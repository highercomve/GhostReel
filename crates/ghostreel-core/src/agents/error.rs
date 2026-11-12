//! What went wrong when an agent was asked something, typed.
//!
//! Every backend fails in its own words: claude prints a JSON `result` with `is_error`, agy writes
//! `error: … AGY_ERROR: {…}` to stderr, codex reports JSONL events, a server answers with an HTTP
//! status. [`classify_cli`] and [`classify_http`] read those into one [`AgentError`]: a kind the
//! pool acts on ([`AgentErrorKind::handling`]) and the UI shows as a chip, plus one short line in
//! the tool's own words and the raw tail for a details view.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Why a call failed, as far as the caller needs to know to decide what to do next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentErrorKind {
    /// The configured model does not exist or this account cannot use it.
    InvalidModel {
        model: String,
    },
    /// The account's quota is used up until it resets.
    QuotaExhausted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resets_in_s: Option<u64>,
    },
    /// Rate limited or too busy right now: worth asking again shortly.
    Busy {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_s: Option<u64>,
    },
    /// Not signed in, or the key is wrong.
    Auth,
    /// The tool is not installed (or cannot be started).
    NotInstalled,
    Timeout {
        after_s: u64,
    },
    /// It answered, but not with something usable (no JSON, wrong shape).
    BadAnswer,
    /// A server that could not be reached at all.
    Unreachable,
    /// A server answered with an error status nothing above explains.
    Http {
        status: u16,
    },
    /// A CLI exited with an error nothing above explains.
    Exit {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<i32>,
    },
    /// No agent of a pool could take the work; each one's reason is listed.
    NoAgentLeft {
        role: String,
        agents: Vec<AgentError>,
    },
    Other,
}

/// What a caller does about a failed call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Handling {
    /// Ask again after a pause (this agent or another).
    Retry { after: Option<Duration> },
    /// The agent is out until its quota resets: every feature skips it until then.
    Cooldown { for_: Option<Duration> },
    /// The agent cannot work at all as configured: stop sending it anything.
    Drop,
    /// This one call failed: try it once more, elsewhere when there is an elsewhere.
    RetryElsewhere,
}

/// A rate limit that clears in less than this is a pause, not a used-up quota.
const SHORT_QUOTA: Duration = Duration::from_secs(120);

impl AgentErrorKind {
    /// A short plain label for a chip: "quota used up", "model not available".
    pub fn label(&self) -> String {
        match self {
            AgentErrorKind::InvalidModel { model } if model.is_empty() => "model not available".into(),
            AgentErrorKind::InvalidModel { model } => format!("model {model} not available"),
            AgentErrorKind::QuotaExhausted { resets_in_s: Some(s) } => {
                format!("quota used up, resets in {}", human_duration(Duration::from_secs(*s)))
            }
            AgentErrorKind::QuotaExhausted { resets_in_s: None } => "quota used up".into(),
            AgentErrorKind::Busy { .. } => "busy, retry later".into(),
            AgentErrorKind::Auth => "not signed in".into(),
            AgentErrorKind::NotInstalled => "not installed".into(),
            AgentErrorKind::Timeout { after_s: 0 } => "timed out".into(),
            AgentErrorKind::Timeout { after_s } => format!("timed out after {after_s} s"),
            AgentErrorKind::BadAnswer => "unusable answer".into(),
            AgentErrorKind::Unreachable => "not reachable".into(),
            AgentErrorKind::Http { status } => format!("HTTP {status}"),
            AgentErrorKind::Exit { code: Some(c) } => format!("exit code {c}"),
            AgentErrorKind::Exit { code: None } => "stopped by a signal".into(),
            AgentErrorKind::NoAgentLeft { role, .. } => format!("no {role} agent left"),
            AgentErrorKind::Other => "failed".into(),
        }
    }

    /// The kind's key as it is serialized ("quota_exhausted"), for logs and tests.
    pub fn key(&self) -> &'static str {
        match self {
            AgentErrorKind::InvalidModel { .. } => "invalid_model",
            AgentErrorKind::QuotaExhausted { .. } => "quota_exhausted",
            AgentErrorKind::Busy { .. } => "busy",
            AgentErrorKind::Auth => "auth",
            AgentErrorKind::NotInstalled => "not_installed",
            AgentErrorKind::Timeout { .. } => "timeout",
            AgentErrorKind::BadAnswer => "bad_answer",
            AgentErrorKind::Unreachable => "unreachable",
            AgentErrorKind::Http { .. } => "http",
            AgentErrorKind::Exit { .. } => "exit",
            AgentErrorKind::NoAgentLeft { .. } => "no_agent_left",
            AgentErrorKind::Other => "other",
        }
    }

    pub fn handling(&self) -> Handling {
        match self {
            AgentErrorKind::Busy { retry_after_s } => Handling::Retry { after: retry_after_s.map(Duration::from_secs) },
            AgentErrorKind::QuotaExhausted { resets_in_s } => {
                Handling::Cooldown { for_: resets_in_s.map(Duration::from_secs) }
            }
            AgentErrorKind::InvalidModel { .. }
            | AgentErrorKind::Auth
            | AgentErrorKind::NotInstalled
            | AgentErrorKind::NoAgentLeft { .. } => Handling::Drop,
            _ => Handling::RetryElsewhere,
        }
    }
}

/// One failed call: who, what kind of failure, and what the tool itself said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentError {
    /// The agent's id, or the tool / server when no agent id is known yet.
    pub agent: String,
    #[serde(flatten)]
    pub kind: AgentErrorKind,
    /// One short line, in the tool's own words where it gave any.
    pub message: String,
    /// The tail of what it printed, for an expandable details view.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub details: String,
}

impl AgentError {
    pub fn new(agent: impl Into<String>, kind: AgentErrorKind, message: impl Into<String>) -> Self {
        Self { agent: agent.into(), kind, message: message.into(), details: String::new() }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = tail(&details.into(), 4000);
        self
    }

    /// The same error, told about the agent by its id.
    pub fn for_agent(mut self, id: &str) -> Self {
        if !id.is_empty() {
            self.agent = id.to_string();
        }
        self
    }

    /// Any [`crate::Error`] as an agent error: typed when it already is one, else `kind` with the
    /// error's text.
    pub fn from_error(agent: &str, e: &crate::Error, kind: AgentErrorKind) -> Self {
        match e {
            crate::Error::Agent(a) => a.clone().for_agent(agent),
            other => AgentError::new(agent, kind, other.to_string()),
        }
    }

    /// Every agents' reason in one error, when none of them is left to take the work.
    pub fn no_agent_left(role: &str, agents: Vec<AgentError>) -> Self {
        let message = agents.iter().map(|a| a.to_string()).collect::<Vec<_>>().join("; ");
        AgentError::new("pool", AgentErrorKind::NoAgentLeft { role: role.to_string(), agents }, message)
    }
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let AgentErrorKind::NoAgentLeft { role, .. } = &self.kind {
            return write!(f, "no {role} agent could take the work — {}", self.message);
        }
        let label = self.kind.label();
        if self.message.is_empty() || self.message.eq_ignore_ascii_case(&label) {
            write!(f, "{}: {label}", self.agent)
        } else {
            write!(f, "{}: {label} — {}", self.agent, self.message)
        }
    }
}

impl std::error::Error for AgentError {}

/// "2 h 3 min", "45 min", "30 s".
pub fn human_duration(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..=89 => format!("{s} s"),
        90..=3599 => format!("{} min", (s + 30) / 60),
        _ => {
            let (h, m) = (s / 3600, (s % 3600 + 30) / 60);
            if m == 0 || m == 60 { format!("{} h", h + m / 60) } else { format!("{h} h {m} min") }
        }
    }
}

fn tail(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max { s.to_string() } else { format!("…{}", s.chars().skip(n - max).collect::<String>()) }
}

fn shorten(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max { s.to_string() } else { format!("{}…", s.chars().take(max).collect::<String>()) }
}

// ---------------------------------------------------------------------------------------------
// Classification

/// Read what a CLI printed into a typed error. `code` is its exit code (`None`: killed by a
/// signal); `exited_ok` is a zero exit whose output still reports an error (claude's
/// `is_error`).
pub fn classify_cli(tool: &str, model: &str, code: Option<i32>, stdout: &str, stderr: &str) -> AgentError {
    let said = cli_message(stdout, stderr);
    let details =
        [stderr.trim(), stdout.trim()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("\n");
    let kind = classify_text(&format!("{said}\n{stderr}\n{stdout}"), model).unwrap_or(match code {
        Some(0) => AgentErrorKind::BadAnswer,
        code => AgentErrorKind::Exit { code },
    });
    let message = if said.is_empty() {
        match code {
            Some(c) => format!("{tool} exited with code {c} and printed nothing"),
            None => format!("{tool} was stopped by a signal"),
        }
    } else {
        said
    };
    AgentError::new(tool, kind, message).with_details(details)
}

/// Read a server's error answer into a typed error.
pub fn classify_http(server: &str, model: &str, status: u16, retry_after: Option<u64>, body: &str) -> AgentError {
    let said = http_message(body);
    let from_text = classify_text(&format!("{said}\n{body}"), model);
    let kind = match (status, from_text) {
        (_, Some(AgentErrorKind::InvalidModel { model })) => AgentErrorKind::InvalidModel { model },
        (401 | 403, _) => AgentErrorKind::Auth,
        (429, Some(AgentErrorKind::QuotaExhausted { resets_in_s })) => {
            AgentErrorKind::QuotaExhausted { resets_in_s: resets_in_s.or(retry_after) }
        }
        (429 | 503 | 529, _) => {
            AgentErrorKind::Busy { retry_after_s: retry_after.or_else(|| retry_in(body).map(|d| d.as_secs().max(1))) }
        }
        (_, Some(AgentErrorKind::Busy { retry_after_s })) => {
            AgentErrorKind::Busy { retry_after_s: retry_after_s.or(retry_after) }
        }
        (_, Some(k)) => k,
        (404, None) if !model.is_empty() && body.to_lowercase().contains("model") => {
            AgentErrorKind::InvalidModel { model: model.to_string() }
        }
        (s, None) => AgentErrorKind::Http { status: s },
    };
    let message = if said.is_empty() { format!("HTTP {status}") } else { said };
    AgentError::new(server, kind, message).with_details(body)
}

/// A server's non-success answer, read into a typed error (status, `Retry-After`, body).
pub async fn from_response(server: &str, model: &str, resp: reqwest::Response) -> AgentError {
    let status = resp.status().as_u16();
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    let body = resp.text().await.unwrap_or_default();
    classify_http(server, model, status, retry_after, &body)
}

/// A request that got no answer at all.
pub fn from_request(server: &str, e: &reqwest::Error) -> AgentError {
    let kind = if e.is_timeout() { AgentErrorKind::Timeout { after_s: 0 } } else { AgentErrorKind::Unreachable };
    let message =
        if e.is_timeout() { "no answer in time".to_string() } else { format!("could not reach {server}: {e}") };
    AgentError::new(server, kind, message)
}

/// The kind a piece of tool or server output describes, when it describes one.
pub fn classify_text(text: &str, model: &str) -> Option<AgentErrorKind> {
    let t = text.to_lowercase();
    let any = |needles: &[&str]| needles.iter().any(|n| t.contains(n));
    if any(&[
        "issue with the selected model",
        "model not found",
        "model_not_found",
        "modelnotfound",
        "unknown model",
        "invalid model",
        "not a valid model",
        "model is not supported",
        "model does not exist",
        "no such model",
        "model is not available",
    ]) {
        return Some(AgentErrorKind::InvalidModel { model: model_named(text).unwrap_or_else(|| model.to_string()) });
    }
    if any(&[
        "not logged in",
        "please log in",
        "please login",
        "run /login",
        "invalid api key",
        "invalid_api_key",
        "invalid x-api-key",
        "authentication_error",
        "authentication failed",
        "unauthorized",
        "oauth token has expired",
        "not authenticated",
        "login required",
    ]) {
        return Some(AgentErrorKind::Auth);
    }
    if any(&[
        "resource_exhausted",
        "quota",
        "usage limit",
        "limit reached",
        "insufficient_quota",
        "credit balance is too low",
        "out of credits",
    ]) {
        let resets = resets_in(text);
        // A per-minute limit is a pause, not a used-up quota.
        if resets.is_some_and(|d| d <= SHORT_QUOTA) {
            return Some(AgentErrorKind::Busy { retry_after_s: resets.map(|d| d.as_secs().max(1)) });
        }
        return Some(AgentErrorKind::QuotaExhausted { resets_in_s: resets.map(|d| d.as_secs()) });
    }
    if any(&[
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "too many chat requests",
        "overloaded",
        "server is busy",
        "busy:",
        "retry later",
        "try again later",
        "temporarily unavailable",
        "service unavailable",
    ]) {
        return Some(AgentErrorKind::Busy { retry_after_s: retry_in(text).map(|d| d.as_secs().max(1)) });
    }
    None
}

/// The model a "selected model (x)" / "model 'x'" message names.
fn model_named(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    for open in ["selected model (", "model ("] {
        if let Some(i) = lower.find(open) {
            let rest = &text[i + open.len()..];
            if let Some(end) = rest.find(')') {
                let m = rest[..end].trim();
                if !m.is_empty() && !m.contains(' ') {
                    return Some(m.to_string());
                }
            }
        }
    }
    for q in ['\'', '"', '`'] {
        if let Some(i) = lower.find(&format!("model {q}")) {
            let rest = &text[i + 7..];
            if let Some(end) = rest.find(q) {
                return Some(rest[..end].to_string());
            }
        }
        if let Some(i) = lower.find(&format!("{q} model")) {
            let before = &text[..i];
            if let Some(start) = before.rfind(q) {
                return Some(before[start + 1..].to_string());
            }
        }
    }
    None
}

/// How long until a quota resets, from "Resets in 2h2m31s", "try again in 47 minutes",
/// "usage limit reached|1731340800".
pub fn resets_in(text: &str) -> Option<Duration> {
    let lower = text.to_lowercase();
    for key in
        ["resets in", "reset in", "resets after", "try again in", "retry in", "retry after", "available again in"]
    {
        if let Some(i) = lower.find(key)
            && let Some(d) = parse_duration(&lower[i + key.len()..])
        {
            return Some(d);
        }
    }
    // claude's older usage-limit line ends with the reset time as a Unix timestamp.
    if let Some(i) = lower.find("limit reached|") {
        let digits: String = lower[i + 14..].chars().take_while(char::is_ascii_digit).collect();
        if let Ok(at) = digits.parse::<u64>() {
            let now =
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            return Some(Duration::from_secs(at.saturating_sub(now)));
        }
    }
    None
}

/// How long a busy server asks to wait ("retry after 30 seconds", "Please retry in 23.4s").
fn retry_in(text: &str) -> Option<Duration> {
    let lower = text.to_lowercase();
    for key in ["retry after", "retry in", "try again in", "retry-after:"] {
        if let Some(i) = lower.find(key)
            && let Some(d) = parse_duration(&lower[i + key.len()..])
        {
            return Some(d);
        }
    }
    None
}

/// "2h2m31s", "2 h 3 min", "47 minutes", "23.5s", "1 hour and 5 minutes" at the start of `s`.
fn parse_duration(s: &str) -> Option<Duration> {
    let mut total = 0.0f64;
    let mut found = false;
    let mut rest = s.trim_start();
    loop {
        let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
        if num.is_empty() {
            break;
        }
        let Ok(n) = num.parse::<f64>() else { break };
        let after = rest[num.len()..].trim_start();
        let unit: String = after.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        let secs = match unit.as_str() {
            "d" | "day" | "days" => 86400.0,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3600.0,
            "m" | "min" | "mins" | "minute" | "minutes" => 60.0,
            "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
            // A bare number (Retry-After: 30) is seconds, but only on its own.
            "" if !found => 1.0,
            _ => break,
        };
        total += n * secs;
        found = true;
        rest = after[unit.len()..].trim_start_matches([' ', ',']).trim_start();
        if let Some(r) = rest.strip_prefix("and ") {
            rest = r;
        }
        if unit.is_empty() {
            break;
        }
    }
    found.then(|| Duration::from_secs_f64(total))
}

/// The one line a CLI said about why it failed.
fn cli_message(stdout: &str, stderr: &str) -> String {
    // claude: {"type":"result","is_error":true,"result":"There's an issue with the selected model…"}
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(stdout.trim())
        && let Some(r) = v["result"].as_str().or_else(|| v["error"].as_str()).or_else(|| v["error"]["message"].as_str())
        && !r.trim().is_empty()
    {
        return tidy(r);
    }
    // codex: JSONL error events on stdout.
    if let Some(m) = crate::cliagent::jsonl_error(stdout) {
        return tidy(&m);
    }
    let pick = |s: &str| -> Option<String> {
        let lines: Vec<&str> = s
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("[opencode-") && !l.starts_with("Reading additional input"))
            .collect();
        // The line that says "error" if there is one, else the last.
        let line = lines.iter().find(|l| l.to_lowercase().contains("error")).or(lines.last())?;
        Some(tidy(line))
    };
    pick(stderr).or_else(|| pick(stdout)).unwrap_or_default()
}

/// What an error body says: `{"error":{"message":…}}`, `{"error":"…"}`, `{"detail":…}` or text.
fn http_message(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim()) {
        let m = v["error"]["message"]
            .as_str()
            .or_else(|| v["error"].as_str())
            .or_else(|| v["message"].as_str())
            .or_else(|| v["detail"].as_str());
        if let Some(m) = m {
            return tidy(m);
        }
    }
    tidy(body.lines().find(|l| !l.trim().is_empty()).unwrap_or_default())
}

/// One readable line: no "error:" prefix, no trailing machine dump, a sane length.
fn tidy(s: &str) -> String {
    let mut s = s.trim();
    // agy appends its machine-readable copy: "… Resets in 2h2m31s. AGY_ERROR: {…}".
    if let Some(i) = s.find("AGY_ERROR:") {
        s = s[..i].trim();
    }
    for p in ["error:", "Error:", "ERROR:"] {
        if let Some(r) = s.strip_prefix(p) {
            s = r.trim();
        }
    }
    shorten(s.lines().next().unwrap_or_default(), 300)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGY_QUOTA: &str = "error: Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 2h2m31s. AGY_ERROR: {\"short_error\":\"RESOURCE_EXHAUSTED (code 429)\",\"details\":\"quota\"}";
    const CLAUDE_MODEL: &str = r#"{"type":"result","subtype":"success","is_error":true,"result":"There's an issue with the selected model (claude-opus-5-5-medium). It may not exist or you may not have access to it. Run --model to pick a different model.","session_id":"x"}"#;
    const GHOSTPEN_BUSY: &str =
        r#"{"error":{"message":"busy: too many chat requests waiting, retry later","type":"server_busy"}}"#;

    #[test]
    fn real_failures_are_classified() {
        let e = classify_cli("agy", "", Some(3), "", AGY_QUOTA);
        assert_eq!(e.kind, AgentErrorKind::QuotaExhausted { resets_in_s: Some(2 * 3600 + 2 * 60 + 31) });
        assert_eq!(
            e.message,
            "Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 2h2m31s."
        );
        assert!(e.details.contains("RESOURCE_EXHAUSTED"));
        assert_eq!(e.to_string(), format!("agy: quota used up, resets in 2 h 3 min — {}", e.message));

        // Exit 1 with the reason on stdout and nothing on stderr: the case that read "exited Some(1):".
        let e = classify_cli("claude", "claude-opus-5-5-medium", Some(1), CLAUDE_MODEL, "");
        assert_eq!(e.kind, AgentErrorKind::InvalidModel { model: "claude-opus-5-5-medium".into() });
        assert!(e.message.starts_with("There's an issue with the selected model"));
        assert!(!e.to_string().contains("Some("));
        // The same with exit 0 and plain text.
        let e = classify_cli(
            "claude",
            "x",
            Some(0),
            "There's an issue with the selected model (claude-opus-5-5-medium). It may not exist or you may not have access to it.",
            "",
        );
        assert_eq!(e.kind, AgentErrorKind::InvalidModel { model: "claude-opus-5-5-medium".into() });

        let e = classify_http("ghostpen", "", 503, None, GHOSTPEN_BUSY);
        assert_eq!(e.kind, AgentErrorKind::Busy { retry_after_s: None });
        assert_eq!(e.message, "busy: too many chat requests waiting, retry later");
        let e = classify_http("srv", "", 503, Some(7), "busy: too many chat requests waiting, retry later");
        assert_eq!(e.kind, AgentErrorKind::Busy { retry_after_s: Some(7) });
    }

    #[test]
    fn classification_table() {
        let cases: &[(&str, &str)] = &[
            ("You've hit your usage limit. Upgrade to Pro or try again in 1 hour 5 minutes.", "quota_exhausted"),
            ("Claude AI usage limit reached|4102444800", "quota_exhausted"),
            ("Error: 429 RESOURCE_EXHAUSTED. Please retry in 23.5s.", "busy"),
            (
                "API Error: 529 {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
                "busy",
            ),
            ("Rate limit exceeded, retry after 30 seconds", "busy"),
            ("Invalid API key · Please run /login", "auth"),
            ("Not logged in · Please run /login", "auth"),
            ("The 'gpt-9' model is not supported when using Codex with a ChatGPT account.", "invalid_model"),
            ("ProviderModelNotFoundError: Model not found: opencode/nope", "invalid_model"),
        ];
        for (text, want) in cases {
            let got = classify_text(text, "").map(|k| k.key()).unwrap_or("none");
            assert_eq!(got, *want, "{text}");
        }
        assert_eq!(classify_text("all good", ""), None);
        let e = classify_cli("codex", "", Some(1), "", "");
        assert_eq!(e.kind, AgentErrorKind::Exit { code: Some(1) });
        assert_eq!(e.to_string(), "codex: exit code 1 — codex exited with code 1 and printed nothing");
        assert_eq!(
            classify_text("try again in 1 hour 5 minutes, quota", ""),
            Some(AgentErrorKind::QuotaExhausted { resets_in_s: Some(3900) })
        );
        assert_eq!(
            classify_text("The 'gpt-9' model is not supported", ""),
            Some(AgentErrorKind::InvalidModel { model: "gpt-9".into() })
        );
        assert_eq!(classify_http("s", "m", 401, None, "nope").kind, AgentErrorKind::Auth);
        assert_eq!(
            classify_http("s", "m", 404, None, "model m not loaded").kind,
            AgentErrorKind::InvalidModel { model: "m".into() }
        );
        assert_eq!(classify_http("s", "", 500, None, "boom").kind, AgentErrorKind::Http { status: 500 });
    }

    #[test]
    fn errors_round_trip_as_json_with_a_flat_kind() {
        let e = classify_cli("agy", "", Some(3), "", AGY_QUOTA).for_agent("agy-pro");
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["kind"], "quota_exhausted");
        assert_eq!(v["agent"], "agy-pro");
        assert_eq!(v["resets_in_s"], 7351);
        let back: AgentError = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
        let all =
            AgentError::no_agent_left("worker", vec![e.clone(), AgentError::new("claude", AgentErrorKind::Auth, "")]);
        let s = all.to_string();
        assert!(s.starts_with("no worker agent could take the work — agy-pro: quota used up"), "{s}");
        assert!(s.ends_with("claude: not signed in"), "{s}");
        let back: AgentError = serde_json::from_str(&serde_json::to_string(&all).unwrap()).unwrap();
        assert_eq!(back, all);
    }

    #[test]
    fn durations_read_and_print() {
        assert_eq!(parse_duration("2h2m31s."), Some(Duration::from_secs(7351)));
        assert_eq!(parse_duration("47 minutes"), Some(Duration::from_secs(2820)));
        assert_eq!(parse_duration("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_duration("soon"), None);
        assert_eq!(human_duration(Duration::from_secs(7351)), "2 h 3 min");
        assert_eq!(human_duration(Duration::from_secs(45)), "45 s");
        assert_eq!(human_duration(Duration::from_secs(600)), "10 min");
        assert_eq!(human_duration(Duration::from_secs(7200)), "2 h");
    }
}
