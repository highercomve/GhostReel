//! Delegate vision / chat to an installed coding-agent CLI (claude, agy, opencode, codex).
//!
//! Each tool has a different invocation shape:
//! - **claude**: `claude -p "<prompt>" --output-format json --allowedTools Read [--model <m>]`
//!   stdout is JSON; answer is `result` (a string). Chat turns send the prompt on stdin instead,
//!   as a plain brief, and pin the conversation with `--session-id` / `--resume <id>`.
//! - **agy**: `agy --dangerously-skip-permissions --add-dir <dir> --output-format json [--model <m>] -p "<prompt>"`
//!   stdout is JSON; answer is `response` (a string).
//! - **opencode**: `opencode run [-m <provider/model>] "<prompt>"`
//!   plain stdout; ignore `[opencode-*]` plugin log lines; answer is the first `{...}` object.
//! - **codex**: `codex exec --json --skip-git-repo-check -s read-only [-i <image>] [-m <m>] "<prompt>"`
//!   stdout is JSONL events; the answer is the last `item.completed` of type `agent_message`.
//!   It attaches images itself via `-i`, so the prompt does not ask it to read the file.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::Error;
use crate::agents::{AgentError, AgentErrorKind, Effort, classify_cli};
use crate::config::CliAgentConfig;

/// Resolves a binary path for a named CLI tool.
/// Checks `command` field first, then `GHOSTREEL_<TOOL>` env var, then PATH.
pub fn find_binary(cfg: &CliAgentConfig) -> Option<PathBuf> {
    // Explicit command wins.
    if !cfg.command.is_empty() {
        let p = PathBuf::from(&cfg.command);
        if p.is_file() {
            return Some(p);
        }
        return None;
    }
    // Honour the GHOSTREEL_<TOOL> env override (same pattern as doctor::locate).
    let tool = if cfg.tool.is_empty() { return None } else { &cfg.tool };
    if let Some(p) = std::env::var_os(format!("GHOSTREEL_{}", tool.to_uppercase())) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    // Fall back to PATH, then to the usual per-user install dirs.
    let exe = if cfg!(windows) { format!("{tool}.exe") } else { tool.to_string() };
    if let Some(path) = std::env::var_os("PATH")
        && let Some(hit) = std::env::split_paths(&path).map(|d| d.join(&exe)).find(|p| p.is_file())
    {
        return Some(hit);
    }
    user_bin_dirs().into_iter().map(|d| d.join(&exe)).find(|p| p.is_file())
}

/// Where these CLIs actually install themselves. A desktop-launched app inherits the session's
/// PATH (`/usr/local/bin:/usr/bin` on a stock Linux login), not the one a shell builds from the
/// user's profile, so a tool installed in `~/.local/bin` is invisible unless we look here.
fn user_bin_dirs() -> Vec<PathBuf> {
    let Some(home) = home_dir() else { return Vec::new() };
    let rel: &[&str] = if cfg!(windows) {
        &["AppData/Local/Programs", "AppData/Roaming/npm", ".bun/bin", ".local/bin"]
    } else {
        &[
            ".local/bin",
            "bin",
            ".bun/bin",
            ".opencode/bin",
            ".deno/bin",
            ".cargo/bin",
            ".volta/bin",
            ".npm-global/bin",
            ".local/share/pnpm",
            ".yarn/bin",
        ]
    };
    let mut dirs: Vec<PathBuf> = rel.iter().map(|r| home.join(r)).collect();
    if !cfg!(windows) {
        dirs.push(PathBuf::from("/opt/homebrew/bin"));
        dirs.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin"));
    }
    dirs
}

fn home_dir() -> Option<PathBuf> {
    let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(key).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Turn the ChatML transcript into something a coding agent reads as its own brief.
///
/// claude and agy take `<|im_start|>system … <|im_end|>` as instructions and get on with it.
/// opencode reads it as somebody else's conversation pasted at it and says so — verbatim: "I'm
/// not going to echo that back or invent fake entries — I'm opencode, not your tool runtime."
/// The markers go, the turns are labelled in plain words, and the job is stated first.
fn as_plain_brief(prompt: &str) -> String {
    let mut out = String::from(
        "You are the editor described below. These are your instructions, not a transcript to \
         comment on: follow them and reply with only the JSON object they ask for.\n\n",
    );
    for chunk in prompt.split("<|im_start|>") {
        let chunk = chunk.trim_end_matches("<|im_end|>").trim_end().trim_end_matches("<|im_end|>");
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (role, body) = chunk.split_once('\n').unwrap_or(("", chunk));
        let body = body.replace("<|im_end|>", "").trim().to_string();
        if body.is_empty() {
            continue;
        }
        match role.trim() {
            "system" => out.push_str(&format!("{body}\n\n")),
            "user" => out.push_str(&format!("WHAT YOU WERE ASKED\n{body}\n\n")),
            "assistant" => out.push_str(&format!("WHAT YOU ANSWERED LAST\n{body}\n\n")),
            _ => out.push_str(&format!("{body}\n\n")),
        }
    }
    out
}

pub struct CliAgent {
    pub cfg: CliAgentConfig,
    /// claude's conversation for this agent, pinned with `--session-id` on the first call and
    /// resumed by id afterwards. `--continue` resumed whatever ran last in the working folder,
    /// which could be the user's own session in a terminal.
    session: std::sync::Mutex<Option<String>>,
}

/// A fresh random UUID (v4 layout) for `claude --session-id`, which rejects anything else.
fn new_session_id() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut h = blake3::Hasher::new();
    h.update(&std::process::id().to_le_bytes());
    h.update(&SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_le_bytes());
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    h.update(&nanos.to_le_bytes());
    let mut b = [0u8; 16];
    b.copy_from_slice(&h.finalize().as_bytes()[..16]);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let x: String = b.iter().map(|v| format!("{v:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &x[0..8], &x[8..12], &x[12..16], &x[16..20], &x[20..32])
}

/// The most a whole command line may be before the prompt is spilled to a file instead.
///
/// Windows caps a command line at 32767 characters and `CreateProcess` fails with os error 206,
/// "The filename or extension is too long" — a message that points at the binary and not at the
/// argument that is actually too big. The script chat's prompt is the whole project's speech:
/// 75 KB on a 96-video project, so every CLI turn failed there before it started.
///
/// Unix allows megabytes, so the threshold is set so high that the working path never spills and
/// behaves exactly as it did.
const MAX_ARGV_CHARS: usize = if cfg!(windows) { 30_000 } else { 1_000_000 };

/// Write a prompt too long to pass as an argument, and return the file and the directory to
/// grant. `None` when it fits, which is the ordinary case.
fn spill_prompt(prompt: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    if prompt.len() <= MAX_ARGV_CHARS {
        return None;
    }
    let dir = std::env::temp_dir().join("ghostreel-prompts");
    std::fs::create_dir_all(&dir).ok()?;
    // One file per call, cleaned by the OS: two turns must not read each other's prompt.
    let name = format!(
        "prompt-{}-{}.md",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    );
    let file = dir.join(name);
    std::fs::write(&file, prompt).ok()?;
    Some((file, dir))
}

/// What to say instead of the prompt when it had to be written to a file.
fn read_the_prompt(file: &Path) -> String {
    format!(
        "Read the file {} — it contains your full instructions, including the footage you may use \
         and the format of your answer — and then do exactly what it says. Do not reply about the \
         file itself.",
        file.display()
    )
}

/// Parse any `[User attached image: <path>]` references inside the prompt.
fn extract_attached_image_paths(prompt: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let marker = "[User attached image: ";
    let mut rest = prompt;
    while let Some(idx) = rest.find(marker) {
        let after = &rest[idx + marker.len()..];
        if let Some(end) = after.find(']') {
            let path_str = &after[..end];
            let p = PathBuf::from(path_str.trim());
            if p.is_file() && !out.contains(&p) {
                out.push(p);
            }
            rest = &after[end + 1..];
        } else {
            break;
        }
    }
    out
}

impl CliAgent {
    pub fn new(cfg: CliAgentConfig) -> Self {
        Self { cfg, session: std::sync::Mutex::new(None) }
    }

    /// Resolve binary or return `None`.
    pub fn available(&self) -> Option<PathBuf> {
        find_binary(&self.cfg)
    }

    /// Describe one generated test image, so the user can check the CLI works (and see what it
    /// costs) before starting an index run that calls it once per keyframe.
    pub async fn self_test(&self, data_dir: &Path) -> Result<String, Error> {
        let dir = data_dir.join("tmp");
        std::fs::create_dir_all(&dir).map_err(|e| Error::Io(dir.clone(), e))?;
        let path = dir.join("cli-agent-test.jpg");
        // A red square with a green stripe: enough for the answer to show the model really looked.
        let mut img = image::RgbImage::from_pixel(320, 180, image::Rgb([200, 30, 30]));
        for y in 70..110 {
            for x in 0..320 {
                img.put_pixel(x, y, image::Rgb([30, 180, 60]));
            }
        }
        img.save(&path).map_err(|e| Error::Vision(format!("writing the test image: {e}")))?;
        let started = std::time::Instant::now();
        let json = self.describe(&path, &serde_json::to_string(&crate::vision::schema()).unwrap_or_default()).await?;
        let _ = std::fs::remove_file(&path);
        Ok(format!("{} answered in {:.1} s:\n{json}", self.cfg.tool, started.elapsed().as_secs_f64()))
    }

    /// Build argv for a describe call (image path + text prompt).
    fn build_argv_describe(&self, bin: &Path, image: &Path, prompt: &str) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        match self.cfg.tool.as_str() {
            "claude" => {
                args.push(bin.as_os_str().to_owned());
                args.push("-p".into());
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
                args.push("--output-format".into());
                args.push("json".into());
                args.push("--allowedTools".into());
                args.push("Read".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.extend(self.effort_args());
            }
            "agy" => {
                args.push(bin.as_os_str().to_owned());
                args.push("--dangerously-skip-permissions".into());
                // Grant access to the directory containing the frame.
                if let Some(dir) = image.parent() {
                    args.push("--add-dir".into());
                    args.push(dir.as_os_str().to_owned());
                }
                args.push("--output-format".into());
                args.push("json".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push("-p".into());
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
            }
            "opencode" => {
                args.push(bin.as_os_str().to_owned());
                args.push("run".into());
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                let full_prompt =
                    format!("Read the image file {} and reply with ONLY compact JSON: {}", image.display(), prompt);
                args.push(full_prompt.into());
            }
            "codex" => {
                args.extend(codex_exec_prefix(bin));
                // codex attaches the image itself, so the prompt is only the schema.
                args.push("-i".into());
                args.push(image.as_os_str().to_owned());
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.extend(self.effort_args());
                args.push(format!("Describe this image. Reply with ONLY compact JSON: {prompt}").into());
            }
            other => {
                // Unknown tool — return an empty argv; the caller will error.
                tracing_warn(other);
            }
        }
        for extra in &self.cfg.extra_args {
            args.push(extra.clone().into());
        }
        args
    }

    /// Build argv for a text-only complete call.
    ///
    /// With `continued`, the tool picks up its own most recent conversation and `prompt` is only
    /// what is new. Every round otherwise re-sends the whole transcript as a fresh invocation,
    /// which grows quadratically: on a 96-video library agy needed more than three minutes for a
    /// single round and hit the timeout.
    fn build_argv_complete_from(&self, bin: &Path, prompt: &str, continued: bool) -> Vec<std::ffi::OsString> {
        let mut args = self.build_argv_complete(bin, prompt, continued);
        if !continued {
            return args;
        }
        match self.cfg.tool.as_str() {
            // claude resumes its own session by id (see `session`); `build_argv_complete` does it.
            "claude" => {}
            // agy takes the flag anywhere; the prompt stays as the new message.
            "agy" => args.insert(1, "--continue".into()),
            // codex resumes by subcommand: `codex exec resume --last`.
            "codex" => {
                if let Some(i) = args.iter().position(|a| a == "exec") {
                    args.insert(i + 1, "resume".into());
                    args.insert(i + 2, "--last".into());
                }
            }
            // opencode continues too — `run -c`. It has to come after the `run` subcommand.
            "opencode" => {
                if let Some(i) = args.iter().position(|a| a == "run") {
                    args.insert(i + 1, "-c".into());
                }
            }
            _ => {}
        }
        args
    }

    fn build_argv_complete(&self, bin: &Path, prompt: &str, continued: bool) -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = Vec::new();
        // Too long to pass as an argument on this platform: write it down and point at it. The
        // agents all read files; what they cannot do is take 75 KB through argv on Windows.
        // claude is the exception: it takes the prompt on stdin (`stdin_prompt`), because told to
        // read a file it treats the file as data and answers about it instead of following it.
        let spilled = if self.cfg.tool == "claude" { None } else { spill_prompt(prompt) };
        let owned;
        let effective_prompt: &str = match &spilled {
            Some((file, _)) => {
                owned = read_the_prompt(file);
                &owned
            }
            None => prompt,
        };
        let mut grant_dirs: Vec<PathBuf> = Vec::new();
        if let Some((_, dir)) = &spilled {
            grant_dirs.push(dir.clone());
        }
        // Extract any user-attached images mentioned in prompt so CLI agents can read them
        let attached_imgs = extract_attached_image_paths(prompt);
        for img in &attached_imgs {
            if let Some(parent) = img.parent()
                && !grant_dirs.contains(&parent.to_path_buf())
            {
                grant_dirs.push(parent.to_path_buf());
            }
        }

        match self.cfg.tool.as_str() {
            "claude" => {
                args.push(bin.as_os_str().to_owned());
                // The prompt itself arrives on stdin: see `stdin_prompt`.
                args.push("-p".into());
                args.push("--output-format".into());
                args.push("json".into());
                // First call pins a new session; later ones in this agent resume exactly it.
                {
                    let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
                    match (continued, session.as_ref()) {
                        (true, Some(id)) => {
                            args.push("--resume".into());
                            args.push(id.clone().into());
                        }
                        _ => {
                            let id = new_session_id();
                            args.push("--session-id".into());
                            args.push(id.clone().into());
                            *session = Some(id);
                        }
                    }
                }
                args.push("--allowedTools".into());
                // Reading attached images is the only tool it needs.
                args.push(if !grant_dirs.is_empty() { "Read".into() } else { std::ffi::OsString::from("none") });
                for dir in &grant_dirs {
                    args.push("--add-dir".into());
                    args.push(dir.as_os_str().to_owned());
                }
                // Nothing here edits anything: the agent is asked for JSON, and a prompt it
                // cannot answer is a turn that hangs until the timeout.
                args.push("--dangerously-skip-permissions".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.extend(self.effort_args());
            }
            "agy" => {
                args.push(bin.as_os_str().to_owned());
                args.push("--dangerously-skip-permissions".into());
                for dir in &grant_dirs {
                    args.push("--add-dir".into());
                    args.push(dir.as_os_str().to_owned());
                }
                args.push("--output-format".into());
                args.push("json".into());
                if !self.cfg.model.is_empty() {
                    args.push("--model".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push("-p".into());
                args.push(effective_prompt.into());
            }
            "opencode" => {
                args.push(bin.as_os_str().to_owned());
                args.push("run".into());
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.push(as_plain_brief(effective_prompt).into());
            }
            "codex" => {
                args.extend(codex_exec_prefix(bin));
                args.push("--dangerously-bypass-approvals-and-sandbox".into());
                for img in &attached_imgs {
                    args.push("-i".into());
                    args.push(img.as_os_str().to_owned());
                }
                if !self.cfg.model.is_empty() {
                    args.push("-m".into());
                    args.push(self.cfg.model.clone().into());
                }
                args.extend(self.effort_args());
                args.push(effective_prompt.into());
            }
            other => {
                tracing_warn(other);
            }
        }
        for extra in &self.cfg.extra_args {
            args.push(extra.clone().into());
        }
        args
    }

    /// The effort flag for the tools that take one: claude `--effort <level>`, codex
    /// `-c model_reasoning_effort=<level>` (which has no "max": its top is "xhigh"). Others none.
    fn effort_args(&self) -> Vec<std::ffi::OsString> {
        let Some(effort) = self.cfg.effort else { return Vec::new() };
        match self.cfg.tool.as_str() {
            "claude" => vec!["--effort".into(), effort.as_str().into()],
            "codex" => {
                let level = if effort == Effort::Max { Effort::Xhigh } else { effort };
                vec!["-c".into(), format!("model_reasoning_effort=\"{level}\"").into()]
            }
            _ => Vec::new(),
        }
    }

    /// What to write to the tool's stdin for a complete call. claude reads its prompt there, as a
    /// plain brief: handed the ChatML transcript (or a file holding it) it described the prompt
    /// instead of answering it. Every other tool gets nothing and must not wait for input.
    fn stdin_prompt(&self, prompt: &str, continued: bool) -> Option<String> {
        match self.cfg.tool.as_str() {
            // A continued turn is only the new tool results, already plain text.
            "claude" if continued => Some(prompt.to_string()),
            "claude" => Some(as_plain_brief(prompt)),
            _ => None,
        }
    }

    /// How long this agent is allowed to take. Read by the chat, which gives a script-writing
    /// agent a longer clock than a frame-describing one.
    pub fn timeout_secs(&self) -> u64 {
        self.cfg.timeout_secs
    }

    /// Run a CLI and return stdout.
    async fn run_cli(&self, bin: &Path, args: &[std::ffi::OsString], stdin: Option<String>) -> Result<String, Error> {
        if args.is_empty() {
            return Err(self.fail(AgentErrorKind::NotInstalled, format!("unknown CLI tool '{}'", self.cfg.tool)));
        }
        // args[0] is the binary itself; pass args[1..] to the command.
        let mut cmd = crate::proc::command(bin);
        for a in args.iter().skip(1) {
            cmd.arg(a);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        // codex treats a non-TTY stdin as extra prompt input and waits for EOF: a tool that is
        // not handed a prompt there must not read from us at all.
        cmd.stdin(if stdin.is_some() { std::process::Stdio::piped() } else { std::process::Stdio::null() });
        // ETXTBSY: the binary was written moments ago (a fresh install, or a test fixture) and the
        // kernel still holds it open. One short retry is enough.
        let child = match cmd.spawn() {
            Err(e) if e.raw_os_error() == Some(26) => {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                cmd.spawn()
            }
            other => other,
        }
        .map_err(|e| {
            let kind = if e.kind() == std::io::ErrorKind::NotFound {
                AgentErrorKind::NotInstalled
            } else {
                AgentErrorKind::Exit { code: None }
            };
            self.fail(kind, format!("cannot run {}: {e}", bin.display()))
        })?;
        let mut child = child;
        if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
            use tokio::io::AsyncWriteExt as _;
            // Written from its own task: a 100 KB prompt fills the pipe, and the tool may start
            // writing stdout before it has read all of it.
            tokio::spawn(async move {
                let _ = pipe.write_all(text.as_bytes()).await;
                let _ = pipe.shutdown().await;
            });
        }
        let after_s = self.cfg.timeout_secs;
        let output = tokio::time::timeout(Duration::from_secs(after_s), child.wait_with_output())
            .await
            .map_err(|_| {
                self.fail(AgentErrorKind::Timeout { after_s }, format!("no answer within {after_s} s"))
            })?
            .map_err(|e| self.fail(AgentErrorKind::Other, format!("I/O error: {e}")))?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.status.success() {
            // Why it failed may be on stdout (claude's JSON `result`, codex's JSONL events) or
            // stderr (agy); `classify_cli` reads both.
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Agent(classify_cli(&self.cfg.tool, &self.cfg.model, output.status.code(), &stdout, &stderr)));
        }
        // claude can exit 0 and still report an error (an unknown model, say) in its JSON.
        if self.cfg.tool == "claude"
            && serde_json::from_str::<serde_json::Value>(stdout.trim()).is_ok_and(|v| v["is_error"] == true)
        {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Agent(classify_cli(&self.cfg.tool, &self.cfg.model, Some(0), &stdout, &stderr)));
        }
        Ok(stdout)
    }

    /// A typed failure of this tool.
    fn fail(&self, kind: AgentErrorKind, message: impl Into<String>) -> Error {
        Error::Agent(AgentError::new(self.cfg.tool.clone(), kind, message))
    }

    fn not_installed(&self) -> Error {
        self.fail(AgentErrorKind::NotInstalled, format!("'{}' was not found on PATH", self.cfg.tool))
    }

    /// Extract the model's text answer from the tool's stdout.
    fn extract_text(tool: &str, raw: &str) -> Result<String, Error> {
        match tool {
            "claude" => {
                // stdout is JSON: {"result": "<answer>", ...}
                let v: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|_| unusable("claude", raw))?;
                let text = v["result"]
                    .as_str()
                    .ok_or_else(|| unusable("claude", raw))?;
                Ok(text.to_string())
            }
            "agy" => {
                // stdout is JSON: {"response": "<answer>", ...}
                let v: serde_json::Value = serde_json::from_str(raw)
                    .map_err(|_| unusable("agy", raw))?;
                let text = v["response"]
                    .as_str()
                    .ok_or_else(|| unusable("agy", raw))?;
                Ok(text.to_string())
            }
            "opencode" => {
                // Plain stdout; plugin log lines start with `[opencode-…]`. Skip them.
                let filtered: Vec<&str> = raw
                    .lines()
                    .filter(|l| {
                        let t = l.trim();
                        !t.starts_with("[opencode-") && !t.is_empty()
                    })
                    .collect();
                Ok(filtered.join("\n"))
            }
            "codex" => {
                // stdout is JSONL events; the answer is the last agent_message item.
                let mut answer: Option<String> = None;
                for line in raw.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
                    if v["type"] == "item.completed"
                        && v["item"]["type"] == "agent_message"
                        && let Some(t) = v["item"]["text"].as_str()
                    {
                        answer = Some(t.to_string());
                    }
                }
                answer.ok_or_else(|| unusable("codex", raw))
            }
            other => Err(Error::Agent(AgentError::new(other, AgentErrorKind::NotInstalled, format!("unknown CLI tool '{other}'")))),
        }
    }

    /// Pull the first `{...}` JSON object out of `text`, stripping any ``` fences or prose.
    pub fn extract_json_object(text: &str) -> Result<String, Error> {
        // Strip ```json ... ``` or ``` ... ``` fences.
        let stripped = strip_code_fences(text);
        let haystack = stripped.as_deref().unwrap_or(text);
        let start = haystack.find('{');
        let end = haystack.rfind('}');
        match (start, end) {
            (Some(s), Some(e)) if e > s => Ok(haystack[s..=e].to_string()),
            _ => Err(Error::Invalid(format!("no JSON object in the answer: {}", preview(text)))),
        }
    }

    /// Describe a frame image: returns the extracted JSON string (matching `vision::parse_description`).
    pub async fn describe(&self, image: &Path, schema_hint: &str) -> Result<String, Error> {
        let bin = self.available().ok_or_else(|| self.not_installed())?;
        let args = self.build_argv_describe(&bin, image, schema_hint);
        let raw = self.run_cli(&bin, &args, None).await?;
        let text = Self::extract_text(&self.cfg.tool, &raw)?;
        Self::extract_json_object(&text)
    }

    /// Send a plain text prompt and return the first JSON object in the response.
    pub async fn complete(&self, prompt: &str) -> Result<String, Error> {
        self.complete_continuing(prompt, false).await
    }

    /// `complete`, optionally continuing the tool's own most recent conversation so only the new
    /// message is sent.
    pub async fn complete_continuing(&self, prompt: &str, continued: bool) -> Result<String, Error> {
        let bin = self.available().ok_or_else(|| self.not_installed())?;
        let args = self.build_argv_complete_from(&bin, prompt, continued);
        let stdin = self.stdin_prompt(prompt, continued);
        let shown = args.iter().skip(1).map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" ");
        crate::llmlog::push("prompt", format!("{} {shown}\n\n{}", self.cfg.tool, stdin.as_deref().unwrap_or(prompt)));
        let raw = match self.run_cli(&bin, &args, stdin).await {
            Ok(raw) => raw,
            Err(e) => {
                crate::llmlog::push("answer", format!("error: {e}"));
                return Err(e);
            }
        };
        crate::llmlog::push("answer", raw.as_str());
        let text = Self::extract_text(&self.cfg.tool, &raw)?;
        Self::extract_json_object(&text)
    }
}

/// `codex exec` in headless shape: JSONL events, no git-repo requirement (frames live in the data
/// dir), read-only sandbox — it only has to look, never edit.
/// The last error a JSONL-speaking tool reported on stdout, with any nested API error unwrapped
/// to its message: `{"type":"error","message":"{\"error\":{\"message\":\"…\"}}"}`.
pub(crate) fn jsonl_error(stdout: &str) -> Option<String> {
    let mut found = None;
    for line in stdout.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else { continue };
        let msg = match v["type"].as_str() {
            Some("error") => v["message"].as_str(),
            Some("turn.failed") => v["error"]["message"].as_str(),
            _ => None,
        };
        if let Some(m) = msg {
            let inner = serde_json::from_str::<serde_json::Value>(m)
                .ok()
                .and_then(|j| j["error"]["message"].as_str().map(str::to_string));
            found = Some(inner.unwrap_or_else(|| m.to_string()));
        }
    }
    found
}

fn codex_exec_prefix(bin: &Path) -> Vec<std::ffi::OsString> {
    vec![
        bin.as_os_str().to_owned(),
        "exec".into(),
        "--json".into(),
        "--skip-git-repo-check".into(),
        "-s".into(),
        "read-only".into(),
    ]
}

fn tracing_warn(tool: &str) {
    // Avoid a hard dependency on `tracing`; just write to stderr in debug builds.
    #[cfg(debug_assertions)]
    eprintln!("cliagent: unknown tool '{tool}'");
    let _ = tool;
}

/// Output a tool printed that is not the answer shape: an error it reported in plain words
/// (an unknown model, a used-up quota) when it reads as one, else an unusable answer.
fn unusable(tool: &str, raw: &str) -> Error {
    let mut e = classify_cli(tool, "", Some(0), raw, "");
    if e.kind == AgentErrorKind::BadAnswer {
        e.message = format!("{tool} answered in an unexpected shape: {}", preview(raw));
    }
    Error::Agent(e)
}

fn preview(s: &str) -> String {
    let p: String = s.chars().take(200).collect();
    if s.chars().count() > 200 { format!("{p}…") } else { p }
}

/// Strip leading ``` fences and return the inner text if any fence was found.
fn strip_code_fences(s: &str) -> Option<String> {
    let trimmed = s.trim();
    let after_open = trimmed.strip_prefix("```json").or_else(|| trimmed.strip_prefix("```"))?;
    let inner = match after_open.rfind("```") {
        Some(pos) => &after_open[..pos],
        None => after_open,
    };
    Some(inner.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CliAgentConfig;

    #[cfg(unix)]
    fn write_fake_cli(dir: &Path, name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        // Write, close, then chmod: exec'ing a file whose handle is still open fails with ETXTBSY.
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(format!("#!/bin/sh\n{script}").as_bytes()).unwrap();
            f.sync_all().unwrap();
        }
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn cfg(tool: &str, command: &Path) -> CliAgentConfig {
        CliAgentConfig {
            tool: tool.to_string(),
            command: command.to_str().unwrap_or("").to_string(),
            timeout_secs: 30,
            concurrency: 1,
            ..Default::default()
        }
    }

    // ---- extract_text --------------------------------------------------------

    #[test]
    fn extract_text_claude_shape() {
        let raw = r#"{"result": "{\"description\":\"a cat\"}", "cost_usd": 0.04}"#;
        let text = CliAgent::extract_text("claude", raw).unwrap();
        assert_eq!(text, r#"{"description":"a cat"}"#);
    }

    #[test]
    fn extract_text_agy_shape() {
        let raw = r#"{"response": "{\"description\":\"a dog\"}", "duration_ms": 1200}"#;
        let text = CliAgent::extract_text("agy", raw).unwrap();
        assert_eq!(text, r#"{"description":"a dog"}"#);
    }

    #[test]
    fn extract_text_opencode_strips_plugin_lines() {
        let raw = "[opencode-lmstudio] connecting…\n{\"description\":\"a bird\"}\n";
        let text = CliAgent::extract_text("opencode", raw).unwrap();
        assert!(text.contains(r#"{"description":"a bird"}"#), "got: {text}");
    }

    // ---- extract_json_object -------------------------------------------------

    #[test]
    fn extract_json_object_from_plain_json() {
        let s = r#"{"description":"test","visible_text":[]}"#;
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), s);
    }

    #[test]
    fn extract_json_object_from_fenced() {
        let s = "```json\n{\"description\":\"test\"}\n```";
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), r#"{"description":"test"}"#);
    }

    #[test]
    fn extract_json_object_strips_prose() {
        let s = "Here you go: {\"description\":\"hi\"} thanks";
        assert_eq!(CliAgent::extract_json_object(s).unwrap(), r#"{"description":"hi"}"#);
    }

    #[test]
    fn extract_json_object_no_json_is_err() {
        assert!(CliAgent::extract_json_object("no json here").is_err());
    }

    // ---- effort ----------------------------------------------------------------

    #[test]
    fn effort_reaches_the_tools_that_take_it() {
        let bin = Path::new("/bin/true");
        let with = |tool: &str, effort| {
            let agent = CliAgent::new(CliAgentConfig { tool: tool.into(), effort, ..Default::default() });
            let mut argv = agent.build_argv_complete(bin, "hi", false);
            argv.extend(agent.build_argv_describe(bin, Path::new("/tmp/f.jpg"), "{}"));
            argv.iter().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>()
        };
        let claude = with("claude", Some(Effort::Medium));
        assert_eq!(claude.iter().filter(|a| *a == "--effort").count(), 2, "complete and describe: {claude:?}");
        assert!(claude.windows(2).any(|w| w[0] == "--effort" && w[1] == "medium"));
        // codex tops out at xhigh.
        let codex = with("codex", Some(Effort::Max));
        assert!(codex.windows(2).any(|w| w[0] == "-c" && w[1] == "model_reasoning_effort=\"xhigh\""), "{codex:?}");
        for tool in ["agy", "opencode"] {
            let argv = with(tool, Some(Effort::High));
            assert!(!argv.iter().any(|a| a.contains("effort")), "{tool} has no effort flag: {argv:?}");
        }
        assert!(!with("claude", None).iter().any(|a| a == "--effort"));
    }

    // ---- typed failures from fake CLIs (unix only) -----------------------------

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_cli_failures_come_back_typed() {
        let tmp = tempfile::tempdir().unwrap();
        // agy with a used-up quota: the reason on stderr, exit 3.
        let agy = write_fake_cli(
            tmp.path(),
            "agy",
            r#"echo 'error: Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 2h2m31s. AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429)"}' >&2; exit 3"#,
        );
        let e = CliAgent::new(cfg("agy", &agy)).complete("hi").await.unwrap_err();
        let a = e.agent_error().expect("typed");
        assert_eq!(a.kind, AgentErrorKind::QuotaExhausted { resets_in_s: Some(7351) });
        assert!(!e.to_string().starts_with("vision:"), "{e}");

        // claude with an unknown model: the reason on stdout as JSON, exit 1, stderr empty.
        let claude = write_fake_cli(
            tmp.path(),
            "claude",
            r#"cat >/dev/null; echo '{"type":"result","is_error":true,"result":"There'"'"'s an issue with the selected model (claude-opus-5-5-medium). It may not exist or you may not have access to it."}'; exit 1"#,
        );
        let mut c = cfg("claude", &claude);
        c.model = "claude-opus-5-5-medium".into();
        let e = CliAgent::new(c.clone()).complete("hi").await.unwrap_err();
        assert_eq!(e.agent_error().unwrap().kind, AgentErrorKind::InvalidModel { model: "claude-opus-5-5-medium".into() });
        assert!(!e.to_string().contains("Some("), "{e}");

        // The same with exit 0: claude still says is_error.
        let claude0 = write_fake_cli(
            tmp.path(),
            "claude0",
            r#"cat >/dev/null; echo '{"type":"result","is_error":true,"result":"There'"'"'s an issue with the selected model (x). It may not exist."}'"#,
        );
        let e = CliAgent::new(CliAgentConfig { tool: "claude".into(), ..cfg("claude", &claude0) }).complete("hi").await.unwrap_err();
        assert_eq!(e.agent_error().unwrap().kind.key(), "invalid_model");

        // Nothing printed at all: an exit code in words, never Rust's Debug.
        let mute = write_fake_cli(tmp.path(), "mute", "exit 1");
        let e = CliAgent::new(CliAgentConfig { tool: "agy".into(), ..cfg("agy", &mute) }).complete("hi").await.unwrap_err();
        assert_eq!(e.to_string(), "agy: exit code 1 — agy exited with code 1 and printed nothing");
    }

    // ---- integration: fake CLI scripts (unix only) ---------------------------

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_claude_describe() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // Fake claude: echo the JSON response format
        let answer = r#"{\"description\":\"two boards\",\"visible_text\":[],\"objects\":[\"board\"],\"setting\":\"desk\",\"shot\":\"close-up\",\"tags\":[]}"#;
        let script = format!(r#"printf '%s\n' '{{"result": "{answer}", "cost_usd": 0.04}}'"#);
        let bin = write_fake_cli(tmp.path(), "claude", &script);

        let agent = CliAgent::new(cfg("claude", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("two boards"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_agy_describe() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let answer = r#"{\"description\":\"a mountain\",\"visible_text\":[],\"objects\":[\"peak\"],\"setting\":\"outdoors\",\"shot\":\"wide\",\"tags\":[]}"#;
        let script = format!(r#"printf '%s\n' '{{"response": "{answer}", "duration_ms": 1200}}'"#);
        let bin = write_fake_cli(tmp.path(), "agy", &script);

        let agent = CliAgent::new(cfg("agy", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a mountain"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_opencode_describe_with_noise() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // opencode emits plugin noise then the JSON
        let script = r#"printf '[opencode-lmstudio] loading...\n{"description":"a desk","visible_text":[],"objects":["laptop"],"setting":"office","shot":"medium","tags":[]}\n'"#;
        let bin = write_fake_cli(tmp.path(), "opencode", script);

        let agent = CliAgent::new(cfg("opencode", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a desk"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fenced_json_response_is_handled() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // claude returns fenced JSON in the result field
        let result_val = r#"```json\n{\"description\":\"a monitor\",\"visible_text\":[],\"objects\":[],\"setting\":\"office\",\"shot\":\"close-up\",\"tags\":[]}\n```"#;
        let script = format!(r#"printf '%s\n' '{{"result": "{result_val}"}}'  "#);
        let bin = write_fake_cli(tmp.path(), "claude", &script);

        let agent = CliAgent::new(cfg("claude", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a monitor"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_codex_describe() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        // codex emits JSONL events; only the last agent_message counts.
        let script = r#"printf '%s\n' \
'{"type":"thread.started","thread_id":"t1"}' \
'{"type":"turn.started"}' \
'{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"{\"description\":\"a canyon\",\"visible_text\":[],\"objects\":[],\"setting\":\"outdoors\",\"shot\":\"wide\",\"tags\":[]}"}}' \
'{"type":"turn.completed","usage":{"output_tokens":9}}'"#;
        let bin = write_fake_cli(tmp.path(), "codex", script);

        let agent = CliAgent::new(cfg("codex", &bin));
        let json = agent.describe(&img, "schema").await.unwrap();
        assert!(json.contains("a canyon"), "got: {json}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn codex_argv_is_headless_and_attaches_the_image() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        let bin = tmp.path().join("codex");

        let agent = CliAgent::new(cfg("codex", &bin));
        let argv: Vec<String> =
            agent.build_argv_describe(&bin, &img, "schema").iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(argv.contains(&"exec".to_string()), "{argv:?}");
        assert!(argv.contains(&"--json".to_string()), "{argv:?}");
        assert!(argv.contains(&"--skip-git-repo-check".to_string()), "{argv:?}");
        // The frame is attached with -i rather than described by path in the prompt.
        let i = argv.iter().position(|a| a == "-i").expect("-i");
        assert_eq!(argv[i + 1], img.to_string_lossy());
    }

    #[test]
    fn codex_ignores_non_message_events_and_takes_the_last() {
        let raw = concat!(
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"reasoning\",\"text\":\"thinking\"}}\n",
            "not json at all\n",
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"first\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"last\"}}\n",
        );
        assert_eq!(CliAgent::extract_text("codex", raw).unwrap(), "last");
        assert!(CliAgent::extract_text("codex", "{\"type\":\"turn.started\"}").is_err());
    }

    #[test]
    fn codex_failure_reports_the_api_error_not_stderr() {
        let out = r#"{"type":"turn.started"}
{"type":"error","message":"{\"type\":\"error\",\"status\":400,\"error\":{\"message\":\"The 'x' model is not supported.\"}}"}
{"type":"turn.failed","error":{"message":"{\"error\":{\"message\":\"The 'x' model is not supported.\"}}"}}"#;
        assert_eq!(jsonl_error(out).as_deref(), Some("The 'x' model is not supported."));
        assert_eq!(jsonl_error("plain text"), None);
    }

    #[test]
    fn a_tool_outside_path_is_still_found_in_the_usual_install_dirs() {
        // A desktop-launched app gets a bare PATH; the tool lives in ~/.local/bin.
        let home = tempfile::tempdir().unwrap();
        let bindir = home.path().join(".local/bin");
        std::fs::create_dir_all(&bindir).unwrap();
        let exe = bindir.join(if cfg!(windows) { "codex.exe" } else { "codex" });
        std::fs::write(&exe, b"#!/bin/sh\n").unwrap();

        let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        let prev_home = std::env::var_os(key);
        let prev_path = std::env::var_os("PATH");
        // SAFETY: single-threaded test; restored below.
        unsafe {
            std::env::set_var(key, home.path());
            std::env::set_var("PATH", "/nonexistent-bin");
        }
        let found = find_binary(&CliAgentConfig { tool: "codex".into(), ..Default::default() });
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
            match prev_path {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }
        assert_eq!(found.as_deref(), Some(exe.as_path()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn non_zero_exit_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let bin = write_fake_cli(tmp.path(), "claude", "exit 1");
        let agent = CliAgent::new(cfg("claude", &bin));
        let err = agent.describe(&img, "schema").await.unwrap_err();
        assert!(err.to_string().contains("exited"), "got: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("frame.jpg");
        std::fs::write(&img, b"fake").unwrap();

        let bin = write_fake_cli(tmp.path(), "claude", "sleep 60");
        let short_cfg = CliAgentConfig {
            tool: "claude".to_string(),
            command: bin.to_str().unwrap_or("").to_string(),
            timeout_secs: 1,
            concurrency: 1,
            ..Default::default()
        };
        let agent = CliAgent::new(short_cfg);
        let err = agent.describe(&img, "schema").await.unwrap_err();
        assert!(err.to_string().contains("timed out"), "got: {err}");
    }

    #[test]
    fn find_binary_uses_command_field() {
        let tmp = tempfile::tempdir().unwrap();
        // A non-existent path: should return None.
        let missing_cfg = CliAgentConfig {
            tool: "claude".into(),
            command: tmp.path().join("nonexistent").to_str().unwrap_or("").into(),
            ..Default::default()
        };
        assert!(find_binary(&missing_cfg).is_none());

        // An existing file: should return Some.
        let real = tmp.path().join("mybin");
        std::fs::write(&real, b"fake").unwrap();
        let real_cfg =
            CliAgentConfig { tool: "claude".into(), command: real.to_str().unwrap_or("").into(), ..Default::default() };
        assert_eq!(find_binary(&real_cfg), Some(real));
    }
}

/// The models a coding-agent CLI will accept, asked of the tool itself.
///
/// Every one of these has its own catalogue that changes without us, so the list is fetched
/// rather than kept here — a hard-coded one would be wrong within a month. `opencode models` and
/// `agy models` print one per line; `claude` has no such command and `codex` wants a terminal, so
/// those return nothing and the field stays free text.
///
/// Runs with stdin closed and a short timeout: this is called to fill a menu, and a tool that
/// decides to wait for input must not hang the window.
pub fn list_models(cfg: &CliAgentConfig) -> Vec<String> {
    if cfg.tool == "claude" {
        return vec![
            "opus",
            "sonnet",
            "haiku",
            "fable",
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
    }
    if cfg.tool == "codex" {
        let codex_home = std::env::var_os("CODEX_HOME")
            .map(PathBuf::from)
            .or_else(home_dir)
            .map(|p| if std::env::var_os("CODEX_HOME").is_some() { p } else { p.join(".codex") });
        let Some(dir) = codex_home else { return Vec::new() };
        let mut models = std::fs::read_to_string(dir.join("models_cache.json"))
            .ok()
            .map(|s| parse_codex_models(&s))
            .unwrap_or_default();
        if let Ok(config) = std::fs::read_to_string(dir.join("config.toml")) {
            if let Some(model) = config
                .lines()
                .find_map(|l| l.trim().strip_prefix("model = ").map(|v| v.trim().trim_matches(['"', '\'']).to_string()))
                .filter(|s| !s.is_empty())
            {
                models.push(model);
            }
        }
        models.sort();
        models.dedup();
        return models;
    }
    let Some(bin) = find_binary(cfg) else { return Vec::new() };
    if !matches!(cfg.tool.as_str(), "opencode" | "agy") {
        return Vec::new();
    }

    let mut cmd = crate::proc::std_command(&bin);
    cmd.arg("models").stdin(std::process::Stdio::null());
    let Ok(out) = cmd.output() else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }

    parse_model_list(&String::from_utf8_lossy(&out.stdout))
}

fn parse_codex_models(text: &str) -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|v| v.get("models").and_then(|m| m.as_array()).cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|m| {
            (m.get("visibility").and_then(|v| v.as_str()) == Some("list"))
                .then(|| m.get("slug").and_then(|v| v.as_str()).map(str::to_string))
                .flatten()
        })
        .collect()
}

/// Pull model ids out of what a tool prints.
///
/// Both print one per line as `<id>` or `<id>\t<description>`, mixed with prose it writes while
/// it works and, in opencode's case, `[provider]` headings. An id has a `/` or a `-` in it; a
/// heading is bracketed and a status line ends in an ellipsis.
fn parse_model_list(text: &str) -> Vec<String> {
    let mut models: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('['))
        .filter_map(|l| l.split(['\t', ' ']).next())
        .filter(|id| id.contains(['/', '-']) && !id.ends_with("...") && !id.starts_with('-'))
        .map(str::to_string)
        .collect();
    models.sort();
    models.dedup();
    models
}

#[cfg(test)]
mod model_list_tests {
    use super::{parse_codex_models, parse_model_list};

    #[test]
    fn codex_cache_only_returns_visible_slugs() {
        let cache = r#"{"models":[{"slug":"a","visibility":"list"},{"slug":"b","visibility":"hide"},{"slug":"c","visibility":"list"}]}"#;
        assert_eq!(parse_codex_models(cache), vec!["a", "c"]);
    }

    #[test]
    fn claude_model_catalog_is_static() {
        let cfg = crate::config::CliAgentConfig { tool: "claude".into(), ..Default::default() };
        assert_eq!(
            super::list_models(&cfg),
            vec![
                "opus",
                "sonnet",
                "haiku",
                "fable",
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-sonnet-5-5",
                "claude-haiku-5-5"
            ]
        );
    }

    /// Real output from both tools, trimmed: a heading, a status line, tab-separated descriptions.
    #[test]
    fn model_ids_are_picked_out_of_what_the_tools_print() {
        let opencode = "[opencode-go]\nopencode-go/glm-5.3\nopencode-go/glm-5.3-flash\n\
                        [opencode-lmstudio]\nlmstudio/bonsai-27b\n";
        assert_eq!(
            parse_model_list(opencode),
            vec!["lmstudio/bonsai-27b", "opencode-go/glm-5.3", "opencode-go/glm-5.3-flash"]
        );

        let agy = "Fetching available models...\n\
                   gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n\
                   claude-sonnet-4-6\tClaude Sonnet 4.6\n";
        assert_eq!(parse_model_list(agy), vec!["claude-sonnet-4-6", "gemini-3.8-flash-high"]);

        // Nothing usable in, nothing out — the field stays free text.
        assert!(parse_model_list("no models configured\n").is_empty());
    }
}

#[cfg(test)]
mod brief_tests {
    use super::{CliAgent, MAX_ARGV_CHARS, as_plain_brief, read_the_prompt, spill_prompt};
    use crate::config::CliAgentConfig;
    use std::path::Path;

    /// opencode read the ChatML transcript as someone else's conversation and refused to play:
    /// "I'm opencode, not your tool runtime." The same content, framed as its own brief, is a job.
    #[test]
    fn a_transcript_becomes_a_brief_it_can_act_on() {
        let transcript = "<|im_start|>system\nYou are an editor. Reply with JSON.<|im_end|>\n\
                          <|im_start|>user\nMake a 40 second teaser.<|im_end|>\n\
                          <|im_start|>assistant\n{\"action\":\"tool\"}<|im_end|>\n";
        let brief = as_plain_brief(transcript);

        assert!(!brief.contains("<|im_start|>"), "the markers are gone:\n{brief}");
        assert!(!brief.contains("<|im_end|>"));
        assert!(brief.starts_with("You are the editor described below"), "the job is stated first");
        assert!(brief.contains("You are an editor. Reply with JSON."));
        assert!(brief.contains("WHAT YOU WERE ASKED\nMake a 40 second teaser."));
        assert!(brief.contains("WHAT YOU ANSWERED LAST"));
    }

    /// A prompt too long for the platform's command line is written down and pointed at, rather
    /// than failing on spawn with "The filename or extension is too long" — a Windows message
    /// that blames the binary for an argument's size.
    #[test]
    fn a_prompt_too_long_for_argv_is_spilled_to_a_file() {
        // The threshold is platform-dependent on purpose, so drive the helper directly: on unix
        // it is a megabyte, and the working path must never spill. Asserting on the constant
        // itself is not a test, it is a restatement.
        assert!(spill_prompt("a short prompt").is_none(), "an ordinary prompt is passed as an argument");

        let huge = "x".repeat(MAX_ARGV_CHARS + 1);
        let Some((file, dir)) = spill_prompt(&huge) else { panic!("a prompt past the limit must spill") };
        assert_eq!(std::fs::read_to_string(&file).unwrap().len(), huge.len(), "written whole");
        assert!(file.starts_with(&dir));

        // What the agent is told instead names the file and nothing else.
        let told = read_the_prompt(&file);
        assert!(told.contains(&file.display().to_string()));
        assert!(told.contains("do exactly what it says"));
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn spilling_grants_the_agent_the_directory_it_must_read() {
        let dir = std::env::temp_dir().join("ghostreel-prompts");
        {
            let tool = "agy";
            let agent = CliAgent::new(CliAgentConfig { tool: tool.into(), ..Default::default() });
            let argv = agent.build_argv_complete(Path::new("/bin/true"), &"x".repeat(MAX_ARGV_CHARS + 1), false);
            let flat: Vec<String> = argv.iter().map(|a| a.to_string_lossy().into_owned()).collect();

            assert!(flat.iter().any(|a| a == "--add-dir"), "{tool}: {flat:?}");
            assert!(flat.iter().any(|a| a == &dir.display().to_string()), "{tool}: {flat:?}");
            // And the prompt itself is a short instruction, not the payload.
            assert!(flat.iter().all(|a| a.len() < 1000), "{tool} still passes the payload in argv");
        }
    }

    /// claude told to "read the file" answered about the file ("it's a system prompt for…")
    /// instead of producing JSON. It gets the prompt on stdin as a plain brief, never a file.
    #[test]
    fn claude_takes_the_prompt_on_stdin_as_a_plain_brief() {
        let claude = CliAgent::new(CliAgentConfig { tool: "claude".into(), ..Default::default() });
        let huge = format!("<|im_start|>system\nYou are an editor.<|im_end|>\n{}", "x".repeat(MAX_ARGV_CHARS + 1));
        let argv = claude.build_argv_complete(Path::new("/bin/true"), &huge, false);
        let flat: Vec<String> = argv.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(flat.iter().all(|a| a.len() < 1000), "no payload in argv: {flat:?}");
        assert!(!flat.iter().any(|a| a.contains("ghostreel-prompts")), "no spilled file: {flat:?}");
        assert!(flat.contains(&"none".to_string()), "no tools needed: {flat:?}");

        let stdin = claude.stdin_prompt(&huge, false).expect("claude reads stdin");
        assert!(!stdin.contains("<|im_start|>"));
        assert!(stdin.starts_with("You are the editor described below"));
        assert!(
            CliAgent::new(CliAgentConfig { tool: "agy".into(), ..Default::default() })
                .stdin_prompt("p", false)
                .is_none()
        );
    }

    /// `--continue` resumed whatever claude session ran last in the folder. The agent pins its
    /// own id on the first call and resumes exactly that one.
    #[test]
    fn claude_resumes_its_own_session_not_the_latest() {
        let claude = CliAgent::new(CliAgentConfig { tool: "claude".into(), ..Default::default() });
        let flat = |argv: Vec<std::ffi::OsString>| -> Vec<String> {
            argv.iter().map(|a| a.to_string_lossy().into_owned()).collect()
        };
        let first = flat(claude.build_argv_complete_from(Path::new("/bin/true"), "p", false));
        let i = first.iter().position(|a| a == "--session-id").expect("--session-id");
        let id = first[i + 1].clone();
        assert_eq!(id.len(), 36);
        assert_eq!(&id[14..15], "4");

        let next = flat(claude.build_argv_complete_from(Path::new("/bin/true"), "p", true));
        assert!(!next.contains(&"--continue".to_string()), "{next:?}");
        let r = next.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(next[r + 1], id);
    }
}
