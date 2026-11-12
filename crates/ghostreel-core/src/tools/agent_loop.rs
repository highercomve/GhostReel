//! The action loop for jobs that end in a JSON answer: a clip worker scoring a window of a
//! transcript, the orchestrator choosing the final clips.
//!
//! It is the loop the script chat runs, over the same registry: the model may look things up
//! first — read more of a transcript, see what a picture shows, check who is speaking — and still
//! ends in the one JSON object its schema describes. The prompt already carries what the job
//! needs (the transcript slice, the candidates), so a model that calls no tool answers at once and
//! costs one request, as before.
//!
//! The three kinds of backend are driven the way the chat drives them: a server is given the tools
//! as an OpenAI `tools` array and ends with a call held to the schema; the local helper and a CLI
//! agent read the tools as prose in the prompt and answer in `{"action": …}` JSON, the local one
//! under a grammar. Every call goes through a [`ToolSession`], so the cap on rounds (none by
//! default), the repeat guard, the result size limits and the error wording are the chat's.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::{Value, json};

use super::{Detail, Origin, RenderOpts, ToolBudget, ToolCtx, ToolSession, action_branches, render_openai, tools_prose};
use crate::Error;
use crate::agents::{AgentError, AgentErrorKind};
use crate::chat::{ChatBackend, workers};
use crate::config::Config;
use crate::db::Db;
use crate::embed::Embedder;
use crate::paths::Paths;
use crate::projects::Project;

/// How a job reaches the registry.
#[derive(Debug, Clone)]
pub struct ToolAccess {
    pub paths: Paths,
    pub config: Arc<Config>,
    /// The project the job works on: the default `project` of every call.
    pub project_id: i64,
    pub cancel: Option<Arc<AtomicBool>>,
    /// An optional cap on the rounds of tool calls (0 = none, the default).
    pub max_rounds: u32,
    /// Where the job's calls are noted ("tool: summary"), for its run log.
    pub trail: Option<ToolTrail>,
}

/// The calls one job made, one line each, shared with whoever logs the job.
pub type ToolTrail = Arc<std::sync::Mutex<Vec<String>>>;

impl ToolAccess {
    pub fn new(paths: Paths, config: Arc<Config>, project_id: i64, cancel: Option<Arc<AtomicBool>>) -> Self {
        Self { paths, config, project_id, cancel, max_rounds: 0, trail: None }
    }

    /// The same access noting its calls in a fresh trail, returned for the job's log.
    pub fn with_trail(&self) -> (Self, ToolTrail) {
        let trail = ToolTrail::default();
        (Self { trail: Some(trail.clone()), ..self.clone() }, trail)
    }

    /// The same access with the agent's own cap on rounds.
    pub fn with_max_rounds(&self, max_rounds: u32) -> Self {
        Self { max_rounds, ..self.clone() }
    }
}

/// The window a server job is assumed to have: conversations are trimmed to three quarters of it.
const SERVER_WINDOW_TOKENS: usize = 32_768;

const PROTOCOL: &str = "You can look things up before you answer, with the tools listed below. Reply with ONE JSON \
object each time: either a tool action {\"action\":\"tool\",\"tool\":\"<name>\",\"args\":{...}}, or, when you are \
ready, {\"action\":\"final\",\"answer\":<your answer>}. Everything the request needs is usually in it already: call \
a tool only to check what it leaves open (what a picture shows, more of a transcript, who is speaking), and answer \
as soon as you can. Tools marked [changes things] or [destructive] act on the real project: use one only if the \
request asks for it.";

const SERVER_PROTOCOL: &str = "You can call the tools you are given to check what the request leaves open (what a \
picture shows, more of a transcript, who is speaking), but everything the request needs is usually in it already: \
answer as soon as you can. Tools that change things act on the real project: use one only if the request asks for \
it. When you answer, reply with ONLY the JSON object the request describes: no prose, no code fences.";

fn unusable(msg: impl Into<String>) -> Error {
    Error::Agent(AgentError::new("", AgentErrorKind::BadAnswer, msg))
}

/// Whether `v` has every field the schema requires at its top level.
fn satisfies(schema: &Value, v: &Value) -> bool {
    v.is_object()
        && schema["required"]
            .as_array()
            .map(|req| req.iter().filter_map(Value::as_str).all(|k| v.get(k).is_some()))
            .unwrap_or(true)
}

/// What a model's reply to the action protocol asked for.
enum Step {
    Tool { name: String, args: Value },
    Answer(Value),
}

/// Read one reply: a tool action, a final answer in its wrapper, or (from a model that skipped
/// the wrapper) the answer itself.
fn read_step(reply: Value, schema: &Value) -> Result<Step, Error> {
    match reply.get("action").and_then(Value::as_str) {
        Some("tool") => {
            let name = reply.get("tool").and_then(Value::as_str).unwrap_or_default().to_string();
            Ok(Step::Tool { name, args: reply.get("args").cloned().filter(Value::is_object).unwrap_or_else(|| json!({})) })
        }
        Some("final") => match reply.get("answer") {
            Some(a) if satisfies(schema, a) => Ok(Step::Answer(a.clone())),
            _ => Err(unusable("the final answer does not match the schema")),
        },
        _ if satisfies(schema, &reply) => Ok(Step::Answer(reply)),
        _ => Err(unusable(format!("not a tool action or an answer: {}", super::util::clip_chars(&reply.to_string(), 160)))),
    }
}

/// `{"action":"final","answer":<schema>}` alone, or with every tool branch beside it.
fn action_schema(answer: &Value, opts: Option<RenderOpts>) -> Value {
    let mut branches = vec![json!({
        "type": "object",
        "properties": { "action": { "type": "string", "enum": ["final"] }, "answer": answer },
        "required": ["action", "answer"],
        "additionalProperties": false
    })];
    if let Some(opts) = opts {
        branches.extend(action_branches(opts));
    }
    if branches.len() == 1 { branches.remove(0) } else { json!({ "type": "object", "oneOf": branches }) }
}

/// One job's resources: its own database connection (a job runs on a task of its own), its
/// embedding model when a search asks for one, and the session every call goes through.
struct Run<'a> {
    access: &'a ToolAccess,
    db: Db,
    embedder: Option<Embedder>,
    project: Project,
    session: ToolSession,
    budget: ToolBudget,
    detail: Detail,
}

impl<'a> Run<'a> {
    fn open(access: &'a ToolAccess, backend: &ChatBackend) -> Result<Self, Error> {
        let db = Db::open(&access.paths.db_file())?;
        let project = db.project(access.project_id)?;
        let script = &access.config.script;
        let (max_chars, detail) = match backend {
            ChatBackend::Local { ctx_tokens, .. } => (
                script.local_tool_result_chars,
                match script.tool_docs.as_str() {
                    "full" => Detail::Full,
                    "short" => Detail::Short,
                    _ if *ctx_tokens < 65_536 => Detail::Short,
                    _ => Detail::Full,
                },
            ),
            _ => (
                script.roomy_tool_result_chars,
                if script.tool_docs == "short" { Detail::Short } else { Detail::Full },
            ),
        };
        Ok(Self {
            access,
            db,
            embedder: None,
            project,
            session: ToolSession::new(max_chars),
            budget: ToolBudget::from_setting(access.max_rounds),
            detail,
        })
    }

    fn opts(&self) -> RenderOpts {
        RenderOpts::new(self.detail, true)
    }

    fn check_cancel(&self) -> Result<(), Error> {
        if self.access.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst)) {
            return Err(Error::Invalid(crate::chat::CANCELLED.into()));
        }
        Ok(())
    }

    async fn call(&mut self, name: &str, args: &Value) -> super::Called {
        let mut ctx = ToolCtx::new(&self.access.paths, &self.access.config, &mut self.db, &mut self.embedder, Origin::ClipRun)
            .with_project(self.project.clone())
            .with_cancel(self.access.cancel.clone())
            .on_demand_embedder();
        let called = self.session.call(&mut ctx, name, args).await;
        if let Some(trail) = &self.access.trail
            && let Ok(mut t) = trail.lock()
        {
            t.push(format!("{}: {}", called.tool, called.summary));
        }
        called
    }
}

/// Ask a backend for one JSON object matching `schema`, letting it use the tool registry first
/// when `access` is given. Without tools this is the plain single request
/// ([`workers::ask_json`]).
pub async fn run_agent_json(
    backend: &mut ChatBackend,
    prompt: &str,
    schema: &Value,
    max_tokens: usize,
    timeout_s: u64,
    access: Option<&ToolAccess>,
) -> Result<Value, Error> {
    let Some(access) = access else {
        return workers::ask_json(backend, prompt, schema, max_tokens, timeout_s).await;
    };
    let mut run = Run::open(access, backend)?;
    match backend {
        ChatBackend::Server { url, model, api_key, .. } => {
            server_loop(&mut run, (url, model, api_key), prompt, schema, max_tokens, timeout_s).await
        }
        ChatBackend::Local { helper, ctx_tokens, think } => {
            local_loop(&mut run, helper, *ctx_tokens, *think, prompt, schema, max_tokens).await
        }
        ChatBackend::Cli(agent) => cli_loop(&mut run, agent, prompt, schema).await,
    }
}

async fn server_loop(
    run: &mut Run<'_>,
    (url, model, api_key): (&str, &str, &str),
    prompt: &str,
    schema: &Value,
    max_tokens: usize,
    timeout_s: u64,
) -> Result<Value, Error> {
    let tools = render_openai(run.opts());
    let mut messages =
        vec![json!({ "role": "system", "content": SERVER_PROTOCOL }), json!({ "role": "user", "content": prompt })];
    let window = SERVER_WINDOW_TOKENS * 3 / 4;
    loop {
        run.check_cancel()?;
        if run.budget.exhausted() || !crate::chat::make_room(&mut messages, window) {
            break;
        }
        run.budget.spend();
        let msg = workers::server_complete(url, model, api_key, &messages, Some(&tools), None, max_tokens, timeout_s).await?;
        let calls = msg["tool_calls"].as_array().filter(|c| !c.is_empty()).cloned();
        let Some(calls) = calls else {
            // No tool wanted: the content is the answer, if it is one.
            if let Ok(v) = workers::parse_object(msg["content"].as_str().unwrap_or_default())
                && let Ok(Step::Answer(a)) = read_step(v, schema)
            {
                return Ok(a);
            }
            messages.push(json!({ "role": "assistant", "content": msg["content"] }));
            break;
        };
        messages.push(msg.clone());
        let mut stuck = false;
        for tc in calls {
            let func = tc.get("function").cloned().unwrap_or(Value::Null);
            let name = func.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            let args: Value = func
                .get("arguments")
                .and_then(|a| if let Some(s) = a.as_str() { serde_json::from_str(s).ok() } else { Some(a.clone()) })
                .unwrap_or_else(|| json!({}));
            let called = run.call(&name, &args).await;
            if called.cancelled {
                return Err(Error::Invalid(crate::chat::CANCELLED.into()));
            }
            stuck |= called.stop;
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tc.get("id").and_then(Value::as_str).unwrap_or_default(),
                "content": called.model_text,
            }));
        }
        if stuck {
            break;
        }
    }
    // The answer, held to the schema.
    messages.push(json!({
        "role": "user",
        "content": "Now give your final answer: ONLY the JSON object the request describes, using what you have found."
    }));
    let msg = workers::server_complete(url, model, api_key, &messages, None, Some(schema), max_tokens, timeout_s).await?;
    let v = workers::parse_object(msg["content"].as_str().unwrap_or_default()).map_err(|e| unusable(e.to_string()))?;
    match read_step(v, schema)? {
        Step::Answer(a) => Ok(a),
        Step::Tool { .. } => Err(unusable("asked for a tool instead of answering")),
    }
}

async fn local_loop(
    run: &mut Run<'_>,
    helper: &mut crate::vision::LocalLlm,
    ctx_tokens: u32,
    think: bool,
    prompt: &str,
    schema: &Value,
    max_tokens: usize,
) -> Result<Value, Error> {
    let output = crate::chat::local_output_budget(ctx_tokens, max_tokens as u32);
    let grammar = action_schema(schema, Some(run.opts()));
    let mut transcript = format!(
        "<|im_start|>system\n{PROTOCOL}{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n",
        tools_prose(run.opts())
    );
    loop {
        run.check_cancel()?;
        if run.budget.exhausted() {
            break;
        }
        run.budget.spend();
        crate::chat::fit_local_prompt(helper, &mut transcript, prompt, ctx_tokens, output, think, None).await?;
        let out = helper.complete_full(&transcript, Some(grammar.clone()), output, think).await?;
        let reply = workers::parse_object(&out).map_err(|e| unusable(e.to_string()))?;
        match read_step(reply, schema)? {
            Step::Answer(a) => return Ok(a),
            Step::Tool { name, args } => {
                let called = run.call(&name, &args).await;
                if called.cancelled {
                    helper.kill().await;
                    return Err(Error::Invalid(crate::chat::CANCELLED.into()));
                }
                transcript.push_str(&format!(
                    "<|im_start|>assistant\n{out}\n<|im_end|>\n<|im_start|>user\nTool result for {}:\n{}\n<|im_end|>\n",
                    called.tool, called.model_text
                ));
                if called.stop {
                    break;
                }
            }
        }
    }
    transcript.push_str("<|im_start|>user\nGive your final answer now, using what you have found.<|im_end|>\n");
    crate::chat::fit_local_prompt(helper, &mut transcript, prompt, ctx_tokens, output, think, None).await?;
    let out = helper.complete_full(&transcript, Some(action_schema(schema, None)), output, think).await?;
    let reply = workers::parse_object(&out).map_err(|e| unusable(e.to_string()))?;
    match read_step(reply, schema)? {
        Step::Answer(a) => Ok(a),
        Step::Tool { .. } => Err(unusable("asked for a tool instead of answering")),
    }
}

async fn cli_loop(
    run: &mut Run<'_>,
    agent: &crate::cliagent::CliAgent,
    prompt: &str,
    schema: &Value,
) -> Result<Value, Error> {
    let schema_text = serde_json::to_string(schema).unwrap_or_default();
    let mut next = format!(
        "{PROTOCOL}{}\n\nThe answer must be a JSON object matching this JSON schema:\n{schema_text}\n\nThe request:\n{prompt}\n\nReply ONLY with one JSON object (no prose, no code fences).",
        tools_prose(run.opts())
    );
    let mut continued = false;
    loop {
        run.check_cancel()?;
        if run.budget.exhausted() {
            break;
        }
        run.budget.spend();
        // The first round carries the whole request; after that the tool continues its own
        // conversation and hears only what is new.
        let out = agent.complete_continuing(&next, continued).await?;
        continued = true;
        let reply = workers::parse_object(&out).map_err(|e| unusable(e.to_string()))?;
        match read_step(reply, schema)? {
            Step::Answer(a) => return Ok(a),
            Step::Tool { name, args } => {
                let called = run.call(&name, &args).await;
                if called.cancelled {
                    return Err(Error::Invalid(crate::chat::CANCELLED.into()));
                }
                next = format!(
                    "Tool result for {}:\n{}\n\nReply ONLY with one JSON object: another tool action, or {{\"action\":\"final\",\"answer\":...}}.",
                    called.tool, called.model_text
                );
                if called.stop {
                    break;
                }
            }
        }
    }
    // A fresh session (the cap was zero or nothing ran) has not seen the request yet.
    let ask = if continued {
        "Give your final answer now. Reply ONLY with {\"action\":\"final\",\"answer\":<a JSON object matching the schema>}.".to_string()
    } else {
        format!("{prompt}\n\nReply ONLY with one JSON object matching this JSON schema (no prose, no code fences):\n{schema_text}")
    };
    let out = agent.complete_continuing(&ask, continued).await?;
    let reply = workers::parse_object(&out).map_err(|e| unusable(e.to_string()))?;
    match read_step(reply, schema)? {
        Step::Answer(a) => Ok(a),
        Step::Tool { .. } => Err(unusable("asked for a tool instead of answering")),
    }
}

/// For the tests: the grammar a local worker is held to.
#[cfg(test)]
pub(super) fn action_schema_for_test(answer: &Value, opts: RenderOpts) -> Value {
    action_schema(answer, Some(opts))
}

/// For the tests: what a reply is read as.
#[cfg(test)]
pub(super) fn read_step_for_test(reply: Value, schema: &Value) -> String {
    match read_step(reply, schema) {
        Ok(Step::Answer(_)) => "answer".into(),
        Ok(Step::Tool { name, .. }) => format!("tool {name}"),
        Err(_) => "unusable".into(),
    }
}
