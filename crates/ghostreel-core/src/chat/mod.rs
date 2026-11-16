//! Project-scoped script chat agent (plan §4a, M8c).
//!
//! Grounded script writing through iterative tool calling over local footage
//! and transcripts. Supports server backends (OpenAI-compatible) and local
//! `ghostreel-llm` helper backends.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub mod build;
pub mod judge;
pub mod metrics;
pub mod refine;
pub mod repair;
mod research;
#[cfg(test)]
mod review_tests;
pub use enforce_grounding_and_pacing as repair_draft;
pub mod tools;
pub mod workers;
pub use crate::tools::{Grounding, ToolMemo};
pub use judge::Judgement;
pub use metrics::{ScriptMetrics, ScriptScore, measure, score};
pub use tools::{Detail, tools_definition};

use crate::Error;
use crate::db::Db;
use crate::embed::Embedder;
use crate::finish::{MIN_CLIP_WORDS, clip_has_speech, clip_speech};
use crate::projects::{Project, now};
use crate::script::{ChatStyle, Issue, IssueSeverity, Script, ScriptClip};
use crate::tools::{Origin, RenderOpts, ToolCtx, ToolSession};

/// Events emitted during agent execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    ToolStarted { tool: String, args: Value },
    ToolFinished { tool: String, summary: String },
    Drafting,
    Validating,
}

/// Record of an executed tool call in a chat turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args: Value,
    pub summary: String,
}

fn log_tool_record(record: &ToolCallRecord) {
    let text = serde_json::to_string(record).unwrap_or_default();
    crate::llmlog::push("tool", &text);
    if record.summary.starts_with("error:") {
        crate::llmlog::push("error", text);
    }
}

/// A stored chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatSession {
    pub id: i64,
    pub project_id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub style: ChatStyle,
}

/// A message in a chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub id: i64,
    pub session_id: i64,
    pub role: String,
    pub content: String,
    pub images: Option<Vec<String>>,
    pub tool_calls: Option<Vec<ToolCallRecord>>,
    pub script_id: Option<i64>,
    pub created_at: i64,
    #[serde(default)]
    pub performance: Option<crate::inference::ChatPerformance>,
}

/// Result of a single agent turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnResult {
    pub session_id: i64,
    pub reply: String,
    pub script_id: Option<i64>,
    pub script: Option<Script>,
    pub issues: Vec<Issue>,
    pub tool_calls: Vec<ToolCallRecord>,
    /// The editorial read, when `[jev]` is on. `None` means nothing was asked, not a good cut.
    pub judgement: Option<judge::Judgement>,
}

/// Backends for script chat.
pub enum ChatBackend {
    Server {
        url: String,
        model: String,
        api_key: String,
        /// The window the server was started with (`chat_model.ctx_tokens`). Every round re-sends
        /// the whole conversation, so the loop has to know when it is about to run out of room.
        ctx_tokens: u32,
    },
    Local {
        helper: Box<crate::vision::LocalLlm>,
        /// The helper's context window: tool results are trimmed to a fraction of it.
        ctx_tokens: u32,
        /// Whether the draft may reason before answering (`chat_model.think`).
        think: bool,
    },
    /// Coding-agent CLI (claude / agy / opencode) drives the existing local action loop.
    Cli(crate::cliagent::CliAgent),
}

impl ChatBackend {
    pub async fn from_runtime(rt: &crate::runtime::Runtime, script_timeout_s: u64) -> Result<Self, Error> {
        Self::from_vision_setup(&rt.vision, script_timeout_s).await
    }

    /// The window this backend is talking to, for deciding when a conversation has grown too
    /// long to send again. Zero when it does not apply (a CLI agent manages its own).
    pub fn window_tokens(&self) -> u32 {
        match self {
            ChatBackend::Server { ctx_tokens, .. } => *ctx_tokens,
            ChatBackend::Local { ctx_tokens, .. } => *ctx_tokens,
            ChatBackend::Cli(_) => 0,
        }
    }

    /// Records the window a server was configured with; `from_vision_setup` cannot know it.
    pub fn with_window(mut self, tokens: u32) -> Self {
        if let ChatBackend::Server { ctx_tokens, .. } = &mut self {
            *ctx_tokens = tokens;
        }
        self
    }

    /// Chat uses the vision model (Bonsai): the vision server, or the local helper with the model
    /// and mmproj (downloaded once when missing, like the describe stage).
    /// `script_timeout_s` is the floor for a CLI agent's own timeout here. `CliAgentConfig`
    /// serves two jobs with one number: describing a frame, which takes seconds, and writing a
    /// whole script, which takes minutes — agy needs about nine of them on a 96-video project.
    /// At the 180 s default the second was killed every time, reported as "agy timed out after
    /// 180s". The server path already has `script.server_timeout_s` for exactly this reason.
    pub async fn from_vision_setup(setup: &crate::runtime::VisionSetup, script_timeout_s: u64) -> Result<Self, Error> {
        match setup {
            crate::runtime::VisionSetup::Server(s) => {
                // The caller sets the window with `with_window`: a VisionSetup does not carry it.
                Ok(ChatBackend::Server {
                    url: s.url.clone(),
                    model: s.model.clone(),
                    api_key: s.api_key.clone(),
                    ctx_tokens: 32768,
                })
            }
            crate::runtime::VisionSetup::Cli(cfg) => {
                Ok(ChatBackend::Cli(crate::cliagent::CliAgent::new(crate::config::CliAgentConfig {
                    timeout_secs: cfg.timeout_secs.max(script_timeout_s),
                    ..cfg.clone()
                })))
            }
            crate::runtime::VisionSetup::Mlx { helper, models_dir, entry, runtime } => {
                let dir = crate::models::download_bundle(entry, models_dir, |_, _| {}).await?;
                let llm = crate::vision::LocalLlm::start_mlx(helper, &dir, runtime).await?;
                Ok(ChatBackend::Local {
                    ctx_tokens: llm.ctx_tokens.unwrap_or(runtime.ctx_tokens),
                    helper: Box::new(llm),
                    think: runtime.think,
                })
            }
            crate::runtime::VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime } => {
                let mut paths = Vec::with_capacity(2);
                for spec in std::iter::once(model).chain(mmproj.iter()) {
                    if let Some(p) = found.iter().find(|p| {
                        p == &std::path::Path::new(&spec.file_name)
                            || p.file_name().is_some_and(|n| n == spec.file_name.as_str())
                    }) {
                        paths.push(p.clone());
                    } else {
                        paths.push(crate::models::download(spec, models_dir, |_, _| {}).await?);
                    }
                }
                let vision = Some((paths[0].clone(), paths.get(1).cloned()));
                let models = crate::vision::LocalModels {
                    helper: helper.clone(),
                    vision,
                    embed: None,
                    cpu: false,
                    runtime: runtime.clone(),
                    concurrency: 1,
                    free_ghostpen: crate::ghost_service::models_to_free(None),
                };
                let llm = crate::vision::LocalLlm::start(&models).await?;
                Ok(ChatBackend::Local {
                    ctx_tokens: llm.ctx_tokens.unwrap_or(runtime.ctx_tokens),
                    helper: Box::new(llm),
                    think: runtime.think,
                })
            }
            crate::runtime::VisionSetup::Unavailable(why) => {
                Err(Error::Invalid(format!("the model is unavailable: {why}")))
            }
        }
    }
}

/// The chat backend for a turn, from the agent pool, with the endpoint slot it holds for the turn.
pub struct ChatLease {
    pub backend: ChatBackend,
    /// The agent answering; `None` when no agent has the chat role and `[chat_model]` answers.
    pub agent: Option<crate::agents::AgentSpec>,
    /// Held until the turn ends, so no other feature asks this model for more than it allows.
    pub slot: Option<crate::agents::Slot>,
    /// The agent's `max_tool_rounds` (0 = unlimited).
    pub max_tool_rounds: u32,
    /// What it resolved to ("cli claude (claude-opus-5-5)").
    pub describe: String,
}

/// The agent that answers script chat on a cut: the run's choice over the cut's over the app's
/// (see [`crate::agents::resolve_selection`]). Waits for the endpoint's slot — a local model
/// busy describing frames or scoring clips is finished with first — and calls `on_wait` once if
/// it has to.
pub async fn lease_backend(
    paths: &crate::paths::Paths,
    config: &crate::config::Config,
    run: Option<&crate::agents::AgentSelection>,
    cut: Option<&crate::agents::AgentSelection>,
    on_wait: impl FnOnce(&str),
) -> Result<ChatLease, Error> {
    use crate::agents::{AgentRole, EndpointKey, SlotRegistry};
    let selection = config.resolve_agents(run, cut)?;
    let (setup, llm) = match &selection.chat {
        Some(a) => (
            crate::runtime::resolve_agent(paths, config, a, AgentRole::Chat).await,
            a.llm_config(config, AgentRole::Chat),
        ),
        None => (crate::runtime::resolve_chat(paths, config).await, config.chat_model()),
    };
    let capacity = selection.chat.as_ref().map(|a| a.capacity()).unwrap_or(1);
    let slot = match EndpointKey::for_setup(&setup) {
        Some(key) => {
            let registry = SlotRegistry::global();
            match registry.try_acquire(&key, capacity) {
                Some(s) => Some(s),
                None => {
                    on_wait(&format!("Waiting for {} to be free", key.label()));
                    Some(registry.acquire(&key, capacity).await)
                }
            }
        }
        None => None,
    };
    let describe = setup.describe();
    let backend =
        ChatBackend::from_vision_setup(&setup, config.script.server_timeout_s).await?.with_window(llm.ctx_tokens);
    Ok(ChatLease { backend, agent: selection.chat, slot, max_tool_rounds: llm.max_tool_rounds, describe })
}

/// The agent choice of the cut a turn works on: `script_id`'s, else the newest script of the
/// session. `None` when there is no cut yet or it has no choice of its own.
pub fn cut_agents(db: &Db, script_id: Option<i64>, session_id: Option<i64>) -> Option<crate::agents::AgentSelection> {
    let id = script_id.or_else(|| {
        let sid = session_id?;
        db.conn
            .query_row(
                "SELECT id FROM scripts WHERE session_id = ?1 AND trash_id IS NULL ORDER BY id DESC LIMIT 1",
                [sid],
                |r| r.get(0),
            )
            .ok()
    })?;
    crate::script::load(db, id).ok()?.script.agents
}

/// Context owning all resources needed for a chat turn.
pub struct ChatContext {
    pub db: Db,
    /// Where the data and the config file live: the tools the chat calls read both.
    pub paths: crate::paths::Paths,
    pub backend: ChatBackend,
    pub embedder: Option<Embedder>,
    /// House rules from Settings, added to the harness instructions; `None`/empty = none.
    pub system_prompt: Option<String>,
    /// `chat_model.max_tool_rounds`: an optional cap on model rounds; 0 (the default) = no cap.
    pub max_tool_rounds: u32,
    /// The `[script]` settings: clip lengths, target tolerances, narration pace, research budget.
    pub script: crate::config::ScriptConfig,
    /// The optional editorial judge. Default — disabled — asks nothing and sends nothing.
    pub jev: crate::config::JevConfig,
    /// Set by Stop. Checked between tool rounds and before every model call, so a turn ends
    /// within a round instead of after the whole draft.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// The cap on a turn's model rounds: none by default (`chat_model.max_tool_rounds` = 0), because
/// with the whole registry to use a small number makes no sense. A turn still ends when the model
/// drafts, when the user stops it, when the backend times out, or when the [`ToolSession`] gives up
/// on a model repeating the same call.
fn round_cap(configured: u32) -> Option<u32> {
    (configured > 0).then_some(configured)
}

/// Returned when a turn is stopped; the caller reports it as cancelled, not as a failure.
pub const CANCELLED: &str = "stopped";

/// Pacing problems the model can fix in a redraft: clips too long or too short, and (when
/// `enforce_target`) a total far from the target.
pub fn pacing_issues(script: &Script, enforce_target: bool, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    // No single shot may dominate the piece. `max_clip_s` is an absolute 30 s, which says nothing
    // useful about a 40 s cut: a local model filled one with eight clips of twenty-odd seconds,
    // every one of them "legal", and the result ran 349% over with nothing to trim — speech is
    // never scaled, and dropping beats could not reach the target without gutting the cut. The
    // only cure is upstream, while the model can still choose differently.
    let dominant_s = script
        .target_duration_s
        .filter(|t| *t > 0.0 && enforce_target)
        .map(|t| (t / 3.0).max(cfg.min_clip_s * 2.0))
        .unwrap_or(f64::MAX);
    for beat in &script.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let len = c.out_s - c.in_s;
            if len > dominant_s && len <= cfg.max_clip_s {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "clip is {len:.1} s of a {:.0} s piece (video #{} {:.1}–{:.1}); no shot should take more \
                         than a third — keep it under {dominant_s:.0} s or use fewer beats",
                        script.target_duration_s.unwrap_or_default(),
                        c.video_id,
                        c.in_s,
                        c.out_s
                    ),
                });
            }
            if len > cfg.max_clip_s {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "clip is {len:.1} s long (video #{} {:.1}–{:.1}); use a shorter excerpt",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
            } else if len < cfg.min_clip_s {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "clip is only {len:.1} s (video #{} {:.1}–{:.1}); hold each shot at least {:.0} s so viewers can see and read it",
                        c.video_id, c.in_s, c.out_s, cfg.min_clip_s
                    ),
                });
            }
        }
    }
    if let Some(target) = script.target_duration_s.filter(|t| *t > 0.0 && enforce_target) {
        let total = script.total_duration_s();
        if (total - target).abs() > target * (cfg.target_overshoot - 1.0) {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: None,
                clip_index: None,
                message: format!("total clip duration is {total:.1} s but the target is {target:.0} s"),
            });
        }
    }
    issues
}

/// What a chat turn holds its editor's draft to that needs the turn's own context, before the
/// finishing chain every cut goes through ([`crate::finish::run`], which keeps the project's
/// footage, a steady camera, the clip length, the sound and the narration for every model's cut):
/// tidy beat ids, a montage's sound, footage the model never opened that nothing is indexed for,
/// and with `enforce_target` a squeeze toward the target so the model hears about a cut far too long
/// while it can still choose differently. Revisions don't enforce the target: the user's feedback
/// ("slower", "longer") must be able to change the length. Returns the issues describing the changes.
pub fn enforce_grounding_and_pacing(
    db: &Db,
    _project_id: i64,
    s: &mut Script,
    grounding: &Grounding,
    enforce_target: bool,
    cfg: &crate::config::ScriptConfig,
) -> Vec<Issue> {
    let mut issues = tidy_beat_ids(s);
    if cfg.style.broll {
        issues.extend(settle_broll_audio(db, s, cfg.style));
    }
    for beat in &mut s.beats {
        let mut kept = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if !grounding.is_grounded(c.video_id, c.in_s, c.out_s, cfg.grounding_slack_s) {
                // The model reached for footage it never opened. That used to be fatal for the
                // clip, and with enough ungrounded picks a whole script collapsed to nothing —
                // the loudest failure in this pipeline, and it got worse the more the model
                // explored. But "not looked at" is not the same as "not real": if the range sits
                // inside the video and we have described keyframes or speech for it, the footage
                // exists and we know what is in it, so keep it and say it went unchecked.
                match verify_clip(db, c.video_id, c.in_s, c.out_s) {
                    Some(seen) => {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Info,
                            beat_id: Some(beat.id.clone()),
                            clip_index: None,
                            message: format!(
                                "kept an unopened clip after checking the footage: video #{} {:.1}–{:.1} s — {seen}",
                                c.video_id, c.in_s, c.out_s
                            ),
                        });
                    }
                    None => {
                        issues.push(Issue {
                            fixes: Vec::new(),
                            severity: IssueSeverity::Warning,
                            beat_id: Some(beat.id.clone()),
                            clip_index: None,
                            message: format!(
                                "dropped clip: nothing indexed at video #{} {:.1}–{:.1} s",
                                c.video_id, c.in_s, c.out_s
                            ),
                        });
                        continue;
                    }
                }
            }
            kept.push(c);
        }
        beat.clips = kept;
    }
    s.beats.retain(|b| !b.clips.is_empty());
    let before = s.total_duration_s();
    // In a montage nothing is speech, so every clip may be trimmed.
    let speaking = |c: &ScriptClip| !cfg.style.broll && clip_has_speech(db, c.video_id, c.in_s, c.out_s);
    if enforce_target && crate::finish::fit::trim_to_target_with(s, speaking, cfg) {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("clips shortened proportionally: {before:.1} s → {:.1} s", s.total_duration_s()),
        });
    }
    issues
}

/// A montage's sound, decided rather than drafted: no narration, no beds, and each clip muted or
/// playing its own ambience as the chat chose.
///
/// A clip over someone speaking on the microphone is dropped. Muting it was tried first: the cut
/// then held a talking head with no voice, which reads as a broken export, not as b-roll. The fit
/// that follows holds the remaining shots longer to make up the time.
fn settle_broll_audio(db: &Db, s: &mut Script, style: ChatStyle) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut voiced = 0usize;
    let mut unvoiced = 0usize;
    for beat in &mut s.beats {
        if beat.narration.as_deref().is_some_and(|n| !n.trim().is_empty()) {
            unvoiced += 1;
        }
        beat.narration = Some(String::new());
        beat.bed = None;
        let before = beat.clips.len();
        beat.clips.retain(|c| !clip_has_speech(db, c.video_id, c.in_s, c.out_s));
        voiced += before - beat.clips.len();
        for c in &mut beat.clips {
            c.audio = if style.natural_sound { crate::script::Audio::Source } else { crate::script::Audio::Mute };
        }
    }
    s.beats.retain(|b| !b.clips.is_empty());
    if unvoiced > 0 {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("removed narration from {unvoiced} beat(s): this chat is b-roll only"),
        });
    }
    if voiced > 0 {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("dropped {voiced} clip(s) where someone speaks on the microphone"),
        });
    }
    issues
}

/// Local constrained action representation.
// A parse result that lives for one model answer: boxing the script would only add noise.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LocalAction {
    Tool {
        tool: String,
        args: Value,
    },
    Final {
        script: Script,
    },
    /// Answer in words and draft nothing. Without this the only legal moves are "search again"
    /// or "emit a whole script", so a message that is not editing feedback — a question, a pasted
    /// note, a brief the footage cannot support — still produced a script nobody asked for.
    Reply {
        text: String,
    },
}

/// JSON Schema for Script v1, friendly to llama.cpp grammar.
pub fn script_json_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "target_duration_s": { "type": "number" },
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "purpose": { "type": "string" },
                        "narration": { "type": "string" },
                        "on_screen_text": { "type": "string" },
                        "clips": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "video_id": { "type": "integer" },
                                    "in_s": { "type": "number" },
                                    "out_s": { "type": "number" },
                                    "from": { "type": "string" },
                                    "to": { "type": "string" },
                                    "audio": { "type": "string", "enum": ["source", "mute"] },
                                    "why": { "type": "string" }
                                },
                                "required": ["video_id", "audio"],
                                "additionalProperties": false
                            }
                        },
                        "bed": {
                            "type": "object",
                            "properties": {
                                "video_id": { "type": "integer" },
                                "in_s": { "type": "number" },
                                "out_s": { "type": "number" },
                                "from": { "type": "string" },
                                "to": { "type": "string" },
                                "why": { "type": "string" }
                            },
                            "required": ["video_id"],
                            "additionalProperties": false
                        },
                        "notes": { "type": "string" }
                    },
                    "required": ["id", "purpose", "narration", "clips"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["title", "beats"],
        "additionalProperties": false
    })
}

/// Local LLM action schema (oneOf tool action or final script action).
pub fn local_action_schema() -> Value {
    let mut branches = vec![
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["final"] },
                "script": script_json_schema()
            },
            "required": ["action", "script"],
            "additionalProperties": false
        }),
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["reply"] },
                "text": { "type": "string" }
            },
            "required": ["action", "text"],
            "additionalProperties": false
        }),
    ];
    // Each tool's arguments are constrained by the schema the server models get. A generic object
    // permits filenames in numeric video_id fields and wastes whole research rounds.
    branches.extend(crate::tools::action_branches(RenderOpts::new(Detail::Full, true)));
    json!({ "type": "object", "oneOf": branches })
}

/// Local LLM final action schema only.
/// The last call of a turn. `allow_reply` is false once the model has opened footage.
///
/// Leaving the reply branch here is what a grammar-constrained model reaches for at the end: one
/// local run did the whole job — searched, chose four quotes, laid out a 40 s structure with real
/// timecodes — and then handed it over as markdown prose inside a `reply`, because that branch
/// was still reachable and prose is easier than a schema. A model that has looked at the footage
/// has nothing left to ask, so the grammar stops offering.
pub fn local_final_action_schema(allow_reply: bool) -> Value {
    let final_branch = json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "enum": ["final"] },
            "script": script_json_schema()
        },
        "required": ["action", "script"],
        "additionalProperties": false
    });
    if !allow_reply {
        return final_branch;
    }
    json!({
        "type": "object",
        "oneOf": [
            final_branch,
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["reply"] },
                    "text": { "type": "string" }
                },
                "required": ["action", "text"],
                "additionalProperties": false
            }
        ]
    })
}

/// Roughly how much of a context window a conversation takes, at four characters a token.
/// Images in multimodal messages are estimated at ~1000 tokens each rather than raw base64 char length.
fn approx_tokens(messages: &[Value]) -> usize {
    let mut chars = 0usize;
    let mut image_count = 0usize;
    for m in messages {
        if let Some(content) = m.get("content") {
            if let Some(s) = content.as_str() {
                chars += s.len();
            } else if let Some(arr) = content.as_array() {
                for part in arr {
                    if part.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                        image_count += 1;
                    } else if let Some(txt) = part.get("text").and_then(|t| t.as_str()) {
                        chars += txt.len();
                    } else {
                        chars += part.to_string().len();
                    }
                }
            } else {
                chars += content.to_string().len();
            }
        }
        if let Some(role) = m.get("role").and_then(|r| r.as_str()) {
            chars += role.len();
        }
        if let Some(tools) = m.get("tool_calls") {
            chars += tools.to_string().len();
        }
    }
    (chars / 4) + (image_count * 1000)
}

/// Make room for another round by forgetting the oldest tool results.
///
/// Every round re-sends the whole conversation, so a model that keeps looking eventually fills the
/// window — Bonsai 2 made 199 tool calls and the turn died on a raw "exceeds the available context
/// size" from the server, with nothing drafted. The oldest results are the ones it has already
/// used or discarded; forgetting them keeps the research going. Returns false when there is
/// nothing left to shrink, which means it is time to draft with what we have.
///
/// The response is stubbed, not removed: an assistant message carrying `tool_calls` whose
/// responses are gone is protocol-invalid and OpenAI-compatible servers reject it outright (400),
/// which would kill the turn exactly when this exists to prevent.
const DROPPED_RESULT: &str = "[result dropped to save room]";
/// What an old assistant `tool_call`'s arguments become when even the tool results are not enough:
/// short, valid JSON, keeping `id`, `type` and `function.name` so the call/response pairing and the
/// protocol stay valid (an argument the server can parse is all a forgotten call needs).
const DROPPED_ARGS: &str = r#"{"dropped":"to save room"}"#;
pub(crate) fn make_room(messages: &mut [Value], budget_tokens: usize) -> bool {
    if approx_tokens(messages) <= budget_tokens {
        return true;
    }
    // Keep the system prompt and the first user message: the brief and the rules are why we are
    // here. Everything between is fair game, oldest first.
    while approx_tokens(messages) > budget_tokens {
        let victim = messages
            .iter()
            .enumerate()
            .skip(2)
            .find(|(_, m)| {
                m.get("role").and_then(Value::as_str) == Some("tool")
                    && m.get("content").and_then(Value::as_str) != Some(DROPPED_RESULT)
            })
            .map(|(i, _)| i);
        match victim {
            Some(i) => {
                messages[i]["content"] = json!(DROPPED_RESULT);
            }
            None => {
                // Every result is already stubbed: the arguments themselves are the room that is
                // left — one carries a whole script, and `approx_tokens` counts them. Shrink the
                // oldest assistant message's tool_call arguments, skipping the newest assistant
                // message (the round being extended keeps its calls whole). All three fields of
                // the call stay, so its responses still pair with it.
                let newest_assistant =
                    messages.iter().rposition(|m| m.get("role").and_then(Value::as_str) == Some("assistant"));
                let arg_victim = messages
                    .iter()
                    .enumerate()
                    .skip(2)
                    .find(|(i, m)| {
                        Some(*i) != newest_assistant
                            && m.get("role").and_then(Value::as_str) == Some("assistant")
                            && m.get("tool_calls").and_then(Value::as_array).is_some_and(|calls| {
                                !calls.is_empty()
                                    && calls.iter().any(|c| {
                                        c.get("function").and_then(|f| f.get("arguments")).and_then(Value::as_str)
                                            != Some(DROPPED_ARGS)
                                    })
                            })
                    })
                    .map(|(i, _)| i);
                match arg_victim {
                    Some(i) => {
                        if let Some(calls) = messages[i].get_mut("tool_calls").and_then(Value::as_array_mut) {
                            for c in calls {
                                if let Some(args) = c.get_mut("function").and_then(|f| f.get_mut("arguments"))
                                    && args.as_str() != Some(DROPPED_ARGS)
                                {
                                    *args = json!(DROPPED_ARGS);
                                }
                            }
                        }
                    }
                    None => return false,
                }
            }
        }
    }
    true
}

pub(crate) fn local_output_budget(context: u32, answer_limit: u32) -> usize {
    (context as usize / 2).clamp(512, 12288).min(answer_limit as usize)
}

/// Discard old complete exchanges while keeping the system rules, current request, and
/// newest exchange. Grounding remains in the harness and is supplied again for the draft.
fn drop_oldest_local_exchange(transcript: &mut String, brief: &str) -> bool {
    let starts: Vec<usize> = transcript.match_indices("<|im_start|>").map(|(i, _)| i).collect();
    let pinned = format!("<|im_start|>user\n{brief}<|im_end|>");
    for i in 1..starts.len().saturating_sub(2) {
        let start = starts[i];
        let end = starts[i + 1];
        if transcript[start..end].starts_with(&pinned) {
            continue;
        }
        let mut stop = end;
        // Keep both halves of a historical conversation or tool exchange together.
        let tool_pair = transcript[start..].starts_with("<|im_start|>assistant\n")
            && transcript[end..].starts_with("<|im_start|>user\nTool result for ");
        let history_pair = transcript[start..].starts_with("<|im_start|>user\n")
            && transcript[end..].starts_with("<|im_start|>assistant\n");
        if tool_pair || history_pair {
            if i + 1 >= starts.len().saturating_sub(2) {
                continue;
            }
            stop = starts.get(i + 2).copied().unwrap_or(transcript.len());
        }
        transcript.replace_range(start..stop, "");
        return true;
    }
    false
}

pub(crate) async fn fit_local_prompt(
    helper: &mut crate::vision::LocalLlm,
    transcript: &mut String,
    brief: &str,
    context: u32,
    output: usize,
    think: bool,
    image: Option<&Path>,
) -> Result<(), Error> {
    let mut dropped = false;
    loop {
        let tokens = helper.prompt_tokens(transcript, think, image).await?;
        if tokens.saturating_add(output) <= context as usize {
            if dropped {
                crate::llmlog::push(
                    "info",
                    format!(
                        "Compacted old exchanges: {tokens} prompt + {output} output tokens in a {context}-token window"
                    ),
                );
            }
            return Ok(());
        }
        if !drop_oldest_local_exchange(transcript, brief) {
            return Err(Error::Invalid(format!(
                "The script request and latest research need {tokens} tokens plus {output} for the draft, but the local model loaded a {context}-token context. Raise the context window or shorten the request."
            )));
        }
        dropped = true;
    }
}

/// The project's reference edits (finished videos made by a person) as a study guide for the model:
/// length, keyframe timeline and transcript. Empty when there are none.
pub fn reference_edits_text(db: &Db, project_id: i64, max_chars: usize) -> String {
    let Ok(refs) = db.excluded_videos(project_id) else { return String::new() };
    let refs: Vec<_> = refs.into_iter().filter(|(_, role, _)| role == "reference").collect();
    if refs.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\nREFERENCE EDITS\nFinished videos a person edited for this project. They are NOT footage: never put their \
         video ids in a script. Study them and match their quality: total length, how long shots are held (keyframe \
         times show where the picture changes), how interviews and scenery alternate, and how the story opens and ends.\n",
    );
    let per_ref = max_chars / refs.len();
    for (video_id, _, path) in refs {
        let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or(path);
        let duration = db
            .conn
            .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
            .ok()
            .flatten()
            .unwrap_or(0.0);
        let mut block = format!("\n\"{name}\" ({duration:.0} s)\nPicture:\n");
        if let Ok(mut st) = db.conn.prepare("SELECT t_s, description_json FROM frames WHERE video_id = ?1 ORDER BY t_s")
        {
            let rows = st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, Option<String>>(1)?)));
            for (t, d) in rows.into_iter().flatten().flatten() {
                let desc: String = d
                    .and_then(|j| serde_json::from_str::<Value>(&j).ok())
                    .and_then(|v| v.get("description").and_then(|x| x.as_str()).map(str::to_string))
                    .unwrap_or_default()
                    .chars()
                    .take(110)
                    .collect();
                block.push_str(&format!("  {t:.1}s {desc}\n"));
            }
        }
        block.push_str("Speech:\n");
        if let Ok(mut st) =
            db.conn.prepare("SELECT start_s, end_s, text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")
        {
            let rows =
                st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?)));
            for (a, b, t) in rows.into_iter().flatten().flatten() {
                block.push_str(&format!("  {a:.1}-{b:.1}s {}\n", t.trim()));
            }
        }
        if block.len() > per_ref {
            let mut cut = per_ref;
            while !block.is_char_boundary(cut) {
                cut -= 1;
            }
            block.truncate(cut);
            block.push_str("…\n");
        }
        out.push_str(&block);
    }
    out
}

/// After this many searches in a row come back empty, the model is told to look at the footage
/// directly instead: a weak local model otherwise repeats the same query until it runs out of turns.
const EMPTY_SEARCHES_BEFORE_HINT: usize = 2;

/// How often the loop insists a local model looks before it answers. Two: the first nudge names
/// the tools, the second shows the JSON. A third would be arguing with it.
const MAX_PUSHBACKS: usize = 2;

const EMPTY_SEARCH_HINT: &str = "Those searches found nothing — the words you are searching for are not in this \
footage. Stop searching: call list_videos, then get_video and get_transcript on the videos that look useful, and \
build the script from what they actually show.";

/// Check whether a video belongs to a project.
pub fn is_video_in_project(db: &Db, project_id: i64, video_id: i64) -> bool {
    db.is_video_in_project(project_id, video_id)
}

/// Tool results for the local helper (8k context) stay small.
pub const LOCAL_TOOL_RESULT_CHARS: usize = 1500;
/// Servers have large contexts (highllama: 120k); richer results make much better edits.
pub const SERVER_TOOL_RESULT_CHARS: usize = 8000;

/// Check grounding and project constraints for a script. Returns error issues for ungrounded clips.
pub fn check_grounding(
    db: &Db,
    project_id: i64,
    script: &Script,
    grounding: &Grounding,
    cfg: &crate::config::ScriptConfig,
) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            if !is_video_in_project(db, project_id, clip.video_id) {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Error,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!("video #{} does not belong to project", clip.video_id),
                });
            } else if !grounding.is_grounded(clip.video_id, clip.in_s, clip.out_s, cfg.grounding_slack_s) {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Error,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!(
                        "clip not grounded in tool results: video #{} [{:.1}s–{:.1}s]",
                        clip.video_id, clip.in_s, clip.out_s
                    ),
                });
            }
        }
    }
    issues
}

/// A video length stated in the user's message: "60 second", "90s", "1.5 minutes", "2 minutos".
pub fn requested_duration_s(message: &str) -> Option<f64> {
    let lower = message.to_lowercase();
    let tokens: Vec<&str> =
        lower.split(|c: char| c.is_whitespace() || c == '-' || c == ',').filter(|t| !t.is_empty()).collect();
    for (i, tok) in tokens.iter().enumerate() {
        // "90s" / "2min" glued forms, or a number followed by a unit word.
        let split = tok.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(tok.len());
        let (num, glued) = tok.split_at(split);
        let Ok(n) = num.parse::<f64>() else { continue };
        let unit = if glued.is_empty() { tokens.get(i + 1).copied().unwrap_or("") } else { glued };
        let unit = unit.trim_matches(|c: char| !c.is_alphabetic());
        let secs = if ["s", "sec", "secs", "second", "seconds", "seg", "segs", "segundo", "segundos"].contains(&unit) {
            n
        } else if ["m", "min", "mins", "minute", "minutes", "minuto", "minutos"].contains(&unit) {
            n * 60.0
        } else {
            continue;
        };
        if (5.0..=3600.0).contains(&secs) {
            return Some(secs);
        }
    }
    None
}

/// The project's videos a text names by file name ("use only 481_2635.mp4 and 481_2636.mp4").
pub fn named_videos(db: &Db, project_id: i64, text: &str) -> Vec<i64> {
    let lower = text.to_lowercase();
    let Ok(videos) = crate::index::videos(db, Some(project_id)) else { return Vec::new() };
    let mut ids: Vec<i64> = videos
        .iter()
        .filter(|v| {
            std::path::Path::new(&v.path)
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.len() > 4 && lower.contains(&n.to_lowercase()))
        })
        .map(|v| v.id)
        .collect();
    ids.dedup();
    ids
}

/// The shape a new-cut brief asks for, from its header: `[Draft "Title" — Target: 30s — Format: 9:16]`.
/// `None` when the message has no such header or names no format (the project's shape).
pub fn requested_format(message: &str) -> Option<crate::clips::ClipFormat> {
    let header = message.lines().next()?.trim();
    if !header.starts_with("[Draft") {
        return None;
    }
    let rest = header[header.find("Format:")? + "Format:".len()..].trim_start();
    let token: String = rest.chars().take_while(|c| !c.is_whitespace() && *c != ']').collect();
    crate::clips::ClipFormat::parse(&token).ok()
}

/// The shape a turn's cut must have, `(width, height, crop)` where `crop` means the footage is cut
/// to that shape and its framing follows the speaker:
/// - the format the new-cut brief chose (Original = the project's own size) wins over the model;
/// - else the cut keeps the shape it already has (a revision does not change it);
/// - else, a first draft on Auto, the shape the model drafted, from the brief ("an instagram
///   video" is vertical), with the framing a crop needs; the project's own when it set none.
fn turn_shape(
    project: &crate::projects::Project,
    requested: Option<crate::clips::ClipFormat>,
    baseline: Option<&Script>,
    drafted: &Script,
) -> Option<(i64, i64, bool)> {
    if project.width <= 0 || project.height <= 0 {
        return None;
    }
    let own = (project.width, project.height, false);
    let crops =
        |w: i64, h: i64| crate::framing::needs_crop(project.width as f64, project.height as f64, w as f64, h as f64);
    Some(match requested {
        Some(fmt) => fmt.size().map_or(own, |(w, h)| (w, h, true)),
        None => match baseline.and_then(|b| Some((b.width?, b.height?))) {
            Some((w, h)) => (w, h, crops(w, h)),
            None => match (drafted.width, drafted.height) {
                (Some(w), Some(h)) if w > 0 && h > 0 => (w, h, crops(w, h)),
                _ => own,
            },
        },
    })
}

/// Give a cut its shape (see [`turn_shape`]), then the framing repair a save would do. True when
/// anything changed.
fn apply_shape(db: &Db, s: &mut Script, (w, h, crop): (i64, i64, bool), cfg: &crate::config::ScriptConfig) -> bool {
    let framed = !crop || s.framing.is_some();
    if s.width == Some(w) && s.height == Some(h) && framed {
        return false;
    }
    s.width = Some(w);
    s.height = Some(h);
    if crop && s.framing.is_none() {
        s.framing = Some(crate::framing::Framing::default());
    }
    if !crop {
        // The footage's own shape: no crop to plan, and no follow-the-speaker framing left over.
        s.framing = None;
    }
    crate::script_fix::repair_framing(db, s, cfg.speech_overrun_s, None);
    true
}

/// What the script chat adds to the harness instructions ([`crate::tools::INSTRUCTIONS`], the same
/// ones an MCP client reads): only how a chat turn ends. Every model the chat drives, local or not,
/// gets these same words and the same tools; `{project}`, `{fps}`, `{width}` and `{height}` are filled in.
pub const CHAT_TURN_PROMPT: &str =
    "You are the editor on GhostReel project \"{project}\" ({fps} fps, {width}x{height}). \
The project is set, so `project` is optional in every tool. The instructions below are the ones every editor \
working with GhostReel follows, with the same tools.

IN THIS CHAT
- Save the edit with save_script, as below: the last version you save is what this turn delivers. (Answering \
with the script as one JSON object, the `script` save_script takes, saves it too.)
- Rendering (preview_script, export_script) and review_render come after the turn: do not render from here, and do \
not start another draft or a clip run.
- Answer {\"action\":\"reply\",\"text\":...} instead of a script when the request is ambiguous and one question \
would settle it, or when the footage cannot support it (say what is missing). Do not reply to avoid work.
- When revising, change what the user asked for and keep what they did not mention. A line marked [applied after \
drafting] in an earlier reply is a change already made to the saved script: build on it.
- Reply in the user's language, and look at any image they attach.

";

/// What a b-roll montage chat adds to the harness instructions: a theme told in pictures, nobody
/// speaking, no voice-over. `{audio_rule}` is filled with the chat's choice of natural sound or silence.
pub const BROLL_EDITOR_PROMPT: &str = "THIS CHAT IS A B-ROLL MONTAGE\nA piece told in pictures only, on one theme, with no interviews and no voice-over. Work in this order.
 a. Read the brief and decide the theme and the feeling: what the piece is about, and what a viewer should come away with.
 b. Find the pictures: search for the subjects of the theme, then get_video to see the keyframes, the shot sizes and the shaky stretches. A range you have not seen returned is a guess, and guesses are dropped.
 c. Lay out beats - each a short visual sequence on one idea of the theme - and order them so the piece goes somewhere.
 d. Add up the clip lengths and fix the total yourself before you answer.
 e. Answer with one JSON object and nothing else.
Before you answer, check: every range came from a tool; nobody is talking to the camera in any clip; no clip is under 3 s; no two neighbouring shots look alike; the total is within 10% of the target; every beat belongs to the theme.

HOW TO EDIT
1. One theme, held all the way: every shot earns its place by belonging to it. A string of nice shots with nothing in common is not a piece.
2. Structure: an establishing opening (a wide of the place or the subject), 3-6 beats that each develop one idea of the theme, and a closing image that lands - a strong wide, a detail that sums the piece up, a movement that settles.
3. Cohesion between shots: inside a beat move through shot sizes (wide, then medium, then a close detail); keep the same setting, light and time of day together; carry movement across a cut (motion in the same direction, a pan into a pan); when you change place, match on subject, shape or colour. Never put two near-identical shots side by side, and never reuse the same footage twice.
4. Nobody talking: this piece has no interviews. Do not use a range where someone speaks to the camera or into a microphone - a moving mouth with no voice looks like a mistake. People doing things (walking, working, playing, laughing, looking around) are good b-roll. get_transcript shows where people speak on a tape: cut around those stretches.
5. Choose only strong shots: a clear subject (people at something, a landmark, a building, a sign, activity, a striking view). Skip footage whose keyframes describe black or blank frames, blur, transitions, the ground or sky only, or the same view as the previous clip.
6. Pacing: hold shots 3-8 s - longer for wides and slow movement, shorter for details. Let the rhythm build and settle; do not cut at the same length every time, and never flash a shot past.
7. Camera work: every search hit and video says whether the camera is static, tripod, stabilised or handheld, and get_video lists shaky stretches as shaky_at timestamps. A shaky shot looks wrong in a finished cut: cut around those stretches, and when two clips cover the same moment prefer the mounted or stabilised one.
8. Audio: {audio_rule}
9. Narration: none. Set narration to \"\" on every beat and do not give any beat a \"bed\". on_screen_text may carry a short title or a place name when it helps the piece.
10. Length: the clips add up to the length the user asked for; set target_duration_s to it, and give it about 10% more - a cut that comes in long is trimmed to fit, one that comes in short cannot be fixed. If the user gave no length, choose what the material supports (usually 30-90 s).
";

/// The system prompt of a chat turn, the same for every model the chat drives: the project and how a
/// chat turn ends ([`CHAT_TURN_PROMPT`]), the harness instructions every editor gets
/// ([`crate::tools::INSTRUCTIONS`]), a b-roll montage's brief when the chat is one, the house rules
/// from Settings (`custom`), the tools, and the current draft. Only the rendering of the tools
/// (`detail`) follows the model's window.
pub fn build_system_prompt(
    project: &Project,
    latest_script_json: Option<&str>,
    custom: Option<&str>,
    detail: tools::Detail,
    style: ChatStyle,
) -> String {
    let fps = if project.fps_den == 1 {
        project.fps_num.to_string()
    } else {
        format!("{:.3}", project.fps_num as f64 / project.fps_den as f64)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    };
    let fill = |s: &str| {
        s.replace("{project}", &project.name)
            .replace("{fps}", &fps)
            .replace("{width}", &project.width.to_string())
            .replace("{height}", &project.height.to_string())
    };
    let mut prompt = fill(CHAT_TURN_PROMPT);
    prompt.push_str(crate::tools::INSTRUCTIONS);
    prompt.push('\n');
    if style.broll {
        prompt.push('\n');
        prompt.push_str(&BROLL_EDITOR_PROMPT.replace(
            "{audio_rule}",
            if style.natural_sound {
                "set every clip's audio to \"source\": its own ambient sound (street, wind, water, a crowd) carries the piece. That is one more reason to avoid anyone speaking."
            } else {
                "set every clip's audio to \"mute\": the piece is pictures only, and music is laid over it later."
            },
        ));
    }
    if let Some(house) = custom.map(str::trim).filter(|c| !c.is_empty()) {
        prompt.push_str("\nHOUSE RULES (from Settings; they win over the defaults above)\n");
        prompt.push_str(&fill(house));
        prompt.push('\n');
    }
    prompt.push_str(&tools::tools_prose(RenderOpts::new(detail, true)));

    if let Some(draft) = latest_script_json {
        prompt.push_str(&format!(
            "\nCurrent script draft from this session:\n{}\n\
             When the user asks for changes, revise this draft accordingly. \
             Note: This draft may contain deliberate manual edits from the user — treat them as intentional and preserve them unless the user explicitly requests changes to them.\n",
            draft
        ));
    }

    if let Some(section) = project.glossary.prompt_section() {
        prompt.push('\n');
        prompt.push_str(&section);
        prompt.push('\n');
    }

    prompt
}

/// Writes the prompt and raw answer of one local call to `$GHOSTREEL_DEBUG_CHAT/<stage>.*.txt`.
/// Off unless the variable is set: what the model actually emitted is otherwise unknowable from
/// the outside, and "the narration is empty" has very different causes before and after parsing.
fn debug_dump(stage: &str, prompt: &str, answer: &str) {
    let Some(dir) = std::env::var_os("GHOSTREEL_DEBUG_CHAT") else { return };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let _ = std::fs::write(dir.join(format!("{stage}.prompt.txt")), prompt);
    let _ = std::fs::write(dir.join(format!("{stage}.answer.txt")), answer);
}

/// What the keyframes in a clip's range show, as a few short lines.
fn frame_summaries(db: &Db, video_id: i64, in_s: f64, out_s: f64, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(mut st) = db
        .conn
        .prepare("SELECT description_json FROM frames WHERE video_id = ?1 AND t_s >= ?2 AND t_s <= ?3 ORDER BY t_s")
    else {
        return out;
    };
    let Ok(rows) = st.query_map(params![video_id, in_s - 0.5, out_s + 0.5], |r| r.get::<_, Option<String>>(0)) else {
        return out;
    };
    for row in rows.flatten().flatten() {
        let summary = serde_json::from_str::<Value>(&row)
            .ok()
            .and_then(|v| v["description"].as_str().map(str::to_string))
            .unwrap_or(row);
        if !summary.trim().is_empty() {
            out.push(summary.chars().take(240).collect::<String>());
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

/// The clips the model is actually allowed to cut, as the draft prompt states them.
///
/// Grounding is enforced after the fact — anything outside it is dropped, and a draft built
/// entirely from unseen video ids collapses to nothing. `list_videos` shows every id in the
/// project, so a small model readily drafts with footage it never opened; spelling the permitted
/// ranges out costs a few lines and removes the guesswork.
fn allowed_clips_text(grounding: &Grounding) -> String {
    if grounding.ranges.is_empty() {
        return String::new();
    }
    let mut merged: Vec<(i64, f64, f64)> = Vec::new();
    for &(vid, s, e) in &grounding.ranges {
        match merged.iter_mut().find(|(v, _, _)| *v == vid) {
            Some(r) => {
                r.1 = r.1.min(s);
                r.2 = r.2.max(e);
            }
            None => merged.push((vid, s, e)),
        }
    }
    merged.sort_by_key(|(v, _, _)| *v);
    let list: Vec<String> = merged.iter().map(|(v, s, e)| format!("  video #{v}: {s:.1}-{e:.1} s")).collect();
    format!(
        "You may only use these videos and time ranges — every clip must lie inside one of them, \
         and any clip outside them is thrown away:\n{}\n",
        list.join("\n")
    )
}

/// The beats that need a voice-over written, as (beat index, prompt) pairs.
///
/// Kept separate from the asking so that no database handle is alive across an await — the
/// connection is not `Send`, and `run_turn`'s future has to be.
fn narration_jobs(db: &Db, script: &Script, cfg: &crate::config::ScriptConfig) -> Vec<(usize, String)> {
    let mut jobs = Vec::new();
    for (idx, beat) in script.beats.iter().enumerate() {
        let beat_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        let empty = beat.narration.as_deref().map(str::trim).is_none_or(str::is_empty);
        let has_speech = beat.clips.iter().any(|c| clip_has_speech(db, c.video_id, c.in_s, c.out_s));
        if !empty || has_speech || beat_s < 2.0 {
            continue;
        }
        let words = (beat_s * cfg.narration_words_per_s).round().max(5.0) as usize;
        let seen: Vec<String> =
            beat.clips.iter().flat_map(|c| frame_summaries(db, c.video_id, c.in_s, c.out_s, 3)).collect();
        let purpose = if beat.purpose.trim().is_empty() { "introduce what is on screen" } else { beat.purpose.trim() };
        let seen = if seen.is_empty() { "(no description available)".to_string() } else { seen.join("\n") };
        jobs.push((
            idx,
            format!(
                "<|im_start|>system\nYou write documentary voice-over. Write natural, specific sentences about what \
                 is on screen and why it matters. No filler, no lists, no stage directions.<|im_end|>\n\
                 <|im_start|>user\nThis shot runs {beat_s:.0} seconds. Its purpose: {purpose}\n\
                 On screen:\n{seen}\n\n\
                 Write about {words} words of narration for it.<|im_end|>\n"
            ),
        ));
    }
    jobs
}

/// Ask for each silent beat's narration on its own.
///
/// A small local model reliably writes good voice-over when that is the only thing it is asked
/// for, and reliably leaves `narration` empty when it is one field among a whole nested script —
/// so the beats that come back silent get a second, much smaller question instead of a redraft.
async fn fill_missing_narration(
    helper: &mut crate::vision::LocalLlm,
    script: &mut Script,
    jobs: Vec<(usize, String)>,
) -> usize {
    let schema = json!({
        "type": "object",
        "properties": { "narration": { "type": "string" } },
        "required": ["narration"],
        "additionalProperties": false
    });
    let mut filled = 0usize;
    for (idx, prompt) in jobs {
        if let Ok(answer) = helper.complete_limited(&prompt, Some(schema.clone()), 512).await
            && let Ok(v) = serde_json::from_str::<Value>(&answer)
            && let Some(n) = v["narration"].as_str().map(str::trim).filter(|n| !n.is_empty())
            && let Some(beat) = script.beats.get_mut(idx)
        {
            beat.narration = Some(n.to_string());
            filled += 1;
        }
    }
    filled
}

/// Editorial problems the model can fix in a redraft: narration too short for its beat, and clips
/// over footage the tools know nothing about (no speech and no described keyframe nearby).
pub fn content_issues(db: &Db, script: &Script, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    if cfg.style.broll {
        return broll_issues(db, script);
    }
    let mut issues = empty_speech_issues(db, script);
    let mut seen_narration: Vec<String> = Vec::new();
    for beat in &script.beats {
        let Some(n) = beat.narration.as_deref().map(str::trim).filter(|n| !n.is_empty()) else { continue };
        let key = n.to_lowercase();
        if seen_narration.contains(&key) {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: None,
                message:
                    "narration repeats an earlier beat word for word; write new narration for what this beat shows"
                        .into(),
            });
        } else {
            seen_narration.push(key);
        }
    }
    for beat in &script.beats {
        let beat_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        let words = beat.narration.as_deref().map(|n| n.split_whitespace().count()).unwrap_or(0);
        let has_speech = beat.clips.iter().any(|c| clip_has_speech(db, c.video_id, c.in_s, c.out_s));
        if words > 0 || !has_speech {
            let spoken_s = words as f64 / cfg.narration_words_per_s;
            if beat_s > 0.0 && spoken_s < beat_s * cfg.min_narration_coverage {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "narration is {words} words (~{spoken_s:.0} s spoken) but the beat runs {beat_s:.0} s; \
                         write about {:.0} words or shorten the beat",
                        beat_s * cfg.narration_words_per_s
                    ),
                });
            }
        }
        for (i, c) in beat.clips.iter().enumerate() {
            if !clip_has_speech(db, c.video_id, c.in_s, c.out_s)
                && !clip_has_described_frame(db, c.video_id, c.in_s, c.out_s)
            {
                issues.push(Issue {
                    fixes: Vec::new(),
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "video #{} {:.1}–{:.1} s has no speech and no described keyframe; pick a range you have seen described",
                        c.video_id, c.in_s, c.out_s
                    ),
                });
            }
        }
    }
    issues
}

/// What a montage can get wrong that the model can fix in a redraft: a shot of somebody talking,
/// and a range nobody has described.
fn broll_issues(db: &Db, script: &Script) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let message = if clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                format!(
                    "video #{} {:.1}–{:.1} s has someone speaking on the microphone; this cut is b-roll only — \
                     pick a stretch where nobody talks",
                    c.video_id, c.in_s, c.out_s
                )
            } else if !clip_has_described_frame(db, c.video_id, c.in_s, c.out_s) {
                format!(
                    "video #{} {:.1}–{:.1} s has no described keyframe; pick a range you have seen described",
                    c.video_id, c.in_s, c.out_s
                )
            } else {
                continue;
            };
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(i),
                message,
            });
        }
    }
    issues
}

/// Clips whose audible speech says nothing: "Okay.", "Thank you.", "Yeah."
///
/// Reported rather than dropped. The words may be the point — somebody thanking a neighbour is a
/// moment — and the model is the only thing that knows; what it must not do is spend five seconds
/// on one by accident.
fn empty_speech_issues(db: &Db, script: &Script) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        // A bed carries the beat's sound, so its clips are not meant to be heard.
        if beat.bed.is_some() {
            continue;
        }
        for (i, c) in beat.clips.iter().enumerate() {
            if c.audio != crate::script::Audio::Source {
                continue;
            }
            let said = clip_speech(db, c.video_id, c.in_s, c.out_s);
            let said = said.trim();
            if said.is_empty() || said.split_whitespace().count() >= MIN_CLIP_WORDS {
                continue;
            }
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(i),
                message: format!(
                    "clip plays {:.1} s of \"{said}\" (video #{} {:.1}–{:.1}) — that is an acknowledgement, not \
                     something said; quote a whole thought or mute the clip and let another voice carry it",
                    (c.out_s - c.in_s).max(0.0),
                    c.video_id,
                    c.in_s,
                    c.out_s
                ),
            });
        }
    }
    issues
}

/// What is actually at a clip's range, when the model never opened it: a short description of the
/// nearest keyframe or the words spoken there. `None` when the range falls outside the video or
/// nothing was indexed for it, which is the only case worth throwing the clip away for.
fn verify_clip(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> Option<String> {
    let duration: Option<f64> =
        db.conn.query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get(0)).ok().flatten();
    if in_s < 0.0 || out_s <= in_s || duration.is_some_and(|d| out_s > d + 1.0) {
        return None;
    }
    let said: Option<String> = db
        .conn
        .query_row(
            "SELECT text FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 \
             ORDER BY start_s LIMIT 1",
            params![video_id, in_s, out_s],
            |r| r.get(0),
        )
        .ok();
    if let Some(t) = said.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        return Some(format!("says \"{}\"", t.chars().take(60).collect::<String>()));
    }
    let seen: Option<String> = db
        .conn
        .query_row(
            "SELECT description_json FROM frames WHERE video_id = ?1 AND t_s >= ?2 - 2.0 AND t_s <= ?3 + 2.0 \
             AND description_json IS NOT NULL ORDER BY t_s LIMIT 1",
            params![video_id, in_s, out_s],
            |r| r.get(0),
        )
        .ok();
    seen.and_then(|j| serde_json::from_str::<Value>(&j).ok())
        .and_then(|v| v["description"].as_str().map(|d| d.chars().take(60).collect::<String>()))
        .map(|d| format!("shows {d}"))
}

/// A described keyframe inside the clip, or shortly before it (the view it continues from).
fn clip_has_described_frame(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL
               AND t_s >= ?2 - 10.0 AND t_s <= ?3)",
            params![video_id, in_s, out_s],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

/// Create a new chat session in the database.
pub fn create_session(db: &Db, project_id: i64, title: &str) -> Result<i64, Error> {
    let t = now();
    db.conn.execute(
        "INSERT INTO chat_sessions(project_id, title, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![project_id, title, t, t],
    )?;
    Ok(db.conn.last_insert_rowid())
}

fn parse_style(json: Option<&str>) -> ChatStyle {
    json.and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default()
}

/// How this chat builds its cuts; the default for a chat that never chose.
pub fn session_style(db: &Db, session_id: i64) -> ChatStyle {
    let json: Option<String> = db
        .conn
        .query_row("SELECT style_json FROM chat_sessions WHERE id = ?1", [session_id], |r| r.get(0))
        .ok()
        .flatten();
    parse_style(json.as_deref())
}

/// Record how this chat builds its cuts. Every later turn in it follows the same style.
pub fn set_session_style(db: &Db, session_id: i64, style: ChatStyle) -> Result<(), Error> {
    let json = serde_json::to_string(&style).map_err(|e| Error::Invalid(e.to_string()))?;
    db.conn.execute("UPDATE chat_sessions SET style_json = ?1 WHERE id = ?2", params![json, session_id])?;
    Ok(())
}

/// Delete a chat session and its messages. Scripts drafted in it are kept and merely lose the
/// link (the schema sets their `session_id` to NULL): a script is a deliverable, a chat is not.
pub fn delete_session(db: &Db, session_id: i64) -> Result<bool, Error> {
    let n = db.conn.execute("DELETE FROM chat_sessions WHERE id = ?1", [session_id])?;
    if n > 0 {
        crate::llmlog::delete(session_id);
    }
    Ok(n > 0)
}

pub fn sessions(db: &Db, project_id: i64) -> Result<Vec<ChatSession>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, project_id, title, created_at, updated_at, style_json
         FROM chat_sessions WHERE project_id = ?1 ORDER BY updated_at DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok(ChatSession {
            id: r.get(0)?,
            project_id: r.get(1)?,
            title: r.get(2)?,
            created_at: r.get(3)?,
            updated_at: r.get(4)?,
            style: parse_style(r.get::<_, Option<String>>(5)?.as_deref()),
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Summary of a chat session including message count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionSummary {
    pub id: i64,
    pub project_id: i64,
    pub title: String,
    pub message_count: usize,
    pub updated_at: i64,
}

/// List all chat sessions for a project with their message count.
pub fn sessions_with_counts(db: &Db, project_id: i64) -> Result<Vec<SessionSummary>, Error> {
    let mut st = db.conn.prepare(
        "SELECT cs.id, cs.project_id, cs.title, cs.updated_at, COUNT(cm.id)
         FROM chat_sessions cs
         LEFT JOIN chat_messages cm ON cm.session_id = cs.id
         WHERE cs.project_id = ?1
         GROUP BY cs.id
         ORDER BY cs.updated_at DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok(SessionSummary {
            id: r.get(0)?,
            project_id: r.get(1)?,
            title: r.get(2)?,
            updated_at: r.get(3)?,
            message_count: r.get::<_, i64>(4)? as usize,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// Retrieve all messages for a session.
pub fn messages(db: &Db, session_id: i64) -> Result<Vec<ChatMessage>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, session_id, role, content, tool_calls_json, created_at, images_json
         FROM chat_messages WHERE session_id = ?1 ORDER BY id ASC",
    )?;
    let rows = st.query_map([session_id], |r| {
        let id: i64 = r.get(0)?;
        let sid: i64 = r.get(1)?;
        let role: String = r.get(2)?;
        let content: String = r.get(3)?;
        let tool_calls_json: Option<String> = r.get(4)?;
        let created_at: i64 = r.get(5)?;
        let images_json: Option<String> = r.get(6)?;

        let mut tool_calls = None;
        let mut script_id = None;
        let mut performance = None;

        if let Some(json_str) = tool_calls_json
            && let Ok(val) = serde_json::from_str::<Value>(&json_str)
        {
            if let Some(arr) = val.as_array() {
                tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(Value::Array(arr.clone())).ok();
            } else if let Some(obj) = val.as_object() {
                script_id = obj.get("script_id").and_then(|v| v.as_i64());
                performance = obj.get("performance").and_then(|v| serde_json::from_value(v.clone()).ok());
                if let Some(tc) = obj.get("tool_calls") {
                    tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(tc.clone()).ok();
                }
            }
        }

        let images = images_json
            .as_deref()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .filter(|imgs| !imgs.is_empty());

        Ok(ChatMessage { id, session_id: sid, role, content, images, tool_calls, script_id, created_at, performance })
    })?;

    Ok(rows.filter_map(Result::ok).collect())
}

/// Run one turn of the script chat agent.
/// The first thing the editor asked for in this session, which is what every later cut is still
/// meant to serve.
fn session_brief(db: &Db, session_id: i64) -> Option<String> {
    db.conn
        .query_row(
            "SELECT content FROM chat_messages WHERE session_id = ?1 AND role = 'user' ORDER BY id LIMIT 1",
            params![session_id],
            |r| r.get::<_, String>(0),
        )
        .ok()
        .filter(|b| !b.trim().is_empty())
}

/// Save any data URLs (from pasted images) into `<data_dir>/chat_images`, and keep any valid
/// existing file paths. Returns the absolute paths on disk.
fn persist_chat_images(data_dir: &Path, session_id: i64, images: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if images.is_empty() {
        return out;
    }
    let chat_images_dir = data_dir.join("chat_images");
    let _ = std::fs::create_dir_all(&chat_images_dir);
    let now_ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);

    for (idx, img_str) in images.iter().enumerate() {
        let trimmed = img_str.trim();
        if trimmed.starts_with("data:image/")
            && let Some((header, b64)) = trimmed.split_once(',')
        {
            let ext = if header.contains("jpeg") || header.contains("jpg") {
                "jpg"
            } else if header.contains("webp") {
                "webp"
            } else {
                "png"
            };
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64.trim()) {
                let filename = format!("chat_{}_{}_{}.{}", session_id, now_ts, idx, ext);
                let target = chat_images_dir.join(filename);
                if std::fs::write(&target, bytes).is_ok() {
                    out.push(target);
                    continue;
                }
            }
        }
        let p = PathBuf::from(trimmed);
        if p.is_file() {
            out.push(p);
        }
    }
    out
}

pub async fn run_turn(
    ctx: &mut ChatContext,
    project_id: i64,
    session_id: Option<i64>,
    message: &str,
    images: &[String],
    baseline_script_id: Option<i64>,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
) -> Result<TurnResult, Error> {
    let turn_started = std::time::Instant::now();
    let (session_id, is_new_session) = match session_id {
        Some(id) => (id, false),
        None => {
            let title: String = message.chars().take(60).collect();
            let sid = create_session(&ctx.db, project_id, &title)?;
            (sid, true)
        }
    };

    let _log_session = crate::llmlog::session(session_id);
    if let ChatBackend::Server { api_key, .. } = &ctx.backend {
        crate::llmlog::redact_secret(api_key);
    }
    crate::llmlog::redact_secret(&ctx.jev.api_key);
    let (backend, model, tool, think) = match &ctx.backend {
        ChatBackend::Server { model, .. } => ("server", model.as_str(), None, false),
        ChatBackend::Local { helper, think, .. } => ("local", helper.model_name(), None, *think),
        ChatBackend::Cli(agent) => ("cli", agent.cfg.model.as_str(), Some(agent.cfg.tool.as_str()), false),
    };
    let logged_baseline = baseline_script_id.or_else(|| {
        ctx.db
            .conn
            .query_row(
                "SELECT id FROM scripts WHERE session_id = ?1 AND trash_id IS NULL ORDER BY id DESC LIMIT 1",
                [session_id],
                |r| r.get::<_, i64>(0),
            )
            .ok()
    });
    crate::llmlog::push(
        "info",
        json!({
            "event": "turn_start", "started_at_ms": crate::llmlog::timestamp(),
            "session_id": session_id, "project_id": project_id, "backend": backend,
            "model": model, "tool": tool, "max_tool_rounds": ctx.max_tool_rounds,
            "think": think, "ctx_tokens": ctx.backend.window_tokens(),
            "mode": if ctx.script.style.broll { "broll" } else { "documentary" },
            "method": "run_turn", "style": ctx.script.style, "settings": ctx.script,
            "message": message, "baseline_script_id": logged_baseline,
        })
        .to_string(),
    );
    let mut rounds_used = 0u32;
    let result = run_turn_inner(
        ctx,
        project_id,
        session_id,
        is_new_session,
        message,
        images,
        baseline_script_id,
        on_event,
        &mut rounds_used,
    )
    .await;
    match &result {
        Ok(turn) => {
            let version = turn.script_id.and_then(|id| crate::script::load(&ctx.db, id).ok()).map(|s| s.version);
            crate::llmlog::push(
                "info",
                json!({ "event": "turn_end", "duration_ms": turn_started.elapsed().as_millis(),
                "rounds_used": rounds_used, "saved_script_id": turn.script_id, "version": version,
                "issues_count": turn.issues.len() })
                .to_string(),
            );
        }
        Err(error) => {
            crate::llmlog::push("error", error.to_string());
            crate::llmlog::push("info", json!({ "event": "turn_end", "duration_ms": turn_started.elapsed().as_millis(),
                "rounds_used": rounds_used, "status": "error", "saved_script_id": null, "version": null, "issues_count": 0 }).to_string());
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_inner(
    ctx: &mut ChatContext,
    project_id: i64,
    session_id: i64,
    is_new_session: bool,
    message: &str,
    images: &[String],
    baseline_script_id: Option<i64>,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
    rounds_used: &mut u32,
) -> Result<TurnResult, Error> {
    let turn_started = std::time::Instant::now();
    let mut inference = Vec::new();
    if let ChatBackend::Local { helper, .. } = &mut ctx.backend {
        helper.take_inference();
    }
    let project = ctx.db.project(project_id)?;
    // Zero means unlimited; cap pathological non-terminating tool loops at 500 rounds.
    // Other stop conditions (cancellation, memoization, and context limits) still apply.
    let cancel_flag = ctx.cancel.clone();
    let cancelled = move || cancel_flag.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst));
    let round_cap = round_cap(ctx.max_tool_rounds);

    let saved_image_paths = persist_chat_images(&ctx.paths.data_dir, session_id, images);
    let image_strings: Vec<String> = saved_image_paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
    let images_json = if !image_strings.is_empty() { serde_json::to_string(&image_strings).ok() } else { None };

    let mut grounding = Grounding::default();
    // Every tool call goes through the registry by way of this session: what the turn has already
    // fetched is not fetched twice, a model repeating itself is told to answer, results are cut to
    // the room the context has, and errors come back as something the model can read.
    let mut session = ToolSession::new(match &ctx.backend {
        ChatBackend::Local { .. } => ctx.script.local_tool_result_chars,
        _ => ctx.script.roomy_tool_result_chars,
    });
    let mut prior_messages = Vec::new();
    let mut latest_script_json = None;
    let mut baseline_script: Option<Script> = None;

    // Explicit baseline script_id has top priority (handles multi-cut, sessionless cuts, or revision of specific versions)
    if let Some(sid) = baseline_script_id
        && let Ok(stored) = crate::script::load(&ctx.db, sid)
    {
        baseline_script = Some(stored.script);
    }

    if !is_new_session {
        prior_messages = messages(&ctx.db, session_id)?;
        for m in &prior_messages {
            if let Some(calls) = &m.tool_calls {
                for c in calls {
                    grounding.record_tool_call(&c.tool, &c.args, &ctx.db);
                }
            }
        }
        // Fall back to latest script from this session if no explicit baseline was given
        if baseline_script.is_none() {
            let stored_json: Option<String> = ctx
                .db
                .conn
                .query_row(
                    "SELECT script_json FROM scripts WHERE session_id = ?1 AND trash_id IS NULL ORDER BY id DESC LIMIT 1",
                    [session_id],
                    |r| r.get(0),
                )
                .ok();
            if let Some(ref json_str) = stored_json
                && let Ok(prev_script) = serde_json::from_str::<Script>(json_str)
            {
                baseline_script = Some(prev_script);
            }
        }
    }

    if let Some(ref prev_script) = baseline_script {
        for beat in &prev_script.beats {
            for clip in &beat.clips {
                grounding.add(clip.video_id, clip.in_s, clip.out_s);
            }
        }
        latest_script_json = serde_json::to_string_pretty(prev_script).ok();
    }

    // Write down what was asked before doing any of it. Everything used to be stored when the turn
    // finished, so a chat looked empty if you left the panel and came back while it ran — the
    // message lived only in the window's own state — and closing the app mid-turn lost it
    // altogether. After `prior_messages` is read, so it is not replayed twice.
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at, images_json)
         VALUES (?1, 'user', ?2, NULL, ?3, ?4)",
        params![session_id, message, now(), images_json],
    )?;
    ctx.db.conn.execute("UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2", params![now(), session_id])?;

    // A roomy brain gets the whole of every tool description; a small local model, whose window
    // is already half transcripts, gets the summary, the parameters and the costliest warning.
    let detail = match ctx.script.tool_docs.as_str() {
        "full" => tools::Detail::Full,
        "short" => tools::Detail::Short,
        // The full descriptions of the whole registry are about 10k tokens: a window under 64k
        // keeps the short ones.
        _ => match &ctx.backend {
            ChatBackend::Local { ctx_tokens, .. } if *ctx_tokens < 65_536 => tools::Detail::Short,
            _ => tools::Detail::Full,
        },
    };
    // A server is sent the descriptions as the `tools` array; its prompt only needs the short
    // prose, unless the setting insists on the full one.
    let prose_detail = match (&ctx.backend, ctx.script.tool_docs.as_str()) {
        (ChatBackend::Server { .. }, "full") => tools::Detail::Full,
        (ChatBackend::Server { .. }, _) => tools::Detail::Short,
        _ => detail,
    };
    // Every repair pass reads the chat's style from here.
    ctx.script.style = session_style(&ctx.db, session_id);
    // The footage the brief named by file, if it did: the bounds of what a repair may add.
    let brief_text = session_brief(&ctx.db, session_id).unwrap_or_else(|| message.to_string());
    ctx.script.footage = named_videos(&ctx.db, project_id, &brief_text);
    let style = ctx.script.style;
    // The registry's tools read the whole config; the chat's own `[script]` settings win in it.
    let tool_cfg = {
        let mut c = crate::config::Config::load(&ctx.paths.config_file).unwrap_or_default();
        c.script = ctx.script.clone();
        c
    };
    let mut sys_prompt =
        build_system_prompt(&project, latest_script_json.as_deref(), ctx.system_prompt.as_deref(), prose_detail, style);
    let reference_chars = match &ctx.backend {
        ChatBackend::Server { .. } => 6000,
        // Roughly an eighth of the window (~4 chars per token), so results leave room for the draft.
        ChatBackend::Local { ctx_tokens, .. } => ((*ctx_tokens as usize) * 4 / 8).clamp(1500, 12000),
        // CLI: treat like a large-context backend; tool results are limited by LOCAL_TOOL_RESULT_CHARS anyway.
        ChatBackend::Cli(_) => 6000,
    };
    sys_prompt.push_str(&reference_edits_text(&ctx.db, project_id, reference_chars));
    // Nothing else goes into the prompt: no transcript digest, no picture digest. Every model starts
    // where an editor over MCP starts, with get_outline, and reads only what its story needs.
    // A length the user states ("60 second promo", "2 minutos") wins over whatever the model sets.
    let requested_s = requested_duration_s(message);
    let requested_fmt = requested_format(message);

    // Only a first draft (or an explicit length) is squeezed to its target; revisions follow feedback.
    let enforce_target = latest_script_json.is_none() || requested_s.is_some();

    let mut tool_records = Vec::new();
    let mut raw_reply = String::new();
    let mut pre_issues: Vec<Issue> = Vec::new();
    // What the editor delivered, reviewed the same way whichever backend it is (`repair::deliver`).
    let mut delivered = repair::Delivered::default();
    // What the turn asked for, which every draft it reviews is held to. A macro, not a value: the
    // research loops borrow the grounding mutably until the editor delivers.
    macro_rules! rules {
        () => {
            repair::Rules {
                project: &project,
                grounding: &grounding,
                cfg: &ctx.script,
                enforce_target,
                requested_s,
                requested_fmt,
                baseline: baseline_script.as_ref(),
                read: repair::read_back,
            }
        };
    }

    match &mut ctx.backend {
        ChatBackend::Server { url, model, api_key, ctx_tokens } => {
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                // A local model at 40 tokens a second needs minutes for a long script, and reads the
                // whole prompt before it starts. Waiting is cheaper than losing the turn.
                .timeout(Duration::from_secs(ctx.script.server_timeout_s))
                .build()
                .map_err(|e| Error::Invalid(format!("reqwest client: {e}")))?;

            let mut req_messages = Vec::new();
            req_messages.push(json!({ "role": "system", "content": sys_prompt }));
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    if pm.role == "user" && pm.images.as_ref().is_some_and(|imgs| !imgs.is_empty()) {
                        let mut parts = vec![json!({ "type": "text", "text": &pm.content })];
                        for img_str in pm.images.as_ref().unwrap() {
                            let p = PathBuf::from(img_str);
                            if let Ok(bytes) = std::fs::read(&p) {
                                let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("jpeg");
                                let mime = if ext == "png" {
                                    "image/png"
                                } else if ext == "webp" {
                                    "image/webp"
                                } else {
                                    "image/jpeg"
                                };
                                let data_url = format!(
                                    "data:{mime};base64,{}",
                                    base64::engine::general_purpose::STANDARD.encode(bytes)
                                );
                                parts.push(json!({ "type": "image_url", "image_url": { "url": data_url } }));
                            }
                        }
                        req_messages.push(json!({ "role": "user", "content": parts }));
                    } else {
                        req_messages.push(json!({ "role": &pm.role, "content": &pm.content }));
                    }
                }
            }
            if !saved_image_paths.is_empty() {
                let mut parts = vec![json!({ "type": "text", "text": message })];
                for p in &saved_image_paths {
                    if let Ok(bytes) = std::fs::read(p) {
                        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("jpeg");
                        let mime = if ext == "png" {
                            "image/png"
                        } else if ext == "webp" {
                            "image/webp"
                        } else {
                            "image/jpeg"
                        };
                        let data_url =
                            format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
                        parts.push(json!({ "type": "image_url", "image_url": { "url": data_url } }));
                    }
                }
                req_messages.push(json!({ "role": "user", "content": parts }));
            } else {
                req_messages.push(json!({ "role": "user", "content": message }));
            }

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            let mut stuck = false;
            let mut last_assistant_text = String::new();

            // Three quarters of the window for the conversation; the draft needs the rest.
            let round_budget = (*ctx_tokens as usize).saturating_mul(3) / 4;
            while round_cap.is_none_or(|cap| tool_rounds < cap) {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                if !make_room(&mut req_messages, round_budget) {
                    // Nothing left to forget: stop looking and write the script.
                    break;
                }
                tool_rounds += 1;
                *rounds_used += 1;
                let mut body = json!({
                    "messages": req_messages,
                    // A tool call is short, but a model that reasons first spends the same budget
                    // on the thought, so it gets the same ceiling.
                    "max_tokens": ctx.script.max_answer_tokens,
                    "tools": tools::render_openai(RenderOpts::new(detail, true)),
                    "tool_choice": "auto",
                    "temperature": 0.3,
                    "chat_template_kwargs": { "enable_thinking": false },
                });
                if !model.is_empty() {
                    body["model"] = json!(model);
                }
                // Ollama's name for the context size: a GhostPen service loads the model with this
                // window (llama-server ignores it; it was started with its own).
                body["options"] = json!({ "num_ctx": *ctx_tokens });

                crate::llmlog::push("prompt", body.to_string());
                let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&body);
                if !api_key.is_empty() {
                    req = req.bearer_auth(api_key.as_str());
                }

                let request_started = std::time::Instant::now();
                let resp = req.send().await.map_err(|e| Error::Agent(crate::agents::error_from_request(url, &e)))?;
                if !resp.status().is_success() {
                    return Err(Error::Agent(crate::agents::error_from_response(url, model, resp).await));
                }

                let v: Value = resp.json().await.map_err(|e| Error::Invalid(format!("bad response JSON: {e}")))?;
                let logged_message = &v["choices"][0]["message"];
                if let Some(thinking) = logged_message["reasoning_content"].as_str() {
                    crate::llmlog::push("thinking", thinking);
                }
                crate::llmlog::push("answer", logged_message.to_string());
                inference.push(crate::inference::InferenceCall::from_response(
                    &v,
                    "server",
                    if model.is_empty() { "Server model not reported" } else { model },
                    request_started.elapsed().as_secs_f64(),
                ));
                let choice = v.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message"));
                let msg_obj = match choice {
                    Some(m) => m,
                    None => break,
                };

                if let Some(txt) = msg_obj.get("content").and_then(|c| c.as_str())
                    && !txt.trim().is_empty()
                {
                    last_assistant_text = txt.to_string();
                }

                let tool_calls = msg_obj.get("tool_calls").and_then(|t| t.as_array());
                if tool_calls.is_none() || tool_calls.unwrap().is_empty() {
                    // Assistant finished tool calling
                    break;
                }

                req_messages.push(msg_obj.clone());

                for tc in tool_calls.unwrap() {
                    let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let func = tc.get("function").cloned().unwrap_or(Value::Null);
                    let tool_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                    let tool_args: Value =
                        func.get("arguments")
                            .and_then(|a| {
                                if let Some(s) = a.as_str() { serde_json::from_str(s).ok() } else { Some(a.clone()) }
                            })
                            .unwrap_or(json!({}));

                    let tool_name = crate::tools::canonical_name(&tool_name).to_string();
                    on_event(ChatEvent::ToolStarted { tool: tool_name.clone(), args: tool_args.clone() });

                    let called = {
                        let mut tctx =
                            ToolCtx::new(&ctx.paths, &tool_cfg, &mut ctx.db, &mut ctx.embedder, Origin::ScriptChat)
                                .with_project(project.clone())
                                .with_cancel(ctx.cancel.clone())
                                .with_grounding(&mut grounding);
                        session.call(&mut tctx, &tool_name, &tool_args).await
                    };
                    if called.cancelled {
                        return Err(Error::Invalid(CANCELLED.into()));
                    }

                    on_event(ChatEvent::ToolFinished { tool: tool_name.clone(), summary: called.summary.clone() });

                    let was_search = tool_name == "search";
                    let record = ToolCallRecord { tool: tool_name, args: tool_args, summary: called.summary.clone() };
                    log_tool_record(&record);
                    tool_records.push(record);

                    if called.empty {
                        empty_searches += 1;
                    } else if !was_search {
                        empty_searches = 0;
                    }
                    stuck |= called.stop;

                    req_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": called.model_text,
                    }));
                }
                // The hint goes after the tool responses: between the assistant `tool_calls`
                // message and its responses even a `user` hint is protocol-invalid on strict
                // servers (400), and the model hears it on the next round either way.
                if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                    hinted = true;
                    req_messages.push(json!({ "role": "user", "content": EMPTY_SEARCH_HINT }));
                }
                if stuck {
                    // It will not stop repeating itself: write the script with what it has.
                    break;
                }
            }

            // A model that saved its edit with save_script has delivered it, the way an MCP session
            // does: only one that never saved is asked for the script here.
            let first = if let Some(id) = session.saved_script {
                on_event(ChatEvent::Validating);
                repair::Answer::Saved(id)
            } else {
                on_event(ChatEvent::Drafting);
                req_messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "Now produce the final complete Script JSON using only the footage returned by the tools. \
                         Follow the pacing rules, and apply this request from the user: {message}"
                    )
                }));
                let final_body = server_script_request(&req_messages, model, *ctx_tokens, ctx.script.max_answer_tokens);
                crate::llmlog::push("prompt", final_body.to_string());
                let mut req =
                    client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&final_body);
                if !api_key.is_empty() {
                    req = req.bearer_auth(api_key.as_str());
                }
                let request_started = std::time::Instant::now();
                let resp = req.send().await.map_err(|e| Error::Agent(crate::agents::error_from_request(url, &e)))?;
                if !resp.status().is_success() {
                    return Err(Error::Agent(crate::agents::error_from_response(url, model, resp).await));
                }
                let v: Value = resp.json().await.map_err(|e| Error::Invalid(format!("bad final JSON: {e}")))?;
                log_server_answer(&v);
                inference.push(crate::inference::InferenceCall::from_response(
                    &v,
                    "server",
                    if model.is_empty() { "Server model not reported" } else { model },
                    request_started.elapsed().as_secs_f64(),
                ));
                let content = v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
                let first = repair::normalise(&ctx.db, &rules!(), repair::Answer::Text(content.clone()));

                // Grounding + pacing check, redraft once on the server backend
                on_event(ChatEvent::Validating);
                let redraft_reasons = match &first {
                    repair::Ready::Draft(d) => {
                        let mut i = check_grounding(&ctx.db, project_id, &d.script, &grounding, &ctx.script);
                        i.extend(pacing_issues(&d.script, enforce_target, &ctx.script));
                        i.extend(content_issues(&ctx.db, &d.script, &ctx.script));
                        i
                    }
                    _ => Vec::new(),
                };
                for issue in &redraft_reasons {
                    crate::llmlog::push("info", format!("Validation: {}", issue.message));
                }
                let mut first = first;
                if !redraft_reasons.is_empty() {
                    // Retry once
                    on_event(ChatEvent::Drafting);
                    let issue_text: Vec<String> = redraft_reasons.iter().map(|i| i.message.clone()).collect();
                    req_messages.push(json!({ "role": "assistant", "content": content }));
                    req_messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "Fix these problems and return the complete corrected Script JSON:\n{}\n\
                             Every clip must lie inside a range returned by the tools and follow the pacing rules. \
                             Keep applying the user's request: {message}",
                            issue_text.join("\n")
                        )
                    }));
                    let retry_body =
                        server_script_request(&req_messages, model, *ctx_tokens, ctx.script.max_answer_tokens);
                    crate::llmlog::push("prompt", retry_body.to_string());
                    let mut retry_req =
                        client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&retry_body);
                    if !api_key.is_empty() {
                        retry_req = retry_req.bearer_auth(api_key.as_str());
                    }
                    let retry_started = std::time::Instant::now();
                    let retry_response = async { retry_req.send().await?.json::<Value>().await }.await;
                    if let Err(error) = &retry_response {
                        crate::llmlog::push("error", format!("Optional redraft: {error}"));
                    }
                    if let Ok(rv) = retry_response {
                        log_server_answer(&rv);
                        inference.push(crate::inference::InferenceCall::from_response(
                            &rv,
                            "server",
                            if model.is_empty() { "Server model not reported" } else { model },
                            retry_started.elapsed().as_secs_f64(),
                        ));
                        let retry_content =
                            rv["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string();
                        if let repair::Ready::Draft(redraft) =
                            repair::normalise(&ctx.db, &rules!(), repair::Answer::Text(retry_content))
                        {
                            // The redraft used to be accepted whatever it was, so a model that
                            // answered one complaint by breaking something else quietly won. Keep it
                            // only when it is better by the same measure the eval uses.
                            let after_issues = {
                                let mut i =
                                    check_grounding(&ctx.db, project_id, &redraft.script, &grounding, &ctx.script);
                                i.extend(pacing_issues(&redraft.script, enforce_target, &ctx.script));
                                i.extend(content_issues(&ctx.db, &redraft.script, &ctx.script));
                                i
                            };
                            let after = score(&measure(&ctx.db, None, &redraft.script, &after_issues)).total;
                            let before = match &first {
                                repair::Ready::Draft(d) => {
                                    score(&measure(&ctx.db, None, &d.script, &redraft_reasons)).total
                                }
                                _ => f64::MIN,
                            };
                            if after >= before {
                                first = repair::Ready::Draft(redraft);
                            } else {
                                pre_issues.push(Issue {
                                    fixes: Vec::new(),
                                    severity: IssueSeverity::Info,
                                    beat_id: None,
                                    clip_index: None,
                                    message: format!(
                                        "kept the first draft: the redraft scored {after:.0} against {before:.0}"
                                    ),
                                });
                            }
                        }
                    }
                }
                repair::Answer::Ready(first)
            };

            // Read back before it is saved, as every writer's cut is: what the finished cut still
            // needs goes back to the model, which answers with the whole script.
            let mut transport = ServerTransport {
                client: &client,
                url,
                model,
                api_key,
                ctx_tokens: *ctx_tokens,
                max_tokens: ctx.script.max_answer_tokens,
                messages: &mut req_messages,
                message,
                inference: &mut inference,
                on_event: &mut *on_event,
                cancel: ctx.cancel.clone(),
            };
            delivered = repair::deliver(&mut ctx.db, &rules!(), &mut transport, first).await?;
            raw_reply = last_assistant_text;
        }
        ChatBackend::Local { helper, ctx_tokens, think } => {
            let mut transcript = format!("<|im_start|>system\n{sys_prompt}<|im_end|>\n");
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    if pm.role == "user" && pm.images.as_ref().is_some_and(|imgs| !imgs.is_empty()) {
                        let notes = pm
                            .images
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|p| format!("[User attached image: {p}]"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        transcript
                            .push_str(&format!("<|im_start|>{}\n{}\n{}\n<|im_end|>\n", pm.role, pm.content, notes));
                    } else {
                        transcript.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", pm.role, pm.content));
                    }
                }
            }
            let user_turn_text = if !saved_image_paths.is_empty() {
                let notes = saved_image_paths
                    .iter()
                    .map(|p| format!("[User attached image: {}]", p.display()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{message}\n\nPlease examine the attached image(s) as part of this request:\n{notes}")
            } else {
                message.to_string()
            };
            transcript.push_str(&format!("<|im_start|>user\n{user_turn_text}<|im_end|>\n"));

            // A full script's JSON is far longer than a tool action, so the draft call gets its
            // own budget out of the configured context rather than the default.
            // Reasoning is spent from the same budget as the answer, so leave room for both.
            let script_token_budget = local_output_budget(*ctx_tokens, ctx.script.max_answer_tokens);

            let mut budget = research::ResearchBudget::new(round_cap);
            let mut empty_searches = 0usize;
            let mut hinted = false;
            // Whether any footage has been opened, and whether the "you have not looked yet"
            // nudge has already been spent.
            let mut tools_used = 0usize;
            let mut pushed_back = 0usize;
            // The editor's answer, when it gave one: reviewed after the loop, like every backend's.
            let mut delivered_answer: Option<repair::Answer> = None;
            while budget.can_research() {
                if cancelled() {
                    // The generation runs inside the helper process; ending the turn means
                    // ending it, or the model keeps going and Stop waits for it.
                    helper.kill().await;
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                budget.calls += 1;
                *rounds_used += 1;
                // A tool action is short, but a model that reasons first spends the same budget on
                // the thought — at the default 2048 it ran out mid-round and the turn died.
                fit_local_prompt(
                    helper,
                    &mut transcript,
                    &user_turn_text,
                    *ctx_tokens,
                    script_token_budget,
                    *think,
                    saved_image_paths.first().map(|p| p.as_path()),
                )
                .await?;
                let out_str = helper
                    .complete_full_with_image(
                        &transcript,
                        saved_image_paths.first().map(|p| p.as_path()),
                        Some(local_action_schema()),
                        script_token_budget,
                        *think,
                    )
                    .await?;
                // Every round, not just the final draft: a turn that ends in `Reply` never
                // reaches the draft call, so the one dump that existed showed nothing at all
                // about why a local model talked instead of editing.
                debug_dump(&format!("local-round-{}", budget.calls), &transcript, &out_str);
                let action: Result<LocalAction, _> = serde_json::from_str(&out_str);
                match action {
                    // Answering in words is a complete turn: nothing is drafted and the previous
                    // version stays as it is.
                    Ok(LocalAction::Reply { text }) => {
                        // Replying before looking at anything is the cheapest branch of the
                        // grammar and models take it: asked for a 40 s piece about a
                        // neighbourhood, two different local models declined in one round —
                        // "nobody explicitly states 'I live here'" — while the transcripts named
                        // the place nineteen times and three speakers said they grew up there.
                        // The prompt already forbids this ("do not reply to avoid work"); a rule
                        // the harness can enforce should not be left to the model's manners.
                        //
                        // Pushed back on once, and only once: a second reply is a considered one
                        // and ends the turn, which is what the escape hatch is for.
                        if tools_used == 0 && pushed_back < MAX_PUSHBACKS {
                            pushed_back += 1;
                            transcript.push_str(&format!(
                                "<|im_start|>assistant\n{{\"action\":\"reply\",\"text\":{}}}<|im_end|>\n",
                                serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".into())
                            ));
                            // The second nudge spells out the action, because the first earns
                            // answers like "I need to verify the visual content for the
                            // interviewee" — the model knows what to do and describes doing it
                            // instead of doing it. Saying "use a tool" is not enough; showing the
                            // JSON is.
                            transcript.push_str(if pushed_back == 1 {
                                "<|im_start|>user\nYou have not opened any footage yet, so that answer is a guess. \
                                 Start where the instructions say: get_outline, then read what your story needs, and \
                                 then draft. Only say the footage cannot support the request after you have \
                                 looked.<|im_end|>\n"
                            } else {
                                "<|im_start|>user\nDo not describe what you need to check: check it. Answer with \
                                 the tool action itself, for example \
                                 {\"action\":\"tool\",\"tool\":\"search\",\"args\":{\"query\":\"quiet \
                                 residential street\"}} or \
                                 {\"action\":\"tool\",\"tool\":\"get_video\",\"args\":{\"video_id\":41}}. The \
                                 length you were given is the target; it is not something to ask about.<|im_end|>\n"
                            });
                            continue;
                        }
                        raw_reply = text;
                        break;
                    }
                    Ok(LocalAction::Tool { tool, args }) => {
                        tools_used += 1;
                        budget.tools += 1;
                        let tool = crate::tools::canonical_name(&tool).to_string();
                        on_event(ChatEvent::ToolStarted { tool: tool.clone(), args: args.clone() });

                        // Remembered, like the server and CLI paths do. Without it a local model
                        // asked the same question forever: four identical get_video calls for
                        // video #100, 52-79 s, one after another, each answered afresh and none of
                        // them advancing the turn.
                        let called = {
                            let mut tctx =
                                ToolCtx::new(&ctx.paths, &tool_cfg, &mut ctx.db, &mut ctx.embedder, Origin::ScriptChat)
                                    .with_project(project.clone())
                                    .with_cancel(ctx.cancel.clone())
                                    .with_grounding(&mut grounding);
                            session.call(&mut tctx, &tool, &args).await
                        };
                        if called.cancelled {
                            helper.kill().await;
                            return Err(Error::Invalid(CANCELLED.into()));
                        }
                        on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: called.summary.clone() });
                        let res = called.model_text.clone();
                        let record =
                            ToolCallRecord { tool: tool.clone(), args: args.clone(), summary: called.summary.clone() };
                        log_tool_record(&record);
                        tool_records.push(record);

                        if called.empty {
                            empty_searches += 1;
                        } else if tool != "search" {
                            empty_searches = 0;
                        }
                        let hint = if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                            hinted = true;
                            format!("\n{EMPTY_SEARCH_HINT}")
                        } else {
                            String::new()
                        };
                        transcript.push_str(&format!(
                            "<|im_start|>assistant\n{}\n<|im_end|>\n<|im_start|>user\nTool result for {tool}:\n{res}{hint}\n<|im_end|>\n",
                            out_str
                        ));
                        if called.stop {
                            // It will not stop repeating itself: draft with what it has.
                            break;
                        }
                    }
                    // The script: delivered, and reviewed below like every backend's.
                    Ok(LocalAction::Final { script }) => {
                        delivered_answer = Some(repair::Answer::Script(script));
                        break;
                    }
                    // Not an action: fed back below with the reason, like any unusable answer.
                    Err(_) => {
                        delivered_answer = Some(repair::Answer::Text(out_str));
                        break;
                    }
                }
            }

            // What the editor delivered: the script it answered with, else the version it saved
            // itself, else (when it did not answer in words) the script asked for now.
            let first = match delivered_answer {
                Some(answer) => Some(answer),
                None => match session.saved_script {
                    Some(id) => Some(repair::Answer::Saved(id)),
                    None if !raw_reply.is_empty() => None,
                    None => {
                        if cancelled() {
                            return Err(Error::Invalid(CANCELLED.into()));
                        }
                        on_event(ChatEvent::Drafting);
                        transcript.push_str(&format!(
                            "<|im_start|>user\n{}Produce the final script action.<|im_end|>\n",
                            allowed_clips_text(&grounding)
                        ));
                        fit_local_prompt(
                            helper,
                            &mut transcript,
                            &user_turn_text,
                            *ctx_tokens,
                            script_token_budget,
                            *think,
                            None,
                        )
                        .await?;
                        let final_str = helper
                            .complete_full(
                                &transcript,
                                Some(local_final_action_schema(tools_used == 0)),
                                script_token_budget,
                                *think,
                            )
                            .await?;
                        debug_dump("final-draft", &transcript, &final_str);
                        Some(repair::Answer::Text(final_str))
                    }
                },
            };

            on_event(ChatEvent::Validating);

            if let Some(first) = first {
                let mut first = repair::normalise(&ctx.db, &rules!(), first);
                // Redraft once against the same checks the server backend uses. Without this the
                // local model's first answer is final, and its usual miss — beats with a purpose
                // but no narration — reached the user as a finished script full of silent beats.
                // A version the editor saved itself has been through its own save.
                if let repair::Ready::Draft(d) = &first
                    && d.saved.is_none()
                {
                    let mut reasons = check_grounding(&ctx.db, project_id, &d.script, &grounding, &ctx.script);
                    reasons.extend(pacing_issues(&d.script, enforce_target, &ctx.script));
                    reasons.extend(content_issues(&ctx.db, &d.script, &ctx.script));
                    for issue in &reasons {
                        crate::llmlog::push("info", format!("Validation: {}", issue.message));
                    }
                    if !reasons.is_empty() {
                        on_event(ChatEvent::Drafting);
                        let issue_text: Vec<String> = reasons.iter().map(|i| i.message.clone()).collect();
                        transcript.push_str(&format!(
                            "<|im_start|>assistant\n{}<|im_end|>\n",
                            serde_json::to_string(&LocalAction::Final { script: d.script.clone() }).unwrap_or_default()
                        ));
                        transcript.push_str(&format!(
                            "<|im_start|>user\nFix these problems and return the complete corrected script action:\n{}\n\
                             {}Keep applying the user's request: {message}<|im_end|>\n",
                            issue_text.join("\n"),
                            allowed_clips_text(&grounding)
                        ));
                        let fits = fit_local_prompt(
                            helper,
                            &mut transcript,
                            &user_turn_text,
                            *ctx_tokens,
                            script_token_budget,
                            *think,
                            saved_image_paths.first().map(|p| p.as_path()),
                        )
                        .await;
                        if let Err(error) = &fits {
                            crate::llmlog::push("info", format!("Skipped optional redraft: {error}"));
                        }
                        if fits.is_ok()
                            && let Ok(retry_str) = helper
                                .complete_full_with_image(
                                    &transcript,
                                    saved_image_paths.first().map(|p| p.as_path()),
                                    Some(local_final_action_schema(false)),
                                    script_token_budget,
                                    *think,
                                )
                                .await
                            && let Ok(LocalAction::Final { script }) = serde_json::from_str::<LocalAction>(&retry_str)
                            && let redraft @ repair::Ready::Draft(_) =
                                repair::normalise(&ctx.db, &rules!(), repair::Answer::Script(script))
                        {
                            first = redraft;
                        }
                    }
                }
                let mut transport = LocalTransport {
                    helper,
                    transcript: &mut transcript,
                    brief: &user_turn_text,
                    ctx_tokens: *ctx_tokens,
                    output: script_token_budget,
                    think: *think,
                    image: saved_image_paths.first().map(|p| p.as_path()),
                    cfg: &ctx.script,
                    on_event: &mut *on_event,
                    cancel: ctx.cancel.clone(),
                };
                delivered =
                    repair::deliver(&mut ctx.db, &rules!(), &mut transport, repair::Answer::Ready(first)).await?;
            }
        }
        ChatBackend::Cli(agent) => {
            // Drive the same local action schema loop as the Local backend.
            // The agent CLI receives the whole transcript as a single prompt each turn.
            let mut transcript = format!("<|im_start|>system\n{sys_prompt}<|im_end|>\n");
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    if pm.role == "user" && pm.images.as_ref().is_some_and(|imgs| !imgs.is_empty()) {
                        let notes = pm
                            .images
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|p| format!("[User attached image: {p}]"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        transcript
                            .push_str(&format!("<|im_start|>{}\n{}\n{}\n<|im_end|>\n", pm.role, pm.content, notes));
                    } else {
                        transcript.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", pm.role, pm.content));
                    }
                }
            }
            let user_turn_text = if !saved_image_paths.is_empty() {
                let notes = saved_image_paths
                    .iter()
                    .map(|p| format!("[User attached image: {}]", p.display()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{message}\n\nPlease examine the attached image(s) as part of this request:\n{notes}")
            } else {
                message.to_string()
            };
            transcript.push_str(&format!("<|im_start|>user\n{user_turn_text}<|im_end|>\n"));

            let schema_json = serde_json::to_string(&local_action_schema()).unwrap_or_default();
            let final_schema_json = serde_json::to_string(&local_final_action_schema(true)).unwrap_or_default();

            let events_file = crate::tools::TempTrailFile::create_named("chat-mcp-events");
            let events_path = events_file.as_ref().map(|f| f.path().to_path_buf());
            let mut seen_mcp_events = 0usize;
            let project_id_str = project_id.to_string();
            let events_path_str = events_path.as_ref().map(|p| p.to_string_lossy().into_owned());
            let mut cli_env: Vec<(&str, &str)> = vec![
                ("GHOSTREEL_MCP_ORIGIN", "script_chat"),
                ("GHOSTREEL_MCP_PROJECT", &project_id_str),
                ("GHOSTREEL_MCP_PROFILE", "all"),
            ];
            if let Some(p) = events_path_str.as_deref() {
                cli_env.push(("GHOSTREEL_MCP_EVENTS_FILE", p));
            }

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            // The editor's answer, when it gave one: reviewed after the loop, like every backend's.
            let mut delivered_answer: Option<repair::Answer> = None;
            // What has happened since the tool last spoke: sent on its own when continuing.
            let mut new_since_last = String::new();
            while round_cap.is_none_or(|cap| tool_rounds < cap) {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                tool_rounds += 1;
                *rounds_used += 1;
                // The first round carries the whole transcript; after that the tool continues its
                // own conversation and hears only what is new. Re-sending everything each round
                // grows quadratically — on this project's library agy timed out on round one.
                let first = tool_rounds == 1;
                let cli_prompt = if first {
                    format!(
                        "{transcript}<|im_start|>assistant\n\
                         Reply ONLY with a JSON object matching this schema:\n{schema_json}\n<|im_end|>\n"
                    )
                } else {
                    format!("{new_since_last}\nReply ONLY with a JSON object matching the same schema.")
                };
                let answered = complete_cli_streaming_mcp(
                    agent,
                    &cli_prompt,
                    !first,
                    &cli_env,
                    events_path.as_deref(),
                    &mut seen_mcp_events,
                    on_event,
                    &mut tool_records,
                    &log_tool_record,
                    &mut session.saved_script,
                )
                .await;
                // An answer with no JSON in it is an error from the CLI; after the editor saved a
                // version itself it is the editor saying so in words, and the version is the answer.
                let out_str = match (answered, session.saved_script) {
                    (Ok(text), _) => text,
                    (Err(e), Some(id)) => {
                        crate::llmlog::push("error", format!("CLI answer after saving #{id}: {e}"));
                        delivered_answer = Some(repair::Answer::Saved(id));
                        break;
                    }
                    (Err(e), None) => return Err(e),
                };
                debug_dump(&format!("cli-round-{tool_rounds}"), &cli_prompt, &out_str);
                new_since_last.clear();
                let action: Result<LocalAction, _> = serde_json::from_str(&out_str);
                match action {
                    // Answering in words is a complete turn: nothing is drafted and the previous
                    // version stays as it is.
                    Ok(LocalAction::Reply { text }) => {
                        raw_reply = text;
                        break;
                    }
                    Ok(LocalAction::Tool { tool, args }) => {
                        let tool = crate::tools::canonical_name(&tool).to_string();
                        on_event(ChatEvent::ToolStarted { tool: tool.clone(), args: args.clone() });

                        let called = {
                            let mut tctx =
                                ToolCtx::new(&ctx.paths, &tool_cfg, &mut ctx.db, &mut ctx.embedder, Origin::ScriptChat)
                                    .with_project(project.clone())
                                    .with_cancel(ctx.cancel.clone())
                                    .with_grounding(&mut grounding);
                            session.call(&mut tctx, &tool, &args).await
                        };
                        if called.cancelled {
                            return Err(Error::Invalid(CANCELLED.into()));
                        }
                        on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: called.summary.clone() });
                        let res = called.model_text.clone();
                        let record =
                            ToolCallRecord { tool: tool.clone(), args: args.clone(), summary: called.summary.clone() };
                        log_tool_record(&record);
                        tool_records.push(record);

                        if called.empty {
                            empty_searches += 1;
                        } else if tool != "search" {
                            empty_searches = 0;
                        }
                        let hint = if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                            hinted = true;
                            format!("\n{EMPTY_SEARCH_HINT}")
                        } else {
                            String::new()
                        };
                        transcript.push_str(&format!(
                            "<|im_start|>assistant\n{out_str}\n<|im_end|>\n<|im_start|>user\nTool result for {tool}:\n{res}{hint}\n<|im_end|>\n"
                        ));
                        // The same thing, for a tool that is continuing and has the rest already.
                        new_since_last.push_str(&format!("Tool result for {tool}:\n{res}{hint}\n"));
                        if called.stop {
                            // It will not stop repeating itself: ask for the final script now.
                            break;
                        }
                    }
                    // The script: delivered, and reviewed below like every backend's.
                    Ok(LocalAction::Final { script }) => {
                        delivered_answer = Some(repair::Answer::Script(script));
                        break;
                    }
                    // Not an action: perhaps a script wrapped in prose, perhaps nothing usable. Either
                    // way it is reviewed below (or its reason fed back); prose with no script in it
                    // after the editor saved a version itself means the saved version.
                    Err(_) => {
                        delivered_answer = Some(match session.saved_script {
                            Some(id) if !repair::holds_script(&out_str) => repair::Answer::Saved(id),
                            _ => repair::Answer::Text(out_str),
                        });
                        break;
                    }
                }
            }

            // What the editor delivered: the script it answered with, else the version it saved
            // itself, else (when it did not answer in words) the script asked for now.
            let first = match delivered_answer {
                Some(answer) => Some(answer),
                None => match session.saved_script {
                    Some(id) => Some(repair::Answer::Saved(id)),
                    None if !raw_reply.is_empty() => None,
                    None => {
                        if cancelled() {
                            return Err(Error::Invalid(CANCELLED.into()));
                        }
                        on_event(ChatEvent::Drafting);
                        let cli_final_prompt = if tool_rounds > 0 {
                            format!(
                                "{new_since_last}\nProduce the final script. Reply ONLY with a JSON object matching \
                                 this schema:\n{final_schema_json}"
                            )
                        } else {
                            format!(
                                "{transcript}<|im_start|>assistant\n\
                                 Produce the final script. Reply ONLY with a JSON object matching this schema:\n{final_schema_json}\n<|im_end|>\n"
                            )
                        };
                        let before = session.saved_script;
                        let answered = complete_cli_streaming_mcp(
                            agent,
                            &cli_final_prompt,
                            tool_rounds > 0,
                            &cli_env,
                            events_path.as_deref(),
                            &mut seen_mcp_events,
                            on_event,
                            &mut tool_records,
                            &log_tool_record,
                            &mut session.saved_script,
                        )
                        .await;
                        match (answered, session.saved_script) {
                            (Ok(final_str), saved) => {
                                debug_dump("cli-final", &cli_final_prompt, &final_str);
                                Some(cli_answer(final_str, before, saved))
                            }
                            // Saved, then answered in words: the saved version.
                            (Err(e), Some(id)) => {
                                crate::llmlog::push("error", format!("CLI final answer after saving #{id}: {e}"));
                                Some(repair::Answer::Saved(id))
                            }
                            (Err(e), None) => return Err(e),
                        }
                    }
                },
            };

            on_event(ChatEvent::Validating);
            if let Some(first) = first {
                let mut transport = CliTransport {
                    agent,
                    env: &cli_env,
                    events_path: events_path.as_deref(),
                    seen: &mut seen_mcp_events,
                    on_event: &mut *on_event,
                    tool_records: &mut tool_records,
                    saved: &mut session.saved_script,
                    final_schema: &final_schema_json,
                    cancel: ctx.cancel.clone(),
                };
                delivered = repair::deliver(&mut ctx.db, &rules!(), &mut transport, first).await?;
            }
        }
    }

    let mut script_id = None;
    let mut parsed_script: Option<Script> = None;
    let mut issues = pre_issues;
    issues.extend(delivered.issues);
    if raw_reply.trim().is_empty()
        && let Some(reply) = delivered.reply
    {
        raw_reply = reply;
    }
    // Why the turn's script could not be used, when it could not: said in the reply instead of a
    // bland "unable to assemble", so the user (and the next turn) know what to change.
    let assembly_error = delivered.error;
    // The cut exactly as it was reviewed: finished once, saved as read.
    if let Some(s) = delivered.script {
        script_id = Some(repair::save(&ctx.db, project_id, session_id, &s, delivered.saved)?);
        parsed_script = Some(s);
    }

    // What the pipeline changed after the draft was written — beds laid, clips moved off shaky
    // stretches, ranges dropped. The model is never told this in the turn that caused it, because
    // the repair runs after the last redraft, so it repeats the same edit next time and the
    // pipeline undoes it again. Recording it in the reply puts it in what the next turn replays.
    let repair_note = repair_note(&issues);

    // The editorial read, when one is configured. A repair pass can move a cut onto a sentence;
    // it cannot make a shot of a road illustrate a sentence about a dog, so what Jev finds goes
    // where the mechanical repairs go — onto the assistant message, which every backend replays —
    // and the only thing that can choose a different shot reads it next turn.
    //
    // A judgement that fails is not a failed turn: the script is already saved and the cut is
    // already good or bad on its own terms.
    // Read the cut first, ask second: the database is not `Sync`, and a future still holding it
    // at an await point cannot be spawned by the app's queue worker.
    // Judged against the *brief*, which is the first thing asked in this session — not `message`,
    // which on any turn after the first is an instruction about the cut rather than a description
    // of it. Refining scored "says what was asked" against "Improve this cut. The editorial read
    // above says exactly where it is losing points…", and the same script read 67 in the turn and
    // 76 when judged against what was actually asked for.
    let brief = session_brief(&ctx.db, session_id).unwrap_or_else(|| message.to_string());
    // The judge reads a cut against its voice — openings, the shots under what is said. A
    // montage has none, and its answers would be about a piece that was not asked for.
    let planned =
        parsed_script.as_ref().filter(|_| !style.broll).and_then(|s| judge::plan(&ctx.db, s, Some(&brief), &ctx.jev));
    let judgement = match planned {
        Some(p) => match p.ask().await {
            Ok(j) => Some(j),
            Err(e) => {
                tracing::warn!("editorial judge: {e}");
                None
            }
        },
        None => None,
    };
    let judge_note = judgement
        .as_ref()
        .map(Judgement::notes)
        .filter(|n| !n.is_empty())
        .map(|notes| format!("[editorial read of the saved cut, fix these next time: {}]", notes.join("; ")));

    let reply = if !raw_reply.trim().is_empty() {
        raw_reply
    } else if let Some(s) = &parsed_script {
        format!(
            "Drafted '{}': {} beats, {} clips, {:.1} s",
            s.title,
            s.beats.len(),
            s.clip_count(),
            s.total_duration_s()
        )
    } else if let Some(why) = &assembly_error {
        format!("I could not assemble a script: {why}")
    } else {
        "I was unable to assemble a script from the available footage.".to_string()
    };

    if let ChatBackend::Local { helper, .. } = &mut ctx.backend {
        inference.extend(helper.take_inference());
    }
    if inference.is_empty() {
        let (backend, model) = match &ctx.backend {
            ChatBackend::Cli(agent) => (
                format!("CLI · {}", agent.cfg.tool),
                if agent.cfg.model.is_empty() { "Model not reported".to_string() } else { agent.cfg.model.clone() },
            ),
            ChatBackend::Local { .. } => ("local".to_string(), "Local model not reported".to_string()),
            ChatBackend::Server { model, .. } => (
                "server".to_string(),
                if model.is_empty() { "Server model not reported".to_string() } else { model.clone() },
            ),
        };
        inference.push(crate::inference::InferenceCall::from_response(
            &Value::Null,
            &backend,
            &model,
            turn_started.elapsed().as_secs_f64(),
        ));
    }
    let performance =
        crate::inference::ChatPerformance { elapsed_seconds: turn_started.elapsed().as_secs_f64(), calls: inference };

    // Persist messages in database
    let now_ts = now();

    // 1. The user's message was stored before the work started.

    // 2. Tool summary row if any tools were called
    if !tool_records.is_empty() {
        let tool_summary = format!(
            "Used {} tools: {}",
            tool_records.len(),
            tool_records.iter().map(|t| t.tool.as_str()).collect::<Vec<_>>().join(", ")
        );
        let tool_json = serde_json::to_string(&tool_records).ok();
        ctx.db.conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
             VALUES (?1, 'tool', ?2, ?3, ?4)",
            params![session_id, tool_summary, tool_json, now_ts],
        )?;
    }

    // 3. Assistant message. The repairs ride along with it: this is the text every backend
    // replays next turn, so it is the one place a note reaches the server, local and CLI brains
    // alike.
    let assistant_tc_json = Some(json!({ "script_id": script_id, "performance": performance }).to_string());
    let mut stored_reply = reply.clone();
    let research_note =
        issues.iter().find(|i| i.message.starts_with("research incomplete:")).map(|i| format!("[{}]", i.message));
    for note in [repair_note.as_ref(), judge_note.as_ref(), research_note.as_ref()].into_iter().flatten() {
        stored_reply.push_str("\n\n");
        stored_reply.push_str(note);
        crate::llmlog::push("info", note);
    }
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'assistant', ?2, ?3, ?4)",
        params![session_id, stored_reply, assistant_tc_json, now_ts],
    )?;

    // Update chat_sessions updated_at
    ctx.db.conn.execute("UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2", params![now_ts, session_id])?;

    for issue in &issues {
        crate::llmlog::push(if issue.severity == IssueSeverity::Error { "error" } else { "info" }, &issue.message);
    }
    Ok(TurnResult { session_id, reply, script_id, script: parsed_script, issues, tool_calls: tool_records, judgement })
}

/// The request that asks a server model for the whole script, constrained to the script schema.
fn server_script_request(messages: &[Value], model: &str, ctx_tokens: u32, max_tokens: u32) -> Value {
    let mut body = json!({
        "messages": messages,
        "temperature": 0.3,
        // Deliberately generous: a long script must never be cut off. The cap only stops a model
        // that will not stop at all, so the turn fails as itself rather than as a dead socket —
        // Bonsai-27B at 1 bit produced 11 347 tokens of one draft that way.
        "max_tokens": max_tokens,
        "chat_template_kwargs": { "enable_thinking": false },
        "response_format": {
            "type": "json_schema",
            "json_schema": { "name": "script", "strict": true, "schema": script_json_schema() }
        }
    });
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    body["options"] = json!({ "num_ctx": ctx_tokens });
    body
}

/// Log a server model's answer (and its reasoning, when it shows it).
fn log_server_answer(v: &Value) {
    let message = &v["choices"][0]["message"];
    if let Some(thinking) = message["reasoning_content"].as_str() {
        crate::llmlog::push("thinking", thinking);
    }
    crate::llmlog::push("answer", message.to_string());
}

/// What a CLI's answer delivered: the version it saved itself through MCP during this call, when
/// it saved one and did not answer with a script (in any form a script is read from); else its
/// text, reviewed as any answer is.
fn cli_answer(text: String, saved_before: Option<i64>, saved_now: Option<i64>) -> repair::Answer {
    let answered_with_script = repair::holds_script(&text);
    match saved_now {
        Some(id) if saved_now != saved_before && !answered_with_script => repair::Answer::Saved(id),
        _ => repair::Answer::Text(text),
    }
}

fn is_cancelled(cancel: &Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) -> bool {
    cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst))
}

/// A review round on a server model: the note goes in as the next user message, after the script
/// it is about, and the answer comes back constrained to the script schema.
struct ServerTransport<'a, 'e> {
    client: &'a reqwest::Client,
    url: &'a str,
    model: &'a str,
    api_key: &'a str,
    ctx_tokens: u32,
    max_tokens: u32,
    messages: &'a mut Vec<Value>,
    /// The user's request, repeated with every note.
    message: &'a str,
    inference: &'a mut Vec<crate::inference::InferenceCall>,
    on_event: &'a mut (dyn FnMut(ChatEvent) + Send + 'e),
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl repair::Transport for ServerTransport<'_, '_> {
    async fn ask(&mut self, note: &str, replay: repair::Replay<'_>) -> repair::Answer {
        if is_cancelled(&self.cancel) {
            return repair::Answer::Cancelled;
        }
        (self.on_event)(ChatEvent::Drafting);
        let replay = match replay {
            repair::Replay::Script(s) => serde_json::to_string(s).unwrap_or_default(),
            repair::Replay::Text(t) => t.to_string(),
        };
        self.messages.push(json!({ "role": "assistant", "content": replay }));
        self.messages.push(json!({
            "role": "user",
            "content": format!(
                "{note}\nReturn the complete corrected Script JSON. Keep applying the user's request: {}",
                self.message
            )
        }));
        let body = server_script_request(self.messages, self.model, self.ctx_tokens, self.max_tokens);
        crate::llmlog::push("prompt", body.to_string());
        let mut req = self.client.post(format!("{}/v1/chat/completions", self.url.trim_end_matches('/'))).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(self.api_key);
        }
        let started = std::time::Instant::now();
        match async { req.send().await?.json::<Value>().await }.await {
            Ok(rv) => {
                log_server_answer(&rv);
                self.inference.push(crate::inference::InferenceCall::from_response(
                    &rv,
                    "server",
                    if self.model.is_empty() { "Server model not reported" } else { self.model },
                    started.elapsed().as_secs_f64(),
                ));
                repair::Answer::Text(rv["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string())
            }
            Err(error) => {
                // The cut under review is still a cut: keep it.
                crate::llmlog::push("error", format!("Review round: {error}"));
                repair::Answer::Nothing
            }
        }
    }
}

/// A review round on a local model: the note goes into its transcript after the script it is
/// about, and the answer comes back constrained to the final-script action. Each draft gets
/// narration written for its silent beats before it is finished.
struct LocalTransport<'a, 'e> {
    helper: &'a mut crate::vision::LocalLlm,
    transcript: &'a mut String,
    /// The user's turn, kept when the transcript is trimmed to fit.
    brief: &'a str,
    ctx_tokens: u32,
    output: usize,
    think: bool,
    image: Option<&'a Path>,
    cfg: &'a crate::config::ScriptConfig,
    on_event: &'a mut (dyn FnMut(ChatEvent) + Send + 'e),
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl repair::Transport for LocalTransport<'_, '_> {
    async fn ask(&mut self, note: &str, replay: repair::Replay<'_>) -> repair::Answer {
        if is_cancelled(&self.cancel) {
            // The generation runs inside the helper process: ending the turn means ending it.
            self.helper.kill().await;
            return repair::Answer::Cancelled;
        }
        (self.on_event)(ChatEvent::Drafting);
        let replay = match replay {
            repair::Replay::Script(s) => {
                serde_json::to_string(&LocalAction::Final { script: s.clone() }).unwrap_or_default()
            }
            repair::Replay::Text(t) => t.to_string(),
        };
        self.transcript
            .push_str(&format!("<|im_start|>assistant\n{replay}\n<|im_end|>\n<|im_start|>user\n{note}<|im_end|>\n"));
        if let Err(error) = fit_local_prompt(
            self.helper,
            self.transcript,
            self.brief,
            self.ctx_tokens,
            self.output,
            self.think,
            self.image,
        )
        .await
        {
            crate::llmlog::push("error", format!("Review round: {error}"));
            return repair::Answer::Nothing;
        }
        match self
            .helper
            .complete_full_with_image(
                self.transcript,
                self.image,
                Some(local_final_action_schema(false)),
                self.output,
                self.think,
            )
            .await
        {
            Ok(answer) => {
                debug_dump("local-review", self.transcript, &answer);
                repair::Answer::Text(answer)
            }
            Err(error) => {
                crate::llmlog::push("error", format!("Review round: {error}"));
                repair::Answer::Nothing
            }
        }
    }

    async fn fill(&mut self, db: &mut Db, draft: &mut Script) -> Vec<Issue> {
        // A montage has no voice-over to write.
        if self.cfg.style.broll {
            return Vec::new();
        }
        let jobs = narration_jobs(db, draft, self.cfg);
        let filled = fill_missing_narration(self.helper, draft, jobs).await;
        if filled == 0 {
            return Vec::new();
        }
        vec![Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("wrote narration for {filled} silent beat(s)"),
        }]
    }
}

/// A review round on a coding-agent CLI: it continues its own conversation, so it hears only the
/// note, and it may answer with the script or by saving a version through its MCP tools.
struct CliTransport<'a, 'e> {
    agent: &'a crate::cliagent::CliAgent,
    env: &'a [(&'a str, &'a str)],
    events_path: Option<&'a Path>,
    seen: &'a mut usize,
    on_event: &'a mut (dyn FnMut(ChatEvent) + Send + 'e),
    tool_records: &'a mut Vec<ToolCallRecord>,
    saved: &'a mut Option<i64>,
    final_schema: &'a str,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl repair::Transport for CliTransport<'_, '_> {
    async fn ask(&mut self, note: &str, _replay: repair::Replay<'_>) -> repair::Answer {
        if is_cancelled(&self.cancel) {
            return repair::Answer::Cancelled;
        }
        (self.on_event)(ChatEvent::Drafting);
        let prompt = format!("{note}\nReply ONLY with a JSON object matching this schema:\n{}", self.final_schema);
        let before = *self.saved;
        let answer = complete_cli_streaming_mcp(
            self.agent,
            &prompt,
            true,
            self.env,
            self.events_path,
            self.seen,
            &mut *self.on_event,
            self.tool_records,
            &log_tool_record,
            self.saved,
        )
        .await;
        debug_dump("cli-review", &prompt, answer.as_deref().unwrap_or_default());
        match answer {
            Ok(text) => cli_answer(text, before, *self.saved),
            Err(error) => {
                crate::llmlog::push("error", format!("Review round: {error}"));
                match *self.saved {
                    Some(id) if *self.saved != before => repair::Answer::Saved(id),
                    // The cut under review is still a cut: keep it.
                    _ => repair::Answer::Nothing,
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn complete_cli_streaming_mcp(
    agent: &crate::cliagent::CliAgent,
    prompt: &str,
    continued: bool,
    env: &[(&str, &str)],
    events_path: Option<&std::path::Path>,
    seen_lines: &mut usize,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
    tool_records: &mut Vec<ToolCallRecord>,
    log_tool_record: &(dyn Fn(&ToolCallRecord) + Sync),
    saved_script: &mut Option<i64>,
) -> Result<String, Error> {
    let fut = agent.complete_continuing_with_env(prompt, continued, env);
    let Some(path) = events_path else {
        return fut.await;
    };
    tokio::pin!(fut);
    let mut tick = tokio::time::interval(Duration::from_millis(150));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let result = loop {
        tokio::select! {
            res = &mut fut => break res,
            _ = tick.tick() => {
                drain_mcp_events(path, seen_lines, on_event, tool_records, log_tool_record, saved_script);
            }
        }
    };
    drain_mcp_events(path, seen_lines, on_event, tool_records, log_tool_record, saved_script);
    result
}

/// The lines of a file being written as we read it that are complete: those ending in a newline,
/// blanks dropped. A trailing partial line stays out, so the next drain sees it once the writer
/// has finished it — counting a torn line as seen now would skip it forever.
pub(crate) fn complete_lines(content: &str) -> Vec<&str> {
    content
        .split_inclusive('\n')
        .filter_map(|line| {
            let line = line.strip_suffix('\n')?;
            if line.trim().is_empty() { None } else { Some(line) }
        })
        .collect()
}

fn drain_mcp_events(
    path: &std::path::Path,
    seen_lines: &mut usize,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
    tool_records: &mut Vec<ToolCallRecord>,
    log_tool_record: &(dyn Fn(&ToolCallRecord) + Sync),
    saved_script: &mut Option<i64>,
) {
    let Ok(content) = std::fs::read_to_string(path) else { return };
    let lines = complete_lines(&content);
    if lines.len() <= *seen_lines {
        return;
    }
    for line in &lines[*seen_lines..] {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        let tool = v.get("tool").and_then(Value::as_str).unwrap_or_default().to_string();
        let args = v.get("args").cloned().unwrap_or_else(|| json!({}));
        match v.get("kind").and_then(Value::as_str) {
            Some("tool_started") if !tool.is_empty() => {
                on_event(ChatEvent::ToolStarted { tool, args });
            }
            Some("tool_finished") if !tool.is_empty() => {
                let summary = v.get("summary").and_then(Value::as_str).unwrap_or_default().to_string();
                if let Some(id) = v.get("script_id").and_then(Value::as_i64) {
                    *saved_script = Some(id);
                }
                on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: summary.clone() });
                let record = ToolCallRecord { tool, args, summary };
                log_tool_record(&record);
                tool_records.push(record);
            }
            _ => {}
        }
    }
    *seen_lines = lines.len();
}

/// What the pipeline changed after the draft was written — beds laid, clips moved off shaky
/// stretches, ranges dropped — written for the model rather than the user.
///
/// The repair runs after the last redraft, so nothing tells the model in the turn that caused it;
/// it makes the same edit next time and the pipeline undoes it again. Carried on the assistant
/// message, this reaches the server, local and CLI brains alike, since all three replay it.
fn repair_note(issues: &[Issue]) -> Option<String> {
    let repairs: Vec<&str> =
        issues.iter().filter(|i| i.severity != IssueSeverity::Error).map(|i| i.message.as_str()).take(6).collect();
    if repairs.is_empty() {
        return None;
    }
    Some(format!("[applied after drafting, already in the saved script: {}]", repairs.join("; ")))
}

/// A beat's id is a handle, not a sentence: the preview groups clips by it, titles and captions
/// are keyed on it, and the editor scrolls to it. Models treat it loosely — one pasted the whole
/// quote it had just found, another left it empty on every beat — so an unusable id is replaced
/// with a slug of the beat's purpose, and duplicates are numbered.
fn tidy_beat_ids(script: &mut Script) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut fixed = 0usize;

    for (i, beat) in script.beats.iter_mut().enumerate() {
        let original = beat.id.clone();
        let usable = !original.trim().is_empty() && original.chars().count() <= 40;
        let mut id = if usable {
            original.trim().to_string()
        } else {
            let slug: String = beat
                .purpose
                .split_whitespace()
                .take(4)
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join("-");
            if slug.is_empty() { format!("beat-{}", i + 1) } else { slug }
        };
        if id != original {
            fixed += 1;
        }
        let base = id.clone();
        let mut n = 2;
        while !seen.insert(id.clone()) {
            id = format!("{base}-{n}");
            n += 1;
        }
        beat.id = id;
    }

    if fixed > 0 {
        issues.push(Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("named {fixed} beat(s) after their purpose"),
        });
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    // The finishing passes these tests drive, as a turn's finish runs them.
    use crate::finish::fit::drop_beats_to_target;
    use crate::finish::{
        clamp_beds_to_beats, drop_closing_picture, end_on_sentences, end_on_turns, fit_to_target,
        hold_the_last_picture, lay_audio_beds, mute_clips_without_speech, pad_speech, trim_pictures_to_bed,
    };

    #[test]
    fn no_research_limit_means_no_cap_and_a_set_one_is_not_clamped() {
        assert_eq!(round_cap(0), None);
        assert_eq!(round_cap(7), Some(7));
        assert_eq!(round_cap(200), Some(200));
    }

    #[test]
    fn local_compaction_keeps_rules_brief_latest_evidence_and_tool_pairs() {
        let turn = |role: &str, text: &str| format!("<|im_start|>{role}\n{text}<|im_end|>\n");
        let system = turn("system", "rules and footage digest");
        let brief = turn("user", "make a 60 second script");
        let old = turn("assistant", "old search call") + &turn("user", "Tool result for search: old evidence");
        let latest = turn("assistant", "latest call") + &turn("user", "Tool result for get_video: latest evidence");
        let mut prompt = system.clone() + &brief + &old + &latest;
        assert!(drop_oldest_local_exchange(&mut prompt, "make a 60 second script"));
        assert_eq!(prompt, system + &brief + &latest);
        assert!(!drop_oldest_local_exchange(&mut prompt, "make a 60 second script"));
    }

    /// Five seconds of "Thank you." is a third of a forty-second piece, and every existing check
    /// passed it: the clip is long enough, it ends on a sentence, the footage is real, and the
    /// total came to 39.9 s against 40. Nothing asked whether the words were worth hearing.
    #[test]
    fn a_clip_that_only_says_thank_you_is_reported() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,60)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 1.0, 6.0, 'Thank you.', 0),
                 (1, 10.0, 18.0, 'We moved here in 2021 and we have loved every minute of it.', 0)",
                [],
            )
            .unwrap();

        let beat = |id: &str, in_s: f64, out_s: f64| Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s,
                out_s,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            bed: None,
            notes: None,
            card: None,
        };
        let script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(40.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![beat("thanks", 1.0, 6.0), beat("real", 10.0, 18.0)],
            card_style: None,
        };

        let issues = empty_speech_issues(&db, &script);
        assert_eq!(issues.len(), 1, "only the acknowledgement: {issues:?}");
        assert_eq!(issues[0].beat_id.as_deref(), Some("thanks"));
        assert!(issues[0].message.contains("Thank you."), "{}", issues[0].message);
        assert!(issues[0].message.contains("acknowledgement"));
    }

    #[test]
    fn a_clip_under_a_bed_is_not_judged_on_words_nobody_hears() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,60)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (1, 1.0, 6.0, 'Okay.', 0)",
                [],
            )
            .unwrap();
        let script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(40.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![Beat {
                id: "bedded".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                // Muted pictures under somebody else's voice: what they say is irrelevant.
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 1.0,
                    out_s: 6.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 18.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                notes: None,
                card: None,
            }],
            card_style: None,
        };
        assert!(empty_speech_issues(&db, &script).is_empty());
    }

    /// One `CliAgentConfig` serves two jobs with very different clocks: describing a frame takes
    /// seconds, writing a script takes minutes — agy needs about nine on a 96-video project. At
    /// the 180 s default every chat turn was killed and reported as "agy timed out after 180s".
    #[tokio::test]
    async fn a_cli_brain_writing_a_script_gets_a_script_length_clock() {
        let cli = crate::config::CliAgentConfig { tool: "agy".into(), timeout_secs: 180, ..Default::default() };
        let setup = crate::runtime::VisionSetup::Cli(cli);

        let backend = ChatBackend::from_vision_setup(&setup, 1800).await.unwrap();
        let ChatBackend::Cli(agent) = backend else { panic!("expected a CLI backend") };
        assert_eq!(agent.timeout_secs(), 1800, "the frame clock must not govern a script");

        // A setting longer than the script default is the operator's choice and is kept.
        let patient = crate::config::CliAgentConfig { tool: "agy".into(), timeout_secs: 3600, ..Default::default() };
        let backend = ChatBackend::from_vision_setup(&crate::runtime::VisionSetup::Cli(patient), 1800).await.unwrap();
        let ChatBackend::Cli(agent) = backend else { panic!("expected a CLI backend") };
        assert_eq!(agent.timeout_secs(), 3600);
    }

    /// A speech-heavy cut far over target loses whole quotes, not seconds off each one.
    ///
    /// Picture scaling cannot touch it — `speech_factor` is 1.0 on purpose — so before this a
    /// local model's seven-beat draft stayed at 179.6 s against a 40 s target and the only thing
    /// that happened was a warning.
    #[test]
    fn a_cut_made_of_speech_loses_beats_rather_than_sentences() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        let cfg = crate::config::ScriptConfig::default();

        let beat = |id: &str, secs: f64| Beat {
            id: id.into(),
            purpose: format!("beat {id}"),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: secs,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            bed: None,
            notes: None,
            card: None,
        };
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(40.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![beat("open", 25.0), beat("mid-1", 25.0), beat("mid-2", 25.0), beat("close", 25.0)],
            card_style: None,
        };

        assert!(drop_beats_to_target(&db, &mut script, &cfg), "100 s against 40 s must lose something");

        let ids: Vec<&str> = script.beats.iter().map(|b| b.id.as_str()).collect();
        // The hook and the ending are the two positions a viewer notices; middles go from the back.
        assert_eq!(ids.first(), Some(&"open"));
        assert_eq!(ids.last(), Some(&"close"));
        assert!(!ids.contains(&"mid-2"), "the last middle goes first: {ids:?}");
        assert!(script.total_duration_s() <= 40.0 * cfg.target_overshoot);
    }

    /// The 9B's six beats against 60 s (77.4 s, from cut #188): dropping stopped as soon as the cut
    /// was under the 75 s ceiling, after the 3 s "Thank you", and left it at 74.4 s. The next
    /// middle brings it to 61.8 s, nearer, so it goes too. A cut just over keeps its quotes.
    #[test]
    fn dropping_continues_while_it_brings_the_cut_nearer_its_target() {
        use crate::script::Script;
        let db = Db::open_in_memory().unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let cut = |lens: &[f64], target: f64| -> Script {
            let beats: Vec<_> = lens
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    serde_json::json!({ "id": format!("b{i}"), "purpose": "p", "narration": "",
                        "clips": [{ "video_id": 1, "in_s": 0.0, "out_s": l, "audio": "source" }] })
                })
                .collect();
            serde_json::from_value(serde_json::json!({ "title": "t", "target_duration_s": target, "beats": beats }))
                .unwrap()
        };
        let mut s = cut(&[8.2, 15.3, 29.6, 12.6, 3.0, 8.7], 60.0);
        assert!(drop_beats_to_target(&db, &mut s, &cfg));
        let ids: Vec<&str> = s.beats.iter().map(|b| b.id.as_str()).collect();
        assert_eq!(ids, ["b0", "b1", "b2", "b5"], "{ids:?}");
        assert!((s.total_duration_s() - 61.8).abs() < 0.01, "{}", s.total_duration_s());
        // 1.5 s over a 60 s target: losing a 12 s quote lands further away.
        let mut s = cut(&[20.0, 12.0, 20.0, 9.5], 60.0);
        let before = s.total_duration_s();
        assert!(before > 60.0 * cfg.final_target_tolerance, "{before}");
        assert!(!drop_beats_to_target(&db, &mut s, &cfg));
        assert_eq!(s.beats.len(), 4);
    }

    #[test]
    fn dropping_stops_before_it_stops_being_the_script() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let beat = |id: &str| Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 90.0,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            bed: None,
            notes: None,
            card: None,
        };
        // Two beats of 90 s against a 10 s target: hopeless, and dropping to one would not be
        // the script any more. The score reports the overrun instead.
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(10.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![beat("open"), beat("close")],
            card_style: None,
        };
        assert!(!drop_beats_to_target(&db, &mut script, &cfg));
        assert_eq!(script.beats.len(), 2);
    }

    /// Default script settings for tests.
    fn sc() -> crate::config::ScriptConfig {
        crate::config::ScriptConfig::default()
    }

    use crate::projects::NewProject;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    /// What someone says is never shortened to reach a length. A cut made only of interview
    /// keeps its sentences whole and runs as long as they do.
    #[test]
    fn what_people_say_is_never_trimmed_to_hit_the_target() {
        use crate::script::{Audio, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 600.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        // Speech throughout: every clip is someone talking, and whole sentences end on the second.
        for i in 0..80 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, ?1, ?2, 'a sentence')",
                    params![i as f64, (i + 1) as f64],
                )
                .unwrap();
        }
        let clip = |in_s: f64, out_s: f64| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s,
            out_s,
            audio: Audio::Source,
            why: None,
            tracks: None,
        };
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(60.0),
            fps: Default::default(),
            width: None,
            height: None,
            // 75 s of interview and no pictures at all.
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: Some(String::new()),
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![clip(0.0, 25.0), clip(25.0, 50.0), clip(50.0, 75.0)],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        let before = script.total_duration_s();
        // Nothing here is a picture, so there is nothing that may be trimmed: the fit leaves it.
        fit_to_target(&db, &mut script, &sc());
        let total = script.total_duration_s();
        assert!((total - before).abs() < 1e-9, "speech kept whole: {before} -> {total}");
        assert!(
            script.beats[0].clips.iter().all(|c| (c.out_s - c.in_s - 25.0).abs() < 1e-9),
            "every clip is the length its sentences are"
        );
    }

    /// The cut used to end on two seconds of the speaker still moving his hands with the sound
    /// already finished. The editor: it closes "without an image that represents what we were
    /// talking about".
    #[test]
    fn the_cut_closes_on_a_picture_rather_than_a_talking_head() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(project.id, tmp.path(), true).unwrap();
        for (id, hash) in [(1, 'h'), (2, 'i'), (3, 'j')] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 300.0)",
                    params![id, hash.to_string()],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, format!("{id}.mp4")],
                )
                .unwrap();
        }
        db.conn
            .execute("INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 20.0, 'everyone here is on each other side in this community')", [])
            .unwrap();
        // #3 is somebody else being interviewed: described, but a talking head all the same.
        db.conn
            .execute("INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (3, 0.0, 9.0, 'and then I started the business')", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES
                 (2, 5.0, '{\"description\":\"a wide shot of the community gathered in the lobby\"}'),
                 (3, 1.0, '{\"description\":\"a man with a lav mic on his collar, mid sentence\"}')",
                [],
            )
            .unwrap();
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 20.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 20.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        let held = hold_the_last_picture(&db, project.id, &mut script, &sc());
        assert!(held > 0.0, "there is a hold to add");
        let clips = &script.beats[0].clips;
        assert_eq!(clips.len(), 2, "the hold is a new picture, not more of the same one");
        assert_eq!(clips[1].video_id, 2, "not #3 — a lav mic mid-sentence is another talking head");
        assert_eq!(clips[1].audio, Audio::Mute);
        assert_eq!(clips[0].out_s, 20.0, "the speaker's own picture is left where it was");

        // And a second repair neither adds another nor leaves the first behind.
        assert_eq!(hold_the_last_picture(&db, project.id, &mut script, &sc()), 0.0, "held twice");
        assert!(drop_closing_picture(&mut script), "the repair can take back what it added");
        assert_eq!(script.beats[0].clips.len(), 1);
    }

    /// The Greet Mag tape, verbatim: she finishes, both say "Yeah", and the interviewer starts a
    /// story of their own. That last line is ten words of plain English — no word count and no
    /// off-mic flag reaches it (Jev scores it 0.20). The pair of "Yeah"s in front of it does.
    /// Refining judged "says what was asked" against the refine instruction rather than the
    /// brief, so the same script read 67 in the turn and 76 against what was actually asked for —
    /// and the loop that climbs that score was climbing the wrong number.
    #[test]
    fn a_cut_is_judged_against_the_brief_not_against_the_last_thing_typed() {
        let db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        let session = create_session(&db, project.id, "s").unwrap();
        let t = now();
        for (role, content) in [
            ("user", "Make a 40 second piece about the Northwest Hills neighbourhood."),
            ("assistant", "Drafted it."),
            ("user", "Improve this cut. Change something substantial this time."),
        ] {
            db.conn
                .execute(
                    "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
                     VALUES (?1, ?2, ?3, NULL, ?4)",
                    params![session, role, content, t],
                )
                .unwrap();
        }
        let brief = session_brief(&db, session).expect("the session opens with the brief");
        assert!(brief.contains("Northwest Hills"), "got {brief:?}");
        assert!(!brief.contains("Improve this cut"), "the instruction is not the brief: {brief:?}");
        // A session with nothing in it yet falls back to the caller's message.
        assert_eq!(session_brief(&db, 999_999), None);
    }

    #[test]
    fn a_beat_ends_where_the_turn_does_not_where_the_reply_starts() {
        use crate::script::{AudioBed, Beat};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (100,'h',1,200.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (100, 72.0, 77.0, 'And it feels like you are in more of a small town neighborhood and we just love it.'),
                 (100, 77.0, 79.0, 'And we love the deer running around too.'),
                 (100, 79.0, 80.0, 'So, yeah, it is beautiful.'),
                 (100, 80.0, 80.0, 'Yeah.'),
                 (100, 80.0, 81.0, 'Yeah.'),
                 (100, 81.0, 82.0, 'I used to live up on like Spicewood Springs road.')",
                [],
            )
            .unwrap();
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 100,
                    in_s: 72.0,
                    out_s: 82.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        assert_eq!(end_on_turns(&db, &mut script, &sc()), 1, "the handover is there to be found");
        let bed = script.beats[0].bed.clone().unwrap();
        assert!(
            (bed.out_s - 80.0).abs() < 0.05,
            "keeps what she said and drops the reply over the top, got {:.1}",
            bed.out_s
        );
    }

    /// Two "Yeah"s in the middle of somebody's own answer are them thinking. Trimming there would
    /// throw away the answer to keep its opening.
    #[test]
    fn a_pause_inside_an_answer_is_not_a_handover() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (7,'h7',1,200.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (7, 0.0, 4.0, 'We moved here in 2021 and we have loved every minute of it.'),
                 (7, 4.0, 4.5, 'Yeah.'),
                 (7, 4.5, 5.0, 'Yeah.'),
                 (7, 5.0, 9.0, 'And the thing that keeps us here is how easily people talk to each other.'),
                 (7, 9.0, 13.0, 'That is rarer than it sounds, and it is why we are staying put.')",
                [],
            )
            .unwrap();
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 7,
                    in_s: 0.0,
                    out_s: 13.0,
                    audio: Audio::Source,
                    why: None,
                    tracks: None,
                }],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        assert_eq!(end_on_turns(&db, &mut script, &sc()), 0, "kept: {:?}", script.beats[0].clips[0]);
    }

    /// The fit that runs last can trim a speech clip far past what snapping reaches, leaving a
    /// speaker cut off mid-thought — four clips in one run, one of them mid-word.
    #[test]
    fn a_trimmed_speech_clip_is_put_back_on_a_whole_sentence() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 300.0)", [])
            .unwrap();
        // Two sentences: 10–20 s and 20–31 s.
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (1, 10.0, 20.0, 'the first whole sentence'), (1, 20.0, 31.0, 'the second whole sentence')",
                [],
            )
            .unwrap();
        let beat = |in_s: f64, out_s: f64| Beat {
            id: "b1".into(),
            purpose: "p".into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            notes: None,
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s,
                out_s,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            bed: None,
            card: None,
        };
        let script_of = |b: Beat| Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![b],
            card_style: None,
        };

        // Cut 2.6 s early: within reach, so it runs on to the end of the sentence.
        let mut s = script_of(beat(10.0, 28.4));
        assert_eq!(end_on_sentences(&db, &mut s, &sc()), 1);
        assert!((s.beats[0].clips[0].out_s - 31.35).abs() < 0.01, "{}", s.beats[0].clips[0].out_s);

        // Cut so early that finishing would add more than max_speech_extend_s: fall back to where
        // the previous sentence ended rather than stop in the middle of this one.
        let mut cfg = sc();
        cfg.max_speech_extend_s = 2.0;
        let mut s = script_of(beat(10.0, 22.0));
        assert_eq!(end_on_sentences(&db, &mut s, &cfg), 1);
        assert!((s.beats[0].clips[0].out_s - 20.35).abs() < 0.01, "{}", s.beats[0].clips[0].out_s);

        // A clip sitting exactly on a boundary gains the overrun and nothing else: whisper's
        // times are approximate, so cutting on the number clips the last word — it is how
        // "…the deer running around too" and "Genuine, real neighborhood." lost their endings.
        let mut s = script_of(beat(10.0, 31.0));
        assert_eq!(end_on_sentences(&db, &mut s, &sc()), 1);
        assert!(
            (s.beats[0].clips[0].out_s - 31.35).abs() < 0.01,
            "the last word keeps its decay: {}",
            s.beats[0].clips[0].out_s
        );

        // And b-roll is not speech: nothing to keep whole.
        let mut s = script_of(beat(10.0, 28.4));
        s.beats[0].clips[0].audio = Audio::Mute;
        assert_eq!(end_on_sentences(&db, &mut s, &sc()), 0);
    }

    /// Every round re-sends the whole conversation, so a model that keeps looking fills the
    /// window. Bonsai 2 made 199 tool calls and the turn died on a raw "exceeds the available
    /// context size" with nothing drafted.
    #[test]
    fn a_long_search_forgets_its_oldest_results_rather_than_overflowing() {
        let msg = |role: &str, text: &str| json!({ "role": role, "content": text });
        let mut messages = vec![msg("system", &"rules ".repeat(200)), msg("user", "make me a teaser")];
        for i in 0..40 {
            messages.push(msg("assistant", "calling a tool"));
            messages.push(msg("tool", &format!("result {i} {}", "keyframe description ".repeat(100))));
        }
        let before = approx_tokens(&messages);
        assert!(make_room(&mut messages, before / 2), "room was made");
        assert!(approx_tokens(&messages) <= before / 2);
        // The brief and the rules survive; the oldest results are the ones forgotten.
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "make me a teaser");
        let kept: Vec<&str> = messages.iter().filter_map(|m| m["content"].as_str()).collect();
        assert!(!kept.iter().any(|c| c.starts_with("result 0 ")), "oldest went first");
        assert!(kept.iter().any(|c| c.starts_with("result 39 ")), "newest stayed");

        // Nothing left to forget: the caller is told to stop looking and draft.
        let mut only_prompt = vec![msg("system", &"rules ".repeat(500)), msg("user", "hello")];
        assert!(!make_room(&mut only_prompt, 10));
    }

    /// Trimming must not break the tool protocol: an assistant message with `tool_calls` whose
    /// responses are gone is rejected outright (400) by OpenAI-compatible servers — which would
    /// kill the turn exactly when `make_room` exists to prevent. Before, it removed the tool
    /// messages; now their content is stubbed and every tool_call keeps a response.
    #[test]
    fn making_room_stubs_tool_results_and_keeps_every_tool_call_answered() {
        let mut messages = vec![
            json!({ "role": "system", "content": "rules rules rules" }),
            json!({ "role": "user", "content": "make me a teaser" }),
        ];
        for i in 0..4 {
            messages.push(json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "id": format!("call_{i}"), "type": "function",
                    "function": { "name": "search", "arguments": "{}" } }]
            }));
            messages.push(json!({
                "role": "tool", "tool_call_id": format!("call_{i}"),
                "content": format!("result {i} {}", "keyframe description ".repeat(50))
            }));
        }
        let before = approx_tokens(&messages);
        assert!(make_room(&mut messages, before / 2), "room was made");
        assert!(approx_tokens(&messages) <= before / 2);
        // Every assistant tool_call still has its response: the pairing survives.
        let answered: std::collections::HashSet<String> = messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["tool_call_id"].as_str().map(str::to_string))
            .collect();
        for m in messages.iter().filter(|m| m["role"] == "assistant") {
            for tc in m["tool_calls"].as_array().cloned().unwrap_or_default() {
                let id = tc["id"].as_str().expect("tool_call id");
                assert!(answered.contains(id), "tool_call {id} lost its response");
            }
        }
        // The oldest results are the stubs; the newest kept its text.
        let stubbed = messages.iter().filter(|m| m["content"].as_str() == Some(DROPPED_RESULT)).count();
        assert!(stubbed >= 1, "oldest were stubbed: {stubbed}");
        assert!(
            messages.iter().any(|m| m["content"].as_str().is_some_and(|c| c.starts_with("result 3"))),
            "the newest result kept its text"
        );
        // Everything already stubbed and still over budget: nothing more to shrink.
        assert!(!make_room(&mut messages, 1), "nothing left to shrink");
        // And an already-stubbed result is not stubbed twice.
        let mut already = vec![
            json!({ "role": "system", "content": "rules" }),
            json!({ "role": "user", "content": "go" }),
            json!({ "role": "assistant", "content": "", "tool_calls": [{ "id": "c", "type": "function",
                "function": { "name": "search", "arguments": "{}" } }] }),
            json!({ "role": "tool", "tool_call_id": "c", "content": DROPPED_RESULT }),
        ];
        let snapshot = already.clone();
        assert!(!make_room(&mut already, 1), "nothing left to shrink");
        assert_eq!(already, snapshot, "an already-stubbed result is left alone");
    }

    /// Arguments count too (`approx_tokens` reads `tool_calls`): a `save_script` call carries a
    /// whole script and can overflow on its own. Once every tool result is stubbed, the oldest
    /// assistant arguments are replaced with a short valid-JSON placeholder — keeping `id`,
    /// `type` and `function.name` so the pairing stays valid — and the newest assistant message
    /// keeps its arguments whole.
    #[test]
    fn making_room_stubs_old_assistant_arguments_once_the_results_are_gone() {
        let mut messages = vec![
            json!({ "role": "system", "content": "rules" }),
            json!({ "role": "user", "content": "make me a teaser" }),
        ];
        for i in 0..3 {
            let args = if i == 0 { json!({ "script": "x".repeat(4000) }).to_string() } else { "{}".into() };
            messages.push(json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "id": format!("call_{i}"), "type": "function",
                    "function": { "name": "save_script", "arguments": args } }]
            }));
            messages.push(json!({
                "role": "tool", "tool_call_id": format!("call_{i}"),
                "content": DROPPED_RESULT
            }));
        }
        let before = approx_tokens(&messages);
        assert!(make_room(&mut messages, before / 2), "the arguments gave room");
        assert!(approx_tokens(&messages) <= before / 2);
        // The pairing survives: every call keeps its id, type and function name, and its
        // arguments are still valid JSON.
        for m in messages.iter().filter(|m| m["role"] == "assistant") {
            for tc in m["tool_calls"].as_array().cloned().unwrap_or_default() {
                assert_eq!(tc["type"], "function", "type kept");
                assert!(tc["id"].as_str().is_some_and(|id| id.starts_with("call_")), "id kept");
                assert_eq!(tc["function"]["name"], "save_script", "name kept");
                serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().expect("arguments are a string"))
                    .expect("arguments are still valid JSON");
            }
        }
        // The oldest call's script payload went; the newest assistant message kept its arguments.
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"].as_str(),
            Some(DROPPED_ARGS),
            "the oldest arguments were stubbed"
        );
        assert_eq!(
            messages[6]["tool_calls"][0]["function"]["arguments"],
            json!("{}"),
            "the newest assistant message keeps its arguments"
        );
        // Nothing left at all: the caller is told to stop looking and draft.
        assert!(!make_room(&mut messages, 1), "nothing left to shrink");
    }

    /// A drain must not count a torn, still-unterminated last line as seen: the writer is mid-
    /// line, and once it is consumed it never comes back. The torn line stays out until it is
    /// completed, then the next drain delivers it.
    #[test]
    fn draining_mcp_events_leaves_a_torn_last_line_for_the_next_drain() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let path = file.path();
        std::fs::write(path, "{\"kind\":\"tool_started\",\"tool\":\"search\",\"args\":{}}\n")
            .expect("first complete line");
        let mut seen = 0usize;
        let mut events: Vec<ChatEvent> = Vec::new();
        let mut records: Vec<ToolCallRecord> = Vec::new();
        let mut saved = None;
        drain_mcp_events(path, &mut seen, &mut |e| events.push(e), &mut records, &|_| {}, &mut saved);
        assert_eq!(seen, 1, "the complete line was consumed");

        // The writer starts the next line but has not finished it.
        std::fs::write(
            path,
            "{\"kind\":\"tool_started\",\"tool\":\"search\",\"args\":{}}\n{\"kind\":\"tool_finished\",\"tool\":\"search\",\"su",
        )
        .expect("torn second line");
        drain_mcp_events(path, &mut seen, &mut |e| events.push(e), &mut records, &|_| {}, &mut saved);
        assert_eq!(seen, 1, "the torn line was not counted as seen");

        // The writer finishes the line; the next drain sees it exactly once.
        std::fs::write(
            path,
            "{\"kind\":\"tool_started\",\"tool\":\"search\",\"args\":{}}\n{\"kind\":\"tool_finished\",\"tool\":\"search\",\"summary\":\"found 2\",\"script_id\":7}\n",
        )
        .expect("second line completed");
        drain_mcp_events(path, &mut seen, &mut |e| events.push(e), &mut records, &|_| {}, &mut saved);
        assert_eq!(seen, 2);
        assert_eq!(
            events,
            vec![
                ChatEvent::ToolStarted { tool: "search".into(), args: json!({}) },
                ChatEvent::ToolFinished { tool: "search".into(), summary: "found 2".into() },
            ],
            "the completed line was delivered exactly once"
        );
        assert_eq!(records.len(), 1, "the tool_finished record was kept");
        assert_eq!(saved, Some(7));
    }

    /// The model never hears about a repair in the turn that caused it, so the note rides on the
    /// assistant message the next turn replays. Errors are the model's problem to fix and are
    /// reported separately; this is only what was already done for it.
    /// What was asked is written down before the work starts. Everything used to be stored when
    /// the turn finished, so a turn that died took the question with it and a chat reopened
    /// mid-turn looked empty.
    #[tokio::test]
    async fn a_failed_turn_still_records_what_was_asked() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();

        let mut ctx = ChatContext {
            db,
            paths: crate::paths::Paths::in_dir(tmp.path()),
            // Nothing is listening here, so the turn fails as soon as it tries to think.
            backend: ChatBackend::Server {
                url: "http://127.0.0.1:1".into(),
                model: "none".into(),
                api_key: String::new(),
                ctx_tokens: 8192,
            },
            embedder: None,
            system_prompt: None,
            max_tool_rounds: 0,
            script: sc(),
            jev: Default::default(),
            cancel: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "a 40 second teaser", &[], None, &mut |_| {}).await;
        assert!(res.is_err(), "the turn fails without a server");

        let asked: String = ctx
            .db
            .conn
            .query_row("SELECT content FROM chat_messages WHERE role = 'user'", [], |r| r.get(0))
            .expect("the question was written down before the work started");
        assert_eq!(asked, "a 40 second teaser");
    }

    #[test]
    fn repairs_are_written_down_for_the_next_turn() {
        let issue = |sev: IssueSeverity, msg: &str| Issue {
            fixes: Vec::new(),
            severity: sev,
            beat_id: None,
            clip_index: None,
            message: msg.into(),
        };
        assert!(repair_note(&[]).is_none());
        assert!(repair_note(&[issue(IssueSeverity::Error, "unable to assemble a script")]).is_none());

        let note = repair_note(&[
            issue(IssueSeverity::Info, "let the speaker's voice run under the pictures in 2 beat(s)"),
            issue(IssueSeverity::Error, "ignored"),
            issue(IssueSeverity::Warning, "moved clip off a shaky stretch"),
        ])
        .expect("a note");
        assert!(note.starts_with("[applied after drafting"));
        assert!(note.contains("voice run under the pictures"));
        assert!(note.contains("shaky stretch"));
        assert!(!note.contains("ignored"), "an error is not a repair: {note}");
    }

    /// Fitting trims the pictures after the beds are laid. A bed left at its old length plays
    /// on over the next beat, and the export puts two clips on top of each other on A1.
    #[test]
    fn a_bed_is_cut_back_when_its_pictures_are() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        // No transcript: nothing to cut into, so the bed is simply trimmed to its pictures.
        let db = Db::open_in_memory().unwrap();
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 6.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 30.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        clamp_beds_to_beats(&db, &mut script, &sc());
        let bed = script.beats[0].bed.clone().unwrap();
        assert!((bed.duration_s() - 6.0).abs() < 1e-9, "bed matches its pictures: {}", bed.duration_s());
    }

    /// One model pasted the quote it had just found into the id; another left every id empty.
    #[test]
    fn unusable_beat_ids_are_named_after_their_purpose() {
        use crate::script::Beat;
        let beat = |id: &str, purpose: &str| Beat {
            id: id.into(),
            purpose: purpose.into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            notes: None,
            clips: vec![],
            bed: None,
            card: None,
        };
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![
                beat(
                    "So was that Northwest Hills where you grew up? I grew up in Northwest Hills.",
                    "Establish the deep roots",
                ),
                beat("", "Establish the deep roots"),
                beat("keep-me", "Something else"),
            ],
            card_style: None,
        };
        let issues = tidy_beat_ids(&mut script);
        assert_eq!(script.beats[0].id, "establish-the-deep-roots");
        assert_eq!(script.beats[1].id, "establish-the-deep-roots-2", "duplicates are numbered");
        assert_eq!(script.beats[2].id, "keep-me", "a usable id is left alone");
        assert!(issues.iter().any(|i| i.message.contains("named 2 beat(s)")));
    }

    /// A beat that opens on a speaker and cuts away used to go silent at the cut. The voice now
    /// runs on underneath, and the pictures are muted so nobody is heard twice.
    #[test]
    fn a_speakers_voice_runs_under_the_pictures_that_follow() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (2, 'i', 1, 60.0)", []).unwrap();
        // One person talking in whole sentences from 10 s to 30 s of video 1.
        for i in 0..10 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                     VALUES (1, ?1, ?2, 'a whole sentence', 0)",
                    params![10.0 + i as f64 * 2.0, 12.0 + i as f64 * 2.0],
                )
                .unwrap();
        }
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // Her face for 4 s, then two pictures of what she is describing.
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 10.0,
                        out_s: 14.0,
                        audio: Audio::Source,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 5.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 20.0,
                        out_s: 25.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        let issues = lay_audio_beds(&db, &mut script, &sc());
        let bed = script.beats[0].bed.clone().expect("a bed was laid");
        assert_eq!(bed.video_id, 1);
        assert!(bed.inferred);
        // It starts where she starts and runs across the whole beat, not just her own clip.
        assert!((bed.in_s - 10.0).abs() < 1e-9, "bed starts with her: {}", bed.in_s);
        assert!(bed.duration_s() > 13.0, "bed covers the cutaways too: {}", bed.duration_s());
        assert!(script.beats[0].clips.iter().all(|c| c.audio == Audio::Mute), "pictures play silent");
        assert!(issues.iter().any(|i| i.message.contains("run under the pictures")));

        // When the second picture is somebody else talking, it is not a cutaway: muted under her
        // voice, his lips would move to her words. No bed; every clip keeps its own sound.
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                 VALUES (2, 20.0, 25.0, 'and I think the events are great', 0)",
                [],
            )
            .unwrap();
        let beat = &mut script.beats[0];
        beat.bed = None;
        beat.clips[0].audio = Audio::Source;
        beat.clips[2].audio = Audio::Source;
        lay_audio_beds(&db, &mut script, &sc());
        assert!(script.beats[0].bed.is_none(), "no voice over a talking head");
        assert_eq!(script.beats[0].clips[2].audio, Audio::Source);
        // And a bed written that way is flagged.
        let mut bedded = script.clone();
        bedded.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: 1,
            in_s: 10.0,
            out_s: 24.0,
            why: None,
            inferred: false,
            tracks: None,
            gain_db: None,
        });
        let issues = crate::script_check::check(&db, project_id_of(&db), &bedded, &sc(), &Default::default()).unwrap();
        assert!(issues.iter().any(|i| i.clip_index == Some(2) && i.message.contains("their lips move")), "{issues:?}");
    }

    fn project_id_of(db: &Db) -> i64 {
        db.conn.query_row("SELECT id FROM projects LIMIT 1", [], |r| r.get(0)).unwrap_or(0)
    }

    /// The bed has to stay in step with the face: when the picture of the speaker comes second,
    /// the sound starts early so their lips still match when we cut to them.
    #[test]
    fn a_bed_starts_early_when_the_face_comes_after_the_picture() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (2, 'i', 1, 60.0)", []).unwrap();
        for i in 0..10 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                     VALUES (1, ?1, ?2, 'a whole sentence', 0)",
                    params![10.0 + i as f64 * 2.0, 12.0 + i as f64 * 2.0],
                )
                .unwrap();
        }
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 3.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 14.0,
                        out_s: 20.0,
                        audio: Audio::Source,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        lay_audio_beds(&db, &mut script, &sc());
        let bed = script.beats[0].bed.clone().expect("a bed was laid");
        // Her clip sits 3 s into the beat, so her audio starts 3 s before her own in-point.
        assert!((bed.in_s - 11.0).abs() < 1e-9, "sound leads the picture by the cutaway: {}", bed.in_s);
    }

    /// A bed the editor asked for is checked. One with nobody speaking in it is thrown away
    /// rather than rendered as silence under the pictures.
    #[test]
    fn a_bed_over_silence_is_dropped() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 5.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 40.0,
                    out_s: 50.0,
                    why: None,
                    inferred: false,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        let issues = lay_audio_beds(&db, &mut script, &sc());
        assert!(script.beats[0].bed.is_none(), "a bed with no speech in it is dropped");
        assert!(issues.iter().any(|i| i.message.contains("nobody speaks")));
    }

    /// A draft that opens on the interviewer is left for the finishing chain's span rule, which
    /// trims it to the answer. The draft stage used to slide the whole clip forward to the next
    /// line on the microphone, keeping its length, so it played on past the answer into whatever
    /// came next.
    #[test]
    fn a_draft_opening_on_the_interviewer_is_trimmed_not_slid() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 6.5, 8.5, 'What made you want to sponsor the bar?', 1),
                 (1, 9.0, 14.0, 'We love this hotel and the people who come here.', 0),
                 (1, 14.5, 17.0, 'And what about the music, then?', 1)",
                [],
            )
            .unwrap();
        let mut s: Script = serde_json::from_value(json!({ "title": "t", "beats": [{ "id": "b1", "purpose": "p",
            "narration": "", "clips": [{ "video_id": 1, "in_s": 6.5, "out_s": 14.0, "audio": "source" }] }] }))
        .unwrap();
        let mut g = Grounding::default();
        g.add(1, 0.0, 60.0);
        enforce_grounding_and_pacing(&db, p.id, &mut s, &g, false, &sc());
        let c = &s.beats[0].clips[0];
        assert_eq!((c.in_s, c.out_s), (6.5, 14.0), "the draft keeps its edges for the finish to judge");
        repair::finish_script(&db, p.id, &mut s, false, &sc()).unwrap().expect("a script");
        let c = &s.beats[0].clips[0];
        assert!((8.5..=9.0).contains(&c.in_s), "opens on the answer: {c:?}");
        assert!(c.out_s < 14.5, "and stops before the next question: {c:?}");
    }

    #[test]
    fn a_clip_on_a_shaky_stretch_is_moved_to_the_steady_part_of_the_shot() {
        use crate::script::{Audio, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 18.6)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn.execute("INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 9.0, '{}')", []).unwrap();
        // Measured like the clip that kept getting picked: violent for eight seconds, then steady.
        let w = |a: f64, b: f64, jerk: f64, sway: f64| crate::steadiness::Window {
            start_s: a,
            end_s: b,
            jerk,
            motion: 0.5,
            sway,
        };
        db.set_motion_windows(
            1,
            &[
                w(0.0, 4.0, 3.25, 11.1),
                w(4.0, 8.0, 0.77, 2.46),
                w(8.0, 12.0, 0.18, 0.02),
                w(12.0, 16.0, 0.11, 0.02),
                w(16.0, 18.6, 0.14, 0.03),
            ],
        )
        .unwrap();
        let mut grounding = Grounding::default();
        grounding.add(1, 0.0, 18.6);
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: Some("some words".into()),
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 0.5,
                    out_s: 4.7,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        // A finishing rule now, the same for every editor's cut (`finish::draft`).
        let _ = &grounding;
        let issues = crate::finish::draft::steady_camera(&db, &mut script, &sc());
        let c = &script.beats[0].clips[0];
        assert!(
            (c.in_s - 8.0).abs() < 1e-9 && (c.out_s - 12.2).abs() < 1e-9,
            "moved to the steady run: {c:?} {issues:?}"
        );
        assert!(issues.iter().any(|i| i.contains("moved clip off a shaky stretch")), "{issues:?}");
    }

    /// The three actions a turn can take. A reply carries words and no script, which is what lets
    /// the editor ask a question instead of drafting around a guess.
    #[cfg(unix)]
    #[tokio::test]
    async fn local_planning_replies_do_not_spend_the_research_budget() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("Research test")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        for id in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id,content_hash,size,duration_s) VALUES (?1,?2,100,10)",
                    params![id, format!("research-{id}")],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id,folder_id,path,size,mtime,last_seen) VALUES (?1,?2,?3,100,0,0)",
                    params![id, folder.id, tmp.path().join(format!("{id}.mp4")).to_str().unwrap()],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO frames(video_id,t_s,description_json) VALUES (?1,2.5,?2)",
                    params![id, r#"{"description":"Neighbors meeting on a residential street."}"#],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id,start_s,end_s,text) VALUES (?1,0,5,?2)",
                    params![id, "This neighborhood brings people together and makes everyone feel at home."],
                )
                .unwrap();
        }
        let final_action = json!({"action":"final","script":{
            "title":"Community","target_duration_s":5,"beats":[{"id":"b1","purpose":"A community brings people together",
                "clips":[{"video_id":2,"in_s":0,"out_s":5,"audio":"source"}]}]}});
        let helper_path = tmp.path().join("helper");
        let marker_path = tmp.path().join("reviewed");
        let source = format!(
            r#"#!/usr/bin/env python3
import sys,json
final=json.loads({:?})
count=0
print(json.dumps({{'ready':True,'ctx':32768,'token_count':True,'model':'test-model'}}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['cmd']=='count_tokens':
  out={{'tokens':len(r.get('prompt',''))//4}}
 else:
  count+=1
  if count<=2: action={{'action':'reply','text':'I will inspect the footage now.'}}
  elif count==3: action={{'action':'tool','tool':'get_video','args':{{'video_id':1}}}}
  else: action=final
  if 'newly inspected shot details' in r.get('prompt',''):
   open({:?},'w').write('reviewed')
  out={{'content':json.dumps(action),'gen_tokens':10,'generation_seconds':0.1,'model':'test-model'}}
 out.update(id=r['id'],ok=True)
 print(json.dumps(out),flush=True)
"#,
            final_action.to_string(),
            marker_path.to_str().unwrap()
        );
        std::fs::write(&helper_path, source).unwrap();
        std::fs::set_permissions(&helper_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let helper = crate::vision::LocalLlm::start(&crate::vision::LocalModels {
            helper: helper_path,
            vision: Some(("model".into(), Some("projector".into()))),
            embed: None,
            cpu: true,
            runtime: Default::default(),
            concurrency: 1,
            free_ghostpen: Vec::new(),
        })
        .await
        .unwrap();
        let mut ctx = ChatContext {
            db,
            paths: crate::paths::Paths::in_dir(tmp.path()),
            backend: ChatBackend::Local { helper: Box::new(helper), ctx_tokens: 32768, think: false },
            embedder: None,
            system_prompt: None,
            max_tool_rounds: 3,
            script: crate::config::ScriptConfig {
                closing_hold_s: 0.0,
                infer_audio_beds: false,
                speech_tail_s: 0.0,
                speech_lead_s: 0.0,
                speech_overrun_s: 0.0,
                ..sc()
            },
            jev: Default::default(),
            cancel: None,
        };
        let result =
            run_turn(&mut ctx, p.id, None, "Create a 5 second community video", &[], None, &mut |_| {}).await.unwrap();
        assert!(result.script_id.is_some());
        // The calls are the model's own: the harness runs no inspections of its own on a local
        // model's draft that it would not run on any other model's.
        assert_eq!(result.tool_calls.len(), 1, "{:?}", result.tool_calls);
        assert_eq!(result.tool_calls[0].tool, "get_video");
        assert!(!marker_path.exists(), "no extra review round");
    }

    #[test]
    fn local_tool_arguments_use_the_shared_typed_schemas() {
        let schema = local_action_schema();
        let branches = schema["oneOf"].as_array().unwrap();
        for definition in tools::tools_definition().as_array().unwrap() {
            let function = &definition["function"];
            let branch = branches.iter().find(|b| b["properties"]["tool"]["enum"][0] == function["name"]).unwrap();
            // The grammar leaves out the parameters' docs (the prompt has them); the types match.
            fn without_docs(v: &Value) -> Value {
                match v {
                    Value::Object(m) => Value::Object(
                        m.iter()
                            .filter(|(k, v)| !(*k == "description" && v.is_string()))
                            .map(|(k, x)| (k.clone(), without_docs(x)))
                            .collect(),
                    ),
                    Value::Array(a) => Value::Array(a.iter().map(without_docs).collect()),
                    other => other.clone(),
                }
            }
            assert_eq!(branch["properties"]["args"], without_docs(&function["parameters"]));
        }
        let video = branches.iter().find(|b| b["properties"]["tool"]["enum"][0] == "get_video").unwrap();
        assert_eq!(video["properties"]["args"]["properties"]["video_id"]["type"], "integer");
        assert!(video["properties"]["args"]["required"].as_array().unwrap().contains(&json!("video_id")));
    }

    #[test]
    fn a_turn_may_answer_in_words_instead_of_drafting() {
        let reply: LocalAction =
            serde_json::from_str(r#"{"action":"reply","text":"Which three voices do you want?"}"#).unwrap();
        assert!(matches!(reply, LocalAction::Reply { text } if text.contains("voices")));

        // The schema the model is constrained to must actually permit it, or it can never be
        // produced no matter what the rules say.
        let schema = serde_json::to_string(&local_action_schema()).unwrap();
        assert!(schema.contains("reply"), "the action schema must allow a reply");
        let forced = serde_json::to_string(&local_final_action_schema(true)).unwrap();
        assert!(forced.contains("reply"), "even when pushed to finish, it may answer instead");

        // A tool call and a script still parse as before.
        assert!(matches!(
            serde_json::from_str::<LocalAction>(r#"{"action":"tool","tool":"list_videos","args":{}}"#).unwrap(),
            LocalAction::Tool { .. }
        ));
    }

    #[test]
    fn an_unopened_clip_is_verified_against_the_footage_not_thrown_away() {
        use crate::script::{Audio, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 2.0, '{\"description\":\"a hillside\"}')",
                [],
            )
            .unwrap();

        // The model opened nothing, so nothing is grounded by tool results.
        let grounding = Grounding::default();
        let clip = |in_s: f64, out_s: f64| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s,
            out_s,
            audio: Audio::Mute,
            why: None,
            tracks: None,
        };
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: Some("some words about the hillside here".into()),
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // Real footage it never opened, then a range past the end of the video.
                clips: vec![clip(0.0, 5.0), clip(9000.0, 9005.0)],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        let issues = enforce_grounding_and_pacing(&db, p.id, &mut script, &grounding, false, &sc());
        let kept: Vec<(f64, f64)> = script.beats.iter().flat_map(|b| &b.clips).map(|c| (c.in_s, c.out_s)).collect();
        assert_eq!(kept, vec![(0.0, 5.0)], "real footage survives, invented footage does not: {issues:?}");
        assert!(
            issues.iter().any(|i| i.message.contains("kept an unopened clip") && i.message.contains("a hillside")),
            "should report what it checked: {issues:?}"
        );
        assert!(issues.iter().any(|i| i.message.contains("nothing indexed")), "should drop the unreal one: {issues:?}");
    }
    #[test]
    fn local_action_parsing() {
        let tool_json = r#"{"action":"tool","tool":"search","args":{"query":"unbox","limit":5}}"#;
        let action: LocalAction = serde_json::from_str(tool_json).unwrap();
        match action {
            LocalAction::Tool { tool, args } => {
                assert_eq!(tool, "search");
                assert_eq!(args["query"], "unbox");
                assert_eq!(args["limit"], 5);
            }
            _ => panic!("expected Tool action"),
        }

        let final_json = r#"{
            "action": "final",
            "script": {
                "title": "Teaser",
                "beats": [
                    {
                        "id": "b1",
                        "purpose": "hook",
                        "clips": [
                            { "video_id": 1, "in_s": 0.0, "out_s": 3.5 }
                        ]
                    }
                ]
            }
        }"#;
        let action: LocalAction = serde_json::from_str(final_json).unwrap();
        match action {
            LocalAction::Final { script } => {
                assert_eq!(script.title, "Teaser");
                assert_eq!(script.beats.len(), 1);
                assert_eq!(script.beats[0].clips[0].video_id, 1);
            }
            _ => panic!("expected Final action"),
        }
    }

    #[test]
    fn grounding_checks() {
        let mut g = Grounding::default();
        g.add(1, 10.0, 20.0);

        // Within range
        assert!(g.is_grounded(1, 12.0, 18.0, sc().grounding_slack_s));
        // Overlap within 5s slack (e.g. 5.0 to 12.0 overlaps 10.0..20.0)
        assert!(g.is_grounded(1, 6.0, 11.0, sc().grounding_slack_s));
        // Completely outside slack
        assert!(!g.is_grounded(1, 0.0, 4.0, sc().grounding_slack_s));
        // Different video
        assert!(!g.is_grounded(2, 12.0, 18.0, sc().grounding_slack_s));
        // Overlapping but far outside: a whole-window clip touching the hit is not grounded
        assert!(!g.is_grounded(1, 0.0, 45.0, sc().grounding_slack_s));
    }

    #[test]
    fn pacing_redraft_and_trim() {
        use crate::script::{Audio, Beat, ScriptClip};
        let clip = |in_s: f64, out_s: f64| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s,
            out_s,
            audio: Audio::Source,
            why: None,
            tracks: None,
        };
        let mut s = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: Some(20.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                clips: vec![clip(0.0, 45.0), clip(90.0, 144.0), clip(200.0, 206.0)],
                notes: None,
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        let issues = pacing_issues(&s, true, &sc());
        // two over-long clips + total far from target
        assert_eq!(issues.len(), 3);
        assert_eq!(pacing_issues(&s, false, &sc()).len(), 2, "revisions don't enforce the target");

        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 400.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        let mut g = Grounding::default();
        g.add(1, 0.0, 400.0);
        let mut revision = s.clone();
        enforce_grounding_and_pacing(&db, p.id, &mut revision, &g, false, &sc());
        assert!(revision.total_duration_s() > 20.0 * sc().target_overshoot, "revision keeps its length");
        let applied = enforce_grounding_and_pacing(&db, p.id, &mut s, &g, true, &sc());
        assert!(!applied.is_empty());
        assert!(s.beats[0].clips.iter().all(|c| c.out_s - c.in_s <= sc().trimmed_clip_s + 1e-9));
        assert!(s.total_duration_s() <= 20.0 * sc().target_overshoot);
    }

    #[test]
    fn removed_and_reference_videos_leave_the_library() {
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        for (id, name) in [(1, "a.mp4"), (2, "wife-edit.mp4"), (3, "c.mp4")] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 30.0)",
                    params![id, name],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, name],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (2, 0.0, 4.0, 'we love it')",
                [],
            )
            .unwrap();
        db.exclude_video(p.id, 2, "reference").unwrap();
        db.exclude_video(p.id, 3, "removed").unwrap();
        assert!(db.exclude_video(p.id, 1, "bogus").is_err());

        assert!(is_video_in_project(&db, p.id, 1));
        assert!(!is_video_in_project(&db, p.id, 2), "a reference edit is never footage");
        assert!(!is_video_in_project(&db, p.id, 3));
        let st = crate::index::status(&db, Some(p.id)).unwrap();
        assert_eq!(st.videos, 1);

        let text = reference_edits_text(&db, p.id, 6000);
        assert!(text.contains("wife-edit.mp4") && text.contains("we love it"), "{text}");
        assert!(!text.contains("c.mp4"));

        db.include_video(p.id, 2).unwrap();
        assert!(is_video_in_project(&db, p.id, 2));
        assert!(reference_edits_text(&db, p.id, 6000).is_empty());
    }

    #[test]
    fn the_hint_after_empty_searches_points_at_the_tools_that_look() {
        assert!(EMPTY_SEARCH_HINT.contains("list_videos"));
        assert!(EMPTY_SEARCH_HINT.contains("get_video"));
    }

    #[test]
    fn requested_duration_from_message() {
        assert_eq!(requested_duration_s("Create a 60 second promo video"), Some(60.0));
        assert_eq!(requested_duration_s("make it 90s long"), Some(90.0));
        assert_eq!(requested_duration_s("un video de 2 minutos"), Some(120.0));
        assert_eq!(requested_duration_s("about 1.5 minutes, please"), Some(90.0));
        assert_eq!(requested_duration_s("a 60-second teaser"), Some(60.0));
        assert_eq!(requested_duration_s("show the 3 houses"), None);
        assert_eq!(requested_duration_s("leave people talking longer"), None);
    }

    #[test]
    fn the_format_chosen_wins_auto_follows_the_draft_and_a_cut_keeps_its_shape() {
        use crate::clips::ClipFormat;
        let db = Db::open_in_memory().unwrap();
        let mut p = db.create_project(&NewProject::named("P")).unwrap();
        p.width = 3840;
        p.height = 2160;
        let draft = |w: Option<i64>, h: Option<i64>| -> Script {
            let mut s: Script = serde_json::from_value(serde_json::json!({"title": "t", "beats": []})).unwrap();
            s.width = w;
            s.height = h;
            s
        };
        let vertical = draft(Some(1080), Some(1920));
        let none = draft(None, None);
        // Chosen: wins over what the model drafted.
        assert_eq!(turn_shape(&p, Some(ClipFormat::Original), None, &vertical), Some((3840, 2160, false)));
        assert_eq!(turn_shape(&p, Some(ClipFormat::Vertical), None, &none), Some((1080, 1920, true)));
        // Auto: the model's shape, cropped; the project's when it set none.
        assert_eq!(turn_shape(&p, None, None, &vertical), Some((1080, 1920, true)));
        assert_eq!(turn_shape(&p, None, None, &none), Some((3840, 2160, false)));
        // A revision keeps the cut's shape whatever the model writes.
        let landscape = draft(Some(3840), Some(2160));
        assert_eq!(turn_shape(&p, None, Some(&landscape), &vertical), Some((3840, 2160, false)));
        // Back to the footage's own shape: no crop framing left over.
        let mut s = vertical.clone();
        s.framing = Some(crate::framing::Framing::default());
        assert!(apply_shape(&db, &mut s, (3840, 2160, false), &sc()));
        assert_eq!((s.width, s.height, s.framing.is_none()), (Some(3840), Some(2160), true));
    }

    #[test]
    fn a_new_cut_brief_says_its_shape_in_its_header() {
        use crate::clips::ClipFormat;
        assert_eq!(
            requested_format("[Draft \"Reel\" — Target: 30s — Format: 9:16]\nUpbeat"),
            Some(ClipFormat::Vertical)
        );
        assert_eq!(requested_format("[Draft \"Sq\" — Target: 30s — Format: 1:1]"), Some(ClipFormat::Square));
        assert_eq!(requested_format("[Draft \"Old\" — Target: 30s]\nbrief"), None);
        assert_eq!(requested_format("make it Format: 9:16 please"), None, "only a draft header says it");
        assert_eq!(requested_format("[Draft \"A\" — Target: 30s — Format: auto]"), None, "auto: the model decides");
    }

    /// A b-roll chat: the sound is decided, not drafted — ambience or silence as chosen, anyone on
    /// the microphone muted either way, and no narration or bed survives. Its checks are about
    /// pictures: a shot of somebody talking is the thing to fix, missing narration is not.
    #[test]
    fn broll_style_decides_the_sound_and_drops_the_voice() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 400.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 100.0, 110.0, 'We moved here in 2021 and have loved it since.')",
                [],
            )
            .unwrap();
        for t in [20.0, 105.0] {
            db.conn.execute("INSERT INTO frames(video_id, t_s, description_json) VALUES (1, ?1, '{}')", [t]).unwrap();
        }
        let clip = |in_s: f64, out_s: f64, audio| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            video_id: 1,
            in_s,
            out_s,
            audio,
            why: None,
            tracks: None,
            framing: None,
        };
        let script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: None,
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: Some("The hills at dawn.".into()),
                on_screen_text: None,
                title_layout: None,
                clips: vec![clip(22.0, 30.0, Audio::Mute), clip(100.0, 108.0, Audio::Source)],
                notes: None,
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 100.0,
                    out_s: 110.0,
                    why: None,
                    inferred: false,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        let natural = ChatStyle { broll: true, natural_sound: true };
        let mut s = script.clone();
        let notes = settle_broll_audio(&db, &mut s, natural);
        let b = &s.beats[0];
        assert_eq!(b.narration.as_deref(), Some(""));
        assert!(b.bed.is_none());
        assert_eq!(b.clips.len(), 1, "the talking head is gone, not muted");
        assert_eq!(b.clips[0].audio, Audio::Source, "scenery keeps its ambience");
        assert_eq!(notes.len(), 2, "{notes:?}");

        let mut s = script.clone();
        settle_broll_audio(&db, &mut s, ChatStyle { broll: true, natural_sound: false });
        assert!(s.beats[0].clips.iter().all(|c| c.audio == Audio::Mute), "silent means silent");

        // A draft that still has one (a redraft the model is asked to fix) is told about it.
        let cfg = crate::config::ScriptConfig { style: natural, ..sc() };
        let issues = content_issues(&db, &script, &cfg);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].clip_index, Some(1));
        assert!(issues[0].message.contains("speaking"), "{issues:?}");

        // Nothing is padded out to a sentence: the talking clip stays exactly where it was cut.
        let mut s = script.clone();
        repair::finish_script(&db, p.id, &mut s, false, &cfg).unwrap();
        assert_eq!((s.beats[0].clips[1].in_s, s.beats[0].clips[1].out_s), (100.0, 108.0));

        // And the prompt carries the montage's brief after the harness instructions, with the sound
        // rule filled in and the house rules after it.
        let prompt = build_system_prompt(&p, None, Some("custom house rule"), tools::Detail::Short, natural);
        assert!(prompt.contains("B-ROLL MONTAGE"));
        assert!(prompt.contains(crate::tools::INSTRUCTIONS));
        assert!(prompt.find("B-ROLL MONTAGE").unwrap() < prompt.find("custom house rule").unwrap());
        assert!(!prompt.contains("{audio_rule}"));
        assert!(prompt.contains("ambient sound"));
    }

    #[test]
    fn content_checks_narration_and_undescribed_footage() {
        use crate::script::{Audio, Beat, ScriptClip};
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 400.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'a.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 100.0, 110.0, 'We moved here in 2021 and have loved it since.')",
                [],
            )
            .unwrap();
        db.conn.execute("INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 20.0, '{}')", []).unwrap();
        let clip = |in_s: f64, out_s: f64| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id: 1,
            in_s,
            out_s,
            audio: Audio::Source,
            why: None,
            tracks: None,
        };
        let beat = |id: &str, narration: Option<&str>, clips| Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: narration.map(Into::into),
            on_screen_text: None,
            title_layout: None,
            clips,
            notes: None,
            bed: None,
            card: None,
        };
        let mut s = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: None,
            width: None,
            height: None,
            beats: vec![
                // 20 s of scenery with 5 words of narration: too short.
                beat("views", Some("A view of the hills"), vec![clip(22.0, 42.0)]),
                // someone talking, no narration needed
                beat("talk", None, vec![clip(100.0, 110.0)]),
                // nothing known about 300-306 s
                beat(
                    "dead",
                    Some("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"),
                    vec![clip(300.0, 306.0)],
                ),
            ],
            card_style: None,
        };
        let issues = content_issues(&db, &s, &sc());
        let msgs: Vec<_> = issues.iter().map(|i| (i.beat_id.clone().unwrap(), i.clip_index)).collect();
        assert_eq!(msgs, vec![("views".to_string(), None), ("dead".to_string(), Some(0))], "{issues:?}");

        // clip cut right at the end of the words: gains the 1.5 s tail and a short lead-in
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 111.0, 115.0, 'next')",
                [],
            )
            .unwrap();
        // The transcript changed under the stored sentences: as after a new transcription.
        crate::sentence_split::invalidate(&db.conn, 1).unwrap();
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 110.0);
        pad_speech(&db, &mut talk, &sc());
        let c = &talk.beats[1].clips[0];
        assert!((c.in_s - 99.5).abs() < 1e-9, "{c:?}");
        assert!((c.out_s - 110.7).abs() < 1e-9, "stops before the next sentence: {c:?}");
        db.conn.execute("DELETE FROM transcript_segments WHERE text = 'next'", []).unwrap();
        crate::sentence_split::invalidate(&db.conn, 1).unwrap();
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 105.0);
        pad_speech(&db, &mut talk, &sc());
        assert!((talk.beats[1].clips[0].out_s - 111.5).abs() < 1e-9, "finishes the sentence, then holds");

        assert_eq!(mute_clips_without_speech(&db, &mut s), 2);
        assert_eq!(s.beats[1].clips[0].audio, Audio::Source, "speech without narration keeps its audio");
    }

    #[tokio::test]
    async fn tool_dispatch_in_memory() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("TestProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"data").unwrap();

        let c = &db.conn;
        c.execute(
            "INSERT INTO videos(id, content_hash, size, duration_s, fps, width, height, language, summary)
             VALUES (10, 'hash10', 1000, 30.0, 25.0, 1920, 1080, 'en', 'CM5 unboxing overview')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (10, ?1, ?2, 1000, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();

        c.execute(
            "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
             VALUES (10, 0.0, 5.0, 'welcome to cm5 unboxing.'),
                    (10, 5.0, 12.0, 'here is the compute module board.')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO frames(video_id, t_s, description_json)
             VALUES (10, 2.0, '{\"description\": \"Hands holding a green board.\"}'),
                    (10, 8.0, '{\"description\": \"Close-up of the chip.\"}')",
            [],
        )
        .unwrap();

        c.execute(
            "INSERT INTO chunks(video_id, kind, start_s, end_s, text)
             VALUES (10, 'moment', 0.0, 5.0, 'Hands holding a green board. welcome to cm5 unboxing')",
            [],
        )
        .unwrap();

        let mut grounding = Grounding::default();
        let config = crate::config::Config { script: sc(), ..Default::default() };
        let paths = crate::paths::Paths::in_dir(tmp.path());
        let mut embedder = None;
        let mut session = ToolSession::new(crate::chat::LOCAL_TOOL_RESULT_CHARS);
        macro_rules! call {
            ($name:expr, $args:expr) => {{
                let mut tctx = ToolCtx::new(&paths, &config, &mut db, &mut embedder, Origin::ScriptChat)
                    .with_project(p.clone())
                    .with_grounding(&mut grounding);
                session.call(&mut tctx, $name, &$args).await
            }};
        }

        // 1. list_videos
        let called = call!("list_videos", json!({}));
        assert!(called.model_text.len() <= 1500);
        assert!(called.model_text.contains("clip.mp4"));
        assert_eq!(called.summary, "1 videos");

        // 2. get_video
        let called = call!("get_video", json!({"video_id": 10}));
        assert!(called.model_text.len() <= 1500);
        assert!(called.model_text.contains("Hands holding a green board"));
        assert!(called.summary.contains("30s"));
        assert!(grounding.is_grounded(10, 0.0, 30.0, sc().grounding_slack_s));

        // 3. get_transcript
        let called = call!("get_transcript", json!({"video_id": 10, "start_s": 0.0, "end_s": 6.0}));
        assert!(called.model_text.len() <= 1500);
        assert!(called.model_text.contains("welcome to cm5 unboxing"));
        assert_eq!(called.summary, "2 sentences");

        // 4. search (keyword only), asked for by its old name
        let called = call!("search_moments", json!({"query": "unboxing"}));
        assert!(called.model_text.len() <= 1500);
        assert!(called.model_text.contains("unboxing"));
        assert_eq!(called.summary, "1 hits");
        assert_eq!(called.tool, "search", "the old name resolves to the registry's");
        assert!(!called.empty);
        let nothing = call!("search", json!({"query": "zzzz"}));
        assert!(nothing.empty, "an empty search is flagged, not parsed out of its summary");

        // 5. Foreign video rejected
        let called = call!("get_video", json!({"video_id": 999}));
        assert!(called.model_text.contains("error"));
        assert!(called.summary.starts_with("error:"));
        assert!(called.is_error);
    }

    /// Every brain tested cuts on the final syllable. A piece needs a moment to land.
    #[test]
    fn the_last_picture_is_held_in_silence() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',1,120.0)", []).unwrap();

        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "last".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 20.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 20.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        let held = hold_the_last_picture(&db, 1, &mut script, &sc());
        assert!((held - 2.0).abs() < 0.01, "held {held}");
        // The shot runs on as a clip of its own, so the next finish can take the hold back
        // (`drop_closing_picture`) instead of finding a longer picture and holding again.
        let clip = &script.beats[0].clips[0];
        assert!((clip.out_s - 20.0).abs() < 0.01, "the chosen picture keeps its length: {}", clip.out_s);
        let hold = &script.beats[0].clips[1];
        assert_eq!(hold.why.as_deref(), Some(crate::script::CLOSING_PICTURE));
        assert_eq!((hold.video_id, hold.audio), (1, Audio::Mute));
        assert!((hold.in_s - 20.0).abs() < 0.01 && (hold.out_s - 22.0).abs() < 0.01, "the picture runs on: {hold:?}");
        assert!(drop_closing_picture(&mut script), "and the next finish takes it back");
        assert_eq!(script.beats[0].clips.len(), 1);
        // The voice is not extended with it: the quiet is the point.
        assert!((script.beats[0].bed.as_ref().unwrap().out_s - 20.0).abs() < 0.01);
    }

    /// There has to be footage left to hold, and a beat already read over does not need one.
    #[test]
    fn the_hold_takes_what_the_footage_and_the_beat_allow() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',1,20.6)", []).unwrap();
        let beat_of = |narration: Option<&str>, out_s: f64| Beat {
            id: "last".into(),
            purpose: "p".into(),
            narration: narration.map(str::to_string),
            on_screen_text: None,
            title_layout: None,
            notes: None,
            // A beat that ends on someone speaking: the case a hold is for. A last picture that
            // is already silent has ended the piece by itself and is left alone.
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                sentences: None,
                framing: None,
                video_id: 1,
                in_s: 10.0,
                out_s,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            bed: None,
            card: None,
        };
        let script_of = |b: Beat| Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![b],
            card_style: None,
        };

        // Only 0.6 s of the file is left: hold that and no more.
        let mut s = script_of(beat_of(None, 20.0));
        let held = hold_the_last_picture(&db, 1, &mut s, &sc());
        assert!((held - 0.6).abs() < 0.01, "held {held}");

        // Narration runs to the end: the piece already has an ending.
        let mut s = script_of(beat_of(Some("a closing line"), 15.0));
        assert_eq!(hold_the_last_picture(&db, 1, &mut s, &sc()), 0.0);

        // Turned off.
        let mut cfg = sc();
        cfg.closing_hold_s = 0.0;
        let mut s = script_of(beat_of(None, 15.0));
        assert_eq!(hold_the_last_picture(&db, 1, &mut s, &cfg), 0.0);
    }

    /// A one-clip cut of someone talking (a social clip) keeps its sound: the hold is a silent
    /// clip continuing the same shot, not the speaking clip extended and muted. Muting it left
    /// eight podcast clips with no audio at all (−91 dB).
    #[test]
    fn the_hold_after_a_speaker_keeps_their_voice() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',1,600.0)", []).unwrap();
        let mut s = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "clip".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 100.0,
                    out_s: 160.0,
                    audio: Audio::Source,
                    why: None,
                    tracks: None,
                }],
                bed: None,
                card: None,
            }],
            card_style: None,
        };
        let held = hold_the_last_picture(&db, 1, &mut s, &sc());
        assert!((held - 2.0).abs() < 0.01, "held {held}");
        let clips = &s.beats[0].clips;
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].audio, Audio::Source, "the speaker is still heard");
        assert_eq!((clips[0].in_s, clips[0].out_s), (100.0, 160.0));
        assert_eq!(clips[1].audio, Audio::Mute);
        assert_eq!((clips[1].video_id, clips[1].in_s), (1, 160.0), "the same shot carries on");
        assert!((clips[1].out_s - 162.0).abs() < 0.01);
        // Held once only, and a repair can take it back.
        assert_eq!(hold_the_last_picture(&db, 1, &mut s, &sc()), 0.0);
        assert!(drop_closing_picture(&mut s));
        assert_eq!(s.beats[0].clips.len(), 1);
    }

    /// A cutaway that outlasts the voice under it is a pause between interviews, which is the
    /// one thing b-roll should not be. Two of four beats in an agy cut ran on 2.6 s and 4.4 s.
    #[test]
    fn pictures_do_not_outlast_the_voice_carrying_them() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // Her face, then a cutaway: 14.5 s of picture over 11.9 s of voice.
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 0.0,
                        out_s: 10.5,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 4.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 11.9,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        assert_eq!(trim_pictures_to_bed(&mut script, &sc()), 1);
        let beat = &script.beats[0];
        let pictures: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        assert!((pictures - 11.9).abs() < 0.05, "pictures end with the voice: {pictures}");
        assert_eq!(beat.clips.len(), 2, "the cutaway survives");
        assert!(beat.clips[0].out_s < 10.5, "the face gave some of it: {}", beat.clips[0].out_s);
        assert!(beat.clips[1].out_s - beat.clips[1].in_s > 2.0, "and the cutaway is still a shot");
    }

    /// The trim takes a tail, not a piece: a beat whose voice covers a fraction of its pictures
    /// is a different mistake, and cutting to the bed threw most of a Qwen cut away.
    #[test]
    fn a_beat_whose_voice_covers_little_of_it_is_left_alone() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // 40 s of pictures over 8 s of voice.
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 0.0,
                        out_s: 8.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 16.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 3,
                        in_s: 0.0,
                        out_s: 16.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 8.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        assert_eq!(trim_pictures_to_bed(&mut script, &sc()), 0, "the cut is not thrown away");
        assert_eq!(script.beats[0].clips.len(), 3);
    }

    /// When there is not enough give in the shots to reach the voice, the trailing pictures go
    /// rather than becoming flashes — and never the opening face.
    #[test]
    fn a_cutaway_is_dropped_only_when_trimming_cannot_reach() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // 12 s of picture over 7 s of voice: a tail worth taking, but not enough give in
                // the two shots to reach it without one of them going.
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 0.0,
                        out_s: 8.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 4.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 7.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        trim_pictures_to_bed(&mut script, &sc());
        let beat = &script.beats[0];
        assert_eq!(beat.clips.len(), 1, "the cutaway goes rather than flashing by");
        assert_eq!(beat.clips[0].video_id, 1, "the face stays");
        // As close to the voice as the minimum shot length allows: a picture held under 4 s
        // reads as a flash, which is worse than a second of tail.
        let pictures: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        assert!((pictures - 7.0).abs() < 0.05, "what is left ends with the voice: {pictures}");
    }

    /// A beat read over by narration may hold its pictures: there is something to hear.
    #[test]
    fn a_narrated_beat_keeps_its_pictures() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: Some("a line read across the whole beat".into()),
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 2,
                    in_s: 0.0,
                    out_s: 12.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 4.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };
        assert_eq!(trim_pictures_to_bed(&mut script, &sc()), 0);
        assert!((script.beats[0].clips[0].out_s - 12.0).abs() < 0.01);
    }

    /// The bed is where most of the speech lives once one is laid, and the sentence guarantee
    /// never covered it: three clips in a Bonsai 2 cut and one in agy's stopped mid-sentence.
    #[test]
    fn a_bed_ends_on_a_whole_sentence_too() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',1,300.0)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (2,'i',1,300.0)", []).unwrap();
        // She speaks from 10 s to 31 s, in two sentences.
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (1, 10.0, 20.0, 'the first whole sentence'), (1, 20.0, 31.0, 'the second whole sentence')",
                [],
            )
            .unwrap();

        // A beat of 25 s of pictures, with her voice under it cut off at 28.4 s — mid-sentence.
        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                clips: vec![
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 1,
                        in_s: 10.0,
                        out_s: 20.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                    ScriptClip {
                        exact: false,
                        gain_db: None,
                        sentences: None,
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 15.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                    },
                ],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 28.4,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        assert_eq!(end_on_sentences(&db, &mut script, &sc()), 1);
        let bed = script.beats[0].bed.clone().unwrap();
        assert!((bed.out_s - 31.35).abs() < 0.01, "the bed runs to the end of her sentence: {}", bed.out_s);
        // And it still fits the pictures, so clamp_beds_to_beats will not cut it back again.
        clamp_beds_to_beats(&db, &mut script, &sc());
        assert!((script.beats[0].bed.as_ref().unwrap().out_s - 31.35).abs() < 0.01, "clamped away again");
    }

    /// A bed may not outlast its beat: finishing the sentence would, so it falls back instead of
    /// being cut off mid-word by the clamp that runs afterwards.
    #[test]
    fn a_bed_that_cannot_finish_the_sentence_falls_back_to_the_last_one() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',1,300.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (1, 10.0, 20.0, 'the first whole sentence'), (1, 20.0, 60.0, 'a very long answer')",
                [],
            )
            .unwrap();

        let mut script = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                notes: None,
                // Only 12 s of pictures.
                clips: vec![ScriptClip {
                    exact: false,
                    gain_db: None,
                    sentences: None,
                    framing: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 22.0,
                    audio: Audio::Mute,
                    why: None,
                    tracks: None,
                }],
                bed: Some(AudioBed {
                    sentences: None,
                    video_id: 1,
                    in_s: 10.0,
                    out_s: 22.0,
                    why: None,
                    inferred: true,
                    tracks: None,
                    gain_db: None,
                }),
                card: None,
            }],
            card_style: None,
        };

        end_on_sentences(&db, &mut script, &sc());
        let bed = script.beats[0].bed.clone().unwrap();
        assert!(bed.out_s <= 22.05, "the bed stays inside its pictures: {}", bed.out_s);
        assert!((bed.out_s - 20.35).abs() < 0.01, "and stops where the first sentence did: {}", bed.out_s);
    }

    /// Bonsai 2 spent 24 of 55 calls re-fetching four ranges it already had, seven times round a
    /// four-call cycle, and ran out of budget with nothing drafted.
    #[test]
    fn asking_for_the_same_thing_twice_costs_nothing_and_a_third_time_is_refused() {
        let mut memo = ToolMemo::default();
        let args = json!({ "video_id": 100, "start_s": 52.0, "end_s": 80.0 });

        assert!(memo.recall("get_video", &args).is_none(), "nothing is known yet");
        memo.remember("get_video", &args, "241s, 14 frames", "14 frames");

        // Asked again: the answer it already had, and told so.
        let (second, summary) = memo.recall("get_video", &args).expect("remembered");
        assert!(second.contains("241s, 14 frames"), "the answer is still there: {second}");
        assert!(second.contains("already called this"));
        assert!(summary.contains("repeat"));

        // Still asking: the payload is what it is stuck on, so it is withheld.
        let (third, _) = memo.recall("get_video", &args).expect("remembered");
        assert!(!third.contains("241s, 14 frames"), "the payload should be withheld: {third}");
        assert!(third.contains("Stop looking"));
    }

    /// The same call written two ways is the same call.
    #[test]
    fn argument_order_does_not_make_a_new_call() {
        let mut memo = ToolMemo::default();
        memo.remember("get_video", &json!({ "start_s": 1.0, "video_id": 7 }), "result", "sum");
        assert!(memo.recall("get_video", &json!({ "video_id": 7, "start_s": 1.0 })).is_some());
        // A different range is a different question.
        assert!(memo.recall("get_video", &json!({ "video_id": 7, "start_s": 2.0 })).is_none());
    }

    /// A redraft used to be accepted whatever came back, so a model that answered one complaint
    /// by breaking something worse quietly won. Here the second draft is four times its target;
    /// the first has to survive.
    #[tokio::test]
    async fn a_redraft_that_scores_worse_is_thrown_away() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_url = format!("http://127.0.0.1:{port}");

        // A clip of a video the tools never returned: enough to trigger the one redraft.
        // The first draft is the right length but its narration is too thin for the beat, which
        // is what asks for a redraft. The redraft answers by making the cut four times as long.
        let draft = |title: &str, out_s: f64, narration: &str| {
            json!({
                "title": title,
                "target_duration_s": 5.0,
                "beats": [{
                    "id": "b1", "purpose": "hook", "narration": narration,
                    "clips": [{ "video_id": 1, "in_s": 0.0, "out_s": out_s, "audio": "mute" }]
                }]
            })
            .to_string()
        };

        tokio::spawn(async move {
            for step in 1..=3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let mut total = 0;
                loop {
                    if total == buf.len() {
                        buf.resize(buf.len() * 2, 0);
                    }
                    let n = socket.read(&mut buf[total..]).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    total += n;
                    let text = String::from_utf8_lossy(&buf[..total]);
                    if let Some(head) = text.find("\r\n\r\n")
                        && let Some(idx) = text.to_lowercase().find("content-length:")
                    {
                        let rest = &text[idx + 15..];
                        let len: usize = rest[..rest.find("\r\n").unwrap()].trim().parse().unwrap();
                        if total >= head + 4 + len {
                            break;
                        }
                    }
                }
                let content = match step {
                    1 => "Looked at the footage.".to_string(),
                    2 => draft("first", 5.0, "a short line"),
                    _ => draft("worse", 20.0, "a short line"),
                };
                let body =
                    json!({ "choices": [{ "message": { "role": "assistant", "content": content } }] }).to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                let _ = socket.shutdown().await;
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("Redraft")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"test").unwrap();
        let c = &db.conn;
        c.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'h',500,60.0)", []).unwrap();
        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, ?2, 500, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();
        // Something really is visible there, so the clip survives grounding and the two drafts
        // differ only in the thing under test: their length.
        for t in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0] {
            c.execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (1, ?1, '{\"description\":\"a street\"}')",
                params![t],
            )
            .unwrap();
        }

        let mut ctx = ChatContext {
            db,
            paths: crate::paths::Paths::in_dir(tmp.path()),
            backend: ChatBackend::Server {
                url: server_url,
                model: "m".into(),
                api_key: String::new(),
                ctx_tokens: 32768,
            },
            embedder: None,
            system_prompt: None,
            max_tool_rounds: 0,
            script: sc(),
            jev: Default::default(),
            cancel: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "a 5 second teaser", &[], None, &mut |_| {}).await.unwrap();
        let script = res.script.expect("a script");
        assert_eq!(script.title, "first", "the worse redraft was kept");
        assert!(
            res.issues.iter().any(|i| i.message.contains("kept the first draft")),
            "the turn should say it threw one away: {:?}",
            res.issues.iter().map(|i| &i.message).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn server_loop_against_fake_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_url = format!("http://127.0.0.1:{port}");

        // Spawn minimal HTTP server handling 3 sequential calls
        tokio::spawn(async move {
            for step in 1..=3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                // Grows with the request: a fixed buffer silently truncates the body, and the
                // assertions below then fail on a prompt that merely got longer.
                let mut buf = vec![0u8; 8192];
                let mut total_read = 0;
                loop {
                    if total_read == buf.len() {
                        buf.resize(buf.len() * 2, 0);
                    }
                    let n = socket.read(&mut buf[total_read..]).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    total_read += n;
                    let s = String::from_utf8_lossy(&buf[..total_read]);
                    if let Some(header_end) = s.find("\r\n\r\n") {
                        if let Some(cl_idx) = s.to_lowercase().find("content-length:") {
                            let rest = &s[cl_idx + 15..];
                            let end_line = rest.find("\r\n").unwrap();
                            let cl: usize = rest[..end_line].trim().parse().unwrap();
                            if total_read >= header_end + 4 + cl {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }

                let req_text = String::from_utf8_lossy(&buf[..total_read]);

                let mut resp_body = match step {
                    1 => {
                        assert!(req_text.contains("tools"));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_moments",
                                            "arguments": "{\"query\":\"unboxing\"}"
                                        }
                                    }]
                                }
                            }]
                        })
                    }
                    2 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "Found the unboxing footage. Now drafting your script.",
                                    "tool_calls": null
                                }
                            }]
                        })
                    }
                    3 => {
                        assert!(req_text.contains("response_format"));
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": json!({
                                        "title": "CM5 Fast Teaser",
                                        "beats": [
                                            {
                                                "id": "b1",
                                                "purpose": "hook",
                                                "narration": "Meet the new compute module.",
                                                "clips": [
                                                    { "video_id": 1, "in_s": 1.0, "out_s": 4.0, "audio": "source" }
                                                ]
                                            }
                                        ]
                                    }).to_string()
                                }
                            }]
                        })
                    }
                    _ => unreachable!(),
                };

                resp_body["model"] = json!("actual-served-model");
                resp_body["usage"] = json!({"completion_tokens":80,"prompt_tokens":9000});
                resp_body["timings"] = json!({"predicted_ms":2000.0});
                let resp_bytes = resp_body.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp_bytes.len(),
                    resp_bytes
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                let _ = socket.shutdown().await;
            }
        });

        // Set up project and database
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("FakeServerProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"test").unwrap();

        let c = &db.conn;
        c.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'hash1', 500, 10.0)", []).unwrap();
        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (1, ?1, ?2, 500, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();
        c.execute(
            "INSERT INTO chunks(video_id, kind, start_s, end_s, text)
             VALUES (1, 'moment', 0.0, 5.0, 'unboxing the board')",
            [],
        )
        .unwrap();

        let mut events = Vec::new();
        let mut ctx = ChatContext {
            db,
            paths: crate::paths::Paths::in_dir(tmp.path()),
            backend: ChatBackend::Server {
                url: server_url,
                model: "test-model".into(),
                api_key: String::new(),
                ctx_tokens: 32768,
            },
            embedder: None,
            system_prompt: None,
            max_tool_rounds: 0,
            script: sc(),
            jev: Default::default(),
            cancel: None,
        };

        let res =
            run_turn(&mut ctx, p.id, None, "Create a 5s teaser about unboxing", &[], None, &mut |e| events.push(e))
                .await
                .unwrap();

        assert!(res.script_id.is_some());
        let script = res.script.unwrap();
        assert_eq!(script.title, "CM5 Fast Teaser");
        assert_eq!(script.beats.len(), 1);
        assert_eq!(res.tool_calls.len(), 1);
        // The model asked by the old name; the record and the log use the registry's.
        assert_eq!(res.tool_calls[0].tool, "search");

        // Verify events
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolStarted { tool, .. } if tool == "search")));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolFinished { tool, .. } if tool == "search")));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::Drafting)));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::Validating)));

        // Verify messages in DB
        let msgs = messages(&ctx.db, res.session_id).unwrap();
        assert_eq!(msgs.len(), 3); // user, tool, assistant
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "tool");
        assert!(msgs[1].tool_calls.is_some());
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].script_id, res.script_id);
        assert!(msgs[0].performance.is_none());
        let performance = msgs[2].performance.as_ref().expect("persisted answer metrics");
        assert!(performance.elapsed_seconds > 0.0);
        assert_eq!(performance.calls.len(), 3);
        for call in &performance.calls {
            assert_eq!(call.model, "actual-served-model");
            assert_eq!(call.output_tokens, Some(80));
            assert_eq!(call.generation_seconds, Some(2.0));
        }
    }

    #[tokio::test]
    async fn grounding_rejection_drops_ungrounded_clips() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_url = format!("http://127.0.0.1:{port}");

        // Server does 1 round: returns no tool calls, then returns script with ungrounded clip range (100.0..105.0)
        // on retry returns the same ungrounded script
        tokio::spawn(async move {
            for step in 1..=3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = socket.read(&mut buf).await.unwrap();
                let _req_text = String::from_utf8_lossy(&buf[..n]);

                let resp_body = match step {
                    1 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "No tools needed, drafting.",
                                    "tool_calls": null
                                }
                            }]
                        })
                    }
                    2 | 3 => {
                        json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": json!({
                                        "title": "Ungrounded Script",
                                        "beats": [
                                            {
                                                "id": "b1",
                                                "purpose": "hook",
                                                "clips": [
                                                    { "video_id": 1, "in_s": 100.0, "out_s": 105.0 }
                                                ]
                                            }
                                        ]
                                    }).to_string()
                                }
                            }]
                        })
                    }
                    _ => unreachable!(),
                };

                let resp_bytes = resp_body.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp_bytes.len(),
                    resp_bytes
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                let _ = socket.shutdown().await;
            }
        });

        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("UngroundedProj")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        let file_path = tmp.path().join("clip.mp4");
        std::fs::write(&file_path, b"test").unwrap();

        let c = &db.conn;
        c.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'hash1', 500, 200.0)", [])
            .unwrap();
        c.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (1, ?1, ?2, 500, 0, 0)",
            params![folder.id, file_path.to_str().unwrap()],
        )
        .unwrap();

        let mut ctx = ChatContext {
            db,
            paths: crate::paths::Paths::in_dir(tmp.path()),
            backend: ChatBackend::Server {
                url: server_url,
                model: "test-model".into(),
                api_key: String::new(),
                ctx_tokens: 32768,
            },
            embedder: None,
            system_prompt: None,
            max_tool_rounds: 0,
            script: sc(),
            jev: Default::default(),
            cancel: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "Make video", &[], None, &mut |_| {}).await.unwrap();

        // The clip was not grounded in tool results -> dropped -> no beats left -> script_id None
        assert_eq!(res.script_id, None);
        assert!(res.script.is_none());
    }

    #[test]
    fn run_turn_future_is_send() {
        fn assert_send<T: Send>(_: T) {}
        let _ = |ctx: &mut ChatContext, mut cb: Box<dyn FnMut(ChatEvent) + Send>| {
            assert_send(run_turn(ctx, 1, None, "test", &[], None, &mut *cb));
        };
    }

    #[test]
    fn test_persist_chat_images_and_messages_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("ImgProj")).unwrap();
        let sid = create_session(&db, p.id, "Session with image").unwrap();

        // 1x1 PNG in base64
        let b64_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let saved = persist_chat_images(tmp.path(), sid, &[b64_png.to_string()]);
        assert_eq!(saved.len(), 1);
        assert!(saved[0].is_file());

        let imgs_json = serde_json::to_string(&vec![saved[0].to_string_lossy().to_string()]).ok();
        db.conn
            .execute(
                "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at, images_json)
             VALUES (?1, 'user', 'look at this image', NULL, ?2, ?3)",
                params![sid, now(), imgs_json],
            )
            .unwrap();

        let msgs = messages(&db, sid).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "look at this image");
        let loaded_imgs = msgs[0].images.as_ref().unwrap();
        assert_eq!(loaded_imgs.len(), 1);
        assert_eq!(loaded_imgs[0], saved[0].to_str().unwrap());
    }
}
