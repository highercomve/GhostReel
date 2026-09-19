//! Project-scoped script chat agent (plan §4a, M8c).
//!
//! Grounded script writing through iterative tool calling over local footage
//! and transcripts. Supports server backends (OpenAI-compatible) and local
//! `ghostreel-llm` helper backends.

use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Error;
use crate::db::Db;
use crate::embed::Embedder;
use crate::projects::{Project, now};
use crate::script::{Issue, IssueSeverity, Script, ScriptClip, save_version, snap_to_segments, validate};
use crate::search::{SearchOptions, query_vector, search_with_vector};

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

/// A stored chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatSession {
    pub id: i64,
    pub project_id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A message in a chat session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub id: i64,
    pub session_id: i64,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallRecord>>,
    pub script_id: Option<i64>,
    pub created_at: i64,
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
    pub async fn from_runtime(rt: &crate::runtime::Runtime) -> Result<Self, Error> {
        Self::from_vision_setup(&rt.vision).await
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
    pub async fn from_vision_setup(setup: &crate::runtime::VisionSetup) -> Result<Self, Error> {
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
            crate::runtime::VisionSetup::Cli(cfg) => Ok(ChatBackend::Cli(crate::cliagent::CliAgent::new(cfg.clone()))),
            crate::runtime::VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime } => {
                let mut paths = Vec::with_capacity(2);
                for spec in [model, mmproj] {
                    if let Some(p) = found.iter().find(|p| p.file_name().is_some_and(|n| n == spec.file_name.as_str()))
                    {
                        paths.push(p.clone());
                    } else {
                        paths.push(crate::models::download(spec, models_dir, |_, _| {}).await?);
                    }
                }
                let vision = Some((paths[0].clone(), paths[1].clone()));
                let models = crate::vision::LocalModels {
                    helper: helper.clone(),
                    vision,
                    embed: None,
                    cpu: false,
                    runtime: runtime.clone(),
                };
                let llm = crate::vision::LocalLlm::start(&models).await?;
                Ok(ChatBackend::Local { helper: Box::new(llm), ctx_tokens: runtime.ctx_tokens, think: runtime.think })
            }
            crate::runtime::VisionSetup::Unavailable(why) => {
                Err(Error::Vision(format!("vision/chat model unavailable: {why}")))
            }
        }
    }
}

/// Context owning all resources needed for a chat turn.
pub struct ChatContext {
    pub db: Db,
    pub data_dir: PathBuf,
    pub backend: ChatBackend,
    pub embedder: Option<Embedder>,
    /// Custom editing instructions (Settings); `None`/empty = [`DEFAULT_EDITOR_PROMPT`].
    pub system_prompt: Option<String>,
    /// `chat_model.max_tool_rounds`; 0 = the backend's own default.
    pub max_tool_rounds: u32,
    /// The `[script]` settings: clip lengths, target tolerances, narration pace, research budget.
    pub script: crate::config::ScriptConfig,
    /// Set by Stop. Checked between tool rounds and before every model call, so a turn ends
    /// within a round instead of after the whole draft.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Returned when a turn is stopped; the caller reports it as cancelled, not as a failure.
pub const CANCELLED: &str = "stopped";

/// Pacing problems the model can fix in a redraft: clips too long or too short, and (when
/// `enforce_target`) a total far from the target.
pub fn pacing_issues(script: &Script, enforce_target: bool, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    for beat in &script.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let len = c.out_s - c.in_s;
            if len > cfg.max_clip_s {
                issues.push(Issue {
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
                severity: IssueSeverity::Warning,
                beat_id: None,
                clip_index: None,
                message: format!("total clip duration is {total:.1} s but the target is {target:.0} s"),
            });
        }
    }
    issues
}

/// Drop ungrounded/foreign clips and trim over-long ones; with `enforce_target`, also squeeze the
/// total toward the target. Revisions don't enforce it: the user's feedback ("slower", "longer")
/// must be able to change the length. Returns the issues describing the changes.
fn enforce_grounding_and_pacing(
    db: &Db,
    project_id: i64,
    s: &mut Script,
    grounding: &Grounding,
    enforce_target: bool,
    cfg: &crate::config::ScriptConfig,
) -> Vec<Issue> {
    let mut issues = tidy_beat_ids(s);
    // A clip on a stretch the camera shakes through is moved to the nearest steady stretch of
    // the same shot, or dropped when there is none. Telling the model was not enough: the same
    // stretch of a clip measured at eleven times the sway limit was cut into four scripts.
    for beat in &mut s.beats {
        let mut kept = Vec::with_capacity(beat.clips.len());
        for mut c in beat.clips.drain(..) {
            let windows = db.motion_windows(c.video_id).unwrap_or_default();
            if windows.is_empty() || cfg.max_shake_jerk <= 0.0 {
                kept.push(c);
                continue;
            }
            let limit = crate::steadiness::shake_limit(&windows, cfg.max_shake_jerk, cfg.shake_relative);
            let spans = crate::steadiness::shaky_spans_with_sway(&windows, c.in_s, c.out_s, limit, cfg.max_sway);
            let len = c.out_s - c.in_s;
            let shaky_s: f64 = spans.iter().map(|(a, b)| (b.min(c.out_s) - a.max(c.in_s)).max(0.0)).sum();
            if len <= 0.0 || shaky_s <= len * 0.5 {
                kept.push(c);
                continue;
            }
            match crate::steadiness::steady_stretch_near(&windows, c.in_s, len, limit, cfg.max_sway) {
                Some(start) => {
                    issues.push(Issue {
                        severity: IssueSeverity::Info,
                        beat_id: Some(beat.id.clone()),
                        clip_index: None,
                        message: format!(
                            "moved clip off a shaky stretch: video #{} {:.1}–{:.1} s → {:.1}–{:.1} s",
                            c.video_id,
                            c.in_s,
                            c.out_s,
                            start,
                            start + len
                        ),
                    });
                    c.in_s = start;
                    c.out_s = start + len;
                    kept.push(c);
                }
                None => issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "dropped clip: the camera shakes through video #{} {:.1}–{:.1} s and nothing steady \
                         of that length exists in it",
                        c.video_id, c.in_s, c.out_s
                    ),
                }),
            }
        }
        beat.clips = kept;
    }
    // A clip playing its own audio must start on someone close to the microphone: left alone it
    // opens on the interviewer's question or the slate, which is what "the sound is awful" meant.
    let mut retimed = 0usize;
    for beat in &mut s.beats {
        for c in &mut beat.clips {
            if c.audio != crate::script::Audio::Source || db.opens_on_mic(c.video_id, c.in_s, c.out_s) {
                continue;
            }
            // It opens on the interviewer: move to the first on-mic segment, keeping the length.
            let next: Option<f64> = db
                .conn
                .query_row(
                    "SELECT start_s FROM transcript_segments WHERE video_id = ?1 AND start_s >= ?2 \
                     AND COALESCE(off_mic, 0) = 0 ORDER BY start_s LIMIT 1",
                    params![c.video_id, c.in_s],
                    |r| r.get(0),
                )
                .ok();
            if let Some(start) = next {
                let len = c.out_s - c.in_s;
                c.in_s = start;
                c.out_s = start + len;
                retimed += 1;
            }
        }
    }
    if retimed > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("moved {retimed} clip(s) off the interviewer's questions to where someone answers"),
        });
    }
    // A clip with no transcript in its range cannot carry source audio, whatever the model said.
    // Left alone, a scenery shot marked "source" reads as "someone speaks here", which is exactly
    // the case the editing rules tell the model to leave narration empty for — so a whole script
    // of b-roll ends up silent.
    let mut unmuted = 0usize;
    for beat in &mut s.beats {
        for c in &mut beat.clips {
            if c.audio == crate::script::Audio::Source && !clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                c.audio = crate::script::Audio::Mute;
                unmuted += 1;
            }
        }
    }
    let mut hushed = 0usize;
    for beat in &mut s.beats {
        let speaks = beat.clips.iter().any(|c| c.audio == crate::script::Audio::Source);
        let narrated = beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty());
        if speaks && narrated {
            beat.narration = Some(String::new());
            hushed += 1;
        }
    }
    if hushed > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("dropped narration from {hushed} beat(s) that play the speaker's own audio"),
        });
    }
    if unmuted > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("muted {unmuted} clip(s) with no speech in range"),
        });
    }
    // Now that it is settled who speaks, let their voice carry the pictures that follow.
    issues.extend(lay_audio_beds(db, s, cfg));
    for beat in &mut s.beats {
        let mut kept = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if !is_video_in_project(db, project_id, c.video_id) {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!("dropped clip from video #{} — not in this project", c.video_id),
                });
                continue;
            }
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
        for (i, c) in kept.iter_mut().enumerate() {
            if c.out_s - c.in_s > cfg.max_clip_s {
                issues.push(Issue {
                    severity: IssueSeverity::Info,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(i),
                    message: format!(
                        "trimmed {:.1} s clip to {:.0} s (video #{} from {:.1} s)",
                        c.out_s - c.in_s,
                        cfg.trimmed_clip_s,
                        c.video_id,
                        c.in_s
                    ),
                });
                c.out_s = c.in_s + cfg.trimmed_clip_s;
            }
        }
        beat.clips = kept;
    }
    s.beats.retain(|b| !b.clips.is_empty());
    clamp_to_duration(db, s);
    let repeats = drop_repeated_footage(s);
    if repeats > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Warning,
            beat_id: None,
            clip_index: None,
            message: format!("dropped {repeats} clip(s) that repeated footage already used earlier"),
        });
    }
    let merged = merge_contiguous_clips(s, cfg);
    if merged > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("joined {merged} back-to-back cut(s) of the same shot into continuous clips"),
        });
    }
    // Pad before trimming, so the target is met with the people's pauses already in.
    pad_speech(db, s, cfg);
    // Padding can grow two clips of the same video into each other: check again.
    let overlapped = drop_repeated_footage(s);
    if overlapped > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("dropped {overlapped} clip(s) that overlapped another once padded"),
        });
    }
    let before = s.total_duration_s();
    let speaking = |c: &ScriptClip| clip_has_speech(db, c.video_id, c.in_s, c.out_s);
    if enforce_target && trim_to_target_with(s, speaking, cfg) {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("clips shortened proportionally: {before:.1} s → {:.1} s", s.total_duration_s()),
        });
    }
    issues
}

/// Grounding tracker recording footage ranges inspected by tools.
#[derive(Debug, Clone, Default)]
pub struct Grounding {
    pub ranges: Vec<(i64, f64, f64)>,
}

impl Grounding {
    pub fn add(&mut self, video_id: i64, start_s: f64, end_s: f64) {
        self.ranges.push((video_id, start_s, end_s));
    }

    /// Check if a clip lies inside a grounded range for that video (±5 s slack at both ends).
    /// Containment, not overlap: a 45 s clip touching a 5 s search hit is not grounded.
    pub fn is_grounded(&self, video_id: i64, in_s: f64, out_s: f64, slack: f64) -> bool {
        self.ranges.iter().any(|&(vid, s, e)| vid == video_id && in_s >= s - slack && out_s <= e + slack)
    }

    /// Record grounding ranges from a tool invocation.
    pub fn record_tool_call(&mut self, tool: &str, args: &Value, db: &Db) {
        match tool {
            "get_transcript" => {
                if let (Some(vid), Some(s), Some(e)) = (
                    args.get("video_id").and_then(|v| v.as_i64()),
                    args.get("start_s").and_then(|v| v.as_f64()),
                    args.get("end_s").and_then(|v| v.as_f64()),
                ) {
                    self.add(vid, s, e);
                }
            }
            "get_video" => {
                if let Some(vid) = args.get("video_id").and_then(|v| v.as_i64()) {
                    let dur: Option<f64> =
                        db.conn.query_row("SELECT duration_s FROM videos WHERE id = ?1", [vid], |r| r.get(0)).ok();
                    self.add(vid, 0.0, dur.unwrap_or(0.0));
                }
            }
            _ => {}
        }
    }
}

/// Local constrained action representation.
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
                                    "audio": { "type": "string", "enum": ["source", "mute"] },
                                    "why": { "type": "string" }
                                },
                                "required": ["video_id", "in_s", "out_s", "audio"],
                                "additionalProperties": false
                            }
                        },
                        "bed": {
                            "type": "object",
                            "properties": {
                                "video_id": { "type": "integer" },
                                "in_s": { "type": "number" },
                                "out_s": { "type": "number" },
                                "why": { "type": "string" }
                            },
                            "required": ["video_id", "in_s", "out_s"],
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

/// OpenAI tools schema definition for the 4 tools.
pub fn tools_definition() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "search_moments",
                "description": "Search indexed footage by keyword or meaning in the project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query terms" },
                        "limit": { "type": "integer", "description": "Max results (1-20, default 10)" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_transcript",
                "description": "Get timestamped transcript segments for a video within a time range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": { "type": "integer", "description": "Video ID" },
                        "start_s": { "type": "number", "description": "Start timestamp in seconds" },
                        "end_s": { "type": "number", "description": "End timestamp in seconds" }
                    },
                    "required": ["video_id", "start_s", "end_s"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_video",
                "description": "Get video metadata and keyframe descriptions (what is visible at each time). Pass start_s/end_s to see every keyframe in a range before choosing a clip.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": { "type": "integer", "description": "Video ID" },
                        "start_s": { "type": "number", "description": "Optional range start in seconds" },
                        "end_s": { "type": "number", "description": "Optional range end in seconds" }
                    },
                    "required": ["video_id"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_videos",
                "description": "List all videos belonging to the project with summaries",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }
    ])
}

/// Local LLM action schema (oneOf tool action or final script action).
pub fn local_action_schema() -> Value {
    json!({
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["tool"] },
                    "tool": { "type": "string", "enum": ["search_moments", "get_transcript", "get_video", "list_videos"] },
                    "args": { "type": "object" }
                },
                "required": ["action", "tool", "args"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["final"] },
                    "script": script_json_schema()
                },
                "required": ["action", "script"],
                "additionalProperties": false
            },
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

/// Local LLM final action schema only.
pub fn local_final_action_schema() -> Value {
    json!({
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["final"] },
                    "script": script_json_schema()
                },
                "required": ["action", "script"],
                "additionalProperties": false
            },
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
fn approx_tokens(messages: &[Value]) -> usize {
    messages.iter().map(|m| m.to_string().len()).sum::<usize>() / 4
}

/// Make room for another round by forgetting the oldest tool results.
///
/// Every round re-sends the whole conversation, so a model that keeps looking eventually fills the
/// window — Bonsai 2 made 199 tool calls and the turn died on a raw "exceeds the available context
/// size" from the server, with nothing drafted. The oldest results are the ones it has already
/// used or discarded; dropping them keeps the research going. Returns false when there is nothing
/// left to drop, which means it is time to draft with what we have.
fn make_room(messages: &mut Vec<Value>, budget_tokens: usize) -> bool {
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
            .find(|(_, m)| m.get("role").and_then(Value::as_str) == Some("tool"))
            .map(|(i, _)| i);
        match victim {
            Some(i) => {
                messages.remove(i);
            }
            None => return false,
        }
    }
    true
}

/// Everything anyone says in the project, with timestamps, laid out for the editor to read
/// before it cuts anything.
///
/// A model that has to *ask* for each transcript only reads the tapes it already suspects: one run
/// looked at two videos, found a good speaker and built the whole teaser out of him, while three
/// other people said better things on tapes it never opened. An editor does not work that way —
/// they read the interviews first, decide what the story is, and then go looking for pictures.
///
/// It is affordable: every word of a 96-video project is about 34 000 characters. When that will
/// not fit, the videos with the most speech go in first and the rest are named with a pointer to
/// `get_transcript`, so nothing is hidden, only deferred.
pub fn speech_digest(db: &Db, project_id: i64, max_chars: usize) -> String {
    let Ok(mut st) = db.conn.prepare(
        "SELECT ts.video_id, MIN(vf.path), SUM(LENGTH(ts.text))
           FROM transcript_segments ts
           JOIN video_files vf ON vf.video_id = ts.video_id
           JOIN folders f ON f.id = vf.folder_id
           JOIN project_folders pf ON pf.folder_id = f.id
          WHERE pf.project_id = ?1
            AND NOT EXISTS (SELECT 1 FROM project_exclusions x
                             WHERE x.project_id = pf.project_id AND x.video_id = ts.video_id)
          GROUP BY ts.video_id
          ORDER BY SUM(LENGTH(ts.text)) DESC",
    ) else {
        return String::new();
    };
    let videos: Vec<(i64, String, i64)> = st
        .query_map([project_id], |r| Ok((r.get(0)?, r.get::<_, String>(1)?, r.get(2)?)))
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();
    if videos.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "\nWHAT PEOPLE SAY\nEvery word spoken in this project, with the timestamps to cut on. Read it first and \
         decide whose words carry the story; then use search_moments and get_video to find pictures for what they \
         describe. A line marked [off-mic] is the interviewer or someone away from the microphone: never open a clip \
         or build a beat on one.\n",
    );
    let mut left_out = Vec::new();

    for (video_id, path, _) in &videos {
        let name = Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut block = format!("\n#{video_id} {name}\n");
        if let Ok(mut q) = db.conn.prepare(
            "SELECT start_s, end_s, text, COALESCE(off_mic, 0) FROM transcript_segments
              WHERE video_id = ?1 ORDER BY start_s",
        ) {
            let rows = q.query_map([video_id], |r| {
                Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?, r.get::<_, i64>(3)? != 0))
            });
            for (start, end, text, off) in rows.into_iter().flatten().flatten() {
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }
                block.push_str(&format!(
                    "  {start:.2}-{end:.2}{} {text}\n",
                    if off { " [off-mic]" } else { "" }
                ));
            }
        }
        if out.len() + block.len() <= max_chars {
            out.push_str(&block);
            continue;
        }
        // Out of room. Keep as much of this tape as fits rather than dropping it: the list is in
        // order of how much is said, so the one being cut is the one most worth reading.
        let room = max_chars.saturating_sub(out.len());
        match block[..block.len().min(room)].rfind('\n') {
            Some(cut) if cut > 120 => {
                out.push_str(&block[..=cut]);
                out.push_str("  … the rest of this one with get_transcript\n");
            }
            _ => left_out.push(format!("#{video_id}")),
        }
    }

    if !left_out.is_empty() {
        out.push_str(&format!(
            "\nAlso speech in {} more: {} — read them with get_transcript.\n",
            left_out.len(),
            left_out.join(", ")
        ));
    }
    out
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

const EMPTY_SEARCH_HINT: &str = "Those searches found nothing — the words you are searching for are not in this \
footage. Stop searching: call list_videos, then get_video and get_transcript on the videos that look useful, and \
build the script from what they actually show.";

/// A `search_moments` result with no hits.
fn is_empty_search(tool: &str, summary: &str) -> bool {
    tool == "search_moments" && (summary.starts_with("0 hits") || summary == "no matches")
}

/// Check whether a video belongs to a project.
pub fn is_video_in_project(db: &Db, project_id: i64, video_id: i64) -> bool {
    let res: Result<i64, _> = db.conn.query_row(
        "SELECT 1 FROM video_files vf
         JOIN folders f ON f.id = vf.folder_id
         JOIN project_folders pf ON pf.folder_id = f.id
         WHERE pf.project_id = ?1 AND vf.video_id = ?2
           AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
         LIMIT 1",
        params![project_id, video_id],
        |r| r.get(0),
    );
    res.is_ok()
}

/// Truncate a JSON array of items until its serialized string length <= max_len.
fn truncate_json_list<T: Serialize>(items: &mut Vec<T>, max_len: usize) -> String {
    while !items.is_empty() {
        if let Ok(s) = serde_json::to_string(items)
            && s.len() <= max_len
        {
            return s;
        }
        items.pop();
    }
    serde_json::to_string(items).unwrap_or_else(|_| "[]".into())
}

/// Dispatch a tool call using DB and optional precomputed query vector. Returns (json_result, summary).
pub fn dispatch_tool(
    db: &Db,
    data_dir: &Path,
    project_id: i64,
    tool: &str,
    args: &Value,
    grounding: &mut Grounding,
    vector: Option<&[f32]>,
    cfg: &crate::config::ScriptConfig,
) -> (String, String) {
    dispatch_tool_limited(
        db,
        data_dir,
        project_id,
        tool,
        args,
        grounding,
        vector,
        cfg.local_tool_result_chars,
        cfg.max_shake_jerk,
        cfg.shake_relative,
        cfg.max_sway,
    )
}

/// Tool results for the local helper (8k context) stay small.
pub const LOCAL_TOOL_RESULT_CHARS: usize = 1500;
/// Servers have large contexts (highllama: 120k); richer results make much better edits.
pub const SERVER_TOOL_RESULT_CHARS: usize = 8000;

/// [`dispatch_tool`] with an explicit cap on the JSON result length.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tool_limited(
    db: &Db,
    data_dir: &Path,
    project_id: i64,
    tool: &str,
    args: &Value,
    grounding: &mut Grounding,
    vector: Option<&[f32]>,
    max_chars: usize,
    max_shake: f64,
    shake_relative: f64,
    max_sway: f64,
) -> (String, String) {
    match tool {
        "search_moments" => {
            let query = match args.get("query").and_then(|v| v.as_str()) {
                Some(q) if !q.trim().is_empty() => q,
                _ => return (json!({"error": "missing or invalid query"}).to_string(), "error: missing query".into()),
            };
            // Exactly what the Library search box does: whole project, every kind of chunk. A
            // `kind` filter here hid everything whenever that part of the index wasn't built yet
            // (frame descriptions still running) while the same words sat in the speech.
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10).clamp(1, 20) as usize;
            let opts = SearchOptions { project_id: Some(project_id), limit, kinds: None };

            let hits = match search_with_vector(db, data_dir, query, vector, &opts) {
                Ok(h) => h,
                Err(e) => {
                    return (
                        json!({"error": format!("search failed: {e}")}).to_string(),
                        "error: search failed".into(),
                    );
                }
            };

            #[derive(Serialize)]
            struct CompactHit {
                video_id: i64,
                file: String,
                start_s: f64,
                end_s: f64,
                /// static, tripod, stabilised, handheld — the model picks clips straight from a
                /// hit, so what it needs to know about the camera has to be here.
                camera: &'static str,
                /// The hit lies on a stretch the camera shakes through.
                shaky: bool,
                snippet: String,
            }

            let mut compact = Vec::new();
            for hit in &hits {
                grounding.add(hit.video_id, hit.start_s, hit.end_s);
                let file = hit.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let snippet: String = hit.snippet.chars().take(200).collect();
                let windows = db.motion_windows(hit.video_id).unwrap_or_default();
                let limit = crate::steadiness::shake_limit(&windows, max_shake, shake_relative);
                let shaky =
                    !crate::steadiness::shaky_spans_with_sway(&windows, hit.start_s, hit.end_s, limit, max_sway)
                        .is_empty();
                compact.push(CompactHit {
                    video_id: hit.video_id,
                    file,
                    start_s: (hit.start_s * 100.0).round() / 100.0,
                    end_s: (hit.end_s * 100.0).round() / 100.0,
                    camera: crate::steadiness::camera_style(&windows).as_str(),
                    shaky,
                    snippet,
                });
            }

            let summary = format!("{} hits", hits.len());
            let json_str = truncate_json_list(&mut compact, max_chars);
            (json_str, summary)
        }
        "get_transcript" => {
            let video_id = match args.get("video_id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return (json!({"error": "missing video_id"}).to_string(), "error: missing video_id".into()),
            };
            if !is_video_in_project(db, project_id, video_id) {
                return (
                    json!({"error": format!("video #{video_id} not in project")}).to_string(),
                    format!("error: video #{video_id} not in project"),
                );
            }
            let start_s = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end_s = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);

            grounding.add(video_id, start_s, end_s);

            let mut st = match db.conn.prepare(
                "SELECT start_s, end_s, text, COALESCE(off_mic, 0) FROM transcript_segments
                 WHERE video_id = ?1 AND end_s >= ?2 AND start_s <= ?3
                 ORDER BY start_s",
            ) {
                Ok(s) => s,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db prepare".into()),
            };

            #[derive(Serialize)]
            struct CompactSeg {
                start_s: f64,
                end_s: f64,
                /// Spoken away from the microphone — in an interview, the person asking the
                /// questions rather than the one answering.
                off_mic: bool,
                text: String,
            }

            let rows = match st.query_map(params![video_id, start_s, end_s], |r| {
                Ok(CompactSeg {
                    start_s: (r.get::<_, f64>(0)? * 100.0).round() / 100.0,
                    end_s: (r.get::<_, f64>(1)? * 100.0).round() / 100.0,
                    off_mic: r.get::<_, bool>(3)?,
                    text: r.get(2)?,
                })
            }) {
                Ok(rows) => rows,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db query".into()),
            };

            let mut segs: Vec<CompactSeg> = rows.filter_map(Result::ok).collect();
            let summary = format!("{} segments", segs.len());
            let json_str = truncate_json_list(&mut segs, max_chars);
            (json_str, summary)
        }
        "get_video" => {
            let video_id = match args.get("video_id").and_then(|v| v.as_i64()) {
                Some(id) => id,
                None => return (json!({"error": "missing video_id"}).to_string(), "error: missing video_id".into()),
            };
            if !is_video_in_project(db, project_id, video_id) {
                return (
                    json!({"error": format!("video #{video_id} not in project")}).to_string(),
                    format!("error: video #{video_id} not in project"),
                );
            }

            struct VideoMeta {
                duration_s: Option<f64>,
                fps: Option<f64>,
                width: Option<i64>,
                height: Option<i64>,
                language: Option<String>,
            }

            let vid_meta: Option<VideoMeta> = db
                .conn
                .query_row(
                    "SELECT duration_s, fps, width, height, language FROM videos WHERE id = ?1",
                    [video_id],
                    |r| {
                        Ok(VideoMeta {
                            duration_s: r.get(0)?,
                            fps: r.get(1)?,
                            width: r.get(2)?,
                            height: r.get(3)?,
                            language: r.get(4)?,
                        })
                    },
                )
                .ok();

            let (duration_s, fps, width, height, language) = match vid_meta {
                Some(m) => (m.duration_s, m.fps, m.width, m.height, m.language),
                None => {
                    return (
                        json!({"error": format!("video #{video_id} not found")}).to_string(),
                        "error: not found".into(),
                    );
                }
            };

            grounding.add(video_id, 0.0, duration_s.unwrap_or(0.0));

            let file_path: Option<String> = db
                .conn
                .query_row("SELECT path FROM video_files WHERE video_id = ?1 LIMIT 1", [video_id], |r| r.get(0))
                .ok();
            let file = file_path
                .as_ref()
                .map(|p| Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())
                .unwrap_or_default();

            #[derive(Serialize)]
            struct FrameSummary {
                t_s: f64,
                summary: String,
            }

            let mut all_frames: Vec<(f64, Option<String>)> = Vec::new();
            if let Ok(mut st) =
                db.conn.prepare("SELECT t_s, description_json FROM frames WHERE video_id = ?1 ORDER BY t_s")
                && let Ok(rows) =
                    st.query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, Option<String>>(1)?)))
            {
                for r in rows.flatten() {
                    all_frames.push(r);
                }
            }

            let range_start = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let range_end = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
            all_frames.retain(|(t, _)| *t >= range_start - 0.5 && *t <= range_end + 0.5);
            let max_sampled = if max_chars > LOCAL_TOOL_RESULT_CHARS { 40 } else { 12 };
            let sampled_raw = if all_frames.len() <= max_sampled {
                all_frames
            } else {
                let count = all_frames.len();
                let mut chosen = Vec::with_capacity(max_sampled);
                for i in 0..max_sampled {
                    let idx = (i * (count - 1)) / (max_sampled - 1);
                    chosen.push(all_frames[idx].clone());
                }
                chosen
            };

            let mut frames = Vec::new();
            for (t_s, desc_json) in sampled_raw {
                let summary_text = if let Some(j) = desc_json {
                    if let Ok(v) = serde_json::from_str::<Value>(&j) {
                        {
                            let desc_chars = if max_chars > LOCAL_TOOL_RESULT_CHARS { 300 } else { 120 };
                            let mut text: String = v
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .chars()
                                .take(desc_chars)
                                .collect();
                            if let Some(vt) = v.get("visible_text").and_then(|t| t.as_array()).filter(|a| !a.is_empty())
                            {
                                let vt: Vec<&str> = vt.iter().filter_map(|x| x.as_str()).collect();
                                text.push_str(&format!(" [text: {}]", vt.join(" | ")));
                            }
                            text
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                frames.push(FrameSummary { t_s: (t_s * 100.0).round() / 100.0, summary: summary_text });
            }

            let summary = format!("{}s, {} frames", duration_s.unwrap_or(0.0) as i64, frames.len());

            // Whether anyone speaks in the asked-for range. Without this the model has no evidence
            // either way and picks the first `audio` the schema offers — "source" — for scenery,
            // which then reads as "people speak here" and suppresses the beat's narration.
            let speech_end = if range_end == f64::MAX { duration_s.unwrap_or(0.0) } else { range_end };
            let has_speech = clip_has_speech(db, video_id, range_start, speech_end);
            // How steady the camera is here. No frame description mentions it, and a shaky shot
            // looks wrong in a cut whatever it shows.
            let measured = db.motion_windows(video_id).unwrap_or_default();
            // Judged against the clip's own ordinary level as well as the floor: a handheld
            // clip's usual stretches are what it is, the worse ones are what to avoid.
            let limit = crate::steadiness::shake_limit(&measured, max_shake, shake_relative);
            let camera = crate::steadiness::camera_style(&measured).as_str();
            let shaky_spans =
                crate::steadiness::shaky_spans_with_sway(&measured, range_start, speech_end, limit, max_sway);
            // Timestamps, not a verdict on the whole file: most of a shaky clip is usually fine,
            // and the editor is choosing a range, not a video.
            let shaky_at: Vec<String> = shaky_spans.iter().map(|(a, b)| format!("{a:.1}-{b:.1}")).collect();
            let steady = if measured.is_empty() {
                "unknown"
            } else if shaky_at.is_empty() {
                "yes"
            } else {
                "not everywhere — see shaky_at"
            };

            let mut obj = json!({
                "video_id": video_id,
                "file": file,
                "duration_s": duration_s,
                "fps": fps,
                "width": width,
                "height": height,
                "language": language,
                "has_speech": has_speech,
                "audio": if has_speech { "source" } else { "mute" },
                "steady": steady,
                "camera": camera,
                "shaky_at": shaky_at,
                "frames": frames,
            });

            // Drop frames until the JSON fits.
            while obj.to_string().len() > max_chars && !frames.is_empty() {
                frames.pop();
                obj["frames"] = json!(frames);
            }

            (obj.to_string(), summary)
        }
        "list_videos" => {
            let mut st = match db.conn.prepare(
                "SELECT DISTINCT v.id, v.duration_s, v.language, v.summary
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN folders f ON f.id = vf.folder_id
                 JOIN project_folders pf ON pf.folder_id = f.id
                 WHERE pf.project_id = ?1
                   AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
                 ORDER BY v.id",
            ) {
                Ok(s) => s,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db prepare".into()),
            };

            #[derive(Serialize)]
            struct VideoItem {
                video_id: i64,
                file: String,
                duration_s: Option<f64>,
                language: Option<String>,
                /// Someone talks in this video: it can carry a beat on its own audio.
                has_speech: bool,
                /// static, tripod, stabilised, handheld — or unknown before measuring.
                camera: &'static str,
                summary: String,
            }

            let rows = match st.query_map([project_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<f64>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            }) {
                Ok(rows) => rows,
                Err(e) => return (json!({"error": e.to_string()}).to_string(), "error: db query".into()),
            };

            let mut videos = Vec::new();
            for r in rows.flatten() {
                let (vid, duration_s, language, mut sum) = r;
                let file_path: Option<String> = db
                    .conn
                    .query_row("SELECT path FROM video_files WHERE video_id = ?1 LIMIT 1", [vid], |row| row.get(0))
                    .ok();
                let file = file_path
                    .as_ref()
                    .map(|p| Path::new(p).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())
                    .unwrap_or_default();

                if sum.as_ref().is_none_or(|s| s.trim().is_empty()) {
                    let frame_desc: Option<String> = db
                        .conn
                        .query_row(
                            "SELECT description_json FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s LIMIT 1",
                            [vid],
                            |row| row.get(0),
                        )
                        .ok();
                    if let Some(fd) = frame_desc
                        && let Ok(v) = serde_json::from_str::<Value>(&fd)
                    {
                        sum = v.get("description").and_then(|d| d.as_str()).map(|s| s.to_string());
                    }
                }

                if sum.as_ref().is_none_or(|s| s.trim().is_empty()) {
                    sum = db
                        .conn
                        .query_row(
                            "SELECT text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s LIMIT 1",
                            [vid],
                            |row| row.get(0),
                        )
                        .ok();
                }

                let summary_text: String = sum.unwrap_or_default().chars().take(150).collect();
                let has_speech = db
                    .conn
                    .query_row("SELECT EXISTS(SELECT 1 FROM transcript_segments WHERE video_id = ?1)", [vid], |r| {
                        r.get::<_, bool>(0)
                    })
                    .unwrap_or(false);
                let camera = crate::steadiness::camera_style(&db.motion_windows(vid).unwrap_or_default()).as_str();
                videos.push(VideoItem {
                    video_id: vid,
                    file,
                    duration_s,
                    language,
                    has_speech,
                    camera,
                    summary: summary_text,
                });
            }

            let summary = format!("{} videos", videos.len());
            let json_str = truncate_json_list(&mut videos, max_chars);
            (json_str, summary)
        }
        unknown => {
            (json!({"error": format!("unknown tool: {unknown}")}).to_string(), format!("error: unknown tool {unknown}"))
        }
    }
}

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
                    severity: IssueSeverity::Error,
                    beat_id: Some(beat.id.clone()),
                    clip_index: Some(clip_idx),
                    message: format!("video #{} does not belong to project", clip.video_id),
                });
            } else if !grounding.is_grounded(clip.video_id, clip.in_s, clip.out_s, cfg.grounding_slack_s) {
                issues.push(Issue {
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

/// Last resort when the model ignores the target: shorten every clip proportionally (keeping its
/// in point, never below `script.min_trimmed_clip_s`). Returns whether anything changed.
pub fn trim_to_target(script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    trim_to_target_with(script, |_| false, cfg)
}

/// Keep every bed inside the beat it plays under.
///
/// Beds are laid before the cut is fitted to its target, and fitting trims the pictures. A bed
/// left at its old length then outlives them: it plays on over the next beat's pictures, and the
/// export puts overlapping clips on A1. Whatever moved the clips, this puts the sound back inside
/// its beat.
fn clamp_beds_to_beats(script: &mut Script) {
    for beat in &mut script.beats {
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        match &mut beat.bed {
            Some(bed) if beat_len <= 0.0 => {
                let _ = bed;
                beat.bed = None;
            }
            Some(bed) if bed.duration_s() > beat_len => bed.out_s = bed.in_s + beat_len,
            _ => {}
        }
    }
}

/// [`trim_to_target`] that leaves `keep` clips (people speaking) whole and shortens the others.
/// Fit a cut to its target in one pass.
///
/// Three mechanisms used to shorten a script independently — speech capped to a share of the
/// target, b-roll trimmed proportionally, and a final pass doing both again — each measuring from
/// its own view of the total. Together they overshot badly: a cut sitting at 58.6 s against a 60 s
/// target came out at 29.5 s. One pass, measuring once, cannot do that.
///
/// Over target, the pictures give way first and the speaking clips only if that was not enough;
/// under it, the pictures are held longer. Speech is never stretched: a sentence is as long as it
/// is, and the rest is dead air.
fn fit_to_target(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let total = script.total_duration_s();
    let slack = target * (cfg.final_target_tolerance - 1.0);
    if (total - target).abs() <= slack {
        return false;
    }
    if total < target {
        let grew = grow_to_target(db, script, cfg);
        clamp_beds_to_beats(script);
        return grew;
    }

    let speaking: Vec<bool> =
        script.beats.iter().flat_map(|b| &b.clips).map(|c| clip_has_speech(db, c.video_id, c.in_s, c.out_s)).collect();
    let spoken: f64 = script
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .zip(&speaking)
        .filter(|(_, s)| **s)
        .map(|(c, _)| c.out_s - c.in_s)
        .sum();
    let pictures = total - spoken;

    // What the pictures must come down to, never below the floor for each shot.
    let picture_floor: f64 = speaking.iter().filter(|s| !**s).count() as f64 * cfg.min_trimmed_clip_s;
    let want_pictures = (target - spoken).max(picture_floor).min(pictures);
    let picture_factor = if pictures > 0.0 { want_pictures / pictures } else { 1.0 };

    // Speech is never scaled. Scaling it cut people off mid-sentence to hit a number: three
    // interview clips in a row ended inside a word. A sentence is the unit here, and a cut that
    // runs a few seconds long is worth more than one that lands exactly and sounds broken.
    let speech_factor = 1.0;

    let mut changed = false;
    let mut i = 0;
    for beat in &mut script.beats {
        for c in &mut beat.clips {
            let factor = if speaking[i] { speech_factor } else { picture_factor };
            i += 1;
            if factor >= 1.0 {
                continue;
            }
            let len = c.out_s - c.in_s;
            let new_len = (len * factor).max(cfg.min_trimmed_clip_s).min(len);
            if (new_len - len).abs() > 0.05 {
                c.out_s = c.in_s + new_len;
                changed = true;
            }
        }
    }
    clamp_beds_to_beats(script);
    changed
}

/// Hold the b-roll longer when the cut came in short.
///
/// Two independent mechanisms shorten a script — speech is capped at a share of the target, and
/// b-roll is trimmed proportionally — so together they undershoot, and a 60 s teaser lands at 49 s
/// with no way back. Speaking clips are left alone: they are as long as the sentence is. Scenery
/// can simply be held, up to what the footage actually has.
fn grow_to_target(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let total = script.total_duration_s();
    if total >= target * 0.97 {
        return false;
    }
    let mut room: Vec<(usize, usize, f64)> = Vec::new();
    for (bi, beat) in script.beats.iter().enumerate() {
        // A narrated beat can only be held as long as its voice-over lasts, or the picture runs
        // on in silence after the last word — the very gap content_issues complains about.
        let mut beat_room = f64::MAX;
        let words = beat.narration.as_deref().map(|n| n.split_whitespace().count()).unwrap_or(0);
        if words > 0 {
            let spoken_s = words as f64 / cfg.narration_words_per_s;
            let beat_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
            beat_room = (spoken_s - beat_s).max(0.0);
        }
        if beat_room <= 0.1 {
            continue;
        }
        let mut left = beat_room;
        for (ci, c) in beat.clips.iter().enumerate() {
            if left <= 0.1 {
                break;
            }
            if clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                continue;
            }
            let duration: Option<f64> = db
                .conn
                .query_row("SELECT duration_s FROM videos WHERE id = ?1", [c.video_id], |r| r.get(0))
                .ok()
                .flatten();
            let ceiling = duration.unwrap_or(c.out_s).min(c.in_s + cfg.max_clip_s);
            let spare = (ceiling - c.out_s).min(left);
            if spare > 0.1 {
                room.push((bi, ci, spare));
                left -= spare;
            }
        }
    }
    let spare_total: f64 = room.iter().map(|(_, _, s)| s).sum();
    if spare_total <= 0.0 {
        return false;
    }
    let needed = target - total;
    let share = (needed / spare_total).min(1.0);
    let mut changed = false;
    for (bi, ci, spare) in room {
        let add = spare * share;
        if add > 0.1 {
            script.beats[bi].clips[ci].out_s += add;
            changed = true;
        }
    }
    changed
}

fn trim_to_target_with(
    script: &mut Script,
    keep: impl Fn(&ScriptClip) -> bool,
    cfg: &crate::config::ScriptConfig,
) -> bool {
    trim_to_target_within(script, keep, cfg.target_overshoot, cfg)
}

/// `trim_to_target_with` with an explicit tolerance. The draft stage is loose — the model may yet
/// redraft — but the last pass before saving has no such luxury: whatever it leaves is the length
/// the user gets, and a cut left 18% long is one the report then complains about.
fn trim_to_target_within(
    script: &mut Script,
    keep: impl Fn(&ScriptClip) -> bool,
    tolerance: f64,
    cfg: &crate::config::ScriptConfig,
) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let total = script.total_duration_s();
    if total <= target * tolerance {
        return false;
    }
    let kept: f64 = script.beats.iter().flat_map(|b| &b.clips).filter(|c| keep(c)).map(|c| c.out_s - c.in_s).sum();
    let flexible = total - kept;
    if flexible <= 0.0 {
        return false;
    }
    let factor = ((target - kept) / flexible).clamp(0.0, 1.0);
    let mut changed = false;
    for clip in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        if keep(clip) {
            continue;
        }
        let len = clip.out_s - clip.in_s;
        let new_len = (len * factor).max(cfg.min_trimmed_clip_s).min(len);
        if new_len < len {
            clip.out_s = clip.in_s + new_len;
            changed = true;
        }
    }
    changed
}

fn video_duration(db: &Db, video_id: i64) -> Option<f64> {
    db.conn
        .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
        .ok()
        .flatten()
}

/// Keep clips inside their video; drop the ones that start past its end.
fn clamp_to_duration(db: &Db, script: &mut Script) {
    for beat in &mut script.beats {
        beat.clips.retain_mut(|c| {
            if let Some(d) = video_duration(db, c.video_id) {
                c.out_s = c.out_s.min(d);
                c.in_s = c.in_s.max(0.0);
            }
            c.out_s - c.in_s >= 1.0
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
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

/// A later clip overlapping footage an earlier clip already showed (by more than 1 s) is dropped.
fn drop_repeated_footage(script: &mut Script) -> usize {
    let mut used: Vec<(i64, f64, f64)> = Vec::new();
    let mut dropped = 0;
    for beat in &mut script.beats {
        beat.clips.retain(|c| {
            let repeat = used.iter().any(|&(v, a, b)| v == c.video_id && c.out_s.min(b) - c.in_s.max(a) > 1.0);
            if repeat {
                dropped += 1;
            } else {
                used.push((c.video_id, c.in_s, c.out_s));
            }
            !repeat
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
    dropped
}

/// Back-to-back clips of the same video in one beat (`0-6`, `6-16`, `16-23`) are jump cuts inside a
/// single continuous take: join them. Returns how many cuts were removed.
fn merge_contiguous_clips(script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut merged = 0;
    for beat in &mut script.beats {
        let mut out: Vec<ScriptClip> = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if let Some(prev) = out.last_mut()
                && prev.video_id == c.video_id
                && (c.in_s - prev.out_s).abs() <= 0.5
                && c.out_s - prev.in_s <= cfg.max_clip_s
            {
                prev.out_s = prev.out_s.max(c.out_s);
                if c.audio == crate::script::Audio::Source {
                    prev.audio = crate::script::Audio::Source;
                }
                merged += 1;
                continue;
            }
            out.push(c);
        }
        beat.clips = out;
    }
    merged
}

/// Default editing instructions. Users can replace them in Settings (`[chat] system_prompt`);
/// `{project}`, `{fps}`, `{width}` and `{height}` are filled in. The tool list and the clip-range
/// rule are always appended, since scripts can't be built without them.
pub const DEFAULT_EDITOR_PROMPT: &str = "You are a senior documentary and promo video editor working on project \"{project}\" \
({fps} fps, {width}x{height}). You cut real footage into a watchable, well-paced story and write the voice-over for it.

HOW THIS WORKS
The whole project is in front of you before you call anything: WHAT PEOPLE SAY has every word spoken, with the timestamps to cut on. Work in this order.
 a. Read it. Decide whose sentences carry the story and copy their timestamps exactly.
 b. For each of those, find a picture of what is being described: search_moments for the subject, then get_video to see the keyframes and the shaky stretches. A range you have not seen returned is a guess, and guesses are dropped.
 c. Lay the beat out: the speaker's face first, then cut to the picture and give the beat a \"bed\" so their voice keeps running under it.
 d. Add up the clip lengths and fix the total yourself before you answer.
 e. Answer with one JSON object and nothing else.
Before you answer, check: every range came from WHAT PEOPLE SAY or from a tool; every clip with \"source\" audio starts and ends where a sentence does; no clip is under 4 s; the total is within 10% of the target; every beat has a purpose that moves the story on.

HOW TO EDIT
1. Understand the material first: list the videos, look at the keyframes of the promising ones, and read the transcripts of the videos where people talk.
2. Build one story out of what the footage actually has, not a list of nice moments: a hook, 3-6 beats that each make one point, and an ending that lands. Each beat follows from the one before - someone names the place, someone says what it is like to live there, someone shows what that looks like. If two clips could swap places without anyone noticing, the piece has no story yet. Every beat has a purpose, and the purpose says how it moves the story on.
3. Choose only strong shots: a clear subject (people, a landmark, a building, a sign, activity, a striking view). Skip footage whose keyframes describe black or blank frames, blur, transitions, the ground or sky only, empty hillsides with nothing to see, or the same view as the previous clip.
4. Pacing - slower is better than frantic:
   - hold every shot at least 4 s so viewers can see it and read any text; views and b-roll 5-10 s;
   - when someone speaks, keep the clip from just before their first word to the end of their sentences (use the transcript timestamps; up to ~25 s) with audio \"source\", and never cut the moment they stop: hold 1-2 s of the person on screen after the last word;
   - prefer fewer, longer clips over many quick cuts; never jump between unrelated shots every 2 s.
5. Audio: use \"source\" when a person is speaking in the clip; use \"mute\" for scenery and b-roll so wind and handling noise don't play under the voice-over.
6. Every search hit and video says what the camera is doing: static, tripod, stabilised or handheld, and a hit marked shaky sits on a stretch the camera shakes through - do not cut from it. list_videos and get_video report the camera work too. When two clips cover the same moment, prefer the mounted or stabilised one. get_video also lists a clip's shaky stretches as shaky_at timestamps (\"12.0-16.0\") - the parts worse than that clip's own ordinary level. A shaky shot looks wrong in a finished cut whatever it shows: cut around those stretches rather than dropping the video, since the rest of it is usually fine. Use a shaky range only when nothing else covers the moment, and keep it short when you do.
7. When the footage has people talking on camera (interviews), build the story out of what they say. Everything anyone says is already in front of you under WHAT PEOPLE SAY - read it before you decide anything, and choose whose words carry the piece from all of the tapes, not from the first speaker you come across. Use get_transcript only for a tape listed as left out there. Cut the clip to whole sentences and set that clip's audio to \"source\". A talking head is not b-roll - never mute someone mid-sentence to speak over them, and never write narration for a beat whose clips use \"source\" audio. Voice-over is for the scenery between what people say, not a replacement for it.
8. Do not sit on a talking head for a whole beat, and do not bury them either. Show the person first - long enough for a viewer to take in their face, a few seconds - then cut to a picture of what they are describing while they keep talking, and come back to them if the point lands on them. The viewer should recognise that face when it returns later in the piece. To keep a voice running while the picture changes, give the beat a \"bed\": {video_id, in_s, out_s} naming the stretch of speech that carries the whole beat. The beat's clips are then pictures only - their own audio is not played - and you can cut between as many as you like without interrupting the speaker. Use a bed whenever an answer continues under a cutaway; the range must be one speaker talking, on the microphone, ending on a whole sentence. Without a bed the sound stops dead the moment you cut away.
9. get_transcript marks segments spoken away from the microphone: in an interview those are the questions and the slate, not the answers. Never start a clip on one and never build a beat around one - cut to where the person answers. They sound as far away as they were.
10. Use what people say deliberately: a statement, an explanation, a line with a point to it. Chatter, half-sentences, thinking aloud, banter between takes and answers that go nowhere do not belong in a teaser, however clearly they are recorded - unless the user asks for that kind of material.
11. How much of the piece is people talking is your decision, and it follows what the user asked for: a teaser built on what people say can be almost all interview, a scenic one almost none. Cutting to a picture of what is being described is usually better than staying on a face for a long time - but do it because it helps the story, not to hit a quota.
12. Narration (voice-over) must cover the beat: about 2.5 spoken words per second of the beat's clips (a 20 s beat needs ~50 words). Set narration to \"\" for beats where people speak on camera (never copy their words into the narration). Every beat has its own narration; never repeat text from another beat, and never reuse the same footage twice. Write natural, specific sentences about what is on screen and why it matters; no filler. on_screen_text is short (a title or a name).
13. Length: the clips add up to the length the user asked for; set target_duration_s to it. Give it a little more than asked - about 10% - and pick one more moment than you think you need: a cut that comes in long is trimmed to fit, but a cut that comes in short can only be fixed by holding shots after the voice-over has stopped, which looks like a mistake. If the user gave no length, choose what the material supports (usually 60-180 s).
14. You can answer in words instead of drafting: reply with {\"action\":\"reply\",\"text\":...} when the request is ambiguous and one question would settle it, when a message is not about the video (notes, something pasted by mistake), or when the footage cannot support what was asked - say what is missing. A reply leaves the previous version alone, which is better than rebuilding it around a guess. Do not reply to avoid work: when the request is clear, draft.
15a. A line marked [applied after drafting] in an earlier reply is a change already made to the saved script - a bed laid, a clip moved off a shaky stretch, a range dropped. It is done: build on it rather than undoing it, and do not make the same mistake again in this session.
15. The user's feedback overrides these defaults. When revising, change what they asked for (slower, longer, different shots, more narration) and keep what they didn't mention; never return the previous draft unchanged.
16. Always reply in the user's language.
";

/// Construct the system prompt for the editor agent: the editing instructions (`custom` or the
/// default), the fixed tool contract, and the current draft.
pub fn build_system_prompt(project: &Project, latest_script_json: Option<&str>, custom: Option<&str>) -> String {
    let template = custom.map(str::trim).filter(|c| !c.is_empty()).unwrap_or(DEFAULT_EDITOR_PROMPT);
    let fps = if project.fps_den == 1 {
        project.fps_num.to_string()
    } else {
        format!("{:.3}", project.fps_num as f64 / project.fps_den as f64)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    };
    let mut prompt = template
        .replace("{project}", &project.name)
        .replace("{fps}", &fps)
        .replace("{width}", &project.width.to_string())
        .replace("{height}", &project.height.to_string());
    prompt.push_str(
        "\n\nTOOLS (use them generously before writing; your clips can only come from what they return)\n\
         - list_videos(): every video with a short summary.\n\
         - search_moments(query, limit): find moments by meaning or keyword across speech, on-screen text and visuals.\n\
         - get_video(video_id, start_s, end_s): keyframe descriptions - what is actually visible and when.\n\
         - get_transcript(video_id, start_s, end_s): what people say, with timestamps - only needed for a tape \
         listed as left out of WHAT PEOPLE SAY below.\n\n\
         CLIP RANGES: in_s/out_s must lie inside ranges returned by the tools; never use a video_id or range you \
         have not inspected. Search results are search windows, not clips: cut a sub-range out of them.\n",
    );

    if let Some(draft) = latest_script_json {
        prompt.push_str(&format!(
            "\nCurrent script draft from this session:\n{}\n\
             When the user asks for changes, revise this draft accordingly.\n",
            draft
        ));
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

/// Spoken voice-over rate used to check that narration fills its beat.

/// Narration covering less than this share of its beat triggers a redraft.

/// Editorial problems the model can fix in a redraft: narration too short for its beat, and clips
/// over footage the tools know nothing about (no speech and no described keyframe nearby).
pub fn content_issues(db: &Db, script: &Script, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut seen_narration: Vec<String> = Vec::new();
    for beat in &script.beats {
        let Some(n) = beat.narration.as_deref().map(str::trim).filter(|n| !n.is_empty()) else { continue };
        let key = n.to_lowercase();
        if seen_narration.contains(&key) {
            issues.push(Issue {
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

fn clip_has_speech(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3)",
            params![video_id, in_s, out_s],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
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

/// Silence kept after the last words of a speaking clip, so the cut doesn't clip the person off.

/// Breath kept before the first words.

/// Never extend a clip further than this to finish a sentence.

/// Make every clip that carries someone's voice begin and end on a whole sentence.
///
/// Snapping only reaches 0.75 s and the fit that runs last can trim far more than that, so a clip
/// ended wherever the arithmetic landed: four in one run stopped between 0.27 s and 2.65 s before
/// the speaker finished, one of them mid-word. The length is a target; a sentence is not.
///
/// An edge moves out to the sentence boundary when that is within `max_speech_extend_s`, and back
/// to the previous one when it is further, so a clip is never left in the middle of a thought.
fn end_on_sentences(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut fixed = 0usize;
    for c in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        if c.audio != crate::script::Audio::Source {
            continue;
        }
        let Ok(mut st) = db
            .conn
            .prepare_cached("SELECT start_s, end_s FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")
        else {
            continue;
        };
        let segs: Vec<(f64, f64)> = st
            .query_map([c.video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?)))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default();
        let overlapping: Vec<usize> =
            (0..segs.len()).filter(|&i| segs[i].1 > c.in_s + 0.05 && segs[i].0 < c.out_s - 0.05).collect();
        let (Some(&first), Some(&last)) = (overlapping.first(), overlapping.last()) else { continue };
        let duration = video_duration(db, c.video_id).unwrap_or(f64::MAX);
        let (old_in, old_out) = (c.in_s, c.out_s);

        // The end. Run on to the last word, or fall back to where the previous sentence stopped.
        let sentence_end = segs[last].1;
        if sentence_end > c.out_s + 0.05 {
            let wanted = (sentence_end + cfg.speech_overrun_s).min(duration);
            if wanted - c.out_s <= cfg.max_speech_extend_s {
                c.out_s = wanted;
            } else {
                let back = overlapping
                    .iter()
                    .rev()
                    .map(|&i| segs[i].1)
                    .find(|&e| e <= c.out_s + 0.05 && e - c.in_s >= cfg.min_trimmed_clip_s);
                match back {
                    Some(end) => c.out_s = (end + cfg.speech_overrun_s).min(duration),
                    // Nothing to fall back to: a whole sentence long is better than half of one.
                    None => c.out_s = wanted,
                }
            }
        }

        // The start. Never open in the middle of a sentence.
        let sentence_start = segs[first].0;
        if sentence_start < c.in_s - 0.05 {
            if c.in_s - sentence_start <= cfg.max_speech_extend_s {
                c.in_s = (sentence_start - cfg.speech_lead_s).max(0.0);
            } else if let Some(&next) = overlapping.iter().find(|&&i| segs[i].0 >= c.in_s) {
                if c.out_s - segs[next].0 >= cfg.min_trimmed_clip_s {
                    c.in_s = (segs[next].0 - cfg.speech_lead_s).max(0.0);
                }
            }
        }

        if (c.in_s - old_in).abs() > 0.05 || (c.out_s - old_out).abs() > 0.05 {
            fixed += 1;
        }
    }
    fixed
}

/// Let speaking clips breathe: finish the sentence the clip is in, then hold ~1.5 s of the person
/// before cutting (without running into their next sentence), and start slightly before the first
/// words. Runs after [`snap_to_segments`], which lands cuts exactly on segment boundaries.
fn pad_speech(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut changed = 0;
    for c in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        let Ok(mut st) = db
            .conn
            .prepare_cached("SELECT start_s, end_s FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")
        else {
            continue;
        };
        let segs: Vec<(f64, f64)> = st
            .query_map([c.video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?)))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default();
        let overlapping: Vec<usize> =
            (0..segs.len()).filter(|&i| segs[i].1 > c.in_s + 0.05 && segs[i].0 < c.out_s - 0.05).collect();
        let (Some(&first), Some(&last)) = (overlapping.first(), overlapping.last()) else { continue };
        let duration = video_duration(db, c.video_id).unwrap_or(f64::MAX);
        let (old_in, old_out) = (c.in_s, c.out_s);

        let speech_end = segs[last].1;
        let mut out = speech_end + cfg.speech_tail_s;
        if let Some(next) = segs.get(last + 1) {
            // Stop short of the next sentence — but when the speaker runs straight on there is
            // no pause to stop in, and clipping at the final sample cuts the last word's decay.
            // A little overrun is the difference between a clean end and a chopped one.
            let room = (next.0 - 0.3).max(speech_end + cfg.speech_overrun_s);
            out = out.min(room);
        }
        let out = out.min(old_out + cfg.max_speech_extend_s).min(duration);
        if out > c.out_s {
            c.out_s = out;
        }

        let speech_start = segs[first].0;
        let mut lead = (speech_start - cfg.speech_lead_s).max(0.0);
        if first > 0 {
            lead = lead.max(segs[first - 1].1);
        }
        // Also pulls a clip that starts mid-sentence back to the start of that sentence.
        if c.in_s > lead {
            c.in_s = lead.max(old_in - cfg.max_speech_extend_s);
        }
        if (c.in_s, c.out_s) != (old_in, old_out) {
            changed += 1;
        }
    }
    changed
}

/// Mute clips nobody speaks in when their beat has narration: ambient wind and handling noise
/// shouldn't play under the voice-over.
fn mute_silent_clips(db: &Db, script: &mut Script) -> usize {
    let mut muted = 0;
    for beat in &mut script.beats {
        if beat.narration.as_deref().is_none_or(|n| n.trim().is_empty()) {
            continue;
        }
        for c in &mut beat.clips {
            if c.audio == crate::script::Audio::Source && !clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                c.audio = crate::script::Audio::Mute;
                muted += 1;
            }
        }
    }
    muted
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

/// List all chat sessions for a project ordered by most recently updated.
/// Delete a chat session and its messages. Scripts drafted in it are kept and merely lose the
/// link (the schema sets their `session_id` to NULL): a script is a deliverable, a chat is not.
pub fn delete_session(db: &Db, session_id: i64) -> Result<bool, Error> {
    let n = db.conn.execute("DELETE FROM chat_sessions WHERE id = ?1", [session_id])?;
    Ok(n > 0)
}

pub fn sessions(db: &Db, project_id: i64) -> Result<Vec<ChatSession>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, project_id, title, created_at, updated_at
         FROM chat_sessions WHERE project_id = ?1 ORDER BY updated_at DESC",
    )?;
    let rows = st.query_map([project_id], |r| {
        Ok(ChatSession {
            id: r.get(0)?,
            project_id: r.get(1)?,
            title: r.get(2)?,
            created_at: r.get(3)?,
            updated_at: r.get(4)?,
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
        "SELECT id, session_id, role, content, tool_calls_json, created_at
         FROM chat_messages WHERE session_id = ?1 ORDER BY id ASC",
    )?;
    let rows = st.query_map([session_id], |r| {
        let id: i64 = r.get(0)?;
        let sid: i64 = r.get(1)?;
        let role: String = r.get(2)?;
        let content: String = r.get(3)?;
        let tool_calls_json: Option<String> = r.get(4)?;
        let created_at: i64 = r.get(5)?;

        let mut tool_calls = None;
        let mut script_id = None;

        if let Some(json_str) = tool_calls_json
            && let Ok(val) = serde_json::from_str::<Value>(&json_str)
        {
            if let Some(arr) = val.as_array() {
                tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(Value::Array(arr.clone())).ok();
            } else if let Some(obj) = val.as_object() {
                script_id = obj.get("script_id").and_then(|v| v.as_i64());
                if let Some(tc) = obj.get("tool_calls") {
                    tool_calls = serde_json::from_value::<Vec<ToolCallRecord>>(tc.clone()).ok();
                }
            }
        }

        Ok(ChatMessage { id, session_id: sid, role, content, tool_calls, script_id, created_at })
    })?;

    Ok(rows.filter_map(Result::ok).collect())
}

/// Run one turn of the script chat agent.
pub async fn run_turn(
    ctx: &mut ChatContext,
    project_id: i64,
    session_id: Option<i64>,
    message: &str,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
) -> Result<TurnResult, Error> {
    let project = ctx.db.project(project_id)?;
    // A configured budget wins; otherwise the backend decides how much research it can afford.
    // Captured before the backend is borrowed, so every arm can check it.
    let cancel_flag = ctx.cancel.clone();
    let cancelled = move || cancel_flag.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::SeqCst));
    let rounds_budget = match (ctx.max_tool_rounds, &ctx.backend) {
        (n, _) if n > 0 => n,
        (_, ChatBackend::Local { .. }) => ctx.script.local_tool_rounds,
        _ => ctx.script.roomy_tool_rounds,
    };

    let (session_id, is_new_session) = match session_id {
        Some(id) => (id, false),
        None => {
            let title: String = message.chars().take(60).collect();
            let sid = create_session(&ctx.db, project_id, &title)?;
            (sid, true)
        }
    };

    let mut grounding = Grounding::default();
    let mut prior_messages = Vec::new();
    let mut latest_script_json = None;

    if !is_new_session {
        prior_messages = messages(&ctx.db, session_id)?;
        for m in &prior_messages {
            if let Some(calls) = &m.tool_calls {
                for c in calls {
                    grounding.record_tool_call(&c.tool, &c.args, &ctx.db);
                }
            }
        }
        let stored_json: Option<String> = ctx
            .db
            .conn
            .query_row(
                "SELECT script_json FROM scripts WHERE session_id = ?1 ORDER BY version DESC LIMIT 1",
                [session_id],
                |r| r.get(0),
            )
            .ok();
        if let Some(ref json_str) = stored_json
            && let Ok(prev_script) = serde_json::from_str::<Script>(json_str)
        {
            for beat in &prev_script.beats {
                for clip in &beat.clips {
                    grounding.add(clip.video_id, clip.in_s, clip.out_s);
                }
            }
        }
        latest_script_json = stored_json;
    }

    // Write down what was asked before doing any of it. Everything used to be stored when the turn
    // finished, so a chat looked empty if you left the panel and came back while it ran — the
    // message lived only in the window's own state — and closing the app mid-turn lost it
    // altogether. After `prior_messages` is read, so it is not replayed twice.
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'user', ?2, NULL, ?3)",
        params![session_id, message, now()],
    )?;

    let mut sys_prompt = build_system_prompt(&project, latest_script_json.as_deref(), ctx.system_prompt.as_deref());
    let reference_chars = match &ctx.backend {
        ChatBackend::Server { .. } => 6000,
        // Roughly an eighth of the window (~4 chars per token), so results leave room for the draft.
        ChatBackend::Local { ctx_tokens, .. } => ((*ctx_tokens as usize) * 4 / 8).clamp(1500, 12000),
        // CLI: treat like a large-context backend; tool results are limited by LOCAL_TOOL_RESULT_CHARS anyway.
        ChatBackend::Cli(_) => 6000,
    };
    sys_prompt.push_str(&reference_edits_text(&ctx.db, project_id, reference_chars));
    // The tapes themselves, before any tool is called. A whole project's speech is a few thousand
    // tokens; a model that has to ask for each transcript reads two and builds the teaser out of
    // whoever it happened to find there.
    let speech_chars = match &ctx.backend {
        ChatBackend::Server { .. } | ChatBackend::Cli(_) => 60_000,
        // Half the window at ~4 chars a token, leaving the other half for tools and the draft.
        ChatBackend::Local { ctx_tokens, .. } => ((*ctx_tokens as usize) * 4 / 2).clamp(4_000, 60_000),
    };
    if ctx.script.speech_in_prompt {
        sys_prompt.push_str(&speech_digest(&ctx.db, project_id, speech_chars));
    }
    // A length the user states ("60 second promo", "2 minutos") wins over whatever the model sets.
    let requested_s = requested_duration_s(message);
    // Only a first draft (or an explicit length) is squeezed to its target; revisions follow feedback.
    let enforce_target = latest_script_json.is_none() || requested_s.is_some();

    let mut tool_records = Vec::new();
    let mut raw_reply = String::new();
    let mut parsed_script: Option<Script> = None;
    let mut pre_issues: Vec<Issue> = Vec::new();

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
                    req_messages.push(json!({ "role": &pm.role, "content": &pm.content }));
                }
            }
            req_messages.push(json!({ "role": "user", "content": message }));

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            let mut last_assistant_text = String::new();

            // Three quarters of the window for the conversation; the draft needs the rest.
            let round_budget = (*ctx_tokens as usize).saturating_mul(3) / 4;
            while tool_rounds < rounds_budget {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                if !make_room(&mut req_messages, round_budget) {
                    // Nothing left to forget: stop looking and write the script.
                    break;
                }
                tool_rounds += 1;
                let mut body = json!({
                    "messages": req_messages,
                    // A tool call is short, but a model that reasons first spends the same budget
                    // on the thought, so it gets the same ceiling.
                    "max_tokens": ctx.script.max_answer_tokens,
                    "tools": tools_definition(),
                    "tool_choice": "auto",
                    "temperature": 0.3,
                    "chat_template_kwargs": { "enable_thinking": false },
                });
                if !model.is_empty() {
                    body["model"] = json!(model);
                }

                let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&body);
                if !api_key.is_empty() {
                    req = req.bearer_auth(api_key.as_str());
                }

                let resp = req.send().await.map_err(|e| Error::Vision(format!("server at {url}: {e}")))?;
                if !resp.status().is_success() {
                    let err_text = resp.text().await.unwrap_or_default();
                    return Err(Error::Vision(format!("server HTTP error: {err_text}")));
                }

                let v: Value = resp.json().await.map_err(|e| Error::Vision(format!("bad response JSON: {e}")))?;
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

                if empty_searches >= EMPTY_SEARCHES_BEFORE_HINT && !hinted {
                    hinted = true;
                    req_messages.push(json!({ "role": "user", "content": EMPTY_SEARCH_HINT }));
                }

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

                    on_event(ChatEvent::ToolStarted { tool: tool_name.clone(), args: tool_args.clone() });

                    let vector = if tool_name == "search_moments" {
                        if let (Some(e), Some(q)) =
                            (ctx.embedder.as_mut(), tool_args.get("query").and_then(|v| v.as_str()))
                        {
                            query_vector(e, q).await.ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let (result_str, summary) = dispatch_tool_limited(
                        &ctx.db,
                        &ctx.data_dir,
                        project_id,
                        &tool_name,
                        &tool_args,
                        &mut grounding,
                        vector.as_deref(),
                        ctx.script.roomy_tool_result_chars,
                        ctx.script.max_shake_jerk,
                        ctx.script.shake_relative,
                        ctx.script.max_sway,
                    );

                    on_event(ChatEvent::ToolFinished { tool: tool_name.clone(), summary: summary.clone() });

                    let empty = is_empty_search(&tool_name, &summary);
                    let was_search = tool_name.starts_with("search");
                    tool_records.push(ToolCallRecord { tool: tool_name, args: tool_args, summary });

                    if empty {
                        empty_searches += 1;
                    } else if !was_search {
                        empty_searches = 0;
                    }

                    req_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str,
                    }));
                }
            }

            // Drafting final script call
            on_event(ChatEvent::Drafting);

            req_messages.push(json!({
                "role": "user",
                "content": format!(
                    "Now produce the final complete Script JSON using only the footage returned by the tools. \
                     Follow the pacing rules, and apply this request from the user: {message}"
                )
            }));

            let mut final_body = json!({
                "messages": req_messages,
                "temperature": 0.3,
                // Deliberately generous: a long script must never be cut off. The cap only stops
                // a model that will not stop at all, so the turn fails as itself rather than as a
                // dead socket — Bonsai-27B at 1 bit produced 11 347 tokens of one draft that way.
                "max_tokens": ctx.script.max_answer_tokens,
                "chat_template_kwargs": { "enable_thinking": false },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "script",
                        "strict": true,
                        "schema": script_json_schema()
                    }
                }
            });
            if !model.is_empty() {
                final_body["model"] = json!(model);
            }

            let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&final_body);
            if !api_key.is_empty() {
                req = req.bearer_auth(api_key.as_str());
            }
            let resp = req.send().await.map_err(|e| Error::Vision(format!("server at {url}: {e}")))?;
            if !resp.status().is_success() {
                let err_text = resp.text().await.unwrap_or_default();
                return Err(Error::Vision(format!("server HTTP error (final script): {err_text}")));
            }
            let v: Value = resp.json().await.map_err(|e| Error::Vision(format!("bad final JSON: {e}")))?;
            let content = v["choices"][0]["message"]["content"].as_str().unwrap_or_default();

            let mut script_res = Script::parse_for_project(content, &project);
            if let (Ok(s), Some(t)) = (&mut script_res, requested_s) {
                s.target_duration_s = Some(t);
            }

            // Grounding + pacing check, redraft once on the server backend
            on_event(ChatEvent::Validating);

            let redraft_reasons = match &script_res {
                Ok(s) => {
                    let mut i = check_grounding(&ctx.db, project_id, s, &grounding, &ctx.script);
                    i.extend(pacing_issues(s, enforce_target, &ctx.script));
                    i.extend(content_issues(&ctx.db, s, &ctx.script));
                    i
                }
                Err(_) => Vec::new(),
            };

            if !redraft_reasons.is_empty() && script_res.is_ok() {
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
                final_body["messages"] = json!(req_messages);

                let mut retry_req =
                    client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&final_body);
                if !api_key.is_empty() {
                    retry_req = retry_req.bearer_auth(api_key.as_str());
                }
                if let Ok(retry_resp) = retry_req.send().await
                    && let Ok(rv) = retry_resp.json::<Value>().await
                {
                    let retry_content = rv["choices"][0]["message"]["content"].as_str().unwrap_or_default();
                    if let Ok(rescript) = Script::parse_for_project(retry_content, &project) {
                        script_res = Ok(rescript);
                    }
                }
            }

            if let Ok(mut s) = script_res {
                if requested_s.is_some() {
                    s.target_duration_s = requested_s;
                }
                pre_issues =
                    enforce_grounding_and_pacing(&ctx.db, project_id, &mut s, &grounding, enforce_target, &ctx.script);
                parsed_script = Some(s);
            }

            raw_reply = last_assistant_text;
        }
        ChatBackend::Local { helper, ctx_tokens, think } => {
            let mut transcript = format!("<|im_start|>system\n{sys_prompt}<|im_end|>\n");
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    transcript.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", pm.role, pm.content));
                }
            }
            transcript.push_str(&format!("<|im_start|>user\n{message}<|im_end|>\n"));

            // A full script's JSON is far longer than a tool action, so the draft call gets its
            // own budget out of the configured context rather than the default.
            // Reasoning is spent from the same budget as the answer, so leave room for both.
            let script_token_budget = (*ctx_tokens as usize / 2).clamp(4096, 12288);

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            while tool_rounds < rounds_budget {
                if cancelled() {
                    // The generation runs inside the helper process; ending the turn means
                    // ending it, or the model keeps going and Stop waits for it.
                    helper.kill().await;
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                tool_rounds += 1;
                // A tool action is short, but a model that reasons first spends the same budget on
                // the thought — at the default 2048 it ran out mid-round and the turn died.
                let out_str =
                    helper.complete_full(&transcript, Some(local_action_schema()), script_token_budget, *think).await?;
                let action: Result<LocalAction, _> = serde_json::from_str(&out_str);
                match action {
                    // Answering in words is a complete turn: nothing is drafted and the previous
                    // version stays as it is.
                    Ok(LocalAction::Reply { text }) => {
                        raw_reply = text;
                        break;
                    }
                    Ok(LocalAction::Tool { tool, args }) => {
                        on_event(ChatEvent::ToolStarted { tool: tool.clone(), args: args.clone() });
                        let vector = if tool == "search_moments" {
                            if let (Some(e), Some(q)) =
                                (ctx.embedder.as_mut(), args.get("query").and_then(|v| v.as_str()))
                            {
                                query_vector(e, q).await.ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let (res, summary) = dispatch_tool(
                            &ctx.db,
                            &ctx.data_dir,
                            project_id,
                            &tool,
                            &args,
                            &mut grounding,
                            vector.as_deref(),
                            &ctx.script,
                        );
                        on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: summary.clone() });
                        let empty = is_empty_search(&tool, &summary);
                        tool_records.push(ToolCallRecord { tool: tool.clone(), args: args.clone(), summary });

                        if empty {
                            empty_searches += 1;
                        } else if !tool.starts_with("search") {
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
                    }
                    Ok(LocalAction::Final { script }) => {
                        let mut s = script;
                        s.fill_from_project(&project);
                        parsed_script = Some(s);
                        break;
                    }
                    Err(e) => {
                        return Err(Error::Invalid(format!("local helper output did not match action schema: {e}")));
                    }
                }
            }

            if parsed_script.is_none() {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                on_event(ChatEvent::Drafting);
                transcript.push_str(&format!(
                    "<|im_start|>user\n{}Produce the final script action.<|im_end|>\n",
                    allowed_clips_text(&grounding)
                ));
                let final_str = helper
                    .complete_full(&transcript, Some(local_final_action_schema()), script_token_budget, *think)
                    .await?;
                debug_dump("final-draft", &transcript, &final_str);
                // Failing to parse here used to leave `parsed_script` as None, which surfaced as
                // the bland "unable to assemble a script" reply with no hint of what went wrong —
                // the same silence a mid-loop parse failure is loud about.
                match serde_json::from_str::<LocalAction>(&final_str) {
                    Ok(LocalAction::Final { mut script }) => {
                        script.fill_from_project(&project);
                        parsed_script = Some(script);
                    }
                    Ok(LocalAction::Reply { text }) => {
                        raw_reply = text;
                    }
                    Ok(LocalAction::Tool { .. }) => {
                        return Err(Error::Invalid(
                            "the local model asked for another tool instead of drafting the script".into(),
                        ));
                    }
                    Err(e) => {
                        return Err(Error::Invalid(format!(
                            "the local model's final script was not valid JSON: {e}: {}",
                            final_str.chars().take(200).collect::<String>()
                        )));
                    }
                }
            }

            on_event(ChatEvent::Validating);

            // Redraft once against the same checks the server backend uses. Without this the
            // local model's first answer is final, and its usual miss — beats with a purpose but
            // no narration — reached the user as a finished script full of silent beats.
            if let Some(s) = &parsed_script {
                let mut reasons = check_grounding(&ctx.db, project_id, s, &grounding, &ctx.script);
                reasons.extend(pacing_issues(s, enforce_target, &ctx.script));
                reasons.extend(content_issues(&ctx.db, s, &ctx.script));
                if !reasons.is_empty() {
                    on_event(ChatEvent::Drafting);
                    let issue_text: Vec<String> = reasons.iter().map(|i| i.message.clone()).collect();
                    transcript.push_str(&format!(
                        "<|im_start|>assistant\n{}<|im_end|>\n",
                        serde_json::to_string(&LocalAction::Final { script: s.clone() }).unwrap_or_default()
                    ));
                    transcript.push_str(&format!(
                        "<|im_start|>user\nFix these problems and return the complete corrected script action:\n{}\n\
                         {}Keep applying the user's request: {message}<|im_end|>\n",
                        issue_text.join("\n"),
                        allowed_clips_text(&grounding)
                    ));
                    if let Ok(retry_str) = helper
                        .complete_full(&transcript, Some(local_final_action_schema()), script_token_budget, *think)
                        .await
                        && let Ok(LocalAction::Final { mut script }) = serde_json::from_str::<LocalAction>(&retry_str)
                    {
                        script.fill_from_project(&project);
                        parsed_script = Some(script);
                    }
                }
            }

            if let Some(s) = &mut parsed_script {
                if requested_s.is_some() {
                    s.target_duration_s = requested_s;
                }
                pre_issues =
                    enforce_grounding_and_pacing(&ctx.db, project_id, s, &grounding, enforce_target, &ctx.script);
                // Muting happens above, so by now the beats that need a voice-over are known.
                let jobs = narration_jobs(&ctx.db, s, &ctx.script);
                let filled = fill_missing_narration(helper, s, jobs).await;
                if filled > 0 {
                    pre_issues.push(Issue {
                        severity: IssueSeverity::Info,
                        beat_id: None,
                        clip_index: None,
                        message: format!("wrote narration for {filled} silent beat(s)"),
                    });
                }
            }
        }
        ChatBackend::Cli(agent) => {
            // Drive the same local action schema loop as the Local backend.
            // The agent CLI receives the whole transcript as a single prompt each turn.
            let mut transcript = format!("<|im_start|>system\n{sys_prompt}<|im_end|>\n");
            for pm in &prior_messages {
                if pm.role == "user" || pm.role == "assistant" {
                    transcript.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", pm.role, pm.content));
                }
            }
            transcript.push_str(&format!("<|im_start|>user\n{message}<|im_end|>\n"));

            let schema_json = serde_json::to_string(&local_action_schema()).unwrap_or_default();
            let final_schema_json = serde_json::to_string(&local_final_action_schema()).unwrap_or_default();

            let mut tool_rounds = 0;
            let mut empty_searches = 0usize;
            let mut hinted = false;
            // What has happened since the tool last spoke: sent on its own when continuing.
            let mut new_since_last = String::new();
            while tool_rounds < rounds_budget {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                tool_rounds += 1;
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
                let out_str = agent.complete_continuing(&cli_prompt, !first).await?;
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
                        on_event(ChatEvent::ToolStarted { tool: tool.clone(), args: args.clone() });
                        let vector = if tool == "search_moments" {
                            if let (Some(e), Some(q)) =
                                (ctx.embedder.as_mut(), args.get("query").and_then(|v| v.as_str()))
                            {
                                query_vector(e, q).await.ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let (res, summary) = dispatch_tool_limited(
                            &ctx.db,
                            &ctx.data_dir,
                            project_id,
                            &tool,
                            &args,
                            &mut grounding,
                            vector.as_deref(),
                            ctx.script.roomy_tool_result_chars,
                            ctx.script.max_shake_jerk,
                            ctx.script.shake_relative,
                            ctx.script.max_sway,
                        );
                        on_event(ChatEvent::ToolFinished { tool: tool.clone(), summary: summary.clone() });
                        let empty = is_empty_search(&tool, &summary);
                        tool_records.push(ToolCallRecord { tool: tool.clone(), args: args.clone(), summary });

                        if empty {
                            empty_searches += 1;
                        } else if !tool.starts_with("search") {
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
                    }
                    Ok(LocalAction::Final { script }) => {
                        let mut s = script;
                        s.fill_from_project(&project);
                        parsed_script = Some(s);
                        break;
                    }
                    Err(_) => {
                        // A CLI may wrap the JSON in prose: pull the object out and retry once.
                        if let Some(start) = out_str.find('{')
                            && let Ok(LocalAction::Final { mut script }) =
                                serde_json::from_str::<LocalAction>(&out_str[start..])
                        {
                            script.fill_from_project(&project);
                            parsed_script = Some(script);
                            break;
                        }
                        // Give up this round and ask for the final script.
                        break;
                    }
                }
            }

            if parsed_script.is_none() {
                if cancelled() {
                    return Err(Error::Invalid(CANCELLED.into()));
                }
                on_event(ChatEvent::Drafting);
                let cli_final_prompt = if tool_rounds > 0 {
                    format!(
                        "{new_since_last}\nProduce the final script. Reply ONLY with a JSON object matching this \
                         schema:\n{final_schema_json}"
                    )
                } else {
                    format!(
                        "{transcript}<|im_start|>assistant\n\
                         Produce the final script. Reply ONLY with a JSON object matching this schema:\n{final_schema_json}\n<|im_end|>\n"
                    )
                };
                let final_str = agent.complete_continuing(&cli_final_prompt, tool_rounds > 0).await?;
                debug_dump("cli-final", &cli_final_prompt, &final_str);
                if let Ok(LocalAction::Reply { text }) = serde_json::from_str::<LocalAction>(&final_str) {
                    raw_reply = text;
                } else if let Ok(LocalAction::Final { mut script }) = serde_json::from_str::<LocalAction>(&final_str) {
                    script.fill_from_project(&project);
                    parsed_script = Some(script);
                }
            }

            on_event(ChatEvent::Validating);
            if let Some(s) = &mut parsed_script {
                if requested_s.is_some() {
                    s.target_duration_s = requested_s;
                }
                pre_issues =
                    enforce_grounding_and_pacing(&ctx.db, project_id, s, &grounding, enforce_target, &ctx.script);
            }
        }
    }

    let mut script_id = None;
    let mut issues = pre_issues;

    if let Some(mut s) = parsed_script {
        if s.clip_count() > 0 && !s.beats.is_empty() {
            let _ = snap_to_segments(&ctx.db, &mut s)?;
            pad_speech(&ctx.db, &mut s, &ctx.script);
            clamp_to_duration(&ctx.db, &mut s);
            let muted = mute_silent_clips(&ctx.db, &mut s);
            if muted > 0 {
                issues.push(Issue {
                    severity: IssueSeverity::Info,
                    beat_id: None,
                    clip_index: None,
                    message: format!("muted {muted} clip(s) without speech under the narration"),
                });
            }
            // snap_to_segments and pad_speech above pull clips out to whole sentences, which
            // undoes the trim enforce_grounding_and_pacing just made: an interview-led cut came
            // out 37% over target because the last word on the subject was padding, not trimming.
            // Fit once more, now that the clips are their final length.
            if enforce_target {
                let before = s.total_duration_s();
                if fit_to_target(&ctx.db, &mut s, &ctx.script) {
                    // Trimming can cut a sentence short again. Snapping only reaches 0.75 s, so
                    // this puts every voice back on a whole sentence whatever the trim did — the
                    // length gives way to the speaker, not the other way round.
                    let _ = snap_to_segments(&ctx.db, &mut s)?;
                    let mended = end_on_sentences(&ctx.db, &mut s, &ctx.script);
                    if mended > 0 {
                        issues.push(Issue {
                            severity: IssueSeverity::Info,
                            beat_id: None,
                            clip_index: None,
                            message: format!("put {mended} clip(s) back on whole sentences after fitting"),
                        });
                    }
                    issues.push(Issue {
                        severity: IssueSeverity::Info,
                        beat_id: None,
                        clip_index: None,
                        message: format!(
                            "fitted to target after padding: {before:.1} s → {:.1} s",
                            s.total_duration_s()
                        ),
                    });
                }
            }
            issues.extend(content_issues(&ctx.db, &s, &ctx.script));
            issues.extend(validate(&ctx.db, project_id, &s)?);
            let sid = save_version(&ctx.db, project_id, &s, Some(session_id))?;
            script_id = Some(sid);
            parsed_script = Some(s);
        } else {
            parsed_script = None;
        }
    }

    // What the pipeline changed after the draft was written — beds laid, clips moved off shaky
    // stretches, ranges dropped. The model is never told this in the turn that caused it, because
    // the repair runs after the last redraft, so it repeats the same edit next time and the
    // pipeline undoes it again. Recording it in the reply puts it in what the next turn replays.
    let repair_note = repair_note(&issues);

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
    } else {
        "I was unable to assemble a script from the available footage.".to_string()
    };

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
    let assistant_tc_json = script_id.map(|sid| json!({ "script_id": sid }).to_string());
    let stored_reply = match &repair_note {
        Some(note) => format!("{reply}\n\n{note}"),
        None => reply.clone(),
    };
    ctx.db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'assistant', ?2, ?3, ?4)",
        params![session_id, stored_reply, assistant_tc_json, now_ts],
    )?;

    // Update chat_sessions updated_at
    ctx.db.conn.execute("UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2", params![now_ts, session_id])?;

    Ok(TurnResult { session_id, reply, script_id, script: parsed_script, issues, tool_calls: tool_records })
}

/// What the pipeline changed after the draft was written — beds laid, clips moved off shaky
/// stretches, ranges dropped — written for the model rather than the user.
///
/// The repair runs after the last redraft, so nothing tells the model in the turn that caused it;
/// it makes the same edit next time and the pipeline undoes it again. Carried on the assistant
/// message, this reaches the server, local and CLI brains alike, since all three replay it.
fn repair_note(issues: &[Issue]) -> Option<String> {
    let repairs: Vec<&str> = issues
        .iter()
        .filter(|i| i.severity != IssueSeverity::Error)
        .map(|i| i.message.as_str())
        .take(6)
        .collect();
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
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("named {fixed} beat(s) after their purpose"),
        });
    }
    issues
}

/// How far a voice can keep going from `from_s` towards `wanted_end`: never into the interviewer's
/// next question, and stopping on a whole sentence. Shared by the beds the editor asks for and the
/// ones laid here, so both end the same way.
fn speech_runs_until(db: &Db, video_id: i64, from_s: f64, wanted_end: f64, cfg: &crate::config::ScriptConfig) -> f64 {
    let mut end = wanted_end;
    let segs: Vec<(f64, f64, bool)> = db
        .conn
        .prepare(
            "SELECT start_s, end_s, COALESCE(off_mic, 0) FROM transcript_segments
             WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 ORDER BY start_s",
        )
        .and_then(|mut st| {
            st.query_map(params![video_id, from_s, end], |r| {
                Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, i64>(2)? != 0))
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
        })
        .unwrap_or_default();
    if let Some((off_start, _, _)) = segs.iter().find(|(_, _, off)| *off) {
        end = end.min(*off_start);
    }
    match segs.iter().rev().find(|(_, e, off)| !*off && *e <= end + cfg.speech_overrun_s) {
        // Stop where the sentence stops, the way a clip does.
        Some((_, seg_end, _)) => (*seg_end + cfg.speech_overrun_s).min(wanted_end),
        // Nobody speaks out here. Running the bed on would only add silence that has been faded
        // out anyway, and would claim in the export that there is audio to hear.
        None => from_s,
    }
}

/// Let a speaker's voice run under the pictures that follow, instead of stopping at the cutaway.
///
/// A beat usually opens on the person talking and then cuts to what they are describing. Until
/// beds existed the second half went silent, because sound belonged to whichever clip was on
/// screen — so every teaser alternated a talking head with a mute postcard. Here the beat's
/// speech is extended across the whole beat and the pictures play under it.
///
/// The bed is aligned so the face stays in sync: it starts as far *before* the speaking clip's
/// own in-point as that clip sits into the beat, so when we reach the face, the audio is exactly
/// where it would have been. It stops at a sentence end, before the interviewer's next question,
/// and never runs further than `max_bed_extend_s` past the clip it came from.
pub fn lay_audio_beds(db: &Db, s: &mut Script, cfg: &crate::config::ScriptConfig) -> Vec<Issue> {
    let mut issues = Vec::new();
    if !cfg.infer_audio_beds {
        return issues;
    }
    let mut laid = 0usize;

    for beat in &mut s.beats {
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();

        // A bed the editor asked for is checked, not trusted: it has to be speech, and it cannot
        // outlast the pictures it plays under or it would run into the next beat.
        if let Some(bed) = beat.bed.take() {
            let mut bed = bed;
            if bed.duration_s() > beat_len && beat_len > 0.0 {
                bed.out_s = bed.in_s + beat_len;
            } else if beat_len > bed.duration_s() + 0.05 {
                // Written to the length of the speaker's own clip, not the beat's: the pictures
                // after it would play silent, which is the thing a bed exists to prevent.
                let wanted = (bed.in_s + beat_len).min(bed.out_s + cfg.max_bed_extend_s);
                let grown = speech_runs_until(db, bed.video_id, bed.out_s, wanted, cfg);
                if grown > bed.out_s + 0.05 {
                    bed.out_s = grown;
                }
            }
            if !clip_has_speech(db, bed.video_id, bed.in_s, bed.out_s) {
                issues.push(Issue {
                    severity: IssueSeverity::Warning,
                    beat_id: Some(beat.id.clone()),
                    clip_index: None,
                    message: format!(
                        "dropped the audio bed: nobody speaks at video #{} {:.1}–{:.1} s",
                        bed.video_id, bed.in_s, bed.out_s
                    ),
                });
            } else {
                for c in &mut beat.clips {
                    c.audio = crate::script::Audio::Mute;
                }
                beat.bed = Some(bed);
                continue;
            }
        }

        if beat.clips.len() < 2 {
            continue;
        }
        // Where does the voice come from, and how far into the beat does its picture sit?
        let mut offset = 0.0f64;
        let mut source: Option<(usize, f64)> = None;
        for (i, c) in beat.clips.iter().enumerate() {
            if c.audio == crate::script::Audio::Source && clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                source = Some((i, offset));
                break;
            }
            offset += (c.out_s - c.in_s).max(0.0);
        }
        let Some((idx, lead)) = source else { continue };

        let beat_dur = beat_len;
        let clip = &beat.clips[idx];
        let clip_dur = (clip.out_s - clip.in_s).max(0.0);
        // Only worth a bed when something other than this clip is on screen.
        if beat_dur <= clip_dur + 0.05 {
            continue;
        }

        let start = (clip.in_s - lead).max(0.0);
        let wanted_end = start + beat_dur;
        let cap = clip.out_s + cfg.max_bed_extend_s;
        let capped = wanted_end.min(cap);

        // Never run into the interviewer's next question, and stop on a whole sentence.
        let end = speech_runs_until(db, clip.video_id, clip.out_s, capped, cfg);
        // Worth a bed when the voice carries past its own picture, or begins under one before
        // it — the two halves of a J-cut. When it does neither, the old behaviour is already right.
        let trails = end > clip.out_s + 0.05;
        let leads = start < clip.in_s - 0.05;
        if !trails && !leads {
            continue;
        }

        beat.bed = Some(crate::script::AudioBed {
            video_id: clip.video_id,
            in_s: start,
            out_s: end,
            why: None,
            inferred: true,
        });
        for c in &mut beat.clips {
            c.audio = crate::script::Audio::Mute;
        }
        laid += 1;
    }

    if laid > 0 {
        issues.push(Issue {
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: format!("let the speaker's voice run under the pictures in {laid} beat(s)"),
        });
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None };
        let mut script = Script {
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
                notes: None,
                clips: vec![clip(0.0, 25.0), clip(25.0, 50.0), clip(50.0, 75.0)],
                bed: None,
            }],
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

    /// The fit that runs last can trim a speech clip far past what snapping reaches, leaving a
    /// speaker cut off mid-thought — four clips in one run, one of them mid-word.
    #[test]
    fn a_trimmed_speech_clip_is_put_back_on_a_whole_sentence() {
        use crate::script::{Audio, Beat, ScriptClip};
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 300.0)", []).unwrap();
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
            notes: None,
            clips: vec![ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None }],
            bed: None,
        };
        let script_of = |b: Beat| Script {
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![b],
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

        // A clip already on a boundary is left alone.
        let mut s = script_of(beat(10.0, 31.0));
        assert_eq!(end_on_sentences(&db, &mut s, &sc()), 0);

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

    /// The editor reads the tapes before it cuts. Everything spoken goes in the prompt, marked
    /// where it is off the microphone, and when it will not all fit the tapes with the most
    /// speech go first and the rest are named rather than hidden.
    #[test]
    fn every_word_spoken_goes_in_the_prompt() {
        let mut db = Db::open_in_memory().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        let folder = db.add_folder(p.id, tmp.path(), true).unwrap();
        for (id, name) in [(1, "talky.mp4"), (2, "quiet.mp4")] {
            db.conn
                .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60.0)", params![
                    id,
                    format!("h{id}")
                ])
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                     VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, tmp.path().join(name).to_str().unwrap()],
                )
                .unwrap();
        }
        // The talkative tape, and one question asked from across the room.
        for i in 0..8 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                     VALUES (1, ?1, ?2, 'I grew up in Northwest Hills and stayed', 0)",
                    params![i as f64 * 3.0, i as f64 * 3.0 + 3.0],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                 VALUES (1, 30.0, 32.0, 'So where did you grow up?', 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                 VALUES (2, 0.0, 2.0, 'a word from the quiet tape', 0)",
                [],
            )
            .unwrap();

        let all = speech_digest(&db, p.id, 60_000);
        assert!(all.contains("#1 talky.mp4"), "{all}");
        assert!(all.contains("#2 quiet.mp4"));
        assert!(all.contains("I grew up in Northwest Hills"));
        assert!(all.contains("[off-mic] So where did you grow up?"), "the questions are marked: {all}");
        assert!(all.contains("0.00-3.00"), "timestamps to cut on");

        // Squeezed: the tape with the most speech is kept and cut short, not dropped for a
        // shorter one that happens to fit.
        let tight = speech_digest(&db, p.id, 700);
        assert!(tight.contains("#1 talky.mp4"), "{tight}");
        assert!(tight.contains("the rest of this one with get_transcript"), "{tight}");
        assert!(tight.len() <= 800, "stays near its budget: {}", tight.len());
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
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();

        let mut ctx = ChatContext {
            db,
            data_dir: tmp.path().to_path_buf(),
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
            cancel: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "a 40 second teaser", &mut |_| {}).await;
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
        let mut script = Script {
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
                notes: None,
                clips: vec![ScriptClip { video_id: 1, in_s: 0.0, out_s: 6.0, audio: Audio::Mute, why: None }],
                bed: Some(AudioBed { video_id: 1, in_s: 10.0, out_s: 30.0, why: None, inferred: true }),
            }],
        };
        clamp_beds_to_beats(&mut script);
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
            notes: None,
            clips: vec![],
            bed: None,
        };
        let mut script = Script {
            title: "t".into(),
            target_duration_s: None,
            fps: Default::default(),
            width: None,
            height: None,
            beats: vec![
                beat("So was that Northwest Hills where you grew up? I grew up in Northwest Hills.", "Establish the deep roots"),
                beat("", "Establish the deep roots"),
                beat("keep-me", "Something else"),
            ],
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
                notes: None,
                // Her face for 4 s, then two pictures of what she is describing.
                clips: vec![
                    ScriptClip { video_id: 1, in_s: 10.0, out_s: 14.0, audio: Audio::Source, why: None },
                    ScriptClip { video_id: 2, in_s: 0.0, out_s: 5.0, audio: Audio::Mute, why: None },
                    ScriptClip { video_id: 2, in_s: 20.0, out_s: 25.0, audio: Audio::Mute, why: None },
                ],
                bed: None,
            }],
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
                notes: None,
                clips: vec![
                    ScriptClip { video_id: 2, in_s: 0.0, out_s: 3.0, audio: Audio::Mute, why: None },
                    ScriptClip { video_id: 1, in_s: 14.0, out_s: 20.0, audio: Audio::Source, why: None },
                ],
                bed: None,
            }],
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
                notes: None,
                clips: vec![ScriptClip { video_id: 1, in_s: 0.0, out_s: 5.0, audio: Audio::Mute, why: None }],
                bed: Some(AudioBed { video_id: 1, in_s: 40.0, out_s: 50.0, why: None, inferred: false }),
            }],
        };
        let issues = lay_audio_beds(&db, &mut script, &sc());
        assert!(script.beats[0].bed.is_none(), "a bed with no speech in it is dropped");
        assert!(issues.iter().any(|i| i.message.contains("nobody speaks")));
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
                notes: None,
                clips: vec![ScriptClip { video_id: 1, in_s: 0.5, out_s: 4.7, audio: Audio::Mute, why: None }],
                bed: None,
            }],
        };
        let issues = enforce_grounding_and_pacing(&db, p.id, &mut script, &grounding, false, &sc());
        let c = &script.beats[0].clips[0];
        assert!(
            (c.in_s - 8.0).abs() < 1e-9 && (c.out_s - 12.2).abs() < 1e-9,
            "moved to the steady run: {c:?} {issues:?}"
        );
        assert!(issues.iter().any(|i| i.message.contains("moved clip off a shaky stretch")), "{issues:?}");
    }

    /// The three actions a turn can take. A reply carries words and no script, which is what lets
    /// the editor ask a question instead of drafting around a guess.
    #[test]
    fn a_turn_may_answer_in_words_instead_of_drafting() {
        let reply: LocalAction =
            serde_json::from_str(r#"{"action":"reply","text":"Which three voices do you want?"}"#).unwrap();
        assert!(matches!(reply, LocalAction::Reply { text } if text.contains("voices")));

        // The schema the model is constrained to must actually permit it, or it can never be
        // produced no matter what the rules say.
        let schema = serde_json::to_string(&local_action_schema()).unwrap();
        assert!(schema.contains("reply"), "the action schema must allow a reply");
        let forced = serde_json::to_string(&local_final_action_schema()).unwrap();
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
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Mute, why: None };
        let mut script = Script {
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
                notes: None,
                // Real footage it never opened, then a range past the end of the video.
                clips: vec![clip(0.0, 5.0), clip(9000.0, 9005.0)],
                bed: None,
            }],
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
        let tool_json = r#"{"action":"tool","tool":"search_moments","args":{"query":"unbox","limit":5}}"#;
        let action: LocalAction = serde_json::from_str(tool_json).unwrap();
        match action {
            LocalAction::Tool { tool, args } => {
                assert_eq!(tool, "search_moments");
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
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None };
        let mut s = Script {
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
                clips: vec![clip(0.0, 45.0), clip(90.0, 144.0), clip(200.0, 206.0)],
                notes: None,
                bed: None,
            }],
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
    fn empty_searches_are_recognised() {
        assert!(is_empty_search("search_moments", "0 hits"));
        assert!(is_empty_search("search_moments", "no matches"));
        assert!(!is_empty_search("search_moments", "8 hits"));
        assert!(!is_empty_search("get_transcript", "0 segments"), "only searches count");
        assert!(EMPTY_SEARCH_HINT.contains("list_videos"));
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
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 100.0, 110.0, 'hi')",
                [],
            )
            .unwrap();
        db.conn.execute("INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 20.0, '{}')", []).unwrap();
        let clip = |in_s: f64, out_s: f64| ScriptClip { video_id: 1, in_s, out_s, audio: Audio::Source, why: None };
        let beat = |id: &str, narration: Option<&str>, clips| Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: narration.map(Into::into),
            on_screen_text: None,
            clips,
            notes: None,
            bed: None,
        };
        let mut s = Script {
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
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 110.0);
        pad_speech(&db, &mut talk, &sc());
        let c = &talk.beats[1].clips[0];
        assert!((c.in_s - 99.5).abs() < 1e-9, "{c:?}");
        assert!((c.out_s - 110.7).abs() < 1e-9, "stops before the next sentence: {c:?}");
        db.conn.execute("DELETE FROM transcript_segments WHERE text = 'next'", []).unwrap();
        let mut talk = s.clone();
        talk.beats[1].clips[0] = clip(100.0, 105.0);
        pad_speech(&db, &mut talk, &sc());
        assert!((talk.beats[1].clips[0].out_s - 111.5).abs() < 1e-9, "finishes the sentence, then holds");

        assert_eq!(mute_silent_clips(&db, &mut s), 2);
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
             VALUES (10, 0.0, 5.0, 'welcome to cm5 unboxing'),
                    (10, 5.0, 12.0, 'here is the compute module board')",
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

        // 1. list_videos
        let (res, sum) = dispatch_tool(&db, tmp.path(), p.id, "list_videos", &json!({}), &mut grounding, None, &sc());
        assert!(res.len() <= 1500);
        assert!(res.contains("clip.mp4"));
        assert_eq!(sum, "1 videos");

        // 2. get_video
        let (res, sum) =
            dispatch_tool(&db, tmp.path(), p.id, "get_video", &json!({"video_id": 10}), &mut grounding, None, &sc());
        assert!(res.len() <= 1500);
        assert!(res.contains("Hands holding a green board"));
        assert!(sum.contains("30s"));
        assert!(grounding.is_grounded(10, 0.0, 30.0, sc().grounding_slack_s));

        // 3. get_transcript
        let (res, sum) = dispatch_tool(
            &db,
            tmp.path(),
            p.id,
            "get_transcript",
            &json!({"video_id": 10, "start_s": 0.0, "end_s": 6.0}),
            &mut grounding,
            None,
            &sc(),
        );
        assert!(res.len() <= 1500);
        assert!(res.contains("welcome to cm5 unboxing"));
        assert_eq!(sum, "2 segments");

        // 4. search_moments (keyword only)
        let (res, sum) = dispatch_tool(
            &db,
            tmp.path(),
            p.id,
            "search_moments",
            &json!({"query": "unboxing"}),
            &mut grounding,
            None,
            &sc(),
        );
        assert!(res.len() <= 1500);
        assert!(res.contains("unboxing"));
        assert_eq!(sum, "1 hits");

        // 5. Foreign video rejected
        let (res, sum) =
            dispatch_tool(&db, tmp.path(), p.id, "get_video", &json!({"video_id": 999}), &mut grounding, None, &sc());
        assert!(res.contains("error"));
        assert!(sum.contains("error"));
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

                let resp_body = match step {
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
            data_dir: tmp.path().to_path_buf(),
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
            cancel: None,
        };

        let res =
            run_turn(&mut ctx, p.id, None, "Create a 5s teaser about unboxing", &mut |e| events.push(e)).await.unwrap();

        assert!(res.script_id.is_some());
        let script = res.script.unwrap();
        assert_eq!(script.title, "CM5 Fast Teaser");
        assert_eq!(script.beats.len(), 1);
        assert_eq!(res.tool_calls.len(), 1);
        assert_eq!(res.tool_calls[0].tool, "search_moments");

        // Verify events
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolStarted { tool, .. } if tool == "search_moments")));
        assert!(events.iter().any(|e| matches!(e, ChatEvent::ToolFinished { tool, .. } if tool == "search_moments")));
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
            data_dir: tmp.path().to_path_buf(),
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
            cancel: None,
        };

        let res = run_turn(&mut ctx, p.id, None, "Make video", &mut |_| {}).await.unwrap();

        // The clip was not grounded in tool results -> dropped -> no beats left -> script_id None
        assert_eq!(res.script_id, None);
        assert!(res.script.is_none());
    }

    #[test]
    fn run_turn_future_is_send() {
        fn assert_send<T: Send>(_: T) {}
        let _ = |ctx: &mut ChatContext, mut cb: Box<dyn FnMut(ChatEvent) + Send>| {
            assert_send(run_turn(ctx, 1, None, "test", &mut *cb));
        };
    }
}
