//! User configuration (`config.toml`), shared by the desktop app and the CLI.
//!
//! Every AI capability has an independent `backend` switch (plan §2a):
//! `auto` probes the configured server and falls back to running the model locally,
//! `server` requires the server, `local` never touches it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;

/// Default endpoints of the sibling services on a dev box.
pub const DEFAULT_VISION_URL: &str = "http://127.0.0.1:8089"; // highllama chat/vision
pub const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:8091"; // highllama embeddings server
pub const DEFAULT_STT_URL: &str = "http://127.0.0.1:8771"; // GhostPen transcription server

/// The one embedding model GhostReel uses everywhere (plan D5): vectors from the server and
/// the local runtime must be interchangeable.
pub const EMBED_MODEL: &str = "embeddinggemma-300M-Q8_0";
pub const EMBED_DIM: usize = 768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Use the server when it is reachable and capable, else run locally.
    #[default]
    Auto,
    /// Always run in-process.
    Local,
    /// Always use the server; fail if it is unavailable.
    Server,
    /// Delegate to an installed coding-agent CLI (claude, agy, opencode). Never chosen by `auto`
    /// because it costs money / quota; the user must set this explicitly.
    Cli,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Backend::Auto => "auto",
            Backend::Local => "local",
            Backend::Server => "server",
            Backend::Cli => "cli",
        })
    }
}

/// Valid coding-agent CLI tools.
pub const CLI_TOOLS: &[&str] = &["claude", "agy", "opencode", "codex"];

/// Configuration for a coding-agent CLI backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CliAgentConfig {
    /// Which tool to use: `claude`, `agy`, `opencode`, or `codex`.
    pub tool: String,
    /// Explicit path to the binary. Empty = resolved from PATH (or `GHOSTREEL_<TOOL>` env var).
    pub command: String,
    /// Model to pass via `--model`; empty = CLI default.
    pub model: String,
    /// Extra flags appended verbatim to every invocation.
    pub extra_args: Vec<String>,
    /// Hard timeout for one describe/complete call, in seconds (default 180).
    pub timeout_secs: u64,
    /// Maximum concurrent CLI invocations during the describe stage (default 2).
    pub concurrency: usize,
}

impl Default for CliAgentConfig {
    fn default() -> Self {
        Self {
            tool: String::new(),
            command: String::new(),
            model: String::new(),
            extra_args: Vec::new(),
            timeout_secs: 180,
            concurrency: 2,
        }
    }
}

impl CliAgentConfig {
    pub fn validate(&self, section: &str) -> Result<(), String> {
        if !self.tool.is_empty() && !CLI_TOOLS.contains(&self.tool.as_str()) {
            return Err(format!("{section}.cli.tool must be one of {}, got '{}'", CLI_TOOLS.join(", "), self.tool));
        }
        if self.timeout_secs == 0 {
            return Err(format!("{section}.cli.timeout_secs must be > 0"));
        }
        if self.concurrency == 0 {
            return Err(format!("{section}.cli.concurrency must be > 0"));
        }
        Ok(())
    }
}

/// Settings for one model-using capability. Frame descriptions (`[vision]`) and the script chat
/// (`[chat_model]`) each have their own: describing a keyframe is a short prompt plus one image run
/// hundreds of times, while a script chat needs room for tool results and a whole draft, so they
/// want different context windows even though both run the same helper binary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub backend: Backend,
    pub url: String,
    /// Model id sent to the server; empty = whatever the server has loaded.
    pub model: String,
    /// Bearer token for non-local servers.
    pub api_key: String,
    /// Local vision model catalog id (pair: model + projector). Default: "bonsai-27b".
    pub local_model: String,
    /// Context window for the local helper, in tokens (2048–131072).
    pub ctx_tokens: u32,
    /// KV cache precision for the local helper: `f16`, `q8_0` or `q4_0`. `q4_0` holds roughly 4×
    /// the context of `f16` in the same VRAM (what highllama runs with).
    pub kv_cache: String,
    /// Flash attention for the local helper: `auto`, `on` or `off`.
    pub flash_attn: String,
    /// How many tool calls the script chat may make before it has to draft: how much footage it
    /// gets to search, watch and read transcripts of. 0 = pick from the backend (a local model
    /// gets fewer, since every round costs it context it cannot spare).
    pub max_tool_rounds: u32,
    /// Let the local model reason before it answers. Worth its cost for a whole script, not for
    /// a keyframe: descriptions run once per frame, so they default to off and scripts to on.
    pub think: bool,
    /// Frames described at once when the backend is a server. Generating a token means reading
    /// every weight out of VRAM, so the GPU spends most of decode waiting on memory; a batch
    /// reads those weights once and answers several frames from it. The server must be started
    /// with a matching `--parallel` or the requests queue — which is harmless (2.87 s a frame
    /// against 3.03 s sequential), so over-asking costs nothing and under-asking leaves the card
    /// idle. Ignored for local and CLI backends.
    #[serde(default = "default_describe_concurrency")]
    pub describe_concurrency: u32,
    /// CLI agent settings (used when `backend = "cli"`).
    pub cli: CliAgentConfig,
}

/// Context window of the frame-description helper: a short prompt plus one image.
pub const DESCRIBE_CTX_TOKENS: u32 = 8192;
/// Context window of the script chat: tool results plus a whole draft.
pub const CHAT_CTX_TOKENS: u32 = 32768;
pub const KV_CACHE_KINDS: &[&str] = &["f16", "q8_0", "q4_0"];
pub const FLASH_ATTN_KINDS: &[&str] = &["auto", "on", "off"];

fn default_describe_concurrency() -> u32 {
    4
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            url: DEFAULT_VISION_URL.into(),
            model: String::new(),
            api_key: String::new(),
            local_model: "bonsai-27b".into(),
            ctx_tokens: DESCRIBE_CTX_TOKENS,
            kv_cache: "q4_0".into(),
            flash_attn: "auto".into(),
            max_tool_rounds: 0,
            think: false,
            describe_concurrency: default_describe_concurrency(),
            cli: CliAgentConfig::default(),
        }
    }
}

impl VisionConfig {
    pub fn validate(&self, section: &str) -> Result<(), String> {
        if !(2048..=131_072).contains(&self.ctx_tokens) {
            return Err(format!("{section}.ctx_tokens must be between 2048 and 131072, got {}", self.ctx_tokens));
        }
        if !KV_CACHE_KINDS.contains(&self.kv_cache.as_str()) {
            return Err(format!(
                "{section}.kv_cache must be one of {}, got '{}'",
                KV_CACHE_KINDS.join(", "),
                self.kv_cache
            ));
        }
        if !FLASH_ATTN_KINDS.contains(&self.flash_attn.as_str()) {
            return Err(format!(
                "{section}.flash_attn must be one of {}, got '{}'",
                FLASH_ATTN_KINDS.join(", "),
                self.flash_attn
            ));
        }
        self.cli.validate(section)?;
        Ok(())
    }
}

/// Every number the script generator uses to turn a draft into a timeline.
///
/// These were constants, which meant a house style could only be changed by rebuilding. They are
/// deliberately plain seconds, words and fractions: a documentary cut and a fast promo disagree
/// about nearly all of them, and so do two editors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScriptConfig {
    /// Slack around a range a tool returned, when checking a clip came from real footage (s).
    pub grounding_slack_s: f64,
    /// Longer than this and the model pasted a whole tool range rather than choosing a shot (s).
    pub max_clip_s: f64,
    /// Over-long clips are cut back to this, keeping their start (s).
    pub trimmed_clip_s: f64,
    /// Shorter than this and a shot flashes past before it can be read (s).
    pub min_clip_s: f64,
    /// Trimming to fit never takes a shot below this (s).
    pub min_trimmed_clip_s: f64,
    /// How far over target a draft may sit before it is squeezed (1.25 = 25% over).
    pub target_overshoot: f64,
    /// The last pass before saving works to this (1.02 = 2% over). Whatever it leaves is final.
    pub final_target_tolerance: f64,
    /// Hard limit on the share of the target that speaking clips may fill. 1.0 — the default —
    /// leaves the balance of interview and pictures to the editor, which is the only thing that
    /// knows whether this piece was asked for in people's own words or as a scenic teaser. Lower
    /// it to impose a house rule (0.7 keeps a third of the running time for pictures).
    pub speech_budget: f64,
    /// Spoken words per second, for checking narration covers its beat.
    pub narration_words_per_s: f64,
    /// Narration shorter than this share of its beat is reported as a gap (0.6).
    pub min_narration_coverage: f64,
    /// Held after someone's last word before cutting away (s).
    pub speech_tail_s: f64,
    /// When the speaker runs straight on with no pause, how far into the next sentence the cut
    /// may reach so the last word's decay is not chopped off (s). Whisper's segments are
    /// contiguous in continuous speech, so without this the clip ends on the final sample.
    pub speech_overrun_s: f64,
    /// Audio faded in and out at each join (s). A cut in the middle of someone's breath is a
    /// click and an abrupt stop however well the sentence ended.
    pub audio_fade_s: f64,
    /// Held before someone's first word (s).
    pub speech_lead_s: f64,
    /// Let a speaker's voice run on under the pictures that follow it in the same beat, instead of
    /// stopping when we cut away. The editor can name a bed itself; this decides whether the
    /// pipeline lays one where it sees a beat that would otherwise fall silent.
    pub infer_audio_beds: bool,
    /// Put every word spoken in the project in the prompt, before any tool is called. A model
    /// that has to ask for each transcript reads a couple of tapes and builds the story out of
    /// whoever it found there; an editor reads the interviews first. Off for a brain with a small
    /// context that needs the room for tool results.
    pub speech_in_prompt: bool,
    /// How much of each tool's description to send: `auto`, `full` or `short`. Rich descriptions
    /// — what a result means, what the repair passes will do to a draft that misuses it — are
    /// worth far more than their tokens, but a small local model's window is already half
    /// transcripts, so `auto` shortens them there.
    pub tool_docs: String,
    /// Picture held after the last voice stops, in silence, so the piece lands instead of
    /// stopping (s). 0 turns it off. Every brain tested ends on the last word; an editor holds
    /// the closing image for a beat and lets it go quiet.
    pub closing_hold_s: f64,
    /// Ceiling on a single model answer, in tokens. Generous: a long script with forty clips is
    /// thousands of tokens and must never be cut off. It exists only so a model that will not
    /// stop fails as itself instead of as a dead socket.
    pub max_answer_tokens: u32,
    /// How long to wait for a server brain to answer one request (s). A local model producing a
    /// long script at 40 tokens a second needs minutes, and the prompt has to be read first.
    pub server_timeout_s: u64,
    /// How far a laid bed may run past the clip it came from (s). Long enough for a cutaway or
    /// two, short enough that a beat does not swallow a whole answer nobody asked for.
    pub max_bed_extend_s: f64,
    /// A clip is never stretched further than this to reach whole sentences (s).
    pub max_speech_extend_s: f64,
    /// Tool rounds for a local model, whose context every result is spent from.
    pub local_tool_rounds: u32,
    /// Tool rounds for a server or CLI brain: a stop against looping, not a research budget.
    pub roomy_tool_rounds: u32,
    /// Characters of tool result a local model is given.
    pub local_tool_result_chars: usize,
    /// Characters of tool result a server or CLI brain is given.
    pub roomy_tool_result_chars: usize,
    /// Above this, a stretch is called shaky and the editor is told to cut around it. The number
    /// is high-frequency camera movement as a percentage of the frame width per frame, and it
    /// agrees with ffmpeg's vid.stab to within 0.1: a static shot, a tripod and a stabilised action
    /// camera all sit at 0.2–0.45 (the floor), handheld footage at 0.6–1.3 with its worst
    /// stretches around 2. 0 turns the check off.
    pub max_shake_jerk: f64,
    /// A stretch also has to be this many times shakier than its own clip's ordinary level to be
    /// called shaky: handheld footage is judged against itself, not condemned wholesale. The
    /// higher of this and `max_shake_jerk` applies. 0 uses `max_shake_jerk` alone.
    pub shake_relative: f64,
    /// Sway — movement within a second that is undone — above which a stretch is shaky, as a
    /// percentage of the frame width per second. Mounted and stabilised footage measured 0–0.6,
    /// an unstabilised walking shot 1.3. 0 turns the sway check off.
    ///
    /// Calibrated against the editor's own eye on the Greet Mag footage (Sep 2026): stretches at
    /// sway 1.35 and 2.01 were called shaky, one at 0.09 was called fine, and a stretch with
    /// tremor 2.87 was shaky by both measures. vid.stab's motion paths, scored the same way,
    /// agreed with GhostReel's to within ~0.1 across eleven clips. Change these with evidence.
    pub max_sway: f64,
    /// Seconds measured at a time when checking how steady a video is.
    pub shake_window_s: f64,
    /// Seconds between the start of one measured window and the next. Equal to the window by
    /// default, so every second of a video is covered: the shaky stretches are reported as
    /// timestamps, and a gap between windows would hide one.
    pub shake_stride_s: f64,
}

/// Jev, TypeSafe's System One model, used as an editorial judge.
///
/// Everything else in GhostReel runs on this machine. This does not: it is a hosted model, and a
/// judgement sends the cut's narration, the words spoken under it and the descriptions of what is
/// on screen to `api.typesafe.ai`. That is why it is off by default and why turning it on takes
/// both a switch and a key — a shortcut through either would mean a local-first tool quietly
/// posting somebody's interview to a third party.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JevConfig {
    /// Ask Jev at all. Without this nothing here is reached, whatever else is set.
    pub enabled: bool,
    /// The API key. `TYPESAFE_API_KEY` in the environment wins over this, because a config file
    /// is committed by accident far more often than an environment is.
    pub api_key: String,
    /// Which model answers. `jev-latest` moves with each release; pin a version once thresholds
    /// have been tuned against one.
    pub model: String,
    pub base_url: String,
    /// One request carries every question, so this covers the whole judgement.
    pub timeout_s: u64,
    /// How many beats are asked about individually. The per-beat questions are the ones that
    /// grow with the script; the whole-cut ones are fixed.
    pub max_beats: usize,
    /// Characters of transcript quoted per beat, and of picture description per clip. Jev reads
    /// 32k tokens of state; a forty-clip cut with the full transcript under it would spend that
    /// on material no judgement needs.
    pub max_quote_chars: usize,
    /// Above this probability a transcript line is the interviewer rather than the subject, and
    /// is marked off-mic so no draft can quote it (`script interviewer`).
    ///
    /// Measured on the Greet Mag footage: the unmistakable lines — "so just tell me your name",
    /// a mic check, a countdown — sit at 0.84–0.97, and real answers wrongly caught sit at
    /// 0.70–0.76. 0.78 is the gap between them. Lower it to catch more chatter at the cost of
    /// losing the occasional good quote; raise it to keep every answer and do the last of the
    /// weeding by hand.
    pub interviewer_threshold: f64,
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: String::new(),
            model: "jev-latest".into(),
            base_url: "https://api.typesafe.ai".into(),
            timeout_s: 60,
            max_beats: 24,
            max_quote_chars: 600,
            interviewer_threshold: 0.78,
        }
    }
}

impl JevConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("jev.model cannot be empty".into());
        }
        if !self.base_url.starts_with("http") {
            return Err(format!("jev.base_url must be an http(s) URL, got '{}'", self.base_url));
        }
        if self.timeout_s == 0 {
            return Err("jev.timeout_s must be greater than 0".into());
        }
        if !(0.5..=1.0).contains(&self.interviewer_threshold) {
            return Err(format!(
                "jev.interviewer_threshold is a probability above a coin flip (0.5–1.0), got {}",
                self.interviewer_threshold
            ));
        }
        Ok(())
    }
}

impl Default for ScriptConfig {
    fn default() -> Self {
        Self {
            grounding_slack_s: 5.0,
            max_clip_s: 30.0,
            trimmed_clip_s: 20.0,
            min_clip_s: 3.0,
            min_trimmed_clip_s: 4.0,
            target_overshoot: 1.25,
            final_target_tolerance: 1.02,
            speech_budget: 1.0,
            narration_words_per_s: 2.5,
            min_narration_coverage: 0.6,
            speech_tail_s: 1.5,
            speech_overrun_s: 0.35,
            audio_fade_s: 0.12,
            speech_lead_s: 0.5,
            infer_audio_beds: true,
            speech_in_prompt: true,
            tool_docs: "auto".into(),
            closing_hold_s: 2.0,
            max_answer_tokens: 16384,
            server_timeout_s: 1800,
            max_bed_extend_s: 20.0,
            max_speech_extend_s: 12.0,
            local_tool_rounds: 10,
            roomy_tool_rounds: 60,
            local_tool_result_chars: 1500,
            roomy_tool_result_chars: 8000,
            max_shake_jerk: 1.0,
            shake_relative: 2.0,
            max_sway: 1.0,
            shake_window_s: 4.0,
            shake_stride_s: 4.0,
        }
    }
}

impl ScriptConfig {
    pub fn validate(&self) -> Result<(), String> {
        let positive: [(&str, f64); 9] = [
            ("max_clip_s", self.max_clip_s),
            ("trimmed_clip_s", self.trimmed_clip_s),
            ("min_clip_s", self.min_clip_s),
            ("min_trimmed_clip_s", self.min_trimmed_clip_s),
            ("narration_words_per_s", self.narration_words_per_s),
            ("speech_tail_s", self.speech_tail_s),
            ("max_speech_extend_s", self.max_speech_extend_s),
            ("target_overshoot", self.target_overshoot),
            ("final_target_tolerance", self.final_target_tolerance),
        ];
        for (name, v) in positive {
            if !(v > 0.0) {
                return Err(format!("script.{name} must be greater than 0, got {v}"));
            }
        }
        if self.min_clip_s > self.max_clip_s {
            return Err("script.min_clip_s cannot exceed script.max_clip_s".into());
        }
        if self.target_overshoot < 1.0 || self.final_target_tolerance < 1.0 {
            return Err("script target tolerances are multipliers of the target, so at least 1.0".into());
        }
        for (name, v) in
            [("speech_budget", self.speech_budget), ("min_narration_coverage", self.min_narration_coverage)]
        {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("script.{name} is a fraction between 0 and 1, got {v}"));
            }
        }
        if self.max_shake_jerk < 0.0 {
            return Err(format!("script.max_shake_jerk cannot be negative, got {}", self.max_shake_jerk));
        }
        if self.max_sway < 0.0 {
            return Err(format!("script.max_sway cannot be negative, got {}", self.max_sway));
        }
        if self.shake_relative < 0.0 {
            return Err(format!("script.shake_relative cannot be negative, got {}", self.shake_relative));
        }
        if self.shake_window_s <= 0.0 || self.shake_stride_s <= 0.0 {
            return Err("script shake window and stride must be greater than 0".into());
        }
        if self.local_tool_rounds == 0 || self.roomy_tool_rounds == 0 {
            return Err("script tool rounds must be at least 1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    pub backend: Backend,
    pub url: String,
    pub model: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self { backend: Backend::Auto, url: DEFAULT_EMBED_URL.into(), model: EMBED_MODEL.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    pub backend: Backend,
    pub url: String,
    /// Whisper model for the local backend: `auto` (large-v3-turbo on an NVIDIA GPU with ≥ 6 GB,
    /// else `small`), or a name like `large-v3-turbo`, `small`, `base`.
    pub model: String,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self { backend: Backend::Auto, url: DEFAULT_STT_URL.into(), model: "auto".into() }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Custom directory for downloaded model files. If unset, defaults to `Paths::models_dir()`.
    pub dir: Option<PathBuf>,
    /// Extra directories searched for model files before downloading
    /// (e.g. `~/.lmstudio/models`), so an existing copy is reused.
    pub search_paths: Vec<PathBuf>,
}

/// Script chat settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatConfig {
    /// Editing instructions for the script chat; empty = the built-in default
    /// (`chat::DEFAULT_EDITOR_PROMPT`). `{project}`, `{fps}`, `{width}`, `{height}` are filled in.
    pub system_prompt: String,
}

/// Keyframe extraction settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FramesConfig {
    /// Maximum gap between keyframes in seconds. Default 8 s; valid range 1–60.
    /// Shorter = more detail for search and scripts; longer = faster indexing.
    /// Scene changes always get a frame regardless of this setting.
    pub max_interval_s: f64,
    /// How much accumulated picture change earns a keyframe, on top of cuts and the interval
    /// clock. 0 turns it off.
    ///
    /// A fixed interval is wrong in both directions at once: it wastes frames on a locked-off
    /// talking head and starves a moving camera, which never trips a cut threshold and so used to
    /// be sampled by the clock alone. Measured on real footage, ffmpeg's scene score accumulates
    /// at ~0.12/s while travelling and ~0.02/s on a static interview, so 1.0 asks for a frame
    /// every ~8 s of travel and never fires on a talking head. Lower it for more detail on
    /// moving footage, at a describe call per extra frame.
    pub change_budget: f64,
    /// Never sample closer together than this (s). The budget is self-calibrating — footage that
    /// changes five times faster gets five times the frames, with no notion of what a car is —
    /// and this is what stops that from becoming an overnight describe job.
    pub min_interval_s: f64,
}

impl Default for FramesConfig {
    fn default() -> Self {
        Self { max_interval_s: 8.0, change_budget: 1.0, min_interval_s: 2.0 }
    }
}

impl FramesConfig {
    /// Clamp `min_interval_s` to 0.5–30 s, and never above `max_interval_s`: a floor longer than
    /// the ceiling would silently disable the interval clock.
    pub fn clamped_min_interval(&self) -> f64 {
        self.min_interval_s.clamp(0.5, 30.0).min(self.clamped_interval())
    }

    /// Clamp `max_interval_s` to the valid range (1–60 s) without erroring.
    pub fn clamped_interval(&self) -> f64 {
        self.max_interval_s.clamp(1.0, 60.0)
    }

    /// Validate that `max_interval_s` is within the 1–60 range, returning an error message if not.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_interval_s < 1.0 || self.max_interval_s > 60.0 {
            Err(format!("frames.max_interval_s must be between 1 and 60, got {}", self.max_interval_s))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Frame descriptions (indexing).
    pub vision: VisionConfig,
    /// Script chat. Absent in configs written before the split: it then follows `vision`, with the
    /// chat's bigger context window.
    #[serde(default)]
    pub chat_model: Option<VisionConfig>,
    pub embed: EmbedConfig,
    pub stt: SttConfig,
    pub models: ModelsConfig,
    pub frames: FramesConfig,
    pub chat: ChatConfig,
    /// The script generator's timing and research settings.
    #[serde(default)]
    pub script: ScriptConfig,
    /// The optional editorial judge. Off, and inert, unless a key is put in front of it.
    #[serde(default)]
    pub jev: JevConfig,
}

impl Config {
    /// The script chat's model settings: its own `[chat_model]` section, or the frame-description
    /// settings with the chat's larger context window when the file predates the split.
    pub fn chat_model(&self) -> VisionConfig {
        match &self.chat_model {
            Some(c) => c.clone(),
            None => VisionConfig { ctx_tokens: CHAT_CTX_TOKENS, think: true, ..self.vision.clone() },
        }
    }

    /// Apply one `section.field = value` setting, the form `ghostreel config set` and the MCP
    /// `set_settings` tool both speak. Values arrive as strings because that is what both callers
    /// have. Returns a message naming the problem, never a partially applied change.
    pub fn set_key(&mut self, key: &str, value: &str) -> Result<(), String> {
        fn backend(v: &str) -> Result<Backend, String> {
            match v.to_lowercase().as_str() {
                "auto" => Ok(Backend::Auto),
                "local" => Ok(Backend::Local),
                "server" => Ok(Backend::Server),
                "cli" => Ok(Backend::Cli),
                other => Err(format!("invalid backend '{other}'; expected 'auto', 'local', 'server', or 'cli'")),
            }
        }
        fn flag(v: &str) -> bool {
            matches!(v.to_lowercase().as_str(), "true" | "on" | "yes" | "1")
        }
        fn num<T: std::str::FromStr>(key: &str, v: &str) -> Result<T, String> {
            v.parse().map_err(|_| format!("{key} must be a number, got '{v}'"))
        }

        let mut parts = key.split('.');
        let section = parts.next().unwrap_or_default();
        let field = parts.next().unwrap_or_default();
        let sub = parts.next().unwrap_or_default();
        let is_llm = matches!(section, "vision" | "chat_model" | "chat-model");

        // vision.* = frame descriptions, chat_model.* = the script chat.
        if is_llm && field == "cli" {
            let mut llm = if section == "vision" { self.vision.clone() } else { self.chat_model() };
            match sub {
                "tool" => llm.cli.tool = value.to_string(),
                "command" => llm.cli.command = value.to_string(),
                "model" => llm.cli.model = value.to_string(),
                "timeout_secs" => llm.cli.timeout_secs = num(key, value)?,
                "concurrency" => llm.cli.concurrency = num(key, value)?,
                "extra_args" => {
                    return Err(format!("{key}: extra_args is not settable here; edit the config file directly"));
                }
                other => return Err(format!("unknown config key '{section}.cli.{other}'")),
            }
            llm.cli.validate(section)?;
            if section == "vision" {
                self.vision = llm
            } else {
                self.chat_model = Some(llm)
            }
            return Ok(());
        }
        if is_llm {
            let mut llm = if section == "vision" { self.vision.clone() } else { self.chat_model() };
            match field {
                "backend" => llm.backend = backend(value)?,
                "url" => llm.url = value.to_string(),
                "model" => llm.model = value.to_string(),
                "local_model" => llm.local_model = value.to_string(),
                "ctx_tokens" => llm.ctx_tokens = num(key, value)?,
                "kv_cache" => llm.kv_cache = value.to_string(),
                "flash_attn" => llm.flash_attn = value.to_string(),
                "think" => llm.think = flag(value),
                "max_tool_rounds" => llm.max_tool_rounds = num(key, value)?,
                "describe_concurrency" => llm.describe_concurrency = num(key, value)?,
                other => return Err(format!("unknown config key '{section}.{other}'")),
            }
            llm.validate(section)?;
            if section == "vision" {
                self.vision = llm
            } else {
                self.chat_model = Some(llm)
            }
            return Ok(());
        }
        if section == "jev" {
            let f = &mut self.jev;
            match field {
                "enabled" => f.enabled = flag(value),
                "api_key" => f.api_key = value.to_string(),
                "model" => f.model = value.to_string(),
                "base_url" => f.base_url = value.to_string(),
                "timeout_s" => f.timeout_s = num(key, value)?,
                "max_beats" => f.max_beats = num(key, value)?,
                "max_quote_chars" => f.max_quote_chars = num(key, value)?,
                "interviewer_threshold" => f.interviewer_threshold = num(key, value)?,
                other => return Err(format!("unknown config key 'jev.{other}'")),
            }
            return self.jev.validate();
        }
        if section == "script" {
            let f = &mut self.script;
            match field {
                "grounding_slack_s" => f.grounding_slack_s = num(key, value)?,
                "max_clip_s" => f.max_clip_s = num(key, value)?,
                "trimmed_clip_s" => f.trimmed_clip_s = num(key, value)?,
                "min_clip_s" => f.min_clip_s = num(key, value)?,
                "min_trimmed_clip_s" => f.min_trimmed_clip_s = num(key, value)?,
                "target_overshoot" => f.target_overshoot = num(key, value)?,
                "final_target_tolerance" => f.final_target_tolerance = num(key, value)?,
                "speech_budget" => f.speech_budget = num(key, value)?,
                "narration_words_per_s" => f.narration_words_per_s = num(key, value)?,
                "min_narration_coverage" => f.min_narration_coverage = num(key, value)?,
                "speech_tail_s" => f.speech_tail_s = num(key, value)?,
                "speech_overrun_s" => f.speech_overrun_s = num(key, value)?,
                "audio_fade_s" => f.audio_fade_s = num(key, value)?,
                "speech_lead_s" => f.speech_lead_s = num(key, value)?,
                "infer_audio_beds" => f.infer_audio_beds = flag(value),
                "speech_in_prompt" => f.speech_in_prompt = flag(value),
                "tool_docs" => f.tool_docs = value.to_string(),
                "closing_hold_s" => f.closing_hold_s = num(key, value)?,
                "max_answer_tokens" => f.max_answer_tokens = num(key, value)?,
                "server_timeout_s" => f.server_timeout_s = num(key, value)?,
                "max_bed_extend_s" => f.max_bed_extend_s = num(key, value)?,
                "max_speech_extend_s" => f.max_speech_extend_s = num(key, value)?,
                "local_tool_rounds" => f.local_tool_rounds = num(key, value)?,
                "roomy_tool_rounds" => f.roomy_tool_rounds = num(key, value)?,
                "local_tool_result_chars" => f.local_tool_result_chars = num(key, value)?,
                "roomy_tool_result_chars" => f.roomy_tool_result_chars = num(key, value)?,
                "max_shake_jerk" => f.max_shake_jerk = num(key, value)?,
                "shake_relative" => f.shake_relative = num(key, value)?,
                "max_sway" => f.max_sway = num(key, value)?,
                "shake_window_s" => f.shake_window_s = num(key, value)?,
                "shake_stride_s" => f.shake_stride_s = num(key, value)?,
                other => return Err(format!("unknown config key 'script.{other}'")),
            }
            return self.script.validate();
        }
        match key {
            "stt.backend" => self.stt.backend = backend(value)?,
            "stt.url" => self.stt.url = value.to_string(),
            "embed.backend" | "embeddings.backend" => self.embed.backend = backend(value)?,
            "embed.url" | "embeddings.url" => self.embed.url = value.to_string(),
            "frames.max_interval_s" => {
                let v: f64 = num(key, value)?;
                if !(1.0..=60.0).contains(&v) {
                    return Err(format!("frames.max_interval_s must be between 1 and 60, got {v}"));
                }
                self.frames.max_interval_s = v;
            }
            "frames.change_budget" => {
                let v: f64 = num(key, value)?;
                if !(0.0..=20.0).contains(&v) {
                    return Err(format!("frames.change_budget must be between 0 (off) and 20, got {v}"));
                }
                self.frames.change_budget = v;
            }
            "frames.min_interval_s" => {
                let v: f64 = num(key, value)?;
                if !(0.5..=30.0).contains(&v) {
                    return Err(format!("frames.min_interval_s must be between 0.5 and 30, got {v}"));
                }
                self.frames.min_interval_s = v;
            }
            other => {
                return Err(format!(
                    "unknown or unsupported config key '{other}'; supported keys: \
                     vision.* and chat_model.* (backend, url, model, local_model, ctx_tokens, kv_cache, \
                     flash_attn, think), vision.cli.* and chat_model.cli.* (tool, command, model, \
                     timeout_secs, concurrency), stt.backend, stt.url, embed.backend, embed.url, \
                     frames.max_interval_s"
                ));
            }
        }
        Ok(())
    }

    /// Check every section that has rules. Returns the first problem as a message.
    pub fn validate(&self) -> Result<(), String> {
        self.vision.validate("vision")?;
        self.chat_model().validate("chat_model")?;
        self.script.validate()?;
        self.jev.validate()?;
        self.frames.validate()
    }

    /// Load `path`, or defaults when the file does not exist.
    pub fn load(path: &Path) -> Result<Self, Error> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(Error::Io(path.to_path_buf(), e)),
        }
    }

    pub fn to_toml(&self) -> Result<String, Error> {
        toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let text = self.to_toml()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
        }
        std::fs::write(path, text).map_err(|e| Error::Io(path.to_path_buf(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_and_chat_have_their_own_model_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        // Defaults: same model, different windows.
        let cfg = Config::default();
        assert_eq!(cfg.vision.ctx_tokens, DESCRIBE_CTX_TOKENS);
        assert_eq!(cfg.chat_model().ctx_tokens, CHAT_CTX_TOKENS);
        assert_eq!(cfg.chat_model().local_model, cfg.vision.local_model);
        assert_eq!(cfg.vision.kv_cache, "q4_0");

        // A file written before the split: the chat inherits vision's backend and model.
        std::fs::write(
            &path,
            "[vision]\nbackend = \"server\"\nurl = \"http://x:1/v1\"\nlocal_model = \"qwen2.5-vl-3b\"\n",
        )
        .unwrap();
        let old = Config::load(&path).unwrap();
        let chat = old.chat_model();
        assert_eq!(chat.backend, Backend::Server);
        assert_eq!(chat.url, "http://x:1/v1");
        assert_eq!(chat.local_model, "qwen2.5-vl-3b");
        assert_eq!(chat.ctx_tokens, CHAT_CTX_TOKENS, "but with the chat's window");

        // Once set, the two are independent and survive a roundtrip.
        let mut cfg = Config::default();
        cfg.vision.ctx_tokens = 4096;
        cfg.chat_model = Some(VisionConfig { ctx_tokens: 65536, kv_cache: "q8_0".into(), ..VisionConfig::default() });
        cfg.save(&path).unwrap();
        let back = Config::load(&path).unwrap();
        assert_eq!(back, cfg);
        assert_eq!(back.vision.ctx_tokens, 4096);
        assert_eq!(back.chat_model().ctx_tokens, 65536);
        assert_eq!(back.chat_model().kv_cache, "q8_0");

        // Validation.
        assert!(VisionConfig { ctx_tokens: 1024, ..VisionConfig::default() }.validate("vision").is_err());
        assert!(VisionConfig { kv_cache: "q2_k".into(), ..VisionConfig::default() }.validate("vision").is_err());
        assert!(VisionConfig { flash_attn: "maybe".into(), ..VisionConfig::default() }.validate("chat_model").is_err());
        assert!(back.validate().is_ok());
    }

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load(&dir.path().join("nope.toml")).unwrap();
        assert_eq!(cfg, Config::default());
        assert_eq!(cfg.embed.url, DEFAULT_EMBED_URL);
    }

    #[test]
    fn roundtrip_and_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/config.toml");
        let mut cfg = Config::default();
        cfg.vision.backend = Backend::Server;
        cfg.stt.backend = Backend::Local;
        cfg.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), cfg);

        // Unspecified sections/keys fall back to defaults.
        std::fs::write(&path, "[vision]\nbackend = \"local\"\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.vision.backend, Backend::Local);
        assert_eq!(cfg.vision.url, DEFAULT_VISION_URL);
        assert_eq!(cfg.stt, SttConfig::default());
    }

    #[test]
    fn invalid_backend_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[vision]\nbackend = \"cloud\"\n").unwrap();
        assert!(matches!(Config::load(&path), Err(Error::Config(_))));
    }

    #[test]
    fn models_dir_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[models]\ndir = \"/custom/models\"\nsearch_paths = [\"/extra/path\"]\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.models.dir, Some(PathBuf::from("/custom/models")));
        assert_eq!(cfg.models.search_paths, vec![PathBuf::from("/extra/path")]);
    }

    #[test]
    fn vision_local_model_default_and_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::default();
        assert_eq!(cfg.vision.local_model, "bonsai-27b");

        std::fs::write(&path, "[vision]\nlocal_model = \"gemma-3-4b-it\"\n").unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.vision.local_model, "gemma-3-4b-it");
        assert_eq!(loaded.vision.backend, Backend::Auto);

        let mut saved_cfg = Config::default();
        saved_cfg.vision.local_model = "qwen2.5-vl-7b".into();
        saved_cfg.save(&path).unwrap();
        let roundtrip = Config::load(&path).unwrap();
        assert_eq!(roundtrip.vision.local_model, "qwen2.5-vl-7b");
    }

    #[test]
    fn frames_config_default_is_8s() {
        let cfg = Config::default();
        assert_eq!(cfg.frames.max_interval_s, 8.0);
        assert_eq!(cfg.frames.clamped_interval(), 8.0);
        assert!(cfg.frames.validate().is_ok());
    }

    #[test]
    fn frames_config_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut cfg = Config::default();
        cfg.frames.max_interval_s = 5.0;
        cfg.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.frames.max_interval_s, 5.0);

        // Partial file: no [frames] section → default 8 s
        std::fs::write(&path, "[vision]\nbackend = \"local\"\n").unwrap();
        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.frames.max_interval_s, 8.0);
    }

    #[test]
    fn frames_config_clamping_and_validation() {
        let mut fc = FramesConfig { max_interval_s: 0.5, ..Default::default() };
        assert_eq!(fc.clamped_interval(), 1.0);
        assert!(fc.validate().is_err());

        fc.max_interval_s = 90.0;
        assert_eq!(fc.clamped_interval(), 60.0);
        assert!(fc.validate().is_err());

        fc.max_interval_s = 15.0;
        assert_eq!(fc.clamped_interval(), 15.0);
        assert!(fc.validate().is_ok());
    }

    #[test]
    fn cli_agent_config_defaults_and_validation() {
        let default = CliAgentConfig::default();
        assert_eq!(default.timeout_secs, 180);
        assert_eq!(default.concurrency, 2);
        assert!(default.tool.is_empty());
        // Empty tool is valid (not required when backend != cli in the config file).
        assert!(default.validate("vision").is_ok());

        // Valid tools pass.
        for tool in CLI_TOOLS {
            let cfg = CliAgentConfig { tool: tool.to_string(), ..CliAgentConfig::default() };
            assert!(cfg.validate("vision").is_ok(), "{tool} should be valid");
        }

        // Unknown tool fails.
        let bad_tool = CliAgentConfig { tool: "gpt4all".into(), ..CliAgentConfig::default() };
        assert!(bad_tool.validate("vision").is_err());

        // Zero timeout fails.
        let bad_timeout = CliAgentConfig { timeout_secs: 0, ..CliAgentConfig::default() };
        assert!(bad_timeout.validate("vision").is_err());

        // Zero concurrency fails.
        let bad_conc = CliAgentConfig { concurrency: 0, ..CliAgentConfig::default() };
        assert!(bad_conc.validate("vision").is_err());
    }

    #[test]
    fn cli_backend_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        // Write a config with backend = "cli" and cli settings.
        std::fs::write(
            &path,
            "[vision]\nbackend = \"cli\"\n[vision.cli]\ntool = \"claude\"\nmodel = \"claude-opus-4-5\"\ntimeout_secs = 300\nconcurrency = 3\n",
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.vision.backend, Backend::Cli);
        assert_eq!(cfg.vision.cli.tool, "claude");
        assert_eq!(cfg.vision.cli.model, "claude-opus-4-5");
        assert_eq!(cfg.vision.cli.timeout_secs, 300);
        assert_eq!(cfg.vision.cli.concurrency, 3);

        // Full roundtrip through save/load.
        cfg.save(&path).unwrap();
        let back = Config::load(&path).unwrap();
        assert_eq!(back, cfg);
        assert!(back.validate().is_ok());
    }
}
