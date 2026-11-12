//! What a tool handler is given and what it gives back.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::Value;

use crate::config::Config;
use crate::db::Db;
use crate::embed::Embedder;
use crate::paths::Paths;
use crate::projects::Project;

/// Where a tool call comes from. A tool that starts a model run refuses to start one from inside
/// another (see [`ToolError::Nested`]): the run holds its model's slot, and a second one would
/// wait for it forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// An agent outside GhostReel (MCP) or a person at the CLI.
    External,
    /// The script chat (Sonar).
    ScriptChat,
    /// A clip run's workers or orchestrator.
    ClipRun,
}

impl Origin {
    /// Who to record as the source of a change made by a tool ("cli", "script chat", "clip run").
    pub fn source(self) -> &'static str {
        match self {
            Origin::External => "cli",
            Origin::ScriptChat => "script chat",
            Origin::ClipRun => "clip run",
        }
    }

    /// "a script chat", for "already inside a script chat".
    pub fn article(self) -> &'static str {
        match self {
            Origin::External => "an outside agent",
            Origin::ScriptChat => "a script chat",
            Origin::ClipRun => "a clip run",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("unknown tool '{0}'")]
    Unknown(String),
    /// A missing or malformed argument, or a value the footage cannot satisfy.
    #[error("{0}")]
    Invalid(String),
    /// A tool that starts a model run, called from inside one.
    #[error("{tool} starts its own run and this call is already inside {}; do the work with the other tools instead", .inside.article())]
    Nested { tool: &'static str, inside: Origin },
    #[error("stopped")]
    Cancelled,
    #[error(transparent)]
    Core(#[from] crate::Error),
}

impl ToolError {
    pub fn invalid(msg: impl Into<String>) -> Self {
        ToolError::Invalid(msg.into())
    }

    /// Whether the call was stopped rather than failed.
    pub fn is_cancelled(&self) -> bool {
        match self {
            ToolError::Cancelled => true,
            ToolError::Core(e) => e.is_cancelled(),
            _ => false,
        }
    }
}

impl From<rusqlite::Error> for ToolError {
    fn from(e: rusqlite::Error) -> Self {
        ToolError::Core(e.into())
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(e: serde_json::Error) -> Self {
        ToolError::Invalid(e.to_string())
    }
}

/// A picture a tool shows the caller (JPEG), with what it is a picture of.
#[derive(Debug, Clone)]
pub struct ToolImage {
    pub caption: String,
    pub jpeg: Vec<u8>,
}

/// What a tool returns.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// The structured result.
    pub value: Value,
    /// A text rendering that reads better than the JSON (a transcript). `None`: the JSON is it.
    pub text: Option<String>,
    pub images: Vec<ToolImage>,
    /// One line for the activity log: "3 hits", "saved script #12".
    pub summary: String,
    /// The call found nothing it was asked for (a search with no hits).
    pub empty: bool,
}

impl ToolOutput {
    pub fn json(value: Value) -> Self {
        Self { value, ..Self::default() }
    }

    pub fn summary(mut self, s: impl Into<String>) -> Self {
        self.summary = s.into();
        self
    }

    pub fn text(mut self, t: impl Into<String>) -> Self {
        self.text = Some(t.into());
        self
    }

    pub fn images(mut self, images: Vec<ToolImage>) -> Self {
        self.images = images;
        self
    }

    pub fn empty(mut self, empty: bool) -> Self {
        self.empty = empty;
        self
    }

    /// What a model that cannot see pictures is told.
    pub fn model_text(&self) -> String {
        match &self.text {
            Some(t) => t.clone(),
            None => self.value.to_string(),
        }
    }
}

pub type ToolResult = Result<ToolOutput, ToolError>;

/// Footage ranges the caller has been shown, so a script can be checked against them.
#[derive(Debug, Clone, Default)]
pub struct Grounding {
    pub ranges: Vec<(i64, f64, f64)>,
}

impl Grounding {
    pub fn add(&mut self, video_id: i64, start_s: f64, end_s: f64) {
        self.ranges.push((video_id, start_s, end_s));
    }

    /// Check if a clip lies inside a grounded range for that video (±`slack` s at both ends).
    /// Containment, not overlap: a 45 s clip touching a 5 s search hit is not grounded.
    pub fn is_grounded(&self, video_id: i64, in_s: f64, out_s: f64, slack: f64) -> bool {
        self.ranges.iter().any(|&(vid, s, e)| vid == video_id && in_s >= s - slack && out_s <= e + slack)
    }

    /// Record grounding ranges from an earlier call's arguments (a session's stored tool calls).
    pub fn record_tool_call(&mut self, tool: &str, args: &Value, db: &Db) {
        match crate::tools::canonical_name(tool) {
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

/// Everything a handler may use. Built by the caller of the registry; the same handler runs for
/// an MCP client, the script chat and a clip worker.
pub struct ToolCtx<'a> {
    pub paths: &'a Paths,
    pub config: &'a Config,
    pub db: &'a mut Db,
    /// The project the caller works on: the default for every `project` argument, and the scope
    /// video ids are checked against. `None` for an MCP client, which names its project.
    pub project: Option<Project>,
    pub origin: Origin,
    pub cancel: Option<Arc<AtomicBool>>,
    /// Records what was looked at (the script chat checks its draft against it).
    pub grounding: Option<&'a mut Grounding>,
    /// The embedding model, when the caller already has one.
    pub embedder: &'a mut Option<Embedder>,
    /// Start the embedding model on the first search when `embedder` is empty.
    pub embed_on_demand: bool,
    /// A size the caller would like results to fit in, in characters. Handlers with lists use it
    /// to sample; the session cuts whatever is still over.
    pub max_chars: Option<usize>,
    /// The caller can look at pictures. Without it a handler skips decoding frames.
    pub can_see: bool,
}

impl<'a> ToolCtx<'a> {
    pub fn new(
        paths: &'a Paths,
        config: &'a Config,
        db: &'a mut Db,
        embedder: &'a mut Option<Embedder>,
        origin: Origin,
    ) -> Self {
        Self {
            paths,
            config,
            db,
            project: None,
            origin,
            cancel: None,
            grounding: None,
            embedder,
            embed_on_demand: false,
            max_chars: None,
            can_see: origin == Origin::External,
        }
    }

    pub fn with_project(mut self, project: Project) -> Self {
        self.project = Some(project);
        self
    }

    pub fn with_cancel(mut self, cancel: Option<Arc<AtomicBool>>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn with_grounding(mut self, grounding: &'a mut Grounding) -> Self {
        self.grounding = Some(grounding);
        self
    }

    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = Some(max_chars);
        self
    }

    pub fn on_demand_embedder(mut self) -> Self {
        self.embed_on_demand = true;
        self
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst))
    }

    /// Fails with [`ToolError::Cancelled`] once the caller has asked to stop.
    pub fn check_cancel(&self) -> Result<(), ToolError> {
        if self.is_cancelled() { Err(ToolError::Cancelled) } else { Ok(()) }
    }

    /// Note that `[start_s, end_s]` of a video has been shown to the caller.
    pub fn ground(&mut self, video_id: i64, start_s: f64, end_s: f64) {
        if let Some(g) = self.grounding.as_deref_mut() {
            g.add(video_id, start_s, end_s);
        }
    }

    /// The project a call is about: its `project` argument (a name, or an id), else the current one.
    pub fn project_for(&self, args: Args<'_>) -> Result<Project, ToolError> {
        match args.get("project") {
            Some(Value::String(name)) if !name.trim().is_empty() => Ok(self.db.require_project(name)?),
            Some(v) if v.is_number() => {
                let id = v.as_i64().ok_or_else(|| ToolError::invalid("project must be a name or a whole number id"))?;
                Ok(self.db.project(id)?)
            }
            _ => self
                .project
                .clone()
                .ok_or_else(|| ToolError::invalid("missing argument 'project' (the project's name)")),
        }
    }

    /// Refuse a video outside the current project, when there is one. A caller without a current
    /// project (MCP) may read any video of the library.
    pub fn check_video(&self, video_id: i64) -> Result<(), ToolError> {
        if let Some(p) = &self.project
            && !crate::chat::is_video_in_project(self.db, p.id, video_id)
        {
            return Err(ToolError::invalid(format!("video #{video_id} not in project")));
        }
        Ok(())
    }

    /// A tool that starts a model run cannot be called from inside one.
    pub fn refuse_nested(&self, tool: &'static str) -> Result<(), ToolError> {
        match self.origin {
            Origin::External => Ok(()),
            inside => Err(ToolError::Nested { tool, inside }),
        }
    }
}

/// The arguments of a call, read leniently: a model writes `"5"` for 5 and `"true"` for true, and
/// a whole number as `5.0`.
#[derive(Clone, Copy)]
pub struct Args<'a>(pub &'a Value);

impl<'a> Args<'a> {
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        self.0.get(key).filter(|v| !v.is_null())
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn opt_str(&self, key: &str) -> Option<&'a str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn str(&self, key: &str) -> Result<&'a str, ToolError> {
        self.opt_str(key).ok_or_else(|| ToolError::Invalid(format!("missing string argument '{key}'")))
    }

    pub fn opt_i64(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn i64(&self, key: &str) -> Result<i64, ToolError> {
        self.opt_i64(key).ok_or_else(|| ToolError::Invalid(format!("missing integer argument '{key}'")))
    }

    pub fn opt_u64(&self, key: &str) -> Option<u64> {
        self.opt_i64(key).and_then(|v| u64::try_from(v).ok())
    }

    pub fn opt_f64(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn f64(&self, key: &str) -> Result<f64, ToolError> {
        self.opt_f64(key).ok_or_else(|| ToolError::Invalid(format!("missing number argument '{key}'")))
    }

    pub fn opt_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            Value::Bool(b) => Some(*b),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        self.opt_bool(key).unwrap_or(default)
    }

    /// A list of whole numbers (`[1, "2"]`); entries that are not numbers are skipped.
    pub fn i64_list(&self, key: &str) -> Vec<i64> {
        self.get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| Args(v).as_i64()).collect())
            .unwrap_or_default()
    }

    pub fn f64_list(&self, key: &str) -> Vec<f64> {
        self.get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))).collect())
            .unwrap_or_default()
    }

    pub fn str_list(&self, key: &str) -> Vec<String> {
        self.get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default()
    }

    /// The value itself as a whole number (for list entries).
    fn as_i64(&self) -> Option<i64> {
        match self.0 {
            Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64)),
            Value::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }
}
