//! The one tool registry.
//!
//! Every operation a model may ask GhostReel to do is defined here once: a [`ToolSpec`] (what it
//! is called, what it changes, its parameters and the words a model reads about it) and a handler
//! (a [`ToolCtx`] and the call's [`Args`] in, a [`ToolOutput`] out). Every surface exposes the same
//! set:
//!
//! - **MCP** (`ghostreel mcp`) lists the registry with [`render_mcp`] and dispatches `tools/call`
//!   into [`call`].
//! - **The script chat** (Sonar) renders it for the model with [`render_openai`] (server backends)
//!   or [`tools_prose`] (local and CLI action loops) and calls it through a [`ToolSession`], which
//!   adds what is particular to a chat: grounding, result size limits, one-line summaries, the
//!   optional call cap and the guard against a model repeating itself.
//! - **Clip workers and the orchestrator** run the same action loop ([`agent_loop`]) over the
//!   same renderings and the same session.
//!
//! What differs between callers is not the list but the context: who is calling ([`Origin`], so a
//! tool that starts a model run refuses to start one from inside another), the current project (the
//! default for every `project` argument), and how much the caller can take (`max_chars`, `can_see`).

mod agent_loop;
mod clips;
mod ctx;
mod instructions;
mod library;
mod people;
mod project;
mod render;
mod scripts;
mod session;
mod settings;
mod spec;
mod util;

use futures_util::future::BoxFuture;
use serde_json::Value;

pub(crate) use agent_loop::TempTrailFile;
pub use agent_loop::{ToolAccess, ToolTrail, run_agent_json};
pub use ctx::{Args, Grounding, Origin, ToolCtx, ToolError, ToolImage, ToolOutput, ToolResult};
pub use instructions::INSTRUCTIONS;
pub use render::{
    RenderOpts, TOOL_CONTRACT, action_branches, render_mcp, render_mcp_for, render_mcp_with, render_openai, tools_prose,
};
pub use session::{CallGuard, Called, ToolBudget, ToolMemo, ToolSession};
pub use spec::{Detail, Effect, ParamType, ToolParam, ToolSpec, params_schema};

/// Which subset of the registry a surface exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolProfile {
    /// The full registry (Sonar script chat, standard MCP server).
    #[default]
    All,
    /// Read-only footage, person, and clip-verification tools for clip runs.
    ClipRun,
}

/// The read-only tools offered to clip workers and orchestrators inside a clip run.
pub const CLIP_RUN_TOOLS: &[&str] = &[
    "get_outline",
    "list_videos",
    "search",
    "get_transcript",
    "get_manifest",
    "get_video",
    "get_frames",
    "get_steadiness",
    "get_faces",
    "get_scenes",
    "get_people",
    "get_speakers",
    "preview_crop",
    "check_clips",
];

/// A handler: runs one tool. Async, because search, indexing and the model runs are.
pub type Handler = for<'a, 'b> fn(&'a mut ToolCtx<'b>, Args<'a>) -> BoxFuture<'a, ToolResult>;

pub struct Tool {
    pub spec: &'static ToolSpec,
    pub run: Handler,
}

/// Wraps an `async fn(&mut ToolCtx, Args) -> ToolResult` as a registry entry.
macro_rules! tool {
    ($spec:expr, $handler:path) => {
        $crate::tools::Tool { spec: &$spec, run: |ctx, args| Box::pin($handler(ctx, args)) }
    };
}
pub(crate) use tool;

/// Names models used before the registry had one name per tool. Accepted by [`call`] and
/// [`lookup`], never listed: old sessions and logs still carry them.
const ALIASES: &[(&str, &str)] = &[("search_moments", "search")];

/// The registry's name for `name` (an alias resolves to its tool; anything else is returned as is).
pub fn canonical_name(name: &str) -> &str {
    ALIASES.iter().find(|(old, _)| *old == name).map(|(_, new)| *new).unwrap_or(name)
}

/// Every tool, in the order they are listed to a model: looking at footage first, then the
/// people in it, scripts, clips, then running the project and the machine.
pub fn all() -> impl Iterator<Item = &'static Tool> {
    [
        library::TOOLS.iter(),
        people::TOOLS.iter(),
        scripts::TOOLS.iter(),
        clips::TOOLS.iter(),
        project::TOOLS.iter(),
        settings::TOOLS.iter(),
    ]
    .into_iter()
    .flatten()
}

/// Tools included in `profile`, in registry order.
pub fn for_profile(profile: ToolProfile) -> Vec<&'static Tool> {
    match profile {
        ToolProfile::All => all().collect(),
        ToolProfile::ClipRun => all().filter(|t| CLIP_RUN_TOOLS.contains(&t.spec.name)).collect(),
    }
}

/// Tool names included in `profile`, in listing order.
pub fn profile_names(profile: ToolProfile) -> Vec<&'static str> {
    for_profile(profile).into_iter().map(|t| t.spec.name).collect()
}

/// Whether `name` (or its canonical alias) is permitted under `profile`.
pub fn allows(profile: ToolProfile, name: &str) -> bool {
    let name = canonical_name(name);
    match profile {
        ToolProfile::All => lookup(name).is_some(),
        ToolProfile::ClipRun => CLIP_RUN_TOOLS.contains(&name),
    }
}

pub fn lookup(name: &str) -> Option<&'static Tool> {
    let name = canonical_name(name);
    all().find(|t| t.spec.name == name)
}

/// The names the action schema accepts, in listing order.
pub fn names() -> Vec<&'static str> {
    all().map(|t| t.spec.name).collect()
}

/// Run one tool.
pub async fn call(ctx: &mut ToolCtx<'_>, name: &str, args: &Value) -> ToolResult {
    let tool = lookup(name).ok_or_else(|| ToolError::Unknown(name.to_string()))?;
    // An argument the tool does not take is an error, not something to drop: a model that passes
    // `from`/`to` to a tool without them otherwise gets the whole video back and never learns why.
    if let Some(obj) = args.as_object() {
        // `project` is the one exception: every editor is told it may pass it to any tool, and a tool
        // that works from a video id simply has no use for it — refusing it costs the model a round.
        let unknown: Vec<&str> = obj
            .keys()
            .map(String::as_str)
            .filter(|k| *k != "project" && !tool.spec.params.iter().any(|p| p.name == *k))
            .collect();
        if !unknown.is_empty() {
            let takes: Vec<&str> = tool.spec.params.iter().map(|p| p.name).collect();
            return Err(ToolError::Invalid(format!(
                "{} does not take {}; it takes: {}",
                tool.spec.name,
                unknown.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(", "),
                takes.join(", ")
            )));
        }
    }
    let args = Args(args);
    for p in tool.spec.params.iter().filter(|p| p.required) {
        // The current project stands in for a missing `project`.
        let satisfied = args.has(p.name) || (p.name == "project" && ctx.project.is_some());
        if !satisfied {
            return Err(ToolError::Invalid(format!("missing argument '{}'", p.name)));
        }
    }
    (tool.run)(ctx, args).await
}

#[cfg(test)]
pub(crate) mod tests;
