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

pub use agent_loop::{ToolAccess, ToolTrail, run_agent_json};
pub use ctx::{Args, Grounding, Origin, ToolCtx, ToolError, ToolImage, ToolOutput, ToolResult};
pub use render::{RenderOpts, TOOL_CONTRACT, action_branches, render_mcp, render_openai, tools_prose};
pub use session::{Called, CallGuard, ToolBudget, ToolMemo, ToolSession};
pub use spec::{Detail, Effect, ParamType, ToolParam, ToolSpec, params_schema};

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
