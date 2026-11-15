//! The registry, rendered for each surface. Three renderings of one catalog, so a fact stated to
//! one backend is stated to all: the MCP `tools/list`, the OpenAI `tools` array a server backend
//! receives, and the prose block the local and CLI action loops read in their prompt.

use serde_json::{Value, json};

use super::ToolProfile;
use super::spec::{Detail, Effect, ToolSpec};

/// The rules that apply to every tool, stated once.
pub const TOOL_CONTRACT: &str = "Every clip you write must lie inside a range one of these tools \
    returned for that video. A range nothing returned is checked against the index and dropped if \
    nothing was ever described or spoken there — so look before you cut. Opening a video with \
    get_video makes the whole of that video legal to cut, get_manifest or get_transcript the range it read; a search hit makes \
    only the window it returned legal; list_videos makes nothing legal.";

/// How a rendering is adjusted for who reads it.
#[derive(Clone, Copy, Debug)]
pub struct RenderOpts {
    pub detail: Detail,
    /// The reader works on one project already: `project` is optional in every schema.
    pub project_default: bool,
    /// Which subset of tools is exposed.
    pub profile: ToolProfile,
}

impl RenderOpts {
    pub fn new(detail: Detail, project_default: bool) -> Self {
        Self { detail, project_default, profile: ToolProfile::All }
    }

    pub fn with_profile(mut self, profile: ToolProfile) -> Self {
        self.profile = profile;
        self
    }
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self { detail: Detail::Full, project_default: false, profile: ToolProfile::All }
    }
}

/// The arguments' schema for a reader. `Short` leaves out the parameters' docs: a small model
/// gets the names and types, and the tool's one-line summary says the rest.
fn input_schema(spec: &ToolSpec, opts: RenderOpts, docs: bool) -> Value {
    spec.input_schema(opts.project_default, docs)
}

/// The MCP `tools/list` result entries for the full registry.
pub fn render_mcp() -> Vec<Value> {
    render_mcp_for(ToolProfile::All)
}

/// The MCP `tools/list` result entries for `profile`: name, description, input schema, and
/// annotations taken from what the tool does (`readOnlyHint`, `destructiveHint`).
pub fn render_mcp_for(profile: ToolProfile) -> Vec<Value> {
    render_mcp_with(RenderOpts::new(Detail::Full, false).with_profile(profile))
}

/// The MCP `tools/list` result entries configured by `opts`.
pub fn render_mcp_with(opts: RenderOpts) -> Vec<Value> {
    super::for_profile(opts.profile)
        .into_iter()
        .map(|t| {
            let s = t.spec;
            json!({
                "name": s.name,
                "description": s.describe(opts.detail),
                "inputSchema": s.input_schema(opts.project_default, opts.detail == Detail::Full),
                "annotations": {
                    "readOnlyHint": s.effect == Effect::Read,
                    "destructiveHint": s.effect == Effect::Destructive,
                },
            })
        })
        .collect()
}

/// The OpenAI `tools` array a server backend is sent.
pub fn render_openai(opts: RenderOpts) -> Value {
    Value::Array(
        super::for_profile(opts.profile)
            .into_iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.spec.name,
                        "description": t.spec.describe(opts.detail),
                        "parameters": input_schema(t.spec, opts, opts.detail == Detail::Full),
                    }
                })
            })
            .collect(),
    )
}

/// The `oneOf` branches `{action: "tool", tool: <name>, args: <schema>}` of a grammar-constrained
/// action loop: each tool's arguments constrained by its own schema, so a numeric `video_id`
/// cannot be a filename and a research round is not wasted on one. Without the parameters' docs,
/// which a grammar ignores and the prompt already carries.
pub fn action_branches(opts: RenderOpts) -> Vec<Value> {
    super::for_profile(opts.profile)
        .into_iter()
        .map(|t| {
            json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["tool"] },
                    "tool": { "type": "string", "enum": [t.spec.name] },
                    "args": input_schema(t.spec, opts, false),
                },
                "required": ["action", "tool", "args"],
                "additionalProperties": false
            })
        })
        .collect()
}

/// The TOOLS block of a prompt, for the loops that read prose rather than a schema. Every tool
/// in `opts.profile` is listed at every detail level: `Short` drops the prose around a tool and
/// the parameters' docs, not the tool.
pub fn tools_prose(opts: RenderOpts) -> String {
    let mut out = String::from("\n\nTOOLS\n");
    out.push_str(TOOL_CONTRACT);
    out.push('\n');
    for t in super::for_profile(opts.profile) {
        let spec = t.spec;
        let args: Vec<String> = spec
            .params
            .iter()
            .map(|p| {
                let optional = !p.required || (opts.project_default && p.name == "project");
                if optional { format!("{}?", p.name) } else { p.name.to_string() }
            })
            .collect();
        let marker = match spec.effect {
            Effect::Read => "",
            Effect::Write => " [changes things]",
            Effect::Destructive => " [destructive]",
        };
        out.push_str(&format!("- {}({}){marker}: {}\n", spec.name, args.join(", "), spec.describe(opts.detail)));
        if opts.detail == Detail::Full {
            for p in spec.params {
                out.push_str(&format!("    {} ({}) — {}\n", p.name, p.ty.label(), p.doc));
            }
        }
    }
    out
}
