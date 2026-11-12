//! What a tool is, as data: its name, what it does to the world, its parameters, and the words
//! the model reads about it. Written once here and rendered for every surface (MCP, the OpenAI
//! `tools` array, the prose block of the action loops).

use serde_json::{Value, json};

/// What calling a tool changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Looks only; calling it again is free.
    Read,
    /// Creates or changes something (a script version, a setting, a person's name, indexed data).
    Write,
    /// Removes something that cannot be put back by another tool call.
    Destructive,
}

impl Effect {
    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Read => "read",
            Effect::Write => "write",
            Effect::Destructive => "destructive",
        }
    }
}

/// The type of one parameter. Const-constructible so a spec can be a `static`.
#[derive(Clone, Copy)]
pub enum ParamType {
    Integer,
    Number,
    String,
    Boolean,
    /// An object whose shape the parameter's doc describes.
    Object,
    Enum(&'static [&'static str]),
    List(&'static ParamType),
    /// An object with known fields.
    Shape(&'static [ToolParam]),
}

#[derive(Clone, Copy)]
pub struct ToolParam {
    pub name: &'static str,
    pub ty: ParamType,
    pub required: bool,
    /// Units, range, default, and what happens when it is left out.
    pub doc: &'static str,
}

impl ToolParam {
    pub const fn new(name: &'static str, ty: ParamType, required: bool, doc: &'static str) -> Self {
        Self { name, ty, required, doc }
    }
}

impl ParamType {
    /// The JSON Schema of the type, with the parameters' docs as descriptions when `docs`.
    pub fn schema(self, docs: bool) -> Value {
        match self {
            ParamType::Integer => json!({ "type": "integer" }),
            ParamType::Number => json!({ "type": "number" }),
            ParamType::String => json!({ "type": "string" }),
            ParamType::Boolean => json!({ "type": "boolean" }),
            ParamType::Object => json!({ "type": "object" }),
            ParamType::Enum(values) => json!({ "type": "string", "enum": values }),
            ParamType::List(inner) => json!({ "type": "array", "items": inner.schema(docs) }),
            ParamType::Shape(params) => params_schema(params, false, false, docs),
        }
    }

    /// A short name for the prose rendering: `integer`, `[integer]`, `a|b`.
    pub fn label(self) -> String {
        match self {
            ParamType::Integer => "integer".into(),
            ParamType::Number => "number".into(),
            ParamType::String => "string".into(),
            ParamType::Boolean => "boolean".into(),
            ParamType::Object => "object".into(),
            ParamType::Enum(values) => values.join("|"),
            ParamType::List(inner) => format!("[{}]", inner.label()),
            ParamType::Shape(_) => "object".into(),
        }
    }
}

/// The `{type: object, properties, required}` schema of a parameter list. With `top`, extra keys
/// are refused (a tool call is closed); `project_optional` drops `project` from `required` for
/// callers that already have a current project.
pub fn params_schema(params: &[ToolParam], top: bool, project_optional: bool, docs: bool) -> Value {
    let mut props = serde_json::Map::new();
    let mut required = Vec::new();
    for p in params {
        let mut s = p.ty.schema(docs);
        if docs && !p.doc.is_empty() {
            s["description"] = json!(p.doc);
        }
        props.insert(p.name.to_string(), s);
        if p.required && !(project_optional && p.name == "project") {
            required.push(json!(p.name));
        }
    }
    let mut schema = json!({ "type": "object", "properties": Value::Object(props), "required": required });
    if top {
        schema["additionalProperties"] = json!(false);
    }
    schema
}

pub struct ToolSpec {
    pub name: &'static str,
    pub effect: Effect,
    /// One sentence. Survives at every detail level.
    pub summary: &'static str,
    /// What must be true before calling it, and where it sits in the order of work.
    pub when: &'static str,
    /// Each field of the result, and what it means.
    pub returns: &'static str,
    /// What goes wrong when it is misused, costliest first: `misuse[0]` is the line that
    /// survives at [`Detail::Short`].
    pub misuse: &'static [&'static str],
    /// Which tool to call next, and why.
    pub chains_with: &'static [&'static str],
    pub params: &'static [ToolParam],
}

/// How much of each spec to render. A roomy brain gets everything; a small local model, whose
/// window is already half transcripts, gets the summary, the parameters and the costliest warning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Detail {
    Full,
    Short,
}

impl ToolSpec {
    /// The description a model reads.
    pub fn describe(&self, detail: Detail) -> String {
        let mut out = String::from(self.summary);
        if detail == Detail::Full {
            if !self.when.is_empty() {
                out.push(' ');
                out.push_str(self.when);
            }
            if !self.returns.is_empty() {
                out.push_str(" Returns: ");
                out.push_str(self.returns);
            }
            for line in self.chains_with {
                out.push(' ');
                out.push_str(line);
            }
        }
        // The costliest warning is worth its tokens even in the short rendering.
        let warnings: &[&str] = if detail == Detail::Full { self.misuse } else { &self.misuse[..self.misuse.len().min(1)] };
        for line in warnings {
            out.push(' ');
            out.push_str(line);
        }
        out
    }

    /// The JSON Schema of the arguments, with the parameters' docs when `docs`.
    pub fn input_schema(&self, project_optional: bool, docs: bool) -> Value {
        params_schema(self.params, true, project_optional, docs)
    }
}

/// The `project` parameter most tools take. Required for a caller that names its project (MCP);
/// optional for one that already works on a project (the script chat, clip workers).
pub const PROJECT: ToolParam = ToolParam::new(
    "project",
    ParamType::String,
    true,
    "Project name. Leave it out to use the project you are working on.",
);

/// `project`, for tools that work across projects when it is left out.
pub const PROJECT_OPTIONAL: ToolParam = ToolParam::new(
    "project",
    ParamType::String,
    false,
    "Project name. Leave it out to use the project you are working on (all projects if there is none).",
);
