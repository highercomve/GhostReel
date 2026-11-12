//! What an agent is: one model behind one endpoint, the jobs it may take and how many at once.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::config::{Backend, CLI_TOOLS, CliAgentConfig, Config, VisionConfig};

/// The kinds of work an agent can be given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentRole {
    /// Reads every worker's answer and makes the final choice (one long structured call).
    Orchestrator,
    /// Small structured-JSON jobs run in parallel (scoring a window of footage).
    Worker,
    /// Script chat: a conversation with tools that writes a cut.
    Chat,
    /// Describing keyframes while indexing.
    Vision,
}

impl AgentRole {
    pub const ALL: [AgentRole; 4] = [AgentRole::Orchestrator, AgentRole::Worker, AgentRole::Chat, AgentRole::Vision];

    pub fn as_str(self) -> &'static str {
        match self {
            AgentRole::Orchestrator => "orchestrator",
            AgentRole::Worker => "worker",
            AgentRole::Chat => "chat",
            AgentRole::Vision => "vision",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        Self::ALL.into_iter().find(|r| r.as_str() == s.trim()).ok_or_else(|| {
            format!("unknown agent role '{s}'; expected orchestrator, worker, chat or vision")
        })
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How hard a model thinks before it answers. Passed to the CLIs that take it (claude
/// `--effort`, codex `model_reasoning_effort`); the others ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Effort {
    pub const ALL: [Effort; 5] = [Effort::Low, Effort::Medium, Effort::High, Effort::Xhigh, Effort::Max];

    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh => "xhigh",
            Effort::Max => "max",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_lowercase();
        Self::ALL
            .into_iter()
            .find(|e| e.as_str() == s)
            .ok_or_else(|| format!("unknown effort '{s}'; expected low, medium, high, xhigh or max"))
    }

    /// `parse`, with an empty value or "default" meaning none.
    pub fn parse_opt(s: &str) -> Result<Option<Self>, String> {
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("default") { Ok(None) } else { Self::parse(t).map(Some) }
    }

    /// A legacy claude model id with the effort glued on ("claude-opus-5-5-medium") as the real
    /// model and the effort. Only claude's ids: other tools have real models ending in "-max"
    /// or "-high" ("gpt-5.1-codex-max").
    pub fn split_model(tool: &str, model: &str) -> Option<(String, Effort)> {
        let m = model.trim();
        let is_claude = tool == "claude" || m.starts_with("claude-");
        if !is_claude {
            return None;
        }
        let (base, suffix) = m.rsplit_once('-')?;
        let effort = Self::parse(suffix).ok()?;
        (!base.is_empty()).then(|| (base.to_string(), effort))
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The model fields a linked agent never stores: they come from its section.
pub const LINKED_FIELDS: &[&str] =
    &["backend", "tool", "model", "effort", "url", "api_key", "local_model", "ctx_tokens", "timeout_secs"];

/// An agent whose model settings live in an existing config section rather than on the agent.
/// Configs written before `[[agents]]` existed are read as linked agents, so they behave exactly
/// as before; the Settings sections for those capabilities keep editing them.
///
/// The section is the one source of truth: a linked agent's model fields ([`LINKED_FIELDS`]) are
/// derived from it when the config is read ([`crate::config::Config::normalize`]) and whenever
/// agents are listed ([`crate::config::Config::agents`]), and never written to the file. Setting
/// one of them on the agent (`agents.<id>.backend = …`) unlinks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSource {
    /// `[chat_model]` (or `[vision]` with the chat's window on older files).
    ChatModel,
    /// `[chat_worker]` laid over `[chat_model]`.
    ChatWorker,
    /// `[vision]`.
    Vision,
}

impl AgentSource {
    pub fn label(self) -> &'static str {
        match self {
            AgentSource::ChatModel => "Script chat model",
            AgentSource::ChatWorker => "Clip workers",
            AgentSource::Vision => "Frame descriptions",
        }
    }
}

/// One configured agent (`[[agents]]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSpec {
    /// Short handle used by selections, the CLI and the run log ("claude", "ghostpen").
    pub id: String,
    /// Display name; the id when empty.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Linked to a config section: the model fields below are ignored, the section's apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<AgentSource>,
    /// local, server or cli (`auto`: the server when it answers, else the local model).
    pub backend: Backend,
    /// CLI tool (claude, agy, opencode, codex) when `backend = "cli"`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub tool: String,
    /// Server model id, or the CLI's model; empty = the endpoint's default.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// How hard the model thinks (CLI agents that take it); `None` = the tool's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// OpenAI-compatible server base URL when `backend = "server"`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub url: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub api_key: String,
    /// Catalog id or `external:<path>` when `backend = "local"`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub local_model: String,
    /// Context window for a local model; 0 = the chat model's.
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub ctx_tokens: u32,
    /// Per-call timeout for a CLI tool; 0 = the default.
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub timeout_secs: u64,
    /// Calls this agent takes at once. A local model always takes one: it is one process on one GPU.
    pub max_concurrent: usize,
    pub roles: BTreeSet<AgentRole>,
    pub enabled: bool,
    /// Lower runs first when several agents have a free slot.
    pub priority: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

impl Default for AgentSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            source: None,
            backend: Backend::Cli,
            tool: String::new(),
            model: String::new(),
            effort: None,
            url: String::new(),
            api_key: String::new(),
            local_model: String::new(),
            ctx_tokens: 0,
            timeout_secs: 0,
            max_concurrent: 1,
            roles: BTreeSet::new(),
            enabled: true,
            priority: 100,
        }
    }
}

/// Most calls one agent may be allowed at once.
pub const MAX_CONCURRENT: usize = 32;

impl AgentSpec {
    pub fn display_name(&self) -> &str {
        if self.name.trim().is_empty() { &self.id } else { &self.name }
    }

    pub fn has_role(&self, role: AgentRole) -> bool {
        self.roles.contains(&role)
    }

    /// The model settings this agent runs with for `role`, in the shape the backend resolver
    /// speaks. Runtime knobs a spec does not carry (KV cache, flash attention…) come from the
    /// chat model's section, or `[vision]` for the vision role.
    pub fn llm_config(&self, config: &Config, role: AgentRole) -> VisionConfig {
        let mut cfg = match self.source {
            Some(AgentSource::ChatModel) => config.chat_model(),
            Some(AgentSource::ChatWorker) => config.chat_worker(),
            Some(AgentSource::Vision) => config.vision.clone(),
            None => {
                let mut base = if role == AgentRole::Vision { config.vision.clone() } else { config.chat_model() };
                base.backend = self.backend;
                if !self.url.trim().is_empty() {
                    base.url = self.url.clone();
                }
                base.model = self.model.clone();
                base.api_key = self.api_key.clone();
                if !self.local_model.trim().is_empty() {
                    base.local_model = self.local_model.clone();
                }
                if self.ctx_tokens > 0 {
                    base.ctx_tokens = self.ctx_tokens;
                }
                let defaults = CliAgentConfig::default();
                base.cli = CliAgentConfig {
                    tool: self.tool.clone(),
                    model: self.model.clone(),
                    effort: self.effort,
                    timeout_secs: if self.timeout_secs > 0 { self.timeout_secs } else { defaults.timeout_secs },
                    concurrency: self.max_concurrent.max(1),
                    ..defaults
                };
                if self.backend == Backend::Cli {
                    // The CLI's model is `cli.model`; `model` is a server field.
                    base.model.clear();
                }
                base
            }
        };
        // A worker answers one small structured question: reasoning only costs time there.
        if role == AgentRole::Worker {
            cfg.think = false;
        }
        cfg
    }

    /// The spec with a linked agent's model fields filled from its section, for showing what it
    /// really runs. Unlinked agents come back unchanged.
    pub fn effective(&self, config: &Config) -> AgentSpec {
        let Some(_) = self.source else { return self.clone() };
        let role = self.roles.iter().next().copied().unwrap_or(AgentRole::Chat);
        let llm = self.llm_config(config, role);
        AgentSpec {
            backend: llm.backend,
            tool: if llm.backend == Backend::Cli { llm.cli.tool.clone() } else { String::new() },
            model: if llm.backend == Backend::Cli { llm.cli.model.clone() } else { llm.model.clone() },
            effort: if llm.backend == Backend::Cli { llm.cli.effort } else { None },
            url: if matches!(llm.backend, Backend::Server | Backend::Auto) { llm.url.clone() } else { String::new() },
            api_key: String::new(),
            local_model: if matches!(llm.backend, Backend::Local | Backend::Auto) {
                llm.local_model.clone()
            } else {
                String::new()
            },
            ctx_tokens: llm.ctx_tokens,
            timeout_secs: llm.cli.timeout_secs,
            ..self.clone()
        }
    }

    /// Calls at once as configured, with a local model held to one. `auto` may still turn out
    /// local when it runs; the slot registry clamps it then.
    pub fn capacity(&self) -> usize {
        if self.backend == Backend::Local { 1 } else { self.max_concurrent.clamp(1, MAX_CONCURRENT) }
    }

    /// A legacy model id with the effort in it becomes the model and the effort (CLI agents).
    pub fn normalize_effort(&mut self) {
        if self.backend == Backend::Cli
            && let Some((model, effort)) = Effort::split_model(&self.tool, &self.model)
        {
            self.model = model;
            self.effort.get_or_insert(effort);
        }
    }

    /// One plain line: "cli claude · claude-opus-5-5 · medium effort", "server qwen3.5-9b @ http://…",
    /// "local qwen3.5-9b".
    pub fn summary(&self) -> String {
        match self.backend {
            Backend::Cli => {
                let mut line =
                    if self.model.is_empty() { format!("cli {}", self.tool) } else { format!("cli {} · {}", self.tool, self.model) };
                if let Some(e) = self.effort {
                    line.push_str(&format!(" · {e} effort"));
                }
                line
            }
            Backend::Server => format!(
                "server {} @ {}",
                if self.model.is_empty() { "(default model)" } else { &self.model },
                self.url
            ),
            Backend::Local => format!("local {}", self.local_model),
            Backend::Auto => format!("auto (server {} or local {})", self.url, self.local_model),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let id = self.id.trim();
        if id.is_empty() {
            return Err("an agent needs an id".into());
        }
        if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(format!("agent id '{id}' may only use letters, digits, '-' and '_'"));
        }
        if self.max_concurrent == 0 || self.max_concurrent > MAX_CONCURRENT {
            return Err(format!("agent '{id}': max_concurrent must be between 1 and {MAX_CONCURRENT}"));
        }
        if self.source.is_some() {
            return Ok(());
        }
        match self.backend {
            Backend::Cli => {
                if !CLI_TOOLS.contains(&self.tool.as_str()) {
                    return Err(format!(
                        "agent '{id}': tool must be one of {}, got '{}'",
                        CLI_TOOLS.join(", "),
                        self.tool
                    ));
                }
            }
            Backend::Server => {
                if !self.url.starts_with("http") {
                    return Err(format!("agent '{id}': a server agent needs an http(s) url, got '{}'", self.url));
                }
            }
            Backend::Local => {
                if self.local_model.trim().is_empty() {
                    return Err(format!("agent '{id}': a local agent needs local_model"));
                }
            }
            // The server at `url` when it answers, else `local_model` (or the chat model's).
            Backend::Auto => {}
        }
        Ok(())
    }
}

/// How many chat turns the chat model takes at once when nothing says otherwise.
fn chat_capacity(llm: &VisionConfig) -> usize {
    match llm.backend {
        Backend::Local => 1,
        Backend::Cli => llm.cli.concurrency.max(1),
        Backend::Server | Backend::Auto => 4,
    }
}

/// The agents a config without `[[agents]]` stands for, linked to the sections it has, with the
/// roles and capacities those sections always had: the chat model orchestrates and chats (and
/// scores clips when there is no `[chat_worker]`), `[chat_worker]` scores clips, `[vision]`
/// describes frames.
pub fn implicit_agents(config: &Config) -> Vec<AgentSpec> {
    let chat = config.chat_model();
    let has_worker = config.chat_worker.is_some();
    let mut chat_roles: BTreeSet<AgentRole> = [AgentRole::Orchestrator, AgentRole::Chat].into();
    if !has_worker {
        chat_roles.insert(AgentRole::Worker);
    }
    let mut out = vec![AgentSpec {
        id: "chat".into(),
        name: "Script chat model".into(),
        source: Some(AgentSource::ChatModel),
        backend: chat.backend,
        max_concurrent: if has_worker { chat_capacity(&chat) } else { config.worker_concurrency() },
        roles: chat_roles,
        priority: 10,
        ..AgentSpec::default()
    }];
    if has_worker {
        out.push(AgentSpec {
            id: "worker".into(),
            name: "Clip workers".into(),
            source: Some(AgentSource::ChatWorker),
            backend: config.chat_worker().backend,
            max_concurrent: config.worker_concurrency(),
            roles: [AgentRole::Worker].into(),
            priority: 20,
            ..AgentSpec::default()
        });
    }
    let vision_cap = match config.vision.backend {
        Backend::Cli => config.vision.cli.concurrency.max(1),
        _ => (config.vision.describe_concurrency as usize).max(1),
    };
    out.push(AgentSpec {
        id: "vision".into(),
        name: "Frame descriptions".into(),
        source: Some(AgentSource::Vision),
        backend: config.vision.backend,
        max_concurrent: vision_cap.min(MAX_CONCURRENT),
        roles: [AgentRole::Vision].into(),
        priority: 30,
        ..AgentSpec::default()
    });
    out
}

/// Implicit agents as explicit ones, for the first edit of a config that has none: the chat
/// model and vision stay linked (their Settings sections keep editing them); `[chat_worker]`
/// becomes a standalone agent, since agents replace that section.
pub fn materialize(config: &Config) -> Vec<AgentSpec> {
    implicit_agents(config)
        .into_iter()
        .map(|a| match a.source {
            Some(AgentSource::ChatWorker) => {
                let w = config.chat_worker();
                AgentSpec {
                    source: None,
                    backend: w.backend,
                    tool: w.cli.tool.clone(),
                    model: if w.backend == Backend::Cli { w.cli.model.clone() } else { w.model.clone() },
                    effort: if w.backend == Backend::Cli { w.cli.effort } else { None },
                    url: w.url.clone(),
                    api_key: w.api_key.clone(),
                    local_model: w.local_model.clone(),
                    ctx_tokens: config.chat_worker.as_ref().map(|c| c.ctx_tokens).unwrap_or(0),
                    timeout_secs: w.cli.timeout_secs,
                    ..a
                }
            }
            _ => a,
        })
        .collect()
}
