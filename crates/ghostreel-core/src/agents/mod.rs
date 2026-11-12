//! The agent pool: configurable model agents across backends (a local model, OpenAI-compatible
//! servers, coding-agent CLIs), shared call slots per endpoint, and a dispatcher that spreads
//! parallel jobs over them.
//!
//! This module knows nothing about what the jobs are for. A feature asks for the agents of a role
//! ([`AgentRole`]) under a selection ([`AgentSelection`], resolved run → cut → app by
//! [`resolve_selection`]), gets an [`AgentPool`] and hands it jobs. Every model call in the app —
//! pool jobs, script chat turns, frame descriptions — takes a [`Slot`] for its endpoint from the
//! process-wide [`SlotRegistry`], so features running side by side never ask a model for more
//! than it allows; a local model has exactly one.
//!
//! Configs without `[[agents]]` are read as [`implicit_agents`] linked to `[chat_model]`,
//! `[chat_worker]` and `[vision]`, and behave as they always did.

mod cooldown;
mod error;
mod pool;
mod preflight;
mod selection;
mod slots;
mod spec;

pub use pool::{
    AgentJob, AgentLoad, AgentPool, AgentTarget, Attempt, JobOutcome, JobRunner, ModelRunner, PoolAction, PoolAgent,
    PoolEvent, PoolProgress,
    Skipped, running_line, share_line,
};
pub use selection::{
    AgentSelection, ResolvedSelection, SelectionError, SelectionSource, WorkTarget, resolve_selection, validate_selection,
    with_role,
};
pub use slots::{EndpointKey, EndpointSlots, EndpointUse, Slot, SlotRegistry, acquire_for};
pub use cooldown::{Cooldown, Cooldowns, call_once};
pub use error::{
    AgentError, AgentErrorKind, Handling, classify_cli, classify_http, classify_text, from_request as error_from_request,
    from_response as error_from_response, human_duration, resets_in,
};
pub use preflight::{agent_checks, check_agent};
pub use spec::{
    AgentRole, AgentSource, AgentSpec, Effort, LINKED_FIELDS, MAX_CONCURRENT, implicit_agents, materialize,
};

use serde::Serialize;

use crate::config::{Backend, Config};
use crate::paths::Paths;
use crate::preflight::Level;
use crate::runtime::VisionSetup;

/// Whether an agent can run on this machine right now.
#[derive(Debug, Clone, Serialize)]
pub struct Availability {
    pub id: String,
    /// Not `Error`: it can take jobs (a `Warn` says what may still go wrong).
    pub ok: bool,
    pub level: Level,
    /// What it resolved to ("cli claude (claude-opus-5-5)", "qwen3.5-9b @ http://…").
    pub resolved: String,
    /// One line on what was checked or what is wrong.
    pub detail: String,
    /// Whether the configured model is among those the endpoint lists. `None` when no model is
    /// set or the endpoint cannot list its models.
    pub model_listed: Option<bool>,
    /// The typed reason when it cannot be used (resting after a used-up quota, not installed…).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentError>,
}

impl Availability {
    fn new(id: String, level: Level, resolved: String, detail: String, model_listed: Option<bool>) -> Self {
        Self { id, ok: level < Level::Error, level, resolved, detail, model_listed, error: None }
    }
}

/// Check one agent: the CLI is installed, the server answers, the local model is present, the
/// endpoint is not resting after a used-up quota ([`Cooldowns`]) — and whether the model it names
/// is one the endpoint offers (a `Warn` when a CLI does not list it). Makes no model call.
pub async fn check_availability(paths: &Paths, config: &Config, spec: &AgentSpec) -> Availability {
    let role = spec.roles.iter().next().copied().unwrap_or(AgentRole::Chat);
    let setup = crate::runtime::resolve_agent(paths, config, spec, role).await;
    let id = spec.id.clone();
    let resolved = setup.describe();
    if let Some(c) = EndpointKey::for_setup(&setup).and_then(|k| Cooldowns::global().get(&k)) {
        let mut av = Availability::new(id, Level::Error, resolved, c.reason(), None);
        av.error = Some(c.error.clone().for_agent(&spec.id));
        return av;
    }
    match &setup {
        VisionSetup::Unavailable(why) => {
            let mut av = Availability::new(id.clone(), Level::Error, resolved, why.clone(), None);
            // A server that does not answer is unreachable; anything else is missing here.
            let kind = if spec.llm_config(config, role).backend == Backend::Server {
                AgentErrorKind::Unreachable
            } else {
                AgentErrorKind::NotInstalled
            };
            av.error = Some(AgentError::new(id, kind, why.clone()));
            av
        }
        VisionSetup::Cli(cfg) => {
            let probe = cfg.clone();
            let models = tokio::task::spawn_blocking(move || crate::cliagent::list_models(&probe)).await.unwrap_or_default();
            let path = crate::cliagent::find_binary(cfg).map(|p| p.display().to_string()).unwrap_or_default();
            let model_listed = (!cfg.model.is_empty() && !models.is_empty()).then(|| models.contains(&cfg.model));
            let (level, detail) = match model_listed {
                Some(false) => (
                    Level::Warn,
                    format!("installed at {path}; model '{}' is not among those {} lists", cfg.model, cfg.tool),
                ),
                _ => (Level::Ok, format!("installed at {path}")),
            };
            Availability::new(id, level, resolved, detail, model_listed)
        }
        VisionSetup::Server(s) => match crate::probe::server_models(&s.url).await {
            Ok(models) => {
                let model_listed = (!s.model.is_empty()).then(|| models.contains(&s.model));
                let (level, detail) = match model_listed {
                    Some(false) => (
                        Level::Error,
                        format!("server answers, but does not list '{}' ({} models)", s.model, models.len()),
                    ),
                    _ => (
                        Level::Ok,
                        format!("server answers with {} model{}", models.len(), if models.len() == 1 { "" } else { "s" }),
                    ),
                };
                let mut av = Availability::new(id.clone(), level, resolved, detail.clone(), model_listed);
                if level == Level::Error {
                    av.error = Some(AgentError::new(id, AgentErrorKind::InvalidModel { model: s.model.clone() }, detail));
                }
                av
            }
            Err(e) => {
                let detail = format!("server not reachable: {e}");
                let mut av = Availability::new(id.clone(), Level::Error, resolved, detail.clone(), None);
                av.error = Some(AgentError::new(id, AgentErrorKind::Unreachable, detail));
                av
            }
        },
        VisionSetup::Local { model, found, .. } => {
            let present = found.iter().any(|p| p.file_name().is_some_and(|n| n == model.file_name.as_str()) || p == std::path::Path::new(&model.file_name));
            let detail = if present {
                "model file present; one at a time on this computer".to_string()
            } else {
                format!("model {} downloads on first use", model.file_name)
            };
            Availability::new(id, Level::Ok, resolved, detail, Some(present))
        }
        VisionSetup::Mlx { entry, models_dir, .. } => {
            let present = entry.bundle_installed(models_dir);
            let detail = if present { "MLX model present".into() } else { "MLX model downloads on first use".into() };
            Availability::new(id, Level::Ok, resolved, detail, Some(present))
        }
    }
}

/// How an agent stands right now, for the dot beside it in a picker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AgentState {
    /// Can take a call now.
    Ready,
    /// Every call slot of its endpoint is taken.
    Busy { in_use: usize, capacity: usize },
    /// Its quota is used up; it rests until `until` (unix seconds).
    OutOfQuota { until: i64 },
    NotInstalled,
    Unreachable,
    /// Something else stops it (wrong model, not signed in…): see the detail.
    Unavailable,
}

impl AgentState {
    /// "ready", "busy", "out of quota"… (the picker adds the reset time).
    pub fn label(&self) -> &'static str {
        match self {
            AgentState::Ready => "ready",
            AgentState::Busy { .. } => "busy",
            AgentState::OutOfQuota { .. } => "out of quota",
            AgentState::NotInstalled => "not installed",
            AgentState::Unreachable => "unreachable",
            AgentState::Unavailable => "unavailable",
        }
    }
}

/// One agent's [`AgentState`], with its label and one line on why.
#[derive(Debug, Clone, Serialize)]
pub struct AgentStatus {
    pub id: String,
    #[serde(flatten)]
    pub state: AgentState,
    pub label: String,
    pub detail: String,
}

/// Where an agent stands now: resting after a used-up quota ([`Cooldowns`]), missing or
/// unreachable ([`check_availability`]), every slot of its endpoint taken ([`SlotRegistry`]), or
/// ready. Makes no model call.
pub async fn agent_status(paths: &Paths, config: &Config, spec: &AgentSpec) -> AgentStatus {
    let role = spec.roles.iter().next().copied().unwrap_or(AgentRole::Chat);
    let setup = crate::runtime::resolve_agent(paths, config, spec, role).await;
    let key = EndpointKey::for_setup(&setup);
    let status = |state: AgentState, detail: String| AgentStatus { id: spec.id.clone(), label: state.label().into(), state, detail };
    if let Some(c) = key.as_ref().and_then(|k| Cooldowns::global().get(k)) {
        return status(AgentState::OutOfQuota { until: c.until }, c.reason());
    }
    let av = check_availability(paths, config, spec).await;
    if !av.ok {
        let state = match av.error.as_ref().map(|e| &e.kind) {
            Some(AgentErrorKind::NotInstalled) => AgentState::NotInstalled,
            Some(AgentErrorKind::Unreachable) => AgentState::Unreachable,
            Some(AgentErrorKind::QuotaExhausted { resets_in_s }) => AgentState::OutOfQuota {
                until: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0)
                    + resets_in_s.unwrap_or(0) as i64,
            },
            _ => AgentState::Unavailable,
        };
        return status(state, av.detail);
    }
    if let Some(k) = &key
        && let Some(u) = SlotRegistry::global().snapshot().into_iter().find(|u| &u.key == k)
        && u.in_use >= u.capacity
    {
        return status(AgentState::Busy { in_use: u.in_use, capacity: u.capacity }, format!("{} of {} calls in flight", u.in_use, u.capacity));
    }
    status(AgentState::Ready, av.detail)
}

impl Config {
    /// The agents this config has: `[[agents]]`, or the implicit ones its sections stand for.
    ///
    /// Linked agents come back with their model fields derived from their sections, so every
    /// caller sees the one value the section holds.
    pub fn agents(&self) -> Vec<AgentSpec> {
        let list = if self.agents.is_empty() { implicit_agents(self) } else { self.agents.clone() };
        list.into_iter().map(|a| a.effective(self)).collect()
    }

    /// Resolve a selection against this config's agents and app default.
    pub fn resolve_agents(
        &self,
        run: Option<&AgentSelection>,
        cut: Option<&AgentSelection>,
    ) -> Result<ResolvedSelection, SelectionError> {
        resolve_selection(&self.agents(), run, cut, &self.agent_selection)
    }

    /// The model settings frame descriptions run with: the first enabled vision agent's, or
    /// `[vision]` when none is set up.
    pub fn vision_llm(&self) -> crate::config::VisionConfig {
        match with_role(&self.agents(), AgentRole::Vision).first() {
            Some(a) => a.llm_config(self, AgentRole::Vision),
            None => self.vision.clone(),
        }
    }

    /// Turn the implicit agents into `[[agents]]` (first edit). A no-op once there are some.
    pub fn materialize_agents(&mut self) {
        if self.agents.is_empty() {
            self.agents = materialize(self);
            if !self.agents.iter().any(|a| a.source == Some(AgentSource::ChatWorker)) {
                self.chat_worker = None;
            }
        }
    }

    /// Add or replace one agent (by id), after checking it. Nothing changes when it is refused.
    pub fn upsert_agent(&mut self, spec: AgentSpec) -> Result<(), String> {
        spec.validate()?;
        let mut next = self.clone();
        next.materialize_agents();
        match next.agents.iter_mut().find(|a| a.id == spec.id) {
            Some(a) => *a = spec,
            None => next.agents.push(spec),
        }
        next.check_agent_selection()?;
        *self = next;
        Ok(())
    }

    /// Replace the whole agent list and app default at once (the Settings page saves this way).
    pub fn set_agents(&mut self, agents: Vec<AgentSpec>, selection: AgentSelection) -> Result<(), String> {
        for a in &agents {
            a.validate()?;
        }
        let mut next = self.clone();
        next.materialize_agents();
        next.agents = agents;
        next.agent_selection = selection;
        if next.agents.is_empty() {
            return Err("keep at least one agent".into());
        }
        next.check_agent_selection()?;
        *self = next;
        Ok(())
    }

    /// Remove one agent. The app default may not keep naming it.
    pub fn remove_agent(&mut self, id: &str) -> Result<(), String> {
        self.materialize_agents();
        let before = self.agents.len();
        self.agents.retain(|a| a.id != id);
        if self.agents.len() == before {
            return Err(format!("no agent '{id}'"));
        }
        if self.agent_selection.orchestrator.as_deref() == Some(id) {
            self.agent_selection.orchestrator = None;
        }
        if let Some(w) = &mut self.agent_selection.workers {
            w.retain(|x| x != id);
            if w.is_empty() {
                self.agent_selection.workers = None;
            }
        }
        if let Some(c) = &mut self.agent_selection.concurrency {
            c.remove(id);
            if c.is_empty() {
                self.agent_selection.concurrency = None;
            }
        }
        Ok(())
    }

    fn check_agent_selection(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for a in &self.agents {
            if !seen.insert(a.id.as_str()) {
                return Err(format!("two agents are called '{}'", a.id));
            }
        }
        validate_selection(&self.agents(), &self.agent_selection).map_err(|e| e.to_string())
    }

    /// `agents.<id>.<field> = value` (creating the agent when it is new) and
    /// `agent_selection.<field> = value`, for `ghostreel config set` and MCP `set_settings`.
    pub(crate) fn set_agent_key(&mut self, section: &str, rest: &[&str], value: &str) -> Result<(), String> {
        if section == "agent_selection" {
            let field = rest.first().copied().unwrap_or_default();
            let none = value.trim().is_empty() || value == "default";
            match field {
                "orchestrator" => self.agent_selection.orchestrator = (!none).then(|| value.trim().to_string()),
                "pick_by_score" => {
                    self.agent_selection.pick_by_score =
                        matches!(value.trim().to_lowercase().as_str(), "true" | "on" | "yes" | "1").then_some(true)
                }
                "workers" => {
                    self.agent_selection.workers = (!none).then(|| {
                        value.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
                    })
                }
                "max_parallel" => {
                    self.agent_selection.max_parallel = if none {
                        None
                    } else {
                        Some(value.parse().map_err(|_| format!("agent_selection.max_parallel must be a number, got '{value}'"))?)
                    }
                }
                other => return Err(format!("unknown config key 'agent_selection.{other}'")),
            }
            return self.check_agent_selection();
        }
        let (Some(id), Some(field)) = (rest.first(), rest.get(1)) else {
            return Err("agent keys look like agents.<id>.<field>".into());
        };
        self.materialize_agents();
        let existing = self.agents().into_iter().find(|a| a.id == *id);
        let is_new = existing.is_none();
        let mut spec = existing.unwrap_or_else(|| AgentSpec { id: id.to_string(), ..AgentSpec::default() });
        // A new agent's first field says what it is: a url is a server, a local model is local.
        if is_new {
            match *field {
                "url" => spec.backend = Backend::Server,
                "local_model" => spec.backend = Backend::Local,
                _ => {}
            }
        }
        // A linked agent's model fields belong to its section: setting one here makes the agent
        // its own (its fields already hold the section's values, so only this one changes).
        if LINKED_FIELDS.contains(field) && spec.source.is_some() {
            let role = spec.roles.iter().next().copied().unwrap_or(AgentRole::Chat);
            spec.api_key = spec.llm_config(self, role).api_key;
            spec.source = None;
        }
        let num = |v: &str| v.trim().parse::<u64>().map_err(|_| format!("agents.{id}.{field} must be a number, got '{v}'"));
        match *field {
            "name" => spec.name = value.to_string(),
            "backend" => {
                spec.backend = match value.trim().to_lowercase().as_str() {
                    "local" => Backend::Local,
                    "server" => Backend::Server,
                    "cli" => Backend::Cli,
                    "auto" => Backend::Auto,
                    other => return Err(format!("invalid backend '{other}'; expected local, server or cli")),
                };
                spec.source = None;
            }
            "tool" => spec.tool = value.trim().to_string(),
            "model" => {
                spec.model = value.trim().to_string();
                spec.normalize_effort();
            }
            "effort" => spec.effort = Effort::parse_opt(value)?,
            "url" => spec.url = value.trim().to_string(),
            "api_key" => spec.api_key = value.to_string(),
            "local_model" => spec.local_model = value.trim().to_string(),
            "ctx_tokens" => spec.ctx_tokens = num(value)? as u32,
            "timeout_secs" => spec.timeout_secs = num(value)?,
            "max_concurrent" | "concurrency" => spec.max_concurrent = num(value)? as usize,
            "priority" => spec.priority = num(value)? as u32,
            "enabled" => spec.enabled = matches!(value.trim().to_lowercase().as_str(), "true" | "on" | "yes" | "1"),
            "roles" => {
                spec.roles = value
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(AgentRole::parse)
                    .collect::<Result<_, _>>()?;
            }
            other => {
                return Err(format!(
                    "unknown agent field '{other}'; expected name, backend, tool, model, effort, url, api_key, local_model, \
                     ctx_tokens, timeout_secs, max_concurrent, priority, enabled, roles"
                ));
            }
        }
        self.upsert_agent(spec)
    }
}

#[cfg(test)]
mod tests;
