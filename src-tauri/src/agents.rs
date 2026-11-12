//! The agent pool as the window sees it: list, save, test, and resolve a selection. Every rule
//! lives in `ghostreel_core::agents`; these commands only carry its structures to the page.

use ghostreel_core::agents::{
    self, AgentRole, AgentSelection, AgentSpec, AgentStatus, Availability, ResolvedSelection, SelectionSource,
    WorkTarget,
};
use std::collections::BTreeMap;
use ghostreel_core::config::{Backend, Config};
use serde::Serialize;

use crate::{CmdResult, err, paths};

/// One agent: what is stored (to edit) and what it runs (to show).
#[derive(Serialize)]
pub struct AgentView {
    /// As stored; `api_key` is never sent, `has_api_key` says whether one is set.
    pub spec: AgentSpec,
    /// Model fields filled from the linked section, for linked agents.
    pub effective: AgentSpec,
    pub has_api_key: bool,
    /// Calls at once, with a local model held to one.
    pub capacity: usize,
    /// Why "at once" cannot be changed, when it cannot.
    pub capacity_locked: Option<String>,
    /// For linked agents: the Settings section that sets its model.
    pub linked_to: Option<String>,
    /// "cli claude · claude-opus-5-5".
    pub summary: String,
    /// The name to show: its own name, or its id when it has none.
    pub display_name: String,
}

/// A resolved selection, with every agent named by id.
#[derive(Serialize)]
pub struct ResolvedView {
    pub orchestrator: Option<String>,
    pub orchestrator_from: SelectionSource,
    pub workers: Vec<String>,
    pub workers_from: SelectionSource,
    pub chat: Option<String>,
    pub max_parallel: usize,
    pub capacity: usize,
    /// "up to 6 at once: claude ×2 · codex ×2 · ghostpen ×2".
    pub capacity_line: String,
    /// Chosen to run without an orchestrator model (code picks by score).
    pub picks_by_score: bool,
    /// Calls at once each worker takes, by id.
    pub concurrency: BTreeMap<String, usize>,
    /// "GhostPen (2 at a time) and Claude (1) score the episode; Claude picks the 2 clips."
    pub summary: String,
}

impl ResolvedView {
    fn new(r: &ResolvedSelection, target: &WorkTarget) -> Self {
        Self {
            orchestrator: r.orchestrator.as_ref().map(|a| a.id.clone()),
            orchestrator_from: r.orchestrator_from,
            workers: r.workers.iter().map(|a| a.id.clone()).collect(),
            workers_from: r.workers_from,
            chat: r.chat.as_ref().map(|a| a.id.clone()),
            max_parallel: r.max_parallel,
            capacity: r.capacity,
            capacity_line: r.capacity_line(),
            picks_by_score: r.picks_by_score(),
            concurrency: r.concurrency(),
            summary: r.summary(target),
        }
    }
}

#[derive(Serialize)]
pub struct AgentsView {
    /// No `[[agents]]` yet: these stand for the older sections; the first save writes them out.
    pub implicit: bool,
    pub agents: Vec<AgentView>,
    pub app_selection: AgentSelection,
    /// The app selection resolved (`None` when it is invalid, see `error`).
    pub resolved: Option<ResolvedView>,
    pub error: Option<String>,
    pub roles: Vec<AgentRole>,
    pub cli_tools: Vec<String>,
}

fn view(config: &Config) -> AgentsView {
    let agents = config
        .agents()
        .into_iter()
        .map(|a| {
            let effective = AgentSpec { api_key: String::new(), ..a.effective(config) };
            let local = effective.backend == Backend::Local;
            AgentView {
                has_api_key: !a.api_key.is_empty(),
                capacity: if local { 1 } else { a.capacity() },
                capacity_locked: local.then(|| "A local model runs one chat at a time.".into()),
                linked_to: a.source.map(|s| s.label().to_string()),
                summary: effective.summary(),
                display_name: a.display_name().to_string(),
                effective,
                spec: AgentSpec { api_key: String::new(), ..a },
            }
        })
        .collect();
    let resolved = config.resolve_agents(None, None);
    AgentsView {
        implicit: config.agents.is_empty(),
        agents,
        app_selection: config.agent_selection.clone(),
        resolved: resolved.as_ref().ok().map(|r| ResolvedView::new(r, &WorkTarget::default())),
        error: resolved.err().map(|e| e.to_string()),
        roles: AgentRole::ALL.to_vec(),
        cli_tools: ghostreel_core::config::CLI_TOOLS.iter().map(|t| t.to_string()).collect(),
    }
}

#[tauri::command]
pub fn list_agents() -> CmdResult<AgentsView> {
    let config = Config::load(&paths()?.config_file).map_err(err)?;
    Ok(view(&config))
}

/// Save the whole list and the app-wide choice. A key left empty keeps the one stored for that id.
#[tauri::command]
pub fn save_agents(agents: Vec<AgentSpec>, selection: AgentSelection) -> CmdResult<AgentsView> {
    let p = paths()?;
    let mut config = Config::load(&p.config_file).map_err(err)?;
    let stored = config.agents();
    let agents = agents
        .into_iter()
        .map(|mut a| {
            if a.api_key.is_empty()
                && let Some(old) = stored.iter().find(|o| o.id == a.id)
            {
                a.api_key = old.api_key.clone();
            }
            a
        })
        .collect();
    config.set_agents(agents, selection)?;
    config.save(&p.config_file).map_err(err)?;
    Ok(view(&config))
}

/// Whether one agent can run now (installed / reachable / model listed). No model call.
#[tauri::command]
pub async fn test_agent(id: String) -> CmdResult<Availability> {
    let p = paths()?;
    let config = Config::load(&p.config_file).map_err(err)?;
    let spec = config.agents().into_iter().find(|a| a.id == id).ok_or_else(|| format!("no agent '{id}'"))?;
    Ok(agents::check_availability(&p, &config, &spec).await)
}

/// What a run with `run` on a cut with `cut` would use, and the sentence under a picker
/// (naming the clip count and format of `target` when given).
#[tauri::command]
pub fn resolve_agents(
    run: Option<AgentSelection>,
    cut: Option<AgentSelection>,
    target: Option<WorkTarget>,
) -> CmdResult<ResolvedView> {
    let config = Config::load(&paths()?.config_file).map_err(err)?;
    let r = config.resolve_agents(run.as_ref(), cut.as_ref()).map_err(err)?;
    Ok(ResolvedView::new(&r, &target.unwrap_or_default()))
}

/// Where every agent stands now (ready, busy, out of quota, missing…), for the dots in a picker.
/// Makes no model call.
#[tauri::command]
pub async fn agent_status() -> CmdResult<Vec<AgentStatus>> {
    let p = paths()?;
    let config = Config::load(&p.config_file).map_err(err)?;
    let agents = config.agents();
    Ok(futures_util::future::join_all(agents.iter().map(|a| agents::agent_status(&p, &config, a))).await)
}

/// A cut's or run's choice, checked against the configured agents before it is saved or queued.
pub fn check_selection(sel: Option<&AgentSelection>) -> CmdResult<()> {
    let Some(sel) = sel else { return Ok(()) };
    let config = Config::load(&paths()?.config_file).map_err(err)?;
    agents::validate_selection(&config.agents(), sel).map_err(err)
}
