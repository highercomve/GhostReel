//! Which agents a piece of work uses: chosen per run, per cut and app-wide, resolved in one place.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::spec::{AgentRole, AgentSpec, MAX_CONCURRENT};
use crate::config::Backend;

/// A choice of agents. Every field is optional: what is left out falls through to the next level
/// (run → cut → app → every enabled agent with the role).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSelection {
    /// The lead agent: chooses among the workers' answers, and writes the cut in script chat
    /// when it can chat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestrator: Option<String>,
    /// No orchestrator model: code picks from the workers' scores (fastest). Set at a level, it
    /// wins over that level's `orchestrator` and stops the lower levels' choice of one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick_by_score: Option<bool>,
    /// Agents that take parallel worker jobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workers: Option<Vec<String>>,
    /// Cap on worker jobs in flight across all workers (default: their summed capacity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallel: Option<usize>,
    /// Calls at once per worker for this choice, by agent id, instead of the agent's own
    /// `max_concurrent` (a local model still takes one).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<BTreeMap<String, usize>>,
}

impl AgentSelection {
    pub fn is_empty(&self) -> bool {
        self.orchestrator.is_none()
            && self.pick_by_score.is_none()
            && self.workers.is_none()
            && self.max_parallel.is_none()
            && self.concurrency.is_none()
    }

    /// The orchestrator this level chooses: `Some(None)` = none (pick by score).
    fn orchestrator_choice(&self) -> Option<Option<&String>> {
        if self.pick_by_score == Some(true) {
            Some(None)
        } else {
            self.orchestrator.as_ref().map(Some)
        }
    }
}

/// What the work is asked to make, for [`ResolvedSelection::summary`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkTarget {
    /// How many clips the run asks for.
    pub count: Option<usize>,
    /// Their frame ("9:16"); none or "original" says nothing.
    pub format: Option<crate::clips::ClipFormat>,
}

/// Which level a resolved field came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionSource {
    Run,
    Cut,
    App,
    /// Nothing chose: every enabled agent with the role, by priority.
    Default,
}

/// Why a selection cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    UnknownAgent { id: String, known: Vec<String> },
    Disabled { id: String },
    MissingRole { id: String, role: AgentRole },
    NoAgent { role: AgentRole },
    ZeroParallel,
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionError::UnknownAgent { id, known } => {
                write!(f, "unknown agent '{id}' (configured: {})", if known.is_empty() { "none".into() } else { known.join(", ") })
            }
            SelectionError::Disabled { id } => write!(f, "agent '{id}' is disabled; enable it in Models > Agents"),
            SelectionError::MissingRole { id, role } => {
                write!(f, "agent '{id}' cannot be used as {}: give it the {role} role in Models > Agents", article(*role))
            }
            SelectionError::NoAgent { role } => write!(f, "no enabled agent has the {role} role"),
            SelectionError::ZeroParallel => write!(f, "max_parallel and calls at once must be at least 1"),
        }
    }
}

fn article(role: AgentRole) -> String {
    match role {
        AgentRole::Orchestrator => "an orchestrator".into(),
        r => format!("a {r}"),
    }
}

impl std::error::Error for SelectionError {}

impl From<SelectionError> for crate::Error {
    fn from(e: SelectionError) -> Self {
        crate::Error::Invalid(e.to_string())
    }
}

/// A selection with every level applied and every agent checked.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedSelection {
    /// `None` when the choice is to pick by score, or no enabled agent can orchestrate (code
    /// then picks).
    pub orchestrator: Option<AgentSpec>,
    pub orchestrator_from: SelectionSource,
    pub workers: Vec<AgentSpec>,
    pub workers_from: SelectionSource,
    /// Worker jobs in flight at most: the requested cap, never above the workers' capacity.
    pub max_parallel: usize,
    /// The workers' summed capacity (a local model counts one).
    pub capacity: usize,
    /// Who answers script chat: the orchestrator when it can chat, else the first chat agent.
    pub chat: Option<AgentSpec>,
}

impl ResolvedSelection {
    /// The selection as it will run, to save on what the run made.
    pub fn effective(&self) -> AgentSelection {
        AgentSelection {
            orchestrator: self.orchestrator.as_ref().map(|a| a.id.clone()),
            pick_by_score: self.picks_by_score().then_some(true),
            workers: Some(self.workers.iter().map(|a| a.id.clone()).collect()),
            max_parallel: Some(self.max_parallel),
            concurrency: Some(self.concurrency()),
        }
    }

    /// Chosen to run without an orchestrator model (not merely left without one).
    pub fn picks_by_score(&self) -> bool {
        self.orchestrator.is_none() && self.orchestrator_from != SelectionSource::Default
    }

    /// Calls at once each worker takes, by id.
    pub fn concurrency(&self) -> BTreeMap<String, usize> {
        self.workers.iter().map(|a| (a.id.clone(), a.capacity())).collect()
    }

    /// "up to 7 at once: claude ×2 · codex ×2 · ghostpen ×2 · local ×1".
    pub fn capacity_line(&self) -> String {
        let parts: Vec<String> = self.workers.iter().map(|a| format!("{} ×{}", a.id, a.capacity())).collect();
        format!("up to {} at once: {}", self.max_parallel, parts.join(" · "))
    }

    /// Who does what, in one plain sentence: "GhostPen (2 at a time) and Claude (1) score the
    /// episode; Claude picks the 2 clips (9:16)."
    pub fn summary(&self, target: &WorkTarget) -> String {
        let workers: Vec<String> = self
            .workers
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let n = a.capacity();
                if i == 0 { format!("{} ({n} at a time)", a.display_name()) } else { format!("{} ({n})", a.display_name()) }
            })
            .collect();
        let scorers = match workers.as_slice() {
            [] => "No agent".to_string(),
            [one] => one.clone(),
            [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
        };
        let verb = if workers.len() == 1 { "scores" } else { "score" };
        let mut line = format!("{scorers} {verb} the episode");
        if self.max_parallel < self.capacity {
            line.push_str(&format!(", at most {} at once", self.max_parallel));
        }
        let clips = match target.count {
            Some(1) => "the clip".to_string(),
            Some(n) => format!("the {n} clips"),
            None => "the clips".to_string(),
        };
        let frame = match target.format {
            Some(f) if f != crate::clips::ClipFormat::Original => format!(" ({})", f.label()),
            _ => String::new(),
        };
        match &self.orchestrator {
            Some(o) => line.push_str(&format!("; {} picks {clips}{frame}.", o.display_name())),
            None => {
                let top = match target.count {
                    Some(1) => "the best score becomes the clip".to_string(),
                    Some(n) => format!("the top {n} by score become the clips"),
                    None => "the best scores become the clips".to_string(),
                };
                line.push_str(&format!("; no model picks: {top}{frame}."));
            }
        }
        line
    }
}

fn find<'a>(agents: &'a [AgentSpec], id: &str, role: AgentRole) -> Result<&'a AgentSpec, SelectionError> {
    let a = agents.iter().find(|a| a.id == id).ok_or_else(|| SelectionError::UnknownAgent {
        id: id.to_string(),
        known: agents.iter().map(|a| a.id.clone()).collect(),
    })?;
    if !a.enabled {
        return Err(SelectionError::Disabled { id: id.to_string() });
    }
    if !a.has_role(role) {
        return Err(SelectionError::MissingRole { id: id.to_string(), role });
    }
    Ok(a)
}

/// Enabled agents with `role`, by priority (ties keep config order).
pub fn with_role(agents: &[AgentSpec], role: AgentRole) -> Vec<AgentSpec> {
    let mut v: Vec<AgentSpec> = agents.iter().filter(|a| a.enabled && a.has_role(role)).cloned().collect();
    v.sort_by_key(|a| a.priority);
    v
}

/// The first level that sets a field, and which level that was.
fn pick<'a, T>(
    levels: &[(Option<&'a AgentSelection>, SelectionSource)],
    field: impl Fn(&'a AgentSelection) -> Option<&'a T>,
) -> Option<(&'a T, SelectionSource)> {
    levels.iter().find_map(|(sel, src)| sel.and_then(&field).map(|v| (v, *src)))
}

/// The one resolver: a run's choice overrides the cut's, which overrides the app's; what nobody
/// chose is every enabled agent with the role. Every named agent must exist, be enabled and
/// have the role it is named for.
pub fn resolve_selection(
    agents: &[AgentSpec],
    run: Option<&AgentSelection>,
    cut: Option<&AgentSelection>,
    app: &AgentSelection,
) -> Result<ResolvedSelection, SelectionError> {
    let levels = [(run, SelectionSource::Run), (cut, SelectionSource::Cut), (Some(app), SelectionSource::App)];

    let orch = levels.iter().find_map(|(sel, src)| sel.and_then(|s| s.orchestrator_choice()).map(|c| (c, *src)));
    let (orchestrator, orchestrator_from) = match orch {
        Some((Some(id), src)) => (Some(find(agents, id, AgentRole::Orchestrator)?.clone()), src),
        Some((None, src)) => (None, src),
        None => (with_role(agents, AgentRole::Orchestrator).into_iter().next(), SelectionSource::Default),
    };

    let (workers, workers_from) = match pick(&levels, |s| s.workers.as_ref()) {
        Some((ids, src)) if !ids.is_empty() => {
            let mut out: Vec<AgentSpec> = Vec::new();
            for id in ids {
                let a = find(agents, id, AgentRole::Worker)?;
                if !out.iter().any(|o| o.id == a.id) {
                    out.push(a.clone());
                }
            }
            (out, src)
        }
        // An empty list says nothing: fall back to every worker.
        _ => (with_role(agents, AgentRole::Worker), SelectionSource::Default),
    };
    // Calls at once: the highest level that sets one for an agent wins; a local model takes one.
    let workers: Vec<AgentSpec> = workers
        .into_iter()
        .map(|mut a| {
            let set = levels.iter().find_map(|(sel, _)| sel.and_then(|s| s.concurrency.as_ref()?.get(&a.id).copied()));
            if let Some(n) = set {
                a.max_concurrent = if a.backend == Backend::Local { 1 } else { n.clamp(1, MAX_CONCURRENT) };
            }
            a
        })
        .collect();

    let capacity: usize = workers.iter().map(AgentSpec::capacity).sum();
    let max_parallel = match pick(&levels, |s| s.max_parallel.as_ref()) {
        Some((0, _)) => return Err(SelectionError::ZeroParallel),
        Some((n, _)) => (*n).min(capacity.max(1)),
        None => capacity.max(1),
    };

    let chat = match &orchestrator {
        Some(o) if o.has_role(AgentRole::Chat) => Some(o.clone()),
        _ => with_role(agents, AgentRole::Chat).into_iter().next(),
    };

    Ok(ResolvedSelection { orchestrator, orchestrator_from, workers, workers_from, max_parallel, capacity, chat })
}

/// Check a selection against the agents without resolving the rest (for saving it on a cut or
/// as the app default).
pub fn validate_selection(agents: &[AgentSpec], sel: &AgentSelection) -> Result<(), SelectionError> {
    if let Some(id) = &sel.orchestrator {
        find(agents, id, AgentRole::Orchestrator)?;
    }
    for id in sel.workers.iter().flatten() {
        find(agents, id, AgentRole::Worker)?;
    }
    if sel.max_parallel == Some(0) || sel.concurrency.iter().flatten().any(|(_, n)| *n == 0) {
        return Err(SelectionError::ZeroParallel);
    }
    Ok(())
}
