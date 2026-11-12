//! Spreading parallel jobs over several agents.
//!
//! A pool is the agents chosen for one role in one run, each resolved to an endpoint. Jobs go to
//! whichever agent has a free slot — its own `max_concurrent` and the endpoint's shared slots,
//! which other features may be using at the same moment — by priority, then least loaded.
//!
//! A failed job is handled by what kind of failure it was ([`AgentErrorKind::handling`]): a busy
//! or rate-limited agent pauses and the job is asked again (there or elsewhere, a bounded number
//! of times); an agent whose quota is gone rests process-wide ([`Cooldowns`]) and its jobs go to
//! the others; an agent that cannot work as configured (unknown model, not signed in, not
//! installed) is dropped from the run. Anything else is tried once more, on a different agent when
//! there is one. When no agent is left the run stops at once with one error naming each agent and
//! why ([`AgentErrorKind::NoAgentLeft`]). Every pause, rest and drop is reported as a
//! [`PoolEvent`].

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use serde::Serialize;
use serde_json::Value;

use super::cooldown::Cooldowns;
use super::error::{AgentError, AgentErrorKind, Handling};
use super::slots::{EndpointKey, Slot, SlotRegistry};
use super::spec::{AgentRole, AgentSpec};
use crate::Error;
use crate::chat::ChatBackend;
use crate::config::{CliAgentConfig, Config};
use crate::runtime::VisionSetup;

/// One structured-JSON job: a prompt and the shape of the answer.
#[derive(Debug, Clone)]
pub struct AgentJob {
    /// Caller's tag, echoed in the result ("v1 0–600 s").
    pub label: String,
    pub prompt: String,
    pub schema: Value,
    /// Output token cap (servers and the local helper).
    pub max_tokens: usize,
    /// The tool registry, when the agent may use it before it answers. `None`: one request.
    pub tools: Option<Arc<crate::tools::ToolAccess>>,
}

/// How a resolved agent is reached.
#[derive(Clone)]
pub enum AgentTarget {
    Server { url: String, model: String, api_key: String, timeout_s: u64 },
    Cli(CliAgentConfig),
    /// Local / MLX helper: started on the agent's first job, stopped after its last.
    Local(VisionSetup),
    /// Run by the caller's own [`JobRunner`] (tests, or a backend this module does not know).
    External,
}

/// An agent of a pool, resolved.
pub struct PoolAgent {
    pub spec: AgentSpec,
    pub key: EndpointKey,
    /// Jobs it takes at once (the endpoint's shared slots can make that fewer).
    pub capacity: usize,
    pub target: AgentTarget,
    /// What it resolved to, for the log ("cli claude · claude-opus-5-5").
    pub describe: String,
    /// The agent's optional cap on rounds of tool calls per job (0 = none, the default).
    pub max_tool_rounds: u32,
}

impl PoolAgent {
    pub fn from_setup(spec: AgentSpec, setup: &VisionSetup, timeout_s: u64) -> Result<Self, String> {
        let key = EndpointKey::for_setup(setup).ok_or_else(|| match setup {
            VisionSetup::Unavailable(why) => why.clone(),
            _ => "unavailable".to_string(),
        })?;
        let target = match setup {
            VisionSetup::Server(s) => AgentTarget::Server {
                url: s.url.trim_end_matches('/').to_string(),
                model: s.model.clone(),
                api_key: s.api_key.clone(),
                timeout_s,
            },
            VisionSetup::Cli(cfg) => AgentTarget::Cli(CliAgentConfig {
                timeout_secs: cfg.timeout_secs.max(timeout_s.min(1800)),
                ..cfg.clone()
            }),
            other => AgentTarget::Local(other.clone()),
        };
        Ok(Self { capacity: key.capacity(spec.max_concurrent), describe: setup.describe(), key, target, spec, max_tool_rounds: 0 })
    }
}

/// One try of a job.
#[derive(Debug, Clone, Serialize)]
pub struct Attempt {
    pub agent: String,
    pub seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentError>,
}

/// A job's outcome, in job order.
#[derive(Debug, Clone, Serialize)]
pub struct JobOutcome {
    pub index: usize,
    pub label: String,
    #[serde(skip)]
    pub value: Result<Value, AgentError>,
    /// The agent of the last try.
    pub agent: String,
    /// Seconds over every try.
    pub seconds: f64,
    pub attempts: Vec<Attempt>,
}

/// One agent's share of a run so far.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgentLoad {
    pub id: String,
    pub capacity: usize,
    pub running: usize,
    /// Jobs it answered.
    pub done: usize,
    /// Tries that failed on it (a retried job counts on the agent that failed it).
    pub failed: usize,
    /// Seconds spent on its answered jobs.
    pub busy_s: f64,
}

/// What the pool did about a failure, beyond trying the job again.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PoolAction {
    /// Busy or rate limited: the agent pauses and the job is asked again.
    Retrying { job: String, after_s: f64 },
    /// Quota gone: the agent rests (for every feature) until `until`, Unix seconds.
    CoolingDown { until: i64 },
    /// It cannot work as configured: no more jobs this run.
    Dropped,
}

/// A failure the pool acted on, for the run's log.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PoolEvent {
    pub agent: String,
    #[serde(flatten)]
    pub action: PoolAction,
    pub error: AgentError,
    /// The agents still taking jobs.
    pub remaining: Vec<String>,
}

impl PoolEvent {
    /// One line: "agy: quota used up, resets in 2 h 3 min — resting it; its jobs go to claude".
    pub fn message(&self) -> String {
        let to = if self.remaining.is_empty() {
            "no other agent left".to_string()
        } else {
            format!("its jobs go to {}", self.remaining.join(", "))
        };
        match &self.action {
            PoolAction::Retrying { job, after_s } => {
                format!("{} is busy — asking again for {job} in {after_s:.0} s", self.agent)
            }
            PoolAction::CoolingDown { .. } => format!("{}: {} — resting it; {to}", self.agent, self.error.kind.label()),
            PoolAction::Dropped => format!("{}: {} — dropped from this run; {to}", self.agent, self.error.kind.label()),
        }
    }
}

/// Where a run is, after a job started or ended, or the pool acted on a failure.
pub struct PoolProgress<'a> {
    pub done: usize,
    pub total: usize,
    /// The job that just ended (`None` when one just started).
    pub finished: Option<&'a JobOutcome>,
    pub agents: &'a [AgentLoad],
    /// A pause, rest or drop that just happened.
    pub event: Option<&'a PoolEvent>,
}

impl PoolProgress<'_> {
    /// "claude 3 · codex 2 · local 1 running", or "" when nothing runs.
    pub fn running_line(&self) -> String {
        running_line(self.agents)
    }
}

pub fn running_line(agents: &[AgentLoad]) -> String {
    let parts: Vec<String> =
        agents.iter().filter(|a| a.running > 0).map(|a| format!("{} {}", a.id, a.running)).collect();
    if parts.is_empty() { String::new() } else { format!("{} running", parts.join(" · ")) }
}

/// "claude 4 · ghostpen 3 (1 failed)": who answered how many, for the end of a run.
pub fn share_line(agents: &[AgentLoad]) -> String {
    agents
        .iter()
        .filter(|a| a.done + a.failed > 0)
        .map(|a| {
            if a.failed > 0 { format!("{} {} ({} failed)", a.id, a.done, a.failed) } else { format!("{} {}", a.id, a.done) }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Runs one job on one agent. The model runner is the real one; tests bring fakes.
pub trait JobRunner: Send + Sync {
    fn run(&self, agent: Arc<PoolAgent>, job: Arc<AgentJob>) -> BoxFuture<'static, Result<Value, AgentError>>;
    /// The agent gets nothing more from this run (stop a local helper).
    fn release(&self, _agent: &PoolAgent) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

/// Runs jobs on real models: an HTTP request per job, a fresh CLI session per job, or the local
/// helper (started once, kept for the agent's following jobs).
#[derive(Default)]
pub struct ModelRunner {
    local: Mutex<HashMap<String, Arc<tokio::sync::Mutex<Option<ChatBackend>>>>>,
}

impl ModelRunner {
    fn local_cell(&self, id: &str) -> Arc<tokio::sync::Mutex<Option<ChatBackend>>> {
        self.local.lock().unwrap_or_else(|e| e.into_inner()).entry(id.to_string()).or_default().clone()
    }
}

impl JobRunner for ModelRunner {
    fn run(&self, agent: Arc<PoolAgent>, job: Arc<AgentJob>) -> BoxFuture<'static, Result<Value, AgentError>> {
        use crate::tools::run_agent_json;
        let id = agent.spec.id.clone();
        // A failure the backend did not type is a failed call; an answer that does not parse is
        // an unusable one (the loop types those itself).
        let failed = move |id: &str, e: crate::Error| AgentError::from_error(id, &e, AgentErrorKind::Other);
        let tools = job.tools.as_ref().map(|t| Arc::new(t.with_max_rounds(agent.max_tool_rounds)));
        match agent.target.clone() {
            AgentTarget::Server { url, model, api_key, timeout_s } => Box::pin(async move {
                // The window is only used to trim a long conversation; a server job is short.
                let mut backend = ChatBackend::Server { url, model, api_key, ctx_tokens: 32_768 };
                run_agent_json(&mut backend, &job.prompt, &job.schema, job.max_tokens, timeout_s, tools.as_deref())
                    .await
                    .map_err(|e| failed(&id, e))
            }),
            AgentTarget::Cli(cfg) => Box::pin(async move {
                // A fresh agent per job: its own session, nothing carried over from another job.
                let mut backend = ChatBackend::Cli(crate::cliagent::CliAgent::new(cfg));
                run_agent_json(&mut backend, &job.prompt, &job.schema, job.max_tokens, 0, tools.as_deref())
                    .await
                    .map_err(|e| failed(&id, e))
            }),
            AgentTarget::Local(setup) => {
                let cell = self.local_cell(&agent.spec.id);
                Box::pin(async move {
                    let mut guard = cell.lock().await;
                    if guard.is_none() {
                        *guard = Some(ChatBackend::from_vision_setup(&setup, 600).await.map_err(|e| failed(&id, e))?);
                    }
                    let backend = guard.as_mut().expect("started above");
                    run_agent_json(backend, &job.prompt, &job.schema, job.max_tokens, 600, tools.as_deref())
                        .await
                        .map_err(|e| failed(&id, e))
                })
            }
            AgentTarget::External => {
                Box::pin(async move { Err(AgentError::new(id, AgentErrorKind::Other, "this agent has no model runner")) })
            }
        }
    }

    fn release(&self, agent: &PoolAgent) -> BoxFuture<'static, ()> {
        let cell = self.local.lock().unwrap_or_else(|e| e.into_inner()).remove(&agent.spec.id);
        Box::pin(async move {
            let Some(cell) = cell else { return };
            if let Some(ChatBackend::Local { mut helper, .. }) = cell.lock().await.take() {
                helper.kill().await;
            }
        })
    }
}

/// An agent that could not be used this run, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Skipped {
    pub id: String,
    pub reason: String,
}

/// The agents of one role for one run.
pub struct AgentPool {
    agents: Vec<Arc<PoolAgent>>,
    role: AgentRole,
    max_parallel: usize,
    slots: Arc<SlotRegistry>,
    cooldowns: Arc<Cooldowns>,
    runner: Arc<dyn JobRunner>,
    /// Selected agents that could not be resolved (not installed, server down, resting…).
    pub skipped: Vec<Skipped>,
}

/// Times a job is asked again because its agent was busy, before that counts as its failure.
const MAX_BUSY_RETRIES: u32 = 5;
/// The longest pause a busy agent gets before it is asked again.
const MAX_BUSY_WAIT: Duration = Duration::from_secs(60);

/// Whether an agent still takes jobs this run.
#[derive(Debug, Clone)]
enum Health {
    Ready,
    /// Busy: no new job before this.
    Paused(Instant),
    /// Resting or dropped, with the failure that did it.
    Out(AgentError),
}

impl AgentPool {
    /// Resolve `specs` for `role` on this machine. Agents that are unavailable are skipped (and
    /// listed); none available is an error naming why each was skipped.
    pub async fn resolve(
        paths: &crate::paths::Paths,
        config: &Config,
        role: AgentRole,
        specs: &[AgentSpec],
        max_parallel: usize,
        timeout_s: u64,
    ) -> Result<Self, Error> {
        let setups = futures_util::future::join_all(
            specs.iter().map(|s| crate::runtime::resolve_agent(paths, config, s, role)),
        )
        .await;
        let mut agents = Vec::new();
        let mut skipped = Vec::new();
        let cooldowns = Cooldowns::global();
        for (spec, setup) in specs.iter().zip(setups) {
            match PoolAgent::from_setup(spec.clone(), &setup, timeout_s) {
                Ok(mut a) => match cooldowns.get(&a.key) {
                    Some(c) => skipped.push(Skipped { id: spec.id.clone(), reason: c.reason() }),
                    None => {
                        a.max_tool_rounds = spec.llm_config(config, role).max_tool_rounds;
                        agents.push(a)
                    }
                },
                Err(reason) => skipped.push(Skipped { id: spec.id.clone(), reason }),
            }
        }
        if agents.is_empty() {
            let why = if skipped.is_empty() {
                format!("no enabled agent has the {role} role")
            } else {
                skipped.iter().map(|s| format!("{}: {}", s.id, s.reason)).collect::<Vec<_>>().join("; ")
            };
            return Err(Error::Invalid(format!("no {role} agent is available ({why})")));
        }
        let mut pool = Self::new(agents, max_parallel, SlotRegistry::global(), Arc::new(ModelRunner::default()));
        pool.skipped = skipped;
        pool.role = role;
        Ok(pool)
    }

    pub fn new(agents: Vec<PoolAgent>, max_parallel: usize, slots: Arc<SlotRegistry>, runner: Arc<dyn JobRunner>) -> Self {
        let cap: usize = agents.iter().map(|a| a.capacity).sum();
        Self {
            agents: agents.into_iter().map(Arc::new).collect(),
            role: AgentRole::Worker,
            max_parallel: max_parallel.clamp(1, cap.max(1)),
            slots,
            cooldowns: Cooldowns::global(),
            runner,
            skipped: Vec::new(),
        }
    }

    /// Record quota cooldowns here instead of the process's registry (tests).
    pub fn with_cooldowns(mut self, cooldowns: Arc<Cooldowns>) -> Self {
        self.cooldowns = cooldowns;
        self
    }

    pub fn with_role(mut self, role: AgentRole) -> Self {
        self.role = role;
        self
    }

    pub fn agents(&self) -> impl Iterator<Item = &PoolAgent> {
        self.agents.iter().map(|a| a.as_ref())
    }

    pub fn max_parallel(&self) -> usize {
        self.max_parallel
    }

    /// "claude ×2 · ghostpen ×2 (up to 4 at once)".
    pub fn describe(&self) -> String {
        let parts: Vec<String> = self.agents.iter().map(|a| format!("{} ×{}", a.spec.id, a.capacity)).collect();
        format!("{} (up to {} at once)", parts.join(" · "), self.max_parallel)
    }

    /// The best agent with room for a job that should avoid `avoid`, with its slot. Agents that
    /// are paused or out are passed over; `avoid` is only honoured while another agent is in.
    fn choose(
        &self,
        running: &[usize],
        health: &[Health],
        avoid: Option<usize>,
        held: &mut HashMap<usize, Slot>,
    ) -> Option<(usize, Slot)> {
        let now = Instant::now();
        let in_run = |i: usize| !matches!(health[i], Health::Out(_));
        let avoid = avoid.filter(|&a| (0..self.agents.len()).any(|i| i != a && in_run(i)));
        let mut order: Vec<usize> = (0..self.agents.len())
            .filter(|&i| {
                let ready = match &health[i] {
                    Health::Ready => true,
                    Health::Paused(until) => *until <= now,
                    Health::Out(_) => false,
                };
                ready && running[i] < self.agents[i].capacity && Some(i) != avoid
            })
            .collect();
        order.sort_by(|&a, &b| {
            let (sa, sb) = (&self.agents[a], &self.agents[b]);
            sa.spec.priority.cmp(&sb.spec.priority).then_with(|| {
                let la = running[a] as f64 / sa.capacity as f64;
                let lb = running[b] as f64 / sb.capacity as f64;
                la.total_cmp(&lb)
            })
        });
        order.into_iter().find_map(|i| {
            let slot = held.remove(&i).or_else(|| self.slots.try_acquire(&self.agents[i].key, self.agents[i].capacity))?;
            Some((i, slot))
        })
    }

    /// Run every job, at most `max_parallel` at once, and return the outcomes in job order.
    /// Cancelling stops new jobs and drops those in flight. When every agent is out (resting or
    /// dropped) the run stops with [`AgentErrorKind::NoAgentLeft`].
    pub async fn run_jobs(
        &self,
        jobs: Vec<AgentJob>,
        cancel: Option<Arc<AtomicBool>>,
        on_progress: &mut (dyn FnMut(&PoolProgress) + Send),
    ) -> Result<Vec<JobOutcome>, Error> {
        struct Pending {
            index: usize,
            job: Arc<AgentJob>,
            avoid: Option<usize>,
            attempts: Vec<Attempt>,
            /// Failures that count toward its one retry (not busy pauses, not dropped agents).
            failures: u32,
            busy: u32,
        }
        type Done = (usize, usize, Slot, Result<Value, AgentError>, f64);

        let total = jobs.len();
        let is_cancelled = || cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst));
        let mut pending: VecDeque<Pending> = jobs
            .into_iter()
            .enumerate()
            .map(|(index, job)| Pending { index, job: Arc::new(job), avoid: None, attempts: Vec::new(), failures: 0, busy: 0 })
            .collect();
        let mut loads: Vec<AgentLoad> = self
            .agents
            .iter()
            .map(|a| AgentLoad { id: a.spec.id.clone(), capacity: a.capacity, running: 0, done: 0, failed: 0, busy_s: 0.0 })
            .collect();
        let mut running = vec![0usize; self.agents.len()];
        let mut health = vec![Health::Ready; self.agents.len()];
        let mut in_flight: HashMap<usize, Pending> = HashMap::new();
        // A local agent keeps its slot between jobs: its helper holds the model in memory, and a
        // gap would let another feature load a second model beside it.
        let mut held: HashMap<usize, Slot> = HashMap::new();
        let mut used = vec![false; self.agents.len()];
        let mut outcomes: Vec<JobOutcome> = Vec::with_capacity(total);
        let mut set: tokio::task::JoinSet<Done> = tokio::task::JoinSet::new();

        let result = loop {
            if is_cancelled() {
                set.abort_all();
                break Err(Error::Invalid(crate::chat::CANCELLED.into()));
            }
            let mut started = false;
            while set.len() < self.max_parallel {
                let Some(front) = pending.front() else { break };
                let Some((ai, slot)) = self.choose(&running, &health, front.avoid, &mut held) else { break };
                let p = pending.pop_front().expect("front exists");
                running[ai] += 1;
                loads[ai].running += 1;
                used[ai] = true;
                let (agent, job, runner, index) = (self.agents[ai].clone(), p.job.clone(), self.runner.clone(), p.index);
                in_flight.insert(index, p);
                set.spawn(async move {
                    let t0 = Instant::now();
                    let value = runner.run(agent, job).await;
                    (index, ai, slot, value, t0.elapsed().as_secs_f64())
                });
                started = true;
            }
            if started {
                on_progress(&PoolProgress { done: outcomes.len(), total, finished: None, agents: &loads, event: None });
            }
            if set.is_empty() && pending.is_empty() {
                break Ok(());
            }
            // Wake now and then: to notice a cancel, a paused agent's turn, and a slot another
            // feature gave back.
            let joined = match tokio::time::timeout(Duration::from_millis(250), set.join_next()).await {
                Err(_) | Ok(None) => continue,
                Ok(Some(j)) => j,
            };
            let (index, ai, slot, value, seconds) = match joined {
                Ok(d) => d,
                Err(e) => {
                    set.abort_all();
                    break Err(Error::Invalid(format!("agent job failed: {e}")));
                }
            };
            running[ai] -= 1;
            loads[ai].running -= 1;
            let mut p = in_flight.remove(&index).expect("job was in flight");
            let agent_id = self.agents[ai].spec.id.clone();
            let value = value.map_err(|e| e.for_agent(&agent_id));
            p.attempts.push(Attempt { agent: agent_id.clone(), seconds, error: value.as_ref().err().cloned() });
            // What to do with the job: `None` = it is finished (answered or failed for good).
            let requeue = match &value {
                Ok(_) => None,
                Err(e) => {
                    loads[ai].failed += 1;
                    let event = match e.kind.handling() {
                        Handling::Retry { after } if p.busy < MAX_BUSY_RETRIES => {
                            p.busy += 1;
                            let wait = after
                                .unwrap_or(Duration::from_secs(1 << p.busy.min(5)))
                                .clamp(Duration::from_millis(100), MAX_BUSY_WAIT);
                            if !matches!(health[ai], Health::Out(_)) {
                                health[ai] = Health::Paused(Instant::now() + wait);
                            }
                            Some(PoolAction::Retrying { job: p.job.label.clone(), after_s: wait.as_secs_f64() })
                        }
                        Handling::Cooldown { for_ } => {
                            let c = self.cooldowns.put(&self.agents[ai].key, for_, e.clone());
                            let fresh = !matches!(health[ai], Health::Out(_));
                            health[ai] = Health::Out(e.clone());
                            fresh.then_some(PoolAction::CoolingDown { until: c.until })
                        }
                        Handling::Drop => {
                            let fresh = !matches!(health[ai], Health::Out(_));
                            health[ai] = Health::Out(e.clone());
                            fresh.then_some(PoolAction::Dropped)
                        }
                        Handling::Retry { .. } | Handling::RetryElsewhere => {
                            p.failures += 1;
                            None
                        }
                    };
                    if let Some(action) = event {
                        let remaining = (0..self.agents.len())
                            .filter(|&i| !matches!(health[i], Health::Out(_)))
                            .map(|i| self.agents[i].spec.id.clone())
                            .collect();
                        let ev = PoolEvent { agent: agent_id.clone(), action, error: e.clone(), remaining };
                        on_progress(&PoolProgress {
                            done: outcomes.len(),
                            total,
                            finished: None,
                            agents: &loads,
                            event: Some(&ev),
                        });
                    }
                    match e.kind.handling() {
                        // Busy within its retries, or its agent is out: the job goes back as is.
                        Handling::Retry { .. } if p.failures == 0 => Some(false),
                        Handling::Cooldown { .. } | Handling::Drop => Some(false),
                        // Its first real failure: once more, elsewhere when there is an elsewhere.
                        _ if p.failures == 1 => Some(true),
                        _ => None,
                    }
                }
            };
            match requeue {
                Some(elsewhere) => {
                    if elsewhere {
                        p.avoid = (self.agents.len() > 1).then_some(ai);
                    }
                    pending.push_front(p);
                }
                None => {
                    if value.is_ok() {
                        loads[ai].done += 1;
                        loads[ai].busy_s += seconds;
                    }
                    let seconds = p.attempts.iter().map(|a| a.seconds).sum();
                    outcomes.push(JobOutcome {
                        index: p.index,
                        label: p.job.label.clone(),
                        value,
                        agent: agent_id,
                        seconds,
                        attempts: p.attempts,
                    });
                    let last = outcomes.last().expect("just pushed");
                    on_progress(&PoolProgress { done: outcomes.len(), total, finished: Some(last), agents: &loads, event: None });
                }
            }
            let out = matches!(health[ai], Health::Out(_));
            if self.agents[ai].key.is_local() && !pending.is_empty() && !out {
                held.insert(ai, slot);
            } else {
                drop(slot);
                if self.agents[ai].key.is_local() {
                    self.runner.release(&self.agents[ai]).await;
                    used[ai] = false;
                }
            }
            // Nobody left to take what is still waiting: stop now, with every agent's reason.
            if health.iter().all(|h| matches!(h, Health::Out(_))) && (!pending.is_empty() || !set.is_empty()) {
                set.abort_all();
                let reasons =
                    health.iter().filter_map(|h| if let Health::Out(e) = h { Some(e.clone()) } else { None }).collect();
                break Err(Error::Agent(AgentError::no_agent_left(self.role.as_str(), reasons)));
            }
        };
        drop(held);
        for (i, agent) in self.agents.iter().enumerate() {
            if used[i] {
                self.runner.release(agent).await;
            }
        }
        result?;
        outcomes.sort_by_key(|o| o.index);
        Ok(outcomes)
    }

    /// Each agent's share once a run is over.
    pub fn loads_of(outcomes: &[JobOutcome]) -> Vec<AgentLoad> {
        let mut loads: Vec<AgentLoad> = Vec::new();
        for o in outcomes {
            for a in &o.attempts {
                let i = match loads.iter().position(|l| l.id == a.agent) {
                    Some(i) => i,
                    None => {
                        loads.push(AgentLoad {
                            id: a.agent.clone(),
                            capacity: 0,
                            running: 0,
                            done: 0,
                            failed: 0,
                            busy_s: 0.0,
                        });
                        loads.len() - 1
                    }
                };
                if a.error.is_some() {
                    loads[i].failed += 1;
                } else {
                    loads[i].done += 1;
                    loads[i].busy_s += a.seconds;
                }
            }
        }
        loads
    }
}
