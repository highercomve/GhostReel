use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures_util::future::BoxFuture;
use serde_json::{Value, json};

use super::*;
use crate::config::{Backend, Config, VisionConfig, WorkerConfig};

fn agent(id: &str, roles: &[AgentRole], cap: usize) -> AgentSpec {
    AgentSpec {
        id: id.into(),
        backend: Backend::Cli,
        tool: "claude".into(),
        max_concurrent: cap,
        roles: roles.iter().copied().collect(),
        ..AgentSpec::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Config back-compat

#[test]
fn configs_without_agents_keep_their_backends_roles_and_concurrency() {
    // The sandbox's shape: a CLI chat model and a server worker.
    let mut cfg = Config::default();
    let mut chat = VisionConfig { backend: Backend::Cli, ..VisionConfig::default() };
    chat.cli.tool = "claude".into();
    chat.cli.model = "claude-opus-5-5".into();
    cfg.chat_model = Some(chat);
    cfg.chat_worker = Some(WorkerConfig {
        backend: Some(Backend::Server),
        url: "http://127.0.0.1:8771".into(),
        model: "qwen3.5-9b".into(),
        ..WorkerConfig::default()
    });
    let agents = cfg.agents();
    assert_eq!(agents.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(), ["chat", "worker", "vision"]);
    let chat = &agents[0];
    assert!(chat.has_role(AgentRole::Orchestrator) && chat.has_role(AgentRole::Chat));
    assert!(!chat.has_role(AgentRole::Worker), "workers are [chat_worker]'s job");
    // The settings each role resolves with are exactly what the sections said.
    assert_eq!(chat.llm_config(&cfg, AgentRole::Chat), cfg.chat_model());
    assert_eq!(agents[1].llm_config(&cfg, AgentRole::Worker), cfg.chat_worker());
    assert_eq!(agents[1].max_concurrent, cfg.worker_concurrency());
    assert_eq!(agents[2].llm_config(&cfg, AgentRole::Vision), cfg.vision);
    assert_eq!(cfg.vision_llm(), cfg.vision);

    let sel = cfg.resolve_agents(None, None).unwrap();
    assert_eq!(sel.orchestrator.as_ref().unwrap().id, "chat");
    assert_eq!(sel.workers.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(), ["worker"]);
    assert_eq!(sel.max_parallel, 4, "a server worker ran four at once");
    assert_eq!(sel.chat.as_ref().unwrap().id, "chat");

    // Without [chat_worker] the chat model scores clips too, at the worker concurrency.
    cfg.chat_worker = None;
    let agents = cfg.agents();
    assert_eq!(agents.len(), 2);
    assert!(agents[0].has_role(AgentRole::Worker));
    assert_eq!(agents[0].llm_config(&cfg, AgentRole::Worker), cfg.chat_worker());
    assert_eq!(cfg.resolve_agents(None, None).unwrap().max_parallel, cfg.worker_concurrency());

    // A config file without the new keys still loads, and saving it writes none of them.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "[chat_model]\nbackend = \"local\"\n").unwrap();
    let loaded = Config::load(&path).unwrap();
    assert!(loaded.agents.is_empty() && loaded.agent_selection.is_empty());
    assert_eq!(loaded.agents()[0].capacity(), 1, "a local chat model takes one call at a time");
    assert!(!loaded.to_toml().unwrap().contains("agents"));
}

#[test]
fn explicit_agents_roundtrip_and_materialize_from_old_sections() {
    let mut cfg = Config {
        chat_worker: Some(WorkerConfig {
            backend: Some(Backend::Server),
            url: "http://127.0.0.1:8771".into(),
            model: "qwen3.5-9b".into(),
            concurrency: 2,
            ..WorkerConfig::default()
        }),
        ..Config::default()
    };
    let old = cfg.clone();
    cfg.set_key("agents.codex.tool", "codex").unwrap();
    cfg.set_key("agents.codex.roles", "worker,orchestrator").unwrap();
    cfg.set_key("agents.codex.max_concurrent", "2").unwrap();
    // The worker section became a standalone agent and the section is gone.
    assert!(cfg.chat_worker.is_none());
    let worker = cfg.agents.iter().find(|a| a.id == "worker").unwrap();
    assert_eq!((worker.backend, worker.url.as_str(), worker.model.as_str()), (Backend::Server, "http://127.0.0.1:8771", "qwen3.5-9b"));
    let (now, was) = (worker.llm_config(&cfg, AgentRole::Worker), old.chat_worker());
    assert_eq!((now.backend, now.url, now.model, now.think), (was.backend, was.url, was.model, was.think));
    assert_eq!(worker.max_concurrent, old.worker_concurrency());
    assert!(cfg.set_key("chat_worker.model", "x").is_err(), "no second place to set workers");

    cfg.set_key("agent_selection.workers", "worker,codex").unwrap();
    assert!(cfg.set_key("agent_selection.orchestrator", "worker").is_err(), "worker cannot orchestrate");
    assert!(cfg.set_key("agent_selection.workers", "nobody").is_err());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    cfg.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("[[agents]]"), "{text}");
    let back = Config::load(&path).unwrap();
    // Read back, linked agents carry their sections' values (derived, never stored).
    let mut want = cfg.clone();
    want.normalize();
    assert_eq!(back, want);
    assert_eq!(back.agents(), cfg.agents());
    assert_eq!(back.validate(), Ok(()));

    cfg.remove_agent("codex").unwrap();
    assert_eq!(cfg.agent_selection.workers.as_deref(), Some(&["worker".to_string()][..]));
}

#[test]
fn agent_specs_are_validated() {
    assert!(agent("a b", &[], 1).validate().is_err());
    assert!(agent("a", &[], 0).validate().is_err());
    assert!(AgentSpec { tool: "gpt".into(), ..agent("a", &[], 1) }.validate().is_err());
    assert!(AgentSpec { backend: Backend::Server, url: "nope".into(), ..agent("a", &[], 1) }.validate().is_err());
    assert!(AgentSpec { backend: Backend::Local, ..agent("a", &[], 1) }.validate().is_err());
    assert!(agent("claude", &[AgentRole::Worker], 2).validate().is_ok());
    assert_eq!(AgentSpec { backend: Backend::Local, local_model: "m".into(), ..agent("l", &[], 4) }.capacity(), 1);
}

// ---------------------------------------------------------------------------------------------
// Selection

#[test]
fn selection_resolves_run_over_cut_over_app_field_by_field() {
    use AgentRole::*;
    let agents = vec![
        AgentSpec { priority: 1, ..agent("claude", &[Orchestrator, Worker, Chat], 2) },
        AgentSpec { priority: 2, ..agent("codex", &[Orchestrator, Worker], 2) },
        AgentSpec { priority: 3, backend: Backend::Local, local_model: "m".into(), ..agent("local", &[Worker], 4) },
        AgentSpec { enabled: false, ..agent("off", &[Worker], 1) },
    ];
    // Nothing chosen: the first orchestrator, every enabled worker, their capacity.
    let d = resolve_selection(&agents, None, None, &AgentSelection::default()).unwrap();
    assert_eq!(d.orchestrator.as_ref().unwrap().id, "claude");
    assert_eq!(d.orchestrator_from, SelectionSource::Default);
    assert_eq!(d.workers.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(), ["claude", "codex", "local"]);
    assert_eq!((d.capacity, d.max_parallel), (5, 5), "local counts one");
    assert_eq!(d.capacity_line(), "up to 5 at once: claude ×2 · codex ×2 · local ×1");

    let app = AgentSelection { workers: Some(vec!["local".into()]), max_parallel: Some(3), ..Default::default() };
    let cut = AgentSelection { orchestrator: Some("codex".into()), ..Default::default() };
    let run = AgentSelection { workers: Some(vec!["claude".into(), "codex".into()]), ..Default::default() };
    let r = resolve_selection(&agents, Some(&run), Some(&cut), &app).unwrap();
    assert_eq!((r.orchestrator.as_ref().unwrap().id.as_str(), r.orchestrator_from), ("codex", SelectionSource::Cut));
    assert_eq!(r.workers_from, SelectionSource::Run);
    assert_eq!(r.max_parallel, 3, "the app's cap still applies");
    // codex cannot chat, so script chat falls to the first chat agent.
    assert_eq!(r.chat.as_ref().unwrap().id, "claude");
    assert_eq!(
        r.effective(),
        AgentSelection {
            orchestrator: Some("codex".into()),
            workers: Some(vec!["claude".into(), "codex".into()]),
            max_parallel: Some(3),
            concurrency: Some([("claude".to_string(), 2), ("codex".to_string(), 2)].into()),
            ..Default::default()
        }
    );
    // The app alone; a cap above the capacity is clamped.
    let only_app = resolve_selection(&agents, None, None, &app).unwrap();
    assert_eq!((only_app.workers.len(), only_app.max_parallel, only_app.workers_from), (1, 1, SelectionSource::App));
}

#[test]
fn selection_errors_name_the_problem() {
    use AgentRole::*;
    let agents = vec![agent("claude", &[Orchestrator, Worker], 2), AgentSpec { enabled: false, ..agent("off", &[Worker], 1) }, agent("eyes", &[Vision], 1)];
    let app = AgentSelection::default();
    let err = |sel: AgentSelection| resolve_selection(&agents, Some(&sel), None, &app).unwrap_err();
    assert!(matches!(err(AgentSelection { orchestrator: Some("gpt".into()), ..Default::default() }), SelectionError::UnknownAgent { id, .. } if id == "gpt"));
    assert_eq!(err(AgentSelection { workers: Some(vec!["off".into()]), ..Default::default() }), SelectionError::Disabled { id: "off".into() });
    let missing = err(AgentSelection { orchestrator: Some("eyes".into()), ..Default::default() });
    assert_eq!(missing, SelectionError::MissingRole { id: "eyes".into(), role: Orchestrator });
    assert!(missing.to_string().contains("an orchestrator"), "{missing}");
    assert_eq!(err(AgentSelection { max_parallel: Some(0), ..Default::default() }), SelectionError::ZeroParallel);
    assert!(err(AgentSelection { orchestrator: Some("gpt".into()), ..Default::default() }).to_string().contains("claude, off, eyes"));
}

// ---------------------------------------------------------------------------------------------
// Slots

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_local_model_has_one_slot_shared_by_every_caller() {
    let reg = SlotRegistry::new();
    // Two features each think the local model can take four.
    let a = reg.try_acquire(&EndpointKey::Local, 4).expect("free");
    assert!(reg.try_acquire(&EndpointKey::Local, 4).is_none(), "a second caller must wait");
    let reg2 = reg.clone();
    let waiter = tokio::spawn(async move {
        let _s = reg2.acquire(&EndpointKey::Local, 4).await;
        reg2.in_use(&EndpointKey::Local)
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!waiter.is_finished(), "blocked while the first holds it");
    drop(a);
    assert_eq!(tokio::time::timeout(Duration::from_secs(2), waiter).await.unwrap().unwrap(), 1);
    assert_eq!(reg.in_use(&EndpointKey::Local), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn endpoint_capacities_are_respected_and_separate() {
    let reg = SlotRegistry::new();
    let claude = EndpointKey::Cli { tool: "claude".into(), model: String::new() };
    let server = EndpointKey::Server { url: "http://x".into(), model: "m".into() };
    let held: Vec<Slot> = (0..2).map(|_| reg.try_acquire(&claude, 2).unwrap()).collect();
    assert!(reg.try_acquire(&claude, 2).is_none());
    assert!(reg.try_acquire(&server, 1).is_some(), "another endpoint is not affected");
    // Many concurrent callers never exceed the capacity.
    drop(held);
    let peak = Arc::new(AtomicUsize::new(0));
    let now = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();
    for _ in 0..12 {
        let (reg, peak, now, key) = (reg.clone(), peak.clone(), now.clone(), claude.clone());
        tasks.push(tokio::spawn(async move {
            let _s = reg.acquire(&key, 3).await;
            let n = now.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(n, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(10)).await;
            now.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    assert_eq!(peak.load(Ordering::SeqCst), 3);
    let snap = reg.snapshot();
    assert!(snap.iter().all(|e| e.in_use == 0));
}

// ---------------------------------------------------------------------------------------------
// Dispatcher

/// Answers after `delay_ms`; fails every job on agents listed in `broken`.
struct FakeRunner {
    delay_ms: u64,
    broken: Vec<&'static str>,
    released: AtomicUsize,
}

impl JobRunner for FakeRunner {
    fn run(&self, agent: Arc<PoolAgent>, job: Arc<AgentJob>) -> BoxFuture<'static, Result<Value, AgentError>> {
        let fail = self.broken.contains(&agent.spec.id.as_str());
        let delay = self.delay_ms;
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            if fail {
                Err(AgentError::new(agent.spec.id.clone(), AgentErrorKind::Exit { code: Some(1) }, "broke"))
            } else {
                Ok(json!({ "by": agent.spec.id, "job": job.label }))
            }
        })
    }
    fn release(&self, _agent: &PoolAgent) -> BoxFuture<'static, ()> {
        self.released.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {})
    }
}

fn pool_agent(id: &str, cap: usize, priority: u32, key: EndpointKey) -> PoolAgent {
    PoolAgent {
        spec: AgentSpec { priority, max_concurrent: cap, ..agent(id, &[AgentRole::Worker], cap) },
        capacity: key.capacity(cap),
        key,
        target: AgentTarget::External,
        describe: id.into(),
        max_tool_rounds: 0,
    }
}

fn cli(tool: &str) -> EndpointKey {
    EndpointKey::Cli { tool: tool.into(), model: String::new() }
}

fn jobs(n: usize) -> Vec<AgentJob> {
    (0..n).map(|i| AgentJob { label: format!("j{i}"), prompt: String::new(), schema: json!({}), max_tokens: 16, tools: None }).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jobs_spread_over_agents_by_capacity_and_report_counts() {
    let runner = Arc::new(FakeRunner { delay_ms: 60, broken: vec![], released: AtomicUsize::new(0) });
    let pool = AgentPool::new(
        vec![
            pool_agent("claude", 2, 1, cli("claude")),
            pool_agent("codex", 2, 1, cli("codex")),
            pool_agent("local", 4, 2, EndpointKey::Local),
        ],
        16,
        SlotRegistry::new(),
        runner.clone(),
    );
    assert_eq!(pool.max_parallel(), 5);
    let mut peak_line = String::new();
    let mut peak = 0;
    let out = pool
        .run_jobs(jobs(10), None, &mut |p: &PoolProgress| {
            let running: usize = p.agents.iter().map(|a| a.running).sum();
            if running > peak {
                peak = running;
                peak_line = p.running_line();
            }
        })
        .await
        .unwrap();
    assert_eq!(out.len(), 10);
    assert_eq!(out.iter().map(|o| o.index).collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
    assert!(out.iter().all(|o| o.value.is_ok() && o.attempts.len() == 1));
    assert_eq!(peak, 5, "every slot filled");
    assert_eq!(peak_line, "claude 2 · codex 2 · local 1 running");
    let loads = AgentPool::loads_of(&out);
    let by = |id: &str| loads.iter().find(|l| l.id == id).map(|l| l.done).unwrap_or(0);
    assert!(by("claude") >= 3 && by("codex") >= 3 && by("local") >= 1, "{loads:?}");
    assert_eq!(by("claude") + by("codex") + by("local"), 10);
    assert!(share_line(&loads).contains("claude"));
    assert_eq!(runner.released.load(Ordering::SeqCst), 3, "every agent used is released");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_job_is_retried_once_on_another_agent() {
    let runner = Arc::new(FakeRunner { delay_ms: 5, broken: vec!["flaky"], released: AtomicUsize::new(0) });
    let pool = AgentPool::new(
        vec![pool_agent("flaky", 2, 1, cli("agy")), pool_agent("steady", 1, 5, cli("claude"))],
        3,
        SlotRegistry::new(),
        runner,
    );
    let out = pool.run_jobs(jobs(4), None, &mut |_| {}).await.unwrap();
    for o in &out {
        assert!(o.value.is_ok(), "{:?}", o.attempts);
        assert_eq!(o.agent, "steady");
        if o.attempts.len() == 2 {
            assert_eq!(o.attempts[0].agent, "flaky");
            assert!(o.attempts[0].error.is_some());
        }
    }
    assert!(out.iter().any(|o| o.attempts.len() == 2), "flaky took some first");

    // With only one agent the retry goes back to it, once; then the failure is reported.
    let runner = Arc::new(FakeRunner { delay_ms: 1, broken: vec!["flaky"], released: AtomicUsize::new(0) });
    let pool = AgentPool::new(vec![pool_agent("flaky", 1, 1, cli("agy"))], 1, SlotRegistry::new(), runner);
    let out = pool.run_jobs(jobs(1), None, &mut |_| {}).await.unwrap();
    assert_eq!(out[0].attempts.len(), 2);
    assert!(out[0].value.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_pool_waits_for_slots_other_features_hold() {
    let reg = SlotRegistry::new();
    // Script chat is using the local model.
    let chat_turn = reg.try_acquire(&EndpointKey::Local, 1).unwrap();
    let runner = Arc::new(FakeRunner { delay_ms: 5, broken: vec![], released: AtomicUsize::new(0) });
    let pool = AgentPool::new(vec![pool_agent("local", 1, 1, EndpointKey::Local)], 1, reg.clone(), runner);
    let task = tokio::spawn(async move { pool.run_jobs(jobs(2), None, &mut |_| {}).await.map(|o| o.len()) });
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(!task.is_finished(), "never two calls on the local model");
    drop(chat_turn);
    assert_eq!(tokio::time::timeout(Duration::from_secs(3), task).await.unwrap().unwrap().unwrap(), 2);
    assert_eq!(reg.in_use(&EndpointKey::Local), 0, "the pool gave the slot back");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_stops_the_run() {
    let runner = Arc::new(FakeRunner { delay_ms: 2000, broken: vec![], released: AtomicUsize::new(0) });
    let pool = AgentPool::new(vec![pool_agent("a", 1, 1, cli("claude"))], 1, SlotRegistry::new(), runner);
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        c2.store(true, Ordering::SeqCst);
    });
    let err = pool.run_jobs(jobs(3), Some(cancel), &mut |_| {}).await.unwrap_err();
    assert!(err.to_string().contains(crate::chat::CANCELLED));
}

// ---------------------------------------------------------------------------------------------
// Failures by kind

/// How an agent answers its n-th call: `Some` = fails that way.
type Script = fn(usize) -> Option<AgentError>;

/// Fails each agent's calls the way the real tools do, from the captured outputs.
struct ScriptedRunner {
    /// agent id → how it fails (`None`: it answers).
    fails: std::collections::HashMap<&'static str, Script>,
    calls: std::sync::Mutex<std::collections::HashMap<String, usize>>,
}

impl ScriptedRunner {
    fn new(fails: &[(&'static str, Script)]) -> Arc<Self> {
        Arc::new(Self { fails: fails.iter().copied().collect(), calls: Default::default() })
    }
    fn calls(&self, id: &str) -> usize {
        self.calls.lock().unwrap().get(id).copied().unwrap_or(0)
    }
}

impl JobRunner for ScriptedRunner {
    fn run(&self, agent: Arc<PoolAgent>, job: Arc<AgentJob>) -> BoxFuture<'static, Result<Value, AgentError>> {
        let n = {
            let mut calls = self.calls.lock().unwrap();
            let c = calls.entry(agent.spec.id.clone()).or_default();
            *c += 1;
            *c
        };
        let result = match self.fails.get(agent.spec.id.as_str()).and_then(|f| f(n)) {
            Some(e) => Err(e),
            None => Ok(json!({ "by": agent.spec.id, "job": job.label })),
        };
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            result
        })
    }
}

const AGY_QUOTA: &str = "error: Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 2h2m31s. AGY_ERROR: {\"short_error\":\"RESOURCE_EXHAUSTED (code 429)\"}";
const CLAUDE_BAD_MODEL: &str = r#"{"type":"result","is_error":true,"result":"There's an issue with the selected model (claude-opus-5-5-medium). It may not exist or you may not have access to it."}"#;

fn quota(_: usize) -> Option<AgentError> {
    Some(classify_cli("agy", "", Some(3), "", AGY_QUOTA))
}
fn bad_model(_: usize) -> Option<AgentError> {
    Some(classify_cli("claude", "claude-opus-5-5-medium", Some(1), CLAUDE_BAD_MODEL, ""))
}
/// Busy for its first two calls, then fine.
fn busy_twice(n: usize) -> Option<AgentError> {
    (n <= 2).then(|| classify_http("http://ghostpen", "", 503, Some(0), "busy: too many chat requests waiting, retry later"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_used_up_quota_rests_the_agent_and_its_jobs_go_elsewhere() {
    let runner = ScriptedRunner::new(&[("agy", quota)]);
    let cooldowns = Cooldowns::new();
    // agy first by priority: it gets the first jobs, fails, and must not get the other five.
    let pool = AgentPool::new(
        vec![pool_agent("agy", 2, 1, cli("agy")), pool_agent("codex", 2, 5, cli("codex"))],
        4,
        SlotRegistry::new(),
        runner.clone(),
    )
    .with_cooldowns(cooldowns.clone());
    let mut events = Vec::new();
    let out = pool
        .run_jobs(jobs(7), None, &mut |p: &PoolProgress| {
            if let Some(e) = p.event {
                events.push(e.clone());
            }
        })
        .await
        .unwrap();
    assert!(out.iter().all(|o| o.value.is_ok() && o.agent == "codex"), "{out:?}");
    assert!(runner.calls("agy") <= 2, "only the jobs already in flight reached agy: {}", runner.calls("agy"));
    assert_eq!(events.len(), 1, "one rest, not one line per job: {events:?}");
    assert!(matches!(events[0].action, PoolAction::CoolingDown { .. }));
    assert_eq!(events[0].remaining, ["codex"]);
    assert!(events[0].message().starts_with("agy: quota used up, resets in 2 h 3 min — resting it; its jobs go to codex"));
    // Process-wide: the next run (any feature) sees agy resting.
    let rest = cooldowns.get(&cli("agy")).expect("agy rests");
    assert!(rest.remaining() > Duration::from_secs(7000));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unknown_model_drops_the_agent_and_with_none_left_the_run_stops_with_every_reason() {
    let runner = ScriptedRunner::new(&[("agy", quota), ("claude", bad_model)]);
    let pool = AgentPool::new(
        vec![pool_agent("agy", 1, 1, cli("agy")), pool_agent("claude", 1, 2, cli("claude"))],
        2,
        SlotRegistry::new(),
        runner.clone(),
    )
    .with_cooldowns(Cooldowns::new());
    let err = pool.run_jobs(jobs(7), None, &mut |_| {}).await.unwrap_err();
    let e = err.agent_error().expect("typed");
    let AgentErrorKind::NoAgentLeft { role, agents } = &e.kind else { panic!("{e:?}") };
    assert_eq!(role, "worker");
    let kinds: Vec<(&str, &str)> = agents.iter().map(|a| (a.agent.as_str(), a.kind.key())).collect();
    assert!(kinds.contains(&("agy", "quota_exhausted")) && kinds.contains(&("claude", "invalid_model")), "{kinds:?}");
    let msg = err.to_string();
    assert!(msg.starts_with("no worker agent could take the work — "), "{msg}");
    assert!(msg.contains("claude: model claude-opus-5-5-medium not available"), "{msg}");
    assert!(!msg.contains("Some("), "{msg}");
    // It stopped at once: one call each, not seven failures.
    assert_eq!(runner.calls("agy") + runner.calls("claude"), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_busy_server_is_asked_again_after_a_pause() {
    let runner = ScriptedRunner::new(&[("ghostpen", busy_twice)]);
    let pool = AgentPool::new(
        vec![pool_agent("ghostpen", 1, 1, EndpointKey::Server { url: "http://ghostpen".into(), model: String::new() })],
        1,
        SlotRegistry::new(),
        runner.clone(),
    )
    .with_cooldowns(Cooldowns::new());
    let mut retries = 0;
    let out = pool
        .run_jobs(jobs(2), None, &mut |p: &PoolProgress| {
            if p.event.is_some_and(|e| matches!(e.action, PoolAction::Retrying { .. })) {
                retries += 1;
            }
        })
        .await
        .unwrap();
    assert!(out.iter().all(|o| o.value.is_ok()));
    assert_eq!(retries, 2);
    assert_eq!(out[0].attempts.len(), 3, "busy, busy, answered");
    assert_eq!(out[0].attempts[0].error.as_ref().unwrap().kind.key(), "busy");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resolving_skips_an_agent_that_is_resting() {
    // The global registry: what another run recorded is what this one sees.
    let key = EndpointKey::Cli { tool: "agy".into(), model: "resting-test-model".into() };
    Cooldowns::global().put(&key, Some(Duration::from_secs(600)), quota(1).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let paths = crate::paths::Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
    let cfg = Config::default();
    let resting = AgentSpec { model: "resting-test-model".into(), tool: "agy".into(), ..agent("agy", &[AgentRole::Worker], 1) };
    let av = check_availability(&paths, &cfg, &resting).await;
    // Installed or not, a resting endpoint is reported as resting first.
    assert_eq!(av.level, crate::preflight::Level::Error);
    assert!(av.detail.starts_with("quota used up"), "{}", av.detail);
    assert_eq!(av.error.unwrap().kind.key(), "quota_exhausted");
    Cooldowns::global().clear(&key);
}

// ---------------------------------------------------------------------------------------------
// Effort and linked agents

#[test]
fn legacy_effort_suffixes_become_a_model_and_an_effort() {
    assert_eq!(Effort::split_model("claude", "claude-opus-5-5-medium"), Some(("claude-opus-5-5".into(), Effort::Medium)));
    assert_eq!(Effort::split_model("claude", "opus-max"), Some(("opus".into(), Effort::Max)));
    assert_eq!(Effort::split_model("claude", "claude-opus-5-5"), None);
    // Other tools have real models that end like that.
    assert_eq!(Effort::split_model("codex", "gpt-5.1-codex-max"), None);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(
        &path,
        "[chat_model]\nbackend = \"cli\"\n[chat_model.cli]\ntool = \"claude\"\nmodel = \"claude-opus-5-5-medium\"\n\n\
         [[agents]]\nid = \"opus\"\nbackend = \"cli\"\ntool = \"claude\"\nmodel = \"claude-opus-5-5-high\"\nroles = [\"worker\"]\n",
    )
    .unwrap();
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.chat_model().cli.model, "claude-opus-5-5");
    assert_eq!(cfg.chat_model().cli.effort, Some(Effort::Medium));
    let opus = cfg.agents().into_iter().find(|a| a.id == "opus").unwrap();
    assert_eq!((opus.model.as_str(), opus.effort), ("claude-opus-5-5", Some(Effort::High)));
    assert_eq!(opus.summary(), "cli claude · claude-opus-5-5 · high effort");
    // It reaches the CLI config the runner uses.
    assert_eq!(opus.llm_config(&cfg, AgentRole::Worker).cli.effort, Some(Effort::High));
    // And `config set` splits a pasted legacy id the same way.
    let mut cfg = cfg;
    cfg.set_key("agents.opus.model", "claude-sonnet-5-5-low").unwrap();
    let opus = cfg.agents().into_iter().find(|a| a.id == "opus").unwrap();
    assert_eq!((opus.model.as_str(), opus.effort), ("claude-sonnet-5-5", Some(Effort::High)), "an effort already set stays");
    cfg.set_key("agents.opus.effort", "low").unwrap();
    assert_eq!(cfg.agents().into_iter().find(|a| a.id == "opus").unwrap().effort, Some(Effort::Low));
    assert!(cfg.set_key("agents.opus.effort", "extreme").is_err());
}

#[test]
fn a_linked_agent_has_one_source_for_its_model() {
    // The user's file: the agent still says cli while [chat_model] was switched to a server.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(
        &path,
        "[chat_model]\nbackend = \"server\"\nurl = \"http://127.0.0.1:8090\"\nmodel = \"ghostpen/qwen\"\n\
         [chat_model.cli]\ntool = \"claude\"\nmodel = \"claude-opus-5-5\"\n\n\
         [[agents]]\nid = \"chat\"\nsource = \"chat_model\"\nbackend = \"cli\"\nmax_concurrent = 1\n\
         roles = [\"orchestrator\", \"worker\", \"chat\"]\n",
    )
    .unwrap();
    let cfg = Config::load(&path).unwrap();
    let chat = cfg.agents().into_iter().find(|a| a.id == "chat").unwrap();
    assert_eq!(chat.backend, Backend::Server, "the section decides");
    assert_eq!(chat.model, "ghostpen/qwen");
    assert_eq!(cfg.agents[0].backend, Backend::Server, "derived when read, not kept from the file");
    // Saved, the agent no longer carries a copy of the section's fields.
    let text = cfg.to_toml().unwrap();
    let agents_part = &text[text.find("[[agents]]").unwrap()..];
    for k in LINKED_FIELDS {
        assert!(!agents_part.lines().any(|l| l.starts_with(&format!("{k} ="))), "{k} written for a linked agent:\n{agents_part}");
    }
    assert!(agents_part.contains("source = \"chat_model\""));
    // The section comes first, as before (the file keeps its order).
    assert!(text.find("[vision]").unwrap() < text.find("[[agents]]").unwrap());
    // Switching the section switches the agent, with nothing to keep in step.
    let mut cfg = Config::load(&path).unwrap();
    cfg.set_key("chat_model.backend", "cli").unwrap();
    let chat = cfg.agents().into_iter().find(|a| a.id == "chat").unwrap();
    assert_eq!((chat.backend, chat.tool.as_str(), chat.model.as_str()), (Backend::Cli, "claude", "claude-opus-5-5"));
    // Setting a model field on the agent itself makes it its own.
    cfg.set_key("agents.chat.model", "claude-sonnet-5-5").unwrap();
    let chat = cfg.agents().into_iter().find(|a| a.id == "chat").unwrap();
    assert_eq!(chat.source, None);
    assert_eq!((chat.tool.as_str(), chat.model.as_str()), ("claude", "claude-sonnet-5-5"));
    assert_eq!(cfg.chat_model().cli.model, "claude-opus-5-5", "the section is untouched");
}

#[tokio::test]
async fn preflight_downgrades_an_agent_others_can_cover() {
    let dir = tempfile::tempdir().unwrap();
    let paths = crate::paths::Paths { config_file: dir.path().join("config.toml"), data_dir: dir.path().join("data") };
    let cfg = Config::default();
    // Two servers nothing answers at.
    let down = |id: &str| AgentSpec {
        backend: Backend::Server,
        url: "http://127.0.0.1:9".into(),
        ..agent(id, &[AgentRole::Worker, AgentRole::Orchestrator], 1)
    };
    let (a, b) = (down("gone"), down("gone2"));
    // Next to one that works, it only warns.
    let mut ok = agent("ok", &[AgentRole::Worker], 1);
    ok.backend = Backend::Local;
    ok.local_model = "qwen3.5-9b".into();
    let checks = agent_checks(&paths, &cfg, &[(AgentRole::Worker, vec![a.clone(), ok])], &[]).await;
    assert_eq!(checks.len(), 1, "{checks:?}");
    assert_eq!(checks[0].level, crate::preflight::Level::Warn);
    assert!(checks[0].message.ends_with("the other worker agents take its jobs"), "{}", checks[0].message);
    // Alone, an unreachable worker is an error.
    let checks = agent_checks(&paths, &cfg, &[(AgentRole::Worker, vec![a.clone()])], &[]).await;
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].level, crate::preflight::Level::Error);
    assert_eq!(checks[0].error.as_ref().unwrap().kind.key(), "unreachable");
    // As an optional orchestrator it only warns.
    let checks = agent_checks(&paths, &cfg, &[(AgentRole::Orchestrator, vec![b.clone()])], &[AgentRole::Orchestrator]).await;
    assert_eq!(checks[0].level, crate::preflight::Level::Warn);
    assert!(checks[0].message.ends_with("the orchestrator step is skipped"), "{}", checks[0].message);
}

#[test]
fn a_selection_can_pick_by_score_and_set_calls_at_once() {
    use AgentRole::*;
    let agents = vec![
        AgentSpec { name: "Claude".into(), ..agent("claude", &[Orchestrator, Worker], 2) },
        AgentSpec { name: "GhostPen".into(), backend: Backend::Local, local_model: "q".into(), ..agent("pen", &[Worker], 1) },
    ];
    let app = AgentSelection { orchestrator: Some("claude".into()), ..Default::default() };
    // A run that picks by score wins over the app's orchestrator.
    let run = AgentSelection { pick_by_score: Some(true), orchestrator: Some("claude".into()), ..Default::default() };
    let r = resolve_selection(&agents, Some(&run), None, &app).unwrap();
    assert!(r.orchestrator.is_none() && r.picks_by_score());
    assert_eq!(r.orchestrator_from, SelectionSource::Run);
    assert_eq!(r.effective().pick_by_score, Some(true));
    // Calls at once per worker; a local model stays at one.
    let run = AgentSelection { concurrency: Some([("claude".into(), 4), ("pen".into(), 3)].into()), ..Default::default() };
    let r = resolve_selection(&agents, Some(&run), None, &app).unwrap();
    assert_eq!(r.concurrency(), [("claude".to_string(), 4), ("pen".to_string(), 1)].into());
    assert_eq!((r.capacity, r.max_parallel), (5, 5));
    assert!(!r.picks_by_score());
    let bad = AgentSelection { concurrency: Some([("claude".into(), 0)].into()), ..Default::default() };
    assert_eq!(validate_selection(&agents, &bad), Err(SelectionError::ZeroParallel));
}

#[test]
fn a_resolved_selection_reads_as_one_sentence() {
    use AgentRole::*;
    use crate::clips::ClipFormat;
    let agents = vec![
        AgentSpec { name: "Claude".into(), ..agent("claude", &[Orchestrator, Worker], 1) },
        AgentSpec { name: "GhostPen".into(), ..agent("pen", &[Worker], 2) },
    ];
    let run = AgentSelection { workers: Some(vec!["pen".into(), "claude".into()]), ..Default::default() };
    let r = resolve_selection(&agents, Some(&run), None, &AgentSelection::default()).unwrap();
    let target = WorkTarget { count: Some(2), format: Some(ClipFormat::Vertical) };
    assert_eq!(r.summary(&target), "GhostPen (2 at a time) and Claude (1) score the episode; Claude picks the 2 clips (9:16).");
    assert_eq!(r.summary(&WorkTarget::default()), "GhostPen (2 at a time) and Claude (1) score the episode; Claude picks the clips.");
    let run = AgentSelection { workers: Some(vec!["pen".into()]), pick_by_score: Some(true), max_parallel: Some(1), ..Default::default() };
    let r = resolve_selection(&agents, Some(&run), None, &AgentSelection::default()).unwrap();
    assert_eq!(
        r.summary(&WorkTarget { count: Some(3), format: Some(ClipFormat::Original) }),
        "GhostPen (2 at a time) scores the episode, at most 1 at once; no model picks: the top 3 by score become the clips."
    );
}
