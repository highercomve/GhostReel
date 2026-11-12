//! `ghostreel agents …`: the agent pool — list it, check agents, remove one.
//!
//! Agents are edited with `ghostreel config set agents.<id>.<field> <value>` (creating the agent
//! when the id is new) and the app-wide choice with `agent_selection.<field>`.

use std::process::ExitCode;

use clap::Subcommand;
use ghostreel_core::agents::{self, AgentError, AgentErrorKind, AgentJob, AgentPool, AgentRole, AgentSpec, Effort};
use ghostreel_core::preflight::Level;
use ghostreel_core::config::Config;
use ghostreel_core::paths::Paths;
use serde_json::json;

#[derive(Subcommand)]
pub enum AgentsAction {
    /// List the configured agents (or the implicit ones an older config stands for).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Check agents are available (installed / reachable / model listed). `--call` also sends
    /// each one tiny structured question.
    Test {
        /// Only this agent.
        id: Option<String>,
        #[arg(long)]
        call: bool,
        #[arg(long)]
        json: bool,
    },
    /// Add an agent (or replace the one with this id).
    Add {
        id: String,
        /// cli, server or local.
        #[arg(long, default_value = "cli")]
        backend: String,
        /// claude, agy, opencode or codex (cli).
        #[arg(long, default_value = "")]
        tool: String,
        /// Model id (server or CLI); empty = the endpoint's default.
        #[arg(long, default_value = "")]
        model: String,
        /// low, medium, high, xhigh or max (claude, codex); empty = the tool's default.
        #[arg(long, default_value = "")]
        effort: String,
        /// Server base URL (server).
        #[arg(long, default_value = "")]
        url: String,
        /// Catalog id or external:<path> (local).
        #[arg(long, default_value = "")]
        local_model: String,
        /// Comma-separated: orchestrator, worker, chat, vision.
        #[arg(long, value_delimiter = ',', default_value = "worker")]
        roles: Vec<String>,
        /// Calls at once (a local model always takes one).
        #[arg(long, default_value_t = 1)]
        max_concurrent: usize,
        /// Lower gets jobs first.
        #[arg(long, default_value_t = 100)]
        priority: u32,
        #[arg(long, default_value = "")]
        name: String,
    },
    /// Remove an agent.
    Remove { id: String },
}

fn roles(a: &AgentSpec) -> String {
    a.roles.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(",")
}

pub async fn run(paths: &Paths, action: AgentsAction) -> anyhow::Result<ExitCode> {
    let mut config = Config::load(&paths.config_file)?;
    match action {
        AgentsAction::List { json } => {
            let list = config.agents();
            let sel = config.resolve_agents(None, None);
            if json {
                let views: Vec<_> = list.iter().map(|a| a.effective(&config)).collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "implicit": config.agents.is_empty(),
                        "agents": views,
                        "app_selection": config.agent_selection,
                        "resolved": sel.as_ref().ok(),
                    }))?
                );
                return Ok(ExitCode::SUCCESS);
            }
            if config.agents.is_empty() {
                println!("(implicit agents from [chat_model], [chat_worker] and [vision]; any `config set agents.…` makes them explicit)");
            }
            for a in &list {
                let e = a.effective(&config);
                println!(
                    "{:<10} {:<4} {:<42} ×{:<2} prio {:<3} roles {}{}{}",
                    a.id,
                    if a.enabled { "on" } else { "off" },
                    e.summary(),
                    a.capacity(),
                    a.priority,
                    roles(a),
                    a.source.map(|s| format!(" · linked to {}", s.label())).unwrap_or_default(),
                    if a.backend == ghostreel_core::config::Backend::Local { " · one at a time" } else { "" }
                );
            }
            match sel {
                Ok(s) => println!(
                    "\ndefault: orchestrator {} · chat {} · {}",
                    s.orchestrator.as_ref().map(|a| a.id.as_str()).unwrap_or("(code picks)"),
                    s.chat.as_ref().map(|a| a.id.as_str()).unwrap_or("[chat_model]"),
                    s.capacity_line()
                ),
                Err(e) => println!("\napp selection is invalid: {e}"),
            }
        }
        AgentsAction::Test { id, call, json } => {
            let list: Vec<AgentSpec> =
                config.agents().into_iter().filter(|a| id.as_ref().is_none_or(|i| &a.id == i)).collect();
            if list.is_empty() {
                anyhow::bail!("no agent '{}'", id.unwrap_or_default());
            }
            let mut reports = Vec::new();
            let mut all_ok = true;
            for a in &list {
                let av = agents::check_availability(paths, &config, a).await;
                let mut answer = None;
                if call && av.ok {
                    let role = if a.has_role(AgentRole::Worker) { AgentRole::Worker } else { AgentRole::Orchestrator };
                    let started = std::time::Instant::now();
                    let res = match AgentPool::resolve(paths, &config, role, std::slice::from_ref(a), 1, 120).await {
                        Ok(pool) => pool
                            .run_jobs(
                                vec![AgentJob {
                                    label: "test".into(),
                                    prompt: "Reply with ok set to true and the word 'ready' as note.".into(),
                                    schema: json!({"type":"object","properties":{"ok":{"type":"boolean"},"note":{"type":"string"}},"required":["ok","note"]}),
                                    max_tokens: 64,
                                    // A probe of whether the agent answers: no tools to wander off with.
                                    tools: None,
                                }],
                                None,
                                &mut |_| {},
                            )
                            .await
                            .map_err(|e| match e.agent_error() {
                                // One agent tested: its own reason, not "no agent left".
                                Some(AgentError { kind: AgentErrorKind::NoAgentLeft { agents, .. }, .. })
                                    if agents.len() == 1 =>
                                {
                                    agents[0].clone()
                                }
                                Some(a) => a.clone(),
                                None => AgentError::new(a.id.clone(), AgentErrorKind::Other, e.to_string()),
                            })
                            .and_then(|mut o| o.remove(0).value),
                        Err(e) => Err(AgentError::new(a.id.clone(), AgentErrorKind::Other, e.to_string())),
                    };
                    answer = Some(match res {
                        Ok(v) => json!({ "ok": true, "seconds": started.elapsed().as_secs_f64(), "answer": v }),
                        Err(e) => json!({ "ok": false, "seconds": started.elapsed().as_secs_f64(), "error": e.to_string(), "error_detail": e }),
                    });
                }
                all_ok &= av.ok && answer.as_ref().is_none_or(|a| a["ok"] == true);
                if !json {
                    let status = match av.level {
                        Level::Ok => "ok  ",
                        Level::Warn => "warn",
                        Level::Error => "FAIL",
                    };
                    println!("{:<10} {status} {} — {}", a.id, av.resolved, av.detail);
                    if let Some(ans) = &answer {
                        match ans["ok"].as_bool() {
                            Some(true) => println!("           answered in {:.1} s: {}", ans["seconds"].as_f64().unwrap_or(0.0), ans["answer"]),
                            _ => println!("           call failed: {}", ans["error"].as_str().unwrap_or_default()),
                        }
                    }
                }
                reports.push(json!({ "availability": av, "call": answer }));
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&reports)?);
            }
            return Ok(if all_ok { ExitCode::SUCCESS } else { ExitCode::FAILURE });
        }
        AgentsAction::Add { id, backend, tool, model, effort, url, local_model, roles, max_concurrent, priority, name } => {
            let backend = match backend.as_str() {
                "cli" => ghostreel_core::config::Backend::Cli,
                "server" => ghostreel_core::config::Backend::Server,
                "local" => ghostreel_core::config::Backend::Local,
                other => anyhow::bail!("backend must be cli, server or local, got '{other}'"),
            };
            let mut spec = AgentSpec {
                id: id.clone(),
                name,
                backend,
                tool,
                model,
                effort: Effort::parse_opt(&effort).map_err(|e| anyhow::anyhow!(e))?,
                url,
                local_model,
                roles: roles.iter().map(|r| AgentRole::parse(r)).collect::<Result<_, _>>().map_err(|e| anyhow::anyhow!(e))?,
                max_concurrent,
                priority,
                ..AgentSpec::default()
            };
            spec.normalize_effort();
            config.upsert_agent(spec).map_err(|e| anyhow::anyhow!(e))?;
            config.save(&paths.config_file)?;
            println!("saved agent {id}");
        }
        AgentsAction::Remove { id } => {
            config.remove_agent(&id).map_err(|e| anyhow::anyhow!(e))?;
            config.save(&paths.config_file)?;
            println!("removed agent {id}");
        }
    }
    Ok(ExitCode::SUCCESS)
}
