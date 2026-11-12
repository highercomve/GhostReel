//! `ghostreel clips …`: generate a clip set from a project's footage, list sets, export them.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Context;
use clap::Subcommand;
use ghostreel_core::agents::AgentSelection;
use ghostreel_core::clips::{self, ClipFormat, ClipOptions};
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::doctor;
use ghostreel_core::paths::Paths;
use ghostreel_core::runs::{EventLevel, RunLink, RunRecorder, RunSink, StepEvent};
use ghostreel_core::script::IssueSeverity;

#[derive(Subcommand)]
pub enum ClipsAction {
    /// Pick N short self-contained clips from the project's videos and save them as a clip set.
    Generate {
        #[arg(long, short)]
        project: String,
        /// Optional editorial brief ("moments about cost", "the funny bits").
        #[arg(long, default_value = "")]
        brief: String,
        #[arg(long, short = 'n', default_value_t = clips::DEFAULT_COUNT)]
        count: usize,
        /// Shortest clip, seconds.
        #[arg(long, default_value_t = clips::DEFAULT_MIN_S)]
        min: f64,
        /// Longest clip, seconds.
        #[arg(long, default_value_t = clips::DEFAULT_MAX_S)]
        max: f64,
        /// original, 9:16 or 1:1.
        #[arg(long, default_value = "original")]
        format: String,
        /// Only these video ids (comma-separated; default: the whole library).
        #[arg(long, value_delimiter = ',')]
        videos: Vec<i64>,
        /// Skip the orchestrator model: code merges and picks from the workers' scores.
        #[arg(long)]
        worker_only: bool,
        /// Framing for 9:16 / 1:1: follow_speaker (default), stacked or fixed.
        #[arg(long, default_value = "follow_speaker")]
        layout: String,
        /// Agent that picks the final clips (default: the app's choice; see `ghostreel agents list`).
        #[arg(long)]
        orchestrator: Option<String>,
        /// Agents that score windows in parallel, comma-separated (default: the app's choice).
        #[arg(long, value_delimiter = ',')]
        agents: Vec<String>,
        /// Most worker jobs at once across all agents (default: their summed capacity).
        #[arg(long)]
        max_parallel: Option<usize>,
        #[arg(long)]
        json: bool,
    },
    /// List the sentence-bounded candidate windows code finds (no model involved).
    Candidates {
        #[arg(long, short)]
        project: String,
        #[arg(long, default_value_t = clips::DEFAULT_MIN_S)]
        min: f64,
        #[arg(long, default_value_t = clips::DEFAULT_MAX_S)]
        max: f64,
        #[arg(long, default_value = "original")]
        format: String,
        #[arg(long, value_delimiter = ',')]
        videos: Vec<i64>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// List a project's clip sets and their clips.
    List {
        #[arg(long, short)]
        project: String,
        #[arg(long)]
        json: bool,
    },
    /// Render each clip of a set to an MP4 (titles + speech captions burnt in) and a sidecar .srt.
    Export {
        set_id: String,
        #[arg(long, short = 'o', alias = "output")]
        out: PathBuf,
        /// Only clips marked accepted.
        #[arg(long)]
        accepted: bool,
        /// Include rejected clips too.
        #[arg(long, conflicts_with = "accepted")]
        all: bool,
    },
}

fn open_db(paths: &Paths) -> anyhow::Result<Db> {
    Ok(Db::open(&paths.db_file())?)
}

fn mmss(s: f64) -> String {
    let s = s.max(0.0);
    format!("{}:{:04.1}", (s / 60.0).floor() as i64, s % 60.0)
}

pub async fn run(paths: &Paths, action: ClipsAction) -> anyhow::Result<ExitCode> {
    match action {
        ClipsAction::Generate {
            project,
            brief,
            count,
            min,
            max,
            format,
            videos,
            worker_only,
            layout,
            orchestrator,
            agents,
            max_parallel,
            json,
        } => {
            let mut db = open_db(paths)?;
            let p = db.require_project(&project)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            let opts = ClipOptions {
                brief,
                count,
                min_s: min,
                max_s: max,
                format: ClipFormat::parse(&format)?,
                video_ids: videos,
                worker_only,
                layout: layout.parse()?,
                agents: AgentSelection {
                    orchestrator,
                    workers: (!agents.is_empty()).then_some(agents),
                    max_parallel,
                    ..Default::default()
                },
                ..ClipOptions::default()
            };
            opts.validate()?;
            let backends = clips::Backends::resolve(paths, &config, &opts).await?;
            eprintln!("{}", backends.describe());
            let cancel = Arc::new(AtomicBool::new(false));
            let c2 = cancel.clone();
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    c2.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            });
            let started = std::time::Instant::now();
            // Recorded as a run so the app's history shows CLI runs next to its own.
            let label = format!("Generate {count} {} “{}”", if count == 1 { "clip" } else { "clips" }, p.name);
            let mut sink = CliSink { rec: RunRecorder::start("clips", label, Some(p.id), "cli"), started };
            let res = clips::run(&mut db, p.id, &config.script, &backends, &opts, Some(cancel), &mut sink).await;
            let mut rec = sink.rec;
            if let Err(e) = &res
                && let Some(ev) = StepEvent::from_error(e)
            {
                rec.event(ev);
            }
            rec.finish_with(
                &res,
                |r| {
                    let n = r.set.items.len();
                    let summary = format!("Saved {n} {} · {}", if n == 1 { "clip" } else { "clips" }, r.stats_line());
                    (Some(summary), vec![RunLink::ClipSet { id: r.set_id().to_string() }])
                },
                |e| e.to_string().contains(ghostreel_core::chat::CANCELLED),
            );
            if let Err(e) = rec.save(&db) {
                eprintln!("ghostreel: could not record the run: {e}");
            }
            let res = res?;
            if json {
                println!("{}", serde_json::to_string_pretty(&res)?);
                return Ok(ExitCode::SUCCESS);
            }
            println!("clip set {} — {} clips ({})", res.set.id, res.set.items.len(), opts.format.label());
            for (k, c) in res.clips.iter().enumerate() {
                let Some(id) = c.script_id else {
                    println!("  [skipped] {} — errors: {:?}", c.pick.title, c.issues.iter().map(|i| &i.message).collect::<Vec<_>>());
                    continue;
                };
                println!(
                    "{:>2}. #{id} v{} {}–{} ({:.0} s) score {:.1} · {}\n    hook: {}\n    title: {}\n    why: {}\n    quote: {}",
                    k + 1,
                    c.pick.video_id,
                    mmss(c.pick.in_s),
                    mmss(c.pick.out_s),
                    c.pick.duration_s(),
                    c.pick.score,
                    c.pick.visual.summary(),
                    c.script.beats[0].on_screen_text.as_deref().unwrap_or_default(),
                    c.pick.title,
                    c.pick.why,
                    c.pick.quote
                );
                for i in c.issues.iter().filter(|i| i.severity != IssueSeverity::Info) {
                    println!("    [{:?}] {}", i.severity, i.message);
                }
            }
            println!("timings: {}", res.log["timings_s"]);
            Ok(ExitCode::SUCCESS)
        }
        ClipsAction::Candidates { project, min, max, format, videos, limit, json } => {
            let db = open_db(paths)?;
            let p = db.require_project(&project)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            let opts = ClipOptions {
                min_s: min,
                max_s: max,
                format: ClipFormat::parse(&format)?,
                video_ids: videos,
                ..ClipOptions::default()
            };
            opts.validate()?;
            let mut cands = clips::find_candidates(&db, p.id, &config.script, &opts)?;
            let total = cands.len();
            cands.sort_by(|a, b| b.prescore.total_cmp(&a.prescore));
            cands.truncate(limit);
            if json {
                println!("{}", serde_json::to_string_pretty(&cands)?);
            } else {
                println!("{total} candidates ({min:.0}–{max:.0} s); best {}:", cands.len());
                for c in &cands {
                    let text: String = c.text.chars().take(100).collect();
                    println!(
                        "{:<8} v{} {}–{} ({:.0} s) pre {:.2} · {} · {text}…",
                        c.id,
                        c.video_id,
                        mmss(c.in_s),
                        mmss(c.out_s),
                        c.duration_s(),
                        c.prescore,
                        c.visual.summary()
                    );
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        ClipsAction::List { project, json } => {
            let db = open_db(paths)?;
            let p = db.require_project(&project)?;
            let sets = clips::list_sets(&db, p.id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&sets)?);
                return Ok(ExitCode::SUCCESS);
            }
            if sets.is_empty() {
                println!("no clip sets yet; run `ghostreel clips generate -p \"{}\"`", p.name);
            }
            for s in sets {
                let brief = if s.brief.is_empty() { "(no brief)".to_string() } else { s.brief.clone() };
                println!("{} · {} clips · {} · {brief}", s.id, s.items.len(), s.options["format"].as_str().unwrap_or("original"));
                for i in &s.items {
                    println!(
                        "   #{} [{}] {:.1} v{} {}–{} {}",
                        i.script_id,
                        i.status,
                        i.score,
                        i.video_id,
                        mmss(i.in_s),
                        mmss(i.out_s),
                        if i.title.is_empty() { &i.hook } else { &i.title }
                    );
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        ClipsAction::Export { set_id, out, accepted, all } => {
            let db = open_db(paths)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            let ffmpeg = doctor::locate("ffmpeg").context("ffmpeg not found")?;
            let which = if accepted {
                clips::ExportWhich::Accepted
            } else if all {
                clips::ExportWhich::All
            } else {
                clips::ExportWhich::NotRejected
            };
            let files = clips::export_set(&db, &paths.data_dir, &ffmpeg, &set_id, &out, which, &config.script, None, |k, n, name| {
                if k < n {
                    eprintln!("rendering {}/{n}: {name}", k + 1);
                }
            })?;
            for f in files {
                println!("{}", f.display());
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// A clip run from the terminal: progress and events to stderr, and into the recorded run.
struct CliSink {
    rec: RunRecorder,
    started: std::time::Instant,
}

impl RunSink for CliSink {
    fn progress(&mut self, frac: f64, phase: &str, note: &str) {
        self.rec.progress(frac, phase, note);
        eprintln!("[{:>5.1}s {:>3.0}%] {phase}: {note}", self.started.elapsed().as_secs_f64(), frac * 100.0);
        let _ = std::io::stderr().flush();
    }

    fn event(&mut self, event: StepEvent) {
        let tag = match event.level {
            EventLevel::Info => "note",
            EventLevel::Warning => "warning",
            EventLevel::Error => "error",
        };
        eprintln!("[{:>5.1}s] {tag}: {}", self.started.elapsed().as_secs_f64(), event.message);
        self.rec.event(event);
    }
}
