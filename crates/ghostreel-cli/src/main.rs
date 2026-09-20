//! `ghostreel` — headless GhostReel CLI.

mod mcp;

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::export::ExportFormat;
use ghostreel_core::index::{self, Event, IndexLock};
use ghostreel_core::paths::Paths;
use ghostreel_core::probe::{Resolution, Target};
use ghostreel_core::progress::{Progress, eta_text};
use ghostreel_core::projects::NewProject;
use ghostreel_core::runtime;
use ghostreel_core::script::{Audio, IssueSeverity, Script};
use ghostreel_core::watch::FolderWatcher;

#[derive(Parser)]
#[command(name = "ghostreel", version, about = "Search inside your videos — locally")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the MCP tools (projects, indexing, search, scripts) on stdio for an AI agent.
    Mcp,
    /// Check ffmpeg, GPU, database, models and the AI backends GhostReel would use.
    Doctor {
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
    /// Show or create the configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Manage projects.
    Project {
        #[command(subcommand)]
        action: ProjectAction,
    },
    /// Manage the video folders a project watches.
    Folder {
        #[command(subcommand)]
        action: FolderAction,
    },
    /// Scan folders and run pending indexing jobs (resumable).
    Index {
        /// Only this project's folders (default: all projects).
        #[arg(long, short)]
        project: Option<String>,
        /// Keep running and re-index when files change.
        #[arg(long)]
        watch: bool,
        /// Also retry jobs that already failed the maximum number of times.
        #[arg(long)]
        retry_failed: bool,
        #[arg(long)]
        json: bool,
        /// Reset a stage (and all later stages) back to pending before indexing.
        /// Accepted values: frames, transcribe, describe, embed.
        /// When --redo frames is used, old keyframe files are deleted and re-extracted.
        #[arg(long, value_name = "STAGE")]
        redo: Option<String>,
    },
    /// Print a video's transcript.
    Transcript {
        /// Video id (see `ghostreel status --videos`).
        video_id: i64,
        /// SubRip subtitles instead of plain text.
        #[arg(long)]
        srt: bool,
        #[arg(long)]
        json: bool,
    },
    /// Search a project's videos by meaning and keywords.
    Search {
        query: String,
        #[arg(long, short)]
        project: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Only these chunk kinds: moment, transcript, frame.
        #[arg(long, value_delimiter = ',')]
        kind: Option<Vec<String>>,
        /// Keywords only (don't load an embedding model).
        #[arg(long)]
        keywords: bool,
        #[arg(long)]
        json: bool,
    },
    /// List a video's keyframes (JPEG paths).
    Frames {
        video_id: i64,
        #[arg(long)]
        json: bool,
    },
    /// Show indexing progress.
    Status {
        #[arg(long, short)]
        project: Option<String>,
        /// List every video.
        #[arg(long)]
        videos: bool,
        #[arg(long)]
        json: bool,
    },
    /// Manage and export video editing scripts.
    Script {
        #[command(subcommand)]
        action: ScriptAction,
    },
    /// Manage AI models (whisper, vision, embeddings).
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    /// List available models and installation status.
    List {
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Download a model by id.
    Download {
        /// Model id (e.g. tiny, large-v3-turbo, bonsai-27b, embeddinggemma-300M-Q8_0).
        id: String,
    },
    /// Remove a model from GhostReel's models directory.
    Remove {
        /// Model id.
        id: String,
    },
    /// Print the effective models directory.
    Dir,
    /// Select a model to use for speech (whisper) or vision.
    Use {
        /// Model id (e.g. large-v3-turbo, gemma-3-4b-it, bonsai-27b).
        id: String,
    },
}

#[derive(Subcommand)]
enum ScriptAction {
    /// Import a script JSON file into a project.
    Import {
        file: PathBuf,
        #[arg(long, short)]
        project: String,
        /// Save even if validation reported errors.
        #[arg(long)]
        force: bool,
    },
    /// List scripts in a project.
    List {
        #[arg(long, short)]
        project: String,
        #[arg(long)]
        json: bool,
    },
    /// Measure a saved script: length, voices, cut sentences, silent picture, and one score.
    Score {
        id: i64,
        #[arg(long)]
        json: bool,
    },
    /// Find the interviewer's own lines — questions, prompts, "okay", "perfect" — and mark them
    /// off-mic so no draft can quote them. The acoustic test only catches an interviewer who is
    /// quieter than the subject; this reads what was said. Needs `jev.enabled` and a key.
    Interviewer {
        #[arg(long, short)]
        project: String,
        /// Show what would be flagged without changing anything.
        #[arg(long)]
        dry_run: bool,
        /// Clear the flags this pass set, leaving the acoustic ones alone.
        #[arg(long, conflicts_with = "dry_run")]
        undo: bool,
    },
    /// Build a cut by choosing rather than writing: Jev picks the quotes and the shots out of the
    /// index, and nothing is generated, so no clip can refer to footage that does not exist.
    /// Fast, and limited to what the interviews already say. Needs `jev.enabled` and a key.
    Build {
        #[arg(long, short)]
        project: String,
        /// What the piece is for. Every choice is made against this.
        #[arg(long, short)]
        brief: String,
        #[arg(long, short = 't', default_value_t = 40.0)]
        target_s: f64,
    },
    /// Ask Jev what is wrong with a saved script editorially: do the pictures show what is being
    /// said, does the opening earn attention, does the ending land. Needs `jev.enabled` and a key.
    Judge {
        id: i64,
        /// What the cut was asked for. Judged against it when given.
        #[arg(long)]
        brief: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Display a script draft.
    Show {
        id: i64,
        #[arg(long)]
        json: bool,
    },
    /// Export a script as an OpenTimelineIO (.otio) or Final Cut Pro 7 XML (.xml) timeline.
    Export {
        id: i64,
        #[arg(long, default_value = "fcp_xml")]
        format: String,
        #[arg(long, short = 'o', alias = "output")]
        out: PathBuf,
    },
    /// Render a preview MP4 of a script timeline.
    Preview {
        id: i64,
        #[arg(long, short = 'o', alias = "output")]
        out: Option<PathBuf>,
        #[arg(long)]
        burn_titles: bool,
        #[arg(long)]
        burn_narration: bool,
        /// Bring every clip to the same loudness.
        #[arg(long)]
        normalize_audio: bool,
    },
    /// Chat with the editing agent to draft or refine a script.
    Chat {
        #[arg(long, short)]
        project: String,
        #[arg(long)]
        session: Option<i64>,
        message: String,
    },
    /// List chat sessions in a project.
    Sessions {
        #[arg(long, short)]
        project: String,
    },
}

#[derive(Subcommand)]
enum ProjectAction {
    /// Create a project (sequence settings are used by timeline exports).
    Create {
        name: String,
        #[arg(long, default_value = "")]
        description: String,
        /// Frame rate, e.g. 25, 30, 29.97, 23.976 or 30000/1001.
        #[arg(long, default_value = "25")]
        fps: String,
        #[arg(long, default_value_t = 1920)]
        width: i64,
        #[arg(long, default_value_t = 1080)]
        height: i64,
    },
    List {
        #[arg(long)]
        json: bool,
    },
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Rename a project.
    Rename { name: String, new_name: String },
    /// Take a video out of the project's library (by video id or file name). With --reference the
    /// script chat studies it as a finished edit instead of ignoring it.
    Exclude {
        name: String,
        video: String,
        #[arg(long)]
        reference: bool,
    },
    /// Put an excluded video back into the library.
    Include { name: String, video: String },
    /// Delete a project. Indexed data is kept for reuse unless --purge is given.
    Remove {
        name: String,
        /// Also delete what was indexed: keyframes, preview renders, and the transcripts,
        /// descriptions and vectors of footage no other project uses. Video files are untouched.
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Subcommand)]
enum FolderAction {
    Add {
        path: PathBuf,
        #[arg(long, short)]
        project: String,
        /// Only files directly in the folder.
        #[arg(long)]
        no_recursive: bool,
    },
    List {
        #[arg(long, short)]
        project: Option<String>,
    },
    Remove {
        path: PathBuf,
        #[arg(long, short)]
        project: String,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Run one describe call through the configured CLI agent (claude / agy / opencode) and print
    /// its answer, to check the setup before an index run calls it once per keyframe.
    TestCli {
        /// Which capability's CLI settings to use: `vision` or `chat_model`.
        #[arg(default_value = "vision")]
        capability: String,
    },
    /// Print the config file path.
    Path,
    /// Print the effective configuration (defaults applied).
    Show,
    /// Write the default configuration if no config file exists yet.
    Init,
    /// Set a configuration value.
    Set {
        /// Key to set: vision.* (frame descriptions) or chat_model.* (script chat) with backend,
        /// url, model, local_model, ctx_tokens, kv_cache, flash_attn; plus stt.backend, stt.url,
        /// embed.backend, embed.url, frames.max_interval_s.
        key: String,
        /// Value to assign.
        value: String,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    let paths = Paths::resolve()?;
    match cli.command {
        Command::Doctor { json } => {
            let report = doctor::run(&paths).await;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_report(&report);
            }
            Ok(if report.blockers().is_empty() { ExitCode::SUCCESS } else { ExitCode::from(2) })
        }
        Command::Project { action } => project_cmd(&paths, action),
        Command::Folder { action } => folder_cmd(&paths, action),
        Command::Index { project, watch, retry_failed, json, redo } => {
            index_cmd(&paths, project.as_deref(), watch, retry_failed, json, redo.as_deref()).await
        }
        Command::Status { project, videos, json } => status_cmd(&paths, project.as_deref(), videos, json),
        Command::Transcript { video_id, srt, json } => transcript_cmd(&paths, video_id, srt, json),
        Command::Search { query, project, limit, kind, keywords, json } => {
            search_cmd(&paths, &query, project.as_deref(), limit, kind, keywords, json).await
        }
        Command::Frames { video_id, json } => {
            let rows = index::frames(&open_db(&paths)?, &paths.data_dir, video_id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else if rows.is_empty() {
                println!("no frames for video #{video_id} yet");
            } else {
                for f in rows {
                    println!("[{}] {}", human_duration(f.t_s), f.path.display());
                    if let Some(d) =
                        f.description.as_deref().and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok())
                    {
                        if let Some(err) = d["error"].as_str() {
                            println!("      ✗ {err}");
                        } else {
                            println!("      {}", d["description"].as_str().unwrap_or_default());
                            if let Some(t) = f.visible_text.as_deref().filter(|t| !t.is_empty()) {
                                println!("      on screen: {}", t.replace('\n', " · "));
                            }
                        }
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Mcp => {
            mcp::serve(paths).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Script { action } => script_cmd(&paths, action).await,
        Command::Models { action } => models_cmd(&paths, action).await,
        Command::Config { action } => {
            match action {
                ConfigAction::TestCli { capability } => {
                    let cfg = Config::load(&paths.config_file).unwrap_or_default();
                    let llm = if capability == "chat_model" { cfg.chat_model() } else { cfg.vision.clone() };
                    let agent = ghostreel_core::cliagent::CliAgent::new(llm.cli);
                    println!("{}", agent.self_test(&paths.data_dir).await.map_err(|e| anyhow::anyhow!("{e}"))?);
                }
                ConfigAction::Path => println!("{}", paths.config_file.display()),
                ConfigAction::Show => {
                    let cfg = Config::load(&paths.config_file)?;
                    print!("{}", cfg.to_toml()?);
                }
                ConfigAction::Init => {
                    if paths.config_file.exists() {
                        println!("exists: {}", paths.config_file.display());
                    } else {
                        Config::default().save(&paths.config_file)?;
                        println!("created: {}", paths.config_file.display());
                    }
                }
                ConfigAction::Set { key, value } => {
                    let mut cfg = Config::load(&paths.config_file).unwrap_or_default();
                    cfg.set_key(&key, &value).map_err(|e| anyhow::anyhow!("{e}"))?;
                    cfg.save(&paths.config_file)?;
                    println!("{key} = \"{value}\"");
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn score_meter(n: u8) -> String {
    let filled = n.min(5) as usize;
    let empty = 5 - filled;
    "▰".repeat(filled) + &"▱".repeat(empty)
}

fn print_models_table(models: &[ghostreel_core::models::ModelStatus]) {
    println!("{:<26} {:<17} {:<9} {:<7} {:<9} STATUS", "ID", "KIND", "SIZE", "SPEED", "ACCURACY");
    for m in models {
        let size_str = human_size(m.entry.size_bytes as i64);
        let speed_str = score_meter(m.entry.speed);
        let acc_str = score_meter(m.entry.accuracy);
        let kind_str = m.entry.kind.to_string();
        let status_str = if let Some(path) = &m.installed_path {
            if m.in_own_dir {
                format!("✓ installed ({})", path.display())
            } else {
                format!("✓ found in {}", path.display())
            }
        } else if let Some(part) = m.partial_bytes {
            format!("– partial ({})", human_size(part as i64))
        } else {
            "– not installed".to_string()
        };

        println!("{:<26} {:<17} {:<9} {:<7} {:<9} {}", m.entry.id, kind_str, size_str, speed_str, acc_str, status_str);
    }
}

async fn models_cmd(paths: &Paths, action: ModelsAction) -> anyhow::Result<ExitCode> {
    let config = Config::load(&paths.config_file).unwrap_or_default();
    let models_dir = ghostreel_core::models::effective_models_dir(paths, &config);
    match action {
        ModelsAction::Dir => {
            println!("{}", models_dir.display());
            Ok(ExitCode::SUCCESS)
        }
        ModelsAction::List { json } => {
            let list = ghostreel_core::models::status(&models_dir, &config.models.search_paths);
            if json {
                println!("{}", serde_json::to_string_pretty(&list)?);
                return Ok(ExitCode::SUCCESS);
            }
            print_models_table(&list);
            Ok(ExitCode::SUCCESS)
        }
        ModelsAction::Download { id } => {
            let specs = match ghostreel_core::models::find_entry(&id) {
                Some(entry) => {
                    let mut s = vec![entry.spec()];
                    if let Some(m) = entry.mmproj_spec() {
                        s.push(m);
                    }
                    s
                }
                None => vec![ghostreel_core::models::whisper(&id).map_err(|e| anyhow::anyhow!("{e}"))?],
            };
            let is_tty = std::io::stdout().is_terminal();
            for spec in specs {
                let mut last_pct = 0;
                let dest = ghostreel_core::models::download(&spec, &models_dir, |done, total| {
                    if let Some(total) = total {
                        let pct = (done * 100).checked_div(total).unwrap_or(0);
                        let done_h = human_size(done as i64);
                        let total_h = human_size(total as i64);
                        if is_tty {
                            print!("\rDownloading {}: {} / {} ({}%)   ", spec.file_name, done_h, total_h, pct);
                            let _ = std::io::stdout().flush();
                        } else if pct >= last_pct + 10 || done == total {
                            last_pct = pct;
                            println!("Downloading {}: {} / {} ({}%)", spec.file_name, done_h, total_h, pct);
                        }
                    } else if is_tty {
                        print!("\rDownloading {}: {}   ", spec.file_name, human_size(done as i64));
                        let _ = std::io::stdout().flush();
                    }
                })
                .await?;
                if is_tty {
                    println!();
                }
                println!("Downloaded {} to {}", spec.file_name, dest.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        ModelsAction::Remove { id } => {
            let dest = ghostreel_core::models::remove(&models_dir, &id).map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("Removed {}", dest.display());
            Ok(ExitCode::SUCCESS)
        }
        ModelsAction::Use { id } => {
            if id == ghostreel_core::config::EMBED_MODEL
                || ghostreel_core::models::find_entry(&id)
                    .is_some_and(|e| e.kind == ghostreel_core::models::ModelKind::Embedding)
            {
                bail!(
                    "embedding model is locked to {} to keep search vectors compatible between computers",
                    ghostreel_core::config::EMBED_MODEL
                );
            }
            let mut cfg = Config::load(&paths.config_file).unwrap_or_default();
            if id == "auto" {
                cfg.stt.model = "auto".into();
                cfg.save(&paths.config_file)?;
                println!("Selected speech model: auto");
                return Ok(ExitCode::SUCCESS);
            }
            // Check vision catalog first — vision IDs are explicit catalog entries and must
            // not be matched by the generic whisper() fallback (which accepts any valid name).
            if ghostreel_core::models::vision_pair(&id).is_some()
                || ghostreel_core::models::find_entry(&id)
                    .is_some_and(|e| e.kind == ghostreel_core::models::ModelKind::Vision)
            {
                cfg.vision.local_model = id.clone();
                cfg.save(&paths.config_file)?;
                println!("Selected vision model: {id}");
                return Ok(ExitCode::SUCCESS);
            }
            if ghostreel_core::models::find_entry(&id)
                .is_some_and(|e| e.kind == ghostreel_core::models::ModelKind::Whisper)
                // Custom ggml names are fine once the file is in the models folder.
                || ghostreel_core::models::whisper(&id).is_ok_and(|s| models_dir.join(&s.file_name).is_file())
            {
                cfg.stt.model = id.clone();
                cfg.save(&paths.config_file)?;
                println!("Selected speech model: {id}");
                return Ok(ExitCode::SUCCESS);
            }
            bail!(
                "unknown model '{id}'; use a whisper model (e.g. tiny, small, large-v3-turbo) or vision model (e.g. bonsai-27b, gemma-3-4b-it, qwen2.5-vl-7b, qwen2.5-vl-3b)"
            );
        }
    }
}

/// A video by id ("8") or by file name / path suffix ("NW Hills.mp4").
fn find_video(db: &ghostreel_core::db::Db, key: &str) -> anyhow::Result<i64> {
    if let Ok(id) = key.parse::<i64>() {
        return Ok(id);
    }
    let mut st = db.conn.prepare("SELECT DISTINCT video_id FROM video_files WHERE path LIKE '%' || ?1")?;
    let ids: Vec<i64> = st.query_map([key], |r| r.get(0))?.collect::<Result<_, _>>()?;
    match ids.as_slice() {
        [id] => Ok(*id),
        [] => bail!("no video matches '{key}'"),
        _ => bail!("'{key}' matches {} videos; use the video id", ids.len()),
    }
}

/// "25", "29.97", "23.976", "30000/1001" → (num, den).
fn parse_fps(s: &str) -> anyhow::Result<(i64, i64)> {
    if let Some((n, d)) = s.split_once('/') {
        return Ok((n.trim().parse()?, d.trim().parse()?));
    }
    Ok(match s.trim() {
        "23.976" | "23.98" => (24000, 1001),
        "29.97" => (30000, 1001),
        "59.94" => (60000, 1001),
        other => {
            let f: f64 = other.parse().with_context(|| format!("invalid fps '{other}'"))?;
            if f.fract() != 0.0 {
                bail!("use a fraction for non-integer fps, e.g. 30000/1001");
            }
            (f as i64, 1)
        }
    })
}

fn open_db(paths: &Paths) -> anyhow::Result<Db> {
    Ok(Db::open(&paths.db_file())?)
}

fn project_id(db: &Db, name: Option<&str>) -> anyhow::Result<Option<i64>> {
    Ok(match name {
        Some(n) => Some(db.require_project(n)?.id),
        None => None,
    })
}

fn fps_label(num: i64, den: i64) -> String {
    if den == 1 { num.to_string() } else { format!("{:.3} ({num}/{den})", num as f64 / den as f64) }
}

fn project_cmd(paths: &Paths, action: ProjectAction) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    match action {
        ProjectAction::Create { name, description, fps, width, height } => {
            let (fps_num, fps_den) = parse_fps(&fps)?;
            let p = db.create_project(&NewProject { name, description, fps_num, fps_den, width, height })?;
            println!(
                "created project '{}' ({}x{} @ {} fps)",
                p.name,
                p.width,
                p.height,
                fps_label(p.fps_num, p.fps_den)
            );
        }
        ProjectAction::List { json } => {
            let projects = db.projects()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&projects)?);
            } else if projects.is_empty() {
                println!("no projects — create one with: ghostreel project create <name>");
            } else {
                for p in projects {
                    let st = index::status(&db, Some(p.id))?;
                    println!(
                        "{:<24} {:>3} folders {:>5} videos  {}x{} @ {}",
                        p.name,
                        st.folders,
                        st.videos,
                        p.width,
                        p.height,
                        fps_label(p.fps_num, p.fps_den)
                    );
                }
            }
        }
        ProjectAction::Show { name, json } => {
            let p = db.require_project(&name)?;
            let folders = db.folders(Some(p.id))?;
            let st = index::status(&db, Some(p.id))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({ "project": p, "folders": folders, "status": st })
                    )?
                );
            } else {
                println!("{} — {}x{} @ {} fps", p.name, p.width, p.height, fps_label(p.fps_num, p.fps_den));
                if !p.description.is_empty() {
                    println!("  {}", p.description);
                }
                for f in folders {
                    println!("  folder {}{}", f.path.display(), if f.recursive { "" } else { " (not recursive)" });
                }
                print_status(&st);
            }
        }
        ProjectAction::Exclude { name, video, reference } => {
            let p = db.require_project(&name)?;
            let id = find_video(&db, &video)?;
            let role = if reference { "reference" } else { "removed" };
            db.exclude_video(p.id, id, role)?;
            println!("video #{id} is now '{role}' in project '{}'", p.name);
        }
        ProjectAction::Include { name, video } => {
            let p = db.require_project(&name)?;
            let id = find_video(&db, &video)?;
            db.include_video(p.id, id)?;
            println!("video #{id} is back in project '{}'", p.name);
        }
        ProjectAction::Rename { name, new_name } => {
            let p = db.require_project(&name)?;
            let renamed = db.rename_project(p.id, &new_name)?;
            println!("renamed project '{}' to '{}'", p.name, renamed.name);
        }
        ProjectAction::Remove { name, purge } => {
            let p = db.require_project(&name)?;
            if purge {
                let s = db.purge_project_data(&paths.data_dir, p.id)?;
                println!(
                    "deleted indexed data: {} videos, {} files, {}",
                    s.videos,
                    s.files_deleted,
                    human_size(s.bytes_freed as i64)
                );
            }
            db.remove_project(p.id)?;
            println!("removed project '{}'{}", p.name, if purge { "" } else { " (indexed data kept)" });
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn folder_cmd(paths: &Paths, action: FolderAction) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    match action {
        FolderAction::Add { path, project, no_recursive } => {
            let p = db.require_project(&project)?;
            let f = db.add_folder(p.id, &path, !no_recursive)?;
            println!("'{}' now watches {} — run: ghostreel index -p \"{}\"", p.name, f.path.display(), p.name);
        }
        FolderAction::List { project } => {
            let pid = project_id(&db, project.as_deref())?;
            for f in db.folders(pid)? {
                let exists = if f.path.is_dir() { "" } else { "  (missing)" };
                println!("{}{}", f.path.display(), exists);
            }
        }
        FolderAction::Remove { path, project } => {
            let p = db.require_project(&project)?;
            db.remove_folder(p.id, &path)?;
            println!("'{}' no longer watches {}", p.name, path.display());
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn script_cmd(paths: &Paths, action: ScriptAction) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    match action {
        ScriptAction::Import { file, project, force } => {
            let p = db.require_project(&project)?;
            let content = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read script file '{}'", file.display()))?;
            let mut script = Script::parse_for_project(&content, &p).with_context(|| "failed to parse script JSON")?;
            let snapped = ghostreel_core::script::snap_to_segments(&db, &mut script)?;
            if snapped > 0 {
                println!("snapped {snapped} clip boundary(ies) to speech segments");
            }
            // Same treatment a drafted script gets: a beat that would fall silent when it cuts
            // away keeps the voice running under the pictures.
            let script_cfg = Config::load(&paths.config_file).unwrap_or_default().script;
            for issue in ghostreel_core::chat::lay_audio_beds(&db, &mut script, &script_cfg) {
                println!("  [info] {}", issue.message);
            }
            // And the same guarantee a drafted script gets: nothing stops mid-sentence.
            let mended = ghostreel_core::chat::end_on_sentences(&db, &mut script, &script_cfg);
            if mended > 0 {
                println!("  [info] put {mended} range(s) back on whole sentences");
            }
            let shortened = ghostreel_core::chat::trim_pictures_to_bed(&mut script, &script_cfg);
            if shortened > 0 {
                println!("  [info] cut the pictures back to the voice in {shortened} beat(s)");
            }
            let held = ghostreel_core::chat::hold_the_last_picture(&db, &mut script, &script_cfg);
            if held > 0.0 {
                println!("  [info] held the closing picture for {held:.1} s of quiet");
            }
            let issues = ghostreel_core::script::validate(&db, p.id, &script)?;
            for issue in &issues {
                let tag = match issue.severity {
                    IssueSeverity::Error => "error",
                    IssueSeverity::Warning => "warning",
                    IssueSeverity::Info => "info",
                };
                if let Some(beat) = &issue.beat_id {
                    if let Some(clip) = issue.clip_index {
                        println!("  [{tag}] beat {beat} clip {}: {}", clip + 1, issue.message);
                    } else {
                        println!("  [{tag}] beat {beat}: {}", issue.message);
                    }
                } else {
                    println!("  [{tag}] {}", issue.message);
                }
            }
            let has_errors = issues.iter().any(|i| i.severity == IssueSeverity::Error);
            if has_errors && !force {
                bail!("script validation failed with errors; use --force to save anyway");
            }
            let id = ghostreel_core::script::save_version(&db, p.id, &script, None)?;
            let stored = ghostreel_core::script::load(&db, id)?;
            println!("saved script #{} \"{}\" v{}", stored.id, stored.title, stored.version);
        }
        ScriptAction::List { project, json } => {
            let p = db.require_project(&project)?;
            let list = ghostreel_core::script::list(&db, p.id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&list)?);
            } else if list.is_empty() {
                println!("no scripts in project '{}'", p.name);
            } else {
                for s in list {
                    println!(
                        "#{:<4} {:<24} v{:<3} {:>2} beats {:>2} clips {:>6.1}s",
                        s.id, s.title, s.version, s.beats, s.clips, s.duration_s
                    );
                }
            }
        }
        ScriptAction::Score { id, json } => {
            let stored = ghostreel_core::script::load(&db, id)?;
            let issues = ghostreel_core::script::validate(&db, stored.project_id, &stored.script)?;
            let m = ghostreel_core::chat::measure(&db, None, &stored.script, &issues);
            let sc = ghostreel_core::chat::score(&m);
            if json {
                println!("{}", serde_json::to_string(&serde_json::json!({ "metrics": m, "score": sc }))?);
            } else {
                println!("#{id} \"{}\"  score {:.0}/100", stored.script.title, sc.total);
                println!(
                    "  {:.1} s{}  ·  {} beats, {} clips  ·  {} of {} voices",
                    m.total_s,
                    m.target_s.map(|t| format!(" against {t:.0} s ({:+.0}%)", m.duration_error.unwrap_or(0.0) * 100.0))
                        .unwrap_or_default(),
                    m.beats,
                    m.clips,
                    m.speaking_sources,
                    m.speaking_sources_available
                );
                println!(
                    "  {} cut mid-sentence  ·  {:.0} s of silent picture  ·  {} errors, {} warnings",
                    m.mid_sentence_cuts, m.silent_picture_s, m.errors, m.warnings
                );
                for (part, cost) in &sc.parts {
                    println!("  -{cost:>5.1}  {part}");
                }
            }
        }
        ScriptAction::Interviewer { project, dry_run, undo } => {
            let p = db.require_project(&project)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            if undo {
                let n = ghostreel_core::interviewer::undo(&db, p.id)?;
                println!("cleared {n} line(s) this pass had flagged; the measured ones are untouched");
                return Ok(ExitCode::SUCCESS);
            }
            let verdicts = ghostreel_core::interviewer::find(&db, p.id, &config.jev).await?;

            let found: Vec<_> = verdicts.iter().filter(|v| v.is_interviewer(&config.jev)).collect();
            println!("read {} line(s) still believed to be the subject", verdicts.len());
            for v in &found {
                println!("  #{} {:>7.1}s  p={:.2}  {}", v.video_id, v.start_s, v.p, v.text.trim());
            }
            if dry_run {
                println!("\n{} line(s) would be marked off-mic (nothing changed)", found.len());
            } else {
                let n = ghostreel_core::interviewer::apply(&db, &verdicts, &config.jev)?;
                println!("\nmarked {n} line(s) off-mic");
            }
        }
        ScriptAction::Build { project, brief, target_s } => {
            let project = db.require_project(&project)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();

            // Read the index, then ask: the database must not be held open across the requests.
            let footage = ghostreel_core::chat::build::survey(&db, project.id)?;
            println!(
                "{} quotable line(s) and {} described shot(s) in the index",
                footage.quotes.len(),
                footage.shots.len()
            );
            let script =
                ghostreel_core::chat::build::build(&footage, &project, &brief, target_s, &config.jev).await?;
            println!("chose {} line(s):", script.beats.len());
            for beat in &script.beats {
                println!("  {}", beat.purpose);
            }

            let mut s = script;
            let mut issues =
                ghostreel_core::chat::repair::finish_script(&db, project.id, &mut s, true, &config.script)?
                    .unwrap_or_default();
            issues.retain(|i| !i.message.is_empty());
            for i in &issues {
                println!("  [{:?}] {}", i.severity, i.message);
            }
            let id = ghostreel_core::script::save_version(&db, project.id, &s, None)?;
            println!("saved script #{id} \"{}\" ({:.1} s)", s.title, s.total_duration_s());
        }
        ScriptAction::Judge { id, brief, json } => {
            let stored = ghostreel_core::script::load(&db, id)?;
            let Some(j) =
                ghostreel_core::chat::judge::judge(&db, &stored.script, brief.as_deref(), &Config::load(&paths.config_file).unwrap_or_default().jev).await?
            else {
                eprintln!(
                    "the editorial judge is off. Turn it on with:\n  \
                     ghostreel config set jev.enabled true\n  \
                     ghostreel config set jev.api_key <key>   (or set TYPESAFE_API_KEY)"
                );
                std::process::exit(1);
            };
            if json {
                println!("{}", serde_json::to_string(&j)?);
            } else {
                println!("#{id} \"{}\"  editorial {:.0}/100  ({})", stored.script.title, j.total, j.model);
                for p in &j.parts {
                    println!("  {:>5.1}/{:<4.0}  {:<26} {:.2}", p.earned, p.possible, p.name, p.value);
                }
                if j.unjudged_beats > 0 {
                    println!(
                        "  {} beat(s) not judged: nothing describes what is on screen there — index their frames",
                        j.unjudged_beats
                    );
                }
                for m in &j.mismatched {
                    println!("  pictures do not match the voice in '{}' (p={:.2}): \"{}\"", m.beat_id, m.match_p, m.heard);
                }
                for n in j.notes() {
                    println!("  → {n}");
                }
            }
        }
        ScriptAction::Show { id, json } => {
            let stored = ghostreel_core::script::load(&db, id)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stored)?);
            } else {
                println!(
                    "script #{} \"{}\" v{} ({} beats, {:.1}s)",
                    stored.id,
                    stored.title,
                    stored.version,
                    stored.script.beats.len(),
                    stored.script.total_duration_s()
                );
                for beat in &stored.script.beats {
                    println!("\nbeat [{}] {}:", beat.id, beat.purpose);
                    if let Some(narr) = &beat.narration {
                        println!("  narration: \"{narr}\"");
                    }
                    if let Some(text) = &beat.on_screen_text {
                        println!("  on-screen: \"{text}\"");
                    }
                    if let Some(notes) = &beat.notes {
                        println!("  notes: {notes}");
                    }
                    if let Some(bed) = &beat.bed {
                        println!(
                            "  sound: #{} {:.2}-{:.2} ({:.2}s){}{}",
                            bed.video_id,
                            bed.in_s,
                            bed.out_s,
                            bed.duration_s(),
                            if bed.inferred { " carried under the pictures" } else { "" },
                            bed.why.as_deref().map(|w| format!(" — {w}")).unwrap_or_default()
                        );
                    }
                    for clip in &beat.clips {
                        let dur = (clip.out_s - clip.in_s).max(0.0);
                        let audio = match clip.audio {
                            Audio::Source => "source",
                            Audio::Mute => "mute",
                        };
                        let why = clip.why.as_deref().map(|w| format!(" {w}")).unwrap_or_default();
                        println!(
                            "    #{} {:.2}-{:.2} ({:.2}s) {}{}",
                            clip.video_id, clip.in_s, clip.out_s, dur, audio, why
                        );
                    }
                }
            }
        }
        ScriptAction::Export { id, format, out } => {
            let export_fmt = ExportFormat::from_str(&format)?;
            let res = ghostreel_core::export::export_script(&db, id, export_fmt, &out)?;
            println!("{}", res.path.display());
            match ghostreel_core::export::validate_export(&res.path) {
                Ok(v) => println!("{}", serde_json::to_string(&v)?),
                Err(e) => eprintln!("warning: validation failed: {e}"),
            }
        }
        ScriptAction::Preview { id, out, burn_titles, burn_narration, normalize_audio } => {
            let ffmpeg = doctor::locate("ffmpeg").context("ffmpeg not found")?;
            let opts = ghostreel_core::preview::PreviewOptions {
                burn_titles,
                burn_narration,
                normalize_audio,
                audio_fade_s: Config::load(&paths.config_file).unwrap_or_default().script.audio_fade_s,
                speech_overrun_s: Config::load(&paths.config_file).unwrap_or_default().script.speech_overrun_s,
                out,
                cancel: None,
            };
            let t0 = std::time::Instant::now();
            let is_tty = std::io::stderr().is_terminal();
            let res = ghostreel_core::preview::render_preview(&db, &paths.data_dir, &ffmpeg, id, &opts, |pct, msg| {
                if is_tty {
                    eprint!("\r\x1b[2K[{:>3.0}%] {}", pct * 100.0, msg);
                    let _ = std::io::stderr().flush();
                } else {
                    eprintln!("[{:>3.0}%] {}", pct * 100.0, msg);
                }
            })?;
            if is_tty {
                eprint!("\r\x1b[2K");
            }
            println!(
                "preview: {} ({:.1} s, {} segments, {} built / {} cached, {}) in {:.1} s",
                res.path.display(),
                res.duration_s,
                res.segments,
                res.proxies_built,
                res.proxies_cached,
                res.encoder,
                t0.elapsed().as_secs_f64()
            );
        }
        ScriptAction::Chat { project, session, message } => {
            let p = db.require_project(&project)?;
            let config = Config::load(&paths.config_file)?;
            let vision_setup = runtime::resolve_chat(paths, &config).await;
            eprintln!("Chat model: {}", vision_setup.describe());
            let backend = ghostreel_core::chat::ChatBackend::from_vision_setup(&vision_setup).await?.with_window(config.chat_model().ctx_tokens);

            let mut embedder = None;
            let embed_setup = runtime::resolve_embed(paths, &config).await;
            match runtime::start_embedder(&embed_setup, |_, _| {}).await {
                Ok(e) => embedder = Some(e),
                Err(why) => eprintln!("(meaning search unavailable: {why}; using keywords only)"),
            }

            let mut ctx = ghostreel_core::chat::ChatContext {
                db,
                data_dir: paths.data_dir.clone(),
                backend,
                embedder,
                system_prompt: Some(config.chat.system_prompt.clone()),
                max_tool_rounds: config.chat_model().max_tool_rounds,
                script: config.script.clone(),
                jev: config.jev.clone(),
                cancel: None,
            };

            let t0 = std::time::Instant::now();
            let mut on_event = |event: ghostreel_core::chat::ChatEvent| match event {
                ghostreel_core::chat::ChatEvent::ToolStarted { tool, args } => {
                    println!("→ {tool} {args}");
                }
                ghostreel_core::chat::ChatEvent::ToolFinished { summary, .. } => {
                    println!("  ← {summary}");
                }
                ghostreel_core::chat::ChatEvent::Drafting => {
                    println!("Drafting script…");
                }
                ghostreel_core::chat::ChatEvent::Validating => {
                    println!("Validating…");
                }
            };

            let res = ghostreel_core::chat::run_turn(&mut ctx, p.id, session, &message, &mut on_event).await?;

            println!("\n{}\n", res.reply);
            println!("Session: #{}", res.session_id);
            if let (Some(sid), Some(script)) = (res.script_id, &res.script) {
                let version: i64 = ctx
                    .db
                    .conn
                    .query_row("SELECT version FROM scripts WHERE id = ?1", [sid], |r| r.get(0))
                    .unwrap_or(1);
                println!("Script: #{sid} \"{}\" (v{version})", script.title);
                println!("\n{:<6} {:<14} {:<30} NARRATION", "BEAT", "PURPOSE", "CLIPS");
                for beat in &script.beats {
                    let clips_str = beat
                        .clips
                        .iter()
                        .map(|c| format!("#{} {:.1}–{:.1}s", c.video_id, c.in_s, c.out_s))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let narr = beat.narration.as_deref().unwrap_or("-");
                    println!("{:<6} {:<14} {:<30} {}", beat.id, beat.purpose, clips_str, narr);
                }
                let total_dur = script.total_duration_s();
                let target_str = script.target_duration_s.map(|t| format!(" (target {:.1}s)", t)).unwrap_or_default();
                println!("\nDuration: {:.1}s{}", total_dur, target_str);
            }
            if !res.issues.is_empty() {
                println!("\nIssues:");
                for issue in &res.issues {
                    let tag = match issue.severity {
                        IssueSeverity::Error => "error",
                        IssueSeverity::Warning => "warning",
                        IssueSeverity::Info => "info",
                    };
                    println!("  [{tag}] {}", issue.message);
                }
            }
            println!("Turn completed in {:.1}s", t0.elapsed().as_secs_f64());
        }
        ScriptAction::Sessions { project } => {
            let p = db.require_project(&project)?;
            let list = ghostreel_core::chat::sessions(&db, p.id)?;
            if list.is_empty() {
                println!("no chat sessions in project '{}'", p.name);
            } else {
                println!("{:<6} {:<40} {:<10} UPDATED", "ID", "TITLE", "MESSAGES");
                for s in list {
                    let count: i64 = db
                        .conn
                        .query_row("SELECT count(*) FROM chat_messages WHERE session_id = ?1", [s.id], |r| r.get(0))
                        .unwrap_or(0);
                    let now_ts = ghostreel_core::projects::now();
                    let diff = now_ts.saturating_sub(s.updated_at);
                    let updated = if diff < 60 {
                        format!("{diff}s ago")
                    } else if diff < 3600 {
                        format!("{}m ago", diff / 60)
                    } else if diff < 86400 {
                        format!("{}h ago", diff / 3600)
                    } else {
                        format!("{}d ago", diff / 86400)
                    };
                    let title: String = s.title.chars().take(38).collect();
                    println!("{:<6} {:<40} {:<10} {}", s.id, title, count, updated);
                }
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn index_cmd(
    paths: &Paths,
    project: Option<&str>,
    watch: bool,
    retry_failed: bool,
    json: bool,
    redo: Option<&str>,
) -> anyhow::Result<ExitCode> {
    let mut db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let config = Config::load(&paths.config_file)?;
    let opts = index::Options { project_id: pid, retry_failed, settle_secs: if watch { 10 } else { 0 }, cancel: None };

    // --redo: reset the specified stage (and later stages) back to pending.
    if let Some(stage) = redo {
        let valid = index::STAGES;
        if !valid.contains(&stage) {
            anyhow::bail!("unknown stage '{stage}'; valid values: {}", valid.join(", "));
        }
        index::reset_stages(&db, &paths.data_dir, pid, stage).map_err(|e| anyhow::anyhow!("{e}"))?;
        if !json {
            println!("reset stage '{stage}' (and later stages) to pending");
        }
    }

    let tty = !json && std::io::stderr().is_terminal();
    let mut bar_visible = false;
    let mut print = |e: Event| {
        if json {
            if let Ok(line) = serde_json::to_string(&e) {
                println!("{line}");
            }
            return;
        }
        if bar_visible && !matches!(e, Event::Progress(_) | Event::JobDone { .. } | Event::JobStarted { .. }) {
            eprint!("\r\x1b[2K");
            bar_visible = false;
        }
        match e {
            Event::ScanFolder { path } => println!("scan  {}", path.display()),
            Event::FolderMissing { path } => println!("  ! folder not available (kept index): {}", path.display()),
            Event::Scanned { new, changed, unchanged, removed } => {
                println!("  {new} new, {changed} changed, {unchanged} unchanged, {removed} removed")
            }
            Event::JobStarted { .. } => {}
            Event::StageBackend { stage, backend } => println!("{stage}: {backend}"),
            Event::StageUnavailable { stage, reason } => println!("  ! {stage} postponed: {reason}"),
            Event::DownloadingModel { file } => println!("downloading {file}…"),
            Event::JobDone { video_id, stage } => {
                if !tty {
                    println!("  ✓ {stage} #{video_id}")
                }
            }
            Event::JobFailed { video_id, stage, error } => println!("  ✗ {stage} #{video_id}: {error}"),
            Event::Progress(p) => {
                if tty {
                    eprint!("\r\x1b[2K{}", progress_line(&p));
                    let _ = std::io::stderr().flush();
                    bar_visible = true;
                }
            }
        }
    };

    let mut failed_total = 0;
    loop {
        // Hold the indexer lock only while a run is active, so a long `--watch` doesn't block
        // other projects from indexing while it sits idle.
        let lock = match IndexLock::acquire(&paths.data_dir) {
            Ok(l) => l,
            Err(e @ ghostreel_core::Error::Busy(_)) if watch => {
                if !json {
                    println!("{e}; retrying in 15 s");
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(15)) => continue,
                    _ = shutdown_signal() => break,
                }
            }
            Err(e) => return Err(e.into()),
        };
        // Re-resolved every run: GhostPen/highllama may have started or stopped meanwhile.
        let rt = runtime::resolve(paths, &config).await?;
        let s = index::run(&mut db, &rt, &opts, &mut print).await?;
        drop(lock);
        if tty {
            eprint!("\r\x1b[2K");
        }
        failed_total += s.jobs_failed;
        if !json {
            println!(
                "done: {} jobs ok, {} failed{}",
                s.jobs_done,
                s.jobs_failed,
                if s.unsettled > 0 { format!(", {} still copying", s.unsettled) } else { String::new() }
            );
        }
        if !watch {
            break;
        }
        let folders: Vec<(PathBuf, bool)> = db.folders(pid)?.into_iter().map(|f| (f.path, f.recursive)).collect();
        let mut watcher = FolderWatcher::new(&folders)?;
        if !json {
            println!("watching {} folder(s) — Ctrl+C to stop", folders.len());
        }
        let wait_settle = s.unsettled > 0;
        tokio::select! {
            changed = watcher.changed(Duration::from_secs(3)) => if !changed { break },
            _ = tokio::time::sleep(Duration::from_secs(12)), if wait_settle => {},
            _ = shutdown_signal() => break,
        }
    }
    Ok(if failed_total == 0 { ExitCode::SUCCESS } else { ExitCode::from(3) })
}

/// `[██████░░░░░░]  48%  probe 12/25  · about 2 min left · clip.mp4`
fn progress_line(p: &Progress) -> String {
    const WIDTH: usize = 24;
    let filled = ((p.fraction * WIDTH as f64).round() as usize).min(WIDTH);
    let eta = match p.eta_secs {
        Some(s) if p.fraction < 1.0 => format!(" · {} left", eta_text(s)),
        _ => String::new(),
    };
    let current = p
        .current
        .as_ref()
        .and_then(|c| c.file_name())
        .map(|n| format!(" · {}", n.to_string_lossy()))
        .unwrap_or_default();
    let phase = match p.phase.as_str() {
        "hash" => "reading files",
        "probe" => "video details",
        "download" => "downloading model",
        "transcribe_server" | "transcribe_local" => "transcribing",
        "frames" => "keyframes",
        "download_vision" => "downloading vision model",
        "describe_server" | "describe_local" => "describing frames",
        "download_embed" => "downloading embedding model",
        "embed" => "search index",
        other => other,
    };
    format!(
        "[{}{}] {:>3.0}%  {phase} {}/{}{eta}{current}",
        "█".repeat(filled),
        "░".repeat(WIDTH - filled),
        p.fraction * 100.0,
        p.phase_done,
        p.phase_total
    )
}

/// Ctrl+C, or SIGTERM on Unix (systemd / `kill`).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => return std::future::pending().await,
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn human_size(bytes: i64) -> String {
    let b = bytes as f64;
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else { format!("{:.0} MB", b / 1e6) }
}

fn human_duration(s: f64) -> String {
    let s = s.round() as i64;
    if s >= 3600 { format!("{}h{:02}m", s / 3600, s % 3600 / 60) } else { format!("{}m{:02}s", s / 60, s % 60) }
}

fn print_status(st: &index::Status) {
    println!(
        "  {} videos, {}, {} of footage{}",
        st.videos,
        human_size(st.total_size),
        human_duration(st.total_duration_s),
        if st.vfr_videos > 0 { format!(", {} variable-frame-rate", st.vfr_videos) } else { String::new() }
    );
    for c in &st.stages {
        let skipped = if c.skipped > 0 { format!(", {} skipped", c.skipped) } else { String::new() };
        println!(
            "  {:<10} {} done, {} pending, {} running, {} failed{skipped}",
            c.stage, c.done, c.pending, c.running, c.failed
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn search_cmd(
    paths: &Paths,
    query: &str,
    project: Option<&str>,
    limit: usize,
    kinds: Option<Vec<String>>,
    keywords_only: bool,
    json: bool,
) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let mut embedder = None;
    if !keywords_only {
        let config = Config::load(&paths.config_file)?;
        let setup = runtime::resolve_embed(paths, &config).await;
        match runtime::start_embedder(&setup, |_, _| {}).await {
            Ok(e) => embedder = Some(e),
            Err(why) => eprintln!("(meaning search unavailable: {why}; using keywords only)"),
        }
    }
    let opts = ghostreel_core::search::SearchOptions { project_id: pid, limit, kinds };
    let hits = ghostreel_core::search::search(&db, &paths.data_dir, query, embedder.as_mut(), &opts).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(ExitCode::SUCCESS);
    }
    if hits.is_empty() {
        println!("no matches");
    }
    for (i, h) in hits.iter().enumerate() {
        let name = h.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        println!(
            "{:>2}. {name} @ {}–{}  [{}; {}]",
            i + 1,
            human_duration(h.start_s),
            human_duration(h.end_s),
            h.kinds.join("+"),
            h.matched_by.join("+")
        );
        println!("    {}", h.snippet.replace('\n', " · "));
    }
    Ok(ExitCode::SUCCESS)
}

fn srt_time(s: f64) -> String {
    let ms = (s.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn transcript_cmd(paths: &Paths, video_id: i64, srt: bool, json: bool) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let segments = index::transcript(&db, video_id)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&segments)?);
    } else if srt {
        for (i, s) in segments.iter().enumerate() {
            println!("{}\n{} --> {}\n{}\n", i + 1, srt_time(s.start), srt_time(s.end), s.text);
        }
    } else if segments.is_empty() {
        println!("no transcript for video #{video_id} (not transcribed yet, or no speech)");
    } else {
        for s in segments {
            println!("[{}] {}", human_duration(s.start), s.text);
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn status_cmd(paths: &Paths, project: Option<&str>, videos: bool, json: bool) -> anyhow::Result<ExitCode> {
    let db = open_db(paths)?;
    let pid = project_id(&db, project)?;
    let st = index::status(&db, pid)?;
    let rows = if videos { Some(index::videos(&db, pid)?) } else { None };
    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "status": st, "videos": rows }))?);
        return Ok(ExitCode::SUCCESS);
    }
    println!("{}", project.map(|p| format!("Project {p}")).unwrap_or_else(|| "All projects".into()));
    print_status(&st);
    for v in rows.unwrap_or_default() {
        let dims = match (v.width, v.height) {
            (Some(w), Some(h)) => format!("{w}x{h}"),
            _ => "-".into(),
        };
        let fps = v.fps.map(|f| format!("{f:.2}fps")).unwrap_or_default();
        let dur = v.duration_s.map(human_duration).unwrap_or_else(|| "-".into());
        let flags = format!(
            "{}{}{}{}{}",
            if v.vfr { " VFR" } else { "" },
            if v.has_audio == Some(false) { " no-audio" } else { "" },
            if v.segments > 0 {
                format!(" 📝{}{}", v.segments, v.language.as_deref().map(|l| format!(" {l}")).unwrap_or_default())
            } else {
                String::new()
            },
            if v.frames > 0 { format!(" 🖼{}", v.frames) } else { String::new() },
            if v.copies > 1 { format!(" ×{}", v.copies) } else { String::new() }
        );
        println!("  #{:<4} {:<8} {:>7} {:>9} {:<9}{} {}", v.id, v.status, dur, dims, fps, flags, v.path.display());
        if let Some(e) = v.error {
            println!("        {e}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn mark(ok: bool) -> &'static str {
    if ok { "✓" } else { "✗" }
}

fn print_backend(name: &str, r: &Resolution) {
    let target = match r.target {
        Target::Server => "server",
        Target::Local => "local",
        Target::Unavailable => "UNAVAILABLE",
    };
    let where_ = r
        .probe
        .as_ref()
        .filter(|_| r.target == Target::Server)
        .map(|p| format!(" {} @ {}", p.model.as_deref().unwrap_or("?"), p.url))
        .unwrap_or_default();
    println!(
        "  {} {name:<11} {target}{where_}  [backend = {}] {}",
        mark(r.target != Target::Unavailable),
        r.backend,
        r.reason
    );
}

fn print_report(r: &Report) {
    println!("GhostReel {}", r.version);
    println!(
        "  config  {}{}",
        r.config_file.display(),
        r.config_error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default()
    );
    println!("  data    {}", r.data_dir.display());

    println!("\nDatabase");
    match r.db.ok {
        true => println!(
            "  ✓ {} (schema v{}, sqlite-vec {})",
            r.db.path.display(),
            r.db.schema_version.unwrap_or(0),
            r.db.sqlite_vec.as_deref().unwrap_or("?")
        ),
        false => println!("  ✗ {}: {}", r.db.path.display(), r.db.error.as_deref().unwrap_or("?")),
    }

    println!("\nTools");
    for t in [&r.ffmpeg, &r.ffprobe] {
        match &t.path {
            Some(p) => println!("  ✓ {:<8} {} ({})", t.name, t.version.as_deref().unwrap_or("?"), p.display()),
            None => println!("  ✗ {:<8} not found", t.name),
        }
    }

    println!("\nGPU");
    if r.gpu.is_empty() {
        println!("  - no NVIDIA GPU detected (local models would run on CPU)");
    }
    for g in &r.gpu {
        println!("  ✓ {} — {} / {} MiB used, driver {}", g.name, g.vram_used_mib, g.vram_total_mib, g.driver);
    }

    println!("\nAI backends");
    print_backend("frame descriptions", &r.vision);
    print_backend("script chat", &r.chat);
    println!(
        "  local windows: descriptions {} tokens (kv {}), chat {} tokens (kv {})",
        r.local_runtime.describe_ctx, r.local_runtime.describe_kv, r.local_runtime.chat_ctx, r.local_runtime.chat_kv
    );
    // A server's window is divided among its slots, so `--parallel 4 -c 65536` gives each request
    // 16k — not the 65536 the chat was configured for. Nothing errors when that is wrong; the
    // model just runs out of room mid-script, which is a confusing way to find out.
    if let Some(p) = &r.chat.probe
        && let Some(slot_ctx) = p.caps.slot_ctx
        && r.local_runtime.chat_ctx > slot_ctx
    {
        println!(
            "  ! the chat is set to {} tokens but each of the server's slots has {}: \n    \
             lower chat_model.ctx_tokens, or restart the server with fewer slots or a bigger -c",
            r.local_runtime.chat_ctx, slot_ctx
        );
    }
    print_backend("embeddings", &r.embeddings);
    print_backend("stt", &r.stt);

    println!("\nCLI agents");
    if r.cli_tools.installed.is_empty() {
        println!("  - none installed (claude, agy, opencode not found in PATH)");
    } else {
        for (name, path) in &r.cli_tools.installed {
            println!("  ✓ {:<10} {}", name, path.display());
        }
    }
    if !r.cli_tools.active_for.is_empty() {
        for note in &r.cli_tools.active_for {
            println!("  → using CLI for {note}");
        }
    }

    println!("\nLocal model files");
    for m in &r.models {
        match &m.found {
            Some(p) => println!("  ✓ {:<25} {}", m.role, p.display()),
            None => println!("  - {:<25} {} (not downloaded)", m.role, m.pattern),
        }
    }

    let blockers = r.blockers();
    println!();
    if blockers.is_empty() {
        println!("Ready.");
    } else {
        println!("Blockers:");
        for b in blockers {
            println!("  ✗ {b}");
        }
    }
}
