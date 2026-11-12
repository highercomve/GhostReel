//! GhostReel desktop app. UI logic lives in the React frontend; everything else is
//! `ghostreel-core`, shared with the CLI.

mod agents;
mod media;
mod menu;
mod queue;
mod webui;

use std::path::PathBuf;

use ghostreel_core::config::{Backend, Config, EMBED_MODEL};
use ghostreel_core::db::Db;
use ghostreel_core::doctor::{self, Report};
use ghostreel_core::index::{self, FrameRow, Status, TranscriptSegment, VideoRow};
use ghostreel_core::models;
use ghostreel_core::paths::Paths;
use ghostreel_core::probe::{self, Resolution};
use ghostreel_core::projects::{Folder, NewProject, PipelineConfig, Project};
use ghostreel_core::runtime;
use ghostreel_core::storage;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn paths() -> CmdResult<Paths> {
    Paths::resolve().map_err(err)
}

fn open_db() -> CmdResult<Db> {
    Db::open(&paths()?.db_file()).map_err(err)
}

/// Doctor report + the blockers list the UI shows at the top.
#[derive(Serialize)]
struct DoctorView {
    report: Report,
    blockers: Vec<String>,
}

#[tauri::command]
async fn doctor() -> CmdResult<DoctorView> {
    let report = doctor::run(&paths()?).await;
    let blockers = report.blockers();
    Ok(DoctorView { report, blockers })
}

#[derive(Serialize)]
struct ProjectSummary {
    project: Project,
    status: Status,
}

#[tauri::command]
fn list_projects() -> CmdResult<Vec<ProjectSummary>> {
    let db = open_db()?;
    db.projects()
        .map_err(err)?
        .into_iter()
        .map(|project| Ok(ProjectSummary { status: index::status(&db, Some(project.id)).map_err(err)?, project }))
        .collect()
}

#[tauri::command]
fn create_project(
    name: String,
    fps_num: i64,
    fps_den: i64,
    width: i64,
    height: i64,
    pipeline: Option<PipelineConfig>,
) -> CmdResult<Project> {
    open_db()?
        .create_project(&NewProject { name, description: String::new(), fps_num, fps_den, width, height, pipeline })
        .map_err(err)
}

#[tauri::command]
fn set_project_pipeline(project_id: i64, pipeline: PipelineConfig) -> CmdResult<Project> {
    open_db()?.update_project_pipeline(project_id, &pipeline).map_err(err)
}

#[tauri::command]
fn get_project_glossary(project_id: i64) -> CmdResult<ghostreel_core::glossary::Glossary> {
    let db = open_db()?;
    let p = db.project(project_id).map_err(err)?;
    Ok(p.glossary)
}

#[tauri::command]
fn set_project_glossary(
    project_id: i64,
    glossary: ghostreel_core::glossary::Glossary,
) -> CmdResult<ghostreel_core::glossary::Glossary> {
    let mut db = open_db()?;
    db.set_project_glossary(project_id, &glossary).map_err(err)
}

#[tauri::command]
fn rename_project(project_id: i64, name: String) -> CmdResult<Project> {
    open_db()?.rename_project(project_id, &name).map_err(err)
}

/// `purge`: also delete keyframes, preview renders and the transcripts/descriptions/vectors of
/// footage no other project uses. Video files are never touched.
#[tauri::command]
fn remove_project(project_id: i64, purge: bool) -> CmdResult<ghostreel_core::projects::PurgeStats> {
    let p = paths()?;
    let mut db = open_db()?;
    let stats = if purge { db.purge_project_data(&p.data_dir, project_id).map_err(err)? } else { Default::default() };
    db.remove_project(project_id).map_err(err)?;
    Ok(stats)
}

/// How much space GhostReel uses on disk, overall and per project.
#[tauri::command]
async fn storage_report() -> CmdResult<storage::Report> {
    tokio::task::spawn_blocking(|| storage::report(&open_db()?, &paths()?.data_dir).map_err(err)).await.map_err(err)?
}

/// `scope`: `all`, `project` (with `project_id`) or `unused` (ignores `kinds`). `kinds`: proxies,
/// keyframes, previews, chat_images, exports, temp. Video files are never touched.
#[tauri::command]
async fn clean_storage(scope: String, project_id: Option<i64>, kinds: Vec<String>) -> CmdResult<storage::CleanStats> {
    tokio::task::spawn_blocking(move || {
        let scope = match (scope.as_str(), project_id) {
            ("all", _) => storage::Scope::All,
            ("unused", _) => storage::Scope::Unused,
            ("project", Some(id)) => storage::Scope::Project(id),
            _ => return Err(format!("unknown scope '{scope}'")),
        };
        let kinds = kinds.iter().map(|k| storage::Kind::parse(k)).collect::<Result<Vec<_>, _>>().map_err(err)?;
        storage::clean(&mut open_db()?, &paths()?.data_dir, scope, &kinds).map_err(err)
    })
    .await
    .map_err(err)?
}

/// Shrink the library database after deletions. Returns the bytes saved.
#[tauri::command]
async fn compact_database() -> CmdResult<u64> {
    tokio::task::spawn_blocking(|| {
        let p = paths()?;
        storage::compact(&open_db()?, &p.db_file()).map_err(err)
    })
    .await
    .map_err(err)?
}

#[derive(Serialize)]
struct ProjectView {
    project: Project,
    folders: Vec<FolderView>,
    status: Status,
    videos: Vec<VideoRow>,
    /// Videos taken out of the library: removed, or kept as reference edits for the script chat.
    excluded: Vec<ExcludedView>,
    face_models_available: bool,
    /// Videos with speech whose voices are not analysed yet (speakers stage).
    speakers_pending: usize,
    /// People found in the project so far.
    people_count: usize,
    /// Each pipeline stage's status for the Project menu (off / unavailable / pending / failed /
    /// done), from the same gate the index pipeline uses.
    stages: Vec<index::StageStatus>,
}

#[derive(Serialize)]
struct ExcludedView {
    video_id: i64,
    role: String,
    path: String,
}

#[derive(Serialize)]
struct FolderView {
    #[serde(flatten)]
    folder: Folder,
    available: bool,
}

#[tauri::command]
fn project_view(project_id: i64) -> CmdResult<ProjectView> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let face_models_available = ghostreel_core::models::face_models_available(&p, &config);
    let db = open_db()?;
    let folders = db
        .folders(Some(project_id))
        .map_err(err)?
        .into_iter()
        .map(|f| FolderView { available: f.path.is_dir(), folder: f })
        .collect();
    Ok(ProjectView {
        project: db.project(project_id).map_err(err)?,
        folders,
        status: index::status(&db, Some(project_id)).map_err(err)?,
        videos: index::videos_with_shake(&db, Some(project_id), shake_limit(), shake_relative(), max_sway())
            .map_err(err)?,
        excluded: db
            .excluded_videos(project_id)
            .map_err(err)?
            .into_iter()
            .map(|(video_id, role, path)| ExcludedView { video_id, role, path })
            .collect(),
        face_models_available,
        speakers_pending: ghostreel_core::speakers::pending_videos(&db, project_id).map_err(err)?,
        people_count: db.people(project_id).map_err(err)?.len(),
        stages: index::stage_statuses(&db, project_id, &config, face_models_available).map_err(err)?,
    })
}

/// `role`: `removed` (ignored) or `reference` (a finished edit the script chat learns from).
#[tauri::command]
fn exclude_video(project_id: i64, video_id: i64, role: String) -> CmdResult<()> {
    open_db()?.exclude_video(project_id, video_id, &role).map_err(err)
}

#[tauri::command]
fn include_video(project_id: i64, video_id: i64) -> CmdResult<()> {
    open_db()?.include_video(project_id, video_id).map_err(err)
}

#[tauri::command]
fn set_video_category(video_id: i64, category: String) -> CmdResult<()> {
    let db = open_db()?;
    let tag = if category == "broll" || category == "audio" {
        db.video_category(video_id).ok().and_then(|(_, _, _, t)| t)
    } else {
        None
    };
    db.set_video_category(video_id, &category, 1.0, "manual", tag.as_deref()).map_err(err)
}

#[tauri::command]
fn plan_organize_project(project_id: i64) -> CmdResult<ghostreel_core::organize::OrganizePlan> {
    let db = open_db()?;
    ghostreel_core::organize::plan_organize_project(&db, project_id).map_err(err)
}

#[tauri::command]
async fn execute_organize_project(project_id: i64) -> CmdResult<ghostreel_core::organize::OrganizeReport> {
    tokio::task::spawn_blocking(move || {
        let mut db = open_db()?;
        ghostreel_core::organize::execute_organize_project(&mut db, project_id).map_err(err)
    })
    .await
    .map_err(err)?
}

#[tauri::command]
fn add_folder(app: AppHandle, project_id: i64, path: PathBuf, recursive: bool) -> CmdResult<Folder> {
    let folder = open_db()?.add_folder(project_id, &path, recursive).map_err(err)?;
    allow_media_dir(&app, &folder.path);
    Ok(folder)
}

/// Let the webview load videos from a watched folder (player).
fn allow_media_dir(app: &AppHandle, dir: &std::path::Path) {
    use tauri::Manager;
    let _ = app.asset_protocol_scope().allow_directory(dir, true);
}

/// Cached embedder for search (a local helper takes ~1 s to start; reuse it across queries).
#[derive(Default)]
struct SearchState(tokio::sync::Mutex<Option<ghostreel_core::embed::Embedder>>);

#[tauri::command]
async fn search(
    state: State<'_, SearchState>,
    project_id: Option<i64>,
    query: String,
    folder_id: Option<i64>,
    video_ids: Option<Vec<i64>>,
    limit: Option<usize>,
) -> CmdResult<SearchView> {
    let p = paths()?;
    let mut cached = state.0.lock().await;
    let mut note = None;
    if cached.is_none() {
        let config = Config::load(&p.config_file).map_err(err)?;
        let setup = runtime::resolve_embed(&p, &config).await;
        match runtime::start_embedder(&setup, |_, _| {}).await {
            Ok(e) => *cached = Some(e),
            Err(why) => note = Some(format!("Keyword search only ({why})")),
        }
    }
    let opts = ghostreel_core::search::SearchOptions {
        project_id,
        folder_id,
        video_ids,
        limit: limit.unwrap_or(30),
        kinds: None,
    };
    let vector = match cached.as_mut() {
        Some(e) => match ghostreel_core::search::query_vector(e, &query).await {
            Ok(v) => Some(v),
            Err(e) => {
                // A dead server/helper: drop the cache and fall back to keywords for this query.
                *cached = None;
                note = Some(format!("Keyword search only ({e})"));
                None
            }
        },
        None => None,
    };
    drop(cached);
    let db = Db::open(&p.db_file()).map_err(err)?;
    let hits =
        ghostreel_core::search::search_with_vector(&db, &p.data_dir, &query, vector.as_deref(), &opts).map_err(err)?;
    Ok(SearchView { hits, note })
}

/// Base URL of the local media server (append `?path=<url-encoded absolute path>`).
#[tauri::command]
async fn media_base(server: State<'_, tokio::sync::OnceCell<media::MediaServer>>) -> CmdResult<String> {
    let s = server.get_or_try_init(media::start).await?;
    Ok(s.base.clone())
}

#[derive(Serialize)]
struct SearchView {
    hits: Vec<ghostreel_core::search::Hit>,
    note: Option<String>,
}

/// The models the chosen coding-agent CLI will accept, asked of the tool.
///
/// Empty when the tool has no way to say (claude has no such command, codex wants a terminal) or
/// when it is not installed — the page then leaves the field as free text.
#[tauri::command]
async fn cli_models(tool: String) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let cfg = ghostreel_core::config::CliAgentConfig { tool, ..Default::default() };
        ghostreel_core::cliagent::list_models(&cfg)
    })
    .await
    .map_err(|e| e.to_string())
}

/// Open a video in the system player at `t` seconds (mpv/VLC when installed, else the default app).
#[tauri::command]
fn open_external(app: AppHandle, path: PathBuf, t: Option<f64>) -> CmdResult<()> {
    let t = t.unwrap_or(0.0).max(0.0);
    if let Some(mpv) = doctor::locate("mpv") {
        std::process::Command::new(mpv).arg(format!("--start={t:.1}")).arg(&path).spawn().map_err(err)?;
        return Ok(());
    }
    if let Some(vlc) = doctor::locate("vlc") {
        std::process::Command::new(vlc).arg(format!("--start-time={t:.1}")).arg(&path).spawn().map_err(err)?;
        return Ok(());
    }
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(path.to_string_lossy(), None::<&str>).map_err(err)
}

#[tauri::command]
fn remove_folder(project_id: i64, path: PathBuf) -> CmdResult<()> {
    open_db()?.remove_folder(project_id, &path).map_err(err)
}

#[tauri::command]
async fn enqueue_index(app: AppHandle, queue: State<'_, queue::Queue>, project_id: i64) -> CmdResult<u64> {
    let name = open_db()?.project(project_id).map_err(err)?.name;
    Ok(queue.enqueue(&app, queue::TaskKind::Index { project_id }, format!("Index “{name}”")).await)
}

#[tauri::command]
async fn enqueue_steadiness(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    force: Option<bool>,
) -> CmdResult<u64> {
    let name = open_db()?.project(project_id).map_err(err)?.name;
    let force = force.unwrap_or(false);
    let label = if force {
        format!("Re-analyze camera shake “{name}”")
    } else {
        format!("Analyze camera shake “{name}”")
    };
    Ok(queue.enqueue(&app, queue::TaskKind::Steadiness { project_id, force }, label).await)
}

#[tauri::command]
async fn enqueue_faces(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    force: Option<bool>,
) -> CmdResult<u64> {
    let name = open_db()?.project(project_id).map_err(err)?.name;
    let force = force.unwrap_or(false);
    let label = if force {
        format!("Re-analyze faces “{name}”")
    } else {
        format!("Analyze faces “{name}”")
    };
    Ok(queue.enqueue(&app, queue::TaskKind::Faces { project_id, force }, label).await)
}

#[tauri::command]
async fn enqueue_speakers(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    force: Option<bool>,
) -> CmdResult<u64> {
    let name = open_db()?.project(project_id).map_err(err)?.name;
    let force = force.unwrap_or(false);
    let label = if force { format!("Re-analyze speakers “{name}”") } else { format!("Analyze speakers “{name}”") };
    Ok(queue.enqueue(&app, queue::TaskKind::Speakers { project_id, force }, label).await)
}

#[tauri::command]
fn project_people(project_id: i64) -> CmdResult<Vec<ghostreel_core::speakers::PersonSummary>> {
    open_db()?.person_summaries(project_id).map_err(err)
}

#[tauri::command]
fn set_person(
    project_id: i64,
    person_id: i64,
    name: Option<String>,
    handle: Option<String>,
    role: Option<String>,
) -> CmdResult<ghostreel_core::speakers::Person> {
    open_db()?
        .set_person(project_id, person_id, name.as_deref(), handle.as_deref(), role.as_deref())
        .map_err(err)
}

#[tauri::command]
fn merge_people(project_id: i64, keep_id: i64, merge_id: i64) -> CmdResult<ghostreel_core::speakers::Person> {
    open_db()?.merge_people(project_id, keep_id, merge_id).map_err(err)
}

#[tauri::command]
fn video_speakers(
    video_id: i64,
    from_s: Option<f64>,
    to_s: Option<f64>,
) -> CmdResult<Vec<ghostreel_core::speakers::SpeakerTurn>> {
    open_db()?.speaker_turns(video_id, from_s.unwrap_or(0.0), to_s.unwrap_or(f64::MAX)).map_err(err)
}

/// Path of a cached face thumbnail for a person (cropped from the video on first request).
#[tauri::command]
async fn person_thumbnail(project_id: i64, person_id: i64) -> CmdResult<Option<String>> {
    tokio::task::spawn_blocking(move || {
        let p = paths()?;
        let db = open_db()?;
        db.person(project_id, person_id).map_err(err)?;
        let ffmpeg = doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
        Ok(ghostreel_core::speakers::person_thumbnail(&db, &p.data_dir, &ffmpeg, person_id)
            .map_err(err)?
            .map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .map_err(err)?
}

/// Reset a stage (and later stages) back to pending for a project's videos, then enqueue an index run.
/// `stage` must be one of: frames, transcribe, describe, embed.
#[tauri::command]
async fn redo_project_stage(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    stage: String,
) -> CmdResult<u64> {
    // Project-level analysis stages re-run in place rather than through the job pipeline.
    match stage.as_str() {
        "speakers" => return enqueue_speakers(app, queue, project_id, Some(true)).await,
        "faces" => return enqueue_faces(app, queue, project_id, Some(true)).await,
        _ => {}
    }
    let p = paths()?;
    let db = open_db()?;
    index::reset_stages(&db, &p.data_dir, Some(project_id), &stage).map_err(err)?;
    let name = db.project(project_id).map_err(err)?.name;
    let label = format!("Rebuild keyframes — “{name}”");
    Ok(queue.enqueue(&app, queue::TaskKind::Index { project_id }, label).await)
}

#[derive(Serialize)]
struct ChatSettingsView {
    /// What the user saved; empty = default in use.
    system_prompt: String,
    default_system_prompt: &'static str,
}

#[tauri::command]
fn get_chat_settings() -> CmdResult<ChatSettingsView> {
    let config = Config::load(&paths()?.config_file).map_err(err)?;
    Ok(ChatSettingsView {
        system_prompt: config.chat.system_prompt,
        default_system_prompt: ghostreel_core::chat::DEFAULT_EDITOR_PROMPT,
    })
}

/// Save the chat system prompt; an empty prompt (or the default text) restores the default.
#[tauri::command]
fn set_chat_system_prompt(prompt: String) -> CmdResult<ChatSettingsView> {
    let p = paths()?;
    let mut config = Config::load(&p.config_file).map_err(err)?;
    let trimmed = prompt.trim();
    config.chat.system_prompt =
        if trimmed == ghostreel_core::chat::DEFAULT_EDITOR_PROMPT.trim() { String::new() } else { trimmed.to_string() };
    config.save(&p.config_file).map_err(err)?;
    get_chat_settings()
}

#[tauri::command]
async fn enqueue_preview(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    script_id: i64,
    burn_titles: bool,
    burn_narration: bool,
    burn_captions: bool,
    normalize_audio: bool,
    out: Option<String>,
) -> CmdResult<u64> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let label = match &out {
        Some(_) => format!("Export MP4 “{}” v{}", stored.title, stored.version),
        None => format!("Preview “{}” v{}", stored.title, stored.version),
    };
    Ok(queue
        .enqueue(
            &app,
            queue::TaskKind::RenderPreview { script_id, burn_titles, burn_narration, burn_captions, normalize_audio, out },
            label,
        )
        .await)
}

#[tauri::command]
async fn enqueue_export(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    script_id: i64,
    format: String,
    path: String,
) -> CmdResult<u64> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let label = format!("Export “{}” → {}", stored.title, format);
    Ok(queue.enqueue(&app, queue::TaskKind::Export { script_id, format, path }, label).await)
}

#[tauri::command]
fn preview_plan(script_id: i64) -> CmdResult<Vec<ghostreel_core::preview::PlannedSegment>> {
    let db = open_db()?;
    ghostreel_core::preview::preview_plan(&db, script_id).map_err(err)
}

#[tauri::command]
fn get_script_preview(script_id: i64) -> CmdResult<Option<String>> {
    let db = open_db()?;
    let p = Paths::resolve().map_err(err)?;
    let found = ghostreel_core::preview::find_preview(&db, &p.data_dir, script_id).map_err(err)?;
    Ok(found.map(|p| p.to_string_lossy().to_string()))
}

#[tauri::command]
async fn queue_list(queue: State<'_, queue::Queue>) -> CmdResult<Vec<queue::Task>> {
    Ok(queue.snapshot().await)
}

/// Saved runs (finished tasks of the app's queue and of the CLI), newest first: one project's, or
/// every project's when `project_id` is absent.
#[tauri::command]
fn list_runs(project_id: Option<i64>, limit: Option<usize>) -> CmdResult<Vec<ghostreel_core::runs::Run>> {
    let db = open_db()?;
    let limit = limit.unwrap_or(100).min(ghostreel_core::runs::RUNS_KEPT);
    match project_id {
        Some(id) => db.runs_for_project(id, limit),
        None => db.recent_runs(limit),
    }
    .map_err(err)
}

/// For each run link, where its target stands: there, in the trash (restorable), or gone.
#[tauri::command]
fn run_links_state(links: Vec<ghostreel_core::runs::RunLink>) -> CmdResult<Vec<ghostreel_core::runs::LinkState>> {
    let db = open_db()?;
    ghostreel_core::runs::links_state(&db, &links).map_err(err)
}

/// What is in a project's trash (deleted cuts, versions and clip sets), newest first.
#[tauri::command]
fn list_trash(project_id: i64) -> CmdResult<Vec<ghostreel_core::trash::TrashItem>> {
    let db = open_db()?;
    ghostreel_core::trash::list(&db, Some(project_id)).map_err(err)
}

/// Put a trash entry back exactly as it was.
#[tauri::command]
fn restore_trash(trash_id: String) -> CmdResult<ghostreel_core::trash::TrashItem> {
    let db = open_db()?;
    ghostreel_core::trash::restore(&db, &trash_id, "app").map_err(err)
}

/// Restore whatever a run link points at from the trash (a run's deleted clip set or cut).
#[tauri::command]
fn restore_run_link(link: ghostreel_core::runs::RunLink) -> CmdResult<ghostreel_core::trash::TrashItem> {
    let db = open_db()?;
    ghostreel_core::trash::restore_link(&db, &link, "app").map_err(err)
}

#[tauri::command]
async fn cancel_task(app: AppHandle, queue: State<'_, queue::Queue>, id: u64) -> CmdResult<bool> {
    Ok(queue.cancel(&app, id).await)
}

#[tauri::command]
async fn clear_finished_tasks(app: AppHandle, queue: State<'_, queue::Queue>) -> CmdResult<()> {
    queue.clear_finished(&app).await;
    Ok(())
}

#[derive(Serialize)]
struct ModelsStatusView {
    dir: PathBuf,
    models: Vec<ghostreel_core::models::ModelStatus>,
    current_whisper_model: String,
    current_vision_model: String,
}

#[tauri::command]
fn models_status() -> CmdResult<ModelsStatusView> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    let models = ghostreel_core::models::status(&dir, &config.models.search_paths);
    let current_whisper_model = config.stt.model;
    let current_vision_model = config.vision.local_model;
    Ok(ModelsStatusView { dir, models, current_whisper_model, current_vision_model })
}

#[derive(Serialize, Deserialize)]
struct VisionSettingsView {
    backend: String,
    url: String,
    model: String,
    local_model: String,
    api_key_set: bool,
    ctx_tokens: u32,
    kv_cache: String,
    flash_attn: String,
    cpu_moe_layers: u32,
    moe_blocks: Option<u32>,
    max_tool_rounds: u32,
    think: bool,
    /// Frames described at once against a server. Meaningless without slots, which is why the
    /// settings page shows it only when the server reports them.
    describe_concurrency: u32,
    cli: CliSettingsView,
}

/// A coding-agent CLI (claude / agy / opencode) used instead of a model.
#[derive(Serialize, Deserialize)]
struct CliSettingsView {
    tool: String,
    command: String,
    model: String,
    timeout_secs: u64,
    concurrency: usize,
    /// Whether the binary was found on this machine.
    installed: bool,
}

#[derive(Serialize, Deserialize)]
struct SttSettingsView {
    backend: String,
    url: String,
    model: String,
}

#[derive(Serialize, Deserialize)]
struct EmbedSettingsView {
    backend: String,
    url: String,
    model: String,
}

#[derive(Serialize, Deserialize)]
struct AiSettingsView {
    /// Frame descriptions (indexing).
    vision: VisionSettingsView,
    /// Script chat: its own model settings, with a bigger context window by default.
    chat_model: VisionSettingsView,
    stt: SttSettingsView,
    embed: EmbedSettingsView,
    frames: FrameSettingsView,
    jev: JevSettingsView,
    /// What the frame-description server admits it can do. The settings page offers a control
    /// only when the server behind it can honour it — a slider that silently does nothing on LM
    /// Studio is worse than no slider.
    vision_caps: ServerCapsView,
}

/// What one server admits it can do, for the settings page to gate on.
#[derive(Serialize, Deserialize)]
struct ServerCapsView {
    /// Requests the server really works on at once. `None` = it did not say (LM Studio, Ollama,
    /// anything not llama.cpp), which is not the same as one.
    slots: Option<u32>,
    slot_ctx: Option<u32>,
    /// llama.cpp router: models and their flags can be changed without a restart.
    router: bool,
}

/// The editorial judge. `has_key` rather than the key itself: a secret that has been typed in
/// does not need to travel back to the window that typed it, and the settings page only has to
/// show whether there is one.
#[derive(Serialize, Deserialize)]
struct JevSettingsView {
    enabled: bool,
    /// Whether every finished cut is read editorially. Separate from `enabled`, which also gates
    /// choosing — building a cut with Jev and having it score every draft are different wants.
    judge: bool,
    has_key: bool,
    /// The key is in the environment rather than the config file, so the field is not editable.
    key_from_env: bool,
    model: String,
}

#[derive(Deserialize, Default)]
struct JevSettingsPatch {
    enabled: Option<bool>,
    judge: Option<bool>,
    api_key: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize, Default)]
struct VisionSettingsPatch {
    backend: Option<String>,
    describe_concurrency: Option<u32>,
    url: Option<String>,
    model: Option<String>,
    local_model: Option<String>,
    api_key: Option<String>,
    ctx_tokens: Option<u32>,
    kv_cache: Option<String>,
    flash_attn: Option<String>,
    cpu_moe_layers: Option<u32>,
    max_tool_rounds: Option<u32>,
    think: Option<bool>,
    cli: Option<CliSettingsPatch>,
}

#[derive(Deserialize, Default)]
struct CliSettingsPatch {
    tool: Option<String>,
    command: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
    concurrency: Option<usize>,
}

#[derive(Deserialize, Default)]
struct SttSettingsPatch {
    backend: Option<String>,
    url: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize, Default)]
struct EmbedSettingsPatch {
    backend: Option<String>,
    url: Option<String>,
    model: Option<String>,
}

#[derive(Deserialize, Default)]
struct AiSettingsPatch {
    vision: Option<VisionSettingsPatch>,
    chat_model: Option<VisionSettingsPatch>,
    stt: Option<SttSettingsPatch>,
    embed: Option<EmbedSettingsPatch>,
    frames: Option<FrameSettingsPatch>,
    jev: Option<JevSettingsPatch>,
}

#[derive(Serialize, Deserialize)]
struct FrameSettingsView {
    max_interval_s: f64,
    long_side: u32,
}

#[derive(Deserialize, Default)]
struct FrameSettingsPatch {
    max_interval_s: Option<f64>,
    long_side: Option<u32>,
}

#[derive(Serialize)]
struct BackendsResolutionView {
    vision: Resolution,
    chat: Resolution,
    embeddings: Resolution,
    stt: Resolution,
}

fn llm_view(c: &ghostreel_core::config::VisionConfig) -> VisionSettingsView {
    VisionSettingsView {
        backend: c.backend.to_string(),
        url: c.url.clone(),
        model: c.model.clone(),
        local_model: c.local_model.clone(),
        api_key_set: !c.api_key.trim().is_empty(),
        ctx_tokens: c.ctx_tokens,
        kv_cache: c.kv_cache.clone(),
        flash_attn: c.flash_attn.clone(),
        cpu_moe_layers: c.cpu_moe_layers,
        moe_blocks: ghostreel_core::models::moe_blocks(&c.local_model),
        max_tool_rounds: c.max_tool_rounds,
        think: c.think,
        describe_concurrency: c.describe_concurrency,
        cli: CliSettingsView {
            tool: c.cli.tool.clone(),
            command: c.cli.command.clone(),
            model: c.cli.model.clone(),
            timeout_secs: c.cli.timeout_secs,
            concurrency: c.cli.concurrency,
            installed: ghostreel_core::cliagent::CliAgent::new(c.cli.clone()).available().is_some(),
        },
    }
}

/// Apply a patch to one capability's model settings.
fn apply_llm_patch(
    cfg: &mut ghostreel_core::config::VisionConfig,
    v: VisionSettingsPatch,
    section: &str,
    parse_backend: impl Fn(&str) -> CmdResult<Backend>,
) -> CmdResult<()> {
    if let Some(b) = v.backend {
        cfg.backend = parse_backend(&b)?;
        // The tool defaults to empty, but the picker has no empty entry and so renders its
        // first option: without this the UI would claim `claude` is selected while the
        // config says nothing is, and the run would fail with "unknown CLI tool ''".
        if cfg.backend == Backend::Cli && cfg.cli.tool.is_empty() {
            cfg.cli.tool = ghostreel_core::config::CLI_TOOLS[0].to_string();
        }
    }
    if let Some(url) = v.url {
        cfg.url = url;
    }
    if let Some(model) = v.model {
        cfg.model = model;
    }
    if let Some(lm) = v.local_model {
        if !models::external_model(&lm).is_some_and(|m| m.entry.kind == models::ModelKind::Vision)
            && models::vision_pair(&lm).is_none()
            && !models::find_entry(&lm).is_some_and(|e| e.kind == models::ModelKind::Vision)
        {
            return Err(format!("unknown local vision model '{lm}'"));
        }
        cfg.local_model = lm;
    }
    if let Some(key) = v.api_key {
        cfg.api_key = key;
    }
    if let Some(ctx) = v.ctx_tokens {
        cfg.ctx_tokens = ctx;
    }
    if let Some(kv) = v.kv_cache {
        cfg.kv_cache = kv;
    }
    if let Some(fa) = v.flash_attn {
        cfg.flash_attn = fa;
    }
    if let Some(layers) = v.cpu_moe_layers {
        cfg.cpu_moe_layers = layers;
    }
    if let Some(rounds) = v.max_tool_rounds {
        cfg.max_tool_rounds = rounds;
    }
    if let Some(t) = v.think {
        cfg.think = t;
    }
    if let Some(n) = v.describe_concurrency {
        cfg.describe_concurrency = n.clamp(1, 16);
    }
    if let Some(c) = v.cli {
        if let Some(t) = c.tool {
            cfg.cli.tool = t;
        }
        if let Some(cmd) = c.command {
            cfg.cli.command = cmd;
        }
        if let Some(m) = c.model {
            cfg.cli.model = m;
        }
        if let Some(t) = c.timeout_secs {
            cfg.cli.timeout_secs = t;
        }
        if let Some(n) = c.concurrency {
            cfg.cli.concurrency = n;
        }
    }
    cfg.validate(section)
}

/// Probe the frame-description server for what it supports. Best-effort and short-timeout: the
/// settings page must open whether or not a server is up.
async fn vision_caps_view(config: &Config) -> ServerCapsView {
    use ghostreel_core::config::Backend;
    if !matches!(config.vision.backend, Backend::Auto | Backend::Server) {
        return ServerCapsView { slots: None, slot_ctx: None, router: false };
    }
    let client = ghostreel_core::probe::probe_client();
    let p = ghostreel_core::probe::vision(&client, &config.vision.url, &config.vision.model).await;
    ServerCapsView { slots: p.caps.slots, slot_ctx: p.caps.slot_ctx, router: p.caps.router }
}

fn jev_view(cfg: &ghostreel_core::config::JevConfig) -> JevSettingsView {
    let env_key = std::env::var("TYPESAFE_API_KEY").ok().filter(|k| !k.trim().is_empty());
    JevSettingsView {
        enabled: cfg.enabled,
        judge: cfg.judge,
        has_key: env_key.is_some() || !cfg.api_key.trim().is_empty(),
        key_from_env: env_key.is_some(),
        model: cfg.model.clone(),
    }
}

/// Build a cut by choosing rather than writing: Jev picks the quotes and the shots out of the
/// index and code assembles them. Not a chat turn — there is no conversation to have with a model
/// that answers in probabilities — so it runs here rather than through the queue: it takes about
/// as long as a preview, and there is nothing to stream.
#[tauri::command]
async fn build_script_with_jev(
    project_id: i64,
    session_id: Option<i64>,
    brief: String,
    target_s: f64,
) -> CmdResult<BuiltCut> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    if config.jev.api_key.trim().is_empty() && std::env::var("TYPESAFE_API_KEY").is_err() {
        return Err("building this way needs a Jev API key — add one in Settings".into());
    }

    // Read the index and then ask: the database is not `Sync`, so it must not be held open
    // across the requests. Both halves run off the UI thread.
    let db_path = p.db_file();
    let (footage, project) = tokio::task::spawn_blocking(move || -> CmdResult<_> {
        let db = Db::open(&db_path).map_err(err)?;
        let project = db.project(project_id).map_err(err)?;
        let footage = ghostreel_core::chat::build::survey(&db, project_id).map_err(err)?;
        Ok((footage, project))
    })
    .await
    .map_err(|e| e.to_string())??;

    let script =
        ghostreel_core::chat::build::build(&footage, &project, &brief, target_s, &config.jev).await.map_err(err)?;

    let brief_for_judge = brief.clone();
    let db_path = p.db_file();
    let script_cfg = config.script.clone();
    let (script_id, script) =
        tokio::task::spawn_blocking(move || -> CmdResult<(i64, ghostreel_core::script::Script)> {
            let db = Db::open(&db_path).map_err(err)?;
            let mut s = script;
            ghostreel_core::chat::repair::finish_script(&db, project_id, &mut s, true, &script_cfg).map_err(err)?;
            let id = ghostreel_core::script::save_version(&db, project_id, &s, None).map_err(err)?;
            Ok((id, s))
        })
        .await
        .map_err(|e| e.to_string())??;

    // Judge it and open it as a conversation, the same handoff `ghostreel script build` does.
    // The builder picks pictures well and structures poorly and a chat brain is the other way
    // round, so the useful thing is not the cut but the cut plus a critique, in a session the
    // next turn can carry on from. Without this the button is a dead end: the best pipeline
    // measured here (build, then refine) was reachable only from the CLI.
    //
    // `plan` reads the database and `ask` does not, which is the whole reason the judge is split
    // in two — a `Db` is not `Sync` and must not be held across the request.
    let db_path = p.db_file();
    let jev_cfg = config.jev.clone();
    let (planned, script) = tokio::task::spawn_blocking(move || -> CmdResult<_> {
        let db = Db::open(&db_path).map_err(err)?;
        let planned = ghostreel_core::chat::judge::plan(&db, &script, Some(&brief_for_judge), &jev_cfg);
        Ok((planned, script))
    })
    .await
    .map_err(|e| e.to_string())??;
    // A judgement is a nicety; the handoff is the point. Losing the notes must not lose the
    // session, so a judge that is off, unkeyed or unreachable is not an error here.
    let notes = match planned {
        Some(p) => p.ask().await.map(|j| j.notes()).unwrap_or_default(),
        None => Vec::new(),
    };

    let db_path = p.db_file();
    let final_session_id = tokio::task::spawn_blocking(move || -> CmdResult<i64> {
        let db = Db::open(&db_path).map_err(err)?;
        ghostreel_core::chat::build::save_as_session(&db, project_id, session_id, &brief, script_id, &script, &notes)
            .map_err(err)
    })
    .await
    .map_err(|e| e.to_string())??;

    Ok(BuiltCut { script_id, session_id: final_session_id })
}

/// What the build hands back: the cut, and the conversation to refine it in.
#[derive(serde::Serialize)]
struct BuiltCut {
    script_id: i64,
    session_id: i64,
}

#[tauri::command]
async fn get_ai_settings() -> CmdResult<AiSettingsView> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let vision_caps = vision_caps_view(&config).await;
    Ok(AiSettingsView {
        vision_caps,
        vision: llm_view(&config.vision),
        chat_model: llm_view(&config.chat_model()),
        stt: SttSettingsView { backend: config.stt.backend.to_string(), url: config.stt.url, model: config.stt.model },
        embed: EmbedSettingsView {
            backend: config.embed.backend.to_string(),
            url: config.embed.url,
            model: config.embed.model,
        },
        frames: FrameSettingsView { max_interval_s: config.frames.max_interval_s, long_side: config.frames.long_side },
        jev: jev_view(&config.jev),
    })
}

#[tauri::command]
async fn set_ai_settings(patch: AiSettingsPatch, search_state: State<'_, SearchState>) -> CmdResult<AiSettingsView> {
    let p = paths()?;
    let mut config = Config::load(&p.config_file).unwrap_or_default();

    fn parse_backend(s: &str) -> CmdResult<Backend> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "local" => Ok(Backend::Local),
            "server" => Ok(Backend::Server),
            other => Err(format!("invalid backend '{other}'; expected 'auto', 'local', or 'server'")),
        }
    }

    /// Vision and the script chat can also delegate to a coding-agent CLI; speech and
    /// embeddings cannot, so they keep `parse_backend` above.
    fn parse_backend_cli_ok(s: &str) -> CmdResult<Backend> {
        match s.to_lowercase().as_str() {
            "cli" => Ok(Backend::Cli),
            other => parse_backend(other)
                .map_err(|_| format!("invalid backend '{other}'; expected 'auto', 'local', 'server', or 'cli'")),
        }
    }

    if let Some(v) = patch.vision {
        apply_llm_patch(&mut config.vision, v, "vision", parse_backend_cli_ok)?;
    }

    if let Some(v) = patch.chat_model {
        let mut cfg = config.chat_model();
        apply_llm_patch(&mut cfg, v, "chat_model", parse_backend_cli_ok)?;
        config.chat_model = Some(cfg);
    }

    if let Some(s) = patch.stt {
        if let Some(b) = s.backend {
            config.stt.backend = parse_backend(&b)?;
        }
        if let Some(url) = s.url {
            config.stt.url = url;
        }
        if let Some(model) = s.model {
            // Reject vision catalog IDs (whisper() accepts any valid name, so we must
            // explicitly exclude known vision entries to avoid ambiguity).
            let is_vision = models::find_entry(&model).is_some_and(|e| e.kind == models::ModelKind::Vision);
            let is_valid_whisper = model == "auto"
                || models::find_entry(&model).is_some_and(|e| e.kind == models::ModelKind::Whisper)
                || (!is_vision && models::whisper(&model).is_ok());
            if !is_valid_whisper {
                return Err(format!("unknown speech model '{model}'"));
            }
            config.stt.model = model;
        }
    }

    let mut embed_changed = false;
    if let Some(e) = patch.embed {
        if let Some(b) = e.backend {
            let parsed = parse_backend(&b)?;
            if parsed != config.embed.backend {
                embed_changed = true;
                config.embed.backend = parsed;
            }
        }
        if let Some(url) = e.url
            && url != config.embed.url
        {
            embed_changed = true;
            config.embed.url = url;
        }
        if let Some(model) = e.model
            && !model.is_empty()
            && model != EMBED_MODEL
        {
            return Err(format!("embedding model cannot be changed from {EMBED_MODEL}: vectors must remain portable"));
        }
    }

    if embed_changed {
        *search_state.0.lock().await = None;
    }

    if let Some(f) = patch.frames {
        if let Some(v) = f.max_interval_s {
            if v < 1.0 || v > 60.0 {
                return Err(format!("frames.max_interval_s must be between 1 and 60, got {v}"));
            }
            config.frames.max_interval_s = v;
        }
        if let Some(s) = f.long_side {
            if !(256..=3840).contains(&s) {
                return Err(format!("frames.long_side must be between 256 and 3840, got {s}"));
            }
            config.frames.long_side = s;
        }
    }

    if let Some(j) = patch.jev {
        if let Some(on) = j.enabled {
            config.jev.enabled = on;
        }
        if let Some(on) = j.judge {
            config.jev.judge = on;
        }
        if let Some(key) = j.api_key {
            // An empty string is how the page clears a stored key, so it is not ignored.
            config.jev.api_key = key.trim().to_string();
        }
        if let Some(m) = j.model {
            config.jev.model = m;
        }
        config.jev.validate()?;
        if config.jev.enabled
            && config.jev.api_key.trim().is_empty()
            && std::env::var("TYPESAFE_API_KEY").ok().filter(|k| !k.trim().is_empty()).is_none()
        {
            return Err("turning Jev on needs an API key: paste one, or set TYPESAFE_API_KEY".into());
        }
    }

    config.save(&p.config_file).map_err(err)?;

    let vision_caps = vision_caps_view(&config).await;
    Ok(AiSettingsView {
        vision_caps,
        vision: llm_view(&config.vision),
        chat_model: llm_view(&config.chat_model()),
        stt: SttSettingsView { backend: config.stt.backend.to_string(), url: config.stt.url, model: config.stt.model },
        embed: EmbedSettingsView {
            backend: config.embed.backend.to_string(),
            url: config.embed.url,
            model: config.embed.model,
        },
        frames: FrameSettingsView { max_interval_s: config.frames.max_interval_s, long_side: config.frames.long_side },
        jev: jev_view(&config.jev),
    })
}

/// Run one describe call through the configured CLI agent, so the user can check it works before
/// starting an index run. `capability`: "vision" or "chat_model".
#[tauri::command]
async fn test_cli_agent(capability: String) -> CmdResult<String> {
    let p = paths()?;
    let config = Config::load(&p.config_file).map_err(err)?;
    let cfg = if capability == "chat_model" { config.chat_model() } else { config.vision.clone() };
    ghostreel_core::cliagent::CliAgent::new(cfg.cli).self_test(&p.data_dir).await.map_err(err)
}

#[tauri::command]
async fn server_models(url: String) -> CmdResult<Vec<String>> {
    probe::server_models(&url).await.map_err(err)
}

#[tauri::command]
async fn probe_backends() -> CmdResult<BackendsResolutionView> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let client = probe::probe_client();

    let vision_probe = async {
        match config.vision.backend {
            Backend::Local => None,
            _ => Some(probe::vision(&client, &config.vision.url, &config.vision.model).await),
        }
    };
    let embed_probe = async {
        match config.embed.backend {
            Backend::Local => None,
            _ => Some(probe::embeddings(&client, &config.embed.url, &config.embed.model).await),
        }
    };
    let chat_cfg = config.chat_model();
    let chat_probe = async {
        match chat_cfg.backend {
            Backend::Local => None,
            _ => Some(probe::vision(&client, &chat_cfg.url, &chat_cfg.model).await),
        }
    };
    let stt_probe = async {
        match config.stt.backend {
            Backend::Local => None,
            _ => Some(probe::stt(&client, &config.stt.url).await),
        }
    };
    let (vision_p, chat_p, embed_p, stt_p) = tokio::join!(vision_probe, chat_probe, embed_probe, stt_probe);
    let vision = probe::resolve(config.vision.backend, vision_p);
    let chat = probe::resolve(chat_cfg.backend, chat_p);
    let embeddings = probe::resolve(config.embed.backend, embed_p);
    let stt = probe::resolve(config.stt.backend, stt_p);
    Ok(BackendsResolutionView { vision, chat, embeddings, stt })
}

#[tauri::command]
async fn enqueue_model_download(app: AppHandle, queue: State<'_, queue::Queue>, model_id: String) -> CmdResult<u64> {
    let file_name = if let Some(e) = ghostreel_core::models::find_entry(&model_id) {
        e.file_name
    } else if let Ok(s) = ghostreel_core::models::whisper(&model_id) {
        s.file_name
    } else {
        return Err(format!("unknown model '{model_id}'"));
    };
    let label = format!("Download {file_name}");
    Ok(queue.enqueue(&app, queue::TaskKind::DownloadModel { model_id }, label).await)
}

#[tauri::command]
fn remove_model(model_id: String) -> CmdResult<()> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    ghostreel_core::models::remove(&dir, &model_id).map_err(err)?;
    Ok(())
}

#[tauri::command]
fn set_whisper_model(model_id: String) -> CmdResult<()> {
    let p = paths()?;
    let mut config = Config::load(&p.config_file).unwrap_or_default();
    config.stt.model = model_id;
    config.save(&p.config_file).map_err(err)?;
    Ok(())
}

#[tauri::command]
fn open_models_dir(app: AppHandle) -> CmdResult<()> {
    let p = paths()?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let dir = ghostreel_core::models::effective_models_dir(&p, &config);
    let _ = std::fs::create_dir_all(&dir);
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(dir.to_string_lossy(), None::<&str>).map_err(err)
}

#[tauri::command]
fn video_transcript(video_id: i64) -> CmdResult<Vec<TranscriptSegment>> {
    index::transcript(&open_db()?, video_id).map_err(err)
}

/// `script.max_shake_jerk` from the config; 0 when unset or the check is off.
fn shake_limit() -> f64 {
    paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script.max_shake_jerk).unwrap_or(0.0)
}

/// `script.max_sway` from the config.
fn max_sway() -> f64 {
    paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script.max_sway).unwrap_or(0.0)
}

/// `script.shake_relative` from the config.
fn shake_relative() -> f64 {
    paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script.shake_relative).unwrap_or(0.0)
}

/// How steady the camera is through one video, window by window, plus the limit the app calls
/// shaky — so the panel can mark the stretches to cut around.
#[derive(Serialize)]
struct SteadinessView {
    windows: Vec<ghostreel_core::steadiness::Window>,
    /// The global floor.
    max_shake: f64,
    /// The line this clip is judged against: the floor, or a multiple of its own level.
    limit: f64,
    /// Sway above this is shaky too.
    max_sway: f64,
    camera: String,
}

#[tauri::command]
fn video_steadiness(video_id: i64) -> CmdResult<SteadinessView> {
    let p = paths()?;
    let db = Db::open(&p.db_file()).map_err(err)?;
    let windows = db.motion_windows(video_id).map_err(err)?;
    let limit = ghostreel_core::steadiness::shake_limit(&windows, shake_limit(), shake_relative());
    let camera = ghostreel_core::steadiness::camera_style(&windows).as_str().to_string();
    Ok(SteadinessView { windows, max_shake: shake_limit(), limit, max_sway: max_sway(), camera })
}

/// What the player should load for a video: the original when the browser can play it, else a
/// proxy built now (and kept) — 4K 10-bit camera files otherwise sit black for a long time.
#[derive(Serialize)]
struct PlaybackView {
    path: PathBuf,
    proxy: bool,
}

#[tauri::command]
async fn video_playback(video_id: i64) -> CmdResult<PlaybackView> {
    let p = paths()?;
    let (path, hash, duration, width, height): (String, String, Option<f64>, Option<u32>, Option<u32>) = {
        let db = Db::open(&p.db_file()).map_err(err)?;
        db.conn
            .query_row(
                "SELECT vf.path, v.content_hash, v.duration_s, v.width, v.height FROM videos v JOIN video_files vf ON vf.video_id = v.id WHERE v.id = ?1 LIMIT 1",
                [video_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .map_err(err)?
    };
    let long_side = Config::load(&p.config_file).map_err(err)?.frames.clamped_long_side();
    let original = PathBuf::from(&path);
    let ffprobe = doctor::locate("ffprobe").ok_or_else(|| "ffprobe not found".to_string())?;
    if width.zip(height).is_some_and(|(w, h)| w.max(h) <= long_side)
        && ghostreel_core::preview::plays_natively(&ffprobe, &original).await
    {
        return Ok(PlaybackView { path: original, proxy: false });
    }
    let ffmpeg = doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let proxy = ghostreel_core::preview::playback_proxy_with_progress(
        &ffmpeg,
        &p.data_dir,
        &original,
        &hash,
        duration,
        long_side,
    )
    .await
    .map_err(err)?;
    Ok(PlaybackView { path: proxy, proxy: true })
}

/// Every audio track of a video's file, from its header.
#[tauri::command]
async fn video_audio_tracks(video_id: i64) -> CmdResult<Vec<ghostreel_core::audio_tracks::AudioTrackInfo>> {
    let src = {
        let db = open_db()?;
        ghostreel_core::audio_tracks::track_source(&db, video_id).map_err(err)?
    };
    let ffprobe = doctor::locate("ffprobe").ok_or_else(|| "ffprobe not found".to_string())?;
    ghostreel_core::audio_tracks::list_tracks(&ffprobe, &src.path, src.auto_track).await.map_err(err)
}

/// Mean level in dB of each audio track over a source range (-91 = silence), so the editor can
/// show which tracks carry sound.
#[tauri::command]
async fn audio_levels(video_id: i64, in_s: f64, out_s: f64) -> CmdResult<Vec<f64>> {
    let src = {
        let db = open_db()?;
        ghostreel_core::audio_tracks::track_source(&db, video_id).map_err(err)?
    };
    let ffprobe = doctor::locate("ffprobe").ok_or_else(|| "ffprobe not found".to_string())?;
    let ffmpeg = doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let count = ghostreel_core::audio::track_count(&ffprobe, &src.path).await.map_err(err)?;
    Ok(ghostreel_core::audio_tracks::track_levels(&ffmpeg, &src.path, count, in_s, out_s).await)
}

/// One audio track as a small cached `.m4a`, for the live viewer to play beside the picture.
#[tauri::command]
async fn audio_track_proxy(video_id: i64, track: u32) -> CmdResult<PathBuf> {
    let p = paths()?;
    let src = {
        let db = Db::open(&p.db_file()).map_err(err)?;
        ghostreel_core::audio_tracks::track_source(&db, video_id).map_err(err)?
    };
    let cached = ghostreel_core::audio_tracks::track_proxy_path(&p.data_dir, &src.content_hash, track);
    if cached.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(cached);
    }
    let ffprobe = doctor::locate("ffprobe").ok_or_else(|| "ffprobe not found".to_string())?;
    let ffmpeg = doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let channels = ghostreel_core::audio::track_channels(&ffprobe, &src.path, track)
        .await
        .ok_or_else(|| format!("video #{video_id} has no audio track {}", track + 1))?;
    ghostreel_core::audio_tracks::track_proxy(&ffmpeg, &p.data_dir, &src.path, &src.content_hash, track, channels)
        .await
        .map_err(err)
}

#[tauri::command]
fn video_playback_progress(video_id: i64) -> CmdResult<Option<ghostreel_core::preview::PlaybackProgress>> {
    let p = paths()?;
    let db = Db::open(&p.db_file()).map_err(err)?;
    let hash: String =
        db.conn.query_row("SELECT content_hash FROM videos WHERE id = ?1", [video_id], |r| r.get(0)).map_err(err)?;
    let long_side = Config::load(&p.config_file).map_err(err)?.frames.clamped_long_side();
    Ok(ghostreel_core::preview::playback_proxy_progress(&p.data_dir, &hash, long_side))
}

#[tauri::command]
fn video_frames(video_id: i64) -> CmdResult<Vec<FrameRow>> {
    let p = paths()?;
    index::frames(&Db::open(&p.db_file()).map_err(err)?, &p.data_dir, video_id).map_err(err)
}

#[tauri::command]
fn video_details(video_id: i64) -> CmdResult<VideoRow> {
    let p = paths()?;
    let db = Db::open(&p.db_file()).map_err(err)?;
    index::video_by_id(&db, video_id, shake_limit(), shake_relative(), max_sway()).map_err(err)
}

#[tauri::command]
async fn open_search_window(app: tauri::AppHandle) -> CmdResult<()> {
    use tauri::Manager;
    if let Some(win) = app.get_webview_window("search") {
        let _ = win.unminimize();
        let _ = win.show();
        let _ = win.set_focus();
        return Ok(());
    }
    let builder =
        tauri::WebviewWindowBuilder::new(&app, "search", tauri::WebviewUrl::App("index.html?window=search".into()))
            .title("Search Material — GhostReel")
            .inner_size(1020.0, 720.0)
            .min_inner_size(600.0, 450.0)
            .resizable(true)
            .center();

    builder.build().map_err(err)?;
    Ok(())
}

#[tauri::command]
async fn focus_main_window(app: tauri::AppHandle) -> CmdResult<()> {
    use tauri::Manager;
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.unminimize();
        let _ = win.show();
        let _ = win.set_focus();
    }
    Ok(())
}

#[derive(Serialize)]
struct ScriptView {
    stored: ghostreel_core::script::StoredScript,
    issues: Vec<ghostreel_core::script::Issue>,
}

#[derive(Serialize)]
struct SaveScriptView {
    script_id: i64,
    issues: Vec<ghostreel_core::script::Issue>,
    /// What the deterministic repair changed before saving (empty: saved as given).
    notes: Vec<String>,
    /// The script as saved (after repairs).
    script: ghostreel_core::script::Script,
}

/// A script after one fix, checked again (not saved: the editor saves it).
#[derive(Serialize)]
struct FixedScriptView {
    script: ghostreel_core::script::Script,
    issues: Vec<ghostreel_core::script::Issue>,
    note: String,
}

/// The checks of a script against the index as it is now. A clip of a clip run keeps that run's
/// minimum length when a fix trims it.
fn live_issues(
    db: &ghostreel_core::db::Db,
    project_id: i64,
    script: &ghostreel_core::script::Script,
) -> CmdResult<Vec<ghostreel_core::script::Issue>> {
    let cfg = paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script).unwrap_or_default();
    let min = ghostreel_core::clips::length_range(db, project_id, &script.title).map(|r| r.0);
    let opts = ghostreel_core::script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    ghostreel_core::script_check::check(db, project_id, script, &cfg, &opts).map_err(err)
}

/// Queue the indexing stages the issues ask for (the project has them on, nothing queued or
/// running for them yet, and the last try didn't fail) and say so on the issues. When it
/// finishes, `task-finished` makes the editor check and plan the framing again.
async fn queue_needed_stages(
    app: &AppHandle,
    queue: &queue::Queue,
    project_id: i64,
    issues: &mut [ghostreel_core::script::Issue],
) {
    for stage in ghostreel_core::script_check::stages_to_queue(issues) {
        let (kind, label) = match stage.as_str() {
            "faces" => (queue::TaskKind::Faces { project_id, force: false }, "Find faces for framing"),
            "speakers" => (queue::TaskKind::Speakers { project_id, force: false }, "Find speakers for framing"),
            _ => (queue::TaskKind::Index { project_id }, "Index for framing"),
        };
        let tasks = queue.snapshot().await;
        let same = |t: &&queue::Task| t.kind.key() == kind.key() && t.project_id == Some(project_id);
        let active = tasks.iter().filter(same).any(|t| t.state.is_active());
        let failed = tasks.iter().filter(same).last().is_some_and(|t| t.state == queue::TaskState::Failed);
        if failed {
            continue;
        }
        if !active {
            queue.enqueue(app, kind, label.to_string()).await;
        }
        ghostreel_core::script_check::mark_queued(issues, &stage);
    }
}

#[tauri::command]
async fn chat_turn(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    session_id: Option<i64>,
    message: String,
    images: Option<Vec<String>>,
    style: Option<ghostreel_core::script::ChatStyle>,
    script_id: Option<i64>,
) -> CmdResult<queue::ChatTurnView> {
    let session_id = match session_id {
        Some(id) => id,
        None => {
            let title: String = message.chars().take(60).collect();
            let db = open_db()?;
            ghostreel_core::chat::create_session(&db, project_id, &title).map_err(err)?
        }
    };
    // The composer's choice, recorded on the chat so every later turn keeps it.
    if let Some(style) = style {
        ghostreel_core::chat::set_session_style(&open_db()?, session_id, style).map_err(err)?;
    }
    let label = format!("Script chat: {}", message.chars().take(40).collect::<String>());
    let (tx, rx) = tokio::sync::oneshot::channel();
    queue.enqueue_chat(&app, project_id, session_id, message, images.unwrap_or_default(), label, script_id, tx).await;
    match rx.await {
        Ok(res) => res,
        Err(_) => Err("chat task cancelled or failed".into()),
    }
}

#[tauri::command]
fn chat_sessions(project_id: i64) -> CmdResult<Vec<ghostreel_core::chat::ChatSession>> {
    let db = open_db()?;
    ghostreel_core::chat::sessions(&db, project_id).map_err(err)
}

#[tauri::command]
fn delete_chat_session(session_id: i64) -> CmdResult<bool> {
    let db = open_db()?;
    ghostreel_core::chat::delete_session(&db, session_id).map_err(err)
}

#[tauri::command]
fn chat_messages(session_id: i64) -> CmdResult<Vec<ghostreel_core::chat::ChatMessage>> {
    let db = open_db()?;
    ghostreel_core::chat::messages(&db, session_id).map_err(err)
}

/// Reveal a saved chat trace in the desktop file manager.
#[tauri::command]
fn open_chat_log(app: tauri::AppHandle, session_id: i64) -> CmdResult<()> {
    use tauri_plugin_opener::OpenerExt;
    let path = paths()?.data_dir.join("logs/chat").join(format!("{session_id}.jsonl"));
    if !path.is_file() {
        return Err("No saved log for this chat".into());
    }
    app.opener().reveal_item_in_dir(path).map_err(err)
}

/// Read a bounded batch from the chat trace after `after_seq`.
#[tauri::command]
fn chat_log(session_id: i64, after_seq: u64) -> CmdResult<Vec<ghostreel_core::llmlog::LogEntry>> {
    Ok(ghostreel_core::llmlog::since(session_id, after_seq))
}

#[tauri::command]
async fn generate_clips(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    options: ghostreel_core::clips::ClipOptions,
) -> CmdResult<u64> {
    options.validate().map_err(err)?;
    agents::check_selection(Some(&options.agents))?;
    let db = open_db()?;
    let name = db.project(project_id).map(|p| p.name).unwrap_or_else(|_| format!("{project_id}"));
    let label = format!("Generate {} {} “{name}”", options.count, if options.count == 1 { "clip" } else { "clips" });
    Ok(queue.enqueue(&app, queue::TaskKind::Clips { project_id, options }, label).await)
}

/// What a clip run with these options would run into, before it is queued: agents that cannot
/// be used and indexing still to do (`ghostreel_core::clips::preflight`).
#[tauri::command]
async fn clips_preflight(
    project_id: i64,
    options: ghostreel_core::clips::ClipOptions,
) -> CmdResult<ghostreel_core::preflight::Preflight> {
    let p = paths()?;
    let config = Config::load(&p.config_file).map_err(err)?;
    ghostreel_core::clips::preflight(&p, &config, project_id, &options).await.map_err(err)
}

#[tauri::command]
fn list_clip_sets(project_id: i64) -> CmdResult<Vec<ghostreel_core::clips::ClipSet>> {
    let db = open_db()?;
    let data_dir = paths()?.data_dir;
    let mut sets = ghostreel_core::clips::list_sets(&db, project_id).map_err(err)?;
    let cfg = script_cfg();
    for s in &mut sets {
        ghostreel_core::clips::with_thumbs(&db, &data_dir, s);
        ghostreel_core::clips::with_live_checks(&db, &cfg, s);
    }
    Ok(sets)
}

/// The script settings (defaults when the config can't be read).
fn script_cfg() -> ghostreel_core::config::ScriptConfig {
    paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script).unwrap_or_default()
}

#[tauri::command]
fn get_clip_set(set_id: String) -> CmdResult<ghostreel_core::clips::ClipSet> {
    let db = open_db()?;
    let mut set = ghostreel_core::clips::load_set(&db, &set_id).map_err(err)?;
    ghostreel_core::clips::with_thumbs(&db, &paths()?.data_dir, &mut set);
    ghostreel_core::clips::with_live_checks(&db, &script_cfg(), &mut set);
    Ok(set)
}

#[tauri::command]
fn set_clip_status(set_id: String, script_id: i64, status: String) -> CmdResult<ghostreel_core::clips::ClipSet> {
    let db = open_db()?;
    let mut set = ghostreel_core::clips::set_item_status(&db, &set_id, script_id, &status).map_err(err)?;
    ghostreel_core::clips::with_thumbs(&db, &paths()?.data_dir, &mut set);
    ghostreel_core::clips::with_live_checks(&db, &script_cfg(), &mut set);
    Ok(set)
}

/// Move a clip set and its clips to the trash (restorable; recorded as a run).
#[tauri::command]
fn delete_clip_set(set_id: String) -> CmdResult<ghostreel_core::trash::TrashItem> {
    let db = open_db()?;
    ghostreel_core::clips::delete_set(&db, &set_id, "app").map_err(err)
}

/// Render a set's clips (accepted only, or everything not rejected) into `dir`.
#[tauri::command]
async fn export_clip_set(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    set_id: String,
    dir: String,
    accepted_only: bool,
) -> CmdResult<u64> {
    let db = open_db()?;
    let set = ghostreel_core::clips::load_set(&db, &set_id).map_err(err)?;
    let n = set.items.iter().filter(|i| if accepted_only { i.status == "accepted" } else { i.status != "rejected" }).count();
    if n == 0 {
        return Err(if accepted_only { "no accepted clips to export".into() } else { "no clips to export".into() });
    }
    Ok(queue
        .enqueue(&app, queue::TaskKind::ExportClips { set_id, dir, accepted_only }, format!("Export {n} {}", if n == 1 { "clip" } else { "clips" }))
        .await)
}

#[tauri::command]
fn list_scripts(project_id: i64) -> CmdResult<Vec<ghostreel_core::script::ScriptSummary>> {
    let db = open_db()?;
    ghostreel_core::script::list(&db, project_id).map_err(err)
}

#[tauri::command]
async fn get_script(app: AppHandle, queue: State<'_, queue::Queue>, script_id: i64) -> CmdResult<ScriptView> {
    let db = open_db()?;
    let stored = ghostreel_core::script::load(&db, script_id).map_err(err)?;
    let mut issues = live_issues(&db, stored.project_id, &stored.script)?;
    queue_needed_stages(&app, &queue, stored.project_id, &mut issues).await;
    Ok(ScriptView { stored, issues })
}

/// Check an (unsaved) script against the index as it is now: the editor asks whenever the script
/// changes or a task finishes, so what it shows is never stale.
#[tauri::command]
async fn check_script(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    script: ghostreel_core::script::Script,
) -> CmdResult<Vec<ghostreel_core::script::Issue>> {
    let db = open_db()?;
    let mut issues = live_issues(&db, project_id, &script)?;
    queue_needed_stages(&app, &queue, project_id, &mut issues).await;
    Ok(issues)
}

/// Apply one fix an issue offered to an (unsaved) script and check it again. A run_stage fix
/// turns the stage on when asked and queues it; the script comes back unchanged.
#[tauri::command]
async fn apply_script_fix(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    mut script: ghostreel_core::script::Script,
    fix: ghostreel_core::script::FixAction,
) -> CmdResult<FixedScriptView> {
    let mut db = open_db()?;
    let applied = ghostreel_core::script_fix::apply_fix(&mut db, project_id, &mut script, &fix).map_err(err)?;
    if let Some(stage) = &applied.run_stage {
        let kind = match stage.as_str() {
            "faces" => queue::TaskKind::Faces { project_id, force: false },
            "speakers" => queue::TaskKind::Speakers { project_id, force: false },
            _ => queue::TaskKind::Index { project_id },
        };
        queue.enqueue(&app, kind, format!("Run {stage} (asked by a framing fix)")).await;
    }
    let mut issues = live_issues(&db, project_id, &script)?;
    queue_needed_stages(&app, &queue, project_id, &mut issues).await;
    Ok(FixedScriptView { script, issues, note: applied.note })
}

#[tauri::command]
async fn save_script(
    app: AppHandle,
    queue: State<'_, queue::Queue>,
    project_id: i64,
    mut script: ghostreel_core::script::Script,
    session_id: Option<i64>,
) -> CmdResult<SaveScriptView> {
    agents::check_selection(script.agents.as_ref())?;
    let db = open_db()?;
    let _ = ghostreel_core::script::snap_to_segments(&db, &mut script).map_err(err)?;
    // The deterministic framing work (a vertical clip off a screen share, the closing hold on a
    // face), the same pass clip runs and imports get; what it did comes back as notes.
    let cfg = paths().map(|p| Config::load(&p.config_file).unwrap_or_default().script).unwrap_or_default();
    let min = ghostreel_core::clips::length_range(&db, project_id, &script.title).map(|r| r.0);
    let notes = ghostreel_core::script_fix::repair_framing(&db, &mut script, cfg.speech_overrun_s, min);
    let mut issues = live_issues(&db, project_id, &script)?;
    queue_needed_stages(&app, &queue, project_id, &mut issues).await;
    let script_id = ghostreel_core::script::save_version(&db, project_id, &script, session_id).map_err(err)?;
    Ok(SaveScriptView { script_id, issues, notes, script })
}

/// Move one script version to the trash (restorable; recorded as a run).
#[tauri::command]
fn delete_script(script_id: i64) -> CmdResult<ghostreel_core::trash::TrashItem> {
    let db = open_db()?;
    ghostreel_core::script::delete(&db, script_id, "app").map_err(err)
}

/// Move every version of a cut to the trash, as one restorable entry.
#[tauri::command]
fn delete_script_cut(project_id: i64, title: String) -> CmdResult<ghostreel_core::trash::TrashItem> {
    let db = open_db()?;
    ghostreel_core::script::delete_cut(&db, project_id, &title, "app").map_err(err)
}

#[tauri::command]
fn script_captions(script: ghostreel_core::script::Script) -> CmdResult<Vec<ghostreel_core::captions::Cue>> {
    let db = open_db()?;
    ghostreel_core::captions::cues_for_script(&db, &script).map_err(err)
}

/// Every burnt-in title and caption of a script, laid out exactly as a render places them, so
/// the live viewer draws the same lines in the same spots.
#[tauri::command]
fn script_overlays(script: ghostreel_core::script::Script) -> CmdResult<ghostreel_core::overlays::Overlays> {
    let db = open_db()?;
    ghostreel_core::overlays::for_script(&db, &script).map_err(err)
}

/// The crop path of every framed clip in a script (vertical cuts of wide footage), so the live
/// player crops exactly as a render will.
#[tauri::command]
fn script_framing(
    script: ghostreel_core::script::Script,
) -> CmdResult<Vec<ghostreel_core::framing::ClipFraming>> {
    let db = open_db()?;
    let ffmpeg = doctor::locate("ffmpeg");
    ghostreel_core::framing::plans_for_script(&db, ffmpeg.as_deref(), &script).map_err(err)
}

/// WebKitGTK's DMABUF renderer dies with "Error 71 (Protocol error) dispatching to Wayland
/// display" on wlroots compositors (Hyprland, Sway) — same workaround as GhostPen. Only on
/// Wayland, and only if the user hasn't chosen a value themselves.
#[cfg(target_os = "linux")]
fn apply_wayland_webkit_workaround() {
    let on_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    if on_wayland && std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        // SAFETY: called first thing in `run`, before any other thread exists.
        unsafe { std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1") };
    }
}

/// Return the running app version (from Cargo.toml, baked in at compile time).
#[tauri::command]
fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Web access: whether the UI is being served to other devices, and where.
#[tauri::command]
fn web_status(server: State<'_, webui::WebServer>) -> CmdResult<webui::WebStatus> {
    let config = Config::load(&paths()?.config_file).unwrap_or_default();
    Ok(webui::status(&server, &config.web))
}

/// Saves a web access change and restarts the server to match it.
#[tauri::command]
async fn set_web_config(app: AppHandle, patch: webui::WebConfigPatch) -> CmdResult<webui::WebStatus> {
    use tauri::Manager;
    let p = paths()?;
    let mut config = Config::load(&p.config_file).map_err(err)?;
    webui::merge(&mut config.web, patch)?;
    config.save(&p.config_file).map_err(err)?;
    webui::apply(&app, &config.web).await;
    Ok(webui::status(&app.state::<webui::WebServer>(), &config.web))
}

/// Write a database snapshot when the newest is older than `max_age`. Failures are logged,
/// never fatal.
fn backup_if_older_than(max_age: std::time::Duration) {
    let Ok(p) = Paths::resolve() else { return };
    let db = match Db::open(&p.db_file()) {
        Ok(db) => db,
        Err(e) => return eprintln!("housekeeping: cannot open the library: {e}"),
    };
    if let Err(e) = ghostreel_core::backup::ensure_newer_than(&db, &p.data_dir, std::time::SystemTime::now(), max_age) {
        eprintln!("housekeeping: database backup failed: {e}");
    }
}

/// At start-up: write a database snapshot when the newest is older than
/// `backup::BACKUP_EVERY`, then purge trash entries past `trash::KEEP_DAYS` (the snapshot goes
/// first, so it still holds what the purge removes). Failures are logged, never fatal.
fn start_up_housekeeping() {
    backup_if_older_than(ghostreel_core::backup::BACKUP_EVERY);
    let Ok(p) = Paths::resolve() else { return };
    let db = match Db::open(&p.db_file()) {
        Ok(db) => db,
        Err(e) => return eprintln!("housekeeping: cannot open the library: {e}"),
    };
    if let Err(e) = ghostreel_core::trash::purge_expired(&db, &p.data_dir, ghostreel_core::projects::now(), "app") {
        eprintln!("housekeeping: emptying old trash failed: {e}");
    }
}

/// For an app left open for days: every `backup::BACKUP_CHECK_EVERY`, write a snapshot when the
/// newest is older than `backup::BACKUP_CHECK_AGE`.
async fn periodic_backups() {
    use ghostreel_core::backup::{BACKUP_CHECK_AGE, BACKUP_CHECK_EVERY};
    // The first tick is a full period away: start-up has just taken care of this moment.
    let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + BACKUP_CHECK_EVERY, BACKUP_CHECK_EVERY);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let _ = tauri::async_runtime::spawn_blocking(|| backup_if_older_than(BACKUP_CHECK_AGE)).await;
    }
}

/// What of the main window is remembered across restarts (everything but visibility).
fn window_state_flags() -> tauri_plugin_window_state::StateFlags {
    use tauri_plugin_window_state::StateFlags;
    StateFlags::SIZE | StateFlags::POSITION | StateFlags::MAXIMIZED | StateFlags::FULLSCREEN
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "linux")]
    apply_wayland_webkit_workaround();

    // CI pins Vulkan bundles to their own signed updater payloads.
    let mut updater = tauri_plugin_updater::Builder::new();
    if let Some(target) = option_env!("GHOSTREEL_UPDATER_TARGET") {
        updater = updater.target(target);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(updater.build())
        .plugin(tauri_plugin_process::init())
        // The main window comes back at its last size, place and maximized state. It starts
        // hidden (tauri.conf.json) and the plugin restores it as it is created, before `setup`
        // shows it, so it never visibly jumps. Visibility itself is not restored: a window
        // hidden at quit must still open. On Wayland the compositor owns the position, so only
        // size and maximized apply there. The search window opens fresh each time.
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_state_flags(window_state_flags())
                .with_denylist(&["search"])
                .build(),
        )
        .manage(SearchState::default())
        .manage(tokio::sync::OnceCell::<media::MediaServer>::new())
        .manage(queue::Queue::default())
        .manage(webui::WebServer::default())
        .setup(|app| {
            // Frames/thumbnails are served from the data dir via the asset protocol; scope it
            // at runtime because GHOSTREEL_DATA can move it.
            use tauri::Manager;
            #[cfg(target_os = "macos")]
            menu::install(app)?;
            if let Ok(p) = Paths::resolve() {
                ghostreel_core::llmlog::init(&p.data_dir);
                let _ = std::fs::create_dir_all(&p.data_dir);
                app.asset_protocol_scope().allow_directory(&p.data_dir, true)?;
                let previews_dir = p.data_dir.join("previews");
                let _ = std::fs::create_dir_all(&previews_dir);
                let _ = app.asset_protocol_scope().allow_directory(&previews_dir, true);
                let proxies_dir = p.data_dir.join("proxies");
                let _ = std::fs::create_dir_all(&proxies_dir);
                let _ = app.asset_protocol_scope().allow_directory(&proxies_dir, true);
                let chat_images_dir = p.data_dir.join("chat_images");
                let _ = std::fs::create_dir_all(&chat_images_dir);
                let _ = app.asset_protocol_scope().allow_directory(&chat_images_dir, true);
                // Watched folders, so the player can load the videos.
                if let Ok(db) = Db::open(&p.db_file()) {
                    for f in db.folders(None).unwrap_or_default() {
                        let _ = app.asset_protocol_scope().allow_directory(&f.path, true);
                    }
                }
            }
            // Config windows are built (and restored by the window-state plugin) before setup.
            if let Some(main) = app.get_webview_window("main") {
                let _ = main.show();
                let _ = main.set_focus();
            }
            // House keeping off the UI thread: snapshot the library when the newest backup is a
            // day old, then empty the trash of what is past keeping. A running app snapshots
            // again every few hours it stays open.
            std::thread::spawn(start_up_housekeeping);
            tauri::async_runtime::spawn(periodic_backups());
            tauri::async_runtime::spawn(queue::worker(app.handle().clone()));
            webui::forward_events(app.handle());
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Ok(p) = Paths::resolve() {
                    let web = Config::load(&p.config_file).unwrap_or_default().web;
                    webui::apply(&handle, &web).await;
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            agents::list_agents,
            agents::save_agents,
            agents::test_agent,
            agents::resolve_agents,
            agents::agent_status,
            doctor,
            cli_models,
            list_projects,
            create_project,
            set_project_pipeline,
            get_project_glossary,
            set_project_glossary,
            rename_project,
            exclude_video,
            include_video,
            set_video_category,
            plan_organize_project,
            execute_organize_project,
            remove_project,
            storage_report,
            clean_storage,
            compact_database,
            project_view,
            add_folder,
            remove_folder,
            enqueue_index,
            enqueue_steadiness,
            enqueue_faces,
            enqueue_speakers,
            project_people,
            set_person,
            merge_people,
            video_speakers,
            person_thumbnail,
            enqueue_preview,
            enqueue_export,
            preview_plan,
            get_script_preview,
            queue_list,
            list_runs,
            run_links_state,
            list_trash,
            restore_trash,
            restore_run_link,
            cancel_task,
            clear_finished_tasks,
            redo_project_stage,
            get_chat_settings,
            set_chat_system_prompt,
            video_transcript,
            video_frames,
            video_steadiness,
            video_playback,
            video_playback_progress,
            video_audio_tracks,
            audio_levels,
            audio_track_proxy,
            video_details,
            search,
            open_search_window,
            focus_main_window,
            open_external,
            media_base,
            chat_turn,
            chat_sessions,
            chat_messages,
            chat_log,
            open_chat_log,
            delete_chat_session,
            list_scripts,
            generate_clips,
            clips_preflight,
            list_clip_sets,
            get_clip_set,
            set_clip_status,
            delete_clip_set,
            export_clip_set,
            get_script,
            check_script,
            apply_script_fix,
            save_script,
            delete_script,
            delete_script_cut,
            script_captions,
            script_overlays,
            script_framing,
            models_status,
            enqueue_model_download,
            remove_model,
            set_whisper_model,
            open_models_dir,
            build_script_with_jev,
            get_ai_settings,
            set_ai_settings,
            server_models,
            probe_backends,
            test_cli_agent,
            app_version,
            web_status,
            set_web_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running GhostReel");
}
