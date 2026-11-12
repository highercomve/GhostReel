//! The app's background work queue: everything slow (indexing, renders, exports, clip runs, model
//! downloads, script chat) runs one task at a time, in order, with progress the UI can show and
//! cancel.
//!
//! Every task is a run (`ghostreel_core::runs`): whatever its kind, it reports through one
//! [`Reporter`] — `(fraction, phase, note)`, a tracker snapshot, or a status note — which keeps the
//! task's progress, note and steps the same way for all of them. When a task ends it gets typed
//! links to what it made and a summary, and project-scoped runs are saved so the history
//! survives restarts.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ghostreel_core::chat::ChatEvent;
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::index::{self, IndexLock};
use ghostreel_core::paths::Paths;
use ghostreel_core::progress::{Progress, Tracker};
use ghostreel_core::runs::{self, RunLink, RunState, StepLog};
use ghostreel_core::runtime;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::{Mutex, Notify, mpsc, oneshot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurnView {
    pub session_id: i64,
    pub reply: String,
    pub script_id: Option<i64>,
    pub issues: Vec<ghostreel_core::script::Issue>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskKind {
    Index {
        project_id: i64,
    },
    /// `out`: save the render there (MP4 export for demos) instead of the previews folder.
    RenderPreview {
        script_id: i64,
        burn_titles: bool,
        burn_narration: bool,
        burn_captions: bool,
        normalize_audio: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        out: Option<String>,
    },
    Export {
        script_id: i64,
        format: String,
        path: String,
    },
    Chat {
        project_id: i64,
        session_id: i64,
    },
    DownloadModel {
        model_id: String,
    },
    Steadiness {
        project_id: i64,
        #[serde(default)]
        force: bool,
    },
    Faces {
        project_id: i64,
        #[serde(default)]
        force: bool,
    },
    Speakers {
        project_id: i64,
        #[serde(default)]
        force: bool,
    },
    /// Generate a clip set (`clips::run`).
    Clips {
        project_id: i64,
        options: ghostreel_core::clips::ClipOptions,
    },
    /// Render a clip set's clips into a folder.
    ExportClips {
        set_id: String,
        dir: String,
        accepted_only: bool,
    },
}

impl TaskKind {
    /// The run kind key (the same as the serde tag): "index", "clips", "render_preview"…
    pub fn key(&self) -> &'static str {
        match self {
            TaskKind::Index { .. } => "index",
            TaskKind::RenderPreview { .. } => "render_preview",
            TaskKind::Export { .. } => "export",
            TaskKind::Chat { .. } => "chat",
            TaskKind::DownloadModel { .. } => "download_model",
            TaskKind::Steadiness { .. } => "steadiness",
            TaskKind::Faces { .. } => "faces",
            TaskKind::Speakers { .. } => "speakers",
            TaskKind::Clips { .. } => "clips",
            TaskKind::ExportClips { .. } => "export_clips",
        }
    }

    /// The project the task works for, when it has one (renders and exports through their script
    /// or clip set).
    fn project_id(&self) -> Option<i64> {
        let db = || Paths::resolve().ok().and_then(|p| Db::open(&p.db_file()).ok());
        match self {
            TaskKind::Index { project_id }
            | TaskKind::Chat { project_id, .. }
            | TaskKind::Steadiness { project_id, .. }
            | TaskKind::Faces { project_id, .. }
            | TaskKind::Speakers { project_id, .. }
            | TaskKind::Clips { project_id, .. } => Some(*project_id),
            TaskKind::RenderPreview { script_id, .. } | TaskKind::Export { script_id, .. } => {
                db()?.conn.query_row("SELECT project_id FROM scripts WHERE id = ?1", [script_id], |r| r.get(0)).ok()
            }
            TaskKind::ExportClips { set_id, .. } => {
                db()?.conn.query_row("SELECT project_id FROM clip_sets WHERE id = ?1", [set_id], |r| r.get(0)).ok()
            }
            TaskKind::DownloadModel { .. } => None,
        }
    }
}

/// A task's state is its run's state.
pub type TaskState = RunState;

#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub id: u64,
    /// The run this task is (stable: the saved run keeps it after a restart).
    pub run_id: String,
    pub project_id: Option<i64>,
    pub kind: TaskKind,
    pub label: String,
    pub state: TaskState,
    pub progress: Option<Progress>,
    /// Latest human-readable status line ("Transcribing: GhostPen @ …").
    pub note: Option<String>,
    /// Every phase so far with its last note.
    pub steps: StepLog,
    /// What the task made, typed so the UI knows how to open it.
    pub links: Vec<RunLink>,
    /// One line on how it went ("Saved 8 clips · 41 candidates…").
    pub summary: Option<String>,
    /// The raw result (a rendered file's path), for callers waiting on this task.
    pub output: Option<String>,
    pub error: Option<String>,
    pub created_at: i64,
    pub finished_at: Option<i64>,
    /// A chat turn's tool calls, for the Sonar turn UI.
    #[serde(default)]
    pub chat_events: Vec<ChatEvent>,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
    #[serde(skip)]
    started: Option<Instant>,
    /// When the running task was last saved to the runs table (see [`runs::LIVE_SAVE_EVERY`]).
    #[serde(skip)]
    saved: Option<Instant>,
}

impl Task {
    fn new(id: u64, kind: TaskKind, label: String) -> Self {
        Task {
            id,
            run_id: runs::new_run_id(),
            project_id: kind.project_id(),
            kind,
            label,
            state: TaskState::Queued,
            progress: None,
            note: None,
            steps: StepLog::new(),
            links: Vec::new(),
            summary: None,
            output: None,
            error: None,
            created_at: ghostreel_core::projects::now(),
            finished_at: None,
            chat_events: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            started: None,
            saved: None,
        }
    }

    /// Apply one progress report. Reports that arrive after the task ended are ignored: the
    /// outcome (summary, closed steps) wins.
    fn apply(&mut self, report: Report) {
        if self.state != TaskState::Running {
            return;
        }
        let elapsed = self.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
        match report {
            Report::Step { fraction, phase, note, indeterminate } => {
                self.steps.report(&phase, &note);
                if !note.is_empty() {
                    self.note = Some(note);
                }
                self.progress = Some(Progress::fraction(&phase, fraction, elapsed, indeterminate));
            }
            Report::Snapshot { progress, note } => {
                self.steps.report(&progress.phase, note.as_deref().unwrap_or(&progress.count_note()));
                if note.is_some() {
                    self.note = note;
                }
                self.progress = Some(progress);
            }
            Report::Note(note) => {
                self.steps.note(&note);
                self.note = Some(note);
            }
            Report::Chat { progress, note, events } => {
                self.steps.report(&progress.phase, "");
                self.note = Some(note);
                self.progress = Some(progress);
                self.chat_events = events;
            }
            Report::Event(event) => {
                let phase = self.progress.as_ref().map(|p| p.phase.clone()).unwrap_or_else(|| "run".into());
                self.steps.event(&phase, event);
            }
            Report::PhaseEvent { phase, event } => self.steps.event_in(&phase, event),
        }
    }

    /// The run to save now, when the task is project-scoped, running and was not saved for a
    /// while: other processes (the CLI, a reload) then see its live steps, not a stepless row.
    fn live_run(&mut self) -> Option<runs::Run> {
        if self.project_id.is_none() || self.state != TaskState::Running {
            return None;
        }
        if self.saved.is_some_and(|t| t.elapsed() < runs::LIVE_SAVE_EVERY) {
            return None;
        }
        self.saved = Some(Instant::now());
        Some(self.to_run())
    }

    /// Write how the task ended.
    fn finish(&mut self, result: Result<Outcome, String>) {
        let outcome = result.unwrap_or_else(Outcome::failed);
        self.state = outcome.state;
        // A failure's error closes the step it happened in; a partial run's reason is the run's.
        let note = if outcome.state == TaskState::Failed { outcome.error.as_deref().unwrap_or("") } else { "" };
        self.steps.finish(outcome.state, note);
        self.summary = outcome.summary;
        self.links = outcome.links;
        self.output = outcome.output;
        self.error = outcome.error;
        self.finished_at = Some(ghostreel_core::projects::now());
        if let Some(p) = &mut self.progress
            && matches!(self.state, TaskState::Done | TaskState::Partial)
        {
            p.fraction = 1.0;
            p.eta_secs = Some(0.0);
        }
    }

    fn to_run(&self) -> runs::Run {
        runs::Run {
            id: self.run_id.clone(),
            project_id: self.project_id,
            kind: self.kind.key().to_string(),
            label: self.label.clone(),
            state: self.state,
            steps: self.steps.steps().to_vec(),
            links: self.links.clone(),
            summary: self.summary.clone(),
            error: self.error.clone(),
            source: "app".into(),
            created_at: self.created_at,
            finished_at: self.finished_at,
        }
    }
}

/// One progress report from a running task (see [`Reporter`]).
enum Report {
    /// The producer's `(fraction, phase, note)`.
    Step { fraction: f64, phase: String, note: String, indeterminate: bool },
    /// A full snapshot (an index [`Tracker`]'s), with an optional status line.
    Snapshot { progress: Progress, note: Option<String> },
    /// A status line for the current step.
    Note(String),
    /// A chat turn's progress with its tool calls so far.
    Chat { progress: Progress, note: String, events: Vec<ChatEvent> },
    /// Something the current step keeps (a warning, an agent rerouted, the typed failure).
    Event(runs::StepEvent),
    /// Something `phase`'s step keeps (a stage postponed, a video that failed in it).
    PhaseEvent { phase: String, event: runs::StepEvent },
}

/// How a running task reports, whatever its kind. Cheap to clone and usable from synchronous
/// producer callbacks (and blocking threads): reports are queued and applied in order.
#[derive(Clone)]
pub struct Reporter {
    tx: mpsc::UnboundedSender<Report>,
}

impl Reporter {
    pub fn step(&self, fraction: f64, phase: &str, note: &str) {
        self.send(Report::Step { fraction, phase: phase.into(), note: note.into(), indeterminate: false });
    }

    /// A step with no measurable end (a model generating): the bar animates.
    pub fn working(&self, fraction: f64, phase: &str, note: &str) {
        self.send(Report::Step { fraction, phase: phase.into(), note: note.into(), indeterminate: true });
    }

    pub fn snapshot(&self, progress: Progress, note: Option<String>) {
        self.send(Report::Snapshot { progress, note });
    }

    pub fn note(&self, note: impl Into<String>) {
        self.send(Report::Note(note.into()));
    }

    /// Keep an event on the current step.
    pub fn event(&self, event: runs::StepEvent) {
        self.send(Report::Event(event));
    }

    fn chat(&self, progress: Progress, note: String, events: Vec<ChatEvent>) {
        self.send(Report::Chat { progress, note, events });
    }

    /// What the indexer (and its steadiness / faces / speakers passes) says, as reports.
    pub fn index_event(&self, e: &index::Event) {
        match e.run_report() {
            Some(index::EventReport::Progress(p)) => self.snapshot(p, None),
            Some(index::EventReport::Note(n)) => self.note(n),
            Some(index::EventReport::Event { phase, event }) => self.send(Report::PhaseEvent { phase, event }),
            None => {}
        }
    }

    fn send(&self, r: Report) {
        let _ = self.tx.send(r);
    }
}

type ChatSender = oneshot::Sender<Result<ChatTurnView, String>>;
type ChatRequest = (String, Vec<String>, Option<i64>, ChatSender);

#[derive(Default)]
pub struct Queue {
    tasks: Mutex<VecDeque<Task>>,
    next_id: std::sync::atomic::AtomicU64,
    wake: Notify,
    chat_requests: Mutex<HashMap<u64, ChatRequest>>,
}

/// Finished tasks kept for the Activity list (saved runs outlive them).
const HISTORY: usize = 30;

/// How long the worker waits for a task's last reports before writing its outcome.
const DRAIN: Duration = Duration::from_secs(3);

impl Queue {
    pub async fn snapshot(&self) -> Vec<Task> {
        self.tasks.lock().await.iter().cloned().collect()
    }

    /// Add a task unless an identical one is already waiting. Returns the task id.
    pub async fn enqueue(&self, app: &AppHandle, kind: TaskKind, label: String) -> u64 {
        if !matches!(kind, TaskKind::Chat { .. })
            && let Some(t) = self.tasks.lock().await.iter().find(|t| t.kind == kind && t.state == TaskState::Queued)
        {
            return t.id;
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.push(app, Task::new(id, kind, label)).await;
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn enqueue_chat(
        &self,
        app: &AppHandle,
        project_id: i64,
        session_id: i64,
        message: String,
        images: Vec<String>,
        label: String,
        script_id: Option<i64>,
        tx: oneshot::Sender<Result<ChatTurnView, String>>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        self.chat_requests.lock().await.insert(id, (message, images, script_id, tx));
        self.push(app, Task::new(id, TaskKind::Chat { project_id, session_id }, label)).await;
        id
    }

    async fn push(&self, app: &AppHandle, task: Task) {
        self.tasks.lock().await.push_back(task);
        self.wake.notify_one();
        self.emit(app).await;
    }

    pub async fn cancel(&self, app: &AppHandle, id: u64) -> bool {
        let mut tasks = self.tasks.lock().await;
        let Some(t) = tasks.iter_mut().find(|t| t.id == id) else { return false };
        match t.state {
            TaskState::Queued => {
                t.state = TaskState::Cancelled;
                t.finished_at = Some(ghostreel_core::projects::now());
                if let Some((_, _, _, tx)) = self.chat_requests.lock().await.remove(&id) {
                    let _ = tx.send(Err("task cancelled".into()));
                }
            }
            TaskState::Running => {
                t.cancel.store(true, Ordering::SeqCst);
                t.note = Some("Stopping…".into());
            }
            _ => return false,
        }
        drop(tasks);
        self.emit(app).await;
        true
    }

    pub async fn clear_finished(&self, app: &AppHandle) {
        self.tasks.lock().await.retain(|t| t.state.is_active());
        self.emit(app).await;
    }

    async fn emit(&self, app: &AppHandle) {
        let _ = app.emit("queue", self.snapshot().await);
    }

    async fn update(&self, app: &AppHandle, id: u64, f: impl FnOnce(&mut Task)) {
        {
            let mut tasks = self.tasks.lock().await;
            if let Some(t) = tasks.iter_mut().find(|t| t.id == id) {
                f(t);
            }
            // Trim history.
            let finished = tasks.iter().filter(|t| !t.state.is_active()).count();
            if finished > HISTORY {
                let mut drop_n = finished - HISTORY;
                tasks.retain(|t| {
                    if drop_n > 0 && !t.state.is_active() {
                        drop_n -= 1;
                        false
                    } else {
                        true
                    }
                });
            }
        }
        self.emit(app).await;
    }

    /// The one way a task reports progress, whatever its kind.
    async fn report(&self, app: &AppHandle, id: u64, report: Report) {
        let mut live = None;
        self.update(app, id, |t| {
            t.apply(report);
            live = t.live_run();
        })
        .await;
        if let Some(run) = live {
            let _ = save_run(&run);
        }
    }

    /// A [`Reporter`] for task `id`, and the forwarder applying its reports (it ends once every
    /// clone of the reporter is dropped).
    fn reporter(app: &AppHandle, id: u64) -> (Reporter, tauri::async_runtime::JoinHandle<()>) {
        let (tx, mut rx) = mpsc::unbounded_channel::<Report>();
        let app = app.clone();
        let forward = tauri::async_runtime::spawn(async move {
            while let Some(r) = rx.recv().await {
                app.state::<Queue>().report(&app, id, r).await;
            }
        });
        (Reporter { tx }, forward)
    }

    async fn next_queued(&self) -> Option<(u64, TaskKind, Arc<AtomicBool>)> {
        let mut tasks = self.tasks.lock().await;
        let t = tasks.iter_mut().find(|t| t.state == TaskState::Queued)?;
        t.state = TaskState::Running;
        t.started = Some(Instant::now());
        Some((t.id, t.kind.clone(), t.cancel.clone()))
    }

    async fn task(&self, id: u64) -> Option<Task> {
        self.tasks.lock().await.iter().find(|t| t.id == id).cloned()
    }
}

/// How a task ended, in the run's terms.
#[derive(Debug)]
struct Outcome {
    state: RunState,
    summary: Option<String>,
    links: Vec<RunLink>,
    output: Option<String>,
    error: Option<String>,
}

impl Default for Outcome {
    fn default() -> Self {
        Outcome { state: RunState::Done, summary: None, links: Vec::new(), output: None, error: None }
    }
}

impl Outcome {
    fn done(summary: impl Into<String>) -> Self {
        Outcome { summary: Some(summary.into()), ..Default::default() }
    }

    fn cancelled() -> Self {
        Outcome { state: RunState::Cancelled, ..Default::default() }
    }

    /// An index run (or one analysis pass) by how it ended: stopped, partial with the reason
    /// (a stage postponed, videos that failed), or done.
    fn from_index(s: &index::Summary) -> Self {
        Outcome { state: s.run_state(), summary: Some(s.line()), error: s.problems_line(), ..Default::default() }
    }

    fn failed(error: String) -> Self {
        Outcome { state: RunState::Failed, error: Some(error), ..Default::default() }
    }

    fn link(mut self, link: RunLink) -> Self {
        self.links.push(link);
        self
    }

    /// The task produced a file: its path is the output, and a file link.
    fn file(mut self, path: String) -> Self {
        self.links.push(RunLink::File { path: path.clone() });
        self.output = Some(path);
        self
    }
}

/// The single worker: runs queued tasks forever, in order.
pub async fn worker(app: AppHandle) {
    // Runs the last session left running were cut short when it closed.
    if let Ok(p) = Paths::resolve()
        && let Ok(db) = Db::open(&p.db_file())
    {
        let _ = db.interrupt_stale_runs("app");
    }
    loop {
        let queue = app.state::<Queue>();
        let Some((id, kind, cancel)) = queue.next_queued().await else {
            queue.wake.notified().await;
            continue;
        };
        queue.emit(&app).await;
        // Saved as running too, so a run cut short by closing the app still shows (as interrupted).
        if let Some(task) = queue.task(id).await
            && task.project_id.is_some()
        {
            let _ = save_run(&task.to_run());
        }
        let (rep, forward) = Queue::reporter(&app, id);
        let result = until_stopped(run_task(&app, id, kind, &rep, cancel.clone()), &cancel).await;
        // Let the task's last reports land before its outcome; anything later is ignored.
        drop(rep);
        let _ = tokio::time::timeout(DRAIN, forward).await;
        let queue = app.state::<Queue>();
        queue.update(&app, id, |t| t.finish(result)).await;
        if let Some(task) = queue.task(id).await
            && task.project_id.is_some()
            && let Err(e) = save_run(&task.to_run())
        {
            eprintln!("could not save run {}: {e}", task.run_id);
        }
        let _ = app.emit("task-finished", id);
    }
}

/// How long a stopped task gets to wind down on its own (keeping what it finished) before the
/// queue drops it. Every long loop checks the stop flag far more often than this.
const STOP_GRACE: Duration = Duration::from_millis(1500);

/// Run a task, never leaving a Stop unanswered: once stop is asked for, the task has
/// [`STOP_GRACE`] to end by itself; after that it is dropped — with the `kill_on_drop` child
/// processes it was waiting on — and counts as stopped, so the queue moves on whatever the task
/// was stuck in.
async fn until_stopped(
    task: impl std::future::Future<Output = Result<Outcome, String>>,
    cancel: &ghostreel_core::cancel::CancelFlag,
) -> Result<Outcome, String> {
    tokio::pin!(task);
    tokio::select! {
        r = &mut task => r,
        _ = ghostreel_core::cancel::wait(Some(cancel)) => {
            tokio::time::timeout(STOP_GRACE, &mut task).await.unwrap_or_else(|_| Ok(Outcome::cancelled()))
        }
    }
}

fn save_run(run: &runs::Run) -> Result<(), String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    db.save_run(run).map_err(|e| e.to_string())
}

async fn run_task(
    app: &AppHandle,
    id: u64,
    kind: TaskKind,
    rep: &Reporter,
    cancel: Arc<AtomicBool>,
) -> Result<Outcome, String> {
    match kind {
        TaskKind::Index { project_id } => {
            let summary = run_index(rep, project_id, cancel).await?;
            Ok(Outcome::from_index(&summary).link(RunLink::Project { id: project_id }))
        }
        TaskKind::Steadiness { project_id, force } => {
            run_analysis(rep, project_id, force, cancel, Analysis::Steadiness).await
        }
        TaskKind::Faces { project_id, force } => run_analysis(rep, project_id, force, cancel, Analysis::Faces).await,
        TaskKind::Speakers { project_id, force } => {
            run_analysis(rep, project_id, force, cancel, Analysis::Speakers).await
        }
        TaskKind::RenderPreview { script_id, burn_titles, burn_narration, burn_captions, normalize_audio, out } => {
            let opts = PreviewRequest { burn_titles, burn_narration, burn_captions, normalize_audio, out };
            run_preview(rep, script_id, opts, cancel).await
        }
        TaskKind::Export { script_id, format, path } => run_export(rep, script_id, &format, &path).await,
        TaskKind::Chat { project_id, session_id } => {
            let req = app.state::<Queue>().chat_requests.lock().await.remove(&id);
            let Some((message, images, script_id, tx)) = req else {
                return Err("chat request missing".into());
            };
            let res = run_chat(app, rep, project_id, session_id, message, images, script_id, cancel).await;
            let outcome = match &res {
                Ok(turn) => {
                    let mut o = Outcome::default();
                    if let Some(id) = turn.script_id {
                        o.summary = Some(format!("Saved script #{id}"));
                        o = o.link(RunLink::Script { id });
                    }
                    Ok(o)
                }
                // Stop is not a failure: report it as the cancellation it is.
                Err(e) if e.contains(ghostreel_core::chat::CANCELLED) => Ok(Outcome::cancelled()),
                Err(e) => Err(e.clone()),
            };
            let _ = tx.send(res);
            outcome
        }
        TaskKind::DownloadModel { model_id } => run_download_model(rep, &model_id, cancel).await,
        TaskKind::Clips { project_id, options } => run_clips(rep, project_id, options, cancel).await,
        TaskKind::ExportClips { set_id, dir, accepted_only } => {
            run_export_clips(rep, set_id, dir, accepted_only, cancel).await
        }
    }
}

/// Take the indexer lock, waiting (as a "waiting" step) while another process holds it. `None`
/// when the task was stopped meanwhile.
async fn index_lock(rep: &Reporter, p: &Paths, cancel: &AtomicBool) -> Result<Option<IndexLock>, String> {
    loop {
        match IndexLock::acquire(&p.data_dir) {
            Ok(l) => return Ok(Some(l)),
            Err(ghostreel_core::Error::Busy(_)) => {
                rep.working(0.0, "waiting", &waiting_note(p));
                // Checked often, so a Stop while waiting shows at once.
                for _ in 0..10 {
                    if cancel.load(Ordering::SeqCst) {
                        return Ok(None);
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// What the queue is waiting for while another process holds the indexer lock: that process's
/// live run when it records one (`ghostreel index` does), with the step it is on.
fn waiting_note(p: &Paths) -> String {
    let other =
        Db::open(&p.db_file()).ok().and_then(|db| db.active_runs("cli").ok()).and_then(|r| r.into_iter().next());
    match other {
        Some(run) => match run.steps.last() {
            Some(s) if !s.note.is_empty() => {
                format!("Waiting for the command line: {} · {} · {}", run.label, s.label, s.note)
            }
            Some(s) => format!("Waiting for the command line: {} · {}", run.label, s.label),
            None => format!("Waiting for the command line: {}", run.label),
        },
        None => "Waiting for another GhostReel process to finish indexing…".into(),
    }
}

async fn run_index(rep: &Reporter, project_id: i64, cancel: Arc<AtomicBool>) -> Result<index::Summary, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    // The CLI may be indexing: wait for it instead of failing.
    let Some(lock) = index_lock(rep, &p, &cancel).await? else {
        return Ok(index::Summary { cancelled: true, ..Default::default() });
    };
    let mut db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let rt = runtime::resolve(&p, &config).await.map_err(|e| e.to_string())?;
    let opts = index::Options { project_id: Some(project_id), cancel: Some(cancel), ..Default::default() };
    // The pipeline decides which stages run (the project's enabled stages) and how the run ended;
    // nothing is queued after it — a stage left pending runs on the next index.
    let result = index::run_pipeline(&mut db, &p, &config, &rt, &opts, |e| rep.index_event(&e)).await;
    drop(lock);
    result.map_err(|e| e.to_string())
}

/// The per-video analysis passes that can be re-run on their own from the Project menu.
#[derive(Clone, Copy)]
enum Analysis {
    Steadiness,
    Faces,
    Speakers,
}

/// "Measured camera steadiness on 3 of 5 videos" / "All 5 videos already … (up to date)".
fn measured_line(measured: usize, total: usize, all: &str, some: &str, up_to_date: &str) -> String {
    if measured == 0 {
        format!("All {total} videos {up_to_date} (up to date)")
    } else if measured == total {
        format!("{all} {measured} videos")
    } else {
        format!("{some} {measured} of {total} videos")
    }
}

async fn run_analysis(
    rep: &Reporter,
    project_id: i64,
    force: bool,
    cancel: Arc<AtomicBool>,
    what: Analysis,
) -> Result<Outcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    let Some(lock) = index_lock(rep, &p, &cancel).await? else { return Ok(Outcome::cancelled()) };
    let mut db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let rt = runtime::resolve(&p, &config).await.map_err(|e| e.to_string())?;
    let opts = index::Options { project_id: Some(project_id), cancel: Some(cancel.clone()), ..Default::default() };
    // The same problem bookkeeping as an index run: a pass with failed videos is partial.
    let mut problems = Vec::new();
    let mut sink = |e: index::Event| {
        index::StageProblem::observe(&mut problems, &e);
        rep.index_event(&e)
    };
    let summary = match what {
        Analysis::Steadiness => {
            let mut tracker = Tracker::new(&[("steadiness", 1.0)]);
            let s = index::run_steadiness_jobs(&mut db, &rt, &opts, force, &mut tracker, &mut sink)
                .await
                .map_err(|e| e.to_string())?;
            measured_line(
                s.measured,
                s.total,
                "Measured camera steadiness on all",
                "Measured camera steadiness on",
                "already measured",
            )
        }
        Analysis::Faces => {
            let mut tracker = Tracker::new(&[("faces", 1.0)]);
            let s = index::run_face_jobs(&mut db, &p, &config, &rt, &opts, force, &mut tracker, &mut sink)
                .await
                .map_err(|e| e.to_string())?;
            measured_line(
                s.measured,
                s.total,
                "Found and tracked faces on all",
                "Tracked faces on",
                "already tracked for faces",
            )
        }
        Analysis::Speakers => {
            let mut tracker = Tracker::new(&[("speakers", 1.0)]);
            let reports = index::run_speaker_jobs(&mut db, &p, &config, &rt, &opts, force, &mut tracker, &mut sink)
                .await
                .map_err(|e| e.to_string())?;
            speakers_line(&reports)
        }
    };
    drop(lock);
    if cancel.load(Ordering::SeqCst) {
        return Ok(Outcome::cancelled());
    }
    let outcome = index::Summary { problems, ..Default::default() };
    Ok(Outcome { summary: Some(summary), ..Outcome::from_index(&outcome) }.link(RunLink::Project { id: project_id }))
}

fn speakers_line(reports: &[(i64, ghostreel_core::speakers::SpeakersReport)]) -> String {
    match reports.first() {
        None => "Speakers not analysed (voice model unavailable)".to_string(),
        Some((_, r)) if r.up_to_date => "Speakers already up to date".to_string(),
        Some((_, r)) => {
            let linked = r.links.iter().filter(|l| l.person_id.is_some()).count();
            format!(
                "Found {} speaker{} and {} {}; linked {} voice{} to faces",
                r.speakers.len(),
                if r.speakers.len() == 1 { "" } else { "s" },
                r.people.people.len(),
                if r.people.people.len() == 1 { "person" } else { "people" },
                linked,
                if linked == 1 { "" } else { "s" },
            )
        }
    }
}

/// What a preview render burns in, and where it goes.
struct PreviewRequest {
    burn_titles: bool,
    burn_narration: bool,
    burn_captions: bool,
    normalize_audio: bool,
    out: Option<String>,
}

async fn run_preview(
    rep: &Reporter,
    script_id: i64,
    req: PreviewRequest,
    cancel: Arc<AtomicBool>,
) -> Result<Outcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let opts = ghostreel_core::preview::PreviewOptions {
        burn_titles: req.burn_titles,
        burn_narration: req.burn_narration,
        burn_captions: req.burn_captions,
        normalize_audio: req.normalize_audio,
        audio_fade_s: config.script.audio_fade_s,
        speech_overrun_s: config.script.speech_overrun_s,
        // A file the user asked for is a deliverable: the cut's own size. The in-app
        // preview (no file) stays light.
        size: if req.out.is_some() {
            ghostreel_core::preview::RenderSize::Full
        } else {
            ghostreel_core::preview::RenderSize::Preview
        },
        out: req.out.map(std::path::PathBuf::from),
        cancel: Some(cancel),
    };
    let rep = rep.clone();
    let res = tokio::task::spawn_blocking(move || {
        ghostreel_core::preview::render_preview(&db, &p.data_dir, &ffmpeg, script_id, &opts, |frac, msg| {
            rep.step(frac, "preview", msg)
        })
    })
    .await
    .map_err(|e| e.to_string())?;
    match res {
        Ok(r) => Ok(Outcome::done("Rendered the preview")
            .link(RunLink::Script { id: script_id })
            .file(r.path.to_string_lossy().to_string())),
        Err(ghostreel_core::Error::Preview(s)) if s == "cancelled" => Ok(Outcome::cancelled()),
        Err(e) => Err(e.to_string()),
    }
}

/// A clip run's progress and events, as task reports.
struct ClipSink<'a>(&'a Reporter);

impl runs::RunSink for ClipSink<'_> {
    fn progress(&mut self, frac: f64, phase: &str, note: &str) {
        // The orchestrator is one long model call: nothing to count.
        if phase == "orchestrator" { self.0.working(frac, phase, note) } else { self.0.step(frac, phase, note) }
    }

    fn event(&mut self, event: runs::StepEvent) {
        self.0.event(event);
    }
}

async fn run_clips(
    rep: &Reporter,
    project_id: i64,
    options: ghostreel_core::clips::ClipOptions,
    cancel: Arc<AtomicBool>,
) -> Result<Outcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    let backends = ghostreel_core::clips::Backends::resolve(&p, &config, &options).await.map_err(|e| e.to_string())?;
    rep.note(backends.describe());
    let mut db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let mut sink = ClipSink(rep);
    let res = ghostreel_core::clips::run(
        &mut db,
        project_id,
        &config.script,
        &backends,
        &options,
        Some(cancel.clone()),
        &mut sink,
    )
    .await;
    // A run that failed on its agents keeps the typed failure in its last step.
    if let Err(e) = &res
        && let Some(ev) = runs::StepEvent::from_error(e)
    {
        rep.event(ev);
    }
    match res {
        Ok(r) => {
            let n = r.set.items.len();
            let summary = format!("Saved {n} {} · {}", if n == 1 { "clip" } else { "clips" }, r.stats_line());
            Ok(Outcome::done(summary).link(RunLink::ClipSet { id: r.set_id().to_string() }))
        }
        Err(e) if e.to_string().contains(ghostreel_core::chat::CANCELLED) => Ok(Outcome::cancelled()),
        Err(e) => Err(e.to_string()),
    }
}

async fn run_export_clips(
    rep: &Reporter,
    set_id: String,
    dir: String,
    accepted_only: bool,
    cancel: Arc<AtomicBool>,
) -> Result<Outcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let ffmpeg = ghostreel_core::doctor::locate("ffmpeg").ok_or_else(|| "ffmpeg not found".to_string())?;
    let out_dir = dir.clone();
    let which = if accepted_only {
        ghostreel_core::clips::ExportWhich::Accepted
    } else {
        ghostreel_core::clips::ExportWhich::NotRejected
    };
    let rep = rep.clone();
    let set = set_id.clone();
    let res = tokio::task::spawn_blocking(move || {
        let db = Db::open(&p.db_file())?;
        ghostreel_core::clips::export_set(
            &db,
            &p.data_dir,
            &ffmpeg,
            &set,
            std::path::Path::new(&out_dir),
            which,
            &config.script,
            Some(cancel),
            |k, n, name| {
                let note = if k < n { format!("Rendering {}/{n}: {name}", k + 1) } else { "Done".into() };
                rep.step(k as f64 / n.max(1) as f64, "export", &note)
            },
        )
    })
    .await
    .map_err(|e| e.to_string())?;
    match res {
        Ok(files) => Ok(Outcome::done(format!(
            "Exported {} {} to {dir}",
            files.len(),
            if files.len() == 1 { "clip" } else { "clips" }
        ))
            .link(RunLink::ClipSet { id: set_id })
            .file(dir)),
        Err(ghostreel_core::Error::Preview(s)) if s == "cancelled" => Ok(Outcome::cancelled()),
        Err(e) if e.to_string().contains(ghostreel_core::chat::CANCELLED) => Ok(Outcome::cancelled()),
        Err(e) => Err(e.to_string()),
    }
}

async fn run_export(rep: &Reporter, script_id: i64, format_str: &str, out_path_str: &str) -> Result<Outcome, String> {
    use std::path::PathBuf;
    use std::str::FromStr;

    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    let format = ghostreel_core::export::ExportFormat::from_str(format_str).map_err(|e| e.to_string())?;
    let out_path = PathBuf::from(out_path_str);
    rep.working(0.0, "export_script", &format!("Exporting to {format}…"));
    let res =
        tokio::task::spawn_blocking(move || ghostreel_core::export::export_script(&db, script_id, format, &out_path))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
    Ok(Outcome::done(format!("Exported to {format}"))
        .link(RunLink::Script { id: script_id })
        .file(res.path.to_string_lossy().to_string()))
}

#[allow(clippy::too_many_arguments)]
async fn run_chat(
    app: &AppHandle,
    rep: &Reporter,
    project_id: i64,
    session_id: i64,
    message: String,
    images: Vec<String>,
    script_id: Option<i64>,
    cancel: Arc<AtomicBool>,
) -> Result<ChatTurnView, String> {
    // Everything the model is sent and says during this turn goes to the chat's "Full log".
    let _log = ghostreel_core::llmlog::session(session_id);
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).map_err(|e| e.to_string())?;
    let db = Db::open(&p.db_file()).map_err(|e| e.to_string())?;
    // The cut's agent choice (else the app's) answers, holding its endpoint's slot for the turn.
    let cut = ghostreel_core::chat::cut_agents(&db, script_id, Some(session_id));
    let lease = ghostreel_core::chat::lease_backend(&p, &config, None, cut.as_ref(), |w| rep.working(0.0, "waiting", w))
        .await
        .map_err(|e| e.to_string())?;
    let (backend, _slot, max_tool_rounds) = (lease.backend, lease.slot, lease.max_tool_rounds);
    let setup = runtime::resolve_embed(&p, &config).await;
    let embedder = runtime::start_embedder(&setup, |_, _| {}).await.ok();

    let mut ctx = ghostreel_core::chat::ChatContext {
        db,
        paths: p.clone(),
        backend,
        embedder,
        system_prompt: Some(config.chat.system_prompt.clone()),
        max_tool_rounds,
        script: config.script.clone(),
        jev: config.jev.clone(),
        cancel: Some(cancel.clone()),
    };

    // A chat turn has two halves and only one of them can be measured. Researching is countable —
    // tool rounds against the budget — and drafting is not: the model generates for minutes with
    // nothing to count, so the bar says so rather than inventing a number.
    let configured_budget = max_tool_rounds;
    let round_budget = configured_budget.max(1) as f64;
    let unlimited = configured_budget == 0;
    let mut rounds = 0u64;
    let started = Instant::now();
    let mut events_log: Vec<ChatEvent> = Vec::new();
    let mut on_event = move |event: ChatEvent| {
        if matches!(event, ChatEvent::ToolStarted { .. }) {
            rounds += 1;
        }
        let drafting = matches!(event, ChatEvent::Drafting | ChatEvent::Validating);
        // Research is capped at 0.8: the draft is still to come, and a bar that reaches the end
        // while the work continues is worse than one that stops short.
        let fraction = if drafting {
            0.85
        } else if unlimited {
            0.1
        } else {
            (rounds as f64 / round_budget).min(0.8)
        };
        let phase = if drafting { "drafting" } else { "researching" };
        let progress = Progress {
            phase_done: rounds,
            phase_total: round_budget as u64,
            ..Progress::fraction(phase, fraction, started.elapsed().as_secs_f64(), drafting || unlimited)
        };
        let note = match &event {
            ChatEvent::ToolStarted { tool, args } => {
                let s = serde_json::to_string(args).unwrap_or_default();
                // Char-based: args often hold non-ASCII queries ("configuración").
                let snippet =
                    if s.chars().count() > 40 { format!("{}…", s.chars().take(40).collect::<String>()) } else { s };
                if unlimited { format!("Researching · {rounds} lookups") } else { format!("{tool}: {snippet}") }
            }
            ChatEvent::ToolFinished { tool, summary } => format!("{tool}: {summary}"),
            ChatEvent::Drafting => "Drafting script".to_string(),
            ChatEvent::Validating => "Validating script".to_string(),
        };
        events_log.push(event.clone());
        let _ = app.emit("chat-progress", serde_json::json!({ "session_id": session_id, "event": event }));
        rep.chat(progress, note, events_log.clone());
    };

    if cancel.load(Ordering::SeqCst) {
        return Err("task cancelled".into());
    }

    let turn_res = ghostreel_core::chat::run_turn(
        &mut ctx,
        project_id,
        Some(session_id),
        &message,
        &images,
        script_id,
        &mut on_event,
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(ChatTurnView {
        session_id: turn_res.session_id,
        reply: turn_res.reply,
        script_id: turn_res.script_id,
        issues: turn_res.issues,
    })
}

fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else { format!("{:.0} MB", b / 1e6) }
}

async fn run_download_model(rep: &Reporter, model_id: &str, cancel: Arc<AtomicBool>) -> Result<Outcome, String> {
    let p = Paths::resolve().map_err(|e| e.to_string())?;
    let config = Config::load(&p.config_file).unwrap_or_default();
    let models_dir = ghostreel_core::models::effective_models_dir(&p, &config);

    let entry = ghostreel_core::models::find_entry(model_id);
    let whisper_spec = if entry.is_none() { ghostreel_core::models::whisper(model_id).ok() } else { None };

    if entry.is_none() && whisper_spec.is_none() {
        return Err(format!("unknown model '{model_id}'"));
    }

    let specs: Vec<(ghostreel_core::models::ModelSpec, u64)> =
        if let Some(ref e) = entry { e.downloads() } else { vec![(whisper_spec.unwrap(), 0)] };

    let total_size: u64 = specs.iter().map(|(_, size)| *size).sum();
    let mut completed_bytes = 0;
    let started = Instant::now();
    let mut last_path = String::new();
    for (spec, size_bytes) in specs {
        if cancel.load(Ordering::SeqCst) {
            return Ok(Outcome::cancelled());
        }

        let dest = models_dir.join(&spec.file_name);
        if dest.is_file()
            && dest
                .metadata()
                .map(|m| {
                    if entry.as_ref().is_some_and(|e| e.bundle.is_some()) { m.len() == size_bytes } else { m.len() > 0 }
                })
                .unwrap_or(false)
        {
            last_path = dest.to_string_lossy().to_string();
            completed_bytes += size_bytes;
            continue;
        }

        let size_str = if size_bytes > 0 { format!(" ({})", format_size(size_bytes)) } else { String::new() };
        let frac_so_far = if total_size > 0 { completed_bytes as f64 / total_size as f64 } else { 0.0 };
        rep.step(frac_so_far, "download_model", &format!("Downloading {}{size_str}", spec.file_name));

        let file_name = spec.file_name.clone();
        let mut last_emit: Option<Instant> = None;
        let on_bytes = |done: u64, total_opt: Option<u64>| {
            let file_total = total_opt.unwrap_or(size_bytes).max(1);
            let is_final = done >= file_total;
            if !is_final && last_emit.is_some_and(|t| t.elapsed() < Duration::from_millis(200)) {
                return;
            }
            last_emit = Some(Instant::now());
            let total = if total_size > 0 { total_size } else { file_total };
            let done = completed_bytes + done;
            let frac = (done as f64 / total as f64).clamp(0.0, 1.0);
            let elapsed = started.elapsed().as_secs_f64();
            let eta_secs = if frac > 0.01 && elapsed > 0.5 { Some((elapsed / frac) * (1.0 - frac)) } else { None };
            rep.snapshot(
                Progress {
                    phase_done: done,
                    phase_total: total,
                    eta_secs,
                    current: Some(std::path::PathBuf::from(&file_name)),
                    ..Progress::fraction("download_model", frac, elapsed, false)
                },
                None,
            );
        };
        let download_fut = ghostreel_core::models::download(&spec, &models_dir, on_bytes);

        let cancel_clone = cancel.clone();
        let cancel_fut = async {
            while !cancel_clone.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
        };

        tokio::select! {
            res = download_fut => {
                let path = res.map_err(|e| e.to_string())?;
                last_path = path.to_string_lossy().to_string();
            }
            _ = cancel_fut => return Ok(Outcome::cancelled()),
        };

        if entry.as_ref().is_some_and(|e| e.bundle.is_some()) && !dest.metadata().is_ok_and(|m| m.len() == size_bytes) {
            return Err(format!("incomplete MLX file {}", spec.file_name));
        }
        completed_bytes += size_bytes;
    }

    Ok(Outcome::done(format!("Downloaded {model_id}")).file(last_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(kind: TaskKind) -> Task {
        let mut t = Task::new(1, kind, "t".into());
        t.state = TaskState::Running;
        t.started = Some(Instant::now());
        t
    }

    fn step(fraction: f64, phase: &str, note: &str) -> Report {
        Report::Step { fraction, phase: phase.into(), note: note.into(), indeterminate: false }
    }

    #[test]
    fn reports_build_steps_and_progress_for_any_kind() {
        let mut t = running(TaskKind::DownloadModel { model_id: "m".into() });
        t.apply(step(0.1, "candidates", "Reading transcripts"));
        t.apply(Report::Note("Scored 1/3".into()));
        t.apply(step(0.5, "workers", "Scored 3/3"));
        assert_eq!(t.steps.steps().len(), 2);
        assert_eq!(t.steps.steps()[0].note, "Scored 1/3");
        let p = t.progress.as_ref().unwrap();
        assert_eq!((p.phase.as_str(), p.label.as_str()), ("workers", "Scoring candidates"));
        assert_eq!(t.note.as_deref(), Some("Scored 3/3"));
    }

    #[test]
    fn late_reports_do_not_overwrite_the_outcome() {
        let mut t = running(TaskKind::DownloadModel { model_id: "m".into() });
        t.apply(step(0.2, "workers", "Scored 1/3"));
        t.finish(Ok(Outcome::done("Saved 1 clip").link(RunLink::ClipSet { id: "cs1".into() })));
        t.apply(step(0.4, "save", "late"));
        assert_eq!(t.state, TaskState::Done);
        assert_eq!(t.steps.steps().len(), 1);
        assert_eq!(t.progress.as_ref().unwrap().fraction, 1.0);
        let run = t.to_run();
        assert_eq!(run.kind, "download_model");
        assert_eq!(run.links, vec![RunLink::ClipSet { id: "cs1".into() }]);
        assert_eq!(run.summary.as_deref(), Some("Saved 1 clip"));
    }

    /// Stop is answered even when the task ignores it: after the grace period the task is
    /// dropped and the queue moves on with it stopped.
    #[tokio::test]
    async fn a_stuck_task_is_dropped_after_stop() {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            flag.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let stuck = async {
            std::future::pending::<()>().await;
            Ok(Outcome::done("never"))
        };
        let r = until_stopped(stuck, &cancel).await.unwrap();
        assert_eq!(r.state, RunState::Cancelled);
        assert!(started.elapsed() < STOP_GRACE + Duration::from_secs(1), "{:?}", started.elapsed());

        // A task that winds down by itself within the grace keeps its own outcome.
        let cancel = Arc::new(AtomicBool::new(true));
        let quick = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(Outcome::done("kept what it finished"))
        };
        assert_eq!(until_stopped(quick, &cancel).await.unwrap().summary.as_deref(), Some("kept what it finished"));
    }

    #[test]
    fn a_partial_index_run_keeps_its_steps_and_gives_the_reason() {
        let mut t = running(TaskKind::Index { project_id: 1 });
        t.apply(step(0.5, "embed", "3 of 3"));
        let summary = index::Summary {
            problems: vec![index::StageProblem::Postponed {
                stage: "describe".into(),
                reason: "no vision projector".into(),
            }],
            ..Default::default()
        };
        t.finish(Ok(Outcome::from_index(&summary)));
        assert_eq!(t.state, TaskState::Partial);
        assert_eq!(t.steps.steps()[0].note, "3 of 3");
        assert_eq!(t.error.as_deref(), Some("Describing frames postponed: no vision projector"));
    }

    #[test]
    fn a_failure_closes_the_open_step_with_the_error() {
        let mut t = running(TaskKind::DownloadModel { model_id: "m".into() });
        t.apply(step(0.2, "orchestrator", "Choosing"));
        t.finish(Err("model crashed".into()));
        let s = &t.steps.steps()[0];
        assert_eq!((s.state, s.note.as_str()), (runs::StepState::Failed, "model crashed"));
        assert_eq!(t.error.as_deref(), Some("model crashed"));
    }
}
