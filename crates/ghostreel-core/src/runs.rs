//! Runs: one shape for every long-running job, whoever starts it (the app's queue, the CLI, an
//! agent) and whatever it does (indexing, clips, faces, a render…).
//!
//! A run is a label, a state, the [`Step`]s it went through, typed [`RunLink`]s to what it made
//! and a one-line summary. Producers don't know about runs: they keep reporting
//! `(fraction, phase, note)` through their own callbacks, and the caller feeds those into a
//! [`StepLog`] (or a [`RunRecorder`], which also keeps the run's other fields and saves it).
//! Finished runs are kept in the `runs` table so every front end can list them again later.
//!
//! [`phase_label`] is the single place a phase key ("workers", "transcribe_local") gets its human
//! name; front ends show the label that arrives with each step and progress snapshot.

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;

/// Persisted runs kept before the oldest are pruned.
pub const RUNS_KEPT: usize = 500;

/// Pseudo-phase a producer reports when it is finished ("done": "Saved 8 clips"). It closes the
/// open step instead of adding one.
pub const DONE_PHASE: &str = "done";

/// The human name of a phase key. Unknown keys are humanised ("export_clips" → "Export clips").
pub fn phase_label(phase: &str) -> String {
    let known = match phase {
        // Script chat.
        "researching" => "Reading the footage",
        "drafting" => "Writing the script",
        // Indexing.
        "hash" => "Reading new files",
        "probe" => "Reading video details",
        "download" => "Downloading speech model",
        "download_model" => "Downloading model",
        "download_embed" => "Downloading search model",
        "download_vision" => "Downloading vision model",
        "transcribe" | "transcribe_server" | "transcribe_local" => "Transcribing",
        "frames" => "Picking keyframes",
        "describe" | "describe_server" | "describe_local" => "Describing frames",
        "embed" => "Building the search index",
        "steadiness" => "Measuring camera shake",
        "faces" => "Finding faces",
        "speakers" => "Finding speakers",
        // Clips.
        "candidates" => "Finding candidate moments",
        "workers" => "Scoring candidates",
        "orchestrator" => "Choosing the best clips",
        "save" => "Checking and saving",
        // Rendering and export.
        "preview" => "Rendering preview",
        "export" => "Rendering clips",
        "export_script" => "Exporting",
        "waiting" => "Waiting",
        DONE_PHASE => "Done",
        _ => "",
    };
    if !known.is_empty() {
        return known.to_string();
    }
    let words = phase.replace(['_', '-'], " ");
    let mut chars = words.trim().chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Where a run is (and, once finished, how it ended).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Queued,
    Running,
    Done,
    /// Finished, but some of the work could not be done now (a stage postponed, videos that
    /// failed): what did run is kept, the rest stays pending for a later run.
    Partial,
    Failed,
    Cancelled,
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            RunState::Queued => "queued",
            RunState::Running => "running",
            RunState::Done => "done",
            RunState::Partial => "partial",
            RunState::Failed => "failed",
            RunState::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "queued" => RunState::Queued,
            "running" => RunState::Running,
            "done" => RunState::Done,
            "partial" => RunState::Partial,
            "cancelled" => RunState::Cancelled,
            _ => RunState::Failed,
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, RunState::Queued | RunState::Running)
    }

    /// How the open step ends when the run ends this way.
    fn step_state(self) -> StepState {
        match self {
            RunState::Done | RunState::Partial => StepState::Done,
            RunState::Failed => StepState::Failed,
            RunState::Cancelled => StepState::Stopped,
            RunState::Queued | RunState::Running => StepState::Running,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepState {
    Running,
    Done,
    /// Finished with problems worth a look (its warning events say which): a stage postponed,
    /// some videos failed.
    Warning,
    Failed,
    Stopped,
}

/// How much a step event matters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EventLevel {
    Info,
    Warning,
    Error,
}

/// Something a step wants kept beside its last note: a preflight warning, an agent rerouted or
/// resting, the typed failure that ended the run. Notes are overwritten as a step goes on; events
/// stay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepEvent {
    pub level: EventLevel,
    pub message: String,
    /// The typed agent failure behind it, for a kind chip and details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<crate::agents::AgentError>,
    pub at: i64,
}

impl StepEvent {
    pub fn new(level: EventLevel, message: impl Into<String>) -> Self {
        Self { level, message: message.into(), error: None, at: crate::projects::now() }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(EventLevel::Warning, message)
    }

    pub fn with_error(mut self, error: crate::agents::AgentError) -> Self {
        self.error = Some(error);
        self
    }

    /// The typed failure of a run that failed on an agent, as the event that ends it.
    pub fn from_error(e: &Error) -> Option<Self> {
        e.agent_error().map(|a| Self::new(EventLevel::Error, a.to_string()).with_error(a.clone()))
    }

    /// A preflight check as an event (`None` for one that passed).
    pub fn from_check(c: &crate::preflight::Check) -> Option<Self> {
        let level = match c.level {
            crate::preflight::Level::Ok => return None,
            crate::preflight::Level::Warn => EventLevel::Warning,
            crate::preflight::Level::Error => EventLevel::Error,
        };
        let ev = Self::new(level, c.message.clone());
        Some(match &c.error {
            Some(e) => ev.with_error(e.clone()),
            None => ev,
        })
    }
}

/// Where a producer reports: its `(fraction, phase, note)` progress, and the events worth
/// keeping. Any `FnMut(f64, &str, &str)` is a sink that drops events, so producers that only
/// report progress take a closure as before.
pub trait RunSink: Send {
    fn progress(&mut self, fraction: f64, phase: &str, note: &str);
    fn event(&mut self, _event: StepEvent) {}
}

impl<F: FnMut(f64, &str, &str) + Send> RunSink for F {
    fn progress(&mut self, fraction: f64, phase: &str, note: &str) {
        self(fraction, phase, note)
    }
}

/// One phase of a run with its last note ("Scoring candidates · Scored 7/7 windows").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Step {
    pub phase: String,
    pub label: String,
    pub note: String,
    pub state: StepState,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    /// What happened in it worth keeping (warnings, reroutes, the failure).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<StepEvent>,
}

/// A run's steps, built from `(phase, note)` reports: a new phase closes the open step with its
/// last note, the same phase updates that note.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct StepLog {
    steps: Vec<Step>,
}

impl StepLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    pub fn into_steps(self) -> Vec<Step> {
        self.steps
    }

    fn open_step(&mut self) -> Option<&mut Step> {
        self.steps.last_mut().filter(|s| s.state == StepState::Running)
    }

    /// A progress report. An empty note keeps the step's last one. [`DONE_PHASE`] closes the open
    /// step rather than adding one.
    pub fn report(&mut self, phase: &str, note: &str) {
        if phase.is_empty() {
            self.note(note);
            return;
        }
        if phase == DONE_PHASE {
            self.close(StepState::Done, "");
            return;
        }
        // One step per human phase: `describe` and `describe_server` are the same step.
        if let Some(open) = self.open_step()
            && (open.phase == phase || open.label == phase_label(phase))
        {
            if !note.is_empty() {
                open.note = note.to_string();
            }
            return;
        }
        self.close(StepState::Done, "");
        let now = crate::projects::now();
        self.steps.push(Step {
            phase: phase.to_string(),
            label: phase_label(phase),
            note: note.to_string(),
            state: StepState::Running,
            started_at: now,
            finished_at: None,
            events: Vec::new(),
        });
    }

    /// Keep an event on the step of `phase`: the open step when it is that phase (by label, so
    /// `describe_server` and `describe` are one step), else a new step for it. A stage that is
    /// postponed before it starts still gets its own step, instead of its warning landing on
    /// whatever ran before it.
    pub fn event_in(&mut self, phase: &str, event: StepEvent) {
        let same = self.open_step().is_some_and(|s| s.phase == phase || s.label == phase_label(phase));
        if !same {
            self.report(phase, "");
        }
        if let Some(step) = self.steps.last_mut() {
            step.events.push(event);
        }
    }

    /// Whether any step ended badly or kept a warning.
    pub fn has_problems(&self) -> bool {
        self.steps.iter().any(|s| {
            matches!(s.state, StepState::Warning | StepState::Failed)
                || s.events.iter().any(|e| e.level >= EventLevel::Warning)
        })
    }

    /// Keep an event on the open step (the last one when none is open). An event before any
    /// step opens one of its own phase, `phase`.
    pub fn event(&mut self, phase: &str, event: StepEvent) {
        if self.steps.is_empty() {
            self.report(phase, "");
        }
        if let Some(step) = self.steps.last_mut() {
            step.events.push(event);
        }
    }

    /// A status line for the open step without changing phase ("Finding faces: a.mp4").
    pub fn note(&mut self, note: &str) {
        if let Some(open) = self.open_step()
            && !note.is_empty()
        {
            open.note = note.to_string();
        }
    }

    /// The run ended: close the open step the way the run ended (`note` replaces its last note
    /// when not empty — the error of a failed run, say).
    pub fn finish(&mut self, state: RunState, note: &str) {
        self.close(state.step_state(), note);
    }

    fn close(&mut self, state: StepState, note: &str) {
        if state == StepState::Running {
            return;
        }
        if let Some(open) = self.open_step() {
            // A step that kept warnings did not simply finish.
            let warned = open.events.iter().any(|e| e.level >= EventLevel::Warning);
            open.state = if state == StepState::Done && warned { StepState::Warning } else { state };
            open.finished_at = Some(crate::projects::now());
            if !note.is_empty() {
                open.note = note.to_string();
            }
        }
    }
}

/// Something a run made or worked on, typed so a front end knows how to open it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunLink {
    ClipSet { id: String },
    Script { id: i64 },
    File { path: String },
    Video { id: i64 },
    Project { id: i64 },
}

/// Where a link's target stands now.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LinkState {
    /// There, and can be opened.
    Live,
    /// Deleted into the trash ([`crate::trash`]): it can be restored from entry `trash_id`.
    Trashed { trash_id: String },
    /// Gone for good (purged, its project deleted, a file moved).
    Gone,
}

impl RunLink {
    /// Whether what the link points at is still there, in the trash, or gone (a clip set, cut,
    /// video or project can be deleted after the run that made it; a file can be moved). Front
    /// ends offer Restore for a trashed target and show a gone one as such instead of a button
    /// that does nothing.
    pub fn state(&self, db: &Db) -> Result<LinkState, Error> {
        let row = |sql: &str, id: &dyn rusqlite::ToSql| -> Result<bool, Error> {
            Ok(db.conn.query_row(sql, [id], |r| r.get::<_, i64>(0)).optional()?.is_some())
        };
        // A trashable row: missing, live (no trash id) or trashed.
        let trashable = |sql: &str, id: &dyn rusqlite::ToSql| -> Result<LinkState, Error> {
            Ok(match db.conn.query_row(sql, [id], |r| r.get::<_, Option<String>>(0)).optional()? {
                None => LinkState::Gone,
                Some(None) => LinkState::Live,
                Some(Some(trash_id)) => LinkState::Trashed { trash_id },
            })
        };
        let live = |there: bool| if there { LinkState::Live } else { LinkState::Gone };
        Ok(match self {
            RunLink::ClipSet { id } => trashable("SELECT trash_id FROM clip_sets WHERE id = ?1", id)?,
            RunLink::Script { id } => trashable("SELECT trash_id FROM scripts WHERE id = ?1", id)?,
            RunLink::Video { id } => live(row("SELECT 1 FROM videos WHERE id = ?1", id)?),
            RunLink::Project { id } => live(row("SELECT 1 FROM projects WHERE id = ?1", id)?),
            RunLink::File { path } => live(std::path::Path::new(path).exists()),
        })
    }

    /// Whether the target can be opened now (there and not in the trash).
    pub fn exists(&self, db: &Db) -> Result<bool, Error> {
        Ok(self.state(db)? == LinkState::Live)
    }
}

/// [`RunLink::state`] for each link, in order.
pub fn links_state(db: &Db, links: &[RunLink]) -> Result<Vec<LinkState>, Error> {
    links.iter().map(|l| l.state(db)).collect()
}

/// A run as front ends list it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Run {
    /// Stable across restarts (the queue's task ids are not).
    pub id: String,
    pub project_id: Option<i64>,
    /// What kind of job ("clips", "index", "faces"…): the same key the queue's task kinds use.
    pub kind: String,
    pub label: String,
    pub state: RunState,
    pub steps: Vec<Step>,
    pub links: Vec<RunLink>,
    pub summary: Option<String>,
    pub error: Option<String>,
    /// Who ran it: "app" or "cli".
    pub source: String,
    pub created_at: i64,
    pub finished_at: Option<i64>,
}

/// A new run id: unique across processes (the app and the CLI write to the same table).
pub fn new_run_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("r{:x}{:x}{seq:x}", nanos & 0xffff_ffff_ffff_ffff, std::process::id())
}

/// Records one run for a producer that is not the app's queue (a CLI command, a tool): feed it the
/// producer's progress, finish it with what it made, save it.
#[derive(Debug, Clone)]
pub struct RunRecorder {
    run: Run,
    log: StepLog,
    fraction: f64,
    last_saved: Option<std::time::Instant>,
}

/// How often a running run is saved while it goes, so other processes (the app showing a CLI
/// run, the CLI listing the app's) see its live steps.
pub const LIVE_SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(2);

impl RunRecorder {
    pub fn start(kind: &str, label: impl Into<String>, project_id: Option<i64>, source: &str) -> Self {
        Self {
            run: Run {
                id: new_run_id(),
                project_id,
                kind: kind.to_string(),
                label: label.into(),
                state: RunState::Running,
                steps: Vec::new(),
                links: Vec::new(),
                summary: None,
                error: None,
                source: source.to_string(),
                created_at: crate::projects::now(),
                finished_at: None,
            },
            log: StepLog::new(),
            fraction: 0.0,
            last_saved: None,
        }
    }

    pub fn id(&self) -> &str {
        &self.run.id
    }

    /// The producer's `(fraction, phase, note)`.
    pub fn progress(&mut self, fraction: f64, phase: &str, note: &str) {
        self.fraction = fraction.clamp(0.0, 1.0);
        self.log.report(phase, note);
    }

    /// A status line for the current step.
    pub fn note(&mut self, note: &str) {
        self.log.note(note);
    }

    /// Keep an event on the current step.
    pub fn event(&mut self, event: StepEvent) {
        self.log.event("run", event);
    }

    /// Keep an event on `phase`'s step (see [`StepLog::event_in`]).
    pub fn event_in(&mut self, phase: &str, event: StepEvent) {
        self.log.event_in(phase, event);
    }

    /// Save the run if [`LIVE_SAVE_EVERY`] passed since the last save: call it as progress
    /// arrives. Errors are ignored — a missed live update is not worth stopping the work.
    pub fn save_live(&mut self, db: &Db) {
        if self.last_saved.is_some_and(|t| t.elapsed() < LIVE_SAVE_EVERY) {
            return;
        }
        self.last_saved = Some(std::time::Instant::now());
        let _ = self.save(db);
    }

    /// A full progress snapshot (an indexer's): its phase, with the count as the step's note.
    pub fn snapshot(&mut self, p: &crate::progress::Progress) {
        self.progress(p.fraction, &p.phase, &p.count_note());
    }

    pub fn fraction(&self) -> f64 {
        self.fraction
    }

    pub fn steps(&self) -> &[Step] {
        self.log.steps()
    }

    pub fn finish(&mut self, state: RunState, summary: Option<String>, links: Vec<RunLink>, error: Option<String>) {
        // A failure's error closes the step it happened in; a partial run's reason is the run's.
        let note = if state == RunState::Failed { error.as_deref().unwrap_or("") } else { "" };
        self.log.finish(state, note);
        self.run.state = state;
        self.run.summary = summary;
        self.run.links = links;
        self.run.error = error;
        self.run.finished_at = Some(crate::projects::now());
        if matches!(state, RunState::Done | RunState::Partial) {
            self.fraction = 1.0;
        }
    }

    /// Finish from a producer's result: `Ok` with its summary and links, or the error (a stop
    /// counts as cancelled when `is_cancel` says so).
    pub fn finish_with<T, E: std::fmt::Display>(
        &mut self,
        res: &Result<T, E>,
        ok: impl FnOnce(&T) -> (Option<String>, Vec<RunLink>),
        is_cancel: impl FnOnce(&E) -> bool,
    ) {
        match res {
            Ok(v) => {
                let (summary, links) = ok(v);
                self.finish(RunState::Done, summary, links, None);
            }
            Err(e) if is_cancel(e) => self.finish(RunState::Cancelled, None, Vec::new(), None),
            Err(e) => self.finish(RunState::Failed, None, Vec::new(), Some(e.to_string())),
        }
    }

    /// The run as it stands.
    pub fn run(&self) -> Run {
        Run { steps: self.log.steps().to_vec(), ..self.run.clone() }
    }

    pub fn save(&self, db: &Db) -> Result<(), Error> {
        db.save_run(&self.run())
    }
}

fn row_to_run(r: &rusqlite::Row) -> rusqlite::Result<Run> {
    let steps: String = r.get("steps_json")?;
    let links: String = r.get("links_json")?;
    let state: String = r.get("state")?;
    Ok(Run {
        id: r.get("id")?,
        project_id: r.get("project_id")?,
        kind: r.get("kind")?,
        label: r.get("label")?,
        state: RunState::parse(&state),
        steps: serde_json::from_str(&steps).unwrap_or_default(),
        links: serde_json::from_str(&links).unwrap_or_default(),
        summary: r.get("summary")?,
        error: r.get("error")?,
        source: r.get("source")?,
        created_at: r.get("created_at")?,
        finished_at: r.get("finished_at")?,
    })
}

impl Db {
    /// Insert or replace a run, then prune the oldest beyond [`RUNS_KEPT`].
    pub fn save_run(&self, run: &Run) -> Result<(), Error> {
        self.conn.execute(
            "INSERT INTO runs(id, project_id, kind, label, state, source, steps_json, links_json, summary, error,
                              created_at, finished_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(id) DO UPDATE SET project_id = excluded.project_id, kind = excluded.kind,
                 label = excluded.label, state = excluded.state, source = excluded.source,
                 steps_json = excluded.steps_json, links_json = excluded.links_json, summary = excluded.summary,
                 error = excluded.error, created_at = excluded.created_at, finished_at = excluded.finished_at",
            params![
                run.id,
                run.project_id,
                run.kind,
                run.label,
                run.state.as_str(),
                run.source,
                serde_json::to_string(&run.steps).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&run.links).unwrap_or_else(|_| "[]".into()),
                run.summary,
                run.error,
                run.created_at,
                run.finished_at,
            ],
        )?;
        self.conn.execute(
            "DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?1)",
            [RUNS_KEPT as i64],
        )?;
        Ok(())
    }

    /// Runs `source` left marked running (it was closed or crashed mid-run): mark them failed.
    /// Call once at start-up, before that source starts new runs. Returns how many.
    pub fn interrupt_stale_runs(&self, source: &str) -> Result<usize, Error> {
        let now = crate::projects::now();
        let mut stmt =
            self.conn.prepare("SELECT * FROM runs WHERE source = ?1 AND state IN ('running', 'queued')")?;
        let stale = stmt.query_map([source], row_to_run)?.collect::<Result<Vec<_>, _>>()?;
        for mut run in stale.iter().cloned() {
            let mut log = StepLog { steps: run.steps };
            log.finish(RunState::Failed, "");
            run.steps = log.into_steps();
            run.state = RunState::Failed;
            run.error = Some("Interrupted: GhostReel closed before it finished".into());
            run.finished_at = Some(now);
            self.save_run(&run)?;
        }
        Ok(stale.len())
    }

    /// The runs `source` has going now (running or queued), newest first.
    pub fn active_runs(&self, source: &str) -> Result<Vec<Run>, Error> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM runs WHERE source = ?1 AND state IN ('running', 'queued') ORDER BY created_at DESC, rowid DESC",
        )?;
        let rows = stmt.query_map([source], row_to_run)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn run(&self, id: &str) -> Result<Option<Run>, Error> {
        Ok(self.conn.query_row("SELECT * FROM runs WHERE id = ?1", [id], row_to_run).optional()?)
    }

    /// A project's runs, newest first.
    pub fn runs_for_project(&self, project_id: i64, limit: usize) -> Result<Vec<Run>, Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM runs WHERE project_id = ?1 ORDER BY created_at DESC, rowid DESC LIMIT ?2")?;
        let rows = stmt.query_map(params![project_id, limit as i64], row_to_run)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Every project's runs (and those of no project), newest first.
    pub fn recent_runs(&self, limit: usize) -> Result<Vec<Run>, Error> {
        let mut stmt = self.conn.prepare("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?1")?;
        let rows = stmt.query_map([limit as i64], row_to_run)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phases(log: &StepLog) -> Vec<(&str, &str, StepState)> {
        log.steps().iter().map(|s| (s.phase.as_str(), s.note.as_str(), s.state)).collect()
    }

    #[test]
    fn links_report_whether_their_target_still_exists() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'p', 0);
                 INSERT INTO clip_sets(id, project_id, created_at) VALUES ('cs1', 1, 5);",
            )
            .unwrap();
        let file = std::env::temp_dir().join(format!("ghostreel-link-{}", new_run_id()));
        std::fs::write(&file, b"x").unwrap();
        let links = vec![
            RunLink::ClipSet { id: "cs1".into() },
            RunLink::ClipSet { id: "gone".into() },
            RunLink::Project { id: 1 },
            RunLink::Script { id: 99 },
            RunLink::Video { id: 99 },
            RunLink::File { path: file.to_string_lossy().into() },
            RunLink::File { path: "/nonexistent/ghostreel/file.mp4".into() },
        ];
        let live = |states: Vec<LinkState>| states.into_iter().map(|s| s == LinkState::Live).collect::<Vec<_>>();
        assert_eq!(live(links_state(&db, &links).unwrap()), vec![true, false, true, false, false, true, false]);
        db.conn.execute("UPDATE clip_sets SET trash_id = 't1' WHERE id = 'cs1'", []).unwrap();
        assert_eq!(links[0].state(&db).unwrap(), LinkState::Trashed { trash_id: "t1".into() });
        assert!(!links[0].exists(&db).unwrap(), "a trashed target can't be opened");
        db.conn.execute("DELETE FROM clip_sets WHERE id = 'cs1'", []).unwrap();
        assert_eq!(links[0].state(&db).unwrap(), LinkState::Gone);
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn steps_keep_each_phase_with_its_last_note() {
        let mut log = StepLog::new();
        log.report("candidates", "Reading transcripts");
        log.report("workers", "Scored 1/7 windows");
        log.report("workers", "Scored 7/7 windows");
        log.report("workers", "");
        log.report("save", "Checking and saving clips");
        assert_eq!(
            phases(&log),
            vec![
                ("candidates", "Reading transcripts", StepState::Done),
                ("workers", "Scored 7/7 windows", StepState::Done),
                ("save", "Checking and saving clips", StepState::Running),
            ]
        );
        assert_eq!(log.steps()[1].label, "Scoring candidates");
        assert!(log.steps()[0].finished_at.is_some());
        log.report(DONE_PHASE, "Saved 8 clips");
        assert_eq!(log.steps().len(), 3, "done closes the open step instead of adding one");
        assert_eq!(log.steps()[2].state, StepState::Done);
    }

    #[test]
    fn finish_closes_the_open_step_the_way_the_run_ended() {
        let mut log = StepLog::new();
        log.report("faces", "a.mp4");
        log.note("Finding faces: b.mp4");
        log.finish(RunState::Failed, "model missing");
        assert_eq!(phases(&log), vec![("faces", "model missing", StepState::Failed)]);

        let mut log = StepLog::new();
        log.report("workers", "Scored 2/7");
        log.finish(RunState::Cancelled, "");
        assert_eq!(phases(&log), vec![("workers", "Scored 2/7", StepState::Stopped)]);
        // A closed log ignores notes.
        log.note("late");
        assert_eq!(log.steps()[0].note, "Scored 2/7");
    }

    #[test]
    fn a_stage_warning_gets_its_own_step_and_a_partial_run_keeps_the_notes() {
        let mut log = StepLog::new();
        log.report("probe", "1 of 1");
        log.event_in("describe", StepEvent::warning("Describing frames postponed: no vision projector"));
        log.report("describe_server", "");
        log.report("embed", "3 of 3");
        log.finish(RunState::Partial, "");
        assert_eq!(
            phases(&log),
            vec![
                ("probe", "1 of 1", StepState::Done),
                ("describe", "", StepState::Warning),
                ("embed", "3 of 3", StepState::Done),
            ]
        );
        assert!(log.has_problems());

        let mut rec = RunRecorder::start("index", "Index", None, "cli");
        rec.progress(0.5, "embed", "3 of 3");
        rec.finish(RunState::Partial, Some("0 new".into()), vec![], Some("Describing frames postponed: x".into()));
        let run = rec.run();
        assert_eq!((run.state, run.steps[0].note.as_str()), (RunState::Partial, "3 of 3"), "the reason is the run's");
        assert_eq!(RunState::parse("partial"), RunState::Partial);
        assert!(!RunState::Partial.is_active());
    }

    #[test]
    fn active_runs_are_listed_per_source() {
        let db = Db::open_in_memory().unwrap();
        let mut live = RunRecorder::start("index", "Index “P”", None, "cli");
        live.progress(0.2, "faces", "12 of 600");
        live.save_live(&db);
        live.progress(0.3, "faces", "18 of 600");
        live.save_live(&db); // within LIVE_SAVE_EVERY: not saved again
        let active = db.active_runs("cli").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].steps[0].note, "12 of 600");
        assert!(db.active_runs("app").unwrap().is_empty());
    }

    #[test]
    fn labels_come_from_one_table() {
        assert_eq!(phase_label("transcribe_local"), "Transcribing");
        assert_eq!(phase_label("orchestrator"), "Choosing the best clips");
        assert_eq!(phase_label("steadiness"), "Measuring camera shake");
        assert_eq!(phase_label("export_clips"), "Export clips");
        assert_eq!(phase_label(""), "");
    }

    #[test]
    fn links_are_tagged_snake_case() {
        let links = vec![
            RunLink::ClipSet { id: "cs1".into() },
            RunLink::Script { id: 4 },
            RunLink::File { path: "/tmp/a.mp4".into() },
        ];
        let json = serde_json::to_string(&links).unwrap();
        assert_eq!(
            json,
            r#"[{"kind":"clip_set","id":"cs1"},{"kind":"script","id":4},{"kind":"file","path":"/tmp/a.mp4"}]"#
        );
        let back: Vec<RunLink> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, links);
    }

    #[test]
    fn recorder_saves_and_lists_runs() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'P', 0)", []).unwrap();
        let mut rec = RunRecorder::start("clips", "Generate 1 clip", Some(1), "cli");
        rec.progress(0.1, "candidates", "Reading transcripts");
        rec.progress(0.5, "workers", "Scored 3/3");
        rec.finish(RunState::Done, Some("Saved 1 clip".into()), vec![RunLink::ClipSet { id: "cs9".into() }], None);
        rec.save(&db).unwrap();

        let other = RunRecorder::start("download_model", "Download", None, "app");
        other.save(&db).unwrap();

        let runs = db.runs_for_project(1, 10).unwrap();
        assert_eq!(runs.len(), 1);
        let r = &runs[0];
        assert_eq!((r.kind.as_str(), r.state, r.source.as_str()), ("clips", RunState::Done, "cli"));
        assert_eq!(r.links, vec![RunLink::ClipSet { id: "cs9".into() }]);
        assert_eq!(r.steps.len(), 2);
        assert!(r.steps.iter().all(|s| s.state == StepState::Done));
        assert_eq!(db.recent_runs(10).unwrap().len(), 2);
        assert_eq!(db.run(rec.id()).unwrap().unwrap().summary.as_deref(), Some("Saved 1 clip"));
    }

    #[test]
    fn finish_with_maps_results() {
        let mut rec = RunRecorder::start("faces", "Faces", None, "cli");
        let res: Result<usize, String> = Err("stopped: cancelled".into());
        rec.finish_with(&res, |_| (None, vec![]), |e| e.contains("cancelled"));
        assert_eq!(rec.run().state, RunState::Cancelled);
        let res: Result<usize, String> = Err("boom".into());
        rec.finish_with(&res, |_| (None, vec![]), |_| false);
        assert_eq!(rec.run().error.as_deref(), Some("boom"));
    }

    #[test]
    fn prunes_to_the_cap() {
        let db = Db::open_in_memory().unwrap();
        for i in 0..(RUNS_KEPT + 5) {
            let mut r = RunRecorder::start("index", format!("run {i}"), None, "app").run();
            r.created_at = i as i64;
            db.save_run(&r).unwrap();
        }
        let n: i64 = db.conn.query_row("SELECT count(*) FROM runs", [], |r| r.get(0)).unwrap();
        assert_eq!(n as usize, RUNS_KEPT);
        assert_eq!(db.recent_runs(1).unwrap()[0].label, format!("run {}", RUNS_KEPT + 4));
    }

    #[test]
    fn stale_running_runs_are_marked_interrupted() {
        let db = Db::open_in_memory().unwrap();
        let mut rec = RunRecorder::start("steadiness", "Shake", None, "app");
        rec.progress(0.1, "steadiness", "a.mp4");
        rec.save(&db).unwrap();
        RunRecorder::start("clips", "CLI still going", None, "cli").save(&db).unwrap();
        assert_eq!(db.interrupt_stale_runs("app").unwrap(), 1);
        let r = db.run(rec.id()).unwrap().unwrap();
        assert_eq!(r.state, RunState::Failed);
        assert_eq!(r.steps[0].state, StepState::Failed);
        assert!(r.error.unwrap().starts_with("Interrupted"));
        assert_eq!(db.recent_runs(10).unwrap().iter().filter(|r| r.state == RunState::Running).count(), 1);
    }

    #[test]
    fn project_delete_takes_its_runs() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (2, 'Q', 0)", []).unwrap();
        RunRecorder::start("index", "Index", Some(2), "app").save(&db).unwrap();
        db.conn.execute("DELETE FROM projects WHERE id = 2", []).unwrap();
        assert!(db.runs_for_project(2, 10).unwrap().is_empty());
    }
}
