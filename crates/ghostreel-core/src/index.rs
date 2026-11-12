//! Indexing: scan watched folders, then run the persisted, resumable job pipeline (plan §3).
//!
//! Every video walks the stages in [`STAGES`] order. Job rows live in the database, so a crash
//! or quit resumes where it stopped; `running` rows found at startup are reset to `pending`.
//! Stages so far: `probe` (M1), `transcribe` (M2); frames / describe / embed follow.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::{OptionalExtension, params};
use serde::Serialize;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::Error;
use crate::db::Db;
use crate::media::{self, MediaInfo};
use crate::progress::{Progress, Tracker};
use crate::projects::now;
use crate::runtime::{EmbedSetup, Runtime, SttSetup, VisionSetup};
use crate::stt::{self, Engine};

/// Pipeline stages in execution order.
pub const STAGES: &[&str] = &["probe", "transcribe", "frames", "describe", "embed"];

/// All indexing stages including post-index analysis.
pub const ALL_STAGES: &[&str] =
    &["probe", "transcribe", "frames", "describe", "embed", "steadiness", "faces", "speakers"];

/// The stage whose completion a stage waits for.
fn prerequisite(stage: &str) -> Option<&'static str> {
    match stage {
        "probe" => None,
        "describe" => Some("frames"),
        _ => Some("probe"),
    }
}

/// A failed job is retried on later runs until it has failed this many times.
pub const MAX_ATTEMPTS: i64 = 3;

const PARALLEL_HASH: usize = 4;
const PARALLEL_PROBE: usize = 4;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    ScanFolder {
        path: PathBuf,
    },
    FolderMissing {
        path: PathBuf,
    },
    Scanned {
        new: usize,
        changed: usize,
        unchanged: usize,
        removed: usize,
    },
    JobStarted {
        video_id: i64,
        stage: String,
        path: PathBuf,
    },
    JobDone {
        video_id: i64,
        stage: String,
    },
    JobFailed {
        video_id: i64,
        stage: String,
        error: String,
    },
    /// Which backend a stage uses in this run (e.g. "GhostPen @ http://127.0.0.1:8771").
    StageBackend {
        stage: String,
        backend: String,
    },
    /// A stage can't run now; its jobs stay pending for a later run.
    StageUnavailable {
        stage: String,
        reason: String,
    },
    DownloadingModel {
        file: String,
    },
    /// A stage is waiting on something outside it (a free slot on a busy model endpoint, a model
    /// loading): said so the run never sits silent.
    Waiting {
        stage: String,
        what: String,
    },
    /// Files in watched folders that GhostReel rendered itself (previews, exports) and that were
    /// left out of the library instead of indexed as footage.
    RendersSkipped {
        count: usize,
    },
    /// Overall progress and time remaining (throttled to ~4 per second).
    Progress(Progress),
}

/// What an indexer event means for whoever keeps a run's log (`runs`): a progress snapshot, or a
/// status line for the current step.
#[derive(Debug, Clone, PartialEq)]
pub enum EventReport {
    Progress(Progress),
    Note(String),
    /// Something to keep on `phase`'s step: a stage postponed, a video that failed.
    Event {
        phase: String,
        event: crate::runs::StepEvent,
    },
}

impl Event {
    /// The event as a run report; `None` for events a run log has no use for.
    pub fn run_report(&self) -> Option<EventReport> {
        let label = crate::runs::phase_label;
        let file = |p: &Path| p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        Some(match self {
            Event::Progress(p) => EventReport::Progress(p.clone()),
            Event::StageBackend { stage, backend } => EventReport::Note(format!("{}: {backend}", label(stage))),
            Event::StageUnavailable { stage, reason } => EventReport::Event {
                phase: stage.clone(),
                event: crate::runs::StepEvent::warning(format!("{} postponed: {reason}", label(stage))),
            },
            Event::JobFailed { video_id, stage, error } => EventReport::Event {
                phase: stage.clone(),
                event: crate::runs::StepEvent::warning(format!("Video #{video_id}: {error}")),
            },
            Event::DownloadingModel { file } => EventReport::Note(format!("Downloading {file} (once)…")),
            Event::Waiting { stage, what } => EventReport::Note(format!("{}: waiting for {what}", label(stage))),
            Event::RendersSkipped { count } => EventReport::Event {
                phase: "hash".into(),
                event: crate::runs::StepEvent::new(
                    crate::runs::EventLevel::Info,
                    format!(
                        "Skipped {count} {} GhostReel rendered itself (previews, exports): not footage",
                        if *count == 1 { "file" } else { "files" }
                    ),
                ),
            },
            // Per-file passes say which file they are on; the main stages are too quick for that.
            Event::JobStarted { stage, path, .. } if matches!(stage.as_str(), "steadiness" | "faces" | "speakers") => {
                EventReport::Note(format!("{}: {}", label(stage), file(path)))
            }
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Summary {
    pub new: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub jobs_done: usize,
    pub jobs_failed: usize,
    /// Files skipped because they were modified less than `settle_secs` ago (still copying).
    pub unsettled: usize,
    /// Files skipped because GhostReel rendered them (a preview or export saved into a watched
    /// folder): they are its own output, not footage.
    pub renders: usize,
    /// The run was stopped early by [`Options::cancel`].
    pub cancelled: bool,
    /// Stages that could not do all their work: postponed (backend unavailable) or with videos
    /// that failed. A run with problems is [`RunState::Partial`](crate::runs::RunState), not done.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<StageProblem>,
}

/// How a stage fell short in a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageProblem {
    /// The stage could not run now (its backend is unavailable); its work stays pending.
    Postponed { stage: String, reason: String },
    /// Some videos failed in it.
    Failed { stage: String, videos: usize, error: String },
}

impl StageProblem {
    pub fn stage(&self) -> &str {
        match self {
            StageProblem::Postponed { stage, .. } | StageProblem::Failed { stage, .. } => stage,
        }
    }

    /// "Describing frames postponed: …" / "Finding faces failed on 2 videos: …".
    pub fn message(&self) -> String {
        let label = crate::runs::phase_label(self.stage());
        match self {
            StageProblem::Postponed { reason, .. } => format!("{label} postponed: {reason}"),
            StageProblem::Failed { videos, error, .. } => {
                format!("{label} failed on {videos} video{}: {error}", if *videos == 1 { "" } else { "s" })
            }
        }
    }

    /// Fold an indexer event into a run's problems.
    pub fn observe(problems: &mut Vec<StageProblem>, e: &Event) {
        // Phase keys name a stage's backend ("describe_server"); problems are per stage.
        let stage_of = |s: &str| s.split('_').next().unwrap_or(s).to_string();
        match e {
            Event::StageUnavailable { stage, reason } => {
                let stage = stage_of(stage);
                if !problems.iter().any(|p| matches!(p, StageProblem::Postponed { stage: s, .. } if *s == stage)) {
                    problems.push(StageProblem::Postponed { stage, reason: reason.clone() });
                }
            }
            Event::JobFailed { stage, error, .. } => {
                let stage = stage_of(stage);
                match problems.iter_mut().find(|p| matches!(p, StageProblem::Failed { stage: s, .. } if *s == stage)) {
                    Some(StageProblem::Failed { videos, .. }) => *videos += 1,
                    _ => problems.push(StageProblem::Failed { stage, videos: 1, error: error.clone() }),
                }
            }
            _ => {}
        }
    }
}

impl Summary {
    /// How the run ended, in a run's terms: stopped, finished short (problems), or done.
    pub fn run_state(&self) -> crate::runs::RunState {
        use crate::runs::RunState;
        if self.cancelled {
            RunState::Cancelled
        } else if self.problems.is_empty() {
            RunState::Done
        } else {
            RunState::Partial
        }
    }

    /// The reason a partial run gives, one line per problem; `None` when there were none.
    pub fn problems_line(&self) -> Option<String> {
        (!self.problems.is_empty())
            .then(|| self.problems.iter().map(StageProblem::message).collect::<Vec<_>>().join("; "))
    }

    /// "0 new · 1 changed · 0 removed · 3 steps done · 1 failed".
    pub fn line(&self) -> String {
        let mut line = format!(
            "{} new · {} changed · {} removed · {} steps done",
            self.new, self.changed, self.removed, self.jobs_done
        );
        if self.jobs_failed > 0 {
            line.push_str(&format!(" · {} failed", self.jobs_failed));
        }
        line
    }
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Limit scanning and jobs to one project's folders.
    pub project_id: Option<i64>,
    /// Retry jobs that exhausted their attempts.
    pub retry_failed: bool,
    /// Skip files modified within this many seconds (a copy in progress would hash garbage).
    /// The watcher uses ~10 s and re-runs later; one-shot `index` uses 0.
    pub settle_secs: i64,
    /// Set to stop the run at the next safe point (between videos/frames); unfinished work stays
    /// pending for a later run.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Explicit stages to run (overrides project/default configuration).
    pub stages: Option<Vec<String>>,
}

impl Options {
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// [`stage_enabled`] for this run's explicit stages.
    pub fn is_stage_enabled(&self, stage: &str, project_pipeline: Option<&crate::projects::PipelineConfig>) -> bool {
        stage_enabled(stage, self.stages.as_deref(), project_pipeline)
    }
}

/// Whether `stage` is to run: the explicit list a caller gave (`ghostreel index --stages`), else
/// the project's pipeline, else every stage. The one gate the index pipeline, its follow-up
/// analysis passes and the Project menu's stage status all ask.
pub fn stage_enabled(
    stage: &str,
    explicit: Option<&[String]>,
    project_pipeline: Option<&crate::projects::PipelineConfig>,
) -> bool {
    match (explicit, project_pipeline) {
        (Some(stages), _) => stages.iter().any(|s| s == stage),
        (None, Some(pipeline)) => pipeline.is_enabled(stage),
        (None, None) => true,
    }
}

/// Exclusive indexer lock (`<data>/indexer.lock`): the app and the CLI may both be open, but only
/// one of them runs the pipeline. Released when dropped.
pub struct IndexLock {
    _file: File,
}

impl IndexLock {
    pub fn acquire(data_dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(data_dir).map_err(|e| Error::Io(data_dir.to_path_buf(), e))?;
        let path = data_dir.join("indexer.lock");
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| Error::Io(path.clone(), e))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(std::fs::TryLockError::WouldBlock) => Err(Error::Busy(path.display().to_string())),
            Err(std::fs::TryLockError::Error(e)) => Err(Error::Io(path, e)),
        }
    }
}

/// Default speeds for a machine that has never indexed (units per second); replaced by measured
/// rates after the first runs.
///
/// `download` is bytes/s; `transcribe_*` are seconds of audio per second.
const PHASES: &[(&str, f64)] = &[
    ("hash", 40.0),
    ("probe", 15.0),
    ("download", 20_000_000.0),
    ("transcribe_server", 40.0),
    ("transcribe_local", 20.0),
    ("frames", 40.0),
    ("download_vision", 20_000_000.0),
    ("describe_server", 0.2),
    ("describe_local", 0.2),
    ("download_embed", 20_000_000.0),
    ("embed", 30.0),
];

/// Assumed length of a video whose duration isn't known yet (for the first estimate only).
const UNKNOWN_DURATION_S: f64 = 120.0;

/// Scan + run all pending jobs. Callers hold an [`IndexLock`].
///
/// Emits [`Event::Progress`] (throttled) with overall completion and time remaining.
pub async fn run(db: &mut Db, rt: &Runtime, opts: &Options, on_event: impl FnMut(Event)) -> Result<Summary, Error> {
    let result = run_jobs(db, rt, opts, on_event).await;
    // Done with the models: a GhostPen service frees them now, as a local helper exits.
    crate::ghost_service::release_after_run(rt).await;
    result
}

/// The whole indexing pipeline for a scope: the scan and per-video stages ([`run`]), then the
/// analysis passes (camera shake, faces, speakers) that [`stage_enabled`] lets through. The one
/// sequence the app's queue, `ghostreel index` and the MCP tool run, so they agree on what an
/// index run is and how it ended.
///
/// Every stage reports problems through its events; they are collected into
/// [`Summary::problems`] (a run with any is partial, not done) and a postponed stage's reason is
/// kept on its pending jobs for the Project menu. A stop anywhere sets [`Summary::cancelled`].
/// Callers hold an [`IndexLock`].
pub async fn run_pipeline(
    db: &mut Db,
    paths: &crate::paths::Paths,
    config: &crate::config::Config,
    rt: &Runtime,
    opts: &Options,
    mut on_event: impl FnMut(Event),
) -> Result<Summary, Error> {
    let mut problems = Vec::new();
    let mut sink = |e: Event| {
        StageProblem::observe(&mut problems, &e);
        on_event(e)
    };
    let mut summary = run(db, rt, opts, &mut sink).await?;
    if !summary.cancelled && !opts.cancelled() {
        let pipeline = opts.project_id.and_then(|pid| db.project(pid).ok()).map(|p| p.pipeline);
        let enabled = |stage: &str| opts.is_stage_enabled(stage, pipeline.as_ref()) && !opts.cancelled();
        // A pass that errors out as a whole (its model would not load) is a postponed stage, not
        // a reason to hide what the other passes did.
        let failed = |stage: &str, e: Error, sink: &mut dyn FnMut(Event)| {
            if !e.is_cancelled() {
                sink(Event::StageUnavailable { stage: stage.into(), reason: e.to_string() });
            }
        };
        if enabled("steadiness") {
            let mut tracker = Tracker::new(&[("steadiness", 1.0)]);
            if let Err(e) = run_steadiness_jobs(db, rt, opts, false, &mut tracker, &mut sink).await {
                failed("steadiness", e, &mut sink);
            }
        }
        if enabled("faces") {
            let mut tracker = Tracker::new(&[("faces", 1.0)]);
            if let Err(e) = run_face_jobs(db, paths, config, rt, opts, false, &mut tracker, &mut sink).await {
                failed("faces", e, &mut sink);
            }
        }
        if enabled("speakers") {
            let mut tracker = Tracker::new(&[("speakers", 1.0)]);
            if let Err(e) = run_speaker_jobs(db, paths, config, rt, opts, false, &mut tracker, &mut sink).await {
                failed("speakers", e, &mut sink);
            }
        }
    }
    summary.cancelled |= opts.cancelled();
    note_postponed(db, opts, &problems)?;
    summary.problems = problems;
    Ok(summary)
}

/// Keep a postponed stage's reason on its pending jobs in scope, so the Project menu can say why
/// the stage is still pending. Starting the job again clears it.
fn note_postponed(db: &Db, opts: &Options, problems: &[StageProblem]) -> Result<(), Error> {
    for p in problems {
        let StageProblem::Postponed { stage, reason } = p else { continue };
        if !STAGES.contains(&stage.as_str()) {
            continue;
        }
        db.conn.execute(
            "UPDATE jobs SET last_error = ?1
              WHERE stage = ?2 AND state = 'pending'
                AND (?3 IS NULL OR video_id IN (SELECT vf.video_id FROM video_files vf
                                                  JOIN project_folders pf ON pf.folder_id = vf.folder_id
                                                 WHERE pf.project_id = ?3))",
            params![format!("postponed: {reason}"), stage, opts.project_id],
        )?;
    }
    Ok(())
}

/// Record how an analysis pass (camera shake, faces) went for a video as its job row, so the
/// status and the next run know the video was analysed even when there was nothing to find —
/// a video with no faces stores no face tracks, and without this it was analysed again on every
/// run and shown as pending for ever.
fn record_analysis(db: &Db, video_id: i64, stage: &str, error: Option<&str>) -> Result<(), Error> {
    let (state, bump) = if error.is_some() { ("failed", 1) } else { ("done", 0) };
    db.conn.execute(
        "INSERT INTO jobs(video_id, stage, state, attempts, last_error, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(video_id, stage) DO UPDATE SET state = excluded.state, last_error = excluded.last_error,
             attempts = CASE WHEN excluded.state = 'done' THEN 0 ELSE jobs.attempts + 1 END,
             updated_at = excluded.updated_at",
        params![video_id, stage, state, bump, error, now()],
    )?;
    Ok(())
}

/// SQL: video `v` still needs analysis pass `stage` (`?` placeholders bound by the caller are not
/// used: the stage is a constant). `output` is the table its results live in.
fn needs_analysis_sql(stage: &str, output: &str, retry_failed: bool) -> String {
    let max_attempts = if retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    format!(
        "NOT EXISTS (SELECT 1 FROM {output} o WHERE o.video_id = v.id)
         AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.video_id = v.id AND j.stage = '{stage}'
                            AND (j.state = 'done' OR (j.state = 'failed' AND j.attempts >= {max_attempts})))"
    )
}

/// SQL: the video in column `video` has been through analysis pass `stage` — it has results in
/// `output`, or its job row says it was done (with nothing to find).
fn analysed_sql(stage: &str, output: &str, video: &str) -> String {
    format!(
        "(EXISTS (SELECT 1 FROM {output} o WHERE o.video_id = {video})
          OR EXISTS (SELECT 1 FROM jobs aj WHERE aj.video_id = {video} AND aj.stage = '{stage}' AND aj.state = 'done'))"
    )
}

async fn run_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    mut on_event: impl FnMut(Event),
) -> Result<Summary, Error> {
    let mut summary = Summary::default();
    let mut tracker = Tracker::new(PHASES);
    tracker.load_rates(db)?;

    // 1. Walk every folder first, so the run's totals are known before the slow work starts.
    let mut pending = Vec::new();
    for folder in db.folders(opts.project_id)?.into_iter().filter(|f| f.enabled) {
        on_event(Event::ScanFolder { path: folder.path.clone() });
        if !folder.path.is_dir() {
            // Unplugged drive / network share: keep the index, don't treat files as deleted.
            on_event(Event::FolderMissing { path: folder.path.clone() });
            continue;
        }
        let w = walk_folder(db, folder.id, &folder.path, folder.recursive, opts.settle_secs).await?;
        summary.unchanged += w.unchanged;
        summary.unsettled += w.unsettled;
        pending.push(w);
    }

    reset_interrupted(db)?;
    ensure_jobs(db)?;
    sync_pipeline_jobs(db, opts.project_id)?;

    let project_pipeline = opts.project_id.and_then(|pid| db.project(pid).ok()).map(|p| p.pipeline);
    let stage_active = |stage: &str| -> bool { opts.is_stage_enabled(stage, project_pipeline.as_ref()) };

    let to_hash: Vec<ToHash> = pending.iter_mut().flat_map(|w| std::mem::take(&mut w.to_hash)).collect();
    let hash_work = to_hash.iter().filter(|f| f.known_hash.is_none()).count() as u64;
    let already_queued = claimable_jobs(db, "probe", opts)?.len() as u64;
    tracker.set_total("hash", hash_work);
    // Upper bound until hashing tells us which files are genuinely new content.
    if stage_active("probe") {
        tracker.set_total("probe", already_queued + to_hash.len() as u64);
    }
    if rt.frames.is_some() && stage_active("frames") {
        let (known, unknown) = stage_work(db, "frames", opts)?;
        tracker.set_total("frames", (known + (unknown as f64 + to_hash.len() as f64) * UNKNOWN_DURATION_S) as u64);
    }
    let stt_phase = stt_phase(&rt.stt);
    if let Some(phase) = stt_phase
        && stage_active("transcribe")
    {
        let (known, unknown) = transcribe_work(db, opts)?;
        tracker.set_total(phase, (known + (unknown as f64 + to_hash.len() as f64) * UNKNOWN_DURATION_S) as u64);
    }

    // 2. Hash new/changed files (all folders together).
    tracker.start("hash");
    on_event(Event::Progress(tracker.snapshot(None)));
    let (new, changed, renders) = hash_and_store(db, to_hash, &mut tracker, &mut on_event).await?;
    summary.new = new;
    summary.changed = changed;
    // Renders indexed before they were recognised leave the library too.
    summary.renders = renders + forget_indexed_renders(db)?;
    if summary.renders > 0 {
        on_event(Event::RendersSkipped { count: summary.renders });
    }
    for w in &pending {
        summary.removed += db
            .conn
            .execute("DELETE FROM video_files WHERE folder_id = ?1 AND last_seen < ?2", params![w.folder_id, w.seq])?;
    }
    on_event(Event::Scanned {
        new: summary.new,
        changed: summary.changed,
        unchanged: summary.unchanged,
        removed: summary.removed,
    });

    if new > 0 || changed > 0 {
        sync_pipeline_jobs(db, opts.project_id)?;
    }

    // 3. Jobs, stage by stage.
    if stage_active("probe") {
        if opts.cancelled() {
            summary.cancelled = true;
            return Ok(summary);
        }
        let (done, failed) = run_probe_jobs(db, &rt.ffprobe, opts, &mut tracker, &mut on_event).await?;
        summary.jobs_done += done;
        summary.jobs_failed += failed;
    }

    if stage_active("transcribe") {
        if opts.cancelled() {
            summary.cancelled = true;
            return Ok(summary);
        }
        let (done, failed) = run_transcribe_jobs(db, rt, opts, &mut tracker, &mut on_event).await?;
        summary.jobs_done += done;
        summary.jobs_failed += failed;
    }

    if stage_active("frames") {
        if opts.cancelled() {
            summary.cancelled = true;
            return Ok(summary);
        }
        let (done, failed) = run_frame_jobs(db, rt, opts, &mut tracker, &mut on_event).await?;
        summary.jobs_done += done;
        summary.jobs_failed += failed;
    }

    if stage_active("describe") {
        if opts.cancelled() {
            summary.cancelled = true;
            return Ok(summary);
        }
        let (done, failed) = run_describe_jobs(db, rt, opts, &mut tracker, &mut on_event).await?;
        summary.jobs_done += done;
        summary.jobs_failed += failed;
    }

    if stage_active("embed") {
        if opts.cancelled() {
            summary.cancelled = true;
            return Ok(summary);
        }
        let (done, failed) = run_embed_jobs(db, rt, opts, &mut tracker, &mut on_event).await?;
        summary.jobs_done += done;
        summary.jobs_failed += failed;
    }

    // Auto-classify indexed videos that do not have manual category overrides.
    if let Some(pid) = opts.project_id {
        if let Err(error) = crate::classify::classify_project_videos(db, pid) {
            tracing::warn!(%error, "automatic classification failed");
        }
    } else if let Ok(mut st) =
        db.conn.prepare("SELECT id FROM videos WHERE COALESCE(category_source, 'auto') <> 'manual'")
        && let Ok(vids) = st.query_map([], |r| r.get::<_, i64>(0)).map(|m| m.filter_map(Result::ok).collect::<Vec<_>>())
    {
        for vid in vids {
            if let Ok(res) = crate::classify::classify_video(db, vid)
                && let Err(error) =
                    db.set_video_category(vid, &res.category, res.confidence, "auto", res.tag.as_deref())
            {
                tracing::warn!(%error, video_id = vid, "automatic classification update failed");
            }
        }
    }

    tracker.finish();
    tracker.save_rates(db)?;
    on_event(Event::Progress(tracker.snapshot(None)));
    Ok(summary)
}

struct ToHash {
    folder_id: i64,
    seq: i64,
    path: PathBuf,
    size: i64,
    mtime: i64,
    existed: bool,
    /// Identity already known through an overlapping folder (no hashing needed).
    known_hash: Option<String>,
}

struct Walked {
    folder_id: i64,
    seq: i64,
    unchanged: usize,
    unsettled: usize,
    to_hash: Vec<ToHash>,
}

fn next_scan_seq(db: &Db) -> Result<i64, Error> {
    let cur: Option<String> =
        db.conn.query_row("SELECT value FROM meta WHERE key = 'scan_seq'", [], |r| r.get(0)).optional()?;
    let next = cur.and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) + 1;
    db.conn.execute(
        "INSERT INTO meta(key, value) VALUES ('scan_seq', ?1) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [next.to_string()],
    )?;
    Ok(next)
}

fn mtime_secs(m: &std::fs::Metadata) -> i64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// List a folder's video files; mark unchanged/unsettled ones as seen and return the rest.
async fn walk_folder(
    db: &mut Db,
    folder_id: i64,
    root: &Path,
    recursive: bool,
    settle_secs: i64,
) -> Result<Walked, Error> {
    let seq = next_scan_seq(db)?;
    let mut w = Walked { folder_id, seq, unchanged: 0, unsettled: 0, to_hash: Vec::new() };

    // Walk (blocking IO) off the async threads.
    let root_owned = root.to_path_buf();
    let files: Vec<(PathBuf, i64, i64)> = tokio::task::spawn_blocking(move || {
        let walker = walkdir::WalkDir::new(&root_owned).max_depth(if recursive { usize::MAX } else { 1 });
        walker
            .into_iter()
            .filter_entry(|e| {
                e.depth() == 0
                    || if e.file_type().is_dir() {
                        !media::is_ignored_dir(&e.file_name().to_string_lossy())
                    } else {
                        !media::is_ignored_file(&e.file_name().to_string_lossy())
                    }
            })
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file() && media::is_video_path(e.path()))
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((e.into_path(), m.len() as i64, mtime_secs(&m)))
            })
            .collect()
    })
    .await
    .map_err(|e| Error::Invalid(format!("scan task failed: {e}")))?;

    let known: HashMap<String, (i64, i64)> = {
        let mut st = db.conn.prepare("SELECT path, size, mtime FROM video_files WHERE folder_id = ?1")?;
        st.query_map([folder_id], |r| Ok((r.get::<_, String>(0)?, (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))))?
            .collect::<Result<_, _>>()?
    };

    let settled_before = now() - settle_secs;
    let tx = db.conn.transaction()?;
    for (path, size, mtime) in files {
        let key = path.to_string_lossy().to_string();
        match known.get(&key) {
            Some(&(s, m)) if s == size && m == mtime => {
                tx.execute(
                    "UPDATE video_files SET last_seen = ?1 WHERE folder_id = ?2 AND path = ?3",
                    params![seq, folder_id, key],
                )?;
                w.unchanged += 1;
            }
            _ if settle_secs > 0 && mtime > settled_before => {
                // Still being written: keep any existing row alive, look again later.
                tx.execute(
                    "UPDATE video_files SET last_seen = ?1 WHERE folder_id = ?2 AND path = ?3",
                    params![seq, folder_id, key],
                )?;
                w.unsettled += 1;
            }
            prev => {
                // Same file already hashed through an overlapping folder: reuse its identity.
                let known_hash: Option<String> = tx
                    .query_row(
                        "SELECT v.content_hash FROM video_files vf JOIN videos v ON v.id = vf.video_id
                          WHERE vf.path = ?1 AND vf.size = ?2 AND vf.mtime = ?3 LIMIT 1",
                        params![key, size, mtime],
                        |r| r.get(0),
                    )
                    .optional()?;
                w.to_hash.push(ToHash { folder_id, seq, path, size, mtime, existed: prev.is_some(), known_hash });
            }
        }
    }
    tx.commit()?;
    Ok(w)
}

/// Remember that GhostReel wrote `path` (a preview or an export render), by content hash, so a
/// scan of a folder it lands in skips it. `script_id` is the cut it was rendered from.
pub fn record_render(db: &Db, path: &Path, script_id: Option<i64>) -> Result<(), Error> {
    let hash = media::content_hash(path)?;
    db.conn.execute(
        "INSERT INTO rendered_outputs(content_hash, script_id, path, created_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(content_hash) DO UPDATE SET script_id = excluded.script_id, path = excluded.path,
             created_at = excluded.created_at",
        params![hash, script_id, path.to_string_lossy(), now()],
    )?;
    Ok(())
}

/// Whether a file is GhostReel's own render: its content was recorded by [`record_render`], or
/// it sits where a recorded preview/export render was written (renders made before the hashes
/// were kept).
fn is_render(conn: &rusqlite::Connection, content_hash: &str, path: &Path) -> Result<bool, Error> {
    Ok(conn
        .query_row(
            "SELECT 1 WHERE EXISTS (SELECT 1 FROM rendered_outputs WHERE content_hash = ?1)
                         OR EXISTS (SELECT 1 FROM exports WHERE format = 'preview_mp4' AND path = ?2)",
            params![content_hash, path.to_string_lossy()],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

/// Drop the file locations of renders indexed as footage before they were recognised (their
/// index stays, unused, like any file no folder holds). Returns how many.
fn forget_indexed_renders(db: &Db) -> Result<usize, Error> {
    Ok(db.conn.execute(
        "DELETE FROM video_files
          WHERE video_id IN (SELECT v.id FROM videos v JOIN rendered_outputs r ON r.content_hash = v.content_hash)
             OR path IN (SELECT path FROM exports WHERE format = 'preview_mp4')",
        [],
    )?)
}

/// Hash files concurrently and record videos, file locations and their pending jobs. Files that
/// are GhostReel's own renders are left out. Returns (new, changed, renders skipped).
async fn hash_and_store(
    db: &mut Db,
    files: Vec<ToHash>,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize, usize), Error> {
    let sem = Arc::new(Semaphore::new(PARALLEL_HASH));
    let mut set = JoinSet::new();
    for mut f in files {
        let sem = sem.clone();
        set.spawn(async move {
            if let Some(h) = f.known_hash.take() {
                return (f, false, Ok(h));
            }
            let _permit = sem.acquire_owned().await;
            let p = f.path.clone();
            let hash = tokio::task::spawn_blocking(move || media::content_hash(&p))
                .await
                .unwrap_or_else(|e| Err(Error::Invalid(format!("hash task failed: {e}"))));
            (f, true, hash)
        });
    }
    let (mut new, mut changed, mut renders) = (0, 0, 0);
    while let Some(joined) = set.join_next().await {
        let Ok((f, hashed, hash)) = joined else { continue };
        if hashed {
            tracker.advance("hash", 1);
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(f.path.clone()))));
            }
        }
        // A file that vanished or can't be read mid-scan is skipped; next scan retries it.
        let Ok(hash) = hash else { continue };
        if is_render(&db.conn, &hash, &f.path)? {
            renders += 1;
            continue;
        }
        let tx = db.conn.transaction()?;
        tx.execute("INSERT OR IGNORE INTO videos(content_hash, size) VALUES (?1, ?2)", params![hash, f.size])?;
        let video_id: i64 = tx.query_row("SELECT id FROM videos WHERE content_hash = ?1", [&hash], |r| r.get(0))?;
        tx.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(folder_id, path) DO UPDATE SET video_id = excluded.video_id,
                 size = excluded.size, mtime = excluded.mtime, last_seen = excluded.last_seen",
            params![video_id, f.folder_id, f.path.to_string_lossy(), f.size, f.mtime, f.seq],
        )?;
        for stage in STAGES {
            tx.execute(
                "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) VALUES (?1, ?2, 'pending', ?3)",
                params![video_id, stage, now()],
            )?;
        }
        tx.commit()?;
        if f.existed { changed += 1 } else { new += 1 }
    }
    Ok((new, changed, renders))
}

/// Videos indexed before a stage existed get its job rows now.
fn ensure_jobs(db: &Db) -> Result<(), Error> {
    for stage in STAGES {
        db.conn.execute(
            "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) SELECT id, ?1, 'pending', ?2 FROM videos",
            params![stage, now()],
        )?;
    }
    Ok(())
}

fn reset_interrupted(db: &Db) -> Result<(), Error> {
    db.conn.execute("UPDATE jobs SET state = 'pending', updated_at = ?1 WHERE state = 'running'", [now()])?;
    Ok(())
}

/// Keep job states in sync with project pipeline settings.
/// When a stage is disabled for all projects watching a video, pending/failed jobs become `skipped`.
/// When a stage is enabled for a project, previously `skipped` jobs for its videos become `pending`.
pub fn sync_pipeline_jobs(db: &Db, project_id: Option<i64>) -> Result<(), Error> {
    let projects = db.projects()?;
    let project_map: HashMap<i64, &crate::projects::Project> = projects.iter().map(|p| (p.id, p)).collect();

    // Map each video in scope to the project ids that watch it.
    let mut video_projects: HashMap<i64, Vec<i64>> = HashMap::new();
    {
        let sql = "
            SELECT vf.video_id, pf.project_id
              FROM video_files vf
              JOIN project_folders pf ON pf.folder_id = vf.folder_id
             WHERE (?1 IS NULL OR vf.video_id IN (
                 SELECT vf2.video_id FROM video_files vf2
                 JOIN project_folders pf2 ON pf2.folder_id = vf2.folder_id
                 WHERE pf2.project_id = ?1
             ))
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id AND x.role = 'removed')";
        let mut st = db.conn.prepare(sql)?;
        let rows = st.query_map([project_id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
        for row in rows {
            let (vid, pid) = row?;
            video_projects.entry(vid).or_default().push(pid);
        }
    }

    let timestamp = now();
    for &stage in STAGES {
        let mut to_skip = Vec::new();
        let mut to_unskip = Vec::new();

        for (&vid, pids) in &video_projects {
            let any_enabled = pids.iter().any(|pid| project_map.get(pid).is_none_or(|p| p.pipeline.is_enabled(stage)));
            if any_enabled {
                to_unskip.push(vid);
            } else {
                to_skip.push(vid);
            }
        }

        for chunk in to_skip.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "UPDATE jobs SET state = 'skipped', updated_at = ?1
                  WHERE stage = ?2 AND state IN ('pending', 'failed') AND video_id IN ({placeholders})"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(chunk.len() + 2);
            params.push(&timestamp);
            params.push(&stage);
            for id in chunk {
                params.push(id);
            }
            db.conn.execute(&sql, rusqlite::params_from_iter(params))?;
        }

        for chunk in to_unskip.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = if stage == "transcribe" {
                format!(
                    "UPDATE jobs SET state = 'pending', attempts = 0, last_error = NULL, updated_at = ?1
                      WHERE stage = ?2 AND state = 'skipped'
                        AND video_id IN (SELECT id FROM videos WHERE COALESCE(has_audio, 1) != 0)
                        AND video_id IN ({placeholders})"
                )
            } else {
                format!(
                    "UPDATE jobs SET state = 'pending', attempts = 0, last_error = NULL, updated_at = ?1
                      WHERE stage = ?2 AND state = 'skipped' AND video_id IN ({placeholders})"
                )
            };
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(chunk.len() + 2);
            params.push(&timestamp);
            params.push(&stage);
            for id in chunk {
                params.push(id);
            }
            db.conn.execute(&sql, rusqlite::params_from_iter(params))?;
        }
    }
    Ok(())
}

/// Reset `from_stage` and all later stages (in [`STAGES`] order) back to `pending` for the
/// videos of `project_id` (or all videos when `None`).
///
/// For the `frames` stage:
/// - Deletes `frames` rows (and their related FTS / embed chunks via cascade / explicit delete).
/// - Deletes the JPEG files on disk (`<data_dir>/frames/…`).
///
/// For `describe`/`embed`: existing rows are replaced when the stage re-runs, so no pre-deletion
/// is needed — `store_frames` + `store_chunks` both start with a DELETE.
pub fn reset_stages(
    db: &Db,
    data_dir: &std::path::Path,
    project_id: Option<i64>,
    from_stage: &str,
) -> Result<(), Error> {
    // Find the index of from_stage in STAGES.
    let start_idx = STAGES
        .iter()
        .position(|&s| s == from_stage)
        .ok_or_else(|| Error::Invalid(format!("unknown stage '{from_stage}'; valid stages: {}", STAGES.join(", "))))?;

    // Collect video ids in scope.
    let video_ids: Vec<i64> = if let Some(pid) = project_id {
        let mut st = db.conn.prepare(
            "SELECT DISTINCT j.video_id FROM jobs j
              JOIN video_files vf ON vf.video_id = j.video_id
              JOIN project_folders pf ON pf.folder_id = vf.folder_id
             WHERE pf.project_id = ?1
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id AND x.role = 'removed')",
        )?;
        st.query_map([pid], |r| r.get(0))?.collect::<Result<_, _>>()?
    } else {
        let mut st = db.conn.prepare("SELECT id FROM videos")?;
        st.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?
    };

    for &stage in &STAGES[start_idx..] {
        if stage == "frames" {
            // Collect frame file paths before deleting the rows.
            for &vid in &video_ids {
                let paths: Vec<String> = {
                    let mut st = db.conn.prepare("SELECT thumb_path FROM frames WHERE video_id = ?1")?;
                    st.query_map([vid], |r| r.get(0))?.collect::<Result<_, _>>()?
                };
                for rel in paths {
                    let abs = data_dir.join(&rel);
                    let _ = std::fs::remove_file(&abs);
                }
                // Delete frame directory if now empty.
                let hash: Option<String> =
                    db.conn.query_row("SELECT content_hash FROM videos WHERE id = ?1", [vid], |r| r.get(0)).ok();
                if let Some(h) = hash {
                    let dir = data_dir.join(frames_rel_dir(&h));
                    let _ = std::fs::remove_dir(&dir); // only removes if empty
                }
                // Delete the frame rows themselves; their descriptions go with them.
                db.conn.execute("DELETE FROM frames WHERE video_id = ?1", [vid])?;
                // Chunks / embeddings derived from frames will be regenerated by embed.
            }
        }
        if stage == "describe" {
            // The describe stage only looks at frames with no description yet, so resetting the
            // job rows alone made `--redo describe` a silent no-op: it reported "96 jobs ok" in
            // eighteen seconds and changed nothing. Clearing the output is what makes the work
            // exist again. `--redo frames` deletes the rows outright and does not need this.
            for &vid in &video_ids {
                db.conn.execute(
                    "UPDATE frames SET description_json = NULL, visible_text = NULL WHERE video_id = ?1",
                    [vid],
                )?;
            }
        }
        // Reset job state for this stage for all in-scope videos.
        for &vid in &video_ids {
            db.conn.execute(
                "UPDATE jobs SET state = 'pending', attempts = 0, last_error = NULL, updated_at = ?1
                  WHERE video_id = ?2 AND stage = ?3",
                rusqlite::params![now(), vid, stage],
            )?;
        }
    }
    Ok(())
}

/// Pending (or retryable) jobs for `stage` in scope, with one existing file path per video.
fn claimable_jobs(db: &Db, stage: &str, opts: &Options) -> Result<Vec<(i64, PathBuf)>, Error> {
    let max_attempts = if opts.retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    let sql = "
        SELECT j.video_id, (SELECT vf.path FROM video_files vf
                             JOIN project_folders pf ON pf.folder_id = vf.folder_id
                            WHERE vf.video_id = j.video_id AND (?3 IS NULL OR pf.project_id = ?3)
                              AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id AND x.role = 'removed')
                            ORDER BY vf.id LIMIT 1) AS path
          FROM jobs j
         WHERE j.stage = ?1
           AND (j.state = 'pending' OR (j.state = 'failed' AND j.attempts < ?2))
           AND (?4 IS NULL OR EXISTS (SELECT 1 FROM jobs p WHERE p.video_id = j.video_id
                                          AND p.stage = ?4 AND p.state = 'done'))
         ORDER BY j.video_id";
    let mut st = db.conn.prepare(sql)?;
    let rows = st.query_map(params![stage, max_attempts, opts.project_id, prerequisite(stage)], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?))
    })?;
    Ok(rows.filter_map(Result::ok).filter_map(|(id, path)| path.map(|p| (id, PathBuf::from(p)))).collect())
}

fn set_job(db: &Db, video_id: i64, stage: &str, state: &str, error: Option<&str>) -> Result<(), Error> {
    let bump = if state == "failed" { 1 } else { 0 };
    db.conn.execute(
        "UPDATE jobs SET state = ?1, last_error = ?2, attempts = attempts + ?3, updated_at = ?4
          WHERE video_id = ?5 AND stage = ?6",
        params![state, error, bump, now(), video_id, stage],
    )?;
    Ok(())
}

fn store_media(db: &Db, video_id: i64, m: &MediaInfo) -> Result<(), Error> {
    db.conn.execute(
        "UPDATE videos SET duration_s = ?1, width = ?2, height = ?3, rotation = ?4, fps = ?5, avg_fps = ?6,
                vfr = ?7, vcodec = ?8, acodec = ?9, has_audio = ?10, created_time = ?11,
                status = 'probed', error = NULL
          WHERE id = ?12",
        params![
            m.duration_s,
            m.width,
            m.height,
            m.rotation,
            m.fps,
            m.avg_fps,
            m.vfr,
            m.vcodec,
            m.acodec,
            m.has_audio,
            m.created_time,
            video_id
        ],
    )?;
    Ok(())
}

async fn run_probe_jobs(
    db: &mut Db,
    ffprobe_bin: &Path,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "probe";
    let jobs = claimable_jobs(db, STAGE, opts)?;
    tracker.set_total(STAGE, jobs.len() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));
    let sem = Arc::new(Semaphore::new(PARALLEL_PROBE));
    let mut set = JoinSet::new();
    for (video_id, path) in jobs {
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let (sem, bin) = (sem.clone(), ffprobe_bin.to_path_buf());
        set.spawn(async move {
            let _permit = sem.acquire_owned().await;
            let result = media::ffprobe(&bin, &path).await;
            (video_id, path, result)
        });
    }
    let (mut done, mut failed) = (0, 0);
    while let Some(joined) = set.join_next().await {
        let Ok((video_id, path, result)) = joined else { continue };
        match result {
            Ok(info) => {
                store_media(db, video_id, &info)?;
                set_job(db, video_id, STAGE, "done", None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                set_job(db, video_id, STAGE, "failed", Some(&msg))?;
                db.conn
                    .execute("UPDATE videos SET status = 'error', error = ?1 WHERE id = ?2", params![msg, video_id])?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: msg });
                failed += 1;
            }
        }
        tracker.advance(STAGE, 1);
        if tracker.should_emit() {
            on_event(Event::Progress(tracker.snapshot(Some(path))));
        }
    }
    Ok((done, failed))
}

// ---- transcribe ---------------------------------------------------------------------------

fn stt_phase(setup: &SttSetup) -> Option<&'static str> {
    match setup {
        SttSetup::Ready(Engine::Server { .. }) => Some("transcribe_server"),
        SttSetup::Ready(Engine::Local { .. }) | SttSetup::NeedsModel { .. } => Some("transcribe_local"),
        SttSetup::Unavailable(_) => None,
    }
}

/// (seconds of audio known to need transcription, videos in the queue whose duration is unknown)
fn transcribe_work(db: &Db, opts: &Options) -> Result<(f64, i64), Error> {
    let max_attempts = if opts.retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    Ok(db.conn.query_row(
        "SELECT COALESCE(SUM(v.duration_s), 0), COALESCE(SUM(v.duration_s IS NULL), 0)
           FROM jobs j JOIN videos v ON v.id = j.video_id
          WHERE j.stage = 'transcribe' AND COALESCE(v.has_audio, 1) = 1
            AND (j.state = 'pending' OR (j.state = 'failed' AND j.attempts < ?1))
            AND EXISTS (SELECT 1 FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
                         WHERE vf.video_id = j.video_id AND (?2 IS NULL OR pf.project_id = ?2)
                           AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id AND x.role = 'removed'))",
        params![max_attempts, opts.project_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?)
}

fn store_transcript(db: &mut Db, video_id: i64, t: &stt::Transcript) -> Result<(), Error> {
    let tx = db.conn.transaction()?;
    tx.execute("DELETE FROM transcript_segments WHERE video_id = ?1", [video_id])?;
    for s in &t.segments {
        tx.execute(
            "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (?1, ?2, ?3, ?4)",
            params![video_id, s.start, s.end, s.text],
        )?;
    }
    tx.execute("UPDATE videos SET language = ?1 WHERE id = ?2", params![t.language, video_id])?;
    tx.execute("UPDATE jobs SET state = 'pending', attempts = 0 WHERE video_id = ?1 AND stage = 'embed'", [video_id])?;
    tx.execute(
        "UPDATE jobs SET state = 'done', last_error = NULL, updated_at = ?1 WHERE video_id = ?2 AND stage = 'transcribe'",
        params![now(), video_id],
    )?;
    tx.commit()?;
    Ok(())
}

async fn run_transcribe_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "transcribe";
    // Videos without an audio track have nothing to transcribe.
    db.conn.execute(
        "UPDATE jobs SET state = 'skipped', updated_at = ?1
          WHERE stage = 'transcribe' AND state IN ('pending', 'failed')
            AND video_id IN (SELECT id FROM videos WHERE has_audio = 0)
            AND EXISTS (SELECT 1 FROM jobs p WHERE p.video_id = jobs.video_id AND p.stage = 'probe' AND p.state = 'done')",
        [now()],
    )?;
    let jobs = claimable_jobs(db, STAGE, opts)?;
    let Some(phase) = stt_phase(&rt.stt) else {
        if !jobs.is_empty()
            && let SttSetup::Unavailable(reason) = &rt.stt
        {
            on_event(Event::StageUnavailable { stage: STAGE.into(), reason: reason.clone() });
        }
        return Ok((0, 0));
    };
    if jobs.is_empty() {
        tracker.set_total(phase, 0);
        return Ok((0, 0));
    }

    let durations: HashMap<i64, f64> = {
        let mut st = db.conn.prepare("SELECT id, COALESCE(duration_s, 0) FROM videos")?;
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?
    };
    let total: f64 = jobs.iter().map(|(id, _)| durations.get(id).copied().unwrap_or(0.0)).sum();

    // Local model missing: download it first (resumable), then run locally.
    let engine = match &rt.stt {
        SttSetup::Ready(e) => e.clone(),
        SttSetup::NeedsModel { spec, dir, asr_bin, language } => {
            on_event(Event::DownloadingModel { file: spec.file_name.clone() });
            tracker.start("download");
            let mut last_done = 0u64;
            let result = crate::models::download(spec, dir, |done, len| {
                if let Some(len) = len {
                    tracker.set_total("download", len);
                }
                tracker.advance("download", done.saturating_sub(last_done));
                last_done = done;
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(None)));
                }
            })
            .await;
            match result {
                Ok(model) => Engine::Local { asr_bin: asr_bin.clone(), model, language: language.clone() },
                Err(e) => {
                    on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                    return Ok((0, 0));
                }
            }
        }
        SttSetup::Unavailable(_) => unreachable!("handled above"),
    };
    on_event(Event::StageBackend { stage: STAGE.into(), backend: engine.label() });

    tracker.set_total(phase, total.ceil() as u64);
    tracker.start(phase);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    let mut transcriber = stt::Transcriber::new(&engine);
    for (video_id, path) in jobs {
        if opts.cancelled() {
            break;
        }
        let duration = durations.get(&video_id).copied().unwrap_or(0.0);
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let g = db.glossary_for_video(video_id).unwrap_or_default();
        let prompt = g.whisper_prompt();
        let mut credited = 0.0f64;
        // A stop drops the transcription mid-video (its decoder and worker are killed below); the
        // video stays pending and is transcribed from the start next run.
        let result = crate::cancel::race(
            opts.cancel.as_ref(),
            transcriber.transcribe(
                &rt.ffmpeg,
                &path,
                duration,
                |secs| {
                    let secs = secs.min(duration);
                    if secs > credited {
                        tracker.advance(phase, (secs - credited).round() as u64);
                        credited = secs.round();
                    }
                    if tracker.should_emit() {
                        on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                    }
                },
                prompt.as_deref(),
            ),
        )
        .await;
        let Ok(result) = result else {
            set_job(db, video_id, STAGE, "pending", None)?;
            break;
        };
        match result {
            Ok(mut t) => {
                for seg in &mut t.segments {
                    seg.text = g.correct(&seg.text);
                }
                store_transcript(db, video_id, &t)?;
                // Which track carries the speech, and who is close to the microphone. Measured
                // here because the transcript gives the stretches worth measuring; a failure is
                // not worth failing the stage for, it only leaves the editor less informed.
                if rt.measure_audio
                    && !opts.cancelled()
                    && let Ok(tracks) = crate::audio::track_count(&rt.ffprobe, &path).await
                    && tracks > 0
                {
                    let track = crate::audio::pick_track(&rt.ffmpeg, &path, tracks, duration).await;
                    let _ = db.set_audio_track(video_id, track);
                    if let Some(track) = track {
                        let spans: Vec<(f64, f64)> = t.segments.iter().map(|s| (s.start, s.end)).collect();
                        let levels = if tracks == 1
                            && track == 0
                            && crate::audio::track_channels(&rt.ffprobe, &path, track).await == Some(1)
                        {
                            if let Some(pcm) = transcriber.decoded_audio() {
                                crate::audio::segment_levels_pcm(pcm, &spans).await
                            } else {
                                crate::audio::segment_levels(&rt.ffmpeg, &path, track, &spans).await
                            }
                        } else {
                            crate::audio::segment_levels(&rt.ffmpeg, &path, track, &spans).await
                        };
                        let flags = crate::audio::off_mic_flags(&levels, crate::audio::OFF_MIC_MARGIN_DB);
                        let rows: Vec<(f64, Option<bool>)> = spans.iter().map(|(a, _)| *a).zip(flags).collect();
                        let _ = db.set_off_mic(video_id, &rows);
                    }
                }
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                set_job(db, video_id, STAGE, "failed", Some(&msg))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: msg });
                failed += 1;
                // A server that went away mid-run: stop hammering it, retry next run.
                if matches!(engine, Engine::Server { .. }) && e.to_string().contains("GhostPen at") {
                    let remaining = (duration - credited).max(0.0);
                    tracker.advance(phase, remaining as u64);
                    on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                    break;
                }
            }
        }
        if duration > credited {
            tracker.advance(phase, (duration - credited).round() as u64);
        }
        on_event(Event::Progress(tracker.snapshot(Some(path))));
    }
    transcriber.shutdown().await;
    Ok((done, failed))
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Spoken away from the subject's mic (usually the interviewer's questions).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub off_mic: bool,
}

pub fn transcript(db: &Db, video_id: i64) -> Result<Vec<TranscriptSegment>, Error> {
    let mut st = db.conn.prepare(
        "SELECT start_s, end_s, text, COALESCE(off_mic, 0) FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s",
    )?;
    let rows = st.query_map([video_id], |r| {
        Ok(TranscriptSegment { start: r.get(0)?, end: r.get(1)?, text: r.get(2)?, off_mic: r.get(3)? })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

// ---- frames -------------------------------------------------------------------------------

/// (seconds of video queued for `stage`, queued videos with unknown duration)
fn stage_work(db: &Db, stage: &str, opts: &Options) -> Result<(f64, i64), Error> {
    let max_attempts = if opts.retry_failed { i64::MAX } else { MAX_ATTEMPTS };
    Ok(db.conn.query_row(
        "SELECT COALESCE(SUM(v.duration_s), 0), COALESCE(SUM(v.duration_s IS NULL), 0)
           FROM jobs j JOIN videos v ON v.id = j.video_id
          WHERE j.stage = ?1
            AND (j.state = 'pending' OR (j.state = 'failed' AND j.attempts < ?2))
            AND EXISTS (SELECT 1 FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
                         WHERE vf.video_id = j.video_id AND (?3 IS NULL OR pf.project_id = ?3)
                           AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id AND x.role = 'removed'))",
        params![stage, max_attempts, opts.project_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?)
}

/// `frames/<2 hex>/<hash>/` under the data dir (relative, so the data dir can move).
pub fn frames_rel_dir(content_hash: &str) -> PathBuf {
    let hex = content_hash.rsplit(':').next().unwrap_or(content_hash);
    PathBuf::from("frames").join(&hex[..2.min(hex.len())]).join(hex)
}

fn store_frames(db: &mut Db, video_id: i64, data_dir: &Path, frames: &[crate::frames::Frame]) -> Result<(), Error> {
    let tx = db.conn.transaction()?;
    tx.execute("DELETE FROM frames WHERE video_id = ?1", [video_id])?;
    for f in frames {
        let rel = f.path.strip_prefix(data_dir).unwrap_or(&f.path).to_string_lossy().replace('\\', "/");
        tx.execute(
            "INSERT INTO frames(video_id, t_s, thumb_path, phash) VALUES (?1, ?2, ?3, ?4)",
            params![video_id, f.t_s, rel, f.dhash as i64],
        )?;
    }
    tx.execute(
        "UPDATE jobs SET state = 'done', last_error = NULL, updated_at = ?1 WHERE video_id = ?2 AND stage = 'frames'",
        params![now(), video_id],
    )?;
    tx.commit()?;
    Ok(())
}

async fn run_frame_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "frames";
    let Some(frame_opts) = rt.frames.clone() else { return Ok((0, 0)) };
    let jobs = claimable_jobs(db, STAGE, opts)?;
    if jobs.is_empty() {
        tracker.set_total(STAGE, 0);
        return Ok((0, 0));
    }
    let info: HashMap<i64, (f64, String)> = {
        let mut st = db.conn.prepare("SELECT id, COALESCE(duration_s, 0), content_hash FROM videos")?;
        st.query_map([], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?.collect::<Result<_, _>>()?
    };
    let total: f64 = jobs.iter().map(|(id, _)| info.get(id).map(|i| i.0).unwrap_or(0.0)).sum();
    tracker.set_total(STAGE, total.ceil() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    for (video_id, path) in jobs {
        if opts.cancelled() {
            break;
        }
        let Some((duration, hash)) = info.get(&video_id).cloned() else { continue };
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let dir = rt.data_dir.join(frames_rel_dir(&hash));
        // Start clean: a previous interrupted run may have left files.
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let mut credited = 0.0f64;
        let result = crate::cancel::race(
            opts.cancel.as_ref(),
            crate::frames::extract_keyframes(&rt.ffmpeg, &path, duration, &dir, &frame_opts, |fraction| {
                let secs = (fraction * duration).floor();
                if secs > credited {
                    tracker.advance(STAGE, (secs - credited) as u64);
                    credited = secs;
                }
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                }
            }),
        )
        .await;
        // Stopped mid-video: its half-written frames go, the video is redone next run.
        let Ok(result) = result else {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            set_job(db, video_id, STAGE, "pending", None)?;
            break;
        };
        match result {
            Ok(frames) => {
                store_frames(db, video_id, &rt.data_dir, &frames)?;
                set_job(db, video_id, STAGE, "done", None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                set_job(db, video_id, STAGE, "failed", Some(&msg))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: msg });
                failed += 1;
            }
        }
        if duration > credited {
            tracker.advance(STAGE, (duration - credited).round() as u64);
        }
        on_event(Event::Progress(tracker.snapshot(Some(path))));
    }
    Ok((done, failed))
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SteadinessSummary {
    pub measured: usize,
    pub failed: usize,
    pub total: usize,
}

/// Measure camera steadiness for videos that haven't been measured yet (or all videos if `force` is true).
/// Runs lazily as a post-indexing action or dedicated background task.
pub async fn run_steadiness_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    force: bool,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<SteadinessSummary, Error> {
    const STAGE: &str = "steadiness";
    let total_in_scope: usize = match opts.project_id {
        Some(pid) => {
            let mut st = db.conn.prepare(
                "SELECT COUNT(DISTINCT v.id)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0",
            )?;
            st.query_row([pid], |r| r.get::<_, i64>(0)).map(|v| v as usize).unwrap_or(0)
        }
        None => {
            let mut st = db.conn.prepare("SELECT COUNT(DISTINCT id) FROM videos WHERE duration_s > 0")?;
            st.query_row([], |r| r.get::<_, i64>(0)).map(|v| v as usize).unwrap_or(0)
        }
    };

    let Some(steadiness_opts) = rt.steadiness else {
        return Ok(SteadinessSummary { measured: 0, failed: 0, total: total_in_scope });
    };

    // Find videos for the project (or all) that have duration > 0.
    let unmeasured: Vec<(i64, PathBuf, f64)> = match opts.project_id {
        Some(pid) => {
            let sql = if force {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0
                 GROUP BY v.id
                 ORDER BY v.id"
            } else {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0 AND {needs}
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let sql = sql.replace("{needs}", &needs_analysis_sql(STAGE, "motion_windows", opts.retry_failed));
            let mut st = db.conn.prepare(&sql)?;
            let rows = st.query_map([pid], |r| Ok((r.get(0)?, PathBuf::from(r.get::<_, String>(1)?), r.get(2)?)))?;
            rows.collect::<Result<_, _>>()?
        }
        None => {
            let sql = if force {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 WHERE v.duration_s > 0
                 GROUP BY v.id
                 ORDER BY v.id"
            } else {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 WHERE v.duration_s > 0 AND {needs}
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let sql = sql.replace("{needs}", &needs_analysis_sql(STAGE, "motion_windows", opts.retry_failed));
            let mut st = db.conn.prepare(&sql)?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, PathBuf::from(r.get::<_, String>(1)?), r.get(2)?)))?;
            rows.collect::<Result<_, _>>()?
        }
    };

    if unmeasured.is_empty() {
        tracker.set_total(STAGE, 0);
        return Ok(SteadinessSummary { measured: 0, failed: 0, total: total_in_scope });
    }

    let total: f64 = unmeasured.iter().map(|(_, _, d)| *d).sum();
    tracker.set_total(STAGE, total.ceil() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    for (video_id, path, duration) in unmeasured {
        if opts.cancelled() {
            break;
        }
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        // One video is one long decode: progress comes from inside it, and a stop drops it (and
        // kills its ffmpeg) at once. A stopped video stores nothing; it is measured next run.
        let mut credited = 0u64;
        let measured = crate::cancel::race(
            opts.cancel.as_ref(),
            crate::steadiness::measure(
                &rt.ffmpeg,
                &path,
                duration,
                steadiness_opts.window_s,
                steadiness_opts.stride_s,
                |secs| {
                    let secs = secs.min(duration).floor() as u64;
                    if secs > credited {
                        tracker.advance(STAGE, secs - credited);
                        credited = secs;
                    }
                    if tracker.should_emit() {
                        on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                    }
                },
            ),
        )
        .await;
        match measured {
            Err(_) => break,
            Ok(Ok(windows)) => {
                db.set_motion_windows(video_id, &windows)?;
                record_analysis(db, video_id, STAGE, None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Ok(Err(e)) => {
                record_analysis(db, video_id, STAGE, Some(&e.to_string()))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: e.to_string() });
                failed += 1;
            }
        }
        tracker.advance(STAGE, (duration.ceil() as u64).saturating_sub(credited));
        on_event(Event::Progress(tracker.snapshot(Some(path))));
    }
    Ok(SteadinessSummary { measured: done, failed, total: total_in_scope })
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct FaceSummary {
    pub measured: usize,
    pub failed: usize,
    pub total: usize,
}

/// Find and track faces for videos that haven't been processed yet (or all videos if `force` is true).
#[allow(clippy::too_many_arguments)]
pub async fn run_face_jobs(
    db: &mut Db,
    paths: &crate::paths::Paths,
    config: &crate::config::Config,
    rt: &Runtime,
    opts: &Options,
    force: bool,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<FaceSummary, Error> {
    const STAGE: &str = "faces";
    let total_in_scope: usize = match opts.project_id {
        Some(pid) => {
            let mut st = db.conn.prepare(
                "SELECT COUNT(DISTINCT v.id)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0",
            )?;
            st.query_row([pid], |r| r.get::<_, i64>(0)).map(|v| v as usize).unwrap_or(0)
        }
        None => {
            let mut st = db.conn.prepare("SELECT COUNT(DISTINCT id) FROM videos WHERE duration_s > 0")?;
            st.query_row([], |r| r.get::<_, i64>(0)).map(|v| v as usize).unwrap_or(0)
        }
    };

    // Find videos for the project (or all) that have duration > 0.
    let unmeasured: Vec<(i64, PathBuf, f64, u32, u32)> = match opts.project_id {
        Some(pid) => {
            let sql = if force {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0), COALESCE(v.width, 1280), COALESCE(v.height, 720)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0
                 GROUP BY v.id
                 ORDER BY v.id"
            } else {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0), COALESCE(v.width, 1280), COALESCE(v.height, 720)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
                 WHERE v.duration_s > 0 AND {needs}
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let sql = sql.replace("{needs}", &needs_analysis_sql(STAGE, "face_tracks", opts.retry_failed));
            let mut st = db.conn.prepare(&sql)?;
            let rows = st.query_map([pid], |r| {
                Ok((
                    r.get(0)?,
                    PathBuf::from(r.get::<_, String>(1)?),
                    r.get(2)?,
                    r.get::<_, i64>(3)? as u32,
                    r.get::<_, i64>(4)? as u32,
                ))
            })?;
            rows.collect::<Result<_, _>>()?
        }
        None => {
            let sql = if force {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0), COALESCE(v.width, 1280), COALESCE(v.height, 720)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 WHERE v.duration_s > 0
                 GROUP BY v.id
                 ORDER BY v.id"
            } else {
                "SELECT v.id, vf.path, COALESCE(v.duration_s, 0), COALESCE(v.width, 1280), COALESCE(v.height, 720)
                 FROM videos v
                 JOIN video_files vf ON vf.video_id = v.id
                 WHERE v.duration_s > 0 AND {needs}
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let sql = sql.replace("{needs}", &needs_analysis_sql(STAGE, "face_tracks", opts.retry_failed));
            let mut st = db.conn.prepare(&sql)?;
            let rows = st.query_map([], |r| {
                Ok((
                    r.get(0)?,
                    PathBuf::from(r.get::<_, String>(1)?),
                    r.get(2)?,
                    r.get::<_, i64>(3)? as u32,
                    r.get::<_, i64>(4)? as u32,
                ))
            })?;
            rows.collect::<Result<_, _>>()?
        }
    };

    if unmeasured.is_empty() {
        tracker.set_total(STAGE, 0);
        return Ok(FaceSummary { measured: 0, failed: 0, total: total_in_scope });
    }

    let models_dir = crate::models::effective_models_dir(paths, config);
    let roots = crate::models::search_roots(&models_dir, &config.models.search_paths);
    let yunet_path = crate::models::find(&roots, "face_detection_yunet_2023mar.onnx");
    let sface_path = crate::models::find(&roots, "face_recognition_sface_2021dec.onnx");

    let (Some(yunet_file), Some(sface_file)) = (yunet_path, sface_path) else {
        on_event(Event::StageUnavailable {
            stage: STAGE.into(),
            reason: "face models not downloaded — Models page".into(),
        });
        return Ok(FaceSummary { measured: 0, failed: 0, total: total_in_scope });
    };

    let env = crate::onnx::environment()?;
    let detector = crate::faces::OrtDetector::new(&env, &yunet_file.to_string_lossy())?;
    let embedder = crate::faces::OrtEmbedder::new(&env, &sface_file.to_string_lossy())?;
    // The models travel to the blocking thread with each video and come back with its tracks.
    let mut models = Some((detector, embedder));

    let total: f64 = unmeasured.iter().map(|(_, _, d, _, _)| *d).sum();
    tracker.set_total(STAGE, total.ceil() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    for (video_id, path, duration, w, h) in unmeasured {
        if opts.cancelled() {
            break;
        }
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });

        // Decoding and ONNX inference are blocking: on their own thread, so the run's progress
        // and Stop keep flowing. The loop checks the stop flag every frame and kills its ffmpeg.
        let Some((mut detector, mut embedder)) = models.take() else { break };
        let (ffmpeg, video, cancel) = (rt.ffmpeg.clone(), path.clone(), opts.cancel.clone());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<f64>();
        let job = tokio::task::spawn_blocking(move || {
            let res = crate::faces::process_video_faces(
                &ffmpeg,
                &video,
                video_id,
                w,
                h,
                &mut detector,
                &mut embedder,
                cancel.as_ref(),
                |t| {
                    let _ = tx.send(t);
                },
            );
            (res, detector, embedder)
        });
        tokio::pin!(job);
        let mut credited = 0u64;
        let joined = loop {
            tokio::select! {
                j = &mut job => break j,
                Some(t) = rx.recv() => {
                    let secs = t.min(duration).floor() as u64;
                    if secs > credited {
                        tracker.advance(STAGE, secs - credited);
                        credited = secs;
                    }
                    if tracker.should_emit() {
                        on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                    }
                }
            }
        };
        let (res, detector, embedder) =
            joined.map_err(|e| Error::Invalid(format!("face tracking thread failed: {e}")))?;
        models = Some((detector, embedder));

        match res {
            Err(e) if e.is_cancelled() => break,
            Ok(tracks) => {
                db.set_face_tracks(video_id, &tracks)?;
                record_analysis(db, video_id, STAGE, None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                record_analysis(db, video_id, STAGE, Some(&e.to_string()))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: e.to_string() });
                failed += 1;
            }
        }
        tracker.advance(STAGE, (duration.ceil() as u64).saturating_sub(credited));
        on_event(Event::Progress(tracker.snapshot(Some(path))));
    }
    Ok(FaceSummary { measured: done, failed, total: total_in_scope })
}

/// Analyse who speaks when (voice), who is on screen (people) and which face each voice is, for
/// one project or every project. Needs transcripts; uses face tracks when there are any. Only
/// runs again when voices or faces changed since the last run, unless `force`.
#[allow(clippy::too_many_arguments)]
pub async fn run_speaker_jobs(
    db: &mut Db,
    paths: &crate::paths::Paths,
    config: &crate::config::Config,
    rt: &Runtime,
    opts: &Options,
    force: bool,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<Vec<(i64, crate::speakers::SpeakersReport)>, Error> {
    const STAGE: &str = "speakers";
    let projects: Vec<i64> = match opts.project_id {
        Some(pid) => vec![pid],
        None => db.projects()?.into_iter().map(|p| p.id).collect(),
    };
    // Progress counts seconds of speech-bearing video to embed.
    let mut total = 0.0f64;
    let mut need_model = false;
    for &pid in &projects {
        for vid in db.project_video_ids(pid)? {
            let has_speech: bool = db.conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM transcript_segments WHERE video_id = ?1)",
                [vid],
                |r| r.get(0),
            )?;
            if !has_speech {
                continue;
            }
            let has_windows: bool =
                db.conn.query_row("SELECT EXISTS (SELECT 1 FROM speaker_windows WHERE video_id = ?1)", [vid], |r| {
                    r.get(0)
                })?;
            if force || !has_windows {
                need_model = true;
                total += db.conn.query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [vid], |r| {
                    r.get::<_, f64>(0)
                })?;
            }
        }
    }

    let models_dir = crate::models::effective_models_dir(paths, config);
    let roots = crate::models::search_roots(&models_dir, &config.models.search_paths);
    let mut model_path = crate::models::find(&roots, crate::speakers::SPEAKER_MODEL_FILE);
    if model_path.is_none() && need_model {
        // Small (~29 MB): fetch it on first use rather than send people to the Models page.
        let entry = crate::models::catalog().into_iter().find(|e| e.file_name == crate::speakers::SPEAKER_MODEL_FILE);
        if let Some(entry) = entry {
            on_event(Event::DownloadingModel { file: entry.file_name.clone() });
            let spec = crate::models::ModelSpec { file_name: entry.file_name.clone(), url: entry.url.clone() };
            match crate::models::download(&spec, &models_dir, |_, _| {}).await {
                Ok(p) => model_path = Some(p),
                Err(e) => {
                    on_event(Event::StageUnavailable {
                        stage: STAGE.into(),
                        reason: format!("voice model not downloaded ({e}) — Models page"),
                    });
                }
            }
        }
    }
    let env;
    let mut embedder = match &model_path {
        Some(p) => {
            env = crate::onnx::environment()?;
            Some(crate::speakers::OrtSpeakerEmbedder::new(&env, &p.to_string_lossy())?)
        }
        None => None,
    };
    if need_model && embedder.is_none() {
        return Ok(Vec::new());
    }

    tracker.set_total(STAGE, total.ceil() as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));
    let mut reports = Vec::new();
    for pid in projects {
        if opts.cancelled() {
            break;
        }
        let mut current: Option<PathBuf> = None;
        let mut step = |s: crate::speakers::Step<'_>| match s {
            crate::speakers::Step::Started { video_id, path, phase: _ } => {
                current = Some(path.to_path_buf());
                on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.to_path_buf() });
            }
            crate::speakers::Step::Advanced { seconds } => {
                tracker.advance(STAGE, seconds.round() as u64);
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(current.clone())));
                }
            }
            crate::speakers::Step::Done { video_id } => on_event(Event::JobDone { video_id, stage: STAGE.into() }),
            crate::speakers::Step::Failed { video_id, error } => {
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error })
            }
        };
        let report = crate::speakers::analyse_project(
            db,
            &rt.ffmpeg,
            pid,
            embedder.as_mut(),
            force,
            opts.cancel.as_ref(),
            &mut step,
        );
        match report {
            Ok(r) => reports.push((pid, r)),
            Err(e) if opts.cancelled() || e.is_cancelled() => break,
            Err(e) => return Err(e),
        }
    }
    Ok(reports)
}

// ---- describe -----------------------------------------------------------------------------

fn describe_phase(setup: &VisionSetup) -> Option<&'static str> {
    match setup {
        VisionSetup::Server(_) => Some("describe_server"),
        VisionSetup::Local { .. } | VisionSetup::Mlx { .. } => Some("describe_local"),
        VisionSetup::Cli(_) => Some("describe_server"), // CLI is billed externally, similar wall time
        VisionSetup::Unavailable(_) => None,
    }
}

/// Speech within ±15 s of `t_s`.
fn speech_near(db: &Db, video_id: i64, t_s: f64) -> Result<String, Error> {
    let mut st = db.conn.prepare(
        "SELECT text FROM transcript_segments WHERE video_id = ?1 AND end_s >= ?2 AND start_s <= ?3 ORDER BY start_s",
    )?;
    let parts: Vec<String> =
        st.query_map(params![video_id, t_s - 15.0, t_s + 15.0], |r| r.get(0))?.collect::<Result<_, _>>()?;
    Ok(parts.join(" "))
}

/// A frame awaiting description: (frame id, t, absolute path).
type PendingFrame = (i64, f64, PathBuf);

/// Frames of `video_id` still without a description.
fn undescribed_frames(db: &Db, data_dir: &Path, video_id: i64) -> Result<Vec<PendingFrame>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, t_s, thumb_path FROM frames WHERE video_id = ?1 AND description_json IS NULL ORDER BY t_s",
    )?;
    let rows = st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, data_dir.join(r.get::<_, String>(2)?))))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Consecutive failures that mean the backend is broken rather than the frames.
///
/// Five, because a genuinely corrupt frame is rare and never arrives in runs, while a model whose
/// chat template this build cannot apply fails on the very first one and every one after.
const BROKEN_BACKEND_RUN: usize = 5;

/// How many frames to describe at once against a server or local helper.
fn describe_concurrency(rt: &Runtime) -> usize {
    match &rt.vision {
        VisionSetup::Server(_) => rt.describe_concurrency.max(1),
        VisionSetup::Local { .. } => rt.describe_concurrency.max(1),
        _ => 1,
    }
}

/// Start the describer: the server client, CLI agent, or download missing local models and launch the helper.
async fn start_describer(
    rt: &Runtime,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<crate::vision::Describer, String> {
    use crate::vision::{Describer, LocalLlm, LocalModels};
    match &rt.vision {
        VisionSetup::Server(s) => Ok(Describer::Server(s.clone())),
        VisionSetup::Cli(cfg) => {
            let agent = crate::cliagent::CliAgent::new(cfg.clone());
            Ok(Describer::Cli(agent))
        }
        VisionSetup::Unavailable(why) => Err(why.clone()),
        VisionSetup::Mlx { helper, models_dir, entry, runtime } => {
            if !entry.bundle_installed(models_dir) {
                on_event(Event::DownloadingModel { file: format!("{} (MLX)", entry.id) });
            }
            tracker.start("download_vision");
            let mut last = 0u64;
            let dir = crate::models::download_bundle(entry, models_dir, |done, total| {
                if let Some(total) = total {
                    tracker.set_total("download_vision", total);
                }
                tracker.advance("download_vision", done.saturating_sub(last));
                last = done;
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(None)));
                }
            })
            .await
            .map_err(|e| e.to_string())?;
            LocalLlm::start_mlx(helper, &dir, runtime)
                .await
                .map(|l| Describer::Local(Box::new(l)))
                .map_err(|e| e.to_string())
        }
        VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime } => {
            let mut paths = Vec::new();
            for spec in std::iter::once(model).chain(mmproj.iter()) {
                if let Some(p) = found.iter().find(|p| {
                    p == &std::path::Path::new(&spec.file_name)
                        || p.file_name().is_some_and(|n| n == spec.file_name.as_str())
                }) {
                    paths.push(p.clone());
                    continue;
                }
                on_event(Event::DownloadingModel { file: spec.file_name.clone() });
                tracker.start("download_vision");
                let mut last = 0u64;
                let p = crate::models::download(spec, models_dir, |done, len| {
                    if let Some(len) = len {
                        tracker.set_total("download_vision", len);
                    }
                    tracker.advance("download_vision", done.saturating_sub(last));
                    last = done;
                    if tracker.should_emit() {
                        on_event(Event::Progress(tracker.snapshot(None)));
                    }
                })
                .await
                .map_err(|e| e.to_string())?;
                paths.push(p);
            }
            let models = LocalModels {
                helper: helper.clone(),
                vision: Some((paths[0].clone(), paths.get(1).cloned())),
                embed: None,
                cpu: false,
                runtime: runtime.clone(),
                concurrency: describe_concurrency(rt),
                free_ghostpen: crate::ghost_service::models_to_free(Some(rt)),
            };
            LocalLlm::start(&models).await.map(|l| Describer::Local(Box::new(l))).map_err(|e| e.to_string())
        }
    }
}

async fn run_describe_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "describe";
    let jobs = claimable_jobs(db, STAGE, opts)?;
    let Some(phase) = describe_phase(&rt.vision) else {
        if !jobs.is_empty()
            && let VisionSetup::Unavailable(reason) = &rt.vision
        {
            on_event(Event::StageUnavailable { stage: STAGE.into(), reason: reason.clone() });
        }
        return Ok((0, 0));
    };
    let work: Vec<(i64, PathBuf, Vec<PendingFrame>)> = jobs
        .into_iter()
        .map(|(id, path)| Ok((id, path, undescribed_frames(db, &rt.data_dir, id)?)))
        .collect::<Result<_, Error>>()?;
    let total: usize = work.iter().map(|w| w.2.len()).sum();
    // Videos whose frames are all described already (e.g. interrupted after the last frame).
    for (id, _, frames) in &work {
        if frames.is_empty() {
            set_job(db, *id, STAGE, "done", None)?;
        }
    }
    if total == 0 {
        tracker.set_total(phase, 0);
        return Ok((0, 0));
    }

    // Every frame description takes a slot on the model's endpoint, shared with script chat and
    // clip runs. A local model (and a server asked one frame at a time) is held for the whole
    // stage: its helper keeps the model loaded between frames.
    let n_at_once = describe_concurrency(rt);
    let hold_stage_slot = match &rt.vision {
        VisionSetup::Local { .. } | VisionSetup::Mlx { .. } => true,
        VisionSetup::Server(_) => n_at_once < 2,
        _ => false,
    };
    // The endpoint may be busy with a script chat or a clip run: say so while waiting, and let a
    // stop through.
    let _stage_slot = if hold_stage_slot {
        let wait = crate::agents::acquire_for(&rt.vision, 1);
        tokio::pin!(wait);
        match tokio::time::timeout(std::time::Duration::from_millis(500), &mut wait).await {
            Ok(slot) => slot,
            Err(_) => {
                let what = crate::agents::EndpointKey::for_setup(&rt.vision)
                    .map(|k| format!("a free slot on {}", k.label()))
                    .unwrap_or_else(|| "the model".into());
                on_event(Event::Waiting { stage: STAGE.into(), what });
                match crate::cancel::race(opts.cancel.as_ref(), wait).await {
                    Ok(slot) => slot,
                    Err(_) => return Ok((0, 0)),
                }
            }
        }
    } else {
        None
    };
    let started = crate::cancel::race(opts.cancel.as_ref(), start_describer(rt, tracker, on_event)).await;
    let Ok(started) = started else { return Ok((0, 0)) };
    let mut describer = match started {
        Ok(d) => d,
        Err(reason) => {
            on_event(Event::StageUnavailable { stage: STAGE.into(), reason });
            return Ok((0, 0));
        }
    };
    on_event(Event::StageBackend { stage: STAGE.into(), backend: rt.vision.describe() });
    tracker.set_total(phase, total as u64);
    tracker.start(phase);
    on_event(Event::Progress(tracker.snapshot(None)));

    let cli_concurrency = if let VisionSetup::Cli(c) = &rt.vision { Some(c.concurrency) } else { None };

    let (mut done, mut failed) = (0, 0);

    if let Some(concurrency) = cli_concurrency {
        // CLI backend: run up to `concurrency` frames at a time across all videos.
        use std::sync::Arc;
        let sem = Arc::new(crate::agents::EndpointSlots::for_setup(&rt.vision, concurrency.max(1)));
        let agent_cfg = if let VisionSetup::Cli(c) = &rt.vision { c.clone() } else { unreachable!() };

        'videos: for (video_id, path, frames) in work {
            if frames.is_empty() {
                continue;
            }
            set_job(db, video_id, STAGE, "running", None)?;
            on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });

            let mut tasks = JoinSet::new();
            for (frame_id, t_s, image) in frames {
                if opts.cancelled() {
                    tasks.abort_all();
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                }
                let speech = speech_near(db, video_id, t_s)?;
                let agent = crate::cliagent::CliAgent::new(agent_cfg.clone());
                let sem = sem.clone();
                tasks.spawn(async move {
                    let _permit = sem.acquire().await;
                    use crate::vision::schema;
                    let schema_hint = serde_json::to_string(&schema()).unwrap_or_default();
                    // Mirror Describer::Cli: pass schema + speech context as the schema_hint.
                    let hint = if !speech.is_empty() {
                        format!("{schema_hint}\n\nAudio near this frame: {speech}")
                    } else {
                        schema_hint
                    };
                    let json_text = agent.describe(&image, &hint).await;
                    (frame_id, json_text)
                });
            }

            loop {
                // A stop drops the frames still in flight; those already back are saved, the
                // rest stay undescribed for the next run.
                let Ok(next) = crate::cancel::race(opts.cancel.as_ref(), tasks.join_next()).await else {
                    tasks.abort_all();
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                };
                let Some(joined) = next else { break };
                let Ok((frame_id, result)) = joined else { continue };
                let result = result.and_then(|j| crate::vision::parse_description(&j));
                match result {
                    Ok(d) => {
                        let json = serde_json::to_string(&d).unwrap_or_default();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1, visible_text = ?2 WHERE id = ?3",
                            params![json, d.visible_text.join("\n"), frame_id],
                        )?;
                    }
                    Err(e) => {
                        let json = serde_json::json!({ "error": e.to_string() }).to_string();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1 WHERE id = ?2",
                            params![json, frame_id],
                        )?;
                    }
                }
                tracker.advance(phase, 1);
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                }
            }

            let ok: i64 = db.conn.query_row(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?1 AND description_json NOT LIKE '{\"error\"%'",
                [video_id],
                |r| r.get(0),
            )?;
            if ok == 0 {
                set_job(db, video_id, STAGE, "failed", Some("no frame could be described"))?;
                on_event(Event::JobFailed {
                    video_id,
                    stage: STAGE.into(),
                    error: "no frame could be described".into(),
                });
                failed += 1;
            } else {
                set_job(db, video_id, STAGE, "done", None)?;
                db.conn.execute(
                    "UPDATE jobs SET state = 'pending', attempts = 0 WHERE video_id = ?1 AND stage = 'embed'",
                    [video_id],
                )?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
        }
    } else if let (VisionSetup::Server(server), n @ 2..) = (&rt.vision, describe_concurrency(rt)) {
        // Server: several frames in flight at once.
        //
        // Describing is memory-bandwidth bound, not compute bound — generating one token means
        // reading every weight in the model out of VRAM, which is why an RTX 4070 tops out near
        // 90 tok/s on a 5.5 GB model however idle its compute units are. A batch reads those
        // weights *once* and produces a token for every request in it, so the bandwidth cost
        // amortises and throughput scales with the batch. Sequentially this stage ran at 3.0 s a
        // frame; the GPU was waiting on memory for most of it.
        //
        // The server has to be started with a matching `--parallel`, or the requests simply
        // queue. Queuing is harmless — measured at 2.87 s a frame against 3.03 s sequential — so
        // over-asking costs nothing and under-asking leaves the GPU idle.
        use std::sync::Arc;
        let sem = Arc::new(crate::agents::EndpointSlots::for_setup(&rt.vision, n));

        'videos: for (video_id, path, frames) in work {
            if frames.is_empty() {
                continue;
            }
            set_job(db, video_id, STAGE, "running", None)?;
            on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });

            // The database is not `Sync`, so everything it holds is read before anything is
            // spawned and written back after.
            let mut prepared = Vec::with_capacity(frames.len());
            for (frame_id, t_s, image) in frames {
                prepared.push((frame_id, speech_near(db, video_id, t_s)?, image));
            }

            let mut tasks = JoinSet::new();
            for (frame_id, speech, image) in prepared {
                if opts.cancelled() {
                    tasks.abort_all();
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                }
                let (sem, server) = (sem.clone(), server.clone());
                tasks.spawn(async move {
                    let _permit = sem.acquire().await;
                    let speech = (!speech.is_empty()).then_some(speech.as_str());
                    // One retry for a bad answer; a transport error is the server's problem.
                    let mut r = server.describe(&image, speech).await;
                    if matches!(&r, Err(e) if !is_transport_error(e)) {
                        r = server.describe(&image, speech).await;
                    }
                    (frame_id, r)
                });
            }

            let mut gone = None;
            loop {
                // A stop drops the frames still in flight; those already back are saved, the
                // rest stay undescribed for the next run.
                let Ok(next) = crate::cancel::race(opts.cancel.as_ref(), tasks.join_next()).await else {
                    tasks.abort_all();
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                };
                let Some(joined) = next else { break };
                let Ok((frame_id, result)) = joined else { continue };
                match result {
                    Ok(d) => {
                        let json = serde_json::to_string(&d).unwrap_or_default();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1, visible_text = ?2 WHERE id = ?3",
                            params![json, d.visible_text.join("\n"), frame_id],
                        )?;
                    }
                    Err(e) if is_transport_error(&e) => gone = Some(e.to_string()),
                    Err(e) => {
                        let json = serde_json::json!({ "error": e.to_string() }).to_string();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1 WHERE id = ?2",
                            params![json, frame_id],
                        )?;
                    }
                }
                tracker.advance(phase, 1);
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                }
            }

            // Server gone / helper crashed: leave the rest pending for a later run. Checked after
            // the batch drains so the frames that did come back are still saved.
            if let Some(reason) = gone {
                set_job(db, video_id, STAGE, "pending", Some(&reason))?;
                on_event(Event::StageUnavailable { stage: STAGE.into(), reason });
                break 'videos;
            }

            let ok: i64 = db.conn.query_row(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?1 AND description_json NOT LIKE '{\"error\"%'",
                [video_id],
                |r| r.get(0),
            )?;
            if ok == 0 {
                set_job(db, video_id, STAGE, "failed", Some("no frame could be described"))?;
                on_event(Event::JobFailed {
                    video_id,
                    stage: STAGE.into(),
                    error: "no frame could be described".into(),
                });
                failed += 1;
            } else {
                set_job(db, video_id, STAGE, "done", None)?;
                db.conn.execute(
                    "UPDATE jobs SET state = 'pending', attempts = 0 WHERE video_id = ?1 AND stage = 'embed'",
                    [video_id],
                )?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
        }
    } else if let (crate::vision::Describer::Local(local_llm), n @ 2..) = (&mut describer, describe_concurrency(rt)) {
        // Local: batch decode multiple frames concurrently in one forward pass.
        let mut recent_errors: Vec<i64> = Vec::new();
        'videos: for (video_id, path, frames) in work {
            if frames.is_empty() {
                continue;
            }
            set_job(db, video_id, STAGE, "running", None)?;
            on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });

            for chunk in frames.chunks(n) {
                if opts.cancelled() {
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                }
                let mut prepared = Vec::with_capacity(chunk.len());
                for (frame_id, t_s, image) in chunk {
                    let speech = speech_near(db, video_id, *t_s)?;
                    prepared.push((*frame_id, image, speech));
                }

                let batch_input: Vec<(&Path, Option<&str>)> = prepared
                    .iter()
                    .map(|(_, img, sp)| (img.as_path(), (!sp.is_empty()).then_some(sp.as_str())))
                    .collect();

                let Ok(results) =
                    crate::cancel::race(opts.cancel.as_ref(), local_llm.batch_describe(&batch_input)).await
                else {
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                };
                let results = match results {
                    Ok(r) => r,
                    Err(e) => {
                        set_job(db, video_id, STAGE, "pending", Some(&e.to_string()))?;
                        on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                        break 'videos;
                    }
                };

                for (idx, result) in results.into_iter().enumerate() {
                    let (frame_id, _, _) = prepared[idx];
                    match result {
                        Ok(d) => {
                            let json = serde_json::to_string(&d).unwrap_or_default();
                            db.conn.execute(
                                "UPDATE frames SET description_json = ?1, visible_text = ?2 WHERE id = ?3",
                                params![json, d.visible_text.join("\n"), frame_id],
                            )?;
                            recent_errors.clear();
                        }
                        Err(e) => {
                            let json = serde_json::json!({ "error": e.to_string() }).to_string();
                            db.conn.execute(
                                "UPDATE frames SET description_json = ?1 WHERE id = ?2",
                                params![json, frame_id],
                            )?;
                            recent_errors.push(frame_id);

                            if recent_errors.len() >= BROKEN_BACKEND_RUN {
                                for id in &recent_errors {
                                    db.conn.execute(
                                        "UPDATE frames SET description_json = NULL, visible_text = NULL WHERE id = ?1",
                                        [id],
                                    )?;
                                }
                                let reason = format!(
                                    "{BROKEN_BACKEND_RUN} frames in a row failed to describe ({e}); \
                                     leaving the rest for a later run"
                                );
                                set_job(db, video_id, STAGE, "pending", Some(&reason))?;
                                on_event(Event::StageUnavailable { stage: STAGE.into(), reason });
                                break 'videos;
                            }
                        }
                    }
                    tracker.advance(phase, 1);
                    if tracker.should_emit() {
                        on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                    }
                }
            }

            let ok: i64 = db.conn.query_row(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?1 AND description_json NOT LIKE '{\"error\"%'",
                [video_id],
                |r| r.get(0),
            )?;
            if ok == 0 {
                set_job(db, video_id, STAGE, "failed", Some("no frame could be described"))?;
                on_event(Event::JobFailed {
                    video_id,
                    stage: STAGE.into(),
                    error: "no frame could be described".into(),
                });
                failed += 1;
            } else {
                set_job(db, video_id, STAGE, "done", None)?;
                db.conn.execute(
                    "UPDATE jobs SET state = 'pending', attempts = 0 WHERE video_id = ?1 AND stage = 'embed'",
                    [video_id],
                )?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
        }
    } else {
        // Local with concurrency 1, or a server asked to describe one frame at a time: sequential single-describer
        // path.
        //
        // Frames that errored and have not yet been vindicated by a success. Kept across videos:
        // a broken backend does not repair itself at a video boundary.
        let mut recent_errors: Vec<i64> = Vec::new();
        'videos: for (video_id, path, frames) in work {
            if frames.is_empty() {
                continue;
            }
            set_job(db, video_id, STAGE, "running", None)?;
            on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
            for (frame_id, t_s, image) in frames {
                if opts.cancelled() {
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                }
                let speech = speech_near(db, video_id, t_s)?;
                let speech = (!speech.is_empty()).then_some(speech.as_str());
                // One retry for a bad/unparseable answer; transport errors stop the stage.
                let attempt = crate::cancel::race(opts.cancel.as_ref(), async {
                    let mut result = describer.describe(&image, speech).await;
                    if matches!(&result, Err(e) if !is_transport_error(e)) {
                        result = describer.describe(&image, speech).await;
                    }
                    result
                })
                .await;
                let Ok(result) = attempt else {
                    set_job(db, video_id, STAGE, "pending", None)?;
                    break 'videos;
                };
                match result {
                    Ok(d) => {
                        let json = serde_json::to_string(&d).unwrap_or_default();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1, visible_text = ?2 WHERE id = ?3",
                            params![json, d.visible_text.join("\n"), frame_id],
                        )?;
                        recent_errors.clear(); // the backend works; earlier failures were frames
                    }
                    Err(e) if is_transport_error(&e) => {
                        // Server gone / helper crashed: leave the rest pending for a later run.
                        set_job(db, video_id, STAGE, "pending", Some(&e.to_string()))?;
                        on_event(Event::StageUnavailable { stage: STAGE.into(), reason: e.to_string() });
                        break 'videos;
                    }
                    Err(e) => {
                        // This frame can't be described; record why and move on. An isolated
                        // failure is a bad frame and must not be retried forever.
                        let json = serde_json::json!({ "error": e.to_string() }).to_string();
                        db.conn.execute(
                            "UPDATE frames SET description_json = ?1 WHERE id = ?2",
                            params![json, frame_id],
                        )?;
                        recent_errors.push(frame_id);

                        // A run of them is not bad frames, it is a bad backend — a model whose
                        // chat template this build cannot apply answers every request the same
                        // way. Storing that as each frame's description turns a working index
                        // into errors that read as finished work: 576 good descriptions were
                        // overwritten before anyone looked at the text rather than the ticks.
                        // Roll them back so a retry picks them up, and stop.
                        if recent_errors.len() >= BROKEN_BACKEND_RUN {
                            for id in &recent_errors {
                                db.conn.execute(
                                    "UPDATE frames SET description_json = NULL, visible_text = NULL WHERE id = ?1",
                                    [id],
                                )?;
                            }
                            let reason = format!(
                                "{BROKEN_BACKEND_RUN} frames in a row failed to describe ({e}); \
                                 leaving the rest for a later run"
                            );
                            set_job(db, video_id, STAGE, "pending", Some(&reason))?;
                            on_event(Event::StageUnavailable { stage: STAGE.into(), reason });
                            break 'videos;
                        }
                    }
                }
                tracker.advance(phase, 1);
                if tracker.should_emit() {
                    on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
                }
            }
            let ok: i64 = db.conn.query_row(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?1 AND description_json NOT LIKE '{\"error\"%'",
                [video_id],
                |r| r.get(0),
            )?;
            if ok == 0 {
                set_job(db, video_id, STAGE, "failed", Some("no frame could be described"))?;
                on_event(Event::JobFailed {
                    video_id,
                    stage: STAGE.into(),
                    error: "no frame could be described".into(),
                });
                failed += 1;
            } else {
                set_job(db, video_id, STAGE, "done", None)?;
                db.conn.execute(
                    "UPDATE jobs SET state = 'pending', attempts = 0 WHERE video_id = ?1 AND stage = 'embed'",
                    [video_id],
                )?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
        }
    }
    Ok((done, failed))
}

// ---- embed ----------------------------------------------------------------------------------

/// Build the retrieval chunks for a video from its transcript and described frames.
fn build_chunks(db: &Db, video_id: i64) -> Result<Vec<crate::chunks::Chunk>, Error> {
    use crate::chunks::{DescribedFrame, Seg, frame_chunks, transcript_windows};
    let segs: Vec<(f64, f64, String)> = {
        let mut st = db
            .conn
            .prepare("SELECT start_s, end_s, text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
        st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?
    };
    let frames: Vec<(i64, f64, crate::vision::FrameDescription)> = {
        let mut st = db.conn.prepare(
            "SELECT id, t_s, description_json FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s",
        )?;
        st.query_map([video_id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?)))?
            .filter_map(|row| {
                let (id, t, json) = row.ok()?;
                let d: crate::vision::FrameDescription = serde_json::from_str(&json).ok()?;
                (!d.description.is_empty()).then_some((id, t, d))
            })
            .collect()
    };
    let duration: f64 =
        db.conn.query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [video_id], |r| r.get(0))?;
    let seg_refs: Vec<Seg> = segs.iter().map(|(s, e, t)| Seg { start: *s, end: *e, text: t }).collect();
    let frame_refs: Vec<DescribedFrame> =
        frames.iter().map(|(id, t, d)| DescribedFrame { id: *id, t_s: *t, description: d }).collect();
    let mut chunks = transcript_windows(&seg_refs);
    chunks.extend(frame_chunks(&frame_refs, &seg_refs, duration));
    Ok(chunks)
}

/// Replace a video's chunks (FTS follows via triggers). With `vectors`, also stores embeddings.
fn store_chunks(
    db: &mut Db,
    video_id: i64,
    chunks: &[crate::chunks::Chunk],
    vectors: Option<&[Vec<f32>]>,
) -> Result<(), Error> {
    let tx = db.conn.transaction()?;
    tx.execute("DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE video_id = ?1)", [video_id])?;
    tx.execute("DELETE FROM chunks WHERE video_id = ?1", [video_id])?;
    for (i, c) in chunks.iter().enumerate() {
        tx.execute(
            "INSERT INTO chunks(video_id, kind, start_s, end_s, text, frame_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![video_id, c.kind, c.start_s, c.end_s, c.text, c.frame_id],
        )?;
        if let Some(v) = vectors {
            let id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)",
                params![id, crate::embed::to_blob(&v[i])],
            )?;
        }
    }
    tx.commit()?;
    Ok(())
}

async fn run_embed_jobs(
    db: &mut Db,
    rt: &Runtime,
    opts: &Options,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
    const STAGE: &str = "embed";
    const BATCH: usize = 32;
    let jobs = claimable_jobs(db, STAGE, opts)?;
    if jobs.is_empty() {
        tracker.set_total(STAGE, 0);
        return Ok((0, 0));
    }
    let mut work = Vec::with_capacity(jobs.len());
    for (video_id, path) in jobs {
        let chunks = build_chunks(db, video_id)?;
        work.push((video_id, path, chunks));
    }
    let total: usize = work.iter().map(|w| w.2.len()).sum();

    let mut embedder = if matches!(rt.embed, EmbedSetup::Unavailable(_)) || total == 0 {
        None
    } else {
        let mut last = 0u64;
        let started = crate::cancel::race(
            opts.cancel.as_ref(),
            crate::runtime::start_embedder(&rt.embed, |done, len| {
                if let Some(len) = len {
                    tracker.set_total("download_embed", len);
                }
                tracker.start("download_embed");
                tracker.advance("download_embed", done.saturating_sub(last));
                last = done;
            }),
        )
        .await;
        let Ok(started) = started else { return Ok((0, 0)) };
        match started {
            Ok(e) => Some(e),
            Err(reason) => {
                on_event(Event::StageUnavailable { stage: STAGE.into(), reason });
                None
            }
        }
    };
    if let EmbedSetup::Unavailable(reason) = &rt.embed {
        on_event(Event::StageUnavailable { stage: STAGE.into(), reason: reason.clone() });
    }
    match &embedder {
        Some(e) => on_event(Event::StageBackend { stage: STAGE.into(), backend: e.label() }),
        None => {
            // Keyword search still works: store chunks without vectors, keep the job pending.
            for (video_id, _, chunks) in &work {
                store_chunks(db, *video_id, chunks, None)?;
            }
            tracker.set_total(STAGE, 0);
            return Ok((0, 0));
        }
    }
    tracker.set_total(STAGE, total as u64);
    tracker.start(STAGE);
    on_event(Event::Progress(tracker.snapshot(None)));

    let (mut done, mut failed) = (0, 0);
    for (video_id, path, chunks) in work {
        if opts.cancelled() {
            break;
        }
        set_job(db, video_id, STAGE, "running", None)?;
        let texts: Vec<String> = chunks.iter().map(|c| crate::embed::doc_text(&c.text)).collect();
        let mut vectors = Vec::with_capacity(texts.len());
        let mut error = None;
        let mut stopped = false;
        for batch in texts.chunks(BATCH) {
            let Ok(res) =
                crate::cancel::race(opts.cancel.as_ref(), embedder.as_mut().expect("checked above").embed(batch)).await
            else {
                stopped = true;
                break;
            };
            match res {
                Ok(v) => vectors.extend(v),
                Err(e) => {
                    error = Some(e.to_string());
                    break;
                }
            }
            tracker.advance(STAGE, batch.len() as u64);
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
            }
        }
        if stopped {
            // Nothing of this video is stored; it is embedded again next run.
            set_job(db, video_id, STAGE, "pending", None)?;
            break;
        }
        match error {
            None => {
                store_chunks(db, video_id, &chunks, Some(&vectors))?;
                set_job(db, video_id, STAGE, "done", None)?;
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Some(e) => {
                store_chunks(db, video_id, &chunks, None)?;
                set_job(db, video_id, STAGE, "failed", Some(&e))?;
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: e });
                failed += 1;
            }
        }
    }
    Ok((done, failed))
}

fn is_transport_error(e: &Error) -> bool {
    let s = e.to_string();
    s.contains("vision server at") && !s.contains("bad response")
        || s.contains("helper exited")
        || s.contains("helper write")
        || s.contains("helper read")
        || s.contains("timed out")
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameRow {
    pub id: i64,
    pub t_s: f64,
    /// Absolute path of the JPEG.
    pub path: PathBuf,
    pub description: Option<String>,
    pub visible_text: Option<String>,
}

pub fn frames(db: &Db, data_dir: &Path, video_id: i64) -> Result<Vec<FrameRow>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, t_s, thumb_path, description_json, visible_text FROM frames WHERE video_id = ?1 ORDER BY t_s",
    )?;
    let rows = st.query_map([video_id], |r| {
        let rel: String = r.get(2)?;
        Ok(FrameRow {
            id: r.get(0)?,
            t_s: r.get(1)?,
            path: data_dir.join(rel),
            description: r.get(3)?,
            visible_text: r.get(4)?,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

// ---- status -------------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize)]
pub struct StageCounts {
    pub stage: String,
    pub pending: i64,
    pub running: i64,
    pub done: i64,
    pub failed: i64,
    pub skipped: i64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Status {
    pub folders: i64,
    pub videos: i64,
    pub total_size: i64,
    pub total_duration_s: f64,
    pub vfr_videos: i64,
    pub stages: Vec<StageCounts>,
    #[serde(default)]
    pub steadiness_measured: i64,
    #[serde(default)]
    pub faces_measured: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoRow {
    pub id: i64,
    pub folder_ids: Vec<i64>,
    pub path: PathBuf,
    pub copies: i64,
    pub size: i64,
    pub duration_s: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub fps: Option<f64>,
    pub vfr: bool,
    pub vcodec: Option<String>,
    pub has_audio: Option<bool>,
    pub status: String,
    pub error: Option<String>,
    pub language: Option<String>,
    /// Transcript segments stored (0 before transcription).
    pub segments: i64,
    /// State of the transcribe job (`pending`, `done`, `failed`, `skipped`, …).
    pub transcribe: Option<String>,
    /// Keyframes stored.
    pub frames: i64,
    /// Seconds where the camera measured shakier than the configured limit (0 = none, or not
    /// measured, or the check is off).
    pub shaky_s: f64,
    /// Whether the camera was measured at all: "steady" is only a verdict when it was.
    pub steadiness_measured: bool,
    /// Whether faces were tracked for this video.
    pub faces_measured: bool,
    /// static, tripod, stabilised, handheld, or unknown.
    pub camera: String,
    /// Category: interview, broll, screencast, drone, audio, other, or custom.
    pub category: String,
    /// Confidence of auto-classification (0.0 to 1.0).
    pub category_confidence: f64,
    /// Source: 'auto' or 'manual'.
    pub category_source: String,
    /// Secondary category tag (e.g. 'music').
    pub category_tag: Option<String>,
}

/// Videos visible to a project (or all), one row per content with its first path.
const SCOPE: &str = "
    SELECT vf.video_id, MIN(vf.path) AS path, COUNT(DISTINCT vf.path) AS copies, json_group_array(DISTINCT vf.folder_id) AS folder_ids
      FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
     WHERE (?1 IS NULL OR pf.project_id = ?1)
       AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
     GROUP BY vf.video_id";

pub fn status(db: &Db, project_id: Option<i64>) -> Result<Status, Error> {
    let mut s = Status { folders: db.folders(project_id)?.len() as i64, ..Default::default() };
    (s.videos, s.total_size, s.total_duration_s, s.vfr_videos) = db.conn.query_row(
        &format!(
            "SELECT COUNT(*), COALESCE(SUM(v.size), 0), COALESCE(SUM(v.duration_s), 0), COALESCE(SUM(v.vfr), 0)
               FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id"
        ),
        [project_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    for stage in STAGES {
        let mut c = StageCounts { stage: stage.to_string(), ..Default::default() };
        let mut st = db.conn.prepare(&format!(
            "SELECT j.state, COUNT(*) FROM ({SCOPE}) sc JOIN jobs j ON j.video_id = sc.video_id
              WHERE j.stage = ?2 GROUP BY j.state"
        ))?;
        for row in st.query_map(params![project_id, stage], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (state, n) = row?;
            match state.as_str() {
                "pending" => c.pending = n,
                "running" => c.running = n,
                "done" => c.done = n,
                "failed" => c.failed = n,
                "skipped" => c.skipped = n,
                _ => {}
            }
        }
        s.stages.push(c);
    }
    s.steadiness_measured = db
        .conn
        .query_row(
            &format!(
                "SELECT COUNT(DISTINCT sc.video_id)
                   FROM ({SCOPE}) sc
                  WHERE {}",
                analysed_sql("steadiness", "motion_windows", "sc.video_id")
            ),
            [project_id],
            |r| r.get(0),
        )
        .unwrap_or(0);
    s.faces_measured = db
        .conn
        .query_row(
            &format!(
                "SELECT COUNT(DISTINCT sc.video_id)
                   FROM ({SCOPE}) sc
                  WHERE {}",
                analysed_sql("faces", "face_tracks", "sc.video_id")
            ),
            [project_id],
            |r| r.get(0),
        )
        .unwrap_or(0);
    Ok(s)
}

/// Where a stage stands for a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    /// Turned off for the project (or by a setting it depends on): never run, never pending.
    Off,
    /// On, but cannot run on this machine yet (its models are missing).
    Unavailable,
    /// Work left for the next run (possibly postponed: see the reason).
    Pending,
    /// Every remaining video failed it as often as it is retried.
    Failed,
    Done,
}

/// One stage of a project's pipeline as the Project menu shows it. Whether a stage is *running*
/// is the queue's to say; this is what the database says.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StageStatus {
    pub stage: String,
    /// The stage's name in the menu ("Camera shake").
    pub label: String,
    pub state: StageState,
    /// Videos done and in scope for the stage.
    pub done: i64,
    pub total: i64,
    /// Why it is off, unavailable, pending or failed ("postponed: vision server …").
    pub reason: Option<String>,
}

/// The stages the Project menu lists, in pipeline order, with their names there.
pub const MENU_STAGES: &[(&str, &str)] = &[
    ("frames", "Keyframes"),
    ("transcribe", "Transcription"),
    ("describe", "Frame descriptions"),
    ("embed", "Search embeddings"),
    ("steadiness", "Camera shake"),
    ("faces", "Faces"),
    ("speakers", "Speakers"),
];

/// Every menu stage's status for a project. A stage the project turned off is `off` — through
/// [`stage_enabled`], the same gate the pipeline uses — whatever its videos' state.
/// `face_models`: whether the face models are installed.
pub fn stage_statuses(
    db: &Db,
    project_id: i64,
    config: &crate::config::Config,
    face_models: bool,
) -> Result<Vec<StageStatus>, Error> {
    let pipeline = db.project(project_id)?.pipeline;
    let scope =
        |extra: &str| format!("SELECT COUNT(*) FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id WHERE {extra}");
    let count = |sql: String| -> Result<i64, Error> { Ok(db.conn.query_row(&sql, [project_id], |r| r.get(0))?) };
    let mut out = Vec::new();
    for &(stage, label) in MENU_STAGES {
        let mut st = StageStatus {
            stage: stage.into(),
            label: label.into(),
            state: StageState::Done,
            done: 0,
            total: 0,
            reason: None,
        };
        if !stage_enabled(stage, None, Some(&pipeline)) {
            st.state = StageState::Off;
            out.push(st);
            continue;
        }
        match stage {
            "steadiness" | "faces" => {
                let output = if stage == "steadiness" { "motion_windows" } else { "face_tracks" };
                st.total = count(scope("COALESCE(v.duration_s, 0) > 0"))?;
                st.done = count(scope(&format!(
                    "COALESCE(v.duration_s, 0) > 0 AND {}",
                    analysed_sql(stage, output, "v.id")
                )))?;
                let given_up = count(scope(&format!(
                    "EXISTS (SELECT 1 FROM jobs j WHERE j.video_id = v.id AND j.stage = '{stage}'
                                AND j.state = 'failed' AND j.attempts >= {MAX_ATTEMPTS})"
                )))?;
                if stage == "steadiness" && config.script.max_shake_jerk <= 0.0 {
                    st.state = StageState::Off;
                    st.reason = Some("the camera shake check is off in Settings".into());
                } else if stage == "faces" && !face_models {
                    st.state = StageState::Unavailable;
                    st.reason = Some("face models not downloaded — Models page".into());
                } else if st.done < st.total {
                    st.state = if given_up >= st.total - st.done { StageState::Failed } else { StageState::Pending };
                    st.reason = last_error(db, project_id, stage)?;
                }
            }
            "speakers" => {
                let pending = crate::speakers::pending_videos(db, project_id)? as i64;
                st.total = count(scope("EXISTS (SELECT 1 FROM transcript_segments t WHERE t.video_id = v.id)"))?;
                st.done = (st.total - pending).max(0);
                if pending > 0 {
                    st.state = StageState::Pending;
                }
            }
            _ => {
                let by_state = |state: &str| {
                    count(scope(&format!(
                        "EXISTS (SELECT 1 FROM jobs j WHERE j.video_id = v.id AND j.stage = '{stage}' AND j.state = '{state}')"
                    )))
                };
                // A job left `running` by a run that was cut short is pending work.
                let (pending, done, failed) =
                    (by_state("pending")? + by_state("running")?, by_state("done")?, by_state("failed")?);
                st.total = pending + done + failed;
                st.done = done;
                if pending > 0 {
                    st.state = StageState::Pending;
                } else if failed > 0 {
                    st.state = StageState::Failed;
                }
                if st.state != StageState::Done {
                    st.reason = last_error(db, project_id, stage)?;
                }
            }
        }
        out.push(st);
    }
    Ok(out)
}

/// The most recent reason a not-done job of `stage` in the project gave.
fn last_error(db: &Db, project_id: i64, stage: &str) -> Result<Option<String>, Error> {
    Ok(db
        .conn
        .query_row(
            &format!(
                "SELECT j.last_error FROM ({SCOPE}) sc JOIN jobs j ON j.video_id = sc.video_id
                  WHERE j.stage = ?2 AND j.state <> 'done' AND j.last_error IS NOT NULL
                  ORDER BY j.updated_at DESC LIMIT 1"
            ),
            params![project_id, stage],
            |r| r.get(0),
        )
        .optional()?)
}

pub fn videos(db: &Db, project_id: Option<i64>) -> Result<Vec<VideoRow>, Error> {
    videos_with_shake(db, project_id, 0.0, 0.0, 0.0)
}

/// [`videos`], also classifying each one's camera work and totalling the seconds it measured
/// shakier than its limit (`script.max_shake_jerk` raised by `script.shake_relative` times the
/// clip's own level; a `max_shake` of 0 reports none).
pub fn videos_with_shake(
    db: &Db,
    project_id: Option<i64>,
    max_shake: f64,
    shake_relative: f64,
    max_sway: f64,
) -> Result<Vec<VideoRow>, Error> {
    let mut st = db.conn.prepare(&format!(
        "SELECT v.id, sc.path, sc.copies, v.size, v.duration_s, v.width, v.height, v.fps, COALESCE(v.vfr, 0),
                v.vcodec, v.has_audio, v.status, v.error, v.language,
                (SELECT COUNT(*) FROM transcript_segments t WHERE t.video_id = v.id),
                (SELECT j.state FROM jobs j WHERE j.video_id = v.id AND j.stage = 'transcribe'),
                (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id),
                COALESCE(v.category, 'other'), COALESCE(v.category_confidence, 0.0), COALESCE(v.category_source, 'auto'),
                v.category_tag, sc.folder_ids,
                (EXISTS (SELECT 1 FROM face_tracks ft WHERE ft.video_id = v.id)
                 OR EXISTS (SELECT 1 FROM jobs fj WHERE fj.video_id = v.id AND fj.stage = 'faces' AND fj.state = 'done'))
           FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id
          ORDER BY sc.path"
    ))?;
    let rows = st.query_map([project_id], |r| {
        Ok(VideoRow {
            id: r.get(0)?,
            folder_ids: serde_json::from_str(&r.get::<_, String>(21)?).unwrap_or_default(),
            path: PathBuf::from(r.get::<_, String>(1)?),
            copies: r.get(2)?,
            size: r.get(3)?,
            duration_s: r.get(4)?,
            width: r.get(5)?,
            height: r.get(6)?,
            fps: r.get(7)?,
            vfr: r.get(8)?,
            vcodec: r.get(9)?,
            has_audio: r.get(10)?,
            status: r.get(11)?,
            error: r.get(12)?,
            language: r.get(13)?,
            segments: r.get(14)?,
            transcribe: r.get(15)?,
            frames: r.get(16)?,
            shaky_s: 0.0,
            steadiness_measured: false,
            faces_measured: r.get(22)?,
            camera: String::new(),
            category: r.get(17)?,
            category_confidence: r.get(18)?,
            category_source: r.get(19)?,
            category_tag: r.get(20)?,
        })
    })?;
    let mut out: Vec<VideoRow> = rows.collect::<Result<_, _>>()?;
    for row in &mut out {
        let windows = db.motion_windows(row.id).unwrap_or_default();
        row.steadiness_measured = !windows.is_empty();
        row.camera = crate::steadiness::camera_style(&windows).as_str().to_string();
        if max_shake > 0.0 && !windows.is_empty() {
            let limit = crate::steadiness::shake_limit(&windows, max_shake, shake_relative);
            row.shaky_s = windows
                .iter()
                .filter(|w| w.jerk > limit || (max_sway > 0.0 && w.sway > max_sway))
                .map(|w| w.end_s - w.start_s)
                .sum();
        }
    }
    Ok(out)
}

/// A single [`VideoRow`] with steadiness, camera style, and shakes.
pub fn video_by_id(
    db: &Db,
    video_id: i64,
    max_shake: f64,
    shake_relative: f64,
    max_sway: f64,
) -> Result<VideoRow, Error> {
    let mut st = db.conn.prepare(
        "SELECT v.id, sc.path, sc.copies, v.size, v.duration_s, v.width, v.height, v.fps, COALESCE(v.vfr, 0),
                v.vcodec, v.has_audio, v.status, v.error, v.language,
                (SELECT COUNT(*) FROM transcript_segments t WHERE t.video_id = v.id),
                (SELECT j.state FROM jobs j WHERE j.video_id = v.id AND j.stage = \x27transcribe\x27),
                (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id),
                COALESCE(v.category, 'other'), COALESCE(v.category_confidence, 0.0), COALESCE(v.category_source, 'auto'),
                v.category_tag, sc.folder_ids,
                (EXISTS (SELECT 1 FROM face_tracks ft WHERE ft.video_id = v.id)
                 OR EXISTS (SELECT 1 FROM jobs fj WHERE fj.video_id = v.id AND fj.stage = 'faces' AND fj.state = 'done'))
           FROM (
             SELECT vf.video_id, MIN(vf.path) AS path, COUNT(DISTINCT vf.path) AS copies, json_group_array(DISTINCT vf.folder_id) AS folder_ids
               FROM video_files vf
              WHERE vf.video_id = ?1
              GROUP BY vf.video_id
           ) sc JOIN videos v ON v.id = sc.video_id",
    )?;
    let mut row = st.query_row([video_id], |r| {
        Ok(VideoRow {
            id: r.get(0)?,
            folder_ids: serde_json::from_str(&r.get::<_, String>(21)?).unwrap_or_default(),
            path: PathBuf::from(r.get::<_, String>(1)?),
            copies: r.get(2)?,
            size: r.get(3)?,
            duration_s: r.get(4)?,
            width: r.get(5)?,
            height: r.get(6)?,
            fps: r.get(7)?,
            vfr: r.get(8)?,
            vcodec: r.get(9)?,
            has_audio: r.get(10)?,
            status: r.get(11)?,
            error: r.get(12)?,
            language: r.get(13)?,
            segments: r.get(14)?,
            transcribe: r.get(15)?,
            frames: r.get(16)?,
            shaky_s: 0.0,
            steadiness_measured: false,
            faces_measured: r.get(22)?,
            camera: String::new(),
            category: r.get(17)?,
            category_confidence: r.get(18)?,
            category_source: r.get(19)?,
            category_tag: r.get(20)?,
        })
    })?;
    let windows = db.motion_windows(row.id).unwrap_or_default();
    row.steadiness_measured = !windows.is_empty();
    row.camera = crate::steadiness::camera_style(&windows).as_str().to_string();
    if max_shake > 0.0 && !windows.is_empty() {
        let limit = crate::steadiness::shake_limit(&windows, max_shake, shake_relative);
        row.shaky_s = windows
            .iter()
            .filter(|w| w.jerk > limit || (max_sway > 0.0 && w.sway > max_sway))
            .map(|w| w.end_s - w.start_s)
            .sum();
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    /// A fake ffprobe: prints fixed JSON, or fails for files whose name contains "broken".
    fn fake_ffprobe(dir: &Path) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let p = dir.join("ffprobe");
            std::fs::write(
                &p,
                r#"#!/bin/sh
for a in "$@"; do last="$a"; done
case "$last" in *broken*) echo "Invalid data found when processing input" >&2; exit 1;; esac
echo '{"streams":[{"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"r_frame_rate":"25/1","avg_frame_rate":"25/1"},{"codec_type":"audio","codec_name":"aac"}],"format":{"duration":"10.0"}}'
"#,
            )
            .unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        }
        #[cfg(not(unix))]
        {
            let _ = dir;
            unimplemented!("fake ffprobe script is unix-only")
        }
    }

    fn rt(ffprobe: &Path) -> Runtime {
        Runtime {
            ffmpeg: "ffmpeg".into(),
            describe_concurrency: 1,
            ffprobe: ffprobe.to_path_buf(),
            stt: SttSetup::Unavailable("not configured in this test".into()),
            data_dir: std::env::temp_dir(),
            frames: None,
            vision: VisionSetup::Unavailable("not configured in this test".into()),
            embed: EmbedSetup::Unavailable("not configured in this test".into()),
            steadiness: None,
            measure_audio: false,
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ghostreels_own_renders_are_not_indexed_as_footage() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("interview.mp4"), b"real footage");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.renders), (1, 0));

        // A clip export rendered elsewhere, recorded, then moved into the watched folder.
        let rendered = tmp.path().join("out/01-hook.mp4");
        write(&rendered, b"a ghostreel render");
        record_render(&db, &rendered, None).unwrap();
        std::fs::create_dir_all(media.join("clips")).unwrap();
        std::fs::rename(&rendered, media.join("clips/01-hook-renamed.mp4")).unwrap();
        // A render indexed as footage before renders were recognised.
        let old = media.join("previews/script_3_v1.mp4");
        write(&old, b"an older render");
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.renders), (1, 1), "the moved render is skipped, the older one indexed");
        assert_eq!(videos(&db, Some(p.id)).unwrap().len(), 2);

        // Once it is known as a render (its exports row), the next scan takes it out of the library.
        db.conn
            .execute(
                "INSERT INTO scripts(id, project_id, title, version, script_json, created_at) VALUES (3, ?1, 't', 1, '{}', 0)",
                [p.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO exports(script_id, format, path, created_at) VALUES (3, 'preview_mp4', ?1, 0)",
                [old.to_string_lossy()],
            )
            .unwrap();
        let mut events = Vec::new();
        let s = run(&mut db, &rt(&bin), &opts, |e| events.push(e)).await.unwrap();
        assert_eq!((s.new, s.renders), (0, 2), "both renders are skipped, reported, not failed");
        assert!(events.iter().any(|e| matches!(e, Event::RendersSkipped { count: 2 })));
        let left: Vec<String> = videos(&db, Some(p.id))
            .unwrap()
            .into_iter()
            .map(|v| v.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, vec!["interview.mp4".to_string()]);
        assert_eq!(s.jobs_failed, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scan_probe_rescan_move_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("a.mp4"), b"video-a");
        write(&media.join("sub/b.MOV"), b"video-b");
        write(&media.join("sub/copy-of-a.mp4"), b"video-a");
        write(&media.join("broken.mkv"), b"junk");
        write(&media.join("notes.txt"), b"not a video");
        write(&media.join(".hidden/c.mp4"), b"video-c");
        write(&media.join("Adobe Premiere Pro Video Previews/Seq 01.PRV/render.mov"), b"render");
        write(&media.join("proj/Adobe Premiere Pro Auto-Save/x.mp4"), b"autosave");
        write(&media.join("proj/Media Cache Files/cache.mpeg"), b"cache");
        let bin = fake_ffprobe(tmp.path());

        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };

        let mut events = Vec::new();
        let s = run(&mut db, &rt(&bin), &opts, |e| events.push(e)).await.unwrap();
        assert_eq!(
            (s.new, s.unchanged, s.removed),
            (4, 0, 0),
            "4 files; hidden dir, Premiere renders/caches and .txt skipped"
        );
        assert_eq!((s.jobs_done, s.jobs_failed), (2, 1), "a/copy-of-a share content; broken fails");

        let st = status(&db, Some(p.id)).unwrap();
        assert_eq!(st.videos, 3);
        assert_eq!((st.stages[0].done, st.stages[0].failed), (2, 1));
        let rows = videos(&db, Some(p.id)).unwrap();
        let a = rows.iter().find(|v| v.path.ends_with("a.mp4")).unwrap();
        assert_eq!((a.copies, a.duration_s, a.width), (2, Some(10.0), Some(1920)));

        // Unchanged rescan does nothing; the failed job is retried (attempt 2).
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unchanged, s.jobs_done, s.jobs_failed), (0, 4, 0, 1));

        // Move a file: same content → no new video, no new jobs.
        std::fs::rename(media.join("sub/b.MOV"), media.join("b-moved.mov")).unwrap();
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.new, s.removed, s.jobs_done), (1, 1, 0));
        assert_eq!(status(&db, Some(p.id)).unwrap().videos, 3);

        // Third failure exhausts retries; --retry-failed forces it.
        run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!(s.jobs_failed, 0, "gave up after {MAX_ATTEMPTS} attempts");
        let retry = Options { retry_failed: true, ..opts.clone() };
        assert_eq!(run(&mut db, &rt(&bin), &retry, |_| {}).await.unwrap().jobs_failed, 1);

        // Delete: the video disappears from the project.
        std::fs::remove_file(media.join("b-moved.mov")).unwrap();
        run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!(status(&db, Some(p.id)).unwrap().videos, 2);
        assert!(events.iter().any(|e| matches!(e, Event::JobFailed { .. })));
        let last = events.iter().rev().find_map(|e| match e {
            Event::Progress(p) => Some(p.clone()),
            _ => None,
        });
        let last = last.expect("progress events emitted");
        assert_eq!(last.fraction, 1.0);
        assert_eq!(last.phase_done, last.phase_total);
    }

    /// A fake ffmpeg that writes its pid to `pidfile` and then hangs without output: a decode
    /// that never ends, to prove a stop reaches inside a long per-video stage.
    #[cfg(unix)]
    fn hanging_ffmpeg(dir: &Path, pidfile: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("ffmpeg-hang");
        std::fs::write(&p, format!("#!/bin/sh\necho $$ > '{}'\nexec sleep 60\n", pidfile.display())).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[cfg(unix)]
    fn process_gone(pid: u32) -> bool {
        (0..100).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Err(_) => true,
                Ok(stat) => stat.contains(") Z"),
            }
        })
    }

    /// A project with one probed 10-minute video, at `media`.
    #[cfg(unix)]
    async fn one_probed_video(db: &mut Db, tmp: &Path, pipeline: Option<crate::projects::PipelineConfig>) -> i64 {
        let media = tmp.join("media");
        write(&media.join("long.mp4"), b"video-long");
        let mut np = NewProject::named("P");
        if let Some(pl) = pipeline {
            np = np.with_pipeline(pl);
        }
        let p = db.create_project(&np).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };
        run(db, &rt(&fake_ffprobe(tmp)), &opts, |_| {}).await.unwrap();
        db.conn.execute("UPDATE videos SET duration_s = 600", []).unwrap();
        p.id
    }

    /// Stop reaches inside one long video: the camera-shake decode is dropped within the poll,
    /// its ffmpeg is killed, and nothing is stored for the video (it is measured next run).
    #[cfg(unix)]
    #[tokio::test]
    async fn stop_reaches_inside_a_long_video_and_kills_its_child() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pid = one_probed_video(&mut db, tmp.path(), None).await;
        let pidfile = tmp.path().join("ffmpeg.pid");
        let mut runtime = rt(&fake_ffprobe(tmp.path()));
        runtime.ffmpeg = hanging_ffmpeg(tmp.path(), &pidfile);
        runtime.steadiness = Some(crate::runtime::SteadinessOptions { window_s: 2.0, stride_s: 1.0 });

        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopper = flag.clone();
        let pidfile2 = pidfile.clone();
        tokio::spawn(async move {
            // Stop once the decode is under way.
            while !pidfile2.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            stopper.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        let opts = Options { project_id: Some(pid), cancel: Some(flag), ..Default::default() };
        let mut tracker = Tracker::new(&[("steadiness", 1.0)]);
        let started = std::time::Instant::now();
        let s = run_steadiness_jobs(&mut db, &runtime, &opts, false, &mut tracker, &mut |_| {}).await.unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "took {:?}", started.elapsed());
        assert_eq!((s.measured, s.failed), (0, 0));
        let child: u32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
        assert!(process_gone(child), "ffmpeg {child} survived the stop");
        let rows: i64 = db.conn.query_row("SELECT COUNT(*) FROM motion_windows", [], |r| r.get(0)).unwrap();
        let jobs: i64 =
            db.conn.query_row("SELECT COUNT(*) FROM jobs WHERE stage = 'steadiness'", [], |r| r.get(0)).unwrap();
        assert_eq!((rows, jobs), (0, 0), "a stopped video stores nothing and stays to do");
    }

    /// A decode that ffmpeg did not finish is a failure, not a short (and falsely steady)
    /// measurement; the failure is recorded so the status can say why.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_decode_is_a_failure_not_a_short_measurement() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pid = one_probed_video(&mut db, tmp.path(), None).await;
        // Some frames, then a crash.
        let bad = tmp.path().join("ffmpeg-crash");
        std::fs::write(&bad, "#!/bin/sh\nhead -c 2000000 /dev/zero\nexit 1\n").unwrap();
        std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut runtime = rt(&fake_ffprobe(tmp.path()));
        runtime.ffmpeg = bad;
        runtime.steadiness = Some(crate::runtime::SteadinessOptions { window_s: 2.0, stride_s: 1.0 });
        let opts = Options { project_id: Some(pid), ..Default::default() };
        let mut tracker = Tracker::new(&[("steadiness", 1.0)]);
        let mut events = Vec::new();
        let s =
            run_steadiness_jobs(&mut db, &runtime, &opts, false, &mut tracker, &mut |e| events.push(e)).await.unwrap();
        assert_eq!((s.measured, s.failed), (0, 1));
        let rows: i64 = db.conn.query_row("SELECT COUNT(*) FROM motion_windows", [], |r| r.get(0)).unwrap();
        assert_eq!(rows, 0);
        let mut problems = Vec::new();
        for e in &events {
            StageProblem::observe(&mut problems, e);
        }
        assert!(matches!(&problems[..], [StageProblem::Failed { stage, videos: 1, .. }] if stage == "steadiness"));
    }

    /// A stage the project turned off never runs — not in the pipeline, not as a follow-up — and
    /// the Project menu shows it `off`, through the same gate.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_disabled_stage_never_runs_and_shows_off() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pipeline = crate::projects::PipelineConfig { steadiness: false, ..Default::default() };
        let pid = one_probed_video(&mut db, tmp.path(), Some(pipeline)).await;
        let pidfile = tmp.path().join("ffmpeg.pid");
        let mut runtime = rt(&fake_ffprobe(tmp.path()));
        // Were steadiness to run, this ffmpeg would hang the test.
        runtime.ffmpeg = hanging_ffmpeg(tmp.path(), &pidfile);
        runtime.steadiness = Some(crate::runtime::SteadinessOptions { window_s: 2.0, stride_s: 1.0 });
        let paths =
            crate::paths::Paths { config_file: tmp.path().join("config.toml"), data_dir: tmp.path().to_path_buf() };
        let mut config = crate::config::Config::default();
        config.script.max_shake_jerk = 1.0;
        let opts = Options { project_id: Some(pid), stages: None, ..Default::default() };
        let mut events = Vec::new();
        let s = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            run_pipeline(&mut db, &paths, &config, &runtime, &opts, |e| events.push(e)),
        )
        .await
        .expect("a disabled stage must not run")
        .unwrap();
        assert!(!pidfile.exists(), "camera shake ran although the project turned it off");
        assert!(!events.iter().any(|e| matches!(e, Event::JobStarted { stage, .. } if stage == "steadiness")));
        assert!(!s.cancelled);

        let statuses = stage_statuses(&db, pid, &config, false).unwrap();
        let shake = statuses.iter().find(|s| s.stage == "steadiness").unwrap();
        assert_eq!(shake.state, StageState::Off);
        let faces = statuses.iter().find(|s| s.stage == "faces").unwrap();
        assert_eq!(faces.state, StageState::Unavailable, "on, but its models are missing");
        assert!(stage_enabled("steadiness", Some(&["steadiness".to_string()]), Some(&pipeline_off_shake())));
    }

    fn pipeline_off_shake() -> crate::projects::PipelineConfig {
        crate::projects::PipelineConfig { steadiness: false, ..Default::default() }
    }

    /// A stage postponed because its backend is unavailable makes the run partial with the
    /// reason, keeps its jobs pending and lets the Project menu say why.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_postponed_stage_makes_the_run_partial() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pipeline =
            crate::projects::PipelineConfig { steadiness: false, faces: false, speakers: false, ..Default::default() };
        let pid = one_probed_video(&mut db, tmp.path(), Some(pipeline)).await;
        // Its keyframes exist and wait to be described.
        let vid: i64 = db.conn.query_row("SELECT id FROM videos", [], |r| r.get(0)).unwrap();
        db.conn.execute("UPDATE jobs SET state = 'done' WHERE stage = 'frames'", []).unwrap();
        db.conn
            .execute("INSERT INTO frames(video_id, t_s, thumb_path, phash) VALUES (?1, 1.0, 'f.jpg', 0)", [vid])
            .unwrap();
        let mut runtime = rt(&fake_ffprobe(tmp.path()));
        runtime.vision = VisionSetup::Unavailable("the built-in model can't read images".into());
        let paths =
            crate::paths::Paths { config_file: tmp.path().join("config.toml"), data_dir: tmp.path().to_path_buf() };
        let config = crate::config::Config::default();
        let opts = Options { project_id: Some(pid), ..Default::default() };
        let mut events = Vec::new();
        let s = run_pipeline(&mut db, &paths, &config, &runtime, &opts, |e| events.push(e)).await.unwrap();

        assert_eq!(s.run_state(), crate::runs::RunState::Partial);
        assert!(
            s.problems_line().unwrap().contains("Describing frames postponed: the built-in model can't read images")
        );
        let describe =
            stage_statuses(&db, pid, &config, true).unwrap().into_iter().find(|s| s.stage == "describe").unwrap();
        assert_eq!(describe.state, StageState::Pending);
        assert!(describe.reason.unwrap().contains("can't read images"));

        // The run log puts the warning on its own step, which ends as a warning.
        let mut log = crate::runs::StepLog::new();
        log.report("probe", "1 of 1");
        for e in &events {
            if let Some(EventReport::Event { phase, event }) = e.run_report() {
                log.event_in(&phase, event);
            }
        }
        log.finish(s.run_state(), "");
        let step = log.steps().iter().find(|st| st.label == "Describing frames").expect("its own step");
        assert_eq!(step.state, crate::runs::StepState::Warning);
        assert_eq!(log.steps()[0].state, crate::runs::StepState::Done, "the warning did not land on probe");
    }

    /// Faces in the pipeline: a video with no faces is recorded as analysed (not redone on every
    /// run nor shown pending for ever). Needs the face models; skipped unless
    /// `GHOSTREEL_FACE_MODELS` names their folder.
    #[cfg(unix)]
    #[tokio::test]
    async fn faces_are_recorded_even_when_there_are_none() {
        let models = std::env::var_os("GHOSTREEL_FACE_MODELS").map(PathBuf::from).filter(|d| {
            d.join("face_detection_yunet_2023mar.onnx").is_file()
                && d.join("face_recognition_sface_2021dec.onnx").is_file()
        });
        let ffmpeg_ok = std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success());
        let (Some(models), true) = (models, ffmpeg_ok) else {
            eprintln!("skipped: face models or ffmpeg not available");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        let video = media.join("bars.mp4");
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest", "-pix_fmt", "yuv420p"])
            .arg(&video)
            .status()
            .unwrap();
        assert!(ok.success());
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let mut runtime = rt(&fake_ffprobe(tmp.path()));
        runtime.ffmpeg = "ffmpeg".into();
        run(&mut db, &runtime, &Options { project_id: Some(p.id), ..Default::default() }, |_| {}).await.unwrap();
        db.conn.execute("UPDATE videos SET duration_s = 3, width = 320, height = 240", []).unwrap();

        let paths =
            crate::paths::Paths { config_file: tmp.path().join("config.toml"), data_dir: tmp.path().to_path_buf() };
        let mut config = crate::config::Config::default();
        config.models.search_paths = vec![models];
        let opts = Options { project_id: Some(p.id), ..Default::default() };
        let mut tracker = Tracker::new(&[("faces", 1.0)]);
        let s =
            run_face_jobs(&mut db, &paths, &config, &runtime, &opts, false, &mut tracker, &mut |_| {}).await.unwrap();
        assert_eq!((s.measured, s.failed), (1, 0));
        let state: String =
            db.conn.query_row("SELECT state FROM jobs WHERE stage = 'faces'", [], |r| r.get(0)).unwrap();
        assert_eq!(state, "done");
        assert_eq!(status(&db, Some(p.id)).unwrap().faces_measured, 1, "analysed, with nothing to find");
        let again =
            run_face_jobs(&mut db, &paths, &config, &runtime, &opts, false, &mut tracker, &mut |_| {}).await.unwrap();
        assert_eq!(again.measured, 0, "not analysed again");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn projects_share_indexed_videos_and_missing_folders_keep_index() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("a.mp4"), b"video-a");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p1 = db.create_project(&NewProject::named("One")).unwrap();
        let p2 = db.create_project(&NewProject::named("Two")).unwrap();
        db.add_folder(p1.id, &media, true).unwrap();
        db.add_folder(p2.id, &media, true).unwrap();

        let s =
            run(&mut db, &rt(&bin), &Options { project_id: Some(p1.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!(s.jobs_done, 1);
        // Project Two sees the already-probed video without any work.
        let s =
            run(&mut db, &rt(&bin), &Options { project_id: Some(p2.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!((s.new, s.jobs_done), (0, 0));
        assert_eq!(status(&db, Some(p2.id)).unwrap().stages[0].done, 1);

        // Unmounted drive: folder missing → files are NOT dropped.
        std::fs::rename(&media, tmp.path().join("unplugged")).unwrap();
        let mut missing = false;
        run(&mut db, &rt(&bin), &Options::default(), |e| missing |= matches!(e, Event::FolderMissing { .. }))
            .await
            .unwrap();
        assert!(missing);
        assert_eq!(status(&db, None).unwrap().videos, 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn overlapping_folders_of_different_projects_both_see_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        write(&lib.join("screen/demo.mp4"), b"video-demo");
        write(&lib.join("other.mp4"), b"video-other");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let teaser = db.create_project(&NewProject::named("Teaser")).unwrap();
        let docs = db.create_project(&NewProject::named("Docs")).unwrap();
        db.add_folder(teaser.id, &lib, true).unwrap();
        db.add_folder(docs.id, &lib.join("screen"), true).unwrap();

        for _ in 0..3 {
            run(&mut db, &rt(&bin), &Options::default(), |_| {}).await.unwrap();
        }
        assert_eq!(status(&db, Some(teaser.id)).unwrap().videos, 2);
        assert_eq!(status(&db, Some(docs.id)).unwrap().videos, 1);
        let s = run(&mut db, &rt(&bin), &Options::default(), |_| {}).await.unwrap();
        assert_eq!((s.new, s.changed, s.removed, s.unchanged), (0, 0, 0, 3), "stable across rescans");
        let rows = videos(&db, None).unwrap();
        assert!(rows.iter().all(|v| v.copies == 1), "one file seen through two folders is not a copy");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fresh_files_wait_to_settle() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("copying.mp4"), b"partial");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let watch = Options { project_id: Some(p.id), settle_secs: 3600, ..Default::default() };
        let s = run(&mut db, &rt(&bin), &watch, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unsettled), (0, 1));
        let s =
            run(&mut db, &rt(&bin), &Options { project_id: Some(p.id), ..Default::default() }, |_| {}).await.unwrap();
        assert_eq!((s.new, s.unsettled), (1, 0));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_stage_end_to_end() {
        let have =
            |b: &str| crate::proc::std_command(b).arg("-version").output().map(|o| o.status.success()).unwrap_or(false);
        if !have("ffmpeg") || !have("ffprobe") {
            eprintln!("skipping: ffmpeg/ffprobe not installed");
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        let make = |args: &[&str], out: &str| {
            let ok = crate::proc::std_command("ffmpeg")
                .args(["-v", "error"])
                .args(args)
                .arg(media.join(out))
                .status()
                .unwrap()
                .success();
            assert!(ok, "ffmpeg {out}");
        };
        make(
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=d=3",
                "-t",
                "3",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
            ],
            "talk.mp4",
        );
        make(&["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "2", "-c:v", "mpeg4"], "silent.mp4");

        let asr = tmp.path().join("asr");
        std::fs::write(
            &asr,
            "#!/bin/sh\nread -r request\necho '{\"type\":\"loaded\"}'\necho '{\"type\":\"progress\",\"percent\":100}'\necho '{\"type\":\"segment\",\"start\":0.5,\"end\":2.5,\"text\":\"hello world\",\"no_speech\":0.1}'\necho '{\"type\":\"done\",\"language\":\"en\",\"audio_s\":3}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&asr, std::fs::Permissions::from_mode(0o755)).unwrap();

        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let opts = Options { project_id: Some(p.id), ..Default::default() };

        // 1. Transcription unavailable: probe runs, transcribe stays pending (not failed).
        let unavailable = Runtime {
            describe_concurrency: 1,
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            stt: SttSetup::Unavailable("GhostPen down".into()),
            data_dir: tmp.path().join("data"),
            frames: None,
            vision: VisionSetup::Unavailable("not configured in this test".into()),
            embed: EmbedSetup::Unavailable("not configured in this test".into()),
            steadiness: None,
            measure_audio: false,
        };
        let mut events = Vec::new();
        let s = run(&mut db, &unavailable, &opts, |e| events.push(e)).await.unwrap();
        assert_eq!((s.jobs_done, s.jobs_failed), (2, 0));
        assert!(events.iter().any(|e| matches!(e, Event::StageUnavailable { .. })));
        let st = status(&db, Some(p.id)).unwrap();
        let tr = st.stages.iter().find(|c| c.stage == "transcribe").unwrap();
        assert_eq!((tr.pending, tr.skipped, tr.failed), (1, 1, 0), "silent video skipped, talk waits");

        // 2. Local engine available: the waiting video gets its transcript.
        let model = tmp.path().join("ggml-test.bin");
        std::fs::write(&model, b"m").unwrap();
        let local = Runtime {
            stt: SttSetup::Ready(Engine::Local { asr_bin: asr, model, language: "auto".into() }),
            frames: Some(crate::frames::FrameOptions::default()),
            ..unavailable
        };
        let mut events = Vec::new();
        let s = run(&mut db, &local, &opts, |e| events.push(e)).await.unwrap();
        let failures: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Event::JobFailed { video_id, stage, error } => Some(format!("#{video_id} {stage}: {error}")),
                _ => None,
            })
            .collect();
        assert_eq!((s.jobs_done, s.jobs_failed), (3, 0), "1 transcript + 2 frame jobs; failures: {failures:?}");
        let rows = videos(&db, Some(p.id)).unwrap();
        let talk = rows.iter().find(|v| v.path.ends_with("talk.mp4")).unwrap();
        assert_eq!(
            (talk.segments, talk.language.as_deref(), talk.transcribe.as_deref()),
            (1, Some("en"), Some("done"))
        );
        let silent = rows.iter().find(|v| v.path.ends_with("silent.mp4")).unwrap();
        assert_eq!(silent.transcribe.as_deref(), Some("skipped"));
        assert_eq!(transcript(&db, talk.id).unwrap()[0].text, "hello world");
        // Keyframes: both videos get frames stored under the data dir, paths resolve to files.
        assert!(talk.frames >= 1 && silent.frames >= 1, "{} {}", talk.frames, silent.frames);
        let fr = frames(&db, &local.data_dir, talk.id).unwrap();
        assert!(fr.iter().all(|f| f.path.is_file() && f.path.starts_with(&local.data_dir)));
        let last =
            events.iter().rev().find_map(|e| if let Event::Progress(p) = e { Some(p.clone()) } else { None }).unwrap();
        assert_eq!(last.fraction, 1.0);
        assert!(events.iter().any(|e| matches!(e, Event::StageBackend { .. })));

        // 3. Local vision helper describes every frame (speech context passed for the talking video).
        let helper = tmp.path().join("llm");
        let log = tmp.path().join("prompts.log");
        std::fs::write(
            &helper,
            format!(
                r#"#!/bin/sh
echo '{{"ready":true,"vision":true,"embed_dim":null}}'
while IFS= read -r line; do
  printf '%s\n' "$line" >> {log}
  id=$(printf '%s' "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  printf '{{"id":%s,"ok":true,"content":"{{\\"description\\":\\"test pattern\\",\\"visible_text\\":[\\"PM5544\\"],\\"objects\\":[],\\"setting\\":\\"studio\\",\\"shot\\":\\"title card\\",\\"tags\\":[]}}"}}\n' "$id"
done
"#,
                log = log.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (model, mmproj) = crate::models::bonsai_vision();
        let fake_model = tmp.path().join(&model.file_name);
        let fake_mmproj = tmp.path().join(&mmproj.file_name);
        std::fs::write(&fake_model, b"m").unwrap();
        std::fs::write(&fake_mmproj, b"p").unwrap();
        let with_vision = Runtime {
            vision: VisionSetup::Local {
                helper,
                models_dir: tmp.path().join("models"),
                model,
                mmproj: Some(mmproj),
                found: vec![fake_model, fake_mmproj],
                runtime: Default::default(),
            },
            ..local.clone()
        };
        let s = run(&mut db, &with_vision, &opts, |_| {}).await.unwrap();
        assert_eq!((s.jobs_done, s.jobs_failed), (2, 0), "describe job per video");
        let fr = frames(&db, &with_vision.data_dir, talk.id).unwrap();
        assert!(fr.iter().all(|f| f.visible_text.as_deref() == Some("PM5544")));
        let prompts = std::fs::read_to_string(&log).unwrap();
        assert!(prompts.contains("hello world"), "speech context sent for the talking video");

        // 4. Nothing left: another run does no work.
        let s = run(&mut db, &with_vision, &opts, |_| {}).await.unwrap();
        assert_eq!(s.jobs_done, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancel_stops_before_jobs() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        write(&media.join("a.mp4"), b"video-a");
        let bin = fake_ffprobe(tmp.path());
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let opts = Options { project_id: Some(p.id), cancel: Some(flag.clone()), ..Default::default() };
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert!(s.cancelled);
        assert_eq!((s.new, s.jobs_done), (1, 0), "scan happened, jobs did not");
        flag.store(false, std::sync::atomic::Ordering::Relaxed);
        let s = run(&mut db, &rt(&bin), &opts, |_| {}).await.unwrap();
        assert_eq!((s.cancelled, s.jobs_done), (false, 1), "resumes on the next run");
    }

    #[test]
    fn lock_is_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        let first = IndexLock::acquire(tmp.path()).unwrap();
        assert!(matches!(IndexLock::acquire(tmp.path()), Err(Error::Busy(_))));
        drop(first);
        // Other tests spawn processes concurrently; a child can hold an inherited copy of the fd for
        // an instant between fork and exec, so allow a short retry.
        let reacquired = (0..50).any(|_| {
            IndexLock::acquire(tmp.path()).is_ok() || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                false
            }
        });
        assert!(reacquired);
    }

    #[test]
    fn redo_describe_clears_the_descriptions_it_is_meant_to_redo() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let mut db = crate::db::Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.conn.execute("INSERT INTO videos(content_hash, size) VALUES ('sha3:ccdd', 100)", []).unwrap();
        let vid: i64 =
            db.conn.query_row("SELECT id FROM videos WHERE content_hash = 'sha3:ccdd'", [], |r| r.get(0)).unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let folder_id: i64 = db.conn.query_row("SELECT id FROM folders LIMIT 1", [], |r| r.get(0)).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (?1, ?2, '/a.mp4', 100, 0, 1)",
                rusqlite::params![vid, folder_id],
            )
            .unwrap();
        for stage in STAGES {
            db.conn
                .execute(
                    "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) VALUES (?1, ?2, 'done', 0)",
                    rusqlite::params![vid, stage],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, thumb_path, phash, description_json, visible_text)
                 VALUES (?1, 0.0, 'frames/cc/ccdd/t0.jpg', 0, '{\"description\":\"a street\"}', 'STOP')",
                rusqlite::params![vid],
            )
            .unwrap();

        reset_stages(&db, &data_dir, Some(p.id), "describe").unwrap();

        // The frame row survives — re-extracting it is `--redo frames`, which is slower and was
        // not asked for. What must go is the description, because the describe stage only picks
        // up frames where it is NULL: leaving it made `--redo describe` report success in
        // eighteen seconds having described nothing.
        let (frames, described): (i64, i64) = db
            .conn
            .query_row(
                "SELECT COUNT(*), SUM(CASE WHEN description_json IS NOT NULL THEN 1 ELSE 0 END)
                 FROM frames WHERE video_id = ?1",
                [vid],
                |r| Ok((r.get(0)?, r.get::<_, Option<i64>>(1)?.unwrap_or(0))),
            )
            .unwrap();
        assert_eq!(frames, 1, "the keyframe itself is kept");
        assert_eq!(described, 0, "but its description is cleared, so there is work to do");

        let state: String = db
            .conn
            .query_row("SELECT state FROM jobs WHERE video_id = ?1 AND stage = 'describe'", [vid], |r| r.get(0))
            .unwrap();
        assert_eq!(state, "pending");
    }

    #[test]
    fn reset_stages_clears_frames_and_resets_jobs() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let mut db = crate::db::Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();

        // Insert a fake video and mark all stages done.
        db.conn.execute("INSERT INTO videos(content_hash, size) VALUES ('sha3:aabb', 100)", []).unwrap();
        let vid: i64 =
            db.conn.query_row("SELECT id FROM videos WHERE content_hash = 'sha3:aabb'", [], |r| r.get(0)).unwrap();

        // Add a folder so the video is in scope.
        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let folder_id: i64 = db.conn.query_row("SELECT id FROM folders LIMIT 1", [], |r| r.get(0)).unwrap();
        db.conn.execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, '/a.mp4', 100, 0, 1)",
            rusqlite::params![vid, folder_id],
        ).unwrap();

        // Insert jobs for all stages (all marked done).
        for stage in STAGES {
            db.conn
                .execute(
                    "INSERT OR IGNORE INTO jobs(video_id, stage, state, updated_at) VALUES (?1, ?2, 'done', 0)",
                    rusqlite::params![vid, stage],
                )
                .unwrap();
        }

        // Insert a fake frame row and a JPEG file.
        let frames_dir = data_dir.join("frames/aa/aabb");
        std::fs::create_dir_all(&frames_dir).unwrap();
        let jpeg = frames_dir.join("t000000000.jpg");
        std::fs::write(&jpeg, b"fake").unwrap();
        let rel = "frames/aa/aabb/t000000000.jpg".to_string();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, thumb_path, phash) VALUES (?1, 0.0, ?2, 0)",
                rusqlite::params![vid, rel],
            )
            .unwrap();

        // Reset from "frames" stage.
        reset_stages(&db, &data_dir, Some(p.id), "frames").unwrap();

        // Frame file deleted.
        assert!(!jpeg.exists(), "frame JPEG should be deleted");
        // Frame rows gone.
        let count: i64 =
            db.conn.query_row("SELECT COUNT(*) FROM frames WHERE video_id = ?1", [vid], |r| r.get(0)).unwrap();
        assert_eq!(count, 0, "frame rows deleted");

        // Jobs for frames, describe, embed → pending.
        for stage in &["frames", "describe", "embed"] {
            let state: String = db
                .conn
                .query_row(
                    "SELECT state FROM jobs WHERE video_id = ?1 AND stage = ?2",
                    rusqlite::params![vid, stage],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(state, "pending", "stage {stage} should be pending after reset");
        }
        // Jobs before frames (probe, transcribe) stay done.
        for stage in &["probe", "transcribe"] {
            let state: String = db
                .conn
                .query_row(
                    "SELECT state FROM jobs WHERE video_id = ?1 AND stage = ?2",
                    rusqlite::params![vid, stage],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(state, "done", "stage {stage} should still be done");
        }
    }

    #[tokio::test]
    async fn pipeline_disabled_stage_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let custom_pipeline = crate::projects::PipelineConfig {
            probe: true,
            transcribe: false,
            frames: true,
            describe: false,
            embed: true,
            steadiness: true,
            faces: true,
            speakers: true,
        };
        let p = db.create_project(&NewProject::named("NoSpeech").with_pipeline(custom_pipeline)).unwrap();

        let media = tmp.path().join("media");
        std::fs::create_dir_all(&media).unwrap();
        db.add_folder(p.id, &media, true).unwrap();
        let folder_id: i64 = db.conn.query_row("SELECT id FROM folders LIMIT 1", [], |r| r.get(0)).unwrap();

        // Insert a video with audio
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, has_audio) VALUES (1, 'hash_no_speech', 500, 10.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, '/a.mp4', 500, 0, 1)",
                [folder_id],
            )
            .unwrap();

        // Ensure jobs and sync
        ensure_jobs(&db).unwrap();
        sync_pipeline_jobs(&db, Some(p.id)).unwrap();

        let tr_state: String = db
            .conn
            .query_row("SELECT state FROM jobs WHERE video_id = 1 AND stage = 'transcribe'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tr_state, "skipped", "transcribe job should be skipped because project disabled it");

        let desc_state: String = db
            .conn
            .query_row("SELECT state FROM jobs WHERE video_id = 1 AND stage = 'describe'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(desc_state, "skipped", "describe job should be skipped because project disabled it");

        let frames_state: String = db
            .conn
            .query_row("SELECT state FROM jobs WHERE video_id = 1 AND stage = 'frames'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(frames_state, "pending", "frames job should still be pending");

        // Re-enable transcribe
        let mut new_pipeline = p.pipeline.clone();
        new_pipeline.transcribe = true;
        db.update_project_pipeline(p.id, &new_pipeline).unwrap();

        let tr_state_after: String = db
            .conn
            .query_row("SELECT state FROM jobs WHERE video_id = 1 AND stage = 'transcribe'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tr_state_after, "pending", "transcribe job should be pending after re-enabling");
    }
}
