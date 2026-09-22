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
    /// Overall progress and time remaining (throttled to ~4 per second).
    Progress(Progress),
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
    /// The run was stopped early by [`Options::cancel`].
    pub cancelled: bool,
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

    pub fn is_stage_enabled(&self, stage: &str, project_pipeline: Option<&crate::projects::PipelineConfig>) -> bool {
        if let Some(ref stages) = self.stages {
            return stages.iter().any(|s| s == stage);
        }
        if let Some(pipeline) = project_pipeline {
            return pipeline.is_enabled(stage);
        }
        true
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
pub async fn run(db: &mut Db, rt: &Runtime, opts: &Options, mut on_event: impl FnMut(Event)) -> Result<Summary, Error> {
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
    let (new, changed) = hash_and_store(db, to_hash, &mut tracker, &mut on_event).await?;
    summary.new = new;
    summary.changed = changed;
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

/// Hash files concurrently and record videos, file locations and their pending jobs.
async fn hash_and_store(
    db: &mut Db,
    files: Vec<ToHash>,
    tracker: &mut Tracker,
    on_event: &mut impl FnMut(Event),
) -> Result<(usize, usize), Error> {
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
    let (mut new, mut changed) = (0, 0);
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
    Ok((new, changed))
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
            let any_enabled = pids.iter().any(|pid| {
                project_map.get(pid).is_none_or(|p| p.pipeline.is_enabled(stage))
            });
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
    for (video_id, path) in jobs {
        if opts.cancelled() {
            break;
        }
        let duration = durations.get(&video_id).copied().unwrap_or(0.0);
        set_job(db, video_id, STAGE, "running", None)?;
        on_event(Event::JobStarted { video_id, stage: STAGE.into(), path: path.clone() });
        let mut credited = 0.0f64;
        let result = stt::transcribe(&engine, &rt.ffmpeg, &path, duration, |secs| {
            let secs = secs.min(duration);
            if secs > credited {
                tracker.advance(phase, (secs - credited).round() as u64);
                credited = secs.round();
            }
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
            }
        })
        .await;
        match result {
            Ok(t) => {
                store_transcript(db, video_id, &t)?;
                // Which track carries the speech, and who is close to the microphone. Measured
                // here because the transcript gives the stretches worth measuring; a failure is
                // not worth failing the stage for, it only leaves the editor less informed.
                if rt.measure_audio
                    && let Ok(tracks) = crate::audio::track_count(&rt.ffprobe, &path).await
                    && tracks > 0
                {
                    let track = crate::audio::pick_track(&rt.ffmpeg, &path, tracks, duration).await;
                    let _ = db.set_audio_track(video_id, track);
                    if let Some(track) = track {
                        let spans: Vec<(f64, f64)> = t.segments.iter().map(|s| (s.start, s.end)).collect();
                        let levels = crate::audio::segment_levels(&rt.ffmpeg, &path, track, &spans).await;
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
    Ok((done, failed))
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

pub fn transcript(db: &Db, video_id: i64) -> Result<Vec<TranscriptSegment>, Error> {
    let mut st =
        db.conn.prepare("SELECT start_s, end_s, text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
    let rows =
        st.query_map([video_id], |r| Ok(TranscriptSegment { start: r.get(0)?, end: r.get(1)?, text: r.get(2)? }))?;
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
        let result = crate::frames::extract_keyframes(&rt.ffmpeg, &path, duration, &dir, &frame_opts, |fraction| {
            let secs = (fraction * duration).floor();
            if secs > credited {
                tracker.advance(STAGE, (secs - credited) as u64);
                credited = secs;
            }
            if tracker.should_emit() {
                on_event(Event::Progress(tracker.snapshot(Some(path.clone()))));
            }
        })
        .await;
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
                 WHERE v.duration_s > 0
                   AND NOT EXISTS (SELECT 1 FROM motion_windows mw WHERE mw.video_id = v.id)
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let mut st = db.conn.prepare(sql)?;
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
                 WHERE v.duration_s > 0
                   AND NOT EXISTS (SELECT 1 FROM motion_windows mw WHERE mw.video_id = v.id)
                 GROUP BY v.id
                 ORDER BY v.id"
            };
            let mut st = db.conn.prepare(sql)?;
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
        match crate::steadiness::measure(
            &rt.ffmpeg,
            &path,
            duration,
            steadiness_opts.window_s,
            steadiness_opts.stride_s,
        )
        .await
        {
            Ok(windows) => {
                let _ = db.set_motion_windows(video_id, &windows);
                on_event(Event::JobDone { video_id, stage: STAGE.into() });
                done += 1;
            }
            Err(e) => {
                on_event(Event::JobFailed { video_id, stage: STAGE.into(), error: e.to_string() });
                failed += 1;
            }
        }
        tracker.advance(STAGE, duration.ceil() as u64);
        if tracker.should_emit() {
            on_event(Event::Progress(tracker.snapshot(Some(path))));
        }
    }
    Ok(SteadinessSummary { measured: done, failed, total: total_in_scope })
}

// ---- describe -----------------------------------------------------------------------------

fn describe_phase(setup: &VisionSetup) -> Option<&'static str> {
    match setup {
        VisionSetup::Server(_) => Some("describe_server"),
        VisionSetup::Local { .. } => Some("describe_local"),
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
        VisionSetup::Local { helper, models_dir, model, mmproj, found, runtime } => {
            let mut paths = Vec::new();
            for spec in [model, mmproj] {
                if let Some(p) = found.iter().find(|p| p.file_name().is_some_and(|n| n == spec.file_name.as_str())) {
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
                vision: Some((paths[0].clone(), paths[1].clone())),
                embed: None,
                cpu: false,
                runtime: runtime.clone(),
                concurrency: describe_concurrency(rt),
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

    let mut describer = match start_describer(rt, tracker, on_event).await {
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
        let sem = Arc::new(Semaphore::new(concurrency.max(1)));
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
                    let _permit = sem.acquire_owned().await;
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

            while let Some(joined) = tasks.join_next().await {
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
        let sem = Arc::new(Semaphore::new(n));

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
                    let _permit = sem.acquire_owned().await;
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
            while let Some(joined) = tasks.join_next().await {
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

                let results = match local_llm.batch_describe(&batch_input).await {
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
                let mut result = describer.describe(&image, speech).await;
                if matches!(&result, Err(e) if !is_transport_error(e)) {
                    result = describer.describe(&image, speech).await;
                }
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
        let started = crate::runtime::start_embedder(&rt.embed, |done, len| {
            if let Some(len) = len {
                tracker.set_total("download_embed", len);
            }
            tracker.start("download_embed");
            tracker.advance("download_embed", done.saturating_sub(last));
            last = done;
        })
        .await;
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
        for batch in texts.chunks(BATCH) {
            match embedder.as_mut().expect("checked above").embed(batch).await {
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
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoRow {
    pub id: i64,
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
    /// static, tripod, stabilised, handheld, or unknown.
    pub camera: String,
}

/// Videos visible to a project (or all), one row per content with its first path.
const SCOPE: &str = "
    SELECT vf.video_id, MIN(vf.path) AS path, COUNT(DISTINCT vf.path) AS copies
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
    Ok(s)
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
                (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id)
           FROM ({SCOPE}) sc JOIN videos v ON v.id = sc.video_id
          ORDER BY sc.path"
    ))?;
    let rows = st.query_map([project_id], |r| {
        Ok(VideoRow {
            id: r.get(0)?,
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
            camera: String::new(),
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
            "#!/bin/sh\ncat > /dev/null\necho '{\"type\":\"loaded\"}'\necho '{\"type\":\"progress\",\"percent\":100}'\necho '{\"type\":\"segment\",\"start\":0.5,\"end\":2.5,\"text\":\"hello world\",\"no_speech\":0.1}'\necho '{\"type\":\"done\",\"language\":\"en\",\"audio_s\":3}'\n",
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
                mmproj,
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
        };
        let p = db
            .create_project(&NewProject::named("NoSpeech").with_pipeline(custom_pipeline))
            .unwrap();

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
