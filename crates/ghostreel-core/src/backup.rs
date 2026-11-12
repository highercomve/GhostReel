//! Daily database snapshots: a cheap safety net under the trash.
//!
//! [`ensure_recent`] (the app calls it at start-up) writes a consistent copy of the library with
//! `VACUUM INTO` to `<data_dir>/backups/ghostreel-YYYYMMDD-HHMM.db` (UTC) when the newest one is
//! older than [`BACKUP_EVERY`]; an app left open calls [`ensure_newer_than`] with
//! [`BACKUP_CHECK_AGE`] on a timer. Rotation keeps the newest snapshot of each of the last
//! [`BACKUPS_KEPT`] days, so frequent snapshots never shorten the history. Storage cleanup never
//! touches this folder; the report counts it on its own line.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::Error;
use crate::db::Db;

/// How many days of snapshots are kept (the newest snapshot of each).
pub const BACKUPS_KEPT: usize = 7;
/// How old the newest snapshot may get before start-up writes another.
pub const BACKUP_EVERY: Duration = Duration::from_secs(24 * 3600);
/// How often a running app looks at whether the newest snapshot is old.
pub const BACKUP_CHECK_EVERY: Duration = Duration::from_secs(3 * 3600);
/// How old the newest snapshot may get before a running app writes another.
pub const BACKUP_CHECK_AGE: Duration = Duration::from_secs(6 * 3600);

const PREFIX: &str = "ghostreel-";
const SUFFIX: &str = ".db";

/// One snapshot on disk.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Backup {
    pub path: PathBuf,
    pub bytes: u64,
    /// When the file was last written (seconds since the epoch, from its modification time).
    pub created_at: i64,
    /// The moment the snapshot stands for (seconds since the epoch): the time in its name, or
    /// the file's modification time for a name that carries none. Snapshots order by this.
    pub taken_at: i64,
}

pub fn dir(data_dir: &Path) -> PathBuf {
    data_dir.join("backups")
}

/// The snapshots, newest first (by [`Backup::taken_at`], then by modification time).
pub fn list(data_dir: &Path) -> Vec<Backup> {
    let mut found: Vec<Backup> = std::fs::read_dir(dir(data_dir))
        .map(|rd| rd.flatten().collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with(PREFIX) && name.ends_with(SUFFIX) && e.path().is_file()
        })
        .map(|e| {
            let meta = e.metadata().ok();
            let created_at = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            Backup {
                bytes: meta.as_ref().map(|m| m.len()).unwrap_or(0),
                created_at,
                taken_at: taken_at(&e.file_name().to_string_lossy()).unwrap_or(created_at),
                path: e.path(),
            }
        })
        .collect();
    found.sort_by(|a, b| (b.taken_at, b.created_at).cmp(&(a.taken_at, a.created_at)).then_with(|| b.path.cmp(&a.path)));
    found
}

/// Bytes all snapshots take.
pub fn total_bytes(data_dir: &Path) -> u64 {
    list(data_dir).iter().map(|b| b.bytes).sum()
}

/// `YYYYMMDD-HHMM` in UTC for `secs` since the epoch.
fn stamp(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}", rem / 3600, (rem % 3600) / 60)
}

/// The inverse of [`stamp`]: seconds since the epoch for a snapshot's file name, `None` when the
/// name carries no `YYYYMMDD-HHMM` stamp.
fn taken_at(file_name: &str) -> Option<i64> {
    let stamp = file_name.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?;
    if stamp.len() != 13 || !stamp.is_ascii() {
        return None;
    }
    let (date, time) = stamp.split_at(8);
    let time = time.strip_prefix('-')?;
    if !date.bytes().chain(time.bytes()).all(|b| b.is_ascii_digit()) {
        return None;
    }
    let num = |s: &str| s.parse::<i64>().ok();
    let (y, m, d) = (num(&date[..4])?, num(&date[4..6])?, num(&date[6..])?);
    let (hh, mm) = (num(&time[..2])?, num(&time[2..])?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 {
        return None;
    }
    // Days since 1970-01-01 from a civil date (Howard Hinnant's algorithm, as in `stamp`).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hh * 3600 + mm * 60)
}

/// Where a snapshot is written before it is renamed into place; unique per process.
fn part_path(dir: &Path, secs: i64) -> PathBuf {
    dir.join(format!(".{PREFIX}{}{SUFFIX}.{}.part", stamp(secs), std::process::id()))
}

/// Write a snapshot of `db` now, then keep the newest [`BACKUPS_KEPT`]. Returns its path.
pub fn snapshot(db: &Db, data_dir: &Path, now: SystemTime) -> Result<PathBuf, Error> {
    let d = dir(data_dir);
    std::fs::create_dir_all(&d).map_err(|e| Error::Io(d.clone(), e))?;
    let secs = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let path = d.join(format!("{PREFIX}{}{SUFFIX}", stamp(secs)));
    if !path.exists() {
        // Written beside its final name and renamed, so a crash never leaves half a backup that
        // looks whole. The process id keeps two processes backing up in the same minute apart.
        let part = part_path(&d, secs);
        let _ = std::fs::remove_file(&part);
        db.conn.execute("VACUUM INTO ?1", [part.to_string_lossy().as_ref()])?;
        std::fs::rename(&part, &path).map_err(|e| Error::Io(path.clone(), e))?;
    }
    rotate(data_dir, BACKUPS_KEPT);
    Ok(path)
}

/// Keep the newest snapshot of each of the newest `keep` days (UTC) and remove the rest (never
/// the newest one). Returns how many were removed.
pub fn rotate(data_dir: &Path, keep: usize) -> usize {
    let keep = keep.max(1);
    let mut days: Vec<i64> = Vec::new();
    let mut removed = 0;
    for b in list(data_dir) {
        let day = b.taken_at.div_euclid(86_400);
        if days.last() != Some(&day) && days.len() < keep {
            days.push(day);
        } else if std::fs::remove_file(&b.path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Write a snapshot when the newest is older than [`BACKUP_EVERY`] (or there is none). Returns
/// the new snapshot's path, or `None` when a recent one exists.
pub fn ensure_recent(db: &Db, data_dir: &Path, now: SystemTime) -> Result<Option<PathBuf>, Error> {
    ensure_newer_than(db, data_dir, now, BACKUP_EVERY)
}

/// Write a snapshot when the newest is older than `max_age` (or there is none). Returns the new
/// snapshot's path, or `None` when a newer one exists.
pub fn ensure_newer_than(
    db: &Db,
    data_dir: &Path,
    now: SystemTime,
    max_age: Duration,
) -> Result<Option<PathBuf>, Error> {
    let now_s = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    if let Some(newest) = list(data_dir).first()
        && now_s - newest.created_at < max_age.as_secs() as i64
    {
        return Ok(None);
    }
    snapshot(db, data_dir, now).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_are_utc_calendar_minutes() {
        assert_eq!(stamp(0), "19700101-0000");
        assert_eq!(stamp(951_782_400 + 3600 * 13 + 60 * 7), "20000229-1307");
        assert_eq!(stamp(1_794_355_200), "20261111-0000");
    }

    #[test]
    fn snapshots_are_whole_databases_and_rotate_to_the_newest_seven() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let db = Db::open(&data.join("ghostreel.db")).unwrap();
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'Kept', 0)", []).unwrap();

        let t0 = UNIX_EPOCH + Duration::from_secs(1_794_355_200);
        let first = ensure_recent(&db, data, t0).unwrap().expect("no backup yet: one is written");
        let copy = Db::open(&first).unwrap();
        let name: String = copy.conn.query_row("SELECT name FROM projects WHERE id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(name, "Kept");
        drop(copy);
        assert!(first.file_name().unwrap().to_string_lossy().starts_with("ghostreel-20261111-0000"));

        // Its file time is now, so a start-up right after writes nothing.
        assert!(ensure_recent(&db, data, SystemTime::now()).unwrap().is_none());

        for day in 1..=9u64 {
            snapshot(&db, data, t0 + Duration::from_secs(day * 86_400)).unwrap();
        }
        let kept = list(data);
        assert_eq!(kept.len(), BACKUPS_KEPT);
        assert!(kept[0].path.to_string_lossy().contains("20261120"), "newest first: {:?}", kept[0].path);
        assert!(!kept.iter().any(|b| b.path == first), "the oldest went");
        assert!(total_bytes(data) > 0);
        assert_eq!(rotate(data, 0), BACKUPS_KEPT - 1, "rotation always keeps the newest");
        assert_eq!(list(data).len(), 1);
    }

    fn db_in(data: &Path) -> Db {
        Db::open(&data.join("ghostreel.db")).unwrap()
    }

    fn touch(path: &Path, secs: i64) {
        let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.set_modified(UNIX_EPOCH + Duration::from_secs(secs as u64)).unwrap();
    }

    #[test]
    fn names_parse_back_to_their_stamp() {
        for secs in [0, 951_782_400 + 3600 * 13 + 60 * 7, 1_794_355_200 + 86_399 - 59] {
            assert_eq!(taken_at(&format!("{PREFIX}{}{SUFFIX}", stamp(secs))), Some(secs - secs % 60));
        }
        assert_eq!(taken_at("ghostreel-manual.db"), None);
        assert_eq!(taken_at("ghostreel-20261301-0000.db"), None, "no 13th month");
        assert_eq!(taken_at("ghostreel-20261111-0000.db.part"), None);
    }

    #[test]
    fn the_part_file_is_named_for_the_process() {
        let part = part_path(Path::new("/b"), 0);
        let name = part.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.contains(&format!(".{}.part", std::process::id())), "{name}");
        // Nothing is left behind after a snapshot.
        let dir = tempfile::tempdir().unwrap();
        snapshot(&db_in(dir.path()), dir.path(), SystemTime::now()).unwrap();
        let left: Vec<_> = std::fs::read_dir(super::dir(dir.path())).unwrap().flatten().collect();
        assert_eq!(left.len(), 1);
    }

    #[test]
    fn order_follows_time_not_the_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let db = db_in(data);
        let t0: i64 = 1_794_355_200;
        for day in 0..3 {
            snapshot(&db, data, UNIX_EPOCH + Duration::from_secs((t0 + day * 86_400) as u64)).unwrap();
        }
        // A hand-named copy sorts after the stamped names as a string, but was written long ago.
        let manual = super::dir(data).join("ghostreel-manual.db");
        std::fs::write(&manual, b"x").unwrap();
        touch(&manual, t0 - 100 * 86_400);

        let listed = list(data);
        assert_eq!(listed.len(), 4);
        assert_eq!(listed.last().unwrap().path, manual, "oldest by its modification time");
        assert!(listed[0].path.to_string_lossy().contains("20261113"));
        assert_eq!(rotate(data, 3), 1);
        assert!(!manual.exists(), "the old file went, not a newer snapshot");
        assert_eq!(list(data).len(), 3);
    }

    #[test]
    fn frequent_snapshots_do_not_shorten_the_history() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let db = db_in(data);
        let t0: i64 = 1_794_355_200;
        // Every six hours for ten days.
        for i in 0..40u64 {
            snapshot(&db, data, UNIX_EPOCH + Duration::from_secs(t0 as u64 + i * 6 * 3600)).unwrap();
        }
        let kept = list(data);
        assert_eq!(kept.len(), BACKUPS_KEPT, "one per day");
        let days: std::collections::HashSet<i64> = kept.iter().map(|b| b.taken_at.div_euclid(86_400)).collect();
        assert_eq!(days.len(), BACKUPS_KEPT);
        assert_eq!(stamp(kept[0].taken_at), stamp(t0 + 39 * 6 * 3600), "the newest of today stays");
    }

    #[test]
    fn a_running_app_backs_up_again_after_six_hours() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let db = db_in(data);
        let now = SystemTime::now();
        let secs = now.duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let first = ensure_newer_than(&db, data, now, BACKUP_CHECK_AGE).unwrap().expect("none yet");
        assert!(ensure_newer_than(&db, data, now, BACKUP_CHECK_AGE).unwrap().is_none());

        touch(&first, secs - 5 * 3600);
        assert!(ensure_newer_than(&db, data, now, BACKUP_CHECK_AGE).unwrap().is_none(), "5 hours old: enough");
        touch(&first, secs - 7 * 3600);
        assert!(ensure_newer_than(&db, data, now, BACKUP_CHECK_AGE).unwrap().is_some(), "7 hours old: new one");
        // A day-old backup is still fine for the start-up rule.
        let newest = list(data)[0].path.clone();
        touch(&newest, secs - 12 * 3600);
        assert!(ensure_recent(&db, data, now).unwrap().is_none());
    }
}
