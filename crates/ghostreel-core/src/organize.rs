//! Filesystem folder organizer: groups original media files on disk into categorized subfolders
//! (e.g. `Interviews/`, `B-Roll/`, `Screen Recordings/`, `Drones & Aerials/`, `Other/`)
//! inside each watched folder, updates database paths, and moves companion sidecars (.xmp, .wav, .srt, .thm).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::classify::folder_for_category;
use crate::db::Db;

/// Known sidecar file extensions commonly found on camera cards and production sets.
pub const SIDECAR_EXTENSIONS: &[&str] = &["xmp", "thm", "lrf", "wav", "aac", "srt", "vtt", "xml", "json", "nfo"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarMove {
    pub from: PathBuf,
    pub to: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMoveItem {
    pub video_id: i64,
    pub file_name: String,
    pub current_path: PathBuf,
    pub target_path: PathBuf,
    pub category: String,
    pub category_folder: String,
    pub watched_folder_path: PathBuf,
    pub already_organized: bool,
    pub conflict: Option<String>,
    pub sidecars: Vec<SidecarMove>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizePlan {
    pub project_id: i64,
    pub items: Vec<FileMoveItem>,
    pub total_files: usize,
    pub to_move_count: usize,
    pub already_organized_count: usize,
    pub conflict_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizeReport {
    pub project_id: i64,
    pub moved_videos: usize,
    pub moved_sidecars: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

/// Find companion sidecar files sharing the same file stem in the same folder.
pub fn find_sidecars(video_path: &Path, target_dir: &Path, target_stem: &str) -> Vec<SidecarMove> {
    let mut sidecars = Vec::new();
    let Some(parent) = video_path.parent() else {
        return sidecars;
    };
    let Some(stem) = video_path.file_stem().and_then(|s| s.to_str()) else {
        return sidecars;
    };

    let entries = match std::fs::read_dir(parent) {
        Ok(e) => e,
        Err(_) => return sidecars,
    };

    let video_canonical = video_path.canonicalize().unwrap_or_else(|_| video_path.to_path_buf());

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() || path.canonicalize().is_ok_and(|canon| canon == video_canonical) {
            continue;
        }
        if let Some(entry_stem) = path.file_stem().and_then(|s| s.to_str())
            && entry_stem.eq_ignore_ascii_case(stem)
            && let Some(ext) = path.extension().and_then(|e| e.to_str())
            && SIDECAR_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
        {
            sidecars.push(SidecarMove { from: path.clone(), to: target_dir.join(format!("{target_stem}.{ext}")) });
        }
    }

    sidecars
}

/// Move without replacing an existing destination, including sidecars and rollback targets.
/// Hard links avoid copying on supported filesystems; otherwise publish via exclusive creation.
/// Filesystem operations and SQLite are still separate: this is not crash-atomic.
pub fn move_file(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let from_c = std::ffi::CString::new(from.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NUL in source path"))?;
        let to_c = std::ffi::CString::new(to.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NUL in target path"))?;
        // Atomic no-replace rename keeps same-volume moves fast even on exFAT (no hard links).
        // SAFETY: both C strings remain alive and terminated for the duration of this call.
        let result = unsafe {
            libc::renameat2(libc::AT_FDCWD, from_c.as_ptr(), libc::AT_FDCWD, to_c.as_ptr(), libc::RENAME_NOREPLACE)
        };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            return Err(error);
        }
    }
    if std::fs::hard_link(from, to).is_ok() {
        if let Err(error) = std::fs::remove_file(from) {
            let _ = std::fs::remove_file(to);
            return Err(error);
        }
        return Ok(());
    }
    let mut source = std::fs::File::open(from)?;
    let mut target = std::fs::OpenOptions::new().write(true).create_new(true).open(to)?;
    let copied = (|| {
        std::io::copy(&mut source, &mut target)?;
        target.set_permissions(source.metadata()?.permissions())?;
        target.sync_all()?;
        std::fs::remove_file(from)
    })();
    drop(target);
    if copied.is_err() {
        let _ = std::fs::remove_file(to);
    }
    copied
}

/// Generate a collision-safe destination path: if `file_name` collides with an existing file
/// or an already planned target in `target_dir`, generates `{stem}_{counter}.{ext}`.
pub fn collision_safe_target(
    target_dir: &Path,
    file_name: &str,
    current_path: &Path,
    planned_targets: &HashSet<PathBuf>,
) -> (PathBuf, String) {
    let base_target = target_dir.join(file_name);
    if !planned_targets.contains(&base_target) && (!base_target.exists() || base_target == current_path) {
        return (base_target, file_name.to_string());
    }

    let p = Path::new(file_name);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(file_name);
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");

    let mut counter = 1;
    loop {
        let candidate_name =
            if ext.is_empty() { format!("{stem}_{counter}") } else { format!("{stem}_{counter}.{ext}") };
        let candidate_path = target_dir.join(&candidate_name);
        if !planned_targets.contains(&candidate_path) && (!candidate_path.exists() || candidate_path == current_path) {
            return (candidate_path, candidate_name);
        }
        counter += 1;
    }
}

/// Prepare an organization plan without touching any files on disk.
pub fn plan_organize_project(db: &Db, project_id: i64) -> Result<OrganizePlan, Error> {
    let mut st = db.conn.prepare(
        "SELECT vf.video_id, vf.path, f.path AS folder_path, COALESCE(v.category, 'other')
         FROM video_files vf
         JOIN folders f ON f.id = vf.folder_id
         JOIN project_folders pf ON pf.folder_id = f.id
         JOIN videos v ON v.id = vf.video_id
         WHERE pf.project_id = ?1
           AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
         ORDER BY vf.path",
    )?;

    let rows = st.query_map([project_id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            PathBuf::from(r.get::<_, String>(1)?),
            PathBuf::from(r.get::<_, String>(2)?),
            r.get::<_, String>(3)?,
        ))
    })?;

    let mut items = Vec::new();
    let mut planned_targets = HashSet::new();

    for r in rows {
        let (video_id, current_path, watched_folder_path, category) = r?;
        let file_name = current_path.file_name().and_then(|n| n.to_str()).unwrap_or("video.mp4").to_string();

        let category_folder = folder_for_category(&category);
        if category_folder.is_empty()
            || category_folder == "."
            || category_folder == ".."
            || category_folder.contains(['/', '\\', ':'])
        {
            return Err(Error::Invalid(format!("Invalid category folder: {category_folder}")));
        }
        let target_dir = watched_folder_path.join(&category_folder);
        if target_dir.exists() {
            let root = watched_folder_path.canonicalize().map_err(|e| Error::Io(watched_folder_path.clone(), e))?;
            let target = target_dir.canonicalize().map_err(|e| Error::Io(target_dir.clone(), e))?;
            if !target.starts_with(root) {
                return Err(Error::Invalid(format!(
                    "Category folder escapes watched folder: {}",
                    target_dir.display()
                )));
            }
        }

        // Check if file is already in the category folder
        let already_organized = current_path.parent().map(|p| p == target_dir).unwrap_or(false);

        let (target_path, safe_file_name) = if already_organized {
            (current_path.clone(), file_name.clone())
        } else {
            collision_safe_target(&target_dir, &file_name, &current_path, &planned_targets)
        };
        planned_targets.insert(target_path.clone());

        let target_stem = Path::new(&safe_file_name).file_stem().and_then(|s| s.to_str()).unwrap_or(&file_name);

        let sidecars =
            if !already_organized { find_sidecars(&current_path, &target_dir, target_stem) } else { Vec::new() };

        let sidecar_collision = sidecars.iter().find(|sc| sc.to.exists() || planned_targets.contains(&sc.to));
        let conflict = if let Some(sc) = sidecar_collision {
            Some(format!("Sidecar target already exists: {}", sc.to.display()))
        } else if !current_path.exists() {
            Some(format!("Source file missing: {}", current_path.display()))
        } else {
            None
        };

        planned_targets.extend(sidecars.iter().map(|sc| sc.to.clone()));

        items.push(FileMoveItem {
            video_id,
            file_name: safe_file_name,
            current_path,
            target_path,
            category,
            category_folder,
            watched_folder_path,
            already_organized,
            conflict,
            sidecars,
        });
    }

    let total_files = items.len();
    let to_move_count = items.iter().filter(|i| !i.already_organized && i.conflict.is_none()).count();
    let already_organized_count = items.iter().filter(|i| i.already_organized).count();
    let conflict_count = items.iter().filter(|i| i.conflict.is_some()).count();

    Ok(OrganizePlan { project_id, items, total_files, to_move_count, already_organized_count, conflict_count })
}

/// Execute the organization plan: creates directories, moves video files and sidecars,
/// and updates database paths, rolling back the video move on database failure.
pub fn execute_organize_project(db: &mut Db, project_id: i64) -> Result<OrganizeReport, Error> {
    let plan = plan_organize_project(db, project_id)?;
    let mut moved_videos = 0;
    let mut moved_sidecars = 0;
    let mut skipped = 0;
    let mut errors = Vec::new();

    for item in plan.items {
        if item.already_organized {
            skipped += 1;
            continue;
        }

        if let Some(err) = item.conflict {
            errors.push(err);
            skipped += 1;
            continue;
        }

        if !item.current_path.exists() {
            errors.push(format!("Source file missing: {}", item.current_path.display()));
            skipped += 1;
            continue;
        }

        // Perform main video move
        match move_file(&item.current_path, &item.target_path) {
            Ok(_) => {
                // Update SQLite database atomically
                match db.update_video_file_path(&item.current_path, &item.target_path) {
                    Ok(true) => {
                        // DB update succeeded! Now move companion sidecars
                        for sc in item.sidecars {
                            if sc.from.exists() {
                                if let Err(e) = move_file(&sc.from, &sc.to) {
                                    errors.push(format!("Failed to move sidecar {}: {e}", sc.from.display()));
                                } else {
                                    moved_sidecars += 1;
                                }
                            }
                        }
                        moved_videos += 1;
                    }
                    Ok(false) => {
                        // Database update affected 0 rows (old path was not found in DB)!
                        // RECOVERY: Roll back filesystem move immediately to keep filesystem and DB in sync.
                        let rb = move_file(&item.target_path, &item.current_path);
                        let rb_msg = match rb {
                            Ok(_) => "filesystem move rolled back successfully".to_string(),
                            Err(e) => format!("CRITICAL: rollback failed: {e}"),
                        };
                        errors.push(format!("DB update affected 0 rows for {}; {rb_msg}", item.current_path.display()));
                        skipped += 1;
                    }
                    Err(db_err) => {
                        // Database query returned an error!
                        // RECOVERY: Roll back filesystem move immediately to keep filesystem and DB in sync.
                        let rb = move_file(&item.target_path, &item.current_path);
                        let rb_msg = match rb {
                            Ok(_) => "filesystem move rolled back successfully".to_string(),
                            Err(e) => format!("CRITICAL: rollback failed: {e}"),
                        };
                        errors
                            .push(format!("DB update failed for {}: {db_err}; {rb_msg}", item.current_path.display()));
                        skipped += 1;
                    }
                }
            }
            Err(e) => {
                errors.push(format!("Failed to move {}: {e}", item.current_path.display()));
                skipped += 1;
            }
        }
    }

    // Ensure watched folders for this project have recursive = 1 so subfolders are watched
    db.conn.execute(
        "UPDATE folders SET recursive = 1
         WHERE id IN (SELECT folder_id FROM project_folders WHERE project_id = ?1)",
        params![project_id],
    )?;

    Ok(OrganizeReport { project_id, moved_videos, moved_sidecars, skipped, errors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_never_replace_existing_destination_files() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("clip.xmp");
        let target = temp.path().join("existing.xmp");
        std::fs::write(&source, "source sidecar").unwrap();
        std::fs::write(&target, "valuable existing sidecar").unwrap();
        assert!(move_file(&source, &target).is_err());
        assert_eq!(std::fs::read_to_string(&source).unwrap(), "source sidecar");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "valuable existing sidecar");
        let free = temp.path().join("free.xmp");
        move_file(&source, &free).unwrap();
        assert!(!source.exists());
        assert_eq!(std::fs::read_to_string(free).unwrap(), "source sidecar");
    }

    #[test]
    fn test_sidecar_extensions_contains_standard() {
        assert!(SIDECAR_EXTENSIONS.contains(&"xmp"));
        assert!(SIDECAR_EXTENSIONS.contains(&"thm"));
        assert!(SIDECAR_EXTENSIONS.contains(&"wav"));
        assert!(SIDECAR_EXTENSIONS.contains(&"srt"));
    }

    #[test]
    fn test_plan_and_execute_organize() {
        let tmp = tempfile::tempdir().unwrap();
        let folder_dir = tmp.path().join("Footage");
        std::fs::create_dir_all(&folder_dir).unwrap();

        // Create sample video file and sidecar
        let vid1 = folder_dir.join("interview_01.mp4");
        let sidecar1 = folder_dir.join("interview_01.xmp");
        std::fs::write(&vid1, "fake video data").unwrap();
        std::fs::write(&sidecar1, "fake xmp data").unwrap();

        let vid2 = folder_dir.join("scenery_01.mp4");
        std::fs::write(&vid2, "fake broll data").unwrap();

        let mut db = Db::open_in_memory().unwrap();
        // Insert project, folder, project_folders
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'TestProj', 0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO folders(id, path, recursive, enabled, added_at) VALUES (1, ?1, 1, 1, 0)",
                [folder_dir.to_str().unwrap()],
            )
            .unwrap();
        db.conn.execute("INSERT INTO project_folders(project_id, folder_id) VALUES (1, 1)", []).unwrap();

        // Insert videos
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (1, 'hash1', 100, 'interview')", [])
            .unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (2, 'hash2', 100, 'broll')", [])
            .unwrap();

        // Insert video_files
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, 1, ?1, 100, 0, 0)",
                [vid1.to_str().unwrap()],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (2, 1, ?1, 100, 0, 0)",
                [vid2.to_str().unwrap()],
            )
            .unwrap();

        // Plan organize
        let plan = plan_organize_project(&db, 1).unwrap();
        assert_eq!(plan.total_files, 2);
        assert_eq!(plan.to_move_count, 2);
        assert_eq!(plan.conflict_count, 0);

        // Execute organize
        let report = execute_organize_project(&mut db, 1).unwrap();
        assert_eq!(report.moved_videos, 2);
        assert_eq!(report.moved_sidecars, 1);
        assert_eq!(report.errors.len(), 0);

        // Verify files on disk
        let target_vid1 = folder_dir.join("Interviews").join("interview_01.mp4");
        let target_sidecar1 = folder_dir.join("Interviews").join("interview_01.xmp");
        let target_vid2 = folder_dir.join("B-Roll").join("scenery_01.mp4");

        assert!(target_vid1.exists(), "interview video should exist in Interviews/");
        assert!(target_sidecar1.exists(), "xmp sidecar should exist in Interviews/");
        assert!(target_vid2.exists(), "broll video should exist in B-Roll/");
        assert!(!vid1.exists(), "original interview file should be moved");
        assert!(!vid2.exists(), "original broll file should be moved");

        // Verify DB path updated
        let db_path1: String =
            db.conn.query_row("SELECT path FROM video_files WHERE video_id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(db_path1, target_vid1.to_str().unwrap());

        // Re-running organize should find all files already organized
        let plan2 = plan_organize_project(&db, 1).unwrap();
        assert_eq!(plan2.to_move_count, 0);
        assert_eq!(plan2.already_organized_count, 2);

        // A new clip sharing a stem must not overwrite an already organized sidecar.
        let extra = folder_dir.join("interview_01.mov");
        std::fs::write(&extra, "new original").unwrap();
        std::fs::write(&sidecar1, "new sidecar").unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (3, 'hash3', 12, 'interview')", [])
            .unwrap();
        db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (3, 1, ?1, 12, 0, 0)", [extra.to_str().unwrap()]).unwrap();
        let plan = plan_organize_project(&db, 1).unwrap();
        assert!(plan.items.iter().find(|i| i.video_id == 3).unwrap().conflict.is_some());
        let report = execute_organize_project(&mut db, 1).unwrap();
        assert!(!report.errors.is_empty());
        assert!(extra.exists());
        assert_eq!(std::fs::read_to_string(&sidecar1).unwrap(), "new sidecar");
        assert_eq!(std::fs::read_to_string(&target_sidecar1).unwrap(), "fake xmp data");
        db.conn.execute("UPDATE videos SET category = '../escape' WHERE id = 3", []).unwrap();
        assert!(plan_organize_project(&db, 1).is_err());
    }

    #[test]
    fn test_collision_safe_renaming_and_sidecars() {
        let tmp = tempfile::tempdir().unwrap();
        let folder_dir = tmp.path().join("Footage");
        let cam_a = folder_dir.join("CamA");
        let cam_b = folder_dir.join("CamB");
        std::fs::create_dir_all(&cam_a).unwrap();
        std::fs::create_dir_all(&cam_b).unwrap();

        // Both CamA and CamB have clip.mp4 and clip.xmp
        let clip_a = cam_a.join("clip.mp4");
        let xmp_a = cam_a.join("clip.xmp");
        let clip_b = cam_b.join("clip.mp4");
        let xmp_b = cam_b.join("clip.xmp");
        std::fs::write(&clip_a, "cam A video").unwrap();
        std::fs::write(&xmp_a, "cam A sidecar").unwrap();
        std::fs::write(&clip_b, "cam B video").unwrap();
        std::fs::write(&xmp_b, "cam B sidecar").unwrap();

        let mut db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'CollisionProj', 0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO folders(id, path, recursive, enabled, added_at) VALUES (1, ?1, 1, 1, 0)",
                [folder_dir.to_str().unwrap()],
            )
            .unwrap();
        db.conn.execute("INSERT INTO project_folders(project_id, folder_id) VALUES (1, 1)", []).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (1, 'h1', 100, 'interview')", [])
            .unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (2, 'h2', 100, 'interview')", [])
            .unwrap();
        db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, 1, ?1, 100, 0, 0)", [clip_a.to_str().unwrap()]).unwrap();
        db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (2, 1, ?1, 100, 0, 0)", [clip_b.to_str().unwrap()]).unwrap();

        let plan = plan_organize_project(&db, 1).unwrap();
        assert_eq!(plan.total_files, 2);
        assert_eq!(plan.to_move_count, 2);
        assert_eq!(plan.conflict_count, 0);

        // First gets clip.mp4, second gets clip_1.mp4
        let rep = execute_organize_project(&mut db, 1).unwrap();
        assert_eq!(rep.moved_videos, 2);
        assert_eq!(rep.moved_sidecars, 2);
        assert_eq!(rep.errors.len(), 0);

        let int_dir = folder_dir.join("Interviews");
        assert!(int_dir.join("clip.mp4").exists());
        assert!(int_dir.join("clip.xmp").exists());
        assert!(int_dir.join("clip_1.mp4").exists());
        assert!(int_dir.join("clip_1.xmp").exists());
    }

    #[test]
    fn test_rollback_recovery_when_db_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("Footage");
        std::fs::create_dir_all(&root).unwrap();
        let video = root.join("clip.mp4");
        let sidecar = root.join("clip.xmp");
        std::fs::write(&video, "original video").unwrap();
        std::fs::write(&sidecar, "original sidecar").unwrap();
        let mut db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'Rollback', 0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO folders(id, path, recursive, enabled, added_at) VALUES (1, ?1, 1, 1, 0)",
                [root.to_str().unwrap()],
            )
            .unwrap();
        db.conn.execute("INSERT INTO project_folders(project_id, folder_id) VALUES (1, 1)", []).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, category) VALUES (1, 'hash', 14, 'interview')", [])
            .unwrap();
        db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, 1, ?1, 14, 0, 0)", [video.to_str().unwrap()]).unwrap();
        db.conn.execute_batch("CREATE TRIGGER fail_path_update BEFORE UPDATE OF path ON video_files BEGIN SELECT RAISE(ABORT, 'test database failure'); END;").unwrap();
        let report = execute_organize_project(&mut db, 1).unwrap();
        assert_eq!(report.moved_videos, 0);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(std::fs::read_to_string(&video).unwrap(), "original video");
        assert_eq!(std::fs::read_to_string(&sidecar).unwrap(), "original sidecar");
        assert!(!root.join("Interviews/clip.mp4").exists());
        let path: String =
            db.conn.query_row("SELECT path FROM video_files WHERE video_id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(path, video.to_str().unwrap());
    }
}
