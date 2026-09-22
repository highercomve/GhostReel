//! Projects and their watched folders (plan D13).
//!
//! A folder row is shared: several projects can watch the same directory, and its videos are
//! indexed once. Removing a folder from its last project deletes the folder (and its file
//! locations); the videos' index data stays, so re-adding the footage later is instant.

use std::path::{Path, PathBuf};

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;

fn default_true() -> bool {
    true
}

/// Which pipeline stages are enabled for a project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default = "default_true")]
    pub probe: bool,
    #[serde(default = "default_true")]
    pub transcribe: bool,
    #[serde(default = "default_true")]
    pub frames: bool,
    #[serde(default = "default_true")]
    pub describe: bool,
    #[serde(default = "default_true")]
    pub embed: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { probe: true, transcribe: true, frames: true, describe: true, embed: true }
    }
}

impl PipelineConfig {
    pub fn is_enabled(&self, stage: &str) -> bool {
        match stage {
            "probe" => self.probe,
            "transcribe" => self.transcribe,
            "frames" => self.frames,
            "describe" => self.describe,
            "embed" => self.embed,
            _ => true,
        }
    }

    pub fn set_enabled(&mut self, stage: &str, enabled: bool) -> Result<(), Error> {
        match stage {
            "probe" => self.probe = enabled,
            "transcribe" => self.transcribe = enabled,
            "frames" => self.frames = enabled,
            "describe" => self.describe = enabled,
            "embed" => self.embed = enabled,
            _ => return Err(Error::Invalid(format!("unknown pipeline stage '{stage}'"))),
        }
        Ok(())
    }

    pub fn enabled_stages(&self) -> Vec<&'static str> {
        let mut stages = Vec::new();
        if self.probe {
            stages.push("probe");
        }
        if self.transcribe {
            stages.push("transcribe");
        }
        if self.frames {
            stages.push("frames");
        }
        if self.describe {
            stages.push("describe");
        }
        if self.embed {
            stages.push("embed");
        }
        stages
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i64,
    pub height: i64,
    pub created_at: i64,
    pub pipeline: PipelineConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Folder {
    pub id: i64,
    pub path: PathBuf,
    pub recursive: bool,
    pub enabled: bool,
}

/// Sequence settings for a new project (used by the M8 timeline export).
#[derive(Debug, Clone)]
pub struct NewProject {
    pub name: String,
    pub description: String,
    pub fps_num: i64,
    pub fps_den: i64,
    pub width: i64,
    pub height: i64,
    pub pipeline: Option<PipelineConfig>,
}

impl NewProject {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            fps_num: 25,
            fps_den: 1,
            width: 1920,
            height: 1080,
            pipeline: None,
        }
    }

    pub fn with_pipeline(mut self, pipeline: PipelineConfig) -> Self {
        self.pipeline = Some(pipeline);
        self
    }
}

/// What [`Db::purge_project_data`] removed. `videos` counts footage that only this project saw:
/// its transcripts, keyframes, descriptions and search vectors go with it. Footage shared with
/// another project is kept.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PurgeStats {
    pub videos: i64,
    pub files_deleted: usize,
    pub bytes_freed: u64,
}

/// Total size of a directory tree (best effort).
fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    entries
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn row_to_project(r: &rusqlite::Row) -> rusqlite::Result<Project> {
    let pipeline_json: String = r.get(8)?;
    let pipeline = serde_json::from_str(&pipeline_json).unwrap_or_default();
    Ok(Project {
        id: r.get(0)?,
        name: r.get(1)?,
        description: r.get(2)?,
        fps_num: r.get(3)?,
        fps_den: r.get(4)?,
        width: r.get(5)?,
        height: r.get(6)?,
        created_at: r.get(7)?,
        pipeline,
    })
}

const PROJECT_COLS: &str =
    "id, name, description, fps_num, fps_den, width, height, created_at, COALESCE(pipeline_json, '')";

impl Db {
    pub fn create_project(&self, p: &NewProject) -> Result<Project, Error> {
        let name = p.name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("project name is empty".into()));
        }
        if p.fps_num <= 0 || p.fps_den <= 0 || p.width <= 0 || p.height <= 0 {
            return Err(Error::Invalid("fps and resolution must be positive".into()));
        }
        if self.project_by_name(name)?.is_some() {
            return Err(Error::Invalid(format!("project '{name}' already exists")));
        }
        let pipeline_json = serde_json::to_string(&p.pipeline.clone().unwrap_or_default()).unwrap_or_default();
        self.conn.execute(
            "INSERT INTO projects(name, description, fps_num, fps_den, width, height, created_at, pipeline_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![name, p.description, p.fps_num, p.fps_den, p.width, p.height, now(), pipeline_json],
        )?;
        let project = self.project(self.conn.last_insert_rowid())?;
        let _ = crate::index::sync_pipeline_jobs(self, Some(project.id));
        Ok(project)
    }

    /// Update the pipeline stage configuration for a project.
    pub fn update_project_pipeline(&mut self, id: i64, pipeline: &PipelineConfig) -> Result<Project, Error> {
        let pipeline_json = serde_json::to_string(pipeline).map_err(|e| Error::Invalid(e.to_string()))?;
        if self.conn.execute("UPDATE projects SET pipeline_json = ?1 WHERE id = ?2", params![pipeline_json, id])? == 0 {
            return Err(Error::NotFound(format!("project #{id}")));
        }
        crate::index::sync_pipeline_jobs(self, Some(id))?;
        self.project(id)
    }

    pub fn projects(&self) -> Result<Vec<Project>, Error> {
        let mut st = self.conn.prepare(&format!("SELECT {PROJECT_COLS} FROM projects ORDER BY name COLLATE NOCASE"))?;
        let rows = st.query_map([], row_to_project)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn project(&self, id: i64) -> Result<Project, Error> {
        self.conn
            .query_row(&format!("SELECT {PROJECT_COLS} FROM projects WHERE id = ?1"), [id], row_to_project)
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("project #{id}")))
    }

    pub fn project_by_name(&self, name: &str) -> Result<Option<Project>, Error> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {PROJECT_COLS} FROM projects WHERE name = ?1 COLLATE NOCASE"),
                [name.trim()],
                row_to_project,
            )
            .optional()?)
    }

    /// Look a project up by name, failing with a helpful error.
    pub fn require_project(&self, name: &str) -> Result<Project, Error> {
        self.project_by_name(name)?.ok_or_else(|| Error::NotFound(format!("project '{name}'")))
    }

    /// Rename a project. Names stay unique (case-insensitive); changing only the case is allowed.
    pub fn rename_project(&mut self, id: i64, name: &str) -> Result<Project, Error> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("project name is empty".into()));
        }
        if self.project_by_name(name)?.is_some_and(|p| p.id != id) {
            return Err(Error::Invalid(format!("project '{name}' already exists")));
        }
        if self.conn.execute("UPDATE projects SET name = ?1 WHERE id = ?2", params![name, id])? == 0 {
            return Err(Error::NotFound(format!("project #{id}")));
        }
        self.project(id)
    }

    /// Take a video out of a project's library. `role` is `removed` (ignored, not indexed) or
    /// `reference` (a finished edit the script chat learns from; still indexed, never used as
    /// footage). The file isn't touched; other projects that see the same file keep it.
    pub fn exclude_video(&self, project_id: i64, video_id: i64, role: &str) -> Result<(), Error> {
        if role != "removed" && role != "reference" {
            return Err(Error::Invalid(format!("unknown role '{role}' (removed or reference)")));
        }
        self.conn.execute(
            "INSERT INTO project_exclusions(project_id, video_id, role, excluded_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(project_id, video_id) DO UPDATE SET role = excluded.role",
            params![project_id, video_id, role, now()],
        )?;
        Ok(())
    }

    /// Put a removed video back into the project's library.
    pub fn include_video(&self, project_id: i64, video_id: i64) -> Result<(), Error> {
        self.conn.execute(
            "DELETE FROM project_exclusions WHERE project_id = ?1 AND video_id = ?2",
            params![project_id, video_id],
        )?;
        Ok(())
    }

    /// Videos taken out of a project: `(video_id, role, one path)`.
    pub fn excluded_videos(&self, project_id: i64) -> Result<Vec<(i64, String, String)>, Error> {
        let mut st = self.conn.prepare(
            "SELECT x.video_id, x.role,
                    COALESCE((SELECT MIN(vf.path) FROM video_files vf WHERE vf.video_id = x.video_id), '')
               FROM project_exclusions x WHERE x.project_id = ?1 ORDER BY x.excluded_at DESC",
        )?;
        let rows = st.query_map([project_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
        Ok(rows)
    }

    /// Delete everything indexed for a project: keyframe images, preview renders and proxies, plus
    /// the transcripts, descriptions and vectors of footage no other project uses. The video files
    /// themselves are never touched. Call before [`remove_project`](Self::remove_project).
    pub fn purge_project_data(&mut self, data_dir: &Path, project_id: i64) -> Result<PurgeStats, Error> {
        let mut stats = PurgeStats::default();
        let delete = |path: &Path, stats: &mut PurgeStats| {
            if let Ok(meta) = std::fs::metadata(path) {
                let ok = if meta.is_dir() {
                    std::fs::remove_dir_all(path).is_ok()
                } else {
                    std::fs::remove_file(path).is_ok()
                };
                if ok {
                    stats.files_deleted += 1;
                    stats.bytes_freed += if meta.is_dir() { dir_size(path) } else { meta.len() };
                }
            }
        };

        // Preview renders of this project's scripts (previews/script_<id>_v<n>.mp4).
        let script_ids: Vec<i64> = {
            let mut st = self.conn.prepare("SELECT id FROM scripts WHERE project_id = ?1")?;
            st.query_map([project_id], |r| r.get(0))?.collect::<Result<_, _>>()?
        };
        let previews = data_dir.join("previews");
        if let Ok(entries) = std::fs::read_dir(&previews) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if script_ids.iter().any(|id| name.starts_with(&format!("script_{id}_v"))) {
                    let p = e.path();
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    if std::fs::remove_file(&p).is_ok() {
                        stats.files_deleted += 1;
                        stats.bytes_freed += size;
                    }
                }
            }
        }

        // Footage only this project sees.
        let videos: Vec<(i64, String)> = {
            let mut st = self.conn.prepare(
                "SELECT DISTINCT v.id, v.content_hash FROM videos v
                   JOIN video_files vf ON vf.video_id = v.id
                   JOIN project_folders pf ON pf.folder_id = vf.folder_id
                  WHERE pf.project_id = ?1
                    AND NOT EXISTS (SELECT 1 FROM video_files vf2
                                      JOIN project_folders pf2 ON pf2.folder_id = vf2.folder_id
                                     WHERE vf2.video_id = v.id AND pf2.project_id <> ?1)",
            )?;
            st.query_map([project_id], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?
        };
        let proxies = data_dir.join("proxies");
        for (video_id, hash) in &videos {
            delete(&data_dir.join(crate::index::frames_rel_dir(hash)), &mut stats);
            // Preview proxies are named after the content hash.
            let prefix: String =
                hash.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
            if let Ok(entries) = std::fs::read_dir(&proxies) {
                for e in entries.flatten() {
                    if e.file_name().to_string_lossy().starts_with(&prefix) {
                        let p = e.path();
                        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                        if std::fs::remove_file(&p).is_ok() {
                            stats.files_deleted += 1;
                            stats.bytes_freed += size;
                        }
                    }
                }
            }
            // Cascades to video_files, jobs, transcripts, frames, chunks and vectors.
            stats.videos += self.conn.execute("DELETE FROM videos WHERE id = ?1", [video_id])? as i64;
        }
        Ok(stats)
    }

    /// Delete a project; folders no other project uses are removed with it.
    pub fn remove_project(&mut self, id: i64) -> Result<(), Error> {
        let tx = self.conn.transaction()?;
        let n = tx.execute("DELETE FROM projects WHERE id = ?1", [id])?;
        if n == 0 {
            return Err(Error::NotFound(format!("project #{id}")));
        }
        tx.execute("DELETE FROM folders WHERE id NOT IN (SELECT folder_id FROM project_folders)", [])?;
        tx.commit()?;
        Ok(())
    }

    /// Watch `path` for `project`. Returns the (possibly shared) folder.
    pub fn add_folder(&mut self, project_id: i64, path: &Path, recursive: bool) -> Result<Folder, Error> {
        let canonical = normalize_dir(path)?;
        let path_str = canonical.to_string_lossy().to_string();
        let tx = self.conn.transaction()?;
        // Refuse nesting within the same project: files would belong to two folders.
        let mut st = tx.prepare(
            "SELECT f.path FROM folders f JOIN project_folders pf ON pf.folder_id = f.id WHERE pf.project_id = ?1",
        )?;
        let existing: Vec<String> = st.query_map([project_id], |r| r.get(0))?.collect::<Result<_, _>>()?;
        drop(st);
        for other in &existing {
            if other == &path_str {
                continue;
            }
            let (a, b) = (Path::new(other), canonical.as_path());
            if b.starts_with(a) || a.starts_with(b) {
                return Err(Error::Invalid(format!("{} overlaps folder {other} already in this project", b.display())));
            }
        }
        tx.execute(
            "INSERT INTO folders(path, recursive, enabled, added_at) VALUES (?1, ?2, 1, ?3)
             ON CONFLICT(path) DO UPDATE SET recursive = excluded.recursive, enabled = 1",
            params![path_str, recursive, now()],
        )?;
        let folder_id: i64 = tx.query_row("SELECT id FROM folders WHERE path = ?1", [&path_str], |r| r.get(0))?;
        let n = tx.execute(
            "INSERT OR IGNORE INTO project_folders(project_id, folder_id) VALUES (?1, ?2)",
            params![project_id, folder_id],
        )?;
        if n == 0 && !existing.contains(&path_str) {
            return Err(Error::NotFound(format!("project #{project_id}")));
        }
        tx.commit()?;
        Ok(Folder { id: folder_id, path: canonical, recursive, enabled: true })
    }

    pub fn folders(&self, project_id: Option<i64>) -> Result<Vec<Folder>, Error> {
        let map = |r: &rusqlite::Row| -> rusqlite::Result<Folder> {
            Ok(Folder {
                id: r.get(0)?,
                path: PathBuf::from(r.get::<_, String>(1)?),
                recursive: r.get(2)?,
                enabled: r.get(3)?,
            })
        };
        Ok(match project_id {
            Some(pid) => {
                let mut st = self.conn.prepare(
                    "SELECT f.id, f.path, f.recursive, f.enabled FROM folders f
                     JOIN project_folders pf ON pf.folder_id = f.id WHERE pf.project_id = ?1 ORDER BY f.path",
                )?;
                st.query_map([pid], map)?.collect::<Result<_, _>>()?
            }
            None => {
                let mut st = self.conn.prepare("SELECT id, path, recursive, enabled FROM folders ORDER BY path")?;
                st.query_map([], map)?.collect::<Result<_, _>>()?
            }
        })
    }

    /// Stop watching `path` for `project`; drop the folder if no project uses it any more.
    pub fn remove_folder(&mut self, project_id: i64, path: &Path) -> Result<(), Error> {
        let path_str = normalize_dir(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().to_string();
        let tx = self.conn.transaction()?;
        let n = tx.execute(
            "DELETE FROM project_folders WHERE project_id = ?1
               AND folder_id = (SELECT id FROM folders WHERE path = ?2)",
            params![project_id, path_str],
        )?;
        if n == 0 {
            return Err(Error::NotFound(format!("folder {path_str} in this project")));
        }
        tx.execute("DELETE FROM folders WHERE id NOT IN (SELECT folder_id FROM project_folders)", [])?;
        tx.commit()?;
        Ok(())
    }
}

/// Absolute, symlink-resolved directory path (UNC prefix stripped on Windows so paths stay
/// readable and usable as `file://` URLs later).
pub fn normalize_dir(path: &Path) -> Result<PathBuf, Error> {
    let canonical = std::fs::canonicalize(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    if !canonical.is_dir() {
        return Err(Error::Invalid(format!("{} is not a directory", canonical.display())));
    }
    #[cfg(windows)]
    {
        let s = canonical.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            if !rest.starts_with("UNC\\") {
                return Ok(PathBuf::from(rest));
            }
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purge_deletes_only_this_project_s_footage() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let mut db = Db::open_in_memory().unwrap();
        let solo = db.create_project(&NewProject::named("Solo")).unwrap();
        let other = db.create_project(&NewProject::named("Other")).unwrap();
        std::fs::create_dir_all(tmp.path().join("a")).unwrap();
        std::fs::create_dir_all(tmp.path().join("b")).unwrap();
        let f_solo = db.add_folder(solo.id, &tmp.path().join("a"), true).unwrap();
        let f_shared = db.add_folder(solo.id, &tmp.path().join("b"), true).unwrap();
        db.add_folder(other.id, &tmp.path().join("b"), true).unwrap();

        // Two videos: one only in Solo, one in the folder both projects watch.
        for (id, hash, folder) in [(1, "sha3:aa11", f_solo.id), (2, "sha3:bb22", f_shared.id)] {
            db.conn
                .execute("INSERT INTO videos(id, content_hash, size) VALUES (?1, ?2, 1)", params![id, hash])
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                     VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder, format!("{hash}.mp4")],
                )
                .unwrap();
            let dir = data.join(crate::index::frames_rel_dir(hash));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("t000000000.jpg"), b"0123456789").unwrap();
        }
        db.conn
            .execute("INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 0, 1, 'hi')", [])
            .unwrap();
        // A preview render of a Solo script, and a proxy of its video.
        std::fs::create_dir_all(data.join("previews")).unwrap();
        std::fs::create_dir_all(data.join("proxies")).unwrap();
        db.conn
            .execute(
                "INSERT INTO scripts(id, project_id, title, version, script_json, created_at)
                 VALUES (7, ?1, 't', 1, '{}', 0)",
                [solo.id],
            )
            .unwrap();
        std::fs::write(data.join("previews/script_7_v1.mp4"), b"preview").unwrap();
        std::fs::write(data.join("proxies/sha3-aa11_0_1000_25_1_960x540.mp4"), b"proxy").unwrap();

        let stats = db.purge_project_data(&data, solo.id).unwrap();
        assert_eq!(stats.videos, 1, "only the video no other project sees: {stats:?}");
        assert!(stats.bytes_freed > 0);
        assert!(!data.join(crate::index::frames_rel_dir("sha3:aa11")).exists());
        assert!(data.join(crate::index::frames_rel_dir("sha3:bb22")).is_dir(), "shared footage keeps its keyframes");
        assert!(!data.join("previews/script_7_v1.mp4").exists());
        assert!(!data.join("proxies/sha3-aa11_0_1000_25_1_960x540.mp4").exists());
        let segs: i64 = db.conn.query_row("SELECT COUNT(*) FROM transcript_segments", [], |r| r.get(0)).unwrap();
        assert_eq!(segs, 0, "transcripts of the deleted video are gone");
        db.remove_project(solo.id).unwrap();
        assert_eq!(db.projects().unwrap().len(), 1);
    }

    #[test]
    fn project_crud_and_unique_names() {
        let mut db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("Teaser")).unwrap();
        assert_eq!((p.fps_num, p.width), (25, 1920));
        assert!(db.create_project(&NewProject::named("teaser")).is_err(), "names are case-insensitive");
        assert!(db.create_project(&NewProject::named("  ")).is_err());
        assert_eq!(db.require_project("TEASER").unwrap().id, p.id);
        let other = db.create_project(&NewProject::named("Other")).unwrap();
        assert!(db.rename_project(other.id, "teaser").is_err(), "rename can't take another project's name");
        assert!(db.rename_project(other.id, " ").is_err());
        assert_eq!(db.rename_project(p.id, " TEASER cut ").unwrap().name, "TEASER cut");
        assert_eq!(db.rename_project(p.id, "teaser CUT").unwrap().name, "teaser CUT", "case-only change");
        db.remove_project(other.id).unwrap();
        db.remove_project(p.id).unwrap();
        assert!(db.projects().unwrap().is_empty());
    }

    #[test]
    fn folders_are_shared_and_cleaned_up() {
        let tmp = tempfile::tempdir().unwrap();
        let media = tmp.path().join("media");
        std::fs::create_dir_all(media.join("sub")).unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let a = db.create_project(&NewProject::named("A")).unwrap();
        let b = db.create_project(&NewProject::named("B")).unwrap();

        let fa = db.add_folder(a.id, &media, true).unwrap();
        let fb = db.add_folder(b.id, &media, true).unwrap();
        assert_eq!(fa.id, fb.id, "same directory → one shared folder row");
        assert!(db.add_folder(a.id, &media, true).is_ok(), "re-adding is idempotent");
        assert!(db.add_folder(a.id, &media.join("sub"), true).is_err(), "nested folder rejected");
        assert!(db.add_folder(a.id, &tmp.path().join("missing"), true).is_err());

        db.remove_folder(a.id, &media).unwrap();
        assert_eq!(db.folders(None).unwrap().len(), 1, "still used by B");
        db.remove_project(b.id).unwrap();
        assert!(db.folders(None).unwrap().is_empty(), "orphan folder removed with last project");
    }

    #[test]
    fn project_pipeline_configuration() {
        let mut db = Db::open_in_memory().unwrap();
        let default_proj = db.create_project(&NewProject::named("Default")).unwrap();
        assert!(default_proj.pipeline.probe);
        assert!(default_proj.pipeline.transcribe);
        assert!(default_proj.pipeline.frames);
        assert!(default_proj.pipeline.describe);
        assert!(default_proj.pipeline.embed);
        assert_eq!(default_proj.pipeline.enabled_stages(), vec!["probe", "transcribe", "frames", "describe", "embed"]);

        let custom_pipeline =
            PipelineConfig { probe: true, transcribe: false, frames: true, describe: false, embed: true };
        let b_roll = db.create_project(&NewProject::named("BRoll").with_pipeline(custom_pipeline)).unwrap();
        assert!(!b_roll.pipeline.transcribe);
        assert!(!b_roll.pipeline.describe);
        assert!(b_roll.pipeline.frames);
        assert_eq!(b_roll.pipeline.enabled_stages(), vec!["probe", "frames", "embed"]);

        // Update pipeline on existing project
        let mut updated_pipeline = b_roll.pipeline.clone();
        updated_pipeline.transcribe = true;
        let updated = db.update_project_pipeline(b_roll.id, &updated_pipeline).unwrap();
        assert!(updated.pipeline.transcribe);
        assert!(db.project(b_roll.id).unwrap().pipeline.transcribe);
    }
}
