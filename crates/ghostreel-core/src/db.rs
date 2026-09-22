//! The index database: SQLite + FTS5 (keyword) + sqlite-vec (vectors), one file (plan D4, §4).

use std::path::Path;
use std::sync::Once;

use rusqlite::{Connection, OptionalExtension, params};

use crate::Error;
use crate::config::{EMBED_DIM, EMBED_MODEL};

/// Ordered migrations; index + 1 is the schema version it produces. Append only.
const MIGRATIONS: &[&str] = &[
    // v1 — initial schema (plan §4)
    r#"
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

    CREATE TABLE folders (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        recursive INTEGER NOT NULL DEFAULT 1,
        enabled INTEGER NOT NULL DEFAULT 1,
        added_at INTEGER NOT NULL
    );

    CREATE TABLE videos (
        id INTEGER PRIMARY KEY,
        content_hash TEXT NOT NULL UNIQUE,
        path TEXT NOT NULL,
        folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
        size INTEGER NOT NULL,
        mtime INTEGER NOT NULL,
        duration_s REAL,
        width INTEGER,
        height INTEGER,
        fps REAL,
        vcodec TEXT,
        acodec TEXT,
        has_audio INTEGER,
        language TEXT,
        summary TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        error TEXT,
        indexed_at INTEGER
    );
    CREATE INDEX videos_path ON videos(path);

    CREATE TABLE jobs (
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        state TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (video_id, stage)
    );

    CREATE TABLE transcript_segments (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        confidence REAL
    );
    CREATE INDEX transcript_segments_video ON transcript_segments(video_id, start_s);

    CREATE TABLE frames (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        t_s REAL NOT NULL,
        thumb_path TEXT,
        phash INTEGER,
        description_json TEXT,
        visible_text TEXT
    );
    CREATE INDEX frames_video ON frames(video_id, t_s);

    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('moment', 'transcript', 'frame', 'summary')),
        start_s REAL,
        end_s REAL,
        text TEXT NOT NULL,
        frame_id INTEGER REFERENCES frames(id) ON DELETE SET NULL
    );
    CREATE INDEX chunks_video ON chunks(video_id);

    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        text, content='chunks', content_rowid='id', tokenize='unicode61'
    );
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;

    CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[768]);
    "#,
    // v2 — projects (D13) and file locations separate from video identity. A video is its
    // content (hash); `video_files` are the paths where that content was found, each inside a
    // watched folder. Projects see videos through project_folders → folders → video_files.
    r#"
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL DEFAULT '',
        fps_num INTEGER NOT NULL DEFAULT 25,
        fps_den INTEGER NOT NULL DEFAULT 1,
        width INTEGER NOT NULL DEFAULT 1920,
        height INTEGER NOT NULL DEFAULT 1080,
        created_at INTEGER NOT NULL
    );

    CREATE TABLE project_folders (
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
        PRIMARY KEY (project_id, folder_id)
    );

    CREATE TABLE videos_v2 (
        id INTEGER PRIMARY KEY,
        content_hash TEXT NOT NULL UNIQUE,
        size INTEGER NOT NULL,
        duration_s REAL,
        width INTEGER,
        height INTEGER,
        rotation INTEGER,
        fps REAL,
        avg_fps REAL,
        vfr INTEGER,
        vcodec TEXT,
        acodec TEXT,
        has_audio INTEGER,
        created_time TEXT,
        language TEXT,
        summary TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        error TEXT,
        indexed_at INTEGER
    );
    INSERT INTO videos_v2 (id, content_hash, size, duration_s, width, height, fps, vcodec, acodec,
                           has_audio, language, summary, status, error, indexed_at)
        SELECT id, content_hash, size, duration_s, width, height, fps, vcodec, acodec,
               has_audio, language, summary, status, error, indexed_at FROM videos;

    CREATE TABLE video_files (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        size INTEGER NOT NULL,
        mtime INTEGER NOT NULL,
        last_seen INTEGER NOT NULL,
        -- Per folder: overlapping folders of different projects each track the file.
        UNIQUE (folder_id, path)
    );
    INSERT INTO video_files (video_id, folder_id, path, size, mtime, last_seen)
        SELECT id, folder_id, path, size, mtime, 0 FROM videos WHERE folder_id IS NOT NULL;

    DROP INDEX videos_path;
    DROP TABLE videos;
    ALTER TABLE videos_v2 RENAME TO videos;
    CREATE INDEX video_files_video ON video_files(video_id);
    CREATE INDEX video_files_path ON video_files(path);
    CREATE INDEX jobs_state ON jobs(stage, state);
    "#,
    // v3 — script chat & timeline export (plan §4, §4a, M8).
    r#"
    CREATE TABLE chat_sessions (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    );
    CREATE INDEX chat_sessions_project ON chat_sessions(project_id);

    CREATE TABLE chat_messages (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_calls_json TEXT,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX chat_messages_session ON chat_messages(session_id);

    CREATE TABLE scripts (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        session_id INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
        title TEXT NOT NULL,
        version INTEGER NOT NULL,
        script_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        UNIQUE (project_id, title, version)
    );
    CREATE INDEX scripts_project ON scripts(project_id);
    CREATE INDEX scripts_session ON scripts(session_id);

    CREATE TABLE exports (
        id INTEGER PRIMARY KEY,
        script_id INTEGER NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
        format TEXT NOT NULL CHECK (format IN ('otio', 'fcp_xml', 'preview_mp4')),
        path TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX exports_script ON exports(script_id);
    "#,
    // v4 — videos taken out of a project's library (the files stay where they are). `removed`:
    // ignored entirely; `reference`: a finished human edit the script chat learns from, never footage.
    r#"
    CREATE TABLE project_exclusions (
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        role TEXT NOT NULL DEFAULT 'removed' CHECK (role IN ('removed', 'reference')),
        excluded_at INTEGER NOT NULL,
        PRIMARY KEY (project_id, video_id)
    );
    "#,
    // v5 — how steady the camera is, in windows across each video. Measured from the pictures at
    // index time: no frame description says whether the operator was on a tripod, and a shaky shot
    // looks bad in a cut whatever is in it.
    r#"
    CREATE TABLE motion_windows (
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        jerk REAL NOT NULL,
        PRIMARY KEY (video_id, start_s)
    );
    CREATE INDEX idx_motion_windows_video ON motion_windows(video_id);
    "#,
    // v6 — how fast the camera moves on purpose in each window, alongside how much it shakes.
    // Together they say whether a clip is locked off, on a tripod, stabilised or handheld.
    r#"
    ALTER TABLE motion_windows ADD COLUMN motion REAL NOT NULL DEFAULT 0;
    "#,
    // v7 — sway: movement within a second that is undone. The slow back-and-forth of an
    // unstabilised walking shot has no tremor to measure, and this is what catches it.
    r#"
    ALTER TABLE motion_windows ADD COLUMN sway REAL NOT NULL DEFAULT 0;
    "#,
    // v8 — which audio track carries the usable speech (a field recording has several, some of
    // them silent), and which transcript segments are someone away from the microphone: in an
    // interview the subject is on a lav and the interviewer is across the room, 12 dB down.
    r#"
    ALTER TABLE videos ADD COLUMN audio_track INTEGER;
    ALTER TABLE transcript_segments ADD COLUMN off_mic INTEGER;
    "#,
    // v9 — why a segment is off-mic, and how sure we are. There are two ways of being the wrong
    // voice and they are not interchangeable: 'level' is the acoustic test in audio.rs, which
    // only catches an interviewer quieter than the subject, and 'speech' is a judgement about
    // what was said (interviewer.rs), which catches one sitting next to the mic. Without this the
    // column is one bit and a semantic flag cannot be reviewed, re-judged at a different
    // threshold, or undone without re-running the audio pass over every video.
    r#"
    ALTER TABLE transcript_segments ADD COLUMN off_mic_source TEXT;
    ALTER TABLE transcript_segments ADD COLUMN off_mic_p REAL;
    "#,
    // v10 — attached images on chat messages (pasted screenshots or reference frames)
    r#"
    ALTER TABLE chat_messages ADD COLUMN images_json TEXT;
    "#,
    // v11 — project pipeline configuration (which stages run or are skipped)
    r#"
    ALTER TABLE projects ADD COLUMN pipeline_json TEXT;
    "#,
];

static REGISTER_VEC: Once = Once::new();

/// Make sqlite-vec available to every connection opened afterwards.
fn register_sqlite_vec() {
    REGISTER_VEC.call_once(|| {
        // SAFETY: sqlite3_vec_init has the sqlite3 extension-entry signature expected by
        // sqlite3_auto_extension; registering it once, before any connection, is the
        // documented way to load a statically linked extension.
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ())));
        }
    });
}

pub struct Db {
    pub conn: Connection,
}

impl Db {
    /// Open (creating if needed) and migrate the database at `path`.
    pub fn open(path: &Path) -> Result<Self, Error> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
        }
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self, Error> {
        register_sqlite_vec();
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self, Error> {
        // WAL lets the app and the CLI read while one of them indexes.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    pub fn schema_version(&self) -> Result<u32, Error> {
        Ok(self.conn.pragma_query_value(None, "user_version", |r| r.get(0))?)
    }

    /// Record how steady a video is, window by window, replacing any earlier measurement.
    pub fn set_motion_windows(&self, video_id: i64, windows: &[crate::steadiness::Window]) -> Result<(), Error> {
        self.conn.execute("DELETE FROM motion_windows WHERE video_id = ?1", [video_id])?;
        let mut st = self.conn.prepare(
            "INSERT INTO motion_windows(video_id, start_s, end_s, jerk, motion, sway) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for w in windows {
            st.execute(rusqlite::params![video_id, w.start_s, w.end_s, w.jerk, w.motion, w.sway])?;
        }
        Ok(())
    }

    /// The steadiness windows of a video, in order.
    pub fn motion_windows(&self, video_id: i64) -> Result<Vec<crate::steadiness::Window>, Error> {
        let mut st = self.conn.prepare(
            "SELECT start_s, end_s, jerk, motion, sway FROM motion_windows WHERE video_id = ?1 ORDER BY start_s",
        )?;
        let rows = st.query_map([video_id], |r| {
            Ok(crate::steadiness::Window {
                start_s: r.get(0)?,
                end_s: r.get(1)?,
                jerk: r.get(2)?,
                motion: r.get(3)?,
                sway: r.get(4)?,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// Record which audio track of a video carries the speech.
    pub fn set_audio_track(&self, video_id: i64, track: Option<u32>) -> Result<(), Error> {
        self.conn.execute("UPDATE videos SET audio_track = ?2 WHERE id = ?1", rusqlite::params![video_id, track])?;
        Ok(())
    }

    /// The audio track to use for a video, when one was measured.
    pub fn audio_track(&self, video_id: i64) -> Option<u32> {
        self.conn
            .query_row("SELECT audio_track FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<u32>>(0))
            .ok()
            .flatten()
    }

    /// Mark which transcript segments are someone off the microphone, by segment start.
    pub fn set_off_mic(&self, video_id: i64, flags: &[(f64, Option<bool>)]) -> Result<(), Error> {
        // 'level': measured, not read. The semantic pass records itself separately and never
        // overwrites one of these, so re-running either is safe.
        let mut st = self.conn.prepare(
            "UPDATE transcript_segments SET off_mic = ?3, off_mic_source = 'level', off_mic_p = NULL
             WHERE video_id = ?1 AND start_s = ?2",
        )?;
        for (start_s, flag) in flags {
            st.execute(rusqlite::params![video_id, start_s, flag])?;
        }
        Ok(())
    }

    /// Whether a clip *opens* on someone close to the microphone.
    ///
    /// Not "is there any on-mic speech in range": a clip holding six seconds of the interviewer
    /// and clipping a third of a second of the subject's "Okay." at the end satisfies that and
    /// still opens on the wrong voice. What matters is the first thing heard.
    pub fn opens_on_mic(&self, video_id: i64, in_s: f64, out_s: f64) -> bool {
        let first: Option<bool> = self
            .conn
            .query_row(
                "SELECT COALESCE(off_mic, 0) FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 \
                 AND start_s < ?3 ORDER BY start_s LIMIT 1",
                rusqlite::params![video_id, in_s, out_s],
                |r| r.get::<_, bool>(0),
            )
            .ok();
        // No speech at all in range is not a problem for this check; silence opens nothing.
        first.is_none_or(|off_mic| !off_mic)
    }

    fn migrate(&mut self) -> Result<(), Error> {
        let current = self.schema_version()? as usize;
        if current > MIGRATIONS.len() {
            return Err(Error::SchemaTooNew { found: current as u32, supported: MIGRATIONS.len() as u32 });
        }
        if current == MIGRATIONS.len() {
            return Ok(());
        }
        // Table rebuilds (v2) drop tables other tables reference; with enforcement on, that would
        // cascade-delete their rows. Enforcement can only change outside a transaction.
        self.conn.pragma_update(None, "foreign_keys", "OFF")?;
        let result = self.apply_migrations(current);
        self.conn.pragma_update(None, "foreign_keys", "ON")?;
        result
    }

    fn apply_migrations(&mut self, current: usize) -> Result<(), Error> {
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(current) {
            let tx = self.conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.pragma_update(None, "user_version", (i + 1) as u32)?;
            if i == 0 {
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES ('embed_model', ?1), ('embed_dim', ?2)",
                    params![EMBED_MODEL, EMBED_DIM.to_string()],
                )?;
            }
            let violations: i64 = tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r.get(0))?;
            if violations > 0 {
                return Err(Error::Migration(format!("v{} left {violations} foreign key violations", i + 1)));
            }
            tx.commit()?;
        }
        Ok(())
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>, Error> {
        Ok(self.conn.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0)).optional()?)
    }

    /// sqlite-vec version string, proving the extension is loaded.
    pub fn vec_version(&self) -> Result<String, Error> {
        Ok(self.conn.query_row("SELECT vec_version()", [], |r| r.get(0))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_and_records_embedding_model() {
        let db = Db::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        assert_eq!(db.meta("embed_model").unwrap().as_deref(), Some(EMBED_MODEL));
        assert_eq!(db.meta("embed_dim").unwrap().as_deref(), Some("768"));
        assert!(db.vec_version().unwrap().starts_with('v'));
    }

    #[test]
    fn v1_to_v2_keeps_videos_and_their_children() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            tx.execute_batch(MIGRATIONS[0]).unwrap();
            tx.pragma_update(None, "user_version", 1).unwrap();
            tx.execute_batch(
                "INSERT INTO folders(id, path, added_at) VALUES (1, '/media', 0);
                 INSERT INTO videos(id, content_hash, path, folder_id, size, mtime, duration_s)
                     VALUES (7, 'abc', '/media/a.mp4', 1, 10, 5, 12.5);
                 INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (7, 0, 1, 'hi');",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        let (hash, dur): (String, f64) = db
            .conn
            .query_row("SELECT content_hash, duration_s FROM videos WHERE id = 7", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((hash.as_str(), dur), ("abc", 12.5));
        let file: String =
            db.conn.query_row("SELECT path FROM video_files WHERE video_id = 7", [], |r| r.get(0)).unwrap();
        assert_eq!(file, "/media/a.mp4");
        let segs: i64 = db.conn.query_row("SELECT count(*) FROM transcript_segments", [], |r| r.get(0)).unwrap();
        assert_eq!(segs, 1, "rebuilding videos must not cascade-delete children");
        let fk: bool = db.conn.pragma_query_value(None, "foreign_keys", |r| r.get(0)).unwrap();
        assert!(fk);
    }

    #[test]
    fn v2_to_v3_adds_chat_and_scripts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..2] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 2).unwrap();
            tx.execute_batch(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'TestProject', 100);
                 INSERT INTO folders(id, path, added_at) VALUES (1, '/media', 0);
                 INSERT INTO project_folders(project_id, folder_id) VALUES (1, 1);
                 INSERT INTO videos(id, content_hash, size, duration_s) VALUES (42, 'hash42', 1024, 15.5);
                 INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                     VALUES (42, 1, '/media/clip.mp4', 1024, 10, 0);",
            )
            .unwrap();
            tx.commit().unwrap();
        }

        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);

        let (pname, pcreated): (String, i64) = db
            .conn
            .query_row("SELECT name, created_at FROM projects WHERE id = 1", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((pname.as_str(), pcreated), ("TestProject", 100));

        let (hash, dur): (String, f64) = db
            .conn
            .query_row("SELECT content_hash, duration_s FROM videos WHERE id = 42", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((hash.as_str(), dur), ("hash42", 15.5));

        let chat_sessions_count: i64 =
            db.conn.query_row("SELECT count(*) FROM chat_sessions", [], |r| r.get(0)).unwrap();
        assert_eq!(chat_sessions_count, 0);
        let chat_messages_count: i64 =
            db.conn.query_row("SELECT count(*) FROM chat_messages", [], |r| r.get(0)).unwrap();
        assert_eq!(chat_messages_count, 0);
        let scripts_count: i64 = db.conn.query_row("SELECT count(*) FROM scripts", [], |r| r.get(0)).unwrap();
        assert_eq!(scripts_count, 0);
        let exports_count: i64 = db.conn.query_row("SELECT count(*) FROM exports", [], |r| r.get(0)).unwrap();
        assert_eq!(exports_count, 0);

        let pipeline_col_exists: bool = db
            .conn
            .query_row(
                "SELECT count(*) FROM pragma_table_info('projects') WHERE name = 'pipeline_json'",
                [],
                |r| Ok(r.get::<_, i64>(0)? > 0),
            )
            .unwrap();
        assert!(pipeline_col_exists);

        let fk: bool = db.conn.pragma_query_value(None, "foreign_keys", |r| r.get(0)).unwrap();
        assert!(fk);
    }

    #[test]
    fn reopen_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ghostreel.db");
        drop(Db::open(&path).unwrap());
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
    }

    #[test]
    fn fts_and_vector_search_work_together() {
        let db = Db::open_in_memory().unwrap();
        let c = &db.conn;
        c.execute("INSERT INTO videos(content_hash, size) VALUES ('h', 1)", []).unwrap();
        for (text, axis) in [("person unboxing a raspberry pi", 0usize), ("cat sleeping on a sofa", 1)] {
            c.execute(
                "INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES (1, 'moment', 0, 5, ?1)",
                [text],
            )
            .unwrap();
            let id = c.last_insert_rowid();
            let mut v = vec![0f32; EMBED_DIM];
            v[axis] = 1.0;
            let blob: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            c.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)", params![id, blob]).unwrap();
        }

        let fts: i64 =
            c.query_row("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'unboxing'", [], |r| r.get(0)).unwrap();
        assert_eq!(fts, 1);

        let mut q = vec![0f32; EMBED_DIM];
        q[1] = 1.0;
        let blob: Vec<u8> = q.iter().flat_map(|f| f.to_le_bytes()).collect();
        let nearest: i64 = c
            .query_row("SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 ORDER BY distance LIMIT 1", [blob], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(nearest, 2);

        // Deleting a chunk keeps FTS in sync.
        c.execute("DELETE FROM chunks WHERE id = 1", []).unwrap();
        let n: i64 =
            c.query_row("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'unboxing'", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0);
    }
}
