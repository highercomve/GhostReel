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
    // v12 — per-chat style (b-roll montage, natural sound)
    r#"
    ALTER TABLE chat_sessions ADD COLUMN style_json TEXT;
    "#,
    // v13 — video categorization (interview, b-roll, screencast, drone, audio, other)
    r#"
    ALTER TABLE videos ADD COLUMN category TEXT DEFAULT 'broll';
    ALTER TABLE videos ADD COLUMN category_confidence REAL DEFAULT 0.0;
    ALTER TABLE videos ADD COLUMN category_source TEXT DEFAULT 'auto';
    "#,
    // v14 — secondary category tag (e.g. 'music' for live performance B-Roll)
    r#"
    ALTER TABLE videos ADD COLUMN category_tag TEXT;
    "#,
    // v15 — transcription glossary (technical terms, names, Spanglish words and Whisper mishearings)
    r#"
    ALTER TABLE projects ADD COLUMN glossary_json TEXT;
    "#,
    // v16 — face tracking: face_tracks (track duration, face count, mean embedding, avg box height)
    // and face_boxes (box normalized coordinates, score, landmarks, sharpness, cut_off).
    r#"
    CREATE TABLE face_tracks (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        n_faces INTEGER NOT NULL,
        mean_embedding BLOB NOT NULL,
        avg_box_h REAL NOT NULL
    );
    CREATE INDEX idx_face_tracks_video ON face_tracks(video_id);

    CREATE TABLE face_boxes (
        track_id INTEGER NOT NULL REFERENCES face_tracks(id) ON DELETE CASCADE,
        t_s REAL NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        w REAL NOT NULL,
        h REAL NOT NULL,
        score REAL NOT NULL,
        landmarks_json TEXT NOT NULL,
        sharpness REAL NOT NULL,
        cut_off INTEGER NOT NULL
    );
    CREATE INDEX idx_face_boxes_track_ts ON face_boxes(track_id, t_s);
    "#,
    // v17 — people (project identities grouping face tracks, with name/handle/role for name
    // tags), speaker turns (who talks when, by voice) and the link between them. `speaker` is a
    // 1-based voice cluster label, stable within one run over a project. speaker_windows caches
    // the voice embedding of every ~2 s window of speech so a re-run only embeds new videos;
    // person_embeddings remembers faces merged into a person by hand so re-clustering keeps them
    // together.
    r#"
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        name TEXT, handle TEXT, role TEXT,
        face_embedding BLOB,
        voice_embedding BLOB
    );
    CREATE INDEX idx_people_project ON people(project_id);
    ALTER TABLE face_tracks ADD COLUMN person_id INTEGER REFERENCES people(id) ON DELETE SET NULL;
    CREATE TABLE speaker_turns (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL, end_s REAL NOT NULL,
        speaker INTEGER NOT NULL,
        person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
        confidence REAL
    );
    CREATE INDEX idx_speaker_turns_video ON speaker_turns(video_id, start_s);
    ALTER TABLE transcript_segments ADD COLUMN speaker INTEGER;
    ALTER TABLE transcript_segments ADD COLUMN person_id INTEGER;

    CREATE TABLE speaker_windows (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        segment_id INTEGER REFERENCES transcript_segments(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        embedding BLOB NOT NULL
    );
    CREATE INDEX idx_speaker_windows_video ON speaker_windows(video_id, start_s);
    CREATE TABLE person_embeddings (
        person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        embedding BLOB NOT NULL
    );
    CREATE INDEX idx_person_embeddings_person ON person_embeddings(person_id);
    "#,
    // v18 — clips. A clip set is one "Generate clips" run: its clips are ordinary one-clip scripts tagged with the set and `kind = 'clip'`, so
    // check, preview, export and the editor work on them unchanged. `items_json` keeps what the
    // run said about each clip (score, quote, why, accepted/rejected) for the grid.
    r#"
    ALTER TABLE scripts ADD COLUMN set_id TEXT;
    ALTER TABLE scripts ADD COLUMN kind TEXT;
    CREATE INDEX scripts_set ON scripts(set_id);
    CREATE TABLE clip_sets (
        id TEXT PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        brief TEXT NOT NULL DEFAULT '',
        created_at INTEGER NOT NULL,
        options_json TEXT NOT NULL DEFAULT '{}',
        items_json TEXT NOT NULL DEFAULT '[]',
        log_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX clip_sets_project ON clip_sets(project_id);
    "#,
    // v19 — runs (`runs.rs`): every long job's label, state, steps, links to what it made and
    // summary, whoever ran it (the app's queue or the CLI), so the history survives restarts.
    r#"
    CREATE TABLE runs (
        id TEXT PRIMARY KEY,
        project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        label TEXT NOT NULL,
        state TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'app',
        steps_json TEXT NOT NULL DEFAULT '[]',
        links_json TEXT NOT NULL DEFAULT '[]',
        summary TEXT,
        error TEXT,
        created_at INTEGER NOT NULL,
        finished_at INTEGER
    );
    CREATE INDEX runs_project ON runs(project_id, created_at);
    CREATE INDEX runs_created ON runs(created_at);
    "#,
    // v20 — the trash (`trash.rs`): deleting a cut, a script version or a clip set moves its rows
    // here instead of removing them. One `trash` row per delete; the rows it took carry its id in
    // `trash_id` (NULL = live) and every listing skips them. Restoring clears the id; rows trashed
    // longer than `trash::KEEP_DAYS` are purged for good. No foreign key on `trash_id`: dropping a
    // trash row must never bring its rows back by itself.
    r#"
    CREATE TABLE trash (
        id TEXT PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        scripts INTEGER NOT NULL DEFAULT 0,
        clips INTEGER NOT NULL DEFAULT 0,
        link_json TEXT NOT NULL,
        deleted_at INTEGER NOT NULL
    );
    CREATE INDEX trash_project ON trash(project_id, deleted_at);
    ALTER TABLE scripts ADD COLUMN trash_id TEXT;
    ALTER TABLE clip_sets ADD COLUMN trash_id TEXT;
    CREATE INDEX scripts_trash ON scripts(trash_id);
    CREATE INDEX clip_sets_trash ON clip_sets(trash_id);
    "#,
    // v21 — what GhostReel rendered itself (previews, clip and cut exports), by content hash, so
    // the scanner never indexes its own output as footage when it lands in a watched folder.
    // Hash, not path: a render moved or renamed is still a render. No foreign key on the script:
    // a render stays one after its cut is deleted.
    r#"
    CREATE TABLE rendered_outputs (
        content_hash TEXT PRIMARY KEY,
        script_id INTEGER,
        path TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    "#,
    // v22 — where somebody is talking, measured from the audio itself (`speech.rs`): one row per
    // region of speech on the video's chosen audio track, and the quietest 10 ms frame of the
    // pause that follows it. Whisper's segment times are approximate; cuts are placed on these.
    r#"
    CREATE TABLE speech_regions (
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        quiet_after_s REAL,
        PRIMARY KEY (video_id, start_s)
    );
    "#,
    // v23 — when each word of a transcript sentence was said (`sentences.rs`), from the speech
    // engine's word timestamps, so a cut can be made at a word inside a 29 s sentence
    // (`from: "s412@3"`). A word belongs to its sentence (the transcript segment) and goes with it
    // when the video is transcribed again. Engines that give no word timings leave the table empty
    // for the video; nothing is invented.
    r#"
    CREATE TABLE transcript_words (
        segment_id INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        PRIMARY KEY (segment_id, idx)
    );
    "#,
    // v24 — chapters (`chapters.rs`): each speech video split into topical stretches, with a title,
    // a one-sentence summary and 1-3 standout lines, for `get_outline`. A chapter is bounded by two
    // transcript sentences; it goes with them when the video is transcribed again. `titled` is 0
    // when the title is only the first sentence (no chat model was available) so the stage can
    // refresh it later.
    r#"
    CREATE TABLE chapters (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        first_sentence_id INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
        last_sentence_id INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        standout_json TEXT NOT NULL DEFAULT '[]',
        titled INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX chapters_video ON chapters(video_id, idx);
    "#,
    // v25 — real sentences (`sentence_split.rs`). Whisper's segments are ~2 s display fragments or
    // 30 s run-ons, not sentences; these are rebuilt from them at full stops, speaker changes and
    // pauses, each boundary on a pause of the measured speech. Sentence ids (`s412`) name these
    // rows. They are derived data, built on first use and dropped when the transcript or the
    // measured speech changes. Chapters are bounded by sentences now, so they are split again.
    r#"
    CREATE TABLE transcript_sentences (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL
    );
    CREATE INDEX transcript_sentences_video ON transcript_sentences(video_id, start_s);
    DROP TABLE chapters;
    CREATE TABLE chapters (
        id INTEGER PRIMARY KEY,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        first_sentence_id INTEGER NOT NULL REFERENCES transcript_sentences(id) ON DELETE CASCADE,
        last_sentence_id INTEGER NOT NULL REFERENCES transcript_sentences(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        standout_json TEXT NOT NULL DEFAULT '[]',
        titled INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX chapters_video ON chapters(video_id, idx);
    DELETE FROM jobs WHERE stage = 'chapters';
    "#,
    // v26 — each sentence keeps its own words with their times (`words_json`, one [start, end]
    // per word of its text), so "s412@3" names the third word of what the sentence says. Looking
    // the words up by time instead lost or borrowed words wherever the engine's word timings drift
    // from the measured pauses the sentence edges sit on. Sentences are built again to fill it.
    r#"
    ALTER TABLE transcript_sentences ADD COLUMN words_json TEXT;
    DELETE FROM chapters;
    DELETE FROM transcript_sentences;
    DELETE FROM jobs WHERE stage = 'chapters';
    "#,
    // v27 — which version of the splitting rules built a video's sentences
    // (`sentence_split::SPLIT_VERSION`). A video built by an older one is split again on first use,
    // keeping its chapters (their sentence ids are found again by time), so a change to the rules
    // reaches indexed projects without a model run to title chapters again.
    r#"
    ALTER TABLE videos ADD COLUMN sentence_split INTEGER NOT NULL DEFAULT 0;
    "#,
    // v28 — keep sentence ids (`transcript_sentences.id`) monotonic across rebuilds and
    // invalidations (`AUTOINCREMENT`), so an old script clip or reference from before a rebuild
    // never silently resolves to another video's newly inserted sentence that happened to reuse the
    // freed integer row id.
    r#"
    CREATE TABLE transcript_sentences_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        words_json TEXT
    );
    INSERT INTO transcript_sentences_new(id, video_id, start_s, end_s, text, words_json)
        SELECT id, video_id, start_s, end_s, text, words_json FROM transcript_sentences;
    DROP TABLE transcript_sentences;
    ALTER TABLE transcript_sentences_new RENAME TO transcript_sentences;
    CREATE INDEX transcript_sentences_video ON transcript_sentences(video_id, start_s);
    "#,
    // v29 — `premiere_package` exports (`package.rs`: the zip a co-worker opens in Premiere) are
    // recorded like the timeline exports; the format CHECK gains the value.
    r#"
    CREATE TABLE exports_new (
        id INTEGER PRIMARY KEY,
        script_id INTEGER NOT NULL REFERENCES scripts(id) ON DELETE CASCADE,
        format TEXT NOT NULL CHECK (format IN ('otio', 'fcp_xml', 'preview_mp4', 'premiere_package')),
        path TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    INSERT INTO exports_new(id, script_id, format, path, created_at)
        SELECT id, script_id, format, path, created_at FROM exports;
    DROP TABLE exports;
    ALTER TABLE exports_new RENAME TO exports;
    CREATE INDEX exports_script ON exports(script_id);
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
        db.backfill_voice_off_mic();
        Ok(db)
    }

    /// Once per library: mark the interviewer's lines in videos whose speakers were found before
    /// `interviewer::mark_by_voice` existed (new ones get it from the speakers stage). Cheap SQL,
    /// and a failure only leaves the old flags, so it never stops the library from opening.
    fn backfill_voice_off_mic(&self) {
        const KEY: &str = "voice_off_mic_v1";
        match self.meta(KEY) {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("voice_off_mic backfill meta check failed: {e}");
                return;
            }
        }
        let videos: Vec<i64> = match self
            .conn
            .prepare("SELECT DISTINCT video_id FROM transcript_segments WHERE speaker IS NOT NULL")
            .and_then(|mut st| st.query_map([], |r| r.get(0))?.collect::<Result<Vec<_>, _>>())
        {
            Ok(vs) => vs,
            Err(e) => {
                tracing::warn!("voice_off_mic backfill video query failed: {e}");
                return;
            }
        };
        let mut all_ok = true;
        for v in videos {
            if let Err(e) = crate::interviewer::mark_by_voice(self, v) {
                tracing::warn!(video_id = v, "voice_off_mic backfill mark_by_voice failed: {e}");
                all_ok = false;
            }
        }
        if all_ok && let Err(e) = self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?1, '1')", [KEY]) {
            tracing::warn!("voice_off_mic backfill failed to record completion: {e}");
        }
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

    /// Record face tracks for a video, replacing any earlier tracks.
    pub fn set_face_tracks(&self, video_id: i64, tracks: &[crate::faces::FaceTrack]) -> Result<(), Error> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM face_tracks WHERE video_id = ?1", [video_id])?;
        {
            let mut insert_track = tx.prepare(
                "INSERT INTO face_tracks (video_id, start_s, end_s, n_faces, mean_embedding, avg_box_h)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut insert_box = tx.prepare(
                "INSERT INTO face_boxes (track_id, t_s, x, y, w, h, score, landmarks_json, sharpness, cut_off)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for track in tracks {
                let blob = crate::faces::embedding_to_blob(&track.mean_embedding);
                insert_track.execute(rusqlite::params![
                    video_id,
                    track.start_s,
                    track.end_s,
                    track.n_faces as i64,
                    blob,
                    track.avg_box_h as f64,
                ])?;
                let track_id = tx.last_insert_rowid();
                for b in &track.boxes {
                    insert_box.execute(rusqlite::params![
                        track_id,
                        b.t_s,
                        b.x as f64,
                        b.y as f64,
                        b.w as f64,
                        b.h as f64,
                        b.score as f64,
                        b.landmarks_json,
                        b.sharpness as f64,
                        if b.cut_off { 1 } else { 0 },
                    ])?;
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// The face tracks of a video, ordered by start time.
    pub fn face_tracks(&self, video_id: i64) -> Result<Vec<crate::faces::FaceTrack>, Error> {
        let mut st = self.conn.prepare(
            "SELECT id, video_id, start_s, end_s, n_faces, mean_embedding, avg_box_h
             FROM face_tracks WHERE video_id = ?1 ORDER BY start_s, id",
        )?;
        let rows = st.query_map([video_id], |r| {
            let blob: Vec<u8> = r.get(5)?;
            let mean_embedding = crate::faces::blob_to_embedding(&blob).unwrap_or_default();
            Ok(crate::faces::FaceTrack {
                id: r.get(0)?,
                video_id: r.get(1)?,
                start_s: r.get(2)?,
                end_s: r.get(3)?,
                n_faces: r.get::<_, i64>(4)? as usize,
                mean_embedding,
                avg_box_h: r.get::<_, f64>(6)? as f32,
                boxes: Vec::new(),
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// The face boxes for a video within [from_s, to_s].
    pub fn face_boxes(&self, video_id: i64, from_s: f64, to_s: f64) -> Result<Vec<crate::faces::FaceBoxRecord>, Error> {
        let mut st = self.conn.prepare(
            "SELECT fb.track_id, fb.t_s, fb.x, fb.y, fb.w, fb.h, fb.score, fb.landmarks_json, fb.sharpness, fb.cut_off
             FROM face_boxes fb
             JOIN face_tracks ft ON ft.id = fb.track_id
             WHERE ft.video_id = ?1 AND fb.t_s >= ?2 AND fb.t_s <= ?3
             ORDER BY fb.t_s, fb.track_id",
        )?;
        let rows = st.query_map(rusqlite::params![video_id, from_s, to_s], |r| {
            Ok(crate::faces::FaceBoxRecord {
                track_id: r.get(0)?,
                t_s: r.get(1)?,
                x: r.get::<_, f64>(2)? as f32,
                y: r.get::<_, f64>(3)? as f32,
                w: r.get::<_, f64>(4)? as f32,
                h: r.get::<_, f64>(5)? as f32,
                score: r.get::<_, f64>(6)? as f32,
                landmarks_json: r.get(7)?,
                sharpness: r.get::<_, f64>(8)? as f32,
                cut_off: r.get::<_, i32>(9)? != 0,
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    /// All face boxes for a specific track, ordered by time.
    pub fn face_boxes_for_track(&self, track_id: i64) -> Result<Vec<crate::faces::FaceBoxRecord>, Error> {
        let mut st = self.conn.prepare(
            "SELECT track_id, t_s, x, y, w, h, score, landmarks_json, sharpness, cut_off
             FROM face_boxes WHERE track_id = ?1 ORDER BY t_s",
        )?;
        let rows = st.query_map([track_id], |r| {
            Ok(crate::faces::FaceBoxRecord {
                track_id: r.get(0)?,
                t_s: r.get(1)?,
                x: r.get::<_, f64>(2)? as f32,
                y: r.get::<_, f64>(3)? as f32,
                w: r.get::<_, f64>(4)? as f32,
                h: r.get::<_, f64>(5)? as f32,
                score: r.get::<_, f64>(6)? as f32,
                landmarks_json: r.get(7)?,
                sharpness: r.get::<_, f64>(8)? as f32,
                cut_off: r.get::<_, i32>(9)? != 0,
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
        let off_mic_ids = |conn: &rusqlite::Connection| -> Result<Vec<i64>, Error> {
            let mut st = conn.prepare(
                "SELECT id FROM transcript_segments WHERE video_id = ?1 AND COALESCE(off_mic, 0) = 1 ORDER BY id",
            )?;
            Ok(st.query_map([video_id], |r| r.get(0))?.collect::<Result<_, _>>()?)
        };
        let tx = self.conn.unchecked_transaction()?;
        let before = off_mic_ids(&tx)?;
        {
            let mut st = tx.prepare(
                "UPDATE transcript_segments SET off_mic = ?3, off_mic_source = 'level', off_mic_p = NULL
                 WHERE video_id = ?1 AND start_s = ?2",
            )?;
            for (start_s, flag) in flags {
                st.execute(rusqlite::params![video_id, start_s, flag])?;
            }
        }
        let after = off_mic_ids(&tx)?;
        if before != after {
            crate::sentence_split::invalidate(&tx, video_id)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Project membership shared by tools and editorial callers. Excluded reference/removed videos
    /// do not count as footage; a failed lookup fails closed.
    pub fn is_video_in_project(&self, project_id: i64, video_id: i64) -> bool {
        self.conn.query_row(
            "SELECT 1 FROM video_files vf
             JOIN folders f ON f.id = vf.folder_id
             JOIN project_folders pf ON pf.folder_id = f.id
             WHERE pf.project_id = ?1 AND vf.video_id = ?2
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
             LIMIT 1",
            rusqlite::params![project_id, video_id],
            |r| r.get::<_, i64>(0),
        ).is_ok()
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

    /// Record or override the category and optional secondary tag of a video ('interview', 'broll', 'screencast', 'drone', 'audio', 'other', etc.).
    pub fn set_video_category(
        &self,
        video_id: i64,
        category: &str,
        confidence: f64,
        source: &str,
        tag: Option<&str>,
    ) -> Result<(), Error> {
        self.conn.execute(
            "UPDATE videos SET category = ?2, category_confidence = ?3, category_source = ?4, category_tag = ?5 WHERE id = ?1",
            params![video_id, category, confidence, source, tag],
        )?;
        Ok(())
    }

    /// Retrieve the category, confidence, source, and secondary tag of a video.
    pub fn video_category(&self, video_id: i64) -> Result<(String, f64, String, Option<String>), Error> {
        self.conn
            .query_row(
                "SELECT COALESCE(category, 'broll'), COALESCE(category_confidence, 0.0), COALESCE(category_source, 'auto'), category_tag FROM videos WHERE id = ?1",
                [video_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .map_err(Error::from)
    }

    /// Update the recorded path of a video file when moved on the filesystem.
    pub fn update_video_file_path(&self, old_path: &Path, new_path: &Path) -> Result<bool, Error> {
        let old_str = old_path.to_string_lossy();
        let new_str = new_path.to_string_lossy();
        let updated =
            self.conn.execute("UPDATE video_files SET path = ?2 WHERE path = ?1", params![old_str, new_str])?;
        Ok(updated > 0)
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
    fn clip_migration_keeps_scripts_and_adds_sets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v16.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..16] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 16).unwrap();
            tx.execute_batch(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'P', 100);
                 INSERT INTO scripts(id, project_id, title, version, script_json, created_at)
                     VALUES (5, 1, 'Cut', 1, '{\"title\":\"Cut\",\"beats\":[]}', 1);",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        let (set_id, kind): (Option<String>, Option<String>) = db
            .conn
            .query_row("SELECT set_id, kind FROM scripts WHERE id = 5", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!((set_id, kind), (None, None));
        let sets: i64 = db.conn.query_row("SELECT count(*) FROM clip_sets", [], |r| r.get(0)).unwrap();
        assert_eq!(sets, 0);
        let s = crate::script::load(&db, 5).unwrap();
        assert!(s.script.captions.is_none(), "old scripts keep captions off");
    }

    #[test]
    fn runs_migration_keeps_clip_sets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v18.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..18] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 18).unwrap();
            tx.execute_batch(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'P', 100);
                 INSERT INTO clip_sets(id, project_id, created_at) VALUES ('cs1', 1, 5);",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        let sets: i64 = db.conn.query_row("SELECT count(*) FROM clip_sets", [], |r| r.get(0)).unwrap();
        assert_eq!(sets, 1);
        assert!(db.runs_for_project(1, 10).unwrap().is_empty());
    }

    #[test]
    fn trash_migration_keeps_every_cut_and_clip_set_live() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v19.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..19] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 19).unwrap();
            tx.execute_batch(
                "INSERT INTO projects(id, name, created_at) VALUES (1, 'P', 100);
                 INSERT INTO clip_sets(id, project_id, created_at) VALUES ('cs1', 1, 5);
                 INSERT INTO scripts(id, project_id, title, version, script_json, created_at, set_id, kind)
                     VALUES (5, 1, 'Clip', 1, '{\"title\":\"Clip\",\"beats\":[]}', 1, 'cs1', 'clip');",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        assert_eq!(crate::script::list(&db, 1).unwrap().len(), 1);
        assert_eq!(crate::clips::list_sets(&db, 1).unwrap().len(), 1);
        assert!(crate::trash::list(&db, None).unwrap().is_empty());
    }

    #[test]
    fn speech_regions_migration_keeps_videos_and_marks_them_unmeasured() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v21.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..21] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 21).unwrap();
            tx.execute_batch(
                "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (7, 'h7', 1, 60.0);
                 INSERT INTO jobs(video_id, stage, state, attempts, updated_at) VALUES (7, 'steadiness', 'done', 0, 1);",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        assert!(MIGRATIONS.len() >= 22, "the speech_regions table is v22");
        let videos: i64 = db.conn.query_row("SELECT COUNT(*) FROM videos", [], |r| r.get(0)).unwrap();
        assert_eq!(videos, 1);
        // An already-indexed video has no regions and no speech job: the stage is pending for it.
        assert!(!crate::speech::analysed(&db, 7));
        db.set_speech_regions(7, &[crate::speech::SpeechRegion::new(1.0, 2.0)]).unwrap();
        assert!(crate::speech::analysed(&db, 7));
    }

    #[test]
    fn words_and_chapters_migrations_keep_the_index_and_start_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v22.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..22] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 22).unwrap();
            tx.execute_batch(
                "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (7, 'h7', 1, 600.0);
                 INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES (41, 7, 0, 5, 'hola');",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        assert_eq!(
            MIGRATIONS.len(),
            29,
            "transcript_words is v23, chapters v24, sentences v25, their words v26, the split version v27, monotonic sentence ids v28, premiere_package exports v29"
        );
        // Nothing lost; the sentence the word and the chapter will hang on is still there.
        let text: String =
            db.conn.query_row("SELECT text FROM transcript_segments WHERE id = 41", [], |r| r.get(0)).unwrap();
        assert_eq!(text, "hola");
        // An already-indexed video has no word timings, sentences or chapters: they come later.
        let sentences: i64 = db.conn.query_row("SELECT COUNT(*) FROM transcript_sentences", [], |r| r.get(0)).unwrap();
        assert_eq!(sentences, 0);
        assert_eq!(crate::sentences::word_count(&db, 7), 0);
        assert!(crate::chapters::load(&db, 7).unwrap().is_empty());
        // Both tables work, and go with their sentences.
        db.conn
            .execute(
                "INSERT INTO transcript_words(segment_id, idx, start_s, end_s, text) VALUES (41, 1, 0.1, 0.4, 'hola')",
                [],
            )
            .unwrap();
        crate::sentence_split::ensure(&db, 7).unwrap();
        let sid: i64 =
            db.conn.query_row("SELECT id FROM transcript_sentences WHERE video_id = 7", [], |r| r.get(0)).unwrap();
        crate::chapters::store(
            &db,
            7,
            &[crate::chapters::Chapter {
                video_id: 7,
                idx: 0,
                start_s: 0.0,
                end_s: 5.0,
                first_sentence_id: sid,
                last_sentence_id: sid,
                title: "hola".into(),
                summary: String::new(),
                standout: Vec::new(),
                titled: false,
            }],
        )
        .unwrap();
        assert_eq!(crate::sentences::word_count(&db, 7), 1);
        assert_eq!(crate::chapters::load(&db, 7).unwrap().len(), 1);
        // Sentences built by older splitting rules are built again on first use, and the chapters
        // stay: title kept, sentence ids found again among the new sentences.
        db.conn.execute("UPDATE videos SET sentence_split = 0 WHERE id = 7", []).unwrap();
        crate::sentence_split::ensure(&db, 7).unwrap();
        let kept = crate::chapters::load(&db, 7).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].title, "hola");
        let first_exists: bool = db
            .conn
            .query_row(
                "SELECT EXISTS (SELECT 1 FROM transcript_sentences WHERE id = ?1)",
                [kept[0].first_sentence_id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(first_exists);
        let version: i64 =
            db.conn.query_row("SELECT sentence_split FROM videos WHERE id = 7", [], |r| r.get(0)).unwrap();
        assert_eq!(version, crate::sentence_split::SPLIT_VERSION);
        // Transcribed again: the words go with their segment, the chapters with their sentences.
        crate::sentence_split::invalidate(&db.conn, 7).unwrap();
        db.conn.execute("DELETE FROM transcript_segments WHERE id = 41", []).unwrap();
        assert_eq!(crate::sentences::word_count(&db, 7), 0);
        assert!(crate::chapters::load(&db, 7).unwrap().is_empty());
        // The video going takes the rest with it.
        db.conn.execute("DELETE FROM videos WHERE id = 7", []).unwrap();
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
            .query_row("SELECT count(*) FROM pragma_table_info('projects') WHERE name = 'pipeline_json'", [], |r| {
                Ok(r.get::<_, i64>(0)? > 0)
            })
            .unwrap();
        assert!(pipeline_col_exists);

        let glossary_col_exists: bool = db
            .conn
            .query_row("SELECT count(*) FROM pragma_table_info('projects') WHERE name = 'glossary_json'", [], |r| {
                Ok(r.get::<_, i64>(0)? > 0)
            })
            .unwrap();
        assert!(glossary_col_exists);

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

    #[test]
    fn backfill_records_completion_only_when_every_video_succeeds() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("DELETE FROM meta WHERE key = 'voice_off_mic_v1'", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, speaker, person_id, off_mic, off_mic_source) VALUES
                 (1, 0.0, 2.0, 'Q1', 3, NULL, 1, 'level'),
                 (1, 3.0, 5.0, 'Q2', 3, NULL, 1, 'level'),
                 (1, 6.0, 8.0, 'Q3', 3, NULL, 0, NULL)",
                [],
            )
            .unwrap();
        // Force mark_by_voice to fail on UPDATE.
        db.conn
            .execute_batch(
                "CREATE TRIGGER fail_voice_update BEFORE UPDATE ON transcript_segments BEGIN
                     SELECT RAISE(ABORT, 'simulated failure');
                 END;",
            )
            .unwrap();
        db.backfill_voice_off_mic();
        assert_eq!(db.meta("voice_off_mic_v1").unwrap(), None, "failed backfill must not record completion");

        db.conn.execute("DROP TRIGGER fail_voice_update", []).unwrap();
        db.backfill_voice_off_mic();
        assert_eq!(db.meta("voice_off_mic_v1").unwrap().as_deref(), Some("1"));
    }

    #[test]
    fn sentence_ids_are_not_reused_after_invalidation() {
        let db = Db::open_in_memory().unwrap();
        for v in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60.0)",
                    params![v, format!("h{v}")],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (?1, 0.0, 5.0, 'Hello world.')",
                    [v],
                )
                .unwrap();
        }
        crate::sentence_split::ensure(&db, 1).unwrap();
        let v1_id: i64 =
            db.conn.query_row("SELECT MAX(id) FROM transcript_sentences WHERE video_id = 1", [], |r| r.get(0)).unwrap();
        crate::sentence_split::invalidate(&db.conn, 1).unwrap();
        crate::sentence_split::ensure(&db, 2).unwrap();
        let v2_id: i64 =
            db.conn.query_row("SELECT MIN(id) FROM transcript_sentences WHERE video_id = 2", [], |r| r.get(0)).unwrap();
        assert!(v2_id > v1_id, "video 2 reused deleted sentence id {v1_id} (got {v2_id})");
    }

    #[test]
    fn v27_to_v28_keeps_sentence_ids_and_prevents_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v27.db");
        register_sqlite_vec();
        {
            let mut conn = Connection::open(&path).unwrap();
            let tx = conn.transaction().unwrap();
            for sql in &MIGRATIONS[0..27] {
                tx.execute_batch(sql).unwrap();
            }
            tx.pragma_update(None, "user_version", 27).unwrap();
            tx.execute_batch(
                "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h1', 1, 60.0), (2, 'h2', 1, 60.0);
                 INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                     (1, 0.0, 5.0, 'First.'),
                     (2, 0.0, 5.0, 'Second.');
                 INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text) VALUES (42, 1, 0.0, 5.0, 'First.');
                 INSERT INTO chapters(id, video_id, idx, start_s, end_s, first_sentence_id, last_sentence_id, title)
                     VALUES (1, 1, 0, 0.0, 5.0, 42, 42, 'Ch1');",
            )
            .unwrap();
            tx.commit().unwrap();
        }
        let db = Db::open(&path).unwrap();
        assert_eq!(db.schema_version().unwrap(), MIGRATIONS.len() as u32);
        let kept_ch: i64 =
            db.conn.query_row("SELECT first_sentence_id FROM chapters WHERE id = 1", [], |r| r.get(0)).unwrap();
        assert_eq!(kept_ch, 42);
        crate::sentence_split::invalidate(&db.conn, 1).unwrap();
        crate::sentence_split::ensure(&db, 2).unwrap();
        let v2_id: i64 =
            db.conn.query_row("SELECT MIN(id) FROM transcript_sentences WHERE video_id = 2", [], |r| r.get(0)).unwrap();
        assert!(v2_id > 42, "new sentence id after v28 migration reused or went below 42 (got {v2_id})");
    }
}
