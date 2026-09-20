//! The slice of the Greet Mag project the recorded drafts refer to, built in memory.
//!
//! Shared by `eval_replay` (the repair passes) and `eval_judge` (the editorial read), so the two
//! measure the same footage. Lives under `tests/common/` because cargo turns every `.rs` directly
//! in `tests/` into its own test binary.

use ghostreel_core::db::Db;
use ghostreel_core::projects::NewProject;

/// The slice of a project a recorded draft refers to: what was said, what is visible, and how
/// steady the camera was. Enough to repair a draft, and nothing else.
#[derive(serde::Deserialize)]
struct ProjectFixture {
    videos: Vec<VideoRow>,
    segments: Vec<SegmentRow>,
    frames: Vec<FrameRow>,
    motion: Vec<MotionRow>,
}

#[derive(serde::Deserialize)]
struct VideoRow {
    id: i64,
    duration_s: Option<f64>,
    fps: Option<f64>,
    has_audio: Option<i64>,
    audio_track: Option<i64>,
}

#[derive(serde::Deserialize)]
struct SegmentRow {
    video_id: i64,
    start_s: f64,
    end_s: f64,
    text: String,
    off_mic: i64,
}

#[derive(serde::Deserialize)]
struct FrameRow {
    video_id: i64,
    t_s: f64,
    description_json: Option<String>,
}

#[derive(serde::Deserialize)]
struct MotionRow {
    video_id: i64,
    start_s: f64,
    end_s: f64,
    jerk: f64,
    motion: f64,
    sway: f64,
}

/// Build the project in memory. The files themselves are never opened — the repair passes read
/// the index, not the footage — so a placeholder path per video is enough.
pub fn fixture_project(tmp: &std::path::Path) -> (Db, i64) {
    let raw = include_str!("../fixtures/eval/project.json");
    let fx: ProjectFixture = serde_json::from_str(raw).expect("project fixture");

    let mut db = Db::open_in_memory().unwrap();
    let project = db.create_project(&NewProject::named("Eval")).unwrap();
    let folder = db.add_folder(project.id, tmp, true).unwrap();

    for v in &fx.videos {
        let path = tmp.join(format!("{}.mp4", v.id));
        std::fs::write(&path, b"x").unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio, audio_track)
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)",
                rusqlite::params![v.id, format!("h{}", v.id), v.duration_s, v.fps, v.has_audio, v.audio_track],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (?1, ?2, ?3, 1, 0, 0)",
                rusqlite::params![v.id, folder.id, path.to_str().unwrap()],
            )
            .unwrap();
    }
    for s in &fx.segments {
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![s.video_id, s.start_s, s.end_s, s.text, s.off_mic],
            )
            .unwrap();
    }
    for f in &fx.frames {
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (?1, ?2, ?3)",
                rusqlite::params![f.video_id, f.t_s, f.description_json],
            )
            .unwrap();
    }
    for m in &fx.motion {
        db.conn
            .execute(
                "INSERT INTO motion_windows(video_id, start_s, end_s, jerk, motion, sway)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![m.video_id, m.start_s, m.end_s, m.jerk, m.motion, m.sway],
            )
            .unwrap();
    }
    (db, project.id)
}

