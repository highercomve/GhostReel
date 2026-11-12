//! Speakers: who talks when (voice), people (who is on screen), and the link between them.
//!
//! 1. Each transcript segment is cut into ~2 s windows and embedded with a WeSpeaker CAM++
//!    model (Kaldi fbank → 512-d). Embeddings are stored per video (`speaker_windows`), so a
//!    re-run only embeds new videos.
//! 2. All windows of a project are clustered (average linkage, cosine) into speakers; adjacent
//!    windows of one speaker become `speaker_turns`, and each transcript segment gets the speaker
//!    it overlaps most.
//! 3. Face tracks are clustered into project `people` (names survive re-runs).
//! 4. Per video, each person's mouth motion is correlated with each speaker's speech to decide
//!    whose face a voice is.
//!
//! Speaker labels are 1-based, ordered by total speech in the project (1 talks most), and are
//! stable only within one run over a project. A video shared by two projects keeps the turns of
//! whichever project ran last.

pub mod cluster;
pub mod embedder;
pub mod fbank;
pub mod link;
pub mod people;
pub mod turns;

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rusqlite::{OptionalExtension, params};
use serde::Serialize;

pub use embedder::OrtSpeakerEmbedder;
pub use people::{Person, PersonSummary, PeopleReport};
pub use turns::{Turn, WindowSpan};

use crate::Error;
use crate::db::Db;
use crate::faces::{blob_to_embedding, embedding_to_blob};

/// The voice model file (sherpa-onnx export of WeSpeaker's VoxCeleb CAM++).
pub const SPEAKER_MODEL_FILE: &str = "wespeaker_en_voxceleb_CAM++.onnx";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerWindow {
    pub id: i64,
    pub video_id: i64,
    pub segment_id: Option<i64>,
    pub start_s: f64,
    pub end_s: f64,
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

/// A speaker turn as tools show it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeakerTurn {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: i64,
    pub person_id: Option<i64>,
    pub name: Option<String>,
    pub confidence: Option<f64>,
}

/// Decode a video's speech track at 16 kHz mono and embed each window (sorted by start).
/// Windows the model cannot embed (too short after decoding) are left out.
pub fn embed_windows(
    ffmpeg: &Path,
    video: &Path,
    audio_track: Option<u32>,
    windows: &[WindowSpan],
    embedder: &mut OrtSpeakerEmbedder,
    cancel: Option<&Arc<AtomicBool>>,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<(WindowSpan, Vec<f32>)>, Error> {
    const SR: f64 = fbank::SAMPLE_RATE as f64;
    let mut out = Vec::with_capacity(windows.len());
    if windows.is_empty() {
        return Ok(out);
    }
    let end_s = windows.iter().map(|w| w.end_s).fold(0.0f64, f64::max) + 1.0;
    let mut cmd = crate::proc::std_command(ffmpeg);
    cmd.args(["-nostdin", "-v", "error", "-i"]).arg(video);
    cmd.args(["-map", &format!("0:a:{}", audio_track.unwrap_or(0)), "-t", &format!("{end_s:.3}")]);
    cmd.args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]);
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    let mut child = cmd.spawn().map_err(|e| Error::Io(video.to_path_buf(), e))?;
    let mut stdout = child.stdout.take().ok_or_else(|| Error::Invalid("failed to capture ffmpeg stdout".into()))?;

    let mut buf: Vec<f32> = Vec::new();
    let mut base = 0usize; // sample index of buf[0]
    let mut leftover: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 256 * 1024];
    let mut eof = false;
    let mut result = Ok(());
    for w in windows {
        if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
            result = Err(Error::Cancelled);
            break;
        }
        let s0 = (w.start_s.max(0.0) * SR) as usize;
        let s1 = (w.end_s * SR) as usize;
        while !eof && base + buf.len() < s1 {
            match stdout.read(&mut chunk) {
                Ok(0) => eof = true,
                Ok(n) => {
                    leftover.extend_from_slice(&chunk[..n]);
                    let whole = leftover.len() / 4 * 4;
                    buf.extend(leftover[..whole].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
                    leftover.drain(..whole);
                    // Keep memory flat: nothing before this window is needed again.
                    if s0 > base {
                        let drop = (s0 - base).min(buf.len());
                        buf.drain(..drop);
                        base += drop;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    result = Err(Error::Io(video.to_path_buf(), e));
                    eof = true;
                }
            }
        }
        if result.is_err() {
            break;
        }
        let a = s0.max(base) - base;
        let b = s1.saturating_sub(base).min(buf.len());
        if b > a
            && let Some(emb) = embedder.embed(&buf[a..b])?
        {
            out.push((*w, emb));
        }
        on_progress(w.end_s);
    }
    let _ = child.kill();
    let _ = child.wait();
    result.map(|_| out)
}

impl Db {
    pub fn set_speaker_windows(&self, video_id: i64, windows: &[(WindowSpan, Vec<f32>)]) -> Result<(), Error> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM speaker_windows WHERE video_id = ?1", [video_id])?;
        {
            let mut st = tx.prepare(
                "INSERT INTO speaker_windows (video_id, segment_id, start_s, end_s, embedding) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (w, e) in windows {
                st.execute(params![video_id, w.segment_id, w.start_s, w.end_s, embedding_to_blob(e)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn speaker_windows(&self, video_id: i64) -> Result<Vec<SpeakerWindow>, Error> {
        let mut st = self.conn.prepare(
            "SELECT id, video_id, segment_id, start_s, end_s, embedding FROM speaker_windows
              WHERE video_id = ?1 ORDER BY start_s",
        )?;
        let rows = st.query_map([video_id], |r| {
            let blob: Vec<u8> = r.get(5)?;
            Ok(SpeakerWindow {
                id: r.get(0)?,
                video_id: r.get(1)?,
                segment_id: r.get(2)?,
                start_s: r.get(3)?,
                end_s: r.get(4)?,
                embedding: blob_to_embedding(&blob).unwrap_or_default(),
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Replace a video's speaker turns and give each transcript segment its majority speaker.
    /// Person links are cleared (the link step sets them again).
    pub fn set_speaker_turns(&self, video_id: i64, turns: &[Turn]) -> Result<(), Error> {
        let segs: Vec<(i64, f64, f64)> = {
            let mut st = self.conn.prepare("SELECT id, start_s, end_s FROM transcript_segments WHERE video_id = ?1")?;
            st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?
        };
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM speaker_turns WHERE video_id = ?1", [video_id])?;
        {
            let mut ins =
                tx.prepare("INSERT INTO speaker_turns (video_id, start_s, end_s, speaker) VALUES (?1, ?2, ?3, ?4)")?;
            for t in turns {
                ins.execute(params![video_id, t.start_s, t.end_s, t.speaker])?;
            }
            let mut upd =
                tx.prepare("UPDATE transcript_segments SET speaker = ?2, person_id = NULL WHERE id = ?1")?;
            for (id, s, e) in segs {
                upd.execute(params![id, turns::majority_speaker(s, e, turns)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Record that `speaker` in a video is `person` (or nobody), with the link's confidence.
    pub fn link_speaker(
        &self,
        video_id: i64,
        speaker: i64,
        person_id: Option<i64>,
        confidence: Option<f64>,
    ) -> Result<(), Error> {
        self.conn.execute(
            "UPDATE speaker_turns SET person_id = ?3, confidence = ?4 WHERE video_id = ?1 AND speaker = ?2",
            params![video_id, speaker, person_id, confidence],
        )?;
        self.conn.execute(
            "UPDATE transcript_segments SET person_id = ?3 WHERE video_id = ?1 AND speaker = ?2",
            params![video_id, speaker, person_id],
        )?;
        Ok(())
    }

    /// Speaker turns of a video overlapping [from_s, to_s], with the linked person's name.
    pub fn speaker_turns(&self, video_id: i64, from_s: f64, to_s: f64) -> Result<Vec<SpeakerTurn>, Error> {
        let mut st = self.conn.prepare(
            "SELECT st.start_s, st.end_s, st.speaker, st.person_id, p.name, st.confidence
               FROM speaker_turns st LEFT JOIN people p ON p.id = st.person_id
              WHERE st.video_id = ?1 AND st.end_s >= ?2 AND st.start_s <= ?3
              ORDER BY st.start_s",
        )?;
        let rows = st.query_map(params![video_id, from_s, to_s], |r| {
            Ok(SpeakerTurn {
                start_s: r.get(0)?,
                end_s: r.get(1)?,
                speaker: r.get(2)?,
                person_id: r.get(3)?,
                name: r.get(4)?,
                confidence: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Person id per face track of a video (tracks not assigned to anyone are absent).
    pub fn face_track_people(&self, video_id: i64) -> Result<HashMap<i64, i64>, Error> {
        let mut st = self
            .conn
            .prepare("SELECT id, person_id FROM face_tracks WHERE video_id = ?1 AND person_id IS NOT NULL")?;
        let rows = st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

/// What a speakers run did, for the CLI, the task card and tools.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SpeakersReport {
    /// Nothing changed since the last run, so nothing was done.
    pub up_to_date: bool,
    /// Videos whose voice windows were embedded in this run.
    pub embedded: usize,
    pub failed: usize,
    /// Videos with speech in the project.
    pub total: usize,
    pub windows: usize,
    pub speakers: Vec<SpeakerInfo>,
    pub people: PeopleReport,
    pub links: Vec<VideoLink>,
    pub timings: Timings,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SpeakerInfo {
    pub speaker: i64,
    pub speech_s: f64,
    pub windows: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoLink {
    pub video_id: i64,
    pub speaker: i64,
    pub person_id: Option<i64>,
    pub name: Option<String>,
    pub correlation: Option<f32>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Timings {
    pub embed_s: f64,
    pub cluster_s: f64,
    pub people_s: f64,
    pub link_s: f64,
}

/// Progress of a speakers run.
pub enum Step<'a> {
    Started { video_id: i64, path: &'a Path, phase: &'static str },
    Advanced { seconds: f64 },
    Done { video_id: i64 },
    Failed { video_id: i64, error: String },
}

struct VideoInfo {
    id: i64,
    path: PathBuf,
    duration: f64,
    width: u32,
    height: u32,
}

fn project_videos(db: &Db, project_id: i64) -> Result<Vec<VideoInfo>, Error> {
    let ids = db.project_video_ids(project_id)?;
    let mut out = Vec::new();
    for id in ids {
        let row: Option<(String, f64, u32, u32)> = db
            .conn
            .query_row(
                "SELECT vf.path, COALESCE(v.duration_s, 0), COALESCE(v.width, 1280), COALESCE(v.height, 720)
                   FROM videos v JOIN video_files vf ON vf.video_id = v.id WHERE v.id = ?1 LIMIT 1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? as u32, r.get::<_, i64>(3)? as u32)),
            )
            .optional()?;
        if let Some((path, duration, width, height)) = row {
            out.push(VideoInfo { id, path: PathBuf::from(path), duration, width, height });
        }
    }
    Ok(out)
}

fn eligible_windows(db: &Db, video_id: i64) -> Result<Vec<WindowSpan>, Error> {
    let mut st = db.conn.prepare("SELECT id, start_s, end_s FROM transcript_segments WHERE video_id = ?1")?;
    let segs: Vec<(i64, f64, f64)> =
        st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
    Ok(turns::windows_from_segments(&segs))
}

/// What the last run saw: when faces or voice windows change, the project is analysed again.
fn fingerprint(db: &Db, videos: &[VideoInfo]) -> Result<String, Error> {
    let mut parts = Vec::new();
    for v in videos {
        let (nf, mf): (i64, i64) = db.conn.query_row(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM face_tracks WHERE video_id = ?1",
            [v.id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let (nw, mw): (i64, i64) = db.conn.query_row(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM speaker_windows WHERE video_id = ?1",
            [v.id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        parts.push(format!("{}:{nf}/{mf}:{nw}/{mw}", v.id));
    }
    Ok(parts.join(","))
}

/// Videos of a project whose speakers are not analysed yet (all of them when `force`).
pub fn pending_videos(db: &Db, project_id: i64) -> Result<usize, Error> {
    let videos = project_videos(db, project_id)?;
    let mut n = 0;
    for v in &videos {
        let has: bool = db
            .conn
            .query_row("SELECT EXISTS (SELECT 1 FROM speaker_windows WHERE video_id = ?1)", [v.id], |r| r.get(0))?;
        if !has && !eligible_windows(db, v.id)?.is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

/// Analyse a project's speakers, people and links (see the module docs).
///
/// `embedder` is only needed when some video's voice windows are missing; without it those
/// videos are reported failed.
pub fn analyse_project(
    db: &Db,
    ffmpeg: &Path,
    project_id: i64,
    embedder: Option<&mut OrtSpeakerEmbedder>,
    force: bool,
    cancel: Option<&Arc<AtomicBool>>,
    on_step: &mut dyn FnMut(Step<'_>),
) -> Result<SpeakersReport, Error> {
    let cancelled = || cancel.is_some_and(|c| c.load(Ordering::SeqCst));
    let videos = project_videos(db, project_id)?;
    let mut report = SpeakersReport::default();
    let fp_key = format!("speakers_fp:{project_id}");

    // 1. Voice windows for videos that have none (or all, when forced).
    let t0 = Instant::now();
    let mut todo: Vec<(&VideoInfo, Vec<WindowSpan>)> = Vec::new();
    for v in &videos {
        let spans = eligible_windows(db, v.id)?;
        if spans.is_empty() {
            continue;
        }
        report.total += 1;
        let has: bool = db
            .conn
            .query_row("SELECT EXISTS (SELECT 1 FROM speaker_windows WHERE video_id = ?1)", [v.id], |r| r.get(0))?;
        if force || !has {
            todo.push((v, spans));
        } else {
            on_step(Step::Advanced { seconds: v.duration });
        }
    }
    if todo.is_empty() && !force && db.meta(&fp_key)? == Some(fingerprint(db, &videos)?) {
        report.up_to_date = true;
        return Ok(report);
    }
    let mut embedder = embedder;
    for (v, spans) in todo {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        on_step(Step::Started { video_id: v.id, path: &v.path, phase: "voices" });
        let Some(emb) = embedder.as_deref_mut() else {
            on_step(Step::Failed { video_id: v.id, error: "speaker model not available".into() });
            report.failed += 1;
            continue;
        };
        let mut last = 0.0;
        let res = embed_windows(ffmpeg, &v.path, db.audio_track(v.id), &spans, emb, cancel, |t| {
            on_step(Step::Advanced { seconds: (t - last).max(0.0) });
            last = t;
        });
        match res {
            Ok(windows) => {
                db.set_speaker_windows(v.id, &windows)?;
                on_step(Step::Advanced { seconds: (v.duration - last).max(0.0) });
                on_step(Step::Done { video_id: v.id });
                report.embedded += 1;
            }
            Err(e) => {
                if cancelled() {
                    return Err(e);
                }
                on_step(Step::Failed { video_id: v.id, error: e.to_string() });
                report.failed += 1;
            }
        }
    }
    report.timings.embed_s = t0.elapsed().as_secs_f64();

    // 2. Cluster all of the project's windows into speakers; turns and segment speakers.
    let t1 = Instant::now();
    let mut all: Vec<SpeakerWindow> = Vec::new();
    for v in &videos {
        all.extend(db.speaker_windows(v.id)?);
    }
    report.windows = all.len();
    let embs: Vec<&[f32]> = all.iter().map(|w| w.embedding.as_slice()).collect();
    let durs: Vec<f64> = all.iter().map(|w| w.end_s - w.start_s).collect();
    let (labels, _cents) = cluster::label_speakers(&embs, &durs, &cluster::SpeakerClusterParams::default());
    let labels: Vec<i64> = labels.into_iter().map(|l| l as i64 + 1).collect();
    let mut info: BTreeMap<i64, SpeakerInfo> = BTreeMap::new();
    for (w, &l) in all.iter().zip(&labels) {
        let e = info.entry(l).or_insert(SpeakerInfo { speaker: l, ..Default::default() });
        e.speech_s += w.end_s - w.start_s;
        e.windows += 1;
    }
    report.speakers = info.into_values().collect();
    // Per video: (start, end, speaker) of its windows.
    let mut per_video: HashMap<i64, Vec<(f64, f64, i64)>> = HashMap::new();
    for (w, &l) in all.iter().zip(&labels) {
        per_video.entry(w.video_id).or_default().push((w.start_s, w.end_s, l));
    }
    for v in &videos {
        let wins = per_video.get(&v.id).cloned().unwrap_or_default();
        let turns = turns::merge_turns(&wins, turns::MAX_TURN_GAP_S);
        db.set_speaker_turns(v.id, &turns)?;
    }
    report.timings.cluster_s = t1.elapsed().as_secs_f64();

    // 3. People from face tracks.
    let t2 = Instant::now();
    report.people = db.update_people(project_id)?;
    report.timings.people_s = t2.elapsed().as_secs_f64();

    // 4. Link speakers to people, video by video.
    let t3 = Instant::now();
    let mut voice: HashMap<i64, Vec<usize>> = HashMap::new(); // person -> window indices
    for v in &videos {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let Some(wins) = per_video.get(&v.id) else { continue };
        let track_people = db.face_track_people(v.id)?;
        let mut speakers: Vec<i64> = wins.iter().map(|w| w.2).collect();
        speakers.sort();
        speakers.dedup();
        let decisions = if track_people.is_empty() {
            Vec::new()
        } else {
            on_step(Step::Started { video_id: v.id, path: &v.path, phase: "link" });
            let boxes: Vec<link::MouthBox> = db
                .face_boxes(v.id, 0.0, f64::MAX)?
                .into_iter()
                .filter_map(|b| {
                    let person_id = *track_people.get(&b.track_id)?;
                    let lm: Vec<(f32, f32)> = serde_json::from_str(&b.landmarks_json).ok()?;
                    let landmarks: [(f32, f32); 5] = lm.try_into().ok()?;
                    Some(link::MouthBox { t_s: b.t_s, track_id: b.track_id, person_id, landmarks })
                })
                .collect();
            let motion = match link::mouth_motion(ffmpeg, &v.path, v.width, v.height, &boxes, cancel) {
                Ok(m) => m,
                Err(e) => {
                    if cancelled() {
                        return Err(e);
                    }
                    on_step(Step::Failed { video_id: v.id, error: e.to_string() });
                    HashMap::new()
                }
            };
            let mut persons: Vec<i64> = motion.keys().copied().collect();
            persons.sort();
            if persons.is_empty() {
                // No visible mouths: a single person still maps trivially.
                let mut ps: Vec<i64> = track_people.values().copied().collect();
                ps.sort();
                ps.dedup();
                persons = ps;
            }
            let n_frames = (v.duration * link::FPS).ceil() as usize + 2;
            let corr: Vec<Vec<Option<f32>>> = speakers
                .iter()
                .map(|&s| {
                    let mut act = vec![0.0f32; n_frames];
                    for &(a, b, l) in wins {
                        if l == s {
                            let (fa, fb) = ((a * link::FPS) as usize, ((b * link::FPS).ceil() as usize).min(n_frames));
                            for x in act.iter_mut().take(fb).skip(fa) {
                                *x = 1.0;
                            }
                        }
                    }
                    persons
                        .iter()
                        .map(|p| motion.get(p).and_then(|m| link::motion_activity_correlation(m, &act)))
                        .collect()
                })
                .collect();
            link::assign_links(&speakers, &persons, &corr)
        };
        for &s in &speakers {
            let d = decisions.iter().find(|d| d.speaker == s);
            let person_id = d.map(|d| d.person_id);
            db.link_speaker(v.id, s, person_id, d.map(|d| d.confidence as f64))?;
            let name = match person_id {
                Some(p) => db.person(project_id, p).ok().and_then(|p| p.name),
                None => None,
            };
            report.links.push(VideoLink {
                video_id: v.id,
                speaker: s,
                person_id,
                name,
                correlation: d.and_then(|d| d.correlation),
                confidence: d.map(|d| d.confidence),
            });
            if let Some(p) = person_id {
                let idx = all.iter().enumerate().filter(|(i, w)| w.video_id == v.id && labels[*i] == s).map(|(i, _)| i);
                voice.entry(p).or_default().extend(idx);
            }
        }
        if !track_people.is_empty() {
            on_step(Step::Done { video_id: v.id });
        }
    }
    for p in db.people(project_id)? {
        let blob = voice.get(&p.id).map(|idx| {
            let c = cluster::centroid(&idx.iter().map(|&i| all[i].embedding.as_slice()).collect::<Vec<_>>());
            embedding_to_blob(&c)
        });
        db.conn.execute("UPDATE people SET voice_embedding = ?2 WHERE id = ?1", params![p.id, blob])?;
    }
    report.timings.link_s = t3.elapsed().as_secs_f64();

    db.conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![fp_key, fingerprint(db, &videos)?],
    )?;
    Ok(report)
}

/// A face thumbnail for a person: the sharpest stored box, cropped from the source video with
/// some margin and cached as `<data>/people/p<person>-v<video>-<ms>.jpg`. None when the person
/// has no usable face box.
pub fn person_thumbnail(db: &Db, data_dir: &Path, ffmpeg: &Path, person_id: i64) -> Result<Option<PathBuf>, Error> {
    let Some((video_id, t_s, x, y, w, h)) = db.person_best_box(person_id)? else { return Ok(None) };
    let dir = data_dir.join("people");
    let out = dir.join(format!("p{person_id}-v{video_id}-{}.jpg", (t_s * 1000.0).round() as i64));
    if out.is_file() {
        return Ok(Some(out));
    }
    let row: Option<(String, i64, i64)> = db
        .conn
        .query_row(
            "SELECT vf.path, COALESCE(v.width, 1280), COALESCE(v.height, 720)
               FROM videos v JOIN video_files vf ON vf.video_id = v.id WHERE v.id = ?1 LIMIT 1",
            [video_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((src, vw, vh)) = row else { return Ok(None) };
    let (vw, vh) = (vw.max(2) as f32, vh.max(2) as f32);
    // Square around the face, 1.7× its size, kept inside the frame.
    let side = ((w * vw).max(h * vh) * 1.7).min(vw).min(vh);
    let cx = (x + w / 2.0) * vw;
    let cy = (y + h / 2.0) * vh;
    let left = (cx - side / 2.0).clamp(0.0, vw - side);
    let top = (cy - side / 2.0).clamp(0.0, vh - side);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Io(dir.clone(), e))?;
    let status = crate::proc::std_command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-y", "-ss", &format!("{t_s:.3}"), "-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &format!("crop={:.0}:{:.0}:{:.0}:{:.0},scale=192:192", side, side, left, top),
            "-q:v",
            "3",
        ])
        .arg(&out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| Error::Io(ffmpeg.to_path_buf(), e))?;
    Ok(if status.success() && out.is_file() { Some(out) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn migration_v17_schema() {
        let db = Db::open_in_memory().unwrap();
        let cols = |t: &str| -> Vec<String> {
            let mut st = db.conn.prepare(&format!("SELECT name FROM pragma_table_info('{t}')")).unwrap();
            st.query_map([], |r| r.get(0)).unwrap().collect::<Result<_, _>>().unwrap()
        };
        assert_eq!(cols("people"), ["id", "project_id", "name", "handle", "role", "face_embedding", "voice_embedding"]);
        assert_eq!(cols("speaker_turns"), ["id", "video_id", "start_s", "end_s", "speaker", "person_id", "confidence"]);
        assert!(cols("face_tracks").contains(&"person_id".to_string()));
        let ts = cols("transcript_segments");
        assert!(ts.contains(&"speaker".to_string()) && ts.contains(&"person_id".to_string()));
        assert!(cols("speaker_windows").contains(&"embedding".to_string()));
    }

    #[test]
    fn turns_and_links_roundtrip_and_cascade() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO videos (id, content_hash, size, duration_s) VALUES (1, 'h', 1, 60.0);
                 INSERT INTO transcript_segments (id, video_id, start_s, end_s, text) VALUES
                    (1, 1, 0.0, 4.0, 'hola'), (2, 1, 4.0, 8.0, 'qué tal'), (3, 1, 8.0, 8.3, 'sí');",
            )
            .unwrap();
        db.conn.execute("INSERT INTO people (id, project_id, name) VALUES (5, ?1, 'Ana')", [p.id]).unwrap();
        let spans = eligible_windows(&db, 1).unwrap();
        assert_eq!(spans.len(), 2 + 2, "0.3 s segment skipped; 4 s → 2 windows each");
        let windows: Vec<(WindowSpan, Vec<f32>)> = spans.iter().map(|w| (*w, vec![1.0, 0.0])).collect();
        db.set_speaker_windows(1, &windows).unwrap();
        assert_eq!(db.speaker_windows(1).unwrap().len(), 4);

        let turns = vec![Turn { start_s: 0.0, end_s: 4.0, speaker: 1 }, Turn { start_s: 4.0, end_s: 8.0, speaker: 2 }];
        db.set_speaker_turns(1, &turns).unwrap();
        db.link_speaker(1, 2, Some(5), Some(0.9)).unwrap();
        let got = db.speaker_turns(1, 0.0, 100.0).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].name.as_deref(), Some("Ana"));
        assert_eq!(got[0].person_id, None);
        let seg: (Option<i64>, Option<i64>) = db
            .conn
            .query_row("SELECT speaker, person_id FROM transcript_segments WHERE id = 2", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap();
        assert_eq!(seg, (Some(2), Some(5)));
        let short: Option<i64> =
            db.conn.query_row("SELECT speaker FROM transcript_segments WHERE id = 3", [], |r| r.get(0)).unwrap();
        assert_eq!(short, None, "a segment no turn overlaps has no speaker");
    }
}
