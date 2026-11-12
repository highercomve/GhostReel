//! Generate clips: short, self-contained moments mined from every selected video of a project,
//! saved as a **clip set** of one-clip scripts (plan: `.agents/clips/plan.md`, Milestone 1 §4).
//!
//! The run is deterministic wherever it can be:
//! 1. **Candidates (code).** Sentence-bounded windows in the length range, from the stored
//!    transcript; never starting on an off-mic line, never across a hole in the transcript, not
//!    on shaky stretches. Each carries a visual label read from the keyframe descriptions (screen
//!    share vs people), face numbers from `face_tracks`, and speakers when segments have them.
//! 2. **Workers (model, parallel).** One job per ~10 min of a video: brief + glossary + transcript
//!    slice + candidates → scored picks. Any backend: server, CLI agent, local.
//! 3. **Orchestrator (code + model).** Merge, drop overlaps, prefer faces for vertical formats,
//!    spread across videos and speakers; the orchestrator model picks N and writes the hooks.
//!    When it is missing or fails, code picks.
//! 4. **Check + save (code).** The same repair passes `script import` runs, then
//!    `script_check::check`; saved with `kind = 'clip'` and a shared `set_id`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Error;
use crate::agents::{AgentJob, AgentPool, AgentRole, AgentSelection, AgentSpec, PoolProgress, ResolvedSelection};
use crate::config::{Config, ScriptConfig};
use crate::db::Db;
use crate::projects::Project;
use crate::script::{Audio, Beat, CaptionStyle, Issue, IssueSeverity, Script, ScriptClip};

pub const DEFAULT_COUNT: usize = 8;
pub const DEFAULT_MIN_S: f64 = 20.0;
pub const DEFAULT_MAX_S: f64 = 60.0;
/// Transcript one worker reads at once.
pub const DEFAULT_WINDOW_S: f64 = 600.0;
/// Candidates shown to one worker; more only crowds a small model's context.
const MAX_CANDIDATES_PER_JOB: usize = 30;
/// A silence longer than this between two lines is a hole in the transcript (dropped speech),
/// not a pause: a clip never spans one.
const TRANSCRIPT_HOLE_S: f64 = 6.0;
/// A sentence unit longer than this is a run-on segment chain: it is split at the weak edges.
const LONG_UNIT_S: f64 = 25.0;

/// Output frame for the clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClipFormat {
    #[default]
    #[serde(rename = "original")]
    Original,
    #[serde(rename = "9:16", alias = "vertical")]
    Vertical,
    #[serde(rename = "1:1", alias = "square")]
    Square,
}

impl ClipFormat {
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s.trim().to_lowercase().as_str() {
            "" | "original" | "16:9" => Ok(Self::Original),
            "9:16" | "vertical" | "portrait" => Ok(Self::Vertical),
            "1:1" | "square" => Ok(Self::Square),
            other => Err(Error::Invalid(format!("unknown clip format '{other}': use original, 9:16 or 1:1"))),
        }
    }

    /// Width/height to write on the script; `None` = the project's sequence size.
    pub fn size(&self) -> Option<(i64, i64)> {
        match self {
            Self::Original => None,
            Self::Vertical => Some((1080, 1920)),
            Self::Square => Some((1080, 1080)),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Vertical => "9:16",
            Self::Square => "1:1",
        }
    }
}

/// What a run is asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipOptions {
    /// Optional editorial brief ("moments about cost", "funny bits").
    pub brief: String,
    pub count: usize,
    pub min_s: f64,
    pub max_s: f64,
    pub format: ClipFormat,
    /// Videos to mine; empty = every video in the project's library.
    pub video_ids: Vec<i64>,
    pub window_s: f64,
    /// Skip the orchestrator model: code merges and picks from the workers' scores.
    pub worker_only: bool,
    /// How vertical and square clips are framed (ignored for the original format).
    pub layout: crate::framing::Layout,
    /// Which agents this run uses; what it leaves out comes from the app's choice.
    pub agents: AgentSelection,
}

impl Default for ClipOptions {
    fn default() -> Self {
        Self {
            brief: String::new(),
            count: DEFAULT_COUNT,
            min_s: DEFAULT_MIN_S,
            max_s: DEFAULT_MAX_S,
            format: ClipFormat::Original,
            video_ids: Vec::new(),
            window_s: DEFAULT_WINDOW_S,
            worker_only: false,
            layout: crate::framing::Layout::FollowSpeaker,
            agents: AgentSelection::default(),
        }
    }
}

impl ClipOptions {
    pub fn validate(&self) -> Result<(), Error> {
        if self.count == 0 || self.count > 50 {
            return Err(Error::Invalid("clip count must be between 1 and 50".into()));
        }
        if !(3.0..=600.0).contains(&self.min_s) || self.max_s < self.min_s + 2.0 || self.max_s > 900.0 {
            return Err(Error::Invalid(format!(
                "clip length range {:.0}–{:.0} s is not usable (min ≥ 3 s, max ≥ min + 2 s)",
                self.min_s, self.max_s
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Survey: what the index knows about one video, read once.

/// One stored transcript segment.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
    pub off_mic: bool,
    pub speaker: Option<String>,
}

/// Consecutive segments that make one sentence (or as near as an unpunctuated transcript allows).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Unit {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
    /// The first line is someone off the microphone.
    pub starts_off_mic: bool,
    /// Ends on punctuation (rather than on a pause / capital letter / forced split).
    pub strong_end: bool,
    pub speakers: Vec<String>,
}

/// Faces in a range, from `face_tracks`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceInfo {
    /// Average number of faces on screen.
    pub faces: f64,
    /// Average face box height as a share of the frame height (0–1).
    pub avg_box_h: f64,
}

/// What the picture looks like in a range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Visual {
    /// `faces`, `screen`, `mixed` or `unknown`.
    pub label: String,
    /// Share of the described keyframes that are a screen share / recording.
    pub screen_share: f64,
    pub frames: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<FaceInfo>,
    /// How much of the range is a two-shot, a single face, a screen share or empty, from the
    /// face boxes (`crate::scenes`); absent when the faces were never indexed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<crate::scenes::SceneMix>,
}

impl Visual {
    pub fn summary(&self) -> String {
        if let Some(m) = &self.layout {
            return match &self.faces {
                Some(f) => format!("{}; face height {:.0}%", m.summary(), f.avg_box_h * 100.0),
                None => m.summary(),
            };
        }
        let base = match self.label.as_str() {
            "screen" => "screen share".to_string(),
            "mixed" => format!("mixed ({:.0}% screen)", self.screen_share * 100.0),
            "faces" => "people on camera".to_string(),
            _ => "picture unknown".to_string(),
        };
        match &self.faces {
            Some(f) => format!("{base}; {:.1} faces, face height {:.0}%", f.faces, f.avg_box_h * 100.0),
            None => base,
        }
    }

    /// How well this suits a vertical crop: a big face is ideal, a screen share with thumbnails
    /// is not.
    fn vertical_bonus(&self) -> f64 {
        // Measured from the faces: a two-shot crops best (two people to cut between), a single
        // face well; a screen share only as screen over a small tile, an empty picture not at all.
        if let Some(m) = &self.layout {
            return 2.0 * m.two_shot + 1.4 * m.single_face - 1.5 * m.screen_share - 1.0 * m.no_face;
        }
        let label = match self.label.as_str() {
            "faces" => 1.5,
            "mixed" => 0.2,
            "screen" => -1.5,
            _ => 0.0,
        };
        let face = self.faces.as_ref().map(|f| {
            if f.avg_box_h >= 0.2 {
                0.5
            } else if f.avg_box_h < 0.08 {
                -0.3
            } else {
                0.0
            }
        });
        label + face.unwrap_or(0.0)
    }
}

struct FrameNote {
    t_s: f64,
    screen: bool,
}

struct Track {
    start_s: f64,
    end_s: f64,
    box_h: f64,
}

/// Everything the run needs from one video.
pub struct VideoSurvey {
    pub video_id: i64,
    pub duration_s: f64,
    pub units: Vec<Unit>,
    frames: Vec<FrameNote>,
    tracks: Vec<Track>,
    motion: Vec<crate::steadiness::Window>,
    /// What the picture is, stretch by stretch (`None`: faces never indexed).
    scenes: Option<Vec<crate::scenes::SceneSpan>>,
}

/// Lowercase words a sentence does not end on (Spanish + English): an unpunctuated segment
/// ending on one is cut mid-sentence ("…entonces aníbal creó un").
const FUNCTION_WORDS: &[&str] = &[
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "entonces", "es", "esa", "ese", "esta", "este", "hay",
    "la", "las", "lo", "los", "me", "mi", "muy", "ni", "no", "o", "para", "pero", "por", "porque", "que", "se", "si",
    "sin", "su", "sus", "te", "tu", "un", "una", "unas", "unos", "y", "ya", "yo", "an", "and", "are", "as", "at", "be",
    "because", "but", "by", "for", "from", "i", "if", "in", "is", "it", "my", "of", "on", "or", "so", "that", "the",
    "their", "then", "this", "to", "was", "we", "with", "you", "your",
];

fn last_word(text: &str) -> String {
    text.split_whitespace().last().unwrap_or_default().trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

fn starts_upper(text: &str) -> bool {
    text.trim_start()
        .chars()
        .find(|c| c.is_alphabetic() || *c == '¿' || *c == '¡')
        .is_some_and(|c| c.is_uppercase() || c == '¿' || c == '¡')
}

/// Whether a sentence may end after `seg`: `Some(true)` on punctuation, `Some(false)` on a
/// weaker sign (a pause, the next line capitalised, the end of the video), `None` when the line
/// runs on.
pub fn sentence_end(seg: &Segment, next: Option<&Segment>) -> Option<bool> {
    let t = seg.text.trim_end();
    if t.ends_with(['.', '!', '?', '…']) {
        return Some(true);
    }
    if FUNCTION_WORDS.contains(&last_word(t).as_str()) {
        return None;
    }
    let Some(next) = next else { return Some(false) };
    if next.start_s - seg.end_s >= 0.6 || starts_upper(&next.text) {
        return Some(false);
    }
    None
}

/// Group segments into sentence units. Long run-on chains (the server transcriber returns 20–50 s
/// lowercase lines without punctuation) are closed at the first edge that is not mid-phrase once
/// they pass [`LONG_UNIT_S`], so they still offer cut points; a clip never ends inside a segment.
pub fn sentence_units(segs: &[Segment]) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut cur: Option<Unit> = None;
    for (i, seg) in segs.iter().enumerate() {
        let next = segs.get(i + 1);
        let u = cur.get_or_insert_with(|| Unit {
            start_s: seg.start_s,
            end_s: seg.end_s,
            text: String::new(),
            starts_off_mic: seg.off_mic,
            strong_end: false,
            speakers: Vec::new(),
        });
        if !u.text.is_empty() {
            u.text.push(' ');
        }
        u.text.push_str(seg.text.trim());
        u.end_s = seg.end_s;
        if let Some(s) = &seg.speaker
            && !u.speakers.contains(s)
        {
            u.speakers.push(s.clone());
        }
        let hole = next.is_some_and(|n| n.start_s - seg.end_s > TRANSCRIPT_HOLE_S);
        let end = sentence_end(seg, next);
        let forced = end.is_none()
            && u.end_s - u.start_s > LONG_UNIT_S
            && !FUNCTION_WORDS.contains(&last_word(&seg.text).as_str());
        if end.is_some() || hole || forced || next.is_none() {
            let mut done = cur.take().unwrap();
            done.strong_end = end == Some(true);
            units.push(done);
        }
    }
    units
}

/// Speaker columns another track may add to `transcript_segments` (v17); absent on older DBs.
fn is_screen_frame(raw: &str) -> bool {
    let v: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
    let shot = v["shot"].as_str().unwrap_or_default().to_lowercase();
    let desc = v["description"].as_str().unwrap_or(raw).to_lowercase();
    if shot.contains("split") {
        return false;
    }
    if shot.contains("screen") {
        return true;
    }
    ["screen recording", "screen share", "screenshare", "code editor", "terminal window", "slide", "desktop"]
        .iter()
        .any(|k| desc.contains(k))
}

/// The videos a run mines: the requested ones that are in the project, or the whole library.
pub fn project_videos(db: &Db, project_id: i64, requested: &[i64]) -> Result<Vec<i64>, Error> {
    let all: Vec<i64> = db
        .conn
        .prepare(
            "SELECT DISTINCT vf.video_id FROM video_files vf
             JOIN project_folders pf ON pf.folder_id = vf.folder_id
             WHERE pf.project_id = ?1
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x
                               WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)
             ORDER BY vf.video_id",
        )?
        .query_map([project_id], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    if requested.is_empty() {
        return Ok(all);
    }
    let missing: Vec<_> = requested.iter().filter(|v| !all.contains(v)).collect();
    if !missing.is_empty() {
        return Err(Error::Invalid(format!("videos {missing:?} are not in this project's library")));
    }
    Ok(requested.to_vec())
}

impl VideoSurvey {
    pub fn load(db: &Db, video_id: i64) -> Result<Self, Error> {
        // Who speaks each line: the person's name, else the person or voice number.
        let sql = "SELECT ts.start_s, ts.end_s, ts.text, COALESCE(ts.off_mic, 0),
                    COALESCE(NULLIF(TRIM(p.name), ''),
                             CASE WHEN ts.person_id IS NOT NULL THEN 'Person ' || ts.person_id END,
                             CASE WHEN ts.speaker IS NOT NULL THEN 'Speaker ' || ts.speaker END)
             FROM transcript_segments ts LEFT JOIN people p ON p.id = ts.person_id
             WHERE ts.video_id = ?1 AND TRIM(ts.text) <> '' ORDER BY ts.start_s";
        let glossary = db.glossary_for_video(video_id).unwrap_or_default();
        let segs: Vec<Segment> = db
            .conn
            .prepare(sql)?
            .query_map([video_id], |r| {
                Ok(Segment {
                    start_s: r.get(0)?,
                    end_s: r.get(1)?,
                    text: r.get(2)?,
                    off_mic: r.get::<_, i64>(3)? != 0,
                    speaker: r.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|mut s| {
                s.text = glossary.correct(&s.text);
                s
            })
            .collect();
        let frames = db
            .conn
            .prepare("SELECT t_s, description_json FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s")?
            .query_map([video_id], |r| Ok((r.get::<_, f64>(0)?, r.get::<_, String>(1)?)))?
            .flatten()
            .filter(|(_, raw)| !raw.contains("\"error\""))
            .map(|(t_s, raw)| FrameNote { t_s, screen: is_screen_frame(&raw) })
            .collect();
        let tracks = db
            .face_tracks(video_id)
            .unwrap_or_default()
            .into_iter()
            .map(|t| Track { start_s: t.start_s, end_s: t.end_s, box_h: t.avg_box_h as f64 })
            .collect();
        let duration_s: f64 = db
            .conn
            .query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [video_id], |r| r.get(0))
            .optional()?
            .unwrap_or(0.0);
        let scenes = crate::scenes::for_range(db, video_id, 0.0, duration_s.max(0.0))?;
        Ok(Self {
            video_id,
            duration_s,
            units: sentence_units(&segs),
            frames,
            tracks,
            motion: db.motion_windows(video_id).unwrap_or_default(),
            scenes,
        })
    }

    /// Build a survey from parts (tests).
    pub fn from_units(video_id: i64, units: Vec<Unit>) -> Self {
        Self {
            video_id,
            duration_s: units.last().map(|u| u.end_s).unwrap_or(0.0),
            units,
            frames: Vec::new(),
            tracks: Vec::new(),
            motion: Vec::new(),
            scenes: None,
        }
    }

    /// The same survey with its scenes classified from these face observations (tests).
    pub fn with_faces(mut self, obs: &[crate::scenes::FaceObs]) -> Self {
        self.scenes = Some(crate::scenes::segment(obs, 0.0, self.duration_s));
        self
    }

    pub fn visual(&self, in_s: f64, out_s: f64) -> Visual {
        let mut inside: Vec<&FrameNote> = self.frames.iter().filter(|f| f.t_s >= in_s && f.t_s <= out_s).collect();
        if inside.is_empty()
            && let Some(f) = self.frames.iter().rev().find(|f| f.t_s < in_s && in_s - f.t_s < 30.0)
        {
            inside.push(f);
        }
        let len = (out_s - in_s).max(0.01);
        let mut face_s = 0.0;
        let mut box_w = 0.0;
        for t in &self.tracks {
            let ov = (t.end_s.min(out_s) - t.start_s.max(in_s)).max(0.0);
            face_s += ov;
            box_w += ov * t.box_h;
        }
        let faces = (face_s > 0.0).then(|| FaceInfo {
            faces: (face_s / len * 10.0).round() / 10.0,
            avg_box_h: (box_w / face_s * 1000.0).round() / 1000.0,
        });
        let frames = inside.len();
        let screen_share =
            if frames == 0 { 0.0 } else { inside.iter().filter(|f| f.screen).count() as f64 / frames as f64 };
        let layout = self.scenes.as_ref().map(|s| crate::scenes::mix(s, in_s, out_s));
        let label = if let Some(m) = &layout {
            // The faces say it better than a few keyframe descriptions can.
            if m.people() >= 0.75 {
                "faces"
            } else if m.screen_share + m.no_face >= 0.6 {
                "screen"
            } else {
                "mixed"
            }
        } else if frames == 0 {
            if faces.as_ref().is_some_and(|f| f.faces >= 0.5) { "faces" } else { "unknown" }
        } else if screen_share >= 0.6 {
            "screen"
        } else if screen_share <= 0.25 {
            "faces"
        } else {
            "mixed"
        };
        let screen_share = layout.map(|m| m.screen_share).unwrap_or(screen_share);
        Visual { label: label.into(), screen_share: (screen_share * 100.0).round() / 100.0, frames, faces, layout }
    }

    /// Seconds of the range on shaky stretches, with the run's shake settings.
    pub fn shaky_s(&self, cfg: &ScriptConfig, in_s: f64, out_s: f64) -> f64 {
        if cfg.max_shake_jerk <= 0.0 {
            return 0.0;
        }
        crate::script_check::Steadiness::measure(&self.motion, cfg, in_s, out_s).map(|s| s.shaky_s).unwrap_or(0.0)
    }

    pub fn speakers(&self, in_s: f64, out_s: f64) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for u in self.units.iter().filter(|u| u.end_s > in_s && u.start_s < out_s) {
            for s in &u.speakers {
                if !out.contains(s) {
                    out.push(s.clone());
                }
            }
        }
        out
    }

    fn text(&self, in_s: f64, out_s: f64) -> String {
        self.units
            .iter()
            .filter(|u| u.start_s >= in_s - 0.05 && u.end_s <= out_s + 0.05)
            .map(|u| u.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Move a range onto unit edges: the nearest unit start at or around `in_s` (not off-mic) and
    /// the nearest unit end around `out_s`.
    pub fn snap(&self, in_s: f64, out_s: f64) -> Option<(f64, f64)> {
        let start = self
            .units
            .iter()
            .filter(|u| !u.starts_off_mic)
            .min_by(|a, b| (a.start_s - in_s).abs().total_cmp(&(b.start_s - in_s).abs()))?
            .start_s;
        let end = self
            .units
            .iter()
            .filter(|u| u.end_s > start)
            .min_by(|a, b| {
                let pa = (a.end_s - out_s).abs() - if a.strong_end { 1.5 } else { 0.0 };
                let pb = (b.end_s - out_s).abs() - if b.strong_end { 1.5 } else { 0.0 };
                pa.total_cmp(&pb)
            })?
            .end_s;
        (end > start).then_some((start, end))
    }

    /// Whether `in_s..out_s` spans a hole in the transcript.
    fn spans_hole(&self, in_s: f64, out_s: f64) -> bool {
        let inside: Vec<&Unit> =
            self.units.iter().filter(|u| u.end_s > in_s + 0.05 && u.start_s < out_s - 0.05).collect();
        inside.windows(2).any(|w| w[1].start_s - w[0].end_s > TRANSCRIPT_HOLE_S)
    }
}

// ---------------------------------------------------------------------------------------------
// Candidates

/// A sentence-bounded window that could be a clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    pub text: String,
    pub visual: Visual,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speakers: Vec<String>,
    pub shaky_s: f64,
    /// Ends on punctuation, not on a pause.
    pub strong_end: bool,
    /// Code's own estimate (sentence quality, steadiness, picture); used to cap the list.
    pub prescore: f64,
}

impl Candidate {
    pub fn duration_s(&self) -> f64 {
        self.out_s - self.in_s
    }
}

fn overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0.0)
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Every usable sentence-bounded window of one video, best first per start, de-duplicated.
pub fn candidates(survey: &VideoSurvey, cfg: &ScriptConfig, opts: &ClipOptions) -> Vec<Candidate> {
    let units = &survey.units;
    let mut raw: Vec<Candidate> = Vec::new();
    let mid = (opts.min_s + opts.max_s) / 2.0;
    for i in 0..units.len() {
        if units[i].starts_off_mic {
            continue;
        }
        // A clip opens where a sentence opens: after a unit that ended one, or at the top.
        let clean_start = i == 0 || units[i - 1].strong_end || units[i].start_s - units[i - 1].end_s >= 0.6;
        let mut best: Option<Candidate> = None;
        for j in i..units.len() {
            if j > i && units[j].start_s - units[j - 1].end_s > TRANSCRIPT_HOLE_S {
                break;
            }
            let (in_s, out_s) = (units[i].start_s, units[j].end_s);
            let dur = out_s - in_s;
            if dur > opts.max_s {
                break;
            }
            if dur < opts.min_s {
                continue;
            }
            let text: String = units[i..=j].iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" ");
            let words = text.split_whitespace().count() as f64;
            if words / dur < 1.0 {
                continue; // mostly silence
            }
            let strong_end = units[j].strong_end;
            let shaky_s = survey.shaky_s(cfg, in_s, out_s);
            let visual = survey.visual(in_s, out_s);
            let mut prescore = 0.0;
            prescore += if strong_end { 1.0 } else { 0.3 };
            prescore += if clean_start { 0.5 } else { 0.0 };
            prescore -= if shaky_s > dur * crate::script_check::SHAKY_SHARE {
                2.0
            } else if shaky_s > 0.0 {
                0.4
            } else {
                0.0
            };
            prescore -= ((dur - mid).abs() / (opts.max_s - opts.min_s).max(1.0)) * 0.6;
            if opts.format != ClipFormat::Original {
                prescore += visual.vertical_bonus() * 0.4;
            }
            let c = Candidate {
                id: String::new(),
                video_id: survey.video_id,
                in_s: round2(in_s),
                out_s: round2(out_s),
                text,
                speakers: survey.speakers(in_s, out_s),
                visual,
                shaky_s: round2(shaky_s),
                strong_end,
                prescore: round2(prescore),
            };
            if best.as_ref().is_none_or(|b| c.prescore >= b.prescore) {
                best = Some(c);
            }
        }
        if let Some(c) = best
            && c.shaky_s <= c.duration_s() * crate::script_check::SHAKY_SHARE
        {
            raw.push(c);
        }
    }
    let mut out = dedupe(raw, 0.5, usize::MAX);
    out.sort_by(|a, b| a.in_s.total_cmp(&b.in_s));
    for (k, c) in out.iter_mut().enumerate() {
        c.id = format!("v{}c{}", c.video_id, k + 1);
    }
    out
}

/// Keep the best candidates, dropping any that shares more than `max_share` of its length with
/// one already kept.
fn dedupe(mut items: Vec<Candidate>, max_share: f64, cap: usize) -> Vec<Candidate> {
    items.sort_by(|a, b| b.prescore.total_cmp(&a.prescore).then(a.in_s.total_cmp(&b.in_s)));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in items {
        if kept.len() >= cap {
            break;
        }
        let clash = kept.iter().any(|k| {
            k.video_id == c.video_id
                && overlap((k.in_s, k.out_s), (c.in_s, c.out_s)) > c.duration_s().min(k.duration_s()) * max_share
        });
        if !clash {
            kept.push(c);
        }
    }
    kept
}

/// Candidates for every requested video of a project.
pub fn find_candidates(
    db: &Db,
    project_id: i64,
    cfg: &ScriptConfig,
    opts: &ClipOptions,
) -> Result<Vec<Candidate>, Error> {
    let mut all = Vec::new();
    for vid in project_videos(db, project_id, &opts.video_ids)? {
        let survey = VideoSurvey::load(db, vid)?;
        all.extend(candidates(&survey, cfg, opts));
    }
    Ok(all)
}

// ---------------------------------------------------------------------------------------------
// Picks

/// A scored clip proposal (from a worker, or from an outside agent through `save_clips`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    #[serde(default)]
    pub id: String,
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub quote: String,
    #[serde(default)]
    pub why: String,
    #[serde(default)]
    pub score: f64,
    /// Short on-screen hook title.
    #[serde(default)]
    pub hook: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<String>,
    #[serde(default)]
    pub visual: Visual,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speakers: Vec<String>,
}

impl Pick {
    pub fn duration_s(&self) -> f64 {
        self.out_s - self.in_s
    }
    fn range(&self) -> (f64, f64) {
        (self.in_s, self.out_s)
    }
    fn primary_speaker(&self) -> Option<&str> {
        self.speakers.first().map(String::as_str)
    }
}

pub fn worker_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["picks"],
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["candidate", "in_s", "out_s", "title", "quote", "why", "score", "hook"],
                    "properties": {
                        "candidate": { "type": "string" },
                        "in_s": { "type": "number" },
                        "out_s": { "type": "number" },
                        "title": { "type": "string" },
                        "quote": { "type": "string" },
                        "why": { "type": "string" },
                        "score": { "type": "number" },
                        "hook": { "type": "string" }
                    }
                }
            }
        }
    })
}

pub fn orchestrator_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["clips"],
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "title", "hook", "why"],
                    "properties": {
                        "id": { "type": "string" },
                        "title": { "type": "string" },
                        "hook": { "type": "string" },
                        "why": { "type": "string" }
                    }
                }
            }
        }
    })
}

fn shorten(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n).collect::<String>() + "…" }
}

fn tail(s: &str, n: usize) -> String {
    let c: Vec<char> = s.trim().chars().collect();
    if c.len() <= n {
        c.into_iter().collect()
    } else {
        "…".to_string() + &c[c.len() - n..].iter().collect::<String>()
    }
}

fn format_line(opts: &ClipOptions) -> String {
    match opts.format {
        ClipFormat::Vertical => "vertical 9:16 (phone): people on camera crop well (a two-shot best: the crop cuts \
             between them), screen shares with tiny faces do not; each proposal's picture says how much of it is \
             which"
            .into(),
        ClipFormat::Square => "square 1:1: people on camera crop well (a two-shot best), screen shares less so; \
             each proposal's picture says how much of it is which"
            .into(),
        ClipFormat::Original => "the original frame".into(),
    }
}

/// One worker's prompt: a slice of one video's transcript and the candidates inside it.
pub fn worker_prompt(
    survey: &VideoSurvey,
    from_s: f64,
    to_s: f64,
    cands: &[&Candidate],
    opts: &ClipOptions,
    glossary: Option<&str>,
    want: usize,
) -> String {
    let mut p = String::new();
    p.push_str(
        "You pick short clips for social media from a long recording. A good clip makes complete sense \
         with no context (no \"as I said\", no \"this\"/\"that\" pointing at something earlier), grabs \
         attention in its first seconds, lands one complete thought, story or punchline, and ends on a \
         finished sentence.\n",
    );
    if opts.brief.trim().is_empty() {
        p.push_str("Brief: none — pick the strongest self-contained moments.\n");
    } else {
        p.push_str(&format!("Brief: {}\n", opts.brief.trim()));
    }
    if let Some(g) = glossary {
        p.push_str(g);
        p.push('\n');
    }
    p.push_str(&format!(
        "Output format: {}. Clip length {:.0}–{:.0} s.\n\n",
        format_line(opts),
        opts.min_s,
        opts.max_s
    ));
    p.push_str(&format!(
        "Transcript of video #{} from {:.0} s to {:.0} s ([start–end] seconds; (off-mic) = someone away from the microphone):\n",
        survey.video_id, from_s, to_s
    ));
    for u in survey.units.iter().filter(|u| u.end_s > from_s && u.start_s < to_s) {
        let who = if u.speakers.is_empty() { String::new() } else { format!(" {}:", u.speakers.join("/")) };
        let off = if u.starts_off_mic { " (off-mic)" } else { "" };
        p.push_str(&format!("[{:.1}–{:.1}]{off}{who} {}\n", u.start_s, u.end_s, u.text.trim()));
    }
    p.push_str("\nCandidate clips (sentence-bounded; prefer these, or move in_s/out_s to other line edges above):\n");
    for c in cands {
        p.push_str(&format!(
            "{}  {:.1}–{:.1}  {:.0} s  {}  \"{}\" … \"{}\"\n",
            c.id,
            c.in_s,
            c.out_s,
            c.duration_s(),
            c.visual.summary(),
            shorten(&c.text, 90),
            tail(&c.text, 60)
        ));
    }
    p.push_str(&format!(
        "\nPick up to {want} of the best clips here (fewer if few are good, none if none are). For each give: \
         candidate (its id, or \"\" if you moved the bounds), in_s, out_s, title (≤ 60 characters, in the \
         transcript's language), quote (its key line, verbatim), why (one sentence), score (0–10: 10 = works \
         perfectly on its own), hook (a 2–6 word on-screen hook in the transcript's language)."
    ));
    p
}

/// Turn a worker's answer into picks: bounds resolved against the candidates and snapped onto
/// sentence edges, too-short / too-long ranges dropped.
pub fn parse_worker_picks(
    answer: &Value,
    survey: &VideoSurvey,
    cands: &[&Candidate],
    opts: &ClipOptions,
    cfg: &ScriptConfig,
) -> Vec<Pick> {
    let mut out = Vec::new();
    for item in answer["picks"].as_array().cloned().unwrap_or_default() {
        let cid = item["candidate"].as_str().unwrap_or_default().trim().to_string();
        let cand = cands.iter().find(|c| c.id == cid);
        let (mut in_s, mut out_s) = (item["in_s"].as_f64().unwrap_or(0.0), item["out_s"].as_f64().unwrap_or(0.0));
        let bounds = match cand {
            // Asked bounds within a second of the candidate's: the candidate's exact edges.
            Some(c) if (in_s - c.in_s).abs() <= 1.0 && (out_s - c.out_s).abs() <= 1.0 => Some((c.in_s, c.out_s)),
            Some(c) if out_s <= in_s => Some((c.in_s, c.out_s)),
            _ if out_s > in_s => survey.snap(in_s, out_s),
            _ => None,
        };
        let Some(b) = bounds else { continue };
        (in_s, out_s) = b;
        let dur = out_s - in_s;
        if dur < opts.min_s * 0.85 || dur > opts.max_s + 2.0 || survey.spans_hole(in_s, out_s) {
            // A nudge that broke the range falls back to the candidate it came from.
            match cand {
                Some(c) => (in_s, out_s) = (c.in_s, c.out_s),
                None => continue,
            }
        }
        if survey.shaky_s(cfg, in_s, out_s) > (out_s - in_s) * crate::script_check::SHAKY_SHARE {
            continue;
        }
        let quote = item["quote"].as_str().unwrap_or_default().trim().to_string();
        out.push(Pick {
            id: String::new(),
            video_id: survey.video_id,
            in_s: round2(in_s),
            out_s: round2(out_s),
            title: shorten(item["title"].as_str().unwrap_or_default(), 80),
            quote: if quote.is_empty() { shorten(&survey.text(in_s, out_s), 160) } else { quote },
            why: item["why"].as_str().unwrap_or_default().trim().to_string(),
            score: item["score"].as_f64().unwrap_or(0.0).clamp(0.0, 10.0),
            hook: shorten(item["hook"].as_str().unwrap_or_default(), 48),
            candidate: cand.map(|c| c.id.clone()),
            visual: survey.visual(in_s, out_s),
            speakers: survey.speakers(in_s, out_s),
        });
    }
    out
}

/// Merge every worker's picks: best score first, any pick overlapping a better one dropped.
pub fn merge_picks(mut picks: Vec<Pick>) -> Vec<Pick> {
    picks.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.in_s.total_cmp(&b.in_s)));
    let mut kept: Vec<Pick> = Vec::new();
    for p in picks {
        if kept.iter().any(|k| k.video_id == p.video_id && overlap(k.range(), p.range()) > 0.0) {
            continue;
        }
        kept.push(p);
    }
    for (i, p) in kept.iter_mut().enumerate() {
        p.id = format!("p{}", i + 1);
    }
    kept
}

/// The score code ranks by: the worker's score, adjusted for how the picture suits the format.
pub fn rank_score(p: &Pick, format: ClipFormat) -> f64 {
    match format {
        ClipFormat::Original => p.score,
        _ => p.score + p.visual.vertical_bonus(),
    }
}

/// Pick `n` from a merged pool by code: rank score, minus a penalty for each clip already taken
/// from the same video (only when there are several) and the same speaker.
pub fn select(pool: &[Pick], n: usize, format: ClipFormat) -> Vec<Pick> {
    let videos: std::collections::HashSet<i64> = pool.iter().map(|p| p.video_id).collect();
    let mut taken: Vec<Pick> = Vec::new();
    let mut left: Vec<&Pick> = pool.iter().collect();
    while taken.len() < n && !left.is_empty() {
        let adjusted = |p: &Pick| {
            let same_video = taken.iter().filter(|t| t.video_id == p.video_id).count() as f64;
            let same_speaker = p
                .primary_speaker()
                .map(|s| taken.iter().filter(|t| t.primary_speaker() == Some(s)).count() as f64)
                .unwrap_or(0.0);
            rank_score(p, format) - if videos.len() > 1 { same_video * 1.0 } else { 0.0 } - same_speaker * 0.5
        };
        let (idx, _) = left
            .iter()
            .enumerate()
            .max_by(|a, b| adjusted(a.1).total_cmp(&adjusted(b.1)).then(b.1.in_s.total_cmp(&a.1.in_s)))
            .unwrap();
        let p = left.remove(idx).clone();
        if !taken.iter().any(|t| t.video_id == p.video_id && overlap(t.range(), p.range()) > 0.0) {
            taken.push(p);
        }
    }
    taken
}

/// What the orchestrator chooses from: the merged picks by rank score, at most 3×N. For a
/// cropped format, screen shares only make the pool when there are too few people-on-camera
/// picks to fill the set — a rule, so it holds whatever the model thinks of a good quote.
pub fn orchestrator_pool(merged: &[Pick], opts: &ClipOptions) -> Vec<Pick> {
    let mut pool: Vec<Pick> = merged.to_vec();
    pool.sort_by(|a, b| rank_score(b, opts.format).total_cmp(&rank_score(a, opts.format)));
    if opts.format != ClipFormat::Original {
        let framed = pool.iter().filter(|p| p.visual.label != "screen").count();
        if framed >= opts.count + 2 {
            pool.retain(|p| p.visual.label != "screen");
        }
    }
    pool.truncate((opts.count * 3).max(12));
    pool
}

/// The orchestrator's prompt: the merged pool, best first.
pub fn orchestrator_prompt(pool: &[Pick], opts: &ClipOptions, glossary: Option<&str>) -> String {
    let mut p = String::new();
    p.push_str(&format!(
        "You are the editor choosing the final {} clips for social media from proposals your assistants scored. \
         Choose exactly {} (or all, if fewer are offered). Prefer the strongest self-contained moments; spread them \
         across videos and speakers; never choose two that overlap. Output format: {}.\n",
        opts.count,
        opts.count,
        format_line(opts)
    ));
    if !opts.brief.trim().is_empty() {
        p.push_str(&format!("Brief: {}\n", opts.brief.trim()));
    }
    if let Some(g) = glossary {
        p.push_str(g);
        p.push('\n');
    }
    p.push_str("\nProposals (id · video · seconds · picture · speakers · assistant score · title · why · quote):\n");
    for c in pool {
        p.push_str(&format!(
            "{} · v{} {:.1}–{:.1} ({:.0} s) · {} · {} · {:.1} · {} · {} · \"{}\"\n",
            c.id,
            c.video_id,
            c.in_s,
            c.out_s,
            c.duration_s(),
            c.visual.summary(),
            if c.speakers.is_empty() { "-".to_string() } else { c.speakers.join("/") },
            c.score,
            c.title,
            c.why,
            shorten(&c.quote, 200)
        ));
    }
    p.push_str(
        "\nFor each chosen clip, in the order they should be published, give: id; title (≤ 60 characters, the \
         clip's name, in the transcript's language); hook (2–6 words shown on screen in the first seconds, in \
         the transcript's language, punchy, no emoji, no hashtags); why (one sentence).",
    );
    p
}

/// Apply the orchestrator's answer to the pool: known ids in its order, overlaps and duplicates
/// skipped, its titles and hooks taken; topped up by code when it chose too few.
pub fn apply_orchestrator(pool: &[Pick], answer: &Value, n: usize, format: ClipFormat) -> Vec<Pick> {
    let mut chosen: Vec<Pick> = Vec::new();
    for item in answer["clips"].as_array().cloned().unwrap_or_default() {
        if chosen.len() >= n {
            break;
        }
        let id = item["id"].as_str().unwrap_or_default().trim();
        let Some(p) = pool.iter().find(|p| p.id == id) else { continue };
        if chosen.iter().any(|c| c.id == p.id || (c.video_id == p.video_id && overlap(c.range(), p.range()) > 0.0)) {
            continue;
        }
        let mut p = p.clone();
        for (field, key, max) in [(&mut p.title, "title", 80), (&mut p.hook, "hook", 48), (&mut p.why, "why", 400)] {
            let v = item[key].as_str().unwrap_or_default().trim();
            if !v.is_empty() {
                *field = shorten(v, max);
            }
        }
        chosen.push(p);
    }
    if chosen.len() < n {
        let rest: Vec<Pick> = pool
            .iter()
            .filter(|p| {
                !chosen
                    .iter()
                    .any(|c| c.id == p.id || (c.video_id == p.video_id && overlap(c.range(), p.range()) > 0.0))
            })
            .cloned()
            .collect();
        chosen.extend(select(&rest, n - chosen.len(), format));
    }
    chosen
}

// ---------------------------------------------------------------------------------------------
// Check + save

/// One clip as checked (and, when saved, its script id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckedClip {
    pub pick: Pick,
    pub script: Script,
    pub notes: Vec<String>,
    pub issues: Vec<Issue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script_id: Option<i64>,
}

/// A clip's script: one beat, the speech clip with its hook as the on-screen title, captions on.
/// The calls a job noted, for its run log (empty when it had no tools).
fn trail_lines(trail: &Option<crate::tools::ToolTrail>) -> Vec<String> {
    trail.as_ref().and_then(|t| t.lock().ok().map(|v| v.clone())).unwrap_or_default()
}

pub fn clip_script(pick: &Pick, index: usize, total: usize, opts: &ClipOptions) -> Script {
    let hook = if pick.hook.trim().is_empty() { pick.title.clone() } else { pick.hook.clone() };
    let name = if pick.title.trim().is_empty() { hook.clone() } else { pick.title.clone() };
    let (width, height) = match opts.format.size() {
        Some((w, h)) => (Some(w), Some(h)),
        None => (None, None),
    };
    // Vertical and square clips crop on the faces instead of padding the original frame.
    let framing = opts.format.size().map(|_| crate::framing::Framing { layout: opts.layout, ..Default::default() });
    Script {
        title: format!("Clip {}/{} · {}", index + 1, total, shorten(&name, 70)),
        target_duration_s: Some(round2(pick.duration_s())),
        fps: None,
        width,
        height,
        beats: vec![Beat {
            id: "b1".into(),
            purpose: if pick.why.is_empty() { name.clone() } else { pick.why.clone() },
            narration: None,
            on_screen_text: Some(hook),
            title_layout: None,
            clips: vec![ScriptClip {
                video_id: pick.video_id,
                in_s: pick.in_s,
                out_s: pick.out_s,
                audio: Audio::Source,
                why: (!pick.quote.is_empty()).then(|| pick.quote.clone()),
                tracks: None, framing: None,
            }],
            bed: None,
            notes: None,
        }],
        captions: Some(CaptionStyle::default()),
        title_style: None,
        caption_edits: Vec::new(),
        agents: None,
        framing,
    }
}

/// Build, repair and check each clip, plus overlaps between clips of the set. Saves nothing.
pub fn check_clips(
    db: &Db,
    project: &Project,
    cfg: &ScriptConfig,
    picks: &[Pick],
    opts: &ClipOptions,
) -> Result<Vec<CheckedClip>, Error> {
    let mut out: Vec<CheckedClip> = Vec::new();
    for (i, pick) in picks.iter().enumerate() {
        let mut script = clip_script(pick, i, picks.len(), opts);
        script.fill_from_project(project);
        let notes = crate::script_check::repair_for_import(db, project.id, &mut script, cfg, Some(opts.min_s))?;
        // A clip's target is its own length: the picked range as the repair left it (snapped to
        // sentences, trimmed off a screen share) plus the closing hold it added. Recording the
        // pick's bare length made every held clip run over its own target the moment it was made.
        script.target_duration_s = Some(round2(script.total_duration_s()));
        let mut issues = crate::script_check::check(
            db,
            project.id,
            &script,
            cfg,
            &crate::script_check::CheckOptions { forbid_overlap: true, trim_min_s: Some(opts.min_s), ..Default::default() },
        )?;
        let speech: f64 = script
            .beats
            .iter()
            .flat_map(|b| &b.clips)
            .filter(|c| c.audio == Audio::Source)
            .map(|c| c.out_s - c.in_s)
            .sum();
        if speech < opts.min_s * 0.8 || speech > opts.max_s * 1.25 {
            issues.push(Issue {
                fixes: Vec::new(),
                severity: IssueSeverity::Warning,
                beat_id: None,
                clip_index: None,
                message: format!(
                    "clip runs {speech:.1} s, outside the {:.0}–{:.0} s asked for",
                    opts.min_s, opts.max_s
                ),
            });
        }
        let main = script.beats.first().and_then(|b| b.clips.first()).cloned();
        if let Some(m) = &main {
            for (j, other) in out.iter().enumerate() {
                if let Some(o) = other.script.beats.first().and_then(|b| b.clips.first())
                    && o.video_id == m.video_id
                    && overlap((o.in_s, o.out_s), (m.in_s, m.out_s)) > 0.1
                {
                    issues.push(Issue {
                        fixes: Vec::new(),
                        severity: IssueSeverity::Warning,
                        beat_id: None,
                        clip_index: None,
                        message: format!("overlaps clip {} of this set", j + 1),
                    });
                }
            }
        }
        // The pick follows the repaired bounds, so the grid shows what will play.
        let mut pick = pick.clone();
        if let Some(m) = main {
            pick.in_s = round2(m.in_s);
            pick.out_s = round2(m.out_s);
        }
        out.push(CheckedClip { pick, script, notes, issues, script_id: None });
    }
    Ok(out)
}

/// One clip as the set remembers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipItem {
    pub script_id: i64,
    pub title: String,
    pub hook: String,
    pub quote: String,
    pub why: String,
    pub score: f64,
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    pub visual: Visual,
    /// `new`, `accepted` or `rejected`.
    pub status: String,
    /// Warnings the checks find. Recomputed from the index by [`with_live_checks`] whenever the
    /// set is shown; the stored count is only what the run saw.
    #[serde(default)]
    pub warnings: usize,
    /// The checks' issues (with their fixes) as of now. Filled by [`with_live_checks`], never
    /// stored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<Issue>,
    /// Keyframe near the clip's start, for the grid. Filled by [`with_thumbs`], never stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumb: Option<String>,
    /// For a clip that crops its footage (9:16, 1:1): what its planned framing shows of `thumb`,
    /// so the grid shows the clip as it will look. Filled by [`with_thumbs`], never stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumb_view: Option<crate::framing::FrameView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSet {
    pub id: String,
    pub project_id: i64,
    pub brief: String,
    pub created_at: i64,
    pub options: Value,
    pub items: Vec<ClipItem>,
    pub log: Value,
}

fn new_set_id() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("cs{:x}", nanos & 0xffff_ffff_ffff)
}

/// Check and save picks as a new clip set. Clips with errors are skipped (and reported).
pub fn save_clips(
    db: &Db,
    project: &Project,
    cfg: &ScriptConfig,
    picks: &[Pick],
    opts: &ClipOptions,
    agents: Option<&AgentSelection>,
    log: Value,
) -> Result<(ClipSet, Vec<CheckedClip>), Error> {
    let mut checked = check_clips(db, project, cfg, picks, opts)?;
    let set_id = new_set_id();
    let created_at = crate::projects::now();
    let mut items = Vec::new();
    for c in checked.iter_mut() {
        if c.issues.iter().any(|i| i.severity == IssueSeverity::Error) {
            continue;
        }
        // The agents that made it keep working on it.
        c.script.agents = agents.cloned();
        let id = crate::script::save_version(db, project.id, &c.script, None)?;
        db.conn.execute("UPDATE scripts SET set_id = ?1, kind = 'clip' WHERE id = ?2", params![set_id, id])?;
        c.script_id = Some(id);
        items.push(ClipItem {
            script_id: id,
            title: c.pick.title.clone(),
            hook: c.script.beats[0].on_screen_text.clone().unwrap_or_default(),
            quote: c.pick.quote.clone(),
            why: c.pick.why.clone(),
            score: c.pick.score,
            video_id: c.pick.video_id,
            in_s: c.pick.in_s,
            out_s: c.pick.out_s,
            visual: c.pick.visual.clone(),
            status: "new".into(),
            warnings: c.issues.iter().filter(|i| i.severity == IssueSeverity::Warning).count(),
            issues: Vec::new(),
            thumb: None,
            thumb_view: None,
        });
    }
    if items.is_empty() {
        return Err(Error::Invalid("no clip passed the checks; nothing saved".into()));
    }
    let options = serde_json::to_value(opts).unwrap_or(Value::Null);
    db.conn.execute(
        "INSERT INTO clip_sets(id, project_id, brief, created_at, options_json, items_json, log_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            set_id,
            project.id,
            opts.brief,
            created_at,
            options.to_string(),
            serde_json::to_string(&items).unwrap_or_else(|_| "[]".into()),
            log.to_string()
        ],
    )?;
    Ok((
        ClipSet { id: set_id, project_id: project.id, brief: opts.brief.clone(), created_at, options, items, log },
        checked,
    ))
}

fn row_to_set(r: &rusqlite::Row) -> rusqlite::Result<ClipSet> {
    let options: String = r.get(4)?;
    let items: String = r.get(5)?;
    let log: String = r.get(6)?;
    Ok(ClipSet {
        id: r.get(0)?,
        project_id: r.get(1)?,
        brief: r.get(2)?,
        created_at: r.get(3)?,
        options: serde_json::from_str(&options).unwrap_or(Value::Null),
        items: serde_json::from_str(&items).unwrap_or_default(),
        log: serde_json::from_str(&log).unwrap_or(Value::Null),
    })
}

/// The newest live version, among the set's own scripts, of the clip `script_id` is a version
/// of (an edit in the editor saves a new one). `None` when every version of it is in the trash.
fn latest_live_version(db: &Db, set_id: &str, script_id: i64) -> Option<i64> {
    db.conn
        .query_row(
            "SELECT s2.id FROM scripts s1 JOIN scripts s2
               ON s2.project_id = s1.project_id AND s2.title = s1.title AND s2.set_id = ?2 AND s2.trash_id IS NULL
             WHERE s1.id = ?1 ORDER BY s2.version DESC LIMIT 1",
            params![script_id, set_id],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten()
}

/// What the set shows: each item points at the newest version of its clip, and an item whose
/// clip is in the trash is left out. The stored list keeps every item (see [`read_set`]), so
/// restoring a trashed clip brings its place in the grid back with it.
fn refresh_items(db: &Db, set: &mut ClipSet) {
    let set_id = set.id.clone();
    set.items.retain_mut(|item| match latest_live_version(db, &set_id, item.script_id) {
        Some(id) => {
            item.script_id = id;
            true
        }
        None => false,
    });
}

pub fn list_sets(db: &Db, project_id: i64) -> Result<Vec<ClipSet>, Error> {
    let mut sets: Vec<ClipSet> = db
        .conn
        .prepare(
            "SELECT id, project_id, brief, created_at, options_json, items_json, log_json FROM clip_sets
             WHERE project_id = ?1 AND trash_id IS NULL ORDER BY created_at DESC, id DESC",
        )?
        .query_map([project_id], row_to_set)?
        .collect::<Result<_, _>>()?;
    for s in &mut sets {
        refresh_items(db, s);
    }
    Ok(sets)
}

/// The set as stored, items of trashed clips included. For writing back; show [`load_set`].
fn read_set(db: &Db, set_id: &str) -> Result<ClipSet, Error> {
    db.conn
        .query_row(
            "SELECT id, project_id, brief, created_at, options_json, items_json, log_json FROM clip_sets
             WHERE id = ?1 AND trash_id IS NULL",
            [set_id],
            row_to_set,
        )
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("clip set {set_id}")))
}

pub fn load_set(db: &Db, set_id: &str) -> Result<ClipSet, Error> {
    let mut set = read_set(db, set_id)?;
    refresh_items(db, &mut set);
    Ok(set)
}

/// The length range a clip run asked for, for a clip (any version) of that run.
pub fn length_range(db: &Db, project_id: i64, title: &str) -> Option<(f64, f64)> {
    let options: String = db
        .conn
        .query_row(
            "SELECT c.options_json FROM scripts s JOIN clip_sets c ON c.id = s.set_id
              WHERE s.project_id = ?1 AND s.title = ?2 AND s.trash_id IS NULL AND c.trash_id IS NULL
              ORDER BY s.version DESC LIMIT 1",
            params![project_id, title],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten()?;
    let v: Value = serde_json::from_str(&options).ok()?;
    Some((v["min_s"].as_f64()?, v["max_s"].as_f64()?))
}

/// Check each clip of a set against the index as it is now: its issues (with fixes), its warning
/// count, and the bounds its newest version plays. What the run saw at creation goes stale as
/// soon as faces or speakers are indexed, or the clip is edited.
pub fn with_live_checks(db: &Db, cfg: &ScriptConfig, set: &mut ClipSet) {
    let min_s = set.options["min_s"].as_f64();
    for item in &mut set.items {
        let Ok(stored) = crate::script::load(db, item.script_id) else { continue };
        if let Some(c) = stored.script.beats.iter().flat_map(|b| &b.clips).find(|c| c.audio == Audio::Source) {
            item.in_s = round2(c.in_s);
            item.out_s = round2(c.out_s);
        }
        let opts = crate::script_check::CheckOptions { trim_min_s: min_s, forbid_overlap: true, ..Default::default() };
        if let Ok(issues) = crate::script_check::check(db, set.project_id, &stored.script, cfg, &opts) {
            item.warnings = issues.iter().filter(|i| i.severity == IssueSeverity::Warning).count();
            item.issues = issues;
        }
    }
}

/// Fill each item's `thumb` with the keyframe nearest a couple of seconds into the clip and, when
/// the clip crops its footage, `thumb_view` with the clip's own framing plan at that keyframe (the
/// plan the player and the render use). Planned without ffmpeg's black-bar detection, so listing
/// sets stays quick; the crop follows the faces either way.
pub fn with_thumbs(db: &Db, data_dir: &std::path::Path, set: &mut ClipSet) {
    for item in &mut set.items {
        let t = item.in_s + 2.0;
        let frame = db
            .conn
            .query_row(
                "SELECT thumb_path, t_s FROM frames WHERE video_id = ?1 AND thumb_path IS NOT NULL
                 ORDER BY ABS(t_s - ?2) LIMIT 1",
                params![item.video_id, t],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
            )
            .ok();
        item.thumb_view = frame.as_ref().and_then(|(_, t_s)| thumb_view(db, item.script_id, item.video_id, *t_s));
        item.thumb = frame.map(|(rel, _)| data_dir.join(rel).to_string_lossy().into_owned());
    }
}

/// What a clip's framing shows at `t_s` (source seconds of `video_id`): its first framed clip of
/// that video, at that moment clamped into the clip.
fn thumb_view(db: &Db, script_id: i64, video_id: i64, t_s: f64) -> Option<crate::framing::FrameView> {
    let stored = crate::script::load(db, script_id).ok()?;
    let plans = crate::framing::plans_for_script(db, None, &stored.script).ok()?;
    let f = plans.into_iter().find(|f| f.video_id == video_id)?;
    Some(f.plan.view_at((t_s - f.in_s).max(0.0)))
}

/// Accept or reject one clip of a set (`new`, `accepted`, `rejected`).
pub fn set_item_status(db: &Db, set_id: &str, script_id: i64, status: &str) -> Result<ClipSet, Error> {
    if !matches!(status, "new" | "accepted" | "rejected") {
        return Err(Error::Invalid(format!("unknown clip status '{status}'")));
    }
    // Change that one entry of the stored list; the items of trashed clips stay in it. The caller
    // names the clip by the version the set shows, which may be newer than the one stored.
    let mut stored = read_set(db, set_id)?;
    let item = stored
        .items
        .iter_mut()
        .find(|i| i.script_id == script_id || latest_live_version(db, set_id, i.script_id) == Some(script_id))
        .ok_or_else(|| Error::NotFound(format!("clip script #{script_id} in set {set_id}")))?;
    item.status = status.to_string();
    db.conn.execute(
        "UPDATE clip_sets SET items_json = ?1 WHERE id = ?2",
        params![serde_json::to_string(&stored.items).unwrap_or_else(|_| "[]".into()), set_id],
    )?;
    load_set(db, set_id)
}

/// Delete a clip set and its scripts: they go to the trash together (`trash.rs`), restorable for
/// `trash::KEEP_DAYS` days, and the delete is recorded as a run from `source`.
pub fn delete_set(db: &Db, set_id: &str, source: &str) -> Result<crate::trash::TrashItem, Error> {
    crate::trash::trash_clip_set(db, set_id, source)
}

// ---------------------------------------------------------------------------------------------
// The run

/// The agent that makes the final choice, resolved.
pub struct Orchestrator {
    pub agent: AgentSpec,
    pub setup: crate::runtime::VisionSetup,
    /// The agent's optional cap on rounds of tool calls (0 = none, the default).
    pub max_tool_rounds: u32,
}

/// Who does the model work: the worker agents as a pool, and the orchestrator.
pub struct Backends {
    pub workers: AgentPool,
    /// `None` = code picks (worker-only run, or no orchestrator available).
    pub orchestrator: Option<Orchestrator>,
    /// Why the orchestrator chosen could not be used, when it could not.
    pub orchestrator_skipped: Option<String>,
    pub orchestrator_timeout_s: u64,
    /// The selection as resolved (run → app), saved on every clip the run makes.
    pub selection: ResolvedSelection,
    /// How workers and the orchestrator reach the tool registry. `None`: they answer from the
    /// prompt alone (a run with no data directory behind it).
    pub tools: Option<crate::tools::ToolAccess>,
}

impl Backends {
    /// Resolve the run's agents: its own choice over the app's.
    pub async fn resolve(paths: &crate::paths::Paths, config: &Config, opts: &ClipOptions) -> Result<Self, Error> {
        Self::resolve_for(paths, config, Some(&opts.agents), None, opts.worker_only).await
    }

    /// Resolve with a cut's choice between the run's and the app's.
    pub async fn resolve_for(
        paths: &crate::paths::Paths,
        config: &Config,
        run: Option<&AgentSelection>,
        cut: Option<&AgentSelection>,
        worker_only: bool,
    ) -> Result<Self, Error> {
        let selection = config.resolve_agents(run, cut)?;
        if selection.workers.is_empty() {
            return Err(crate::agents::SelectionError::NoAgent { role: AgentRole::Worker }.into());
        }
        let worker_timeout = config.script.server_timeout_s.min(900);
        let workers = AgentPool::resolve(
            paths,
            config,
            AgentRole::Worker,
            &selection.workers,
            selection.max_parallel,
            worker_timeout,
        )
        .await?;
        let (mut orchestrator, mut orchestrator_skipped, mut orchestrator_timeout_s) =
            (None, None, config.script.server_timeout_s.min(1800));
        if let (false, Some(agent)) = (worker_only, &selection.orchestrator) {
            let llm = agent.llm_config(config, AgentRole::Orchestrator);
            orchestrator_timeout_s = llm.cli.timeout_secs.max(orchestrator_timeout_s);
            match crate::runtime::resolve_agent(paths, config, agent, AgentRole::Orchestrator).await {
                crate::runtime::VisionSetup::Unavailable(why) => orchestrator_skipped = Some(format!("{}: {why}", agent.id)),
                setup => {
                    orchestrator = Some(Orchestrator { agent: agent.clone(), setup, max_tool_rounds: llm.max_tool_rounds })
                }
            }
        }
        let tools = Some(crate::tools::ToolAccess::new(paths.clone(), Arc::new(config.clone()), 0, None));
        Ok(Self { workers, orchestrator, orchestrator_skipped, orchestrator_timeout_s, selection, tools })
    }

    /// "workers: claude ×2 · ghostpen ×2 (up to 4 at once); orchestrator: claude (cli claude)".
    pub fn describe(&self) -> String {
        let orch = match &self.orchestrator {
            Some(o) => format!("{} ({})", o.agent.id, o.setup.describe()),
            None => match &self.orchestrator_skipped {
                Some(why) => format!("code (orchestrator unavailable: {why})"),
                None => "code (worker-only)".into(),
            },
        };
        let mut line = format!("workers: {}; orchestrator: {orch}", self.workers.describe());
        for s in &self.workers.skipped {
            line.push_str(&format!("; skipped {}: {}", s.id, s.reason));
        }
        line
    }
}

/// What a clip run would run into, checked before it is queued: the agents it would use
/// (installed, reachable, model known, not resting) and whether the project is indexed far
/// enough. The Clips form shows it before Start; [`run`] repeats the project part itself.
/// Opens the database only after the agent checks, so the future holds no connection.
pub async fn preflight(
    paths: &crate::paths::Paths,
    config: &Config,
    project_id: i64,
    opts: &ClipOptions,
) -> Result<crate::preflight::Preflight, Error> {
    let mut checks = Vec::new();
    match config.resolve_agents(Some(&opts.agents), None) {
        Ok(sel) => {
            let mut needed = vec![(AgentRole::Worker, sel.workers.clone())];
            if let (false, Some(o)) = (opts.worker_only, &sel.orchestrator) {
                needed.push((AgentRole::Orchestrator, vec![o.clone()]));
            }
            checks.extend(crate::agents::agent_checks(paths, config, &needed, &[AgentRole::Orchestrator]).await);
        }
        Err(e) => checks.push(crate::preflight::Check::new(
            crate::preflight::Level::Error,
            crate::preflight::Scope::Agent,
            "selection",
            e.to_string(),
        )),
    }
    let db = Db::open(&paths.db_file())?;
    checks.extend(crate::preflight::project_readiness(
        &db,
        project_id,
        &opts.video_ids,
        crate::preflight::Needs { transcripts: true, analysis: true },
    )?);
    Ok(crate::preflight::Preflight { checks })
}

/// A run's outcome and log.
#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub set: ClipSet,
    pub clips: Vec<CheckedClip>,
    pub log: Value,
}

impl RunResult {
    /// The saved clip set's id, for whoever reports the run (a `runs::RunLink::ClipSet`).
    pub fn set_id(&self) -> &str {
        &self.set.id
    }

    /// One line on how the run went: "41 candidates · 7 windows · 12 picks · orchestrator model · 93 s".
    pub fn stats_line(&self) -> String {
        let l = &self.log;
        let share = l["agents_line"].as_str().unwrap_or_default();
        format!(
            "{} candidates · {} windows{} · {} picks · orchestrator {} · {:.0} s",
            l["candidates"],
            l["worker_jobs"],
            if share.is_empty() { String::new() } else { format!(" ({share})") },
            l["picks_merged"],
            if l["orchestrator"]["used"].as_bool().unwrap_or(false) { "model" } else { "code" },
            l["timings_s"]["total"].as_f64().unwrap_or(0.0)
        )
    }
}

/// Where a run reports: `(fraction 0–1, phase, note)` progress and the events worth keeping
/// (preflight warnings, agents rerouted or resting). A plain closure takes the progress only.
pub type OnProgress<'a> = &'a mut dyn crate::runs::RunSink;

fn cancelled(cancel: &Option<Arc<AtomicBool>>) -> Result<(), Error> {
    if cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
        return Err(Error::Invalid(crate::chat::CANCELLED.into()));
    }
    Ok(())
}

/// Worker jobs for every video: one per `window_s` of timeline, with the candidates that start in it.
struct Job {
    survey_idx: usize,
    from_s: f64,
    to_s: f64,
    cand_ids: Vec<String>,
    want: usize,
}

fn plan_jobs(surveys: &[VideoSurvey], cands: &[Vec<Candidate>], opts: &ClipOptions) -> Vec<Job> {
    let total_s: f64 = surveys.iter().map(|s| s.units.last().map(|u| u.end_s).unwrap_or(0.0)).sum::<f64>().max(1.0);
    let mut jobs = Vec::new();
    for (si, survey) in surveys.iter().enumerate() {
        let end = survey.units.last().map(|u| u.end_s).unwrap_or(0.0);
        let mut from = 0.0;
        while from < end {
            let to = (from + opts.window_s).min(end);
            let in_window: Vec<Candidate> =
                cands[si].iter().filter(|c| c.in_s >= from && c.in_s < to).cloned().collect();
            if !in_window.is_empty() {
                let mut chosen = dedupe(in_window, 0.5, MAX_CANDIDATES_PER_JOB);
                chosen.sort_by(|a, b| a.in_s.total_cmp(&b.in_s));
                // Ask for more than this window's share of the set so the orchestrator has a choice.
                let share = (to - from) / total_s * opts.count as f64 * 2.0;
                let want = (share.ceil() as usize).clamp(2, 5);
                jobs.push(Job {
                    survey_idx: si,
                    from_s: from,
                    to_s: to,
                    cand_ids: chosen.into_iter().map(|c| c.id).collect(),
                    want,
                });
            }
            from = to;
        }
    }
    jobs
}

/// The whole run: candidates → workers → orchestrator → check → save.
pub async fn run(
    db: &mut Db,
    project_id: i64,
    cfg: &ScriptConfig,
    backends: &Backends,
    opts: &ClipOptions,
    cancel: Option<Arc<AtomicBool>>,
    on_progress: OnProgress<'_>,
) -> Result<RunResult, Error> {
    opts.validate()?;
    let started = Instant::now();
    let project = db.project(project_id)?;
    let glossary = project.glossary.prompt_section();

    // 1. Candidates.
    // 0. Preflight: is the project indexed far enough, and who could not be used.
    on_progress.progress(0.01, "preflight", "Checking the project and the agents");
    let readiness = crate::preflight::project_readiness(
        db,
        project_id,
        &opts.video_ids,
        crate::preflight::Needs { transcripts: true, analysis: true },
    )?;
    for c in &readiness {
        if let Some(ev) = crate::runs::StepEvent::from_check(c) {
            on_progress.event(ev);
        }
    }
    crate::preflight::Preflight { checks: readiness }.into_result()?;
    for s in &backends.workers.skipped {
        on_progress.event(crate::runs::StepEvent::warning(format!("worker {} is not used: {}", s.id, s.reason)));
    }
    if let Some(why) = &backends.orchestrator_skipped {
        on_progress.event(crate::runs::StepEvent::warning(format!("orchestrator not used, code picks instead: {why}")));
    }
    cancelled(&cancel)?;

    on_progress.progress(0.02, "candidates", "Reading transcripts");
    let t0 = Instant::now();
    let video_ids = project_videos(db, project_id, &opts.video_ids)?;
    let mut surveys = Vec::new();
    let mut cands: Vec<Vec<Candidate>> = Vec::new();
    for vid in &video_ids {
        let s = VideoSurvey::load(db, *vid)?;
        if s.units.is_empty() {
            continue;
        }
        cands.push(candidates(&s, cfg, opts));
        surveys.push(s);
    }
    let n_cands: usize = cands.iter().map(Vec::len).sum();
    if n_cands == 0 {
        return Err(Error::Invalid(format!(
            "no {:.0}–{:.0} s sentence-bounded stretches of speech in the selected videos (are they transcribed?)",
            opts.min_s, opts.max_s
        )));
    }
    let candidates_s = t0.elapsed().as_secs_f64();
    cancelled(&cancel)?;

    // 2. Workers. They get the whole tool registry for this project, as the script chat does.
    let t1 = Instant::now();
    let access = backends
        .tools
        .as_ref()
        .map(|t| Arc::new(crate::tools::ToolAccess { project_id, cancel: cancel.clone(), ..t.clone() }));
    let plans = plan_jobs(&surveys, &cands, opts);
    // Each job notes its own tool calls, for the run log.
    let (job_access, trails): (Vec<_>, Vec<_>) = plans
        .iter()
        .map(|_| match &access {
            Some(a) => {
                let (a, t) = a.with_trail();
                (Some(Arc::new(a)), Some(t))
            }
            None => (None, None),
        })
        .unzip();
    let jobs: Vec<AgentJob> = plans
        .iter()
        .zip(job_access)
        .map(|(j, job_access)| {
            let s = &surveys[j.survey_idx];
            let cs: Vec<&Candidate> = cands[j.survey_idx].iter().filter(|c| j.cand_ids.contains(&c.id)).collect();
            AgentJob {
                label: format!("v{} {:.0}–{:.0} s", s.video_id, j.from_s, j.to_s),
                prompt: worker_prompt(
                    s,
                    j.from_s,
                    (j.to_s + opts.max_s).min(s.duration_s.max(j.to_s)),
                    &cs,
                    opts,
                    glossary.as_deref(),
                    j.want,
                ),
                schema: worker_schema(),
                max_tokens: 2048,
                tools: job_access,
            }
        })
        .collect();
    let n_jobs = jobs.len();
    on_progress.progress(0.08, "workers", &format!("Scoring {n_jobs} windows with {}", backends.workers.describe()));
    let results = {
        let mut cb = |p: &PoolProgress| {
            // A rest, a drop or a busy pause: kept in the step, not just its passing note.
            if let Some(ev) = p.event {
                let level =
                    if matches!(ev.action, crate::agents::PoolAction::Retrying { .. }) { crate::runs::EventLevel::Info } else { crate::runs::EventLevel::Warning };
                on_progress.event(crate::runs::StepEvent::new(level, ev.message()).with_error(ev.error.clone()));
                return;
            }
            let note = match p.finished {
                Some(r) => match &r.value {
                    Ok(_) => format!("Scored {}/{} windows ({} on {}: {:.0} s)", p.done, p.total, r.label, r.agent, r.seconds),
                    Err(e) => format!("Window {} failed: {}", r.label, shorten(&e.to_string(), 160)),
                },
                None => format!("Scored {}/{} windows", p.done, p.total),
            };
            // Which agents are busy right now: "claude 2 · ghostpen 2 running".
            let running = p.running_line();
            let note = if running.is_empty() { note } else { format!("{note} · {running}") };
            on_progress.progress(0.08 + 0.62 * p.done as f64 / p.total.max(1) as f64, "workers", &note);
        };
        backends.workers.run_jobs(jobs, cancel.clone(), &mut cb).await?
    };
    let loads = AgentPool::loads_of(&results);
    let mut picks = Vec::new();
    let mut worker_log = Vec::new();
    for r in &results {
        let plan = &plans[r.index];
        let s = &surveys[plan.survey_idx];
        let cs: Vec<&Candidate> = cands[plan.survey_idx].iter().filter(|c| plan.cand_ids.contains(&c.id)).collect();
        match &r.value {
            Ok(v) => {
                let got = parse_worker_picks(v, s, &cs, opts, cfg);
                worker_log.push(json!({ "window": r.label, "agent": r.agent, "attempts": r.attempts, "seconds": round2(r.seconds), "candidates": cs.len(), "picks": got.len(), "tool_calls": trail_lines(&trails[r.index]) }));
                picks.extend(got);
            }
            Err(e) => worker_log.push(json!({ "window": r.label, "agent": r.agent, "attempts": r.attempts, "seconds": round2(r.seconds), "error": e, "tool_calls": trail_lines(&trails[r.index]) })),
        }
    }
    let workers_s = t1.elapsed().as_secs_f64();
    let failed = results.iter().filter(|r| r.value.is_err()).count();
    if picks.is_empty() {
        // Every window failed: the first failure, typed, says why.
        if failed == n_jobs
            && let Some(mut e) = results.iter().find_map(|r| r.value.as_ref().err().cloned())
        {
            e.message = format!("{} (all {n_jobs} windows failed)", e.message);
            return Err(Error::Agent(e));
        }
        return Err(Error::Invalid(format!(
            "the workers found nothing worth clipping ({failed}/{n_jobs} windows failed, the rest returned no usable picks)"
        )));
    }
    cancelled(&cancel)?;

    // 3. Orchestrator.
    let t2 = Instant::now();
    let merged = merge_picks(picks);
    let pool = orchestrator_pool(&merged, opts);
    let mut orchestrator_log = json!({ "used": false });
    let chosen = match &backends.orchestrator {
        Some(Orchestrator { agent, setup, max_tool_rounds }) if pool.len() > 1 => {
            on_progress.progress(
                0.72,
                "orchestrator",
                &format!("Choosing {} of {} with {} ({})", opts.count, pool.len(), agent.id, setup.describe()),
            );
            let prompt = orchestrator_prompt(&pool, opts, glossary.as_deref());
            // Its endpoint's slot, shared with every other feature using that model.
            let _slot = crate::agents::acquire_for(setup, agent.capacity()).await;
            // Busy: asked again after a pause; quota gone: the endpoint rests for every feature.
            let key = crate::agents::EndpointKey::for_setup(setup);
            let timeout_s = backends.orchestrator_timeout_s;
            let (orchestrator_access, orchestrator_trail) = match access.as_ref().map(|a| a.with_max_rounds(*max_tool_rounds).with_trail()) {
                Some((a, t)) => (Some(a), Some(t)),
                None => (None, None),
            };
            let answer = crate::agents::call_once(&agent.id, key.as_ref(), || {
                let prompt = &prompt;
                let access = orchestrator_access.as_ref();
                async move {
                    let mut backend = crate::chat::ChatBackend::from_vision_setup(setup, timeout_s).await?;
                    let v =
                        crate::tools::run_agent_json(&mut backend, prompt, &orchestrator_schema(), 4096, timeout_s, access)
                            .await;
                    if let crate::chat::ChatBackend::Local { helper, .. } = &mut backend {
                        helper.kill().await;
                    }
                    v
                }
            })
            .await;
            match answer {
                Ok(v) => {
                    orchestrator_log = json!({ "used": true, "agent": agent.id, "backend": setup.describe(), "answered": v["clips"].as_array().map(|a| a.len()), "tool_calls": trail_lines(&orchestrator_trail) });
                    apply_orchestrator(&pool, &v, opts.count, opts.format)
                }
                Err(e) => {
                    orchestrator_log = json!({ "used": false, "agent": agent.id, "backend": setup.describe(), "error": e.to_string(), "error_detail": e.agent_error(), "tool_calls": trail_lines(&orchestrator_trail) });
                    let ev = crate::runs::StepEvent::warning(format!("orchestrator failed, code picks instead: {e}"));
                    on_progress.event(match e.agent_error() {
                        Some(a) => ev.with_error(a.clone()),
                        None => ev,
                    });
                    on_progress.progress(
                        0.85,
                        "orchestrator",
                        &format!("Orchestrator failed, code picks instead: {}", shorten(&e.to_string(), 120)),
                    );
                    select(&pool, opts.count, opts.format)
                }
            }
        }
        _ => select(&pool, opts.count, opts.format),
    };
    let orchestrator_s = t2.elapsed().as_secs_f64();
    cancelled(&cancel)?;

    // 4. Check + save.
    on_progress.progress(0.9, "save", "Checking and saving clips");
    let t3 = Instant::now();
    let mut log = json!({
        "videos": video_ids,
        "candidates": n_cands,
        "worker_jobs": n_jobs,
        "worker_failures": failed,
        "worker_backend": backends.workers.describe(),
        "worker_concurrency": backends.workers.max_parallel(),
        "agents": loads,
        "agents_line": crate::agents::share_line(&loads),
        "agents_skipped": backends.workers.skipped,
        "orchestrator_skipped": backends.orchestrator_skipped,
        "selection": backends.selection.effective(),
        "picks_merged": merged.len(),
        "orchestrator": orchestrator_log,
        "workers": worker_log,
        "timings_s": { "candidates": round2(candidates_s), "workers": round2(workers_s), "orchestrator": round2(orchestrator_s) },
    });
    let effective = backends.selection.effective();
    let (mut set, clips) = save_clips(db, &project, cfg, &chosen, opts, Some(&effective), log.clone())?;
    let save_s = t3.elapsed().as_secs_f64();
    log["timings_s"]["save"] = json!(round2(save_s));
    log["timings_s"]["total"] = json!(round2(started.elapsed().as_secs_f64()));
    log["saved"] = json!(set.items.len());
    db.conn.execute("UPDATE clip_sets SET log_json = ?1 WHERE id = ?2", params![log.to_string(), set.id])?;
    set.log = log.clone();
    let n = set.items.len();
    on_progress.progress(1.0, "done", &format!("Saved {n} {}", if n == 1 { "clip" } else { "clips" }));
    Ok(RunResult { set, clips, log })
}

/// Which clips of a set to export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportWhich {
    Accepted,
    /// Everything not rejected.
    NotRejected,
    All,
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').chars().take(48).collect()
}

/// Render each chosen clip of a set to `dir` as `NN-title.mp4` with its hook title and speech
/// captions burnt in, plus a sidecar `.srt`. Returns the MP4 paths.
#[allow(clippy::too_many_arguments)]
pub fn export_set(
    db: &Db,
    data_dir: &std::path::Path,
    ffmpeg: &std::path::Path,
    set_id: &str,
    dir: &std::path::Path,
    which: ExportWhich,
    cfg: &ScriptConfig,
    cancel: Option<Arc<AtomicBool>>,
    mut on_progress: impl FnMut(usize, usize, &str),
) -> Result<Vec<std::path::PathBuf>, Error> {
    let set = load_set(db, set_id)?;
    let items: Vec<&ClipItem> = set
        .items
        .iter()
        .filter(|i| match which {
            ExportWhich::Accepted => i.status == "accepted",
            ExportWhich::NotRejected => i.status != "rejected",
            ExportWhich::All => true,
        })
        .collect();
    if items.is_empty() {
        return Err(Error::Invalid("no clips to export (accept some first, or export all)".into()));
    }
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    let mut out = Vec::new();
    for (k, item) in items.iter().enumerate() {
        cancelled(&cancel)?;
        let name = if item.title.is_empty() { item.hook.clone() } else { item.title.clone() };
        let path = dir.join(format!("{:02}-{}.mp4", k + 1, slug(&name)));
        on_progress(k, items.len(), &name);
        let opts = crate::preview::PreviewOptions {
            burn_titles: true,
            burn_narration: false,
            burn_captions: true,
            normalize_audio: true,
            audio_fade_s: cfg.audio_fade_s,
            speech_overrun_s: cfg.speech_overrun_s,
            out: Some(path.clone()),
            cancel: cancel.clone(),
            size: crate::preview::RenderSize::Full,
        };
        crate::preview::render_preview(db, data_dir, ffmpeg, item.script_id, &opts, |_, _| {})?;
        out.push(path);
    }
    on_progress(items.len(), items.len(), "done");
    Ok(out)
}

/// Video file name for display.
pub fn video_name(db: &Db, video_id: i64) -> String {
    db.conn
        .query_row("SELECT path FROM video_files WHERE video_id = ?1 LIMIT 1", [video_id], |r| r.get::<_, String>(0))
        .ok()
        .and_then(|p| std::path::Path::new(&p).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| format!("video #{video_id}"))
}

/// Count of picks overlapping any of `others` (same video, any overlap) — for comparing a run
/// against a hand-picked baseline.
pub fn overlap_count(picks: &[(i64, f64, f64)], others: &[(i64, f64, f64)]) -> usize {
    picks.iter().filter(|p| others.iter().any(|o| o.0 == p.0 && overlap((o.1, o.2), (p.1, p.2)) > 0.0)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(a: f64, b: f64, t: &str) -> Segment {
        Segment { start_s: a, end_s: b, text: t.into(), off_mic: false, speaker: None }
    }

    fn cfg() -> ScriptConfig {
        ScriptConfig { max_shake_jerk: 0.0, ..ScriptConfig::default() }
    }

    fn opts(min: f64, max: f64) -> ClipOptions {
        ClipOptions { min_s: min, max_s: max, count: 3, ..ClipOptions::default() }
    }

    #[test]
    fn units_end_on_punctuation_and_never_on_a_dangling_word() {
        let segs = vec![
            seg(0.0, 4.0, "Hola a todos."),
            seg(4.0, 9.0, "hoy hablamos de agentes y entonces aníbal creó un"),
            seg(9.0, 14.0, "sistema que corrige solo"),
            seg(14.0, 18.0, "Y funciona."),
        ];
        let u = sentence_units(&segs);
        // "creó un" never closes a unit; "corrige solo" closes on the capital that follows.
        assert_eq!(u.len(), 3);
        assert!(u[0].strong_end);
        assert_eq!((u[1].start_s, u[1].end_s), (4.0, 14.0));
        assert!(!u[1].strong_end);
        assert!(u[2].strong_end);
    }

    #[test]
    fn a_long_run_on_chain_is_split_at_a_weak_edge() {
        let segs: Vec<Segment> = (0..8)
            .map(|i| {
                seg(
                    i as f64 * 8.0,
                    i as f64 * 8.0 + 8.0,
                    if i % 2 == 0 { "seguimos hablando de esto y de" } else { "cosas que pasan siempre bien" },
                )
            })
            .collect();
        let u = sentence_units(&segs);
        assert!(u.len() >= 2, "{u:?}");
        assert!(u.iter().all(|u| u.end_s - u.start_s <= LONG_UNIT_S + 8.0 + 0.01));
        // Never closed on "de".
        assert!(u.iter().all(|u| !u.text.ends_with(" de") || u.end_s == 64.0));
    }

    #[test]
    fn a_hole_in_the_transcript_breaks_units_and_windows() {
        let segs = vec![
            seg(0.0, 10.0, "Primera parte larga de la charla."),
            seg(30.0, 40.0, "Otra cosa distinta aquí dicha."),
        ];
        let survey = VideoSurvey::from_units(1, sentence_units(&segs));
        let c = candidates(&survey, &cfg(), &opts(15.0, 60.0));
        assert!(c.is_empty(), "a window may not span 20 s of missing speech: {c:?}");
    }

    fn talk(n: usize, each: f64) -> Vec<Segment> {
        (0..n)
            .map(|i| {
                seg(
                    i as f64 * each,
                    (i + 1) as f64 * each,
                    &format!("Esta es la frase número {i} con varias palabras dentro."),
                )
            })
            .collect()
    }

    /// A face every 0.2 s over `a..b`.
    fn faces(a: f64, b: f64, cx: f64, cy: f64, h: f64) -> Vec<crate::scenes::FaceObs> {
        let n = ((b - a) * 5.0).round() as usize;
        (0..n).map(|i| crate::scenes::FaceObs { t: a + i as f64 * 0.2, cx, cy, w: h * 0.6, h }).collect()
    }

    #[test]
    fn candidates_carry_the_scene_mix_and_vertical_runs_prefer_two_shots() {
        // 0–60 s two hosts on camera, 60–120 s a screen share with their tiles in the corner.
        let mut obs = faces(0.0, 60.0, 0.25, 0.42, 0.45);
        obs.extend(faces(0.0, 60.0, 0.75, 0.42, 0.45));
        obs.extend(faces(60.0, 120.0, 0.72, 0.88, 0.076));
        obs.extend(faces(60.0, 120.0, 0.9, 0.88, 0.07));
        let survey = VideoSurvey::from_units(1, sentence_units(&talk(24, 5.0))).with_faces(&obs);
        let two = survey.visual(10.0, 40.0);
        let share = survey.visual(70.0, 100.0);
        let half = survey.visual(45.0, 75.0);
        assert_eq!(two.layout.unwrap().two_shot, 1.0);
        assert_eq!(two.label, "faces");
        assert_eq!(share.layout.unwrap().screen_share, 1.0);
        assert_eq!(share.label, "screen");
        assert_eq!(half.label, "mixed");
        assert!(two.vertical_bonus() > half.vertical_bonus() && half.vertical_bonus() > share.vertical_bonus());
        assert!(two.summary().starts_with("two-shot 100%"), "{}", two.summary());
        // Ranked for a vertical run, the two-shot windows come first; the summary reaches prompts.
        let o = ClipOptions { format: ClipFormat::Vertical, ..opts(20.0, 30.0) };
        let mut c = candidates(&survey, &cfg(), &o);
        c.sort_by(|a, b| b.prescore.total_cmp(&a.prescore));
        assert!(c[0].out_s <= 60.0, "{:?}", c[0]);
        assert!(c.iter().any(|x| x.visual.summary().contains("screen share")));
        // Without indexed faces, the keyframe labels still work as before.
        let plain = VideoSurvey::from_units(1, sentence_units(&talk(24, 5.0)));
        assert!(plain.visual(10.0, 40.0).layout.is_none());
    }

    #[test]
    fn candidates_respect_length_off_mic_and_overlap() {
        let mut segs = talk(30, 5.0);
        segs[4].off_mic = true;
        let survey = VideoSurvey::from_units(7, sentence_units(&segs));
        let c = candidates(&survey, &cfg(), &opts(20.0, 40.0));
        assert!(!c.is_empty());
        for x in &c {
            assert!(x.duration_s() >= 20.0 - 1e-6 && x.duration_s() <= 40.0 + 1e-6, "{x:?}");
            assert!((x.in_s - 20.0).abs() > 0.01, "never starts on the off-mic line");
            assert_eq!(x.video_id, 7);
            assert!(x.id.starts_with("v7c"));
        }
        for (i, a) in c.iter().enumerate() {
            for b in c.iter().skip(i + 1) {
                assert!(
                    overlap((a.in_s, a.out_s), (b.in_s, b.out_s)) <= a.duration_s().min(b.duration_s()) * 0.5 + 1e-6
                );
            }
        }
    }

    fn pick(id: &str, v: i64, a: f64, b: f64, score: f64, label: &str) -> Pick {
        Pick {
            id: id.into(),
            video_id: v,
            in_s: a,
            out_s: b,
            title: format!("t{id}"),
            quote: String::new(),
            why: String::new(),
            score,
            hook: String::new(),
            candidate: None,
            visual: Visual { label: label.into(), ..Visual::default() },
            speakers: Vec::new(),
        }
    }

    #[test]
    fn merging_keeps_the_better_of_two_overlapping_picks() {
        let m = merge_picks(vec![
            pick("", 1, 0.0, 30.0, 6.0, "faces"),
            pick("", 1, 20.0, 50.0, 8.0, "faces"),
            pick("", 1, 50.0, 80.0, 5.0, "faces"),
            pick("", 2, 0.0, 30.0, 4.0, "faces"),
        ]);
        let ranges: Vec<(i64, f64)> = m.iter().map(|p| (p.video_id, p.in_s)).collect();
        assert_eq!(ranges, vec![(1, 20.0), (1, 50.0), (2, 0.0)]);
        assert_eq!(m[0].id, "p1");
    }

    #[test]
    fn vertical_prefers_faces_and_spreads_videos() {
        let pool = vec![
            pick("p1", 1, 0.0, 30.0, 8.0, "screen"),
            pick("p2", 1, 100.0, 130.0, 7.0, "faces"),
            pick("p3", 1, 200.0, 230.0, 7.5, "faces"),
            pick("p4", 2, 0.0, 30.0, 6.5, "faces"),
        ];
        let v = select(&pool, 2, ClipFormat::Vertical);
        let ids: Vec<&str> = v.iter().map(|p| p.id.as_str()).collect();
        // p3 is the best face shot; then video 2 beats a second clip from video 1.
        assert_eq!(ids, vec!["p3", "p4"]);
        let o = select(&pool, 1, ClipFormat::Original);
        assert_eq!(o[0].id, "p1");
    }

    #[test]
    fn a_cropped_set_leaves_screen_shares_out_when_it_can() {
        let mut merged: Vec<Pick> = (0..5).map(|i| pick(&format!("p{i}"), 1, i as f64 * 100.0, i as f64 * 100.0 + 30.0, 5.0, "faces")).collect();
        merged.push(pick("s", 1, 900.0, 930.0, 10.0, "screen"));
        let o = ClipOptions { count: 3, format: ClipFormat::Vertical, ..ClipOptions::default() };
        assert!(orchestrator_pool(&merged, &o).iter().all(|p| p.id != "s"));
        let o = ClipOptions { count: 4, ..o };
        assert!(orchestrator_pool(&merged, &o).iter().any(|p| p.id == "s"), "too few face shots: keep it");
        let o = ClipOptions { format: ClipFormat::Original, ..o };
        assert_eq!(orchestrator_pool(&merged, &o)[0].id, "s");
    }

    #[test]
    fn orchestrator_answer_is_applied_and_topped_up() {
        let pool = vec![
            pick("p1", 1, 0.0, 30.0, 8.0, "faces"),
            pick("p2", 1, 10.0, 40.0, 7.0, "faces"),
            pick("p3", 1, 100.0, 130.0, 6.0, "faces"),
            pick("p4", 1, 200.0, 230.0, 5.0, "faces"),
        ];
        let answer = json!({ "clips": [
            { "id": "p2", "title": "Corregite", "hook": "Corregí, corregí", "why": "w" },
            { "id": "p1", "title": "overlaps p2", "hook": "x", "why": "w" },
            { "id": "nope", "title": "?", "hook": "?", "why": "?" },
            { "id": "p2", "title": "dup", "hook": "dup", "why": "w" }
        ]});
        let out = apply_orchestrator(&pool, &answer, 3, ClipFormat::Original);
        let ids: Vec<&str> = out.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["p2", "p3", "p4"]);
        assert_eq!(out[0].hook, "Corregí, corregí");
        assert_eq!(out[0].title, "Corregite");
        assert_eq!(out[1].title, "tp3", "a topped-up pick keeps the worker's title");
    }

    #[test]
    fn worker_picks_snap_to_candidates_and_sentence_edges() {
        let segs = talk(20, 5.0);
        let survey = VideoSurvey::from_units(3, sentence_units(&segs));
        let o = opts(20.0, 40.0);
        let cands = candidates(&survey, &cfg(), &o);
        let refs: Vec<&Candidate> = cands.iter().collect();
        let c0 = &cands[0];
        let answer = json!({ "picks": [
            { "candidate": c0.id, "in_s": c0.in_s + 0.4, "out_s": c0.out_s - 0.3, "title": "A", "quote": "q", "why": "w", "score": 7.5, "hook": "h" },
            { "candidate": "", "in_s": 51.2, "out_s": 78.9, "title": "B", "quote": "", "why": "w", "score": 12, "hook": "h" },
            { "candidate": "", "in_s": 10.0, "out_s": 12.0, "title": "too short", "quote": "", "why": "", "score": 9, "hook": "" }
        ]});
        let picks = parse_worker_picks(&answer, &survey, &refs, &o, &cfg());
        assert_eq!(picks.len(), 2);
        assert_eq!((picks[0].in_s, picks[0].out_s), (c0.in_s, c0.out_s));
        assert_eq!((picks[1].in_s, picks[1].out_s), (50.0, 80.0));
        assert_eq!(picks[1].score, 10.0, "scores are clamped");
        assert!(!picks[1].quote.is_empty(), "a missing quote comes from the transcript");
    }

    #[test]
    fn formats_parse_and_size() {
        assert_eq!(ClipFormat::parse("9:16").unwrap().size(), Some((1080, 1920)));
        assert_eq!(ClipFormat::parse("square").unwrap().size(), Some((1080, 1080)));
        assert_eq!(ClipFormat::parse("").unwrap().size(), None);
        assert!(ClipFormat::parse("4:3").is_err());
        let o: ClipOptions = serde_json::from_value(json!({ "format": "9:16", "count": 5 })).unwrap();
        assert_eq!(o.format, ClipFormat::Vertical);
        assert_eq!(o.min_s, DEFAULT_MIN_S);
    }

    #[test]
    fn a_clip_script_has_its_hook_captions_and_size() {
        let mut p = pick("p1", 1, 10.0, 40.0, 7.0, "faces");
        p.hook = "El hook".into();
        p.title = "Título".into();
        let s = clip_script(&p, 0, 8, &ClipOptions { format: ClipFormat::Vertical, ..ClipOptions::default() });
        assert_eq!(s.title, "Clip 1/8 · Título");
        assert_eq!((s.width, s.height), (Some(1080), Some(1920)));
        assert_eq!(s.beats[0].on_screen_text.as_deref(), Some("El hook"));
        assert!(s.captions.is_some());
        assert_eq!(s.beats[0].clips[0].audio, Audio::Source);
    }

    fn library() -> (Db, i64, tempfile::TempDir) {
        populate(Db::open_in_memory().unwrap(), tempfile::tempdir().unwrap())
    }

    /// The same library in a database file, for jobs that open the database themselves.
    fn library_on_disk() -> (Db, i64, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        populate(Db::open(&crate::paths::Paths::in_dir(dir.path()).db_file()).unwrap(), dir)
    }

    fn populate(mut db: Db, dir: tempfile::TempDir) -> (Db, i64, tempfile::TempDir) {
        let project = db.create_project(&crate::projects::NewProject::named("clips")).unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        let path = dir.path().join("talk.mp4");
        std::fs::write(&path, b"x").unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h1', 1, 600.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, ?2, 1, 0, 0)",
                params![folder.id, path.to_str().unwrap()],
            )
            .unwrap();
        for i in 0..100 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, ?1, ?2, ?3)",
                    params![i as f64 * 5.0, i as f64 * 5.0 + 5.0, format!("Esta es la frase número {i} con varias palabras.")],
                )
                .unwrap();
        }
        (db, project.id, dir)
    }

    #[test]
    fn saved_clips_form_a_set_that_follows_new_versions() {
        let (db, pid, _dir) = library();
        let project = db.project(pid).unwrap();
        let o = ClipOptions { format: ClipFormat::Square, ..opts(20.0, 40.0) };
        let picks = vec![pick("p1", 1, 100.0, 130.0, 8.0, "faces"), pick("p2", 1, 200.0, 225.0, 6.0, "faces")];
        let (set, checked) = save_clips(&db, &project, &cfg(), &picks, &o, None, json!({})).unwrap();
        assert_eq!(set.items.len(), 2);
        assert!(checked.iter().all(|c| !c.issues.iter().any(|i| i.severity == IssueSeverity::Error)), "{checked:?}");
        let listed = crate::script::list(&db, pid).unwrap();
        assert!(listed.iter().all(|s| s.kind.as_deref() == Some("clip") && s.set_id.as_deref() == Some(set.id.as_str())));
        let stored = crate::script::load(&db, set.items[0].script_id).unwrap();
        assert_eq!((stored.script.width, stored.script.height), (Some(1080), Some(1080)));
        assert!(stored.script.captions.is_some());

        // An edit saves a new version: it stays in the set and the set points at it.
        let v2 = crate::script::save_version(&db, pid, &stored.script, None).unwrap();
        let set2 = set_item_status(&db, &set.id, v2, "accepted").unwrap();
        assert_eq!(set2.items[0].script_id, v2);
        assert_eq!(set2.items[0].status, "accepted");
        let kind: Option<String> = db.conn.query_row("SELECT kind FROM scripts WHERE id = ?1", [v2], |r| r.get(0)).unwrap();
        assert_eq!(kind.as_deref(), Some("clip"));
        assert_eq!(list_sets(&db, pid).unwrap().len(), 1);
        let trashed = delete_set(&db, &set.id, "app").unwrap();
        assert_eq!((trashed.scripts, trashed.clips), (3, 2));
        assert!(list_sets(&db, pid).unwrap().is_empty());
        assert!(crate::script::list(&db, pid).unwrap().is_empty(), "its clips go with it");

        // Restored exactly: the same clips, statuses and newest versions.
        crate::trash::restore(&db, &trashed.id, "app").unwrap();
        let back = load_set(&db, &set.id).unwrap();
        assert_eq!(back.items.len(), 2);
        assert_eq!((back.items[0].script_id, back.items[0].status.as_str()), (v2, "accepted"));
        assert_eq!(crate::script::list(&db, pid).unwrap().len(), 3);
    }

    fn stored_item_ids(db: &Db, set_id: &str) -> Vec<(i64, String)> {
        let json: String =
            db.conn.query_row("SELECT items_json FROM clip_sets WHERE id = ?1", [set_id], |r| r.get(0)).unwrap();
        serde_json::from_str::<Vec<ClipItem>>(&json).unwrap().into_iter().map(|i| (i.script_id, i.status)).collect()
    }

    #[test]
    fn a_trashed_clip_keeps_its_place_in_the_set_through_a_status_change() {
        let (db, pid, _dir) = library();
        let project = db.project(pid).unwrap();
        let o = ClipOptions { format: ClipFormat::Square, ..opts(20.0, 40.0) };
        let picks = vec![pick("p1", 1, 100.0, 130.0, 8.0, "faces"), pick("p2", 1, 200.0, 225.0, 6.0, "faces")];
        let (set, _) = save_clips(&db, &project, &cfg(), &picks, &o, None, json!({})).unwrap();
        let (first, second) = (set.items[0].script_id, set.items[1].script_id);
        let first_title = crate::script::load(&db, first).unwrap().title;

        // The first clip's cut goes to the trash; the grid shows the other.
        let gone = crate::script::delete_cut(&db, pid, &first_title, "app").unwrap();
        assert_eq!(load_set(&db, &set.id).unwrap().items.len(), 1);

        // Accepting the other must not rewrite the stored list without the trashed one.
        let shown = set_item_status(&db, &set.id, second, "accepted").unwrap();
        assert_eq!(shown.items.len(), 1);
        assert_eq!(stored_item_ids(&db, &set.id), vec![(first, "new".into()), (second, "accepted".into())]);

        // Restoring the cut brings its grid entry back, as it was.
        crate::trash::restore(&db, &gone.id, "app").unwrap();
        let back = load_set(&db, &set.id).unwrap();
        assert_eq!(
            back.items.iter().map(|i| (i.script_id, i.status.as_str())).collect::<Vec<_>>(),
            vec![(first, "new"), (second, "accepted")]
        );
        // And a status can be set on it again.
        let again = set_item_status(&db, &set.id, first, "rejected").unwrap();
        assert_eq!(again.items[0].status, "rejected");
    }

    #[test]
    fn a_set_follows_only_its_own_scripts_and_the_length_range_skips_the_trash() {
        let (db, pid, _dir) = library();
        let project = db.project(pid).unwrap();
        let o = ClipOptions { format: ClipFormat::Square, ..opts(20.0, 40.0) };
        let (set, _) = save_clips(&db, &project, &cfg(), &[pick("p1", 1, 100.0, 130.0, 8.0, "faces")], &o, None, json!({}))
            .unwrap();
        let v1 = set.items[0].script_id;
        let stored = crate::script::load(&db, v1).unwrap();

        assert_eq!(length_range(&db, pid, &stored.title), Some((20.0, 40.0)));

        // Another set holds a newer version of the same title (here put there by hand).
        db.conn
            .execute("INSERT INTO clip_sets(id, project_id, brief, created_at) VALUES ('other', ?1, 'Other', 9)", [pid])
            .unwrap();
        let foreign = crate::script::save_version(&db, pid, &stored.script, None).unwrap();
        db.conn.execute("UPDATE scripts SET set_id = 'other', kind = 'clip' WHERE id = ?1", [foreign]).unwrap();
        assert_eq!(load_set(&db, &set.id).unwrap().items[0].script_id, v1, "not the other set's version");
        assert_eq!(set_item_status(&db, &set.id, v1, "accepted").unwrap().items[0].script_id, v1);

        // A deleted set no longer answers for its clips' length, whether or not the clip rows
        // themselves are marked (only the set is here).
        crate::trash::trash_clip_set(&db, "other", "app").unwrap();
        let entry = crate::trash::trash_clip_set(&db, &set.id, "app").unwrap();
        db.conn.execute("UPDATE scripts SET trash_id = NULL WHERE trash_id = ?1", [&entry.id]).unwrap();
        assert_eq!(length_range(&db, pid, &stored.title), None);
    }

    #[test]
    fn a_fresh_clip_meets_its_own_target_closing_hold_included() {
        let (db, pid, _dir) = library();
        let project = db.project(pid).unwrap();
        let config = ScriptConfig { closing_hold_s: 2.0, ..cfg() };
        let picks = vec![pick("p1", 1, 100.0, 130.0, 8.0, "faces")];
        for format in [ClipFormat::Original, ClipFormat::Vertical] {
            let o = ClipOptions { format, ..opts(20.0, 40.0) };
            let checked = check_clips(&db, &project, &config, &picks, &o).unwrap();
            let s = &checked[0].script;
            assert!(
                crate::script_fix::closing_hold(s).is_some(),
                "the scenario needs a closing hold: {:?}",
                s.beats[0].clips
            );
            let target = s.target_duration_s.unwrap();
            assert!((s.total_duration_s() - target).abs() < 0.011, "{format:?}: runs {} against {target}", s.total_duration_s());
            assert!(
                !checked[0].issues.iter().any(|i| i.severity != IssueSeverity::Info && i.message.starts_with("total duration")),
                "{:?}",
                checked[0].issues
            );
        }
    }

    #[test]
    fn a_cropped_clip_thumbnail_carries_its_planned_framing() {
        let (db, pid, dir) = library();
        db.conn.execute("UPDATE videos SET width = 1920, height = 1080 WHERE id = 1", []).unwrap();
        db.conn.execute("INSERT INTO frames(video_id, t_s, thumb_path) VALUES (1, 102.0, 'f/1.jpg')", []).unwrap();
        // One person on camera, left of centre, all through the clip.
        db.conn
            .execute(
                "INSERT INTO face_tracks(id, video_id, start_s, end_s, n_faces, mean_embedding, avg_box_h)
                 VALUES (1, 1, 90.0, 140.0, 0, x'', 0.4)",
                [],
            )
            .unwrap();
        for i in 0..250 {
            db.conn
                .execute(
                    "INSERT INTO face_boxes(track_id, t_s, x, y, w, h, score, landmarks_json, sharpness, cut_off)
                     VALUES (1, ?1, 0.248, 0.2, 0.224, 0.4, 0.9, '[]', 100.0, 0)",
                    [90.0 + i as f64 * 0.2],
                )
                .unwrap();
        }
        let project = db.project(pid).unwrap();
        let picks = vec![pick("p1", 1, 100.0, 130.0, 8.0, "faces")];
        for (format, framed) in [(ClipFormat::Vertical, true), (ClipFormat::Original, false)] {
            let o = ClipOptions { format, ..opts(20.0, 40.0) };
            let (mut set, _) = save_clips(&db, &project, &cfg(), &picks, &o, None, json!({})).unwrap();
            with_thumbs(&db, dir.path(), &mut set);
            let item = &set.items[0];
            assert!(item.thumb.as_deref().is_some_and(|t| t.ends_with("1.jpg")));
            assert_eq!(item.thumb_view.is_some(), framed, "{format:?}: {:?}", item.thumb_view);
            if let Some(v) = &item.thumb_view {
                assert_eq!((v.out_w, v.out_h), (1080.0, 1920.0));
                // A 9:16 window of the 16:9 picture, inside it.
                let c = v.crops.first().expect("one crop");
                assert!((c.w / c.h - 1080.0 / 1920.0).abs() < 0.02 && c.x >= 0.0 && c.x + c.w <= 1920.5, "{c:?}");
                // On the face (x ≈ 0.36 of the width), not the centre of the picture.
                assert!(c.x < 0.36 * 1920.0 && c.x + c.w > 0.36 * 1920.0 && c.x + c.w < 1920.0 * 0.75, "{c:?}");
            }
        }
    }

    #[test]
    fn overlapping_clips_in_a_set_are_reported() {
        let (db, pid, _dir) = library();
        let project = db.project(pid).unwrap();
        let picks = vec![pick("p1", 1, 100.0, 130.0, 8.0, "faces"), pick("p2", 1, 120.0, 150.0, 6.0, "faces")];
        let checked = check_clips(&db, &project, &cfg(), &picks, &opts(20.0, 40.0)).unwrap();
        assert!(checked[1].issues.iter().any(|i| i.message.contains("overlaps clip 1")));
    }

    async fn fake_worker_server() -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut got = Vec::new();
                    let mut buf = vec![0u8; 65536];
                    loop {
                        let n = sock.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        got.extend_from_slice(&buf[..n]);
                        let t = String::from_utf8_lossy(&got);
                        if let Some(h) = t.find("\r\n\r\n") {
                            let len = t[..h]
                                .lines()
                                .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                                .unwrap_or(0);
                            if got.len() >= h + 4 + len {
                                break;
                            }
                        }
                    }
                    let t = String::from_utf8_lossy(&got).to_string();
                    // Pick the first candidate the prompt lists.
                    let id = t.split("v1c").nth(1).map(|r| format!("v1c{}", r.chars().take_while(|c| c.is_ascii_digit()).collect::<String>())).unwrap_or_default();
                    let content = json!({ "picks": [{ "candidate": id, "in_s": 0, "out_s": 0, "title": "T", "quote": "Q", "why": "W", "score": 7, "hook": "H" }] }).to_string();
                    let body = json!({ "choices": [{ "message": { "content": content } }] }).to_string();
                    let resp = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn readiness_blocks_on_transcripts_and_warns_on_analysis() {
        use crate::preflight::{Level, Needs, project_readiness};
        let (db, pid, _dir) = library();
        let needs = Needs { transcripts: true, analysis: true };
        assert!(project_readiness(&db, pid, &[], needs).unwrap().is_empty(), "indexed: nothing to say");
        let job = |stage: &str, state: &str| {
            db.conn
                .execute(
                    "INSERT OR REPLACE INTO jobs(video_id, stage, state, updated_at) VALUES (1, ?1, ?2, 0)",
                    params![stage, state],
                )
                .unwrap();
        };
        job("describe", "running");
        job("faces", "pending");
        job("speakers", "pending");
        let checks = project_readiness(&db, pid, &[], needs).unwrap();
        assert_eq!(checks.iter().map(|c| (c.subject.as_str(), c.level)).collect::<Vec<_>>(), [
            ("describe", Level::Warn),
            ("faces", Level::Warn),
            ("speakers", Level::Warn)
        ]);
        job("transcribe", "pending");
        let pre = crate::preflight::Preflight { checks: project_readiness(&db, pid, &[], needs).unwrap() };
        assert_eq!(pre.level(), Level::Error);
        let err = pre.into_result().unwrap_err().to_string();
        assert!(err.contains("transcription is still running for the video"), "{err}");
    }

    #[tokio::test]
    async fn a_worker_only_run_saves_a_clip_set() {
        let (mut db, pid, _dir) = library();
        let url = fake_worker_server().await;
        use crate::agents::{AgentTarget, EndpointKey, ModelRunner, PoolAgent, SlotRegistry};
        let agent = PoolAgent {
            spec: AgentSpec { id: "srv".into(), max_concurrent: 2, roles: [AgentRole::Worker].into(), ..AgentSpec::default() },
            key: EndpointKey::Server { url: url.clone(), model: String::new() },
            capacity: 2,
            target: AgentTarget::Server { url, model: String::new(), api_key: String::new(), timeout_s: 10 },
            describe: "srv".into(),
            max_tool_rounds: 0,
        };
        let selection = crate::agents::resolve_selection(
            std::slice::from_ref(&agent.spec),
            None,
            None,
            &AgentSelection::default(),
        )
        .unwrap();
        let backends = Backends {
            workers: AgentPool::new(vec![agent], 2, SlotRegistry::new(), Arc::new(ModelRunner::default())),
            orchestrator: None,
            orchestrator_skipped: None,
            orchestrator_timeout_s: 10,
            selection,
            tools: None,
        };
        let o = ClipOptions { count: 2, window_s: 200.0, ..opts(20.0, 40.0) };
        let mut phases = Vec::new();
        let res = run(&mut db, pid, &cfg(), &backends, &o, None, &mut |_: f64, phase: &str, _: &str| phases.push(phase.to_string())).await.unwrap();
        assert_eq!(res.set.items.len(), 2);
        assert_eq!(res.log["worker_jobs"], 3);
        assert!(phases.contains(&"workers".to_string()) && phases.last().map(String::as_str) == Some("done"));
        assert_eq!(res.log["agents_line"], "srv 3");
        // The clips remember the agents that made them.
        let clip = crate::script::load(&db, res.set.items[0].script_id).unwrap();
        assert_eq!(clip.script.agents.unwrap().workers, Some(vec!["srv".to_string()]));
        let a = &res.set.items[0];
        let b = &res.set.items[1];
        assert!(overlap((a.in_s, a.out_s), (b.in_s, b.out_s)) == 0.0);
    }

    /// Workers and the orchestrator get the tool registry for the run's project: here each asks
    /// for something before it answers, and the answer to the ask comes from the database.
    #[tokio::test]
    async fn a_clip_run_gives_workers_and_the_orchestrator_the_tool_registry() {
        let (mut db, pid, dir) = library_on_disk();
        let paths = crate::paths::Paths::in_dir(dir.path());
        let (url, seen) = crate::tools::tests::scripted_server(|request, n| {
            let text = request.to_string();
            let looked = request["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool");
            let call = |name: &str, args: Value| {
                json!({ "content": null, "tool_calls": [{ "id": format!("c{n}"), "type": "function",
                    "function": { "name": name, "arguments": args.to_string() } }] })
            };
            if text.contains("Proposals") {
                if !looked {
                    return call("get_video", json!({ "video_id": 1, "limit": 2 }));
                }
                return json!({ "content": json!({ "clips": [
                    { "id": "p1", "title": "T1", "hook": "H1", "why": "W" },
                    { "id": "p2", "title": "T2", "hook": "H2", "why": "W" } ] }).to_string() });
            }
            if !looked {
                return call("get_transcript", json!({ "video_id": 1, "start_s": 0, "end_s": 20 }));
            }
            let id = text.split("v1c").nth(1).map(|r| format!("v1c{}", r.chars().take_while(|c| c.is_ascii_digit()).collect::<String>())).unwrap_or_default();
            json!({ "content": json!({ "picks": [{ "candidate": id, "in_s": 0, "out_s": 0, "title": "T", "quote": "Q", "why": "W", "score": 7, "hook": "H" }] }).to_string() })
        })
        .await;
        use crate::agents::{AgentTarget, EndpointKey, ModelRunner, PoolAgent, SlotRegistry};
        let worker = PoolAgent {
            spec: AgentSpec { id: "srv".into(), max_concurrent: 2, roles: [AgentRole::Worker].into(), ..AgentSpec::default() },
            key: EndpointKey::Server { url: url.clone(), model: String::new() },
            capacity: 2,
            target: AgentTarget::Server { url: url.clone(), model: String::new(), api_key: String::new(), timeout_s: 10 },
            describe: "srv".into(),
            max_tool_rounds: 0,
        };
        let selection =
            crate::agents::resolve_selection(std::slice::from_ref(&worker.spec), None, None, &AgentSelection::default()).unwrap();
        let backends = Backends {
            workers: AgentPool::new(vec![worker], 2, SlotRegistry::new(), Arc::new(ModelRunner::default())),
            orchestrator: Some(Orchestrator {
                agent: AgentSpec { id: "boss".into(), roles: [AgentRole::Orchestrator].into(), ..AgentSpec::default() },
                setup: crate::runtime::VisionSetup::Server(crate::vision::ServerVision::new(&url, "", "")),
                max_tool_rounds: 0,
            }),
            orchestrator_skipped: None,
            orchestrator_timeout_s: 10,
            selection,
            tools: Some(crate::tools::ToolAccess::new(paths, Arc::new(Config::default()), 0, None)),
        };
        let o = ClipOptions { count: 2, window_s: 200.0, ..opts(20.0, 40.0) };
        let res = run(&mut db, pid, &cfg(), &backends, &o, None, &mut |_: f64, _: &str, _: &str| {}).await.unwrap();
        assert_eq!(res.set.items.len(), 2);
        assert_eq!(res.log["orchestrator"]["used"], true, "{}", res.log["orchestrator"]);

        let seen = seen.lock().unwrap();
        let tool_answers = |needle: &str| -> Vec<String> {
            seen.iter()
                .filter(|r| r.to_string().contains(needle))
                .flat_map(|r| r["messages"].as_array().unwrap().clone())
                .filter(|m| m["role"] == "tool")
                .map(|m| m["content"].as_str().unwrap_or_default().to_string())
                .collect()
        };
        assert!(tool_answers("Candidate clips").iter().any(|t| t.contains("Esta es la frase")), "a worker read the transcript");
        assert!(tool_answers("Proposals").iter().any(|t| t.contains("keyframes_total")), "the orchestrator opened a video");
        // The run log says which tools each job called.
        let calls = |v: &Value| v["tool_calls"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert!(res.log["workers"].as_array().unwrap().iter().all(|w| calls(w).iter().any(|c| c.starts_with("get_transcript: "))), "{}", res.log["workers"]);
        assert!(calls(&res.log["orchestrator"]).iter().any(|c| c.starts_with("get_video: ")), "{}", res.log["orchestrator"]);
    }
}
