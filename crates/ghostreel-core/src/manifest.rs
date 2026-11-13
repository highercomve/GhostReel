//! The manifest: everything the index knows about a video, joined on one clock.
//!
//! An editor (a person or a model) picks moments by what is said, who says it, what the picture
//! shows while it is said, and whether a cut there will be clean. Those facts live in separate
//! stages of the index — sentences, measured speech, speakers, face tracks, scene kinds, keyframe
//! descriptions, camera motion, chapters — and an editor that has to ask for each one separately
//! spends its rounds cross-referencing them. The manifest lays them side by side, in time order.
//!
//! It holds measurements, not judgements: pauses in seconds, faces as positions and sizes, the
//! camera's jerk as a number next to the clip's own shake limit. What is usable, what is a good
//! line, where a thought begins — that is the editor's call. It is read straight from the index
//! each time, so it is never out of date with it.

use serde::Serialize;

use crate::Error;
use crate::db::Db;
use crate::scenes::{Scene, SceneSpan};
use crate::sentences::{clock, format_id};

/// How many keyframe descriptions the manifest carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Frames {
    /// One at each change of picture (and at a chapter's start).
    #[default]
    Changes,
    /// Every described keyframe.
    All,
    None,
}

/// What to read and how much of it.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub start_s: f64,
    pub end_s: f64,
    pub frames: Frames,
    /// Each sentence's words with their times.
    pub words: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { start_s: 0.0, end_s: f64::MAX, frames: Frames::Changes, words: false }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub video_id: i64,
    pub file: String,
    pub duration_s: f64,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub fps: Option<f64>,
    pub language: Option<String>,
    /// The camera's high-frequency movement above which this video counts as shaky (% of the
    /// frame width per frame); the motion events are measured against it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shake_limit: Option<f64>,
    pub events: Vec<Event>,
}

/// One fact on the video's clock.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Chapter {
        idx: i64,
        start_s: f64,
        end_s: f64,
        title: String,
        summary: String,
    },
    /// A change of picture: what kind, and every face on screen through it.
    Picture {
        start_s: f64,
        end_s: f64,
        scene: Scene,
        faces: Vec<FaceOnScreen>,
        /// The change into this picture as the pixels show it: a dissolve or an animated change of
        /// layout takes a span, and the faces put the change at one instant inside or near it.
        #[serde(skip_serializing_if = "Option::is_none")]
        change: Option<crate::transitions::Transition>,
    },
    Frame {
        t_s: f64,
        description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        visible_text: Option<String>,
    },
    Sentence {
        #[serde(serialize_with = "crate::sentences::serialize_id")]
        id: i64,
        start_s: f64,
        end_s: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        speaker: Option<String>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        off_mic: bool,
        /// Silence in the measured speech right before and after the sentence (s): how much room
        /// a cut there has. None where the speech was not measured.
        #[serde(skip_serializing_if = "Option::is_none")]
        pause_before_s: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pause_after_s: Option<f64>,
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        words: Vec<(String, f64, f64)>,
    },
    /// A stretch where the camera's jerk is over the video's shake limit.
    Motion {
        start_s: f64,
        end_s: f64,
        jerk: f64,
        sway: f64,
    },
}

impl Event {
    fn t(&self) -> f64 {
        match self {
            Event::Chapter { start_s, .. }
            | Event::Picture { start_s, .. }
            | Event::Sentence { start_s, .. }
            | Event::Motion { start_s, .. } => *start_s,
            Event::Frame { t_s, .. } => *t_s,
        }
    }

    /// Ties at one instant read as heading, picture, frame, motion, words.
    fn order(&self) -> u8 {
        match self {
            Event::Chapter { .. } => 0,
            Event::Picture { .. } => 1,
            Event::Frame { .. } => 2,
            Event::Motion { .. } => 3,
            Event::Sentence { .. } => 4,
        }
    }
}

/// A face through one picture span: who, where and how big (all 0..1 of the frame), measured
/// over every detection of it in the span.
#[derive(Debug, Clone, Serialize)]
pub struct FaceOnScreen {
    /// The person's name or handle, else `person N`, else `face N` (a track nobody was matched to).
    pub who: String,
    /// Centre of the face box.
    pub x: f64,
    pub y: f64,
    /// Box height: median, and the 10th..90th percentile range (how much it grows or shrinks).
    pub h: f64,
    pub h_range: (f64, f64),
    /// Centre height range, 10th..90th percentile (how much it moves up and down).
    pub y_range: (f64, f64),
    /// Share of the span the face is on screen at all.
    pub present: f64,
    /// Share of its detections where the box touches the frame's edge.
    pub cut_off: f64,
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Faces seen between `a` and `b`, one entry per track (tracks of one person are kept apart:
/// two tiles of the same person in a grid are two faces).
fn faces_in(db: &Db, video_id: i64, a: f64, b: f64) -> Result<Vec<FaceOnScreen>, Error> {
    let mut st = db.conn.prepare_cached(
        "SELECT t.id, t.person_id, NULLIF(TRIM(p.name), ''), NULLIF(TRIM(p.handle), ''),
                fb.x + fb.w / 2.0, fb.y + fb.h / 2.0, fb.h, fb.cut_off
           FROM face_boxes fb JOIN face_tracks t ON t.id = fb.track_id
           LEFT JOIN people p ON p.id = t.person_id
          WHERE t.video_id = ?1 AND fb.t_s >= ?2 AND fb.t_s < ?3
          ORDER BY t.id",
    )?;
    type Row = (i64, Option<i64>, Option<String>, Option<String>, f64, f64, f64, bool);
    let rows: Vec<Row> = st
        .query_map(rusqlite::params![video_id, a, b], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?))
        })?
        .collect::<Result<_, _>>()?;
    // Detections run five a second (`faces`); a span's expected count is its length at that rate.
    let expected = ((b - a) * 5.0).max(1.0);
    let mut out = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let track = rows[i].0;
        let j = rows[i..].iter().position(|r| r.0 != track).map_or(rows.len(), |k| i + k);
        let group = &rows[i..j];
        let (_, person, name, handle, ..) = &group[0];
        let who = name
            .clone()
            .or_else(|| handle.clone())
            .or_else(|| person.map(|p| format!("person {p}")))
            .unwrap_or_else(|| format!("face {track}"));
        let sorted = |f: fn(&Row) -> f64| {
            let mut v: Vec<f64> = group.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let (xs, ys, hs) = (sorted(|r| r.4), sorted(|r| r.5), sorted(|r| r.6));
        let n = group.len() as f64;
        out.push(FaceOnScreen {
            who,
            x: r2(pct(&xs, 0.5)),
            y: r2(pct(&ys, 0.5)),
            h: r2(pct(&hs, 0.5)),
            h_range: (r2(pct(&hs, 0.1)), r2(pct(&hs, 0.9))),
            y_range: (r2(pct(&ys, 0.1)), r2(pct(&ys, 0.9))),
            present: r2((n / expected).min(1.0)),
            cut_off: r2(group.iter().filter(|r| r.7).count() as f64 / n),
        });
        i = j;
    }
    // A face seen for a moment is a detection's noise, not someone on screen.
    out.retain(|f| f.present >= 0.1);
    out.sort_by(|p, q| p.x.total_cmp(&q.x));
    Ok(out)
}

/// The silence in the measured speech that contains `t`, or touches it: (start, end).
/// `file` is the video's (start, end) when the regions reach that far: the silence before the first
/// voice runs from the start of the file, and after the last one to its end.
fn pause_around(regions: &[crate::speech::SpeechRegion], t: f64, file: (Option<f64>, Option<f64>)) -> Option<f64> {
    if regions.is_empty() {
        return None;
    }
    // Inside a stretch of continuous speech (a sentence that starts or ends without a pause, as at
    // a speaker change that talks over the last word): no silence at all.
    if regions.iter().any(|r| r.start_s < t - 0.05 && r.end_s > t + 0.05) {
        return Some(0.0);
    }
    // The last region that ends at or before t (a little slack: sentence edges sit on pauses) and
    // the first that starts at or after it.
    let before = regions.iter().rev().find(|r| r.end_s <= t + 0.05).map(|r| r.end_s).or(file.0);
    let after = regions.iter().find(|r| r.start_s >= t - 0.05).map(|r| r.start_s).or(file.1);
    match (before, after) {
        (Some(a), Some(b)) if b >= a => Some(r2(b - a)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn frame_text(json: Option<&str>) -> Option<String> {
    json.and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok())
        .and_then(|v| v.get("description").and_then(|d| d.as_str()).map(str::to_string))
}

/// The manifest of one video over `opts`' range.
pub fn build(db: &Db, video_id: i64, opts: &Options) -> Result<Manifest, Error> {
    let (from, to) = (opts.start_s.max(0.0), opts.end_s);
    let (duration_s, width, height, fps, language): (f64, Option<i64>, Option<i64>, Option<f64>, Option<String>) =
        db.conn.query_row(
            "SELECT COALESCE(duration_s, 0), width, height, COALESCE(avg_fps, fps), language FROM videos WHERE id = ?1",
            [video_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )?;
    let to = to.min(duration_s.max(from));
    let mut events = Vec::new();

    // Chapters that overlap the range.
    {
        let mut st = db.conn.prepare(
            "SELECT idx, start_s, end_s, title, summary FROM chapters
              WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 ORDER BY start_s",
        )?;
        let rows = st.query_map(rusqlite::params![video_id, from, to], |r| {
            Ok(Event::Chapter {
                idx: r.get(0)?,
                start_s: r.get::<_, f64>(1)?.max(from),
                end_s: r.get(2)?,
                title: r.get(3)?,
                summary: r.get(4)?,
            })
        })?;
        for e in rows {
            events.push(e?);
        }
    }
    let chapter_starts: Vec<f64> = events.iter().map(Event::t).collect();

    // Pictures: the scene spans, each with its faces.
    let spans: Vec<SceneSpan> = crate::scenes::for_range(db, video_id, from, to)?.unwrap_or_default();
    // Each change of picture measured from the pixels — for a reading of a stretch, not a whole
    // video, where it would mean decoding a few seconds around every one of hundreds of changes.
    let measure_changes = spans.len() <= MEASURED_CHANGES;
    for s in &spans {
        let change = (measure_changes && s.start > from + 0.05)
            .then(|| {
                crate::transitions::around(db, video_id, s.start)
                    .into_iter()
                    .filter(|t| !t.is_cut())
                    .min_by(|a, b| ((a.start + a.end) / 2.0 - s.start).abs().total_cmp(&((b.start + b.end) / 2.0 - s.start).abs()))
            })
            .flatten();
        events.push(Event::Picture {
            start_s: s.start,
            end_s: s.end,
            scene: s.scene,
            faces: faces_in(db, video_id, s.start, s.end)?,
            change,
        });
    }

    // Keyframes.
    if opts.frames != Frames::None {
        let mut st = db.conn.prepare(
            "SELECT t_s, description_json, NULLIF(TRIM(visible_text), '') FROM frames
              WHERE video_id = ?1 AND t_s >= ?2 AND t_s <= ?3 AND description_json IS NOT NULL ORDER BY t_s",
        )?;
        let frames: Vec<(f64, Option<String>, Option<String>)> = st
            .query_map(rusqlite::params![video_id, from - 0.5, to + 0.5], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<Result<_, _>>()?;
        // The first frame at or after each picture change and chapter start.
        let marks: Vec<f64> = spans.iter().map(|s| s.start).chain(chapter_starts.iter().copied()).collect();
        let mut keep = vec![opts.frames == Frames::All; frames.len()];
        if opts.frames == Frames::Changes {
            for m in marks {
                if let Some(k) = frames.iter().position(|f| f.0 >= m - 0.5) {
                    keep[k] = true;
                }
            }
            if frames.first().is_some() && spans.is_empty() {
                keep[0] = true;
            }
        }
        for ((t, json, text), k) in frames.into_iter().zip(keep) {
            if let (true, Some(description)) = (k, frame_text(json.as_deref())) {
                events.push(Event::Frame { t_s: r2(t), description, visible_text: text });
            }
        }
    }

    // Sentences, with the pauses around them and, when asked, their words.
    let regions = crate::speech::speech_regions(db, video_id, from - 30.0, to + 30.0).unwrap_or_default();
    let file = ((from - 30.0 <= 0.0).then_some(0.0), (to + 30.0 >= duration_s).then_some(duration_s));
    // A sentence's own words, numbered as "s412@n" counts them: word n of its text, with its time.
    let words_of = |id: i64, text: &str| -> Vec<(String, f64, f64)> {
        let times = crate::sentences::own_words(db, id);
        if times.is_empty() {
            return Vec::new();
        }
        text.split_whitespace().zip(times).map(|(w, (a, b))| (w.to_string(), r2(a), r2(b))).collect()
    };
    for s in crate::index::transcript(db, video_id)?.into_iter().filter(|s| s.end > from && s.start < to) {
        events.push(Event::Sentence {
            id: s.id,
            start_s: r2(s.start),
            end_s: r2(s.end),
            speaker: s.speaker,
            off_mic: s.off_mic,
            pause_before_s: pause_around(&regions, s.start, file),
            pause_after_s: pause_around(&regions, s.end, file),
            text: s.text.trim().to_string(),
            words: if opts.words { words_of(s.id, &s.text) } else { Vec::new() },
        });
    }

    // Camera motion over the video's own shake limit, adjacent windows joined.
    let windows = db.motion_windows(video_id).unwrap_or_default();
    let cfg = crate::config::ScriptConfig::default();
    let shake_limit =
        (!windows.is_empty()).then(|| crate::steadiness::shake_limit(&windows, cfg.max_shake_jerk, cfg.shake_relative));
    if let Some(limit) = shake_limit {
        let mut run: Option<(f64, f64, f64, f64)> = None;
        for w in windows.iter().filter(|w| w.end_s > from && w.start_s < to) {
            if w.jerk > limit {
                run = Some(match run {
                    Some((a, b, j, s)) if w.start_s <= b + 0.01 => (a, w.end_s, j.max(w.jerk), s.max(w.sway)),
                    Some(done) => {
                        events.push(Event::Motion {
                            start_s: done.0,
                            end_s: done.1,
                            jerk: r2(done.2),
                            sway: r2(done.3),
                        });
                        (w.start_s, w.end_s, w.jerk, w.sway)
                    }
                    None => (w.start_s, w.end_s, w.jerk, w.sway),
                });
            }
        }
        if let Some(done) = run {
            events.push(Event::Motion { start_s: done.0, end_s: done.1, jerk: r2(done.2), sway: r2(done.3) });
        }
    }

    events.sort_by(|a, b| a.t().total_cmp(&b.t()).then(a.order().cmp(&b.order())));
    Ok(Manifest {
        video_id,
        file: db
            .conn
            .query_row("SELECT MIN(path) FROM video_files WHERE video_id = ?1", [video_id], |r| {
                r.get::<_, Option<String>>(0)
            })
            .ok()
            .flatten()
            .and_then(|p| std::path::Path::new(&p).file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default(),
        duration_s,
        width,
        height,
        fps: fps.map(r2),
        language,
        shake_limit: shake_limit.map(r2),
        events,
    })
}

fn scene_word(s: Scene) -> &'static str {
    match s {
        Scene::TwoShot => "two faces",
        Scene::SingleFace => "one face",
        Scene::ScreenShare => "screen share",
        Scene::NoFace => "no face",
    }
}

fn face_text(f: &FaceOnScreen) -> String {
    let mut s = format!("{} x{:.2} y{:.2} h{:.2}", f.who, f.x, f.y, f.h);
    if f.h_range.1 - f.h_range.0 >= 0.05 {
        s.push_str(&format!(" (h {:.2}–{:.2})", f.h_range.0, f.h_range.1));
    }
    if f.y_range.1 - f.y_range.0 >= 0.05 {
        s.push_str(&format!(" (y {:.2}–{:.2})", f.y_range.0, f.y_range.1));
    }
    if f.present < 0.9 {
        s.push_str(&format!(" on {:.0}%", f.present * 100.0));
    }
    if f.cut_off > 0.0 {
        s.push_str(&format!(" cut off {:.0}%", f.cut_off * 100.0));
    }
    s
}

impl Manifest {
    /// The header line and the key to the lines below it.
    fn header(&self) -> String {
        let mut h = format!("=== video {} ({}, {}", self.video_id, self.file, clock(self.duration_s));
        if let (Some(w), Some(ht)) = (self.width, self.height) {
            h.push_str(&format!(", {w}x{ht}"));
        }
        if let Some(f) = self.fps {
            h.push_str(&format!(", {f} fps"));
        }
        if let Some(l) = &self.language {
            h.push_str(&format!(", {l}"));
        }
        h.push_str(")\n");
        h
    }

    /// The manifest as compact text: one line per event, in time order.
    pub fn render(&self) -> String {
        self.render_events(&self.events)
    }

    pub fn render_events(&self, events: &[Event]) -> String {
        let mut out = self.header();
        for e in events {
            out.push_str(&line(e, self.shake_limit));
            out.push('\n');
        }
        out
    }
}

/// At most this many changes of picture in one reading are measured from the pixels.
const MEASURED_CHANGES: usize = 24;

/// One event as a manifest line.
pub fn line(e: &Event, shake_limit: Option<f64>) -> String {
    match e {
        Event::Chapter { idx, start_s, end_s, title, summary } => {
            format!("## ch{idx} {}–{} {title}\n   {summary}", clock(*start_s), clock(*end_s))
        }
        Event::Picture { start_s, end_s, scene, faces, change } => {
            let mut s = format!("[picture {}–{} {}", clock(*start_s), clock(*end_s), scene_word(*scene));
            if let Some(c) = change {
                s.push_str(&format!(" (changing over {}–{})", clock(c.start), clock(c.end)));
            }
            if !faces.is_empty() {
                s.push_str(": ");
                s.push_str(&faces.iter().map(face_text).collect::<Vec<_>>().join(" · "));
            }
            s.push(']');
            s
        }
        Event::Frame { t_s, description, visible_text } => {
            let mut s = format!("[frame {}] {description}", clock(*t_s));
            if let Some(t) = visible_text {
                s.push_str(&format!(" | on screen: {}", t.replace('\n', " / ")));
            }
            s
        }
        Event::Motion { start_s, end_s, jerk, sway } => format!(
            "[camera {}–{} jerk {jerk:.2} (limit {:.2}) sway {sway:.2}]",
            clock(*start_s),
            clock(*end_s),
            shake_limit.unwrap_or(0.0)
        ),
        Event::Sentence { id, start_s, end_s, speaker, off_mic, pause_before_s, pause_after_s, text, words } => {
            let mut s = format!("{} {}–{}", format_id(*id), clock(*start_s), clock(*end_s));
            if let Some(who) = speaker {
                s.push(' ');
                s.push_str(who);
            }
            if *off_mic {
                s.push_str(" off-mic");
            }
            let p = |x: &Option<f64>| x.map_or("?".to_string(), |v| format!("{v:.2}"));
            if pause_before_s.is_some() || pause_after_s.is_some() {
                s.push_str(&format!(" pause {}/{}", p(pause_before_s), p(pause_after_s)));
            }
            s.push_str(": ");
            s.push_str(text);
            if !words.is_empty() {
                s.push_str("\n   words: ");
                s.push_str(
                    &words
                        .iter()
                        .enumerate()
                        .map(|(n, (w, a, _))| format!("{}:{w}@{a:.2}", n + 1))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::SpeechRegion;

    #[test]
    fn a_boundary_inside_continuous_speech_has_no_pause() {
        // B3's "8.33 s gap" at a speaker change: the sentence starts inside one stretch of speech,
        // and the gap to the next stretch is not a pause before it.
        let regions = [SpeechRegion::new(1.0, 9.0), SpeechRegion::new(12.0, 14.0)];
        assert_eq!(pause_around(&regions, 5.0, (None, None)), Some(0.0));
        assert_eq!(pause_around(&regions, 9.0, (None, None)), Some(3.0));
    }
}
