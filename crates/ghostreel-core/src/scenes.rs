//! Layout segmentation: what kind of picture each stretch of a video is, read from the face boxes
//! the faces stage stored (about 5 per second).
//!
//! - `two_shot`: two or more people on camera at a size worth framing (tiles side by side, or a
//!   room with two people in it).
//! - `single_face`: one person filling the picture.
//! - `screen_share`: only small faces, sitting in camera tiles at the edge of the frame, over a
//!   shared screen (a slide, an app, a browser).
//! - `no_face`: nobody on screen (b-roll, a full-screen share, a transition).
//!
//! One function for everyone who needs to know: framing plans one layout per stretch, clip
//! candidates prefer people on camera for vertical formats, checks warn about the screen-share
//! edges of a vertical clip, and the MCP tools show it. Pure geometry: no model, no pixels.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;

/// Faces at least this tall (share of the picture height) are people we frame on their own; below
/// it they are camera tiles or a wide shot.
pub const FRAMED_FACE_H: f64 = 0.15;
/// Faces smaller than this are noise: a face on a slide, a poster, a detection on a pattern.
pub const MIN_FACE_H: f64 = 0.03;
/// Small faces whose centre is within this band of the top or bottom of the picture, or within
/// [`TILE_SIDE_BAND`] of a side, are camera tiles.
pub const TILE_BAND: f64 = 0.30;
pub const TILE_SIDE_BAND: f64 = 0.15;
/// Classification step.
pub const BIN_S: f64 = 0.4;
/// Stretches shorter than this are folded into a neighbour: a transition, a missed detection, a
/// head turned away for a moment. About the shortest shot that reads as a shot.
pub const MIN_SCENE_S: f64 = 1.5;

/// What a stretch of picture is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scene {
    TwoShot,
    SingleFace,
    ScreenShare,
    NoFace,
}

impl Scene {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TwoShot => "two_shot",
            Self::SingleFace => "single_face",
            Self::ScreenShare => "screen_share",
            Self::NoFace => "no_face",
        }
    }

    /// Words for people: "a two-shot", "a screen share".
    pub fn phrase(&self) -> &'static str {
        match self {
            Self::TwoShot => "a two-shot",
            Self::SingleFace => "a single face",
            Self::ScreenShare => "a screen share",
            Self::NoFace => "a shot with no one in it",
        }
    }

    /// People fill the picture: what a vertical crop wants.
    pub fn has_people(&self) -> bool {
        matches!(self, Self::TwoShot | Self::SingleFace)
    }
}

/// One face seen at one instant, in picture coordinates (0..1 of the picture with any black bars
/// left out). `t` in seconds, on whatever clock the caller uses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaceObs {
    pub t: f64,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
}

/// A stretch of one kind of picture, on the caller's clock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneSpan {
    pub start: f64,
    pub end: f64,
    pub scene: Scene,
    /// Typical number of faces on screen.
    pub faces: f64,
    /// Typical height of the biggest face (share of the picture height).
    pub face_h: f64,
}

impl SceneSpan {
    pub fn len(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 0.0
    }
}

/// How much of a range is each kind of picture (shares 0..1, summing to 1 when anything was
/// measured).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SceneMix {
    pub two_shot: f64,
    pub single_face: f64,
    pub screen_share: f64,
    pub no_face: f64,
}

impl SceneMix {
    /// Share with people on camera at a framable size.
    pub fn people(&self) -> f64 {
        self.two_shot + self.single_face
    }

    pub fn share(&self, scene: Scene) -> f64 {
        match scene {
            Scene::TwoShot => self.two_shot,
            Scene::SingleFace => self.single_face,
            Scene::ScreenShare => self.screen_share,
            Scene::NoFace => self.no_face,
        }
    }

    /// "two-shot 55% · screen share 45%": the parts over 5%, biggest first.
    pub fn summary(&self) -> String {
        let mut parts = [
            ("two-shot", self.two_shot),
            ("single face", self.single_face),
            ("screen share", self.screen_share),
            ("no face", self.no_face),
        ];
        parts.sort_by(|a, b| b.1.total_cmp(&a.1));
        let s: Vec<String> =
            parts.iter().filter(|p| p.1 >= 0.05).map(|(n, v)| format!("{n} {:.0}%", v * 100.0)).collect();
        if s.is_empty() { "no faces measured".into() } else { s.join(" · ") }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// A small face is in a camera tile when it sits along an edge of the picture.
fn in_tile_band(o: &FaceObs) -> bool {
    o.cy >= 1.0 - TILE_BAND || o.cy <= TILE_BAND * 0.7 || o.cx <= TILE_SIDE_BAND || o.cx >= 1.0 - TILE_SIDE_BAND
}

/// Two tile faces are a pair when they sit in a row (same height in the picture) or a column (same
/// distance from the side) and are about the same size: how a conferencing tool lays out its
/// participants.
fn tile_pair(a: &FaceObs, b: &FaceObs) -> bool {
    let same_size = a.h.max(b.h) <= a.h.min(b.h) * 1.6;
    let row = (a.cy - b.cy).abs() < 0.06 && (a.cx - b.cx).abs() > 0.03;
    let column = (a.cx - b.cx).abs() < 0.06 && (a.cy - b.cy).abs() > 0.03;
    same_size && (row || column)
}

/// Whether the faces include camera tiles over a shared screen: a pair of small faces in the tile
/// band, or two or more tall participant tiles stacked along one side of the picture. The shared
/// content around them can itself hold faces (a slide, a gallery, a video), big or small, which say
/// nothing about the camera.
fn has_camera_tiles(faces: &[&&FaceObs]) -> bool {
    let tiles: Vec<&&&FaceObs> = faces.iter().filter(|o| o.h < FRAMED_FACE_H && in_tile_band(o)).collect();
    let small_pair = tiles.iter().enumerate().any(|(i, a)| tiles[i + 1..].iter().any(|b| tile_pair(a, b)));
    let tall: Vec<&&&FaceObs> = faces.iter().filter(|o| o.h >= FRAMED_FACE_H).collect();
    // Tall tiles hug one side and stack (two hosts side by side in a real two-shot do not).
    let on_one_side = |side: fn(f64) -> bool| tall.len() >= 2 && tall.iter().all(|o| side(o.cx));
    let spread = |tall: &[&&&FaceObs]| {
        let (lo, hi) = tall.iter().fold((1.0f64, 0.0f64), |(lo, hi), o| (lo.min(o.cx), hi.max(o.cx)));
        hi - lo
    };
    let tall_column = (on_one_side(|cx| cx >= 0.7) || on_one_side(|cx| cx <= 0.3)) && spread(&tall) < 0.12;
    small_pair || tall_column
}

/// The scene of one instant's faces.
fn classify(faces: &[&FaceObs]) -> Scene {
    let faces: Vec<&&FaceObs> = faces.iter().filter(|o| o.h >= MIN_FACE_H).collect();
    let big = faces.iter().filter(|o| o.h >= FRAMED_FACE_H).count();
    // Camera tiles with anything but one framed person beside them are a screen share: with one
    // person framed it is a speaker view with thumbnails, and that person is the picture.
    if big != 1 && has_camera_tiles(&faces) {
        return Scene::ScreenShare;
    }
    match big {
        0 if faces.is_empty() => Scene::NoFace,
        0 if faces.iter().all(|o| in_tile_band(o)) => Scene::ScreenShare,
        // Small faces in the middle of the picture: a wide shot of the room.
        0 => {
            if faces.len() >= 2 {
                Scene::TwoShot
            } else {
                Scene::SingleFace
            }
        }
        1 => Scene::SingleFace,
        _ => Scene::TwoShot,
    }
}

/// Classify `from..to` from the faces seen in it (any order; observations outside the range are
/// ignored). Every instant of the range lands in exactly one span; spans are in order, touch end
/// to end, and none is shorter than [`MIN_SCENE_S`] unless the range itself is.
/// The picture in bins of [`BIN_S`] over `from..to`: the observations in each bin, and each bin's
/// scene (the most common among its instants).
fn bin_labels(obs: &[FaceObs], from: f64, to: f64) -> (Vec<Vec<&FaceObs>>, Vec<Scene>) {
    let dur = to - from;
    if dur <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let n = ((dur / BIN_S).ceil() as usize).max(1);
    let mut bins: Vec<Vec<&FaceObs>> = vec![Vec::new(); n];
    for o in obs.iter().filter(|o| o.t >= from && o.t < to) {
        let i = (((o.t - from) / BIN_S) as usize).min(n - 1);
        bins[i].push(o);
    }
    // Each bin: the most common scene among its instants (a bin has about two).
    let labels: Vec<Scene> = bins
        .iter()
        .map(|b| {
            let mut instants: Vec<(f64, Vec<&FaceObs>)> = Vec::new();
            for o in b {
                match instants.iter_mut().find(|(t, _)| (t - o.t).abs() < 1e-3) {
                    Some((_, v)) => v.push(o),
                    None => instants.push((o.t, vec![*o])),
                }
            }
            let votes: Vec<Scene> = instants.iter().map(|(_, f)| classify(f)).collect();
            majority(&votes).unwrap_or(Scene::NoFace)
        })
        .collect();
    (bins, labels)
}

pub fn segment(obs: &[FaceObs], from: f64, to: f64) -> Vec<SceneSpan> {
    let (bins, labels) = bin_labels(obs, from, to);
    let n = labels.len();
    if n == 0 {
        return Vec::new();
    }
    // Smooth over one bin either side, so a single missed detection doesn't flicker.
    let smoothed: Vec<Scene> = (0..n)
        .map(|i| {
            let win: Vec<Scene> = labels[i.saturating_sub(1)..(i + 2).min(n)].to_vec();
            match majority(&win) {
                Some(s) if win.iter().filter(|&&x| x == s).count() * 2 > win.len() => s,
                _ => labels[i],
            }
        })
        .collect();

    let mut runs: Vec<(usize, usize, Scene)> = Vec::new();
    for (i, &s) in smoothed.iter().enumerate() {
        match runs.last_mut() {
            Some(r) if r.2 == s => r.1 = i + 1,
            _ => runs.push((i, i + 1, s)),
        }
    }
    // Fold short runs into a neighbour (the longer one), shortest first, until none is left.
    let min_bins = (MIN_SCENE_S / BIN_S).ceil() as usize;
    loop {
        if runs.len() <= 1 {
            break;
        }
        let Some((k, _)) = runs.iter().enumerate().filter(|(_, r)| r.1 - r.0 < min_bins).min_by_key(|(_, r)| r.1 - r.0)
        else {
            break;
        };
        let prev = k.checked_sub(1).map(|p| runs[p].1 - runs[p].0);
        let next = runs.get(k + 1).map(|r| r.1 - r.0);
        let into_prev = match (prev, next) {
            (Some(p), Some(nx)) => p >= nx,
            (Some(_), None) => true,
            _ => false,
        };
        let (a, b) = (runs[k].0, runs[k].1);
        runs.remove(k);
        if into_prev {
            runs[k - 1].1 = b;
        } else {
            runs[k].0 = a;
        }
        // Neighbours that now touch with the same scene become one.
        let mut merged: Vec<(usize, usize, Scene)> = Vec::new();
        for r in runs.drain(..) {
            match merged.last_mut() {
                Some(m) if m.2 == r.2 => m.1 = r.1,
                _ => merged.push(r),
            }
        }
        runs = merged;
    }

    runs.into_iter()
        .map(|(a, b, scene)| {
            let start = from + a as f64 * BIN_S;
            let end = if b >= n { to } else { from + b as f64 * BIN_S };
            let inside: Vec<&FaceObs> = bins[a..b].iter().flatten().copied().filter(|o| o.h >= MIN_FACE_H).collect();
            let mut per_instant: Vec<(f64, usize, f64)> = Vec::new();
            for o in &inside {
                match per_instant.iter_mut().find(|(t, _, _)| (t - o.t).abs() < 1e-3) {
                    Some((_, c, h)) => {
                        *c += 1;
                        *h = h.max(o.h);
                    }
                    None => per_instant.push((o.t, 1, o.h)),
                }
            }
            let instants = ((end - start) * crate::framing::SAMPLE_HZ).max(1.0);
            SceneSpan {
                start: round3(start),
                end: round3(end),
                scene,
                faces: round2(per_instant.iter().map(|p| p.1 as f64).sum::<f64>() / instants),
                face_h: round3(median(per_instant.iter().map(|p| p.2).collect())),
            }
        })
        .collect()
}

fn majority(v: &[Scene]) -> Option<Scene> {
    // Ties go to the scene with people in it, then to the first seen.
    let order = [Scene::TwoShot, Scene::SingleFace, Scene::ScreenShare, Scene::NoFace];
    order.iter().copied().filter(|s| v.contains(s)).max_by_key(|s| {
        let count = v.iter().filter(|x| *x == s).count();
        (count, std::cmp::Reverse(order.iter().position(|o| o == s)))
    })
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// The span covering `t`, if any.
pub fn at(spans: &[SceneSpan], t: f64) -> Option<&SceneSpan> {
    spans.iter().find(|s| t >= s.start && t < s.end).or_else(|| spans.last().filter(|s| (t - s.end).abs() < 1e-6))
}

/// The scene mix of `from..to`.
pub fn mix(spans: &[SceneSpan], from: f64, to: f64) -> SceneMix {
    let len = (to - from).max(0.0);
    let mut m = SceneMix::default();
    if len <= 0.0 {
        return m;
    }
    let mut covered = 0.0;
    for s in spans {
        let ov = (s.end.min(to) - s.start.max(from)).max(0.0);
        covered += ov;
        match s.scene {
            Scene::TwoShot => m.two_shot += ov,
            Scene::SingleFace => m.single_face += ov,
            Scene::ScreenShare => m.screen_share += ov,
            Scene::NoFace => m.no_face += ov,
        }
    }
    let total = covered.max(1e-9);
    if covered <= 0.0 {
        return m;
    }
    SceneMix {
        two_shot: round2(m.two_shot / total),
        single_face: round2(m.single_face / total),
        screen_share: round2(m.screen_share / total),
        no_face: round2(m.no_face / total),
    }
}

/// Whether the faces stage has looked at this video (found faces, or found none).
pub fn faces_analysed(db: &Db, video_id: i64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM face_tracks WHERE video_id = ?1)
                 OR EXISTS (SELECT 1 FROM jobs WHERE video_id = ?1 AND stage = 'faces' AND state = 'done')",
            [video_id],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false)
}

/// Every face box of `from..to` of a video, normalised to the frame, on the source clock.
pub fn load_obs(db: &Db, video_id: i64, from: f64, to: f64) -> Result<Vec<FaceObs>, Error> {
    let mut st = db.conn.prepare_cached(
        "SELECT fb.t_s, fb.x, fb.y, fb.w, fb.h FROM face_boxes fb JOIN face_tracks ft ON ft.id = fb.track_id
          WHERE ft.video_id = ?1 AND fb.t_s >= ?2 AND fb.t_s < ?3 ORDER BY fb.t_s",
    )?;
    let rows = st.query_map(rusqlite::params![video_id, from, to], |r| {
        let (t, x, y, w, h): (f64, f64, f64, f64, f64) = (r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?);
        Ok(FaceObs { t, cx: x + w / 2.0, cy: y + h / 2.0, w, h })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// What most of `a..b` shows when it is not faces: `"screen"` for a screen share, `"no face"` when
/// nobody is on screen. `None` when faces fill most of it, or nothing covers it.
pub fn picture_label(spans: &[SceneSpan], a: f64, b: f64) -> Option<&'static str> {
    let len = (b - a).max(1e-6);
    let share = |kind: Scene| -> f64 {
        spans.iter().filter(|s| s.scene == kind).map(|s| (s.end.min(b) - s.start.max(a)).max(0.0)).sum::<f64>() / len
    };
    if share(Scene::ScreenShare) > 0.5 {
        Some("screen")
    } else if share(Scene::NoFace) > 0.5 {
        Some("no face")
    } else {
        None
    }
}

/// The share of `a..b` with a face on screen (a single face or a two-shot), 0-1.
pub fn face_share(spans: &[SceneSpan], a: f64, b: f64) -> f64 {
    let len = (b - a).max(1e-6);
    spans
        .iter()
        .filter(|s| matches!(s.scene, Scene::SingleFace | Scene::TwoShot))
        .map(|s| (s.end.min(b) - s.start.max(a)).max(0.0))
        .sum::<f64>()
        / len
}

/// The scenes of `from..to` of a video on the source clock; `None` when its faces were never
/// looked for (then nothing is known, which is not the same as "no faces").
///
/// The whole video is segmented and the spans clipped to the range: segmenting only the range
/// smooths differently near its edges, so two tools asking about the same moment over different
/// ranges were told different scene edges.
pub fn for_range(db: &Db, video_id: i64, from: f64, to: f64) -> Result<Option<Vec<SceneSpan>>, Error> {
    if !faces_analysed(db, video_id) {
        return Ok(None);
    }
    let duration: f64 = db
        .conn
        .query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [video_id], |r| r.get(0))
        .unwrap_or(0.0);
    let end = if duration > 0.0 { duration } else { to };
    let all = segment(&load_obs(db, video_id, 0.0, end)?, 0.0, end);
    Ok(Some(
        all.into_iter()
            .filter(|s| s.end > from && s.start < to)
            .map(|s| SceneSpan { start: s.start.max(from), end: s.end.min(to), ..s })
            .filter(|s| !s.is_empty())
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stretch_is_labelled_by_the_picture_most_of_it_shows() {
        let span = |start, end, scene| SceneSpan { start, end, scene, faces: 1.0, face_h: 0.3 };
        let spans =
            [span(0.0, 10.0, Scene::TwoShot), span(10.0, 20.0, Scene::ScreenShare), span(20.0, 25.0, Scene::NoFace)];
        assert_eq!(picture_label(&spans, 2.0, 8.0), None, "faces need no label");
        assert_eq!(picture_label(&spans, 9.0, 15.0), Some("screen"));
        assert_eq!(picture_label(&spans, 21.0, 24.0), Some("no face"));
        assert!((face_share(&spans, 0.0, 20.0) - 0.5).abs() < 1e-9);
    }

    /// A face every 0.2 s over `a..b`.
    fn faces(a: f64, b: f64, cx: f64, cy: f64, h: f64) -> Vec<FaceObs> {
        let n = ((b - a) * 5.0).round() as usize;
        (0..n).map(|i| FaceObs { t: a + i as f64 * 0.2, cx, cy, w: h * 0.6, h }).collect()
    }

    #[test]
    fn classifies_two_shot_screen_share_single_and_empty() {
        let mut obs = Vec::new();
        // 0–10 s: two hosts side by side, big faces.
        obs.extend(faces(0.0, 10.0, 0.25, 0.4, 0.45));
        obs.extend(faces(0.0, 10.0, 0.75, 0.4, 0.45));
        // 10–20 s: a screen share with two camera tiles at the bottom right.
        obs.extend(faces(10.0, 20.0, 0.72, 0.88, 0.076));
        obs.extend(faces(10.0, 20.0, 0.90, 0.88, 0.07));
        // 20–25 s: nobody. 25–30 s: one person full frame.
        obs.extend(faces(25.0, 30.0, 0.5, 0.4, 0.4));
        let spans = segment(&obs, 0.0, 30.0);
        let kinds: Vec<Scene> = spans.iter().map(|s| s.scene).collect();
        assert_eq!(kinds, vec![Scene::TwoShot, Scene::ScreenShare, Scene::NoFace, Scene::SingleFace], "{spans:?}");
        assert!((spans[1].start - 10.0).abs() < 0.41 && (spans[1].end - 20.0).abs() < 0.41, "{spans:?}");
        assert_eq!(spans.first().unwrap().start, 0.0);
        assert_eq!(spans.last().unwrap().end, 30.0);
        assert!(spans.windows(2).all(|w| w[0].end == w[1].start));
        assert!((spans[0].faces - 2.0).abs() < 0.1 && spans[0].face_h > 0.4);
        let m = mix(&spans, 0.0, 20.0);
        assert!((m.two_shot - 0.5).abs() < 0.03 && (m.screen_share - 0.5).abs() < 0.03, "{m:?}");
        assert!(m.summary().starts_with("two-shot 5") || m.summary().starts_with("screen share 5"));
    }

    #[test]
    fn short_glitches_fold_into_their_neighbours() {
        let mut obs = Vec::new();
        // A two-shot where one face is lost for 0.8 s, and a one-second transition face.
        obs.extend(faces(0.0, 6.0, 0.25, 0.4, 0.45));
        obs.extend(faces(6.8, 12.0, 0.25, 0.4, 0.45));
        obs.extend(faces(0.0, 12.0, 0.75, 0.4, 0.45));
        obs.extend(faces(12.0, 13.0, 0.7, 0.74, 0.18));
        obs.extend(faces(13.0, 20.0, 0.72, 0.88, 0.076));
        let spans = segment(&obs, 0.0, 20.0);
        let kinds: Vec<Scene> = spans.iter().map(|s| s.scene).collect();
        assert_eq!(kinds, vec![Scene::TwoShot, Scene::ScreenShare], "{spans:?}");
        assert!(spans.iter().all(|s| s.len() >= MIN_SCENE_S));
    }

    #[test]
    fn small_faces_in_the_middle_are_a_wide_shot_not_a_screen_share() {
        let mut obs = faces(0.0, 5.0, 0.4, 0.45, 0.09);
        obs.extend(faces(0.0, 5.0, 0.6, 0.45, 0.09));
        assert_eq!(segment(&obs, 0.0, 5.0)[0].scene, Scene::TwoShot);
        assert_eq!(segment(&[], 0.0, 5.0)[0].scene, Scene::NoFace);
        assert!(segment(&[], 3.0, 3.0).is_empty());
    }

    #[test]
    fn camera_tiles_over_a_shared_screen_are_a_screen_share_whatever_the_screen_shows() {
        // Two tall participant tiles stacked down the right side, and faces in the shared content.
        let mut obs = faces(0.0, 10.0, 0.80, 0.22, 0.23);
        obs.extend(faces(0.0, 10.0, 0.82, 0.71, 0.23));
        obs.extend(faces(0.0, 10.0, 0.30, 0.72, 0.08));
        obs.extend(faces(0.0, 10.0, 0.16, 0.67, 0.11));
        assert_eq!(segment(&obs, 0.0, 10.0)[0].scene, Scene::ScreenShare);
        // Small tiles in a row at the bottom, with bigger faces on the shared screen: the same.
        let mut obs = faces(0.0, 10.0, 0.72, 0.88, 0.07);
        obs.extend(faces(0.0, 10.0, 0.90, 0.88, 0.07));
        obs.extend(faces(0.0, 10.0, 0.50, 0.52, 0.26));
        obs.extend(faces(0.0, 10.0, 0.28, 0.53, 0.27));
        assert_eq!(segment(&obs, 0.0, 10.0)[0].scene, Scene::ScreenShare);
        // Small faces in the middle among small tiles: still the tiles decide.
        let mut obs = faces(0.0, 10.0, 0.72, 0.87, 0.07);
        obs.extend(faces(0.0, 10.0, 0.90, 0.88, 0.07));
        obs.extend(faces(0.0, 10.0, 0.45, 0.41, 0.07));
        assert_eq!(segment(&obs, 0.0, 10.0)[0].scene, Scene::ScreenShare);
        // One person framed with thumbnails beside them is that person, not a screen.
        let mut obs = faces(0.0, 10.0, 0.5, 0.4, 0.4);
        obs.extend(faces(0.0, 10.0, 0.72, 0.88, 0.07));
        obs.extend(faces(0.0, 10.0, 0.90, 0.88, 0.07));
        assert_eq!(segment(&obs, 0.0, 10.0)[0].scene, Scene::SingleFace);
        // Two hosts side by side stay a two-shot, tall as they are.
        let mut obs = faces(0.0, 10.0, 0.23, 0.42, 0.45);
        obs.extend(faces(0.0, 10.0, 0.74, 0.40, 0.44));
        assert_eq!(segment(&obs, 0.0, 10.0)[0].scene, Scene::TwoShot);
    }

    #[test]
    fn a_short_screen_share_survives_a_long_range_as_it_does_a_narrow_one() {
        // Half an hour of two hosts, with an 8 s screen share and a 3.4 s one inside it.
        let mut obs = Vec::new();
        for (a, b, share) in [
            (0.0, 900.0, false),
            (900.0, 908.0, true),
            (908.0, 1200.0, false),
            (1200.0, 1203.4, true),
            (1203.4, 1800.0, false),
        ] {
            if share {
                obs.extend(faces(a, b, 0.72, 0.88, 0.07));
                obs.extend(faces(a, b, 0.90, 0.88, 0.07));
            } else {
                obs.extend(faces(a, b, 0.23, 0.42, 0.45));
                obs.extend(faces(a, b, 0.74, 0.40, 0.44));
            }
        }
        let long = segment(&obs, 0.0, 1800.0);
        let shares = |spans: &[SceneSpan]| -> Vec<(f64, f64)> {
            spans.iter().filter(|s| s.scene == Scene::ScreenShare).map(|s| (s.start, s.end)).collect()
        };
        let long_shares = shares(&long);
        assert_eq!(long_shares.len(), 2, "{long:?}");
        for ((s, e), (want_s, want_e)) in long_shares.iter().zip([(900.0, 908.0), (1200.0, 1203.4)]) {
            assert!((s - want_s).abs() <= BIN_S && (e - want_e).abs() <= BIN_S, "{long_shares:?}");
        }
        // The same stretches when only a minute around each is asked for.
        for (from, to, want) in [(870.0, 940.0, (900.0, 908.0)), (1170.0, 1240.0, (1200.0, 1203.4))] {
            let narrow = shares(&segment(&obs, from, to));
            assert_eq!(narrow.len(), 1, "{narrow:?}");
            assert!((narrow[0].0 - want.0).abs() <= BIN_S && (narrow[0].1 - want.1).abs() <= BIN_S, "{narrow:?}");
        }
        // And the mix over the long range counts them.
        let m = mix(&long, 0.0, 1800.0);
        assert!((m.screen_share - 11.4 / 1800.0).abs() < 0.01, "{m:?}");
    }

    #[test]
    fn absolute_clock_and_lookup() {
        let obs = faces(100.0, 110.0, 0.5, 0.4, 0.4);
        let spans = segment(&obs, 98.0, 110.0);
        assert_eq!(spans.first().unwrap().start, 98.0);
        assert_eq!(at(&spans, 105.0).unwrap().scene, Scene::SingleFace);
        assert_eq!(at(&spans, 110.0).unwrap().scene, Scene::SingleFace, "the end belongs to the last span");
        assert!(at(&spans, 120.0).is_none());
    }
}
