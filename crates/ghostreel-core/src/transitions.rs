//! Changes of picture measured from the pixels: hard cuts, dissolves and layout animations in the
//! source, each with the span it takes.
//!
//! The scene spans (`scenes`) come from face detections, which carry on through a crossfade, so a
//! change of picture there is a single instant, and the wrong one by up to half a second. A clip
//! edge placed "on" that instant can open on the tail of a dissolve or close on the start of one,
//! and nothing said so until someone watched the render. This reads the frames themselves: how much
//! the picture changes from one frame to the next, at low resolution, around the moment asked about.
//! It is measured on demand (a few seconds of video around each edge), and remembered for the life
//! of the process.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use serde::Serialize;

use crate::db::Db;

/// A change of picture in the source: `start == end` for a hard cut, a span for a dissolve or an
/// animated change of layout.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Transition {
    pub start: f64,
    pub end: f64,
}

impl Transition {
    pub fn is_cut(&self) -> bool {
        self.end - self.start < 1.5 / FPS
    }
}

/// Frames looked at per second, and their size.
const FPS: f64 = 15.0;
const W: usize = 64;
const H: usize = 36;
/// A frame that differs from the one before by more than this share (0–1 of the full range, per
/// pixel on average) is part of a change of picture. Measured on a podcast: a talking two-shot moves
/// under 0.008 between frames, a dissolve 0.02–0.09, a hard cut 0.1–0.5.
const CHANGE: f64 = 0.018;
/// And somewhere in the run the picture's tones must shift by this much (half the L1 distance of
/// 16-bin luma histograms, 0–1): a hand moving in front of a face changes pixels as much as a
/// dissolve does, but not the tones (0.03 against 0.29 for the dissolve, 0.6 for a hard cut).
const TONE_SHIFT: f64 = 0.1;
/// Around a moment, how far to look either side (s).
const RADIUS_S: f64 = 1.5;

/// One frame's change from the one before: per-pixel (0–1) and in its tones (0–1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Change {
    pub pixels: f64,
    pub tones: f64,
}

/// Frame-to-frame change over `from..to` of `video`: one value per frame after the first, at time
/// `from + (i + 1) / FPS`.
pub fn measure(ffmpeg: &Path, video: &Path, from: f64, to: f64) -> Option<Vec<Change>> {
    let out = crate::proc::std_command(ffmpeg)
        .args(["-v", "error", "-ss", &format!("{:.3}", from.max(0.0)), "-t", &format!("{:.3}", (to - from).max(0.1))])
        .arg("-i")
        .arg(video)
        .args(["-vf", &format!("fps={FPS},scale={W}:{H},format=gray"), "-f", "rawvideo", "-"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let histogram = |f: &[u8]| {
        let mut h = [0u32; 16];
        for v in f {
            h[usize::from(v >> 4)] += 1;
        }
        h
    };
    #[allow(clippy::chunks_exact_to_as_chunks)] // one frame per chunk; frames are compared by slice
    let frames: Vec<&[u8]> = out.stdout.chunks_exact(W * H).collect();
    Some(
        frames
            .windows(2)
            .map(|p| {
                let sum: u64 = p[0].iter().zip(p[1]).map(|(a, b)| u64::from(a.abs_diff(*b))).sum();
                let (ha, hb) = (histogram(p[0]), histogram(p[1]));
                let shift: u32 = ha.iter().zip(hb).map(|(a, b)| a.abs_diff(b)).sum();
                Change { pixels: sum as f64 / (W * H * 255) as f64, tones: f64::from(shift) / (2 * W * H) as f64 }
            })
            .collect(),
    )
}

/// The changes of picture in a run of frame-to-frame changes starting at `from` (see [`measure`]). A
/// frame counts when its pixels change more than [`CHANGE`], or three times the window's median (a
/// busy shot moves more); runs one quiet frame apart are one change; and a run counts only when the
/// tones shift by [`TONE_SHIFT`] somewhere in it.
pub fn find(changes: &[Change], from: f64) -> Vec<Transition> {
    if changes.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = changes.iter().map(|c| c.pixels).collect();
    sorted.sort_by(f64::total_cmp);
    let threshold = CHANGE.max(3.0 * sorted[sorted.len() / 2]);
    let at = |i: usize| from + (i + 1) as f64 / FPS;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (i, c) in changes.iter().enumerate() {
        if c.pixels <= threshold {
            continue;
        }
        match runs.last_mut() {
            Some(run) if i <= run.1 + 2 => run.1 = i,
            _ => runs.push((i, i)),
        }
    }
    // A change starts on the last frame that was still the old picture.
    runs.into_iter()
        .filter(|&(a, b)| changes[a..=b].iter().any(|c| c.tones >= TONE_SHIFT))
        .map(|(a, b)| Transition { start: round2(at(a) - 1.0 / FPS), end: round2(at(b)) })
        .map(|t| if t.end - t.start < 1.5 / FPS { Transition { start: t.end, end: t.end } } else { t })
        .collect()
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

type Cache = Mutex<HashMap<(i64, i64), Vec<Transition>>>;

fn cache() -> &'static Cache {
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// The changes of picture within [`RADIUS_S`] of `t` in a video, measured from its file (empty when
/// the file or ffmpeg is not there). Remembered per tenth of a second.
pub fn around(db: &Db, video_id: i64, t: f64) -> Vec<Transition> {
    match crate::framing::source_path(db, video_id).ok().flatten() {
        Some(path) => around_file(video_id, &path, t),
        None => Vec::new(),
    }
}

/// [`around`] for a file already looked up: needs no database, so many edges can be measured on
/// threads at once.
pub fn around_file(video_id: i64, video: &Path, t: f64) -> Vec<Transition> {
    let key = (video_id, (t * 10.0).round() as i64);
    if let Some(hit) = cache().lock().ok().and_then(|c| c.get(&key).cloned()) {
        return hit;
    }
    let found = (|| {
        let ffmpeg = crate::doctor::locate("ffmpeg")?;
        if !video.is_file() {
            return None;
        }
        let from = (t - RADIUS_S).max(0.0);
        Some(find(&measure(&ffmpeg, video, from, t + RADIUS_S)?, from))
    })()
    .unwrap_or_default();
    if let Ok(mut c) = cache().lock() {
        c.insert(key, found.clone());
    }
    found
}

/// The changes of picture around many `(video_id, t)` moments, measured a few at a time in
/// parallel (decoding is the cost: a 4K source takes about a second per moment).
pub fn around_many(db: &Db, moments: &[(i64, f64)]) -> HashMap<(i64, i64), Vec<Transition>> {
    let mut paths: HashMap<i64, Option<std::path::PathBuf>> = HashMap::new();
    for &(id, _) in moments {
        paths.entry(id).or_insert_with(|| crate::framing::source_path(db, id).ok().flatten());
    }
    let jobs: Vec<(i64, f64, std::path::PathBuf)> =
        moments.iter().filter_map(|&(id, t)| Some((id, t, paths.get(&id)?.clone()?))).collect();
    let out = Mutex::new(HashMap::new());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 6);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers.min(jobs.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((id, t, path)) = jobs.get(i) else { break };
                    let found = around_file(*id, path, *t);
                    if let Ok(mut o) = out.lock() {
                        o.insert((*id, (*t * 10.0).round() as i64), found);
                    }
                }
            });
        }
    });
    out.into_inner().unwrap_or_default()
}

/// How a clip edge sits against the changes of picture near it, when it is in trouble: an edge
/// inside a dissolve, or a clip that opens just before a change (a flash of the old picture) or
/// closes just after one (a flash of the new). `opening` is true for an in point.
pub fn edge_hazard(transitions: &[Transition], edge: f64, opening: bool) -> Option<Transition> {
    /// How close to an edge a change still shows as a flash (s).
    const FLASH_S: f64 = 0.6;
    transitions.iter().copied().find(|t| {
        let inside = t.start < edge - 0.02 && t.end > edge + 0.02;
        let flash = if opening {
            t.start >= edge - 0.02 && t.start < edge + FLASH_S
        } else {
            t.end <= edge + 0.02 && t.end > edge - FLASH_S
        };
        // A hard cut right on the edge is the cleanest place to cut: nothing of either side shows.
        let on_cut = t.is_cut() && (t.start - edge).abs() < 1.0 / FPS;
        (inside || flash) && !on_cut
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changes(pixels: &[f64], tones: &[f64]) -> Vec<Change> {
        pixels.iter().zip(tones).map(|(&p, &t)| Change { pixels: p, tones: t }).collect()
    }

    #[test]
    fn a_dissolve_is_a_span_and_a_hard_cut_an_instant() {
        // 0.0–1.0 s quiet, a hard cut, quiet, then a dissolve decaying over half a second
        // (the values measured on the podcast's dissolve at 15:30).
        let mut p = vec![0.005; 15];
        let mut t = vec![0.01; 15];
        p.push(0.4);
        t.push(0.6);
        p.extend(vec![0.006; 10]);
        t.extend(vec![0.01; 10]);
        p.extend([0.09, 0.07, 0.05, 0.03, 0.025, 0.021, 0.019]);
        t.extend([0.29, 0.26, 0.17, 0.11, 0.06, 0.03, 0.03]);
        p.extend(vec![0.005; 10]);
        t.extend(vec![0.01; 10]);
        let found = find(&changes(&p, &t), 10.0);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].is_cut(), "{found:?}");
        assert!((found[0].start - (10.0 + 16.0 / FPS)).abs() < 0.01, "{found:?}");
        assert!(!found[1].is_cut() && found[1].end - found[1].start > 0.4, "{found:?}");
    }

    #[test]
    fn talking_heads_moving_are_no_change() {
        let p: Vec<f64> = (0..45).map(|i| 0.004 + 0.003 * ((i % 5) as f64 / 4.0)).collect();
        assert!(find(&changes(&p, &vec![0.015; 45]), 0.0).is_empty());
        // A hand in front of a face: as many pixels change as in a dissolve, the tones do not.
        let mut p = vec![0.005; 20];
        p.extend([0.028, 0.016, 0.026]);
        p.extend(vec![0.005; 20]);
        let mut t = vec![0.012; 20];
        t.extend([0.032, 0.024, 0.032]);
        t.extend(vec![0.012; 20]);
        assert!(find(&changes(&p, &t), 0.0).is_empty());
    }

    #[test]
    fn an_edge_in_a_dissolve_or_just_before_a_change_is_a_hazard_but_one_on_a_hard_cut_is_not() {
        let dissolve = Transition { start: 930.0, end: 930.8 };
        assert!(edge_hazard(&[dissolve], 930.3, true).is_some(), "inside");
        assert!(edge_hazard(&[dissolve], 929.6, true).is_some(), "opens 0.4 s before it: a flash of the old picture");
        assert!(edge_hazard(&[dissolve], 931.2, true).is_none(), "opens after it");
        assert!(edge_hazard(&[dissolve], 930.9, false).is_some(), "closes just after it: a flash of the new");
        let cut = Transition { start: 50.0, end: 50.0 };
        assert!(edge_hazard(&[cut], 50.02, true).is_none(), "on the cut");
        assert!(edge_hazard(&[cut], 49.7, true).is_some(), "0.3 s of the old shot first");
    }
}
