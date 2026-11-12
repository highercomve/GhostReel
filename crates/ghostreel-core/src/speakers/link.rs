//! Active-speaker link: which person's face moves its mouth when a voice speaks.
//!
//! Frames are decoded again at the face stage's 5 fps (grayscale, 640 px wide). For each stored
//! face box, a patch around the mouth (between the mouth-corner landmarks, extended down to the
//! chin) is sampled on a fixed grid, so its size does not depend on how big the face is. The mean
//! absolute difference to the same track's patch one frame earlier is that person's mouth motion.
//! Each person's motion series, smoothed over a second, is correlated with each speaker's speech
//! activity; the best-correlated pairs are linked.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use crate::Error;

pub const FPS: f64 = 5.0;
const DECODE_W: usize = 640;
const GRID_W: usize = 24;
const GRID_H: usize = 16;
/// Below this mouth width (decoded pixels) the patch is mostly blur.
const MIN_MOUTH_PX: f32 = 5.0;
/// Fewer overlapping frames than this (10 s) make a correlation meaningless.
const MIN_FRAMES: usize = 50;
/// A link needs at least this correlation, unless it is the only face and the only voice.
pub const MIN_CORR: f32 = 0.02;

#[derive(Debug, Clone)]
pub struct MouthBox {
    pub t_s: f64,
    pub track_id: i64,
    pub person_id: i64,
    /// YuNet landmarks normalised to the frame: right eye, left eye, nose, right/left mouth corner.
    pub landmarks: [(f32, f32); 5],
}

/// Decoded frame height for a source of `w`×`h`, even, at 640 px wide.
fn decode_height(w: u32, h: u32) -> usize {
    if w == 0 || h == 0 {
        return 360;
    }
    let raw = DECODE_W as f64 * h as f64 / w as f64;
    ((raw / 2.0).round() as usize * 2).clamp(2, 4096)
}

#[allow(clippy::too_many_arguments)]
/// Sample a GRID_W×GRID_H patch centred on (cx, cy) of size (pw, ph) pixels.
fn sample_patch(img: &[u8], w: usize, h: usize, cx: f32, cy: f32, pw: f32, ph: f32, out: &mut [f32]) {
    for gy in 0..GRID_H {
        let y = (cy - ph / 2.0 + (gy as f32 + 0.5) / GRID_H as f32 * ph).clamp(0.0, (h - 1) as f32) as usize;
        for gx in 0..GRID_W {
            let x = (cx - pw / 2.0 + (gx as f32 + 0.5) / GRID_W as f32 * pw).clamp(0.0, (w - 1) as f32) as usize;
            out[gy * GRID_W + gx] = img[y * w + x] as f32;
        }
    }
}

/// The mouth patch of a face, or None when the mouth is too small to see.
pub fn mouth_patch(img: &[u8], w: usize, h: usize, lm: &[(f32, f32); 5]) -> Option<Vec<f32>> {
    let (rx, ry) = (lm[3].0 * w as f32, lm[3].1 * h as f32);
    let (lx, ly) = (lm[4].0 * w as f32, lm[4].1 * h as f32);
    let mw = ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt();
    if !mw.is_finite() || mw < MIN_MOUTH_PX {
        return None;
    }
    let cx = (rx + lx) / 2.0;
    let cy = (ry + ly) / 2.0 + 0.25 * mw;
    let mut out = vec![0.0f32; GRID_W * GRID_H];
    sample_patch(img, w, h, cx, cy, 1.6 * mw, 1.2 * mw, &mut out);
    Some(out)
}

/// Mouth motion per person: `(frame index at 5 fps, energy)`, from boxes sorted by time.
pub fn mouth_motion(
    ffmpeg: &Path,
    video: &Path,
    src_w: u32,
    src_h: u32,
    boxes: &[MouthBox],
    cancel: Option<&Arc<AtomicBool>>,
) -> Result<HashMap<i64, Vec<(usize, f32)>>, Error> {
    let mut by_frame: HashMap<usize, Vec<&MouthBox>> = HashMap::new();
    for b in boxes {
        by_frame.entry((b.t_s * FPS).round() as usize).or_default().push(b);
    }
    let last_frame = by_frame.keys().copied().max().unwrap_or(0);
    let mut motion: HashMap<i64, Vec<(usize, f32)>> = HashMap::new();
    if by_frame.is_empty() {
        return Ok(motion);
    }
    let h = decode_height(src_w, src_h);
    let w = DECODE_W;
    let mut cmd = crate::proc::std_command(ffmpeg);
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"]).arg(video);
    cmd.args(["-map", "0:v:0", "-an", "-vf", &format!("fps=5,scale={w}:{h}"), "-f", "rawvideo", "-pix_fmt", "gray"]);
    cmd.arg("pipe:1");
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    let mut child = cmd.spawn().map_err(|e| Error::Io(video.to_path_buf(), e))?;
    let stdout = child.stdout.take().ok_or_else(|| Error::Invalid("failed to capture ffmpeg stdout".into()))?;
    let mut reader = std::io::BufReader::with_capacity(4 * 1024 * 1024, stdout);
    let mut frame = vec![0u8; w * h];
    let mut prev: HashMap<i64, (usize, Vec<f32>)> = HashMap::new();
    let mut idx = 0usize;
    loop {
        if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Cancelled);
        }
        if idx > last_frame + 1 {
            break;
        }
        match reader.read_exact(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Io(video.to_path_buf(), e));
            }
        }
        if let Some(list) = by_frame.get(&idx) {
            for b in list {
                let Some(patch) = mouth_patch(&frame, w, h, &b.landmarks) else { continue };
                if let Some((pf, pp)) = prev.get(&b.track_id)
                    && *pf + 1 == idx
                {
                    let e = patch.iter().zip(pp).map(|(a, b)| (a - b).abs()).sum::<f32>() / patch.len() as f32;
                    motion.entry(b.person_id).or_default().push((idx, e));
                }
                prev.insert(b.track_id, (idx, patch));
            }
        }
        idx += 1;
    }
    let _ = child.kill();
    let _ = child.wait();
    for series in motion.values_mut() {
        // Two tracks of one person in a frame (rare): keep the mean.
        series.sort_by_key(|(f, _)| *f);
        let mut merged: Vec<(usize, f32, u32)> = Vec::with_capacity(series.len());
        for &(f, e) in series.iter() {
            match merged.last_mut() {
                Some((lf, le, n)) if *lf == f => {
                    *le += e;
                    *n += 1;
                }
                _ => merged.push((f, e, 1)),
            }
        }
        *series = merged.into_iter().map(|(f, e, n)| (f, e / n as f32)).collect();
    }
    Ok(motion)
}

/// Pearson correlation between a person's mouth motion (smoothed over ±2 frames, kept where at
/// least 3 of the 5 frames have a value) and a 0/1 speech-activity series indexed by frame.
pub fn motion_activity_correlation(motion: &[(usize, f32)], activity: &[f32]) -> Option<f32> {
    let &(last, _) = motion.last()?;
    let len = (last + 1).max(activity.len());
    let mut sum = vec![0.0f32; len];
    let mut cnt = vec![0u8; len];
    for &(f, e) in motion {
        sum[f] = e;
        cnt[f] = 1;
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for f in 0..len {
        let (lo, hi) = (f.saturating_sub(2), (f + 2).min(len - 1));
        let n: u32 = cnt[lo..=hi].iter().map(|&c| c as u32).sum();
        if n >= 3 && cnt[f] == 1 {
            xs.push(sum[lo..=hi].iter().sum::<f32>() / n as f32);
            ys.push(activity.get(f).copied().unwrap_or(0.0));
        }
    }
    pearson(&xs, &ys)
}

pub fn pearson(xs: &[f32], ys: &[f32]) -> Option<f32> {
    let n = xs.len();
    if n < MIN_FRAMES || n != ys.len() {
        return None;
    }
    let mx = xs.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let my = ys.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in xs.iter().zip(ys) {
        let (dx, dy) = (*x as f64 - mx, *y as f64 - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 1e-12 || syy <= 1e-12 {
        return None;
    }
    Some((sxy / (sxx.sqrt() * syy.sqrt())) as f32)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LinkDecision {
    pub speaker: i64,
    pub person_id: i64,
    pub correlation: Option<f32>,
    pub confidence: f32,
}

/// Link speakers to people from `corr[speaker][person]` (None = not enough overlap).
///
/// Greedy on the strongest correlation, one person per speaker and vice versa. Confidence is
/// tanh(5 × margin), where the margin is how far the pair's correlation beats the best rival
/// (the speaker with another person, or the person with another speaker; 0 when there is none).
/// A video with a single face and a single voice links them trivially.
pub fn assign_links(speakers: &[i64], persons: &[i64], corr: &[Vec<Option<f32>>]) -> Vec<LinkDecision> {
    if speakers.len() == 1 && persons.len() == 1 {
        let c = corr.first().and_then(|r| r.first()).copied().flatten();
        let confidence = match c {
            Some(v) if v < 0.0 => 0.5,
            _ => 0.8,
        };
        return vec![LinkDecision { speaker: speakers[0], person_id: persons[0], correlation: c, confidence }];
    }
    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
    for (si, row) in corr.iter().enumerate() {
        for (pi, c) in row.iter().enumerate() {
            if let Some(c) = c
                && *c >= MIN_CORR
            {
                pairs.push((si, pi, *c));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut used_s = vec![false; speakers.len()];
    let mut used_p = vec![false; persons.len()];
    let mut out = Vec::new();
    for (si, pi, c) in pairs {
        if used_s[si] || used_p[pi] {
            continue;
        }
        used_s[si] = true;
        used_p[pi] = true;
        let rival_p = (0..persons.len()).filter(|&q| q != pi).filter_map(|q| corr[si][q]).fold(0.0f32, f32::max);
        let rival_s = (0..speakers.len()).filter(|&t| t != si).filter_map(|t| corr[t][pi]).fold(0.0f32, f32::max);
        let margin = c - rival_p.max(rival_s);
        let confidence = (5.0 * margin).tanh().clamp(0.0, 1.0);
        out.push(LinkDecision { speaker: speakers[si], person_id: persons[pi], correlation: Some(c), confidence });
    }
    out.sort_by_key(|d| d.speaker);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_follows_the_talking_mouth() {
        // Speaker 0 talks in the first half of every 20-frame cycle, speaker 1 in the second.
        let n = 400;
        let act0: Vec<f32> = (0..n).map(|f| if (f / 10) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let act1: Vec<f32> = act0.iter().map(|a| 1.0 - a).collect();
        let mouth_a: Vec<(usize, f32)> = (0..n).map(|f| (f, 2.0 + 6.0 * act0[f] + (f % 3) as f32)).collect();
        let mouth_b: Vec<(usize, f32)> = (0..n).map(|f| (f, 2.0 + 6.0 * act1[f] + (f % 5) as f32 * 0.5)).collect();
        let c = [
            [motion_activity_correlation(&mouth_a, &act0), motion_activity_correlation(&mouth_b, &act0)],
            [motion_activity_correlation(&mouth_a, &act1), motion_activity_correlation(&mouth_b, &act1)],
        ];
        assert!(c[0][0].unwrap() > 0.5 && c[0][1].unwrap() < 0.0);
        assert!(c[1][1].unwrap() > 0.5 && c[1][0].unwrap() < 0.0);
        let corr: Vec<Vec<Option<f32>>> = c.iter().map(|r| r.to_vec()).collect();
        let links = assign_links(&[1, 2], &[10, 20], &corr);
        assert_eq!(links.len(), 2);
        assert_eq!((links[0].speaker, links[0].person_id), (1, 10));
        assert_eq!((links[1].speaker, links[1].person_id), (2, 20));
        assert!(links.iter().all(|l| l.confidence > 0.9));
    }

    #[test]
    fn crossed_correlations_assign_crossed() {
        let corr = vec![vec![Some(-0.2), Some(0.15)], vec![Some(0.25), Some(-0.1)]];
        let links = assign_links(&[1, 2], &[10, 20], &corr);
        assert_eq!(links[0].person_id, 20);
        assert_eq!(links[1].person_id, 10);
        // Weak evidence gives lower confidence than strong.
        let weak = assign_links(&[1, 2], &[10, 20], &[vec![Some(0.05), Some(0.04)], vec![Some(0.03), Some(0.04)]]);
        assert!(weak.iter().all(|l| l.confidence < 0.2), "{weak:?}");
    }

    #[test]
    fn single_face_single_voice_is_trivial_and_off_camera_voice_stays_unlinked() {
        let links = assign_links(&[3], &[7], &[vec![None]]);
        assert_eq!(links, vec![LinkDecision { speaker: 3, person_id: 7, correlation: None, confidence: 0.8 }]);
        // Two voices, one face: only the voice that moves the mouth is linked.
        let links = assign_links(&[1, 2], &[7], &[vec![Some(0.3)], vec![Some(-0.3)]]);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].speaker, 1);
    }

    #[test]
    fn mouth_patch_needs_a_visible_mouth() {
        let img = vec![128u8; 640 * 360];
        let lm = [(0.40, 0.40), (0.50, 0.40), (0.45, 0.45), (0.41, 0.50), (0.49, 0.50)];
        let p = mouth_patch(&img, 640, 360, &lm).expect("51 px mouth");
        assert_eq!(p.len(), GRID_W * GRID_H);
        let tiny = [(0.4, 0.4), (0.401, 0.4), (0.4, 0.41), (0.4, 0.42), (0.403, 0.42)];
        assert!(mouth_patch(&img, 640, 360, &tiny).is_none());
        assert_eq!(decode_height(1280, 720), 360);
        assert_eq!(decode_height(1080, 1920), 1138);
    }
}
