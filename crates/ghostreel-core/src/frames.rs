//! Keyframes (plan §3 step 5): pick moments worth describing, save them as JPEGs, drop duplicates.
//!
//! 1. One fast low-resolution decode reads every frame's scene score at 4 fps. A score over
//!    `scene_threshold` is a cut; the scores *below* it are accumulated, and when the running sum
//!    passes `change_budget` that moment is worth a frame too. This is the cumulative half of the
//!    twin-comparison algorithm (Zhang, Kankanhalli & Smoliar, 1993), and it exists because a
//!    moving camera is formally an endless gradual transition: it never trips a cut threshold, so
//!    a drive through a neighbourhood used to be sampled by the 20 s fallback alone and every
//!    keyframe showed a different street with everything between them unindexed.
//! 2. Gaps longer than `max_interval_s` are filled so static footage (talking heads, screen
//!    recordings) still gets a frame every so often; cuts closer than `min_interval_s` are thinned.
//! 3. Each chosen moment is extracted at full quality (long side 1280 px — enough for a vision model
//!    to read on-screen text, see S0 findings).
//! 4. A 64-bit difference hash drops frames that look the same as the previous kept frame.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::Error;

#[derive(Debug, Clone)]
pub struct FrameOptions {
    pub scene_threshold: f64,
    /// Accumulated sub-cut change that earns a keyframe. 0 turns it off, leaving cuts and the
    /// `max_interval_s` clock alone.
    ///
    /// Measured on the Greet Mag footage: ffmpeg's scene score accumulates at 0.12/s on a moving
    /// camera and 0.02/s on a locked-off interview, so 1.0 asks for a frame every ~8 s of travel
    /// and never fires on a talking head, which the interval clock already covers.
    pub change_budget: f64,
    pub max_interval_s: f64,
    /// Floor between keyframes, and the only thing standing between fast footage and a describe
    /// job that runs all night: whatever the budget wants, nothing is sampled closer than this.
    pub min_interval_s: f64,
    pub long_side: u32,
    /// Hamming distance at or below which two frames count as duplicates.
    pub dup_distance: u32,
    /// Keep a "duplicate" anyway once this long has passed since the last kept frame: the hash is
    /// coarse, and screen recordings change text without changing layout.
    pub max_dup_gap_s: f64,
    pub max_frames: usize,
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            scene_threshold: 0.3,
            change_budget: 1.0,
            max_interval_s: 20.0,
            min_interval_s: 2.0,
            long_side: 1280,
            dup_distance: 5,
            max_dup_gap_s: 60.0,
            max_frames: 600,
        }
    }
}

impl FrameOptions {
    /// Build options from the user's [`FramesConfig`], clamping `max_interval_s` to the valid
    /// range (1–60 s) and deriving related fields:
    ///
    /// - `max_dup_gap_s = (2 * max_interval_s).max(4.0)` — keeps the de-dupe window proportional
    ///   so static sections still collapse to one frame even with short intervals.
    /// - `max_frames` raised to `max(600, duration/interval)` via the caller; here we use a safe
    ///   global cap of 5000 so very long videos aren't silently truncated.
    pub fn from_config(cfg: &crate::config::FramesConfig) -> Self {
        let interval = cfg.clamped_interval();
        Self {
            max_interval_s: interval,
            max_dup_gap_s: (2.0 * interval).max(4.0),
            max_frames: 5000,
            change_budget: cfg.change_budget,
            min_interval_s: cfg.clamped_min_interval(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Frame {
    pub t_s: f64,
    pub path: PathBuf,
    pub dhash: u64,
}

/// What one low-resolution decode found: where the picture cut, and where it had drifted far
/// enough to be worth another look.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneScan {
    /// Scores over `scene_threshold` — an outright cut.
    pub cuts: Vec<f64>,
    /// Moments where the accumulated sub-cut change passed `change_budget`. Empty when the budget
    /// is 0. These are what a travelling shot gets instead of nothing.
    pub changes: Vec<f64>,
}

/// One low-resolution decode: cuts, accumulated change, and progress in seconds decoded.
///
/// Every sampled frame prints its scene score (`metadata=print`), so progress advances steadily
/// instead of only at cuts — and the scores below the cut threshold, which used to be read and
/// thrown away, are the signal a moving camera has. Accumulating them costs a few additions on a
/// line we already parse: no second pass, no extra decode.
///
/// Decoding uses CUDA when available (4K HEVC is ~8× faster than on the CPU); ffmpeg falls back
/// to software decoding by itself when it isn't.
pub async fn scene_scan(
    ffmpeg: &Path,
    video: &Path,
    opts: &FrameOptions,
    mut on_progress: impl FnMut(f64),
) -> Result<SceneScan, Error> {
    let threshold = opts.scene_threshold;
    let filter = "fps=4,scale=256:-2,select='gte(scene\\,0)',metadata=print:key=lavfi.scene_score";
    let mut child = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-nostats"])
        .args(hwaccel_args())
        .arg("-i")
        .arg(video)
        .args(["-an", "-sn", "-dn", "-vf", filter, "-f", "null", "-"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Frames(format!("cannot run {}: {e}", ffmpeg.display())))?;

    let stderr = child.stderr.take().ok_or_else(|| Error::Frames("ffmpeg stderr unavailable".into()))?;
    let mut scan = SceneScan::default();
    let mut last_error = String::new();
    let mut pts = 0.0f64;
    // The running sum since the last frame this pass asked for, and when that was. A cut resets
    // both: a new shot starts its own drift, and carrying the old one over would ask for a frame
    // moments after the cut already got one.
    let mut accumulated = 0.0f64;
    let mut last_pick = f64::NEG_INFINITY;
    let budget = opts.change_budget;
    let floor = opts.min_interval_s;

    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(l)) = lines.next_line().await {
        if let Some(t) = l.split("pts_time:").nth(1).and_then(|v| v.split_whitespace().next()) {
            if let Ok(t) = t.parse::<f64>() {
                pts = t;
                on_progress(t);
            }
        } else if let Some(score) = l.split("lavfi.scene_score=").nth(1) {
            let Ok(score) = score.trim().parse::<f64>() else { continue };
            accumulate(score, pts, threshold, budget, floor, &mut accumulated, &mut last_pick, &mut scan);
        } else if !l.trim().is_empty() {
            last_error = l;
        }
    }
    let status = child.wait().await.map_err(|e| Error::Frames(e.to_string()))?;
    if !status.success() {
        return Err(Error::Frames(format!("scene detection failed: {last_error}")));
    }
    Ok(scan)
}

/// One scene score, folded into the running scan. Split out so the rule can be tested against a
/// list of numbers instead of against ffmpeg.
///
/// A cut resets the accumulator: a new shot starts its own drift, and carrying the old one over
/// would ask for a frame moments after the cut already got one. The floor is enforced here rather
/// than in the plan, and it does *not* reset the accumulator — change that happened is change
/// that happened, so footage moving faster than the floor allows gets a frame the moment it is
/// allowed one instead of losing the overflow.
#[allow(clippy::too_many_arguments)]
fn accumulate(
    score: f64,
    pts: f64,
    threshold: f64,
    budget: f64,
    floor: f64,
    accumulated: &mut f64,
    last_pick: &mut f64,
    scan: &mut SceneScan,
) {
    if score > threshold {
        scan.cuts.push(pts);
        *accumulated = 0.0;
        *last_pick = pts;
    } else if budget > 0.0 {
        *accumulated += score;
        if *accumulated >= budget && pts - *last_pick >= floor {
            scan.changes.push(pts);
            *accumulated = 0.0;
            *last_pick = pts;
        }
    }
}

/// Hardware decoding for the full-length scan. macOS uses VideoToolbox; elsewhere CUDA (NVIDIA).
fn hwaccel_args() -> &'static [&'static str] {
    if std::env::var_os("GHOSTREEL_NO_HWACCEL").is_some() {
        &[]
    } else if cfg!(target_os = "macos") {
        &["-hwaccel", "videotoolbox"]
    } else {
        &["-hwaccel", "cuda"]
    }
}

/// Final sampling plan: an early frame, scene cuts (thinned to `min_interval_s`), the moments the
/// picture had drifted far enough, and fillers so no gap exceeds `max_interval_s`. Sorted, within
/// `[0, duration)`, at most `max_frames`.
pub fn plan_times(duration_s: f64, scan: &SceneScan, opts: &FrameOptions) -> Vec<f64> {
    if duration_s <= 0.0 {
        return vec![0.0];
    }
    // Frame 0 is often black/fade-in: start a little in.
    let first = (duration_s * 0.05).min(1.0);
    let last_ok = (duration_s - 0.05).max(0.0);
    let mut picks = vec![first];

    // Cuts land 0.2 s late so the new shot is fully on screen; a drift moment is already the
    // picture we want, so it is taken where it is. Merged in time order, thinned once.
    let mut sorted: Vec<(f64, bool)> = scan
        .cuts
        .iter()
        .map(|t| (*t, true))
        .chain(scan.changes.iter().map(|t| (*t, false)))
        .filter(|(t, _)| *t > first && *t < last_ok)
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (t, is_cut) in sorted {
        let t = if is_cut { (t + 0.2).min(last_ok) } else { t };
        if t - picks.last().unwrap() >= opts.min_interval_s {
            picks.push(t);
        }
    }
    // Fill long gaps (including up to the end).
    let mut filled = Vec::with_capacity(picks.len());
    let bounds: Vec<f64> = picks.iter().copied().chain(std::iter::once(duration_s)).collect();
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        filled.push(a);
        let gap = b - a;
        if gap > opts.max_interval_s {
            let n = (gap / opts.max_interval_s).ceil() as usize;
            for k in 1..n {
                let t = a + gap * k as f64 / n as f64;
                if t < last_ok {
                    filled.push(t);
                }
            }
        }
    }
    if filled.len() > opts.max_frames {
        // Keep an even spread rather than only the beginning.
        let step = filled.len() as f64 / opts.max_frames as f64;
        filled = (0..opts.max_frames).map(|i| filled[(i as f64 * step) as usize]).collect();
    }
    filled
}

/// Extract the frame at `t_s` as a JPEG (long side ≤ `long_side`).
pub async fn extract(ffmpeg: &Path, video: &Path, t_s: f64, out: &Path, long_side: u32) -> Result<(), Error> {
    if let Some(dir) = out.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    }
    let scale = format!("scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'");
    let output = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-y", "-ss", &format!("{t_s:.3}"), "-i"])
        .arg(video)
        .args(["-frames:v", "1", "-an", "-vf", &scale, "-q:v", "3"])
        .arg(out)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Frames(format!("cannot run {}: {e}", ffmpeg.display())))?;
    if !output.status.success() || !out.is_file() {
        let msg = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Frames(format!("extract at {t_s:.1}s: {}", msg.lines().next().unwrap_or("no frame"))));
    }
    Ok(())
}

/// 64-bit difference hash: grayscale 9×8, one bit per horizontal neighbour comparison.
pub fn dhash(path: &Path) -> Result<u64, Error> {
    let img = image::open(path).map_err(|e| Error::Frames(format!("{}: {e}", path.display())))?;
    let small = img.grayscale().resize_exact(9, 8, image::imageops::FilterType::Triangle).to_luma8();
    let mut hash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let bit = small.get_pixel(x, y)[0] > small.get_pixel(x + 1, y)[0];
            hash = (hash << 1) | bit as u64;
        }
    }
    Ok(hash)
}

pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Full pipeline for one video. Frames land in `dir` as `t<milliseconds>.jpg`; duplicates are
/// deleted. `on_progress` gets a 0–1 fraction (decode ≈ 70 %, extraction ≈ 30 %).
pub async fn extract_keyframes(
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    dir: &Path,
    opts: &FrameOptions,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<Frame>, Error> {
    let scan = scene_scan(ffmpeg, video, opts, |secs| {
        if duration_s > 0.0 {
            on_progress(0.7 * (secs / duration_s).min(1.0));
        }
    })
    .await?;
    let times = plan_times(duration_s, &scan, opts);

    // Extract in parallel (ffmpeg seeks are cheap), then dedupe in time order.
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut set = tokio::task::JoinSet::new();
    for (i, t) in times.iter().copied().enumerate() {
        let (sem, ffmpeg, video) = (sem.clone(), ffmpeg.to_path_buf(), video.to_path_buf());
        let out = dir.join(format!("t{:09}.jpg", (t * 1000.0).round() as u64));
        let long_side = opts.long_side;
        set.spawn(async move {
            let _permit = sem.acquire_owned().await;
            let r = extract(&ffmpeg, &video, t, &out, long_side).await;
            let h = match r {
                Ok(()) => {
                    let p = out.clone();
                    tokio::task::spawn_blocking(move || dhash(&p))
                        .await
                        .unwrap_or_else(|e| Err(Error::Frames(e.to_string())))
                }
                Err(e) => Err(e),
            };
            (i, t, out, h)
        });
    }
    let mut extracted = Vec::with_capacity(times.len());
    let mut done = 0usize;
    let mut first_error = None;
    while let Some(j) = set.join_next().await {
        let Ok((i, t, out, h)) = j else { continue };
        done += 1;
        on_progress(0.7 + 0.3 * done as f64 / times.len().max(1) as f64);
        match h {
            Ok(h) => extracted.push((i, Frame { t_s: t, path: out, dhash: h })),
            Err(e) => {
                // Seeking past the last decodable frame fails on some files; skip that moment.
                let _ = tokio::fs::remove_file(&out).await;
                first_error.get_or_insert(e);
            }
        }
    }
    extracted.sort_by_key(|(i, _)| *i);
    if extracted.is_empty() {
        return Err(first_error.unwrap_or_else(|| Error::Frames("no frames extracted".into())));
    }

    let mut kept: Vec<Frame> = Vec::with_capacity(extracted.len());
    for (_, f) in extracted {
        let dup = kept
            .last()
            .is_some_and(|k| hamming(k.dhash, f.dhash) <= opts.dup_distance && f.t_s - k.t_s < opts.max_dup_gap_s);
        if dup {
            let _ = tokio::fs::remove_file(&f.path).await;
        } else {
            kept.push(f);
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> FrameOptions {
        FrameOptions::default()
    }

    fn cuts(at: &[f64]) -> SceneScan {
        SceneScan { cuts: at.to_vec(), changes: Vec::new() }
    }

    #[test]
    fn plan_fills_gaps_and_thins_cuts() {
        // 69 s video, cuts at 8, 8.5 (flash), 53, 63.
        let t = plan_times(69.0, &cuts(&[8.0, 8.5, 53.0, 63.0]), &opts());
        assert_eq!(t[0], 1.0);
        assert!(t.contains(&8.2) && !t.contains(&8.7), "cut 0.5 s after another is thinned: {t:?}");
        assert!(t.windows(2).all(|w| w[1] - w[0] <= 20.0 + 1e-9), "no gap over 20 s: {t:?}");
        assert!(t.windows(2).all(|w| w[1] > w[0]));
        assert!(*t.last().unwrap() < 69.0);
        // Static 5-minute screen recording: a frame at least every 20 s.
        let t = plan_times(300.0, &cuts(&[]), &opts());
        assert!(t.len() >= 15, "{}", t.len());
        // Tiny clip.
        assert_eq!(plan_times(0.8, &cuts(&[]), &opts()), vec![0.04000000000000001]);
    }

    #[test]
    fn plan_caps_frame_count_evenly() {
        let o = FrameOptions { max_frames: 10, ..opts() };
        let t = plan_times(3600.0, &cuts(&[]), &o);
        assert_eq!(t.len(), 10);
        assert!(*t.last().unwrap() > 3000.0, "spread across the video, not just the start");
    }

    fn have_ffmpeg() -> bool {
        crate::proc::std_command("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    #[tokio::test]
    async fn keyframes_of_a_video_with_scene_cuts() {
        if !have_ffmpeg() {
            eprintln!("skipping: no ffmpeg");
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("scenes.mp4");
        // 8 s bars, 30 s static red, 6 s testsrc (moving), 6 s blue.
        let ok = crate::proc::std_command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "smptebars=size=640x360:rate=10:d=8"])
            .args(["-f", "lavfi", "-i", "color=c=red:size=640x360:rate=10:d=30"])
            .args(["-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:d=6"])
            .args(["-f", "lavfi", "-i", "color=c=blue:size=640x360:rate=10:d=6"])
            .args(["-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,format=yuv420p", "-c:v", "mpeg4"])
            .arg(&video)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let dir = tmp.path().join("frames");
        let mut progress = Vec::new();
        let frames =
            extract_keyframes(Path::new("ffmpeg"), &video, 50.0, &dir, &opts(), |p| progress.push(p)).await.unwrap();

        let times: Vec<f64> = frames.iter().map(|f| f.t_s).collect();
        assert!(frames.iter().all(|f| f.path.is_file()));
        // The static red section yields filler frames that are all duplicates of the first red one.
        let red = times.iter().filter(|t| **t > 8.0 && **t < 38.0).count();
        assert_eq!(red, 1, "static red section collapses to one frame: {times:?}");
        assert!(times.iter().any(|t| (38.0..44.0).contains(t)), "testsrc section present: {times:?}");
        assert!(times.iter().any(|t| *t >= 44.0), "blue section present: {times:?}");
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, frames.len(), "duplicate JPEGs deleted");
        let img = image::image_dimensions(&frames[0].path).unwrap();
        assert_eq!(img, (640, 360), "never upscaled");
        assert!((progress.last().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn dhash_distinguishes_images() {
        let tmp = tempfile::tempdir().unwrap();
        let mk = |name: &str, f: &dyn Fn(u32, u32) -> u8| {
            let img = image::GrayImage::from_fn(64, 64, |x, y| image::Luma([f(x, y)]));
            let p = tmp.path().join(name);
            img.save(&p).unwrap();
            p
        };
        let a = dhash(&mk("a.jpg", &|x, _| (x * 4) as u8)).unwrap();
        let a2 = dhash(&mk("a2.jpg", &|x, _| (x * 4).saturating_add(3) as u8)).unwrap();
        let b = dhash(&mk("b.jpg", &|x, _| (255 - x * 4) as u8)).unwrap();
        assert!(hamming(a, a2) <= 5);
        assert!(hamming(a, b) > 20);
    }

    #[test]
    fn from_config_derives_dup_gap_and_raises_max_frames() {
        use crate::config::FramesConfig;

        let mut fc = FramesConfig { max_interval_s: 8.0, ..Default::default() };
        // Default 8 s → dup_gap = 16 s
        let opts = FrameOptions::from_config(&fc);
        assert_eq!(opts.max_interval_s, 8.0);
        assert_eq!(opts.max_dup_gap_s, 16.0);
        assert!(opts.max_frames >= 5000);

        // 1 s → dup_gap = max(2, 4) = 4 s (minimum floor)
        fc.max_interval_s = 1.0;
        let opts = FrameOptions::from_config(&fc);
        assert_eq!(opts.max_interval_s, 1.0);
        assert_eq!(opts.max_dup_gap_s, 4.0);

        // 30 s → dup_gap = 60 s
        fc.max_interval_s = 30.0;
        let opts = FrameOptions::from_config(&fc);
        assert_eq!(opts.max_dup_gap_s, 60.0);

        // Values out of range are clamped
        fc.max_interval_s = 0.1;
        let opts = FrameOptions::from_config(&fc);
        assert_eq!(opts.max_interval_s, 1.0, "out-of-range clamped to minimum");

        fc.max_interval_s = 120.0;
        let opts = FrameOptions::from_config(&fc);
        assert_eq!(opts.max_interval_s, 60.0, "out-of-range clamped to maximum");
    }

    #[test]
    fn short_interval_enforced_in_plan() {
        use crate::config::FramesConfig;

        let fc = FramesConfig { max_interval_s: 5.0, ..Default::default() };
        let opts = FrameOptions::from_config(&fc);
        // 60 s video, no scene cuts: frames should be at most 5 s apart
        let t = plan_times(60.0, &cuts(&[]), &opts);
        assert!(t.windows(2).all(|w| w[1] - w[0] <= 5.0 + 1e-9), "gap exceeds interval: {t:?}");
        assert!(t.len() >= 11, "expected ≥ 11 frames for 60 s / 5 s interval, got {}", t.len());
    }

    /// Run a list of per-frame scores through the rule, at 4 fps, as the decode loop would.
    fn scan_scores(scores: &[f64], opts: &FrameOptions) -> SceneScan {
        let mut scan = SceneScan::default();
        let (mut acc, mut last) = (0.0, f64::NEG_INFINITY);
        for (i, &sc) in scores.iter().enumerate() {
            let pts = i as f64 / 4.0;
            accumulate(
                sc,
                pts,
                opts.scene_threshold,
                opts.change_budget,
                opts.min_interval_s,
                &mut acc,
                &mut last,
                &mut scan,
            );
        }
        scan
    }

    #[test]
    fn a_moving_camera_earns_frames_a_cut_threshold_never_would() {
        // 60 s at 4 fps. The measured rate on the DJI walk: 0.031 per frame, 0.12 per second,
        // and not one frame anywhere near the 0.3 cut threshold.
        let travelling = vec![0.031; 240];
        let scan = scan_scores(&travelling, &opts());
        assert!(scan.cuts.is_empty(), "nothing here is a cut, which is the whole problem");
        // Budget 1.0 at 0.12/s ≈ a frame every 8 s.
        assert!(
            (7..=9).contains(&scan.changes.len()),
            "expected ~8 frames over 60 s, got {}: {:?}",
            scan.changes.len(),
            scan.changes
        );

        // The same footage with the budget off is what GhostReel did before: nothing at all, and
        // the 20 s interval clock left to do the whole job.
        let off = FrameOptions { change_budget: 0.0, ..opts() };
        assert!(scan_scores(&travelling, &off).changes.is_empty());
    }

    #[test]
    fn a_locked_off_interview_is_left_to_the_interval_clock() {
        // The measured rate on a tripod interview: 0.005 per frame, 0.02 per second.
        let scan = scan_scores(&vec![0.005; 240], &opts());
        assert!(scan.cuts.is_empty());
        assert!(scan.changes.len() <= 1, "a talking head must not be resampled by drift: {:?}", scan.changes);
    }

    #[test]
    fn the_floor_holds_when_the_footage_is_far_faster_than_anything_measured() {
        // A car at speed: nine times the drift of the DJI walk, and still never a cut — which is
        // the point, a continuously moving camera has no cuts at any speed. The budget alone
        // would ask for a frame every 0.9 s; min_interval_s is the only thing between that and an
        // overnight describe job.
        let scan = scan_scores(&vec![0.29; 240], &opts());
        assert!(scan.cuts.is_empty(), "fast is not the same as cut");
        let gaps: Vec<f64> = scan.changes.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|g| *g >= opts().min_interval_s - 1e-9), "the floor was crossed: {gaps:?}");
        // 60 s at a 2 s floor: about 30 frames, not the ~69 the budget alone would have asked for.
        assert!(scan.changes.len() <= 31, "{} frames in 60 s", scan.changes.len());
        assert!(scan.changes.len() >= 25, "the floor must not starve it either: {}", scan.changes.len());
    }

    #[test]
    fn a_cut_starts_the_drift_over() {
        // Drift almost to the budget, then cut. The cut gets its own frame and the leftover
        // drift is discarded, so the next frame is not asked for a moment later.
        let mut scores = vec![0.09; 10];
        scores.push(0.9);
        scores.extend(vec![0.09; 10]);
        let scan = scan_scores(&scores, &opts());
        assert_eq!(scan.cuts.len(), 1);
        assert!(scan.changes.is_empty(), "the cut absorbed the drift: {:?}", scan.changes);
    }

    #[test]
    fn drift_moments_join_the_plan_where_they_happened() {
        let scan = SceneScan { cuts: vec![10.0], changes: vec![20.0, 40.0] };
        let t = plan_times(60.0, &scan, &opts());
        // A cut lands 0.2 s late so the new shot is on screen; a drift moment is already the
        // picture we want, so it is taken where it is.
        assert!(t.contains(&10.2), "{t:?}");
        assert!(t.contains(&20.0) && t.contains(&40.0), "{t:?}");
        assert!(t.windows(2).all(|w| w[1] > w[0]), "still in order: {t:?}");
    }
}
