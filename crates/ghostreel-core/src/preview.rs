//! Timeline preview: proxy cache, concat demuxing, and title/narration burning (plan §4a, M8b).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::otio::{self, ResolvedMedia};
use crate::script::{self, Audio, Fps, Script};

/// A segment in the planned preview timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedSegment {
    pub video_id: i64,
    pub path: PathBuf,
    pub in_s: f64,
    pub out_s: f64,
    pub timeline_start_s: f64,
    pub beat_id: String,
    pub mute: bool,
    pub has_audio: bool,
    /// Which of the file's audio tracks carries the speech. A field recording has several — a
    /// camera mic, lavs, tracks left silent — and the first is a guess.
    #[serde(default)]
    pub audio_track: u32,
}

/// The sound running under a beat: where it sits on the preview timeline and where it comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BedSpan {
    pub beat_id: String,
    pub video_id: i64,
    pub path: PathBuf,
    /// Range inside the source file.
    pub in_s: f64,
    pub out_s: f64,
    /// Where the beat starts on the preview timeline.
    pub timeline_start_s: f64,
    pub audio_track: u32,
}

/// A beat's span on the preview timeline for on-screen titles and narration captions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatSpan {
    pub beat_id: String,
    pub start_s: f64,
    pub end_s: f64,
    pub on_screen_text: Option<String>,
    pub narration: Option<String>,
}

/// Video encoder configuration (nvenc or libx264).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoder {
    pub name: String,
    pub args: Vec<String>,
}

impl Encoder {
    pub fn nvenc() -> Self {
        Self {
            name: "h264_nvenc".into(),
            args: vec!["-c:v".into(), "h264_nvenc".into(), "-preset".into(), "p4".into(), "-cq".into(), "23".into()],
        }
    }

    pub fn libx264() -> Self {
        Self {
            name: "libx264".into(),
            args: vec![
                "-c:v".into(),
                "libx264".into(),
                "-preset".into(),
                "veryfast".into(),
                "-crf".into(),
                "23".into(),
            ],
        }
    }
}

impl std::fmt::Display for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Options controlling preview rendering.
#[derive(Debug, Clone, Default)]
pub struct PreviewOptions {
    pub burn_titles: bool,
    pub burn_narration: bool,
    /// Bring every clip to the same loudness. Cutting between a lav at -20 dB and a room mic at
    /// -33 dB is jarring however good each one is on its own.
    pub normalize_audio: bool,
    /// Seconds of fade at each join; `script.audio_fade_s`.
    pub audio_fade_s: f64,
    /// How far a cut runs past the last word so its decay survives; `script.speech_overrun_s`.
    /// The fade out spends this, rather than starting on the word's final sample.
    pub speech_overrun_s: f64,
    pub out: Option<PathBuf>,
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Result of rendering a timeline preview.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreviewResult {
    pub path: PathBuf,
    pub segments: usize,
    pub proxies_built: usize,
    pub proxies_cached: usize,
    pub encoder: String,
    pub duration_s: f64,
}

/// Calculate proxy dimensions preserving aspect ratio with height 540 and even width.
/// For 16:9 (1920x1080), gives 960x540.
pub fn proxy_dimensions(w: u32, h: u32) -> (u32, u32) {
    if h == 0 {
        return (960, 540);
    }
    let raw_w = 540.0 * (w as f64) / (h as f64);
    let mut rounded = raw_w.round() as u32;
    if !rounded.is_multiple_of(2) {
        if raw_w < rounded as f64 {
            rounded -= 1;
        } else {
            rounded += 1;
        }
    }
    (rounded.max(2), 540)
}

/// Generate cache key file name for a proxy segment.
/// Format: `<content_hash>_<in_ms>_<out_ms>_<fps_num>_<fps_den>_<w>x<h>.mp4`.
/// Content hashes look like `b3e:<hex>`; characters that aren't valid in file names everywhere
/// (`:` on Windows) are replaced with `-`.
pub fn proxy_file_name(content_hash: &str, in_s: f64, out_s: f64, fps: Fps, w: u32, h: u32) -> String {
    proxy_file_name_with_audio(content_hash, in_s, out_s, fps, w, h, false, 0.0)
}

/// [`proxy_file_name`], distinguished also by how the audio was treated.
///
/// A proxy holds encoded audio, so two clips that differ only in levelling or fade are different
/// files. Without this a rendered clip kept whatever audio it was first built with, and changing
/// the setting appeared to do nothing.
#[allow(clippy::too_many_arguments)]
pub fn proxy_file_name_with_audio(
    content_hash: &str,
    in_s: f64,
    out_s: f64,
    fps: Fps,
    w: u32,
    h: u32,
    normalized: bool,
    fade_s: f64,
) -> String {
    let hash: String =
        content_hash.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
    let in_ms = (in_s * 1000.0).round() as i64;
    let out_ms = (out_s * 1000.0).round() as i64;
    let audio = format!("{}{}", if normalized { "n" } else { "r" }, (fade_s * 1000.0).round() as i64);
    format!("{hash}_{in_ms}_{out_ms}_{}_{}_{w}x{h}_{audio}.mp4", fps.num, fps.den)
}

/// Snap clip duration to whole frames at the project sequence fps:
/// `dur = (round(raw_dur * seq_fps) / seq_fps).max(1.0 / seq_fps)`.
/// `out_s` is set to `in_s + dur` so that the timeline positions and proxy durations agree.
pub fn plan_segments(script: &Script, media: &HashMap<i64, ResolvedMedia>) -> Result<Vec<PlannedSegment>, Error> {
    let seq_fps = script.fps.unwrap_or(Fps::new(25, 1));
    let seq_rate = seq_fps.as_f64();
    let mut segments = Vec::new();
    let mut timeline_start_s = 0.0;

    for beat in &script.beats {
        for clip in &beat.clips {
            let res = media
                .get(&clip.video_id)
                .ok_or_else(|| Error::NotFound(format!("resolved media for video #{}", clip.video_id)))?;

            let raw_dur = (clip.out_s - clip.in_s).max(0.0);
            let frames = (raw_dur * seq_rate).round().max(1.0);
            let dur = frames / seq_rate;
            let in_s = clip.in_s;
            let out_s = clip.in_s + dur;

            segments.push(PlannedSegment {
                video_id: clip.video_id,
                path: res.path.clone(),
                in_s,
                out_s,
                timeline_start_s,
                beat_id: beat.id.clone(),
                // A beat with a bed is carried by the bed; its pictures play silent, or the
                // speaker would be heard twice, a few seconds out of step with themselves.
                mute: clip.audio == Audio::Mute || beat.bed.is_some(),
                has_audio: res.has_audio,
                audio_track: res.audio_track,
            });

            timeline_start_s += dur;
        }
    }

    Ok(segments)
}

/// Compute spans (start_s .. end_s) on the timeline for each beat.
pub fn plan_beat_spans(script: &Script, segments: &[PlannedSegment]) -> Vec<BeatSpan> {
    let mut spans = Vec::new();
    for beat in &script.beats {
        let beat_segs: Vec<&PlannedSegment> = segments.iter().filter(|s| s.beat_id == beat.id).collect();
        let (start_s, end_s) = if let Some(first) = beat_segs.first() {
            let last = beat_segs.last().unwrap();
            (first.timeline_start_s, last.timeline_start_s + (last.out_s - last.in_s).max(0.0))
        } else {
            (0.0, 0.0)
        };

        spans.push(BeatSpan {
            beat_id: beat.id.clone(),
            start_s,
            end_s,
            on_screen_text: beat.on_screen_text.clone().filter(|t| !t.trim().is_empty()),
            narration: beat.narration.clone().filter(|t| !t.trim().is_empty()),
        });
    }
    spans
}

/// Where each beat's audio bed sits on the timeline, and which file it is cut from.
pub fn plan_bed_spans(
    script: &Script,
    segments: &[PlannedSegment],
    media: &HashMap<i64, ResolvedMedia>,
) -> Vec<BedSpan> {
    let mut beds = Vec::new();
    for beat in &script.beats {
        let Some(bed) = &beat.bed else { continue };
        let Some(res) = media.get(&bed.video_id) else { continue };
        let Some(first) = segments.iter().find(|s| s.beat_id == beat.id) else { continue };
        if bed.duration_s() <= 0.0 {
            continue;
        }
        beds.push(BedSpan {
            beat_id: beat.id.clone(),
            video_id: bed.video_id,
            path: res.path.clone(),
            in_s: bed.in_s,
            out_s: bed.out_s,
            timeline_start_s: first.timeline_start_s,
            audio_track: res.audio_track,
        });
    }
    beds
}

/// Format seconds into SubRip timestamp format `HH:MM:SS,mmm`.
pub fn format_srt_time(sec: f64) -> String {
    let total_ms = (sec.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Generate SubRip (.srt) content from beat spans with narration text.
pub fn generate_narration_srt(spans: &[BeatSpan]) -> String {
    let mut srt = String::new();
    let mut idx = 1;
    for span in spans {
        if let Some(text) = &span.narration {
            let start = format_srt_time(span.start_s);
            let end = format_srt_time(span.end_s);
            srt.push_str(&format!("{idx}\n{start} --> {end}\n{text}\n\n"));
            idx += 1;
        }
    }
    srt
}

/// Escape a file path for use as an *unquoted* filter option value inside a `-vf` filtergraph
/// (e.g. `subtitles=<escaped>` or `drawtext=textfile=<escaped>`).
///
/// ffmpeg unescapes twice: first the filtergraph parser (`\ ' [ ] , ;`), then the filter's option
/// parser (`\ ' :`). Windows backslashes become forward slashes, which ffmpeg accepts.
pub fn escape_filter_path(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    let mut level1 = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '\\' | '\'' | ':') {
            level1.push('\\');
        }
        level1.push(c);
    }
    let mut out = String::with_capacity(level1.len());
    for c in level1.chars() {
        if matches!(c, '\\' | '\'' | '[' | ']' | ',' | ';') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

static ENCODER_CACHE: OnceLock<Encoder> = OnceLock::new();

/// Test whether ffmpeg has working `h264_nvenc` support.
fn check_nvenc(ffmpeg: &Path) -> bool {
    let Ok(encoders_output) = crate::proc::std_command(ffmpeg).args(["-hide_banner", "-encoders"]).output() else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&encoders_output.stdout);
    if !stdout.contains("h264_nvenc") {
        return false;
    }

    let Ok(status) = crate::proc::std_command(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=320x240:d=1",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "23",
            "-f",
            "null",
            "-",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    else {
        return false;
    };

    status.success()
}

/// Pick the best available video encoder: `h264_nvenc` if working, else `libx264`.
pub fn pick_encoder(ffmpeg: &Path) -> Encoder {
    ENCODER_CACHE.get_or_init(|| if check_nvenc(ffmpeg) { Encoder::nvenc() } else { Encoder::libx264() }).clone()
}

/// Load script and footage from DB to generate planned segments.
pub fn preview_plan(db: &Db, script_id: i64) -> Result<Vec<PlannedSegment>, Error> {
    let mut stored = script::load(db, script_id)?;
    let project = db.project(stored.project_id)?;
    stored.script.fill_from_project(&project);
    let project_fps = stored.script.fps.unwrap_or_else(|| Fps::new(project.fps_num, project.fps_den));
    let media = otio::resolve_media_for_script(db, stored.project_id, &stored.script, project_fps)?;
    plan_segments(&stored.script, &media)
}

/// Render a preview MP4 for a script draft.
pub fn render_preview(
    db: &Db,
    data_dir: &Path,
    ffmpeg: &Path,
    script_id: i64,
    opts: &PreviewOptions,
    mut on_progress: impl FnMut(f64, &str),
) -> Result<PreviewResult, Error> {
    let mut stored = script::load(db, script_id)?;
    let project = db.project(stored.project_id)?;
    stored.script.fill_from_project(&project);

    let project_fps = stored.script.fps.unwrap_or_else(|| Fps::new(project.fps_num, project.fps_den));
    let proj_w = stored.script.width.unwrap_or(project.width) as u32;
    let proj_h = stored.script.height.unwrap_or(project.height) as u32;
    let (proxy_w, proxy_h) = proxy_dimensions(proj_w, proj_h);

    let media = otio::resolve_media_for_script(db, stored.project_id, &stored.script, project_fps)?;
    let segments = plan_segments(&stored.script, &media)?;
    if segments.is_empty() {
        return Err(Error::Preview("script has no clips to preview".into()));
    }

    let beat_spans = plan_beat_spans(&stored.script, &segments);
    let bed_spans = plan_bed_spans(&stored.script, &segments, &media);
    let encoder = pick_encoder(ffmpeg);

    let proxies_dir = data_dir.join("proxies");
    std::fs::create_dir_all(&proxies_dir).map_err(|e| Error::Io(proxies_dir.clone(), e))?;
    let previews_dir = data_dir.join("previews");
    std::fs::create_dir_all(&previews_dir).map_err(|e| Error::Io(previews_dir.clone(), e))?;

    let out_path =
        opts.out.clone().unwrap_or_else(|| previews_dir.join(format!("script_{}_v{}.mp4", stored.id, stored.version)));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::Io(parent.to_path_buf(), e))?;
    }

    let total_dur: f64 = segments.iter().map(|s| (s.out_s - s.in_s).max(0.0)).sum();
    let mut completed_dur = 0.0;
    let mut proxies_built = 0;
    let mut proxies_cached = 0;
    let mut proxy_paths = Vec::new();

    let g_val = project_fps.as_f64().round().max(1.0) as i64;
    let vf = format!(
        "scale={proxy_w}:{proxy_h}:force_original_aspect_ratio=decrease,pad={proxy_w}:{proxy_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={}/{},format=yuv420p",
        project_fps.num, project_fps.den
    );

    for (i, seg) in segments.iter().enumerate() {
        if opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
            return Err(Error::Preview("cancelled".into()));
        }

        let res = media
            .get(&seg.video_id)
            .ok_or_else(|| Error::NotFound(format!("resolved media for video #{}", seg.video_id)))?;

        let p_name = proxy_file_name_with_audio(
            &res.content_hash,
            seg.in_s,
            seg.out_s,
            project_fps,
            proxy_w,
            proxy_h,
            opts.normalize_audio,
            opts.audio_fade_s,
        );
        let p_path = proxies_dir.join(p_name);

        let exists_and_non_empty = p_path.is_file() && std::fs::metadata(&p_path).map(|m| m.len() > 0).unwrap_or(false);

        if exists_and_non_empty {
            proxies_cached += 1;
        } else {
            proxies_built += 1;
            let seg_dur = (seg.out_s - seg.in_s).max(0.01);
            let in_str = format!("{:.3}", seg.in_s);
            let dur_str = format!("{:.3}", seg_dur);
            let tmp_path = p_path.with_extension(format!("tmp.{}.mp4", std::process::id()));

            let mut cmd = crate::proc::std_command(ffmpeg);
            cmd.arg("-y").arg("-ss").arg(&in_str).arg("-t").arg(&dur_str).arg("-i").arg(&seg.path);

            // Levelling happens per clip, before they are joined: loudnorm needs a whole clip
            // to measure, and the point is that clips match each other.
            // Fade each clip in and out: segments are joined end to end, and a cut taken in the
            // middle of a breath stops dead without one.
            //
            // The fade out starts where the speaking stops, not where the clip does. A clip runs
            // a little past the last word so its decay survives, and in continuous speech that
            // overrun reaches into the next sentence — audible as a stray "And" after the point
            // has been made. Fading from the sentence end keeps the decay and loses the word.
            let fade = opts.audio_fade_s.max(0.0).min(seg_dur / 3.0);
            let cfg_overrun = opts.speech_overrun_s.max(0.0);
            let speech_end_rel: Option<f64> = db
                .conn
                .query_row(
                    "SELECT MAX(end_s) FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND end_s <= ?3",
                    rusqlite::params![seg.video_id, seg.in_s, seg.out_s + 0.01],
                    |r| r.get::<_, Option<f64>>(0),
                )
                .ok()
                .flatten()
                .map(|e| (e - seg.in_s).max(0.0));
            // The fade runs *across* the decay of the last word rather than starting on it.
            // Whisper's segment times are approximate and often quantised to whole seconds, so
            // "the sentence ends here" is only nearly true; fading from that instant clipped
            // "…running around too" and "Genuine, real neighborhood." The clip carries an overrun
            // for exactly this, and the fade now spends it.
            let (fade_out_at, fade_len) = match speech_end_rel {
                Some(end) if end < seg_dur => {
                    let room = (seg_dur - end).max(0.0);
                    (end.min((seg_dur - fade).max(0.0)), fade.max(room.min(cfg_overrun)))
                }
                _ => ((seg_dur - fade).max(0.0), fade),
            };
            let fades = if fade > 0.005 {
                format!("afade=t=in:st=0:d={fade:.3},afade=t=out:st={fade_out_at:.3}:d={fade_len:.3},")
            } else {
                String::new()
            };
            let af = if opts.normalize_audio {
                format!("{fades}loudnorm=I=-16:TP=-1.5:LRA=11,apad")
            } else {
                format!("{fades}apad")
            };
            if !seg.mute && seg.has_audio {
                cmd.args(["-map", "0:v:0", "-map", &format!("0:a:{}", seg.audio_track), "-af", &af]);
            } else {
                let null_audio = "anullsrc=r=48000:cl=stereo";
                cmd.args(["-f", "lavfi", "-t", &dur_str, "-i", null_audio]);
                cmd.args(["-map", "0:v:0", "-map", "1:a:0", "-af", "apad"]);
            }

            cmd.args(["-vf", &vf]);
            cmd.args(["-g", &g_val.to_string()]);
            for arg in &encoder.args {
                cmd.arg(arg);
            }
            cmd.args([
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-t",
                &dur_str,
                "-shortest",
                "-movflags",
                "+faststart",
            ]);
            cmd.arg(&tmp_path);

            let out = cmd.output().map_err(|e| Error::Preview(format!("failed to run ffmpeg for proxy: {e}")))?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let tail =
                    stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
                let _ = std::fs::remove_file(&tmp_path);
                return Err(Error::Preview(format!("ffmpeg proxy encode failed: {tail}")));
            }

            std::fs::rename(&tmp_path, &p_path).map_err(|e| Error::Io(p_path.clone(), e))?;
        }

        proxy_paths.push(p_path);
        completed_dur += (seg.out_s - seg.in_s).max(0.0);
        let frac = if total_dur > 0.0 { (completed_dur / total_dur) * 0.85 } else { 0.85 };
        on_progress(frac, &format!("Building proxies {}/{} ({})", i + 1, segments.len(), encoder.name));
    }

    if opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
        return Err(Error::Preview("cancelled".into()));
    }

    // Prepare concat list
    let concat_list_path = previews_dir.join(format!("concat_{}_{}.txt", stored.id, std::process::id()));
    let mut concat_content = String::from("ffconcat version 1.0\n");
    for p in &proxy_paths {
        let abs_path = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        let escaped = abs_path.to_string_lossy().replace('\'', "'\\''");
        concat_content.push_str(&format!("file '{escaped}'\n"));
    }
    std::fs::write(&concat_list_path, concat_content).map_err(|e| Error::Io(concat_list_path.clone(), e))?;

    let has_titles = opts.burn_titles && beat_spans.iter().any(|b| b.on_screen_text.is_some());
    let has_narration = opts.burn_narration && beat_spans.iter().any(|b| b.narration.is_some());
    let burn_pass_needed = has_titles || has_narration;

    let bed_pass_needed = !bed_spans.is_empty();
    let concat_output = if burn_pass_needed || bed_pass_needed {
        out_path.with_extension(format!("concat.tmp.{}.mp4", std::process::id()))
    } else {
        out_path.clone()
    };

    on_progress(0.86, "Concatenating segments…");

    let concat_res = crate::proc::std_command(ffmpeg)
        .args(["-y", "-f", "concat", "-safe", "0", "-i"])
        .arg(&concat_list_path)
        .args(["-c", "copy", "-movflags", "+faststart"])
        .arg(&concat_output)
        .output();

    let _ = std::fs::remove_file(&concat_list_path);

    let concat_out = concat_res.map_err(|e| Error::Preview(format!("failed to spawn ffmpeg concat: {e}")))?;
    if !concat_out.status.success() {
        let stderr = String::from_utf8_lossy(&concat_out.stderr);
        let tail = stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        let _ = std::fs::remove_file(&concat_output);
        return Err(Error::Preview(format!("concat failed: {tail}")));
    }

    // Lay the beds: one voice running across a beat while its pictures change under it. The
    // pictures were encoded silent, so this is a mix onto silence rather than over anything.
    let mut burn_input = concat_output.clone();
    if bed_pass_needed {
        on_progress(0.87, "Laying audio beds…");
        let mut bed_files = Vec::new();
        for (i, bed) in bed_spans.iter().enumerate() {
            let dur = (bed.out_s - bed.in_s).max(0.01);
            let wav = previews_dir.join(format!("bed_{}_{i}_{}.wav", stored.id, std::process::id()));
            let fade = opts.audio_fade_s.max(0.0).min(dur / 3.0);
            let cfg_overrun = opts.speech_overrun_s.max(0.0);
            // As for a clip, the fade out starts where the speaking stops, not where the bed does.
            let speech_end_rel: Option<f64> = db
                .conn
                .query_row(
                    "SELECT MAX(end_s) FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND end_s <= ?3",
                    params![bed.video_id, bed.in_s, bed.out_s + 0.01],
                    |r| r.get::<_, Option<f64>>(0),
                )
                .ok()
                .flatten()
                .map(|e| (e - bed.in_s).max(0.0));
            let (fade_out_at, fade_len) = match speech_end_rel {
                Some(end) if end < dur => {
                    let room = (dur - end).max(0.0);
                    (end.min((dur - fade).max(0.0)), fade.max(room.min(cfg_overrun)))
                }
                _ => ((dur - fade).max(0.0), fade),
            };
            let mut af = String::new();
            if fade > 0.005 {
                af.push_str(&format!("afade=t=in:st=0:d={fade:.3},afade=t=out:st={fade_out_at:.3}:d={fade_len:.3},"));
            }
            if opts.normalize_audio {
                af.push_str("loudnorm=I=-16:TP=-1.5:LRA=11,");
            }
            af.push_str("aresample=48000,apad");

            let out = crate::proc::std_command(ffmpeg)
                .args(["-y", "-ss", &format!("{:.3}", bed.in_s), "-t", &format!("{dur:.3}"), "-i"])
                .arg(&bed.path)
                .args(["-map", &format!("0:a:{}", bed.audio_track), "-af", &af])
                .args(["-ac", "2", "-ar", "48000", "-t", &format!("{dur:.3}"), "-c:a", "pcm_s16le"])
                .arg(&wav)
                .output()
                .map_err(|e| Error::Preview(format!("failed to run ffmpeg for audio bed: {e}")))?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let tail =
                    stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
                return Err(Error::Preview(format!("audio bed extract failed: {tail}")));
            }
            bed_files.push((wav, bed.timeline_start_s));
        }

        let mixed = if burn_pass_needed {
            out_path.with_extension(format!("bedded.tmp.{}.mp4", std::process::id()))
        } else {
            out_path.clone()
        };
        let mut cmd = crate::proc::std_command(ffmpeg);
        cmd.args(["-y", "-i"]).arg(&concat_output);
        for (wav, _) in &bed_files {
            cmd.arg("-i").arg(wav);
        }
        let mut graph = String::new();
        for (i, (_, start)) in bed_files.iter().enumerate() {
            let delay_ms = (start * 1000.0).round().max(0.0) as i64;
            graph.push_str(&format!("[{}:a]adelay={delay_ms}:all=1[b{i}];", i + 1));
        }
        graph.push_str("[0:a]");
        for i in 0..bed_files.len() {
            graph.push_str(&format!("[b{i}]"));
        }
        // normalize=0: a bed mixed onto silence must keep its own level, not be halved because
        // there are two inputs. duration=first: the cut is as long as its pictures.
        graph.push_str(&format!(
            "amix=inputs={}:normalize=0:duration=first:dropout_transition=0[a]",
            bed_files.len() + 1
        ));
        cmd.args(["-filter_complex", &graph]);
        cmd.args(["-map", "0:v", "-map", "[a]", "-c:v", "copy"]);
        cmd.args(["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart"]);
        cmd.arg(&mixed);

        let mix_out = cmd.output().map_err(|e| Error::Preview(format!("failed to spawn audio bed mix: {e}")))?;
        for (wav, _) in &bed_files {
            let _ = std::fs::remove_file(wav);
        }
        if !mix_out.status.success() {
            let stderr = String::from_utf8_lossy(&mix_out.stderr);
            let tail =
                stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            let _ = std::fs::remove_file(&mixed);
            return Err(Error::Preview(format!("audio bed mix failed: {tail}")));
        }
        let _ = std::fs::remove_file(&concat_output);
        burn_input = mixed;
    }

    // Burn pass (if requested and content exists)
    if burn_pass_needed {
        on_progress(0.88, "Burning titles and narration…");

        // Title texts go through `textfile=` (no text escaping pitfalls) with `expansion=none`, so
        // `%` in a title is literal.
        let mut temp_files_to_remove = Vec::new();
        let mut titles = Vec::new(); // (escaped textfile path, start, end)
        if has_titles {
            for (i, span) in beat_spans.iter().enumerate() {
                if let Some(text) = &span.on_screen_text {
                    let tmp_txt = previews_dir.join(format!("title_{}_{i}_{}.txt", stored.id, std::process::id()));
                    std::fs::write(&tmp_txt, text).map_err(|e| Error::Io(tmp_txt.clone(), e))?;
                    titles.push((escape_filter_path(&tmp_txt), span.start_s, span.end_s));
                    temp_files_to_remove.push(tmp_txt);
                }
            }
        }
        let srt_escaped = if has_narration {
            let srt_path = out_path.parent().unwrap_or(Path::new(".")).join("narration.srt");
            std::fs::write(&srt_path, generate_narration_srt(&beat_spans))
                .map_err(|e| Error::Io(srt_path.clone(), e))?;
            Some(escape_filter_path(&srt_path))
        } else {
            None
        };

        let filtergraph = |font: Option<&str>| {
            let font = font.map(|f| format!(":font={f}")).unwrap_or_default();
            let mut filters: Vec<String> = titles
                .iter()
                .map(|(file, start, end)| {
                    format!(
                        "drawtext=textfile={file}{font}:expansion=none:enable='between(t,{start:.3},{end:.3})':\
                         fontsize=h/12:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=12:\
                         x=(w-text_w)/2:y=h*0.08"
                    )
                })
                .collect();
            if let Some(srt) = &srt_escaped {
                filters.push(format!("subtitles={srt}"));
            }
            filters.join(",")
        };
        let burn = |graph: String| {
            let mut cmd = crate::proc::std_command(ffmpeg);
            cmd.args(["-y", "-i"]).arg(&burn_input).args(["-vf", &graph]);
            cmd.args(&encoder.args);
            cmd.args(["-c:a", "copy", "-movflags", "+faststart"]).arg(&out_path);
            cmd.output()
        };

        let mut burn_res = burn(filtergraph(None));
        // No default fontconfig match for drawtext: retry naming a generic family.
        if has_titles && !matches!(&burn_res, Ok(o) if o.status.success()) {
            burn_res = burn(filtergraph(Some("Sans")));
        }
        for tmp in &temp_files_to_remove {
            let _ = std::fs::remove_file(tmp);
        }
        let _ = std::fs::remove_file(&burn_input);

        let burn_out = burn_res.map_err(|e| Error::Preview(format!("failed to spawn burn pass: {e}")))?;
        if !burn_out.status.success() {
            let stderr = String::from_utf8_lossy(&burn_out.stderr);
            let tail =
                stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            return Err(Error::Preview(format!("burn pass failed: {tail}")));
        }
    }

    // Insert exports row
    let now = crate::projects::now();
    db.conn.execute(
        "INSERT INTO exports(script_id, format, path, created_at) VALUES (?1, 'preview_mp4', ?2, ?3)",
        params![stored.id, out_path.to_string_lossy(), now],
    )?;

    on_progress(1.0, "Ready");

    Ok(PreviewResult {
        path: out_path,
        segments: segments.len(),
        proxies_built,
        proxies_cached,
        encoder: encoder.name,
        duration_s: total_dur,
    })
}

/// Where a video's playback proxy lives under the data dir.
pub fn playback_proxy_path(data_dir: &Path, content_hash: &str) -> PathBuf {
    let hex = content_hash.rsplit(':').next().unwrap_or(content_hash);
    data_dir.join("proxies").join(&hex[..2.min(hex.len())]).join(format!("{hex}.mp4"))
}

/// Whether a browser engine can be expected to play this file as it is: 8-bit 4:2:0 H.264 at
/// 1080p or less. Camera originals — 4K, 10-bit 4:2:2, HEVC — are none of that, and WebKit
/// software-decodes them so slowly the player sits black for a long time.
pub async fn plays_natively(ffprobe: &Path, video: &Path) -> bool {
    let out = crate::proc::command(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,height",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await;
    let Ok(out) = out else { return false };
    let line = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = line.lines().next().unwrap_or("").trim().split(',').collect();
    f.len() >= 3 && f[0] == "h264" && f[1] == "yuv420p" && f[2].parse::<u32>().is_ok_and(|h| h <= 1080)
}

/// A copy of the video the player can actually play: 720p, 8-bit 4:2:0 H.264 with AAC audio,
/// built once and kept. The original stays the source for keyframes, previews and exports.
pub async fn playback_proxy(
    ffmpeg: &Path,
    data_dir: &Path,
    video: &Path,
    content_hash: &str,
) -> Result<PathBuf, Error> {
    let out = playback_proxy_path(data_dir, content_hash);
    if out.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(out);
    }
    if let Some(dir) = out.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    }
    let tmp = out.with_extension("part.mp4");
    let enc = pick_encoder(ffmpeg);
    let mut cmd = crate::proc::command(ffmpeg);
    cmd.args(["-y", "-v", "error", "-i"])
        .arg(video)
        .args(["-vf", "scale=-2:720,format=yuv420p"])
        .args(enc.args.iter().map(String::as_str))
        .args(["-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart"])
        .arg(&tmp)
        .kill_on_drop(true);
    let status = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| Error::Vision(format!("ffmpeg: {e}")))?;
    if !status.status.success() {
        let _ = tokio::fs::remove_file(&tmp).await;
        let msg: String = String::from_utf8_lossy(&status.stderr).chars().take(300).collect();
        return Err(Error::Vision(format!("proxy encode failed: {msg}")));
    }
    tokio::fs::rename(&tmp, &out).await.map_err(|e| Error::Io(out.clone(), e))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn plan_math_snapping_and_beat_spans() {
        // 2 beats / 3 clips:
        // Beat 1: clip 1 (0.0..1.0), clip 2 (2.0..3.51) at 25 fps
        // 1.0s = 25 frames -> 1.0s
        // 1.51s = 37.75 frames -> rounds to 38 frames = 1.52s
        // Beat 2: clip 3 (5.0..7.0) -> 2.0s = 50 frames
        let script = Script {
            title: "Plan Test".into(),
            target_duration_s: Some(5.0),
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![
                script::Beat {
                    id: "b1".into(),
                    purpose: "intro".into(),
                    narration: Some("Voice 1".into()),
                    on_screen_text: Some("Title 1".into()),
                    clips: vec![
                        script::ScriptClip { video_id: 1, in_s: 0.0, out_s: 1.0, audio: Audio::Source, why: None },
                        script::ScriptClip { video_id: 2, in_s: 2.0, out_s: 3.51, audio: Audio::Mute, why: None },
                    ],
                    notes: None,
                    bed: None,
                },
                script::Beat {
                    id: "b2".into(),
                    purpose: "body".into(),
                    narration: Some("Voice 2".into()),
                    on_screen_text: None,
                    clips: vec![script::ScriptClip {
                        video_id: 1,
                        in_s: 5.0,
                        out_s: 7.0,
                        audio: Audio::Source,
                        why: None,
                    }],
                    notes: None,
                    bed: None,
                },
            ],
        };

        let mut media = HashMap::new();
        media.insert(
            1,
            ResolvedMedia {
                video_id: 1,
                path: PathBuf::from("/media/v1.mp4"),
                duration_s: 10.0,
                has_audio: true,
                fps: Fps::new(25, 1),
                content_hash: "hash1".into(),
                audio_track: 0,
            },
        );
        media.insert(
            2,
            ResolvedMedia {
                video_id: 2,
                path: PathBuf::from("/media/v2.mp4"),
                duration_s: 10.0,
                has_audio: false,
                fps: Fps::new(25, 1),
                content_hash: "hash2".into(),
                audio_track: 0,
            },
        );

        let segments = plan_segments(&script, &media).unwrap();
        assert_eq!(segments.len(), 3);

        // Segment 1:
        assert_eq!(segments[0].timeline_start_s, 0.0);
        assert_eq!(segments[0].in_s, 0.0);
        assert!((segments[0].out_s - 1.0).abs() < 1e-6);
        assert!(!segments[0].mute);
        assert!(segments[0].has_audio);

        // Segment 2 (snapped from 1.51s to 38 frames = 1.52s):
        assert!((segments[1].timeline_start_s - 1.0).abs() < 1e-6);
        assert_eq!(segments[1].in_s, 2.0);
        assert!((segments[1].out_s - 3.52).abs() < 1e-6);
        assert!(segments[1].mute);
        assert!(!segments[1].has_audio);

        // Segment 3:
        assert!((segments[2].timeline_start_s - 2.52).abs() < 1e-6);
        assert_eq!(segments[2].in_s, 5.0);
        assert!((segments[2].out_s - 7.0).abs() < 1e-6);

        // Beat spans:
        let spans = plan_beat_spans(&script, &segments);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].beat_id, "b1");
        assert_eq!(spans[0].start_s, 0.0);
        assert!((spans[0].end_s - 2.52).abs() < 1e-6);
        assert_eq!(spans[0].on_screen_text.as_deref(), Some("Title 1"));
        assert_eq!(spans[0].narration.as_deref(), Some("Voice 1"));

        assert_eq!(spans[1].beat_id, "b2");
        assert!((spans[1].start_s - 2.52).abs() < 1e-6);
        assert!((spans[1].end_s - 4.52).abs() < 1e-6);
        assert_eq!(spans[1].on_screen_text, None);
        assert_eq!(spans[1].narration.as_deref(), Some("Voice 2"));
    }

    #[test]
    fn proxy_file_name_and_dimensions() {
        // proxy_file_name exact string for 25/1 and 30000/1001:
        let name_25 = proxy_file_name("hash123", 1.234, 4.567, Fps::new(25, 1), 960, 540);
        assert_eq!(name_25, "hash123_1234_4567_25_1_960x540_r0.mp4");

        let name_ntsc = proxy_file_name("hash123", 1.234, 4.567, Fps::new(30000, 1001), 960, 540);
        assert_eq!(name_ntsc, "hash123_1234_4567_30000_1001_960x540_r0.mp4");

        // Proxy dimensions for 16:9, 9:16, 4:3:
        assert_eq!(proxy_dimensions(1920, 1080), (960, 540));
        assert_eq!(proxy_dimensions(1080, 1920), (304, 540));
        assert_eq!(proxy_dimensions(1440, 1080), (720, 540));
        assert_eq!(proxy_dimensions(640, 480), (720, 540));
    }

    /// Two clips that differ only in how their audio was treated are different files: a proxy
    /// carries encoded audio, and reusing one silently kept the old levelling and fade.
    #[test]
    fn audio_treatment_gives_a_proxy_its_own_name() {
        let plain = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, false, 0.0);
        let faded = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, false, 0.12);
        let levelled = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, true, 0.12);
        assert_ne!(plain, faded);
        assert_ne!(faded, levelled);
    }

    #[test]
    fn srt_formatting_and_escaping() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(65.432), "00:01:05,432");
        assert_eq!(format_srt_time(3661.05), "01:01:01,050");

        let spans = vec![
            BeatSpan {
                beat_id: "b1".into(),
                start_s: 0.0,
                end_s: 2.5,
                on_screen_text: None,
                narration: Some("Hello world".into()),
            },
            BeatSpan { beat_id: "b2".into(), start_s: 2.5, end_s: 5.0, on_screen_text: None, narration: None },
            BeatSpan {
                beat_id: "b3".into(),
                start_s: 5.0,
                end_s: 7.25,
                on_screen_text: None,
                narration: Some("Goodbye!".into()),
            },
        ];

        let srt = generate_narration_srt(&spans);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"));
        assert!(srt.contains("2\n00:00:05,000 --> 00:00:07,250\nGoodbye!\n\n"));
        assert!(!srt.contains("3\n"));

        // Path escaping:
        // Two escaping levels (option value, then filtergraph); verified against ffmpeg 8 with
        // `drawtext=textfile=` and `subtitles=` on a directory named `we:ird it's [x],y`.
        let p_unix = Path::new("/var/we:ird it's [x],y/n.srt");
        assert_eq!(escape_filter_path(p_unix), r"/var/we\\:ird it\\\'s \[x\]\,y/n.srt");

        let p_win = Path::new("C:\\Users\\test\\clip.srt");
        assert_eq!(escape_filter_path(p_win), r"C\\:/Users/test/clip.srt");

        // Proxy names never contain ':' (content hashes are `b3e:<hex>`).
        assert_eq!(
            proxy_file_name("b3e:ab12", 0.0, 1.0, Fps::new(25, 1), 960, 540),
            "b3e-ab12_0_1000_25_1_960x540_r0.mp4"
        );
    }

    #[test]
    fn render_preview_ffmpeg_integration() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else {
            eprintln!("skipping render_preview integration test: ffmpeg not found");
            return;
        };
        let Some(ffprobe) = crate::doctor::locate("ffprobe") else {
            eprintln!("skipping render_preview integration test: ffprobe not found");
            return;
        };

        let temp = tempfile::tempdir().unwrap();
        let vid_a = temp.path().join("vid_a.mp4");
        let vid_b = temp.path().join("vid_b.mp4");

        // Generate A: testsrc2 3 s 1280x720 30fps with sine audio
        let gen_a = crate::proc::std_command(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-nostats",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1280x720:rate=30:duration=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:duration=3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                vid_a.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(gen_a.success(), "failed to generate vid_a");

        // Generate B: testsrc 2 s 640x480 25fps WITHOUT audio
        let gen_b = crate::proc::std_command(&ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-nostats",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x480:rate=25:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                vid_b.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(gen_b.success(), "failed to generate vid_b");

        // Build DB with project (fps 25/1, 1920x1080)
        let mut db = Db::open_in_memory().unwrap();
        let project = db
            .create_project(&NewProject {
                name: "PreviewIntTest".into(),
                description: "".into(),
                fps_num: 25,
                fps_den: 1,
                width: 1920,
                height: 1080,
            })
            .unwrap();
        let folder = db.add_folder(project.id, temp.path(), true).unwrap();

        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash_a', 100, 3.0, 30.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, ?1, ?2, 100, 0, 0)",
                params![folder.id, vid_a.to_str().unwrap()],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (2, 'hash_b', 100, 2.0, 25.0, 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (2, ?1, ?2, 100, 0, 0)",
                params![folder.id, vid_b.to_str().unwrap()],
            )
            .unwrap();

        // Script with 2 beats:
        // Beat 1: clip A (0.5..2.5, duration 2.0s), audio=source, with on_screen_text + narration
        // Beat 2: clip B (0.0..1.5, duration 1.5s)
        let script = Script {
            title: "Preview Integration".into(),
            target_duration_s: Some(3.5),
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![
                script::Beat {
                    id: "b1".into(),
                    purpose: "intro".into(),
                    narration: Some("Narration for Beat 1".into()),
                    on_screen_text: Some("Preview Title".into()),
                    clips: vec![script::ScriptClip {
                        video_id: 1,
                        in_s: 0.5,
                        out_s: 2.5,
                        audio: Audio::Source,
                        why: None,
                    }],
                    notes: None,
                    bed: None,
                },
                script::Beat {
                    id: "b2".into(),
                    purpose: "outro".into(),
                    narration: None,
                    on_screen_text: None,
                    clips: vec![script::ScriptClip {
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 1.5,
                        audio: Audio::Source,
                        why: None,
                    }],
                    notes: None,
                    bed: None,
                },
            ],
        };

        let script_id = script::save_version(&db, project.id, &script, None).unwrap();

        // Awkward data dir so the filtergraph escaping of the title/SRT paths is exercised.
        let data_dir = temp.path().join(if cfg!(windows) { "data it's [x],y" } else { "da:ta it's [x],y" });
        std::fs::create_dir_all(&data_dir).unwrap();

        // First render (cold)
        let res1 = render_preview(&db, &data_dir, &ffmpeg, script_id, &PreviewOptions::default(), |_, _| {}).unwrap();
        assert_eq!(res1.segments, 2);
        assert_eq!(res1.proxies_built, 2);
        assert_eq!(res1.proxies_cached, 0);
        assert!(res1.path.is_file());

        // Second render (warm) - must report proxies_cached == 2, proxies_built == 0
        let res2 = render_preview(&db, &data_dir, &ffmpeg, script_id, &PreviewOptions::default(), |_, _| {}).unwrap();
        assert_eq!(res2.segments, 2);
        assert_eq!(res2.proxies_built, 0);
        assert_eq!(res2.proxies_cached, 2);

        // Third render with burn_titles + burn_narration
        let burned_out = data_dir.join("burned.mp4");
        let res3 = render_preview(
            &db,
            temp.path(),
            &ffmpeg,
            script_id,
            &PreviewOptions {
                burn_titles: true,
                burn_narration: true,
                normalize_audio: false,
                audio_fade_s: 0.12,
                speech_overrun_s: 0.35,
                out: Some(burned_out.clone()),
                cancel: None,
            },
            |_, _| {},
        )
        .unwrap();
        assert_eq!(res3.path, burned_out);
        assert!(burned_out.is_file());

        // Probe both outputs with ffprobe
        for file in [&res1.path, &burned_out] {
            let probe_out = crate::proc::std_command(&ffprobe)
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type",
                    "-of",
                    "json",
                    file.to_str().unwrap(),
                ])
                .output()
                .unwrap();
            assert!(probe_out.status.success());
            let probe_json: serde_json::Value = serde_json::from_slice(&probe_out.stdout).unwrap();

            let streams = probe_json["streams"].as_array().unwrap();
            assert_eq!(streams.len(), 2, "must have 2 streams in {}", file.display());
            let v_streams = streams.iter().filter(|s| s["codec_type"] == "video").count();
            let a_streams = streams.iter().filter(|s| s["codec_type"] == "audio").count();
            assert_eq!(v_streams, 1, "must have 1 video stream");
            assert_eq!(a_streams, 1, "must have 1 audio stream");

            let dur_str = probe_json["format"]["duration"].as_str().unwrap();
            let dur: f64 = dur_str.parse().unwrap();
            assert!((dur - 3.5).abs() < 0.1, "duration {} should be ≈ 3.5s (tolerance 0.1s)", dur);
        }
    }
}
