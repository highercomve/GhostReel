//! Timeline preview: proxy cache, concat demuxing, and title/narration burning (plan §4a, M8b).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use rusqlite::{OptionalExtension, params};
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
    /// Tracks the script chose for this audio (see `ScriptClip::tracks`); None = `audio_track`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracks: Option<Vec<u32>>,
    /// How the picture is cut to the cut's shape (the clip's framing, else the script's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framing: Option<crate::framing::Framing>,
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
    /// Tracks the script chose for this audio (see `ScriptClip::tracks`); None = `audio_track`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracks: Option<Vec<u32>>,
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
    pub burn_captions: bool,
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
    /// How big the frames are: a light 540p preview, or the cut's own size for a finished export.
    pub size: RenderSize,
}

/// Output frame size of a render.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderSize {
    /// 540 px tall: quick to build, for watching the cut while editing.
    #[default]
    Preview,
    /// The cut's width × height (e.g. 1080×1920 for a vertical clip): what gets delivered.
    Full,
}

impl RenderSize {
    /// Frame size for a cut of `w`×`h`, both even (h264 needs it).
    pub fn dimensions(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            RenderSize::Preview => proxy_dimensions(w, h),
            RenderSize::Full if w >= 2 && h >= 2 => (w & !1, h & !1),
            RenderSize::Full => proxy_dimensions(w, h),
        }
    }
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
    proxy_file_name_with_audio(content_hash, in_s, out_s, fps, w, h, false, 0.0, None)
}

/// [`proxy_file_name`], distinguished also by how the audio was treated.
///
/// A proxy holds encoded audio, so two clips that differ only in levelling or fade are different
/// files. Without this a rendered clip kept whatever audio it was first built with, and changing
/// the setting appeared to do nothing. A clip's own track selection (`tracks`) is part of the
/// name too; `None` (the automatic track) leaves the name as it always was.
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
    tracks: Option<&[u32]>,
) -> String {
    let hash: String =
        content_hash.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
    let in_ms = (in_s * 1000.0).round() as i64;
    let out_ms = (out_s * 1000.0).round() as i64;
    let mut audio = format!("{}{}", if normalized { "n" } else { "r" }, (fade_s * 1000.0).round() as i64);
    if let Some(tracks) = tracks {
        let mut sorted = tracks.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        audio.push_str("_t");
        audio.push_str(&sorted.iter().map(u32::to_string).collect::<Vec<_>>().join("-"));
    }
    format!("{hash}_{in_ms}_{out_ms}_{}_{}_{w}x{h}_{audio}.mp4", fps.num, fps.den)
}

/// The source tracks a clip or bed plays: its own choice, without repeats or tracks the file does
/// not have (when the count is known), else the automatic track. Empty = silent.
pub fn effective_tracks(tracks: Option<&[u32]>, auto_track: u32, stream_count: Option<u32>) -> Vec<u32> {
    let Some(tracks) = tracks else { return vec![auto_track] };
    let mut out: Vec<u32> = Vec::with_capacity(tracks.len());
    for &t in tracks {
        if stream_count.is_none_or(|n| t < n) && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// ffmpeg arguments that take the audio of input 0 from `tracks` and run it through `af`. One
/// track is a plain map; several are summed first (`normalize=0`: each keeps its own level, as
/// they would on a mixer). `tracks` must not be empty.
pub fn audio_map_args(tracks: &[u32], af: &str) -> Vec<String> {
    match tracks {
        [one] => vec!["-map".into(), format!("0:a:{one}"), "-af".into(), af.into()],
        many => {
            let inputs: String = many.iter().map(|t| format!("[0:a:{t}]")).collect();
            let graph = format!("{inputs}amix=inputs={}:normalize=0:dropout_transition=0,{af}[mixa]", many.len());
            vec!["-filter_complex".into(), graph, "-map".into(), "[mixa]".into()]
        }
    }
}

/// How many audio streams a file has, from its header. `None` when ffprobe is missing or fails;
/// callers then trust the indexes they were given.
fn audio_stream_count(ffprobe: Option<&Path>, path: &Path) -> Option<u32> {
    let out = crate::proc::std_command(ffprobe?)
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.trim().is_empty()).count() as u32)
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
                tracks: clip.tracks.clone(),
                framing: crate::framing::effective(script, clip).cloned(),
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
            tracks: bed.tracks.clone(),
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

/// Format seconds into SubStation Alpha timestamp format `H:MM:SS.cc`.
pub fn format_ass_time(sec: f64) -> String {
    let total_cs = (sec.max(0.0) * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_s = total_cs / 100;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

/// Caption font size and bottom margin for a frame. A vertical frame (9:16) is watched in an app
/// whose buttons and description cover the bottom fifth, so captions sit above that, and smaller
/// type keeps a line to a few words on the narrow frame.
pub fn caption_metrics(width: u32, height: u32) -> (u32, u32) {
    let h = height as f64;
    if height > width {
        ((h * 0.042).round().max(12.0) as u32, (h * 0.22).round().max(10.0) as u32)
    } else {
        ((h * 0.055).round().max(12.0) as u32, (h * 0.08).round().max(10.0) as u32)
    }
}

/// Break a burned title into lines that fit the frame. drawtext does not wrap, and on a narrow
/// (9:16) frame a one-line title runs off both edges. The font size matches the title filter's
/// `min(h/12, w/14)`; a glyph averages about 0.55 of the font size.
pub fn wrap_title(text: &str, width: u32, height: u32) -> String {
    let font = (height as f64 / 12.0).min(width as f64 / 14.0);
    let max_chars = ((width as f64 * 0.9) / (font * 0.55)).floor().max(8.0) as usize;
    let mut lines: Vec<String> = Vec::new();
    for para in text.lines() {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > max_chars {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    lines.join("\n")
}

/// Generate Advanced SubStation Alpha (.ass) content from speech caption cues.
pub fn generate_captions_ass(cues: &[crate::captions::Cue], width: u32, height: u32) -> String {
    generate_captions_ass_styled(cues, width, height, None, false)
}

/// `generate_captions_ass` with a cut's caption style: position (bottom / center) and size.
/// Sized from the shorter side so a 9:16 frame gets captions as readable as a 16:9 one.
/// `at_seam` (a stacked layout) puts them on the middle of the frame, the seam between the two
/// faces, where they cover neither.
pub fn generate_captions_ass_styled(
    cues: &[crate::captions::Cue],
    width: u32,
    height: u32,
    style: Option<&crate::script::CaptionStyle>,
    at_seam: bool,
) -> String {
    let scale = style.map(|s| s.size).filter(|s| *s > 0.2 && *s < 4.0).unwrap_or(1.0);
    let basis = if height > width { (width as f64) * 0.95 } else { height as f64 };
    let font_size = (basis * 0.055 * scale).round().max(12.0) as u32;
    let center = at_seam || style.is_some_and(|s| s.position == "center");
    // ASS alignment: 2 = bottom centre, 5 = middle centre (MarginV is then ignored). In 9:16 the
    // bottom margin clears the platform UI (the lower fifth).
    let alignment = if center { 5 } else { 2 };
    let bottom = if height > width { 0.2 } else { 0.08 };
    let margin_v = if center { 0 } else { ((height as f64) * bottom).round().max(10.0) as u32 };
    let margin_h = ((width as f64) * 0.05).round().max(10.0) as u32;

    let mut ass = format!(
        "[Script Info]\n\
         ScriptType: v4.00+\n\
         PlayResX: {width}\n\
         PlayResY: {height}\n\
         ScaledBorderAndShadow: yes\n\n\
         [V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
         Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1.5,{alignment},{margin_h},{margin_h},{margin_v},1\n\n\
         [Events]\n\
         Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    );

    for cue in cues {
        let start = format_ass_time(cue.start_s);
        let end = format_ass_time(cue.end_s);
        let mut lines: Vec<&str> = cue.text.lines().collect();
        if lines.len() > 2 {
            lines.truncate(2);
        }
        let text = lines.join("\\N");
        use std::fmt::Write;
        writeln!(&mut ass, "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}").unwrap();
    }

    ass
}

/// Greedy word wrap to lines of at most `max` characters (a longer word gets its own line).
pub fn wrap_words(text: &str, max: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        match lines.last_mut() {
            Some(l) if l.chars().count() + 1 + word.chars().count() <= max => {
                l.push(' ');
                l.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines.join("\n")
}

/// Assemble the `-vf` filtergraph for titles, narration subtitles, and speech captions.
pub fn build_burn_filtergraph(
    titles: &[(String, f64, f64)],
    font: Option<&str>,
    narration_path: Option<&str>,
    captions_path: Option<&str>,
    _captions_margin_v: u32,
    narration_margin_v: u32,
) -> String {
    let font_clause = font.map(|f| format!(":font={f}")).unwrap_or_default();
    let mut filters: Vec<String> = titles
        .iter()
        .map(|(file, start, end)| {
            format!(
                "drawtext=textfile={file}{font_clause}:expansion=none:enable='between(t,{start:.3},{end:.3})':\
                 fontsize='min(h/12,w/14)':fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=12:\
                 x=(w-text_w)/2:y=h*0.08"
            )
        })
        .collect();
    if let Some(narration) = narration_path {
        if captions_path.is_some() {
            filters.push(format!("subtitles={narration}:force_style='MarginV={narration_margin_v}'"));
        } else {
            filters.push(format!("subtitles={narration}"));
        }
    }
    if let Some(captions) = captions_path {
        filters.push(format!("subtitles={captions}"));
    }
    filters.join(",")
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

/// Where a script version's preview is rendered unless the caller picks a path.
pub fn default_preview_path(data_dir: &Path, script_id: i64, version: i64) -> PathBuf {
    data_dir.join("previews").join(format!("script_{script_id}_v{version}.mp4"))
}

/// Delete the default preview render of a script version, and its caption sidecar. Best effort:
/// a file that is already gone is fine. Used when the script row is purged, so a later script
/// that reuses the id never picks up the old render.
pub fn remove_default_preview(data_dir: &Path, script_id: i64, version: i64) {
    let mp4 = default_preview_path(data_dir, script_id, version);
    for path in [mp4.with_extension("srt"), mp4] {
        if let Err(e) = std::fs::remove_file(&path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("cannot remove preview {}: {e}", path.display());
        }
    }
}

/// Check if an existing preview MP4 render exists on disk for this script.
pub fn find_preview(db: &Db, data_dir: &Path, script_id: i64) -> Result<Option<PathBuf>, Error> {
    let path: Option<String> = db
        .conn
        .query_row(
            "SELECT path FROM exports WHERE script_id = ?1 AND format = 'preview_mp4' ORDER BY created_at DESC LIMIT 1",
            params![script_id],
            |r| r.get(0),
        )
        .optional()?;

    if let Some(p_str) = path {
        let p = PathBuf::from(p_str);
        if p.is_file() {
            return Ok(Some(p));
        }
    }

    if let Ok(stored) = script::load(db, script_id) {
        let candidate = default_preview_path(data_dir, stored.id, stored.version);
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }

    Ok(None)
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
    let (proxy_w, proxy_h) = opts.size.dimensions(proj_w, proj_h);

    let media = otio::resolve_media_for_script(db, stored.project_id, &stored.script, project_fps)?;
    let segments = plan_segments(&stored.script, &media)?;
    if segments.is_empty() {
        return Err(Error::Preview("script has no clips to preview".into()));
    }

    let beat_spans = plan_beat_spans(&stored.script, &segments);
    // Only explicit track choices need the stream count, so files are probed lazily, once each.
    let ffprobe = crate::doctor::locate("ffprobe");
    let mut stream_counts: HashMap<i64, Option<u32>> = HashMap::new();
    let mut stream_count = |video_id: i64, path: &Path| {
        *stream_counts.entry(video_id).or_insert_with(|| audio_stream_count(ffprobe.as_deref(), path))
    };
    // A bed whose chosen tracks are all silent (or missing) lays nothing: its pictures are muted
    // already, so the beat simply plays silent.
    let bed_spans: Vec<(BedSpan, Vec<u32>)> = plan_bed_spans(&stored.script, &segments, &media)
        .into_iter()
        .filter_map(|bed| {
            let count = if bed.tracks.is_some() { stream_count(bed.video_id, &bed.path) } else { None };
            let chosen = effective_tracks(bed.tracks.as_deref(), bed.audio_track, count);
            (!chosen.is_empty()).then_some((bed, chosen))
        })
        .collect();
    let encoder = pick_encoder(ffmpeg);
    // A cut that carries a caption style (clips do) always burns its speech captions.
    let burn_captions = opts.burn_captions || stored.script.captions.is_some();

    let proxies_dir = data_dir.join("proxies");
    std::fs::create_dir_all(&proxies_dir).map_err(|e| Error::Io(proxies_dir.clone(), e))?;
    let previews_dir = data_dir.join("previews");
    std::fs::create_dir_all(&previews_dir).map_err(|e| Error::Io(previews_dir.clone(), e))?;

    let out_path =
        opts.out.clone().unwrap_or_else(|| default_preview_path(data_dir, stored.id, stored.version));
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::Io(parent.to_path_buf(), e))?;
    }

    let total_dur: f64 = segments.iter().map(|s| (s.out_s - s.in_s).max(0.0)).sum();
    let mut completed_dur = 0.0;
    let mut proxies_built = 0;
    let mut proxies_cached = 0;
    let mut proxy_paths = Vec::new();

    let g_val = project_fps.as_f64().round().max(1.0) as i64;
    let fps_tail = format!("fps={}/{},format=yuv420p", project_fps.num, project_fps.den);
    let letterbox_vf = format!(
        "scale={proxy_w}:{proxy_h}:force_original_aspect_ratio=decrease,pad={proxy_w}:{proxy_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,{fps_tail}"
    );

    for (i, seg) in segments.iter().enumerate() {
        if opts.cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst)) {
            return Err(Error::Preview("cancelled".into()));
        }

        let res = media
            .get(&seg.video_id)
            .ok_or_else(|| Error::NotFound(format!("resolved media for video #{}", seg.video_id)))?;

        // A framed clip is cropped along its path instead of letterboxed; the path is part of
        // the proxy's name, so moving a face or changing the layout builds a new one.
        let plan = match &seg.framing {
            Some(fr) => crate::framing::plan_clip(
                db,
                Some(ffmpeg),
                Some(&seg.path),
                seg.video_id,
                seg.in_s,
                seg.out_s,
                fr,
                proj_w as f64,
                proj_h as f64,
            )?,
            None => None,
        };
        let vf = match &plan {
            Some(p) => format!("{},{fps_tail}", p.ffmpeg_filter(proxy_w, proxy_h)),
            None => letterbox_vf.clone(),
        };
        let mut p_name = proxy_file_name_with_audio(
            &res.content_hash,
            seg.in_s,
            seg.out_s,
            project_fps,
            proxy_w,
            proxy_h,
            opts.normalize_audio,
            opts.audio_fade_s,
            // A silent segment is cached as "no tracks": sharing a name with the same range's
            // sounding version replayed whichever one was built first after a mute toggle.
            if seg.mute { Some(&[]) } else { seg.tracks.as_deref() },
        );
        if let Some(p) = &plan {
            p_name = p_name.replace(".mp4", &format!("_f{}.mp4", p.digest()));
        }
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
            let chosen = if seg.mute || !seg.has_audio {
                Vec::new()
            } else {
                let count = if seg.tracks.is_some() { stream_count(seg.video_id, &seg.path) } else { None };
                effective_tracks(seg.tracks.as_deref(), seg.audio_track, count)
            };
            if !chosen.is_empty() {
                cmd.args(["-map", "0:v:0"]);
                cmd.args(audio_map_args(&chosen, &af));
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

            let out = crate::cancel::output(&mut cmd, opts.cancel.as_ref())
                .map_err(ffmpeg_err("failed to run ffmpeg for proxy"))?;
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

    let cues = if burn_captions || opts.out.is_some() {
        crate::captions::cues_for_script(db, &stored.script)?
    } else {
        Vec::new()
    };
    let has_captions = burn_captions && !cues.is_empty();
    let has_titles = opts.burn_titles && beat_spans.iter().any(|b| b.on_screen_text.is_some());
    let has_narration = opts.burn_narration && beat_spans.iter().any(|b| b.narration.is_some());
    let burn_pass_needed = has_titles || has_narration || has_captions;

    let bed_pass_needed = !bed_spans.is_empty();
    let concat_output = if burn_pass_needed || bed_pass_needed {
        out_path.with_extension(format!("concat.tmp.{}.mp4", std::process::id()))
    } else {
        out_path.clone()
    };

    on_progress(0.86, "Concatenating segments…");

    let concat_res = crate::cancel::output(
        crate::proc::std_command(ffmpeg)
            .args(["-y", "-f", "concat", "-safe", "0", "-i"])
            .arg(&concat_list_path)
            .args(["-c", "copy", "-movflags", "+faststart"])
            .arg(&concat_output),
        opts.cancel.as_ref(),
    );

    let _ = std::fs::remove_file(&concat_list_path);

    let concat_out = concat_res.map_err(ffmpeg_err("failed to spawn ffmpeg concat"))?;
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
        for (i, (bed, chosen)) in bed_spans.iter().enumerate() {
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
                .args(audio_map_args(chosen, &af))
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

        let mix_out = crate::cancel::output(&mut cmd, opts.cancel.as_ref())
            .map_err(ffmpeg_err("failed to spawn audio bed mix"))?;
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
        on_progress(0.88, "Burning titles, subtitles and captions…");

        let mut temp_files_to_remove = Vec::new();
        let srt_escaped = if has_narration {
            let srt_path = out_path.parent().unwrap_or(Path::new(".")).join("narration.srt");
            std::fs::write(&srt_path, generate_narration_srt(&beat_spans))
                .map_err(|e| Error::Io(srt_path.clone(), e))?;
            temp_files_to_remove.push(srt_path.clone());
            Some(escape_filter_path(&srt_path))
        } else {
            None
        };
        // Titles and captions share one ASS file, each line at an explicit position, laid out
        // exactly as the editor's live viewer draws them.
        let overlays_escaped = if has_titles || has_captions {
            let mut ov = crate::overlays::layout(&stored.script, if has_captions { &cues } else { &[] });
            if !has_titles {
                ov.titles.clear();
            }
            let ass_path = out_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("overlays_{}_{}.ass", stored.id, std::process::id()));
            std::fs::write(&ass_path, crate::overlays::to_ass(&ov, proxy_w, proxy_h))
                .map_err(|e| Error::Io(ass_path.clone(), e))?;
            temp_files_to_remove.push(ass_path.clone());
            Some(escape_filter_path(&ass_path))
        } else {
            None
        };

        let (caption_font_size, captions_margin_v) = caption_metrics(proxy_w, proxy_h);
        let narration_margin_v = captions_margin_v + (caption_font_size as f64 * 2.2).round() as u32 + 10;

        // Narration moves up out of the captions' way only when captions are burnt.
        let mut graph = build_burn_filtergraph(
            &[],
            None,
            srt_escaped.as_deref(),
            overlays_escaped.as_deref().filter(|_| has_captions),
            captions_margin_v,
            narration_margin_v,
        );
        if let Some(ov) = overlays_escaped.as_deref().filter(|_| !has_captions) {
            if !graph.is_empty() {
                graph.push(',');
            }
            graph.push_str(&format!("subtitles={ov}"));
        }
        let burn = |graph: String| {
            let mut cmd = crate::proc::std_command(ffmpeg);
            cmd.args(["-y", "-i"]).arg(&burn_input).args(["-vf", &graph]);
            cmd.args(&encoder.args);
            cmd.args(["-c:a", "copy", "-movflags", "+faststart"]).arg(&out_path);
            crate::cancel::output(&mut cmd, opts.cancel.as_ref())
        };

        let burn_res = burn(graph);
        for tmp in &temp_files_to_remove {
            let _ = std::fs::remove_file(tmp);
        }
        let _ = std::fs::remove_file(&burn_input);

        let burn_out = burn_res.map_err(ffmpeg_err("failed to spawn burn pass"))?;
        if !burn_out.status.success() {
            let stderr = String::from_utf8_lossy(&burn_out.stderr);
            let tail =
                stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            return Err(Error::Preview(format!("burn pass failed: {tail}")));
        }
    }

    // Sidecar .srt for export when burn_captions is on
    if burn_captions && opts.out.is_some() {
        let sidecar_path = out_path.with_extension("srt");
        let srt_text = crate::captions::to_srt(&cues);
        std::fs::write(&sidecar_path, srt_text).map_err(|e| Error::Io(sidecar_path.clone(), e))?;
    }

    // Insert exports row
    let now = crate::projects::now();
    db.conn.execute(
        "INSERT INTO exports(script_id, format, path, created_at) VALUES (?1, 'preview_mp4', ?2, ?3)",
        params![stored.id, out_path.to_string_lossy(), now],
    )?;
    // Our own output: a scan of a folder it is saved into must not index it as footage.
    crate::index::record_render(db, &out_path, Some(stored.id))?;

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
    playback_proxy_path_for_size(data_dir, content_hash, 768)
}

/// What sets current playback proxies apart from older formats, which nothing reads any more.
pub const PLAYBACK_PROXY_TAG: &str = "-playback-v2-";

pub fn playback_proxy_path_for_size(data_dir: &Path, content_hash: &str, long_side: u32) -> PathBuf {
    let long_side = long_side.clamp(256, 3840);
    let hex = content_hash.rsplit(':').next().unwrap_or(content_hash);
    data_dir.join("proxies").join(&hex[..2.min(hex.len())]).join(format!("{hex}{PLAYBACK_PROXY_TAG}{long_side}.mp4"))
}

/// Whether a browser engine can be expected to play this file as it is: 8-bit 4:2:0 H.264 at
/// 1080p or less. Camera originals — 4K, 10-bit 4:2:2, HEVC — are none of that, and WebKit
/// software-decodes them so slowly the player sits black for a long time.
pub async fn plays_natively(ffprobe: &Path, video: &Path) -> bool {
    // The elementary codec alone does not make camera containers or PCM audio browser-playable.
    let extension = video.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if !matches!(extension.as_str(), "mp4" | "m4v" | "mov") {
        return false;
    }
    let Ok(out) = crate::proc::command(ffprobe)
        .args(["-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt,height", "-of", "json"])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let Ok(info) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else {
        return false;
    };
    let Some(streams) = info["streams"].as_array() else {
        return false;
    };
    let video_ok = streams.iter().find(|s| s["codec_type"] == "video").is_some_and(|s| {
        s["codec_name"] == "h264" && s["pix_fmt"] == "yuv420p" && s["height"].as_u64().is_some_and(|h| h <= 1080)
    });
    let audio_ok = streams
        .iter()
        .filter(|s| s["codec_type"] == "audio")
        .all(|s| matches!(s["codec_name"].as_str(), Some("aac" | "mp3")));
    video_ok && audio_ok
}

#[derive(Debug, Clone, Serialize)]
pub struct PlaybackProgress {
    pub phase: String,
    pub fraction: Option<f64>,
    pub encoded_s: f64,
    pub encoder: Option<String>,
}

struct ProxyJob {
    gate: tokio::sync::Mutex<()>,
    progress: std::sync::Mutex<PlaybackProgress>,
}

static PROXY_LOCKS: OnceLock<std::sync::Mutex<HashMap<PathBuf, std::sync::Weak<ProxyJob>>>> = OnceLock::new();

fn proxy_job_for(path: &Path) -> Arc<ProxyJob> {
    let mut map = PROXY_LOCKS.get_or_init(Default::default).lock().unwrap();
    map.retain(|_, job| job.strong_count() > 0);
    if let Some(job) = map.get(path).and_then(std::sync::Weak::upgrade) {
        return job;
    }
    let job = Arc::new(ProxyJob {
        gate: tokio::sync::Mutex::new(()),
        progress: std::sync::Mutex::new(PlaybackProgress {
            phase: "Checking playable copy…".into(),
            fraction: None,
            encoded_s: 0.0,
            encoder: None,
        }),
    });
    map.insert(path.to_path_buf(), Arc::downgrade(&job));
    job
}

/// Shared progress for an active encode, including callers waiting for the same proxy.
pub fn playback_proxy_progress(data_dir: &Path, hash: &str, long_side: u32) -> Option<PlaybackProgress> {
    let map = PROXY_LOCKS.get()?.lock().ok()?;
    let job = map.get(&playback_proxy_path_for_size(data_dir, hash, long_side))?.upgrade()?;
    let progress = job.progress.lock().ok()?.clone();
    Some(progress)
}

fn encoding_progress(line: &str, duration_s: Option<f64>) -> Option<(f64, Option<f64>)> {
    let micros = line.strip_prefix("out_time_us=")?.parse::<f64>().ok()?;
    if !micros.is_finite() {
        return None;
    }
    let seconds = (micros / 1_000_000.0).max(0.0);
    let fraction = duration_s.filter(|d| d.is_finite() && *d > 0.0).map(|d| (seconds / d).clamp(0.0, 0.99));
    Some((seconds, fraction))
}

struct ProxyTemporaryFile(PathBuf);
impl Drop for ProxyTemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Inspect actual decoded video and audio, rather than trusting a possibly corrupt MP4 header.
/// This samples the start of the file; it is not a full-file integrity scan.
pub async fn validate_proxy_file(ffmpeg: &Path, path: &Path) -> bool {
    if !path.metadata().is_ok_and(|m| m.len() > 0) {
        return false;
    }
    let Ok(out) = crate::proc::command(ffmpeg)
        .args(["-v", "error", "-xerror", "-i"])
        .arg(path)
        .args(["-t", "1", "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"])
        .kill_on_drop(true)
        .output()
        .await
    else {
        return false;
    };
    out.status.success()
}

/// A copy of the video the player can actually play: 720p, 8-bit 4:2:0 H.264 with AAC audio,
/// built once and kept. The original stays the source for keyframes, previews and exports.
pub async fn playback_proxy(
    ffmpeg: &Path,
    data_dir: &Path,
    video: &Path,
    content_hash: &str,
) -> Result<PathBuf, Error> {
    playback_proxy_with_progress(ffmpeg, data_dir, video, content_hash, None, 768).await
}

pub async fn playback_proxy_with_progress(
    ffmpeg: &Path,
    data_dir: &Path,
    video: &Path,
    content_hash: &str,
    duration_s: Option<f64>,
    long_side: u32,
) -> Result<PathBuf, Error> {
    let long_side = long_side.clamp(256, 3840);
    let out = playback_proxy_path_for_size(data_dir, content_hash, long_side);

    // 1. Per-video locking: prevents concurrent duplicate encodes for the same content_hash
    let job = proxy_job_for(&out);
    let _guard = job.gate.lock().await;

    // 2. Cache validation & corrupted copy rebuilding
    if out.exists() {
        if validate_proxy_file(ffmpeg, &out).await {
            return Ok(out);
        }
        // Remove corrupted or incomplete proxy copy so it can be rebuilt cleanly
        let _ = tokio::fs::remove_file(&out).await;
    }

    if let Some(dir) = out.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    }

    // 3. Unique temporary file per attempt and process
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let tmp = out.with_extension(format!("part-{}-{}.mp4", std::process::id(), nanos));
    let _cleanup = ProxyTemporaryFile(tmp.clone());

    let enc = tokio::task::spawn_blocking({
        let ffmpeg = ffmpeg.to_path_buf();
        move || pick_encoder(&ffmpeg)
    })
    .await
    .map_err(|e| Error::Vision(format!("encoder detection: {e}")))?;
    *job.progress.lock().unwrap() = PlaybackProgress {
        phase: "Preparing a playable copy…".into(),
        fraction: duration_s.map(|_| 0.0),
        encoded_s: 0.0,
        encoder: Some(enc.name.clone()),
    };
    let mut cmd = crate::proc::command(ffmpeg);
    cmd.args(["-y", "-v", "error", "-progress", "pipe:1", "-nostats", "-i"])
        .arg(video)
        .args(["-vf", &format!("scale=w=\'min({long_side},iw)\':h=\'min({long_side},ih)\':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,format=yuv420p")])
        .args(["-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"])
        .args(enc.args.iter().map(String::as_str))
        .args(["-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart"])
        .arg(&tmp)
        .kill_on_drop(true);
    use tokio::io::AsyncBufReadExt;
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| Error::Vision(format!("ffmpeg: {e}")))?;
    let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
    let read_progress = async {
        while let Some(line) = lines.next_line().await? {
            if let Some((encoded_s, fraction)) = encoding_progress(&line, duration_s) {
                let mut progress = job.progress.lock().unwrap();
                progress.encoded_s = encoded_s.max(progress.encoded_s);
                progress.fraction = fraction;
            }
        }
        Ok::<_, std::io::Error>(())
    };
    // Drain stderr concurrently so FFmpeg cannot block on a full pipe.
    let (read, output) = tokio::join!(read_progress, child.wait_with_output());
    read.map_err(|e| Error::Vision(format!("proxy progress: {e}")))?;
    let status = output.map_err(|e| Error::Vision(format!("ffmpeg: {e}")))?;

    if !status.status.success() {
        let _ = tokio::fs::remove_file(&tmp).await;
        let msg: String = String::from_utf8_lossy(&status.stderr).chars().take(300).collect();
        return Err(Error::Vision(format!("proxy encode failed: {msg}")));
    }

    {
        let mut progress = job.progress.lock().unwrap();
        progress.phase = "Finishing playable copy…".into();
        progress.fraction = None;
    }
    // 4. Validate the temporary file before promoting to final cache destination
    if !validate_proxy_file(ffmpeg, &tmp).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(Error::Vision("proxy encode produced corrupt or unplayable output".into()));
    }

    tokio::fs::rename(&tmp, &out).await.map_err(|e| Error::Io(out.clone(), e))?;
    Ok(out)
}

/// An ffmpeg call's I/O error as a render error: a stop stays a stop (`preview: cancelled`).
fn ffmpeg_err(what: &'static str) -> impl Fn(std::io::Error) -> Error {
    move |e| {
        if crate::cancel::is_cancel_io(&e) {
            Error::Preview("cancelled".into())
        } else {
            Error::Preview(format!("{what}: {e}"))
        }
    }
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
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
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
                    title_layout: None,
                    clips: vec![
                        script::ScriptClip {
                            framing: None,
                            video_id: 1,
                            in_s: 0.0,
                            out_s: 1.0,
                            audio: Audio::Source,
                            why: None,
                            tracks: None,
                        },
                        script::ScriptClip {
                            framing: None,
                            video_id: 2,
                            in_s: 2.0,
                            out_s: 3.51,
                            audio: Audio::Mute,
                            why: None,
                            tracks: None,
                        },
                    ],
                    notes: None,
                    bed: None,
                },
                script::Beat {
                    id: "b2".into(),
                    purpose: "body".into(),
                    narration: Some("Voice 2".into()),
                    on_screen_text: None,
                    title_layout: None,
                    clips: vec![script::ScriptClip {
                        framing: None,
                        video_id: 1,
                        in_s: 5.0,
                        out_s: 7.0,
                        audio: Audio::Source,
                        why: None,
                        tracks: None,
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

    #[test]
    fn a_full_render_keeps_the_cut_size_and_a_preview_stays_light() {
        // A vertical clip exported for delivery is 1080x1920, not the 304x540 preview.
        assert_eq!(RenderSize::Full.dimensions(1080, 1920), (1080, 1920));
        assert_eq!(RenderSize::Preview.dimensions(1080, 1920), (304, 540));
        assert_eq!(RenderSize::Full.dimensions(1081, 1921), (1080, 1920));
        assert_eq!(RenderSize::default(), RenderSize::Preview);
    }

    /// Two clips that differ only in how their audio was treated are different files: a proxy
    /// carries encoded audio, and reusing one silently kept the old levelling and fade.
    #[test]
    fn audio_treatment_gives_a_proxy_its_own_name() {
        let plain = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, false, 0.0, None);
        let faded = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, false, 0.12, None);
        let levelled = proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, true, 0.12, None);
        assert_ne!(plain, faded);
        assert_ne!(faded, levelled);
    }

    /// A clip's track choice changes its sound, so it changes the proxy; the automatic track
    /// keeps the old name, and order or repeats in the choice don't matter.
    #[test]
    fn track_selection_gives_a_proxy_its_own_name() {
        let name =
            |t: Option<&[u32]>| proxy_file_name_with_audio("h", 0.0, 1.0, Fps::new(25, 1), 960, 540, false, 0.0, t);
        assert_eq!(name(None), "h_0_1000_25_1_960x540_r0.mp4");
        assert_eq!(name(Some(&[0, 1])), "h_0_1000_25_1_960x540_r0_t0-1.mp4");
        assert_eq!(name(Some(&[1, 0, 1])), name(Some(&[0, 1])));
        assert_ne!(name(Some(&[0])), name(None));
        assert_ne!(name(Some(&[])), name(Some(&[0])));
    }

    #[test]
    fn effective_tracks_drop_repeats_and_missing_streams() {
        assert_eq!(effective_tracks(None, 2, Some(4)), vec![2]);
        assert_eq!(effective_tracks(Some(&[1, 0, 1]), 2, Some(4)), vec![1, 0]);
        assert_eq!(effective_tracks(Some(&[0, 5]), 0, Some(4)), vec![0]);
        assert_eq!(effective_tracks(Some(&[5]), 0, Some(4)), Vec::<u32>::new());
        assert_eq!(effective_tracks(Some(&[5]), 0, None), vec![5]);
        assert!(effective_tracks(Some(&[]), 0, Some(4)).is_empty());
    }

    #[test]
    fn several_tracks_are_mixed_before_the_audio_chain() {
        assert_eq!(audio_map_args(&[1], "apad"), ["-map", "0:a:1", "-af", "apad"]);
        assert_eq!(
            audio_map_args(&[0, 1], "afade=t=in:st=0:d=0.100,apad"),
            [
                "-filter_complex",
                "[0:a:0][0:a:1]amix=inputs=2:normalize=0:dropout_transition=0,afade=t=in:st=0:d=0.100,apad[mixa]",
                "-map",
                "[mixa]"
            ]
        );
    }

    #[test]
    fn planned_segments_and_beds_carry_the_track_choice() {
        let script: Script = serde_json::from_value(serde_json::json!({
            "title": "t", "beats": [{
                "id": "b1", "purpose": "p",
                "clips": [{"video_id": 1, "in_s": 0.0, "out_s": 2.0, "tracks": [0, 1]}],
                "bed": {"video_id": 1, "in_s": 5.0, "out_s": 7.0, "tracks": []}
            }]
        }))
        .unwrap();
        let media = HashMap::from([(
            1,
            ResolvedMedia {
                video_id: 1,
                path: PathBuf::from("/x.mxf"),
                duration_s: 60.0,
                has_audio: true,
                fps: Fps::new(25, 1),
                content_hash: "h".into(),
                audio_track: 1,
            },
        )]);
        let segs = plan_segments(&script, &media).unwrap();
        assert_eq!(segs[0].tracks, Some(vec![0, 1]));
        let beds = plan_bed_spans(&script, &segs, &media);
        assert_eq!(beds[0].tracks, Some(vec![]));
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
                pipeline: None,
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
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
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
                    title_layout: None,
                    clips: vec![script::ScriptClip {
                        framing: None,
                        video_id: 1,
                        in_s: 0.5,
                        out_s: 2.5,
                        audio: Audio::Source,
                        why: None,
                        tracks: None,
                    }],
                    notes: None,
                    bed: None,
                },
                script::Beat {
                    id: "b2".into(),
                    purpose: "outro".into(),
                    narration: None,
                    on_screen_text: None,
                    title_layout: None,
                    clips: vec![script::ScriptClip {
                        framing: None,
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 1.5,
                        audio: Audio::Source,
                        why: None,
                        tracks: None,
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
                burn_captions: false,
                normalize_audio: false,
                audio_fade_s: 0.12,
                speech_overrun_s: 0.35,
                out: Some(burned_out.clone()),
                cancel: None,
                size: RenderSize::Preview,
            },
            |_, _| {},
        )
        .unwrap();
        assert_eq!(res3.path, burned_out);
        assert!(burned_out.is_file());

        // Fourth render with burn_titles + burn_captions + sidecar .srt export
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 0.5, 2.5, 'Captioned beat 1')",
                [],
            )
            .unwrap();
        let export_out = data_dir.join("export_captions.mp4");
        let res4 = render_preview(
            &db,
            temp.path(),
            &ffmpeg,
            script_id,
            &PreviewOptions {
                burn_titles: true,
                burn_narration: false,
                burn_captions: true,
                normalize_audio: false,
                audio_fade_s: 0.12,
                speech_overrun_s: 0.35,
                out: Some(export_out.clone()),
                cancel: None,
                size: RenderSize::Preview,
            },
            |_, _| {},
        )
        .unwrap();
        assert_eq!(res4.path, export_out);
        assert!(export_out.is_file());
        let sidecar_srt = export_out.with_extension("srt");
        assert!(sidecar_srt.is_file(), "sidecar .srt must exist next to export MP4");
        let srt_text = std::fs::read_to_string(&sidecar_srt).unwrap();
        assert!(srt_text.contains("Captioned beat 1"));

        // Fifth: a moved, restyled title and a hand-edited caption burn, and the sidecar keeps the edit.
        let mut edited = script.clone();
        edited.beats[0].title_layout = Some(script::TitleLayout {
            start_s: Some(0.5),
            end_s: Some(1.5),
            x: Some(0.06),
            y: Some(0.78),
            style: Some(script::TextStyle { align: Some("left".into()), bold: Some(true), ..Default::default() }),
        });
        edited.captions = Some(script::CaptionStyle { y: Some(0.3), ..Default::default() });
        edited.caption_edits = vec![script::CueEdit {
            video_id: 1,
            src_start_s: 0.5,
            src_end_s: 2.5,
            text: Some("Edited {caption}".into()),
            start_s: Some(1.0),
            end_s: None,
            hidden: false,
        }];
        let edited_id = script::save_version(&db, project.id, &edited, None).unwrap();
        let edited_out = data_dir.join("edited.mp4");
        render_preview(
            &db,
            temp.path(),
            &ffmpeg,
            edited_id,
            &PreviewOptions {
                burn_titles: true,
                burn_captions: true,
                out: Some(edited_out.clone()),
                ..PreviewOptions::default()
            },
            |_, _| {},
        )
        .unwrap();
        let srt_text = std::fs::read_to_string(edited_out.with_extension("srt")).unwrap();
        assert!(srt_text.contains("00:00:00,500 --> 00:00:02,000\nEdited {caption}"), "{srt_text}");

        // Probe outputs with ffprobe
        for file in [&res1.path, &burned_out, &export_out] {
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

    #[test]
    fn progress_uses_encoded_time_and_reserves_completion() {
        assert_eq!(encoding_progress("out_time_us=2500000", Some(10.0)), Some((2.5, Some(0.25))));
        assert_eq!(encoding_progress("out_time_us=12000000", Some(10.0)), Some((12.0, Some(0.99))));
        assert_eq!(encoding_progress("out_time_us=1000000", None), Some((1.0, None)));
        assert_eq!(encoding_progress("out_time_us=N/A", Some(10.0)), None);
        assert_eq!(encoding_progress("out_time_us=NaN", Some(10.0)), None);
    }

    #[test]
    fn titles_wrap_for_narrow_frames() {
        assert_eq!(wrap_words("Mi tiempo vale más que diez dólares", 16), "Mi tiempo vale\nmás que diez\ndólares");
        assert_eq!(wrap_words("corto", 16), "corto");
    }

    #[test]
    fn test_burn_captions_filtergraph_and_timing() {
        let cues = vec![
            crate::captions::Cue { start_s: 1.25, end_s: 3.5, text: "First line\nSecond line".into(), ..Default::default() },
            crate::captions::Cue { start_s: 4.0, end_s: 6.75, text: "Single line caption".into(), ..Default::default() },
        ];

        // 1. ASS text has the right times and formatting
        let ass = generate_captions_ass(&cues, 960, 540);
        assert!(ass.contains("Dialogue: 0,0:00:01.25,0:00:03.50,Default,,0,0,0,,First line\\NSecond line"));
        assert!(ass.contains("Dialogue: 0,0:00:04.00,0:00:06.75,Default,,0,0,0,,Single line caption"));
        assert!(ass.contains("PlayResX: 960"));
        assert!(ass.contains("PlayResY: 540"));
        // Style check: bold, white, black outline 3, margin_v ~ 8% of 540 = 43, font size ~ 5.5% of 540 = 30
        assert!(ass.contains("Style: Default,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1.5,2,48,48,43,1"));

        // SRT text has the right times
        let srt = crate::captions::to_srt(&cues);
        assert!(srt.contains("1\n00:00:01,250 --> 00:00:03,500\nFirst line\nSecond line\n\n"));
        assert!(srt.contains("2\n00:00:04,000 --> 00:00:06,750\nSingle line caption\n\n"));

        // 2. Filtergraph with burn_captions on and cues exist
        let cap_path = "escaped_captions.ass";
        let narr_path = "escaped_narration.srt";
        let fg_captions_only = build_burn_filtergraph(&[], None, None, Some(cap_path), 43, 119);
        assert!(fg_captions_only.contains("subtitles=escaped_captions.ass"));
        assert!(!fg_captions_only.contains("subtitles=escaped_narration.srt"));

        // 3. Filtergraph when burn_captions is off (no captions)
        let fg_narr_only = build_burn_filtergraph(&[], None, Some(narr_path), None, 43, 119);
        assert!(!fg_narr_only.contains("escaped_captions.ass"));
        assert!(fg_narr_only.contains("subtitles=escaped_narration.srt"));
        assert!(!fg_narr_only.contains("force_style='MarginV=119'"));

        // 4. Captions + narration together use different MarginV
        let fg_both = build_burn_filtergraph(&[], None, Some(narr_path), Some(cap_path), 43, 119);
        assert!(fg_both.contains("subtitles=escaped_narration.srt:force_style='MarginV=119'"));
        assert!(fg_both.contains("subtitles=escaped_captions.ass"));
        // Captions ASS uses MarginV 43, narration uses MarginV 119
        assert_ne!(43, 119);
    }

    #[tokio::test]
    async fn concurrent_playback_encodes_share_a_valid_resolution_specific_copy() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else {
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.mp4");
        let generated = crate::proc::command(&ffmpeg)
            .args(["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=960x540:rate=25:duration=2", "-c:v", "libx264"])
            .arg(&source)
            .status()
            .await
            .unwrap();
        assert!(generated.success());
        let hash = "b3e:concurrent";
        let (a, b) = tokio::join!(
            playback_proxy_with_progress(&ffmpeg, tmp.path(), &source, hash, Some(2.0), 512),
            playback_proxy_with_progress(&ffmpeg, tmp.path(), &source, hash, Some(2.0), 512)
        );
        let a = a.unwrap();
        assert_eq!(a, b.unwrap());
        assert!(validate_proxy_file(&ffmpeg, &a).await);
        assert_ne!(a, playback_proxy_path_for_size(tmp.path(), hash, 768));
        let ffprobe = crate::doctor::locate("ffprobe").unwrap();
        let out = crate::proc::command(ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
            .arg(&a)
            .output()
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "512,288");
        // Preserve the container metadata while corrupting its media data: header-only checks miss this.
        let mut bytes = std::fs::read(&a).unwrap();
        let mdat = bytes.windows(4).position(|w| w == b"mdat").unwrap();
        let end = (mdat + 4 + 8192).min(bytes.len());
        bytes[mdat + 4..end].fill(0);
        std::fs::write(&a, bytes).unwrap();
        assert!(!validate_proxy_file(&ffmpeg, &a).await);
        let rebuilt = playback_proxy_with_progress(&ffmpeg, tmp.path(), &source, hash, Some(2.0), 512).await.unwrap();
        assert!(validate_proxy_file(&ffmpeg, &rebuilt).await);
    }

    #[tokio::test]
    async fn test_playback_proxy_cache_validation_and_rebuilding() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else {
            return;
        };
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Create a fake corrupted proxy file (invalid data)
        let hash = "b3e:corrupt123";
        let proxy_path = playback_proxy_path(&data_dir, hash);
        std::fs::create_dir_all(proxy_path.parent().unwrap()).unwrap();
        std::fs::write(&proxy_path, "corrupted partial file with no moov atom").unwrap();

        // validate_proxy_file should detect corruption
        assert!(!validate_proxy_file(&ffmpeg, &proxy_path).await);

        // Generate a small 1s source video
        let video_path = temp.path().join("source.mp4");
        let status = crate::proc::command(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25:duration=1",
                "-c:v",
                "libx264",
                "-f",
                "mp4",
            ])
            .arg(&video_path)
            .status()
            .await
            .unwrap();
        assert!(status.success());

        // playback_proxy must detect the corrupted cache file, remove it, and rebuild a valid one
        let built_proxy = playback_proxy(&ffmpeg, &data_dir, &video_path, hash).await.unwrap();
        assert!(built_proxy.exists());
        assert!(validate_proxy_file(&ffmpeg, &built_proxy).await);
    }
}
