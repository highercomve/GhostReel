//! Every audio track of a video, for the editor: what the tracks are, how loud each one is over a
//! stretch, and a small playable copy of one track for the live viewer.
//!
//! Pro cameras record several tracks per file (a lav, the camera mic, tracks left silent). The
//! index picks one of them for transcription; the editor shows them all and lets a clip choose.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use rusqlite::OptionalExtension;
use serde::Serialize;

use crate::Error;
use crate::db::Db;

/// What volumedetect reports for digital silence; used too when a track could not be measured.
pub const SILENCE_DB: f64 = -91.0;
/// Longest stretch measured for levels; longer ranges are sampled from the middle.
pub const MAX_LEVEL_SPAN_S: f64 = 20.0;

/// One audio stream of a file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioTrackInfo {
    /// 0-based among the file's audio streams (what `0:a:{index}` selects).
    pub index: u32,
    pub channels: u32,
    pub codec: String,
    /// The stream's title or handler name, when the file carries one.
    pub label: Option<String>,
    /// The track GhostReel plays when a clip does not choose.
    pub auto: bool,
}

/// A video's source file, its content hash and the track the index picked.
pub struct TrackSource {
    pub path: PathBuf,
    pub content_hash: String,
    pub auto_track: u32,
}

/// The first of a video's files that is still on disk.
pub fn track_source(db: &Db, video_id: i64) -> Result<TrackSource, Error> {
    let content_hash: Option<String> =
        db.conn.query_row("SELECT content_hash FROM videos WHERE id = ?1", [video_id], |r| r.get(0)).optional()?;
    let content_hash = content_hash.ok_or_else(|| Error::NotFound(format!("video #{video_id}")))?;
    let mut stmt = db.conn.prepare("SELECT path FROM video_files WHERE video_id = ?1 ORDER BY id")?;
    let paths: Vec<String> = stmt.query_map([video_id], |r| r.get(0))?.collect::<Result<_, _>>()?;
    let path = paths
        .iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
        .or_else(|| paths.first().map(PathBuf::from))
        .ok_or_else(|| Error::NotFound(format!("file for video #{video_id}")))?;
    Ok(TrackSource { path, content_hash, auto_track: db.audio_track(video_id).unwrap_or(0) })
}

/// Parse `ffprobe -show_streams -select_streams a -of json` output.
pub fn parse_tracks(json: &[u8], auto_track: u32) -> Result<Vec<AudioTrackInfo>, Error> {
    let info: serde_json::Value = serde_json::from_slice(json).map_err(|e| Error::Probe(format!("bad json: {e}")))?;
    let streams = info["streams"].as_array().cloned().unwrap_or_default();
    let mut tracks: Vec<AudioTrackInfo> = streams
        .iter()
        .filter(|s| s["codec_type"].as_str().is_none_or(|t| t == "audio"))
        .enumerate()
        .map(|(i, s)| {
            let tag = |k: &str| s["tags"][k].as_str().map(str::trim).filter(|v| !v.is_empty()).map(str::to_string);
            // Generic handler names say nothing about the track.
            let handler = tag("handler_name").filter(|h| !matches!(h.as_str(), "SoundHandler" | "Sound Handler"));
            AudioTrackInfo {
                index: i as u32,
                channels: s["channels"].as_u64().unwrap_or(0) as u32,
                codec: s["codec_name"].as_str().unwrap_or("unknown").to_string(),
                label: tag("title").or(handler),
                auto: false,
            }
        })
        .collect();
    let auto = if (auto_track as usize) < tracks.len() { auto_track as usize } else { 0 };
    if let Some(t) = tracks.get_mut(auto) {
        t.auto = true;
    }
    Ok(tracks)
}

/// The file's audio tracks, read from the header.
pub async fn list_tracks(ffprobe: &Path, video: &Path, auto_track: u32) -> Result<Vec<AudioTrackInfo>, Error> {
    let out = crate::proc::command(ffprobe)
        .args(["-v", "error", "-select_streams", "a", "-show_streams", "-of", "json"])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Probe(format!("{e}")))?;
    if !out.status.success() {
        return Err(Error::Probe(String::from_utf8_lossy(&out.stderr).trim().to_string()));
    }
    parse_tracks(&out.stdout, auto_track)
}

/// The stretch actually measured: at most [`MAX_LEVEL_SPAN_S`], from the middle of the range.
pub fn level_window(in_s: f64, out_s: f64) -> (f64, f64) {
    let (a, b) = (in_s.max(0.0), out_s.max(0.0));
    let (a, b) = if b < a { (b, a) } else { (a, b) };
    let span = (b - a).min(MAX_LEVEL_SPAN_S);
    let start = a + ((b - a) - span) / 2.0;
    (start, span.max(0.05))
}

/// One volumedetect per track in a single filtergraph, all sent to the null muxer.
pub fn level_args(tracks: u32) -> Vec<String> {
    let graph: Vec<String> = (0..tracks).map(|i| format!("[0:a:{i}]volumedetect[o{i}]")).collect();
    let mut args = vec!["-filter_complex".to_string(), graph.join(";")];
    for i in 0..tracks {
        args.extend(["-map".to_string(), format!("[o{i}]")]);
    }
    args.extend(["-f", "null", "-"].map(String::from));
    args
}

/// Per-track mean volume from ffmpeg's stderr. Filters in a `-filter_complex` are numbered in the
/// order written, so `Parsed_volumedetect_{i}` is track `i`.
pub fn parse_levels(stderr: &str, tracks: u32) -> Vec<f64> {
    let mut levels = vec![SILENCE_DB; tracks as usize];
    for line in stderr.lines() {
        let Some(rest) = line.split("Parsed_volumedetect_").nth(1) else { continue };
        let Some(value) = line.split("mean_volume:").nth(1) else { continue };
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let (Ok(i), Ok(db)) = (digits.parse::<usize>(), value.trim().trim_end_matches("dB").trim().parse::<f64>())
        else {
            continue;
        };
        if let Some(slot) = levels.get_mut(i) {
            *slot = if db.is_finite() { db.max(SILENCE_DB) } else { SILENCE_DB };
        }
    }
    levels
}

/// Mean volume in dB of every audio track over `in_s..out_s` of the source (capped, see
/// [`level_window`]). [`SILENCE_DB`] means silence or nothing measured.
pub async fn track_levels(ffmpeg: &Path, video: &Path, tracks: u32, in_s: f64, out_s: f64) -> Vec<f64> {
    if tracks == 0 {
        return Vec::new();
    }
    let (start, span) = level_window(in_s, out_s);
    let out = crate::proc::command(ffmpeg)
        // volumedetect reports at info level; no -v error.
        .args([
            "-nostdin",
            "-nostats",
            "-hide_banner",
            "-ss",
            &format!("{start:.3}"),
            "-t",
            &format!("{span:.3}"),
            "-i",
        ])
        .arg(video)
        .args(level_args(tracks))
        .kill_on_drop(true)
        .output()
        .await;
    match out {
        Ok(out) => parse_levels(&String::from_utf8_lossy(&out.stderr), tracks),
        Err(_) => vec![SILENCE_DB; tracks as usize],
    }
}

/// Where one track's playable copy is cached.
pub fn track_proxy_path(data_dir: &Path, content_hash: &str, track: u32) -> PathBuf {
    let hash: String =
        content_hash.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
    data_dir.join("proxies").join("audio").join(format!("{hash}_a{track}.m4a"))
}

static TRACK_LOCKS: OnceLock<std::sync::Mutex<HashMap<PathBuf, std::sync::Weak<tokio::sync::Mutex<()>>>>> =
    OnceLock::new();

fn lock_for(path: &Path) -> Arc<tokio::sync::Mutex<()>> {
    let mut map = TRACK_LOCKS.get_or_init(Default::default).lock().unwrap();
    map.retain(|_, l| l.strong_count() > 0);
    if let Some(l) = map.get(path).and_then(std::sync::Weak::upgrade) {
        return l;
    }
    let l = Arc::new(tokio::sync::Mutex::new(()));
    map.insert(path.to_path_buf(), Arc::downgrade(&l));
    l
}

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The whole of one track as a small AAC file (mono stays mono, more channels become stereo),
/// built once and reused. Concurrent callers for the same track wait for one encode.
pub async fn track_proxy(
    ffmpeg: &Path,
    data_dir: &Path,
    video: &Path,
    content_hash: &str,
    track: u32,
    channels: u32,
) -> Result<PathBuf, Error> {
    let out = track_proxy_path(data_dir, content_hash, track);
    let lock = lock_for(&out);
    let _guard = lock.lock().await;
    if out.metadata().is_ok_and(|m| m.len() > 0) {
        return Ok(out);
    }
    if let Some(dir) = out.parent() {
        tokio::fs::create_dir_all(dir).await.map_err(|e| Error::Io(dir.to_path_buf(), e))?;
    }
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let tmp = out.with_extension(format!("part-{}-{nanos}.m4a", std::process::id()));
    let _cleanup = TempFile(tmp.clone());
    let (ac, bitrate) = if channels <= 1 { ("1", "96k") } else { ("2", "128k") };
    let result = crate::proc::command(ffmpeg)
        .args(["-y", "-nostdin", "-v", "error", "-i"])
        .arg(video)
        .args(["-map", &format!("0:a:{track}"), "-vn", "-sn", "-dn", "-c:a", "aac", "-b:a", bitrate, "-ac", ac])
        .args(["-ar", "48000", "-movflags", "+faststart", "-f", "mp4"])
        .arg(&tmp)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Preview(format!("failed to run ffmpeg for audio track: {e}")))?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let tail = stderr.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(Error::Preview(format!("audio track extract failed: {tail}")));
    }
    tokio::fs::rename(&tmp, &out).await.map_err(|e| Error::Io(out.clone(), e))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_are_numbered_among_audio_streams_and_auto_marked() {
        let json = br#"{"streams":[
            {"index":1,"codec_name":"pcm_s24le","codec_type":"audio","channels":1,"tags":{"title":"Lav"}},
            {"index":2,"codec_name":"pcm_s24le","codec_type":"audio","channels":1,"tags":{"handler_name":"SoundHandler"}},
            {"index":3,"codec_name":"aac","codec_type":"audio","channels":2}
        ]}"#;
        let tracks = parse_tracks(json, 1).unwrap();
        assert_eq!(tracks.len(), 3);
        assert_eq!(tracks[0].index, 0);
        assert_eq!(tracks[0].label.as_deref(), Some("Lav"));
        assert_eq!(tracks[1].label, None);
        assert_eq!(tracks[2].channels, 2);
        assert_eq!(tracks[2].codec, "aac");
        assert_eq!(tracks.iter().filter(|t| t.auto).map(|t| t.index).collect::<Vec<_>>(), vec![1]);
        // An auto track the file doesn't have falls back to the first.
        assert!(parse_tracks(json, 9).unwrap()[0].auto);
        assert!(parse_tracks(br#"{"streams":[]}"#, 0).unwrap().is_empty());
    }

    #[test]
    fn levels_are_read_per_filter_and_default_to_silence() {
        let stderr = "\
[Parsed_volumedetect_1 @ 0x55] n_samples: 960000
[Parsed_volumedetect_1 @ 0x55] mean_volume: -91.0 dB
[Parsed_volumedetect_0 @ 0x54] mean_volume: -27.4 dB
[Parsed_volumedetect_0 @ 0x54] max_volume: -6.1 dB
[Parsed_volumedetect_2 @ 0x56] mean_volume: -inf dB";
        assert_eq!(parse_levels(stderr, 4), vec![-27.4, -91.0, SILENCE_DB, SILENCE_DB]);
    }

    #[test]
    fn level_args_map_every_track() {
        let args = level_args(2);
        assert_eq!(args[1], "[0:a:0]volumedetect[o0];[0:a:1]volumedetect[o1]");
        assert_eq!(&args[2..6], ["-map", "[o0]", "-map", "[o1]"]);
        assert_eq!(&args[6..], ["-f", "null", "-"]);
    }

    #[test]
    fn long_ranges_are_sampled_from_the_middle() {
        assert_eq!(level_window(10.0, 15.0), (10.0, 5.0));
        assert_eq!(level_window(0.0, 100.0), (40.0, 20.0));
        assert_eq!(level_window(15.0, 10.0), (10.0, 5.0));
    }

    #[test]
    fn proxy_paths_are_file_name_safe() {
        let p = track_proxy_path(Path::new("/d"), "b3e:ab12", 2);
        assert_eq!(p, Path::new("/d/proxies/audio/b3e-ab12_a2.m4a"));
    }
}
