//! Which audio track to use, and who is close to the microphone.
//!
//! Two facts about a video's sound that nothing else in the index knows, and that a script is
//! poor without:
//!
//! - A field recording often carries several tracks — a camera mic, one or two lavs, and tracks
//!   left silent. Taking the first one is a guess: it can be the roomy camera mic, or nothing at
//!   all. The one with usable speech is measured instead.
//! - In an interview the subject is on a lav and the interviewer is off-mic across the room.
//!   Their speech is measurably quieter — 12 dB apart on this project's footage — so the slate
//!   and the questions can be told from the answers without reading a word.

use std::path::Path;

use crate::Error;

/// Below this mean level a track is silence rather than a quiet recording.
const SILENT_DBFS: f64 = -70.0;
/// A segment this far under the video's own speech level is someone away from the microphone.
pub const OFF_MIC_MARGIN_DB: f64 = 8.0;

/// Mean level of a stretch of one audio track, in dBFS. `None` when ffmpeg reported nothing.
pub async fn mean_level(ffmpeg: &Path, video: &Path, stream: u32, start_s: f64, seconds: f64) -> Option<f64> {
    let out = crate::proc::command(ffmpeg)
        // No -v error here: volumedetect reports its measurement at info level, and quieting
        // ffmpeg hides the very number being asked for.
        .args(["-nostats", "-hide_banner", "-ss"])
        .arg(format!("{start_s}"))
        .arg("-t")
        .arg(format!("{seconds}"))
        .arg("-i")
        .arg(video)
        .args(["-map", &format!("0:a:{stream}"), "-filter:a", "volumedetect", "-f", "null", "-"])
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;
    // volumedetect reports on stderr: "[Parsed_volumedetect_0 @ …] mean_volume: -23.0 dB"
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text.lines().find(|l| l.contains("mean_volume:"))?;
    line.split("mean_volume:").nth(1)?.trim().trim_end_matches(" dB").trim().parse().ok()
}

/// The audio track to use: the loudest one that is not silence. `None` when the video has no
/// usable audio at all.
///
/// Sampled from the middle of the video, where someone is most likely to be talking; a track that
/// happens to be quiet at the top of the file is not thereby the wrong one.
pub async fn pick_track(ffmpeg: &Path, video: &Path, tracks: u32, duration_s: f64) -> Option<u32> {
    if tracks == 0 {
        return None;
    }
    let seconds = 20.0_f64.min(duration_s.max(1.0));
    let start = ((duration_s - seconds) / 2.0).max(0.0);
    let mut best: Option<(u32, f64)> = None;
    for stream in 0..tracks {
        let Some(level) = mean_level(ffmpeg, video, stream, start, seconds).await else { continue };
        if level <= SILENT_DBFS {
            continue;
        }
        if best.is_none_or(|(_, b)| level > b) {
            best = Some((stream, level));
        }
    }
    best.map(|(s, _)| s)
}

/// Mean level of each transcript segment, in the order given.
pub async fn segment_levels(ffmpeg: &Path, video: &Path, stream: u32, segments: &[(f64, f64)]) -> Vec<Option<f64>> {
    if segments.is_empty() {
        return Vec::new();
    }
    let end = segments.iter().map(|(_, end)| *end).filter(|v| v.is_finite()).fold(0.0_f64, f64::max);
    if end <= 0.0 {
        return vec![None; segments.len()];
    }
    let Some(ffprobe) = crate::doctor::locate("ffprobe") else { return vec![None; segments.len()] };
    let Some(channels) = track_channels(&ffprobe, video, stream).await else { return vec![None; segments.len()] };
    // Decode the track once, rather than seeking the source and starting ffmpeg for every line.
    let mut child = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(video)
        .args(["-map", &format!("0:a:{stream}"), "-t", &end.to_string(), "-vn", "-ar", "16000", "-f", "f32le", "-"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn();
    let Ok(ref mut child) = child else { return vec![None; segments.len()] };
    let Some(stdout) = child.stdout.take() else { return vec![None; segments.len()] };
    let levels = measure_pcm(stdout, segments, 16000 * channels as u64).await;
    if !child.wait().await.is_ok_and(|s| s.success()) {
        return vec![None; segments.len()];
    }
    levels
}

/// Measure the exact mono PCM already decoded for local transcription.
pub async fn segment_levels_pcm(path: &Path, segments: &[(f64, f64)]) -> Vec<Option<f64>> {
    match tokio::fs::File::open(path).await {
        Ok(file) => measure_pcm(file, segments, 16000).await,
        Err(_) => vec![None; segments.len()],
    }
}

/// Keep cumulative energy only at segment boundaries: bounded memory even for hours of audio,
/// with overlapping/out-of-order segments supported. Interleaved channels contribute their
/// individual energy rather than cancelling in a downmix.
async fn measure_pcm(
    reader: impl tokio::io::AsyncRead + Unpin,
    segments: &[(f64, f64)],
    samples_per_second: u64,
) -> Vec<Option<f64>> {
    use tokio::io::AsyncReadExt;
    let mut bounds = vec![0_u64];
    let spans: Vec<_> = segments
        .iter()
        .map(|&(start, end)| {
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end - start < 0.4 {
                return None;
            }
            let span =
                ((start * samples_per_second as f64).round() as u64, (end * samples_per_second as f64).round() as u64);
            bounds.extend([span.0, span.1]);
            Some(span)
        })
        .collect();
    bounds.sort_unstable();
    bounds.dedup();
    let mut energy = vec![0.0; bounds.len()];
    let mut count = vec![0_u64; bounds.len()];
    let mut reader = tokio::io::BufReader::new(reader);
    let (mut samples, mut sum, mut next) = (0_u64, 0.0_f64, 0_usize);
    let mut buffer = [0_u8; 16384];
    let mut carry = Vec::with_capacity(buffer.len() + 3);
    loop {
        while next < bounds.len() && bounds[next] <= samples {
            energy[next] = sum;
            count[next] = samples;
            next += 1;
        }
        let n = match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => return vec![None; segments.len()],
        };
        carry.extend_from_slice(&buffer[..n]);
        let complete = carry.len() / 4 * 4;
        for bytes in carry[..complete].as_chunks::<4>().0 {
            while next < bounds.len() && bounds[next] <= samples {
                energy[next] = sum;
                count[next] = samples;
                next += 1;
            }
            let value = f32::from_le_bytes(*bytes) as f64;
            if !value.is_finite() {
                return vec![None; segments.len()];
            }
            sum += value.clamp(-1.0, 1.0).powi(2);
            samples += 1;
        }
        carry.drain(..complete);
    }
    for i in next..bounds.len() {
        energy[i] = sum;
        count[i] = samples;
    }
    spans
        .into_iter()
        .map(|span| {
            let (start, end) = span?;
            let a = bounds.binary_search(&start).ok()?;
            let b = bounds.binary_search(&end).ok()?;
            let n = count[b].saturating_sub(count[a]);
            (n > 0).then(|| (10.0 * ((energy[b] - energy[a]).max(0.0) / n as f64).log10()).max(-91.0))
        })
        .collect()
}

/// Whether each segment is someone away from the microphone, judged against the video's own
/// speech. Returns `None` for a segment where nothing could be measured.
///
/// The reference is the median of the measured levels, not the loudest: an interview is mostly
/// the subject, so the middle of the distribution is the person on the lav.
pub fn off_mic_flags(levels: &[Option<f64>], margin_db: f64) -> Vec<Option<bool>> {
    let mut measured: Vec<f64> = levels.iter().flatten().copied().collect();
    if measured.len() < 3 {
        return levels.iter().map(|_| None).collect();
    }
    measured.sort_by(f64::total_cmp);
    let reference = measured[measured.len() / 2];
    levels.iter().map(|l| l.map(|v| v < reference - margin_db)).collect()
}

/// Channel count for one track. Keep channels separate when measuring their energy: a mono
/// downmix could cancel opposite-phase stereo and wrongly label speech as off-mic.
pub async fn track_channels(ffprobe: &Path, video: &Path, stream: u32) -> Option<u32> {
    let out = crate::proc::command(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("a:{stream}"),
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
        ])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse::<u32>().ok().filter(|n| *n > 0)
}

/// How many audio tracks a video has.
pub async fn track_count(ffprobe: &Path, video: &Path) -> Result<u32, Error> {
    let out = crate::proc::command(ffprobe)
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"])
        .arg(video)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Vision(format!("ffprobe: {e}")))?;
    Ok(String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.trim().is_empty()).count() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pcm_levels_support_overlaps_invalid_spans_and_partial_reads() {
        let mut pcm = Vec::new();
        for amplitude in [0.5_f32, 0.1] {
            for _ in 0..16000 {
                pcm.extend_from_slice(&amplitude.to_le_bytes());
            }
        }
        let spans = [(1.0, 2.0), (0.0, 1.0), (0.5, 1.5), (0.0, 0.2), (-1.0, 1.0), (f64::NAN, 1.0), (3.0, 4.0)];
        let levels = measure_pcm(pcm.as_slice(), &spans, 16000).await;
        assert!((levels[0].unwrap() + 20.0).abs() < 0.001);
        assert!((levels[1].unwrap() + 6.0206).abs() < 0.001);
        assert!((levels[2].unwrap() - 10.0 * 0.13_f64.log10()).abs() < 0.001);
        assert!(levels[3..].iter().all(Option::is_none));
        // A reader can split a sample across reads; no bytes may be discarded.
        let (mut writer, reader) = tokio::io::duplex(7);
        let task = tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            writer.write_all(&pcm).await.unwrap();
        });
        let levels = measure_pcm(reader, &[(0.0, 2.0)], 16000).await;
        task.await.unwrap();
        assert!((levels[0].unwrap() - 10.0 * 0.13_f64.log10()).abs() < 0.001);
    }

    #[tokio::test]
    async fn one_pass_levels_preserve_stereo_energy() {
        if crate::doctor::locate("ffmpeg").is_none() || crate::doctor::locate("ffprobe").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("stereo.wav");
        let status = crate::proc::command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "aevalsrc=0.2*sin(2*PI*440*t)|-0.2*sin(2*PI*440*t):s=16000:d=2"])
            .arg(&video)
            .status()
            .await
            .unwrap();
        assert!(status.success());
        let spans = [(0.1, 0.8), (0.7, 1.9)];
        let levels = segment_levels(Path::new("ffmpeg"), &video, 0, &spans).await;
        for (level, (start, end)) in levels.iter().zip(spans) {
            let old = mean_level(Path::new("ffmpeg"), &video, 0, start, end - start).await.unwrap();
            assert!((level.unwrap() - old).abs() < 0.15, "{level:?} vs {old}");
        }
    }

    /// The interviewer's levels from this project's footage, against the subject's.
    #[test]
    fn the_quiet_speaker_is_the_one_off_mic() {
        let levels = vec![
            Some(-33.6), // interviewer
            Some(-36.1), // interviewer
            Some(-20.1), // subject
            Some(-24.3),
            Some(-24.2),
            Some(-24.2),
            Some(-36.2), // interviewer
            Some(-20.1),
            Some(-23.6),
        ];
        let flags = off_mic_flags(&levels, OFF_MIC_MARGIN_DB);
        assert_eq!(
            flags,
            vec![
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false)
            ]
        );
    }

    /// One person talking throughout: nobody is off-mic, however their level varies.
    #[test]
    fn a_single_speaker_has_nobody_off_mic() {
        let levels = vec![Some(-22.0), Some(-24.0), Some(-21.0), Some(-25.0), Some(-23.0)];
        assert!(off_mic_flags(&levels, OFF_MIC_MARGIN_DB).iter().all(|f| f == &Some(false)));
    }

    /// Too little measured to have an opinion: say so rather than guess.
    #[test]
    fn too_few_measurements_decide_nothing() {
        assert_eq!(off_mic_flags(&[Some(-20.0), None], OFF_MIC_MARGIN_DB), vec![None, None]);
    }
}
