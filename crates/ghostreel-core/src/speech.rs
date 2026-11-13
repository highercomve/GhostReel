//! Where somebody is talking, found from the audio itself.
//!
//! Whisper's segment times are approximate: often late at a start, early or quantised at an end.
//! A cut placed on them can land on the attack of the next speaker's first word (a clip that ends
//! at 12.86 s when the next voice enters at 12.86 s). This stage measures the sound instead.
//!
//! The video's chosen audio track ([`Db::audio_track`]) is decoded at 16 kHz mono through a pipe,
//! never held whole, and reduced to the energy of every 10 ms frame in dBFS. Voiced frames are
//! the ones above an adaptive threshold: a noise floor and a speech level, each taken from the
//! frames around (a low and a high percentile over a rolling window), so a quiet lav recording
//! and a loud camera mic are each judged against their own room. A speaker 12 dB off the
//! microphone (see [`crate::audio`]) is still found, because the threshold sits well under the
//! video's own speech level rather than at a fixed one.
//!
//! What is stored is compact: one row per region of speech (`speech_regions`), plus the quietest
//! moment of the pause that follows it, which is where a cut goes when the pause is too short to
//! leave room for a tail and a guard (see [`crate::cuts`]).

use std::path::Path;

use rusqlite::params;
use serde::Serialize;
use tokio::io::AsyncReadExt;

use crate::Error;
use crate::db::Db;

/// Sample rate the audio is decoded at.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples in one energy frame (10 ms).
pub const FRAME_SAMPLES: usize = 160;
/// Seconds in one energy frame.
pub const FRAME_S: f64 = FRAME_SAMPLES as f64 / SAMPLE_RATE as f64;
/// Digital silence reads as this, not as minus infinity.
const FLOOR_DBFS: f32 = -100.0;

/// How speech is told from the room, in dB and seconds. [`Default`] is what the stage uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadParams {
    /// Length of the rolling window the noise floor and speech level are taken over (s).
    pub window_s: f64,
    /// Percentile of the window that is the noise floor (the quiet end).
    pub floor_percentile: f64,
    /// Percentile of the window that is the speech level (the loud end).
    pub speech_percentile: f64,
    /// A frame is voiced above the floor by at least this much (dB)…
    pub floor_margin_db: f32,
    /// …and above the speech level less this much (dB): a speaker 12 dB off the mic still passes,
    /// a breath or a chair does not.
    pub speech_margin_db: f32,
    /// A voiced stretch goes on while frames stay above the threshold less this (dB), so a word
    /// is kept from its soft attack to its fading end instead of its loudest middle.
    pub hysteresis_db: f32,
    /// Never voiced below this level, whatever the room (dBFS).
    pub min_level_db: f32,
    /// A voiced stretch shorter than this is a click, not speech (s).
    pub min_voiced_s: f64,
    /// A gap shorter than this is inside a word or between two, not a pause (s).
    pub min_gap_s: f64,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            window_s: 30.0,
            floor_percentile: 0.10,
            speech_percentile: 0.95,
            floor_margin_db: 11.0,
            speech_margin_db: 30.0,
            hysteresis_db: 6.0,
            min_level_db: -60.0,
            min_voiced_s: 0.06,
            min_gap_s: 0.08,
        }
    }
}

/// One stretch of speech.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SpeechRegion {
    pub start_s: f64,
    pub end_s: f64,
    /// The quietest 10 ms frame between this region and the next, when there is a next one.
    pub quiet_after_s: Option<f64>,
}

impl SpeechRegion {
    pub fn new(start_s: f64, end_s: f64) -> Self {
        Self { start_s, end_s, quiet_after_s: None }
    }
}

/// Turns PCM into 10 ms frame levels as it streams past, keeping at most one frame of samples.
#[derive(Debug, Default)]
pub struct FrameMeter {
    sum: f64,
    n: usize,
    carry: Vec<u8>,
    levels: Vec<f32>,
}

impl FrameMeter {
    pub fn push_samples(&mut self, samples: impl IntoIterator<Item = f32>) {
        for x in samples {
            // A broken decode must not poison the levels: NaN and infinity count as silence.
            let x = if x.is_finite() { x.clamp(-1.0, 1.0) as f64 } else { 0.0 };
            self.sum += x * x;
            self.n += 1;
            if self.n == FRAME_SAMPLES {
                self.levels.push(dbfs(self.sum / FRAME_SAMPLES as f64));
                self.sum = 0.0;
                self.n = 0;
            }
        }
    }

    /// Little-endian `f32` bytes, as ffmpeg writes them; a sample split across two reads is kept.
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.carry.extend_from_slice(bytes);
        let whole = self.carry.len() / 4 * 4;
        let carry = std::mem::take(&mut self.carry);
        self.push_samples(carry[..whole].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        self.carry = carry[whole..].to_vec();
    }

    /// Frames measured so far.
    pub fn frames(&self) -> usize {
        self.levels.len()
    }

    pub fn finish(self) -> Vec<f32> {
        self.levels
    }
}

fn dbfs(mean_square: f64) -> f32 {
    if mean_square <= 0.0 { FLOOR_DBFS } else { ((10.0 * mean_square.log10()) as f32).max(FLOOR_DBFS) }
}

/// The level at `p` (0..1) of `values`, by nearest rank. `scratch` is reused between calls.
fn percentile(values: &[f32], p: f64, scratch: &mut Vec<f32>) -> f32 {
    scratch.clear();
    scratch.extend_from_slice(values);
    if scratch.is_empty() {
        return FLOOR_DBFS;
    }
    let rank = ((scratch.len() as f64 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    *scratch.select_nth_unstable_by(rank, f32::total_cmp).1
}

/// Regions of speech in a track's frame levels (10 ms each, dBFS).
pub fn detect(levels: &[f32], vad: &VadParams) -> Vec<SpeechRegion> {
    let n = levels.len();
    if n == 0 {
        return Vec::new();
    }
    let thresholds = thresholds(levels, vad);
    let hi: Vec<bool> = (0..n).map(|i| levels[i] > thresholds[i]).collect();
    let lo: Vec<bool> = (0..n).map(|i| levels[i] > thresholds[i] - vad.hysteresis_db).collect();

    // Hysteresis: a stretch above the lower line is speech when it rises above the upper one.
    let mut runs: Vec<(usize, usize)> = Vec::new(); // [start, end) in frames
    let mut i = 0;
    while i < n {
        if !lo[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut loud = false;
        while i < n && lo[i] {
            loud |= hi[i];
            i += 1;
        }
        if loud {
            runs.push((start, i));
        }
    }

    // Pauses shorter than `min_gap_s` are not pauses; then what is left too short is not speech.
    let frames = |s: f64| (s / FRAME_S).round() as usize;
    let (min_gap, min_voiced) = (frames(vad.min_gap_s), frames(vad.min_voiced_s));
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for run in runs {
        match merged.last_mut() {
            Some(last) if run.0 - last.1 < min_gap => last.1 = run.1,
            _ => merged.push(run),
        }
    }
    merged.retain(|&(a, b)| b - a >= min_voiced);

    let mut regions: Vec<SpeechRegion> =
        merged.iter().map(|&(a, b)| SpeechRegion::new(a as f64 * FRAME_S, b as f64 * FRAME_S)).collect();
    for (k, pair) in merged.windows(2).enumerate() {
        regions[k].quiet_after_s = quietest_frame(levels, pair[0].1, pair[1].0).map(|f| f as f64 * FRAME_S);
    }
    regions
}

/// The quietest frame in `[from, to)`; among equals, the one nearest the middle (a flat pause has
/// no better place than its centre).
fn quietest_frame(levels: &[f32], from: usize, to: usize) -> Option<usize> {
    let to = to.min(levels.len());
    if from >= to {
        return None;
    }
    let mid = (from + to - 1) as f64 / 2.0;
    (from..to).min_by(|&a, &b| {
        levels[a].total_cmp(&levels[b]).then((a as f64 - mid).abs().total_cmp(&(b as f64 - mid).abs()))
    })
}

/// The voiced threshold of every frame: from the floor and speech level of the window around it,
/// refreshed once a second (what changes that quickly is speech, not the room).
fn thresholds(levels: &[f32], vad: &VadParams) -> Vec<f32> {
    let n = levels.len();
    let step = (1.0 / FRAME_S).round() as usize;
    let half = ((vad.window_s / FRAME_S / 2.0).round() as usize).max(step);
    let mut out = vec![0.0f32; n];
    let mut scratch = Vec::with_capacity(2 * half + 1);
    let mut block = 0;
    while block < n {
        let end = (block + step).min(n);
        let centre = (block + end) / 2;
        let window = &levels[centre.saturating_sub(half)..(centre + half).min(n)];
        let floor = percentile(window, vad.floor_percentile, &mut scratch);
        let speech = percentile(window, vad.speech_percentile, &mut scratch);
        let threshold = (floor + vad.floor_margin_db).max(speech - vad.speech_margin_db).max(vad.min_level_db);
        out[block..end].fill(threshold);
        block = end;
    }
    out
}

/// Regions of speech in raw 16 kHz mono samples (the same path the stage takes, for callers that
/// already hold the audio).
pub fn detect_samples(samples: &[f32], vad: &VadParams) -> Vec<SpeechRegion> {
    let mut meter = FrameMeter::default();
    meter.push_samples(samples.iter().copied());
    detect(&meter.finish(), vad)
}

/// Decode `audio_track` of `video` (0-based among its audio streams) and measure it, streaming:
/// only the 10 ms levels are kept, so a 70-minute interview costs a couple of megabytes.
/// `on_progress` gets the seconds decoded so far. Dropping the future kills ffmpeg.
pub async fn measure(
    ffmpeg: &Path,
    video: &Path,
    audio_track: u32,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<f32>, Error> {
    let mut child = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(video)
        .args(["-map", &format!("0:a:{audio_track}"), "-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string()])
        .args(["-f", "f32le", "-"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Io(ffmpeg.to_path_buf(), e))?;
    let mut stdout = child.stdout.take().ok_or_else(|| Error::Invalid("ffmpeg stdout unavailable".into()))?;
    let mut stderr = child.stderr.take();
    let errors = tokio::spawn(async move {
        let mut text = String::new();
        if let Some(e) = stderr.as_mut() {
            let _ = e.read_to_string(&mut text).await;
        }
        text
    });

    let mut meter = FrameMeter::default();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let n = stdout.read(&mut buffer).await.map_err(|e| Error::Io(video.to_path_buf(), e))?;
        if n == 0 {
            break;
        }
        meter.push_bytes(&buffer[..n]);
        on_progress(meter.frames() as f64 * FRAME_S);
    }
    let status = child.wait().await.map_err(|e| Error::Io(ffmpeg.to_path_buf(), e))?;
    let stderr = errors.await.unwrap_or_default();
    if !status.success() {
        let why = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("ffmpeg failed").trim().to_string();
        return Err(Error::Invalid(format!("could not decode audio track {audio_track}: {why}")));
    }
    Ok(meter.finish())
}

impl Db {
    /// Replace a video's speech regions.
    pub fn set_speech_regions(&self, video_id: i64, regions: &[SpeechRegion]) -> Result<(), Error> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM speech_regions WHERE video_id = ?1", [video_id])?;
        // Sentence boundaries sit on the pauses; with new pauses they are placed again.
        crate::sentence_split::invalidate(&tx, video_id)?;
        {
            let mut st = tx.prepare(
                "INSERT INTO speech_regions(video_id, start_s, end_s, quiet_after_s) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for r in regions {
                st.execute(params![video_id, r.start_s, r.end_s, r.quiet_after_s])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

/// The speech regions of a video that touch `from_s..to_s`, in order. Empty when the stage has not
/// run for the video, and when it found no speech: [`analysed`] tells the two apart.
pub fn speech_regions(db: &Db, video_id: i64, from_s: f64, to_s: f64) -> Result<Vec<SpeechRegion>, Error> {
    let mut st = db.conn.prepare_cached(
        "SELECT start_s, end_s, quiet_after_s FROM speech_regions
          WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 ORDER BY start_s",
    )?;
    let rows = st.query_map(params![video_id, from_s, to_s], |r| {
        Ok(SpeechRegion { start_s: r.get(0)?, end_s: r.get(1)?, quiet_after_s: r.get(2)? })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Whether the stage has measured this video (it may have found no speech).
pub fn analysed(db: &Db, video_id: i64) -> bool {
    db.conn
        .query_row(
            "SELECT EXISTS (SELECT 1 FROM speech_regions WHERE video_id = ?1)
                 OR EXISTS (SELECT 1 FROM jobs WHERE video_id = ?1 AND stage = 'speech' AND state = 'done')",
            [video_id],
            |r| r.get(0),
        )
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Deterministic noise in `[-1, 1]` (a small LCG: no dependency, same every run).
    struct Noise(u64);
    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        }
    }

    /// A track of `seconds` at `room` dBFS (RMS) with tone bursts `(start, end, dBFS)` laid over it.
    pub fn track(seconds: f64, room_db: f32, bursts: &[(f64, f64, f32)]) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f64) as usize;
        let mut noise = Noise(7);
        // Uniform noise in [-1, 1] has an RMS of 1/sqrt(3).
        let room = 10f32.powf(room_db / 20.0) * 3f32.sqrt();
        let mut out: Vec<f32> = (0..n).map(|_| noise.next() * room).collect();
        for &(a, b, db) in bursts {
            // A sine of amplitude A has an RMS of A/sqrt(2).
            let amp = 10f32.powf(db / 20.0) * 2f32.sqrt();
            for (i, s) in out
                .iter_mut()
                .enumerate()
                .take((b * SAMPLE_RATE as f64) as usize)
                .skip((a * SAMPLE_RATE as f64) as usize)
            {
                *s += amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin();
            }
        }
        out
    }

    /// Write `pcm` (mono, 16 kHz) as a 16-bit WAV file.
    pub fn write_wav(path: &std::path::Path, pcm: &[f32]) {
        let mut bytes = Vec::new();
        let data_len = (pcm.len() * 2) as u32;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for s in pcm {
            bytes.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{track, write_wav};
    use super::*;

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() <= 0.03
    }

    #[test]
    fn finds_bursts_in_silence_and_the_off_mic_speaker_too() {
        let pcm = track(
            40.0,
            -70.0,
            &[
                (2.0, 4.0, -12.0),
                (5.0, 5.5, -12.0),
                // The same breath of silence inside a word: 40 ms is not a pause.
                (7.0, 7.5, -12.0),
                (7.54, 8.0, -12.0),
                // A click, 30 ms: not speech.
                (10.0, 10.03, -12.0),
                // Somebody 12 dB further from the microphone.
                (12.0, 13.5, -24.0),
                // 100 ms of quiet is a pause: two regions.
                (15.0, 16.0, -12.0),
                (16.1, 17.0, -12.0),
            ],
        );
        let regions = detect_samples(&pcm, &VadParams::default());
        let spans: Vec<(f64, f64)> = regions.iter().map(|r| (r.start_s, r.end_s)).collect();
        let want = [(2.0, 4.0), (5.0, 5.5), (7.0, 8.0), (12.0, 13.5), (15.0, 16.0), (16.1, 17.0)];
        assert_eq!(spans.len(), want.len(), "{spans:?}");
        for ((s, e), (ws, we)) in spans.iter().zip(want) {
            assert!(near(*s, ws) && near(*e, we), "{spans:?} vs {want:?}");
        }
    }

    #[test]
    fn the_quietest_frame_of_each_pause_is_kept() {
        let mut pcm = track(10.0, -70.0, &[(1.0, 2.0, -12.0), (3.0, 4.0, -12.0)]);
        // A dip in the pause: -90 dB at 2.6 s, everything else in it at the room's -70.
        for s in pcm.iter_mut().skip((2.6 * 16000.0) as usize).take(160) {
            *s *= 0.1;
        }
        let regions = detect_samples(&pcm, &VadParams::default());
        assert_eq!(regions.len(), 2);
        let q = regions[0].quiet_after_s.expect("a pause follows the first region");
        assert!((q - 2.6).abs() < 0.02, "{q}");
        assert!(regions[1].quiet_after_s.is_none(), "nothing follows the last one");
        assert!(q > regions[0].end_s && q < regions[1].start_s);
    }

    #[test]
    fn a_room_with_no_speech_has_no_regions() {
        // Steady noise, and a track that is digital silence: nothing stands out of either.
        assert!(detect_samples(&track(20.0, -50.0, &[]), &VadParams::default()).is_empty());
        assert!(detect_samples(&vec![0.0; 16000 * 20], &VadParams::default()).is_empty());
        assert!(detect_samples(&[], &VadParams::default()).is_empty());
    }

    #[test]
    fn the_threshold_follows_the_room() {
        // The same speaker in a loud room: speech at -20, room at -45. A fixed -40 line would
        // call the whole room speech; the adaptive one still finds the two bursts.
        let pcm = track(30.0, -45.0, &[(5.0, 8.0, -20.0), (15.0, 18.0, -20.0)]);
        let regions = detect_samples(&pcm, &VadParams::default());
        assert_eq!(regions.len(), 2, "{regions:?}");
        assert!(near(regions[0].start_s, 5.0) && near(regions[1].end_s, 18.0));
    }

    #[test]
    fn bytes_split_anywhere_measure_like_samples() {
        let pcm = track(3.0, -70.0, &[(1.0, 2.0, -12.0)]);
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut whole = FrameMeter::default();
        whole.push_samples(pcm.iter().copied());
        let mut split = FrameMeter::default();
        for chunk in bytes.chunks(7) {
            split.push_bytes(chunk);
        }
        assert_eq!(whole.finish(), split.finish());
    }

    #[test]
    fn regions_round_trip_through_the_database() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size) VALUES (1, 'h', 1)", []).unwrap();
        assert!(!analysed(&db, 1));
        let regions =
            vec![SpeechRegion { start_s: 1.0, end_s: 2.0, quiet_after_s: Some(2.4) }, SpeechRegion::new(3.0, 4.0)];
        db.set_speech_regions(1, &regions).unwrap();
        assert!(analysed(&db, 1));
        assert_eq!(speech_regions(&db, 1, 0.0, 100.0).unwrap(), regions);
        // Only what touches the window.
        assert_eq!(speech_regions(&db, 1, 2.5, 3.5).unwrap(), vec![regions[1]]);
        assert_eq!(speech_regions(&db, 1, 2.0, 3.0).unwrap(), Vec::new(), "touching at an edge is not inside");
        // Replacing, not adding.
        db.set_speech_regions(1, &regions[..1]).unwrap();
        assert_eq!(speech_regions(&db, 1, 0.0, 100.0).unwrap().len(), 1);
        // The video going takes its regions with it.
        db.conn.execute("DELETE FROM videos WHERE id = 1", []).unwrap();
        assert!(speech_regions(&db, 1, 0.0, 100.0).unwrap().is_empty());
    }

    /// The real decode path: ffmpeg writes a track, `measure` streams it back.
    #[tokio::test]
    async fn decodes_the_chosen_audio_track_through_ffmpeg() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else { return };
        let tmp = tempfile::tempdir().unwrap();
        let wav = tmp.path().join("speech.wav");
        let pcm = track(6.0, -70.0, &[(1.0, 2.5, -12.0), (4.0, 5.0, -24.0)]);
        write_wav(&wav, &pcm);

        let mut last = 0.0;
        let levels = measure(&ffmpeg, &wav, 0, |s| last = s).await.unwrap();
        assert!((levels.len() as f64 * FRAME_S - 6.0).abs() < 0.05, "{} frames", levels.len());
        assert!((last - 6.0).abs() < 0.05);
        let regions = detect(&levels, &VadParams::default());
        assert_eq!(regions.len(), 2, "{regions:?}");
        assert!(near(regions[0].start_s, 1.0) && near(regions[1].end_s, 5.0));
        // A track that is not there is an error, not an empty video.
        assert!(measure(&ffmpeg, &wav, 3, |_| {}).await.is_err());
    }
}
