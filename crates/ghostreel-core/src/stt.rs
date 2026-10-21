//! Speech-to-text: GhostPen's transcription server or the local `ghostreel-asr` helper
//! (plan D2, D12, §2b). Both produce the same timestamped segments.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Transcript {
    pub segments: Vec<Segment>,
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Engine {
    /// GhostPen `/v1/audio/transcriptions` with `verbose_json`.
    Server { url: String, chunk_secs: f64 },
    /// `ghostreel-asr` helper + whisper model file.
    Local { asr_bin: PathBuf, model: PathBuf, language: String },
}

impl Engine {
    pub fn server(url: &str) -> Self {
        Engine::Server { url: url.trim_end_matches('/').to_string(), chunk_secs: SERVER_CHUNK_SECS }
    }

    /// Short label for logs/UI.
    pub fn label(&self) -> String {
        match self {
            Engine::Server { url, .. } => format!("GhostPen @ {url}"),
            Engine::Local { model, .. } => format!(
                "local whisper ({})",
                model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
        }
    }
}

/// Requests to GhostPen stay short so its shared model isn't locked for minutes (dictation and
/// live captions keep working while GhostReel indexes).
pub const SERVER_CHUNK_SECS: f64 = 300.0;
/// Extra audio after each chunk so words on the boundary are heard whole.
const CHUNK_OVERLAP_SECS: f64 = 2.0;
/// Whisper hallucinates text in silence; drop segments it is confident contain no speech.
const MAX_NO_SPEECH: f64 = 0.8;

/// Transcribe `video` (whose audio is `duration_s` long). `on_progress` gets seconds of audio done.
pub async fn transcribe(
    engine: &Engine,
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    mut on_progress: impl FnMut(f64),
) -> Result<Transcript, Error> {
    let mut t = match engine {
        Engine::Server { url, chunk_secs } => {
            transcribe_server(url, *chunk_secs, ffmpeg, video, duration_s, &mut on_progress).await?
        }
        Engine::Local { asr_bin, model, language } => {
            match transcribe_local(asr_bin, model, language, false, ffmpeg, video, duration_s, &mut on_progress).await {
                Ok(t) => t,
                // Typically the GPU is busy with other models (out of memory): CPU is slower but works.
                Err(Error::Stt(e)) if e.contains("[gpu]") => {
                    tracing::warn!("local whisper failed on GPU, retrying on CPU: {e}");
                    transcribe_local(asr_bin, model, language, true, ffmpeg, video, duration_s, &mut on_progress)
                        .await
                        .map_err(|cpu_err| Error::Stt(format!("{e}; CPU retry: {cpu_err}")))?
                }
                Err(e) => return Err(e),
            }
        }
    };
    t.segments = collapse_repeats(std::mem::take(&mut t.segments));
    Ok(t)
}

/// One transcription stage owns one worker. Dropping the session releases its GPU allocation
/// before frame descriptions start. Server backends retain their existing request protocol.
pub struct Transcriber<'a> {
    engine: &'a Engine,
    worker: Option<LocalWorker>,
    audio: Option<tempfile::NamedTempFile>,
}

impl<'a> Transcriber<'a> {
    pub fn new(engine: &'a Engine) -> Self {
        Self { engine, worker: None, audio: None }
    }

    /// Mono PCM from ffmpeg’s selected audio track, valid until the next request.
    pub fn decoded_audio(&self) -> Option<&Path> {
        self.audio.as_ref().map(|f| f.path())
    }

    /// Wait for process exit before another model claims the GPU. Drop still kills the worker
    /// on cancellation/error paths where an async shutdown cannot run.
    pub async fn shutdown(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            let _ = worker.child.kill().await;
            worker.stderr.abort();
        }
        self.audio = None;
    }

    pub async fn transcribe(
        &mut self,
        ffmpeg: &Path,
        video: &Path,
        duration_s: f64,
        mut progress: impl FnMut(f64),
    ) -> Result<Transcript, Error> {
        self.audio = None;
        let Engine::Local { asr_bin, model, language } = self.engine else {
            return transcribe(self.engine, ffmpeg, video, duration_s, progress).await;
        };
        let audio = tempfile::Builder::new()
            .prefix("ghostreel-audio-")
            .suffix(".f32")
            .tempfile()
            .map_err(|e| Error::Stt(format!("temporary audio: {e}")))?;
        let decode = crate::proc::command(ffmpeg)
            .args(["-nostdin", "-v", "error", "-y", "-i"])
            .arg(video)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le"])
            .arg(audio.path())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| Error::Stt(format!("ffmpeg audio decode: {e}")))?;
        if !decode.status.success() {
            return Err(Error::Stt(format!("ffmpeg audio decode: {}", String::from_utf8_lossy(&decode.stderr))));
        }
        self.audio = Some(audio);
        if self.worker.is_none() {
            self.worker = Some(LocalWorker::start(asr_bin, model, language)?);
        }
        let result = self
            .worker
            .as_mut()
            .unwrap()
            .transcribe(self.audio.as_ref().unwrap().path(), duration_s, &mut progress)
            .await;
        match result {
            Ok(mut t) => {
                t.segments = collapse_repeats(t.segments);
                Ok(t)
            }
            Err(error) => {
                // Kill a failed worker before a retry, releasing any GPU memory it owns.
                self.shutdown().await;
                tracing::warn!("persistent whisper failed, retrying with isolated helper: {error}");
                transcribe(self.engine, ffmpeg, video, duration_s, progress).await
            }
        }
    }
}

struct LocalWorker {
    child: tokio::process::Child,
    input: tokio::process::ChildStdin,
    lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    stderr: tokio::task::JoinHandle<()>,
}

impl Drop for LocalWorker {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        self.stderr.abort();
    }
}

impl LocalWorker {
    fn start(bin: &Path, model: &Path, language: &str) -> Result<Self, Error> {
        let mut child = crate::proc::command(bin)
            .arg("--model")
            .arg(model)
            .args(["--language", language, "--serve"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::Stt(format!("cannot run {}: {e}", bin.display())))?;
        let input = child.stdin.take().ok_or_else(|| Error::Stt("asr stdin unavailable".into()))?;
        let output = child.stdout.take().ok_or_else(|| Error::Stt("asr stdout unavailable".into()))?;
        let errors = child.stderr.take();
        // Drain stderr continuously: ggml can otherwise fill its pipe and stall inference.
        let stderr = tokio::spawn(async move {
            if let Some(errors) = errors {
                let mut lines = BufReader::new(errors).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!("whisper: {line}");
                }
            }
        });
        Ok(Self { child, input, lines: BufReader::new(output).lines(), stderr })
    }

    async fn transcribe(
        &mut self,
        audio: &Path,
        duration_s: f64,
        progress: &mut impl FnMut(f64),
    ) -> Result<Transcript, Error> {
        let request =
            serde_json::to_vec(&serde_json::json!({ "audio": audio })).map_err(|e| Error::Stt(e.to_string()))?;
        self.input.write_all(&request).await.map_err(|e| Error::Stt(e.to_string()))?;
        self.input.write_all(b"\n").await.map_err(|e| Error::Stt(e.to_string()))?;
        self.input.flush().await.map_err(|e| Error::Stt(e.to_string()))?;
        let mut out = Transcript::default();
        while let Some(line) = self.lines.next_line().await.map_err(|e| Error::Stt(e.to_string()))? {
            match serde_json::from_str::<AsrLine>(&line) {
                Ok(AsrLine::Progress { percent }) => progress(duration_s * percent.clamp(0.0, 100.0) / 100.0),
                Ok(AsrLine::Segment { start, end, text, no_speech }) => {
                    if no_speech.unwrap_or(0.0) <= MAX_NO_SPEECH && !text.trim().is_empty() {
                        out.segments.push(Segment { start, end, text: text.trim().into() });
                    }
                }
                Ok(AsrLine::Done { language }) => {
                    out.language = language;
                    progress(duration_s);
                    return Ok(out);
                }
                Ok(AsrLine::Error { error }) => return Err(Error::Stt(error)),
                _ => {}
            }
        }
        Err(Error::Stt("Whisper worker exited before completing the clip".into()))
    }
}

fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whisper tends to repeat the previous sentence in long silences ("Chapter 5. The green…" twice,
/// the first copy cut short). When a segment repeats or extends the one before it within a minute,
/// keep one segment: the earlier start time (when the words were actually spoken) with the more
/// complete text.
pub fn collapse_repeats(segments: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for seg in segments {
        if is_non_speech(&seg.text) {
            continue;
        }
        if let Some(prev) = out.last_mut() {
            let (a, b) = (normalize(&prev.text), normalize(&seg.text));
            let close = seg.start - prev.start <= 60.0;
            let shorter = a.len().min(b.len());
            // Same opening for most of the shorter sentence: a re-hearing, possibly with a
            // different (mis-heard) ending.
            let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            if close && shorter >= 12 && shared * 10 >= shorter * 7 {
                if b.len() > a.len() {
                    // Rough speaking rate (~15 chars/s) for the extra words; never past the repeat's end.
                    let extra = (b.len() - a.len()) as f64 / 15.0;
                    prev.end = (prev.end + extra).min(seg.end.max(prev.end));
                    prev.text = seg.text;
                }
                continue;
            }
        }
        out.push(seg);
    }
    out
}

/// Whisper's non-speech annotations: "[Music]", "(applause)", "♪♪", "[BLANK_AUDIO]".
fn is_non_speech(text: &str) -> bool {
    let t = text.trim();
    let bracketed = (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')'));
    bracketed || t.chars().all(|c| !c.is_alphanumeric())
}

// ---- server (GhostPen) ------------------------------------------------------------------------

async fn transcribe_server(
    url: &str,
    chunk_secs: f64,
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    on_progress: &mut impl FnMut(f64),
) -> Result<Transcript, Error> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        // A 5-minute chunk takes seconds on GPU; allow slow CPU servers generously.
        .timeout(Duration::from_secs(900))
        .build()
        .map_err(|e| Error::Stt(e.to_string()))?;
    let mut out = Transcript::default();
    let silences = if duration_s > chunk_secs { detect_silences(ffmpeg, video).await } else { Vec::new() };
    let bounds = chunk_bounds(duration_s, chunk_secs, &silences);
    for (i, pair) in bounds.windows(2).enumerate() {
        let (start, end) = (pair[0], pair[1]);
        let last = i + 2 == bounds.len();
        // Audio starts a little before the cut and runs a little past it, so a word on the boundary
        // is heard whole by one of the two requests; merge_chunk drops the duplicates.
        let from = (start - CHUNK_OVERLAP_SECS).max(0.0);
        let len = (end - from) + if last { 1.0 } else { CHUNK_OVERLAP_SECS };
        let audio = extract_opus(ffmpeg, video, from, len).await?;
        let body: VerboseJson = post_chunk(&client, url, audio).await?;
        if out.language.is_none() {
            out.language = body.language.filter(|l| !l.is_empty());
        }
        merge_chunk(&mut out.segments, from, start, if last { f64::INFINITY } else { end }, body.segments);
        on_progress(end.min(duration_s));
    }
    Ok(out)
}

/// Chunk boundaries `[0, b1, …, duration]`: roughly every `chunk_secs`, moved to the middle of the
/// nearest silence (within ±20 % of a chunk) so cuts don't split words.
fn chunk_bounds(duration_s: f64, chunk_secs: f64, silences: &[(f64, f64)]) -> Vec<f64> {
    let mut bounds = vec![0.0];
    let window = chunk_secs * 0.2;
    let mut target = chunk_secs;
    while target < duration_s - window.min(chunk_secs * 0.5) {
        let prev = *bounds.last().unwrap();
        let cut = silences
            .iter()
            .map(|&(a, b)| (a + b) / 2.0)
            .filter(|m| (m - target).abs() <= window && *m > prev + chunk_secs * 0.5)
            .min_by(|x, y| (x - target).abs().total_cmp(&(y - target).abs()))
            .unwrap_or(target);
        bounds.push(cut);
        target = cut + chunk_secs;
    }
    bounds.push(duration_s);
    bounds
}

/// `(start, end)` of silences ≥ 0.4 s in the audio track (one fast decode pass). Empty on failure:
/// chunking then falls back to fixed boundaries.
async fn detect_silences(ffmpeg: &Path, video: &Path) -> Vec<(f64, f64)> {
    let out = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-nostats", "-i"])
        .arg(video)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"])
        .kill_on_drop(true)
        .output()
        .await;
    let Ok(out) = out else { return Vec::new() };
    parse_silences(&String::from_utf8_lossy(&out.stderr))
}

fn parse_silences(log: &str) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut start = None;
    for line in log.lines() {
        if let Some(v) = line.split("silence_start: ").nth(1) {
            start = v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok());
        } else if let Some(v) = line.split("silence_end: ").nth(1)
            && let (Some(a), Some(b)) = (start.take(), v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok()))
        {
            out.push((a.max(0.0), b));
        }
    }
    out
}

#[derive(Deserialize)]
struct VerboseJson {
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Option<Vec<Segment>>,
}

async fn post_chunk(client: &reqwest::Client, url: &str, audio: Vec<u8>) -> Result<VerboseJson, Error> {
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(audio)
                .file_name("chunk.ogg")
                .mime_str("audio/ogg")
                .map_err(|e| Error::Stt(e.to_string()))?,
        )
        .text("response_format", "verbose_json");
    let resp = client
        .post(format!("{url}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| Error::Stt(format!("GhostPen at {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Stt(format!("GhostPen at {url}: HTTP {status} {}", text.trim())));
    }
    let body: VerboseJson =
        resp.json().await.map_err(|e| Error::Stt(format!("GhostPen at {url}: bad response: {e}")))?;
    if body.segments.is_none() {
        return Err(Error::Stt(format!("GhostPen at {url} returned no segments (update GhostPen)")));
    }
    Ok(body)
}

/// Append a chunk's segments (times relative to `audio_from`) to `acc`. The chunk owns segments
/// starting in `[chunk_start, chunk_end)`; the pre-roll before `chunk_start` only exists so a
/// boundary word is heard whole, and anything the previous chunk already covered is dropped.
fn merge_chunk(
    acc: &mut Vec<Segment>,
    audio_from: f64,
    chunk_start: f64,
    chunk_end: f64,
    segments: Option<Vec<Segment>>,
) {
    let last_end = acc.last().map(|s| s.end).unwrap_or(f64::NEG_INFINITY);
    for s in segments.unwrap_or_default() {
        let seg = Segment { start: s.start + audio_from, end: s.end + audio_from, text: s.text.trim().to_string() };
        if seg.text.is_empty() || seg.start >= chunk_end || seg.start < last_end - 0.3 {
            continue;
        }
        // In the pre-roll and not already covered: a boundary word the previous chunk missed.
        if seg.start < chunk_start - 0.3 && seg.end <= last_end {
            continue;
        }
        acc.push(seg);
    }
}

/// Cut `[start, start+len]` of the audio as mono 16 kHz Opus in Ogg (small uploads).
async fn extract_opus(ffmpeg: &Path, video: &Path, start: f64, len: f64) -> Result<Vec<u8>, Error> {
    let out = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-ss", &format!("{start:.3}"), "-t", &format!("{len:.3}"), "-i"])
        .arg(video)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k", "-f", "ogg", "-"])
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| Error::Stt(format!("cannot run {}: {e}", ffmpeg.display())))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Stt(format!("ffmpeg audio extract: {}", msg.lines().next().unwrap_or("failed"))));
    }
    Ok(out.stdout)
}

// ---- local (ghostreel-asr) --------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum AsrLine {
    Loaded {
        #[serde(default)]
        gpu: bool,
    },
    Progress {
        percent: f64,
    },
    Segment {
        start: f64,
        end: f64,
        text: String,
        #[serde(default)]
        no_speech: Option<f64>,
    },
    Done {
        language: Option<String>,
    },
    Error {
        error: String,
    },
}

#[allow(clippy::too_many_arguments)]
async fn transcribe_local(
    asr_bin: &Path,
    model: &Path,
    language: &str,
    cpu: bool,
    ffmpeg: &Path,
    video: &Path,
    duration_s: f64,
    on_progress: &mut impl FnMut(f64),
) -> Result<Transcript, Error> {
    let mut decoder = crate::proc::command(ffmpeg)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(video)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Stt(format!("cannot run {}: {e}", ffmpeg.display())))?;
    let pcm: Stdio = decoder
        .stdout
        .take()
        .ok_or_else(|| Error::Stt("ffmpeg stdout unavailable".into()))?
        .try_into()
        .map_err(|e| Error::Stt(format!("pipe ffmpeg → asr: {e}")))?;

    let mut cmd = crate::proc::command(asr_bin);
    cmd.arg("--model").arg(model).args(["--language", language]);
    if cpu {
        cmd.arg("--cpu");
    }
    let mut asr = cmd
        .stdin(pcm)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Stt(format!("cannot run {}: {e}", asr_bin.display())))?;

    let stderr = asr.stderr.take();
    // Keep the lines that explain a failure (our own error, CUDA/ggml errors), not the chatter.
    let err_task = tokio::spawn(async move {
        let mut keep: Vec<String> = Vec::new();
        if let Some(e) = stderr {
            let mut lines = BufReader::new(e).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let lower = l.to_lowercase();
                if l.starts_with("ghostreel-asr:") || lower.contains("error") || lower.contains("out of memory") {
                    keep.push(l.trim().to_string());
                    if keep.len() > 4 {
                        keep.remove(0);
                    }
                }
            }
        }
        keep.join(" | ")
    });

    let mut out = Transcript::default();
    let mut done = false;
    let mut used_gpu = false;
    let stdout = asr.stdout.take().ok_or_else(|| Error::Stt("asr stdout unavailable".into()))?;
    let mut lines = BufReader::new(stdout).lines();
    while let Some(line) = lines.next_line().await.map_err(|e| Error::Stt(e.to_string()))? {
        match serde_json::from_str::<AsrLine>(&line) {
            Ok(AsrLine::Progress { percent }) => on_progress(duration_s * percent.clamp(0.0, 100.0) / 100.0),
            Ok(AsrLine::Segment { start, end, text, no_speech }) => {
                if no_speech.unwrap_or(0.0) <= MAX_NO_SPEECH && !text.trim().is_empty() {
                    out.segments.push(Segment { start, end, text: text.trim().to_string() });
                }
            }
            Ok(AsrLine::Done { language }) => {
                out.language = language;
                done = true;
            }
            Ok(AsrLine::Loaded { gpu }) => used_gpu = gpu,
            Ok(AsrLine::Error { error }) => return Err(Error::Stt(error)),
            Err(_) => {}
        }
    }
    let status = asr.wait().await.map_err(|e| Error::Stt(e.to_string()))?;
    let decode = decoder.wait_with_output().await.map_err(|e| Error::Stt(e.to_string()))?;
    let asr_err = err_task.await.unwrap_or_default();
    if !status.success() || !done {
        let mut reason = if asr_err.is_empty() { format!("local whisper exited with {status}") } else { asr_err };
        // Marks failures worth retrying on CPU (crash after loading onto the GPU, or CUDA errors).
        if !cpu && (used_gpu || reason.contains("CUDA") || reason.contains("out of memory")) {
            reason.push_str(" [gpu]");
        }
        return Err(Error::Stt(reason));
    }
    if !decode.status.success() {
        let msg = String::from_utf8_lossy(&decode.stderr);
        return Err(Error::Stt(format!("ffmpeg audio decode: {}", msg.lines().next().unwrap_or("failed"))));
    }
    on_progress(duration_s);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn persistent_worker_reuses_process_across_requests() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("worker");
        let starts = tmp.path().join("starts");
        std::fs::write(
            &bin,
            format!(
                r#"#!/bin/sh
echo started >> '{}'
echo '{{"type":"loaded","gpu":true}}'
while IFS= read -r request; do
 echo '{{"type":"segment","start":0,"end":1,"text":"hello"}}'
 echo '{{"type":"done","language":"en"}}'
done
"#,
                starts.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut worker = LocalWorker::start(&bin, Path::new("model"), "auto").unwrap();
        let pid = worker.child.id().unwrap();
        for _ in 0..2 {
            let t = worker.transcribe(Path::new("audio.f32"), 1.0, &mut |_| {}).await.unwrap();
            assert_eq!(t.segments, vec![seg(0.0, 1.0, "hello")]);
            assert_eq!(t.language.as_deref(), Some("en"));
            assert_eq!(worker.child.id(), Some(pid));
        }
        assert_eq!(std::fs::read_to_string(starts).unwrap().lines().count(), 1);
        drop(worker);
    }

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment { start, end, text: text.into() }
    }

    #[test]
    fn chunk_seams_are_deduplicated() {
        let mut acc = Vec::new();
        // chunk 0 owns [0, 10); a segment starting at/after 10 belongs to chunk 1.
        merge_chunk(
            &mut acc,
            0.0,
            0.0,
            10.0,
            Some(vec![seg(0.0, 4.0, " one"), seg(4.0, 9.8, "two"), seg(10.0, 12.0, "Chap")]),
        );
        // chunk 1 audio starts 2 s early (at 8): it re-hears "two" and hears "Chapter ten" whole.
        merge_chunk(
            &mut acc,
            8.0,
            10.0,
            f64::INFINITY,
            Some(vec![seg(0.0, 1.8, "two"), seg(2.0, 5.0, "Chapter ten"), seg(5.0, 5.1, "  ")]),
        );
        let texts: Vec<_> = acc.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["one", "two", "Chapter ten"]);
        assert_eq!((acc[2].start, acc[2].end), (10.0, 13.0));
    }

    #[test]
    fn repeated_sentences_in_silence_collapse() {
        let segs = vec![
            seg(
                120.0,
                126.0,
                "Chapter 5. The green light links twice, which means the boot loader found the operating system.",
            ),
            seg(
                143.0,
                150.0,
                "Chapter 5. The green light links twice, which means the boot loader found the operating system, and is starting the containers.",
            ),
            seg(174.0, 180.0, "Chapter 6. We open the provisioning page."),
            seg(181.0, 183.0, "Okay."),
            seg(184.0, 186.0, "Okay."),
            seg(
                240.0,
                246.0,
                "Chapter 9. The update is transactional, so if something fails the device rolls back automatically to the computer.",
            ),
            seg(
                264.0,
                270.0,
                "Chapter 9. The update is transactional, so if something fails the device rolls back automatically to the previous working revision.",
            ),
            seg(346.0, 350.0, "[Music]"),
            seg(351.0, 352.0, "♪ ♪"),
        ];
        let out = collapse_repeats(segs);
        assert_eq!(out.len(), 5, "short fillers like 'Okay.' are not treated as repeats; tags dropped");
        assert!(out[4].text.ends_with("previous working revision."));
        assert_eq!(out[4].start, 240.0);
        assert_eq!(out[0].start, 120.0);
        assert!(out[0].text.ends_with("starting the containers."));
        assert!(out[0].end > 126.0 && out[0].end <= 150.0);
        assert!(out[1].text.starts_with("Chapter 6"));
    }

    #[test]
    fn boundaries_prefer_silence() {
        let silences = [(95.0, 97.0), (290.0, 291.0), (318.0, 320.0), (610.0, 612.0)];
        let b = chunk_bounds(900.0, 300.0, &silences);
        assert_eq!(b, vec![0.0, 290.5, 611.0, 900.0]);
        assert_eq!(chunk_bounds(250.0, 300.0, &[]), vec![0.0, 250.0]);
        // Tail shorter than 20 % of a chunk is merged into the last chunk.
        assert_eq!(chunk_bounds(640.0, 300.0, &[]), vec![0.0, 300.0, 640.0]);
        let log = "[silencedetect @ 0x1] silence_start: 12.5\n[silencedetect @ 0x1] silence_end: 14.25 | silence_duration: 1.75\n";
        assert_eq!(parse_silences(log), vec![(12.5, 14.25)]);
    }

    fn have(bin: &str) -> bool {
        crate::proc::std_command(bin).arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_engine_parses_helper_output() {
        if !have("ffmpeg") {
            eprintln!("skipping: ffmpeg not installed");
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("v.mp4");
        let ok = crate::proc::std_command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=2", "-c:a", "aac"])
            .arg(&video)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let asr = tmp.path().join("asr");
        std::fs::write(
            &asr,
            r#"#!/bin/sh
cat > /dev/null
echo '{"type":"loaded","model":"m","gpu":false}'
echo '{"type":"progress","percent":50}'
echo '{"type":"segment","start":0.0,"end":1.0,"text":" hello ","no_speech":0.01}'
echo '{"type":"segment","start":1.0,"end":2.0,"text":"Thank you.","no_speech":0.97}'
echo '{"type":"done","language":"en","audio_s":2.0}'
"#,
        )
        .unwrap();
        std::fs::set_permissions(&asr, std::fs::Permissions::from_mode(0o755)).unwrap();
        let engine = Engine::Local { asr_bin: asr, model: tmp.path().join("m.bin"), language: "auto".into() };
        let mut progress = Vec::new();
        let t = transcribe(&engine, Path::new("ffmpeg"), &video, 2.0, |p| progress.push(p)).await.unwrap();
        assert_eq!(t.segments, vec![seg(0.0, 1.0, "hello")], "no-speech hallucination dropped");
        assert_eq!(t.language.as_deref(), Some("en"));
        assert_eq!(progress, vec![1.0, 2.0]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_engine_reports_helper_errors() {
        if !have("ffmpeg") {
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("v.mp4");
        crate::proc::std_command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "sine=d=1", "-c:a", "aac"])
            .arg(&video)
            .status()
            .unwrap();
        let asr = tmp.path().join("asr");
        std::fs::write(&asr, "#!/bin/sh\ncat >/dev/null\necho 'ghostreel-asr: loading model m: Failed' >&2\nexit 1\n")
            .unwrap();
        std::fs::set_permissions(&asr, std::fs::Permissions::from_mode(0o755)).unwrap();
        let engine = Engine::Local { asr_bin: asr, model: "m".into(), language: "auto".into() };
        let err = transcribe(&engine, Path::new("ffmpeg"), &video, 1.0, |_| {}).await.unwrap_err();
        assert!(err.to_string().contains("loading model"), "{err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn gpu_crash_retries_on_cpu() {
        if !have("ffmpeg") {
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("v.mp4");
        crate::proc::std_command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "sine=d=1", "-c:a", "aac"])
            .arg(&video)
            .status()
            .unwrap();
        let asr = tmp.path().join("asr");
        std::fs::write(
            &asr,
            r#"#!/bin/sh
cat > /dev/null
case " $* " in
  *" --cpu "*) echo '{"type":"loaded","gpu":false}'
               echo '{"type":"segment","start":0,"end":1,"text":"on cpu","no_speech":0}'
               echo '{"type":"done","language":"en"}' ;;
  *) echo '{"type":"loaded","gpu":true}'
     echo 'ggml-cuda.cu:96: CUDA error: out of memory' >&2
     kill -ABRT $$ ;;
esac
"#,
        )
        .unwrap();
        std::fs::set_permissions(&asr, std::fs::Permissions::from_mode(0o755)).unwrap();
        let engine = Engine::Local { asr_bin: asr, model: "m".into(), language: "auto".into() };
        let t = transcribe(&engine, Path::new("ffmpeg"), &video, 1.0, |_| {}).await.unwrap();
        assert_eq!(t.segments[0].text, "on cpu");
    }

    #[tokio::test]
    async fn server_engine_chunks_and_offsets() {
        if !have("ffmpeg") {
            return;
        }
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("v.mp4");
        crate::proc::std_command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "sine=d=25", "-c:a", "aac"])
            .arg(&video)
            .status()
            .unwrap();

        // Fake GhostPen: every chunk "hears" a segment at 1–3 s and one at 9.5–10.5 s.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let counter = counter.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 8192];
                    // read until the multipart terminator arrives
                    loop {
                        let n = sock.read(&mut tmp).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        if String::from_utf8_lossy(&buf).contains("verbose_json") && buf.ends_with(b"--\r\n") {
                            break;
                        }
                    }
                    let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let body = format!(
                        r#"{{"language":"en","segments":[{{"id":0,"start":1.0,"end":3.0,"text":"chunk {n} a"}},{{"id":1,"start":9.5,"end":10.5,"text":"chunk {n} b"}}]}}"#
                    );
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let engine = Engine::Server { url, chunk_secs: 10.0 };
        // (sine tone has no silences → fixed 10 s boundaries; chunks 2+ start with 2 s pre-roll)
        let mut progress = Vec::new();
        let t = transcribe(&engine, Path::new("ffmpeg"), &video, 25.0, |p| progress.push(p)).await.unwrap();
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 3, "25 s in 10 s chunks");
        let got: Vec<(f64, &str)> = t.segments.iter().map(|s| (s.start, s.text.as_str())).collect();
        // chunk 1 audio starts at 8 s: its "a" at 9 s overlaps chunk 0's last segment (ends 10.5)
        // → dropped. Chunk 2 audio starts at 18 s: its "a" at 19 s is in the pre-roll but nothing
        // earlier covered it (a boundary word) → kept.
        assert_eq!(
            got,
            [(1.0, "chunk 0 a"), (9.5, "chunk 0 b"), (17.5, "chunk 1 b"), (19.0, "chunk 2 a"), (27.5, "chunk 2 b")]
        );
        assert_eq!(progress, vec![10.0, 20.0, 25.0]);
        assert_eq!(t.language.as_deref(), Some("en"));
    }
}
