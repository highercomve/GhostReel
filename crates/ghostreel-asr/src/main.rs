//! `ghostreel-asr` — local speech-to-text helper for GhostReel.
//!
//! Reads 16 kHz mono f32le PCM on stdin and writes JSON lines on stdout:
//!
//! ```text
//! {"type":"loaded","model":"…","gpu":true}
//! {"type":"progress","percent":42}
//! {"type":"segment","start":0.0,"end":6.2,"text":"…","no_speech":0.01,"words":[{"w":"Esto","s":0.1,"e":0.4}]}
//! {"type":"done","language":"en","audio_s":123.4}
//! ```
//!
//! `words` are the segment's words with whisper's own token timestamps (seconds in the audio): a
//! rough clock, good to a word and not to a phoneme, which GhostReel snaps to the measured speech.
//!
//! Errors go to stderr with a non-zero exit code. Runs as its own process because whisper-rs and
//! llama-cpp-2 each vendor ggml (duplicate symbols in one binary), and so the GPU memory is
//! released for certain when transcription ends.

use std::io::{BufRead, Read, Write};
use std::process::ExitCode;

use serde_json::json;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

struct Args {
    model: String,
    language: String,
    threads: i32,
    cpu: bool,
    serve: bool,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut language = "auto".to_string();
    let mut threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).min(8);
    let mut cpu = false;
    let mut serve = false;
    let mut prompt = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next(),
            "--language" => language = it.next().ok_or("--language needs a value")?,
            "--threads" => {
                threads = it.next().and_then(|v| v.parse().ok()).ok_or("--threads needs a number")?;
            }
            "--prompt" => prompt = it.next(),
            // Force CPU inference (GPU out of memory, or no usable GPU).
            "--cpu" => cpu = true,
            "--serve" => serve = true,
            "--version" => {
                println!("ghostreel-asr {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "usage: ghostreel-asr --model <ggml-*.bin> [--language auto|en|es…] [--threads N] [--cpu] [--serve] [--prompt <text>] < pcm_f32le_16k_mono"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args { model: model.ok_or("--model is required")?, language, threads, cpu, serve, prompt })
}

fn emit(v: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // whisper.cpp/ggml logs go to stderr as-is: GhostReel keeps the last lines to explain failures
    // (e.g. "CUDA error: out of memory").
    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.flash_attn(true);
    if args.cpu {
        ctx_params.use_gpu(false);
    }
    let gpu = ctx_params.use_gpu;
    let ctx = WhisperContext::new_with_params(&args.model, ctx_params)
        .map_err(|e| format!("loading model {}: {e}", args.model))?;
    emit(json!({ "type": "loaded", "model": args.model, "gpu": gpu }));

    if args.serve {
        // One JSON request per decoded clip. Retain weights on the GPU, but use fresh decoder
        // state for each request so language/context from one video cannot leak into the next.
        for line in std::io::stdin().lock().lines() {
            let result = (|| {
                let line = line.map_err(|e| format!("reading request: {e}"))?;
                let request: serde_json::Value =
                    serde_json::from_str(&line).map_err(|e| format!("invalid request: {e}"))?;
                let path = request["audio"].as_str().ok_or("request needs an audio path")?;
                let prompt = request["prompt"].as_str().map(|s| s.to_string());
                let raw = std::fs::read(path).map_err(|e| format!("reading decoded audio: {e}"))?;
                transcribe(&ctx, &args, &raw, prompt.as_deref())
            })();
            if let Err(error) = result {
                emit(json!({ "type": "error", "error": error }));
            }
        }
        Ok(())
    } else {
        let mut raw = Vec::new();
        std::io::stdin().read_to_end(&mut raw).map_err(|e| format!("reading audio from stdin: {e}"))?;
        transcribe(&ctx, &args, &raw, args.prompt.as_deref())
    }
}

fn transcribe(ctx: &WhisperContext, args: &Args, raw: &[u8], prompt: Option<&str>) -> Result<(), String> {
    if !raw.len().is_multiple_of(4) {
        return Err("audio is not f32le PCM (length not a multiple of 4)".into());
    }
    let samples: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let audio_s = samples.len() as f64 / 16_000.0;
    if samples.is_empty() {
        emit(json!({ "type": "done", "language": null, "audio_s": 0.0 }));
        return Ok(());
    }

    let mut state = ctx.create_state().map_err(|e| format!("whisper state: {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(args.threads);
    let lang = (!args.language.eq_ignore_ascii_case("auto")).then_some(args.language.as_str());
    params.set_language(lang);
    if let Some(p) = prompt.filter(|p| !p.trim().is_empty()) {
        params.set_initial_prompt(p);
    }
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_nst(true);
    // Per-token times, for cutting at a word inside a sentence.
    params.set_token_timestamps(true);
    let mut last = -1;
    params.set_progress_callback_safe(move |p: i32| {
        if p != last {
            last = p;
            emit(json!({ "type": "progress", "percent": p }));
        }
    });

    state.full(params, &samples).map_err(|e| format!("transcription failed: {e}"))?;

    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else { continue };
        let text = seg.to_str_lossy().map(|t| t.trim().to_string()).unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let eot = ctx.token_eot();
        let tokens: Vec<Token> = (0..seg.n_tokens())
            .filter_map(|k| {
                let token = seg.get_token(k)?;
                let data = token.token_data();
                // The end-of-text and timestamp tokens ([_BEG_], [_TT_n]) are not words.
                if data.id >= eot {
                    return None;
                }
                Some(Token { text: token.to_str_lossy().ok()?.into_owned(), t0: data.t0, t1: data.t1 })
            })
            .collect();
        let words: Vec<serde_json::Value> = group_words(&tokens, seg.start_timestamp(), seg.end_timestamp())
            .into_iter()
            .map(|w| json!({ "w": w.text, "s": w.start, "e": w.end }))
            .collect();
        emit(json!({
            "type": "segment",
            "start": seg.start_timestamp() as f64 / 100.0,
            "end": seg.end_timestamp() as f64 / 100.0,
            "text": text,
            "no_speech": seg.no_speech_probability(),
            "words": words,
        }));
    }
    let language = whisper_rs::get_lang_str(state.full_lang_id_from_state());
    emit(json!({ "type": "done", "language": language, "audio_s": audio_s }));
    Ok(())
}

/// One token of a segment, with whisper's times for it in centiseconds (-1 when it has none).
struct Token {
    text: String,
    t0: i64,
    t1: i64,
}

#[derive(Debug, PartialEq)]
struct Word {
    text: String,
    start: f64,
    end: f64,
}

/// Group a segment's tokens into words. A token that starts with a space starts a word; one that
/// does not (a suffix, a punctuation mark) belongs to the word before it. A word runs from its
/// first token's start to its last token's end, clamped to the segment and kept in order: whisper
/// leaves some tokens without a time, and those take their neighbours'.
fn group_words(tokens: &[Token], seg_start: i64, seg_end: i64) -> Vec<Word> {
    let mut words: Vec<(String, i64, i64)> = Vec::new();
    for t in tokens {
        if t.text.trim().is_empty() {
            continue;
        }
        let (a, b) = (t.t0.max(seg_start), t.t1.max(t.t0).min(seg_end.max(seg_start)));
        let new_word = t.text.starts_with(' ') || words.is_empty();
        match words.last_mut() {
            Some(w) if !new_word => {
                w.0.push_str(&t.text);
                if t.t1 >= 0 {
                    w.2 = w.2.max(b);
                }
            }
            _ => words.push((t.text.trim_start().to_string(), a, b)),
        }
    }
    // A word without its own time starts where the one before ended, and never ends before it starts.
    let mut cursor = seg_start;
    words
        .into_iter()
        .map(|(text, a, b)| {
            let start = a.max(cursor);
            let end = b.max(start);
            cursor = end;
            Word { text, start: start as f64 / 100.0, end: end as f64 / 100.0 }
        })
        .collect()
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ghostreel-asr: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, t0: i64, t1: i64) -> Token {
        Token { text: text.into(), t0, t1 }
    }

    #[test]
    fn tokens_make_words_with_their_pieces_and_punctuation_kept_together() {
        let tokens = [
            tok(" Esto", 100, 125),
            tok(" es", 125, 140),
            tok(" O", 150, 170),
            tok(" Cor", 170, 185),
            tok("res", 185, 205),
            tok(".", 205, 206),
        ];
        let words = group_words(&tokens, 100, 210);
        let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(text, ["Esto", "es", "O", "Corres."]);
        assert_eq!((words[0].start, words[0].end), (1.0, 1.25));
        assert_eq!((words[3].start, words[3].end), (1.7, 2.06));
    }

    #[test]
    fn a_token_without_a_time_takes_its_neighbours_and_nothing_runs_backwards() {
        let tokens = [tok(" uno", 100, 120), tok(" dos", -1, -1), tok(" tres", 90, 150)];
        let words = group_words(&tokens, 100, 200);
        assert_eq!(words.len(), 3);
        assert!(words.windows(2).all(|w| w[0].end <= w[1].start + 1e-9), "{words:?}");
        assert!(words.iter().all(|w| w.start >= 1.0 && w.end >= w.start && w.end <= 2.0), "{words:?}");
        assert!(group_words(&[], 0, 100).is_empty());
    }
}
