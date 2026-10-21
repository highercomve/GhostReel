//! `ghostreel-asr` — local speech-to-text helper for GhostReel.
//!
//! Reads 16 kHz mono f32le PCM on stdin and writes JSON lines on stdout:
//!
//! ```text
//! {"type":"loaded","model":"…","gpu":true}
//! {"type":"progress","percent":42}
//! {"type":"segment","start":0.0,"end":6.2,"text":"…","no_speech":0.01}
//! {"type":"done","language":"en","audio_s":123.4}
//! ```
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
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut language = "auto".to_string();
    let mut threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).min(8);
    let mut cpu = false;
    let mut serve = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next(),
            "--language" => language = it.next().ok_or("--language needs a value")?,
            "--threads" => {
                threads = it.next().and_then(|v| v.parse().ok()).ok_or("--threads needs a number")?;
            }
            // Force CPU inference (GPU out of memory, or no usable GPU).
            "--cpu" => cpu = true,
            "--serve" => serve = true,
            "--version" => {
                println!("ghostreel-asr {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "usage: ghostreel-asr --model <ggml-*.bin> [--language auto|en|es…] [--threads N] [--cpu] [--serve] < pcm_f32le_16k_mono"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args { model: model.ok_or("--model is required")?, language, threads, cpu, serve })
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
                let raw = std::fs::read(path).map_err(|e| format!("reading decoded audio: {e}"))?;
                transcribe(&ctx, &args, &raw)
            })();
            if let Err(error) = result {
                emit(json!({ "type": "error", "error": error }));
            }
        }
        Ok(())
    } else {
        let mut raw = Vec::new();
        std::io::stdin().read_to_end(&mut raw).map_err(|e| format!("reading audio from stdin: {e}"))?;
        transcribe(&ctx, &args, &raw)
    }
}

fn transcribe(ctx: &WhisperContext, args: &Args, raw: &[u8]) -> Result<(), String> {
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
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_nst(true);
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
        emit(json!({
            "type": "segment",
            "start": seg.start_timestamp() as f64 / 100.0,
            "end": seg.end_timestamp() as f64 / 100.0,
            "text": text,
            "no_speech": seg.no_speech_probability(),
        }));
    }
    let language = whisper_rs::get_lang_str(state.full_lang_id_from_state());
    emit(json!({ "type": "done", "language": language, "audio_s": audio_s }));
    Ok(())
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
