//! `ghostreel-llm` — GhostReel's local model helper (llama.cpp via llama-cpp-2).
//!
//! Loads models once, then serves JSON-lines requests on stdin until EOF:
//!
//! ```text
//! → {"id":1,"cmd":"describe","image":"/f.jpg","prompt":"…","schema":{…},"max_tokens":1024}
//! ← {"id":1,"ok":true,"content":"{…json…}","prompt_tokens":1003,"gen_tokens":231,"secs":5.1}
//! → {"id":2,"cmd":"embed","texts":["…","…"]}
//! ← {"id":2,"ok":true,"embeddings":[[…],[…]],"secs":0.02}
//! ← {"id":3,"ok":false,"error":"…"}
//! ```
//!
//! Arguments: `--model <gguf> --mmproj <gguf>` (vision), `--embed-model <gguf>` (embeddings),
//! `--ctx 8192`, `--ngl 999`, `--cpu`. The first stdout line is `{"ready":true,…}` once models load.
//!
//! Sampling with a JSON-schema grammar: sample from the unconstrained distribution, check the token
//! against the grammar, and only if rejected apply the grammar to the full vocabulary and sample
//! again. That keeps generation at full speed (grammar over 248 k tokens each step is ~2.5× slower)
//! without the abort a pre-filtered candidate set can trigger when no candidate is grammatical.

use std::io::{BufRead, Write};
use std::num::NonZeroU32;
use std::process::ExitCode;
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use serde::Deserialize;
use serde_json::{Value, json};

struct Args {
    model: Option<String>,
    mmproj: Option<String>,
    embed_model: Option<String>,
    ctx: u32,
    ngl: u32,
    /// KV cache precision: `f16`, `q8_0` or `q4_0`. `q4_0` holds ~4× the context of `f16` in the
    /// same VRAM, at some quality cost (what highllama runs with).
    kv_type: String,
    /// `auto`, `on` or `off`.
    flash_attn: String,
    concurrency: u32,
    cpu_moe_layers: u32,
    explicit_ngl: bool,
    threads: i32,
    threadpool: bool,
    mmap: Option<bool>,
}

fn default_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    // Match GhostPen: avoid SMT contention on common 8-core / 16-thread CPUs.
    (if logical >= 16 { logical / 2 } else { logical }).clamp(1, 256) as i32
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: None,
        mmproj: None,
        embed_model: None,
        ctx: 8192,
        ngl: 999,
        kv_type: "q8_0".into(),
        flash_attn: "auto".into(),
        concurrency: 4,
        cpu_moe_layers: 0,
        explicit_ngl: false,
        threads: default_threads(),
        threadpool: true,
        mmap: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => a.model = Some(val()?),
            "--mmproj" => a.mmproj = Some(val()?),
            "--embed-model" => a.embed_model = Some(val()?),
            "--ctx" => a.ctx = val()?.parse().map_err(|_| "--ctx needs a number")?,
            "--concurrency" => a.concurrency = val()?.parse().map_err(|_| "--concurrency needs a number")?,
            "--ngl" => {
                a.ngl = val()?.parse().map_err(|_| "--ngl needs a number")?;
                a.explicit_ngl = true;
            }
            "--cpu-moe" => {
                a.cpu_moe_layers = val()?.parse().map_err(|_| "--cpu-moe needs a number")?;
                if a.cpu_moe_layers > 1024 {
                    return Err("--cpu-moe must be between 0 and 1024".into());
                }
            }
            "--cpu" => {
                a.ngl = 0;
                a.explicit_ngl = true;
            }
            "--kv-type" => {
                a.kv_type = val()?;
                if !["f16", "q8_0", "q4_0"].contains(&a.kv_type.as_str()) {
                    return Err(format!("--kv-type must be f16, q8_0 or q4_0 (got {})", a.kv_type));
                }
            }
            "--flash-attn" => {
                a.flash_attn = val()?;
                if !["auto", "on", "off"].contains(&a.flash_attn.as_str()) {
                    return Err(format!("--flash-attn must be auto, on or off (got {})", a.flash_attn));
                }
            }
            "--threads" => {
                a.threads = val()?.parse().map_err(|_| "--threads needs a number")?;
                if !(1..=256).contains(&a.threads) {
                    return Err("--threads must be 1–256".into());
                }
            }
            "--no-threadpool" => a.threadpool = false,
            "--mmap" => {
                a.mmap = Some(match val()?.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--mmap must be on or off".into()),
                });
            }
            "--version" => {
                println!("ghostreel-llm {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "usage: ghostreel-llm [--model m.gguf --mmproj p.gguf] [--embed-model e.gguf] [--ctx N] \
[--concurrency N] [--kv-type f16|q8_0|q4_0] [--flash-attn auto|on|off] [--cpu-moe N] [--threads N] [--mmap on|off] [--no-threadpool] [--ngl N|--cpu]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.model.is_some() != a.mmproj.is_some() {
        return Err("--model and --mmproj go together".into());
    }
    if a.model.is_none() && a.embed_model.is_none() {
        return Err("nothing to load: pass --model/--mmproj and/or --embed-model".into());
    }
    Ok(a)
}

#[derive(Clone, Deserialize)]
struct BatchDescribeItem {
    image: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct Request {
    id: Value,
    cmd: String,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    schema: Option<Value>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    texts: Option<Vec<String>>,
    #[serde(default)]
    items: Option<Vec<BatchDescribeItem>>,
    /// Let the model reason before answering. The grammar only binds after `</think>`, so the
    /// reasoning is free text; without this a schema forces JSON from the very first token.
    #[serde(default)]
    think: Option<bool>,
    /// Sampling seed. The same seed and prompt give the same answer, which is what makes a run
    /// reproducible — and what makes asking twice pointless unless the caller varies it.
    #[serde(default)]
    seed: Option<u32>,
    /// Sampling temperature. 0 is greedy; the default is deliberately low, because the answer is
    /// usually a structured draft rather than prose.
    #[serde(default)]
    temperature: Option<f32>,
}

const OPEN_THINK: &str = "<think>";
const CLOSE_THINK: &str = "</think>";

fn reply(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

struct Vision<'a> {
    model: &'a LlamaModel,
    model_name: String,
    mtmd: MtmdContext,
    ctx: LlamaContext<'a>,
    n_batch: u32,
    concurrency: usize,
    cached_prompt: Vec<llama_cpp_2::token::LlamaToken>,
    prompt_checkpoint: Option<(usize, SeqState)>,
}

fn format_prompt_with_template(model: &LlamaModel, user_message: &str) -> String {
    if let Ok(tmpl) = model.chat_template(None)
        && let Ok(msg) = LlamaChatMessage::new("user".to_string(), user_message.to_string())
        && let Ok(mut text) = model.apply_chat_template(&tmpl, &[msg], true)
    {
        if text.ends_with("<|im_start|>assistant\n") {
            text.push_str("<think>\n\n</think>\n\n");
        } else if text.ends_with("<|im_start|>assistant") {
            text.push_str("\n<think>\n\n</think>\n\n");
        }
        return text;
    }
    // Fallback: ChatML with thinking disabled
    format!("<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

impl Vision<'_> {
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        prompt_tokens: usize,
        n_past: i32,
        schema: Option<&Value>,
        max_tokens: usize,
        t0: Instant,
        think: bool,
        seed: u32,
        temperature: f32,
    ) -> Result<Value, String> {
        let prompt_seconds = t0.elapsed().as_secs_f64();
        let generation_start = Instant::now();
        let mut grammar = match schema {
            Some(s) if !s.is_null() => {
                let g = llama_cpp_2::json_schema_to_grammar(&s.to_string()).map_err(|e| format!("schema: {e:?}"))?;
                Some(LlamaSampler::grammar(self.model, &g, "root").map_err(|e| format!("grammar: {e:?}"))?)
            }
            _ => None,
        };
        let mut chain = LlamaSampler::chain_simple([
            LlamaSampler::penalties(self.model.n_vocab(), 64, 1.1, 0.0, 0.0),
            LlamaSampler::top_k(40),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(seed),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut batch = LlamaBatch::new(1, 1);
        let mut gen_tokens = 0usize;
        // While the model is still reasoning, the grammar must stay out of the way: it would
        // otherwise reject the very first word of the thought. It binds once `</think>` closes.
        let mut thinking = think;
        // Reasoning and answer share one budget, so an unbounded thought starves the answer and
        // the reply arrives cut off mid-JSON. Give the thought half, then close it ourselves.
        let think_max = (max_tokens / 2).max(256);
        let mut pos = n_past;
        while gen_tokens < max_tokens {
            if thinking && out.contains(CLOSE_THINK) {
                thinking = false;
            } else if thinking && gen_tokens >= think_max {
                let close = format!("{CLOSE_THINK}\n\n");
                let toks = self
                    .model
                    .str_to_token(&close, llama_cpp_2::model::AddBos::Never)
                    .map_err(|e| format!("close think: {e}"))?;
                for t in toks {
                    if gen_tokens >= max_tokens || pos >= self.ctx.n_ctx() as i32 {
                        break;
                    }
                    batch.clear();
                    batch.add(t, pos, &[0], true).map_err(|e| e.to_string())?;
                    self.ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
                    chain.accept(t);
                    pos += 1;
                    gen_tokens += 1;
                }
                out.push_str(&close);
                thinking = false;
                continue;
            }
            let mut cur = self.ctx.token_data_array();
            cur.apply_sampler(&chain);
            let mut tok = cur.selected_token().ok_or("sampler selected no token")?;
            if let Some(g) = grammar.as_ref().filter(|_| !thinking) {
                let mut single = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 1.0, 0.0)], false);
                single.apply_sampler(g);
                if !single.data[0].logit().is_finite() {
                    // Rejected: constrain the full vocabulary, then sample.
                    let mut full = self.ctx.token_data_array();
                    full.apply_sampler(g);
                    full.apply_sampler(&chain);
                    tok = full.selected_token().ok_or("no grammatical token")?;
                }
            }
            if let Some(g) = grammar.as_mut().filter(|_| !thinking) {
                g.accept(tok);
            }
            chain.accept(tok);
            if self.model.is_eog_token(tok) {
                break;
            }
            out.push_str(&self.model.token_to_piece(tok, &mut decoder, false, None).map_err(|e| e.to_string())?);
            gen_tokens += 1;
            batch.clear();
            batch.add(tok, pos, &[0], true).map_err(|e| e.to_string())?;
            self.ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
            pos += 1;
        }
        // The reasoning is not the answer: hand back what follows `</think>`, and the thought
        // separately so a caller can log it.
        let (thought, answer) = match out.split_once(CLOSE_THINK) {
            Some((t, a)) => (Some(t.trim_start_matches(OPEN_THINK).trim().to_string()), a.trim().to_string()),
            None => (None, out.clone()),
        };
        Ok(json!({
            "content": answer,
            "thinking": thought,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            // A thought that never closed means the answer never started.
            "truncated": gen_tokens >= max_tokens || (think && !out.contains(CLOSE_THINK)),
            "secs": t0.elapsed().as_secs_f64(),
            "prompt_seconds": prompt_seconds,
            "generation_seconds": generation_start.elapsed().as_secs_f64(),
            "model": self.model_name,
            "backend": "llama.cpp",
        }))
    }

    fn batch_describe(&mut self, items: &[BatchDescribeItem]) -> Result<Vec<Value>, String> {
        let mut all_results = Vec::with_capacity(items.len());
        let chunk_size = self.concurrency.max(1);

        for chunk in items.chunks(chunk_size) {
            let t0 = Instant::now();
            self.ctx.clear_kv_cache();
            self.cached_prompt.clear();
            self.prompt_checkpoint = None;

            struct SlotState {
                original_idx: usize,
                prompt_tokens: usize,
                pos: i32,
                seq_id: i32,
                max_tokens: usize,
                chain: LlamaSampler,
                grammar: Option<LlamaSampler>,
                decoder: encoding_rs::Decoder,
                out: String,
                gen_tokens: usize,
                next_token: llama_cpp_2::token::LlamaToken,
                active: bool,
            }

            let mut slots: Vec<SlotState> = Vec::with_capacity(chunk.len());
            let mut chunk_results: Vec<Option<Value>> = vec![None; chunk.len()];

            for (slot_idx, item) in chunk.iter().enumerate() {
                let marker = llama_cpp_2::mtmd::mtmd_default_marker();
                let prompt_str = item.prompt.as_deref().unwrap_or("Describe this image.");
                let user_msg = format!("{marker}{prompt_str}");
                let text = format_prompt_with_template(self.model, &user_msg);
                let bitmap = match MtmdBitmap::from_file(&self.mtmd, &item.image, false) {
                    Ok(bm) => bm,
                    Err(e) => {
                        chunk_results[slot_idx] = Some(json!({
                            "ok": false,
                            "error": format!("image {}: {e:?}", item.image),
                        }));
                        continue;
                    }
                };
                let chunks = match self
                    .mtmd
                    .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[&bitmap])
                {
                    Ok(c) => c,
                    Err(e) => {
                        chunk_results[slot_idx] = Some(json!({
                            "ok": false,
                            "error": format!("tokenize: {e:?}"),
                        }));
                        continue;
                    }
                };
                let prompt_tokens = chunks.total_tokens();
                let seq_id = slot_idx as i32;
                let n_past = match chunks.eval_chunks(&self.mtmd, &self.ctx, 0, seq_id, self.n_batch as i32, true) {
                    Ok(p) => p,
                    Err(e) => {
                        chunk_results[slot_idx] = Some(json!({
                            "ok": false,
                            "error": format!("prompt eval: {e:?}"),
                        }));
                        continue;
                    }
                };

                let mut grammar = match &item.schema {
                    Some(s) if !s.is_null() => match llama_cpp_2::json_schema_to_grammar(&s.to_string()) {
                        Ok(g) => match LlamaSampler::grammar(self.model, &g, "root") {
                            Ok(samp) => Some(samp),
                            Err(e) => {
                                chunk_results[slot_idx] = Some(json!({
                                    "ok": false,
                                    "error": format!("grammar: {e:?}"),
                                }));
                                continue;
                            }
                        },
                        Err(e) => {
                            chunk_results[slot_idx] = Some(json!({
                                "ok": false,
                                "error": format!("schema: {e:?}"),
                            }));
                            continue;
                        }
                    },
                    _ => None,
                };

                let mut chain = LlamaSampler::chain_simple([
                    LlamaSampler::penalties(self.model.n_vocab(), 64, 1.1, 0.0, 0.0),
                    LlamaSampler::top_k(40),
                    LlamaSampler::temp(0.2),
                    LlamaSampler::dist(42),
                ]);

                let mut cur = self.ctx.token_data_array();
                cur.apply_sampler(&chain);
                let Some(mut tok) = cur.selected_token() else {
                    chunk_results[slot_idx] = Some(json!({
                        "ok": false,
                        "error": "sampler selected no token",
                    }));
                    continue;
                };
                if let Some(g) = grammar.as_ref() {
                    let mut single = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 1.0, 0.0)], false);
                    single.apply_sampler(g);
                    if !single.data[0].logit().is_finite() {
                        let mut full = self.ctx.token_data_array();
                        full.apply_sampler(g);
                        full.apply_sampler(&chain);
                        if let Some(good_tok) = full.selected_token() {
                            tok = good_tok;
                        } else {
                            chunk_results[slot_idx] = Some(json!({
                                "ok": false,
                                "error": "no grammatical token",
                            }));
                            continue;
                        }
                    }
                }
                if let Some(g) = grammar.as_mut() {
                    g.accept(tok);
                }
                chain.accept(tok);

                let mut decoder = encoding_rs::UTF_8.new_decoder();
                let mut out = String::new();
                let mut active = true;
                let mut gen_tokens = 0;
                let max_tok = item.max_tokens.unwrap_or(1024);
                if self.model.is_eog_token(tok) {
                    active = false;
                } else {
                    match self.model.token_to_piece(tok, &mut decoder, false, None) {
                        Ok(piece) => {
                            out.push_str(&piece);
                            gen_tokens = 1;
                        }
                        Err(e) => {
                            chunk_results[slot_idx] = Some(json!({
                                "ok": false,
                                "error": e.to_string(),
                            }));
                            continue;
                        }
                    }
                }

                slots.push(SlotState {
                    original_idx: slot_idx,
                    prompt_tokens,
                    pos: n_past,
                    seq_id,
                    max_tokens: max_tok,
                    chain,
                    grammar,
                    decoder,
                    out,
                    gen_tokens,
                    next_token: tok,
                    active,
                });
            }

            let mut batch = LlamaBatch::new(slots.len().max(512), 1);
            while slots.iter().any(|s| s.active) {
                batch.clear();
                let mut active_indices = Vec::new();
                for (idx, slot) in slots.iter_mut().enumerate() {
                    if slot.active {
                        active_indices.push(idx);
                        if let Err(e) = batch.add(slot.next_token, slot.pos, &[slot.seq_id], true) {
                            slot.active = false;
                            chunk_results[slot.original_idx] = Some(json!({
                                "ok": false,
                                "error": format!("batch add error: {e}"),
                            }));
                        } else {
                            slot.pos += 1;
                        }
                    }
                }
                if active_indices.is_empty() {
                    break;
                }
                if let Err(e) = self.ctx.decode(&mut batch) {
                    for &idx in &active_indices {
                        slots[idx].active = false;
                        chunk_results[slots[idx].original_idx] = Some(json!({
                            "ok": false,
                            "error": format!("decode error: {e}"),
                        }));
                    }
                    break;
                }

                for (batch_offset, &slot_idx) in active_indices.iter().enumerate() {
                    let slot = &mut slots[slot_idx];
                    if !slot.active {
                        continue;
                    }
                    let mut cur = self.ctx.token_data_array_ith(batch_offset as i32);
                    cur.apply_sampler(&slot.chain);
                    let Some(mut tok) = cur.selected_token() else {
                        slot.active = false;
                        continue;
                    };
                    if let Some(g) = slot.grammar.as_ref() {
                        let mut single = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 1.0, 0.0)], false);
                        single.apply_sampler(g);
                        if !single.data[0].logit().is_finite() {
                            let mut full = self.ctx.token_data_array_ith(batch_offset as i32);
                            full.apply_sampler(g);
                            full.apply_sampler(&slot.chain);
                            if let Some(good_tok) = full.selected_token() {
                                tok = good_tok;
                            } else {
                                slot.active = false;
                                continue;
                            }
                        }
                    }
                    if let Some(g) = slot.grammar.as_mut() {
                        g.accept(tok);
                    }
                    slot.chain.accept(tok);

                    if self.model.is_eog_token(tok) {
                        slot.active = false;
                        continue;
                    }
                    if let Ok(piece) = self.model.token_to_piece(tok, &mut slot.decoder, false, None) {
                        slot.out.push_str(&piece);
                    }
                    slot.gen_tokens += 1;
                    slot.next_token = tok;
                    if slot.gen_tokens >= slot.max_tokens {
                        slot.active = false;
                    }
                }
            }

            let chunk_secs = t0.elapsed().as_secs_f64();
            for slot in slots {
                if chunk_results[slot.original_idx].is_none() {
                    let (thought, answer) = match slot.out.split_once(CLOSE_THINK) {
                        Some((t, a)) => {
                            (Some(t.trim_start_matches(OPEN_THINK).trim().to_string()), a.trim().to_string())
                        }
                        None => (None, slot.out.clone()),
                    };
                    chunk_results[slot.original_idx] = Some(json!({
                        "ok": true,
                        "content": answer,
                        "thinking": thought,
                        "prompt_tokens": slot.prompt_tokens,
                        "gen_tokens": slot.gen_tokens,
                        "truncated": slot.gen_tokens >= slot.max_tokens,
                        "secs": chunk_secs,
                    }));
                }
            }

            for res in chunk_results {
                all_results.push(res.unwrap_or_else(|| json!({ "ok": false, "error": "unknown slot error" })));
            }
        }

        Ok(all_results)
    }

    fn describe(
        &mut self,
        image: &str,
        prompt: &str,
        schema: Option<&Value>,
        max_tokens: usize,
    ) -> Result<Value, String> {
        let item = BatchDescribeItem {
            image: image.to_string(),
            prompt: Some(prompt.to_string()),
            schema: schema.cloned(),
            max_tokens: Some(max_tokens),
        };
        let mut results = self.batch_describe(&[item])?;
        if let Some(res) = results.pop() {
            if res["ok"].as_bool().unwrap_or(false) {
                Ok(res)
            } else {
                Err(res["error"].as_str().unwrap_or("describe failed").to_string())
            }
        } else {
            Err("no result returned".into())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn complete(
        &mut self,
        prompt: &str,
        schema: Option<&Value>,
        max_tokens: usize,
        think: bool,
        seed: u32,
        temperature: f32,
        image: Option<&str>,
    ) -> Result<Value, String> {
        let t0 = Instant::now();
        let mut text = completion_text(self.model, prompt, think);

        let bitmap = if let Some(img_path) = image {
            let bm =
                MtmdBitmap::from_file(&self.mtmd, img_path, false).map_err(|e| format!("image {img_path}: {e:?}"))?;
            insert_image_marker(&mut text);
            Some(bm)
        } else {
            None
        };

        let (prompt_tokens, n_past) = if let Some(bitmap) = &bitmap {
            self.cached_prompt.clear();
            self.prompt_checkpoint = None;
            self.ctx.clear_kv_cache();
            let chunks = self
                .mtmd
                .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[bitmap])
                .map_err(|e| format!("tokenize: {e:?}"))?;
            ensure_room(chunks.total_tokens(), max_tokens, self.ctx.n_ctx() as usize)?;
            let n = chunks
                .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.n_batch as i32, true)
                .map_err(|e| format!("prompt eval: {e:?}"))?;
            (chunks.total_tokens(), n)
        } else {
            let tokens = self
                .model
                .str_to_token(&text, llama_cpp_2::model::AddBos::Always)
                .map_err(|e| format!("tokenize: {e}"))?;
            ensure_room(tokens.len(), max_tokens, self.ctx.n_ctx() as usize)?;
            let mut common = reusable_prefix(&self.cached_prompt, &tokens);
            // A checkpoint before the assistant header survives appending a tool result.
            // Restore recurrent/SWA state first, then remove the attention cache's tail.
            if let Some((position, state)) = self.prompt_checkpoint.take()
                && position <= common
            {
                if self.ctx.state_seq_set(&state, 0).is_ok() {
                    common = position;
                    self.prompt_checkpoint = Some((position, state));
                } else {
                    self.ctx.clear_kv_cache();
                    common = 0;
                }
            }
            if !self
                .ctx
                .clear_kv_cache_seq(Some(0), Some(common as u32), None)
                .map_err(|e| format!("prompt cache: {e}"))?
            {
                self.ctx.clear_kv_cache();
                common = 0;
            }
            self.cached_prompt.clear();
            let prefix_end = text.rfind("<|im_start|>assistant").unwrap_or(text.len());
            let prefix = self
                .model
                .str_to_token(&text[..prefix_end], llama_cpp_2::model::AddBos::Always)
                .map_err(|e| format!("tokenize prefix: {e}"))?;
            // Leave the trailing whitespace token to be re-tokenized when a new turn is appended.
            let checkpoint = reusable_prefix(&prefix, &tokens).saturating_sub(1);
            let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
            let mut position = common;
            while position < tokens.len() {
                let boundary = if position < checkpoint { checkpoint } else { tokens.len() };
                let end = (position + self.n_batch as usize).min(boundary);
                batch.clear();
                for (i, token) in tokens.iter().enumerate().take(end).skip(position) {
                    batch.add(*token, i as i32, &[0], i + 1 == tokens.len()).map_err(|e| e.to_string())?;
                }
                self.ctx.decode(&mut batch).map_err(|e| format!("prompt eval: {e}"))?;
                position = end;
                if position == checkpoint {
                    let flags = LlamaStateSeqFlags::PARTIAL_ONLY;
                    // One bounded host snapshot; skip caching unusually large recurrent states.
                    if self.ctx.state_seq_get_size_ext(0, flags) <= 256 * 1024 * 1024 {
                        self.prompt_checkpoint = self.ctx.state_seq_get(0, flags).ok().map(|state| (position, state));
                    }
                }
            }
            self.cached_prompt.clone_from(&tokens);
            if common > 0 {
                eprintln!("ghostreel-llm: reused {common}/{} prompt tokens", tokens.len());
            }
            (tokens.len(), tokens.len() as i32)
        };

        self.sample(prompt_tokens, n_past, schema, max_tokens, t0, think, seed, temperature)
    }
}

struct Embedder<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    n_ctx: usize,
}

impl Embedder<'_> {
    fn embed(&mut self, texts: &[String]) -> Result<Value, String> {
        let t0 = Instant::now();
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let mut tokens = self
                .model
                .str_to_token(text, llama_cpp_2::model::AddBos::Always)
                .map_err(|e| format!("tokenize: {e}"))?;
            tokens.truncate(self.n_ctx);
            self.ctx.clear_kv_cache();
            let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
            for (i, t) in tokens.iter().enumerate() {
                batch.add(*t, i as i32, &[0], true).map_err(|e| e.to_string())?;
            }
            if self.ctx.decode(&mut batch).is_err() {
                self.ctx.encode(&mut batch).map_err(|e| format!("encode: {e}"))?;
            }
            let v = self.ctx.embeddings_seq_ith(0).map_err(|e| format!("embeddings: {e}"))?;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            vectors.push(v.iter().map(|x| x / norm).collect::<Vec<f32>>());
        }
        Ok(json!({ "embeddings": vectors, "secs": t0.elapsed().as_secs_f64() }))
    }
}

fn completion_text(model: &LlamaModel, prompt: &str, think: bool) -> String {
    // Thinking opens the block and leaves it to the model to close; otherwise the block is
    // pre-closed, which is what tells a Qwen-style model to answer straight away.
    let head = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
    if prompt.starts_with("<|im_start|>") {
        if prompt.ends_with("<think>\n\n</think>\n\n") {
            if think {
                // Caller wants reasoning: reopen the block this prompt already closed.
                format!("{}{head}", prompt.trim_end_matches("<think>\n\n</think>\n\n"))
            } else {
                prompt.to_string()
            }
        } else if prompt.ends_with("<|im_start|>assistant\n") {
            format!("{prompt}{head}")
        } else if prompt.ends_with("<|im_start|>assistant") {
            format!("{prompt}\n{head}")
        } else {
            format!("{prompt}\n<|im_start|>assistant\n{head}")
        }
    } else {
        format_prompt_with_template(model, prompt)
    }
}

fn insert_image_marker(text: &mut String) {
    let marker = llama_cpp_2::mtmd::mtmd_default_marker();
    if let Some(pos) = text.rfind("<|im_start|>user\n") {
        text.insert_str(pos + "<|im_start|>user\n".len(), marker);
    } else {
        *text = format!("{marker}{text}");
    }
}

fn reusable_prefix<T: PartialEq>(cached: &[T], tokens: &[T]) -> usize {
    cached.iter().zip(tokens).take_while(|(a, b)| a == b).count().min(tokens.len().saturating_sub(1))
}

fn ensure_room(prompt: usize, output: usize, context: usize) -> Result<(), String> {
    if output == 0 || prompt.saturating_add(output) > context {
        return Err(format!(
            "prompt needs {prompt} tokens plus {output} output tokens, but the local context holds {context}; reduce research/history or raise the context window"
        ));
    }
    Ok(())
}

/// Auto-fit uses llama.cpp's own model geometry and free-device-memory measurements.
/// Expert-only offload can free VRAM for a larger context, but is a capacity tradeoff,
/// not a promise that CPU experts run faster than GPU experts.
fn load_model(backend: &LlamaBackend, path: &str, args: &Args) -> Result<(LlamaModel, i32, Vec<String>), String> {
    use std::ffi::CString;
    let gpu = args.ngl > 0 && unsafe { llama_cpp_sys_2::llama_supports_gpu_offload() };
    let mut params = Box::pin(LlamaModelParams::default());
    let pattern = CString::new(moe_pattern(args.cpu_moe_layers)).map_err(|e| e.to_string())?;
    if !gpu || args.explicit_ngl || args.cpu_moe_layers > 0 {
        params.set(LlamaModelParams::default().with_n_gpu_layers(if gpu { args.ngl } else { 0 }));
        if gpu && args.cpu_moe_layers > 0 {
            params.as_mut().add_cpu_buft_override(&pattern);
        }
    } else {
        let path_c = CString::new(path).map_err(|e| e.to_string())?;
        let kv = match (args.flash_attn.as_str(), args.kv_type.as_str()) {
            ("off", _) | (_, "f16") => KvCacheType::F16,
            (_, "q4_0") => KvCacheType::Q4_0,
            _ => KvCacheType::Q8_0,
        };
        let flash = match args.flash_attn.as_str() {
            "on" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "off" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED,
            _ => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO,
        };
        let mut context = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(args.ctx))
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_type_k(kv)
            .with_type_v(kv)
            .with_flash_attention_policy(flash);
        // Leave room for the projector as well as runtime allocations.
        let projector = args.mmproj.as_ref().and_then(|p| std::fs::metadata(p).ok()).map_or(0, |m| m.len());
        let margin = (512_u64 * 1024 * 1024).saturating_add(projector).min(usize::MAX as u64) as usize;
        let mut margins = vec![margin; unsafe { llama_cpp_sys_2::llama_max_devices() }];
        if let Err(e) =
            params.as_mut().fit_params(&path_c, &mut context, &mut margins, 2048, llama_cpp_sys_2::GGML_LOG_LEVEL_WARN)
        {
            eprintln!("ghostreel-llm: auto-fit failed ({e}); loading on CPU");
            params.set(LlamaModelParams::default().with_n_gpu_layers(0));
        }
        eprintln!(
            "ghostreel-llm: auto-fit selected {} GPU layers, expert overrides {:?}",
            params.n_gpu_layers(),
            params.tensor_buft_override_patterns()
        );
    }
    let ngl = params.n_gpu_layers();
    let expert_overrides = params.tensor_buft_override_patterns();
    let mmap = args.mmap.unwrap_or(expert_overrides.is_empty());
    params.as_mut().set_use_mmap(mmap);
    eprintln!("ghostreel-llm: mmap={mmap}, CPU threads={}", args.threads);
    LlamaModel::load_from_file(backend, path, &params)
        .map(|model| {
            // Auto-fit leaves -1 unchanged when the whole model already fits. llama.cpp
            // treats that sentinel as full offload, not CPU-only placement.
            let actual_ngl = effective_gpu_layers(ngl, model.n_layer());
            (model, actual_ngl, expert_overrides)
        })
        .map_err(|e| format!("loading {path}: {e}"))
}

fn moe_pattern(layers: u32) -> String {
    let blocks = (0..layers).map(|i| i.to_string()).collect::<Vec<_>>().join("|");
    format!(r"blk\.({blocks})\.ffn_(up|down|gate_up|gate)_(ch|)exps")
}

fn effective_gpu_layers(configured: i32, layers: u32) -> i32 {
    let maximum = layers.saturating_add(1).min(i32::MAX as u32) as i32;
    if configured < 0 { maximum } else { configured.min(maximum) }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let backend = LlamaBackend::init().map_err(|e| e.to_string())?;
    let t0 = Instant::now();

    let (vision_model, effective_ngl, expert_overrides) = match &args.model {
        Some(path) => {
            let model = load_model(&backend, path, &args)?;
            (Some(model.0), model.1, model.2)
        }
        None => (None, 0, Vec::new()),
    };
    let mut effective_ctx = args.ctx;
    let mut effective_kv = args.kv_type.clone();
    let mut vision = match (&vision_model, &args.mmproj) {
        (Some(model), Some(mmproj)) => {
            let mtmd = MtmdContext::init_from_file(
                mmproj,
                model,
                &MtmdContextParams {
                    use_gpu: effective_ngl > 0,
                    n_threads: args.threads,
                    print_timings: false,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("loading {mmproj}: {e:?}"))?;
            let n_batch = 512;
            let kv = match args.kv_type.as_str() {
                "f16" => KvCacheType::F16,
                "q4_0" => KvCacheType::Q4_0,
                _ => KvCacheType::Q8_0,
            };
            let flash = match args.flash_attn.as_str() {
                "on" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
                "off" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED,
                _ => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO,
            };
            let mut ctx_tokens = args.ctx.min(model.n_ctx_train().max(512)).max(512);
            let mut active_kv = if args.flash_attn == "off" { KvCacheType::F16 } else { kv };
            let mut ctx = loop {
                let params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(ctx_tokens))
                    .with_n_batch(n_batch)
                    .with_n_ubatch(n_batch)
                    .with_n_threads(args.threads)
                    .with_n_threads_batch(args.threads)
                    .with_flash_attention_policy(flash)
                    .with_type_k(active_kv)
                    .with_type_v(active_kv)
                    .with_offload_kqv(effective_ngl > 0)
                    .with_op_offload(effective_ngl > 0);
                match model.new_context(&backend, params) {
                    Ok(c) => break c,
                    Err(e) => {
                        if active_kv != KvCacheType::F16 && args.flash_attn == "auto" {
                            active_kv = KvCacheType::F16;
                            eprintln!("ghostreel-llm: quantized KV context failed ({e}); trying f16");
                        } else if ctx_tokens > 2048 {
                            ctx_tokens = (ctx_tokens / 2).max(2048);
                            if args.flash_attn != "off" {
                                active_kv = kv;
                            }
                            eprintln!("ghostreel-llm: retrying with a {ctx_tokens}-token context ({e})");
                        } else {
                            return Err(format!("local context: {e}"));
                        }
                    }
                }
            };
            if args.threadpool && !ctx.attach_resident_threadpool(std::num::NonZeroI32::new(args.threads).unwrap()) {
                return Err("could not allocate the resident CPU threadpool".into());
            }
            effective_ctx = ctx.n_ctx();
            effective_kv = match active_kv {
                KvCacheType::F16 => "f16",
                KvCacheType::Q4_0 => "q4_0",
                _ => "q8_0",
            }
            .into();
            eprintln!("ghostreel-llm: actual context {effective_ctx}, KV {active_kv:?}, GPU layers {effective_ngl}");
            Some(Vision {
                model,
                model_name: std::path::Path::new(args.model.as_ref().unwrap())
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                mtmd,
                ctx,
                n_batch,
                concurrency: args.concurrency.max(1) as usize,
                cached_prompt: Vec::new(),
                prompt_checkpoint: None,
            })
        }
        _ => None,
    };

    let embed_model = match &args.embed_model {
        Some(path) => Some(
            // Small model: fine on CPU, keeps VRAM for the vision model.
            LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(0))
                .map_err(|e| format!("loading {path}: {e}"))?,
        ),
        None => None,
    };
    let mut embedder = match &embed_model {
        Some(model) => {
            let n_ctx = 2048u32;
            let ctx = model
                .new_context(
                    &backend,
                    LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(n_ctx))
                        .with_n_batch(n_ctx)
                        .with_n_ubatch(n_ctx)
                        .with_embeddings(true),
                )
                .map_err(|e| format!("embedding context: {e}"))?;
            Some(Embedder { model, ctx, n_ctx: n_ctx as usize })
        }
        None => None,
    };

    let dim = embed_model.as_ref().map(|m| m.n_embd());
    reply(json!({
        "ready": true,
        "vision": vision.is_some(),
        "embed_dim": dim,
        "ctx": effective_ctx,
        "concurrency": args.concurrency,
        "kv_type": effective_kv,
        "flash_attn": args.flash_attn,
        "ngl": effective_ngl,
        "cpu_moe_layers": args.cpu_moe_layers,
        "expert_overrides": expert_overrides,
        "model": args.model.as_ref().and_then(|p| std::path::Path::new(p).file_stem()).map(|p| p.to_string_lossy()),
        "backend": "llama.cpp",
        "threads": args.threads,
        "resident_threadpool": args.threadpool,
        "token_count": true,
        "load_secs": t0.elapsed().as_secs_f64(),
    }));

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                reply(json!({ "id": null, "ok": false, "error": format!("bad request: {e}") }));
                continue;
            }
        };
        let result = match req.cmd.as_str() {
            "count_tokens" => match (&vision, req.prompt.as_deref()) {
                (Some(v), Some(prompt)) => {
                    let text = completion_text(v.model, prompt, req.think.unwrap_or(false));
                    if let Some(path) = req.image.as_deref() {
                        let bitmap =
                            MtmdBitmap::from_file(&v.mtmd, path, false).map_err(|e| format!("image {path}: {e:?}"));
                        bitmap.and_then(|bitmap| {
                            let mut text = text;
                            insert_image_marker(&mut text);
                            v.mtmd
                                .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &[&bitmap])
                                .map(|chunks| json!({"tokens":chunks.total_tokens()}))
                                .map_err(|e| format!("tokenize: {e:?}"))
                        })
                    } else {
                        v.model
                            .str_to_token(&text, llama_cpp_2::model::AddBos::Always)
                            .map(|tokens| json!({"tokens": tokens.len()}))
                            .map_err(|e| format!("tokenize: {e}"))
                    }
                }
                _ => Err("count_tokens needs a loaded model and prompt".into()),
            },
            "batch_describe" => match (&mut vision, &req.items) {
                (Some(v), Some(items)) => {
                    let t0 = Instant::now();
                    match v.batch_describe(items) {
                        Ok(results) => Ok(json!({
                            "results": results,
                            "secs": t0.elapsed().as_secs_f64(),
                        })),
                        Err(e) => Err(e),
                    }
                }
                (None, _) => Err("vision model not loaded".into()),
                (_, None) => Err("batch_describe needs items".into()),
            },
            "describe" => match (&mut vision, &req.image) {
                (Some(v), Some(image)) => v.describe(
                    image,
                    req.prompt.as_deref().unwrap_or("Describe this image."),
                    req.schema.as_ref(),
                    req.max_tokens.unwrap_or(1024),
                ),
                (None, _) => Err("vision model not loaded".into()),
                (_, None) => Err("describe needs image".into()),
            },
            "complete" => match (&mut vision, &req.prompt) {
                (Some(v), Some(prompt)) => {
                    // 42 and 0.2 were the hard-coded values: keep them as the defaults so a
                    // caller that says nothing gets exactly what it always got.
                    v.complete(
                        prompt,
                        req.schema.as_ref(),
                        req.max_tokens.unwrap_or(2048),
                        req.think.unwrap_or(false),
                        req.seed.unwrap_or(42),
                        req.temperature.unwrap_or(0.2),
                        req.image.as_deref(),
                    )
                }
                (None, _) => Err("vision/llm model not loaded".into()),
                (_, None) => Err("complete needs prompt".into()),
            },
            "embed" => match (&mut embedder, &req.texts) {
                (Some(e), Some(texts)) => e.embed(texts),
                (None, _) => Err("embedding model not loaded".into()),
                (_, None) => Err("embed needs texts".into()),
            },
            other => Err(format!("unknown cmd {other}")),
        };
        match result {
            Ok(mut v) => {
                v["id"] = req.id;
                v["ok"] = json!(true);
                reply(v);
            }
            Err(e) => reply(json!({ "id": req.id, "ok": false, "error": e })),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ghostreel-llm: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_fit_full_offload_sentinel_keeps_gpu_context_and_projector_enabled() {
        assert_eq!(effective_gpu_layers(-1, 32), 33);
        assert_eq!(effective_gpu_layers(999, 32), 33);
        assert_eq!(effective_gpu_layers(0, 32), 0);
        assert_eq!(effective_gpu_layers(8, 32), 8);
    }

    #[test]
    fn cache_reuses_only_shared_tokens_and_always_evaluates_logits() {
        assert_eq!(reusable_prefix(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3]), 2);
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2]), 1);
        assert_eq!(reusable_prefix(&[1, 2], &[9, 2]), 0);
        assert_eq!(reusable_prefix(&[1, 2], &[]), 0);
    }

    #[test]
    fn generation_reserves_output_and_rejects_overflow() {
        assert!(ensure_room(512, 512, 1024).is_ok());
        assert!(ensure_room(513, 512, 1024).is_err());
        assert!(ensure_room(usize::MAX, 512, 1024).is_err());
        assert!(ensure_room(512, 0, 1024).is_err());
    }

    #[test]
    fn expert_offload_targets_only_requested_blocks_expert_tensors() {
        assert_eq!(moe_pattern(2), r"blk\.(0|1)\.ffn_(up|down|gate_up|gate)_(ch|)exps");
    }
}
