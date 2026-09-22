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
            "--ngl" => a.ngl = val()?.parse().map_err(|_| "--ngl needs a number")?,
            "--cpu" => a.ngl = 0,
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
            "--version" => {
                println!("ghostreel-llm {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!(
                    "usage: ghostreel-llm [--model m.gguf --mmproj p.gguf] [--embed-model e.gguf] [--ctx N] \
[--concurrency N] [--kv-type f16|q8_0|q4_0] [--flash-attn auto|on|off] [--ngl N|--cpu]"
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
    mtmd: MtmdContext,
    ctx: LlamaContext<'a>,
    n_batch: u32,
    concurrency: usize,
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
        }))
    }

    fn batch_describe(&mut self, items: &[BatchDescribeItem]) -> Result<Vec<Value>, String> {
        let mut all_results = Vec::with_capacity(items.len());
        let chunk_size = self.concurrency.max(1);

        for chunk in items.chunks(chunk_size) {
            let t0 = Instant::now();
            self.ctx.clear_kv_cache();

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
        self.ctx.clear_kv_cache();
        // Thinking opens the block and leaves it to the model to close; otherwise the block is
        // pre-closed, which is what tells a Qwen-style model to answer straight away.
        let head = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        let mut text = if prompt.starts_with("<|im_start|>") {
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
            format_prompt_with_template(self.model, prompt)
        };

        let bitmap = if let Some(img_path) = image {
            let bm =
                MtmdBitmap::from_file(&self.mtmd, img_path, false).map_err(|e| format!("image {img_path}: {e:?}"))?;
            let marker = llama_cpp_2::mtmd::mtmd_default_marker();
            if let Some(pos) = text.rfind("<|im_start|>user\n") {
                text.insert_str(pos + "<|im_start|>user\n".len(), marker);
            } else {
                text = format!("{marker}{text}");
            }
            Some(bm)
        } else {
            None
        };

        let bitmaps: Vec<&MtmdBitmap> = bitmap.as_ref().into_iter().collect();
        let chunks = self
            .mtmd
            .tokenize(MtmdInputText { text, add_special: true, parse_special: true }, &bitmaps)
            .map_err(|e| format!("tokenize: {e:?}"))?;
        let prompt_tokens = chunks.total_tokens();
        let n_past = chunks
            .eval_chunks(&self.mtmd, &self.ctx, 0, 0, self.n_batch as i32, true)
            .map_err(|e| format!("prompt eval: {e:?}"))?;

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

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let backend = LlamaBackend::init().map_err(|e| e.to_string())?;
    let t0 = Instant::now();

    let vision_model = match &args.model {
        Some(path) => Some(
            LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(args.ngl))
                .map_err(|e| format!("loading {path}: {e}"))?,
        ),
        None => None,
    };
    let mut effective_ctx = args.ctx;
    let mut vision = match (&vision_model, &args.mmproj) {
        (Some(model), Some(mmproj)) => {
            let mtmd = MtmdContext::init_from_file(
                mmproj,
                model,
                &MtmdContextParams { use_gpu: args.ngl > 0, print_timings: false, ..Default::default() },
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
            let params = |ctx_tokens: u32| {
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(ctx_tokens))
                    .with_n_batch(n_batch)
                    .with_n_ubatch(n_batch)
                    .with_flash_attention_policy(flash)
                    .with_type_k(kv)
                    .with_type_v(kv)
            };
            // Out of VRAM for the asked-for window: half it once rather than failing to start.
            let (ctx, ctx_tokens) = match model.new_context(&backend, params(args.ctx)) {
                Ok(c) => (c, args.ctx),
                Err(first) => {
                    let smaller = (args.ctx / 2).max(2048);
                    eprintln!("ghostreel-llm: {} ctx failed ({first}); retrying with {smaller}", args.ctx);
                    let c = model
                        .new_context(&backend, params(smaller))
                        .map_err(|e| format!("vision context: {e} (after {first})"))?;
                    (c, smaller)
                }
            };
            effective_ctx = ctx_tokens;
            Some(Vision { model, mtmd, ctx, n_batch, concurrency: args.concurrency.max(1) as usize })
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
        "kv_type": args.kv_type,
        "flash_attn": args.flash_attn,
        "ngl": args.ngl,
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
