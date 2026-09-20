# AGENTS.md — GhostReel

Instructions for AI coding agents in this repo (symlinked as `CLAUDE.md`).

## What this is

GhostReel (formerly "HighVid") indexes folders of video so you can search inside them: whisper
transcripts, keyframes described by a local vision model, embeddings, hybrid keyword + vector
search that jumps to the timestamp. Tauri v2 desktop app + `ghostreel` CLI sharing
`ghostreel-core`. Sibling of `~/Code/ghostpen`.

Source of truth: [`.agents/plan.md`](.agents/plan.md) (decisions D1–D12, runtime profiles §2a,
S0 findings §11). Progress: [`.agents/TODO.md`](.agents/TODO.md).

## Layout

| path | what |
|---|---|
| `crates/ghostreel-core` | config, paths, SQLite+FTS5+sqlite-vec DB, server probing, doctor, projects, media (hash/ffprobe), index (scan + jobs), watch, models, script, otio, fcpxml, export, preview |
| `crates/ghostreel-cli` | `ghostreel` binary (`doctor`, `config`, `project`, `folder`, `index [--watch]`, `status`, `models [list|download|remove|dir|use]`, `script [import|list|show|export|preview|chat]`, `mcp` = MCP stdio server in `mcp.rs`) |
| `src-tauri` | desktop app (package `ghostreel-app` → `target/*/ghostreel-app`, bundled as `GhostReel`; lib `ghostreel_lib`). Never name it `ghostreel`: it would overwrite the CLI binary in `target/` |
| `src/` | React + TS frontend (Vite) |
| `scripts/` | helper build, sidecar fetch/stage, CLI tarball (see Packaging), `install-local.sh`, `clean-stale-target.sh` |
| `spikes/s0-local` | throwaway S0 spike (separate workspace, CUDA builds) |

## Build & run

```bash
scripts/install-local.sh          # build + install app and CLI into ~/.local (see -h)
scripts/clean-stale-target.sh     # after moving the checkout: drop build caches pointing at the old path
cargo test --workspace            # core tests (no GPU, no servers needed)
cargo run -p ghostreel-cli -- doctor
npm install && npx tauri dev      # desktop app
npx tauri build --no-bundle       # release app binary (embedded frontend)
```
`GHOSTREEL_CONFIG=<file>` / `GHOSTREEL_DATA=<dir>` override config/data locations (tests, demos).
`GHOSTREEL_DEBUG_CHAT=<file>` dumps the prompt and reply of every chat turn — the way to check what
the editor was actually told rather than assuming.

## Packaging (plan §8, M7)

```bash
scripts/build-helpers.sh                       # ghostreel-asr + ghostreel-llm (CUDA if toolkit present)
node scripts/fetch-sidecars.mjs                # pinned BtbN LGPL ffmpeg/ffprobe (scripts/sidecars.json, sha256)
node scripts/stage-helpers.mjs [--cuda]        # helpers → src-tauri/binaries/<name>-<triple>, CUDA libs → src-tauri/lib/
                                               # also writes src-tauri/tauri.bundle.json (externalBin + resources)
NO_STRIP=true npx tauri build --config src-tauri/tauri.bundle.json --bundles appimage,deb
scripts/fix-appimage.sh                        # AppImage only: drop host driver libcuda, dedupe CUDA libs, repack
scripts/package-cli.sh                         # target/dist/ghostreel-cli-linux-x64.tar.gz
```
`npm run bundle:linux` / `bundle:windows` chain these. CI: `.github/workflows/{check,release}.yml`.
- `externalBin`/`resources` live only in the generated overlay, never in `tauri.conf.json`:
  tauri-build fails when a sidecar file is missing, which would break `tauri dev` and clippy.
- Tauri strips the triple and puts sidecars next to the app exe (`usr/bin/` on Linux), where
  `doctor::locate` already looks. Resources land in `usr/lib/GhostReel/` (Linux) / install dir (Windows).
- Helpers are linked with RUNPATH `$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib` (their `build.rs`;
  CLI tarball / deb+AppImage layouts). Never bundle `libcuda.so`/`libnvidia-*` (host driver).
  Staging patchelfs binaries built before the build.rs existed. Inside the AppImage, Tauri's
  linuxdeploy rewrites RUNPATH to `$ORIGIN/../lib` and copies libs into `usr/lib` (incl. the host's
  `libcuda.so.1`) — hence `fix-appimage.sh`; always run it after an AppImage build.
- ffmpeg pins must be month-end BtbN `autobuild-YYYY-MM-DD-*` tags (kept long-term), never `latest`.

## Script chat (`chat.rs`)

The editor gets the whole project's speech in the prompt before it calls anything
(`speech_digest`, ~11.5k tokens for 96 videos): a model that has to *ask* for each transcript
reads two tapes and builds the teaser out of whoever it found there. Tools are for pictures.

A draft then goes through repair passes, in this order, and the order matters:
`tidy_beat_ids` → shaky stretches moved → off-mic openings retimed → clips without speech muted →
`lay_audio_beds` → grounding → `snap_to_segments` / `pad_speech` → `fit_to_target` →
`end_on_sentences` → `clamp_beds_to_beats`. Invariants worth not breaking:

- **Speech is never scaled and never ends mid-sentence.** Length is a target; a sentence is not.
  Anything that trims clips must be followed by `end_on_sentences`.
- **A bed is the beat's sound**, so its clips play muted and it cannot outlast them — anything that
  moves clips must be followed by `clamp_beds_to_beats`. That clamp is the *last* thing to touch a
  bed and so the last chance to cut a speaker off: when trimming the bed to its pictures would land
  inside a sentence, it holds the beat's last shot for the seconds the voice needs instead. The
  pictures give way to the voice, never the other way round.
- **A muted clip has no voice to protect**, so `pad_speech` and `end_on_sentences` both skip it.
  Padding b-roll to finish speech nobody can hear turned a two-second cutaway into twenty.
- **Repairs are reported back to the model** on the assistant message (`repair_note`), because the
  repair runs after the last redraft and the model otherwise repeats the same mistake.
- **The conversation is re-sent every round**, so the loop keeps three quarters of the window and
  forgets the oldest tool results (`make_room`) rather than dying on the server's context error.

## The editorial judge (`chat/judge.rs`, `jev.rs`)

`chat/metrics.rs` counts craft faults and every one has a pass that drives it to zero. It cannot
see whether the shot on screen shows what the voice is talking about, whether the opening earns
attention, or whether the ending lands — the three things the editor actually complained about.
Jev (TypeSafe's System One model) answers those as probabilities; the weights that turn them into
one number live in `compose`, in code, so they can be changed without asking anything again.

- **Off unless told otherwise.** It is the only part of GhostReel that leaves the machine, so it
  needs both `jev.enabled` and a key (`TYPESAFE_API_KEY` beats `jev.api_key`). Settings has both.
- **One request, every question.** Jev reads the state once and answers all of them in parallel.
  Splitting them costs ~12x more for the same answers.
- **The state is what is heard and seen**, never timecodes or video ids: transcript text for each
  clip and bed, the vision model's frame descriptions, and whether the cut closes on a held image.
  Jev reads text only, and a judgement it cannot ground is a judgement of nothing.
- **A beat with no frame descriptions is not asked about.** Asking anyway returned a flat "no" and
  scored good footage as filler; `unjudged_beats` reports the gap instead of averaging it away.
- **What it finds goes back to the model** on the assistant message, beside the repair note — a
  repair pass cannot make a shot of a road illustrate a sentence about a dog, so only the brain
  that picked the shot can fix it.
- `ghostreel script judge <id> --brief "…"`; `tests/eval_judge.rs` (`--ignored`) re-runs the four
  recorded drafts when the questions change, since a reworded question is a different measurement.

## Building a cut by choosing (`chat/build.rs`)

A third way to get a script, beside the chat brains and importing JSON: `ghostreel script build`
and the *Build with Jev* button. No model writes anything. Code enumerates every quotable run of
on-mic speech and every described shot out of the index, Jev picks among them, and code assembles
the result — so an invented timecode is not a bug that was fixed, it is a thing that cannot be
expressed. It is also the fastest brain here: ~14 s against minutes for a local model, and on the
Greet Mag brief it scores 98/100 mechanically (40.7 s against a 40 s target) and 67/100
editorially, which beats every model draft recorded so far.

It cannot write narration, invent a framing device, or say anything the interviews do not. Rules
that questions only *suggest* are enforced in code, because "mostly" is not a rule:

- **A quote is a whole sentence with something in it**: 4–12 s, ≥10 words, on-mic, and it may not
  *end* on an acknowledgement — Jev chose a closing line finishing "…lost in some places. Okay."
  and then scored that ending 0.29.
- **Roles are picked one request at a time.** Independent questions cannot see each other's
  answers: asked together, one strong line won opening, middle *and* closing, and the cut was two
  quotes long.
- **A voice already used is off the menu** while any unheard one remains. Suggesting it in the
  question gave a four-line cut drawn from two people out of fifty-six.
- **Shots come only from footage nobody in the cut speaks in**, so a talking head can never be the
  b-roll under somebody else's voice.
- **The cutaway falls on a transcript boundary**, never inside a sentence. Cutting anywhere else
  is undone by `end_on_sentences` and the first real build came out 84% over its target.

## Timeline export

`.otio` (`otio.rs`) and Final Cut Pro 7 XML (`fcpxml.rs`) are both written here — there is no
Python sidecar, and re-adding one would be a regression. `tests/fixtures/golden.xml` came from the
official `otio-fcp-adapter` and is compared byte for byte; its one deliberate difference is noted
in the test. In xmeml, `start`/`end` count in the sequence rate and `in`/`out`/`duration` in the
source's, gaps are positions rather than elements, and a `<file>` is spelled out once then
referenced by id — which is what lets a bed export as a J-cut.

## Critical rules

1. **Two first-class runtime profiles** (plan §2a): *standalone* (models in-process, Windows +
   Linux, nothing else installed) and *shared servers* (highllama vision `:8089`, highllama
   embeddings `:8091`, GhostPen STT `:8771`). Each capability has `auto|local|server`, and the
   script chat additionally has `cli` (claude / agy / opencode / codex). Never load
   a local model while the matching server is usable — the dev box must not hold models twice.
2. **Only use a server that is reachable *and capable*** (`probe.rs`): vision needs
   `modalities.vision`; embeddings must be 768-dim `embeddinggemma-300M-Q8_0` (a chat model's
   CLS vectors silently ruin search); STT needs `capabilities.segments` (timestamps).
3. **One embedding model everywhere** (`embeddinggemma-300M-Q8_0`, 768-dim) so indexes are
   portable between profiles. The DB records `embed_model`/`embed_dim`.
4. **whisper-rs and llama-cpp-2 can't share a binary** (duplicate ggml symbols, S0) →
   local transcription will run in a separate `ghostreel-asr` helper.
5. **VRAM budget is the RTX 3070 8 GB.** On the 4070 dev box keep highllama at `KVTYPE=q4_0`;
   GhostPen's whisper segfaults when VRAM runs out.
6. CUDA builds: `CMAKE_BUILD_PARALLEL_LEVEL=4` — full parallel llama.cpp+whisper.cpp CUDA builds OOM.
7. Wayland: keep `apply_wayland_webkit_workaround()` (WebKit DMABUF "Error 71").
8. Migrations in `db.rs` are append-only once committed. Table rebuilds run with foreign keys off
   (see `Db::migrate`) — otherwise dropping `videos` cascades into jobs/transcripts.
9. Video identity = content hash; `video_files` are locations, unique per *(folder, path)* so
   overlapping folders of different projects don't fight over a file.
10. Don't create branches in `~/Code/ghostpen` or `~/Code/highllama`; work on main there.
11. **Vision model selection** (`config.vision.local_model`): the catalog has 4 pairs (bonsai-27b,
    gemma-3-4b-it, qwen2.5-vl-7b, qwen2.5-vl-3b); `models use <id>` checks the vision catalog
    **before** the generic `whisper()` function (which accepts any valid name). `ghostreel-llm`
    uses the model's own chat template via `apply_chat_template`; Qwen-style thinking is suppressed
    by appending `<think>\n\n</think>\n\n` when the template emits the ChatML assistant header.
    Known gap: non-Bonsai templates are untested on GPU. Ternary Bonsai 2 (`PQ2_0`/`PTQ1_0`, ggml
    types 142/143) needs PrismML's llama.cpp fork, which highllama installs beside its own build
    (`highllama prism install`) and selects by reading the GGUF's tensor types.
12. **A server brain gets a token ceiling and a long timeout**, not the reverse: `max_answer_tokens`
    exists so a model that will not stop fails as itself instead of as a dead socket, and
    `server_timeout_s` is generous because a local model needs minutes for a long script.
