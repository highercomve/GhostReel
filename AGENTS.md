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
  Anything that trims clips must be followed by `end_on_sentences`. A cut made mostly of speech
  therefore cannot be squeezed at all — what it loses instead is a whole beat, which ends on a
  sentence by construction (`drop_beats_to_target`). The opening and the closing stay; middles go
  from the back. It is planned up front rather than greedily, because a greedy loop that stops at
  its floor leaves the cut over the ceiling and the next pass drops again (53.0 s → 43.4 s), and
  it never takes a script below half the clips the model chose — past that the score reports the
  overrun instead.
- **The length is spelled out as arithmetic.** "40 seconds" plus "a clip may run to 30 s" is a
  contradiction, and a model resolves it by ignoring the first. The prompt states the beats and
  the per-shot limit a requested duration implies; it is the only lever that works before the
  draft exists, and it took a local run from 179.6 s to 60.7 s against a 40 s target.
- **No shot may take more than a third of the target.** `max_clip_s` is an absolute 30 s and says
  nothing about a 40 s cut: a local model filled one with eight clips of twenty-odd seconds, every
  one legal, and it ran 349% over with nothing to trim. `pacing_issues` flags it while the model
  can still choose differently.
- **A bed is the beat's sound**, so its clips play muted and it cannot outlast them — anything that
  moves clips must be followed by `clamp_beds_to_beats`. That clamp is the *last* thing to touch a
  bed and so the last chance to cut a speaker off: when trimming the bed to its pictures would land
  inside a sentence, it holds the beat's last shot for the seconds the voice needs instead. The
  pictures give way to the voice, never the other way round.
- **A muted clip has no voice to protect**, so `pad_speech` and `end_on_sentences` both skip it.
  Padding b-roll to finish speech nobody can hear turned a two-second cutaway into twenty.
- **A clip must carry a whole thought.** Length, sentence boundaries and grounding were all
  checked; nothing asked whether the words were worth hearing. A cut came back at 39.9 s against
  a 40 s target — a perfect duration score — with three of eight clips being five seconds each of
  "Thank you." and "Okay.", a third of the piece. `empty_speech_issues` reports them and the
  prompt's checklist says it; `chat/build.rs` had held the rule since it was written, but only
  for the cut it builds itself.
- **A CLI brain writing a script gets a script-length clock.** One `CliAgentConfig` serves
  describing a frame (seconds) and writing a script (minutes: agy needs about nine on a 96-video
  project), and at the 180 s default every chat turn was killed. The chat takes the larger of the
  agent's own timeout and `script.server_timeout_s`.
- **Repairs are reported back to the model** on the assistant message (`repair_note`), because the
  repair runs after the last redraft and the model otherwise repeats the same mistake.
- **The conversation is re-sent every round**, so the loop keeps three quarters of the window and
  forgets the oldest tool results (`make_room`) rather than dying on the server's context error.

## The local (standalone) chat path

A local model is grammar-constrained to one action per round — `tool`, `final` or `reply` — and
`reply` is the cheapest branch to commit to. Left alone it is chosen constantly, and the turn ends
having done nothing. Four things keep standalone working; each was a real failure first:

- **The prompt must describe the pictures, not only the speech.** `speech_digest` gave every word
  and nothing about what is on screen, so a model that called no tool concluded there was no
  b-roll — both local models refused a project holding 639 described frames. `picture_digest` is
  one line per video and removes the whole class.
- **A `reply` before any tool call is pushed back on, twice** (`MAX_PUSHBACKS`). The first nudge
  names the tools; the second shows the JSON, because the first earns "I need to verify the visual
  content" — the model describing the tool call instead of making it.
- **`ToolMemo` applies here too.** Without it a local model asked the same question forever: four
  identical `get_video` calls for the same range, each answered afresh.
- **The final call drops the `reply` branch once footage has been opened**
  (`local_final_action_schema(allow_reply)`). A model that had done the whole job handed it over as
  markdown prose because that branch was still reachable.

`GHOSTREEL_DEBUG_CHAT` dumps every local round, not just the final draft — a turn that ends in
`reply` never reaches the draft, which is exactly the failure worth seeing.

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
- **Narration is a voice.** A beat with a written line over muted pictures was described to the
  judge as "nothing — the pictures play silent" and reported as hearing `""`, so it was scored
  against an absence and the editor was told to "pick shots of what is being talked about ("")".
  `state` labels it, and a mismatch quotes the narration when there is no transcript speech.
- **A beat with no frame descriptions is not asked about.** Asking anyway returned a flat "no" and
  scored good footage as filler; `unjudged_beats` reports the gap instead of averaging it away.
- **A stretch with no frame of its own is described by the last frame before it.** Not a guess:
  keyframes are de-duped by perceptual hash, so no frame means nothing changed — a locked-off
  interview collapses to one every 16–24 s. Without this a bedded beat over a static camera looked
  unindexed and was judged on its cutaway alone.
- **What it finds goes back to the model** on the assistant message, beside the repair note — a
  repair pass cannot make a shot of a road illustrate a sentence about a dog, so only the brain
  that picked the shot can fix it.
- `ghostreel script judge <id> --brief "…"`; `tests/eval_judge.rs` (`--ignored`) re-runs the four
  recorded drafts when the questions change, since a reworded question is a different measurement.

## What a server admits it can do (`probe.rs`)

GhostReel talks to anything OpenAI-compatible, but only llama.cpp describes its own shape. Never
offer a control the server behind it cannot honour — a setting that silently does nothing is worse
than one that is not there. `Probe::caps` carries what was actually asked:

- **`slots`** — `/props.total_slots`. Describing several frames at once only pays against a server
  started with matching slots. `None` means the server did not say (LM Studio, Ollama, a hosted
  endpoint) and is *not* the same as one; the settings page says so rather than guessing.
- **`slot_ctx`** — from `/slots`, because a server's window is divided among them. `--parallel 4
  -c 65536` gives each request 16k, and nothing errors when `chat_model.ctx_tokens` is larger —
  the model just runs out of room mid-script. `doctor` warns.
- **`router`** — llama.cpp router mode, detected by `/models` entries carrying a `status` field
  (a plain server's carry only id/aliases/meta/tags). There is no version endpoint; that
  difference *is* the detection. Only then can the model or its flags change without a restart
  (`highllama router use <preset>`).

## Keyframe sampling (`frames.rs`)

One low-resolution decode at 4 fps reads every frame's scene score. A score over
`scene_threshold` is a cut; the scores *below* it are accumulated, and when the running sum passes
`frames.change_budget` that moment earns a frame too — the cumulative half of the **twin-comparison
algorithm** (Zhang, Kankanhalli & Smoliar, 1993). A moving camera is formally an endless gradual
transition: it never trips a cut threshold, so a drive through a neighbourhood used to be sampled
by the interval clock alone, every keyframe a different street and everything between them
unindexed.

- **Accumulate, never compare to the last kept frame.** Measured on the DJI walk, distance from a
  fixed frame saturates at ~0.25 after four seconds and never grows: at 90 s and a completely
  different street it reads the same as at 4 s. Same budget, accumulation picked 10 keyframes and
  reference-distance picked 1. Do not "improve" this into a reference comparison.
- **`min_interval_s` is the only cost control that matters.** The budget is self-calibrating —
  footage changing five times faster gets five times the frames, with no notion of what a car is —
  and the floor is what stops that from becoming an overnight describe job. It is enforced in the
  scan, and it does *not* reset the accumulator, so fast footage gets its frame the moment it is
  allowed one instead of losing the overflow.
- **Cost lives in frame count**, not in the scan: the decode is O(duration) and unchanged, but
  extraction spawns an ffmpeg per frame and the describe stage is one vision call per frame.
- Calibration (Greet Mag, Sep 2026): ffmpeg's scene score accumulates at ~0.12/s on a moving
  camera and ~0.02/s on a locked-off interview. `change_budget = 1.0` therefore asks for a frame
  every ~8 s of travel and never fires on a talking head. Change it with measurements.
- PySceneDetect's `AdaptiveDetector` solves the *opposite* problem (suppressing false cuts during
  camera motion); AKS/Q-Frame score frames against the *query*, which an index built once and
  searched later cannot do. Neither applies here.

## Who is the interviewer (`interviewer.rs`)

`off_mic` decides whose voice to ignore, and everything reads it: the prompt's speech digest, the
quote candidates, the judge's "heard". `audio.rs` sets it acoustically — a segment a margin below
the video's median level is somebody off the lav — which only works when the interviewer is
*quieter*. On one Greet Mag tape they were not (9 off-mic of 76, against 24–26 elsewhere) and a
finished cut opened with "Okay, cool. So just tell me your name and the line of business that
you're in." The editor's verdict: "it's like I asked for bloopers."

`ghostreel script interviewer -p <project> [--dry-run|--undo]` asks Jev instead, one Noul per
line. It only ever *adds* flags — the acoustic test is evidence too — and records how each one was
set (`off_mic_source` 'level' or 'speech', with `off_mic_p`), because one bit cannot be reviewed,
re-judged at a different threshold, or undone without re-measuring every video.

- **Batch per video, never across the project.** "The surrounding lines are context" is only true
  if they are the same conversation: batched project-wide that line scored 0.53 and survived; among
  its own interview it is 0.97.
- **`jev.interviewer_threshold` is 0.78**, which is the gap the footage showed: unmistakable lines
  (a question, a mic check, a countdown) sit at 0.84–0.97 and real answers wrongly caught at
  0.70–0.76. Change it with evidence, not taste.
- It also catches slates ("Three, two, one, two"), mic checks, and — below the threshold — Whisper's
  silence hallucination "ご視聴ありがとうございました".

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
- **The option budget is spent on relevance, not on a sample.** A Choice takes at most 255
  options and the state shares that budget, so 639 described shots must become a few dozen. Each
  line now gets its own shortlist, ranked by content words shared with what is said over it, and
  a line that names nothing visible ("and we just love it") falls back to a spread. Picture-match
  went 0.70 → 0.84 with no mismatched beats, on the same four quotes: the deer shot finally
  reached the deer line. Deliberately word overlap and not embeddings — the builder's shape is
  that code enumerates and Jev judges, and pulling the search subsystem in here would widen what
  a change to this file can break.
- **A capped pool is spread, never truncated.** `.take(200)` in video order showed Jev shots from
  25 of 96 videos, and re-indexing made it *worse*: more frames per video pushed more videos out
  of the window, so half again as much footage produced the identical cut. `spread` takes a turn
  from each video instead.
- **The cutaway falls on a transcript boundary**, never inside a sentence. Cutting anywhere else
  is undone by `end_on_sentences` and the first real build came out 84% over its target.

## Jev builds, a chat brain refines

The two ways of getting a cut are good at opposite things. The builder picks pictures well —
0.84 picture-to-voice against agy's 0.77 — and structures poorly, because it chooses four quotes
independently and nothing ever asks whether they make a story (flow 0.59). agy is the reverse.

`script build` therefore judges its own cut and writes it into a chat session as turn one: the
brief as the user message, the script JSON and the judge's notes as the assistant's. A chat turn
on that session starts from real timecodes and a critique instead of fifteen rounds of looking.

    ghostreel script build -p P -b "…"            # ~10 s, prints the session id
    ghostreel script chat  -p P --session N "…"   # refine

Measured on the same brief: **76 editorial in 2m45s**, against 65 for the builder alone and 76 for
agy alone at 9m23s. Flow 0.59 → 0.73, ending 0.37 → 0.86, opening 0.59 → 0.71 — the last two the
best recorded. Attempts to fix the builder's structure *in the builder*, by rewording how middles
are chosen, made it worse twice; handing the problem to something that can hold a story did not.

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
