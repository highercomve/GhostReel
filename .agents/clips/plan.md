# Clips: generate short clips from a project's footage

Idea source: the O Corres O Te Encaramas episode "Agentes de IA en producción: así hacemos este
pódcast" (2026-11, ~38:00–60:00), where the hosts describe their own podcast clipping pipeline. We
take the ideas, not the implementation: their pipeline is built for one Restream composite
recording; ours works on any project, across many videos that are already indexed.

## Principles

- **The project is the source.** Clips are mined from every video selected in a project, using
  what indexing already stored (transcripts, keyframe descriptions, embeddings, steadiness,
  off-mic flags). New knowledge about footage is a new **indexing stage**, stored once and reused
  by search, scripts and clips, never a clip-only analysis.
- **Deterministic first.** Anything a script can decide (sentence bounds, breath, length,
  overlap, shake, framing) is code. Models only make editorial choices.
- **Clips are scripts.** A clip is a one-clip script, so check, preview, export, the editor and
  versions all work unchanged.
- **Every backend works.** Orchestrator and workers can each be local, a custom server, or a
  CLI agent (claude, agy, opencode, codex), using the same JSON action loop as the chat today.

## What exists (map, 2026-11-10)

- Backends: `config::Backend` (`config.rs:25`), per-capability `VisionConfig` (`config.rs:104`),
  `[chat_model]` via `Config::chat_model()` (`config.rs:675`), `runtime::resolve_chat`
  (`runtime.rs:277`), `chat::ChatBackend` (`chat/mod.rs:99`). CLI agents use the local action
  loop (`chat/mod.rs:4452`), not MCP. No subagents; `JoinSet` parallelism only in indexing.
- `chat/build.rs`: sentence-bounded quote selection (`choose_quotes`, `build.rs:254`), the
  closest existing code to candidate mining.
- Validation is split: `save_script` only runs `script::validate` (`script.rs:282`) +
  `shake_issues` (`mcp.rs:474`); whole sentences / breath / shake repair live in the chat
  (`enforce_grounding_and_pacing` `chat/mod.rs:306`, `end_on_sentences` `:3025`, breath `:3152`).
- STT (`stt.rs`): no initial prompt, segment timestamps only (no words). No speaker IDs, only
  `off_mic` (`db.rs:255`). Preview pads instead of cropping (`preview.rs:516`). Scripts may
  override width/height. No per-project settings beyond `pipeline_json` (`db.rs:273`).
- Scripts UI: `ScriptsPanel.tsx` (modes `:91`, New Cut presets `:148`, modal `:2007`),
  `editor/CutsPanel.tsx` (groups versions by title), `ScriptEditor.tsx`, `PreviewPlayer.tsx`.

## Decisions (2026-11-10)

1. Roles: `[chat_model]` is the **orchestrator**; new `[chat_worker]` (backend, model,
   concurrency) for parallel **workers**, defaulting to the orchestrator's backend. No migration.
2. CLI agents reach footage tools through the **same JSON action loop** as today; workers only
   return structured JSON, so every backend can be a worker.
3. Results appear as a **clip set**: one row in the cuts list opening a grid of its clips.
   Needs `set_id` + `kind` on the `scripts` table.
4. Output format (original / 9:16 / 1:1) is chosen per run in the UI.
5. Milestone 1 = clip candidates + glossary.

## Milestone 1: clip candidates + glossary

### 1. Shared `script::check`
One checker for whole sentences, breath before cut, shake, length range, overlap between clips
of a set, and existing bounds checks. Used by `save_script`, the chat (its repair passes call
the same rules) and clips. Reports issues; repair stays explicit.

### 2. Project glossary
- Storage: `projects.glossary_json` (terms + optional "heard as" variants), exposed on `Project`,
  editable in project settings, CLI and MCP (`get/set` via project tools).
- Transcription: glossary → Whisper initial prompt in both engines (local helper param;
  `prompt` form field for the server). Post-pass replaces known mis-hearings.
- Prompts: chat and clip prompts include the glossary.
- Learning: editing a transcript word offers "Add to glossary". "Re-transcribe with glossary"
  re-queues STT for selected videos (normal indexing queue).

### 3. Worker role
`[chat_worker]` config + `runtime::resolve_worker`; a small `chat::workers` module that runs N
structured-JSON jobs on a bounded `JoinSet`, each with its own backend instance (CLI agents get
fresh sessions). Reusable later by script drafting.

### 4. Clip run
1. **Candidates (code):** for each selected video, sentence-bounded windows in the length range
   from stored segments; not starting on an off-mic line; avoiding shaky stretches; optional
   ranking against the brief with embeddings. Builds on `chat/build.rs`.
2. **Workers:** one job per video or ~10 min window: brief + glossary + transcript slice +
   candidates → scored picks `{in_s, out_s, title, quote, why, score}` (may nudge bounds).
3. **Orchestrator:** merge, drop overlaps, spread across videos/speakers, pick N, write title and
   caption text.
4. **Check + save:** `script::check` each pick; save as one-clip scripts in a clip set
   (`kind = clip`, `set_id`), with the chosen width/height.

Progress and cancellation go through the existing job/progress plumbing.

### 5. MCP + CLI
`generate_clips` (whole run); `find_clip_candidates`, `check_clips`, `save_clips` so an outside
agent can orchestrate with its own subagents. CLI: `ghostreel clips generate|list|export`.

### 5b. Speech captions (titles + subtitles for every cut, not only clips)
Today: titles = beat `on_screen_text` (drawtext, works); "narration subtitles" = beat narration
(voice-over text) — nothing captions what people *say* on camera.
- `captions.rs`: cues from stored transcript segments under every Source clip (and beds), mapped
  source → timeline time, cut to the clip range, split into ≤2 lines × ~42 chars / ~3 s cues
  (time split by character count until word timestamps exist). Glossary-corrected text.
- One cue builder feeds both the live player (Tauri command) and the render, like titles do.
- Render: `burn_captions` in preview/MP4 export — ASS style (bold, outline, bottom or centre,
  sized from frame height so 9:16 works); MP4 export also writes a sidecar `.srt`.
- `Script.captions: Option<CaptionStyle>` stored per cut; clip runs default it on. Titles stay
  `on_screen_text`; the orchestrator writes a short hook title per clip.
- UI: player toolbar "Speech captions" toggle (separate from narration subtitles) + position/size.
  Caption text is fixed via glossary / transcript edits; per-cue editing later.
- Later: word-by-word highlight once word timestamps exist (see finding 3).

### 6. UI
- New Cut modal: **Clips** type — brief (optional), count, length range, videos (selected
  footage), format, orchestrator/worker backend (defaults from Settings).
- Settings: Workers section next to the chat model.
- Clip set view: grid with score, title, quote, why; accept/reject; open in editor; batch export
  of accepted clips.

## Findings from the first end-to-end test (2026-11-11)

Sandbox (`GHOSTREEL_HOME`), the 70 min source episode, GhostPen (Whisper large-v3-turbo-q5_0,
qwen3.5-9b, embeddinggemma), 23-term glossary. 8 clips picked by hand as the orchestrator from
the indexed transcript, saved through `script import` (repair passes + `script_check`), previewed.

1. **Glossary prompt helps, heard_as lists must come from our own transcripts.** Zero Claw,
   JEPA, Kubernetes, homelab spelled right; but Whisper invented new mis-hearings ("ciroclot",
   "dipstick", "Deep Sea", "restring", "FFMP", "tome") the YouTube-derived lists did not cover.
   → "Add to glossary" from transcript edits matters; add **re-apply glossary corrections**
   to stored text without re-transcribing (milliseconds, no Whisper).
2. **Ambiguous mis-hearings need context.** "Claude" → "cloud" ×10 can't be a word swap.
   → follow-up: worker-role LLM correction pass on segments that fuzzy-match a glossary term.
3. **Long run-on segments break sentence-aligned cuts.** Server transcription returns some
   20–50 s lowercase segments without punctuation; a cut on a segment edge can still land
   mid-sentence ("…entonces aníbal creó un"), and `script_check` can't see it. Two transcript
   gaps of 14 s and 23 s were dropped. → word timestamps (planned M4) should move up, or
   re-segment long segments on pauses/punctuation; flag gaps in `script_check`.
4. **Visual suitability is half the pick.** 4 of 8 strong editorial picks are screen shares
   with tiny faces; 4 are two-host shots ideal for vertical. Keyframe descriptions already
   say which is which. → candidates carry a visual label from existing frame descriptions in
   M1 (no new stage); M3 faces/framing refines it.
5. Repair passes + check worked: every clip snapped to segments, got a 2 s closing hold, no
   warnings. Previews render in 2–3 s each.

## Milestone 2: people, speakers and faces (indexing + harness tools) — 2026-11-11

Reference look (youtube.com/shorts/gSohndlAyC8): 9:16 crop that follows the speaker (face in the
upper third, full bleed), cut to the other face when they respond; 2–3 word uppercase captions
centred with a highlighted keyword; lower-third name tag (name, handle, role).

Decisions: models run **locally with ONNX Runtime** (`ort`; small models from the model manager;
GhostPen server option later, like other capabilities). Rationale: per-frame detection ~1–3 ms vs
~750 ms for Qwen (70 min at 5 fps: ~1 min vs ~4.4 h), pixel-accurate stable boxes, face/voice
embeddings for identity, and Qwen cannot hear. Qwen stays for meaning: name suggestions from
on-screen name tags, expressions in keyframes (reactions), judging a crop preview.
Layouts in the first framing release: **follow the speaker** and **stacked**. People are
**auto-grouped and named in a People tab** (name, handle, role for the name tag).

Spike result (spikes/faces/README.md, 10 min @ 5 fps, CPU): **ort** YuNet 3.4 ms/frame, SFace
4.9 ms/face, 43.5 s total (13.8× real time); tract 19.4 / 35.4 ms, 274.6 s; identical embeddings;
tract had no op failures. → **ort** (static on Linux/macOS, onnxruntime.dll on Windows; GhostReel
already ships native llama.cpp/whisper.cpp). 2 faces in 99.3% of frames, but naive clustering of
unaligned crops gave 12 clusters for 2 people → production must align on YuNet's 5 landmarks
(ArcFace 112×112 template), drop small/blurry faces, and cluster *tracks* (IoU-linked over time)
by mean embedding, not single faces.

Indexing stages (stored once; used by search, scripts, clips, captions):
1. `faces`: YuNet-class detector at ~5 fps, boxes linked into tracks (IoU + embedding). Works for
   composite recordings (several faces in one frame) and per-person files alike.
2. `people`: SFace-class embeddings cluster tracks into project people across videos.
3. `speakers`: voice embeddings (ECAPA/WeSpeaker-class) cluster speech into speaker turns;
   each transcript segment gets a speaker (supersedes the one-bit off_mic).
4. `active_speaker`: link speakers to face tracks by mouth movement vs speech activity →
   for any moment: who talks, where their face is, who listens.

Harness tools (chat tools + MCP): `get_people`, `get_speakers(video, range)`,
`get_faces(video, t|range)` (boxes + framing quality), `get_reactions(video, range, person)`,
`preview_crop(video, range, layout)` (frames or a short proxy the agent can look at).

## Milestone 3: framing render + live preview
Per-clip `framing` {layout: follow_speaker | stacked | fixed, person?, box?}; one smoothed crop
path (dead zone, ease, hard cut on speaker change, fallback when no face) drives both the ffmpeg
render and the live player (CSS crop), so live preview = render. Name tag lower third from People.
Word timestamps (whisper.cpp token timestamps; GhostPen word granularity if supported) →
word-by-word caption style like the reference, building on the live captions.

## Later milestones (each adds an indexing stage)

- **M2 Speakers + passages:** speaker IDs per segment (grows `off_mic`; works for shared or
  per-person files) for speaker balance; "passages" = self-contained thoughts with summary +
  embedding, also used by search.
- **M3 Faces + framing:** per-keyframe face boxes; reject clips whose face leaves frame;
  face-following 9:16 crop; stacked speaker/listener layout.
- **M4 Words + captions:** word timestamps; word-synced burned-in captions; social post copy.
- **M5 Clip library:** scheduling, publishing state, link tracking (their "parrilla").
