# AGENTS.md — GhostReel

Instructions and architecture reference for AI coding agents (symlinked as `CLAUDE.md`).

## What this is

GhostReel indexes folders of video for search and automated script editing: Whisper speech-to-text, keyframe extraction, vision model frame descriptions, vector embeddings, and hybrid keyword + semantic search with timestamp playback. Includes a Tauri v2 desktop app, an embedded Axum web server for remote LAN access, and the `ghostreel` CLI, all sharing `ghostreel-core`. Sibling of `ghostpen`.

- Decisions & spec: `.agents/plan.md`
- Tasks & progress: `.agents/TODO.md`

## Layout

| Path | Purpose |
|---|---|
| `crates/ghostreel-core` | Core library: DB (SQLite + FTS5 + sqlite-vec), config, paths, projects, media (ffprobe/hash), index pipeline, watch, models, script engine, otio, fcpxml, export, preview |
| `crates/ghostreel-cli` | `ghostreel` binary (`doctor`, `config`, `project`, `folder`, `index`, `status`, `models`, `script`, `mcp`) |
| `src-tauri` | Tauri v2 desktop app (`ghostreel-app` package, binary `target/*/ghostreel-app`). Includes Axum WebUI server (`webui.rs`) |
| `src/` | React 19 + TypeScript frontend (Vite) |
| `crates/ghostreel-asr` | Helper binary: standalone Whisper speech-to-text |
| `crates/ghostreel-llm` | Helper binary: standalone llama.cpp vision / chat helper with batched concurrency |
| `scripts/` | Sidecar fetch/stage, local installer, packaging scripts |

## Build & Test Commands

```bash
cargo test --workspace            # All tests (no GPU or external servers needed)
cargo clippy --workspace          # Lint checks
cargo fmt --all -- --check        # Formatting checks
cargo run -p ghostreel-cli -- doctor
npm install && npm run build      # Frontend build check
npx tauri dev                     # Run desktop app in dev mode
scripts/install-local.sh          # Build + install app and CLI into ~/.local
```

- `GHOSTREEL_CONFIG=<file>` / `GHOSTREEL_DATA=<dir>`: Override config and data directories for tests/demos.
- `GHOSTREEL_DEBUG_CHAT=<file>`: Dumps prompt and reply of every script chat turn.

## Packaging

```bash
scripts/build-helpers.sh                       # ghostreel-asr + ghostreel-llm
node scripts/fetch-sidecars.mjs                # Download pinned ffmpeg/ffprobe
node scripts/stage-helpers.mjs [--cuda]        # Stage sidecars into src-tauri/
NO_STRIP=true npx tauri build --config src-tauri/tauri.bundle.json --bundles appimage
scripts/fix-appimage.sh                        # Repack AppImage dropping host libcuda
scripts/package-cli.sh                         # target/dist/ghostreel-cli-linux-x64.tar.gz
```
CI: `.github/workflows/{check,release}.yml`. `externalBin` and `resources` live exclusively in the generated `tauri.bundle.json`, never in `tauri.conf.json`.

## Core Subsystems & Invariants

### 1. Indexing & Pipeline Stages (`index.rs`, `projects.rs`)
- **5 Pipeline Stages**: `probe` -> `transcribe` -> `frames` -> `describe` -> `embed`.
- **Configurable Stages**: Each project stores its `PipelineConfig` in `projects.pipeline_json` (migration 11).
  - CLI: `ghostreel project create <name> --no-transcribe`, `ghostreel project config -p <P> --disable transcribe`, `ghostreel index --stages / --skip`.
  - UI: Project creation modal and Project header settings dropdown (`⚙ Project ▾`).
- **Job Synchronization (`sync_pipeline_jobs`)**: If a stage is disabled across all projects watching a video, its jobs are marked `skipped`. Re-enabling a stage resets them to `pending`. Videos with `has_audio == 0` remain `skipped` for transcription.
- **Keyframe Sampling (`frames.rs`)**: Twin-comparison algorithm (Zhang et al.). Decodes at 4 fps; cuts trigger on `scene_threshold`, gradual transitions accumulate below threshold up to `change_budget`. `min_interval_s` enforces the minimum spacing.
- **Identity & Files**: Videos are identified by `blake3` content hash. `video_files` tracks filesystem locations per `(folder, path)`.

### 2. Remote Web Server (`src-tauri/src/webui.rs`, `src/events.ts`)
- Embedded Axum server runs inside Tauri app if enabled in Settings (`config.toml: [web]`).
- Serves embedded UI bundle, media at `/media`, and proxies commands via `POST /api/call`.
- Server-sent events at `/api/events` forward background queue updates, task completion, and chat progress.
- Frontend uses `call()` in `api.ts` (falls back to Tauri `invoke` when in desktop window) and `onEvent()` in `events.ts` (falls back to Tauri `listen`).
- Optional HTTP Basic auth (`auth_enabled`, `auth_user`, `auth_password`).

### 3. Script Chat & Repair Invariants (`chat.rs`, `repair.rs`)
- The prompt includes a full speech digest before any tools are called. Local models also receive a one-line per video `picture_digest`.
- **Repairs run in strict order**: `tidy_beat_ids` -> shaky stretches moved -> off-mic openings retimed -> clips without speech muted -> `lay_audio_beds` -> grounding -> `snap_to_segments` / `pad_speech` -> `fit_to_target` -> `end_on_turns` -> `end_on_sentences` -> `clamp_beds_to_beats`.
- **Invariants**:
  - Speech is never scaled and never cut mid-sentence (`end_on_sentences`). Overrun cuts drop whole beats from the back instead (`drop_beats_to_target`).
  - Audio beds belong to the beat: clips under beds play muted; beds never outlast pictures (`clamp_beds_to_beats`). If clamping lands mid-sentence, the beat holds the last picture.
  - Muted clips do not pad speech (`pad_speech` and `end_on_sentences` skip muted clips).
  - Speaker turns: `end_on_turns` cleans acknowledgements / handovers after `pad_speech`.
  - Empty thoughts ("Okay", "Thank you") are flagged by `empty_speech_issues`.

### 4. Editorial Judge & Builder (`judge.rs`, `build.rs`, `jev.rs`)
- **Judge (Jev / System One)**: Optional remote evaluation (`TYPESAFE_API_KEY`). Answers questions about picture grounding, openings, flow, and endings in parallel. State is ordered alphabetically (`BTreeMap`), framing text must sort before the data it frames.
- **Jev Builder (`chat/build.rs`)**: Assembles cuts algorithmically without LLM hallucination: enumerates on-mic quotes (4–12s, whole sentences, non-acknowledgements) and described shots, scores picture match via keyword overlap, and creates turn 1 of chat for refinement.
- **Off-mic detection (`interviewer.rs`)**: Acoustic level filter (`audio.rs`) + Jev question scoring (`jev.interviewer_threshold = 0.78`). Never removes flags, only adds evidence.

### 5. Timeline Export (`otio.rs`, `fcpxml.rs`)
- Native Rust implementations for `.otio` and Final Cut Pro 7 XML (`fcpxml.rs`). No Python dependencies. Audio beds export as J-cuts with shared source references.

## Key Rules & Architectural Constraints

1. **Two Runtime Profiles**: *standalone* (in-process helpers `ghostreel-asr`, `ghostreel-llm`) and *shared servers* (OpenAI-compatible endpoints). Always probe before use (`probe.rs`).
2. **Fixed Embedding Model**: Always `embeddinggemma-300M-Q8_0` (768-dim). Do not swap or mix embedding models.
3. **No Duplicate GGML Symbols**: `whisper-rs` and `llama-cpp-2` cannot link in the same binary; local models run in separate helpers.
4. **VRAM & Parallelism**: Target baseline GPU is 8 GB VRAM. Limit CUDA builds to `CMAKE_BUILD_PARALLEL_LEVEL=4` to avoid compiler OOM.
5. **Database Migrations**: Migrations in `db.rs` are strictly append-only. Table rebuilds must disable foreign keys during migration to prevent cascading deletes.
6. **Code Formatting & Quality**: Always ensure `cargo fmt --all -- --check` and `cargo clippy --workspace` pass before pushing.
