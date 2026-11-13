# A harness that makes AI editing easier, cheaper and faster

Source: scratchpad brief `harness-brief.md` (no plan.md / research.md for this task). TASK = harness-editing-tools.

## Stage 1: sentence ids, compact transcript, clips by sentence range
- [x] `sentences.rs`: ids `s<segment id>`, typed `SentenceError`, `resolve_range` / `resolve_script`, `parse_for_project`
- [x] `ScriptClip` / `AudioBed`: `from`/`to` (SentenceRange), seconds win when both; `try_from` wire types
- [x] resolved in save_script, chat draft parse (server, local, CLI loops), CLI `script import`, Tauri `save_script`
- [x] get_transcript compact text `s412 [12:03.4 Freddy off-mic] words`; JSON keeps `id` and `speaker`
- [x] speech_digest (the prompt's WHAT PEOPLE SAY) shows sentence ids
- [x] migration v23 `transcript_words`
- [x] check issue for cut edges judges from speech regions (`cuts::judge_*`), transcript rule only as fallback
- [x] tests: resolve + repair end to end through save_script, typed errors, judge, check

## Stage 2: read_cut
- [x] `readcut.rs` typed reading + compact text; scenes per moment; framing; boundary flags; check issues
- [x] `read_cut` tool (Read), registered, spec
- [x] tests: clipped word and leak from stored regions; draft read; muted/bed; unmeasured

## Stage 3: chapters stage + get_outline
- [x] migration v24 `chapters`
- [x] `chapters.rs`: boundary finder on stored embeddings (pure), length fallback, titling prompt/apply, store/load
- [x] stage `chapters`: PipelineConfig, index.rs (`run_chapter_jobs`), runs label, CLI `ghostreel chapters`, queue task, Project menu row + checkbox
- [x] `outline.rs` + `get_outline` tool (Read)
- [x] tests: boundary finder on synthetic embeddings, stage end to end, size budget on a 70-minute fixture

## Stage 4: benchmark fixes
- [x] get_frames exact frames when times are given (proxy, then source; kept under frames/exact), unique names
- [x] export_script `format: mp4` (also CLI), `PreviewOptions::deliverable` shared with clip export
- [x] get_scenes: classification (camera tiles) + long-range answers keep screen shares >= 3 s
- [x] word timestamps from both STT engines into `transcript_words`; `s412@3`

## Stage 5: prompts
- [x] Sonar editor prompt, clip worker + orchestrator prompts, MCP INSTRUCTIONS, README, CHANGELOG

## Stage 6: tests
- [x] parity test with the two new tools; MCP_TOOLS list; migration test; counts
- [x] `cargo test --workspace`, `cargo check` on src-tauri, `npx tsc --noEmit`, `npm run lint`
- [ ] on the real project (user): `ghostreel chapters`, `ghostreel index --stages speech` (see report), listen to a cut
