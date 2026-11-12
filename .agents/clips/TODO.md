# Clips TODO

See [plan.md](./plan.md).

## Milestone 1: clip candidates + glossary
- [x] Shared `script::check` (sentences, breath, shake, length, overlap); wire into `save_script` and the chat
- [x] Project glossary: `projects.glossary_json`, project settings UI, CLI/MCP access
- [x] Glossary into STT (local initial prompt, server `prompt` field) + post-pass corrections
- [x] Glossary into chat/clip prompts; re-transcribe action
- [ ] "Add to glossary" from transcript edits
- [x] `[chat_worker]` config, `resolve_worker`, `chat::workers` bounded parallel JSON jobs
- [x] Candidate mining from indexed transcripts (reuse `chat/build.rs`)
- [x] Clip run: workers score → orchestrator picks → check → save clip set
- [x] Schema: `scripts.set_id`, `scripts.kind`
- [x] Glossary in its own project tab, filter, "Apply to transcribed videos"
- [x] Fix: closing hold muted a speaking last clip (one-clip cuts had no audio)
- [x] Re-apply glossary corrections to stored transcripts without re-transcribing ("Apply to transcribed videos")
- [x] Speech captions: `captions.rs` cue builder, `burn_captions` render + sidecar .srt, `Script.captions`, live player toggle
- [x] Faces stage (face tracks, models in catalog, `ghostreel faces`, MCP `get_faces`) — delegated
- [x] Captions B: burn captions into renders + sidecar .srt — delegated
- [x] Project menu: every index option in one place — pipeline stage toggles incl. Camera shake and Faces (later Speakers/People); per-stage Run/Re-run with status (via redo_project_stage); after the faces stage lands
- [x] Comparison run (after faces + captions B verified; again after framing): same sandbox project,
      re-index new stages only, regenerate the 8 baseline clips (same in/out as v2) with captions +
      face-informed checks (+ 9:16 framing when M3 lands), then a fresh orchestrator pick using the
      face/speaker tools; side-by-side HTML page (frames, captions, screen-share flags, framing)
- [x] MCP: `generate_clips`, `find_clip_candidates`, `check_clips`, `save_clips`; CLI `ghostreel clips`
- [x] UI: Clips type in New Cut, Workers in Settings, clip set grid view, batch export

## Milestone 2/3 (done 2026-11-11)
- [x] Speakers stage (CAM++ voice embeddings, `speaker_turns`, `transcript_segments.speaker/person_id`), v17
- [x] People (face-track clusters, names kept across re-runs, merge), People tab, MCP `get_people`/`set_person`/`merge_people`/`get_speakers`
- [x] Active-speaker link (mouth motion vs speech activity)
- [x] Framing: `framing.rs` crop path (follow_speaker / stacked / fixed), render, live player crop, `preview_crop`, framing checks
- [x] Clip runs set follow_speaker (or stacked) framing for 9:16 / 1:1; clips name speakers by person

## Next (found in the comparison run)
- [ ] Framing per layout segment: a clip that switches between two-shot and screen share keeps one crop size (baseline 1, 4, 5)
- [ ] Visual label from face boxes over time (small camera tiles = screen-share layout), not only sparse keyframes
- [ ] Burned title covers the top face in stacked; move/shrink it
- [ ] Live player truncates long speech captions ("Porque Freddy …")
- [ ] Person thumbnail: prefer frontal, unoccluded boxes (mic in front of the face)
- [ ] 9:16 previews render at 304×540; offer a full-size preview
- [ ] Re-import re-runs repair passes on an already repaired script and lengthens it slightly

## Later
- [x] M2 speaker IDs + passages (indexing stages)
- [x] M3 faces/framing, face-following crop, stacked layout
- [ ] M4 word timestamps, word-synced captions, post copy
- [ ] M5 clip library (scheduling, tracking)
