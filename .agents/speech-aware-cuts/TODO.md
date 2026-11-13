# speech-aware-cuts

- [x] snap_to_segments: in -> segment starts, out -> segment ends (muted clips left alone); regression test with the 12.02 / 12.86 handover
- [x] speech.rs: streamed 16 kHz decode, 10 ms levels, adaptive VAD, regions + quiet_after_s; schema v22 `speech_regions`
- [x] `speech` stage: PipelineConfig, index.rs run_speech_jobs / stage_statuses / run_pipeline, runs label, CLI `ghostreel speech`, queue task, Project menu row + checkbox
- [x] cuts.rs: place_cuts as the last pass of repair_for_import, chat finish_script, MCP save_script (repair) and CLI `script repair`
- [x] config: cut_to_speech, cut_guard_s, cut_lead_s, cut_max_move_s (tail = speech_overrun_s)
- [x] preview.rs: fade out never starts before the last voiced sound; no transcript-end fade; beds too; closing hold plays room tone
- [x] tests: VAD, cut rule (real case), in point, short pause, idempotence, snap, fades, migration, stage end to end
- [ ] run `ghostreel speech --project <name>` on the real project and listen to the benchmark cut
