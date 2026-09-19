# TODO

## Pending
- [ ] Carve `chat/tools/{spec,dispatch}.rs` out of `chat.rs`; `chat.rs` → `chat/mod.rs` with `pub use`
- [ ] `ToolSpec` catalog + `render_openai(detail)` + `tools_prose(detail)` + `TOOL_CONTRACT`
- [ ] Wire `build_system_prompt` and `local_action_schema()` to the catalog; delete the hand-written TOOLS paragraph
- [ ] `script.tool_docs` (auto|full|short) in ScriptConfig, validate, set_key
- [ ] Drift test (both renderings + action-schema enum from one list) and Short-rendering size budget test
- [ ] Tests named after the claims: listing_the_videos_does_not_make_their_footage_legal_to_cut, opening_a_video_makes_the_whole_of_it_legal_to_cut, a_search_hit_grounds_only_the_window_it_returned
- [ ] Split out `chat/{prompt,schema,repair,issues}.rs`; extract `repair_draft` + `finish_script`
- [ ] `chat/metrics.rs`: ScriptMetrics, measure, ScriptScore, score + unit tests
- [ ] `eval/{mod,fixture,replay,report}.rs`
- [ ] Fixture project (≤12 videos) + 2 briefs + 3 recorded drafts + 1 pathological draft
- [ ] `tests/eval_replay.rs` asserting each draft's `expect` block
- [ ] `debug_dump("server-final"/"server-retry")` + `grounding.json` capture on the Server backend
- [ ] `ghostreel eval run|dump|replay` (hidden subcommand), with research_s/draft_s split and `--record`
- [ ] `seed` + `temperature` in the `ghostreel-llm` Request and `LocalLlm::complete_sampled`
- [ ] `chat/draft.rs`: FinalDrafter, draft_candidates, Candidate, best
- [ ] `script.best_of`, `script.best_of_temperature`, `script.redraft_on_issues`
- [ ] Keep a redraft only when it scores higher (fixes today's unconditional accept)
- [ ] `ChatEvent::Candidate` + the 5 match sites (CLI, queue.rs, api.ts, TaskCard.tsx, ScriptsPanel.tsx)
- [ ] `TurnResult.candidates` (`#[serde(default)]`) and the "best of N: a / b / c" reply line
- [ ] AGENTS.md: ToolSpec rule, fixture-privacy rule, new `[script]` keys

## In Progress
- [x] Architecture design (this folder)

## Completed
- [x] Read chat.rs, config.rs, script.rs, db.rs, cliagent.rs, vision.rs, ghostreel-llm/main.rs, AGENTS.md, plan §4a
