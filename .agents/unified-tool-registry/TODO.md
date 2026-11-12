# One tool registry for Sonar, clip generation and MCP

Source: scratchpad brief `tools-brief.md` (no plan.md / research.md for this task).

- [x] Stage 1: `ghostreel_core::tools` registry (spec, ctx, session, six area modules); MCP handlers moved in; `mcp.rs` is a thin adapter
- [x] Stage 2: Sonar renders and dispatches the whole registry (`chat/tools.rs` is a view over it); `search_moments` kept as an alias; editor prompts updated
- [x] Stage 3: clip workers and the orchestrator run the same action loop (`tools/agent_loop.rs`) through `AgentJob.tools`
- [x] Stage 4: no cap on tool rounds by default (`max_tool_rounds` 0 = none, clamps to 100 removed), repeat guard, Settings and composer show "Unlimited"
- [x] Stage 5: parity test (`mcp_sonar_and_the_clip_workers_are_given_exactly_the_same_tools`) plus MCP adapter tests
- [ ] Not done: the local helper and CLI loops of the script chat (`chat/mod.rs`) still carry their own drafting phase; they share the session, renderings and schemas but not the loop itself
