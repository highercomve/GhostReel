# Changelog

All notable changes to GhostReel. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/) while the project is 0.x (minor = features, patch = fixes).

## [Unreleased]

### Added
- **`scripts/install-local.sh`** — build from source and install for the current user, the way
  GhostPen does it: `ghostreel-app` and `ghostreel` into `~/.local/bin` with a desktop entry and
  icon, `--helpers` to add the local model helpers, `--no-build` to install what is already there.
  It builds through `tauri build`, since a bare `cargo build` leaves the app pointing at the Vite
  dev server and opening on a blank window.

## [0.2.0] — 2026-09-19

The script chat stopped being a lottery. Every word spoken in the project goes to the editor before
it cuts, a voice can run on under the pictures instead of stopping at the cutaway, no clip ever ends
in the middle of a sentence, and the timeline is edited by ear rather than by dragging. Timeline
export is written here now, so nothing Python ships in the bundle.

Interviews survive the cut in particular: sentences are never scaled to hit a length, clips no
longer stop on the last sample of a word, every join is faded, and the editor can answer in words
instead of always redrafting.

### Added
- **A voice can run under the pictures.** Sound was bound to whatever clip was on screen — one
  `audio` field per clip, `source` or `mute` — so the moment a beat cut away from a speaker it
  fell silent, and every teaser came out as a talking head followed by a mute postcard. A beat can
  now carry a *bed*: the stretch of speech that plays across it while the pictures change
  underneath. The editing rules asked for exactly this ("cut to a picture of what they are
  describing while they keep talking"); the schema had no way to say it.

  The pipeline lays beds itself where a beat would otherwise go quiet, so it works with a local
  model that knows nothing about the field, and the editor can name one explicitly. A laid bed
  starts as far before the speaker's own clip as that clip sits into the beat, so their lips still
  match when we cut to them, and it stops at a sentence end before the interviewer's next question
  (`script.infer_audio_beds`, `script.max_bed_extend_s`). On the reference footage the b-roll went
  from digital silence to her voice at a normal level.

  Exports carry it as a real J-cut: A1 holds one audio clip spanning several video clips, which is
  what an editor would cut by hand in Premiere.
- **The script editor shows the cut on three lanes** — titles, picture and sound — the way an NLE
  does, because a bed is invisible in a list of beats. A striped bar is a voice the pipeline
  carried under the pictures.

  It is edited by keyboard as much as by mouse, because the decisions here are fractions of a
  second and a drag cannot hit them: the preview player's playhead is drawn across the lanes,
  clicking seeks, `←`/`→` trim the out point (`shift` the in point) by a second, `alt` by 0.2 s and
  `ctrl` by 5 s, `i`/`o` put an edge exactly where you are listening, `s` pulls both edges onto the
  nearest sentence, and `z` zooms to the selected beat. The keys are listed under the lanes.
- **The editor can reply.** A third action carries text and drafts nothing, so a question, an
  ambiguous brief or a message that is not about the video gets an answer instead of a script. A
  note pasted into the chat by mistake used to come back as a 120 s single-beat draft. A reply
  leaves the previous version alone.
- **The audio track with the speech.** A field recording carries several tracks — on this project
  four, two of them digital silence — and the preview took the first on faith. The index records
  the loudest non-silent one and previews use it.
- **Off-microphone speech, marked.** In an interview the subject is on a lav and the questions come
  from across the room, 12 dB down. Each transcript segment is compared with the video's median
  speech level and flagged; the rules say never to open a clip or build a beat on one.
- **Even out audio** — a checkbox beside Burn titles that levels loudness across clips
  (`loudnorm`), for cuts that mix a lav with a room mic.
- **Task cards open**, showing what the turn is working on and a live log of each tool call.
- **Timeline export works again, and carries nothing with it.** Export failed with "ghostreel-otio
  sidecar not found": staging and packaging both knew about the OpenTimelineIO sidecar, but nothing
  ever built it, so every release shipped without it. Rather than freeze 11.9 MB of Python into
  each bundle, Final Cut Pro 7 XML is now written here (`fcpxml.rs`) — frame counts on the right
  clock, NTSC rates, gaps as positions, files declared once and referenced by id — and read back
  for validation. `.otio` was always written in Rust; both formats now are, and the sidecar,
  its build script, its CI steps and the Python toolchain are gone.

  On the reference footage the Rust writer's XML is byte for byte what the official
  `otio-fcp-adapter` produced, save one filename it left percent-encoded, and OpenTimelineIO reads
  the two files back identically. A fixture from that adapter is checked in and compared byte for
  byte on every test run.


- **Every word spoken is in front of the editor before it cuts.** A model that had to *ask* for
  each transcript only read the tapes it already suspected: one run opened two videos, built the
  whole teaser out of the speaker it found there, and never looked at the three other people who
  said better things. The project's speech now goes in the prompt up front — with the timestamps to
  cut on and the interviewer's questions marked — which is the order an editor works in: read the
  interviews, decide the story, then go looking for pictures. It costs about 11 500 tokens for a
  96-video project. Qwen3.5-9B went from 113–135 % over its target across four runs to 20 %, and
  Bonsai-27B, which had never produced a usable script, found five different residents.
  (`script.speech_in_prompt`; a local model gets half its window, and tapes that do not fit are
  named rather than hidden.)
- **The editor is told how to work here**, not only what a good cut is: a short method at the top
  of the prompt — read the transcripts, find a picture for each chosen sentence, show the face then
  cut away with a bed, fix the total yourself — and a checklist to run before answering.
- **The repairs are fed back to the model.** The repair pass runs after the last redraft, so
  nothing ever told the editor what had been changed for it; it made the same edit next turn and
  the pipeline undid it again. What was applied now rides on the assistant message every backend
  replays, and a rule says what it means: already done, build on it.

### Changed
- **Length is a target, not a quota.** Speech clips are never scaled; pictures are the only thing
  trimmed or held. A 40 s ask now lands at 43.9 s with every sentence whole.
- **The interview/b-roll balance is the editor's decision**, not a hidden 70 % cap
  (`script.speech_budget` defaults to 1.0 and is a house rule when lowered).
- **CLI agent brains continue their own conversation.** Every tool round re-sent the whole
  transcript as a fresh invocation, so each round cost more than the last — on a 96-video library
  agy spent over three minutes on round one and hit the timeout. Now only the new tool result is
  sent (`claude`/`agy --continue`, `codex exec resume --last`; opencode has no equivalent).
  `claude` and `codex` also skip permission prompts, as agy already did: a prompt nobody can answer
  hangs the turn until it times out.
- The rules ask for one story rather than a list of moments, for a speaker's face to be seen before
  cutting to what they describe, and for no small talk unless it is asked for.

### Fixed
- **A cut never stops in the middle of a sentence.** The fit that runs last can trim a speech clip
  by seconds, and the only repair after it reached 0.75 s — so clips ended 0.27 s to 2.65 s before
  the speaker finished, one of them mid-word, four of four in a single run. Every clip carrying a
  voice is now put back on a whole sentence after fitting: out to the end of the sentence when that
  is within `script.max_speech_extend_s`, back to where the previous one ended when it is further.
  The length gives way to the speaker.
- **A long search no longer kills the turn.** Every round re-sends the whole conversation, so a
  model that kept looking eventually could not fit another one: 199 tool calls, then a raw
  "exceeds the available context size" from the server with nothing drafted and twenty minutes of
  research thrown away. Three quarters of the window is kept for the conversation, the oldest tool
  results are forgotten to make room, and when there is nothing left to forget the loop stops
  looking and writes the script.
- **A model that would not stop generating hung the turn.** Server requests had no `max_tokens` and
  a 300 s ceiling, so 11 347 tokens of one draft ran until the socket timed out and the turn died
  as "error sending request". Both are now generous and configurable
  (`script.max_answer_tokens` 16384, `script.server_timeout_s` 1800): the cap exists so a runaway
  fails as itself, not to ration a long script from a slow model.
- **A bed could outlive the beat it played under.** Beds are laid before the cut is fitted, and
  fitting trims the pictures — one came back 19.9 s long under an 11.9 s beat, playing over the
  next beat and stacking two clips on A1 in the export.
- **A beat id is a handle, not a sentence.** The preview groups clips by it and titles are keyed on
  it, but one model pasted a whole transcript quote into it and another left every one empty. An
  unusable id is replaced with a slug of the beat's purpose, and duplicates are numbered.
- `config set` accepts `max_tool_rounds`, which had a field and documentation but no key — the
  research budget could only be changed by editing the file by hand.
- **Sentences cut mid-word.** Three interview clips in a row ended inside a word: the fitting pass
  scaled speech after it had been snapped to sentences, so the number won (203.92 instead of
  204.68, 75.55 instead of 77.00, 155.86 instead of 156.78).
- **The last word chopped.** Whisper's segments are contiguous in continuous speech, so the guard
  against running into the next sentence collapsed to the current segment's end and the clip
  stopped on the final sample. A clip may now run a little past the last word
  (`script.speech_overrun_s`, 0.35 s), and every clip is faded in and out at its join
  (`script.audio_fade_s`, 120 ms).
- **A word from the next sentence, audible.** That overrun reached into what came next, so a clip
  ending on "community" came out as "community and". The fade now starts at the last sentence end
  inside the clip: the decay survives, the next word is already at zero.
- **Stop did nothing to a script chat.** It set a flag only the index loop read between items, and
  a chat turn is one long item. The turn now checks between tool rounds and before each model
  call, and the local backend kills the helper process, where the generation actually runs. A
  stopped turn reports as cancelled, not failed.
- **Changing the audio settings appeared to do nothing** — proxies carry encoded audio, so a clip
  kept whatever levelling and fade it was first built with. They are keyed on it now.
- **The chat scrolled the whole application**; it scrolls itself.
- Tool rounds get the draft's token budget; at the old 2048 default a model that reasons first ran
  out mid-round and took the turn with it.
- A chat session can be removed from the Scripts panel (its scripts are kept).

## [0.1.7] — 2026-09-18

### Fixed
- **An interview-led cut came in at 70 % of its target.** Speech was held to a fixed share of the
  length (70 %) to leave room for the pictures — but a teaser built from what people say has no
  pictures to fill the rest, so a 60 s target produced 41.6 s, exactly the share. Speech now gives
  way only as far as the pictures actually present can cover; the same prompt and footage now
  lands at 60.7 s. The share was also hardcoded rather than read from `script.speech_budget`.

## [0.1.6] — 2026-09-18

The standalone script chat went from producing empty, silent scripts to complete, narrated,
interview-led cuts that land on the target length; the index now measures how steady the camera
is and the editor cuts around the shaky stretches; camera originals play in the app through a proxy.

### Added
- **Camera steadiness, measured at index time.** Each video is analysed from its own pictures — a
  patch grid with contrast selection, a robust similarity fit (translation, rotation, scale) with
  a median consensus so a person moving in the shot is not mistaken for the camera, and the camera
  path split by frequency at 30 fps. Two quantities per 4 s window: *tremor* (fast movement the
  hand adds) and *sway* (movement within a second that is undone — the slow back-and-forth of an
  unstabilised walking shot). Decoding runs on the GPU when it can (8× faster), CPU otherwise.
  Validated against ffmpeg's vid.stab on eleven clips (agreement within ~0.1) and against the
  editor's own eye on four labelled stretches.
- **Camera style per clip** — `static`, `tripod`, `stabilised` or `handheld` — read from the
  tremor floor and how much the camera moves on purpose. Shown in the Library's new **Camera**
  column and reported to the script editor by `list_videos`, `search_moments` and `get_video`.
- **Shaky stretches as timestamps**, not a verdict on the file: `get_video` lists `shaky_at`
  ranges, search hits carry a `shaky` flag, and the video panel draws a clickable shake bar under
  the player. A stretch is shaky when tremor or sway is over its line, and — for handheld footage
  — when it is twice the clip's own ordinary level, so a handheld clip keeps its usual stretches
  and loses only the worse ones.
- **The script editor never cuts from a shaky stretch.** A clip that lands on one is moved to the
  nearest steady stretch of the same shot, or dropped with the reason when the shot has none.
- **Thinking, configurable per capability.** The local model can reason before it answers; the
  grammar that constrains its JSON now binds only after `</think>`, and the thought is capped at
  half the output budget so it cannot starve the answer. On for script drafts, off for per-keyframe
  descriptions; `vision.think` / `chat_model.think` in config, CLI and the Models page.
- **`[script]` settings** — every number the script generator uses (clip lengths, target
  tolerances, narration pace, research budget, shake limits), settable with `ghostreel config set`
  and over MCP.
- **MCP**: `get_settings`, `set_settings` (same keys as the CLI) and `preview_script`, so a remote
  GhostReel can be configured, driven and checked from another machine.
- **codex** as a fourth coding-agent CLI backend (`codex exec --json`, images attached with `-i`).
- **Remove a chat session** from the Scripts panel (its scripts are kept).
- **Playback proxy** for camera originals: a 4K 10-bit 4:2:2 file sat black in the player for a
  long time; the app now builds a 720p copy on first open (NVENC when available) and keeps it.
  Keyframes, previews and exports still use the original.
- Rule 6a for the editor: when people talk on camera, build the story from what they say — cut to
  whole sentences on their own audio; voice-over is for the scenery between.

### Changed
- **Research budget**: a server or coding-agent CLI brain gets 60 tool rounds and full-size tool
  results (it was 8 rounds and 1500 characters, sized for a small local model); a local model gets
  10. `chat_model.max_tool_rounds` overrides either.
- **Grounding**: a clip the model never opened is no longer thrown away. If the range is inside the
  video and has indexed keyframes or speech, it is kept and reported as checked; only ranges with
  nothing indexed are dropped. With enough unopened picks a whole script used to collapse to
  "unable to assemble a script".
- **Fitting to target** is one pass that measures once: pictures give way first, speech only if
  that is not enough, and pictures are held longer (never past the voice-over) when the cut is
  short. The final pass before saving works to 2 % of the target. Three independent cutters used to
  compound, and a cut at 58.6 s once came out at 29.5 s.
- **Speech-aware audio**: `get_video` and `list_videos` report whether anyone speaks; a clip with
  no transcript in range cannot carry source audio, and a beat playing someone's own audio cannot
  also carry narration. The editor is told to prefer a mounted or stabilised take when two clips
  cover the same moment.
- The draft prompt lists the footage the model is allowed to cut; it is asked to come in ~10 %
  long, since trimming is safe and growing is not.
- Beats left silent get their narration written one at a time: a small model does that reliably,
  and reliably leaves `narration` empty when it is one field of a nested draft.
- Tool result and round limits, and every other pipeline constant, moved to `[script]`.

### Fixed
- **Models page: "CLI agent" could not be selected** — the desktop app rejected `backend = "cli"`
  outright, so the control snapped back. Selecting it also fills in the default tool; the picker
  used to show `claude` while the config held an empty string.
- **CLI agents looked "not installed" in the app.** A desktop launch inherits the session's PATH
  (`/usr/local/bin:/usr/bin`), not the shell's; `claude`, `agy`, `opencode` and `codex` in
  `~/.local/bin` or `~/.bun/bin` are now found.
- **Standalone script chat lost its draft to a 2048-token cap.** Every local completion asked for
  at most 2048 tokens; a real script runs past that, the helper stopped mid-object, and the one
  place that swallowed the parse error reported "unable to assemble a script" with no reason. The
  draft now gets a budget out of the configured context, and a cut-off answer is an error that says
  so.
- **Silent scripts.** `get_video` never said whether anyone speaks, so the model marked scenery
  `source` and then obeyed the rule that silences narration where people talk. Interviews were
  invisible in `list_videos`, so talking heads were cut as b-roll and muted mid-sentence under a
  voice-over.
- `codex` treats a non-TTY stdin as extra prompt input and hangs; every CLI tool now runs with stdin
  closed.
- Tests: fake CLI scripts used `echo` with `\n`, which dash (Ubuntu's `/bin/sh`) expands; CI failed
  while the Arch dev box passed. They use `printf` now.
- Repo references point at `highercomve/GhostReel` after the transfer; the updater endpoint follows
  the redirect either way.

### Notes for the standalone profile
- Measured on the reference footage (96 clips, 88.7 GB): a full index peaks at 6.5 GB of VRAM and a
  script turn at 6.8 GB with `chat_model.ctx_tokens = 16384`, `kv_cache = q4_0` — inside the 8 GB
  budget.
- A rebuilt `ghostreel-llm` must be built with `scripts/build-helpers.sh` (CUDA); a plain
  `cargo build` silently produces a CPU-only helper.
- Moving the repository leaves stale absolute paths in `target/`; clean `target/*/build/` for
  `tauri` and `llama-cpp-sys-2` after a move.

## [0.1.5] and earlier

See the git history: `git log v0.1.5`.

[Unreleased]: https://github.com/highercomve/GhostReel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/highercomve/GhostReel/compare/v0.1.7...v0.2.0
[0.1.7]: https://github.com/highercomve/GhostReel/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/highercomve/GhostReel/compare/v0.1.5...v0.1.6
