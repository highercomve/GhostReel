# Multi-track audio — implementation brief

Read `/home/projects/personal/ghostreel/.agents/editor-features/BRIEF.md` first for the project
context and general rules (own files only, no git writes, don't restart the app, theme variables,
`npx tsc --noEmit` / `npm run build`; for Rust: `cargo test -p ghostreel-core --lib` must stay green —
the baseline is 306 passed, 0 failed).

## Why

Pro cameras record several audio tracks per file. This project's Sony MXF files carry 4 mono
tracks: A1 (lav) and A2 (camera mic) have sound, A3/A4 are silent. GhostReel today picks one track
per video (`videos.audio_track`, `ResolvedMedia.audio_track`) and hides the rest; the clip only has
`audio: "source" | "mute"`. Editors need every audio track as its own layer in the timeline and to
choose, per clip, which tracks play — like Premiere/Resolve.

## Data model (both agents rely on this exactly)

`ScriptClip` and `AudioBed` (Rust `crates/ghostreel-core/src/script.rs`, TS `src/api.ts`) gain:

```rust
/// Source audio tracks to play (0-based among the file's audio streams). None = the file's
/// automatic track (today's behaviour). Empty = silent.
#[serde(default, skip_serializing_if = "Option::is_none")]
pub tracks: Option<Vec<u32>>,
```
```ts
tracks?: number[];
```

`audio: "mute"` still silences the clip entirely regardless of `tracks`. A beat with a bed still
mutes its pictures (unchanged); the bed plays `bed.tracks` (or its automatic track).

## New commands (Tauri `src-tauri/src/lib.rs` + web UI dispatch `src-tauri/src/webui.rs`, same names/args)

1. `video_audio_tracks { videoId }` → `AudioTrackInfo[]`:
   `{ index: number /* 0-based among audio streams */, channels: number, codec: string,
      label: string | null /* stream title/handler if any */, auto: boolean /* = the track
      GhostReel picks today */ }`. ffprobe on the file (header only, fast).
2. `audio_levels { videoId, inS, outS }` → `number[]`: mean volume in dB per audio track over that
   source range (ffmpeg `volumedetect` per mapped stream; one ffmpeg run with several outputs or
   `astats`), `-91` (or lower) meaning silence. Cap the measured range at 20 s (sample the middle)
   so it stays quick.
3. `audio_track_proxy { videoId, track }` → `string` path: the whole track extracted once to a small
   playable file (mono or stereo AAC in `.m4a`, 96–128 kbps) cached under the data dir
   (e.g. `proxies/audio/<content_hash>_a<track>.m4a`), reused on later calls. Used by the live
   viewer to play chosen tracks in sync with the picture.

## Agent A — backend (owns `crates/ghostreel-core/src/script.rs`, `preview.rs`, `otio.rs` (only if
needed), a new `crates/ghostreel-core/src/audio_tracks.rs` if you like, `src-tauri/src/lib.rs`,
`src-tauri/src/webui.rs`; note `crates/ghostreel-core/src/otio.rs` and `cliagent.rs` were just
edited by the lead — keep those edits)

- Add the `tracks` field (Rust) with serde defaults so old scripts load unchanged; update any
  struct literals/tests that construct `ScriptClip`/`AudioBed`.
- Renderer (`preview.rs`): when a segment's clip has `tracks: Some(list)` and isn't muted, map each
  listed `0:a:{i}` and mix them (`amix=inputs=N:normalize=0`, then the existing fades/loudnorm
  chain); `Some([])` = silent (use the existing null-audio path). Same for beds with `tracks`.
  Include the track selection in the segment proxy cache key (`proxy_file_name_with_audio`) so a
  changed selection re-renders. Ignore (and skip) indexes beyond the file's audio stream count
  rather than failing the render.
- The three commands above, registered for Tauri and the web UI.
- Unit tests: `tracks` round-trips through serde (absent stays absent), and the ffmpeg argument /
  filter building for multi-track segments if that logic is testable as a pure function.
- Don't change the chat/AI pipeline beyond what compiles; the AI never sets `tracks`.

## Agent B — frontend (owns `src/api.ts` (only the additions), `src/ScriptTimeline.tsx`,
`src/editor/timeline.css`, `src/editor/LivePlayer.tsx`, `src/editor/player.css`; the lead owns
`src/ScriptEditor.tsx` and will wire props you ask for — put the wiring you need in your report, or
make minimal, clearly-scoped edits to `ScriptEditor.tsx` limited to passing new props and the
inspector's audio section)

- `api.ts`: `tracks?: number[]` on `ScriptClip` and `AudioBed`; `AudioTrackInfo` type;
  `videoAudioTracks(videoId)`, `audioLevels(videoId, inS, outS)`, `audioTrackProxy(videoId, track)`
  (use the existing `call<T>(name, args)` helper like the other commands).
- **Timeline audio lanes**: replace the single "Sound" lane with one lane per audio track,
  **A1…An**, n = the most audio tracks among the videos in the cut (fetch `videoAudioTracks` per
  video once, cache it). Each clip (and each bed) draws a block on every lane its file has: enabled
  tracks look like today's sound blocks; disabled ones are dim/striped; a lane the file doesn't have
  shows nothing. A bed's tracks draw on the same lanes across its beat (beds replace the pictures'
  own audio in that beat, as today). Show per-block level from `audioLevels` (lazy, cached) as a
  small dB label or a simple level bar so silent tracks are obvious ("silent").
  Lane headers: name (A1…), and **M** (mute) / **S** (solo) toggles. Lane M/S is a monitoring
  state for the live viewer (not saved). Clicking a block selects it (key format of your choice
  that keeps `p-…`/`t-…` working; tell the lead); a selected audio block + `m` key toggles that
  track for that clip (edit → the lead's handler, see below). Keep lanes compact (≈24–28 px each)
  and let the timeline scroll vertically when there are many.
- Provide a new optional prop on `ScriptTimeline`:
  `onToggleTrack?: (beat: number, clip: number | null /* null = bed */, track: number) => void`
  and `audioTracks?: Record<number, AudioTrackInfo[]>` if you'd rather the lead fetch them
  (your choice; say which).
- **Inspector audio section** (in `ScriptEditor.tsx`, clip inspector, under the clip fields):
  "Audio tracks": one row per track of the clip's file — checkbox, `A1`, label/channels, level
  (dB or "silent"), and a small play/preview button optional. Checking/unchecking writes
  `tracks` (when the user first touches it, start from the auto track). Also for a selected bed.
- **Live viewer**: when a clip (or bed) has explicit `tracks`, mute the picture element and play
  each chosen track from `audioTrackProxy` in hidden `<audio>` elements synced to the cut clock
  (same drift-correction approach as the bed); honour lane M/S monitoring state (pass it in as a
  prop, e.g. `trackMonitor: { muted: number[]; solo: number[] }`, from the editor). `tracks`
  undefined → today's behaviour (the picture element's own audio).
- Verify in the browser at http://127.0.0.1:4317 (Scripts → Social Reel (30s)); don't save.
