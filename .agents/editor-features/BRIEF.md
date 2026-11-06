# Editor features — implementation brief

GhostReel (Tauri v2 + React 19 + TS + Vite frontend in `src/`, Rust core in `crates/`) turns indexed
footage into a "script": a cut made of beats, each beat a list of clips (`video_id`, `in_s`, `out_s`,
`audio`), optional `on_screen_text`, `narration`, and an optional audio `bed` (a J-cut: one source's
audio across the beat while its pictures change). Types: `src/api.ts` (`Script`, `Beat`, `ScriptClip`,
`AudioBed`, `VideoRow`, `FrameRow`).

The Scripts tab was just rebuilt as an editor workspace (`src/ScriptEditor.tsx` renders a grid):
left panel (Cuts / Beats / Media), center viewer (`src/PreviewPlayer.tsx`), right inspector, bottom
timeline (`src/ScriptTimeline.tsx`). We now add real video-editor functionality, taking *ideas* from
Remotion Studio (live preview without rendering, transport controls, zoomable timeline with
thumbnails, keyboard-driven editing). **Do not copy or depend on Remotion code or packages** (license);
write our own.

## Rules

- Own only the files listed for you. You may create new files under `src/editor/`.
- Put your CSS in your own new stylesheet (`src/editor/<name>.css`) imported from your component; do
  not edit `src/styles.css` except a tiny targeted edit if truly unavoidable (say so in your report).
  Use the existing theme variables: `--bg --panel --border --text --muted --accent --accent-2 --good
  --bad --warn` (light and dark themes both exist; never hard-code text colors on themed surfaces).
- No git writes. Do not start/stop/restart the app; it's already running (`npx tauri dev`, web UI on
  http://127.0.0.1:4317 serving `dist/`).
- Verify with `npx tsc --noEmit` (must be clean) and `npm run build`. You may check the UI with the
  Playwright MCP tools against http://127.0.0.1:4317 after `npm run build` (Scripts tab → pick a cut
  in the left panel's Cuts tab). Don't create, delete, or save scripts while checking.
- Keep comments short and about *why*. Match the existing code style.
- Report: what you built, files touched, prop/interface changes, anything left undone.

## Shared interfaces (the lead wires these in `ScriptEditor.tsx`; implement your side exactly)

### Agent 1 — Live viewer (owns `src/PreviewPlayer.tsx`, new `src/editor/LivePlayer.tsx`, `src/editor/player.css`)

`PreviewPlayer` gains two props: `script: Script` and `videos: Record<number, VideoRow>`. It keeps the
existing contract: `onTime(seconds)` reports the playhead on the *cut's* clock, and
`onControls({ seek(s), toggle() })` hands controls to the timeline (seek must not start playback).

Build a **Live** mode (default) next to the existing **Rendered** mode (the rendered `preview.mp4`,
render/export buttons — keep those working):

- Plays the current `script` prop live (including unsaved edits) by laying the clips end to end:
  beat order, then clip order; cut time = running sum of `out_s - in_s`.
- Source per video: `videoPlayback(videoId)` → `{ path }` → `mediaUrl(path)` (both in `src/api.ts`).
  The first call may build a proxy and take a while: show a "Preparing <file>…" state, using
  `videoPlaybackProgress(videoId)` if useful (see `src/VideoPanel.tsx` for how it's used). Cache URLs.
- Gapless-ish playback: two `<video>` elements (A/B); preload the next clip at its `in_s` while the
  current plays; swap at the boundary. Seeking to any cut time picks the right clip and offset.
- `audio: "mute"` clips play muted. If a beat has a `bed`, mute the pictures in that beat and play the
  bed's source audio (a hidden media element at `bed.in_s + offset`) across the beat. Best effort.
- Overlay `on_screen_text` as a title (lower third style) during its beat, and optionally narration as
  a subtitle (toggle).
- Transport (Remotion-Studio-like): start, previous/next clip boundary, frame step back/forward
  (1/30 s, or `script.fps`), play/pause, J/K/L shuttle (J reverse step/slower, K pause, L play/faster),
  playback rate menu (0.25×, 0.5×, 1×, 1.5×, 2×), loop toggle, fullscreen of the stage, timecode
  `m:ss.ff` (frames) and total. Keyboard shortcuts while the viewer has focus: Space, J/K/L, ←/→ frame
  step, Shift+←/→ 1 s, Home/End, F fullscreen.
- Header: segmented toggle **Live | Rendered**; keep the Export menu. When the script is empty, show a
  helpful empty state.
- Props/state for `isAiWorking`, `isDirty`, `onSaveBeforeAction` keep their current meaning for
  render/export.

### Agent 2 — Timeline (owns `src/ScriptTimeline.tsx`, new `src/editor/timeline.css`)

Keep all existing props and keyboard behavior (←/→ trim, Shift, Alt, Ctrl, i/o, s snap, z zoom-to-beat,
space, Escape). Add these props (the lead implements them):

```ts
frames?: Record<number, FrameRow[]>;            // keyframes per video, for filmstrip thumbnails
onMoveClip?: (from: { beat: number; clip: number }, to: { beat: number; clip: number }) => void;
                                                // drag a picture clip; `to.clip` = insert-before index in `to.beat`
onSplit?: (beat: number, clip: number, atSourceSeconds: number) => void;  // blade at the playhead
onDelete?: (beat: number, clip: number | null) => void;  // delete selected clip (clip null = the beat's bed)
```

Build:

- **Zoom & scroll** like a real NLE: a zoom slider + "fit" button in a small timeline toolbar,
  Ctrl/⌘+wheel zooms around the pointer, horizontal scroll (wheel/shift+wheel, scrollbar) when zoomed,
  `=`/`-` keys zoom, `\` fits. The ruler adapts its tick step and shows `m:ss` labels; keep `z`.
- The playhead stays visible: auto-scroll to keep it in view while playing; a draggable playhead
  handle on the ruler (click/drag on the ruler scrubs via `onSeek`).
- **Filmstrip thumbnails** in picture clips (nearest keyframe(s) from `frames[video_id]` by source
  time; render with `MediaImage` from `src/MediaImage.tsx`), with the filename/time label over them.
  Wide clips show several thumbs.
- **Drag to reorder** picture clips: drag a clip body, show an insertion marker, drop → `onMoveClip`.
  Edge handles keep trimming as today.
- **Blade**: `c` key or Ctrl/⌘+K splits the selected clip (or the clip under the playhead if none) at
  the playhead → `onSplit` with the source-time position.
- **Delete/Backspace** on a selected clip or bed → `onDelete`.
- A compact toolbar row above the ruler: zoom −/slider/+, Fit, and a `?` button that toggles the key
  hints (keep the existing hints, collapsed by default to save height).
- Lanes slightly taller for thumbnails (picture ~52px). Beat boundaries drawn as subtle vertical
  separators with the beat id in the ruler area.
