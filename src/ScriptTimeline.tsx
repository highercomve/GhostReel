import { formatDuration } from "./timeFormat";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { AudioTrackInfo, Cue, FrameRow, Script, VideoRow } from "./api";
import { AudioLines, ChevronDown, EyeOff, Film, MessageSquareText, Type, Volume2 } from "lucide-react";
import { cueKey } from "./editor/textOverlay";
import MediaImage from "./MediaImage";
import { PICK_MIME, type Pick } from "./editor/MediaBin";
import {
  audible,
  dbLabel,
  effectiveTracks,
  levelFraction,
  SILENT_DB,
  trackCount,
  trackDescription,
  trackName,
  useAudioLevels,
  useAudioTracks,
  type LevelRequest,
  type TrackMonitor,
} from "./editor/audioTracks";
import "./editor/timeline.css";

/** One block on a lane, already positioned in seconds on the cut's timeline. */
type Block = {
  key: string;
  kind: "title" | "picture" | "bed" | "cue";
  beat: number;
  clip: number | null;
  /** Position in the finished cut. */
  start: number;
  end: number;
  /** Where this block reads from inside its source file. */
  sourceIn: number;
  videoId: number | null;
  label: string;
  detail: string;
  muted: boolean;
  inferred: boolean;
  /** Audio blocks: which of the file's tracks (and so which A-lane) this is. */
  track?: number;
  /** Audio blocks: the file has this track but the clip doesn't play it. */
  off?: boolean;
  /** Titles and caption cues: how far they may move on the cut's clock. */
  lo?: number;
  hi?: number;
  /** Caption blocks: the cue. */
  cue?: Cue;
};

type ClipRef = { beat: number; clip: number };

/** A pointer gesture in progress; window listeners read it so a drag survives leaving the block. */
type Gesture =
  | { type: "trim"; beat: number; clip: number | null; edge: "in" | "out"; x: number }
  | { type: "move"; from: ClipRef; key: string; x0: number; y0: number; active: boolean }
  /** A title or caption moved or retimed by its edge, within its beat (or clip). */
  | { type: "item"; key: string; mode: "move" | "in" | "out"; x0: number; s0: number; e0: number; lo: number; hi: number; active: boolean }
  | { type: "scrub" };

/** The shortest a title or caption may be made by dragging. */
const MIN_ITEM = 0.2;

/** Air on both sides of the content so the playhead handle at 0 and at the end isn't clipped. */
const GUTTER = 8;
/** The closest zoom: about 20 px per frame at 30 fps, enough to place an edge on a frame. */
const MAX_PPS = 600;
/** Roughly how wide one filmstrip thumbnail is; the clip's width is shared out evenly. */
const THUMB_W = 84;
/** Picture blocks narrower than this move only once the pointer has really travelled. */
const DRAG_THRESHOLD = 4;
/** A blade this close to an edge would leave a sliver nobody can see. */
const MIN_SPLIT = 0.05;

/** Lane heights at rest; the picture and sound lanes grow into a taller dock. */

/** Ruler labels: whole seconds unless the ticks are closer than a second. */
function rulerLabel(t: number, step: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  if (step >= 1) return `${m}:${String(Math.round(s)).padStart(2, "0")}`;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function name(videos: Record<number, VideoRow>, id: number): string {
  const v = videos[id];
  if (!v) return `#${id}`;
  const base = (v.path ?? "").split(/[\\/]/).pop();
  return base || `#${id}`;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

/** The keyframe closest to a source time; frames must be sorted by `t_s`. */
function nearestFrame(frames: FrameRow[], t: number): FrameRow | null {
  if (frames.length === 0) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t_s < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(frames[lo - 1].t_s - t) <= Math.abs(frames[lo].t_s - t)) return frames[lo - 1];
  return frames[lo];
}

/** Lay the script out in seconds: where every picture, title and bed sits in the finished cut. */
function layout(
  script: Script,
  videos: Record<number, VideoRow>,
  infos: Record<number, AudioTrackInfo[] | null>,
): {
  blocks: Block[];
  total: number;
  beatStarts: { beat: number; id: string; start: number; end: number }[];
  laneCount: number;
} {
  const blocks: Block[] = [];
  const fmt = formatDuration;
  let laneCount = 1;
  const beatStarts: { beat: number; id: string; start: number; end: number }[] = [];
  let at = 0;

  script.beats.forEach((beat, b) => {
    const beatStart = at;
    beat.clips.forEach((c, i) => {
      const dur = Math.max(0, c.out_s - c.in_s);
      // Lead with what the shot is for; the file name and source range are the detail.
      const why = c.why?.trim();
      const file = name(videos, c.video_id);
      blocks.push({
        key: `p-${b}-${i}`,
        kind: "picture",
        beat: b,
        clip: i,
        start: at,
        end: at + dur,
        sourceIn: c.in_s,
        videoId: c.video_id,
        label: why || file,
        detail: `${why ? `${file} · ` : ""}${fmt(c.in_s)}–${fmt(c.out_s)}`,
        muted: c.audio === "mute" || beat.bed != null,
        inferred: false,
      });
      at += dur;
    });
    const beatEnd = at;
    beatStarts.push({ beat: b, id: beat.id, start: beatStart, end: beatEnd });

    if (beat.on_screen_text?.trim() && beatEnd > beatStart) {
      const dur = beatEnd - beatStart;
      const rel0 = clamp(beat.title_layout?.start_s ?? 0, 0, Math.max(0, dur - 0.1));
      const rel1 = clamp(beat.title_layout?.end_s ?? dur, rel0 + 0.1, Math.max(dur, rel0 + 0.1));
      blocks.push({
        key: `t-${b}`,
        kind: "title",
        beat: b,
        clip: null,
        start: beatStart + rel0,
        end: Math.min(beatEnd, beatStart + rel1),
        lo: beatStart,
        hi: beatEnd,
        sourceIn: 0,
        videoId: null,
        label: beat.on_screen_text,
        detail: "",
        muted: false,
        inferred: false,
      });
    }

    if (beat.bed) {
      const bed = beat.bed;
      const dur = Math.max(0, bed.out_s - bed.in_s);
      const info = infos[bed.video_id];
      const on = new Set(effectiveTracks(bed.tracks, info));
      // A bed always keeps one block, even from a file without sound, so it can be selected and removed.
      const n = Math.max(1, trackCount(info));
      laneCount = Math.max(laneCount, n);
      for (let t = 0; t < n; t++) {
        const off = !on.has(t);
        blocks.push({
          key: `b-${b}-${t}`,
          kind: "bed",
          beat: b,
          clip: null,
          start: beatStart,
          end: beatStart + Math.min(dur, beatEnd - beatStart),
          sourceIn: bed.in_s,
          videoId: bed.video_id,
          label: name(videos, bed.video_id),
          detail: off ? "bed · track off" : `bed · ${fmt(bed.in_s)}–${fmt(bed.out_s)}`,
          muted: false,
          inferred: bed.inferred === true,
          track: t,
          off,
        });
      }
    } else {
      // Where a beat has no bed, its speaking clips are what you hear: one block per track of the file.
      let before = 0;
      beat.clips.forEach((c, i) => {
        const silent = c.audio === "mute";
        const dur = Math.max(0, c.out_s - c.in_s);
        const info = infos[c.video_id];
        const on = new Set(effectiveTracks(c.tracks, info));
        const n = trackCount(info);
        laneCount = Math.max(laneCount, n);
        for (let t = 0; t < n; t++) {
          const off = !silent && !on.has(t);
          blocks.push({
            key: `a-${b}-${i}-${t}`,
            kind: "bed",
            beat: b,
            clip: i,
            start: beatStart + before,
            end: beatStart + before + dur,
            sourceIn: c.in_s,
            videoId: c.video_id,
            // A muted clip still gets a block, so a silent stretch reads as a choice you can undo
            // rather than as a gap where the sound went missing.
            label: name(videos, c.video_id),
            detail: silent ? "clip sound off" : off ? "track off · m to use" : "own audio",
            muted: silent,
            inferred: false,
            track: t,
            off,
          });
        }
        before += dur;
      });
    }
  });

  return { blocks, total: at, beatStarts, laneCount };
}

/** "p-1-2", "a-1-2-0" → the clip they belong to; "b-1…" → the beat's bed. */
function ownerOf(key: string | null): string | null {
  if (!key) return null;
  const [kind, beat, clip] = key.split("-");
  if (kind === "p" || kind === "a") return `c${beat}-${clip}`;
  if (kind === "b") return `bed${beat}`;
  return null;
}
const blockOwner = (b: Block) =>
  b.kind === "title" || b.kind === "cue" ? null : b.clip != null ? `c${b.beat}-${b.clip}` : `bed${b.beat}`;

/** Caption cues as blocks on their own lane. */
function cueBlocks(cues: Cue[] | undefined): Block[] {
  return (cues ?? []).map((cue, i) => ({
    key: cueKey(cue, i),
    kind: "cue" as const,
    beat: -1,
    clip: null,
    start: cue.start_s,
    end: cue.end_s,
    sourceIn: cue.src_start_s ?? 0,
    videoId: cue.video_id ?? null,
    label: cue.text.replace(/\n/g, " "),
    detail: cue.hidden ? "hidden" : cue.edited ? "edited" : "",
    muted: !!cue.hidden,
    inferred: false,
    lo: cue.min_s ?? 0,
    hi: cue.max_s ?? cue.end_s,
    cue,
  }));
}

type Props = {
  script: Script;
  videos: Record<number, VideoRow>;
  selected: string | null;
  /** Where the preview player is, in timeline seconds. */
  playhead: number;
  onSelect: (key: string | null, beat: number, clip: number | null) => void;
  onSeek: (seconds: number) => void;
  onTogglePlay: () => void;
  onTrim: (beat: number, clip: number | null, edge: "in" | "out", deltaSeconds: number) => void;
  /** Put an edge exactly here, in the source file's own seconds. */
  onSetEdge: (beat: number, clip: number | null, edge: "in" | "out", sourceSeconds: number) => void;
  /** Pull both edges onto the nearest sentence boundaries. */
  onSnap: (beat: number, clip: number | null) => void;
  /** Keyframes per video, for the filmstrip in picture clips. */
  frames?: Record<number, FrameRow[]>;
  /** A picture clip dragged elsewhere; `to.clip` is the insert-before index in `to.beat`. */
  onMoveClip?: (from: ClipRef, to: ClipRef) => void;
  onInsertPick?: (pick: Pick, to: ClipRef) => void;
  /** Blade: cut this clip in two at a point in its source file's own seconds. */
  onSplit?: (beat: number, clip: number, atSourceSeconds: number) => void;
  /** Remove a clip, or the beat's bed when `clip` is null. */
  onDelete?: (beat: number, clip: number | null) => void;
  /** Turn one source audio track on or off for a clip (or the beat's bed when `clip` is null). */
  onToggleTrack?: (beat: number, clip: number | null, track: number) => void;
  /** Lane mute/solo for the live viewer. Uncontrolled (kept here) when absent. */
  monitor?: TrackMonitor;
  onMonitor?: (m: TrackMonitor) => void;
  /** Turn a muted clip's own sound back on. */
  onUnmute?: (beat: number, clip: number) => void;
  /** Speech caption cues (hidden ones included), for the Captions lane. */
  captions?: Cue[];
  /** A title moved or retimed: seconds after its beat starts. */
  onTitleTime?: (beat: number, startRel: number, endRel: number) => void;
  /** A caption moved or retimed, on the cut's clock. */
  onCueTime?: (cue: Cue, start: number, end: number) => void;
  onCueText?: (cue: Cue, text: string) => void;
  onCueHide?: (cue: Cue, hidden: boolean) => void;
};

/**
 * The cut on three lanes, the way an NLE shows it: what is written over the picture, the picture
 * itself, and the sound. A list of beats hides the one thing that matters most here — that a voice
 * runs on while the pictures change — because a bed has no place in a list.
 *
 * Editing is by keyboard as much as by mouse, because the decisions here are fractions of a
 * second: where a sentence ends is a 0.1 s judgement and a drag cannot hit it. The playhead is the
 * preview player's, so setting an edge from what you just heard is one key.
 */
export default function ScriptTimeline({
  script,
  videos,
  selected,
  playhead,
  onSelect,
  onSeek,
  onTogglePlay,
  onTrim,
  onSetEdge,
  onSnap,
  frames,
  onMoveClip,
  onInsertPick,
  onSplit,
  onDelete,
  onToggleTrack,
  monitor: monitorProp,
  onMonitor,
  onUnmute,
  captions,
  onTitleTime,
  onCueTime,
  onCueText,
  onCueHide,
}: Props) {
  const videoIds = useMemo(
    () => script.beats.flatMap((b) => [...b.clips.map((c) => c.video_id), ...(b.bed ? [b.bed.video_id] : [])]),
    [script],
  );
  const trackInfo = useAudioTracks(videoIds);
  const { blocks: scriptBlocks, total, beatStarts, laneCount } = useMemo(
    () => layout(script, videos, trackInfo),
    [script, videos, trackInfo],
  );
  const blocks = useMemo(() => [...scriptBlocks, ...cueBlocks(captions)], [scriptBlocks, captions]);
  /** A title or caption being dragged: drawn here until it is dropped. */
  const [itemDrag, setItemDrag] = useState<{ key: string; start: number; end: number } | null>(null);
  const itemDragRef = useRef(itemDrag);
  const [editingCue, setEditingCue] = useState<string | null>(null);
  const [ownMonitor, setOwnMonitor] = useState<TrackMonitor>({ muted: [], solo: [] });
  const monitor = monitorProp ?? ownMonitor;
  const setMonitor = (m: TrackMonitor) => {
    if (!monitorProp) setOwnMonitor(m);
    onMonitor?.(m);
  };
  const flipMonitor = (which: "muted" | "solo", track: number) => {
    const list = monitor[which];
    const next = list.includes(track) ? list.filter((t) => t !== track) : [...list, track].sort((a, b) => a - b);
    setMonitor({ ...monitor, [which]: next });
  };
  const wrapRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const namesRef = useRef<HTMLDivElement>(null);
  const [binDrop, setBinDrop] = useState<{ to: ClipRef; x: number } | null>(null);
  const [expandSilent, setExpandSilent] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const [vw, setVw] = useState(900);
  const [scrollX, setScrollX] = useState(0);
  const [tracksOverflow, setTracksOverflow] = useState(false);
  const [tracksAtBottom, setTracksAtBottom] = useState(false);
  const [tracksBelow, setTracksBelow] = useState(0);
  const [laneDensity, setLaneDensity] = useState<"compact" | "normal" | "tall">(() => {
    try {
      const saved = localStorage.getItem("ghostreel.timelineLaneHeight");
      return saved === "compact" || saved === "tall" ? saved : "normal";
    } catch {
      return "normal";
    }
  });
  // Zoom is relative to "fit", so resizing the dock keeps a fitted cut fitted.
  const [zoom, setZoom] = useState(1);
  const zoomRef = useRef(1);
  const pendingScroll = useRef<number | null>(null);
  const [beatZoom, setBeatZoom] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const gesture = useRef<Gesture | null>(null);
  const suppressClick = useRef(false);
  const [moving, setMoving] = useState<{ key: string; to: ClipRef; x: number } | null>(null);
  const movingRef = useRef(moving);

  const sel = blocks.find((b) => b.key === selected) ?? null;
  const pictures = useMemo(() => blocks.filter((b) => b.kind === "picture"), [blocks]);

  const sortedFrames = useMemo(() => {
    const out: Record<number, FrameRow[]> = {};
    for (const [id, list] of Object.entries(frames ?? {})) {
      out[Number(id)] = [...list].sort((a, b) => a.t_s - b.t_s);
    }
    return out;
  }, [frames]);

  const target = script.target_duration_s && script.target_duration_s > 0 ? script.target_duration_s : 0;
  // A target a little past the end is drawn to scale so the room left reads at a glance; a far one
  // would squash the cut into a corner, so it only gets a note in the toolbar.
  const targetOnScale = target > 0 && target <= total * 1.25;
  const extent = targetOnScale ? Math.max(total, target) : total;
  const span = total > 0 ? Math.max(extent, 0.001) : Math.max(target, 30);
  const fitPps = Math.max(vw - GUTTER * 2, 40) / span;
  const maxZoom = Math.max(1, MAX_PPS / fitPps);
  const z = clamp(zoom, 1, maxZoom);
  const pps = fitPps * z;
  const contentW = GUTTER * 2 + span * pps;
  const xOf = useCallback((t: number) => GUTTER + t * pps, [pps]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setVw(el.clientWidth));
    ro.observe(el);
    setVw(el.clientWidth);
    return () => ro.disconnect();
  }, [total > 0]);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    const content = contentRef.current;
    if (!el || !content) return;
    const measure = () => {
      // Measure against the space before the cue is inserted; otherwise the cue can
      // keep itself visible after a resize even when all collapsed lanes would fit.
      const cue = wrapRef.current?.querySelector<HTMLElement>(".tl-track-continuation");
      const cueSpace = cue ? cue.offsetHeight + (parseFloat(getComputedStyle(wrapRef.current!).rowGap) || 6) : 0;
      setTracksOverflow(el.scrollHeight > el.clientHeight + cueSpace + 1);
      setTracksAtBottom(el.scrollTop + el.clientHeight >= el.scrollHeight - 1);
      if (namesRef.current) namesRef.current.scrollTop = el.scrollTop;
      const bottom = el.scrollTop + el.clientHeight;
      const laneEls = content.querySelectorAll<HTMLElement>(".tl-lane");
      let count = 0;
      laneEls.forEach((l) => {
        if (l.offsetTop + l.offsetHeight > bottom + 2) count++;
      });
      setTracksBelow(count);
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    ro.observe(content);
    measure();
    return () => ro.disconnect();
  }, []);

  // A zoom asks for a scroll position that only exists once the wider content is laid out.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || pendingScroll.current == null) return;
    el.scrollLeft = Math.max(0, pendingScroll.current);
    pendingScroll.current = null;
    setScrollX(el.scrollLeft);
  }, [pps]);

  /** Zoom so that `anchorTime` stays under the same pixel of the viewport. */
  const zoomTo = useCallback(
    (next: number, anchorTime?: number, anchorPx?: number) => {
      const el = scrollRef.current;
      if (!el) return;
      const target = clamp(next, 1, maxZoom);
      const curZoom = clamp(zoomRef.current, 1, maxZoom);
      if (Math.abs(target - curZoom) < 1e-6) return;
      const curPps = fitPps * curZoom;
      const curScroll = pendingScroll.current ?? el.scrollLeft;
      const px = anchorPx ?? el.clientWidth / 2;
      const t = anchorTime ?? (curScroll + px - GUTTER) / curPps;
      pendingScroll.current = GUTTER + t * fitPps * target - px;
      zoomRef.current = target;
      setZoom(target);
      setBeatZoom(false);
    },
    [fitPps, maxZoom],
  );

  /** Zoom around the playhead when it is on screen, otherwise around the middle of the view. */
  const zoomStep = useCallback(
    (factor: number) => {
      const el = scrollRef.current;
      if (!el) return;
      const px = xOf(playhead) - el.scrollLeft;
      const onScreen = px >= 0 && px <= el.clientWidth;
      zoomTo(zoomRef.current * factor, onScreen ? playhead : undefined, onScreen ? px : undefined);
    },
    [playhead, xOf, zoomTo],
  );

  const fit = useCallback(() => {
    pendingScroll.current = 0;
    zoomRef.current = 1;
    setZoom(1);
    setBeatZoom(false);
    if (scrollRef.current) scrollRef.current.scrollLeft = 0;
  }, []);

  // Shows the selected beat alone, with a little air around it.
  const zoomToBeat = useCallback(() => {
    if (!sel) return;
    const beatBlocks = blocks.filter((b) => b.beat === sel.beat);
    const start = Math.min(...beatBlocks.map((b) => b.start));
    const end = Math.max(...beatBlocks.map((b) => b.end));
    const pad = Math.max((end - start) * 0.08, 0.3);
    const from = Math.max(0, start - pad);
    const to = Math.min(total, end + pad);
    const target = clamp((vw - GUTTER * 2) / Math.max(to - from, 0.001) / fitPps, 1, maxZoom);
    pendingScroll.current = from * fitPps * target;
    zoomRef.current = target;
    setZoom(target);
    setBeatZoom(true);
  }, [sel, blocks, total, vw, fitPps, maxZoom]);

  // Levels for the audio blocks on screen (and half a screen either side), measured lazily.
  const levelRequests = useMemo(() => {
    const from = (scrollX - GUTTER - vw / 2) / pps;
    const to = (scrollX + vw - GUTTER + vw / 2) / pps;
    const out: LevelRequest[] = [];
    for (const b of blocks) {
      if (b.track == null || b.videoId == null || b.end < from || b.start > to) continue;
      out.push({ videoId: b.videoId, inS: b.sourceIn, outS: b.sourceIn + (b.end - b.start) });
    }
    return out;
  }, [blocks, scrollX, vw, pps]);
  const levelOf = useAudioLevels(levelRequests);

  // The latest values for listeners registered once.
  const live = useRef({ pps, total, zoomTo, onSeek, onTrim, onMoveClip, pictures, blocks, onTitleTime, onCueTime });
  live.current = { pps, total, zoomTo, onSeek, onTrim, onMoveClip, pictures, blocks, onTitleTime, onCueTime };

  const timeAt = useCallback((clientX: number) => {
    const rect = contentRef.current?.getBoundingClientRect();
    if (!rect) return 0;
    return clamp((clientX - rect.left - GUTTER) / live.current.pps, 0, live.current.total);
  }, []);

  // Wheel needs a non-passive listener: Ctrl/⌘ zooms around the pointer, a plain wheel scrolls sideways.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientWidth : 1;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const px = e.clientX - el.getBoundingClientRect().left;
        live.current.zoomTo(zoomRef.current * Math.exp(-e.deltaY * unit * 0.002), undefined, px);
        return;
      }
      if (el.scrollWidth <= el.clientWidth) return;
      const d = e.shiftKey ? e.deltaY || e.deltaX : Math.abs(e.deltaY) > Math.abs(e.deltaX) ? e.deltaY : 0;
      if (!d) return;
      e.preventDefault();
      el.scrollLeft += d * unit;
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [total > 0]);

  // Keep the playhead in view as it moves (playing, stepping, seeking from elsewhere). Zooming
  // doesn't move the playhead, so looking around while paused never yanks the view back.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || el.scrollWidth <= el.clientWidth) return;
    const x = GUTTER + playhead * live.current.pps;
    if (x < el.scrollLeft + 4 || x > el.scrollLeft + el.clientWidth - 24) {
      el.scrollLeft = Math.max(0, x - el.clientWidth * 0.2);
    }
  }, [playhead]);

  /** Where a dragged clip would land: before or after the picture under the pointer. */
  const dropTarget = useCallback((t: number): { to: ClipRef; x: number } | null => {
    const list = live.current.pictures.filter((p) => p.end > p.start);
    if (list.length === 0) return null;
    const over = list.find((p) => t >= p.start && t < p.end) ?? (t <= list[0].start ? list[0] : list[list.length - 1]);
    const before = t < (over.start + over.end) / 2;
    return {
      to: { beat: over.beat, clip: (over.clip ?? 0) + (before ? 0 : 1) },
      x: before ? over.start : over.end,
    };
  }, []);

  // Trims, moves and ruler scrubs follow the pointer anywhere on the page until it is released.
  useEffect(() => {
    const move = (e: PointerEvent) => {
      const g = gesture.current;
      const L = live.current;
      if (!g) return;
      if (g.type === "trim") {
        const delta = (e.clientX - g.x) / L.pps;
        if (Math.abs(delta) < 0.02) return;
        gesture.current = { ...g, x: e.clientX };
        L.onTrim(g.beat, g.clip, g.edge, delta);
      } else if (g.type === "item") {
        const delta = (e.clientX - g.x0) / L.pps;
        if (!g.active) {
          if (Math.abs(e.clientX - g.x0) < DRAG_THRESHOLD) return;
          g.active = true;
        }
        let start = g.s0;
        let end = g.e0;
        if (g.mode === "move") {
          start = clamp(g.s0 + delta, g.lo, Math.max(g.lo, g.hi - (g.e0 - g.s0)));
          end = start + (g.e0 - g.s0);
        } else if (g.mode === "in") {
          start = clamp(g.s0 + delta, g.lo, g.e0 - MIN_ITEM);
        } else {
          end = clamp(g.e0 + delta, g.s0 + MIN_ITEM, g.hi);
        }
        itemDragRef.current = { key: g.key, start, end };
        setItemDrag(itemDragRef.current);
      } else if (g.type === "scrub") {
        L.onSeek(timeAt(e.clientX));
      } else if (g.type === "move") {
        if (!g.active) {
          if (Math.hypot(e.clientX - g.x0, e.clientY - g.y0) < DRAG_THRESHOLD) return;
          g.active = true;
        }
        const target = dropTarget(timeAt(e.clientX));
        movingRef.current = target ? { key: g.key, ...target } : null;
        setMoving(movingRef.current);
      }
    };
    const up = () => {
      const g = gesture.current;
      gesture.current = null;
      if (g?.type === "item") {
        const d = itemDragRef.current;
        itemDragRef.current = null;
        setItemDrag(null);
        if (g.active && d) {
          const b = live.current.blocks.find((x) => x.key === g.key);
          if (b?.kind === "title" && b.lo != null) live.current.onTitleTime?.(b.beat, d.start - b.lo, d.end - b.lo);
          else if (b?.kind === "cue" && b.cue) live.current.onCueTime?.(b.cue, d.start, d.end);
          suppressClick.current = true;
          setTimeout(() => (suppressClick.current = false), 0);
        }
        return;
      }
      if (g?.type === "move" && g.active) {
        const target = movingRef.current;
        movingRef.current = null;
        const { from } = g;
        const noop =
          !target || (target.to.beat === from.beat && (target.to.clip === from.clip || target.to.clip === from.clip + 1));
        if (!noop && target) live.current.onMoveClip?.(from, target.to);
        setMoving(null);
        // The click that ends a drag is not a selection.
        suppressClick.current = true;
        setTimeout(() => (suppressClick.current = false), 0);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [timeAt, dropTarget]);

  /** The clip the blade cuts: the selected one, or the picture under the playhead when none is. */
  const bladeTarget = (): Block | null => {
    const inside = (b: Block) => playhead > b.start + MIN_SPLIT && playhead < b.end - MIN_SPLIT;
    if (sel && sel.clip != null) return inside(sel) ? sel : null;
    return pictures.find(inside) ?? null;
  };

  const keys = (e: React.KeyboardEvent) => {
    // The zoom slider keeps its own arrows.
    if ((e.target as HTMLElement).closest("input, textarea, select")) return;
    const mod = e.ctrlKey || e.metaKey;
    // A step you can feel: a second by default, a nudge with Alt, a stride with Ctrl.
    const step = e.altKey ? 0.2 : mod ? 5 : 1;
    const edge: "in" | "out" = e.shiftKey ? "in" : "out";
    switch (e.key) {
      case "ArrowRight":
      case "ArrowLeft": {
        const dir = e.key === "ArrowRight" ? 1 : -1;
        if (!sel) {
          onSeek(Math.min(Math.max(playhead + dir * step, 0), total));
        } else if (sel.kind === "title" || sel.kind === "cue") {
          // Titles and captions: arrows move them (Shift: only the start, as for a clip's in).
          const lo = sel.lo ?? 0;
          const hi = sel.hi ?? total;
          const d = dir * Math.min(step, 1) * (e.altKey ? 0.5 : 1);
          let start = sel.start;
          let end = sel.end;
          if (e.shiftKey) start = clamp(sel.start + d, lo, sel.end - MIN_ITEM);
          else {
            start = clamp(sel.start + d, lo, Math.max(lo, hi - (sel.end - sel.start)));
            end = start + (sel.end - sel.start);
          }
          if (sel.kind === "title") onTitleTime?.(sel.beat, start - lo, end - lo);
          else if (sel.cue) onCueTime?.(sel.cue, start, end);
        } else {
          onTrim(sel.beat, sel.clip, edge, dir * step);
        }
        break;
      }
      case "h":
        if (mod || sel?.kind !== "cue" || !sel.cue) return;
        onCueHide?.(sel.cue, !sel.cue.hidden);
        break;
      case "Enter":
        if (sel?.kind !== "cue") return;
        setEditingCue(sel.key);
        break;
      case "i":
      case "o": {
        // Put this edge where the playhead is, read back into the source file's own clock.
        if (mod || !sel || playhead < sel.start || playhead > sel.end) return;
        if (sel.kind === "title" || sel.kind === "cue") {
          const lo = sel.lo ?? 0;
          const start = e.key === "i" ? clamp(playhead, lo, sel.end - MIN_ITEM) : sel.start;
          const end = e.key === "o" ? Math.max(playhead, sel.start + MIN_ITEM) : sel.end;
          if (sel.kind === "title") onTitleTime?.(sel.beat, start - lo, end - lo);
          else if (sel.cue) onCueTime?.(sel.cue, start, end);
          break;
        }
        onSetEdge(sel.beat, sel.clip, e.key === "i" ? "in" : "out", sel.sourceIn + (playhead - sel.start));
        break;
      }
      case "s":
        if (mod) return;
        if (sel) onSnap(sel.beat, sel.clip);
        break;
      case "m":
        if (mod || !sel || sel.track == null || !onToggleTrack) return;
        onToggleTrack(sel.beat, sel.clip, sel.track);
        break;
      case "z":
        if (mod) return; // Ctrl/⌘+Z is the editor's undo.
        if (beatZoom) fit();
        else zoomToBeat();
        break;
      case "c":
      case "k":
      case "K": {
        if (e.key === "c" ? mod : !mod) return; // c alone, or Ctrl/⌘+K
        const b = bladeTarget();
        if (!b || b.clip == null || !onSplit) return;
        onSplit(b.beat, b.clip, b.sourceIn + (playhead - b.start));
        break;
      }
      case "Delete":
      case "Backspace": {
        if (sel?.kind === "cue" && sel.cue) {
          // A caption is not deleted, only hidden: the transcript still has it.
          onCueHide?.(sel.cue, true);
          break;
        }
        if (!sel || !onDelete) return;
        // Own audio is the clip's sound, not a block of its own: deleting it would drop the picture too.
        if (sel.kind === "title" || sel.key.startsWith("a-")) return;
        const { beat, clip } = sel;
        onSelect(null, -1, null);
        onDelete(beat, clip);
        break;
      }
      case "=":
      case "+":
        zoomStep(1.5);
        break;
      case "-":
      case "_":
        zoomStep(1 / 1.5);
        break;
      case "\\":
        fit();
        break;
      case " ":
        onTogglePlay();
        break;
      case "Escape":
        onSelect(null, -1, null);
        if (beatZoom) fit();
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  const laneStates = Array.from({ length: laneCount }, (_, track) => {
    const audio = blocks.filter((b) => b.track === track);
    const silent = audio.length > 0 && audio.every((b) => {
      const db = b.videoId == null ? undefined : levelOf(b.videoId, b.sourceIn, b.sourceIn + b.end - b.start)?.[track];
      return db != null && db <= SILENT_DB;
    });
    return { track, silent, off: audio.length > 0 && audio.every((b) => b.off || b.muted) };
  });
  const silentTracks = laneStates.filter((s) => s.silent);
  const collapseSilent = !expandSilent && silentTracks.length > 0;
  const lanes: { id: string; kind: Block["kind"]; label: string; track?: number; summary?: boolean }[] = [
    { id: "title", kind: "title", label: "Titles" },
    ...(captions && captions.length > 0 ? [{ id: "cue", kind: "cue" as const, label: "Captions" }] : []),
    { id: "picture", kind: "picture", label: "Picture" },
    ...laneStates.filter((s) => !collapseSilent || !s.silent).map(({ track }) => ({
      id: `a${track}`, kind: "bed" as const, label: trackName(track), track,
    })),
    ...(collapseSilent ? [{ id: "silent", kind: "bed" as const, label: `${silentTracks.length} silent tracks`, summary: true }] : []),
  ];
  const binTarget = (clientX: number) => {
    const t = timeAt(clientX);
    const clip = pictures.find((p) => p.start <= t && t < p.end);
    if (clip) return { to: { beat: clip.beat, clip: clip.clip! }, x: clip.start };
    const beat = beatStarts.find((b) => b.start <= t && t < b.end) ?? beatStarts[beatStarts.length - 1];
    return beat ? { to: { beat: beat.beat, clip: script.beats[beat.beat].clips.length }, x: beat.end }
      : { to: { beat: 0, clip: 0 }, x: 0 };
  };
  const selOwner = ownerOf(selected);

  // Only what is on screen (and half a screen either side) is drawn; a long cut zoomed in is wide.
  const margin = vw / 2;
  const visFrom = Math.max(0, (scrollX - GUTTER - margin) / pps);
  const visTo = Math.min(extent, (scrollX + vw - GUTTER + margin) / pps);

  // A labelled tick at least ~70 px apart on a round number of seconds, with minor ticks between.
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const step = steps.find((s) => s * pps >= 70) ?? 600;
  const minor = step / (step === 15 || step === 30 ? 3 : step === 0.2 || step === 2 || step === 120 ? 4 : 5);
  const ticks: { t: number; major: boolean }[] = [];
  for (let i = Math.floor(visFrom / minor); i * minor <= visTo + 1e-9; i++) {
    const t = i * minor;
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    ticks.push({ t, major });
  }

  const sliderMax = 1000;
  const sliderValue = maxZoom > 1 ? Math.round((Math.log(z) / Math.log(maxZoom)) * sliderMax) : 0;

  const startGestureFromRuler = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    gesture.current = { type: "scrub" };
    onSeek(timeAt(e.clientX));
    wrapRef.current?.focus();
  };

  const filmstrip = (b: Block) => {
    const list = b.videoId != null ? sortedFrames[b.videoId] : undefined;
    if (!list || list.length === 0) return null;
    const dur = b.end - b.start;
    const w = dur * pps;
    const n = Math.max(1, Math.round(w / THUMB_W));
    const tw = w / n;
    // Thumbs off screen are skipped, so a deep zoom does not ask for hundreds of images.
    const first = clamp(Math.floor(((visFrom - b.start) * pps) / tw), 0, n - 1);
    const last = clamp(Math.ceil(((visTo - b.start) * pps) / tw), 0, n - 1);
    const thumbs = [];
    for (let i = first; i <= last; i++) {
      const f = nearestFrame(list, b.sourceIn + ((i + 0.5) / n) * dur);
      if (!f) continue;
      thumbs.push(
        <span key={i} className="tl-thumb" style={{ left: i * tw, width: tw }}>
          <MediaImage path={f.path} alt="" draggable={false} loading="lazy" />
        </span>,
      );
    }
    return <span className="tl-strip">{thumbs}</span>;
  };

  const selectable = (b: Block) => (e: React.MouseEvent) => {
    e.stopPropagation();
    if (suppressClick.current) return;
    onSelect(b.key, b.beat, b.clip);
    wrapRef.current?.focus();
  };

  return (
    <div
      className="script-timeline tl"
      ref={wrapRef}
      tabIndex={0}
      onKeyDown={keys}
      aria-label="Timeline"
      data-lane-height={laneDensity}
    >
      <div className="tl-toolbar small">
        <button
          type="button"
          className="tl-tool"
          tabIndex={-1}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => zoomStep(1 / 1.5)}
          disabled={z <= 1}
          title="Zoom out (-)"
        >
          −
        </button>
        <input
          type="range"
          className="tl-zoom"
          min={0}
          max={sliderMax}
          value={sliderValue}
          disabled={maxZoom <= 1}
          aria-label="Timeline zoom"
          onChange={(e) => {
            const v = Number(e.target.value) / sliderMax;
            const el = scrollRef.current;
            const px = el ? xOf(playhead) - el.scrollLeft : -1;
            const onScreen = el != null && px >= 0 && px <= el.clientWidth;
            zoomTo(Math.pow(maxZoom, v), onScreen ? playhead : undefined, onScreen ? px : undefined);
          }}
          onPointerUp={() => wrapRef.current?.focus()}
        />
        <button
          type="button"
          className="tl-tool"
          tabIndex={-1}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => zoomStep(1.5)}
          disabled={z >= maxZoom}
          title="Zoom in (=)"
        >
          +
        </button>
        <button
          type="button"
          className="tl-tool tl-fit"
          tabIndex={-1}
          onMouseDown={(e) => e.preventDefault()}
          onClick={fit}
          disabled={z <= 1}
          title="Fit the whole cut (\)"
        >
          Fit
        </button>
        <span className="tl-zoom-label muted">{beatZoom ? "beat" : `${z < 10 ? z.toFixed(1) : Math.round(z)}×`}</span>
        {silentTracks.length > 0 && <button type="button" className="tl-tool" aria-expanded={expandSilent}
          onClick={() => setExpandSilent(!expandSilent)}>{expandSilent ? "Collapse silent tracks" : `${silentTracks.length} silent tracks`}</button>}
        <label className="tl-density">
          <span>Lane height</span>
          <select aria-label="Timeline lane height" value={laneDensity} onChange={(e) => {
            const next = e.target.value as typeof laneDensity;
            setLaneDensity(next);
            try { localStorage.setItem("ghostreel.timelineLaneHeight", next); } catch { /* Session-only when storage is unavailable. */ }
          }}>
            <option value="compact">Compact</option>
            <option value="normal">Normal</option>
            <option value="tall">Tall</option>
          </select>
        </label>
        <span className="tl-spacer" />
        {target > 0 && !targetOnScale && (
          <span className="tl-target-note muted" title="The target is too far past the end to draw to scale">
            Target {rulerLabel(target, 1)} · {rulerLabel(target - total, 1)} to go
          </span>
        )}
        {sel && <span className="tl-sel muted">{sel.label}{sel.detail ? ` · ${sel.detail}` : ""}</span>}
        <button
          type="button"
          className={`tl-tool ${showKeys ? "on" : ""}`}
          tabIndex={-1}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => setShowKeys((v) => !v)}
          aria-pressed={showKeys}
          title="Timeline keys"
        >
          ?
        </button>
      </div>

      <div className="tl-body">
        <div className="tl-names small muted" ref={namesRef}>
          <div className="tl-name tl-name-ruler" />
          {lanes.map((lane) =>
            lane.summary ? (
              <button type="button" key={lane.id} className="tl-name tl-name-bed tl-silent-summary" onClick={() => setExpandSilent(true)}>
                {lane.label}
              </button>
            ) : lane.track == null ? (
              <div
                key={lane.id}
                className={`tl-name tl-name-${lane.kind}`}
                title={
                  lane.kind === "title"
                    ? "On-screen text, one block per beat: drag to move, drag an edge to retime"
                    : lane.kind === "cue"
                      ? "Speech captions: drag to move, drag an edge to retime, double-click to edit, H hides"
                      : "The pictures, in story order"
                }
              >
                {lane.kind === "title" ? (
                  <Type size={13} strokeWidth={2} aria-hidden />
                ) : lane.kind === "cue" ? (
                  <MessageSquareText size={13} strokeWidth={2} aria-hidden />
                ) : (
                  <Film size={13} strokeWidth={2} aria-hidden />
                )}
                <span className="tl-track-name">{lane.label}</span>
              </div>
            ) : (
              <div
                key={lane.id}
                className={`tl-name tl-name-bed tl-name-audio ${audible(lane.track, monitor) ? "" : "silenced"}`}
              >
                <AudioLines size={13} strokeWidth={2} aria-hidden />
                <span className="tl-track-name" title={`Audio track ${lane.track + 1} of the source files`}>{lane.label}</span>
                {(laneStates[lane.track].silent || laneStates[lane.track].off) && <span className="tl-lane-state">
                  {laneStates[lane.track].silent ? "silent" : "off"}
                </span>}
                <button
                  type="button"
                  className={`tl-ms ${monitor.muted.includes(lane.track) ? "on mute" : ""}`}
                  tabIndex={-1}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => flipMonitor("muted", lane.track!)}
                  aria-pressed={monitor.muted.includes(lane.track)}
                  title={`Mute ${lane.label} in the viewer (not saved)`}
                >
                  M
                </button>
                <button
                  type="button"
                  className={`tl-ms ${monitor.solo.includes(lane.track) ? "on solo" : ""}`}
                  tabIndex={-1}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => flipMonitor("solo", lane.track!)}
                  aria-pressed={monitor.solo.includes(lane.track)}
                  title={`Solo ${lane.label} in the viewer (not saved)`}
                >
                  S
                </button>
              </div>
            ),
          )}
        </div>

        <div className="tl-scroll" ref={scrollRef} onScroll={(e) => {
          setScrollX(e.currentTarget.scrollLeft);
          const atBottom = e.currentTarget.scrollTop + e.currentTarget.clientHeight >= e.currentTarget.scrollHeight - 1;
          setTracksAtBottom(atBottom);
          if (namesRef.current) namesRef.current.scrollTop = e.currentTarget.scrollTop;
          const content = contentRef.current;
          if (content) {
            const bottom = e.currentTarget.scrollTop + e.currentTarget.clientHeight;
            const laneEls = content.querySelectorAll<HTMLElement>(".tl-lane");
            let count = 0;
            laneEls.forEach((l) => {
              if (l.offsetTop + l.offsetHeight > bottom + 2) count++;
            });
            setTracksBelow(count);
          }
        }}>
          <div className="tl-content" ref={contentRef} style={{ width: contentW }}>
            <div className="tl-ruler" onPointerDown={startGestureFromRuler} title="Click or drag to scrub">
              {ticks.map(({ t, major }) => (
                <span key={t.toFixed(3)} className={`tl-tick ${major ? "major" : ""}`} style={{ left: xOf(t) }}>
                  {major && !(targetOnScale && xOf(t) < xOf(target) + 8 && xOf(t) + 30 > xOf(target) - 72) && <span className="tl-tick-label">{rulerLabel(t, step)}</span>}
                </span>
              ))}
              {beatStarts
                .filter((bs) => bs.end >= visFrom && bs.start <= visTo)
                .map((bs) => {
                  const isCompact =
                    laneDensity === "compact" ||
                    (typeof window !== "undefined" && window.innerHeight <= 760);
                  const nearTarget =
                    targetOnScale &&
                    ((target - bs.start <= 1.05 && bs.start <= target) ||
                      (xOf(target) - xOf(bs.start) < 70 && bs.start <= target && bs.end >= target - 1.0));
                  const hideBeatName = isCompact && nearTarget;
                  return (
                    <span
                      key={bs.beat}
                      className={`tl-beat-tag ${sel?.beat === bs.beat ? "current" : ""} ${
                        nearTarget ? "near-target" : ""
                      }`.trim()}
                      style={{ left: xOf(bs.start), maxWidth: Math.max(0, (bs.end - bs.start) * pps - 2) }}
                      title={`Beat ${bs.beat + 1}: ${bs.id}`}
                    >
                      {hideBeatName ? "" : bs.id}
                    </span>
                  );
                })}
              {targetOnScale && (
                <span
                  className={`tl-target ${total > target + 1e-6 ? "over" : ""}`}
                  style={{ left: xOf(target) }}
                  title={
                    total > target
                      ? `Target length ${rulerLabel(target, 1)}: the cut runs ${formatDuration(total - target)} over`
                      : `Target length ${rulerLabel(target, 1)}: ${formatDuration(target - total)} of room left`
                  }
                >
                  <span className="tl-target-label">Target {rulerLabel(target, 1)}</span>
                </span>
              )}
              <span className="tl-playhead-handle" style={{ left: xOf(playhead) }} />
            </div>

            {lanes.map((lane) => (
              <div
                key={lane.id}
                className={`tl-lane tl-lane-${lane.kind} ${
                  lane.track != null ? `tl-lane-audio ${audible(lane.track, monitor) ? "" : "silenced"}` : ""
                }`}
                onDragOver={lane.kind === "picture" && onInsertPick ? (e) => {
                  if (!e.dataTransfer.types.includes(PICK_MIME)) return;
                  e.preventDefault();
                  e.dataTransfer.dropEffect = "copy";
                  setBinDrop(binTarget(e.clientX));
                } : undefined}
                onDragLeave={lane.kind === "picture" ? (e) => {
                  if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setBinDrop(null);
                } : undefined}
                onDrop={lane.kind === "picture" && onInsertPick ? (e) => {
                  e.preventDefault();
                  setBinDrop(null);
                  try {
                    const pick = JSON.parse(e.dataTransfer.getData(PICK_MIME)) as Pick;
                    if (!Number.isInteger(pick.video_id) || !Number.isFinite(pick.in_s) ||
                        !Number.isFinite(pick.out_s) || pick.in_s < 0 || pick.out_s <= pick.in_s || !videos[pick.video_id]) return;
                    const duration = videos[pick.video_id].duration_s;
                    if (duration != null && pick.out_s > duration) return;
                    onInsertPick(pick, binTarget(e.clientX).to);
                  } catch { /* Other drags and invalid payloads do not change the cut. */ }
                } : undefined}
                onClick={(e) => {
                  if (lane.summary) { setExpandSilent(true); return; }
                  if (suppressClick.current) return;
                  onSeek(timeAt(e.clientX));
                }}
              >
                {blocks
                  .filter(
                    (b) =>
                      !lane.summary && b.kind === lane.kind &&
                      (lane.track == null || (b.track === lane.track && !laneStates[lane.track].silent)) &&
                      b.end >= visFrom &&
                      b.start <= visTo,
                  )
                  .map((b) => {
                    const db =
                      b.track != null && b.videoId != null
                        ? levelOf(b.videoId, b.sourceIn, b.sourceIn + (b.end - b.start))?.[b.track]
                        : undefined;
                    const info = b.track != null && b.videoId != null ? trackInfo[b.videoId]?.[b.track] : undefined;
                    // A sound cell can be narrower than the room a caption and a level badge share.
                    // The name wins: below that width the cell shows a shortened file name instead of
                    // a bare "−18 dB", and the badge (with the full name in the tooltip) returns when
                    // there is room for both.
                    const dragged = itemDrag?.key === b.key ? itemDrag : null;
                    const bStart = dragged?.start ?? b.start;
                    const bEnd = dragged?.end ?? b.end;
                    const width = Math.max(1, (bEnd - bStart) * pps);
                    const isItem = b.kind === "title" || b.kind === "cue";
                    const itemGesture = (mode: "move" | "in" | "out") => (e: React.PointerEvent) => {
                      if (e.button !== 0 || (b.kind === "title" ? !onTitleTime : !onCueTime)) return;
                      e.stopPropagation();
                      gesture.current = {
                        type: "item", key: b.key, mode, x0: e.clientX, s0: b.start, e0: b.end,
                        lo: b.lo ?? 0, hi: b.hi ?? total, active: false,
                      };
                    };
                    const compact = b.kind === "bed" && width < 84;
                    const captionFrom = b.kind === "bed" ? 34 : 48;
                    return (
                    <div
                      key={b.key}
                      className={[
                        "tl-block",
                        `tl-${b.kind}`,
                        b.muted ? "muted" : "",
                        b.off ? "off" : "",
                        db != null && db <= SILENT_DB ? "silent" : "",
                        b.inferred ? "inferred" : "",
                        selected === b.key ? "selected" : "",
                        selected !== b.key && selOwner != null && blockOwner(b) === selOwner ? "linked" : "",
                        moving?.key === b.key ? "dragging" : "",
                        b.kind === "picture" && onMoveClip ? "movable" : "",
                        isItem ? "movable" : "",
                        dragged ? "item-dragging" : "",
                        b.cue?.edited ? "edited" : "",
                        compact ? "tl-compact" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      style={{ left: xOf(bStart), width }}
                      onClick={selectable(b)}
                      onDoubleClick={b.kind === "cue" && onCueText ? (e) => {
                        e.stopPropagation();
                        onSelect(b.key, -1, null);
                        setEditingCue(b.key);
                      } : undefined}
                      onPointerDown={(e) => {
                        if (isItem) return itemGesture("move")(e);
                        if (e.button !== 0 || b.kind !== "picture" || b.clip == null || !onMoveClip) return;
                        gesture.current = {
                          type: "move",
                          from: { beat: b.beat, clip: b.clip },
                          key: b.key,
                          x0: e.clientX,
                          y0: e.clientY,
                          active: false,
                        };
                      }}
                      title={`${b.kind === "title" ? "On-screen title · " : b.kind === "cue" ? "Caption · " : ""}${b.track != null ? `${trackName(b.track)} · ` : ""}${b.label}${b.detail ? ` · ${b.detail}` : ""}${
                        info ? ` · ${trackDescription(info)}` : ""
                      }${db != null ? ` · ${dbLabel(db)}` : ""}${
                        b.inferred ? " · voice carried under the pictures" : ""
                      }${b.muted ? " · muted: select and press m, or Unmute" : ""}${
                        b.track != null && onToggleTrack && !b.muted ? " · m: track on/off" : ""
                      }`}
                    >
                      {b.kind === "picture" && filmstrip(b)}
                      {isItem ? (
                        <span className="tl-handle in" onPointerDown={itemGesture("in")} />
                      ) : (
                        <span
                          className="tl-handle in"
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            gesture.current = { type: "trim", beat: b.beat, clip: b.clip, edge: "in", x: e.clientX };
                          }}
                        />
                      )}
                      {b.kind === "cue" && b.cue?.hidden && width >= 20 && <EyeOff size={11} strokeWidth={2} aria-hidden className="tl-cue-hidden" />}
                      {width >= captionFrom && <span className="tl-caption">
                        <span className="tl-label">
                          {b.label}
                        </span>
                        {b.kind !== "bed" && b.detail && <span className="tl-detail">{b.detail}</span>}
                      </span>}
                      {b.kind === "bed" && b.muted && b.track === 0 && b.clip != null && onUnmute &&
                        (b.end - b.start) * pps >= 84 && (
                          <button
                            type="button"
                            className="tl-unmute"
                            tabIndex={-1}
                            onPointerDown={(e) => e.stopPropagation()}
                            onClick={(e) => {
                              e.stopPropagation();
                              onUnmute(b.beat, b.clip!);
                            }}
                            title="Turn this clip's own sound back on"
                          >
                            <Volume2 size={12} strokeWidth={2.25} aria-hidden />
                            Unmute
                          </button>
                        )}
                      {db != null && !b.muted && width >= 84 && (
                        <>
                          <span className="tl-level">{dbLabel(db)}</span>
                          <span className="tl-level-bar" style={{ width: `${levelFraction(db) * 100}%` }} />
                        </>
                      )}
                      {isItem ? (
                        <span className="tl-handle out" onPointerDown={itemGesture("out")} />
                      ) : (
                        <span
                          className="tl-handle out"
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            gesture.current = { type: "trim", beat: b.beat, clip: b.clip, edge: "out", x: e.clientX };
                          }}
                        />
                      )}

                    </div>
                    );
                  })}
                {lane.kind === "cue" && editingCue && (() => {
                  const b = blocks.find((x) => x.key === editingCue);
                  if (!b?.cue) return null;
                  const cue = b.cue;
                  return (
                    <CueTextEditor
                      key={editingCue}
                      left={xOf(b.start)}
                      text={cue.text.replace(/\n/g, " ")}
                      onDone={(text) => {
                        setEditingCue(null);
                        if (text != null && text.trim() && text.trim() !== cue.text.replace(/\n/g, " ")) onCueText?.(cue, text.trim());
                        wrapRef.current?.focus();
                      }}
                    />
                  );
                })()}
                {lane.kind === "picture" && total === 0 && <span className="tl-empty-drop-hint">Drag footage here from Media to start the cut</span>}
                {lane.summary && <button type="button" className="tl-silent-summary" onClick={() => setExpandSilent(true)}>
                  {silentTracks.map((s) => trackName(s.track)).join(" · ")} silent
                </button>}
                {lane.kind === "picture" && binDrop && <span className="tl-drop" style={{ left: xOf(binDrop.x) }} />}
                {lane.kind === "picture" && moving && <span className="tl-drop" style={{ left: xOf(moving.x) }} />}
              </div>
            ))}

            {beatStarts
              .filter((bs) => bs.beat > 0 && bs.start >= visFrom && bs.start <= visTo)
              .map((bs) => (
                <span key={bs.beat} className="tl-beat-line" style={{ left: xOf(bs.start) }} />
              ))}
            {targetOnScale && (
              <span className="tl-target-line" style={{ left: xOf(target) }} />
            )}
            {targetOnScale && total > target && (
              <span
                className="tl-overrun"
                style={{ left: xOf(target), width: (total - target) * pps }}
                title={`Over the ${rulerLabel(target, 1)} target`}
              />
            )}
            <span className="tl-playhead" style={{ left: xOf(playhead) }} />
          </div>
        </div>
      </div>

      {tracksOverflow && (
        <button
          type="button"
          className="tl-track-continuation"
          onClick={() => {
            if (expandSilent && silentTracks.length > 0) {
              setExpandSilent(false);
            } else {
              const el = scrollRef.current;
              if (el) el.scrollTo({ top: tracksAtBottom ? 0 : el.scrollHeight, behavior: "smooth" });
            }
          }}
        >
          <ChevronDown
            size={14}
            strokeWidth={2}
            aria-hidden
            style={tracksAtBottom || (expandSilent && silentTracks.length > 0) ? { transform: "rotate(180deg)" } : undefined}
          />
          {expandSilent && silentTracks.length > 0
            ? tracksAtBottom
              ? "Collapse silent tracks"
              : `${tracksBelow > 0 ? tracksBelow : silentTracks.length} more ${(tracksBelow > 0 ? tracksBelow : silentTracks.length) === 1 ? "track" : "tracks"} — scroll, or collapse silent tracks`
            : tracksAtBottom
              ? "Back to upper tracks"
              : `${tracksBelow > 0 ? `${tracksBelow} more ${tracksBelow === 1 ? "track" : "tracks"} below` : "More tracks below"} — scroll, or click to jump`}
        </button>
      )}

      {showKeys && (
        <div className="tl-keys small muted">
          <span>Click: seek</span>
          <span>Drag a clip: move it</span>
          <span>Drag an edge: trim</span>
          <span><kbd>←</kbd><kbd>→</kbd> trim out</span>
          <span><kbd>shift</kbd> trim in</span>
          <span><kbd>alt</kbd> 0.2 s</span>
          <span><kbd>ctrl</kbd> 5 s</span>
          <span><kbd>i</kbd><kbd>o</kbd> edge at playhead</span>
          <span><kbd>s</kbd> snap to sentence</span>
          {onToggleTrack && <span><kbd>m</kbd> audio track on/off</span>}
          <span><kbd>c</kbd> / <kbd>ctrl</kbd><kbd>k</kbd> blade</span>
          <span><kbd>del</kbd> delete clip or bed (caption: hide)</span>
          {captions && captions.length > 0 && <span><kbd>h</kbd> hide caption · <kbd>enter</kbd> edit it</span>}
          <span><kbd>=</kbd><kbd>-</kbd> zoom</span>
          <span><kbd>ctrl</kbd>+wheel zoom</span>
          <span><kbd>\</kbd> fit</span>
          <span><kbd>z</kbd> zoom{beatZoom ? " out" : " to beat"}</span>
          <span><kbd>space</kbd> play</span>
          <span><kbd>esc</kbd> deselect</span>
        </div>
      )}
    </div>
  );
}

/** Inline editor for a caption's text, opened by double-clicking it on the Captions lane. */
function CueTextEditor({ text, left, onDone }: { text: string; left: number; onDone: (text: string | null) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  const finish = (value: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(value);
  };
  return (
    <input
      ref={ref}
      className="tl-cue-editor"
      style={{ left }}
      defaultValue={text}
      aria-label="Caption text"
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Escape") finish(null);
        else if (e.key === "Enter") {
          e.preventDefault();
          finish(e.currentTarget.value);
        }
      }}
      onBlur={(e) => finish(e.currentTarget.value)}
    />
  );
}
