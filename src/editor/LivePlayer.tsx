import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import {
  fileName,
  mediaUrl,
  videoPlayback,
  videoPlaybackProgress,
  type Beat,
  type PlaybackProgress,
  type Script,
  type VideoRow,
} from "../api";
import {
  ArrowLeftToLine,
  Captions,
  Scan,
  Ellipsis,
  Maximize,
  Minimize,
  Pause,
  Play,
  Repeat,
  SkipBack,
  SkipForward,
  StepBack,
  StepForward,
  Type,
} from "lucide-react";
import { audible, autoTrack, loadAudioTracks, trackProxyUrl, type TrackMonitor } from "./audioTracks";
import { fpsValue, frameClock } from "./timecode";
import "./player.css";

/** One picture clip placed on the cut's clock. */
export interface LiveSegment {
  beatIndex: number;
  clipIndex: number;
  videoId: number;
  inS: number;
  outS: number;
  /** Start and end on the cut's clock. */
  start: number;
  end: number;
  /** Its own sound off: the clip asks for it, or a bed carries the beat's audio. */
  muted: boolean;
  /** Source tracks chosen for it (when not muted); undefined = the picture's own audio. */
  tracks?: number[];
}

export interface LiveBeatSpan {
  beat: Beat;
  start: number;
  end: number;
}

export interface LiveLayout {
  segments: LiveSegment[];
  beats: LiveBeatSpan[];
  total: number;
}

/** Lays the clips end to end the way the timeline draws them: beat order, then clip order. */
export function layoutScript(script: Script): LiveLayout {
  const segments: LiveSegment[] = [];
  const beats: LiveBeatSpan[] = [];
  let t = 0;
  script.beats.forEach((beat, beatIndex) => {
    const start = t;
    const hasBed = !!beat.bed && beat.bed.out_s > beat.bed.in_s;
    beat.clips.forEach((c, clipIndex) => {
      const dur = Math.max(0, c.out_s - c.in_s);
      if (dur <= 0) return;
      segments.push({
        beatIndex,
        clipIndex,
        videoId: c.video_id,
        inS: c.in_s,
        outS: c.out_s,
        start: t,
        end: t + dur,
        muted: c.audio === "mute" || hasBed,
        tracks: c.audio === "mute" || hasBed ? undefined : c.tracks,
      });
      t += dur;
    });
    beats.push({ beat, start, end: t });
  });
  return { segments, beats, total: t };
}

// Kept as exports of this module too: the rendered viewer imports them from here.
export { fpsValue, frameClock };

// Shared across mounts: the first request may build a proxy, which should happen once.
const urlCache = new Map<number, Promise<string>>();
function playableUrl(videoId: number, video: VideoRow | undefined): Promise<string> {
  let p = urlCache.get(videoId);
  if (!p) {
    p = videoPlayback(videoId)
      .then((pb) => mediaUrl(pb.path))
      .catch((err) => {
        if (video) return mediaUrl(video.path);
        throw err;
      });
    urlCache.set(videoId, p);
    p.catch(() => urlCache.delete(videoId));
  }
  return p;
}

const RATES = [0.25, 0.5, 1, 1.5, 2];

export interface PlayerControls {
  seek: (s: number) => void;
  toggle: () => void;
}

interface LivePlayerProps {
  script: Script;
  videos: Record<number, VideoRow>;
  /** Where to start, so switching from the rendered view keeps the playhead. */
  initialTime?: number;
  onTime?: (seconds: number) => void;
  onControls?: (controls: PlayerControls) => void;
  /** Extra controls for the right end of the transport. */
  extra?: ReactNode;
  /** Titles and narration shown here are the ones a render burns in: one switch for both. */
  showTitles: boolean;
  onShowTitles: (on: boolean) => void;
  showNarration: boolean;
  onShowNarration: (on: boolean) => void;
  /** Timeline lane mute/solo: which source tracks are heard here. */
  trackMonitor?: TrackMonitor;
}

/** Where chosen tracks play from at some moment of the cut. */
type TrackSource = { videoId: number; srcTime: number; tracks: number[] };
type TrackEl = { el: HTMLAudioElement; url: string; pending: number | null };

type Slot = 0 | 1;

export default function LivePlayer({
  script,
  videos,
  initialTime = 0,
  onTime,
  onControls,
  extra,
  showTitles,
  onShowTitles,
  showNarration: subtitles,
  onShowNarration,
  trackMonitor,
}: LivePlayerProps) {
  const layout = useMemo(() => layoutScript(script), [script]);
  const fps = fpsValue(script.fps);

  const [urls, setUrls] = useState<Record<number, string>>({});
  const [failed, setFailed] = useState<Record<number, string>>({});
  const [playing, setPlaying] = useState(false);
  const [displayTime, setDisplayTime] = useState(initialTime);
  const [curIdx, setCurIdx] = useState(0);
  const [rate, setRateState] = useState(1);
  const [loop, setLoop] = useState(false);
  const [safeGuides, setSafeGuides] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [progress, setProgress] = useState<PlaybackProgress | null>(null);

  const rootRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const vidA = useRef<HTMLVideoElement>(null);
  const vidB = useRef<HTMLVideoElement>(null);
  const bedRef = useRef<HTMLAudioElement>(null);

  // The playback engine runs on refs: it is driven by rAF and media events, not renders.
  const layoutRef = useRef(layout);
  const fpsRef = useRef(fps);
  const urlsRef = useRef<Record<number, string>>({});
  const requested = useRef(new Set<number>());
  const videosRef = useRef(videos);
  const timeRef = useRef(initialTime);
  const idxRef = useRef(0);
  const activeRef = useRef<Slot>(0);
  const slotSeg = useRef<[number | null, number | null]>([null, null]);
  const slotUrl = useRef<[string | null, string | null]>([null, null]);
  const pending = useRef<[number | null, number | null]>([null, null]);
  const bedUrl = useRef<string | null>(null);
  const bedPending = useRef<number | null>(null);
  const playingRef = useRef(false);
  const rateRef = useRef(1);
  const loopRef = useRef(false);
  const lastFrame = useRef(-1);
  const lastReport = useRef(0);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;
  // Chosen source tracks play from hidden audio elements, one per file+track, made on demand.
  const monitorRef = useRef(trackMonitor);
  monitorRef.current = trackMonitor;
  const trackEls = useRef(new Map<string, TrackEl>());
  const trackUrls = useRef(new Map<string, string>());
  const trackFailed = useRef(new Set<string>());
  const trackInfo = useRef(new Map<number, number>()); // videoId → its automatic track
  const [trackTick, setTrackTick] = useState(0);
  videosRef.current = videos;
  fpsRef.current = fps;
  loopRef.current = loop;

  const vid = (slot: Slot) => (slot === 0 ? vidA.current : vidB.current);

  const showSlot = (slot: Slot) => {
    vidA.current?.setAttribute("data-on", slot === 0 ? "1" : "0");
    vidB.current?.setAttribute("data-on", slot === 1 ? "1" : "0");
  };

  const ensureUrl = useCallback((videoId: number) => {
    if (urlsRef.current[videoId] || requested.current.has(videoId)) return;
    requested.current.add(videoId);
    playableUrl(videoId, videosRef.current[videoId])
      .then((url) => {
        urlsRef.current = { ...urlsRef.current, [videoId]: url };
        setUrls(urlsRef.current);
      })
      .catch((err) => {
        requested.current.delete(videoId);
        setFailed((f) => ({ ...f, [videoId]: String(err) }));
      });
  }, []);

  const ensureTrack = (videoId: number, track: number) => {
    const k = `${videoId}:${track}`;
    if (trackUrls.current.has(k) || trackFailed.current.has(k)) return;
    trackUrls.current.set(k, ""); // asked
    trackProxyUrl(videoId, track)
      .then((url) => trackUrls.current.set(k, url))
      .catch(() => {
        // No proxy (an older backend, a track ffmpeg can't read): play what the picture has instead.
        trackUrls.current.delete(k);
        trackFailed.current.add(k);
      })
      .finally(() => setTrackTick((n) => n + 1));
  };

  const ensureInfo = (videoId: number) => {
    if (trackInfo.current.has(videoId)) return;
    trackInfo.current.set(videoId, 0);
    void loadAudioTracks(videoId).then((infos) => {
      trackInfo.current.set(videoId, autoTrack(infos));
      setTrackTick((n) => n + 1);
    });
  };

  /** Explicit tracks play from proxies unless one of them couldn't be had. */
  const usesTracks = (videoId: number, tracks: number[] | undefined): tracks is number[] =>
    tracks != null && !tracks.some((tr) => trackFailed.current.has(`${videoId}:${tr}`));

  /** Whether the media element's own audio (the file's automatic track) is heard under the monitor. */
  const ownAudible = (videoId: number) => audible(trackInfo.current.get(videoId) ?? 0, monitorRef.current);

  const pictureMuted = (seg: LiveSegment) =>
    seg.muted || usesTracks(seg.videoId, seg.tracks) || !ownAudible(seg.videoId);

  const report = (t: number, force = false) => {
    const frame = Math.floor(t * fpsRef.current + 1e-6);
    if (force || frame !== lastFrame.current) {
      lastFrame.current = frame;
      setDisplayTime(t);
    }
    const now = performance.now();
    // The parent re-renders the whole editor on each report; 20 Hz is enough for a playhead.
    if (force || now - lastReport.current > 50) {
      lastReport.current = now;
      onTimeRef.current?.(t);
    }
  };

  const applyRate = (el: HTMLMediaElement | null) => {
    if (!el) return;
    el.defaultPlaybackRate = rateRef.current;
    el.playbackRate = rateRef.current;
  };

  /** Loads segment `idx` into a slot, `offset` seconds in. False while its source is not ready. */
  const prime = (slot: Slot, idx: number, offset: number): boolean => {
    const v = vid(slot);
    const seg = layoutRef.current.segments[idx];
    if (!v || !seg) return false;
    slotSeg.current[slot] = idx;
    v.muted = pictureMuted(seg);
    const url = urlsRef.current[seg.videoId];
    if (!url) {
      ensureUrl(seg.videoId);
      // Drop the previous source: its picture and clock belong to another clip.
      if (slotUrl.current[slot] != null) {
        slotUrl.current[slot] = null;
        pending.current[slot] = null;
        v.pause();
        v.removeAttribute("src");
        v.load();
      }
      return false;
    }
    const target = seg.inS + offset;
    if (slotUrl.current[slot] !== url) {
      slotUrl.current[slot] = url;
      pending.current[slot] = target;
      v.src = url;
      applyRate(v);
      return true;
    }
    applyRate(v);
    if (v.readyState >= 1) {
      // Skip tiny seeks so an edit elsewhere in the cut doesn't stutter the clip on screen.
      const slack = playingRef.current ? 0.12 : 0.02;
      if (Math.abs(v.currentTime - target) > slack) v.currentTime = target;
    } else {
      pending.current[slot] = target;
    }
    return true;
  };

  const beatAt = (t: number): LiveBeatSpan | undefined => {
    const spans = layoutRef.current.beats;
    return spans.find((b) => t >= b.start && t < b.end);
  };

  /** Keeps the bed's audio under the beat's pictures: best effort, resynced when it drifts. */
  const syncBed = (t: number, hard: boolean) => {
    const a = bedRef.current;
    if (!a) return;
    const span = beatAt(t);
    const bed = span?.beat.bed;
    const off = span ? t - span.start : 0;
    if (!span || !bed || off >= bed.out_s - bed.in_s || usesTracks(bed.video_id, bed.tracks)) {
      if (!a.paused) a.pause();
      return;
    }
    a.muted = !ownAudible(bed.video_id);
    const url = urlsRef.current[bed.video_id];
    if (!url) {
      ensureUrl(bed.video_id);
      if (!a.paused) a.pause();
      return;
    }
    const target = bed.in_s + off;
    if (bedUrl.current !== url) {
      bedUrl.current = url;
      bedPending.current = target;
      a.src = url;
    } else if (a.readyState >= 1) {
      if (Math.abs(a.currentTime - target) > (hard ? 0.05 : 0.3)) a.currentTime = target;
    } else {
      bedPending.current = target;
    }
    applyRate(a);
    if (playingRef.current) {
      if (a.paused) a.play().catch(() => {});
    } else if (!a.paused) {
      a.pause();
    }
  };

  /** What chosen tracks should be playing at cut time `t`: a bed's, else the clip's, else none. */
  const trackSourceAt = (t: number): TrackSource | null => {
    const span = beatAt(t);
    const bed = span?.beat.bed;
    if (span && bed && bed.out_s > bed.in_s) {
      const off = t - span.start;
      if (off >= bed.out_s - bed.in_s || !usesTracks(bed.video_id, bed.tracks)) return null;
      return { videoId: bed.video_id, srcTime: bed.in_s + off, tracks: bed.tracks };
    }
    const seg = layoutRef.current.segments[segAt(t)];
    if (!seg || t < seg.start || t > seg.end || !usesTracks(seg.videoId, seg.tracks)) return null;
    return { videoId: seg.videoId, srcTime: seg.inS + (t - seg.start), tracks: seg.tracks };
  };

  /** Keeps the chosen tracks in step with the cut clock, the same way the bed is kept. */
  const syncTracks = (t: number, hard: boolean) => {
    const want = trackSourceAt(t);
    const live = new Set<string>();
    for (const track of want?.tracks ?? []) {
      const k = `${want!.videoId}:${track}`;
      const url = trackUrls.current.get(k);
      if (!url) {
        ensureTrack(want!.videoId, track);
        continue;
      }
      live.add(k);
      let te = trackEls.current.get(k);
      if (!te) {
        const el = new Audio();
        el.preload = "auto";
        const entry: TrackEl = { el, url, pending: want!.srcTime };
        el.addEventListener("loadedmetadata", () => {
          if (entry.pending != null) {
            el.currentTime = entry.pending;
            entry.pending = null;
          }
          syncTracks(timeRef.current, true);
        });
        el.src = url;
        trackEls.current.set(k, entry);
        te = entry;
      }
      const { el } = te;
      el.muted = !audible(track, monitorRef.current);
      if (el.readyState >= 1) {
        if (Math.abs(el.currentTime - want!.srcTime) > (hard ? 0.05 : 0.15)) el.currentTime = want!.srcTime;
      } else {
        te.pending = want!.srcTime;
      }
      applyRate(el);
      if (playingRef.current) {
        if (el.paused) el.play().catch(() => {});
      } else if (!el.paused) {
        el.pause();
      }
    }
    for (const [k, te] of trackEls.current) if (!live.has(k) && !te.el.paused) te.el.pause();
  };

  const syncAudio = (t: number, hard: boolean) => {
    syncBed(t, hard);
    syncTracks(t, hard);
  };

  const pauseTracks = () => {
    for (const te of trackEls.current.values()) te.el.pause();
  };

  const segAt = (t: number) => {
    const segs = layoutRef.current.segments;
    if (!segs.length) return -1;
    for (let i = 0; i < segs.length; i++) if (t < segs[i].end) return i;
    return segs.length - 1;
  };

  /** Puts everything at cut time `t`; keeps playing if it was. */
  const place = (t: number) => {
    const { segments, total } = layoutRef.current;
    const time = Math.min(Math.max(0, t), total);
    timeRef.current = time;
    const idx = segAt(time);
    if (idx < 0) {
      vidA.current?.pause();
      vidB.current?.pause();
      bedRef.current?.pause();
      pauseTracks();
      report(time, true);
      return;
    }
    idxRef.current = idx;
    setCurIdx(idx);
    const seg = segments[idx];
    const active = activeRef.current;
    const other = (1 - active) as Slot;
    // At the very end, hold the last frame rather than a black one past the out point.
    const offset = Math.min(time - seg.start, seg.end - seg.start - 0.5 / fpsRef.current);
    const ready = prime(active, idx, Math.max(0, offset));
    vid(other)?.pause();
    if (idx + 1 < segments.length) prime(other, idx + 1, 0);
    const v = vid(active);
    if (v) {
      if (playingRef.current && ready) v.play().catch(() => {});
      else if (!playingRef.current) v.pause();
    }
    showSlot(active);
    syncAudio(time, true);
    report(time, true);
  };

  const pause = () => {
    playingRef.current = false;
    setPlaying(false);
    vidA.current?.pause();
    vidB.current?.pause();
    bedRef.current?.pause();
    pauseTracks();
    report(timeRef.current, true);
  };

  const play = () => {
    const { segments, total } = layoutRef.current;
    if (!segments.length) return;
    playingRef.current = true;
    setPlaying(true);
    if (timeRef.current >= total - 0.5 / fpsRef.current) place(0);
    else place(timeRef.current);
  };

  const toggle = () => (playingRef.current ? pause() : play());

  /** Moves on to the next clip: same element when it continues the same footage, else swap A/B. */
  const advance = () => {
    const { segments, total } = layoutRef.current;
    const idx = idxRef.current;
    const next = idx + 1;
    if (next >= segments.length) {
      if (loopRef.current) place(0);
      else {
        pause();
        place(total);
      }
      return;
    }
    const seg = segments[idx];
    const ns = segments[next];
    const active = activeRef.current;
    const other = (1 - active) as Slot;
    const v = vid(active);
    const o = vid(other);
    if (v && !v.ended && ns.videoId === seg.videoId && Math.abs(ns.inS - v.currentTime) < 0.15) {
      idxRef.current = next;
      setCurIdx(next);
      v.muted = pictureMuted(ns);
      slotSeg.current[active] = next;
      timeRef.current = ns.start;
      if (next + 1 < segments.length) prime(other, next + 1, 0);
      syncAudio(ns.start, true);
      report(ns.start, true);
      return;
    }
    if (o && slotSeg.current[other] === next && slotUrl.current[other] && pending.current[other] == null && o.readyState >= 2) {
      o.muted = pictureMuted(ns);
      applyRate(o);
      o.play().catch(() => {});
      v?.pause();
      activeRef.current = other;
      showSlot(other);
      idxRef.current = next;
      setCurIdx(next);
      timeRef.current = ns.start;
      if (next + 1 < segments.length) prime(active, next + 1, 0);
      syncAudio(ns.start, true);
      report(ns.start, true);
      return;
    }
    place(ns.start);
  };

  // The clock: while playing, the cut time follows the active element's source time.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!playingRef.current) return;
      const { segments } = layoutRef.current;
      const idx = idxRef.current;
      const seg = segments[idx];
      const active = activeRef.current;
      const v = vid(active);
      if (!seg || !v) return;
      if (
        slotSeg.current[active] !== idx ||
        slotUrl.current[active] !== urlsRef.current[seg.videoId] ||
        v.readyState < 1 ||
        pending.current[active] != null
      ) return;
      const src = v.currentTime;
      if (v.ended || src >= seg.outS - 0.25 / fpsRef.current) {
        advance();
        return;
      }
      const t = Math.min(seg.end, seg.start + Math.max(0, src - seg.inS));
      timeRef.current = t;
      syncAudio(t, false);
      report(t);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // The engine reads everything through refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // New script (an unsaved edit, too): keep the playhead and re-place the clips around it.
  useEffect(() => {
    layoutRef.current = layout;
    place(timeRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  // A source just became ready: load it where the playhead is.
  useEffect(() => {
    if (Object.keys(urls).length) place(timeRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urls]);

  // Ask for the sources around the playhead only; the first ask may build a proxy.
  useEffect(() => {
    const segs = layout.segments;
    for (const i of [curIdx, curIdx + 1, curIdx + 2]) {
      const seg = segs[i];
      if (!seg) continue;
      ensureUrl(seg.videoId);
      ensureInfo(seg.videoId);
      for (const tr of seg.tracks ?? []) ensureTrack(seg.videoId, tr);
    }
    const cur = segs[curIdx];
    if (cur) {
      for (const b of [cur.beatIndex, cur.beatIndex + 1]) {
        const bed = layout.beats[b]?.beat.bed;
        if (!bed) continue;
        ensureUrl(bed.video_id);
        ensureInfo(bed.video_id);
        for (const tr of bed.tracks ?? []) ensureTrack(bed.video_id, tr);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curIdx, layout, ensureUrl]);

  // A track proxy arrived or failed, a file's tracks became known, or the lane monitor changed:
  // re-decide what is heard without moving the picture.
  useEffect(() => {
    const seg = layoutRef.current.segments[idxRef.current];
    const v = vid(activeRef.current);
    if (seg && v) v.muted = pictureMuted(seg);
    syncAudio(timeRef.current, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackTick, trackMonitor]);

  // Hidden track players aren't in the DOM, so they'd outlive the viewer without this.
  useEffect(() => {
    const els = trackEls.current;
    return () => {
      for (const te of els.values()) {
        te.el.pause();
        te.el.removeAttribute("src");
        te.el.load();
      }
      els.clear();
    };
  }, []);

  useEffect(() => {
    onControls?.({
      seek: (s) => place(s),
      toggle,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onControls]);

  useEffect(() => {
    const onFs = () => setFullscreen(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const current = layout.segments[curIdx];
  const currentVideoId = current?.videoId;
  const waitingFor = currentVideoId != null && !urls[currentVideoId] ? currentVideoId : null;

  // Poll the proxy build while the clip on screen waits for one.
  useEffect(() => {
    setProgress(null);
    if (waitingFor == null || failed[waitingFor]) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const p = await videoPlaybackProgress(waitingFor);
        if (!stop) setProgress(p);
      } catch { /* The wait still ends when the URL arrives. */ }
      if (!stop) timer = setTimeout(poll, 500);
    };
    void poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [waitingFor, failed]);

  const onMeta = (slot: Slot) => {
    const v = vid(slot);
    if (!v) return;
    applyRate(v);
    const target = pending.current[slot];
    if (target != null) {
      v.currentTime = target;
      pending.current[slot] = null;
    }
    if (slot === activeRef.current && playingRef.current && slotSeg.current[slot] === idxRef.current) {
      v.play().catch(() => {});
    }
  };

  const onBedMeta = () => {
    const a = bedRef.current;
    if (!a) return;
    if (bedPending.current != null) {
      a.currentTime = bedPending.current;
      bedPending.current = null;
    }
    syncAudio(timeRef.current, true);
  };

  const setRate = (r: number) => {
    rateRef.current = r;
    setRateState(r);
    applyRate(vidA.current);
    applyRate(vidB.current);
    applyRate(bedRef.current);
    for (const te of trackEls.current.values()) applyRate(te.el);
  };

  const step = (frames: number) => {
    pause();
    const f = fpsRef.current;
    place((Math.round(timeRef.current * f) + frames) / f);
  };

  const seekBy = (s: number) => place(timeRef.current + s);

  const boundaries = useMemo(() => {
    const set = new Set<number>([0, layout.total]);
    for (const s of layout.segments) set.add(s.start);
    return [...set].sort((a, b) => a - b);
  }, [layout]);

  const prevBoundary = () => {
    const t = timeRef.current;
    const b = [...boundaries].reverse().find((x) => x < t - 0.05);
    place(b ?? 0);
  };
  const nextBoundary = () => {
    const t = timeRef.current;
    place(boundaries.find((x) => x > t + 0.01) ?? layout.total);
  };

  // J/K/L as an editor knows them, within what a <video> can do: J slows down, then steps back.
  const shuttleBack = () => {
    if (!playingRef.current) return step(-1);
    const i = RATES.indexOf(rateRef.current);
    if (i > 0) setRate(RATES[i - 1]);
    else pause();
  };
  const shuttleForward = () => {
    if (!playingRef.current) {
      if (rateRef.current < 1) setRate(1);
      return play();
    }
    const i = RATES.indexOf(rateRef.current);
    if (i >= 0 && i < RATES.length - 1) setRate(RATES[i + 1]);
  };

  const toggleFullscreen = () => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else {
      stage.requestFullscreen().catch(() => {});
      // Keys go to the focused element; outside the fullscreen subtree they'd be lost.
      stage.focus();
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    let handled = true;
    switch (e.key) {
      case " ":
        toggle();
        break;
      case "j":
      case "J":
        shuttleBack();
        break;
      case "k":
      case "K":
        pause();
        break;
      case "l":
      case "L":
        shuttleForward();
        break;
      case "ArrowLeft":
        if (e.shiftKey) seekBy(-1);
        else step(-1);
        break;
      case "ArrowRight":
        if (e.shiftKey) seekBy(1);
        else step(1);
        break;
      case "ArrowUp":
        prevBoundary();
        break;
      case "ArrowDown":
        nextBoundary();
        break;
      case "Home":
        place(0);
        break;
      case "End":
        place(layout.total);
        break;
      case "f":
      case "F":
        toggleFullscreen();
        break;
      default:
        handled = false;
    }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const empty = layout.segments.length === 0;
  const span = layout.beats.find((b) => displayTime >= b.start && displayTime < b.end)
    ?? (current ? layout.beats[current.beatIndex] : undefined);
  const title = span?.beat.on_screen_text?.trim();
  const narration = span?.beat.narration?.trim();
  const currentVideo = currentVideoId != null ? videos[currentVideoId] : undefined;
  const currentName = currentVideo ? fileName(currentVideo.path) : currentVideoId != null ? `video ${currentVideoId}` : "";

  return (
    <div className="lp" ref={rootRef} tabIndex={0} onKeyDown={onKeyDown} aria-label="Live viewer">
      <div
        className={`lp-stage nle-viewer-stage ${fullscreen ? "is-fullscreen" : ""}`}
        ref={stageRef}
        tabIndex={-1}
        onClick={() => {
          rootRef.current?.focus({ preventScroll: true });
          if (!empty) toggle();
        }}
        onDoubleClick={toggleFullscreen}
      >
        <video ref={vidA} className="lp-video" data-on="1" preload="auto" playsInline
          onLoadedMetadata={() => onMeta(0)} onEnded={() => activeRef.current === 0 && playingRef.current && advance()} />
        <video ref={vidB} className="lp-video" data-on="0" preload="auto" playsInline
          onLoadedMetadata={() => onMeta(1)} onEnded={() => activeRef.current === 1 && playingRef.current && advance()} />
        <audio ref={bedRef} preload="auto" onLoadedMetadata={onBedMeta} />

        {empty ? (
          <div className="nle-viewer-empty lp-overlay-center">
            <div className="nle-viewer-empty-icon"><Play size={34} strokeWidth={1.75} aria-hidden /></div>
            <div>This cut has no clips yet.</div>
            <div className="lp-hint">Add clips from the Media tab, or ask Sonar to draft the cut, and they play here live.</div>
          </div>
        ) : (
          <>
            {current && (
              <div className="lp-chip" title={`Clip at playhead · Beat: ${span?.beat.id ?? ""} · ${currentVideo?.path ?? currentName}`}>
                {span?.beat.id && <span className="lp-chip-beat">Beat · {span.beat.id.charAt(0).toUpperCase() + span.beat.id.slice(1)}</span>}
                <span>{currentName}</span>
                {span?.beat.bed && <span className="lp-chip-bed" title="This beat plays a bed's audio under its pictures">bed audio</span>}
              </div>
            )}
            {safeGuides && <div className="lp-safe-guides" style={{ "--lp-aspect": (currentVideo?.width || 16) / (currentVideo?.height || 9) } as React.CSSProperties} aria-hidden="true">
              <div className="lp-action-safe"><span>Action safe · 93%</span></div>
              <div className="lp-title-safe"><span>Title safe · 90%</span></div>
            </div>}
            {showTitles && title && <div className="lp-title">{title}</div>}
            {subtitles && narration && <div className="lp-subtitle">{narration}</div>}
            {waitingFor != null && (
              <div className="lp-overlay-center lp-preparing">
                {failed[waitingFor] ? (
                  <>
                    <div>Could not load {currentName}.</div>
                    <div className="lp-hint">{failed[waitingFor]}</div>
                  </>
                ) : (
                  <>
                    <div>Preparing {currentName}…</div>
                    <div className="lp-hint">
                      {progress?.phase ?? "Checking for a playable copy"}
                      {progress?.fraction != null ? ` · ${Math.floor(progress.fraction * 100)}%` : ""}
                    </div>
                    <progress max={1} value={progress?.fraction ?? undefined} aria-label="Preparing playback copy" />
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <div className="nle-transport lp-transport">
        <div className="nle-transport-buttons">
          <button type="button" className="nle-tbtn lp-minor" onClick={() => place(0)} disabled={empty}
            title="Go to start (Home)" aria-label="Go to start">
            <ArrowLeftToLine size={16} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-tbtn lp-minor" onClick={prevBoundary} disabled={empty}
            title="Previous cut point (Up arrow)" aria-label="Previous cut point">
            <SkipBack size={16} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-tbtn" onClick={() => step(-1)} disabled={empty}
            title="Back one frame (Left arrow; Shift+Left: one second)" aria-label="Back one frame">
            <StepBack size={16} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-tbtn play" onClick={toggle} disabled={empty}
            title={`${playing ? "Pause" : "Play"} (Space or K; J/L shuttle back/forward)`} aria-label={playing ? "Pause" : "Play"}>
            {playing ? <Pause size={16} strokeWidth={2} aria-hidden /> : <Play size={16} strokeWidth={2} aria-hidden />}
          </button>
          <button type="button" className="nle-tbtn" onClick={() => step(1)} disabled={empty}
            title="Forward one frame (Right arrow; Shift+Right: one second)" aria-label="Forward one frame">
            <StepForward size={16} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-tbtn lp-minor" onClick={nextBoundary} disabled={empty}
            title="Next cut point (Down arrow)" aria-label="Next cut point">
            <SkipForward size={16} strokeWidth={2} aria-hidden />
          </button>
          <span className="nle-timecode lp-timecode" title={`Minutes:seconds ·frames at ${+fps.toFixed(3)} fps`}>
            {frameClock(displayTime, fps)}
            <span className="muted lp-total"> / {frameClock(layout.total, fps)}</span>
          </span>
        </div>
        <div className="nle-transport-render">
          {/* Speed, loop, titles and captions: inline when there is room, in the ⋯ menu when not. */}
          <div className="lp-secondary">
            <select
              className="lp-rate"
              value={rate}
              onChange={(e) => setRate(Number(e.target.value))}
              title="Playback speed (J slower, L faster)"
              aria-label="Playback speed"
            >
              {RATES.map((r) => <option key={r} value={r}>{r}×</option>)}
            </select>
            <button type="button" className={`nle-tbtn lp-toggle ${loop ? "on" : ""}`} aria-pressed={loop}
              onClick={() => setLoop((l) => !l)} title={loop ? "Loop on: the cut repeats. Click to turn off." : "Loop off. Click to repeat the cut."}
              aria-label="Loop">
              <Repeat size={16} strokeWidth={2} aria-hidden />
            </button>
            <button type="button" className={`nle-tbtn lp-toggle ${showTitles ? "on" : ""}`} aria-pressed={showTitles}
              onClick={() => onShowTitles(!showTitles)} aria-label="Titles"
              title={showTitles ? "Titles on: shown here and burned into renders and MP4 exports. Click to turn off." : "Titles off: not shown, not burned in. Click to turn on."}>
              <Type size={16} strokeWidth={2} aria-hidden />
            </button>
            <button type="button" className={`nle-tbtn lp-toggle ${subtitles ? "on" : ""}`} aria-pressed={subtitles}
              onClick={() => onShowNarration(!subtitles)} aria-label="Narration subtitles"
              title={subtitles ? "Narration subtitles on: shown here and burned into renders and MP4 exports. Click to turn off." : "Narration subtitles off. Click to show and burn them in."}>
              <Captions size={16} strokeWidth={2} aria-hidden />
            </button>
          </div>
          <details className="nle-menu up lp-more">
            <summary className="nle-tbtn" title="Speed, loop, titles and subtitles" aria-label="More playback options">
              <Ellipsis size={16} strokeWidth={2} aria-hidden />
            </summary>
            <div className="nle-menu-list">
              <div className="nle-menu-heading">Speed</div>
              <div className="lp-more-rates">
                {RATES.map((r) => (
                  <button key={r} type="button" className={rate === r ? "on" : ""} aria-pressed={rate === r} onClick={() => setRate(r)}>
                    {r}×
                  </button>
                ))}
              </div>
              <div className="nle-menu-sep" />
              <button type="button" role="menuitemcheckbox" aria-checked={loop} onClick={() => setLoop((l) => !l)}>
                <Repeat size={14} strokeWidth={2} aria-hidden /> Loop <span className="lp-more-state">{loop ? "On" : "Off"}</span>
              </button>
              <button type="button" role="menuitemcheckbox" aria-checked={showTitles} onClick={() => onShowTitles(!showTitles)}>
                <Type size={14} strokeWidth={2} aria-hidden /> Titles <span className="lp-more-state">{showTitles ? "On" : "Off"}</span>
              </button>
              <button type="button" role="menuitemcheckbox" aria-checked={subtitles} onClick={() => onShowNarration(!subtitles)}>
                <Captions size={14} strokeWidth={2} aria-hidden /> Narration subtitles <span className="lp-more-state">{subtitles ? "On" : "Off"}</span>
              </button>
              <div className="small muted lp-more-note">Titles and subtitles shown here are burned into renders.</div>
            </div>
          </details>
          <button type="button" className={`nle-tbtn lp-toggle ${safeGuides ? "on" : ""}`} aria-pressed={safeGuides}
            onClick={() => setSafeGuides(!safeGuides)} aria-label="Safe area guides"
            title="Show action-safe (93%) and title-safe (90%) guides; preview only">
            <Scan size={16} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-tbtn" onClick={toggleFullscreen} disabled={empty}
            title={fullscreen ? "Exit fullscreen (F or Esc)" : "Fullscreen (F; double-click the picture)"} aria-label="Fullscreen">
            {fullscreen ? <Minimize size={16} strokeWidth={2} aria-hidden /> : <Maximize size={16} strokeWidth={2} aria-hidden />}
          </button>
          {extra}
        </div>
      </div>
    </div>
  );
}
