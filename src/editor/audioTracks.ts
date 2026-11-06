import { useEffect, useMemo, useState } from "react";
import { audioLevels, audioTrackProxy, mediaUrl, videoAudioTracks, type AudioTrackInfo } from "../api";

/**
 * Shared, cached access to a file's audio tracks, their levels and their playable proxies, so the
 * timeline, the inspector and the live viewer ask the backend once per file. Every loader resolves
 * to `null` on failure (an older backend without these commands, a file ffprobe can't read): the
 * callers then fall back to one track per file, which is what GhostReel did before.
 */

/** Lane monitoring in the live viewer: not saved, not rendered. */
export interface TrackMonitor {
  muted: number[];
  solo: number[];
}

/** Below this a track is shown as "silent": room tone on an unplugged input sits well under it. */
export const SILENT_DB = -60;

const infoCache = new Map<number, Promise<AudioTrackInfo[] | null>>();
const infoDone = new Map<number, AudioTrackInfo[] | null>();

export function loadAudioTracks(videoId: number): Promise<AudioTrackInfo[] | null> {
  let p = infoCache.get(videoId);
  if (!p) {
    p = videoAudioTracks(videoId)
      .then((list) => (Array.isArray(list) ? [...list].sort((a, b) => a.index - b.index) : null))
      .catch(() => null)
      .then((v) => {
        infoDone.set(videoId, v);
        return v;
      });
    infoCache.set(videoId, p);
  }
  return p;
}

/** A file's tracks if already loaded (undefined when not yet, null when they couldn't be). */
export function cachedAudioTracks(videoId: number): AudioTrackInfo[] | null | undefined {
  return infoDone.has(videoId) ? infoDone.get(videoId) ?? null : undefined;
}

/** Track lists for these videos; a video is absent until loaded and `null` when it couldn't be. */
export function useAudioTracks(videoIds: number[]): Record<number, AudioTrackInfo[] | null> {
  const key = [...new Set(videoIds)].sort((a, b) => a - b).join(",");
  const [map, setMap] = useState<Record<number, AudioTrackInfo[] | null>>(() => {
    const out: Record<number, AudioTrackInfo[] | null> = {};
    for (const id of videoIds) if (infoDone.has(id)) out[id] = infoDone.get(id) ?? null;
    return out;
  });
  useEffect(() => {
    let alive = true;
    const ids = key ? key.split(",").map(Number) : [];
    for (const id of ids) {
      if (infoDone.has(id)) {
        const v = infoDone.get(id) ?? null;
        setMap((m) => (id in m && m[id] === v ? m : { ...m, [id]: v }));
        continue;
      }
      void loadAudioTracks(id).then((v) => {
        if (alive) setMap((m) => ({ ...m, [id]: v }));
      });
    }
    return () => {
      alive = false;
    };
  }, [key]);
  return map;
}

// Levels are keyed to a tenth of a second: a trim nudge re-measures, jitter doesn't.
const levelKey = (videoId: number, inS: number, outS: number) => `${videoId}:${inS.toFixed(1)}:${outS.toFixed(1)}`;
const levelCache = new Map<string, Promise<number[] | null>>();
const levelDone = new Map<string, number[] | null>();
// Once the backend says it has no such command, stop asking for every block.
let levelsUnsupported = false;

// Each measurement is an ffmpeg run; a long cut must not start dozens at once.
const MAX_LEVEL_JOBS = 2;
let levelJobs = 0;
const levelQueue: (() => void)[] = [];
function limited<T>(job: () => Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const run = () => {
      levelJobs++;
      job()
        .then(resolve, reject)
        .finally(() => {
          levelJobs--;
          levelQueue.shift()?.();
        });
    };
    if (levelJobs < MAX_LEVEL_JOBS) run();
    else levelQueue.push(run);
  });
}

function loadLevels(videoId: number, inS: number, outS: number): Promise<number[] | null> {
  const k = levelKey(videoId, inS, outS);
  let p = levelCache.get(k);
  if (!p) {
    p = levelsUnsupported
      ? Promise.resolve(null)
      : limited(() => (levelsUnsupported ? Promise.resolve(null) : audioLevels(videoId, inS, outS)))
          .then((v) => (Array.isArray(v) ? v : null))
          .catch((err) => {
            if (/unknown command|not found|not allowed/i.test(String(err))) levelsUnsupported = true;
            return null;
          });
    p = p.then((v) => {
      levelDone.set(k, v);
      return v;
    });
    levelCache.set(k, p);
  }
  return p;
}

export type LevelRequest = { videoId: number; inS: number; outS: number };

/**
 * Per-track mean levels for source ranges, measured lazily and after the ranges stop changing
 * for a moment (a trim drag would otherwise start one measurement per pixel). The returned lookup
 * gives `undefined` while unknown and `null` when it couldn't be measured.
 */
export function useAudioLevels(requests: LevelRequest[], delayMs = 350) {
  const [, setTick] = useState(0);
  const wanted = useMemo(() => {
    const seen = new Map<string, LevelRequest>();
    for (const r of requests) if (r.outS > r.inS) seen.set(levelKey(r.videoId, r.inS, r.outS), r);
    return seen;
  }, [requests]);
  const wantedKey = [...wanted.keys()].sort().join("|");
  useEffect(() => {
    const missing = [...wanted.entries()].filter(([k]) => !levelDone.has(k));
    if (missing.length === 0) return;
    let alive = true;
    const timer = setTimeout(() => {
      for (const [, r] of missing) {
        void loadLevels(r.videoId, r.inS, r.outS).then(() => {
          if (alive) setTick((t) => t + 1);
        });
      }
    }, delayMs);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // `wantedKey` stands for `wanted`'s contents.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantedKey, delayMs]);
  return (videoId: number, inS: number, outS: number): number[] | null | undefined => {
    const k = levelKey(videoId, inS, outS);
    return levelDone.has(k) ? levelDone.get(k) ?? null : undefined;
  };
}

/** The track GhostReel picks for a file when a clip doesn't choose. */
export function autoTrack(infos: AudioTrackInfo[] | null | undefined): number {
  return infos?.find((t) => t.auto)?.index ?? infos?.[0]?.index ?? 0;
}

/** How many lanes this file fills; unknown files count as one, like before multi-track. */
export function trackCount(infos: AudioTrackInfo[] | null | undefined): number {
  return infos ? infos.length : 1;
}

/** The tracks that actually play for a clip or bed (ignoring `audio: "mute"`). */
export function effectiveTracks(tracks: number[] | undefined, infos: AudioTrackInfo[] | null | undefined): number[] {
  return tracks ?? [autoTrack(infos)];
}

/** Flip one track in a clip's selection; the first touch starts from the automatic track. */
export function toggleTrack(
  tracks: number[] | undefined,
  infos: AudioTrackInfo[] | null | undefined,
  track: number,
): number[] {
  const cur = new Set(effectiveTracks(tracks, infos));
  if (cur.has(track)) cur.delete(track);
  else cur.add(track);
  return [...cur].sort((a, b) => a - b);
}

/** Whether a lane is heard under the monitor's mute/solo state. */
export function audible(track: number, monitor: TrackMonitor | undefined): boolean {
  if (!monitor) return true;
  if (monitor.solo.length > 0) return monitor.solo.includes(track);
  return !monitor.muted.includes(track);
}

export function dbLabel(db: number): string {
  return db <= SILENT_DB ? "silent" : `${Math.round(db)} dB`;
}

/** 0..1 for a level bar: -60 dB and below is empty, 0 dB is full. */
export function levelFraction(db: number): number {
  return Math.min(1, Math.max(0, (db - SILENT_DB) / -SILENT_DB));
}

export function trackName(index: number): string {
  return `A${index + 1}`;
}

/** e.g. "mono · pcm_s24le" or "Lav · stereo". */
export function trackDescription(t: AudioTrackInfo): string {
  const ch = t.channels === 1 ? "mono" : t.channels === 2 ? "stereo" : `${t.channels} ch`;
  return [t.label?.trim() || null, ch].filter(Boolean).join(" · ");
}

const proxyCache = new Map<string, Promise<string>>();

/** A playable URL for one track; the first ask extracts it, so it can take a moment. */
export function trackProxyUrl(videoId: number, track: number): Promise<string> {
  const k = `${videoId}:${track}`;
  let p = proxyCache.get(k);
  if (!p) {
    p = audioTrackProxy(videoId, track).then((path) => mediaUrl(path));
    proxyCache.set(k, p);
    // A failure may be transient (or the backend not updated yet); let a later ask retry.
    p.catch(() => setTimeout(() => proxyCache.delete(k), 10_000));
  }
  return p;
}
