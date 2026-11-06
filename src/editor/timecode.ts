import { useSyncExternalStore } from "react";
import type { Fps } from "../api";

/**
 * Timecode helpers shared by the live viewer, the inspector and the timeline, so every place that
 * shows a time on the cut's clock agrees on frames at the script's fps (30 when it has none).
 */

export const fpsValue = (fps: Fps | undefined): number => {
  if (typeof fps === "number" && fps > 0) return fps;
  if (fps && typeof fps === "object" && fps.num > 0 && fps.den > 0) return fps.num / fps.den;
  return 30;
};

/** Whole seconds and the frame inside the last second. */
function split(s: number, fps: number): { whole: number; frames: number } {
  const t = Math.max(0, s);
  const whole = Math.floor(t + 1e-6);
  const frames = Math.min(Math.ceil(fps) - 1, Math.floor((t - whole) * fps + 1e-6));
  return { whole, frames: Math.max(0, frames) };
}

/** Editorial position: minutes:seconds with an explicitly labelled frame suffix. */
export const frameClock = (s: number, fps: number): string => {
  const { whole, frames } = split(s, fps);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")} ·${String(frames).padStart(2, "0")}f`;
};

/** hh:mm:ss:ff, the broadcast (SMPTE, non-drop) form. */
export const smpteClock = (s: number, fps: number): string => {
  const { whole, frames } = split(s, fps);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(Math.floor(whole / 3600))}:${p(Math.floor((whole % 3600) / 60))}:${p(whole % 60)}:${p(frames)}`;
};

/** Snaps a time onto the nearest frame boundary. */
export const snapToFrame = (s: number, fps: number): number => Math.round(s * fps) / fps;

/**
 * Reads what someone typed into a timecode field, in seconds; null when it isn't a time.
 *
 * - `1:04:12` → minutes:seconds:frames
 * - `1:04.12` / `0:01:04.12` → minutes:seconds.frames (what the fields show)
 * - `00:01:04:12` → SMPTE hh:mm:ss:ff (four parts)
 * - `64.5` / `64` → plain seconds (no colon: the decimal is a fraction of a second)
 * - `90f` → frames
 */
export function parseTimecode(raw: string, fps: number): number | null {
  const text = raw.trim();
  const editorial = /^(\d+):([0-5]\d)\s*·\s*(\d+)f$/i.exec(text);
  if (editorial) {
    const [, m, sec, f] = editorial.map(Number);
    return f < Math.ceil(fps) ? m * 60 + sec + f / fps : null;
  }
  if (!text) return null;
  const framesOnly = /^(\d+)\s*f$/i.exec(text);
  if (framesOnly) return Number(framesOnly[1]) / fps;
  if (!text.includes(":")) {
    const n = Number(text);
    return Number.isFinite(n) && n >= 0 ? n : null;
  }
  const parts = text.split(":");
  if (parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) return null;
  if (parts.length === 4) {
    const [h, m, s, f] = parts.map(Number);
    if (f >= Math.ceil(fps)) return null;
    return h * 3600 + m * 60 + s + f / fps;
  }
  if (parts.length > 3) return null;
  if (parts.length === 3 && !text.includes(".")) {
    const [m, s, f] = parts.map(Number);
    if (s >= 60 || f >= Math.ceil(fps)) return null;
    return m * 60 + s + f / fps;
  }
  // The last part may carry frames after a dot: "04.12" is 4 s and 12 frames.
  const last = parts[parts.length - 1];
  const [secText, frameText] = last.split(".");
  const frames = frameText ? Number(frameText) : 0;
  if (frames >= Math.ceil(fps)) return null;
  const head = parts.slice(0, -1).map(Number);
  const [h, m] = head.length === 2 ? head : [0, head[0] ?? 0];
  return h * 3600 + m * 60 + Number(secText) + frames / fps;
}

/* ---- The viewer's timecode style, remembered across sessions and shared by every transport ---- */

export type TimecodeStyle = "frames" | "smpte";
const KEY = "ghostreel.timecodeStyle";
const listeners = new Set<() => void>();
let style: TimecodeStyle = (() => {
  try {
    return localStorage.getItem(KEY) === "smpte" ? "smpte" : "frames";
  } catch {
    return "frames";
  }
})();

export function setTimecodeStyle(next: TimecodeStyle) {
  style = next;
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* Private mode: the choice lasts for this session. */
  }
  for (const l of listeners) l();
}

export function useTimecodeStyle(): TimecodeStyle {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => style,
  );
}

export const formatTimecode = (s: number, fps: number, how: TimecodeStyle): string =>
  how === "smpte" ? smpteClock(s, fps) : frameClock(s, fps);
