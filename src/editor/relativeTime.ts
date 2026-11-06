import { useEffect, useState } from "react";

/** Seconds or milliseconds since the epoch, as the backend sends either. */
function toMs(timestamp: number): number {
  return timestamp < 1e11 ? timestamp * 1000 : timestamp;
}

/** "just now", "4 min ago", "3 h ago", "2 d ago", then a date. */
export function relativeTime(timestamp: number, now: number = Date.now()): string {
  const secs = Math.max(0, Math.floor((now - toMs(timestamp)) / 1000));
  if (secs < 45) return "just now";
  if (secs < 3600) return `${Math.max(1, Math.round(secs / 60))} min ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)} h ago`;
  if (secs < 7 * 86400) return `${Math.floor(secs / 86400)} d ago`;
  return new Date(toMs(timestamp)).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

/** The full local date and time, for tooltips. */
export function fullTime(timestamp: number): string {
  return new Date(toMs(timestamp)).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** The current time, refreshed once a minute so "4 min ago" stays true on a screen left open. */
export function useMinuteNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}
