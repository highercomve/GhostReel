/** Shared editorial durations: m:ss; source positions use editor/timecode for frames. */
export function formatDuration(seconds: number): string {
  const total = Number.isFinite(seconds) ? Math.max(0, Math.round(seconds)) : 0;
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** Normalize generated duration diagnostics without changing saved scripts or prompts. */
export function formatDurationText(text: string): string {
  return text.replace(/\b(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b/g, (_, value: string) => formatDuration(Number(value)));
}
