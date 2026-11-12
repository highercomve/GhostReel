import { etaText, type Progress } from "../api";

/** "42% · about 2 min left", or what it is doing when nothing can be counted. */
export function progressText(p: Progress): string {
  if (p.indeterminate) return `${p.label || "Working"}…`;
  const pct = Math.round(p.fraction * 100);
  return `${pct}%${p.eta_secs != null && p.fraction < 1 ? ` · ${etaText(p.eta_secs)} left` : ""}`;
}

/** A running task's bar and phase line ("Transcribing · 3 of 12 · a.mp4"). */
export default function ProgressBar({ progress: p }: { progress: Progress | null | undefined }) {
  const pct = Math.round((p?.fraction ?? 0) * 100);
  return (
    <>
      <div className={`meter big${p?.indeterminate ? " indeterminate" : ""}`}>
        <div style={p?.indeterminate ? undefined : { width: `${Math.max(2, pct)}%` }} />
      </div>
      {p && (
        <div className="muted small">
          {p.label || p.phase}
          {p.phase_total > 0 ? ` · ${p.phase_done} of ${p.phase_total}` : ""}
          {p.current ? ` · ${p.current.split(/[\\/]/).pop()}` : ""}
        </div>
      )}
    </>
  );
}
