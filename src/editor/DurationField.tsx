import { useState } from "react";
import { formatDuration } from "../timeFormat";

export function parseDuration(text: string): number | null {
  const match = /^(\d+):([0-5]\d)$/.exec(text.trim());
  return match ? Number(match[1]) * 60 + Number(match[2]) : null;
}

/** Keep incomplete typing local; only valid m:ss values reach the saved script. */
export default function DurationField({ value, onChange, min = 0, max = Infinity, allowEmpty = false }: {
  value?: number; onChange: (value: number | undefined) => void;
  min?: number; max?: number; allowEmpty?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const parsed = draft === null ? value : parseDuration(draft);
  const valid = draft === null || (allowEmpty && draft.trim() === "") ||
    (parsed != null && parsed >= min && parsed <= max);
  const commit = () => {
    if (!valid) return;
    if (draft !== null) onChange(draft.trim() === "" ? undefined : parseDuration(draft)!);
    setDraft(null);
  };
  return <input type="text" inputMode="text" placeholder="1:00" aria-label="Target length (m:ss)"
    title="Minutes:seconds, for example 1:00" aria-invalid={!valid}
    value={draft ?? (value == null ? "" : formatDuration(value))}
    onChange={(e) => setDraft(e.target.value)} onBlur={commit}
    onKeyDown={(e) => {
      if (e.key === "Enter") { e.preventDefault(); commit(); }
      if (e.key === "Escape") { e.preventDefault(); setDraft(null); }
    }} />;
}
