import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Crosshair } from "lucide-react";
import { frameClock, parseTimecode, snapToFrame } from "./timecode";

interface TimecodeFieldProps {
  label: string;
  /** Seconds in the source file. */
  value: number;
  fps: number;
  onChange: (seconds: number) => void;
  /** The source time under the playhead, or null when the playhead isn't over this clip. */
  playheadValue: number | null;
  /** The timeline key that does the same, for the tooltip ("i" / "o"). */
  playheadKey?: string;
  disabled?: boolean;
}

/** Pixels of drag per frame when scrubbing the label. */
const PX_PER_FRAME = 4;

/**
 * An In/Out point as an editor expects it: m:ss:ff at the cut's frame rate, typed, nudged a frame
 * at a time, scrubbed by dragging the label, or taken from the playhead.
 */
export default function TimecodeField({
  label,
  value,
  fps,
  onChange,
  playheadValue,
  playheadKey,
  disabled,
}: TimecodeFieldProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const drag = useRef<{ x: number; start: number; frames: number } | null>(null);
  const frame = 1 / fps;

  const commit = () => {
    if (draft == null) return;
    const v = parseTimecode(draft, fps);
    setDraft(null);
    setInvalid(false);
    if (v != null) onChange(snapToFrame(v, fps));
  };

  const nudge = (frames: number) => onChange(snapToFrame(value + frames * frame, fps));

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
      (e.target as HTMLInputElement).blur();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setDraft(null);
      setInvalid(false);
      (e.target as HTMLInputElement).blur();
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      setDraft(null);
      nudge((e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 10 : 1));
    }
  };

  const onScrubStart = (e: PointerEvent<HTMLSpanElement>) => {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, start: value, frames: 0 };
  };
  const onScrubMove = (e: PointerEvent<HTMLSpanElement>) => {
    const d = drag.current;
    if (!d) return;
    const frames = Math.round((e.clientX - d.x) / PX_PER_FRAME) * (e.shiftKey ? 10 : 1);
    if (frames === d.frames) return;
    d.frames = frames;
    onChange(snapToFrame(d.start + frames * frame, fps));
  };
  const onScrubEnd = (e: PointerEvent<HTMLSpanElement>) => {
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const fpsText = +fps.toFixed(3);

  return (
    <div className={`tc-field ${disabled ? "disabled" : ""}`}>
      <span
        className="tc-label"
        onPointerDown={onScrubStart}
        onPointerMove={onScrubMove}
        onPointerUp={onScrubEnd}
        onPointerCancel={onScrubEnd}
        title={`Drag left or right to scrub ${label} by frames (Shift: 10 frames)`}
      >
        {label}
      </span>
      <input
        className={`tc-input ${invalid ? "invalid" : ""}`}
        value={draft ?? frameClock(value, fps)}
        disabled={disabled}
        spellCheck={false}
        inputMode="decimal"
        aria-label={`${label} point`}
        title={`Minutes:seconds ·frames at ${fpsText} fps. Type 1:04 ·12f or 1:04:12. Up/Down: one frame.`}
        onFocus={(e) => e.currentTarget.select()}
        onChange={(e) => {
          setDraft(e.target.value);
          setInvalid(parseTimecode(e.target.value, fps) == null);
        }}
        onBlur={commit}
        onKeyDown={onKeyDown}
      />
      <span className="tc-buttons">
        <button type="button" className="tc-btn" disabled={disabled} onClick={() => nudge(-1)} title={`${label} one frame earlier`}>
          −1f
        </button>
        <button type="button" className="tc-btn" disabled={disabled} onClick={() => nudge(1)} title={`${label} one frame later`}>
          +1f
        </button>
        <button
          type="button"
          className="tc-btn tc-playhead"
          disabled={disabled || playheadValue == null}
          onClick={() => playheadValue != null && onChange(snapToFrame(playheadValue, fps))}
          aria-label={`Set ${label} from the playhead`}
          title={
            playheadValue == null
              ? `Put the playhead over this clip to set ${label} from it`
              : `Set ${label} to the playhead${playheadKey ? ` (${playheadKey} on the timeline)` : ""}`
          }
        >
          <Crosshair size={14} strokeWidth={2} aria-hidden /> Set to playhead
        </button>
      </span>
    </div>
  );
}
