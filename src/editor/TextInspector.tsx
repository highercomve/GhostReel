import { AlignCenter, AlignLeft, AlignRight, Bold, CaseUpper, Eye, EyeOff, RotateCcw, Square, Type } from "lucide-react";
import type { Cue, ResolvedStyle, TextStyle } from "../api";
import TimecodeField from "./TimecodeField";
import { FONTS, presetPosition, type Preset } from "./textOverlay";
import { formatDuration as clock } from "../timeFormat";

interface Common {
  style: ResolvedStyle;
  x: number;
  y: number;
  portrait: boolean;
  fps: number;
  /** Cut time under the playhead, for the "set to playhead" buttons. */
  playhead: number;
  disabled?: boolean;
  onStyle: (patch: Partial<TextStyle>) => void;
  onPosition: (x: number, y: number) => void;
  onResetStyle: () => void;
}

interface TitleProps extends Common {
  kind: "title";
  text: string;
  beatId: string;
  beatStart: number;
  beatDuration: number;
  startRel: number;
  endRel: number;
  onText: (text: string) => void;
  onTimes: (startRel: number | null, endRel: number | null) => void;
  onApplyAll: () => void;
}

interface CaptionProps extends Common {
  kind: "caption";
  cue: Cue | null;
  maxChars: number;
  maxLines: number;
  editCount: number;
  onCueText: (text: string) => void;
  onCueTimes: (start: number, end: number) => void;
  onCueHide: (hidden: boolean) => void;
  onCueReset: () => void;
  onResetAllEdits: () => void;
  onLayout: (maxChars: number, maxLines: number) => void;
}

const PRESETS: { id: Preset; label: string }[] = [
  { id: "top", label: "Top" },
  { id: "center", label: "Centre" },
  { id: "bottom", label: "Bottom" },
  { id: "lower", label: "Lower third" },
];

const pct = (v: number) => Math.round(v * 1000) / 10;

/** The Text tab: how a title or the speech captions look, where they sit, and when they show. */
export default function TextInspector(props: TitleProps | CaptionProps) {
  const { style, x, y, portrait, fps, playhead, disabled, onStyle, onPosition, onResetStyle } = props;

  const preset = (p: Preset) => {
    const pos = presetPosition(props.kind, p, portrait);
    if (pos.align !== style.align) onStyle({ align: pos.align });
    onPosition(pos.x, pos.y);
  };

  return (
    <section className="nle-insp-section insp-text">
      {props.kind === "title" ? (
        <>
          <div className="insp-beat-head">
            <span className="insp-crumb-id">{props.beatId}</span>
            <span className="small muted">Title · beat runs {clock(props.beatDuration)}</span>
          </div>
          <label className="nle-insp-field insp-field-title">
            <span className="insp-field-label">
              <Type size={13} strokeWidth={2} aria-hidden /> On-screen text
            </span>
            <input value={props.text} disabled={disabled} onChange={(e) => props.onText(e.target.value)} placeholder="Title or lower third…" />
          </label>
          <div className="nle-insp-subtitle">When</div>
          <div className="insp-timecodes">
            <TimecodeField
              label="In"
              value={props.beatStart + props.startRel}
              fps={fps}
              disabled={disabled}
              playheadValue={playhead >= props.beatStart && playhead <= props.beatStart + props.beatDuration ? playhead : null}
              onChange={(v) => props.onTimes(Math.min(Math.max(0, v - props.beatStart), props.endRel - 0.1), props.endRel)}
            />
            <TimecodeField
              label="Out"
              value={props.beatStart + props.endRel}
              fps={fps}
              disabled={disabled}
              playheadValue={playhead >= props.beatStart && playhead <= props.beatStart + props.beatDuration ? playhead : null}
              onChange={(v) => props.onTimes(props.startRel, Math.max(props.startRel + 0.1, Math.min(v - props.beatStart, props.beatDuration)))}
            />
            <div className="insp-tc-facts">
              <span>
                On screen <strong>{clock(props.endRel - props.startRel)}</strong>
              </span>
              <button type="button" className="ghost small" disabled={disabled} onClick={() => props.onTimes(null, null)}
                title="Show the title for the whole beat">
                Whole beat
              </button>
            </div>
          </div>
        </>
      ) : (
        <CaptionCue {...props} />
      )}

      <div className="nle-insp-subtitle">{props.kind === "title" ? "Look of this title" : "Look of every caption"}</div>
      <label className="nle-insp-field">
        Font
        <select value={style.font} disabled={disabled} onChange={(e) => onStyle({ font: e.target.value })}>
          {(FONTS.includes(style.font) ? FONTS : [style.font, ...FONTS]).map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </label>
      <div className="insp-text-grid">
        <label className="nle-insp-field">
          Size (% of height)
          <input type="number" min={1} max={30} step={0.1} value={pct(style.size)} disabled={disabled}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (v > 0) onStyle({ size: v / 100 });
            }} />
        </label>
        <div className="nle-insp-field">
          Weight and case
          <span className="insp-seg" role="group" aria-label="Bold and uppercase">
            <button type="button" aria-pressed={style.bold} className={style.bold ? "on" : ""} disabled={disabled}
              onClick={() => onStyle({ bold: !style.bold })} title="Bold">
              <Bold size={14} strokeWidth={2} aria-hidden />
            </button>
            <button type="button" aria-pressed={style.uppercase} className={style.uppercase ? "on" : ""} disabled={disabled}
              onClick={() => onStyle({ uppercase: !style.uppercase })} title="Uppercase">
              <CaseUpper size={14} strokeWidth={2} aria-hidden />
            </button>
          </span>
        </div>
        <label className="nle-insp-field">
          Text colour
          <input type="color" value={style.color} disabled={disabled} onChange={(e) => onStyle({ color: e.target.value })} />
        </label>
        <div className="nle-insp-field">
          Alignment
          <span className="insp-seg" role="group" aria-label="Alignment">
            {(["left", "center", "right"] as const).map((a) => (
              <button key={a} type="button" aria-pressed={style.align === a} className={style.align === a ? "on" : ""} disabled={disabled}
                onClick={() => onStyle({ align: a })} title={a === "center" ? "Centre" : a === "left" ? "Left" : "Right"}>
                {a === "left" ? <AlignLeft size={14} strokeWidth={2} aria-hidden /> : a === "right" ? <AlignRight size={14} strokeWidth={2} aria-hidden /> : <AlignCenter size={14} strokeWidth={2} aria-hidden />}
              </button>
            ))}
          </span>
        </div>
      </div>

      <div className="insp-switch-row">
        <span className="insp-switch-label">
          <Square size={14} strokeWidth={2} aria-hidden /> Box behind text
        </span>
        <button type="button" role="switch" aria-checked={style.box} aria-label="Box behind text" disabled={disabled}
          className={`insp-switch ${style.box ? "on" : ""}`} onClick={() => onStyle({ box: !style.box })}>
          <span className="insp-switch-knob" />
        </button>
        <span className={`insp-switch-state ${style.box ? "" : "off"}`}>{style.box ? "On" : "Off"}</span>
      </div>
      {style.box ? (
        <div className="insp-text-grid">
          <label className="nle-insp-field">
            Box colour
            <input type="color" value={style.box_color} disabled={disabled} onChange={(e) => onStyle({ box_color: e.target.value })} />
          </label>
          <label className="nle-insp-field">
            Box opacity {Math.round(style.box_opacity * 100)}%
            <input type="range" min={0} max={1} step={0.05} value={style.box_opacity} disabled={disabled}
              onChange={(e) => onStyle({ box_opacity: Number(e.target.value) })} />
          </label>
        </div>
      ) : (
        <div className="insp-text-grid">
          <label className="nle-insp-field">
            Outline colour
            <input type="color" value={style.outline_color} disabled={disabled} onChange={(e) => onStyle({ outline_color: e.target.value })} />
          </label>
          <label className="nle-insp-field">
            Outline {Math.round(style.outline * 100)}%
            <input type="range" min={0} max={0.3} step={0.01} value={style.outline} disabled={disabled}
              onChange={(e) => onStyle({ outline: Number(e.target.value) })} />
          </label>
          <label className="nle-insp-field">
            Shadow {Math.round(style.shadow * 100)}%
            <input type="range" min={0} max={0.3} step={0.01} value={style.shadow} disabled={disabled}
              onChange={(e) => onStyle({ shadow: Number(e.target.value) })} />
          </label>
        </div>
      )}
      <label className="nle-insp-field">
        Line width {Math.round(style.max_width * 100)}% of the frame
        <input type="range" min={0.2} max={1} step={0.01} value={style.max_width} disabled={disabled}
          onChange={(e) => onStyle({ max_width: Number(e.target.value) })} />
      </label>

      <div className="nle-insp-subtitle">Where</div>
      <div className="insp-seg insp-presets" role="group" aria-label="Position presets">
        {PRESETS.map((p) => (
          <button key={p.id} type="button" disabled={disabled} onClick={() => preset(p.id)}>{p.label}</button>
        ))}
      </div>
      <div className="insp-text-grid">
        <label className="nle-insp-field">
          X (% across)
          <input type="number" min={0} max={100} step={0.5} value={pct(x)} disabled={disabled}
            onChange={(e) => onPosition(Math.min(1, Math.max(0, Number(e.target.value) / 100)), y)} />
        </label>
        <label className="nle-insp-field">
          Y (% down)
          <input type="number" min={0} max={100} step={0.5} value={pct(y)} disabled={disabled}
            onChange={(e) => onPosition(x, Math.min(1, Math.max(0, Number(e.target.value) / 100)))} />
        </label>
      </div>
      <div className="small muted">
        {props.kind === "title" ? "Y is the middle of the title. " : "Y is the bottom of the caption, so two lines grow upwards. "}
        Drag the text in the viewer to move it; Shift-drag or its corner resizes it.
      </div>

      <div className="nle-insp-actions">
        {props.kind === "title" && (
          <button type="button" className="small" disabled={disabled} onClick={props.onApplyAll}
            title="Make this look the style of every title in the cut">
            Apply to all titles
          </button>
        )}
        <button type="button" className="ghost small" disabled={disabled} onClick={onResetStyle}
          title={props.kind === "title" ? "Back to the cut's title style and the default place" : "Back to the default caption look and place"}>
          <RotateCcw size={13} strokeWidth={2} aria-hidden /> Reset look and place
        </button>
      </div>
    </section>
  );
}

function CaptionCue(props: CaptionProps) {
  const { cue, fps, playhead, disabled } = props;
  return (
    <>
      <div className="insp-beat-head">
        <span className="insp-crumb-id">Caption</span>
        <span className="small muted">
          {cue ? `${clock(cue.start_s)}–${clock(cue.end_s)}${cue.edited ? " · edited" : ""}${cue.hidden ? " · hidden" : ""}` : "No caption selected"}
        </span>
      </div>
      {cue && (
        <>
          <label className="nle-insp-field">
            Text
            <textarea
              key={`${cue.video_id}:${cue.src_start_s}`}
              rows={2}
              defaultValue={cue.text}
              disabled={disabled}
              onBlur={(e) => {
                const v = e.target.value.trim();
                if (v && v !== cue.text) props.onCueText(v);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.blur();
                }
              }}
            />
          </label>
          <div className="insp-timecodes">
            <TimecodeField
              label="In"
              value={cue.start_s}
              fps={fps}
              disabled={disabled}
              playheadValue={cue.min_s != null && cue.max_s != null && playhead >= cue.min_s && playhead <= cue.max_s ? playhead : null}
              onChange={(v) => props.onCueTimes(Math.min(v, cue.end_s - 0.1), cue.end_s)}
            />
            <TimecodeField
              label="Out"
              value={cue.end_s}
              fps={fps}
              disabled={disabled}
              playheadValue={cue.min_s != null && cue.max_s != null && playhead >= cue.min_s && playhead <= cue.max_s ? playhead : null}
              onChange={(v) => props.onCueTimes(cue.start_s, Math.max(v, cue.start_s + 0.1))}
            />
          </div>
          <div className="nle-insp-actions">
            <button type="button" className="small" disabled={disabled} onClick={() => props.onCueHide(!cue.hidden)}
              title={cue.hidden ? "Show this caption again" : "Leave this caption out of the cut (H on the timeline)"}>
              {cue.hidden ? <Eye size={13} strokeWidth={2} aria-hidden /> : <EyeOff size={13} strokeWidth={2} aria-hidden />}
              {cue.hidden ? " Show" : " Hide"}
            </button>
            <button type="button" className="ghost small" disabled={disabled || (!cue.edited && !cue.hidden)} onClick={props.onCueReset}
              title="Back to the transcript's words at their own time">
              <RotateCcw size={13} strokeWidth={2} aria-hidden /> Reset to transcript
            </button>
          </div>
        </>
      )}
      <div className="nle-insp-subtitle">Lines</div>
      <div className="insp-text-grid">
        <label className="nle-insp-field">
          Characters a line
          <input type="number" min={8} max={120} step={1} value={props.maxChars} disabled={disabled}
            onChange={(e) => Number(e.target.value) >= 8 && props.onLayout(Number(e.target.value), props.maxLines)} />
        </label>
        <label className="nle-insp-field">
          Lines a caption
          <input type="number" min={1} max={6} step={1} value={props.maxLines} disabled={disabled}
            onChange={(e) => Number(e.target.value) >= 1 && props.onLayout(props.maxChars, Number(e.target.value))} />
        </label>
      </div>
      {props.editCount > 0 && (
        <button type="button" className="ghost small" disabled={disabled} onClick={props.onResetAllEdits}
          title="Undo every caption edit in this cut: all back to the transcript">
          <RotateCcw size={13} strokeWidth={2} aria-hidden /> Reset all {props.editCount} caption {props.editCount === 1 ? "edit" : "edits"}
        </button>
      )}
    </>
  );
}
