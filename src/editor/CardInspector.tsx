import { useState } from "react";
import { Square, Trash2 } from "lucide-react";
import type { Beat, CardStyle, Script, SectionCard } from "../api";
import { CARD_S_RANGE, cardBackground, cardSeconds, defaultCardSeconds, hasCard } from "./cards";

/** Seconds typed as a plain number. A half-typed value ("1.") stays local; a whole one is passed on
 * as it is typed, and empty means "use the default". */
function SecondsField({
  value,
  placeholder,
  label,
  disabled,
  onChange,
}: {
  value?: number;
  placeholder: number;
  label: string;
  disabled?: boolean;
  onChange: (value: number | undefined) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="number"
      inputMode="decimal"
      min={CARD_S_RANGE[0]}
      max={CARD_S_RANGE[1]}
      step={0.5}
      aria-label={label}
      title={`Seconds, ${CARD_S_RANGE[0]}–${CARD_S_RANGE[1]}. Empty uses the default.`}
      placeholder={String(placeholder)}
      disabled={disabled}
      value={draft ?? (value == null ? "" : String(value))}
      onChange={(e) => {
        const text = e.target.value;
        setDraft(text);
        if (text.trim() === "") onChange(undefined);
        else if (Number.isFinite(Number(text))) onChange(Number(text));
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

/** The Beat tab's section card: its text (empty = no card), how long it stays and its colour. Every
 * edit goes out as a patch; the caller marks the card as hand-edited. */
export function BeatCardFields({
  script,
  beat,
  selected,
  disabled,
  onEdit,
  onRemove,
}: {
  script: Script;
  beat: Beat;
  /** The card is the selected block of the timeline. */
  selected: boolean;
  disabled?: boolean;
  onEdit: (patch: Partial<SectionCard>) => void;
  onRemove: () => void;
}) {
  const card = beat.card;
  return (
    <div className={`insp-card ${selected ? "selected" : ""}`}>
      <label className="nle-insp-field insp-field-card">
        <span className="insp-field-label">
          <Square size={13} strokeWidth={2} aria-hidden /> Card
          <span className="insp-field-hint">full-screen title before this beat</span>
        </span>
        <input
          value={card?.text ?? ""}
          disabled={disabled}
          onChange={(e) => onEdit({ text: e.target.value })}
          placeholder="No card. Type a title to add one…"
        />
      </label>
      {card && (
        <div className="insp-card-row">
          <label className="nle-insp-field">
            Seconds
            <SecondsField
              value={card.duration_s}
              placeholder={defaultCardSeconds(script)}
              label="Card duration in seconds"
              disabled={disabled}
              onChange={(duration_s) => onEdit({ duration_s })}
            />
          </label>
          <label className="nle-insp-field">
            Background
            <span className="insp-card-colour">
              <input
                type="color"
                aria-label="Card background colour"
                value={cardBackground(script, beat)}
                disabled={disabled}
                onChange={(e) => onEdit({ background: e.target.value })}
              />
              {card.background && (
                <button
                  type="button"
                  className="ghost small"
                  disabled={disabled}
                  onClick={() => onEdit({ background: undefined })}
                  title="Use the cut's card colour"
                >
                  Cut colour
                </button>
              )}
            </span>
          </label>
          <button
            type="button"
            className="small danger insp-card-remove"
            disabled={disabled}
            onClick={onRemove}
            title="Remove this beat's card (Ctrl+Z undoes)"
            aria-label="Remove card"
          >
            <Trash2 size={14} strokeWidth={2} aria-hidden /> Remove
          </button>
        </div>
      )}
      {card && hasCard(beat) && (
        <div className="small muted">
          {cardSeconds(script, beat).toFixed(1)} s on screen{card.generated ? " · written by GhostReel; editing it keeps your version" : ""}
        </div>
      )}
    </div>
  );
}

/** The Cut tab's section cards: one switch for the whole cut, and the look each card starts from. */
export function CutCardSettings({
  script,
  on,
  busy,
  disabled,
  onToggle,
  onStyle,
}: {
  script: Script;
  /** Any beat has a card. */
  on: boolean;
  busy: boolean;
  disabled?: boolean;
  onToggle: (on: boolean) => void;
  onStyle: (patch: Partial<CardStyle>) => void;
}) {
  const style = script.card_style;
  return (
    <div className="insp-card">
      <div className="insp-switch-row">
        <span className="insp-switch-label">
          <Square size={14} strokeWidth={2} aria-hidden />
          Section cards
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Section cards"
          className={`insp-switch ${on ? "on" : ""}`}
          disabled={disabled || busy}
          onClick={() => onToggle(!on)}
          title={
            on
              ? "On: a title card opens each chapter. Click to remove every card."
              : "Off: click to add a title card before each beat that starts a new chapter."
          }
        >
          <span className="insp-switch-knob" />
        </button>
        <span className={`insp-switch-state ${on ? "" : "off"}`}>{busy ? "…" : on ? "On" : "Off"}</span>
      </div>
      <div className="small muted">
        {on
          ? "Cards you edit stay when you turn this on again."
          : "A full-screen title before each beat that opens a new chapter."}
      </div>
      <div className="insp-card-row">
        <label className="nle-insp-field">
          Default seconds
          <SecondsField
            value={style?.duration_s}
            placeholder={defaultCardSeconds({ ...script, card_style: undefined })}
            label="Default card duration in seconds"
            disabled={disabled}
            onChange={(duration_s) => onStyle({ duration_s })}
          />
        </label>
        <label className="nle-insp-field">
          Default background
          <span className="insp-card-colour">
            <input
              type="color"
              aria-label="Default card background colour"
              value={style?.background ?? "#000000"}
              disabled={disabled}
              onChange={(e) => onStyle({ background: e.target.value })}
            />
            {style?.background && (
              <button
                type="button"
                className="ghost small"
                disabled={disabled}
                onClick={() => onStyle({ background: undefined })}
                title="Back to black"
              >
                Black
              </button>
            )}
          </span>
        </label>
      </div>
    </div>
  );
}
