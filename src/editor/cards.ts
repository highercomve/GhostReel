import type { Beat, CardStyle, Script, ScriptClip, SectionCard } from "../api";

/** How long a section card stays on screen unless it or the cut says otherwise (s). */
export const DEFAULT_CARD_S = 2;
/** The range a card's duration is kept in (s). */
export const CARD_S_RANGE: readonly [number, number] = [0.5, 10];
/** The colour of a card nobody coloured. */
export const DEFAULT_CARD_BACKGROUND = "#000000";

/** Seconds a clip plays. */
export const clipSeconds = (c: Pick<ScriptClip, "in_s" | "out_s">) => Math.max(0, c.out_s - c.in_s);

/** Seconds of footage in a beat's clips (its card is not counted). */
export const clipsSeconds = (beat: Beat) => beat.clips.reduce((sum, c) => sum + clipSeconds(c), 0);

/** Whether the beat has a card that shows: one with no text takes no time and draws nothing. */
export const hasCard = (beat: Beat) => !!beat.card && beat.card.text.trim() !== "";

/** The duration a card without its own takes: the cut's, else the default. */
export function defaultCardSeconds(script: Script): number {
  return clampCard(script.card_style?.duration_s);
}

function clampCard(d: number | null | undefined): number {
  const v = d != null && Number.isFinite(d) ? d : DEFAULT_CARD_S;
  return Math.min(Math.max(v, CARD_S_RANGE[0]), CARD_S_RANGE[1]);
}

/**
 * How long `beat`'s card stays on screen: 0 when it has none (or one with no text). Mirrors
 * `Script::card_s`: the card's own duration, else the cut's, else 2 s, kept within 0.5..10.
 */
export function cardSeconds(script: Script, beat: Beat): number {
  if (!hasCard(beat)) return 0;
  return clampCard(beat.card?.duration_s ?? script.card_style?.duration_s);
}

/** The colour behind `beat`'s card, `#rrggbb`: the card's, else the cut's, else black. */
export function cardBackground(script: Script, beat: Beat): string {
  const ok = (c: string | null | undefined): c is string => !!c && /^#[0-9a-fA-F]{6}$/.test(c);
  const pick = beat.card?.background ?? script.card_style?.background;
  return ok(pick) ? pick.toLowerCase() : DEFAULT_CARD_BACKGROUND;
}

/** Where a beat sits on the cut's clock: its card plays `cardStart`..`clipsStart`, its clips
 * `clipsStart`..`end`. A beat without a card has `cardStart === clipsStart`. */
export interface BeatTimes {
  cardStart: number;
  clipsStart: number;
  end: number;
}

/** Where every beat sits on the cut's timeline, from raw durations (mirrors `Script::beat_times`). */
export function beatTimes(script: Script): BeatTimes[] {
  let t = 0;
  return script.beats.map((beat) => {
    const cardStart = t;
    const clipsStart = t + cardSeconds(script, beat);
    t = clipsStart + clipsSeconds(beat);
    return { cardStart, clipsStart, end: t };
  });
}

/** Time taken by section cards. */
export const cardsSeconds = (script: Script) => script.beats.reduce((sum, b) => sum + cardSeconds(script, b), 0);

/** How long the cut plays: its clips and its section cards. */
export const scriptLength = (script: Script) =>
  script.beats.reduce((sum, b) => sum + cardSeconds(script, b) + clipsSeconds(b), 0);

/** A beat's card after a hand edit: the patch applied, and no longer one GhostReel wrote. A card
 * left with no text and no settings of its own is dropped. */
export function editedCard(card: SectionCard | null | undefined, patch: Partial<SectionCard>): SectionCard | undefined {
  const { generated: _generated, ...rest } = { ...(card ?? { text: "" }), ...patch };
  const next: SectionCard = { ...rest, text: rest.text ?? "" };
  for (const k of Object.keys(next) as (keyof SectionCard)[]) if (next[k] === undefined) delete next[k];
  const custom = Object.keys(next).some((k) => k !== "text");
  return next.text === "" && !custom ? undefined : next;
}

/** The cut's card style after an edit; empty once nothing is set. */
export function editedCardStyle(style: CardStyle | null | undefined, patch: Partial<CardStyle>): CardStyle | undefined {
  const next: CardStyle = { ...(style ?? {}), ...patch };
  for (const k of Object.keys(next) as (keyof CardStyle)[]) if (next[k] === undefined) delete next[k];
  return Object.keys(next).length > 0 ? next : undefined;
}
