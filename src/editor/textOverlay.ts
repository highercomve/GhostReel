import type { CaptionStyle, Cue, CueEdit, Overlay, ResolvedStyle, Script, TextStyle, TitleLayout } from "../api";

/** Fonts offered in the style menu: each exists on Windows and macOS and maps to a
 * metric-compatible Liberation font on Linux, so renders and the viewer break lines alike. */
export const FONTS = ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Impact"];

const FALLBACKS: Record<string, string> = {
  arial: `"Liberation Sans", Arimo, Helvetica, sans-serif`,
  "times new roman": `"Liberation Serif", Tinos, serif`,
  "courier new": `"Liberation Mono", Cousine, monospace`,
  georgia: "serif",
};

export function fontStack(font: string): string {
  return `"${font.replace(/"/g, "")}", ${FALLBACKS[font.toLowerCase()] ?? "sans-serif"}`;
}

/** Selection key for a title (`t-<beat>`) or a caption (`cap-<video>_<source start ms>`). The
 * caption key comes from the cue's source anchor, so it survives moving clips around. */
export function cueKey(cue: Cue, index = 0): string {
  return cue.video_id != null && cue.src_start_s != null
    ? `cap-${cue.video_id}_${Math.round(cue.src_start_s * 1000)}`
    : `cap-i${index}`;
}

export function overlayKey(o: Overlay, index = 0): string {
  return o.kind === "title" ? `t-${o.beat_index ?? index}` : cueKey(o.cue ?? { start_s: 0, end_s: 0, text: "" }, index);
}

const hexToRgba = (hex: string, alpha: number) => {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 6 ? h : "000000", 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

/** A ring of text-shadows: the nearest CSS gets to libass's outline (drawn outside the glyph). */
function ring(radius: number, color: string, dx = 0, dy = 0): string[] {
  if (radius <= 0.05) return dx || dy ? [`${dx}px ${dy}px 0 ${color}`] : [];
  const steps = radius < 1.5 ? 8 : 16;
  const out: string[] = [];
  for (let i = 0; i < steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    out.push(`${(dx + Math.cos(a) * radius).toFixed(2)}px ${(dy + Math.sin(a) * radius).toFixed(2)}px 0 ${color}`);
  }
  return out;
}

/** CSS for an overlay in a frame `frameH` pixels tall, matching the ASS style a render uses. */
export function overlayCss(style: ResolvedStyle, frameH: number, sizeOverride?: number) {
  const fs = (sizeOverride ?? style.size) * frameH;
  const shadows: string[] = [];
  if (!style.box) {
    const ow = style.outline * fs;
    const sh = style.shadow * fs;
    if (sh > 0.05) {
      // libass shadows the glyph and its outline together, offset down and right.
      shadows.push(...ring(ow, "rgba(0, 0, 0, 0.5)", sh, sh));
      if (ow <= 0.05) shadows.push(`${sh}px ${sh}px 0 rgba(0, 0, 0, 0.5)`);
    }
    shadows.unshift(...ring(ow, style.outline_color));
  }
  const pad = style.box ? style.box_pad * fs : 0;
  return {
    block: {
      fontFamily: fontStack(style.font),
      fontSize: `${fs / style.cell_em}px`,
      lineHeight: `${fs}px`,
      fontWeight: style.bold ? 700 : 400,
      color: style.color,
      textAlign: style.align,
      textShadow: shadows.length ? shadows.join(", ") : "none",
    } as React.CSSProperties,
    line: style.box
      ? ({
          background: hexToRgba(style.box_color, style.box_opacity),
          padding: `${pad}px`,
          margin: `0 ${-pad}px`,
          boxDecorationBreak: "clone",
          WebkitBoxDecorationBreak: "clone",
        } as React.CSSProperties)
      : undefined,
  };
}

/** translate() that puts the anchor point of the text block at its x,y. */
export function anchorTransform(align: string, vAnchor: string): string {
  const tx = align === "left" ? "0" : align === "right" ? "-100%" : "-50%";
  const ty = vAnchor === "bottom" ? "-100%" : "-50%";
  return `translate(${tx}, ${ty})`;
}

// ---- edits ------------------------------------------------------------------------------

export function patchTitleLayout(script: Script, beat: number, patch: Partial<TitleLayout>): Script {
  const beats = [...script.beats];
  const b = beats[beat];
  if (!b) return script;
  const layout: TitleLayout = { ...(b.title_layout ?? {}), ...patch };
  for (const k of Object.keys(layout) as (keyof TitleLayout)[]) if (layout[k] == null) delete layout[k];
  beats[beat] = { ...b, title_layout: Object.keys(layout).length ? layout : undefined };
  return { ...script, beats };
}

export function patchTitleStyle(script: Script, beat: number, patch: Partial<TextStyle>): Script {
  const cur = script.beats[beat]?.title_layout?.style ?? {};
  const style = cleanStyle({ ...cur, ...patch });
  return patchTitleLayout(script, beat, { style: Object.keys(style).length ? style : undefined });
}

/** Makes this title's look the cut's title style and drops every title's own override. */
export function applyTitleStyleToAll(script: Script, beat: number, effective: TextStyle): Script {
  const beats = script.beats.map((b) => {
    if (!b.title_layout?.style) return b;
    const rest = { ...b.title_layout };
    delete rest.style;
    return { ...b, title_layout: Object.keys(rest).length ? rest : undefined };
  });
  void beat;
  return { ...script, beats, title_style: cleanStyle({ ...(script.title_style ?? {}), ...effective }) };
}

function cleanStyle(s: TextStyle): TextStyle {
  const out: TextStyle = { ...s };
  for (const k of Object.keys(out) as (keyof TextStyle)[]) if (out[k] == null) delete out[k];
  return out;
}

const DEFAULT_CAPTIONS: CaptionStyle = { position: "bottom", size: 1 };

export function patchCaptions(script: Script, patch: Partial<CaptionStyle>): Script {
  const captions: CaptionStyle = { ...DEFAULT_CAPTIONS, ...(script.captions ?? {}), ...patch };
  for (const k of Object.keys(captions) as (keyof CaptionStyle)[]) if (captions[k] == null) delete captions[k];
  return { ...script, captions };
}

export function patchCaptionStyle(script: Script, patch: Partial<TextStyle>): Script {
  const style = cleanStyle({ ...(script.captions?.style ?? {}), ...patch });
  return patchCaptions(script, { style: Object.keys(style).length ? style : undefined });
}

/** The edit that names this cue, by the same rule the backend uses. */
export function findCueEdit(edits: CueEdit[] | undefined, cue: Cue): number {
  if (!edits || cue.video_id == null || cue.src_start_s == null || cue.src_end_s == null) return -1;
  let best = -1;
  let bestScore = Infinity;
  edits.forEach((e, i) => {
    if (e.video_id !== cue.video_id) return;
    const da = Math.abs(e.src_start_s - cue.src_start_s!);
    const db = Math.abs(e.src_end_s - cue.src_end_s!);
    let score: number | null = null;
    if (da <= 0.35 && db <= 0.35) score = da + db;
    else {
      const overlap = Math.max(0, Math.min(e.src_end_s, cue.src_end_s!) - Math.max(e.src_start_s, cue.src_start_s!));
      const union = Math.max(e.src_end_s, cue.src_end_s!) - Math.min(e.src_start_s, cue.src_start_s!);
      if (union > 0 && overlap / union >= 0.6) score = 1 + (1 - overlap / union);
    }
    if (score != null && score < bestScore) {
      bestScore = score;
      best = i;
    }
  });
  return best;
}

/** Adds to (or starts) the hand edit of a cue. `patch` start/end are source seconds. */
export function patchCueEdit(script: Script, cue: Cue, patch: Partial<CueEdit>): Script {
  if (cue.video_id == null || cue.src_start_s == null || cue.src_end_s == null) return script;
  const edits = [...(script.caption_edits ?? [])];
  const i = findCueEdit(edits, cue);
  const base: CueEdit = i >= 0 ? edits[i] : { video_id: cue.video_id, src_start_s: cue.src_start_s, src_end_s: cue.src_end_s };
  const next: CueEdit = { ...base, ...patch };
  if (!next.hidden) delete next.hidden;
  for (const k of ["text", "start_s", "end_s"] as const) if (next[k] == null) delete next[k];
  const empty = next.text == null && next.start_s == null && next.end_s == null && !next.hidden;
  if (i >= 0) {
    if (empty) edits.splice(i, 1);
    else edits[i] = next;
  } else if (!empty) edits.push(next);
  return { ...script, caption_edits: edits };
}

/** Back to what the transcript says, at its own time. */
export function resetCue(script: Script, cue: Cue): Script {
  const i = findCueEdit(script.caption_edits, cue);
  if (i < 0) return script;
  const edits = [...(script.caption_edits ?? [])];
  edits.splice(i, 1);
  return { ...script, caption_edits: edits };
}

/** A cue moved on the cut's clock, written back in source seconds. */
export function retimeCue(script: Script, cue: Cue, start: number, end: number): Script {
  const off = cue.offset_s ?? 0;
  return patchCueEdit(script, cue, { start_s: +(start - off).toFixed(3), end_s: +(end - off).toFixed(3) });
}

/** Position presets for titles (y = middle) and captions (y = bottom of the text). */
export type Preset = "top" | "center" | "bottom" | "lower";
export function presetPosition(kind: "title" | "caption", preset: Preset, portrait: boolean) {
  const bottom = portrait ? 0.8 : 0.92;
  if (kind === "title") {
    switch (preset) {
      case "top": return { x: 0.5, y: 0.13, align: "center" as const };
      case "center": return { x: 0.5, y: 0.5, align: "center" as const };
      case "bottom": return { x: 0.5, y: portrait ? 0.72 : 0.85, align: "center" as const };
      case "lower": return { x: 0.06, y: portrait ? 0.68 : 0.78, align: "left" as const };
    }
  }
  switch (preset) {
    case "top": return { x: 0.5, y: 0.22, align: "center" as const };
    case "center": return { x: 0.5, y: 0.56, align: "center" as const };
    case "bottom": return { x: 0.5, y: bottom, align: "center" as const };
    case "lower": return { x: 0.06, y: portrait ? 0.72 : 0.84, align: "left" as const };
  }
}
