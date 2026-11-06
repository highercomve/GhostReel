import { formatDuration, formatDurationText } from "./timeFormat";
// Turns what Sonar's backend records into words for an editor: tool calls become "Looked
// through 9 clips", the bracketed notes the pipeline appends for the model become bullet lists.
import { clock, type ToolCallRecord } from "./api";

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** "put 2 range(s) back" → "put 2 ranges back". */
export function fixPlurals(s: string): string {
  // The noun may follow the number after a qualifier: "3 shot/transcript range(s)".
  return s.replace(/(\d+(?:\.\d+)?)([^\d()]{0,40}?)\b([A-Za-z]+)\(s\)/g, (_, n: string, mid: string, w: string) =>
    `${n}${mid}${Number(n) === 1 ? w : `${w}s`}`,
  );
}

function sentence(s: string): string {
  const t = fixPlurals(s.trim()).replace(/[.;,]\s*$/, "");
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : "";
}

/** "a", "a and b", "a, b and c". */
function joinList(parts: string[]): string {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

function argsOf(call: { args: unknown }): Record<string, unknown> {
  const a = call.args;
  if (a && typeof a === "object") return a as Record<string, unknown>;
  if (typeof a === "string") {
    try {
      const parsed = JSON.parse(a);
      if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
    } catch {
      // A bare string argument: treat it as a query.
      return { query: a };
    }
  }
  return {};
}

function videoIdOf(call: { args: unknown }): number | null {
  const v = argsOf(call).video_id;
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
}

/** One sentence for a whole research pass: "Looked through 9 clips and ran 2 searches". */
export function describeResearch(calls: ToolCallRecord[]): string {
  const clips = new Set<number | string>();
  const transcripts = new Set<number | string>();
  let searches = 0;
  let listed = false;
  let other = 0;
  calls.forEach((c, i) => {
    const id = videoIdOf(c) ?? `?${i}`;
    if (c.tool === "get_video") clips.add(id);
    else if (c.tool === "get_transcript") transcripts.add(id);
    else if (c.tool.startsWith("search")) searches++;
    else if (c.tool === "list_videos") listed = true;
    else other++;
  });
  const parts: string[] = [];
  if (clips.size) parts.push(`looked through ${plural(clips.size, "clip")}`);
  if (transcripts.size) parts.push(`read ${plural(transcripts.size, "transcript")}`);
  if (searches) parts.push(`ran ${plural(searches, "search", "searches")}`);
  if (listed) parts.push("checked the library");
  if (other) parts.push(plural(other, "other step"));
  return sentence(joinList(parts)) || "Researched the footage";
}

/** One line per tool call, for the expanded research list. */
export function describeCall(
  call: { tool: string; args: unknown; summary?: string },
  names: Map<number, string>,
): { text: string; detail?: string } {
  const args = argsOf(call);
  const id = videoIdOf(call);
  const name = id != null ? (names.get(id) ?? `video #${id}`) : "a video";
  const detail = call.summary && !call.summary.startsWith("error") ? call.summary : call.summary ? "failed" : undefined;
  switch (call.tool) {
    case "get_video":
      return { text: `Looked at ${name}`, detail };
    case "get_transcript": {
      const s = Number(args.start_s);
      const e = Number(args.end_s);
      const range = Number.isFinite(s) && Number.isFinite(e) ? ` ${clock(s)}–${clock(e)}` : "";
      return { text: `Read ${name}${range}`, detail };
    }
    case "list_videos":
      return { text: "Checked the library", detail };
    default:
      if (call.tool.startsWith("search")) {
        const q = typeof args.query === "string" ? args.query : "";
        return { text: q ? `Searched for “${q}”` : "Searched the footage", detail };
      }
      return { text: sentence(call.tool.replace(/_/g, " ")), detail };
  }
}

/** The step in progress, in words, for the live research line. */
export function describeLiveStep(tool: string, args: unknown, names: Map<number, string>): string {
  const call = { tool, args };
  const id = videoIdOf(call);
  const name = id != null ? (names.get(id) ?? "a clip") : "a clip";
  if (tool === "get_video") return `Looking at ${name}`;
  if (tool === "get_transcript") return `Reading ${name}`;
  if (tool === "list_videos") return "Checking the library";
  if (tool.startsWith("search")) {
    const q = argsOf(call).query;
    return typeof q === "string" && q ? `Searching for “${q}”` : "Searching the footage";
  }
  return sentence(tool.replace(/_/g, " "));
}

const NOTE_KINDS: { prefix: RegExp; kind: AnswerSection["kind"]; title: string; split: boolean }[] = [
  { prefix: /^applied after drafting(?:,[^:]*)?:\s*/i, kind: "fixed", title: "Fixed after drafting", split: true },
  { prefix: /^editorial read of [^:]*:\s*/i, kind: "editorial", title: "Editorial notes", split: true },
  { prefix: /^research incomplete:\s*/i, kind: "check", title: "Check before export", split: false },
];

export interface AnswerSection {
  kind: "fixed" | "editorial" | "check";
  title: string;
  items: string[];
}

export interface ParsedAnswer {
  summary: string;
  sections: AnswerSection[];
}

/**
 * Splits an assistant answer into a short summary and bullet lists. The backend appends notes
 * for the model ("[applied after drafting, already in the saved script: a; b]") and the script
 * JSON in a code fence; neither is meant to be read as prose.
 */
export function parseAnswer(content: string): ParsedAnswer {
  let hadScript = false;
  let text = content.replace(/```[a-z]*\s*[\s\S]*?```/gi, () => {
    hadScript = true;
    return "";
  });
  const sections: AnswerSection[] = [];
  text = text.replace(/\[([^[\]]+)\]/g, (whole, inner: string) => {
    for (const k of NOTE_KINDS) {
      if (k.prefix.test(inner)) {
        const body = inner.replace(k.prefix, "");
        const items = (k.split ? body.split(/;\s*/) : [body]).map(formatRepairNote).map(sentence).filter(Boolean);
        if (items.length) sections.push({ kind: k.kind, title: k.title, items });
        return "";
      }
    }
    return whole;
  });
  // "Drafted 'Social Reel (30s)': 4 beats, 6 clips, 34.3 s" → readable, with a clock duration.
  text = text.replace(
    /Drafted '([^']+)':\s*(\d+) beats?,\s*(\d+) clips?,\s*([\d.]+)\s*s\b/g,
    (_, title: string, b: string, c: string, s: string) =>
      `Drafted “${title}”: ${plural(Number(b), "beat")}, ${plural(Number(c), "clip")}, ${formatDuration(Number(s))} long.`,
  );
  let summary = formatDurationText(text).replace(/\n{3,}/g, "\n\n").trim();
  if (!summary && hadScript) summary = "Here is the new version.";
  return { summary, sections };
}

function formatRepairNote(note: string): string {
  const duration = note.match(/total duration\s+([\d.]+)\s*s\s*\(target\s+([\d.]+)\s*s,\s*difference\s+([\d.]+)%\)/i);
  if (duration) {
    const actual = Number(duration[1]);
    const target = Number(duration[2]);
    const delta = Math.abs(actual - target);
    const relation = actual > target ? "over" : actual < target ? "under" : "on target";
    return `${formatDuration(actual)} total · target ${formatDuration(target)} · ${formatDuration(delta)} ${relation}`;
  }
  return note.replace(/\b([\d.]+)\s*s of quiet\b/gi, (_, seconds: string) => `${Math.round(Number(seconds))} s of quiet`);
}

/** A new-cut brief starts with `[Draft "Title" — Target: 30s]`; show it as a chip, not text. */
export function parseDraftHeader(text: string): { title: string; target_s: number; rest: string } | null {
  const m = text.match(/^\[Draft "(.+?)" — Target: (\d+)s\]\s*\n?/);
  if (!m) return null;
  return { title: m[1], target_s: Number(m[2]), rest: text.slice(m[0].length) };
}

/** Editorial duration, shared with the cut and timeline. */
export const formatSeconds = clock;
