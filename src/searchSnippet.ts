/** Index field labels: the value is evidence, the label is bookkeeping. */
const FIELD_LABEL = /^(?:Shot|Tags?|Objects?|Setting|On\s+screen|Visible text|Description|Speech)$/i;
const SHOT_SIZE = /^(?:extreme\s+close[- ]?up|medium\s+close[- ]?up|close[- ]?up|medium\s+shot|wide\s+shot|establishing|wide|medium)$/i;

/**
 * One result excerpt, cleaned of index bookkeeping.
 *
 * Stored picture text is field-per-line ("Setting: outdoor patio", "Shot: medium", "Tags: …"),
 * and FTS already reduced that to a token window. Two things are therefore removed at display
 * time, because the stored text cannot be rewritten without re-indexing:
 *
 * - every field label, and the whole `Shot:` field. The shot size is a classification, not a
 *   sentence part; left in, it welds two words together ("entrance medium musician").
 * - repeated terms. "patio" twice in one excerpt reads as if the index double-counts.
 */
export function cleanSearchSnippet(snippet: string): string {
  // The stored snippet marks the edges it cut with "…". Trimming other edge punctuation must not
  // eat those: without them a stopped excerpt reads as a sentence that just ends mid-thought.
  const cutHead = /^\s*…/.test(snippet);
  const cutTail = /…\s*$/.test(snippet);
  const cleaned = snippet
    // Two adjacent matches are one phrase in the text ("[live] [music]"): merge them before the
    // brackets are dropped, so the excerpt shows one highlight chip instead of two.
    .replace(/\] +\[/g, " ")
    // The Shot field is dropped whole, value included. Two shapes occur: FTS keyword snippets keep
    // the stored newlines, semantic hits flatten them to " · ", and the label itself is bracketed
    // when the query contains the word "shot" ("[Shot]: wide").
    .replace(/(?:^|[\s\n·])\[?Shot\]?\s*:\s*[^\n·]*/gi, " · ")
    .replace(/\s*·\s*(?:·\s*)+/g, " · ")
    .replace(/(^|[\s\n])Shot\s*:[^\n]*(?=\n|$)/gi, "$1")
    .replace(/\[([^\]]+)\]\s*:\s*/g, (match, label: string) => (FIELD_LABEL.test(label.trim()) ? "" : match))
    .replace(/\b(?:Shot|Tags?|Objects?|Setting|On\s+screen|Visible text|Description|Speech)\s*:\s*/gi, "")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([,.;])/g, "$1")
    .replace(/^[\s,.;…-]+|[\s,.;…-]+$/g, "");
  const seen = new Set<string>();
  const words = new Set<string>();
  const joined = cleaned.split(/(,\s*)/).map((part, index) => {
    if (index % 2 === 1) return part;
    const key = part.replace(/\[[^\]]+\]/g, (value) => value.slice(1, -1)).replace(/^[\s·]+/, "").trim().toLocaleLowerCase();
    if (!key) return "";
    const partWords = key.split(/\s+/).filter(Boolean);
    // A shot-size word left behind as a term of its own (the label was cut off by the token
    // window) is bookkeeping too — but a phrase inside a sentence ("a wide shot of …") is not.
    if (SHOT_SIZE.test(key)) return "";
    // A short, term-like part that only repeats words already shown adds noise, not evidence.
    if (seen.has(key) || (partWords.length <= 2 && partWords.every((word) => words.has(word)))) return "";
    seen.add(key);
    for (const word of partWords) words.add(word);
    return part;
  }).join("").replace(/(?:,\s*)+/g, ", ").replace(/^[,\s]+|[,\s]+$/g, "");
  return `${cutHead ? "…" : ""}${joined}${cutTail ? "…" : ""}`;
}

/** The matched terms an excerpt marks with [brackets], labels and duplicates removed. */
export function matchedTerms(snippet: string): string[] {
  const terms = [...snippet.matchAll(/\[([^\]]+)\]/g)]
    .map((match) => cleanSearchSnippet(match[1]))
    .filter((term) => term && !FIELD_LABEL.test(term));
  return [...new Map(terms.map((term) => [term.toLocaleLowerCase(), term])).values()];
}
