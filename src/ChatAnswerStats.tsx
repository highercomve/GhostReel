import type { ChatPerformance, InferenceCall } from "./api";
import { formatSeconds } from "./sonarFormat";

function summarize(calls: InferenceCall[]) {
  const groups = new Map<string, InferenceCall[]>();
  for (const call of calls) {
    const key = `${call.backend}\0${call.model}`;
    const group = groups.get(key) ?? [];
    group.push(call);
    groups.set(key, group);
  }
  return [...groups.entries()].map(([key, group]) => {
    const known = group.every((c) => c.output_tokens != null);
    const output = known ? group.reduce((n, c) => n + c.output_tokens!, 0) : null;
    const generation = group.every((c) => c.generation_seconds != null && c.generation_seconds > 0);
    const seconds = group.reduce((n, c) => n + (generation ? c.generation_seconds! : c.request_seconds), 0);
    return { key, model: group[0].model, backend: group[0].backend, calls: group.length,
      output, generation, rate: output != null && seconds > 0 ? output / seconds : null };
  });
}

/** A model name the backend actually reported, not a placeholder for one it didn't. */
function realModel(model: string): boolean {
  return Boolean(model.trim()) && !/not reported/i.test(model);
}

/**
 * One muted line under an answer with only the values that were measured: model, speed, time.
 * Anything unknown is left out rather than printed as "unavailable"; the full numbers sit in the
 * tooltip and in Full log.
 */
export default function ChatAnswerStats({ performance }: { performance?: ChatPerformance | null }) {
  if (!performance) return null;
  const models = summarize(performance.calls);
  const parts: string[] = [];
  const details: string[] = [];
  for (const m of models) {
    if (realModel(m.model)) parts.push(m.model);
    if (m.rate != null) parts.push(`${m.rate.toFixed(0)} tok/s${m.generation ? "" : " overall"}`);
    details.push(
      [
        realModel(m.model) ? `${m.model} (${m.backend})` : m.backend,
        m.output != null ? `${m.output.toLocaleString()} tokens` : null,
        m.calls > 1 ? `${m.calls} model calls` : null,
      ]
        .filter(Boolean)
        .join(", "),
    );
  }
  if (performance.elapsed_seconds > 0) parts.push(`Generated in ${formatSeconds(performance.elapsed_seconds)}`);
  if (parts.length === 0) return null;
  return (
    <div
      className="chat-answer-stats"
      aria-label="Answer model and timing"
      title={[...details, `${performance.elapsed_seconds.toFixed(1)} s total, research and checks included`].join("\n")}
    >
      {parts.join(" · ")}
    </div>
  );
}
