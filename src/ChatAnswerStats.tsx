import type { ChatPerformance, InferenceCall } from "./api";

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

export default function ChatAnswerStats({ performance }: { performance?: ChatPerformance | null }) {
  if (!performance) {
    return <div className="chat-answer-stats" title="Earlier answers did not record their model or timing.">Model / speed not recorded</div>;
  }
  return (
    <div className="chat-answer-stats" aria-label="Answer model and performance">
      {summarize(performance.calls).map((model) => (
        <div className="chat-answer-stat-row" key={model.key}>
          <span className="chat-answer-model" title={`${model.model} · ${model.backend}`}>{model.model}</span>
          <span className="chat-answer-rate" title={model.generation
            ? "Generated tokens divided by generation time; excludes prompt processing. Includes research and repair calls."
            : "The provider did not report generation time. This overall rate includes prompt processing and request overhead."}>
            {model.rate != null ? `${model.rate.toFixed(1)} tok/s${model.generation ? "" : " overall"}` : "Speed unavailable"}
          </span>
          {model.output != null && <span>{model.output.toLocaleString()} tokens</span>}
          {model.calls > 1 && <span title="Primary-model requests including tool research and repairs">{model.calls} model calls</span>}
        </div>
      ))}
      <span title="Complete answer time, including research, tool execution and validation">{performance.elapsed_seconds.toFixed(1)}s total</span>
    </div>
  );
}
