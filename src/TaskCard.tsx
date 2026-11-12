import { useEffect, useRef, useState } from "react";
import { cancelTask, etaText, type ChatProgress, type Task } from "./api";
import { ChevronDown, ChevronRight } from "lucide-react";
import { onEvent } from "./events";
import ProgressBar, { progressText } from "./runs/ProgressBar";
import { RunSteps, STATE_LABEL } from "./runs/RunLog";
import { RunLinks, type LinkContext } from "./runs/links";
import { toMs } from "./runs/model";

/** "burn_titles" → "Burn titles". */
const humanise = (key: string) => {
  const s = key.replace(/_id$/, "").replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
};

/** A kind field's value as text: ids as "#4", flags as yes/no, small objects as "key value" pairs. */
function factValue(key: string, v: unknown): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return key.endsWith("_id") ? `#${v}` : String(v);
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.length ? v.join(", ") : null;
  if (typeof v === "object") {
    const parts = Object.entries(v as Record<string, unknown>)
      .filter(([, x]) => ["string", "number", "boolean"].includes(typeof x) && x !== "")
      .slice(0, 6)
      .map(([k, x]) => `${humanise(k).toLowerCase()} ${String(x)}`);
    return parts.length ? parts.join(" · ") : null;
  }
  return null;
}

/** The task's own facts, as name/value rows — what it is working on, not how far along it is. */
function details(task: Task): [string, string][] {
  const rows: [string, string][] = [];
  for (const [k, v] of Object.entries(task.kind)) {
    if (k === "type") continue;
    const value = factValue(k, v);
    if (value != null) rows.push([humanise(k), value]);
  }
  const p = task.progress;
  if (p) {
    if (p.phase_total > 0) rows.push(["Step", `${p.phase_done} of ${p.phase_total}`]);
    if (p.elapsed_secs > 0) rows.push(["Running for", etaText(p.elapsed_secs)]);
    if (p.current) rows.push(["Now", p.current]);
  }
  rows.push(["Started", new Date(toMs(task.created_at)).toLocaleTimeString()]);
  if (task.finished_at) rows.push(["Finished", new Date(toMs(task.finished_at)).toLocaleTimeString()]);
  return rows;
}

/** One line of the live log for a script chat: what the editor just did. */
function chatLine(e: ChatProgress["event"]): string {
  switch (e.kind) {
    case "tool_started": {
      const a = e.args;
      const arg =
        typeof a === "string" ? a : a?.query ?? a?.video_id ?? (a ? Object.values(a)[0] : undefined);
      return `${e.tool}${arg != null ? ` ${JSON.stringify(arg)}` : ""}…`;
    }
    case "tool_finished":
      return `${e.tool} → ${e.summary}`;
    case "drafting":
      return "Drafting the script";
    case "validating":
      return "Checking the draft";
  }
}

export default function TaskCard({
  task,
  compact = false,
  linkCtx = {},
}: {
  task: Task;
  compact?: boolean;
  linkCtx?: LinkContext;
}) {
  const p = task.progress;
  const running = task.state === "running";
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  // A script chat's real progress is the tools it calls; its steps alone say almost nothing.
  // Only listened to while the card is open, so closed cards cost nothing.
  const sessionId = task.kind.type === "chat" ? task.kind.session_id : null;
  useEffect(() => {
    if (!open || sessionId == null) return;
    return onEvent<ChatProgress>("chat-progress", (p) => {
      if (p.session_id !== sessionId) return;
      setLog((prev) => [...prev, chatLine(p.event)].slice(-200));
    });
  }, [open, sessionId]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  return (
    <div className={`card task ${task.state} ${open ? "open" : ""}`}>
      <div
        className="progress-head clickable"
        onClick={() => setOpen((v) => !v)}
        title={open ? "Hide details" : "Show details"}
      >
        <span className="label">
          <span className="task-caret">{open ? <ChevronDown size={14} strokeWidth={2} aria-hidden /> : <ChevronRight size={14} strokeWidth={2} aria-hidden />}</span>{" "}
          {task.label || humanise(task.kind.type)}
        </span>
        <span className="muted small">{running && p ? progressText(p) : STATE_LABEL[task.state]}</span>
      </div>
      {running && <ProgressBar progress={p} />}
      {task.note && (running || !compact) && <div className="muted small">{task.note}</div>}

      {open && (
        <div className="task-details">
          {task.label && <div className="task-full-label">{task.label}</div>}
          <dl>
            {details(task).map(([k, v]) => (
              <div key={k}>
                <dt className="muted small">{k}</dt>
                <dd className="small">{v}</dd>
              </div>
            ))}
          </dl>
          {task.steps.length > 0 ? (
            <RunSteps steps={task.steps} />
          ) : (
            sessionId == null && <div className="muted small">{running ? "Waiting for the first step…" : "No steps recorded."}</div>
          )}
          {sessionId != null && (
            <div className="task-log" ref={logRef}>
              {log.length === 0 ? (
                <div className="muted small">{running ? "Waiting for the next tool call…" : "No tool calls recorded."}</div>
              ) : (
                log.map((line, i) => (
                  <div key={i} className="small">
                    {line}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}

      {(task.state === "done" || task.state === "partial") && task.summary && (
        <div className="muted small">{task.summary}</div>
      )}
      {task.state === "done" && task.output && !compact && <div className="muted small path">Output: {task.output}</div>}
      {task.error && <div className="bad-text small">{task.error}</div>}
      {!compact && task.state === "done" && <RunLinks links={task.links} ctx={linkCtx} />}
      {(task.state === "queued" || running) && (
        <div>
          <button className="ghost small" onClick={() => cancelTask(task.id)}>
            {running ? "Stop" : "Remove from queue"}
          </button>
        </div>
      )}
    </div>
  );
}
