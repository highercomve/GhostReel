import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { cancelTask, etaText, PHASE_LABELS, type ChatProgress, type Task } from "./api";

const STATE_LABEL: Record<string, string> = {
  queued: "Waiting",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function taskLabel(task: Task): string {
  if (task.label) return task.label;
  switch (task.kind.type) {
    case "index":
      return "Indexing";
    case "chat":
      return "Script chat";
    case "render_preview":
      return `Rendering preview (script #${task.kind.script_id})`;
    case "export":
      return `Exporting (${task.kind.format})`;
    case "download_model":
      return `Downloading model (${task.kind.model_id})`;
  }
}

/** The task's own facts, as name/value rows — what it is working on, not how far along it is. */
function details(task: Task): [string, string][] {
  const rows: [string, string][] = [];
  const k = task.kind;
  switch (k.type) {
    case "index":
      rows.push(["Project", `#${k.project_id}`]);
      break;
    case "chat":
      rows.push(["Project", `#${k.project_id}`], ["Chat", `#${k.session_id}`]);
      break;
    case "render_preview":
      rows.push(
        ["Script", `#${k.script_id}`],
        ["Burn in", [k.burn_titles && "titles", k.burn_narration && "narration"].filter(Boolean).join(", ") || "nothing"],
      );
      break;
    case "export":
      rows.push(["Script", `#${k.script_id}`], ["Format", k.format], ["To", k.path]);
      break;
    case "download_model":
      rows.push(["Model", k.model_id]);
      break;
  }
  const p = task.progress;
  if (p) {
    if (p.phase_total > 0) rows.push(["Step", `${p.phase_done} of ${p.phase_total}`]);
    if (p.elapsed_secs > 0) rows.push(["Running for", etaText(p.elapsed_secs)]);
    if (p.current) rows.push(["Now", p.current]);
  }
  rows.push(["Started", new Date((task.created_at < 1e11 ? task.created_at * 1000 : task.created_at)).toLocaleTimeString()]);
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

export default function TaskCard({ task, compact = false }: { task: Task; compact?: boolean }) {
  const p = task.progress;
  const running = task.state === "running";
  const pct = Math.round((p?.fraction ?? 0) * 100);
  const s = task.summary;
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  // A script chat's real progress is the tools it calls; the bar alone says almost nothing.
  // Only listened to while the card is open, so closed cards cost nothing.
  const sessionId = task.kind.type === "chat" ? task.kind.session_id : null;
  useEffect(() => {
    if (!open || sessionId == null) return;
    const un = listen<ChatProgress>("chat-progress", (e) => {
      if (e.payload.session_id !== sessionId) return;
      setLog((prev) => [...prev, chatLine(e.payload.event)].slice(-200));
    });
    return () => {
      un.then((f) => f());
    };
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
          <span className="task-caret">{open ? "▾" : "▸"}</span> {taskLabel(task)}
        </span>
        <span className="muted small">
          {running && p ? (
            p.indeterminate ? (
              // Nothing to count: say what it is doing instead of a number that would be a guess.
              <>{PHASE_LABELS[p.phase] ?? "Working"}…</>
            ) : (
              <>
                {pct}%{p.eta_secs != null && p.fraction < 1 ? ` · ${etaText(p.eta_secs)} left` : ""}
              </>
            )
          ) : (
            STATE_LABEL[task.state]
          )}
        </span>
      </div>
      {running && (
        <>
          <div className={`meter big${p?.indeterminate ? " indeterminate" : ""}`}>
            <div style={p?.indeterminate ? undefined : { width: `${Math.max(2, pct)}%` }} />
          </div>
          {p && (
            <div className="muted small">
              {PHASE_LABELS[p.phase] ?? p.phase}
              {p.phase_total > 0 ? ` · ${p.phase_done} of ${p.phase_total}` : ""}
              {p.current ? ` · ${p.current.split(/[\\/]/).pop()}` : ""}
            </div>
          )}
        </>
      )}
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
          {sessionId != null && (
            <div className="task-log" ref={logRef}>
              {log.length === 0 ? (
                <div className="muted small">{running ? "Waiting for the next step…" : "No steps recorded."}</div>
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

      {task.state === "done" && s && !compact && (
        <div className="muted small">
          {s.new} new · {s.changed} changed · {s.removed} removed · {s.jobs_done} steps done
          {s.jobs_failed ? ` · ${s.jobs_failed} failed` : ""}
        </div>
      )}
      {task.state === "done" && task.output && <div className="muted small path">Output: {task.output}</div>}
      {task.error && <div className="bad-text small">{task.error}</div>}
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
