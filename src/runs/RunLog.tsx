import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  ArchiveRestore,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  Clapperboard,
  Download,
  FileOutput,
  MessageSquare,
  Minus,
  Play,
  ScanFace,
  TriangleAlert,
  ScanLine,
  Trash2,
  Vibrate,
  X,
} from "lucide-react";
import { cancelTask, type AgentError, type RunState, type RunStep, type StepEvent } from "../api";
import AgentErrorView, { agentErrorLabel } from "../agents/AgentErrorView";
import { RunLinks, type LinkContext } from "./links";
import { isLive, toMs, type RunView } from "./model";
import ProgressBar, { progressText } from "./ProgressBar";

export const STATE_LABEL: Record<RunState, string> = {
  queued: "Waiting",
  running: "Running",
  done: "Done",
  partial: "Partly done",
  failed: "Failed",
  cancelled: "Stopped",
};

/** An icon per run kind; unknown kinds get a generic one. */
const KIND_ICONS: Record<string, LucideIcon> = {
  clips: Clapperboard,
  index: ScanLine,
  faces: ScanFace,
  speakers: AudioLines,
  steadiness: Vibrate,
  render_preview: Play,
  export: FileOutput,
  export_clips: FileOutput,
  download_model: Download,
  chat: MessageSquare,
  delete: Trash2,
  restore: ArchiveRestore,
  purge: Trash2,
};

/** The typed failure that ended a run: the last error event that carries one. */
export function runFailure(steps: RunStep[]): AgentError | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    const evs = steps[i].events ?? [];
    for (let j = evs.length - 1; j >= 0; j--) {
      const e = evs[j];
      if (e.level === "error" && e.error) return e.error;
    }
  }
  return null;
}

/** A step's kept events: warnings, rerouted or resting agents (the failure itself is shown once, below). */
function StepEvents({ events }: { events: StepEvent[] }) {
  const shown = events.filter((e) => !(e.level === "error" && e.error));
  if (shown.length === 0) return null;
  return (
    <ul className="run-log-events">
      {shown.map((e, i) => (
        <li key={i} className={`run-log-event level-${e.level}`}>
          {e.error && <span className={`agent-error-kind kind-${e.error.kind}`}>{agentErrorLabel(e.error)}</span>}
          <span>{e.message}</span>
        </li>
      ))}
    </ul>
  );
}

/** A run's steps, each with its last note and the events it kept. `hideFailedNote`: the
 *  failure is shown on its own, so the failed step does not repeat it. */
export function RunSteps({ steps, hideFailedNote = false }: { steps: RunStep[]; hideFailedNote?: boolean }) {
  if (steps.length === 0) return null;
  return (
    <ol className="run-log-steps">
      {steps.map((s, i) => (
        <li key={i} className={`run-log-step ${s.state}`}>
          <span className="run-log-step-icon" aria-hidden="true">
            {s.state === "running" ? (
              <span className="research-dot on" />
            ) : s.state === "failed" ? (
              <X size={12} strokeWidth={2.5} />
            ) : s.state === "stopped" ? (
              <Minus size={12} strokeWidth={2.5} />
            ) : s.state === "warning" ? (
              <TriangleAlert size={12} strokeWidth={2.5} />
            ) : (
              <Check size={12} strokeWidth={2.5} />
            )}
          </span>
          <span>{s.label}</span>
          {s.note && !(hideFailedNote && s.state === "failed") && <span className="muted"> · {s.note}</span>}
          {s.events && s.events.length > 0 && <StepEvents events={s.events} />}
        </li>
      ))}
    </ol>
  );
}

const timeOf = (t: number) =>
  new Date(toMs(t)).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });

/**
 * Any run — a live task or a saved run, any kind: what it is, how far along, the steps it went
 * through, how it ended and buttons for what it made. `compact` folds a finished run's steps.
 */
export default function RunLog({
  run,
  ctx = {},
  compact = false,
}: {
  run: RunView;
  ctx?: LinkContext;
  compact?: boolean;
}) {
  const live = isLive(run);
  const [showSteps, setShowSteps] = useState(!compact);
  const stepsOpen = live || showSteps;
  const Icon = KIND_ICONS[run.kind] ?? Activity;
  const task = run.task;
  const p = task?.progress;
  const failure = run.state === "failed" ? runFailure(run.steps) : null;

  return (
    <div className={`run-log ${run.state}`}>
      <div className="run-log-head">
        <Icon size={14} aria-hidden="true" />
        <span className="run-log-label" title={run.label}>{run.label}</span>
        {/* A short chip that never truncates: the percentage when it can be counted, else the
            state (an uncounted phase's own words are on the line under the bar). */}
        <span className={`run-log-state ${run.state}`}>
          {run.state === "running" && p && !p.indeterminate ? progressText(p) : STATE_LABEL[run.state]}
        </span>
        {run.source === "cli" && (
          <span className="run-log-source" title="Run from the ghostreel command line">
            CLI
          </span>
        )}
        <time className="run-log-time muted small" dateTime={new Date(toMs(run.createdAt)).toISOString()}>
          {timeOf(run.createdAt)}
        </time>
      </div>

      {run.state === "running" && <ProgressBar progress={p} />}
      {/* With steps, the open step already carries the latest note. */}
      {live && task?.note && run.steps.length === 0 && <div className="muted small">{task.note}</div>}

      {run.steps.length > 0 &&
        (live || !compact ? (
          <RunSteps steps={run.steps} hideFailedNote={failure != null} />
        ) : (
          <>
            <button
              type="button"
              className="run-log-toggle ghost small"
              aria-expanded={stepsOpen}
              onClick={() => setShowSteps((v) => !v)}
            >
              {stepsOpen ? <ChevronDown size={13} aria-hidden="true" /> : <ChevronRight size={13} aria-hidden="true" />}
              {run.steps.length} {run.steps.length === 1 ? "step" : "steps"}
            </button>
            {stepsOpen && <RunSteps steps={run.steps} hideFailedNote={failure != null} />}
          </>
        ))}

      {run.summary && !live && <div className="run-log-summary small">{run.summary}</div>}
      {failure ? (
        <div className="run-log-failure">
          <AgentErrorView error={failure} compact />
        </div>
      ) : (
        run.error && <div className="banner bad small run-log-error">{run.error}</div>
      )}
      {!live && <RunLinks links={run.links} ctx={ctx} />}

      {live && task && (
        <div>
          <button type="button" className="ghost small" onClick={() => void cancelTask(task.id)}>
            {run.state === "running" ? "Stop" : "Remove from queue"}
          </button>
        </div>
      )}
    </div>
  );
}
