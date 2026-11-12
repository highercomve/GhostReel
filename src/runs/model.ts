import type { Run, RunLink, RunState, RunStep, Task } from "../api";

/**
 * One run, whichever way it reached the UI: a live queue task or a saved run (which outlives the
 * queue and includes CLI runs). Everything that shows runs works on this shape.
 */
export interface RunView {
  /** The run id: the same for a task and the run it was saved as. */
  key: string;
  kind: string;
  label: string;
  state: RunState;
  projectId: number | null;
  steps: RunStep[];
  links: RunLink[];
  summary: string | null;
  error: string | null;
  /** "app" or "cli". */
  source: string;
  /** Seconds since the epoch. */
  createdAt: number;
  finishedAt: number | null;
  /** The queue task, while the queue still has it (progress, note, Stop). */
  task?: Task;
}

export const isLive = (r: { state: RunState }) => r.state === "queued" || r.state === "running";

/** Seconds or milliseconds since the epoch → milliseconds. */
export const toMs = (t: number) => (t < 1e11 ? t * 1000 : t);

export function fromTask(task: Task): RunView {
  return {
    key: task.run_id,
    kind: task.kind.type,
    label: task.label,
    state: task.state,
    projectId: task.project_id,
    steps: task.steps ?? [],
    links: task.links ?? [],
    summary: task.summary,
    error: task.error,
    source: "app",
    createdAt: task.created_at,
    finishedAt: task.finished_at,
    task,
  };
}

export function fromRun(run: Run): RunView {
  return {
    key: run.id,
    kind: run.kind,
    label: run.label,
    state: run.state,
    projectId: run.project_id,
    steps: run.steps,
    links: run.links,
    summary: run.summary,
    error: run.error,
    source: run.source,
    createdAt: run.created_at,
    finishedAt: run.finished_at,
  };
}

/** Live tasks and saved runs as one list, newest first; a task wins over its own saved run. */
export function mergeRuns(tasks: Task[], saved: Run[]): RunView[] {
  const live = tasks.map(fromTask);
  const seen = new Set(live.map((r) => r.key));
  return [...live, ...saved.filter((r) => !seen.has(r.id)).map(fromRun)].sort(
    (a, b) => toMs(b.createdAt) - toMs(a.createdAt),
  );
}
