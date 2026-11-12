import type { StageStatus, Task } from "./api";

/** A stage as the Project menu shows it: the backend's status, or `running` while the queue is
 *  working on that stage right now. */
export interface StageDisplay {
  state: StageStatus["state"] | "running";
  reason: string | null;
}

/** The queue task's stage: its kind for a stand-alone pass, else the index run's current phase
 *  ("describe_server" → "describe"). */
function runningStage(task: Task | undefined): string | null {
  if (!task) return null;
  if (task.kind.type === "steadiness" || task.kind.type === "faces" || task.kind.type === "speakers") {
    return task.kind.type;
  }
  if (task.kind.type !== "index") return null;
  return task.progress?.phase?.split("_")[0] ?? null;
}

export function stageDisplay(status: StageStatus | undefined, running: Task | undefined): StageDisplay {
  if (!status) return { state: "pending", reason: null };
  if (status.state !== "off" && status.state !== "unavailable" && runningStage(running) === status.stage) {
    return { state: "running", reason: null };
  }
  return { state: status.state, reason: status.reason };
}

export function stageTagClass(state: string): string {
  switch (state) {
    case "running":
      return "accent";
    case "done":
      return "good";
    case "failed":
      return "bad";
    case "off":
    case "unavailable":
      return "muted";
    default:
      return "warn";
  }
}
