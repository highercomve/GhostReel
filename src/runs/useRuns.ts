import { useEffect, useMemo, useState } from "react";
import { listRuns, type Run } from "../api";
import { onEvent } from "../events";
import { useQueue } from "../useQueue";
import { mergeRuns, type RunView } from "./model";
import { onTrashChanged } from "./trashEvents";

/**
 * Runs for one project (or every project when `projectId` is null): the queue's live tasks merged
 * with saved runs, newest first. Saved runs reload when a task finishes and when the window gets
 * focus again (a CLI run may have finished meanwhile).
 */
export function useRuns(projectId: number | null, limit = 50): RunView[] {
  const tasks = useQueue();
  const [saved, setSaved] = useState<Run[]>([]);
  useEffect(() => {
    let alive = true;
    const load = () =>
      listRuns(projectId, limit)
        .then((r) => alive && setSaved(r))
        .catch(() => {});
    setSaved([]);
    load();
    const off = onEvent<number>("task-finished", load);
    // Deletes and restores are recorded as runs.
    const offTrash = onTrashChanged(load);
    window.addEventListener("focus", load);
    return () => {
      alive = false;
      off();
      offTrash();
      window.removeEventListener("focus", load);
    };
  }, [projectId, limit]);
  return useMemo(
    () => mergeRuns(projectId == null ? tasks : tasks.filter((t) => t.project_id === projectId), saved),
    [tasks, saved, projectId],
  );
}

/** One project's runs, live and saved. */
export const useProjectRuns = (projectId: number, limit?: number) => useRuns(projectId, limit);
