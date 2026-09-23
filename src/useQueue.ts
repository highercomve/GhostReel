import { useEffect, useState } from "react";
import { queueList, type Task } from "./api";
import { onEvent } from "./events";

/** Live view of the background task queue (updates on every `queue` event). */
export function useQueue(): Task[] {
  const [tasks, setTasks] = useState<Task[]>([]);
  useEffect(() => {
    queueList().then(setTasks).catch(() => {});
    return onEvent<Task[]>("queue", setTasks);
  }, []);
  return tasks;
}

export const isActive = (t: Task) => t.state === "queued" || t.state === "running";
