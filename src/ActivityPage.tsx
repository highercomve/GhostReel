import { useEffect, useState } from "react";
import { Inbox } from "lucide-react";
import { clearFinishedTasks, enqueueIndex, listProjects, type ProjectSummary } from "./api";
import TaskCard from "./TaskCard";
import { isActive, useQueue } from "./useQueue";

export default function ActivityPage() {
  const tasks = useQueue();
  const active = tasks.filter(isActive);
  const finished = tasks.filter((t) => !isActive(t)).reverse();

  // The empty state doubles as the shortest path to the most common job here: indexing.
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [picked, setPicked] = useState<number | "">("");
  const [startError, setStartError] = useState<string | null>(null);
  useEffect(() => {
    listProjects().then(setProjects).catch(() => {});
  }, []);

  const startIndexing = async () => {
    if (picked === "") return;
    try {
      await enqueueIndex(picked);
      setStartError(null);
    } catch (e) {
      setStartError(String(e));
    }
  };

  return (
    <main>
      <header>
        <div>
          <h1>Activity</h1>
          <p className="muted">Background work runs one task at a time, in order.</p>
        </div>
        {finished.length > 0 && (
          <button className="ghost" onClick={() => clearFinishedTasks()}>
            Clear finished
          </button>
        )}
      </header>
      {startError && <div className="banner bad">{startError}</div>}

      <h2>Now and next</h2>
      <section className="stack">
        {active.length === 0 ? (
          <div className="card empty-state">
            <span className="empty-icon">
              <Inbox size={22} strokeWidth={2} aria-hidden />
            </span>
            <div className="empty-body">
              <div className="empty-title">Nothing running</div>
              <p className="muted small">
                Indexing, exports, rendering and model downloads queue here and run one at a time. Finished tasks
                stay listed under Recent until you clear them.
              </p>
              {projects.length > 0 && (
                <div className="empty-actions">
                  <select
                    aria-label="Project to index"
                    value={picked}
                    onChange={(e) => setPicked(e.currentTarget.value === "" ? "" : Number(e.currentTarget.value))}
                  >
                    <option value="" disabled>
                      Pick a project…
                    </option>
                    {projects.map((p) => (
                      <option key={p.project.id} value={p.project.id}>
                        {p.project.name}
                      </option>
                    ))}
                  </select>
                  <button disabled={picked === ""} onClick={startIndexing}>
                    Start indexing
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          active.map((t) => <TaskCard key={t.id} task={t} />)
        )}
      </section>

      <h2>Recent</h2>
      <section className="stack">
        {finished.length === 0 ? (
          <div className="muted small">
            Finished tasks appear here — indexing runs, exports and downloads stay listed until you clear them.
          </div>
        ) : (
          finished.map((t) => <TaskCard key={t.id} task={t} />)
        )}
      </section>
    </main>
  );
}
