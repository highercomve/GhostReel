import { useCallback, useEffect, useState } from "react";
import { createProject, FPS_PRESETS, listProjects, type ProjectSummary } from "./api";
import ActivityPage from "./ActivityPage";
import ModelsPage from "./ModelsPage";
import ProjectPage from "./ProjectPage";
import StatusPage from "./StatusPage";
import SettingsPage from "./SettingsPage";
import { isActive, useQueue } from "./useQueue";

type Page = { kind: "status" } | { kind: "models" } | { kind: "settings" } | { kind: "activity" } | { kind: "project"; id: number };

function NewProject({ onCreated, onCancel }: { onCreated: (id: number) => void; onCancel: () => void }) {
  const [name, setName] = useState("");
  const [fps, setFps] = useState("25");
  const [res, setRes] = useState("1920x1080");
  const [transcribe, setTranscribe] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const preset = FPS_PRESETS.find((p) => p.label === fps)!;
    const [w, h] = res.split("x").map(Number);
    try {
      const p = await createProject(name, preset.num, preset.den, w, h, { transcribe });
      onCreated(p.id);
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <form className="new-project" onSubmit={submit}>
      <input autoFocus placeholder="Project name" value={name} onChange={(e) => setName(e.target.value)} />
      <div className="inline">
        <select value={fps} onChange={(e) => setFps(e.target.value)} title="Sequence frame rate">
          {FPS_PRESETS.map((p) => (
            <option key={p.label}>{p.label}</option>
          ))}
        </select>
        <select value={res} onChange={(e) => setRes(e.target.value)} title="Sequence resolution">
          <option value="1920x1080">1080p</option>
          <option value="3840x2160">4K</option>
          <option value="1080x1920">9:16</option>
          <option value="1080x1080">1:1</option>
        </select>
      </div>
      <label className="checkbox-row" title="Uncheck if your footage is b-roll or has no speech to transcribe">
        <input type="checkbox" checked={transcribe} onChange={(e) => setTranscribe(e.target.checked)} />
        <span>Transcribe audio (speech-to-text)</span>
      </label>
      {error && <div className="bad-text small">{error}</div>}
      <div className="inline">
        <button type="submit" disabled={!name.trim()}>
          Create
        </button>
        <button type="button" className="ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [page, setPage] = useState<Page>({ kind: "status" });
  const [creating, setCreating] = useState(false);
  const tasks = useQueue();
  const activeCount = tasks.filter(isActive).length;
  const runningTask = tasks.find((t) => t.state === "running");

  const refresh = useCallback(async () => {
    const list = await listProjects().catch(() => []);
    setProjects(list);
    setPage((p) => (p.kind === "project" && !list.some((x) => x.project.id === p.id) ? { kind: "status" } : p));
  }, []);

  useEffect(() => {
    refresh().then(() =>
      listProjects()
        .then((l) => l.length > 0 && setPage({ kind: "project", id: l[0].project.id }))
        .catch(() => {}),
    );
  }, [refresh]);

  return (
    <div className="shell">
      <nav>
        <div className="brand">
          <img src="/icon.png" alt="" width={28} height={28} />
          <span>GhostReel</span>
        </div>
        <div className="nav-title">Projects</div>
        {projects.map(({ project, status }) => (
          <button
            key={project.id}
            className={`nav-item ${page.kind === "project" && page.id === project.id ? "active" : ""}`}
            onClick={() => setPage({ kind: "project", id: project.id })}
          >
            <span>{project.name}</span>
            <span className="count">{status.videos}</span>
          </button>
        ))}
        {creating ? (
          <NewProject
            onCancel={() => setCreating(false)}
            onCreated={async (id) => {
              setCreating(false);
              await refresh();
              setPage({ kind: "project", id });
            }}
          />
        ) : (
          <button className="nav-item add" onClick={() => setCreating(true)}>
            + New project
          </button>
        )}
        <div className="spacer" />
        <button
          className={`nav-item ${page.kind === "activity" ? "active" : ""}`}
          onClick={() => setPage({ kind: "activity" })}
          title={runningTask?.label}
        >
          <span>Activity</span>
          {activeCount > 0 && (
            <span className="count busy">
              {runningTask?.progress ? `${Math.round(runningTask.progress.fraction * 100)}%` : activeCount}
            </span>
          )}
        </button>
        <button
          className={`nav-item ${page.kind === "models" ? "active" : ""}`}
          onClick={() => setPage({ kind: "models" })}
        >
          Models
        </button>
        <button
          className={`nav-item ${page.kind === "settings" ? "active" : ""}`}
          onClick={() => setPage({ kind: "settings" })}
        >
          Settings
        </button>
        <button
          className={`nav-item ${page.kind === "status" ? "active" : ""}`}
          onClick={() => setPage({ kind: "status" })}
        >
          Status
        </button>
      </nav>
      <div className="content">
        {page.kind === "status" ? (
          <StatusPage />
        ) : page.kind === "models" ? (
          <ModelsPage />
        ) : page.kind === "settings" ? (
          <SettingsPage />
        ) : page.kind === "activity" ? (
          <ActivityPage />
        ) : (
          <ProjectPage key={page.id} projectId={page.id} onChanged={refresh} />
        )}
      </div>
    </div>
  );
}
