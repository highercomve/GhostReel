import { useCallback, useEffect, useState } from "react";
import { createProject, fpsLabel, FPS_PRESETS, isDesktop, listProjects, openSearchWindow, focusMainWindow, type ProjectSummary } from "./api";
import { Activity, Cpu, HardDrive, Info, PanelLeftClose, PanelLeftOpen, Plus, Search, Settings } from "lucide-react";
import ActivityPage from "./ActivityPage";
import ModelsPage from "./ModelsPage";
import ProjectPage from "./ProjectPage";
import StatusPage from "./StatusPage";
import StoragePage from "./StoragePage";
import { useUpdater } from "./UpdaterPanel";
import { onEvent } from "./events";
import SettingsPage from "./SettingsPage";
import { isActive, useQueue } from "./useQueue";
import { NAVIGATE_EVENT } from "./navigate";

type Page = { kind: "status" } | { kind: "models" } | { kind: "settings" } | { kind: "storage" } | { kind: "activity" } | { kind: "project"; id: number; openVideo?: { videoId: number; t: number } };

function NewProject({ onCreated, onCancel }: { onCreated: (id: number) => void; onCancel: () => void }) {
  const [name, setName] = useState("");
  const [fps, setFps] = useState("29.97");
  const [res, setRes] = useState("1920x1080");
  const [transcribe, setTranscribe] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userSelected, setUserSelected] = useState(false);

  useEffect(() => {
    listProjects()
      .then((list) => {
        if (!userSelected && list.length > 0) {
          const p = list[0].project;
          const detectedFps = fpsLabel(p.fps_num, p.fps_den);
          if (FPS_PRESETS.some((preset) => preset.label === detectedFps)) {
            setFps(detectedFps);
          }
          const detectedRes = `${p.width}x${p.height}`;
          if (["1920x1080", "3840x2160", "1080x1920", "1080x1080"].includes(detectedRes)) {
            setRes(detectedRes);
          }
        }
      })
      .catch(() => {});
  }, [userSelected]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const preset = FPS_PRESETS.find((p) => p.label === fps) ?? FPS_PRESETS[0];
    const [w, h] = res.split("x").map(Number);
    try {
      const p = await createProject(name.trim(), preset.num, preset.den, w, h, { transcribe });
      onCreated(p.id);
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <form className="new-project" onSubmit={submit}>
      <div className="new-project-field">
        <label htmlFor="np-name" className="new-project-label">Project name</label>
        <input
          id="np-name"
          autoFocus
          placeholder="e.g. Vacation reel"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>
      <div className="new-project-row">
        <div className="new-project-field">
          <label htmlFor="np-fps" className="new-project-label">Frame rate</label>
          <select
            id="np-fps"
            value={fps}
            onChange={(e) => {
              setFps(e.target.value);
              setUserSelected(true);
            }}
            title="Sequence frame rate"
          >
            {FPS_PRESETS.map((p) => (
              <option key={p.label} value={p.label}>{p.label} fps</option>
            ))}
          </select>
        </div>
        <div className="new-project-field">
          <label htmlFor="np-res" className="new-project-label">Resolution</label>
          <select
            id="np-res"
            value={res}
            onChange={(e) => {
              setRes(e.target.value);
              setUserSelected(true);
            }}
            title="Sequence resolution"
          >
            <option value="1920x1080">1080p</option>
            <option value="3840x2160">4K</option>
            <option value="1080x1920">9:16</option>
            <option value="1080x1080">1:1</option>
          </select>
        </div>
      </div>
      <label className="checkbox-row" title="Uncheck if your footage is b-roll or has no speech to transcribe">
        <input type="checkbox" checked={transcribe} onChange={(e) => setTranscribe(e.target.checked)} />
        <span>Transcribe audio (speech-to-text)</span>
      </label>
      {error && <div className="bad-text small">{error}</div>}
      <div className="inline new-project-actions">
        <button type="submit" className="primary" disabled={!name.trim()}>
          Create project
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
  const updater = useUpdater();
  const { checkForUpdates } = updater;
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

  useEffect(() => {
    if (!isDesktop) return;
    return onEvent("check-for-updates", () => {
      setPage({ kind: "status" });
      void checkForUpdates();
    });
  }, [checkForUpdates]);

  // Cross-page links (e.g. "Set up on the Models page" on Status) dispatch ghostreel:navigate.
  useEffect(() => {
    const onPageNavigate = (e: Event) => {
      const kind = (e as CustomEvent<{ kind?: unknown }>).detail?.kind;
      if (kind === "models" || kind === "settings" || kind === "storage" || kind === "activity" || kind === "status") {
        setPage({ kind });
      }
    };
    window.addEventListener(NAVIGATE_EVENT, onPageNavigate);
    return () => window.removeEventListener(NAVIGATE_EVENT, onPageNavigate);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        void openSearchWindow();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    return onEvent<{ projectId: number; videoId?: number; t?: number }>(
      "navigate-to-project",
      async ({ projectId, videoId, t }) => {
        await focusMainWindow();
        setPage({ kind: "project", id: projectId,
          openVideo: videoId !== undefined && t !== undefined ? { videoId, t } : undefined,
        });
      },
    );
  }, []);

  // The sidebar folds to a rail of icons, for room in the editor; the choice is remembered.
  const [editorMode, setEditorMode] = useState(false);
  const [libraryCollapsed, setLibraryCollapsed] = useState(() => localStorage.getItem("ghostreel.navCollapsed") === "1");
  const [editorCollapsed, setEditorCollapsed] = useState(() => localStorage.getItem("ghostreel.editorNavCollapsed") !== "0");
  const navCollapsed = editorMode ? editorCollapsed : libraryCollapsed;
  useEffect(() => {
    const onMode = (event: Event) => setEditorMode(Boolean((event as CustomEvent<boolean>).detail));
    window.addEventListener("ghostreel:editor-mode", onMode);
    return () => window.removeEventListener("ghostreel:editor-mode", onMode);
  }, []);
  const toggleNav = () => {
    const setCollapsed = editorMode ? setEditorCollapsed : setLibraryCollapsed;
    const key = editorMode ? "ghostreel.editorNavCollapsed" : "ghostreel.navCollapsed";
    setCollapsed((c) => {
      localStorage.setItem(key, c ? "0" : "1");
      return !c;
    });
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === "b") {
        const t = e.target as HTMLElement | null;
        if (t?.closest("input, textarea, [contenteditable=true]")) return;
        e.preventDefault();
        toggleNav();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editorMode]);

  return (
    <div className={`shell${navCollapsed ? " nav-collapsed" : ""}`}>
      <nav>
        <div className="brand">
          <img src="/icon.png" alt="" width={28} height={28} />
          <span className="nav-label">GhostReel</span>
          <button
            type="button"
            className="nav-collapse-btn"
            onClick={toggleNav}
            title={navCollapsed ? "Show the sidebar (Ctrl+B)" : "Hide the sidebar (Ctrl+B)"}
            aria-label={navCollapsed ? "Show sidebar" : "Hide sidebar"}
            aria-expanded={!navCollapsed}
          >
            {navCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>
        <button
          type="button"
          className="nav-item search-material-btn"
          onClick={() => void openSearchWindow()}
          title="Search Material Globally (Ctrl+Shift+F / ⇧⌘F)"
        >
          <span className="search-nav-content">
            <span className="nav-icon"><Search size={16} /></span>
            <span className="search-nav-label nav-label">Search Material</span>
            <kbd className="nav-shortcut">⇧⌘F</kbd>
          </span>
        </button>
        <div className="nav-title">Projects</div>
        {projects.map(({ project, status }) => (
          <button
            key={project.id}
            className={`nav-item ${page.kind === "project" && page.id === project.id ? "active" : ""}`}
            onClick={() => setPage({ kind: "project", id: project.id })}
            title={navCollapsed ? project.name : undefined}
          >
            <span className="nav-icon nav-avatar">{project.name.trim().charAt(0).toUpperCase() || "?"}</span>
            <span className="nav-label">{project.name}</span>
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
          <button className="nav-item add" onClick={() => setCreating(true)} title="New project">
            <span className="nav-icon"><Plus size={16} /></span>
            <span className="nav-label">New project</span>
          </button>
        )}
        <div className="spacer" />
        <button
          className={`nav-item ${page.kind === "activity" ? "active" : ""}`}
          onClick={() => setPage({ kind: "activity" })}
          title={runningTask?.label ?? (navCollapsed ? "Activity" : undefined)}
        >
          <span className="nav-icon"><Activity size={16} /></span>
          <span className="nav-label">Activity</span>
          {activeCount > 0 && (
            <span className="count busy">
              {runningTask?.progress ? `${Math.round(runningTask.progress.fraction * 100)}%` : activeCount}
            </span>
          )}
        </button>
        <button
          className={`nav-item ${page.kind === "models" ? "active" : ""}`}
          onClick={() => setPage({ kind: "models" })}
          title={navCollapsed ? "Models" : undefined}
        >
          <span className="nav-icon"><Cpu size={16} /></span>
          <span className="nav-label">Models</span>
        </button>
        <button
          className={`nav-item ${page.kind === "storage" ? "active" : ""}`}
          onClick={() => setPage({ kind: "storage" })}
          title={navCollapsed ? "Storage" : undefined}
        >
          <span className="nav-icon"><HardDrive size={16} /></span>
          <span className="nav-label">Storage</span>
        </button>
        <button
          className={`nav-item ${page.kind === "settings" ? "active" : ""}`}
          onClick={() => setPage({ kind: "settings" })}
          title={navCollapsed ? "Settings" : undefined}
        >
          <span className="nav-icon"><Settings size={16} /></span>
          <span className="nav-label">Settings</span>
        </button>
        <button
          className={`nav-item ${page.kind === "status" ? "active" : ""}`}
          onClick={() => setPage({ kind: "status" })}
          title={navCollapsed ? "Status" : undefined}
        >
          <span className="nav-icon"><Info size={16} /></span>
          <span className="nav-label">Status</span>
        </button>
      </nav>
      <div className="content">
        {page.kind === "status" ? (
          <StatusPage updater={updater} />
        ) : page.kind === "models" ? (
          <ModelsPage />
        ) : page.kind === "storage" ? (
          <StoragePage onChanged={refresh} />
        ) : page.kind === "settings" ? (
          <SettingsPage />
        ) : page.kind === "activity" ? (
          <ActivityPage />
        ) : (
          <ProjectPage key={page.id} projectId={page.id} openVideo={page.openVideo} onChanged={refresh} />
        )}
      </div>
    </div>
  );
}
