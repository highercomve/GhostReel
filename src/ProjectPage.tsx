import MediaImage from "./MediaImage";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ask, open } from "@tauri-apps/plugin-dialog";
import {
  addFolder,
  clock,
  enqueueIndex,
  enqueueSteadiness,
  fileName,
  fpsLabel,
  humanDuration,
  humanSize,
  projectView,
  removeFolder,
  redoProjectStage,
  getAiSettings,
  setAiSettings,
  setProjectPipeline,
  removeProject,
  excludeVideo,
  includeVideo,
  renameProject,
  search,
  setVideoCategory,
  CATEGORIES,
  type Hit,
  type ProjectView,
  type VideoRow,
} from "./api";
import { onEvent } from "./events";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";
import VideoPanel from "./VideoPanel";
import ScriptsPanel from "./ScriptsPanel";
import OrganizeModal from "./OrganizeModal";
import { useVideoSelection } from "./useVideoSelection";

function SpeechCell({ v, transcribeEnabled }: { v: VideoRow; transcribeEnabled?: boolean }) {
  if (v.segments > 0) return <span className="good-text">{v.language ? v.language.toUpperCase() : "✓"}</span>;
  switch (v.transcribe) {
    case "skipped":
      return <span className="muted">{v.has_audio ? "skipped" : "no audio"}</span>;
    case "failed":
      return <span className="bad-text">failed</span>;
    case "done":
      return <span className="muted">no speech</span>;
    default:
      return <span className="muted">{transcribeEnabled === false ? "disabled" : "waiting"}</span>;
  }
}

function CategoryCell({
  v,
  onCategoryChange,
}: {
  v: VideoRow;
  onCategoryChange: (newCategory: string) => void;
}) {
  const current = CATEGORIES.find((c) => c.id === v.category) ?? {
    id: v.category,
    label: v.category,
    folder: v.category,
    icon: "📁",
  };

  return (
    <div className="cat-cell" onClick={(e) => e.stopPropagation()}>
      <select
        value={v.category}
        onChange={(e) => onCategoryChange(e.target.value)}
        className={`cat-select cat-${v.category}`}
        title={`Category: ${current.label} (${v.category_source === "manual" ? "manual" : `${Math.round((v.category_confidence || 0) * 100)}% confidence`})`}
      >
        {CATEGORIES.map((cat) => (
          <option key={cat.id} value={cat.id}>
            {cat.icon} {cat.label}
          </option>
        ))}
      </select>
      {v.category_tag && (
        <span
          className="cat-subtag"
          title={`Detected tag: ${v.category_tag}`}
        >
          {v.category_tag === "music" ? "🎵 music" : v.category_tag}
        </span>
      )}
    </div>
  );
}

/** "…the [compute] module…" → highlighted spans. */
function Snippet({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <span>
      {parts.map((p, i) =>
        p.startsWith("[") && p.endsWith("]") ? <mark key={i}>{p.slice(1, -1)}</mark> : <span key={i}>{p}</span>,
      )}
    </span>
  );
}

export default function ProjectPage({ projectId, onChanged, openVideo }: {
  projectId: number; onChanged: () => void; openVideo?: { videoId: number; t: number };
}) {
  const [tab, setTab] = useState<"library" | "scripts">("library");
  const [view, setView] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [seek, setSeek] = useState<{ t: number; nonce: number } | null>(null);
  const [query, setQuery] = useState("");
  const [searchResult, setSearchResult] = useState<{ key: string; hits: Hit[]; note: string | null } | null>(null);
  const [searchingKey, setSearchingKey] = useState<string | null>(null);
  const searchSequence = useRef(0);
  const [folderId, setFolderId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string | null>(null);
  const [samplingSize, setSamplingSize] = useState<number>(768);
  const [organizeModalOpen, setOrganizeModalOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const tasks = useQueue();
  const searchInput = useRef<HTMLInputElement>(null);
  const scopeFolder = view?.folders.find((f) => f.id === folderId) ?? null;
  const folderVideos = useMemo(() => view?.videos.filter((v) =>
    folderId == null || v.folder_ids?.includes(folderId)) ?? [], [view, folderId]);
  const visibleVideos = useMemo(() => folderVideos.filter((v) =>
    categoryFilter == null || v.category === categoryFilter), [folderVideos, categoryFilter]);
  const selection = useVideoSelection(visibleVideos.map((v) => v.id));
  const selectedIds = folderVideos.filter((v) => selection.ids.has(v.id)).map((v) => v.id).sort((a, b) => a - b);
  const hiddenSelected = selectedIds.length - visibleVideos.filter((v) => selection.ids.has(v.id)).length;
  const searchKey = JSON.stringify([projectId, folderId, selectedIds, query.trim()]);
  const hits = searchResult?.key === searchKey ? searchResult.hits : null;
  const searchNote = searchResult?.key === searchKey ? searchResult.note : null;
  const searching = searchingKey === searchKey;

  const changeFolder = (id: number | null) => { selection.clear(); setFolderId(id); };
  useEffect(() => {
    // A response for an older query or scope must not replace the current search.
    searchSequence.current += 1;
    setSearchingKey(null);
    setSearchResult(null);
  }, [searchKey]);

  useEffect(() => {
    const current = new Set(view?.videos.map((v) => v.id) ?? []);
    selection.setIds((previous) => {
      const next = new Set([...previous].filter((id) => current.has(id)));
      return next.size === previous.size ? previous : next;
    });
    if (view && folderId != null && !view.folders.some((f) => f.id === folderId)) changeFolder(null);
  }, [view]); // selection is pruned on refresh, not when category filters hide rows

  useEffect(() => {
    if (!menuOpen) return;
    const onClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onClickOutside);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onClickOutside);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  const refresh = useCallback(async () => {
    try {
      setView(await projectView(projectId));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
    try {
      const s = await getAiSettings();
      setSamplingSize(s.frames.long_side ?? 768);
    } catch {
      /* ignore */
    }
  }, [projectId]);

  useEffect(() => {
    setView(null);
    setSelected(null);
    setSearchResult(null);
    setSearchingKey(null);
    searchSequence.current += 1;
    selection.clear();
    selection.setEnabled(false);
    setFolderId(null);
    setCategoryFilter(null);
    setQuery("");
    setEditName(null);
    refresh();
  }, [refresh]);

  // Keep the table fresh while this project's task runs, and once it finishes.
  const myTasks = tasks.filter((t) => t.kind.type === "index" && t.kind.project_id === projectId);
  const running = myTasks.find((t) => t.state === "running");
  const queued = myTasks.find((t) => t.state === "queued");
  const lastFinished = [...myTasks].reverse().find((t) => t.state !== "running" && t.state !== "queued");
  useEffect(() => {
    if (!running) return;
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, [running?.id, refresh]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    return onEvent<number>("task-finished", () => {
      refresh();
      onChanged();
    });
  }, [refresh, onChanged]);

  const run = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      await refresh();
      onChanged();
    } catch (e) {
      setError(String(e));
    }
  };

  const onAddFolder = async () => {
    const dir = await open({ directory: true, multiple: false, title: "Add a video folder" });
    if (typeof dir === "string") run(() => addFolder(projectId, dir));
  };

  const onRebuildFrames = async () => {
    let interval = "the current";
    let size = samplingSize;
    try {
      const s = await getAiSettings();
      interval = `${s.frames.max_interval_s} s`;
      size = s.frames.long_side ?? samplingSize;
    } catch {
      /* keep generic wording */
    }
    const ok = await ask(
      `Re-extract and re-describe all keyframes of this project with ${interval} interval at ${size}px sampling size? Transcripts are kept. This re-runs frame descriptions and search embeddings, which can take a while.`,
      { title: "Rebuild keyframes", kind: "warning" },
    );
    if (ok) run(() => redoProjectStage(projectId, "frames"));
  };

  const onDelete = async () => {
    if (!view) return;
    const ok = await ask(`Delete project "${view.project.name}"? Your video files are not touched.`, {
      title: "Delete project",
      kind: "warning",
    });
    if (!ok) return;
    // Second question: the index (keyframes, transcripts, previews) can be kept for reuse, since
    // re-indexing the same footage is slow.
    const purge = await ask(
      "Also delete what was indexed for it — keyframes, preview renders, and the transcripts, descriptions and search index of footage no other project uses?\n\nKeep it if you may add this footage to another project later. Your video files are not touched either way.",
      { title: "Delete the indexed data too?", kind: "warning", okLabel: "Delete indexed data", cancelLabel: "Keep it" },
    );
    run(() => removeProject(projectId, purge));
  };

  const onRename = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (editName == null || !view) return;
    const name = editName.trim();
    if (!name || name === view.project.name) {
      setEditName(null);
      return;
    }
    try {
      await renameProject(projectId, name);
      setEditName(null);
      await refresh();
      onChanged();
    } catch (err) {
      setError(String(err));
    }
  };

  const onSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) { setSearchResult(null); return; }
    const sequence = ++searchSequence.current;
    setSearchingKey(searchKey);
    setError(null);
    try {
      const r = await search(projectId, q, 30, {
        folderId, videoIds: selectedIds.length > 0 ? selectedIds : null,
      });
      if (sequence === searchSequence.current) setSearchResult({ key: searchKey, ...r });
    } catch (err) {
      if (sequence === searchSequence.current) setError(String(err));
    } finally {
      if (sequence === searchSequence.current) setSearchingKey(null);
    }
  };

  const openAt = (videoId: number, t: number) => {
    setSelected(videoId);
    setSeek({ t, nonce: Date.now() });
    setTimeout(() => panelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  useEffect(() => {
    return onEvent<{ videoId: number; t: number }>("open-video-at", ({ videoId, t }) => {
      openAt(videoId, t);
    });
  }, []);

  const navigationReady = !!openVideo && view?.project.id === projectId &&
    view.videos.some((video) => video.id === openVideo.videoId);
  useEffect(() => {
    if (navigationReady && openVideo) openAt(openVideo.videoId, openVideo.t);
  }, [navigationReady, openVideo]);

  if (!view) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const { project: p, status: st } = view;
  const stage = (name: string) => st.stages.find((s) => s.stage === name);
  const selectedVideo = view.videos.find((v) => v.id === selected) ?? null;

  return (
    <main className="project-workspace">
      <header>
        <div>
          {editName != null ? (
            <form className="rename" onSubmit={onRename}>
              <input
                autoFocus
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onBlur={() => onRename()}
                onKeyDown={(e) => e.key === "Escape" && setEditName(null)}
              />
            </form>
          ) : (
            <h1 className="editable" title="Click to rename" onClick={() => setEditName(p.name)}>
              {p.name} <span className="edit-hint">✎</span>
            </h1>
          )}
          <p className="muted">
            {p.width}×{p.height} · {fpsLabel(p.fps_num, p.fps_den)} fps · {st.videos} videos ·{" "}
            {humanDuration(st.total_duration_s)} · {humanSize(st.total_size)}
          </p>
        </div>
        <div className="header-actions">
          <div className="project-menu-container" ref={menuRef}>
            <button
              type="button"
              className="ghost"
              onClick={() => setMenuOpen(!menuOpen)}
              aria-expanded={menuOpen}
              title="Project settings, video analysis & management"
            >
              ⚙ Project ▾
            </button>
            {menuOpen && (
              <div className="project-menu">
                <div className="project-menu-section-title">Video Analysis</div>

                <div className="project-menu-field">
                  <span className="project-menu-label">Vision / playback size</span>
                  <select
                    className="project-menu-select"
                    value={samplingSize}
                    onChange={async (e) => {
                      const v = Number(e.target.value);
                      setSamplingSize(v);
                      try {
                        await setAiSettings({ frames: { long_side: v } });
                      } catch {
                        /* ignore */
                      }
                    }}
                    disabled={!!running || !!queued}
                  >
                    <option value={512}>512 px (fastest)</option>
                    <option value={640}>640 px (fast)</option>
                    <option value={768}>768 px (default)</option>
                    <option value={1024}>1024 px (high detail)</option>
                    <option value={1280}>1280 px (full detail)</option>
                  </select>
                </div>

                <button
                  type="button"
                  className="project-menu-item"
                  onClick={() => {
                    setMenuOpen(false);
                    onRebuildFrames();
                  }}
                  disabled={!!running || !!queued || view.videos.length === 0}
                  title="Re-extract and re-describe keyframes with current interval and sampling size"
                >
                  <span>Rebuild keyframes…</span>
                </button>

                {view.videos.some((v) => !v.steadiness_measured) ? (
                  <button
                    type="button"
                    className="project-menu-item"
                    onClick={() => {
                      setMenuOpen(false);
                      run(() => enqueueSteadiness(projectId, false));
                    }}
                    disabled={!!running || !!queued}
                    title="Analyze camera steadiness/shake for unmeasured videos in this project"
                  >
                    <span>Analyze camera shake</span>
                    <span className="tag warn small">pending</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="project-menu-item"
                    onClick={async () => {
                      setMenuOpen(false);
                      const ok = await ask("Re-measure camera steadiness for all videos in this project?", {
                        title: "Re-analyze camera shake",
                      });
                      if (ok) run(() => enqueueSteadiness(projectId, true));
                    }}
                    disabled={!!running || !!queued || view.videos.length === 0}
                    title="Re-measure camera steadiness for all videos in this project"
                  >
                    <span>Re-analyze camera shake…</span>
                  </button>
                )}

                <div className="project-menu-divider" />

                <div className="project-menu-section-title">Pipeline Stages</div>

                <label className="project-menu-checkbox" title="Transcribe speech with Whisper">
                  <input
                    type="checkbox"
                    checked={p.pipeline.transcribe}
                    disabled={!!running || !!queued}
                    onChange={async (e) => {
                      const next = { ...p.pipeline, transcribe: e.target.checked };
                      await run(() => setProjectPipeline(projectId, next));
                    }}
                  />
                  <span>Transcription</span>
                </label>

                <label className="project-menu-checkbox" title="Sample keyframes from video">
                  <input
                    type="checkbox"
                    checked={p.pipeline.frames}
                    disabled={!!running || !!queued}
                    onChange={async (e) => {
                      const next = { ...p.pipeline, frames: e.target.checked };
                      await run(() => setProjectPipeline(projectId, next));
                    }}
                  />
                  <span>Keyframes</span>
                </label>

                <label className="project-menu-checkbox" title="Describe keyframes with vision model">
                  <input
                    type="checkbox"
                    checked={p.pipeline.describe}
                    disabled={!!running || !!queued}
                    onChange={async (e) => {
                      const next = { ...p.pipeline, describe: e.target.checked };
                      await run(() => setProjectPipeline(projectId, next));
                    }}
                  />
                  <span>Frame descriptions</span>
                </label>

                <label className="project-menu-checkbox" title="Generate vector embeddings and search chunks">
                  <input
                    type="checkbox"
                    checked={p.pipeline.embed}
                    disabled={!!running || !!queued}
                    onChange={async (e) => {
                      const next = { ...p.pipeline, embed: e.target.checked };
                      await run(() => setProjectPipeline(projectId, next));
                    }}
                  />
                  <span>Search embeddings</span>
                </label>

                <div className="project-menu-divider" />

                <div className="project-menu-section-title">Project</div>

                <button
                  type="button"
                  className="project-menu-item"
                  onClick={() => {
                    setMenuOpen(false);
                    setEditName(p.name);
                  }}
                  title="Rename this project"
                >
                  <span>Rename project</span>
                </button>

                <button
                  type="button"
                  className="project-menu-item danger"
                  onClick={() => {
                    setMenuOpen(false);
                    onDelete();
                  }}
                  title="Delete this project"
                >
                  <span>Delete project…</span>
                </button>
              </div>
            )}
          </div>

          <button onClick={() => run(() => enqueueIndex(projectId))} disabled={!!queued || view.folders.length === 0}>
            {running ? "Index again" : queued ? "Queued…" : "Index now"}
          </button>
        </div>
      </header>

      <div className="tab-nav">
        <button
          type="button"
          className={`tab-btn ${tab === "library" ? "active" : ""}`}
          onClick={() => setTab("library")}
        >
          Library
        </button>
        <button
          type="button"
          className={`tab-btn ${tab === "scripts" ? "active" : ""}`}
          onClick={() => setTab("scripts")}
        >
          Scripts
        </button>
      </div>

      {tab === "scripts" ? (
        <ScriptsPanel projectId={projectId} />
      ) : (
        <>
          {error && <div className="banner bad">{error}</div>}
          {running && <TaskCard task={running} compact />}
          {!running && queued && <TaskCard task={queued} compact />}
          {!running && !queued && lastFinished && lastFinished.state !== "done" && <TaskCard task={lastFinished} compact />}

      <div className="search-scope-bar">
        <label htmlFor="search-folder">Search in</label>
        <select id="search-folder" value={folderId ?? ""}
          onChange={(e) => changeFolder(e.target.value ? Number(e.target.value) : null)}>
          <option value="">All watched folders</option>
          {view.folders.map((f) => <option key={f.id} value={f.id}>{f.path}</option>)}
        </select>
        <span className="search-scope-summary" role="status">
          {selectedIds.length > 0 ? `${selectedIds.length} selected video${selectedIds.length === 1 ? "" : "s"}`
            : `${folderVideos.length} video${folderVideos.length === 1 ? "" : "s"} · ${scopeFolder ? "this folder" : "whole project"}`}
        </span>
        {selectedIds.length > 0 && <button type="button" className="ghost small" onClick={selection.clear}>Clear selection</button>}
      </div>
      <form className="search" onSubmit={onSearch}>
        <input
          ref={searchInput}
          aria-label="Search project footage"
          placeholder={selectedIds.length ? `Search ${selectedIds.length} selected videos…`
            : scopeFolder ? "Search this watched folder…" : "Search this project: “unboxing the board”, “wifi password”, on-screen text…"}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (!e.target.value.trim()) setSearchResult(null);
          }}
        />
        <button type="submit" disabled={searching || !query.trim()}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>
      {searchNote && <div className="muted small">{searchNote}</div>}
      {hits && (
        <section className="hits">
          {hits.length === 0 && <div className="muted">No matches.</div>}
          {hits.map((h, i) => (
            <div key={i} className="card hit" onClick={() => openAt(h.video_id, h.start_s)}>
              {h.frame ? <MediaImage path={h.frame} alt="" loading="lazy" /> : <div className="noframe" />}
              <div className="hit-body">
                <div className="hit-head">
                  <span className="label">{fileName(h.path)}</span>
                  <span className="time">
                    {clock(h.start_s)}–{clock(h.end_s)}
                  </span>
                </div>
                <div className="small">
                  <Snippet text={h.snippet} />
                </div>
                <div>
                  {h.kinds.map((k) => (
                    <span key={k} className="tag">
                      {k === "moment" ? "picture" : k === "frame" ? "on screen" : "speech"}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </section>
      )}

      <div ref={panelRef}>
        {selectedVideo && (
          <VideoPanel
            video={selectedVideo}
            seek={seek}
            onClose={() => setSelected(null)}
            onExclude={async (role) => {
              const name = fileName(selectedVideo.path);
              const ok = await ask(
                role === "reference"
                  ? `Use "${name}" as a reference edit? The script chat will study it (length, pacing, structure) but never use it as footage. It disappears from search.`
                  : `Remove "${name}" from this project's library? It won't be searched, indexed or used in scripts. The file is not deleted.`,
                { title: role === "reference" ? "Reference edit" : "Remove from library", kind: "warning" },
              );
              if (!ok) return;
              setSelected(null);
              run(() => excludeVideo(projectId, selectedVideo.id, role));
            }}
          />
        )}
      </div>

      <h2>Folders</h2>
      <section className="card list">
        {view.folders.length === 0 && <div className="muted">Add the folders that hold this project's footage.</div>}
        {view.folders.map((f) => (
          <div key={f.id} className="row folder">
            <span className={f.available ? "" : "bad-text"}>{f.available ? "📁" : "⚠"}</span>
            <span className="path">
              {f.path}
              {!f.available && <span className="muted small"> — not available (drive unplugged?)</span>}
            </span>
            <div className="folder-actions">
              <button className={`ghost small${folderId === f.id ? " active" : ""}`}
                onClick={() => { changeFolder(f.id); searchInput.current?.focus(); }}>
                Search here
              </button>
              <button className="ghost small" onClick={() => run(() => removeFolder(projectId, f.path))}>
                Remove
              </button>
            </div>
          </div>
        ))}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button className="ghost" onClick={onAddFolder}>
            + Add folder
          </button>
          <button
            className="ghost"
            onClick={() => setOrganizeModalOpen(true)}
            disabled={view.videos.length === 0 || !!running}
            title="Organize original video files into categorized subfolders on disk"
          >
            🗂️ Organize folders…
          </button>
        </div>
      </section>

      <h2>
        Videos
        <span className="muted small normal">
          {" "}
          · {stage("probe")?.done ?? 0} ready ·{" "}
          {p.pipeline?.transcribe ? `${stage("transcribe")?.done ?? 0} transcribed` : "transcription disabled"} ·{" "}
          {p.pipeline?.describe ? `${stage("describe")?.done ?? 0} described` : "descriptions disabled"} ·{" "}
          {p.pipeline?.embed ? `${stage("embed")?.done ?? 0} searchable` : "search indexing disabled"}
          {st.vfr_videos ? ` · ${st.vfr_videos} variable frame rate` : ""}
        </span>
      </h2>

      {view.videos.length > 0 && (
        <div className="category-filter-bar">
          <button
            type="button"
            className={`filter-pill ${categoryFilter === null ? "active" : ""}`}
            onClick={() => setCategoryFilter(null)}
          >
            All ({folderVideos.length})
          </button>
          {CATEGORIES.map((cat) => {
            const count = folderVideos.filter((v) => v.category === cat.id).length;
            if (count === 0 && categoryFilter !== cat.id) return null;
            return (
              <button
                key={cat.id}
                type="button"
                className={`filter-pill cat-${cat.id} ${categoryFilter === cat.id ? "active" : ""}`}
                onClick={() => setCategoryFilter(categoryFilter === cat.id ? null : cat.id)}
              >
                {cat.icon} {cat.label} ({count})
              </button>
            );
          })}
        </div>
      )}

      <div className="video-selection-toolbar">
        <button type="button" className={`ghost small${selection.enabled ? " active" : ""}`}
          aria-pressed={selection.enabled} onClick={() => selection.setEnabled(!selection.enabled)}>Select videos</button>
        <span className="small" role="status">{selectedIds.length ? `${selectedIds.length} selected` : "Select videos to narrow your search"}
          {hiddenSelected > 0 ? ` · ${hiddenSelected} hidden by category` : ""}</span>
        {selectedIds.length > 0 && <>
          <button type="button" className="ghost small" onClick={() => {
            searchInput.current?.focus(); searchInput.current?.scrollIntoView({ behavior: "smooth", block: "center" });
          }}>Search selected</button>
          <button type="button" className="ghost small" onClick={selection.clear}>Clear</button>
        </>}
        <span className="muted small selection-hint">{selection.enabled
          ? "Drag across rows · Ctrl/⌘ adds · Shift-click selects a range · Esc cancels a drag"
          : "Use checkboxes or Ctrl/⌘-click · Shift-click selects a range"}</span>
      </div>
      <section ref={selection.container} {...selection.pointer}
        className={`card video-library${selection.enabled ? " selection-mode" : ""}`}
        tabIndex={0} aria-label="Video library, scroll horizontally to see all columns">
        {view.videos.length === 0 ? (
          <div className="muted">No videos yet — add a folder and press “Index now”.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th className="selection-cell"><input type="checkbox" aria-label="Select all visible videos"
                  checked={selection.all} disabled={visibleVideos.length === 0}
                  ref={(element) => { if (element) element.indeterminate = !selection.all && visibleVideos.some((v) => selection.ids.has(v.id)); }}
                  onChange={selection.toggleAll} /></th>
                <th className="video-name-heading">Name</th>
                <th>Category</th>
                <th>Length</th>
                <th>Size</th>
                <th>Format</th>
                <th>Speech</th>
                <th>Camera</th>
                <th>Frames</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {visibleVideos.map((v) => (
                <tr
                  key={v.id}
                  title={v.path}
                  data-video-id={v.id}
                  className={`clickable ${v.id === selected ? "selected" : ""}${selection.ids.has(v.id) ? " search-selected" : ""}`}
                  onClick={(e) => {
                    if (selection.enabled || e.ctrlKey || e.metaKey || e.shiftKey) {
                      if (e.detail < 2) selection.toggle(v.id, e.shiftKey);
                    }
                    else if (v.id === selected) setSelected(null);
                    else openAt(v.id, 0);
                  }}
                  onDoubleClick={() => { if (selection.enabled) openAt(v.id, 0); }}
                >
                  <td className="selection-cell" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" aria-label={`Select ${fileName(v.path)}`}
                      checked={selection.ids.has(v.id)}
                      onChange={(e) => selection.toggle(v.id, !!(e.nativeEvent as MouseEvent).shiftKey)} />
                  </td>
                  <td className="name">
                    {fileName(v.path)}
                    {v.copies > 1 && <span className="tag">×{v.copies}</span>}
                    {v.vfr && (
                      <span className="tag warn" title="Variable frame rate (phone footage) — may drift in Premiere">
                        VFR
                      </span>
                    )}
                    {v.has_audio === false && <span className="tag">no audio</span>}
                  </td>
                  <td>
                    <CategoryCell
                      v={v}
                      onCategoryChange={(cat) => run(() => setVideoCategory(v.id, cat))}
                    />
                  </td>
                  <td>{v.duration_s != null ? humanDuration(v.duration_s) : "–"}</td>
                  <td>{humanSize(v.size)}</td>
                  <td className="muted">
                    {v.width && v.height ? `${v.width}×${v.height}` : "–"}
                    {v.fps ? ` · ${v.fps.toFixed(v.fps % 1 ? 2 : 0)} fps` : ""}
                    {v.vcodec ? ` · ${v.vcodec}` : ""}
                  </td>
                  <td>
                    <SpeechCell v={v} transcribeEnabled={p.pipeline?.transcribe} />
                  </td>
                  <td>
                    {v.steadiness_measured ? (
                      <>
                        <span className={v.camera === "handheld" ? "" : "muted"}>{v.camera}</span>
                        {v.shaky_s > 0 && (
                          <span
                            className="tag warn"
                            title="Stretches shakier than this clip's own level; open the video to see them"
                          >
                            shaky {Math.round(v.shaky_s)}s
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </td>
                  <td className="muted">{v.frames || "–"}</td>
                  <td>
                    {v.status === "error" ? (
                      <span className="bad-text" title={v.error ?? ""}>
                        error
                      </span>
                    ) : v.status === "probed" ? (
                      <span className="good-text">ready</span>
                    ) : (
                      <span className="muted">waiting</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {view.videos.length > 0 && visibleVideos.length === 0 && <p className="muted">No videos in this folder/category.</p>}
        {selection.box && <div className="video-selection-box" aria-hidden="true" style={selection.box} />}
      </section>

      {view.excluded.length > 0 && (
        <>
          <h2>Not in the library</h2>
          <section className="card list">
            {view.excluded.map((x) => (
              <div key={x.video_id} className="row folder">
                <span className="tag">{x.role === "reference" ? "reference edit" : "removed"}</span>
                <span className="path">{fileName(x.path)}</span>
                <button className="ghost small" onClick={() => run(() => includeVideo(projectId, x.video_id))}>
                  Put back
                </button>
              </div>
            ))}
          </section>
        </>
      )}

      <p className="danger-zone">
        <button className="ghost small danger" onClick={onDelete}>
          Delete project
        </button>
      </p>
        </>
      )}

      {organizeModalOpen && (
        <OrganizeModal
          projectId={projectId}
          onClose={() => setOrganizeModalOpen(false)}
          onOrganized={() => {
            refresh();
            onChanged();
          }}
        />
      )}
    </main>
  );
}
