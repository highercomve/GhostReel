import MediaImage from "./MediaImage";
import { confirmAction } from "./dialogs";
import { cleanSearchSnippet, matchedTerms } from "./searchSnippet";
import { Check, ChevronDown, Folder, FolderTree, Music, Pencil, Plus, Settings, TriangleAlert, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { navigate } from "./navigate";
import {
  addFolder,
  clock,
  enqueueFaces,
  enqueueIndex,
  enqueueSpeakers,
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
  setProjectGlossary,
  removeProject,
  excludeVideo,
  includeVideo,
  renameProject,
  search,
  setVideoCategory,
  videoFrames,
  videoTranscript,
  CATEGORIES,
  type Glossary,
  type Hit,
  type ProjectView,
  type VideoRow,
} from "./api";
import { onEvent } from "./events";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";
import VideoPanel from "./VideoPanel";
import ScriptsPanel from "./ScriptsPanel";
import PeoplePanel from "./PeoplePanel";
import OrganizeModal from "./OrganizeModal";
import { useVideoSelection } from "./useVideoSelection";
import { stageDisplay, stageTagClass } from "./stageStatus";

function SpeechCell({ v, transcribeEnabled }: { v: VideoRow; transcribeEnabled?: boolean }) {
  if (v.segments > 0) return <span className="good-text">{v.language ? v.language.toUpperCase() : <Check size={14} strokeWidth={2} aria-label="Has speech" />}</span>;
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
  const [open, setOpen] = useState(false);
  const current = CATEGORIES.find((c) => c.id === v.category) ?? {
    id: v.category,
    label: v.category,
    folder: v.category,
  };

  return (
    <div className="cat-cell" onClick={(e) => e.stopPropagation()}>
      <button type="button" className={`cat-select cat-${v.category}`} aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        title={`Category: ${current.label} (${v.category_source === "manual" ? "manual" : `${Math.round((v.category_confidence || 0) * 100)}% confidence`})`}>{current.label}</button>
      {open && <div className="library-category-menu" role="listbox" aria-label="Choose category">
        {CATEGORIES.map((cat) => <button type="button" role="option" aria-selected={cat.id === v.category} key={cat.id}
          className={cat.id === v.category ? "selected" : ""} onClick={() => { setOpen(false); if (cat.id !== v.category) onCategoryChange(cat.id); }}>{cat.label}</button>)}
      </div>}
      {v.category_tag && (
        <span
          className="cat-subtag"
          title={`Detected tag: ${v.category_tag}`}
        >
          {v.category_tag === "music" ? (
            <span className="icon-text">
              <Music size={12} strokeWidth={2} aria-hidden="true" /> music
            </span>
          ) : (
            v.category_tag
          )}
        </span>
      )}
    </div>
  );
}

const libraryFrameCache = new Map<number, { path: string | null; description: string }>();
function resultReason(h: Hit, query: string): string {
  const terms = matchedTerms(h.snippet);
  const evidenceTerms = [...new Map(terms.map((term) => [term.toLocaleLowerCase(), term])).values()];
  const visualConstraint = /\b(drone|aerial|overhead|wide shot|close[- ]?up|panoramic)\b/i.test(query);
  const hasPicture = h.kinds.some((kind) => kind === "moment" || kind === "frame");
  const kinds = [...new Set(h.kinds)];
  const names = kinds.map((kind) => kind === "moment" ? "picture" : kind === "frame" ? "on-screen text" : kind === "transcript" ? "speech" : null).filter((name): name is Exclude<typeof name, null> => name !== null);
  const evidence = evidenceTerms.length ? ` ${evidenceTerms.join(", ")}` : " this query";
  const reason = names.length === 1
    ? `${names[0][0].toUpperCase()}${names[0].slice(1)} matches${evidence}`
    : names.length > 1 ? `Matched in ${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : "Footage matches this query";
  const constraint = query.match(/\b(drone|aerial|overhead|wide shot|close[- ]?up|panoramic)\b/i)?.[0];
  return visualConstraint && !hasPicture && constraint ? `${reason}; no picture match for '${constraint}'` : reason;
}

function isSubstantiveTranscriptLine(line: string): boolean {
  const words = line.match(/[\p{L}\p{N}]+/gu) ?? [];
  if (words.length < 4) return false;
  return !/\b(?:hi|hello|hey|okay|ok|um|uh|can you hear|do you hear|test(?:ing)? audio|check(?:ing)? (?:the )?(?:camera|mic|audio)|set(?:ting)? up (?:the )?(?:camera|shot)|look at me|look into the camera|tell me your name|what(?:'s| is) your name|one two three|i don'?t see|whatever you feel comfortable|what do you do|tell me your whole name|tell me about yourself)\b/i.test(line)
    && !/^\s*(?:who|what|when|where|why|how|could you|can you|would you)\b[^.!?]*\?\s*$/i.test(line);
}

function LibraryThumbnail({ videoId, children }: { videoId: number; children: React.ReactNode }) {
  const host = useRef<HTMLSpanElement>(null);
  const [item, setItem] = useState(() => libraryFrameCache.get(videoId));
  useEffect(() => {
    if (item) return;
    const element = host.current;
    if (!element) return;
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      Promise.all([videoFrames(videoId).catch(() => []), videoTranscript(videoId).catch(() => [])]).then(([frames, transcript]) => {
        const middle = frames.length ? frames[Math.floor((frames.length - 1) / 2)] : null;
        const lines = transcript.filter((segment) => !segment.off_mic).map((segment) => segment.text.trim()).filter(Boolean);
        const substantive = lines.filter(isSubstantiveTranscriptLine);
        const identifying = substantive.find((line) => /\b(?:I['’]?m|I am|my name is)\s+[A-Z][\p{L}'’-]+/u.test(line));
        const usefulLine = identifying ?? substantive[0] ?? lines[0] ?? "";
        let sceneDescription = "";
        if (middle?.description) {
          try { sceneDescription = JSON.parse(middle.description).description ?? ""; } catch { sceneDescription = ""; }
        }
        const description = usefulLine || sceneDescription;
        const result = { path: middle?.path ?? null, description };
        libraryFrameCache.set(videoId, result);
        setItem(result);
      });
    }, { rootMargin: "160px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, [videoId, item]);
  return <span className="library-visual-cell"><span ref={host} className="library-thumb" title={item?.description}>{item?.path ? <MediaImage path={item.path} alt="" loading="lazy" /> : <span aria-hidden="true" />}</span><span className="video-name-copy">{children}{item?.description && <span className="video-description" title={item.description}>{item.description}</span>}</span></span>;
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
  const [tab, setTab] = useState<"library" | "scripts" | "glossary" | "people">("library");
  const [peopleRefresh, setPeopleRefresh] = useState(0);
  const [view, setView] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [seek, setSeek] = useState<{ t: number; nonce: number } | null>(null);
  const [query, setQuery] = useState("");
  const [searchResult, setSearchResult] = useState<{ key: string; hits: Hit[]; note: string | null } | null>(null);
  const [searchingKey, setSearchingKey] = useState<string | null>(null);
  const searchSequence = useRef(0);
  const [folderId, setFolderId] = useState<number | null>(null);
  const [glossaryFilter, setGlossaryFilter] = useState("");
  const [newTerm, setNewTerm] = useState("");
  const [newHeardAs, setNewHeardAs] = useState("");
  const [editName, setEditName] = useState<string | null>(null);
  const [samplingSize, setSamplingSize] = useState<number>(768);
  const [organizeModalOpen, setOrganizeModalOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const tasks = useQueue();
  const searchInput = useRef<HTMLInputElement>(null);
  const clearSearch = () => { setQuery(""); setSearchResult(null); setSearchingKey(null); searchInput.current?.focus(); };
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape" && query) { event.preventDefault(); clearSearch(); } };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [query]);
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
    setGlossaryFilter("");
    setNewTerm("");
    setNewHeardAs("");
    setEditName(null);
    refresh();
  }, [refresh]);

  // Keep the table fresh while this project's task runs, and once it finishes.
  const myTasks = tasks.filter(
    (t) =>
      (t.kind.type === "index" || t.kind.type === "steadiness" || t.kind.type === "faces" || t.kind.type === "speakers") &&
      t.kind.project_id === projectId,
  );
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
      setPeopleRefresh((n) => n + 1);
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
    const ok = await confirmAction(
      `Re-extract and re-describe all keyframes of this project with ${interval} interval at ${size}px sampling size? Transcripts are kept. This re-runs frame descriptions and search embeddings, which can take a while.`,
      { title: "Rebuild keyframes", destructive: true },
    );
    if (ok) run(() => redoProjectStage(projectId, "frames"));
  };

  const onDelete = async () => {
    if (!view) return;
    const ok = await confirmAction(`Delete project "${view.project.name}"? Your video files are not touched.`, {
      title: "Delete project",
      destructive: true,
    });
    if (!ok) return;
    // Second question: the index (keyframes, transcripts, previews) can be kept for reuse, since
    // re-indexing the same footage is slow.
    const purge = await confirmAction(
      "Also delete what was indexed for it — keyframes, preview renders, and the transcripts, descriptions and search index of footage no other project uses?\n\nKeep it if you may add this footage to another project later. Your video files are not touched either way.",
      { title: "Delete the indexed data too?", destructive: true, okLabel: "Delete indexed data", cancelLabel: "Keep it" },
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

  // Before the loading return: hooks must run on every render, data or not.
  useEffect(() => {
    window.dispatchEvent(new CustomEvent("ghostreel:editor-mode", { detail: tab === "scripts" }));
    return () => { window.dispatchEvent(new CustomEvent("ghostreel:editor-mode", { detail: false })); };
  }, [tab]);

  if (!view) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const { project: p, status: st } = view;
  const stage = (name: string) => st.stages.find((s) => s.stage === name);
  const selectedVideo = view.videos.find((v) => v.id === selected) ?? null;

  const glossary: Glossary = p.glossary ?? { terms: [] };

  const onAddTerm = async (e: React.FormEvent) => {
    e.preventDefault();
    const term = newTerm.trim();
    if (!term) return;
    const variants = newHeardAs
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const existingIndex = glossary.terms.findIndex(
      (t) => t.term.toLowerCase() === term.toLowerCase()
    );
    let nextTerms = [...glossary.terms];
    if (existingIndex >= 0) {
      const existing = nextTerms[existingIndex];
      const mergedHeardAs = [...existing.heard_as];
      for (const v of variants) {
        if (
          v.toLowerCase() !== term.toLowerCase() &&
          !mergedHeardAs.some((h) => h.toLowerCase() === v.toLowerCase())
        ) {
          mergedHeardAs.push(v);
        }
      }
      nextTerms[existingIndex] = { ...existing, heard_as: mergedHeardAs };
    } else {
      const cleanVariants = variants.filter(
        (v) => v.toLowerCase() !== term.toLowerCase()
      );
      nextTerms.push({ term, heard_as: cleanVariants });
    }
    await run(() => setProjectGlossary(projectId, { terms: nextTerms }));
    setNewTerm("");
    setNewHeardAs("");
  };

  const onRemoveTerm = async (termToRemove: string) => {
    const nextTerms = glossary.terms.filter(
      (t) => t.term.toLowerCase() !== termToRemove.toLowerCase()
    );
    await run(() => setProjectGlossary(projectId, { terms: nextTerms }));
  };

  const onRemoveHeardAs = async (term: string, variantToRemove: string) => {
    const nextTerms = glossary.terms.map((t) => {
      if (t.term.toLowerCase() === term.toLowerCase()) {
        return {
          ...t,
          heard_as: t.heard_as.filter(
            (h) => h.toLowerCase() !== variantToRemove.toLowerCase()
          ),
        };
      }
      return t;
    });
    await run(() => setProjectGlossary(projectId, { terms: nextTerms }));
  };

  // Plain computation, not useMemo: this runs after the `if (!view) return` above, and a hook
  // there is called on some renders and not others (React error #310, a blank window).
  const glossaryQuery = glossaryFilter.trim().toLowerCase();
  const filteredGlossaryTerms = !glossaryQuery
    ? glossary.terms
    : glossary.terms.filter(
        (t) =>
          t.term.toLowerCase().includes(glossaryQuery) ||
          t.heard_as.some((h) => h.toLowerCase().includes(glossaryQuery)),
      );

  const onApplyGlossary = async () => {
    if (
      !(await confirmAction(
        "Re-transcribe every video in this project with the glossary? This runs speech-to-text again.",
        { okLabel: "Re-transcribe" },
      ))
    ) {
      return;
    }
    await run(() => redoProjectStage(projectId, "transcribe"));
  };

  const editorMode = tab === "scripts";
  const statsLine = `${p.width}×${p.height} · ${fpsLabel(p.fps_num, p.fps_den)} fps · ${st.videos} videos · ${humanDuration(st.total_duration_s)} · ${humanSize(st.total_size)}`;
  // In the editor the project header, the tabs and the actions share one slim row, so the cut
  // gets the height; on the Library tab they keep their roomier two-row layout.
  const tabs = (
    <div className="tab-nav" role="tablist" aria-label="Project views">
      <button
        type="button"
        role="tab"
        aria-selected={tab === "library"}
        className={`tab-btn ${tab === "library" ? "active" : ""}`}
        onClick={() => setTab("library")}
      >
        Library
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "scripts"}
        className={`tab-btn ${tab === "scripts" ? "active" : ""}`}
        onClick={() => setTab("scripts")}
      >
        Scripts
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "glossary"}
        className={`tab-btn ${tab === "glossary" ? "active" : ""}`}
        onClick={() => setTab("glossary")}
      >
        Glossary{glossary.terms.length > 0 && (
          <span className="muted small" style={{ marginLeft: "0.35rem" }}>
            {glossary.terms.length}
          </span>
        )}
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "people"}
        className={`tab-btn ${tab === "people" ? "active" : ""}`}
        onClick={() => setTab("people")}
      >
        People{(view?.people_count ?? 0) > 0 && (
          <span className="muted small" style={{ marginLeft: "0.35rem" }}>
            {view?.people_count}
          </span>
        )}
      </button>
    </div>
  );

  return (
    <main className={`project-workspace ${editorMode ? "editor-mode" : ""}`}>
      <header className={editorMode ? "project-head-compact" : undefined}>
        <div className="project-title-block" title={editorMode ? statsLine : undefined}>
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
              {p.name}{" "}
              <span className="edit-hint">
                <Pencil size={14} strokeWidth={2} aria-hidden="true" />
              </span>
            </h1>
          )}
          <p className="muted project-stats">
            {editorMode ? `${st.videos} videos · ${humanDuration(st.total_duration_s)}` : statsLine}
          </p>
        </div>
        {editorMode && tabs}
        <div className="header-actions">
          <div className="project-menu-container" ref={menuRef}>
            <button
              type="button"
              className={editorMode ? "ghost small" : "ghost"}
              onClick={() => setMenuOpen(!menuOpen)}
              aria-expanded={menuOpen}
              title="Project settings, video analysis & management"
            >
              <span className="icon-text">
                <Settings size={14} strokeWidth={2} aria-hidden="true" /> Project
                <ChevronDown size={14} strokeWidth={2} aria-hidden="true" />
              </span>
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

                {(() => {
                  const totalVideos = view.videos.length;

                  // Every stage's status comes from the backend's one stage gate (off for a stage
                  // the project turned off); the queue only adds whether it is running right now.
                  const shown = (name: string) => stageDisplay(view.stages?.find((x) => x.stage === name), running);
                  const framesCount = stage("frames");
                  const transcribeCount = stage("transcribe");
                  const describeCount = stage("describe");
                  const embedCount = stage("embed");
                  const [framesStatus, transcribeStatus, describeStatus, embedStatus] = [
                    shown("frames"),
                    shown("transcribe"),
                    shown("describe"),
                    shown("embed"),
                  ];
                  const measuredSteadiness = view.videos.filter((v) => v.steadiness_measured).length;
                  const steadinessStatus = shown("steadiness");
                  const faceModelsMissing = view.face_models_available === false;
                  const measuredFaces = view.videos.filter((v) => v.faces_measured).length;
                  const facesStatus = shown("faces");
                  const speakersPending = view.speakers_pending ?? 0;
                  const speakersEverRan = (view.people_count ?? 0) > 0 || (totalVideos > 0 && speakersPending === 0);
                  const speakersStatus = shown("speakers");
                  const tag = (d: { state: string; reason: string | null }) => (
                    <span className={`tag ${stageTagClass(d.state)} small`} title={d.reason ?? undefined}>
                      {d.state}
                    </span>
                  );

                  return (
                    <>
                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Keyframes</span>
                          {tag(framesStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={() => {
                            setMenuOpen(false);
                            onRebuildFrames();
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Re-extract and re-describe keyframes with current interval and sampling size"
                        >
                          {(framesCount?.done ?? 0) > 0 ? "Re-run…" : "Run"}
                        </button>
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Transcription</span>
                          {tag(transcribeStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={async () => {
                            setMenuOpen(false);
                            const isRerun = (transcribeCount?.done ?? 0) > 0;
                            if (isRerun) {
                              const ok = await confirmAction(
                                "Re-transcribe all videos in this project? This will redo transcription and subsequent stages.",
                                { title: "Re-run transcription", destructive: true },
                              );
                              if (!ok) return;
                            }
                            run(() => redoProjectStage(projectId, "transcribe"));
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Transcribe speech with Whisper"
                        >
                          {(transcribeCount?.done ?? 0) > 0 ? "Re-run…" : "Run"}
                        </button>
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Frame descriptions</span>
                          {tag(describeStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={async () => {
                            setMenuOpen(false);
                            const isRerun = (describeCount?.done ?? 0) > 0;
                            if (isRerun) {
                              const ok = await confirmAction(
                                "Re-describe keyframes for all videos in this project? This will redo frame descriptions and search embeddings.",
                                { title: "Re-run frame descriptions", destructive: true },
                              );
                              if (!ok) return;
                            }
                            run(() => redoProjectStage(projectId, "describe"));
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Describe keyframes with vision model"
                        >
                          {(describeCount?.done ?? 0) > 0 ? "Re-run…" : "Run"}
                        </button>
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Search embeddings</span>
                          {tag(embedStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={async () => {
                            setMenuOpen(false);
                            const isRerun = (embedCount?.done ?? 0) > 0;
                            if (isRerun) {
                              const ok = await confirmAction(
                                "Re-generate search embeddings for all videos in this project?",
                                { title: "Re-run search embeddings", destructive: true },
                              );
                              if (!ok) return;
                            }
                            run(() => redoProjectStage(projectId, "embed"));
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Generate vector embeddings and search chunks"
                        >
                          {(embedCount?.done ?? 0) > 0 ? "Re-run…" : "Run"}
                        </button>
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Camera shake</span>
                          {tag(steadinessStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={async () => {
                            setMenuOpen(false);
                            const isRerun = totalVideos > 0 && measuredSteadiness === totalVideos;
                            if (isRerun) {
                              const ok = await confirmAction("Re-measure camera steadiness for all videos in this project?", {
                                title: "Re-analyze camera shake",
                              });
                              if (ok) run(() => enqueueSteadiness(projectId, true));
                            } else {
                              run(() => enqueueSteadiness(projectId, false));
                            }
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Analyze camera steadiness and shake for videos in this project"
                        >
                          {totalVideos > 0 && measuredSteadiness === totalVideos ? "Re-run…" : "Run"}
                        </button>
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Faces</span>
                          {tag(facesStatus)}
                        </div>
                        {faceModelsMissing ? (
                          <button
                            type="button"
                            className="project-menu-stage-link"
                            onClick={() => {
                              setMenuOpen(false);
                              navigate("models");
                            }}
                            title="Download YuNet and SFace models on the Models page"
                          >
                            Download face models on the Models page
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="small"
                            onClick={async () => {
                              setMenuOpen(false);
                              const isRerun = totalVideos > 0 && measuredFaces === totalVideos;
                              if (isRerun) {
                                const ok = await confirmAction("Re-analyze faces for all videos in this project?", {
                                  title: "Re-analyze faces",
                                });
                                if (ok) run(() => enqueueFaces(projectId, true));
                              } else {
                                run(() => enqueueFaces(projectId, false));
                              }
                            }}
                            disabled={!!running || !!queued || totalVideos === 0}
                            title="Find and track faces through video footage"
                          >
                            {totalVideos > 0 && measuredFaces === totalVideos ? "Re-run…" : "Run"}
                          </button>
                        )}
                      </div>

                      <div className="project-menu-stage-row">
                        <div className="project-menu-stage-info">
                          <span className="project-menu-stage-name">Speakers</span>
                          {tag(speakersStatus)}
                        </div>
                        <button
                          type="button"
                          className="small"
                          onClick={async () => {
                            setMenuOpen(false);
                            if (speakersEverRan && speakersPending === 0) {
                              const ok = await confirmAction(
                                "Re-analyze speakers and people for all videos in this project? Names you gave people are kept.",
                                { title: "Re-analyze speakers" },
                              );
                              if (ok) run(() => enqueueSpeakers(projectId, true));
                            } else {
                              run(() => enqueueSpeakers(projectId, false));
                            }
                          }}
                          disabled={!!running || !!queued || totalVideos === 0}
                          title="Find who speaks when, group faces into people and link each voice to a face"
                        >
                          {speakersEverRan && speakersPending === 0 ? "Re-run…" : "Run"}
                        </button>
                      </div>
                    </>
                  );
                })()}

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

                <label className="project-menu-checkbox" title="Measure camera motion and shake">
                  <input
                    type="checkbox"
                    checked={p.pipeline.steadiness ?? true}
                    disabled={!!running || !!queued}
                    onChange={async (e) => {
                      const next = { ...p.pipeline, steadiness: e.target.checked };
                      await run(() => setProjectPipeline(projectId, next));
                    }}
                  />
                  <span>Camera shake</span>
                </label>

                <div>
                  <label className="project-menu-checkbox" title="Find and track faces through video footage">
                    <input
                      type="checkbox"
                      checked={p.pipeline.faces ?? true}
                      disabled={!!running || !!queued}
                      onChange={async (e) => {
                        const next = { ...p.pipeline, faces: e.target.checked };
                        await run(() => setProjectPipeline(projectId, next));
                      }}
                    />
                    <span>Faces</span>
                  </label>
                  <div className="project-menu-hint">
                    Finds and follows faces — needed for speaker framing. ~5 min per hour of video.
                  </div>
                </div>

                <div>
                  <label className="project-menu-checkbox" title="Find who speaks when and which face each voice is">
                    <input
                      type="checkbox"
                      checked={p.pipeline.speakers ?? true}
                      disabled={!!running || !!queued}
                      onChange={async (e) => {
                        const next = { ...p.pipeline, speakers: e.target.checked };
                        await run(() => setProjectPipeline(projectId, next));
                      }}
                    />
                    <span>Speakers</span>
                  </label>
                  <div className="project-menu-hint">
                    Tells voices apart, groups faces into people and links them. ~1 min per hour of video.
                  </div>
                </div>

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

          <button
            className={editorMode ? "small" : "primary"}
            onClick={() => run(() => enqueueIndex(projectId))}
            disabled={!!queued || view.folders.length === 0}
          >
            {running ? "Index again" : queued ? "Queued…" : "Index now"}
          </button>
        </div>
      </header>

      {!editorMode && tabs}

      {editorMode ? (
                <ScriptsPanel projectId={projectId} selectedVideoIds={selectedIds} videoCount={view?.videos.length ?? 0} />
      ) : tab === "people" ? (
        <>
          {running && <TaskCard task={running} compact />}
          {!running && queued && <TaskCard task={queued} compact />}
          <PeoplePanel projectId={projectId} refreshKey={peopleRefresh} />
        </>
      ) : tab === "glossary" ? (
        <>
          {error && <div className="banner bad">{error}</div>}
          {running && <TaskCard task={running} compact />}
          {!running && queued && <TaskCard task={queued} compact />}
          {!running && !queued && lastFinished && lastFinished.state !== "done" && <TaskCard task={lastFinished} compact />}

          <p className="muted" style={{ margin: "0 0 1rem" }}>
            {'Terms Whisper should spell correctly. "Heard as" lists known mis-hearings that are fixed automatically when a video is transcribed.'}
          </p>

          {glossary.terms.length > 8 && (
            <div style={{ marginBottom: "1rem" }}>
              <input
                type="text"
                placeholder="Filter terms"
                value={glossaryFilter}
                onChange={(e) => setGlossaryFilter(e.target.value)}
                style={{ maxWidth: "320px" }}
              />
            </div>
          )}

          <section className="card list" style={{ gap: "0.75rem" }}>
            <form
              onSubmit={onAddTerm}
              style={{
                display: "flex",
                gap: "0.5rem",
                alignItems: "center",
                flexWrap: "wrap",
                paddingBottom: glossary.terms.length > 0 ? "0.75rem" : undefined,
                borderBottom: glossary.terms.length > 0 ? "1px solid var(--border)" : undefined,
              }}
            >
              <input
                type="text"
                placeholder="Term (e.g. GhostReel)"
                value={newTerm}
                onChange={(e) => setNewTerm(e.target.value)}
                style={{ flex: "1 1 180px" }}
              />
              <input
                type="text"
                placeholder="Heard as (comma-separated, e.g. ghost real, ghost rail)"
                value={newHeardAs}
                onChange={(e) => setNewHeardAs(e.target.value)}
                style={{ flex: "2 1 260px" }}
              />
              <button type="submit" className="small" disabled={!newTerm.trim()}>
                Add term
              </button>
            </form>

            {glossary.terms.length === 0 && (
              <div className="muted">
                No glossary terms yet — add technical terms or names Whisper tends to mishear.
              </div>
            )}
            {glossary.terms.length > 0 && filteredGlossaryTerms.length === 0 && (
              <div className="muted">No terms match &ldquo;{glossaryFilter}&rdquo;.</div>
            )}
            {filteredGlossaryTerms.map((t) => (
              <div
                key={t.term}
                className="row"
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "1rem",
                  flexWrap: "wrap",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
                  <span style={{ fontWeight: 600 }}>{t.term}</span>
                  {t.heard_as.length > 0 && (
                    <div style={{ display: "flex", gap: "0.35rem", alignItems: "center", flexWrap: "wrap" }}>
                      <span className="muted small">heard as:</span>
                      {t.heard_as.map((h) => (
                        <span
                          key={h}
                          className="chip"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.25rem",
                            padding: "2px 8px",
                            fontSize: "12px",
                          }}
                        >
                          <span>{h}</span>
                          <button
                            type="button"
                            className="ghost small"
                            style={{
                              padding: "0",
                              lineHeight: 1,
                              border: "none",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                            }}
                            onClick={() => onRemoveHeardAs(t.term, h)}
                            title={`Remove variant "${h}"`}
                            aria-label={`Remove variant "${h}"`}
                          >
                            <X size={12} strokeWidth={2} />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  className="ghost small"
                  onClick={() => onRemoveTerm(t.term)}
                  title={`Remove term "${t.term}"`}
                >
                  Remove
                </button>
              </div>
            ))}
          </section>

          <div style={{ marginTop: "1rem" }}>
            <button
              type="button"
              onClick={onApplyGlossary}
              disabled={!!running || !!queued || view.videos.length === 0}
              title="Re-transcribe every video in this project with the glossary"
            >
              Apply to transcribed videos
            </button>
          </div>
        </>
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
        {query && <button type="button" className="ghost search-clear" onClick={clearSearch} aria-label="Clear search" title="Clear search (Esc)"><X size={15} strokeWidth={2} /></button>}
        <button type="submit" className="primary" disabled={searching || !query.trim()}>
          {searching ? "Searching…" : "Search"}
        </button>
      </form>
      {hits && <div className="library-search-summary" role="status"><strong>Found {hits.length} {hits.length === 1 ? "moment" : "moments"}</strong>{searchNote && <span className="muted small">{searchNote}</span>}<span className="muted small">Esc to clear</span></div>}
      {hits && (
        <section className="hits">
          {hits.length === 0 && <div className="muted">No matches.</div>}
          {hits.map((h, i) => (
            <div key={i} className="card hit">
              {h.frame ? <MediaImage path={h.frame} alt="" loading="lazy" /> : <div className="noframe" />}
              <div className="hit-body">
                <div className="hit-head">
                  <span className="label">{fileName(h.path)}</span>
                  <span className="time">
                    {clock(h.start_s)}–{clock(h.end_s)}
                  </span>
                </div>
                <div className="small">
                  <Snippet text={cleanSearchSnippet(h.snippet)} />
                </div>
                <div>
                  {h.kinds.map((k) => (
                    <span key={k} className="tag">
                      {k === "moment" ? "picture" : k === "frame" ? "on screen" : "speech"}
                    </span>
                  ))}
                </div>
                <div className="library-hit-reason">{resultReason(h, query)}</div>
                <div className="hit-actions"><button type="button" className="small" onClick={() => openAt(h.video_id, h.start_s)}>Preview moment</button><button type="button" className="ghost small" onClick={() => openAt(h.video_id, 0)}>Open video</button></div>
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
              const ok = await confirmAction(
                role === "reference"
                  ? `Use "${name}" as a reference edit? The script chat will study it (length, pacing, structure) but never use it as footage. It disappears from search.`
                  : `Remove "${name}" from this project's library? It won't be searched, indexed or used in scripts. The file is not deleted.`,
                { title: role === "reference" ? "Reference edit" : "Remove from library", destructive: true },
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
            <span className={`icon-text ${f.available ? "muted" : "bad-text"}`}>
              {f.available ? (
                <Folder size={15} strokeWidth={2} aria-label="Folder" />
              ) : (
                <TriangleAlert size={15} strokeWidth={2} aria-label="Folder not available" />
              )}
            </span>
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
          <button className="ghost icon-text" onClick={onAddFolder}>
            <Plus size={15} strokeWidth={2} aria-hidden="true" /> Add folder
          </button>
          <button
            className="ghost"
            onClick={() => setOrganizeModalOpen(true)}
            disabled={view.videos.length === 0 || !!running}
            title="Organize original video files into categorized subfolders on disk"
          >
            <span className="icon-text">
              <FolderTree size={15} strokeWidth={2} aria-hidden="true" /> Organize folders…
            </span>
          </button>
        </div>
      </section>

      <h2>
        Videos
        <span className="muted small normal">
          {" "}
          · {p.pipeline?.transcribe ? `${stage("transcribe")?.done ?? 0} of ${st.videos} transcribed${(st.videos - (stage("transcribe")?.done ?? 0)) > 0 ? ` · ${st.videos - (stage("transcribe")?.done ?? 0)} without speech` : ""}` : "transcription disabled"} ·{" "}
          {p.pipeline?.describe ? `${stage("describe")?.done ?? 0} described` : "descriptions disabled"} ·{" "}
          {p.pipeline?.embed ? `${stage("embed")?.done ?? 0} searchable (speech and frame descriptions)` : "search indexing disabled"}
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
                {cat.label} ({count})
              </button>
            );
          })}
        </div>
      )}

      <div className="video-selection-toolbar">
        <button type="button" className={`ghost small${selection.enabled ? " active" : ""}`}
          aria-pressed={selection.enabled} onClick={() => selection.setEnabled(!selection.enabled)}>Select videos</button>
        <span className="small" role="status">{selectedIds.length ? `${selectedIds.length} selected` : ""}
          {hiddenSelected > 0 ? ` · ${hiddenSelected} hidden by category` : ""}</span>
        {selectedIds.length > 0 && <>
          <button type="button" className="ghost small" onClick={() => {
            searchInput.current?.focus(); searchInput.current?.scrollIntoView({ behavior: "smooth", block: "center" });
          }}>Search selected</button>
          <button type="button" className="ghost small" onClick={selection.clear}>Clear</button>
        </>}
        <span className="muted small selection-hint">{selection.enabled
          ? "Drag across rows · Ctrl/⌘ adds · Shift-click selects a range · Esc cancels a drag"
          : "Ctrl/⌘-click · Shift-click selects a range"}</span>
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
                <th className="video-technical-heading">Size</th>
                <th className="video-technical-heading">Format</th>
                <th>Speech</th>
                <th className="video-technical-heading">Camera</th>
                <th className="video-technical-heading">Frames</th>
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
                    <LibraryThumbnail videoId={v.id}>
                    {fileName(v.path)}
                    {v.copies > 1 && <span className="tag">×{v.copies}</span>}
                    {v.vfr && (
                      <span className="tag warn" title="Variable frame rate (phone footage) — may drift in Premiere">
                        VFR
                      </span>
                    )}
                    {v.has_audio === false && <span className="tag">no audio</span>}
                    </LibraryThumbnail>
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
