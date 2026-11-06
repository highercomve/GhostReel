import MediaImage from "./MediaImage";
import { cleanSearchSnippet, matchedTerms } from "./searchSnippet";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  clock,
  fileName,
  focusMainWindow,
  Hit,
  listProjects,
  openExternal,
  ProjectSummary,
  search,
  videoDetails,
  VideoRow,
} from "./api";
import { Clapperboard, ExternalLink, Folder, Play, Search, X } from "lucide-react";
import { emitEvent } from "./events";
import VideoPanel from "./VideoPanel";

function Snippet({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <span>
      {parts.map((p, i) =>
        p.startsWith("[") && p.endsWith("]") ? (
          <mark key={i}>{p.slice(1, -1)}</mark>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </span>
  );
}

type KindFilter = "all" | "moment" | "transcript" | "frame";

function explainSearchMatch(h: Hit, query: string): string {
  const matchedWords = matchedTerms(h.snippet);
  const terms = matchedWords.join(", ");
  const kinds = [...new Set(h.kinds)];
  const sources = kinds.map((kind) => kind === "moment" ? "picture" : kind === "frame" ? "on-screen text" : kind === "transcript" ? "speech" : null).filter((source): source is Exclude<typeof source, null> => source !== null);
  const visualConstraint = /\b(drone|aerial|overhead|wide shot|close[- ]?up|panoramic)\b/i.test(query);
  const hasPicture = h.kinds.some((kind) => kind === "moment" || kind === "frame");
  const evidence = terms ? ` ${terms}` : " this query";
  const reason = sources.length === 1
    ? `${sources[0][0].toUpperCase()}${sources[0].slice(1)} matches${evidence}`
    : sources.length > 1 ? `Matched in ${sources.slice(0, -1).join(", ")} and ${sources[sources.length - 1]}` : "Footage matches this query";
  const constraint = query.match(/\b(drone|aerial|overhead|wide shot|close[- ]?up|panoramic)\b/i)?.[0];
  return visualConstraint && !hasPicture && constraint ? `${reason}; no picture match for '${constraint}'` : reason;
}

export default function SearchWindow() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kindFilter, setKindFilter] = useState<KindFilter>("all");

  const [previewVideo, setPreviewVideo] = useState<VideoRow | null>(null);
  const [previewSeek, setPreviewSeek] = useState<{ t: number; nonce: number } | null>(null);
  const [loadingPreviewId, setLoadingPreviewId] = useState<number | null>(null);

  const searchRequest = useRef(0);
  const previewRequest = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((err) => console.error("Failed to load projects:", err));
  }, []);

  const runSearch = useCallback(
    async (q: string, projId: number | null) => {
      const request = ++searchRequest.current;
      const trimmed = q.trim();
      if (!trimmed) {
        setHits(null);
        setSearchNote(null);
        setSearching(false);
        return;
      }
      setSearching(true);
      setError(null);
      try {
        const res = await search(projId, trimmed, 60);
        if (request !== searchRequest.current) return;
        setHits(res.hits);
        setSearchNote(res.note);
      } catch (e) {
        if (request === searchRequest.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (request === searchRequest.current) setSearching(false);
      }
    },
    [],
  );

  const onSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void runSearch(query, selectedProjectId);
  };

  const onProjectChange = (id: number | null) => {
    setSelectedProjectId(id);
    if (query.trim()) {
      void runSearch(query, id);
    }
  };

  const handleOpenPreview = async (hit: Hit) => {
    if (loadingPreviewId === hit.video_id) return;
    const request = ++previewRequest.current;
    setLoadingPreviewId(hit.video_id);
    try {
      const details = await videoDetails(hit.video_id);
      if (request !== previewRequest.current) return;
      setPreviewVideo(details);
      setPreviewSeek({ t: hit.start_s, nonce: Date.now() });
      setTimeout(() => {
        previewRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 50);
    } catch (e) {
      if (request === previewRequest.current) setError(`Failed to load video details: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      if (request === previewRequest.current) setLoadingPreviewId(null);
    }
  };

  const handleOpenExternal = async (hit: Hit, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await openExternal(hit.path, hit.start_s);
    } catch (err) {
      console.error("Failed to open external:", err);
    }
  };

  const handleOpenInProject = async (projectId: number, hit: Hit, e: React.MouseEvent) => {
    e.stopPropagation();
    await emitEvent("navigate-to-project", {
      projectId,
      videoId: hit.video_id,
      t: hit.start_s,
    });
    await focusMainWindow();
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      } else if (e.key === "Escape") {
        if (previewVideo || loadingPreviewId !== null) {
          ++previewRequest.current;
          setLoadingPreviewId(null);
          setPreviewVideo(null);
        } else if (query) {
          ++searchRequest.current;
          setSearching(false);
          setQuery("");
          setHits(null);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [previewVideo, query, loadingPreviewId]);

  // Filter hits by kind if requested
  const filteredHits = useMemo(() => {
    if (!hits) return null;
    if (kindFilter === "all") return hits;
    return hits.filter((h) => h.kinds.includes(kindFilter));
  }, [hits, kindFilter]);

  const uniqueProjectsInHits = useMemo(() => {
    if (!hits) return 0;
    const projectIds = new Set<number>();
    for (const h of hits) {
      if (h.projects) {
        for (const p of h.projects) projectIds.add(p.id);
      }
    }
    return projectIds.size;
  }, [hits]);

  return (
    <div className="search-window-shell">
      <header className="search-window-header">
        <div className="header-title-row">
          <div className="title-with-badge">
            <span className="search-icon"><Search size={18} strokeWidth={2} aria-hidden /></span>
            <h1>Search Material</h1>
          </div>
          <span className="global-badge">Global</span>
          <div className="search-shortcut-hint">
            Press <kbd>/</kbd> to focus · <kbd>Esc</kbd> to clear
          </div>
        </div>
        <p className="search-subhead">
          Search footage, dialogue, and on-screen text across all your projects
        </p>

        <form className="search-window-form" onSubmit={onSearchSubmit}>
          <div className="search-input-wrapper">
            <span className="input-search-icon"><Search size={16} strokeWidth={2} aria-hidden /></span>
            <input
              ref={inputRef}
              autoFocus
              type="text"
              placeholder="Search footage, e.g. “unboxing”"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                if (!e.target.value.trim()) {
                  ++searchRequest.current;
                  setSearching(false);
                  setHits(null);
                  setSearchNote(null);
                }
              }}
            />
            {query && (
              <button
                type="button"
                className="clear-search-btn"
                title="Clear search (Esc)"
                onClick={() => {
                  ++searchRequest.current;
                  setSearching(false);
                  setQuery("");
                  setHits(null);
                  inputRef.current?.focus();
                }}
              >
                <X size={16} strokeWidth={2} aria-hidden />
              </button>
            )}
          </div>
          <button type="submit" className="search-submit-btn primary" disabled={searching || !query.trim()}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="search-filters-bar">
          <div className="filter-group">
            <label htmlFor="project-filter">Project:</label>
            <select
              id="project-filter"
              value={selectedProjectId ?? ""}
              onChange={(e) => onProjectChange(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">All projects ({projects.length})</option>
              {projects.map(({ project }) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <span className="filter-label">Kind:</span>
            <div className="kind-chips">
              <button
                type="button"
                className={`chip ${kindFilter === "all" ? "active" : ""}`}
                onClick={() => setKindFilter("all")}
              >
                All
              </button>
              <button
                type="button"
                className={`chip ${kindFilter === "moment" ? "active" : ""}`}
                onClick={() => setKindFilter("moment")}
              >
                Picture
              </button>
              <button
                type="button"
                className={`chip ${kindFilter === "transcript" ? "active" : ""}`}
                onClick={() => setKindFilter("transcript")}
              >
                Speech
              </button>
              <button
                type="button"
                className={`chip ${kindFilter === "frame" ? "active" : ""}`}
                onClick={() => setKindFilter("frame")}
              >
                On screen
              </button>
            </div>
          </div>
        </div>
      </header>

      {error && <div className="banner bad">{error}</div>}
      {searchNote && <div className="search-note muted small">{searchNote}</div>}

      <div ref={previewRef} className="search-preview-anchor">
        {previewVideo && (
          <div className="search-preview-container card">
            <div className="search-preview-header">
              <div className="preview-title">
                <strong>Preview:</strong> {fileName(previewVideo.path)}
              </div>
              <button
                type="button"
                className="ghost small close-preview-btn"
                onClick={() => { ++previewRequest.current; setLoadingPreviewId(null); setPreviewVideo(null); }}
                title="Close preview (Esc)"
              >
                <X size={14} strokeWidth={2} aria-hidden /> Close preview
              </button>
            </div>
            <VideoPanel
              video={previewVideo}
              seek={previewSeek}
              onClose={() => { ++previewRequest.current; setLoadingPreviewId(null); setPreviewVideo(null); }}
            />
          </div>
        )}
      </div>

      <main className="search-window-results">
        {hits === null && !searching && (
          <div className="search-empty-state">
            <div className="empty-icon"><Clapperboard size={34} strokeWidth={1.75} aria-hidden /></div>
            <h3>Discover material across your library</h3>
            <p className="muted">
              GhostReel indexes scenes, spoken words, and text inside videos across every project.
            </p>
            <div className="search-suggestions">
              <span className="muted small">Quick searches:</span>
              {["interview", "outdoor", "screen recording", "music", "presentation"].map((term) => (
                <button
                  key={term}
                  type="button"
                  className="ghost small chip-btn"
                  onClick={() => {
                    setQuery(term);
                    void runSearch(term, selectedProjectId);
                  }}
                >
                  “{term}”
                </button>
              ))}
            </div>
          </div>
        )}

        {searching && (
          <div className="search-loading-state">
            <div className="spinner" />
            <p className="muted">Searching across projects…</p>
          </div>
        )}

        {filteredHits !== null && !searching && (
          <>
            <div className="search-results-summary">
              <span className="results-count">
                {filteredHits.length === 0
                  ? "No matching moments found"
                  : `Found ${filteredHits.length} moment${filteredHits.length === 1 ? "" : "s"}${
                      uniqueProjectsInHits > 0
                        ? ` across ${uniqueProjectsInHits} project${uniqueProjectsInHits === 1 ? "" : "s"}`
                        : ""
                    }`}
              </span>
            </div>

            {filteredHits.length === 0 ? (
              <div className="search-no-results muted">
                No footage matched your query. Try broadening your terms or switching project filters.
              </div>
            ) : (
              <div className="search-hits-grid">
                {filteredHits.map((h, i) => {
                  const defaultProj = h.projects && h.projects.length > 0 ? h.projects[0] : null;
                  const isCurrentPreview = previewVideo?.id === h.video_id;
                  const matchExplanation = explainSearchMatch(h, query);

                  return (
                    <div
                      key={i}
                      className={`card search-hit-card ${isCurrentPreview ? "active-preview" : ""}`}
                      onClick={() => handleOpenPreview(h)}
                      title={isCurrentPreview ? "Currently previewing this moment" : "Click card to preview this moment here"}
                    >
                      <div className="hit-thumbnail-container">
                        {h.frame ? (
                          <MediaImage path={h.frame} alt="" loading="lazy" />
                        ) : (
                          <div className="noframe media-image-placeholder" role="img" aria-label="No thumbnail available">No thumbnail available</div>
                        )}
                        {isCurrentPreview ? (
                          <span className="active-preview-badge">
                            <Play size={12} strokeWidth={2.5} aria-hidden /> Previewing
                          </span>
                        ) : (
                          <span className="thumbnail-play-overlay" title="Click to preview">
                            <Play size={16} strokeWidth={2} aria-hidden />
                          </span>
                        )}
                        <span className="thumbnail-timecode">
                          {clock(h.start_s)}–{clock(h.end_s)}
                        </span>
                      </div>

                      <div className="hit-details">
                        <div className="hit-title-row">
                          <span className="hit-filename" title={h.path}>
                            {fileName(h.path)}
                          </span>
                          <div className="hit-project-badges">
                            {h.projects?.map((p) => (
                              <span
                                key={p.id}
                                className="project-badge"
                                title={`Belongs to project: ${p.name}. Click to view in project.`}
                                onClick={(e) => handleOpenInProject(p.id, h, e)}
                              >
                                <Folder size={12} strokeWidth={2} aria-hidden /> {p.name}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="hit-snippet">
                          <Snippet text={cleanSearchSnippet(h.snippet)} />
                        </div>
                        <div className="search-match-explanation" aria-label={`Why this result matched: ${matchExplanation}`}>{matchExplanation}</div>

                        <div className="hit-footer">
                          <div className="hit-tags">
                            {h.kinds.map((k) => {
                              const label = k === "moment" ? "picture" : k === "frame" ? "on screen" : "speech";
                              const tooltip =
                                k === "moment"
                                  ? "Visual scene description match"
                                  : k === "frame"
                                    ? "On-screen text (OCR) match"
                                    : "Spoken transcript match";
                              return (
                                <span key={k} className="tag kind-tag" title={tooltip}>
                                  {label}
                                </span>
                              );
                            })}
                          </div>

                          <div className="hit-actions">
                            <button
                              type="button"
                              className={`small hit-action-btn ${isCurrentPreview ? "active-preview-btn" : ""}`}
                              title={isCurrentPreview ? "Currently previewing this moment" : "Preview clip in the embedded player"}
                              disabled={loadingPreviewId === h.video_id}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleOpenPreview(h);
                              }}
                            >
                              {loadingPreviewId === h.video_id ? (
                                "Loading…"
                              ) : isCurrentPreview ? (
                                <><Play size={14} strokeWidth={2} aria-hidden /> Previewing</>
                              ) : (
                                <><Play size={14} strokeWidth={2} aria-hidden /> Preview</>
                              )}
                            </button>
                            <button
                              type="button"
                              className="ghost small hit-action-btn"
                              title="Open in external video player"
                              onClick={(e) => handleOpenExternal(h, e)}
                            >
                              <ExternalLink size={14} strokeWidth={2} aria-hidden /> Open externally
                            </button>
                            {defaultProj && (
                              <button
                                type="button"
                                className="ghost small hit-action-btn"
                                title={`Open project "${defaultProj.name}" at this moment`}
                                onClick={(e) => handleOpenInProject(defaultProj.id, h, e)}
                              >
                                Open in project
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
