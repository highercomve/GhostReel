import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, Search, X } from "lucide-react";
import MediaImage from "../MediaImage";
import { clock, fileName, search, videoFrames, type Hit, type VideoRow } from "../api";

/** A stretch of footage the bin hands to the editor. */
export interface Pick {
  video_id: number;
  in_s: number;
  out_s: number;
}

/** The drag payload type a drop target (e.g. the timeline) can accept from the bin. */
export const PICK_MIME = "application/x-ghostreel-pick";

interface MediaBinProps {
  projectId: number;
  videos: Record<number, VideoRow>;
  /** What the actions apply to, shown on the buttons so the target is never a guess. */
  beatLabel: string | null;
  canReplace: boolean;
  disabled?: boolean;
  onAdd: (pick: Pick) => void;
  onReplace: (pick: Pick) => void;
}

const CATEGORY_LABEL: Record<string, string> = {
  all: "All",
  interview: "Interview",
  broll: "B-roll",
  drone: "Drone",
  screencast: "Screen",
  audio: "Audio",
};

const KIND_LABEL: Record<string, string> = { moment: "picture", frame: "on screen", speech: "speech" };

// One poster frame per video for the whole session: the bin remounts on every tab switch.
const posterCache = new Map<number, Promise<string | null>>();
function posterFor(videoId: number): Promise<string | null> {
  let p = posterCache.get(videoId);
  if (!p) {
    p = videoFrames(videoId)
      .then((frames) => (frames.length ? frames[Math.floor(frames.length / 2)].path : null))
      .catch(() => null);
    posterCache.set(videoId, p);
  }
  return p;
}

/** A video's middle keyframe, fetched only once its row scrolls into view. */
function Poster({ videoId, label }: { videoId: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [path, setPath] = useState<string | null | undefined>(undefined);

  useEffect(() => {
    const el = ref.current;
    if (!el || visible) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "120px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    posterFor(videoId).then((p) => !cancelled && setPath(p));
    return () => {
      cancelled = true;
    };
  }, [visible, videoId]);

  return (
    <div ref={ref} className={`nle-bin-thumb ${path === null ? "noframe" : ""}`}>
      {path ? <MediaImage path={path} alt="" loading="lazy" /> : path === null ? <span>{label}</span> : null}
    </div>
  );
}

/** "…the [live music] here…" → the bracketed words highlighted, brackets gone. */
function Snippet({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("[") && p.endsWith("]") ? <mark key={i}>{p.slice(1, -1)}</mark> : <span key={i}>{p}</span>,
      )}
    </>
  );
}

/** A bin of the project's footage: search by what is said or seen, or browse by kind. */
export default function MediaBin({ projectId, videos, beatLabel, canReplace, disabled, onAdd, onReplace }: MediaBinProps) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searchedFor, setSearchedFor] = useState("");
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<string>("all");
  const searchSeq = useRef(0);

  const list = useMemo(() => Object.values(videos), [videos]);
  const categories = useMemo(() => {
    const counts = new Map<string, number>();
    for (const v of list) counts.set(v.category, (counts.get(v.category) ?? 0) + 1);
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [list]);

  const clearSearch = () => {
    searchSeq.current++;
    setQuery("");
    setHits(null);
    setSearchedFor("");
    setSearching(false);
    setError(null);
  };

  const runSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) {
      clearSearch();
      return;
    }
    const seq = ++searchSeq.current;
    setSearching(true);
    setError(null);
    try {
      const res = await search(projectId, q, 20);
      if (seq !== searchSeq.current) return;
      setHits(res.hits);
      setSearchedFor(q);
    } catch (err) {
      if (seq === searchSeq.current) setError(String(err));
    } finally {
      if (seq === searchSeq.current) setSearching(false);
    }
  };

  const canAdd = !disabled && beatLabel != null;
  const actions = (pick: Pick) => (
    <div className="nle-bin-actions">
      <button
        type="button"
        className="small nle-bin-add icon-text"
        disabled={!canAdd}
        onClick={() => onAdd(pick)}
        title={beatLabel ? `Add to the end of beat “${beatLabel}”` : "Select a beat first: clips are added to the end of it"}
        aria-label={beatLabel ? `Add to beat ${beatLabel}` : "Add (select a beat first)"}
      >
        <Plus size={14} strokeWidth={2} aria-hidden="true" />
          <span className="nle-bin-add-label">Add</span>
      </button>
      {canReplace && (
        <button
          type="button"
          className="small ghost"
          disabled={disabled}
          onClick={() => onReplace(pick)}
          title="Swap the selected clip for this footage"
        >
          Replace clip
        </button>
      )}
    </div>
  );

  const dragProps = (pick: Pick) => ({
    draggable: !disabled,
    onDragStart: (e: React.DragEvent) => {
      if (disabled) { e.preventDefault(); return; }
      e.dataTransfer.setData(PICK_MIME, JSON.stringify(pick));
      e.dataTransfer.effectAllowed = "copy";
    },
  });

  const browse = list
    .filter((v) => category === "all" || v.category === category)
    .sort((a, b) => fileName(a.path).localeCompare(fileName(b.path)));

  return (
    <div className="nle-bin">
      <form className="nle-panel-tools nle-bin-search" onSubmit={runSearch} role="search">
        <div className="nle-field-wrap with-icon">
          <Search className="nle-field-icon" size={14} strokeWidth={2} aria-hidden="true" />
          <input
            className="nle-filter"
            placeholder="Search media"
            aria-label="Search footage by what is said or seen; press Enter"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              if (!e.target.value.trim()) clearSearch();
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape" && (query || hits)) {
                e.stopPropagation();
                clearSearch();
              }
            }}
          />
          {(query || hits) && (
            <button type="button" className="nle-field-clear" onClick={clearSearch} title="Clear search" aria-label="Clear search">
              <X size={14} strokeWidth={2} aria-hidden="true" />
            </button>
          )}
        </div>
      </form>

      <div className={`nle-bin-hint ${canAdd ? "" : "idle"}`}>
        {disabled
          ? "Adding is paused while Sonar AI edits the cut."
          : beatLabel
            ? <>Add puts the clip at the end of beat <strong>{beatLabel}</strong></>
            : "Select a beat on the timeline, then Add puts clips at its end."}
      </div>
      {error && <div className="banner bad small">{error}</div>}

      {searching && !hits && <div className="muted small nle-panel-empty">Searching…</div>}

      {hits ? (
        <>
          <div className="nle-bin-count" role="status">
            {searching
              ? "Searching…"
              : `${hits.length === 0 ? "No" : hits.length} ${hits.length === 1 ? "moment" : "moments"} for “${searchedFor}”`}
            <span className="muted"> · speech and pictures</span>
          </div>
          <ul className="nle-bin-list">
            {hits.length === 0 && (
              <li className="muted small nle-panel-empty">Try other words, or describe what is on screen.</li>
            )}
            {hits.map((hit, i) => {
              const pick = { video_id: hit.video_id, in_s: hit.start_s, out_s: hit.end_s };
              return (
                <li key={`${hit.video_id}-${hit.start_s}-${i}`} className="nle-bin-hit" {...dragProps(pick)}>
                  <div className={`nle-bin-thumb ${hit.frame ? "" : "noframe"}`}>
                    {hit.frame && <MediaImage path={hit.frame} alt="" loading="lazy" />}
                  </div>
                  <div className="nle-bin-info">
                    <div className="nle-bin-name" title={hit.path}>
                      {fileName(hit.path)}
                    </div>
                    <div className="nle-bin-meta">
                      at {clock(hit.start_s)}–{clock(hit.end_s)}
                      {hit.kinds.length > 0 && (
                        <span className="nle-bin-kinds">
                          {hit.kinds.map((k) => (
                            <span key={k} className="nle-bin-kind">
                              {KIND_LABEL[k] ?? k}
                            </span>
                          ))}
                        </span>
                      )}
                    </div>
                    <div className="nle-bin-snippet">
                      <Snippet text={hit.snippet} />
                    </div>
                    {actions(pick)}
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      ) : (
        <>
          <div className="nle-bin-cats" role="group" aria-label="Filter by kind">
            {[["all", list.length] as [string, number], ...categories].map(([cat, n]) => (
              <button
                key={cat}
                type="button"
                className={`nle-chip ${category === cat ? "active" : ""}`}
                aria-pressed={category === cat}
                onClick={() => setCategory(cat)}
              >
                {CATEGORY_LABEL[cat] ?? cat} <span className="nle-chip-n">{n}</span>
              </button>
            ))}
          </div>
          <ul className="nle-bin-list">
            {browse.map((v) => {
              const dur = v.duration_s ?? 0;
              // The opening seconds, as a starting point to trim on the timeline.
              const pick = { video_id: v.id, in_s: 0, out_s: Math.min(4, dur || 4) };
              return (
                <li key={v.id} className="nle-bin-video" {...dragProps(pick)}>
                  <Poster videoId={v.id} label={CATEGORY_LABEL[v.category] ?? v.category} />
                  <div className="nle-bin-info">
                    <div className="nle-bin-name" title={v.path}>
                      {fileName(v.path)}
                    </div>
                    <div className="nle-bin-meta">
                      {clock(dur)} · {CATEGORY_LABEL[v.category] ?? v.category}
                      {v.segments > 0 ? " · speech" : ""}
                    </div>
                    {actions(pick)}
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
