import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Ellipsis, Plus, X } from "lucide-react";
import { chatMessages, type ScriptSummary } from "../api";
import { formatDuration as clock } from "../timeFormat";
import { fullTime, relativeTime, useMinuteNow } from "./relativeTime";

export interface ScriptCut {
  title: string;
  latestScript: ScriptSummary;
  /** Newest first. */
  versions: ScriptSummary[];
}

interface CutsPanelProps {
  cuts: ScriptCut[];
  selectedScriptId: number | null;
  onSelect: (scriptId: number, sessionId: number | null) => void;
  onNewCut: () => void;
  onDeleteCut: (title: string) => void;
  onRestore: (scriptId: number) => Promise<void>;
  onDeleteVersion: (scriptId: number) => void;
}

const versionNote = (text: string) => {
  if (!text || text === "Saved by hand") return text;
  let s = text.replace(/^\[(?:new cut|draft)[^\]]*\]\s*/i, "").trim();
  s = s.replace(
    /^create\s+(?:an?\s+)?(?:(?:upbeat|quick|short)\s+)?(?:(?:\d+|[a-z]+)[ -]?(?:minute|second|min|sec|m|s)s?\s+)?(?:[a-z0-9-]+-style\s+)?(?:cut|video|edit|feature)\s*(?:[;,]\s*)?(?:\b(?:highlighting|featuring|about|of|for|with)\b\s*(?::\s*)?(?:\bthe\b\s+)?)?[.;,]?\s*/i,
    "",
  ).trim();
  s = s.replace(/^create\s+an?\s+edit\s+titled\s+["'][^"']*["']\s+targeting\s+\d+\s+seconds?[.;,]?\s*/i, "").trim();
  s = s.replace(/\bbest insights\b/gi, "insights").replace(/\bkeynote soundbites\b/gi, "keynote quotes")
    .replace(/,?\s+and\s+audience takeaways/gi, ", audience takeaways");
  if (!s) return text;
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/**
 * The project's cuts, like a composition list: one row per edit of the footage, its versions
 * folded underneath. The open cut is expanded so going back a version is one click.
 * Deleting sits behind each row's ⋯ menu so a stray click on a row never reaches it.
 */
export default function CutsPanel({
  cuts,
  selectedScriptId,
  onSelect,
  onNewCut,
  onDeleteCut,
  onRestore,
  onDeleteVersion,
}: CutsPanelProps) {
  const now = useMinuteNow();
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [restoring, setRestoring] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const ids = [...new Set(cuts.flatMap(c => c.versions.map(v => v.session_id)).filter((id): id is number => id != null))];
    void Promise.all(ids.map(async id => {
      const messages = await chatMessages(id);
      const result: Record<number, string> = {};
      let prompt = "";
      for (const message of messages) {
        if (message.role === "user") prompt = message.content.replace(/^\[Draft[^\n]*\]\s*/, "").trim();
        if (message.script_id != null && prompt) result[message.script_id] = prompt;
      }
      return result;
    })).then(parts => { if (!cancelled) setNotes(Object.assign({}, ...parts)); }).catch(() => {});
    return () => { cancelled = true; };
  }, [cuts]);
  const [filter, setFilter] = useState("");
  // Cuts the user folded or unfolded by hand; the open cut is unfolded unless folded here.
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  // The row whose ⋯ menu is open: "cut:<title>" or "v:<script id>".
  const [menuFor, setMenuFor] = useState<string | null>(null);

  const q = filter.trim().toLowerCase();
  const shown = q ? cuts.filter((c) => c.title.toLowerCase().includes(q)) : cuts;

  const toggleMenu = (key: string) => setMenuFor((m) => (m === key ? null : key));

  return (
    <div className="nle-cuts" onKeyDown={(e) => e.key === "Escape" && setMenuFor(null)}>
      <div className="nle-panel-tools">
        <div className="nle-field-wrap">
          <input
            className="nle-filter"
            placeholder="Filter cuts"
            aria-label="Filter cuts by name"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && filter && (e.stopPropagation(), setFilter(""))}
          />
          {filter && (
            <button type="button" className="nle-field-clear" onClick={() => setFilter("")} title="Clear filter" aria-label="Clear filter">
              <X size={14} strokeWidth={2} aria-hidden="true" />
            </button>
          )}
        </div>
        <button type="button" className="ghost small icon-text" onClick={onNewCut} title="Start a new cut of this footage">
          <Plus size={14} strokeWidth={2} aria-hidden="true" /> New cut
        </button>
      </div>

      {shown.length === 0 && (
        <div className="muted small nle-panel-empty">{cuts.length === 0 ? "No cuts yet." : "No cut matches."}</div>
      )}

      <ul className="nle-tree">
        {shown.map((cut) => {
          const isActive = cut.versions.some((v) => v.id === selectedScriptId);
          const open = toggled[cut.title] ?? isActive;
          const latest = cut.latestScript;
          // The oldest version tells how the cut began: Sonar drafts are saved with their chat.
          const first = cut.versions[cut.versions.length - 1];
          const fromSonar = first?.session_id != null;
          const cutKey = `cut:${cut.title}`;
          return (
            <li key={cut.title} className={`nle-tree-cut ${isActive ? "active" : ""}`}>
              <div className="nle-tree-row">
                <button
                  type="button"
                  className="nle-tree-caret"
                  onClick={() => setToggled((t) => ({ ...t, [cut.title]: !open }))}
                  aria-label={open ? "Hide versions" : "Show versions"}
                  aria-expanded={open}
                >
                  {open ? <ChevronDown size={14} strokeWidth={2} aria-hidden="true" /> : <ChevronRight size={14} strokeWidth={2} aria-hidden="true" />}
                </button>
                <button
                  type="button"
                  className="nle-tree-label nle-tree-label-2"
                  onClick={() => onSelect(latest.id, latest.session_id)}
                  title={`${cut.title}\n${plural(cut.versions.length, "version")} · last saved ${fullTime(latest.created_at)}`}
                >
                  <span className="nle-tree-name">{cut.title}</span>
                  <span className="nle-tree-sub">
                    <span className="nle-tree-len">{clock(latest.duration_s)}</span>
                    <span>{cut.versions.length} v</span>
                    <span>{relativeTime(latest.created_at, now)}</span>
                    {fromSonar && (
                      <span className="nle-tree-origin" title="First drafted by Sonar AI">
                        Sonar
                      </span>
                    )}
                  </span>
                </button>
                <button
                  type="button"
                  className={`nle-tree-more ${menuFor === cutKey ? "open" : ""}`}
                  onClick={() => toggleMenu(cutKey)}
                  aria-expanded={menuFor === cutKey}
                  aria-label={`More for ${cut.title}`}
                  title="More"
                >
                  <Ellipsis size={16} strokeWidth={2} aria-hidden="true" />
                </button>
              </div>
              {menuFor === cutKey && (
                <div className="nle-tree-menu">
                  <button
                    type="button"
                    className="small danger"
                    onClick={() => {
                      setMenuFor(null);
                      onDeleteCut(cut.title);
                    }}
                  >
                    Delete cut{cut.versions.length > 1 ? ` and its ${cut.versions.length} versions` : ""}…
                  </button>
                  <button type="button" className="small ghost" onClick={() => setMenuFor(null)}>
                    Cancel
                  </button>
                </div>
              )}
              {open && (
                <ul className="nle-tree-versions">
                  {cut.versions.map((v) => {
                    const vKey = `v:${v.id}`;
                    return (
                      <li key={v.id} className={v.id === selectedScriptId ? "active" : ""}>
                        <div className="nle-tree-vrow">
                          <button
                            type="button"
                            className="nle-tree-label"
                            onClick={() => onSelect(v.id, v.session_id)}
                            title={`v${v.version} · saved ${fullTime(v.created_at)} · ${plural(v.beats, "beat")}, ${plural(v.clips, "clip")}`}
                          >
                            <span className="nle-tree-name">
                              v{v.version}
                              {v.id === latest.id && <span className="nle-tree-latest">latest</span>}
                            </span>
                            <span className="nle-tree-meta">
                              {clock(v.duration_s)} · {relativeTime(v.created_at, now)}
                            </span>
                          </button>
                          {(
                            <button
                              type="button"
                              className={`nle-tree-more ${menuFor === vKey ? "open" : ""}`}
                              onClick={() => toggleMenu(vKey)}
                              aria-expanded={menuFor === vKey}
                              aria-label={`More for v${v.version}`}
                              title="More"
                            >
                              <Ellipsis size={16} strokeWidth={2} aria-hidden="true" />
                            </button>
                          )}
                        </div>
                        <details className="nle-version-details">
                          <summary><span className="nle-version-note" title={notes[v.id] ?? "Saved by hand"}><span>{versionNote(notes[v.id] ?? "Saved by hand")}</span></span><span className="nle-version-more">More</span></summary>
                          <div className="nle-version-full">{notes[v.id] ?? "Saved by hand"}</div>
                        </details>
                        {v.id !== latest.id && (
                          <button type="button" className="ghost small nle-version-restore" disabled={restoring} onClick={async () => {
                            setRestoring(true);
                            try { await onRestore(v.id); } finally { setRestoring(false); }
                          }}>Restore as latest</button>
                        )}
                        {menuFor === vKey && (
                          <div className="nle-tree-menu">
                            <button
                              type="button"
                              className="small danger"
                              onClick={() => {
                                setMenuFor(null);
                                onDeleteVersion(v.id);
                              }}
                            >
                              Delete v{v.version}…
                            </button>
                            <button type="button" className="small ghost" onClick={() => setMenuFor(null)}>
                              Cancel
                            </button>
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
