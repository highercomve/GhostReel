import Modal from "../ui/Modal";
import { useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { promptText } from "../dialogs";
import { Check, Download, SquarePen, X } from "lucide-react";
import {
  exportClipSet,
  isDesktop,
  setClipStatus,
  type ClipItem,
  type ClipSet,
  type ClipStatus,
} from "../api";
import MediaImage from "../MediaImage";
import FramedImage from "./FramedImage";
import { formatDuration } from "../timeFormat";
import { fullTime, relativeTime } from "./relativeTime";

type Filter = "all" | ClipStatus;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "accepted", label: "Accepted" },
  { id: "rejected", label: "Rejected" },
  { id: "new", label: "New" },
];

const VISUAL_LABELS: Record<string, string> = { faces: "Faces", screen: "Screen share", mixed: "Mixed" };

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** "Two-shot 55% · Screen share 45%": what the picture is, measured from the faces. */
function mixText(mix: NonNullable<ClipItem["visual"]["layout"]>): string {
  const parts: [string, number][] = [
    ["Two-shot", mix.two_shot],
    ["Single face", mix.single_face],
    ["Screen share", mix.screen_share],
    ["No face", mix.no_face],
  ];
  return parts
    .filter(([, v]) => v >= 0.05)
    .sort((a, b) => b[1] - a[1])
    .map(([n, v]) => `${n} ${Math.round(v * 100)}%`)
    .join(" · ");
}

/** Total seconds the run took: `timings_s` may be one number or a map of phase to seconds. */
function totalSeconds(timings: unknown): number | null {
  if (typeof timings === "number") return timings;
  if (typeof timings !== "object" || timings === null) return null;
  const map = timings as Record<string, unknown>;
  if (typeof map.total === "number") return map.total;
  const parts = Object.values(map).filter((v): v is number => typeof v === "number");
  return parts.length > 0 ? parts.reduce((a, b) => a + b, 0) : null;
}

/** Who chose the clips, if the run recorded it. */
function chosenBy(log: Record<string, unknown>): string | null {
  const orchestrator = log.orchestrator;
  if (typeof orchestrator !== "object" || orchestrator === null) return null;
  const used = (orchestrator as { used?: unknown }).used;
  if (typeof used !== "boolean") return null;
  return used ? "Chosen by the orchestrator model" : "Chosen by code";
}

/** A folder to export into: the native picker on the desktop, a typed path in the browser build. */
async function pickFolder(): Promise<string | null> {
  if (isDesktop) {
    const picked = await open({ directory: true, multiple: false, title: "Export clips to" });
    return typeof picked === "string" ? picked : null;
  }
  return promptText("Folder to export into (a path on the computer running GhostReel)");
}

function ClipCard({
  item,
  busy,
  onStatus,
  onOpen,
}: {
  item: ClipItem;
  busy: boolean;
  onStatus: (item: ClipItem, status: ClipStatus) => void;
  onOpen: (scriptId: number) => void;
}) {
  const visual = VISUAL_LABELS[item.visual.label];
  const toggle = (status: "accepted" | "rejected") => onStatus(item, item.status === status ? "new" : status);
  return (
    <li className={`card clips-card ${item.status}`}>
      <div className="clips-thumb">
        {item.thumb && item.thumb_view ? (
          // A cropped clip (9:16, 1:1): the keyframe as its framing plan frames it.
          <FramedImage path={item.thumb} view={item.thumb_view} />
        ) : item.thumb ? (
          <MediaImage path={item.thumb} alt="" loading="lazy" />
        ) : (
          <span className="clips-thumb-empty muted small">No thumbnail</span>
        )}
        <span className="clips-thumb-time">{formatDuration(item.out_s - item.in_s)}</span>
      </div>
      <div className="clips-tags small">
        <span className="clips-score" title="Score">{item.score.toFixed(1)}</span>
        {item.visual.layout ? (
          <span className="clips-tag" title="What the picture is, measured from the faces">{mixText(item.visual.layout)}</span>
        ) : (
          visual && <span className="clips-tag">{visual}</span>
        )}
        {item.warnings > 0 && (
          <span
            className="clips-tag warn"
            title={
              item.issues?.filter((i) => i.severity === "warning").map((i) => i.message).join("\n") ||
              "Issues found when checking this clip"
            }
          >
            {plural(item.warnings, "warning")}
          </span>
        )}
        <span className="muted clips-source">
          v{item.video_id} {formatDuration(item.in_s)}–{formatDuration(item.out_s)}
        </span>
      </div>
      <div className="clips-hook">{item.hook || item.title}</div>
      {item.hook && item.title && item.title !== item.hook && <div className="small">{item.title}</div>}
      {item.quote && <blockquote className="clips-quote small">{item.quote}</blockquote>}
      {item.why && <div className="muted small">{item.why}</div>}
      <div className="clips-actions">
        <button
          type="button"
          className={`small icon-text clips-accept${item.status === "accepted" ? " on" : ""}`}
          aria-pressed={item.status === "accepted"}
          disabled={busy}
          onClick={() => toggle("accepted")}
        >
          <Check size={14} strokeWidth={2} aria-hidden="true" /> Accept
        </button>
        <button
          type="button"
          className={`small icon-text clips-reject${item.status === "rejected" ? " on" : ""}`}
          aria-pressed={item.status === "rejected"}
          disabled={busy}
          onClick={() => toggle("rejected")}
        >
          <X size={14} strokeWidth={2} aria-hidden="true" /> Reject
        </button>
        <button type="button" className="ghost small icon-text clips-open" onClick={() => onOpen(item.script_id)}>
          <SquarePen size={14} strokeWidth={2} aria-hidden="true" /> Open in editor
        </button>
      </div>
    </li>
  );
}

/** One clip set as a grid of candidate clips to accept, reject, open in the editor, or export. */
export default function ClipSetGrid({
  set,
  onChange,
  onClose,
  onOpenInEditor,
}: {
  set: ClipSet;
  onChange: (set: ClipSet) => void;
  onClose: () => void;
  onOpenInEditor: (scriptId: number) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [busyId, setBusyId] = useState<number | null>(null);
  const [exporting, setExporting] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const counts = useMemo(() => {
    const c = { accepted: 0, rejected: 0, new: 0 };
    for (const item of set.items) c[item.status] += 1;
    return c;
  }, [set.items]);
  const shown = filter === "all" ? set.items : set.items.filter((i) => i.status === filter);

  const seconds = totalSeconds(set.log.timings_s);
  const by = chosenBy(set.log);
  const format = set.options.format ?? "original";

  const changeStatus = async (item: ClipItem, status: ClipStatus) => {
    setBusyId(item.script_id);
    setError(null);
    try {
      onChange(await setClipStatus(set.id, item.script_id, status));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyId(null);
    }
  };

  const runExport = async (acceptedOnly: boolean) => {
    setError(null);
    setNote(null);
    setExporting(true);
    try {
      const dir = await pickFolder();
      if (!dir) return;
      await exportClipSet(set.id, dir, acceptedOnly);
      setNote("Export started. Progress shows in the cuts panel.");
    } catch (e) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  };

  return (
    <Modal onClose={onClose} className="clips-modal" labelledBy="clip-set-title">
        <div className="nle-modal-header clips-header">
          <div className="clips-header-text">
            <h3 id="clip-set-title">{set.brief.trim() || "No brief"}</h3>
            <div className="muted small" title={fullTime(set.created_at)}>
              {plural(set.items.length, "clip")} · {format} · {relativeTime(set.created_at)}
              {seconds != null && ` · made in ${Math.max(1, Math.round(seconds))} s`}
              {by && ` · ${by}`}
            </div>
            <div className="muted small">
              {counts.accepted} accepted · {counts.rejected} rejected · {counts.new} new
            </div>
          </div>
          <div className="clips-header-actions">
            <button
              type="button"
              className="primary small icon-text"
              disabled={counts.accepted === 0 || exporting}
              onClick={() => void runExport(true)}
            >
              <Download size={14} strokeWidth={2} aria-hidden="true" /> Export accepted…
            </button>
            <button
              type="button"
              className="small icon-text"
              disabled={set.items.length === 0 || exporting}
              onClick={() => void runExport(false)}
            >
              <Download size={14} strokeWidth={2} aria-hidden="true" /> Export all…
            </button>
            <button
              type="button"
              className="ghost small nle-sonar-icon-btn"
              onClick={onClose}
              title="Close (Esc)"
              aria-label="Close"
            >
              <X size={16} strokeWidth={2} aria-hidden="true" />
            </button>
          </div>
        </div>

        {error && <div className="banner bad">{error}</div>}
        {note && <div className="banner good small">{note}</div>}

        <div className="nle-preset-row" role="group" aria-label="Filter clips by status">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              className={`nle-preset-chip${filter === f.id ? " on" : ""}`}
              aria-pressed={filter === f.id}
              onClick={() => setFilter(f.id)}
            >
              {f.label}
              {f.id !== "all" && ` (${counts[f.id]})`}
            </button>
          ))}
        </div>

        {shown.length === 0 ? (
          <p className="muted small">{set.items.length === 0 ? "This set has no clips." : "No clips match this filter."}</p>
        ) : (
          <ul className="clips-grid">
            {shown.map((item) => (
              <ClipCard
                key={item.script_id}
                item={item}
                busy={busyId === item.script_id}
                onStatus={changeStatus}
                onOpen={onOpenInEditor}
              />
            ))}
          </ul>
        )}
    </Modal>
  );
}
