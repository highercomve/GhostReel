import DurationField from "./editor/DurationField";
import { confirmAction, notify } from "./dialogs";
import { fullTime } from "./editor/relativeTime";
import MediaImage from "./MediaImage";
import ChatAnswerStats from "./ChatAnswerStats";
import { useCallback, useEffect, useRef, useState } from "react";
import { useMemo } from "react";
import {
  getAiSettings,
  setAiSettings,
  modelsStatus,
  chatMessages,
  chatSessions,
  deleteChatSession,
  chatTurn,
  buildScriptWithJev,
  listScripts,
  saveScript,
  getScript,
  deleteScript, deleteScriptCut,
  clipsPreflight,
  generateClips,
  listClipSets,
  deleteClipSet,
  listTrash,
  restoreTrash,
  projectView,
  clock,
  fileName,
  type Script,
  type AiSettings,
  type ClipFormat,
  type ClipSet,
  type AgentSelection,
  type ClipOptions,
  type VisionSettings,
  type ChatEvent,
  type ChatMessage,
  type ChatProgress,
  type ChatSession,
  type ChatStyle,
  type Issue,
  type ModelStatus,
  type ScriptSummary,
  type ToolCallRecord,
  type TrashItem,
  type VisionSettingsPatch,
} from "./api";
import { onEvent } from "./events";
import ChatLogModal from "./ChatLogModal";
import ClipSetGrid from "./editor/ClipSetGrid";
import CutsPanel from "./editor/CutsPanel";
import ModelSelect from "./editor/ModelSelect";
import ScriptEditor, { type ScriptEditorHandle } from "./ScriptEditor";
import TaskCard from "./TaskCard";
import Modal from "./ui/Modal";
import AgentPicker, { agentSelectionProblem } from "./agents/AgentPicker";
import { useAgents, useResolvedAgents } from "./agents/useAgents";
import RunLog from "./runs/RunLog";
import PreflightList, { usePreflight } from "./runs/PreflightList";
import { useProjectRuns } from "./runs/useRuns";
import { toMs, type RunView } from "./runs/model";
import type { LinkContext } from "./runs/links";
import { onTrashChanged, trashChanged } from "./runs/trashEvents";
import { useQueue } from "./useQueue";
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clapperboard,
  Copy,
  FilePlus,
  PanelRight,
  Paperclip,
  PictureInPicture2,
  Plus,
  Radar,
  Scissors,
  Settings2,
  Sparkles,
  SquarePen,
  X,
} from "lucide-react";
import {
  describeCall,
  describeLiveStep,
  describeResearch,
  parseAnswer,
  parseDraftHeader,
} from "./sonarFormat";

function cleanTitle(title: string | null | undefined): string {
  if (!title) return "Untitled chat";
  // A chat opened from the New cut dialog is titled after its brief header.
  const draft = parseDraftHeader(title.trim() + "\n");
  if (draft) return `${draft.title} · target ${clock(draft.target_s)}`;
  const firstLine = title.split("\n")[0].trim();
  const cleaned = firstLine.replace(/^```[a-z]*\s*/i, "").replace(/[`"{}[\]]/g, "").trim();
  return cleaned.length > 55 ? cleaned.slice(0, 55) + "…" : cleaned || "Untitled chat";
}

function formatRelativeTime(timestamp: number): string {
  const ms = timestamp < 1e11 ? timestamp * 1000 : timestamp;
  const diffSecs = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (diffSecs < 60) return "just now";
  if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
  if (diffSecs < 86400) return `${Math.floor(diffSecs / 3600)}h ago`;
  return `${Math.floor(diffSecs / 86400)}d ago`;
}

/** The three ways a brief becomes a cut, in the order they cost wall-clock time. */
type Mode = "chat" | "jev" | "both";

const CLI_NAMES: Record<string, string> = { claude: "Claude", agy: "Agy", opencode: "OpenCode", codex: "Codex" };

/** One plain line naming the engine, for the folded options. Never a URL. */
function brainSummary(m: VisionSettings): string {
  switch (m.backend) {
    case "local":
      return `Local · ${m.local_model}`;
    case "server":
      return m.model ? `Server · ${m.model}` : "Server model";
    case "cli":
      return `${CLI_NAMES[m.cli.tool] ?? m.cli.tool} (CLI)`;
    default:
      return "Auto engine";
  }
}

const ENGINE_HINTS: Record<VisionSettings["backend"], string> = {
  auto: "Uses the best engine that is available.",
  local: "Runs a model on this computer.",
  server: "Uses a model server you run; set its address below.",
  cli: "Uses an AI command-line tool, such as Claude Code.",
};

const JEV_BLOCKED = "Needs the quote-builder key (Jev). Set it in Settings.";

const MODES: { id: Mode; label: string; short: string; action: string; hint: string }[] = [
  {
    id: "chat",
    label: "AI writes it",
    short: "AI writes",
    action: "Send",
    hint: "AI researches the footage and writes the cut. Takes a few minutes.",
  },
  {
    id: "jev",
    label: "Quick build from quotes",
    short: "Quick build",
    action: "Build",
    hint: "Builds a cut from interview quotes in about 15 seconds.",
  },
  {
    id: "both",
    label: "Quick build, then AI polish",
    short: "Quick build + polish",
    action: "Build & polish",
    hint: "Builds a first cut from interview quotes, then AI improves it. Takes a few minutes.",
  },
];

/** How long "Clip set deleted · Undo" stays after a delete. */
const UNDO_MS = 8000;

/** Runs shown in the Sonar thread (the newest; older ones are on the Activity page). */
const SONAR_RUNS = 5;

const NEW_CHAT_EXAMPLES = [
  "A 30-second teaser that opens on a strong spoken hook",
  "A 2-minute recap of the best interview moments",
  "A B-roll montage of the venue, no talking",
];

type CutPreset = "social_30s" | "interview_3m" | "doc_15m";

const CUT_PRESETS: { id: CutPreset; label: string; title: string; target: number; prompt: string }[] = [
  {
    id: "social_30s",
    label: "Social reel",
    title: "Social Reel (30s)",
    target: 30,
    prompt:
      "Create an upbeat 30-second video for social media from this footage. Fast pacing, strong speech hook, energetic B-roll montage.",
  },
  {
    id: "interview_3m",
    label: "Highlights",
    title: "Highlights (3m)",
    target: 180,
    prompt: "Create a 3-minute cut highlighting the best insights, keynote soundbites, and audience takeaways.",
  },
  {
    id: "doc_15m",
    label: "Documentary",
    title: "Documentary (15m)",
    target: 900,
    prompt:
      "Create a 15-minute documentary-style feature. Build a thoughtful narrative arc with deep speaker quotes, context, and rich natural sound.",
  },
];

/** Survives component unmount/remount (e.g. navigating to Activity/Settings/Library and back). */
const sessionEventsCache = new Map<number, ChatEvent[]>();

/** File names by video id, per project, so research steps can name the clips they looked at. */
const videoNamesCache = new Map<number, Map<number, string>>();

const DOCK_KEY = "ghostreel.sonar.docked";

/** An on/off setting that says which it is, not a pill whose colour has to be decoded. */
function Switch({
  checked,
  disabled,
  onChange,
  label,
  hint,
  title,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (next: boolean) => void;
  label: string;
  hint?: string;
  title?: string;
}) {
  return (
    <label className={`nle-switch${disabled ? " disabled" : ""}`} title={title}>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="nle-switch-track" aria-hidden="true">
        <span className="nle-switch-thumb" />
      </span>
      <span className="nle-switch-text">
        <span className="nle-switch-label">
          {label} <span className={`nle-switch-state${checked ? " on" : ""}`}>{checked ? "On" : "Off"}</span>
        </span>
        {hint && <span className="nle-switch-hint">{hint}</span>}
      </span>
    </label>
  );
}

/** A research pass in one line ("Looked through 9 clips"), each step on demand. */
function ResearchSummary({ calls, names }: { calls: ToolCallRecord[]; names: Map<number, string> }) {
  // The same clip looked at twice is one line, counted.
  const lines: { text: string; detail?: string; count: number; title: string }[] = [];
  for (const c of calls) {
    const d = describeCall(c, names);
    const same = lines.find((l) => l.text === d.text);
    if (same) same.count++;
    else
      lines.push({
        ...d,
        count: 1,
        title: `${c.tool} ${typeof c.args === "string" ? c.args : JSON.stringify(c.args)}`,
      });
  }
  return (
    <details className="research-steps done">
      <summary>
        <span className="research-dot" />
        <span className="research-now">{describeResearch(calls)}</span>
        <ChevronRight size={14} strokeWidth={2} aria-hidden="true" className="research-caret" />
      </summary>
      <ol>
        {lines.map((l, k) => (
          <li key={k} title={l.title}>
            {l.text}
            {l.count > 1 && <span className="muted"> ×{l.count}</span>}
            {l.detail && <span className="muted"> · {l.detail}</span>}
          </li>
        ))}
      </ol>
    </details>
  );
}

/** An assistant answer: a short summary, then what changed as bullet lists. */
function AnswerBody({ content }: { content: string }) {
  const { summary, sections } = parseAnswer(content);
  return (
    <>
      {summary && <div className="bubble-text">{summary}</div>}
      {sections.map((s, k) => (
        <div key={k} className={`answer-section ${s.kind}`}>
          <div className="answer-section-title">{s.title}</div>
          <ul>
            {s.items.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      ))}
    </>
  );
}

/** A user message; a new-cut brief shows its title and target as a chip. */
function UserText({ text }: { text: string }) {
  const draft = parseDraftHeader(text);
  if (!draft) return <div className="bubble-text">{text}</div>;
  return (
    <>
      <div className="bubble-draft-chip">
        New cut · {draft.title} · {clock(draft.target_s)} target
      </div>
      {draft.rest.trim() && <div className="bubble-text">{draft.rest.trim()}</div>}
    </>
  );
}

const CLIP_FORMATS: { id: ClipFormat; label: string }[] = [
  { id: "original", label: "Original" },
  { id: "9:16", label: "9:16" },
  { id: "1:1", label: "1:1" },
];

const clampInt = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(value)));

export default function ScriptsPanel({
  projectId,
  selectedVideoIds = [],
  videoCount = 0,
}: {
  projectId: number;
  /** Videos picked in the library; empty means every video. */
  selectedVideoIds?: number[];
  videoCount?: number;
}) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [scripts, setScripts] = useState<ScriptSummary[]>([]);
  const [selectedScriptId, setSelectedScriptId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [turnRunning, setTurnRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<ChatEvent[]>([]);
  const [optimisticUser, setOptimisticUser] = useState<{ text: string; images: string[] } | null>(null);
  const [turnError, setTurnError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // NLE Sonar panel states
  const [sonarOpen, setSonarOpen] = useState(false);
  const [sonarTab, setSonarTab] = useState<"chat" | "history">("chat");
  // Docked or floating is a working preference: remembered across sessions.
  const [sonarDocked, setSonarDocked] = useState(() => {
    try {
      return localStorage.getItem(DOCK_KEY) === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(DOCK_KEY, sonarDocked ? "1" : "0");
    } catch {
      // Storage unavailable: the choice just isn't remembered.
    }
  }, [sonarDocked]);
  const scriptEditorRef = useRef<ScriptEditorHandle>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  // Whether the open cut has edits not saved as a version. The editor keeps this to itself
  // (a handle, not a prop), so the toolbar reads it on a short interval.
  const [editorDirty, setEditorDirty] = useState(false);
  useEffect(() => {
    const id = window.setInterval(() => {
      const dirty = Boolean(scriptEditorRef.current?.isDirty());
      setEditorDirty((prev) => (prev === dirty ? prev : dirty));
    }, 600);
    return () => window.clearInterval(id);
  }, []);

  // New cut dialog: how to start first, presets only for a Sonar draft.
  const [newCutModalOpen, setNewCutModalOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<CutPreset | null>("social_30s");
  const [newCutTitle, setNewCutTitle] = useState(CUT_PRESETS[0].title);
  const [newCutTargetS, setNewCutTargetS] = useState(CUT_PRESETS[0].target);
  const [newCutType, setNewCutType] = useState<"ai" | "blank" | "branch" | "clips">("ai");
  const [newCutPrompt, setNewCutPrompt] = useState(CUT_PRESETS[0].prompt);

  const applyPreset = (preset: CutPreset) => {
    const p = CUT_PRESETS.find((x) => x.id === preset) ?? CUT_PRESETS[0];
    setSelectedPreset(p.id);
    setNewCutType("ai");
    setNewCutTitle(p.title);
    setNewCutTargetS(p.target);
    setNewCutPrompt(p.prompt);
  };

  // Clip sets: the generated-clips form, the sets themselves, and the one open in the grid.
  const [clipBrief, setClipBrief] = useState("");
  const [clipCount, setClipCount] = useState(8);
  const [clipMinS, setClipMinS] = useState(20);
  const [clipMaxS, setClipMaxS] = useState(60);
  const [clipFormat, setClipFormat] = useState<ClipFormat>("original");
  /** This run's agents; `null` = the app default. */
  const [clipAgents, setClipAgents] = useState<AgentSelection | null>(null);
  const [clipLayout, setClipLayout] = useState<"follow_speaker" | "stacked">("follow_speaker");
  const [clipSets, setClipSets] = useState<ClipSet[]>([]);
  const [openClipSetId, setOpenClipSetId] = useState<string | null>(null);
  // The lists as of the latest render: a handler that waits on a dialog reads these afterwards
  // instead of the values its closure captured before the wait.
  const latest = useRef({ scripts, selectedScriptId, clipSets });
  latest.current = { scripts, selectedScriptId, clipSets };

  const reloadClipSets = () => {
    listClipSets(projectId).then(setClipSets).catch(() => {});
  };
  useEffect(() => {
    setClipSets([]);
    setOpenClipSetId(null);
    reloadClipSets();
  }, [projectId]);

  // Deletes go to the trash: what is in it (Recently deleted), and the last delete, offered for
  // undo for a few seconds.
  const [trash, setTrash] = useState<TrashItem[]>([]);
  const [undoItem, setUndoItem] = useState<TrashItem | null>(null);
  // Restores in flight. The undo toast stays up meanwhile, so it never vanishes mid-restore.
  const [restoresPending, setRestoresPending] = useState(0);
  useEffect(() => {
    let alive = true;
    const reload = () => {
      listTrash(projectId)
        .then((t) => alive && setTrash(t))
        .catch(() => {});
    };
    setTrash([]);
    setUndoItem(null);
    reload();
    // A restore from anywhere in this window (a run's link, Recently deleted) brings rows back.
    const off = onTrashChanged(() => {
      reload();
      listScripts(projectId)
        .then((s) => alive && setScripts(s))
        .catch(() => {});
      listClipSets(projectId)
        .then((c) => alive && setClipSets(c))
        .catch(() => {});
    });
    return () => {
      alive = false;
      off();
    };
  }, [projectId]);
  useEffect(() => {
    if (!undoItem || restoresPending > 0) return;
    const t = window.setTimeout(() => setUndoItem(null), UNDO_MS);
    return () => window.clearTimeout(t);
  }, [undoItem, restoresPending]);
  const afterDelete = (item: TrashItem) => {
    setUndoItem(item);
    trashChanged();
  };
  const restoreFromTrash = async (trashId: string) => {
    setRestoresPending((n) => n + 1);
    try {
      await restoreTrash(trashId);
      trashChanged();
    } catch (err) {
      setTurnError(String(err));
    } finally {
      // Done either way: the toast for this item has served its purpose (on failure the item is
      // still in Recently deleted, and the error is shown).
      setUndoItem((u) => (u?.id === trashId ? null : u));
      setRestoresPending((n) => n - 1);
    }
  };

  const chooseCutMethod = (method: "ai" | "blank" | "branch" | "clips") => {
    if (method === "clips") {
      setNewCutType("clips");
      return;
    }
    if (method === "ai") {
      applyPreset(selectedPreset ?? "social_30s");
      return;
    }
    setNewCutType(method);
    if (method === "blank") {
      setNewCutTitle("Untitled cut");
      setNewCutTargetS(60);
    } else {
      const current = scriptEditorRef.current?.getScript();
      setNewCutTitle(activeScript ? `${activeScript.title} (copy)` : "Copy of cut");
      setNewCutTargetS(
        Math.round(current?.target_duration_s || activeScript?.duration_s || 60),
      );
    }
  };

  // Stable, so the dialog's Escape listener is not re-registered on every render.
  const closeNewCutModal = useCallback(() => setNewCutModalOpen(false), []);

  // Names for the clips Sonar researched; read once per project.
  const [videoNames, setVideoNames] = useState<Map<number, string>>(
    () => videoNamesCache.get(projectId) ?? new Map(),
  );
  useEffect(() => {
    const cached = videoNamesCache.get(projectId);
    if (cached) {
      setVideoNames(cached);
      return;
    }
    let cancelled = false;
    projectView(projectId)
      .then((view) => {
        const names = new Map(view.videos.map((v) => [v.id, fileName(v.path)] as [number, string]));
        videoNamesCache.set(projectId, names);
        if (!cancelled) setVideoNames(names);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // Toggle Sonar overlay with keyboard shortcut (Cmd+J / Ctrl+J)
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") {
        e.preventDefault();
        setSonarOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);


  const addImages = (files: FileList | File[]) => {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (loadEvent) => {
          const res = loadEvent.target?.result;
          if (typeof res === "string") {
            setAttachedImages((prev) => [...prev, res]);
          }
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const imageFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          imageFiles.push(file);
        }
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      addImages(imageFiles);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      let hasImg = false;
      for (let i = 0; i < files.length; i++) {
        if (files[i].type.startsWith("image/")) {
          hasImg = true;
          break;
        }
      }
      if (hasImg) {
        e.preventDefault();
        addImages(files);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addImages(e.target.files);
      e.target.value = "";
    }
  };
  /** Jev is building a cut by choosing. One shot, no conversation, ~15 s. */
  const [building, setBuilding] = useState(false);
  /**
   * How this brief becomes a cut.
   *
   * One decision, not two buttons and a checkbox. The three ways differ by which brain chooses
   * and which writes: a model researches the project and writes (slow, good structure); Jev only
   * chooses out of the index (~15 s, grounded, cannot write narration); or Jev chooses and the
   * model improves what it chose, which measured best — 76 editorial against 65 for Jev alone
   * and 76 for agy alone at three times the wall clock. The local models, worst at the research,
   * gain the most from not having to do it.
   */
  const [mode, setMode] = useState<Mode>("chat");
  // The build options fold into one summary line; a narrow drawer has no room for them all.
  const [optionsOpen, setOptionsOpen] = useState(false);
  /** Whether the Jev modes can run at all, so they are disabled with a reason, not failed with one. */
  const [jevReady, setJevReady] = useState<boolean | null>(null);
  /** Whether every finished cut is read editorially. `null` until settings have been read. */
  const [judging, setJudging] = useState<boolean | null>(null);
  /**
   * B-roll only: a montage on a theme with no interviews and no voice-over. Chosen per chat and
   * kept on it, so a follow-up turn cannot quietly drift back to an interview cut.
   */
  const [style, setStyle] = useState<ChatStyle>({ broll: false, natural_sound: true });
  const [latestIssues, setLatestIssues] = useState<Issue[] | undefined>(undefined);

  const messagesRef = useRef<HTMLDivElement>(null);
  const tasks = useQueue();
  const projectRuns = useProjectRuns(projectId, 20);
  const userSelectedRef = useRef<boolean>(false);

  // A turn's session exists from the moment it is queued — the backend creates it before the
  // task starts — but the list was only re-read when the turn returned, so a chat that took ten
  // minutes was invisible for ten minutes: "No sessions yet" beside a card reading "Chat #1".
  // Follow the running task instead: show its session as soon as it has one, and select it if
  // nothing else is selected, so the conversation is traceable while it happens.
  const liveSessionId = (() => {
    for (const t of tasks) {
      if ((t.state === "running" || t.state === "queued") && t.kind.type === "chat" && t.kind.project_id === projectId) {
        return t.kind.session_id;
      }
    }
    return null;
  })();

  // Find active chat task in queue (running or queued)
  const activeChatTask = tasks.find(
    (t) =>
      (t.state === "running" || t.state === "queued") &&
      t.kind.type === "chat" &&
      t.kind.project_id === projectId &&
      (selectedSessionId == null || t.kind.session_id === (selectedSessionId ?? liveSessionId)),
  );

  const isTurnRunning = turnRunning || Boolean(activeChatTask);

  const removeSession = async (id: number) => {
    if (!(await confirmAction("Remove this chat? Scripts drafted in it are kept.", { okLabel: "Remove", destructive: true }))) return;
    try {
      await deleteChatSession(id);
    } catch (e) {
      console.error(e);
      return;
    }
    setSessions((prev) => prev.filter((s) => s.id !== id));
    listScripts(projectId).then(setScripts).catch(() => {});
    if (selectedSessionId === id) {
      setSelectedSessionId(null);
      setSelectedScriptId(null);
      setMessages([]);
    }
  };

  const [aiSettings, setAiSettingsState] = useState<AiSettings | null>(null);
  const [researchLimit, setResearchLimit] = useState(0);
  const researchLimitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelStatus[]>([]);

  // Whether the Jev modes are offerable at all. Asked once: a mode that cannot run should say so
  // before it is chosen, not fail after.
  useEffect(() => {
    let cancelled = false;
    getAiSettings()
      .then((ai) => {
        if (cancelled) return;
        setAiSettingsState(ai);
        setResearchLimit(ai.chat_model.max_tool_rounds ?? 0);
        setJevReady(ai.jev.enabled && (ai.jev.has_key || ai.jev.key_from_env));
        setJudging(ai.jev.judge);
      })
      .catch(() => {
        // Unknown, not unavailable: leave the modes enabled rather than hiding them over a
        // settings read that happened to fail.
        if (!cancelled) setJevReady(null);
      });

    modelsStatus()
      .then((st) => {
        if (!cancelled) {
          setAvailableModels(st.models.filter((m) => m.entry.kind === "vision"));
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, []);

  const handleUpdateChatModel = async (patch: VisionSettingsPatch) => {
    try {
      const nextAi = await setAiSettings({ chat_model: patch });
      setAiSettingsState(nextAi);
    } catch (err) {
      setTurnError(String(err));
    }
  };

  const saveResearchLimit = (raw: string) => {
    if (researchLimitTimer.current) clearTimeout(researchLimitTimer.current);
    // Empty means no cap.
    const value = raw.trim() === "" ? 0 : Number(raw);
    if (!Number.isInteger(value) || value < 0) {
      setResearchLimit(aiSettings?.chat_model.max_tool_rounds ?? 0);
      return;
    }
    setResearchLimit(value);
    if (value !== (aiSettings?.chat_model.max_tool_rounds ?? 0)) {
      void handleUpdateChatModel({ max_tool_rounds: value });
    }
  };

  useEffect(() => {
    if (!aiSettings) return;
    setResearchLimit(aiSettings.chat_model.max_tool_rounds ?? 0);
  }, [aiSettings?.chat_model.max_tool_rounds]);

  useEffect(() => () => {
    if (researchLimitTimer.current) clearTimeout(researchLimitTimer.current);
  }, []);

  // Load sessions and scripts on mount / projectId change
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      chatSessions(projectId).catch(() => [] as ChatSession[]),
      listScripts(projectId).catch(() => [] as ScriptSummary[]),
    ]).then(([sessList, scriptList]) => {
      if (cancelled) return;
      // Newest sessions first
      const sortedSess = [...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at));
      setSessions(sortedSess);
      setScripts(scriptList);

      // Default select the live running session if any, otherwise the latest session
      const targetSessionId = liveSessionId ?? (sortedSess.length > 0 ? sortedSess[0].id : null);
      if (targetSessionId != null) {
        setSelectedSessionId(targetSessionId);
        chatMessages(targetSessionId).then((msgs) => {
          if (!cancelled) setMessages(msgs);
        }).catch(() => {});

        // Find latest script for this session
        const sessScripts = scriptList.filter((s) => s.session_id === targetSessionId);
        setSelectedScriptId(sessScripts.length > 0 ? Math.max(...sessScripts.map((s) => s.id)) : null);
      } else {
        setSelectedSessionId(null);
        setSelectedScriptId(null);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // When selected session changes, load messages
  useEffect(() => {
    if (selectedSessionId == null) {
      setMessages([]);
      return;
    }
    chatMessages(selectedSessionId)
      .then(setMessages)
      .catch(() => setMessages([]));
  }, [selectedSessionId]);

  // Listen to live chat-progress events
  useEffect(() => {
    return onEvent<ChatProgress>("chat-progress", (p) => {
      const cached = sessionEventsCache.get(p.session_id) ?? [];
      const updated = [...cached, p.event];
      sessionEventsCache.set(p.session_id, updated);
      if (selectedSessionId == null || p.session_id === selectedSessionId || p.session_id === liveSessionId) {
        setLiveEvents(updated);
      }
    });
  }, [selectedSessionId, liveSessionId]);

  // Restore live events on mount or session change
  useEffect(() => {
    const sid = selectedSessionId ?? liveSessionId;
    if (!sid) return;
    const fromCache = sessionEventsCache.get(sid);
    const fromTask = activeChatTask?.chat_events;
    if (fromCache && fromCache.length > 0) {
      setLiveEvents(fromCache);
    } else if (fromTask && fromTask.length > 0) {
      sessionEventsCache.set(sid, fromTask);
      setLiveEvents(fromTask);
    }
  }, [selectedSessionId, liveSessionId, activeChatTask?.id]);

  // Sync if task has more events from backend
  useEffect(() => {
    const sid = selectedSessionId ?? liveSessionId;
    if (activeChatTask?.chat_events && activeChatTask.chat_events.length > liveEvents.length) {
      if (sid) {
        sessionEventsCache.set(sid, activeChatTask.chat_events);
      }
      setLiveEvents(activeChatTask.chat_events);
    }
  }, [activeChatTask?.chat_events, liveEvents.length, selectedSessionId, liveSessionId]);

  const displayEvents = liveEvents.length > 0 ? liveEvents : (activeChatTask?.chat_events ?? []);

  // Keep the chat pinned to its latest message. Scrolling the element itself rather than calling
  // scrollIntoView on a marker: that walks up every scrollable ancestor, so a new message dragged
  // the whole page down with it.
  useEffect(() => {
    const list = messagesRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, displayEvents, optimisticUser, isTurnRunning]);

  // Detect when a running chat task finishes while viewing it
  const prevActiveRef = useRef<boolean>(false);
  useEffect(() => {
    const wasActive = prevActiveRef.current;
    const nowActive = Boolean(activeChatTask);
    prevActiveRef.current = nowActive;
    if (wasActive && !nowActive) {
      const sid = selectedSessionId ?? liveSessionId;
      if (sid != null) {
        sessionEventsCache.delete(sid);
        chatMessages(sid).then(setMessages).catch(() => {});
      }
      setLiveEvents([]);
      setOptimisticUser(null);
      setTurnRunning(false);
      chatSessions(projectId).then((list) => {
        setSessions([...list].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
      }).catch(() => {});
      listScripts(projectId).then((list) => {
        setScripts(list);
        if (sid != null) {
          const sessScripts = list.filter((s) => s.session_id === sid);
          if (sessScripts.length > 0) {
            setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
          }
        }
      }).catch(() => {});
    }
  }, [activeChatTask, selectedSessionId, liveSessionId, projectId]);

  // Also listen for task-finished event as a fallback
  useEffect(() => {
    return onEvent<number>("task-finished", async () => {
      reloadClipSets();
      const [sessList, scriptList] = await Promise.all([
        chatSessions(projectId).catch(() => [] as ChatSession[]),
        listScripts(projectId).catch(() => [] as ScriptSummary[]),
      ]);
      setSessions([...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
      setScripts(scriptList);
      const sid = selectedSessionId ?? liveSessionId;
      if (sid != null) {
        sessionEventsCache.delete(sid);
        const msgs = await chatMessages(sid).catch(() => [] as ChatMessage[]);
        setMessages(msgs);
        const sessScripts = scriptList.filter((s) => s.session_id === sid);
        if (sessScripts.length > 0) {
          setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
        }
      }
      setLiveEvents([]);
      setOptimisticUser(null);
      setTurnRunning(false);
    });
  }, [projectId, selectedSessionId, liveSessionId]);

  useEffect(() => {
    if (liveSessionId == null) return;
    let cancelled = false;
    if (!sessions.some((s) => s.id === liveSessionId)) {
      chatSessions(projectId)
        .then((list) => {
          if (cancelled) return;
          setSessions([...list].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
        })
        .catch(() => {});
    }
    if (!userSelectedRef.current && selectedSessionId !== liveSessionId) {
      setSelectedSessionId(liveSessionId);
    }
    return () => {
      cancelled = true;
    };
  }, [liveSessionId, projectId, selectedSessionId]);

  const handleSelectSession = async (sessId: number) => {
    // Auto-select latest script belonging to this session if any, otherwise null
    const sessScripts = scripts.filter((s) => s.session_id === sessId);
    const targetScriptId = sessScripts.length > 0 ? Math.max(...sessScripts.map((s) => s.id)) : null;
    await selectScript(targetScriptId, sessId);
  };

  const handleNewChat = async () => {
    await selectScript(null, null);
    setTurnError(null);
    setLiveEvents([]);
    setSonarOpen(true);
    setSonarTab("chat");
  };

  // A chat that chose a style keeps it: follow the selected chat's own. A new chat keeps
  // whatever the composer is set to.
  const selectedStyle = sessions.find((s) => s.id === selectedSessionId)?.style;
  // Only a b-roll chat has chosen its sound; an interview chat leaves the composer's choice alone.
  useEffect(() => {
    if (selectedStyle) setStyle((s) => (selectedStyle.broll ? selectedStyle : { ...s, broll: false }));
  }, [selectedSessionId, selectedStyle?.broll, selectedStyle?.natural_sound]);

  // Jev builds from interview quotes; a montage has none, so only the model can write one.
  useEffect(() => {
    if (style.broll && mode !== "chat") setMode("chat");
  }, [style.broll, mode]);

  const handleSend = async () => {
    const text = inputMessage.trim();
    const imagesToSend = [...attachedImages];
    if ((!text && imagesToSend.length === 0) || isTurnRunning) return;
    userSelectedRef.current = true;
    const promptText = text || (imagesToSend.length > 0 ? "Review the attached image(s)." : "");
    setInputMessage("");
    setAttachedImages([]);
    setTurnRunning(true);
    setTurnError(null);
    setLiveEvents([]);
    setOptimisticUser({ text: promptText, images: imagesToSend });

    try {
      // If the editor has unsaved manual changes, SAVE THEM FIRST!
      // This ensures the manual changes are written to the database under the current session,
      // so the AI reads the manual edits as its baseline for the next turn.
      let baselineScriptId = selectedScriptId;
      if (scriptEditorRef.current?.isDirty()) {
        const savedId = await scriptEditorRef.current.save();
        if (savedId == null) {
          // Abort turn to prevent AI generating against stale baseline without user edits
          setTurnError("Failed to save manual edits before generating. Edit not sent.");
          setTurnRunning(false);
          setOptimisticUser(null);
          setInputMessage(text);
          setAttachedImages(imagesToSend);
          return;
        }
        baselineScriptId = savedId;
        setSelectedScriptId(savedId);
        const updatedScripts = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
        setScripts(updatedScripts);
      }

      // Jev first, when asked. The build lands in the open chat (or opens one), so the model's
      // turn that follows already holds the brief, the cut and the editorial notes — and a build
      // asked for mid-conversation continues that conversation rather than starting another.
      let session = selectedSessionId;
      let message = promptText;
      if (mode !== "chat") {
        setBuilding(true);
        const built = await buildScriptWithJev(projectId, session, promptText, 40).finally(() =>
          setBuilding(false),
        );
        session = built.session_id;
        setSelectedSessionId(built.session_id);
        setSelectedScriptId(built.script_id);
        setScripts(await listScripts(projectId).catch(() => [] as ScriptSummary[]));
        setSessions(
          [...(await chatSessions(projectId).catch(() => [] as ChatSession[]))].sort(
            (a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at),
          ),
        );
        // Jev does not write, so it cannot hold a conversation: this mode ends here, with the
        // session left selected so anything typed next refines the cut rather than starting over.
        if (mode === "jev") {
          setMessages(await chatMessages(built.session_id).catch(() => [] as ChatMessage[]));
          return;
        }
        message =
          "Improve this cut. Keep it to the brief, fix the editorial notes above, and keep every " +
          "quote a whole sentence. Reply with the full script JSON.";
      }
      const res = await chatTurn(projectId, session, message, imagesToSend, style, baselineScriptId);
      setSelectedSessionId(res.session_id);
      userSelectedRef.current = true;
      sessionEventsCache.delete(res.session_id);

      // Refresh sessions, messages, and scripts
      const [sessList, msgs, scriptList] = await Promise.all([
        chatSessions(projectId).catch(() => [] as ChatSession[]),
        chatMessages(res.session_id).catch(() => [] as ChatMessage[]),
        listScripts(projectId).catch(() => [] as ScriptSummary[]),
      ]);
      const sortedSess = [...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at));
      setSessions(sortedSess);
      setMessages(msgs);
      setScripts(scriptList);

      if (res.script_id != null) {
        setSelectedScriptId(res.script_id);
      }
      setLatestIssues(res.issues);
    } catch (err) {
      setTurnError(String(err));
    } finally {
      setTurnRunning(false);
      setOptimisticUser(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isTurnRunning && !building) {
        handleSend();
      }
    }
  };

  const handleScriptSaved = (newScriptId: number, issues: Issue[]) => {
    listScripts(projectId).then(setScripts).catch(() => {});
    setSelectedScriptId(newScriptId);
    setLatestIssues(issues);
  };

  const findScriptVersion = (sid: number | null) => {
    if (sid == null) return "?";
    const found = scripts.find((s) => s.id === sid);
    return found ? found.version : "?";
  };

  // Group scripts by session_id
  const sessionScripts =
    selectedSessionId != null
      ? scripts.filter((s) => s.session_id === selectedSessionId)
      : [];
  const otherScripts = scripts.filter(
    (s) => s.kind !== "clip" && (s.session_id == null || !sessions.some((sess) => sess.id === s.session_id)),
  );

  // Multi-script grouping: Group all script versions by title into distinct Script Cuts
  const scriptCuts = useMemo(() => {
    const map = new Map<string, { title: string; latestScript: ScriptSummary; versions: ScriptSummary[] }>();
    for (const s of scripts) {
      if (s.kind === "clip") continue; // clip scripts are shown as part of their clip set
      const existing = map.get(s.title);
      if (!existing) {
        map.set(s.title, { title: s.title, latestScript: s, versions: [s] });
      } else {
        existing.versions.push(s);
        if (s.version > existing.latestScript.version) {
          existing.latestScript = s;
        }
      }
    }
    for (const cut of map.values()) {
      cut.versions.sort((a, b) => b.version - a.version);
    }
    return Array.from(map.values()).sort(
      (a, b) => b.latestScript.created_at - a.latestScript.created_at
    );
  }, [scripts]);

  const activeScript = scripts.find((s) => s.id === selectedScriptId);

  // The open cut's own agent choice: Sonar writes with the agent it resolves to.
  const [cutAgents, setCutAgents] = useState<AgentSelection | null>(null);
  useEffect(() => {
    let alive = true;
    if (selectedScriptId == null) {
      setCutAgents(null);
      return;
    }
    getScript(selectedScriptId)
      .then(({ stored }) => alive && setCutAgents(stored.script.agents ?? null))
      .catch(() => alive && setCutAgents(null));
    return () => {
      alive = false;
    };
  }, [selectedScriptId]);
  const { view: agentsView } = useAgents();
  const { resolved: cutResolved } = useResolvedAgents(null, cutAgents, agentsView);
  const chatAgent = agentsView?.agents.find((a) => a.spec.id === cutResolved?.chat) ?? null;
  // The app's chat model answers unless the cut (or the app default) picked another agent.
  const chatIsAppModel = !chatAgent || chatAgent.spec.source === "chat_model";
  const chatEngineLabel = (ai: AiSettings) =>
    chatIsAppModel || !chatAgent
      ? brainSummary(ai.chat_model)
      : `${chatAgent.spec.name || chatAgent.spec.id}${cutAgents ? " (this cut)" : ""}`;

  // Robust guard against losing manual edits when switching cuts, versions, sessions, or starting fresh
  const guardUnsaved = async (): Promise<boolean> => {
    if (scriptEditorRef.current?.isDirty()) {
      const saveNow = await confirmAction(
        "You have unsaved edits in the script editor. Save them as a new version before switching? Cancel keeps you on this draft.",
        { title: "Unsaved edits", okLabel: "Save and switch", cancelLabel: "Keep editing" },
      );
      if (!saveNow) {
        return false;
      }
      const savedId = await scriptEditorRef.current.save();
      if (savedId == null) {
        await notify("Failed to save changes, so you stay on this draft.");
        return false;
      }
      const updated = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
      setScripts(updated);
    }
    return true;
  };

  // Centralized script selection with dirty checking and issues isolation
  const selectScript = async (targetScriptId: number | null, targetSessionId?: number | null, skipGuard = false) => {
    if (targetScriptId === selectedScriptId && (targetSessionId === undefined || targetSessionId === selectedSessionId)) {
      return;
    }
    if (!skipGuard && !(await guardUnsaved())) return;
    userSelectedRef.current = true;
    setSelectedScriptId(targetScriptId);
    setLatestIssues(undefined); // Clear stale issues on script transition
    setTurnError(null);
    if (targetSessionId !== undefined) {
      setSelectedSessionId(targetSessionId);
      setLiveEvents(targetSessionId != null ? (sessionEventsCache.get(targetSessionId) ?? []) : []);
      if (targetSessionId != null) {
        chatMessages(targetSessionId).then(setMessages).catch(() => setMessages([]));
      } else {
        setMessages([]);
      }
    } else if (targetScriptId != null) {
      const scr = scripts.find((s) => s.id === targetScriptId);
      if (scr?.session_id != null && sessions.some((sess) => sess.id === scr.session_id)) {
        setSelectedSessionId(scr.session_id);
        setLiveEvents(sessionEventsCache.get(scr.session_id) ?? []);
        chatMessages(scr.session_id).then(setMessages).catch(() => setMessages([]));
      } else {
        setSelectedSessionId(null);
        setLiveEvents([]);
        setMessages([]);
      }
    }
  };

  const handleDeleteVersion = async (scriptId: number) => {
    const s = scripts.find((x) => x.id === scriptId);
    if (!s) return;
    if (!(await confirmAction(`Delete version v${s.version} of "${s.title}"? It goes to Recently deleted for 30 days.`, { okLabel: "Delete", destructive: true }))) return;
    // Look again: the list may have changed while the question was open.
    const current = latest.current.scripts.find((x) => x.id === scriptId);
    if (!current) return;
    try {
      if (latest.current.selectedScriptId === scriptId) {
        scriptEditorRef.current?.resetDirty();
      }
      afterDelete(await deleteScript(scriptId));
      const updated = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
      setScripts(updated);
      const cutVersions = updated.filter((x) => x.title === current.title && x.kind !== "clip");
      const plainCuts = updated.filter((x) => x.kind !== "clip");
      if (cutVersions.length > 0) {
        void selectScript(cutVersions[0].id, cutVersions[0].session_id, true);
      } else if (plainCuts.length > 0) {
        void selectScript(plainCuts[0].id, plainCuts[0].session_id, true);
      } else {
        void selectScript(null, null, true);
      }
    } catch (err) {
      setTurnError(String(err));
    }
  };

  const clipOptions = (): ClipOptions => {
    const min_s = clampInt(clipMinS, 3, 600);
    const max_s = Math.max(min_s + 2, clampInt(clipMaxS, 5, 900));
    return {
      brief: clipBrief.trim(),
      count: clampInt(clipCount, 1, 50),
      min_s,
      max_s,
      format: clipFormat,
      video_ids: selectedVideoIds,
      // "No model — pick by score" is part of the agent choice (`pick_by_score`).
      worker_only: false,
      layout: clipLayout,
      ...(clipAgents ? { agents: clipAgents } : {}),
    };
  };
  // What the run would run into (agents that cannot be used, indexing still to do), shown before
  // Start. Only what changes the answer re-runs it.
  const clipPreflight = usePreflight(
    newCutModalOpen && newCutType === "clips" ? () => clipsPreflight(projectId, clipOptions()) : null,
    JSON.stringify([projectId, selectedVideoIds, clipAgents]),
  );

  const clipAgentProblem = agentSelectionProblem(clipAgents);

  const handleGenerateClips = async () => {
    setNewCutModalOpen(false);
    try {
      await generateClips(projectId, clipOptions());
      // Follow the run in Sonar: every phase with its last note, then the result.
      setSonarOpen(true);
    } catch (err) {
      setTurnError(String(err));
    }
  };

  const handleDeleteClipSet = async (setId: string) => {
    const set = clipSets.find((c) => c.id === setId);
    if (!set) return;
    if (
      !(await confirmAction(
        `Delete this clip set and its ${set.items.length === 1 ? "clip" : `${set.items.length} clips`}? It goes to Recently deleted for 30 days; your footage is not touched.`,
        { okLabel: "Delete", destructive: true },
      ))
    )
      return;
    // Look again: the set or the open cut may have changed while the question was open.
    const current = latest.current.clipSets.find((c) => c.id === setId);
    if (!current) return;
    try {
      if (current.items.some((i) => i.script_id === latest.current.selectedScriptId)) {
        scriptEditorRef.current?.resetDirty();
        void selectScript(null, null, true);
      }
      afterDelete(await deleteClipSet(setId));
      if (openClipSetId === setId) setOpenClipSetId(null);
      reloadClipSets();
      setScripts(await listScripts(projectId).catch(() => [] as ScriptSummary[]));
    } catch (err) {
      setTurnError(String(err));
    }
  };

  const openClipInEditor = async (scriptId: number) => {
    if (!(await guardUnsaved())) return;
    setOpenClipSetId(null);
    await selectScript(scriptId, null, true);
  };

  const handleCreateNewCut = async () => {
    if (newCutType === "clips") {
      await handleGenerateClips();
      return;
    }
    const title = newCutTitle.trim() || "Untitled cut";
    setNewCutModalOpen(false);

    if (newCutType === "blank") {
      const blankScript: Script = {
        title,
        target_duration_s: newCutTargetS,
        beats: [
          {
            id: "beat-1",
            purpose: "Opening",
            clips: [],
          },
        ],
      };
      try {
        const res = await saveScript(projectId, blankScript, null);
        const updated = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
        setScripts(updated);
        setSelectedScriptId(res.script_id);
        setSelectedSessionId(null);
      } catch (err) {
        setTurnError(String(err));
      }
    } else if (newCutType === "branch") {
      const current = scriptEditorRef.current?.getScript();
      if (!current) {
        setTurnError("Open a cut first to copy it.");
        return;
      }
      const branchedScript: Script = {
        ...current,
        title,
        target_duration_s: newCutTargetS || current.target_duration_s,
      };
      try {
        const res = await saveScript(projectId, branchedScript, null);
        const updated = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
        setScripts(updated);
        setSelectedScriptId(res.script_id);
        setSelectedSessionId(null);
      } catch (err) {
        setTurnError(String(err));
      }
    } else {
      // AI-generated cut: create a new session with targeted brief and open Sonar
      userSelectedRef.current = true;
      setSelectedSessionId(null);
      setSelectedScriptId(null);
      setMessages([]);
      setSonarOpen(true);
      setSonarTab("chat");
      const promptToUse = newCutPrompt.trim() || `Create an edit titled "${title}" targeting ${newCutTargetS} seconds.`;
      setInputMessage(`[Draft "${title}" — Target: ${newCutTargetS}s]\n${promptToUse}`);
      // The brief is ready to send; put the caret there so Enter starts the draft.
      window.setTimeout(() => composerRef.current?.focus(), 50);
    }
  };

  const handleDeleteCut = async (cutTitle: string) => {
    if (!(await confirmAction(`Delete the cut "${cutTitle}" and all its versions? It goes to Recently deleted for 30 days.`, { okLabel: "Delete", destructive: true }))) {
      return;
    }
    try {
      afterDelete(await deleteScriptCut(projectId, cutTitle));
      const updated = await listScripts(projectId).catch(() => [] as ScriptSummary[]);
      setScripts(updated);
      if (activeScript?.title === cutTitle) {
        if (updated.length > 0) {
          setSelectedScriptId(updated[0].id);
          if (updated[0].session_id) {
            setSelectedSessionId(updated[0].session_id);
          }
        } else {
          setSelectedScriptId(null);
          setSelectedSessionId(null);
        }
      }
    } catch (err) {
      setTurnError(String(err));
    }
  };

  const openNewCut = async () => {
    if (!(await guardUnsaved())) return;
    setSelectedPreset("social_30s");
    applyPreset("social_30s");
    setNewCutModalOpen(true);
  };

  const clipTasks = tasks.filter(
    (t) =>
      (t.state === "running" || t.state === "queued") &&
      ((t.kind.type === "clips" && t.kind.project_id === projectId) ||
        (t.kind.type === "export_clips" && clipSets.some((c) => c.id === (t.kind as { set_id: string }).set_id))),
  );

  // This project's runs (clip generation, indexing, faces, renders… from the app or the CLI) in the
  // Sonar thread, in time order among the messages. Chat turns are left out: they already show as
  // messages.
  const sonarRuns = projectRuns
    .filter((r) => r.kind !== "chat")
    .slice(0, SONAR_RUNS)
    .reverse();
  type ThreadItem = { at: number; msg: (typeof messages)[number]; run?: undefined } | { at: number; run: RunView; msg?: undefined };
  const threadItems: ThreadItem[] = [
    ...messages.map((msg) => ({ at: toMs(msg.created_at), msg })),
    ...sonarRuns.map((run) => ({ at: toMs(run.createdAt), run })),
  ].sort((a, b) => a.at - b.at);
  const runLinks: LinkContext = {
    openClipSet: (id) => {
      if (!clipSets.some((c) => c.id === id)) reloadClipSets();
      setOpenClipSetId(id);
    },
    openScript: (id) => void selectScript(id),
  };

  const cutsPanel = (
    <>
    {clipTasks.length > 0 && (
      <div className="clips-tasks">
        {clipTasks.map((t) => (
          <TaskCard key={t.id} task={t} compact />
        ))}
      </div>
    )}
    <CutsPanel
      clipSets={clipSets}
      onOpenClipSet={setOpenClipSetId}
      onDeleteClipSet={(id) => void handleDeleteClipSet(id)}
      cuts={scriptCuts}
      selectedScriptId={selectedScriptId}
      onSelect={(id, sid) => void selectScript(id, sid)}
      onNewCut={openNewCut}
      onDeleteCut={handleDeleteCut}
      onRestore={async (id) => {
        if (!(await guardUnsaved())) return;
        try {
          const { stored } = await getScript(id);
          const saved = await saveScript(projectId, { ...stored.script, title: stored.title }, stored.session_id);
          setScripts(await listScripts(projectId));
          await selectScript(saved.script_id, stored.session_id, true);
        } catch (error) { setTurnError(String(error)); }
      }}
      onDeleteVersion={handleDeleteVersion}
      trash={trash}
      onRestoreTrash={restoreFromTrash}
      undoItem={undoItem}
      onDismissUndo={() => setUndoItem(null)}
    />
    </>
  );

  const openClipSet = clipSets.find((c) => c.id === openClipSetId) ?? null;

  const currentMode = MODES.find((m) => m.id === mode) ?? MODES[0];
  const judgingOn = Boolean(jevReady && judging);
  const viewingSessionId = selectedSessionId ?? liveSessionId;

  // Live research, one line per step: a finished call replaces its own "started" line.
  const liveSteps = (() => {
    const steps: { label: string; detail?: string; running: boolean; tool?: string }[] = [];
    for (const evt of displayEvents) {
      if (evt.kind === "tool_started") {
        steps.push({ label: describeLiveStep(evt.tool, evt.args, videoNames), tool: evt.tool, running: true });
      } else if (evt.kind === "tool_finished") {
        const open = [...steps].reverse().find((st) => st.running && st.tool === evt.tool);
        if (open) {
          open.running = false;
          open.detail = evt.summary;
        } else {
          steps.push({ label: describeLiveStep(evt.tool, null, videoNames), tool: evt.tool, detail: evt.summary, running: false });
        }
      } else if (evt.kind === "drafting") steps.push({ label: "Writing the cut", running: true });
      else if (evt.kind === "validating") steps.push({ label: "Checking the cut", running: true });
    }
    // Only the last step can still be in progress.
    steps.forEach((st, k) => k < steps.length - 1 && (st.running = false));
    return steps;
  })();

  return (
    <div className={`nle-workspace-layout ${sonarOpen ? "sonar-visible" : ""} ${sonarOpen && sonarDocked ? "sonar-docked" : ""}`}>
      {/* 1. Cut toolbar: what is open, whether it is saved, and the two actions on it. */}
      <div className="nle-top-bar">
        <div className="nle-top-bar-left">
          <span className="nle-script-name" title={activeScript?.title}>
            {activeScript ? activeScript.title : "No cut open"}
          </span>
          {activeScript && (
            <span className="pill small version-pill" title={`Version ${activeScript.version} of this cut`}>
              v{activeScript.version}
            </span>
          )}
          {activeScript && (
            <span className="muted small nle-script-meta">
              {activeScript.beats} {activeScript.beats === 1 ? "beat" : "beats"} · {activeScript.clips}{" "}
              {activeScript.clips === 1 ? "clip" : "clips"} · {clock(activeScript.duration_s)}
            </span>
          )}
          {activeScript && (
            <span
              className={`nle-save-state${editorDirty ? " dirty" : ""}`}
              title={editorDirty ? "Save a version to keep these edits (Ctrl+S)" : "Everything is saved in this version"}
            >
              <span className="nle-save-dot" aria-hidden="true" />
              {editorDirty ? "Unsaved changes" : `Saved ${formatRelativeTime(activeScript.created_at)}`}
            </span>
          )}
        </div>

        <div className="nle-top-bar-right">
          <button
            type="button"
            className="small nle-new-cut-btn"
            onClick={openNewCut}
            title="Start another cut of this footage, e.g. a 0:30 reel next to a 3:00 recap"
          >
            <Plus size={14} strokeWidth={2} aria-hidden="true" />
            New cut
          </button>
          <button
            type="button"
            className={`small nle-sonar-toggle-btn${sonarOpen ? " active" : ""}${isTurnRunning ? " running" : ""}`}
            onClick={() => setSonarOpen((prev) => !prev)}
            aria-pressed={sonarOpen}
            title={`${sonarOpen ? "Hide" : "Show"} Sonar, the AI assistant (Ctrl+J)`}
          >
            <Radar size={16} strokeWidth={2} aria-hidden="true" />
            <span className="sonar-label">Sonar</span>
            {isTurnRunning && (
              <span className="sonar-pulse-pill">
                <span className="sonar-pulse-dot" /> Working…
              </span>
            )}
            <kbd className="nle-kbd">Ctrl+J</kbd>
          </button>
        </div>
      </div>

      {/* 2. The editor, with Sonar beside it (docked) or over it (floating) — never over the toolbar. */}
      <div className="nle-body-row">
      <main className={`nle-main-canvas ${sonarOpen && sonarDocked ? "with-docked-sonar" : ""}`}>
        {selectedScriptId != null ? (
          <ScriptEditor
            ref={scriptEditorRef}
            projectId={projectId}
            scriptId={selectedScriptId}
            sessionId={selectedSessionId}
            sessionTitle={
              selectedSessionId != null
                ? cleanTitle(sessions.find((s) => s.id === selectedSessionId)?.title)
                : undefined
            }
            latestIssues={latestIssues}
            isAiWorking={isTurnRunning}
            onScriptSaved={handleScriptSaved}
            cutsPanel={cutsPanel}
          />
        ) : (
          <div className="nle-grid nle-grid-empty">
            <aside className="nle-panel nle-left">
              <div className="nle-panel-tabs">
                <span className="nle-panel-tab active">Cuts</span>
              </div>
              <div className="nle-panel-body">{cutsPanel}</div>
            </aside>
            <section className="nle-panel nle-center nle-empty-stage">
              <div className="nle-viewer-empty">
                <div className="nle-viewer-empty-icon"><Clapperboard size={36} strokeWidth={1.5} aria-hidden="true" /></div>
                <h3>{isTurnRunning ? "Sonar is drafting this cut…" : "No cut open"}</h3>
                <p className="small muted">
                  Pick a cut on the left, or start a new one and let Sonar draft it from your footage.
                </p>
                <div className="inline" style={{ gap: "8px", justifyContent: "center" }}>
                  <button type="button" className="primary" onClick={openNewCut}>
                    <Plus size={16} strokeWidth={2} aria-hidden="true" />
                    New cut
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSonarOpen(true);
                      setSonarTab("chat");
                    }}
                  >
                    <Radar size={16} strokeWidth={2} aria-hidden="true" />
                    Open Sonar
                  </button>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      {/* 3. Sonar drawer. Toggled from the toolbar or Ctrl+J; no edge tab to cover the inspector. */}
      {sonarOpen && (
        <aside className={`nle-sonar-overlay card ${sonarDocked ? "docked" : ""}`} aria-label="Sonar assistant">
          <div className="nle-sonar-header">
            <div className="nle-sonar-header-left">
              <Radar size={16} strokeWidth={2} aria-hidden="true" />
              <span className="nle-sonar-title">Sonar</span>
            </div>

            <div className="nle-sonar-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={sonarTab === "chat"}
                className={`nle-sonar-tab ${sonarTab === "chat" ? "active" : ""}`}
                onClick={() => setSonarTab("chat")}
              >
                Chat
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={sonarTab === "history"}
                className={`nle-sonar-tab ${sonarTab === "history" ? "active" : ""}`}
                onClick={() => setSonarTab("history")}
              >
                History{sessions.length > 0 ? ` (${sessions.length})` : ""}
              </button>
            </div>

            <div className="nle-sonar-header-right">
              <button
                type="button"
                className="ghost small nle-sonar-icon-btn"
                onClick={handleNewChat}
                title="New chat"
                aria-label="New chat"
              >
                <SquarePen size={16} strokeWidth={2} aria-hidden="true" />
              </button>
              <button
                type="button"
                className="ghost small nle-sonar-icon-btn"
                onClick={() => setSonarDocked((d) => !d)}
                title={sonarDocked ? "Float over the editor" : "Dock beside the editor"}
                aria-label={sonarDocked ? "Float over the editor" : "Dock beside the editor"}
              >
                {sonarDocked ? <PictureInPicture2 size={16} strokeWidth={2} aria-hidden="true" /> : <PanelRight size={16} strokeWidth={2} aria-hidden="true" />}
              </button>
              <button
                type="button"
                className="ghost small nle-sonar-icon-btn"
                onClick={() => setSonarOpen(false)}
                title="Hide Sonar (Ctrl+J)"
                aria-label="Hide Sonar"
              >
                <X size={16} strokeWidth={2} aria-hidden="true" />
              </button>
            </div>
          </div>

          <div className="nle-sonar-body">
            {sonarTab === "history" ? (
              <div className="nle-sonar-history-view">
                <div className="nle-history-actions">
                  <button type="button" className="new-chat-btn small" onClick={handleNewChat}>
                    <SquarePen size={14} strokeWidth={2} aria-hidden="true" />
                    New chat
                  </button>
                </div>

                <div className="sidebar-section-title">Chats</div>
                <div className="sessions-list">
                  {sessions.length === 0 && liveSessionId == null && (
                    <div className="muted small">No chats yet.</div>
                  )}
                  {liveSessionId != null && !sessions.some((s) => s.id === liveSessionId) && (
                    <div
                      className={`session-item ${selectedSessionId === liveSessionId ? "active" : ""}`}
                      onClick={() => {
                        handleSelectSession(liveSessionId);
                        setSonarTab("chat");
                      }}
                    >
                      <span className="session-title">Current chat</span>
                      <span className="pill local small">Working</span>
                    </div>
                  )}
                  {sessions.map((sess) => {
                    const isSelected = sess.id === selectedSessionId;
                    return (
                      <div key={sess.id} className="session-group">
                        <div
                          className={`session-item ${isSelected ? "active" : ""}`}
                          onClick={() => {
                            handleSelectSession(sess.id);
                            setSonarTab("chat");
                          }}
                        >
                          <span className="session-title" title={cleanTitle(sess.title)}>{cleanTitle(sess.title)}</span>
                          {liveSessionId === sess.id && <span className="pill local small">Working</span>}
                          <span className="session-time muted small">
                            {formatRelativeTime(sess.updated_at || sess.created_at)}
                          </span>
                          <button
                            className="ghost small session-remove"
                            title="Remove this chat (its cuts are kept)"
                            aria-label="Remove this chat"
                            onClick={(e) => {
                              e.stopPropagation();
                              void removeSession(sess.id);
                            }}
                          >
                            <X size={14} strokeWidth={2} aria-hidden="true" />
                          </button>
                        </div>

                        {/* Versions drafted in the selected chat */}
                        {isSelected && sessionScripts.length > 0 && (
                          <div className="session-scripts-sublist">
                            {sessionScripts.map((s) => {
                              const isScriptSelected = s.id === selectedScriptId;
                              return (
                                <div
                                  key={s.id}
                                  className={`script-version-item ${isScriptSelected ? "active" : ""}`}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    void selectScript(s.id, sess.id);
                                  }}
                                >
                                  <span className="version-tag">v{s.version}</span>
                                  <span className="version-info muted small">
                                    {s.beats} {s.beats === 1 ? "beat" : "beats"} · {s.clips} {s.clips === 1 ? "clip" : "clips"} · {clock(s.duration_s)}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {otherScripts.length > 0 && (
                  <div className="other-scripts-section" style={{ marginTop: "16px" }}>
                    <div className="sidebar-section-title">Cuts made without a chat</div>
                    <div className="session-scripts-sublist standalone">
                      {otherScripts.map((s) => {
                        const isScriptSelected = s.id === selectedScriptId;
                        return (
                          <div
                            key={s.id}
                            className={`script-version-item ${isScriptSelected ? "active" : ""}`}
                            onClick={() => {
                              void selectScript(s.id);
                            }}
                            title={s.title}
                          >
                            <span className="version-tag">v{s.version}</span>
                            <span className="version-info muted small">
                              {s.title} · {clock(s.duration_s)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Chat Conversation View */
              <div className={`nle-sonar-chat-view ${optionsOpen ? "show-options" : ""}`}>
                <div className="chat-header">
                  <span className="label chat-header-title">
                    {selectedSessionId != null
                      ? cleanTitle(sessions.find((s) => s.id === selectedSessionId)?.title)
                      : "New chat"}
                  </span>
                  <span className="chat-header-actions">
                    {viewingSessionId != null && (
                      <button
                        type="button"
                        className="ghost small"
                        onClick={() => setLogOpen(true)}
                        title="Every prompt, the model’s thinking and answers, timings, and the helper’s output"
                      >
                        Full log
                      </button>
                    )}
                  </span>
                </div>

                <div className="chat-messages" ref={messagesRef}>
                  {messages.length === 0 && !optimisticUser && (
                    <div className="chat-empty">
                      <p className="chat-empty-lead">Describe the video you want.</p>
                      <p className="small muted">
                        Sonar looks through this project’s footage, picks the clips and writes a cut you can edit.
                      </p>
                      <div className="chat-example-chips" aria-label="Example requests">
                        {NEW_CHAT_EXAMPLES.map((ex) => (
                          <button
                            key={ex}
                            type="button"
                            className="chat-example-chip"
                            disabled={isTurnRunning || building}
                            onClick={() => {
                              setInputMessage(ex);
                              composerRef.current?.focus();
                            }}
                          >
                            {ex}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {threadItems.map(({ msg, run }) => run ? (
                    <div key={run.key} className="chat-message-row assistant run">
                      <RunLog run={run} ctx={runLinks} compact />
                    </div>
                  ) : (
                    <div key={msg.id} className={`chat-message-row ${msg.role}`}>
                      {msg.tool_calls && msg.tool_calls.length > 0 && (
                        <ResearchSummary calls={msg.tool_calls} names={videoNames} />
                      )}

                      {/* A tool row's text is the raw call list; the summary above replaces it. */}
                      {msg.role !== "tool" && (msg.content || (msg.images && msg.images.length > 0)) && (
                        <div className={`chat-bubble ${msg.role}`}>
                          {msg.images && msg.images.length > 0 && (
                            <div className="chat-attached-images">
                              {msg.images.map((img, idx) => (
                                <MediaImage
                                  key={idx}
                                  path={img}
                                  alt={`Attached image ${idx + 1}`}
                                  className="chat-attached-image"
                                  onClick={() => setPreviewImage(img)}
                                  title="Click to view full image"
                                />
                              ))}
                            </div>
                          )}
                          {msg.content &&
                            (msg.role === "assistant" ? (
                              <AnswerBody content={msg.content} />
                            ) : (
                              <UserText text={msg.content} />
                            ))}
                          {msg.script_id != null && (
                            <div className="bubble-script-link">
                              {selectedScriptId === msg.script_id ? (
                                <span className="bubble-viewing"><Check size={14} strokeWidth={2} aria-hidden="true" />Viewing v{findScriptVersion(msg.script_id)}</span>
                              ) : (
                                <button type="button" className="small" onClick={() => void selectScript(msg.script_id!)}>
                                  Open v{findScriptVersion(msg.script_id)}
                                </button>
                              )}
                              <button
                                  type="button"
                                  className="ghost small"
                                  title="Open this version and hide Sonar so the whole timeline shows"
                                  onClick={async () => {
                                    await selectScript(msg.script_id!);
                                    setSonarOpen(false);
                                  }}
                                >
                                  Show on timeline
                                </button>
                            </div>
                          )}
                          {msg.role === "assistant" && <>
                            <ChatAnswerStats performance={msg.performance} />
                            <time className="chat-message-time small muted" title={fullTime(msg.created_at)}
                              dateTime={new Date(msg.created_at < 1e11 ? msg.created_at * 1000 : msg.created_at).toISOString()}>
                              {new Date(msg.created_at < 1e11 ? msg.created_at * 1000 : msg.created_at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit", hour12: true })}
                            </time>
                          </>}
                        </div>
                      )}
                    </div>
                  ))}

                  {optimisticUser && !messages.some((m) => m.role === "user" && m.content === optimisticUser.text) && (
                    <div className="chat-message-row user">
                      <div className="chat-bubble user">
                        {optimisticUser.images.length > 0 && (
                          <div className="chat-attached-images">
                            {optimisticUser.images.map((img, idx) => (
                              <MediaImage
                                key={idx}
                                path={img}
                                alt={`Attached image ${idx + 1}`}
                                className="chat-attached-image"
                                onClick={() => setPreviewImage(img)}
                                title="Click to view full image"
                              />
                            ))}
                          </div>
                        )}
                        {optimisticUser.text && <UserText text={optimisticUser.text} />}
                      </div>
                    </div>
                  )}

                  {isTurnRunning && (
                    <div className="chat-message-row assistant live">
                      <div className="live-progress-container">
                        {liveSteps.length > 0 && (
                          <details className="research-steps">
                            <summary>
                              <span className={`research-dot${liveSteps[liveSteps.length - 1].running ? " on" : ""}`} />
                              <span className="research-now">
                                {liveSteps[liveSteps.length - 1].label}
                                {liveSteps[liveSteps.length - 1].running ? "…" : ""}
                              </span>
                              <span className="research-count">
                                {liveSteps.length} {liveSteps.length === 1 ? "step" : "steps"}
                              </span>
                            </summary>
                            <ol>
                              {liveSteps.map((st, k) => (
                                <li key={k} className={st.running ? "running" : ""}>
                                  <span className="research-step-icon" aria-hidden="true">{st.running ? <span className="research-dot on" /> : <Check size={12} strokeWidth={2.5} />}</span> {st.label}
                                  {st.detail && <span className="muted"> · {st.detail}</span>}
                                </li>
                              ))}
                            </ol>
                          </details>
                        )}
                        {activeChatTask && <TaskCard task={activeChatTask} compact />}
                      </div>
                    </div>
                  )}

                  {latestIssues && latestIssues.length > 0 && !isTurnRunning && (
                    <div className="issues-box small">
                      <div className="label">To check in this cut ({latestIssues.length})</div>
                      {latestIssues.map((iss, i) => (
                        <div key={i} className={`issue-row ${iss.severity === "error" ? "bad-text" : "warn"}`}>
                          <span className={`pill ${iss.severity}`}>{iss.severity}</span>
                          {iss.beat_id && <strong>Beat {iss.beat_id}</strong>}
                          {iss.clip_index != null && <span>clip {iss.clip_index + 1}:</span>}
                          <span>{iss.message}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {turnError && <div className="banner bad small">{turnError}</div>}
                </div>

                {/* Composer */}
                <div className="chat-composer" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
                  {attachedImages.length > 0 && (
                    <div className="chat-composer-attachments">
                      {attachedImages.map((img, idx) => (
                        <div key={idx} className="chat-composer-attachment-thumb">
                          <img
                            src={img}
                            alt={`Attachment ${idx + 1}`}
                            onClick={() => setPreviewImage(img)}
                            title="Click to view full image"
                          />
                          <button
                            type="button"
                            className="chat-composer-attachment-remove"
                            onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                            title="Remove image"
                            aria-label="Remove image"
                          >
                            <X size={12} strokeWidth={2.5} aria-hidden="true" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  <textarea
                    ref={composerRef}
                    rows={3}
                    aria-label="Message to Sonar"
                    placeholder={
                      isTurnRunning
                        ? "Working…"
                        : selectedSessionId != null
                          ? "Ask for changes: tighter, a different opening, steadier shots…"
                          : "Describe the cut you want…"
                    }
                    value={inputMessage}
                    disabled={isTurnRunning || building}
                    onChange={(e) => {
                      setInputMessage(e.target.value);
                      e.currentTarget.style.height = "auto";
                      e.currentTarget.style.height = `${e.currentTarget.scrollHeight}px`;
                    }}
                    onKeyDown={handleKeyDown}
                    onPaste={handlePaste}
                  />

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    style={{ display: "none" }}
                    onChange={handleFileSelect}
                  />

                  <div className="composer-toolbar">
                    <button
                      type="button"
                      className="composer-icon-btn"
                      disabled={isTurnRunning}
                      onClick={() => fileInputRef.current?.click()}
                      title="Attach an image or screenshot (or paste one)"
                      aria-label="Attach image"
                    >
                      <Paperclip size={16} strokeWidth={2} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className={`composer-options-toggle${optionsOpen ? " open" : ""}`}
                      onClick={() => setOptionsOpen((o) => !o)}
                      aria-expanded={optionsOpen}
                      title="How Sonar builds the cut, its style, and the engine"
                    >
                      <Settings2 size={14} strokeWidth={2} aria-hidden="true" className="composer-options-icon" />
                      <span className="composer-options-summary">
                        <strong>{currentMode.short}</strong>
                        {aiSettings && mode !== "jev" && <span> · {chatEngineLabel(aiSettings)}</span>}
                        {style.broll && <span> · B-roll montage</span>}
                        {judgingOn && <span> · Review on</span>}
                      </span>
                      <span className="composer-options-caret" aria-hidden="true">{optionsOpen ? <ChevronUp size={14} strokeWidth={2} /> : <ChevronDown size={14} strokeWidth={2} />}</span>
                    </button>
                  </div>

                  {optionsOpen && (
                    <div className="composer-options" role="region" aria-label="Sonar build options">
                      <button type="button" className="ghost small composer-options-done" onClick={() => setOptionsOpen(false)}>
                        <ChevronLeft size={14} strokeWidth={2} aria-hidden="true" /> Back to chat
                      </button>
                      <div className="composer-section-label" id="sonar-method-label">How to build</div>
                      <div className="composer-methods" role="radiogroup" aria-labelledby="sonar-method-label">
                        {MODES.map((m) => {
                          const needsJev = m.id !== "chat";
                          const blocked = needsJev && (jevReady === false || style.broll);
                          return (
                            <button
                              key={m.id}
                              type="button"
                              role="radio"
                              aria-checked={mode === m.id}
                              className={`composer-method${mode === m.id ? " on" : ""}${blocked ? " blocked" : ""}`}
                              disabled={isTurnRunning || (needsJev && style.broll)}
                              onClick={() => setMode(m.id)}
                            >
                              <span className="composer-method-radio" aria-hidden="true" />
                              <span className="composer-method-text">
                                <span className="composer-method-name">{m.label}</span>
                                <span className="composer-method-hint">
                                  {needsJev && style.broll
                                    ? "Not for B-roll montages: this method builds from interview quotes."
                                    : needsJev && jevReady === false
                                      ? "Requires quote-builder setup."
                                      : m.hint}
                                </span>
                              </span>
                            </button>
                          );
                        })}
                      </div>

                      <div className="composer-section-label">Style</div>
                      <div className="composer-switches">
                        <Switch
                          label="B-roll montage"
                          hint="Pictures only, on a theme: no interviews, no voice-over."
                          checked={style.broll}
                          disabled={isTurnRunning}
                          onChange={(on) => setStyle((s) => ({ ...s, broll: on }))}
                        />
                        {style.broll && (
                          <div className="composer-audio" role="radiogroup" aria-label="Montage sound">
                            {(
                              [
                                [true, "Natural sound", "Each clip plays its own ambient sound; anyone speaking is muted."],
                                [false, "Silent", "Every clip is muted, ready for music."],
                              ] as const
                            ).map(([natural, label, hint]) => (
                              <button
                                key={label}
                                type="button"
                                role="radio"
                                aria-checked={style.natural_sound === natural}
                                className={`composer-sound${style.natural_sound === natural ? " on" : ""}`}
                                disabled={isTurnRunning}
                                title={hint}
                                onClick={() => setStyle((s) => ({ ...s, natural_sound: natural }))}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        )}
                        <Switch
                          label="Editorial review"
                          hint={
                            jevReady === false
                              ? "Requires quote-builder setup."
                              : "Checks shots against the voice, opening and ending, then suggests fixes."
                          }
                          checked={judgingOn}
                          disabled={jevReady === false || judging === null}
                          onChange={async (next) => {
                            setJudging(next);
                            try {
                              const ai = await setAiSettings({ jev: { judge: next } });
                              setJudging(ai.jev.judge);
                            } catch (e) {
                              setJudging(!next);
                              setTurnError(String(e));
                            }
                          }}
                        />
                      </div>

                      {aiSettings && (
                        <details className="composer-advanced">
                          <summary>
                            <span>Advanced: engine</span>
                            <span className="muted">{chatEngineLabel(aiSettings)}</span>
                          </summary>
                          {!chatIsAppModel && chatAgent && (
                            <p className="composer-engine-hint small muted">
                              This cut writes with the agent “{chatAgent.spec.name || chatAgent.spec.id}” ({chatAgent.summary}),
                              chosen in the cut&apos;s Inspector › Agents{cutAgents ? "" : " or Models › Agents"}. The engine
                              below is the app&apos;s script chat model.
                            </p>
                          )}
                          <div className="composer-model-bar">
                            <div className="composer-backend-pills" role="radiogroup" aria-label="Where the AI runs">
                              {(["auto", "local", "server", "cli"] as const).map((b) => (
                                <button
                                  key={b}
                                  type="button"
                                  role="radio"
                                  aria-checked={aiSettings.chat_model.backend === b}
                                  className={`backend-pill${aiSettings.chat_model.backend === b ? " active" : ""}`}
                                  disabled={isTurnRunning}
                                  title={ENGINE_HINTS[b]}
                                  onClick={() => handleUpdateChatModel({ backend: b })}
                                >
                                  {b === "auto" ? "Auto" : b === "local" ? "Local" : b === "server" ? "Server" : "CLI"}
                                </button>
                              ))}
                            </div>
                            <p className="composer-engine-hint small muted">{ENGINE_HINTS[aiSettings.chat_model.backend]}</p>

                            <div className="composer-model-controls">
                              {aiSettings.chat_model.backend === "local" && (
                                <>
                                  <label className="composer-field">Model
                                    <select
                                      className="composer-model-select"
                                      value={aiSettings.chat_model.local_model}
                                      disabled={isTurnRunning}
                                      onChange={(e) => handleUpdateChatModel({ local_model: e.target.value })}
                                    >
                                      {!availableModels.some((m) => m.entry.id === aiSettings.chat_model.local_model) && (
                                        <option value={aiSettings.chat_model.local_model}>
                                          {aiSettings.chat_model.local_model}
                                        </option>
                                      )}
                                      {availableModels.map((m) => (
                                        <option key={m.entry.id} value={m.entry.id}>
                                          {m.external?.name ?? m.entry.id}{m.external?.quant ? ` · ${m.external.quant}` : ""} {m.installed_path ? "" : "(not downloaded)"}
                                        </option>
                                      ))}
                                    </select>
                                  </label>

                                  <label className="composer-think-label" title="Let the model reason before it answers: slower, often better">
                                    <input
                                      type="checkbox"
                                      checked={aiSettings.chat_model.think}
                                      disabled={isTurnRunning}
                                      onChange={(e) => handleUpdateChatModel({ think: e.target.checked })}
                                    />
                                    Think before answering
                                  </label>
                                </>
                              )}

                              {aiSettings.chat_model.backend === "cli" && (
                                <>
                                  <label className="composer-field">Tool
                                    <select
                                      className="composer-model-select"
                                      value={aiSettings.chat_model.cli.tool}
                                      disabled={isTurnRunning}
                                      onChange={(e) => handleUpdateChatModel({ cli: { tool: e.target.value } })}
                                    >
                                      <option value="claude">Claude</option>
                                      <option value="agy">Agy</option>
                                      <option value="opencode">OpenCode</option>
                                      <option value="codex">Codex</option>
                                    </select>
                                  </label>

                                  <label className="composer-field">Model
                                    <ModelSelect kind="cli" tool={aiSettings.chat_model.cli.tool} value={aiSettings.chat_model.cli.model} disabled={isTurnRunning} onChange={(model) => handleUpdateChatModel({ cli: { model } })} />
                                  </label>

                                  <span className={aiSettings.chat_model.cli.installed ? "muted small" : "bad-text small"}>
                                    {aiSettings.chat_model.cli.installed
                                      ? "Installed"
                                      : `${CLI_NAMES[aiSettings.chat_model.cli.tool] ?? aiSettings.chat_model.cli.tool} isn't installed (not found in PATH)`}
                                  </span>
                                </>
                              )}

                              {aiSettings.chat_model.backend === "server" && (
                                <>
                                  <label className="composer-field">Server address
                                    <input
                                      type="text"
                                      className="composer-model-input"
                                      defaultValue={aiSettings.chat_model.url}
                                      placeholder="http://127.0.0.1:8089"
                                      disabled={isTurnRunning}
                                      onBlur={(e) => handleUpdateChatModel({ url: e.target.value })}
                                    />
                                  </label>
                                  <label className="composer-field">Model
                                    <ModelSelect kind="server" url={aiSettings.chat_model.url} value={aiSettings.chat_model.model} placeholder="Server default" disabled={isTurnRunning} onChange={(model) => handleUpdateChatModel({ model })} />
                                  </label>
                                </>
                              )}

                              <label
                                className="composer-think-label"
                                title="The most rounds of lookups (searches, transcripts, pictures) the AI may make while researching. Leave it empty for no cap: it stops when it has what it needs, when you stop it, or when it keeps asking for the same thing."
                              >
                                Research limit
                                <input
                                  type="number"
                                  className="composer-budget-input"
                                  min={1}
                                  step={1}
                                  placeholder="Unlimited"
                                  value={researchLimit === 0 ? "" : researchLimit}
                                  disabled={isTurnRunning}
                                  onChange={(e) => {
                                    const raw = e.currentTarget.value;
                                    setResearchLimit(raw === "" ? 0 : Number(raw));
                                    if (researchLimitTimer.current) clearTimeout(researchLimitTimer.current);
                                    researchLimitTimer.current = setTimeout(() => saveResearchLimit(raw), 400);
                                  }}
                                  onKeyDown={(e) => { if (e.key === "Enter") saveResearchLimit(e.currentTarget.value); }}
                                  onBlur={(e) => saveResearchLimit(e.currentTarget.value)}
                                />
                                rounds · empty = unlimited
                              </label>
                            </div>
                          </div>
                        </details>
                      )}
                    </div>
                  )}

                  <div className="composer-go">
                    <p className="composer-hint small muted">
                      {style.broll
                        ? `A montage on the theme you describe. ${
                            style.natural_sound ? "Clips keep their natural sound." : "Clips are muted, ready for music."
                          }`
                        : jevReady === false && mode !== "chat"
                          ? <><span>{JEV_BLOCKED}</span> <button type="button" className="ghost small sonar-settings-link" onClick={() => window.dispatchEvent(new CustomEvent("ghostreel:navigate", { detail: { kind: "settings" } }))}>Open Settings</button></>
                          : currentMode.hint}
                    </p>
                    <button
                      type="button"
                      className="primary"
                      disabled={
                        isTurnRunning ||
                        building ||
                        (!inputMessage.trim() && attachedImages.length === 0) ||
                        (mode !== "chat" && jevReady === false)
                      }
                      title={mode !== "chat" && jevReady === false ? JEV_BLOCKED : "Send (Enter)"}
                      onClick={handleSend}
                    >
                      {building ? "Choosing…" : isTurnRunning ? "Working…" : currentMode.action}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>
      )}
      </div>

      {logOpen && viewingSessionId != null && (
        <ChatLogModal sessionId={viewingSessionId} onClose={() => setLogOpen(false)} />
      )}
      {/* Lightbox for attached images */}
      {previewImage && (
        <div className="chat-image-modal-overlay" onClick={() => setPreviewImage(null)}>
          <div className="chat-image-modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="chat-image-modal-close"
              onClick={() => setPreviewImage(null)}
              title="Close"
              aria-label="Close"
            >
              <X size={16} strokeWidth={2} aria-hidden="true" />
            </button>
            <MediaImage path={previewImage} alt="Expanded preview" />
          </div>
        </div>
      )}

      {/* 4. New cut dialog: how to start first; presets only for a Sonar draft. */}
      {newCutModalOpen && (
        <Modal onClose={closeNewCutModal} className="nle-new-cut" labelledBy="new-cut-title">
            <div className="nle-modal-header">
              <h3 id="new-cut-title">New cut</h3>
              <button
                type="button"
                className="ghost small nle-sonar-icon-btn"
                onClick={() => setNewCutModalOpen(false)}
                title="Close (Esc)"
                aria-label="Close"
              >
                <X size={16} strokeWidth={2} aria-hidden="true" />
              </button>
            </div>

            <p className="small muted nle-modal-intro">
              A cut is one edit of this footage, with its own versions and Sonar chat. Make as many as you
              like: a 0:30 reel next to a 15:00 documentary.
            </p>

            <div className="nle-field-label" id="new-cut-method">How to start</div>
            <div className="nle-method-cards" role="radiogroup" aria-labelledby="new-cut-method">
              {(
                [
                  ["ai", Sparkles, "Draft with Sonar", "Describe it; Sonar picks the clips and writes the first version."],
                  ["blank", FilePlus, "Blank", "One empty beat. Build it yourself from the media bin."],
                  [
                    "branch",
                    Copy,
                    "Copy current cut",
                    activeScript ? `Start from “${activeScript.title}” and change it from there.` : "Open a cut first to copy it.",
                  ],
                  ["clips", Scissors, "Clips", "Pick short self-contained clips from the footage for social media."],
                ] as const
              ).map(([id, MethodIcon, label, hint]) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={newCutType === id}
                  className={`nle-method-card${newCutType === id ? " on" : ""}`}
                  disabled={id === "branch" && !activeScript}
                  onClick={() => chooseCutMethod(id)}
                >
                  <MethodIcon size={16} strokeWidth={2} aria-hidden="true" className="nle-method-icon" />
                  <span className="composer-method-text">
                    <span className="composer-method-name">{label}</span>
                    <span className="composer-method-hint">{hint}</span>
                  </span>
                </button>
              ))}
            </div>

            {newCutType === "ai" && (
              <>
                <div className="nle-field-label" id="new-cut-presets">Start from a preset</div>
                <div className="nle-preset-row" role="group" aria-labelledby="new-cut-presets">
                  {CUT_PRESETS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className={`nle-preset-chip${selectedPreset === p.id ? " on" : ""}`}
                      aria-pressed={selectedPreset === p.id}
                      onClick={(e) => { e.currentTarget.focus(); applyPreset(p.id); }}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </>
            )}

            {newCutType === "clips" && (
              <div className="clips-form">
                <label className="nle-field">
                  <span className="nle-field-label">Brief (optional)</span>
                  <textarea
                    className="nle-brief"
                    rows={3}
                    autoFocus
                    value={clipBrief}
                    onChange={(e) => setClipBrief(e.target.value)}
                    placeholder="What the clips should be about; leave empty for the strongest moments"
                  />
                </label>
                <div className="nle-modal-row">
                  <label className="nle-field">
                    <span className="nle-field-label">Count</span>
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={clipCount}
                      onChange={(e) => setClipCount(Number(e.target.value))}
                    />
                  </label>
                  <label className="nle-field">
                    <span className="nle-field-label">Shortest (s)</span>
                    <input
                      type="number"
                      min={1}
                      max={600}
                      value={clipMinS}
                      onChange={(e) => setClipMinS(Number(e.target.value))}
                    />
                  </label>
                  <label className="nle-field">
                    <span className="nle-field-label">Longest (s)</span>
                    <input
                      type="number"
                      min={1}
                      max={600}
                      value={clipMaxS}
                      onChange={(e) => setClipMaxS(Number(e.target.value))}
                    />
                  </label>
                </div>
                {/* Chips take focus on click (WebKit leaves it on the previous one), so the focus
                    ring never sits on a chip other than the selected one. */}
                <div className="nle-field-label" id="clips-format">Format</div>
                <div className="nle-preset-row" role="group" aria-labelledby="clips-format">
                  {CLIP_FORMATS.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      className={`nle-preset-chip${clipFormat === f.id ? " on" : ""}`}
                      aria-pressed={clipFormat === f.id}
                      onClick={(e) => { e.currentTarget.focus(); setClipFormat(f.id); }}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
                {clipFormat !== "original" && (
                  <>
                    <div className="nle-field-label" id="clips-layout">Framing</div>
                    <div className="nle-preset-row" role="group" aria-labelledby="clips-layout">
                      {([
                        ["follow_speaker", "Follow the speaker"],
                        ["stacked", "Speaker and listener stacked"],
                      ] as const).map(([id, label]) => (
                        <button
                          key={id}
                          type="button"
                          className={`nle-preset-chip${clipLayout === id ? " on" : ""}`}
                          aria-pressed={clipLayout === id}
                          onClick={(e) => { e.currentTarget.focus(); setClipLayout(id); }}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </>
                )}
                <div className="small clips-videos">
                  <span className="nle-field-label">Videos</span>
                  {selectedVideoIds.length > 0
                    ? `${selectedVideoIds.length} selected ${selectedVideoIds.length === 1 ? "video" : "videos"}`
                    : `All ${videoCount} ${videoCount === 1 ? "video" : "videos"} in the library`}
                </div>
                <div className="clips-engines small">
                  <span className="nle-field-label">Agents</span>
                  <AgentPicker
                    level="run"
                    value={clipAgents}
                    onChange={setClipAgents}
                    target={{ count: clampInt(clipCount, 1, 50), format: clipFormat }}
                    onManage={() => setNewCutModalOpen(false)}
                  />
                </div>
                {clipPreflight.checking && clipPreflight.checks.length === 0 ? (
                  <div className="small muted">Checking the agents and the project…</div>
                ) : (
                  <PreflightList checks={clipPreflight.checks} />
                )}
              </div>
            )}

            {newCutType !== "clips" && (
            <div className="nle-modal-row">
              <label className="nle-field nle-field-grow">
                <span className="nle-field-label">Name</span>
                <input
                  type="text"
                  value={newCutTitle}
                  autoFocus
                  onChange={(e) => {
                    setNewCutTitle(e.target.value);
                  }}
                  placeholder="e.g. Keynote highlights"
                />
              </label>
              <label className="nle-field nle-field-target">
                <span className="nle-field-label">Target length</span>
                <span className="nle-target-input">
                  <DurationField value={newCutTargetS} min={5} max={7200}
                    onChange={(value) => { if (value != null) setNewCutTargetS(value); setSelectedPreset(null); }} />
                  <span className="small muted">m:ss</span>
                </span>
                <span className="nle-duration-presets" role="group" aria-label="Target duration presets">
                  {[[30, "0:30"], [60, "1:00"], [180, "3:00"], [900, "15:00"]].map(([seconds, label]) => (
                    <button key={seconds} type="button" className="ghost small" aria-pressed={newCutTargetS === seconds}
                      onClick={() => { setNewCutTargetS(Number(seconds)); setSelectedPreset(null); }}>{label}</button>
                  ))}
                </span>
              </label>
            </div>
            )}

            {newCutType === "ai" && (
              <label className="nle-field">
                <span className="nle-field-label">Brief for Sonar</span>
                <textarea
                  className="nle-brief"
                  rows={5}
                  value={newCutPrompt}
                  onChange={(e) => setNewCutPrompt(e.target.value)}
                  placeholder="The tone, the pacing, who should speak, what it should leave people with…"
                />
              </label>
            )}

            <div className="nle-modal-actions">
              {newCutType === "clips" && clipAgentProblem && (
                <span className="small warn-text nle-modal-actions-reason">{clipAgentProblem}</span>
              )}
              <button type="button" onClick={() => setNewCutModalOpen(false)}>
                Cancel
              </button>
              <button
                type="button"
                className="primary"
                title={newCutType === "clips" && clipAgentProblem ? clipAgentProblem : undefined}
                disabled={
                  newCutType === "clips"
                    ? !(clipCount >= 1 && clipMinS >= 1 && clipMaxS >= 1) || clipPreflight.blocked || clipAgentProblem != null
                    : !(newCutTargetS >= 5) || (newCutType === "branch" && !activeScript)
                }
                onClick={handleCreateNewCut}
              >
                {newCutType === "ai"
                  ? "Start with Sonar"
                  : newCutType === "branch"
                    ? "Copy cut"
                    : newCutType === "clips"
                      ? "Generate clips"
                      : "Create blank cut"}
              </button>
            </div>
        </Modal>
      )}

      {openClipSet && (
        <ClipSetGrid
          set={openClipSet}
          onChange={(next) => setClipSets((prev) => prev.map((c) => (c.id === next.id ? next : c)))}
          onClose={() => setOpenClipSetId(null)}
          onOpenInEditor={(id) => void openClipInEditor(id)}
        />
      )}
    </div>
  );
}
