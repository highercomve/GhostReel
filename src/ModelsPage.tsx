import { useCallback, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Download, Trash2, X } from "lucide-react";
import {
  cancelTask,
  enqueueModelDownload,
  etaText,
  getAiSettings,
  humanSize,
  modelsStatus,
  openModelsDir,
  probeBackends,
  removeModel,
  setAiSettings,
  setWhisperModel,
  type AiSettings,
  type AiSettingsPatch,
  type BackendsResolution,
  type Backend,
  type CliSettings,
  type ModelStatus,
  type ModelsStatusView,
  type Resolution,
  type Task,
  type VisionSettings,
  type VisionSettingsPatch,
  testCliAgent,
} from "./api";
import { onEvent } from "./events";
import { useQueue } from "./useQueue";
import ModelSelect from "./editor/ModelSelect";

// ─── helpers ─────────────────────────────────────────────────────────────────

/** The chosen model, for "Using: <model> · <where>" summaries. */
function probeLabel(r: Resolution, model?: string): { text: string; cls: string; title?: string } {
  const raw = r.reason.trim();
  if (r.target === "server" && r.probe) {
    const m = r.probe.model || model;
    const host = (() => {
      try {
        return new URL(r.probe.url).host;
      } catch {
        return r.probe.url;
      }
    })();
    return { text: `Using: ${m || "the server"} · server @ ${host}`, cls: "probe-ok" };
  }
  if (r.target === "local") {
    // "backend = local" and friends are raw config echoes, not copy — the summary says it in words.
    const note = /^backend\s*=/.test(raw) || raw === "server not probed" ? undefined : raw;
    // Without a known model the "where" is the whole answer; "this computer · this computer" reads
    // as a bug. With one, name the model first (the same pattern the server route uses).
    const text = model ? `Using: ${model} · this computer` : "Using: this computer";
    return { text, cls: "probe-ok", title: note };
  }
  if (r.target === "unavailable") {
    const but = raw.match(/^backend\s*=\s*server\s+but\s+([\s\S]+)$/);
    return { text: but ? `The server can't do this job: ${but[1]}` : raw || "Not available", cls: "probe-bad" };
  }
  return { text: raw, cls: "" };
}

/** Configured CLI tool ids stay plain in settings; the UI once names each tool. */
const CLI_TOOL_LABELS: Record<string, string> = {
  claude: "Claude",
  agy: "agy",
  opencode: "OpenCode",
  codex: "Codex",
};

type Summary = { text: string; cls: string; title?: string };

/** "Using:" for a CLI agent selection — derived from the card's own Tool/Model controls,
 *  so the summary can never contradict the selected route. */
function cliSummary(cli: CliSettings): Summary {
  const tool = CLI_TOOL_LABELS[cli.tool] ?? cli.tool;
  const model = cli.model || "the tool's default model";
  return cli.installed
    ? { text: `Using: ${tool} (CLI) · ${model}`, cls: "probe-ok" }
    : { text: `Using: ${tool} (CLI) · ${model} — not installed yet`, cls: "probe-bad" };
}

/**
 * The "Using:" line for one section, derived from the visible selection: a CLI backend names the
 * tool; auto that fell back to this computer says so instead of silently claiming the server route.
 */
function sectionSummary(r: Resolution, opts: { model?: string; cli?: CliSettings } = {}): Summary {
  if (r.backend === "cli" && opts.cli) return cliSummary(opts.cli);
  const item = probeLabel(r, opts.model);
  if (r.backend === "auto" && r.target === "local" && /server not usable/.test(r.reason)) {
    return { ...item, text: `${item.text} · (auto — the server was not usable)` };
  }
  return item;
}

// ─── segmented control ────────────────────────────────────────────────────────

interface SegmentedProps {
  value: Backend;
  onChange: (v: Backend) => void;
  /** Frame descriptions and the script chat can also run a coding-agent CLI. */
  withCli?: boolean;
}
/** Which coding-agent CLI answers for this capability, and a button to try it once. */
function CliAgentFields({
  cfg,
  capability,
  onPatch,
}: {
  cfg: VisionSettings;
  capability: "vision" | "chat_model";
  onPatch: (p: VisionSettingsPatch) => void;
}) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const perFrame = capability === "vision";
  return (
    <>
      <div className="settings-fields">
        <div className="settings-field">
          <label>Tool</label>
          <select
            value={cfg.cli.tool}
            title="agy is short for the Agenty CLI; the other tools run under their own accounts or bills"
            onChange={(e) => onPatch({ cli: { tool: e.currentTarget.value } })}
          >
            <option value="claude">Claude</option>
            <option value="agy">agy</option>
            <option value="opencode">OpenCode</option>
            <option value="codex">Codex</option>
          </select>
          <span className={cfg.cli.installed ? "good-text small" : "bad-text small"}>
            {cfg.cli.installed ? "Found on this computer." : "Not found — install it or set a full path"}
          </span>
        </div>
        <div className="settings-field">
          <label>Model</label>
          <ModelSelect
            kind="cli"
            tool={cfg.cli.tool}
            value={cfg.cli.model}
            placeholder="— the tool's default —"
            onChange={(model) => onPatch({ cli: { model } })}
          />
        </div>
        {perFrame && (
          <div className="settings-field">
            <label>At a time</label>
            <input
              type="number"
              min={1}
              max={8}
              style={{ width: "4em" }}
              defaultValue={cfg.cli.concurrency}
              onChange={(e) => {
                const v = Number(e.currentTarget.value);
                if (v >= 1 && v <= 8) onPatch({ cli: { concurrency: v } });
              }}
            />
            <span className="muted small">parallel calls while indexing</span>
          </div>
        )}
      </div>
      <p className="muted small">
        {perFrame
          ? "One call per keyframe. Claude bills per call (about $0.04 per frame here, so a 1,500-frame library runs into tens of dollars); agy (the Agenty CLI) and OpenCode instead run under their own accounts, so hours of indexing cost nothing extra."
          : "A handful of calls per script — far cheaper than describing every frame. Claude bills per call; agy (the Agenty CLI) and OpenCode run under their own accounts."}
      </p>
      <div className="row">
        <button
          className="ghost small"
          disabled={testing}
          onClick={async () => {
            setTesting(true);
            setResult(null);
            try {
              setResult(await testCliAgent(capability));
            } catch (e) {
              setResult(String(e));
            } finally {
              setTesting(false);
            }
          }}
        >
          {testing ? "Testing…" : "Test one call"}
        </button>
      </div>
      {result && <pre className="cli-test-result">{result}</pre>}
    </>
  );
}

/** Context window, KV cache and flash attention of one local helper. */
function LocalRuntimeFields({
  cfg,
  onPatch,
  thinkHint,
  mlx = false,
}: {
  cfg: VisionSettings;
  onPatch: (p: VisionSettingsPatch) => void;
  /** Why thinking is or isn't worth its cost for this capability. */
  thinkHint: string;
  mlx?: boolean;
}) {
  return (
    <div className="settings-fields">
      <div className="settings-field">
        <label>Context window</label>
        <input
          type="number"
          min={2048}
          max={131072}
          step={2048}
          style={{ width: "7em" }}
          defaultValue={cfg.ctx_tokens}
          onChange={(e) => {
            const v = Number(e.currentTarget.value);
            if (v >= 2048 && v <= 131072) onPatch({ ctx_tokens: v });
          }}
        />
        <span className="muted small">tokens · more room for the model, more VRAM</span>
      </div>
      <div className="settings-field">
        <label>KV cache</label>
        <select value={cfg.kv_cache} onChange={(e) => onPatch({ kv_cache: e.currentTarget.value })}>
          <option value="q4_0">{mlx ? "4-bit — lower memory" : "q4_0 — ~4× the context per GB"}</option>
          <option value="q8_0">{mlx ? "8-bit — balanced" : "q8_0 — balanced"}</option>
          <option value="f16">{mlx ? "Full precision — higher memory" : "f16 — best quality, 4× the VRAM"}</option>
        </select>
      </div>
      {!mlx && <div className="settings-field">
        <label>Flash attention</label>
        <select value={cfg.flash_attn} onChange={(e) => onPatch({ flash_attn: e.currentTarget.value })}>
          <option value="auto">auto</option>
          <option value="on">on</option>
          <option value="off">off</option>
        </select>
      </div>}
      <div className="settings-field check">
        <label>Think first</label>
        <input type="checkbox" checked={cfg.think} onChange={(e) => onPatch({ think: e.currentTarget.checked })} />
        <span className="muted small">
          {mlx ? "Used for free-text replies; structured JSON replies skip reasoning." : thinkHint}
        </span>
      </div>
    </div>
  );
}

function BackendSegmented({ value, onChange, withCli = false }: SegmentedProps) {
  const opts: { label: string; v: Backend }[] = [
    { label: "Auto", v: "auto" },
    { label: "This computer", v: "local" },
    { label: "Server", v: "server" },
    ...(withCli ? [{ label: "CLI agent", v: "cli" as Backend }] : []),
  ];
  return (
    <div className="segmented-control">
      {opts.map(({ label, v }) => (
        <button key={v} className={value === v ? "active" : ""} onClick={() => onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

// ─── model list (GhostPen-style) ─────────────────────────────────────────────

/** The app a model file outside GhostReel's folder was found in. */
const externalOrigin = (path: string): string => {
  const lower = path.toLowerCase();
  if (lower.includes("ghostpen")) return "GhostPen";
  if (lower.includes("lmstudio") || lower.includes("lm-studio") || lower.includes("lm studio"))
    return "LM Studio";
  if (lower.includes("ollama")) return "Ollama";
  return "another app";
};

/** A catalog note as a sentence: "recommended — …" becomes the Recommended badge plus the rest. */
const modelQuant = (m: ModelStatus): string | undefined =>
  m.external?.quant ?? m.entry.file_name.match(/(?:^|[-.])((?:i?q|pq)\d[\w]*|bf16|f16|f32)(?=[-.]|$)/i)?.[1]?.toLowerCase();

const noteInfo = (m: ModelStatus): { recommended: boolean; text: string } => {
  let note = m.entry.note.trim();
  let recommended = false;
  const found = note.match(/^recommended\s*[—–-]\s*(.*)$/i);
  if (found) {
    recommended = true;
    note = found[1];
  }
  return { recommended, text: note.charAt(0).toUpperCase() + note.slice(1) };
};

/** A 1-5 score as five small bar segments. */
function ScoreMeter({ label, value }: { label: string; value: number }) {
  return (
    <span className="mp-meter-item">
      {label}
      <span
        className="mp-meter"
        role="img"
        aria-label={`${label} ${value} of 5`}
        title={`${label}: ${value} of 5`}
      >
        {[1, 2, 3, 4, 5].map((i) => (
          <i key={i} className={i <= value ? "on" : ""} />
        ))}
      </span>
    </span>
  );
}

/** One model row: name, size, badges, meters, note — one action on the right. */
function ModelRow(props: {
  m: ModelStatus;
  /** A running or queued download for this model, if any. */
  task?: Task;
  /** The row is the chosen model: an accent border instead of radio buttons. */
  selected?: boolean;
  onUse?: () => void;
  sizeText: string;
  note: string;
  recommended?: boolean;
  readsImages?: boolean;
  /** A quiet badge such as "English-only". */
  dimBadge?: string;
  /** Muted extra line, e.g. "From LM Studio, reused". */
  origin?: string;
  onDownload: () => void;
  onRemove: () => void;
}) {
  const { m, task } = props;
  const [confirmRemove, setConfirmRemove] = useState(false);
  const pct = task && task.progress ? Math.min(100, Math.round(task.progress.fraction * 100)) : 0;
  const partialPct = m.partial_bytes && m.entry.size_bytes
    ? Math.round((m.partial_bytes / m.entry.size_bytes) * 100)
    : 0;
  const eta = task && task.progress?.eta_secs != null ? etaText(task.progress.eta_secs) : null;
  return (
    <div className={`mp-row${props.selected ? " active" : ""}`} title={m.installed_path ?? undefined}>
      <div className="mp-info">
        <div className="mp-name">
          <span className="mp-title">{m.external?.name ?? m.entry.id}</span>
          <span className="muted small">{props.sizeText}</span>
          {/* The row's real state, next to the matching action on the right. */}
          {task ? (
            <span className="mp-badge busy">{task.state === "queued" ? "Queued" : "Downloading"}</span>
          ) : m.installed_path ? (
            props.selected ? null : (
              <span className="mp-badge good">Downloaded</span>
            )
          ) : m.partial_bytes ? (
            <span className="mp-badge dim">Partial · {partialPct}%</span>
          ) : (
            <span className="mp-badge dim">Not downloaded</span>
          )}
          {props.recommended && <span className="mp-badge rec">Recommended</span>}
          {props.readsImages && <span className="mp-badge good">Reads images</span>}
          {props.dimBadge && <span className="mp-badge dim">{props.dimBadge}</span>}
        </div>
        <div className="mp-meta">
          {!m.external && <ScoreMeter label="Speed" value={m.entry.speed} />}
          {!m.external && <ScoreMeter label="Quality" value={m.entry.accuracy} />}
          <span className="mp-note">{props.note}</span>
        </div>
        {props.origin && <div className="muted small">{props.origin}</div>}
        {task && (
          <div className="mp-progress">
            <div className="mp-bar">
              <div style={{ width: `${task.state === "running" ? Math.max(4, pct) : 0}%` }} />
            </div>
            <span className="muted small">
              {task.state === "queued"
                ? "Queued, waiting…"
                : `Downloading ${pct}%${eta ? ` · about ${eta} left` : ""}`}
            </span>
          </div>
        )}
      </div>
      <div className="mp-actions">
        {task ? (
          <button
            className="small"
            onClick={() => cancelTask(task.id)}
            title="Cancel the download (keeps the partial file)"
          >
            <X size={14} strokeWidth={2} aria-hidden /> Cancel
          </button>
        ) : m.installed_path ? (
          <>
            {props.selected ? (
              <span className="mp-state" title="This model is chosen for this section">
                <Check size={15} strokeWidth={2} aria-hidden /> In use
              </span>
            ) : props.onUse ? (
              <button className="small" onClick={props.onUse}>
                Use
              </button>
            ) : null}
            {m.in_own_dir && !m.external &&
              (confirmRemove ? (
                <>
                  <button className="ghost small" onClick={() => setConfirmRemove(false)}>
                    Keep
                  </button>
                  <button
                    className="danger small"
                    onClick={() => {
                      setConfirmRemove(false);
                      props.onRemove();
                    }}
                  >
                    <Trash2 size={14} strokeWidth={2} aria-hidden /> Remove?
                  </button>
                </>
              ) : (
                <button
                  className="ghost small danger"
                  onClick={() => setConfirmRemove(true)}
                  title="Delete from GhostReel's models folder"
                >
                  <Trash2 size={14} strokeWidth={2} aria-hidden /> Remove
                </button>
              ))}
          </>
        ) : m.partial_bytes ? (
          <>
            <button className="small" onClick={props.onDownload} title="Resume download">
              <Download size={14} strokeWidth={2} aria-hidden /> Resume ({partialPct}%)
            </button>
            <button
              className="ghost small danger"
              onClick={props.onRemove}
              title="Discard the partial file"
            >
              <Trash2 size={14} strokeWidth={2} aria-hidden /> Clear
            </button>
          </>
        ) : (
          <button className="small" onClick={props.onDownload} title="Download this model">
            <Download size={14} strokeWidth={2} aria-hidden /> Download
          </button>
        )}
      </div>
    </div>
  );
}

/** "Available models" card: one row per model, plus a collapsible list of models found in other apps. */
function ModelListCard(props: {
  main: ModelStatus[];
  /** Models installed in another app's folder ("Found in LM Studio"). */
  others: ModelStatus[];
  /** Select this model id for the section; omit when the section has no choice (embeddings, server mode). */
  onUse?: (id: string) => void;
  selectedId?: string;
  /** Rows show the green "Reads images" badge (frame descriptions list). */
  readsImages?: boolean;
  /** The download task for a model id, if one is running or queued. */
  taskFor: (id: string) => Task | undefined;
  onDownload: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  const { main, others } = props;
  const primary = main
    .filter((m) => Boolean(m.installed_path) || m.entry.id === props.selectedId)
    .sort((a, b) => Number(b.entry.id === props.selectedId) - Number(a.entry.id === props.selectedId));
  const catalog = main.filter((m) => !primary.includes(m));
  // Nothing installed yet: offer the downloadable rows directly instead of an empty
  // "Available models" body above a collapsed catalog. Sources that need no file of their own
  // (a server, a CLI tool) show only the other-app rows, or nothing at all.
  const listed = primary.length > 0 ? primary : catalog;
  const rest = primary.length > 0 ? catalog : [];
  const apps =
    others.length > 0
      ? [...new Set(others.map((o) => o.external?.source ?? externalOrigin(o.installed_path ?? "")))]
      : [];
  const row = (m: ModelStatus, selected: boolean) => {
    const info = noteInfo(m);
    // The VRAM column is gone: mention GPU memory when the note itself doesn't.
    let note = info.text;
    if (props.readsImages && m.entry.vram_mb && !/(GB|MB|VRAM|RAM)/.test(note)) {
      const gb =
        m.entry.vram_mb >= 1024 ? `${(m.entry.vram_mb / 1024).toFixed(0)} GB` : `${m.entry.vram_mb} MB`;
      note = `${note.replace(/\.$/, "")}. Needs about ${gb} of GPU memory.`;
    }
    return (
      <ModelRow
        key={m.entry.id}
        m={m}
        task={props.taskFor(m.entry.id)}
        selected={selected}
        onUse={props.onUse ? () => props.onUse!(m.entry.id) : undefined}
        sizeText={humanSize(m.entry.size_bytes + (m.entry.mmproj_size_bytes ?? 0))}
        note={note}
        recommended={info.recommended}
        readsImages={Boolean(m.entry.mmproj_file_name)}
        dimBadge={m.entry.languages === "english" ? "English-only" : undefined}
        origin={
          m.installed_path && !m.in_own_dir
            ? `From ${m.external?.source ?? externalOrigin(m.installed_path)}${modelQuant(m) ? ` · ${modelQuant(m)}` : ""}`
            : undefined
        }
        onDownload={() => props.onDownload(m.entry.id)}
        onRemove={() => props.onRemove(m.entry.id)}
      />
    );
  };
  return (
    <div className="card mp-list-card">
      {listed.length > 0 && (
        <>
          <div className="mp-list-head">Available models</div>
          <div className="mp-list">
            {listed.map((m) => row(m, Boolean(props.selectedId && props.selectedId === m.entry.id)))}
          </div>
        </>
      )}
      {rest.length > 0 && (
        <details className="mp-others mp-catalog">
          <summary>
            <ChevronDown size={15} strokeWidth={2} aria-hidden />
            <span>Browse other models ({rest.length})</span>
          </summary>
          <div className="mp-list mp-others-body">
            {rest.map((m) => row(m, false))}
          </div>
        </details>
      )}
      {others.length > 0 && (
        <details
          className="mp-others"
          open={Boolean(props.selectedId && others.some((o) => o.entry.id === props.selectedId))}
        >
          <summary>
            <ChevronDown size={15} strokeWidth={2} aria-hidden />
            <span>Models from other apps ({others.length})</span>
            <span className="muted small">{apps.join(", ")}</span>
          </summary>
          <div className="mp-list mp-others-body">
            {others.map((m) => row(m, Boolean(props.selectedId && props.selectedId === m.entry.id)))}
          </div>
        </details>
      )}
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

export default function ModelsPage() {
  const [data, setData] = useState<ModelsStatusView | null>(null);
  const [aiSettings, setAiSettingsState] = useState<AiSettings | null>(null);
  const [resolution, setResolution] = useState<BackendsResolution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // API-key masked display
  const [apiKeyInput, setApiKeyInput] = useState("");
  const apiKeyDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const urlDebounceRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const tasks = useQueue();

  const refresh = useCallback(async () => {
    try {
      const [res, ai, probe] = await Promise.all([modelsStatus(), getAiSettings(), probeBackends()]);
      setData(res);
      setAiSettingsState(ai);
      setResolution(probe);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    return onEvent<number>("task-finished", () => refresh());
  }, [refresh]);

  // ─── patch helpers ──────────────────────────────────────────────────────────

  const applyPatch = useCallback(
    async (patch: AiSettingsPatch) => {
      try {
        setError(null);
        const updated = await setAiSettings(patch);
        setAiSettingsState(updated);
        // Re-probe after settings change
        const probe = await probeBackends();
        setResolution(probe);
      } catch (e) {
        setError(String(e));
      }
    },
    [],
  );

  // ─── model actions ──────────────────────────────────────────────────────────

  const handleDownload = async (modelId: string) => {
    try {
      setError(null);
      await enqueueModelDownload(modelId);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRemove = async (modelId: string) => {
    try {
      setError(null);
      await removeModel(modelId);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  // ─── model status cells ─────────────────────────────────────────────────────

  const downloadTask = (modelId: string): Task | undefined =>
    tasks.find(
      (t) =>
        t.kind.type === "download_model" &&
        t.kind.model_id === modelId &&
        (t.state === "running" || t.state === "queued"),
    );

  // ─── debounced URL setter ───────────────────────────────────────────────────

  function debouncedUrlPatch(key: string, patch: AiSettingsPatch, delayMs = 600) {
    if (urlDebounceRef.current[key]) clearTimeout(urlDebounceRef.current[key]);
    urlDebounceRef.current[key] = setTimeout(() => applyPatch(patch), delayMs);
  }

  // ─── computed lists ─────────────────────────────────────────────────────────

  const whisperModels = data?.models.filter((m) => m.entry.kind === "whisper") ?? [];
  const visionModels = data?.models.filter((m) => m.entry.kind === "vision") ?? [];
  const embeddingModels = data?.models.filter((m) => m.entry.kind === "embedding") ?? [];

  /** "Total downloaded: 24.1 GB across 3 models" — the disk story in one line at the top,
   *  with models reused from other apps called out, so the total reconciles with Storage
   *  (which only counts files inside GhostReel's folder). */
  const installedModels = data?.models.filter((m) => m.installed_path) ?? [];
  const externalModels = installedModels.filter((m) => !m.in_own_dir);
  const modelSize = (m: ModelStatus) => m.entry.size_bytes + (m.entry.mmproj_size_bytes ?? 0);
  const totalBytes = installedModels.reduce((n, m) => n + modelSize(m), 0);
  const externalBytes = externalModels.reduce((n, m) => n + modelSize(m), 0);
  let totalSummary: string | null = null;
  if (data) {
    if (installedModels.length === 0) {
      totalSummary = "No models downloaded yet.";
    } else if (externalModels.length > 0) {
      const apps = [...new Set(externalModels.map((m) => m.external?.source ?? externalOrigin(m.installed_path ?? "")))].join(" and ");
      totalSummary = `Total downloaded: ${humanSize(totalBytes)} across ${installedModels.length} model${installedModels.length === 1 ? "" : "s"} (${humanSize(externalBytes)} reused from ${apps}).`;
    } else {
      totalSummary = `Total downloaded: ${humanSize(totalBytes)} across ${installedModels.length} model${installedModels.length === 1 ? "" : "s"}.`;
    }
  }

  /** Installed in GhostReel's own folder (or not downloaded) vs. reused from another app. */
  const splitModels = (list: ModelStatus[]) => ({
    main: list.filter((m) => !m.installed_path || m.in_own_dir),
    others: list.filter((m) => m.installed_path && !m.in_own_dir),
  });

  const ai = aiSettings;
  const sttBackend = ai?.stt.backend ?? "auto";
  const visionBackend = ai?.vision.backend ?? "auto";
  const embedBackend = ai?.embed.backend ?? "auto";

  /** whisper section data for the GhostPen-style list. */
  const stt = splitModels(whisperModels);
  const sttSelected = ai?.stt.model ?? "auto";
  const useWhisper = (id: string) => {
    applyPatch({ stt: { model: id } });
    setWhisperModel(id).catch(() => null);
  };

  /** frame descriptions section data; selecting is hidden while the server backend answers. */
  const vis = splitModels(visionModels.filter((m) => !m.external || m.entry.mmproj_file_name));
  const chatModels = splitModels(visionModels);
  const visSelectionOff = visionBackend === "server";
  const visSelected = ai?.vision.local_model ?? "qwen3.5-9b";
  const useVision = (id: string) => applyPatch({ vision: { local_model: id } });

  /** embeddings section data; the model is fixed, no selection. */
  const emb = splitModels(embeddingModels);

  /** "Using: <model> · <where>" summary per section — derived from each card's visible selection,
   *  with the raw probe text kept for the tooltip. */
  /** "Using: <model> · <where>" summary per section — derived from each card's visible selection,
   *  with the raw probe text kept for the tooltip. With the automatic speech model, the actually
   *  loaded whisper model names the summary once it is known. */
  const sttAutoLoaded = sttSelected === "auto" && resolution?.stt.target === "local"
    ? data?.current_whisper_model
    : undefined;
  const sttSummary = resolution
    ? sectionSummary(resolution.stt, { model: sttSelected === "auto" ? sttAutoLoaded ?? "Auto" : sttSelected })
    : null;
  const visionSummary = resolution
    ? sectionSummary(resolution.vision, visSelectionOff ? { model: undefined } : { model: visSelected, cli: ai?.vision.cli })
    : null;
  const chatSummary = resolution?.chat
    ? sectionSummary(resolution.chat,
        ai?.chat_model.backend === "server"
          ? { model: undefined }
          : { model: ai?.chat_model.local_model, cli: ai?.chat_model.cli })
    : null;
  // Embeddings have no choice of model (adding one would break compatibility between computers),
  // so the summary names the single file that is actually in use instead of only the location.
  // Only an installed file is named; an absent one is not "in use".
  const embedFile = (emb.main.find((m) => Boolean(m.installed_path)) ?? emb.others[0])?.entry.file_name;
  const embedModel = embedFile?.replace(/\.gguf$/i, "") ?? undefined;
  const embedSummary = resolution ? sectionSummary(resolution.embeddings, { model: embedModel }) : null;

  // ─── render ─────────────────────────────────────────────────────────────────

  return (
    <main>
      <header>
        <div>
          <h1>Models</h1>
          <p className="muted">Download and choose local AI models for speech, vision, and embeddings</p>
        </div>
        <button
          className="ghost"
          onClick={async () => {
            setBusy(true);
            await refresh();
            setBusy(false);
          }}
          disabled={busy}
        >
          {busy ? "Checking…" : "Refresh"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}

      {data && (
        <div className="card models-dir-card">
          <div className="card-head" style={{ gap: "12px" }}>
            <div style={{ minWidth: 0 }}>
              <div className="label">Models directory</div>
              <div className="muted small path">{data.dir}</div>
              {totalSummary && <div className="muted small">{totalSummary}</div>}
            </div>
            <button
              className="ghost small"
              onClick={openModelsDir}
              style={{ whiteSpace: "nowrap", flexShrink: 0 }}
            >
              Open folder
            </button>
          </div>
        </div>
      )}

      {/* ── Speech (whisper) ── */}
      <h2>Speech (whisper)</h2>

      {ai && (
        <div className="card ai-settings-card">
          <div>
            <BackendSegmented
              value={sttBackend}
              onChange={(b) => {
                // The section's single "Auto" lives here: automatic backend and automatic model.
                if (b === "auto") {
                  applyPatch({ stt: { backend: "auto", model: "auto" } });
                  setWhisperModel("auto").catch(() => null);
                } else {
                  applyPatch({ stt: { backend: b } });
                }
              }}
            />
          </div>

          {ai.stt.model === "auto" && (
            <p className="muted small">
              The speech model is picked automatically: large-v3-turbo on a GPU with 6&nbsp;GB free,
              otherwise small. Nothing to download until it runs the first time.
            </p>
          )}

          {(sttBackend === "server" || sttBackend === "auto") && (
            <div className="settings-fields">
              <div className="settings-field">
                <label>URL</label>
                <input
                  type="text"
                  defaultValue={ai.stt.url}
                  placeholder="http://127.0.0.1:8771"
                  onBlur={(e) => debouncedUrlPatch("stt.url", { stt: { url: e.currentTarget.value } })}
                  onChange={(e) => debouncedUrlPatch("stt.url", { stt: { url: e.currentTarget.value } })}
                />
              </div>
              <div className="settings-field">
                <label>Model</label>
                <ModelSelect
                  kind="server"
                  url={ai.stt.url}
                  value={ai.stt.model}
                  onChange={(model) => applyPatch({ stt: { model } })}
                />
              </div>
            </div>
          )}

          {sttSummary && (
            <div className="probe-status" title={sttSummary.title}>
              <span className={sttSummary.cls}>{sttSummary.text}</span>
            </div>
          )}
        </div>
      )}

      {/* whisper model list: the stored "auto" marks the model the app actually loads — there is no
          separate Auto row, the section's Auto lives in the switches above */}
      <ModelListCard
        main={stt.main}
        others={stt.others}
        selectedId={
          sttSelected === "auto"
            ? data?.current_whisper_model
            : sttSelected
        }
        onUse={useWhisper}
        taskFor={downloadTask}
        onDownload={handleDownload}
        onRemove={handleRemove}
      />

      {/* ── Frame descriptions (indexing) ── */}
      <h2>Frame descriptions</h2>
      <p className="muted small">
        Describes keyframes while indexing: a short prompt and one image, run once per keyframe.
      </p>

      {ai && (
        <div className="card ai-settings-card">
          <div>
            <BackendSegmented
              value={visionBackend}
              withCli
              onChange={(b) => applyPatch({ vision: { backend: b } })}
            />
          </div>

          <div className="settings-field">
            <label>Take a frame at least every</label>
            <input
              type="number"
              min={1}
              max={60}
              step={1}
              style={{ width: "5em" }}
              defaultValue={ai.frames.max_interval_s}
              onChange={(e) => {
                const v = Number(e.currentTarget.value);
                if (v >= 1 && v <= 60) debouncedUrlPatch("frames.max_interval_s", { frames: { max_interval_s: v } });
              }}
            />
            <span className="muted small">
              s · shorter = more detail for search and scripts, longer indexing. Scene changes always get a frame.
              Use “Rebuild keyframes” on a project to apply it to indexed videos.
            </span>
          </div>

          <div className="settings-field">
            <label>Sampling size</label>
            <input
              type="number"
              min={256}
              max={3840}
              step={64}
              style={{ width: "6em" }}
              defaultValue={ai.frames.long_side ?? 768}
              onChange={(e) => {
                const v = Number(e.currentTarget.value);
                if (v >= 256 && v <= 3840) debouncedUrlPatch("frames.long_side", { frames: { long_side: v } });
              }}
            />
            <span className="muted small">
              px on long side (default 768) · lower (512–768) = fewer tokens, faster AI description and lower VRAM; higher = finer visual detail.
            </span>
          </div>

          {(visionBackend === "server" || visionBackend === "auto") && (
            <div className="settings-fields">
              <div className="settings-field">
                <label>URL</label>
                <input
                  type="text"
                  defaultValue={ai.vision.url}
                  placeholder="http://127.0.0.1:8089"
                  onBlur={(e) => debouncedUrlPatch("vision.url", { vision: { url: e.currentTarget.value } })}
                  onChange={(e) => debouncedUrlPatch("vision.url", { vision: { url: e.currentTarget.value } })}
                />
              </div>
              <div className="settings-field">
                <label>Model</label>
                <ModelSelect
                  kind="server"
                  url={ai.vision.url}
                  value={ai.vision.model}
                  placeholder="— server default —"
                  onChange={(model) => applyPatch({ vision: { model } })}
                />
              </div>
              <div className="settings-field">
                <label>API key</label>
                <input
                  type="password"
                  value={apiKeyInput}
                  placeholder={ai.vision.api_key_set ? "••••••••  (set — enter new to change)" : "optional"}
                  onChange={(e) => {
                    setApiKeyInput(e.currentTarget.value);
                    if (apiKeyDebounceRef.current) clearTimeout(apiKeyDebounceRef.current);
                    apiKeyDebounceRef.current = setTimeout(
                      () => applyPatch({ vision: { api_key: e.currentTarget.value } }),
                      800,
                    );
                  }}
                />
              </div>
            </div>
          )}

          {visionBackend === "cli" && (
            <CliAgentFields cfg={ai.vision} capability="vision" onPatch={(v) => applyPatch({ vision: v })} />
          )}

          {(visionBackend === "auto" || visionBackend === "local") && (
            <LocalRuntimeFields
              cfg={ai.vision}
              mlx={visionModels.some((m) => Boolean(m.entry.bundle))}
              onPatch={(v) => applyPatch({ vision: v })}
              thinkHint="slower — this runs once per keyframe, so it is usually off"
            />
          )}

          {visionSummary && (
            <div className="probe-status" title={visionSummary.title}>
              <span className={visionSummary.cls}>{visionSummary.text}</span>
            </div>
          )}
        </div>
      )}

      {visionModels.some((m) => Boolean(m.entry.bundle)) && (
        <p className="muted small">
          Recommended vision and chat models use MLX on Apple Silicon. Downloads include the complete model folder.
          Other-app GGUF models use the native llama.cpp helper.
        </p>
      )}
      {/* vision model list; selection only when this computer answers (local/auto) */}
      <ModelListCard
        main={vis.main}
        others={vis.others}
        selectedId={visSelectionOff ? undefined : visSelected}
        onUse={visSelectionOff ? undefined : useVision}
        readsImages
        taskFor={downloadTask}
        onDownload={handleDownload}
        onRemove={handleRemove}
      />

      {/* ── Script chat ── */}
      <h2>Script chat</h2>
      <p className="muted small">
        Drafts and revises scripts. It reads search results and transcripts, so it wants a much bigger context
        window than the frame descriptions do.
      </p>

      {ai && (
        <div className="card ai-settings-card">
          <div>
            <BackendSegmented
              value={ai.chat_model.backend}
              withCli
              onChange={(b) => applyPatch({ chat_model: { backend: b } })}
            />
          </div>

          {(ai.chat_model.backend === "server" || ai.chat_model.backend === "auto") && (
            <div className="settings-fields">
              <div className="settings-field">
                <label>URL</label>
                <input
                  type="text"
                  defaultValue={ai.chat_model.url}
                  placeholder="http://127.0.0.1:8089"
                  onChange={(e) => debouncedUrlPatch("chat_model.url", { chat_model: { url: e.currentTarget.value } })}
                />
              </div>
              <div className="settings-field">
                <label>Model</label>
                <ModelSelect
                  kind="server"
                  url={ai.chat_model.url}
                  value={ai.chat_model.model}
                  placeholder="— server default —"
                  onChange={(model) => applyPatch({ chat_model: { model } })}
                />
              </div>
            </div>
          )}

          {ai.chat_model.backend === "cli" && (
            <CliAgentFields
              cfg={ai.chat_model}
              capability="chat_model"
              onPatch={(v) => applyPatch({ chat_model: v })}
            />
          )}

          {(ai.chat_model.backend === "auto" || ai.chat_model.backend === "local") && (
            <>
              <div className="settings-fields">
                <div className="settings-field">
                  <label>Model on this computer</label>
                  <select
                    value={ai.chat_model.local_model}
                    onChange={(e) => applyPatch({ chat_model: { local_model: e.currentTarget.value } })}
                  >
                    {visionModels.map((m) => (
                      <option key={m.entry.id} value={m.entry.id}>
                        {m.external?.name ?? m.entry.id}{m.external?.quant ? ` · ${m.external.quant}` : ""}
                        {m.installed_path ? "" : " (not downloaded)"}
                      </option>
                    ))}
                  </select>
                  <span className="muted small">Choose an installed model or download one below.</span>
                </div>
              </div>
              <LocalRuntimeFields
                cfg={ai.chat_model}
                mlx={visionModels.some((m) => Boolean(m.entry.bundle))}
                onPatch={(v) => applyPatch({ chat_model: v })}
                thinkHint="a few calls per script — reasoning makes a noticeably better edit"
              />
            </>
          )}

          {chatSummary && (
            <div className="probe-status" title={chatSummary.title}>
              <span className={chatSummary.cls}>{chatSummary.text}</span>
            </div>
          )}
        </div>
      )}

      <ModelListCard
        main={chatModels.main}
        others={chatModels.others}
        selectedId={ai?.chat_model.backend === "server" || ai?.chat_model.backend === "cli" ? undefined : ai?.chat_model.local_model}
        onUse={ai?.chat_model.backend === "server" || ai?.chat_model.backend === "cli" ? undefined : (id) => applyPatch({ chat_model: { local_model: id } })}
        taskFor={downloadTask}
        onDownload={handleDownload}
        onRemove={handleRemove}
      />

      {/* ── Search embeddings ── */}
      <h2>Search embeddings</h2>

      {ai && (
        <div className="card ai-settings-card">
          <div>
            <BackendSegmented
              value={embedBackend}
              onChange={(b) => applyPatch({ embed: { backend: b } })}
            />
          </div>

          {(embedBackend === "server" || embedBackend === "auto") && (
            <div className="settings-fields">
              <div className="settings-field">
                <label>URL</label>
                <input
                  type="text"
                  defaultValue={ai.embed.url}
                  placeholder="http://127.0.0.1:8091"
                  onBlur={(e) => debouncedUrlPatch("embed.url", { embed: { url: e.currentTarget.value } })}
                  onChange={(e) => debouncedUrlPatch("embed.url", { embed: { url: e.currentTarget.value } })}
                />
              </div>
              <div className="settings-field">
                <label>Model</label>
                <ModelSelect
                  kind="server"
                  url={ai.embed.url}
                  value={ai.embed.model}
                  onChange={(model) => applyPatch({ embed: { model } })}
                />
              </div>
            </div>
          )}

          {embedSummary && (
            <div className="probe-status" title={embedSummary.title}>
              <span className={embedSummary.cls}>{embedSummary.text}</span>
            </div>
          )}
        </div>
      )}

      {/* embedding model list; the model is fixed, so rows only manage downloads */}
      <ModelListCard
        main={emb.main}
        others={emb.others}
        taskFor={downloadTask}
        onDownload={handleDownload}
        onRemove={handleRemove}
      />
      <div className="embed-locked-note">
        The embedding model is fixed so search indexes stay compatible between computers.
      </div>
    </main>
  );
}
