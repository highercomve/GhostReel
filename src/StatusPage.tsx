import { useCallback, useEffect, useState } from "react";
import { Check, Minus, X } from "lucide-react";
import UpdaterPanel, { type Updater } from "./UpdaterPanel";
import { appVersion, doctor, getAiSettings, type AiSettings, type DoctorView, type ModelFile, type Resolution } from "./api";
import { isActive, useQueue } from "./useQueue";
import { navigate } from "./navigate";

const CAPABILITIES: { key: "vision" | "embeddings" | "stt"; label: string; hint: string }[] = [
  { key: "vision", label: "Frame descriptions", hint: "describes keyframes while indexing" },
  { key: "embeddings", label: "Search embeddings", hint: "matches meanings across transcripts and descriptions" },
  { key: "stt", label: "Transcription", hint: "turns speech in the audio into text" },
];

/** "just now", "5 min ago", "2 h ago", "3 d ago" — for the indexer's last-run line. */
function agoText(ts: number): string {
  const ms = ts < 1e11 ? ts * 1000 : ts;
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

/**
 * The probe's reason is sometimes a raw config echo ("backend = local"), not copy — keep only
 * the human parts. The pill already says where the capability runs.
 */
function cleanReason(r: Resolution): string {
  const s = r.reason.trim();
  const but = s.match(/^backend\s*=\s*server\s+but\s+([\s\S]+)$/);
  if (but) return `the server can't do this: ${but[1]}`;
  if (/^backend\s*=/.test(s) || s === "server not probed") return "";
  return s;
}

function BackendCard({
  label,
  hint,
  r,
  model,
  missing = [],
  missingNote,
}: {
  label: string;
  hint: string;
  r: Resolution;
  /** The model this capability is set to use, shown next to the role so "Models page" points
   *  at a concrete download. Servers show it in the where line instead. */
  model?: string | null;
  /** Required model files still to be downloaded, in Status wording. */
  missing?: string[];
  /** Overrides the generic missing line — lets the copy say GPU/CPU files are alternatives. */
  missingNote?: string;
}) {
  const where =
    r.target === "server" && r.probe
      ? `${r.probe.model ?? "server"} · ${r.probe.url.replace(/^https?:\/\//, "")}`
      : r.target === "local"
        ? model
          ? `${model} · runs on this computer`
          : "Runs on this computer"
        : "Not available";
  const pill = r.target === "local" ? "Local" : r.target === "server" ? "Server" : "Unavailable";
  const reason = cleanReason(r);
  const missingLine =
    missingNote ??
    (missing.length === 1
      ? `needs one model file on this computer: ${missing[0]}`
      : missing.length
        ? `needs model files on this computer: ${missing.join(" · ")}`
        : "");
  return (
    <div className={`card backend ${r.target}`}>
      <div className="card-head">
        <span className="label">{label}</span>
        <span className={`pill ${r.target}`}>{pill}</span>
      </div>
      <div className="where">{where}</div>
      <div className="muted small">{[hint, reason].filter(Boolean).join(" · ")}</div>
      {missingLine && <div className="bad-text small missing-models">{missingLine}</div>}
      {missing.length > 0 && (
        <div className="missing-models">
          <button className="ghost small" onClick={() => navigate("models")}>
            Set up on the Models page
          </button>
        </div>
      )}
    </div>
  );
}

/** Friendly name for a model file row; the raw file name stays in the row's tooltip. */
function friendlyRole(role: string): string {
  const r = role.toLowerCase();
  if (r.includes("mlx")) return "Vision & chat models (Apple MLX)";
  if (r.startsWith("whisper")) return "Transcription model";
  if (r.startsWith("vision projector")) return "Vision projector";
  if (r.startsWith("vision")) return "Vision model";
  if (r.startsWith("embedding")) return "Embeddings model";
  return role;
}

/** Which capability a model file belongs to (matches the doctor's role names). */
function capabilityOf(role: string): "vision" | "embeddings" | "stt" | null {
  const r = role.toLowerCase();
  if (r.startsWith("whisper")) return "stt";
  if (r.startsWith("embedding")) return "embeddings";
  if (r.startsWith("vision")) return "vision";
  return null;
}

const CAP_LABEL: Record<"vision" | "embeddings" | "stt", string> = {
  vision: "Frame descriptions",
  embeddings: "Search embeddings",
  stt: "Transcription",
};

/**
 * A missing file is genuinely required when its capability answers from this computer.
 * A CLI backend runs the tool instead of a model, so its files are optional.
 */
function fileRequired(r: Resolution): boolean {
  return r.target === "local" && r.backend !== "cli";
}

/** Missing required files grouped by capability, in Status wording ("Transcription model (GPU)"). */
function missingByCapability(
  models: ModelFile[],
  resolutions: Record<"vision" | "embeddings" | "stt", Resolution>,
): Record<"vision" | "embeddings" | "stt", string[]> {
  const out: Record<"vision" | "embeddings" | "stt", string[]> = { vision: [], embeddings: [], stt: [] };
  for (const m of models) {
    if (m.found) continue;
    const cap = capabilityOf(m.role);
    if (!cap || !fileRequired(resolutions[cap])) continue;
    out[cap].push(friendlyRole(m.role));
  }
  return out;
}

const baseName = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** The indexer as a row like the others, with its last run — not a lone floating banner.
 *  "Ready" is only claimed when every model file the local route needs is present. */
function IndexerRow({
  lastRun,
  runningNow,
  missing = [],
}: {
  lastRun: number | null;
  runningNow: boolean;
  /** Required model files still to be downloaded, in Status wording. */
  missing?: string[];
}) {
  // The transcription GPU and CPU files are alternatives (one per hardware), not two downloads —
  // so the count and the list treat them as one requirement if both are missing.
  const shown = missing.filter((m) => m !== "Transcription model (CPU)");
  const bothVariants = shown.length !== missing.length;
  const altNote = bothVariants
    ? " (two transcription entries are alternatives — either one is enough)"
    : "";
  const state = runningNow ? "Working" : missing.length ? "Needs models" : "Ready";
  return (
    <div className="card backend local">
      <div className="card-head">
        <span className="label">Indexer</span>
        <span className={`pill ${runningNow ? "local" : missing.length ? "warn" : "good"}`}>{state}</span>
      </div>
      <div className="where">{runningNow ? "Indexing now" : missing.length ? "Not ready to index" : "Ready to index"}</div>
      <div className="muted small">
        {runningNow
          ? "watch the queue under Activity"
          : missing.length
            ? `${shown.length === 1 ? "One model file is missing" : `${shown.length} model files are missing`}: ${shown.join(" · ")}${altNote} — set them up below or on the Models page`
            : lastRun
              ? `last run ${agoText(lastRun)}`
              : "no indexing run yet — it starts when videos are added to a project"}
      </div>
      {!runningNow && missing.length > 0 && (
        <button className="ghost small" onClick={() => navigate("models")}>
          Set up on the Models page
        </button>
      )}
    </div>
  );
}

// ---- Updater section -------------------------------------------------------

// ---- Main page -------------------------------------------------------------

export default function StatusPage({ updater }: { updater: Updater }) {
  const [view, setView] = useState<DoctorView | null>(null);
  const [ai, setAi] = useState<AiSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [appVer, setAppVer] = useState<string>("");

  // The queue is the only history of indexing runs; the latest finished one dates "last run".
  const tasks = useQueue();
  const finishedRuns = tasks
    .filter((t) => t.kind.type === "index" && t.state === "done" && t.finished_at)
    .map((t) => t.finished_at as number);
  const lastRun = finishedRuns.length ? Math.max(...finishedRuns) : null;
  const runningNow = tasks.some((t) => t.kind.type === "index" && isActive(t));

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setView(await doctor());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    appVersion()
      .then(setAppVer)
      .catch(() => {});
    // Also read which models the settings pick, so each capability card names the concrete model.
    getAiSettings().then(setAi).catch(() => {});
  }, [refresh]);

  const r = view?.report;
  // Which required model files are still missing, per capability (local backends only; a CLI
  // backend runs the tool instead of a model file).
  const missingFiles = r
    ? missingByCapability(
        r.models,
        { vision: r.vision, embeddings: r.embeddings, stt: r.stt },
      )
    : null;
  const indexerMissing = missingFiles ? [...new Set([...missingFiles.vision, ...missingFiles.embeddings, ...missingFiles.stt])] : [];

  /** Whether this computer's GPU qualifies for the heavyweight transcription file (same 6 GB
   *  rule the backend uses to pick between the whisper GPU and CPU files). */
  const gpuForTurbo = (r?.gpu ?? []).some((g) => g.vram_total_mib >= 6000);

  /** The selected model behind a capability, named so "Models page" points at a concrete download.
   *  Server routes already name their model in the where line; a CLI route names its tool. */
  const selectedModel = (cap: "vision" | "embeddings" | "stt", res: Resolution): string | null => {
    if (res.backend === "cli") {
      const tool = cap === "vision" ? ai?.vision.cli.tool : undefined;
      return tool ? `${tool} (CLI)` : null;
    }
    if (res.target !== "local") return null;
    const installed = r?.models.find(m => capabilityOf(m.role) === cap && m.found);
    if (cap === "stt" && installed) return installed.pattern.replace(/^ggml-/, "").replace(/\.bin$/, "");
    if (cap === "vision") return ai?.vision.local_model ?? null;
    if (cap === "embeddings") return "embeddinggemma-300M-Q8_0";
    if (ai?.stt.model && ai.stt.model !== "auto") return ai.stt.model;
    return gpuForTurbo ? "large-v3-turbo" : "small";
  };

  /** The missing-files line for a card, with transcription's CPU/GPU files explained as
   *  alternatives (either one works — the hardware decides) instead of two separate downloads. */
  const missingNoteFor = (cap: "vision" | "embeddings" | "stt"): string | undefined => {
    const list = missingFiles?.[cap];
    if (cap !== "stt" || !list || !ai) return undefined;
    if (!list.includes("Transcription model (GPU)") || !list.includes("Transcription model (CPU)")) {
      return undefined;
    }
    const chosen = ai.stt.model && ai.stt.model !== "auto" ? ai.stt.model : gpuForTurbo ? "large-v3-turbo" : "small";
    const pick = chosen.startsWith("large-v3-turbo") ? "the GPU file" : "the CPU file";
    return `needs one transcription file — the two entries are alternatives, not both: large-v3-turbo (GPU) or small (CPU), and ${pick} is the one that fits this computer`;
  };

  return (
    <main>
      <header>
        <div>
          <h1>Status</h1>
          <p className="muted">What GhostReel will use on this computer</p>
        </div>
        <button onClick={refresh} disabled={busy}>
          {busy ? "Checking…" : "Check again"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}

      {/* Updates card — always visible */}
      {appVer && <UpdaterPanel appVer={appVer} updater={updater} />}

      {view && r && (
        <>
          {view.blockers.length === 0 ? (
            <IndexerRow lastRun={lastRun} runningNow={runningNow} missing={indexerMissing} />
          ) : (
            <div className="banner bad">
              <strong>Needs attention</strong>
              <ul>
                {view.blockers.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </div>
          )}

          <h2>AI</h2>
          <section className="grid">
            {CAPABILITIES.map((c) => (
              <BackendCard
                key={c.key}
                label={c.label}
                hint={c.hint}
                r={r[c.key]}
                model={selectedModel(c.key, r[c.key])}
                missing={missingFiles?.[c.key] ?? []}
                missingNote={missingNoteFor(c.key)}
              />
            ))}
          </section>

          <h2>This computer</h2>
          <section className="grid">
            <div className="card">
              <div className="label">GPU</div>
              {r.gpu.length === 0 ? (
                <div className="muted">No NVIDIA GPU — local models run on the CPU (slow).</div>
              ) : (
                r.gpu.map((g) => (
                  <div key={g.name}>
                    <div className="where">{g.name}</div>
                    <div className="meter">
                      <div style={{ width: `${(100 * g.vram_used_mib) / Math.max(g.vram_total_mib, 1)}%` }} />
                    </div>
                    <div className="muted small">
                      {(g.vram_used_mib / 1024).toFixed(1)} / {(g.vram_total_mib / 1024).toFixed(1)} GB used ·
                      driver {g.driver}
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="card">
              <div className="label">Video tools</div>
              {[r.ffmpeg, r.ffprobe].map((t) => (
                <div key={t.name} className={`icon-text ${t.path ? "" : "bad-text"}`}>
                  {t.path ? <Check size={14} strokeWidth={2} aria-label="found" /> : <X size={14} strokeWidth={2} aria-label="missing" />} {t.name} <span className="muted small">{t.version ?? "not found"}</span>
                </div>
              ))}
            </div>
            <div className="card">
              <div className="label">Library database</div>
              <div className={`icon-text ${r.db.ok ? "" : "bad-text"}`}>
                {r.db.ok ? <Check size={14} strokeWidth={2} aria-label="ok" /> : <X size={14} strokeWidth={2} aria-label="error" />}
                {r.db.ok ? `schema v${r.db.schema_version} · sqlite-vec ${r.db.sqlite_vec}` : r.db.error}
              </div>
              <div className="muted small path">{r.db.path}</div>
            </div>
          </section>

          <h2>Local model files</h2>
          <section className="card list status-files">
            {r.models.map((m) => {
              const cap = capabilityOf(m.role);
              const res = cap ? r[cap] : null;
              const needed = Boolean(cap && res && res.target === "local" && res.backend !== "cli");
              return (
                <div key={m.pattern} className="row" title={m.found ?? m.pattern}>
                  <span className="icon-text">
                    {m.found ? <Check size={14} strokeWidth={2} aria-label="downloaded" /> : <Minus size={14} strokeWidth={2} aria-label="not downloaded" />}
                  </span>
                  <span className="role">{friendlyRole(m.role)}</span>
                  {m.found ? (
                    <span className="muted small path">{baseName(m.found)}</span>
                  ) : (
                    <span className="muted small">
                      {needed && cap
                        ? `not downloaded — needed by ${CAP_LABEL[cap]} · get it on the Models page`
                        : cap && res && res.target === "server"
                          ? `not downloaded — optional here, ${CAP_LABEL[cap]} run on the server`
                          : "not downloaded · get it on the Models page"}
                    </span>
                  )}
                </div>
              );
            })}
          </section>
          <p className="muted small">
            Missing models download from the Models page — the raw file names are in each row's tooltip.
          </p>

          <p className="muted small path">
            Config: {r.config_file}
            {r.config_error ? ` — ${r.config_error}` : ""}
          </p>
        </>
      )}
    </main>
  );
}
