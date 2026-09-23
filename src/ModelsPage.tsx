import { useCallback, useEffect, useRef, useState } from "react";
import {
  cliModels,
  cancelTask,
  enqueueModelDownload,
  getAiSettings,
  humanSize,
  modelsStatus,
  openModelsDir,
  probeBackends,
  removeModel,
  scoreMeter,
  serverModels,
  setAiSettings,
  setWhisperModel,
  type AiSettings,
  type AiSettingsPatch,
  type BackendsResolution,
  type Backend,
  type ModelStatus,
  type ModelsStatusView,
  type Resolution,
  type VisionSettings,
  type VisionSettingsPatch,
  testCliAgent,
} from "./api";
import { onEvent } from "./events";
import { useQueue } from "./useQueue";

// ─── helpers ─────────────────────────────────────────────────────────────────

function probeLabel(r: Resolution): { text: string; cls: string } {
  if (r.target === "server" && r.probe) {
    const m = r.probe.model ? ` · ${r.probe.model}` : "";
    const host = (() => {
      try {
        return new URL(r.probe.url).host;
      } catch {
        return r.probe.url;
      }
    })();
    return { text: `Now using: server${m} @ ${host}`, cls: "probe-ok" };
  }
  if (r.target === "local") {
    return { text: r.reason || "Now using: this computer", cls: "probe-ok" };
  }
  if (r.target === "unavailable") {
    return { text: r.reason || "Unavailable", cls: "probe-bad" };
  }
  return { text: r.reason || "", cls: "" };
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
  // What this tool says it can run. Fetched when the tool changes rather than kept in the app:
  // every one of these catalogues moves without us.
  const [cliModelList, setCliModelList] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    setCliModelList([]);
    cliModels(cfg.cli.tool)
      .then((m) => {
        if (!cancelled) setCliModelList(m);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [cfg.cli.tool]);
  const perFrame = capability === "vision";
  return (
    <>
      <div className="settings-fields">
        <div className="settings-field">
          <label>Tool</label>
          <select value={cfg.cli.tool} onChange={(e) => onPatch({ cli: { tool: e.currentTarget.value } })}>
            <option value="claude">claude</option>
            <option value="agy">agy</option>
            <option value="opencode">opencode</option>
            <option value="codex">codex</option>
          </select>
          <span className={cfg.cli.installed ? "good-text small" : "bad-text small"}>
            {cfg.cli.installed ? "found on this computer" : "not found — install it or set a full path"}
          </span>
        </div>
        <div className="settings-field">
          <label>Model</label>
          {cliModelList.length > 0 ? (
            // The tool told us what it has. Its own default stays available, and a model it did
            // not list is still shown, so switching tools never silently discards a setting.
            <select
              value={cfg.cli.model}
              onChange={(e) => onPatch({ cli: { model: e.currentTarget.value } })}
            >
              <option value="">— the tool's default —</option>
              {!cliModelList.includes(cfg.cli.model) && cfg.cli.model && (
                <option value={cfg.cli.model}>{cfg.cli.model} (not in its list)</option>
              )}
              {cliModelList.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              defaultValue={cfg.cli.model}
              placeholder="— the tool's default —"
              onChange={(e) => onPatch({ cli: { model: e.currentTarget.value } })}
            />
          )}
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
          ? "One call per keyframe. claude bills per call (about $0.04 a frame here, so a 1500 frame library runs into tens of dollars); agy and opencode spend their own quota instead."
          : "A handful of calls per script — far cheaper than describing every frame. claude bills per call; agy and opencode spend their own quota."}
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
}: {
  cfg: VisionSettings;
  onPatch: (p: VisionSettingsPatch) => void;
  /** Why thinking is or isn't worth its cost for this capability. */
  thinkHint: string;
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
          <option value="q4_0">q4_0 — ~4× the context per GB</option>
          <option value="q8_0">q8_0 — balanced</option>
          <option value="f16">f16 — best quality, 4× the VRAM</option>
        </select>
      </div>
      <div className="settings-field">
        <label>Flash attention</label>
        <select value={cfg.flash_attn} onChange={(e) => onPatch({ flash_attn: e.currentTarget.value })}>
          <option value="auto">auto</option>
          <option value="on">on</option>
          <option value="off">off</option>
        </select>
      </div>
      <div className="settings-field">
        <label>Think first</label>
        <input type="checkbox" checked={cfg.think} onChange={(e) => onPatch({ think: e.currentTarget.checked })} />
        <span className="muted small">
          {thinkHint}
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

// ─── main component ───────────────────────────────────────────────────────────

export default function ModelsPage() {
  const [data, setData] = useState<ModelsStatusView | null>(null);
  const [aiSettings, setAiSettingsState] = useState<AiSettings | null>(null);
  const [resolution, setResolution] = useState<BackendsResolution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Server model lists per section (fetched on demand)
  const [sttServerModels, setSttServerModels] = useState<string[]>([]);
  const [visionServerModels, setVisionServerModels] = useState<string[]>([]);
  const [embedServerModels, setEmbedServerModels] = useState<string[]>([]);
  const [fetchingServerModels, setFetchingServerModels] = useState<Record<string, boolean>>({});

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

  const fetchServerModelList = useCallback(
    async (
      url: string,
      setter: (list: string[]) => void,
      key: string,
    ) => {
      if (!url) return;
      setFetchingServerModels((p) => ({ ...p, [key]: true }));
      try {
        const list = await serverModels(url);
        setter(list);
      } catch {
        setter([]);
      } finally {
        setFetchingServerModels((p) => ({ ...p, [key]: false }));
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

  const renderStatus = (m: ModelStatus) => {
    const activeTask = tasks.find(
      (t) =>
        t.kind.type === "download_model" &&
        t.kind.model_id === m.entry.id &&
        (t.state === "running" || t.state === "queued"),
    );

    if (activeTask) {
      if (activeTask.state === "running") {
        const pct = Math.round((activeTask.progress?.fraction ?? 0) * 100);
        return (
          <div className="status-cell">
            <span className="status-badge downloading">Downloading {pct}%</span>
            <div className="meter small-meter">
              <div style={{ width: `${Math.max(4, pct)}%` }} />
            </div>
          </div>
        );
      }
      return <span className="status-badge queued">Queued</span>;
    }

    if (m.installed_path) {
      if (m.in_own_dir) {
        return (
          <span className="status-badge installed" title={m.installed_path}>
            Installed
          </span>
        );
      }
      const lower = m.installed_path.toLowerCase();
      const origin = lower.includes("ghostpen")
        ? "GhostPen"
        : lower.includes("lmstudio")
          ? "LM Studio"
          : "external";
      return (
        <span className="status-badge external" title={m.installed_path}>
          Found in {origin}
        </span>
      );
    }

    if (m.partial_bytes) {
      return (
        <span className="status-badge partial">
          Partial ({humanSize(m.partial_bytes)})
        </span>
      );
    }

    return <span className="status-badge not-installed">Not installed</span>;
  };

  const renderActions = (m: ModelStatus) => {
    const activeTask = tasks.find(
      (t) =>
        t.kind.type === "download_model" &&
        t.kind.model_id === m.entry.id &&
        (t.state === "running" || t.state === "queued"),
    );

    if (activeTask) {
      return (
        <button
          className="ghost small"
          onClick={() => cancelTask(activeTask.id)}
          title="Cancel download (keeps partial file)"
        >
          Cancel
        </button>
      );
    }

    if (m.installed_path && m.in_own_dir) {
      return (
        <button
          className="ghost small danger"
          onClick={() => handleRemove(m.entry.id)}
          title="Delete from GhostReel's models folder"
        >
          Remove
        </button>
      );
    }

    if (m.installed_path && !m.in_own_dir) {
      return <span className="muted small">Reused</span>;
    }

    if (m.partial_bytes) {
      return (
        <div className="button-group">
          <button className="small" onClick={() => handleDownload(m.entry.id)} title="Resume download">
            Resume
          </button>
          <button
            className="ghost small danger"
            onClick={() => handleRemove(m.entry.id)}
            title="Discard partial file"
          >
            Clear
          </button>
        </div>
      );
    }

    return (
      <button className="small" onClick={() => handleDownload(m.entry.id)}>
        Download
      </button>
    );
  };

  // ─── debounced URL setter ───────────────────────────────────────────────────

  function debouncedUrlPatch(key: string, patch: AiSettingsPatch, delayMs = 600) {
    if (urlDebounceRef.current[key]) clearTimeout(urlDebounceRef.current[key]);
    urlDebounceRef.current[key] = setTimeout(() => applyPatch(patch), delayMs);
  }

  // ─── computed lists ─────────────────────────────────────────────────────────

  const whisperModels = data?.models.filter((m) => m.entry.kind === "whisper") ?? [];
  const visionModels = data?.models.filter((m) => m.entry.kind === "vision") ?? [];
  const embeddingModels = data?.models.filter((m) => m.entry.kind === "embedding") ?? [];

  const ai = aiSettings;
  const sttBackend = ai?.stt.backend ?? "auto";
  const visionBackend = ai?.vision.backend ?? "auto";
  const embedBackend = ai?.embed.backend ?? "auto";

  const sttProbe = resolution ? probeLabel(resolution.stt) : null;
  const visionProbe = resolution ? probeLabel(resolution.vision) : null;
  const chatProbe = resolution?.chat ? probeLabel(resolution.chat) : null;
  const embedProbe = resolution ? probeLabel(resolution.embeddings) : null;

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
          <div className="card-head">
            <div>
              <div className="label">Models directory</div>
              <div className="muted small path">{data.dir}</div>
            </div>
            <button className="ghost small" onClick={openModelsDir}>
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
              onChange={(b) => applyPatch({ stt: { backend: b } })}
            />
          </div>

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
                <button
                  className="ghost small"
                  disabled={!!fetchingServerModels["stt"]}
                  onClick={() => fetchServerModelList(ai.stt.url, setSttServerModels, "stt")}
                >
                  {fetchingServerModels["stt"] ? "…" : "Refresh"}
                </button>
              </div>
              {sttServerModels.length > 0 && (
                <div className="settings-field">
                  <label>Model</label>
                  <select
                    value={ai.stt.model}
                    onChange={(e) => applyPatch({ stt: { model: e.currentTarget.value } })}
                  >
                    {sttServerModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}

          {sttProbe && (
            <div className="probe-status">
              <span className={sttProbe.cls}>{sttProbe.text}</span>
            </div>
          )}
        </div>
      )}

      {/* whisper model table */}
      <div className="card whisper-config-card">
        <div className="whisper-auto-row">
          <label className="radio-label">
            <input
              type="radio"
              name="whisper-select"
              value="auto"
              checked={(ai?.stt.model ?? "auto") === "auto"}
              onChange={() => {
                applyPatch({ stt: { model: "auto" } });
                setWhisperModel("auto").catch(() => null);
              }}
            />
            <span className="radio-text">
              <strong>Auto</strong> — large-v3-turbo on ≥6 GB NVIDIA GPU, otherwise small (recommended)
            </span>
          </label>
        </div>
      </div>

      <div className="card table-card">
        <table className="models-table">
          <thead>
            <tr>
              <th className="col-radio">Use</th>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {whisperModels.map((m) => {
              const isSelected = (ai?.stt.model ?? "auto") === m.entry.id;
              return (
                <tr key={m.entry.id} className={isSelected ? "selected-row" : ""}>
                  <td className="col-radio">
                    <input
                      type="radio"
                      name="whisper-select"
                      value={m.entry.id}
                      checked={isSelected}
                      onChange={() => {
                        applyPatch({ stt: { model: m.entry.id } });
                        setWhisperModel(m.entry.id).catch(() => null);
                      }}
                      title={`Use ${m.entry.id} for transcription`}
                    />
                  </td>
                  <td>
                    <div className="model-name">
                      <strong>{m.entry.id}</strong>
                      {m.entry.languages === "english" && (
                        <span className="pill en-pill">English-only</span>
                      )}
                    </div>
                  </td>
                  <td className="col-size">{humanSize(m.entry.size_bytes)}</td>
                  <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                  </td>
                  <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                  </td>
                  <td className="muted small col-note">{m.entry.note}</td>
                  <td>{renderStatus(m)}</td>
                  <td className="col-action">{renderActions(m)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

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
                <button
                  className="ghost small"
                  disabled={!!fetchingServerModels["vision"]}
                  onClick={() => fetchServerModelList(ai.vision.url, setVisionServerModels, "vision")}
                >
                  {fetchingServerModels["vision"] ? "…" : "Refresh"}
                </button>
              </div>
              {visionServerModels.length > 0 && (
                <div className="settings-field">
                  <label>Model</label>
                  <select
                    value={ai.vision.model}
                    onChange={(e) => applyPatch({ vision: { model: e.currentTarget.value } })}
                  >
                    <option value="">— server default —</option>
                    {visionServerModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}
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
              onPatch={(v) => applyPatch({ vision: v })}
              thinkHint="slower — this runs once per keyframe, so it is usually off"
            />
          )}

          {visionProbe && (
            <div className="probe-status">
              <span className={visionProbe.cls}>{visionProbe.text}</span>
            </div>
          )}
        </div>
      )}

      {/* vision model table */}
      <div className="card table-card">
        <table className="models-table vision-table">
          <thead>
            <tr>
              {visionBackend !== "server" && <th className="col-radio">Use</th>}
              <th>Model</th>
              <th>Size</th>
              <th>VRAM</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {visionModels.map((m) => {
              const isSelected = (ai?.vision.local_model ?? "bonsai-27b") === m.entry.id;
              const totalSize = m.entry.size_bytes + (m.entry.mmproj_size_bytes ?? 0);
              return (
                <tr key={m.entry.id} className={isSelected && visionBackend !== "server" ? "selected-row" : ""}>
                  {visionBackend !== "server" && (
                    <td className="col-radio">
                      <input
                        type="radio"
                        name="vision-select"
                        value={m.entry.id}
                        checked={isSelected}
                        onChange={() => applyPatch({ vision: { local_model: m.entry.id } })}
                        title={`Use ${m.entry.id} for vision`}
                      />
                    </td>
                  )}
                  <td>
                    <div className="model-name">
                      <strong>{m.entry.id}</strong>
                      {m.entry.mmproj_file_name && (
                        <span className="pill" title={`Model + projector: ${m.entry.file_name}, ${m.entry.mmproj_file_name}`}>
                          model + projector
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="col-size">{humanSize(totalSize)}</td>
                  <td className="col-size">
                    {m.entry.vram_mb != null ? `${m.entry.vram_mb >= 1024 ? `${(m.entry.vram_mb / 1024).toFixed(0)} GB` : `${m.entry.vram_mb} MB`}` : "—"}
                  </td>
                  <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                  </td>
                  <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                    <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                  </td>
                  <td className="muted small col-note">{m.entry.note}</td>
                  <td>{renderStatus(m)}</td>
                  <td className="col-action">{renderActions(m)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

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
                <input
                  type="text"
                  defaultValue={ai.chat_model.model}
                  placeholder="— server default —"
                  onChange={(e) =>
                    debouncedUrlPatch("chat_model.model", { chat_model: { model: e.currentTarget.value } })
                  }
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
                        {m.entry.id}
                        {m.installed_path ? "" : " (not downloaded)"}
                      </option>
                    ))}
                  </select>
                  <span className="muted small">Download it in the table above.</span>
                </div>
              </div>
              <LocalRuntimeFields
                cfg={ai.chat_model}
                onPatch={(v) => applyPatch({ chat_model: v })}
                thinkHint="a few calls per script — reasoning makes a noticeably better edit"
              />
            </>
          )}

          {chatProbe && (
            <div className="probe-status">
              <span className={chatProbe.cls}>{chatProbe.text}</span>
            </div>
          )}
        </div>
      )}

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
                <button
                  className="ghost small"
                  disabled={!!fetchingServerModels["embed"]}
                  onClick={() => fetchServerModelList(ai.embed.url, setEmbedServerModels, "embed")}
                >
                  {fetchingServerModels["embed"] ? "…" : "Refresh"}
                </button>
              </div>
              {embedServerModels.length > 0 && (
                <div className="settings-field">
                  <label>Model</label>
                  <select value={ai.embed.model} onChange={(e) => applyPatch({ embed: { model: e.currentTarget.value } })}>
                    {embedServerModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}

          {embedProbe && (
            <div className="probe-status">
              <span className={embedProbe.cls}>{embedProbe.text}</span>
            </div>
          )}
        </div>
      )}

      <div className="card table-card">
        <table className="models-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Size</th>
              <th>Speed</th>
              <th>Accuracy</th>
              <th>Note</th>
              <th>Status</th>
              <th className="col-action">Action</th>
            </tr>
          </thead>
          <tbody>
            {embeddingModels.map((m) => (
              <tr key={m.entry.id}>
                <td>
                  <div className="model-name">
                    <strong>{m.entry.id}</strong>
                    <span className="pill">Embedding</span>
                  </div>
                </td>
                <td className="col-size">{humanSize(m.entry.size_bytes)}</td>
                <td className="col-meter" title={`Speed ${m.entry.speed}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.speed)}</span>
                </td>
                <td className="col-meter" title={`Accuracy ${m.entry.accuracy}/5`}>
                  <span className="score-meter">{scoreMeter(m.entry.accuracy)}</span>
                </td>
                <td className="muted small col-note">{m.entry.note}</td>
                <td>{renderStatus(m)}</td>
                <td className="col-action">{renderActions(m)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="embed-locked-note">
        The embedding model is fixed to keep search indexes compatible between computers (AGENTS.md rule 3).
      </div>
    </main>
  );
}
