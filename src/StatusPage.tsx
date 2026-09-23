import { useCallback, useEffect, useState } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { appVersion, doctor, type DoctorView, type Resolution } from "./api";

const CAPABILITIES: { key: "vision" | "embeddings" | "stt"; label: string; hint: string }[] = [
  { key: "vision", label: "Frame descriptions", hint: "vision model" },
  { key: "embeddings", label: "Search embeddings", hint: "embeddinggemma" },
  { key: "stt", label: "Transcription", hint: "whisper" },
];

function BackendCard({ label, hint, r }: { label: string; hint: string; r: Resolution }) {
  const where =
    r.target === "server" && r.probe
      ? `${r.probe.model ?? "server"} · ${r.probe.url.replace(/^https?:\/\//, "")}`
      : r.target === "local"
        ? "runs on this computer"
        : "not available";
  return (
    <div className={`card backend ${r.target}`}>
      <div className="card-head">
        <span className="label">{label}</span>
        <span className={`pill ${r.target}`}>{r.target}</span>
      </div>
      <div className="where">{where}</div>
      <div className="muted small">
        {hint} · backend = {r.backend} · {r.reason}
      </div>
    </div>
  );
}

// ---- Updater section -------------------------------------------------------

type UpdaterState =
  | { phase: "idle" }
  | { phase: "checking" }
  | { phase: "up_to_date" }
  | { phase: "available"; update: Update }
  | { phase: "downloading"; downloaded: number; total: number | null }
  | { phase: "ready" }
  | { phase: "error"; message: string };

function fmtBytes(n: number) {
  return `${(n / 1_048_576).toFixed(1)} MB`;
}

function UpdaterPanel({ appVer }: { appVer: string }) {
  const [state, setState] = useState<UpdaterState>({ phase: "idle" });

  const checkForUpdates = useCallback(async () => {
    setState({ phase: "checking" });
    try {
      const update = await check();
      if (!update) {
        setState({ phase: "up_to_date" });
      } else {
        setState({ phase: "available", update });
      }
    } catch (e) {
      setState({ phase: "error", message: String(e) });
    }
  }, []);

  const downloadAndInstall = useCallback(async () => {
    if (state.phase !== "available") return;
    const { update } = state;
    setState({ phase: "downloading", downloaded: 0, total: null });
    try {
      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case "Started":
            setState({ phase: "downloading", downloaded: 0, total: event.data.contentLength ?? null });
            break;
          case "Progress":
            setState((prev) =>
              prev.phase === "downloading"
                ? { ...prev, downloaded: prev.downloaded + event.data.chunkLength }
                : prev
            );
            break;
          case "Finished":
            setState({ phase: "ready" });
            break;
        }
      });
      setState({ phase: "ready" });
    } catch (e) {
      setState({ phase: "error", message: String(e) });
    }
  }, [state]);

  const restart = useCallback(async () => {
    await relaunch();
  }, []);

  const isChecking = state.phase === "checking";

  return (
    <div className="card" style={{ marginTop: "1rem" }}>
      <div className="card-head">
        <span className="label">GhostReel {appVer}</span>
        {state.phase === "idle" || state.phase === "error" || state.phase === "up_to_date" ? (
          <button onClick={checkForUpdates} disabled={isChecking}>
            Check for updates
          </button>
        ) : state.phase === "available" ? (
          <button onClick={downloadAndInstall}>Download and install</button>
        ) : state.phase === "ready" ? (
          <button onClick={restart}>Restart to finish</button>
        ) : null}
      </div>

      {state.phase === "checking" && <div className="muted small">Checking…</div>}

      {state.phase === "up_to_date" && (
        <div className="muted small">You are on the latest version.</div>
      )}

      {state.phase === "available" && (
        <div>
          <div className="where">
            Version {state.update.version} available
          </div>
          {state.update.body && (
            <pre className="muted small" style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem" }}>
              {state.update.body}
            </pre>
          )}
        </div>
      )}

      {state.phase === "downloading" && (
        <div>
          <div className="muted small">
            Downloading…{" "}
            {fmtBytes(state.downloaded)}
            {state.total != null
              ? ` / ${fmtBytes(state.total)} (${((state.downloaded / state.total) * 100).toFixed(0)}%)`
              : ""}
          </div>
          {state.total != null && (
            <div className="meter" style={{ marginTop: "0.4rem" }}>
              <div style={{ width: `${(state.downloaded / state.total) * 100}%` }} />
            </div>
          )}
        </div>
      )}

      {state.phase === "ready" && (
        <div className="muted small">Download complete. Click "Restart to finish" to apply.</div>
      )}

      {state.phase === "error" && (
        <div className="bad-text small" style={{ marginTop: "0.4rem" }}>
          {state.message}
        </div>
      )}
    </div>
  );
}

// ---- Main page -------------------------------------------------------------

export default function StatusPage() {
  const [view, setView] = useState<DoctorView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [appVer, setAppVer] = useState<string>("");

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
  }, [refresh]);

  const r = view?.report;

  return (
    <main>
      <header>
        <div>
          <h1>Status</h1>
          <p className="muted">What GhostReel will use on this computer{r ? ` · v${r.version}` : ""}</p>
        </div>
        <button onClick={refresh} disabled={busy}>
          {busy ? "Checking…" : "Check again"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}

      {/* Updates card — always visible */}
      {appVer && <UpdaterPanel appVer={appVer} />}

      {view && r && (
        <>
          {view.blockers.length === 0 ? (
            <div className="banner good">Ready to index.</div>
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
              <BackendCard key={c.key} label={c.label} hint={c.hint} r={r[c.key]} />
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
                <div key={t.name} className={t.path ? "" : "bad-text"}>
                  {t.path ? "✓" : "✗"} {t.name} <span className="muted small">{t.version ?? "not found"}</span>
                </div>
              ))}
            </div>
            <div className="card">
              <div className="label">Library database</div>
              <div className={r.db.ok ? "" : "bad-text"}>
                {r.db.ok ? `✓ schema v${r.db.schema_version} · sqlite-vec ${r.db.sqlite_vec}` : `✗ ${r.db.error}`}
              </div>
              <div className="muted small path">{r.db.path}</div>
            </div>
          </section>

          <h2>Local model files</h2>
          <section className="card list">
            {r.models.map((m) => (
              <div key={m.pattern} className="row">
                <span>{m.found ? "✓" : "–"}</span>
                <span className="role">{m.role}</span>
                <span className="muted small path">{m.found ?? `${m.pattern} (not downloaded)`}</span>
              </div>
            ))}
          </section>

          <p className="muted small path">
            Config: {r.config_file}
            {r.config_error ? ` — ${r.config_error}` : ""}
          </p>
        </>
      )}
    </main>
  );
}
