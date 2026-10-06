import { useCallback, useRef, useState } from "react";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

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

export function useUpdater() {
  const [state, setState] = useState<UpdaterState>({ phase: "idle" });

  const locked = useRef(false);

  const checkForUpdates = useCallback(async () => {
    if (locked.current) return;
    locked.current = true;
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
    } finally {
      locked.current = false;
    }
  }, []);

  const downloadAndInstall = useCallback(async () => {
    if (state.phase !== "available" || locked.current) return;
    locked.current = true;
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
      locked.current = false;
      setState({ phase: "error", message: String(e) });
    }
  }, [state]);

  const restart = useCallback(async () => {
    await relaunch();
  }, []);

  return { state, checkForUpdates, downloadAndInstall, restart };
}

export type Updater = ReturnType<typeof useUpdater>;

export default function UpdaterPanel({ appVer, updater }: { appVer: string; updater: Updater }) {
  const { state, checkForUpdates, downloadAndInstall, restart } = updater;
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
