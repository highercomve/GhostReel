import { useEffect, useRef, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import {
  ArrowLeftToLine,
  ArrowRightToLine,
  ChevronDown,
  Maximize,
  Pause,
  Play,
  Repeat,
  SkipBack,
  StepBack,
  StepForward,
} from "lucide-react";
import {
  enqueueExport,
  enqueuePreview,
  getScriptPreview,
  mediaUrl,
  previewPlan,
  type PlannedSegment,
  type Script,
  type VideoRow,
} from "./api";
import LivePlayer, { fpsValue, frameClock } from "./editor/LivePlayer";
import { relativeTime, useMinuteNow } from "./editor/relativeTime";
import type { TrackMonitor } from "./editor/audioTracks";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";

interface PreviewPlayerProps {
  scriptId: number;
  scriptTitle: string;
  /** The script as edited, unsaved changes included: Live mode plays this. */
  script: Script;
  videos: Record<number, VideoRow>;
  isDirty?: boolean;
  isAiWorking?: boolean;
  onSaveBeforeAction?: () => Promise<number | null>;
  /** Where playback is, so the timeline lanes can draw a playhead and edit against it. */
  onTime?: (seconds: number) => void;
  /** Hands the parent the controls, so a key pressed over the timeline reaches this player. */
  onControls?: (controls: { seek: (s: number) => void; toggle: () => void }) => void;
  /** Timeline lane mute/solo, passed through to the live viewer. */
  trackMonitor?: TrackMonitor;
}

export default function PreviewPlayer({
  scriptId,
  scriptTitle,
  script,
  videos,
  isDirty = false,
  isAiWorking = false,
  onSaveBeforeAction,
  onTime,
  onControls,
  trackMonitor,
}: PreviewPlayerProps) {
  // Remembered across cuts and sessions: most edits go out the same way every time.
  const [burnTitles, setBurnTitles] = useStoredFlag("ghostreel.burnTitles", false);
  const [burnNarration, setBurnNarration] = useStoredFlag("ghostreel.burnNarration", false);
  const [normalizeAudio, setNormalizeAudio] = useStoredFlag("ghostreel.normalizeAudio", false);
  const [previewTaskId, setPreviewTaskId] = useState<number | null>(null);
  const [exportTaskId, setExportTaskId] = useState<number | null>(null);
  const [previewVideoSrc, setPreviewVideoSrc] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlannedSegment[] | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [mode, setMode] = useState<"live" | "rendered">("live");
  const videoRef = useRef<HTMLVideoElement>(null);
  // The playhead survives a switch between Live and Rendered.
  const lastTime = useRef(0);
  const reportTime = (s: number) => {
    lastTime.current = s;
    onTime?.(s);
  };

  const tasks = useQueue();
  const previewTask =
    previewTaskId != null
      ? tasks.find((t) => t.id === previewTaskId)
      : tasks.find(
          (t) =>
            t.kind.type === "render_preview" &&
            t.kind.script_id === scriptId &&
            (t.state === "running" || t.state === "queued"),
        ) ??
        tasks
          .filter(
            (t) =>
              t.kind.type === "render_preview" &&
              t.kind.script_id === scriptId &&
              t.state === "done" &&
              t.output,
          )
          .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0))[0] ??
        null;

  // The newest finished render of this version: when it was made and with which burn-ins.
  const lastRender =
    tasks
      .filter((t) => t.kind.type === "render_preview" && t.kind.script_id === scriptId && t.state === "done" && t.output)
      .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0))[0] ?? null;

  const exportTask =
    exportTaskId != null
      ? tasks.find((t) => t.id === exportTaskId)
      : tasks.find(
          (t) =>
            t.kind.type === "export" &&
            t.kind.script_id === scriptId &&
            (t.state === "running" || t.state === "queued"),
        ) ??
        tasks
          .filter(
            (t) =>
              t.kind.type === "export" &&
              t.kind.script_id === scriptId &&
              t.state === "done" &&
              t.output,
          )
          .sort((a, b) => (b.finished_at || 0) - (a.finished_at || 0))[0] ??
        null;

  // Load plan and check for existing preview render on disk when script changes
  useEffect(() => {
    setPlan(null);
    setPreviewVideoSrc(null);
    setError(null);
    previewPlan(scriptId)
      .then(setPlan)
      .catch(() => {});

    getScriptPreview(scriptId)
      .then((path) => {
        if (path) {
          mediaUrl(path).then(setPreviewVideoSrc).catch(() => {});
        }
      })
      .catch(() => {});
  }, [scriptId]);

  // When preview task finishes, load video
  useEffect(() => {
    if (!previewTask) return;
    if (previewTask.state === "done" && previewTask.output) {
      setRendering(false);
      mediaUrl(previewTask.output)
        .then((url) => {
          setPreviewVideoSrc(url);
          // Refresh plan for newly rendered script
          previewPlan(scriptId).then(setPlan).catch(() => {});
        })
        .catch((err) => setError(String(err)));
    } else if (previewTask.state === "failed") {
      setRendering(false);
      setError(previewTask.error || "Preview render failed");
    } else if (previewTask.state === "cancelled") {
      setRendering(false);
    }
  }, [previewTask?.state, previewTask?.output, previewTask?.error, scriptId]);

  const onRenderPreview = async () => {
    if (isAiWorking) {
      setError("Sonar AI is drafting an update. Please wait for the turn to complete before rendering.");
      return;
    }
    setError(null);
    setRendering(true);
    let targetScriptId = scriptId;
    if (isDirty && onSaveBeforeAction) {
      const savedId = await onSaveBeforeAction();
      if (savedId == null) {
        setRendering(false);
        return;
      }
      targetScriptId = savedId;
    }

    try {
      const taskId = await enqueuePreview(targetScriptId, burnTitles, burnNarration, normalizeAudio);
      setPreviewTaskId(taskId);
      previewPlan(targetScriptId).then(setPlan).catch(() => {});
    } catch (err) {
      setRendering(false);
      setError(String(err));
    }
  };

  const onExportMp4 = async () => {
    if (isAiWorking) {
      setError("Sonar AI is drafting an update. Please wait for the turn to complete before exporting.");
      return;
    }
    setError(null);
    let targetScriptId = scriptId;
    if (isDirty && onSaveBeforeAction) {
      const savedId = await onSaveBeforeAction();
      if (savedId == null) return;
      targetScriptId = savedId;
    }
    const cleanTitle = scriptTitle.trim().replace(/[^a-zA-Z0-9_\-]/g, "_") || "preview";
    try {
      const path = await save({
        defaultPath: `${cleanTitle}.mp4`,
        filters: [{ name: "MP4 video", extensions: ["mp4"] }],
      });
      if (typeof path === "string") {
        const taskId = await enqueuePreview(targetScriptId, burnTitles, burnNarration, normalizeAudio, path);
        setExportTaskId(taskId);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const onExport = async (format: "fcp_xml" | "otio") => {
    if (isAiWorking) {
      setError("Sonar AI is drafting an update. Please wait for the turn to complete before exporting.");
      return;
    }
    setError(null);
    let targetScriptId = scriptId;
    if (isDirty && onSaveBeforeAction) {
      const savedId = await onSaveBeforeAction();
      if (savedId == null) return;
      targetScriptId = savedId;
    }

    const cleanTitle = scriptTitle.trim().replace(/[^a-zA-Z0-9_\-]/g, "_") || "timeline";
    const extension = format === "fcp_xml" ? "xml" : "otio";
    const filterName = format === "fcp_xml" ? "Final Cut Pro XML" : "OpenTimelineIO";

    try {
      const path = await save({
        defaultPath: `${cleanTitle}.${extension}`,
        filters: [{ name: filterName, extensions: [extension] }],
      });
      if (typeof path === "string") {
        const taskId = await enqueueExport(targetScriptId, format, path);
        setExportTaskId(taskId);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  // Calculate timeline segments
  const totalDuration = (plan ?? []).reduce(
    (acc, item) => acc + Math.max(0, item.out_s - item.in_s),
    0,
  );

  const seekTo = (s: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = s;
      videoRef.current.play().catch(() => {});
    }
  };

  // The timeline seeks without starting playback: you are placing an edge, not watching.
  // In Live mode the LivePlayer hands over its own controls.
  useEffect(() => {
    if (mode !== "rendered") return;
    onControls?.({
      seek: (s: number) => {
        const v = videoRef.current;
        if (!v) return;
        v.currentTime = Math.max(0, s);
        setCurrentTime(v.currentTime);
        lastTime.current = v.currentTime;
        onTime?.(v.currentTime);
      },
      toggle: () => {
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) v.play().catch(() => {});
        else v.pause();
      },
    });
  }, [onControls, onTime, previewVideoSrc, mode]);

  const [playing, setPlaying] = useState(false);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const now = useMinuteNow();
  const fps = fpsValue(script.fps);
  const total = mediaDuration || totalDuration;
  const busy = rendering || (previewTask != null && (previewTask.state === "running" || previewTask.state === "queued"));
  const nudge = (d: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.min(Math.max(0, v.currentTime + d), v.duration || v.currentTime + d);
  };
  // Clip boundaries in the rendered file, from the plan it was rendered from.
  const boundaries = (() => {
    const out = [0];
    let t = 0;
    for (const seg of plan ?? []) {
      t += Math.max(0, seg.out_s - seg.in_s);
      out.push(t);
    }
    return out;
  })();
  const jumpBoundary = (dir: -1 | 1) => {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    const target =
      dir < 0
        ? [...boundaries].reverse().find((b) => b < t - 0.05) ?? 0
        : boundaries.find((b) => b > t + 0.05) ?? (v.duration || t);
    v.currentTime = target;
  };
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  };
  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else stageRef.current?.requestFullscreen().catch(() => {});
  };
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate, previewVideoSrc, mode]);

  // Whether the rendered file still shows this cut. Each saved version renders to its own file,
  // so it goes stale through unsaved edits, or through burn-in options changed since the render.
  const renderKind = lastRender?.kind.type === "render_preview" ? lastRender.kind : null;
  const optionsChanged =
    renderKind != null &&
    (renderKind.burn_titles !== burnTitles ||
      renderKind.burn_narration !== burnNarration ||
      renderKind.normalize_audio !== normalizeAudio);
  const freshness: { tone: "ok" | "stale" | "busy"; text: string; hint: string } = busy
    ? { tone: "busy", text: "Rendering…", hint: "The viewer switches to the new render when it is done" }
    : !previewVideoSrc
      ? { tone: "stale", text: "Not rendered yet", hint: "Render this version to watch it as it will export" }
      : isDirty
        ? { tone: "stale", text: "Out of date · unsaved edits", hint: "Re-render saves a new version and renders it" }
        : optionsChanged
          ? { tone: "stale", text: "Out of date · burn-in changed", hint: "Titles, subtitles or audio levelling changed since this render" }
          : {
              tone: "ok",
              text: lastRender?.finished_at ? `Up to date · rendered ${relativeTime(lastRender.finished_at, now)}` : "Up to date",
              hint: "This render matches the saved version",
            };
  const stale = freshness.tone === "stale";

  return (
    <div className="nle-viewer">
      <div className="nle-viewer-head">
        <span className="nle-panel-title">Viewer</span>
        <div className="lp-mode" role="group" aria-label="Viewer mode">
          <button
            type="button"
            className={mode === "live" ? "active" : ""}
            aria-pressed={mode === "live"}
            onClick={() => setMode("live")}
            title="Play the cut as edited, unsaved changes included, without rendering"
          >
            Live
          </button>
          <button
            type="button"
            className={mode === "rendered" ? "active" : ""}
            aria-pressed={mode === "rendered"}
            onClick={() => setMode("rendered")}
            title="The rendered preview.mp4 of the saved version"
          >
            Rendered
          </button>
        </div>
        {mode === "live" ? (
          <span className="nle-viewer-name" title={scriptTitle}>{scriptTitle}</span>
        ) : (
          <span className="nle-viewer-status">
            <span className={`nle-render-chip ${freshness.tone}`} title={freshness.hint} role="status">
              <span className="nle-render-dot" aria-hidden="true" />
              <span className="nle-render-text">{freshness.text}</span>
            </span>
            {previewVideoSrc && (
              <button
                type="button"
                className={`small ${stale ? "primary" : ""}`}
                onClick={onRenderPreview}
                disabled={busy || isAiWorking}
                title={
                  isAiWorking
                    ? "Rendering waits until Sonar AI finishes"
                    : `Render this version again (titles ${burnTitles ? "on" : "off"}, subtitles ${burnNarration ? "on" : "off"}${normalizeAudio ? ", audio evened out" : ""})`
                }
              >
                {busy ? "Rendering…" : "Re-render"}
              </button>
            )}
          </span>
        )}
        <details className="nle-menu">
          <summary className="small ghost-btn icon-text">
            Export <ChevronDown size={14} strokeWidth={2} aria-hidden="true" />
          </summary>
          <div className="nle-menu-list">
            <div className="nle-menu-heading">MP4 includes</div>
            <label className="checkbox-label small">
              <input type="checkbox" checked={burnTitles} onChange={(e) => setBurnTitles(e.target.checked)} />
              Titles burned in
            </label>
            <label className="checkbox-label small">
              <input type="checkbox" checked={burnNarration} onChange={(e) => setBurnNarration(e.target.checked)} />
              Narration subtitles
            </label>
            <label className="checkbox-label small" title="Bring every clip to the same loudness">
              <input type="checkbox" checked={normalizeAudio} onChange={(e) => setNormalizeAudio(e.target.checked)} />
              Even out audio
            </label>
            <button type="button" onClick={onExportMp4} title="Save the preview (540p) as an MP4 with the options above">
              MP4 video…
            </button>
            <div className="nle-menu-sep" />
            <div className="nle-menu-heading">For another editor (no burn-in)</div>
            <button type="button" onClick={() => onExport("fcp_xml")} title="Final Cut Pro 7 XML, imported by Premiere Pro">
              Premiere (FCP XML)…
            </button>
            <button type="button" onClick={() => onExport("otio")} title="OpenTimelineIO interchange">
              OpenTimelineIO (.otio)…
            </button>
          </div>
        </details>
      </div>

      {mode === "live" ? (
        <>
          <LivePlayer
            script={script}
            videos={videos}
            initialTime={lastTime.current}
            onTime={reportTime}
            onControls={onControls}
            showTitles={burnTitles}
            onShowTitles={setBurnTitles}
            showNarration={burnNarration}
            onShowNarration={setBurnNarration}
            trackMonitor={trackMonitor}
          />
          {error && <div className="banner bad small">{error}</div>}
          {exportTask && (
            <div className="export-task-wrap">
              <TaskCard task={exportTask} compact={exportTask.state === "done"} />
              {exportTask.state === "done" && exportTask.output && (
                <div className="good-text small">Exported: {exportTask.output}</div>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <div className="nle-viewer-stage" ref={stageRef}>
            {previewVideoSrc ? (
              <video
                ref={videoRef}
                src={previewVideoSrc}
                loop={loop}
                onClick={togglePlay}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  setMediaDuration(v.duration || 0);
                  v.playbackRate = rate;
                  if (lastTime.current > 0) v.currentTime = Math.min(lastTime.current, v.duration || lastTime.current);
                }}
                onTimeUpdate={(e) => {
                  setCurrentTime(e.currentTarget.currentTime);
                  reportTime(e.currentTarget.currentTime);
                }}
              />
            ) : (
              <div className="nle-viewer-empty">
                <div className="nle-viewer-empty-icon">
                  <Play size={30} strokeWidth={1.5} aria-hidden="true" />
                </div>
                <div>{isDirty ? "Unsaved changes — render to see them" : "No preview rendered for this version yet"}</div>
                <button type="button" className="primary" onClick={onRenderPreview} disabled={busy || isAiWorking}>
                  {busy ? "Rendering…" : "Render preview"}
                </button>
              </div>
            )}
            {previewTask && (previewTask.state === "running" || previewTask.state === "queued") && (
              <div className="nle-viewer-task">
                <TaskCard task={previewTask} compact />
              </div>
            )}
          </div>

          {error && <div className="banner bad small">{error}</div>}
          {exportTask && (
            <div className="export-task-wrap">
              <TaskCard task={exportTask} compact={exportTask.state === "done"} />
              {exportTask.state === "done" && exportTask.output && (
                <div className="good-text small">Exported: {exportTask.output}</div>
              )}
            </div>
          )}

          <div className="nle-transport lp-transport">
            <div className="nle-transport-buttons">
              <button type="button" className="nle-tbtn" onClick={() => seekTo(0)} disabled={!previewVideoSrc} title="Go to start" aria-label="Go to start"><SkipBack size={15} strokeWidth={2} aria-hidden="true" /></button>
              <button type="button" className="nle-tbtn" onClick={() => jumpBoundary(-1)} disabled={!previewVideoSrc} title="Previous clip boundary" aria-label="Previous clip boundary"><ArrowLeftToLine size={15} strokeWidth={2} aria-hidden="true" /></button>
              <button type="button" className="nle-tbtn" onClick={() => nudge(-1 / fps)} disabled={!previewVideoSrc} title="Back one frame" aria-label="Back one frame"><StepBack size={15} strokeWidth={2} aria-hidden="true" /></button>
              <button type="button" className="nle-tbtn play" disabled={!previewVideoSrc} onClick={togglePlay} title="Play / pause (Space on the timeline)" aria-label={playing ? "Pause" : "Play"}>
                {playing ? <Pause size={15} strokeWidth={2} aria-hidden="true" /> : <Play size={15} strokeWidth={2} aria-hidden="true" />}
              </button>
              <button type="button" className="nle-tbtn" onClick={() => nudge(1 / fps)} disabled={!previewVideoSrc} title="Forward one frame" aria-label="Forward one frame"><StepForward size={15} strokeWidth={2} aria-hidden="true" /></button>
              <button type="button" className="nle-tbtn" onClick={() => jumpBoundary(1)} disabled={!previewVideoSrc} title="Next clip boundary" aria-label="Next clip boundary"><ArrowRightToLine size={15} strokeWidth={2} aria-hidden="true" /></button>
              <span className="nle-timecode lp-timecode">
                {frameClock(currentTime, fps)} <span className="muted">/ {frameClock(total, fps)}</span>
              </span>
            </div>
            <div className="nle-transport-render">
              <select
                className="lp-rate"
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
                disabled={!previewVideoSrc}
                title="Playback speed"
                aria-label="Playback speed"
              >
                {[0.25, 0.5, 1, 1.5, 2].map((r) => (
                  <option key={r} value={r}>
                    {r}×
                  </option>
                ))}
              </select>
              <button type="button" className={`nle-tbtn lp-toggle ${loop ? "on" : ""}`} aria-pressed={loop} onClick={() => setLoop((l) => !l)} title="Loop the render" aria-label="Loop">
                <Repeat size={15} strokeWidth={2} aria-hidden="true" />
              </button>
              <button
                type="button"
                className={`nle-tbtn lp-toggle ${burnTitles ? "on" : ""}`}
                aria-pressed={burnTitles}
                onClick={() => setBurnTitles(!burnTitles)}
                title={burnTitles ? "Titles burned in. Click to leave them out of the next render." : "Titles left out. Click to burn them into the next render."}
              >
                T
              </button>
              <button
                type="button"
                className={`nle-tbtn lp-toggle ${burnNarration ? "on" : ""}`}
                aria-pressed={burnNarration}
                onClick={() => setBurnNarration(!burnNarration)}
                title={burnNarration ? "Narration subtitles burned in. Click to leave them out of the next render." : "No narration subtitles. Click to burn them into the next render."}
              >
                CC
              </button>
              <button type="button" className="nle-tbtn" onClick={toggleFullscreen} disabled={!previewVideoSrc} title="Fullscreen" aria-label="Fullscreen">
                <Maximize size={15} strokeWidth={2} aria-hidden="true" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function useStoredFlag(key: string, initial: boolean): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(() => {
    const v = localStorage.getItem(key);
    return v == null ? initial : v === "1";
  });
  const set = (v: boolean) => {
    localStorage.setItem(key, v ? "1" : "0");
    setValue(v);
  };
  return [value, set];
}
