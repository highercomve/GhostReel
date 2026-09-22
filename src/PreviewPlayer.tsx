import { useEffect, useRef, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import {
  clock,
  enqueueExport,
  enqueuePreview,
  getScriptPreview,
  mediaUrl,
  previewPlan,
  type PlannedSegment,
} from "./api";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";

interface PreviewPlayerProps {
  scriptId: number;
  scriptTitle: string;
  isDirty?: boolean;
  onSaveBeforeAction?: () => Promise<number | null>;
  /** Where playback is, so the timeline lanes can draw a playhead and edit against it. */
  onTime?: (seconds: number) => void;
  /** Hands the parent the controls, so a key pressed over the timeline reaches this player. */
  onControls?: (controls: { seek: (s: number) => void; toggle: () => void }) => void;
}

export default function PreviewPlayer({
  scriptId,
  scriptTitle,
  isDirty = false,
  onSaveBeforeAction,
  onTime,
  onControls,
}: PreviewPlayerProps) {
  const [burnTitles, setBurnTitles] = useState(false);
  const [burnNarration, setBurnNarration] = useState(false);
  const [normalizeAudio, setNormalizeAudio] = useState(false);
  const [previewTaskId, setPreviewTaskId] = useState<number | null>(null);
  const [exportTaskId, setExportTaskId] = useState<number | null>(null);
  const [previewVideoSrc, setPreviewVideoSrc] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlannedSegment[] | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

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
  useEffect(() => {
    onControls?.({
      seek: (s: number) => {
        const v = videoRef.current;
        if (!v) return;
        v.currentTime = Math.max(0, s);
        setCurrentTime(v.currentTime);
        onTime?.(v.currentTime);
      },
      toggle: () => {
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) v.play().catch(() => {});
        else v.pause();
      },
    });
  }, [onControls, onTime, previewVideoSrc]);

  return (
    <div className="card preview-player">
      <div className="card-head">
        <span className="label">Timeline Preview & Export</span>
        <div className="inline">
          <button
            type="button"
            className="ghost small"
            onClick={onExportMp4}
            title="Save the preview (540p, with the burn options below) as an MP4 to send as a demo"
          >
            Export MP4
          </button>
          <button
            type="button"
            className="ghost small"
            onClick={() => onExport("fcp_xml")}
            title="Export Final Cut Pro 7 XML for Adobe Premiere Pro"
          >
            Export for Premiere (FCP XML)
          </button>
          <button
            type="button"
            className="ghost small"
            onClick={() => onExport("otio")}
            title="Export OpenTimelineIO (.otio) interchange format"
          >
            Export .otio
          </button>
        </div>
      </div>

      <div className="preview-controls">
        <label className="checkbox-label small">
          <input
            type="checkbox"
            checked={burnTitles}
            onChange={(e) => setBurnTitles(e.target.checked)}
          />
          Burn titles
        </label>
        <label className="checkbox-label small">
          <input
            type="checkbox"
            checked={burnNarration}
            onChange={(e) => setBurnNarration(e.target.checked)}
          />
          Burn narration
        </label>
        <label
          className="checkbox-label small"
          title="Bring every clip to the same loudness — a lav and a room mic are far apart"
        >
          <input
            type="checkbox"
            checked={normalizeAudio}
            onChange={(e) => setNormalizeAudio(e.target.checked)}
          />
          Even out audio
        </label>
        <button
          type="button"
          onClick={onRenderPreview}
          disabled={rendering || (previewTask != null && previewTask.state === "running")}
        >
          {rendering ? "Enqueueing..." : "Render preview"}
        </button>
      </div>

      {error && <div className="banner bad small">{error}</div>}

      {previewTask && (previewTask.state === "running" || previewTask.state === "queued") && (
        <TaskCard task={previewTask} compact />
      )}

      {exportTask && (
        <div className="export-task-wrap">
          <TaskCard task={exportTask} compact={exportTask.state === "done"} />
          {exportTask.state === "done" && exportTask.output && (
            <div className="good-text small">Exported: {exportTask.output}</div>
          )}
        </div>
      )}

      {previewVideoSrc && (
        <div className="player-wrap">
          <video
            ref={videoRef}
            src={previewVideoSrc}
            controls
            onTimeUpdate={(e) => {
              setCurrentTime(e.currentTarget.currentTime);
              onTime?.(e.currentTarget.currentTime);
            }}
          />
        </div>
      )}

      {plan && plan.length > 0 && (
        <div className="timeline-container">
          <div className="timeline-head muted small">
            <span>Timeline: {plan.length} clips · {totalDuration.toFixed(1)} s</span>
            <span>{clock(currentTime)} / {clock(totalDuration)}</span>
          </div>
          <div className="timeline-bar" title="Click segment to seek">
            {plan.map((item, idx) => {
              const segDur = Math.max(0, item.out_s - item.in_s);
              const widthPct = totalDuration > 0 ? (segDur / totalDuration) * 100 : 0;
              const isCurrent =
                currentTime >= item.timeline_start_s &&
                currentTime < item.timeline_start_s + segDur;
              return (
                <div
                  key={idx}
                  className={`timeline-segment ${isCurrent ? "current" : ""}`}
                  style={{ width: `${Math.max(1, widthPct)}%` }}
                  onClick={() => seekTo(item.timeline_start_s)}
                  title={`Beat ${item.beat_id} · ${clock(item.in_s)}–${clock(item.out_s)} (${segDur.toFixed(1)}s)`}
                >
                  <span className="segment-label">{item.beat_id}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
