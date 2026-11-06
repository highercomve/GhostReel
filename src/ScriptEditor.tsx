import { formatDuration as clock, formatDurationText } from "./timeFormat";
import MediaImage from "./MediaImage";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from "react";
import MediaBin, { type Pick } from "./editor/MediaBin";
import AudioTracksSection from "./editor/AudioTracksSection";
import { cachedAudioTracks, toggleTrack, type TrackMonitor } from "./editor/audioTracks";
import TimecodeField from "./editor/TimecodeField";
import ScrollPanel from "./editor/ScrollPanel";
import DurationField from "./editor/DurationField";
import { fpsValue } from "./editor/timecode";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ChevronRight,
  Film,
  Keyboard,
  Mic,
  Plus,
  PanelLeft,
  PanelRight,
  Redo2,
  Replace,
  Sparkles,
  Trash2,
  Type,
  Undo2,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import "./editor/inspector.css";

export interface ScriptEditorHandle {
  save: () => Promise<number | null>;
  isDirty: () => boolean;
  resetDirty: () => void;
  getScript: () => Script | null;
}
import {
  fileName,
  getScript,
  projectView,
  saveScript,
  videoFrames,
  videoTranscript,
  type Beat,
  type FrameRow,
  type Issue,
  type Script,
  type ScriptClip,
  type TranscriptSegment,
  type VideoRow,
} from "./api";
import PreviewPlayer from "./PreviewPlayer";
import ScriptTimeline from "./ScriptTimeline";
import GridSplitters from "./editor/GridSplitters";

interface ScriptEditorProps {
  projectId: number;
  scriptId: number;
  sessionId: number | null;
  sessionTitle?: string;
  latestIssues?: Issue[];
  isAiWorking?: boolean;
  onScriptSaved: (newScriptId: number, issues: Issue[]) => void;
  /** The project's cuts, shown in the left panel's first tab. */
  cutsPanel?: ReactNode;
}

/** The shortest a clip may be, as on the timeline. */
const MIN_CLIP = 0.2;

const ScriptEditor = forwardRef<ScriptEditorHandle, ScriptEditorProps>(function ScriptEditor(
  {
    projectId,
    scriptId,
    sessionId,
    sessionTitle,
    latestIssues,
    isAiWorking,
    onScriptSaved,
    cutsPanel,
  },
  ref
) {
  const [script, setScriptRaw] = useState<Script | null>(null);
  // Undo history: snapshots before each edit. Keystrokes in one field within a moment are one step.
  const past = useRef<Script[]>([]);
  const future = useRef<Script[]>([]);
  const lastPush = useRef(0);
  const [, setHistoryTick] = useState(0);
  const [showKeys, setShowKeys] = useState(false);
  const setScript = (next: Script) => {
    const now = Date.now();
    if (script && now - lastPush.current > 600) {
      past.current.push(script);
      if (past.current.length > 200) past.current.shift();
    }
    lastPush.current = now;
    future.current = [];
    setScriptRaw(next);
    setHistoryTick((t) => t + 1);
  };
  const undo = () => {
    const prev = past.current.pop();
    if (!prev || !script) return;
    future.current.push(script);
    lastPush.current = 0;
    setScriptRaw(prev);
    setHistoryTick((t) => t + 1);
  };
  const redo = () => {
    const next = future.current.pop();
    if (!next || !script) return;
    past.current.push(script);
    lastPush.current = 0;
    setScriptRaw(next);
    setHistoryTick((t) => t + 1);
  };
  const [initialJson, setInitialJson] = useState<string | null>(null);
  const [loadedSessionId, setLoadedSessionId] = useState<number | null>(null);
  const [version, setVersion] = useState<number | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Video cache & frame cache
  const [videosMap, setVideosMap] = useState<Record<number, VideoRow>>({});
  const [framesMap, setFramesMap] = useState<Record<number, FrameRow[]>>({});
  const requestedVideos = useRef<Set<number>>(new Set());

  // Which block of the timeline is selected, so the lane and the beat below agree.
  const [selectedBlock, setSelectedBlock] = useState<string | null>(null);
  // The preview player's position, and its controls, so a key over the timeline reaches it.
  const [playhead, setPlayhead] = useState(0);
  const playerRef = useRef<{ seek: (s: number) => void; toggle: () => void } | null>(null);
  // Sentences per video, fetched once, for snapping an edge to where someone stops talking.
  const transcriptCache = useRef<Record<number, TranscriptSegment[]>>({});
  // Stable, and declared with the other hooks: the JSX below sits after two early returns, so a
  // hook called down there would change the hook order while the script is still loading.
  const handleControls = useCallback((c: { seek: (s: number) => void; toggle: () => void }) => {
    playerRef.current = c;
  }, []);

  const [compactPanel, setCompactPanel] = useState<"left" | "inspector" | null>(null);
  const [leftTab, setLeftTab] = useState<"cuts" | "beats" | "media">("beats");
  // The inspector tab someone picked; a new selection goes back to the tab that fits it.
  const [inspPick, setInspPick] = useState<"clip" | "beat" | "cut" | null>(null);
  useEffect(() => setInspPick(null), [selectedBlock]);
  // Timeline lane mute/solo: what the live viewer lets you hear. Not saved.
  const [trackMonitor, setTrackMonitor] = useState<TrackMonitor>({ muted: [], solo: [] });

  // Load project videos map for video names
  useEffect(() => {
    requestedVideos.current.clear();
    setFramesMap({});
    projectView(projectId)
      .then((pv) => {
        const map: Record<number, VideoRow> = {};
        for (const v of pv.videos) {
          map[v.id] = v;
        }
        setVideosMap(map);
      })
      .catch(() => {});
  }, [projectId]);

  // Load script
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getScript(scriptId)
      .then((view) => {
        if (cancelled) return;
        setScriptRaw(view.stored.script);
        past.current = [];
        future.current = [];
        setInitialJson(JSON.stringify(view.stored.script));
        setLoadedSessionId(view.stored.session_id);
        setVersion(view.stored.version);
        setIssues(view.issues);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(String(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [scriptId]);

  // Update issues when latestIssues changes without clobbering dirty script state
  useEffect(() => {
    if (latestIssues) {
      setIssues(latestIssues);
    }
  }, [latestIssues]);

  // Request frames for video
  const requestFrames = useCallback((videoId: number) => {
    if (requestedVideos.current.has(videoId)) return;
    requestedVideos.current.add(videoId);
    videoFrames(videoId)
      .then((rows) => setFramesMap((prev) => ({ ...prev, [videoId]: rows })))
      .catch(() => setFramesMap((prev) => ({ ...prev, [videoId]: [] })));
  }, []);

  // Request frames for all clips in the script
  useEffect(() => {
    if (!script) return;
    for (const beat of script.beats) {
      for (const clip of beat.clips) {
        requestFrames(clip.video_id);
      }
    }
  }, [script, requestFrames]);

  const getNearestFrame = (videoId: number, in_s: number): FrameRow | null => {
    const list = framesMap[videoId];
    if (!list || list.length === 0) return null;
    let closest = list[0];
    let minDiff = Math.abs(list[0].t_s - in_s);
    for (let i = 1; i < list.length; i++) {
      const diff = Math.abs(list[i].t_s - in_s);
      if (diff < minDiff) {
        minDiff = diff;
        closest = list[i];
      }
    }
    return closest;
  };

  const isDirty =
    script != null && initialJson != null && JSON.stringify(script) !== initialJson;

  const handleSave = async (): Promise<number | null> => {
    if (!script) return null;
    setSaving(true);
    setError(null);
    try {
      const res = await saveScript(projectId, script, loadedSessionId ?? sessionId);
      setInitialJson(JSON.stringify(script));
      setIssues(res.issues);
      onScriptSaved(res.script_id, res.issues);
      return res.script_id;
    } catch (err) {
      setError(String(err));
      return null;
    } finally {
      setSaving(false);
    }
  };

  useImperativeHandle(
    ref,
    () => ({
      save: handleSave,
      isDirty: () => isDirty,
      resetDirty: () => setInitialJson(JSON.stringify(script)),
      getScript: () => script,
    }),
    [handleSave, isDirty, script]
  );

  // Beat edits
  const updateBeat = (idx: number, patch: Partial<Beat>) => {
    if (!script) return;
    const newBeats = [...script.beats];
    newBeats[idx] = { ...newBeats[idx], ...patch };
    setScript({ ...script, beats: newBeats });
  };

  const moveBeat = (idx: number, dir: -1 | 1) => {
    if (!script) return;
    const target = idx + dir;
    if (target < 0 || target >= script.beats.length) return;
    const newBeats = [...script.beats];
    const temp = newBeats[idx];
    newBeats[idx] = newBeats[target];
    newBeats[target] = temp;
    setScript({ ...script, beats: newBeats });
  };

  const removeBeat = (idx: number) => {
    if (!script) return;
    const newBeats = script.beats.filter((_, i) => i !== idx);
    setScript({ ...script, beats: newBeats });
  };

  const addBeat = () => {
    if (!script) return;
    const nextId = `b${script.beats.length + 1}`;
    const newBeats = [
      ...script.beats,
      {
        id: nextId,
        purpose: "new beat",
        clips: [],
      },
    ];
    setScript({ ...script, beats: newBeats });
  };

  // Clip edits
  const updateClip = (bIdx: number, cIdx: number, patch: Partial<ScriptClip>) => {
    if (!script) return;
    const beat = script.beats[bIdx];
    const newClips = [...beat.clips];
    newClips[cIdx] = { ...newClips[cIdx], ...patch };
    updateBeat(bIdx, { clips: newClips });
  };

  const moveClip = (bIdx: number, cIdx: number, dir: -1 | 1) => {
    if (!script) return;
    const beat = script.beats[bIdx];
    const target = cIdx + dir;
    if (target < 0 || target >= beat.clips.length) return;
    const newClips = [...beat.clips];
    const temp = newClips[cIdx];
    newClips[cIdx] = newClips[target];
    newClips[target] = temp;
    updateBeat(bIdx, { clips: newClips });
  };

  const removeClip = (bIdx: number, cIdx: number) => {
    if (!script) return;
    const beat = script.beats[bIdx];
    const newClips = beat.clips.filter((_, i) => i !== cIdx);
    updateBeat(bIdx, { clips: newClips });
  };

  if (loading) {
    return <div className="card muted">Loading script...</div>;
  }

  if (!script) {
    return <div className="card muted">No script loaded.</div>;
  }

  const totalDuration = script.beats
    .flatMap((b) => b.clips)
    .reduce((sum, c) => sum + Math.max(0, c.out_s - c.in_s), 0);

  const targetDuration = script.target_duration_s ?? 0;
  const isOverTarget = targetDuration > 0 && totalDuration > targetDuration;
  const pctOfTarget = targetDuration > 0 ? (totalDuration / targetDuration) * 100 : 0;

  const errors = issues.filter((i) => i.severity === "error");
  const warnings = issues.filter((i) => i.severity === "warning");
  const infos = issues.filter((i) => i.severity === "info" && !/^total duration\b/i.test(i.message));
  const feedbackCount = errors.length + warnings.length + infos.length;

  // What the timeline has selected, read back as a beat and maybe a clip.
  const selParts = selectedBlock?.split("-") ?? [];
  const selKind = selParts[0] ?? null;
  const selBeatRaw = selParts.length > 1 ? Number(selParts[1]) : null;
  const selBeat = selBeatRaw != null && script.beats[selBeatRaw] ? selBeatRaw : null;
  // A clip's own-audio block selects the clip, so its mute switch is right there.
  const selClipRaw = selKind === "p" || selKind === "a" ? Number(selParts[2]) : null;
  const selClip =
    selBeat != null && selClipRaw != null && script.beats[selBeat].clips[selClipRaw] ? selClipRaw : null;
  const beat = selBeat != null ? script.beats[selBeat] : null;
  const clip = beat && selClip != null ? beat.clips[selClip] : null;

  const inspTab: "clip" | "beat" | "cut" =
    inspPick === "clip" && clip ? "clip"
    : inspPick === "beat" && beat ? "beat"
    : inspPick === "cut" ? "cut"
    : clip ? "clip"
    : beat ? "beat"
    : "cut";
  const fps = fpsValue(script.fps);

  const selectBeat = (bIdx: number) => setSelectedBlock(`beat-${bIdx}`);
  const beatDuration = (b: Beat) => b.clips.reduce((sum, c) => sum + Math.max(0, c.out_s - c.in_s), 0);
  const issueTone = (beatId: string) => {
    const mine = issues.filter((i) => i.beat_id === beatId);
    if (mine.some((i) => i.severity === "error")) return "bad";
    if (mine.some((i) => i.severity === "warning")) return "warn";
    return null;
  };

  const addFromBin = (pick: Pick) => {
    if (selBeat == null) return;
    const clips: ScriptClip[] = [...script.beats[selBeat].clips, { ...pick, audio: "source" }];
    updateBeat(selBeat, { clips });
    setSelectedBlock(`p-${selBeat}-${clips.length - 1}`);
    requestFrames(pick.video_id);
  };
  const replaceFromBin = (pick: Pick) => {
    if (selBeat == null || selClip == null) return;
    updateClip(selBeat, selClip, pick);
    requestFrames(pick.video_id);
  };

  const moveClipTo = (from: { beat: number; clip: number }, to: { beat: number; clip: number }) => {
    const beats = script.beats.map((b) => ({ ...b, clips: [...b.clips] }));
    const moved = beats[from.beat]?.clips[from.clip];
    if (!moved || !beats[to.beat]) return;
    beats[from.beat].clips.splice(from.clip, 1);
    // Dropping later in the same beat: the removal shifted everything after it by one.
    const at = from.beat === to.beat && from.clip < to.clip ? to.clip - 1 : to.clip;
    const idx = Math.max(0, Math.min(at, beats[to.beat].clips.length));
    beats[to.beat].clips.splice(idx, 0, moved);
    setScript({ ...script, beats });
    setSelectedBlock(`p-${to.beat}-${idx}`);
  };
  const splitClip = (bIdx: number, cIdx: number, at: number) => {
    const c = script.beats[bIdx]?.clips[cIdx];
    // Both halves must stay long enough to see.
    if (!c || at <= c.in_s + 0.1 || at >= c.out_s - 0.1) return;
    const clips = [...script.beats[bIdx].clips];
    clips.splice(cIdx, 1, { ...c, out_s: at }, { ...c, in_s: at });
    updateBeat(bIdx, { clips });
    setSelectedBlock(`p-${bIdx}-${cIdx + 1}`);
  };
  // `m` on an audio block: flip that source track for the clip (or the bed when `cIdx` is null).
  const toggleTrackAt = (bIdx: number, cIdx: number | null, track: number) => {
    const b = script.beats[bIdx];
    if (!b) return;
    if (cIdx == null) {
      if (!b.bed) return;
      updateBeat(bIdx, { bed: { ...b.bed, tracks: toggleTrack(b.bed.tracks, cachedAudioTracks(b.bed.video_id), track) } });
      return;
    }
    const c = b.clips[cIdx];
    if (!c) return;
    // On a muted clip the key means "let me hear this track".
    if (c.audio === "mute") updateClip(bIdx, cIdx, { audio: "source", tracks: [track] });
    else updateClip(bIdx, cIdx, { tracks: toggleTrack(c.tracks, cachedAudioTracks(c.video_id), track) });
  };
  const deleteAt = (bIdx: number, cIdx: number | null) => {
    if (cIdx == null) updateBeat(bIdx, { bed: null });
    else removeClip(bIdx, cIdx);
    selectBeat(bIdx);
  };
  const onEditorKeys = (e: React.KeyboardEvent) => {
    const t = e.target as HTMLElement;
    // Text fields keep their own undo.
    if (t.closest("input, textarea, select, [contenteditable=true]")) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (isAiWorking) return;
      if (e.shiftKey) redo();
      else undo();
    } else if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      if (!isAiWorking) redo();
    } else if (mod && e.key.toLowerCase() === "s") {
      e.preventDefault();
      if (isDirty && !isAiWorking) void handleSave();
    } else if (e.key === "?") {
      setShowKeys((v) => !v);
    }
  };

  const issueRow = (iss: Issue, i: number) => (
    <button
      type="button"
      key={i}
      className={`nle-issue ${iss.severity}`}
      onClick={() => {
        const bIdx = script.beats.findIndex((b) => b.id === iss.beat_id);
        if (bIdx < 0) return;
        if (iss.clip_index != null && script.beats[bIdx].clips[iss.clip_index]) {
          setSelectedBlock(`p-${bIdx}-${iss.clip_index}`);
        } else {
          selectBeat(bIdx);
        }
      }}
    >
      <span className={`pill ${iss.severity === "error" ? "bad" : iss.severity === "warning" ? "warn" : "info"}`}>
        {iss.severity}
      </span>
      <span>
        {iss.beat_id && <strong>{iss.beat_id} </strong>}
        {iss.clip_index != null && `clip ${iss.clip_index + 1}: `}
        {formatDurationText(iss.message)}
      </span>
    </button>
  );

  const locked = isAiWorking ? "editor-locked" : "";

  return (
    <div className="nle-grid" data-compact-panel={compactPanel ?? "none"} onKeyDown={onEditorKeys}>
      <div className="nle-compact-panels" role="group" aria-label="Editor panels">
        <button type="button" className="ghost small" aria-pressed={compactPanel === "left"}
          onClick={() => setCompactPanel(compactPanel === "left" ? null : "left")}>
          <PanelLeft size={14} strokeWidth={2} aria-hidden /> Cuts, beats & media
        </button>
        <button type="button" className="ghost small" aria-pressed={compactPanel === "inspector"}
          onClick={() => setCompactPanel(compactPanel === "inspector" ? null : "inspector")}>
          <PanelRight size={14} strokeWidth={2} aria-hidden /> Inspector
        </button>
      </div>
      <GridSplitters />
      {/* Left: the project's cuts, this cut's beats, and the footage bin */}
      <aside className="nle-panel nle-left">
        <div className="nle-panel-tabs" role="tablist">
          {(["cuts", "beats", "media"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={leftTab === t}
              className={`nle-panel-tab ${leftTab === t ? "active" : ""}`}
              onClick={() => setLeftTab(t)}
            >
              {t === "cuts" ? "Cuts" : t === "beats" ? `Beats ${script.beats.length}` : "Media"}
            </button>
          ))}
        </div>
        <div className="nle-panel-body">
          {leftTab === "cuts" && cutsPanel}
          {leftTab === "beats" && (
            <div className={`nle-beats ${locked}`}>
              <div className="nle-panel-tools">
                <span className="small muted">Story order</span>
                <button type="button" className="small" onClick={addBeat} title="Add a beat at the end of the story">
                  <Plus size={14} strokeWidth={2} aria-hidden /> Beat
                </button>
              </div>
              {script.beats.length === 0 && <div className="muted small nle-panel-empty">No beats yet.</div>}
              <ol className="nle-beat-list">
                {script.beats.map((b, bIdx) => {
                  const tone = issueTone(b.id);
                  const n = b.clips.length;
                  return (
                    <li key={b.id || bIdx} className={selBeat === bIdx ? "active" : ""}>
                      <button
                        type="button"
                        className="nle-beat-row"
                        onClick={() => selectBeat(bIdx)}
                        title={`${b.id}: ${b.purpose || "untitled"}`}
                      >
                        <span className="nle-beat-id">{b.id}</span>
                        <span className="nle-beat-purpose">{b.purpose || "untitled"}</span>
                        <span className="nle-beat-meta">
                          {tone && <span className={`nle-dot ${tone}`} title={tone === "bad" ? "Has errors" : "Has warnings"} />}
                          {n} {n === 1 ? "clip" : "clips"} · {clock(beatDuration(b))}
                        </span>
                      </button>
                      <span className="nle-beat-tools">
                        <button
                          type="button"
                          disabled={bIdx === 0}
                          onClick={() => moveBeat(bIdx, -1)}
                          title="Move this beat up"
                          aria-label="Move beat up"
                        >
                          <ArrowUp size={14} strokeWidth={2} aria-hidden />
                        </button>
                        <button
                          type="button"
                          disabled={bIdx === script.beats.length - 1}
                          onClick={() => moveBeat(bIdx, 1)}
                          title="Move this beat down"
                          aria-label="Move beat down"
                        >
                          <ArrowDown size={14} strokeWidth={2} aria-hidden />
                        </button>
                        <button
                          type="button"
                          className="danger"
                          onClick={() => {
                            removeBeat(bIdx);
                            setSelectedBlock(null);
                          }}
                          title="Delete this beat and its clips (Ctrl+Z undoes)"
                          aria-label="Delete beat"
                        >
                          <Trash2 size={14} strokeWidth={2} aria-hidden />
                        </button>
                      </span>
                    </li>
                  );
                })}
              </ol>
            </div>
          )}
          {leftTab === "media" && (
            <MediaBin
              projectId={projectId}
              videos={videosMap}
              beatLabel={beat ? beat.id : null}
              canReplace={clip != null}
              disabled={isAiWorking}
              onAdd={addFromBin}
              onReplace={replaceFromBin}
            />
          )}
        </div>
      </aside>

      {/* Center: the viewer */}
      <section className="nle-panel nle-center">
        <PreviewPlayer
          scriptId={scriptId}
          scriptTitle={script.title}
          isDirty={isDirty}
          isAiWorking={isAiWorking}
          onSaveBeforeAction={handleSave}
          script={script}
          videos={videosMap}
          onTime={setPlayhead}
          onControls={handleControls}
          trackMonitor={trackMonitor}
        />
      </section>

      {/* Right: the inspector, as Clip / Beat / Cut tabs; a selection opens the tab that fits it */}
      <aside className={`nle-panel nle-inspector ${locked}`}>
        <div className="nle-panel-tabs insp-tabs" role="tablist" aria-label="Inspector">
          {(["clip", "beat", "cut"] as const).map((t) => {
            const available = t === "clip" ? clip != null : t === "beat" ? beat != null : true;
            return (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={inspTab === t}
                disabled={!available}
                className={`nle-panel-tab ${inspTab === t ? "active" : ""}`}
                onClick={() => setInspPick(t)}
                title={
                  available
                    ? undefined
                    : t === "clip"
                      ? "Select a clip on the timeline"
                      : "Select a beat in Beats or on the timeline"
                }
              >
                {t === "clip" ? "Clip" : t === "beat" ? "Beat" : "Cut"}
              </button>
            );
          })}
        </div>
        <ScrollPanel className="nle-panel-body nle-inspector-body" label="Inspector">
          {inspTab === "clip" && clip && beat && selBeat != null && selClip != null && (() => {
            const len = Math.max(0, clip.out_s - clip.in_s);
            const clipStart =
              script.beats.slice(0, selBeat).reduce((t, b) => t + beatDuration(b), 0) +
              beat.clips.slice(0, selClip).reduce((t, c) => t + Math.max(0, c.out_s - c.in_s), 0);
            const underPlayhead = playhead >= clipStart - 1e-6 && playhead <= clipStart + len + 1e-6;
            const playheadSource = underPlayhead ? clip.in_s + (playhead - clipStart) : null;
            const maxOut = videosMap[clip.video_id]?.duration_s ?? Number.MAX_SAFE_INTEGER;
            const soundOn = clip.audio !== "mute";
            const file = videosMap[clip.video_id] ? fileName(videosMap[clip.video_id].path) : `Video #${clip.video_id}`;
            return (
              <section className="nle-insp-section">
                <button
                  type="button"
                  className="ghost small insp-crumb"
                  onClick={() => setInspPick("beat")}
                  title="Open this clip's beat"
                >
                  <span className="insp-crumb-id">{beat.id}</span>
                  <span className="insp-crumb-text" title={beat.purpose}>{beat.purpose || "untitled beat"}</span>
                  <span className="insp-crumb-pos">
                    clip {selClip + 1} of {beat.clips.length}
                  </span>
                  <ChevronRight size={14} strokeWidth={2} aria-hidden />
                </button>
                <div className="nle-insp-actions">
                  <button type="button" className="small" onClick={() => { setLeftTab("media"); setCompactPanel("left"); }} title="Pick a replacement in the Media bin">
                    <Replace size={14} strokeWidth={2} aria-hidden /> Replace…
                  </button>
                  <span className="insp-move">
                    <button
                      type="button"
                      className="small"
                      disabled={selClip === 0}
                      onClick={() => {
                        moveClip(selBeat, selClip, -1);
                        setSelectedBlock(`p-${selBeat}-${selClip - 1}`);
                      }}
                      title="Move earlier in the beat (or drag it on the timeline)"
                    >
                      <ArrowLeft size={14} strokeWidth={2} aria-hidden /> Earlier
                    </button>
                    <button
                      type="button"
                      className="small"
                      disabled={selClip === beat.clips.length - 1}
                      onClick={() => {
                        moveClip(selBeat, selClip, 1);
                        setSelectedBlock(`p-${selBeat}-${selClip + 1}`);
                      }}
                      title="Move later in the beat (or drag it on the timeline)"
                    >
                      Later <ArrowRight size={14} strokeWidth={2} aria-hidden />
                    </button>
                  </span>
                  <button
                    type="button"
                    className="small danger"
                    disabled={isAiWorking}
                    onClick={() => {
                      removeClip(selBeat, selClip);
                      selectBeat(selBeat);
                    }}
                    title="Remove this clip from the cut (Ctrl+Z undoes)"
                  >
                    <Trash2 size={14} strokeWidth={2} aria-hidden /> Remove
                  </button>
                </div>
                <div className="nle-insp-file" title={videosMap[clip.video_id]?.path ?? ""}>
                  <Film size={13} strokeWidth={2} aria-hidden /> {file}
                </div>
                <div className="insp-timecodes">
                  <TimecodeField
                    label="In"
                    value={clip.in_s}
                    fps={fps}
                    playheadValue={playheadSource}
                    playheadKey="i"
                    disabled={isAiWorking}
                    onChange={(v) => updateClip(selBeat, selClip, { in_s: Math.min(Math.max(0, v), clip.out_s - MIN_CLIP) })}
                  />
                  <TimecodeField
                    label="Out"
                    value={clip.out_s}
                    fps={fps}
                    playheadValue={playheadSource}
                    playheadKey="o"
                    disabled={isAiWorking}
                    onChange={(v) => updateClip(selBeat, selClip, { out_s: Math.max(clip.in_s + MIN_CLIP, Math.min(v, maxOut)) })}
                  />
                  <div className="insp-tc-facts">
                    <span title="How long the clip runs">
                      Length <strong>{clock(len)}</strong>
                    </span>
                    <span title="Where the clip sits in the finished cut">
                      In the cut <strong>{clock(clipStart)}–{clock(clipStart + len)}</strong>
                    </span>
                  </div>
                </div>

                <div className="nle-insp-subtitle">Sound</div>
                <div className="insp-switch-row">
                  <span className="insp-switch-label">
                    {soundOn ? <Volume2 size={14} strokeWidth={2} aria-hidden /> : <VolumeX size={14} strokeWidth={2} aria-hidden />}
                    Clip sound
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={soundOn}
                    aria-label="Clip sound"
                    className={`insp-switch ${soundOn ? "on" : ""}`}
                    disabled={isAiWorking}
                    onClick={() => updateClip(selBeat, selClip, { audio: soundOn ? "mute" : "source" })}
                    title={soundOn ? "On: the clip plays its own sound. Click to mute it." : "Muted: click to play the clip's own sound."}
                  >
                    <span className="insp-switch-knob" />
                  </button>
                  <span className={`insp-switch-state ${soundOn ? "" : "off"}`}>{soundOn ? "On" : "Muted"}</span>
                </div>
                <div className="small muted">{beat.bed ? "The beat’s sound bed plays instead of clip sound." : soundOn ? "Plays the selected source tracks." : "Muted: source tracks are off."}</div>
                <div className="nle-insp-subtitle">Audio tracks</div>
                <AudioTracksSection
                  videoId={clip.video_id}
                  inS={clip.in_s}
                  outS={clip.out_s}
                  tracks={clip.tracks}
                  muted={clip.audio === "mute" || beat.bed != null}
                  disabled={isAiWorking}
                  onChange={(tracks) => updateClip(selBeat, selClip, { tracks })}
                />

                <details className="insp-source-image">
                  <summary>Source image</summary>
                <div className="nle-insp-thumb">
                  {(() => {
                    const frame = getNearestFrame(clip.video_id, clip.in_s);
                    return frame ? <MediaImage path={frame.path} alt="" loading="lazy" /> : <div className="noframe" />;
                  })()}
                </div>
                </details>

                <label className="nle-insp-field">
                  Why this clip
                  <AutoGrowTextarea
                    value={clip.why ?? ""}
                    onChange={(v) => updateClip(selBeat, selClip, { why: v })}
                    placeholder="What it does for the story…"
                  />
                </label>

              </section>
            );
          })()}

          {inspTab === "beat" && beat && selBeat != null && (
            <section className="nle-insp-section">
              <div className="insp-beat-head">
                <span className="insp-crumb-id">{beat.id}</span>
                <span className="small muted">
                  Beat {selBeat + 1} of {script.beats.length} · {beat.clips.length} {beat.clips.length === 1 ? "clip" : "clips"} ·{" "}
                  {clock(beatDuration(beat))}
                </span>
              </div>
              <label className="nle-insp-field">
                Purpose
                <AutoGrowTextarea value={beat.purpose} onChange={(v) => updateBeat(selBeat, { purpose: v })} />
              </label>
              <label className="nle-insp-field insp-field-narration">
                <span className="insp-field-label">
                  <Mic size={13} strokeWidth={2} aria-hidden /> Narration
                  <span className="insp-field-hint">spoken voice-over</span>
                </span>
                <textarea
                  rows={3}
                  value={beat.narration ?? ""}
                  onChange={(e) => updateBeat(selBeat, { narration: e.target.value })}
                  placeholder="Words spoken over this beat…"
                />
              </label>
              <label className="nle-insp-field insp-field-title">
                <span className="insp-field-label">
                  <Type size={13} strokeWidth={2} aria-hidden /> On-screen text
                  <span className="insp-field-hint">Titles lane</span>
                </span>
                <input
                  value={beat.on_screen_text ?? ""}
                  onChange={(e) => updateBeat(selBeat, { on_screen_text: e.target.value })}
                  placeholder="Title or lower third…"
                />
              </label>
              <div className="nle-insp-row">
                <button type="button" className="small" onClick={() => setLeftTab("media")} title="Pick footage in the Media bin; it is added to the end of this beat">
                  <Plus size={14} strokeWidth={2} aria-hidden /> Add clip from Media
                </button>
                {clip && selClip != null && (
                  <button type="button" className="ghost small" onClick={() => setInspPick("clip")}>
                    Back to clip {selClip + 1}
                  </button>
                )}
              </div>
              {selKind === "b" && beat.bed && (
                <>
                  <div className="nle-insp-subtitle">Bed audio tracks</div>
                  <AudioTracksSection
                    videoId={beat.bed.video_id}
                    inS={beat.bed.in_s}
                    outS={beat.bed.out_s}
                    tracks={beat.bed.tracks}
                    disabled={isAiWorking}
                    onChange={(tracks) => beat.bed && updateBeat(selBeat, { bed: { ...beat.bed, tracks } })}
                  />
                </>
              )}
              {issues.filter((i) => i.beat_id === beat.id).map(issueRow)}
            </section>
          )}

          {inspTab === "cut" && (
            <section className="nle-insp-section">
              <label className="nle-insp-field">
                Title
                <input value={script.title} onChange={(e) => setScript({ ...script, title: e.target.value })} />
              </label>
              <label className="nle-insp-field">
                Target length (m:ss)
                <DurationField value={script.target_duration_s} allowEmpty
                  onChange={(value) => setScript({ ...script, target_duration_s: value })} />
                {targetDuration > 0 && (
                  <span className={`insp-target-note ${isOverTarget ? "over" : ""}`}>
                    {clock(totalDuration)} of {clock(targetDuration)}
                    {isOverTarget
                      ? ` · ${clock(totalDuration - targetDuration)} over`
                      : ` · ${clock(targetDuration - totalDuration)} left`}
                  </span>
                )}
              </label>
              <div className="nle-insp-row">
                <span className="small muted">
                  Version {version ?? "–"} · {script.beats.length} {script.beats.length === 1 ? "beat" : "beats"} ·{" "}
                  {script.beats.reduce((n, b) => n + b.clips.length, 0)} clips · {+fps.toFixed(3)} fps
                </span>
              </div>
              {sessionTitle && (
                <div className="small muted insp-chat-line">
                  <Sparkles size={13} strokeWidth={2} aria-hidden /> Drafted by Sonar in “{sessionTitle}”
                </div>
              )}
              <div className="nle-insp-subtitle">Cut overview</div>
              <ScrollPanel className="insp-overview-scroll" label="Cut overview"><ol className="insp-cut-overview">
                {script.beats.map((b, i) => (
                  <li key={b.id}>
                    <button type="button" className="ghost small" onClick={() => { selectBeat(i); setInspPick("beat"); }} title={b.purpose}>
                      <span>{b.purpose || b.id}</span><strong>{clock(beatDuration(b))}</strong>
                    </button>
                  </li>
                ))}
              </ol></ScrollPanel>
              <div className="nle-insp-subtitle">Feedback ({feedbackCount})</div>
              {feedbackCount === 0 && <div className="small muted">{targetDuration > 0
                ? `The cut runs ${clock(totalDuration)} against a ${clock(targetDuration)} target.`
                : "No issues found."}</div>}
              {[...errors, ...warnings, ...infos].map(issueRow)}
            </section>
          )}
        </ScrollPanel>
      </aside>

      {/* Bottom: the timeline across the whole width */}
      <section className="nle-panel nle-timeline-dock">
        <div className="nle-timeline-head">
          <span className="nle-panel-title">Timeline</span>
          <span className="nle-timeline-name" title={script.title}>
            {script.title}
          </span>
          {version != null && <span className="pill small version-pill">v{version}</span>}
          {isDirty && <span className="tag warn">Unsaved</span>}
          <div className={`nle-duration ${isOverTarget ? "over" : ""}`} title="Cut length against its target">
            <span>
              {clock(totalDuration)}
              {targetDuration > 0 && <span className="muted"> / {clock(targetDuration)}</span>}
            </span>
            {targetDuration > 0 && (
              <span className="nle-duration-bar">
                <span style={{ width: `${Math.min(100, Math.max(2, pctOfTarget))}%` }} />
              </span>
            )}
          </div>
          {isAiWorking && (
            <span className="nle-ai-working-banner">
              <Sparkles size={14} strokeWidth={2} aria-hidden /> Sonar is revising this cut; editing is paused
            </span>
          )}
          {error && <span className="bad-text small">{error}</span>}
          <span className="nle-history">
            <button type="button" className="nle-icon-btn" onClick={undo} disabled={past.current.length === 0 || isAiWorking} title="Undo (Ctrl+Z)" aria-label="Undo">
              <Undo2 size={15} strokeWidth={2} aria-hidden />
            </button>
            <button type="button" className="nle-icon-btn" onClick={redo} disabled={future.current.length === 0 || isAiWorking} title="Redo (Ctrl+Shift+Z)" aria-label="Redo">
              <Redo2 size={15} strokeWidth={2} aria-hidden />
            </button>
            <button type="button" className="nle-icon-btn" onClick={() => setShowKeys(true)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
              <Keyboard size={15} strokeWidth={2} aria-hidden />
            </button>
          </span>
          <button
            type="button"
            className="primary small"
            onClick={() => handleSave()}
            disabled={saving || !isDirty || isAiWorking}
            title={isAiWorking ? "AI is currently generating an update" : "Save these edits as a new version"}
          >
            {saving ? "Saving…" : "Save version"}
          </button>
        </div>
        <div className={locked}>
          <ScriptTimeline
              script={script}
              videos={videosMap}
              selected={selectedBlock}
              frames={framesMap}
              onMoveClip={moveClipTo}
              onInsertPick={isAiWorking ? undefined : (pick, to) => {
                if (!videosMap[pick.video_id]) return;
                if (script.beats.length === 0) {
                  setScript({ ...script, beats: [{ id: "b1", purpose: "New beat", clips: [{ ...pick, audio: "source" }] }] });
                  setSelectedBlock("p-0-0");
                  requestFrames(pick.video_id);
                  return;
                }
                if (!script.beats[to.beat]) return;
                const clips = [...script.beats[to.beat].clips];
                clips.splice(to.clip, 0, { ...pick, audio: "source" });
                updateBeat(to.beat, { clips });
                setSelectedBlock(`p-${to.beat}-${to.clip}`);
                requestFrames(pick.video_id);
              }}
              onSplit={splitClip}
              onDelete={deleteAt}
              onToggleTrack={toggleTrackAt}
              onUnmute={(bIdx, cIdx) => updateClip(bIdx, cIdx, { audio: "source" })}
              monitor={trackMonitor}
              onMonitor={setTrackMonitor}
              onSelect={(key) => setSelectedBlock(key)}
              playhead={playhead}
              onSeek={(t) => {
                setPlayhead(t);
                playerRef.current?.seek(t);
              }}
              onTogglePlay={() => playerRef.current?.toggle()}
              onSetEdge={(bIdx, cIdx, edge, sourceSeconds) => {
                const beats = [...script.beats];
                const b = { ...beats[bIdx] };
                if (cIdx == null) {
                  if (!b.bed) return;
                  const bed = { ...b.bed };
                  if (edge === "in") bed.in_s = Math.min(Math.max(0, sourceSeconds), bed.out_s - 0.2);
                  else bed.out_s = Math.max(bed.in_s + 0.2, sourceSeconds);
                  b.bed = bed;
                } else {
                  const clips = [...b.clips];
                  const c = { ...clips[cIdx] };
                  if (edge === "in") c.in_s = Math.min(Math.max(0, sourceSeconds), c.out_s - 0.2);
                  else c.out_s = Math.max(c.in_s + 0.2, sourceSeconds);
                  clips[cIdx] = c;
                  b.clips = clips;
                }
                beats[bIdx] = b;
                setScript({ ...script, beats });
              }}
              onSnap={async (bIdx, cIdx) => {
                const sb = script.beats[bIdx];
                const videoId = cIdx == null ? sb.bed?.video_id : sb.clips[cIdx]?.video_id;
                if (videoId == null) return;
                let segs = transcriptCache.current[videoId];
                if (!segs) {
                  segs = await videoTranscript(videoId).catch(() => [] as TranscriptSegment[]);
                  transcriptCache.current[videoId] = segs;
                }
                if (segs.length === 0) return;
                // Pull each edge onto the nearest sentence boundary, but only if one is close: a clip
                // deliberately cut mid-thought should not jump half a sentence.
                const near = (t: number, candidates: number[]) => {
                  let best = t;
                  let dist = 1.5;
                  for (const c of candidates) {
                    if (Math.abs(c - t) < dist) {
                      dist = Math.abs(c - t);
                      best = c;
                    }
                  }
                  return best;
                };
                const starts = segs.map((x) => x.start);
                const ends = segs.map((x) => x.end);
                const beats = [...script.beats];
                const b = { ...beats[bIdx] };
                if (cIdx == null) {
                  if (!b.bed) return;
                  b.bed = { ...b.bed, in_s: near(b.bed.in_s, starts), out_s: near(b.bed.out_s, ends) };
                } else {
                  const clips = [...b.clips];
                  const c = { ...clips[cIdx] };
                  c.in_s = near(c.in_s, starts);
                  c.out_s = near(c.out_s, ends);
                  if (c.out_s - c.in_s < 0.2) return;
                  clips[cIdx] = c;
                  b.clips = clips;
                }
                beats[bIdx] = b;
                setScript({ ...script, beats });
              }}
              onTrim={(bIdx, cIdx, edge, delta) => {
                const beats = [...script.beats];
                const b = { ...beats[bIdx] };
                if (cIdx == null) {
                  // The sound bed: its own range moves, the pictures stay where they are.
                  if (!b.bed) return;
                  const bed = { ...b.bed };
                  if (edge === "in") bed.in_s = Math.max(0, bed.in_s + delta);
                  else bed.out_s = Math.max(bed.in_s + 0.2, bed.out_s + delta);
                  if (bed.out_s - bed.in_s < 0.2) return;
                  b.bed = bed;
                } else {
                  const clips = [...b.clips];
                  const c = { ...clips[cIdx] };
                  const max = videosMap[c.video_id]?.duration_s ?? Number.MAX_SAFE_INTEGER;
                  if (edge === "in") c.in_s = Math.min(Math.max(0, c.in_s + delta), c.out_s - 0.2);
                  else c.out_s = Math.max(c.in_s + 0.2, Math.min(c.out_s + delta, max));
                  clips[cIdx] = c;
                  b.clips = clips;
                }
                beats[bIdx] = b;
                setScript({ ...script, beats });
              }}
            />
        </div>
      </section>
      {showKeys && <ShortcutsSheet onClose={() => setShowKeys(false)} />}
    </div>
  );
});

/** A text box that wraps and grows with what is written, for one-line-ish notes that run long. */
function AutoGrowTextarea({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [value]);
  return (
    <textarea
      ref={ref}
      rows={1}
      className="nle-autogrow"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

/** One shortcut: alternatives (outer), each a key combination (inner). */
type KeyRow = { keys: string[][]; what: string };
type KeyGroup = { title: string; note?: string; rows: KeyRow[] };

const KEY_GROUPS: KeyGroup[] = [
  {
    title: "Anywhere in the editor",
    rows: [
      { keys: [["Ctrl", "Z"]], what: "Undo" },
      { keys: [["Ctrl", "Shift", "Z"], ["Ctrl", "Y"]], what: "Redo" },
      { keys: [["Ctrl", "S"]], what: "Save version" },
      { keys: [["Ctrl", "J"]], what: "Show or hide Sonar" },
      { keys: [["Ctrl", "B"]], what: "Collapse or expand the app sidebar" },
      { keys: [["Ctrl", "Shift", "F"]], what: "Search Material (all projects)" },
      { keys: [["?"]], what: "This sheet" },
    ],
  },
  {
    title: "Timeline",
    note: "Click the timeline first; a highlight around it shows the timeline will receive the key presses.",
    rows: [
      { keys: [["←"], ["→"]], what: "Trim the selected clip's out point (nothing selected: move the playhead)" },
      { keys: [["Shift", "←"], ["Shift", "→"]], what: "Trim the in point" },
      { keys: [["Alt"], ["Ctrl"]], what: "With an arrow: 0.2 s / 5 s steps" },
      { keys: [["I"], ["O"]], what: "Set in / out at the playhead" },
      { keys: [["S"]], what: "Snap both edges to sentence ends" },
      { keys: [["C"], ["Ctrl", "K"]], what: "Blade: cut the clip at the playhead" },
      { keys: [["M"]], what: "Audio block: track on or off (a muted clip: unmute)" },
      { keys: [["Delete"]], what: "Remove the selected clip or bed" },
      { keys: [["Z"]], what: "Zoom to the selected beat, again to fit" },
      { keys: [["="], ["-"], ["\\"]], what: "Zoom in / out / fit" },
      { keys: [["Space"]], what: "Play / pause" },
      { keys: [["Esc"]], what: "Clear the selection" },
    ],
  },
  {
    title: "Viewer",
    note: "Click the picture first.",
    rows: [
      { keys: [["Space"], ["K"]], what: "Play / pause" },
      { keys: [["J"], ["L"]], what: "Shuttle back / forward (press again to go faster)" },
      { keys: [["←"], ["→"]], what: "One frame" },
      { keys: [["Shift", "←"], ["Shift", "→"]], what: "One second" },
      { keys: [["↑"], ["↓"]], what: "Previous / next cut point" },
      { keys: [["Home"], ["End"]], what: "Start / end" },
      { keys: [["F"]], what: "Fullscreen" },
    ],
  },
];

/** Mouse actions: described, not drawn as keys. */
const MOUSE_ROWS: [string, string][] = [
  ["Click the ruler or a lane", "Move the playhead there"],
  ["Drag along the ruler", "Scrub"],
  ["Click a block", "Select it (the inspector follows)"],
  ["Drag a picture clip", "Move it; a line shows where it lands"],
  ["Drag a clip's edge", "Trim it"],
  ["Ctrl + wheel over the timeline", "Zoom around the pointer"],
  ["Drag the In or Out label", "Scrub that point by frames"],
  ["Double-click the picture", "Fullscreen"],
];

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MAC_KEYS: Record<string, string> = { Ctrl: "⌘", Alt: "⌥", Shift: "⇧" };
const keyLabel = (k: string) => (IS_MAC ? MAC_KEYS[k] ?? k : k);

function KeyCombo({ combo }: { combo: string[] }) {
  return (
    <span className="keycombo">
      {combo.map((k, i) => (
        <span key={i} className="keycombo-part">
          {i > 0 && !IS_MAC && <span className="keycombo-plus">+</span>}
          <kbd className="keycap">{keyLabel(k)}</kbd>
        </span>
      ))}
    </span>
  );
}

/** The keyboard and mouse reference, opened with ? or the keyboard button. */
function ShortcutsSheet({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="nle-keys-backdrop" onClick={onClose}>
      <div className="nle-keys-sheet" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Keyboard shortcuts">
        <div className="nle-panel-head">
          <span className="nle-panel-title">Keyboard shortcuts</span>
          <button type="button" className="ghost small" onClick={onClose} aria-label="Close" title="Close (Esc)">
            <X size={14} strokeWidth={2} aria-hidden />
          </button>
        </div>
        <div className="nle-keys-cols">
          {KEY_GROUPS.map((g) => (
            <section key={g.title}>
              <h4>{g.title}</h4>
              {g.note && <p className="keys-note">{g.note}</p>}
              {g.rows.map((row) => (
                <div key={row.what} className="nle-key-row">
                  <span className="nle-key-keys">
                    {row.keys.map((combo, i) => (
                      <span key={i} className="keycombo-alt">
                        {i > 0 && <span className="keycombo-or">/</span>}
                        <KeyCombo combo={combo} />
                      </span>
                    ))}
                  </span>
                  <span>{row.what}</span>
                </div>
              ))}
            </section>
          ))}
          <section>
            <h4>Mouse</h4>
            {MOUSE_ROWS.map(([action, what]) => (
              <div key={action} className="nle-key-row nle-mouse-row">
                <span className="nle-mouse-action">{action}</span>
                <span>{what}</span>
              </div>
            ))}
          </section>
        </div>
      </div>
    </div>
  );
}

export default ScriptEditor;
