import { useCallback, useEffect, useRef, useState } from "react";
import {
  clock,
  fileName,
  fileUrl,
  getScript,
  projectView,
  saveScript,
  search,
  videoFrames,
  videoTranscript,
  type Beat,
  type FrameRow,
  type Hit,
  type Issue,
  type Script,
  type ScriptClip,
  type TranscriptSegment,
  type VideoRow,
} from "./api";
import PreviewPlayer from "./PreviewPlayer";
import ScriptTimeline from "./ScriptTimeline";

interface ScriptEditorProps {
  projectId: number;
  scriptId: number;
  sessionId: number | null;
  sessionTitle?: string;
  latestIssues?: Issue[];
  onScriptSaved: (newScriptId: number, issues: Issue[]) => void;
}

export default function ScriptEditor({
  projectId,
  scriptId,
  sessionId,
  sessionTitle,
  latestIssues,
  onScriptSaved,
}: ScriptEditorProps) {
  const [script, setScript] = useState<Script | null>(null);
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

  // Inline replace search state
  const [replacingKey, setReplacingKey] = useState<string | null>(null); // e.g. "bIdx-cIdx"
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);

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
        setScript(view.stored.script);
        setInitialJson(JSON.stringify(view.stored.script));
        setLoadedSessionId(view.stored.session_id);
        setVersion(view.stored.version);
        setIssues(latestIssues ?? view.issues);
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
  }, [scriptId, latestIssues]);

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
      const res = await saveScript(projectId, script, sessionId ?? loadedSessionId);
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

  const addClip = (bIdx: number) => {
    if (!script) return;
    const beat = script.beats[bIdx];
    // Find first available video or default to 1
    const firstVid = Object.keys(videosMap)[0] ? Number(Object.keys(videosMap)[0]) : 1;
    const newClips: ScriptClip[] = [
      ...beat.clips,
      {
        video_id: firstVid,
        in_s: 0,
        out_s: 4,
        audio: "source",
      },
    ];
    updateBeat(bIdx, { clips: newClips });
    // Open replace search for this new clip
    setReplacingKey(`${bIdx}-${newClips.length - 1}`);
    setSearchQuery("");
    setSearchResults(null);
  };

  // Inline search replacement
  const onSearchFootage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await search(projectId, searchQuery.trim(), 8);
      setSearchResults(res.hits);
    } catch (err) {
      setError(String(err));
    } finally {
      setSearching(false);
    }
  };

  const onSelectReplacement = (bIdx: number, cIdx: number, hit: Hit) => {
    updateClip(bIdx, cIdx, {
      video_id: hit.video_id,
      in_s: hit.start_s,
      out_s: hit.end_s,
    });
    setReplacingKey(null);
    setSearchResults(null);
    setSearchQuery("");
    requestFrames(hit.video_id);
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
  const infos = issues.filter((i) => i.severity === "info");

  return (
    <div className="script-editor-wrap">
      {/* Preview Player & Export Card */}
      <PreviewPlayer
        scriptId={scriptId}
        scriptTitle={script.title}
        isDirty={isDirty}
        onSaveBeforeAction={handleSave}
        onTime={setPlayhead}
        onControls={handleControls}
      />

      <div className="card script-editor">
        {/* Editor Head */}
        <div className="card-head">
          <div className="inline align-center">
            <span className="label">
              Script {version != null ? `v${version}` : ""}
            </span>
            {sessionTitle && (
              <span className="pill small" style={{ marginLeft: "8px", fontWeight: "normal", opacity: 0.8 }} title="Chat session this script belongs to">
                Chat: {sessionTitle}
              </span>
            )}
            {isDirty && <span className="tag warn">Unsaved changes</span>}
          </div>
          <div className="inline">
            <button
              type="button"
              onClick={() => handleSave()}
              disabled={saving || !isDirty}
            >
              {saving ? "Saving..." : "Save version"}
            </button>
          </div>
        </div>

        {error && <div className="banner bad small">{error}</div>}

        {/* Script Title & Target Duration */}
        <div className="script-meta-fields">
          <div className="field-group">
            <label className="small muted">Script Title</label>
            <input
              value={script.title}
              onChange={(e) => setScript({ ...script, title: e.target.value })}
              placeholder="Script title"
            />
          </div>
          <div className="field-group">
            <label className="small muted">Target Duration (s)</label>
            <input
              type="number"
              min="0"
              step="1"
              value={script.target_duration_s ?? ""}
              onChange={(e) =>
                setScript({
                  ...script,
                  target_duration_s: e.target.value ? Number(e.target.value) : undefined,
                })
              }
              placeholder="Target seconds"
            />
          </div>
        </div>

        {/* Duration Meter */}
        <div className="duration-meter-block">
          <div className="progress-head">
            <span className="small">
              Duration: <strong>{totalDuration.toFixed(1)} s</strong>
              {targetDuration > 0 && ` / ${targetDuration} s target`}
            </span>
            {targetDuration > 0 && (
              <span className={`small ${isOverTarget ? "warn-text" : "good-text"}`}>
                {Math.round(pctOfTarget)}%
                {isOverTarget && " (exceeds target)"}
              </span>
            )}
          </div>
          {targetDuration > 0 && (
            <div className={`meter ${isOverTarget ? "over-target" : ""}`}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(2, pctOfTarget))}%`,
                  background: isOverTarget ? "var(--warn)" : undefined,
                }}
              />
            </div>
          )}
        </div>

        {/* The cut on three lanes: titles, picture, sound */}
        <ScriptTimeline
          script={script}
          videos={videosMap}
          selected={selectedBlock}
          onSelect={(key, bIdx) => {
            setSelectedBlock(key);
            const id = script.beats[bIdx]?.id;
            if (id) document.getElementById(`beat-${id}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
          }}
          playhead={playhead}
          onSeek={(t) => {
            setPlayhead(t);
            playerRef.current?.seek(t);
          }}
          onTogglePlay={() => playerRef.current?.toggle()}
          onSetEdge={(bIdx, cIdx, edge, sourceSeconds) => {
            const beats = [...script.beats];
            const beat = { ...beats[bIdx] };
            if (cIdx == null) {
              if (!beat.bed) return;
              const bed = { ...beat.bed };
              if (edge === "in") bed.in_s = Math.min(Math.max(0, sourceSeconds), bed.out_s - 0.2);
              else bed.out_s = Math.max(bed.in_s + 0.2, sourceSeconds);
              beat.bed = bed;
            } else {
              const clips = [...beat.clips];
              const c = { ...clips[cIdx] };
              if (edge === "in") c.in_s = Math.min(Math.max(0, sourceSeconds), c.out_s - 0.2);
              else c.out_s = Math.max(c.in_s + 0.2, sourceSeconds);
              clips[cIdx] = c;
              beat.clips = clips;
            }
            beats[bIdx] = beat;
            setScript({ ...script, beats });
          }}
          onSnap={async (bIdx, cIdx) => {
            const beat = script.beats[bIdx];
            const videoId = cIdx == null ? beat.bed?.video_id : beat.clips[cIdx]?.video_id;
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
            const beat = { ...beats[bIdx] };
            if (cIdx == null) {
              // The sound bed: its own range moves, the pictures stay where they are.
              if (!beat.bed) return;
              const bed = { ...beat.bed };
              if (edge === "in") bed.in_s = Math.max(0, bed.in_s + delta);
              else bed.out_s = Math.max(bed.in_s + 0.2, bed.out_s + delta);
              if (bed.out_s - bed.in_s < 0.2) return;
              beat.bed = bed;
            } else {
              const clips = [...beat.clips];
              const c = { ...clips[cIdx] };
              const max = videosMap[c.video_id]?.duration_s ?? Number.MAX_SAFE_INTEGER;
              if (edge === "in") c.in_s = Math.min(Math.max(0, c.in_s + delta), c.out_s - 0.2);
              else c.out_s = Math.max(c.in_s + 0.2, Math.min(c.out_s + delta, max));
              clips[cIdx] = c;
              beat.clips = clips;
            }
            beats[bIdx] = beat;
            setScript({ ...script, beats });
          }}
        />

        {/* Issues List Grouped by Severity */}
        {issues.length > 0 && (
          <div className="issues-box">
            <div className="small label">
              Script Feedback ({issues.length} {issues.length === 1 ? "issue" : "issues"})
            </div>
            {errors.map((iss, i) => (
              <div
                key={`err-${i}`}
                className="issue-row bad-text small clickable-issue"
                onClick={() => {
                  if (iss.beat_id) {
                    document.getElementById(`beat-${iss.beat_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }
                }}
              >
                <span className="pill bad">error</span>
                {iss.beat_id && <strong>[Beat {iss.beat_id}]</strong>}
                {iss.clip_index != null && <span>Clip #{iss.clip_index + 1}:</span>}
                <span>{iss.message}</span>
              </div>
            ))}
            {warnings.map((iss, i) => (
              <div
                key={`warn-${i}`}
                className="issue-row warn-text small clickable-issue"
                onClick={() => {
                  if (iss.beat_id) {
                    document.getElementById(`beat-${iss.beat_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }
                }}
              >
                <span className="pill warn">warning</span>
                {iss.beat_id && <strong>[Beat {iss.beat_id}]</strong>}
                {iss.clip_index != null && <span>Clip #{iss.clip_index + 1}:</span>}
                <span>{iss.message}</span>
              </div>
            ))}
            {infos.map((iss, i) => (
              <div
                key={`info-${i}`}
                className="issue-row muted small clickable-issue"
                onClick={() => {
                  if (iss.beat_id) {
                    document.getElementById(`beat-${iss.beat_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }
                }}
              >
                <span className="pill info">info</span>
                {iss.beat_id && <strong>[Beat {iss.beat_id}]</strong>}
                {iss.clip_index != null && <span>Clip #{iss.clip_index + 1}:</span>}
                <span>{iss.message}</span>
              </div>
            ))}
          </div>
        )}

        {/* Beats List */}
        <div className="beats-list">
          <div className="beats-head">
            <span className="label">Story Beats ({script.beats.length})</span>
            <button type="button" className="ghost small" onClick={addBeat}>
              + Add beat
            </button>
          </div>

          {script.beats.length === 0 && (
            <div className="muted small">No beats yet. Click “+ Add beat” to begin.</div>
          )}

          {script.beats.map((beat, bIdx) => {
            const beatIssues = issues.filter((i) => i.beat_id === beat.id);
            const hasError = beatIssues.some((i) => i.severity === "error");
            const hasWarn = beatIssues.some((i) => i.severity === "warning");

            return (
              <div
                key={beat.id || bIdx}
                id={`beat-${beat.id}`}
                className={`beat-card card ${hasError ? "beat-error" : hasWarn ? "beat-warn" : ""}`}
              >
                <div className="beat-header">
                  <div className="inline align-center">
                    <span className="pill beat-id-pill">{beat.id}</span>
                    <input
                      className="beat-purpose-input"
                      value={beat.purpose}
                      onChange={(e) => updateBeat(bIdx, { purpose: e.target.value })}
                      placeholder="Beat purpose (e.g. hook, demo, punchline)"
                    />
                  </div>
                  <div className="inline">
                    <button
                      type="button"
                      className="ghost small"
                      disabled={bIdx === 0}
                      onClick={() => moveBeat(bIdx, -1)}
                      title="Move beat up"
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="ghost small"
                      disabled={bIdx === script.beats.length - 1}
                      onClick={() => moveBeat(bIdx, 1)}
                      title="Move beat down"
                    >
                      ↓
                    </button>
                    <button
                      type="button"
                      className="ghost small danger"
                      onClick={() => removeBeat(bIdx)}
                      title="Delete beat"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Beat-specific issue banners */}
                {beatIssues.map((iss, i) => (
                  <div
                    key={i}
                    className={`small ${iss.severity === "error" ? "bad-text" : "warn"}`}
                  >
                    ⚠ {iss.clip_index != null ? `Clip #${iss.clip_index + 1}: ` : ""}
                    {iss.message}
                  </div>
                ))}

                {/* Narration & On-screen text */}
                <div className="beat-text-fields">
                  <div className="field-group">
                    <label className="small muted">Narration (Voice-over)</label>
                    <textarea
                      rows={2}
                      value={beat.narration ?? ""}
                      onChange={(e) => updateBeat(bIdx, { narration: e.target.value })}
                      placeholder="Spoken words during this beat..."
                    />
                  </div>
                  <div className="field-group">
                    <label className="small muted">On-Screen Text</label>
                    <input
                      value={beat.on_screen_text ?? ""}
                      onChange={(e) => updateBeat(bIdx, { on_screen_text: e.target.value })}
                      placeholder="Titles / lower-thirds..."
                    />
                  </div>
                </div>

                {/* Clips in Beat */}
                <div className="beat-clips-section">
                  <div className="clips-head">
                    <span className="small muted font-bold">
                      Clips ({beat.clips.length})
                    </span>
                    <button
                      type="button"
                      className="ghost small"
                      onClick={() => addClip(bIdx)}
                    >
                      + Add clip
                    </button>
                  </div>

                  {beat.clips.length === 0 && (
                    <div className="muted small">No clips in this beat yet.</div>
                  )}

                  <div className="clips-list">
                    {beat.clips.map((clip, cIdx) => {
                      const clipKey = `${bIdx}-${cIdx}`;
                      const isReplacing = replacingKey === clipKey;
                      const frame = getNearestFrame(clip.video_id, clip.in_s);
                      const video = videosMap[clip.video_id];
                      const clipDur = Math.max(0, clip.out_s - clip.in_s);

                      return (
                        <div key={cIdx} className="clip-card">
                          <div className="clip-row">
                            {/* Thumbnail */}
                            <div className="clip-thumb-wrap">
                              {frame ? (
                                <img
                                  src={fileUrl(frame.path)}
                                  alt=""
                                  className="clip-thumb"
                                  loading="lazy"
                                />
                              ) : (
                                <div className="clip-thumb noframe" />
                              )}
                              <span className="clip-thumb-time small">
                                {clock(clip.in_s)}
                              </span>
                            </div>

                            {/* Info & Controls */}
                            <div className="clip-info">
                              <div className="clip-title-bar">
                                <span className="label small" title={video?.path ?? ""}>
                                  {video ? fileName(video.path) : `Video #${clip.video_id}`}
                                </span>
                                <span className="tag">
                                  {clipDur.toFixed(1)} s
                                </span>
                              </div>

                              <div className="clip-trim-inputs">
                                <label className="small muted">
                                  In:
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={clip.in_s}
                                    onChange={(e) =>
                                      updateClip(bIdx, cIdx, {
                                        in_s: Math.max(0, parseFloat(e.target.value) || 0),
                                      })
                                    }
                                  />
                                </label>
                                <label className="small muted">
                                  Out:
                                  <input
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    value={clip.out_s}
                                    onChange={(e) =>
                                      updateClip(bIdx, cIdx, {
                                        out_s: Math.max(0, parseFloat(e.target.value) || 0),
                                      })
                                    }
                                  />
                                </label>
                                <button
                                  type="button"
                                  className={`ghost small ${clip.audio === "mute" ? "warn" : ""}`}
                                  onClick={() =>
                                    updateClip(bIdx, cIdx, {
                                      audio: clip.audio === "mute" ? "source" : "mute",
                                    })
                                  }
                                  title="Toggle audio source vs mute"
                                >
                                  {clip.audio === "mute" ? "🔇 Muted" : "🔊 Audio"}
                                </button>
                              </div>

                              <div className="clip-why-input">
                                <input
                                  className="small"
                                  value={clip.why ?? ""}
                                  onChange={(e) =>
                                    updateClip(bIdx, cIdx, { why: e.target.value })
                                  }
                                  placeholder="Why this clip was chosen..."
                                />
                              </div>

                              <div className="clip-actions inline">
                                <button
                                  type="button"
                                  className={`ghost small ${isReplacing ? "active" : ""}`}
                                  onClick={() =>
                                    setReplacingKey(isReplacing ? null : clipKey)
                                  }
                                >
                                  {isReplacing ? "Cancel search" : "Replace from search"}
                                </button>
                                <button
                                  type="button"
                                  className="ghost small"
                                  disabled={cIdx === 0}
                                  onClick={() => moveClip(bIdx, cIdx, -1)}
                                  title="Move clip up"
                                >
                                  ↑
                                </button>
                                <button
                                  type="button"
                                  className="ghost small"
                                  disabled={cIdx === beat.clips.length - 1}
                                  onClick={() => moveClip(bIdx, cIdx, 1)}
                                  title="Move clip down"
                                >
                                  ↓
                                </button>
                                <button
                                  type="button"
                                  className="ghost small danger"
                                  onClick={() => removeClip(bIdx, cIdx)}
                                  title="Remove clip"
                                >
                                  ✕
                                </button>
                              </div>
                            </div>
                          </div>

                          {/* Replace from search inline popup/box */}
                          {isReplacing && (
                            <div className="inline-search-card">
                              <form className="inline-search-form" onSubmit={onSearchFootage}>
                                <input
                                  autoFocus
                                  placeholder="Search footage to replace clip: e.g. unboxing, logo..."
                                  value={searchQuery}
                                  onChange={(e) => setSearchQuery(e.target.value)}
                                />
                                <button type="submit" disabled={searching || !searchQuery.trim()}>
                                  {searching ? "Searching..." : "Find"}
                                </button>
                              </form>

                              {searchResults && (
                                <div className="search-hits-list">
                                  {searchResults.length === 0 && (
                                    <div className="muted small">No matches found.</div>
                                  )}
                                  {searchResults.map((hit, hIdx) => (
                                    <div
                                      key={hIdx}
                                      className="search-hit-item"
                                      onClick={() => onSelectReplacement(bIdx, cIdx, hit)}
                                    >
                                      {hit.frame ? (
                                        <img src={fileUrl(hit.frame)} alt="" loading="lazy" />
                                      ) : (
                                        <div className="noframe" />
                                      )}
                                      <div className="hit-info">
                                        <div className="hit-head">
                                          <span className="label small">
                                            {fileName(hit.path)}
                                          </span>
                                          <span className="time small">
                                            {clock(hit.start_s)}–{clock(hit.end_s)}
                                          </span>
                                        </div>
                                        <div className="small muted hit-snippet">
                                          {hit.snippet}
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
