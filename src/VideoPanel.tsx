import MediaImage from "./MediaImage";
import type { SteadinessView } from "./api";
import { formatDuration } from "./timeFormat";
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight, Music, Pause, Play } from "lucide-react";
import {
  clock,
  fileName,
  mediaUrl,
  videoPlayback,
  videoPlaybackProgress,
  type PlaybackProgress,
  openExternal,
  videoFrames,
  videoSteadiness,
  videoTranscript,
  type FrameRow,
  type TranscriptSegment,
  type VideoRow,
} from "./api";

interface Description {
  description?: string;
  objects?: string[];
  setting?: string;
  shot?: string;
  tags?: string[];
  error?: string;
}

const parseDescription = (f: FrameRow): Description => {
  try {
    return f.description ? JSON.parse(f.description) : {};
  } catch {
    return {};
  }
};

/** Player + keyframes + transcript for one video; `seek` jumps the player (e.g. from a search hit). */
export default function VideoPanel({
  video,
  seek,
  onClose,
  onExclude,
}: {
  video: VideoRow;
  seek: { t: number; nonce: number } | null;
  onClose: () => void;
  /** Take the video out of the project's library (removed, or kept as a reference edit). */
  onExclude?: (role: "removed" | "reference") => void;
}) {
  const player = useRef<HTMLVideoElement>(null);
  const frameStrip = useRef<HTMLDivElement>(null);
  const [segments, setSegments] = useState<TranscriptSegment[] | null>(null);
  const [frames, setFrames] = useState<FrameRow[]>([]);
  const [steadiness, setSteadiness] = useState<SteadinessView | null>(null);
  const [activeFrame, setActiveFrame] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const [now, setNow] = useState(0);
  const [playError, setPlayError] = useState(false);
  const [src, setSrc] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const pendingSeek = useRef<number | null>(null);

  useEffect(() => {
    setSegments(null);
    videoTranscript(video.id).then(setSegments).catch(() => setSegments([]));
  }, [video.id, video.segments]);
  useEffect(() => {
    videoFrames(video.id).then(setFrames).catch(() => setFrames([]));
  }, [video.id, video.frames]);
  useEffect(() => {
    setSteadiness(null);
    videoSteadiness(video.id).then(setSteadiness).catch(() => setSteadiness(null));
  }, [video.id, video.frames]);
  const shaky = (steadiness?.windows ?? []).filter(
    (w) => steadiness!.max_shake > 0 && (w.jerk > steadiness!.limit || (steadiness!.max_sway > 0 && w.sway > steadiness!.max_sway)),
  );
  const [preparing, setPreparing] = useState(false);
  const [proxyProgress, setProxyProgress] = useState<PlaybackProgress | null>(null);
  const [proxyError, setProxyError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setPlayError(false);
    setSrc(null);
    setPreparing(true);
    setProxyProgress(null);
    setProxyError(null);
    let finished = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const progress = await videoPlaybackProgress(video.id);
        if (!cancelled && !finished) setProxyProgress(progress);
      } catch { /* Playback remains usable if a progress poll fails. */ }
      if (!cancelled && !finished) timer = setTimeout(poll, 500);
    };
    void poll();
    // A 4K 10-bit camera original sits black in the player for a long time; ask for something
    // it can play (a proxy, built once), and fall back to the original if that fails.
    videoPlayback(video.id)
      .then((pb) => mediaUrl(pb.path))
      .catch((error) => {
        if (!cancelled) setProxyError(String(error));
        return mediaUrl(video.path);
      })
      .then((url) => {
        if (!cancelled) setSrc(url);
      })
      .catch(() => {
        if (!cancelled) setPlayError(true);
      })
      .finally(() => {
        finished = true;
        clearTimeout(timer);
        if (!cancelled) setPreparing(false);
      });
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [video.id, video.path]);

  const jump = (t: number) => {
    const v = player.current;
    // Before metadata loads, remember the seek and apply it in onLoadedMetadata.
    if (!v || v.readyState < 1) {
      pendingSeek.current = t;
      return;
    }
    v.currentTime = t;
    v.play().catch(() => {});
  };
  useEffect(() => {
    if (seek) jump(seek.t);
  }, [seek?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const current = useMemo(
    () => (segments ?? []).findIndex((s) => now >= s.start && now < s.end),
    [segments, now],
  );
  const q = filter.trim().toLowerCase();
  const shown = (segments ?? []).map((s, i) => ({ s, i })).filter(({ s }) => !q || s.text.toLowerCase().includes(q));
  const active = frames.find((f) => f.id === activeFrame);
  const d = active ? parseDescription(active) : null;
  const duration = player.current?.duration || video.duration_s || 0;
  const timeLabel = formatDuration;
  const togglePlayback = () => {
    const element = player.current;
    if (!element) return;
    if (element.paused) element.play().catch(() => {});
    else element.pause();
  };

  return (
    <section className="card video-panel">
      <div className="card-head">
        <span className="label">
          {fileName(video.path)}
          {video.category && <span className={`tag cat-badge cat-${video.category}`}>{video.category}</span>}
          {video.category_tag && (
            <span className="tag cat-badge cat-tag" title={`Detected tag: ${video.category_tag}`}>
              {video.category_tag === "music" ? (
                <span className="icon-text"><Music size={12} strokeWidth={2} aria-hidden /> Music</span>
              ) : (
                video.category_tag
              )}
            </span>
          )}
          {video.language && <span className="tag">{video.language.toUpperCase()}</span>}
        </span>
        <span className="inline">
          <button className="ghost small" onClick={() => openExternal(video.path, player.current?.currentTime ?? 0)}>
            Open in player
          </button>
          {onExclude && (
            <>
              <button
                className="ghost small"
                title="A finished edit made by a person: the script chat studies it but never uses it as footage"
                onClick={() => onExclude("reference")}
              >
                Use as reference edit
              </button>
              <button
                className="ghost small danger video-remove-library"
                title="Stop using this video in search and scripts (the file is not deleted)"
                onClick={() => onExclude("removed")}
              >
                Remove from library
              </button>
            </>
          )}
          <button className="ghost small" onClick={onClose}>
            Close
          </button>
        </span>
      </div>

      <div className="video-panel-content">
      <div className="video-panel-visual">
      <div className="player-wrap">
        <video
          ref={player}
          src={src ?? undefined}
          preload="metadata"
          onLoadedMetadata={(e) => {
            if (pendingSeek.current != null) {
              e.currentTarget.currentTime = pendingSeek.current;
              pendingSeek.current = null;
              e.currentTarget.play().catch(() => {});
            }
          }}
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onError={() => { if (src && !preparing) setPlayError(true); }}
        />
        {preparing && !src && (
          <div className="proxy-loading" role="status" aria-live="polite">
            <div className="proxy-loading-label">
              <span>{proxyProgress?.phase ?? "Checking playable copy…"}</span>
              <span>{proxyProgress?.fraction != null ? `${Math.floor(proxyProgress.fraction * 100)}%` : ""}</span>
            </div>
            <progress aria-label="Playback copy creation" max={1}
              value={proxyProgress?.fraction ?? undefined} />
            {proxyProgress?.encoder && <div className="muted small">
              {proxyProgress.encoder === "h264_nvenc" ? "GPU encoding" : "CPU encoding"}
              {proxyProgress.encoded_s > 0 ? ` · ${clock(proxyProgress.encoded_s)} encoded` : ""}
            </div>}
          </div>
        )}
        {playError && (
          <div className="play-error">
            {proxyError ? "Could not prepare the playback copy." : "This video format can't play here."}{" "}
            <button className="ghost small" onClick={() => openExternal(video.path, now)}>
              Open in system player
            </button>
            {proxyError && <details className="small"><summary>Error details</summary>{proxyError}</details>}
          </div>
        )}
      </div>
      <div className="video-mini-transport">
        <button type="button" className="ghost small icon-text" onClick={togglePlayback} disabled={!src} aria-label={isPlaying ? "Pause video" : "Play video"} title={isPlaying ? "Pause" : "Play"}>
          {isPlaying ? <Pause size={15} strokeWidth={2} /> : <Play size={15} strokeWidth={2} />}{isPlaying ? "Pause" : "Play"}
        </button>
        <span className="video-time-readout" aria-live="off">{timeLabel(now)} / {timeLabel(duration)}</span>
        <input type="range" min={0} max={duration || 1} step={0.1} value={Math.min(now, duration || 1)} aria-label="Seek video"
          onChange={(event) => { const time = Number(event.target.value); if (player.current) player.current.currentTime = time; setNow(time); }} />
      </div>

      {video.duration_s != null && steadiness && steadiness.windows.length > 0 && (
        <div className="shake-bar-group">
        <div className="video-section-label">Camera steadiness · red marks shakier moments</div>
        <div
          className="shake-bar"
          title={
            (steadiness.camera !== "unknown" ? `${steadiness.camera} camera. ` : "") +
            (shaky.length ? "Red: shakier than this clip's own level — cut around it. Click to jump." : "Steady throughout")
          }
        >
          {shaky.map((w) => (
            <span
              key={w.start_s}
              className="shaky"
              style={{
                left: `${(w.start_s / video.duration_s!) * 100}%`,
                width: `${((w.end_s - w.start_s) / video.duration_s!) * 100}%`,
              }}
              title={`shaky ${clock(w.start_s)}–${clock(w.end_s)} (tremor ${w.jerk.toFixed(2)}, sway ${w.sway.toFixed(2)})`}
              onClick={() => jump(w.start_s)}
            />
          ))}
          <span className="playhead" style={{ left: `${(now / video.duration_s!) * 100}%` }} />
        </div>
        </div>
      )}
      {frames.length > 0 && (
        <>
        <div className="video-section-label">Keyframes · click to jump</div>
        <div className={`frame-strip-wrap${frames.length > 4 ? " has-next" : ""}`}>
        <div className="frame-strip" ref={frameStrip}>
          {frames.map((f) => (
            <figure
              key={f.id}
              className={f.id === activeFrame ? "active" : ""}
              onClick={() => {
                setActiveFrame(f.id === activeFrame ? null : f.id);
                jump(f.t_s);
              }}
            >
              <MediaImage path={f.path} alt="" loading="lazy" />
              <figcaption>{clock(f.t_s)}</figcaption>
            </figure>
          ))}
        </div>
        {frames.length > 4 && <button type="button" className="frame-strip-next ghost" aria-label="Scroll to later keyframes" title="More keyframes" onClick={() => {
          const strip = frameStrip.current;
          const firstTile = strip?.firstElementChild as HTMLElement | null;
          if (!strip || !firstTile) return;
          const gap = Number.parseFloat(getComputedStyle(strip).columnGap) || 0;
          strip.scrollBy({ left: firstTile.getBoundingClientRect().width + gap, behavior: "smooth" });
        }}><ChevronRight size={16} strokeWidth={2} aria-hidden="true" /></button>}
        </div>
        </>
      )}
      {active && d && (
        <div className="frame-detail">
          <div className="muted small">
            {clock(active.t_s)}
            {d.shot ? ` · ${d.shot}` : ""}
            {d.setting ? ` · ${d.setting}` : ""}
          </div>
          {d.error ? (
            <div className="bad-text small">{d.error}</div>
          ) : d.description ? (
            <>
              <div>{d.description}</div>
              {active.visible_text && (
                <div className="small">
                  <span className="muted">On screen: </span>
                  {active.visible_text.split("\n").join(" · ")}
                </div>
              )}
              {(d.tags?.length ?? 0) > 0 && (
                <div>
                  {d.tags!.map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="muted small">Not described yet.</div>
          )}
        </div>
      )}

      </div>
      <aside className="video-panel-transcript" aria-label="Transcript">
      {segments && segments.length > 0 && (
        <input placeholder="Find in transcript…" value={filter} onChange={(e) => setFilter(e.target.value)} />
      )}
      {segments === null ? (
        <div className="muted">Loading…</div>
      ) : segments.length === 0 ? (
        <div className="muted">
          {video.transcribe === "skipped" ? "This video has no audio." : "No transcript yet."}
        </div>
      ) : (
        <div className="segments">
          {shown.map(({ s, i }) => (
            <div key={i} className={`segment ${i === current ? "current" : ""}`} onClick={() => jump(s.start)}>
              <span className="time">{clock(s.start)}</span>
              <span>{s.text}</span>
            </div>
          ))}
          {shown.length === 0 && <div className="muted">No match.</div>}
        </div>
      )}
      </aside>
      </div>
    </section>
  );
}
