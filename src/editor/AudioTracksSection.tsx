import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Square } from "lucide-react";
import {
  dbLabel,
  effectiveTracks,
  levelFraction,
  SILENT_DB,
  toggleTrack,
  trackDescription,
  trackName,
  trackProxyUrl,
  useAudioLevels,
  useAudioTracks,
} from "./audioTracks";
import "./audioTracks.css";

interface AudioTracksSectionProps {
  videoId: number;
  /** The source range that plays, for the level readout and the preview. */
  inS: number;
  outS: number;
  /** The clip's or bed's `tracks`: undefined = the file's automatic track. */
  tracks: number[] | undefined;
  /** The clip is muted as a whole: the choice still applies once it's unmuted. */
  muted?: boolean;
  disabled?: boolean;
  onChange: (tracks: number[] | undefined) => void;
}

/**
 * The inspector's "Audio tracks": which of the file's tracks a clip (or bed) plays. Pro cameras
 * record a lav and a camera mic on separate tracks, and unplugged inputs as silent ones; the level
 * beside each makes the silent ones obvious before anyone listens.
 */
export default function AudioTracksSection({
  videoId,
  inS,
  outS,
  tracks,
  muted,
  disabled,
  onChange,
}: AudioTracksSectionProps) {
  const ids = useMemo(() => [videoId], [videoId]);
  const infos = useAudioTracks(ids)[videoId];
  const requests = useMemo(() => [{ videoId, inS, outS }], [videoId, inS, outS]);
  const levelOf = useAudioLevels(requests);
  const levels = levelOf(videoId, inS, outS);
  const on = new Set(effectiveTracks(tracks, infos));

  const [previewing, setPreviewing] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // The playing state and range live on refs so the element's persistent listeners can read them
  // without going stale; `token` invalidates async work (proxy URLs) from superseded plays.
  const previewingRef = useRef<number | null>(null);
  const tokenRef = useRef(0);
  const rangeRef = useRef({ inS, outS });
  rangeRef.current = { inS, outS };

  const stopPreview = () => {
    tokenRef.current++;
    previewingRef.current = null;
    const a = audioRef.current;
    if (a) {
      a.pause();
      try {
        a.currentTime = 0;
      } catch {
        // Setting a time before metadata is ready throws; there is nothing to rewind then.
      }
    }
    setPreviewing(null);
  };
  const stopRef = useRef(stopPreview);
  stopRef.current = stopPreview;
  // A new selection (or unmount) must not leave a preview running.
  useEffect(() => () => stopRef.current(), [videoId, inS, outS]);

  /** The one preview element, created on first use; its listeners are attached once and guarded. */
  const ensureAudio = (): HTMLAudioElement => {
    let a = audioRef.current;
    if (a) return a;
    a = new Audio();
    audioRef.current = a;
    a.addEventListener("loadedmetadata", () => {
      if (previewingRef.current === null) return; // stopped before loading finished: stay stopped
      a.currentTime = rangeRef.current.inS;
      void a.play().catch(() => stopRef.current());
    });
    a.addEventListener("timeupdate", () => {
      if (previewingRef.current !== null && a.currentTime >= rangeRef.current.outS) stopRef.current();
    });
    a.addEventListener("ended", () => stopRef.current());
    a.addEventListener("error", () => {
      if (previewingRef.current === null) return;
      setPreviewError("Couldn't play this track.");
      stopRef.current();
    });
    return a;
  };

  const preview = async (track: number) => {
    if (previewingRef.current === track) return stopPreview();
    stopPreview();
    setPreviewError(null);
    const token = ++tokenRef.current;
    previewingRef.current = track;
    setPreviewing(track);
    try {
      const url = await trackProxyUrl(videoId, track);
      if (token !== tokenRef.current) return; // stopped or moved on while the proxy was extracting
      const a = ensureAudio();
      if (a.getAttribute("src") !== url) {
        a.src = url;
        a.load();
      } else if (a.readyState >= HTMLMediaElement.HAVE_METADATA) {
        a.currentTime = rangeRef.current.inS;
        void a.play().catch(() => stopRef.current());
      } // else the same URL is still buffering: `loadedmetadata` starts it at the in point.
    } catch (err) {
      if (token !== tokenRef.current) return;
      previewingRef.current = null;
      setPreviewing(null);
      setPreviewError(String(err));
    }
  };

  if (infos === undefined) return <div className="small muted">Reading audio tracks…</div>;
  if (infos === null) return <div className="small muted">Audio track details aren't available for this file.</div>;
  if (infos.length === 0) return <div className="small muted">This file has no audio.</div>;

  return (
    <div className="at-section" aria-label={muted ? "Source tracks (clip sound muted)" : "Source tracks"}>

      {infos.map((t) => {
        const db = levels?.[t.index];
        return (
          <div key={t.index} className={`at-row ${on.has(t.index) ? "on" : ""}`}>
            <label className="at-check">
              <input
                type="checkbox"
                checked={on.has(t.index)}
                disabled={disabled}
                onChange={() => onChange(toggleTrack(tracks, infos, t.index))}
              />
              <span className="at-name">{trackName(t.index)}</span>
            </label>
            <span className="at-desc" title={trackDescription(t)}>
              {trackDescription(t)}
              {t.auto && <span className="at-auto"> · auto</span>}
            </span>
            <span className={`at-level ${db != null && db <= SILENT_DB ? "silent" : ""}`}>
              {db == null ? (levels === undefined ? "…" : "") : dbLabel(db)}
              {db != null && <span className="at-bar" style={{ width: `${levelFraction(db) * 100}%` }} />}
            </span>
            <button
              type="button"
              className="at-play"
              onClick={() => void preview(t.index)}
              title={previewing === t.index ? "Stop" : `Listen to ${trackName(t.index)} over this clip's range`}
              aria-label={previewing === t.index ? "Stop" : `Listen to ${trackName(t.index)}`}
            >
              {previewing === t.index ? <Square size={14} strokeWidth={2} aria-hidden /> : <Play size={14} strokeWidth={2} aria-hidden />}
            </button>
          </div>
        );
      })}
      <div className="at-foot small muted">
        {tracks === undefined ? (
          <span>Playing the automatic track.</span>
        ) : (
          <>
            <span>{tracks.length === 0 ? "No tracks: silent." : `Playing ${tracks.map(trackName).join(" + ")}.`}</span>
            <button type="button" className="ghost small" disabled={disabled} onClick={() => onChange(undefined)}>
              Automatic
            </button>
          </>
        )}
      </div>
      {previewError && <div className="small bad-text">{previewError}</div>}
    </div>
  );
}
