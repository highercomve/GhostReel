import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { Overlay } from "../api";
import { anchorTransform, overlayCss, overlayKey } from "./textOverlay";

/** How close (px) an edge or the middle must come to a guide to snap onto it. */
const SNAP_PX = 7;
/** The title-safe margin the edges snap to, as a fraction of the frame. */
export const SAFE = 0.05;

type Drag = {
  key: string;
  mode: "move" | "resize";
  pointerId: number;
  px0: number;
  py0: number;
  x0: number;
  y0: number;
  size0: number;
  /** The text block at the start, normalised to the frame. */
  box: { left: number; top: number; w: number; h: number };
  frame: { w: number; h: number };
  moved: boolean;
  /** Captions are stored by the bottom of their text: added to y when one centred on the
   * frame (the default for stacked cuts) is first moved. */
  yAdj: number;
};

type Live = { key: string; x: number; y: number; size: number };

interface Props {
  /** The overlays on screen now (already filtered to the playhead and the on/off switches). */
  items: Overlay[];
  frameH: number;
  frameRef: React.RefObject<HTMLDivElement | null>;
  editable: boolean;
  selected: string | null;
  onSelect: (key: string | null) => void;
  onMove: (key: string, x: number, y: number) => void;
  onResize: (key: string, size: number) => void;
}

/**
 * Titles and captions drawn over the picture where the render burns them, from the backend's
 * own layout. Selectable and draggable: move by dragging, resize with Shift-drag or the corner
 * handle; edges snap to the middle lines and the safe margin, with guides while dragging.
 */
export default function TextOverlays({ items, frameH, frameRef, editable, selected, onSelect, onMove, onResize }: Props) {
  const drag = useRef<Drag | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [guides, setGuides] = useState<{ v: number[]; h: number[] }>({ v: [], h: [] });
  const liveRef = useRef<Live | null>(null);
  const cb = useRef({ onMove, onResize });
  cb.current = { onMove, onResize };

  // A drag survives the pointer leaving the text: listen on the window until it is released.
  useEffect(() => {
    const move = (e: PointerEvent) => {
      const d = drag.current;
      if (!d || e.pointerId !== d.pointerId) return;
      const dxPx = e.clientX - d.px0;
      const dyPx = e.clientY - d.py0;
      if (!d.moved && Math.hypot(dxPx, dyPx) < 3) return;
      d.moved = true;
      if (d.mode === "resize") {
        const span = Math.max(8, (d.box.w * d.frame.w + d.box.h * d.frame.h) / 2);
        const scale = Math.max(0.2, 1 + (dxPx + dyPx) / span);
        const size = Math.min(0.3, Math.max(0.012, d.size0 * scale));
        liveRef.current = { key: d.key, x: d.x0, y: d.y0, size };
        setLive(liveRef.current);
        return;
      }
      let dx = dxPx / d.frame.w;
      let dy = dyPx / d.frame.h;
      const snapX = SNAP_PX / d.frame.w;
      const snapY = SNAP_PX / d.frame.h;
      const v: number[] = [];
      const h: number[] = [];
      const left = d.box.left + dx;
      const top = d.box.top + dy;
      // Middle first, then the safe edges: the nearest guide within reach wins.
      const xCands: [number, number][] = [
        [0.5, left + d.box.w / 2],
        [SAFE, left],
        [1 - SAFE, left + d.box.w],
      ];
      const yCands: [number, number][] = [
        [0.5, top + d.box.h / 2],
        [SAFE, top],
        [1 - SAFE, top + d.box.h],
      ];
      const bestX = xCands.map(([g, at]) => [g, g - at] as const).filter(([, diff]) => Math.abs(diff) < snapX)
        .sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]))[0];
      const bestY = yCands.map(([g, at]) => [g, g - at] as const).filter(([, diff]) => Math.abs(diff) < snapY)
        .sort((a, b) => Math.abs(a[1]) - Math.abs(b[1]))[0];
      if (bestX && !e.altKey) {
        dx += bestX[1];
        v.push(bestX[0]);
      }
      if (bestY && !e.altKey) {
        dy += bestY[1];
        h.push(bestY[0]);
      }
      const x = Math.min(1, Math.max(0, d.x0 + dx));
      const y = Math.min(1, Math.max(0, d.y0 + dy));
      liveRef.current = { key: d.key, x, y, size: d.size0 };
      setLive(liveRef.current);
      setGuides({ v, h });
    };
    const up = (e: PointerEvent) => {
      const d = drag.current;
      if (!d || e.pointerId !== d.pointerId) return;
      drag.current = null;
      const l = liveRef.current;
      liveRef.current = null;
      setGuides({ v: [], h: [] });
      if (d.moved && l) {
        if (d.mode === "resize") cb.current.onResize(d.key, +l.size.toFixed(4));
        else cb.current.onMove(d.key, +l.x.toFixed(4), +Math.min(1, l.y + d.yAdj).toFixed(4));
      }
      // Keep showing where it was dropped until the new layout arrives.
      setTimeout(() => setLive((cur) => (cur === l ? null : cur)), 400);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, []);

  // A fresh layout (a moved or resized item) replaces the dragged preview.
  const sig = items.map((o, i) => `${overlayKey(o, i)}:${o.x}:${o.y}:${o.style.size}`).join("|");
  useEffect(() => {
    if (!drag.current) setLive(null);
  }, [sig]);

  const start = (e: ReactPointerEvent, key: string, o: Overlay, mode: "move" | "resize") => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    onSelect(key);
    if (!editable) return;
    const frame = frameRef.current?.getBoundingClientRect();
    const el = (e.currentTarget as HTMLElement).closest<HTMLElement>(".lp-ovl");
    const r = el?.getBoundingClientRect();
    if (!frame || !r || frame.width <= 0) return;
    drag.current = {
      key,
      mode: mode === "move" && e.shiftKey ? "resize" : mode,
      pointerId: e.pointerId,
      px0: e.clientX,
      py0: e.clientY,
      x0: o.x,
      y0: o.y,
      size0: o.style.size,
      box: {
        left: (r.left - frame.left) / frame.width,
        top: (r.top - frame.top) / frame.height,
        w: r.width / frame.width,
        h: r.height / frame.height,
      },
      frame: { w: frame.width, h: frame.height },
      moved: false,
      yAdj: o.kind === "caption" && o.v_anchor === "middle" ? r.height / frame.height / 2 : 0,
    };
  };

  if (frameH <= 0) return null;
  return (
    <>
      {items.map((o, i) => {
        const key = overlayKey(o, i);
        const l = live?.key === key ? live : null;
        const css = overlayCss(o.style, frameH, l?.size);
        const isSel = selected === key;
        return (
          <div
            key={key}
            className={`lp-ovl lp-ovl-${o.kind} ${editable ? "editable" : ""} ${isSel ? "selected" : ""}`}
            style={{
              ...css.block,
              left: `${(l?.x ?? o.x) * 100}%`,
              top: `${(l?.y ?? o.y) * 100}%`,
              transform: anchorTransform(o.style.align, o.v_anchor),
            }}
            onPointerDown={(e) => start(e, key, o, "move")}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
            title={editable ? `${o.kind === "title" ? "Title" : "Caption"}: drag to move, Shift-drag or the corner to resize, arrows nudge` : undefined}
          >
            {o.lines.map((line, j) => (
              <div key={j} className="lp-ovl-line">
                <span style={css.line}>{line}</span>
              </div>
            ))}
            {isSel && editable && (
              <span className="lp-ovl-handle" onPointerDown={(e) => start(e, key, o, "resize")} aria-hidden="true" />
            )}
          </div>
        );
      })}
      {guides.v.map((g) => (
        <span key={`v${g}`} className="lp-guide v" style={{ left: `${g * 100}%` }} aria-hidden="true" />
      ))}
      {guides.h.map((g) => (
        <span key={`h${g}`} className="lp-guide h" style={{ top: `${g * 100}%` }} aria-hidden="true" />
      ))}
    </>
  );
}
