import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * Drag handles for the editor grid: left panel ↔ viewer ↔ inspector widths, and the height of the
 * timeline. Rendered inside `.nle-grid`; it reads the panels' boxes and writes the grid's size
 * variables (`--nle-left-w`, `--nle-insp-w`, `--nle-tl-h`), remembered across sessions.
 * Double-click a handle to go back to the default size.
 */
type Key = "left" | "insp" | "tl";
const VARS: Record<Key, string> = { left: "--nle-left-w", insp: "--nle-insp-w", tl: "--nle-tl-h" };
const STORE = "ghostreel.editorPanels";
const LIMITS: Record<Key, [number, number]> = { left: [160, 520], insp: [220, 560], tl: [120, 2000] };

type Boxes = { left?: DOMRect; center?: DOMRect; insp?: DOMRect; tl?: DOMRect; grid: DOMRect };

function load(): Partial<Record<Key, number>> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

export default function GridSplitters() {
  const anchor = useRef<HTMLSpanElement>(null);
  const [boxes, setBoxes] = useState<Boxes | null>(null);
  const sizes = useRef<Partial<Record<Key, number>>>(load());

  const grid = () => anchor.current?.parentElement ?? null;

  const apply = useCallback(() => {
    const g = grid();
    if (!g) return;
    for (const k of Object.keys(VARS) as Key[]) {
      const v = sizes.current[k];
      if (v == null) g.style.removeProperty(VARS[k]);
      else g.style.setProperty(VARS[k], `${Math.round(v)}px`);
    }
  }, []);

  const measure = useCallback(() => {
    const g = grid();
    if (!g) return;
    const pick = (sel: string) => (g.querySelector(`:scope > ${sel}`) as HTMLElement | null)?.getBoundingClientRect();
    setBoxes({
      grid: g.getBoundingClientRect(),
      left: pick(".nle-left"),
      center: pick(".nle-center"),
      insp: pick(".nle-inspector"),
      tl: pick(".nle-timeline-dock"),
    });
  }, []);

  useLayoutEffect(() => {
    apply();
    measure();
    const g = grid();
    if (!g) return;
    const ro = new ResizeObserver(measure);
    ro.observe(g);
    for (const el of Array.from(g.children)) ro.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [apply, measure]);

  const save = () => localStorage.setItem(STORE, JSON.stringify(sizes.current));

  const start = (key: Key) => (e: React.PointerEvent) => {
    e.preventDefault();
    const g = grid();
    if (!g || !boxes) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const startSize =
      key === "left" ? boxes.left?.width ?? 270 : key === "insp" ? boxes.insp?.width ?? 330 : boxes.tl?.height ?? 240;
    // Leave the viewer and the top row a usable minimum, whatever the handle is dragged to.
    const viewerMin = 320;
    const maxFor = (k: Key) => {
      const [, hard] = LIMITS[k];
      if (k === "tl") return Math.min(hard, boxes.grid.height - 220);
      const other = k === "left" ? boxes.insp?.width ?? 0 : boxes.left?.width ?? 0;
      return Math.min(hard, boxes.grid.width - other - viewerMin - 16);
    };
    document.body.classList.add(key === "tl" ? "nle-resizing-row" : "nle-resizing-col");
    const move = (ev: PointerEvent) => {
      const delta = key === "left" ? ev.clientX - startX : key === "insp" ? startX - ev.clientX : startY - ev.clientY;
      const next = Math.max(LIMITS[key][0], Math.min(maxFor(key), startSize + delta));
      sizes.current = { ...sizes.current, [key]: next };
      apply();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("nle-resizing-row", "nle-resizing-col");
      save();
      measure();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const reset = (key: Key) => () => {
    const { [key]: _, ...rest } = sizes.current;
    sizes.current = rest;
    apply();
    save();
    measure();
  };

  // Keyboard: focus a handle, then arrows resize in 16 px steps.
  const keys = (key: Key) => (e: React.KeyboardEvent) => {
    if (!boxes) return;
    const step = e.shiftKey ? 64 : 16;
    const grow = key === "tl" ? e.key === "ArrowUp" : key === "left" ? e.key === "ArrowRight" : e.key === "ArrowLeft";
    const shrink = key === "tl" ? e.key === "ArrowDown" : key === "left" ? e.key === "ArrowLeft" : e.key === "ArrowRight";
    if (!grow && !shrink) return;
    e.preventDefault();
    const cur =
      sizes.current[key] ?? (key === "left" ? boxes.left?.width : key === "insp" ? boxes.insp?.width : boxes.tl?.height) ?? 0;
    sizes.current = { ...sizes.current, [key]: Math.max(LIMITS[key][0], cur + (grow ? step : -step)) };
    apply();
    save();
  };

  useEffect(() => () => document.body.classList.remove("nle-resizing-row", "nle-resizing-col"), []);

  const g = boxes?.grid;
  const handle = (key: Key, style: React.CSSProperties, label: string) => (
    <div
      key={key}
      role="separator"
      aria-orientation={key === "tl" ? "horizontal" : "vertical"}
      aria-label={label}
      title={`${label} — drag to resize, double-click to reset`}
      tabIndex={0}
      className={`nle-splitter ${key === "tl" ? "row" : "col"}`}
      style={style}
      onPointerDown={start(key)}
      onDoubleClick={reset(key)}
      onKeyDown={keys(key)}
    />
  );

  return (
    <>
      <span ref={anchor} className="nle-splitter-anchor" aria-hidden />
      {g && boxes?.left && boxes.center &&
        handle(
          "left",
          { left: (boxes.left.right + boxes.center.left) / 2 - g.left, top: boxes.left.top - g.top, height: boxes.left.height },
          "Resize the left panel",
        )}
      {g && boxes?.insp && boxes.center &&
        handle(
          "insp",
          { left: (boxes.center.right + boxes.insp.left) / 2 - g.left, top: boxes.insp.top - g.top, height: boxes.insp.height },
          "Resize the inspector",
        )}
      {g && boxes?.tl && boxes.center &&
        handle(
          "tl",
          { top: (boxes.center.bottom + boxes.tl.top) / 2 - g.top, left: 0, width: g.width },
          "Resize the timeline",
        )}
    </>
  );
}
