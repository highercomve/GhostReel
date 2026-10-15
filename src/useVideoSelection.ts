import { useEffect, useRef, useState } from "react";
import type { MouseEvent, PointerEvent } from "react";

type Box = { left: number; top: number; width: number; height: number };
type Drag = { pointerId: number; x: number; y: number; clientX: number; clientY: number; startClientX: number; startClientY: number;
  scroll: HTMLElement | null; base: Set<number>; additive: boolean; active: boolean };

/** Selection is separate from the clip opened in the player. Dragging is opt-in and mouse-only. */
export function useVideoSelection(visibleIds: number[]) {
  const [ids, setIds] = useState<Set<number>>(() => new Set());
  const [enabled, setEnabled] = useState(false);
  const [box, setBox] = useState<Box | null>(null);
  const container = useRef<HTMLElement | null>(null);
  const anchor = useRef<number | null>(null);
  const drag = useRef<Drag | null>(null);
  const suppressClick = useRef(false);
  const animation = useRef<number | null>(null);

  const stop = (restore = false) => {
    const current = drag.current;
    if (restore) suppressClick.current = false;
    if (restore && current) setIds(new Set(current.base));
    drag.current = null;
    if (current && container.current?.hasPointerCapture(current.pointerId)) {
      container.current.releasePointerCapture(current.pointerId);
    }
    if (animation.current != null) cancelAnimationFrame(animation.current);
    animation.current = null;
    setBox(null);
  };

  useEffect(() => {
    const cancel = (event: KeyboardEvent) => {
      if (event.key === "Escape" && drag.current) stop(true);
    };
    window.addEventListener("keydown", cancel);
    return () => {
      window.removeEventListener("keydown", cancel);
      if (animation.current != null) cancelAnimationFrame(animation.current);
    };
  }, []);

  const clear = () => { stop(); setIds(new Set()); anchor.current = null; };
  const toggle = (id: number, range = false) => {
    const from = anchor.current == null ? -1 : visibleIds.indexOf(anchor.current);
    const to = visibleIds.indexOf(id);
    setIds((previous) => {
      const next = new Set(previous);
      if (range && from >= 0 && to >= 0) {
        for (const value of visibleIds.slice(Math.min(from, to), Math.max(from, to) + 1)) next.add(value);
      } else if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    if (!range || from < 0) anchor.current = id;
  };
  const all = visibleIds.length > 0 && visibleIds.every((id) => ids.has(id));
  const toggleAll = () => setIds((previous) => {
    const next = new Set(previous);
    for (const id of visibleIds) { if (all) next.delete(id); else next.add(id); }
    return next;
  });

  const update = () => {
    const current = drag.current;
    const root = container.current;
    if (!current?.active || !root) return;
    const rect = root.getBoundingClientRect();
    const x = Math.max(0, Math.min(root.scrollWidth, current.clientX - rect.left + root.scrollLeft));
    const y = Math.max(0, Math.min(root.scrollHeight, current.clientY - rect.top + root.scrollTop));
    const area = { left: Math.min(current.x, x), top: Math.min(current.y, y),
      width: Math.abs(x - current.x), height: Math.abs(y - current.y) };
    const next = current.additive ? new Set(current.base) : new Set<number>();
    root.querySelectorAll<HTMLElement>("tbody tr[data-video-id]").forEach((row) => {
      const r = row.getBoundingClientRect();
      const left = r.left - rect.left + root.scrollLeft;
      const top = r.top - rect.top + root.scrollTop;
      if (left <= area.left + area.width && left + r.width >= area.left &&
          top <= area.top + area.height && top + r.height >= area.top) next.add(Number(row.dataset.videoId));
    });
    setIds((previous) => previous.size === next.size && [...previous].every((id) => next.has(id)) ? previous : next);
    setBox((previous) => previous && previous.left === area.left && previous.top === area.top &&
      previous.width === area.width && previous.height === area.height ? previous : area);
  };
  const tick = () => {
    const current = drag.current;
    if (!current?.active) return;
    // Keep selecting through long libraries without requiring a second drag.
    const bounds = current.scroll?.getBoundingClientRect() ?? { top: 0, bottom: window.innerHeight };
    const target = current.scroll ?? window;
    if (current.clientY > bounds.bottom - 44) target.scrollBy(0, 14);
    else if (current.clientY < bounds.top + 44) target.scrollBy(0, -14);
    update();
    animation.current = requestAnimationFrame(tick);
  };

  const onPointerDown = (event: PointerEvent<HTMLElement>) => {
    suppressClick.current = false;
    if (!enabled || event.button !== 0 || event.pointerType !== "mouse" ||
      (event.target as Element).closest("input,select,button,a")) return;
    const root = event.currentTarget;
    const rect = root.getBoundingClientRect();
    let scroll = root.parentElement;
    while (scroll && !/auto|scroll/.test(getComputedStyle(scroll).overflowY)) scroll = scroll.parentElement;
    drag.current = { scroll, pointerId: event.pointerId, x: event.clientX - rect.left + root.scrollLeft,
      y: event.clientY - rect.top + root.scrollTop, clientX: event.clientX, clientY: event.clientY, startClientX: event.clientX, startClientY: event.clientY,
      base: new Set(ids), additive: event.ctrlKey || event.metaKey || event.shiftKey, active: false };
  };
  const onPointerMove = (event: PointerEvent<HTMLElement>) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (!current.active && Math.hypot(event.clientX - current.startClientX, event.clientY - current.startClientY) >= 6) {
      current.active = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      suppressClick.current = true;
      animation.current = requestAnimationFrame(tick);
    }
    current.clientX = event.clientX;
    current.clientY = event.clientY;
    if (current.active) event.preventDefault();
  };
  const onClickCapture = (event: MouseEvent<HTMLElement>) => {
    if (!suppressClick.current) return;
    suppressClick.current = false;
    event.preventDefault(); event.stopPropagation();
  };

  return { ids, setIds, enabled, setEnabled, box, container, clear, toggle, all, toggleAll,
    pointer: { onPointerDown, onPointerMove, onPointerUp: () => { update(); stop(); },
      onPointerCancel: () => stop(true), onLostPointerCapture: () => stop(), onClickCapture,
      onPointerLeave: () => { if (!drag.current?.active) stop(); } } };
}
