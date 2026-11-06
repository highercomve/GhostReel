import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

/** A bounded scroll region with a solid, line-safe bottom edge and an explicit continuation. */
export default function ScrollPanel({ children, className = "", label }: {
  children: ReactNode; className?: string; label: string;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);
  const [guard, setGuard] = useState(0);
  useLayoutEffect(() => {
    const el = viewport.current;
    const inner = content.current;
    if (!el || !inner) return;
    const measure = () => {
      const continues = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
      setMore(continues);
      // Cover only a line that crosses the viewport edge, using a solid panel-colored strip.
      // Measuring text ranges handles wrapped captions, variable fonts and resized panels.
      const bottom = el.getBoundingClientRect().bottom;
      let cover = 0;
      const lines: DOMRect[] = [];
      const walker = document.createTreeWalker(inner, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.textContent?.trim()) continue;
        const range = document.createRange();
        range.selectNodeContents(node);
        lines.push(...range.getClientRects());
      }
      // Whole controls (a track row's checkbox, its play button) can straddle the edge even when
      // their text does not, so their boxes are measured as well: the edge must fall on a row
      // boundary, never through half a control.
      for (const node of inner.querySelectorAll(".at-row, button, label, li, tr, .nle-insp-row, .nle-insp-file, .insp-timecodes, .probe-status, .nle-version-row")) {
        lines.push(node.getBoundingClientRect());
      }
      // Inline elements can have different line boxes. Move the solid edge upward until
      // none of those boxes straddles it, rather than covering one caption but slicing another.
      let changed = true;
      while (changed) {
        changed = false;
        const edge = bottom - cover;
        for (const rect of lines) {
          if (rect.top < edge - 0.1 && rect.bottom > edge + 0.1) {
            const next = Math.ceil(bottom - rect.top);
            if (next > cover) { cover = next; changed = true; }
          }
        }
      }
      setGuard(continues ? Math.ceil(cover) : 0);
    };
    const resize = new ResizeObserver(measure);
    resize.observe(el);
    resize.observe(inner);
    const mutation = new MutationObserver(measure);
    mutation.observe(inner, { subtree: true, childList: true, characterData: true, attributes: true });
    el.addEventListener("scroll", measure, { passive: true });
    measure();
    return () => { resize.disconnect(); mutation.disconnect(); el.removeEventListener("scroll", measure); };
  }, []);
  return <div className={`editor-scroll-panel ${className}`}>
    <div className="editor-scroll-stage">
      <div className="editor-scroll-viewport" ref={viewport} tabIndex={0} aria-label={label}>
        <div className="editor-scroll-content" ref={content}>{children}</div>
      </div>
      <div className="editor-scroll-guard" style={{ height: guard }} aria-hidden="true" />
    </div>
    <button type="button" className="editor-scroll-more" disabled={!more}
      aria-label={`More below in ${label}`} onClick={() => {
        const el = viewport.current;
        if (el) el.scrollBy({ top: Math.max(48, el.clientHeight * 0.65), behavior: "smooth" });
      }}>
      {more ? <>More below <ChevronDown size={13} aria-hidden="true" /></> : "End of panel"}
    </button>
  </div>;
}
