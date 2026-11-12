import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";

let openCount = 0;

/**
 * Every dialog in the app: rendered at the top of the document (outside any scroll container or
 * stacking context of the page that opened it), closed by Escape or a click on the backdrop.
 * While one is open, `html.modal-open` lets the page behind stand still (see scrollbars.css).
 */
export default function Modal({
  onClose,
  className = "",
  labelledBy,
  children,
}: {
  onClose: () => void;
  /** Extra classes for the dialog box (sizes, layout), next to `nle-modal-content card`. */
  className?: string;
  /** Id of the element holding the dialog's title. */
  labelledBy?: string;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    openCount += 1;
    document.documentElement.classList.add("modal-open");
    return () => {
      window.removeEventListener("keydown", onKey);
      openCount -= 1;
      if (openCount === 0) document.documentElement.classList.remove("modal-open");
    };
  }, [onClose]);

  return createPortal(
    <div className="nle-modal-backdrop" onClick={onClose}>
      <div
        className={`nle-modal-content card ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}
