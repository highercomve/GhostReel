/**
 * Cross-page links without prop drilling: a page dispatches "Set up on the Models page"
 * or "open Settings" as this event, and the app shell (App.tsx) listens and switches page.
 */
export type NavigateKind = "models" | "settings" | "storage" | "activity" | "status";

export const NAVIGATE_EVENT = "ghostreel:navigate";

/** Ask the app shell to switch to another page. */
export function navigate(kind: NavigateKind) {
  window.dispatchEvent(new CustomEvent(NAVIGATE_EVENT, { detail: { kind } }));
}
