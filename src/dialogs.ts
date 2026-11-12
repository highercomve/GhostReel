// Questions and notices for the user, one way for the whole app.
//
// Inside the desktop app `window.confirm` / `alert` / `prompt` do not show anything: WebKitGTK
// under Tauri answers `confirm` with "yes" straight away, so a "Delete…?" guarded by it deleted on
// the first click. On the desktop these go through the native dialog plugin; the browser build
// (web UI) keeps the browser's own dialogs, which do work there.
import { ask, message } from "@tauri-apps/plugin-dialog";
import { isDesktop } from "./api";

export interface ConfirmOptions {
  title?: string;
  okLabel?: string;
  cancelLabel?: string;
  /** A warning icon, for anything that removes or overwrites. */
  destructive?: boolean;
}

/** Ask a yes/no question; resolves true only when the user says yes. */
export async function confirmAction(text: string, opts: ConfirmOptions = {}): Promise<boolean> {
  if (!isDesktop) return window.confirm(text);
  try {
    return await ask(text, {
      title: opts.title ?? "GhostReel",
      kind: opts.destructive ? "warning" : "info",
      okLabel: opts.okLabel ?? "OK",
      cancelLabel: opts.cancelLabel ?? "Cancel",
    });
  } catch {
    // No dialog could be shown: the safe answer is no.
    return false;
  }
}

/** Tell the user something that needs no answer. */
export async function notify(text: string, title = "GhostReel"): Promise<void> {
  if (!isDesktop) {
    window.alert(text);
    return;
  }
  try {
    await message(text, { title, kind: "info" });
  } catch {
    console.warn(text);
  }
}

/** Ask for a line of text; null when cancelled or empty. Only the browser build can (the native
 *  dialog plugin has no text field), so on the desktop it resolves null: offer a picker there. */
export function promptText(text: string): string | null {
  if (isDesktop) return null;
  return window.prompt(text)?.trim() || null;
}
