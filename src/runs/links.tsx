import { useEffect, useState } from "react";
import type { LucideIcon } from "lucide-react";
import { ArchiveRestore, Film, FolderKanban, FolderOpen, LayoutGrid, Video } from "lucide-react";
import { fileUrl, isDesktop, restoreRunLink, runLinksState, type LinkState, type RunLink } from "../api";
import { onTrashChanged, trashChanged } from "./trashEvents";

/**
 * What the page showing a run can open. Each handler is optional: a link whose handler the page
 * does not provide is not shown there.
 */
export interface LinkContext {
  openClipSet?: (id: string) => void;
  openScript?: (id: number) => void;
  openVideo?: (id: number) => void;
  openProject?: (id: number) => void;
}

type LinkOf<K extends RunLink["kind"]> = Extract<RunLink, { kind: K }>;

interface LinkKind<K extends RunLink["kind"]> {
  icon: LucideIcon;
  label: string;
  /** Shown instead of the button once the target is gone for good. */
  goneLabel: string;
  /** How to open it from this page, or null when this page can't. */
  opener: (link: LinkOf<K>, ctx: LinkContext) => (() => void) | null;
}

/** Show a file in its folder (desktop) or open it in a tab (browser). */
async function revealFile(path: string) {
  if (isDesktop) {
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(path);
  } else {
    window.open(fileUrl(path), "_blank", "noopener");
  }
}

/** Every kind of link a run can carry. Adding a kind is one entry here (and one in `runs::RunLink`). */
export const LINK_KINDS: { [K in RunLink["kind"]]: LinkKind<K> } = {
  clip_set: {
    icon: LayoutGrid,
    label: "Open clip set",
    goneLabel: "Clip set deleted",
    opener: (l, ctx) => (ctx.openClipSet ? () => ctx.openClipSet!(l.id) : null),
  },
  script: {
    icon: Film,
    label: "Open cut",
    goneLabel: "Cut deleted",
    opener: (l, ctx) => (ctx.openScript ? () => ctx.openScript!(l.id) : null),
  },
  file: {
    icon: FolderOpen,
    label: isDesktop ? "Show in folder" : "Open file",
    goneLabel: "File no longer there",
    opener: (l) => () => void revealFile(l.path).catch(() => {}),
  },
  video: {
    icon: Video,
    label: "Open video",
    goneLabel: "Video removed",
    opener: (l, ctx) => (ctx.openVideo ? () => ctx.openVideo!(l.id) : null),
  },
  project: {
    icon: FolderKanban,
    label: "Open project",
    goneLabel: "Project deleted",
    opener: (l, ctx) => (ctx.openProject ? () => ctx.openProject!(l.id) : null),
  },
};

/** A link's identity: what it points at, not where it sits in the list. */
function linkKey(link: RunLink): string {
  return link.kind === "file" ? `file:${link.path}` : `${link.kind}:${link.id}`;
}

function opener(link: RunLink, ctx: LinkContext): (() => void) | null {
  const kind = LINK_KINDS[link.kind] as LinkKind<typeof link.kind> | undefined;
  return kind ? (kind.opener as (l: RunLink, c: LinkContext) => (() => void) | null)(link, ctx) : null;
}

/** A run's links as buttons; the ones this page can't open are left out. Core says where each
 *  target stands (again on click): one in the trash offers Restore on any page, one gone for good
 *  shows as a disabled button with why. */
export function RunLinks({ links, ctx }: { links: RunLink[]; ctx: LinkContext }) {
  // Everything per link is keyed by `linkKey`, so it stays with its link when the list changes.
  const [states, setStates] = useState<Record<string, LinkState>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  // Restored from here: still listed (with what happened) on a page that can't open it.
  const [restored, setRestored] = useState<string[]>([]);
  const key = JSON.stringify(links);
  useEffect(() => {
    let live = true;
    const load = () => {
      if (links.length === 0) return;
      runLinksState(links)
        .then((s) => live && setStates(Object.fromEntries(links.map((l, i) => [linkKey(l), s[i]]))))
        .catch(() => {});
    };
    setStates({});
    load();
    const off = onTrashChanged(load);
    return () => {
      live = false;
      off();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const buttons = links
    .map((link) => ({ link, id: linkKey(link), open: opener(link, ctx), state: states[linkKey(link)] }))
    // A trashed target can be restored from anywhere; opening needs this page's handler.
    .filter((b) => b.open != null || b.state?.state === "trashed" || restored.includes(b.id));
  if (buttons.length === 0) return null;

  const setState = (id: string, s: LinkState) => setStates((all) => ({ ...all, [id]: s }));

  // Deleted (or restored) since the card was drawn: check before acting.
  const follow = async (link: RunLink, id: string, open: (() => void) | null) => {
    const [now] = await runLinksState([link]).catch(() => [{ state: "live" } as LinkState]);
    setState(id, now);
    if (now.state === "live" && open) open();
  };

  const restore = async (link: RunLink, id: string) => {
    setBusy(id);
    setFailed(null);
    try {
      await restoreRunLink(link);
      setState(id, { state: "live" });
      setRestored((r) => (r.includes(id) ? r : [...r, id]));
      trashChanged();
    } catch (e) {
      setFailed(String(e));
    } finally {
      setBusy((b) => (b === id ? null : b));
    }
  };

  return (
    <div className="run-log-links">
      {buttons.map(({ link, id, open, state }) => {
        const { icon: Icon, label, goneLabel } = LINK_KINDS[link.kind];
        if (state?.state === "trashed") {
          return (
            <button
              key={id}
              type="button"
              className="small icon-text"
              disabled={busy === id}
              onClick={() => void restore(link, id)}
              title="Put it back exactly as it was"
            >
              <ArchiveRestore size={13} aria-hidden="true" /> {busy === id ? "Restoring…" : "Restore"}
            </button>
          );
        }
        const isGone = state?.state === "gone";
        if (!open) {
          // Restored here, but this page has no way to open it.
          return (
            <span key={id} className="muted small">
              Restored
            </span>
          );
        }
        return (
          <button
            key={id}
            type="button"
            className="small icon-text"
            disabled={isGone}
            onClick={() => void follow(link, id, open)}
            title={isGone ? goneLabel : link.kind === "file" ? link.path : undefined}
          >
            <Icon size={13} aria-hidden="true" /> {isGone ? goneLabel : label}
          </button>
        );
      })}
      {failed && <span className="banner bad small">{failed}</span>}
    </div>
  );
}
