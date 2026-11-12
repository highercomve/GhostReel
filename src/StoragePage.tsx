import { useCallback, useEffect, useState } from "react";
import { confirmAction } from "./dialogs";
import {
  cleanStorage,
  compactDatabase,
  humanSize,
  modelsStatus,
  removeProject,
  storageReport,
  type ModelsStatusView,
  type ProjectStorage,
  type StorageKind,
  type StorageReport,
} from "./api";
import { isActive, useQueue } from "./useQueue";

type StorageCategory = "cache" | "data" | "export";

interface KindDef {
  kind: StorageKind;
  label: string;
  category: StorageCategory;
  categoryLabel: string;
  hint: string;
  warning: string;
  actionLabel: string;
  isDestructive: boolean;
}

const KINDS: KindDef[] = [
  {
    kind: "proxies",
    label: "Proxies",
    category: "cache",
    categoryLabel: "Rebuildable cache",
    hint: "Light copies for playback and preview renders. Rebuilt when needed.",
    warning: "They are rebuilt the next time a video plays or a preview renders, which takes a moment.",
    actionLabel: "Clear",
    isDestructive: false,
  },
  {
    kind: "previews",
    label: "Script previews",
    category: "cache",
    categoryLabel: "Rebuildable cache",
    hint: "Rendered preview videos of scripts.",
    warning: "Previews are rendered again the next time you preview a script.",
    actionLabel: "Clear",
    isDestructive: false,
  },
  {
    kind: "temp",
    label: "Temporary files",
    category: "cache",
    categoryLabel: "Rebuildable cache",
    hint: "Scratch files of the chat helpers.",
    warning: "Nothing is lost.",
    actionLabel: "Clear",
    isDestructive: false,
  },
  {
    kind: "keyframes",
    label: "Keyframes",
    category: "data",
    categoryLabel: "Generated data",
    hint: "The images frame descriptions were made from, also used as thumbnails.",
    warning:
      "Thumbnails disappear until you rebuild keyframes on the project page, which re-runs frame descriptions. Descriptions, transcripts and search are kept.",
    actionLabel: "Clear",
    isDestructive: true,
  },
  {
    kind: "chat_images",
    label: "Chat images",
    category: "data",
    categoryLabel: "Saved media",
    hint: "Images pasted into script chats.",
    warning: "The chats keep their text, but the pasted images are permanently removed.",
    actionLabel: "Delete",
    isDestructive: true,
  },
  {
    kind: "chat_logs",
    label: "Chat logs",
    category: "data",
    categoryLabel: "Saved logs",
    hint: "Saved prompts, answers, tool calls and turn details.",
    warning: "Saved model logs are permanently deleted. Chat messages and scripts are kept.",
    actionLabel: "Delete",
    isDestructive: true,
  },
  {
    kind: "exports",
    label: "Exports",
    category: "export",
    categoryLabel: "Rendered export",
    hint: "Timeline exports saved in GhostReel's own folder.",
    warning: "Only exports inside GhostReel's folder are deleted; exports you saved elsewhere are not touched.",
    actionLabel: "Delete",
    isDestructive: true,
  },
];

/** Clearing caches: what is rebuilt on demand, so nothing visible is lost. */
const CACHES: StorageKind[] = ["proxies", "previews", "temp"];

const confirmStorage = (message: string, title: string, okLabel: string) =>
  confirmAction(message, { title, okLabel, destructive: true });

const total = (u: Record<string, number>) => Object.values(u).reduce((a, b) => a + b, 0);

/** The app a model file outside GhostReel's folder was found in (same wording as the Models page). */
const externalModelOrigin = (path: string): string => {
  const lower = path.toLowerCase();
  if (lower.includes("ghostpen")) return "GhostPen";
  if (lower.includes("lmstudio") || lower.includes("lm-studio") || lower.includes("lm studio"))
    return "LM Studio";
  if (lower.includes("ollama")) return "Ollama";
  return "another app";
};

function Bar({ bytes, of }: { bytes: number; of: number }) {
  const pct = of > 0 ? (100 * bytes) / of : 0;
  return (
    <div className="meter storage-bar" title={`${bytes.toLocaleString()} bytes (${humanSize(bytes)})`}>
      <div
        style={{
          width: bytes > 0 ? `${pct}%` : "0%",
          minWidth: bytes > 0 ? "4px" : "0",
        }}
      />
    </div>
  );
}

export default function StoragePage({ onChanged }: { onChanged: () => void }) {
  const [report, setReport] = useState<StorageReport | null>(null);
  const [models, setModels] = useState<ModelsStatusView | null>(null);
  const [loading, setLoading] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const busy = useQueue().some(isActive);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rep, mods] = await Promise.all([
        storageReport(),
        modelsStatus().catch(() => null),
      ]);
      setReport(rep);
      setModels(mods);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const run = async (action: () => Promise<string>) => {
    setWorking(true);
    setError(null);
    setNotice(null);
    try {
      setNotice(await action());
    } catch (e) {
      setError(String(e));
    } finally {
      setWorking(false);
      await refresh();
    }
  };

  const freed = (s: { files_deleted: number; bytes_freed: number; videos?: number }) =>
    `Freed ${humanSize(s.bytes_freed)} (${s.files_deleted} ${s.files_deleted === 1 ? "item" : "items"})` +
    (s.videos ? `, forgot the index of ${s.videos} ${s.videos === 1 ? "video" : "videos"}.` : ".");

  const clean = async (kinds: StorageKind[], bytes: number, project?: ProjectStorage) => {
    const isSingle = kinds.length === 1;
    const kindDef = isSingle ? KINDS.find((k) => k.kind === kinds[0]) : null;
    const what = kindDef ? kindDef.label.toLowerCase() : "caches";
    const where = project ? ` of "${project.name}"` : "";
    const actionVerb = kindDef?.actionLabel ?? "Clear";
    const why = kindDef ? kindDef.warning : "Proxies, previews and temporary files are rebuilt automatically when needed.";
    const shared = project && project.shared > 0 && kinds.some((k) => k === "proxies" || k === "keyframes")
      ? `\n\nSome of this footage is also in other projects; their copies go too.`
      : "";
    const ok = await confirmStorage(
      `${actionVerb} ${humanSize(bytes)} of ${what}${where}?\n\n${why}${shared}`,
      `${actionVerb} ${what}`,
      actionVerb,
    );
    if (ok) await run(async () => freed(await cleanStorage(project ? "project" : "all", kinds, project?.project_id)));
  };

  const cleanUnused = async () => {
    if (!report) return;
    const ok = await confirmStorage(
      `Delete ${humanSize(total(report.unused))} that no project uses?\n\nThis covers footage no project watches any more (${report.unused_videos} ${report.unused_videos === 1 ? "video" : "videos"}, including their transcripts and descriptions), files of deleted scripts and chats, proxies in formats this version no longer reads, and temporary files. Your video files are not touched.`,
      "Clean up unused files",
      "Delete",
    );
    if (ok) await run(async () => freed(await cleanStorage("unused", [])));
  };

  const compact = async () => {
    await run(async () => `Database compacted, ${humanSize(await compactDatabase())} saved.`);
  };

  const deleteProject = async (p: ProjectStorage) => {
    const ok = await confirmStorage(
      `Delete project "${p.name}" and everything GhostReel made for it?\n\nIts keyframes, proxies, previews and chat images go, and so do the transcripts, descriptions and search index of footage no other project uses. Your video files are not touched.`,
      "Delete project",
      "Delete project",
    );
    if (!ok) return;
    await run(async () => {
      const s = await removeProject(p.project_id, true);
      onChanged();
      return `Deleted "${p.name}". ${freed(s)}`;
    });
  };

  const disabled = busy || working;
  const r = report;
  const cacheBytes = r ? CACHES.reduce((a, k) => a + r.usage[k], 0) : 0;

  // Models reused from other apps live outside GhostReel's folder, so this table doesn't count
  // them; naming the difference reconciles the AI-models row with the Models page total.
  const reusedModels = models ? models.models.filter((m) => m.installed_path && !m.in_own_dir) : [];
  const reusedBytes = reusedModels.reduce((n, m) => n + m.entry.size_bytes + (m.entry.mmproj_size_bytes ?? 0), 0);
  const reusedApps = [...new Set(reusedModels.map((m) => externalModelOrigin(m.installed_path ?? "")))].join(" and ");
  const modelsHint = reusedModels.length
    ? `Model files kept in this folder. ${reusedModels.length} more model${reusedModels.length === 1 ? " is" : "s are"} reused from ${reusedApps} (${humanSize(reusedBytes)}) and live outside GhostReel, so they aren't counted here — manage them on the Models page.`
    : "Downloaded models. Remove them on the Models page.";

  return (
    <main>
      <header>
        <div>
          <h1>Storage</h1>
          <p className="muted">
            What GhostReel keeps on this computer{r ? ` · ${humanSize(r.total)} in ` : ""}
            {r && <span className="path">{r.data_dir}</span>}
          </p>
        </div>
        <button onClick={() => void refresh()} disabled={loading}>
          {loading ? "Measuring…" : "Measure again"}
        </button>
      </header>

      {error && <div className="banner bad">{error}</div>}
      {notice && <div className="banner good">{notice}</div>}
      {busy && <div className="banner info">Background work is running. Cleaning is available once it finishes.</div>}

      {r && (
        <>
          <h2>Overview</h2>
          <section className="card">
            <table className="storage-table">
              <thead>
                <tr>
                  <th>What</th>
                  <th className="num">Size</th>
                  <th className="bar-col" aria-hidden />
                  <th />
                </tr>
              </thead>
              <tbody>
                {KINDS.map(({ kind, label, category, categoryLabel, hint, actionLabel, isDestructive }) => (
                  <tr key={kind}>
                    <td className="name">
                      <div className="storage-name-row">
                        <span>{label}</span>
                        <span className={`storage-category-badge ${category}`}>{categoryLabel}</span>
                      </div>
                      <div className="muted small">{hint}</div>
                    </td>
                    <td className="num">{humanSize(r.usage[kind])}</td>
                    <td className="bar-col"><Bar bytes={r.usage[kind]} of={r.total} /></td>
                    <td className="actions">
                      <button
                        className={`small ghost ${isDestructive ? "danger" : ""}`}
                        disabled={disabled || r.usage[kind] === 0}
                        onClick={() => void clean([kind], r.usage[kind])}
                        title={`${actionLabel} ${label.toLowerCase()}`}
                      >
                        {actionLabel}
                      </button>
                    </td>
                  </tr>
                ))}
                <tr>
                  <td className="name">
                    <div className="storage-name-row">
                      <span>AI models</span>
                      <span className="storage-category-badge model">Model files</span>
                    </div>
                    <div className="muted small">{modelsHint}</div>
                  </td>
                  <td className="num">{humanSize(r.models)}</td>
                  <td className="bar-col"><Bar bytes={r.models} of={r.total} /></td>
                  <td />
                </tr>
                <tr>
                  <td className="name">
                    <div className="storage-name-row">
                      <span>Library database</span>
                      <span className="storage-category-badge data">Index & transcripts</span>
                    </div>
                    <div className="muted small">Transcripts, descriptions and the search index.</div>
                  </td>
                  <td className="num">{humanSize(r.database)}</td>
                  <td className="bar-col"><Bar bytes={r.database} of={r.total} /></td>
                  <td className="actions">
                    <button className="small ghost" disabled={disabled} onClick={() => void compact()} title="Hand back the space deleted entries left behind">
                      Compact
                    </button>
                  </td>
                </tr>
                {r.backups > 0 && (
                  <tr>
                    <td className="name">
                      <div className="storage-name-row">
                        <span>Database backups</span>
                        <span className="storage-category-badge data">Safety copies</span>
                      </div>
                      <div className="muted small">A copy of the library a day; only the newest few are kept. Cleanup never removes them.</div>
                    </td>
                    <td className="num">{humanSize(r.backups)}</td>
                    <td className="bar-col"><Bar bytes={r.backups} of={r.total} /></td>
                    <td />
                  </tr>
                )}
                {r.other > 0 && (
                  <tr>
                    <td className="name">
                      <div className="storage-name-row">
                        <span>Other</span>
                        <span className="storage-category-badge other">Logs & config</span>
                      </div>
                      <div className="muted small">Settings, logs and anything else.</div>
                    </td>
                    <td className="num">{humanSize(r.other)}</td>
                    <td className="bar-col"><Bar bytes={r.other} of={r.total} /></td>
                    <td />
                  </tr>
                )}
              </tbody>
            </table>
            <div className="inline storage-actions">
              <button className="ghost" disabled={disabled || cacheBytes === 0} onClick={() => void clean(CACHES, cacheBytes)}>
                Clear all rebuildable caches ({humanSize(cacheBytes)})
              </button>
            </div>
          </section>

          <section className="card storage-unused">
            <div className="card-head">
              <span className="label">Unused · {humanSize(total(r.unused))}</span>
              <button className="small ghost danger" disabled={disabled || total(r.unused) === 0} onClick={() => void cleanUnused()}>
                Clean up unused
              </button>
            </div>
            <div className="muted small">
              Files no project uses: {r.unused_videos} indexed {r.unused_videos === 1 ? "video" : "videos"} no project watches any more,
              files of deleted scripts and chats, old-format proxies and temporary files.
              {total(r.unused) > 0 &&
                " " +
                  KINDS.filter((k) => r.unused[k.kind] > 0)
                    .map((k) => `${k.label} ${humanSize(r.unused[k.kind])}`)
                    .join(" · ")}
            </div>
          </section>

          <h2>Projects</h2>
          {r.projects.length === 0 && <p className="muted">No projects.</p>}
          {r.projects.map((p) => {
            const projectTotal = total(p.usage);
            const projectCaches = CACHES.reduce((a, k) => a + p.usage[k], 0);
            return (
              <section key={p.project_id} className="card storage-project">
                <div className="card-head">
                  <span className="label">
                    {p.name} <span className="muted small">· {p.videos} {p.videos === 1 ? "video" : "videos"} · {humanSize(projectTotal)}</span>
                  </span>
                  <span className="inline">
                    <button
                      className="small ghost"
                      disabled={disabled || projectCaches === 0}
                      onClick={() => void clean(CACHES, projectCaches, p)}
                      title="Clear proxies and script previews for this project (rebuilt on demand)"
                    >
                      Clear caches ({humanSize(projectCaches)})
                    </button>
                    <button
                      className="small ghost danger"
                      disabled={disabled}
                      onClick={() => void deleteProject(p)}
                      title="Permanently delete project and its indexed data"
                    >
                      Delete project…
                    </button>
                  </span>
                </div>
                <table className="storage-table">
                  <tbody>
                    {KINDS.filter((k) => k.kind !== "temp").map(({ kind, label, actionLabel, isDestructive }) => (
                      <tr key={kind}>
                        <td className="name">{label}</td>
                        <td className="num">{humanSize(p.usage[kind])}</td>
                        <td className="bar-col"><Bar bytes={p.usage[kind]} of={r.total} /></td>
                        <td className="actions">
                          <button
                            className={`small ghost ${isDestructive ? "danger" : ""}`}
                            disabled={disabled || p.usage[kind] === 0}
                            onClick={() => void clean([kind], p.usage[kind], p)}
                            title={`${actionLabel} ${label.toLowerCase()} for this project`}
                          >
                            {actionLabel}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {p.shared > 0 && (
                  <div className="muted small">{humanSize(p.shared)} of this is footage other projects use too.</div>
                )}
              </section>
            );
          })}
        </>
      )}
    </main>
  );
}
