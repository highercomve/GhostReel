import { useEffect, useState } from "react";
import {
  executeOrganizeProject,
  planOrganizeProject,
  CATEGORIES,
  type OrganizePlan,
  type OrganizeReport,
} from "./api";

export default function OrganizeModal({
  projectId,
  onClose,
  onOrganized,
}: {
  projectId: number;
  onClose: () => void;
  onOrganized: () => void;
}) {
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [plan, setPlan] = useState<OrganizePlan | null>(null);
  const [report, setReport] = useState<OrganizeReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    planOrganizeProject(projectId)
      .then((p) => {
        if (active) setPlan(p);
      })
      .catch((e) => {
        if (active) setError(String(e));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [projectId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !executing) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, executing]);

  const onExecute = async () => {
    setExecuting(true);
    setError(null);
    try {
      const rep = await executeOrganizeProject(projectId);
      setReport(rep);
      onOrganized();
    } catch (e) {
      setError(String(e));
    } finally {
      setExecuting(false);
    }
  };

  // Group items by category folder
  const categoryCounts: Record<string, number> = {};
  let totalSidecars = 0;
  if (plan) {
    for (const item of plan.items) {
      if (!item.already_organized && !item.conflict) {
        categoryCounts[item.category_folder] = (categoryCounts[item.category_folder] || 0) + 1;
        totalSidecars += item.sidecars.length;
      }
    }
  }

  return (
    <div className="chat-log-overlay" onClick={() => !executing && onClose()}>
      <div className="chat-log card organize-modal" onClick={(e) => e.stopPropagation()}>
        <div className="chat-log-header">
          <span className="label">Organize Files on Disk</span>
          <button type="button" className="small" onClick={onClose} disabled={executing} title="Close">
            ×
          </button>
        </div>

        <div className="organize-modal-body">
          {error && <div className="banner bad">{error}</div>}

          {loading && <p className="muted">Scanning project footage and planning organization…</p>}

          {!loading && report && (
            <div className="organize-success-state">
              <div className="good-text" style={{ fontSize: "1.2rem", marginBottom: "0.5rem" }}>
                {report.errors.length ? "Organization completed with errors" : "✓ Files Organized Successfully"}
              </div>
              <p>
                Moved <strong>{report.moved_videos}</strong> video files and{" "}
                <strong>{report.moved_sidecars}</strong> companion sidecar files into categorized subfolders.
              </p>
              {report.skipped > 0 && (
                <p className="muted small">
                  {report.skipped} files were already organized or skipped.
                </p>
              )}
              {report.errors.length > 0 && (
                <div className="banner bad" style={{ marginTop: "1rem" }}>
                  <strong>Warnings:</strong>
                  <ul>
                    {report.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div style={{ marginTop: "1.5rem" }}>
                <button type="button" onClick={onClose}>
                  Done
                </button>
              </div>
            </div>
          )}

          {!loading && !report && plan && (
            <>
              <p>
                GhostReel will organize your original video files on your system into categorized subfolders
                inside each watched directory (e.g. <code>Interviews/</code>, <code>B-Roll/</code>, <code>Screen Recordings/</code>, <code>Drones & Aerials/</code>, <code>Music & Audio/</code>, <code>Other/</code>).
              </p>

              {plan.to_move_count === 0 ? (
                <div className="banner" style={{ margin: "1rem 0" }}>
                  All video files in this project are already in their categorized subfolders!
                </div>
              ) : (
                <div className="organize-summary-box">
                  <h3>Folders to create / organize:</h3>
                  <div className="organize-category-chips">
                    {Object.entries(categoryCounts).map(([folder, count]) => {
                      const matched = CATEGORIES.find((c) => c.folder === folder);
                      return (
                        <div key={folder} className="organize-chip">
                          <span className="icon">{matched?.icon ?? "📁"}</span>
                          <span className="name">{folder}</span>
                          <span className="count">{count} {count === 1 ? "file" : "files"}</span>
                        </div>
                      );
                    })}
                  </div>

                  {totalSidecars > 0 && (
                    <p className="muted small" style={{ marginTop: "0.75rem" }}>
                      ℹ️ Also moving <strong>{totalSidecars}</strong> companion sidecar {totalSidecars === 1 ? "file" : "files"} (.xmp, .wav, .srt, .thm) alongside videos.
                    </p>
                  )}

                  {plan.already_organized_count > 0 && (
                    <p className="muted small">
                      {plan.already_organized_count} files are already in their correct category subfolder and will not be moved.
                    </p>
                  )}

                  {plan.conflict_count > 0 && (
                    <div className="banner bad small" style={{ marginTop: "0.5rem" }}>
                      ⚠️ {plan.conflict_count} file(s) have naming collisions at their destination and will be skipped.
                    </div>
                  )}
                </div>
              )}

              <div style={{ margin: "1rem 0" }}>
                <button
                  type="button"
                  className="ghost small"
                  onClick={() => setShowDetails(!showDetails)}
                >
                  {showDetails ? "Hide file move details ▲" : `Show all ${plan.items.length} files ▼`}
                </button>

                {showDetails && (
                  <div className="organize-file-list">
                    <table>
                      <thead>
                        <tr>
                          <th>File</th>
                          <th>Category</th>
                          <th>Target Subfolder</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.items.map((item, idx) => (
                          <tr key={idx}>
                            <td className="small" title={item.current_path}>
                              {item.file_name}
                              {item.sidecars.length > 0 && (
                                <span className="muted"> (+{item.sidecars.length} sidecars)</span>
                              )}
                            </td>
                            <td>
                              <span className={`cat-badge cat-${item.category}`}>
                                {item.category}
                              </span>
                            </td>
                            <td className="small muted">
                              {item.category_folder}/
                            </td>
                            <td className="small">
                              {item.already_organized ? (
                                <span className="muted">already organized</span>
                              ) : item.conflict ? (
                                <span className="bad-text" title={item.conflict}>collision</span>
                              ) : (
                                <span className="good-text">ready to move</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="organize-modal-footer">
                <button type="button" className="ghost" onClick={onClose} disabled={executing}>
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={onExecute}
                  disabled={executing || plan.to_move_count === 0}
                >
                  {executing ? "Organizing files…" : `Organize ${plan.to_move_count} ${plan.to_move_count === 1 ? "File" : "Files"}`}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
