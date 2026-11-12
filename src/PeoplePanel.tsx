import { Clock, Film, Merge, Mic, UserRound } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { confirmAction } from "./dialogs";
import MediaImage from "./MediaImage";
import { mergePeople, personThumbnail, projectPeople, setPerson, type PersonSummary } from "./api";

function minutes(s: number): string {
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

const label = (p: { id: number; name: string | null }) => p.name || `Person ${p.id}`;

function Thumb({ projectId, personId, version }: { projectId: number; personId: number; version: number }) {
  const [path, setPath] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    setPath(undefined);
    personThumbnail(projectId, personId)
      .then((p) => { if (!cancelled) setPath(p); })
      .catch(() => { if (!cancelled) setPath(null); });
    return () => { cancelled = true; };
  }, [projectId, personId, version]);
  if (path) return <MediaImage className="person-thumb" path={path} alt="" />;
  return (
    <span className="person-thumb person-thumb-empty" aria-hidden>
      <UserRound size={36} strokeWidth={1.5} />
    </span>
  );
}

/** One editable field: saves on blur or Enter, Escape restores. */
function Field({ value, placeholder, onSave }: { value: string | null; placeholder: string; onSave: (v: string) => Promise<void> }) {
  const [text, setText] = useState(value ?? "");
  useEffect(() => setText(value ?? ""), [value]);
  const commit = () => { if (text.trim() !== (value ?? "")) onSave(text.trim()); };
  return (
    <input
      type="text"
      value={text}
      placeholder={placeholder}
      aria-label={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") setText(value ?? "");
      }}
    />
  );
}

/** People in a project: one card per person (face, name tag fields, how much they are seen and heard). */
export default function PeoplePanel({ projectId, refreshKey }: { projectId: number; refreshKey: number }) {
  const [people, setPeople] = useState<PersonSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mergeFrom, setMergeFrom] = useState<number | null>(null);
  const [thumbVersion, setThumbVersion] = useState(0);

  const refresh = useCallback(async () => {
    try {
      setPeople(await projectPeople(projectId));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [projectId]);
  useEffect(() => { refresh(); }, [refresh, refreshKey]);

  const save = async (id: number, fields: { name?: string; handle?: string; role?: string }) => {
    try {
      await setPerson(projectId, id, fields);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const merge = async (keep: PersonSummary, other: PersonSummary) => {
    const ok = await confirmAction(
      `Merge ${label(other)} into ${label(keep)}? Their faces and speech become one person; ${label(keep)}'s name is kept.`,
      { title: "Merge people", destructive: true },
    );
    if (!ok) return;
    try {
      await mergePeople(projectId, keep.id, other.id);
      setMergeFrom(null);
      setThumbVersion((v) => v + 1);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  if (people == null) return error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>;

  const source = people.find((p) => p.id === mergeFrom) ?? null;
  return (
    <>
      {error && <div className="banner bad">{error}</div>}
      <p className="muted" style={{ margin: "0 0 1rem" }}>
        Faces grouped by person, with the voice that belongs to each. The name, handle and role appear on name tags.
      </p>
      {people.length === 0 ? (
        <section className="card">
          <div className="muted">
            No people yet. Run Faces and then Speakers from the project menu; people appear here once both are done.
          </div>
        </section>
      ) : (
        <>
          {source && (
            <div className="banner people-merge-banner">
              <span>Choose the person to merge {label(source)} into.</span>
              <button type="button" className="small" onClick={() => setMergeFrom(null)}>Cancel</button>
            </div>
          )}
          <div className="people-grid">
            {people.map((p) => {
              const isSource = mergeFrom === p.id;
              return (
                <section key={p.id} className={`card person-card ${isSource ? "selected" : ""}`}>
                  <div className="person-card-head">
                    <Thumb projectId={projectId} personId={p.id} version={thumbVersion} />
                    <div className="person-fields">
                      <Field value={p.name} placeholder="Name" onSave={(v) => save(p.id, { name: v })} />
                      <Field value={p.handle} placeholder="Handle (@name)" onSave={(v) => save(p.id, { handle: v })} />
                      <Field value={p.role} placeholder="Role (e.g. Host)" onSave={(v) => save(p.id, { role: v })} />
                    </div>
                  </div>
                  <div className="person-stats small muted">
                    <span title="Time on screen"><Clock size={13} strokeWidth={1.75} /> {minutes(p.screen_time_s)} on screen</span>
                    <span title="Time speaking"><Mic size={13} strokeWidth={1.75} /> {p.speaking_time_s > 0 ? `${minutes(p.speaking_time_s)} speaking` : "voice not linked"}</span>
                    <span title="Videos"><Film size={13} strokeWidth={1.75} /> {p.videos.length} video{p.videos.length === 1 ? "" : "s"}</span>
                  </div>
                  <div className="person-actions">
                    {source && !isSource ? (
                      <button type="button" className="small primary" onClick={() => merge(p, source)}>
                        Merge {label(source)} here
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="small ghost"
                        disabled={people.length < 2}
                        onClick={() => setMergeFrom(isSource ? null : p.id)}
                        title="This is the same person as another card"
                      >
                        <Merge size={13} strokeWidth={1.75} /> {isSource ? "Cancel merge" : "Merge into…"}
                      </button>
                    )}
                  </div>
                </section>
              );
            })}
          </div>
        </>
      )}
    </>
  );
}
