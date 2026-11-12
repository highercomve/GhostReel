import { useId, useState } from "react";
import { Minus, Plus, Settings2 } from "lucide-react";
import { saveAgents, type AgentSelection, type AgentStatus, type AgentView, type WorkTarget } from "../api";
import { navigate } from "../navigate";
import { announceAgentsChanged, useAgents, useAgentStatus, useResolvedAgents } from "./useAgents";
import "./agents.css";

type Level = "run" | "cut" | "app";

interface Props {
  /** Which level this choice is: a run's overrides a cut's, which overrides the app's. */
  level: Level;
  /** `null` = no choice of its own at this level (it follows the default). */
  value: AgentSelection | null | undefined;
  onChange(next: AgentSelection | null): void;
  /** What the run makes, so the summary can name it ("picks the 8 clips (9:16)"). */
  target?: WorkTarget;
  disabled?: boolean;
  /** Called before "Manage agents…" leaves the page (e.g. to close a dialog). */
  onManage?: () => void;
}

/** Most calls one agent may take at once (`agents::MAX_CONCURRENT`). */
const MAX_AT_ONCE = 32;

/** Why a run with this choice cannot start, or `null`. */
export function agentSelectionProblem(sel: AgentSelection | null | undefined): string | null {
  return sel?.workers && sel.workers.length === 0 ? "Turn on at least one agent to score the episode." : null;
}

const NOTE: Record<Level, [string, string]> = {
  run: ["Same as the app default.", "Changed for this run."],
  cut: ["Follows the app default.", "Set for this cut."],
  app: ["", ""],
};

/**
 * Choose who does a clip run's work: one agent picks the final clips (or none: code picks by
 * score), and every worker agent that is on scores the episode, each taking so many calls at a
 * time. Always editable, starting from the default; the sentence under it is the backend's own
 * reading of what will run. Used for a run, a cut and the app default.
 */
export default function AgentPicker({ level, value, onChange, target, disabled, onManage }: Props) {
  const uid = useId();
  const { view, error: listError } = useAgents();
  const statuses = useAgentStatus();
  const [saveError, setSaveError] = useState<string | null>(null);
  const problem = agentSelectionProblem(value);
  const { resolved, error } = useResolvedAgents(
    level === "cut" ? null : problem ? null : value,
    level === "cut" ? (problem ? null : value) : null,
    view,
    target,
  );

  if (!view) return <div className="agent-picker small muted">{listError ?? "Reading the agents…"}</div>;

  const usable = (role: "orchestrator" | "worker") => view.agents.filter((a) => a.spec.enabled && a.spec.roles.includes(role));
  const orchestrators = usable("orchestrator");
  const workers = usable("worker");
  // null = still checking; undefined = no status for it (it could not be read).
  const status = (id: string) => (statuses == null ? null : statuses.find((s) => s.id === id));

  // What the picker shows: this level's own choice where it has one, else what resolves.
  const sel = value ?? {};
  const picksByScore = sel.pick_by_score === true || (!sel.orchestrator && resolved != null && resolved.orchestrator == null);
  const orchestrator = picksByScore ? null : (sel.orchestrator ?? resolved?.orchestrator ?? null);
  const on = new Set(sel.workers ?? resolved?.workers ?? []);
  const atOnce = (a: AgentView) =>
    a.capacity_locked ? 1 : (sel.concurrency?.[a.spec.id] ?? resolved?.concurrency[a.spec.id] ?? a.capacity);

  /** The whole choice as shown, with `patch` applied: what an edit or "Save as default" stores. */
  const explicit = (patch: { orchestrator?: string | null; workers?: string[]; concurrency?: Record<string, number> } = {}): AgentSelection => {
    const orch = "orchestrator" in patch ? patch.orchestrator : orchestrator;
    const conc: Record<string, number> = {};
    for (const a of workers) {
      const n = patch.concurrency?.[a.spec.id] ?? atOnce(a);
      if (!a.capacity_locked && n !== a.capacity) conc[a.spec.id] = n;
    }
    const ids = patch.workers ?? workers.map((a) => a.spec.id).filter((id) => on.has(id));
    return {
      ...(orch ? { orchestrator: orch } : { pick_by_score: true }),
      workers: ids,
      ...(Object.keys(conc).length ? { concurrency: conc } : {}),
    };
  };

  const toggle = (id: string) => {
    const next = on.has(id) ? [...on].filter((w) => w !== id) : [...on, id];
    onChange(explicit({ workers: workers.map((a) => a.spec.id).filter((w) => next.includes(w)) }));
  };
  const step = (a: AgentView, by: number) => {
    const n = Math.min(MAX_AT_ONCE, Math.max(1, atOnce(a) + by));
    onChange(explicit({ concurrency: { [a.spec.id]: n } }));
  };
  const saveAsDefault = async () => {
    try {
      const next = await saveAgents(view.agents.map((a) => a.spec), explicit());
      void announceAgentsChanged(next);
      setSaveError(null);
      onChange(null);
    } catch (e) {
      setSaveError(String(e));
    }
  };
  const manage = () => {
    onManage?.();
    navigate("models", "agents");
  };

  const orchView = orchestrators.find((a) => a.spec.id === orchestrator) ?? null;
  // At the app and cut levels a choice is stored at once, so the last worker stays on.
  const lockLast = level !== "run" && on.size <= 1;
  const changed = value != null;

  return (
    <div className={`agent-picker${disabled ? " disabled" : ""}`}>
      <div className="agent-field">
        <label className="agent-field-label" htmlFor={`${uid}-orch`}>
          Picks the final clips
        </label>
        <div className="agent-orch">
          <select
            id={`${uid}-orch`}
            value={orchestrator ?? ""}
            disabled={disabled}
            aria-describedby={`${uid}-orch-detail`}
            onChange={(e) => onChange(explicit({ orchestrator: e.target.value || null }))}
          >
            {orchestrators.map((a) => {
              const s = status(a.spec.id);
              return (
                <option key={a.spec.id} value={a.spec.id}>
                  {a.display_name}
                  {s && s.state !== "ready" ? ` (${statusText(s)})` : ""}
                </option>
              );
            })}
            <option value="">No model — pick by score (fastest)</option>
          </select>
          {orchView && <StatusDot status={status(orchView.spec.id)} />}
        </div>
        <span className="agent-secondary" id={`${uid}-orch-detail`}>
          {orchView ? orchView.summary : "The best-scored moments become the clips; no model reads them again."}
        </span>
      </div>

      <fieldset className="agent-field agent-workers-field" disabled={disabled}>
        <legend className="agent-field-label">Scores the episode</legend>
        {workers.length === 0 ? (
          <span className="small muted">No enabled agent can score. Give one the worker role under Manage agents.</span>
        ) : (
          <ul className="agent-workers">
            {workers.map((a) => {
              const isOn = on.has(a.spec.id);
              const n = atOnce(a);
              const nameId = `${uid}-w-${a.spec.id}`;
              const lockedOn = isOn && lockLast;
              return (
                <li key={a.spec.id} className={`agent-worker${isOn ? "" : " off"}`}>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={isOn}
                    aria-labelledby={nameId}
                    className={`agent-switch${isOn ? " on" : ""}`}
                    disabled={lockedOn}
                    title={lockedOn ? "At least one agent scores the episode" : isOn ? "On: scores the episode" : "Off"}
                    onClick={() => toggle(a.spec.id)}
                  >
                    <span className="agent-switch-knob" />
                  </button>
                  <span className="agent-worker-name">
                    <span id={nameId} className="agent-worker-title">
                      {a.display_name}
                    </span>
                    <span className="agent-secondary">{a.summary}</span>
                  </span>
                  <span
                    className="agent-stepper"
                    role="group"
                    aria-label={`${a.display_name}: calls at a time`}
                    title={a.capacity_locked ?? undefined}
                  >
                    <button
                      type="button"
                      className="nle-icon-btn"
                      disabled={!isOn || a.capacity_locked != null || n <= 1}
                      onClick={() => step(a, -1)}
                      aria-label={`Fewer at a time for ${a.display_name}`}
                    >
                      <Minus size={13} strokeWidth={2} aria-hidden />
                    </button>
                    <strong aria-live="polite">{n}</strong>
                    <button
                      type="button"
                      className="nle-icon-btn"
                      disabled={!isOn || a.capacity_locked != null || n >= MAX_AT_ONCE}
                      onClick={() => step(a, 1)}
                      aria-label={`More at a time for ${a.display_name}`}
                    >
                      <Plus size={13} strokeWidth={2} aria-hidden />
                    </button>
                    <span className="agent-secondary">at a time</span>
                  </span>
                  <StatusDot status={status(a.spec.id)} />
                </li>
              );
            })}
          </ul>
        )}
      </fieldset>

      <p className="agent-summary-line" aria-live="polite">
        {problem ? (
          <span className="warn-text">{problem}</span>
        ) : error ? (
          <span className="bad-text">{error}</span>
        ) : resolved ? (
          resolved.summary
        ) : (
          <span className="muted">Working out which agents run…</span>
        )}
      </p>
      {saveError && <p className="bad-text small">{saveError}</p>}

      <div className="agent-picker-foot">
        {level !== "app" && <span className="agent-note">{NOTE[level][changed ? 1 : 0]}</span>}
        {level !== "app" && changed && (
          <button type="button" className="link-btn" disabled={disabled} onClick={() => onChange(null)}>
            Reset to default
          </button>
        )}
        {level !== "app" && changed && (
          <button type="button" className="link-btn" disabled={disabled || problem != null} onClick={() => void saveAsDefault()}>
            Save as default
          </button>
        )}
        {level !== "app" && (
          <button type="button" className="link-btn agent-manage" onClick={manage}>
            <Settings2 size={13} strokeWidth={2} aria-hidden /> Manage agents…
          </button>
        )}
      </div>
    </div>
  );
}

/** "ready", "busy", "out of quota until 14:05"… */
function statusText(s: AgentStatus): string {
  if (s.state === "out_of_quota") {
    const at = new Date(s.until * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return `${s.label} until ${at}`;
  }
  return s.label;
}

const TONE: Record<AgentStatus["state"], string> = {
  ready: "good",
  busy: "warn",
  out_of_quota: "warn",
  not_installed: "bad",
  unreachable: "bad",
  unavailable: "bad",
};

/** A coloured dot and the state in words; the detail on hover. Nothing while it is checked. */
function StatusDot({ status }: { status: AgentStatus | null | undefined }) {
  if (status === undefined) return null;
  if (status === null) return <span className="agent-status checking">checking…</span>;
  return (
    <span className={`agent-status ${TONE[status.state]}`} title={status.detail}>
      <span className="agent-status-dot" aria-hidden />
      {statusText(status)}
    </span>
  );
}
