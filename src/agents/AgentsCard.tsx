import { useRef, useState } from "react";
import { confirmAction } from "../dialogs";
import { ArrowDown, ArrowUp, Lock, Minus, Plus, Trash2 } from "lucide-react";
import ModelSelect from "../editor/ModelSelect";
import {
  EFFORTS,
  EFFORT_TOOLS,
  saveAgents,
  testAgent,
  type AgentAvailability,
  type AgentRole,
  type AgentSelection,
  type AgentSpec,
  type AgentView,
  type AgentsView,
  type Backend,
  type Effort,
} from "../api";
import { agentErrorLabel } from "./AgentErrorView";
import AgentPicker from "./AgentPicker";
import { announceAgentsChanged, useAgents } from "./useAgents";
import "./agents.css";

const ROLE_LABEL: Record<AgentRole, string> = {
  orchestrator: "Orchestrator",
  worker: "Worker",
  chat: "Chat",
  vision: "Frames",
};
const ROLE_HINT: Record<AgentRole, string> = {
  orchestrator: "Makes the final choice from the workers' answers (clips)",
  worker: "Takes small parallel jobs (scoring clip windows)",
  chat: "Writes and revises cuts in Sonar",
  vision: "Describes keyframes while indexing",
};
const TOOL_LABEL: Record<string, string> = { claude: "Claude", agy: "agy", opencode: "OpenCode", codex: "Codex" };

/** A fresh id from `base`: "codex", then "codex-2", "codex-3"… */
function freshId(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}-${i}`)) return `${base}-${i}`;
}

/** Settings → Agents: every agent with its backend, model, calls at once, roles, order and a
 *  check, plus the app-wide default choice every run and cut starts from. Saves on each change. */
export default function AgentsCard() {
  const { view, error: loadError, setView } = useAgents();
  const [error, setError] = useState<string | null>(null);
  const [tests, setTests] = useState<Record<string, AgentAvailability | "testing">>({});
  // The agents as of the latest render, for `remove`, which waits on a dialog and must act on
  // what is there afterwards, not on what the click saw.
  const latestView = useRef(view);
  latestView.current = view;

  if (!view) return <div className="card">{loadError ? <span className="bad-text">{loadError}</span> : <span className="muted">Reading the agents…</span>}</div>;

  const specs = view.agents.map((a) => a.spec);

  const save = async (agents: AgentSpec[], selection: AgentSelection = view.app_selection) => {
    try {
      const next: AgentsView = await saveAgents(agents, selection);
      setView(next);
      void announceAgentsChanged(next);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  };
  const patch = (id: string, p: Partial<AgentSpec>) => save(specs.map((s) => (s.id === id ? { ...s, ...p } : s)));
  const move = (i: number, by: number) => {
    const order = [...specs];
    const [a] = order.splice(i, 1);
    order.splice(i + by, 0, a);
    void save(order.map((s, k) => ({ ...s, priority: (k + 1) * 10 })));
  };
  const add = () => {
    const id = freshId("claude", specs.map((s) => s.id));
    void save([
      ...specs,
      {
        id,
        backend: "cli",
        tool: "claude",
        max_concurrent: 2,
        roles: ["orchestrator", "worker", "chat"],
        enabled: true,
        priority: (specs.length + 1) * 10,
      },
    ]);
  };
  const remove = async (id: string) => {
    if (!(await confirmAction(`Remove the agent “${id}”?`, { okLabel: "Remove", destructive: true }))) return;
    const now = latestView.current;
    if (!now || !now.agents.some((a) => a.spec.id === id)) return;
    const sel = now.app_selection;
    void save(
      now.agents.map((a) => a.spec).filter((s) => s.id !== id),
      {
        ...sel,
        orchestrator: sel.orchestrator === id ? null : sel.orchestrator,
        workers: sel.workers?.filter((w) => w !== id).length ? sel.workers.filter((w) => w !== id) : null,
      },
    );
  };
  const test = async (id: string) => {
    setTests((t) => ({ ...t, [id]: "testing" }));
    try {
      const r = await testAgent(id);
      setTests((t) => ({ ...t, [id]: r }));
    } catch (e) {
      setTests((t) => ({ ...t, [id]: { id, ok: false, level: "error", resolved: "", detail: String(e), model_listed: null } }));
    }
  };

  return (
    <div className="card agents-card">
      {view.implicit && (
        <p className="muted small">
          These agents come from your model settings above. Changing anything here saves them as your agent list.
        </p>
      )}
      {(error || view.error) && <div className="banner bad small">{error ?? view.error}</div>}
      <ol className="agents-list">
        {view.agents.map((a, i) => (
          <AgentRow
            key={a.spec.id}
            agent={a}
            first={i === 0}
            last={i === view.agents.length - 1}
            roles={view.roles}
            tools={view.cli_tools}
            test={tests[a.spec.id]}
            onPatch={(p) => void patch(a.spec.id, p)}
            onMove={(by) => move(i, by)}
            onTest={() => void test(a.spec.id)}
            onRemove={view.agents.length > 1 ? () => remove(a.spec.id) : undefined}
          />
        ))}
      </ol>
      <div className="row">
        <button type="button" className="ghost small" onClick={add}>
          <Plus size={14} strokeWidth={2} aria-hidden /> Add agent
        </button>
      </div>
      <div className="agents-default">
        <div className="settings-card-title">Default for runs and cuts</div>
        <p className="muted small">A cut or a single run can choose differently; what it leaves out comes from here.</p>
        <AgentPicker level="app" value={view.app_selection} onChange={(sel) => void save(specs, sel ?? {})} />
      </div>
    </div>
  );
}

function AgentRow(props: {
  agent: AgentView;
  first: boolean;
  last: boolean;
  roles: AgentRole[];
  tools: string[];
  test: AgentAvailability | "testing" | undefined;
  onPatch(p: Partial<AgentSpec>): void;
  onMove(by: number): void;
  onTest(): void;
  onRemove?: () => void;
}) {
  const { agent, test, onPatch } = props;
  const s = agent.spec;
  const linked = agent.linked_to != null;
  const locked = agent.capacity_locked != null;
  const cap = agent.capacity;
  return (
    <li className={`agent-row${s.enabled ? "" : " off"}`}>
      <div className="agent-row-head">
        <label className="agent-enabled" title={s.enabled ? "Enabled" : "Disabled"}>
          <input type="checkbox" checked={s.enabled} onChange={(e) => onPatch({ enabled: e.target.checked })} />
          <span className="sr-only">Enabled</span>
        </label>
        <input
          className="agent-name"
          defaultValue={s.name || s.id}
          aria-label="Agent name"
          onBlur={(e) => {
            const v = e.currentTarget.value.trim();
            if (v !== (s.name || s.id)) onPatch({ name: v === s.id ? "" : v });
          }}
        />
        <span className="agent-id small muted" title="Id used by runs, cuts and the CLI">
          {s.id}
        </span>
        <span className="agent-summary small muted">{agent.summary}</span>
        <span className="agent-row-actions">
          <button type="button" className="nle-icon-btn" disabled={props.first} onClick={() => props.onMove(-1)} title="Earlier: gets jobs first" aria-label="Move up">
            <ArrowUp size={14} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="nle-icon-btn" disabled={props.last} onClick={() => props.onMove(1)} title="Later" aria-label="Move down">
            <ArrowDown size={14} strokeWidth={2} aria-hidden />
          </button>
          <button type="button" className="ghost small" disabled={test === "testing"} onClick={props.onTest}>
            {test === "testing" ? "Testing…" : "Test"}
          </button>
          {props.onRemove && (
            <button type="button" className="nle-icon-btn" onClick={props.onRemove} title="Remove" aria-label={`Remove ${s.id}`}>
              <Trash2 size={14} strokeWidth={2} aria-hidden />
            </button>
          )}
        </span>
      </div>

      <div className="agent-row-body">
        {linked ? (
          <span className="small muted agent-linked">
            <Lock size={12} strokeWidth={2} aria-hidden /> Model set in “{agent.linked_to}” above
          </span>
        ) : (
          <>
            <select
              value={s.backend}
              aria-label="Backend"
              onChange={(e) => onPatch({ backend: e.target.value as Backend })}
            >
              <option value="cli">CLI agent</option>
              <option value="server">Server</option>
              <option value="local">Local model</option>
            </select>
            {s.backend === "cli" && (
              <>
                <select value={s.tool ?? ""} aria-label="Tool" onChange={(e) => onPatch({ tool: e.target.value })}>
                  {props.tools.map((t) => (
                    <option key={t} value={t}>
                      {TOOL_LABEL[t] ?? t}
                    </option>
                  ))}
                </select>
                <ModelSelect kind="cli" tool={s.tool ?? ""} value={s.model ?? ""} placeholder="the tool's default" onChange={(model) => onPatch({ model })} />
                {EFFORT_TOOLS.includes(s.tool ?? "") && (
                  <select
                    className="agent-effort"
                    value={s.effort ?? ""}
                    aria-label="Effort"
                    title="How hard the model thinks before it answers"
                    onChange={(e) => onPatch({ effort: (e.target.value || null) as Effort | null })}
                  >
                    <option value="">Default effort</option>
                    {EFFORTS.map((x) => (
                      <option key={x} value={x}>
                        {x} effort
                      </option>
                    ))}
                  </select>
                )}
              </>
            )}
            {s.backend === "server" && (
              <>
                <input
                  className="agent-url"
                  defaultValue={s.url ?? ""}
                  placeholder="http://127.0.0.1:8771"
                  aria-label="Server URL"
                  onBlur={(e) => e.currentTarget.value !== (s.url ?? "") && onPatch({ url: e.currentTarget.value.trim() })}
                />
                <ModelSelect kind="server" url={s.url ?? ""} value={s.model ?? ""} placeholder="server default" onChange={(model) => onPatch({ model })} />
              </>
            )}
            {s.backend === "local" && (
              <input
                defaultValue={s.local_model ?? ""}
                placeholder="qwen3.5-9b"
                aria-label="Local model"
                onBlur={(e) => e.currentTarget.value !== (s.local_model ?? "") && onPatch({ local_model: e.currentTarget.value.trim() })}
              />
            )}
          </>
        )}
      </div>

      <div className="agent-row-body">
        <span className="agent-stepper" role="group" aria-label="Calls at once">
          <span className="small muted">At once</span>
          <button type="button" className="nle-icon-btn" disabled={locked || cap <= 1} onClick={() => onPatch({ max_concurrent: cap - 1 })} aria-label="Fewer at once">
            <Minus size={13} strokeWidth={2} aria-hidden />
          </button>
          <strong>{cap}</strong>
          <button type="button" className="nle-icon-btn" disabled={locked || cap >= 32} onClick={() => onPatch({ max_concurrent: cap + 1 })} aria-label="More at once">
            <Plus size={13} strokeWidth={2} aria-hidden />
          </button>
        </span>
        <span className="agent-roles" role="group" aria-label="Roles">
          {props.roles.map((r) => {
            const on = s.roles.includes(r);
            return (
              <button
                key={r}
                type="button"
                className={`nle-preset-chip${on ? " on" : ""}`}
                aria-pressed={on}
                title={ROLE_HINT[r]}
                onClick={() => onPatch({ roles: on ? s.roles.filter((x) => x !== r) : [...s.roles, r] })}
              >
                {ROLE_LABEL[r]}
              </button>
            );
          })}
        </span>
      </div>
      {locked && <div className="small muted">{agent.capacity_locked}</div>}
      {test && test !== "testing" && (
        <div className={`small ${test.level === "ok" ? "good-text" : test.level === "warn" ? "warn-text" : "bad-text"}`}>
          {test.error && <span className={`agent-error-kind kind-${test.error.kind}`}>{agentErrorLabel(test.error)}</span>}{" "}
          {test.level === "ok" ? "Available" : test.level === "warn" ? "Available, but" : "Not available"} · {test.detail}
        </div>
      )}
    </li>
  );
}
