import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { AgentError } from "../api";
import "./agents.css";

/** "2 h 3 min", "45 min", "30 s" (the backend's `human_duration`). */
function duration(s: number): string {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return m === 0 || m === 60 ? `${h + m / 60} h` : `${h} h ${m} min`;
}

/** The chip's words for each kind of failure. */
export function agentErrorLabel(e: AgentError): string {
  switch (e.kind) {
    case "invalid_model":
      return e.model ? `model ${e.model} not available` : "model not available";
    case "quota_exhausted":
      return e.resets_in_s != null ? `quota used up · resets in ${duration(e.resets_in_s)}` : "quota used up";
    case "busy":
      return "busy";
    case "auth":
      return "not signed in";
    case "not_installed":
      return "not installed";
    case "timeout":
      return e.after_s ? `timed out after ${e.after_s} s` : "timed out";
    case "bad_answer":
      return "unusable answer";
    case "unreachable":
      return "not reachable";
    case "http":
      return `HTTP ${e.status}`;
    case "exit":
      return e.code != null ? `exit code ${e.code}` : "stopped";
    case "no_agent_left":
      return `no ${e.role} agent left`;
    default:
      return "failed";
  }
}

/**
 * A typed agent failure, compact: who, a kind chip and the tool's own line; the raw output (or,
 * when no agent was left, each agent's reason) folds out.
 */
export default function AgentErrorView({ error, compact = false }: { error: AgentError; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const nested = error.kind === "no_agent_left" ? error.agents : [];
  const hasDetails = nested.length > 0 || !!error.details;
  return (
    <div className="agent-error">
      <div className="agent-error-line">
        {error.kind !== "no_agent_left" && <strong className="agent-error-agent">{error.agent}</strong>}
        <span className={`agent-error-kind kind-${error.kind}`}>{agentErrorLabel(error)}</span>
        {error.kind === "no_agent_left" && (
          <span className="agent-error-message">every agent is out: {nested.map((n) => n.agent).join(", ")}</span>
        )}
        {error.kind !== "no_agent_left" && error.message && (
          <span className={`agent-error-message${compact ? " clamp" : ""}`}>{error.message}</span>
        )}
      </div>
      {hasDetails && (
        <button
          type="button"
          className="ghost small agent-error-toggle"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <ChevronDown size={13} aria-hidden="true" /> : <ChevronRight size={13} aria-hidden="true" />}
          {nested.length > 0 ? `${nested.length} ${nested.length === 1 ? "agent" : "agents"}` : "Details"}
        </button>
      )}
      {open &&
        (nested.length > 0 ? (
          <ul className="agent-error-list">
            {nested.map((n, i) => (
              <li key={`${n.agent}-${i}`}>
                <AgentErrorView error={n} compact />
              </li>
            ))}
          </ul>
        ) : (
          <pre className="agent-error-details">{error.details}</pre>
        ))}
    </div>
  );
}
