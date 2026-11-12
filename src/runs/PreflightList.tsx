import { useEffect, useState } from "react";
import { CircleAlert, TriangleAlert } from "lucide-react";
import type { Preflight, PreflightCheck } from "../api";
import { agentErrorLabel } from "../agents/AgentErrorView";

/**
 * What a run would run into before it starts (`preflight::Preflight`): errors that stop it and
 * warnings worth knowing. Checks that passed are not listed. Any feature's form can show it.
 */
export default function PreflightList({ checks }: { checks: PreflightCheck[] }) {
  const shown = checks.filter((c) => c.level !== "ok");
  if (shown.length === 0) return null;
  return (
    <ul className="preflight-list" aria-label="Before you start">
      {shown.map((c, i) => (
        <li key={`${c.subject}-${i}`} className={`preflight-item level-${c.level}`}>
          {c.level === "error" ? (
            <CircleAlert size={13} strokeWidth={2} aria-label="Blocks the run" />
          ) : (
            <TriangleAlert size={13} strokeWidth={2} aria-label="Warning" />
          )}
          <span className="preflight-text">
            {c.error && <span className={`agent-error-kind kind-${c.error.kind}`}>{agentErrorLabel(c.error)}</span>}
            {c.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Run a feature's preflight while its form is open, again whenever `key` changes. `load` null =
 * nothing to check (the form is closed). A failed check call is shown as a warning, never blocks.
 */
export function usePreflight(load: (() => Promise<Preflight>) | null, key: string): {
  checks: PreflightCheck[];
  checking: boolean;
  blocked: boolean;
} {
  const [checks, setChecks] = useState<PreflightCheck[]>([]);
  const [checking, setChecking] = useState(false);
  const active = load != null;
  useEffect(() => {
    if (!load) {
      setChecks([]);
      return;
    }
    let alive = true;
    setChecking(true);
    // Typing in the form should not fire a check per keystroke.
    const t = window.setTimeout(() => {
      load()
        .then((p) => alive && setChecks(p.checks))
        .catch((e) => alive && setChecks([{ level: "warn", scope: "project", subject: "preflight", message: `Could not check before starting: ${String(e)}` }]))
        .finally(() => alive && setChecking(false));
    }, 250);
    return () => {
      alive = false;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, active]);
  return { checks, checking, blocked: checks.some((c) => c.level === "error") };
}
