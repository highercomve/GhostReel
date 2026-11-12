import { useEffect, useState } from "react";
import { emitEvent, onEvent } from "../events";
import {
  agentStatus,
  listAgents,
  resolveAgents,
  type AgentSelection,
  type AgentStatus,
  type AgentsView,
  type ResolvedAgents,
  type WorkTarget,
} from "../api";

/** Fired after the agent list or the app default changes, so every picker reloads. */
export const AGENTS_CHANGED = "agents-changed";

export const announceAgentsChanged = (view: AgentsView) => emitEvent(AGENTS_CHANGED, view);

/** The configured agents, kept fresh across pickers and windows. */
export function useAgents(): { view: AgentsView | null; error: string | null; setView: (v: AgentsView) => void } {
  const [view, setView] = useState<AgentsView | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      listAgents()
        .then((v) => {
          if (alive) {
            setView(v);
            setError(null);
          }
        })
        .catch((e) => alive && setError(String(e)));
    load();
    const off = onEvent<AgentsView>(AGENTS_CHANGED, (v) => alive && setView(v));
    return () => {
      alive = false;
      off();
    };
  }, []);
  return { view, error, setView };
}

/** What a run / cut choice resolves to, from the backend's one resolver. */
export function useResolvedAgents(
  run: AgentSelection | null | undefined,
  cut: AgentSelection | null | undefined,
  reloadKey: unknown,
  target?: WorkTarget | null,
): { resolved: ResolvedAgents | null; error: string | null } {
  const [resolved, setResolved] = useState<ResolvedAgents | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify([run ?? null, cut ?? null, target ?? null]);
  useEffect(() => {
    let alive = true;
    resolveAgents(run, cut, target)
      .then((r) => {
        if (alive) {
          setResolved(r);
          setError(null);
        }
      })
      .catch((e) => {
        if (alive) {
          setResolved(null);
          setError(String(e));
        }
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reloadKey]);
  return { resolved, error };
}

/** How often the agents' status is read again while a picker is open. */
const STATUS_EVERY_MS = 30_000;

/**
 * Where every agent stands now (ready, busy, out of quota, missing), read on mount, every 30 s
 * and after the agents change. `null` while the first read runs; `[]` when it cannot be read.
 */
export function useAgentStatus(): AgentStatus[] | null {
  const [list, setList] = useState<AgentStatus[] | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      agentStatus()
        .then((v) => alive && setList(v))
        .catch(() => alive && setList([]));
    load();
    const timer = window.setInterval(load, STATUS_EVERY_MS);
    const off = onEvent(AGENTS_CHANGED, () => void load());
    return () => {
      alive = false;
      window.clearInterval(timer);
      off();
    };
  }, []);
  return list;
}
