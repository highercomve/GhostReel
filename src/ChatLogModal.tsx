import { useEffect, useRef, useState } from "react";
import { chatLog, type LogEntry } from "./api";

/** Survives closing the log: reopening it fetches only what is new. */
const logCache = new Map<number, LogEntry[]>();

const KINDS: { kind: LogEntry["kind"]; label: string }[] = [
  { kind: "prompt", label: "Prompts" },
  { kind: "thinking", label: "Thinking" },
  { kind: "answer", label: "Answers" },
  { kind: "helper", label: "Helper output" },
];

/** A run of consecutive helper lines is one block; everything else is its own. */
type Block = { kind: LogEntry["kind"]; ts: number; text: string; key: number };

function toBlocks(entries: LogEntry[]): Block[] {
  const out: Block[] = [];
  for (const e of entries) {
    const last = out[out.length - 1];
    if (e.kind === "helper" && last?.kind === "helper") {
      last.text += `\n${e.text}`;
    } else {
      out.push({ kind: e.kind, ts: e.ts_ms, text: e.text, key: e.seq });
    }
  }
  return out;
}

function time(ms: number) {
  return new Date(ms).toLocaleTimeString([], { hour12: false });
}

/**
 * Everything the model was sent and said during this chat: prompts, thinking, answers, and the
 * local helper's own output. Polls while open, so a running turn fills in as it goes.
 */
export default function ChatLogModal({ sessionId, onClose }: { sessionId: number; onClose: () => void }) {
  const [entries, setEntries] = useState<LogEntry[]>(() => logCache.get(sessionId) ?? []);
  const [shown, setShown] = useState<Set<LogEntry["kind"]>>(() => new Set(["prompt", "thinking", "answer", "helper"]));
  const [error, setError] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      const have = logCache.get(sessionId) ?? [];
      const after = have.length > 0 ? have[have.length - 1].seq : 0;
      try {
        const fresh = await chatLog(sessionId, after);
        if (stopped) return;
        setError(null);
        if (fresh.length > 0) {
          const all = [...have, ...fresh];
          logCache.set(sessionId, all);
          setEntries(all);
        }
      } catch (e) {
        if (!stopped) setError(String(e));
      }
    };
    poll();
    const timer = setInterval(poll, 1000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [sessionId]);

  // Follow the end while the reader is at the end; leave them be once they scroll up.
  useEffect(() => {
    const el = bodyRef.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [entries, shown]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const visible = entries.filter((e) => shown.has(e.kind));
  const blocks = toBlocks(visible);

  const toggle = (kind: LogEntry["kind"]) =>
    setShown((s) => {
      const next = new Set(s);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });

  const copyAll = () => {
    const text = visible.map((e) => `--- ${time(e.ts_ms)} ${e.kind} ---\n${e.text}`).join("\n\n");
    navigator.clipboard?.writeText(text).catch(() => {});
  };

  return (
    <div className="chat-log-overlay" onClick={onClose}>
      <div className="chat-log card" onClick={(e) => e.stopPropagation()}>
        <div className="chat-log-header">
          <span className="label">Model log</span>
          <div className="chat-log-filters">
            {KINDS.map((k) => (
              <label key={k.kind} className="small">
                <input type="checkbox" checked={shown.has(k.kind)} onChange={() => toggle(k.kind)} /> {k.label}
              </label>
            ))}
          </div>
          <button type="button" className="small" onClick={copyAll} disabled={visible.length === 0}>
            Copy
          </button>
          <button type="button" className="small" onClick={onClose} title="Close">
            ×
          </button>
        </div>
        <div
          className="chat-log-body"
          ref={bodyRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
        >
          {error && <div className="chat-log-error small">{error}</div>}
          {blocks.length === 0 && (
            <div className="muted small">
              Nothing logged for this chat yet. The log covers turns run since the app started.
            </div>
          )}
          {blocks.map((b, i) => (
            <details
              key={b.key}
              className={`chat-log-entry ${b.kind}`}
              // Prompts are the whole project's speech: open on request. The newest stays open.
              open={b.kind !== "prompt" || i === blocks.length - 1}
            >
              <summary>
                <span className="chat-log-kind">{b.kind}</span>
                <span className="muted small">
                  {time(b.ts)} · {b.text.length.toLocaleString()} chars
                </span>
              </summary>
              <pre>{b.text}</pre>
            </details>
          ))}
        </div>
      </div>
    </div>
  );
}
