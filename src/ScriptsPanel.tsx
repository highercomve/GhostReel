import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  getAiSettings,
  setAiSettings,
  chatMessages,
  chatSessions,
  deleteChatSession,
  chatTurn,
  buildScriptWithJev,
  listScripts,
  type ChatEvent,
  type ChatMessage,
  type ChatProgress,
  type ChatSession,
  type Issue,
  type ScriptSummary,
  type ToolCallRecord,
} from "./api";
import ScriptEditor from "./ScriptEditor";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";

function formatRelativeTime(timestamp: number): string {
  const ms = timestamp < 1e11 ? timestamp * 1000 : timestamp;
  const diffSecs = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (diffSecs < 60) return "just now";
  if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
  if (diffSecs < 86400) return `${Math.floor(diffSecs / 3600)}h ago`;
  return `${Math.floor(diffSecs / 86400)}d ago`;
}

function formatToolCall(tc: ToolCallRecord): string {
  let argSnippet = "";
  if (tc.args) {
    if (typeof tc.args === "string") {
      argSnippet = `"${tc.args}"`;
    } else if (tc.args.query) {
      argSnippet = `"${tc.args.query}"`;
    } else {
      const firstVal = Object.values(tc.args)[0];
      if (firstVal && (typeof firstVal === "string" || typeof firstVal === "number")) {
        argSnippet = `"${firstVal}"`;
      }
    }
  }
  const summaryPart = tc.summary ? ` → ${tc.summary}` : "";
  return `${tc.tool}${argSnippet ? ` ${argSnippet}` : ""}${summaryPart}`;
}

/** The three ways a brief becomes a cut, in the order they cost wall-clock time. */
type Mode = "chat" | "jev" | "both";

const MODES: { id: Mode; label: string; action: string; hint: string }[] = [
  {
    id: "chat",
    label: "Model",
    action: "Send",
    hint: "The model researches the footage and writes the cut. Slowest, strongest at structure.",
  },
  {
    id: "jev",
    label: "Jev",
    action: "Build",
    hint: "Jev picks quotes and shots out of the index. ~15 s, and it can only use what was actually said.",
  },
  {
    id: "both",
    label: "Jev → model",
    action: "Build & refine",
    hint: "Jev chooses a first cut, the model improves it. Best result measured, and a few minutes.",
  },
];

export default function ScriptsPanel({ projectId }: { projectId: number }) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [scripts, setScripts] = useState<ScriptSummary[]>([]);
  const [selectedScriptId, setSelectedScriptId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [turnRunning, setTurnRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<ChatEvent[]>([]);
  const [optimisticUser, setOptimisticUser] = useState<string | null>(null);
  const [turnError, setTurnError] = useState<string | null>(null);
  /** Jev is building a cut by choosing. One shot, no conversation, ~15 s. */
  const [building, setBuilding] = useState(false);
  /**
   * How this brief becomes a cut.
   *
   * One decision, not two buttons and a checkbox. The three ways differ by which brain chooses
   * and which writes: a model researches the project and writes (slow, good structure); Jev only
   * chooses out of the index (~15 s, grounded, cannot write narration); or Jev chooses and the
   * model improves what it chose, which measured best — 76 editorial against 65 for Jev alone
   * and 76 for agy alone at three times the wall clock. The local models, worst at the research,
   * gain the most from not having to do it.
   */
  const [mode, setMode] = useState<Mode>("chat");
  /** Whether the Jev modes can run at all, so they are disabled with a reason, not failed with one. */
  const [jevReady, setJevReady] = useState<boolean | null>(null);
  /** Whether every finished cut is read editorially. `null` until settings have been read. */
  const [judging, setJudging] = useState<boolean | null>(null);
  const [latestIssues, setLatestIssues] = useState<Issue[] | undefined>(undefined);

  const messagesRef = useRef<HTMLDivElement>(null);
  const tasks = useQueue();

  const removeSession = async (id: number) => {
    if (!window.confirm("Remove this chat? Scripts drafted in it are kept.")) return;
    try {
      await deleteChatSession(id);
    } catch (e) {
      console.error(e);
      return;
    }
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (selectedSessionId === id) {
      setSelectedSessionId(null);
      setMessages([]);
    }
  };

  // Whether the Jev modes are offerable at all. Asked once: a mode that cannot run should say so
  // before it is chosen, not fail after.
  useEffect(() => {
    let cancelled = false;
    getAiSettings()
      .then((ai) => {
        if (cancelled) return;
        setJevReady(ai.jev.enabled && (ai.jev.has_key || ai.jev.key_from_env));
        setJudging(ai.jev.judge);
      })
      .catch(() => {
        // Unknown, not unavailable: leave the modes enabled rather than hiding them over a
        // settings read that happened to fail.
        if (!cancelled) setJevReady(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load sessions and scripts on mount / projectId change
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      chatSessions(projectId).catch(() => [] as ChatSession[]),
      listScripts(projectId).catch(() => [] as ScriptSummary[]),
    ]).then(([sessList, scriptList]) => {
      if (cancelled) return;
      // Newest sessions first
      const sortedSess = [...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at));
      setSessions(sortedSess);
      setScripts(scriptList);

      // Default select the latest session if available
      if (sortedSess.length > 0) {
        const firstSess = sortedSess[0];
        setSelectedSessionId(firstSess.id);
        chatMessages(firstSess.id).then((msgs) => {
          if (!cancelled) setMessages(msgs);
        }).catch(() => {});

        // Find latest script for this session
        const sessScripts = scriptList.filter((s) => s.session_id === firstSess.id);
        if (sessScripts.length > 0) {
          setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
        } else if (scriptList.length > 0) {
          setSelectedScriptId(Math.max(...scriptList.map((s) => s.id)));
        }
      } else if (scriptList.length > 0) {
        setSelectedScriptId(Math.max(...scriptList.map((s) => s.id)));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // When selected session changes, load messages
  useEffect(() => {
    if (selectedSessionId == null) {
      setMessages([]);
      return;
    }
    chatMessages(selectedSessionId)
      .then(setMessages)
      .catch(() => setMessages([]));
  }, [selectedSessionId]);

  // Listen to live chat-progress events
  useEffect(() => {
    const un = listen<ChatProgress>("chat-progress", (e) => {
      const p = e.payload;
      if (selectedSessionId == null || p.session_id === selectedSessionId) {
        setLiveEvents((prev) => [...prev, p.event]);
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, [selectedSessionId]);

  // Keep the chat pinned to its latest message. Scrolling the element itself rather than calling
  // scrollIntoView on a marker: that walks up every scrollable ancestor, so a new message dragged
  // the whole page down with it.
  useEffect(() => {
    const list = messagesRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, liveEvents, optimisticUser]);

  // Find running chat task in queue
  const runningChatTask = tasks.find(
    (t) =>
      t.state === "running" &&
      t.kind.type === "chat" &&
      t.kind.project_id === projectId &&
      (selectedSessionId == null || t.kind.session_id === selectedSessionId),
  );

  const handleSelectSession = (sessId: number) => {
    setSelectedSessionId(sessId);
    setTurnError(null);
    // Auto-select latest script belonging to this session if any
    const sessScripts = scripts.filter((s) => s.session_id === sessId);
    if (sessScripts.length > 0) {
      setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
    }
  };

  const handleNewChat = () => {
    setSelectedSessionId(null);
    setMessages([]);
    setTurnError(null);
    setLiveEvents([]);
  };

  const handleSend = async () => {
    const text = inputMessage.trim();
    if (!text || turnRunning) return;
    setInputMessage("");
    setTurnRunning(true);
    setTurnError(null);
    setLiveEvents([]);
    setOptimisticUser(text);

    try {
      // Jev first, when asked. The model's turn then happens in the session the build opened,
      // which already holds the brief, the cut and the editorial notes — so the instruction is to
      // improve it rather than to write one.
      let session = selectedSessionId;
      let message = text;
      if (mode !== "chat") {
        setBuilding(true);
        const built = await buildScriptWithJev(projectId, text, 40).finally(() => setBuilding(false));
        session = built.session_id;
        setSelectedSessionId(built.session_id);
        setSelectedScriptId(built.script_id);
        setScripts(await listScripts(projectId).catch(() => [] as ScriptSummary[]));
        setSessions(
          [...(await chatSessions(projectId).catch(() => [] as ChatSession[]))].sort(
            (a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at),
          ),
        );
        // Jev does not write, so it cannot hold a conversation: this mode ends here, with the
        // session left selected so anything typed next refines the cut rather than starting over.
        if (mode === "jev") {
          setMessages(await chatMessages(built.session_id).catch(() => [] as ChatMessage[]));
          return;
        }
        message =
          "Improve this cut. Keep it to the brief, fix the editorial notes above, and keep every " +
          "quote a whole sentence. Reply with the full script JSON.";
      }
      const res = await chatTurn(projectId, session, message);
      setSelectedSessionId(res.session_id);

      // Refresh sessions, messages, and scripts
      const [sessList, msgs, scriptList] = await Promise.all([
        chatSessions(projectId).catch(() => [] as ChatSession[]),
        chatMessages(res.session_id).catch(() => [] as ChatMessage[]),
        listScripts(projectId).catch(() => [] as ScriptSummary[]),
      ]);
      const sortedSess = [...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at));
      setSessions(sortedSess);
      setMessages(msgs);
      setScripts(scriptList);

      if (res.script_id != null) {
        setSelectedScriptId(res.script_id);
      }
      setLatestIssues(res.issues);
    } catch (err) {
      setTurnError(String(err));
    } finally {
      setTurnRunning(false);
      setOptimisticUser(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleScriptSaved = (newScriptId: number, issues: Issue[]) => {
    listScripts(projectId).then(setScripts).catch(() => {});
    setSelectedScriptId(newScriptId);
    setLatestIssues(issues);
  };

  const findScriptVersion = (sid: number | null) => {
    if (sid == null) return "?";
    const found = scripts.find((s) => s.id === sid);
    return found ? found.version : "?";
  };

  // Group scripts by session_id
  const sessionScripts =
    selectedSessionId != null
      ? scripts.filter((s) => s.session_id === selectedSessionId)
      : [];
  const otherScripts = scripts.filter(
    (s) => s.session_id == null || !sessions.some((sess) => sess.id === s.session_id),
  );

  return (
    <div className="scripts-layout">
      {/* 1. Left Column: Sessions & Script Versions */}
      <aside className="scripts-sidebar card">
        <button type="button" className="new-chat-btn" onClick={handleNewChat}>
          + New chat
        </button>

        <div className="sidebar-section-title">Chat Sessions</div>
        <div className="sessions-list">
          {sessions.length === 0 && (
            <div className="muted small">No sessions yet.</div>
          )}
          {sessions.map((sess) => {
            const isSelected = sess.id === selectedSessionId;
            return (
              <div key={sess.id} className="session-group">
                <div
                  className={`session-item ${isSelected ? "active" : ""}`}
                  onClick={() => handleSelectSession(sess.id)}
                >
                  <span className="session-title">{sess.title || "Untitled chat"}</span>
                  <span className="session-time muted small">
                    {formatRelativeTime(sess.updated_at || sess.created_at)}
                  </span>
                  <button
                    className="ghost small session-remove"
                    title="Remove this chat (its scripts are kept)"
                    onClick={(e) => {
                      e.stopPropagation();
                      void removeSession(sess.id);
                    }}
                  >
                    ×
                  </button>
                </div>

                {/* Script versions under selected session */}
                {isSelected && sessionScripts.length > 0 && (
                  <div className="session-scripts-sublist">
                    {sessionScripts.map((s) => {
                      const isScriptSelected = s.id === selectedScriptId;
                      return (
                        <div
                          key={s.id}
                          className={`script-version-item ${isScriptSelected ? "active" : ""}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedScriptId(s.id);
                          }}
                        >
                          <span className="version-tag">v{s.version}</span>
                          <span className="version-info muted small">
                            {s.beats} {s.beats === 1 ? "beat" : "beats"} · {s.clips} {s.clips === 1 ? "clip" : "clips"} · {s.duration_s.toFixed(1)} s
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Other scripts without matching session */}
        {otherScripts.length > 0 && (
          <div className="other-scripts-section">
            <div className="sidebar-section-title">Other Scripts</div>
            <div className="session-scripts-sublist standalone">
              {otherScripts.map((s) => {
                const isScriptSelected = s.id === selectedScriptId;
                return (
                  <div
                    key={s.id}
                    className={`script-version-item ${isScriptSelected ? "active" : ""}`}
                    onClick={() => setSelectedScriptId(s.id)}
                  >
                    <span className="version-tag">v{s.version}</span>
                    <span className="version-info muted small">
                      {s.beats} {s.beats === 1 ? "beat" : "beats"} · {s.clips} {s.clips === 1 ? "clip" : "clips"} · {s.duration_s.toFixed(1)} s
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </aside>

      {/* 2. Center Column: Chat */}
      <section className="chat-panel card">
        <div className="chat-header">
          <span className="label">
            {selectedSessionId != null
              ? sessions.find((s) => s.id === selectedSessionId)?.title || "Script Chat"
              : "New Script Chat"}
          </span>
          {turnRunning && <span className="pill local small">Running</span>}
        </div>

        <div className="chat-messages" ref={messagesRef}>
          {messages.length === 0 && !optimisticUser && (
            <div className="chat-empty muted small">
              Describe your video concept to generate a script with clips from this project’s footage.
              <div className="chat-prompt-hint">
                e.g. “Write a 60-second teaser showcasing the unboxing and initial test drive.”
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`chat-message-row ${msg.role}`}>
              {/* Tool calls rendered as chips */}
              {msg.tool_calls && msg.tool_calls.length > 0 && (
                <div className="tool-chips-row">
                  {msg.tool_calls.map((tc, idx) => (
                    <span
                      key={idx}
                      className="tag tool-chip"
                      title={typeof tc.args === "string" ? tc.args : JSON.stringify(tc.args, null, 2)}
                    >
                      ⚙ {formatToolCall(tc)}
                    </span>
                  ))}
                </div>
              )}

              {msg.content && (
                <div className={`chat-bubble ${msg.role}`}>
                  <div className="bubble-text">{msg.content}</div>
                  {msg.script_id != null && (
                    <div className="bubble-script-link">
                      <button
                        type="button"
                        className="ghost small link-button"
                        onClick={() => setSelectedScriptId(msg.script_id!)}
                      >
                        Open v{findScriptVersion(msg.script_id)}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Optimistic user message while turn runs */}
          {optimisticUser && (
            <div className="chat-message-row user">
              <div className="chat-bubble user">
                <div className="bubble-text">{optimisticUser}</div>
              </div>
            </div>
          )}

          {/* Live progress during turn */}
          {turnRunning && (
            <div className="chat-message-row assistant live">
              <div className="live-progress-container">
                {liveEvents.map((evt, idx) => (
                  <span
                    key={idx}
                    className="tag tool-chip live-chip"
                    title={
                      evt.kind === "tool_started" && evt.args
                        ? typeof evt.args === "string"
                          ? evt.args
                          : JSON.stringify(evt.args, null, 2)
                        : undefined
                    }
                  >
                    {evt.kind === "tool_started" && `⚙ ${evt.tool}...`}
                    {evt.kind === "tool_finished" && `✓ ${evt.tool} → ${evt.summary}`}
                    {evt.kind === "drafting" && `✎ Drafting script...`}
                    {evt.kind === "validating" && `🔍 Validating script...`}
                  </span>
                ))}
                {runningChatTask && <TaskCard task={runningChatTask} compact />}
              </div>
            </div>
          )}

          {/* Show issues on finish if present */}
          {latestIssues && latestIssues.length > 0 && !turnRunning && (
            <div className="issues-box small">
              <div className="label">Script Issues ({latestIssues.length})</div>
              {latestIssues.map((iss, i) => (
                <div key={i} className={`issue-row ${iss.severity === "error" ? "bad-text" : "warn"}`}>
                  <span className={`pill ${iss.severity}`}>{iss.severity}</span>
                  {iss.beat_id && <strong>[Beat {iss.beat_id}]</strong>}
                  {iss.clip_index != null && <span>Clip #{iss.clip_index + 1}:</span>}
                  <span>{iss.message}</span>
                </div>
              ))}
            </div>
          )}

          {turnError && <div className="banner bad small">{turnError}</div>}
        </div>

        {/* Composer: what to make, then how to make it. */}
        <div className="chat-composer">
          <textarea
            rows={3}
            placeholder={
              turnRunning ? "Working…" : "Describe the cut you want. Enter to send, Shift+Enter for a new line."
            }
            value={inputMessage}
            disabled={turnRunning}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          {/*
            One decision rather than two buttons and a checkbox: every one of these turns the
            same brief into a cut, and they differ only in which brain chooses and which writes.
          */}
          <div className="composer-modes" role="radiogroup" aria-label="How to build this cut">
            {MODES.map((m) => {
              const needsJev = m.id !== "chat";
              const blocked = needsJev && jevReady === false;
              return (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={mode === m.id}
                  className={`composer-mode${mode === m.id ? " on" : ""}`}
                  disabled={turnRunning || blocked}
                  title={blocked ? "Needs Jev turned on with an API key — see Settings" : m.hint}
                  onClick={() => setMode(m.id)}
                >
                  {m.label}
                </button>
              );
            })}
          </div>

          {/*
            The judge is a separate want from the builder: somebody may have Jev assemble a cut
            and not want every draft scored. `jev.enabled` gates both, so this writes its own flag.
          */}
          <button
            type="button"
            className={`composer-judge${judging ? " on" : ""}`}
            disabled={jevReady === false || judging === null}
            aria-pressed={!!judging}
            title={
              jevReady === false
                ? "Needs Jev turned on with an API key — see Settings"
                : judging
                  ? "Every finished cut is read editorially — the shots against the voice, the opening, the ending. Click to stop."
                  : "Finished cuts are not read editorially. Click to have Jev score them."
            }
            onClick={async () => {
              const next = !judging;
              setJudging(next);
              try {
                const ai = await setAiSettings({ jev: { judge: next } });
                setJudging(ai.jev.judge);
              } catch (e) {
                setJudging(!next);
                setTurnError(String(e));
              }
            }}
          >
            {judging ? "Judging on" : "Judging off"}
          </button>

          <div className="composer-go">
            <p className="composer-hint small muted">
              {jevReady === false && mode !== "chat"
                ? "Needs Jev turned on with an API key — see Settings."
                : MODES.find((m) => m.id === mode)?.hint}
            </p>
            <button
              type="button"
              className="primary"
              disabled={turnRunning || building || !inputMessage.trim()}
              onClick={handleSend}
            >
              {building ? "Choosing…" : turnRunning ? "Working…" : MODES.find((m) => m.id === mode)?.action}
            </button>
          </div>
        </div>
      </section>

      {/* 3. Right Column: Script Editor for selected script */}
      <div className="script-editor-col">
        {selectedScriptId != null ? (
          <ScriptEditor
            projectId={projectId}
            scriptId={selectedScriptId}
            sessionId={selectedSessionId}
            latestIssues={latestIssues}
            onScriptSaved={handleScriptSaved}
          />
        ) : (
          <div className="card muted empty-editor-card">
            <div className="empty-editor-content">
              <h3>No script selected</h3>
              <p className="small">
                Select a script version from the left panel or ask the chat assistant in the center to generate one.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
