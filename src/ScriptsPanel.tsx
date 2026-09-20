import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
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
      const res = await chatTurn(projectId, selectedSessionId, text);
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

        {/* Chat Input */}
        <div className="chat-input-row">
          <textarea
            rows={2}
            placeholder={
              turnRunning
                ? "Waiting for response..."
                : "Ask GhostReel to draft or revise a script (Enter to send, Shift+Enter for newline)..."
            }
            value={inputMessage}
            disabled={turnRunning}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            type="button"
            disabled={turnRunning || !inputMessage.trim()}
            onClick={handleSend}
          >
            Send
          </button>
          {/*
            Jev does not write, so it cannot hold a conversation — this is one shot, not a turn.
            It chooses the quotes and the shots out of the index and code assembles them, which
            takes about as long as a preview and cannot refer to footage that does not exist.
          */}
          <button
            type="button"
            className="ghost"
            title="Build a cut by choosing rather than writing: Jev picks the quotes and the shots out of the index. Fast, grounded, and limited to what the interviews already say. Needs a Jev key in Settings."
            disabled={turnRunning || building || !inputMessage.trim()}
            onClick={async () => {
              const brief = inputMessage.trim();
              setBuilding(true);
              setTurnError(null);
              try {
                const id = await buildScriptWithJev(projectId, brief, 40);
                setInputMessage("");
                setScripts(await listScripts(projectId));
                setSelectedScriptId(id);
              } catch (e) {
                setTurnError(String(e));
              } finally {
                setBuilding(false);
              }
            }}
          >
            {building ? "Choosing…" : "Build with Jev"}
          </button>
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
