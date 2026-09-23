import { useEffect, useRef, useState } from "react";
import {
  fileUrl,
  getAiSettings,
  setAiSettings,
  cliModels,
  modelsStatus,
  chatMessages,
  chatSessions,
  deleteChatSession,
  chatTurn,
  buildScriptWithJev,
  listScripts,
  type AiSettings,
  type ChatEvent,
  type ChatMessage,
  type ChatProgress,
  type ChatSession,
  type Issue,
  type ModelStatus,
  type ScriptSummary,
  type ToolCallRecord,
  type VisionSettingsPatch,
} from "./api";
import { onEvent } from "./events";
import ScriptEditor from "./ScriptEditor";
import TaskCard from "./TaskCard";
import { useQueue } from "./useQueue";

function cleanTitle(title: string | null | undefined): string {
  if (!title) return "Untitled chat";
  const firstLine = title.split("\n")[0].trim();
  const cleaned = firstLine.replace(/^```[a-z]*\s*/i, "").replace(/[`"{}[\]]/g, "").trim();
  return cleaned.length > 55 ? cleaned.slice(0, 55) + "…" : cleaned || "Untitled chat";
}

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

/** Survives component unmount/remount (e.g. navigating to Activity/Settings/Library and back). */
const sessionEventsCache = new Map<number, ChatEvent[]>();

export default function ScriptsPanel({ projectId }: { projectId: number }) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [scripts, setScripts] = useState<ScriptSummary[]>([]);
  const [selectedScriptId, setSelectedScriptId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [turnRunning, setTurnRunning] = useState(false);
  const [liveEvents, setLiveEvents] = useState<ChatEvent[]>([]);
  const [optimisticUser, setOptimisticUser] = useState<{ text: string; images: string[] } | null>(null);
  const [turnError, setTurnError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resolveImgSrc = (img: string) => (img.startsWith("data:") ? img : fileUrl(img));

  const addImages = (files: FileList | File[]) => {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (loadEvent) => {
          const res = loadEvent.target?.result;
          if (typeof res === "string") {
            setAttachedImages((prev) => [...prev, res]);
          }
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const imageFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          imageFiles.push(file);
        }
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      addImages(imageFiles);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      let hasImg = false;
      for (let i = 0; i < files.length; i++) {
        if (files[i].type.startsWith("image/")) {
          hasImg = true;
          break;
        }
      }
      if (hasImg) {
        e.preventDefault();
        addImages(files);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addImages(e.target.files);
      e.target.value = "";
    }
  };
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
  const userSelectedRef = useRef<boolean>(false);

  // A turn's session exists from the moment it is queued — the backend creates it before the
  // task starts — but the list was only re-read when the turn returned, so a chat that took ten
  // minutes was invisible for ten minutes: "No sessions yet" beside a card reading "Chat #1".
  // Follow the running task instead: show its session as soon as it has one, and select it if
  // nothing else is selected, so the conversation is traceable while it happens.
  const liveSessionId = (() => {
    for (const t of tasks) {
      if ((t.state === "running" || t.state === "queued") && t.kind.type === "chat" && t.kind.project_id === projectId) {
        return t.kind.session_id;
      }
    }
    return null;
  })();

  // Find active chat task in queue (running or queued)
  const activeChatTask = tasks.find(
    (t) =>
      (t.state === "running" || t.state === "queued") &&
      t.kind.type === "chat" &&
      t.kind.project_id === projectId &&
      (selectedSessionId == null || t.kind.session_id === (selectedSessionId ?? liveSessionId)),
  );

  const isTurnRunning = turnRunning || Boolean(activeChatTask);

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
      setSelectedScriptId(null);
      setMessages([]);
    }
  };

  const [aiSettings, setAiSettingsState] = useState<AiSettings | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelStatus[]>([]);
  const [cliModelList, setCliModelList] = useState<string[]>([]);

  // Whether the Jev modes are offerable at all. Asked once: a mode that cannot run should say so
  // before it is chosen, not fail after.
  useEffect(() => {
    let cancelled = false;
    getAiSettings()
      .then((ai) => {
        if (cancelled) return;
        setAiSettingsState(ai);
        setJevReady(ai.jev.enabled && (ai.jev.has_key || ai.jev.key_from_env));
        setJudging(ai.jev.judge);
        if (ai.chat_model.cli?.tool) {
          cliModels(ai.chat_model.cli.tool)
            .then((m) => {
              if (!cancelled) setCliModelList(m);
            })
            .catch(() => {});
        }
      })
      .catch(() => {
        // Unknown, not unavailable: leave the modes enabled rather than hiding them over a
        // settings read that happened to fail.
        if (!cancelled) setJevReady(null);
      });

    modelsStatus()
      .then((st) => {
        if (!cancelled) {
          setAvailableModels(st.models.filter((m) => m.entry.kind === "vision"));
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, []);

  const handleUpdateChatModel = async (patch: VisionSettingsPatch) => {
    try {
      const nextAi = await setAiSettings({ chat_model: patch });
      setAiSettingsState(nextAi);
      if (patch.cli?.tool) {
        cliModels(patch.cli.tool)
          .then(setCliModelList)
          .catch(() => setCliModelList([]));
      }
    } catch (err) {
      setTurnError(String(err));
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

      // Default select the live running session if any, otherwise the latest session
      const targetSessionId = liveSessionId ?? (sortedSess.length > 0 ? sortedSess[0].id : null);
      if (targetSessionId != null) {
        setSelectedSessionId(targetSessionId);
        chatMessages(targetSessionId).then((msgs) => {
          if (!cancelled) setMessages(msgs);
        }).catch(() => {});

        // Find latest script for this session
        const sessScripts = scriptList.filter((s) => s.session_id === targetSessionId);
        setSelectedScriptId(sessScripts.length > 0 ? Math.max(...sessScripts.map((s) => s.id)) : null);
      } else {
        setSelectedSessionId(null);
        setSelectedScriptId(null);
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
    return onEvent<ChatProgress>("chat-progress", (p) => {
      const cached = sessionEventsCache.get(p.session_id) ?? [];
      const updated = [...cached, p.event];
      sessionEventsCache.set(p.session_id, updated);
      if (selectedSessionId == null || p.session_id === selectedSessionId || p.session_id === liveSessionId) {
        setLiveEvents(updated);
      }
    });
  }, [selectedSessionId, liveSessionId]);

  // Restore live events on mount or session change
  useEffect(() => {
    const sid = selectedSessionId ?? liveSessionId;
    if (!sid) return;
    const fromCache = sessionEventsCache.get(sid);
    const fromTask = activeChatTask?.chat_events;
    if (fromCache && fromCache.length > 0) {
      setLiveEvents(fromCache);
    } else if (fromTask && fromTask.length > 0) {
      sessionEventsCache.set(sid, fromTask);
      setLiveEvents(fromTask);
    }
  }, [selectedSessionId, liveSessionId, activeChatTask?.id]);

  // Sync if task has more events from backend
  useEffect(() => {
    const sid = selectedSessionId ?? liveSessionId;
    if (activeChatTask?.chat_events && activeChatTask.chat_events.length > liveEvents.length) {
      if (sid) {
        sessionEventsCache.set(sid, activeChatTask.chat_events);
      }
      setLiveEvents(activeChatTask.chat_events);
    }
  }, [activeChatTask?.chat_events, liveEvents.length, selectedSessionId, liveSessionId]);

  const displayEvents = liveEvents.length > 0 ? liveEvents : (activeChatTask?.chat_events ?? []);

  // Keep the chat pinned to its latest message. Scrolling the element itself rather than calling
  // scrollIntoView on a marker: that walks up every scrollable ancestor, so a new message dragged
  // the whole page down with it.
  useEffect(() => {
    const list = messagesRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, displayEvents, optimisticUser, isTurnRunning]);

  // Detect when a running chat task finishes while viewing it
  const prevActiveRef = useRef<boolean>(false);
  useEffect(() => {
    const wasActive = prevActiveRef.current;
    const nowActive = Boolean(activeChatTask);
    prevActiveRef.current = nowActive;
    if (wasActive && !nowActive) {
      const sid = selectedSessionId ?? liveSessionId;
      if (sid != null) {
        sessionEventsCache.delete(sid);
        chatMessages(sid).then(setMessages).catch(() => {});
      }
      setLiveEvents([]);
      setOptimisticUser(null);
      setTurnRunning(false);
      chatSessions(projectId).then((list) => {
        setSessions([...list].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
      }).catch(() => {});
      listScripts(projectId).then((list) => {
        setScripts(list);
        if (sid != null) {
          const sessScripts = list.filter((s) => s.session_id === sid);
          if (sessScripts.length > 0) {
            setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
          }
        }
      }).catch(() => {});
    }
  }, [activeChatTask, selectedSessionId, liveSessionId, projectId]);

  // Also listen for task-finished event as a fallback
  useEffect(() => {
    return onEvent<number>("task-finished", async () => {
      const [sessList, scriptList] = await Promise.all([
        chatSessions(projectId).catch(() => [] as ChatSession[]),
        listScripts(projectId).catch(() => [] as ScriptSummary[]),
      ]);
      setSessions([...sessList].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
      setScripts(scriptList);
      const sid = selectedSessionId ?? liveSessionId;
      if (sid != null) {
        sessionEventsCache.delete(sid);
        const msgs = await chatMessages(sid).catch(() => [] as ChatMessage[]);
        setMessages(msgs);
        const sessScripts = scriptList.filter((s) => s.session_id === sid);
        if (sessScripts.length > 0) {
          setSelectedScriptId(Math.max(...sessScripts.map((s) => s.id)));
        }
      }
      setLiveEvents([]);
      setOptimisticUser(null);
      setTurnRunning(false);
    });
  }, [projectId, selectedSessionId, liveSessionId]);

  useEffect(() => {
    if (liveSessionId == null) return;
    let cancelled = false;
    if (!sessions.some((s) => s.id === liveSessionId)) {
      chatSessions(projectId)
        .then((list) => {
          if (cancelled) return;
          setSessions([...list].sort((a, b) => (b.updated_at || b.created_at) - (a.updated_at || a.created_at)));
        })
        .catch(() => {});
    }
    if (!userSelectedRef.current && selectedSessionId !== liveSessionId) {
      setSelectedSessionId(liveSessionId);
    }
    return () => {
      cancelled = true;
    };
  }, [liveSessionId, projectId, selectedSessionId]);

  const handleSelectSession = (sessId: number) => {
    userSelectedRef.current = true;
    setSelectedSessionId(sessId);
    setTurnError(null);
    setLiveEvents(sessionEventsCache.get(sessId) ?? []);
    // Auto-select latest script belonging to this session if any, otherwise null
    const sessScripts = scripts.filter((s) => s.session_id === sessId);
    setSelectedScriptId(sessScripts.length > 0 ? Math.max(...sessScripts.map((s) => s.id)) : null);
  };

  const handleNewChat = () => {
    userSelectedRef.current = true;
    setSelectedSessionId(null);
    setSelectedScriptId(null);
    setMessages([]);
    setTurnError(null);
    setLiveEvents([]);
  };

  const handleSend = async () => {
    const text = inputMessage.trim();
    const imagesToSend = [...attachedImages];
    if ((!text && imagesToSend.length === 0) || isTurnRunning) return;
    userSelectedRef.current = true;
    const promptText = text || (imagesToSend.length > 0 ? "Review the attached image(s)." : "");
    setInputMessage("");
    setAttachedImages([]);
    setTurnRunning(true);
    setTurnError(null);
    setLiveEvents([]);
    setOptimisticUser({ text: promptText, images: imagesToSend });

    try {
      // Jev first, when asked. The build lands in the open chat (or opens one), so the model's
      // turn that follows already holds the brief, the cut and the editorial notes — and a build
      // asked for mid-conversation continues that conversation rather than starting another.
      let session = selectedSessionId;
      let message = promptText;
      if (mode !== "chat") {
        setBuilding(true);
        const built = await buildScriptWithJev(projectId, session, promptText, 40).finally(() =>
          setBuilding(false),
        );
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
      const res = await chatTurn(projectId, session, message, imagesToSend);
      setSelectedSessionId(res.session_id);
      userSelectedRef.current = true;
      sessionEventsCache.delete(res.session_id);

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
      if (!isTurnRunning && !building) {
        handleSend();
      }
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
          {sessions.length === 0 && liveSessionId == null && (
            <div className="muted small">No sessions yet.</div>
          )}
          {liveSessionId != null && !sessions.some((s) => s.id === liveSessionId) && (
            <div
              className={`session-item ${selectedSessionId === liveSessionId ? "active" : ""}`}
              onClick={() => handleSelectSession(liveSessionId)}
            >
              <span className="session-title">Current chat</span>
              <span className="pill local small">Running</span>
            </div>
          )}
          {sessions.map((sess) => {
            const isSelected = sess.id === selectedSessionId;
            return (
              <div key={sess.id} className="session-group">
                <div
                  className={`session-item ${isSelected ? "active" : ""}`}
                  onClick={() => handleSelectSession(sess.id)}
                >
                  <span className="session-title">{cleanTitle(sess.title)}</span>
                  {liveSessionId === sess.id && <span className="pill local small">Running</span>}
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
                            userSelectedRef.current = true;
                            setSelectedSessionId(sess.id);
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
                    onClick={() => {
                      userSelectedRef.current = true;
                      setSelectedScriptId(s.id);
                      if (s.session_id != null && sessions.some((sess) => sess.id === s.session_id)) {
                        setSelectedSessionId(s.session_id);
                        chatMessages(s.session_id).then(setMessages).catch(() => setMessages([]));
                      } else {
                        setSelectedSessionId(null);
                        setMessages([]);
                      }
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
          </div>
        )}
      </aside>

      {/* 2. Center Column: Chat */}
      <section className="chat-panel card">
        <div className="chat-header">
          <span className="label">
            {selectedSessionId != null
              ? cleanTitle(sessions.find((s) => s.id === selectedSessionId)?.title)
              : "New Script Chat"}
          </span>
          {isTurnRunning && <span className="pill local small">Running</span>}
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

              {(msg.content || (msg.images && msg.images.length > 0)) && (
                <div className={`chat-bubble ${msg.role}`}>
                  {msg.images && msg.images.length > 0 && (
                    <div className="chat-attached-images">
                      {msg.images.map((img, idx) => (
                        <img
                          key={idx}
                          src={resolveImgSrc(img)}
                          alt={`Attached image ${idx + 1}`}
                          className="chat-attached-image"
                          onClick={() => setPreviewImage(resolveImgSrc(img))}
                          title="Click to view full image"
                        />
                      ))}
                    </div>
                  )}
                  {msg.content && <div className="bubble-text">{msg.content}</div>}
                  {msg.script_id != null && (
                    <div className="bubble-script-link">
                      <button
                        type="button"
                        className={`ghost small link-button${selectedScriptId === msg.script_id ? " active-script" : ""}`}
                        onClick={() => setSelectedScriptId(msg.script_id!)}
                      >
                        {selectedScriptId === msg.script_id
                          ? `✓ Viewing v${findScriptVersion(msg.script_id)}`
                          : `Open v${findScriptVersion(msg.script_id)}`}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Optimistic user message while turn runs (only if not already present in loaded messages) */}
          {optimisticUser && !messages.some((m) => m.role === "user" && m.content === optimisticUser.text) && (
            <div className="chat-message-row user">
              <div className="chat-bubble user">
                {optimisticUser.images.length > 0 && (
                  <div className="chat-attached-images">
                    {optimisticUser.images.map((img, idx) => (
                      <img
                        key={idx}
                        src={resolveImgSrc(img)}
                        alt={`Attached image ${idx + 1}`}
                        className="chat-attached-image"
                        onClick={() => setPreviewImage(resolveImgSrc(img))}
                        title="Click to view full image"
                      />
                    ))}
                  </div>
                )}
                {optimisticUser.text && <div className="bubble-text">{optimisticUser.text}</div>}
              </div>
            </div>
          )}

          {/* Live progress during turn */}
          {isTurnRunning && (
            <div className="chat-message-row assistant live">
              <div className="live-progress-container">
                {displayEvents.map((evt, idx) => (
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
                {activeChatTask && <TaskCard task={activeChatTask} compact />}
              </div>
            </div>
          )}

          {/* Show issues on finish if present */}
          {latestIssues && latestIssues.length > 0 && !isTurnRunning && (
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
        <div className="chat-composer" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
          {/* Attached images preview strip */}
          {attachedImages.length > 0 && (
            <div className="chat-composer-attachments">
              {attachedImages.map((img, idx) => (
                <div key={idx} className="chat-composer-attachment-thumb">
                  <img
                    src={img}
                    alt={`Attachment ${idx + 1}`}
                    onClick={() => setPreviewImage(img)}
                    title="Click to view full image"
                  />
                  <button
                    type="button"
                    className="chat-composer-attachment-remove"
                    onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                    title="Remove image"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <textarea
            rows={3}
            placeholder={
              isTurnRunning
                ? "Working…"
                : "Describe the cut you want, or paste screenshots. Enter to send, Shift+Enter for a new line."
            }
            value={inputMessage}
            disabled={isTurnRunning}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
          />

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            style={{ display: "none" }}
            onChange={handleFileSelect}
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
                  className={`composer-mode${mode === m.id ? " on" : ""}${blocked ? " blocked" : ""}`}
                  disabled={isTurnRunning}
                  title={blocked ? "Needs Jev turned on with an API key — see Settings" : m.hint}
                  onClick={() => setMode(m.id)}
                >
                  {m.label}
                </button>
              );
            })}
          </div>

          {/* Model / Brain selector for script chat */}
          {aiSettings && (
            <div className="composer-model-bar">
              <span className="composer-model-label">Brain:</span>
              <div className="composer-backend-pills" role="radiogroup" aria-label="Script chat brain">
                {(["auto", "local", "server", "cli"] as const).map((b) => (
                  <button
                    key={b}
                    type="button"
                    className={`backend-pill${aiSettings.chat_model.backend === b ? " active" : ""}`}
                    disabled={isTurnRunning}
                    onClick={() => handleUpdateChatModel({ backend: b })}
                  >
                    {b === "auto" ? "Auto" : b === "local" ? "Local" : b === "server" ? "Server" : "CLI"}
                  </button>
                ))}
              </div>

              <div className="composer-model-controls">
                {aiSettings.chat_model.backend === "local" && (
                  <>
                    <select
                      className="composer-model-select"
                      value={aiSettings.chat_model.local_model}
                      disabled={isTurnRunning}
                      onChange={(e) => handleUpdateChatModel({ local_model: e.target.value })}
                      title="Local model"
                    >
                      {!availableModels.some((m) => m.entry.id === aiSettings.chat_model.local_model) && (
                        <option value={aiSettings.chat_model.local_model}>
                          {aiSettings.chat_model.local_model}
                        </option>
                      )}
                      {availableModels.map((m) => (
                        <option key={m.entry.id} value={m.entry.id}>
                          {m.entry.id} {m.installed_path ? "" : "(not downloaded)"}
                        </option>
                      ))}
                    </select>

                    <label className="composer-think-label" title="Reasoning / think mode">
                      <input
                        type="checkbox"
                        checked={aiSettings.chat_model.think}
                        disabled={isTurnRunning}
                        onChange={(e) => handleUpdateChatModel({ think: e.target.checked })}
                      />
                      Think
                    </label>
                  </>
                )}

                {aiSettings.chat_model.backend === "cli" && (
                  <>
                    <select
                      className="composer-model-select"
                      value={aiSettings.chat_model.cli.tool}
                      disabled={isTurnRunning}
                      onChange={(e) => handleUpdateChatModel({ cli: { tool: e.target.value } })}
                      title="CLI tool"
                    >
                      <option value="claude">claude</option>
                      <option value="agy">agy</option>
                      <option value="opencode">opencode</option>
                      <option value="codex">codex</option>
                    </select>

                    {cliModelList.length > 0 ? (
                      <select
                        className="composer-model-select"
                        value={aiSettings.chat_model.cli.model}
                        disabled={isTurnRunning}
                        onChange={(e) => handleUpdateChatModel({ cli: { model: e.target.value } })}
                        title="CLI tool model"
                      >
                        <option value="">— default —</option>
                        {!cliModelList.includes(aiSettings.chat_model.cli.model) && aiSettings.chat_model.cli.model && (
                          <option value={aiSettings.chat_model.cli.model}>{aiSettings.chat_model.cli.model}</option>
                        )}
                        {cliModelList.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        className="composer-model-input"
                        defaultValue={aiSettings.chat_model.cli.model}
                        placeholder="— default —"
                        disabled={isTurnRunning}
                        title="CLI model (optional)"
                        onBlur={(e) => handleUpdateChatModel({ cli: { model: e.target.value } })}
                      />
                    )}

                    <span
                      className={aiSettings.chat_model.cli.installed ? "good-text small" : "bad-text small"}
                      title={aiSettings.chat_model.cli.installed ? "Found in PATH" : "Not found in PATH"}
                      style={{ fontSize: "11px" }}
                    >
                      {aiSettings.chat_model.cli.installed ? "✓" : "⚠️ not found"}
                    </span>
                  </>
                )}

                {aiSettings.chat_model.backend === "server" && (
                  <>
                    <input
                      type="text"
                      className="composer-model-input"
                      defaultValue={aiSettings.chat_model.url}
                      placeholder="http://127.0.0.1:8089"
                      disabled={isTurnRunning}
                      title="Server URL"
                      onBlur={(e) => handleUpdateChatModel({ url: e.target.value })}
                    />
                    <input
                      type="text"
                      className="composer-model-input"
                      defaultValue={aiSettings.chat_model.model}
                      placeholder="— server default —"
                      disabled={isTurnRunning}
                      title="Server model name"
                      onBlur={(e) => handleUpdateChatModel({ model: e.target.value })}
                    />
                  </>
                )}

                {aiSettings.chat_model.backend === "auto" && (
                  <span className="muted small" style={{ fontSize: "11px" }}>
                    Auto-selects best available
                  </span>
                )}
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            {/*
              The judge is a separate want from the builder: somebody may have Jev assemble a cut
              and not want every draft scored. `jev.enabled` gates both, so this writes its own flag.
            */}
            <button
              type="button"
              className={`composer-judge${jevReady && judging ? " on" : ""}`}
              disabled={jevReady === false || judging === null}
              aria-pressed={Boolean(jevReady && judging)}
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
              {jevReady && judging ? "Judging on" : "Judging off"}
            </button>

            <button
              type="button"
              className="composer-attach-btn"
              disabled={isTurnRunning}
              onClick={() => fileInputRef.current?.click()}
              title="Attach image or screenshot (or paste directly from clipboard)"
            >
              📎 Attach image
            </button>
          </div>

          <div className="composer-go">
            <p className="composer-hint small muted">
              {jevReady === false && mode !== "chat"
                ? "Needs Jev turned on with an API key — see Settings."
                : MODES.find((m) => m.id === mode)?.hint}
            </p>
            <button
              type="button"
              className="primary"
              disabled={
                isTurnRunning ||
                building ||
                (!inputMessage.trim() && attachedImages.length === 0) ||
                (mode !== "chat" && jevReady === false)
              }
              title={
                mode !== "chat" && jevReady === false
                  ? "Needs Jev turned on with an API key — see Settings"
                  : undefined
              }
              onClick={handleSend}
            >
              {building
                ? "Choosing…"
                : isTurnRunning
                  ? "Working…"
                  : MODES.find((m) => m.id === mode)?.action}
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
            sessionTitle={
              selectedSessionId != null
                ? cleanTitle(sessions.find((s) => s.id === selectedSessionId)?.title)
                : undefined
            }
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

      {/* Lightbox modal for previewing attached images */}
      {previewImage && (
        <div className="chat-image-modal-overlay" onClick={() => setPreviewImage(null)}>
          <div className="chat-image-modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="chat-image-modal-close"
              onClick={() => setPreviewImage(null)}
              title="Close"
            >
              ×
            </button>
            <img src={previewImage} alt="Expanded preview" />
          </div>
        </div>
      )}
    </div>
  );
}
