import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import {
  getAiSettings,
  getChatSettings,
  isDesktop,
  setAiSettings,
  setChatSystemPrompt,
  setWebConfig,
  webStatus,
  type AiSettings,
  type AiSettingsPatch,
  type ChatSettings,
  type WebConfigPatch,
  type WebStatus,
} from "./api";

/** Bound here, nothing else on the network can reach it — which is what makes a password optional. */
const isLoopback = (bind: string) => bind === "127.0.0.1" || bind === "localhost" || bind === "::1";

// ─── per-section Saved cue ───────────────────────────────────────────────────

type CueState = "idle" | "saving" | "saved";

/** A transient "Saved" cue for one settings card: Saving… → Saved → fades out. */
function useSavedCue() {
  const [state, setState] = useState<CueState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bump = (to: CueState) => {
    setState(to);
    if (timer.current) clearTimeout(timer.current);
    if (to === "saved") timer.current = setTimeout(() => setState("idle"), 2200);
  };
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  return {
    state,
    saving: () => bump("saving"),
    saved: () => bump("saved"),
    reset: () => {
      if (timer.current) clearTimeout(timer.current);
      setState("idle");
    },
  };
}

function SavedCue({ state }: { state: CueState }) {
  if (state === "idle") return null;
  return (
    <span className={`saved-cue${state === "saved" ? " fade" : ""}`} role="status">
      {state === "saving" ? (
        "Saving…"
      ) : (
        <>
          <Check size={13} strokeWidth={2} aria-hidden /> Saved
        </>
      )}
    </span>
  );
}

/** One settings write: run the card's cue; failures surface through the page banner via onPatch. */
function useApplier(onPatch: (p: AiSettingsPatch) => Promise<boolean>) {
  const cue = useSavedCue();
  const apply = async (patch: AiSettingsPatch) => {
    cue.saving();
    if (await onPatch(patch)) cue.saved();
    else cue.reset();
  };
  return { cue, apply };
}

// ─── Jev ─────────────────────────────────────────────────────────────────────

function JevCard({
  ai,
  onPatch,
  onError,
}: {
  ai: AiSettings | null;
  onPatch: (p: AiSettingsPatch) => Promise<boolean>;
  onError: (e: string | null) => void;
}) {
  const { cue, apply } = useApplier(onPatch);
  const enabled = ai?.jev.enabled ?? false;
  const keyRef = useRef<HTMLInputElement>(null);
  return (
    <section className="card">
      <div className="settings-card-title">
        Quote builder (Jev) — cloud builds &amp; reviews
        <SavedCue state={cue.state} />
      </div>
      <p className="muted small">
        Jev is TypeSafe&apos;s hosted model: it can assemble a first cut from indexed footage in about 15 s, flag
        off-mic interviewers, and score finished drafts. It is the one part of GhostReel that leaves this computer —
        when it is on, transcripts and frame descriptions go to <code>api.typesafe.ai</code>. Nothing is sent while
        it is off. Sonar&apos;s build cards call this the quote-builder key.
      </p>
      <div className="settings-fields">
        <div className="settings-field check">
          <label htmlFor="jev-enabled">Enable Jev</label>
          <input
            id="jev-enabled"
            type="checkbox"
            checked={enabled}
            disabled={!ai}
            onChange={async (e) => {
              const on = e.currentTarget.checked;
              if (on && !ai?.jev.has_key && !ai?.jev.key_from_env) {
                onError("Turning Jev on needs an API key: enter one below first");
                return;
              }
              await apply({ jev: { enabled: on } });
            }}
          />
          <span className="muted small">
            builds cuts, tells interviewer questions apart from answers and judges finished drafts in the cloud
          </span>
        </div>
        <div className="settings-field check">
          <label htmlFor="jev-judge">Let Jev judge finished cuts</label>
          <input
            id="jev-judge"
            type="checkbox"
            checked={ai?.jev.judge ?? true}
            disabled={!ai || !enabled}
            onChange={async (e) => {
              await apply({ jev: { judge: e.currentTarget.checked } });
            }}
          />
          <span className="muted small">
            scores drafted cuts in the cloud and flags weak moments
            {!enabled && " · needs Jev above"}
          </span>
        </div>
        <div className="settings-field">
          <label htmlFor="jev-key">API key</label>
          {ai?.jev.key_from_env ? (
            <span className="muted small">
              taken from <code>TYPESAFE_API_KEY</code> in the environment, which wins over anything typed here
            </span>
          ) : (
            <>
              <input
                id="jev-key"
                ref={keyRef}
                type="password"
                style={{ width: "22em" }}
                placeholder={ai?.jev.has_key ? "new key" : "apikey_…"}
                onKeyDown={async (e) => {
                  if (e.key !== "Enter") return;
                  const value = e.currentTarget.value.trim();
                  e.currentTarget.value = "";
                  // A new key also enables Jev so it is ready to use; clearing the key disables it.
                  await apply(value ? { jev: { api_key: value, enabled: true } } : { jev: { api_key: "", enabled: false } });
                }}
              />
              <span className="muted small">
                {ai?.jev.has_key ? (
                  <>
                    Key saved ·{" "}
                    <a
                      href="#jev-key"
                      onClick={(e) => {
                        e.preventDefault();
                        if (keyRef.current) {
                          keyRef.current.value = "";
                          keyRef.current.focus();
                        }
                      }}
                    >
                      Replace
                    </a>
                  </>
                ) : (
                  "press Enter to save"
                )}
                {" · "}
                <a href="https://console.typesafe.ai/keys" target="_blank" rel="noreferrer">
                  get one
                </a>
              </span>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

// ─── Research ────────────────────────────────────────────────────────────────

function ResearchCard({
  ai,
  onPatch,
  onError,
}: {
  ai: AiSettings | null;
  onPatch: (p: AiSettingsPatch) => Promise<boolean>;
  onError: (e: string | null) => void;
}) {
  const { cue, apply } = useApplier(onPatch);
  const limit = ai?.chat_model.max_tool_rounds ?? 0;
  // null = untouched; the stored value decides whether the cap box is shown.
  const [capped, setCapped] = useState<boolean | null>(null);
  const showCap = capped ?? limit !== 0;
  return (
    <section className="card">
      <div className="settings-card-title">
        Research
        <SavedCue state={cue.state} />
      </div>
      <div className="settings-fields">
        <div className="settings-field">
          <label htmlFor="research-limit">Research limit</label>
          <select
            id="research-limit"
            value={showCap ? "cap" : "unlimited"}
            disabled={!ai}
            onChange={async (e) => {
              if (e.currentTarget.value === "cap") {
                setCapped(true);
                return;
              }
              setCapped(false);
              await apply({ chat_model: { max_tool_rounds: 0 } });
            }}
          >
            <option value="unlimited">Unlimited</option>
            <option value="cap">{showCap && limit > 0 ? `Cap (${limit})` : "Cap…"}</option>
          </select>
          {showCap && (
            <input
              id="research-limit-custom"
              type="number"
              min={1}
              step={1}
              style={{ width: "80px" }}
              disabled={!ai}
              defaultValue={limit > 0 ? limit : undefined}
              placeholder="rounds"
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
              }}
              onBlur={async (e) => {
                const value = Number(e.currentTarget.value);
                if (!Number.isInteger(value) || value < 1) {
                  onError("A research cap must be a whole number of 1 or more. Choose Unlimited for no cap.");
                  return;
                }
                await apply({ chat_model: { max_tool_rounds: value } });
              }}
            />
          )}
          <span className="muted small">
            How many rounds of footage look-ups the AI may do for one script, clip search or answer.
            Unlimited by default: it stops when it has what it needs, when you stop it, or when it keeps
            asking for the same thing.
          </span>
        </div>
      </div>
    </section>
  );
}

// ─── Editor instructions (system prompt) ─────────────────────────────────────

function PromptCard({
  settings,
  onSettings,
  onError,
}: {
  settings: ChatSettings;
  onSettings: (s: ChatSettings) => void;
  onError: (e: string | null) => void;
}) {
  const [text, setText] = useState(settings.system_prompt);
  // Only show the editing surface when there is something of the editor's own to edit;
  // the factory instructions stay fenced in a collapsed, read-only block.
  const [editing, setEditing] = useState(() => {
    const t = settings.system_prompt.trim();
    return t !== "" && t !== settings.default_system_prompt.trim();
  });
  const cue = useSavedCue();
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const save = (value: string) => {
    cue.saving();
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        onSettings(await setChatSystemPrompt(value));
        cue.saved();
        onError(null);
      } catch (e) {
        onError(String(e));
        cue.reset();
      }
    }, 800);
  };

  const isDefault = text.trim() === settings.default_system_prompt.trim() || !text.trim();

  return (
    <section className="card">
      <div className="settings-card-title">
        Editor instructions
        <SavedCue state={cue.state} />
      </div>
      <p className="muted small">
        The brief the AI follows when drafting scripts — story, shot choice, pacing, narration and audio. The
        footage look-up and the reply format are built in and always applied; your brief sets the rest.
      </p>
      {editing ? (
        <>
          <textarea
            className="prompt-editor"
            value={text}
            spellCheck={false}
            placeholder="e.g. Cut tight one-minute highlights; put the best speaker on screen for the hook; keep narration short and plain; end on what to watch next."
            onChange={(e) => {
              setText(e.target.value);
              save(e.target.value);
            }}
          />
          <div className="row prompt-actions">
            <span className="muted small">{isDefault ? "Using the default instructions" : "Using your instructions"}</span>
            <button
              className="ghost small"
              disabled={isDefault}
              onClick={() => {
                setText(settings.default_system_prompt);
                setEditing(false);
                save("");
              }}
            >
              Reset to default
            </button>
          </div>
        </>
      ) : (
        <div className="prompt-summary">
          <p className="muted small">
            Using the default instructions — the AI picks quotes and clips from your footage, writes the narration,
            and answers in the format GhostReel reads.
          </p>
          <div className="row">
            <button
              className="ghost small"
              onClick={() => {
                setText("");
                setEditing(true);
              }}
            >
              Write your own brief
            </button>
          </div>
        </div>
      )}
      {/* The fixed factory instructions, fenced off: read-only and collapsed by default. */}
      <details className="prompt-scaffold">
        <summary>How the AI is instructed (advanced)</summary>
        <p className="muted small">
          {editing
            ? "Your brief replaces the built-in instructions below. Either way the footage tools are added on top, so scripts keep working."
            : "This is what the AI receives now. Writing your own brief replaces it wholesale — the footage tools are added on top."}
        </p>
        <pre className="prompt-scaffold-text">{settings.default_system_prompt}</pre>
      </details>
    </section>
  );
}

// ─── Web access ──────────────────────────────────────────────────────────────

function WebAccess() {
  const [web, setWeb] = useState<WebStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bind, setBind] = useState("");
  const [port, setPort] = useState("");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const usernameRef = useRef<HTMLInputElement>(null);
  const cue = useSavedCue();

  const load = (s: WebStatus) => {
    setWeb(s);
    setBind(s.bind);
    setPort(String(s.port));
    setUser(s.auth_user);
    setPassword("");
  };

  useEffect(() => {
    // Nothing to show if the backend has no web server at all.
    webStatus().then(load).catch(() => {});
  }, []);

  const apply = async (patch: WebConfigPatch) => {
    cue.saving();
    try {
      load(await setWebConfig(patch));
      cue.saved();
      setError(null);
    } catch (err) {
      cue.reset();
      setError(String(err));
    }
  };

  if (!web) return null;

  // Read-only in a browser: these controls are reached *through* the server they would stop.
  const locked = !isDesktop;
  const exposed = web.enabled && !web.auth_enabled && !isLoopback(web.bind);
  const state = web.running
    ? `running on ${web.bind}:${web.port}`
    : web.enabled
      ? web.error
        ? "enabled, but it is not running"
        : "starting…"
      : "off — nothing is served";

  const commitBind = () => {
    const v = bind.trim();
    if (!v || v === web.bind) return;
    apply({ bind: v });
  };
  const commitPort = () => {
    const v = Number(port);
    if (!Number.isInteger(v) || v < 1 || v > 65535) {
      setError("The port must be a number between 1 and 65535");
      return;
    }
    if (v === web.port) return;
    apply({ port: v });
  };
  const commitUser = () => {
    const v = user.trim();
    if (v === web.auth_user) return;
    apply({ auth_user: v });
  };
  const commitPassword = () => {
    if (!password) return;
    apply({ auth_password: password });
  };
  const onEnter = (commit: () => void) => (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") commit();
  };

  return (
    <section className={`card web-access-card${locked ? " locked" : ""}`}>
      <div className="settings-card-title">
        Web access
        <SavedCue state={cue.state} />
      </div>
      <p className="muted small">
        Serve GhostReel over HTTP too — for a browser here, or on a phone or another computer on this network.
        Everything still runs on this computer; the browser is only a window onto it.
      </p>
      {error && <div className="banner bad">{error}</div>}
      {web.error && <div className="banner bad">The server did not start: {web.error}</div>}
      {exposed && (
        <div className="banner warn web-access-warning">
          <span>Anyone on the same network can open this server without signing in and use GhostReel as you.</span>
          {!locked && (
            <button className="ghost small" onClick={async () => {
              await apply({ auth_enabled: true });
              requestAnimationFrame(() => {
                usernameRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                usernameRef.current?.focus();
              });
            }}>
              Turn on sign-in
            </button>
          )}
          {locked && <span>Turn on sign-in in the desktop app: Settings → Web access.</span>}
        </div>
      )}
      <div className="settings-fields">
        <div className="settings-field check">
          <label htmlFor="web-serve">Serve</label>
          <input
            id="web-serve"
            type="checkbox"
            checked={web.enabled}
            disabled={locked}
            onChange={(e) => apply({ enabled: e.currentTarget.checked })}
          />
          <span className="muted small">{state}</span>
        </div>
        {web.running && web.urls.length > 0 && (
          <div className="settings-field">
            <label>Open at</label>
            <span className="muted small">
              {web.urls.map((u, i) => (
                <span key={u}>
                  {i > 0 && " · "}
                  <code>{u}</code>
                </span>
              ))}
            </span>
          </div>
        )}
        <div className="settings-field">
          <label htmlFor="web-bind">Address</label>
          <input
            id="web-bind"
            type="text"
            value={bind}
            disabled={locked}
            onChange={(e) => setBind(e.currentTarget.value)}
            onBlur={commitBind}
            onKeyDown={onEnter(commitBind)}
          />
          <span className="muted small">
            0.0.0.0 to reach it from other devices · 127.0.0.1 for this computer only
          </span>
        </div>
        <div className="settings-field">
          <label htmlFor="web-port">Port</label>
          <input
            id="web-port"
            type="number"
            min={1}
            max={65535}
            style={{ width: "7em" }}
            value={port}
            disabled={locked}
            onChange={(e) => setPort(e.currentTarget.value)}
            onBlur={commitPort}
            onKeyDown={onEnter(commitPort)}
          />
          <span className="muted small">1–65535</span>
        </div>
        <div className="settings-field check">
          <label htmlFor="web-auth">Sign-in</label>
          <input
            id="web-auth"
            type="checkbox"
            checked={web.auth_enabled}
            disabled={locked}
            onChange={(e) => apply({ auth_enabled: e.currentTarget.checked })}
          />
          <span className="muted small">
            protects this server with the name and password below
          </span>
        </div>
        <div className="settings-field">
          <label htmlFor="web-user">Username</label>
          <input
            id="web-user"
            ref={usernameRef}
            type="text"
            value={user}
            disabled={locked || !web.auth_enabled}
            onChange={(e) => setUser(e.currentTarget.value)}
            onBlur={commitUser}
            onKeyDown={onEnter(commitUser)}
          />
        </div>
        <div className="settings-field">
          <label htmlFor="web-password">Password</label>
          <input
            id="web-password"
            type="password"
            value={password}
            disabled={locked || !web.auth_enabled}
            placeholder="type to set a password"
            onChange={(e) => setPassword(e.currentTarget.value)}
            onBlur={commitPassword}
            onKeyDown={onEnter(commitPassword)}
          />
          <span className="muted small">leave blank to keep the saved password</span>
        </div>
      </div>
    </section>
  );
}

// ─── main page ───────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const [settings, setSettings] = useState<ChatSettings | null>(null);
  const [ai, setAi] = useState<AiSettings | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAiSettings().then(setAi).catch(() => {});
    getChatSettings()
      .then(setSettings)
      .catch((e) => setError(String(e)));
  }, []);

  const patchAi = async (patch: AiSettingsPatch): Promise<boolean> => {
    try {
      setAi(await setAiSettings(patch));
      setError(null);
      return true;
    } catch (err) {
      setError(String(err));
      return false;
    }
  };

  // Advanced · model runtime — collapsed by default; sections appear only when they apply.
  const showChatRuntime = Boolean(ai && ai.chat_model.backend !== "server");
  const showFrameServer = Boolean(ai && (ai.vision.backend === "server" || ai.vision.backend === "auto"));
  const chat = useApplier(patchAi);
  const frame = useApplier(patchAi);

  if (!settings) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const moeBlocks = ai?.chat_model.moe_blocks ?? 64;
  const moeLayers = Math.min(ai?.chat_model.cpu_moe_layers ?? 0, moeBlocks);

  return (
    <main>
      <header>
        <div>
          <h1>Settings</h1>
          <p className="muted">Changes save automatically · stored in ~/.ghostreel/config.toml</p>
        </div>
      </header>

      {error && <div className="banner bad">{error}</div>}

      <h2>Basic</h2>
      <div className="settings-group">
        <JevCard ai={ai} onPatch={patchAi} onError={setError} />
        <ResearchCard ai={ai} onPatch={patchAi} onError={setError} />
        <PromptCard settings={settings} onSettings={setSettings} onError={setError} />
        <WebAccess />
      </div>

      {(showChatRuntime || showFrameServer) && (
        <details className="settings-advanced">
          <summary>
            <ChevronDown size={15} strokeWidth={2} aria-hidden />
            <span>Advanced · model runtime</span>
            <span className="muted small">Recommended: automatic — most editors never need this</span>
          </summary>
          <div className="settings-advanced-body">
            {showChatRuntime && ai && (
              <section className="card">
                <div className="settings-card-title">
                  Script chat · model runtime
                  <SavedCue state={chat.cue.state} />
                </div>
                <p className="muted small">
                  Recommended: automatic — GhostReel fits these to your hardware. Changes apply when the model
                  reloads.
                </p>
                <div className="settings-fields">
                  <div className="settings-field">
                    <label htmlFor="chat-ctx">Context window</label>
                    <input
                      id="chat-ctx"
                      type="number"
                      min={2048}
                      max={131072}
                      step={2048}
                      style={{ width: "7em" }}
                      defaultValue={ai.chat_model.ctx_tokens}
                      onChange={async (e) => {
                        const v = Number(e.currentTarget.value);
                        if (v < 2048 || v > 131072) return;
                        await chat.apply({ chat_model: { ctx_tokens: v } });
                      }}
                    />
                    <span className="muted small">
                      tokens · more room for search results and long scripts — and more VRAM
                    </span>
                  </div>
                  <div className="settings-field">
                    <label htmlFor="chat-flash">Flash attention</label>
                    <select
                      id="chat-flash"
                      value={ai.chat_model.flash_attn}
                      onChange={async (e) => {
                        await chat.apply({ chat_model: { flash_attn: e.currentTarget.value } });
                      }}
                    >
                      <option value="auto">Automatic · use when supported</option>
                      <option value="on">On · require support</option>
                      <option value="off">Off · use f16 KV cache</option>
                    </select>
                    <span className="muted small">leave on Automatic — it turns itself on only when the GPU supports it</span>
                  </div>
                  <div className="settings-field">
                    <label htmlFor="chat-moe-ram">Model memory split (large models)</label>
                    <input
                      id="chat-moe-ram"
                      type="range"
                      min={0}
                      max={moeBlocks}
                      step={1}
                      style={{ width: "min(100%, 28em)" }}
                      value={moeLayers}
                      onChange={(e) =>
                        setAi({
                          ...ai,
                          chat_model: { ...ai.chat_model, cpu_moe_layers: Number(e.currentTarget.value) },
                        })
                      }
                      onPointerUp={async (e) => {
                        await chat.apply({ chat_model: { cpu_moe_layers: Number(e.currentTarget.value) } });
                      }}
                      onKeyUp={async (e) => {
                        await chat.apply({ chat_model: { cpu_moe_layers: Number(e.currentTarget.value) } });
                      }}
                    />
                    <span className="small">
                      {!ai.chat_model.cpu_moe_layers
                        ? "Automatic · fit to available GPU memory"
                        : ai.chat_model.moe_blocks
                          ? `${Math.round(moeLayers / moeBlocks * 100)}% · ${moeLayers} of ${moeBlocks} in system memory`
                          : `${moeLayers} in system memory`}
                    </span>
                    <span className="muted small">
                      Keeps part of a large model in system memory so it fits your GPU. Leave on automatic unless loading fails.
                    </span>
                  </div>
                  <div className="settings-field">
                    <label htmlFor="chat-kv">KV cache</label>
                    <select
                      id="chat-kv"
                      value={ai.chat_model.kv_cache}
                      onChange={async (e) => {
                        await chat.apply({ chat_model: { kv_cache: e.currentTarget.value } });
                      }}
                    >
                      <option value="q4_0">q4_0 — ~4× the context per GB</option>
                      <option value="q8_0">q8_0 — balanced</option>
                      <option value="f16">f16 — best quality, 4× the VRAM</option>
                    </select>
                    <span className="muted small">lower precision fits more context in the same memory</span>
                  </div>
                </div>
              </section>
            )}

            {showFrameServer && ai && (
              <section className="card">
                <div className="settings-card-title">
                  Frame descriptions · server
                  <SavedCue state={frame.cue.state} />
                </div>
                {ai.vision_caps.slots == null ? (
                  <p className="muted small">
                    This server doesn&apos;t say how many frames it can describe at once, so GhostReel sends one at a
                    time — the safe choice for LM Studio, Ollama or a hosted endpoint.
                  </p>
                ) : (
                  <>
                    <p className="muted small">
                      The server runs <strong>{ai.vision_caps.slots} frame description{ai.vision_caps.slots === 1 ? "" : "s"}</strong>
                      {ai.vision_caps.slot_ctx != null && <> of {ai.vision_caps.slot_ctx.toLocaleString()} tokens each</>} at
                      once. Describing frames in parallel reuses the model already in memory — measured about 1.7×
                      faster on real footage. Asking for more than the server has is harmless; the extras queue.
                    </p>
                    <div className="settings-fields">
                      <div className="settings-field">
                        <label htmlFor="frames-concurrency">Frames at once</label>
                        <input
                          id="frames-concurrency"
                          type="number"
                          min={1}
                          max={16}
                          style={{ width: "5em" }}
                          defaultValue={ai.vision.describe_concurrency ?? 4}
                          onChange={async (e) => {
                            const v = Number(e.currentTarget.value);
                            if (v < 1 || v > 16) return;
                            await frame.apply({ vision: { describe_concurrency: v } });
                          }}
                        />
                        <span className="muted small">
                          {ai.vision_caps.slots === 1
                            ? "this server runs one at a time — start it with more slots to gain anything"
                            : `up to ${ai.vision_caps.slots} will run in parallel here`}
                        </span>
                      </div>
                    </div>
                  </>
                )}
                {ai.vision_caps.router && (
                  <p className="muted small">
                    This server is a llama.cpp router: the model it runs can be changed without restarting it.
                  </p>
                )}
              </section>
            )}
          </div>
        </details>
      )}
    </main>
  );
}
