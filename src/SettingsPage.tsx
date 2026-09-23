import { useEffect, useRef, useState } from "react";
import {
  getAiSettings,
  getChatSettings,
  isDesktop,
  setAiSettings,
  setChatSystemPrompt,
  setWebConfig,
  webStatus,
  type AiSettings,
  type ChatSettings,
  type WebConfigPatch,
  type WebStatus,
} from "./api";

/** Bound here, nothing else on the network can reach it — which is what makes a password optional. */
const isLoopback = (bind: string) => bind === "127.0.0.1" || bind === "localhost" || bind === "::1";

function WebAccess() {
  const [web, setWeb] = useState<WebStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bind, setBind] = useState("");
  const [port, setPort] = useState("");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");

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
    try {
      load(await setWebConfig(patch));
      setError(null);
    } catch (err) {
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
    <>
      <h2>Web access</h2>
      <section className="card">
        <p className="muted small">
          Serve GhostReel over HTTP as well as in this window, so it can be opened in a browser — here, or on a
          phone or another computer on the same network. Everything still runs on this computer; the browser is
          only a window onto it.
        </p>
        {locked && (
          <p className="muted small">
            You are looking at GhostReel through this server, so it can only be changed in the desktop app —
            switching it off from here would close the page you are reading.
          </p>
        )}
        {error && <div className="banner bad">{error}</div>}
        {web.error && <div className="banner bad">The server did not start: {web.error}</div>}
        {exposed && (
          <div className="banner bad">
            This server has no password and is not limited to this computer. Anyone on the same network can open
            it and use GhostReel as you: read and change your projects, watch your footage, and reach the files on
            this computer that GhostReel can reach. Turn sign-in on below unless you trust every device on this
            network.
          </div>
        )}
        <div className="settings-fields">
          <div className="settings-field">
            <label>Serve</label>
            <input
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
            <label>Address</label>
            <input
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
            <label>Port</label>
            <input
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
          </div>
          <div className="settings-field">
            <label>Sign-in</label>
            <input
              type="checkbox"
              checked={web.auth_enabled}
              disabled={locked}
              onChange={(e) => apply({ auth_enabled: e.currentTarget.checked })}
            />
            <span className="muted small">
              {web.auth_enabled
                ? "on — the browser asks for the name and password below"
                : "off — anyone who can reach the address is let straight in"}
            </span>
          </div>
          <div className="settings-field">
            <label>Username</label>
            <input
              type="text"
              value={user}
              disabled={locked || !web.auth_enabled}
              onChange={(e) => setUser(e.currentTarget.value)}
              onBlur={commitUser}
              onKeyDown={onEnter(commitUser)}
            />
          </div>
          <div className="settings-field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              disabled={locked || !web.auth_enabled}
              placeholder="type to set a password"
              onChange={(e) => setPassword(e.currentTarget.value)}
              onBlur={commitPassword}
              onKeyDown={onEnter(commitPassword)}
            />
            <span className="muted small">left blank, the saved password stays as it is</span>
          </div>
        </div>
      </section>
    </>
  );
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<ChatSettings | null>(null);
  const [ai, setAi] = useState<AiSettings | null>(null);
  const [text, setText] = useState("");
  const [saved, setSaved] = useState<"idle" | "saving" | "saved">("idle");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getAiSettings().then(setAi).catch(() => {});
    getChatSettings()
      .then((s) => {
        setSettings(s);
        setText(s.system_prompt.trim() ? s.system_prompt : s.default_system_prompt);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const save = (value: string) => {
    setSaved("saving");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        setSettings(await setChatSystemPrompt(value));
        setSaved("saved");
        setError(null);
      } catch (e) {
        setError(String(e));
        setSaved("idle");
      }
    }, 800);
  };

  if (!settings) return <main>{error ? <div className="banner bad">{error}</div> : <p className="muted">Loading…</p>}</main>;

  const isDefault = text.trim() === settings.default_system_prompt.trim() || !text.trim();

  return (
    <main>
      <header>
        <div>
          <h1>Settings</h1>
          <p className="muted">Saved in ~/.ghostreel/config.toml</p>
        </div>
      </header>

      {ai && ai.chat_model.backend !== "server" && (
        <>
          <h2>Script chat · model on this computer</h2>
          <section className="card">
            <p className="muted small">
              How much room the model has while drafting. A bigger window fits more search results and a longer
              script, but uses more VRAM. The frame-description model has its own, smaller window (Models page).
            </p>
            <div className="settings-fields">
              <div className="settings-field">
                <label>Context window</label>
                <input
                  type="number"
                  min={2048}
                  max={131072}
                  step={2048}
                  style={{ width: "7em" }}
                  defaultValue={ai.chat_model.ctx_tokens}
                  onChange={async (e) => {
                    const v = Number(e.currentTarget.value);
                    if (v < 2048 || v > 131072) return;
                    try {
                      setAi(await setAiSettings({ chat_model: { ctx_tokens: v } }));
                    } catch (err) {
                      setError(String(err));
                    }
                  }}
                />
                <span className="muted small">tokens · 32768 is a good default on an 8 GB card</span>
              </div>
              <div className="settings-field">
                <label>KV cache</label>
                <select
                  value={ai.chat_model.kv_cache}
                  onChange={async (e) => {
                    try {
                      setAi(await setAiSettings({ chat_model: { kv_cache: e.currentTarget.value } }));
                    } catch (err) {
                      setError(String(err));
                    }
                  }}
                >
                  <option value="q4_0">q4_0 — ~4× the context per GB</option>
                  <option value="q8_0">q8_0 — balanced</option>
                  <option value="f16">f16 — best quality, 4× the VRAM</option>
                </select>
              </div>
            </div>
          </section>
        </>
      )}

      {ai && (ai.vision.backend === "server" || ai.vision.backend === "auto") && (
        <>
          <h2>Frame descriptions · server</h2>
          <section className="card">
            {ai.vision_caps.slots == null ? (
              <p className="muted small">
                This server doesn't report what it can do, so GhostReel describes one frame at a time. That is the
                safe choice for LM Studio, Ollama or a hosted endpoint — only llama.cpp says how many requests it
                will really work on at once, and guessing wrong just makes requests queue.
              </p>
            ) : (
              <>
                <p className="muted small">
                  The server reports <strong>{ai.vision_caps.slots} slot{ai.vision_caps.slots === 1 ? "" : "s"}</strong>
                  {ai.vision_caps.slot_ctx != null && <> of {ai.vision_caps.slot_ctx.toLocaleString()} tokens each</>}.
                  Generating a token means reading every weight out of VRAM, so the card spends most of its time
                  waiting on memory; describing several frames at once reads those weights once and answers all of
                  them, which measured 1.7× faster on real footage. Asking for more than the server has is harmless —
                  the extra requests simply queue.
                </p>
                <div className="settings-fields">
                  <div className="settings-field">
                    <label>Frames at once</label>
                    <input
                      type="number"
                      min={1}
                      max={16}
                      style={{ width: "5em" }}
                      defaultValue={ai.vision.describe_concurrency ?? 4}
                      onChange={async (e) => {
                        const v = Number(e.currentTarget.value);
                        if (v < 1 || v > 16) return;
                        try {
                          setAi(await setAiSettings({ vision: { describe_concurrency: v } }));
                          setError(null);
                        } catch (err) {
                          setError(String(err));
                        }
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
                This server is a <strong>llama.cpp router</strong>: the model and the flags it runs with can be
                changed without restarting it.
              </p>
            )}
          </section>
        </>
      )}

      <h2>Script chat · Jev (builder & judge)</h2>
      <section className="card">
        <p className="muted small">
          Jev is TypeSafe&apos;s hosted System One model. It can assemble a first-pass cut out of the index in ~15 s
          (by choosing quotes and b-roll directly from indexed footage, used in &ldquo;Jev&rdquo; and &ldquo;Jev → model&rdquo;
          modes), detect off-mic interviewers, and score finished drafts editorially (opening, ending, and shot-to-voice match).
        </p>
        <p className="muted small">
          Off by default, and it is the one part of GhostReel that leaves this computer: when enabled, transcripts
          and frame descriptions are sent to <code>api.typesafe.ai</code>. Nothing is sent while it is off.
        </p>
        <div className="settings-fields">
          <div className="settings-field">
            <label>Enable Jev</label>
            <input
              type="checkbox"
              checked={ai?.jev.enabled ?? false}
              disabled={!ai}
              onChange={async (e) => {
                const on = e.currentTarget.checked;
                if (on && !ai?.jev.has_key && !ai?.jev.key_from_env) {
                  setError("Turning Jev on needs an API key: enter one below first");
                  return;
                }
                try {
                  setAi(await setAiSettings({ jev: { enabled: on } }));
                  setError(null);
                } catch (err) {
                  setError(String(err));
                }
              }}
            />
            <span className="muted small">
              {ai?.jev.enabled ? "on — Jev can build cuts, detect interviewers, and judge" : "off — nothing is sent anywhere"}
            </span>
          </div>
          <div className="settings-field">
            <label>Enable Jev to be use in the Script generation</label>
            <input
              type="checkbox"
              checked={ai?.jev.judge ?? true}
              disabled={!ai || !ai?.jev.enabled}
              onChange={async (e) => {
                const on = e.currentTarget.checked;
                try {
                  setAi(await setAiSettings({ jev: { judge: on } }));
                  setError(null);
                } catch (err) {
                  setError(String(err));
                }
              }}
            />
            <span className="muted small">
              {!ai?.jev.enabled
                ? "disabled — enable Jev above"
                : ai?.jev.judge
                  ? "on — every finished script is scored editorially"
                  : "off — scripts are not scored automatically"}
            </span>
          </div>
          <div className="settings-field">
            <label>API key</label>
            {ai?.jev.key_from_env ? (
              <span className="muted small">
                taken from <code>TYPESAFE_API_KEY</code> in the environment, which wins over anything typed here
              </span>
            ) : (
              <>
                <input
                  type="password"
                  style={{ width: "22em" }}
                  placeholder={ai?.jev.has_key ? "•••••••• saved — type to replace" : "apikey_…"}
                  onKeyDown={async (e) => {
                    if (e.key !== "Enter") return;
                    const value = e.currentTarget.value.trim();
                    e.currentTarget.value = "";
                    try {
                      // If saving a non-empty key, also enable Jev so it's ready to use right away.
                      // If saving an empty key (clearing it), disable Jev.
                      const patch = value
                        ? { api_key: value, enabled: true }
                        : { api_key: "", enabled: false };
                      setAi(await setAiSettings({ jev: patch }));
                      setError(null);
                    } catch (err) {
                      setError(String(err));
                    }
                  }}
                />
                <span className="muted small">
                  {ai?.jev.has_key ? "saved in config.toml · Enter to replace" : "press Enter to save"}
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

      <h2>Script chat · editor instructions</h2>
      <section className="card">
        <p className="muted small">
          The system prompt that tells the model how to edit: story, shot choice, pacing, narration and audio. It applies
          to the next chat message. <code>{"{project}"}</code>, <code>{"{fps}"}</code>, <code>{"{width}"}</code> and{" "}
          <code>{"{height}"}</code> are filled in. The footage tools and the rule that clips must come from indexed
          footage are always added, so scripts keep working whatever you write here.
        </p>
        {error && <div className="banner bad">{error}</div>}
        <textarea
          className="prompt-editor"
          value={text}
          spellCheck={false}
          onChange={(e) => {
            setText(e.target.value);
            save(e.target.value);
          }}
        />
        <div className="row prompt-actions">
          <span className="muted small">
            {isDefault ? "Using the default instructions" : "Using your instructions"}
            {saved === "saving" ? " · saving…" : saved === "saved" ? " · saved" : ""}
          </span>
          <button
            className="ghost small"
            disabled={isDefault}
            onClick={() => {
              setText(settings.default_system_prompt);
              save("");
            }}
          >
            Reset to default
          </button>
        </div>
      </section>

      <WebAccess />
    </main>
  );
}
