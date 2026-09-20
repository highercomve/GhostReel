import { useEffect, useRef, useState } from "react";
import {
  getAiSettings,
  getChatSettings,
  setAiSettings,
  setChatSystemPrompt,
  type AiSettings,
  type ChatSettings,
} from "./api";

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

      <h2>Script chat · editorial judge</h2>
      <section className="card">
        <p className="muted small">
          GhostReel can already tell when a cut runs long or stops someone mid-sentence, and it fixes those itself.
          What it cannot tell is whether the shot on screen shows what the voice is talking about, whether the opening
          is worth watching, or whether the ending lands. Jev — a hosted model that answers in numbers rather than
          words — reads each finished cut and says. What it finds goes back to the editor before it redrafts.
        </p>
        <p className="muted small">
          Off by default, and it is the one part of GhostReel that leaves this computer: judging sends the cut's
          spoken words and the descriptions of what is on screen to <code>api.typesafe.ai</code>. Nothing is sent
          while it is off. A judgement costs a fraction of a cent.
        </p>
        <div className="settings-fields">
          <div className="settings-field">
            <label>Judge each cut</label>
            <input
              type="checkbox"
              checked={ai?.jev.enabled ?? false}
              disabled={!ai}
              onChange={async (e) => {
                const on = e.currentTarget.checked;
                try {
                  setAi(await setAiSettings({ jev: { enabled: on } }));
                  setError(null);
                } catch (err) {
                  setError(String(err));
                }
              }}
            />
            <span className="muted small">
              {ai?.jev.enabled ? "on — every finished script is judged" : "off — nothing is sent anywhere"}
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
                    const value = e.currentTarget.value;
                    e.currentTarget.value = "";
                    try {
                      setAi(await setAiSettings({ jev: { api_key: value } }));
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
    </main>
  );
}
