//! Frame descriptions with a vision model (plan D1, D11, S0 findings): highllama/llama-server over
//! the OpenAI API, or the local `ghostreel-llm` helper. Output is constrained by a JSON schema with
//! length limits (prevents repetition loops) and thinking is disabled (7× faster, fewer invented
//! details).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};

use crate::Error;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameDescription {
    pub description: String,
    pub visible_text: Vec<String>,
    pub objects: Vec<String>,
    pub setting: String,
    pub shot: String,
    pub tags: Vec<String>,
}

impl FrameDescription {
    /// Text used for search: everything a person might type to find this moment.
    pub fn search_text(&self) -> String {
        let mut parts = vec![self.description.clone()];
        if !self.visible_text.is_empty() {
            parts.push(format!("On screen: {}", self.visible_text.join(" · ")));
        }
        if !self.objects.is_empty() {
            parts.push(format!("Objects: {}", self.objects.join(", ")));
        }
        if !self.setting.is_empty() {
            parts.push(format!("Setting: {}", self.setting));
        }
        if !self.shot.is_empty() {
            parts.push(format!("Shot: {}", self.shot));
        }
        if !self.tags.is_empty() {
            parts.push(format!("Tags: {}", self.tags.join(", ")));
        }
        parts.retain(|p| !p.trim().is_empty());
        parts.join("\n")
    }
}

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "description": {"type": "string", "maxLength": 600},
            "visible_text": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 10},
            "objects": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 10},
            "setting": {"type": "string", "maxLength": 120},
            "shot": {"type": "string", "maxLength": 40},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 10}
        },
        "required": ["description", "visible_text", "objects", "setting", "shot", "tags"],
        "additionalProperties": false
    })
}

/// Prompt for one frame. `speech` is what is being said around this moment (helps name things
/// that are visible, e.g. a product name); the model is told not to describe speech it can't see.
pub fn prompt(speech: Option<&str>) -> String {
    let mut p = String::from(
        "Describe this video frame for a video search index. Be concrete: people (no names or \
         identities), actions, objects, products, places, screen content. Keep visible_text to the most \
         important distinct text you can actually read (max 10 short items, no repeats). shot is one of: \
         close-up, medium, wide, screen recording, slide, title card, b-roll, other.",
    );
    if let Some(s) = speech.map(str::trim).filter(|s| !s.is_empty()) {
        let s: String = s.chars().take(600).collect();
        p.push_str(&format!("\nSpeech around this moment (use it only to name things that are visible): \"{s}\""));
    }
    p.push_str(
        "\nReply with ONLY JSON: {\"description\": str, \"visible_text\": [str], \"objects\": [str], \
         \"setting\": str, \"shot\": str, \"tags\": [str]}",
    );
    p
}

/// Tolerant parse: accepts code fences or text around the JSON object.
pub fn parse_description(content: &str) -> Result<FrameDescription, Error> {
    let start = content.find('{');
    let end = content.rfind('}');
    let json = match (start, end) {
        (Some(s), Some(e)) if e > s => &content[s..=e],
        _ => return Err(Error::Vision(format!("no JSON in model output: {}", preview(content)))),
    };
    let d: FrameDescription =
        serde_json::from_str(json).map_err(|e| Error::Vision(format!("bad JSON ({e}): {}", preview(content))))?;
    if d.description.trim().is_empty() {
        return Err(Error::Vision("empty description".into()));
    }
    Ok(d)
}

fn preview(s: &str) -> String {
    let p: String = s.chars().take(160).collect();
    if s.chars().count() > 160 { format!("{p}…") } else { p }
}

// ---- server ---------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ServerVision {
    pub url: String,
    /// Model id to send (empty: omit).
    pub model: String,
    pub api_key: String,
    client: reqwest::Client,
}

impl ServerVision {
    pub fn new(url: &str, model: &str, api_key: &str) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(300))
            .build()
            .expect("static reqwest client config");
        Self { url: url.trim_end_matches('/').to_string(), model: model.into(), api_key: api_key.into(), client }
    }

    pub async fn describe(&self, image: &Path, speech: Option<&str>) -> Result<FrameDescription, Error> {
        let bytes = tokio::fs::read(image).await.map_err(|e| Error::Io(image.to_path_buf(), e))?;
        let data_url = format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt(speech)}
                ]
            }],
            "max_tokens": 1200,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "chat_template_kwargs": {"enable_thinking": false},
            "response_format": {"type": "json_schema", "json_schema": {"name": "frame", "schema": schema()}}
        });
        if !self.model.is_empty() {
            body["model"] = json!(self.model);
        }
        let mut req = self.client.post(format!("{}/v1/chat/completions", self.url)).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.map_err(|e| Error::Vision(format!("vision server at {}: {e}", self.url)))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::Vision(format!("vision server at {}: HTTP {status} {}", self.url, preview(&text))));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::Vision(format!("vision server at {}: bad response: {e}", self.url)))?;
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or_default();
        parse_description(content)
    }
}

// ---- local helper ---------------------------------------------------------------------------

/// A running `ghostreel-llm` process. Models stay loaded until it is dropped (kill on drop).
pub struct LocalLlm {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    next_id: u64,
    pub embed_dim: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LocalModels {
    pub helper: PathBuf,
    pub vision: Option<(PathBuf, PathBuf)>,
    pub embed: Option<PathBuf>,
    pub cpu: bool,
    /// Context window, KV cache precision and flash attention for this helper. Frame descriptions
    /// and the script chat run the same binary with different values (see `config::VisionConfig`).
    pub runtime: HelperRuntime,
}

/// How much room the helper gets, and how it spends VRAM on it.
#[derive(Debug, Clone, PartialEq)]
pub struct HelperRuntime {
    pub ctx_tokens: u32,
    pub kv_cache: String,
    pub flash_attn: String,
    /// Whether this helper's calls may reason before answering.
    pub think: bool,
}

impl Default for HelperRuntime {
    fn default() -> Self {
        Self {
            ctx_tokens: crate::config::DESCRIBE_CTX_TOKENS,
            kv_cache: "q4_0".into(),
            flash_attn: "auto".into(),
            think: false,
        }
    }
}

impl From<&crate::config::VisionConfig> for HelperRuntime {
    fn from(c: &crate::config::VisionConfig) -> Self {
        Self {
            ctx_tokens: c.ctx_tokens,
            kv_cache: c.kv_cache.clone(),
            flash_attn: c.flash_attn.clone(),
            think: c.think,
        }
    }
}

impl LocalLlm {
    pub async fn start(m: &LocalModels) -> Result<Self, Error> {
        let mut cmd = crate::proc::command(&m.helper);
        if let Some((model, mmproj)) = &m.vision {
            cmd.arg("--model").arg(model).arg("--mmproj").arg(mmproj);
        }
        if let Some(e) = &m.embed {
            cmd.arg("--embed-model").arg(e);
        }
        if m.cpu {
            cmd.arg("--cpu");
        }
        cmd.arg("--ctx")
            .arg(m.runtime.ctx_tokens.to_string())
            .arg("--kv-type")
            .arg(&m.runtime.kv_cache)
            .arg("--flash-attn")
            .arg(&m.runtime.flash_attn);
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::Vision(format!("cannot run {}: {e}", m.helper.display())))?;
        let stdin = child.stdin.take().ok_or_else(|| Error::Vision("helper stdin unavailable".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| Error::Vision("helper stdout unavailable".into()))?;
        let stderr = child.stderr.take();
        let err_lines = tokio::spawn(async move {
            let mut keep = Vec::new();
            if let Some(e) = stderr {
                let mut l = BufReader::new(e).lines();
                while let Ok(Some(line)) = l.next_line().await {
                    let lower = line.to_lowercase();
                    if line.starts_with("ghostreel-llm:") || lower.contains("error") || lower.contains("out of memory")
                    {
                        keep.push(line);
                        if keep.len() > 4 {
                            keep.remove(0);
                        }
                    }
                }
            }
            keep.join(" | ")
        });
        let mut lines = BufReader::new(stdout).lines();
        // Loading 4 GB from a cold disk can take a while.
        let ready = tokio::time::timeout(Duration::from_secs(600), lines.next_line()).await;
        match ready {
            Ok(Ok(Some(line))) => {
                let v: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
                if v["ready"] != json!(true) {
                    return Err(Error::Vision(format!("helper did not start: {line}")));
                }
                let embed_dim = v["embed_dim"].as_u64().map(|d| d as usize);
                Ok(Self { child, stdin, lines, next_id: 1, embed_dim })
            }
            Ok(_) => {
                let _ = child.wait().await;
                let why = err_lines.await.unwrap_or_default();
                Err(Error::Vision(if why.is_empty() { "helper exited while loading models".into() } else { why }))
            }
            Err(_) => Err(Error::Vision("helper took more than 10 minutes to load models".into())),
        }
    }

    async fn request(&mut self, mut req: Value) -> Result<Value, Error> {
        let id = self.next_id;
        self.next_id += 1;
        req["id"] = json!(id);
        let line = format!("{req}\n");
        self.stdin.write_all(line.as_bytes()).await.map_err(|e| Error::Vision(format!("helper write: {e}")))?;
        self.stdin.flush().await.map_err(|e| Error::Vision(format!("helper write: {e}")))?;
        loop {
            let next = tokio::time::timeout(Duration::from_secs(600), self.lines.next_line())
                .await
                .map_err(|_| Error::Vision("helper timed out".into()))?
                .map_err(|e| Error::Vision(format!("helper read: {e}")))?;
            let Some(line) = next else {
                return Err(Error::Vision("local model helper exited (crash or out of memory)".into()));
            };
            let v: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            if v["id"] != json!(id) {
                continue;
            }
            if v["ok"] == json!(true) {
                return Ok(v);
            }
            return Err(Error::Vision(v["error"].as_str().unwrap_or("helper error").to_string()));
        }
    }

    pub async fn describe(&mut self, image: &Path, speech: Option<&str>) -> Result<FrameDescription, Error> {
        let v = self
            .request(json!({
                "cmd": "describe",
                "image": image,
                "prompt": prompt(speech),
                "schema": schema(),
                "max_tokens": 1200
            }))
            .await?;
        parse_description(v["content"].as_str().unwrap_or_default())
    }

    pub async fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let v = self.request(json!({ "cmd": "embed", "texts": texts })).await?;
        serde_json::from_value(v["embeddings"].clone()).map_err(|e| Error::Vision(format!("bad embeddings: {e}")))
    }

    /// Kill the helper. A generation runs inside the helper process and does not check anything
    /// we set out here, so stopping a turn means ending the process; the next turn starts a new
    /// one. Without this, Stop only took effect after the model had finished anyway.
    pub async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }

    pub async fn complete(&mut self, prompt: &str, schema: Option<Value>) -> Result<String, Error> {
        self.complete_limited(prompt, schema, DEFAULT_COMPLETE_TOKENS).await
    }

    /// `complete` with an explicit output budget. A whole script's JSON runs far past the default,
    /// and the helper simply stops emitting at the cap — mid-object, so the JSON never parses.
    /// The helper reports that as `truncated`; turn it into an error instead of handing back a
    /// fragment the caller will fail to parse for no stated reason.
    pub async fn complete_limited(
        &mut self,
        prompt: &str,
        schema: Option<Value>,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.complete_full(prompt, schema, max_tokens, false).await
    }

    /// `complete_limited`, optionally letting the model reason before it answers. Thinking costs
    /// tokens and time, so it is for the few calls whose quality carries a whole script, not for
    /// per-keyframe work.
    pub async fn complete_full(
        &mut self,
        prompt: &str,
        schema: Option<Value>,
        max_tokens: usize,
        think: bool,
    ) -> Result<String, Error> {
        self.complete_draw(prompt, schema, max_tokens, think, None).await
    }

    /// One draw of an answer. `draw` picks the sampling seed, so the same prompt asked twice with
    /// different draws gives different answers — without it the helper is deterministic and asking
    /// again is pointless. An older helper ignores the field and simply repeats itself, which the
    /// caller detects by comparing the bytes.
    pub async fn complete_draw(
        &mut self,
        prompt: &str,
        schema: Option<Value>,
        max_tokens: usize,
        think: bool,
        draw: Option<u32>,
    ) -> Result<String, Error> {
        let mut req = json!({
            "cmd": "complete",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "think": think,
        });
        if let Some(n) = draw {
            // 42 is what the helper has always used; later draws move off it, and a little heat
            // is needed or top_k alone would keep returning the same tokens.
            req["seed"] = json!(42u32.wrapping_add(n));
            if n > 0 {
                req["temperature"] = json!(0.7);
            }
        }
        if let Some(s) = schema {
            req["schema"] = s;
        }
        let v = self.request(req).await?;
        if v["truncated"].as_bool().unwrap_or(false) {
            return Err(Error::Vision(format!(
                "the local model hit its {max_tokens}-token output limit and the answer is cut off; \
                 raise chat_model.ctx_tokens or ask for a shorter script"
            )));
        }
        Ok(v["content"].as_str().unwrap_or_default().to_string())
    }
}

/// Output budget for a plain `complete` call: enough for a tool action or a short answer.
pub const DEFAULT_COMPLETE_TOKENS: usize = 2048;

pub enum Describer {
    Server(ServerVision),
    Local(Box<LocalLlm>),
    Cli(crate::cliagent::CliAgent),
}

impl Describer {
    pub async fn describe(&mut self, image: &Path, speech: Option<&str>) -> Result<FrameDescription, Error> {
        match self {
            Describer::Server(s) => s.describe(image, speech).await,
            Describer::Local(l) => l.describe(image, speech).await,
            Describer::Cli(agent) => {
                let schema_hint = serde_json::to_string(&schema()).unwrap_or_default();
                // Add speech context if present. CliAgent::describe prepends "Read image…reply with JSON: ".
                let hint = if let Some(sp) = speech {
                    format!("{schema_hint}\n\nAudio near this frame: {sp}")
                } else {
                    schema_hint
                };
                let json_text = agent.describe(image, &hint).await?;
                parse_description(&json_text)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    #[test]
    fn parses_fenced_and_rejects_empty() {
        let d = parse_description(
            "```json\n{\"description\":\"A terminal\",\"visible_text\":[\"$ ls\"],\"objects\":[],\"setting\":\"desk\",\"shot\":\"screen recording\",\"tags\":[\"cli\"]}\n```",
        )
        .unwrap();
        assert_eq!(d.visible_text, vec!["$ ls"]);
        assert!(d.search_text().contains("On screen: $ ls"));
        assert!(parse_description("{\"description\":\"\"}").is_err());
        assert!(parse_description("I cannot see").is_err());
        // Missing optional arrays default to empty.
        assert!(parse_description("{\"description\":\"x\"}").unwrap().tags.is_empty());
    }

    #[test]
    fn prompt_includes_speech_context_trimmed() {
        let p = prompt(Some(&"word ".repeat(500)));
        assert!(p.contains("Speech around this moment"));
        assert!(p.len() < 2000);
        assert!(!prompt(Some("  ")).contains("Speech"));
    }

    #[tokio::test]
    async fn server_request_shape_and_parse() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let captured = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
        let cap = captured.clone();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 65536];
            loop {
                let n = sock.read(&mut tmp).await.unwrap();
                buf.extend_from_slice(&tmp[..n]);
                let s = String::from_utf8_lossy(&buf);
                if let Some(h) = s.find("\r\n\r\n") {
                    let len: usize = s[..h]
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= h + 4 + len {
                        *cap.lock().await = s[h + 4..].to_string();
                        break;
                    }
                }
            }
            let content = r#"{\"description\":\"Two boards on a desk\",\"visible_text\":[\"CM5\"],\"objects\":[\"board\"],\"setting\":\"office\",\"shot\":\"close-up\",\"tags\":[\"hardware\"]}"#;
            let body = format!(r#"{{"choices":[{{"message":{{"content":"{content}"}}}}]}}"#);
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: application/json\r\n\r\n{body}",
                body.len()
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
        });
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("f.jpg");
        std::fs::write(&img, b"\xff\xd8\xff fake jpeg").unwrap();
        let v = ServerVision::new(&url, "Bonsai-27B-Q1_0", "");
        let d = v.describe(&img, Some("this is the CM5")).await.unwrap();
        assert_eq!(d.visible_text, vec!["CM5"]);
        let sent: Value = serde_json::from_str(&captured.lock().await).unwrap();
        assert_eq!(sent["model"], "Bonsai-27B-Q1_0");
        assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(sent["response_format"]["type"], "json_schema");
        assert!(
            sent["messages"][0]["content"][0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
        assert!(sent["messages"][0]["content"][1]["text"].as_str().unwrap().contains("this is the CM5"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_helper_protocol() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let helper = tmp.path().join("llm");
        // Fake helper: ready line, then answers describe/embed requests by id.
        std::fs::write(
            &helper,
            r#"#!/bin/sh
echo '{"ready":true,"vision":true,"embed_dim":3}'
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  case "$line" in
    *'"cmd":"describe"'*) printf '{"id":%s,"ok":true,"content":"{\\"description\\":\\"a cat\\",\\"visible_text\\":[],\\"objects\\":[\\"cat\\"],\\"setting\\":\\"sofa\\",\\"shot\\":\\"medium\\",\\"tags\\":[]}"}\n' "$id" ;;
    *'"cmd":"embed"'*) printf '{"id":%s,"ok":true,"embeddings":[[1,0,0],[0,1,0]]}\n' "$id" ;;
  esac
done
"#,
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let models = LocalModels {
            helper,
            vision: Some(("m".into(), "p".into())),
            embed: None,
            cpu: false,
            runtime: Default::default(),
        };
        let mut llm = LocalLlm::start(&models).await.unwrap();
        assert_eq!(llm.embed_dim, Some(3));
        let d = llm.describe(Path::new("/x.jpg"), None).await.unwrap();
        assert_eq!(d.objects, vec!["cat"]);
        let e = llm.embed(&["a".into(), "b".into()]).await.unwrap();
        assert_eq!(e, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_truncated_completion_is_an_error_not_a_fragment() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let helper = tmp.path().join("llm");
        // Fake helper: echoes back the requested budget and reports the answer as cut off.
        std::fs::write(
            &helper,
            r#"#!/bin/sh
echo '{"ready":true,"vision":true,"embed_dim":3}'
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed 's/.*"id":\([0-9]*\).*/\1/')
  mt=$(printf '%s' "$line" | sed 's/.*"max_tokens":\([0-9]*\).*/\1/')
  case "$line" in
    *'"cmd":"complete"'*)
      if [ "$mt" -ge 4096 ]; then
        printf '{"id":%s,"ok":true,"content":"{\\"done\\":true}","truncated":false}\n' "$id"
      else
        printf '{"id":%s,"ok":true,"content":"{\\"beats\\":[{\\"narr","truncated":true}\n' "$id"
      fi ;;
  esac
done
"#,
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let models = LocalModels {
            helper,
            vision: Some(("m".into(), "p".into())),
            embed: None,
            cpu: false,
            runtime: Default::default(),
        };
        let mut llm = LocalLlm::start(&models).await.unwrap();

        // Small budget: the helper cuts the JSON off. That must not come back as a fragment.
        let err = llm.complete_limited("prompt", None, 2048).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cut off"), "got: {msg}");
        assert!(msg.contains("2048"), "got: {msg}");

        // A budget big enough for the whole answer succeeds.
        let ok = llm.complete_limited("prompt", None, 4096).await.unwrap();
        assert_eq!(ok, r#"{"done":true}"#);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_helper_start_failure_is_explained() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let helper = tmp.path().join("llm");
        std::fs::write(&helper, "#!/bin/sh\necho 'ghostreel-llm: loading m.gguf: out of memory' >&2\nexit 1\n")
            .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let models = LocalModels {
            helper,
            vision: Some(("m".into(), "p".into())),
            embed: None,
            cpu: false,
            runtime: Default::default(),
        };
        let err = LocalLlm::start(&models).await.err().unwrap();
        assert!(err.to_string().contains("out of memory"), "{err}");
    }
}
