//! Probing the optional AI servers and resolving each `auto | local | server` switch (plan §2a).
//!
//! A server is only used when it is reachable **and capable**:
//! - vision: an OpenAI-compatible chat server whose model accepts images
//!   (llama.cpp `/props` → `modalities.vision`), e.g. highllama on :8089;
//! - embeddings: returns `embeddinggemma` vectors of [`EMBED_DIM`] (anything else would silently
//!   corrupt search, plan §2c), e.g. highllama's embeddings server on :8091;
//! - speech-to-text: GhostPen's server advertising timestamped segments on `/v1/models`.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Backend, EMBED_DIM};

/// Outcome of probing one server.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Probe {
    pub url: String,
    pub reachable: bool,
    pub capable: bool,
    /// Model the server reported/used, when known.
    pub model: Option<String>,
    /// Human-readable explanation (shown by `doctor`).
    pub detail: String,
    /// What this particular server can do beyond answering. Everything here is `None`/false for a
    /// server that does not tell us, which is most of them: GhostReel talks to anything
    /// OpenAI-compatible, and only llama.cpp reports its own shape.
    #[serde(default, skip_serializing_if = "Capabilities::is_empty")]
    pub caps: Capabilities,
}

/// What a server admits about itself.
///
/// This exists so the app never offers a control that quietly does nothing. Describing frames
/// several at a time only pays off against a server started with matching slots, and switching
/// model or context without a restart only works against a router — neither is true of LM Studio,
/// Ollama, or a plain `llama-server`, and a setting that silently no-ops is worse than one that
/// is not there.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Requests the server will genuinely work on at once (llama.cpp `/props.total_slots`). One
    /// means concurrency buys nothing; `None` means the server did not say.
    pub slots: Option<u32>,
    /// Context each slot has, which is the whole window divided by the slots — not the number the
    /// server was started with.
    pub slot_ctx: Option<u32>,
    /// llama.cpp router mode: models can be loaded, unloaded and swapped over HTTP, so the model
    /// and its flags can change without stopping anything.
    pub router: bool,
}

impl Capabilities {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// What to append to a doctor line. Empty for a server that told us nothing, so the report
    /// stays honest about the difference between "one slot" and "did not say".
    pub fn summary(&self) -> String {
        let mut bits = Vec::new();
        if let Some(n) = self.slots {
            let ctx = self.slot_ctx.map(|c| format!(" × {c} tokens")).unwrap_or_default();
            bits.push(format!("{n} slot{}{ctx}", if n == 1 { "" } else { "s" }));
        }
        if self.router {
            bits.push("router (models switchable)".into());
        }
        if bits.is_empty() { String::new() } else { format!(", {}", bits.join(", ")) }
    }
}

impl Probe {
    fn down(url: &str, detail: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            reachable: false,
            capable: false,
            model: None,
            detail: detail.into(),
            caps: Capabilities::default(),
        }
    }
    fn incapable(url: &str, model: Option<String>, detail: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            reachable: true,
            capable: false,
            model,
            detail: detail.into(),
            caps: Capabilities::default(),
        }
    }
    fn ok(url: &str, model: Option<String>, detail: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            reachable: true,
            capable: true,
            model,
            detail: detail.into(),
            caps: Capabilities::default(),
        }
    }
    fn with_caps(mut self, caps: Capabilities) -> Self {
        self.caps = caps;
        self
    }
}

/// Ask a llama.cpp server what it can do. Everything here is best-effort: a server that does not
/// answer these endpoints is not broken, it is simply not llama.cpp, and gets the default.
async fn capabilities(client: &reqwest::Client, base: &str, models: &serde_json::Value) -> Capabilities {
    // Router mode lists models with a `status`; a plain server's entries carry only id, aliases,
    // meta and tags. That difference is the whole detection — there is no version to ask for.
    let router = models["data"]
        .as_array()
        .is_some_and(|rows| rows.iter().any(|m| m.get("status").is_some()));

    let slots = match get_json(client, &format!("{base}/props")).await {
        Ok(props) => props["total_slots"].as_u64().map(|n| n as u32),
        Err(_) => None,
    };
    // Per-slot context, which is what a request actually gets. Asking `/props` for `n_ctx` would
    // report the whole window and overstate it by the number of slots.
    let slot_ctx = match get_json(client, &format!("{base}/slots")).await {
        Ok(v) => v.as_array().and_then(|s| s.first()).and_then(|f| f["n_ctx"].as_u64()).map(|n| n as u32),
        Err(_) => None,
    };
    Capabilities { slots, slot_ctx, router }
}

/// Where a capability will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    Server,
    Local,
    /// `backend = server` but the server can't do the job.
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub backend: Backend,
    pub target: Target,
    pub probe: Option<Probe>,
    pub reason: String,
}

/// Decide where a capability runs. `probe` is `None` when it wasn't needed (`backend = local`).
pub fn resolve(backend: Backend, probe: Option<Probe>) -> Resolution {
    let (target, reason) = match (backend, &probe) {
        (Backend::Local, _) => (Target::Local, "backend = local".to_string()),
        (Backend::Cli, _) => (Target::Local, "backend = cli (handled by runtime)".to_string()),
        (_, None) => (Target::Local, "server not probed".to_string()),
        (Backend::Auto, Some(p)) if p.capable => (Target::Server, format!("server OK: {}{}", p.detail, p.caps.summary())),
        (Backend::Auto, Some(p)) => (Target::Local, format!("server not usable ({}) → local", p.detail)),
        (Backend::Server, Some(p)) if p.capable => {
            (Target::Server, format!("server OK: {}{}", p.detail, p.caps.summary()))
        }
        (Backend::Server, Some(p)) => (Target::Unavailable, format!("backend = server but {}", p.detail)),
    };
    Resolution { backend, target, probe, reason }
}

/// Short-timeout client for probes: a missing server must not stall startup.
pub fn probe_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(5))
        .build()
        .expect("static reqwest client config")
}

fn base(url: &str) -> &str {
    url.trim_end_matches('/')
}

async fn get_json(client: &reqwest::Client, url: &str) -> Result<Value, String> {
    let resp = client.get(url).send().await.map_err(short_err)?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    resp.json().await.map_err(|e| format!("invalid JSON: {e}"))
}

fn short_err(e: reqwest::Error) -> String {
    if e.is_connect() {
        "connection refused".into()
    } else if e.is_timeout() {
        "timed out".into()
    } else {
        e.to_string()
    }
}

fn first_model_id(models: &Value) -> Option<String> {
    models["data"].as_array()?.first()?["id"].as_str().map(str::to_string)
}

/// Vision/chat server (highllama / llama-server). `model` empty = the server's loaded model.
pub async fn vision(client: &reqwest::Client, url: &str, model: &str) -> Probe {
    let b = base(url);
    let models = match get_json(client, &format!("{b}/v1/models")).await {
        Ok(v) => v,
        Err(e) => return Probe::down(url, e),
    };
    let model_id = if model.is_empty() { first_model_id(&models) } else { Some(model.to_string()) };
    let props_url = match &model_id {
        Some(m) => format!("{b}/props?model={m}"),
        None => format!("{b}/props"),
    };
    let caps = capabilities(client, b, &models).await;
    match get_json(client, &props_url).await {
        Ok(props) => match props["modalities"]["vision"].as_bool() {
            Some(true) => Probe::ok(url, model_id, "model accepts images").with_caps(caps),
            Some(false) => {
                Probe::incapable(url, model_id, "loaded model has no vision (no mmproj)").with_caps(caps)
            }
            None => Probe::incapable(url, model_id, "server does not report modalities").with_caps(caps),
        },
        // Not llama.cpp (Ollama, LM Studio…): can't confirm images are supported.
        Err(e) => {
            Probe::incapable(url, model_id, format!("cannot confirm vision support (/props: {e})")).with_caps(caps)
        }
    }
}

/// Embeddings server: must return `model`-compatible vectors of [`EMBED_DIM`].
pub async fn embeddings(client: &reqwest::Client, url: &str, model: &str) -> Probe {
    let resp = client
        .post(format!("{}/v1/embeddings", base(url)))
        .json(&json!({ "input": "ghostreel probe", "model": model }))
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return Probe::down(url, short_err(e)),
    };
    let status = resp.status();
    if !status.is_success() {
        return Probe::incapable(url, None, format!("/v1/embeddings HTTP {status}"));
    }
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return Probe::incapable(url, None, format!("invalid JSON: {e}")),
    };
    let served = body["model"].as_str().map(str::to_string);
    let dim = body["data"][0]["embedding"].as_array().map(Vec::len).unwrap_or(0);
    if dim != EMBED_DIM {
        return Probe::incapable(
            url,
            served.clone(),
            format!("{dim}-dim vectors from {} (need {EMBED_DIM}-dim {model})", served.as_deref().unwrap_or("?")),
        );
    }
    let family = model.split('-').next().unwrap_or(model).to_lowercase();
    match &served {
        Some(s) if !s.to_lowercase().contains(&family) => {
            Probe::incapable(url, served.clone(), format!("served by {s}, not {model}"))
        }
        _ => Probe::ok(url, served, format!("{EMBED_DIM}-dim vectors")),
    }
}

/// GhostPen transcription server: needs timestamped segments.
pub async fn stt(client: &reqwest::Client, url: &str) -> Probe {
    let b = base(url);
    match get_json(client, &format!("{b}/v1/models")).await {
        Ok(models) => {
            let model = first_model_id(&models);
            if models["data"][0]["capabilities"]["segments"].as_bool() == Some(true) {
                Probe::ok(url, model, "timestamped segments")
            } else {
                Probe::incapable(url, model, "no timestamped segments (GhostPen too old)")
            }
        }
        Err(e) => match client.get(format!("{b}/health")).send().await {
            Ok(r) if r.status().is_success() => {
                Probe::incapable(url, None, format!("server up but /v1/models {e} (GhostPen too old for timestamps)"))
            }
            _ => Probe::down(url, e),
        },
    }
}

/// Query `{url}/v1/models` and return the list of model IDs (3 s timeout).
pub async fn server_models(url: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;
    let b = base(url);
    let body = get_json(&client, &format!("{b}/v1/models")).await?;
    let mut out = Vec::new();
    if let Some(arr) = body["data"].as_array() {
        for m in arr {
            if let Some(id) = m["id"].as_str() {
                out.push(id.to_string());
            }
        }
    } else if let Some(arr) = body.as_array() {
        for m in arr {
            if let Some(id) = m["id"].as_str() {
                out.push(id.to_string());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Minimal HTTP server: `routes` maps "METHOD /path" (query ignored) to (status, body).
    async fn serve(routes: Vec<(&'static str, u16, String)>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let mut parts = req.split_whitespace();
                    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                    let key = format!("{method} {}", target.split('?').next().unwrap_or(""));
                    let (status, body) = routes
                        .iter()
                        .find(|(k, _, _)| *k == key)
                        .map(|(_, s, b)| (*s, b.clone()))
                        .unwrap_or((404, "{}".into()));
                    let resp = format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn dead_url() -> String {
        // Bind then drop: nothing listens on this port afterwards.
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    }

    const MODELS: &str = r#"{"data":[{"id":"Bonsai-27B-Q1_0"}]}"#;

    #[tokio::test]
    async fn vision_ok_and_without_mmproj() {
        let c = probe_client();
        let url = serve(vec![
            ("GET /v1/models", 200, MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":true}}"#.into()),
        ])
        .await;
        let p = vision(&c, &url, "").await;
        assert!(p.capable, "{p:?}");
        assert_eq!(p.model.as_deref(), Some("Bonsai-27B-Q1_0"));

        let url = serve(vec![
            ("GET /v1/models", 200, MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":false}}"#.into()),
        ])
        .await;
        let p = vision(&c, &url, "").await;
        assert!(p.reachable && !p.capable);

        let p = vision(&c, &dead_url(), "").await;
        assert!(!p.reachable);
    }

    #[tokio::test]
    async fn embeddings_rejects_chat_model_vectors() {
        let c = probe_client();
        let vec768 = format!("[{}]", vec!["0.1"; 768].join(","));
        let good = format!(r#"{{"model":"embeddinggemma-300M-Q8_0","data":[{{"embedding":{vec768}}}]}}"#);
        let url = serve(vec![("POST /v1/embeddings", 200, good)]).await;
        let p = embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await;
        assert!(p.capable, "{p:?}");

        // The pre-fix highllama: Bonsai CLS-pooled, 5120-dim.
        let vec5120 = format!("[{}]", vec!["0.1"; 5120].join(","));
        let bad = format!(r#"{{"model":"Bonsai-27B-Q1_0","data":[{{"embedding":{vec5120}}}]}}"#);
        let url = serve(vec![("POST /v1/embeddings", 200, bad)]).await;
        let p = embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await;
        assert!(p.reachable && !p.capable);
        assert!(p.detail.contains("5120"), "{}", p.detail);

        let url = serve(vec![("POST /v1/embeddings", 501, "{}".into())]).await;
        assert!(!embeddings(&c, &url, "embeddinggemma-300M-Q8_0").await.capable);
    }

    #[tokio::test]
    async fn stt_needs_segments() {
        let c = probe_client();
        let url = serve(vec![(
            "GET /v1/models",
            200,
            r#"{"data":[{"id":"small","capabilities":{"segments":true}}]}"#.into(),
        )])
        .await;
        let p = stt(&c, &url).await;
        assert!(p.capable);
        assert_eq!(p.model.as_deref(), Some("small"));

        // Old GhostPen: /health only.
        let url = serve(vec![("GET /health", 200, "ok".into())]).await;
        let p = stt(&c, &url).await;
        assert!(p.reachable && !p.capable, "{p:?}");
    }

    #[test]
    fn resolve_rules() {
        let up = Probe::ok("u", None, "fine");
        let down = Probe::down("u", "connection refused");
        assert_eq!(resolve(Backend::Auto, Some(up.clone())).target, Target::Server);
        assert_eq!(resolve(Backend::Auto, Some(down.clone())).target, Target::Local);
        assert_eq!(resolve(Backend::Server, Some(down)).target, Target::Unavailable);
        assert_eq!(resolve(Backend::Local, Some(up)).target, Target::Local);
        assert_eq!(resolve(Backend::Local, None).target, Target::Local);
    }

    #[tokio::test]
    async fn server_models_lists_ids() {
        let url = serve(vec![("GET /v1/models", 200, r#"{"data":[{"id":"model-a"},{"id":"model-b"}]}"#.into())]).await;
        let list = server_models(&url).await.unwrap();
        assert_eq!(list, vec!["model-a", "model-b"]);
    }

    /// The `/models` shapes are copied from a real llama.cpp on this machine: the plain server
    /// lists id/aliases/meta/tags, and only the router adds `status`. There is no version
    /// endpoint to ask, so that difference *is* the detection.
    const PLAIN_MODELS: &str = r#"{"data":[{"id":"Qwen3.5-9B","aliases":[],"created":1,"meta":{},
        "object":"model","owned_by":"llamacpp","tags":[]}],"object":"list"}"#;
    const ROUTER_MODELS: &str = r#"{"data":[
        {"id":"chat","aliases":[],"object":"model","owned_by":"llamacpp","tags":[],
         "status":{"value":"unloaded","args":[]},
         "architecture":{"input_modalities":["text","image"]}},
        {"id":"describe","aliases":[],"object":"model","owned_by":"llamacpp","tags":[],
         "status":{"value":"running","args":[]},
         "architecture":{"input_modalities":["text","image"]}}],"object":"list"}"#;

    #[tokio::test]
    async fn a_plain_llama_server_reports_its_slots_but_not_a_router() {
        let url = serve(vec![
            ("GET /v1/models", 200, PLAIN_MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":true},"total_slots":4}"#.into()),
            ("GET /slots", 200, r#"[{"id":0,"n_ctx":16384},{"id":1,"n_ctx":16384}]"#.into()),
        ])
        .await;
        let p = vision(&reqwest::Client::new(), &url, "").await;
        assert!(p.capable);
        assert_eq!(p.caps.slots, Some(4), "four slots is four frames at a time");
        // Per slot, not the whole window: /props would say 65536 and overstate it four times.
        assert_eq!(p.caps.slot_ctx, Some(16384));
        assert!(!p.caps.router, "one model, no swapping");
    }

    #[tokio::test]
    async fn router_mode_is_recognised_by_the_status_field() {
        let url = serve(vec![
            ("GET /v1/models", 200, ROUTER_MODELS.into()),
            ("GET /props", 200, r#"{"modalities":{"vision":true},"total_slots":4}"#.into()),
            ("GET /slots", 200, r#"[{"id":0,"n_ctx":16384}]"#.into()),
        ])
        .await;
        let p = vision(&reqwest::Client::new(), &url, "").await;
        assert!(p.caps.router, "models carry a status, so this one can load and unload them");
    }

    #[tokio::test]
    async fn a_server_that_is_not_llama_cpp_claims_nothing() {
        // LM Studio, Ollama, a hosted endpoint: answers /v1/models and nothing else here. The
        // app must not offer slots or model-switching on the strength of a guess.
        let url = serve(vec![("GET /v1/models", 200, PLAIN_MODELS.into())]).await;
        let p = vision(&reqwest::Client::new(), &url, "").await;
        assert!(p.reachable && !p.capable, "vision cannot be confirmed without /props");
        assert!(p.caps.is_empty(), "no slots, no router, no claims: {:?}", p.caps);
    }
}
