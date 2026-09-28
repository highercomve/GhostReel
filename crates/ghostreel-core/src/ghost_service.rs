//! The Ghost model service shared by GhostPen: when GhostPen is running it already holds an LLM
//! in VRAM, so two apps need not each load one. GhostPen writes a small discovery file
//! (`models.json`); this module reads it. The service speaks the same OpenAI-compatible API the
//! configured servers do, so the ordinary [`crate::probe`]s validate it — only discovery is new.
//! Everything here degrades silently: no file, a stale pid, an opt-out or a failed probe all mean
//! "behave exactly as if the service did not exist".

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{Backend, Config};
use crate::probe::{self, Probe, Resolution, Target};

/// One advertised GhostPen model service (the contents of `models.json`).
#[derive(Debug, Clone, Serialize)]
pub struct GhostService {
    /// File this was read from.
    pub path: PathBuf,
    pub app: String,
    pub pid: Option<u32>,
    pub url: String,
    pub capabilities: ServiceCaps,
    pub models: ServiceModels,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServiceCaps {
    pub chat: bool,
    pub vision: bool,
    pub embeddings: bool,
    pub stt: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServiceModels {
    pub chat: String,
    pub embeddings: String,
    pub stt: String,
}

/// The file as written. Unknown fields are ignored so a newer GhostPen never breaks an older
/// GhostReel; everything but `url` may be absent.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ServiceFile {
    app: String,
    pid: Option<u32>,
    url: String,
    capabilities: ServiceCaps,
    models: ServiceModels,
}

/// Where the discovery file may live, in order; the first that exists wins.
pub fn candidate_paths() -> Vec<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        dirs::home_dir().map(|h| h.join("Library/Application Support/Ghost/models.json")).into_iter().collect()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mut out = Vec::new();
        if let Some(rt) = std::env::var_os("XDG_RUNTIME_DIR") {
            out.push(PathBuf::from(rt).join("ghost/models.json"));
        }
        if let Some(home) = dirs::home_dir() {
            out.push(home.join(".cache/ghost/models.json"));
        }
        out
    }
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join(r"Ghost\models.json")).into_iter().collect()
    }
}

/// Parse one discovery file. An empty or non-HTTP url means the file is worthless.
fn parse(path: &Path, text: &str) -> Option<GhostService> {
    let f: ServiceFile = serde_json::from_str(text).ok()?;
    let url = f.url.trim().trim_end_matches('/');
    if !url.starts_with("http") {
        return None;
    }
    Some(GhostService {
        path: path.to_path_buf(),
        app: f.app,
        pid: f.pid,
        url: url.to_string(),
        capabilities: f.capabilities,
        models: f.models,
    })
}

#[cfg(target_os = "linux")]
fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// EPERM means the process exists but belongs to somebody else — still alive.
#[cfg(all(unix, not(target_os = "linux")))]
fn pid_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM) }
}

/// No cheap liveness check; the HTTP probe decides.
#[cfg(windows)]
fn pid_alive(_pid: u32) -> bool {
    true
}

/// Testable core of [`discover`]: the first path that exists decides. If it is unreadable,
/// unparseable or names a dead pid the service is stale — return `None` rather than falling to
/// the next path, because a newer file at a lower-priority path is not a healthier service.
pub fn discover_in(paths: &[PathBuf], alive: impl Fn(u32) -> bool) -> Option<GhostService> {
    let path = paths.iter().find(|p| p.exists())?;
    let svc = parse(path, &std::fs::read_to_string(path).ok()?)?;
    match svc.pid {
        Some(pid) if !alive(pid) => None,
        _ => Some(svc),
    }
}

/// The user asked us to leave the service alone (`ai.use_ghost_service = false` in the config,
/// or `GHOSTREEL_NO_GHOST_SERVICE` in the environment). A pure helper so tests need not touch
/// process-global env vars.
pub fn opted_out(cfg: &Config, env: Option<&str>) -> bool {
    !cfg.ai.use_ghost_service
        || env.is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

/// Discover the running service, honouring the opt-outs. The result is probed before use.
pub fn discover(config: &Config) -> Option<GhostService> {
    if opted_out(config, std::env::var("GHOSTREEL_NO_GHOST_SERVICE").ok().as_deref()) {
        return None;
    }
    discover_in(&candidate_paths(), pid_alive)
}

/// One AI capability the service might cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Vision,
    Chat,
    Embeddings,
    Stt,
}

/// Probe the service for `cap`: `Some` only when it advertises the capability *and* the probe
/// confirms it can do the job. `embed_model` is GhostReel's configured embedding model: the
/// service may name its own, and only a served model of the same family is accepted — a
/// different 768-dim model would silently corrupt the search index.
pub async fn probe_service(
    client: &reqwest::Client,
    svc: &GhostService,
    cap: Capability,
    embed_model: &str,
) -> Option<Probe> {
    let p = match cap {
        // Chat and vision are the same server shape (an image-capable chat model); the service
        // advertises them separately because GhostPen may load a text-only brain.
        Capability::Vision if svc.capabilities.vision => probe::vision(client, &svc.url, &svc.models.chat).await,
        Capability::Chat if svc.capabilities.chat => probe::vision(client, &svc.url, &svc.models.chat).await,
        Capability::Embeddings if svc.capabilities.embeddings => {
            let requested = if svc.models.embeddings.is_empty() { embed_model } else { svc.models.embeddings.as_str() };
            let p = probe::embeddings(client, &svc.url, requested).await;
            let served = p.model.as_deref().unwrap_or(requested);
            let family = embed_model.split('-').next().unwrap_or(embed_model).to_lowercase();
            if !served.to_lowercase().contains(&family) {
                return None;
            }
            p
        }
        Capability::Stt if svc.capabilities.stt => probe::stt(client, &svc.url).await,
        _ => return None,
    };
    p.capable.then_some(p)
}

/// The resolution `doctor` and `runtime` share when the service answers for a capability, so
/// both always tell the same story.
pub fn service_resolution(backend: Backend, p: Probe) -> Resolution {
    let reason = format!("GhostPen shared service: {}{}", p.detail, p.caps.summary());
    Resolution { backend, target: Target::Server, probe: Some(p), reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"{
        "version": 1,
        "app": "GhostPen",
        "pid": 12345,
        "url": "http://127.0.0.1:8771",
        "capabilities": {"chat": true, "vision": true, "embeddings": true, "stt": true},
        "models": {"chat": "qwen3.5-9b", "embeddings": "embeddinggemma-300m", "stt": "large-v3-turbo-q5_0"},
        "updated": 1790640000
    }"#;

    #[test]
    fn parses_the_full_example() {
        let svc = parse(Path::new("models.json"), EXAMPLE).unwrap();
        assert_eq!(svc.app, "GhostPen");
        assert_eq!(svc.pid, Some(12345));
        assert_eq!(svc.url, "http://127.0.0.1:8771");
        assert!(
            svc.capabilities.chat && svc.capabilities.vision && svc.capabilities.embeddings && svc.capabilities.stt
        );
        assert_eq!(svc.models.chat, "qwen3.5-9b");
        assert_eq!(svc.models.embeddings, "embeddinggemma-300m");
        assert_eq!(svc.models.stt, "large-v3-turbo-q5_0");
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let svc = parse(Path::new("f"), r#"{"url":"http://x:1","future":{"nested":[1,2]},"extra":"yes"}"#).unwrap();
        assert_eq!(svc.url, "http://x:1");
        assert_eq!(svc.app, "");
        assert!(!svc.capabilities.chat, "absent capabilities default to off");
    }

    #[test]
    fn missing_or_invalid_url_is_no_service() {
        assert!(parse(Path::new("f"), r#"{"app":"GhostPen"}"#).is_none());
        assert!(parse(Path::new("f"), r#"{"url":""}"#).is_none());
        assert!(parse(Path::new("f"), r#"{"url":"  "}"#).is_none());
        assert!(parse(Path::new("f"), r#"{"url":"127.0.0.1:8771"}"#).is_none());
    }

    #[test]
    fn invalid_json_is_no_service() {
        assert!(parse(Path::new("f"), "not json").is_none());
        assert!(parse(Path::new("f"), "").is_none());
    }

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn dead_pid_means_stale_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(dir.path(), "models.json", EXAMPLE);
        assert!(discover_in(std::slice::from_ref(&p), |_| false).is_none());
        // The same file with a live pid is fine.
        assert!(discover_in(&[p], |_| true).is_some());
    }

    #[test]
    fn no_pid_means_the_probe_decides() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(dir.path(), "models.json", r#"{"url":"http://127.0.0.1:8771"}"#);
        assert!(discover_in(&[p], |_| panic!("no pid to check")).is_some());
    }

    #[test]
    fn first_existing_path_wins_and_a_stale_first_file_hides_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let stale = write(dir.path(), "stale.json", EXAMPLE); // pid 12345
        let good = write(dir.path(), "good.json", r#"{"url":"http://127.0.0.1:8771"}"#);
        // First exists but its pid is dead: None, not the second file.
        assert!(discover_in(&[stale.clone(), good.clone()], |_| false).is_none());
        // First missing: the second decides.
        let svc = discover_in(&[dir.path().join("missing.json"), good], |_| false).unwrap();
        assert_eq!(svc.url, "http://127.0.0.1:8771");
        // Nothing exists: nothing found.
        assert!(discover_in(&[dir.path().join("nope.json")], |_| true).is_none());
        assert!(discover_in(&[], |_| true).is_none());
    }

    #[test]
    fn opt_outs() {
        let mut cfg = Config::default();
        assert!(cfg.ai.use_ghost_service, "on by default");
        assert!(!opted_out(&cfg, None));
        assert!(!opted_out(&cfg, Some("0")));
        assert!(!opted_out(&cfg, Some("")));
        for v in ["1", "true", "TRUE", "yes", "on"] {
            assert!(opted_out(&cfg, Some(v)), "{v}");
        }
        cfg.ai.use_ghost_service = false;
        assert!(opted_out(&cfg, None));
    }

    #[tokio::test]
    async fn service_embeddings_must_match_the_configured_family() {
        let vec768 = format!("[{}]", vec!["0.1"; 768].join(","));
        let emb_server = |model: &str| {
            let body = format!(r#"{{"model":"{model}","data":[{{"embedding":{vec768}}}]}}"#);
            crate::probe::tests_support::serve(vec![("POST /v1/embeddings", 200, body)])
        };
        let mut svc =
            parse(Path::new("f"), &format!(r#"{{"url":"{}"}}"#, emb_server("embeddinggemma-300m").await)).unwrap();
        let c = probe::probe_client();

        // Advertised embeddings, same family as the configured model: accepted.
        svc.capabilities.embeddings = true;
        svc.models.embeddings = "embeddinggemma-300m".into();
        let p = probe_service(&c, &svc, Capability::Embeddings, crate::config::EMBED_MODEL).await;
        assert!(p.is_some_and(|p| p.capable));

        // A different 768-dim model must never be used silently, even though it passes the
        // probe's own dim and name checks against the requested model.
        svc.url = emb_server("someother-embed").await;
        svc.models.embeddings = "someother-embed".into();
        assert!(probe_service(&c, &svc, Capability::Embeddings, crate::config::EMBED_MODEL).await.is_none());

        // Not advertised: never probed at all.
        svc.models.embeddings = "embeddinggemma-300m".into();
        svc.capabilities.embeddings = false;
        assert!(probe_service(&c, &svc, Capability::Embeddings, crate::config::EMBED_MODEL).await.is_none());
    }
}
