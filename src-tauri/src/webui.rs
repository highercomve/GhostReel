//! The desktop UI served over HTTP, so a phone or another computer can follow and drive the work.
//!
//! The browser gets the same React bundle the window loads, read from Tauri's embedded assets.
//! What the window reaches through `invoke` it reaches through `POST /api/call`, queue updates
//! arrive as server-sent events on `/api/events`, and media comes from `/media`. Off unless it is
//! enabled in Settings; HTTP Basic auth is optional.

use std::convert::Infallible;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine;
use futures_util::stream::{self, Stream};
use ghostreel_core::config::WebConfig;
use ghostreel_core::db::Db;
use ghostreel_core::paths::Paths;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Listener, Manager};
use tokio::sync::{broadcast, watch};
use tower::ServiceExt;

/// What Settings shows about the server.
#[derive(Serialize)]
pub struct WebStatus {
    pub enabled: bool,
    pub running: bool,
    pub bind: String,
    pub port: u16,
    /// Addresses to open the UI at from another device.
    pub urls: Vec<String>,
    pub auth_enabled: bool,
    pub auth_user: String,
    /// Why the server is not running although it is enabled.
    pub error: Option<String>,
}

/// A change to the web settings. Absent fields keep their value, and so does an empty password:
/// it is never sent back to the UI, so the form has nothing to fill it with.
#[derive(Deserialize)]
pub struct WebConfigPatch {
    enabled: Option<bool>,
    bind: Option<String>,
    port: Option<u16>,
    auth_enabled: Option<bool>,
    auth_user: Option<String>,
    auth_password: Option<String>,
}

struct Running {
    addr: SocketAddr,
    stop: watch::Sender<bool>,
}

/// The server's place in the app's state.
pub struct WebServer {
    running: Mutex<Option<Running>>,
    error: Mutex<Option<String>>,
    /// Queue events, relayed to every connected browser.
    events: broadcast::Sender<(&'static str, String)>,
}

impl Default for WebServer {
    fn default() -> Self {
        let (events, _) = broadcast::channel(64);
        Self { running: Mutex::new(None), error: Mutex::new(None), events }
    }
}

#[derive(Clone)]
struct Ctx {
    app: AppHandle,
    events: broadcast::Sender<(&'static str, String)>,
    /// Flips when the server stops, which also ends the open event streams: a graceful
    /// shutdown would otherwise wait on them forever.
    stop: watch::Receiver<bool>,
    auth: Option<Arc<(String, String)>>,
}

/// Relays the queue events the window listens to, so browsers get them too.
pub fn forward_events(app: &AppHandle) {
    let events = app.state::<WebServer>().events.clone();
    for name in ["queue", "task-finished", "chat-progress"] {
        let events = events.clone();
        app.listen(name, move |event| {
            let _ = events.send((name, event.payload().to_string()));
        });
    }
}

/// Applies a settings change, refusing it whole rather than storing half of it.
pub fn merge(cfg: &mut WebConfig, patch: WebConfigPatch) -> Result<(), String> {
    let mut next = cfg.clone();
    if let Some(enabled) = patch.enabled {
        next.enabled = enabled;
    }
    if let Some(bind) = patch.bind {
        let bind = bind.trim().to_string();
        bind.parse::<IpAddr>().map_err(|_| format!("'{bind}' is not an IP address"))?;
        next.bind = bind;
    }
    if let Some(port) = patch.port {
        if port == 0 {
            return Err("the port must be between 1 and 65535".to_string());
        }
        next.port = port;
    }
    if let Some(auth_enabled) = patch.auth_enabled {
        next.auth_enabled = auth_enabled;
    }
    if let Some(user) = patch.auth_user {
        next.auth_user = user.trim().to_string();
    }
    if let Some(password) = patch.auth_password.filter(|p| !p.is_empty()) {
        next.auth_password = password;
    }
    check_auth(&next)?;
    *cfg = next;
    Ok(())
}

fn check_auth(cfg: &WebConfig) -> Result<(), String> {
    if cfg.auth_enabled && (cfg.auth_user.is_empty() || cfg.auth_password.is_empty()) {
        return Err("basic auth needs a username and a password".to_string());
    }
    Ok(())
}

/// Brings the server in line with the settings: stops it, and starts it again if enabled.
/// A failure to start is kept for Settings to show.
pub async fn apply(app: &AppHandle, cfg: &WebConfig) {
    let server = app.state::<WebServer>();
    if let Some(running) = server.running.lock().unwrap().take() {
        let _ = running.stop.send(true);
    }
    *server.error.lock().unwrap() = None;
    if !cfg.enabled {
        return;
    }
    match start(app, cfg, server.events.clone()).await {
        Ok(running) => *server.running.lock().unwrap() = Some(running),
        Err(e) => *server.error.lock().unwrap() = Some(e),
    }
}

pub fn status(server: &WebServer, cfg: &WebConfig) -> WebStatus {
    let addr = server.running.lock().unwrap().as_ref().map(|r| r.addr);
    WebStatus {
        enabled: cfg.enabled,
        running: addr.is_some(),
        bind: cfg.bind.clone(),
        port: addr.map_or(cfg.port, |a| a.port()),
        urls: addr.map(urls_for).unwrap_or_default(),
        auth_enabled: cfg.auth_enabled,
        auth_user: cfg.auth_user.clone(),
        error: server.error.lock().unwrap().clone(),
    }
}

async fn start(
    app: &AppHandle,
    cfg: &WebConfig,
    events: broadcast::Sender<(&'static str, String)>,
) -> Result<Running, String> {
    let ip: IpAddr = cfg.bind.trim().parse().map_err(|_| format!("'{}' is not an IP address", cfg.bind))?;
    check_auth(cfg)?;
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(ip, cfg.port))
        .await
        .map_err(|e| format!("cannot listen on {}: {e}", SocketAddr::new(ip, cfg.port)))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;

    let (stop, stop_rx) = watch::channel(false);
    let ctx = Ctx {
        app: app.clone(),
        events,
        stop: stop_rx.clone(),
        auth: cfg.auth_enabled.then(|| Arc::new((cfg.auth_user.clone(), cfg.auth_password.clone()))),
    };
    let router = Router::new()
        .route("/api/call", post(call))
        .route("/api/events", get(events_stream))
        .route("/media", get(media))
        .fallback(asset)
        .layer(middleware::from_fn_with_state(ctx.clone(), basic_auth))
        .with_state(ctx);

    let mut shutdown = stop_rx;
    tauri::async_runtime::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown.changed().await;
            })
            .await;
    });
    Ok(Running { addr, stop })
}

/// Where the UI can be opened from. For a wildcard bind that is this machine plus the addresses
/// it routes to the LAN and to a Tailscale tailnet, when it has them.
fn urls_for(addr: SocketAddr) -> Vec<String> {
    let port = addr.port();
    if !addr.ip().is_unspecified() {
        return vec![format!("http://{addr}")];
    }
    let mut hosts: Vec<IpAddr> = vec![IpAddr::from([127, 0, 0, 1])];
    // Connecting a UDP socket sends nothing; it only asks which local address would be used.
    for probe in ["192.0.2.1:9", "100.100.100.100:9"] {
        let local = UdpSocket::bind("0.0.0.0:0").and_then(|s| s.connect(probe).map(|_| s)).and_then(|s| s.local_addr());
        if let Ok(local) = local
            && !hosts.contains(&local.ip())
        {
            hosts.push(local.ip());
        }
    }
    hosts.into_iter().map(|ip| format!("http://{}", SocketAddr::new(ip, port))).collect()
}

async fn basic_auth(State(ctx): State<Ctx>, req: Request, next: Next) -> Response {
    let Some(creds) = &ctx.auth else {
        return next.run(req).await;
    };
    let header = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok());
    if authorized(header, &creds.0, &creds.1) {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Basic realm=\"GhostReel\", charset=\"UTF-8\"")],
        "Sign in to GhostReel",
    )
        .into_response()
}

fn authorized(header: Option<&str>, user: &str, password: &str) -> bool {
    let given = header
        .and_then(|v| v.strip_prefix("Basic "))
        .and_then(|b| base64::engine::general_purpose::STANDARD.decode(b.trim()).ok());
    given.is_some_and(|given| constant_time_eq(&given, format!("{user}:{password}").as_bytes()))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The bundled frontend, falling back to `index.html` so the app's own routes load.
async fn asset(State(ctx): State<Ctx>, req: Request) -> Response {
    let path = req.uri().path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let resolver = ctx.app.asset_resolver();
    match resolver.get(path.to_string()).or_else(|| resolver.get("index.html".to_string())) {
        Some(asset) => ([(header::CONTENT_TYPE, asset.mime_type)], asset.bytes).into_response(),
        // `tauri dev` loads the UI from the Vite dev server, so there is nothing embedded to serve.
        None => {
            (StatusCode::NOT_FOUND, "No bundled UI in this build; open the Vite dev server instead.").into_response()
        }
    }
}

#[derive(Deserialize)]
struct MediaQuery {
    path: PathBuf,
}

/// What the window's asset protocol can load: anything under the data dir, and the watched
/// folders.
fn media_allowed(path: &Path) -> bool {
    let Ok(canonical) = std::fs::canonicalize(path) else { return false };
    if !canonical.is_file() {
        return false;
    }
    let Ok(paths) = Paths::resolve() else { return false };
    if std::fs::canonicalize(&paths.data_dir).is_ok_and(|root| canonical.starts_with(root)) {
        return true;
    }
    let Ok(db) = Db::open(&paths.db_file()) else { return false };
    db.folders(None)
        .unwrap_or_default()
        .iter()
        .any(|f| std::fs::canonicalize(&f.path).is_ok_and(|root| canonical.starts_with(root)))
}

async fn media(Query(q): Query<MediaQuery>, req: Request) -> Response {
    let path = q.path.clone();
    if !tokio::task::spawn_blocking(move || media_allowed(&path)).await.unwrap_or(false) {
        return StatusCode::NOT_FOUND.into_response();
    }
    // ServeFile handles Range/If-Range/HEAD, which seeking in a <video> needs.
    match tower_http::services::ServeFile::new(&q.path).oneshot(req).await {
        Ok(r) => r.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn events_stream(State(ctx): State<Ctx>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = (ctx.events.subscribe(), ctx.stop.clone());
    let stream = stream::unfold(state, |(mut events, mut stop)| async move {
        loop {
            tokio::select! {
                _ = stop.changed() => return None,
                next = events.recv() => match next {
                    Ok((name, data)) => {
                        return Some((Ok(Event::default().event(name).data(data)), (events, stop)));
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                },
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct CallBody {
    cmd: String,
    #[serde(default)]
    args: Value,
}

async fn call(State(ctx): State<Ctx>, body: Bytes) -> Response {
    let body: CallBody = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return failure(format!("bad request: {e}")),
    };
    let args = if body.args.is_null() { Value::Object(Default::default()) } else { body.args };
    match dispatch(&ctx.app, &body.cmd, args).await {
        Ok(value) => (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], value.to_string()).into_response(),
        Err(e) => failure(e),
    }
}

fn failure(message: String) -> Response {
    let body = serde_json::json!({ "error": message }).to_string();
    (StatusCode::BAD_REQUEST, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

fn parse<T: DeserializeOwned>(args: Value) -> Result<T, String> {
    serde_json::from_value(args).map_err(|e| format!("invalid arguments: {e}"))
}

fn ok<T: Serialize>(value: T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|e| e.to_string())
}

/// Reads a command's arguments the way Tauri does: camelCase keys, `Option` ones may be left out.
macro_rules! args {
    ($args:expr; $($field:ident : $ty:ty),* $(,)?) => {{
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Args { $($field: $ty,)* }
        let Args { $($field,)* } = parse::<Args>($args)?;
        ($($field,)*)
    }};
}

/// Runs a desktop command for a browser. Each arm calls the same function the window's `invoke`
/// reaches.
async fn dispatch(app: &AppHandle, cmd: &str, args: Value) -> Result<Value, String> {
    use crate::queue::Queue;
    use ghostreel_core::projects::PipelineConfig;
    use ghostreel_core::script::Script;

    let queue = || app.state::<Queue>();
    match cmd {
        "doctor" => ok(crate::doctor().await?),
        "app_version" => ok(crate::app_version()),
        "list_projects" => ok(crate::list_projects()?),
        "create_project" => {
            let (name, fps_num, fps_den, width, height, pipeline) = args!(args;
                name: String, fps_num: i64, fps_den: i64, width: i64, height: i64,
                pipeline: Option<PipelineConfig>);
            ok(crate::create_project(name, fps_num, fps_den, width, height, pipeline)?)
        }
        "set_project_pipeline" => {
            let (project_id, pipeline) = args!(args; project_id: i64, pipeline: PipelineConfig);
            ok(crate::set_project_pipeline(project_id, pipeline)?)
        }
        "rename_project" => {
            let (project_id, name) = args!(args; project_id: i64, name: String);
            ok(crate::rename_project(project_id, name)?)
        }
        "remove_project" => {
            let (project_id, purge) = args!(args; project_id: i64, purge: bool);
            ok(crate::remove_project(project_id, purge)?)
        }
        "project_view" => {
            let (project_id,) = args!(args; project_id: i64);
            ok(crate::project_view(project_id)?)
        }
        "exclude_video" => {
            let (project_id, video_id, role) = args!(args; project_id: i64, video_id: i64, role: String);
            ok(crate::exclude_video(project_id, video_id, role)?)
        }
        "include_video" => {
            let (project_id, video_id) = args!(args; project_id: i64, video_id: i64);
            ok(crate::include_video(project_id, video_id)?)
        }
        "add_folder" => {
            let (project_id, path, recursive) = args!(args; project_id: i64, path: PathBuf, recursive: bool);
            ok(crate::add_folder(app.clone(), project_id, path, recursive)?)
        }
        "remove_folder" => {
            let (project_id, path) = args!(args; project_id: i64, path: PathBuf);
            ok(crate::remove_folder(project_id, path)?)
        }
        "search" => {
            let (project_id, query, limit) = args!(args; project_id: i64, query: String, limit: Option<usize>);
            ok(crate::search(app.state::<crate::SearchState>(), project_id, query, limit).await?)
        }
        // The window's media server listens on loopback only; a browser uses this server's.
        "media_base" => ok("/media"),
        "cli_models" => {
            let (tool,) = args!(args; tool: String);
            ok(crate::cli_models(tool).await?)
        }
        "open_external" => {
            let (path, t) = args!(args; path: PathBuf, t: Option<f64>);
            ok(crate::open_external(app.clone(), path, t)?)
        }
        "enqueue_index" => {
            let (project_id,) = args!(args; project_id: i64);
            ok(crate::enqueue_index(app.clone(), queue(), project_id).await?)
        }
        "enqueue_steadiness" => {
            let (project_id, force) = args!(args; project_id: i64, force: Option<bool>);
            ok(crate::enqueue_steadiness(app.clone(), queue(), project_id, force).await?)
        }
        "redo_project_stage" => {
            let (project_id, stage) = args!(args; project_id: i64, stage: String);
            ok(crate::redo_project_stage(app.clone(), queue(), project_id, stage).await?)
        }
        "get_chat_settings" => ok(crate::get_chat_settings()?),
        "set_chat_system_prompt" => {
            let (prompt,) = args!(args; prompt: String);
            ok(crate::set_chat_system_prompt(prompt)?)
        }
        "enqueue_preview" => {
            let (script_id, burn_titles, burn_narration, normalize_audio, out) = args!(args;
                script_id: i64, burn_titles: bool, burn_narration: bool, normalize_audio: bool,
                out: Option<String>);
            ok(crate::enqueue_preview(
                app.clone(),
                queue(),
                script_id,
                burn_titles,
                burn_narration,
                normalize_audio,
                out,
            )
            .await?)
        }
        "enqueue_export" => {
            let (script_id, format, path) = args!(args; script_id: i64, format: String, path: String);
            ok(crate::enqueue_export(app.clone(), queue(), script_id, format, path).await?)
        }
        "preview_plan" => {
            let (script_id,) = args!(args; script_id: i64);
            ok(crate::preview_plan(script_id)?)
        }
        "get_script_preview" => {
            let (script_id,) = args!(args; script_id: i64);
            ok(crate::get_script_preview(script_id)?)
        }
        "queue_list" => ok(crate::queue_list(queue()).await?),
        "cancel_task" => {
            let (id,) = args!(args; id: u64);
            ok(crate::cancel_task(app.clone(), queue(), id).await?)
        }
        "clear_finished_tasks" => ok(crate::clear_finished_tasks(app.clone(), queue()).await?),
        "models_status" => ok(crate::models_status()?),
        "build_script_with_jev" => {
            let (project_id, session_id, brief, target_s) = args!(args;
                project_id: i64, session_id: Option<i64>, brief: String, target_s: f64);
            ok(crate::build_script_with_jev(project_id, session_id, brief, target_s).await?)
        }
        "get_ai_settings" => ok(crate::get_ai_settings().await?),
        "set_ai_settings" => {
            let (patch,) = args!(args; patch: crate::AiSettingsPatch);
            ok(crate::set_ai_settings(patch, app.state::<crate::SearchState>()).await?)
        }
        "test_cli_agent" => {
            let (capability,) = args!(args; capability: String);
            ok(crate::test_cli_agent(capability).await?)
        }
        "server_models" => {
            let (url,) = args!(args; url: String);
            ok(crate::server_models(url).await?)
        }
        "probe_backends" => ok(crate::probe_backends().await?),
        "enqueue_model_download" => {
            let (model_id,) = args!(args; model_id: String);
            ok(crate::enqueue_model_download(app.clone(), queue(), model_id).await?)
        }
        "remove_model" => {
            let (model_id,) = args!(args; model_id: String);
            ok(crate::remove_model(model_id)?)
        }
        "set_whisper_model" => {
            let (model_id,) = args!(args; model_id: String);
            ok(crate::set_whisper_model(model_id)?)
        }
        "open_models_dir" => ok(crate::open_models_dir(app.clone())?),
        "video_transcript" => {
            let (video_id,) = args!(args; video_id: i64);
            ok(crate::video_transcript(video_id)?)
        }
        "video_steadiness" => {
            let (video_id,) = args!(args; video_id: i64);
            ok(crate::video_steadiness(video_id)?)
        }
        "video_playback" => {
            let (video_id,) = args!(args; video_id: i64);
            ok(crate::video_playback(video_id).await?)
        }
        "video_frames" => {
            let (video_id,) = args!(args; video_id: i64);
            ok(crate::video_frames(video_id)?)
        }
        "chat_turn" => {
            let (project_id, session_id, message, images) = args!(args;
                project_id: i64, session_id: Option<i64>, message: String, images: Option<Vec<String>>);
            ok(crate::chat_turn(app.clone(), queue(), project_id, session_id, message, images).await?)
        }
        "chat_sessions" => {
            let (project_id,) = args!(args; project_id: i64);
            ok(crate::chat_sessions(project_id)?)
        }
        "delete_chat_session" => {
            let (session_id,) = args!(args; session_id: i64);
            ok(crate::delete_chat_session(session_id)?)
        }
        "chat_messages" => {
            let (session_id,) = args!(args; session_id: i64);
            ok(crate::chat_messages(session_id)?)
        }
        "list_scripts" => {
            let (project_id,) = args!(args; project_id: i64);
            ok(crate::list_scripts(project_id)?)
        }
        "get_script" => {
            let (script_id,) = args!(args; script_id: i64);
            ok(crate::get_script(script_id)?)
        }
        "save_script" => {
            let (project_id, script, session_id) = args!(args;
                project_id: i64, script: Script, session_id: Option<i64>);
            ok(crate::save_script(project_id, script, session_id)?)
        }
        "web_status" => ok(crate::web_status(app.state::<WebServer>())?),
        // Changing who can reach the server is left to someone at the machine.
        "set_web_config" => Err("web access settings can only be changed in the desktop app".to_string()),
        other => Err(format!("unknown command '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch() -> WebConfigPatch {
        WebConfigPatch {
            enabled: None,
            bind: None,
            port: None,
            auth_enabled: None,
            auth_user: None,
            auth_password: None,
        }
    }

    fn basic(pair: &str) -> String {
        format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(pair))
    }

    #[test]
    fn off_by_default_and_without_auth() {
        let cfg = WebConfig::default();
        assert!(!cfg.enabled);
        assert!(!cfg.auth_enabled);
    }

    #[test]
    fn a_rejected_change_leaves_everything_as_it_was() {
        let mut cfg = WebConfig::default();
        let before = cfg.clone();
        let bad = WebConfigPatch { port: Some(9000), bind: Some("not-an-ip".into()), ..patch() };
        assert!(merge(&mut cfg, bad).is_err());
        assert_eq!(cfg, before, "the valid port must not be applied either");

        assert!(merge(&mut cfg, WebConfigPatch { port: Some(0), ..patch() }).is_err());
        assert_eq!(cfg, before);
    }

    #[test]
    fn auth_needs_both_a_user_and_a_password() {
        let mut cfg = WebConfig::default();
        let only_user = WebConfigPatch { auth_enabled: Some(true), auth_user: Some("me".into()), ..patch() };
        assert!(merge(&mut cfg, only_user).is_err());
        assert!(!cfg.auth_enabled);

        let both = WebConfigPatch {
            auth_enabled: Some(true),
            auth_user: Some(" me ".into()),
            auth_password: Some("secret".into()),
            ..patch()
        };
        merge(&mut cfg, both).unwrap();
        assert!(cfg.auth_enabled);
        assert_eq!(cfg.auth_user, "me");
    }

    #[test]
    fn an_empty_password_keeps_the_stored_one() {
        let mut cfg = WebConfig { auth_password: "secret".into(), ..WebConfig::default() };
        merge(&mut cfg, WebConfigPatch { auth_password: Some(String::new()), port: Some(8080), ..patch() }).unwrap();
        assert_eq!(cfg.auth_password, "secret");
        assert_eq!(cfg.port, 8080);
    }

    #[test]
    fn basic_auth_lets_in_only_the_exact_pair() {
        assert!(authorized(Some(&basic("me:secret")), "me", "secret"));
        assert!(!authorized(Some(&basic("me:wrong")), "me", "secret"));
        assert!(!authorized(Some(&basic("me:secret2")), "me", "secret"));
        assert!(!authorized(Some(&basic("you:secret")), "me", "secret"));
        assert!(!authorized(Some("Bearer me:secret"), "me", "secret"));
        assert!(!authorized(Some("Basic not base64!"), "me", "secret"));
        assert!(!authorized(None, "me", "secret"));
    }

    #[test]
    fn a_specific_bind_is_the_only_url() {
        let addr: SocketAddr = "192.168.1.5:4317".parse().unwrap();
        assert_eq!(urls_for(addr), vec!["http://192.168.1.5:4317"]);
    }

    #[test]
    fn a_wildcard_bind_lists_loopback_first_and_no_duplicates() {
        let urls = urls_for("0.0.0.0:4317".parse().unwrap());
        assert_eq!(urls[0], "http://127.0.0.1:4317");
        let mut unique = urls.clone();
        unique.dedup();
        assert_eq!(unique.len(), urls.len());
    }
}
