//! Local media server for the player.
//!
//! WebKitGTK can't stream `<video>` from Tauri's asset protocol (no byte ranges), so videos are
//! served from `http://127.0.0.1:<random port>/<token>/media?path=…` with Range support (seeking).
//! Only files inside watched folders are served, and the random token keeps other local processes
//! and web pages from reading through it.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use ghostreel_core::db::Db;
use ghostreel_core::paths::Paths;
use serde::Deserialize;
use tower::ServiceExt;

#[derive(Clone)]
pub struct MediaServer {
    pub base: String,
}

#[derive(Clone)]
struct AppState {
    token: String,
}

#[derive(Deserialize)]
struct MediaQuery {
    path: PathBuf,
}

fn random_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    let a = h.finish();
    let mut h2 = RandomState::new().build_hasher();
    h2.write_u64(a ^ std::process::id() as u64);
    format!("{a:016x}{:016x}", h2.finish())
}

/// A path may be served when it's a file inside one of the watched folders,
/// or a preview / proxy file under the data dir.
fn allowed(path: &std::path::Path) -> bool {
    let Ok(canonical) = std::fs::canonicalize(path) else { return false };
    if !canonical.is_file() {
        return false;
    }
    let Ok(paths) = Paths::resolve() else { return false };
    for name in ["previews", "proxies", "frames", "chat_images"] {
        if std::fs::canonicalize(paths.data_dir.join(name)).is_ok_and(|root| canonical.starts_with(root)) {
            return true;
        }
    }
    let Ok(db) = Db::open(&paths.db_file()) else { return false };
    let folders = db.folders(None).unwrap_or_default();
    folders.iter().any(|f| std::fs::canonicalize(&f.path).is_ok_and(|root| canonical.starts_with(root)))
}

async fn serve(
    State(state): State<AppState>,
    Path(token): Path<String>,
    Query(q): Query<MediaQuery>,
    req: Request,
) -> Response {
    if token != state.token {
        return StatusCode::FORBIDDEN.into_response();
    }
    let path = q.path.clone();
    let ok = tokio::task::spawn_blocking(move || allowed(&path)).await.unwrap_or(false);
    if !ok {
        return StatusCode::NOT_FOUND.into_response();
    }
    // ServeFile handles Range/If-Range/HEAD and content types.
    match tower_http::services::ServeFile::new(&q.path).oneshot(req).await {
        Ok(r) => r.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

pub async fn start() -> Result<MediaServer, String> {
    let token = random_token();
    let app = Router::new().route("/{token}/media", get(serve)).with_state(AppState { token: token.clone() });
    let listener =
        tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(MediaServer { base: format!("http://{addr}/{token}/media") })
}
