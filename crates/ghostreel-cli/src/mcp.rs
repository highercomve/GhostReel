//! MCP server (`ghostreel mcp`): the tool registry (`ghostreel_core::tools`) over stdio.
//!
//! This file is only the transport: `tools/list` renders the registry and `tools/call` runs one
//! of its tools. The same registry is what the script chat and the clip workers call, so an
//! outside agent has exactly the tools GhostReel's own models have.
//!
//! Protocol: JSON-RPC 2.0, one message per line on stdin/stdout (MCP stdio transport). Logs go
//! to stderr so they never corrupt the stream.

use std::io::Write as _;

use anyhow::Context;
use base64::Engine as _;
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::paths::Paths;
use ghostreel_core::tools::{self, Origin, ToolCtx, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

const PROTOCOL_VERSION: &str = "2024-11-05";

pub async fn serve(paths: Paths) -> anyhow::Result<()> {
    // Handlers hold a Db across awaits (search, chat), which isn't Send: keep them on this thread.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let mut lines = BufReader::new(tokio::io::stdin()).lines();
            eprintln!("ghostreel mcp: ready (data: {})", paths.data_dir.display());
            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                let req: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        send(&error_response(Value::Null, -32700, &format!("invalid JSON: {e}")));
                        continue;
                    }
                };
                let id = req.get("id").cloned().unwrap_or(Value::Null);
                let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("").to_string();
                let params = req.get("params").cloned().unwrap_or(json!({}));

                // Notifications (no id) expect no response.
                if id.is_null() && method.starts_with("notifications/") {
                    continue;
                }

                match method.as_str() {
                    "initialize" => send(&json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": { "tools": {} },
                            "serverInfo": { "name": "ghostreel", "version": env!("CARGO_PKG_VERSION") },
                            "instructions": tools::INSTRUCTIONS,
                        }
                    })),
                    "ping" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": {} })),
                    "tools/list" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": tools_list() })),
                    "tools/call" => {
                        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let result = call_tool_content(&paths, &name, &args).await;
                        send(&match result {
                            Ok(content) => json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": { "content": content, "isError": false }
                            }),
                            // Tool errors belong in the result, so the agent can read and fix them.
                            Err(e) => json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": tool_result(&json!({ "error": e.to_string() }), true)
                            }),
                        });
                    }
                    _ => send(&error_response(id, -32601, &format!("unknown method '{method}'"))),
                }
            }
            Ok(())
        })
        .await
}



fn send(msg: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn tool_result(value: &Value, is_error: bool) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

/// The `tools/list` result: every tool of the registry, with annotations from what it does.
fn tools_list() -> Value {
    json!({ "tools": tools::render_mcp() })
}

/// What a tool returned, as MCP content: the text (the tool's own rendering, or its result as
/// JSON), then each picture preceded by its caption.
fn content(out: ToolOutput) -> Vec<Value> {
    let mut content = Vec::new();
    let head = match &out.text {
        Some(text) => Some(text.clone()),
        None if out.value.is_null() => None,
        None => Some(serde_json::to_string_pretty(&out.value).unwrap_or_else(|_| out.value.to_string())),
    };
    if let Some(text) = head {
        content.push(json!({ "type": "text", "text": text }));
    }
    for image in out.images {
        // The name is in the caption so a client that saves pictures can keep them apart.
        content.push(json!({ "type": "text", "text": format!("{} (file {})", image.caption, image.name) }));
        content.push(json!({
            "type": "image",
            "data": base64::engine::general_purpose::STANDARD.encode(&image.jpeg),
            "mimeType": "image/jpeg",
        }));
    }
    content
}

/// Run one registry tool for an outside agent. A fresh database connection per call keeps the
/// handler's future off any shared state; the embedding model starts when a search needs it.
async fn call_tool_content(paths: &Paths, name: &str, args: &Value) -> anyhow::Result<Vec<Value>> {
    let config = Config::load(&paths.config_file).unwrap_or_default();
    let mut db = Db::open(&paths.db_file()).with_context(|| format!("cannot open {}", paths.db_file().display()))?;
    let mut embedder = None;
    let mut ctx = ToolCtx::new(paths, &config, &mut db, &mut embedder, Origin::External).on_demand_embedder();
    let out = tools::call(&mut ctx, name, args).await?;
    Ok(content(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A data directory of its own, removed afterwards: nothing here may touch the real
    /// `~/.ghostreel`.
    struct TempDir(std::path::PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_paths() -> (TempDir, Paths) {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ghostreel-mcp-test-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = Paths::in_dir(&dir);
        (TempDir(dir), paths)
    }

    #[test]
    fn tools_list_is_the_registry_and_nothing_else() {
        let listed: Vec<String> =
            tools_list()["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
        let registry: Vec<String> = tools::names().iter().map(|n| n.to_string()).collect();
        assert_eq!(listed, registry);
        assert_eq!(listed.len(), 45);
        for t in tools_list()["tools"].as_array().unwrap() {
            assert!(t["annotations"]["readOnlyHint"].is_boolean() && t["annotations"]["destructiveHint"].is_boolean());
        }
    }

    #[test]
    fn pictures_follow_their_captions_after_the_text() {
        let out = ToolOutput::json(json!({ "frames": [] })).images(vec![tools::ToolImage {
            caption: "video 1 at 2.00s".into(),
            jpeg: vec![0xff, 0xd8, 0xff],
            name: "v1_t2000ms.jpg".into(),
        }]);
        let blocks = content(out);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["text"], "video 1 at 2.00s (file v1_t2000ms.jpg)");
        assert_eq!(blocks[2]["type"], "image");
        assert_eq!(blocks[2]["mimeType"], "image/jpeg");
        let text = content(ToolOutput::json(json!({ "ignored": true })).text("0.00-1.00 hello\n"));
        assert_eq!(text[0]["text"], "0.00-1.00 hello\n", "a tool's own text is sent as it is");
    }

    #[tokio::test]
    async fn a_call_runs_the_registry_against_the_data_directory() {
        let (_dir, paths) = temp_paths();
        let made = call_tool_content(&paths, "create_project", &json!({ "name": "Spain", "fps": 30 })).await.unwrap();
        assert!(made[0]["text"].as_str().unwrap().contains("\"Spain\""));
        let listed = call_tool_content(&paths, "list_projects", &json!({})).await.unwrap();
        assert!(listed[0]["text"].as_str().unwrap().contains("Spain"));
        // Without a current project, the project must be named — as it always was over MCP.
        let err = call_tool_content(&paths, "list_scripts", &json!({})).await.unwrap_err();
        assert!(err.to_string().contains("project"), "{err}");
        let err = call_tool_content(&paths, "no_such_tool", &json!({})).await.unwrap_err();
        assert!(err.to_string().contains("unknown tool"), "{err}");
        // The old name resolves, so a client that still sends it keeps working.
        let hits = call_tool_content(
            &paths,
            "search_moments",
            &json!({ "project": "Spain", "query": "x", "keywords_only": true }),
        )
        .await
        .unwrap();
        assert!(hits[0]["text"].as_str().unwrap().contains("\"hits\""));
    }
}
