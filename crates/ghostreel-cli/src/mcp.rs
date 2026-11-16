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
use ghostreel_core::tools::{self, Detail, Origin, RenderOpts, ToolCtx, ToolOutput, ToolProfile, ToolSession};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone)]
struct McpScope {
    origin: Origin,
    profile: ToolProfile,
    project_id: Option<i64>,
    trail_file: Option<std::path::PathBuf>,
    events_file: Option<std::path::PathBuf>,
}

impl Default for McpScope {
    fn default() -> Self {
        Self {
            origin: Origin::External,
            profile: ToolProfile::All,
            project_id: None,
            trail_file: None,
            events_file: None,
        }
    }
}

impl McpScope {
    fn from_env() -> Self {
        let origin = match std::env::var("GHOSTREEL_MCP_ORIGIN").ok().as_deref() {
            Some("clip_run") => Origin::ClipRun,
            Some("script_chat") => Origin::ScriptChat,
            _ => Origin::External,
        };
        let profile = match std::env::var("GHOSTREEL_MCP_PROFILE").ok().as_deref() {
            Some("clip_run") => ToolProfile::ClipRun,
            _ => ToolProfile::All,
        };
        let project_id = std::env::var("GHOSTREEL_MCP_PROJECT").ok().and_then(|s| s.trim().parse::<i64>().ok());
        let trail_file =
            std::env::var_os("GHOSTREEL_MCP_TRAIL_FILE").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        let events_file =
            std::env::var_os("GHOSTREEL_MCP_EVENTS_FILE").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        Self { origin, profile, project_id, trail_file, events_file }
    }

    fn append_trail(&self, line: &str) {
        let Some(path) = &self.trail_file else { return };
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{line}");
        }
    }

    fn append_event(&self, event: &Value) {
        let Some(path) = &self.events_file else { return };
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{event}");
        }
    }
}

pub async fn serve(paths: Paths) -> anyhow::Result<()> {
    let scope = McpScope::from_env();
    // Handlers hold a Db across awaits (search, chat), which isn't Send: keep them on this thread.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let mut session = ToolSession::persistent(32_000);
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
                    "tools/list" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": tools_list_for(&scope) })),
                    "tools/call" => {
                        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args = params.get("arguments").cloned().unwrap_or(json!({}));
                        let result = call_tool_content_in_session(&paths, &scope, &mut session, &name, &args).await;
                        send(&match result {
                            Ok(content) => json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": { "content": content, "isError": false }
                            }),
                            // Tool errors belong in the result, so the agent can read and fix them.
                            Err(e) => json!({
                                "jsonrpc": "2.0", "id": id,
                                "result": tool_result(&mcp_error_value(&e.to_string()), true)
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

fn mcp_error_value(message: &str) -> Value {
    serde_json::from_str::<Value>(message)
        .ok()
        .filter(|value| value.get("error").is_some())
        .unwrap_or_else(|| json!({ "error": message }))
}

/// The `tools/list` result: every tool of the registry, with annotations from what it does.
#[cfg(test)]
fn tools_list() -> Value {
    tools_list_for(&McpScope::default())
}

fn tools_list_for(scope: &McpScope) -> Value {
    let opts = RenderOpts::new(Detail::Full, scope.project_id.is_some()).with_profile(scope.profile);
    json!({ "tools": tools::render_mcp_with(opts) })
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
#[cfg(test)]
async fn call_tool_content(paths: &Paths, name: &str, args: &Value) -> anyhow::Result<Vec<Value>> {
    call_tool_content_scoped(paths, &McpScope::default(), name, args).await
}

#[cfg(test)]
async fn call_tool_content_scoped(
    paths: &Paths,
    scope: &McpScope,
    name: &str,
    args: &Value,
) -> anyhow::Result<Vec<Value>> {
    call_tool_content_in_session(paths, scope, &mut ToolSession::new(32_000), name, args).await
}

async fn call_tool_content_in_session(
    paths: &Paths,
    scope: &McpScope,
    session: &mut ToolSession,
    name: &str,
    args: &Value,
) -> anyhow::Result<Vec<Value>> {
    let canon = tools::canonical_name(name);
    if scope.profile != ToolProfile::All && !tools::allows(scope.profile, name) {
        scope.append_trail(&format!("{canon}: error: tool '{name}' is not available"));
        anyhow::bail!(
            "tool '{name}' is not available in this clip run; available tools: {}",
            tools::profile_names(scope.profile).join(", ")
        );
    }
    scope.append_event(&json!({ "kind": "tool_started", "tool": canon, "args": args }));
    let config = Config::load(&paths.config_file).unwrap_or_default();
    let mut db = Db::open(&paths.db_file()).with_context(|| format!("cannot open {}", paths.db_file().display()))?;
    let scoped_project = match scope.project_id {
        Some(pid) => Some(db.project(pid)?),
        None => None,
    };
    let mut embedder = None;
    let mut ctx = ToolCtx::new(paths, &config, &mut db, &mut embedder, scope.origin).on_demand_embedder();
    if let Some(project) = scoped_project {
        ctx = ctx.with_project(project);
    }
    let called = session.call(&mut ctx, name, args).await;
    scope.append_trail(&format!("{canon}: {}", called.summary));
    let script_id = called.value.as_ref().and_then(|v| v.get("script_id")).and_then(Value::as_i64);
    scope.append_event(&json!({
        "kind": "tool_finished",
        "tool": canon,
        "args": args,
        "summary": called.summary,
        "script_id": script_id,
    }));
    if called.is_error {
        anyhow::bail!("{}", called.model_text);
    }
    Ok(content(ToolOutput::json(called.value.unwrap_or(Value::Null)).text(called.model_text).images(called.images)))
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
    fn structured_tool_errors_are_not_double_encoded() {
        let structured = mcp_error_value(r#"{"error":"bad project"}"#);
        assert_eq!(structured, json!({"error":"bad project"}));
        assert_eq!(mcp_error_value("plain failure"), json!({"error":"plain failure"}));
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

    #[tokio::test]
    async fn mcp_connection_remembers_reads_and_sizes_output() {
        let (_dir, paths) = temp_paths();
        call_tool_content(&paths, "create_project", &json!({"name":"Session", "fps":30})).await.unwrap();
        let scope = McpScope::default();
        let mut session = ToolSession::new(32_000);
        let args = json!({});
        let first = call_tool_content_in_session(&paths, &scope, &mut session, "list_projects", &args).await.unwrap();
        assert!(first[0]["text"].as_str().unwrap().contains("Session"));
        let repeat = call_tool_content_in_session(&paths, &scope, &mut session, "list_projects", &args).await.unwrap();
        assert!(repeat[0]["text"].as_str().unwrap().contains("already called this"));
        call_tool_content_in_session(&paths, &scope, &mut session, "list_projects", &args).await.unwrap();
        let error =
            call_tool_content_in_session(&paths, &scope, &mut session, "list_projects", &args).await.unwrap_err();
        assert!(error.to_string().contains("exact call"));
        let mut small = ToolSession::new(20);
        let sized = call_tool_content_in_session(&paths, &scope, &mut small, "list_projects", &args).await.unwrap();
        assert!(!sized[0]["text"].as_str().unwrap().contains("Session"), "{sized:?}");
    }

    #[tokio::test]
    async fn mcp_save_warns_for_resolved_clips_and_beds_until_their_manifest_is_read() {
        let (_dir, paths) = temp_paths();
        call_tool_content(&paths, "create_project", &json!({"name":"Pictures", "fps":30})).await.unwrap();
        {
            let mut db = Db::open(&paths.db_file()).unwrap();
            db.conn
                .execute_batch(
                    "INSERT INTO videos(id,content_hash,size,duration_s) VALUES
                (1,'a',1,60),(2,'b',1,60),(3,'c',1,60);
                INSERT INTO transcript_sentences(id,video_id,start_s,end_s,text) VALUES
                (12,2,20,30,'Answer.'),(13,3,20,30,'Voice.');",
                )
                .unwrap();
            // The videos belong to the project (a save drops footage from outside it).
            let project = db.require_project("Pictures").unwrap();
            let folder = db.add_folder(project.id, &_dir.0, true).unwrap();
            for video in [1, 2, 3] {
                db.conn
                    .execute_batch(&format!(
                        "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) \
                         VALUES ({video}, {}, '{video}.mp4', 1, 0, 0)",
                        folder.id
                    ))
                    .unwrap();
            }
        }
        let scope = McpScope::default();
        let mut session = ToolSession::new(32_000);
        let args = json!({"project":"Pictures", "dry_run":true, "repair":false, "script":{
            "title":"Resolved", "beats":[{"id":"1","purpose":"Test", "clips":[
                {"video_id":1,"from":"s12","to":"s12"}
            ], "bed":{"video_id":1,"from":"s13","to":"s13"}}]
        }});
        let saved = call_tool_content_in_session(&paths, &scope, &mut session, "save_script", &args).await.unwrap();
        let text = saved[0]["text"].as_str().unwrap();
        assert!(text.contains("NOTE: you chose clips from video #2, #3"), "{text}");
        call_tool_content_in_session(
            &paths,
            &scope,
            &mut session,
            "get_manifest",
            &json!({"video_ids":[2,3], "frames":"none"}),
        )
        .await
        .unwrap();
        let saved = call_tool_content_in_session(&paths, &scope, &mut session, "save_script", &args).await.unwrap();
        assert!(!saved[0]["text"].as_str().unwrap().contains("NOTE: you chose"));
        // State is connection-local, never shared with a later client.
        let saved = call_tool_content_in_session(&paths, &scope, &mut ToolSession::new(32_000), "save_script", &args)
            .await
            .unwrap();
        assert!(saved[0]["text"].as_str().unwrap().contains("NOTE: you chose clips from video #2, #3"));
    }

    #[tokio::test]
    async fn scoped_clip_run_mcp_filters_tools_defaults_project_and_logs_trail() {
        let (_dir, paths) = temp_paths();
        let created =
            call_tool_content(&paths, "create_project", &json!({ "name": "Scoped", "fps": 30 })).await.unwrap();
        let project_id =
            serde_json::from_str::<Value>(created[0]["text"].as_str().unwrap()).unwrap()["id"].as_i64().unwrap();
        let trail = paths.data_dir.join("mcp-trail.log");
        let events = paths.data_dir.join("mcp-events.jsonl");
        let scope = McpScope {
            origin: Origin::ClipRun,
            profile: ToolProfile::ClipRun,
            project_id: Some(project_id),
            trail_file: Some(trail.clone()),
            events_file: Some(events.clone()),
        };

        let listed: Vec<String> = tools_list_for(&scope)["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(listed, tools::profile_names(ToolProfile::ClipRun));
        assert!(!listed.contains(&"save_clips".to_string()));
        // With project_id set, `project` is no longer required in MCP inputSchema.
        let scoped_list = tools_list_for(&scope);
        let outline_schema = &scoped_list["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "get_outline")
            .unwrap()["inputSchema"];
        assert!(
            outline_schema["required"].as_array().is_none_or(|req| !req.iter().any(|v| v == "project")),
            "{outline_schema}"
        );

        // Read-only tool works without passing `project` and logs to trail_file + events_file.
        let out = call_tool_content_scoped(&paths, &scope, "get_outline", &json!({})).await.unwrap();
        assert!(!out.is_empty());
        // Write tool is refused by profile even if called directly.
        let err = call_tool_content_scoped(
            &paths,
            &scope,
            "save_clips",
            &json!({ "clips": [{ "video_id": 1, "in_s": 0.0, "out_s": 1.0, "title": "x", "hook": "h" }] }),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("not available"), "{err}");
        let trail_text = std::fs::read_to_string(&trail).unwrap();
        assert!(trail_text.contains("get_outline: "), "{trail_text}");
        assert!(trail_text.contains("save_clips: error:"), "{trail_text}");
        let events_text = std::fs::read_to_string(&events).unwrap();
        assert!(events_text.contains("\"kind\":\"tool_started\""), "{events_text}");
        assert!(events_text.contains("\"kind\":\"tool_finished\""), "{events_text}");
    }
}
