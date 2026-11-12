//! The registry's own tests: its shape, and the handlers that more than one surface depends on.

use serde_json::{Value, json};

use super::*;
use crate::config::Config;
use crate::db::Db;
use crate::paths::Paths;
use crate::projects::{NewProject, Project};

/// The tools MCP clients have been given since before the registry. Every one must stay, under
/// the same name.
const MCP_TOOLS: &[&str] = &[
    "doctor", "list_projects", "create_project", "get_glossary", "set_glossary", "add_folder", "index",
    "index_status", "list_videos", "search", "get_transcript", "get_steadiness", "get_faces", "get_scenes",
    "get_people", "set_person", "merge_people", "get_speakers", "get_frames", "preview_crop", "get_video",
    "list_scripts", "list_runs", "get_script", "apply_fix", "draft_script", "save_script", "export_script",
    "preview_script", "get_settings", "set_settings", "delete_project", "generate_clips", "list_agents",
    "find_clip_candidates", "check_clips", "save_clips", "list_trash", "restore_trash", "list_clip_sets",
    "set_video_role",
];

#[test]
fn every_tool_is_listed_once_and_none_of_the_mcp_tools_is_lost() {
    let names = names();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "a name is registered twice: {names:?}");
    for want in MCP_TOOLS {
        assert!(names.contains(want), "{want} is gone from the registry");
    }
    assert_eq!(names.len(), MCP_TOOLS.len(), "a tool was added: list it in MCP_TOOLS too");
}

#[test]
fn every_spec_is_complete() {
    for t in all() {
        let s = t.spec;
        assert!(s.name.chars().all(|c| c.is_ascii_lowercase() || c == '_'), "{} is not snake_case", s.name);
        assert!(s.summary.len() > 20, "{} has no real summary", s.name);
        assert!(!s.when.is_empty(), "{} does not say when to call it", s.name);
        assert!(!s.returns.is_empty(), "{} does not say what it returns", s.name);
        let mut seen = std::collections::HashSet::new();
        for p in s.params {
            assert!(seen.insert(p.name), "{} lists {} twice", s.name, p.name);
        }
        let schema = s.input_schema(false, true);
        assert_eq!(schema["type"], "object");
        for r in schema["required"].as_array().unwrap() {
            assert!(schema["properties"].get(r.as_str().unwrap()).is_some(), "{} requires an unknown field", s.name);
        }
    }
}

#[test]
fn mcp_annotations_follow_what_the_tool_does() {
    let listed = render_mcp();
    let note = |name: &str| listed.iter().find(|t| t["name"] == name).unwrap()["annotations"].clone();
    assert_eq!(note("search")["readOnlyHint"], true);
    assert_eq!(note("save_script")["readOnlyHint"], false);
    assert_eq!(note("save_script")["destructiveHint"], false);
    assert_eq!(note("delete_project")["destructiveHint"], true);
    assert_eq!(note("merge_people")["destructiveHint"], true);
    for t in &listed {
        assert!(t["description"].as_str().unwrap().len() > 40);
        assert_eq!(t["inputSchema"]["type"], "object");
    }
}

#[test]
fn an_old_name_still_resolves_but_is_never_listed() {
    assert_eq!(canonical_name("search_moments"), "search");
    assert!(lookup("search_moments").is_some());
    assert!(!names().contains(&"search_moments"));
}

#[test]
fn the_current_project_makes_project_optional_in_the_schema() {
    let with = render_openai(RenderOpts::new(Detail::Full, true));
    let without = render_openai(RenderOpts::new(Detail::Full, false));
    let required = |v: &Value, name: &str| {
        v.as_array().unwrap().iter().find(|t| t["function"]["name"] == name).unwrap()["function"]["parameters"]["required"]
            .clone()
    };
    assert_eq!(required(&with, "search"), json!(["query"]));
    assert_eq!(required(&without, "search"), json!(["project", "query"]));
}

#[test]
fn short_prose_lists_every_tool_in_less_room() {
    let full = tools_prose(RenderOpts::new(Detail::Full, true));
    let short = tools_prose(RenderOpts::new(Detail::Short, true));
    for name in names() {
        assert!(short.contains(&format!("- {name}(")), "{name} is missing from the short prose");
    }
    assert!(short.len() * 2 < full.len(), "short {} vs full {}", short.len(), full.len());
}

/// A project with one video that has speech and a described keyframe, in a throwaway directory.
struct Fixture {
    _tmp: tempfile::TempDir,
    paths: Paths,
    config: Config,
    db: Db,
    project: Project,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let paths = Paths::in_dir(tmp.path());
    let mut db = Db::open(&paths.db_file()).unwrap();
    let project = db.create_project(&NewProject::named("TestProj")).unwrap();
    let folder = db.add_folder(project.id, tmp.path(), true).unwrap();
    let file = tmp.path().join("clip.mp4");
    std::fs::write(&file, b"data").unwrap();
    let c = &db.conn;
    c.execute(
        "INSERT INTO videos(id, content_hash, size, duration_s, fps, width, height, language, summary)
         VALUES (10, 'hash10', 1000, 30.0, 25.0, 1920, 1080, 'en', 'CM5 unboxing overview')",
        [],
    )
    .unwrap();
    c.execute(
        "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (10, ?1, ?2, 1000, 0, 0)",
        rusqlite::params![folder.id, file.to_str().unwrap()],
    )
    .unwrap();
    c.execute(
        "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
         VALUES (10, 0.0, 5.0, 'welcome to cm5 unboxing'), (10, 5.0, 12.0, 'here is the compute module board')",
        [],
    )
    .unwrap();
    c.execute(
        "INSERT INTO frames(video_id, t_s, description_json)
         VALUES (10, 2.0, '{\"description\": \"Hands holding a green board.\"}'),
                (10, 8.0, '{\"description\": \"Close-up of the chip.\"}')",
        [],
    )
    .unwrap();
    c.execute(
        "INSERT INTO chunks(video_id, kind, start_s, end_s, text)
         VALUES (10, 'moment', 0.0, 5.0, 'Hands holding a green board. welcome to cm5 unboxing')",
        [],
    )
    .unwrap();
    Fixture { paths, config: Config::default(), db, project, _tmp: tmp }
}

#[tokio::test]
async fn an_outside_agent_names_its_project_and_reads_transcripts_as_text() {
    let mut f = fixture();
    let mut embedder = None;
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::External);

    let out = call(&mut ctx, "get_transcript", &json!({ "video_id": 10, "end_s": 6.0 })).await.unwrap();
    let text = out.text.expect("text by default");
    assert!(text.contains("=== video 10 (clip.mp4)"), "{text}");
    assert!(text.contains("0.00-5.00 welcome to cm5 unboxing"), "{text}");
    assert_eq!(out.summary, "2 segments");
    assert_eq!(out.value.as_array().unwrap().len(), 2, "the structured value is there too");

    let json_out = call(&mut ctx, "get_transcript", &json!({ "video_id": 10, "format": "json" })).await.unwrap();
    assert_eq!(json_out.value[0]["text"], "welcome to cm5 unboxing");
    assert!(json_out.text.is_none());

    // Without a current project the project has to be named.
    let err = call(&mut ctx, "list_videos", &json!({})).await.unwrap_err();
    assert!(err.to_string().contains("project"), "{err}");
    let out = call(&mut ctx, "list_videos", &json!({ "project": "testproj" })).await.unwrap();
    assert_eq!(out.value["rows"][0]["file"], "clip.mp4");
    assert_eq!(out.value["rows"][0]["has_speech"], true);
}

#[tokio::test]
async fn the_current_project_is_the_default_and_scopes_video_ids() {
    let mut f = fixture();
    f.db.conn
        .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (99, 'other', 1, 10.0)", [])
        .unwrap();
    let mut embedder = None;
    let mut grounding = Grounding::default();
    let project = f.project.clone();
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat)
        .with_project(project)
        .with_grounding(&mut grounding);

    let listed = call(&mut ctx, "list_videos", &json!({})).await.unwrap();
    assert_eq!(listed.summary, "1 videos");

    let video = call(&mut ctx, "get_video", &json!({ "video_id": 10 })).await.unwrap();
    assert!(video.value["keyframes"][0]["description"].as_str().unwrap().contains("green board"));
    assert_eq!(video.value["has_speech"], true);
    assert_eq!(video.summary, "30s, 2 frames");

    // A video of another project is refused.
    let err = call(&mut ctx, "get_video", &json!({ "video_id": 99 })).await.unwrap_err();
    assert!(err.to_string().contains("not in project"), "{err}");
    drop(ctx);
    assert!(grounding.is_grounded(10, 0.0, 30.0, 0.0), "opening a video grounds all of it");
    assert!(!grounding.is_grounded(99, 0.0, 1.0, 0.0));
}

#[tokio::test]
async fn search_finds_keywords_and_says_why_it_is_not_using_meaning() {
    let mut f = fixture();
    let mut embedder = None;
    let project = f.project.clone();
    let mut grounding = Grounding::default();
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat)
        .with_project(project)
        .with_grounding(&mut grounding);
    let out = call(&mut ctx, "search_moments", &json!({ "query": "unboxing" })).await.unwrap();
    assert_eq!(out.summary, "1 hits");
    assert!(!out.empty);
    assert!(out.value["mode"].as_str().unwrap().starts_with("keywords only"), "{}", out.value["mode"]);
    assert_eq!(out.value["hits"][0]["file"], "clip.mp4");
    assert!(out.value["hits"][0]["camera"].as_str().is_some());

    let none = call(&mut ctx, "search", &json!({ "query": "zzzz nothing" })).await.unwrap();
    assert!(none.empty);
    drop(ctx);
    assert!(grounding.is_grounded(10, 1.0, 4.0, 0.0), "a hit grounds the window it returned");
}

#[tokio::test]
async fn a_tool_that_starts_a_run_refuses_to_start_one_from_inside_another() {
    let mut f = fixture();
    let mut embedder = None;
    for (origin, inside) in [(Origin::ScriptChat, "a script chat"), (Origin::ClipRun, "a clip run")] {
        let project = f.project.clone();
        let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, origin).with_project(project);
        for tool in ["draft_script", "generate_clips"] {
            let err = call(&mut ctx, tool, &json!({ "message": "x" })).await.unwrap_err();
            assert!(matches!(err, ToolError::Nested { .. }), "{tool} from {origin:?}: {err}");
            assert!(err.to_string().contains(inside), "{err}");
        }
    }
}

#[tokio::test]
async fn people_and_glossary_changes_go_through_the_same_handlers() {
    let mut f = fixture();
    let mut embedder = None;
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::External);
    let out = call(
        &mut ctx,
        "set_glossary",
        &json!({ "project": "TestProj", "add": [{ "term": "CM5", "heard_as": ["see em five"] }] }),
    )
    .await
    .unwrap();
    assert_eq!(out.summary, "1 terms");
    let got = call(&mut ctx, "get_glossary", &json!({ "project": "TestProj" })).await.unwrap();
    assert_eq!(got.value["terms"][0]["term"], "CM5");
    let missing = call(&mut ctx, "get_people", &json!({ "project": "TestProj" })).await.unwrap();
    assert_eq!(missing.value["people"], json!([]));
    assert!(call(&mut ctx, "no_such_tool", &json!({})).await.is_err());
}

#[test]
fn a_call_that_comes_back_different_is_not_a_loop() {
    let mut guard = CallGuard::default();
    // `index` with the same arguments, each answer further along: never refused.
    for progress in 0..20u64 {
        assert_eq!(guard.check("index{}"), session::Verdict::Allow);
        guard.note("index{}", progress);
    }
    // The same answer three times in a row: the fourth identical call is refused, and a model
    // that asks again anyway is stopped.
    let mut guard = CallGuard::default();
    for _ in 0..3 {
        assert_eq!(guard.check("set_person{1}"), session::Verdict::Allow);
        guard.note("set_person{1}", 7);
    }
    assert!(matches!(guard.check("set_person{1}"), session::Verdict::Refuse { .. }));
    assert!(matches!(guard.check("set_person{1}"), session::Verdict::Stop { .. }));
    // Another call clears it.
    assert_eq!(guard.check("get_people{}"), session::Verdict::Allow);
    guard.note("get_people{}", 1);
    assert_eq!(guard.check("set_person{1}"), session::Verdict::Allow);
}

#[test]
fn the_cap_on_calls_is_optional() {
    let mut none = ToolBudget::from_setting(0);
    for _ in 0..10_000 {
        none.spend();
    }
    assert!(!none.exhausted(), "no cap by default");
    assert_eq!(none.cap(), None);
    let mut three = ToolBudget::from_setting(3);
    three.spend();
    three.spend();
    assert!(!three.exhausted());
    three.spend();
    assert!(three.exhausted());
    // Not clamped to some ceiling.
    assert_eq!(ToolBudget::from_setting(5000).cap(), Some(5000));
}

#[tokio::test]
async fn a_model_that_repeats_a_read_is_told_and_then_stopped() {
    let mut f = fixture();
    let mut embedder = None;
    let project = f.project.clone();
    let mut session = ToolSession::new(8000);
    let mut results = Vec::new();
    for _ in 0..5 {
        let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat)
            .with_project(project.clone());
        results.push(session.call(&mut ctx, "list_videos", &json!({})).await);
    }
    assert!(results[0].value.is_some() && !results[0].is_error, "the first call runs");
    assert!(results[1].model_text.contains("already called this"), "the second is answered from memory");
    assert!(results[1].value.is_none());
    assert!(results[2].model_text.contains("Stop looking"), "the third is withheld: {}", results[2].model_text);
    assert!(results[3].is_error && results[3].model_text.contains("give your final answer"), "{}", results[3].model_text);
    assert!(!results[3].stop);
    assert!(results[4].stop, "still asking: the run moves on to its answer");
}

#[tokio::test]
async fn a_write_is_never_answered_from_memory() {
    let mut f = fixture();
    let mut embedder = None;
    let args = json!({ "project": "TestProj", "add": [{ "term": "CM5" }] });
    let mut session = ToolSession::new(8000);
    for n in 1..=3 {
        let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat);
        let called = session.call(&mut ctx, "set_glossary", &args).await;
        assert!(called.value.is_some(), "call {n} ran: {}", called.model_text);
    }
    // Three identical outcomes in a row: the next one is refused rather than run.
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat);
    let fourth = session.call(&mut ctx, "set_glossary", &args).await;
    assert!(fourth.is_error && fourth.value.is_none());
}

#[tokio::test]
async fn results_are_cut_to_the_room_a_context_has_and_what_is_left_is_what_was_proven() {
    let mut f = fixture();
    for i in 0..200 {
        f.db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (10, ?1, ?2, 'a line of speech')",
                rusqlite::params![100.0 + i as f64, 100.5 + i as f64],
            )
            .unwrap();
    }
    let mut embedder = None;
    let project = f.project.clone();
    let mut session = ToolSession::new(600);
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat).with_project(project);
    let called = session.call(&mut ctx, "get_transcript", &json!({ "video_id": 10 })).await;
    assert!(called.model_text.len() <= 700, "{} chars", called.model_text.len());
    assert!(called.model_text.contains("left out to fit"), "{}", called.model_text);
    // The structured value holds exactly the segments the text shows, so evidence is not overstated.
    let shown = called.model_text.lines().filter(|l| l.contains("a line of speech") || l.contains("welcome") || l.contains("compute")).count();
    assert_eq!(called.value.as_ref().unwrap().as_array().unwrap().len(), shown);
}

#[tokio::test]
async fn errors_come_back_as_something_a_model_can_read() {
    let mut f = fixture();
    let mut embedder = None;
    let mut session = ToolSession::new(8000);
    let mut ctx = ToolCtx::new(&f.paths, &f.config, &mut f.db, &mut embedder, Origin::ScriptChat);
    let unknown = session.call(&mut ctx, "frobnicate", &json!({})).await;
    assert!(unknown.is_error && unknown.summary.starts_with("error:"));
    assert!(unknown.model_text.contains("get_video") && unknown.model_text.contains("the tools are"), "{}", unknown.model_text);
    let missing = session.call(&mut ctx, "get_video", &json!({})).await;
    assert!(missing.model_text.contains("video_id"), "{}", missing.model_text);
    assert!(!missing.cancelled);
}


// ---------------------------------------------------------------------------------------------
// The action loop: clip workers and the orchestrator, over a scripted server.

use std::sync::{Arc, Mutex};

use crate::chat::ChatBackend;

/// A server that answers each chat request with whatever `reply(request, n)` says (the `message`
/// of the completion), and remembers the requests.
pub(crate) async fn scripted_server(
    reply: impl Fn(&Value, usize) -> Value + Send + Sync + 'static,
) -> (String, Arc<Mutex<Vec<Value>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let log = seen.clone();
    let reply = Arc::new(reply);
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let (log, reply) = (log.clone(), reply.clone());
            tokio::spawn(async move {
                let mut got = Vec::new();
                let mut buf = vec![0u8; 65536];
                loop {
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                    let t = String::from_utf8_lossy(&got);
                    if let Some(h) = t.find("\r\n\r\n") {
                        let len = t[..h]
                            .lines()
                            .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                            .unwrap_or(0);
                        if got.len() >= h + 4 + len {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&got).to_string();
                let body = text.split("\r\n\r\n").nth(1).unwrap_or("{}");
                let request: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                let n = {
                    let mut log = log.lock().unwrap();
                    log.push(request.clone());
                    log.len() - 1
                };
                let message = reply(&request, n);
                let body = json!({ "choices": [{ "message": message }] }).to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            });
        }
    });
    (format!("http://{addr}"), seen)
}

fn tool_call(id: usize, name: &str, args: Value) -> Value {
    json!({ "content": null, "tool_calls": [{ "id": format!("call_{id}"), "type": "function",
        "function": { "name": name, "arguments": args.to_string() } }] })
}

fn answer() -> Value {
    json!({ "content": json!({ "picks": [] }).to_string() })
}

fn schema() -> Value {
    json!({ "type": "object", "required": ["picks"], "properties": { "picks": { "type": "array" } } })
}

fn access(f: &Fixture) -> ToolAccess {
    ToolAccess::new(f.paths.clone(), Arc::new(f.config.clone()), f.project.id, None)
}

fn server_backend(url: String) -> ChatBackend {
    ChatBackend::Server { url, model: String::new(), api_key: String::new(), ctx_tokens: 32768 }
}

fn has_tools(request: &Value) -> bool {
    request.get("tools").is_some()
}

#[tokio::test]
async fn a_worker_that_calls_no_tool_answers_in_one_request_and_was_offered_every_tool() {
    let f = fixture();
    let (url, seen) = scripted_server(|_, _| answer()).await;
    let mut backend = server_backend(url);
    let got = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "no tool wanted, no extra request");
    let offered: Vec<&str> =
        seen[0]["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(offered, names(), "the worker is offered the whole registry");
    let prompt = seen[0]["messages"].to_string();
    assert!(!prompt.contains("do not use any tools"), "the old instruction is gone");
}

#[tokio::test]
async fn a_worker_may_look_something_up_before_it_answers() {
    let f = fixture();
    let (url, seen) = scripted_server(|request, n| {
        let looked = request["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool");
        if looked { answer() } else { tool_call(n, "get_transcript", json!({ "video_id": 10, "end_s": 6.0 })) }
    })
    .await;
    let mut backend = server_backend(url);
    let got = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let tool_message = seen[1]["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool").unwrap().clone();
    let text = tool_message["content"].as_str().unwrap();
    assert!(text.contains("welcome to cm5 unboxing"), "the tool ran against the project's database: {text}");
}

#[tokio::test]
async fn a_worker_cannot_start_another_run_from_inside_one() {
    let f = fixture();
    let (url, seen) = scripted_server(|request, n| {
        let looked = request["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool");
        if looked { answer() } else { tool_call(n, "generate_clips", json!({})) }
    })
    .await;
    let mut backend = server_backend(url);
    run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    let seen = seen.lock().unwrap();
    let told = seen[1]["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool").unwrap()["content"].to_string();
    assert!(told.contains("already inside a clip run"), "{told}");
}

#[tokio::test]
async fn a_model_that_keeps_asking_for_the_same_thing_is_made_to_answer() {
    let f = fixture();
    let (url, seen) = scripted_server(|request, n| {
        if request.get("response_format").is_some() {
            answer()
        } else {
            tool_call(n, "get_video", json!({ "video_id": 10 }))
        }
    })
    .await;
    let mut backend = server_backend(url);
    let got = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    let seen = seen.lock().unwrap();
    let rounds = seen.iter().filter(|r| has_tools(r)).count();
    assert!(rounds < 10, "the guard ends an endless loop with no cap set: {rounds} rounds");
    assert!(seen.last().unwrap().get("response_format").is_some(), "and the last request is held to the schema");
}

#[tokio::test]
async fn there_is_no_cap_on_rounds_unless_one_is_set() {
    let f = fixture();
    // Different arguments every time, so the repeat guard has nothing to object to.
    let (url, seen) = scripted_server(|request, n| {
        if request.get("response_format").is_some() || n >= 25 {
            answer()
        } else {
            tool_call(n, "get_transcript", json!({ "video_id": 10, "start_s": n as f64, "end_s": n as f64 + 1.0 }))
        }
    })
    .await;
    let mut backend = server_backend(url);
    run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(seen.lock().unwrap().iter().filter(|r| has_tools(r)).count(), 26, "25 lookups and then the answer");

    // The same model with a cap of 3 is cut off after three rounds and asked to answer.
    let (url, seen) = scripted_server(|request, n| {
        if request.get("response_format").is_some() {
            answer()
        } else {
            tool_call(n, "get_transcript", json!({ "video_id": 10, "start_s": n as f64, "end_s": n as f64 + 1.0 }))
        }
    })
    .await;
    let mut backend = server_backend(url);
    let capped = access(&f).with_max_rounds(3);
    run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&capped)).await.unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.iter().filter(|r| has_tools(r)).count(), 3);
    assert!(seen.last().unwrap().get("response_format").is_some());
}

#[tokio::test]
async fn an_answer_in_prose_is_asked_for_again_under_the_schema() {
    let f = fixture();
    let (url, seen) = scripted_server(|request, _| {
        if request.get("response_format").is_some() { answer() } else { json!({ "content": "Let me think about it." }) }
    })
    .await;
    let mut backend = server_backend(url);
    let got = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn a_stopped_run_stops_the_loop_before_asking_the_model() {
    let f = fixture();
    let (url, seen) = scripted_server(|_, _| answer()).await;
    let mut backend = server_backend(url);
    let mut stopped = access(&f);
    stopped.cancel = Some(Arc::new(std::sync::atomic::AtomicBool::new(true)));
    let err = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&stopped)).await.unwrap_err();
    assert!(err.is_cancelled(), "{err}");
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn without_tools_a_job_is_the_single_request_it_always_was() {
    let (url, seen) = scripted_server(|_, _| answer()).await;
    let mut backend = server_backend(url);
    let got = run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, None).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].get("tools").is_none() && seen[0].get("response_format").is_some());
}

#[test]
fn replies_to_the_action_protocol_are_read_leniently() {
    let schema = schema();
    let step = |v: Value| agent_loop::read_step_for_test(v, &schema);
    assert_eq!(step(json!({ "action": "final", "answer": { "picks": [] } })), "answer");
    assert_eq!(step(json!({ "picks": [] })), "answer", "a model that skips the wrapper still answers");
    assert_eq!(step(json!({ "action": "tool", "tool": "get_video", "args": { "video_id": 1 } })), "tool get_video");
    assert_eq!(step(json!({ "action": "final", "answer": { "nope": 1 } })), "unusable");
    assert_eq!(step(json!({ "hello": 1 })), "unusable");
}

/// A CLI agent that replies with the next of `replies` each time it is called (the last one
/// again after that), and keeps every prompt it was sent in `dir`.
#[cfg(unix)]
fn fake_cli(f: &Fixture, replies: &[&str]) -> (ChatBackend, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let dir = f._tmp.path().join("fake-cli");
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("agent");
    let mut cases = String::new();
    for (i, r) in replies.iter().enumerate() {
        let pattern = if i + 1 == replies.len() { "*".to_string() } else { (i + 1).to_string() };
        cases.push_str(&format!("{pattern}) echo '{r}';;\n"));
    }
    {
        use std::io::Write as _;
        let mut file = std::fs::File::create(&script).unwrap();
        let body = format!(
            "#!/bin/sh\nD={d}\nn=$(cat $D/n 2>/dev/null || echo 0)\nn=$((n+1))\necho $n > $D/n\ncat > $D/stdin$n\necho \"$@\" > $D/args$n\ncase $n in\n{cases}esac\n",
            d = dir.display()
        );
        file.write_all(body.as_bytes()).unwrap();
        file.sync_all().unwrap();
    }
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let agent = crate::cliagent::CliAgent::new(crate::config::CliAgentConfig {
        tool: "opencode".into(),
        command: script.to_str().unwrap().into(),
        timeout_secs: 30,
        ..Default::default()
    });
    (ChatBackend::Cli(agent), dir)
}

/// What the fake CLI was sent on call `n`: its stdin and its arguments.
#[cfg(unix)]
fn cli_sent(dir: &std::path::Path, n: usize) -> String {
    let stdin = std::fs::read_to_string(dir.join(format!("stdin{n}"))).unwrap_or_default();
    let args = std::fs::read_to_string(dir.join(format!("args{n}"))).unwrap_or_default();
    format!("{stdin}\n{args}")
}

#[cfg(unix)]
#[tokio::test]
async fn a_cli_agent_drives_the_same_loop_in_json_actions() {
    let f = fixture();
    let (mut backend, dir) = fake_cli(
        &f,
        &[
            r#"{"action":"tool","tool":"get_transcript","args":{"video_id":10,"end_s":6}}"#,
            r#"{"action":"final","answer":{"picks":[]}}"#,
        ],
    );
    let got = run_agent_json(&mut backend, "Pick clips from the window.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(got["picks"], json!([]));
    let first = cli_sent(&dir, 1);
    assert!(first.contains("Pick clips from the window."), "the request is in the first prompt");
    assert!(first.contains("get_video") && first.contains("generate_clips"), "and so is every tool");
    assert!(!first.contains("do not use any tools"));
    assert!(cli_sent(&dir, 2).contains("welcome to cm5 unboxing"), "the second prompt carries the tool's result: {}", cli_sent(&dir, 2));
}

// ---------------------------------------------------------------------------------------------
// One set of tools everywhere.

/// The names a prose block lists: the `- name(args): …` lines of the TOOLS section.
fn prose_names(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| l.strip_prefix("- "))
        .filter_map(|l| l.split_once('(').map(|(name, _)| name.to_string()))
        .filter(|n| n.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        .collect()
}

fn openai_names(tools: &Value) -> Vec<String> {
    tools.as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()
}

fn grammar_names(schema: &Value) -> Vec<String> {
    schema["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["properties"]["action"]["enum"][0] == "tool")
        .map(|b| b["properties"]["tool"]["enum"][0].as_str().unwrap().to_string())
        .collect()
}

/// The point of the registry: an outside agent over MCP, the script chat (on every backend and at
/// every level of detail) and a clip worker (on every backend) are all given the same tools — the
/// registry's, no more and no fewer. A tool added in one place appears in all of them or this fails.
#[cfg(unix)]
#[tokio::test]
async fn mcp_sonar_and_the_clip_workers_are_given_exactly_the_same_tools() {
    let f = fixture();
    let registry: Vec<String> = names().iter().map(|n| n.to_string()).collect();
    assert_eq!(registry.len(), MCP_TOOLS.len());

    // MCP.
    let mcp: Vec<String> = render_mcp().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
    assert_eq!(mcp, registry, "MCP tools/list");

    // Sonar: the server's tools array, the local grammar, the prose, and the system prompt that
    // carries the prose to the local and CLI loops.
    for detail in [Detail::Full, Detail::Short] {
        let opts = RenderOpts::new(detail, true);
        assert_eq!(openai_names(&render_openai(opts)), registry, "Sonar tools array ({detail:?})");
        assert_eq!(prose_names(&tools_prose(opts)), registry, "Sonar prose ({detail:?})");
        for style in [crate::script::ChatStyle::default(), crate::script::ChatStyle { broll: true, ..Default::default() }] {
            let prompt = crate::chat::build_system_prompt(&f.project, None, None, detail, style);
            assert_eq!(prose_names(&prompt), registry, "Sonar system prompt ({detail:?}, {style:?})");
        }
    }
    assert_eq!(grammar_names(&crate::chat::local_action_schema()), registry, "Sonar local grammar");

    // Clip workers and the orchestrator, on a server...
    let (url, seen) = scripted_server(|_, _| answer()).await;
    let mut backend = server_backend(url);
    run_agent_json(&mut backend, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(openai_names(&seen.lock().unwrap()[0]["tools"]), registry, "worker tools array");

    // ... through a CLI agent ...
    let (mut cli, dir) = fake_cli(&f, &[r#"{"action":"final","answer":{"picks":[]}}"#]);
    run_agent_json(&mut cli, "Pick clips.", &schema(), 512, 10, Some(&access(&f))).await.unwrap();
    assert_eq!(prose_names(&cli_sent(&dir, 1)), registry, "worker prompt for a CLI agent");

    // ... and with the local helper, whose grammar is built the same way.
    let grammar = agent_loop::action_schema_for_test(&schema(), RenderOpts::new(Detail::Short, true));
    assert_eq!(grammar_names(&grammar), registry, "worker grammar for the local helper");
}
