//! A turn's review, end to end on each backend: whatever the editor delivers (its final answer, a
//! forced final draft, a version it saved itself, a script that cannot be used) is reviewed and
//! fed back the same way, by `repair::deliver`. The editors here are scripted stand-ins: a fake
//! OpenAI-style server, a fake local helper process and a fake coding CLI.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::projects::NewProject;

/// One interview: two answers with the interviewer's question (off the microphone) between them.
/// A cut over 1–14 s plays the question and goes back to the editor; 1–6 s reads clean.
fn interview(tmp: &Path) -> (Db, Project) {
    let mut db = Db::open_in_memory().unwrap();
    let project = db.create_project(&NewProject::named("Review")).unwrap();
    let folder = db.add_folder(project.id, tmp, true).unwrap();
    let file = tmp.join("kara.mp4");
    std::fs::write(&file, b"x").unwrap();
    db.conn
        .execute(
            "INSERT INTO videos(id, content_hash, size, duration_s, category) VALUES (7, 'h', 1, 60.0, 'interview')",
            [],
        )
        .unwrap();
    db.conn
        .execute(
            "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (7, ?1, ?2, 1, 0, 0)",
            params![folder.id, file.to_str().unwrap()],
        )
        .unwrap();
    db.conn
        .execute(
            "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
             (7, 1.0, 6.0, 'Every event we have come to, we walked away with a new relationship.', 0),
             (7, 6.5, 8.5, 'What made you want to sponsor the bar?', 1),
             (7, 9.0, 14.0, 'We love this hotel and the people who come here every month.', 0)",
            [],
        )
        .unwrap();
    crate::sentence_split::mirror_segments(&db, 7);
    (db, project)
}

fn cut(in_s: f64, out_s: f64) -> Value {
    json!({ "title": "t", "beats": [{ "id": "b1", "purpose": "p", "narration": "",
        "clips": [{ "video_id": 7, "in_s": in_s, "out_s": out_s, "audio": "source" }] }] })
}

fn final_action(script: Value) -> String {
    json!({ "action": "final", "script": script }).to_string()
}

/// A script naming a sentence that does not exist: it cannot be resolved to seconds.
fn unknown_sentence() -> Value {
    json!({ "title": "t", "beats": [{ "id": "b1", "purpose": "p", "narration": "",
        "clips": [{ "video_id": 7, "from": "s99999", "to": "s99999", "audio": "source" }] }] })
}

/// The spoken clip of the saved one-beat cut.
fn spoken(s: &Script) -> (f64, f64) {
    let c: Vec<_> = s.beats.iter().flat_map(|b| &b.clips).filter(|c| c.audio == crate::script::Audio::Source).collect();
    assert_eq!(c.len(), 1, "{s:?}");
    (c[0].in_s, c[0].out_s)
}

fn context(db: Db, tmp: &Path, backend: ChatBackend, max_tool_rounds: u32) -> ChatContext {
    ChatContext {
        db,
        paths: crate::paths::Paths::in_dir(tmp),
        backend,
        embedder: None,
        system_prompt: None,
        max_tool_rounds,
        script: crate::config::ScriptConfig::default(),
        jev: Default::default(),
        cancel: None,
    }
}

// ---- a fake coding CLI -----------------------------------------------------------------------

/// An `agy` stand-in: each call prints the next answer (in agy's JSON envelope) and logs the
/// prompt it was given. An answer may also write an MCP event line (a tool the CLI ran through
/// `ghostreel mcp`). agy is handed the prompt as written; `opencode` gets it rewritten as a brief.
#[cfg(unix)]
fn fake_cli(dir: &Path, answers: &[(String, Option<Value>)]) -> (ChatBackend, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let answers_file = dir.join("answers.json");
    let list: Vec<Value> = answers.iter().map(|(text, event)| json!({ "text": text, "event": event })).collect();
    std::fs::write(&answers_file, serde_json::to_string(&list).unwrap()).unwrap();
    let log = dir.join("prompts.jsonl");
    let counter = dir.join("count");
    let script = dir.join("fake-agy");
    std::fs::write(
        &script,
        format!(
            r#"#!/usr/bin/env python3
import sys, json, os
answers = json.load(open({answers:?}))
try:
    n = int(open({counter:?}).read())
except Exception:
    n = 0
open({counter:?}, 'w').write(str(n + 1))
open({log:?}, 'a').write(json.dumps(' '.join(sys.argv[1:])) + '\n')
a = answers[n] if n < len(answers) else {{'text': json.dumps({{'action': 'reply', 'text': 'out of answers'}}), 'event': None}}
if a.get('event') and os.environ.get('GHOSTREEL_MCP_EVENTS_FILE'):
    open(os.environ['GHOSTREEL_MCP_EVENTS_FILE'], 'a').write(json.dumps(a['event']) + '\n')
print(json.dumps({{'response': a['text']}}))
"#,
            answers = answers_file.to_str().unwrap(),
            counter = counter.to_str().unwrap(),
            log = log.to_str().unwrap(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let cfg = crate::config::CliAgentConfig {
        tool: "agy".into(),
        command: script.to_str().unwrap().into(),
        timeout_secs: 60,
        ..Default::default()
    };
    (ChatBackend::Cli(crate::cliagent::CliAgent::new(cfg)), log)
}

fn prompts(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str::<String>(l).unwrap())
        .collect()
}

/// A CLI that runs out of rounds is asked for the script at the end. That forced final draft used
/// to be saved as it came, never read back; now it is reviewed like any answer.
#[cfg(unix)]
#[tokio::test]
async fn a_cli_forced_final_draft_is_reviewed() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let (backend, log) = fake_cli(
        tmp.path(),
        &[
            (json!({ "action": "tool", "tool": "get_outline", "args": {} }).to_string(), None),
            (final_action(cut(1.0, 14.0)), None),
            (final_action(cut(1.0, 6.0)), None),
        ],
    );
    let mut ctx = context(db, tmp.path(), backend, 1);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await.unwrap();
    let asked = prompts(&log);
    assert_eq!(asked.len(), 3, "research, the forced final, one review: {asked:#?}");
    assert!(asked[2].contains("REVIEW 1 of 2") && asked[2].contains("off-mic"), "{}", asked[2]);
    let (_, out) = spoken(res.script.as_ref().expect("a cut"));
    assert!(out <= 7.0, "the revision is saved, not the forced draft: ends at {out}");
}

/// A CLI that saved its edit itself through MCP (save_script) delivered that version. It used to be
/// re-shaped and checked, never read back; now it is reviewed and the revision saved.
#[cfg(unix)]
#[tokio::test]
async fn a_version_the_cli_saved_itself_is_reviewed() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let bad: Script = serde_json::from_value(cut(1.0, 14.0)).unwrap();
    let saved = crate::script::save_version(&db, project.id, &bad, None).unwrap();
    let event = json!({ "kind": "tool_finished", "tool": "save_script", "summary": "saved", "script_id": saved });
    let (backend, log) = fake_cli(
        tmp.path(),
        &[
            (json!({ "action": "reply", "text": "Saved the cut." }).to_string(), Some(event)),
            (final_action(cut(1.0, 6.0)), None),
        ],
    );
    let mut ctx = context(db, tmp.path(), backend, 0);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await.unwrap();
    let asked = prompts(&log);
    assert_eq!(asked.len(), 2, "{asked:#?}");
    assert!(asked[1].contains("REVIEW 1 of 2"), "{}", asked[1]);
    assert_ne!(res.script_id, Some(saved), "the reviewed revision is a new version");
    let (_, out) = spoken(res.script.as_ref().expect("a cut"));
    assert!(out <= 7.0, "ends at {out}");
}

/// A CLI that saved its edit through MCP and then said so in plain words (no JSON at all) delivered
/// that version. The words used to end the whole turn as an error (the answer has to hold a JSON
/// object), so the saved cut was never reviewed or delivered.
#[cfg(unix)]
#[tokio::test]
async fn a_cli_that_saved_and_answered_in_plain_words_delivers_the_saved_version() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let bad: Script = serde_json::from_value(cut(1.0, 14.0)).unwrap();
    let saved = crate::script::save_version(&db, project.id, &bad, None).unwrap();
    let event = json!({ "kind": "tool_finished", "tool": "save_script", "summary": "saved", "script_id": saved });
    let (backend, log) = fake_cli(
        tmp.path(),
        &[
            ("I saved the cut: it opens on the relationship line.".to_string(), Some(event)),
            (final_action(cut(1.0, 6.0)), None),
        ],
    );
    let mut ctx = context(db, tmp.path(), backend, 0);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await;
    let res = res.expect("the saved version is delivered, not an error");
    let asked = prompts(&log);
    assert_eq!(asked.len(), 2, "{asked:#?}");
    assert!(asked[1].contains("REVIEW 1 of 2"), "the saved cut is reviewed: {}", asked[1]);
    let (_, out) = spoken(res.script.as_ref().expect("a cut"));
    assert!(out <= 7.0, "ends at {out}");
}

/// The same with words that happen to hold braces but no script ("{saved}"): the saved version, not
/// an unusable answer sent back as "your script could not be used".
#[cfg(unix)]
#[tokio::test]
async fn a_cli_that_saved_and_answered_with_braces_but_no_script_delivers_the_saved_version() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let good: Script = serde_json::from_value(cut(1.0, 6.0)).unwrap();
    let saved = crate::script::save_version(&db, project.id, &good, None).unwrap();
    let event = json!({ "kind": "tool_finished", "tool": "save_script", "summary": "saved", "script_id": saved });
    let (backend, log) = fake_cli(tmp.path(), &[(json!({ "status": "saved", "id": saved }).to_string(), Some(event))]);
    let mut ctx = context(db, tmp.path(), backend, 0);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await.unwrap();
    let asked = prompts(&log);
    assert!(asked.iter().all(|p| !p.contains("could not be used")), "{asked:#?}");
    let (_, out) = spoken(res.script.as_ref().expect("the saved cut"));
    assert!(out <= 7.0, "ends at {out}");
}

/// A CLI's script that cannot be resolved goes back with the reason, as on the server backend; it
/// used to end the turn.
#[cfg(unix)]
#[tokio::test]
async fn a_cli_script_that_cannot_be_used_goes_back_with_the_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let (backend, log) =
        fake_cli(tmp.path(), &[(final_action(unknown_sentence()), None), (final_action(cut(1.0, 6.0)), None)]);
    let mut ctx = context(db, tmp.path(), backend, 0);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await;
    let res = res.expect("the turn goes on");
    let asked = prompts(&log);
    assert_eq!(asked.len(), 2, "{asked:#?}");
    assert!(asked[1].contains("could not be used") && asked[1].contains("s99999"), "{}", asked[1]);
    assert!(res.script_id.is_some());
}

// ---- a fake local helper ---------------------------------------------------------------------

/// A local model's script that cannot be resolved goes back with the reason; it used to end the
/// turn (`resolve_script(...)?` in the local loop).
#[cfg(unix)]
#[tokio::test]
async fn a_local_script_that_cannot_be_used_goes_back_with_the_reason() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let answers = json!([final_action(unknown_sentence()), final_action(cut(1.0, 6.0))]);
    let log = tmp.path().join("prompts.jsonl");
    let helper_path = tmp.path().join("helper");
    std::fs::write(
        &helper_path,
        format!(
            r#"#!/usr/bin/env python3
import sys, json
answers = json.loads({answers:?})
n = 0
print(json.dumps({{'ready': True, 'ctx': 32768, 'token_count': True, 'model': 'test-model'}}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if r['cmd'] == 'count_tokens':
        out = {{'tokens': len(r.get('prompt', '')) // 4}}
    else:
        open({log:?}, 'a').write(json.dumps(r.get('prompt', '')) + '\n')
        text = answers[n] if n < len(answers) else answers[-1]
        n += 1
        out = {{'content': text, 'gen_tokens': 10, 'generation_seconds': 0.1, 'model': 'test-model'}}
    out.update(id=r['id'], ok=True)
    print(json.dumps(out), flush=True)
"#,
            answers = answers.to_string(),
            log = log.to_str().unwrap(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&helper_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let helper = crate::vision::LocalLlm::start(&crate::vision::LocalModels {
        helper: helper_path,
        vision: Some(("model".into(), Some("projector".into()))),
        embed: None,
        cpu: true,
        runtime: Default::default(),
        concurrency: 1,
        free_ghostpen: Vec::new(),
    })
    .await
    .unwrap();
    let backend = ChatBackend::Local { helper: Box::new(helper), ctx_tokens: 32768, think: false };
    let mut ctx = context(db, tmp.path(), backend, 3);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await;
    let res = res.expect("the turn goes on");
    let asked = prompts(&log);
    assert_eq!(asked.len(), 2, "the first answer, then the one fixing it: {}", asked.len());
    assert!(asked[1].contains("could not be used") && asked[1].contains("s99999"));
    let (_, out) = spoken(res.script.as_ref().expect("a cut"));
    assert!(out <= 7.0, "ends at {out}");
}

// ---- a fake server ---------------------------------------------------------------------------

/// An OpenAI-style server answering each request with the next response, recording the requests.
async fn fake_server(responses: Vec<Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        for body in responses {
            let Ok((mut socket, _)) = listener.accept().await else { return };
            let mut buf = vec![0u8; 8192];
            let mut total = 0;
            loop {
                if total == buf.len() {
                    buf.resize(buf.len() * 2, 0);
                }
                let n = socket.read(&mut buf[total..]).await.unwrap();
                if n == 0 {
                    break;
                }
                total += n;
                let s = String::from_utf8_lossy(&buf[..total]);
                if let Some(head) = s.find("\r\n\r\n") {
                    let len = s
                        .to_lowercase()
                        .find("content-length:")
                        .map(|i| s[i + 15..].split("\r\n").next().unwrap().trim().parse::<usize>().unwrap())
                        .unwrap_or(0);
                    if total >= head + 4 + len {
                        break;
                    }
                }
            }
            log.lock().unwrap().push(String::from_utf8_lossy(&buf[..total]).to_string());
            let body = body.to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            let _ = socket.shutdown().await;
        }
    });
    (url, seen)
}

fn said(content: &str) -> Value {
    json!({ "choices": [{ "message": { "role": "assistant", "content": content } }] })
}

/// A server model that saved its edit with save_script delivered that version. It used to be
/// re-shaped and checked, never read back; now it is reviewed and the revision saved.
#[tokio::test]
async fn a_version_the_server_model_saved_itself_is_reviewed() {
    let tmp = tempfile::tempdir().unwrap();
    let (db, project) = interview(tmp.path());
    let save_call = json!({ "choices": [{ "message": { "role": "assistant", "content": null, "tool_calls": [{
        "id": "c1", "type": "function",
        "function": { "name": "save_script", "arguments": json!({ "script": cut(1.0, 14.0) }).to_string() }
    }] } }] });
    let (url, seen) = fake_server(vec![save_call, said("Saved."), said(&cut(1.0, 6.0).to_string())]).await;
    let backend = ChatBackend::Server { url, model: "m".into(), api_key: String::new(), ctx_tokens: 32768 };
    let mut ctx = context(db, tmp.path(), backend, 0);
    let res = run_turn(&mut ctx, project.id, None, "a cut about the bar", &[], None, &mut |_| {}).await.unwrap();
    let requests = seen.lock().unwrap().clone();
    assert_eq!(requests.len(), 3, "research, the save, one review");
    assert!(requests[2].contains("REVIEW 1 of 2"), "the third request carries the review note");
    let (_, out) = spoken(res.script.as_ref().expect("a cut"));
    assert!(out <= 7.0, "the revision is saved, not the tool-saved version: ends at {out}");
}
