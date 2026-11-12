//! The structured-JSON protocol every agent speaks for worker and orchestrator jobs.
//!
//! A job is one prompt and one JSON schema; the answer is one JSON object. Every backend can take
//! one: an OpenAI-compatible server (`response_format` set to the schema), a CLI agent (a fresh
//! session per job, the schema spelled out in the prompt), or the local helper (grammar-constrained
//! by the schema).
//!
//! A job with tools ([`crate::tools::run_agent_json`]) is the same protocol with the tool registry
//! in reach: the model may look things up first and still ends in the same JSON object. This
//! module is how a single answer is asked for and read, and what that loop is built from.
//!
//! Spreading jobs over agents, slots and retries is [`crate::agents::AgentPool`]'s work.

use std::time::Duration;

use serde_json::{Value, json};

use crate::Error;
use crate::chat::ChatBackend;

/// Ask one backend for one JSON object, with no tools. Works on every [`ChatBackend`].
pub async fn ask_json(
    backend: &mut ChatBackend,
    prompt: &str,
    schema: &Value,
    max_tokens: usize,
    timeout_s: u64,
) -> Result<Value, Error> {
    let text = match backend {
        ChatBackend::Server { url, model, api_key, .. } => {
            server_json(url, model, api_key, prompt, schema, max_tokens, timeout_s).await?
        }
        ChatBackend::Local { helper, think, .. } => {
            helper.complete_full(prompt, Some(schema.clone()), max_tokens, *think).await?
        }
        ChatBackend::Cli(agent) => agent.complete(&cli_prompt(prompt, schema)).await?,
    };
    parse_object(&text)
}

pub(crate) fn cli_prompt(prompt: &str, schema: &Value) -> String {
    format!(
        "{prompt}\n\nReply ONLY with one JSON object matching this JSON schema (no prose, no code fences):\n{}",
        serde_json::to_string(schema).unwrap_or_default()
    )
}

/// Pull the JSON object out of an answer (fences, prose and thinking tags tolerated).
pub fn parse_object(text: &str) -> Result<Value, Error> {
    let body = match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    };
    if let Ok(v) = serde_json::from_str::<Value>(body.trim())
        && v.is_object()
    {
        return Ok(v);
    }
    let obj = crate::cliagent::CliAgent::extract_json_object(body)?;
    serde_json::from_str(&obj).map_err(|e| Error::Invalid(format!("the answer is not JSON: {e}")))
}

pub(crate) async fn server_json(
    url: &str,
    model: &str,
    api_key: &str,
    prompt: &str,
    schema: &Value,
    max_tokens: usize,
    timeout_s: u64,
) -> Result<String, Error> {
    let messages = [json!({ "role": "user", "content": prompt })];
    let msg = server_complete(url, model, api_key, &messages, None, Some(schema), max_tokens, timeout_s).await?;
    Ok(msg["content"].as_str().unwrap_or_default().to_string())
}

/// One chat completion on an OpenAI-compatible server, returning the reply message. With `tools`
/// the model may answer with tool calls; with `schema` its content is held to the schema (the
/// two are not sent together: servers differ on whether they can).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn server_complete(
    url: &str,
    model: &str,
    api_key: &str,
    messages: &[Value],
    tools: Option<&Value>,
    schema: Option<&Value>,
    max_tokens: usize,
    timeout_s: u64,
) -> Result<Value, Error> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_s.max(10)))
        .build()
        .map_err(|e| Error::Invalid(e.to_string()))?;
    let mut body = json!({
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
        body["tool_choice"] = json!("auto");
    }
    if let Some(schema) = schema {
        body["response_format"] =
            json!({ "type": "json_schema", "json_schema": { "name": "answer", "strict": true, "schema": schema } });
    }
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    crate::llmlog::push("prompt", body.to_string());
    let mut req = client.post(format!("{}/v1/chat/completions", url.trim_end_matches('/'))).json(&body);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let resp = req.send().await.map_err(|e| Error::Agent(crate::agents::error_from_request(url, &e)))?;
    if !resp.status().is_success() {
        return Err(Error::Agent(crate::agents::error_from_response(url, model, resp).await));
    }
    let v: Value = resp.json().await.map_err(|e| {
        Error::Agent(crate::agents::AgentError::new(
            url,
            crate::agents::AgentErrorKind::BadAnswer,
            format!("the server's answer is not JSON: {e}"),
        ))
    })?;
    let msg = v["choices"][0]["message"].clone();
    crate::llmlog::push("answer", msg.to_string());
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_are_read_through_fences_and_thinking() {
        assert_eq!(parse_object("{\"a\":1}").unwrap()["a"], 1);
        assert_eq!(parse_object("<think>{\"x\":0}</think>\n```json\n{\"a\":2}\n```").unwrap()["a"], 2);
        assert_eq!(parse_object("Sure! {\"a\":3} hope that helps").unwrap()["a"], 3);
        assert!(parse_object("no json here").is_err());
    }

    #[tokio::test]
    async fn server_jobs_run_in_parallel_and_come_back_in_order() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { break };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let mut got = Vec::new();
                    loop {
                        let n = sock.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        got.extend_from_slice(&buf[..n]);
                        let text = String::from_utf8_lossy(&got);
                        if let Some(h) = text.find("\r\n\r\n") {
                            let len = text[..h]
                                .lines()
                                .find_map(|l| {
                                    l.to_lowercase()
                                        .strip_prefix("content-length:")
                                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                                })
                                .unwrap_or(0);
                            if got.len() >= h + 4 + len {
                                break;
                            }
                        }
                    }
                    let text = String::from_utf8_lossy(&got).to_string();
                    let n: i64 = text.split("JOB=").nth(1).and_then(|r| r[..1].parse().ok()).unwrap_or(-1);
                    tokio::time::sleep(Duration::from_millis(if n == 0 { 150 } else { 10 })).await;
                    let content = json!({ "n": n }).to_string();
                    let body = json!({ "choices": [{ "message": { "content": content } }] }).to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        use crate::agents::{AgentJob, AgentPool, AgentSpec, AgentTarget, EndpointKey, ModelRunner, PoolAgent, SlotRegistry};
        let url = format!("http://{addr}");
        let agent = PoolAgent {
            spec: AgentSpec { id: "srv".into(), max_concurrent: 3, ..AgentSpec::default() },
            key: EndpointKey::Server { url: url.clone(), model: String::new() },
            capacity: 3,
            target: AgentTarget::Server { url, model: String::new(), api_key: String::new(), timeout_s: 10 },
            describe: "srv".into(),
            max_tool_rounds: 0,
        };
        let pool = AgentPool::new(vec![agent], 3, SlotRegistry::new(), std::sync::Arc::new(ModelRunner::default()));
        let jobs = (0..4)
            .map(|i| AgentJob {
                label: format!("j{i}"),
                prompt: format!("JOB={i}"),
                schema: json!({"type":"object"}),
                max_tokens: 64,
                tools: None,
            })
            .collect();
        let mut order = Vec::new();
        let res = pool
            .run_jobs(jobs, None, &mut |p| {
                if let Some(r) = p.finished {
                    order.push(r.index)
                }
            })
            .await
            .unwrap();
        assert_eq!(res.iter().map(|r| r.index).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        for r in &res {
            assert_eq!(r.value.as_ref().unwrap()["n"], r.index as i64);
        }
        // Job 0 was slowest, so it finished after a later one: they really overlapped.
        assert_ne!(order[0], 0);
    }
}
