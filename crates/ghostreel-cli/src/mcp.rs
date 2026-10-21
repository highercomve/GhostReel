//! MCP server (`ghostreel mcp`): the same operations the CLI does, as tools an agent can call
//! over stdio — projects and folders, indexing, search, transcripts and keyframes, scripts
//! (draft with the model, save, export).
//!
//! Protocol: JSON-RPC 2.0, one message per line on stdin/stdout (MCP stdio transport). Logs go
//! to stderr so they never corrupt the stream.

use std::io::Write as _;
use std::path::PathBuf;
use std::str::FromStr as _;

use anyhow::Context;
use ghostreel_core::config::Config;
use ghostreel_core::db::Db;
use ghostreel_core::paths::Paths;
use ghostreel_core::projects::{NewProject, PipelineConfig};
use ghostreel_core::{doctor, index, runtime, script};
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
                            "instructions": INSTRUCTIONS,
                        }
                    })),
                    "ping" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": {} })),
                    "tools/list" => send(&json!({ "jsonrpc": "2.0", "id": id, "result": { "tools": tools() } })),
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

const INSTRUCTIONS: &str = "GhostReel indexes folders of video (speech transcripts, keyframe \
descriptions, embeddings) and turns clips into an edit. Typical flow: create_project → add_folder \
→ index (repeat until it reports no pending work) → learn the footage → write the script → \
preview_script / export_script.

You can edit without GhostReel's own model: list_videos gives the library at a glance (interviews \
vs b-roll, speech, camera), get_transcript reads what people said (several videos at once, \
interviewer lines marked or dropped), search finds moments by what is said or visible, get_video \
lists a clip's keyframe descriptions and get_frames shows you the actual pictures; \
get_steadiness tells you whether the stretch you picked is shaky and where the nearest steady one \
is. Then write the script yourself with save_script (dry_run first to check it; it warns about \
clips on shaky stretches); a beat's `bed` lays an interview's \
audio under muted b-roll (a J-cut). draft_script instead hands the whole edit to GhostReel's local \
chat model, which needs that model to fit on this machine. Clip in_s/out_s must lie inside real \
footage; save_script reports issues instead of silently fixing them.";

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

/// The AI settings as the MCP tools report them: one entry per capability, plus which
/// coding-agent CLIs this machine can actually run.
fn settings_view(cfg: &ghostreel_core::config::Config) -> Value {
    let llm = |c: &ghostreel_core::config::VisionConfig| {
        json!({
            "backend": c.backend.to_string(),
            "url": c.url,
            "model": c.model,
            "local_model": c.local_model,
            "ctx_tokens": c.ctx_tokens,
            "kv_cache": c.kv_cache,
            "flash_attn": c.flash_attn,
            "think": c.think,
            "cli": {
                "tool": c.cli.tool,
                "command": c.cli.command,
                "model": c.cli.model,
                "concurrency": c.cli.concurrency,
                "timeout_secs": c.cli.timeout_secs,
                "installed": ghostreel_core::cliagent::CliAgent::new(c.cli.clone()).available().is_some(),
            },
        })
    };
    let agents: Vec<Value> = ghostreel_core::config::CLI_TOOLS
        .iter()
        .map(|tool| {
            let probe = ghostreel_core::config::CliAgentConfig {
                tool: (*tool).to_string(),
                ..ghostreel_core::config::CliAgentConfig::default()
            };
            json!({
                "tool": tool,
                "path": ghostreel_core::cliagent::find_binary(&probe).map(|p| p.display().to_string()),
            })
        })
        .collect();
    json!({
        "vision": llm(&cfg.vision),
        "chat_model": llm(&cfg.chat_model()),
        "stt": { "backend": cfg.stt.backend.to_string(), "url": cfg.stt.url, "model": cfg.stt.model },
        "embed": { "backend": cfg.embed.backend.to_string(), "url": cfg.embed.url, "model": cfg.embed.model },
        "frames": { "max_interval_s": cfg.frames.max_interval_s },
        "cli_agents": agents,
    })
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required, "additionalProperties": false })
}

fn tools() -> Vec<Value> {
    let project = json!({ "type": "string", "description": "Project name" });
    vec![
        json!({
            "name": "doctor",
            "description": "What GhostReel would use on this machine: ffmpeg, GPU, database, models, AI backends.",
            "inputSchema": obj(json!({}), &[]),
        }),
        json!({
            "name": "list_projects",
            "description": "Every project with its sequence settings and indexing progress.",
            "inputSchema": obj(json!({}), &[]),
        }),
        json!({
            "name": "create_project",
            "description": "Create a project. fps/width/height are the timeline settings used by exports.",
            "inputSchema": obj(json!({
                "name": project,
                "description": { "type": "string" },
                "fps": { "type": "number", "description": "e.g. 25, 30, 29.97 (default 25)" },
                "width": { "type": "integer", "description": "default 1920" },
                "height": { "type": "integer", "description": "default 1080" },
            }), &["name"]),
        }),
        json!({
            "name": "add_folder",
            "description": "Watch a folder of footage for a project (whole tree by default). Editor render/cache folders are skipped.",
            "inputSchema": obj(json!({
                "project": project,
                "path": { "type": "string", "description": "Absolute path of the folder" },
                "recursive": { "type": "boolean", "description": "default true" },
            }), &["project", "path"]),
        }),
        json!({
            "name": "index",
            "description": "Run pending indexing work (scan, transcribe, keyframes, descriptions, embeddings) for up to max_seconds, then report progress. Resumable: call again while anything is pending.",
            "inputSchema": obj(json!({
                "project": project,
                "max_seconds": { "type": "integer", "description": "Stop after this long (default 900)" },
                "redo": { "type": "string", "description": "Reset this stage and later ones first: frames | transcribe | describe | embed" },
            }), &["project"]),
        }),
        json!({
            "name": "index_status",
            "description": "Indexing progress per stage, and per-video state.",
            "inputSchema": obj(json!({ "project": project, "videos": { "type": "boolean", "description": "Include the video list" } }), &["project"]),
        }),
        json!({
            "name": "list_videos",
            "description": "A project's videos, one compact row each: id, file, duration, category (interview, broll, drone…), camera style, speech segments, keyframes and what the first keyframe shows. Start here to learn the footage.",
            "inputSchema": obj(json!({
                "project": project,
                "category": { "type": "string", "description": "Only this category, e.g. interview or broll" },
                "speech": { "type": "boolean", "description": "true: only videos with speech; false: only videos without" },
            }), &["project"]),
        }),
        json!({
            "name": "search",
            "description": "Search a project's footage by meaning and keywords. Returns {mode, hits}: moments with video_id and timestamps to cut from. mode says whether meaning search ran; when the embedding model is unavailable it falls back to keywords and says so.",
            "inputSchema": obj(json!({
                "project": project,
                "query": { "type": "string" },
                "limit": { "type": "integer", "description": "default 10" },
                "kinds": {
                    "type": "array",
                    "items": { "type": "string", "enum": ["moment", "transcript", "frame"] },
                    "description": "moment: what is visible (keyframe description) plus nearby speech; transcript: speech only; frame: on-screen text only. Default: all",
                },
                "keywords_only": { "type": "boolean", "description": "Skip the embedding model" },
            }), &["project", "query"]),
        }),
        json!({
            "name": "get_transcript",
            "description": "Timestamped speech of one or more videos, optionally limited to a range. Lines spoken off the subject's mic (usually the interviewer) are marked [off], or dropped with speech_only.",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "video_ids": { "type": "array", "items": { "type": "integer" }, "description": "Several videos at once (instead of video_id)" },
                "start_s": { "type": "number" },
                "end_s": { "type": "number" },
                "speech_only": { "type": "boolean", "description": "default false: drop off-mic (interviewer) lines" },
                "format": { "type": "string", "enum": ["text", "json"], "description": "default text: one 'start-end [off] text' line per segment" },
            }), &[]),
        }),
        json!({
            "name": "get_steadiness",
            "description": "Is the camera shaking in a stretch of a video? From the motion measured at indexing: camera style, worst shake (jerk) and sway against the limits in use, the shaky stretches inside the range, and a verdict (steady / some shake / shaky: over half the range). When the range is shaky, the nearest steady stretch of the same length. Without a range, the whole video.",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "start_s": { "type": "number" },
                "end_s": { "type": "number" },
                "windows": { "type": "boolean", "description": "default false: also list every measured window in the range" },
            }), &["video_id"]),
        }),
        json!({
            "name": "get_frames",
            "description": "Look at a video: returns the pictures themselves (JPEG) with their times and descriptions. Without times, keyframes spread across the whole video; with times, the keyframe nearest each time, or the exact frame when exact=true (decodes the source, which must be reachable).",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "times": { "type": "array", "items": { "type": "number" }, "description": "Seconds into the video (at most 8)" },
                "limit": { "type": "integer", "description": "Without times: how many keyframes (default 6, at most 8)" },
                "exact": { "type": "boolean", "description": "default false: decode the exact frame instead of the nearest keyframe" },
            }), &["video_id"]),
        }),
        json!({
            "name": "get_video",
            "description": "A video's metadata and its keyframes with descriptions and on-screen text (what is visible when).",
            "inputSchema": obj(json!({
                "video_id": { "type": "integer" },
                "limit": { "type": "integer", "description": "Max keyframes to return (default 40)" },
            }), &["video_id"]),
        }),
        json!({
            "name": "list_scripts",
            "description": "Script versions of a project (newest first).",
            "inputSchema": obj(json!({ "project": project }), &["project"]),
        }),
        json!({
            "name": "get_script",
            "description": "One script version: beats, clips, narration, on-screen text.",
            "inputSchema": obj(json!({ "script_id": { "type": "integer" } }), &["script_id"]),
        }),
        json!({
            "name": "draft_script",
            "description": "Ask GhostReel's editing model to draft or revise a script from the project's footage (it searches and reads transcripts itself). Slow: minutes. Returns the saved version and its issues.",
            "inputSchema": obj(json!({
                "project": project,
                "message": { "type": "string", "description": "e.g. 'a 60 second promo using the interviews, slower cuts'" },
                "session_id": { "type": "integer", "description": "Continue an earlier chat session (revisions)" },
            }), &["project", "message"]),
        }),
        json!({
            "name": "save_script",
            "description": "Save your own script as a new version. Clips must lie inside real footage; returns the issues found (errors mean the clip won't work). Beats play in order and a beat's clips play one after another. With a bed, the bed's audio runs under the whole beat and its clips should be audio \"mute\" and add up to the bed's length: an interview answer over b-roll.",
            "inputSchema": obj(json!({
                "project": project,
                "script": { "type": "object", "description": "{title, target_duration_s?, beats:[{id, purpose, narration?, on_screen_text?, notes?, bed?:{video_id, in_s, out_s, why?}, clips:[{video_id, in_s, out_s, audio?: \"source\"|\"mute\", why?}]}]}" },
                "dry_run": { "type": "boolean", "description": "default false: only check it and report the issues, save nothing" },
            }), &["project", "script"]),
        }),
        json!({
            "name": "export_script",
            "description": "Export a script version as an editable timeline: fcp_xml for Premiere Pro, or otio.",
            "inputSchema": obj(json!({
                "script_id": { "type": "integer" },
                "format": { "type": "string", "enum": ["fcp_xml", "otio"], "description": "default fcp_xml" },
                "out": { "type": "string", "description": "Output file path" },
            }), &["script_id", "out"]),
        }),
        json!({
            "name": "preview_script",
            "description": "Render a script version to a preview MP4 and return its path (and duration/encoder). Slow: it cuts and encodes the real footage. Optionally burn in the on-screen titles and the narration as subtitles.",
            "inputSchema": obj(json!({
                "script_id": { "type": "integer" },
                "burn_titles": { "type": "boolean", "description": "default false" },
                "burn_narration": { "type": "boolean", "description": "default false: narration as burnt-in subtitles" },
                "normalize_audio": { "type": "boolean", "description": "default false: bring every clip to the same loudness" },
                "out": { "type": "string", "description": "Output file path; default is the data dir's preview folder" },
            }), &["script_id"]),
        }),
        json!({
            "name": "get_settings",
            "description": "The AI settings: for speech, frame descriptions, the script chat and embeddings — which backend each uses (auto/local/server/cli), its server URL and model, the local context window, KV cache, flash attention and whether the model reasons before answering, plus which coding-agent CLIs are installed.",
            "inputSchema": obj(json!({}), &[]),
        }),
        json!({
            "name": "set_settings",
            "description": "Change AI settings, the same keys `ghostreel config set` takes, e.g. {\"chat_model.backend\": \"local\", \"chat_model.think\": \"true\", \"chat_model.ctx_tokens\": \"16384\"}. Sections: vision (frame descriptions), chat_model (script chat), stt, embed. Returns the settings after the change.",
            "inputSchema": obj(json!({
                "set": { "type": "object", "description": "{\"<section>.<field>\": \"<value>\"} — values as strings" },
            }), &["set"]),
        }),
        json!({
            "name": "delete_project",
            "description": "Delete a project. With purge=true also delete its keyframes, preview renders and the transcripts/descriptions/vectors of footage no other project uses. Video files are never touched.",
            "inputSchema": obj(json!({
                "project": project,
                "purge": { "type": "boolean", "description": "default false: keep the index for reuse" },
            }), &["project"]),
        }),
        json!({
            "name": "set_video_role",
            "description": "Take a video out of a project's library: 'removed' (ignored everywhere) or 'reference' (a finished edit the script chat studies but never cuts from). 'footage' puts it back.",
            "inputSchema": obj(json!({
                "project": project,
                "video_id": { "type": "integer" },
                "role": { "type": "string", "enum": ["footage", "removed", "reference"] },
            }), &["project", "video_id", "role"]),
        }),
    ]
}

fn open_db(paths: &Paths) -> anyhow::Result<Db> {
    Db::open(&paths.db_file()).with_context(|| format!("cannot open {}", paths.db_file().display()))
}

fn arg_str(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .with_context(|| format!("missing string argument '{key}'"))
}

fn arg_i64(args: &Value, key: &str) -> anyhow::Result<i64> {
    args.get(key).and_then(|v| v.as_i64()).with_context(|| format!("missing integer argument '{key}'"))
}

fn project(db: &Db, args: &Value) -> anyhow::Result<ghostreel_core::projects::Project> {
    let name = arg_str(args, "project")?;
    Ok(db.require_project(&name)?)
}

/// Most tools answer JSON, sent as one text block; `get_frames` answers pictures.
async fn call_tool_content(paths: &Paths, name: &str, args: &Value) -> anyhow::Result<Vec<Value>> {
    match name {
        "get_frames" => frames_content(paths, args),
        "get_transcript" if args.get("format").and_then(|v| v.as_str()).unwrap_or("text") == "text" => {
            Ok(vec![json!({ "type": "text", "text": transcript_text(paths, args)? })])
        }
        _ => {
            let v = call_tool(paths, name, args).await?;
            Ok(tool_result(&v, false)["content"].as_array().cloned().unwrap_or_default())
        }
    }
}

/// Share of a clip that may be shaky before it counts as shaky, the line the chat pipeline uses
/// to move clips.
const SHAKY_SHARE: f64 = 0.5;

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// How steady `in_s..out_s` of a video is, judged with the script settings' limits; `None` when
/// its motion was never measured.
struct Steadiness {
    camera: &'static str,
    jerk_limit: f64,
    sway_limit: f64,
    worst_jerk: f64,
    worst_sway: f64,
    spans: Vec<(f64, f64)>,
    shaky_s: f64,
    /// When over half the range is shaky: start of the nearest steady stretch as long as the range.
    steady_start: Option<f64>,
    len: f64,
}

impl Steadiness {
    fn measure(
        windows: &[ghostreel_core::steadiness::Window],
        cfg: &ghostreel_core::config::ScriptConfig,
        in_s: f64,
        out_s: f64,
    ) -> Option<Self> {
        use ghostreel_core::steadiness as st;
        let inside: Vec<_> = windows.iter().filter(|w| w.end_s > in_s && w.start_s < out_s).collect();
        if inside.is_empty() {
            return None;
        }
        let jerk_limit = st::shake_limit(windows, cfg.max_shake_jerk, cfg.shake_relative);
        let spans = st::shaky_spans_with_sway(windows, in_s, out_s, jerk_limit, cfg.max_sway);
        // A fold from 0.0: an empty float `sum()` is -0.0.
        let shaky_s = spans.iter().fold(0.0, |acc, (a, b)| acc + (b - a));
        let len = (out_s - in_s).max(0.0);
        let steady_start = (len > 0.0 && shaky_s > len * SHAKY_SHARE)
            .then(|| st::steady_stretch_near(windows, in_s, len, jerk_limit, cfg.max_sway))
            .flatten();
        Some(Self {
            camera: st::camera_style(windows).as_str(),
            jerk_limit,
            sway_limit: cfg.max_sway,
            worst_jerk: inside.iter().map(|w| w.jerk).fold(0.0, f64::max),
            worst_sway: inside.iter().map(|w| w.sway).fold(0.0, f64::max),
            spans,
            shaky_s,
            steady_start,
            len,
        })
    }

    fn verdict(&self) -> &'static str {
        if self.shaky_s <= 0.0 {
            "steady"
        } else if self.shaky_s > self.len * SHAKY_SHARE {
            "shaky"
        } else {
            "some shake"
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "verdict": self.verdict(),
            "camera": self.camera,
            "shaky_s": round2(self.shaky_s),
            "shaky_spans": self.spans.iter().map(|(a, b)| [round2(*a), round2(*b)]).collect::<Vec<_>>(),
            "worst_jerk": round2(self.worst_jerk),
            "jerk_limit": round2(self.jerk_limit),
            "worst_sway": round2(self.worst_sway),
            "sway_limit": round2(self.sway_limit),
            "steady_alternative": self.steady_start.map(|s| [round2(s), round2(s + self.len)]),
        })
    }
}

/// Warnings for clips that sit mostly on shaky stretches, with where to move them.
fn shake_issues(db: &Db, cfg: &ghostreel_core::config::ScriptConfig, s: &script::Script) -> Vec<script::Issue> {
    let mut issues = Vec::new();
    if cfg.max_shake_jerk <= 0.0 {
        return issues;
    }
    for beat in &s.beats {
        for (i, c) in beat.clips.iter().enumerate() {
            let windows = db.motion_windows(c.video_id).unwrap_or_default();
            let Some(st) = Steadiness::measure(&windows, cfg, c.in_s, c.out_s) else { continue };
            if st.verdict() != "shaky" {
                continue;
            }
            let fix = match st.steady_start {
                Some(a) => format!("; nearest steady stretch {a:.1}–{:.1} s", a + st.len),
                None => "; no steady stretch that long in this video".to_string(),
            };
            issues.push(script::Issue {
                severity: script::IssueSeverity::Warning,
                beat_id: Some(beat.id.clone()),
                clip_index: Some(i),
                message: format!(
                    "shaky: video #{} {:.1}–{:.1} s shakes for {:.1} of {:.1} s{fix}",
                    c.video_id, c.in_s, c.out_s, st.shaky_s, st.len
                ),
            });
        }
    }
    issues
}

const MAX_FRAMES: usize = 8;
/// Long side of a decoded frame, the same size the keyframes are stored at.
const FRAME_LONG_SIDE: u32 = 768;

fn transcript_ids(args: &Value) -> anyhow::Result<Vec<i64>> {
    let mut ids: Vec<i64> = args
        .get("video_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default();
    if let Some(id) = args.get("video_id").and_then(|v| v.as_i64()) {
        ids.insert(0, id);
    }
    anyhow::ensure!(!ids.is_empty(), "give video_id or video_ids");
    Ok(ids)
}

/// The transcript segments asked for: in range, and without the interviewer if `speech_only`.
fn transcript_segments(db: &Db, args: &Value, video_id: i64) -> anyhow::Result<Vec<index::TranscriptSegment>> {
    let start = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let end = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
    let speech_only = args.get("speech_only").and_then(|v| v.as_bool()).unwrap_or(false);
    Ok(index::transcript(db, video_id)?
        .into_iter()
        .filter(|s| s.end > start && s.start < end && !(speech_only && s.off_mic))
        .collect())
}

fn transcript_text(paths: &Paths, args: &Value) -> anyhow::Result<String> {
    let db = open_db(paths)?;
    let mut out = String::new();
    for id in transcript_ids(args)? {
        let file = video_file_name(&db, id);
        out.push_str(&format!("=== video {id} ({file})\n"));
        let segs = transcript_segments(&db, args, id)?;
        if segs.is_empty() {
            out.push_str("(no speech)\n");
        }
        for s in segs {
            let off = if s.off_mic { " [off]" } else { "" };
            out.push_str(&format!("{:.2}-{:.2}{off} {}\n", s.start, s.end, s.text.trim()));
        }
    }
    Ok(out)
}

fn video_file_name(db: &Db, video_id: i64) -> String {
    db.conn
        .query_row("SELECT MIN(path) FROM video_files WHERE video_id = ?1", [video_id], |r| {
            r.get::<_, Option<String>>(0)
        })
        .ok()
        .flatten()
        .and_then(|p| PathBuf::from(p).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default()
}

/// The `description` field of a keyframe's description JSON.
fn frame_description(json: Option<&str>) -> Option<String> {
    json.and_then(|j| serde_json::from_str::<Value>(j).ok())
        .and_then(|v| v.get("description").and_then(|d| d.as_str()).map(str::to_string))
}

/// One frame of the source at `t_s`, decoded by ffmpeg and scaled to the keyframe size.
fn decode_frame(source: &std::path::Path, t_s: f64) -> anyhow::Result<Vec<u8>> {
    let ffmpeg = doctor::locate("ffmpeg").context("ffmpeg not found")?;
    let scale = format!("scale=w={FRAME_LONG_SIDE}:h={FRAME_LONG_SIDE}:force_original_aspect_ratio=decrease");
    let out = ghostreel_core::proc::std_command(ffmpeg)
        .args(["-v", "error", "-ss", &format!("{t_s:.3}"), "-i"])
        .arg(source)
        .args(["-frames:v", "1", "-vf", &scale, "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
        .output()
        .context("cannot run ffmpeg")?;
    anyhow::ensure!(
        out.status.success() && !out.stdout.is_empty(),
        "ffmpeg could not decode {} at {t_s:.2}s: {}",
        source.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(out.stdout)
}

fn frames_content(paths: &Paths, args: &Value) -> anyhow::Result<Vec<Value>> {
    use base64::Engine as _;
    let db = open_db(paths)?;
    let video_id = arg_i64(args, "video_id")?;
    let frames = index::frames(&db, &paths.data_dir, video_id)?;
    let exact = args.get("exact").and_then(|v| v.as_bool()).unwrap_or(false);
    let times: Vec<f64> = args
        .get("times")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|t| t.as_f64()).take(MAX_FRAMES).collect())
        .unwrap_or_default();

    // (time asked for, keyframe nearest to it)
    let picks: Vec<(f64, Option<&index::FrameRow>)> = if times.is_empty() {
        anyhow::ensure!(!frames.is_empty(), "video {video_id} has no keyframes yet; pass times with exact=true");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(6).clamp(1, MAX_FRAMES as u64) as usize;
        let step = frames.len().div_ceil(limit).max(1);
        frames.iter().step_by(step).map(|f| (f.t_s, Some(f))).collect()
    } else {
        times
            .iter()
            .map(|&t| (t, frames.iter().min_by(|a, b| (a.t_s - t).abs().total_cmp(&(b.t_s - t).abs()))))
            .collect()
    };

    let source = if exact { Some(index::video_by_id(&db, video_id, 0.0, 0.0, 0.0)?.path) } else { None };
    let mut content = Vec::new();
    for (t, nearest) in picks {
        let description = nearest.and_then(|f| frame_description(f.description.as_deref()));
        let (shown_t, jpeg) = match (&source, nearest) {
            (Some(src), _) => (t, decode_frame(src, t)?),
            (None, Some(f)) => {
                (f.t_s, std::fs::read(&f.path).with_context(|| format!("cannot read keyframe {}", f.path.display()))?)
            }
            (None, None) => anyhow::bail!("video {video_id} has no keyframes yet; use exact=true"),
        };
        let mut caption = format!("video {video_id} at {shown_t:.2}s");
        if let Some(f) = nearest {
            if exact || (f.t_s - t).abs() > 0.01 {
                caption.push_str(&format!(" (nearest keyframe {:.2}s)", f.t_s));
            }
            if let Some(d) = &description {
                caption.push_str(&format!(": {d}"));
            }
        }
        content.push(json!({ "type": "text", "text": caption }));
        content.push(json!({
            "type": "image",
            "data": base64::engine::general_purpose::STANDARD.encode(&jpeg),
            "mimeType": "image/jpeg",
        }));
    }
    Ok(content)
}

async fn call_tool(paths: &Paths, name: &str, args: &Value) -> anyhow::Result<Value> {
    match name {
        "doctor" => Ok(serde_json::to_value(doctor::run(paths).await)?),
        "list_projects" => {
            let db = open_db(paths)?;
            let mut out = Vec::new();
            for p in db.projects()? {
                let st = index::status(&db, Some(p.id))?;
                out.push(json!({ "project": p, "status": st }));
            }
            Ok(json!(out))
        }
        "create_project" => {
            let db = open_db(paths)?;
            let (fps_num, fps_den) = match args.get("fps").and_then(|v| v.as_f64()) {
                Some(f) => {
                    let fps = script::Fps::from_f64(f);
                    (fps.num, fps.den)
                }
                None => (25, 1),
            };
            let mut pipeline = PipelineConfig::default();
            if let Some(nt) = args.get("no_transcribe").and_then(|v| v.as_bool())
                && nt
            {
                pipeline.transcribe = false;
            }
            if let Some(t) = args.get("transcribe").and_then(|v| v.as_bool()) {
                pipeline.transcribe = t;
            }
            let p = NewProject {
                name: arg_str(args, "name")?,
                description: args.get("description").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                fps_num,
                fps_den,
                width: args.get("width").and_then(|v| v.as_i64()).unwrap_or(1920),
                height: args.get("height").and_then(|v| v.as_i64()).unwrap_or(1080),
                pipeline: Some(pipeline),
            };
            Ok(serde_json::to_value(db.create_project(&p)?)?)
        }
        "add_folder" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let path = PathBuf::from(arg_str(args, "path")?);
            let recursive = args.get("recursive").and_then(|v| v.as_bool()).unwrap_or(true);
            let folder = db.add_folder(p.id, &path, recursive)?;
            Ok(json!({ "folder": folder, "next": "call index to scan it" }))
        }
        "index" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let config = Config::load(&paths.config_file).unwrap_or_default();
            if let Some(stage) = args.get("redo").and_then(|v| v.as_str()) {
                anyhow::ensure!(
                    index::STAGES.contains(&stage),
                    "unknown stage '{stage}'; valid: {}",
                    index::STAGES.join(", ")
                );
                index::reset_stages(&db, &paths.data_dir, Some(p.id), stage)?;
            }
            let max_seconds = args.get("max_seconds").and_then(|v| v.as_u64()).unwrap_or(900);
            let rt = runtime::resolve(paths, &config).await?;
            let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let deadline = cancel.clone();
            let ticker = tokio::task::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(max_seconds)).await;
                deadline.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            let opts = index::Options { project_id: Some(p.id), cancel: Some(cancel), ..Default::default() };
            let summary = index::run(&mut db, &rt, &opts, |_| {}).await?;
            ticker.abort();
            let status = index::status(&db, Some(p.id))?;
            let pending: i64 = status.stages.iter().map(|s| s.pending).sum();
            Ok(json!({
                "summary": summary,
                "status": status,
                "pending_jobs": pending,
                "next": if pending > 0 { "call index again to continue" } else { "indexing complete" },
            }))
        }
        "index_status" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let status = index::status(&db, Some(p.id))?;
            let videos = if args.get("videos").and_then(|v| v.as_bool()).unwrap_or(false) {
                serde_json::to_value(index::videos(&db, Some(p.id))?)?
            } else {
                Value::Null
            };
            Ok(json!({ "status": status, "videos": videos }))
        }
        "search" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let query = arg_str(args, "query")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
            let kinds = args
                .get("kinds")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|k| k.as_str().map(str::to_string)).collect::<Vec<_>>());
            let mut embedder = None;
            let mut mode = json!("keywords (asked for)");
            if !args.get("keywords_only").and_then(|v| v.as_bool()).unwrap_or(false) {
                let config = Config::load(&paths.config_file).unwrap_or_default();
                let setup = runtime::resolve_embed(paths, &config).await;
                match runtime::start_embedder(&setup, |_, _| {}).await {
                    Ok(e) => {
                        embedder = Some(e);
                        mode = json!("meaning + keywords");
                    }
                    Err(why) => mode = json!(format!("keywords only: meaning search unavailable ({why})")),
                }
            }
            let opts =
                ghostreel_core::search::SearchOptions { project_id: Some(p.id), limit, kinds, ..Default::default() };
            let hits = ghostreel_core::search::search(&db, &paths.data_dir, &query, embedder.as_mut(), &opts).await?;
            Ok(json!({ "mode": mode, "hits": hits }))
        }
        "list_videos" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let category = args.get("category").and_then(|v| v.as_str());
            let speech = args.get("speech").and_then(|v| v.as_bool());
            let mut first_desc = db.conn.prepare(
                "SELECT description_json FROM frames
                  WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s LIMIT 1",
            )?;
            let cfg = Config::load(&paths.config_file).unwrap_or_default().script;
            let mut rows = Vec::new();
            for v in index::videos_with_shake(&db, Some(p.id), cfg.max_shake_jerk, cfg.shake_relative, cfg.max_sway)? {
                if category.is_some_and(|c| c != v.category) || speech.is_some_and(|s| s != (v.segments > 0)) {
                    continue;
                }
                let desc: Option<String> = first_desc.query_map([v.id], |r| r.get(0))?.next().transpose()?;
                let shows = frame_description(desc.as_deref()).map(|d| {
                    if d.chars().count() > 120 { format!("{}…", d.chars().take(120).collect::<String>()) } else { d }
                });
                rows.push(json!({
                    "id": v.id,
                    "file": v.path.file_name().map(|n| n.to_string_lossy().into_owned()),
                    "duration_s": v.duration_s.map(|d| (d * 10.0).round() / 10.0),
                    "category": v.category,
                    "tag": v.category_tag,
                    "camera": v.camera,
                    "shaky_s": v.steadiness_measured.then(|| (v.shaky_s * 10.0).round() / 10.0),
                    "speech_segments": v.segments,
                    "keyframes": v.frames,
                    "shows": shows,
                }));
            }
            Ok(json!({ "project": p.name, "videos": rows.len(), "rows": rows }))
        }
        "get_steadiness" => {
            let db = open_db(paths)?;
            let video_id = arg_i64(args, "video_id")?;
            let windows = db.motion_windows(video_id)?;
            let duration = windows.last().map(|w| w.end_s).unwrap_or(0.0);
            let start = args.get("start_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end = args.get("end_s").and_then(|v| v.as_f64()).unwrap_or(duration);
            anyhow::ensure!(end > start, "end_s must be after start_s");
            let cfg = Config::load(&paths.config_file).unwrap_or_default().script;
            let Some(st) = Steadiness::measure(&windows, &cfg, start, end) else {
                return Ok(json!({
                    "video_id": video_id,
                    "range": [start, end],
                    "verdict": "not measured",
                    "next": "the steadiness stage has not run for this video (index it), or the range is outside the video",
                }));
            };
            let mut out = json!({ "video_id": video_id, "range": [round2(start), round2(end)] });
            if let (Value::Object(o), Value::Object(r)) = (&mut out, st.to_json()) {
                o.extend(r);
            }
            if args.get("windows").and_then(|v| v.as_bool()).unwrap_or(false) {
                out["windows"] = json!(
                    windows
                        .iter()
                        .filter(|w| w.end_s > start && w.start_s < end)
                        .map(|w| json!({
                            "start_s": round2(w.start_s),
                            "end_s": round2(w.end_s),
                            "jerk": round2(w.jerk),
                            "sway": round2(w.sway),
                            "motion": round2(w.motion),
                        }))
                        .collect::<Vec<_>>()
                );
            }
            Ok(out)
        }
        "get_transcript" => {
            let db = open_db(paths)?;
            let ids = transcript_ids(args)?;
            if let [id] = ids[..] {
                return Ok(serde_json::to_value(transcript_segments(&db, args, id)?)?);
            }
            let mut out = serde_json::Map::new();
            for id in ids {
                out.insert(id.to_string(), serde_json::to_value(transcript_segments(&db, args, id)?)?);
            }
            Ok(Value::Object(out))
        }
        "get_video" => {
            let db = open_db(paths)?;
            let video_id = arg_i64(args, "video_id")?;
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(40) as usize;
            let frames = index::frames(&db, &paths.data_dir, video_id)?;
            let step = frames.len().div_ceil(limit.max(1)).max(1);
            let sampled: Vec<Value> = frames
                .iter()
                .step_by(step)
                .map(|f| {
                    let desc = frame_description(f.description.as_deref());
                    json!({ "t_s": f.t_s, "description": desc, "visible_text": f.visible_text })
                })
                .collect();
            let row = index::videos(&db, None)?.into_iter().find(|v| v.id == video_id);
            Ok(json!({ "video": row, "keyframes": sampled, "keyframes_total": frames.len() }))
        }
        "list_scripts" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            Ok(serde_json::to_value(script::list(&db, p.id)?)?)
        }
        "get_script" => {
            let db = open_db(paths)?;
            Ok(serde_json::to_value(script::load(&db, arg_i64(args, "script_id")?)?)?)
        }
        "draft_script" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let message = arg_str(args, "message")?;
            let session_id = args.get("session_id").and_then(|v| v.as_i64());
            let config = Config::load(&paths.config_file).unwrap_or_default();
            let vision = runtime::resolve_chat(paths, &config).await;
            let backend = ghostreel_core::chat::ChatBackend::from_vision_setup(&vision, config.script.server_timeout_s)
                .await?
                .with_window(config.chat_model().ctx_tokens);
            let embed_setup = runtime::resolve_embed(paths, &config).await;
            let embedder = runtime::start_embedder(&embed_setup, |_, _| {}).await.ok();
            let mut ctx = ghostreel_core::chat::ChatContext {
                db,
                data_dir: paths.data_dir.clone(),
                backend,
                embedder,
                system_prompt: Some(config.chat.system_prompt.clone()),
                max_tool_rounds: config.chat_model().max_tool_rounds,
                script: config.script.clone(),
                jev: config.jev.clone(),
                cancel: None,
            };
            let images: Vec<String> =
                args.get("images").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
            let turn =
                ghostreel_core::chat::run_turn(&mut ctx, p.id, session_id, &message, &images, &mut |_| {}).await?;
            Ok(serde_json::to_value(turn)?)
        }
        "save_script" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let raw = args.get("script").context("missing 'script' object")?;
            let mut s: script::Script =
                serde_json::from_value(raw.clone()).context("script JSON does not match the schema")?;
            s.fill_from_project(&p);
            let mut issues = script::validate(&db, p.id, &s)?;
            let cfg = Config::load(&paths.config_file).unwrap_or_default().script;
            issues.extend(shake_issues(&db, &cfg, &s));
            let errors = issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count();
            let dry_run = args.get("dry_run").and_then(|v| v.as_bool()).unwrap_or(false);
            let script_id = if dry_run { None } else { Some(script::save_version(&db, p.id, &s, None)?) };
            Ok(json!({
                "script_id": script_id,
                "issues": issues,
                "errors": errors,
                "total_duration_s": s.total_duration_s(),
                "clips": s.clip_count(),
            }))
        }
        "export_script" => {
            let db = open_db(paths)?;
            let script_id = arg_i64(args, "script_id")?;
            let format = args.get("format").and_then(|v| v.as_str()).unwrap_or("fcp_xml");
            let fmt = ghostreel_core::export::ExportFormat::from_str(format)?;
            let out = PathBuf::from(arg_str(args, "out")?);
            let res = ghostreel_core::export::export_script(&db, script_id, fmt, &out)?;
            Ok(json!({ "path": res.path, "format": format }))
        }
        "preview_script" => {
            let db = open_db(paths)?;
            let script_id = arg_i64(args, "script_id")?;
            let ffmpeg = doctor::locate("ffmpeg").context("ffmpeg not found")?;
            let opts = ghostreel_core::preview::PreviewOptions {
                burn_titles: args.get("burn_titles").and_then(|v| v.as_bool()).unwrap_or(false),
                burn_narration: args.get("burn_narration").and_then(|v| v.as_bool()).unwrap_or(false),
                normalize_audio: args.get("normalize_audio").and_then(|v| v.as_bool()).unwrap_or(false),
                audio_fade_s: ghostreel_core::config::Config::load(&paths.config_file)
                    .unwrap_or_default()
                    .script
                    .audio_fade_s,
                speech_overrun_s: ghostreel_core::config::Config::load(&paths.config_file)
                    .unwrap_or_default()
                    .script
                    .speech_overrun_s,
                out: args.get("out").and_then(|v| v.as_str()).map(PathBuf::from),
                cancel: None,
            };
            let res =
                ghostreel_core::preview::render_preview(&db, &paths.data_dir, &ffmpeg, script_id, &opts, |_, _| {})?;
            Ok(serde_json::to_value(res)?)
        }
        "get_settings" => {
            let cfg = ghostreel_core::config::Config::load(&paths.config_file).unwrap_or_default();
            Ok(settings_view(&cfg))
        }
        "set_settings" => {
            let mut cfg = ghostreel_core::config::Config::load(&paths.config_file).unwrap_or_default();
            let set = args
                .get("set")
                .and_then(|v| v.as_object())
                .context("set must be an object of \"section.field\": \"value\"")?;
            let mut applied = Vec::new();
            for (key, value) in set {
                // Numbers and booleans are as natural to write here as strings; take either.
                let value = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                cfg.set_key(key, &value).map_err(|e| anyhow::anyhow!("{e}"))?;
                applied.push(format!("{key} = {value}"));
            }
            cfg.save(&paths.config_file)?;
            Ok(json!({ "applied": applied, "settings": settings_view(&cfg) }))
        }
        "delete_project" => {
            let mut db = open_db(paths)?;
            let p = project(&db, args)?;
            let purge = args.get("purge").and_then(|v| v.as_bool()).unwrap_or(false);
            let stats = if purge { db.purge_project_data(&paths.data_dir, p.id)? } else { Default::default() };
            db.remove_project(p.id)?;
            Ok(json!({ "removed": p.name, "purged": stats }))
        }
        "set_video_role" => {
            let db = open_db(paths)?;
            let p = project(&db, args)?;
            let video_id = arg_i64(args, "video_id")?;
            match arg_str(args, "role")?.as_str() {
                "footage" => db.include_video(p.id, video_id)?,
                role => db.exclude_video(p.id, video_id, role)?,
            }
            Ok(json!({ "video_id": video_id, "project": p.name }))
        }
        other => anyhow::bail!("unknown tool '{other}'"),
    }
}
