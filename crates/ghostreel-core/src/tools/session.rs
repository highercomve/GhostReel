//! What a model run adds around a registry call.
//!
//! The registry's handlers know nothing about a conversation. A [`ToolSession`] wraps them with
//! what only a conversation needs: the old name `search_moments` still working, a repeated read
//! answered from memory instead of the database, a model that will not stop repeating itself told
//! to answer (and then stopped), results cut to the room the context has, errors turned into
//! something a model can read and fix, and a one-line summary of every call for the activity log.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use serde_json::{Value, json};

use super::{Effect, ToolCtx, ToolError, ToolImage, ToolOutput, canonical_name, lookup};

/// An optional cap on how much a run may do. `None` is the default and means no cap: with a
/// registry of forty tools a small number makes no sense, and a run still ends when the model
/// answers, when the user cancels, when the backend times out, or when the [`CallGuard`] gives up
/// on a model that repeats itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolBudget {
    cap: Option<u32>,
    used: u32,
}

impl ToolBudget {
    pub fn new(cap: Option<u32>) -> Self {
        Self { cap, used: 0 }
    }

    /// From a setting where 0 means no cap (`chat_model.max_tool_rounds`).
    pub fn from_setting(setting: u32) -> Self {
        Self::new((setting > 0).then_some(setting))
    }

    pub fn cap(&self) -> Option<u32> {
        self.cap
    }

    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn exhausted(&self) -> bool {
        self.cap.is_some_and(|c| self.used >= c)
    }

    /// Count one round or call.
    pub fn spend(&mut self) {
        self.used = self.used.saturating_add(1);
    }
}

/// What the turn has already asked for, so it is not asked again.
///
/// A model that loses track of what it has read asks for it again, and again: Bonsai 2 spent 24
/// of 55 calls re-fetching the same four ranges, seven times round a four-call cycle, until the
/// round budget stopped it with nothing drafted. The work is wasted twice over — the database is
/// queried again and the same text is appended to a context that was already full.
///
/// The first repeat gets its answer back with a note. A model still asking after that gets only
/// the note: the payload it keeps re-reading is what it is stuck on, and taking it away is what
/// breaks the cycle. Only reads are remembered; a call that changes something runs every time.
#[derive(Default)]
pub struct ToolMemo {
    seen: HashMap<String, (String, String, usize)>,
}

impl ToolMemo {
    /// Tool plus arguments, with object keys in a fixed order so `{a,b}` and `{b,a}` are one call.
    pub fn key(tool: &str, args: &Value) -> String {
        fn canonical(v: &Value) -> String {
            match v {
                Value::Object(map) => {
                    let mut keys: Vec<&String> = map.keys().collect();
                    keys.sort();
                    let inner: Vec<String> = keys.iter().map(|k| format!("{k}:{}", canonical(&map[*k]))).collect();
                    format!("{{{}}}", inner.join(","))
                }
                Value::Array(items) => format!("[{}]", items.iter().map(canonical).collect::<Vec<_>>().join(",")),
                other => other.to_string(),
            }
        }
        format!("{}{}", canonical_name(tool), canonical(args))
    }

    /// The answer to a call already made this turn, if there is one.
    pub fn recall(&mut self, tool: &str, args: &Value) -> Option<(String, String)> {
        let key = Self::key(tool, args);
        let (result, summary, times) = self.seen.get_mut(&key)?;
        *times += 1;
        if *times == 1 {
            Some((
                format!("(you already called this; the answer has not changed)\n{result}"),
                format!("{summary} — repeat"),
            ))
        } else {
            Some((
                format!(
                    "You have asked for this {} times and already have the answer. Stop looking and give your \
                     final answer with what you have.",
                    *times + 1
                ),
                "repeat — withheld".to_string(),
            ))
        }
    }

    pub fn remember(&mut self, tool: &str, args: &Value, result: &str, summary: &str) {
        self.seen.insert(Self::key(tool, args), (result.to_string(), summary.to_string(), 0));
    }

    /// How many times a remembered call has been asked for again.
    pub fn repeats(&self, tool: &str, args: &Value) -> usize {
        self.seen.get(&Self::key(tool, args)).map_or(0, |(_, _, times)| *times)
    }
}

/// How many times in a row a call may come back with the same thing before the next identical
/// call is refused.
pub const MAX_IDENTICAL_CALLS: u32 = 3;
/// Refusals in a row after which the model is taken to be stuck and the run moves to its answer.
pub const STOP_AFTER_REFUSALS: u32 = 2;
/// Answers withheld from a model that keeps asking for what it already has, after which the same.
/// Counted across the whole run, not in a row: a model going round a cycle of calls never repeats
/// the one it just made.
pub const STOP_AFTER_WITHHELD: u32 = 3;
/// Failed calls in a row after which the research ends. A model counting `video_id` up from the
/// last real one (139, 140, … 588, each "not in project") never repeats a call, so neither guard
/// above sees it: what it repeats is failing.
pub const STOP_AFTER_FAILURES: u32 = 8;
/// Calls of one reading tool in a turn after which the research ends: past this a model is walking
/// the library one id at a time rather than reading what it found (the outline lists every video
/// already; a 9B asked for each of them, then for ids that do not exist, 450 calls in an hour).
pub const MAX_CALLS_PER_TOOL: u32 = 60;
/// Versions a turn may save (save_script, apply_fix) before the research ends: a model going round
/// save, fix, read and save again made thirty-six versions of one cut without changing it.
pub const MAX_SAVES_PER_TURN: u32 = 20;

/// Notices a model calling the same tool with the same arguments, and getting the same result,
/// over and over. Judged by result as well as arguments: `index` is called again and again with the
/// same arguments until the work is done, and each answer is different.
#[derive(Default)]
pub struct CallGuard {
    last: Option<(String, u64)>,
    identical: u32,
    refused: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    /// Do not run it: answer with the refusal.
    Refuse {
        times: u32,
    },
    /// Refused and still asking: end the run's research.
    Stop {
        times: u32,
    },
}

impl CallGuard {
    pub fn check(&mut self, key: &str) -> Verdict {
        match &self.last {
            Some((k, _)) if k == key && self.identical >= MAX_IDENTICAL_CALLS => {
                self.refused += 1;
                let times = self.identical + self.refused;
                if self.refused >= STOP_AFTER_REFUSALS { Verdict::Stop { times } } else { Verdict::Refuse { times } }
            }
            _ => Verdict::Allow,
        }
    }

    /// A call ran and returned something with this fingerprint.
    pub fn note(&mut self, key: &str, fingerprint: u64) {
        match &mut self.last {
            Some((k, fp)) if k == key && *fp == fingerprint => self.identical += 1,
            _ => {
                self.last = Some((key.to_string(), fingerprint));
                self.identical = 1;
                self.refused = 0;
            }
        }
    }

    /// A call was answered from memory: the same result by construction.
    pub fn note_same(&mut self, key: &str) {
        match &mut self.last {
            Some((k, _)) if k == key => self.identical += 1,
            _ => {
                self.last = Some((key.to_string(), 0));
                self.identical = 1;
                self.refused = 0;
            }
        }
    }
}

fn fingerprint(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// What one call came to, as a conversation sees it.
#[derive(Debug, Default)]
pub struct Called {
    /// The tool's registry name (an old name resolved).
    pub tool: String,
    pub args: Value,
    /// What the model reads.
    pub model_text: String,
    /// One line for the activity log. Starts with `error:` when the call failed.
    pub summary: String,
    /// The structured result of a call that ran (not of a repeat answered from memory).
    pub value: Option<Value>,
    pub images: Vec<ToolImage>,
    /// The call found nothing it was asked for (a search with no hits).
    pub empty: bool,
    pub is_error: bool,
    /// The user stopped the run while the call was running.
    pub cancelled: bool,
    /// The model will not stop repeating itself: end the research and ask for the answer.
    pub stop: bool,
}

/// The wrapper every conversation calls tools through.
pub struct ToolSession {
    memo: ToolMemo,
    guard: CallGuard,
    /// Characters a result may take (the context's room for it).
    pub max_chars: usize,
    /// Calls made, including repeats and refusals.
    pub calls: u32,
    withheld: u32,
    /// Calls that failed since the last one that did not.
    failed_in_a_row: u32,
    /// Calls of each reading tool this turn.
    per_tool: std::collections::HashMap<String, u32>,
    /// Versions saved this turn (save_script, apply_fix, not dry runs), including refused ones.
    saves: u32,
    /// MCP sessions live for a process, not a chat turn, so their per-turn quotas do not apply.
    persistent: bool,
    /// The last script version a call saved (save_script, apply_fix): what the session delivered,
    /// the way an MCP client's work ends in the version it saved last.
    pub saved_script: Option<i64>,
    /// Videos whose pictures this session has read (get_manifest, get_video, get_frames,
    /// get_scenes, get_faces): a cut chosen from the others was chosen from words alone.
    pictures_seen: std::collections::HashSet<i64>,
}

/// The video ids a call names (`video_id`, `video_ids`).
fn video_ids_of(args: &Value) -> Vec<i64> {
    let mut ids: Vec<i64> =
        args["video_ids"].as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
    ids.extend(args["video_id"].as_i64());
    ids
}

impl ToolSession {
    fn invalidate_reads(&mut self) {
        self.memo = ToolMemo::default();
        self.withheld = 0;
    }

    fn should_refuse_read(&mut self, tool: &str) -> bool {
        if self.persistent {
            return false;
        }
        let n = self.per_tool.entry(tool.to_string()).or_default();
        *n += 1;
        *n > MAX_CALLS_PER_TOOL
    }

    fn should_refuse_save(&mut self, tool: &str, args: &Value) -> bool {
        if self.persistent || !matches!(tool, "save_script" | "apply_fix") || is_dry_run(args) {
            return false;
        }
        self.saves += 1;
        self.saves > MAX_SAVES_PER_TURN
    }

    /// Sound sources in the resolved script, including voices carried by audio beds.
    fn unseen_pictures(&self, db: &crate::db::Db, raw: &Value) -> Vec<i64> {
        let Ok(mut script) = serde_json::from_value::<crate::script::Script>(raw.clone()) else { return Vec::new() };
        // Use the same sentence ownership and resolution as the save handler. In particular a
        // model's raw video_id may be replaced by the video the sentence actually belongs to.
        if crate::sentences::resolve_script(db, &mut script).is_err() {
            return Vec::new();
        }
        let mut out: Vec<i64> = script
            .beats
            .iter()
            .flat_map(|beat| {
                beat.clips
                    .iter()
                    .filter(|c| c.audio != crate::script::Audio::Mute)
                    .map(|c| c.video_id)
                    .chain(beat.bed.iter().map(|b| b.video_id))
            })
            .filter(|v| !self.pictures_seen.contains(v))
            .collect();
        out.sort();
        out.dedup();
        out
    }

    pub fn new(max_chars: usize) -> Self {
        Self::with_persistent(max_chars, false)
    }

    /// Create a session whose lifetime is not bounded by chat-turn quotas.
    pub fn persistent(max_chars: usize) -> Self {
        Self::with_persistent(max_chars, true)
    }

    fn with_persistent(max_chars: usize, persistent: bool) -> Self {
        Self {
            memo: ToolMemo::default(),
            guard: CallGuard::default(),
            max_chars,
            calls: 0,
            withheld: 0,
            failed_in_a_row: 0,
            per_tool: Default::default(),
            saves: 0,
            persistent,
            saved_script: None,
            pictures_seen: Default::default(),
        }
    }

    pub async fn call(&mut self, ctx: &mut ToolCtx<'_>, name: &str, args: &Value) -> Called {
        self.calls += 1;
        let tool = canonical_name(name).to_string();
        let key = ToolMemo::key(&tool, args);
        let mut called = Called { tool: tool.clone(), args: args.clone(), ..Called::default() };

        match if self.persistent { Verdict::Allow } else { self.guard.check(&key) } {
            Verdict::Allow => {}
            Verdict::Refuse { times } => return refused(called, times, false),
            Verdict::Stop { times } => return refused(called, times, true),
        }

        let effect = lookup(&tool).map(|t| t.spec.effect);
        if effect == Some(Effect::Read) {
            let count = self.per_tool.get(&tool).copied().unwrap_or(0) + 1;
            if self.should_refuse_read(&tool) {
                called.is_error = true;
                called.stop = true;
                called.summary = format!("{tool} called {count} times — refused");
                called.model_text = format!(
                    "You have called {tool} {count} times this turn. Stop reading and give your final answer with what \
                     you have."
                );
                return called;
            }
        }
        if self.should_refuse_save(&tool, args) {
            called.is_error = true;
            called.stop = true;
            called.summary = format!("{} saves this turn — refused", self.saves - 1);
            called.model_text = format!(
                "You have saved {} versions this turn. Stop saving and give your final answer: the last \
                 version you saved is the cut.",
                self.saves - 1
            );
            return called;
        }
        if !self.persistent
            && effect == Some(Effect::Read)
            && let Some((text, summary)) = self.memo.recall(&tool, args)
        {
            self.guard.note_same(&key);
            // From the second repeat on the answer is withheld; a model that goes on asking is stuck.
            if self.memo.repeats(&tool, args) >= 2 {
                self.withheld += 1;
                called.stop = self.withheld >= STOP_AFTER_WITHHELD;
            }
            called.model_text = text;
            called.summary = summary;
            return called;
        }

        ctx.max_chars = Some(self.max_chars);
        match super::call(ctx, &tool, args).await {
            Ok(mut out) => {
                self.failed_in_a_row = 0;
                // Inspect the prepared script before output sizing can truncate it. When the
                // handler made no changes it omits `script`, so resolve the original instead.
                let unseen = if tool == "save_script" {
                    self.unseen_pictures(ctx.db, out.value.get("script").unwrap_or(&args["script"]))
                } else {
                    Vec::new()
                };
                fit(&mut out, self.max_chars);
                called.model_text = out.model_text();
                called.summary = if out.summary.is_empty() { "ok".into() } else { out.summary.clone() };
                called.empty = out.empty;
                called.images = std::mem::take(&mut out.images);
                if !self.persistent && effect == Some(Effect::Read) {
                    self.memo.remember(&tool, args, &called.model_text, &called.summary);
                }
                // A write changes what the reads said: answer them afresh after it.
                if matches!(effect, Some(Effect::Write | Effect::Destructive)) && !is_dry_run(args) {
                    self.invalidate_reads();
                }
                self.guard.note(&key, fingerprint(&called.model_text));
                if effect == Some(Effect::Write)
                    && matches!(tool.as_str(), "save_script" | "apply_fix")
                    && !is_dry_run(args)
                    && let Some(id) = out.value["script_id"].as_i64()
                {
                    self.saved_script = Some(id);
                }
                if matches!(tool.as_str(), "get_manifest" | "get_video" | "get_frames" | "get_scenes" | "get_faces") {
                    self.pictures_seen.extend(video_ids_of(args));
                }
                if !unseen.is_empty() {
                    let list = unseen.iter().map(|v| format!("#{v}")).collect::<Vec<_>>().join(", ");
                    called.model_text.push_str(&format!(
                            "\nNOTE: you chose clips from video {list} without reading its pictures in this session. \
                             A clip is a picture as well as words: get_manifest (video_ids: [{}]) shows who is on \
                             screen, the framing and what each moment looks like. Read it, and save again if it changes \
                             your choices.\n",
                            unseen.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                        ));
                }
                called.value = Some(out.value);
            }
            Err(e) => {
                called.is_error = true;
                called.cancelled = e.is_cancelled();
                let message = match &e {
                    ToolError::Unknown(_) => format!("{e}; the tools are: {}", super::names().join(", ")),
                    other => other.to_string(),
                };
                called.summary = format!("error: {}", one_line(&e.to_string(), 100));
                called.model_text = json!({ "error": message }).to_string();
                if !called.cancelled {
                    self.failed_in_a_row += 1;
                    if self.failed_in_a_row >= STOP_AFTER_FAILURES {
                        called.stop = true;
                        called.model_text = json!({
                            "error": message,
                            "stop": format!(
                                "Your last {} calls all failed. Stop calling tools and give your final answer with \
                                 what you have.",
                                self.failed_in_a_row
                            ),
                        })
                        .to_string();
                    }
                }
            }
        }
        called
    }
}

fn refused(mut called: Called, times: u32, stop: bool) -> Called {
    called.is_error = true;
    called.stop = stop;
    called.summary = "repeat — refused".into();
    called.model_text = format!(
        "You have made this exact call {times} times in a row and it keeps returning the same thing. \
         Do not call it again: give your final answer with what you have."
    );
    called
}

fn one_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or_default().trim();
    if line.chars().count() <= max {
        line.to_string()
    } else {
        format!("{}…", line.chars().take(max).collect::<String>())
    }
}

/// Cut a result to `max` characters: the text by whole lines, the JSON by dropping the tail of
/// its longest list.
pub fn fit(out: &mut ToolOutput, max: usize) {
    if let Some(text) = &mut out.text {
        if text.len() > max {
            *text = cut_lines(text, max);
        }
        return;
    }
    shrink_value(&mut out.value, max);
}

fn cut_lines(text: &str, max: usize) -> String {
    let total = text.lines().count();
    let mut kept = String::new();
    let mut n = 0;
    for line in text.lines() {
        if kept.len() + line.len() + 1 > max {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
        n += 1;
    }
    if n < total {
        kept.push_str(&format!("… ({} more lines cut to fit)\n", total - n));
    }
    kept
}

fn longest_array(v: &mut Value, depth: u8) -> Option<&mut Vec<Value>> {
    match v {
        Value::Array(a) => Some(a),
        Value::Object(o) if depth > 0 => {
            let mut best: Option<(&String, usize)> = None;
            for (k, x) in o.iter() {
                if let Value::Array(a) = x
                    && !a.is_empty()
                    && best.is_none_or(|(_, n)| a.len() > n)
                {
                    best = Some((k, a.len()));
                }
            }
            let key = best?.0.clone();
            o.get_mut(&key).and_then(|x| longest_array(x, depth - 1))
        }
        _ => None,
    }
}

/// Drop the tail of the longest list until the value fits, and say so.
pub fn shrink_value(value: &mut Value, max: usize) {
    let mut dropped = 0usize;
    loop {
        let total = value.to_string().len();
        if total <= max {
            break;
        }
        let Some(list) = longest_array(value, 2) else { break };
        if list.is_empty() {
            break;
        }
        // Proportional, so a long list converges in a few passes instead of one pop each.
        let cut = (list.len() * (total - max) / total).clamp(1, list.len());
        let keep = list.len() - cut;
        list.truncate(keep);
        dropped += cut;
    }
    if dropped > 0
        && let Value::Object(o) = value
    {
        o.insert("truncated".into(), json!(format!("{dropped} items left out to fit; ask for a narrower range")));
    }
}

fn is_dry_run(args: &serde_json::Value) -> bool {
    super::Args(args).bool_or("dry_run", false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_dry_runs_do_not_use_save_quota() {
        let mut session = ToolSession::new(8000);
        for _ in 0..(MAX_SAVES_PER_TURN + 5) {
            assert!(!session.should_refuse_save("save_script", &json!({"dry_run":"true"})));
        }
        assert!(is_dry_run(&json!({"dry_run":"true"})));
        assert!(!is_dry_run(&json!({"dry_run":"false"})));
    }

    #[tokio::test]
    async fn persistent_reads_repeat_and_observe_external_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = crate::paths::Paths::in_dir(tmp.path());
        let config = crate::config::Config::default();
        let mut db = crate::db::Db::open_in_memory().unwrap();
        let mut embedder = None;
        let mut ctx = ToolCtx::new(&paths, &config, &mut db, &mut embedder, super::super::Origin::External);
        let mut session = ToolSession::persistent(8000);
        let first = session.call(&mut ctx, "list_projects", &json!({})).await;
        for _ in 0..6 {
            let next = session.call(&mut ctx, "list_projects", &json!({})).await;
            assert!(!next.is_error && !next.stop);
            assert_eq!(first.model_text, next.model_text);
        }
        ctx.db.create_project(&crate::projects::NewProject::named("Fresh")).unwrap();
        let fresh = session.call(&mut ctx, "list_projects", &json!({})).await;
        assert!(fresh.model_text.contains("Fresh"));
        assert!(session.memo.recall("list_projects", &json!({})).is_none());
    }

    #[tokio::test]
    async fn destructive_call_invalidates_reads() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = crate::paths::Paths::in_dir(tmp.path());
        let config = crate::config::Config::default();
        let mut db = crate::db::Db::open_in_memory().unwrap();
        db.create_project(&crate::projects::NewProject::named("Gone")).unwrap();
        let mut embedder = None;
        let mut ctx = ToolCtx::new(&paths, &config, &mut db, &mut embedder, super::super::Origin::External);
        let mut session = ToolSession::new(8000);
        session.call(&mut ctx, "list_projects", &json!({})).await;
        let removed = session.call(&mut ctx, "delete_project", &json!({"project":"Gone"})).await;
        assert!(!removed.is_error, "{}", removed.model_text);
        assert!(session.memo.recall("list_projects", &json!({})).is_none());
        let fresh = session.call(&mut ctx, "list_projects", &json!({})).await;
        assert!(!fresh.model_text.contains("Gone"));
    }

    #[test]
    fn picture_warning_uses_resolved_sentence_video() {
        let db = crate::db::Db::open_in_memory().unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,60), (2,'b',1,60);
            INSERT INTO transcript_sentences(id,video_id,start_s,end_s,text) VALUES (12,2,20,30,'Answer.');",
            )
            .unwrap();
        let mut session = ToolSession::new(8000);
        session.pictures_seen.insert(1);
        let args = json!({"script": {"title":"Sentence", "beats":[{"id":"1", "purpose":"test", "clips":[
            {"video_id":1,"from":"s12","to":"s12"}
        ]}]}});
        // The same correction the save handler applies must determine the reminder's source.
        assert_eq!(session.unseen_pictures(&db, &args["script"]), vec![2]);
    }

    #[test]
    fn picture_warning_includes_audio_beds() {
        let db = crate::db::Db::open_in_memory().unwrap();
        let session = ToolSession::new(8000);
        let args = json!({"script": {"title":"Bed", "beats":[{"id":"1", "purpose":"test", "clips":[
            {"video_id":1,"in_s":0.0,"out_s":5.0,"audio":"mute"}
        ], "bed":{"video_id":2,"in_s":0.0,"out_s":5.0}}]}});
        assert_eq!(session.unseen_pictures(&db, &args["script"]), vec![2]);
    }

    #[test]
    fn persistent_session_has_no_turn_save_or_read_quotas() {
        let mut session = ToolSession::persistent(8000);
        let args = json!({});
        for _ in 0..(MAX_CALLS_PER_TOOL + 5) {
            assert!(!session.should_refuse_read("list_projects"));
        }
        for _ in 0..(MAX_SAVES_PER_TURN + 5) {
            assert!(!session.should_refuse_save("save_script", &args));
        }
    }

    #[test]
    fn successful_write_invalidates_read_memo_and_repeat_state() {
        let mut session = ToolSession::new(8000);
        let args = json!({"project":"A"});
        session.memo.remember("list_scripts", &args, "old answer", "old");
        session.memo.recall("list_scripts", &args).unwrap();
        session.guard.note_same(&ToolMemo::key("list_scripts", &args));
        session.withheld = 2;

        session.invalidate_reads(); // successful non-dry-run write
        session.guard.note("set_glossary{}", fingerprint("write result"));

        assert!(session.memo.recall("list_scripts", &args).is_none());
        assert_eq!(session.withheld, 0);
        assert_eq!(session.guard.check(&ToolMemo::key("list_scripts", &args)), Verdict::Allow);
    }
}
