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

use super::{Effect, ToolCtx, ToolError, ToolOutput, ToolImage, canonical_name, lookup};

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
    Refuse { times: u32 },
    /// Refused and still asking: end the run's research.
    Stop { times: u32 },
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
}

impl ToolSession {
    pub fn new(max_chars: usize) -> Self {
        Self { memo: ToolMemo::default(), guard: CallGuard::default(), max_chars, calls: 0, withheld: 0 }
    }

    pub async fn call(&mut self, ctx: &mut ToolCtx<'_>, name: &str, args: &Value) -> Called {
        self.calls += 1;
        let tool = canonical_name(name).to_string();
        let key = ToolMemo::key(&tool, args);
        let mut called = Called { tool: tool.clone(), args: args.clone(), ..Called::default() };

        match self.guard.check(&key) {
            Verdict::Allow => {}
            Verdict::Refuse { times } => return refused(called, times, false),
            Verdict::Stop { times } => return refused(called, times, true),
        }

        let effect = lookup(&tool).map(|t| t.spec.effect);
        if effect == Some(Effect::Read)
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
                fit(&mut out, self.max_chars);
                called.model_text = out.model_text();
                called.summary = if out.summary.is_empty() { "ok".into() } else { out.summary.clone() };
                called.empty = out.empty;
                called.images = std::mem::take(&mut out.images);
                if effect == Some(Effect::Read) {
                    self.memo.remember(&tool, args, &called.model_text, &called.summary);
                }
                self.guard.note(&key, fingerprint(&called.model_text));
                called.value = Some(out.value);
            }
            Err(e) => {
                called.is_error = true;
                called.cancelled = e.is_cancelled();
                let message = match &e {
                    ToolError::Unknown(_) => format!(
                        "{e}; the tools are: {}",
                        super::names().join(", ")
                    ),
                    other => other.to_string(),
                };
                called.summary = format!("error: {}", one_line(&e.to_string(), 100));
                called.model_text = json!({ "error": message }).to_string();
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
    if line.chars().count() <= max { line.to_string() } else { format!("{}…", line.chars().take(max).collect::<String>()) }
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
