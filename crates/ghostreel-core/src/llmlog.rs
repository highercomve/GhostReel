//! The model's side of a script chat, kept for the "Full log" view: every prompt the model is
//! sent, its thinking, its answer, and whatever the local helper prints while it works.
//!
//! In-process and bounded: a chat turn runs inside the app, so the app can hand the log to the
//! window without a file. A 96-video project sends ~100 KB per prompt, so the buffer is capped
//! by bytes and drops the oldest entries first.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

/// How much text the log holds before it forgets the oldest entries.
const MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LogEntry {
    /// Increasing per process, so a viewer can ask only for what it has not seen.
    pub seq: u64,
    /// Unix milliseconds.
    pub ts_ms: i64,
    /// The chat session running when this was logged, if any.
    pub session_id: Option<i64>,
    /// `prompt`, `thinking`, `answer`, `helper` (the helper's own output) or `info`.
    pub kind: String,
    pub text: String,
}

struct Log {
    entries: VecDeque<LogEntry>,
    bytes: usize,
    next_seq: u64,
    session: Option<i64>,
}

static LOG: Mutex<Log> = Mutex::new(Log { entries: VecDeque::new(), bytes: 0, next_seq: 1, session: None });

fn lock() -> std::sync::MutexGuard<'static, Log> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record one entry, tagged with the session that is running. Outside a chat nothing is kept:
/// indexing talks to the same helper once per keyframe and nobody reads that here.
pub fn push(kind: &str, text: impl Into<String>) {
    if lock().session.is_none() {
        return;
    }
    let text = text.into();
    let ts_ms =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
    let mut log = lock();
    let seq = log.next_seq;
    log.next_seq += 1;
    log.bytes += text.len();
    let session_id = log.session;
    log.entries.push_back(LogEntry { seq, ts_ms, session_id, kind: kind.to_string(), text });
    while log.bytes > MAX_BYTES && log.entries.len() > 1 {
        if let Some(old) = log.entries.pop_front() {
            log.bytes -= old.text.len();
        }
    }
}

/// Everything logged for `session_id` after `after_seq`.
pub fn since(session_id: i64, after_seq: u64) -> Vec<LogEntry> {
    lock().entries.iter().filter(|e| e.seq > after_seq && e.session_id == Some(session_id)).cloned().collect()
}

/// Tags what is logged from now until the guard drops with `session_id`. The helper's output
/// arrives on its own task with no idea which chat it serves; this is how it is attributed.
pub fn session(session_id: i64) -> SessionGuard {
    lock().session = Some(session_id);
    SessionGuard(session_id)
}

pub struct SessionGuard(i64);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        let mut log = lock();
        if log.session == Some(self.0) {
            log.session = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_kept_per_session_and_read_incrementally() {
        // Session ids far from anything another test would use: the log is process-wide.
        let a = 9_000_001;
        let b = 9_000_002;
        {
            let _g = session(a);
            push("prompt", "p1");
            push("answer", "a1");
        }
        push("helper", "between turns");
        {
            let _g = session(b);
            push("prompt", "other chat");
        }
        let all = since(a, 0);
        assert_eq!(all.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(), vec!["p1", "a1"]);
        assert_eq!(since(a, all[0].seq).len(), 1, "only what is new");
        assert_eq!(since(b, 0).len(), 1);
    }
}
