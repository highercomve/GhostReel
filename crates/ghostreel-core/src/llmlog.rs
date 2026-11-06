//! Persistent, per-chat model logs. Logging is best effort and never fails a turn.
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    pub seq: u64,
    pub ts_ms: i64,
    pub session_id: Option<i64>,
    pub kind: String,
    pub text: String,
}

#[derive(Default)]
struct Store {
    dir: Option<PathBuf>,
    next: HashMap<i64, u64>,
    // Recent polling cursors, bounded independently of the size of any log.
    cursors: VecDeque<(i64, u64, u64)>,
    session: Option<i64>,
    secrets: Vec<String>,
}
static LOG: LazyLock<Mutex<Store>> = LazyLock::new(|| Mutex::new(Store::default()));
fn lock() -> std::sync::MutexGuard<'static, Store> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// `dir` is the application's data directory, supplied by the host at startup.
pub fn init(dir: impl AsRef<Path>) {
    let mut log = lock();
    log.dir = Some(dir.as_ref().join("logs/chat"));
    log.next.clear();
    log.cursors.clear();
    log.secrets.clear();
}
pub fn path(session_id: i64) -> Option<PathBuf> {
    lock().path(session_id)
}
pub fn delete(session_id: i64) {
    let mut log = lock();
    if let Some(path) = log.path(session_id) {
        let _ = std::fs::remove_file(path);
    }
    log.next.remove(&session_id);
    log.cursors.retain(|c| c.0 != session_id);
}
impl Store {
    fn path(&self, id: i64) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{id}.jsonl")))
    }
    fn append(&mut self, id: i64, kind: &str, text: String) -> std::io::Result<()> {
        let Some(path) = self.path(id) else {
            return Ok(());
        };
        std::fs::create_dir_all(path.parent().unwrap())?;
        // A deleted/cleared log starts over, including when Storage removed it.
        if !path.exists() {
            self.next.remove(&id);
            self.cursors.retain(|c| c.0 != id);
        }
        let mut file = OpenOptions::new().create(true).read(true).append(true).open(path)?;
        if !self.next.contains_key(&id) {
            let mut reader = BufReader::new(&file);
            let mut line = Vec::new();
            let (mut next, mut end) = (1, 0);
            loop {
                line.clear();
                if reader.read_until(b'\n', &mut line)? == 0 {
                    break;
                }
                if !line.ends_with(b"\n") {
                    break;
                } // discard a crash's partial record
                if let Ok(e) = serde_json::from_slice::<LogEntry>(&line) {
                    next = next.max(e.seq + 1);
                }
                end = reader.stream_position()?;
            }
            file.set_len(end)?;
            self.next.insert(id, next);
        }
        let seq = self.next[&id];
        let entry = LogEntry { seq, ts_ms: timestamp(), session_id: Some(id), kind: kind.into(), text };
        let mut bytes = serde_json::to_vec(&entry)?;
        bytes.push(b'\n');
        file.write_all(&bytes)?;
        self.next.insert(id, seq + 1);
        Ok(())
    }
    fn since(&mut self, id: i64, after: u64) -> std::io::Result<Vec<LogEntry>> {
        let Some(path) = self.path(id) else {
            return Ok(Vec::new());
        };
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        let offset =
            self.cursors.iter().rev().find(|c| c.0 == id && c.1 <= after && c.2 <= len).map(|c| c.2).unwrap_or(0);
        file.seek(SeekFrom::Start(offset))?;
        let mut reader = BufReader::new(file);
        let mut out = Vec::new();
        let mut line = Vec::new();
        let mut bytes = 0;
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 || !line.ends_with(b"\n") {
                break;
            }
            if let Ok(e) = serde_json::from_slice::<LogEntry>(&line) {
                if e.seq > after {
                    bytes += line.len();
                    let seq = e.seq;
                    out.push(e);
                    self.cursors.push_back((id, seq, reader.stream_position()?));
                    while self.cursors.len() > 64 {
                        self.cursors.pop_front();
                    }
                    // Catch up over multiple polls rather than allocating an entire huge log.
                    if bytes >= 2 * 1024 * 1024 || out.len() >= 256 {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}
pub fn timestamp() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}
/// Prevent configured credentials from appearing even if a backend echoes one in an error.
pub fn redact_secret(secret: &str) {
    if !secret.is_empty() {
        let mut log = lock();
        if !log.secrets.iter().any(|s| s == secret) {
            log.secrets.push(secret.into());
        }
    }
}
pub fn push(kind: &str, text: impl Into<String>) {
    let mut log = lock();
    if let Some(id) = log.session {
        let mut text = text.into();
        for secret in &log.secrets {
            text = text.replace(secret, "[redacted]");
        }
        if log.append(id, kind, text).is_err() {
            // Re-scan/truncate a partial write before the next append.
            log.next.remove(&id);
            log.cursors.retain(|c| c.0 != id);
        }
    }
}
pub fn since(session_id: i64, after_seq: u64) -> Vec<LogEntry> {
    lock().since(session_id, after_seq).unwrap_or_default()
}
pub fn session(session_id: i64) -> SessionGuard {
    let previous = lock().session.replace(session_id);
    SessionGuard(session_id, previous)
}
pub struct SessionGuard(i64, Option<i64>);
impl Drop for SessionGuard {
    fn drop(&mut self) {
        let mut log = lock();
        if log.session == Some(self.0) {
            log.session = self.1;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn store(dir: &Path) -> Store {
        Store { dir: Some(dir.into()), ..Store::default() }
    }
    #[test]
    fn append_read_and_restart() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = store(dir.path());
        log.append(1, "prompt", "hello\nworld".into()).unwrap();
        log.append(1, "answer", "answer".into()).unwrap();
        log.append(2, "info", "other".into()).unwrap();
        assert_eq!(log.since(1, 0).unwrap().len(), 2);
        assert_eq!(log.since(1, 1).unwrap()[0].text, "answer");
        let mut restarted = store(dir.path());
        restarted.append(1, "tool", "result".into()).unwrap();
        assert_eq!(restarted.since(1, 2).unwrap()[0].seq, 3);
        assert_eq!(restarted.since(2, 0).unwrap().len(), 1);
    }
    #[test]
    fn missing_chat_log_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(store(dir.path()).since(42, 0).unwrap().is_empty());
    }
    #[test]
    fn bounded_reads_can_catch_up_without_skipping_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = store(dir.path());
        for _ in 0..600 {
            log.append(1, "tool", "result".into()).unwrap();
        }
        let mut after = 0;
        let mut count = 0;
        loop {
            let batch = log.since(1, after).unwrap();
            if batch.is_empty() {
                break;
            }
            assert!(batch.len() <= 256);
            assert_eq!(batch[0].seq, after + 1);
            after = batch.last().unwrap().seq;
            count += batch.len();
        }
        assert_eq!(count, 600);
        assert_eq!(after, 600);
    }
    #[test]
    fn partial_tail_is_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = store(dir.path());
        log.append(1, "info", "first".into()).unwrap();
        OpenOptions::new().append(true).open(log.path(1).unwrap()).unwrap().write_all(b"{broken\xff").unwrap();
        let mut restarted = store(dir.path());
        restarted.append(1, "info", "second".into()).unwrap();
        assert_eq!(restarted.since(1, 0).unwrap().len(), 2);
    }
}
