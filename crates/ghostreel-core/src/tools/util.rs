//! Small helpers the handlers share.

use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use super::ToolCtx;

pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// The `description` field of a keyframe's description JSON.
pub fn frame_description(json: Option<&str>) -> Option<String> {
    json.and_then(|j| serde_json::from_str::<Value>(j).ok())
        .and_then(|v| v.get("description").and_then(|d| d.as_str()).map(str::to_string))
}

/// `text` cut to `n` characters, with an ellipsis.
pub fn clip_chars(text: &str, n: usize) -> String {
    if text.chars().count() > n { format!("{}…", text.chars().take(n).collect::<String>()) } else { text.to_string() }
}

pub fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The file name of a video's first copy.
pub fn video_file_name(db: &crate::db::Db, video_id: i64) -> String {
    db.conn
        .query_row("SELECT MIN(path) FROM video_files WHERE video_id = ?1", [video_id], |r| {
            r.get::<_, Option<String>>(0)
        })
        .ok()
        .flatten()
        .map(|p| file_name(Path::new(&p)))
        .unwrap_or_default()
}

/// Drop items off the end of `items` until they serialize to at most `room` characters.
/// Returns how many were dropped.
pub fn fit_items<T: Serialize>(items: &mut Vec<T>, room: usize) -> usize {
    let mut dropped = 0;
    loop {
        let size = serde_json::to_string(items).map(|s| s.len()).unwrap_or(0);
        if size <= room || items.is_empty() {
            return dropped;
        }
        let cut = (items.len() * (size - room) / size).clamp(1, items.len());
        items.truncate(items.len() - cut);
        dropped += cut;
    }
}

impl ToolCtx<'_> {
    /// The room a list may take in a result, leaving `overhead` characters for the rest of it.
    pub fn room(&self, overhead: usize) -> usize {
        self.max_chars.map(|m| m.saturating_sub(overhead)).unwrap_or(usize::MAX)
    }
}
