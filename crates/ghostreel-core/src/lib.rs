//! GhostReel core — shared by the desktop app and the `ghostreel` CLI.
//!
//! See `.agents/plan.md` for the architecture. M0 provides configuration, paths, the index
//! database, AI server probing and the doctor report.

pub mod audio;
pub mod chat;
pub mod chunks;
pub mod cliagent;
pub mod config;
pub mod db;
pub mod doctor;
pub mod embed;
pub mod export;
pub mod fcpxml;
pub mod frames;
pub mod index;
pub mod interviewer;
pub mod jev;
pub mod media;
pub mod models;
pub mod otio;
pub mod paths;
pub mod preview;
pub mod probe;
pub mod proc;
pub mod progress;
pub mod projects;
pub mod runtime;
pub mod script;
pub mod search;
pub mod steadiness;
pub mod stt;
pub mod vision;
pub mod watch;

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("database: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("database schema v{found} is newer than this GhostReel supports (v{supported}); update GhostReel")]
    SchemaTooNew { found: u32, supported: u32 },
    #[error("database migration failed: {0}")]
    Migration(String),
    #[error("{0}")]
    Invalid(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("ffprobe: {0}")]
    Probe(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("embeddings: {0}")]
    Embed(String),
    #[error("vision: {0}")]
    Vision(String),
    #[error("jev: {0}")]
    Jev(String),
    #[error("frames: {0}")]
    Frames(String),
    #[error("transcription: {0}")]
    Stt(String),
    #[error("export: {0}")]
    Export(String),
    #[error("preview: {0}")]
    Preview(String),
    #[error("another GhostReel process is already indexing ({0})")]
    Busy(String),
    #[error("cannot determine the user's {0} directory")]
    NoHomeDir(&'static str),
}
