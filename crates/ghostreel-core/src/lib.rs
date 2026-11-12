//! GhostReel core — shared by the desktop app and the `ghostreel` CLI.
//!
//! See `.agents/plan.md` for the architecture. M0 provides configuration, paths, the index
//! database, AI server probing and the doctor report.

pub mod agents;
pub mod audio;
pub mod backup;
pub mod audio_tracks;
pub mod cancel;
pub mod captions;
pub mod chat;
pub mod chunks;
pub mod clips;
pub mod classify;
pub mod cliagent;
pub mod config;
pub mod db;
pub mod doctor;
pub mod embed;
pub mod export;
pub mod faces;
pub mod fcpxml;
pub mod frames;
pub mod framing;
pub mod ghost_service;
pub mod glossary;
pub mod index;
pub mod interviewer;
pub mod jev;
pub mod llmlog;
pub mod onnx;
pub mod media;
pub mod preflight;
pub mod models;
pub mod organize;
pub mod otio;
pub mod overlays;
pub mod paths;
pub mod preview;
pub mod probe;
pub mod proc;
pub mod progress;
pub mod projects;
pub mod runs;
pub mod runtime;
pub mod script;
pub mod scenes;
pub mod script_check;
pub mod script_fix;
pub mod search;
pub mod speakers;
pub mod steadiness;
pub mod storage;
pub mod stt;
pub mod tools;
pub mod trash;
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
    /// A model agent's call failed, typed (see [`agents::AgentError`]).
    #[error("{0}")]
    Agent(agents::AgentError),
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
    /// The work was stopped (a task's Stop, Ctrl+C). Displays as [`chat::CANCELLED`] so callers
    /// that still match on the text keep working; prefer [`Error::is_cancelled`].
    #[error("stopped")]
    Cancelled,
}

impl Error {
    /// Whether this error is a stop rather than a failure, whichever layer raised it (the typed
    /// [`Error::Cancelled`], a render's `preview: cancelled`, a chat turn's [`chat::CANCELLED`]).
    pub fn is_cancelled(&self) -> bool {
        match self {
            Error::Cancelled => true,
            Error::Preview(s) => s == "cancelled",
            e => e.to_string().contains(chat::CANCELLED),
        }
    }

    /// The typed agent failure, when this is one.
    pub fn agent_error(&self) -> Option<&agents::AgentError> {
        match self {
            Error::Agent(e) => Some(e),
            _ => None,
        }
    }
}

impl From<agents::AgentError> for Error {
    fn from(e: agents::AgentError) -> Self {
        Error::Agent(e)
    }
}

pub mod inference;
