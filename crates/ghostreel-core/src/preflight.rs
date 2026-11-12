//! Checks before a run starts, for any feature: are the agents it will use there and usable, and
//! is the project ready for it?
//!
//! A feature gathers [`Check`]s from the modules that know each thing — [`crate::agents`] for its
//! agents, [`project_readiness`] for what indexing has done — into one [`Preflight`]. Errors stop
//! the run before it starts; warnings are shown in the form before it starts and in the run's log.

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::agents::AgentError;
use crate::db::Db;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Ok,
    Warn,
    Error,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Ok => "ok",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }
}

/// What a check is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Agent,
    Project,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub level: Level,
    pub scope: Scope,
    /// The agent id, or the indexing stage ("transcribe", "faces").
    pub subject: String,
    pub message: String,
    /// The typed reason, when an agent is what is wrong.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<AgentError>,
}

impl Check {
    pub fn new(level: Level, scope: Scope, subject: impl Into<String>, message: impl Into<String>) -> Self {
        Self { level, scope, subject: subject.into(), message: message.into(), error: None }
    }

    pub fn with_error(mut self, error: AgentError) -> Self {
        self.error = Some(error);
        self
    }
}

/// Everything checked before one run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Preflight {
    pub checks: Vec<Check>,
}

impl Preflight {
    pub fn level(&self) -> Level {
        self.checks.iter().map(|c| c.level).max().unwrap_or(Level::Ok)
    }

    pub fn ok(&self) -> bool {
        self.level() < Level::Error
    }

    pub fn at(&self, level: Level) -> impl Iterator<Item = &Check> {
        self.checks.iter().filter(move |c| c.level == level)
    }

    /// The errors as one message, `None` when there are none.
    pub fn error_message(&self) -> Option<String> {
        let errs: Vec<&str> = self.at(Level::Error).map(|c| c.message.as_str()).collect();
        (!errs.is_empty()).then(|| errs.join("; "))
    }

    /// The run cannot start: an error naming every problem.
    pub fn into_result(self) -> Result<Preflight, Error> {
        match self.error_message() {
            Some(m) => Err(Error::Invalid(format!("cannot start: {m}"))),
            None => Ok(self),
        }
    }
}

/// What a run needs indexing to have done.
#[derive(Debug, Clone, Copy, Default)]
pub struct Needs {
    /// Speech transcribed (an error while any selected video is still waiting).
    pub transcripts: bool,
    /// Frame descriptions, faces and speakers (a warning while any are still waiting: the run
    /// works without them, just with less to go on).
    pub analysis: bool,
}

/// Whether a project's videos (or the chosen ones) are indexed far enough for a run.
pub fn project_readiness(db: &Db, project_id: i64, video_ids: &[i64], needs: Needs) -> Result<Vec<Check>, Error> {
    let mut out = Vec::new();
    let videos: Vec<i64> = if video_ids.is_empty() {
        crate::index::videos(db, Some(project_id))?.into_iter().map(|v| v.id).collect()
    } else {
        video_ids.to_vec()
    };
    if videos.is_empty() {
        out.push(Check::new(Level::Error, Scope::Project, "videos", "the project has no videos yet"));
        return Ok(out);
    }
    let n = videos.len();
    let waiting = |stage: &str| -> Result<usize, Error> {
        let mut count = 0;
        for v in &videos {
            let busy: bool = db.conn.query_row(
                "SELECT EXISTS (SELECT 1 FROM jobs WHERE video_id = ?1 AND stage = ?2 AND state IN ('pending', 'running'))",
                rusqlite::params![v, stage],
                |r| r.get(0),
            )?;
            count += busy as usize;
        }
        Ok(count)
    };
    let of = |k: usize| match k {
        _ if n == 1 => "the video".to_string(),
        k if k == n => format!("all {n} videos"),
        k => format!("{k} of {n} videos"),
    };
    if needs.transcripts {
        let k = waiting("transcribe")?;
        if k > 0 {
            out.push(Check::new(
                Level::Error,
                Scope::Project,
                "transcribe",
                format!("transcription is still running for {} — wait for indexing to finish or pick transcribed videos", of(k)),
            ));
        }
    }
    if needs.analysis {
        let k = waiting("describe")?;
        if k > 0 {
            out.push(Check::new(
                Level::Warn,
                Scope::Project,
                "describe",
                format!(
                    "frame descriptions are still running for {}: clips will know less about what is on screen",
                    of(k)
                ),
            ));
        }
        let k = waiting("faces")?;
        if k > 0 {
            out.push(Check::new(
                Level::Warn,
                Scope::Project,
                "faces",
                format!("faces are not found yet in {}: framing will not follow people there", of(k)),
            ));
        }
        let k = waiting("speakers")?;
        if k > 0 {
            out.push(Check::new(
                Level::Warn,
                Scope::Project,
                "speakers",
                format!("speakers are not identified yet in {}: clips cannot tell who is talking", of(k)),
            ));
        }
    }
    Ok(out)
}
