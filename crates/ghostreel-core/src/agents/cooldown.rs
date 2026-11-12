//! Endpoints whose quota is used up, for the whole process.
//!
//! When a call says the quota is gone until some time, every feature should stop asking that
//! endpoint until then — not just the run that found out. A pool records it here; pools resolved
//! later skip the endpoint, and preflight reports it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use serde::Serialize;

use super::error::AgentError;
use super::slots::EndpointKey;

/// How long an endpoint rests when its quota is gone and it did not say until when.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(30 * 60);

/// One endpoint resting until its quota resets.
#[derive(Debug, Clone, Serialize)]
pub struct Cooldown {
    pub key: EndpointKey,
    /// Unix seconds.
    pub until: i64,
    /// The failure that started it.
    pub error: AgentError,
}

impl Cooldown {
    pub fn remaining(&self) -> Duration {
        Duration::from_secs((self.until - unix_now()).max(0) as u64)
    }

    /// "quota used up — resting for 2 h 3 min".
    pub fn reason(&self) -> String {
        format!("{} — resting for {}", self.error.kind.label(), super::error::human_duration(self.remaining()))
    }
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Resting endpoints. [`Cooldowns::global`] is the process's; tests make their own.
#[derive(Debug, Default)]
pub struct Cooldowns {
    map: Mutex<HashMap<EndpointKey, Cooldown>>,
}

impl Cooldowns {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn global() -> Arc<Cooldowns> {
        static GLOBAL: OnceLock<Arc<Cooldowns>> = OnceLock::new();
        GLOBAL.get_or_init(Cooldowns::new).clone()
    }

    /// Rest `key` for `for_` (or [`DEFAULT_COOLDOWN`]). A later reset time wins over an earlier.
    pub fn put(&self, key: &EndpointKey, for_: Option<Duration>, error: AgentError) -> Cooldown {
        let until = unix_now() + for_.unwrap_or(DEFAULT_COOLDOWN).as_secs().max(1) as i64;
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let entry =
            map.entry(key.clone()).or_insert_with(|| Cooldown { key: key.clone(), until, error: error.clone() });
        if until > entry.until {
            entry.until = until;
            entry.error = error;
        }
        entry.clone()
    }

    /// The endpoint's cooldown while it lasts.
    pub fn get(&self, key: &EndpointKey) -> Option<Cooldown> {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        match map.get(key) {
            Some(c) if c.until > unix_now() => Some(c.clone()),
            Some(_) => {
                map.remove(key);
                None
            }
            None => None,
        }
    }

    /// Every cooldown still running.
    pub fn snapshot(&self) -> Vec<Cooldown> {
        let now = unix_now();
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, c| c.until > now);
        map.values().cloned().collect()
    }

    pub fn clear(&self, key: &EndpointKey) {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).remove(key);
    }
}

/// One call outside a pool (an orchestrator, a single test question), handled the way a pool
/// handles its jobs: a busy endpoint is asked again after a pause (a few times), a used-up quota
/// rests the endpoint for every feature, and the failure comes back typed and named after `agent`.
pub async fn call_once<T, F, Fut>(agent: &str, key: Option<&EndpointKey>, mut call: F) -> Result<T, crate::Error>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, crate::Error>>,
{
    use super::error::{AgentErrorKind, Handling};
    const TRIES: u32 = 4;
    if let Some(c) = key.and_then(|k| Cooldowns::global().get(k)) {
        return Err(crate::Error::Agent(c.error.for_agent(agent)));
    }
    let mut tries = 0;
    loop {
        tries += 1;
        let err = match call().await {
            Ok(v) => return Ok(v),
            Err(e) => AgentError::from_error(agent, &e, AgentErrorKind::Other),
        };
        match err.kind.handling() {
            Handling::Retry { after } if tries < TRIES => {
                let wait = after
                    .unwrap_or(Duration::from_secs(1 << tries))
                    .clamp(Duration::from_millis(100), Duration::from_secs(60));
                tokio::time::sleep(wait).await;
            }
            Handling::Cooldown { for_ } => {
                if let Some(k) = key {
                    Cooldowns::global().put(k, for_, err.clone());
                }
                return Err(crate::Error::Agent(err));
            }
            _ => return Err(crate::Error::Agent(err)),
        }
    }
}
