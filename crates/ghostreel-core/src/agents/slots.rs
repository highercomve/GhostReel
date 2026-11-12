//! Shared call slots, one set per model endpoint, for the whole process.
//!
//! Every model call — a clip worker, an orchestrator, a script chat turn, a frame description —
//! takes a slot for its endpoint first, so features running at the same time never ask one model
//! for more than it was configured to give. A local model is one endpoint with one slot however
//! it is reached: it is one process on one GPU, and a second one loaded beside it would not fit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
use tokio::sync::Notify;

use crate::runtime::VisionSetup;

/// Which model a call reaches.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EndpointKey {
    /// Any model run in-process (llama.cpp or MLX helper): one for the whole app.
    Local,
    Server { url: String, model: String },
    Cli { tool: String, model: String },
}

impl EndpointKey {
    /// The endpoint a resolved setup talks to; `None` when it is unavailable.
    pub fn for_setup(setup: &VisionSetup) -> Option<Self> {
        match setup {
            VisionSetup::Server(s) => {
                Some(EndpointKey::Server { url: s.url.trim_end_matches('/').to_string(), model: s.model.clone() })
            }
            VisionSetup::Cli(c) => Some(EndpointKey::Cli { tool: c.tool.clone(), model: c.model.clone() }),
            VisionSetup::Local { .. } | VisionSetup::Mlx { .. } => Some(EndpointKey::Local),
            VisionSetup::Unavailable(_) => None,
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, EndpointKey::Local)
    }

    /// Slots this endpoint may have when asked for `wanted`: a local model has one.
    pub fn capacity(&self, wanted: usize) -> usize {
        if self.is_local() { 1 } else { wanted.clamp(1, 64) }
    }

    pub fn label(&self) -> String {
        match self {
            EndpointKey::Local => "local model".into(),
            EndpointKey::Server { url, model } if model.is_empty() => format!("server {url}"),
            EndpointKey::Server { url, model } => format!("server {model} @ {url}"),
            EndpointKey::Cli { tool, model } if model.is_empty() => format!("cli {tool}"),
            EndpointKey::Cli { tool, model } => format!("cli {tool} · {model}"),
        }
    }
}

#[derive(Debug, Default)]
struct Counts {
    capacity: usize,
    in_use: usize,
}

#[derive(Debug, Default)]
struct Endpoint {
    counts: Mutex<Counts>,
    freed: Notify,
}

/// The right to make one call to an endpoint, given back when dropped.
#[derive(Debug)]
pub struct Slot {
    endpoint: Arc<Endpoint>,
    key: EndpointKey,
}

impl Slot {
    pub fn key(&self) -> &EndpointKey {
        &self.key
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        {
            let mut c = self.endpoint.counts.lock().unwrap_or_else(|e| e.into_inner());
            c.in_use = c.in_use.saturating_sub(1);
        }
        self.endpoint.freed.notify_waiters();
    }
}

/// One endpoint's state, for status views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EndpointUse {
    pub key: EndpointKey,
    pub capacity: usize,
    pub in_use: usize,
}

/// Slots per endpoint. [`SlotRegistry::global`] is the process's; tests make their own.
#[derive(Debug, Default)]
pub struct SlotRegistry {
    endpoints: Mutex<HashMap<EndpointKey, Arc<Endpoint>>>,
}

impl SlotRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The registry every feature in this process shares.
    pub fn global() -> Arc<SlotRegistry> {
        static GLOBAL: OnceLock<Arc<SlotRegistry>> = OnceLock::new();
        GLOBAL.get_or_init(SlotRegistry::new).clone()
    }

    /// The endpoint, with its capacity set to what this caller was configured with (the newest
    /// configuration wins; calls already running keep their slots).
    fn endpoint(&self, key: &EndpointKey, wanted: usize) -> Arc<Endpoint> {
        let ep = {
            let mut map = self.endpoints.lock().unwrap_or_else(|e| e.into_inner());
            map.entry(key.clone()).or_default().clone()
        };
        let grew = {
            let mut c = ep.counts.lock().unwrap_or_else(|e| e.into_inner());
            let cap = key.capacity(wanted);
            let grew = cap > c.capacity;
            c.capacity = cap;
            grew
        };
        if grew {
            ep.freed.notify_waiters();
        }
        ep
    }

    fn take(ep: &Arc<Endpoint>, key: &EndpointKey) -> Option<Slot> {
        let mut c = ep.counts.lock().unwrap_or_else(|e| e.into_inner());
        if c.in_use < c.capacity {
            c.in_use += 1;
            Some(Slot { endpoint: ep.clone(), key: key.clone() })
        } else {
            None
        }
    }

    /// A slot now, or `None` when the endpoint is full.
    pub fn try_acquire(&self, key: &EndpointKey, capacity: usize) -> Option<Slot> {
        let ep = self.endpoint(key, capacity);
        Self::take(&ep, key)
    }

    /// Wait for a slot.
    pub async fn acquire(&self, key: &EndpointKey, capacity: usize) -> Slot {
        let ep = self.endpoint(key, capacity);
        loop {
            let freed = ep.freed.notified();
            tokio::pin!(freed);
            // Registered before checking, so a release between the check and the wait is not lost.
            freed.as_mut().enable();
            if let Some(slot) = Self::take(&ep, key) {
                return slot;
            }
            freed.await;
        }
    }

    pub fn in_use(&self, key: &EndpointKey) -> usize {
        let map = self.endpoints.lock().unwrap_or_else(|e| e.into_inner());
        map.get(key).map(|ep| ep.counts.lock().unwrap_or_else(|e| e.into_inner()).in_use).unwrap_or(0)
    }

    /// Every endpoint seen so far with its capacity and calls in flight.
    pub fn snapshot(&self) -> Vec<EndpointUse> {
        let map = self.endpoints.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<EndpointUse> = map
            .iter()
            .map(|(k, ep)| {
                let c = ep.counts.lock().unwrap_or_else(|e| e.into_inner());
                EndpointUse { key: k.clone(), capacity: c.capacity, in_use: c.in_use }
            })
            .collect();
        v.sort_by_key(|e| e.key.label());
        v
    }
}

/// Wait for a slot on the process registry for whatever `setup` reaches. `None` for an
/// unavailable setup (the caller's own error path reports that).
pub async fn acquire_for(setup: &VisionSetup, capacity: usize) -> Option<Slot> {
    let key = EndpointKey::for_setup(setup)?;
    Some(SlotRegistry::global().acquire(&key, capacity).await)
}

/// One endpoint's slots on the process registry, cloned into every task that calls it.
#[derive(Debug, Clone)]
pub struct EndpointSlots {
    registry: Arc<SlotRegistry>,
    key: Option<EndpointKey>,
    capacity: usize,
}

impl EndpointSlots {
    pub fn for_setup(setup: &VisionSetup, capacity: usize) -> Self {
        Self { registry: SlotRegistry::global(), key: EndpointKey::for_setup(setup), capacity }
    }

    /// Wait for one slot (`None` for an unavailable setup, which never gets this far).
    pub async fn acquire(&self) -> Option<Slot> {
        let key = self.key.as_ref()?;
        Some(self.registry.acquire(key, self.capacity).await)
    }
}
