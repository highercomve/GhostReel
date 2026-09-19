//! Progress and time-remaining estimates for indexing runs.
//!
//! A run is a sequence of phases (hash files → probe → later: transcribe, describe, embed).
//! Each phase has a work total in its own units (files, or seconds of video for the slow
//! stages) and a *rate* in units per second. The rate starts from what previous runs measured
//! (persisted in `meta` as `rate.<phase>`), and blends towards the rate observed in this run as
//! work completes — so the estimate is sensible from the first second and self-corrects.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rusqlite::OptionalExtension;
use serde::Serialize;

use crate::Error;
use crate::db::Db;

/// How many completed units weigh as much as the prior rate.
const PRIOR_WEIGHT: f64 = 3.0;
/// Minimum interval between progress events for the same phase.
const EMIT_EVERY: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Progress {
    /// Current phase name (`hash`, `probe`, …).
    pub phase: String,
    pub phase_done: u64,
    pub phase_total: u64,
    /// Overall completion of the run, 0.0–1.0 (time-weighted across phases).
    pub fraction: f64,
    /// Estimated seconds remaining for the whole run; `None` until there is work to estimate.
    pub eta_secs: Option<f64>,
    pub elapsed_secs: f64,
    /// File most recently handled.
    pub current: Option<PathBuf>,
    /// The work has no measurable end — a model generating a draft, where nothing can be counted
    /// until it stops. The bar animates instead of claiming a percentage it does not know.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub indeterminate: bool,
}

#[derive(Debug, Clone)]
struct Phase {
    name: &'static str,
    total: u64,
    done: u64,
    prior_rate: f64,
    started: Option<Instant>,
    finished: Option<Duration>,
}

impl Phase {
    fn elapsed(&self) -> Option<Duration> {
        self.finished.or_else(|| self.started.map(|s| s.elapsed()))
    }

    /// Units per second: prior blended with what this run measured.
    fn rate(&self) -> f64 {
        let measured = self
            .elapsed()
            .map(|e| e.as_secs_f64())
            .filter(|&secs| secs > 0.05 && self.done > 0)
            .map(|secs| self.done as f64 / secs);
        match measured {
            Some(m) => (self.prior_rate * PRIOR_WEIGHT + m * self.done as f64) / (PRIOR_WEIGHT + self.done as f64),
            None => self.prior_rate,
        }
    }
}

pub struct Tracker {
    phases: Vec<Phase>,
    current: usize,
    started: Instant,
    last_emit: Option<Instant>,
}

impl Tracker {
    /// `phases`: names in execution order with a default rate (units/s) for machines that have
    /// never indexed before.
    pub fn new(phases: &[(&'static str, f64)]) -> Self {
        Self {
            phases: phases
                .iter()
                .map(|&(name, prior_rate)| Phase {
                    name,
                    total: 0,
                    done: 0,
                    prior_rate: prior_rate.max(1e-6),
                    started: None,
                    finished: None,
                })
                .collect(),
            current: 0,
            started: Instant::now(),
            last_emit: None,
        }
    }

    /// Replace default rates with the ones measured by previous runs.
    pub fn load_rates(&mut self, db: &Db) -> Result<(), Error> {
        for p in &mut self.phases {
            let v: Option<String> = db
                .conn
                .query_row("SELECT value FROM meta WHERE key = ?1", [format!("rate.{}", p.name)], |r| r.get(0))
                .optional()?;
            if let Some(rate) = v.and_then(|s| s.parse::<f64>().ok()).filter(|r| r.is_finite() && *r > 0.0) {
                p.prior_rate = rate;
            }
        }
        Ok(())
    }

    /// Remember this run's measured rates (smoothed) for the next run's first estimate.
    pub fn save_rates(&self, db: &Db) -> Result<(), Error> {
        for p in &self.phases {
            let Some(elapsed) = p.elapsed().map(|e| e.as_secs_f64()) else { continue };
            // Too little work to be a meaningful measurement.
            if p.done < 3 || elapsed < 0.2 {
                continue;
            }
            let measured = p.done as f64 / elapsed;
            let smoothed = 0.7 * p.prior_rate + 0.3 * measured;
            db.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [format!("rate.{}", p.name), smoothed.to_string()],
            )?;
        }
        Ok(())
    }

    fn idx(&self, name: &str) -> usize {
        self.phases.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("unknown phase {name}"))
    }

    pub fn set_total(&mut self, phase: &str, total: u64) {
        let i = self.idx(phase);
        let p = &mut self.phases[i];
        p.total = total.max(p.done);
    }

    /// Enter `phase` (earlier phases are considered finished).
    pub fn start(&mut self, phase: &str) {
        let i = self.idx(phase);
        for p in &mut self.phases[..i] {
            if p.finished.is_none() {
                p.finished = p.elapsed().or(Some(Duration::ZERO));
            }
            p.total = p.done;
        }
        self.phases[i].started.get_or_insert_with(Instant::now);
        self.current = i;
        self.last_emit = None;
    }

    pub fn advance(&mut self, phase: &str, units: u64) {
        let i = self.idx(phase);
        let p = &mut self.phases[i];
        p.done += units;
        p.total = p.total.max(p.done);
    }

    /// Mark everything finished.
    pub fn finish(&mut self) {
        for p in &mut self.phases {
            if p.started.is_some() && p.finished.is_none() {
                p.finished = p.elapsed();
            }
            p.total = p.done;
        }
        self.current = self.phases.len().saturating_sub(1);
    }

    pub fn snapshot(&self, current: Option<PathBuf>) -> Progress {
        let (mut done_time, mut remaining_time) = (0.0, 0.0);
        for p in &self.phases {
            let rate = p.rate();
            done_time += p.done as f64 / rate;
            remaining_time += p.total.saturating_sub(p.done) as f64 / rate;
        }
        let total_time = done_time + remaining_time;
        let fraction = if total_time <= 0.0 { if self.all_zero() { 0.0 } else { 1.0 } } else { done_time / total_time };
        let finished = self.phases.iter().all(|p| p.done >= p.total) && !self.all_zero();
        let phase = &self.phases[self.current];
        Progress {
            phase: phase.name.to_string(),
            phase_done: phase.done,
            phase_total: phase.total,
            fraction: if finished { 1.0 } else { fraction.clamp(0.0, 1.0) },
            eta_secs: if self.all_zero() { None } else { Some(remaining_time) },
            elapsed_secs: self.started.elapsed().as_secs_f64(),
            current,
            indeterminate: false,
        }
    }

    fn all_zero(&self) -> bool {
        self.phases.iter().all(|p| p.total == 0)
    }

    /// Throttle: true at most every 250 ms (and always right after a phase change).
    pub fn should_emit(&mut self) -> bool {
        let now = Instant::now();
        match self.last_emit {
            Some(t) if now.duration_since(t) < EMIT_EVERY => false,
            _ => {
                self.last_emit = Some(now);
                true
            }
        }
    }
}

/// Human "time left": "less than a minute", "about 4 min", "about 1 h 20 min".
pub fn eta_text(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 10 {
        "a few seconds".into()
    } else if s < 60 {
        format!("{s} s")
    } else if s < 3600 {
        format!("about {} min", s.div_ceil(60))
    } else {
        format!("about {} h {:02} min", s / 3600, (s % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_wording() {
        assert_eq!(eta_text(3.0), "a few seconds");
        assert_eq!(eta_text(42.4), "42 s");
        assert_eq!(eta_text(61.0), "about 2 min");
        assert_eq!(eta_text(4800.0), "about 1 h 20 min");
    }

    #[test]
    fn eta_uses_prior_before_any_work() {
        let mut t = Tracker::new(&[("hash", 10.0), ("probe", 2.0)]);
        t.set_total("hash", 20);
        t.set_total("probe", 10);
        t.start("hash");
        let p = t.snapshot(None);
        // 20/10 + 10/2 = 7 s
        assert!((p.eta_secs.unwrap() - 7.0).abs() < 1e-9);
        assert_eq!(p.fraction, 0.0);
        assert_eq!((p.phase.as_str(), p.phase_done, p.phase_total), ("hash", 0, 20));
    }

    #[test]
    fn fraction_is_time_weighted_and_reaches_one() {
        let mut t = Tracker::new(&[("hash", 100.0), ("probe", 1.0)]);
        t.set_total("hash", 100);
        t.set_total("probe", 10);
        t.start("hash");
        t.advance("hash", 100);
        t.start("probe");
        let p = t.snapshot(None);
        // Hashing everything is ~1 s of an ~11 s run: fraction must be small, not 100/110.
        assert!(p.fraction < 0.2, "{}", p.fraction);
        t.advance("probe", 10);
        t.finish();
        let p = t.snapshot(None);
        assert_eq!(p.fraction, 1.0);
        assert!(p.eta_secs.unwrap() < 1e-9);
    }

    #[test]
    fn nothing_to_do() {
        let mut t = Tracker::new(&[("hash", 1.0)]);
        t.start("hash");
        let p = t.snapshot(None);
        assert_eq!((p.fraction, p.eta_secs), (0.0, None));
    }

    #[test]
    fn measured_rate_pulls_the_estimate() {
        let mut t = Tracker::new(&[("probe", 1000.0)]);
        t.set_total("probe", 40);
        t.start("probe");
        std::thread::sleep(Duration::from_millis(120));
        t.advance("probe", 20); // ~166/s measured vs a wildly optimistic prior
        let eta = t.snapshot(None).eta_secs.unwrap();
        assert!(eta > 20.0 / 1000.0 * 2.0, "blended rate must be well below the prior: eta={eta}");
    }

    #[test]
    fn rates_persist_between_runs() {
        let db = Db::open_in_memory().unwrap();
        let mut t = Tracker::new(&[("probe", 5.0)]);
        t.set_total("probe", 10);
        t.start("probe");
        std::thread::sleep(Duration::from_millis(250));
        t.advance("probe", 10);
        t.finish();
        t.save_rates(&db).unwrap();

        let mut next = Tracker::new(&[("probe", 5.0)]);
        next.load_rates(&db).unwrap();
        let saved = next.phases[0].prior_rate;
        assert!(saved > 5.0 && saved < 40.0, "smoothed toward ~40/s measured: {saved}");
    }

    #[test]
    fn throttles_events() {
        let mut t = Tracker::new(&[("hash", 1.0)]);
        assert!(t.should_emit());
        assert!(!t.should_emit());
        t.start("hash");
        assert!(t.should_emit(), "phase change resets the throttle");
    }
}
