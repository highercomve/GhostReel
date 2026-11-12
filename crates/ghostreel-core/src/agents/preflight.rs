//! Agents' part of a preflight: is each agent a run would use there, offering its model, and not
//! resting after a used-up quota?

use super::spec::{AgentRole, AgentSpec};
use super::{Availability, check_availability};
use crate::config::Config;
use crate::paths::Paths;
use crate::preflight::{Check, Level, Scope};

/// One agent's check, as a preflight line.
pub async fn check_agent(paths: &Paths, config: &Config, spec: &AgentSpec) -> Check {
    let av: Availability = check_availability(paths, config, spec).await;
    let message = match av.level {
        Level::Ok => format!("{}: {}", spec.id, av.resolved),
        _ => format!("{}: {}", spec.id, av.detail),
    };
    let check = Check::new(av.level, Scope::Agent, spec.id.clone(), message);
    match av.error {
        Some(e) => check.with_error(e),
        None => check,
    }
}

/// Checks for the agents a run will use, each named once, with the role it is needed for.
///
/// A role that has another usable agent does not depend on one that fails: that agent's error is
/// a warning ("its jobs go to the others"). `optional` roles (an orchestrator the run can do
/// without) only ever warn.
pub async fn agent_checks(
    paths: &Paths,
    config: &Config,
    needed: &[(AgentRole, Vec<AgentSpec>)],
    optional: &[AgentRole],
) -> Vec<Check> {
    let mut seen: Vec<(String, Check)> = Vec::new();
    let mut out = Vec::new();
    for (role, specs) in needed {
        let mut checks = Vec::new();
        for spec in specs {
            let c = match seen.iter().find(|(id, _)| id == &spec.id) {
                Some((_, c)) => c.clone(),
                None => {
                    let c = check_agent(paths, config, spec).await;
                    seen.push((spec.id.clone(), c.clone()));
                    c
                }
            };
            checks.push(c);
        }
        let any_usable = checks.iter().any(|c| c.level < Level::Error);
        for mut c in checks {
            if c.level == Level::Error && (any_usable || optional.contains(role)) {
                c.level = Level::Warn;
                c.message = if optional.contains(role) && !any_usable {
                    format!("{} — the {role} step is skipped", c.message)
                } else {
                    format!("{} — the other {role} agents take its jobs", c.message)
                };
            }
            if c.level > Level::Ok && !out.iter().any(|o: &Check| o.subject == c.subject && o.message == c.message) {
                out.push(c);
            }
        }
    }
    out
}
