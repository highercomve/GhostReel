//! Asking again, with the score as the thing being climbed.
//!
//! One refine turn moves a cut about as far as one instruction goes. Handed Jev's build, the
//! local model reordered the beats and stopped — 62 → 64 editorial — because the only fault it
//! had been told about was a single mismatched beat, and it fixed that beat. The judge knew the
//! ending scored 0.49 and the opening 0.56 and said nothing, since `notes` only spoke below 0.4.
//! That is fixed in `judge::weakest`, and this is the other half: run the turn more than once and
//! keep the best one, rather than keeping the last.
//!
//! Keeping the *best* is the part that matters. A round can make a cut worse — agy's did, twice,
//! in the experiments this was built from — and a loop that keeps the last answer is a random
//! walk that ends wherever it happened to stop.

use crate::Error;
use crate::script::Script;

use super::{ChatContext, ChatEvent, run_turn};

/// What one pass produced.
#[derive(Debug, Clone)]
pub struct Round {
    pub n: usize,
    pub script_id: Option<i64>,
    /// The editorial score, or `None` when Jev is off — in which case there is nothing to climb
    /// and the last round is simply kept.
    pub total: Option<f64>,
}

/// The whole run: every round, and which one won.
#[derive(Debug, Clone)]
pub struct Refined {
    pub session_id: i64,
    pub rounds: Vec<Round>,
    pub best: Option<Round>,
    pub script: Option<Script>,
}

/// How many rounds in a row may fail to beat the best before stopping.
///
/// One is not enough: a round that overshoots is often followed by one that recovers, and the
/// model has just been told what it got wrong. Three is spending minutes to find out nothing.
const PATIENCE: usize = 2;

/// What each round asks for.
///
/// Deliberately not "improve this". The model already has the brief, the cut and — now — the
/// dimensions it is losing points on, attached to the last assistant message by the turn itself.
/// What it needs from here is permission to make a big change, because the failure being fixed is
/// a model handed a working cut and returning it nearly unchanged.
fn instruction(round: usize) -> String {
    if round == 1 {
        "Improve this cut. The editorial read above says exactly where it is losing points — work \
         on those, not on whatever is easiest. You may reorder beats, replace a quote with a \
         better one from another speaker, or change which shots play under a voice. Keep every \
         quote a whole sentence and keep it to the brief. Reply with the full script JSON."
            .to_string()
    } else {
        format!(
            "Round {round}. The read above is of your last version. It is still losing points on \
             the things it names — change something substantial this time rather than adjusting \
             the edges: a different opening line, a different closing line, or different footage \
             under the weakest beat. Keep every quote a whole sentence and keep it to the brief. \
             Reply with the full script JSON."
        )
    }
}

/// Refine a cut in an existing session, `rounds` times, and keep the best one.
///
/// The session is where the cut and its critique already live — `build::save_as_session` puts
/// them there — so this only has to keep asking.
pub async fn run(
    ctx: &mut ChatContext,
    project_id: i64,
    session_id: i64,
    rounds: usize,
    on_event: &mut (dyn FnMut(ChatEvent) + Send),
    on_round: &mut (dyn FnMut(&Round) + Send),
) -> Result<Refined, Error> {
    let mut all: Vec<Round> = Vec::new();
    let mut best: Option<Round> = None;
    let mut best_script: Option<Script> = None;
    let mut since_best = 0usize;

    for n in 1..=rounds.max(1) {
        let res = run_turn(ctx, project_id, Some(session_id), &instruction(n), on_event).await?;
        let round = Round { n, script_id: res.script_id, total: res.judgement.as_ref().map(|j| j.total) };
        on_round(&round);
        all.push(round.clone());

        // No judge means no fitness: keep the last, which is all "better" can mean here.
        let improved = match (round.total, best.as_ref().and_then(|b| b.total)) {
            (Some(now), Some(then)) => now > then,
            _ => true,
        };
        if improved {
            best_script = res.script.clone();
            best = Some(round);
            since_best = 0;
        } else {
            since_best += 1;
            if since_best >= PATIENCE {
                break;
            }
        }
    }

    Ok(Refined { session_id, rounds: all, best, script: best_script })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_round_is_told_to_change_something_substantial() {
        // The whole point of the loop: a model that returns what it was given has not refined it.
        let first = instruction(1);
        let later = instruction(3);
        assert!(first.contains("reorder beats"), "{first}");
        assert!(later.contains("Round 3"), "{later}");
        assert!(later.contains("substantial"), "{later}");
        for i in [first, later] {
            assert!(i.contains("whole sentence"), "the invariant survives every round");
            assert!(i.contains("brief"), "drifting off-brief was the cost measured last time");
        }
    }
}
