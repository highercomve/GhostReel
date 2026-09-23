//! Turning a draft into a finished script, outside the turn that produced it.
//!
//! Everything here used to live inline in `run_turn`, which meant nothing else could reproduce a
//! finished cut: not a test, not an eval, not a second candidate in a best-of-N. The order is the
//! one `AGENTS.md` documents and the one the passes were debugged into — this is a move, and the
//! existing tests are the proof.

use crate::Error;
use crate::db::Db;
use crate::script::{Issue, IssueSeverity, Script, validate};

/// Everything applied once narration exists, up to but not including saving.
///
/// Returns `None` when there is nothing left to finish — every clip dropped — which is the turn's
/// "unable to assemble a script" path.
pub fn finish_script(
    db: &Db,
    project_id: i64,
    s: &mut Script,
    enforce_target: bool,
    cfg: &crate::config::ScriptConfig,
) -> Result<Option<Vec<Issue>>, Error> {
    if s.clip_count() == 0 || s.beats.is_empty() {
        return Ok(None);
    }
    let mut issues = Vec::new();
    let note = |issues: &mut Vec<Issue>, message: String| {
        issues.push(Issue { severity: IssueSeverity::Info, beat_id: None, clip_index: None, message });
    };

    // Anything a previous repair added comes off first: the fit below counts clips, and a
    // picture this pass put there is not one the model chose.
    super::drop_closing_picture(s);

    // A montage has no speech to end on, pad or lay under anything: it is fitted and checked.
    if cfg.style.broll {
        super::clamp_to_duration(db, s);
        if enforce_target {
            let before = s.total_duration_s();
            if super::fit_to_target(db, s, cfg) {
                note(&mut issues, format!("fitted to target: {before:.1} s → {:.1} s", s.total_duration_s()));
            }
        }
        issues.extend(super::content_issues(db, s, cfg));
        issues.extend(validate(db, project_id, s)?);
        return Ok(Some(issues));
    }

    let _ = super::snap_to_segments(db, s)?;
    super::pad_speech(db, s, cfg);
    super::clamp_to_duration(db, s);
    let muted = super::mute_silent_clips(db, s);
    if muted > 0 {
        note(&mut issues, format!("muted {muted} clip(s) without speech under the narration"));
    }

    // snap_to_segments and pad_speech pull clips out to whole sentences, undoing the trim the
    // draft repair just made: an interview-led cut came out 37% over target because the last word
    // on the subject was padding, not trimming. Fit once more, now clips are their final length.
    if enforce_target {
        let before = s.total_duration_s();
        if super::fit_to_target(db, s, cfg) {
            // Trimming can cut a sentence short again, and snapping only reaches 0.75 s.
            let _ = super::snap_to_segments(db, s)?;
            note(&mut issues, format!("fitted to target after padding: {before:.1} s → {:.1} s", s.total_duration_s()));
        }
    }

    // These finish the cut rather than fit it, so they run whether or not the fit moved anything.
    // They used to sit inside that `if`, which meant a script already close to its target — the
    // good ones — was the only kind that never got them.
    // Before the sentences, because this moves an edge and `end_on_sentences` puts it back on a
    // boundary. It cannot run earlier: `pad_speech` grows a clip into the pauses either side, so a
    // trim made before padding is simply undone, and the two passes then take turns — the replay
    // eval caught exactly that, a draft walking from 52.5 s to 43.4 s on a second repair that
    // should have found nothing to do.
    let turns = super::end_on_turns(db, s, cfg);
    if turns > 0 {
        note(&mut issues, format!("ended {turns} range(s) where the speaker stopped, before the reply over the top"));
    }
    let mended = super::end_on_sentences(db, s, cfg);
    if mended > 0 {
        note(&mut issues, format!("put {mended} range(s) back on whole sentences"));
    }
    let shortened = super::trim_pictures_to_bed(s, cfg);
    if shortened > 0 {
        note(&mut issues, format!("cut the pictures back to the voice in {shortened} beat(s)"));
    }
    let held = super::hold_the_last_picture(db, project_id, s, cfg);
    if held > 0.0 {
        note(&mut issues, format!("held the closing picture for {held:.1} s of quiet"));
    }
    super::clamp_beds_to_beats(db, s, cfg);

    issues.extend(super::content_issues(db, s, cfg));
    issues.extend(validate(db, project_id, s)?);
    Ok(Some(issues))
}
