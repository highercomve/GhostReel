//! Fitting a cut to its length: pictures give way first, speech is never scaled, and a whole quote
//! goes only when nothing else reaches the ceiling. Also the clean-ups that change which clips are
//! in the cut at all: footage past a video's end, repeated footage, back-to-back cuts of one take.

use crate::db::Db;
use crate::script::{Script, ScriptClip, Side};

use super::pictures::clamp_beds_to_beats;
use super::{clip_has_speech, video_duration};

/// Last resort when the model ignores the target: shorten every clip proportionally (keeping its
/// in point, never below `script.min_trimmed_clip_s`). Returns whether anything changed.
pub fn trim_to_target(script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    trim_to_target_with(script, |_| false, cfg)
}

/// [`trim_to_target`] that leaves `keep` clips (people speaking) whole and shortens the others.
/// Fit a cut to its target in one pass.
///
/// Three mechanisms used to shorten a script independently — speech capped to a share of the
/// target, b-roll trimmed proportionally, and a final pass doing both again — each measuring from
/// its own view of the total. Together they overshot badly: a cut sitting at 58.6 s against a 60 s
/// target came out at 29.5 s. One pass, measuring once, cannot do that.
///
/// Over target, the pictures give way first and the speaking clips only if that was not enough;
/// under it, the pictures are held longer. Speech is never stretched: a sentence is as long as it
/// is, and the rest is dead air.
pub fn fit_to_target(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    fit_to_target_leaving(db, script, cfg, 0.0)
}

/// [`fit_to_target`] leaving `reserve_s` of the target for what the chain adds after the fit (the
/// closing hold): the fit is of the length that plays, and a cut fitted without its hold came out
/// two seconds over on every finish.
pub fn fit_to_target_leaving(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig, reserve_s: f64) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0).map(|t| (t - reserve_s).max(t * 0.5)) else {
        return false;
    };
    let total = script.total_duration_s();
    let slack = target * (cfg.final_target_tolerance - 1.0);
    if (total - target).abs() <= slack {
        return false;
    }
    if total < target {
        let grew = grow_to_target(db, script, cfg, target);
        clamp_beds_to_beats(db, script, cfg);
        return grew;
    }

    // In a montage nothing is speech: every shot may be trimmed.
    let speaking: Vec<bool> = script
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .map(|c| !c.movable(Side::Out) || (!cfg.style.broll && clip_has_speech(db, c.video_id, c.in_s, c.out_s)))
        .collect();
    let spoken: f64 = script
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .zip(&speaking)
        .filter(|(_, s)| **s)
        .map(|(c, _)| c.out_s - c.in_s)
        .sum();
    let pictures = total - spoken;

    // What the pictures must come down to, never below the floor for each shot.
    let picture_floor: f64 = speaking.iter().filter(|s| !**s).count() as f64 * cfg.min_trimmed_clip_s;
    let want_pictures = (target - spoken).max(picture_floor).min(pictures);
    let picture_factor = if pictures > 0.0 { want_pictures / pictures } else { 1.0 };

    // Speech is never scaled. Scaling it cut people off mid-sentence to hit a number: three
    // interview clips in a row ended inside a word. A sentence is the unit here, and a cut that
    // runs a few seconds long is worth more than one that lands exactly and sounds broken.
    let speech_factor = 1.0;

    let mut changed = false;
    let mut i = 0;
    for beat in &mut script.beats {
        for c in &mut beat.clips {
            let factor = if speaking[i] { speech_factor } else { picture_factor };
            i += 1;
            if factor >= 1.0 || !c.movable(Side::Out) {
                continue;
            }
            let len = c.out_s - c.in_s;
            let new_len = (len * factor).max(cfg.min_trimmed_clip_s).min(len);
            if (new_len - len).abs() > 0.05 {
                c.out_s = c.in_s + new_len;
                changed = true;
            }
        }
    }
    clamp_beds_to_beats(db, script, cfg);

    // Pictures alone cannot save a cut made of speech. A model that chose seven quotes of
    // twenty-five seconds is 349% over and there is nothing to scale: the rule that speech is
    // never scaled is what stops it, and rightly. What it can lose is a whole quote — that ends
    // on a sentence by construction, so dropping one breaks nothing.
    //
    // The opening and the closing stay: they are the two positions a viewer notices, and a piece
    // that keeps its hook and its ending and loses a middle is still the piece. Middles go from
    // the back, so what survives is what was set up earliest.
    //
    // But only when nothing smaller reaches the ceiling. Dropping was the first answer to any
    // overrun the pictures could not absorb, so a cut 0.2 s over lost a 12 s quote, came in well
    // short, and the next finish grew the b-roll back: a fit that falls off a cliff and then
    // climbs back is one whose output depends on noise. The breath each quote holds after its last
    // sentence is the give a cut of speech has; it goes first.
    changed |= trim_speech_tails(db, script, cfg, target);
    if drop_beats_within(db, script, cfg, target) {
        changed = true;
        // A quote is long, so the cut may now be short of its target. Holding the pictures to make
        // it up is part of the same fit: left to the next finish it would find the cut short and
        // grow it then, which is a finish that does not settle.
        grow_to_target(db, script, cfg, target);
        clamp_beds_to_beats(db, script, cfg);
    }
    changed
}

/// Take the breath after the last sentence off speaking clips, from the last beat back, toward the
/// target, the way the pictures give way. A clip keeps its sentence and the overrun after it (the
/// word's decay, `speech_overrun_s`); what goes is the hold of the person `pad_speech` added past
/// that. An end written on a word, or a clip to keep exactly, is left. Returns whether anything
/// moved.
fn trim_speech_tails(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig, target: f64) -> bool {
    let total = script.total_duration_s();
    if total <= target * cfg.final_target_tolerance {
        return false;
    }
    let mut over = total - target;
    if over <= 0.0 {
        return false;
    }
    let mut changed = false;
    for beat in script.beats.iter_mut().rev() {
        for c in beat.clips.iter_mut().rev() {
            if over <= 0.0 {
                return changed;
            }
            if c.audio != crate::script::Audio::Source || !c.movable(Side::Out) {
                continue;
            }
            let edges = super::edges::sentence_edges(db, c.video_id);
            let inside: Vec<usize> =
                (0..edges.len()).filter(|&i| edges[i].1 > c.in_s + 0.05 && edges[i].0 < c.out_s - 0.05).collect();
            let Some((i, end)) = super::edges::finished_end(&edges, &inside, c.out_s, cfg.speech_overrun_s) else {
                continue;
            };
            // Only a breath: a clip that stops inside the next sentence carries words, and what
            // people say is never trimmed to hit a number.
            if edges.get(i + 1).is_some_and(|n| n.0 < c.out_s - super::edges::EDGE_SLACK_S) {
                continue;
            }
            let keep = end + cfg.speech_overrun_s;
            let give = (c.out_s - keep).min(over);
            if give > 0.05 && keep - c.in_s >= cfg.min_clip_s {
                c.out_s -= give;
                over -= give;
                changed = true;
            }
        }
    }
    changed
}

/// Drop whole beats, from the middle and the back, until the cut is near its target.
///
/// Returns whether anything was dropped. Never leaves fewer than two beats: below that it is no
/// longer the script the model wrote, and the score reports the overrun instead.
#[cfg(test)]
pub(crate) fn drop_beats_to_target(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    drop_beats_within(db, script, cfg, target)
}

/// [`drop_beats_to_target`] against `target`.
fn drop_beats_within(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig, target: f64) -> bool {
    let ceiling = target * cfg.target_overshoot;
    if script.total_duration_s() <= target * cfg.final_target_tolerance || script.beats.len() <= 2 {
        return false;
    }

    // Dropping has to stop before the cut stops being the model's. Half the clips is the line the
    // replay eval draws — "a cut assembled from a third of what the model chose is not the cut it
    // wrote" — and a wall of eleven talking heads hit it immediately.
    //
    // Decided up front, not greedily, because a greedy loop that stops at the floor leaves the
    // cut still over the ceiling, and the next pass recomputes a lower floor and drops again:
    // repairing an already-repaired script took it from 53.0 s to 43.4 s. Either the target is
    // reachable while keeping half the clips, or nothing is dropped and the score reports the
    // overrun.
    let floor = script.clip_count().div_ceil(2);
    let mut plan: Vec<usize> = Vec::new();
    let mut length = script.total_duration_s();
    let mut clips = script.clip_count();
    let mut beats = script.beats.len();
    // Walk the middles from the back, as `remove` would. Over the ceiling a beat always goes;
    // under it, only when the cut stays at or above its target and lands nearer it. Stopping at the
    // ceiling left a 60 s cut at 74.4 s when the next quote would have brought it to 61.8 s. A cut
    // a little over keeps its quotes: losing one would leave it short, and growing the pictures
    // back to make that up is a fit the next finish moves again.
    for i in (1..script.beats.len() - 1).rev() {
        if beats <= 2 {
            break;
        }
        let beat = &script.beats[i];
        if clips - beat.clips.len() < floor {
            break;
        }
        let len = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum::<f64>();
        if length <= ceiling && (length - len < target || (length - len - target).abs() >= (length - target).abs()) {
            break;
        }
        length -= len;
        clips -= beat.clips.len();
        beats -= 1;
        plan.push(i);
    }
    if length > ceiling {
        return false; // cannot get there without gutting it
    }

    let mut dropped = false;
    for i in plan {
        script.beats.remove(i);
        dropped = true;
    }
    if dropped {
        clamp_beds_to_beats(db, script, cfg);
    }
    dropped
}

/// Hold the b-roll longer when the cut came in short.
///
/// Two independent mechanisms shorten a script — speech is capped at a share of the target, and
/// b-roll is trimmed proportionally — so together they undershoot, and a 60 s teaser lands at 49 s
/// with no way back. Speaking clips are left alone: they are as long as the sentence is. Scenery
/// can simply be held, up to what the footage actually has.
fn grow_to_target(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig, target: f64) -> bool {
    let total = script.total_duration_s();
    if total >= target * 0.97 {
        return false;
    }
    let mut room: Vec<(usize, usize, f64)> = Vec::new();
    for (bi, beat) in script.beats.iter().enumerate() {
        // A narrated beat can only be held as long as its voice-over lasts, or the picture runs
        // on in silence after the last word — the very gap content_issues complains about.
        let mut beat_room = f64::MAX;
        let words = beat.narration.as_deref().map(|n| n.split_whitespace().count()).unwrap_or(0);
        let beat_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
        if words > 0 {
            let spoken_s = words as f64 / cfg.narration_words_per_s;
            beat_room = (spoken_s - beat_s).max(0.0);
        } else if let Some(len) = super::pictures::bed_picture_len(script, bi, cfg) {
            // The same for a voice under the pictures: held past it and its breath, they hang in
            // silence, and `trim_pictures_to_bed` cuts them back on the next finish.
            beat_room = (len - beat_s).max(0.0);
        }
        if beat_room <= 0.1 {
            continue;
        }
        let mut left = beat_room;
        for (ci, c) in beat.clips.iter().enumerate() {
            if left <= 0.1 {
                break;
            }
            if !c.movable(Side::Out) || clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                continue;
            }
            let duration: Option<f64> = db
                .conn
                .query_row("SELECT duration_s FROM videos WHERE id = ?1", [c.video_id], |r| r.get(0))
                .ok()
                .flatten();
            let ceiling = duration.unwrap_or(c.out_s).min(c.in_s + cfg.max_clip_s);
            let spare = (ceiling - c.out_s).min(left);
            if spare > 0.1 {
                room.push((bi, ci, spare));
                left -= spare;
            }
        }
    }
    let spare_total: f64 = room.iter().map(|(_, _, s)| s).sum();
    if spare_total <= 0.0 {
        return false;
    }
    let needed = target - total;
    let share = (needed / spare_total).min(1.0);
    let mut changed = false;
    for (bi, ci, spare) in room {
        let add = spare * share;
        if add > 0.1 {
            script.beats[bi].clips[ci].out_s += add;
            changed = true;
        }
    }
    changed
}

pub(crate) fn trim_to_target_with(
    script: &mut Script,
    keep: impl Fn(&ScriptClip) -> bool,
    cfg: &crate::config::ScriptConfig,
) -> bool {
    trim_to_target_within(script, keep, cfg.target_overshoot, cfg)
}

/// `trim_to_target_with` with an explicit tolerance. The draft stage is loose — the model may yet
/// redraft — but the last pass before saving has no such luxury: whatever it leaves is the length
/// the user gets, and a cut left 18% long is one the report then complains about.
fn trim_to_target_within(
    script: &mut Script,
    keep: impl Fn(&ScriptClip) -> bool,
    tolerance: f64,
    cfg: &crate::config::ScriptConfig,
) -> bool {
    let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) else { return false };
    let total = script.total_duration_s();
    if total <= target * tolerance {
        return false;
    }
    let kept: f64 = script
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .filter(|c| keep(c) || !c.movable(Side::Out))
        .map(|c| c.out_s - c.in_s)
        .sum();
    let flexible = total - kept;
    if flexible <= 0.0 {
        return false;
    }
    let factor = ((target - kept) / flexible).clamp(0.0, 1.0);
    let mut changed = false;
    for clip in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        if keep(clip) || !clip.movable(Side::Out) {
            continue;
        }
        let len = clip.out_s - clip.in_s;
        let new_len = (len * factor).max(cfg.min_trimmed_clip_s).min(len);
        if new_len < len {
            clip.out_s = clip.in_s + new_len;
            changed = true;
        }
    }
    changed
}

/// Keep clips inside their video; drop the ones that start past its end.
pub fn clamp_to_duration(db: &Db, script: &mut Script) {
    for beat in &mut script.beats {
        beat.clips.retain_mut(|c| {
            if let Some(d) = video_duration(db, c.video_id) {
                c.out_s = c.out_s.min(d);
                c.in_s = c.in_s.max(0.0);
            }
            c.out_s - c.in_s >= 1.0 || (c.pinned() && c.out_s > c.in_s)
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
}

/// A later clip overlapping footage an earlier clip already showed (by more than 1 s) is dropped.
/// Clips that play footage an earlier clip already played (more than a second of the same video):
/// dropped, the first use kept. The same rule for every cut, however it was written (chat, save_script).
pub fn drop_repeated_footage(script: &mut Script) -> usize {
    let mut used: Vec<(i64, f64, f64)> = Vec::new();
    let mut dropped = 0;
    for beat in &mut script.beats {
        beat.clips.retain(|c| {
            let repeat = used.iter().any(|&(v, a, b)| v == c.video_id && c.out_s.min(b) - c.in_s.max(a) > 1.0);
            if repeat {
                dropped += 1;
            } else {
                used.push((c.video_id, c.in_s, c.out_s));
            }
            !repeat
        });
    }
    script.beats.retain(|b| !b.clips.is_empty());
    dropped
}
