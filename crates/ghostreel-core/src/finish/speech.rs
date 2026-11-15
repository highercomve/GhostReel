//! The voice's own edges: a quote ends where its speaker stopped, on a whole sentence, with a breath
//! either side; clips nobody speaks in go quiet under narration.

use rusqlite::params;

use crate::db::Db;
use crate::script::{Script, Side};

use super::edges::{in_owner, out_owner, sentence_edges, snap_range};
use super::{MIN_CLIP_WORDS, video_duration};
use crate::sentence_split::CUT_INTO_SENTENCE_S;

/// Acknowledgements in a row, which is where one person stops talking and another starts.
///
/// Nothing in the index says who is speaking. Diarisation would, and `interviewer.rs` catches the
/// lines that give the interviewer away by their words, but neither reaches the ordinary case: a
/// subject finishes a thought, both people say "Yeah", and the interviewer starts a story of their
/// own. Measured on the Greet Mag tape, the beat meant to end on
///
/// > "And we love the deer running around too." … "So, yeah, it's beautiful."
///
/// ran three segments further, through "Yeah." "Yeah." into "I used to live up on like Spicewood
/// Springs road." — the interviewer, indistinguishable from the subject as text (Jev scores it
/// 0.20) and impossible to catch by word count, since it is a full ten-word sentence. What gives
/// it away is the pair of acknowledgements in front of it.
const TURN_BREAK_ACKS: usize = 2;

/// How close to the end of a range the handover has to be to count as one.
///
/// Without this the pass is not idempotent, which the replay eval caught: having trimmed at one
/// run of acknowledgements it finds an earlier one on the next pass and trims again, and a
/// recorded draft walked from 52.5 s to 43.4 s on a repair that should have found nothing to do.
/// Anchoring the handover to the end of the range fixes that by construction — once the range
/// stops where the speaker stopped, the acknowledgements are outside it and there is nothing left
/// to find.
const TURN_TAIL_S: f64 = 4.0;

/// Whether a transcript line is an acknowledgement rather than something said.
fn is_acknowledgement(text: &str) -> bool {
    let n = text.split_whitespace().count();
    n > 0 && n < MIN_CLIP_WORDS
}

/// End a speech range before somebody else takes over.
///
/// A run of `TURN_BREAK_ACKS` acknowledgements splits the range, and the range keeps the longer
/// side. That last part is what makes this safe: two "Yeah"s in the middle of somebody's own
/// answer are them thinking, and the answer continues for longer than its opening — trimming
/// there would throw the answer away to keep its first sentence. When the acknowledgements are a
/// handover instead, what follows them is a scrap: Adrienne speaks for eight seconds, both say
/// "Yeah", and one second of the interviewer follows.
///
/// This runs before `end_on_sentences`, which puts the new edge back on a sentence boundary, and
/// like every pass that moves a clip it must be followed by `clamp_beds_to_beats`.
pub fn end_on_turns(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut fixed = 0usize;
    for beat in &mut script.beats {
        let trim = |video_id: i64, in_s: f64, out_s: &mut f64| {
            let Some(end) = turn_end(db, video_id, in_s, *out_s, cfg) else { return false };
            if end < *out_s - 0.05 {
                *out_s = end;
                return true;
            }
            false
        };
        for c in beat.clips.iter_mut() {
            if c.audio == crate::script::Audio::Source && c.movable(Side::Out) && trim(c.video_id, c.in_s, &mut c.out_s)
            {
                fixed += 1;
            }
        }
        if let Some(bed) = &mut beat.bed
            && bed.movable(Side::Out)
        {
            let (v, i) = (bed.video_id, bed.in_s);
            if trim(v, i, &mut bed.out_s) {
                fixed += 1;
            }
        }
    }
    fixed
}

/// Where the turn inside `[in_s, out_s]` ends, if somebody else takes over before `out_s`.
fn turn_end(db: &Db, video_id: i64, in_s: f64, out_s: f64, cfg: &crate::config::ScriptConfig) -> Option<f64> {
    let segs: Vec<(f64, f64, String)> = db
        .conn
        .prepare_cached(
            "SELECT start_s, end_s, text FROM transcript_segments
             WHERE video_id = ?1 AND end_s > ?2 + 0.05 AND start_s < ?3 - 0.05 ORDER BY start_s",
        )
        .ok()?
        .query_map(params![video_id, in_s, out_s], |r| {
            Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?))
        })
        .ok()?
        .flatten()
        .collect();
    if segs.len() < TURN_BREAK_ACKS + 2 {
        return None;
    }
    // Walk back from the end looking for the last run of acknowledgements, and keep whatever was
    // said before it. Starting from the end rather than the start is what makes this the *tail*:
    // the first such run in a long answer is a pause, the last one is the handover.
    let mut i = segs.len();
    while i > 0 {
        i -= 1;
        if !is_acknowledgement(&segs[i].2) {
            continue;
        }
        let run_end = i + 1;
        while i > 0 && is_acknowledgement(&segs[i - 1].2) {
            i -= 1;
        }
        if run_end - i < TURN_BREAK_ACKS || i == 0 {
            continue;
        }
        // The handover is at the end of the range or it is not a handover.
        if out_s - segs[i].0 > TURN_TAIL_S {
            return None;
        }
        // Something is said after the run, or the run is simply trailing filler that
        // `end_on_sentences` will tidy on its own.
        if !segs[run_end..].iter().any(|s| !is_acknowledgement(&s.2)) {
            continue;
        }
        let end = segs[i - 1].1;
        // Keep the longer side. Everything after the acknowledgements has to be the scrap for
        // this to be a handover rather than a pause the speaker took in their own answer.
        if out_s - segs[run_end].0 >= end - in_s {
            continue;
        }
        // Never trim a speech range below what a clip is allowed to be.
        if end - in_s < cfg.min_clip_s {
            return None;
        }
        return Some(end);
    }
    None
}

/// Make every clip that carries someone's voice begin and end on a whole sentence.
///
/// Snapping only reaches 0.75 s and the fit that runs last can trim far more than that, so a clip
/// ended wherever the arithmetic landed: four in one run stopped between 0.27 s and 2.65 s before
/// the speaker finished, one of them mid-word. The length is a target; a sentence is not.
///
/// An edge moves out to the sentence boundary when that is within `max_speech_extend_s`, and back
/// to the previous one when it is further, so a clip is never left in the middle of a thought.
pub fn end_on_sentences(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut fixed = 0usize;
    let breaths: Vec<f64> = (0..script.beats.len()).map(|b| super::pictures::breath_after(script, b, cfg)).collect();
    for (beat, breath) in script.beats.iter_mut().zip(breaths) {
        let beat_len: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        // The breath after a bed's voice is the pictures' own: the voice does not grow into it.
        let voice_len = beat_len - breath;

        for c in beat.clips.iter_mut() {
            if c.audio != crate::script::Audio::Source || c.pinned() {
                continue;
            }
            let (keep_in, keep_out) = (!c.movable(Side::In), !c.movable(Side::Out));
            let segs = sentence_edges(db, c.video_id);
            let duration = video_duration(db, c.video_id).unwrap_or(f64::MAX);
            let (mut in_s, mut out_s) = snap_range(&segs, duration, c.in_s, c.out_s, None, cfg);
            if keep_in {
                in_s = c.in_s;
            }
            if keep_out {
                out_s = c.out_s;
            }
            if (in_s - c.in_s).abs() > 0.05 || (out_s - c.out_s).abs() > 0.05 {
                c.in_s = in_s;
                c.out_s = out_s;
                fixed += 1;
            }
        }

        // A bed is where most of the speech lives once one is laid, and it was never covered
        // here: three clips in a Bonsai 2 cut and one in agy's stopped mid-sentence because of
        // it. Its end may not pass the pictures it plays under, or `clamp_beds_to_beats` would
        // cut it back to exactly the place this pass exists to avoid.
        //
        // Only its end. Its start is where the voice meets the pictures — a laid bed starts as far
        // before the face's own in-point as the face sits into the beat — and moving it alone onto
        // the lead of its sentence put the voice out of step with the lips, made the bed outlast
        // the pictures, and had the next finish cut it back a sentence.
        if let Some(bed) = &mut beat.bed {
            let (keep_in, keep_out) = (true, !bed.movable(Side::Out));
            if !keep_out {
                let segs = sentence_edges(db, bed.video_id);
                let duration = video_duration(db, bed.video_id).unwrap_or(f64::MAX);
                let cap = if voice_len > 0.0 { Some(bed.in_s + voice_len) } else { None };
                let (mut in_s, mut out_s) = snap_range(&segs, duration, bed.in_s, bed.out_s, cap, cfg);
                if keep_in {
                    in_s = bed.in_s;
                }
                if keep_out {
                    out_s = bed.out_s;
                }
                if (in_s - bed.in_s).abs() > 0.05 || (out_s - bed.out_s).abs() > 0.05 {
                    bed.in_s = in_s;
                    bed.out_s = out_s;
                    fixed += 1;
                }
            }
        }
    }
    fixed
}

/// Where a speaking range whose last sentence is `last` (an index into `segs`) ends once it has
/// breathed: a hold of the person after the sentence (`speech_tail_s`), stopping short of the next
/// sentence — but when the speaker runs straight on there is no pause to stop in, and clipping at
/// the final sample cuts the last word's decay, so never before the overrun. [`pad_speech`] grows
/// a range to it, and the clip cap ([`super::draft::cap_clip_length`]) ends a range there, so a cut
/// the cap made is one padding leaves alone.
pub(crate) fn padded_end(segs: &[(f64, f64)], last: usize, cfg: &crate::config::ScriptConfig) -> f64 {
    let speech_end = segs[last].1;
    let held = speech_end + cfg.speech_tail_s;
    let out = match segs.get(last + 1) {
        Some(next) => held.min((next.0 - 0.3).max(speech_end + cfg.speech_overrun_s)),
        None => held,
    };
    out.max(speech_end + cfg.speech_overrun_s)
}

/// Let speaking clips breathe: finish the sentence the clip is in, then hold ~1.5 s of the person
/// before cutting (without running into their next sentence), and start slightly before the first
/// words. Runs after the snap to sentences, which lands cuts on sentence boundaries. An edge code
/// may not move ([`crate::script::ScriptClip::movable`]) stays where it was written.
pub fn pad_speech(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut changed = 0;
    for c in script.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        // A muted clip has no voice to protect. Padding one to finish a sentence nobody can hear
        // is not a repair, it is a two-second cutaway turning into twenty: b-roll taken from a
        // video that happens to contain speech was held until that speech finished, and a 40 s
        // cut came out at 77 s. `end_on_sentences` always skipped these; this did not.
        if c.audio != crate::script::Audio::Source || c.pinned() {
            continue;
        }
        // An exact clip or an edge written on a word is an instruction: those edges stay.
        let (move_in, move_out) = (c.movable(Side::In), c.movable(Side::Out));
        let segs = sentence_edges(db, c.video_id);
        let overlapping: Vec<usize> =
            (0..segs.len()).filter(|&i| segs[i].1 > c.in_s + 0.05 && segs[i].0 < c.out_s - 0.05).collect();
        let (Some(&first), Some(&last)) = (overlapping.first(), overlapping.last()) else { continue };
        let duration = video_duration(db, c.video_id).unwrap_or(f64::MAX);
        let (old_in, old_out) = (c.in_s, c.out_s);

        // The sentence to hold after is the one the clip has finished ([`out_owner`], the rule
        // every pass shares) — not one its own padding reached into. Without that distinction each
        // pass swallows another sentence: repairing an already-repaired script grew a cut from
        // 152.9 s to 178.7 s, and every `script import` of a saved script inflated it a little more.
        // A clip that stops inside a sentence is finishing that one, by the rule the sentence ends
        // use ([`snap_range`]): holding after the sentence before it instead left the finishing to
        // `end_on_sentences`, which adds no breath, and the next finish padded it again.
        let last = out_owner(&segs, c.out_s, cfg.speech_overrun_s).unwrap_or_else(|| {
            overlapping.iter().rev().find(|&&i| segs[i].0 < c.out_s - CUT_INTO_SENTENCE_S).copied().unwrap_or(last)
        });
        let out = padded_end(&segs, last, cfg).min(old_out + cfg.max_speech_extend_s).min(duration);
        if move_out && out > c.out_s {
            c.out_s = out;
        }

        // The sentence it opens on: the one whose lead-in it starts in, when it does ([`in_owner`]),
        // not the one before whose last words that lead overlaps.
        let first = in_owner(&segs, c.in_s, cfg.speech_lead_s).unwrap_or(first);
        let speech_start = segs[first].0;
        let mut lead = (speech_start - cfg.speech_lead_s).max(0.0);
        if first > 0 {
            lead = lead.max(segs[first - 1].1);
        }
        // Also pulls a clip that starts mid-sentence back to the start of that sentence.
        if move_in && c.in_s > lead {
            c.in_s = lead.max(old_in - cfg.max_speech_extend_s);
        }
        if (c.in_s, c.out_s) != (old_in, old_out) {
            changed += 1;
        }
    }
    changed
}
