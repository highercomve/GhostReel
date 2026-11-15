//! The rules a model editor is told and can miss, held by code for every cut a model wrote — a chat
//! turn's and an MCP `save_script`'s alike ([`super::Profile::draft`]): footage from this project
//! only, a steady camera, no clip longer than `max_clip_s`, and back-to-back cuts of one take
//! joined. The two rules about who is heard — no room sound under a voice, no narration over a
//! speaking beat — live here too and hold for every cut.
//!
//! A rule that limits where a clip may be or how long it may run speaks in whole sentences, the
//! way the passes after it do: one that cut in the middle of a sentence was undone by the sentence
//! passes, and the next finish fired it again (a cut that does not settle, and a quote that lost a
//! sentence on a later save).
//!
//! They used to run only in the chat's review (`chat::enforce_grounding_and_pacing`), so an editor
//! that saved through the tools got none of them, and a chat turn and a tool save of the same draft
//! came out different.

use crate::config::ScriptConfig;
use crate::db::Db;
use crate::script::{Audio, Beat, Script, ScriptClip, Side};

use super::clip_has_speech;
use super::edges::{EDGE_SLACK_S, lead_into, sentence_edges};
use super::speech::padded_end;

/// Clips of videos outside the project go: a sentence id or a video id from another project names
/// footage this cut cannot play. Returns a note for each.
pub fn drop_foreign_clips(db: &Db, project_id: i64, s: &mut Script) -> Vec<String> {
    let mut notes = Vec::new();
    for beat in &mut s.beats {
        beat.clips.retain(|c| {
            let ours = db.is_video_in_project(project_id, c.video_id);
            if !ours {
                notes.push(format!("dropped clip from video #{} — not in this project", c.video_id));
            }
            ours
        });
        if beat.bed.as_ref().is_some_and(|bed| !db.is_video_in_project(project_id, bed.video_id)) {
            let bed = beat.bed.take().expect("checked");
            notes.push(format!("dropped the audio bed from video #{} — not in this project", bed.video_id));
        }
    }
    s.beats.retain(|b| !b.clips.is_empty());
    notes
}

/// The steady stretches of `in_s..out_s`, longest first: what is left between the shaky spans.
fn steady_parts(in_s: f64, out_s: f64, shaky: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut parts = Vec::new();
    let mut at = in_s;
    for &(a, b) in shaky {
        if a > at {
            parts.push((at, a.min(out_s)));
        }
        at = at.max(b);
    }
    if out_s > at {
        parts.push((at, out_s));
    }
    parts.sort_by(|x, y| (y.1 - y.0).total_cmp(&(x.1 - x.0)));
    parts
}

/// A clip on a stretch the camera shakes through (more than half of it) is taken off it. Telling
/// the model was not enough: the same stretch of a clip measured at eleven times the sway limit was
/// cut into four scripts.
///
/// It judges a clip as it will play, so it runs once the passes that grow and shorten clips are
/// done (sentence ends, the cap, pictures cut back to a bed): run first, it passed a clip that
/// padding then grew into the shake, or one that a bed made a picture and trimmed, and the next
/// finish moved what this one had kept.
///
/// A clip that carries someone's voice keeps the longest run of whole sentences inside the steady
/// parts ([`steady_sentences`]): sliding it to a steady stretch elsewhere, keeping its length,
/// played other words — the same slide that was removed for clips opening off the microphone — and
/// trimming it to raw steady seconds left it mid-sentence, for the sentence passes after this to
/// grow straight back over the shake. A picture nobody speaks in has no words to keep, so it moves
/// to the nearest steady stretch of the same shot. What has no steady part long enough goes.
///
/// A clip with an edge code may not move ([`ScriptClip::movable`]) is kept as written, and the note
/// says so: the editor chose that footage by hand. Trimming the other side of it by the second was
/// not the same rule for a speaking clip (whole sentences) and for the same clip once a bed made it
/// a muted picture, so a second finish trimmed what the first had kept.
pub fn steady_camera(db: &Db, s: &mut Script, cfg: &ScriptConfig) -> Vec<String> {
    let mut notes = Vec::new();
    if cfg.max_shake_jerk <= 0.0 {
        return notes;
    }
    for beat in &mut s.beats {
        let mut kept: Vec<ScriptClip> = Vec::with_capacity(beat.clips.len());
        for mut c in beat.clips.drain(..) {
            let windows = db.motion_windows(c.video_id).unwrap_or_default();
            let len = c.out_s - c.in_s;
            if windows.is_empty() || len <= 0.0 {
                kept.push(c);
                continue;
            }
            let limit = crate::steadiness::shake_limit(&windows, cfg.max_shake_jerk, cfg.shake_relative);
            let spans = crate::steadiness::shaky_spans_with_sway(&windows, c.in_s, c.out_s, limit, cfg.max_sway);
            let shaky_s: f64 = spans.iter().map(|(a, b)| (b.min(c.out_s) - a.max(c.in_s)).max(0.0)).sum();
            if shaky_s <= len * 0.5 {
                kept.push(c);
                continue;
            }
            let (from, to) = (c.in_s, c.out_s);
            if !c.movable(Side::In) || !c.movable(Side::Out) {
                notes.push(format!(
                    "kept a clip as written although the camera shakes through video #{} {from:.1}–{to:.1} s: \
                     its edges were set by hand",
                    c.video_id
                ));
                kept.push(c);
                continue;
            }
            let picture = c.audio != Audio::Source;
            let segs = if picture { Vec::new() } else { sentence_edges(db, c.video_id) };
            let moved = if picture {
                crate::steadiness::steady_stretch_near(&windows, c.in_s, len, limit, cfg.max_sway).map(|a| (a, a + len))
            } else if let Some(run) = steady_sentences(&segs, &c, cfg, |a, b| {
                crate::steadiness::shaky_spans_with_sway(&windows, a, b, limit, cfg.max_sway)
            }) {
                run
            } else {
                // No sentence in it to keep whole: what is steady, by the second.
                steady_parts(c.in_s, c.out_s, &spans).into_iter().find(|&(a, b)| b - a >= cfg.min_clip_s.min(len))
            };
            match moved {
                Some((a, b)) => {
                    notes.push(format!(
                        "{} clip off a shaky stretch: video #{} {from:.1}–{to:.1} s → {a:.1}–{b:.1} s",
                        if picture { "moved" } else { "trimmed" },
                        c.video_id
                    ));
                    (c.in_s, c.out_s) = (a, b);
                    kept.push(c);
                }
                None => notes.push(format!(
                    "dropped clip: the camera shakes through video #{} {from:.1}–{to:.1} s and nothing steady of that \
                     length exists in it",
                    c.video_id
                )),
            }
        }
        beat.clips = kept;
    }
    s.beats.retain(|b| !b.clips.is_empty());
    notes
}

/// The longest run of whole sentences of a speaking clip that the camera holds steady through, as
/// the clip it makes (`in_s`, `out_s`): opening on the lead before its first sentence and ending
/// where padding ends its last ([`padded_end`], at most `max_clip_s` long) — the edges the speaking
/// passes give a clip, so the next finish has nothing to grow back over the shake. `shaky(a, b)`
/// lists the shaky spans of `a..b`.
///
/// The sentences considered are the ones the clip plays (whole or in part, since the sentence
/// passes would finish a part); a run must lie inside one steady part and be at least
/// `min_clip_s` long. A start code may not move stays where it was written, and then the run must
/// open on the clip's first sentence and the start lie in the steady part; the same for an end.
///
/// `None` when the clip plays no sentence (it is steadied by the second instead); `Some(None)` when
/// no run of whole sentences is steady and long enough.
fn steady_sentences(
    segs: &[(f64, f64)],
    c: &ScriptClip,
    cfg: &ScriptConfig,
    shaky: impl Fn(f64, f64) -> Vec<(f64, f64)>,
) -> Option<Option<(f64, f64)>> {
    let played: Vec<usize> =
        (0..segs.len()).filter(|&i| segs[i].1 > c.in_s + EDGE_SLACK_S && segs[i].0 < c.out_s - EDGE_SLACK_S).collect();
    let (&first, &last) = (played.first()?, played.last()?);
    let (keep_in, keep_out) = (!c.movable(Side::In), !c.movable(Side::Out));
    let lo = if keep_in { c.in_s } else { c.in_s.min(lead_into(segs, first, cfg)) };
    let hi = if keep_out { c.out_s } else { c.out_s.max(segs[last].1 + cfg.speech_overrun_s) };
    let parts = steady_parts(lo, hi, &shaky(lo, hi));
    let mut best: Option<(f64, f64)> = None;
    for (n, &i) in played.iter().enumerate() {
        if keep_in && i != first {
            break;
        }
        for &j in &played[n..] {
            if keep_out && j != last {
                continue;
            }
            // What must be steady: the words, and an edge kept where it was written.
            let (from, to) = (if keep_in { c.in_s } else { segs[i].0 }, if keep_out { c.out_s } else { segs[j].1 });
            if !parts.iter().any(|&(a, b)| from >= a - EDGE_SLACK_S && to <= b + EDGE_SLACK_S) {
                continue;
            }
            let open = if keep_in { c.in_s } else { lead_into(segs, i, cfg) };
            let close = if keep_out {
                c.out_s
            } else {
                padded_end(segs, j, cfg).min(open + cfg.max_clip_s).max(segs[j].1 + cfg.speech_overrun_s)
            };
            let run = (open, close);
            if run.1 - run.0 >= cfg.min_clip_s && best.is_none_or(|b| run.1 - run.0 > b.1 - b.0) {
                best = Some(run);
            }
        }
    }
    Some(best)
}

/// No clip runs past `max_clip_s`. A speaking clip longer than that ends on the last sentence that
/// fits with its tail — the overrun after it, and the breath padding holds when `pad` (what
/// [`padded_end`] gives, cut at the cap) — so the clip is cut once, where it would have stopped
/// anyway, and the next finish leaves it there. Cutting it to `trimmed_clip_s` in the middle of a
/// sentence had the sentence passes put a sentence back, past the cap, for the next finish to cut
/// again a whole sentence shorter. A clip whose first sentence alone runs past the cap keeps it
/// whole (a sentence is not cut to fit a number), and a picture is cut to `trimmed_clip_s`. An end
/// code may not move ([`ScriptClip::movable`]) stays.
///
/// It runs after every pass that grows a clip (padding, sentence ends), as the one length limit,
/// so nothing after it grows a clip back. Returns a note for each clip it changed or kept long.
pub fn cap_clip_length(db: &Db, s: &mut Script, cfg: &ScriptConfig, pad: bool) -> Vec<String> {
    let mut notes = Vec::new();
    for c in s.beats.iter_mut().flat_map(|b| b.clips.iter_mut()) {
        let len = c.out_s - c.in_s;
        if len <= cfg.max_clip_s + 1e-9 || !c.movable(Side::Out) {
            continue;
        }
        let segs = if c.audio == Audio::Source { sentence_edges(db, c.video_id) } else { Vec::new() };
        let speaks = segs.iter().any(|&(a, e)| e > c.in_s + EDGE_SLACK_S && a < c.out_s - EDGE_SLACK_S);
        if !speaks {
            notes.push(format!(
                "trimmed {len:.1} s clip to {:.0} s (video #{} from {:.1} s)",
                cfg.trimmed_clip_s, c.video_id, c.in_s
            ));
            c.out_s = c.in_s + cfg.trimmed_clip_s;
            continue;
        }
        let limit = c.in_s + cfg.max_clip_s;
        let fits = (0..segs.len()).rev().find(|&i| {
            let e = segs[i].1;
            e <= c.out_s + EDGE_SLACK_S
                && e + cfg.speech_overrun_s <= limit + 1e-9
                && e - c.in_s >= cfg.min_trimmed_clip_s
        });
        match fits {
            Some(i) => {
                let end = if pad { padded_end(&segs, i, cfg) } else { segs[i].1 + cfg.speech_overrun_s };
                let end = end.min(limit);
                if (end - c.out_s).abs() > 1e-9 {
                    notes.push(format!(
                        "ended a {len:.1} s clip on a whole sentence at {:.1} s ({:.1} s), under the {:.0} s limit \
                         (video #{} from {:.1} s)",
                        end,
                        end - c.in_s,
                        cfg.max_clip_s,
                        c.video_id,
                        c.in_s
                    ));
                    c.out_s = end;
                }
            }
            None => {
                // The first sentence whole, ending where padding ends it, so the next finish's
                // padding finds nothing to add.
                if let Some(i) = segs.iter().position(|&(_, e)| e > c.in_s + EDGE_SLACK_S)
                    && segs[i].1 <= c.out_s + EDGE_SLACK_S
                {
                    let end = if pad { padded_end(&segs, i, cfg) } else { segs[i].1 + cfg.speech_overrun_s };
                    c.out_s = end;
                }
                notes.push(format!(
                    "kept a {len:.1} s clip over the {:.0} s limit: its first sentence alone runs past it (video #{} \
                     from {:.1} s)",
                    cfg.max_clip_s, c.video_id, c.in_s
                ));
            }
        }
    }
    notes
}

/// Whether somebody is heard over `beat`: narration, a bed with speech in it, or a clip that plays
/// somebody speaking.
fn voice_over(db: &Db, beat: &Beat) -> bool {
    beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty()) || beat_speaks(db, beat)
}

/// Whether `beat` plays somebody speaking: a clip that plays its own sound with speech in range,
/// or a bed with speech in it — the voice is the voice whether it lives on a clip or on a bed.
fn beat_speaks(db: &Db, beat: &Beat) -> bool {
    beat.clips.iter().any(|c| {
        c.audio == Audio::Source
            && c.tracks.as_ref().is_none_or(|t| !t.is_empty())
            && clip_has_speech(db, c.video_id, c.in_s, c.out_s)
    }) || beat.bed.as_ref().is_some_and(|b| {
        b.tracks.as_ref().is_none_or(|t| !t.is_empty()) && clip_has_speech(db, b.video_id, b.in_s, b.out_s)
    })
}

/// A clip with nobody transcribed in its range plays no sound of its own when something else is
/// heard over its beat — narration, a bed, somebody speaking in another clip: wind, handling noise
/// and the room do not belong under a voice. The one rule for it (a montage decides its sound
/// itself).
///
/// Its scope is what both rules it replaced meant. The chat's muted every speechless `source` clip,
/// because one left marked `source` read as "someone speaks here" and took a narrated beat's
/// narration away — which the narration rule now decides by speech ([`hush_narration_over_speech`]);
/// the tools' and import's muted one only under narration. A speechless clip alone in its beat is
/// the beat's sound — applause, a crowd, the place — and an editor who marked it `source` meant
/// it; an `exact` clip, or one written on words at both ends, is an instruction and keeps its sound
/// everywhere. Returns how many clips were muted.
pub fn mute_clips_without_speech(db: &Db, s: &mut Script) -> usize {
    let mut muted = 0;
    for beat in &mut s.beats {
        if !voice_over(db, beat) {
            continue;
        }
        for c in beat.clips.iter_mut() {
            if c.audio == Audio::Source && !c.pinned() && !clip_has_speech(db, c.video_id, c.in_s, c.out_s) {
                c.audio = Audio::Mute;
                muted += 1;
            }
        }
    }
    muted
}

/// A beat that plays somebody speaking — on a clip or on a bed under its pictures — has no
/// narration on top of them. Runs once beds are laid and checked, so a bed with no speech in it
/// has gone and one laid under the pictures counts. Returns how many beats lost theirs.
pub fn hush_narration_over_speech(db: &Db, s: &mut Script) -> usize {
    let mut hushed = 0;
    for beat in &mut s.beats {
        let narrated = beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty());
        if narrated && beat_speaks(db, beat) {
            beat.narration = Some(String::new());
            hushed += 1;
        }
    }
    hushed
}

/// Whether `prev` and `next` are one shot cut in two: the same take, back to back, played the same
/// way. Clips that differ in how they are framed, how loud, which tracks or whether they are heard
/// are two choices, not one clip split while looking, and joining them kept only the first's.
fn same_shot(prev: &ScriptClip, next: &ScriptClip) -> bool {
    next.why.as_deref() != Some(crate::script::CLOSING_PICTURE)
        && prev.video_id == next.video_id
        && prev.audio == next.audio
        && prev.framing == next.framing
        && prev.gain_db == next.gain_db
        && prev.tracks == next.tracks
        && prev.movable(Side::Out)
        && next.movable(Side::In)
        && !prev.exact
        && !next.exact
}

/// The range of `prev` and `next` joined, when one range can say what both did at the outer edges:
/// the first's start and the second's end. `Err` when it cannot — one was written on a word at its
/// outer edge and the other has no range to end or start the joined one, so joining would forget
/// that edge was pinned and let code move it.
fn joined_range(
    prev: Option<&crate::script::SentenceRange>,
    next: Option<&crate::script::SentenceRange>,
) -> Result<Option<crate::script::SentenceRange>, ()> {
    match (prev, next) {
        (Some(a), Some(b)) => Ok(Some(crate::script::SentenceRange { from: a.from.clone(), to: b.to.clone() })),
        (None, None) => Ok(None),
        (Some(a), None) if a.starts_on_word() => Err(()),
        (None, Some(b)) if b.ends_on_word() => Err(()),
        // Ids alone, with nothing pinned: the range goes, and is named again from what plays.
        _ => Ok(None),
    }
}

/// Back-to-back clips of the same shot in one beat (`0-6`, `6-16`, `16-23`, played the same way) are
/// jump cuts inside a single continuous take: join them. A range the two were written with is kept
/// as the first's start and the second's end; two that one range cannot join without losing an
/// edge written on a word stay apart ([`joined_range`]). Returns how many cuts were removed.
///
/// Back to back means one ends where the next starts (within [`EDGE_SLACK_S`]). Half a second
/// apart counted too, and two touching cuts too long to join, once the pictures were trimmed to
/// their bed, were 0.4 s apart and short enough: the next finish joined what this one had not.
pub fn merge_contiguous_clips(script: &mut Script, cfg: &ScriptConfig) -> usize {
    let mut merged = 0;
    for beat in &mut script.beats {
        let mut out: Vec<ScriptClip> = Vec::with_capacity(beat.clips.len());
        for c in beat.clips.drain(..) {
            if let Some(prev) = out.last_mut()
                && same_shot(prev, &c)
                && (c.in_s - prev.out_s).abs() <= EDGE_SLACK_S
                && c.out_s - prev.in_s <= cfg.max_clip_s
                && let Ok(range) = joined_range(prev.sentences.as_ref(), c.sentences.as_ref())
            {
                prev.out_s = prev.out_s.max(c.out_s);
                prev.sentences = range;
                merged += 1;
                continue;
            }
            out.push(c);
        }
        beat.clips = out;
    }
    merged
}
