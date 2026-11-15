//! Sentence edges: where a clip's edges belong among the sentences of its video, decided by one
//! rule every pass shares.
//!
//! A speaking range ends a little after its last sentence (`speech_overrun_s`, so the last word
//! decays instead of stopping). When the speaker runs straight on, that tail lies inside the next
//! sentence — and a pass that read "inside the next sentence" from it finished that sentence too,
//! added its own tail, and the next repair did the same again: a cut swallowed one short "Yeah."
//! per repair until the fit dropped a whole quote. [`out_owner`] is the one answer to "which
//! sentence has this range finished", and snapping, padding, sentence ends and cut placement all
//! ask it.

use rusqlite::OptionalExtension;

use crate::Error;
use crate::db::Db;
use crate::script::{Audio, Script, Side};
use crate::sentence_split::CUT_INTO_SENTENCE_S;

/// How far either side of a boundary an edge still counts as on it (s): the passes round, and a
/// cut placed on the sound lands a frame or so off the transcript's time.
pub const EDGE_SLACK_S: f64 = 0.05;

/// The sentence a range ending at `out_s` has finished, as an index into `edges` (start, end pairs
/// in order), when it ends on a sentence's end or in the tail after it.
///
/// An end within `tail + EDGE_SLACK_S` after a sentence's end belongs to that sentence: it is the
/// sentence and its overrun, even when the next sentence has already started there. An end on a
/// boundary (within the slack either side) belongs to the sentence ending there. Of two sentences
/// whose tails both reach `out_s` (one shorter than a tail), the earlier owns it — the later one
/// starts inside the tail, which is what a tail into running speech looks like. `None` when the
/// range stops inside a sentence or in a pause past the tail.
pub fn out_owner(edges: &[(f64, f64)], out_s: f64, tail: f64) -> Option<usize> {
    if let Some(i) = edges.iter().position(|&(_, end)| (out_s - end).abs() <= EDGE_SLACK_S) {
        return Some(i);
    }
    edges.iter().position(|&(_, end)| end < out_s && out_s <= end + tail + EDGE_SLACK_S)
}

/// The sentence a range starting at `in_s` opens on, as an index into `edges`, when it starts on a
/// sentence's start or in the lead before it — the mirror of [`out_owner`]. A speaking range opens
/// a little before its first word (`speech_lead_s`), and a pass that read that lead as "the end of
/// the sentence before" snapped it away, for the next pass to put back.
pub fn in_owner(edges: &[(f64, f64)], in_s: f64, lead: f64) -> Option<usize> {
    if let Some(i) = edges.iter().position(|&(start, _)| (in_s - start).abs() <= EDGE_SLACK_S) {
        return Some(i);
    }
    edges.iter().position(|&(start, _)| start > in_s && start - in_s <= lead + EDGE_SLACK_S)
}

/// The end of the last sentence a range ending at `out_s` really contains: its owner
/// ([`out_owner`]) when it ends on one, else the last of `candidates` (indices into `edges`, in
/// order) that ends by `out_s`.
pub(crate) fn finished_end(edges: &[(f64, f64)], candidates: &[usize], out_s: f64, tail: f64) -> Option<(usize, f64)> {
    if let Some(i) = out_owner(edges, out_s, tail) {
        return Some((i, edges[i].1));
    }
    candidates.iter().rev().map(|&i| (i, edges[i].1)).find(|&(_, e)| e <= out_s + EDGE_SLACK_S)
}

/// Every sentence boundary in a video, as (start, end) pairs in order.
pub fn sentence_edges(db: &Db, video_id: i64) -> Vec<(f64, f64)> {
    crate::sentence_split::edges(db, video_id)
}

/// Pull one range onto whole sentences.
///
/// `cap_out` is the furthest the end may travel — a bed may not outlast the pictures above it.
/// When finishing the sentence would pass it, the range falls back to where an earlier sentence
/// ended rather than stopping in the middle of this one.
pub(crate) fn snap_range(
    segs: &[(f64, f64)],
    duration: f64,
    in_s: f64,
    out_s: f64,
    cap_out: Option<f64>,
    cfg: &crate::config::ScriptConfig,
) -> (f64, f64) {
    let overlapping: Vec<usize> =
        (0..segs.len()).filter(|&i| segs[i].1 > in_s + 0.05 && segs[i].0 < out_s - 0.05).collect();
    let (Some(&first), Some(&_last)) = (overlapping.first(), overlapping.last()) else { return (in_s, out_s) };
    let (mut in_s, mut out_s) = (in_s, out_s);
    let ceiling = cap_out.unwrap_or(f64::MAX).min(duration);

    // The end. The sentence to finish is one the range is genuinely inside — a range that ends on
    // a boundary still reaches a little past it on purpose, and treating that as "inside the next
    // sentence" makes every pass swallow one more: a cut grew 126 s to 154 s being repaired twice.
    // A range that has finished a sentence ([`out_owner`]) is not inside the one after it, however
    // early that one started.
    let owner = out_owner(segs, out_s, cfg.speech_overrun_s);
    let cut_into = overlapping
        .iter()
        .rev()
        .find(|&&i| segs[i].0 < out_s - CUT_INTO_SENTENCE_S && owner.is_none_or(|o| i <= o))
        .copied();
    let sentence_end = cut_into.map(|i| segs[i].1).unwrap_or(out_s);

    // A range that stops exactly on a boundary still clips the last word. Whisper's times are
    // approximate and often quantised — Adrienne's segments land on whole seconds — so "the
    // sentence ends at 79.00" means the speech ends somewhere near it, and cutting at 79.00 takes
    // the tail of "running around too" with it. Give every speaking range the overrun; the render
    // fades across it, so the word decays instead of stopping.
    if let Some((_, last_end)) = finished_end(segs, &overlapping, out_s, cfg.speech_overrun_s)
        && out_s < last_end + cfg.speech_overrun_s - 0.02
    {
        out_s = (last_end + cfg.speech_overrun_s).min(ceiling).min(duration);
    }

    // Where the range will open once its start is on a sentence (below): a fallback for the end is
    // measured from there. Measured from the start as written, a range that opens a lead earlier
    // found no earlier sentence long enough, kept the whole one past its ceiling, and the next
    // finish — starting from the lead — fell back after all.
    let lead_into = |i: usize| lead_into(segs, i, cfg);
    let sentence_start = segs[first].0;
    let opens_mid = in_owner(segs, in_s, cfg.speech_lead_s).is_none() && sentence_start < in_s - 0.05;
    let opens_at = if opens_mid && in_s - sentence_start <= cfg.max_speech_extend_s { lead_into(first) } else { in_s };
    if sentence_end > out_s + 0.05 {
        let wanted = (sentence_end + cfg.speech_overrun_s).min(duration);
        if wanted - out_s <= cfg.max_speech_extend_s && wanted <= ceiling {
            out_s = wanted;
        } else {
            let back = overlapping
                .iter()
                .rev()
                .map(|&i| segs[i].1)
                .find(|&e| e <= out_s + 0.05 && e - opens_at >= cfg.min_trimmed_clip_s);
            match back {
                Some(end) => out_s = (end + cfg.speech_overrun_s).min(ceiling),
                // Nothing to fall back to: a whole sentence long is better than half of one.
                // Keep the whole sentence even past the ceiling.
                None => out_s = wanted,
            }
        }
    }

    // The start. Never open in the middle of a sentence. A start in the lead before a sentence
    // ([`in_owner`]) opens on that sentence, not inside the one whose last words the lead overlaps.
    // The lead itself stops at the end of the sentence before, as `pad_speech`'s does: two
    // sentences in one breath have no pause to lead in from.
    if opens_mid {
        if in_s - sentence_start <= cfg.max_speech_extend_s {
            in_s = opens_at;
        } else if let Some(&next) = overlapping.iter().find(|&&i| segs[i].0 >= in_s)
            && out_s - segs[next].0 >= cfg.min_trimmed_clip_s
        {
            in_s = lead_into(next);
        }
    }
    (in_s, out_s)
}

/// Where a speaking range opening on sentence `i` of `segs` starts: a lead before its first word
/// (`speech_lead_s`), never before the end of the sentence before — two sentences in one breath
/// have no pause to lead in from.
pub(crate) fn lead_into(segs: &[(f64, f64)], i: usize, cfg: &crate::config::ScriptConfig) -> f64 {
    let prev_end = i.checked_sub(1).map_or(0.0, |p| segs[p].1);
    (segs[i].0 - cfg.speech_lead_s).max(prev_end.min(segs[i].0)).max(0.0)
}

/// How far a clip edge may travel to land on a sentence edge (s).
const SNAP_REACH_S: f64 = 0.75;

/// Snap each speaking clip's `in_s` to the nearest sentence START and its `out_s` to the nearest
/// sentence END ([`crate::sentence_split`]), within 0.75 s. Returns the count of clips changed.
///
/// An edge only ever snaps to the kind of boundary it is: a clip opens where somebody starts
/// speaking and closes where somebody stops. Snapping to the nearest boundary of either kind sent
/// an out point to the NEXT speaker's first word (12.86 s, a handover from 12.02 s) and an in
/// point to the end of the PREVIOUS speaker's line, so the cut opened on, or closed with, the
/// wrong voice. Muted clips are left alone: they carry no voice, and the speech of the video a
/// cutaway was taken from is not where it should start or end.
///
/// An end that has already finished a sentence ([`out_owner`]: on its end or in the tail after it)
/// stays where it is, and so does a start on a sentence or in the lead before it ([`in_owner`]). The nearest end to such a tail is often the next sentence's — a short "Yeah."
/// touching the last one — and snapping there finished a sentence nobody chose, once per repair.
pub fn snap_to_sentences(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> Result<usize, Error> {
    let mut clips_changed = 0;

    for beat in &mut script.beats {
        for clip in &mut beat.clips {
            // A muted picture (a cutaway, the quiet image a cut closes on) has no speech to land
            // on; the speech of the video it was taken from is not its edges.
            if clip.audio == Audio::Mute || clip.pinned() {
                continue;
            }
            let (keep_in, keep_out) = (!clip.movable(Side::In), !clip.movable(Side::Out));
            let duration_s: Option<f64> = db
                .conn
                .query_row("SELECT duration_s FROM videos WHERE id = ?1", [clip.video_id], |r| r.get(0))
                .optional()?
                .flatten();
            let max_dur = duration_s.unwrap_or(f64::MAX);

            let edges = sentence_edges(db, clip.video_id);
            if edges.is_empty() {
                continue;
            }
            let starts: Vec<f64> = edges.iter().map(|e| e.0).collect();
            let ends: Vec<f64> = edges.iter().map(|e| e.1).collect();

            let best_in = if keep_in || in_owner(&edges, clip.in_s, cfg.speech_lead_s).is_some() {
                clip.in_s
            } else {
                nearest_within(&starts, clip.in_s, SNAP_REACH_S).map_or(clip.in_s, |b| b.clamp(0.0, max_dur))
            };
            let best_out = if keep_out || out_owner(&edges, clip.out_s, cfg.speech_overrun_s).is_some() {
                clip.out_s
            } else {
                nearest_within(&ends, clip.out_s, SNAP_REACH_S).map_or(clip.out_s, |b| b.clamp(0.0, max_dur))
            };

            // Never produce in >= out; skip snap if it would
            if best_in >= best_out {
                continue;
            }

            let changed = (best_in - clip.in_s).abs() > 1e-4 || (best_out - clip.out_s).abs() > 1e-4;
            if changed {
                clip.in_s = best_in;
                clip.out_s = best_out;
                clips_changed += 1;
            }
        }
    }

    Ok(clips_changed)
}

/// The edge closest to `t` within `reach` (the later one on a tie), if any.
fn nearest_within(edges: &[f64], t: f64, reach: f64) -> Option<f64> {
    let mut best = None;
    let mut best_diff = reach;
    for &b in edges {
        let diff = (b - t).abs();
        if diff <= best_diff {
            best_diff = diff;
            best = Some(b);
        }
    }
    best
}

/// What [`trim_unspoken_edges`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unspoken {
    /// Edges moved.
    pub moved: usize,
    /// Clips dropped because everything said in them is off the microphone.
    pub dropped: usize,
}

/// The edges code always takes off what a clip or a bed says, by the one span rule cut placement
/// uses too ([`crate::cuts::spoken_span`]): someone off the microphone at either end, and at a
/// clip's start a "yeah so" said on its own breath; an edge written on a word stays where it was
/// asked for. A pass of its own, before the passes that pad and fit the cut, so they work on the
/// clip that will play; `place_cuts` then finds nothing left to take off.
///
/// A speaking clip in which everything said is off the microphone (the interviewer's question on
/// its own) goes: there is no answer in it to trim to. It used to be slid forward to the next line
/// on the microphone keeping its length, which played whatever came after — often the next question.
pub fn trim_unspoken_edges(db: &Db, script: &mut Script) -> Unspoken {
    let mut done = Unspoken::default();
    for beat in &mut script.beats {
        // A beat with a bed plays its clips silent under it: the bed's edges are the voice's.
        if beat.bed.is_none() {
            let before = beat.clips.len();
            beat.clips.retain_mut(|c| {
                if c.audio != Audio::Source || c.pinned() {
                    return true;
                }
                let span = crate::cuts::spoken_span(db, c.video_id, c.in_s, c.out_s, c.sentences.as_ref(), true);
                if span.off_mic_only && c.movable(Side::In) && c.movable(Side::Out) {
                    return false;
                }
                done.moved += usize::from(span.in_s != c.in_s) + usize::from(span.out_s != c.out_s);
                (c.in_s, c.out_s) = (span.in_s, span.out_s);
                true
            });
            done.dropped += before - beat.clips.len();
        }
        if let Some(bed) = beat.bed.as_mut() {
            let span = crate::cuts::spoken_span(db, bed.video_id, bed.in_s, bed.out_s, bed.sentences.as_ref(), false);
            done.moved += usize::from(span.in_s != bed.in_s) + usize::from(span.out_s != bed.out_s);
            (bed.in_s, bed.out_s) = (span.in_s, span.out_s);
        }
    }
    script.beats.retain(|b| !b.clips.is_empty());
    done
}

/// How far from a sentence's boundary an edge may sit and still be that sentence's edge, for
/// naming it (s): cut placement settles an edge on the measured sound, up to this far from the
/// transcript's time.
const NAME_SLACK_S: f64 = 0.8;

/// The sentences of a video as (id, start, end), in order.
fn sentence_rows(db: &Db, video_id: i64) -> Vec<(i64, f64, f64)> {
    let _ = crate::sentence_split::ensure(db, video_id);
    let Ok(mut st) = db
        .conn
        .prepare_cached("SELECT id, start_s, end_s FROM transcript_sentences WHERE video_id = ?1 ORDER BY start_s")
    else {
        return Vec::new();
    };
    st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
}

/// The sentence a range starting at `in_s` opens on (an index into `edges`): on its start, in the
/// lead before it, or settled on the sound near it. `None` when it starts inside a sentence.
fn opens_on(edges: &[(f64, f64)], in_s: f64, cfg: &crate::config::ScriptConfig) -> Option<usize> {
    if let Some(i) = in_owner(edges, in_s, cfg.speech_lead_s) {
        return Some(i);
    }
    let k = edges.iter().position(|&(_, end)| end > in_s + NAME_SLACK_S)?;
    let start = edges[k].0;
    (in_s <= start + NAME_SLACK_S && start - in_s <= cfg.speech_lead_s + NAME_SLACK_S).then_some(k)
}

/// The sentence a range ending at `out_s` has finished (an index into `edges`): on its end, in the
/// tail or the held breath after it, or settled on the sound near it. `None` when it stops inside a
/// sentence.
fn closes_on(edges: &[(f64, f64)], out_s: f64, cfg: &crate::config::ScriptConfig) -> Option<usize> {
    if let Some(i) = out_owner(edges, out_s, cfg.speech_overrun_s) {
        return Some(i);
    }
    let k = edges.iter().rposition(|&(start, _)| start < out_s - NAME_SLACK_S)?;
    let end = edges[k].1;
    (out_s >= end - NAME_SLACK_S && out_s <= end + cfg.speech_tail_s.max(cfg.speech_overrun_s) + NAME_SLACK_S)
        .then_some(k)
}

/// The range `range` names for `in_s..out_s` of `video_id` once a pass has moved its edges: a side
/// written on a sentence now names the sentence the edge belongs to, a side written on a word stays
/// (code never moves one off its word). `None` when a moved side lies inside a sentence and the
/// other is not written on a word: no sentence describes it any more, and a stale id would.
fn named_range(
    rows: &[(i64, f64, f64)],
    range: &crate::script::SentenceRange,
    in_s: f64,
    out_s: f64,
    cfg: &crate::config::ScriptConfig,
) -> Option<crate::script::SentenceRange> {
    let edges: Vec<(f64, f64)> = rows.iter().map(|&(_, a, b)| (a, b)).collect();
    let (word_in, word_out) = (range.starts_on_word(), range.ends_on_word());
    let from = if word_in {
        Some(range.from.clone())
    } else {
        opens_on(&edges, in_s, cfg).map(|i| crate::sentences::format_id(rows[i].0))
    };
    let to = if word_out {
        Some(range.to.clone())
    } else {
        closes_on(&edges, out_s, cfg).map(|i| crate::sentences::format_id(rows[i].0))
    };
    match (from, to) {
        (Some(from), Some(to)) => Some(crate::script::SentenceRange { from, to }),
        // A word edge keeps its range: it is what keeps code off that edge. The other side's id is
        // only a label then.
        (from, to) if word_in || word_out => Some(crate::script::SentenceRange {
            from: from.unwrap_or_else(|| range.from.clone()),
            to: to.unwrap_or_else(|| range.to.clone()),
        }),
        _ => None,
    }
}

/// Keep the sentence ids a clip or a bed was written with describing what it plays. The passes
/// move edges by seconds — the interviewer trimmed off the start, a quote ended where its speaker
/// stopped — and a range still naming the sentence trimmed away told `read_cut` and the next cut
/// placement about words that no longer play. Returns how many ranges changed.
pub fn name_ranges(db: &Db, script: &mut Script, cfg: &crate::config::ScriptConfig) -> usize {
    let mut rows_of: std::collections::HashMap<i64, Vec<(i64, f64, f64)>> = std::collections::HashMap::new();
    let mut changed = 0;
    for beat in &mut script.beats {
        let items = beat
            .clips
            .iter_mut()
            // An exact clip is kept as written, its range with it.
            .filter(|c| !c.exact)
            .map(|c| (c.video_id, c.in_s, c.out_s, &mut c.sentences))
            .chain(beat.bed.as_mut().map(|b| (b.video_id, b.in_s, b.out_s, &mut b.sentences)));
        for (video_id, in_s, out_s, sentences) in items {
            let Some(range) = sentences.as_ref() else { continue };
            if range.starts_on_word() && range.ends_on_word() {
                continue;
            }
            let rows = rows_of.entry(video_id).or_insert_with(|| sentence_rows(db, video_id));
            if rows.is_empty() {
                continue;
            }
            let named = named_range(rows, range, in_s, out_s, cfg);
            if named.as_ref() != Some(range) {
                *sentences = named;
                changed += 1;
            }
        }
    }
    changed
}
