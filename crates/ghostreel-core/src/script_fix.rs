//! Deterministic fixes for what the checks find, and the framing repair pass.
//!
//! The checker (`script_check`) reports issues; an issue carries typed [`Fix`]es. Anything a rule
//! can decide is decided here, the same way for every caller: the repair pass applies it when
//! clips are made and scripts are saved or imported, and [`apply_fix`] applies one fix on demand
//! (a button in the editor, MCP `apply_fix`, `ghostreel script fix`).
//!
//! Framing rules (vertical and square cuts of wider footage):
//! - a speaking clip's edges are trimmed off a screen share or an empty picture when the trimmed
//!   clip still starts and ends on whole sentences and keeps its minimum length;
//! - the closing hold stays on a face: shortened to the part that is, dropped when none is.

use crate::Error;
use crate::db::Db;
use crate::framing::{self, Framing, Layout};
use crate::script::{Audio, Fix, FixAction, Script, ScriptClip};

/// An edge stretch shorter than this is left alone: the segmentation folds anything shorter than
/// a shot into its neighbour anyway.
pub const MIN_BAD_EDGE_S: f64 = 1.0;
/// A closing hold shorter than this is not worth keeping.
pub const MIN_HOLD_S: f64 = 0.6;
/// A sentence may start this much before the picture changes to faces and still count as
/// starting on them.
const START_SLACK_S: f64 = 0.3;

/// Whether a layout frames people (rather than a screen or the whole picture).
pub fn frames_people(layout: Layout) -> bool {
    matches!(layout, Layout::FollowSpeaker | Layout::Stacked | Layout::Fixed)
}

/// The shortest a trimmed clip may get when the caller gives no minimum: most of what it was.
pub fn default_min_len(len: f64) -> f64 {
    (len * 0.6).max(5.0)
}

/// Whether `a..b` (source seconds) is framed the way the editor chose: an override covers it.
pub fn chosen(framing: &Framing, a: f64, b: f64) -> bool {
    framing.overrides.iter().any(|o| o.from_s <= a + 0.05 && o.to_s >= b - 0.05)
}

/// The stretches of `from..to` of a video (source seconds) that the framing shows people in (or
/// frames as the editor chose, with an override), when the cut crops that video and its faces
/// were looked for. `None` when nothing is known.
pub fn people_spans(
    db: &Db,
    framing: &Framing,
    video_id: i64,
    from: f64,
    to: f64,
    out_w: u32,
    out_h: u32,
) -> Option<Vec<(f64, f64)>> {
    if to <= from || !crate::scenes::faces_analysed(db, video_id) {
        return None;
    }
    let plan = framing::plan_clip(db, None, None, video_id, from, to, framing, out_w as f64, out_h as f64).ok()??;
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for seg in plan.segments.iter().filter(|s| frames_people(s.layout) || chosen(framing, from + s.start, from + s.end))
    {
        let (a, b) = (from + seg.start, from + seg.end);
        match spans.last_mut() {
            Some(last) if (last.1 - a).abs() < 1e-3 => last.1 = b,
            _ => spans.push((a, b)),
        }
    }
    Some(spans)
}

/// Bounds for a speaking clip `in_s..out_s` that leave out a screen share (or empty picture) at
/// its edges: people are framed from `g0` to `g1`. Both bounds land on sentence edges (`edges`
/// are the transcript's segments), the end keeps the speech overrun, and the clip keeps at least
/// `min_len`. Of the trims that qualify, the one leaving out the most is chosen. `None` when no
/// trim qualifies or none is needed.
pub fn edge_trim(
    edges: &[(f64, f64)],
    in_s: f64,
    out_s: f64,
    g0: f64,
    g1: f64,
    min_len: f64,
    overrun: f64,
) -> Option<(f64, f64)> {
    let head = g0 - in_s >= MIN_BAD_EDGE_S;
    let tail = out_s - g1 >= MIN_BAD_EDGE_S;
    if !head && !tail {
        return None;
    }
    let new_in = head
        .then(|| edges.iter().map(|e| e.0).filter(|&s| s >= g0 - START_SLACK_S && s < g1).min_by(f64::total_cmp))
        .flatten();
    // An end whose overrun stays on the faces, else any end before the faces stop.
    let end_from = |lo: f64| {
        let ends = || edges.iter().map(|e| e.1).filter(move |&e| e > lo + 0.5 && e <= g1 + 0.05);
        ends()
            .filter(|&e| e + overrun <= g1 + 0.1)
            .max_by(f64::total_cmp)
            .or_else(|| ends().max_by(f64::total_cmp))
            .map(|e| (e + overrun).min(out_s))
    };
    let mut options: Vec<(f64, f64)> = Vec::new();
    if let Some(i) = new_in {
        if tail && let Some(o) = end_from(i) {
            options.push((i, o));
        }
        options.push((i, out_s));
    }
    if tail && let Some(o) = end_from(in_s) {
        options.push((in_s, o));
    }
    options
        .into_iter()
        .filter(|(a, b)| b - a >= min_len - 1e-6 && (a - in_s).abs() + (out_s - b).abs() > 0.05)
        .max_by(|x, y| ((x.0 - in_s) + (out_s - x.1)).total_cmp(&((y.0 - in_s) + (out_s - y.1))))
}

/// The closing hold of a cut: the silent picture that follows the last words on the same shot
/// (index of the last clip of the last beat), when there is one.
pub fn closing_hold(script: &Script) -> Option<(usize, usize)> {
    let bi = script.beats.len().checked_sub(1)?;
    let beat = &script.beats[bi];
    let ci = beat.clips.len().checked_sub(1)?;
    let hold = &beat.clips[ci];
    let prev = beat.clips.get(ci.checked_sub(1)?)?;
    (hold.audio == Audio::Mute
        && hold.why.as_deref() == Some(crate::chat::CLOSING_PICTURE)
        && hold.video_id == prev.video_id
        && (hold.in_s - prev.out_s).abs() < 0.05)
        .then_some((bi, ci))
}

/// Move a clip's bounds; a closing hold that followed it moves along, keeping its length.
fn set_bounds(script: &mut Script, bi: usize, ci: usize, in_s: f64, out_s: f64) {
    let hold = closing_hold(script).filter(|&(hb, hc)| hb == bi && hc == ci + 1);
    let clip = &mut script.beats[bi].clips[ci];
    clip.in_s = round3(in_s);
    clip.out_s = round3(out_s);
    if let Some((hb, hc)) = hold {
        let h = &mut script.beats[hb].clips[hc];
        let len = h.out_s - h.in_s;
        h.in_s = round3(out_s);
        h.out_s = round3(out_s + len);
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Where a framed clip's people stretch starts and ends, with the clip's framing and the cut's
/// size, when the cut crops it and something is known about its faces.
struct Framed {
    framing: Framing,
    out_w: u32,
    out_h: u32,
}

fn framed(script: &Script, clip: &ScriptClip) -> Option<Framed> {
    let framing = framing::effective(script, clip)?.clone();
    Some(Framed { framing, out_w: u32::try_from(script.width?).ok()?, out_h: u32::try_from(script.height?).ok()? })
}

/// A trim that takes a speaking clip off a screen share at its edges, when one qualifies.
pub fn framing_trim(
    db: &Db,
    script: &Script,
    bi: usize,
    ci: usize,
    min_clip_s: Option<f64>,
    overrun: f64,
) -> Option<(f64, f64)> {
    let clip = script.beats.get(bi)?.clips.get(ci)?;
    if clip.audio != Audio::Source {
        return None;
    }
    let f = framed(script, clip)?;
    let spans = people_spans(db, &f.framing, clip.video_id, clip.in_s, clip.out_s, f.out_w, f.out_h)?;
    let (g0, g1) = (spans.first()?.0, spans.last()?.1);
    let edges = crate::chat::sentence_edges(db, clip.video_id);
    let len = clip.out_s - clip.in_s;
    let min_len = min_clip_s.unwrap_or_else(|| default_min_len(len)).min(len);
    edge_trim(&edges, clip.in_s, clip.out_s, g0, g1, min_len, overrun)
}

/// How long the closing hold can stay on a face from where it starts (seconds, at most its
/// length), when it is framed and its faces are known.
pub fn hold_on_face(db: &Db, script: &Script) -> Option<(usize, usize, f64)> {
    let (bi, ci) = closing_hold(script)?;
    let hold = &script.beats[bi].clips[ci];
    let f = framed(script, hold)?;
    // Judge the hold with a few seconds either side: a stretch shorter than a shot at the edge of
    // a range is folded into its neighbour, so a hold judged alone could hide a screen share.
    let duration: f64 = db
        .conn
        .query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [hold.video_id], |r| r.get(0))
        .unwrap_or(0.0);
    let from = (hold.in_s - 3.0).max(0.0);
    let to = if duration > hold.out_s { (hold.out_s + 3.0).min(duration) } else { hold.out_s };
    let spans = people_spans(db, &f.framing, hold.video_id, from, to, f.out_w, f.out_h)?;
    let good = spans
        .iter()
        .find(|(a, b)| hold.in_s >= *a - 1e-3 && hold.in_s < *b)
        .map(|(_, b)| b.min(hold.out_s) - hold.in_s)
        .unwrap_or(0.0);
    Some((bi, ci, good.max(0.0)))
}

/// The framing repair pass: trim speaking clips off screen shares at their edges (whole
/// sentences, minimum length kept), then keep the closing hold on a face. Only for cuts that crop
/// (width/height differ from the footage) and footage whose faces were looked for. Returns one
/// note per change, like the other repair passes.
pub fn repair_framing(db: &Db, script: &mut Script, overrun: f64, min_clip_s: Option<f64>) -> Vec<String> {
    let mut notes = Vec::new();
    if script.width.is_none() || script.height.is_none() {
        return notes;
    }
    for bi in 0..script.beats.len() {
        for ci in 0..script.beats[bi].clips.len() {
            let Some((a, b)) = framing_trim(db, script, bi, ci, min_clip_s, overrun) else { continue };
            let c = &script.beats[bi].clips[ci];
            let (old_in, old_out, video) = (c.in_s, c.out_s, c.video_id);
            set_bounds(script, bi, ci, a, b);
            // A target that was the clip's own length follows it.
            if let Some(t) = script.target_duration_s.as_mut() {
                *t = round3((*t - ((a - old_in) + (old_out - b))).max(1.0));
            }
            notes.push(trim_note(video, old_in, old_out, a, b));
        }
    }
    if let Some((bi, ci, good)) = hold_on_face(db, script) {
        let hold = &script.beats[bi].clips[ci];
        let len = hold.out_s - hold.in_s;
        if good < len - 0.05 {
            if good >= MIN_HOLD_S {
                let out = round3(hold.in_s + good);
                script.beats[bi].clips[ci].out_s = out;
                notes.push(format!("shortened the closing hold to {good:.1} s so it stays on a face"));
            } else {
                script.beats[bi].clips.remove(ci);
                notes.push("dropped the closing hold: it would have shown a screen share, not a face".to_string());
            }
        }
    }
    notes
}

fn trim_note(video: i64, old_in: f64, old_out: f64, a: f64, b: f64) -> String {
    let mut parts = Vec::new();
    if a > old_in + 0.05 {
        parts.push(format!("starts {:.1} s later", a - old_in));
    }
    if b < old_out - 0.05 {
        parts.push(format!("ends {:.1} s sooner", old_out - b));
    }
    format!(
        "trimmed video #{video} {old_in:.1}–{old_out:.1} s off a screen share: {} ({a:.1}–{b:.1} s, whole sentences)",
        parts.join(" and ")
    )
}

/// What applying a fix did.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Applied {
    pub note: String,
    /// An indexing stage the caller should now run for the project (the script is unchanged).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_stage: Option<String>,
}

fn find_clip(script: &Script, beat_id: &str, clip_index: usize) -> Result<usize, Error> {
    let bi = script
        .beats
        .iter()
        .position(|b| b.id == beat_id)
        .ok_or_else(|| Error::NotFound(format!("beat '{beat_id}'")))?;
    if clip_index >= script.beats[bi].clips.len() {
        return Err(Error::NotFound(format!("clip {} of beat '{beat_id}'", clip_index + 1)));
    }
    Ok(bi)
}

/// Apply one fix to a script (in memory; saving is the caller's). A `run_stage` fix turns the
/// stage on for the project when asked, and returns it for the caller to run.
pub fn apply_fix(db: &mut Db, project_id: i64, script: &mut Script, fix: &FixAction) -> Result<Applied, Error> {
    match fix {
        FixAction::Trim { beat_id, clip_index, in_s, out_s } => {
            if out_s <= in_s {
                return Err(Error::Invalid(format!("trim to {in_s:.2}–{out_s:.2} s is empty")));
            }
            let bi = find_clip(script, beat_id, *clip_index)?;
            let c = &script.beats[bi].clips[*clip_index];
            let (old_in, old_out, video) = (c.in_s, c.out_s, c.video_id);
            set_bounds(script, bi, *clip_index, *in_s, *out_s);
            Ok(Applied {
                note: format!("video #{video}: {old_in:.1}–{old_out:.1} s is now {in_s:.1}–{out_s:.1} s"),
                run_stage: None,
            })
        }
        FixAction::SetLayout { beat_id, clip_index, from_s, to_s, layout } => {
            let bi = find_clip(script, beat_id, *clip_index)?;
            let base = framing::effective(script, &script.beats[bi].clips[*clip_index]).cloned().unwrap_or_default();
            let clip = &mut script.beats[bi].clips[*clip_index];
            let (from, to) = (from_s.max(clip.in_s), to_s.min(clip.out_s));
            if to <= from {
                return Err(Error::Invalid(format!("{from_s:.2}–{to_s:.2} s is outside the clip")));
            }
            clip.framing = Some(base.with_override(from, to, *layout));
            Ok(Applied {
                note: format!("video #{}: {from:.1}–{to:.1} s framed as {}", clip.video_id, layout.label()),
                run_stage: None,
            })
        }
        FixAction::RunStage { stage, enable } => {
            if !crate::index::ALL_STAGES.contains(&stage.as_str()) {
                return Err(Error::Invalid(format!("unknown stage '{stage}'")));
            }
            let mut pipeline = db.project(project_id)?.pipeline;
            let mut note = format!("{stage} queued");
            if *enable && !pipeline.is_enabled(stage) {
                pipeline.set_enabled(stage, true)?;
                db.update_project_pipeline(project_id, &pipeline)?;
                note = format!("{stage} turned on and queued");
            }
            Ok(Applied { note, run_stage: Some(stage.clone()) })
        }
    }
}

/// Fixes worth offering for a framed stretch: frame it another way.
pub fn layout_fixes(beat_id: &str, clip_index: usize, from_s: f64, to_s: f64, current: Layout) -> Vec<Fix> {
    [Layout::Screen, Layout::Fit, Layout::FollowSpeaker]
        .into_iter()
        .filter(|l| *l != current)
        .map(|layout| {
            Fix::new(
                format!("Frame {from_s:.1}–{to_s:.1} s as {}", layout.label()),
                FixAction::SetLayout { beat_id: beat_id.to_string(), clip_index, from_s, to_s, layout },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_screen_share_edges_on_sentences_and_keeps_the_minimum() {
        // Sentences every 5 s; people framed from 12.8 s to 42.8 s of a 0–48 s clip.
        let edges: Vec<(f64, f64)> = (0..10).map(|i| (i as f64 * 5.0, i as f64 * 5.0 + 4.6)).collect();
        let t = edge_trim(&edges, 0.0, 48.0, 12.8, 42.8, 20.0, 0.35).unwrap();
        assert_eq!(t.0, 15.0, "first sentence starting on the faces");
        assert!((t.1 - 39.95).abs() < 1e-9, "last sentence ending on the faces, overrun kept: {t:?}");
        // Too short once trimmed: trims one side only, or nothing.
        let t = edge_trim(&edges, 0.0, 48.0, 12.8, 42.8, 30.0, 0.35).unwrap();
        assert!(t.1 - t.0 >= 30.0, "{t:?}");
        assert!(edge_trim(&edges, 0.0, 48.0, 12.8, 42.8, 46.0, 0.35).is_none());
        // A sentence starting just before the faces counts as starting on them.
        let edges = vec![(0.0, 12.5), (12.6, 30.0), (30.1, 47.0)];
        assert_eq!(edge_trim(&edges, 0.0, 47.35, 12.8, 47.35, 10.0, 0.35), Some((12.6, 47.35)));
        // One long sentence across the change: nothing to trim to.
        assert!(edge_trim(&[(0.0, 29.0)], 0.0, 29.35, 12.8, 29.35, 10.0, 0.35).is_none());
        // Short bad edges are left alone.
        assert!(edge_trim(&edges, 0.0, 47.35, 0.6, 47.0, 10.0, 0.35).is_none());
    }

    #[test]
    fn overrides_replace_the_stretch_they_cover() {
        let f = Framing::default().with_override(10.0, 20.0, Layout::Fit).with_override(15.0, 25.0, Layout::Screen);
        let got: Vec<(f64, f64, Layout)> = f.overrides.iter().map(|o| (o.from_s, o.to_s, o.layout)).collect();
        assert_eq!(got, vec![(10.0, 15.0, Layout::Fit), (15.0, 25.0, Layout::Screen)]);
    }
}
