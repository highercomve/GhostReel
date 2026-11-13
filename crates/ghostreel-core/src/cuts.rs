//! Cut placement on the sound: where a speaking clip opens and closes, judged from where speech
//! actually is (`speech.rs`) instead of from the transcript's segment times.
//!
//! Whisper's times are approximate: often late at a start, early or quantised at an end. A clip
//! that ends on a segment boundary can end exactly as the next speaker's first word begins, and
//! you hear its attack. This is the last pass of the repair chain (`script_check::repair_for_import`,
//! `chat::repair::finish_script`), after everything that moves a clip has moved it, and it only
//! runs for videos the speech stage has measured.
//!
//! The rules, for a clip or an audio bed with `[in, out]`:
//!
//! - **End.** The speech the range contains is the last region that starts at least a guard before
//!   `out`. The cut goes `speech_overrun_s` after that region ends (the word's decay, a breath),
//!   but at least `cut_guard_s` before the next voice comes in. When the pause is too short for
//!   both, the cut goes at the quietest 10 ms frame of the pause. An end on or inside the next
//!   voice is pulled back; one that cuts into the last word is pushed out.
//! - **Start.** The range opens `cut_lead_s` before the first voice it contains, but not before
//!   the previous speech has ended (plus a margin): it never opens inside somebody else's word.
//! - **Frames.** The renderer snaps a clip's duration to whole frames of the cut's frame rate, so a
//!   start is rounded up and an end down to that grid: what plays is what was placed, and the guard
//!   survives.
//! - **Reach.** An edge moves at most `cut_max_move_s`. Past that the clip is a different sentence,
//!   not a different breath, and is left as it was.
//!
//! Placement depends only on the stored regions and the range's own edges, so repairing an
//! already-repaired script finds nothing to do.

use rusqlite::OptionalExtension;

use crate::config::ScriptConfig;
use crate::db::Db;
use crate::script::{Audio, Script};
use crate::speech::SpeechRegion;

/// A pause shorter than the guard plus this has no room for both a tail and a guard (s).
const MIN_TAIL_S: f64 = 0.05;
/// The least a start is placed before the voice it opens on (s).
const MIN_LEAD_S: f64 = 0.05;
/// How far after the previous speech has ended a start may be placed (s).
const IN_MARGIN_S: f64 = 0.04;
/// A range that opens this far into a region has cut off its first word's attack; deeper in, it
/// opens mid-phrase on purpose (s).
const FIRST_WORD_S: f64 = 0.25;
/// A move smaller than this is not worth a note (s).
const NOTE_MIN_S: f64 = 0.05;
/// A placed range keeps at least this much (s).
const MIN_KEPT_S: f64 = 0.1;
/// How much audio is looked at around a range (s).
const LOOK_BEHIND_S: f64 = 3.0;
const LOOK_AHEAD_S: f64 = 4.0;

/// What a placement did, in the repair chain's typed notes.
#[derive(Debug, Clone, PartialEq)]
pub struct CutMove {
    /// 0-based position of the beat.
    pub beat_index: usize,
    /// 0-based clip in the beat; `None` for the beat's audio bed.
    pub clip_index: Option<usize>,
    pub kind: CutKind,
    pub from_s: f64,
    pub to_s: f64,
    /// More than one beat in the script: say which.
    pub several_beats: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutKind {
    /// The end sat on or inside the next voice's first word.
    EndOffNextSpeaker,
    /// The end cut into the last word.
    EndLetWordFinish,
    /// The end was somewhere the sound says is wrong for it (too tight, or on the wrong side).
    EndRetimed,
    /// The start was inside the previous speech.
    StartOffPreviousSpeaker,
    /// The start cut off the attack of the first word.
    StartBeforeFirstWord,
    /// The start left dead air before the first word, or too little.
    StartRetimed,
    /// A closing hold carried sound; it is muted now.
    HoldMuted,
}

impl CutMove {
    /// The note a repair reports, e.g. "moved the end of clip 1 off the next speaker's first word
    /// (12.86 → 12.40 s)".
    pub fn message(&self) -> String {
        let subject = match (self.clip_index, self.several_beats) {
            (Some(c), false) => format!("clip {}", c + 1),
            (Some(c), true) => format!("clip {} of beat {}", c + 1, self.beat_index + 1),
            (None, false) => "the audio bed".to_string(),
            (None, true) => format!("the audio bed of beat {}", self.beat_index + 1),
        };
        let moved = format!("{:.2} → {:.2} s", self.from_s, self.to_s);
        match self.kind {
            CutKind::EndOffNextSpeaker => {
                format!("moved the end of {subject} off the next speaker's first word ({moved})")
            }
            CutKind::EndLetWordFinish => format!("let the last word of {subject} finish ({moved})"),
            CutKind::EndRetimed => format!("placed the end of {subject} on the pause after the last word ({moved})"),
            CutKind::StartOffPreviousSpeaker => {
                format!("moved the start of {subject} off the previous speaker ({moved})")
            }
            CutKind::StartBeforeFirstWord => format!("opened {subject} before its first word starts ({moved})"),
            CutKind::StartRetimed => format!("placed the start of {subject} just before its first word ({moved})"),
            CutKind::HoldMuted => format!("muted the closing hold after {subject}: it would carry the next voice"),
        }
    }
}

/// The numbers placement works with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rules {
    /// The tail after the last word (s): `speech_overrun_s`.
    pub tail: f64,
    pub guard: f64,
    pub lead: f64,
    pub max_move: f64,
    /// The cut's frame rate.
    pub fps: f64,
    /// The video's length, when known.
    pub duration: f64,
}

impl Rules {
    pub fn from_config(cfg: &ScriptConfig, fps: f64, duration: f64) -> Self {
        Self {
            tail: cfg.speech_overrun_s,
            guard: cfg.cut_guard_s,
            lead: cfg.cut_lead_s,
            max_move: cfg.cut_max_move_s,
            fps,
            duration,
        }
    }

    /// The last frame boundary at or before `t`.
    fn floor_frame(&self, t: f64) -> f64 {
        ((t * self.fps + 1e-6).floor()) / self.fps
    }

    /// The first frame boundary at or after `t`.
    fn ceil_frame(&self, t: f64) -> f64 {
        ((t * self.fps - 1e-6).ceil()) / self.fps
    }

    /// `t` rounded down to a frame, kept inside `lo..=hi`: when the frame below falls under `lo`,
    /// the frame above if that is still under `hi`, else `t` itself.
    fn down_within(&self, t: f64, lo: f64, hi: f64) -> f64 {
        let f = self.floor_frame(t);
        if f >= lo - 1e-9 {
            return f;
        }
        let c = self.ceil_frame(t);
        if c <= hi + 1e-9 { c } else { t }
    }

    /// `t` rounded up to a frame, kept inside `lo..=hi` the same way.
    fn up_within(&self, t: f64, lo: f64, hi: f64) -> f64 {
        let c = self.ceil_frame(t);
        if c <= hi + 1e-9 {
            return c;
        }
        let f = self.floor_frame(t);
        if f >= lo - 1e-9 { f } else { t }
    }
}

/// Where the cut after `regions[last]` goes in the pause before the next region.
fn pause_cut(last: &SpeechRegion, next: &SpeechRegion) -> f64 {
    last.quiet_after_s.filter(|q| *q >= last.end_s && *q < next.start_s).unwrap_or((last.end_s + next.start_s) / 2.0)
}

/// Where a range ending at `out_s` should end, given the speech around it; `None` to leave it.
/// `push_limit`: the furthest an end may be pushed out without reaching into the next sentence the
/// transcript knows of (the audio alone cannot tell two sentences spoken without a pause).
pub fn place_out(
    regions: &[SpeechRegion],
    in_s: f64,
    out_s: f64,
    push_limit: Option<f64>,
    r: &Rules,
) -> Option<(f64, CutKind)> {
    // The speech the range really contains: it starts a guard or more before the end. A voice that
    // comes in inside that last guard is the next speaker's, not this clip's.
    let li = regions.iter().rposition(|x| x.start_s < out_s - r.guard && x.end_s > in_s)?;
    let last = &regions[li];
    let next = regions.get(li + 1);

    let (target, ceiling) = match next {
        Some(n) => {
            let ceiling = n.start_s - r.guard;
            if ceiling - last.end_s < MIN_TAIL_S {
                // No room for a tail and a guard: the quietest moment of the pause.
                let t = r.down_within(pause_cut(last, n), last.end_s, n.start_s - 0.01);
                return finish_out(out_s, t, last, ceiling, r, push_limit);
            }
            ((last.end_s + r.tail).min(ceiling), ceiling)
        }
        None => (last.end_s + r.tail, f64::INFINITY),
    };
    let target = r.down_within(target.min(r.duration), last.end_s, ceiling.min(r.duration));
    finish_out(out_s, target, last, ceiling, r, push_limit)
}

/// Apply the reach and the push limit to a placed end and name what it did.
fn finish_out(
    out_s: f64,
    target: f64,
    last: &SpeechRegion,
    ceiling: f64,
    r: &Rules,
    push_limit: Option<f64>,
) -> Option<(f64, CutKind)> {
    let mut target = target;
    if target > out_s
        && let Some(limit) = push_limit
    {
        let capped = target.min(limit.max(out_s));
        if capped < target {
            target = r.floor_frame(capped).max(out_s);
        }
    }
    if (target - out_s).abs() < 1e-9 || (target - out_s).abs() > r.max_move {
        return None;
    }
    let kind = if out_s > ceiling {
        CutKind::EndOffNextSpeaker
    } else if out_s < last.end_s {
        CutKind::EndLetWordFinish
    } else {
        CutKind::EndRetimed
    };
    Some((target, kind))
}

/// Where a range starting at `in_s` should start, given the speech around it; `None` to leave it.
/// `pull_limit`: the earliest a start may be pulled back without reaching into the previous
/// sentence the transcript knows of (two sentences spoken without a pause are one region of sound,
/// and a lead taken from that region opens on the other sentence's last word).
pub fn place_in(
    regions: &[SpeechRegion],
    in_s: f64,
    out_s: f64,
    pull_limit: Option<f64>,
    r: &Rules,
) -> Option<(f64, CutKind)> {
    // The first speech the range really contains: it ends a guard or more after the start. The
    // tail of the previous speaker, still decaying at the start, is not this clip's.
    let fi = regions.iter().position(|x| x.end_s > in_s + r.guard)?;
    let first = &regions[fi];
    if first.start_s >= out_s - MIN_KEPT_S {
        return None;
    }
    let onset = first.start_s;
    // Deep inside a phrase: opened there on purpose.
    if in_s > onset + FIRST_WORD_S {
        return None;
    }
    let prev = fi.checked_sub(1).map(|i| &regions[i]);
    let (target, lo) = match prev {
        Some(p) if onset - p.end_s < IN_MARGIN_S + MIN_LEAD_S => {
            let q = pause_cut(p, first);
            (q, p.end_s)
        }
        Some(p) => ((p.end_s + IN_MARGIN_S).max(onset - r.lead), p.end_s),
        None => ((onset - r.lead).max(0.0), 0.0),
    };
    let hi = (onset - MIN_LEAD_S).max(lo);
    let mut target = r.up_within(target.max(0.0), lo, hi);
    if target < in_s
        && let Some(limit) = pull_limit
        && target < limit
    {
        target = r.ceil_frame(limit.min(in_s)).min(in_s);
    }
    if (target - in_s).abs() < 1e-9 || (target - in_s).abs() > r.max_move {
        return None;
    }
    let prev_end = prev.map_or(f64::NEG_INFINITY, |p| p.end_s);
    let kind = if in_s < prev_end + IN_MARGIN_S - 1e-9 {
        CutKind::StartOffPreviousSpeaker
    } else if in_s > onset {
        CutKind::StartBeforeFirstWord
    } else {
        CutKind::StartRetimed
    };
    Some((target, kind))
}

/// The tail after the last voiced sound a cut should leave when the pause has room for it (s).
const CLEAN_TAIL_S: f64 = 0.1;
/// A cut this far inside a voiced stretch has cut a word; less is the measurement's own noise (s).
const INSIDE_VOICE_S: f64 = 0.02;
/// A start this far past the first sound of a voice has cut off the attack of its first word (s).
const MIDWORD_START_S: f64 = 0.03;

/// What the sound says about where a range ends. The one judgement the check
/// (`script_check::check`) and `read_cut` both report, with the same words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndFlag {
    /// In silence, after the last word and well before the next voice.
    Clean,
    /// Inside the last word, or so close to its end that the decay is cut.
    ClipsLastWord,
    /// On or inside the next voice's first sound (or within the guard before it).
    NextSpeakerLeaksIn,
    /// Between the word the clip was written to end on and the word after it, inside running
    /// speech: where it was asked to end, with no pause to end in.
    BetweenWords,
}

impl EndFlag {
    pub fn label(self) -> &'static str {
        match self {
            EndFlag::Clean => "clean (in silence)",
            EndFlag::ClipsLastWord => "clips the last word",
            EndFlag::NextSpeakerLeaksIn => "next speaker leaks in",
            EndFlag::BetweenWords => "between words, as written (running speech, no pause)",
        }
    }

    pub fn is_clean(self) -> bool {
        matches!(self, EndFlag::Clean | EndFlag::BetweenWords)
    }
}

/// What the sound says about where a range starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartFlag {
    Clean,
    /// Inside the first word (its attack is gone) or inside the previous speaker's last one.
    StartsMidWord,
    /// Between the word before and the word the clip was written to start on, inside running
    /// speech: where it was asked to start, with no pause to start in.
    BetweenWords,
}

impl StartFlag {
    pub fn label(self) -> &'static str {
        match self {
            StartFlag::Clean => "clean (in silence)",
            StartFlag::StartsMidWord => "starts mid-word",
            StartFlag::BetweenWords => "between words, as written (running speech, no pause)",
        }
    }

    pub fn is_clean(self) -> bool {
        matches!(self, StartFlag::Clean | StartFlag::BetweenWords)
    }
}

/// The end of a range against the measured speech.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct EndReading {
    pub flag: EndFlag,
    /// The cut minus the end of the last voiced stretch the range holds (negative: inside it).
    pub after_voice_s: Option<f64>,
    /// The next voice's first sound minus the cut (below the guard: it leaks in).
    pub before_next_s: Option<f64>,
}

impl EndReading {
    /// "clips the last word (the voice runs 0.12 s past the cut)": the flag and the numbers behind it.
    pub fn detail(&self) -> String {
        let label = self.flag.label();
        match (self.flag, self.after_voice_s, self.before_next_s) {
            (EndFlag::ClipsLastWord, Some(a), _) if a < 0.0 => {
                format!("{label} (the voice runs {:.2} s past the cut)", -a)
            }
            (EndFlag::ClipsLastWord, Some(a), _) => format!("{label} (only {a:.2} s of tail after it)"),
            (EndFlag::NextSpeakerLeaksIn, _, Some(b)) if b >= 0.0 => {
                format!("{label} (the next voice starts {b:.2} s after the cut)")
            }
            (EndFlag::NextSpeakerLeaksIn, _, Some(b)) => {
                format!("{label} (the next voice starts {:.2} s before the cut)", -b)
            }
            (EndFlag::Clean, Some(a), _) => format!("{label}, {a:.2} s after the last word"),
            _ => label.to_string(),
        }
    }
}

/// The start of a range against the measured speech.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct StartReading {
    pub flag: StartFlag,
    /// The first voice's first sound minus the cut (negative: the cut is inside the word).
    pub before_voice_s: Option<f64>,
    /// The cut minus the end of the speech before it (negative: inside the previous speaker's word).
    pub after_previous_s: Option<f64>,
}

impl StartReading {
    pub fn detail(&self) -> String {
        let label = self.flag.label();
        match (self.flag, self.before_voice_s, self.after_previous_s) {
            (StartFlag::StartsMidWord, _, Some(p)) if p < 0.0 => {
                format!("{label} (inside the previous speaker's last word, {:.2} s before it ends)", -p)
            }
            (StartFlag::StartsMidWord, Some(b), _) => {
                format!("{label} (the first word began {:.2} s before the cut)", -b)
            }
            (StartFlag::Clean, Some(b), _) if b >= 0.0 => format!("{label}, {b:.2} s before the first word"),
            _ => label.to_string(),
        }
    }
}

/// Judge where a range ends. `regions` are the speech regions around it.
pub fn judge_end(regions: &[SpeechRegion], in_s: f64, out_s: f64, r: &Rules) -> EndReading {
    let clean = |after_voice_s, before_next_s| EndReading { flag: EndFlag::Clean, after_voice_s, before_next_s };
    let Some(li) = regions.iter().rposition(|x| x.start_s < out_s - r.guard && x.end_s > in_s) else {
        // Nothing the range contains began a guard before its end. A voice that begins inside that
        // last guard is the next speaker's attack; anything else is silence.
        return match regions.iter().find(|x| x.end_s > in_s && x.start_s < out_s + r.guard) {
            Some(n) => EndReading {
                flag: EndFlag::NextSpeakerLeaksIn,
                after_voice_s: None,
                before_next_s: Some(n.start_s - out_s),
            },
            None => clean(None, None),
        };
    };
    let last = &regions[li];
    let next = regions.get(li + 1);
    let after_voice_s = out_s - last.end_s;
    let before_next_s = next.map(|n| n.start_s - out_s);
    // Room between the word's end and the guard before the next voice.
    let room = next.map_or(f64::INFINITY, |n| n.start_s - r.guard - last.end_s);
    let leaks = next.is_some_and(|n| {
        if room < MIN_TAIL_S {
            // Too short a pause for a tail and a guard: anywhere inside the pause will do.
            out_s >= n.start_s - 0.01
        } else {
            out_s > n.start_s - r.guard + 1e-6
        }
    });
    // With room for it the cut leaves a tail (less one frame, which the renderer may take); in a
    // pause too short for one, only a cut inside the word is a clipped word.
    let clips = if room >= CLEAN_TAIL_S {
        after_voice_s < CLEAN_TAIL_S - 1.0 / r.fps.max(1.0) - 1e-6
    } else {
        after_voice_s < -INSIDE_VOICE_S
    };
    let flag = if leaks {
        EndFlag::NextSpeakerLeaksIn
    } else if clips {
        EndFlag::ClipsLastWord
    } else {
        EndFlag::Clean
    };
    EndReading { flag, after_voice_s: Some(after_voice_s), before_next_s }
}

/// Judge where a range starts.
pub fn judge_start(regions: &[SpeechRegion], in_s: f64, out_s: f64, r: &Rules) -> StartReading {
    let clean = StartReading { flag: StartFlag::Clean, before_voice_s: None, after_previous_s: None };
    let Some(fi) = regions.iter().position(|x| x.end_s > in_s + r.guard) else { return clean };
    let first = &regions[fi];
    if first.start_s >= out_s - MIN_KEPT_S {
        return clean;
    }
    let onset = first.start_s;
    let after_previous_s = fi.checked_sub(1).map(|i| in_s - regions[i].end_s);
    let into_attack = in_s > onset + MIDWORD_START_S && in_s <= onset + FIRST_WORD_S;
    let inside_previous = after_previous_s.is_some_and(|p| p < -0.01);
    StartReading {
        flag: if into_attack || inside_previous { StartFlag::StartsMidWord } else { StartFlag::Clean },
        before_voice_s: Some(onset - in_s),
        after_previous_s,
    }
}

/// Both ends of a range on the sound. `None` when the speech stage has not measured the video:
/// callers then fall back to the transcript's times.
/// `words`: the word neighbours of an edge written as a word (`s412@3`); a cut that sits between
/// that word and its neighbour inside running speech is where it was asked to be, and says so.
pub fn judge_range(
    db: &Db,
    video_id: i64,
    in_s: f64,
    out_s: f64,
    words: crate::sentences::WordEdges,
    cfg: &ScriptConfig,
    fps: f64,
) -> Option<(StartReading, EndReading)> {
    if !crate::speech::analysed(db, video_id) {
        return None;
    }
    let regions = crate::speech::speech_regions(db, video_id, in_s - LOOK_BEHIND_S, out_s + LOOK_AHEAD_S).ok()?;
    let rules = Rules::from_config(cfg, fps, f64::MAX);
    let mut start = judge_start(&regions, in_s, out_s, &rules);
    let mut end = judge_end(&regions, in_s, out_s, &rules);
    if let Some((prev, this)) = words.start
        && start.flag == StartFlag::StartsMidWord
        && in_s >= prev - WORD_SEARCH_S
        && in_s <= this + WORD_SEARCH_S
    {
        start.flag = StartFlag::BetweenWords;
    }
    if let Some((this, next)) = words.end
        && matches!(end.flag, EndFlag::ClipsLastWord | EndFlag::NextSpeakerLeaksIn)
        && out_s >= this - WORD_SEARCH_S
        && out_s <= next + WORD_SEARCH_S
    {
        end.flag = EndFlag::BetweenWords;
    }
    Some((start, end))
}

/// Room left between a cut on a word and the neighbouring word (s).
const WORD_GAP_S: f64 = 0.02;
/// How far past its estimated edges a word boundary is looked for (s): word times inside a
/// sentence are estimates, a few tenths out.
const WORD_SEARCH_S: f64 = 0.25;
/// Frames averaged when looking for the quiet between two words: a gap, not one quiet sample
/// inside a consonant.
const DIP_FRAMES: usize = 5;

/// The quietest moment of `levels` (dBFS, one per 10 ms frame, the first at `t0`) between `from`
/// and `to`, averaged over [`DIP_FRAMES`]: where one word ends and the next begins.
pub fn quietest(levels: &[f32], t0: f64, from: f64, to: f64) -> Option<f64> {
    let frame = crate::speech::FRAME_S;
    let idx = |t: f64| (((t - t0) / frame).round().max(0.0) as usize).min(levels.len());
    let (a, b) = (idx(from), idx(to));
    if b <= a || levels.is_empty() {
        return None;
    }
    let half = DIP_FRAMES / 2;
    (a..b)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(levels.len());
            let mean = levels[lo..hi].iter().map(|&l| l as f64).sum::<f64>() / (hi - lo) as f64;
            (i, mean)
        })
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(i, _)| t0 + i as f64 * frame)
}

/// The 10 ms levels of a video's speech track from `a` to `b`, decoded now (a fraction of a
/// second of audio). None when the file or ffmpeg cannot be had.
fn levels_between(db: &Db, video_id: i64, a: f64, b: f64) -> Option<Vec<f32>> {
    let ffmpeg = crate::doctor::locate("ffmpeg")?;
    let (path, track): (String, i64) = db
        .conn
        .query_row(
            "SELECT vf.path, COALESCE(v.audio_track, 0) FROM video_files vf JOIN videos v ON v.id = vf.video_id
              WHERE vf.video_id = ?1 ORDER BY vf.last_seen DESC LIMIT 1",
            [video_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok()?;
    let out = crate::proc::std_command(&ffmpeg)
        .args(["-nostdin", "-v", "error", "-ss", &format!("{:.3}", a.max(0.0)), "-t", &format!("{:.3}", b - a)])
        .arg("-i")
        .arg(&path)
        .args(["-map", &format!("0:a:{track}"), "-vn", "-ac", "1", "-ar", &crate::speech::SAMPLE_RATE.to_string()])
        .args(["-f", "f32le", "-"])
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    let mut meter = crate::speech::FrameMeter::default();
    meter.push_bytes(&out.stdout);
    Some(meter.finish())
}

/// Where the boundary between two words really is: the quiet between them in the sound, looked
/// for around their estimated edges (`end_of_first`, `start_of_second`).
/// `span`: from the first word's start to the second's end — the search never leaves it.
fn word_boundary(
    db: &Db,
    video_id: i64,
    (end_of_first, start_of_second): (f64, f64),
    span: Option<(f64, f64)>,
) -> Option<f64> {
    let (lo, hi) = span.map_or((f64::MIN, f64::MAX), |(a, b)| (a + 0.03, b - 0.03));
    let from = (end_of_first.min(start_of_second) - WORD_SEARCH_S).max(lo);
    let to = (end_of_first.max(start_of_second) + WORD_SEARCH_S).min(hi);
    if to <= from {
        return None;
    }
    let pad = 0.05;
    let levels = levels_between(db, video_id, from - pad, to + pad)?;
    quietest(&levels, (from - pad).max(0.0), from, to)
}

/// The start of the next sentence after `out_s`, minus the guard.
fn push_limit(db: &Db, video_id: i64, out_s: f64, guard: f64) -> Option<f64> {
    db.conn
        .query_row(
            "SELECT MIN(start_s) FROM transcript_sentences WHERE video_id = ?1 AND start_s > ?2",
            rusqlite::params![video_id, out_s - 0.05],
            |r| r.get::<_, Option<f64>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten()
        .map(|s| s - guard)
}

/// The end of the last sentence before `in_s`, when it ends right where the range starts: a start
/// must not be pulled back past it.
fn pull_limit(db: &Db, video_id: i64, in_s: f64) -> Option<f64> {
    db.conn
        .query_row(
            "SELECT MAX(end_s) FROM transcript_sentences WHERE video_id = ?1 AND end_s < ?2 + 0.05 AND end_s > ?2 - 1.0",
            rusqlite::params![video_id, in_s],
            |r| r.get::<_, Option<f64>>(0),
        )
        .optional()
        .ok()
        .flatten()
        .flatten()
}

/// Place one range on the sound. `None` when the video has no measured speech near it.
fn place_range(
    db: &Db,
    video_id: i64,
    in_s: f64,
    out_s: f64,
    words: crate::sentences::WordEdges,
    cfg: &ScriptConfig,
    fps: f64,
) -> Option<RangeMove> {
    let regions = crate::speech::speech_regions(db, video_id, in_s - LOOK_BEHIND_S, out_s + LOOK_AHEAD_S).ok()?;
    if regions.is_empty() {
        return None;
    }
    let duration = db
        .conn
        .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
        .ok()
        .flatten()
        .filter(|d| *d > 0.0)
        .unwrap_or(f64::MAX);
    let rules = Rules::from_config(cfg, fps, duration);
    // A clip written to end on a word stops before the next word starts, and one written to start
    // on a word opens after the previous one ends: inside running speech the next pause is the
    // sentence's end, and the whole rest of the sentence is not what was asked for.
    let mut limit = push_limit(db, video_id, out_s, rules.guard);
    if let Some((_, next)) = words.end {
        limit = Some(limit.map_or(next, |l| l.min(next)) - WORD_GAP_S);
    }
    let mut pull = pull_limit(db, video_id, in_s);
    if let Some((prev, _)) = words.start {
        pull = Some(pull.map_or(prev, |l| l.max(prev)) + WORD_GAP_S);
    }
    // A word edge goes on the quiet between that word and its neighbour, measured in the sound:
    // inside running speech that is the only clean place there is.
    let word_in =
        words.start.and_then(|gap| word_boundary(db, video_id, gap, words.start_span)).map(|t| rules.ceil_frame(t));
    let word_out =
        words.end.and_then(|gap| word_boundary(db, video_id, gap, words.end_span)).map(|t| rules.floor_frame(t));
    let mut new_out = match word_out {
        Some(t) if (t - out_s).abs() > 1e-9 => Some((t, CutKind::EndRetimed)),
        Some(_) => None,
        None => place_out(&regions, in_s, out_s, limit, &rules),
    };
    let mut new_in = match word_in {
        Some(t) if (t - in_s).abs() > 1e-9 => Some((t, CutKind::StartRetimed)),
        Some(_) => None,
        None => place_in(&regions, in_s, out_s, pull, &rules),
    };
    // A word start inside running speech has no pause to lead in from; it gets what lead there is
    // between the words (word times are estimates, and a start on one clips its attack).
    if new_in.is_none()
        && word_in.is_none()
        && let Some((prev, _)) = words.start
    {
        let to = rules.ceil_frame((in_s - rules.lead).max(prev + WORD_GAP_S));
        if to < in_s - 1e-9 {
            new_in = Some((to, CutKind::StartBeforeFirstWord));
        }
    }
    // An edge written on a word never goes past the middle of the gap to the word beside it: word
    // times run a little early or late, and a cut on "the end of the previous word + a breath" lets
    // the tail of that word in ("en" before "s613@12"). The middle is as far as either side's
    // estimate can be trusted.
    if let (Some((prev_end, this_start)), Some(m)) = (words.start, new_in.as_mut()) {
        m.0 = m.0.max(rules.ceil_frame((prev_end + this_start) / 2.0)).min(in_s.max(m.0));
    }
    if let (Some((this_end, next_start)), Some(m)) = (words.end, new_out.as_mut()) {
        m.0 = m.0.min(rules.floor_frame((this_end + next_start) / 2.0)).max(out_s.min(m.0));
    }
    let (a, b) = (new_in.map_or(in_s, |m| m.0), new_out.map_or(out_s, |m| m.0));
    if b - a < MIN_KEPT_S {
        return None;
    }
    (new_in.is_some() || new_out.is_some()).then_some(RangeMove { in_s: a, out_s: b, new_in, new_out })
}

/// Where `in_s..out_s` of a video would be placed on its measured speech, when that is not where it
/// is: what a check offers as the fix for a cut it has flagged.
pub fn placed_range(db: &Db, video_id: i64, in_s: f64, out_s: f64, cfg: &ScriptConfig, fps: f64) -> Option<(f64, f64)> {
    place_range(db, video_id, in_s, out_s, Default::default(), cfg, fps).map(|m| (m.in_s, m.out_s))
}

struct RangeMove {
    in_s: f64,
    out_s: f64,
    new_in: Option<(f64, CutKind)>,
    new_out: Option<(f64, CutKind)>,
}

/// Place every speaking clip and audio bed of `script` on the sound. Returns what moved.
///
/// Clips of a beat that has a bed play silent under it, muted clips and clips with no tracks have
/// no voice to place, and a video the speech stage has not measured keeps the transcript's times.
/// A closing hold this close to the end of its clip continues that shot (s): about a frame, the
/// rounding the passes before leave between the two.
const HOLD_JOIN_S: f64 = 0.05;

pub fn place_cuts(db: &Db, script: &mut Script, cfg: &ScriptConfig) -> Vec<CutMove> {
    let mut moves = Vec::new();
    if !cfg.cut_to_speech {
        return moves;
    }
    let fps = script.fps.map(|f| f.as_f64()).filter(|f| *f > 0.0).unwrap_or(25.0);
    let several_beats = script.beats.len() > 1;
    for (bi, beat) in script.beats.iter_mut().enumerate() {
        let pictures: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
        let has_bed = beat.bed.is_some();
        let note =
            |clip_index, kind, from_s, to_s| CutMove { beat_index: bi, clip_index, kind, from_s, to_s, several_beats };
        for ci in 0..beat.clips.len() {
            let clip = beat.clips[ci].clone();
            // A hold after a speaking clip is quiet, whatever else it is.
            if clip.why.as_deref() == Some(crate::chat::CLOSING_PICTURE) {
                if clip.audio == Audio::Source {
                    beat.clips[ci].audio = Audio::Mute;
                    moves.push(note(Some(ci.saturating_sub(1)), CutKind::HoldMuted, clip.in_s, clip.out_s));
                }
                continue;
            }
            // Written to be kept exactly as it is.
            if clip.exact || has_bed || clip.audio != Audio::Source || clip.tracks.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            let words = crate::sentences::WordEdges::of(db, clip.sentences.as_ref());
            let Some(m) = place_range(db, clip.video_id, clip.in_s, clip.out_s, words, cfg, fps) else { continue };
            if let Some((to, kind)) = m.new_in.filter(|(to, _)| (to - clip.in_s).abs() >= NOTE_MIN_S) {
                moves.push(note(Some(ci), kind, clip.in_s, to));
            }
            if let Some((to, kind)) = m.new_out.filter(|(to, _)| (to - clip.out_s).abs() >= NOTE_MIN_S) {
                moves.push(note(Some(ci), kind, clip.out_s, to));
            }
            beat.clips[ci].in_s = m.in_s;
            beat.clips[ci].out_s = m.out_s;
            // A closing hold continues the shot this clip ends: it follows the cut.
            let delta = m.out_s - clip.out_s;
            if let Some(hold) = beat.clips.get_mut(ci + 1)
                && hold.why.as_deref() == Some(crate::chat::CLOSING_PICTURE)
                && hold.video_id == clip.video_id
                && (hold.in_s - clip.out_s).abs() < HOLD_JOIN_S
                && delta.abs() > 1e-9
            {
                let len = hold.out_s - hold.in_s;
                hold.in_s = m.out_s;
                hold.out_s = m.out_s + len;
            }
        }
        if let Some(bed) = &mut beat.bed
            && !bed.tracks.as_ref().is_some_and(Vec::is_empty)
            && let Some(m) = place_range(
                db,
                bed.video_id,
                bed.in_s,
                bed.out_s,
                crate::sentences::WordEdges::of(db, bed.sentences.as_ref()),
                cfg,
                fps,
            )
        {
            // A bed never outlasts the pictures it plays under.
            let out_s = if pictures > 0.0 && m.out_s - m.in_s > pictures { m.in_s + pictures } else { m.out_s };
            if let Some((to, kind)) = m.new_in.filter(|(to, _)| (to - bed.in_s).abs() >= NOTE_MIN_S) {
                moves.push(note(None, kind, bed.in_s, to));
            }
            if let Some((to, kind)) = m.new_out.filter(|(to, _)| (to - bed.out_s).abs() >= NOTE_MIN_S) {
                moves.push(note(None, kind, bed.out_s, to));
            }
            bed.in_s = m.in_s;
            bed.out_s = out_s;
        }
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;
    use crate::script::{Beat, Fps, ScriptClip};

    fn rules() -> Rules {
        Rules { tail: 0.35, guard: 0.12, lead: 0.15, max_move: 0.8, fps: 30.0, duration: 300.0 }
    }

    fn region(start_s: f64, end_s: f64, quiet_after_s: Option<f64>) -> SpeechRegion {
        SpeechRegion { start_s, end_s, quiet_after_s }
    }

    /// The handover from the benchmark cut: speaker A is voiced until 12.06 s and quiet to 12.84;
    /// speaker B's first word is at 12.86 (-52 dB), 12.88 (-26 dB), 12.92 (-12 dB).
    fn handover() -> Vec<SpeechRegion> {
        vec![region(0.30, 12.06, Some(12.48)), region(12.88, 16.30, None)]
    }

    /// What the renderer plays of `in_s..out_s` at `fps`: the duration snaps to whole frames.
    fn rendered_end(in_s: f64, out_s: f64, fps: f64) -> f64 {
        in_s + ((out_s - in_s) * fps).round() / fps
    }

    fn on_grid(t: f64, fps: f64) -> bool {
        (t * fps - (t * fps).round()).abs() < 1e-6
    }

    #[test]
    fn the_end_comes_off_the_next_speakers_first_word() {
        let (out, kind) = place_out(&handover(), 0.0, 12.86, None, &rules()).unwrap();
        assert_eq!(kind, CutKind::EndOffNextSpeaker);
        assert!((12.30..=12.75).contains(&out), "{out}");
        assert!(on_grid(out, 30.0));
        // 12.06 + the 0.35 s tail, rounded down to a frame.
        assert!((out - 12.4).abs() < 1e-9, "{out}");
        // What the renderer plays still ends well before B's first sound, at 12.86 on the transcript
        // and 12.88 on the audio.
        assert!(12.86 - rendered_end(0.0, out, 30.0) >= 0.12);
    }

    #[test]
    fn an_end_inside_the_next_word_is_pulled_back_and_one_inside_the_last_word_is_pushed_out() {
        let r = rules();
        // Well into B's first word, 0.1 s after the onset: still B's attack, not B's clip.
        let (out, kind) = place_out(&handover(), 0.0, 12.98, None, &r).unwrap();
        assert_eq!((kind, out), (CutKind::EndOffNextSpeaker, 12.4));
        // Cutting into A's last word (voiced to 12.06).
        let (out, kind) = place_out(&handover(), 0.0, 11.9, None, &r).unwrap();
        assert_eq!((kind, out), (CutKind::EndLetWordFinish, 12.4));
        // Already there: nothing to do.
        assert_eq!(place_out(&handover(), 0.0, 12.4, None, &r), None);
    }

    #[test]
    fn the_tail_stops_at_the_guard_when_the_pause_is_short() {
        // A ends 3.00, B comes in at 3.30: the 0.35 s tail would reach 3.35; the guard stops it at
        // 3.18, which is 3.1667 on a 30 fps grid.
        let regions = vec![region(1.0, 3.0, Some(3.12)), region(3.30, 6.0, None)];
        let (out, _) = place_out(&regions, 0.0, 3.30, None, &rules()).unwrap();
        assert!((out - 3.1667).abs() < 1e-3, "{out}");
        assert!(3.30 - out >= 0.12);
    }

    #[test]
    fn a_pause_too_short_for_a_tail_and_a_guard_is_cut_at_its_quietest_frame() {
        // A ends 3.00, B comes in at 3.16: 0.14 s. The quietest 10 ms of it is at 3.07.
        let regions = vec![region(1.0, 3.0, Some(3.07)), region(3.16, 6.0, None)];
        let (out, kind) = place_out(&regions, 0.0, 3.16, None, &rules()).unwrap();
        assert_eq!(kind, CutKind::EndOffNextSpeaker);
        // 3.07 on a 30 fps grid is 3.0667; it is inside the pause.
        assert!((out - 3.0667).abs() < 1e-3, "{out}");
        assert!((3.0..3.16).contains(&out));
        // Pauses of 60 ms: the frame below the quiet one would be under the end of the word, so it
        // takes the frame above if the pause holds one, else the quiet moment itself.
        let tight = vec![region(1.0, 3.0, Some(3.03)), region(3.06, 6.0, None)];
        let (out, _) = place_out(&tight, 0.0, 3.06, None, &rules()).unwrap();
        assert!((3.0..3.06).contains(&out), "{out}");
        // No quiet frame stored: the middle of the pause.
        let plain = vec![region(1.0, 3.0, None), region(3.16, 6.0, None)];
        let (out, _) = place_out(&plain, 0.0, 3.16, None, &rules()).unwrap();
        assert!(out > 3.0 && out < 3.16, "{out}");
    }

    #[test]
    fn an_end_that_would_change_the_sentence_is_left_alone() {
        let r = rules();
        // Deep in B's speech (voiced to 16.30): finishing it is 2.7 s away, beyond the reach.
        assert_eq!(place_out(&handover(), 0.0, 13.6, None, &r), None);
        // No speech in the range at all: a cutaway.
        assert_eq!(place_out(&handover(), 17.0, 18.0, None, &r), None);
        // The reach is the configured one.
        let tight = Rules { max_move: 0.3, ..r };
        assert_eq!(place_out(&handover(), 0.0, 12.86, None, &tight), None);
    }

    #[test]
    fn running_on_never_reaches_into_the_next_sentence_the_transcript_knows_of() {
        // One voiced stretch (the speaker runs on without a pause) of two sentences; the cut ends
        // the first at 5.0 and the second starts at 5.0 on the transcript.
        let regions = vec![region(1.0, 5.4, None)];
        let limit = Some(5.0 - 0.12);
        assert_eq!(place_out(&regions, 0.0, 5.0, limit, &rules()), None, "the next sentence is 0.4 s long");
        // Without the transcript's say, the audio alone would finish the stretch.
        assert!(place_out(&regions, 0.0, 5.0, None, &rules()).is_some());
    }

    #[test]
    fn the_start_comes_in_just_before_the_first_word_and_never_inside_the_previous_speech() {
        let r = rules();
        // B's clip, opening on the transcript's 12.86: the first sound is at 12.88.
        let (start, kind) = place_in(&handover(), 12.86, 16.0, None, &r).unwrap();
        assert_eq!(kind, CutKind::StartRetimed);
        // 12.88 - 0.15 = 12.73, up to the frame at 12.7333.
        assert!((start - 12.7333).abs() < 1e-3 && on_grid(start, 30.0), "{start}");
        assert!(start > 12.06 + IN_MARGIN_S);
        // Opening inside A's decaying last word: moved off it, to B.
        let (start, kind) = place_in(&handover(), 12.0, 16.0, None, &r).unwrap();
        assert_eq!(kind, CutKind::StartOffPreviousSpeaker);
        assert!((start - 12.7333).abs() < 1e-3, "{start}");
        // Opening just after B's attack began (Whisper was late): back to before it.
        let (start, kind) = place_in(&handover(), 13.0, 16.0, None, &r).unwrap();
        assert_eq!(kind, CutKind::StartBeforeFirstWord);
        assert!((start - 12.7333).abs() < 1e-3, "{start}");
        // Opening in the middle of B's phrase is on purpose.
        assert_eq!(place_in(&handover(), 14.0, 16.0, None, &r), None);
        // The first voice of a video, nobody before it: the lead, from the start of the file.
        let (start, _) = place_in(&handover(), 0.0, 8.0, None, &r).unwrap();
        assert!((start - 0.1667).abs() < 1e-3, "{start}");
    }

    #[test]
    fn a_lead_never_reaches_back_into_the_previous_sentence() {
        // "…¿ok? En API…" said in one breath: the sound is one region from 10.0 s, the transcript
        // ends the first sentence at 10.42 and the clip opens on the second at 10.46. The lead
        // would open at 10.31, on the "ok"; the previous sentence's end stops it.
        let regions = [SpeechRegion::new(10.0, 14.0)];
        let r = rules();
        let free = place_in(&regions, 10.2, 14.0, None, &r).map(|m| m.0);
        assert!(free.is_some_and(|t| t < 10.2), "{free:?}");
        let held = place_in(&regions, 10.2, 14.0, Some(10.18), &r).map(|m| m.0);
        assert!(held.is_none_or(|t| t >= 10.18 - 1e-9), "{held:?}");
    }

    #[test]
    fn the_start_stays_after_the_previous_speech_when_the_pause_is_short() {
        // A ends 12.00; B comes in 12.20. 12.20 - 0.15 = 12.05 is only 0.05 after A.
        let regions = vec![region(0.3, 12.0, Some(12.1)), region(12.20, 16.0, None)];
        let (start, _) = place_in(&regions, 12.20, 16.0, None, &rules()).unwrap();
        assert!((12.0 + IN_MARGIN_S - 1e-9..=12.20 - MIN_LEAD_S + 1e-9).contains(&start), "{start}");
        // 80 ms between them: no room for a margin and a lead; the quietest frame of the pause.
        let regions = vec![region(0.3, 12.0, Some(12.04)), region(12.08, 16.0, None)];
        let (start, _) = place_in(&regions, 12.08, 16.0, None, &rules()).unwrap();
        assert!((12.0..=12.08).contains(&start), "{start}");
    }

    #[test]
    fn frames_follow_the_cuts_frame_rate() {
        // NTSC: every placed edge is a whole number of 1001/30000 s frames.
        let ntsc = Rules { fps: 30000.0 / 1001.0, ..rules() };
        let (out, _) = place_out(&handover(), 0.0, 12.86, None, &ntsc).unwrap();
        let (start, _) = place_in(&handover(), 12.86, 16.0, None, &ntsc).unwrap();
        for t in [out, start] {
            assert!(on_grid(t, ntsc.fps), "{t}");
        }
        assert!(out <= 12.06 + 0.35 + 1e-9 && out > 12.06 + 0.35 - 1.001 / 30.0);
        assert!((12.88 - 0.15 - 1e-9..12.88 - 0.15 + 1.001 / 29.97).contains(&start));
    }

    // ---- the same sound, judged ------------------------------------------------------------

    #[test]
    fn a_cut_in_the_silence_after_the_word_is_clean_and_the_placement_always_is() {
        let r = rules();
        // A is voiced to 12.06 and quiet to 12.88. 12.4 is where placement puts the cut.
        let end = judge_end(&handover(), 0.0, 12.4, &r);
        assert_eq!(end.flag, EndFlag::Clean);
        assert!((end.after_voice_s.unwrap() - 0.34).abs() < 1e-9);
        assert!(end.detail().starts_with("clean (in silence)"), "{}", end.detail());
        // Whatever place_out returns is clean, for any starting point it moves.
        for out in [11.9, 12.0, 12.2, 12.86, 12.98] {
            let (placed, _) = place_out(&handover(), 0.0, out, None, &r).unwrap();
            assert_eq!(judge_end(&handover(), 0.0, placed, &r).flag, EndFlag::Clean, "{out} -> {placed}");
        }
        // A short pause: the quiet frame the placement picks is clean as well.
        let tight = vec![region(1.0, 3.0, Some(3.07)), region(3.16, 6.0, None)];
        let (placed, _) = place_out(&tight, 0.0, 3.16, None, &r).unwrap();
        assert_eq!(judge_end(&tight, 0.0, placed, &r).flag, EndFlag::Clean);
    }

    #[test]
    fn a_clipped_word_and_a_leak_are_told_apart() {
        let r = rules();
        let clipped = judge_end(&handover(), 0.0, 11.9, &r);
        assert_eq!(clipped.flag, EndFlag::ClipsLastWord);
        assert_eq!(clipped.flag.label(), "clips the last word");
        assert!(clipped.detail().contains("0.16 s past the cut"), "{}", clipped.detail());
        // Right at the end of the word: no tail left.
        assert_eq!(judge_end(&handover(), 0.0, 12.07, &r).flag, EndFlag::ClipsLastWord);
        // The cut on B's first sound, and inside the guard before it.
        let leak = judge_end(&handover(), 0.0, 12.86, &r);
        assert_eq!(leak.flag, EndFlag::NextSpeakerLeaksIn);
        assert!(leak.detail().contains("next voice starts 0.02 s after the cut"), "{}", leak.detail());
        assert_eq!(judge_end(&handover(), 0.0, 12.80, &r).flag, EndFlag::NextSpeakerLeaksIn);
        // Inside B's word: the range holds A's speech, so B is the one leaking.
        assert_eq!(judge_end(&handover(), 0.0, 13.0, &r).flag, EndFlag::NextSpeakerLeaksIn);
        // No speech at all in the range: nothing to clip.
        assert_eq!(judge_end(&handover(), 17.0, 18.0, &r).flag, EndFlag::Clean);
    }

    #[test]
    fn a_start_is_clean_before_the_word_and_flagged_inside_it() {
        let r = rules();
        // B opens at 12.88; 12.73 is the placement's own lead.
        assert_eq!(judge_start(&handover(), 12.7333, 16.0, &r).flag, StartFlag::Clean);
        let late = judge_start(&handover(), 13.0, 16.0, &r);
        assert_eq!(late.flag, StartFlag::StartsMidWord);
        assert!(late.detail().contains("first word began 0.12 s before the cut"), "{}", late.detail());
        // Inside A's decaying last word (voiced to 12.06).
        let inside_a = judge_start(&handover(), 12.0, 16.0, &r);
        assert_eq!(inside_a.flag, StartFlag::StartsMidWord);
        assert!(inside_a.detail().contains("previous speaker"), "{}", inside_a.detail());
        // Deep in a phrase is a choice, not a flag.
        assert_eq!(judge_start(&handover(), 14.0, 16.0, &r).flag, StartFlag::Clean);
        assert_eq!(StartFlag::StartsMidWord.label(), "starts mid-word");
    }

    #[test]
    fn judging_needs_measured_speech() {
        let (db, _) = db_with_handover();
        let cfg = ScriptConfig::default();
        let (start, end) = judge_range(&db, VIDEO, 12.73, 16.6, Default::default(), &cfg, 30.0).unwrap();
        assert_eq!((start.flag, end.flag), (StartFlag::Clean, EndFlag::Clean));
        let (_, end) = judge_range(&db, VIDEO, 0.0, 12.86, Default::default(), &cfg, 30.0).unwrap();
        assert_eq!(end.flag, EndFlag::NextSpeakerLeaksIn);
        db.conn.execute("DELETE FROM speech_regions", []).unwrap();
        assert!(
            judge_range(&db, VIDEO, 0.0, 12.86, Default::default(), &cfg, 30.0).is_none(),
            "unmeasured: the transcript's rule applies"
        );
    }

    // ---- through the database and the repair chain ------------------------------------------

    const VIDEO: i64 = 104;

    fn db_with_handover() -> (Db, i64) {
        let db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h104', 1, 300.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (104, 0.00, 12.02, 'speaker A'), (104, 12.86, 16.46, 'Esto es O Corres o Te Encaramas')",
                [],
            )
            .unwrap();
        db.set_speech_regions(VIDEO, &handover()).unwrap();
        (db, project.id)
    }

    fn clip(in_s: f64, out_s: f64) -> ScriptClip {
        ScriptClip { exact: false,
            sentences: None,
            video_id: VIDEO,
            in_s,
            out_s,
            audio: Audio::Source,
            why: None,
            tracks: None,
            framing: None,
        }
    }

    fn script(clips: Vec<ScriptClip>) -> Script {
        Script {
            title: "t".into(),
            target_duration_s: None,
            fps: Some(Fps::new(30, 1)),
            width: None,
            height: None,
            framing: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "p".into(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                clips,
                bed: None,
                notes: None,
                card: None,
            }],
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            card_style: None,
        }
    }

    #[test]
    fn the_boundary_between_two_words_is_the_quiet_between_them() {
        // -20 dB speech with a 40 ms dip to -50 dB at 1.30 s, and one quiet frame inside a word at
        // 1.10 s (a stop consonant): the averaged dip wins over the lone frame.
        let mut levels = vec![-20.0f32; 200];
        levels[110] = -60.0;
        for l in &mut levels[129..133] {
            *l = -50.0;
        }
        let t = quietest(&levels, 0.0, 1.0, 1.6).unwrap();
        assert!((t - 1.30).abs() <= 0.02, "{t}");
        assert_eq!(quietest(&levels, 0.0, 3.0, 4.0), None, "outside the levels");
    }

    #[test]
    fn a_clip_written_to_a_word_inside_running_speech_stops_at_that_word() {
        // B3's `to: "s160@13"` played to the end of the sentence: the speech runs on without a
        // pause, and the end was let out to the next one. Words: w1 20.0-20.4, w2 20.45-20.9,
        // w3 20.95-21.4, w4 21.45-22.0, all in one stretch of speech 20.0-26.0.
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 300.0)", [])
            .unwrap();
        // Measured speech first: storing it clears the sentences built on the old one.
        db.set_speech_regions(VIDEO, &[SpeechRegion::new(20.0, 26.0)]).unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text, words_json) VALUES
                   (1, 104, 20.0, 26.0, 'uno dos tres cuatro cinco',
                    '[[20.0,20.4],[20.45,20.9],[20.95,21.4],[21.45,22.0],[22.05,26.0]]');",
            )
            .unwrap();
        let mut c = clip(20.0, 21.4);
        c.sentences = Some(crate::script::SentenceRange { from: "s1".into(), to: "s1@3".into() });
        let mut s = script(vec![c]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        let out = s.beats[0].clips[0].out_s;
        assert!(out >= 21.4 - 1e-9 && out <= 21.45, "ends after word 3, before word 4: {out}");
        // And the check says it ends where it was written to, not that it clips a word.
        let c = &s.beats[0].clips[0];
        let words = crate::sentences::WordEdges::of(&db, c.sentences.as_ref());
        let (_, end) = judge_range(&db, VIDEO, c.in_s, c.out_s, words, &ScriptConfig::default(), 30.0).unwrap();
        assert_eq!(end.flag, EndFlag::BetweenWords);

        // From word 3: a lead-in into the gap after word 2, not into word 2.
        let mut c = clip(20.95, 22.0);
        c.sentences = Some(crate::script::SentenceRange { from: "s1@3".into(), to: "s1@4".into() });
        let mut s = script(vec![c]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        let start = s.beats[0].clips[0].in_s;
        assert!(start >= 20.9 && start < 20.95, "opens between word 2 and word 3: {start}");
    }

    #[test]
    fn a_saved_cut_ending_on_the_next_speakers_first_word_is_repaired_and_reported() {
        let (db, _) = db_with_handover();
        let mut s = script(vec![clip(0.0, 12.86)]);
        let moves = place_cuts(&db, &mut s, &ScriptConfig::default());
        let c = &s.beats[0].clips[0];
        assert!((12.30..=12.75).contains(&c.out_s), "{}", c.out_s);
        assert!(12.86 - rendered_end(c.in_s, c.out_s, 30.0) >= 0.12);
        assert!(on_grid(c.in_s, 30.0) && on_grid(c.out_s, 30.0));
        let messages: Vec<String> = moves.iter().map(CutMove::message).collect();
        assert!(
            messages.iter().any(|m| m.starts_with("moved the end of clip 1 off the next speaker's first word")),
            "{messages:?}"
        );
    }

    #[test]
    fn an_exact_clip_keeps_its_bounds_while_the_others_are_placed() {
        let (db, _) = db_with_handover();
        let mut kept = clip(0.0, 12.86);
        kept.exact = true;
        let mut s = script(vec![kept, clip(0.0, 12.86)]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        assert_eq!(s.beats[0].clips[0].out_s, 12.86, "exact: left as written");
        assert!(s.beats[0].clips[1].out_s < 12.8, "the other clip is still placed");
        // And it says so in the JSON only when set.
        let json = serde_json::to_value(&s.beats[0].clips[1]).unwrap();
        assert!(json.get("exact").is_none());
        let back: ScriptClip = serde_json::from_value(serde_json::json!({"video_id": 104, "in_s": 1.0, "out_s": 2.0, "exact": true})).unwrap();
        assert!(back.exact);
    }

    #[test]
    fn a_start_on_a_word_never_reaches_past_the_middle_of_the_gap_before_it() {
        let (db, _) = db_with_handover();
        // Inside the second speaker's running speech, written to open on a word that starts at
        // 13.50 after one that ends at 13.20; a long lead would reach back to 13.22.
        let cfg = ScriptConfig { cut_lead_s: 0.4, ..ScriptConfig::default() };
        let words = crate::sentences::WordEdges { start: Some((13.20, 13.50)), ..Default::default() };
        let m = place_range(&db, VIDEO, 13.50, 16.20, words, &cfg, 30.0).expect("placed");
        assert!(m.in_s >= 13.35 - 1e-9, "{}", m.in_s);
        assert!(m.in_s <= 13.50, "{}", m.in_s);
    }

    #[test]
    fn dead_air_before_the_first_word_is_flagged_with_a_trim() {
        let (db, pid) = db_with_handover();
        // After A stops (12.06) and 0.68 s before B starts (12.88).
        let s = script(vec![clip(12.20, 16.60)]);
        let issues =
            crate::script_check::check(&db, pid, &s, &ScriptConfig::default(), &Default::default()).unwrap();
        let dead = issues.iter().find(|i| i.message.contains("dead air")).expect("flagged");
        assert!(dead.message.contains("before the first word"), "{}", dead.message);
        let crate::script::FixAction::Trim { in_s, .. } = dead.fixes[0].action else { panic!("{dead:?}") };
        assert!((in_s - (12.88 - 0.15)).abs() < 0.02, "{in_s}");
    }

    #[test]
    fn the_cut_is_left_alone_without_measured_speech_or_when_turned_off() {
        let (db, _) = db_with_handover();
        db.conn.execute("DELETE FROM speech_regions", []).unwrap();
        let mut s = script(vec![clip(0.0, 12.86)]);
        assert!(place_cuts(&db, &mut s, &ScriptConfig::default()).is_empty());
        assert_eq!(s.beats[0].clips[0].out_s, 12.86);

        db.set_speech_regions(VIDEO, &handover()).unwrap();
        let off = ScriptConfig { cut_to_speech: false, ..ScriptConfig::default() };
        assert!(place_cuts(&db, &mut s, &off).is_empty());
        assert_eq!(s.beats[0].clips[0].out_s, 12.86);
    }

    #[test]
    fn clips_with_no_voice_to_place_are_not_touched() {
        let (db, _) = db_with_handover();
        let mut muted = clip(0.0, 12.86);
        muted.audio = Audio::Mute;
        let mut silent = clip(0.0, 12.86);
        silent.tracks = Some(vec![]);
        let mut s = script(vec![muted, silent]);
        // A beat with a bed plays its pictures silent under it.
        let mut bedded = script(vec![clip(0.0, 12.86)]);
        bedded.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: 999,
            in_s: 0.0,
            out_s: 12.86,
            why: None,
            inferred: false,
            tracks: None,
        });
        assert!(place_cuts(&db, &mut s, &ScriptConfig::default()).is_empty());
        assert!(place_cuts(&db, &mut bedded, &ScriptConfig::default()).is_empty());
        assert_eq!(s.beats[0].clips[0].out_s, 12.86);
        assert_eq!(bedded.beats[0].clips[0].out_s, 12.86);
    }

    #[test]
    fn an_audio_bed_is_placed_like_a_clip_and_stays_inside_its_pictures() {
        let (db, _) = db_with_handover();
        // Pictures of another video, 13 s; the voice under them is A's answer, ending on B.
        let mut s = script(vec![ScriptClip { sentences: None, video_id: 7, ..clip(0.0, 13.0) }]);
        s.beats[0].clips[0].audio = Audio::Mute;
        s.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: VIDEO,
            in_s: 0.0,
            out_s: 12.86,
            why: None,
            inferred: true,
            tracks: None,
        });
        let moves = place_cuts(&db, &mut s, &ScriptConfig::default());
        let bed = s.beats[0].bed.as_ref().unwrap();
        assert!((12.30..=12.75).contains(&bed.out_s), "{}", bed.out_s);
        assert!(moves.iter().any(|m| m.clip_index.is_none() && m.kind == CutKind::EndOffNextSpeaker));
        // The pictures are shorter than the voice would like: the bed is capped to them.
        let mut short = s.clone();
        short.beats[0].clips[0].out_s = 12.0;
        short.beats[0].bed.as_mut().unwrap().out_s = 12.0;
        short.beats[0].bed.as_mut().unwrap().in_s = 0.0;
        place_cuts(&db, &mut short, &ScriptConfig::default());
        let bed = short.beats[0].bed.as_ref().unwrap();
        assert!(bed.out_s - bed.in_s <= 12.0 + 1e-9, "{bed:?}");
    }

    #[test]
    fn the_closing_hold_follows_the_cut_and_is_never_sound() {
        let (db, _) = db_with_handover();
        let hold = |audio| ScriptClip { exact: false,
            sentences: None,
            video_id: VIDEO,
            in_s: 12.86,
            out_s: 14.86,
            audio,
            why: Some(crate::chat::CLOSING_PICTURE.into()),
            tracks: None,
            framing: None,
        };
        let mut s = script(vec![clip(0.0, 12.86), hold(Audio::Mute)]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        let (c, h) = (&s.beats[0].clips[0], &s.beats[0].clips[1]);
        assert_eq!(h.audio, Audio::Mute);
        assert!((h.in_s - c.out_s).abs() < 1e-9, "the hold continues the shot: {} vs {}", h.in_s, c.out_s);
        assert!((h.out_s - h.in_s - 2.0).abs() < 1e-9, "and keeps its length");

        // A hold that carried sound is muted, and the note says so.
        let mut loud = script(vec![clip(0.0, 12.4), hold(Audio::Source)]);
        let moves = place_cuts(&db, &mut loud, &ScriptConfig::default());
        assert_eq!(loud.beats[0].clips[1].audio, Audio::Mute);
        assert!(moves.iter().any(|m| m.kind == CutKind::HoldMuted));
    }

    #[test]
    fn repairing_an_already_repaired_script_changes_nothing() {
        let (db, project_id) = db_with_handover();
        let cfg = ScriptConfig::default();
        let mut s = script(vec![clip(0.0, 12.86)]);
        let first = crate::script_check::repair_for_import(&db, project_id, &mut s, &cfg, None).unwrap();
        // The saved cut now: out off B's first word, a muted hold continuing the shot.
        let c = s.beats[0].clips[0].clone();
        assert!((12.30..=12.75).contains(&c.out_s), "{}", c.out_s);
        assert!(12.86 - rendered_end(c.in_s, c.out_s, 30.0) >= 0.12);
        let hold = s.beats[0].clips.get(1).expect("the closing hold");
        assert_eq!(hold.audio, Audio::Mute);
        assert!((hold.in_s - c.out_s).abs() < 1e-9);
        assert!(first.iter().any(|n| n.contains("off the next speaker's first word")), "{first:?}");

        let once = s.clone();
        for round in 0..3 {
            let notes = crate::script_check::repair_for_import(&db, project_id, &mut s, &cfg, None).unwrap();
            assert_eq!(s, once, "round {round} drifted: {notes:?}");
            assert!(notes.is_empty(), "round {round} reported {notes:?}");
        }
    }

    #[test]
    fn the_chat_chain_places_cuts_last_and_is_stable_too() {
        let (db, project_id) = db_with_handover();
        let cfg = ScriptConfig::default();
        let mut s = script(vec![clip(0.0, 12.86), clip(12.86, 16.2)]);
        let issues = crate::chat::repair::finish_script(&db, project_id, &mut s, false, &cfg).unwrap().unwrap();
        let first = &s.beats[0].clips[0];
        assert!(12.86 - rendered_end(first.in_s, first.out_s, 30.0) >= 0.12, "{first:?}");
        let second = &s.beats[0].clips[1];
        assert!(second.in_s >= first.out_s - 1e-9 && second.in_s < 12.88, "{second:?}");
        assert!(issues.iter().any(|i| i.message.contains("first word")), "{issues:?}");
        let once = s.clone();
        crate::chat::repair::finish_script(&db, project_id, &mut s, false, &cfg).unwrap().unwrap();
        assert_eq!(s, once, "a second repair moved the cut");
    }
}
