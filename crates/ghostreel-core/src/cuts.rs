//! Cut placement on the sound: where a speaking clip opens and closes, judged from where speech
//! actually is (`speech.rs`) instead of from the transcript's segment times.
//!
//! Whisper's times are approximate: often late at a start, early or quantised at an end. A clip
//! that ends on a segment boundary can end exactly as the next speaker's first word begins, and
//! you hear its attack. This is the last pass of the finishing chain (`finish::run`), after
//! everything that moves a clip has moved it, and it only runs for videos the speech stage has
//! measured.
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
use crate::script::{Audio, CLOSING_PICTURE, Script, on_word};
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
    /// The clip opened on a reply or a filler said on its own breath ("yeah so…"); it starts on
    /// the words after it now.
    StartAfterFiller,
    /// The clip opened on someone off the microphone (the interviewer); it starts after them now.
    StartAfterOffMic,
    /// The clip ended on someone off the microphone; it ends before them now.
    EndBeforeOffMic,
    /// The clip ended on a sound no word accounts for (an "umm", a laugh); it ends on the words.
    EndBeforeUnworded,
    /// The clip opened on a sound no word accounts for; it opens on the words.
    StartAfterUnworded,
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
            CutKind::StartAfterOffMic => {
                format!("started {subject} after the off-mic line it opened on ({moved})")
            }
            CutKind::EndBeforeOffMic => format!("ended {subject} before the off-mic line it closed on ({moved})"),
            CutKind::EndBeforeUnworded => {
                format!("ended {subject} on its last word, before an \"umm\" or other sound no word accounts for ({moved})")
            }
            CutKind::StartAfterUnworded => {
                format!("opened {subject} on its first word, after an \"umm\" or other sound no word accounts for ({moved})")
            }
            CutKind::StartAfterFiller => {
                format!("started {subject} after its opening reply or filler, said on its own breath ({moved})")
            }
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
    // Nor is a voice that starts after the next sentence does (`push_limit`): it is that sentence.
    let li = regions.iter().rposition(|x| {
        x.start_s < out_s - r.guard && x.end_s > in_s && push_limit.is_none_or(|l| x.start_s < l + r.guard)
    })?;
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
    // The transcript's limit is for a voice the sound cannot split (one region running from this
    // sentence into the next). When the measured speech of this sentence ends before the next one
    // starts, the pause is right there and the target was already placed in it: capping it at the
    // transcript's start-minus-guard would cut inside the last word ("Viata" at 67.87, voice to 67.93,
    // next voice at 68.01) and leave a flag no editor can fix.
    let own_pause = push_limit.is_some_and(|l| last.end_s <= l + r.guard + 0.02);
    if target > out_s
        && !own_pause
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

/// The start of the next sentence after `out_s`, minus the guard. A sentence that starts inside
/// the last `reach` of the range counts as the next one: the passes before this one end a clip a
/// tail past its last sentence, and when the speaker runs straight on that tail is the next
/// sentence's first word, not this one's.
fn push_limit(db: &Db, video_id: i64, out_s: f64, guard: f64, reach: f64) -> Option<f64> {
    db.conn
        .query_row(
            "SELECT MIN(start_s) FROM transcript_sentences WHERE video_id = ?1 AND start_s > ?2",
            rusqlite::params![video_id, out_s - reach.max(0.05)],
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

/// The longest end of a sentence that runs on past an off-mic segment's edge (s): a word or two.
const MAX_CARRIED_TAIL_S: f64 = 1.5;

/// The part of `in_s..out_s` left once lines said off the microphone (the interviewer's question,
/// a crew member's "Okay.") are taken off its start and its end — the same rule a clip-run moment
/// follows (`clips::candidates` never lets one in). A line in the middle stays: taking it out would
/// leave a jump cut, so the check reports it instead. `None` when nothing changes, or when nothing
/// on the microphone would be left.
pub fn on_mic_span(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> Option<(f64, f64)> {
    let segs: Vec<(f64, f64, bool)> = db
        .conn
        .prepare(
            "SELECT start_s, end_s, COALESCE(off_mic, 0) FROM transcript_segments
              WHERE video_id = ?1 AND end_s > ?2 + 0.1 AND start_s < ?3 - 0.1 ORDER BY start_s",
        )
        .and_then(|mut st| {
            st.query_map(rusqlite::params![video_id, in_s, out_s], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect()
        })
        .ok()?;
    let first = segs.iter().position(|s| !s.2)?;
    let last = segs.iter().rposition(|s| !s.2)?;
    let a = if first > 0 { segs[first].0 } else { in_s };
    let mut b = if last + 1 < segs.len() { segs[last].1 } else { out_s };
    if b < out_s {
        // The speaker's sentence can finish a word or two into the next, off-mic segment, where
        // the transcript broke it ("…that's a real" / "thing. So this event…"): sentences are split
        // after that tail (`sentence_split::carry_tails`), so one running across this edge is the
        // speaker's own and ends where it ends.
        let tail: Option<f64> = db
            .conn
            .query_row(
                "SELECT MAX(end_s) FROM transcript_sentences
                  WHERE video_id = ?1 AND start_s < ?2 - 0.05 AND end_s > ?2 + 0.05 AND end_s <= ?2 + ?3",
                rusqlite::params![video_id, b, MAX_CARRIED_TAIL_S],
                |r| r.get(0),
            )
            .ok()
            .flatten();
        if let Some(end) = tail {
            b = end.min(out_s);
        }
    }
    ((a, b) != (in_s, out_s) && b - a >= 1.0).then_some((a, b))
}

/// Why [`spoken_span`] moved an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanCut {
    /// A line said off the microphone at that end.
    OffMic,
    /// A reply or filler said on its own breath before the sentence ([`after_opening_filler`]).
    Filler,
}

/// What [`spoken_span`] keeps of a range, and why each edge moved (`None`: it did not).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub in_s: f64,
    pub out_s: f64,
    pub start: Option<SpanCut>,
    pub end: Option<SpanCut>,
    /// Everything said in the range is off the microphone: nothing of it is anyone's answer.
    pub off_mic_only: bool,
}

/// The one rule for what a speaking range keeps: lines said off the microphone come off either end
/// ([`on_mic_span`]), and with `filler` (a clip, not a bed) a reply or filler said on its own breath
/// comes off the start. An edge written on a word (`s412@3`) is where the editor asked for it and
/// stays. The finishing pass (`finish::trim_unspoken_edges`) and the cut placement
/// ([`place_cuts`]) both ask this, so neither undoes the other.
pub fn spoken_span(
    db: &Db,
    video_id: i64,
    in_s: f64,
    out_s: f64,
    range: Option<&crate::script::SentenceRange>,
    filler: bool,
) -> Span {
    let (word_in, word_out) = (on_word(range, true), on_word(range, false));
    let trimmed = on_mic_span(db, video_id, in_s, out_s);
    let off_mic_only = trimmed.is_none() && says_only_off_mic(db, video_id, in_s, out_s);
    let (mut a, mut b) = trimmed.unwrap_or((in_s, out_s));
    if word_in {
        a = in_s;
    }
    if word_out {
        b = out_s;
    }
    let mut start = (a != in_s).then_some(SpanCut::OffMic);
    let end = (b != out_s).then_some(SpanCut::OffMic);
    if filler
        && !word_in
        && let Some(t) = after_opening_filler(db, video_id, a).filter(|t| *t > a && *t < b - 1.0)
    {
        a = t;
        start = Some(SpanCut::Filler);
    }
    Span { in_s: a, out_s: b, start, end, off_mic_only }
}

/// Whether `in_s..out_s` has transcribed speech and all of it is off the microphone.
fn says_only_off_mic(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT COUNT(*) > 0 AND SUM(COALESCE(off_mic, 0) = 0) = 0 FROM transcript_segments
              WHERE video_id = ?1 AND end_s > ?2 + 0.1 AND start_s < ?3 - 0.1",
            rusqlite::params![video_id, in_s, out_s],
            |r| r.get::<_, Option<bool>>(0),
        )
        .ok()
        .flatten()
        .unwrap_or(false)
}

/// Words a clip should not open on when they are said on their own breath: a reply to a question
/// nobody hears ("yeah", "exacto") or a filler before the thought ("so", "okay", "um").
use crate::lexicon::FILLERS;
/// At most this many such words are dropped.
const MAX_FILLERS: usize = 3;

/// When the sentence a range opens on starts with a reply or filler said on its own breath — a short
/// burst of speech, then a pause, then the rest ("yeah so · hi, my name is…") — where the rest
/// starts. Measured on the sound, so it needs no word timings; a filler run straight into the
/// sentence ("And Mark and Kayla…") has no pause to cut in and is left alone.
pub fn after_opening_filler(db: &Db, video_id: i64, in_s: f64) -> Option<f64> {
    let (start, text): (f64, String) = db
        .conn
        .query_row(
            "SELECT start_s, text FROM transcript_sentences WHERE video_id = ?1 AND end_s > ?2 + 0.05 AND start_s < ?2 + 0.8
              ORDER BY start_s LIMIT 1",
            rusqlite::params![video_id, in_s],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .ok()
        .flatten()?;
    // Only for a range that opens on the sentence's start (the passes before pad it a little
    // earlier), not one placed inside it on purpose.
    if in_s > start + 0.3 {
        return None;
    }
    let words: Vec<String> =
        text.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).collect();
    let n = words.iter().take(MAX_FILLERS).take_while(|w| FILLERS.contains(&w.as_str())).count();
    if n == 0 || words.len() < n + 3 {
        return None;
    }
    let regions = crate::speech::speech_regions(db, video_id, start - 0.3, start + 3.0).ok()?;
    let i = regions.iter().position(|r| r.end_s > start + 0.02)?;
    let (burst, rest) = (&regions[i], regions.get(i + 1)?);
    let pause = rest.start_s - burst.end_s;
    let short = burst.end_s - burst.start_s <= 0.25 + 0.35 * n as f64;
    (burst.start_s <= start + 0.4 && short && (0.08..=1.0).contains(&pause)).then_some(rest.start_s)
}

/// Where a range's words end and start when its edges hold measured speech that no sentence
/// covers: the "umm", "uh" or laugh Whisper does not write, which the sentence split leaves between
/// sentences ([`crate::sentence_split`]). `(start, end)`, each `None` where that edge is on words.
/// A range that holds no sentence at all is left alone: it was chosen for its sound.
fn unworded_edges(db: &Db, video_id: i64, in_s: f64, out_s: f64, regions: &[SpeechRegion]) -> (Option<f64>, Option<f64>) {
    let sentences: Vec<(f64, f64)> = crate::sentence_split::edges(db, video_id)
        .into_iter()
        .filter(|&(a, b)| b > in_s + 0.05 && a < out_s - 0.05)
        .collect();
    let (Some(first), Some(last)) = (sentences.first().copied(), sentences.last().copied()) else {
        return (None, None);
    };
    let worded = |r: &SpeechRegion| sentences.iter().any(|&(a, b)| a < r.end_s - 0.02 && b > r.start_s + 0.02);
    // Voiced sound after the last sentence ends, inside the range, that no sentence covers.
    let tail = last.1 < out_s
        && regions.iter().any(|r| r.start_s >= last.1 && r.start_s < out_s - 0.05 && !worded(r));
    let head = first.0 > in_s
        && regions.iter().any(|r| r.end_s <= first.0 && r.end_s > in_s + 0.05 && !worded(r));
    (head.then_some(first.0), tail.then_some(last.1))
}

/// How long the speaker stays quiet after the last voice a range holds, before they talk again
/// (s): from the end of the last measured speech that starts inside it to the next. `None` when
/// the video is not measured or nobody speaks again nearby. It does not depend on where the out
/// point sits in that pause, so a check made before cut placement agrees with one made after.
pub fn quiet_after_voice(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> Option<f64> {
    let regions = crate::speech::speech_regions(db, video_id, in_s, out_s + 10.0).ok()?;
    let li = regions.iter().rposition(|r| r.start_s < out_s - 0.05 && r.end_s > in_s)?;
    let next = regions.get(li + 1)?;
    Some(next.start_s - regions[li].end_s)
}

/// How far before the speaker's next sound a same-shot closing hold stops (s).
const HOLD_QUIET_MARGIN_S: f64 = 0.15;
/// A closing hold shorter than this is a flash; it goes (s).
const MIN_HOLD_S: f64 = 0.2;

/// Stop a closing hold that continues the last shot before its speaker talks again: the hold is
/// muted, and a face seen talking with the sound gone makes the cut seem to go mute before it ends.
/// Run on the placed cut, so the same cut always gets the same hold.
fn hold_until_quiet(db: &Db, beat: &mut crate::script::Beat) {
    let n = beat.clips.len();
    if n < 2 {
        return;
    }
    let (clip, hold) = (&beat.clips[n - 2], &beat.clips[n - 1]);
    if hold.why.as_deref() != Some(CLOSING_PICTURE)
        || hold.video_id != clip.video_id
        || (hold.in_s - clip.out_s).abs() >= HOLD_JOIN_S
    {
        return;
    }
    let Some(quiet) = quiet_after_voice(db, clip.video_id, clip.in_s, clip.out_s) else { return };
    let Some(voice_end) = crate::speech::speech_regions(db, clip.video_id, clip.in_s, clip.out_s)
        .ok()
        .and_then(|r| r.iter().rev().find(|r| r.start_s < clip.out_s - 0.05).map(|r| r.end_s))
    else {
        return;
    };
    let until = voice_end + quiet - HOLD_QUIET_MARGIN_S;
    let hold = &mut beat.clips[n - 1];
    if hold.out_s > until {
        hold.out_s = until.max(hold.in_s);
    }
    if hold.out_s - hold.in_s < MIN_HOLD_S {
        beat.clips.pop();
    }
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
    // An "umm" at either edge is not the clip's: it ends and opens on its words, then is placed.
    let (head, tail) = unworded_edges(db, video_id, in_s, out_s, &regions);
    let (in_s, out_s) = (head.unwrap_or(in_s), tail.unwrap_or(out_s));
    // A clip written to end on a word stops before the next word starts, and one written to start
    // on a word opens after the previous one ends: inside running speech the next pause is the
    // sentence's end, and the whole rest of the sentence is not what was asked for.
    // Only for a range that starts well before it: a range that opens on that sentence is its own.
    let mut limit = push_limit(db, video_id, out_s, rules.guard, (rules.tail + 0.05).min((out_s - in_s) / 2.0));
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
    if tail.is_some() {
        new_out = Some((new_out.map_or(out_s, |m| m.0), CutKind::EndBeforeUnworded));
    }
    if head.is_some() {
        new_in = Some((new_in.map_or(in_s, |m| m.0), CutKind::StartAfterUnworded));
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
/// How much longer the last picture under a bed may be held so the voice can finish (s).
const MAX_PICTURE_HOLD_S: f64 = 1.0;

fn video_duration(db: &Db, video_id: i64) -> Option<f64> {
    db.conn
        .query_row("SELECT duration_s FROM videos WHERE id = ?1", [video_id], |r| r.get::<_, Option<f64>>(0))
        .ok()
        .flatten()
        .filter(|d| *d > 0.0)
}

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
            if clip.why.as_deref() == Some(CLOSING_PICTURE) {
                if clip.audio == Audio::Source {
                    beat.clips[ci].audio = Audio::Mute;
                    moves.push(note(Some(ci.saturating_sub(1)), CutKind::HoldMuted, clip.in_s, clip.out_s));
                }
                continue;
            }
            // Written to be kept exactly as it is. An edge written on a word is still settled on the
            // quiet beside that word below, never past the middle of the gap to its neighbour
            // (`ScriptClip::movable`'s one exception).
            if clip.exact || has_bed || clip.audio != Audio::Source || clip.tracks.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            let mut words = crate::sentences::WordEdges::of(db, clip.sentences.as_ref());
            // The interviewer off the edges and a filler opener off the start, by the one span rule:
            // the sentences asked for were the subject's, and a question or an "Awesome." at either
            // end is someone else's line. A word edge it moved is no longer the edge written.
            let span = spoken_span(db, clip.video_id, clip.in_s, clip.out_s, clip.sentences.as_ref(), true);
            if span.start.is_some() {
                words.start = None;
                words.start_span = None;
            }
            if span.end.is_some() {
                words.end = None;
                words.end_span = None;
            }
            let (from_in, mic_out) = (span.in_s, span.out_s);
            let trimmed = from_in != clip.in_s || mic_out != clip.out_s;
            let m = place_range(db, clip.video_id, from_in, mic_out, words, cfg, fps).or_else(|| {
                trimmed.then_some(RangeMove { in_s: from_in, out_s: mic_out, new_in: None, new_out: None })
            });
            let Some(m) = m else { continue };
            if span.start == Some(SpanCut::Filler) {
                moves.push(note(Some(ci), CutKind::StartAfterFiller, clip.in_s, m.in_s));
            } else if span.start == Some(SpanCut::OffMic) {
                moves.push(note(Some(ci), CutKind::StartAfterOffMic, clip.in_s, m.in_s));
            } else if let Some((to, kind)) = m.new_in.filter(|(to, _)| (to - clip.in_s).abs() >= NOTE_MIN_S) {
                moves.push(note(Some(ci), kind, clip.in_s, to));
            }
            if span.end.is_some() {
                moves.push(note(Some(ci), CutKind::EndBeforeOffMic, clip.out_s, m.out_s));
            } else if let Some((to, kind)) = m.new_out.filter(|(to, _)| (to - clip.out_s).abs() >= NOTE_MIN_S) {
                moves.push(note(Some(ci), kind, clip.out_s, to));
            }
            beat.clips[ci].in_s = m.in_s;
            beat.clips[ci].out_s = m.out_s;
            // A closing hold continues the shot this clip ends: it follows the cut.
            let delta = m.out_s - clip.out_s;
            if let Some(hold) = beat.clips.get_mut(ci + 1)
                && hold.why.as_deref() == Some(CLOSING_PICTURE)
                && hold.video_id == clip.video_id
                && (hold.in_s - clip.out_s).abs() < HOLD_JOIN_S
                && delta.abs() > 1e-9
            {
                let len = hold.out_s - hold.in_s;
                hold.in_s = m.out_s;
                hold.out_s = m.out_s + len;
            }
        }
        hold_until_quiet(db, beat);
        if let Some(bed) = &mut beat.bed
            && !bed.tracks.as_ref().is_some_and(Vec::is_empty)
        {
            // The voice under the pictures follows the same rule as a clip's: no interviewer at
            // either end.
            let mut words = crate::sentences::WordEdges::of(db, bed.sentences.as_ref());
            let span = spoken_span(db, bed.video_id, bed.in_s, bed.out_s, bed.sentences.as_ref(), false);
            if span.start.is_some() {
                words.start = None;
                words.start_span = None;
            }
            if span.end.is_some() {
                words.end = None;
                words.end_span = None;
            }
            let (mic_in, mic_out) = (span.in_s, span.out_s);
            let trimmed = (mic_in, mic_out) != (bed.in_s, bed.out_s);
            let Some(m) = place_range(db, bed.video_id, mic_in, mic_out, words, cfg, fps)
                .or_else(|| trimmed.then_some(RangeMove { in_s: mic_in, out_s: mic_out, new_in: None, new_out: None }))
            else {
                continue;
            };
            // A bed never outlasts the pictures it plays under. When the voice runs a little past
            // them, the last picture is held for it instead: cutting the bed there takes the last
            // word with it ("welcoming vi—"), and a shot a fraction longer costs nothing.
            let over = m.out_s - m.in_s - pictures;
            let mut covered = pictures;
            if pictures > 0.0
                && over > 1e-6
                && over <= MAX_PICTURE_HOLD_S
                && let Some(last) = beat.clips.last_mut().filter(|c| c.movable(crate::script::Side::Out))
            {
                let room = video_duration(db, last.video_id).map_or(f64::MAX, |d| d - last.out_s);
                if room >= over {
                    last.out_s += over;
                    covered += over;
                }
            }
            let out_s = if pictures > 0.0 && m.out_s - m.in_s > covered { m.in_s + covered } else { m.out_s };
            if span.start.is_some() {
                moves.push(note(None, CutKind::StartAfterOffMic, bed.in_s, m.in_s));
            } else if let Some((to, kind)) = m.new_in.filter(|(to, _)| (to - bed.in_s).abs() >= NOTE_MIN_S) {
                moves.push(note(None, kind, bed.in_s, to));
            }
            if span.end.is_some() {
                moves.push(note(None, CutKind::EndBeforeOffMic, bed.out_s, out_s));
            } else if let Some((to, kind)) = m.new_out.filter(|(to, _)| (to - bed.out_s).abs() >= NOTE_MIN_S) {
                moves.push(note(None, kind, bed.out_s, to));
            }
            // The pictures playing the very moment the bed plays (the face whose voice it is, in
            // sync) move with its start, onto exactly the moment the bed now plays there: settling the bed on the sound alone put the lips a frame
            // or more off their words, which the check then reported as someone else talking.
            let shift = m.in_s - bed.in_s;
            if shift.abs() > 1e-9 {
                let mut at = 0.0;
                for c in beat.clips.iter_mut() {
                    let len = (c.out_s - c.in_s).max(0.0);
                    let in_sync = c.video_id == bed.video_id && (c.in_s - (bed.in_s + at)).abs() <= IN_SYNC_S;
                    if in_sync && c.movable(crate::script::Side::In) && c.movable(crate::script::Side::Out) {
                        c.in_s = m.in_s + at;
                        c.out_s = c.in_s + len;
                    }
                    at += len;
                }
            }
            bed.in_s = m.in_s;
            bed.out_s = out_s;
        }
    }
    moves
}

/// How close a picture's source time must be to the bed's at that point of the beat to be the same
/// moment — the face the voice belongs to (s). The rule `finish::lay_audio_beds` and the check hold
/// a bed to.
const IN_SYNC_S: f64 = 0.05 + 1e-6;

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
    fn a_sentence_that_starts_in_the_tail_is_the_next_ones_and_not_kept() {
        // Kara, v154: s1978 ends 83.22 and s1979 ("And so we…") starts there, voiced 83.25–83.59.
        // An earlier pass ended the clip a 0.35 s tail past 83.22, on "And so"; the end goes back
        // into the pause before it instead of out past it.
        let regions =
            vec![region(82.34, 83.17, Some(83.18)), region(83.25, 83.59, Some(83.83)), region(84.12, 84.51, None)];
        let (out, kind) = place_out(&regions, 74.0, 83.57, Some(83.22 - 0.12), &rules()).unwrap();
        assert_eq!(kind, CutKind::EndOffNextSpeaker);
        assert!((83.17..83.25).contains(&out), "{out}");
    }

    #[test]
    fn a_short_pause_before_the_next_sentence_is_cut_in_not_the_last_word() {
        // Mary Ellen, v151: "I love Hotel Viata." voiced to 67.93; the next sentence starts at 68.00
        // on the transcript and its voice at 68.01. The cut goes in that pause, not at 68.00 - guard
        // (67.88), inside "Viata".
        let regions = vec![region(66.06, 67.93, Some(67.97)), region(68.01, 68.34, Some(68.43))];
        let (out, _) = place_out(&regions, 66.0, 67.6, Some(68.0 - 0.12), &rules()).unwrap();
        assert!(out > 67.93 && out < 68.01, "{out}");
        // And reading it back, the last word is not clipped: no flag left that nobody can fix.
        let read = judge_end(&regions, 66.0, out, &rules());
        assert_ne!(read.flag, EndFlag::ClipsLastWord, "{read:?}");
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
        ScriptClip {
            exact: false,
            gain_db: None,
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
        let back: ScriptClip =
            serde_json::from_value(serde_json::json!({"video_id": 104, "in_s": 1.0, "out_s": 2.0, "exact": true}))
                .unwrap();
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
        let issues = crate::script_check::check(&db, pid, &s, &ScriptConfig::default(), &Default::default()).unwrap();
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

    /// Garlar, v153: "yeah so" on its own breath (0.00–0.46), then "hi, my name is…" from 0.63; and
    /// Kara's "And Mark and Kayla…" in one breath from 20.0.
    fn db_with_openers() -> Db {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 60.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES
                 (104, 0.00, 9.05, 'yeah so hi my name is Garlar Stephenson I live in Westlake'),
                 (104, 20.00, 23.56, 'And Mark and Kayla are just incredible connectors of people.')",
                [],
            )
            .unwrap();
        db.set_speech_regions(
            VIDEO,
            &[region(0.00, 0.46, Some(0.52)), region(0.63, 9.05, Some(9.2)), region(20.0, 23.56, None)],
        )
        .unwrap();
        // After the regions: storing them re-splits the sentences, and these are the ones wanted.
        crate::sentence_split::mirror_segments(&db, 104);
        db
    }

    /// Garlar, v153: Whisper ended his segment one word early ("…that's a real") and the
    /// interviewer's segment opens with his "thing.". The off-mic trim cut the clip at the segment
    /// edge and lost the word; the sentence now ends after it, and so does the clip.
    #[test]
    fn a_word_the_speaker_finishes_in_the_interviewers_segment_stays_in_the_clip() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 60.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (104, 73.68, 81.47, 'and the vibe, welcoming vibe. People are nice and the southern hospitality, that''s a real', 0),
                 (104, 81.47, 87.3, 'thing. So this event there are a lot of other business owners, why is it', 1)",
                [],
            )
            .unwrap();
        crate::sentence_split::rebuild(&db, 104).unwrap();
        let ends: Vec<(String, f64)> = db
            .conn
            .prepare("SELECT text, end_s FROM transcript_sentences WHERE video_id = 104 ORDER BY start_s")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        let hospitality = ends.iter().find(|(t, _)| t.contains("hospitality")).expect("{ends:?}");
        assert!(hospitality.0.ends_with("real thing."), "{ends:?}");
        let (_, b) = on_mic_span(&db, VIDEO, 73.68, 87.3).expect("the question is taken off");
        assert!(b > 81.6 && b < 82.5, "the clip keeps \"thing.\": ends at {b}");
    }

    #[test]
    fn an_interviewers_line_at_either_end_of_a_clip_is_taken_off_and_one_inside_is_left() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 60.0)", [])
            .unwrap();
        // Kara, v154: two lines of hers, then the interviewer's question (off-mic).
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (104, 1.0, 3.6, 'Okay, so we are talking about it.', 1),
                 (104, 4.0, 7.0, 'So these events are always great for networking.', 0),
                 (104, 7.2, 8.6, 'We always meet someone great.', 0),
                 (104, 9.0, 10.8, 'What made you want to sponsor the bar?', 1),
                 (104, 11.2, 14.0, 'Well we love this hotel.', 0)",
                [],
            )
            .unwrap();
        assert_eq!(on_mic_span(&db, VIDEO, 4.0, 10.8), Some((4.0, 8.6)), "the question at the end goes");
        assert_eq!(on_mic_span(&db, VIDEO, 1.0, 8.6), Some((4.0, 8.6)), "the prompt at the start goes");
        assert_eq!(on_mic_span(&db, VIDEO, 4.0, 14.0), None, "one inside is left for the check to report");
        assert_eq!(on_mic_span(&db, VIDEO, 9.0, 10.8), None, "nothing on the mic would be left");

        let mut s = script(vec![clip(4.0, 10.8)]);
        let moves = place_cuts(&db, &mut s, &ScriptConfig::default());
        let c = &s.beats[0].clips[0];
        assert!(c.out_s <= 9.0 && c.out_s >= 8.6, "ends before the question: {}", c.out_s);
        assert!(moves.iter().any(|m| m.kind == CutKind::EndBeforeOffMic), "{moves:?}");
    }

    #[test]
    fn a_clip_opening_on_yeah_so_said_on_its_own_breath_starts_after_it() {
        let db = db_with_openers();
        assert_eq!(after_opening_filler(&db, VIDEO, 0.0), Some(0.63));
        // Padded a little earlier by the passes before: still the same sentence.
        assert_eq!(after_opening_filler(&db, VIDEO, -0.5_f64.max(0.0)), Some(0.63));
        // Run straight into the sentence: no pause to cut in.
        assert_eq!(after_opening_filler(&db, VIDEO, 20.0), None);
        // Placed inside the sentence on purpose: left alone.
        assert_eq!(after_opening_filler(&db, VIDEO, 2.0), None);

        let mut s = script(vec![clip(0.0, 9.05)]);
        let moves = place_cuts(&db, &mut s, &ScriptConfig::default());
        let c = &s.beats[0].clips[0];
        assert!((0.46..0.63).contains(&c.in_s), "opens in the pause before \"hi\": {}", c.in_s);
        assert!(moves.iter().any(|m| m.kind == CutKind::StartAfterFiller), "{moves:?}");
        // And reading it back, that is a clean start, not one inside the sentence.
        let notes = crate::readcut::story(&db, VIDEO, c.in_s, c.out_s);
        assert!(!notes.iter().any(|n| n.contains("mid-sentence") || n.contains("reply")), "{notes:?}");
    }

    /// The face whose voice a bed carries plays the same moment as the bed. Settling the bed's start
    /// on the sound moved the voice alone, and the lips ran 0.2 s ahead of the words (the benchmark
    /// cut's b5, flagged by the check as someone else talking): the picture moves with it.
    #[test]
    fn the_face_under_a_bed_moves_with_the_bed_start() {
        let (db, _) = db_with_handover();
        // The bed opens in the quiet before B's first word (12.88); the face is B at the same moment.
        let mut face = clip(12.40, 15.40);
        face.audio = Audio::Mute;
        let mut broll = ScriptClip { video_id: 7, ..clip(0.0, 1.0) };
        broll.audio = Audio::Mute;
        let mut s = script(vec![face, broll]);
        s.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: VIDEO,
            in_s: 12.40,
            out_s: 16.40,
            why: None,
            inferred: false,
            tracks: None,
        });
        place_cuts(&db, &mut s, &ScriptConfig::default());
        let bed = s.beats[0].bed.as_ref().unwrap();
        assert!(bed.in_s > 12.40 + 0.1, "the bed opens on the voice: {bed:?}");
        let c = &s.beats[0].clips[0];
        assert!((c.in_s - bed.in_s).abs() < 1e-9, "the face plays its own words: {c:?} under {bed:?}");
        assert!((c.out_s - c.in_s - 3.0).abs() < 1e-9, "and keeps its length: {c:?}");
        assert_eq!((s.beats[0].clips[1].in_s, s.beats[0].clips[1].out_s), (0.0, 1.0), "b-roll stays");
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
        // The pictures are a little shorter than the voice: the last picture is held for it, rather
        // than the bed cutting off A's last word.
        let mut short = s.clone();
        short.beats[0].clips[0].out_s = 12.0;
        short.beats[0].bed.as_mut().unwrap().out_s = 12.0;
        short.beats[0].bed.as_mut().unwrap().in_s = 0.0;
        place_cuts(&db, &mut short, &ScriptConfig::default());
        let bed = short.beats[0].bed.as_ref().unwrap();
        let pictures = short.beats[0].clips[0].out_s - short.beats[0].clips[0].in_s;
        assert!(bed.out_s >= 12.30, "the voice finishes: {bed:?}");
        assert!((bed.out_s - bed.in_s - pictures).abs() < 1e-9, "and the pictures cover it: {pictures}");
        // Far shorter: holding a shot that much longer is a different edit, so the bed is capped.
        let mut shorter = s.clone();
        shorter.beats[0].clips[0].out_s = 10.0;
        shorter.beats[0].bed.as_mut().unwrap().in_s = 0.0;
        place_cuts(&db, &mut shorter, &ScriptConfig::default());
        let bed = shorter.beats[0].bed.as_ref().unwrap();
        assert!(bed.out_s - bed.in_s <= 10.0 + 1e-9, "{bed:?}");
        assert!((shorter.beats[0].clips[0].out_s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_closing_hold_follows_the_cut_and_is_never_sound() {
        let (db, _) = db_with_handover();
        let hold = |audio| ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            video_id: VIDEO,
            in_s: 12.86,
            out_s: 14.86,
            audio,
            why: Some(CLOSING_PICTURE.into()),
            tracks: None,
            framing: None,
        };
        let mut s = script(vec![clip(0.0, 12.86), hold(Audio::Mute)]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        let (c, h) = (&s.beats[0].clips[0], &s.beats[0].clips[1]);
        assert_eq!(h.audio, Audio::Mute);
        assert!((h.in_s - c.out_s).abs() < 1e-9, "the hold continues the shot: {} vs {}", h.in_s, c.out_s);
        // B speaks from 12.88 s: the muted hold stops before them rather than showing them talk
        // with the sound gone.
        assert!(h.out_s <= 12.88 - HOLD_QUIET_MARGIN_S + 1e-9, "the hold runs into the next voice: {}", h.out_s);
        assert!(h.out_s - h.in_s >= MIN_HOLD_S, "and is not a flash");

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
        // The two speakers in two beats: inside one beat the chain lays the first voice under the
        // second picture as a bed (`finish::lay_audio_beds`), and there is no cut between them.
        let mut s = script(vec![clip(0.0, 12.86)]);
        let mut reply = s.beats[0].clone();
        reply.id = "b2".into();
        reply.clips = vec![clip(12.86, 16.2)];
        s.beats.push(reply);
        // The chain's passes, not the draft rules (this video is in no project).
        let profile = crate::finish::Profile { draft: false, ..crate::finish::Profile::chat(false) };
        let issues = crate::finish::run(&db, project_id, &mut s, &cfg, &profile).unwrap();
        let first = &s.beats[0].clips[0];
        assert!(12.86 - rendered_end(first.in_s, first.out_s, 30.0) >= 0.12, "{first:?}");
        let second = &s.beats[1].clips[0];
        assert!(second.in_s >= first.out_s - 1e-9 && second.in_s < 12.88, "{second:?}");
        assert!(issues.iter().any(|i| i.message.contains("first word")), "{issues:?}");
        let once = s.clone();
        crate::finish::run(&db, project_id, &mut s, &cfg, &profile).unwrap();
        assert_eq!(s, once, "a second repair moved the cut");
    }

    /// One span rule: an edge written on a word stays where it was asked for in the finishing pass
    /// and in the cut placement alike. Placement used to take the interviewer off it anyway and
    /// forget the word, undoing what the pass before had kept.
    #[test]
    fn an_edge_written_on_a_word_is_kept_by_the_span_rule_and_by_placement() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (7, 'h', 1, 60.0)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (7, 6.5, 8.5, 'What made you want to sponsor it?', 1),
                 (7, 9.0, 14.0, 'We love this hotel and the people in it.', 0)",
                [],
            )
            .unwrap();
        let on_word = crate::script::SentenceRange { from: "s1@4".into(), to: "s2".into() };
        let whole = crate::script::SentenceRange { from: "s1".into(), to: "s2".into() };
        let span = spoken_span(&db, 7, 6.5, 14.0, Some(&on_word), true);
        assert_eq!((span.in_s, span.start), (6.5, None), "asked for on a word");
        let span = spoken_span(&db, 7, 6.5, 14.0, Some(&whole), true);
        assert_eq!((span.in_s, span.start), (9.0, Some(SpanCut::OffMic)), "a whole sentence loses the interviewer");

        let mut s = script(vec![ScriptClip { video_id: 7, sentences: Some(on_word), ..clip(6.5, 14.0) }]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        assert_eq!(s.beats[0].clips[0].in_s, 6.5, "placement keeps the word edge too");
        let mut s = script(vec![ScriptClip { video_id: 7, sentences: Some(whole), ..clip(6.5, 14.0) }]);
        place_cuts(&db, &mut s, &ScriptConfig::default());
        assert_eq!(s.beats[0].clips[0].in_s, 9.0);
    }
}
