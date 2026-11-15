//! Reading a cut as text, without rendering it.
//!
//! An editor who wants to know whether a cut works watches it. A model cannot, and re-transcribing
//! a render is slow and costly. This builds the answer from the index instead: for every beat and
//! clip, the words that are heard (from the transcript, the edge words marked when a cut is close
//! to them), who speaks and who is on screen (the same per-moment scene data framing and
//! `get_scenes` use), how a vertical or square crop is framed, and, for each boundary, what the
//! measured speech says about it (`cuts::judge_*`: clean in silence, clips the last word, the next
//! speaker leaks in, starts mid-word). The running time, the total against the target and the
//! check's issues with their fixes close the reading.
//!
//! The reading is typed ([`CutReading`], serialisable for `format: json`) and renders to a compact
//! text ([`CutReading::render`]), which is what `read_cut` returns by default.

use crate::lexicon::{CONNECTIVES, REPLIES};

use std::collections::HashMap;

use serde::Serialize;

use crate::Error;
use crate::config::ScriptConfig;
use crate::cuts::{EndReading, StartReading};
use crate::db::Db;
use crate::framing::{self, Layout};
use crate::index;
use crate::scenes::{self, Scene};
use crate::script::{Audio, Issue, IssueSeverity, Script, ScriptClip};
use crate::sentences::{clock, format_id};

/// A cut within this of a word's edge, or through the word, marks the word.
const EDGE_WORD_S: f64 = 0.15;
/// A sentence needs this much of itself inside a range to count as heard (s).
const MIN_HEARD_S: f64 = 0.05;
/// Screen shares shorter than this are not worth a mention (s).
const MIN_SCREEN_S: f64 = 1.5;
/// Most scene stretches listed for one clip.
const MAX_STRETCHES: usize = 6;

/// One sentence, or the part of it that plays.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Heard {
    pub sentence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub off_mic: bool,
    /// The words that play. An edge word the cut is close to or through is wrapped in ‹ ›; a
    /// sentence that is cut short at either end has an ellipsis there.
    pub text: String,
}

/// One stretch of one kind of picture, in source seconds.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stretch {
    pub start_s: f64,
    pub end_s: f64,
    pub scene: Scene,
}

/// What is on screen during a range.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OnScreen {
    /// `None` when the faces stage has not looked at the video.
    pub stretches: Option<Vec<Stretch>>,
    /// Seconds of screen share inside the range.
    pub screen_share_s: f64,
    /// Who is seen, the longest first.
    pub people: Vec<String>,
}

/// A clip, read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClipReading {
    pub number: usize,
    pub video_id: i64,
    pub file: String,
    pub in_s: f64,
    pub out_s: f64,
    pub duration_s: f64,
    /// Where it starts in the cut.
    pub at_s: f64,
    pub audio: Audio,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// The sentences the clip was written with, when it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asked_for: Option<(String, String)>,
    pub heard: Vec<Heard>,
    /// Whether the clip holds whole thoughts: it starts or ends inside a sentence, opens on a
    /// reply or a connective that leans on a line we never hear, or ends on a question whose answer
    /// is cut. Empty when it reads as complete.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub story: Vec<String>,
    /// Who is on the microphone, as the transcript knows them.
    pub speakers: Vec<String>,
    pub on_screen: OnScreen,
    /// How a crop frames it, when the cut's shape differs from the footage's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop: Option<String>,
    /// `None`: the video's speech was not measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<StartReading>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<EndReading>,
}

/// A beat's audio bed, read.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BedReading {
    pub video_id: i64,
    pub file: String,
    pub in_s: f64,
    pub out_s: f64,
    pub duration_s: f64,
    pub heard: Vec<Heard>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub story: Vec<String>,
    pub speakers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<StartReading>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<EndReading>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BeatReading {
    pub id: String,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub narration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on_screen_text: Option<String>,
    /// The section card that opens the beat, and how long it shows (it plays from `start_s`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card_s: Option<f64>,
    pub start_s: f64,
    pub end_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bed: Option<BedReading>,
    pub clips: Vec<ClipReading>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CutReading {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_s: Option<f64>,
    pub total_s: f64,
    /// Total minus target, when there is a target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub over_target_s: Option<f64>,
    pub beats: Vec<BeatReading>,
    pub issues: Vec<Issue>,
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Read `script` against the index. `issues` are the check's, already computed for the same script.
pub fn read_cut(
    db: &Db,
    project_id: i64,
    script: &Script,
    cfg: &ScriptConfig,
    issues: Vec<Issue>,
) -> Result<CutReading, Error> {
    let names = person_names(db, project_id);
    let fps = script.fps.map(|f| f.as_f64()).filter(|f| *f > 0.0).unwrap_or(25.0);
    let (out_w, out_h) = (script.width.unwrap_or(0) as f64, script.height.unwrap_or(0) as f64);
    let mut file_names: HashMap<i64, String> = HashMap::new();
    let mut file = |db: &Db, id: i64| -> String {
        file_names.entry(id).or_insert_with(|| crate::clips::video_name(db, id)).clone()
    };

    let mut at = 0.0;
    let mut beats = Vec::with_capacity(script.beats.len());
    for beat in &script.beats {
        let beat_start = at;
        let card_s = script.card_s(beat);
        at += card_s;
        let mut clips = Vec::with_capacity(beat.clips.len());
        // With a bed, its voice carries the beat and the clips play silent under it.
        for (i, clip) in beat.clips.iter().enumerate() {
            let duration = (clip.out_s - clip.in_s).max(0.0);
            let hears = clip.audio == Audio::Source && !clip.tracks.as_ref().is_some_and(Vec::is_empty);
            let heard = if hears {
                heard(db, clip.video_id, clip.in_s, clip.out_s, clip.sentences.as_ref())
            } else {
                Vec::new()
            };
            let mut speakers: Vec<String> = Vec::new();
            for h in &heard {
                if let Some(s) = &h.speaker
                    && !speakers.contains(s)
                {
                    speakers.push(s.clone());
                }
            }
            let judged = if hears {
                let words = crate::sentences::WordEdges::of(db, clip.sentences.as_ref());
                crate::cuts::judge_range(db, clip.video_id, clip.in_s, clip.out_s, words, cfg, fps)
            } else {
                None
            };
            clips.push(ClipReading {
                number: i + 1,
                video_id: clip.video_id,
                file: file(db, clip.video_id),
                in_s: round2(clip.in_s),
                out_s: round2(clip.out_s),
                duration_s: round2(duration),
                at_s: round2(at),
                audio: clip.audio,
                why: clip.why.clone(),
                asked_for: clip.sentences.as_ref().map(|s| (s.from.clone(), s.to.clone())),
                story: if hears { story_as_written(db, clip) } else { Vec::new() },
                heard,
                speakers,
                on_screen: on_screen(db, &names, clip.video_id, clip.in_s, clip.out_s),
                crop: crop(db, &names, script, clip, out_w, out_h),
                start: judged.map(|j| j.0),
                end: judged.map(|j| j.1),
            });
            at += duration;
        }
        let bed = beat.bed.as_ref().map(|b| {
            let heard = if b.tracks.as_ref().is_some_and(Vec::is_empty) {
                Vec::new()
            } else {
                heard(db, b.video_id, b.in_s, b.out_s, b.sentences.as_ref())
            };
            let mut speakers: Vec<String> = Vec::new();
            for h in &heard {
                if let Some(s) = &h.speaker
                    && !speakers.contains(s)
                {
                    speakers.push(s.clone());
                }
            }
            let judged = crate::cuts::judge_range(
                db,
                b.video_id,
                b.in_s,
                b.out_s,
                crate::sentences::WordEdges::of(db, b.sentences.as_ref()),
                cfg,
                fps,
            );
            BedReading {
                video_id: b.video_id,
                file: file(db, b.video_id),
                in_s: round2(b.in_s),
                out_s: round2(b.out_s),
                duration_s: round2(b.duration_s()),
                story: if b.tracks.as_ref().is_some_and(Vec::is_empty) {
                    Vec::new()
                } else {
                    story(db, b.video_id, b.in_s, b.out_s)
                },
                heard,
                speakers,
                start: judged.map(|j| j.0),
                end: judged.map(|j| j.1),
            }
        });
        beats.push(BeatReading {
            id: beat.id.clone(),
            purpose: beat.purpose.clone(),
            narration: beat.narration.clone(),
            on_screen_text: beat.on_screen_text.clone(),
            card: beat.card.clone().filter(|_| card_s > 0.0).map(|c| c.text),
            card_s: (card_s > 0.0).then(|| round2(card_s)),
            start_s: round2(beat_start),
            end_s: round2(at),
            bed,
            clips,
        });
    }
    let total = script.length_s();
    Ok(CutReading {
        title: script.title.clone(),
        width: script.width,
        height: script.height,
        target_s: script.target_duration_s,
        total_s: round2(total),
        over_target_s: script.target_duration_s.map(|t| round2(total - t)),
        beats,
        issues,
    })
}

/// A person's label: name, else handle, else "person N".
fn person_names(db: &Db, project_id: i64) -> HashMap<i64, String> {
    db.people(project_id)
        .unwrap_or_default()
        .into_iter()
        .map(|p| {
            let label = [p.name, p.handle]
                .into_iter()
                .flatten()
                .map(|s| s.trim().to_string())
                .find(|s| !s.is_empty())
                .unwrap_or_else(|| format!("person {}", p.id));
            (p.id, label)
        })
        .collect()
}

/// A word of a sentence and when it is said.
struct TimedWord {
    text: String,
    start: f64,
    end: f64,
}

/// The words of a sentence with their times: the engine's when stored, else spread over the
/// sentence in proportion to their length (good to about a word, which is all the edge marks need).
fn timed_words(db: &Db, sentence_id: i64, text: &str, start: f64, end: f64) -> Vec<TimedWord> {
    // The sentence's own words (`sentences::own_words`): exactly its text, one time per word, on
    // its measured edges. Looking words up by time instead dropped or borrowed words wherever the
    // engine's timings drift, and the read-back showed words the clip does not play.
    let own = crate::sentences::own_words(db, sentence_id);
    let words: Vec<&str> = text.split_whitespace().collect();
    if own.len() == words.len() && !own.is_empty() {
        return words
            .iter()
            .zip(own)
            .map(|(w, (a, b))| TimedWord { text: (*w).to_string(), start: a, end: b })
            .collect();
    }
    let weight: f64 = words.iter().map(|w| w.chars().count() as f64 + 1.0).sum();
    let mut t = start;
    words
        .into_iter()
        .map(|w| {
            let span = (end - start) * (w.chars().count() as f64 + 1.0) / weight.max(1.0);
            let word = TimedWord { text: w.to_string(), start: t, end: t + span };
            t += span;
            word
        })
        .collect()
}

/// A word an edge was written on ("s206@9"): its sentence and 0-based index.
fn named_word(r: Option<&str>) -> Option<(i64, usize)> {
    let r = crate::sentences::SentenceRef::parse(r?).ok()?;
    Some((r.sentence_id, r.word? as usize - 1))
}

/// The story notes of a clip, leaving out "starts/ends mid-sentence" on an edge written as a word:
/// the editor cut inside that sentence on purpose.
fn story_as_written(db: &Db, clip: &crate::script::ScriptClip) -> Vec<String> {
    let from = named_word(clip.sentences.as_ref().map(|s| s.from.as_str())).is_some();
    let to = named_word(clip.sentences.as_ref().map(|s| s.to.as_str())).is_some();
    story(db, clip.video_id, clip.in_s, clip.out_s)
        .into_iter()
        .filter(|n| !(from && n.starts_with("starts mid-sentence")) && !(to && n.starts_with("ends mid-sentence")))
        .collect()
}

/// The sentences that play in `in_s..out_s`, as far as they play, with the cut's edge words marked.
/// An edge written on a word (`asked`, "s206@9") opens or closes on exactly that word: the cut was
/// placed on the sound in front of it, and the word's estimated times can sit a hundredth of a
/// second the wrong side of the cut — "… hay manera" for a clip that plays "No hay manera".
fn heard(db: &Db, video_id: i64, in_s: f64, out_s: f64, asked: Option<&crate::script::SentenceRange>) -> Vec<Heard> {
    let start_word = named_word(asked.map(|s| s.from.as_str()));
    let end_word = named_word(asked.map(|s| s.to.as_str()));
    let sentences: Vec<index::TranscriptSegment> = index::transcript(db, video_id)
        .unwrap_or_default()
        .into_iter()
        .filter(|s| s.end.min(out_s) - s.start.max(in_s) > MIN_HEARD_S)
        .collect();
    let last = sentences.len().saturating_sub(1);
    sentences
        .iter()
        .enumerate()
        .filter_map(|(k, s)| {
            let words = timed_words(db, s.id, &s.text, s.start, s.end);
            // A word plays when most of its estimated span is inside the clip — or, at a cut placed
            // on the measured gap between two words, when its estimated span merely reaches past the
            // cut: word times are estimates a few tenths out, the cut is on the sound.
            let plays = |w: &TimedWord| {
                let mid = (w.start + w.end) / 2.0;
                (in_s..=out_s).contains(&mid)
                    || (w.end > in_s + EDGE_REACH_S
                        && w.start < out_s - EDGE_REACH_S
                        && w.start >= in_s - WORD_SLACK_S
                        && w.end <= out_s + WORD_SLACK_S)
            };
            let mut inside: Vec<usize> = (0..words.len()).filter(|&i| plays(&words[i])).collect();
            let lo = start_word.filter(|(id, w)| *id == s.id && *w < words.len()).map(|(_, w)| w);
            let hi = end_word.filter(|(id, w)| *id == s.id && *w < words.len()).map(|(_, w)| w);
            if lo.is_some() || hi.is_some() {
                let from = lo.unwrap_or_else(|| inside.first().copied().unwrap_or(0));
                let to = hi.unwrap_or_else(|| inside.last().copied().unwrap_or(words.len() - 1));
                inside = (from..=to.max(from)).collect();
            }
            let (&first, &final_) = (inside.first()?, inside.last()?);
            let mut parts: Vec<String> = Vec::with_capacity(inside.len() + 2);
            if first > 0 {
                parts.push("…".into());
            }
            for &i in &inside {
                let w = &words[i];
                // The first and last word of the whole range: marked when the cut is on them.
                let at_start = k == 0 && i == first && w.start < in_s + EDGE_WORD_S;
                let at_end = k == last && i == final_ && w.end > out_s - EDGE_WORD_S;
                parts.push(if at_start || at_end { format!("‹{}›", w.text) } else { w.text.clone() });
            }
            if final_ + 1 < words.len() {
                parts.push("…".into());
            }
            Some(Heard {
                sentence: format_id(s.id),
                speaker: s.speaker.clone(),
                off_mic: s.off_mic,
                text: parts.join(" "),
            })
        })
        .collect()
}

/// How far a word's estimated span must reach past a cut to count as playing (s).
const EDGE_REACH_S: f64 = 0.08;
/// How far before a start (or after an end) a word's estimate may begin and still be the word the
/// cut was placed in front of (s): the cut sits on the measured gap, the estimate is a few tenths out.
const WORD_SLACK_S: f64 = 0.3;

/// A range that starts or ends this far inside a sentence cuts the thought (s).
const MID_SENTENCE_S: f64 = 0.4;

/// The first `n` words of `text`, with an ellipsis when there are more.
fn opening(text: &str, n: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut s = words.iter().take(n).copied().collect::<Vec<_>>().join(" ");
    if words.len() > n {
        s.push('…');
    }
    s
}

/// The last `n` words of `text`.
fn closing(text: &str, n: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    words[words.len().saturating_sub(n)..].join(" ")
}

/// A pause shorter than this after a sentence, before the same speaker's next one, is no pause:
/// they ran on, and the thought was not finished there (s).
const RUN_ON_S: f64 = 0.25;

/// Whether the measured speech runs from a sentence ending at `end_s` into the next one starting at
/// `next_s` with no pause of [`RUN_ON_S`] between them. The transcript's own times butt sentences
/// together whatever the voice did, so only the sound can say; unmeasured, nothing is claimed.
fn runs_on(db: &Db, video_id: i64, end_s: f64, next_s: f64) -> bool {
    let Ok(regions) = crate::speech::speech_regions(db, video_id, end_s - 2.0, next_s + 2.0) else { return false };
    if regions.is_empty() {
        return false;
    }
    let (lo, hi) = (end_s - 0.1, next_s + 0.3);
    let paused = regions.windows(2).any(|w| {
        let (a, b) = (w[0].end_s, w[1].start_s);
        b - a >= RUN_ON_S && b > lo && a < hi
    // Or no voice is measured from the next sentence on: whatever the transcript says, it is quiet.
    }) || regions.last().is_some_and(|r| r.end_s <= next_s);
    !paused
}

/// Whether `in_s..out_s` holds whole thoughts, as notes for the editor (see [`ClipReading::story`]).
pub(crate) fn story(db: &Db, video_id: i64, in_s: f64, out_s: f64) -> Vec<String> {
    let all = index::transcript(db, video_id).unwrap_or_default();
    let inside: Vec<usize> =
        (0..all.len()).filter(|&i| all[i].end.min(out_s) - all[i].start.max(in_s) > MIN_HEARD_S).collect();
    let (Some(&first), Some(&last)) = (inside.first(), inside.last()) else { return Vec::new() };
    let mut notes = Vec::new();
    let (a, z) = (&all[first], &all[last]);
    // A start just after a "yeah so" said on its own breath is where the thought begins: placement
    // drops those words (`cuts::after_opening_filler`), and that is not cutting into the sentence.
    let after_filler = crate::cuts::after_opening_filler(db, video_id, a.start)
        .is_some_and(|t| in_s > a.start + 0.05 && in_s >= t - 0.4 && in_s <= t + 0.1);
    let starts_inside = !after_filler && in_s > a.start + MID_SENTENCE_S && a.end - a.start > MID_SENTENCE_S * 2.0;
    if after_filler {
    } else if starts_inside {
        notes.push(format!(
            "starts mid-sentence: {} began {:.1} s earlier (\"{}\"); start at {} or later",
            format_id(a.id),
            in_s - a.start,
            opening(&a.text, 6),
            format_id(a.id)
        ));
    } else {
        let word =
            a.text.split_whitespace().next().unwrap_or("").trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let before = first.checked_sub(1).map(|i| &all[i]).filter(|p| a.start - p.end < 3.0);
        let quoted = opening(&a.text, 4);
        if let Some(p) = first.checked_sub(1).map(|i| &all[i]).filter(|p| {
            a.start - p.end < 1.5 && p.speaker == a.speaker && crate::lexicon::opens_as_continuation(&a.text)
        }) {
            // "En API de Opus, lo cual…": the clip never says what it is about.
            notes.push(format!(
                "opens mid-thought (\"{quoted}\"): it is the rest of {} (\"…{}\"); start at {}",
                format_id(p.id),
                closing(&p.text, 6),
                format_id(p.id)
            ));
        } else if REPLIES.contains(&word.as_str()) {
            let answers = match before {
                Some(p) => format!(
                    " — it answers {}{}",
                    format_id(p.id),
                    p.speaker.as_deref().map(|s| format!(" ({s})")).unwrap_or_default()
                ),
                None => String::new(),
            };
            notes.push(format!(
                "opens on a reply (\"{quoted}\"){answers}: start on the line it answers, or on one that stands alone"
            ));
        } else if CONNECTIVES.contains(&word.as_str()) && before.is_some() {
            notes.push(format!(
                "opens on a connective (\"{quoted}\"): check it stands without {}",
                format_id(before.map(|p| p.id).unwrap_or(a.id))
            ));
        }
    }
    let ends_inside = out_s < z.end - MID_SENTENCE_S && z.end - z.start > MID_SENTENCE_S * 2.0;
    if ends_inside {
        notes.push(format!(
            "ends mid-sentence: {} runs {:.1} s past the cut; end at the sentence before, or after it",
            format_id(z.id),
            z.end - out_s
        ));
    } else if let Some(next) = all
        .get(last + 1)
        .filter(|n| n.start - z.end < RUN_ON_S && !n.off_mic && !z.off_mic && n.speaker == z.speaker)
        .filter(|n| runs_on(db, video_id, z.end, n.start))
    {
        // "…super common. Like, we go to networking events…": a full stop in the transcript, but
        // no breath in the voice. The speaker was not done; an end there sounds cut off.
        notes.push(format!(
            "ends where the speaker runs straight on: {} (\"{}\") follows {:.2} s later, with no pause; end on a \
             sentence followed by a pause, or take it in",
            format_id(next.id),
            opening(&next.text, 6),
            (next.start - z.end).max(0.0)
        ));
    } else if z.text.trim_end().ends_with('?')
        && !crate::sentence_split::ends_on_tag(&z.text)
        && let Some(next) = all.get(last + 1).filter(|n| n.start - z.end < 3.0 && n.speaker != z.speaker)
    {
        notes.push(format!(
            "ends on a question; the answer ({}{}) is not in the clip",
            format_id(next.id),
            next.speaker.as_deref().map(|s| format!(", {s}")).unwrap_or_default()
        ));
    }
    // An off-mic line (an interviewer's question, a "perfect" from behind the camera) inside a clip
    // whose own voice is on the mic: it plays, quieter, and the viewer hears someone they never see.
    if inside.iter().any(|&i| !all[i].off_mic) {
        for &i in inside.iter().filter(|&&i| all[i].off_mic) {
            notes.push(format!(
                "plays an off-mic line: {} (\"{}\"), someone away from the microphone; end before it or start after it",
                format_id(all[i].id),
                opening(&all[i].text, 8)
            ));
        }
    }
    notes
}

/// The pictures of a range: what kind, for how long, and who.
fn on_screen(db: &Db, names: &HashMap<i64, String>, video_id: i64, in_s: f64, out_s: f64) -> OnScreen {
    let stretches = scenes::for_range(db, video_id, in_s, out_s).ok().flatten();
    let screen_share_s = stretches
        .as_deref()
        .map(|spans| spans.iter().filter(|s| s.scene == Scene::ScreenShare).map(|s| s.len()).sum())
        .unwrap_or(0.0);
    let track_people = db.face_track_people(video_id).unwrap_or_default();
    let mut seen: Vec<(String, f64)> = Vec::new();
    for t in db.face_tracks(video_id).unwrap_or_default() {
        let overlap = t.end_s.min(out_s) - t.start_s.max(in_s);
        if overlap <= 0.0 {
            continue;
        }
        let Some(name) = track_people.get(&t.id).and_then(|p| names.get(p)) else { continue };
        match seen.iter_mut().find(|(n, _)| n == name) {
            Some((_, secs)) => *secs += overlap,
            None => seen.push((name.clone(), overlap)),
        }
    }
    seen.sort_by(|a, b| b.1.total_cmp(&a.1));
    OnScreen {
        stretches: stretches.map(|spans| {
            spans.iter().map(|s| Stretch { start_s: round1(s.start), end_s: round1(s.end), scene: s.scene }).collect()
        }),
        screen_share_s: round1(screen_share_s),
        people: seen.into_iter().take(3).map(|(n, _)| n).collect(),
    }
}

/// How a crop frames a clip, in words: only for a cut whose shape differs from the footage's.
fn crop(
    db: &Db,
    names: &HashMap<i64, String>,
    script: &Script,
    clip: &ScriptClip,
    out_w: f64,
    out_h: f64,
) -> Option<String> {
    let (src_w, src_h) = framing::video_size(db, clip.video_id).ok().flatten()?;
    if !framing::needs_crop(src_w, src_h, out_w, out_h) {
        return None;
    }
    let Some(framing) = framing::effective(script, clip) else {
        return Some("letterboxed (no framing set)".into());
    };
    let plan = framing::plan_clip(db, None, None, clip.video_id, clip.in_s, clip.out_s, framing, out_w, out_h)
        .ok()
        .flatten()?;
    let mut parts: Vec<String> = Vec::new();
    for seg in &plan.segments {
        let who = |shot_subject: Option<usize>| {
            shot_subject
                .and_then(|i| plan.subjects.get(i))
                .and_then(|s| s.person_id)
                .and_then(|p| names.get(&p).cloned())
        };
        let shots: Vec<String> = seg
            .paths
            .first()
            .map(|p| {
                p.shots
                    .iter()
                    .filter_map(|s| who(s.subject).map(|n| format!("{n} {:.1}-{:.1}", s.start, s.end)))
                    .collect()
            })
            .unwrap_or_default();
        let mut part = seg.layout.label().to_string();
        if !shots.is_empty() && matches!(seg.layout, Layout::FollowSpeaker | Layout::Stacked | Layout::Fixed) {
            part.push_str(&format!(" on {}", shots.join(", ")));
        }
        if plan.segments.len() > 1 {
            part.push_str(&format!(" [{} {:.1}-{:.1} s]", seg.scene.as_str().replace('_', " "), seg.start, seg.end));
        }
        if seg.quality.longest_no_face_s > 0.6 {
            part.push_str(&format!(
                ", face lost {:.1} s at {:.1} s",
                seg.quality.longest_no_face_s,
                seg.quality.no_face_at.unwrap_or(0.0)
            ));
        }
        if seg.quality.cut_off_share > 0.2 {
            part.push_str(&format!(", face cut off {:.0}%", seg.quality.cut_off_share * 100.0));
        }
        parts.push(part);
    }
    Some(parts.join("; "))
}

fn stretch_text(on: &OnScreen) -> String {
    let Some(stretches) = &on.stretches else { return "faces not indexed".into() };
    let name = |s: Scene| s.as_str().replace('_', " ");
    let mut text = match stretches.as_slice() {
        [] => "nothing measured".to_string(),
        [one] => name(one.scene),
        many => many
            .iter()
            .take(MAX_STRETCHES)
            .map(|s| format!("{} {}-{}", name(s.scene), clock(s.start_s), clock(s.end_s)))
            .collect::<Vec<_>>()
            .join(", "),
    };
    if stretches.len() > MAX_STRETCHES {
        text.push_str(&format!(", +{} more", stretches.len() - MAX_STRETCHES));
    }
    if !on.people.is_empty() {
        text.push_str(&format!(" ({})", on.people.join(", ")));
    }
    if on.screen_share_s >= MIN_SCREEN_S && stretches.len() > 1 {
        text.push_str(&format!(" — screen share {:.1} s", on.screen_share_s));
    }
    text
}

fn heard_text(heard: &[Heard]) -> String {
    heard
        .iter()
        .map(|h| {
            let who = h.speaker.as_deref().map(|s| format!("{s}: ")).unwrap_or_default();
            let off = if h.off_mic { " (off-mic)" } else { "" };
            format!("{who}{} {}{off}", h.sentence, h.text)
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn edge_text(start: &Option<StartReading>, end: &Option<EndReading>) -> String {
    match (start, end) {
        (Some(s), Some(e)) => format!("in: {} · out: {}", s.detail(), e.detail()),
        _ => "boundaries: speech not measured (run the speech stage)".into(),
    }
}

impl CutReading {
    /// The reading as compact text.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let size = match (self.width, self.height) {
            (Some(w), Some(h)) => format!(" · {w}x{h}"),
            _ => String::new(),
        };
        let target = match (self.target_s, self.over_target_s) {
            (Some(t), Some(d)) => format!(" (target {t:.1} s, {d:+.1} s)"),
            _ => String::new(),
        };
        out.push_str(&format!("CUT \"{}\"{size} · {:.1} s{target}\n", self.title, self.total_s));
        for beat in &self.beats {
            out.push_str(&format!(
                "{} \"{}\" [{}-{}] {:.1} s\n",
                beat.id,
                beat.purpose,
                clock(beat.start_s),
                clock(beat.end_s),
                beat.end_s - beat.start_s
            ));
            if let Some(n) = beat.narration.as_ref().filter(|n| !n.trim().is_empty()) {
                out.push_str(&format!("  narration: {n}\n"));
            }
            if let (Some(t), Some(d)) = (&beat.card, beat.card_s) {
                out.push_str(&format!("  card ({d:.1} s, before the clips): {t}\n"));
            }
            if let Some(t) = beat.on_screen_text.as_ref().filter(|t| !t.trim().is_empty()) {
                out.push_str(&format!("  title: {t}\n"));
            }
            if let Some(bed) = &beat.bed {
                out.push_str(&format!(
                    "  bed v{} {} src {}-{} ({:.1} s): {}\n",
                    bed.video_id,
                    bed.file,
                    clock(bed.in_s),
                    clock(bed.out_s),
                    bed.duration_s,
                    if bed.heard.is_empty() { "silent".to_string() } else { heard_text(&bed.heard) }
                ));
                if !bed.story.is_empty() {
                    out.push_str(&format!("     story: {}\n", bed.story.join(" · ")));
                }
                if bed.start.is_some() {
                    out.push_str(&format!("     {}\n", edge_text(&bed.start, &bed.end)));
                }
            }
            for c in &beat.clips {
                let audio = match c.audio {
                    Audio::Source => "",
                    Audio::Mute => ", muted",
                };
                out.push_str(&format!(
                    "  {}. v{} {} src {}-{} ({:.1} s{audio}) at {}\n",
                    c.number,
                    c.video_id,
                    c.file,
                    clock(c.in_s),
                    clock(c.out_s),
                    c.duration_s,
                    clock(c.at_s)
                ));
                if let Some(why) = &c.why {
                    out.push_str(&format!("     why: {why}\n"));
                }
                if c.audio == Audio::Source {
                    out.push_str(&format!(
                        "     hears: {}\n",
                        if c.heard.is_empty() { "no speech".to_string() } else { heard_text(&c.heard) }
                    ));
                }
                out.push_str(&format!("     sees: {}", stretch_text(&c.on_screen)));
                if let Some(crop) = &c.crop {
                    out.push_str(&format!(" · crop: {crop}"));
                }
                out.push('\n');
                if !c.story.is_empty() {
                    out.push_str(&format!("     story: {}\n", c.story.join(" · ")));
                }
                if c.audio == Audio::Source {
                    out.push_str(&format!("     {}\n", edge_text(&c.start, &c.end)));
                }
            }
        }
        match self.target_s {
            Some(t) => out.push_str(&format!(
                "TOTAL {:.1} s vs target {t:.1} s ({:+.1} s)\n",
                self.total_s,
                self.over_target_s.unwrap_or(0.0)
            )),
            None => out.push_str(&format!("TOTAL {:.1} s (no target)\n", self.total_s)),
        }
        if self.issues.is_empty() {
            out.push_str("ISSUES none\n");
        } else {
            out.push_str(&format!("ISSUES {}\n", self.issues.len()));
            for i in &self.issues {
                // The clip's own story line already says it.
                if i.message.starts_with("story: ") && i.clip_index.is_some() {
                    out.push_str(&format!(
                        "  [warning] {} clip {}: see its story line\n",
                        i.beat_id.as_deref().unwrap_or(""),
                        i.clip_index.unwrap_or(0) + 1
                    ));
                    continue;
                }
                let sev = match i.severity {
                    IssueSeverity::Error => "error",
                    IssueSeverity::Warning => "warning",
                    IssueSeverity::Info => "info",
                };
                let place = match (&i.beat_id, i.clip_index) {
                    (Some(b), Some(c)) => format!("{b} clip {}: ", c + 1),
                    (Some(b), None) => format!("{b}: "),
                    _ => String::new(),
                };
                out.push_str(&format!("  [{sev}] {place}{}\n", i.message));
                for f in &i.fixes {
                    out.push_str(&format!("      fix: {}\n", f.label));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;
    use crate::script::{Beat, Fps};
    use crate::speech::SpeechRegion;

    /// Two voices handing over, as in the benchmark cut: A voiced to 12.06 s, B from 12.88 s.
    fn db() -> (Db, i64) {
        let db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 300.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES
                 (412, 104, 8.00, 12.02, 'Entonces le dije que no'),
                 (413, 104, 12.86, 16.46, 'Esto es O Corres o Te Encaramas')",
                [],
            )
            .unwrap();
        db.set_speech_regions(
            104,
            &[
                SpeechRegion { start_s: 8.10, end_s: 12.06, quiet_after_s: Some(12.48) },
                SpeechRegion { start_s: 12.88, end_s: 16.30, quiet_after_s: None },
            ],
        )
        .unwrap();
        crate::sentence_split::mirror_segments(&db, 104);
        (db, project.id)
    }

    fn clip(in_s: f64, out_s: f64, audio: Audio) -> ScriptClip {
        ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            video_id: 104,
            in_s,
            out_s,
            audio,
            why: None,
            tracks: None,
            framing: None,
        }
    }

    fn script(clips: Vec<ScriptClip>) -> Script {
        Script {
            title: "Test".into(),
            target_duration_s: Some(8.0),
            fps: Some(Fps::new(30, 1)),
            width: Some(1920),
            height: Some(1080),
            framing: None,
            beats: vec![Beat {
                id: "b1".into(),
                purpose: "hook".into(),
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

    /// "…super common. Like, we go to networking events…": a full stop in the transcript, no breath
    /// in the voice. Judged on the measured speech, never on the transcript's butted times.
    #[test]
    fn a_clip_ending_where_the_speaker_runs_straight_on_is_a_story_note() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 300.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES
                 (1, 104, 1.0, 4.0, 'Every event we have come to was worth it.'),
                 (2, 104, 4.6, 6.0, 'I do not think that is common.'),
                 (3, 104, 6.0, 9.0, 'Like we go to networking events all the time.')",
                [],
            )
            .unwrap();
        db.set_speech_regions(104, &[SpeechRegion::new(1.0, 4.0), SpeechRegion::new(4.6, 6.0), SpeechRegion::new(6.09, 9.0)])
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 104);
        let runs_on = |out_s| story(&db, 104, 0.9, out_s).iter().any(|n| n.starts_with("ends where the speaker runs"));
        assert!(runs_on(6.0), "a 0.09 s breath is no pause");
        assert!(!runs_on(4.0), "0.6 s of quiet after it is an end");
    }

    #[test]
    fn an_off_mic_question_inside_a_clip_is_a_story_note() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (104, 'h', 1, 300.0)", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 104, 1.0, 4.0, 'These events are always great for networking.', 0),
                 (2, 104, 4.2, 6.0, 'What made you want to sponsor the bar?', 1),
                 (3, 104, 7.0, 9.0, 'We wanted the visibility.', 0)",
                [],
            )
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 104);
        let notes = story(&db, 104, 0.9, 6.2);
        assert!(notes.iter().any(|n| n.starts_with("plays an off-mic line: s2")), "{notes:?}");
        assert!(story(&db, 104, 0.9, 4.1).iter().all(|n| !n.contains("off-mic")));
    }

    #[test]
    fn a_clipped_word_and_a_leaking_speaker_are_flagged_from_the_stored_speech() {
        let (db, pid) = db();
        let cfg = ScriptConfig::default();
        // Ends on the transcript's 12.86, where B's attack already is; opens on a clean pause.
        let leak = read_cut(&db, pid, &script(vec![clip(8.0, 12.86, Audio::Source)]), &cfg, Vec::new()).unwrap();
        let c = &leak.beats[0].clips[0];
        assert_eq!(c.end.unwrap().flag, crate::cuts::EndFlag::NextSpeakerLeaksIn);
        assert_eq!(c.start.unwrap().flag, crate::cuts::StartFlag::Clean);
        let text = leak.render();
        assert!(text.contains("out: next speaker leaks in (the next voice starts 0.02 s after the cut)"), "{text}");
        assert!(text.contains("in: clean (in silence)"), "{text}");

        // Ends inside A's last word (voiced to 12.06).
        let clipped = read_cut(&db, pid, &script(vec![clip(8.0, 11.9, Audio::Source)]), &cfg, Vec::new()).unwrap();
        assert_eq!(clipped.beats[0].clips[0].end.unwrap().flag, crate::cuts::EndFlag::ClipsLastWord);
        let text = clipped.render();
        assert!(text.contains("out: clips the last word (the voice runs 0.16 s past the cut)"), "{text}");
        // The word the cut goes through is marked, and what is cut off is not heard.
        assert!(text.contains("hears: s412 ‹Entonces› le dije que ‹no›"), "{text}");

        // A clean cut says so, and marks nothing.
        let clean = read_cut(&db, pid, &script(vec![clip(7.5, 12.4, Audio::Source)]), &cfg, Vec::new()).unwrap();
        let text = clean.render();
        assert!(text.contains("out: clean (in silence), 0.34 s after the last word"), "{text}");
        assert!(text.contains("hears: s412 Entonces le dije que no\n"), "{text}");
        assert!(!text.contains('‹'), "{text}");
    }

    #[test]
    fn it_reports_the_words_heard_the_running_time_and_the_total_against_the_target() {
        let (db, pid) = db();
        let s = script(vec![clip(8.0, 12.4, Audio::Source), clip(12.7, 16.6, Audio::Source)]);
        let r = read_cut(&db, pid, &s, &ScriptConfig::default(), Vec::new()).unwrap();
        let (a, b) = (&r.beats[0].clips[0], &r.beats[0].clips[1]);
        assert_eq!((a.at_s, b.at_s), (0.0, 4.4));
        assert!((r.total_s - 8.3).abs() < 1e-9 && (r.over_target_s.unwrap() - 0.3).abs() < 1e-9);
        assert_eq!(b.heard[0].sentence, "s413");
        let text = r.render();
        assert!(text.contains("TOTAL 8.3 s vs target 8.0 s (+0.3 s)"), "{text}");
        assert!(text.contains("2. v104") && text.contains("src 0:12.7-0:16.6") && text.contains("at 0:04.4"), "{text}");
        assert!(text.contains("ISSUES none"), "{text}");
    }

    #[test]
    fn a_cut_through_a_sentence_keeps_the_words_that_play_and_marks_the_ellipsis() {
        let (db, pid) = db();
        // Opens in the middle of B's sentence (about the fourth word, estimated).
        let r =
            read_cut(&db, pid, &script(vec![clip(14.4, 16.6, Audio::Source)]), &ScriptConfig::default(), Vec::new())
                .unwrap();
        let text = &r.beats[0].clips[0].heard[0].text;
        assert!(text.starts_with("… "), "{text}");
        assert!(text.ends_with("Encaramas›") || text.ends_with("Encaramas"), "{text}");
        // Exact word timings, when stored, are used instead of the estimate.
        db.conn
            .execute_batch(
                "INSERT INTO transcript_words(segment_id, idx, start_s, end_s, text) VALUES
                 (413, 1, 12.90, 13.10, 'Esto'), (413, 2, 13.10, 13.20, 'es'), (413, 3, 14.50, 14.70, 'O'),
                 (413, 4, 14.80, 15.20, 'Corres'), (413, 5, 15.30, 15.40, 'o'), (413, 6, 15.50, 15.60, 'Te'),
                 (413, 7, 15.70, 16.30, 'Encaramas');",
            )
            .unwrap();
        // New word timings rebuild the sentences with them, as the indexer does.
        crate::sentence_split::mirror_segments(&db, 104);
        let r =
            read_cut(&db, pid, &script(vec![clip(14.45, 16.6, Audio::Source)]), &ScriptConfig::default(), Vec::new())
                .unwrap();
        assert_eq!(r.beats[0].clips[0].heard[0].text, "… ‹O› Corres o Te Encaramas");
        // A clip written to open on a word ("s413@4") opens on that word, even where its estimated
        // times end a hundredth past the cut placed on the sound in front of it.
        let mut on_word = clip(15.19, 16.6, Audio::Source);
        on_word.sentences = Some(crate::script::SentenceRange { from: "s413@4".into(), to: "s413".into() });
        let r = read_cut(&db, pid, &script(vec![on_word]), &ScriptConfig::default(), Vec::new()).unwrap();
        let c = &r.beats[0].clips[0];
        assert_eq!(c.heard[0].text, "… ‹Corres› o Te Encaramas");
        assert!(!c.story.iter().any(|n| n.starts_with("starts mid-sentence")), "{:?}", c.story);
    }

    #[test]
    fn muted_clips_hear_nothing_and_a_bed_carries_the_beat() {
        let (db, pid) = db();
        let mut s = script(vec![clip(200.0, 204.0, Audio::Mute)]);
        s.beats[0].bed = Some(crate::script::AudioBed {
            sentences: None,
            video_id: 104,
            in_s: 12.7,
            out_s: 16.6,
            why: None,
            inferred: false,
            tracks: None,
        });
        let r = read_cut(&db, pid, &s, &ScriptConfig::default(), Vec::new()).unwrap();
        assert!(r.beats[0].clips[0].heard.is_empty() && r.beats[0].clips[0].end.is_none());
        let bed = r.beats[0].bed.as_ref().unwrap();
        assert_eq!(bed.heard[0].sentence, "s413");
        let text = r.render();
        assert!(text.contains("bed v104") && text.contains("s413 "), "{text}");
        assert!(text.contains("1. v104") && text.contains("muted"), "{text}");
        assert!(!text.contains("hears: no speech"), "a muted clip hears nothing, it is not 'no speech': {text}");
    }

    #[test]
    fn without_measured_speech_the_reading_says_so_instead_of_inventing_flags() {
        let (db, pid) = db();
        db.conn.execute("DELETE FROM speech_regions", []).unwrap();
        let r =
            read_cut(&db, pid, &script(vec![clip(8.0, 12.86, Audio::Source)]), &ScriptConfig::default(), Vec::new())
                .unwrap();
        assert!(r.beats[0].clips[0].end.is_none());
        assert!(r.render().contains("speech not measured"), "{}", r.render());
    }

    #[test]
    fn issues_close_the_reading_with_their_fixes_and_a_screen_share_is_named() {
        let (db, pid) = db();
        // Faces: two big ones for 0..10 s of video 104, then small camera tiles (a screen share).
        db.conn
            .execute(
                "INSERT INTO face_tracks(id, video_id, start_s, end_s, n_faces, mean_embedding, avg_box_h)
                 VALUES (1, 104, 8.0, 14.0, 1, x'00', 0.07)",
                [],
            )
            .unwrap();
        for i in 0..30 {
            db.conn
                .execute(
                    "INSERT INTO face_boxes(track_id, t_s, x, y, w, h, score, landmarks_json, sharpness, cut_off)
                     VALUES (1, ?1, 0.7, 0.85, 0.05, 0.07, 0.9, '[]', 1.0, 0)",
                    [8.0 + i as f64 * 0.2],
                )
                .unwrap();
        }
        let issue = Issue {
            severity: IssueSeverity::Warning,
            beat_id: Some("b1".into()),
            clip_index: Some(0),
            message: "end 12.86 s of video #104: next speaker leaks in".into(),
            fixes: vec![crate::script::Fix::new(
                "place the cut on the speech (8.00–12.40 s)",
                crate::script::FixAction::Trim { beat_id: "b1".into(), clip_index: 0, in_s: 8.0, out_s: 12.4 },
            )],
        };
        let r =
            read_cut(&db, pid, &script(vec![clip(8.0, 12.4, Audio::Source)]), &ScriptConfig::default(), vec![issue])
                .unwrap();
        assert_eq!(r.beats[0].clips[0].on_screen.screen_share_s, 4.4);
        let text = r.render();
        assert!(text.contains("sees: screen share"), "{text}");
        assert!(
            text.contains("ISSUES 1")
                && text.contains("[warning] b1 clip 1: end 12.86 s")
                && text.contains("fix: place the cut"),
            "{text}"
        );
    }

    #[test]
    fn the_story_says_when_a_clip_cuts_a_thought_or_leans_on_a_line_we_never_hear() {
        let (db, pid) = db();
        let read = |clips: Vec<ScriptClip>| {
            read_cut(&db, pid, &script(clips), &ScriptConfig::default(), Vec::new()).unwrap().beats[0].clips[0]
                .story
                .clone()
        };
        // Whole sentences: s413 on its own reads as complete.
        assert!(read(vec![clip(12.7, 16.6, Audio::Source)]).is_empty());
        // In 1.5 s after s412 began, out 2 s before s413 ends.
        let notes = read(vec![clip(9.5, 14.4, Audio::Source)]);
        assert!(notes.iter().any(|n| n.starts_with("starts mid-sentence: s412 began 1.5 s earlier")), "{notes:?}");
        assert!(notes.iter().any(|n| n.starts_with("ends mid-sentence: s413 runs 2.1 s past the cut")), "{notes:?}");
        // s412 opens on "Entonces", but nothing is said before it: nothing to lean on.
        assert!(read(vec![clip(7.9, 12.4, Audio::Source)]).is_empty());
        // A reply right after someone else's question.
        db.conn
            .execute_batch(
                "INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text) VALUES
                 (420, 104, 20.0, 22.0, '¿Y eso funciona?'), (421, 104, 22.5, 25.0, 'Exacto, funciona muy bien.');",
            )
            .unwrap();
        let notes = read(vec![clip(22.4, 25.2, Audio::Source)]);
        assert!(
            notes.iter().any(|n| n.starts_with("opens on a reply (\"Exacto, funciona muy bien.\") — it answers s420")),
            "{notes:?}"
        );
        // The check says it too, so save_script returns it as an issue to answer...
        let check = |clips: Vec<ScriptClip>| {
            crate::script_check::check(
                &db,
                pid,
                &script(clips),
                &ScriptConfig::default(),
                &crate::script_check::CheckOptions::default(),
            )
            .unwrap()
        };
        let issues = check(vec![clip(22.4, 25.2, Audio::Source)]);
        assert!(issues.iter().any(|i| i.message.starts_with("story: opens on a reply")), "{issues:?}");
        // ...unless the line it answers plays right before it.
        let issues = check(vec![clip(19.9, 22.2, Audio::Source), clip(22.4, 25.2, Audio::Source)]);
        assert!(!issues.iter().any(|i| i.message.starts_with("story: opens on")), "{issues:?}");
        let notes = read(vec![clip(19.9, 22.2, Audio::Source)]);
        assert!(
            notes.iter().any(|n| n.starts_with("ends on a question; the answer (s421")) || notes.is_empty(),
            "{notes:?}"
        );
    }

    #[test]
    fn a_clip_that_opens_on_the_rest_of_a_thought_is_flagged_and_a_tag_is_no_question() {
        // The B1 bench's "10 dólares" clip: the opening says what costs ten dollars, the clip
        // started after it, on "En API de Opus".
        let (db, pid) = db();
        db.conn
            .execute_batch(
                "INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text) VALUES
                 (430, 104, 40.0, 46.0, 'Esto nos está costando 10 dólares por episodio, ¿ok?'),
                 (431, 104, 46.3, 52.0, 'En API de Opus, lo cual uno podría pensar que es caro, ¿no?'),
                 (432, 104, 52.4, 55.0, 'Y luego otra cosa.');",
            )
            .unwrap();
        let read = |a: f64, b: f64| {
            read_cut(&db, pid, &script(vec![clip(a, b, Audio::Source)]), &ScriptConfig::default(), Vec::new())
                .unwrap()
                .beats[0]
                .clips[0]
                .story
                .clone()
        };
        let notes = read(46.2, 52.1);
        assert!(
            notes.iter().any(|n| n.starts_with("opens mid-thought (\"En API de Opus,…\"): it is the rest of s430")),
            "{notes:?}"
        );
        // From the start of the thought it reads whole, and "¿ok?" / "¿no?" ask nothing.
        assert!(read(39.9, 52.1).is_empty(), "{:?}", read(39.9, 52.1));
    }
}
