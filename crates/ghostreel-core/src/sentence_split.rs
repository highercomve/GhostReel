//! Real sentences from Whisper's segments.
//!
//! Whisper's segments are not sentences. Most are ~2 s display fragments that stop mid-thought
//! ("y te digo, y te soy honesto" / "la gente que usa Hermes"); a few are 20-30 s run-ons holding
//! several sentences. A model that cuts "by sentence" on those either cuts mid-thought or has to
//! guess where inside 29 s a line begins.
//!
//! This module rebuilds the transcript as sentences, deterministically:
//! - the segments' words are laid end to end, each with a time (the engine's word timings when it
//!   gave them, otherwise the segment's span shared out by characters);
//! - a sentence ends after a full stop, question or exclamation mark, at a speaker change between
//!   segments, and at a long pause in the measured speech when there is no punctuation to go by;
//!   one that still runs past [`MAX_SENTENCE_S`] is split at its best comma or pause;
//! - every boundary is then moved onto the nearest pause in the measured speech
//!   ([`crate::speech`]), so a sentence ends where the voice stops and the next starts where it
//!   starts again.
//!
//! Sentences live in `transcript_sentences` and are what sentence ids (`s412`) name. They are built
//! on first use and rebuilt when the transcript or the measured speech changes. Who says a sentence
//! and whether it is off-mic are read from the segments it overlaps, so naming a speaker never
//! needs a rebuild.

use rusqlite::params;

use crate::Error;
use crate::db::Db;
use crate::speech::SpeechRegion;

/// A pause at least this long ends a sentence where there is no punctuation to go by (s).
pub const PAUSE_BREAK_S: f64 = 0.65;
/// A sentence longer than this is split at its best comma or pause (s): past it, a "sentence" is a
/// run-on an editor cannot cut whole thoughts out of.
pub const MAX_SENTENCE_S: f64 = 14.0;
/// The version of the splitting rules. A video whose sentences were built by an older one
/// (`videos.sentence_split`) has them built again on first use, its chapters kept.
pub const SPLIT_VERSION: i64 = 6;
/// A pause does not end a sentence when the engine wrote punctuation this close to it (s): it is
/// punctuating here, so its full stops say where sentences end.
const PUNCTUATION_NEAR_S: f64 = 10.0;
/// Below this a sentence made by a pause alone is joined to the next (s): "ok" is not a thought.
const MIN_SOFT_SENTENCE_S: f64 = 1.2;
/// A pause this long ends a sentence however short it is (s).
const ALONE_PAUSE_S: f64 = 1.5;
/// How far a boundary that came from a segment's own edge may move to reach a pause (s).
const SNAP_EDGE_S: f64 = 0.35;
/// A pause that ends this long before a timed word ends (or starts this long after the next one
/// starts) is on the wrong side of it: a boundary there hands the word to the other sentence (s).
const WORD_SIDE_S: f64 = 0.1;
/// A timed word longer than this was stretched over the silence beside it, and says nothing about
/// where its sound is (s).
const MAX_WORD_S: f64 = 0.7;
/// The shortest gap in the speech that counts as a pause a boundary can sit in (s).
const MIN_GAP_S: f64 = 0.08;

/// One transcript segment, as the splitter reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct SegIn {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The voice or person speaking, when known: a change between segments ends a sentence.
    pub speaker: Option<i64>,
    /// Whether this segment was flagged off-mic: a change between on-mic and off-mic ends a
    /// sentence even when `speaker` is unknown.
    pub off_mic: bool,
    /// Timed words heard in this segment, in order, when an engine gave them. They need not be
    /// the segment's own words: another engine's words are matched to the text ([`align`]).
    pub words: Vec<WordIn>,
}

/// A recognised word and when it was said.
#[derive(Debug, Clone, PartialEq)]
pub struct WordIn {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A word reduced to what two transcriptions agree on: lower case, letters and digits.
fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// A time for each of `words` (the text's own words, in `start..end`), from the timed words an
/// engine heard. The two are matched word for word (longest common subsequence on the normalised
/// words), so another engine's transcription of the same audio, with a few words heard differently,
/// still times the text. A word with no match shares out the time between its matched neighbours
/// by characters; with no timed words at all, the whole span is shared out that way.
pub fn align(words: &[&str], timed: &[WordIn], start: f64, end: f64) -> Vec<(f64, f64)> {
    let (n, m) = (words.len(), timed.len());
    let a: Vec<String> = words.iter().map(|w| norm(w)).collect();
    let b: Vec<String> = timed.iter().map(|w| norm(&w.text)).collect();
    // lcs[i][j]: the longest match of a[i..] and b[j..].
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] =
                if !a[i].is_empty() && a[i] == b[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let mut anchor: Vec<Option<(f64, f64)>> = vec![None; n];
    let (mut i, mut j) = (0, 0);
    let mut floor = start;
    while i < n && j < m {
        if !a[i].is_empty() && a[i] == b[j] {
            let s0 = timed[j].start.max(floor).clamp(start, end);
            let e0 = timed[j].end.max(s0).min(end);
            anchor[i] = Some((s0, e0));
            floor = e0;
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    // Fill each unmatched run between its neighbours' times, by characters.
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut k = 0;
    while k < n {
        if let Some(t) = anchor[k] {
            out.push(t);
            k += 1;
            continue;
        }
        let run_end = (k..n).find(|&x| anchor[x].is_some()).unwrap_or(n);
        let left = out.last().map(|t| t.1).unwrap_or(start);
        let right = anchor.get(run_end).copied().flatten().map(|t| t.0).unwrap_or(end).max(left);
        let weights: Vec<f64> = words[k..run_end].iter().map(|w| w.chars().count() as f64 + 1.0).collect();
        let total: f64 = weights.iter().sum::<f64>().max(1.0);
        let mut t = left;
        for (x, w) in weights.iter().enumerate() {
            let d = (right - left) * w / total;
            let e = if k + x + 1 == run_end { right } else { t + d };
            out.push((t, e));
            t = e;
        }
        k = run_end;
    }
    out
}

/// One sentence: its span and words.
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceOut {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// One (start, end) per word of `text`, when an engine timed the words: the engine's times
    /// stretched onto the sentence's own measured edges (word timings drift by a few tenths; the
    /// edges sit on the pauses of the measured speech).
    pub words: Option<Vec<(f64, f64)>>,
}

#[derive(Debug, Clone)]
struct Tok {
    text: String,
    start: f64,
    end: f64,
    /// The last word of its segment: the boundary after it is a segment edge, timed by Whisper.
    seg_last: bool,
    /// The speaker changes after this word (between two segments with known, different speakers).
    turn_after: bool,
    /// Its time came from an engine's word timings, not from sharing out the segment's span.
    timed: bool,
}

/// Words of a sentence that end it: "." "?" "!" "…", allowing closing quotes or brackets after.
fn ends_sentence(word: &str) -> bool {
    let w = word.trim_end_matches(['"', '\'', '”', '’', '»', ')', ']']);
    if !w.ends_with(['.', '?', '!', '…']) {
        return false;
    }
    let lower = w.to_lowercase();
    // Abbreviations that end in a full stop without ending a sentence.
    const ABBR: &[&str] =
        &["sr.", "sra.", "dr.", "dra.", "vs.", "etc.", "ej.", "p.ej.", "ud.", "uds.", "mr.", "mrs.", "e.g.", "i.e."];
    if ABBR.contains(&lower.as_str()) {
        return false;
    }
    // "1." "2.5." (a numbered list or a decimal) — a digit right before the stop.
    let before = w.trim_end_matches(['.', '?', '!', '…']);
    !(w.ends_with('.') && before.chars().last().is_some_and(|c| c.is_ascii_digit()) && before.len() <= 2)
}

/// A word reduced for matching against the word lists below: lower case, no punctuation or ¿¡.
fn bare(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

/// Tag questions: "¿ok?", "¿no?", "right?" ask nothing, they check the listener is following.
const TAGS: &[&str] =
    &["ok", "okay", "no", "vale", "verdad", "sí", "si", "eh", "ves", "sabes", "cierto", "right", "yeah", "see"];

/// Whether `word` is a tag question: a short check-in that ends in "?" without asking anything.
pub(crate) fn is_tag_question(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', '”', '’', '»', ')', ']']).ends_with('?') && TAGS.contains(&bare(word).as_str())
}

/// Whether the last words of `text` are a tag question ("…10 dólares episodio, ¿ok?").
pub(crate) fn ends_on_tag(text: &str) -> bool {
    text.split_whitespace().last().is_some_and(is_tag_question)
}

pub(crate) use crate::lexicon::opens_as_continuation;

/// A comma, colon or semicolon after the word: a place to split a run-on that has no full stop.
fn ends_clause(word: &str) -> bool {
    word.ends_with([',', ';', ':'])
}

/// The segments' words end to end, each with a time.
fn tokens(segs: &[SegIn]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut last_known: Option<i64> = None;
    for (si, s) in segs.iter().enumerate() {
        let words: Vec<&str> = s.text.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let times = align(&words, &s.words, s.start, s.end);
        // A mic change (on-mic <-> off-mic) always ends a turn, even when `speaker` is unknown
        // (e.g. semantic interviewer detection without speaker clustering). Off-mic segments never
        // adopt or carry over an on-mic speaker identity.
        if s.off_mic {
            last_known = None;
        } else if s.speaker.is_some() {
            last_known = s.speaker;
        }
        let next = segs[si + 1..].iter().find(|n| n.text.split_whitespace().next().is_some());
        let mic_change = next.is_some_and(|n| n.off_mic != s.off_mic);
        let next_speaker = next.and_then(|n| if n.off_mic { None } else { n.speaker });
        let turn = mic_change || matches!((last_known, next_speaker), (Some(a), Some(b)) if a != b);
        let n = words.len();
        for (i, (w, (a, b))) in words.iter().zip(times).enumerate() {
            out.push(Tok {
                text: (*w).to_string(),
                start: a,
                end: b.max(a),
                seg_last: i + 1 == n,
                turn_after: turn && i + 1 == n,
                timed: !s.words.is_empty(),
            });
        }
    }
    out
}

/// The pauses in the measured speech: the gaps between consecutive regions, as (start, end).
fn gaps(regions: &[SpeechRegion]) -> Vec<(f64, f64)> {
    regions
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (w[0].end_s, w[1].start_s);
            (b - a >= MIN_GAP_S).then_some((a, b))
        })
        .collect()
}

/// Measured speech longer than this between two words is not a filler but a passage the
/// transcript lost; the boundary is placed as usual.
const MAX_UNTRANSCRIBED_S: f64 = 2.0;

/// Where a sentence ending on `last` stops and the next one, opening on `next`, starts, when the
/// measured speech holds sound between them that no word accounts for: Whisper writes no "umm",
/// "uh" or laugh, so a voiced region after the last word ends and before the next one starts is
/// one of those. Only where those times are the engine's own (a segment edge, or timed words):
/// a time estimated inside a segment says nothing about what lies between. `(end, start)`: the
/// pause before the first such region and the one after the last, both on the measured speech.
fn untranscribed_between(last: &Tok, next: &Tok, regions: &[SpeechRegion], gaps: &[(f64, f64)]) -> Option<(f64, f64)> {
    if gaps.is_empty() || !(last.seg_last || (last.timed && next.timed)) {
        return None;
    }
    let mut lost = regions.iter().filter(|r| r.start_s >= last.end - 0.02 && r.end_s <= next.start + 0.02);
    let first = lost.next()?;
    let final_ = lost.next_back().unwrap_or(first);
    if final_.end_s - first.start_s > MAX_UNTRANSCRIBED_S {
        return None;
    }
    let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let before = gaps.iter().find(|g| near(g.1, first.start_s))?;
    let after = gaps.iter().find(|g| near(g.0, final_.end_s))?;
    Some((before.0, after.1))
}

/// How far around the boundary after token `i` its true time may be: a segment edge is Whisper's
/// own timestamp, close; a word inside a segment was estimated, and the estimate drifts with the
/// segment's length.
fn window(toks: &[Tok], i: usize, seg_len: f64) -> f64 {
    if toks[i].seg_last { SNAP_EDGE_S } else { (seg_len * 0.12).clamp(0.4, 1.5) }
}

/// At most this many words, ending a sentence, at the head of a segment across a mic change are
/// the previous speaker's sentence finishing ([`carry_tails`]).
const MAX_TAIL_WORDS: usize = 2;

/// Give the previous speaker back the end of their sentence where the transcript broke it across a
/// mic change: "…that's a real" (on the microphone) then "thing. So this event…" (the interviewer).
/// The engine's segment edge fell one word early, so splitting at the mic change made "that's a
/// real" a sentence and left "thing." on the interviewer's side, where every rule that keeps
/// off-mic lines out cut it. A head of at most [`MAX_TAIL_WORDS`] words that starts lower case and
/// ends a sentence, after a segment that does not end one, is moved back onto that segment — its
/// words, text and time — so the mic change falls after it. A capitalised "Okay." or "Right." is a
/// line of its own and stays.
pub(crate) fn carry_tails(segs: &[SegIn]) -> std::borrow::Cow<'_, [SegIn]> {
    let carries = |a: &SegIn, b: &SegIn| -> Option<usize> {
        if a.off_mic == b.off_mic || a.text.split_whitespace().last().is_none_or(ends_sentence) {
            return None;
        }
        let words: Vec<&str> = b.text.split_whitespace().collect();
        let first = words.first()?;
        if !first.chars().find(|c| c.is_alphabetic()).is_some_and(char::is_lowercase) {
            return None;
        }
        (1..=MAX_TAIL_WORDS.min(words.len())).find(|&k| ends_sentence(words[k - 1]))
    };
    if !segs.windows(2).any(|w| carries(&w[0], &w[1]).is_some()) {
        return std::borrow::Cow::Borrowed(segs);
    }
    let mut out = segs.to_vec();
    for i in 1..out.len() {
        let Some(k) = carries(&out[i - 1], &out[i]) else { continue };
        let b = out[i].clone();
        let words: Vec<&str> = b.text.split_whitespace().collect();
        // Where the head ends: the engine's word time when it timed them, else its share of the
        // segment by letters.
        let cut = if b.words.is_empty() {
            let len = |ws: &[&str]| ws.iter().map(|w| w.chars().count() + 1).sum::<usize>() as f64;
            b.start + (b.end - b.start) * len(&words[..k]) / len(&words).max(1.0)
        } else {
            align(&words, &b.words, b.start, b.end)[k - 1].1
        }
        .clamp(b.start, b.end);
        let head = words[..k].join(" ");
        let rest = words[k..].join(" ");
        let (moved, kept): (Vec<WordIn>, Vec<WordIn>) = b.words.iter().cloned().partition(|w| w.end <= cut + 0.02);
        let a = &mut out[i - 1];
        a.text = format!("{} {head}", a.text.trim_end());
        a.end = a.end.max(cut);
        a.words.extend(moved);
        let b = &mut out[i];
        b.text = rest;
        b.start = cut;
        b.words = kept;
    }
    std::borrow::Cow::Owned(out)
}

/// Split segments into sentences, with each boundary placed on a pause of `regions` (the measured
/// speech, in order; empty when it was not measured).
pub fn split(segs: &[SegIn], regions: &[SpeechRegion]) -> Vec<SentenceOut> {
    let segs = carry_tails(segs);
    let segs = segs.as_ref();
    let toks = tokens(segs);
    if toks.is_empty() {
        return Vec::new();
    }
    let gaps = gaps(regions);
    // The length of the segment each token came from, for its boundary window.
    let seg_len: Vec<f64> = {
        let mut lens = Vec::with_capacity(toks.len());
        for s in segs {
            let n = s.text.split_whitespace().count();
            lens.extend(std::iter::repeat_n((s.end - s.start).max(0.0), n));
        }
        lens
    };
    // The boundary after token `i`: its estimated time.
    let at = |i: usize| (toks[i].end + toks[i + 1].start) / 2.0;
    // 1. Each pause goes to the one word boundary nearest it (inside a long segment many word
    //    estimates are near the same pause; only one of them is where it really is). Measured
    //    pauses when the speech was measured, else the engine's own silences between words.
    let pauses: Vec<(f64, f64)> = if gaps.is_empty() {
        (0..toks.len().saturating_sub(1))
            .filter(|&i| toks[i + 1].start - toks[i].end >= MIN_GAP_S)
            .map(|i| (toks[i].end, toks[i + 1].start))
            .collect()
    } else {
        gaps.clone()
    };
    let mut pause_at: Vec<Option<(f64, f64)>> = vec![None; toks.len()];
    for (ga, gb) in pauses {
        let mid = (ga + gb) / 2.0;
        let near = (0..toks.len().saturating_sub(1))
            .filter(|&i| (at(i) - mid).abs() <= window(&toks, i, seg_len[i]) + (gb - ga) / 2.0)
            .min_by(|&x, &y| {
                // A segment edge or a comma is a likelier place for the pause than a word inside.
                let cost = |i: usize| {
                    (at(i) - mid).abs()
                        - if toks[i].seg_last { 0.3 } else { 0.0 }
                        - if ends_clause(&toks[i].text) { 0.2 } else { 0.0 }
                };
                cost(x).total_cmp(&cost(y))
            });
        if let Some(i) = near
            && pause_at[i].is_none_or(|(a, b)| b - a < gb - ga)
        {
            pause_at[i] = Some((ga, gb));
        }
    }
    let pause_len = |i: usize| pause_at[i].map(|(a, b)| b - a).unwrap_or(0.0);

    // 2. Where sentences end. Hard: a full stop or a change of voice. Soft: a long pause.
    let mut cuts: Vec<usize> = Vec::new(); // a sentence ends after these token indexes
    let mut first = 0usize;
    for i in 0..toks.len() {
        if i + 1 == toks.len() {
            cuts.push(i);
            break;
        }
        // A tag question the same voice talks straight on from, into a clause that cannot stand
        // alone, is a comma said aloud: "cuesta 10 dólares, ¿ok? En API de Opus…" is one thought.
        let tag_inside = is_tag_question(&toks[i].text)
            && !toks[i].turn_after
            && pause_len(i) < PAUSE_BREAK_S
            && opens_as_continuation(
                &toks[i + 1..(i + 3).min(toks.len())].iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" "),
            );
        let hard = (ends_sentence(&toks[i].text) && !tag_inside) || toks[i].turn_after;
        // A short stretch ends at a pause only when the pause is long: then it really stood alone.
        let long_enough = toks[i].end - toks[first].start >= MIN_SOFT_SENTENCE_S || pause_len(i) >= ALONE_PAUSE_S;
        // Where the engine is punctuating, its full stops say where sentences end, and a pause
        // without one is a breath inside a sentence. Pauses only stand in where it wrote none.
        // The same voice only: one speaker's lines can come back punctuated and the other's not,
        // and the first one's full stops say nothing about where the second one's thoughts end.
        let t = toks[i].end;
        let voice_from = (0..i).rev().find(|&j| toks[j].turn_after).map_or(0, |j| j + 1);
        let voice_to = (i..toks.len()).find(|&j| toks[j].turn_after).unwrap_or(toks.len() - 1);
        let punctuating = toks[voice_from..=voice_to]
            .iter()
            .skip_while(|w| w.end < t - PUNCTUATION_NEAR_S)
            .take_while(|w| w.start <= t + PUNCTUATION_NEAR_S)
            .any(|w| ends_sentence(&w.text) || ends_clause(&w.text));
        // Without measured speech there are no pauses to go by: where the engine wrote no
        // punctuation, its own segment edges stand in (they were the sentences before this).
        let soft_here = pause_len(i) >= PAUSE_BREAK_S || (gaps.is_empty() && toks[i].seg_last);
        if hard || (soft_here && long_enough && !punctuating) {
            cuts.push(i);
            first = i + 1;
        }
    }

    // 3. A sentence that still runs too long is split at its best comma or pause, repeatedly.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut a = 0;
    for &b in &cuts {
        ranges.push((a, b));
        a = b + 1;
    }
    let mut i = 0;
    while i < ranges.len() {
        let (a, b) = ranges[i];
        let dur = toks[b].end - toks[a].start;
        if dur <= MAX_SENTENCE_S || b <= a + 1 {
            i += 1;
            continue;
        }
        let mid = (toks[a].start + toks[b].end) / 2.0;
        let best = (a..b)
            .map(|k| {
                let t = toks[k].end;
                // Prefer a long pause, then a comma, near the middle; never leave a scrap under 2 s.
                let scrap = (t - toks[a].start).min(toks[b].end - t) < 2.0;
                // A tag question ("¿ok?", "¿no?") closes the line before it: split after it, never
                // just before it, or the next sentence opens on "¿ok?".
                let before_tag = is_tag_question(&toks[k + 1].text);
                let after_tag = is_tag_question(&toks[k].text);
                let score = pause_len(k).min(1.5) * 2.0
                    // A tag closes a thought the way a full stop would, more surely than a comma or
                    // a hesitation does.
                    + if after_tag { 2.5 } else if ends_clause(&toks[k].text) { 1.0 } else { 0.0 }
                    - ((t - mid).abs() / dur) * 1.5
                    - if scrap { 10.0 } else { 0.0 }
                    - if before_tag { 5.0 } else { 0.0 };
                (k, score)
            })
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(k, _)| k)
            .unwrap_or((a + b) / 2);
        ranges.splice(i..=i, [(a, best), (best + 1, b)]);
    }

    // 4. Times: each boundary on its pause in the speech (the one given to it above, else the nearest).
    let mut out: Vec<SentenceOut> = ranges
        .iter()
        .map(|&(a, b)| SentenceOut {
            start: toks[a].start,
            end: toks[b].end,
            text: toks[a..=b].iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" "),
            words: None,
        })
        .collect();
    for k in 0..out.len().saturating_sub(1) {
        let b = ranges[k].1;
        let lo = out[k].start + 0.05;
        let hi = out[k + 1].end - 0.05;
        // Sound between them that no word accounts for (an "umm", a laugh, a breath) is neither
        // sentence's: each ends at its own words, and a cut on either leaves the sound out.
        if let Some((end, start)) = untranscribed_between(&toks[b], &toks[b + 1], regions, &gaps)
            && end > lo
            && start < hi
        {
            out[k].end = end;
            out[k + 1].start = start;
            continue;
        }
        let w = window(&toks, b, seg_len[b]);
        let t = (out[k].end + out[k + 1].start) / 2.0;
        // The pause nearest the estimate, longer pauses winning ties within the window.
        let snap = pause_at[b].filter(|_| !gaps.is_empty()).or_else(|| {
            gaps.iter()
                .filter(|(ga, gb)| *gb > t - w && *ga < t + w)
                .min_by(|x, y| {
                    let d = |g: &(f64, f64)| {
                        let dist = if t < g.0 {
                            g.0 - t
                        } else if t > g.1 {
                            t - g.1
                        } else {
                            0.0
                        };
                        dist - 0.3 * (g.1 - g.0).min(1.0)
                    };
                    d(x).total_cmp(&d(y))
                })
                .copied()
        });
        if let Some((ga, gb)) = snap {
            // Never past the neighbours' own words; where the words were timed, never across the
            // last word or the first one either (a pause before "¿ok?" would hand the "ok" to the
            // next sentence, and a cut there opens on it).
            // Only a pause that ends inside the last word (or starts inside the next) is on its
            // wrong side: engines stretch words over the silence beside them, and where a timing is
            // further off than a word the engine drifted, and the measured speech is right.
            let (last, next) = (&toks[b], &toks[b + 1]);
            let solid = |t: &Tok| t.timed && t.end - t.start <= MAX_WORD_S;
            let crosses = (solid(last) && gb > last.start && gb < last.end - WORD_SIDE_S)
                || (solid(next) && ga < next.end && ga > next.start + WORD_SIDE_S);
            if !crosses && ga > lo && gb < hi {
                out[k].end = ga;
                out[k + 1].start = gb;
            }
        }
    }
    // 5. Each sentence's words. An engine's word times are right where they fall inside measured
    //    speech and wrong where they fall in its silence (it squeezes a sentence's first words
    //    into the pause before it, then catches up): those words move to where the speech resumes,
    //    and words that pile up there are spread out to the next well-timed word.
    let in_speech = |t: f64| gaps.is_empty() || regions.iter().any(|r| r.start_s - 0.02 <= t && t <= r.end_s + 0.02);
    let resume = |t: f64| regions.iter().find(|r| r.start_s >= t).map(|r| r.start_s);
    for (o, &(a, b)) in out.iter_mut().zip(&ranges) {
        if !toks[a..=b].iter().any(|t| t.timed) {
            continue;
        }
        let mut starts: Vec<f64> = Vec::with_capacity(b - a + 1);
        let mut moved: Vec<bool> = Vec::with_capacity(b - a + 1);
        let mut last = o.start;
        for t in &toks[a..=b] {
            let mut st = t.start.clamp(o.start, o.end);
            let off = !in_speech(st);
            if off {
                st = resume(st).unwrap_or(st).min(o.end);
            }
            st = st.max(last);
            last = st;
            starts.push(st);
            moved.push(off);
        }
        // Spread each run of words that landed on one moment up to the next word's own time.
        let n = starts.len();
        let mut k = 0;
        while k < n {
            let mut e = k + 1;
            while e < n && (starts[e] - starts[k]).abs() < 1e-6 {
                e += 1;
            }
            if e - k > 1 {
                let until = if e < n { starts[e] } else { toks[b].end.clamp(starts[k], o.end) };
                let weights: Vec<f64> =
                    toks[a + k..a + e].iter().map(|t| t.text.chars().count() as f64 + 1.0).collect();
                let total: f64 = weights.iter().sum();
                let mut t = starts[k];
                for (m, w) in (k..e).zip(&weights) {
                    starts[m] = t;
                    t += (until - starts[k]) * w / total;
                }
            }
            k = e;
        }
        let words: Vec<(f64, f64)> = (0..n)
            .map(|m| {
                let st = starts[m];
                let next = if m + 1 < n { starts[m + 1] } else { o.end };
                // A word moved off the silence ends where the next begins; one the engine timed in
                // speech keeps its own end, inside the sentence and before the next word.
                let tok = &toks[a + m];
                let en = if moved[m] { next } else { tok.end.clamp(st, o.end).min(next.max(st)) };
                ((st * 1000.0).round() / 1000.0, (en.max(st) * 1000.0).round() / 1000.0)
            })
            .collect();
        o.words = Some(words);
    }
    out
}

/// The segments of a video as the splitter reads them, with their word timings. The speaker is the
/// person, else the voice (negated, so the two never collide); `off_mic` is tracked separately so
/// a change between on-mic and off-mic always splits a sentence even when `speaker` is unknown.
fn load_segments(db: &Db, video_id: i64) -> Result<Vec<SegIn>, Error> {
    let mut st = db.conn.prepare(
        "SELECT id, start_s, end_s, text, COALESCE(person_id, -speaker), COALESCE(off_mic, 0) = 1
           FROM transcript_segments
          WHERE video_id = ?1 AND TRIM(text) <> '' ORDER BY start_s, id",
    )?;
    let rows: Vec<(i64, SegIn)> = st
        .query_map([video_id], |r| {
            Ok((
                r.get(0)?,
                SegIn {
                    start: r.get(1)?,
                    end: r.get(2)?,
                    text: r.get(3)?,
                    speaker: r.get(4)?,
                    off_mic: r.get(5)?,
                    words: Vec::new(),
                },
            ))
        })?
        .collect::<Result<_, _>>()?;
    let mut words =
        db.conn.prepare("SELECT text, start_s, end_s FROM transcript_words WHERE segment_id = ?1 ORDER BY idx")?;
    let mut out = Vec::with_capacity(rows.len());
    for (id, mut s) in rows {
        s.words = words
            .query_map([id], |r| Ok(WordIn { text: r.get(0)?, start: r.get(1)?, end: r.get(2)? }))?
            .collect::<Result<_, _>>()?;
        out.push(s);
    }
    Ok(out)
}

/// Build the sentences of a video when it has a transcript and none are stored. Cheap when they
/// are: one indexed lookup.
pub fn ensure(db: &Db, video_id: i64) -> Result<(), Error> {
    let have: bool = db.conn.query_row(
        "SELECT (EXISTS (SELECT 1 FROM transcript_sentences WHERE video_id = ?1)
                 AND COALESCE((SELECT sentence_split FROM videos WHERE id = ?1), 0) >= ?2)
             OR NOT EXISTS (SELECT 1 FROM transcript_segments WHERE video_id = ?1)",
        params![video_id, SPLIT_VERSION],
        |r| r.get(0),
    )?;
    if have {
        return Ok(());
    }
    rebuild(db, video_id)
}

/// Split a video's transcript into sentences again and store them, replacing what it had. Its
/// chapters are kept: each one's first, last and standout sentences are found again by time among
/// the new sentences (a chapter's title and summary cost a model run; its sentence ids do not).
pub fn rebuild(db: &Db, video_id: i64) -> Result<(), Error> {
    let segs = load_segments(db, video_id)?;
    let regions = crate::speech::speech_regions(db, video_id, f64::MIN, f64::MAX)?;
    let sentences = split(&segs, &regions);
    // The chapters, with the times of the sentences they name, before those go.
    let old: std::collections::HashMap<i64, (f64, f64)> = {
        let mut st = db.conn.prepare("SELECT id, start_s, end_s FROM transcript_sentences WHERE video_id = ?1")?;
        st.query_map([video_id], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?.collect::<Result<_, _>>()?
    };
    let mut chapters = crate::chapters::load(db, video_id)?;
    let tx = db.conn.unchecked_transaction()?;
    tx.execute("DELETE FROM chapters WHERE video_id = ?1", [video_id])?;
    tx.execute("DELETE FROM transcript_sentences WHERE video_id = ?1", [video_id])?;
    let mut new: Vec<(i64, f64, f64)> = Vec::with_capacity(sentences.len());
    {
        let mut st = tx.prepare(
            "INSERT INTO transcript_sentences(video_id, start_s, end_s, text, words_json) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for s in &sentences {
            let words = s.words.as_ref().map(|w| serde_json::to_string(w).unwrap_or_default());
            st.execute(params![video_id, s.start, s.end, s.text, words])?;
            new.push((tx.last_insert_rowid(), s.start, s.end));
        }
    }
    tx.execute("UPDATE videos SET sentence_split = ?2 WHERE id = ?1", params![video_id, SPLIT_VERSION])?;
    if !chapters.is_empty() && !new.is_empty() {
        // The new sentence at `t`: the one holding it, else the nearest.
        let at = |t: f64| {
            new.iter()
                .min_by(|a, b| {
                    let d = |s: &(i64, f64, f64)| {
                        if s.1 <= t && t <= s.2 { 0.0 } else { (s.1 - t).abs().min((s.2 - t).abs()) }
                    };
                    d(a).total_cmp(&d(b))
                })
                .map(|s| s.0)
                .expect("not empty")
        };
        for c in &mut chapters {
            c.first_sentence_id = at(c.start_s + 0.01);
            c.last_sentence_id = at(c.end_s - 0.01);
            let mut standout: Vec<i64> =
                c.standout.iter().filter_map(|id| old.get(id)).map(|(a, b)| at((a + b) / 2.0)).collect();
            standout.dedup();
            c.standout = standout;
        }
        crate::chapters::insert_all(&tx, video_id, &chapters)?;
    }
    tx.commit()?;
    Ok(())
}

/// Every sentence of a video as (start, end), in order: the edges a cut may land on. Built first
/// when the video has none yet.
///
/// Mic changes already split sentences when [`split`] builds `transcript_sentences`, and every
/// off-mic mutation invalidates them so [`ensure`] rebuilds them cleanly.
pub fn edges(db: &Db, video_id: i64) -> Vec<(f64, f64)> {
    if ensure(db, video_id).is_err() {
        return Vec::new();
    }
    db.conn
        .prepare_cached("SELECT start_s, end_s FROM transcript_sentences WHERE video_id = ?1 ORDER BY start_s")
        .and_then(|mut st| st.query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?)))?.collect())
        .unwrap_or_default()
}

/// How much of a sentence a range has to play before stopping in it counts as cutting into it (s).
/// Below this it is the deliberate overrun past the last word, not an interruption.
pub const CUT_INTO_SENTENCE_S: f64 = 0.6;

/// Whether a range ending at `out_s` stops inside a sentence rather than at the end of one.
pub fn cuts_into_a_sentence(edges: &[(f64, f64)], out_s: f64) -> bool {
    edges.iter().any(|&(start, end)| end > out_s + 0.25 && start < out_s - CUT_INTO_SENTENCE_S)
}

/// Forget a video's sentences (its transcript or measured speech changed); they are built again on
/// first use. The chapters go with them.
pub fn invalidate(conn: &rusqlite::Connection, video_id: i64) -> Result<(), Error> {
    conn.execute("DELETE FROM transcript_sentences WHERE video_id = ?1", [video_id])?;
    conn.execute("DELETE FROM jobs WHERE video_id = ?1 AND stage = 'chapters'", [video_id])?;
    Ok(())
}

/// Tests: one sentence per stored segment, with the segment's id, so a fixture that writes
/// segments as whole sentences can quote `s<segment id>`.
#[cfg(test)]
pub fn mirror_segments(db: &Db, video_id: i64) {
    db.conn.execute("DELETE FROM transcript_sentences WHERE video_id = ?1", [video_id]).unwrap();
    db.conn
        .execute(
            "INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text)
             SELECT id, video_id, start_s, end_s, text FROM transcript_segments WHERE video_id = ?1",
            [video_id],
        )
        .unwrap();
    // A segment's own timed words become its sentence's words.
    db.conn
        .execute(
            "UPDATE transcript_sentences SET words_json = (
                 SELECT json_group_array(json_array(w.start_s, w.end_s))
                   FROM (SELECT start_s, end_s FROM transcript_words WHERE segment_id = transcript_sentences.id
                          ORDER BY idx) w)
              WHERE video_id = ?1 AND EXISTS (SELECT 1 FROM transcript_words WHERE segment_id = transcript_sentences.id)",
            [video_id],
        )
        .unwrap();
    // Built as the current rules would want them left alone.
    db.conn.execute("UPDATE videos SET sentence_split = ?2 WHERE id = ?1", params![video_id, SPLIT_VERSION]).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sentence_running_from_the_subject_into_the_interviewer_is_two_edges() {
        let a = seg(0.0, 6.0, "the subject is speaking here");
        let mut b = seg(6.3, 10.0, "okay, perfect");
        b.off_mic = true;
        let c = seg(11.0, 14.0, "back to the subject again.");
        // On-mic -> off-mic -> on-mic splits even when every speaker is None.
        let out = split(&[a.clone(), b.clone(), c.clone()], &[]);
        assert_eq!(
            out.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            vec![(0.0, 6.0), (6.3, 10.0), (11.0, 14.0)]
        );
        // All on the microphone: joined into one unpunctuated thought until the final stop.
        b.off_mic = false;
        assert_eq!(split(&[a, b, c], &[]).len(), 1);

        // And through DB ensure()/edges() with speaker = NULL:
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 20.0)", []).unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 0.0, 6.0, 'the subject is speaking here', 0),
                 (1, 6.3, 10.0, 'okay, perfect', 1),
                 (1, 11.0, 14.0, 'back to the subject again.', 0);",
            )
            .unwrap();
        assert_eq!(edges(&db, 1), vec![(0.0, 6.0), (6.3, 10.0), (11.0, 14.0)]);
    }

    fn seg(start: f64, end: f64, text: &str) -> SegIn {
        SegIn { start, end, text: text.into(), speaker: None, off_mic: false, words: Vec::new() }
    }

    fn mic(start: f64, end: f64, text: &str, off_mic: bool) -> SegIn {
        SegIn { off_mic, ..seg(start, end, text) }
    }

    /// A word or two that finish the speaker's sentence at the head of the interviewer's segment
    /// go back to the speaker; a capitalised line of the interviewer's own does not, and neither
    /// does a head after a sentence that had already ended.
    #[test]
    fn the_end_of_a_sentence_across_a_mic_change_goes_back_to_its_speaker() {
        let segs = vec![
            mic(73.68, 81.47, "People are nice and the southern hospitality, that's a real", false),
            mic(81.47, 87.3, "thing. So this event there are a lot of business owners, why is it", true),
        ];
        let out = split(&segs, &[]);
        assert!(
            texts(&out).contains(&"People are nice and the southern hospitality, that's a real thing."),
            "{:?}",
            texts(&out)
        );
        let moved = carry_tails(&segs);
        assert!(moved[1].text.starts_with("So this event"), "{:?}", moved[1].text);
        assert!(moved[0].end > 81.6 && moved[0].end < 82.5 && moved[1].start == moved[0].end, "{moved:?}");

        // The interviewer's own short line, and a head after a finished sentence, stay.
        for segs in [
            vec![
                mic(0.0, 4.0, "and that is how we met", false),
                mic(4.0, 6.0, "Okay. And what about the venue?", true),
            ],
            vec![mic(0.0, 4.0, "That is how we met.", false), mic(4.0, 6.0, "right. And the venue?", true)],
            vec![mic(0.0, 4.0, "and that is how we", false), mic(4.0, 6.0, "met at the event in August.", true)],
        ] {
            assert!(matches!(carry_tails(&segs), std::borrow::Cow::Borrowed(_)), "{segs:?}");
        }
    }

    fn texts(out: &[SentenceOut]) -> Vec<&str> {
        out.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn fragments_join_up_to_the_full_stop() {
        let segs = [
            seg(15.38, 17.49, "y te digo, y te soy honesto"),
            seg(17.49, 18.93, "la gente que usa Hermes"),
            seg(18.93, 20.89, "lo usa de esa manera."),
            seg(20.89, 23.17, "Todo el tiempo le están diciendo corregite."),
        ];
        let out = split(&segs, &[]);
        assert_eq!(
            texts(&out),
            [
                "y te digo, y te soy honesto la gente que usa Hermes lo usa de esa manera.",
                "Todo el tiempo le están diciendo corregite."
            ]
        );
        assert_eq!((out[0].start, out[0].end), (15.38, 20.89));
        assert_eq!(out[1].start, 20.89);
    }

    #[test]
    fn a_run_on_splits_at_its_stops_and_each_boundary_lands_on_the_pause() {
        // 20 s segment, three sentences; the speech pauses at 6.1-6.6 and 13.0-13.5.
        let segs = [seg(
            0.0,
            20.0,
            "La primera versión del thumbnail fue esta. Esto me hubiera servido perfectamente bien. Y él normalmente hace dos versiones distintas.",
        )];
        let regions = [SpeechRegion::new(0.0, 6.1), SpeechRegion::new(6.6, 13.0), SpeechRegion::new(13.5, 20.0)];
        let out = split(&segs, &regions);
        assert_eq!(out.len(), 3, "{out:?}");
        assert!(out[0].text.ends_with("esta.") && out[1].text.starts_with("Esto"));
        assert_eq!((out[0].end, out[1].start), (6.1, 6.6));
        assert_eq!((out[1].end, out[2].start), (13.0, 13.5));
        assert_eq!((out[0].start, out[2].end), (0.0, 20.0));
    }

    #[test]
    fn no_punctuation_falls_back_to_long_pauses_and_ignores_short_ones() {
        let segs = [
            seg(0.0, 2.0, "pero si yo como que"),
            seg(2.0, 4.0, "esto es muy ladilla"),
            seg(4.0, 6.0, "entonces this is not"),
            seg(6.0, 8.0, "the way y te digo"),
        ];
        // A 0.2 s breath at 2.0 (no break), a 0.9 s pause at 4.0 (break).
        let regions = [SpeechRegion::new(0.0, 1.9), SpeechRegion::new(2.1, 3.7), SpeechRegion::new(4.6, 8.0)];
        let out = split(&segs, &regions);
        assert_eq!(texts(&out), ["pero si yo como que esto es muy ladilla", "entonces this is not the way y te digo"]);
        assert_eq!((out[0].end, out[1].start), (3.7, 4.6));
    }

    #[test]
    fn an_umm_the_transcript_does_not_write_belongs_to_neither_sentence() {
        // Mary Ellen: "...for our growth." ends at 58.08, then 0.8 s of "uh-mm" Whisper left out,
        // then "It's just a great community." from 59.3. The pause nearest the estimate is the one
        // after the "umm", which made it the first sentence's last sound and every cut on it end
        // there.
        let segs = [
            seg(48.81, 58.08, "We share the same goals for our children and for our growth."),
            seg(59.3, 61.18, "It's just a great community to live in."),
        ];
        let regions =
            [SpeechRegion::new(48.9, 58.03), SpeechRegion::new(58.12, 58.92), SpeechRegion::new(59.45, 61.17)];
        let out = split(&segs, &regions);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!((out[0].end, out[1].start), (58.03, 59.45));
    }

    #[test]
    fn one_pause_makes_one_break_however_many_word_estimates_are_near_it() {
        // One long unpunctuated segment; a single 1 s pause in the middle.
        let words: Vec<String> = (0..40).map(|i| format!("w{i}")).collect();
        let segs = [seg(0.0, 20.0, &words.join(" "))];
        let regions = [SpeechRegion::new(0.0, 9.5), SpeechRegion::new(10.5, 20.0)];
        let out = split(&segs, &regions);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!((out[0].end, out[1].start), (9.5, 10.5));
    }

    #[test]
    fn a_pause_inside_a_punctuated_sentence_is_a_breath() {
        let segs = [seg(0.0, 3.0, "De lo que estoy viendo, esto nos está"), seg(3.0, 6.0, "costando diez dólares.")];
        let regions = [SpeechRegion::new(0.0, 2.6), SpeechRegion::new(3.4, 6.0)];
        assert_eq!(split(&segs, &regions).len(), 1);
    }

    #[test]
    fn a_speaker_change_between_segments_ends_the_sentence() {
        let mut a = seg(0.0, 2.0, "y entonces lo que hicimos");
        let mut b = seg(2.0, 3.0, "exacto exacto");
        a.speaker = Some(1);
        b.speaker = Some(2);
        assert_eq!(split(&[a.clone(), b.clone()], &[]).len(), 2);
        // A segment nobody was matched to between them does not hide the change.
        let mut c = seg(3.0, 4.0, "y tu loop no funciona así");
        c.speaker = Some(1);
        let lone = seg(2.0, 2.5, "terrible ok");
        let mut b2 = b;
        b2.end = 2.0;
        let out = split(&[a, b2, lone, c], &[]);
        assert_eq!(
            texts(&out),
            ["y entonces lo que hicimos", "exacto exacto terrible ok", "y tu loop no funciona así"]
        );
    }

    /// One speaker's lines come back punctuated and the other's not: the first one's full stops
    /// say nothing about the second's, whose measured pauses end its thoughts.
    #[test]
    fn a_pause_ends_an_unpunctuated_voice_next_to_a_punctuated_one() {
        let mut a = seg(0.0, 4.0, "Esto nos cuesta diez dólares, ¿ok? Mi tiempo es lo más caro.");
        a.speaker = Some(1);
        let mut b = seg(4.2, 12.0, "aquí hay algo interesante que es el research no hay manera de hacerlo nosotros");
        b.speaker = Some(2);
        // Speaker 2 pauses 0.8 s after "research".
        let regions = [SpeechRegion::new(0.0, 4.0), SpeechRegion::new(4.2, 7.3), SpeechRegion::new(8.1, 12.0)];
        let out = split(&[a, b], &regions);
        // Two sentences each (without word timings, which word the pause falls after is estimated).
        assert_eq!(out.len(), 4, "{:?}", texts(&out));
        assert!(out[2].text.starts_with("aquí") && out[3].text.ends_with("nosotros"), "{:?}", texts(&out));
    }

    #[test]
    fn a_tag_question_talked_straight_through_does_not_end_the_thought() {
        // The B1 bench's "10 dólares" clip opened on "En API de Opus…": the "¿ok?" had split it
        // off the line that says what costs ten dollars.
        let segs = [
            seg(0.0, 6.0, "Esto nos está costando alrededor de 10 dólares episodio, ¿ok?"),
            seg(6.2, 12.0, "En API de Opus, lo cual uno podría pensar que es caro."),
            seg(12.4, 15.0, "¿No? Pues sí."),
        ];
        let out = split(&segs, &[]);
        assert_eq!(
            texts(&out)[0],
            "Esto nos está costando alrededor de 10 dólares episodio, ¿ok? En API de Opus, lo cual uno podría pensar que es caro."
        );
        // A tag followed by a clause that stands on its own still ends the sentence.
        let segs = [seg(0.0, 4.0, "Es caro, ¿no?"), seg(4.2, 8.0, "Pero lo vale.")];
        assert_eq!(split(&segs, &[]).len(), 2);
    }

    #[test]
    fn a_sentence_keeps_all_its_words_on_its_own_edges_when_the_timings_drift() {
        // B2/B3: the engine timed "De lo que estoy…" 0.6 s early, so looking words up by time
        // filed "De lo que" under the sentence before and "@n" counted from "estoy".
        let mut s = seg(0.0, 12.0, "Ya está. De lo que estoy viendo cuesta diez.");
        s.words = [
            ("Ya", 0.2, 0.5),
            ("está", 0.5, 1.0),
            ("De", 2.4, 2.6),
            ("lo", 2.6, 2.8),
            ("que", 2.8, 3.0),
            ("estoy", 3.0, 3.4),
            ("viendo", 3.4, 3.9),
            ("cuesta", 4.0, 4.5),
            ("diez", 4.5, 5.0),
        ]
        .map(|(t, a, b)| WordIn { text: t.into(), start: a, end: b })
        .to_vec();
        let regions = [
            SpeechRegion { start_s: 0.2, end_s: 1.0, quiet_after_s: None },
            SpeechRegion { start_s: 3.0, end_s: 5.6, quiet_after_s: None },
        ];
        let out = split(&[s], &regions);
        assert_eq!(texts(&out), ["Ya está.", "De lo que estoy viendo cuesta diez."]);
        let words = out[1].words.as_ref().expect("timed words");
        assert_eq!(words.len(), 7, "one time per word of the text");
        assert!((words[0].0 - out[1].start).abs() < 1e-6 && (words[6].1 - out[1].end).abs() < 1e-6, "{words:?}");
        assert!(words.windows(2).all(|w| w[0].0 <= w[1].0), "in order: {words:?}");
    }

    #[test]
    fn words_timed_in_the_silence_move_to_the_speech_and_the_rest_keep_their_times() {
        // The episode at 24:09: the engine put "De lo que estoy" in the silence before the voice
        // starts at 1449.91, then timed "viendo" and "Esto" right. Stretching every word onto the
        // sentence's edges put "Esto" half a second late and a cut on it lost three words.
        let mut s = seg(1447.5, 1453.0, "bien. De lo que estoy viendo, esto nos");
        s.words = [
            ("bien", 1447.5, 1449.18),
            ("De", 1449.22, 1449.37),
            ("lo", 1449.37, 1449.56),
            ("que", 1449.56, 1449.85),
            ("estoy", 1449.85, 1450.33),
            ("viendo", 1450.33, 1451.22),
            ("Esto", 1451.26, 1451.49),
            ("nos", 1451.49, 1451.69),
        ]
        .map(|(t, a, b)| WordIn { text: t.into(), start: a, end: b })
        .to_vec();
        let regions = [
            SpeechRegion { start_s: 1447.51, end_s: 1449.25, quiet_after_s: None },
            SpeechRegion { start_s: 1449.91, end_s: 1453.0, quiet_after_s: None },
        ];
        let out = split(&[s], &regions);
        let w = out[1].words.as_ref().unwrap();
        assert_eq!(out[1].text, "De lo que estoy viendo, esto nos");
        assert!((w[0].0 - 1449.91).abs() < 0.01, "first word where the voice starts: {w:?}");
        assert!(w[1].0 > w[0].0 && w[3].0 < 1450.33, "the piled-up words spread before 'viendo': {w:?}");
        assert!((w[5].0 - 1451.26).abs() < 0.01, "'esto' keeps the engine's time: {w:?}");
    }

    #[test]
    fn abbreviations_and_numbers_do_not_end_a_sentence() {
        let out = split(&[seg(0.0, 6.0, "Hablamos con el Dr. Pérez sobre el punto 2. y más cosas.")], &[]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(ends_sentence("así.") && ends_sentence("¿no?") && ends_sentence("vale.»") && !ends_sentence("hola,"));
    }

    #[test]
    fn a_sentence_too_long_is_split_at_a_comma_near_the_middle() {
        let words = (0..60).map(|i| if i == 29 { "palabra,".to_string() } else { format!("w{i}") }).collect::<Vec<_>>();
        let out = split(&[seg(0.0, 40.0, &words.join(" "))], &[]);
        assert!(out.len() >= 2, "{out:?}");
        assert!(out.iter().all(|s| s.end - s.start <= MAX_SENTENCE_S + 0.01), "{out:?}");
        assert!(out.iter().any(|s| s.text.ends_with("palabra,")), "the comma is one of the splits: {out:?}");
        // Never right before a tag question: it belongs to the line it closes.
        let words = (0..60)
            .map(|i| match i {
                29 => "palabra,".to_string(),
                30 => "¿ok?".to_string(),
                _ => format!("w{i}"),
            })
            .collect::<Vec<_>>();
        let out = split(&[seg(0.0, 40.0, &words.join(" "))], &[]);
        assert!(!out.iter().any(|s| s.text.starts_with("¿ok?")), "{out:?}");
    }

    #[test]
    fn word_timings_when_given_place_the_split_exactly() {
        let mut s = seg(0.0, 10.0, "Uno dos. Tres cuatro.");
        s.words = [("Uno", 0.0, 0.4), ("dos", 0.5, 1.0), ("Tres", 8.0, 8.5), ("cuatro", 8.6, 9.0)]
            .map(|(t, a, b)| WordIn { text: t.into(), start: a, end: b })
            .to_vec();
        let out = split(&[s], &[]);
        assert_eq!((out[0].end, out[1].start), (1.0, 8.0));
    }

    #[test]
    fn a_pause_on_the_far_side_of_a_timed_word_does_not_take_the_word_with_it() {
        // "…episodio. ¿Ok? Pues…": the measured pause sits before "Ok", whose timing says it
        // belongs to the first sentence. Snapping to that pause handed "Ok" to the next one.
        let mut s = seg(0.0, 4.0, "Cuesta diez, ¿ok? Pues sí.");
        s.words = [("Cuesta", 0.0, 0.5), ("diez", 0.5, 1.0), ("ok", 1.1, 1.5), ("Pues", 1.55, 1.9), ("sí", 1.9, 2.3)]
            .map(|(t, a, b)| WordIn { text: t.into(), start: a, end: b })
            .to_vec();
        let regions = [
            SpeechRegion { start_s: 0.0, end_s: 1.0, quiet_after_s: None },
            SpeechRegion { start_s: 1.2, end_s: 2.3, quiet_after_s: None },
        ];
        let out = split(&[s], &regions);
        assert_eq!(texts(&out), ["Cuesta diez, ¿ok?", "Pues sí."]);
        assert!(out[1].start >= 1.5, "{out:?}");
    }

    #[test]
    fn another_engines_words_time_the_text_even_where_they_heard_it_differently() {
        let w = |t: &str, a: f64, b: f64| WordIn { text: t.into(), start: a, end: b };
        // The text says "10, 15 dólares"; the other engine heard "10. 15 dolares" and an extra "eh".
        let text = ["pago", "como", "10,", "15", "dólares", "al", "mes."];
        let timed = [
            w("Pago", 1.0, 1.3),
            w("como", 1.35, 1.6),
            w("eh", 1.6, 1.8),
            w("10.", 1.9, 2.2),
            w("15", 2.3, 2.6),
            w("dolares", 2.7, 3.1),
            w("al", 3.2, 3.3),
            w("mes.", 3.35, 3.7),
        ];
        let t = align(&text, &timed, 0.9, 3.8);
        assert_eq!(t[0], (1.0, 1.3));
        assert_eq!(t[2], (1.9, 2.2), "punctuation does not stop a match");
        // "dólares" did not match: it fills the gap between "15" and "al".
        assert_eq!(t[4], (2.6, 3.2));
        assert_eq!(t[6], (3.35, 3.7));
        // No timed words: the span is shared out.
        let none = align(&text, &[], 0.0, 7.0);
        assert_eq!(none.first().unwrap().0, 0.0);
        assert_eq!(none.last().unwrap().1, 7.0);
    }

    #[test]
    fn ensure_builds_once_and_invalidate_forgets() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 30.0)", []).unwrap();
        for (a, b, t) in [(0.0, 2.0, "Hola a todos."), (2.0, 4.0, "Bienvenidos al"), (4.0, 6.0, "programa de hoy.")] {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, ?1, ?2, ?3)",
                    params![a, b, t],
                )
                .unwrap();
        }
        ensure(&db, 1).unwrap();
        let count = |db: &Db| -> i64 {
            db.conn.query_row("SELECT COUNT(*) FROM transcript_sentences WHERE video_id = 1", [], |r| r.get(0)).unwrap()
        };
        assert_eq!(count(&db), 2);
        let first_id: i64 = db.conn.query_row("SELECT MIN(id) FROM transcript_sentences", [], |r| r.get(0)).unwrap();
        ensure(&db, 1).unwrap();
        let again: i64 = db.conn.query_row("SELECT MIN(id) FROM transcript_sentences", [], |r| r.get(0)).unwrap();
        assert_eq!(first_id, again, "a second ensure keeps the ids");
        invalidate(&db.conn, 1).unwrap();
        assert_eq!(count(&db), 0);
        // A video with no transcript has nothing to build.
        ensure(&db, 2).unwrap();
    }

    #[test]
    fn rebuild_preserves_chapters_on_load_or_store_failure() {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 30.0)", []).unwrap();
        for (a, b, t) in [(0.0, 2.0, "Hola a todos."), (2.0, 6.0, "Bienvenidos al programa de hoy.")] {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, ?1, ?2, ?3)",
                    params![a, b, t],
                )
                .unwrap();
        }
        rebuild(&db, 1).unwrap();
        let old_ids: Vec<i64> = {
            let mut st =
                db.conn.prepare("SELECT id FROM transcript_sentences WHERE video_id = 1 ORDER BY start_s").unwrap();
            st.query_map([], |r| r.get(0)).unwrap().collect::<Result<_, _>>().unwrap()
        };
        assert_eq!(old_ids.len(), 2);

        let chapter = crate::chapters::Chapter {
            video_id: 1,
            idx: 0,
            start_s: 0.0,
            end_s: 6.0,
            first_sentence_id: old_ids[0],
            last_sentence_id: old_ids[1],
            title: "Model Title".into(),
            summary: "Model Summary".into(),
            standout: vec![old_ids[0]],
            titled: true,
        };
        crate::chapters::store(&db, 1, &[chapter]).unwrap();

        // 1. A failure while storing remapped chapters rolls back the whole rebuild.
        db.conn
            .execute(
                "CREATE TRIGGER fail_chapter_insert BEFORE INSERT ON chapters BEGIN SELECT RAISE(ABORT, 'boom'); END;",
                [],
            )
            .unwrap();
        assert!(rebuild(&db, 1).is_err());
        let after_abort = crate::chapters::load(&db, 1).unwrap();
        assert_eq!(after_abort.len(), 1);
        assert_eq!(after_abort[0].title, "Model Title");
        assert!(after_abort[0].titled);
        assert_eq!(after_abort[0].first_sentence_id, old_ids[0]);
        db.conn.execute("DROP TRIGGER fail_chapter_insert", []).unwrap();

        // 2. A successful rebuild remaps the chapter onto the new sentence ids inside the transaction.
        rebuild(&db, 1).unwrap();
        let remapped = crate::chapters::load(&db, 1).unwrap();
        assert_eq!(remapped.len(), 1);
        assert_eq!(remapped[0].title, "Model Title");
        assert!(remapped[0].titled);
        assert_ne!(remapped[0].first_sentence_id, old_ids[0]);
        assert_eq!(remapped[0].standout, vec![remapped[0].first_sentence_id]);

        // 3. A failure loading chapters propagates instead of deleting chapters.
        db.conn.execute("ALTER TABLE chapters RENAME TO chapters_bak", []).unwrap();
        assert!(rebuild(&db, 1).is_err());
        db.conn.execute("ALTER TABLE chapters_bak RENAME TO chapters", []).unwrap();
        assert_eq!(crate::chapters::load(&db, 1).unwrap().len(), 1);
    }
}
