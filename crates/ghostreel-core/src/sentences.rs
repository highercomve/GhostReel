//! Sentence ids: how a model names a stretch of speech without quoting seconds.
//!
//! Every sentence of a transcript ([`crate::sentence_split`] builds them from Whisper's segments)
//! has a stable id, `s<sentence row id>` (`s412`). The id is the row id because it is unique across
//! the library, so a sentence of another video is recognised as such; it is stable for as long as
//! the video's transcript and measured speech stay as they are. A per-video ordinal would be
//! shorter but could not tell videos apart.
//!
//! A script clip (or an audio bed) may be written `{video_id, from: "s412", to: "s418"}`. Code
//! resolves it to seconds, from the first sentence's start to the last sentence's end, and the
//! normal repair chain then places the exact cuts from the measured speech ([`crate::cuts`]).
//! `from: "s412@3"` starts at the third word of the sentence, `to: "s412@5"` ends with its fifth;
//! that needs word timings (`transcript_words`, kept per Whisper segment), which the speech engine
//! gives when it can, and the time is then put on the nearest edge of the measured speech.

use rusqlite::params;
use serde::Serializer;

use crate::db::Db;
use crate::script::{Script, SentenceRange};

/// The id a sentence is quoted by.
pub fn format_id(sentence_id: i64) -> String {
    format!("s{sentence_id}")
}

/// Seconds as `minutes:seconds.tenths` (`12:03.4`; minutes go past 59, a 70-minute episode ends at
/// `70:12.0`): the time the compact transcripts and cut readings quote.
pub fn clock(t: f64) -> String {
    let tenths = (t.max(0.0) * 10.0).round() as u64;
    format!("{}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10)
}

/// Serialises a sentence row id as its sentence id (`#[serde(serialize_with = ...)]`).
pub fn serialize_id<S: Serializer>(id: &i64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format_id(*id))
}

/// How far from a word's recorded time the speech's own edge may be to take its place (s).
/// Word timestamps are approximate; a pause's edge is measured.
const WORD_SNAP_S: f64 = 0.25;

/// Why a sentence id could not be used. Every message says what to do instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SentenceError {
    #[error("'{text}' is not a sentence id: write it as get_transcript shows it (s412), or s412@3 for the third word")]
    Malformed { text: String },
    #[error(
        "sentence {id} does not exist (the video may have been transcribed again since): read the transcript \
         again with get_transcript and use the ids it shows"
    )]
    NotFound { id: String },
    #[error(
        "sentence {id} is in video #{found_in}, not video #{video_id}: a clip takes the sentences of its own \
         video. Use video_id {found_in}, or pick a sentence of video #{video_id} from get_transcript"
    )]
    OtherVideo { id: String, found_in: i64, video_id: i64 },
    #[error("from {from} comes after to {to}: swap them, or give a `to` at or after `from`")]
    Reversed { from: String, to: String },
    #[error(
        "{id}@{word}: video #{video_id} has no word timings (its speech engine gave none, or it was transcribed \
         before they were kept): cut at the whole sentence ({id}) instead"
    )]
    NoWordTimings { id: String, word: u32, video_id: i64 },
    #[error("{id}@{word}: the sentence has {words} words; use @1 to @{words}")]
    WordOutOfRange { id: String, word: u32, words: usize },
    /// One of the above, found in a particular clip.
    #[error("{place}: {error}")]
    At { place: String, error: Box<SentenceError> },
}

impl From<SentenceError> for crate::Error {
    fn from(e: SentenceError) -> Self {
        crate::Error::Invalid(e.to_string())
    }
}

/// A sentence, and optionally the n-th word (1-based) of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceRef {
    pub sentence_id: i64,
    pub word: Option<u32>,
}

impl SentenceRef {
    /// `s412` or `S412`, optionally followed by `@3`.
    pub fn parse(text: &str) -> Result<Self, SentenceError> {
        let malformed = || SentenceError::Malformed { text: text.to_string() };
        let t = text.trim();
        let (id_part, word_part) = match t.split_once('@') {
            Some((a, b)) => (a, Some(b)),
            None => (t, None),
        };
        let digits = id_part.strip_prefix(['s', 'S']).ok_or_else(malformed)?;
        let sentence_id: i64 = digits.parse().map_err(|_| malformed())?;
        let word = match word_part {
            None => None,
            Some(w) => match w.trim().parse::<u32>() {
                Ok(n) if n >= 1 => Some(n),
                _ => return Err(malformed()),
            },
        };
        if sentence_id <= 0 {
            return Err(malformed());
        }
        Ok(Self { sentence_id, word })
    }

    pub fn id(&self) -> String {
        format_id(self.sentence_id)
    }
}

impl std::fmt::Display for SentenceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.word {
            Some(w) => write!(f, "s{}@{w}", self.sentence_id),
            None => write!(f, "s{}", self.sentence_id),
        }
    }
}

/// One end of a transcript or manifest range (`from`/`to`): either a sentence id (`s412`,
/// `S412@3`) or an explicit time in seconds (`15.75`, `412s`, or a JSON number). Bare integer
/// strings like `"412"` are rejected as ambiguous because the caller could mean sentence `s412` or
/// `412` seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeRef {
    Sentence(SentenceRef),
    Seconds(f64),
}

impl EdgeRef {
    pub fn parse(text: &str) -> Result<Self, String> {
        let t = text.trim();
        if t.is_empty() {
            return Err("empty range edge: write a sentence id like 's412', or seconds like '412s' or '412.0'".into());
        }
        if t.starts_with(['s', 'S']) {
            return SentenceRef::parse(t).map(EdgeRef::Sentence).map_err(|e| e.to_string());
        }
        if let Some(rest) = t.strip_suffix(['s', 'S']) {
            let secs: f64 = rest
                .trim()
                .parse()
                .map_err(|_| format!("'{t}' is not a valid time in seconds: write e.g. 's412', '412s', or '412.0'"))?;
            if !secs.is_finite() || secs < 0.0 {
                return Err(format!(
                    "'{t}' is not a finite non-negative time in seconds: write e.g. 's412', '412s', or '412.0'"
                ));
            }
            return Ok(EdgeRef::Seconds(secs));
        }
        if !t.contains('.') {
            if let Ok(n) = t.parse::<i64>() {
                if n >= 0 {
                    return Err(format!(
                        "'{t}' is ambiguous: write 's{n}' for a sentence id, or '{n}s' / '{n}.0' for seconds"
                    ));
                }
                return Err(format!(
                    "'{t}' is not a valid sentence id or non-negative time: write 's412' for a sentence id, or '412s' / '412.0' for seconds"
                ));
            }
            return Err(format!(
                "'{t}' is neither a sentence id nor seconds: write 's412' for a sentence id, or '412s' / '412.0' for seconds"
            ));
        }
        let secs: f64 = t.parse().map_err(|_| {
            format!(
                "'{t}' is not a valid time in seconds: write 's412' for a sentence id, or '412s' / '412.0' for seconds"
            )
        })?;
        if !secs.is_finite() || secs < 0.0 {
            return Err(format!(
                "'{t}' is not a finite non-negative time in seconds: write e.g. 's412', '412s', or '412.0'"
            ));
        }
        Ok(EdgeRef::Seconds(secs))
    }

    pub fn parse_value(v: &serde_json::Value) -> Result<Self, String> {
        if let Some(secs) = v.as_f64() {
            if !secs.is_finite() || secs < 0.0 {
                return Err(format!("{secs} is not a finite non-negative time in seconds"));
            }
            return Ok(EdgeRef::Seconds(secs));
        }
        if let Some(s) = v.as_str() {
            return Self::parse(s);
        }
        Err(format!("invalid range edge {v}: write a sentence id like 's412' or seconds like 15.75"))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Edge {
    Start,
    End,
}

/// The video and bounds of a sentence.
fn segment(db: &Db, r: &SentenceRef) -> Result<(i64, f64, f64), SentenceError> {
    db.conn
        .query_row("SELECT video_id, start_s, end_s FROM transcript_sentences WHERE id = ?1", [r.sentence_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(|_| SentenceError::NotFound { id: r.id() })
}

/// The seconds of one end of a reference, inside `video_id`.
fn seconds(db: &Db, video_id: i64, r: &SentenceRef, edge: Edge) -> Result<f64, SentenceError> {
    let (found_in, start, end) = segment(db, r)?;
    if found_in != video_id {
        return Err(SentenceError::OtherVideo { id: r.id(), found_in, video_id });
    }
    let Some(word) = r.word else {
        return Ok(if edge == Edge::Start { start } else { end });
    };
    let words = own_words(db, r.sentence_id);
    if words.is_empty() {
        return Err(SentenceError::NoWordTimings { id: r.id(), word, video_id });
    }
    let Some(&(w_start, w_end)) = words.get(word as usize - 1) else {
        return Err(SentenceError::WordOutOfRange { id: r.id(), word, words: words.len() });
    };
    Ok(match edge {
        Edge::Start => on_speech_edge(db, video_id, w_start, Edge::Start),
        Edge::End => on_speech_edge(db, video_id, w_end, Edge::End),
    })
}

/// The neighbours of a clip written to start or end on a word ("s412@3"): for the start, where
/// the word before it ends and where it starts; for the end, where it ends and where the word after
/// it starts. A word inside running speech has no pause around it, and these say how far a cut may
/// move without taking a neighbour's sound.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WordEdges {
    pub start: Option<(f64, f64)>,
    pub end: Option<(f64, f64)>,
    /// The two words around each boundary from the first one's start to the second one's end: the
    /// boundary is somewhere in there, and never outside it (before the first word is the pause
    /// before the sentence, not the gap between these two).
    pub start_span: Option<(f64, f64)>,
    pub end_span: Option<(f64, f64)>,
}

impl WordEdges {
    /// Also a whole-sentence edge, when the neighbouring sentence touches it with no pause in the
    /// measured speech (a reply that starts on the last word, two sentences in one breath): then
    /// that edge is a boundary between two words as well.
    pub fn of(db: &Db, range: Option<&crate::script::SentenceRange>) -> Self {
        let Some(range) = range else { return Self::default() };
        // (end of the first word, start of the second), (start of the first, end of the second).
        let near = |text: &str, before: bool| -> Option<((f64, f64), (f64, f64))> {
            let r = SentenceRef::parse(text).ok()?;
            let words = own_words(db, r.sentence_id);
            match r.word {
                Some(k) => {
                    let k = k as usize;
                    let this = *words.get(k.checked_sub(1)?)?;
                    let inside = if before {
                        k.checked_sub(2).and_then(|i| words.get(i)).map(|w| ((w.1, this.0), (w.0, this.1)))
                    } else {
                        words.get(k).map(|w| ((this.1, w.0), (this.0, w.1)))
                    };
                    // The first or last word of the sentence: its neighbour is in the next sentence.
                    inside.or_else(|| touching(db, r.sentence_id, before))
                }
                None => touching(db, r.sentence_id, before),
            }
        };
        let (s, e) = (near(&range.from, true), near(&range.to, false));
        Self { start: s.map(|x| x.0), end: e.map(|x| x.0), start_span: s.map(|x| x.1), end_span: e.map(|x| x.1) }
    }
}

/// The boundary between a sentence and the one before it (`before`) or after it, as (end of the
/// first's last word, start of the second's first word), when no pause of the measured speech
/// separates them. None when a pause does: a cut there is placed on the pause.
fn touching(db: &Db, sentence_id: i64, before: bool) -> Option<((f64, f64), (f64, f64))> {
    let (video_id, start, end): (i64, f64, f64) = db
        .conn
        .query_row("SELECT video_id, start_s, end_s FROM transcript_sentences WHERE id = ?1", [sentence_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .ok()?;
    let sql = if before {
        "SELECT id, end_s FROM transcript_sentences WHERE video_id = ?1 AND start_s < ?2 ORDER BY start_s DESC LIMIT 1"
    } else {
        "SELECT id, start_s FROM transcript_sentences WHERE video_id = ?1 AND start_s > ?2 ORDER BY start_s LIMIT 1"
    };
    let (other, edge): (i64, f64) =
        db.conn.query_row(sql, rusqlite::params![video_id, start], |r| Ok((r.get(0)?, r.get(1)?))).ok()?;
    let (a, b) = if before { (edge, start) } else { (end, edge) };
    // A pause between them in the measured speech: nothing to do here.
    let regions = crate::speech::speech_regions(db, video_id, a - 0.5, b + 0.5).ok()?;
    let continuous = regions.iter().any(|r| r.start_s < a.min(b) - 0.02 && r.end_s > a.max(b) + 0.02);
    if !continuous {
        return None;
    }
    let mine = own_words(db, sentence_id);
    let theirs = own_words(db, other);
    let (first, second) = if before { (theirs.last(), mine.first()) } else { (mine.last(), theirs.first()) };
    let (gap_a, gap_b) = (first.map_or(a, |w| w.1), second.map_or(b, |w| w.0));
    let span = (first.map_or(a - 0.3, |w| w.0), second.map_or(b + 0.3, |w| w.1));
    Some(((gap_a, gap_b), span))
}

/// A sentence's own words, one (start, end) per word of its text (`transcript_sentences.words_json`);
/// empty when its words were never timed.
pub fn own_words(db: &Db, sentence_id: i64) -> Vec<(f64, f64)> {
    db.conn
        .query_row("SELECT words_json FROM transcript_sentences WHERE id = ?1", [sentence_id], |r| {
            r.get::<_, Option<String>>(0)
        })
        .ok()
        .flatten()
        .and_then(|j| serde_json::from_str(&j).ok())
        .unwrap_or_default()
}

/// The timed words of the video spoken inside `start..end` (by their middle), in order.
pub fn sentence_words(db: &Db, video_id: i64, start: f64, end: f64) -> Vec<(f64, f64)> {
    db.conn
        .prepare(
            "SELECT w.start_s, w.end_s FROM transcript_words w JOIN transcript_segments s ON s.id = w.segment_id
              WHERE s.video_id = ?1 AND (w.start_s + w.end_s) / 2 >= ?2 AND (w.start_s + w.end_s) / 2 <= ?3
              ORDER BY w.start_s, w.idx",
        )
        .and_then(|mut st| {
            st.query_map(params![video_id, start - 0.05, end + 0.05], |row| Ok((row.get(0)?, row.get(1)?)))?.collect()
        })
        .unwrap_or_default()
}

/// A word's time put on the measured speech: the start of the speech region that begins nearest to
/// `t` (or the end of the region that ends nearest to it), when one is within [`WORD_SNAP_S`].
/// Without measured speech, or with none that close, the engine's time stands.
fn on_speech_edge(db: &Db, video_id: i64, t: f64, edge: Edge) -> f64 {
    let regions = crate::speech::speech_regions(db, video_id, t - WORD_SNAP_S, t + WORD_SNAP_S).unwrap_or_default();
    let edges = regions.iter().map(|r| if edge == Edge::Start { r.start_s } else { r.end_s });
    edges.filter(|e| (e - t).abs() <= WORD_SNAP_S).min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs())).unwrap_or(t)
}

/// The seconds `from` to `to` cover in `video_id`: the first sentence's start (or word's start) to
/// the last one's end (or word's end).
pub fn resolve_range(db: &Db, video_id: i64, from: &str, to: &str) -> Result<(f64, f64), SentenceError> {
    let (a, b) = (SentenceRef::parse(from)?, SentenceRef::parse(to)?);
    // The ids came from a transcript, which built the sentences; a caller that skipped it still gets
    // the same ones.
    let _ = crate::sentence_split::ensure(db, video_id);
    let start = seconds(db, video_id, &a, Edge::Start)?;
    let end = seconds(db, video_id, &b, Edge::End)?;
    if start >= end {
        return Err(SentenceError::Reversed { from: a.to_string(), to: b.to_string() });
    }
    Ok((start, end))
}

/// A clip's or bed's `from`/`to`, each a sentence id or seconds ([`EdgeRef`]): the seconds they
/// cover, and whether either was written in seconds. Every tool that reads a range takes seconds
/// there ("0.41", "412s", 15.75); a script's clips do too, so a writer that copied times from a
/// manifest gets its cut instead of a round spent on "not a sentence id".
fn resolve_edges(db: &Db, video_id: i64, from: &str, to: &str) -> Result<(f64, f64, bool), SentenceError> {
    match (EdgeRef::parse(from), EdgeRef::parse(to)) {
        (Ok(EdgeRef::Sentence(_)), Ok(EdgeRef::Sentence(_))) => {
            resolve_range(db, video_id, from, to).map(|(a, b)| (a, b, false))
        }
        (Ok(a), Ok(b)) => {
            let _ = crate::sentence_split::ensure(db, video_id);
            let at = |e: EdgeRef, edge: Edge| match e {
                EdgeRef::Seconds(s) => Ok(s),
                EdgeRef::Sentence(r) => seconds(db, video_id, &r, edge),
            };
            let (start, end) = (at(a, Edge::Start)?, at(b, Edge::End)?);
            if start >= end {
                return Err(SentenceError::Reversed { from: from.to_string(), to: to.to_string() });
            }
            Ok((start, end, true))
        }
        // Neither a sentence nor seconds: the sentence parser says how to write it.
        _ => resolve_range(db, video_id, from, to).map(|(a, b)| (a, b, false)),
    }
}

/// Resolve every clip and bed of `script` that was written with sentences to seconds, keeping the
/// range on the item. Items written with seconds are left as they are. Returns how many were
/// resolved; the first failure names the beat and clip.
/// The video and bounds `(video_id, start_s, end_s)` of a sentence id (ids are unique across the library).
fn sentence_info(db: &Db, text: &str) -> Option<(i64, f64, f64)> {
    let r = SentenceRef::parse(text).ok()?;
    db.conn
        .query_row("SELECT video_id, start_s, end_s FROM transcript_sentences WHERE id = ?1", [r.sentence_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .ok()
}

/// How far beyond the sentence's own bounds a resolved clip's trimmed/padded seconds may sit and
/// still clearly belong to that sentence span rather than another video's timeline (s).
const SPAN_TOL_S: f64 = 0.5;

fn outside_span(in_s: f64, out_s: f64, span_start: f64, span_end: f64) -> bool {
    in_s < span_start - SPAN_TOL_S || out_s > span_end + SPAN_TOL_S
}

/// Clips and beds whose sentences belong to another video than the one they name. A sentence id is
/// unique across the library, so the video it is in is not in doubt: a clip with sound takes that
/// video (re-resolved from its sentences); a muted clip with its own seconds keeps its pictures and
/// loses the stray range (it was b-roll, labelled with the line it plays under). An already-resolved
/// clip or bed whose seconds already sit inside the resolved sentence span is left alone so an old
/// saved script is never silently hijacked if an upstream id changes. Returns a note for each, so
/// the editor learns what was meant — a small model mixed these up in two drafts in a row, and each
/// time the whole draft was refused for one clip.
pub fn correct_sentence_videos(db: &Db, script: &mut Script) -> Vec<String> {
    let mut notes = Vec::new();
    let owner_and_span = |r: &SentenceRange| match (sentence_info(db, &r.from), sentence_info(db, &r.to)) {
        (Some((va, sa, ea)), Some((vb, sb, eb))) if va == vb => Some((va, sa.min(sb), ea.max(eb))),
        _ => None,
    };
    for beat in &mut script.beats {
        for (i, clip) in beat.clips.iter_mut().enumerate() {
            let Some((v, span_start, span_end)) =
                clip.sentences.as_ref().and_then(owner_and_span).filter(|(v, _, _)| *v != clip.video_id)
            else {
                continue;
            };
            if !clip.is_unresolved() && !outside_span(clip.in_s, clip.out_s, span_start, span_end) {
                continue;
            }
            let range = clip.sentences.as_ref().map(|r| format!("{}..{}", r.from, r.to)).unwrap_or_default();
            if clip.audio == crate::script::Audio::Mute && clip.out_s > clip.in_s {
                notes.push(format!(
                    "beat {} clip {}: muted, so kept its pictures of video #{} {:.1}-{:.1} s and dropped the sentences \
                     {range}, which are in video #{v}",
                    beat.id,
                    i + 1,
                    clip.video_id,
                    clip.in_s,
                    clip.out_s
                ));
                clip.sentences = None;
            } else {
                notes.push(format!(
                    "beat {} clip {}: sentences {range} are in video #{v}, not #{}: the clip now plays video #{v}",
                    beat.id,
                    i + 1,
                    clip.video_id
                ));
                clip.video_id = v;
                // Seconds written for the other video mean nothing here: the sentences decide.
                clip.in_s = 0.0;
                clip.out_s = 0.0;
            }
        }
        if let Some(bed) = beat.bed.as_mut()
            && let Some((v, span_start, span_end)) =
                bed.sentences.as_ref().and_then(owner_and_span).filter(|(v, _, _)| *v != bed.video_id)
            && (bed.is_unresolved() || outside_span(bed.in_s, bed.out_s, span_start, span_end))
        {
            notes.push(format!(
                "beat {} bed: its sentences are in video #{v}, not #{}: it now plays video #{v}",
                beat.id, bed.video_id
            ));
            bed.video_id = v;
            bed.in_s = 0.0;
            bed.out_s = 0.0;
        }
    }
    notes
}

/// Resolve every clip and bed written with sentence ids to seconds, after correcting the ones that
/// name another video than their sentences ([`correct_sentence_videos`]). Returns how many were
/// resolved; [`resolve_script_noting`] also returns what the correction did.
pub fn resolve_script(db: &Db, script: &mut Script) -> Result<usize, SentenceError> {
    resolve_script_noting(db, script).map(|(resolved, _)| resolved)
}

/// [`resolve_script`], with a note for each clip or bed whose video was corrected, so a caller can
/// tell the editor what was meant instead of changing its cut silently.
pub fn resolve_script_noting(db: &Db, script: &mut Script) -> Result<(usize, Vec<String>), SentenceError> {
    let mut notes = correct_sentence_videos(db, script);
    let mut resolved = 0;
    for beat in &mut script.beats {
        for (i, clip) in beat.clips.iter_mut().enumerate() {
            if !clip.is_unresolved() {
                continue;
            }
            let range: &SentenceRange = clip.sentences.as_ref().expect("unresolved clips have a range");
            let (a, b, in_seconds) = resolve_edges(db, clip.video_id, &range.from, &range.to).map_err(|e| {
                SentenceError::At { place: format!("beat {} clip {}", beat.id, i + 1), error: Box::new(e) }
            })?;
            clip.in_s = a;
            clip.out_s = b;
            if in_seconds {
                notes.push(seconds_note(&format!("beat {} clip {}", beat.id, i + 1), range, a, b));
                // Seconds are a range like any in_s/out_s: the finish puts it on the sentences.
                clip.sentences = None;
            }
            resolved += 1;
        }
        if let Some(bed) = beat.bed.as_mut().filter(|b| b.is_unresolved()) {
            let range: &SentenceRange = bed.sentences.as_ref().expect("unresolved beds have a range");
            let (a, b, in_seconds) = resolve_edges(db, bed.video_id, &range.from, &range.to)
                .map_err(|e| SentenceError::At { place: format!("beat {} bed", beat.id), error: Box::new(e) })?;
            bed.in_s = a;
            bed.out_s = b;
            if in_seconds {
                notes.push(seconds_note(&format!("beat {} bed", beat.id), range, a, b));
                bed.sentences = None;
            }
            resolved += 1;
        }
    }
    Ok((resolved, notes))
}

/// What a range written in seconds was taken as, for the editor.
fn seconds_note(place: &str, range: &SentenceRange, a: f64, b: f64) -> String {
    format!(
        "{place}: from/to \"{}\"–\"{}\" were read as seconds ({a:.2}–{b:.2} s); sentence ids (s412) cut on whole \
         sentences",
        range.from, range.to
    )
}

/// Parse script JSON the way [`Script::parse_for_project`] does, then resolve its sentence ranges
/// to seconds. A sentence that cannot be used is a parse error whose message says how to fix it.
pub fn parse_for_project(db: &Db, json: &str, project: &crate::projects::Project) -> Result<Script, serde_json::Error> {
    let mut script = Script::parse_for_project(json, project)?;
    resolve_script(db, &mut script).map_err(<serde_json::Error as serde::de::Error>::custom)?;
    Ok(script)
}

/// The number of words with timings stored for a video.
pub fn word_count(db: &Db, video_id: i64) -> i64 {
    db.conn
        .query_row(
            "SELECT COUNT(*) FROM transcript_words w JOIN transcript_segments s ON s.id = w.segment_id
              WHERE s.video_id = ?1",
            [video_id],
            |r| r.get(0),
        )
        .unwrap_or(0)
}

/// A recognised word, on the engine's clock (seconds into the video).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Give each word to the sentence it falls in (by its middle) and store them, replacing what the
/// video had. Words outside every sentence (a repeat the transcript dropped) are left out.
/// `segments` are `(row id, start, end)`.
pub fn store_words(db: &Db, segments: &[(i64, f64, f64)], words: &[Word]) -> Result<usize, crate::Error> {
    let mut stored = 0;
    let mut insert = db.conn.prepare_cached(
        "INSERT OR REPLACE INTO transcript_words(segment_id, idx, start_s, end_s, text) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let mut next_idx: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for w in words {
        let mid = (w.start + w.end) / 2.0;
        // Segments are in order; a word's middle sits in one, or just outside an edge by rounding.
        let owner = segments.iter().find(|(_, s, e)| mid >= *s - 0.05 && mid <= *e + 0.05).map(|(id, _, _)| *id);
        let Some(seg) = owner else { continue };
        let idx = next_idx.entry(seg).or_insert(0);
        *idx += 1;
        insert.execute(params![seg, *idx, w.start, w.end, w.text.trim()])?;
        stored += 1;
    }
    Ok(stored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_naming_the_wrong_video_takes_its_sentences_video_unless_it_is_muted_b_roll() {
        let db = Db::open_in_memory().unwrap();
        for v in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60.0)",
                    rusqlite::params![v, format!("h{v}")],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES
                 (100, 1, 2.0, 6.0, 'What I love about Westlake is the community.')",
                [],
            )
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 1);
        let mut s: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{ "id": "b1", "purpose": "p", "clips": [
                { "video_id": 2, "from": "s100", "to": "s100", "audio": "source" },
                { "video_id": 2, "from": "s100", "to": "s100", "audio": "mute", "in_s": 0.0, "out_s": 6.6 }
            ] }]
        }))
        .unwrap();
        let notes = correct_sentence_videos(&db, &mut s);
        assert_eq!(notes.len(), 2, "{notes:?}");
        let (spoken, broll) = (&s.beats[0].clips[0], &s.beats[0].clips[1]);
        assert_eq!(spoken.video_id, 1, "the line is Mary Ellen's: her video plays");
        assert_eq!(
            (broll.video_id, broll.sentences.is_none(), broll.out_s),
            (2, true, 6.6),
            "the b-roll keeps its pictures"
        );
        // And it resolves now instead of refusing the whole script.
        resolve_script(&db, &mut s).unwrap();
        assert!(s.beats[0].clips[0].in_s >= 1.9 && s.beats[0].clips[0].out_s <= 6.1, "{:?}", s.beats[0].clips[0]);
    }

    /// Resolving is where a chat answer's sentences are turned into seconds, so the correction's
    /// notes have to come out of it: they used to be dropped there, and the editor never learned
    /// that its clip now plays another video.
    #[test]
    fn resolving_reports_the_video_corrections_it_made() {
        let db = Db::open_in_memory().unwrap();
        for v in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60.0)",
                    rusqlite::params![v, format!("h{v}")],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES
                 (100, 1, 2.0, 6.0, 'What I love about Westlake is the community.')",
                [],
            )
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 1);
        let mut s: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{ "id": "b1", "purpose": "p", "clips": [
                { "video_id": 2, "from": "s100", "to": "s100", "audio": "source" }
            ] }]
        }))
        .unwrap();
        let (resolved, notes) = resolve_script_noting(&db, &mut s).unwrap();
        assert_eq!(resolved, 1);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("video #1"), "{notes:?}");
        assert_eq!(s.beats[0].clips[0].video_id, 1);
        // A second pass finds nothing left to correct, and says nothing.
        assert_eq!(resolve_script_noting(&db, &mut s).unwrap(), (0, Vec::new()));
    }
    use crate::speech::SpeechRegion;

    fn db_with_two_videos() -> Db {
        let db = Db::open_in_memory().unwrap();
        for v in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 100.0)",
                    params![v, format!("h{v}")],
                )
                .unwrap();
        }
        for (id, video, a, b, text) in [
            (412, 1, 12.0, 15.5, "Esto es O Corres o Te Encaramas."),
            (413, 1, 15.6, 18.0, "Y ya."),
            (418, 1, 30.0, 34.0, "Fin del tema."),
            (500, 2, 3.0, 5.0, "Otro video."),
        ] {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, video, a, b, text],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![id, video, a, b, text],
                )
                .unwrap();
        }
        // Built by the current splitting rules, so nothing splits them again under the test.
        db.conn.execute("UPDATE videos SET sentence_split = ?1", [crate::sentence_split::SPLIT_VERSION]).unwrap();
        db
    }

    #[test]
    fn ids_parse_in_every_spelling_and_say_what_is_wrong_otherwise() {
        assert_eq!(SentenceRef::parse("s412").unwrap(), SentenceRef { sentence_id: 412, word: None });
        assert_eq!(SentenceRef::parse(" S412@3 ").unwrap(), SentenceRef { sentence_id: 412, word: Some(3) });
        for bad in ["", "s", "sx1", "s0", "s-4", "s412@0", "s412@x", "412", "412.5"] {
            assert!(matches!(SentenceRef::parse(bad), Err(SentenceError::Malformed { .. })), "{bad}");
        }
        assert_eq!(format_id(412), "s412");
        assert_eq!(SentenceRef::parse("s9@2").unwrap().to_string(), "s9@2");
        // Every message ends in what to do.
        assert!(SentenceRef::parse("x").unwrap_err().to_string().contains("get_transcript"));
    }

    #[test]
    fn a_range_resolves_from_the_first_start_to_the_last_end() {
        let db = db_with_two_videos();
        assert_eq!(resolve_range(&db, 1, "s412", "s418").unwrap(), (12.0, 34.0));
        assert_eq!(resolve_range(&db, 1, "s412", "s412").unwrap(), (12.0, 15.5));
    }

    #[test]
    fn errors_are_typed_and_carry_a_fix() {
        let db = db_with_two_videos();
        let other = resolve_range(&db, 1, "s412", "s500").unwrap_err();
        assert_eq!(other, SentenceError::OtherVideo { id: "s500".into(), found_in: 2, video_id: 1 });
        assert!(other.to_string().contains("Use video_id 2"), "{other}");
        let reversed = resolve_range(&db, 1, "s418", "s412").unwrap_err();
        assert!(matches!(reversed, SentenceError::Reversed { .. }), "{reversed:?}");
        assert!(reversed.to_string().contains("swap"));
        let missing = resolve_range(&db, 1, "s412", "s999").unwrap_err();
        assert_eq!(missing, SentenceError::NotFound { id: "s999".into() });
        let no_words = resolve_range(&db, 1, "s412@2", "s418").unwrap_err();
        assert!(matches!(no_words, SentenceError::NoWordTimings { word: 2, .. }), "{no_words:?}");
        assert!(no_words.to_string().contains("whole sentence"));
    }

    #[test]
    fn a_word_is_cut_on_the_nearest_edge_of_the_speech() {
        let db = db_with_two_videos();
        // Esto(12.00-12.30) es(12.35-12.5) O(12.62-12.8) Corres(12.9-13.4)...
        let words = [("Esto", 12.0, 12.3), ("es", 12.35, 12.5), ("O", 12.62, 12.8), ("Corres", 12.9, 13.4)]
            .map(|(t, a, b)| Word { start: a, end: b, text: t.into() });
        let stored = store_words(&db, &[(412, 12.0, 15.5), (413, 15.6, 18.0)], &words).unwrap();
        assert_eq!(stored, 4);
        assert_eq!(word_count(&db, 1), 4);
        // New words: the sentences are built again with them (as the indexer does).
        crate::sentence_split::mirror_segments(&db, 1);
        // Without measured speech the engine's times stand.
        assert_eq!(resolve_range(&db, 1, "s412@3", "s412@4").unwrap(), (12.62, 13.4));
        // With it, the edges of the pauses around the word: onset 12.66 (the engine said 12.62),
        // end 13.37 (said 13.4).
        db.set_speech_regions(
            1,
            &[
                SpeechRegion { start_s: 11.9, end_s: 12.55, quiet_after_s: Some(12.6) },
                SpeechRegion { start_s: 12.66, end_s: 13.37, quiet_after_s: None },
            ],
        )
        .unwrap();
        // New pauses rebuild the sentences; this fixture's are its segments.
        crate::sentence_split::mirror_segments(&db, 1);
        let (a, b) = resolve_range(&db, 1, "s412@3", "s412@4").unwrap();
        assert!((a - 12.66).abs() < 1e-9 && (b - 13.37).abs() < 1e-9, "{a} {b}");
        // The fifth word does not exist.
        let err = resolve_range(&db, 1, "s412@1", "s412@5").unwrap_err();
        assert_eq!(err, SentenceError::WordOutOfRange { id: "s412".into(), word: 5, words: 4 });
        // Words go with their segment.
        db.conn.execute("DELETE FROM transcript_segments WHERE id = 412", []).unwrap();
        assert_eq!(word_count(&db, 1), 0);
    }

    #[test]
    fn a_clip_written_with_sentences_parses_resolves_and_keeps_its_range() {
        let db = db_with_two_videos();
        let mut script: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{
                "id": "b1", "purpose": "p",
                "clips": [
                    {"video_id": 1, "from": "s412", "to": "s413"},
                    {"video_id": 1, "in_s": 20.0, "out_s": 22.0}
                ],
                "bed": {"video_id": 1, "from": "s418", "to": "s418"}
            }]
        }))
        .unwrap();
        assert!(script.beats[0].clips[0].is_unresolved());
        assert!(!script.beats[0].clips[1].is_unresolved());
        assert_eq!(resolve_script(&db, &mut script).unwrap(), 2);
        let c = &script.beats[0].clips[0];
        assert_eq!((c.in_s, c.out_s), (12.0, 18.0));
        assert_eq!(c.sentences, Some(SentenceRange { from: "s412".into(), to: "s413".into() }));
        assert_eq!((script.beats[0].clips[1].in_s, script.beats[0].clips[1].out_s), (20.0, 22.0));
        let bed = script.beats[0].bed.as_ref().unwrap();
        assert_eq!((bed.in_s, bed.out_s), (30.0, 34.0));
        // The stored form has both: seconds that play, sentences that explain.
        let json = serde_json::to_value(&script).unwrap();
        let clip = &json["beats"][0]["clips"][0];
        assert_eq!(
            (clip["in_s"].as_f64(), clip["from"].as_str(), clip["to"].as_str()),
            (Some(12.0), Some("s412"), Some("s413"))
        );
        assert!(json["beats"][0]["clips"][1].get("from").is_none(), "a seconds clip stays as it was written");
        let again: Script = serde_json::from_value(json).unwrap();
        assert_eq!(again, script);
        assert_eq!(resolve_script(&db, &mut script).unwrap(), 0, "resolving twice does nothing");
    }

    #[test]
    fn a_clip_with_neither_or_half_of_a_range_is_refused_at_parse() {
        for (clip, hint) in [
            (serde_json::json!({"video_id": 1}), "in_s and out_s"),
            (serde_json::json!({"video_id": 1, "from": "s1"}), "no `to`"),
            (serde_json::json!({"video_id": 1, "to": "s1"}), "no `from`"),
            (serde_json::json!({"video_id": 1, "in_s": 1.0}), "both in_s and out_s"),
        ] {
            let err = serde_json::from_value::<crate::script::ScriptClip>(clip).unwrap_err().to_string();
            assert!(err.contains(hint), "{err}");
        }
        // Seconds win when both are given: a stored clip is written back as it was.
        let both: crate::script::ScriptClip = serde_json::from_value(
            serde_json::json!({"video_id": 1, "in_s": 2.0, "out_s": 3.0, "from": "s1", "to": "s2"}),
        )
        .unwrap();
        assert!(!both.is_unresolved());
        assert_eq!((both.in_s, both.out_s), (2.0, 3.0));
    }

    #[test]
    fn resolve_script_names_the_beat_and_clip_that_failed() {
        let db = db_with_two_videos();
        let mut script: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{"id": "hook", "purpose": "p", "clips": [
                {"video_id": 1, "from": "s412", "to": "s413"},
                {"video_id": 1, "from": "s99999", "to": "s99999"}
            ]}]
        }))
        .unwrap();
        let err = resolve_script(&db, &mut script).unwrap_err();
        let text = err.to_string();
        assert!(text.starts_with("beat hook clip 2:") && text.contains("s99999"), "{text}");
    }

    /// The 9B wrote every clip's from/to as seconds copied from a manifest ("0.41", "8.16"): the
    /// whole answer came back "'0.41' is not a sentence id" and a review round was spent on it.
    /// Seconds are taken as seconds, with a note; a sentence on one edge still resolves; and a bare
    /// integer stays ambiguous.
    #[test]
    fn a_clip_written_in_seconds_resolves_to_those_seconds() {
        let db = db_with_two_videos();
        let mut script: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{"id": "b0", "purpose": "p", "clips": [
                {"video_id": 1, "from": "12.41", "to": "17.9"},
                {"video_id": 1, "from": "s412", "to": "20s"}
            ], "bed": {"video_id": 1, "from": "30.0", "to": "33.5"}}]
        }))
        .unwrap();
        let (resolved, notes) = resolve_script_noting(&db, &mut script).unwrap();
        assert_eq!(resolved, 3);
        let clips = &script.beats[0].clips;
        assert_eq!((clips[0].in_s, clips[0].out_s), (12.41, 17.9));
        assert!(clips[0].sentences.is_none(), "a seconds range is a plain range");
        assert_eq!((clips[1].in_s, clips[1].out_s), (12.0, 20.0));
        let bed = script.beats[0].bed.as_ref().unwrap();
        assert_eq!((bed.in_s, bed.out_s), (30.0, 33.5));
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert!(notes[0].contains("read as seconds"), "{notes:?}");

        let mut ambiguous: Script = serde_json::from_value(serde_json::json!({
            "title": "t", "beats": [{"id": "b0", "purpose": "p", "clips": [{"video_id": 1, "from": "412", "to": "413"}]}]
        }))
        .unwrap();
        assert!(resolve_script(&db, &mut ambiguous).is_err());
    }

    #[test]
    fn correct_sentence_videos_leaves_resolved_clips_inside_span_alone() {
        let db = Db::open_in_memory().unwrap();
        for v in [1, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60.0)",
                    params![v, format!("h{v}")],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES
                 (100, 1, 2.0, 6.0, 'Inside span.')",
                [],
            )
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 1);
        let mut s: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{ "id": "b1", "purpose": "p", "clips": [
                // Resolved and inside s100's span (2.0..6.0): must NOT be rewritten to video 1.
                { "video_id": 2, "in_s": 2.0, "out_s": 6.0, "from": "s100", "to": "s100", "audio": "source" },
                // Resolved but outside s100's span (20.0..25.0): corrected to video 1.
                { "video_id": 2, "in_s": 20.0, "out_s": 25.0, "from": "s100", "to": "s100", "audio": "source" },
                // Unresolved: corrected to video 1.
                { "video_id": 2, "from": "s100", "to": "s100", "audio": "source" }
            ] }]
        }))
        .unwrap();
        let notes = correct_sentence_videos(&db, &mut s);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert_eq!(s.beats[0].clips[0].video_id, 2);
        assert_eq!((s.beats[0].clips[0].in_s, s.beats[0].clips[0].out_s), (2.0, 6.0));
        assert_eq!(s.beats[0].clips[1].video_id, 1);
        assert_eq!(s.beats[0].clips[2].video_id, 1);
    }

    #[test]
    fn sentence_ref_rejects_bare_integers() {
        assert!(matches!(SentenceRef::parse("412"), Err(SentenceError::Malformed { .. })));
    }

    #[test]
    fn edge_ref_parses_sentences_and_seconds_and_rejects_ambiguous_or_non_finite() {
        assert_eq!(EdgeRef::parse("s412").unwrap(), EdgeRef::Sentence(SentenceRef { sentence_id: 412, word: None }));
        assert_eq!(
            EdgeRef::parse("S412@3").unwrap(),
            EdgeRef::Sentence(SentenceRef { sentence_id: 412, word: Some(3) })
        );
        assert_eq!(EdgeRef::parse("412s").unwrap(), EdgeRef::Seconds(412.0));
        assert_eq!(EdgeRef::parse("15.75").unwrap(), EdgeRef::Seconds(15.75));
        assert_eq!(EdgeRef::parse_value(&serde_json::json!(412)).unwrap(), EdgeRef::Seconds(412.0));
        let ambig = EdgeRef::parse("412").unwrap_err();
        assert!(ambig.contains("s412") && ambig.contains("412s") && ambig.contains("412.0"), "{ambig}");
        for bad in ["", "-5", "-5.0", "-5s", "NaN", "inf", "-inf", "NaNs", "infs", "1e309.0"] {
            assert!(EdgeRef::parse(bad).is_err(), "{bad}");
        }
    }
}
