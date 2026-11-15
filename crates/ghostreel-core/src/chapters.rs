//! Chapters: each speech video split into topical stretches a model can read about instead of
//! reading the whole transcript (`get_outline`).
//!
//! Boundaries are found in code. The embed stage stores an embedding for every ~30 s window of
//! transcript (`chunks`, kind `transcript`, overlapping by 5 s); the similarity of the windows
//! before a point to the windows after it dips where the subject changes. A boundary is placed on
//! each deep dip, no closer than [`Params::min_s`] to another, and a stretch longer than
//! [`Params::max_s`] is split at its lowest remaining dip. Every boundary sits on the start of a
//! sentence. A video with no stored embeddings is split by length, at the longest pauses.
//!
//! Titles and summaries come from the configured chat model, through the same backend machinery as
//! the other model stages ([`crate::chat::lease_backend`], [`crate::chat::workers::ask_json`]).
//! With no model available the chapters are stored with their first sentence as the title and
//! `titled = false`, so a later run can refresh them (`ghostreel chapters --refresh-titles`).

use rusqlite::params;
use serde::Serialize;
use serde_json::{Value, json};

use crate::Error;
use crate::db::Db;
use crate::index::TranscriptSegment;
use crate::sentences::format_id;

/// How boundaries are chosen. [`Default`] is what the stage uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// No chapter is shorter than this (s).
    pub min_s: f64,
    /// No chapter is longer than this (s).
    pub max_s: f64,
    /// Windows each side of a point that are averaged for the similarity.
    pub window: usize,
    /// A dip must be this many standard deviations over the average dip to count.
    pub z: f64,
    /// The least a dip may be, whatever the rest looks like (cosine similarity).
    pub min_depth: f64,
    /// The length chapters are cut to when there are no embeddings (s).
    pub fallback_s: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self { min_s: 90.0, max_s: 600.0, window: 4, z: 0.5, min_depth: 0.04, fallback_s: 240.0 }
    }
}

/// One stored window of transcript and its embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub start_s: f64,
    pub end_s: f64,
    pub embedding: Vec<f32>,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn mean(points: &[Point]) -> Vec<f32> {
    let dim = points.first().map_or(0, |p| p.embedding.len());
    let mut sum = vec![0.0f32; dim];
    for p in points {
        for (s, x) in sum.iter_mut().zip(&p.embedding) {
            *s += *x;
        }
    }
    let n = points.len().max(1) as f32;
    sum.iter_mut().for_each(|s| *s /= n);
    sum
}

/// The index of the sentence that starts nearest to `t`.
fn nearest_sentence(sentences: &[(f64, f64)], t: f64) -> usize {
    (0..sentences.len()).min_by(|&a, &b| (sentences[a].0 - t).abs().total_cmp(&(sentences[b].0 - t).abs())).unwrap_or(0)
}

/// Where each chapter after the first begins, as the index of its first sentence (ascending,
/// never 0). `sentences` are `(start, end)` in order; `points` the stored windows of the video.
pub fn find_boundaries(sentences: &[(f64, f64)], points: &[Point], p: &Params) -> Vec<usize> {
    let (Some(first), Some(last)) = (sentences.first(), sentences.last()) else { return Vec::new() };
    let (video_start, video_end) = (first.0, last.1);
    if video_end - video_start < 2.0 * p.min_s {
        return Vec::new();
    }
    let mut chosen: Vec<usize> = Vec::new();
    // Each candidate: (sentence index, dip depth, similarity there).
    let mut dips: Vec<(usize, f64, f64)> = Vec::new();

    let w = p.window.max(1);
    if points.len() >= 2 * w {
        // Similarity at every gap k between window k-1 and window k.
        let sims: Vec<Option<f64>> = (0..points.len())
            .map(|k| {
                if k < w || k + w > points.len() {
                    return None;
                }
                Some(cosine(&mean(&points[k - w..k]), &mean(&points[k..k + w])))
            })
            .collect();
        let at = |k: usize| sims.get(k).copied().flatten();
        for k in w..=points.len() - w {
            let Some(s) = at(k) else { continue };
            let left = k.checked_sub(1).and_then(at);
            let right = at(k + 1);
            if left.is_some_and(|l| l < s) || right.is_some_and(|r| r < s) {
                continue;
            }
            // How far the similarity climbs on each side before it stops climbing.
            let climb = |step: isize| {
                let mut peak = s;
                let mut j = k as isize + step;
                while let Some(v) = (j >= 0).then(|| at(j as usize)).flatten() {
                    if v < peak {
                        break;
                    }
                    peak = v;
                    j += step;
                }
                peak
            };
            let depth = (climb(-1) - s) + (climb(1) - s);
            // The point between the two windows is their overlap: the middle of it.
            let t = (points[k].start_s + points[k - 1].end_s) / 2.0;
            let idx = nearest_sentence(sentences, t);
            if idx > 0 {
                dips.push((idx, depth, s));
            }
        }
        if !dips.is_empty() {
            let n = dips.len() as f64;
            let avg = dips.iter().map(|d| d.1).sum::<f64>() / n;
            let sd = (dips.iter().map(|d| (d.1 - avg).powi(2)).sum::<f64>() / n).sqrt();
            let cutoff = (avg + p.z * sd).max(p.min_depth);
            let mut deep: Vec<&(usize, f64, f64)> = dips.iter().filter(|d| d.1 >= cutoff).collect();
            deep.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &&(idx, _, _) in &deep {
                if far_enough(sentences, &chosen, idx, video_start, video_end, p.min_s) {
                    chosen.push(idx);
                    chosen.sort_unstable();
                }
            }
        }
        // Spans that are still too long: the lowest remaining dip, then the middle.
        loop {
            let mut edges = vec![0usize];
            edges.extend(&chosen);
            edges.push(sentences.len());
            let Some(span) =
                edges.windows(2).find(|e| sentences[e[1] - 1].1 - sentences[e[0]].0 > p.max_s).map(|e| (e[0], e[1]))
            else {
                break;
            };
            let mut remaining: Vec<&(usize, f64, f64)> = dips.iter().filter(|d| d.0 > span.0 && d.0 < span.1).collect();
            remaining.sort_by(|a, b| a.2.total_cmp(&b.2));
            let pick = remaining
                .iter()
                .map(|d| d.0)
                .find(|&idx| far_enough(sentences, &chosen, idx, video_start, video_end, p.min_s))
                .or_else(|| {
                    let mid = (sentences[span.0].0 + sentences[span.1 - 1].1) / 2.0;
                    let idx = nearest_sentence(sentences, mid);
                    (idx > span.0 && idx < span.1).then_some(idx)
                });
            match pick {
                Some(idx) if !chosen.contains(&idx) => {
                    chosen.push(idx);
                    chosen.sort_unstable();
                }
                _ => break,
            }
        }
    } else {
        chosen = by_length(sentences, p);
    }
    chosen
}

/// Whether a boundary at sentence `idx` keeps every chapter at least `min_s` long.
fn far_enough(sentences: &[(f64, f64)], chosen: &[usize], idx: usize, start: f64, end: f64, min_s: f64) -> bool {
    let t = sentences[idx].0;
    t - start >= min_s && end - t >= min_s && chosen.iter().all(|&c| (sentences[c].0 - t).abs() >= min_s)
}

/// Boundaries every `fallback_s` or so, each at the longest pause within reach of its target.
fn by_length(sentences: &[(f64, f64)], p: &Params) -> Vec<usize> {
    let end = sentences.last().map_or(0.0, |s| s.1);
    let mut out = Vec::new();
    let mut from = sentences.first().map_or(0.0, |s| s.0);
    loop {
        let target = from + p.fallback_s;
        if end - target < p.min_s {
            break;
        }
        let reach = p.fallback_s * 0.25;
        let best = (1..sentences.len())
            .filter(|&i| (sentences[i].0 - target).abs() <= reach && sentences[i].0 - from >= p.min_s)
            .max_by(|&a, &b| {
                let (ga, gb) = (sentences[a].0 - sentences[a - 1].1, sentences[b].0 - sentences[b - 1].1);
                ga.total_cmp(&gb)
            });
        let Some(i) = best.or_else(|| {
            let i = nearest_sentence(sentences, target);
            (i > 0).then_some(i)
        }) else {
            break;
        };
        if out.last().is_some_and(|&l| l >= i) {
            break;
        }
        out.push(i);
        from = sentences[i].0;
    }
    out
}

/// A chapter, as stored.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Chapter {
    pub video_id: i64,
    pub idx: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub first_sentence_id: i64,
    pub last_sentence_id: i64,
    pub title: String,
    pub summary: String,
    /// 1-3 sentence ids (sentence row ids) worth quoting.
    pub standout: Vec<i64>,
    /// The title and summary were written by a model (false: the first sentence stands in).
    pub titled: bool,
}

impl Chapter {
    pub fn first_sentence(&self) -> String {
        format_id(self.first_sentence_id)
    }

    pub fn last_sentence(&self) -> String {
        format_id(self.last_sentence_id)
    }
}

/// A short title from a sentence: its first words, up to `max` characters.
pub fn title_from(text: &str, max: usize) -> String {
    let text = text.trim().trim_end_matches(['.', '!', '?', ',', ';', ':']);
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{}…", cut.trim_end_matches([',', ';', ':']))
}

/// Chapters on the given boundaries, with the first sentence as the title.
pub fn from_boundaries(video_id: i64, sentences: &[TranscriptSegment], boundaries: &[usize]) -> Vec<Chapter> {
    let mut edges = vec![0usize];
    edges.extend(boundaries.iter().copied().filter(|&b| b > 0 && b < sentences.len()));
    edges.push(sentences.len());
    edges.dedup();
    edges
        .windows(2)
        .enumerate()
        .filter(|(_, e)| e[0] < e[1])
        .map(|(idx, e)| {
            let (first, last) = (&sentences[e[0]], &sentences[e[1] - 1]);
            Chapter {
                video_id,
                idx,
                start_s: first.start,
                end_s: last.end,
                first_sentence_id: first.id,
                last_sentence_id: last.id,
                title: title_from(&first.text, 60),
                summary: String::new(),
                standout: Vec::new(),
                titled: false,
            }
        })
        .collect()
}

/// The prompt asking for titles, summaries and standout lines for `chapters` of one video.
/// `sentences` is the whole transcript; each chapter's own are taken from it.
pub fn titling_prompt(chapters: &[Chapter], sentences: &[TranscriptSegment]) -> String {
    const PER_CHAPTER_CHARS: usize = 2400;
    let mut out = String::from(
        "You label the chapters of a video transcript for a video editor. For every numbered chapter give a title \
         of at most 6 words, a summary of one sentence, and the ids of 1 to 3 standout lines: the most quotable, \
         self-contained sentences, taken only from that chapter's own ids. Write in the language of the transcript. \
         Name what is discussed, not that something is discussed.\n",
    );
    for (n, c) in chapters.iter().enumerate() {
        out.push_str(&format!("\nChapter {}\n", n + 1));
        let own: Vec<&TranscriptSegment> = sentences
            .iter()
            .filter(|s| s.id >= c.first_sentence_id && s.id <= c.last_sentence_id && !s.off_mic)
            .collect();
        let total: usize = own.iter().map(|s| s.text.len() + 8).sum();
        // A long chapter is sampled evenly rather than cut off: the end says as much as the start.
        let step = total / PER_CHAPTER_CHARS + 1;
        for s in own.iter().step_by(step) {
            out.push_str(&format!("{} {}\n", format_id(s.id), s.text.trim()));
        }
    }
    out
}

/// The JSON schema of the titling answer.
pub fn titling_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": { "type": "integer" },
                        "title": { "type": "string" },
                        "summary": { "type": "string" },
                        "standout": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["n", "title", "summary", "standout"]
                }
            }
        },
        "required": ["chapters"]
    })
}

/// Apply a model's answer to `chapters` (in the order the prompt listed them). Only what is usable
/// is taken: an empty title keeps the first-sentence one, a standout id must belong to the
/// chapter. Returns how many chapters were titled.
pub fn apply_titles(chapters: &mut [Chapter], answer: &Value) -> usize {
    let mut titled = 0;
    let Some(items) = answer["chapters"].as_array() else { return 0 };
    for item in items {
        let Some(c) = item["n"].as_u64().and_then(|n| n.checked_sub(1)).and_then(|n| chapters.get_mut(n as usize))
        else {
            continue;
        };
        let title = item["title"].as_str().map(|t| title_from(t, 80)).filter(|t| !t.is_empty());
        let Some(title) = title else { continue };
        c.title = title;
        c.summary = item["summary"].as_str().map(|s| s.trim().chars().take(240).collect()).unwrap_or_default();
        c.standout = item["standout"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(|s| crate::sentences::SentenceRef::parse(s).ok())
                    .map(|r| r.sentence_id)
                    .filter(|id| *id >= c.first_sentence_id && *id <= c.last_sentence_id)
                    .take(3)
                    .collect()
            })
            .unwrap_or_default();
        c.titled = true;
        titled += 1;
    }
    titled
}

/// The stored windows of a video's transcript, with their embeddings, in order.
pub fn points(db: &Db, video_id: i64) -> Result<Vec<Point>, Error> {
    let mut st = db.conn.prepare(
        "SELECT c.start_s, c.end_s, v.embedding FROM chunks c JOIN chunks_vec v ON v.rowid = c.id
          WHERE c.video_id = ?1 AND c.kind = 'transcript' ORDER BY c.start_s",
    )?;
    let rows = st.query_map([video_id], |r| {
        let blob: Vec<u8> = r.get(2)?;
        let embedding = blob.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok(Point { start_s: r.get(0)?, end_s: r.get(1)?, embedding })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Replace a video's chapters.
pub fn store(db: &Db, video_id: i64, chapters: &[Chapter]) -> Result<(), Error> {
    let tx = db.conn.unchecked_transaction()?;
    tx.execute("DELETE FROM chapters WHERE video_id = ?1", [video_id])?;
    insert_all(&tx, video_id, chapters)?;
    tx.commit()?;
    Ok(())
}

/// Insert `chapters` for a video on `conn` (a transaction the caller owns and commits), without
/// deleting what is there: the caller has cleared the video's chapters first.
pub(crate) fn insert_all(conn: &rusqlite::Connection, video_id: i64, chapters: &[Chapter]) -> Result<(), Error> {
    let mut insert = conn.prepare(
        "INSERT INTO chapters(video_id, idx, start_s, end_s, first_sentence_id, last_sentence_id, title, summary,
                              standout_json, titled)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;
    for c in chapters {
        insert.execute(params![
            video_id,
            c.idx as i64,
            c.start_s,
            c.end_s,
            c.first_sentence_id,
            c.last_sentence_id,
            c.title,
            c.summary,
            serde_json::to_string(&c.standout).unwrap_or_else(|_| "[]".into()),
            c.titled,
        ])?;
    }
    Ok(())
}

/// A video's chapters, in order.
pub fn load(db: &Db, video_id: i64) -> Result<Vec<Chapter>, Error> {
    let mut st = db.conn.prepare(
        "SELECT idx, start_s, end_s, first_sentence_id, last_sentence_id, title, summary, standout_json, titled
           FROM chapters WHERE video_id = ?1 ORDER BY idx",
    )?;
    let rows = st.query_map([video_id], |r| {
        Ok(Chapter {
            video_id,
            idx: r.get::<_, i64>(0)? as usize,
            start_s: r.get(1)?,
            end_s: r.get(2)?,
            first_sentence_id: r.get(3)?,
            last_sentence_id: r.get(4)?,
            title: r.get(5)?,
            summary: r.get(6)?,
            standout: serde_json::from_str(&r.get::<_, String>(7)?).unwrap_or_default(),
            titled: r.get(8)?,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Whether any of a video's chapters is waiting for a model's title.
pub fn has_untitled(db: &Db, video_id: i64) -> bool {
    db.conn
        .query_row("SELECT EXISTS (SELECT 1 FROM chapters WHERE video_id = ?1 AND titled = 0)", [video_id], |r| {
            r.get(0)
        })
        .unwrap_or(false)
}

/// Split one video into chapters, in code: boundaries from the stored embeddings (or the length
/// fallback), first-sentence titles. `None` when the video has no speech.
pub fn split(db: &Db, video_id: i64, p: &Params) -> Result<Option<Vec<Chapter>>, Error> {
    let sentences: Vec<TranscriptSegment> =
        crate::index::transcript(db, video_id)?.into_iter().filter(|s| !s.text.trim().is_empty()).collect();
    if sentences.is_empty() {
        return Ok(None);
    }
    let times: Vec<(f64, f64)> = sentences.iter().map(|s| (s.start, s.end)).collect();
    let boundaries = find_boundaries(&times, &points(db, video_id)?, p);
    Ok(Some(from_boundaries(video_id, &sentences, &boundaries)))
}

/// How many chapters per batch are sent to the model at once.
pub const TITLES_PER_CALL: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector in dimension `topic` of 8, with a little deterministic noise.
    fn topic_vec(topic: usize, noise_seed: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 8];
        v[topic % 8] = 1.0;
        v[(topic + 1 + noise_seed % 3) % 8] += 0.05 * ((noise_seed % 5) as f32 / 5.0);
        v
    }

    /// Windows of 30 s every 25 s over `minutes` minutes; the topic changes at each of `changes`
    /// (seconds).
    fn timeline(minutes: f64, changes: &[f64]) -> (Vec<(f64, f64)>, Vec<Point>) {
        let total = minutes * 60.0;
        // A sentence every 5 s.
        let sentences: Vec<(f64, f64)> =
            (0..(total / 5.0) as usize).map(|i| (i as f64 * 5.0, i as f64 * 5.0 + 4.5)).collect();
        let topic_at = |t: f64| changes.iter().filter(|&&c| t >= c).count();
        let mut points = Vec::new();
        let mut start = 0.0;
        let mut i = 0;
        while start + 30.0 <= total {
            points.push(Point { start_s: start, end_s: start + 30.0, embedding: topic_vec(topic_at(start + 15.0), i) });
            start += 25.0;
            i += 1;
        }
        (sentences, points)
    }

    #[test]
    fn boundaries_fall_where_the_subject_changes() {
        let (sentences, points) = timeline(20.0, &[400.0, 800.0]);
        let found = find_boundaries(&sentences, &points, &Params::default());
        let times: Vec<f64> = found.iter().map(|&i| sentences[i].0).collect();
        assert_eq!(times.len(), 2, "{times:?}");
        // Windows are 25 s apart, so a boundary is placed within one window of the change.
        assert!((times[0] - 400.0).abs() <= 25.0, "{times:?}");
        assert!((times[1] - 800.0).abs() <= 25.0, "{times:?}");
        // A sentence start, never 0.
        assert!(found.iter().all(|&i| i > 0));
    }

    #[test]
    fn one_subject_all_the_way_is_one_chapter_until_it_is_too_long() {
        // Ten minutes of one subject: under the maximum, so nothing is split.
        let (sentences, points) = timeline(10.0, &[]);
        assert!(find_boundaries(&sentences, &points, &Params::default()).is_empty());
        // Thirty: the maximum splits it into chapters no longer than that.
        let (sentences, points) = timeline(30.0, &[]);
        let p = Params::default();
        let found = find_boundaries(&sentences, &points, &p);
        assert!(!found.is_empty());
        let mut edges = vec![0usize];
        edges.extend(&found);
        edges.push(sentences.len());
        for e in edges.windows(2) {
            let len = sentences[e[1] - 1].1 - sentences[e[0]].0;
            assert!(len <= p.max_s + 1e-9, "a chapter of {len} s");
            assert!(len >= p.min_s - 1e-9, "a chapter of {len} s");
        }
    }

    #[test]
    fn two_changes_closer_than_the_minimum_make_one_boundary() {
        let (sentences, points) = timeline(15.0, &[400.0, 440.0]);
        let found = find_boundaries(&sentences, &points, &Params::default());
        assert!(found.len() <= 1, "{found:?}");
        let times: Vec<f64> = found.iter().map(|&i| sentences[i].0).collect();
        assert!(times.iter().all(|t| (350.0..=480.0).contains(t)), "{times:?}");
    }

    #[test]
    fn no_embeddings_means_length_chapters_at_the_pauses() {
        let (mut sentences, _) = timeline(20.0, &[]);
        // A long pause near 240 s and near 480 s.
        for s in sentences.iter_mut().filter(|s| s.0 >= 245.0) {
            s.0 += 3.0;
            s.1 += 3.0;
        }
        for s in sentences.iter_mut().filter(|s| s.0 >= 490.0) {
            s.0 += 3.0;
            s.1 += 3.0;
        }
        let found = find_boundaries(&sentences, &[], &Params::default());
        assert!(found.len() >= 3, "{found:?}");
        let times: Vec<f64> = found.iter().map(|&i| sentences[i].0).collect();
        assert!((times[0] - 248.0).abs() < 1.0, "the first boundary is after the pause: {times:?}");
        assert!(found.windows(2).all(|w| w[0] < w[1]));
        // Short videos are one chapter.
        let (short, _) = timeline(2.0, &[]);
        assert!(find_boundaries(&short, &[], &Params::default()).is_empty());
        assert!(find_boundaries(&[], &[], &Params::default()).is_empty());
    }

    #[test]
    fn titles_apply_only_what_is_usable() {
        let seg = |id, start, text: &str| TranscriptSegment {
            id,
            start,
            end: start + 4.0,
            text: text.into(),
            off_mic: false,
            speaker: None,
            picture: None,
        };
        let sentences = vec![
            seg(10, 0.0, "Welcome to the show, today the engine."),
            seg(11, 5.0, "It runs hot."),
            seg(12, 10.0, "Now the chassis."),
        ];
        let mut chapters = from_boundaries(1, &sentences, &[2]);
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "Welcome to the show, today the engine");
        assert_eq!((chapters[0].first_sentence_id, chapters[0].last_sentence_id), (10, 11));
        assert!(!chapters[0].titled);
        let answer = json!({ "chapters": [
            { "n": 1, "title": "The engine", "summary": "Why it runs hot.", "standout": ["s11", "s12", "s999", "nonsense"] },
            { "n": 2, "title": "", "summary": "ignored", "standout": [] },
            { "n": 7, "title": "no such chapter", "summary": "", "standout": [] }
        ]});
        assert_eq!(apply_titles(&mut chapters, &answer), 1);
        assert_eq!(chapters[0].title, "The engine");
        assert_eq!(chapters[0].standout, vec![11], "only ids of the chapter's own sentences");
        assert!(chapters[0].titled);
        assert!(
            !chapters[1].titled && chapters[1].title == "Now the chassis",
            "an empty title keeps the first sentence"
        );
        let prompt = titling_prompt(&chapters, &sentences);
        assert!(
            prompt.contains("Chapter 1\ns10 Welcome") && prompt.contains("Chapter 2\ns12 Now the chassis."),
            "{prompt}"
        );
        assert!(title_from("a very long opening sentence that keeps going past the limit", 20).ends_with('…'));
    }

    #[test]
    fn chapters_round_trip_and_go_with_their_sentences() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 100.0)", [])
            .unwrap();
        for id in [5, 6] {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES (?1, 1, 0, 1, 'x')",
                    [id],
                )
                .unwrap();
        }
        crate::sentence_split::mirror_segments(&db, 1);
        let c = Chapter {
            video_id: 1,
            idx: 0,
            start_s: 0.0,
            end_s: 1.0,
            first_sentence_id: 5,
            last_sentence_id: 6,
            title: "T".into(),
            summary: "S".into(),
            standout: vec![5],
            titled: true,
        };
        store(&db, 1, std::slice::from_ref(&c)).unwrap();
        assert_eq!(load(&db, 1).unwrap(), vec![c.clone()]);
        assert!(!has_untitled(&db, 1));
        // Stored again: replaced, not added to.
        store(&db, 1, &[Chapter { titled: false, ..c }]).unwrap();
        assert_eq!(load(&db, 1).unwrap().len(), 1);
        assert!(has_untitled(&db, 1));
        // Transcribed again: the old sentences go, and the chapters built on them with them.
        crate::sentence_split::invalidate(&db.conn, 1).unwrap();
        assert!(load(&db, 1).unwrap().is_empty());
    }
}
