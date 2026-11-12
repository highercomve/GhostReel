use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::Error;
use crate::db::Db;
use crate::otio::ResolvedMedia;
use crate::preview::{self, format_srt_time};
use crate::script::{Fps, Script};

/// A speech caption cue positioned on the cut's timeline.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Cue {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
    /// The footage the words were heard in: video and the generated cue's source range. A
    /// `CueEdit` names a cue by these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_start_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src_end_s: Option<f64>,
    /// Cut time minus source time for the clip the cue plays in: source = cut time - offset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_s: Option<f64>,
    /// How far the cue may be moved on the cut's clock: the clip it plays in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_s: Option<f64>,
    /// A hand edit changed its text or timing.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub edited: bool,
    /// A hand edit hides it (kept in the list so it can be shown again).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hidden: bool,
}

/// How long caption lines may be and how many a cue may hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CueShape {
    pub max_chars: usize,
    pub max_lines: usize,
}

impl Default for CueShape {
    fn default() -> Self {
        Self { max_chars: 42, max_lines: 2 }
    }
}

impl CueShape {
    pub fn for_script(script: &Script) -> Self {
        let d = Self::default();
        let c = script.captions.as_ref();
        Self {
            max_chars: c.and_then(|c| c.max_chars).unwrap_or(d.max_chars).clamp(8, 120),
            max_lines: c.and_then(|c| c.max_lines).unwrap_or(d.max_lines).clamp(1, 6),
        }
    }
}

/// Helper to resolve media from the db for any video referenced by the script.
pub(crate) fn resolve_media_for_script(db: &Db, script: &Script) -> Result<HashMap<i64, ResolvedMedia>, Error> {
    let mut resolved_map = HashMap::new();
    let video_ids = script
        .beats
        .iter()
        .flat_map(|b| b.clips.iter().map(|c| c.video_id).chain(b.bed.iter().map(|bed| bed.video_id)));

    for video_id in video_ids {
        if resolved_map.contains_key(&video_id) {
            continue;
        }

        let mut st = db.conn.prepare(
            "SELECT v.duration_s, v.has_audio, v.fps, v.content_hash, vf.path
             FROM videos v
             LEFT JOIN video_files vf ON vf.video_id = v.id
             WHERE v.id = ?1",
        )?;

        let rows = st.query_map([video_id], |r| {
            Ok((
                r.get::<_, Option<f64>>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, Option<f64>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
            ))
        })?;

        let mut candidates = Vec::new();
        for r in rows {
            candidates.push(r?);
        }

        if candidates.is_empty() {
            return Err(Error::NotFound(format!("video #{video_id}")));
        }

        let chosen = candidates
            .iter()
            .find(|(_, _, _, _, p)| p.as_deref().map(Path::new).map(|p| p.exists()).unwrap_or(false))
            .unwrap_or(&candidates[0]);

        let duration_s = chosen.0.unwrap_or(0.0);
        let has_audio = chosen.1.unwrap_or(1) != 0;
        let fps = match chosen.2 {
            Some(f) if f > 0.0 => Fps::from_f64(f),
            _ => script.fps.unwrap_or(Fps::new(25, 1)),
        };
        let content_hash = chosen.3.clone().unwrap_or_default();
        let path =
            chosen.4.as_deref().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("video_{video_id}.mp4")));

        resolved_map.insert(
            video_id,
            ResolvedMedia {
                video_id,
                path,
                duration_s,
                has_audio,
                fps,
                content_hash,
                audio_track: db.audio_track(video_id).unwrap_or(0),
            },
        );
    }

    Ok(resolved_map)
}

/// Format a list of words into at most `shape.max_lines` lines of `shape.max_chars` characters.
fn fit_into_lines(words: &[String], shape: CueShape) -> Option<Vec<String>> {
    let (max, max_lines) = (shape.max_chars, shape.max_lines);
    if words.is_empty() {
        return Some(Vec::new());
    }
    let mut lines = Vec::with_capacity(max_lines);
    let mut cur = String::new();

    for w in words {
        let w_len = w.chars().count();
        if w_len > max {
            return None;
        }
        if cur.is_empty() {
            cur.push_str(w);
        } else if cur.chars().count() + 1 + w_len <= max {
            cur.push(' ');
            cur.push_str(w);
        } else {
            lines.push(cur);
            if lines.len() >= max_lines {
                return None;
            }
            cur = w.clone();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.len() <= max_lines { Some(lines) } else { None }
}

/// Split a piece of speech into cues of at most `max_lines` x `max_chars` and at most ~3.5 s.
fn split_piece_into_cues(text: &str, piece_start_s: f64, piece_end_s: f64, shape: CueShape) -> Vec<Cue> {
    let line_max = shape.max_chars;
    let cue_max = shape.max_chars * shape.max_lines;
    let trimmed = text.trim();
    let piece_dur = (piece_end_s - piece_start_s).max(0.0);
    if trimmed.is_empty() || piece_dur <= 0.0 {
        return Vec::new();
    }

    // Split words, breaking any word longer than a line so each chunk fits on one.
    let mut words: Vec<String> = Vec::new();
    for w in trimmed.split_whitespace() {
        let char_count = w.chars().count();
        if char_count > line_max {
            let chars: Vec<char> = w.chars().collect();
            for chunk in chars.chunks(line_max) {
                words.push(chunk.iter().collect());
            }
        } else {
            words.push(w.to_string());
        }
    }

    if words.is_empty() {
        return Vec::new();
    }

    let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();

    // Target duration is ~3.5s per cue.
    let min_cues_by_time = (piece_dur / 3.5).round() as usize;
    let min_cues_by_chars = (total_chars as f64 / cue_max as f64).ceil() as usize;
    let num_cues = min_cues_by_time.max(min_cues_by_chars).max(1);

    let target_chars_per_cue =
        if num_cues > 1 { ((total_chars as f64 / num_cues as f64).ceil() as usize).max(1) } else { cue_max };
    let max_cue_chars = cue_max.min(target_chars_per_cue);

    let mut cues_words: Vec<Vec<String>> = Vec::new();
    let mut current_words: Vec<String> = Vec::new();
    let mut current_chars: usize = 0;

    for word in words {
        let word_len = word.chars().count();
        let word_space_len = if current_words.is_empty() { word_len } else { word_len + 1 };

        let mut test_words = current_words.clone();
        test_words.push(word.clone());
        let can_fit = fit_into_lines(&test_words, shape).is_some();

        if !current_words.is_empty() && (!can_fit || current_chars + word_space_len > max_cue_chars) {
            cues_words.push(current_words);
            current_words = vec![word];
            current_chars = word_len;
        } else if can_fit {
            current_words.push(word);
            current_chars += word_space_len;
        } else {
            if !current_words.is_empty() {
                cues_words.push(current_words);
            }
            current_words = vec![word];
            current_chars = word_len;
        }
    }
    if !current_words.is_empty() {
        cues_words.push(current_words);
    }

    let mut cue_texts: Vec<String> = cues_words
        .into_iter()
        .map(|cw| fit_into_lines(&cw, shape).map(|lines| lines.join("\n")).unwrap_or_else(|| cw.join(" ")))
        .collect();

    // Capitalise the first letter of a cue that starts a transcript segment.
    if let Some(first_text) = cue_texts.first_mut() {
        let mut chars = first_text.chars();
        if let Some(c) = chars.next() {
            *first_text = c.to_uppercase().collect::<String>() + chars.as_str();
        }
    }

    // Distribute the piece's time across its cues proportionally to character count.
    let total_cue_chars: usize = cue_texts.iter().map(|t| t.chars().count()).sum();
    let count = cue_texts.len();
    let mut cur_start = piece_start_s;
    let mut cues = Vec::with_capacity(count);

    for (i, text) in cue_texts.into_iter().enumerate() {
        let dur = if i == count - 1 {
            (piece_start_s + piece_dur - cur_start).max(0.0)
        } else if total_cue_chars > 0 {
            piece_dur * (text.chars().count() as f64 / total_cue_chars as f64)
        } else {
            piece_dur / count as f64
        };
        let end_s = cur_start + dur;
        cues.push(Cue { start_s: cur_start, end_s, text, ..Cue::default() });
        cur_start = end_s;
    }

    cues
}

/// Retrieve and format speech caption cues for a script cut.
/// The words of a transcript segment spoken between `from` and `to`, when only part of the
/// segment is inside a clip. Segments carry one time range for all their text (some run 20–50 s),
/// so the share is estimated by character position, cut at word boundaries. Keeping the whole
/// text instead crammed 40 s of words into the 5 s a clip actually contained.
fn words_in_range(text: &str, seg_start: f64, seg_end: f64, from: f64, to: f64) -> String {
    let len = seg_end - seg_start;
    if len <= 0.0 || (from <= seg_start + 0.05 && to >= seg_end - 0.05) {
        return text.to_string();
    }
    let total = text.chars().count().max(1) as f64;
    let (a, b) = (((from - seg_start) / len).clamp(0.0, 1.0), ((to - seg_start) / len).clamp(0.0, 1.0));
    let mut out: Vec<&str> = Vec::new();
    let mut pos = 0usize;
    for word in text.split_whitespace() {
        // A word belongs to the range when its middle falls inside it.
        let start = text[pos..].find(word).map(|i| pos + i).unwrap_or(pos);
        let mid_chars = text[..start].chars().count() as f64 + word.chars().count() as f64 / 2.0;
        let mid = mid_chars / total;
        if mid >= a && mid < b {
            out.push(word);
        }
        pos = start + word.len();
    }
    out.join(" ")
}

/// Speech caption cues for a cut, with its hand edits applied; hidden cues left out.
pub fn cues_for_script(db: &Db, script: &Script) -> Result<Vec<Cue>, Error> {
    let mut cues = all_cues_for_script(db, script)?;
    cues.retain(|c| !c.hidden);
    Ok(cues)
}

/// Every cue of a cut, hidden ones included (flagged), for the editor's Captions lane.
pub fn all_cues_for_script(db: &Db, script: &Script) -> Result<Vec<Cue>, Error> {
    let media = resolve_media_for_script(db, script)?;
    let segments = preview::plan_segments(script, &media)?;
    let bed_spans = preview::plan_bed_spans(script, &segments, &media);
    let shape = CueShape::for_script(script);

    // Where speech is heard: sounding picture clips and beds, each a source range on the clock.
    let mut sources: Vec<(i64, f64, f64, f64)> = segments
        .iter()
        .filter(|s| !s.mute && s.has_audio)
        .map(|s| (s.video_id, s.in_s, s.out_s, s.timeline_start_s))
        .collect();
    sources.extend(bed_spans.iter().map(|b| (b.video_id, b.in_s, b.out_s, b.timeline_start_s)));

    let mut all_cues = Vec::new();
    let mut st = db.conn.prepare(
        "SELECT start_s, end_s, text FROM transcript_segments
         WHERE video_id = ?1 AND start_s < ?2 AND end_s > ?3
         ORDER BY start_s ASC",
    )?;
    for (video_id, in_s, out_s, timeline_start_s) in sources {
        let offset = timeline_start_s - in_s;
        let (min_s, max_s) = (timeline_start_s, timeline_start_s + (out_s - in_s).max(0.0));
        let rows = st.query_map(rusqlite::params![video_id, out_s, in_s], |r| {
            Ok((r.get::<_, f64>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?))
        })?;
        for r in rows {
            let (t_start, t_end, raw_text) = r?;
            let clipped_start = t_start.max(in_s);
            let clipped_end = t_end.min(out_s);
            if clipped_end <= clipped_start {
                continue;
            }
            let spoken = words_in_range(&raw_text, t_start, t_end, clipped_start, clipped_end);
            for mut cue in split_piece_into_cues(&spoken, clipped_start + offset, clipped_end + offset, shape) {
                cue.video_id = Some(video_id);
                cue.src_start_s = Some(cue.start_s - offset);
                cue.src_end_s = Some(cue.end_s - offset);
                cue.offset_s = Some(offset);
                cue.min_s = Some(min_s);
                cue.max_s = Some(max_s);
                all_cues.push(cue);
            }
        }
    }

    // Drop empty text
    all_cues.retain(|c| !c.text.trim().is_empty());

    // Sort by start
    all_cues.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(std::cmp::Ordering::Equal));

    // Ensure generated cues don't overlap (trim the earlier one)
    for i in 1..all_cues.len() {
        if all_cues[i - 1].end_s > all_cues[i].start_s {
            all_cues[i - 1].end_s = all_cues[i].start_s;
        }
    }
    all_cues.retain(|c| c.end_s > c.start_s);

    if !script.caption_edits.is_empty() {
        apply_edits(&mut all_cues, &script.caption_edits);
    }

    Ok(all_cues)
}

/// How far apart a cue and an edit's anchor may be and still be the same words: generated cue
/// times move a little when a clip's edges are trimmed.
const ANCHOR_TOLERANCE_S: f64 = 0.35;

/// How well an edit's anchor names a cue: lower is better, None = not this cue.
fn anchor_score(cue: &Cue, edit: &crate::script::CueEdit) -> Option<f64> {
    if cue.video_id != Some(edit.video_id) {
        return None;
    }
    let (a, b) = (cue.src_start_s?, cue.src_end_s?);
    let (da, db) = ((a - edit.src_start_s).abs(), (b - edit.src_end_s).abs());
    if da <= ANCHOR_TOLERANCE_S && db <= ANCHOR_TOLERANCE_S {
        return Some(da + db);
    }
    // A trimmed clip cuts a cue short: same words if most of both ranges overlap.
    let overlap = (b.min(edit.src_end_s) - a.max(edit.src_start_s)).max(0.0);
    let union = b.max(edit.src_end_s) - a.min(edit.src_start_s);
    (union > 0.0 && overlap / union >= 0.6).then(|| 1.0 + (1.0 - overlap / union))
}

/// Apply hand edits to generated cues: each edit to the one cue its anchor names best.
pub fn apply_edits(cues: &mut Vec<Cue>, edits: &[crate::script::CueEdit]) {
    let mut taken = vec![false; cues.len()];
    for edit in edits {
        let best = cues
            .iter()
            .enumerate()
            .filter(|(i, _)| !taken[*i])
            .filter_map(|(i, c)| anchor_score(c, edit).map(|s| (i, s)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let Some((i, _)) = best else { continue };
        taken[i] = true;
        let cue = &mut cues[i];
        let offset = cue.offset_s.unwrap_or(0.0);
        let (lo, hi) = (cue.min_s.unwrap_or(f64::MIN), cue.max_s.unwrap_or(f64::MAX));
        if let Some(text) = edit.text.as_ref().filter(|t| !t.trim().is_empty()) {
            cue.text = text.trim().to_string();
        }
        if let Some(s) = edit.start_s {
            cue.start_s = (s + offset).clamp(lo, hi);
        }
        if let Some(e) = edit.end_s {
            cue.end_s = (e + offset).clamp(lo, hi);
        }
        if cue.end_s < cue.start_s + 0.1 {
            cue.end_s = (cue.start_s + 0.1).min(hi.max(cue.start_s + 0.1));
        }
        cue.edited = edit.text.is_some() || edit.start_s.is_some() || edit.end_s.is_some();
        cue.hidden = edit.hidden;
    }
    cues.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(std::cmp::Ordering::Equal));
    // Retimed cues may now overlap a neighbour: the one moved by hand keeps its time.
    let visible: Vec<usize> = (0..cues.len()).filter(|&i| !cues[i].hidden).collect();
    for w in visible.windows(2) {
        let (i, j) = (w[0], w[1]);
        if cues[i].end_s > cues[j].start_s {
            if cues[i].edited && !cues[j].edited {
                cues[j].start_s = cues[i].end_s;
            } else {
                cues[i].end_s = cues[j].start_s;
            }
        }
    }
    cues.retain(|c| c.end_s > c.start_s + 0.01);
}

/// Convert caption cues to standard SubRip (.srt) format.
pub fn to_srt(cues: &[Cue]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        let num = i + 1;
        let start = format_srt_time(cue.start_s);
        let end = format_srt_time(cue.end_s);
        write!(&mut out, "{num}\n{start} --> {end}\n{}\n\n", cue.text).unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{Audio, AudioBed, Beat, ScriptClip};

    fn make_script(clips: Vec<ScriptClip>, bed: Option<AudioBed>) -> Script {
        Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "Caption Test".to_string(),
            target_duration_s: None,
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![Beat {
                id: "b1".to_string(),
                purpose: "test".to_string(),
                narration: None,
                on_screen_text: None,
                title_layout: None,
                clips,
                notes: None,
                bed,
            }],
        }
    }

    #[test]
    fn test_mapping_offsets() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash_1', 100, 30.0, 25.0, 1)",
                [],
            )
            .unwrap();

        // Transcript segment from 5.0 to 10.0 in the video file
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 5.0, 10.0, 'hello from five to ten')",
                [],
            )
            .unwrap();

        // Clip from 6.0 to 9.0 in source video, starting at timeline 0.0
        let script = make_script(
            vec![ScriptClip {
                framing: None,
                video_id: 1,
                in_s: 6.0,
                out_s: 9.0,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            None,
        );

        let cues = cues_for_script(&db, &script).unwrap();
        assert_eq!(cues.len(), 1);
        // Mapped offset: clipped to [6.0, 9.0], duration = 3.0, timeline_start = 0.0 + (6.0 - 6.0) = 0.0, end = 3.0
        assert!((cues[0].start_s - 0.0).abs() < 1e-4);
        assert!((cues[0].end_s - 3.0).abs() < 1e-4);
        // The clip holds 6–9 s of a 5–10 s segment: the middle words, not all of them.
        assert_eq!(cues[0].text, "From five to");
    }

    #[test]
    fn test_long_segment_split_monotonic_times() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash_1', 100, 60.0, 25.0, 1)",
                [],
            )
            .unwrap();

        // Long segment of ~150 chars spanning 15.0 seconds
        let long_text = "this is a very long segment of speech that has multiple sentences and extends over many seconds to test if the caption splitting algorithm produces cues at most eighty four characters";
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 0.0, 15.0, ?1)",
                rusqlite::params![long_text],
            )
            .unwrap();

        let script = make_script(
            vec![ScriptClip {
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 15.0,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            None,
        );

        let cues = cues_for_script(&db, &script).unwrap();
        assert!(cues.len() > 1, "Should be split into multiple cues");

        let mut prev_time = 0.0;
        for (i, cue) in cues.iter().enumerate() {
            // Text constraints
            assert!(cue.text.chars().count() <= 84, "Cue text length <= 84 chars");
            let lines: Vec<&str> = cue.text.split('\n').collect();
            assert!(lines.len() <= 2, "At most 2 lines");
            for l in &lines {
                assert!(l.chars().count() <= 42, "Line length <= 42 chars");
            }

            // Monotonic times
            assert!(cue.start_s >= prev_time, "Monotonic start time");
            assert!(cue.end_s > cue.start_s, "Positive duration");
            prev_time = cue.end_s;

            // First cue capitalized
            if i == 0 {
                assert!(cue.text.starts_with('T'), "First cue capitalized");
            }
        }
        // Total duration covers the clip
        assert!((cues.last().unwrap().end_s - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_muted_clip_produces_no_cues() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash_1', 100, 30.0, 25.0, 1)",
                [],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (1, 0.0, 5.0, 'muted speech')",
                [],
            )
            .unwrap();

        let script = make_script(
            vec![ScriptClip {
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 5.0,
                audio: Audio::Mute,
                why: None,
                tracks: None,
            }],
            None,
        );

        let cues = cues_for_script(&db, &script).unwrap();
        assert!(cues.is_empty(), "Muted clip must produce no cues");
    }

    #[test]
    fn test_bed_produces_cues() {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash_pic', 100, 30.0, 25.0, 1), (2, 'hash_bed', 100, 30.0, 25.0, 1)",
                [],
            )
            .unwrap();

        // Video 2 is the bed
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text)
                 VALUES (2, 2.0, 6.0, 'interview voice under b-roll')",
                [],
            )
            .unwrap();

        let script = make_script(
            vec![ScriptClip {
                framing: None,
                video_id: 1,
                in_s: 0.0,
                out_s: 4.0,
                audio: Audio::Source,
                why: None,
                tracks: None,
            }],
            Some(AudioBed { video_id: 2, in_s: 2.0, out_s: 6.0, why: None, inferred: false, tracks: None }),
        );

        let cues = cues_for_script(&db, &script).unwrap();
        assert_eq!(cues.len(), 1, "Bed should produce a cue");
        assert_eq!(cues[0].text, "Interview voice under b-roll");
        assert!((cues[0].start_s - 0.0).abs() < 1e-4);
        assert!((cues[0].end_s - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_to_srt_format() {
        let cues = vec![
            Cue { start_s: 1.5, end_s: 4.25, text: "Hello world\nSecond line".to_string(), ..Cue::default() },
            Cue { start_s: 5.0, end_s: 8.0, text: "Another cue".to_string(), ..Cue::default() },
        ];

        let srt = to_srt(&cues);
        let expected = "\
1\n\
00:00:01,500 --> 00:00:04,250\n\
Hello world\n\
Second line\n\
\n\
2\n\
00:00:05,000 --> 00:00:08,000\n\
Another cue\n\
\n";
        assert_eq!(srt, expected);
    }

    #[test]
    fn a_partly_covered_segment_keeps_only_its_words() {
        let text = "uno dos tres cuatro cinco seis siete ocho nueve diez";
        assert_eq!(words_in_range(text, 0.0, 10.0, 0.0, 10.0), text);
        let first = words_in_range(text, 0.0, 10.0, 0.0, 5.0);
        let second = words_in_range(text, 0.0, 10.0, 5.0, 10.0);
        assert!(first.starts_with("uno") && !first.contains("diez"), "{first}");
        assert!(second.ends_with("diez") && !second.contains("uno"), "{second}");
        assert_eq!(format!("{first} {second}"), text, "every word lands on one side");
    }

    fn two_clip_db() -> Db {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio) VALUES (1, 'h1', 100, 60.0, 25.0, 1), (2, 'h2', 100, 60.0, 25.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1, 10.0, 12.0, 'first words'), (2, 30.0, 33.0, 'second voice speaks')",
                [],
            )
            .unwrap();
        db
    }

    fn clip(video_id: i64, in_s: f64, out_s: f64) -> ScriptClip {
        ScriptClip { framing: None, video_id, in_s, out_s, audio: Audio::Source, why: None, tracks: None }
    }

    #[test]
    fn cues_carry_their_source_anchor() {
        let db = two_clip_db();
        let script = make_script(vec![clip(1, 10.0, 12.0), clip(2, 30.0, 33.0)], None);
        let cues = all_cues_for_script(&db, &script).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[1].video_id, Some(2));
        assert!((cues[1].src_start_s.unwrap() - 30.0).abs() < 1e-6);
        assert!((cues[1].start_s - 2.0).abs() < 1e-6);
        assert!((cues[1].offset_s.unwrap() - (2.0 - 30.0)).abs() < 1e-6);
        assert_eq!((cues[1].min_s, cues[1].max_s), (Some(2.0), Some(5.0)));
    }

    #[test]
    fn a_caption_edit_follows_its_words_when_clips_are_reordered() {
        let db = two_clip_db();
        let mut script = make_script(vec![clip(1, 10.0, 12.0), clip(2, 30.0, 33.0)], None);
        script.caption_edits = vec![
            crate::script::CueEdit {
                video_id: 2,
                src_start_s: 30.0,
                src_end_s: 33.0,
                text: Some("Second voice, fixed".into()),
                start_s: Some(30.5),
                end_s: Some(32.0),
                hidden: false,
            },
            crate::script::CueEdit { video_id: 1, src_start_s: 10.0, src_end_s: 12.0, hidden: true, ..Default::default() },
        ];
        let cues = cues_for_script(&db, &script).unwrap();
        assert_eq!(cues.len(), 1, "the hidden cue is left out");
        assert_eq!(cues[0].text, "Second voice, fixed");
        assert!((cues[0].start_s - 2.5).abs() < 1e-6 && (cues[0].end_s - 4.0).abs() < 1e-6, "{:?}", cues[0]);

        // Swap the clips: video 2 now plays first, and the edit goes with it.
        script.beats[0].clips.reverse();
        let all = all_cues_for_script(&db, &script).unwrap();
        let fixed = all.iter().find(|c| c.video_id == Some(2)).unwrap();
        assert_eq!(fixed.text, "Second voice, fixed");
        assert!((fixed.start_s - 0.5).abs() < 1e-6 && (fixed.end_s - 2.0).abs() < 1e-6, "{fixed:?}");
        assert!(fixed.edited);
        let hidden = all.iter().find(|c| c.video_id == Some(1)).unwrap();
        assert!(hidden.hidden && hidden.start_s >= 3.0);

        // Trimming the clip a little keeps the match; cutting the words out drops the edit quietly.
        script.beats[0].clips[0].in_s = 30.2;
        let all = all_cues_for_script(&db, &script).unwrap();
        assert!(all.iter().any(|c| c.text == "Second voice, fixed"));
        script.beats[0].clips.remove(0);
        assert!(cues_for_script(&db, &script).unwrap().is_empty());
    }

    #[test]
    fn an_edit_cannot_move_a_cue_out_of_its_clip() {
        let db = two_clip_db();
        let mut script = make_script(vec![clip(1, 10.0, 12.0), clip(2, 30.0, 33.0)], None);
        script.caption_edits = vec![crate::script::CueEdit {
            video_id: 1,
            src_start_s: 10.0,
            src_end_s: 12.0,
            start_s: Some(5.0),
            end_s: Some(20.0),
            ..Default::default()
        }];
        let cues = cues_for_script(&db, &script).unwrap();
        assert!((cues[0].start_s - 0.0).abs() < 1e-6 && (cues[0].end_s - 2.0).abs() < 1e-6, "{:?}", cues[0]);
    }

    #[test]
    fn caption_shape_sets_line_length_and_count() {
        let long = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce";
        let cues = split_piece_into_cues(long, 0.0, 2.0, CueShape { max_chars: 20, max_lines: 1 });
        assert!(cues.len() >= 4, "{cues:?}");
        for c in &cues {
            assert!(!c.text.contains('\n') && c.text.chars().count() <= 20, "{c:?}");
        }
    }
}
