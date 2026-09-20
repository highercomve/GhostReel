//! What a finished script is like, as numbers, and one number for "is this a good one".
//!
//! Until now the only judge of a script was a person watching the preview. That made every
//! question about the harness unanswerable: whether richer tool descriptions helped, whether a
//! prompt change was worth keeping, whether one brain beats another. Worse, the same model on the
//! same brief produced cuts 20 % and 97 % over their target, so a single run says almost nothing.
//!
//! These measure the things we have actually been burnt by, and each one names a pass that exists
//! to keep it at zero.

use crate::db::Db;
use crate::script::{Audio, Issue, IssueSeverity, Script};

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ScriptMetrics {
    pub total_s: f64,
    pub target_s: Option<f64>,
    /// `(total - target) / target`, signed: negative is short, which is the worse failure — a cut
    /// that comes in under can only be fixed by holding shots after the voice has stopped.
    pub duration_error: Option<f64>,
    pub beats: usize,
    pub clips: usize,
    /// Distinct videos whose own sound is heard: `source` clips, plus each beat's bed.
    pub speaking_sources: usize,
    /// How many videos in the project have any speech at all — the ceiling the above is judged
    /// against, so "one voice" is only a fault when there were others to choose.
    pub speaking_sources_available: usize,
    pub speech_s: f64,
    /// Clips whose out point falls strictly inside a transcript segment. `end_on_sentences` exists
    /// to drive this to zero; any non-zero value is a regression in a named pass.
    pub mid_sentence_cuts: usize,
    /// Seconds of picture with nothing to hear: no source audio, no bed over it, no narration.
    pub silent_picture_s: f64,
    /// Clips the model wrote that did not survive the repair passes.
    pub dropped_clips: usize,
    /// Beats with no purpose, or a purpose repeated word for word from another beat.
    pub beats_without_purpose: usize,
    pub errors: usize,
    pub warnings: usize,
    /// Repairs applied. Heavy repair is a smell, not a fault.
    pub repairs: usize,
}

/// Measure a finished script. `draft` is the script as the model wrote it where that is known, so
/// `dropped_clips` can be counted; without it the field is 0 rather than wrong.
pub fn measure(db: &Db, draft: Option<&Script>, script: &Script, issues: &[Issue]) -> ScriptMetrics {
    let mut m = ScriptMetrics {
        total_s: script.total_duration_s(),
        target_s: script.target_duration_s,
        beats: script.beats.len(),
        clips: script.clip_count(),
        ..Default::default()
    };
    if let Some(target) = script.target_duration_s.filter(|t| *t > 0.0) {
        m.duration_error = Some((m.total_s - target) / target);
    }
    if let Some(d) = draft {
        m.dropped_clips = d.clip_count().saturating_sub(script.clip_count());
    }

    let mut voices = std::collections::HashSet::new();
    for beat in &script.beats {
        let narrated = beat.narration.as_deref().map(str::trim).is_some_and(|n| !n.is_empty());
        let bed_s = beat.bed.as_ref().map(|b| b.duration_s()).unwrap_or(0.0);
        if let Some(bed) = &beat.bed {
            voices.insert(bed.video_id);
            m.speech_s += bed.duration_s();
            if ends_mid_sentence(db, bed.video_id, bed.in_s, bed.out_s) {
                m.mid_sentence_cuts += 1;
            }
        }

        let mut beat_s = 0.0;
        for c in &beat.clips {
            let dur = (c.out_s - c.in_s).max(0.0);
            beat_s += dur;
            if c.audio == Audio::Source && beat.bed.is_none() {
                voices.insert(c.video_id);
                m.speech_s += dur;
                if ends_mid_sentence(db, c.video_id, c.in_s, c.out_s) {
                    m.mid_sentence_cuts += 1;
                }
            }
        }

        // Picture with nothing over it. A bed covers the front of its beat; narration covers the
        // whole of it, since it is read across the beat.
        if !narrated {
            let heard = if beat.bed.is_some() {
                bed_s
            } else {
                beat.clips.iter().filter(|c| c.audio == Audio::Source).map(|c| (c.out_s - c.in_s).max(0.0)).sum()
            };
            m.silent_picture_s += (beat_s - heard).max(0.0);
        }
    }
    m.speaking_sources = voices.len();
    m.speaking_sources_available = db
        .conn
        .query_row("SELECT COUNT(DISTINCT video_id) FROM transcript_segments", [], |r| r.get::<_, i64>(0))
        .unwrap_or(0)
        .max(0) as usize;

    let mut purposes: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for beat in &script.beats {
        let p = beat.purpose.trim();
        if p.is_empty() || p.eq_ignore_ascii_case(beat.id.trim()) {
            m.beats_without_purpose += 1;
        } else {
            *purposes.entry(p).or_default() += 1;
        }
    }
    // Every repetition past the first is a beat that does not move the story on.
    m.beats_without_purpose += purposes.values().map(|n| n - 1).sum::<usize>();

    for issue in issues {
        match issue.severity {
            IssueSeverity::Error => m.errors += 1,
            IssueSeverity::Warning => m.warnings += 1,
            IssueSeverity::Info => m.repairs += 1,
        }
    }
    m
}

/// Does this range stop while someone is still speaking?
///
/// A range that ends on a sentence boundary still reaches a little into the next one: the cut runs
/// past the last word on purpose, so its decay is not chopped off. Touching the next sentence is
/// not cutting into it — only a range that plays a real part of a sentence and then stops counts,
/// which is what a listener would call being cut off.
const CUT_INTO_SENTENCE_S: f64 = 0.6;

fn ends_mid_sentence(db: &Db, video_id: i64, _in_s: f64, out_s: f64) -> bool {
    db.conn
        .query_row(
            "SELECT COUNT(*) FROM transcript_segments
              WHERE video_id = ?1 AND end_s > ?2 + 0.25 AND start_s < ?2 - ?3",
            rusqlite::params![video_id, out_s, CUT_INTO_SENTENCE_S],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ScriptScore {
    pub total: f64,
    /// Which axis cost what. A bare total hides a change that trades one failure for another.
    pub parts: Vec<(&'static str, f64)>,
}

/// One number for "is this a good script", 0–100.
///
/// The weights are constants, not config: a score each machine tunes is a score nobody can
/// compare. They are ordered by what has actually gone wrong here — length was the loudest
/// failure, a cut sentence the one the editor minds most.
pub fn score(m: &ScriptMetrics) -> ScriptScore {
    let mut parts = Vec::new();
    let mut penalty = |name: &'static str, amount: f64, cap: f64| {
        let p = amount.min(cap);
        if p > 0.0 {
            parts.push((name, p));
        }
        p
    };

    let mut total = 100.0;
    total -= penalty("errors", m.errors as f64 * 30.0, 60.0);
    total -= penalty("duration", m.duration_error.map(|e| 60.0 * (e.abs() / 0.5).min(1.0)).unwrap_or(0.0), 60.0);
    total -= penalty("mid-sentence cuts", m.mid_sentence_cuts as f64 * 6.0, 18.0);
    total -= penalty("dropped clips", m.dropped_clips as f64 * 4.0, 20.0);
    // A little silence is editing; a third of the piece is a fault.
    let silent_over = (m.silent_picture_s - m.total_s * 0.10).max(0.0);
    total -= penalty("silent picture", silent_over / 2.0, 15.0);
    total -= penalty("beats without purpose", m.beats_without_purpose as f64 * 5.0, 15.0);
    total -= penalty("warnings", m.warnings as f64 * 1.5, 15.0);
    if m.speaking_sources <= 1 && m.speaking_sources_available >= 3 {
        total -= penalty("one voice", 8.0, 8.0);
    }
    total -= penalty("repairs", m.repairs as f64 * 0.25, 6.0);

    ScriptScore { total: total.clamp(0.0, 100.0), parts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{AudioBed, Beat, Fps, ScriptClip};

    fn db_with_speech() -> Db {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,120.0)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (2,'b',1,120.0)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (3,'c',1,120.0)", []).unwrap();
        // Three people talk; sentences are ten seconds each.
        for v in 1..=3 {
            for i in 0..6 {
                db.conn
                    .execute(
                        "INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (?1, ?2, ?3, 'a sentence')",
                        rusqlite::params![v, i as f64 * 10.0, (i + 1) as f64 * 10.0],
                    )
                    .unwrap();
            }
        }
        db
    }

    fn script_of(beats: Vec<Beat>, target: Option<f64>) -> Script {
        Script {
            title: "t".into(),
            target_duration_s: target,
            fps: Some(Fps::new(25, 1)),
            width: None,
            height: None,
            beats,
        }
    }

    fn beat(id: &str, clips: Vec<ScriptClip>, bed: Option<AudioBed>) -> Beat {
        Beat {
            id: id.into(),
            purpose: format!("purpose of {id}"),
            narration: None,
            on_screen_text: None,
            notes: None,
            clips,
            bed,
        }
    }

    fn clip(video_id: i64, in_s: f64, out_s: f64, audio: Audio) -> ScriptClip {
        ScriptClip { video_id, in_s, out_s, audio, why: None }
    }

    #[test]
    fn a_clean_cut_scores_near_a_hundred() {
        let db = db_with_speech();
        let s = script_of(
            vec![
                beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source)], None),
                beat("b2", vec![clip(2, 0.0, 10.0, Audio::Source)], None),
            ],
            Some(20.0),
        );
        let m = measure(&db, None, &s, &[]);
        assert_eq!(m.mid_sentence_cuts, 0);
        assert_eq!(m.speaking_sources, 2);
        assert_eq!(m.duration_error, Some(0.0));
        assert!(score(&m).total > 99.0, "{:?}", score(&m));
    }

    /// The failure the editor minds most, and the one `end_on_sentences` exists to prevent.
    #[test]
    fn a_clip_that_stops_mid_sentence_is_counted_and_costs() {
        let db = db_with_speech();
        let s = script_of(vec![beat("b1", vec![clip(1, 0.0, 7.0, Audio::Source)], None)], Some(7.0));
        let m = measure(&db, None, &s, &[]);
        assert_eq!(m.mid_sentence_cuts, 1, "the sentence runs to 10 s");
        let sc = score(&m);
        assert!(sc.parts.iter().any(|(n, _)| *n == "mid-sentence cuts"));
        assert!(sc.total < 95.0);
    }

    /// The measured failure that started all of this: 97 % over target.
    #[test]
    fn missing_the_target_length_dominates_the_score() {
        let db = db_with_speech();
        let long = script_of(
            vec![beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source), clip(2, 0.0, 10.0, Audio::Source)], None)],
            Some(10.0),
        );
        let m = measure(&db, None, &long, &[]);
        assert_eq!(m.duration_error, Some(1.0), "20 s against a 10 s ask");
        let sc = score(&m);
        assert_eq!(sc.parts.iter().find(|(n, _)| *n == "duration").map(|(_, p)| *p), Some(60.0), "capped");
        assert!(sc.total <= 40.0);
    }

    /// "Built the whole teaser out of whoever it found first" — only a fault when there were others.
    #[test]
    fn one_voice_costs_only_when_the_project_had_more() {
        let db = db_with_speech();
        let one = script_of(vec![beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source)], None)], Some(10.0));
        let m = measure(&db, None, &one, &[]);
        assert_eq!(m.speaking_sources_available, 3);
        assert!(score(&m).parts.iter().any(|(n, _)| *n == "one voice"));

        let quiet = Db::open_in_memory().unwrap();
        quiet.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,60.0)", []).unwrap();
        quiet
            .conn
            .execute("INSERT INTO transcript_segments(video_id, start_s, end_s, text) VALUES (1,0.0,10.0,'x')", [])
            .unwrap();
        let m = measure(&quiet, None, &one, &[]);
        assert!(!score(&m).parts.iter().any(|(n, _)| *n == "one voice"), "nothing else to cut to");
    }

    /// Silent b-roll: the thing a bed exists to prevent.
    #[test]
    fn picture_with_nothing_to_hear_is_counted() {
        let db = db_with_speech();
        let silent = script_of(
            vec![beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source), clip(2, 0.0, 20.0, Audio::Mute)], None)],
            Some(30.0),
        );
        let m = measure(&db, None, &silent, &[]);
        assert!((m.silent_picture_s - 20.0).abs() < 0.01, "{}", m.silent_picture_s);

        // The same pictures with the voice carried under them: nothing silent.
        let bedded = script_of(
            vec![beat(
                "b1",
                vec![clip(1, 0.0, 10.0, Audio::Mute), clip(2, 0.0, 20.0, Audio::Mute)],
                Some(AudioBed { video_id: 1, in_s: 0.0, out_s: 30.0, why: None, inferred: true }),
            )],
            Some(30.0),
        );
        let m = measure(&db, None, &bedded, &[]);
        assert!(m.silent_picture_s < 0.01, "{}", m.silent_picture_s);
    }

    #[test]
    fn a_gutted_draft_is_counted_against_the_draft_not_the_result() {
        let db = db_with_speech();
        let draft = script_of(
            vec![beat(
                "b1",
                vec![
                    clip(1, 0.0, 10.0, Audio::Source),
                    clip(2, 0.0, 10.0, Audio::Mute),
                    clip(3, 0.0, 10.0, Audio::Mute),
                ],
                None,
            )],
            Some(10.0),
        );
        let kept = script_of(vec![beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source)], None)], Some(10.0));
        let m = measure(&db, Some(&draft), &kept, &[]);
        assert_eq!(m.dropped_clips, 2);
        assert!(score(&m).parts.iter().any(|(n, _)| *n == "dropped clips"));
    }

    #[test]
    fn beats_that_do_not_move_the_story_are_counted() {
        let db = db_with_speech();
        let mut s = script_of(
            vec![
                beat("b1", vec![clip(1, 0.0, 10.0, Audio::Source)], None),
                beat("b2", vec![clip(2, 0.0, 10.0, Audio::Source)], None),
                beat("b3", vec![clip(3, 0.0, 10.0, Audio::Source)], None),
            ],
            Some(30.0),
        );
        s.beats[1].purpose = s.beats[0].purpose.clone(); // said twice
        s.beats[2].purpose = String::new(); // said not at all
        let m = measure(&db, None, &s, &[]);
        assert_eq!(m.beats_without_purpose, 2);
    }
}
