//! Finding the person asking the questions, by what they say rather than how loud they are.
//!
//! `audio.rs` decides who is off-mic acoustically: a segment whose level sits a margin below the
//! video's median is somebody away from the lav. That is the right first test and it is cheap,
//! but it only works when the interviewer is *quieter*. On one interview in the Greet Mag
//! footage they were not — nine segments of seventy-six came back off-mic against twenty-odd on
//! every other tape — and the cut opened with:
//!
//! > "Okay, cool. So just tell me your name and the line of business that you're in. Okay."
//!
//! The editor's verdict was "it's like I asked for bloopers". Nothing downstream could have
//! caught it: the prompt, the quote candidates and the judge all read `off_mic`, and the index
//! had said this was the subject talking.
//!
//! What gives an interviewer away is the words. "Tell me your name", "so first off", "what do you
//! think about today's event" — that is a judgement about meaning, one segment at a time, which
//! is what a System One model is for. One request carries a hundred of them.
//!
//! This only ever *adds* off-mic flags. The acoustic test is evidence too, and a segment it
//! already caught is not re-litigated.

use std::collections::BTreeMap;

use rusqlite::params;
use serde_json::json;

use crate::Error;
use crate::config::JevConfig;
use crate::db::Db;
use crate::jev::{Jev, Question};

/// Segments asked about in one request, within a single video. The state is short — a line of
/// dialogue each — so this is bounded by the number of questions rather than the context.
///
/// Never spanning two videos, which the first run made expensive: batched across the project, the
/// line "So just tell me your name and the line of business that you're in" came back at 0.53 and
/// survived. Asked among its own interview it is 0.97. "The surrounding lines are given for
/// context" is only true if the surrounding lines are the same conversation.
const BATCH: usize = 100;

/// One segment considered, and what came back.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub video_id: i64,
    pub start_s: f64,
    pub text: String,
    /// Probability this is the interviewer rather than the subject.
    pub p: f64,
}

impl Verdict {
    pub fn is_interviewer(&self, cfg: &JevConfig) -> bool {
        self.p >= cfg.interviewer_threshold
    }
}

/// Segments still believed to be the subject, which are the only ones worth asking about.
///
/// Ordered by video then time, so a batch is a run of consecutive dialogue and each question can
/// be read in the context of its neighbours.
fn candidates(db: &Db, project_id: i64) -> Result<Vec<(i64, i64, f64, String)>, Error> {
    let rows = db
        .conn
        .prepare(
            "SELECT ts.rowid, ts.video_id, ts.start_s, ts.text
             FROM transcript_segments ts
             JOIN video_files vf ON vf.video_id = ts.video_id
             JOIN folders f ON f.id = vf.folder_id
             JOIN project_folders pf ON pf.folder_id = f.id
             WHERE pf.project_id = ?1 AND COALESCE(ts.off_mic, 0) = 0 AND TRIM(ts.text) <> ''
             GROUP BY ts.rowid
             ORDER BY ts.video_id, ts.start_s",
        )?
        .query_map(params![project_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// The question asked of every line. Written once, here, so a change is a change to the whole
/// measurement rather than to one call site.
fn question(i: usize) -> Question {
    Question::Noul {
        instructions: json!({
            "line": format!("lines[{i}]"),
            "question": format!(
                "In this interview transcript, is `lines[{i}]` spoken by the INTERVIEWER — the \
                 person running the interview — rather than by the subject being interviewed? \
                 The surrounding lines are given for context; judge only that one."
            ),
        }),
        criteria: Some(json!({
            "true": "The interviewer: asking a question, prompting ('so first off', 'tell me about', \
                     'what do you think'), giving direction, checking the recording, or \
                     acknowledging an answer ('okay', 'perfect', 'great', 'got it').",
            "false": "The subject: answering, telling their own story, describing their business, \
                      their neighbourhood or their life. A subject may ask a rhetorical question \
                      of their own or repeat part of the question back while answering it.",
        })),
    }
}

/// Ask about every segment in the project that still looks like the subject.
///
/// Returns every verdict, not only the interviewer's, so a caller can show what was considered
/// and at what confidence rather than only what changed.
pub async fn find(db: &Db, project_id: i64, cfg: &JevConfig) -> Result<Vec<Verdict>, Error> {
    let client = Jev::from_config(cfg)
        .ok_or_else(|| Error::Jev("finding the interviewer needs Jev: set jev.enabled and an API key".into()))?;
    let rows = candidates(db, project_id)?;
    let mut out = Vec::with_capacity(rows.len());

    // One conversation at a time.
    let mut by_video: Vec<Vec<&(i64, i64, f64, String)>> = Vec::new();
    for row in &rows {
        match by_video.last_mut() {
            Some(g) if g.last().is_some_and(|l| l.1 == row.1) && g.len() < BATCH => g.push(row),
            _ => by_video.push(vec![row]),
        }
    }

    for batch in by_video {
        // Every line of the batch is in the state, so each question can be judged against what
        // was said either side of it — "Okay." is the subject agreeing or the interviewer moving
        // on, and only the neighbours say which.
        let state = json!({
            "what_this_is": "A transcript of one recorded interview, in order. Two people speak: an \
                             interviewer running the interview, and the subject answering.",
            "lines": batch.iter().map(|(_, _, _, t)| t.trim()).collect::<Vec<_>>(),
        });
        let questions: BTreeMap<String, Question> =
            (0..batch.len()).map(|i| (format!("line_{i}"), question(i))).collect();

        let answers = client.ask(&state, &questions).await?;
        for (i, (_, video_id, start_s, text)) in batch.iter().enumerate() {
            let Some(p) = answers.unit(&format!("line_{i}")) else { continue };
            out.push(Verdict { video_id: *video_id, start_s: *start_s, text: text.clone(), p });
        }
    }
    Ok(out)
}

/// Mark the interviewer's lines off-mic. Returns how many were newly flagged.
///
/// Only ever sets the flag. The acoustic test in `audio.rs` is evidence as well, and a segment it
/// already caught is left alone — two different ways of being the wrong voice, one column.
pub fn apply(db: &Db, verdicts: &[Verdict], cfg: &JevConfig) -> Result<usize, Error> {
    let mut n = 0;
    for v in verdicts.iter().filter(|v| v.is_interviewer(cfg)) {
        // 'speech' with the probability that produced it, so this can be reviewed, re-judged at a
        // different threshold, or undone — none of which is possible from one bit. The `off_mic`
        // guard keeps an acoustic flag's own provenance intact.
        n += db.conn.execute(
            "UPDATE transcript_segments SET off_mic = 1, off_mic_source = 'speech', off_mic_p = ?3
             WHERE video_id = ?1 AND ABS(start_s - ?2) < 0.001 AND COALESCE(off_mic, 0) = 0",
            params![v.video_id, v.start_s, v.p],
        )?;
    }
    Ok(n)
}

/// Undo what this pass did, leaving the acoustic flags alone. Returns how many were cleared.
///
/// The point of recording the source: a threshold changed on evidence should be re-runnable, and
/// an editor who disagrees with a call should not have to re-measure every video's levels.
pub fn undo(db: &Db, project_id: i64) -> Result<usize, Error> {
    Ok(db.conn.execute(
        "UPDATE transcript_segments SET off_mic = 0, off_mic_source = NULL, off_mic_p = NULL
         WHERE off_mic_source = 'speech' AND video_id IN (
             SELECT vf.video_id FROM video_files vf
             JOIN folders f ON f.id = vf.folder_id
             JOIN project_folders pf ON pf.folder_id = f.id
             WHERE pf.project_id = ?1)",
        params![project_id],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    fn project_with_an_interview() -> (Db, i64) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("T")).unwrap();
        let tmp = std::env::temp_dir();
        let folder = db.add_folder(project.id, &tmp, true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1,'a',1,60)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, ?1, ?2, 1, 0, 0)",
                params![folder.id, format!("{}/1.mp4", tmp.display())],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 0.0, 2.0, 'Okay, cool.', 0),
                 (1, 2.0, 7.0, 'So just tell me your name and the line of business you are in.', 0),
                 (1, 8.0, 17.0, 'I am Adrienne and my husband and I own a lounge north of campus.', 0),
                 (1, 17.0, 18.0, 'Perfect.', 1)",
                [],
            )
            .unwrap();
        (db, project.id)
    }

    #[test]
    fn only_lines_still_believed_to_be_the_subject_are_asked_about() {
        let (db, project_id) = project_with_an_interview();
        let rows = candidates(&db, project_id).unwrap();
        // "Perfect." is already off-mic acoustically; asking about it again buys nothing.
        assert_eq!(rows.len(), 3);
        assert!(!rows.iter().any(|(_, _, _, t)| t == "Perfect."));
        // In transcript order, so a batch reads as a conversation.
        assert_eq!(rows[0].2, 0.0);
        assert_eq!(rows[2].2, 8.0);
    }

    #[test]
    fn the_question_says_which_line_it_is_about_and_what_each_answer_means() {
        let json = serde_json::to_value(question(3)).unwrap();
        assert_eq!(json["type"], "noul");
        // Ids never reach the model, so the question has to point at its own line.
        assert!(json["instructions"]["question"].as_str().unwrap().contains("lines[3]"));
        assert!(json["criteria"]["true"].as_str().unwrap().contains("tell me about"));
        // And a subject repeating the question back is not the interviewer.
        assert!(json["criteria"]["false"].as_str().unwrap().contains("repeat part of the question back"));
    }

    #[test]
    fn applying_flags_the_interviewer_and_leaves_the_subject_alone() {
        let (db, _) = project_with_an_interview();
        let verdicts = vec![
            Verdict { video_id: 1, start_s: 0.0, text: "Okay, cool.".into(), p: 0.93 },
            Verdict { video_id: 1, start_s: 2.0, text: "So just tell me…".into(), p: 0.97 },
            // Below the threshold: an answer that sounds a little like a prompt stays.
            Verdict { video_id: 1, start_s: 8.0, text: "I am Adrienne…".into(), p: 0.74 },
        ];
        let cfg = JevConfig::default();
        assert_eq!(apply(&db, &verdicts, &cfg).unwrap(), 2);

        let still_subject: Vec<String> = db
            .conn
            .prepare("SELECT text FROM transcript_segments WHERE COALESCE(off_mic,0)=0 ORDER BY start_s")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(still_subject, vec!["I am Adrienne and my husband and I own a lounge north of campus."]);

        // Running it again changes nothing: the flag is set, not toggled.
        assert_eq!(apply(&db, &verdicts, &cfg).unwrap(), 0);
    }

    #[test]
    fn the_threshold_sits_in_the_gap_the_footage_showed() {
        let cfg = JevConfig::default();
        let v = |p| Verdict { video_id: 1, start_s: 0.0, text: String::new(), p };
        // Real answers wrongly caught on the Greet Mag tapes measured 0.70–0.76…
        assert!(!v(0.5).is_interviewer(&cfg), "a coin flip is not enough to delete somebody's answer");
        assert!(!v(0.76).is_interviewer(&cfg), "\"I'm newer to Austin.\" is not a question");
        // …and the unmistakable ones 0.84 and up.
        assert!(v(0.84).is_interviewer(&cfg), "a mic check is not interview content");
        assert!(v(0.97).is_interviewer(&cfg), "\"so just tell me your name\" certainly is not");
    }

    #[test]
    fn a_batch_never_spans_two_interviews() {
        // Two videos, one segment apiece: they must not be asked about as one conversation.
        let rows = vec![
            (1i64, 7i64, 0.0f64, "So tell me your name.".to_string()),
            (2, 7, 5.0, "I am Adrienne.".to_string()),
            (3, 9, 0.0, "Okay, cool.".to_string()),
        ];
        let mut by_video: Vec<Vec<&(i64, i64, f64, String)>> = Vec::new();
        for row in &rows {
            match by_video.last_mut() {
                Some(g) if g.last().is_some_and(|l| l.1 == row.1) && g.len() < BATCH => g.push(row),
                _ => by_video.push(vec![row]),
            }
        }
        assert_eq!(by_video.len(), 2, "one group per interview");
        assert_eq!(by_video[0].len(), 2);
        assert_eq!(by_video[1].len(), 1);
        assert!(by_video.iter().all(|g| g.iter().all(|r| r.1 == g[0].1)));
    }

    #[test]
    fn a_semantic_flag_records_itself_and_can_be_taken_back() {
        let (db, project_id) = project_with_an_interview();
        let cfg = JevConfig::default();
        let verdicts = vec![Verdict { video_id: 1, start_s: 2.0, text: "So just tell me…".into(), p: 0.97 }];
        assert_eq!(apply(&db, &verdicts, &cfg).unwrap(), 1);

        let (source, p): (String, f64) = db
            .conn
            .query_row(
                "SELECT off_mic_source, off_mic_p FROM transcript_segments WHERE start_s = 2.0",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(source, "speech");
        assert!((p - 0.97).abs() < 1e-9, "the probability is kept, so it can be re-judged");

        // Undone, and the acoustic flag on "Perfect." is untouched — it was measured, not read.
        assert_eq!(undo(&db, project_id).unwrap(), 1);
        let still_off: Vec<String> = db
            .conn
            .prepare("SELECT text FROM transcript_segments WHERE COALESCE(off_mic,0)=1")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(still_off, vec!["Perfect."]);
    }
}
