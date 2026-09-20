//! An editorial read on a finished cut, from a model that answers in numbers.
//!
//! [`super::metrics`] measures craft faults — seconds over target, a cut landing inside a
//! sentence, picture hanging in silence — and every one of them has a pass whose job is to hold it
//! at zero. None of that says whether the cut is any good. The complaints that actually came back
//! from watching previews were of a different kind, and no counter reaches them:
//!
//! * *"it always puts the b-roll as a pause between interviews"* — the pictures do not show what
//!   the voice is talking about;
//! * *"one thing that is always missing is a couple of seconds with a close image conclusion"* —
//!   the piece stops rather than ends;
//! * a cut that hops between four speakers and never lands on a point.
//!
//! So this asks, in one request: does each beat's picture match its voice, does the opening earn
//! the next ten seconds, does the ending land, does the whole thing hold together. Jev answers
//! each as a probability; the weights that turn those into one number live here, in code, where
//! they can be argued with and changed without asking anything again.
//!
//! Nothing here runs unless `[jev] enabled` is set and a key exists. Judging sends the cut's
//! words and its picture descriptions to a hosted service, which is a thing a local-first tool
//! does only when told to.

use std::collections::BTreeMap;

use rusqlite::params;
use serde_json::{Value, json};

use crate::Error;
use crate::config::JevConfig;
use crate::db::Db;
use crate::jev::{Answers, Jev, Question};
use crate::script::{Audio, Script};

/// What one dimension contributed, and what it could have.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Part {
    pub name: String,
    /// 0–1, as Jev answered it.
    pub value: f64,
    /// Points earned, out of `possible`.
    pub earned: f64,
    pub possible: f64,
}

/// A beat whose pictures do not show what is being heard over them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mismatch {
    pub beat_id: String,
    /// Probability the pictures match. Low is the complaint.
    pub match_p: f64,
    /// What is heard there, shortened — so a note can say *which* voice was ignored.
    pub heard: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Judgement {
    /// 0–100, on the same scale as [`super::metrics::score`] so the two can be read side by side.
    pub total: f64,
    pub parts: Vec<Part>,
    /// Beats whose pictures were judged not to match their sound, worst first.
    pub mismatched: Vec<Mismatch>,
    /// Beats nothing could be said about, because the vision model never described what is on
    /// screen there. Not a fault in the cut — a gap in the index, and worth saying out loud
    /// rather than hiding inside an average.
    pub unjudged_beats: usize,
    /// Whether the cut already closes on a held image. Changes what a weak ending means.
    pub held_closing_picture: bool,
    pub model: String,
    pub input_tokens: u64,
}

impl Judgement {
    /// The lines worth putting in front of the editor model before it redrafts.
    ///
    /// This is the point of judging during a turn rather than after one: the harness already
    /// reports its mechanical repairs back on the assistant message, because a model that is not
    /// told repeats the mistake. An editorial fault is exactly the kind a repair pass *cannot*
    /// fix — no amount of trimming makes a shot of a road illustrate a sentence about a dog — so
    /// it has to go back to the only thing that can choose a different shot.
    pub fn notes(&self) -> Vec<String> {
        let mut out = Vec::new();
        for m in self.mismatched.iter().take(4) {
            out.push(format!(
                "beat '{}': the pictures do not show what is heard over them (\"{}\") — pick shots of what is being talked about, or move this voice under pictures that fit it",
                m.beat_id, m.heard
            ));
        }
        for p in &self.parts {
            if p.name == "ending" && p.value < 0.4 {
                out.push(if self.held_closing_picture {
                    // The hold is already there, so the fault is the line under it.
                    "the piece does not land: the last thing anybody says is not a conclusion — end on a line that answers what the piece set up".into()
                } else {
                    "the cut stops rather than ends: hold a closing image after the last word".to_string()
                });
            }
            if p.name == "opening" && p.value < 0.4 {
                out.push("the opening does not earn the next ten seconds: lead with the strongest thing anybody says".into());
            }
        }
        out
    }
}

/// The ids used for the whole-cut questions. Kept in one place because the composite reads them
/// back by name and a typo would silently drop a dimension.
const OPENING: &str = "opening";
const FLOW: &str = "flow";
const ENDING: &str = "ending";
const BRIEF: &str = "covers_brief";
const REPEATS: &str = "repeats_itself";
const WHIPLASH: &str = "voice_whiplash";

fn beat_question_id(i: usize) -> String {
    format!("beat_{i}_pictures_match")
}

/// The cut as Jev reads it: what is heard, what is seen, beat by beat.
///
/// Deliberately not the script JSON. Timecodes and video ids are the wrong state for this
/// question — they say nothing about whether a shot fits a sentence, and they cost tokens that
/// the words and the picture descriptions need.
pub fn state(db: &Db, script: &Script, brief: Option<&str>, cfg: &JevConfig) -> Value {
    let cap = cfg.max_quote_chars.max(80);
    let mut beats = Vec::new();

    for beat in script.beats.iter().take(cfg.max_beats.max(1)) {
        let seconds: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();

        // What a viewer hears: the bed if there is one, otherwise the sound of any clip that is
        // not muted. Narration is the editor's own words and is carried separately, because a
        // beat can have both and they are judged against different things.
        let (heard, sound) = match &beat.bed {
            Some(bed) => (speech_text(db, bed.video_id, bed.in_s, bed.out_s, cap), "a voice continues under the pictures"),
            None => {
                let mut text = String::new();
                let mut kind = "nothing — the pictures play silent";
                for c in beat.clips.iter().filter(|c| c.audio == Audio::Source) {
                    let t = speech_text(db, c.video_id, c.in_s, c.out_s, cap);
                    if !t.is_empty() {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&t);
                        kind = "the person on screen speaking";
                    }
                }
                (truncate(&text, cap), kind)
            }
        };

        let mut seen = Vec::new();
        for c in &beat.clips {
            for s in frame_text(db, c.video_id, c.in_s, c.out_s, 2) {
                seen.push(truncate(&s, cap / 2));
            }
            if seen.len() >= 8 {
                break;
            }
        }

        let mut o = serde_json::Map::new();
        o.insert("id".into(), json!(beat.id));
        o.insert("purpose".into(), json!(beat.purpose));
        o.insert("seconds".into(), json!((seconds * 10.0).round() / 10.0));
        if let Some(n) = beat.narration.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
            o.insert("narration".into(), json!(truncate(n, cap)));
        }
        o.insert("heard".into(), json!(heard));
        o.insert("sound".into(), json!(sound));
        // An empty `seen` is not a bad shot, it is an unindexed one — the vision model never
        // described this stretch. Asking anyway got a flat "no": Jev has nothing to match against
        // and rightly says so, which would have scored a perfectly good cut as filler. The beat
        // says plainly that the pictures are unknown, and `questions` does not ask about it.
        o.insert("pictures_described".into(), json!(!seen.is_empty()));
        o.insert("seen".into(), json!(seen));
        beats.push(Value::Object(o));
    }

    let mut s = serde_json::Map::new();
    if let Some(b) = brief.map(str::trim).filter(|b| !b.is_empty()) {
        s.insert("brief".into(), json!(truncate(b, 1200)));
    }
    s.insert("title".into(), json!(script.title));
    if let Some(t) = script.target_duration_s {
        s.insert("target_seconds".into(), json!((t * 10.0).round() / 10.0));
    }
    s.insert("actual_seconds".into(), json!((script.total_duration_s() * 10.0).round() / 10.0));
    // Whether the piece closes on a held image in silence. The passes add that hold, and without
    // it in the state Jev judged an ending that stops on the last word and told us to add a hold
    // that was already there.
    if let Some(hold) = closing_hold_s(script) {
        s.insert("ends_with".into(), json!(format!("{hold:.1} s of held picture in silence after the last word")));
    }
    s.insert("beats".into(), Value::Array(beats));
    Value::Object(s)
}

/// Every question, in one map, asked against one state.
///
/// One request rather than one per beat: Jev reads the state once and answers all of them in
/// parallel, which is both the cheap way and the consistent one — every answer saw the same cut.
pub fn questions(state: &Value, has_brief: bool) -> BTreeMap<String, Question> {
    let mut qs = BTreeMap::new();
    let beats = state["beats"].as_array().map(Vec::len).unwrap_or(0);

    for i in 0..beats {
        // No description of what is on screen, no question about it. Scoring an unindexed beat
        // would measure the index rather than the cut.
        if state["beats"][i]["pictures_described"] == Value::Bool(false) {
            continue;
        }
        // The id is not sent to the model, so the question has to name its own beat. The backtick
        // path is how a question points at part of the state.
        qs.insert(
            beat_question_id(i),
            Question::Noul {
                instructions: json!({
                    "beat": i + 1,
                    "question": format!(
                        "In the video being judged, do the shots listed in `beats[{i}].seen` show what is being talked about in `beats[{i}].heard` and `beats[{i}].narration`?"
                    ),
                }),
                criteria: Some(json!({
                    "true": "The shots show the subject, place, person or action the voice is describing, or something a viewer would naturally read as an illustration of it. A shot that sets the scene the speaker is describing counts.",
                    "false": "The shots are unrelated to what is being said — filler that would play the same way under any other sentence — or there is nothing to hear over them at all, so the pictures read as a pause rather than part of the story.",
                })),
            },
        );
    }

    qs.insert(
        OPENING.into(),
        Question::score(
            "How strongly does the first beat make a viewer want to keep watching? Judge `beats[0]` — what is said and what is shown — as the opening of a short documentary piece.",
            &[
                "Nothing happens: throat-clearing, a title, or a line that could open any video",
                "A competent but ordinary start; a viewer would give it a few more seconds out of politeness",
                "A clear hook — a strong statement, a question or an image that makes the next line worth hearing",
            ],
        ),
    );
    qs.insert(
        FLOW.into(),
        Question::score(
            "Do the beats follow one another as a single piece, in this order? Judge whether each beat leads to the next rather than sitting beside it.",
            &[
                "Disconnected fragments in an order that could be shuffled without loss",
                "Loosely related: a subject holds them together but not a thread",
                "One thread: each beat follows from the one before and sets up the one after",
            ],
        ),
    );
    qs.insert(
        ENDING.into(),
        Question::score(
            "Does the last beat land as an ending? An ending gives the viewer something to take away and closes on it; a piece that merely stops leaves the last speaker mid-thought or ends on a line no more final than any other.",
            &[
                "It just stops — the last line is arbitrary and nothing closes",
                "An ending of sorts: the last line reads as a conclusion but nothing is given to take away",
                "It lands: a conclusion that answers what the piece set up, and the piece closes on it",
            ],
        ),
    );
    qs.insert(
        REPEATS.into(),
        Question::noul_between(
            "Does this cut make the same point more than once, in different beats?",
            "Two or more beats say substantially the same thing, so one of them could be removed without losing anything",
            "Each beat carries something the others do not",
        ),
    );
    qs.insert(
        WHIPLASH.into(),
        Question::noul_between(
            "Is the cut hard to follow because of how it moves between speakers and subjects?",
            "It jumps between voices or topics without a thread a first-time viewer could hold on to",
            "Changes of voice or subject are easy to follow; each one arrives where a viewer expects it",
        ),
    );
    if has_brief {
        qs.insert(
            BRIEF.into(),
            Question::noul_between(
                "Does this cut do what `brief` asked for — the subject, the angle and the kind of piece it describes?",
                "It delivers what was asked for",
                "It is about something else, or answers only a fraction of what was asked",
            ),
        );
    }
    qs
}

/// Turn the answers into one number, with the weights in the open.
///
/// The picture/voice match carries the most because it is the fault that came back the most, and
/// it is the one a repair pass provably cannot fix. Two of the dimensions are complaints rather
/// than virtues, so they count inverted.
pub fn compose(state: &Value, a: &Answers) -> Judgement {
    let beats = state["beats"].as_array().cloned().unwrap_or_default();
    let mut parts = Vec::new();
    let mut mismatched = Vec::new();

    let mut matches = Vec::new();
    let mut unjudged = 0usize;
    for (i, beat) in beats.iter().enumerate() {
        let Some(p) = a.unit(&beat_question_id(i)) else {
            unjudged += 1;
            continue;
        };
        matches.push(p);
        if p < 0.5 {
            mismatched.push(Mismatch {
                beat_id: beat["id"].as_str().unwrap_or("?").to_string(),
                match_p: p,
                heard: truncate(beat["heard"].as_str().unwrap_or(""), 90),
            });
        }
    }
    mismatched.sort_by(|x, y| x.match_p.total_cmp(&y.match_p));

    let mut total = 0.0;
    let mut possible = 0.0;
    let mut add = |name: &str, value: Option<f64>, weight: f64| {
        let Some(value) = value else { return };
        let earned = value.clamp(0.0, 1.0) * weight;
        total += earned;
        possible += weight;
        parts.push(Part { name: name.to_string(), value, earned, possible: weight });
    };

    let picture_match = (!matches.is_empty()).then(|| matches.iter().sum::<f64>() / matches.len() as f64);
    add("pictures match the voice", picture_match, 30.0);
    add("opening", a.unit(OPENING), 15.0);
    add("flow", a.unit(FLOW), 20.0);
    add("ending", a.unit(ENDING), 15.0);
    add("says what was asked", a.unit(BRIEF), 10.0);
    add("says each thing once", a.unit(REPEATS).map(|p| 1.0 - p), 5.0);
    add("easy to follow", a.unit(WHIPLASH).map(|p| 1.0 - p), 5.0);

    // Scaled to 100 by what was actually asked, so a cut judged without a brief is not punished
    // for the ten points that question would have carried.
    let scaled = if possible > 0.0 { total / possible * 100.0 } else { 0.0 };
    Judgement {
        total: scaled,
        parts,
        mismatched,
        unjudged_beats: unjudged,
        held_closing_picture: state.get("ends_with").is_some(),
        model: a.model.clone(),
        input_tokens: a.usage.as_ref().map(|u| u.input_tokens).unwrap_or(0),
    }
}

/// Everything a judgement needs, read off the database and owned outright.
///
/// Split from the asking on purpose. The database connection is not `Sync`, so a future that
/// still holds a `&Db` when it awaits is not `Send`, and the desktop app's queue worker will not
/// spawn it. Reading first and asking second keeps the borrow and the await apart.
pub struct Planned {
    client: Jev,
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

/// Read the cut, or `None` when Jev is off, unkeyed, or there is nothing to judge.
pub fn plan(db: &Db, script: &Script, brief: Option<&str>, cfg: &JevConfig) -> Option<Planned> {
    let client = Jev::from_config(cfg)?;
    if script.beats.is_empty() {
        return None;
    }
    let state = state(db, script, brief, cfg);
    let questions = questions(&state, state.get("brief").is_some());
    (!questions.is_empty()).then_some(Planned { client, state, questions })
}

impl Planned {
    pub async fn ask(&self) -> Result<Judgement, Error> {
        let answers = self.client.ask(&self.state, &self.questions).await?;
        Ok(compose(&self.state, &answers))
    }
}

/// Judge a cut, or return `None` when Jev is not configured. The convenient call, for anywhere
/// that is not holding a database open across the request.
pub async fn judge(
    db: &Db,
    script: &Script,
    brief: Option<&str>,
    cfg: &JevConfig,
) -> Result<Option<Judgement>, Error> {
    let Some(planned) = plan(db, script, brief, cfg) else { return Ok(None) };
    planned.ask().await.map(Some)
}

/// Picture that plays on after the sound has stopped, at the very end. `None` when the piece
/// stops on the last word.
fn closing_hold_s(script: &Script) -> Option<f64> {
    let last = script.beats.last()?;
    let pictures: f64 = last.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
    let sound = match &last.bed {
        Some(bed) => bed.duration_s(),
        None => last.clips.iter().filter(|c| c.audio == Audio::Source).map(|c| (c.out_s - c.in_s).max(0.0)).sum(),
    };
    (pictures - sound > 0.3).then_some(pictures - sound)
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    match cut.rfind(' ') {
        Some(i) if i > max / 2 => format!("{}…", &cut[..i]),
        _ => format!("{cut}…"),
    }
}

/// The words spoken in a range, as one line.
fn speech_text(db: &Db, video_id: i64, in_s: f64, out_s: f64, cap: usize) -> String {
    let Ok(mut st) = db.conn.prepare(
        "SELECT text FROM transcript_segments
         WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3 AND COALESCE(off_mic, 0) = 0
         ORDER BY start_s",
    ) else {
        return String::new();
    };
    let Ok(rows) = st.query_map(params![video_id, in_s, out_s], |r| r.get::<_, String>(0)) else {
        return String::new();
    };
    let mut text = String::new();
    for t in rows.flatten() {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(t.trim());
        if text.chars().count() > cap {
            break;
        }
    }
    truncate(&text, cap)
}

/// What the keyframes in a range show. The vision model's own words, which is all Jev can read —
/// it takes text, never pictures.
fn frame_text(db: &Db, video_id: i64, in_s: f64, out_s: f64, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    // Falling back to the last frame *before* the range is not a guess. Keyframes are de-duped by
    // perceptual hash, so a stretch with no frame of its own is a stretch where nothing changed —
    // a locked-off interview collapses to one frame every 16–24 s, and the last one kept is a
    // literal description of what is still on screen. Without this a bedded beat over a static
    // camera looked unindexed and the judge scored it on its cutaway alone.
    let Ok(mut st) = db.conn.prepare(
        "SELECT description_json FROM frames
         WHERE video_id = ?1 AND t_s <= ?3 AND description_json IS NOT NULL
           AND (t_s >= ?2 OR t_s = (SELECT MAX(t_s) FROM frames
                                    WHERE video_id = ?1 AND t_s < ?2 AND description_json IS NOT NULL))
         ORDER BY t_s",
    ) else {
        return out;
    };
    let Ok(rows) = st.query_map(params![video_id, in_s - 0.5, out_s + 0.5], |r| r.get::<_, String>(0)) else {
        return out;
    };
    for row in rows.flatten() {
        let text = serde_json::from_str::<Value>(&row)
            .ok()
            .and_then(|v| v["description"].as_str().map(str::to_string))
            .unwrap_or(row);
        if !text.trim().is_empty() {
            out.push(text);
        }
        if out.len() >= max {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{AudioBed, Beat, ScriptClip};

    fn db_with_footage() -> Db {
        let db = Db::open_in_memory().unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'a', 1, 60)", []).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (2, 'b', 1, 60)", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 0.0, 4.0, 'The deer come right up to the fence every morning.', 0),
                 (1, 4.0, 8.0, 'So who is asking the questions here?', 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES
                 (2, 1.0, '{\"description\":\"Two deer standing at a wooden fence at dawn.\"}'),
                 (2, 3.0, '{\"description\":\"A close shot of a deer looking at the camera.\"}')",
                [],
            )
            .unwrap();
        db
    }

    fn bedded_script() -> Script {
        Script {
            title: "Deer".into(),
            target_duration_s: Some(30.0),
            fps: None,
            width: None,
            height: None,
            beats: vec![Beat {
                id: "the-deer".into(),
                purpose: "Show what she is talking about".into(),
                narration: None,
                on_screen_text: None,
                clips: vec![ScriptClip { video_id: 2, in_s: 0.0, out_s: 4.0, audio: Audio::Mute, why: None }],
                bed: Some(AudioBed { video_id: 1, in_s: 0.0, out_s: 4.0, why: None, inferred: true }),
                notes: None,
            }],
        }
    }

    #[test]
    fn the_state_is_what_is_heard_and_seen_not_timecodes() {
        let db = db_with_footage();
        let s = state(&db, &bedded_script(), Some("A short piece about the deer"), &JevConfig::default());

        assert_eq!(s["brief"], "A short piece about the deer");
        assert_eq!(s["target_seconds"], 30.0);
        let beat = &s["beats"][0];
        assert_eq!(beat["id"], "the-deer");
        assert!(beat["heard"].as_str().unwrap().contains("deer come right up"));
        // The interviewer's own question is off-mic and is not what the viewer is listening to.
        assert!(!beat["heard"].as_str().unwrap().contains("asking the questions"));
        assert_eq!(beat["sound"], "a voice continues under the pictures");
        assert_eq!(beat["seen"].as_array().unwrap().len(), 2);
        assert!(beat["seen"][0].as_str().unwrap().contains("fence"));
        // Nothing a judgement cannot use: no ids, no in/out points.
        assert!(beat.get("clips").is_none());
        assert!(!s.to_string().contains("video_id"));
    }

    #[test]
    fn a_silent_beat_says_so() {
        let db = db_with_footage();
        let mut script = bedded_script();
        script.beats[0].bed = None;
        let s = state(&db, &script, None, &JevConfig::default());
        assert_eq!(s["beats"][0]["sound"], "nothing — the pictures play silent");
        assert_eq!(s["beats"][0]["heard"], "");
        assert!(s.get("brief").is_none(), "no brief, no brief field — and no brief question");
    }

    #[test]
    fn every_beat_is_asked_about_by_name() {
        let db = db_with_footage();
        let s = state(&db, &bedded_script(), Some("brief"), &JevConfig::default());
        let qs = questions(&s, true);

        assert!(qs.contains_key(BRIEF), "a brief was given, so it is asked about");
        assert!(qs.contains_key(ENDING) && qs.contains_key(OPENING) && qs.contains_key(FLOW));
        let q = &qs["beat_0_pictures_match"];
        let json = serde_json::to_value(q).unwrap();
        // The id never reaches the model, so the question itself has to point at its beat.
        assert!(json["instructions"]["question"].as_str().unwrap().contains("beats[0].seen"));
        assert!(json["criteria"]["false"].as_str().unwrap().contains("pause"));

        assert!(!questions(&s, false).contains_key(BRIEF));
    }

    #[test]
    fn the_questions_stay_within_what_one_request_holds() {
        let db = db_with_footage();
        let s = state(&db, &bedded_script(), Some("brief"), &JevConfig::default());
        let qs = questions(&s, true);
        let chars = serde_json::to_string(&qs).unwrap().chars().count();
        // Six fixed questions plus one per beat. A 24-beat cut adds ~24 short nouls to this, and
        // the budget is 32k tokens for the state plus the single longest question.
        assert!(chars < 6000, "fixed questions are {chars} characters");
    }

    fn answers(json: &str) -> Answers {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn a_cut_whose_pictures_fit_scores_above_one_whose_pictures_do_not() {
        let db = db_with_footage();
        let s = state(&db, &bedded_script(), None, &JevConfig::default());

        let good = compose(
            &s,
            &answers(
                r#"{"model":"jev-1.13.0","answers":{
                "beat_0_pictures_match":{"type":"noul","noul":0.95},
                "opening":{"type":"score","score":2.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "flow":{"type":"score","score":2.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "ending":{"type":"score","score":2.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "repeats_itself":{"type":"noul","noul":0.05},
                "voice_whiplash":{"type":"noul","noul":0.05}}}"#,
            ),
        );
        let bad = compose(
            &s,
            &answers(
                r#"{"model":"jev-1.13.0","answers":{
                "beat_0_pictures_match":{"type":"noul","noul":0.08},
                "opening":{"type":"score","score":0.2,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "flow":{"type":"score","score":1.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "ending":{"type":"score","score":0.1,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9},
                "repeats_itself":{"type":"noul","noul":0.8},
                "voice_whiplash":{"type":"noul","noul":0.7}}}"#,
            ),
        );

        assert!(good.total > 90.0, "a cut with everything right scored {:.0}", good.total);
        assert!(bad.total < 30.0, "a cut with everything wrong scored {:.0}", bad.total);
        assert!(good.mismatched.is_empty());
        assert_eq!(bad.mismatched.len(), 1);
        assert_eq!(bad.mismatched[0].beat_id, "the-deer");
        // The note names the beat and quotes what was ignored, so the model can act on it.
        let notes = bad.notes();
        assert!(notes[0].contains("the-deer") && notes[0].contains("deer come right up"));
        assert!(notes.iter().any(|n| n.contains("closing image")), "a cut that stops is told to hold an ending");
        assert!(good.notes().is_empty(), "a good cut has nothing to report");
    }

    #[test]
    fn a_missing_dimension_costs_nothing_rather_than_everything() {
        let db = db_with_footage();
        let s = state(&db, &bedded_script(), None, &JevConfig::default());
        // Only two dimensions came back — the rest are simply not part of the total.
        let j = compose(
            &s,
            &answers(
                r#"{"model":"jev-1.13.0","answers":{
                "beat_0_pictures_match":{"type":"noul","noul":1.0},
                "ending":{"type":"score","score":2.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9}}}"#,
            ),
        );
        assert_eq!(j.parts.len(), 2);
        assert!((j.total - 100.0).abs() < 1e-6, "scored {:.1} on two perfect answers", j.total);
    }

    #[test]
    fn a_beat_nobody_described_is_not_asked_about_and_is_not_a_fault() {
        let db = db_with_footage();
        let mut script = bedded_script();
        // Video 1 has speech but no keyframe descriptions: the picture is unindexed, not bad.
        script.beats[0].clips[0].video_id = 1;
        let s = state(&db, &script, None, &JevConfig::default());
        assert_eq!(s["beats"][0]["pictures_described"], false);
        assert!(s["beats"][0]["seen"].as_array().unwrap().is_empty());

        let qs = questions(&s, false);
        assert!(
            !qs.contains_key("beat_0_pictures_match"),
            "asking got a flat no on footage the vision model simply never saw"
        );

        // Judged without that dimension, the cut still scores on everything else.
        let j = compose(
            &s,
            &serde_json::from_str::<Answers>(
                r#"{"model":"jev-1.13.0","answers":{
                "ending":{"type":"score","score":2.0,"legend":{"0":"a","1":"b","2":"c"},"confidence":0.9}}}"#,
            )
            .unwrap(),
        );
        assert_eq!(j.unjudged_beats, 1, "and the gap is reported rather than averaged away");
        assert!(!j.parts.iter().any(|p| p.name == "pictures match the voice"));
        assert!((j.total - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_held_ending_is_in_the_state_and_changes_what_a_weak_ending_means() {
        let db = db_with_footage();
        let mut script = bedded_script();
        // The pictures run two seconds past the voice: the closing hold the passes add.
        script.beats[0].clips[0].out_s = 6.0;
        let s = state(&db, &script, None, &JevConfig::default());
        assert_eq!(s["ends_with"], "2.0 s of held picture in silence after the last word");

        let weak = r#"{"model":"m","answers":{"ending":{"type":"score","score":0.2,
            "legend":{"0":"a","1":"b","2":"c"},"confidence":0.9}}}"#;
        let held = compose(&s, &serde_json::from_str::<Answers>(weak).unwrap());
        assert!(held.held_closing_picture);
        assert!(
            held.notes()[0].contains("not a conclusion"),
            "asking for a hold that is already there taught the model nothing: {:?}",
            held.notes()
        );

        // Without the hold, the advice is to add one.
        let stops = state(&db, &bedded_script(), None, &JevConfig::default());
        assert!(stops.get("ends_with").is_none());
        let j = compose(&stops, &serde_json::from_str::<Answers>(weak).unwrap());
        assert!(j.notes()[0].contains("hold a closing image"));
    }

    #[test]
    fn a_deduped_stretch_is_described_by_the_last_frame_before_it() {
        let db = db_with_footage();
        // A locked-off interview: frames at 10 s and 40 s, nothing between, because the de-dupe
        // dropped what did not change. A clip at 20-30 s has no frame of its own.
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES
                 (1, 10.0, '{\"description\":\"A woman on a patio, talking to camera.\"}'),
                 (1, 40.0, '{\"description\":\"The same patio, later.\"}')",
                [],
            )
            .unwrap();

        let mut script = bedded_script();
        script.beats[0].clips[0] = ScriptClip {
            video_id: 1,
            in_s: 20.0,
            out_s: 30.0,
            audio: Audio::Mute,
            why: None,
        };
        let s = state(&db, &script, None, &JevConfig::default());

        let seen = s["beats"][0]["seen"].as_array().unwrap();
        assert_eq!(seen.len(), 1, "one frame carries the stretch: {seen:?}");
        assert!(seen[0].as_str().unwrap().contains("talking to camera"));
        assert_eq!(s["beats"][0]["pictures_described"], true, "this beat is judgeable after all");
        // The later frame is not in range and must not be dragged in.
        assert!(!seen[0].as_str().unwrap().contains("later"));
    }
}
