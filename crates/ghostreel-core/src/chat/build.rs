//! Building a cut by choosing, with nothing generating anything.
//!
//! Every other way GhostReel makes a script has a model write JSON and then spends passes
//! repairing what it wrote: timecodes that do not exist, footage it never opened, the same tool
//! called four times, a sentence cut in half. All of that is the cost of generation.
//!
//! A GhostReel script is not writing. Every clip is `(video, in, out)` that must already be in the
//! footage, and the real decisions are *which quote*, *in what order*, and *what to show over it*.
//! Those are selections, and a System One model selects. So code enumerates the candidates out of
//! the index, Jev picks among them, and code assembles the result. Nothing is generated, so an
//! invented timecode is not a bug that got fixed — it is a thing that cannot be expressed.
//!
//! It is also the fastest brain available: five requests, a few seconds, against minutes for a
//! local model drafting the same piece.
//!
//! What it gives up is real. It cannot write narration, it cannot invent a framing device, and it
//! cannot say anything the interviews do not already say. It chooses well among what exists.

use rusqlite::params;
use serde_json::{Value, json};

use crate::Error;
use crate::config::JevConfig;
use crate::db::Db;
use crate::jev::{Jev, Question};
use crate::projects::Project;
use crate::script::{Audio, AudioBed, Beat, Script, ScriptClip};

/// A run of speech that starts and ends on a sentence: something a person could be quoted saying.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Quote {
    pub video_id: i64,
    pub in_s: f64,
    pub out_s: f64,
    pub text: String,
    /// Where the transcript segments inside this quote end, excluding its own end. The only
    /// places the speaker may be cut away from: anywhere else is inside a sentence, and
    /// `end_on_sentences` will rightly put it back — the first build overshot its target by 84%
    /// doing exactly that, because a cutaway took the tail of a line and the pass restored it.
    pub breaks: Vec<f64>,
}

impl Quote {
    pub fn seconds(&self) -> f64 {
        (self.out_s - self.in_s).max(0.0)
    }

    fn overlaps(&self, other: &Quote) -> bool {
        self.video_id == other.video_id && self.in_s < other.out_s && other.in_s < self.out_s
    }
}

/// A described moment of picture.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Shot {
    pub video_id: i64,
    pub t_s: f64,
    pub text: String,
}

/// Everything the index offers, read once and owned outright — so the asking that follows never
/// holds the database open across an await (see `judge::Planned` for the same reason).
pub struct Footage {
    pub quotes: Vec<Quote>,
    pub shots: Vec<Shot>,
}

/// A quote shorter than this is a fragment; longer than this and it is a monologue, not a cut.
const MIN_QUOTE_S: f64 = 4.0;
const MAX_QUOTE_S: f64 = 12.0;
/// Below this a "sentence" is an acknowledgement — "Okay." "Yeah, exactly." — not a statement.
const MIN_QUOTE_WORDS: usize = 10;
/// And a quote may not *end* on one either. Jev picked a closing line that finished "…lost in
/// some places. Okay." and then scored the ending 0.29: the tail is what a viewer is left with.
const MIN_LAST_SENTENCE_WORDS: usize = 4;
/// A Choice takes at most 255 options, and the state has to fit beside them.
const MAX_OPTIONS: usize = 200;

/// Cut a pool down to `max` by taking a turn from each video, rather than the first `max`.
///
/// Truncating in video order quietly hid most of the project: 639 described shots across 96
/// videos, and the first 200 of them covered 25. Re-indexing made that worse, because more frames
/// per video pushed *more* videos out of the window — the same four quotes came back from a cut
/// built on half again as much footage. A round-robin means every video is represented before any
/// video is represented twice, which is the only honest way to spend a fixed number of options.
fn spread<T>(items: &[T], max: usize, video_of: impl Fn(&T) -> i64) -> Vec<&T> {
    if items.len() <= max {
        return items.iter().collect();
    }
    let mut by_video: Vec<(i64, Vec<&T>)> = Vec::new();
    for it in items {
        let v = video_of(it);
        match by_video.iter_mut().find(|(id, _)| *id == v) {
            Some((_, group)) => group.push(it),
            None => by_video.push((v, vec![it])),
        }
    }
    let mut out = Vec::with_capacity(max);
    let mut round = 0usize;
    while out.len() < max {
        let mut took_any = false;
        for (_, group) in &by_video {
            if let Some(it) = group.get(round) {
                out.push(*it);
                took_any = true;
                if out.len() == max {
                    break;
                }
            }
        }
        if !took_any {
            break;
        }
        round += 1;
    }
    out
}
/// The speaker has to be on screen long enough to be somebody before we cut away from them.
const MIN_SPEAKER_S: f64 = 2.0;
/// A cutaway shorter than this flashes past; longer than this and the speaker is forgotten.
const MIN_CUTAWAY_S: f64 = 1.5;
const MAX_CUTAWAY_S: f64 = 5.0;

/// Every quotable run of on-mic speech in the project, and every described shot.
///
/// No model is involved: this is the index, read the way an editor reads a transcript before
/// deciding anything.
pub fn survey(db: &Db, project_id: i64) -> Result<Footage, Error> {
    let video_ids: Vec<i64> = db
        .conn
        .prepare(
            "SELECT DISTINCT vf.video_id FROM video_files vf
             JOIN folders f ON f.id = vf.folder_id
             JOIN project_folders pf ON pf.folder_id = f.id
             WHERE pf.project_id = ?1
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x
                               WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)",
        )?
        .query_map(params![project_id], |r| r.get(0))?
        .collect::<Result<_, _>>()?;

    let mut quotes = Vec::new();
    let mut shots = Vec::new();
    for vid in video_ids {
        let segs: Vec<(f64, f64, String)> = db
            .conn
            .prepare(
                "SELECT start_s, end_s, text FROM transcript_segments
                 WHERE video_id = ?1 AND COALESCE(off_mic, 0) = 0 AND TRIM(text) <> ''
                 ORDER BY start_s",
            )?
            .query_map(params![vid], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, String>(2)?)))?
            .collect::<Result<_, _>>()?;

        // Every run that starts where a segment starts and ends where a sentence ends. Keeping
        // only the longest run from each start point stops near-identical variants of the same
        // quote crowding out different speakers when the options are capped.
        for i in 0..segs.len() {
            let mut best: Option<Quote> = None;
            let mut text = String::new();
            for seg in segs.iter().skip(i) {
                if seg.0 - segs[i].0 >= MAX_QUOTE_S {
                    break;
                }
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(seg.2.trim());
                let dur = seg.1 - segs[i].0;
                let ends_a_sentence = text.trim_end().ends_with(['.', '!', '?']);
                let ends_on_a_statement = seg.2.split_whitespace().count() >= MIN_LAST_SENTENCE_WORDS;
                if (MIN_QUOTE_S..=MAX_QUOTE_S).contains(&dur)
                    && ends_a_sentence
                    && ends_on_a_statement
                    && text.split_whitespace().count() >= MIN_QUOTE_WORDS
                {
                    let breaks = segs
                        .iter()
                        .skip(i)
                        .map(|s| s.1)
                        .take_while(|&e| e < seg.1 - 0.01)
                        .filter(|&e| e > segs[i].0)
                        .collect();
                    best = Some(Quote { video_id: vid, in_s: segs[i].0, out_s: seg.1, text: text.clone(), breaks });
                }
            }
            if let Some(q) = best {
                quotes.push(q);
            }
        }

        let described: Vec<(f64, String)> = db
            .conn
            .prepare(
                "SELECT t_s, description_json FROM frames
                 WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s",
            )?
            .query_map(params![vid], |r| Ok((r.get(0)?, r.get::<_, String>(1)?)))?
            .collect::<Result<_, _>>()?;
        for (t_s, raw) in described {
            let text = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v["description"].as_str().map(str::to_string))
                .unwrap_or(raw);
            if !text.trim().is_empty() {
                shots.push(Shot { video_id: vid, t_s, text });
            }
        }
    }
    Ok(Footage { quotes, shots })
}

fn shorten(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

/// One Choice over the quotes still available, asked on its own.
///
/// Deliberately not batched with the others. Independent questions cannot see one another's
/// answers, and the first attempt proved it: one strong line won opening, middle *and* closing at
/// once, and the cut was two quotes long. Each request now excludes what is already taken, which
/// is the case the docs say a second request is for.
async fn pick(client: &Jev, pool: &[&Quote], instructions: Value, brief: &str) -> Result<Option<(usize, f64)>, Error> {
    if pool.is_empty() {
        return Ok(None);
    }
    let options: Vec<(String, &Quote)> = pool.iter().enumerate().map(|(i, q)| (format!("q{i}"), *q)).collect();
    let state = json!({
        "brief": brief,
        "quotes": options.iter().map(|(k, q)| (k.clone(), json!(q.text))).collect::<serde_json::Map<_, _>>(),
    });
    let criteria: std::collections::BTreeMap<String, Value> =
        options.iter().map(|(k, q)| (k.clone(), json!(shorten(&q.text, 200)))).collect();
    let mut qs = std::collections::BTreeMap::new();
    qs.insert("pick".to_string(), Question::Choice { instructions, criteria });

    let answers = client.ask(&state, &qs).await?;
    let Some(crate::jev::Answer::Choice { choice, confidence, .. }) = answers.answers.get("pick") else {
        return Ok(None);
    };
    let idx = options.iter().position(|(k, _)| k == choice);
    Ok(idx.map(|i| (i, *confidence)))
}

/// Choose the quotes, in the order they will play.
///
/// Opening and closing first, because those are the two positions a viewer notices, and then the
/// middle filled against what is already in the piece rather than against the brief alone — a
/// middle question asked in isolation returns the same strong line every time.
async fn choose_quotes(client: &Jev, footage: &Footage, brief: &str, target_s: f64) -> Result<Vec<Quote>, Error> {
    let all: Vec<&Quote> = spread(&footage.quotes, MAX_OPTIONS, |q| q.video_id);
    let mut taken: Vec<Quote> = Vec::new();

    // Voices already used are off the menu while anything else is left. The middle question asks
    // for a different speaker and Jev mostly obliges, but "mostly" produced a four-line cut drawn
    // from two people out of fifty-six. A rule belongs in code, where it holds every time.
    let free = |taken: &[Quote]| -> Vec<&Quote> {
        let heard: std::collections::HashSet<i64> = taken.iter().map(|q| q.video_id).collect();
        let unheard: Vec<&Quote> = all.iter().copied().filter(|q| !heard.contains(&q.video_id)).collect();
        if !unheard.is_empty() {
            return unheard;
        }
        all.iter().copied().filter(|q| !taken.iter().any(|t| t.overlaps(q))).collect()
    };

    let pool = free(&taken);
    let Some((i, _)) = pick(
        client,
        &pool,
        json!(
            "Which of the lines in `quotes` is the strongest OPENING for the piece described in \
               `brief`? Pick the one that makes a viewer want to keep watching: a clear statement, \
               a vivid detail, or a claim the rest of the piece can answer. Not a fragment, not an \
               interviewer's question, not somebody correcting themselves."
        ),
        brief,
    )
    .await?
    else {
        return Ok(Vec::new());
    };
    let opening = pool[i].clone();
    taken.push(opening.clone());

    let pool = free(&taken);
    let closing = match pick(
        client,
        &pool,
        json!(
            "Which of the lines in `quotes` is the strongest CLOSING line for the piece described \
               in `brief`? It should give the viewer something to take away and sound final, rather \
               than leave a thought open or trail off into an aside."
        ),
        brief,
    )
    .await?
    {
        Some((i, _)) => {
            let c = pool[i].clone();
            taken.push(c.clone());
            Some(c)
        }
        None => None,
    };

    let mut middles = Vec::new();
    // `- MIN_QUOTE_S`: stop before the next quote would certainly overshoot, rather than after.
    while taken.iter().map(Quote::seconds).sum::<f64>() < target_s - MIN_QUOTE_S {
        let pool = free(&taken);
        let already: Vec<String> = taken.iter().map(|q| shorten(&q.text, 160)).collect();
        let Some((i, _)) = pick(
            client,
            &pool,
            json!({
                "already_in_the_piece": already,
                "question": "Which line in `quotes` adds the most to the piece described in `brief` \
                             that is NOT already said in `already_in_the_piece`? Prefer a different \
                             speaker, and a point none of those lines makes.",
            }),
            brief,
        )
        .await?
        else {
            break;
        };
        let q = pool[i].clone();
        // A quote that would take the cut well past its target is worse than a short cut.
        if taken.iter().map(Quote::seconds).sum::<f64>() + q.seconds() > target_s + MAX_QUOTE_S / 2.0 {
            break;
        }
        taken.push(q.clone());
        middles.push(q);
    }

    let mut order = vec![opening];
    order.extend(middles);
    order.extend(closing);
    Ok(order)
}

/// One request: what to show over each chosen line.
///
/// Only footage nobody in the cut speaks in is on offer. A talking head under somebody else's
/// voice is the exact failure this is meant to prevent, so it is not on the menu at all.
/// Shots offered for one line. Forty is what a Choice can weigh without the distribution going
/// flat, and forty relevant ones beat two hundred arbitrary ones.
const SHORTLIST: usize = 40;

/// Words that appear in every sentence and so distinguish nothing.
const STOPWORDS: &[&str] = &[
    "the", "and", "that", "this", "with", "you", "your", "for", "are", "was", "were", "have", "has", "had", "but",
    "not", "they", "them", "their", "there", "here", "what", "when", "where", "who", "how", "just", "like", "really",
    "kind", "sort", "very", "much", "more", "most", "some", "any", "all", "our", "out", "about", "from", "into",
    "than", "then", "been", "would", "could", "should", "will", "can", "get", "got", "know", "think", "going", "one",
    "its", "it's", "i'm", "we're", "that's", "yeah", "okay", "thing", "things", "stuff", "because", "well",
];

/// The content words of a line or a description, lowercased, long enough to mean something.
fn content_words(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| w.len() >= 4 && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// How well a shot's description answers a line, as shared content words.
///
/// Deliberately not embeddings: the builder's whole shape is that code enumerates and Jev judges,
/// and pulling the search subsystem in here would widen what a change to this file can break. The
/// descriptions are keyword-rich — "deer", "fence", "patio", "hillside" — so overlap is enough to
/// shortlist, and Jev still does the choosing.
fn relevance(line: &std::collections::HashSet<String>, shot: &str) -> usize {
    let words = content_words(shot);
    line.iter().filter(|w| words.contains(*w)).count()
}

/// The shots worth offering for one line: the most relevant, and a spread when nothing matches.
///
/// A Choice takes at most 255 options and the state shares that budget, so 639 described shots
/// have to become a few dozen. Spending them on a round-robin sample offers the model whatever
/// happened to be sampled; spending them on relevance offers it the shots that might actually
/// illustrate the line, and no shot in the project is unreachable for want of a turn.
fn shortlist_for<'a>(line: &Quote, eligible: &'a [Shot]) -> Vec<&'a Shot> {
    let words = content_words(&line.text);
    let mut scored: Vec<(usize, &Shot)> = eligible.iter().map(|s| (relevance(&words, &s.text), s)).collect();
    if scored.iter().all(|(n, _)| *n == 0) {
        // Nothing in the line names anything visible — "we just love it". A spread is the honest
        // answer there: no shot is more relevant than another.
        return spread(eligible, SHORTLIST, |s| s.video_id);
    }
    // Most relevant first; ties by video so one tape cannot monopolise the list.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.video_id.cmp(&b.1.video_id)));
    scored.into_iter().filter(|(n, _)| *n > 0).take(SHORTLIST).map(|(_, s)| s).collect()
}

async fn choose_shots(
    client: &Jev,
    footage: &Footage,
    lines: &[Quote],
    brief: &str,
) -> Result<Vec<Option<Shot>>, Error> {
    let speakers: std::collections::HashSet<i64> = lines.iter().map(|q| q.video_id).collect();
    let eligible: Vec<Shot> = footage.shots.iter().filter(|s| !speakers.contains(&s.video_id)).cloned().collect();
    if eligible.is_empty() {
        return Ok(vec![None; lines.len()]);
    }

    // One shortlist per line rather than one pool for all of them. Every shot keeps a stable key
    // so the state names each one once however many lines it is offered to.
    let mut key_of: std::collections::HashMap<(i64, u64), String> = std::collections::HashMap::new();
    let mut shots_json = serde_json::Map::new();
    let mut per_line: Vec<Vec<(String, &Shot)>> = Vec::with_capacity(lines.len());
    for line in lines {
        let mut keyed = Vec::new();
        for shot in shortlist_for(line, &eligible) {
            let id = (shot.video_id, (shot.t_s * 1000.0) as u64);
            let key = key_of.entry(id).or_insert_with(|| format!("s{}", shots_json.len())).clone();
            shots_json.entry(key.clone()).or_insert_with(|| json!(shorten(&shot.text, 260)));
            keyed.push((key, shot));
        }
        per_line.push(keyed);
    }
    if per_line.iter().all(Vec::is_empty) {
        return Ok(vec![None; lines.len()]);
    }

    let state = json!({
        "brief": brief,
        "shots": shots_json,
        "lines": lines.iter().enumerate().map(|(i, q)| (format!("l{i}"), json!(q.text))).collect::<serde_json::Map<_, _>>(),
    });

    // Every line in one request: they are independent, and Jev reads the shot list once.
    let mut qs = std::collections::BTreeMap::new();
    for (i, q) in lines.iter().enumerate() {
        if per_line[i].is_empty() {
            continue;
        }
        qs.insert(
            format!("shot_for_l{i}"),
            Question::Choice {
                instructions: json!({
                    "line": shorten(&q.text, 300),
                    "question": format!(
                        "The voice in `lines.l{i}` plays over the picture. Which of these shots best \
                         shows what that voice is talking about? Choose one a viewer would read as an \
                         illustration of the line."
                    ),
                }),
                criteria: per_line[i].iter().map(|(k, s)| (k.clone(), json!(shorten(&s.text, 200)))).collect(),
            },
        );
    }

    let answers = client.ask(&state, &qs).await?;
    Ok(lines
        .iter()
        .enumerate()
        // Each line is resolved against its own shortlist: the keys are shared across the state,
        // but a line can only be answered with a shot it was actually offered.
        .map(|(i, _)| match answers.answers.get(&format!("shot_for_l{i}")) {
            Some(crate::jev::Answer::Choice { choice, .. }) => {
                per_line[i].iter().find(|(k, _)| k == choice).map(|(_, s)| (*s).clone())
            }
            _ => None,
        })
        .collect())
}

/// Turn the choices into a script. Pure, so the shape of a cut can be argued with offline.
///
/// Each beat is the speaker on camera, then a cutaway, with the voice running under both as a
/// bed. That is a J-cut and it is the whole point: the picture may change, the sentence may not.
pub fn lay_out(project: &Project, lines: &[Quote], shots: &[Option<Shot>], target_s: f64) -> Script {
    let mut beats = Vec::new();
    for (i, q) in lines.iter().enumerate() {
        // Cut away from the speaker on a sentence end, never inside one, and only for a stretch
        // long enough to read and short enough to still have established who is talking.
        let cut_at = q.breaks.iter().copied().rfind(|&b| {
            let picture = q.out_s - b;
            b - q.in_s >= MIN_SPEAKER_S && (MIN_CUTAWAY_S..=MAX_CUTAWAY_S).contains(&picture)
        });

        let mut clips = vec![ScriptClip {
            video_id: q.video_id,
            in_s: q.in_s,
            out_s: cut_at.unwrap_or(q.out_s),
            audio: Audio::Source,
            why: Some("the person saying it".into()),
        }];
        // No usable break means no cutaway. A beat of the speaker alone is a worse cut than one
        // with b-roll; a beat whose b-roll chops a sentence is a broken one.
        if let (Some(cut_at), Some(Some(shot))) = (cut_at, shots.get(i)) {
            let picture_s = q.out_s - cut_at;
            let start = (shot.t_s - picture_s / 2.0).max(0.0);
            clips.push(ScriptClip {
                video_id: shot.video_id,
                in_s: start,
                out_s: start + picture_s,
                audio: Audio::Mute,
                why: Some("shows what is being said".into()),
            });
        }
        beats.push(Beat {
            id: format!("beat-{}", i + 1),
            purpose: shorten(&q.text, 120),
            narration: None,
            on_screen_text: None,
            clips,
            bed: Some(AudioBed {
                video_id: q.video_id,
                in_s: q.in_s,
                out_s: q.out_s,
                why: Some("the voice runs on under the pictures".into()),
                inferred: false,
            }),
            notes: None,
        });
    }
    Script {
        title: project.name.clone(),
        target_duration_s: Some(target_s),
        fps: Some(crate::script::Fps::new(project.fps_num, project.fps_den)),
        width: Some(project.width),
        height: Some(project.height),
        beats,
    }
}

/// Write a built cut into a chat session, as though a turn had produced it.
///
/// The builder is fast and picks pictures well — 0.84 on picture-to-voice against agy's 0.77 —
/// and it structures poorly, because it chooses four quotes independently and nothing ever asks
/// whether they make a story. agy is the other way round. Handing the built cut to agy as the
/// conversation so far lets each do what it is good at, and saves agy the research: it starts
/// from real timecodes and a critique instead of fifteen tool rounds of looking.
///
/// `notes` is what the judge said, when it was asked. The chat already replays the assistant
/// message and already treats a `[...]` note on it as instructions for the next turn, so the
/// critique arrives the same way a repair note does.
pub fn save_as_session(
    db: &Db,
    project_id: i64,
    brief: &str,
    script_id: i64,
    script: &Script,
    notes: &[String],
) -> Result<i64, Error> {
    let title: String = brief.chars().take(60).collect();
    let session_id = super::create_session(db, project_id, &title)?;
    let t = super::now();

    db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'user', ?2, NULL, ?3)",
        params![session_id, brief, t],
    )?;

    // The script itself, in the form the chat replays: the next turn parses it back out as the
    // version to improve rather than starting from nothing.
    let mut content = format!(
        "Drafted '{}': {} beats, {} clips, {:.1} s\n\n```json\n{}\n```",
        script.title,
        script.beats.len(),
        script.clip_count(),
        script.total_duration_s(),
        serde_json::to_string_pretty(script).unwrap_or_default()
    );
    if !notes.is_empty() {
        content.push_str(&format!("\n\n[editorial read of this cut, fix these: {}]", notes.join("; ")));
    }
    db.conn.execute(
        "INSERT INTO chat_messages(session_id, role, content, tool_calls_json, created_at)
         VALUES (?1, 'assistant', ?2, ?3, ?4)",
        params![session_id, content, serde_json::json!({ "script_id": script_id }).to_string(), t],
    )?;
    db.conn.execute("UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2", params![t, session_id])?;
    Ok(session_id)
}

/// Build a cut out of the footage by choosing, start to finish.
///
/// The result still goes through the ordinary repair passes — it is a draft like any other, and
/// the closing hold and the sentence guarantee are not this module's business.
pub async fn build(
    footage: &Footage,
    project: &Project,
    brief: &str,
    target_s: f64,
    cfg: &JevConfig,
) -> Result<Script, Error> {
    let client = Jev::from_config(cfg)
        .ok_or_else(|| Error::Jev("building a cut this way needs Jev: set jev.enabled and an API key".into()))?;
    if footage.quotes.is_empty() {
        return Err(Error::Jev("no quotable speech in this project — index the transcripts first".into()));
    }
    let lines = choose_quotes(&client, footage, brief, target_s).await?;
    if lines.is_empty() {
        return Err(Error::Jev("nothing was chosen".into()));
    }
    let shots = choose_shots(&client, footage, &lines, brief).await?;
    Ok(lay_out(project, &lines, &shots, target_s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    fn project_with_speech() -> (Db, Project) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Test")).unwrap();
        let tmp = std::env::temp_dir();
        let folder = db.add_folder(project.id, &tmp, true).unwrap();
        for vid in [1i64, 2] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (?1, ?2, 1, 60)",
                    params![vid, format!("h{vid}")],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                     VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![vid, folder.id, format!("{}/{vid}.mp4", tmp.display())],
                )
                .unwrap();
        }
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (1, 0.0, 3.0, 'The deer come right up to the fence here every single morning', 0),
                 (1, 3.0, 6.0, 'and nobody ever seems to mind them at all.', 0),
                 (1, 6.0, 9.0, 'So who is asking you all of these questions anyway today?', 1),
                 (2, 0.0, 5.0, 'Yeah.', 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES
                 (2, 2.0, '{\"description\":\"Two deer at a wooden fence at dawn.\"}')",
                [],
            )
            .unwrap();
        (db, project)
    }

    #[test]
    fn the_index_offers_whole_sentences_and_nothing_else() {
        let (db, project) = project_with_speech();
        let f = survey(&db, project.id).unwrap();

        assert_eq!(f.quotes.len(), 1, "one quotable run: {:?}", f.quotes);
        let q = &f.quotes[0];
        assert_eq!((q.video_id, q.in_s, q.out_s), (1, 0.0, 6.0));
        assert!(q.text.ends_with("mind them at all."), "a quote ends on a sentence: {:?}", q.text);
        // The interviewer is off-mic; "Yeah." is a sentence but not a statement.
        assert!(!q.text.contains("asking you all"));
        assert!(!f.quotes.iter().any(|q| q.video_id == 2));

        assert_eq!(f.shots.len(), 1);
        assert!(f.shots[0].text.contains("deer"));
    }

    #[test]
    fn a_cut_laid_out_this_way_is_a_j_cut_with_nothing_invented() {
        let (db, project) = project_with_speech();
        let f = survey(&db, project.id).unwrap();
        let shot = f.shots[0].clone();
        let s = lay_out(&project, &f.quotes, &[Some(shot)], 30.0);

        assert_eq!(s.beats.len(), 1);
        let beat = &s.beats[0];
        // The voice covers the whole beat; the picture changes under it.
        let bed = beat.bed.as_ref().unwrap();
        assert_eq!((bed.in_s, bed.out_s), (0.0, 6.0));
        assert!(!bed.inferred, "the builder chose this bed, it was not laid for it afterwards");
        assert_eq!(beat.clips.len(), 2);
        assert_eq!(beat.clips[0].audio, Audio::Source);
        assert_eq!(beat.clips[1].audio, Audio::Mute, "a cutaway under a voice plays silent");
        assert_eq!(beat.clips[1].video_id, 2);
        // Every timecode came out of the index: nothing here can have been invented.
        assert!(beat.clips.iter().all(|c| c.in_s >= 0.0 && c.out_s > c.in_s));
        assert!((beat.clips[0].out_s - 3.0).abs() < 1e-9, "the cutaway takes the back half");
    }

    #[test]
    fn a_cut_with_no_shot_to_offer_is_still_a_cut() {
        let (db, project) = project_with_speech();
        let f = survey(&db, project.id).unwrap();
        let s = lay_out(&project, &f.quotes, &[None], 30.0);
        assert_eq!(s.beats[0].clips.len(), 1, "no cutaway, just the speaker");
        assert_eq!(s.beats[0].clips[0].audio, Audio::Source);
    }

    #[tokio::test]
    async fn without_a_key_it_says_so_rather_than_trying() {
        let (db, project) = project_with_speech();
        let f = survey(&db, project.id).unwrap();
        let err = build(&f, &project, "a brief", 30.0, &JevConfig::default()).await.unwrap_err();
        assert!(err.to_string().contains("jev.enabled"), "{err}");
    }

    #[test]
    fn the_cutaway_falls_on_a_sentence_end_or_does_not_happen() {
        let (db, project) = project_with_speech();
        let f = survey(&db, project.id).unwrap();
        let shot = f.shots[0].clone();

        // The only break inside this quote is 3.0 s, and that is where the picture changes.
        assert_eq!(f.quotes[0].breaks, vec![3.0]);
        let s = lay_out(&project, &f.quotes, &[Some(shot.clone())], 30.0);
        assert_eq!(s.beats[0].clips[0].out_s, 3.0);

        // A quote with nowhere legal to cut keeps the speaker on screen throughout. Cutting away
        // anywhere else takes the tail of a sentence, `end_on_sentences` restores it, and the cut
        // comes out far longer than asked for — which is what the first real build did, by 84%.
        let unbroken = vec![Quote { breaks: vec![], ..f.quotes[0].clone() }];
        let s = lay_out(&project, &unbroken, &[Some(shot)], 30.0);
        assert_eq!(s.beats[0].clips.len(), 1, "nowhere to cut, so no cutaway");
        assert_eq!(s.beats[0].clips[0].out_s, unbroken[0].out_s);

        // Nor a break that would leave the speaker unestablished, or a flash of picture.
        let too_early = vec![Quote { breaks: vec![1.0], ..f.quotes[0].clone() }];
        assert_eq!(lay_out(&project, &too_early, &[None], 30.0).beats[0].clips.len(), 1);
        let too_late = vec![Quote { breaks: vec![5.8], ..f.quotes[0].clone() }];
        assert_eq!(lay_out(&project, &too_late, &[None], 30.0).beats[0].clips.len(), 1);
    }

    #[test]
    fn a_quote_may_not_end_on_an_acknowledgement() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("T")).unwrap();
        let tmp = std::env::temp_dir();
        let folder = db.add_folder(project.id, &tmp, true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'a', 1, 60)", []).unwrap();
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
                 (1, 0.0, 3.0, 'Neighborness is something that is kind of lost in some places now', 0),
                 (1, 3.0, 5.0, 'and you really do notice it when you move away.', 0),
                 (1, 5.0, 9.0, 'Okay.', 0)",
                [],
            )
            .unwrap();

        let f = survey(&db, project.id).unwrap();
        assert!(!f.quotes.is_empty(), "the real sentence is still quotable");
        assert!(
            f.quotes.iter().all(|q| !q.text.trim_end().ends_with("Okay.")),
            "a closing line ending on an acknowledgement is what a viewer is left with: {:?}",
            f.quotes.iter().map(|q| &q.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn capping_the_options_takes_a_turn_from_each_video_not_the_first_few() {
        // Three videos, ten shots each, room for six options.
        let shots: Vec<Shot> = (1..=3)
            .flat_map(|v| (0..10).map(move |i| Shot { video_id: v, t_s: i as f64, text: String::new() }))
            .collect();
        let picked = spread(&shots, 6, |s| s.video_id);

        assert_eq!(picked.len(), 6);
        let videos: std::collections::BTreeSet<i64> = picked.iter().map(|s| s.video_id).collect();
        assert_eq!(videos.len(), 3, "every video is represented: {videos:?}");
        // Truncation would have given six shots of video 1 and nothing else, which is what hid
        // 71 of 96 videos from the real build.
        assert_eq!(picked.iter().filter(|s| s.video_id == 1).count(), 2);

        // Round order: one from each, then the next from each.
        let order: Vec<(i64, f64)> = picked.iter().map(|s| (s.video_id, s.t_s)).collect();
        assert_eq!(order, vec![(1, 0.0), (2, 0.0), (3, 0.0), (1, 1.0), (2, 1.0), (3, 1.0)]);
    }

    #[test]
    fn a_pool_that_already_fits_is_left_exactly_as_it_was() {
        let shots: Vec<Shot> = (0..5).map(|i| Shot { video_id: 1, t_s: i as f64, text: String::new() }).collect();
        let picked = spread(&shots, 200, |s| s.video_id);
        assert_eq!(picked.len(), 5);
        assert_eq!(picked.iter().map(|s| s.t_s).collect::<Vec<_>>(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    /// 639 described shots have to become a few dozen, and spending that budget on relevance
    /// rather than on a round-robin sample is the difference between offering the model the shot
    /// that illustrates the line and offering it whatever happened to be sampled.
    #[test]
    fn a_line_is_offered_the_shots_that_match_its_words() {
        let shot = |vid: i64, t: f64, text: &str| Shot { video_id: vid, t_s: t, text: text.into() };
        let eligible = vec![
            shot(1, 0.0, "A hotel lobby with a reception desk and marble floor."),
            shot(2, 0.0, "Two deer standing at a wooden fence at dawn."),
            shot(3, 0.0, "A conference room with a projector screen."),
            shot(4, 0.0, "A quiet residential street lined with oak trees."),
        ];
        let line = Quote {
            video_id: 9,
            in_s: 0.0,
            out_s: 8.0,
            text: "We love the deer running around the fence every morning.".into(),
            breaks: vec![],
        };

        let picked = shortlist_for(&line, &eligible);
        assert_eq!(
            picked[0].video_id,
            2,
            "the deer shot comes first: {:?}",
            picked.iter().map(|s| s.video_id).collect::<Vec<_>>()
        );
        // The lobby and the conference room share no content word with the line at all.
        assert!(!picked.iter().any(|s| s.video_id == 1));
        assert!(!picked.iter().any(|s| s.video_id == 3));
    }

    #[test]
    fn a_line_naming_nothing_visible_still_gets_a_spread() {
        let shot = |vid: i64, text: &str| Shot { video_id: vid, t_s: 0.0, text: text.into() };
        let eligible = vec![shot(1, "A hotel lobby."), shot(2, "A wooden fence."), shot(3, "A conference room.")];
        // Nothing here names anything you could photograph.
        let line = Quote { video_id: 9, in_s: 0.0, out_s: 8.0, text: "And we just love it.".into(), breaks: vec![] };

        let picked = shortlist_for(&line, &eligible);
        assert_eq!(picked.len(), 3, "no shot is more relevant than another, so offer them all");
    }

    #[test]
    fn stopwords_and_short_words_do_not_count_as_a_match() {
        // "that", "with", "have" appear in every sentence and distinguish nothing.
        let words = content_words("And that is the thing that we have here with you, because well.");
        assert!(words.is_empty(), "nothing here names anything: {words:?}");
        assert_eq!(relevance(&content_words("deer at the fence"), "A wooden fence at dawn."), 1);
    }
}
