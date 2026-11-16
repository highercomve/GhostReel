//! Section cards: a plain full-screen background with a title, played before a beat to announce
//! the section it opens.
//!
//! Cards are data in the script ([`crate::script::SectionCard`] on a beat, the cut's look in
//! [`crate::script::CardStyle`]), so they are edited, moved and deleted like anything else in a
//! cut, and every render and export reads the same fields. [`generate`] writes them: a card before
//! each beat that moves the cut to another chapter of the footage, titled after that chapter (or
//! after the beat's purpose where the footage has no chapter titles). A generated card is marked
//! as such until someone edits it, so generating again refreshes the untouched cards and keeps
//! the edited ones.

use rusqlite::OptionalExtension;

use crate::db::Db;
use crate::script::{Script, SectionCard};

/// Longest generated title, in characters; a longer chapter title is cut at a word.
const MAX_TITLE_CHARS: usize = 60;

/// The title of the chapter `t` falls in, when the chapters stage has titled it.
fn chapter_at(db: &Db, video_id: i64, t: f64) -> Option<(i64, String)> {
    db.conn
        .query_row(
            "SELECT idx, title FROM chapters WHERE video_id = ?1 AND start_s <= ?2 + 0.5 AND end_s > ?2
             ORDER BY start_s DESC LIMIT 1",
            rusqlite::params![video_id, t],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
        )
        .optional()
        .ok()
        .flatten()
        .filter(|(_, title)| !title.trim().is_empty())
}

/// `text` cut to [`MAX_TITLE_CHARS`] at a word, without trailing punctuation.
fn short(text: &str) -> String {
    let text = text.trim().trim_end_matches(['.', ':', ';', ',']);
    if text.chars().count() <= MAX_TITLE_CHARS {
        return text.to_string();
    }
    let mut out = String::new();
    for w in text.split_whitespace() {
        if out.chars().count() + w.chars().count() + 1 > MAX_TITLE_CHARS {
            break;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    format!("{}…", out.trim_end_matches([',', ';', ':']))
}

/// Give the cut its section cards: one before every beat (after the first, which opens the cut
/// cold) whose footage comes from another chapter than the beat before it. A beat whose card
/// someone edited keeps it; generated cards are written again, and dropped where the beat no
/// longer opens a section. Returns how many beats have a card afterwards.
pub fn generate(db: &Db, script: &mut Script) -> usize {
    let mut prev: Option<(i64, Option<i64>)> = None;
    for (i, beat) in script.beats.iter_mut().enumerate() {
        let here = beat.clips.first().map(|c| {
            let chapter = chapter_at(db, c.video_id, c.in_s);
            (c.video_id, chapter)
        });
        let key = here.as_ref().map(|(v, ch)| (*v, ch.as_ref().map(|(idx, _)| *idx)));
        let new_section = i > 0
            && match (&prev, &key) {
                (Some(p), Some(k)) => p != k || k.1.is_none(),
                _ => false,
            };
        if key.is_some() {
            prev = key;
        }
        if beat.card.as_ref().is_some_and(|c| !c.generated) {
            continue;
        }
        if !new_section {
            beat.card = None;
            continue;
        }
        let title = here.and_then(|(_, ch)| ch.map(|(_, t)| t)).unwrap_or_else(|| beat.purpose.clone());
        let title = short(&title);
        beat.card = (!title.is_empty()).then(|| SectionCard { text: title, generated: true, ..Default::default() });
    }
    script.beats.iter().filter(|b| b.card.is_some()).count()
}

/// Take the section cards out of the cut. `keep_edited`: leave the cards someone wrote or edited.
/// Returns how many were removed.
pub fn clear(script: &mut Script, keep_edited: bool) -> usize {
    let mut n = 0;
    for beat in &mut script.beats {
        if beat.card.as_ref().is_some_and(|c| c.generated || !keep_edited) {
            beat.card = None;
            n += 1;
        }
    }
    n
}

/// Whether the cut has any section card.
pub fn has_cards(script: &Script) -> bool {
    script.beats.iter().any(|b| b.card.as_ref().is_some_and(|c| !c.text.trim().is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::{Audio, Beat, ScriptClip};

    fn beat(id: &str, purpose: &str, video_id: i64, in_s: f64) -> Beat {
        Beat {
            id: id.into(),
            purpose: purpose.into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            clips: vec![ScriptClip {
                exact: false,
                gain_db: None,
                video_id,
                in_s,
                out_s: in_s + 5.0,
                audio: Audio::Source,
                why: None,
                tracks: None,
                framing: None,
                sentences: None,
            }],
            bed: None,
            notes: None,
            card: None,
        }
    }

    fn script(beats: Vec<Beat>) -> Script {
        let mut s: Script = serde_json::from_value(serde_json::json!({"title": "T", "beats": []})).unwrap();
        s.beats = beats;
        s
    }

    fn db() -> Db {
        let db = Db::open_in_memory().unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO videos(id, content_hash, size, duration_s) VALUES (1, 'h', 1, 600.0);
                 INSERT INTO transcript_sentences(id, video_id, start_s, end_s, text) VALUES
                   (1, 1, 0.0, 5.0, 'a'), (2, 1, 100.0, 105.0, 'b'), (3, 1, 300.0, 305.0, 'c');
                 INSERT INTO chapters(video_id, idx, start_s, end_s, first_sentence_id, last_sentence_id, title, titled)
                 VALUES (1, 0, 0.0, 200.0, 1, 2, 'Cómo empezamos', 1),
                        (1, 1, 200.0, 600.0, 3, 3, 'Lo que cuesta de verdad.', 1);",
            )
            .unwrap();
        db
    }

    #[test]
    fn a_card_opens_each_new_chapter_titled_after_it_and_edits_survive_regenerating() {
        let db = db();
        let mut s = script(vec![
            beat("b1", "hook", 1, 300.0),
            beat("b2", "premise", 1, 10.0),
            beat("b3", "same chapter", 1, 100.0),
            beat("b4", "cost", 1, 310.0),
        ]);
        assert_eq!(generate(&db, &mut s), 2);
        let texts: Vec<Option<&str>> = s.beats.iter().map(|b| b.card.as_ref().map(|c| c.text.as_str())).collect();
        // The cold open has none; b3 continues b2's chapter.
        assert_eq!(texts, [None, Some("Cómo empezamos"), None, Some("Lo que cuesta de verdad")]);
        assert_eq!(s.cards_duration_s(), 4.0);
        assert_eq!(s.length_s(), 24.0);
        let times = s.beat_times();
        assert_eq!((times[1].card_start_s, times[1].clips_start_s), (5.0, 7.0));

        // An edited card stays when the cut is generated again; untouched ones are refreshed.
        let card = s.beats[1].card.as_mut().unwrap();
        card.text = "Así empezó".into();
        card.generated = false;
        s.beats.swap(2, 3);
        generate(&db, &mut s);
        let texts: Vec<Option<&str>> = s.beats.iter().map(|b| b.card.as_ref().map(|c| c.text.as_str())).collect();
        assert_eq!(texts, [None, Some("Así empezó"), Some("Lo que cuesta de verdad"), Some("Cómo empezamos")]);

        assert_eq!(clear(&mut s, true), 2);
        assert!(s.beats[1].card.is_some());
        assert_eq!(clear(&mut s, false), 1);
        assert!(!has_cards(&s));
    }

    #[test]
    fn without_chapters_the_beat_purpose_is_the_title() {
        let db = Db::open_in_memory().unwrap();
        let mut s = script(vec![beat("b1", "hook", 9, 0.0), beat("b2", "Ten dollars an episode", 9, 50.0)]);
        generate(&db, &mut s);
        assert_eq!(s.beats[1].card.as_ref().map(|c| c.text.as_str()), Some("Ten dollars an episode"));
    }
}
