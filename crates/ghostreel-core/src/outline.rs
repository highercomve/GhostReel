//! The project in one call (`get_outline`): what footage there is, who is in it, what is said
//! chapter by chapter, and the words the project cares about.
//!
//! It exists so a model can start an edit by reading a few thousand tokens instead of every
//! transcript: it learns which video and which stretch matter from the chapters' titles,
//! summaries and standout lines, and then reads only those with `get_transcript`.

use serde::Serialize;

use crate::Error;
use crate::chapters::{self, Chapter};
use crate::db::Db;
use crate::index;
use crate::projects::Project;
use crate::sentences::{clock, format_id};

/// A standout line: a sentence id and its words.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Standout {
    pub sentence: String,
    pub text: String,
    /// "screen" or "no face" when the line is said over that picture rather than a face.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub picture: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutlineChapter {
    pub number: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub first_sentence: String,
    pub last_sentence: String,
    pub title: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub summary: String,
    pub standout: Vec<Standout>,
    /// The title is the chapter's first sentence: no model had been asked yet.
    pub stand_in_title: bool,
    /// Percent of the chapter with a face on screen (the rest is screen share or nobody), where the
    /// faces stage has run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faces_pct: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutlineVideo {
    pub id: i64,
    pub file: String,
    pub duration_s: f64,
    pub category: String,
    pub has_speech: bool,
    pub sentences: i64,
    /// Empty for a video with no speech, and for one the chapters stage has not reached.
    pub chapters: Vec<OutlineChapter>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutlinePerson {
    pub id: i64,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub speaking_time_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outline {
    pub project: String,
    pub videos: Vec<OutlineVideo>,
    pub people: Vec<OutlinePerson>,
    pub glossary: Vec<String>,
}

/// How much of each chapter is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail {
    /// Title, summary and up to three standout lines.
    Full,
    /// Title, summary and the best standout line, shortened.
    Medium,
    /// Range and title only.
    Titles,
}

/// Longest standout line shown, in characters.
const STANDOUT_CHARS: usize = 110;
const SUMMARY_CHARS: usize = 200;

/// Build the outline of a project; `only_video` limits it to one video.
pub fn build(db: &Db, project: &Project, only_video: Option<i64>) -> Result<Outline, Error> {
    let rows = index::videos(db, Some(project.id))?;
    let mut videos = Vec::new();
    for v in rows.into_iter().filter(|v| only_video.is_none_or(|id| id == v.id)) {
        let stored = chapters::load(db, v.id)?;
        let spans = crate::scenes::for_range(db, v.id, 0.0, v.duration_s.unwrap_or(0.0)).ok().flatten();
        videos.push(OutlineVideo {
            id: v.id,
            file: v.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            duration_s: v.duration_s.unwrap_or(0.0),
            category: v.category.clone(),
            has_speech: v.segments > 0,
            sentences: v.segments,
            chapters: stored.iter().map(|c| outline_chapter(db, c, spans.as_deref())).collect(),
        });
    }
    let people = db
        .person_summaries(project.id)?
        .into_iter()
        .map(|p| OutlinePerson {
            id: p.id,
            name: [p.name, p.handle]
                .into_iter()
                .flatten()
                .map(|s| s.trim().to_string())
                .find(|s| !s.is_empty())
                .unwrap_or_else(|| format!("person {}", p.id)),
            role: p.role.filter(|r| !r.trim().is_empty()),
            speaking_time_s: p.speaking_time_s,
        })
        .collect();
    let glossary = project.glossary.terms.iter().map(|t| t.term.clone()).collect();
    Ok(Outline { project: project.name.clone(), videos, people, glossary })
}

fn outline_chapter(db: &Db, c: &Chapter, spans: Option<&[crate::scenes::SceneSpan]>) -> OutlineChapter {
    let standout = c
        .standout
        .iter()
        .filter_map(|&id| {
            let (text, a, b): (String, f64, f64) = db
                .conn
                .query_row("SELECT text, start_s, end_s FROM transcript_sentences WHERE id = ?1", [id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })
                .ok()?;
            let picture = spans.and_then(|sp| crate::scenes::picture_label(sp, a, b));
            Some(Standout { sentence: format_id(id), text: text.trim().to_string(), picture })
        })
        .collect();
    OutlineChapter {
        number: c.idx + 1,
        start_s: c.start_s,
        end_s: c.end_s,
        first_sentence: c.first_sentence(),
        last_sentence: c.last_sentence(),
        title: c.title.clone(),
        summary: c.summary.clone(),
        standout,
        stand_in_title: !c.titled,
        faces_pct: spans.map(|sp| (crate::scenes::face_share(sp, c.start_s, c.end_s) * 100.0).round() as u32),
    }
}

fn clip(text: &str, n: usize) -> String {
    if text.chars().count() <= n {
        return text.to_string();
    }
    let cut: String = text.chars().take(n).collect();
    format!("{}…", cut.trim_end())
}

impl Outline {
    /// The outline as compact text, as rich as `detail` allows.
    pub fn render(&self, detail: Detail) -> String {
        let footage: f64 = self.videos.iter().map(|v| v.duration_s).sum();
        let mut out = format!(
            "PROJECT \"{}\" · {} video{} · {} of footage\n",
            self.project,
            self.videos.len(),
            if self.videos.len() == 1 { "" } else { "s" },
            clock(footage)
        );
        out.push_str("VIDEOS (id file length category)\n");
        for v in &self.videos {
            out.push_str(&format!(
                "  v{} {} · {} · {}{}\n",
                v.id,
                v.file,
                clock(v.duration_s),
                v.category,
                if v.has_speech { " · speech" } else { "" }
            ));
        }
        if !self.people.is_empty() {
            out.push_str("PEOPLE (id name, role, speaking time)\n");
            for p in &self.people {
                let role = p.role.as_deref().map(|r| format!(", {r}")).unwrap_or_default();
                out.push_str(&format!("  #{} {}{role}, speaks {}\n", p.id, p.name, clock(p.speaking_time_s)));
            }
        }
        if !self.glossary.is_empty() {
            out.push_str(&format!("GLOSSARY {}\n", self.glossary.join(", ")));
        }
        for v in self.videos.iter().filter(|v| v.has_speech) {
            if v.chapters.is_empty() {
                out.push_str(&format!(
                    "\nv{} {}: {} sentences, no chapters yet (the chapters stage has not run: index with stages \
                     [\"chapters\"], or `ghostreel chapters`); read it with get_transcript\n",
                    v.id, v.file, v.sentences
                ));
                continue;
            }
            out.push_str(&format!(
                "\nv{} {} · {} chapters (range, sentences, share of time a face is on screen, title — summary)\n",
                v.id,
                v.file,
                v.chapters.len()
            ));
            for c in &v.chapters {
                out.push_str(&format!(
                    "  {}. {}-{} {}-{}{} {}",
                    c.number,
                    clock(c.start_s),
                    clock(c.end_s),
                    c.first_sentence,
                    c.last_sentence,
                    c.faces_pct.map(|p| format!(" faces {p}%")).unwrap_or_default(),
                    c.title
                ));
                if detail != Detail::Titles && !c.summary.is_empty() {
                    out.push_str(&format!(" — {}", clip(&c.summary, SUMMARY_CHARS)));
                }
                out.push('\n');
                let (take, chars) = match detail {
                    Detail::Full => (3, STANDOUT_CHARS),
                    Detail::Medium => (1, 80),
                    Detail::Titles => (0, 0),
                };
                if take > 0 && !c.standout.is_empty() {
                    let lines: Vec<String> = c
                        .standout
                        .iter()
                        .take(take)
                        .map(|s| {
                            let pic = s.picture.map(|p| format!(" ({p})")).unwrap_or_default();
                            format!("{}{pic} \"{}\"", s.sentence, clip(&s.text, chars))
                        })
                        .collect();
                    out.push_str(&format!("     ★ {}\n", lines.join(" · ")));
                }
            }
            if v.chapters.iter().any(|c| c.stand_in_title) {
                out.push_str("  (titles marked by the first sentence only: no chat model was available; `ghostreel chapters --refresh-titles`)\n");
            }
        }
        out
    }

    /// The richest rendering that fits in `max_chars` (all of it when there is no limit).
    pub fn render_within(&self, max_chars: Option<usize>) -> String {
        let Some(max) = max_chars else { return self.render(Detail::Full) };
        for detail in [Detail::Full, Detail::Medium] {
            let text = self.render(detail);
            if text.len() <= max {
                return text;
            }
        }
        let mut text = self.render(Detail::Titles);
        if text.len() > max {
            let cut = (0..=max).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
            text.truncate(cut);
            text.push_str("\n… (cut to fit: ask for one video with video_id)\n");
        }
        text
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::projects::NewProject;

    /// A 70-minute interview: 840 sentences of 5 s, 14 chapters of 5 minutes, each titled, summarised
    /// and with three standout lines of realistic length.
    pub(crate) fn seventy_minute_project() -> (Db, Project) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Podcast clips test")).unwrap();
        let tmp = std::env::temp_dir().join("ghostreel-outline-fixture");
        std::fs::create_dir_all(&tmp).unwrap();
        let folder = db.add_folder(project.id, &tmp, true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s, category) VALUES (104, 'h', 1, 4200.0, 'interview')", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (104, ?1, '/m/podcast-episode-014.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        let line = "Esto es lo que yo siempre digo: o corres o te encaramas, y no hay otra manera de hacerlo bien.";
        for i in 0..840 {
            db.conn
                .execute(
                    "INSERT INTO transcript_segments(id, video_id, start_s, end_s, text) VALUES (?1, 104, ?2, ?3, ?4)",
                    rusqlite::params![1000 + i, i as f64 * 5.0, i as f64 * 5.0 + 4.8, line],
                )
                .unwrap();
        }
        crate::sentence_split::mirror_segments(&db, 104);
        let chapters: Vec<Chapter> = (0..14)
            .map(|k| Chapter {
                video_id: 104,
                idx: k,
                start_s: k as f64 * 300.0,
                end_s: k as f64 * 300.0 + 299.0,
                first_sentence_id: 1000 + (k * 60) as i64,
                last_sentence_id: 1000 + (k * 60 + 59) as i64,
                title: format!("Capitulo {k}: el trabajo duro y la disciplina"),
                summary: "Freddy cuenta como empezo en el negocio, por que la disciplina importa mas que el talento y que le dijo su padre.".into(),
                standout: vec![1000 + (k * 60 + 3) as i64, 1000 + (k * 60 + 20) as i64, 1000 + (k * 60 + 41) as i64],
                titled: true,
            })
            .collect();
        chapters::store(&db, 104, &chapters).unwrap();
        db.conn
            .execute("INSERT INTO people(id, project_id, name, role) VALUES (1, ?1, 'Freddy', 'host')", [project.id])
            .unwrap();
        let project = db.project(project.id).unwrap();
        (db, project)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::seventy_minute_project;
    use super::*;

    #[test]
    fn the_outline_of_a_seventy_minute_episode_stays_small() {
        let (db, project) = seventy_minute_project();
        let outline = build(&db, &project, None).unwrap();
        let text = outline.render(Detail::Full);
        // A few thousand tokens at most: about four characters each.
        assert!(text.len() < 9_000, "{} characters", text.len());
        assert!(text.contains("PROJECT \"Podcast clips test\" · 1 video · 70:00.0 of footage"), "{text}");
        assert!(text.contains("v104 podcast-episode-014.mp4 · 70:00.0 · interview · speech"), "{text}");
        assert!(text.contains("#1 Freddy, host, speaks 0:00.0"), "{text}");
        assert!(text.contains("14 chapters"), "{text}");
        assert!(text.contains("  1. 0:00.0-4:59.0 s1000-s1059 Capitulo 0:"), "{text}");
        assert!(text.contains("★ s1003 \"Esto es lo que yo siempre digo"), "{text}");
        // The budgeted rendering gives up detail before it gives up chapters.
        let medium = outline.render_within(Some(6_000));
        assert!(
            medium.len() <= 6_000 && medium.contains("14 chapters") && medium.matches('★').count() == 14,
            "{}",
            medium.len()
        );
        let titles = outline.render_within(Some(3_000));
        assert!(titles.len() <= 3_000 && !titles.contains('★') && titles.contains("14. "), "{}", titles.len());
        assert!(outline.render_within(Some(300)).len() <= 400);
    }

    #[test]
    fn a_video_without_chapters_says_how_to_get_them_and_one_without_speech_is_a_row() {
        let (db, project) = seventy_minute_project();
        db.conn.execute("DELETE FROM chapters", []).unwrap();
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, category) VALUES (5, 'b', 1, 12.0, 'broll')",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 SELECT 5, folder_id, '/m/drone.mp4', 1, 0, 0 FROM video_files WHERE video_id = 104",
                [],
            )
            .unwrap();
        let text = build(&db, &project, None).unwrap().render(Detail::Full);
        assert!(text.contains("no chapters yet") && text.contains("ghostreel chapters"), "{text}");
        assert!(text.contains("v5 drone.mp4 · 0:12.0 · broll\n"), "{text}");
        assert!(!text.contains("v5 drone.mp4:"), "no speech, no chapter line");
        // One video only.
        let one = build(&db, &project, Some(5)).unwrap();
        assert_eq!(one.videos.len(), 1);
    }
}
