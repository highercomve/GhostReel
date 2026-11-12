use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GlossaryTerm {
    pub term: String,
    #[serde(default)]
    pub heard_as: Vec<String>,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Glossary {
    #[serde(default)]
    pub terms: Vec<GlossaryTerm>,
}

/// Case-insensitive comparison that also folds non-ASCII letters ("Él" == "él").
fn same(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

fn chars_match_case_insensitive(a: char, b: char) -> bool {
    if a == b { true } else { a.to_lowercase().eq(b.to_lowercase()) }
}

impl Glossary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clean, deduplicate and sort glossary terms.
    ///
    /// - Trims whitespace from terms, heard_as variants, and notes.
    /// - Drops terms with empty strings.
    /// - Merges duplicate terms case-insensitively, unioning their heard_as variants.
    /// - Drops heard_as variants that equal the term (case-insensitively) or are empty.
    /// - Sorts terms case-insensitively.
    pub fn normalize(&mut self) {
        let mut merged: Vec<GlossaryTerm> = Vec::new();
        for item in std::mem::take(&mut self.terms) {
            let term_trimmed = item.term.trim().to_string();
            if term_trimmed.is_empty() {
                continue;
            }
            let note_trimmed = item.note.trim().to_string();
            let key = term_trimmed.to_lowercase();

            // Collect clean heard_as variants (not empty, not equal to the term, no duplicates within entry)
            let mut heard_cleaned = Vec::new();
            for h in item.heard_as {
                let h_trimmed = h.trim().to_string();
                if !h_trimmed.is_empty()
                    && !same(&h_trimmed, &term_trimmed)
                    && !heard_cleaned.iter().any(|existing: &String| same(&existing, &h_trimmed))
                {
                    heard_cleaned.push(h_trimmed);
                }
            }

            if let Some(existing) = merged.iter_mut().find(|t| same(&t.term, &key)) {
                // Union heard_as
                for h in heard_cleaned {
                    if !existing.heard_as.iter().any(|existing_h| same(&existing_h, &h)) {
                        existing.heard_as.push(h);
                    }
                }
                // Preserve non-empty note if existing has none
                if existing.note.is_empty() && !note_trimmed.is_empty() {
                    existing.note = note_trimmed;
                }
            } else {
                merged.push(GlossaryTerm { term: term_trimmed, heard_as: heard_cleaned, note: note_trimmed });
            }
        }

        // Sort terms case-insensitively
        merged.sort_by(|a, b| a.term.to_lowercase().cmp(&b.term.to_lowercase()));
        self.terms = merged;
    }

    /// Add a term and its heard-as mishearings, then normalize.
    pub fn add(&mut self, term: impl Into<String>, heard_as: &[String]) {
        self.terms.push(GlossaryTerm { term: term.into(), heard_as: heard_as.to_vec(), note: String::new() });
        self.normalize();
    }

    /// Add a full GlossaryTerm entry, then normalize.
    pub fn add_term(&mut self, term: GlossaryTerm) {
        self.terms.push(term);
        self.normalize();
    }

    /// Remove a term (case-insensitive). Returns true if found and removed.
    pub fn remove(&mut self, term: &str) -> bool {
        let term_clean = term.trim();
        let initial_len = self.terms.len();
        self.terms.retain(|t| !same(&t.term, term_clean));
        self.terms.len() < initial_len
    }

    /// The terms as a bare list for Whisper's initial prompt ("Zero Claw, Kubernetes, homelab.").
    /// No English lead-in: with language detection on, a prompt in one language pulls the
    /// transcript toward it, and these terms are spoken inside Spanish, Portuguese... speech.
    /// Returns None when empty; capped at ~600 characters (whisper prompts are ~224 tokens max).
    pub fn whisper_prompt(&self) -> Option<String> {
        let prefix = "";
        let suffix = ".";
        let max_len = 600;
        let mut included: Vec<&str> = Vec::new();
        let mut current_len = prefix.len() + suffix.len();

        for t in &self.terms {
            let term = t.term.trim();
            if term.is_empty() {
                continue;
            }
            let added_len = if included.is_empty() {
                term.chars().count()
            } else {
                2 + term.chars().count() // ", "
            };
            if current_len + added_len <= max_len {
                included.push(term);
                current_len += added_len;
            }
        }

        if included.is_empty() { None } else { Some(format!("{prefix}{}{suffix}", included.join(", "))) }
    }

    /// Replace each heard_as variant with its term, case-insensitive, whole words/phrases only.
    /// Letters including accented ones are treated as word characters. Longest variant matches first.
    pub fn correct(&self, text: &str) -> String {
        if text.is_empty() || self.terms.is_empty() {
            return text.to_string();
        }

        // Collect all (variant, replacement_term) pairs
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        for t in &self.terms {
            let term = t.term.trim();
            if term.is_empty() {
                continue;
            }
            for h in &t.heard_as {
                let h_trim = h.trim();
                if !h_trim.is_empty() {
                    pairs.push((h_trim, term));
                }
            }
        }

        if pairs.is_empty() {
            return text.to_string();
        }

        // Longest variant first (by char count descending, then byte len descending)
        pairs.sort_by(|a, b| b.0.chars().count().cmp(&a.0.chars().count()).then_with(|| b.0.len().cmp(&a.0.len())));

        // Track non-overlapping matched byte ranges in the original text: (start, end, term)
        let mut claimed: Vec<(usize, usize, &str)> = Vec::new();

        for (variant, term) in pairs {
            let v_chars: Vec<char> = variant.chars().collect();
            if v_chars.is_empty() {
                continue;
            }

            for (start, _) in text.char_indices() {
                // Check if start is inside any already claimed range
                if claimed.iter().any(|&(s, e, _)| start >= s && start < e) {
                    continue;
                }

                // Start boundary check: preceding char cannot be alphanumeric
                if start > 0 {
                    if let Some(prev) = text[..start].chars().next_back() {
                        if prev.is_alphanumeric() {
                            continue;
                        }
                    }
                }

                // Try to match variant chars case-insensitively
                let mut text_chars = text[start..].chars();
                let mut matched = true;
                let mut bytes_consumed = 0;

                for &vc in &v_chars {
                    match text_chars.next() {
                        Some(tc) if chars_match_case_insensitive(vc, tc) => {
                            bytes_consumed += tc.len_utf8();
                        }
                        _ => {
                            matched = false;
                            break;
                        }
                    }
                }

                if !matched {
                    continue;
                }

                let end = start + bytes_consumed;

                // End boundary check: following char cannot be alphanumeric
                if end < text.len() {
                    if let Some(next) = text[end..].chars().next() {
                        if next.is_alphanumeric() {
                            continue;
                        }
                    }
                }

                // Check overlap with any already claimed range:
                let overlaps = claimed.iter().any(|&(s, e, _)| start < e && end > s);
                if !overlaps {
                    claimed.push((start, end, term));
                }
            }
        }

        if claimed.is_empty() {
            return text.to_string();
        }

        claimed.sort_by_key(|&(s, _, _)| s);

        let mut out = String::with_capacity(text.len());
        let mut last_idx = 0;
        for (start, end, term) in claimed {
            out.push_str(&text[last_idx..start]);
            out.push_str(term);
            last_idx = end;
        }
        out.push_str(&text[last_idx..]);
        out
    }

    /// Section for LLM prompts: "Project glossary (spell these exactly): term — note; ..."
    /// Returns None when empty.
    pub fn prompt_section(&self) -> Option<String> {
        let mut parts = Vec::new();
        for t in &self.terms {
            let term = t.term.trim();
            if term.is_empty() {
                continue;
            }
            let note = t.note.trim();
            if note.is_empty() {
                parts.push(term.to_string());
            } else {
                parts.push(format!("{term} — {note}"));
            }
        }
        if parts.is_empty() {
            None
        } else {
            let body = parts.join("; ");
            let body = body.trim_end_matches('.');
            Some(format!("Project glossary (spell these exactly): {body}."))
        }
    }
}

impl Db {
    /// Transcription glossary for a video, computed as the union (via `Glossary::add_term` /
    /// `normalize`) of the glossaries of every project that contains the video, excluding projects
    /// where the video is excluded.
    pub fn glossary_for_video(&self, video_id: i64) -> Result<Glossary, Error> {
        let mut st = self.conn.prepare(
            "SELECT DISTINCT COALESCE(p.glossary_json, '')
             FROM projects p
             JOIN project_folders pf ON pf.project_id = p.id
             JOIN folders f ON f.id = pf.folder_id
             JOIN video_files vf ON vf.folder_id = f.id
             WHERE vf.video_id = ?1
               AND NOT EXISTS (
                   SELECT 1 FROM project_exclusions x
                   WHERE x.project_id = p.id AND x.video_id = vf.video_id
               )",
        )?;
        let rows = st.query_map([video_id], |r| r.get::<_, String>(0))?;
        let mut merged = Glossary::default();
        for r in rows {
            let json = r?;
            if json.trim().is_empty() {
                continue;
            }
            if let Ok(g) = serde_json::from_str::<Glossary>(&json) {
                for term in g.terms {
                    merged.add_term(term);
                }
            }
        }
        merged.normalize();
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use crate::projects::NewProject;

    #[test]
    fn test_normalize_and_merge() {
        let mut g = Glossary {
            terms: vec![
                GlossaryTerm {
                    term: "  GhostReel  ".into(),
                    heard_as: vec!["ghost real".into(), "ghostreel".into(), "  ghost rail  ".into()],
                    note: "Product".into(),
                },
                GlossaryTerm {
                    term: "ghostreel".into(),
                    heard_as: vec!["ghost reel".into(), "GHOST REAL".into(), "".into()],
                    note: "".into(),
                },
                GlossaryTerm { term: "   ".into(), heard_as: vec!["nothing".into()], note: "".into() },
                GlossaryTerm {
                    term: "Antigravity".into(),
                    heard_as: vec!["anti gravity".into(), "Antigravity".into()],
                    note: "".into(),
                },
            ],
        };

        g.normalize();

        assert_eq!(g.terms.len(), 2);
        // Sorted case-insensitively: "Antigravity" before "GhostReel"
        assert_eq!(g.terms[0].term, "Antigravity");
        assert_eq!(g.terms[0].heard_as, vec!["anti gravity"]);

        assert_eq!(g.terms[1].term, "GhostReel");
        assert_eq!(g.terms[1].note, "Product");
        assert_eq!(g.terms[1].heard_as, vec!["ghost real", "ghost rail", "ghost reel"]);
    }

    #[test]
    fn accented_capitals_match_their_lowercase() {
        let mut g = Glossary::default();
        g.add("Énfasis", &["enfasis".to_string()]);
        g.add("énfasis", &["en fasis".to_string()]);
        assert_eq!(g.terms.len(), 1);
        assert_eq!(g.terms[0].heard_as, vec!["enfasis".to_string(), "en fasis".to_string()]);
        assert!(g.remove("ÉNFASIS"));
        assert!(g.terms.is_empty());
    }

    #[test]
    fn test_add_and_remove() {
        let mut g = Glossary::default();
        g.add("Whisper", &["whispr".into(), "wisper".into()]);
        assert_eq!(g.terms.len(), 1);
        assert_eq!(g.terms[0].term, "Whisper");
        assert_eq!(g.terms[0].heard_as, vec!["whispr", "wisper"]);

        // Adding duplicate term merges variants
        g.add("whisper", &["whisp".into()]);
        assert_eq!(g.terms.len(), 1);
        assert_eq!(g.terms[0].heard_as, vec!["whispr", "wisper", "whisp"]);

        // Remove case-insensitive
        assert!(g.remove("WHISPER"));
        assert_eq!(g.terms.len(), 0);
        assert!(!g.remove("nonexistent"));
    }

    #[test]
    fn test_db_glossary_roundtrip() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("GlossaryProject")).unwrap();

        // Initially empty
        assert_eq!(project.glossary.terms, Vec::<GlossaryTerm>::new());

        let mut g = Glossary::default();
        g.add("Tauri", &["tory".into(), "towry".into()]);
        g.add("Rust", &["roost".into()]);

        let saved = db.set_project_glossary(project.id, &g).unwrap();
        assert_eq!(saved.terms.len(), 2);
        assert_eq!(saved.terms[0].term, "Rust");
        assert_eq!(saved.terms[1].term, "Tauri");

        // Reload project from DB
        let reloaded = db.project(project.id).unwrap();
        assert_eq!(reloaded.glossary, saved);

        // Update with remove
        let mut updated = reloaded.glossary;
        assert!(updated.remove("rust"));
        let saved2 = db.set_project_glossary(project.id, &updated).unwrap();
        assert_eq!(saved2.terms.len(), 1);
        assert_eq!(saved2.terms[0].term, "Tauri");

        let reloaded2 = db.project(project.id).unwrap();
        assert_eq!(reloaded2.glossary.terms.len(), 1);
        assert_eq!(reloaded2.glossary.terms[0].term, "Tauri");
    }

    #[test]
    fn test_whisper_prompt() {
        let mut g = Glossary::default();
        assert_eq!(g.whisper_prompt(), None);

        g.add("Kubernetes", &[]);
        g.add("homelab", &[]);
        g.add("JEPA", &[]);
        // Terms are sorted alphabetically by normalize: homelab, JEPA, Kubernetes
        assert_eq!(g.whisper_prompt(), Some("homelab, JEPA, Kubernetes.".to_string()));

        // Cap at ~600 chars
        let mut long_g = Glossary::default();
        for i in 0..100 {
            long_g.add(format!("VeryLongTechnicalTermNumber{:03}", i), &[]);
        }
        let prompt = long_g.whisper_prompt().unwrap();
        assert!(prompt.len() <= 600);
        assert!(!prompt.starts_with("Glossary"));
        assert!(prompt.ends_with('.'));
    }

    #[test]
    fn test_prompt_section() {
        let mut g = Glossary::default();
        assert_eq!(g.prompt_section(), None);

        g.add_term(GlossaryTerm {
            term: "Kubernetes".into(),
            heard_as: vec!["cubernetes".into()],
            note: "Container orchestration".into(),
        });
        g.add_term(GlossaryTerm { term: "homelab".into(), heard_as: vec!["home laab".into()], note: "".into() });

        assert_eq!(
            g.prompt_section(),
            Some("Project glossary (spell these exactly): homelab; Kubernetes — Container orchestration.".to_string())
        );
    }

    #[test]
    fn test_correct_replacements_and_boundaries() {
        let mut g = Glossary::default();
        g.add("Kubernetes", &["cubernetes".into(), "kube".into()]);
        g.add("homelab", &["home laab".into(), "laab".into()]);
        g.add("énfasis", &["enfasis".into()]);

        // Longest variant first & word boundaries
        let text = "El cubernetes y home laab funcionan bien con subcubernetesx.";
        let corrected = g.correct(text);
        assert_eq!(corrected, "El Kubernetes y homelab funcionan bien con subcubernetesx.");

        // Inside word checks
        assert_eq!(g.correct("subcubernetes"), "subcubernetes");
        assert_eq!(g.correct("cubernetesx"), "cubernetesx");

        // Punctuation and boundaries
        assert_eq!(g.correct("¡cubernetes!"), "¡Kubernetes!");
        assert_eq!(g.correct("¿home laab?"), "¿homelab?");

        // Spanish accents
        assert_eq!(g.correct("Puso mucho enfasis en el camión"), "Puso mucho énfasis en el camión");
        assert_eq!(g.correct("el camiónenfasis"), "el camiónenfasis");
    }

    #[test]
    fn test_db_glossary_for_video() {
        let mut db = Db::open_in_memory().unwrap();
        let p1 = db.create_project(&NewProject::named("P1")).unwrap();
        let p2 = db.create_project(&NewProject::named("P2")).unwrap();

        let mut g1 = Glossary::default();
        g1.add("Kubernetes", &["cubernetes".into()]);
        db.set_project_glossary(p1.id, &g1).unwrap();

        let mut g2 = Glossary::default();
        g2.add("homelab", &["home laab".into()]);
        db.set_project_glossary(p2.id, &g2).unwrap();

        // Setup folder and video membership
        db.conn.execute("INSERT INTO folders (path, recursive, added_at) VALUES ('/media', 1, 1)", []).unwrap();
        let folder_id = db.conn.last_insert_rowid();

        db.conn
            .execute("INSERT INTO project_folders (project_id, folder_id) VALUES (?1, ?2)", [p1.id, folder_id])
            .unwrap();
        db.conn
            .execute("INSERT INTO project_folders (project_id, folder_id) VALUES (?1, ?2)", [p2.id, folder_id])
            .unwrap();

        db.conn.execute("INSERT INTO videos (content_hash, size) VALUES ('hash1', 100)", []).unwrap();
        let video_id = db.conn.last_insert_rowid();

        db.conn.execute(
            "INSERT INTO video_files (video_id, folder_id, path, mtime, size, last_seen) VALUES (?1, ?2, 'clip.mp4', 1, 100, 1)",
            [video_id, folder_id],
        ).unwrap();

        // Video is in both projects -> union of glossaries
        let union_g = db.glossary_for_video(video_id).unwrap();
        assert_eq!(union_g.terms.len(), 2);
        assert_eq!(union_g.terms[0].term, "homelab");
        assert_eq!(union_g.terms[1].term, "Kubernetes");

        // Now exclude video from P2
        db.conn.execute(
            "INSERT INTO project_exclusions (project_id, video_id, role, excluded_at) VALUES (?1, ?2, 'removed', 1)",
            [p2.id, video_id],
        ).unwrap();

        let filtered_g = db.glossary_for_video(video_id).unwrap();
        assert_eq!(filtered_g.terms.len(), 1);
        assert_eq!(filtered_g.terms[0].term, "Kubernetes");
    }
}
