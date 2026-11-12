//! People: project-level identities that group face tracks (and, once linked, a voice).
//!
//! Face tracks of a project's videos are clustered by mean SFace embedding. Each cluster with
//! enough screen time becomes a person; re-running matches clusters to the people already there
//! by face embedding (including faces merged into them by hand), so names, handles and roles
//! survive a re-run.

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use super::cluster::{centroid, cluster_average_linkage};
use crate::Error;
use crate::db::Db;
use crate::faces::{FaceTrack, blob_to_embedding, cosine_similarity, embedding_to_blob};

/// Same-person cut for track embeddings (the faces report uses the same value).
pub const PEOPLE_THRESHOLD: f32 = 0.5;
/// Tracks good enough to define people: at least this many faces...
const QUALITY_MIN_FACES: usize = 5;
/// ...and a face at least this tall (fraction of the frame).
const QUALITY_MIN_BOX_H: f32 = 0.12;
/// A cluster is a person when it is on screen at least this long...
const MIN_PERSON_S: f64 = 20.0;
/// ...and at least this share of the most-seen person's time (drops faces in screen shares).
const MIN_PERSON_SHARE: f64 = 0.10;
/// Other tracks (small, short) join a person when this similar to them.
const JOIN_THRESHOLD: f32 = 0.3;
/// A new cluster is an existing person when this similar to one of their faces.
const MATCH_THRESHOLD: f32 = 0.5;

/// A face box to crop: (video id, time, x, y, w, h), normalised to the frame.
pub type BestBox = (i64, f64, f32, f32, f32, f32);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Person {
    pub id: i64,
    pub project_id: i64,
    pub name: Option<String>,
    pub handle: Option<String>,
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PersonSummary {
    pub id: i64,
    pub name: Option<String>,
    pub handle: Option<String>,
    pub role: Option<String>,
    pub videos: Vec<i64>,
    pub tracks: usize,
    pub screen_time_s: f64,
    pub speaking_time_s: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PeopleReport {
    /// (person id, tracks, seconds on screen)
    pub people: Vec<(i64, usize, f64)>,
    /// Clusters too small to be a person.
    pub dropped_clusters: usize,
    /// People created in this run (the rest were matched to existing ones).
    pub created: usize,
}

fn track_secs(t: &FaceTrack) -> f64 {
    (t.end_s - t.start_s) + 0.2
}

/// Which tracks form people: indices into `tracks`, one Vec per person, most screen time first.
pub fn group_tracks(tracks: &[FaceTrack]) -> (Vec<Vec<usize>>, usize) {
    if tracks.is_empty() {
        return (Vec::new(), 0);
    }
    let mut quality: Vec<usize> = (0..tracks.len())
        .filter(|&i| tracks[i].n_faces >= QUALITY_MIN_FACES && tracks[i].avg_box_h >= QUALITY_MIN_BOX_H)
        .collect();
    if quality.is_empty() {
        quality = (0..tracks.len()).collect();
    }
    let embs: Vec<&[f32]> = quality.iter().map(|&i| tracks[i].mean_embedding.as_slice()).collect();
    let clusters = cluster_average_linkage(&embs, PEOPLE_THRESHOLD);
    let mut scored: Vec<(f64, Vec<usize>)> = clusters
        .into_iter()
        .map(|c| {
            let members: Vec<usize> = c.into_iter().map(|k| quality[k]).collect();
            (members.iter().map(|&i| track_secs(&tracks[i])).sum(), members)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let largest = scored.first().map(|s| s.0).unwrap_or(0.0);
    let min_s = MIN_PERSON_S.max(MIN_PERSON_SHARE * largest);
    let dropped = scored.iter().filter(|(d, _)| *d < min_s).count();
    let mut people: Vec<Vec<usize>> = scored.into_iter().filter(|(d, _)| *d >= min_s).map(|(_, m)| m).collect();
    if people.is_empty() {
        return (people, dropped);
    }
    let cents: Vec<Vec<f32>> = people
        .iter()
        .map(|m| centroid(&m.iter().map(|&i| tracks[i].mean_embedding.as_slice()).collect::<Vec<_>>()))
        .collect();
    let assigned: std::collections::HashSet<usize> = people.iter().flatten().copied().collect();
    for (i, t) in tracks.iter().enumerate() {
        if assigned.contains(&i) {
            continue;
        }
        let best = cents
            .iter()
            .enumerate()
            .map(|(k, c)| (k, cosine_similarity(&t.mean_embedding, c)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((k, s)) = best
            && s >= JOIN_THRESHOLD
        {
            people[k].push(i);
        }
    }
    (people, dropped)
}

impl Db {
    /// Videos of a project (not removed from it), in id order.
    pub fn project_video_ids(&self, project_id: i64) -> Result<Vec<i64>, Error> {
        let mut st = self.conn.prepare(
            "SELECT DISTINCT vf.video_id FROM video_files vf
               JOIN project_folders pf ON pf.folder_id = vf.folder_id AND pf.project_id = ?1
              WHERE NOT EXISTS (SELECT 1 FROM project_exclusions x
                                 WHERE x.project_id = ?1 AND x.video_id = vf.video_id AND x.role = 'removed')
              ORDER BY vf.video_id",
        )?;
        let rows = st.query_map([project_id], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn people(&self, project_id: i64) -> Result<Vec<Person>, Error> {
        let mut st =
            self.conn.prepare("SELECT id, project_id, name, handle, role FROM people WHERE project_id = ?1 ORDER BY id")?;
        let rows = st.query_map([project_id], |r| {
            Ok(Person { id: r.get(0)?, project_id: r.get(1)?, name: r.get(2)?, handle: r.get(3)?, role: r.get(4)? })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn person(&self, project_id: i64, person_id: i64) -> Result<Person, Error> {
        self.conn
            .query_row(
                "SELECT id, project_id, name, handle, role FROM people WHERE project_id = ?1 AND id = ?2",
                params![project_id, person_id],
                |r| Ok(Person { id: r.get(0)?, project_id: r.get(1)?, name: r.get(2)?, handle: r.get(3)?, role: r.get(4)? }),
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("person {person_id} in project {project_id}")))
    }

    /// Every face a person is known by: their mean face, plus faces merged into them.
    fn person_face_embeddings(&self, person_id: i64) -> Result<Vec<Vec<f32>>, Error> {
        let mut out = Vec::new();
        let main: Option<Vec<u8>> = self
            .conn
            .query_row("SELECT face_embedding FROM people WHERE id = ?1", [person_id], |r| r.get(0))
            .optional()?
            .flatten();
        if let Some(b) = main {
            out.push(blob_to_embedding(&b)?);
        }
        let mut st = self.conn.prepare("SELECT embedding FROM person_embeddings WHERE person_id = ?1")?;
        for b in st.query_map([person_id], |r| r.get::<_, Vec<u8>>(0))? {
            out.push(blob_to_embedding(&b?)?);
        }
        Ok(out)
    }

    /// Re-cluster a project's face tracks into people, keeping existing people (and their names)
    /// that still match, and set `face_tracks.person_id`.
    pub fn update_people(&self, project_id: i64) -> Result<PeopleReport, Error> {
        let video_ids = self.project_video_ids(project_id)?;
        let mut tracks: Vec<FaceTrack> = Vec::new();
        for vid in &video_ids {
            tracks.extend(self.face_tracks(*vid)?);
        }
        let (groups, dropped) = group_tracks(&tracks);

        let existing = self.people(project_id)?;
        let mut known: Vec<(i64, Vec<Vec<f32>>)> = Vec::new();
        for p in &existing {
            known.push((p.id, self.person_face_embeddings(p.id)?));
        }

        let tx = self.conn.unchecked_transaction()?;
        for vid in &video_ids {
            tx.execute("UPDATE face_tracks SET person_id = NULL WHERE video_id = ?1", [vid])?;
        }
        let mut report = PeopleReport { dropped_clusters: dropped, ..Default::default() };
        // person id -> member track indices (several clusters may match one merged person)
        let mut members: Vec<(i64, Vec<usize>)> = Vec::new();
        for g in &groups {
            let c = centroid(&g.iter().map(|&i| tracks[i].mean_embedding.as_slice()).collect::<Vec<_>>());
            let best = known
                .iter()
                .map(|(id, embs)| (*id, embs.iter().map(|e| cosine_similarity(e, &c)).fold(f32::NEG_INFINITY, f32::max)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let id = match best {
                Some((id, s)) if s >= MATCH_THRESHOLD => id,
                _ => {
                    tx.execute(
                        "INSERT INTO people (project_id, face_embedding) VALUES (?1, ?2)",
                        params![project_id, embedding_to_blob(&c)],
                    )?;
                    report.created += 1;
                    let id = tx.last_insert_rowid();
                    known.push((id, vec![c.clone()]));
                    id
                }
            };
            match members.iter_mut().find(|(pid, _)| *pid == id) {
                Some((_, m)) => m.extend(g.iter().copied()),
                None => members.push((id, g.clone())),
            }
        }
        {
            let mut set = tx.prepare("UPDATE face_tracks SET person_id = ?1 WHERE id = ?2")?;
            let mut emb = tx.prepare("UPDATE people SET face_embedding = ?2 WHERE id = ?1")?;
            for (pid, m) in &members {
                for &i in m {
                    set.execute(params![pid, tracks[i].id])?;
                }
                let c = centroid(&m.iter().map(|&i| tracks[i].mean_embedding.as_slice()).collect::<Vec<_>>());
                emb.execute(params![pid, embedding_to_blob(&c)])?;
                report.people.push((*pid, m.len(), m.iter().map(|&i| track_secs(&tracks[i])).sum()));
            }
        }
        // People no cluster matched: forget the unnamed ones, keep anyone the user named.
        for p in &existing {
            let matched = members.iter().any(|(id, _)| *id == p.id);
            let named = p.name.is_some() || p.handle.is_some() || p.role.is_some();
            if !matched && !named {
                tx.execute("DELETE FROM people WHERE id = ?1", [p.id])?;
            }
        }
        tx.commit()?;
        report.people.sort_by(|a, b| b.2.total_cmp(&a.2));
        Ok(report)
    }

    /// Change a person's name, handle or role (`Some("")` clears a field, `None` leaves it).
    pub fn set_person(
        &self,
        project_id: i64,
        person_id: i64,
        name: Option<&str>,
        handle: Option<&str>,
        role: Option<&str>,
    ) -> Result<Person, Error> {
        self.person(project_id, person_id)?;
        let norm = |v: &str| {
            let t = v.trim();
            if t.is_empty() { None } else { Some(t.to_string()) }
        };
        if let Some(v) = name {
            self.conn.execute("UPDATE people SET name = ?2 WHERE id = ?1", params![person_id, norm(v)])?;
        }
        if let Some(v) = handle {
            self.conn.execute("UPDATE people SET handle = ?2 WHERE id = ?1", params![person_id, norm(v)])?;
        }
        if let Some(v) = role {
            self.conn.execute("UPDATE people SET role = ?2 WHERE id = ?1", params![person_id, norm(v)])?;
        }
        self.person(project_id, person_id)
    }

    /// Merge `other` into `keep`: tracks, turns and segments move over, empty fields are filled
    /// from `other`, and `other`'s face is remembered so a re-run keeps them together.
    pub fn merge_people(&self, project_id: i64, keep: i64, other: i64) -> Result<Person, Error> {
        if keep == other {
            return Err(Error::Invalid("cannot merge a person into themselves".into()));
        }
        let k = self.person(project_id, keep)?;
        let o = self.person(project_id, other)?;
        let keep_faces = self.person_face_embeddings(keep)?;
        let other_faces = self.person_face_embeddings(other)?;
        let other_voice: Option<Vec<u8>> = self
            .conn
            .query_row("SELECT voice_embedding FROM people WHERE id = ?1", [other], |r| r.get(0))
            .optional()?
            .flatten();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("UPDATE face_tracks SET person_id = ?1 WHERE person_id = ?2", params![keep, other])?;
        tx.execute("UPDATE speaker_turns SET person_id = ?1 WHERE person_id = ?2", params![keep, other])?;
        tx.execute("UPDATE transcript_segments SET person_id = ?1 WHERE person_id = ?2", params![keep, other])?;
        tx.execute(
            "UPDATE people SET name = COALESCE(name, ?2), handle = COALESCE(handle, ?3), role = COALESCE(role, ?4),
                    voice_embedding = COALESCE(voice_embedding, ?5)
              WHERE id = ?1",
            params![keep, o.name, o.handle, o.role, other_voice],
        )?;
        for e in &other_faces {
            tx.execute(
                "INSERT INTO person_embeddings (person_id, embedding) VALUES (?1, ?2)",
                params![keep, embedding_to_blob(e)],
            )?;
        }
        if let (Some(a), Some(b)) = (keep_faces.first(), other_faces.first()) {
            let c = centroid(&[a.as_slice(), b.as_slice()]);
            tx.execute("UPDATE people SET face_embedding = ?2 WHERE id = ?1", params![keep, embedding_to_blob(&c)])?;
        }
        tx.execute("DELETE FROM people WHERE id = ?1", [other])?;
        tx.commit()?;
        let _ = k;
        self.person(project_id, keep)
    }

    /// People of a project with their videos, screen time and speaking time.
    pub fn person_summaries(&self, project_id: i64) -> Result<Vec<PersonSummary>, Error> {
        let mut out = Vec::new();
        for p in self.people(project_id)? {
            let (tracks, screen): (i64, f64) = self.conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(end_s - start_s + 0.2), 0) FROM face_tracks WHERE person_id = ?1",
                [p.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            let speaking: f64 = self.conn.query_row(
                "SELECT COALESCE(SUM(end_s - start_s), 0) FROM speaker_turns WHERE person_id = ?1",
                [p.id],
                |r| r.get(0),
            )?;
            let mut st = self.conn.prepare(
                "SELECT video_id FROM face_tracks WHERE person_id = ?1
                 UNION SELECT video_id FROM speaker_turns WHERE person_id = ?1 ORDER BY 1",
            )?;
            let videos: Vec<i64> = st.query_map([p.id], |r| r.get(0))?.collect::<Result<_, _>>()?;
            out.push(PersonSummary {
                id: p.id,
                name: p.name,
                handle: p.handle,
                role: p.role,
                videos,
                tracks: tracks as usize,
                screen_time_s: screen,
                speaking_time_s: speaking,
            });
        }
        out.sort_by(|a, b| (b.screen_time_s + b.speaking_time_s).total_cmp(&(a.screen_time_s + a.speaking_time_s)));
        Ok(out)
    }

    /// The clearest face of a person for a thumbnail: (video id, time, x, y, w, h) normalised.
    pub fn person_best_box(&self, person_id: i64) -> Result<Option<BestBox>, Error> {
        Ok(self
            .conn
            .query_row(
                "SELECT ft.video_id, fb.t_s, fb.x, fb.y, fb.w, fb.h
                   FROM face_boxes fb JOIN face_tracks ft ON ft.id = fb.track_id
                  WHERE ft.person_id = ?1 AND fb.cut_off = 0
                  ORDER BY fb.sharpness * fb.h * fb.score DESC LIMIT 1",
                [person_id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get::<_, f64>(2)? as f32,
                        r.get::<_, f64>(3)? as f32,
                        r.get::<_, f64>(4)? as f32,
                        r.get::<_, f64>(5)? as f32,
                    ))
                },
            )
            .optional()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faces::l2_normalize;
    use crate::projects::NewProject;

    pub(crate) fn track(id: i64, video_id: i64, start: f64, secs: f64, dir: usize, h: f32) -> FaceTrack {
        let mut e = vec![0.02f32; 128];
        e[dir] = 1.0;
        e[(id as usize * 7) % 128] += 0.1;
        FaceTrack {
            id,
            video_id,
            start_s: start,
            end_s: start + secs,
            n_faces: (secs * 5.0) as usize + 1,
            mean_embedding: l2_normalize(&e),
            avg_box_h: h,
            boxes: Vec::new(),
        }
    }

    #[test]
    fn groups_two_people_drops_a_glimpse_and_joins_small_faces() {
        let mut tracks = Vec::new();
        for i in 0..10 {
            tracks.push(track(i, 1, i as f64 * 60.0, 50.0, 0, 0.3)); // A: 500 s
        }
        for i in 10..18 {
            tracks.push(track(i, 1, i as f64 * 60.0, 50.0, 1, 0.3)); // B: 400 s
        }
        tracks.push(track(30, 1, 5.0, 3.0, 2, 0.3)); // C: a 3 s glimpse
        tracks.push(track(31, 1, 9.0, 40.0, 0, 0.05)); // A, but small: joins A
        let (groups, dropped) = group_tracks(&tracks);
        assert_eq!(groups.len(), 2);
        assert_eq!(dropped, 1);
        assert!(groups[0].contains(&19), "small face of A joins A");
        assert!(!groups.iter().flatten().any(|&i| i == 18), "glimpse is nobody");
    }

    fn setup() -> (Db, i64) {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project(&NewProject::named("P")).unwrap();
        db.conn
            .execute_batch(
                "INSERT INTO folders (id, path, added_at) VALUES (1, '/m', 0);
                 INSERT INTO videos (id, content_hash, size, duration_s) VALUES (1, 'h1', 1, 1000.0);
                 INSERT INTO video_files (video_id, folder_id, path, size, mtime, last_seen) VALUES (1, 1, '/m/a.mp4', 1, 0, 0);",
            )
            .unwrap();
        db.conn.execute("INSERT INTO project_folders (project_id, folder_id) VALUES (?1, 1)", [p.id]).unwrap();
        (db, p.id)
    }

    #[test]
    fn rerun_keeps_names_and_merge_survives() {
        let (db, pid) = setup();
        let mut tracks = Vec::new();
        for i in 1..=6 {
            tracks.push(track(i, 1, i as f64 * 100.0, 60.0, if i % 2 == 0 { 0 } else { 1 }, 0.3));
        }
        db.set_face_tracks(1, &tracks).unwrap();
        let r = db.update_people(pid).unwrap();
        assert_eq!(r.people.len(), 2);
        assert_eq!(r.created, 2);
        let people = db.people(pid).unwrap();
        let a = people[0].id;
        let b = people[1].id;
        db.set_person(pid, a, Some("Ana"), Some("@ana"), Some("Host")).unwrap();

        // Faces re-run: new track ids, same faces. Ana keeps her name.
        db.set_face_tracks(1, &tracks).unwrap();
        let r = db.update_people(pid).unwrap();
        assert_eq!(r.created, 0);
        let p = db.person(pid, a).unwrap();
        assert_eq!(p.name.as_deref(), Some("Ana"));
        assert_eq!(p.role.as_deref(), Some("Host"));

        // Merge B into Ana; a re-run must not split them again.
        db.merge_people(pid, a, b).unwrap();
        assert_eq!(db.people(pid).unwrap().len(), 1);
        db.update_people(pid).unwrap();
        let people = db.people(pid).unwrap();
        assert_eq!(people.len(), 1, "{people:?}");
        let s = db.person_summaries(pid).unwrap();
        assert_eq!(s[0].tracks, 6);
        assert_eq!(s[0].videos, vec![1]);
        assert!(s[0].screen_time_s > 300.0);
    }
}
