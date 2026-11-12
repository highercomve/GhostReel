//! The trash: editing work the user made (cuts, single script versions, clip sets with their
//! clips) is never deleted outright.
//!
//! A delete moves the rows into one trash entry (`trash` table; the rows keep its id in
//! `trash_id`), and every listing and read skips them. [`restore`] puts them back exactly as they
//! were. Entries older than [`KEEP_DAYS`] are purged for good by [`purge_expired`], which the app
//! runs at start-up.
//!
//! Every delete, restore and purge is recorded as a run (kinds [`DELETE_RUN`], [`RESTORE_RUN`],
//! [`PURGE_RUN`]) linking to what it touched, so the activity log says what went and when, and a
//! link to a trashed item ([`crate::runs::LinkState::Trashed`]) offers to restore it.
//!
//! The words a front end shows ("clip set “Launch” (8 clips)") are made here, in [`describe`].

use std::path::Path;

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::runs::{RunLink, RunRecorder, RunState};

/// How long a deleted item can be restored before it is purged.
pub const KEEP_DAYS: i64 = 30;
const KEEP_S: i64 = KEEP_DAYS * 24 * 3600;

/// Run kinds for what happens to the trash.
pub const DELETE_RUN: &str = "delete";
pub const RESTORE_RUN: &str = "restore";
pub const PURGE_RUN: &str = "purge";

/// What a trash entry holds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrashKind {
    /// Every version of a cut (one title).
    Cut,
    /// One version of a cut.
    ScriptVersion,
    /// A clip set and every version of its clips.
    ClipSet,
}

impl TrashKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TrashKind::Cut => "cut",
            TrashKind::ScriptVersion => "script_version",
            TrashKind::ClipSet => "clip_set",
        }
    }

    fn parse(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "cut" => TrashKind::Cut,
            "script_version" => TrashKind::ScriptVersion,
            "clip_set" => TrashKind::ClipSet,
            other => return Err(Error::Invalid(format!("unknown trash kind '{other}'"))),
        })
    }
}

/// One delete, restorable until `purge_at`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrashItem {
    pub id: String,
    pub project_id: i64,
    pub kind: TrashKind,
    /// The cut's title (a version's too), or the clip set's brief (may be empty).
    pub title: String,
    /// What was deleted, in words, capitalised: "Clip set “Launch” (8 clips)".
    pub label: String,
    /// Script rows it holds (versions of the cut, or every version of the set's clips).
    pub scripts: usize,
    /// Clips of a clip set (0 for cuts).
    pub clips: usize,
    /// The script version number, for a single version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
    pub deleted_at: i64,
    /// When [`purge_expired`] removes it for good.
    pub purge_at: i64,
    /// What it is, as runs link to it (the cut's newest version, the version, the set).
    pub link: RunLink,
}

/// What was deleted, in words, lower case: "cut “Promo” (3 versions)", "version 2 of “Promo”",
/// "clip set “Launch” (8 clips)".
pub fn describe(kind: TrashKind, title: &str, scripts: usize, clips: usize, version: Option<i64>) -> String {
    let plural = |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    match kind {
        TrashKind::Cut if scripts > 1 => format!("cut “{title}” ({})", plural(scripts, "version", "versions")),
        TrashKind::Cut => format!("cut “{title}”"),
        TrashKind::ScriptVersion => match version {
            Some(v) => format!("version {v} of “{title}”"),
            None => format!("a version of “{title}”"),
        },
        TrashKind::ClipSet => {
            let brief = short(title);
            let n = plural(clips, "clip", "clips");
            if brief.is_empty() { format!("clip set ({n})") } else { format!("clip set “{brief}” ({n})") }
        }
    }
}

/// A brief cut to one line of reasonable length.
fn short(s: &str) -> String {
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    if line.chars().count() <= 60 {
        return line.to_string();
    }
    let cut: String = line.chars().take(57).collect();
    format!("{}…", cut.trim_end())
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn new_trash_id() -> String {
    format!("t{}", &crate::runs::new_run_id()[1..])
}

fn row_to_item(r: &rusqlite::Row) -> rusqlite::Result<(String, i64, String, String, i64, i64, String, i64)> {
    Ok((
        r.get("id")?,
        r.get("project_id")?,
        r.get("kind")?,
        r.get("title")?,
        r.get("scripts")?,
        r.get("clips")?,
        r.get("link_json")?,
        r.get("deleted_at")?,
    ))
}

fn build_item(
    db: &Db,
    (id, project_id, kind, title, scripts, clips, link_json, deleted_at): (
        String,
        i64,
        String,
        String,
        i64,
        i64,
        String,
        i64,
    ),
) -> Result<TrashItem, Error> {
    let kind = TrashKind::parse(&kind)?;
    let link: RunLink =
        serde_json::from_str(&link_json).map_err(|e| Error::Invalid(format!("trash {id}: bad link: {e}")))?;
    let version = match (&kind, &link) {
        (TrashKind::ScriptVersion, RunLink::Script { id: sid }) => {
            db.conn.query_row("SELECT version FROM scripts WHERE id = ?1", [sid], |r| r.get(0)).optional()?
        }
        _ => None,
    };
    let (scripts, clips) = (scripts.max(0) as usize, clips.max(0) as usize);
    Ok(TrashItem {
        label: capitalise(&describe(kind, &title, scripts, clips, version)),
        id,
        project_id,
        kind,
        title,
        scripts,
        clips,
        version,
        deleted_at,
        purge_at: deleted_at + KEEP_S,
        link,
    })
}

/// One trash entry.
pub fn get(db: &Db, trash_id: &str) -> Result<TrashItem, Error> {
    let row = db
        .conn
        .query_row("SELECT * FROM trash WHERE id = ?1", [trash_id], row_to_item)
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("trash entry {trash_id}")))?;
    build_item(db, row)
}

/// What is in the trash, newest first: one project's, or everyone's.
pub fn list(db: &Db, project_id: Option<i64>) -> Result<Vec<TrashItem>, Error> {
    let mut st = db.conn.prepare(
        "SELECT * FROM trash WHERE ?1 IS NULL OR project_id = ?1 ORDER BY deleted_at DESC, rowid DESC",
    )?;
    let rows = st.query_map([project_id], row_to_item)?.collect::<Result<Vec<_>, _>>()?;
    rows.into_iter().map(|r| build_item(db, r)).collect()
}

/// Move the rows a delete takes into one new trash entry, record the delete as a run, and return
/// the entry. Only rows still live are taken.
#[allow(clippy::too_many_arguments)]
fn trash_rows(
    db: &Db,
    project_id: i64,
    kind: TrashKind,
    title: &str,
    clips: usize,
    link: RunLink,
    script_ids: &[i64],
    set_id: Option<&str>,
    source: &str,
) -> Result<TrashItem, Error> {
    let id = new_trash_id();
    let now = crate::projects::now();
    let tx = db.conn.unchecked_transaction()?;
    tx.execute(
        "INSERT INTO trash(id, project_id, kind, title, scripts, clips, link_json, deleted_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            id,
            project_id,
            kind.as_str(),
            title,
            script_ids.len() as i64,
            clips as i64,
            serde_json::to_string(&link).unwrap_or_default(),
            now
        ],
    )?;
    {
        let mut st = tx.prepare("UPDATE scripts SET trash_id = ?1 WHERE id = ?2 AND trash_id IS NULL")?;
        for sid in script_ids {
            st.execute(params![id, sid])?;
        }
    }
    if let Some(set_id) = set_id {
        tx.execute("UPDATE clip_sets SET trash_id = ?1 WHERE id = ?2 AND trash_id IS NULL", params![id, set_id])?;
    }
    tx.commit()?;
    let item = get(db, &id)?;
    record_best_effort(db, DELETE_RUN, &item, source);
    Ok(item)
}

/// The run saying what happened to an entry ("Deleted clip set “Launch” (8 clips)"), linking to it.
fn record(db: &Db, kind: &str, item: &TrashItem, source: &str) -> Result<(), Error> {
    let what = describe(item.kind, &item.title, item.scripts, item.clips, item.version);
    let (label, summary) = match kind {
        RESTORE_RUN => (format!("Restored {what}"), "Back where it was".to_string()),
        _ => (
            format!("Deleted {what}"),
            format!("In Recently deleted for {KEEP_DAYS} days; restore it from there or from here"),
        ),
    };
    let mut rec = RunRecorder::start(kind, label, Some(item.project_id), source);
    rec.finish(RunState::Done, Some(summary), vec![item.link.clone()], None);
    rec.save(db)
}

/// [`record`] after the change is committed: the run is only a log of it, so failing to write it
/// is logged and the operation still succeeds.
fn record_best_effort(db: &Db, kind: &str, item: &TrashItem, source: &str) {
    if let Err(e) = record(db, kind, item, source) {
        tracing::warn!("trash: cannot record the {kind} run for {}: {e}", item.id);
    }
}

/// Move every version of a cut to the trash.
pub fn trash_cut(db: &Db, project_id: i64, title: &str, source: &str) -> Result<TrashItem, Error> {
    let ids: Vec<i64> = db
        .conn
        .prepare(
            "SELECT id FROM scripts WHERE project_id = ?1 AND title = ?2 AND trash_id IS NULL ORDER BY version DESC",
        )?
        .query_map(params![project_id, title], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    let Some(&newest) = ids.first() else {
        return Err(Error::NotFound(format!("cut “{title}”")));
    };
    trash_rows(db, project_id, TrashKind::Cut, title, 0, RunLink::Script { id: newest }, &ids, None, source)
}

/// Move one script version to the trash.
pub fn trash_script(db: &Db, script_id: i64, source: &str) -> Result<TrashItem, Error> {
    let (project_id, title): (i64, String) = db
        .conn
        .query_row("SELECT project_id, title FROM scripts WHERE id = ?1 AND trash_id IS NULL", [script_id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("script #{script_id}")))?;
    trash_rows(
        db,
        project_id,
        TrashKind::ScriptVersion,
        &title,
        0,
        RunLink::Script { id: script_id },
        &[script_id],
        None,
        source,
    )
}

/// Move a clip set and every version of its clips to the trash.
pub fn trash_clip_set(db: &Db, set_id: &str, source: &str) -> Result<TrashItem, Error> {
    let (project_id, brief): (i64, String) = db
        .conn
        .query_row("SELECT project_id, brief FROM clip_sets WHERE id = ?1 AND trash_id IS NULL", [set_id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?
        .ok_or_else(|| Error::NotFound(format!("clip set {set_id}")))?;
    let rows: Vec<(i64, String)> = db
        .conn
        .prepare("SELECT id, title FROM scripts WHERE set_id = ?1 AND trash_id IS NULL")?
        .query_map([set_id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    let clips = rows.iter().map(|(_, t)| t.as_str()).collect::<std::collections::HashSet<_>>().len();
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    trash_rows(
        db,
        project_id,
        TrashKind::ClipSet,
        &brief,
        clips,
        RunLink::ClipSet { id: set_id.to_string() },
        &ids,
        Some(set_id),
        source,
    )
}

/// Put an entry's rows back exactly as they were, record it, and return what came back.
pub fn restore(db: &Db, trash_id: &str, source: &str) -> Result<TrashItem, Error> {
    let item = get(db, trash_id)?;
    let tx = db.conn.unchecked_transaction()?;
    tx.execute("UPDATE scripts SET trash_id = NULL WHERE trash_id = ?1", [trash_id])?;
    tx.execute("UPDATE clip_sets SET trash_id = NULL WHERE trash_id = ?1", [trash_id])?;
    // Another process may have purged (or restored) the entry since `get`; then there is nothing
    // to restore, and the transaction rolls back untouched.
    if tx.execute("DELETE FROM trash WHERE id = ?1", [trash_id])? == 0 {
        return Err(Error::NotFound(format!("trash entry {trash_id}")));
    }
    tx.commit()?;
    record_best_effort(db, RESTORE_RUN, &item, source);
    Ok(item)
}

/// The trash entry holding what `link` points at, if it is in the trash.
pub fn entry_for(db: &Db, link: &RunLink) -> Result<Option<String>, Error> {
    let col = |sql: &str, key: &dyn rusqlite::ToSql| -> Result<Option<String>, Error> {
        Ok(db.conn.query_row(sql, [key], |r| r.get::<_, Option<String>>(0)).optional()?.flatten())
    };
    match link {
        RunLink::ClipSet { id } => col("SELECT trash_id FROM clip_sets WHERE id = ?1", id),
        RunLink::Script { id } => col("SELECT trash_id FROM scripts WHERE id = ?1", id),
        RunLink::File { .. } | RunLink::Video { .. } | RunLink::Project { .. } => Ok(None),
    }
}

/// Restore whatever trash entry holds what `link` points at (a run's link to a deleted clip set
/// or cut). Restoring a cut's version link brings the whole delete back, not just that version.
pub fn restore_link(db: &Db, link: &RunLink, source: &str) -> Result<TrashItem, Error> {
    let trash_id = entry_for(db, link)?.ok_or_else(|| Error::NotFound("that item is not in the trash".into()))?;
    restore(db, &trash_id, source)
}

/// Remove for good every entry deleted more than [`KEEP_DAYS`] before `now` (seconds), with its
/// rows and the default preview renders under `data_dir` of the scripts among them (ids are
/// reused after a purge, so a leftover file would show up as a later script's preview). Records
/// one run when anything went. Returns how many entries.
pub fn purge_expired(db: &Db, data_dir: &Path, now: i64, source: &str) -> Result<usize, Error> {
    let cutoff = now - KEEP_S;
    let expired = {
        let mut st = db.conn.prepare("SELECT * FROM trash WHERE deleted_at < ?1 ORDER BY deleted_at")?;
        st.query_map([cutoff], row_to_item)?.collect::<Result<Vec<_>, _>>()?
    };
    if expired.is_empty() {
        return Ok(0);
    }
    let items = expired.into_iter().map(|r| build_item(db, r)).collect::<Result<Vec<_>, _>>()?;
    let tx = db.conn.unchecked_transaction()?;
    let mut gone_scripts: Vec<(i64, i64)> = Vec::new();
    for item in &items {
        gone_scripts.extend(
            tx.prepare("SELECT id, version FROM scripts WHERE trash_id = ?1")?
                .query_map([&item.id], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<Vec<(i64, i64)>, _>>()?,
        );
        tx.execute("DELETE FROM scripts WHERE trash_id = ?1", [&item.id])?;
        tx.execute("DELETE FROM clip_sets WHERE trash_id = ?1", [&item.id])?;
        tx.execute("DELETE FROM trash WHERE id = ?1", [&item.id])?;
    }
    tx.commit()?;
    for (id, version) in gone_scripts {
        crate::preview::remove_default_preview(data_dir, id, version);
    }
    // One run per project, naming what went.
    let mut by_project: std::collections::BTreeMap<i64, Vec<&TrashItem>> = Default::default();
    for item in &items {
        by_project.entry(item.project_id).or_default().push(item);
    }
    for (project_id, gone) in by_project {
        let n = gone.len();
        let label = format!(
            "Emptied {n} {} deleted over {KEEP_DAYS} days ago",
            if n == 1 { "item" } else { "items" }
        );
        let names: Vec<String> =
            gone.iter().map(|i| describe(i.kind, &i.title, i.scripts, i.clips, i.version)).collect();
        let mut rec = RunRecorder::start(PURGE_RUN, label, Some(project_id), source);
        rec.finish(RunState::Done, Some(capitalise(&names.join("; "))), Vec::new(), None);
        if let Err(e) = rec.save(db) {
            tracing::warn!("trash: cannot record the {PURGE_RUN} run: {e}");
        }
    }
    Ok(items.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::LinkState;
    use crate::script::Script;

    fn project(db: &Db) -> i64 {
        db.conn.execute("INSERT INTO projects(id, name, created_at) VALUES (1, 'P', 0)", []).unwrap();
        1
    }

    fn save(db: &Db, title: &str) -> i64 {
        let s: Script = serde_json::from_value(serde_json::json!({
            "title": title,
            "beats": [{ "id": "b1", "purpose": "Open", "clips": [] }],
        }))
        .unwrap();
        crate::script::save_version(db, 1, &s, None).unwrap()
    }

    fn clip_set(db: &Db, id: &str, titles: &[&str]) -> Vec<i64> {
        db.conn
            .execute(
                "INSERT INTO clip_sets(id, project_id, brief, created_at) VALUES (?1, 1, 'Launch day', 5)",
                [id],
            )
            .unwrap();
        titles
            .iter()
            .map(|t| {
                let sid = save(db, t);
                db.conn.execute("UPDATE scripts SET set_id = ?1, kind = 'clip' WHERE id = ?2", params![id, sid]).unwrap();
                sid
            })
            .collect()
    }

    fn titles(db: &Db) -> Vec<(String, i64)> {
        crate::script::list(db, 1).unwrap().into_iter().map(|s| (s.title, s.version)).collect()
    }

    #[test]
    fn a_cut_goes_to_the_trash_and_comes_back_exactly() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        save(&db, "Promo");
        let v2 = save(&db, "Promo");
        save(&db, "Other");

        let item = trash_cut(&db, 1, "Promo", "app").unwrap();
        assert_eq!((item.kind, item.scripts), (TrashKind::Cut, 2));
        assert_eq!(item.label, "Cut “Promo” (2 versions)");
        assert_eq!(item.link, RunLink::Script { id: v2 });
        assert_eq!(titles(&db), vec![("Other".into(), 1)], "listings skip trashed rows");
        assert!(crate::script::load(&db, v2).is_err(), "reads skip trashed rows");
        assert!(matches!(trash_cut(&db, 1, "Promo", "app"), Err(Error::NotFound(_))));

        // A new cut of the same title does not collide with the trashed versions.
        let v3 = save(&db, "Promo");
        assert_eq!(crate::script::load(&db, v3).unwrap().version, 3);

        assert_eq!(list(&db, Some(1)).unwrap().len(), 1);
        let back = restore(&db, &item.id, "app").unwrap();
        assert_eq!(back.id, item.id);
        assert_eq!(
            titles(&db),
            vec![("Promo".into(), 3), ("Other".into(), 1), ("Promo".into(), 2), ("Promo".into(), 1)]
        );
        assert!(list(&db, None).unwrap().is_empty());
        assert!(matches!(restore(&db, &item.id, "app"), Err(Error::NotFound(_))));
    }

    #[test]
    fn one_version_goes_alone() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        let v1 = save(&db, "Promo");
        save(&db, "Promo");
        let item = trash_script(&db, v1, "cli").unwrap();
        assert_eq!(item.label, "Version 1 of “Promo”");
        assert_eq!(titles(&db), vec![("Promo".into(), 2)]);
        restore_link(&db, &RunLink::Script { id: v1 }, "cli").unwrap();
        assert_eq!(titles(&db).len(), 2);
    }

    #[test]
    fn a_clip_set_takes_its_clips_and_brings_them_back() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        let ids = clip_set(&db, "cs1", &["Clip A", "Clip B"]);
        // An edit of a clip saves a new version, which stays in the set.
        let a2 = save(&db, "Clip A");
        save(&db, "Promo");

        let item = trash_clip_set(&db, "cs1", "app").unwrap();
        assert_eq!((item.scripts, item.clips), (3, 2));
        assert_eq!(item.label, "Clip set “Launch day” (2 clips)");
        assert!(crate::clips::list_sets(&db, 1).unwrap().is_empty());
        assert!(crate::clips::load_set(&db, "cs1").is_err());
        assert_eq!(titles(&db), vec![("Promo".into(), 1)]);
        assert_eq!(crate::runs::links_state(&db, &[item.link.clone()]).unwrap(), vec![LinkState::Trashed {
            trash_id: item.id.clone()
        }]);

        restore_link(&db, &RunLink::ClipSet { id: "cs1".into() }, "app").unwrap();
        assert_eq!(crate::clips::list_sets(&db, 1).unwrap().len(), 1);
        assert_eq!(titles(&db).len(), 4);
        for id in ids.iter().chain([&a2]) {
            assert!(crate::script::load(&db, *id).is_ok());
        }
        assert_eq!(crate::runs::links_state(&db, &[item.link]).unwrap(), vec![LinkState::Live]);
    }

    #[test]
    fn deletes_and_restores_are_recorded_as_runs_linking_to_the_item() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        clip_set(&db, "cs1", &["Clip A"]);
        let item = trash_clip_set(&db, "cs1", "app").unwrap();
        let runs = db.runs_for_project(1, 10).unwrap();
        assert_eq!(runs.len(), 1);
        let r = &runs[0];
        assert_eq!((r.kind.as_str(), r.state, r.source.as_str()), (DELETE_RUN, RunState::Done, "app"));
        assert_eq!(r.label, "Deleted clip set “Launch day” (1 clip)");
        assert_eq!(r.links, vec![RunLink::ClipSet { id: "cs1".into() }]);

        restore(&db, &item.id, "cli").unwrap();
        let runs = db.runs_for_project(1, 10).unwrap();
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().any(|r| r.kind == RESTORE_RUN && r.label == "Restored clip set “Launch day” (1 clip)"));
    }

    #[test]
    fn old_entries_are_purged_for_good() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open_in_memory().unwrap();
        project(&db);
        let old = save(&db, "Old");
        save(&db, "Fresh");
        clip_set(&db, "cs1", &["Clip A"]);
        let a = trash_cut(&db, 1, "Old", "app").unwrap();
        let b = trash_cut(&db, 1, "Fresh", "app").unwrap();
        let c = trash_clip_set(&db, "cs1", "app").unwrap();
        let now = crate::projects::now();
        let long_ago = now - KEEP_S - 60;
        db.conn.execute("UPDATE trash SET deleted_at = ?1 WHERE id IN (?2, ?3)", params![long_ago, a.id, c.id]).unwrap();

        assert_eq!(purge_expired(&db, dir.path(), now, "app").unwrap(), 2);
        let left: Vec<String> = list(&db, Some(1)).unwrap().into_iter().map(|i| i.id).collect();
        assert_eq!(left, vec![b.id]);
        let n: i64 = db.conn.query_row("SELECT count(*) FROM scripts WHERE id = ?1", [old], |r| r.get(0)).unwrap();
        assert_eq!(n, 0, "purged rows are gone");
        let sets: i64 = db.conn.query_row("SELECT count(*) FROM clip_sets", [], |r| r.get(0)).unwrap();
        assert_eq!(sets, 0);
        assert!(db.runs_for_project(1, 20).unwrap().iter().any(|r| r.kind == PURGE_RUN));
        assert_eq!(crate::runs::links_state(&db, &[a.link]).unwrap(), vec![LinkState::Gone]);
        assert_eq!(purge_expired(&db, dir.path(), now, "app").unwrap(), 0);
    }

    #[test]
    fn purging_deletes_the_preview_files_so_a_reused_id_shows_none() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let db = Db::open_in_memory().unwrap();
        project(&db);
        let keep = save(&db, "Keep");
        let old = save(&db, "Old");
        let render = |id: i64, version: i64| {
            let mp4 = crate::preview::default_preview_path(data, id, version);
            std::fs::create_dir_all(mp4.parent().unwrap()).unwrap();
            std::fs::write(&mp4, b"mp4").unwrap();
            std::fs::write(mp4.with_extension("srt"), b"srt").unwrap();
            mp4
        };
        let old_mp4 = render(old, 1);
        let keep_mp4 = render(keep, 1);
        let entry = trash_cut(&db, 1, "Old", "app").unwrap();
        trash_cut(&db, 1, "Keep", "app").unwrap();
        db.conn.execute("UPDATE trash SET deleted_at = 0 WHERE id = ?1", [&entry.id]).unwrap();

        assert_eq!(purge_expired(&db, data, crate::projects::now(), "app").unwrap(), 1);
        assert!(!old_mp4.exists() && !old_mp4.with_extension("srt").exists(), "purged script's files go");
        assert!(keep_mp4.exists(), "a trashed script still in the trash keeps its preview");

        // SQLite hands the freed id to the next script; it must not find the old render.
        let reused = save(&db, "New");
        assert_eq!(reused, old, "the id is reused");
        assert_eq!(crate::preview::find_preview(&db, data, reused).unwrap(), None);
    }

    #[test]
    fn a_failing_run_record_does_not_fail_the_delete_or_the_restore() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        save(&db, "Promo");
        db.conn.execute_batch("DROP TABLE runs").unwrap();

        let item = trash_cut(&db, 1, "Promo", "app").unwrap();
        assert!(titles(&db).is_empty(), "the delete went through");
        restore(&db, &item.id, "app").unwrap();
        assert_eq!(titles(&db), vec![("Promo".into(), 1)], "and so did the restore");
    }

    #[test]
    fn restoring_an_entry_purged_meanwhile_is_not_found_and_records_nothing() {
        let db = Db::open_in_memory().unwrap();
        project(&db);
        save(&db, "Promo");
        let item = trash_cut(&db, 1, "Promo", "app").unwrap();
        // The entry was read, then another process purged it before the restore's delete ran:
        // the trigger makes that delete touch no row.
        db.conn
            .execute_batch("CREATE TRIGGER gone BEFORE DELETE ON trash BEGIN SELECT RAISE(IGNORE); END")
            .unwrap();
        assert!(matches!(restore(&db, &item.id, "app"), Err(Error::NotFound(_))));
        assert!(titles(&db).is_empty(), "the restore rolled back");
        let runs = db.runs_for_project(1, 10).unwrap();
        assert!(runs.iter().all(|r| r.kind != RESTORE_RUN), "no “Restored” run");
    }

    #[test]
    fn descriptions() {
        assert_eq!(describe(TrashKind::Cut, "A", 1, 0, None), "cut “A”");
        assert_eq!(describe(TrashKind::ClipSet, "", 4, 4, None), "clip set (4 clips)");
        let long = "x".repeat(80);
        assert!(describe(TrashKind::ClipSet, &long, 1, 1, None).contains('…'));
    }
}
