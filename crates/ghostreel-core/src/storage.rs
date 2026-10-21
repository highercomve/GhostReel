//! House cleaning: what GhostReel keeps under the data dir, who it belongs to, and removing it.
//!
//! Every cache file is traced back to an owner — a video (by content hash), a script or a chat
//! session — and from there to the projects that use it. Footage two projects watch counts for
//! both. Files whose owner is gone (a deleted script, footage no project watches any more) are
//! "unused" and can go without anything visible changing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;

/// A kind of file GhostReel writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Playback proxies and the per-clip proxies preview renders are cut from. Rebuilt on demand.
    Proxies,
    /// Keyframe images: what frame descriptions were made from, and the thumbnails.
    Keyframes,
    /// Script preview renders.
    Previews,
    /// Images pasted into script chats.
    ChatImages,
    /// Timeline exports written into the data dir.
    Exports,
    /// Scratch files of the chat helpers.
    Temp,
}

impl Kind {
    /// What clearing the caches removes: everything that is rebuilt on demand or is scratch.
    pub const CACHES: [Kind; 3] = [Kind::Proxies, Kind::Previews, Kind::Temp];

    pub fn parse(s: &str) -> Result<Kind, Error> {
        Ok(match s {
            "proxies" => Kind::Proxies,
            "keyframes" | "frames" => Kind::Keyframes,
            "previews" => Kind::Previews,
            "chat_images" | "chat-images" => Kind::ChatImages,
            "exports" => Kind::Exports,
            "temp" | "tmp" => Kind::Temp,
            _ => {
                return Err(Error::Invalid(format!(
                    "unknown kind '{s}' (proxies, keyframes, previews, chat_images, exports, temp)"
                )));
            }
        })
    }
}

/// Bytes per kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub proxies: u64,
    pub keyframes: u64,
    pub previews: u64,
    pub chat_images: u64,
    pub exports: u64,
    pub temp: u64,
}

impl Usage {
    fn add(&mut self, kind: Kind, bytes: u64) {
        *match kind {
            Kind::Proxies => &mut self.proxies,
            Kind::Keyframes => &mut self.keyframes,
            Kind::Previews => &mut self.previews,
            Kind::ChatImages => &mut self.chat_images,
            Kind::Exports => &mut self.exports,
            Kind::Temp => &mut self.temp,
        } += bytes;
    }

    pub fn total(&self) -> u64 {
        self.proxies + self.keyframes + self.previews + self.chat_images + self.exports + self.temp
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectUsage {
    pub project_id: i64,
    pub name: String,
    pub videos: i64,
    pub usage: Usage,
    /// The part of `usage` that belongs to footage other projects use too.
    pub shared: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub data_dir: PathBuf,
    /// Everything under the data dir.
    pub total: u64,
    /// All cache files, each counted once.
    pub usage: Usage,
    pub models: u64,
    /// The library database, with its write-ahead log and any legacy copy.
    pub database: u64,
    /// The rest: logs, config, anything not listed above.
    pub other: u64,
    /// Files no project uses: footage no project watches, deleted scripts and chats, scratch.
    pub unused: Usage,
    /// Indexed videos no project watches any more.
    pub unused_videos: i64,
    pub projects: Vec<ProjectUsage>,
}

/// Which files [`clean`] removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Everywhere.
    All,
    /// Files of one project's footage, scripts and chats (shared footage included).
    Project(i64),
    /// Files nothing uses. Also forgets the index of videos no project watches.
    Unused,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CleanStats {
    pub files_deleted: usize,
    pub bytes_freed: u64,
    /// Videos whose index (transcripts, descriptions, vectors) was dropped.
    pub videos: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Owner {
    /// The hex part of a content hash.
    Video(String),
    Script(i64),
    Session(i64),
    None,
}

#[derive(Debug, Clone)]
struct Item {
    kind: Kind,
    path: PathBuf,
    bytes: u64,
    owner: Owner,
}

/// Total size of a file or directory tree (best effort).
pub fn size_of(path: &Path) -> u64 {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => {
            std::fs::read_dir(path).map(|entries| entries.flatten().map(|e| size_of(&e.path())).sum()).unwrap_or(0)
        }
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

/// Not written to for an hour: a temporary file no running job owns.
fn is_stale(entry: &std::fs::DirEntry) -> bool {
    entry
        .metadata()
        .and_then(|m| m.modified())
        .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > std::time::Duration::from_secs(3600)))
}

fn entries(dir: &Path) -> Vec<std::fs::DirEntry> {
    std::fs::read_dir(dir).map(|rd| rd.flatten().collect()).unwrap_or_default()
}

/// The hex key a content hash (`b3e:<hex>`) is filed under.
pub fn hash_key(content_hash: &str) -> &str {
    content_hash.rsplit(':').next().unwrap_or(content_hash)
}

/// The number after `prefix` at the start of `name`: `script_12_v3.mp4` → 12.
fn id_after(name: &str, prefix: &str) -> Option<i64> {
    let rest = name.strip_prefix(prefix)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Every file GhostReel wrote under the data dir, with its owner.
fn inventory(data_dir: &Path) -> Vec<Item> {
    let mut items = Vec::new();

    // frames/<2 hex>/<hex>/
    for shard in entries(&data_dir.join("frames")) {
        for dir in entries(&shard.path()) {
            let key = dir.file_name().to_string_lossy().to_string();
            items.push(Item {
                kind: Kind::Keyframes,
                bytes: size_of(&dir.path()),
                path: dir.path(),
                owner: Owner::Video(key),
            });
        }
    }

    // proxies/<hash>_<in>_<out>_….mp4 (clip proxies; `:` in the hash written as `-`) and
    // proxies/<2 hex>/<hex>-playback-….mp4 (playback proxies).
    for e in entries(&data_dir.join("proxies")) {
        let name = e.file_name().to_string_lossy().to_string();
        if e.path().is_dir() {
            for f in entries(&e.path()) {
                let fname = f.file_name().to_string_lossy().to_string();
                let owner = match fname.split_once(crate::preview::PLAYBACK_PROXY_TAG) {
                    // A render in progress, or one a crash left behind.
                    Some((_, rest)) if rest.contains(".part") && is_stale(&f) => Owner::None,
                    Some((hex, _)) => Owner::Video(hex.to_string()),
                    // An older playback format nothing reads any more.
                    None => Owner::None,
                };
                items.push(Item { kind: Kind::Proxies, bytes: size_of(&f.path()), path: f.path(), owner });
            }
        } else {
            let owner = match name.split_once('_') {
                Some((hash, _)) => Owner::Video(hash.rsplit('-').next().unwrap_or(hash).to_string()),
                None => Owner::None,
            };
            items.push(Item { kind: Kind::Proxies, bytes: size_of(&e.path()), path: e.path(), owner });
        }
    }

    // previews/script_<id>_v<n>.mp4 and its intermediates (concat_<id>_….txt).
    for e in entries(&data_dir.join("previews")) {
        let name = e.file_name().to_string_lossy().to_string();
        let owner =
            id_after(&name, "script_").or_else(|| id_after(&name, "concat_")).map(Owner::Script).unwrap_or(Owner::None);
        items.push(Item { kind: Kind::Previews, bytes: size_of(&e.path()), path: e.path(), owner });
    }

    // chat_images/chat_<session>_<ts>_<n>.<ext>
    for e in entries(&data_dir.join("chat_images")) {
        let name = e.file_name().to_string_lossy().to_string();
        let owner = id_after(&name, "chat_").map(Owner::Session).unwrap_or(Owner::None);
        items.push(Item { kind: Kind::ChatImages, bytes: size_of(&e.path()), path: e.path(), owner });
    }

    // exports/script-<id>-v<n>.otio, exports/<tool>-script-<id>.json
    for e in entries(&data_dir.join("exports")) {
        let name = e.file_name().to_string_lossy().to_string();
        let owner = name
            .find("script-")
            .and_then(|i| id_after(&name[i..], "script-"))
            .map(Owner::Script)
            .unwrap_or(Owner::None);
        items.push(Item { kind: Kind::Exports, bytes: size_of(&e.path()), path: e.path(), owner });
    }

    for e in entries(&data_dir.join("tmp")) {
        items.push(Item { kind: Kind::Temp, bytes: size_of(&e.path()), path: e.path(), owner: Owner::None });
    }
    items
}

/// Which projects each owner belongs to, read from the library.
struct Owners {
    /// hex → (video id, projects watching it). Videos no project watches have an empty set.
    videos: HashMap<String, (i64, HashSet<i64>)>,
    scripts: HashMap<i64, i64>,
    sessions: HashMap<i64, i64>,
}

impl Owners {
    fn load(db: &Db) -> Result<Owners, Error> {
        let mut videos: HashMap<String, (i64, HashSet<i64>)> = HashMap::new();
        {
            let mut st = db.conn.prepare("SELECT id, content_hash FROM videos")?;
            let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            for row in rows {
                let (id, hash) = row?;
                videos.insert(hash_key(&hash).to_string(), (id, HashSet::new()));
            }
        }
        let by_id: HashMap<i64, String> = videos.iter().map(|(k, (id, _))| (*id, k.clone())).collect();
        {
            let mut st = db.conn.prepare(
                "SELECT DISTINCT vf.video_id, pf.project_id FROM video_files vf
                   JOIN project_folders pf ON pf.folder_id = vf.folder_id",
            )?;
            let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?;
            for row in rows {
                let (video_id, project_id) = row?;
                if let Some(entry) = by_id.get(&video_id).and_then(|k| videos.get_mut(k)) {
                    entry.1.insert(project_id);
                }
            }
        }
        let pairs = |sql: &str| -> Result<HashMap<i64, i64>, Error> {
            let mut st = db.conn.prepare(sql)?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            Ok(rows.collect::<Result<_, _>>()?)
        };
        Ok(Owners {
            videos,
            scripts: pairs("SELECT id, project_id FROM scripts")?,
            sessions: pairs("SELECT id, project_id FROM chat_sessions")?,
        })
    }

    /// The projects an item belongs to; empty when nothing uses it.
    fn projects(&self, owner: &Owner) -> Vec<i64> {
        match owner {
            Owner::Video(key) => self.videos.get(key).map(|(_, p)| p.iter().copied().collect()).unwrap_or_default(),
            Owner::Script(id) => self.scripts.get(id).map(|p| vec![*p]).unwrap_or_default(),
            Owner::Session(id) => self.sessions.get(id).map(|p| vec![*p]).unwrap_or_default(),
            Owner::None => Vec::new(),
        }
    }
}

/// How much space GhostReel uses, overall and per project.
pub fn report(db: &Db, data_dir: &Path) -> Result<Report, Error> {
    let owners = Owners::load(db)?;
    let items = inventory(data_dir);

    let mut usage = Usage::default();
    let mut unused = Usage::default();
    let mut per_project: HashMap<i64, (Usage, u64)> = HashMap::new();
    for item in &items {
        usage.add(item.kind, item.bytes);
        let projects = owners.projects(&item.owner);
        if projects.is_empty() {
            unused.add(item.kind, item.bytes);
        }
        for p in &projects {
            let entry = per_project.entry(*p).or_default();
            entry.0.add(item.kind, item.bytes);
            if projects.len() > 1 {
                entry.1 += item.bytes;
            }
        }
    }

    let mut project_videos: HashMap<i64, i64> = HashMap::new();
    for (_, projects) in owners.videos.values() {
        for p in projects {
            *project_videos.entry(*p).or_default() += 1;
        }
    }
    let projects = db
        .projects()?
        .into_iter()
        .map(|p| {
            let (usage, shared) = per_project.get(&p.id).copied().unwrap_or_default();
            ProjectUsage {
                project_id: p.id,
                videos: project_videos.get(&p.id).copied().unwrap_or(0),
                name: p.name,
                usage,
                shared,
            }
        })
        .collect();

    let total = size_of(data_dir);
    let models = size_of(&data_dir.join("models"));
    let database = entries(data_dir)
        .iter()
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.ends_with(".db") || name.ends_with(".db-wal") || name.ends_with(".db-shm")
        })
        .map(|e| size_of(&e.path()))
        .sum();
    Ok(Report {
        data_dir: data_dir.to_path_buf(),
        total,
        other: total.saturating_sub(usage.total() + models + database),
        usage,
        models,
        database,
        unused,
        unused_videos: owners.videos.values().filter(|(_, p)| p.is_empty()).count() as i64,
        projects,
    })
}

fn delete(item: &Item, stats: &mut CleanStats) {
    let removed =
        if item.path.is_dir() { std::fs::remove_dir_all(&item.path) } else { std::fs::remove_file(&item.path) };
    if removed.is_ok() {
        stats.files_deleted += 1;
        stats.bytes_freed += item.bytes;
    }
}

/// Delete the files of `kinds` in `scope`. Video files are never touched.
///
/// [`Scope::Unused`] ignores `kinds` and removes everything nothing uses, including the index
/// of videos no project watches.
pub fn clean(db: &mut Db, data_dir: &Path, scope: Scope, kinds: &[Kind]) -> Result<CleanStats, Error> {
    let owners = Owners::load(db)?;
    let mut stats = CleanStats::default();
    for item in inventory(data_dir) {
        let projects = owners.projects(&item.owner);
        let selected = match scope {
            Scope::All => kinds.contains(&item.kind),
            Scope::Project(id) => kinds.contains(&item.kind) && projects.contains(&id),
            Scope::Unused => projects.is_empty(),
        };
        if selected {
            delete(&item, &mut stats);
        }
    }
    if scope == Scope::Unused {
        for (video_id, projects) in owners.videos.values() {
            if projects.is_empty() {
                // Cascades to video_files, jobs, transcripts, frames, chunks and vectors.
                stats.videos += db.conn.execute("DELETE FROM videos WHERE id = ?1", [video_id])? as i64;
            }
        }
    }
    Ok(stats)
}

/// Delete the files of one project for [`Db::purge_project_data`]: everything of its scripts and
/// chats, and the caches of the given footage (by content hash). Exports are kept.
pub(crate) fn delete_project_files(
    db: &Db,
    data_dir: &Path,
    project_id: i64,
    content_hashes: &[String],
) -> Result<CleanStats, Error> {
    let owners = Owners::load(db)?;
    let keys: HashSet<&str> = content_hashes.iter().map(|h| hash_key(h)).collect();
    let mut stats = CleanStats::default();
    for item in inventory(data_dir) {
        let selected = match &item.owner {
            Owner::Video(key) => keys.contains(key.as_str()),
            Owner::Script(_) | Owner::Session(_) => {
                item.kind != Kind::Exports && owners.projects(&item.owner).contains(&project_id)
            }
            Owner::None => false,
        };
        if selected {
            delete(&item, &mut stats);
        }
    }
    Ok(stats)
}

/// Rebuild the database file to hand back the space deleted rows left behind. Returns the bytes
/// saved.
pub fn compact(db: &Db, db_file: &Path) -> Result<u64, Error> {
    let wal = |p: &Path| PathBuf::from(format!("{}-wal", p.display()));
    let before = size_of(db_file) + size_of(&wal(db_file));
    db.conn.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")?;
    let after = size_of(db_file) + size_of(&wal(db_file));
    Ok(before.saturating_sub(after))
}

#[cfg(test)]
mod tests {
    use rusqlite::params;

    use super::*;
    use crate::projects::NewProject;

    fn write(path: &Path, bytes: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; bytes]).unwrap();
    }

    /// Projects A and B share folder `b`; A also has `a`. Video 1 (aa11) is A's only, video 2
    /// (bb22) is shared, and video 3 (cc33) is in no project.
    fn fixture() -> (tempfile::TempDir, PathBuf, Db, i64, i64) {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("data");
        let mut db = Db::open_in_memory().unwrap();
        let a = db.create_project(&NewProject::named("A")).unwrap().id;
        let b = db.create_project(&NewProject::named("B")).unwrap().id;
        std::fs::create_dir_all(tmp.path().join("a")).unwrap();
        std::fs::create_dir_all(tmp.path().join("b")).unwrap();
        let fa = db.add_folder(a, &tmp.path().join("a"), true).unwrap().id;
        let fb = db.add_folder(a, &tmp.path().join("b"), true).unwrap().id;
        db.add_folder(b, &tmp.path().join("b"), true).unwrap();
        for (id, hash, folder) in [(1, "b3e:aa11", Some(fa)), (2, "b3e:bb22", Some(fb)), (3, "b3e:cc33", None)] {
            db.conn
                .execute("INSERT INTO videos(id, content_hash, size) VALUES (?1, ?2, 1)", params![id, hash])
                .unwrap();
            if let Some(folder) = folder {
                db.conn
                    .execute(
                        "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                         VALUES (?1, ?2, ?3, 1, 0, 0)",
                        params![id, folder, format!("{id}.mp4")],
                    )
                    .unwrap();
            }
            let hex = hash_key(hash);
            write(&data.join(crate::index::frames_rel_dir(hash)).join("t000000000.jpg"), 100);
            write(&data.join(format!("proxies/b3e-{hex}_0_1000_25_1_960x540_r0.mp4")), 1000);
            write(&data.join(format!("proxies/{}/{hex}-playback-v2-768.mp4", &hex[..2])), 2000);
        }
        db.conn
            .execute(
                "INSERT INTO scripts(id, project_id, title, version, script_json, created_at) VALUES (7, ?1, 't', 1, '{}', 0)",
                [a],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO chat_sessions(id, project_id, title, created_at, updated_at) VALUES (5, ?1, 'c', 0, 0)",
                [b],
            )
            .unwrap();
        write(&data.join("previews/script_7_v1.mp4"), 300);
        write(&data.join("previews/script_99_v1.mp4"), 30); // deleted script
        write(&data.join("chat_images/chat_5_1700000000_0.png"), 40);
        write(&data.join("exports/script-7-v1.otio"), 50);
        write(&data.join("proxies/aa/aa11.mp4"), 7); // pre-0.9 playback proxy
        write(&data.join("tmp/scratch.txt"), 5);
        write(&data.join("models/m.gguf"), 10_000);
        (tmp, data, db, a, b)
    }

    #[test]
    fn report_attributes_files_to_projects() {
        let (_tmp, data, db, a, b) = fixture();
        let r = report(&db, &data).unwrap();
        assert_eq!(r.usage.proxies, 3 * 3000 + 7);
        assert_eq!(r.usage.keyframes, 300);
        assert_eq!(r.models, 10_000);
        assert_eq!(r.unused_videos, 1);
        assert_eq!(r.unused, Usage { proxies: 3007, keyframes: 100, previews: 30, temp: 5, ..Default::default() });

        let pa = r.projects.iter().find(|p| p.project_id == a).unwrap();
        assert_eq!(pa.videos, 2);
        assert_eq!(pa.usage, Usage { proxies: 6000, keyframes: 200, previews: 300, exports: 50, ..Default::default() });
        assert_eq!(pa.shared, 3100, "video 2's proxies and keyframes");
        let pb = r.projects.iter().find(|p| p.project_id == b).unwrap();
        assert_eq!(pb.usage, Usage { proxies: 3000, keyframes: 100, chat_images: 40, ..Default::default() });
    }

    #[test]
    fn clean_project_caches_and_unused() {
        let (_tmp, data, mut db, a, b) = fixture();
        let s = clean(&mut db, &data, Scope::Project(b), &[Kind::Proxies]).unwrap();
        assert_eq!(s.bytes_freed, 3000, "only the shared video's proxies");
        assert!(data.join("proxies/b3e-aa11_0_1000_25_1_960x540_r0.mp4").is_file(), "A's own proxy is kept");
        assert!(data.join(crate::index::frames_rel_dir("b3e:bb22")).is_dir(), "keyframes weren't asked for");

        let s = clean(&mut db, &data, Scope::Unused, &[]).unwrap();
        assert_eq!(s.videos, 1);
        assert_eq!(s.bytes_freed, 3007 + 100 + 30 + 5);
        assert!(!data.join("proxies/aa/aa11.mp4").exists(), "legacy proxy is unused");
        assert!(!data.join("previews/script_99_v1.mp4").exists());
        assert!(data.join("previews/script_7_v1.mp4").is_file());
        let n: i64 = db.conn.query_row("SELECT COUNT(*) FROM videos", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 2);

        let s = clean(&mut db, &data, Scope::All, &[Kind::Keyframes]).unwrap();
        assert_eq!(s.bytes_freed, 200);
        let r = report(&db, &data).unwrap();
        assert_eq!(r.usage.keyframes, 0);
        assert_eq!(r.projects.iter().find(|p| p.project_id == a).unwrap().usage.previews, 300);
    }

    #[test]
    fn compact_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("g.db");
        let db = Db::open(&file).unwrap();
        compact(&db, &file).unwrap();
    }
}
