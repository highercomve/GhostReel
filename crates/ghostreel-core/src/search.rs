//! Hybrid search (plan D6, §5): FTS5 keyword ranking + sqlite-vec nearest neighbours, fused with
//! Reciprocal Rank Fusion, scoped to a project and grouped into moments.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Serialize;

use crate::Error;
use crate::db::Db;
use crate::embed::{Embedder, query_text, to_blob};

const CANDIDATES: usize = 200;
const RRF_K: f64 = 60.0;
/// Hits of the same video closer than this merge into one moment.
const MERGE_GAP_S: f64 = 10.0;

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub project_id: Option<i64>,
    pub limit: usize,
    /// Restrict to chunk kinds (`moment`, `transcript`, `frame`).
    pub kinds: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub video_id: i64,
    pub path: PathBuf,
    pub start_s: f64,
    pub end_s: f64,
    pub score: f64,
    /// Kinds of the chunks that matched (`moment`, `transcript`, `frame`).
    pub kinds: Vec<String>,
    /// How it matched: `keyword`, `meaning`, or both.
    pub matched_by: Vec<String>,
    pub snippet: String,
    /// Frame image for the moment (closest keyframe).
    pub frame: Option<PathBuf>,
}

/// Words that match almost every chunk and only add noise to keyword ranking (English + Spanish).
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "we", "what", "when", "where", "which", "who", "with", "you",
    "de", "del", "el", "en", "es", "la", "las", "lo", "los", "un", "una", "y", "o", "que", "con", "por", "para", "se",
    "al", "como",
];
/// A moment never grows beyond this while merging overlapping hits.
const MAX_MOMENT_S: f64 = 60.0;

/// FTS5 query from free text: meaningful words (no stopwords), prefix-matched when long enough
/// ("unbox" → "unboxing"), any word may match; BM25 ranks chunks with more/rarer matches higher.
/// Returns `None` when there is nothing searchable.
pub fn fts_query(q: &str) -> Option<String> {
    let words = search_words(q);
    let chosen = meaningful_words(&words);
    let terms: Vec<String> = chosen
        .into_iter()
        .map(|w| if w.chars().count() >= 4 { format!("\"{w}\"*") } else { format!("\"{w}\"") })
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

fn search_words(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

fn meaningful_words(words: &[String]) -> Vec<&String> {
    let meaningful: Vec<&String> = words.iter().filter(|w| !STOPWORDS.contains(&w.as_str())).collect();
    let mut chosen = if meaningful.is_empty() { words.iter().collect() } else { meaningful };
    let mut seen = HashSet::new();
    chosen.retain(|w| seen.insert(w.as_str()));
    chosen
}

/// Retrieve phrase and all-term matches before broad fallback candidates. Apply scope before
/// the limit so unrelated projects cannot crowd relevant chunks out of the candidate pool.
type KeywordCandidates = (Vec<(i64, String)>, HashMap<i64, u8>);

fn keyword_candidates(db: &Db, query: &str, opts: &SearchOptions) -> Result<KeywordCandidates, Error> {
    let Some(broad) = fts_query(query) else {
        return Ok((Vec::new(), HashMap::new()));
    };
    let words = search_words(query);
    let multi = meaningful_words(&words).len() > 1;
    let mut queries = Vec::new();
    if multi {
        queries.push((format!("\"{}\"", words.join(" ")), 3));
        queries.push((broad.replace(" OR ", " AND "), 2));
    }
    queries.push((broad, if multi { 1 } else { 0 }));
    let kinds =
        opts.kinds.as_ref().map(serde_json::to_string).transpose().map_err(|e| Error::Invalid(e.to_string()))?;
    let mut st = db.conn.prepare(
        "SELECT chunks_fts.rowid, snippet(chunks_fts, 0, '[', ']', '…', 14)
         FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
         WHERE chunks_fts MATCH ?1
           AND (?3 IS NULL OR c.kind IN (SELECT value FROM json_each(?3)))
           AND EXISTS (
             SELECT 1 FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
             WHERE vf.video_id = c.video_id AND (?4 IS NULL OR pf.project_id = ?4)
               AND NOT EXISTS (SELECT 1 FROM project_exclusions x
                               WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id))
         ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?2",
    )?;
    let mut candidates = Vec::new();
    let mut quality = HashMap::new();
    for (fts, tier) in queries {
        let rows = st.query_map(params![fts, CANDIDATES as i64, kinds, opts.project_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (id, snippet) = row?;
            if let std::collections::hash_map::Entry::Vacant(e) = quality.entry(id) {
                e.insert(tier);
                candidates.push((id, snippet));
            }
        }
    }
    Ok((candidates, quality))
}

struct ChunkInfo {
    video_id: i64,
    kind: String,
    start_s: f64,
    end_s: f64,
    text: String,
    frame_id: Option<i64>,
}

/// Search with an optional embedder (convenience for single-threaded callers like the CLI).
pub async fn search(
    db: &Db,
    data_dir: &Path,
    query: &str,
    embedder: Option<&mut Embedder>,
    opts: &SearchOptions,
) -> Result<Vec<Hit>, Error> {
    let vector = match embedder {
        Some(e) => Some(query_vector(e, query).await?),
        None => None,
    };
    search_with_vector(db, data_dir, query, vector.as_deref(), opts)
}

/// Embed a search query (embeddinggemma query prompt).
pub async fn query_vector(embedder: &mut Embedder, query: &str) -> Result<Vec<f32>, Error> {
    Ok(embedder.embed(&[query_text(query)]).await?.remove(0))
}

/// Rank and group results; `vector` is the query embedding (keyword-only when `None`). Synchronous so
/// callers can embed first and keep the database handle off async boundaries.
pub fn search_with_vector(
    db: &Db,
    data_dir: &Path,
    query: &str,
    vector: Option<&[f32]>,
    opts: &SearchOptions,
) -> Result<Vec<Hit>, Error> {
    let limit = if opts.limit == 0 { 20 } else { opts.limit };
    let allowed: HashSet<i64> = {
        let mut st = db.conn.prepare(
            "SELECT DISTINCT vf.video_id FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
              WHERE (?1 IS NULL OR pf.project_id = ?1)
                AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)",
        )?;
        st.query_map([opts.project_id], |r| r.get(0))?.collect::<Result<_, _>>()?
    };

    // Ranked candidate chunk ids from each retriever.
    let (keyword, quality) = keyword_candidates(db, query, opts)?;
    let mut semantic: Vec<i64> = Vec::new();
    if let Some(v) = vector {
        let mut st =
            db.conn.prepare("SELECT rowid FROM chunks_vec WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance")?;
        semantic = st.query_map(params![to_blob(v), CANDIDATES as i64], |r| r.get(0))?.collect::<Result<_, _>>()?;
    }

    let mut ids: Vec<i64> = keyword.iter().map(|(id, _)| *id).chain(semantic.iter().copied()).collect();
    ids.sort_unstable();
    ids.dedup();
    let info: HashMap<i64, ChunkInfo> = {
        let mut map = HashMap::new();
        let mut st =
            db.conn.prepare("SELECT video_id, kind, start_s, end_s, text, frame_id FROM chunks WHERE id = ?1")?;
        for id in &ids {
            if let Ok(c) = st.query_row([id], |r| {
                Ok(ChunkInfo {
                    video_id: r.get(0)?,
                    kind: r.get(1)?,
                    start_s: r.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                    end_s: r.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                    text: r.get(4)?,
                    frame_id: r.get(5)?,
                })
            }) {
                map.insert(*id, c);
            }
        }
        map
    };
    let keep = |id: &i64| {
        info.get(id).is_some_and(|c| {
            allowed.contains(&c.video_id) && opts.kinds.as_ref().is_none_or(|k| k.iter().any(|x| x == &c.kind))
        })
    };

    // RRF over the filtered rankings.
    let mut scores: HashMap<i64, (f64, Vec<&'static str>)> = HashMap::new();
    let snippets: HashMap<i64, String> = keyword.iter().cloned().collect();
    for (rank, (id, _)) in keyword.iter().filter(|(id, _)| keep(id)).enumerate() {
        let e = scores.entry(*id).or_default();
        e.0 += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.1.push("keyword");
    }
    for (rank, id) in semantic.iter().filter(|id| keep(id)).enumerate() {
        let e = scores.entry(*id).or_default();
        e.0 += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.1.push("meaning");
    }
    let mut ranked: Vec<(i64, f64, Vec<&'static str>)> = scores.into_iter().map(|(id, (s, m))| (id, s, m)).collect();
    let tier = |id: i64| quality.get(&id).copied().unwrap_or(0);
    ranked.sort_by(|a, b| tier(b.0).cmp(&tier(a.0)).then_with(|| b.1.total_cmp(&a.1)).then_with(|| a.0.cmp(&b.0)));

    // Group into moments.
    let mut hits: Vec<(Hit, Option<i64>, u8)> = Vec::new();
    for (id, score, matched) in ranked {
        let c = &info[&id];
        let snippet = snippets.get(&id).cloned().unwrap_or_else(|| short(&c.text));
        if let Some((h, _, match_tier)) = hits.iter_mut().find(|(h, _, _)| {
            h.video_id == c.video_id && c.start_s <= h.end_s + MERGE_GAP_S && c.end_s + MERGE_GAP_S >= h.start_s
        }) {
            *match_tier = (*match_tier).max(tier(id));
            h.score += score * 0.5; // corroborating evidence, diminished
            // Grow the moment only while it stays short enough to be "a moment".
            let (start, end) = (h.start_s.min(c.start_s), h.end_s.max(c.end_s));
            if end - start <= MAX_MOMENT_S {
                h.start_s = start;
                h.end_s = end;
            }
            for m in matched {
                if !h.matched_by.iter().any(|x| x == m) {
                    h.matched_by.push(m.to_string());
                }
            }
            if !h.kinds.contains(&c.kind) {
                h.kinds.push(c.kind.clone());
            }
            continue;
        }
        if hits.len() >= limit * 3 {
            continue;
        }
        hits.push((
            Hit {
                video_id: c.video_id,
                path: PathBuf::new(),
                start_s: c.start_s,
                end_s: c.end_s,
                score,
                kinds: vec![c.kind.clone()],
                matched_by: matched.iter().map(|m| m.to_string()).collect(),
                snippet,
                frame: None,
            },
            c.frame_id,
            tier(id),
        ));
    }
    hits.sort_by(|a, b| {
        b.2.cmp(&a.2).then_with(|| b.0.score.total_cmp(&a.0.score)).then_with(|| a.0.video_id.cmp(&b.0.video_id))
    });
    hits.truncate(limit);

    // Paths and frames.
    let mut out = Vec::with_capacity(hits.len());
    for (mut h, frame_id, _) in hits {
        h.path = db
            .conn
            .query_row(
                "SELECT vf.path FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
                  WHERE vf.video_id = ?1 AND (?2 IS NULL OR pf.project_id = ?2) ORDER BY vf.id LIMIT 1",
                params![h.video_id, opts.project_id],
                |r| r.get::<_, String>(0),
            )
            .map(PathBuf::from)
            .unwrap_or_default();
        let rel: Option<String> = match frame_id {
            Some(fid) => db.conn.query_row("SELECT thumb_path FROM frames WHERE id = ?1", [fid], |r| r.get(0)).ok(),
            None => db
                .conn
                .query_row(
                    "SELECT thumb_path FROM frames WHERE video_id = ?1 ORDER BY ABS(t_s - ?2) LIMIT 1",
                    params![h.video_id, h.start_s],
                    |r| r.get(0),
                )
                .ok(),
        };
        h.frame = rel.map(|r| data_dir.join(r));
        out.push(h);
    }
    Ok(out)
}

fn short(text: &str) -> String {
    let flat = text.replace('\n', " · ");
    let s: String = flat.chars().take(220).collect();
    if flat.chars().count() > 220 { format!("{s}…") } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn fts_query_building() {
        assert_eq!(fts_query("CM5 unbox!").as_deref(), Some("\"cm5\" OR \"unbox\"*"));
        assert_eq!(fts_query("  ?!  "), None);
        assert_eq!(
            fts_query("the temperature graph in the browser").as_deref(),
            Some("\"temperature\"* OR \"graph\"* OR \"browser\"*")
        );
        assert_eq!(fts_query("the who").as_deref(), Some("\"the\" OR \"who\""), "all-stopword queries still search");
        assert_eq!(fts_query("la red inalámbrica").as_deref(), Some("\"red\" OR \"inalámbrica\"*"));
    }

    #[tokio::test]
    async fn keyword_search_scoped_grouped() {
        let tmp = tempfile::tempdir().unwrap();
        let (a_dir, b_dir) = (tmp.path().join("a"), tmp.path().join("b"));
        std::fs::create_dir_all(&a_dir).unwrap();
        std::fs::create_dir_all(&b_dir).unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let pa = db.create_project(&NewProject::named("A")).unwrap();
        let pb = db.create_project(&NewProject::named("B")).unwrap();
        let fa = db.add_folder(pa.id, &a_dir, true).unwrap();
        let fb = db.add_folder(pb.id, &b_dir, true).unwrap();
        let c = &db.conn;
        c.execute_batch(&format!(
            "INSERT INTO videos(id, content_hash, size) VALUES (1, 'h1', 1), (2, 'h2', 1);
             INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, {fa}, '{a}/one.mp4', 1, 0, 0), (2, {fb}, '{b}/two.mp4', 1, 0, 0);
             INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES
                 (1, 'transcript', 0, 30, 'we unbox the compute module'),
                 (1, 'moment', 5, 20, 'Hands unbox a green board. On screen: CM5'),
                 (1, 'transcript', 200, 230, 'flash the image'),
                 (2, 'transcript', 0, 30, 'unbox a phone');",
            fa = fa.id,
            fb = fb.id,
            a = a_dir.display(),
            b = b_dir.display()
        ))
        .unwrap();

        let opts = SearchOptions { project_id: Some(pa.id), limit: 10, kinds: None };
        let hits = search(&db, tmp.path(), "unbox", None, &opts).await.unwrap();
        assert_eq!(hits.len(), 1, "two overlapping chunks of video 1 merge; video 2 is another project");
        assert_eq!((hits[0].start_s, hits[0].end_s), (0.0, 30.0));
        assert!(hits[0].kinds.contains(&"moment".to_string()) && hits[0].kinds.contains(&"transcript".to_string()));
        assert!(hits[0].snippet.contains("[unbox]"));
        assert!(hits[0].path.ends_with("one.mp4"));

        let all =
            search(&db, tmp.path(), "unbox", None, &SearchOptions { limit: 10, ..Default::default() }).await.unwrap();
        assert_eq!(all.len(), 2);
        let kinds = SearchOptions { project_id: Some(pa.id), limit: 10, kinds: Some(vec!["transcript".into()]) };
        let hits = search(&db, tmp.path(), "flash", None, &kinds).await.unwrap();
        assert_eq!(hits[0].start_s, 200.0);
        assert!(search(&db, tmp.path(), "zebra", None, &opts).await.unwrap().is_empty());
    }
    #[test]
    fn phrase_and_all_terms_rank_above_partial_matches_after_grouping() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Search")).unwrap();
        let folder = db.add_folder(project.id, tmp.path(), true).unwrap();
        for (id, text) in [
            (1, "A woman stands near school lockers."),
            (2, "A boy wears a black jacket."),
            (3, "A woman is walking, wearing black clothes."),
            (4, "A close-up of a Black woman meditating outdoors."),
        ] {
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size) VALUES (?1, ?2, 1)",
                    params![id, format!("hash-{id}")],
                )
                .unwrap();
            db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 1, 0, 0)",
                params![id, folder.id, format!("{id}.mp4")]).unwrap();
            // Repeated partial evidence must never overpower the full phrase when grouped.
            for start in 0..if id == 1 { 20 } else { 1 } {
                db.conn
                    .execute(
                        "INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES (?1, 'frame', ?2, ?2 + 1, ?3)",
                        params![id, start, text],
                    )
                    .unwrap();
            }
        }
        let opts = SearchOptions { project_id: Some(project.id), limit: 10, kinds: None };
        let hits = search_with_vector(&db, tmp.path(), "black woman", None, &opts).unwrap();
        assert_eq!(hits.iter().map(|h| h.video_id).take(2).collect::<Vec<_>>(), [4, 3]);
        assert!(hits[0].snippet.contains("[Black woman]"));
        assert_eq!(hits.len(), 4, "partial matches remain available as fallback results");
        let single = search_with_vector(&db, tmp.path(), "woman", None, &opts).unwrap();
        assert_eq!(single.len(), 3);
        // Semantic corroboration of partial matches must not push them above the phrase.
        let chunk_id: i64 =
            db.conn.query_row("SELECT id FROM chunks WHERE video_id = 1 LIMIT 1", [], |r| r.get(0)).unwrap();
        let vector = vec![0.0f32; 768];
        db.conn
            .execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)", params![chunk_id, to_blob(&vector)])
            .unwrap();
        let hybrid = search_with_vector(&db, tmp.path(), "black woman", Some(&vector), &opts).unwrap();
        assert_eq!(hybrid[0].video_id, 4);
    }

    #[test]
    fn scoped_phrase_candidates_survive_broad_candidate_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Search")).unwrap();
        let folder = db.add_folder(project.id, tmp.path(), true).unwrap();
        db.conn.execute("INSERT INTO videos(id, content_hash, size) VALUES (1, 'h', 1)", []).unwrap();
        db.conn.execute("INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (1, ?1, 'one.mp4', 1, 0, 0)", [folder.id]).unwrap();
        for i in 0..CANDIDATES + 50 {
            db.conn.execute("INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES (1, 'transcript', ?1, ?1 + 1, 'woman woman woman')", [i as i64 * 100]).unwrap();
        }
        db.conn.execute("INSERT INTO chunks(video_id, kind, start_s, end_s, text) VALUES (1, 'frame', 0, 1, 'A Black woman sits outdoors')", []).unwrap();
        let opts = SearchOptions { project_id: Some(project.id), limit: 1, kinds: Some(vec!["frame".into()]) };
        let hits = search_with_vector(&db, tmp.path(), "black woman", None, &opts).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.contains("[Black woman]"));
        assert_eq!(fts_query("black black woman").as_deref(), Some("\"black\"* OR \"woman\"*"));
    }
}
