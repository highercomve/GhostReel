//! "Export for Premiere": one `.zip` a co-worker on macOS or Windows extracts and imports.
//!
//! ```text
//! <title>/
//!   <title>.xml     FCP7 XML (Premiere: File > Import)
//!   <title>.otio    OpenTimelineIO (Premiere 26+)
//!   media/          the footage the cut uses
//!   preview.mp4     the latest render of this version, when there is one (reference only)
//!   README.txt      how to open it, relinking included
//! ```
//!
//! Media comes in two modes. [`MediaMode::Trimmed`] re-encodes, per source video, the merged
//! stretches the cut uses (clips and beds) plus handles, to a short-GOP H.264 + PCM `.mov`, and
//! builds the timeline from a copy of the script whose clips point at those files with their
//! in/out shifted by each stretch's start. [`MediaMode::Full`] copies the used source files as
//! they are.
//!
//! Where the co-worker extracts the zip is unknown, so the OTIO references media by the relative
//! URL `media/<name>`, and the XML by the `file://localhost/` path the media has when the zip is
//! extracted next to where it was written. Anywhere else, Premiere shows the media offline once
//! and relinks the whole `media/` folder from the first file located (README.txt says how).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Error;
use crate::cancel::CancelFlag;
use crate::db::Db;
use crate::otio::{self, ResolvedMedia};
use crate::script::{self, Fps, Script};

/// The `exports.format` a package is recorded under.
pub const FORMAT: &str = "premiere_package";
/// Handles added on each side of every used stretch in [`MediaMode::Trimmed`], by default.
pub const DEFAULT_HANDLES_S: f64 = 2.0;
/// Synthetic media keys for trimmed stretches start here, far from any real video id.
const RANGE_KEY_BASE: i64 = 1_000_000_000_000;

/// What goes into `media/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaMode {
    /// The stretches the cut uses, plus handles, re-encoded (frame-accurate cut points).
    #[default]
    Trimmed,
    /// Every used source file, copied untouched.
    Full,
}

impl MediaMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trimmed => "trimmed",
            Self::Full => "full",
        }
    }
}

impl FromStr for MediaMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "trimmed" | "used" | "ranges" | "used_ranges" => Ok(Self::Trimmed),
            "full" | "whole" | "whole_files" | "copy" => Ok(Self::Full),
            other => Err(Error::Invalid(format!("unknown media mode '{other}' (expected 'trimmed' or 'full')"))),
        }
    }
}

/// How to build a package.
#[derive(Debug, Clone, Default)]
pub struct PackageOptions {
    pub media: MediaMode,
    /// Seconds of extra footage on each side of a used stretch (trimmed only); `None`: 2 s.
    pub handles_s: Option<f64>,
    /// The app's data dir, to find this version's default preview render; without it only a
    /// recorded render is found.
    pub data_dir: Option<PathBuf>,
    pub cancel: Option<CancelFlag>,
}

impl PackageOptions {
    fn handles(&self) -> f64 {
        self.handles_s.filter(|h| h.is_finite()).unwrap_or(DEFAULT_HANDLES_S).clamp(0.0, 60.0)
    }
}

/// One file in the package's `media/` folder.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PackagedMedia {
    /// Its name inside `media/`.
    pub name: String,
    pub video_id: i64,
    /// The source file's own name (not its path).
    pub source_name: String,
    /// Where in the source file it starts (0 for a whole file).
    pub source_start_s: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PackageResult {
    pub export_id: i64,
    pub path: PathBuf,
    pub format: &'static str,
    pub media_mode: MediaMode,
    pub media: Vec<PackagedMedia>,
    /// Whether a preview render went in.
    pub preview: bool,
    pub bytes: u64,
}

/// A used stretch of one source video (trimmed mode).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Range {
    start: f64,
    end: f64,
}

/// What one file in `media/` is made from.
struct Planned {
    key: i64,
    name: String,
    source: ResolvedMedia,
    range: Option<Range>,
}

/// Build the package for a script version at `out_zip` (overwritten), and record it.
pub fn export_package(
    db: &Db,
    ffmpeg: &Path,
    script_id: i64,
    out_zip: &Path,
    opts: &PackageOptions,
    progress: &mut dyn FnMut(&'static str, f64, &str),
) -> Result<PackageResult, Error> {
    let stored = script::load(db, script_id)?;
    let project = db.project(stored.project_id)?;
    let project_fps = Fps::new(project.fps_num, project.fps_den);
    let media = otio::resolve_media_for_script(db, stored.project_id, &stored.script, project_fps)?;
    for m in media.values() {
        if !m.path.is_file() {
            return Err(Error::NotFound(format!("the footage for video #{} at {}", m.video_id, m.path.display())));
        }
    }

    let title = safe_name(&stored.script.title, "cut");
    let out_dir = match out_zip.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    std::fs::create_dir_all(&out_dir).map_err(|e| Error::Io(out_dir.clone(), e))?;
    let out_dir = std::fs::canonicalize(&out_dir).map_err(|e| Error::Io(out_dir.clone(), e))?;
    // Where the media sits once the zip is extracted next to where it was written.
    let extracted_media = out_dir.join(&title).join("media");

    // Dropped (removed) on every early return: failure and cancel leave nothing behind.
    let work = tempfile::Builder::new()
        .prefix(".ghostreel-package-")
        .tempdir_in(&out_dir)
        .map_err(|e| Error::Io(out_dir.clone(), e))?;
    let staged_media = work.path().join("media");
    std::fs::create_dir_all(&staged_media).map_err(|e| Error::Io(staged_media.clone(), e))?;

    let handles = opts.handles();
    let (plan, cut) = match opts.media {
        MediaMode::Full => plan_full(&stored.script, &media),
        MediaMode::Trimmed => plan_trimmed(&stored.script, &media, handles),
    };

    // Trimmed: encode every stretch first (most of the time goes here), then zip.
    let encode_share = if opts.media == MediaMode::Trimmed { 0.8 } else { 0.0 };
    let ffprobe = crate::doctor::locate("ffprobe");
    let mut timeline_media: HashMap<i64, ResolvedMedia> = HashMap::new();
    let mut packaged = Vec::new();
    let mut zip_sources: Vec<(String, PathBuf)> = Vec::new();
    for (i, p) in plan.iter().enumerate() {
        if crate::cancel::is_set(opts.cancel.as_ref()) {
            return Err(Error::Cancelled);
        }
        let (file, duration_s, start_s) = match p.range {
            Some(r) => {
                progress(
                    "encode",
                    encode_share * i as f64 / plan.len().max(1) as f64,
                    &format!("Encoding {}/{}: {}", i + 1, plan.len(), p.name),
                );
                let dest = staged_media.join(&p.name);
                encode_range(ffmpeg, &p.source, r, &dest, opts.cancel.as_ref())?;
                let planned = r.end - r.start;
                // The file never claims more than it holds: an out point past its end makes
                // Premiere refuse the whole import.
                let measured = ffprobe.as_deref().and_then(|fp| probe_duration(fp, &dest));
                let dur = measured.map_or(planned, |m| m.min(planned));
                (dest, dur, r.start)
            }
            None => (p.source.path.clone(), p.source.duration_s, 0.0),
        };
        timeline_media
            .insert(p.key, ResolvedMedia { path: extracted_media.join(&p.name), duration_s, ..p.source.clone() });
        packaged.push(PackagedMedia {
            name: p.name.clone(),
            video_id: p.source.video_id,
            source_name: file_name(&p.source.path),
            source_start_s: start_s,
            duration_s,
        });
        zip_sources.push((p.name.clone(), file));
    }

    // Built like a plain export (source sizes and framing plans from the index), over the files
    // in the package: each one starts `source_start_s` into the video it was cut from.
    let starts: HashMap<i64, f64> = plan.iter().zip(&packaged).map(|(p, m)| (p.key, m.source_start_s)).collect();
    let mut timeline = otio::build_timeline_from(db, stored.project_id, &cut, &timeline_media, &|key| {
        starts.get(&key).copied().unwrap_or(0.0)
    })?;
    let xml = crate::fcpxml::from_otio(&timeline)?;
    relative_media_urls(&mut timeline, &extracted_media, &packaged);
    let otio_text = serde_json::to_string_pretty(&timeline)
        .map_err(|e| Error::Export(format!("failed to serialize timeline JSON: {e}")))?;

    let preview = match &opts.data_dir {
        Some(dir) => crate::preview::find_preview(db, dir, script_id).ok().flatten(),
        None => crate::render_review::latest_render(db, script_id),
    };
    let readme = readme(&title, opts.media, handles, &packaged, preview.is_some());

    progress("zip", encode_share, "Zipping…");
    let part = work.path().join("package.zip.part");
    let total: u64 = zip_sources
        .iter()
        .map(|(_, path)| path)
        .chain(preview.iter())
        .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
        .sum();
    let mut zipper = Zipper::create(&part, total)?;
    let root = format!("{title}/");
    zipper.dir(&root)?;
    zipper.text(&format!("{root}{title}.xml"), xml.as_bytes())?;
    zipper.text(&format!("{root}{title}.otio"), otio_text.as_bytes())?;
    zipper.text(&format!("{root}README.txt"), readme.as_bytes())?;
    zipper.dir(&format!("{root}media/"))?;
    let mut report = |done: u64, total: u64, name: &str| {
        let f = if total > 0 { done as f64 / total as f64 } else { 1.0 };
        progress("zip", encode_share + (1.0 - encode_share) * f.min(1.0), &format!("Zipping {name}"));
    };
    for (name, path) in &zip_sources {
        zipper.file(&format!("{root}media/{name}"), path, opts.cancel.as_ref(), &mut report)?;
    }
    if let Some(p) = &preview {
        zipper.file(&format!("{root}preview.mp4"), p, opts.cancel.as_ref(), &mut report)?;
    }
    let bytes = zipper.finish()?;

    if crate::cancel::is_set(opts.cancel.as_ref()) {
        return Err(Error::Cancelled);
    }
    if out_zip.is_dir() {
        return Err(Error::Invalid(format!("{} is a folder", out_zip.display())));
    }
    std::fs::rename(&part, out_zip).map_err(|e| Error::Io(out_zip.to_path_buf(), e))?;
    drop(work);

    let export_id = crate::export::record(db, stored.id, FORMAT, out_zip)?;
    progress("zip", 1.0, "Done");
    Ok(PackageResult {
        export_id,
        path: out_zip.to_path_buf(),
        format: FORMAT,
        media_mode: opts.media,
        media: packaged,
        preview: preview.is_some(),
        bytes,
    })
}

/// The video ids a script uses, pictures and beds, in first-use order.
fn used_video_ids(script: &Script) -> Vec<i64> {
    let mut seen = HashSet::new();
    script
        .beats
        .iter()
        .flat_map(|b| b.clips.iter().map(|c| c.video_id).chain(b.bed.iter().map(|bed| bed.video_id)))
        .filter(|id| seen.insert(*id))
        .collect()
}

/// Full mode: one file per used video, the script untouched.
fn plan_full(script: &Script, media: &HashMap<i64, ResolvedMedia>) -> (Vec<Planned>, Script) {
    let mut names = UniqueNames::default();
    let plan = used_video_ids(script)
        .into_iter()
        .filter_map(|id| media.get(&id))
        .map(|m| Planned { key: m.video_id, name: names.take(&file_name(&m.path)), source: m.clone(), range: None })
        .collect();
    (plan, script.clone())
}

/// Trimmed mode: per used video, its merged stretches with handles; and a copy of the script
/// whose clips and beds point at the stretch holding them, in/out shifted by its start.
fn plan_trimmed(script: &Script, media: &HashMap<i64, ResolvedMedia>, handles: f64) -> (Vec<Planned>, Script) {
    let mut spans: BTreeMap<i64, Vec<Range>> = BTreeMap::new();
    for beat in &script.beats {
        let clips = beat.clips.iter().map(|c| (c.video_id, c.in_s, c.out_s));
        for (id, a, b) in clips.chain(beat.bed.iter().map(|bed| (bed.video_id, bed.in_s, bed.out_s))) {
            if let Some(m) = media.get(&id) {
                spans.entry(id).or_default().push(padded(a, b, handles, m.duration_s));
            }
        }
    }

    let mut names = UniqueNames::default();
    let mut plan = Vec::new();
    let mut keys: HashMap<i64, Vec<(Range, i64)>> = HashMap::new();
    // First-use order, so media/ reads in the cut's order.
    for id in used_video_ids(script) {
        let (Some(ranges), Some(m)) = (spans.remove(&id), media.get(&id)) else { continue };
        let stem = Path::new(&file_name(&m.path)).file_stem().and_then(|s| s.to_str()).unwrap_or("clip").to_string();
        for r in merge(ranges) {
            let key = RANGE_KEY_BASE + plan.len() as i64;
            let name = names.take(&format!("{stem}_{}-{}.mov", clock(r.start), clock(r.end)));
            keys.entry(id).or_default().push((r, key));
            plan.push(Planned { key, name, source: m.clone(), range: Some(r) });
        }
    }

    let locate = |id: i64, a: f64, b: f64| -> Option<(i64, f64)> {
        keys.get(&id)?
            .iter()
            .find(|(r, _)| a >= r.start - 1e-6 && b <= r.end + 1e-6)
            .or_else(|| keys.get(&id)?.iter().find(|(r, _)| a >= r.start - 1e-6 && a < r.end))
            .map(|(r, k)| (*k, r.start))
    };
    let mut cut = script.clone();
    for beat in &mut cut.beats {
        for c in &mut beat.clips {
            if let Some((key, start)) = locate(c.video_id, c.in_s, c.out_s) {
                c.video_id = key;
                c.in_s -= start;
                c.out_s -= start;
            }
        }
        if let Some(bed) = &mut beat.bed
            && let Some((key, start)) = locate(bed.video_id, bed.in_s, bed.out_s)
        {
            bed.video_id = key;
            bed.in_s -= start;
            bed.out_s -= start;
        }
    }
    (plan, cut)
}

/// `[a, b]` widened by `handles` each side, inside `[0, duration]` (no upper bound when the
/// duration is unknown).
fn padded(a: f64, b: f64, handles: f64, duration: f64) -> Range {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    let start = (a - handles).max(0.0);
    let mut end = b + handles;
    if duration > 0.0 {
        end = end.min(duration);
    }
    Range { start: start.min(end), end }
}

/// Overlapping or touching stretches as one.
fn merge(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort_by(|x, y| x.start.total_cmp(&y.start));
    let mut out: Vec<Range> = Vec::new();
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end + 1e-6 => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// `0m12s`, `1h02m03s`: a stretch's edge in a file name.
fn clock(s: f64) -> String {
    let t = s.max(0.0).round() as u64;
    let (h, m, sec) = (t / 3600, (t / 60) % 60, t % 60);
    if h > 0 { format!("{h}h{m:02}m{sec:02}s") } else { format!("{m}m{sec:02}s") }
}

/// The ffmpeg arguments that cut `r` out of `src` into an editing-friendly `.mov`: H.264 at
/// high quality with a short GOP, every audio stream as 16-bit PCM, the source frame rate.
/// Re-encoded rather than copied, so the stretch starts exactly where asked.
fn encode_args(src: &Path, fps: Fps, r: Range, dest: &Path) -> Vec<std::ffi::OsString> {
    let fps_ok = fps.num > 0 && fps.den > 0;
    let gop = if fps_ok { ((fps.num as f64 / fps.den as f64) / 2.0).round().max(1.0) as i64 } else { 12 };
    let mut args: Vec<std::ffi::OsString> = Vec::new();
    let mut push = |s: &str| args.push(s.into());
    for a in ["-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss"] {
        push(a);
    }
    push(&format!("{:.6}", r.start));
    push("-i");
    args.push(src.as_os_str().to_owned());
    let mut push = |s: &str| args.push(s.into());
    push("-t");
    push(&format!("{:.6}", (r.end - r.start).max(0.0)));
    for a in ["-map", "0:v:0?", "-map", "0:a?", "-map_metadata", "0", "-c:v", "libx264", "-preset", "medium"] {
        push(a);
    }
    for a in ["-crf", "16", "-pix_fmt", "yuv420p", "-bf", "0", "-g"] {
        push(a);
    }
    push(&gop.to_string());
    if fps_ok {
        push("-r");
        push(&format!("{}/{}", fps.num, fps.den));
    }
    for a in ["-c:a", "pcm_s16le", "-avoid_negative_ts", "make_zero", "-f", "mov"] {
        push(a);
    }
    args.push(dest.as_os_str().to_owned());
    args
}

fn encode_range(
    ffmpeg: &Path,
    src: &ResolvedMedia,
    r: Range,
    dest: &Path,
    cancel: Option<&CancelFlag>,
) -> Result<(), Error> {
    let mut cmd = crate::proc::std_command(ffmpeg);
    cmd.args(encode_args(&src.path, src.fps, r, dest));
    let out = crate::cancel::output(&mut cmd, cancel).map_err(|e| {
        if crate::cancel::is_cancel_io(&e) { Error::Cancelled } else { Error::Io(ffmpeg.to_path_buf(), e) }
    })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Export(format!(
            "ffmpeg could not cut {} ({:.2}-{:.2} s): {}",
            src.path.display(),
            r.start,
            r.end,
            err.trim().lines().last().unwrap_or("unknown error")
        )));
    }
    Ok(())
}

fn probe_duration(ffprobe: &Path, file: &Path) -> Option<f64> {
    let out = crate::proc::std_command(ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(file)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().ok().filter(|d| d.is_finite() && *d > 0.0)
}

/// Point every media reference of the OTIO at `media/<name>` (relative, legal in OTIO), and say
/// where in the source each file starts.
fn relative_media_urls(timeline: &mut Value, extracted_media: &Path, packaged: &[PackagedMedia]) {
    let norm = |s: &str| s.replace('\\', "/");
    let by_path: HashMap<String, &PackagedMedia> =
        packaged.iter().map(|m| (norm(&extracted_media.join(&m.name).to_string_lossy()), m)).collect();
    fn walk(v: &mut Value, f: &dyn Fn(&mut serde_json::Map<String, Value>)) {
        match v {
            Value::Object(map) => {
                f(map);
                for child in map.values_mut() {
                    walk(child, f);
                }
            }
            Value::Array(items) => items.iter_mut().for_each(|c| walk(c, f)),
            _ => {}
        }
    }
    walk(timeline, &|map| {
        let Some(url) = map.get("target_url").and_then(Value::as_str) else { return };
        let Some(m) = by_path.get(&norm(&crate::fcpxml::path_from_url(url))) else { return };
        map.insert("target_url".into(), json!(format!("media/{}", url_segment(&m.name))));
        let meta = map.entry("metadata").or_insert_with(|| json!({}));
        if let Value::Object(meta) = meta {
            let gr = meta.entry("ghostreel").or_insert_with(|| json!({}));
            if let Value::Object(gr) = gr {
                gr.insert("source_name".into(), json!(m.source_name));
                gr.insert("source_start_s".into(), json!(m.source_start_s));
                gr.insert("video_id".into(), json!(m.video_id));
            }
        }
    });
}

/// Percent-encode one URL path segment (unreserved characters kept).
fn url_segment(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "clip.mp4".into())
}

/// A file or folder name that is legal on Windows, macOS and Linux.
fn safe_name(s: &str, fallback: &str) -> String {
    let cleaned: String =
        s.chars().map(|c| if c.is_control() || r#"\/:*?"<>|"#.contains(c) { '_' } else { c }).collect();
    let mut name: String = cleaned.trim().trim_end_matches(['.', ' ']).chars().take(100).collect();
    name = name.trim_end_matches(['.', ' ']).to_string();
    if name.is_empty() {
        name = fallback.to_string();
    }
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit());
    if reserved { format!("_{name}") } else { name }
}

/// Safe names, unique ignoring case (Windows and macOS file systems ignore it).
#[derive(Default)]
struct UniqueNames(HashSet<String>);

impl UniqueNames {
    fn take(&mut self, wanted: &str) -> String {
        let safe = safe_name(wanted, "clip");
        let (stem, ext) = match safe.rfind('.') {
            Some(i) if i > 0 => (&safe[..i], &safe[i..]),
            _ => (safe.as_str(), ""),
        };
        let mut candidate = safe.clone();
        let mut n = 2;
        while !self.0.insert(candidate.to_lowercase()) {
            candidate = format!("{stem} ({n}){ext}");
            n += 1;
        }
        candidate
    }
}

fn readme(title: &str, mode: MediaMode, handles: f64, media: &[PackagedMedia], preview: bool) -> String {
    let mut s = String::new();
    s.push_str(&format!("{title}: an edit to continue in Adobe Premiere Pro\r\n"));
    s.push_str(&"=".repeat(title.chars().count() + 42));
    s.push_str("\r\n\r\nWhat is in this folder\r\n\r\n");
    s.push_str(&format!("  {title}.xml    the sequence, for every Premiere version (File > Import)\r\n"));
    s.push_str(&format!("  {title}.otio   the same sequence as OpenTimelineIO, for Premiere 26 and later\r\n"));
    s.push_str("  media/         the footage the sequence uses\r\n");
    if preview {
        s.push_str("  preview.mp4    how the edit looked when it was exported (for reference only)\r\n");
    }
    s.push_str("  README.txt     this file\r\n\r\n");

    let steps = |s: &mut String, extract: &str| {
        s.push_str(&format!("  1. {extract}\r\n"));
        s.push_str("     Keep the folder together: the .xml and the media/ folder belong side by side.\r\n");
        s.push_str("  2. Open Premiere Pro and open or create a project.\r\n");
        s.push_str(&format!("  3. Choose File > Import, select {title}.xml and click Import (Open).\r\n"));
        s.push_str("     A sequence with the same name appears in the Project panel.\r\n");
        s.push_str("  4. If the clips show as offline (\"Media Offline\", or a Link Media window opens):\r\n");
        s.push_str("     in the Link Media window click Locate, open the media folder in this folder,\r\n");
        s.push_str("     select the file it asks for and click OK. Leave \"Relink others automatically\"\r\n");
        s.push_str("     on: Premiere then finds every other file in the same folder by itself.\r\n");
        s.push_str("     (If no window opens, select the offline clips in the Project panel,\r\n");
        s.push_str("     right-click > Link Media..., and do the same.)\r\n");
        s.push_str("  5. Double-click the sequence to open it and carry on editing.\r\n\r\n");
    };
    s.push_str("How to open it on a Mac\r\n\r\n");
    steps(&mut s, &format!("Double-click the .zip file. Finder makes a folder named \"{title}\" next to it."));
    s.push_str("How to open it on Windows\r\n\r\n");
    steps(
        &mut s,
        &format!("Right-click the .zip file > Extract All..., then Extract. This makes a folder named \"{title}\"."),
    );
    s.push_str("Premiere Pro 26 or later can also import the .otio file the same way (File > Import).\r\n\r\n");

    s.push_str("About the media\r\n\r\n");
    match mode {
        MediaMode::Trimmed => {
            s.push_str(&format!(
                "  To keep the package small, media/ holds only the parts of each video the edit uses,\r\n  \
                 with {handles:.1} s of extra footage before and after each part for trimming. They are\r\n  \
                 high-quality H.264 copies with every audio track of the original. The sequence's\r\n  \
                 in and out points refer to these files. Where each file comes from:\r\n\r\n"
            ));
            for m in media {
                s.push_str(&format!(
                    "  {}  =  {} from {:.2} s, {:.2} s long\r\n",
                    m.name, m.source_name, m.source_start_s, m.duration_s
                ));
            }
        }
        MediaMode::Full => {
            s.push_str("  media/ holds every video the edit uses, copied as it is.\r\n");
        }
    }
    s.push_str(
        "\r\nWhat does not come across\r\n\r\n  The cuts, the clips' in and out points, the tracks, the reframing for the \
         output size,\r\n  audio levels and markers come across. Titles, captions and section cards are left as \
         gaps\r\n  or markers to rebuild in Premiere; look at preview.mp4 (when there is one) to see them.\r\n",
    );
    s
}

/// The zip being written: text deflated, media stored as is (it is compressed already).
struct Zipper {
    zip: zip::ZipWriter<std::fs::File>,
    path: PathBuf,
    total: u64,
    done: u64,
}

impl Zipper {
    fn create(path: &Path, total: u64) -> Result<Self, Error> {
        let file = std::fs::File::create(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
        Ok(Self { zip: zip::ZipWriter::new(file), path: path.to_path_buf(), total, done: 0 })
    }

    fn err(&self, e: impl std::fmt::Display) -> Error {
        Error::Export(format!("cannot write {}: {e}", self.path.display()))
    }

    fn dir(&mut self, name: &str) -> Result<(), Error> {
        let opts = zip::write::SimpleFileOptions::default().unix_permissions(0o755);
        self.zip.add_directory(name, opts).map_err(|e| self.err(e))
    }

    fn text(&mut self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        self.zip.start_file(name, opts).map_err(|e| self.err(e))?;
        self.zip.write_all(bytes).map_err(|e| self.err(e))
    }

    fn file(
        &mut self,
        name: &str,
        src: &Path,
        cancel: Option<&CancelFlag>,
        report: &mut dyn FnMut(u64, u64, &str),
    ) -> Result<(), Error> {
        let mut input = std::fs::File::open(src).map_err(|e| Error::Io(src.to_path_buf(), e))?;
        let len = input.metadata().map(|m| m.len()).unwrap_or(0);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(len >= 0xFFFF_FFFF)
            .unix_permissions(0o644);
        self.zip.start_file(name, opts).map_err(|e| self.err(e))?;
        let short = name.rsplit('/').next().unwrap_or(name).to_string();
        let mut buf = vec![0u8; 4 << 20];
        loop {
            if crate::cancel::is_set(cancel) {
                return Err(Error::Cancelled);
            }
            let n = input.read(&mut buf).map_err(|e| Error::Io(src.to_path_buf(), e))?;
            if n == 0 {
                break;
            }
            self.zip.write_all(&buf[..n]).map_err(|e| self.err(e))?;
            self.done += n as u64;
            report(self.done, self.total, &short);
        }
        Ok(())
    }

    fn finish(self) -> Result<u64, Error> {
        let path = self.path.clone();
        let file = self.zip.finish().map_err(|e| Error::Export(format!("cannot write {}: {e}", path.display())))?;
        file.sync_all().map_err(|e| Error::Io(path.clone(), e))?;
        Ok(file.metadata().map(|m| m.len()).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_used_stretches_with_handles_inside_the_file() {
        let r = merge(vec![padded(10.0, 12.0, 2.0, 30.0), padded(1.0, 3.0, 2.0, 30.0), padded(14.0, 16.0, 2.0, 30.0)]);
        assert_eq!(r, vec![Range { start: 0.0, end: 5.0 }, Range { start: 8.0, end: 18.0 }]);
        assert_eq!(padded(27.0, 29.5, 2.0, 30.0), Range { start: 25.0, end: 30.0 });
        // Unknown duration: no upper clamp.
        assert_eq!(padded(5.0, 6.0, 2.0, 0.0), Range { start: 3.0, end: 8.0 });
    }

    #[test]
    fn names_are_windows_safe_and_unique_ignoring_case() {
        assert_eq!(safe_name("Q&A: why? <now>|\"ok\"", "x"), "Q&A_ why_ _now___ok_");
        assert_eq!(safe_name("  ...  ", "cut"), "cut");
        assert_eq!(safe_name("con.mp4", "x"), "_con.mp4");
        assert_eq!(safe_name("trailing. ", "x"), "trailing");
        let mut n = UniqueNames::default();
        assert_eq!(n.take("a.mp4"), "a.mp4");
        assert_eq!(n.take("A.mp4"), "A (2).mp4");
        assert_eq!(n.take("a:b.mp4"), "a_b.mp4");
        assert_eq!(clock(72.4), "1m12s");
        assert_eq!(clock(3723.0), "1h02m03s");
        assert_eq!(url_segment("a b#1.mov"), "a%20b%231.mov");
    }

    #[test]
    fn encode_keeps_every_audio_stream_and_the_frame_rate() {
        let args = encode_args(
            Path::new("/in.mp4"),
            Fps::new(30000, 1001),
            Range { start: 1.5, end: 4.0 },
            Path::new("/o.mov"),
        );
        let a: Vec<String> = args.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        let joined = a.join(" ");
        assert!(joined.contains("-ss 1.500000 -i /in.mp4 -t 2.500000"), "{joined}");
        assert!(joined.contains("-map 0:v:0? -map 0:a?"));
        assert!(joined.contains("-c:v libx264") && joined.contains("-crf 16") && joined.contains("-g 15"));
        assert!(joined.contains("-r 30000/1001") && joined.contains("-c:a pcm_s16le"));
    }

    // ---- end to end, with real footage made by ffmpeg ----

    use crate::projects::NewProject;
    use rusqlite::params;

    fn have(tool: &str) -> bool {
        std::process::Command::new(tool).arg("-version").output().is_ok_and(|o| o.status.success())
    }

    fn make_video(ffmpeg: &Path, path: &Path, seconds: u32, freq: u32) {
        let st = std::process::Command::new(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("testsrc=size=320x180:rate=25:duration={seconds}"))
            .args(["-f", "lavfi", "-i"])
            .arg(format!("sine=frequency={freq}:sample_rate=48000:duration={seconds}"))
            .args(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
            .arg(path)
            .status()
            .unwrap();
        assert!(st.success());
    }

    /// One frame at `t`, as 64x36 grey pixels.
    fn frame(ffmpeg: &Path, file: &Path, t: f64) -> Vec<u8> {
        let out = std::process::Command::new(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{t:.4}"), "-i"])
            .arg(file)
            .args(["-frames:v", "1", "-vf", "scale=64:36", "-pix_fmt", "gray", "-f", "rawvideo", "-"])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        out.stdout
    }

    fn diff(a: &[u8], b: &[u8]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len().max(1) as f64
    }

    struct Fixture {
        _tmp: tempfile::TempDir,
        dir: PathBuf,
        db: Db,
        script_id: i64,
        sources: HashMap<String, PathBuf>,
    }

    fn fixture(ffmpeg: &Path) -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let footage = dir.join("footage");
        std::fs::create_dir_all(&footage).unwrap();
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Pkg")).unwrap();
        let folder = db.add_folder(project.id, &footage, true).unwrap();
        let mut sources = HashMap::new();
        for (id, name, secs, freq) in [(1, "inter view.mp4", 20, 440), (2, "broll.mp4", 12, 880)] {
            let path = footage.join(name);
            make_video(ffmpeg, &path, secs, freq);
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio) VALUES (?1, ?2, 100, ?3, 25.0, 1)",
                    params![id, format!("hash{id}"), secs as f64],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 100, 0, 0)",
                    params![id, folder.id, path.to_str().unwrap()],
                )
                .unwrap();
            sources.insert(name.to_string(), path);
        }
        let clip = |video_id: i64, in_s: f64, out_s: f64, audio: script::Audio| script::ScriptClip {
            exact: false,
            gain_db: None,
            sentences: None,
            framing: None,
            video_id,
            in_s,
            out_s,
            audio,
            why: None,
            tracks: None,
        };
        let beat = |id: &str, clips: Vec<script::ScriptClip>, bed: Option<script::AudioBed>| script::Beat {
            id: id.into(),
            purpose: "p".into(),
            narration: None,
            on_screen_text: None,
            title_layout: None,
            clips,
            notes: None,
            bed,
            card: None,
        };
        let s = Script {
            framing: None,
            captions: None,
            title_style: None,
            caption_edits: Vec::new(),
            agents: None,
            title: "My: Cut?".into(),
            target_duration_s: None,
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![
                beat(
                    "b1",
                    vec![clip(1, 3.0, 5.0, script::Audio::Source), clip(1, 6.0, 8.0, script::Audio::Source)],
                    None,
                ),
                beat(
                    "b2",
                    vec![clip(2, 4.0, 6.0, script::Audio::Mute)],
                    Some(script::AudioBed {
                        video_id: 1,
                        in_s: 15.0,
                        out_s: 17.0,
                        why: None,
                        inferred: false,
                        tracks: None,
                        gain_db: None,
                        sentences: None,
                    }),
                ),
            ],
            card_style: None,
        };
        let script_id = script::save_version(&db, project.id, &s, None).unwrap();
        Fixture { _tmp: tmp, dir, db, script_id, sources }
    }

    fn unzip(zip_path: &Path, into: &Path) {
        let file = std::fs::File::open(zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        archive.extract(into).unwrap();
    }

    /// Every clip and bed of the OTIO as (media name, start s, end s, source start s).
    fn otio_clips(doc: &Value) -> Vec<(String, f64, f64, f64)> {
        let mut out = Vec::new();
        for track in doc["tracks"]["children"].as_array().unwrap() {
            for item in track["children"].as_array().unwrap() {
                if item["OTIO_SCHEMA"].as_str().is_some_and(|s| s.starts_with("Clip")) {
                    let refs = if item["media_reference"].is_object() {
                        vec![&item["media_reference"]]
                    } else {
                        item["media_references"].as_object().unwrap().values().collect()
                    };
                    let r = refs[0];
                    let url = r["target_url"].as_str().unwrap();
                    assert!(url.starts_with("media/"), "{url}");
                    let sr = &item["source_range"];
                    let rate = sr["start_time"]["rate"].as_f64().unwrap();
                    let a = sr["start_time"]["value"].as_f64().unwrap() / rate;
                    let d = sr["duration"]["value"].as_f64().unwrap() / sr["duration"]["rate"].as_f64().unwrap();
                    let start = r["metadata"]["ghostreel"]["source_start_s"].as_f64().unwrap();
                    out.push((url.trim_start_matches("media/").replace("%20", " "), a, a + d, start));
                }
            }
        }
        out
    }

    fn package(fx: &Fixture, ffmpeg: &Path, mode: MediaMode) -> (PathBuf, PackageResult) {
        let out = fx.dir.join("out").join(format!("cut-{}.zip", mode.as_str()));
        let opts = PackageOptions { media: mode, ..Default::default() };
        let mut steps = Vec::new();
        let res =
            export_package(&fx.db, ffmpeg, fx.script_id, &out, &opts, &mut |_, f, n| steps.push((f, n.to_string())))
                .unwrap();
        assert!(out.is_file());
        assert!(steps.windows(2).all(|w| w[0].0 <= w[1].0 + 1e-9), "progress goes forward");
        // Nothing left of the work dir.
        let leftovers: Vec<_> = std::fs::read_dir(out.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".ghostreel-package-"))
            .collect();
        assert!(leftovers.is_empty());
        let extract = fx.dir.join(format!("x-{}", mode.as_str()));
        unzip(&out, &extract);
        (extract.join("My_ Cut_"), res)
    }

    fn assert_layout(root: &Path, fx: &Fixture) -> (String, Value) {
        assert!(root.join("README.txt").is_file());
        let readme = std::fs::read_to_string(root.join("README.txt")).unwrap();
        assert!(readme.contains("File > Import") && readme.contains("Locate") && readme.contains("Windows"));
        let xml = std::fs::read_to_string(root.join("My_ Cut_.xml")).unwrap();
        let otio_doc: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("My_ Cut_.otio")).unwrap()).unwrap();
        // The XML points into the package's media folder, never at the original footage.
        assert!(xml.contains("/My_%20Cut_/media/"), "{xml}");
        let footage_dir = fx.sources.values().next().unwrap().parent().unwrap().to_string_lossy().into_owned();
        assert!(!xml.contains(&footage_dir.replace(' ', "%20")) && !xml.contains(&footage_dir));
        assert!(!otio_doc.to_string().contains(&footage_dir));
        assert!(!root.join("preview.mp4").exists(), "no render in the fixture");
        (xml, otio_doc)
    }

    #[test]
    fn packages_trimmed_media_with_frame_accurate_ranges() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else { return };
        if !have("ffprobe") {
            return;
        }
        let fx = fixture(&ffmpeg);
        let (root, res) = package(&fx, &ffmpeg, MediaMode::Trimmed);
        let (xml, doc) = assert_layout(&root, &fx);

        // interview: [3,5]+[6,8] -> [1,10]; bed [15,17] -> [13,19]; broll [4,6] -> [2,8].
        let names: Vec<_> = res.media.iter().map(|m| (m.name.as_str(), m.source_start_s, m.duration_s)).collect();
        assert_eq!(names.len(), 3, "{names:?}");
        let files: Vec<String> = std::fs::read_dir(root.join("media"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files.len(), 3);
        for m in &res.media {
            assert!(files.contains(&m.name));
            assert!(xml.contains(&format!("media/{}", m.name.replace(' ', "%20"))), "{}", m.name);
        }
        let starts: Vec<f64> = res.media.iter().map(|m| m.source_start_s).collect();
        assert_eq!(starts, vec![1.0, 13.0, 2.0]);
        for (m, want) in res.media.iter().zip([9.0, 6.0, 6.0]) {
            assert!((m.duration_s - want).abs() < 0.05, "{} lasts {}", m.name, m.duration_s);
        }

        // Each clip's in point inside its file is the original footage time, frame for frame.
        let clips = otio_clips(&doc);
        assert!(clips.len() >= 4, "{clips:?}");
        let mut originals: Vec<(f64, f64)> = clips.iter().map(|(_, a, b, start)| (a + start, b + start)).collect();
        originals.sort_by(|x, y| x.0.total_cmp(&y.0));
        originals.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-6);
        for want in [(3.0, 5.0), (6.0, 8.0), (15.0, 17.0)] {
            assert!(
                originals.iter().any(|(a, b)| (a - want.0).abs() < 0.021 && (b - want.1).abs() < 0.021),
                "{want:?} in {originals:?}"
            );
        }
        for (name, a, _, start) in &clips {
            let source =
                if name.starts_with("broll") { &fx.sources["broll.mp4"] } else { &fx.sources["inter view.mp4"] };
            let original = frame(&ffmpeg, source, start + a + 0.02);
            let packed = frame(&ffmpeg, &root.join("media").join(name), a + 0.02);
            let off = frame(&ffmpeg, source, start + a + 0.52);
            let same = diff(&original, &packed);
            let other = diff(&original, &off);
            assert!(same < other / 2.0, "{name} @ {a}: diff {same} vs a half-second off {other}");
        }
        let n: i64 = fx
            .db
            .conn
            .query_row("SELECT count(*) FROM exports WHERE format = 'premiere_package'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn packages_whole_files() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else { return };
        let fx = fixture(&ffmpeg);
        let (root, res) = package(&fx, &ffmpeg, MediaMode::Full);
        let (xml, doc) = assert_layout(&root, &fx);
        let names: Vec<_> = res.media.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["inter view.mp4", "broll.mp4"]);
        for (name, src) in [("inter view.mp4", &fx.sources["inter view.mp4"]), ("broll.mp4", &fx.sources["broll.mp4"])]
        {
            assert_eq!(std::fs::read(root.join("media").join(name)).unwrap(), std::fs::read(src).unwrap());
            assert!(xml.contains(&format!("media/{}", name.replace(' ', "%20"))));
        }
        // Whole files: times are the original footage times.
        let clips = otio_clips(&doc);
        assert!(clips.iter().any(|(n, a, b, s)| n == "inter view.mp4"
            && *s == 0.0
            && (a - 3.0).abs() < 1e-6
            && (b - 5.0).abs() < 1e-6));
        assert!(clips.iter().any(|(n, a, _, _)| n == "inter view.mp4" && (a - 15.0).abs() < 1e-6));
        assert!(clips.iter().any(|(n, a, _, _)| n == "broll.mp4" && (a - 4.0).abs() < 1e-6));
    }

    #[test]
    fn a_stop_leaves_nothing_behind() {
        let Some(ffmpeg) = crate::doctor::locate("ffmpeg") else { return };
        let fx = fixture(&ffmpeg);
        let out = fx.dir.join("stopped").join("cut.zip");
        let flag: CancelFlag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opts = PackageOptions { media: MediaMode::Trimmed, cancel: Some(flag.clone()), ..Default::default() };
        let err = export_package(&fx.db, &ffmpeg, fx.script_id, &out, &opts, &mut |_, _, _| {
            flag.store(true, std::sync::atomic::Ordering::SeqCst)
        })
        .unwrap_err();
        assert!(err.is_cancelled(), "{err}");
        assert!(!out.exists());
        assert_eq!(std::fs::read_dir(out.parent().unwrap()).unwrap().count(), 0);
    }
}
