//! Final Cut Pro 7 XML (xmeml v4) — the interchange format Premiere and Resolve import.
//!
//! This used to go through a frozen Python copy of OpenTimelineIO and its fcp_xml adapter, 11.9 MB
//! of sidecar to turn one JSON document into one XML document. The timeline itself was always
//! built here (`otio.rs`); this module is the other half, and reads the format back to report what
//! an exported file contains.
//!
//! What the format wants, learned from the adapter and from files Premiere accepts:
//!
//! - Every time is a frame count, and a *rate* says which clock it is counted on. `start`/`end`
//!   place a clip on the sequence and count in the sequence's rate; `in`/`out`/`duration` address
//!   the source file and count in *its* rate. A 4 s clip from 59.94 fps footage in a 30 fps
//!   sequence is therefore 240 frames long and occupies frames 0–120.
//! - A rate is written as an integer timebase plus an NTSC flag: 59.94 is timebase 60, NTSC TRUE.
//! - Gaps are not written. A clip's `start` carries its position, so silence is simply a jump.
//! - A `<file>` is written out once, with an id; every later clip that uses the same media
//!   references it as `<file id="file-3"/>`. Premiere links them by that id, so the ids must be
//!   stable within the document and each one defined before it is referenced.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;

use crate::Error;

// ---------------------------------------------------------------------------------------------
// The timeline, read out of the OTIO document
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct MediaRef {
    target_url: String,
    /// Rate of the source file, frames per second.
    rate: f64,
    /// How much of the source exists, in its own frames.
    available_frames: f64,
    width: i64,
    height: i64,
    has_audio: bool,
    channels: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Marker {
    name: String,
    /// Frames into the *source*, on the clip's rate.
    in_frames: f64,
    out_frames: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Clip {
    name: String,
    rate: f64,
    in_frames: f64,
    duration_frames: f64,
    media: Option<MediaRef>,
    markers: Vec<Marker>,
    gain_db: Option<f64>,
    metadata: Value,
    source_channel: usize,
}

impl Clip {
    fn duration_s(&self) -> f64 {
        if self.rate > 0.0 { self.duration_frames / self.rate } else { 0.0 }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Item {
    Clip(Clip),
    /// Nothing on the track; only its length matters.
    Gap {
        duration_s: f64,
    },
}

impl Item {
    fn duration_s(&self) -> f64 {
        match self {
            Item::Clip(c) => c.duration_s(),
            Item::Gap { duration_s } => *duration_s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Track {
    name: String,
    /// "Video" or "Audio".
    kind: String,
    items: Vec<Item>,
}

impl Track {
    fn duration_s(&self) -> f64 {
        self.items.iter().map(Item::duration_s).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Timeline {
    name: String,
    /// Sequence rate: the clock `start` and `end` are counted on.
    rate: f64,
    tracks: Vec<Track>,
    width: i64,
    height: i64,
    markers: Vec<Marker>,
}

/// What an exported timeline turned out to contain — the report `ghostreel script export` prints
/// and the app shows, kept in the shape the Python sidecar used so nothing downstream changes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub clips: usize,
    pub duration_s: f64,
    pub rate: f64,
    pub tracks: Vec<TrackSummary>,
    /// Media the timeline points at that is not on disk — the one failure worth catching before
    /// an editor opens the file and finds every clip offline.
    pub missing_media: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrackSummary {
    pub name: String,
    pub kind: String,
    pub items: usize,
    pub clips: usize,
    pub duration_s: f64,
}

// ---------------------------------------------------------------------------------------------
// Reading the OTIO document
// ---------------------------------------------------------------------------------------------

fn want<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a Value, Error> {
    v.get(key).ok_or_else(|| Error::Export(format!("timeline {what} has no '{key}'")))
}

fn want_f64(v: &Value, key: &str, what: &str) -> Result<f64, Error> {
    want(v, key, what)?.as_f64().ok_or_else(|| Error::Export(format!("timeline {what}: '{key}' is not a number")))
}

fn str_or_empty(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

/// A TimeRange.1 as (rate, start value, duration value).
fn read_range(v: &Value, what: &str) -> Result<(f64, f64, f64), Error> {
    let range = want(v, "source_range", what)?;
    let start = want(range, "start_time", what)?;
    let dur = want(range, "duration", what)?;
    Ok((
        want_f64(start, "rate", what)?,
        want_f64(start, "value", what)?,
        want_f64(dur, "value", what)? * want_f64(start, "rate", what)? / want_f64(dur, "rate", what)?,
    ))
}

fn read_media_ref(clip: &Value) -> Result<Option<MediaRef>, Error> {
    let key = clip.get("active_media_reference_key").and_then(Value::as_str).unwrap_or("DEFAULT_MEDIA");
    let Some(r) = clip.get("media_references").and_then(|m| m.get(key)) else {
        return Ok(None);
    };
    let Some(url) = r.get("target_url").and_then(Value::as_str) else {
        return Ok(None);
    };
    let (rate, _start, dur) = match r.get("available_range") {
        Some(range) if !range.is_null() => {
            let start = want(range, "start_time", "media reference")?;
            let d = want(range, "duration", "media reference")?;
            (
                want_f64(start, "rate", "media reference")?,
                want_f64(start, "value", "media reference")?,
                want_f64(d, "value", "media reference")?,
            )
        }
        _ => (0.0, 0.0, 0.0),
    };
    let meta = &r["metadata"]["ghostreel"];
    Ok(Some(MediaRef {
        target_url: url.to_string(),
        rate,
        available_frames: dur,
        width: meta["width"].as_i64().unwrap_or(1920),
        height: meta["height"].as_i64().unwrap_or(1080),
        has_audio: meta["has_audio"].as_bool().unwrap_or(true),
        channels: meta["audio_channels"].as_u64().unwrap_or(1) as usize,
    }))
}

fn read_markers(clip: &Value) -> Result<Vec<Marker>, Error> {
    let mut out = Vec::new();
    for m in clip.get("markers").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
        let range = want(m, "marked_range", "marker")?;
        let start = want(range, "start_time", "marker")?;
        let dur = want(range, "duration", "marker")?;
        let in_frames = want_f64(start, "value", "marker")?;
        out.push(Marker {
            name: str_or_empty(m, "name"),
            in_frames,
            out_frames: in_frames + want_f64(dur, "value", "marker")?,
        });
    }
    Ok(out)
}

fn read_gain_db(clip: &Value) -> Option<f64> {
    if let Some(effects) = clip.get("effects").and_then(Value::as_array) {
        for eff in effects {
            let is_volume = eff.get("effect_name").and_then(Value::as_str) == Some("AudioVolume")
                || eff.get("name").and_then(Value::as_str) == Some("AudioVolume");
            if is_volume
                && let Some(db) = eff
                    .get("metadata")
                    .and_then(|m| m.get("ghostreel"))
                    .and_then(|g| g.get("gain_db"))
                    .and_then(Value::as_f64)
                && db.is_finite()
                && db.abs() > 0.05
            {
                return Some(db);
            }
        }
    }
    clip.get("metadata")
        .and_then(|m| m.get("ghostreel"))
        .and_then(|g| g.get("gain_db"))
        .and_then(Value::as_f64)
        .filter(|db| db.is_finite() && db.abs() > 0.05)
}

fn read_timeline(doc: &Value) -> Result<Timeline, Error> {
    let schema = doc.get("OTIO_SCHEMA").and_then(Value::as_str).unwrap_or("");
    if !schema.starts_with("Timeline.") {
        return Err(Error::Export(format!("not an OTIO timeline (OTIO_SCHEMA = '{schema}')")));
    }

    let rate = doc
        .get("global_start_time")
        .filter(|v| !v.is_null())
        .map(|t| want_f64(t, "rate", "global start time"))
        .transpose()?
        .unwrap_or(0.0);

    let stack = want(doc, "tracks", "document")?;
    let mut tracks = Vec::new();
    let mut gap_markers = Vec::new();
    for t in stack.get("children").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
        let kind = str_or_empty(t, "kind");
        let mut items = Vec::new();
        for child in t.get("children").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
            let child_schema = child.get("OTIO_SCHEMA").and_then(Value::as_str).unwrap_or("");
            if child_schema.starts_with("Clip.") {
                let (c_rate, mut in_frames, mut dur) = read_range(child, "clip")?;
                let media = read_media_ref(child)?;
                if let Some(m) = &media
                    && m.available_frames > 0.0
                {
                    let available = m.available_frames * c_rate / if m.rate > 0.0 { m.rate } else { c_rate };
                    in_frames = in_frames.max(0.0).min(available);
                    dur = dur.min(available - in_frames).max(0.0);
                }
                items.push(Item::Clip(Clip {
                    name: str_or_empty(child, "name"),
                    rate: c_rate,
                    in_frames,
                    duration_frames: dur,
                    media,
                    markers: read_markers(child)?,
                    gain_db: read_gain_db(child),
                    metadata: child["metadata"]["ghostreel"].clone(),
                    source_channel: 1,
                }));
            } else if child_schema.starts_with("Gap.") {
                let (g_rate, _start, dur) = read_range(child, "gap")?;
                let offset = items.iter().map(Item::duration_s).sum::<f64>();
                for mut marker in read_markers(child)? {
                    marker.in_frames = (offset + marker.in_frames / g_rate) * if rate > 0.0 { rate } else { g_rate };
                    marker.out_frames = (offset + marker.out_frames / g_rate) * if rate > 0.0 { rate } else { g_rate };
                    gap_markers.push(marker);
                }
                items.push(Item::Gap { duration_s: if g_rate > 0.0 { dur / g_rate } else { 0.0 } });
            } else {
                // Transitions and nested stacks: GhostReel never writes them, and passing one
                // through silently would move every clip after it.
                return Err(Error::Export(format!("unsupported timeline item '{child_schema}'")));
            }
        }
        tracks.push(Track { name: str_or_empty(t, "name"), kind, items });
    }

    let rate = if rate > 0.0 {
        rate
    } else {
        // No global start time: fall back to the first clip's rate so the sequence has a clock.
        tracks
            .iter()
            .flat_map(|t| t.items.iter())
            .find_map(|i| match i {
                Item::Clip(c) if c.rate > 0.0 => Some(c.rate),
                _ => None,
            })
            .unwrap_or(30.0)
    };

    // Premiere writes stereo as two mono clipitems. A selected source stream maps
    // to its interleaved channels; each exported channel has its own source track.
    let mut expanded = Vec::new();
    for track in tracks {
        if track.kind != "Audio" {
            expanded.push(track);
            continue;
        }
        let channels = track
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Clip(c) => Some(
                    c.media.as_ref().map_or(1, |m| m.channels).max(1)
                        * c.metadata["tracks"].as_array().map_or(1, |a| a.len()),
                ),
                _ => None,
            })
            .max()
            .unwrap_or(1);
        for channel in 0..channels {
            let mut t = track.clone();
            if channel > 0 {
                t.name = format!("{} channel {}", t.name, channel + 1);
            }
            for item in &mut t.items {
                if let Item::Clip(c) = item {
                    let count = c.media.as_ref().map_or(1, |m| m.channels).max(1);
                    let streams = c.metadata["tracks"].as_array();
                    let stream = streams.and_then(|a| a.get(channel / count)).and_then(Value::as_u64);
                    if channel >= count * streams.map_or(1, |a| a.len()) {
                        *item = Item::Gap { duration_s: c.duration_s() };
                    } else {
                        c.source_channel = stream.unwrap_or(0) as usize * count + channel % count + 1;
                    }
                }
            }
            expanded.push(t);
        }
    }
    Ok(Timeline {
        name: str_or_empty(doc, "name"),
        rate,
        tracks: expanded,
        width: doc["metadata"]["ghostreel"]["width"].as_i64().or(doc["metadata"]["width"].as_i64()).unwrap_or(1920),
        height: doc["metadata"]["ghostreel"]["height"].as_i64().or(doc["metadata"]["height"].as_i64()).unwrap_or(1080),
        markers: {
            let mut markers = read_markers(stack)?;
            markers.extend(gap_markers);
            markers
        },
    })
}

// ---------------------------------------------------------------------------------------------
// Writing xmeml
// ---------------------------------------------------------------------------------------------

/// An FCP rate: an integer timebase plus a flag for the 1000/1001 rates.
fn timebase_ntsc(fps: f64) -> (i64, bool) {
    let timebase = fps.ceil();
    (timebase as i64, (timebase - fps).abs() > f64::EPSILON)
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Percent-decode a file:// URL back to something worth showing a person.
pub fn path_from_url(url: &str) -> String {
    let rest = url.strip_prefix("file://localhost").or_else(|| url.strip_prefix("file://")).unwrap_or(url);
    let bytes = rest.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok());
            if let Some(b) = hex {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    let decoded = String::from_utf8_lossy(&out).into_owned();
    // file:///C:/x -> C:/x; file:///home/x stays absolute.
    let trimmed = decoded.strip_prefix('/').unwrap_or(&decoded);
    let looks_like_drive = {
        let b = trimmed.as_bytes();
        b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':'
    };
    if looks_like_drive { trimmed.to_string() } else { decoded }
}

fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// A tiny indenting XML writer. The format is small and fixed, so this beats pulling a serializer
/// in to describe it.
struct Xml {
    out: String,
    depth: usize,
}

impl Xml {
    fn new() -> Self {
        Xml { out: String::from("<?xml version=\"1.0\" ?>\n"), depth: 0 }
    }

    fn pad(&mut self) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
    }

    fn open(&mut self, tag: &str) {
        self.pad();
        let _ = writeln!(self.out, "<{tag}>");
        self.depth += 1;
    }

    fn open_attr(&mut self, tag: &str, attrs: &str) {
        self.pad();
        let _ = writeln!(self.out, "<{tag} {attrs}>");
        self.depth += 1;
    }

    fn close(&mut self, tag: &str) {
        self.depth = self.depth.saturating_sub(1);
        self.pad();
        let _ = writeln!(self.out, "</{tag}>");
    }

    fn empty(&mut self, tag: &str) {
        self.pad();
        let _ = writeln!(self.out, "<{tag}/>");
    }

    fn empty_attr(&mut self, tag: &str, attrs: &str) {
        self.pad();
        let _ = writeln!(self.out, "<{tag} {attrs}/>");
    }

    fn text(&mut self, tag: &str, text: &str) {
        self.pad();
        let _ = writeln!(self.out, "<{tag}>{}</{tag}>", esc(text));
    }

    fn num(&mut self, tag: &str, n: f64) {
        self.text(tag, &format!("{}", n.round() as i64));
    }

    fn rate(&mut self, fps: f64) {
        let (timebase, ntsc) = timebase_ntsc(fps);
        self.open("rate");
        self.text("timebase", &timebase.to_string());
        self.text("ntsc", if ntsc { "TRUE" } else { "FALSE" });
        self.close("rate");
    }

    /// Timecode at the head of the media or sequence. GhostReel always starts at zero: the source
    /// `in` points we write are offsets from the start of the file, not broadcast timecode.
    fn timecode(&mut self, fps: f64) {
        self.open("timecode");
        self.rate(fps);
        self.text("string", "00:00:00:00");
        self.text("frame", "0");
        self.text("displayformat", "NDF");
        self.close("timecode");
    }
}

/// Convert an OpenTimelineIO document into Final Cut Pro 7 XML.
pub fn from_otio(doc: &Value) -> Result<String, Error> {
    let tl = read_timeline(doc)?;
    Ok(write_xml(&tl))
}

fn write_xml(tl: &Timeline) -> String {
    let seq_rate = tl.rate;
    let mut x = Xml::new();

    // Ids are handed out in document order, and a file is spelled out the first time it appears.
    let mut file_ids: HashMap<String, usize> = HashMap::new();
    let mut next_file_id = 1usize;
    let mut next_clip_id = 1usize;

    let total_s = tl.tracks.iter().map(Track::duration_s).fold(0.0, f64::max);

    x.open_attr("xmeml", "version=\"4\"");
    x.open("project");
    x.text("name", &tl.name);
    x.open("children");
    x.open_attr("sequence", "id=\"sequence-1\"");
    x.text("name", &tl.name);
    x.num("duration", total_s * seq_rate);
    x.rate(seq_rate);
    for m in &tl.markers {
        x.open("marker");
        x.text("name", &m.name);
        x.num("in", m.in_frames);
        x.num("out", m.out_frames);
        x.close("marker");
    }
    x.open("media");
    let links = clip_links(tl);

    for kind in ["Video", "Audio"] {
        x.open(&kind.to_lowercase());
        if kind == "Video" {
            x.open("format");
            video_characteristics(&mut x, seq_rate, tl.width, tl.height);
            x.close("format");
        } else {
            x.num("numOutputChannels", 2.0);
            x.open("format");
            audio_characteristics(&mut x);
            x.close("format");
            x.open("outputs");
            x.open("group");
            x.num("index", 1.0);
            x.num("numchannels", 2.0);
            x.num("downmix", 0.0);
            for channel in 1..=2 {
                x.open("channel");
                x.num("index", channel as f64);
                x.close("channel");
            }
            x.close("group");
            x.close("outputs");
        }
        for track in tl.tracks.iter().filter(|t| t.kind == kind) {
            write_track(
                &mut x,
                track,
                seq_rate,
                &mut file_ids,
                &mut next_file_id,
                &mut next_clip_id,
                &links,
                tl.width,
                tl.height,
            );
        }
        x.close(&kind.to_lowercase());
    }

    x.close("media");
    x.timecode(seq_rate);
    x.close("sequence");
    x.close("children");
    x.close("project");
    x.close("xmeml");
    x.out
}

#[allow(clippy::too_many_arguments)]
fn write_track(
    x: &mut Xml,
    track: &Track,
    seq_rate: f64,
    file_ids: &mut HashMap<String, usize>,
    next_file_id: &mut usize,
    next_clip_id: &mut usize,
    links: &HashMap<usize, Vec<Link>>,
    width: i64,
    height: i64,
) {
    let clips: Vec<&Item> = track.items.iter().collect();
    if !clips.iter().any(|i| matches!(i, Item::Clip(_))) {
        // A track of nothing but gaps (the title track) still has to exist: the tracks below it
        // are numbered by position.
        x.empty("track");
        return;
    }

    x.open("track");
    x.text("name", &track.name);
    // Position is carried by `start`, so gaps are counted, not written.
    let is_audio = track.kind == "Audio";
    let mut at_s = 0.0f64;
    for item in clips {
        let dur_s = item.duration_s();
        if let Item::Clip(clip) = item {
            write_clip(
                x,
                clip,
                is_audio,
                at_s,
                dur_s,
                seq_rate,
                file_ids,
                next_file_id,
                next_clip_id,
                links,
                width,
                height,
            );
        }
        at_s += dur_s;
    }
    if is_audio {
        let output = track
            .items
            .iter()
            .find_map(|i| match i {
                Item::Clip(c) => Some((c.source_channel - 1) % 2 + 1),
                _ => None,
            })
            .unwrap_or(1);
        x.num("outputchannelindex", output as f64);
    }
    x.close("track");
}

#[derive(Clone)]
struct Link {
    id: usize,
    kind: &'static str,
    track: usize,
    index: usize,
}

fn clip_links(tl: &Timeline) -> HashMap<usize, Vec<Link>> {
    let mut placed = Vec::new();
    let mut id = 0;
    for kind in ["Video", "Audio"] {
        for (track, t) in tl.tracks.iter().filter(|t| t.kind == kind).enumerate() {
            let mut at = 0.0;
            let mut index = 0;
            for item in &t.items {
                if let Item::Clip(c) = item {
                    id += 1;
                    index += 1;
                    placed.push((
                        Link { id, kind: if kind == "Video" { "video" } else { "audio" }, track: track + 1, index },
                        at,
                        c,
                    ));
                }
                at += item.duration_s();
            }
        }
    }
    let mut groups = HashMap::new();
    for (v, at, clip) in placed.iter().filter(|(l, _, _)| l.kind == "video") {
        let mut group = vec![v.clone()];
        for (a, audio_at, audio) in placed.iter().filter(|(l, _, _)| l.kind == "audio") {
            if audio.metadata["bed"] == true {
                continue;
            }
            if (at - audio_at).abs() < 1e-6
                && (clip.in_frames / clip.rate - audio.in_frames / audio.rate).abs() < 1e-6
                && (clip.duration_s() - audio.duration_s()).abs() < 1e-6
                && clip.media.as_ref().map(|m| &m.target_url) == audio.media.as_ref().map(|m| &m.target_url)
            {
                group.push(a.clone());
            }
        }
        if group.len() > 1 {
            for member in &group {
                groups.insert(member.id, group.clone());
            }
        }
    }
    groups
}

fn video_characteristics(x: &mut Xml, rate: f64, width: i64, height: i64) {
    x.open("samplecharacteristics");
    x.rate(rate);
    x.num("width", width as f64);
    x.num("height", height as f64);
    x.text("pixelaspectratio", "square");
    x.text("anamorphic", "FALSE");
    x.text("fielddominance", "none");
    x.close("samplecharacteristics");
}

fn audio_characteristics(x: &mut Xml) {
    x.open("samplecharacteristics");
    x.num("depth", 16.0);
    x.num("samplerate", 48000.0);
    x.close("samplecharacteristics");
}

/// FCP Basic Motion's center is an offset in sequence-width/height units.
/// Move the selected source rectangle's center onto the sequence center.
fn crop_motion(rect: crate::framing::Rect, src_w: f64, src_h: f64, out_w: f64, out_h: f64) -> (f64, f64, f64) {
    let scale = (out_w / rect.w).max(out_h / rect.h);
    (
        100.0 * scale,
        (src_w / 2.0 - rect.x - rect.w / 2.0) * scale / out_w,
        (src_h / 2.0 - rect.y - rect.h / 2.0) * scale / out_h,
    )
}

fn write_motion_filter(x: &mut Xml, clip: &Clip, out_w: f64, out_h: f64) {
    let Ok(plan) = serde_json::from_value::<crate::framing::FramePlan>(clip.metadata["frame_plan"].clone()) else {
        return;
    };
    let mut keys = Vec::new();
    let mut fallback = false;
    for segment in &plan.segments {
        if let Some(path) = segment.paths.first() {
            fallback |= segment.paths.len() > 1;
            for key in &path.keyframes {
                if key.t < segment.start || key.t > segment.end {
                    continue;
                }
                let rect = crate::framing::Rect { x: key.x, y: key.y, w: path.w, h: path.h };
                let (scale, h, v) = crop_motion(rect, plan.src_w, plan.src_h, out_w, out_h);
                keys.push((key.t, scale, h, v, key.cut));
            }
        } else {
            keys.push((segment.start, 100.0 * (out_w / plan.src_w).min(out_h / plan.src_h), 0.0, 0.0, true));
        }
    }
    if keys.is_empty() {
        return;
    }
    if fallback {
        x.open("comments");
        x.text(
            "comment",
            "GhostReel: stacked/screen layout exported as its main crop; recreate the second picture band in Premiere.",
        );
        x.close("comments");
    }
    x.open("filter");
    x.open("effect");
    x.text("name", "Basic Motion");
    x.text("effectid", "basic");
    x.text("effectcategory", "motion");
    x.text("effecttype", "motion");
    x.text("mediatype", "video");
    for parameter in ["scale", "center"] {
        x.open("parameter");
        x.text("parameterid", parameter);
        x.text("name", if parameter == "scale" { "Scale" } else { "Center" });
        for (t, scale, h, v, cut) in &keys {
            let keyed = keys.len() > 1;
            if keyed {
                x.open("keyframe");
                x.num("when", clip.in_frames + t * clip.rate);
            }
            if parameter == "scale" {
                x.text("value", &format!("{scale:.8}"));
            } else {
                x.open("value");
                x.text("horiz", &format!("{h:.8}"));
                x.text("vert", &format!("{v:.8}"));
                x.close("value");
            }
            if keyed {
                x.open("interpolation");
                x.text("name", if *cut { "hold" } else { "linear" });
                x.close("interpolation");
                x.close("keyframe");
            }
        }
        x.close("parameter");
    }
    x.close("effect");
    x.close("filter");
}

fn write_audio_levels_filter(x: &mut Xml, gain_db: f64) {
    let linear = 10.0_f64.powf(gain_db / 20.0);
    x.open("filter");
    x.open("effect");
    x.text("name", "Audio Levels");
    x.text("effectid", "audiolevels");
    x.text("effectcategory", "audiolevels");
    x.text("effecttype", "audiolevels");
    x.text("mediatype", "audio");
    x.open("parameter");
    x.text("parameterid", "level");
    x.text("name", "Level");
    x.text("valuemin", "0");
    x.text("valuemax", "3.98109");
    x.text("value", &format!("{linear:.6}"));
    x.close("parameter");
    x.close("effect");
    x.close("filter");
}

#[allow(clippy::too_many_arguments)]
fn write_clip(
    x: &mut Xml,
    clip: &Clip,
    is_audio: bool,
    at_s: f64,
    dur_s: f64,
    seq_rate: f64,
    file_ids: &mut HashMap<String, usize>,
    next_file_id: &mut usize,
    next_clip_id: &mut usize,
    links: &HashMap<usize, Vec<Link>>,
    width: i64,
    height: i64,
) {
    let clip_id = *next_clip_id;
    *next_clip_id += 1;
    x.open_attr("clipitem", &format!("frameBlend=\"FALSE\" id=\"clipitem-{clip_id}\""));

    match &clip.media {
        Some(media) => {
            let path = path_from_url(&media.target_url);
            let name = base_name(&path);
            match file_ids.get(&media.target_url) {
                Some(id) => x.empty_attr("file", &format!("id=\"file-{id}\"")),
                None => {
                    let id = *next_file_id;
                    *next_file_id += 1;
                    file_ids.insert(media.target_url.clone(), id);

                    let file_rate = if media.rate > 0.0 { media.rate } else { clip.rate };
                    x.open_attr("file", &format!("id=\"file-{id}\""));
                    let url = if let Some(rest) = media.target_url.strip_prefix("file:///") {
                        format!("file://localhost/{rest}")
                    } else if media.target_url.starts_with("file://localhost/") {
                        media.target_url.clone()
                    } else {
                        format!("file://localhost/{}", media.target_url.trim_start_matches('/'))
                    };
                    x.text("pathurl", &url);
                    x.text("name", &name);
                    x.rate(file_rate);
                    if media.available_frames > 0.0 {
                        x.num("duration", media.available_frames);
                    }
                    x.timecode(file_rate);
                    x.open("media");
                    // Which streams the file offers. Both, unless the name says audio only.
                    let audio_only = matches!(
                        name.rsplit('.').next().map(str::to_ascii_lowercase).as_deref(),
                        Some("wav" | "aac" | "mp3" | "aif" | "aiff" | "m4a" | "flac")
                    );
                    if !audio_only && media.width > 0 && media.height > 0 {
                        x.open("video");
                        video_characteristics(x, file_rate, media.width, media.height);
                        x.close("video");
                    }
                    if media.has_audio {
                        x.open("audio");
                        audio_characteristics(x);
                        x.num("channelcount", media.channels.max(1) as f64);
                        x.text("layout", if media.channels == 1 { "mono" } else { "stereo" });
                        for channel in 1..=media.channels.max(1) {
                            x.open("audiochannel");
                            x.num("sourcechannel", channel as f64);
                            x.text("channellabel", if channel == 1 { "left" } else { "right" });
                            x.close("audiochannel");
                        }
                        x.close("audio");
                    }
                    x.close("media");
                    x.close("file");
                }
            }
            x.text("name", if clip.name.is_empty() { &name } else { &clip.name });
        }
        None => {
            x.text("name", &clip.name);
        }
    }

    x.rate(clip.rate);
    for m in &clip.markers {
        x.open("marker");
        x.text("name", &m.name);
        x.num("in", m.in_frames);
        x.num("out", m.out_frames);
        x.close("marker");
    }

    x.text("enabled", "TRUE");
    x.num("duration", clip.media.as_ref().map_or(clip.duration_frames, |m| m.available_frames));
    x.num("start", at_s * seq_rate);
    x.num("end", (at_s + dur_s) * seq_rate);
    let available = clip
        .media
        .as_ref()
        .filter(|m| m.available_frames > 0.0)
        .map_or(f64::INFINITY, |m| m.available_frames / if m.rate > 0.0 { m.rate } else { clip.rate } * clip.rate);
    let source_in = clip.in_frames.max(0.0).min(available);
    x.num("in", source_in);
    x.num("out", (source_in + clip.duration_frames).min(available));
    if is_audio {
        x.open("sourcetrack");
        x.text("mediatype", "audio");
        x.num("trackindex", clip.source_channel as f64);
        x.close("sourcetrack");
    } else {
        write_motion_filter(x, clip, width as f64, height as f64);
    }
    if let Some(group) = links.get(&clip_id) {
        for link in group {
            x.open("link");
            x.text("linkclipref", &format!("clipitem-{}", link.id));
            x.text("mediatype", link.kind);
            x.num("trackindex", link.track as f64);
            x.num("clipindex", link.index as f64);
            if link.kind == "audio" {
                x.num("groupindex", 1.0);
            }
            x.close("link");
        }
    }
    if is_audio && let Some(gain_db) = clip.gain_db {
        write_audio_levels_filter(x, gain_db);
    }
    x.close("clipitem");
}

// ---------------------------------------------------------------------------------------------
// Reading back
// ---------------------------------------------------------------------------------------------

fn summarize(tl: &Timeline) -> Summary {
    let mut missing_media = Vec::new();
    let mut clips = 0usize;
    let mut tracks = Vec::new();

    for t in &tl.tracks {
        let mut track_clips = 0usize;
        for item in &t.items {
            if let Item::Clip(c) = item {
                track_clips += 1;
                if let Some(media) = &c.media {
                    let path = path_from_url(&media.target_url);
                    if !std::path::Path::new(&path).exists() && !missing_media.contains(&path) {
                        missing_media.push(path);
                    }
                }
            }
        }
        clips += track_clips;
        tracks.push(TrackSummary {
            name: t.name.clone(),
            kind: t.kind.clone(),
            items: t.items.len(),
            clips: track_clips,
            duration_s: t.duration_s(),
        });
    }

    Summary {
        clips,
        duration_s: tl.tracks.iter().map(Track::duration_s).fold(0.0, f64::max),
        rate: tl.rate,
        tracks,
        missing_media,
    }
}

/// Summarize an OpenTimelineIO document.
pub fn summarize_otio(doc: &Value) -> Result<Summary, Error> {
    Ok(summarize(&read_timeline(doc)?))
}

/// Summarize a Final Cut Pro 7 XML document by reading it back.
pub fn summarize_fcp_xml(xml: &str) -> Result<Summary, Error> {
    use quick_xml::events::Event;

    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // Where we are: the element path, plus the file ids seen so far (a clip may reference one).
    let mut path: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut file_urls: HashMap<String, String> = HashMap::new();

    let mut seq_rate: Option<f64> = None;
    let mut tracks: Vec<TrackSummary> = Vec::new();
    let mut missing_media: Vec<String> = Vec::new();

    // Current clipitem, and the rate/timebase context it sits in.
    let mut cur_timebase: Option<f64> = None;
    let mut cur_file_id: Option<String> = None;
    let mut cur_end: Option<f64> = None;
    let mut track_end_frames = 0.0f64;
    let mut track_clips = 0usize;
    let mut in_kind = String::new();

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => {
                return Err(Error::Export(format!("malformed FCP XML at byte {}: {e}", reader.buffer_position())));
            }
            Ok(Event::Eof) => {
                // quick-xml stops at the end of input without complaining about what is still
                // open; a truncated export would otherwise read as a valid empty timeline.
                if let Some(open) = path.last() {
                    return Err(Error::Export(format!("FCP XML ends inside <{open}>")));
                }
                break;
            }
            Ok(Event::Start(e)) => {
                let name = e.name().as_ref().to_string();
                match name.as_str() {
                    // The sequence's own <media> sections, not a <file>'s description of its streams.
                    "video" | "audio"
                        if path.len() >= 2 && path[path.len() - 1] == "media" && path[path.len() - 2] == "sequence" =>
                    {
                        in_kind = if name == "video" { "Video".into() } else { "Audio".into() };
                    }
                    "track" => {
                        track_end_frames = 0.0;
                        track_clips = 0;
                    }
                    "clipitem" => {
                        cur_file_id = None;
                        cur_end = None;
                    }
                    "file" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == "id" {
                                cur_file_id = Some(attr.value.to_string());
                            }
                        }
                    }
                    _ => {}
                }
                path.push(name);
                text.clear();
            }
            Ok(Event::Empty(e)) => {
                let name = e.name().as_ref().to_string();
                if name == "file" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == "id" {
                            cur_file_id = Some(attr.value.to_string());
                        }
                    }
                } else if name == "track" {
                    tracks.push(TrackSummary {
                        name: String::new(),
                        kind: in_kind.clone(),
                        items: 0,
                        clips: 0,
                        duration_s: 0.0,
                    });
                }
            }
            Ok(Event::Text(t)) => {
                text = t.xml10_content().into_owned();
            }
            Ok(Event::End(e)) => {
                let name = e.name().as_ref().to_string();
                let in_clipitem = path.iter().any(|p| p == "clipitem");
                let in_file = path.iter().any(|p| p == "file");
                match name.as_str() {
                    "timebase" => cur_timebase = text.trim().parse::<f64>().ok(),
                    "ntsc" => {
                        let ntsc = text.trim().eq_ignore_ascii_case("TRUE");
                        if let Some(tb) = cur_timebase.take() {
                            let fps = if ntsc { tb * 1000.0 / 1001.0 } else { tb };
                            // The first rate in the document is the sequence's own.
                            if seq_rate.is_none() && !in_clipitem && !in_file {
                                seq_rate = Some(fps);
                            }
                        }
                    }
                    "pathurl" if in_file => {
                        if let Some(id) = &cur_file_id {
                            file_urls.insert(id.clone(), text.clone());
                        }
                    }
                    "end" if in_clipitem && !in_file => cur_end = text.trim().parse::<f64>().ok(),
                    "clipitem" => {
                        track_clips += 1;
                        if let Some(end) = cur_end.take() {
                            track_end_frames = track_end_frames.max(end);
                        }
                        if let Some(url) = cur_file_id.as_ref().and_then(|id| file_urls.get(id)) {
                            let p = path_from_url(url);
                            if !std::path::Path::new(&p).exists() && !missing_media.contains(&p) {
                                missing_media.push(p);
                            }
                        }
                    }
                    "track" => {
                        let rate = seq_rate.unwrap_or(30.0);
                        tracks.push(TrackSummary {
                            name: String::new(),
                            kind: in_kind.clone(),
                            items: track_clips,
                            clips: track_clips,
                            duration_s: if rate > 0.0 { track_end_frames / rate } else { 0.0 },
                        });
                    }
                    _ => {}
                }
                path.pop();
                text.clear();
            }
            _ => {}
        }
        buf.clear();
    }

    let rate = seq_rate.unwrap_or(0.0);
    Ok(Summary {
        clips: tracks.iter().map(|t| t.clips).sum(),
        duration_s: tracks.iter().map(|t| t.duration_s).fold(0.0, f64::max),
        rate,
        tracks,
        missing_media,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn clip_json(url: &str, rate: f64, in_f: f64, dur_f: f64, avail_f: f64, name: &str) -> Value {
        json!({
            "OTIO_SCHEMA": "Clip.2",
            "name": name,
            "source_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": in_f},
                "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": dur_f}
            },
            "markers": [],
            "media_references": {"DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": url,
                "available_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": 0.0},
                    "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": avail_f}
                }
            }},
            "active_media_reference_key": "DEFAULT_MEDIA"
        })
    }

    fn two_track_doc() -> Value {
        json!({
            "OTIO_SCHEMA": "Timeline.1",
            "name": "Demo",
            "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 0.0},
            "tracks": {"OTIO_SCHEMA": "Stack.1", "children": [
                {"OTIO_SCHEMA": "Track.1", "name": "V1", "kind": "Video", "children": [
                    clip_json("file:///media/a.mp4", 60000.0 / 1001.0, 30.0, 240.0, 995.0, "a.mp4"),
                    clip_json("file:///media/b.mp4", 24000.0 / 1001.0, 0.0, 84.0, 1392.0, "b.mp4")
                ]},
                {"OTIO_SCHEMA": "Track.1", "name": "A1", "kind": "Audio", "children": [
                    {"OTIO_SCHEMA": "Gap.1", "name": "", "source_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 0.0},
                        "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 120.0}
                    }},
                    clip_json("file:///media/b.mp4", 24000.0 / 1001.0, 0.0, 84.0, 1392.0, "b.mp4")
                ]}
            ]}
        })
    }

    #[test]
    fn crop_rectangle_maps_to_scale_and_sequence_center() {
        let rect = crate::framing::Rect { x: 100.0, y: 0.0, w: 607.5, h: 1080.0 };
        let (scale, h, v) = crop_motion(rect, 1920.0, 1080.0, 1080.0, 1920.0);
        assert!((scale - 177.7777777778).abs() < 1e-8);
        // After scaling and translating, crop center lands at the output center.
        assert!(((rect.x + rect.w / 2.0 - 960.0) * scale / 100.0 + h * 1080.0).abs() < 1e-8);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn motion_keyframes_and_selected_stereo_sources_survive() {
        use crate::framing::{FramingInput, Keyframe, Rect};
        let input = FramingInput {
            src_w: 1920.0,
            src_h: 1080.0,
            duration: 4.0,
            subjects: vec![],
            turns: vec![],
            content: Rect { x: 0.0, y: 0.0, w: 1920.0, h: 1080.0 },
            origin_s: 0.0,
        };
        let framing =
            serde_json::from_value(json!({"layout":"fixed", "box":{"x":0.1,"y":0.1,"w":0.3,"h":0.8}})).unwrap();
        let mut plan = crate::framing::plan(&input, &framing, 1080.0, 1920.0);
        let path = &mut plan.segments[0].paths[0];
        path.keyframes =
            vec![Keyframe { t: 0.0, x: 100.0, y: 0.0, cut: false }, Keyframe { t: 2.0, x: 300.0, y: 0.0, cut: false }];
        let mut video = clip_json("file:///C:/x/a%20b.mp4", 25.0, 25.0, 100.0, 250.0, "speaker");
        video["metadata"] = json!({"ghostreel":{"frame_plan": plan}});
        video["media_references"]["DEFAULT_MEDIA"]["metadata"] = json!({"ghostreel":{"audio_channels":2}});
        let mut audio = video.clone();
        audio["metadata"] = json!({"ghostreel":{"tracks":[1]}});
        let doc = json!({"OTIO_SCHEMA":"Timeline.1", "metadata":{"width":1080,"height":1920},
            "global_start_time":{"OTIO_SCHEMA":"RationalTime.1","rate":25.0,"value":0.0},
            "tracks":{"children":[{"name":"V1","kind":"Video","children":[video]},
                {"name":"A1","kind":"Audio","children":[audio]}]}});
        let xml = from_otio(&doc).unwrap();
        assert_eq!(xml.matches("<keyframe>").count(), 4);
        assert!(xml.contains("<when>25</when>"));
        assert!(xml.contains("<when>75</when>"));
        assert!(xml.contains("file://localhost/C:/x/a%20b.mp4"));
        assert!(xml.contains("<trackindex>3</trackindex>"));
        assert!(xml.contains("<trackindex>4</trackindex>"));
        assert_eq!(xml.matches("<linkclipref>").count(), 9);
        // A file's own <media><audio> must not make the video track read back as audio.
        let kinds: Vec<String> = summarize_fcp_xml(&xml).unwrap().tracks.into_iter().map(|t| t.kind).collect();
        assert_eq!(kinds.first().map(String::as_str), Some("Video"), "{kinds:?}");
        assert!(kinds.iter().any(|k| k == "Audio"), "{kinds:?}");
        let mut bed = doc.clone();
        bed["tracks"]["children"][1]["children"][0]["metadata"]["ghostreel"]["bed"] = json!(true);
        assert!(!from_otio(&bed).unwrap().contains("<linkclipref>"));
    }

    #[test]
    fn rate_becomes_timebase_and_ntsc_flag() {
        assert_eq!(timebase_ntsc(30.0), (30, false));
        assert_eq!(timebase_ntsc(60000.0 / 1001.0), (60, true));
        assert_eq!(timebase_ntsc(24000.0 / 1001.0), (24, true));
        assert_eq!(timebase_ntsc(25.0), (25, false));
        assert_eq!(timebase_ntsc(23.976), (24, true));
    }

    #[test]
    fn urls_decode_back_to_paths() {
        assert_eq!(path_from_url("file:///a%20b/c.mp4"), "/a b/c.mp4");
        assert_eq!(path_from_url("file:///C:/Users/x%20y/a.mp4"), "C:/Users/x y/a.mp4");
        assert_eq!(path_from_url("file:///media/v%C3%ADdeo.mp4"), "/media/vídeo.mp4");
        assert_eq!(path_from_url("file:///media/test%231.mp4"), "/media/test#1.mp4");
    }

    #[test]
    fn source_times_count_in_source_frames_and_positions_in_sequence_frames() {
        let xml = from_otio(&two_track_doc()).unwrap();

        // 240 frames of 59.94 is 4.004 s, which is 120 frames of a 30 fps sequence.
        assert!(xml.contains("<duration>995</duration>"), "{xml}");
        assert!(xml.contains("<in>30</in>"));
        assert!(xml.contains("<out>270</out>"));
        assert!(xml.contains("<start>0</start>"));
        assert!(xml.contains("<end>120</end>"));

        // The second clip starts where the first ended, and is counted on its own 23.976 clock.
        assert!(xml.contains("<start>120</start>"));
        assert!(xml.contains("<end>225</end>"));
        assert!(xml.contains("<duration>1392</duration>"));
    }

    #[test]
    fn a_gap_moves_the_next_clip_without_being_written() {
        let xml = from_otio(&two_track_doc()).unwrap();
        let audio = xml.rsplit("<audio>").next().unwrap();
        assert!(!audio.contains("<gap"), "gaps are positions, not elements");
        // The audio clip sits after a 120-frame gap, and the sequence is 30 fps.
        assert!(audio.contains("<start>120</start>"), "{audio}");
    }

    #[test]
    fn a_file_is_written_once_and_referenced_after() {
        let xml = from_otio(&two_track_doc()).unwrap();
        // b.mp4 appears on both tracks: spelled out on the first, referenced on the second.
        assert_eq!(xml.matches("<pathurl>file://localhost/media/b.mp4</pathurl>").count(), 1);
        assert!(xml.contains("<file id=\"file-2\"/>"), "{xml}");
    }

    #[test]
    fn empty_tracks_still_take_their_place() {
        let mut doc = two_track_doc();
        doc["tracks"]["children"].as_array_mut().unwrap().push(json!({
            "OTIO_SCHEMA": "Track.1", "name": "V2", "kind": "Video", "children": [
                {"OTIO_SCHEMA": "Gap.1", "name": "", "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 0.0},
                    "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 225.0}
                }}
            ]
        }));
        let xml = from_otio(&doc).unwrap();
        assert!(xml.contains("<track/>"), "{xml}");
    }

    #[test]
    fn markers_carry_the_beat_onto_the_clip() {
        let mut doc = two_track_doc();
        doc["tracks"]["children"][0]["children"][0]["markers"] = json!([{
            "OTIO_SCHEMA": "Marker.2",
            "name": "open: establish the place",
            "marked_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 60000.0 / 1001.0, "value": 30.0},
                "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": 60000.0 / 1001.0, "value": 450.0}
            }
        }]);
        let xml = from_otio(&doc).unwrap();
        assert!(xml.contains("<name>open: establish the place</name>"));
        assert!(xml.contains("<marker>"));
        assert!(xml.contains("<out>480</out>"), "marker out is in + duration: {xml}");
    }

    #[test]
    fn names_with_markup_are_escaped() {
        let mut doc = two_track_doc();
        doc["name"] = json!("Ben & Jerry's <best> \"take\"");
        let xml = from_otio(&doc).unwrap();
        assert!(xml.contains("Ben &amp; Jerry's &lt;best&gt; &quot;take&quot;"), "{xml}");
    }

    #[test]
    fn what_was_written_reads_back_the_same() {
        let doc = two_track_doc();
        let xml = from_otio(&doc).unwrap();

        let from_json = summarize_otio(&doc).unwrap();
        let from_xml = summarize_fcp_xml(&xml).unwrap();

        assert_eq!(from_json.clips, 3);
        assert_eq!(from_xml.clips, from_json.clips);
        assert_eq!(from_xml.tracks.len(), from_json.tracks.len());
        assert!((from_xml.rate - 30.0).abs() < 1e-6);
        assert!(
            (from_xml.duration_s - from_json.duration_s).abs() < 0.05,
            "{} vs {}",
            from_xml.duration_s,
            from_json.duration_s
        );
        // Neither file exists on disk, and both notice.
        assert_eq!(from_json.missing_media.len(), 2);
        assert_eq!(from_xml.missing_media.len(), 2);
    }

    #[test]
    fn a_timeline_we_cannot_write_is_an_error_not_a_wrong_edit() {
        let mut doc = two_track_doc();
        doc["tracks"]["children"][0]["children"][1] = json!({"OTIO_SCHEMA": "Transition.1", "name": "x"});
        assert!(from_otio(&doc).is_err());

        assert!(from_otio(&json!({"OTIO_SCHEMA": "Clip.2"})).is_err());
    }

    /// The older adapter fixture still describes the same edit after adding
    /// Premiere's sequence formats, source descriptions and links.
    #[test]
    fn preserves_the_reference_adapter_edit() {
        let doc: Value = serde_json::from_str(include_str!("../tests/fixtures/golden.otio")).unwrap();
        let actual = summarize_fcp_xml(&from_otio(&doc).unwrap()).unwrap();
        let expected = summarize_fcp_xml(include_str!("../tests/fixtures/golden.xml")).unwrap();
        assert_eq!(actual.clips, expected.clips);
        assert_eq!(actual.duration_s, expected.duration_s);
        assert_eq!(actual.rate, expected.rate);
    }

    #[test]
    fn malformed_xml_is_reported() {
        assert!(summarize_fcp_xml("<xmeml><project>").is_err());
    }

    #[test]
    fn audio_clip_with_gain_emits_audio_levels_filter() {
        let mut doc = two_track_doc();
        // Put gain on V1[0] (must be ignored on Video), +6 dB on A1[1] via AudioVolume effect,
        // and -3 dB on a second A1 clip, plus a no-gain A1 clip.
        doc["tracks"]["children"][0]["children"][0]["metadata"] = json!({"ghostreel": {"gain_db": 6.0}});
        let mut a_plus = clip_json("file:///media/b.mp4", 24000.0 / 1001.0, 0.0, 84.0, 1392.0, "b.mp4");
        a_plus["effects"] = json!([{
            "OTIO_SCHEMA": "Effect.1",
            "name": "AudioVolume",
            "effect_name": "AudioVolume",
            "metadata": {"ghostreel": {"gain_db": 6.0}}
        }]);
        let mut a_minus = clip_json("file:///media/a.mp4", 60000.0 / 1001.0, 0.0, 120.0, 995.0, "a.mp4");
        a_minus["metadata"] = json!({"ghostreel": {"gain_db": -3.0}});
        let a_none = clip_json("file:///media/b.mp4", 24000.0 / 1001.0, 0.0, 84.0, 1392.0, "b.mp4");
        doc["tracks"]["children"][1]["children"] = json!([a_plus, a_minus, a_none]);

        let xml = from_otio(&doc).unwrap();
        let video_section = xml.split("<video>").nth(1).unwrap().split("</video>").next().unwrap();
        assert!(!video_section.contains("<filter>"), "video clipitems must not emit audio filters");

        let audio_section = xml.rsplit("<audio>").next().unwrap();
        assert_eq!(audio_section.matches("<filter>").count(), 2);
        assert!(audio_section.contains("<name>Audio Levels</name>"));
        assert!(audio_section.contains("<effectid>audiolevels</effectid>"));
        assert!(audio_section.contains("<effectcategory>audiolevels</effectcategory>"));
        assert!(audio_section.contains("<effecttype>audiolevels</effecttype>"));
        assert!(audio_section.contains("<mediatype>audio</mediatype>"));
        assert!(audio_section.contains("<parameterid>level</parameterid>"));
        assert!(audio_section.contains("<valuemin>0</valuemin>"));
        assert!(audio_section.contains("<valuemax>3.98109</valuemax>"));
        assert!(audio_section.contains(&format!("<value>{:.6}</value>", 10.0_f64.powf(6.0 / 20.0))));
        assert!(audio_section.contains(&format!("<value>{:.6}</value>", 10.0_f64.powf(-3.0 / 20.0))));
        assert!(summarize_fcp_xml(&xml).is_ok());
    }
}
