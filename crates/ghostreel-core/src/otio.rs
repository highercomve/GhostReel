//! OpenTimelineIO timeline export (plan §4a, D14).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde_json::{Value, json};

use crate::Error;
use crate::db::Db;
use crate::script::{Fps, Script};

/// Resolved media file attributes for timeline generation.
#[derive(Debug, Clone)]
pub struct ResolvedMedia {
    pub video_id: i64,
    pub path: PathBuf,
    pub duration_s: f64,
    pub has_audio: bool,
    pub fps: Fps,
    pub content_hash: String,
    /// The track carrying the speech, measured at index time; 0 when nothing was measured.
    pub audio_track: u32,
}

/// Convert a local file path to a file:// URL with proper percent-encoding.
///
/// Handles both Unix ("/a b/c.mp4" -> "file:///a%20b/c.mp4") and
/// Windows ("C:\\Users\\x y\\a.mp4" -> "file:///C:/Users/x%20y/a.mp4").
///
/// Encodes space (%20), % (%25), # (%23), ? (%3F), and non-ASCII UTF-8 bytes.
pub fn file_url(path: &str) -> String {
    let is_windows = {
        let bytes = path.as_bytes();
        bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
    };

    let normalized = if is_windows { path.replace('\\', "/") } else { path.to_string() };

    let prefix = if is_windows {
        "file:///"
    } else if normalized.starts_with('/') {
        "file://"
    } else {
        "file:///"
    };

    let mut encoded = String::new();
    for b in normalized.as_bytes() {
        match *b {
            b' ' => encoded.push_str("%20"),
            b'%' => encoded.push_str("%25"),
            b'#' => encoded.push_str("%23"),
            b'?' => encoded.push_str("%3F"),
            b if b >= 0x80 => {
                use std::fmt::Write;
                write!(&mut encoded, "%{:02X}", b).unwrap();
            }
            b => encoded.push(b as char),
        }
    }

    format!("{prefix}{encoded}")
}

/// Pure timeline builder: script + resolved media map -> OpenTimelineIO JSON (serde_json::Value).
/// A section card's length in frames of the sequence (0 when the beat has none).
fn card_frames(script: &Script, beat: &crate::script::Beat, seq_rate: f64) -> f64 {
    let s = script.card_s(beat);
    if s > 0.0 { (s * seq_rate).round().max(1.0) } else { 0.0 }
}

/// An OTIO gap `frames` long, with a name, metadata and, when given, a marker over all of it.
fn gap(seq_rate: f64, frames: f64, name: &str, metadata: Value, marker: Option<&str>) -> Value {
    let range = json!({
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": 0.0},
        "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": frames}
    });
    let markers: Vec<Value> = marker
        .map(|text| {
            json!({
            "OTIO_SCHEMA": "Marker.2",
                "comment": "",
            "metadata": {"comment": text},
            "name": text,
            "color": "BLUE",
            "marked_range": range,
                        })
        })
        .into_iter()
        .collect();
    json!({
        "OTIO_SCHEMA": "Gap.1",
        "metadata": metadata,
        "name": name,
        "source_range": range,
        "effects": [],
        "markers": markers,
        "enabled": true,
        "color": null
    })
}

fn active_gain_db(gain_db: Option<f64>) -> Option<f64> {
    crate::script::clamp_gain(gain_db).filter(|g| g.is_finite())
}

fn gain_effects(gain_db: Option<f64>) -> Vec<Value> {
    match gain_db {
        Some(db) => vec![json!({
            "OTIO_SCHEMA": "Effect.1",
            "enabled": true,
            "metadata": {
                "ghostreel": {
                    "gain_db": db
                }
            },
            "name": "AudioVolume",
            "effect_name": "AudioVolume"
        })],
        None => Vec::new(),
    }
}

pub fn build_timeline_pure(
    script: &Script,
    project_fps: Fps,
    project_width: i64,
    project_height: i64,
    media: &HashMap<i64, ResolvedMedia>,
) -> Result<serde_json::Value, Error> {
    // Clamp on the source clock before calculating any track positions. Never fabricate
    // available media to accommodate a bad out point.
    let mut bounded = script.clone();
    for beat in &mut bounded.beats {
        for clip in &mut beat.clips {
            if let Some(res) = media.get(&clip.video_id) {
                (clip.in_s, clip.out_s) = bounded_range(res, clip.in_s, clip.out_s)?;
            }
        }
        if let Some(bed) = &mut beat.bed {
            let res = media
                .get(&bed.video_id)
                .ok_or_else(|| Error::NotFound(format!("resolved media for video #{}", bed.video_id)))?;
            let pictures_s: f64 = beat.clips.iter().map(|c| c.out_s - c.in_s).sum();
            (bed.in_s, bed.out_s) = bounded_range(res, bed.in_s, bed.out_s.min(bed.in_s + pictures_s))?;
        }
    }
    let script = &bounded;
    let seq_fps = script.fps.unwrap_or(project_fps);
    let seq_rate = seq_fps.as_f64();
    let seq_width = script.width.unwrap_or(project_width);
    let seq_height = script.height.unwrap_or(project_height);

    let mut v1_children = Vec::new();
    let mut a1_children = Vec::new();

    for beat in &script.beats {
        let beat_dur_s: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();

        // A section card plays before the beat: a gap on every track, named after its title and
        // marked on the picture track, so an editor sees where to drop a title card.
        let card_frames = card_frames(script, beat, seq_rate);
        if card_frames > 0.0 {
            let card = beat.card.as_ref().expect("a card takes time only when there is one");
            let meta = json!({"ghostreel": {"card": {"text": card.text, "background": script.card_background(beat)}}});
            v1_children.push(gap(seq_rate, card_frames, &card.text, meta, Some(&card.text)));
            a1_children.push(gap(seq_rate, card_frames, "", json!({}), None));
        }

        // A bed is one voice across the whole beat: A1 carries it once, and the pictures above
        // contribute no sound of their own. In an NLE that is a J-cut, audio and video cut apart.
        let mut bed_written = false;
        if let Some(bed) = &beat.bed
            && bed.plays_sound()
            && let Some(res) = media.get(&bed.video_id)
            && res.has_audio
        {
            {
                bed_written = true;
                let rate = res.fps.as_f64();
                let in_frames = (bed.in_s * rate).round();
                let dur_frames = ((bed.out_s - bed.in_s).max(0.0) * rate).round().max(1.0);
                let available = (res.duration_s * rate).floor();
                let name = res.path.file_name().and_then(|n| n.to_str()).unwrap_or("clip.mp4").to_string();
                let range = |start: f64, dur: f64| {
                    json!({
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": start},
                        "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": dur}
                    })
                };
                let bed_gain = if bed.plays_sound() { active_gain_db(bed.gain_db) } else { None };
                let mut bed_ghostreel = json!({"bed": true, "video_id": bed.video_id, "tracks": bed.tracks.clone().unwrap_or_else(|| vec![res.audio_track]), "beat_id": beat.id, "why": bed.why});
                if let Some(db) = bed_gain {
                    bed_ghostreel["gain_db"] = json!(db);
                }
                a1_children.push(json!({
                    "OTIO_SCHEMA": "Clip.2",
                    "metadata": {"ghostreel": bed_ghostreel},
                    "name": name,
                    "source_range": range(in_frames, dur_frames),
                    "effects": gain_effects(bed_gain),
                    "markers": [],
                    "enabled": true,
                    "color": null,
                    "media_references": {"DEFAULT_MEDIA": json!({
                        "OTIO_SCHEMA": "ExternalReference.1",
                        "metadata": {},
                        "name": "",
                        "available_range": range(0.0, available),
                        "available_image_bounds": null,
                        "target_url": file_url(&res.path.to_string_lossy())
                    })},
                    "active_media_reference_key": "DEFAULT_MEDIA"
                }));
                // Any shortfall between the bed and the pictures above it stays silent.
                let bed_s = (bed.out_s - bed.in_s).max(0.0);
                if beat_dur_s > bed_s + 0.001 {
                    let gap_frames = ((beat_dur_s - bed_s) * seq_rate).round().max(1.0);
                    a1_children.push(json!({
                        "OTIO_SCHEMA": "Gap.1",
                        "metadata": {},
                        "name": "",
                        "source_range": {
                            "OTIO_SCHEMA": "TimeRange.1",
                            "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": 0.0},
                            "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": gap_frames}
                        },
                        "effects": [],
                        "markers": [],
                        "enabled": true,
                        "color": null
                    }));
                }
            }
        }

        for (clip_idx, clip) in beat.clips.iter().enumerate() {
            let res = media
                .get(&clip.video_id)
                .ok_or_else(|| Error::NotFound(format!("resolved media for video #{}", clip.video_id)))?;

            let source_rate = res.fps.as_f64();
            let file_name = res.path.file_name().and_then(|n| n.to_str()).unwrap_or("clip.mp4").to_string();
            let target_url = file_url(&res.path.to_string_lossy());

            let in_frames = (clip.in_s * source_rate).round();
            let out_frames = (clip.out_s * source_rate).round();
            let clip_dur_frames = (out_frames - in_frames).max(1.0);
            let available_dur_frames = (res.duration_s * source_rate).floor();

            let mut v1_markers = Vec::new();
            if clip_idx == 0 {
                let beat_dur_frames = (beat_dur_s * source_rate).round().max(1.0);
                let comment = beat.narration.as_deref().or(beat.notes.as_deref()).unwrap_or("");

                v1_markers.push(json!({
                    "OTIO_SCHEMA": "Marker.2",
                "comment": "",
                    "name": format!("{}: {}", beat.id, beat.purpose),
                    "color": "RED",
                    "marked_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": source_rate,
                            "value": in_frames
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": source_rate,
                            "value": beat_dur_frames
                        }
                    },
                    "metadata": {"comment": comment}
                }));
            }

            let external_ref = json!({
                "OTIO_SCHEMA": "ExternalReference.1",
                "metadata": {},
                "name": "",
                "available_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": source_rate,
                        "value": 0.0
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": source_rate,
                        "value": available_dur_frames
                    }
                },
                "available_image_bounds": null,
                "target_url": target_url
            });

            let v1_clip = json!({
                "OTIO_SCHEMA": "Clip.2",
                "metadata": {
                    "ghostreel": {
                        "video_id": clip.video_id,
                        "beat_id": beat.id,
                        "purpose": beat.purpose,
                        "why": clip.why,
                        "notes": beat.notes,
                        "tracks": clip.tracks.clone().unwrap_or_else(|| vec![res.audio_track])
                    }
                },
                "name": file_name,
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": source_rate,
                        "value": in_frames
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": source_rate,
                        "value": clip_dur_frames
                    }
                },
                "effects": [],
                "markers": v1_markers,
                "enabled": true,
                "color": null,
                "media_references": {
                    "DEFAULT_MEDIA": external_ref
                },
                "active_media_reference_key": "DEFAULT_MEDIA"
            });
            v1_children.push(v1_clip);

            // A1 Audio Track mirroring
            if bed_written {
                // The bed already spans this beat on A1; a second copy would double the voice.
            } else if clip.plays_sound() && res.has_audio {
                let clip_gain = active_gain_db(clip.gain_db);
                let a1_metadata = match clip_gain {
                    Some(db) => {
                        json!({"ghostreel": {"gain_db": db, "video_id": clip.video_id, "tracks": clip.tracks.clone().unwrap_or_else(|| vec![res.audio_track])}})
                    }
                    None => {
                        json!({"ghostreel": {"video_id": clip.video_id, "tracks": clip.tracks.clone().unwrap_or_else(|| vec![res.audio_track])}})
                    }
                };
                let a1_clip = json!({
                    "OTIO_SCHEMA": "Clip.2",
                    "metadata": a1_metadata,
                    "name": file_name,
                    "source_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": source_rate,
                            "value": in_frames
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": source_rate,
                            "value": clip_dur_frames
                        }
                    },
                    "effects": gain_effects(clip_gain),
                    "markers": [],
                    "enabled": true,
                    "color": null,
                    "media_references": {
                        "DEFAULT_MEDIA": external_ref
                    },
                    "active_media_reference_key": "DEFAULT_MEDIA"
                });
                a1_children.push(a1_clip);
            } else {
                // Preserve exactly the picture duration, including mixed-rate fractions.
                let a1_gap = gap(source_rate, clip_dur_frames, "", json!({}), None);
                a1_children.push(a1_gap);
            }
        }
    }

    let v1_track = json!({
        "OTIO_SCHEMA": "Track.1",
        "metadata": {},
        "name": "V1",
        "source_range": null,
        "effects": [],
        "markers": [],
        "enabled": true,
        "color": null,
        "children": v1_children,
        "kind": "Video"
    });

    let a1_track = json!({
        "OTIO_SCHEMA": "Track.1",
        "metadata": {},
        "name": "A1",
        "source_range": null,
        "effects": [],
        "markers": [],
        "enabled": true,
        "color": null,
        "children": a1_children,
        "kind": "Audio"
    });

    let mut tracks = vec![v1_track, a1_track];

    // Track V2: only if any beat has on_screen_text
    let has_v2 = script.beats.iter().any(|b| b.on_screen_text.as_deref().is_some_and(|t| !t.trim().is_empty()));

    if has_v2 {
        let mut v2_children = Vec::new();
        for beat in &script.beats {
            let beat_dur_s: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();
            let beat_frames = (beat_dur_s * seq_rate).round().max(1.0);
            let card_frames = card_frames(script, beat, seq_rate);
            if card_frames > 0.0 {
                v2_children.push(gap(seq_rate, card_frames, "", json!({}), None));
            }

            if let Some(text) = beat.on_screen_text.as_deref().filter(|t| !t.trim().is_empty()) {
                v2_children.push(json!({
                    "OTIO_SCHEMA": "Gap.1",
                    "metadata": {
                        "ghostreel": {
                            "title": text
                        }
                    },
                    "name": "",
                    "source_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": seq_rate,
                            "value": 0.0
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": seq_rate,
                            "value": beat_frames
                        }
                    },
                    "effects": [],
                    "markers": [
                        {
                            "OTIO_SCHEMA": "Marker.2",
                "comment": "",
                            "metadata": {},
                            "name": text,
                            "color": "GREEN",
                            "marked_range": {
                                "OTIO_SCHEMA": "TimeRange.1",
                                "start_time": {
                                    "OTIO_SCHEMA": "RationalTime.1",
                                    "rate": seq_rate,
                                    "value": 0.0
                                },
                                "duration": {
                                    "OTIO_SCHEMA": "RationalTime.1",
                                    "rate": seq_rate,
                                    "value": beat_frames
                                }
                            },
                                                    }
                    ],
                    "enabled": true,
                    "color": null
                }));
            } else {
                v2_children.push(json!({
                    "OTIO_SCHEMA": "Gap.1",
                    "metadata": {},
                    "name": "",
                    "source_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": seq_rate,
                            "value": 0.0
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "rate": seq_rate,
                            "value": beat_frames
                        }
                    },
                    "effects": [],
                    "markers": [],
                    "enabled": true,
                    "color": null
                }));
            }
        }

        let v2_track = json!({
            "OTIO_SCHEMA": "Track.1",
            "metadata": {},
            "name": "V2",
            "source_range": null,
            "effects": [],
            "markers": [],
            "enabled": true,
            "color": null,
            "children": v2_children,
            "kind": "Video"
        });
        tracks.push(v2_track);
    }

    let mut timeline = json!({
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {
            "width": seq_width,
            "height": seq_height,
            "ghostreel": {
                "schema": 1,
                "width": seq_width,
                "height": seq_height,
                "fps": {
                    "num": seq_fps.num,
                    "den": seq_fps.den
                }
            }
        },
        "name": script.title,
        "global_start_time": {
            "OTIO_SCHEMA": "RationalTime.1",
            "rate": seq_rate,
            "value": 0.0
        },
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "metadata": {},
            "name": "tracks",
            "source_range": null,
            "effects": [],
            "markers": [],
            "enabled": true,
            "color": null,
            "children": tracks
        }
    });

    let mut sequence_markers = Vec::new();
    let mut at = 0.0;
    for beat in &script.beats {
        at += card_frames(script, beat, seq_rate);
        let duration = beat.clips.iter().map(|c| c.out_s - c.in_s).sum::<f64>() * seq_rate;
        sequence_markers.push(json!({
            "OTIO_SCHEMA": "Marker.2",
                "comment": "", "metadata": {},
            "name": format!("{}: {}", beat.id, beat.purpose), "color": "RED",
            "marked_range": {"OTIO_SCHEMA": "TimeRange.1",
                "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": at.round()},
                "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": seq_rate, "value": duration.round()}}
        }));
        at += duration;
    }
    timeline["tracks"]["markers"] = json!(sequence_markers);
    for track in timeline["tracks"]["children"].as_array_mut().unwrap() {
        for child in track["children"].as_array_mut().unwrap() {
            let Some(id) = child["metadata"]["ghostreel"]["video_id"].as_i64() else { continue };
            let res = &media[&id];
            child["media_references"]["DEFAULT_MEDIA"]["metadata"] = json!({"ghostreel": {
                "has_audio": res.has_audio, "audio_channels": 2
            }});
        }
    }
    Ok(timeline)
}

fn bounded_range(res: &ResolvedMedia, start: f64, end: f64) -> Result<(f64, f64), Error> {
    let rate = res.fps.as_f64();
    let available = (res.duration_s * rate).floor();
    let start = (start.max(0.0) * rate).round().min(available);
    let end = (end * rate).round().min(available);
    if !available.is_finite() || end <= start {
        return Err(Error::Export(format!("video #{} has no available frames in requested range", res.video_id)));
    }
    Ok((start / rate, end / rate))
}

/// Resolve video metadata and file paths from the database for all clips in the script.
pub fn resolve_media_for_script(
    db: &Db,
    project_id: i64,
    script: &Script,
    project_fps: Fps,
) -> Result<HashMap<i64, ResolvedMedia>, Error> {
    let mut resolved_map = HashMap::new();

    // Pictures and beds alike: a J-cut's voice usually comes from an interview that never
    // appears on screen, and an unresolved bed used to be dropped without a word.
    let video_ids = script
        .beats
        .iter()
        .flat_map(|b| b.clips.iter().map(|c| c.video_id).chain(b.bed.iter().map(|bed| bed.video_id)));
    for video_id in video_ids {
        {
            if resolved_map.contains_key(&video_id) {
                continue;
            }

            // Query paths belonging to project folders
            let mut st = db.conn.prepare(
                "SELECT vf.path, v.duration_s, v.has_audio, v.fps, v.content_hash
                 FROM video_files vf
                 JOIN folders f ON f.id = vf.folder_id
                 JOIN project_folders pf ON pf.folder_id = f.id
                 JOIN videos v ON v.id = vf.video_id
                 WHERE pf.project_id = ?1 AND v.id = ?2
                   AND NOT EXISTS (SELECT 1 FROM project_exclusions x WHERE x.project_id = pf.project_id AND x.video_id = vf.video_id)",
            )?;

            let rows = st.query_map(params![project_id, video_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<f64>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, Option<f64>>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })?;

            let mut candidates = Vec::new();
            for r in rows {
                candidates.push(r?);
            }

            if candidates.is_empty() {
                return Err(Error::NotFound(format!("a file for video #{} in this project's folders", video_id)));
            }

            // Prefer a path that exists on disk
            let chosen = candidates.iter().find(|(p, _, _, _, _)| Path::new(p).exists()).unwrap_or(&candidates[0]);

            let path = PathBuf::from(&chosen.0);
            let duration_s = chosen.1.unwrap_or(0.0);
            let has_audio = chosen.2.unwrap_or(0) != 0;
            let fps = match chosen.3 {
                Some(f) if f > 0.0 => Fps::from_f64(f),
                _ => project_fps,
            };
            let content_hash = chosen.4.clone();

            resolved_map.insert(
                video_id,
                ResolvedMedia {
                    video_id,
                    path,
                    duration_s,
                    has_audio,
                    fps,
                    content_hash,
                    audio_track: db.audio_track(video_id).unwrap_or(0),
                },
            );
        }
    }

    Ok(resolved_map)
}

/// Build an OpenTimelineIO timeline for a project and script.
pub fn build_timeline(db: &Db, project_id: i64, script: &Script) -> Result<serde_json::Value, Error> {
    let project = db.project(project_id)?;
    let project_fps = Fps::new(project.fps_num, project.fps_den);
    let media = resolve_media_for_script(db, project_id, script, project_fps)?;
    build_timeline_from(db, project_id, script, &media, &|_| 0.0)
}

/// [`build_timeline`] over media the caller resolved: each key of `media` is a `video_id` the
/// script's clips name, and the entry's own `video_id` is the indexed video it was cut from
/// (the same for plain exports; a range file for a package). `source_start_s(key)` is where that
/// media starts inside the indexed video, so source sizes and framing plans come from the index.
pub fn build_timeline_from(
    db: &Db,
    project_id: i64,
    script: &Script,
    media: &HashMap<i64, ResolvedMedia>,
    source_start_s: &dyn Fn(i64) -> f64,
) -> Result<serde_json::Value, Error> {
    let project = db.project(project_id)?;
    let project_fps = Fps::new(project.fps_num, project.fps_den);
    let mut timeline = build_timeline_pure(script, project_fps, project.width, project.height, media)?;
    let width = script.width.unwrap_or(project.width) as f64;
    let height = script.height.unwrap_or(project.height) as f64;
    for track in timeline["tracks"]["children"].as_array_mut().unwrap() {
        let picture = track["kind"] == "Video";
        for child in track["children"].as_array_mut().unwrap() {
            let Some(id) = child["metadata"]["ghostreel"]["video_id"].as_i64() else { continue };
            let Some(res) = media.get(&id) else { continue };
            let (source_id, offset_s) = (res.video_id, source_start_s(id));
            let (w, h): (Option<i64>, Option<i64>) =
                db.conn.query_row("SELECT width, height FROM videos WHERE id = ?1", [source_id], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?;
            let meta = &mut child["media_references"]["DEFAULT_MEDIA"]["metadata"]["ghostreel"];
            meta["width"] = json!(w.unwrap_or(0));
            meta["height"] = json!(h.unwrap_or(0));
            if picture {
                let beat_id = child["metadata"]["ghostreel"]["beat_id"].as_str().unwrap_or("");
                let start_s = child["source_range"]["start_time"]["value"].as_f64().unwrap() / media[&id].fps.as_f64();
                let duration_s = child["source_range"]["duration"]["value"].as_f64().unwrap() / media[&id].fps.as_f64();
                let default_framing =
                    crate::framing::Framing { layout: crate::framing::Layout::Fit, ..Default::default() };
                if let Some(clip) = script
                    .beats
                    .iter()
                    .find(|b| b.id == beat_id)
                    .and_then(|b| b.clips.iter().find(|c| c.video_id == id && (c.in_s - start_s).abs() < 0.1))
                    && let fr = crate::framing::effective(script, clip).unwrap_or(&default_framing)
                    && let Some(plan) = crate::framing::plan_clip(
                        db,
                        None,
                        None,
                        source_id,
                        offset_s + start_s,
                        offset_s + start_s + duration_s,
                        fr,
                        width,
                        height,
                    )?
                {
                    child["metadata"]["ghostreel"]["frame_plan"] =
                        serde_json::to_value(plan).map_err(|e| Error::Export(e.to_string()))?;
                }
            }
        }
    }
    Ok(timeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_url_unix_and_windows() {
        assert_eq!(file_url("/a b/c.mp4"), "file:///a%20b/c.mp4");
        assert_eq!(file_url("C:\\Users\\x y\\a.mp4"), "file:///C:/Users/x%20y/a.mp4");
        assert_eq!(file_url("C:/Users/x y/a.mp4"), "file:///C:/Users/x%20y/a.mp4");
        assert_eq!(file_url("/media/test#1?v=2%20.mp4"), "file:///media/test%231%3Fv=2%2520.mp4");
        // UTF-8 non-ASCII
        assert_eq!(file_url("/media/vídeo.mp4"), "file:///media/v%C3%ADdeo.mp4");
    }

    #[test]
    fn rational_rates() {
        assert_eq!(Fps::from_f64(25.0), Fps::new(25, 1));
        assert_eq!(Fps::from_f64(29.97), Fps::new(30000, 1001));
        assert_eq!(Fps::from_f64(23.976), Fps::new(24000, 1001));
        assert_eq!(Fps::from_f64(59.94), Fps::new(60000, 1001));
        assert_eq!(Fps::from_f64(60.0), Fps::new(60, 1));
    }

    #[test]
    fn golden_test_timeline() {
        let script_str = include_str!("../tests/fixtures/golden_script.json");
        let script: Script = serde_json::from_str(script_str).unwrap();

        let mut media = HashMap::new();
        media.insert(
            1,
            ResolvedMedia {
                video_id: 1,
                path: PathBuf::from("/media/intro.mp4"),
                duration_s: 10.0,
                has_audio: true,
                fps: Fps::new(25, 1),
                content_hash: "hash1".into(),
                audio_track: 0,
            },
        );
        media.insert(
            2,
            ResolvedMedia {
                video_id: 2,
                path: PathBuf::from("/media/broll.mp4"),
                duration_s: 5.0,
                has_audio: false,
                fps: Fps::new(30000, 1001),
                content_hash: "hash2".into(),
                audio_track: 0,
            },
        );

        let timeline = build_timeline_pure(&script, Fps::new(25, 1), 1920, 1080, &media).unwrap();

        let expected_str = include_str!("../tests/fixtures/golden.otio");
        let expected: serde_json::Value = serde_json::from_str(expected_str).unwrap();

        assert_eq!(timeline["name"], expected["name"]);
        for (actual, old) in timeline["tracks"]["children"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["tracks"]["children"].as_array().unwrap())
        {
            assert_eq!(actual["name"], old["name"]);
            for (a, b) in actual["children"].as_array().unwrap().iter().zip(old["children"].as_array().unwrap()) {
                for key in ["start_time", "duration"] {
                    let seconds = |v: &Value| {
                        v["source_range"][key]["value"].as_f64().unwrap()
                            / v["source_range"][key]["rate"].as_f64().unwrap()
                    };
                    assert!((seconds(a) - seconds(b)).abs() < 0.01);
                }
                assert_eq!(a["name"], b["name"]);
            }
        }
    }

    #[test]
    fn clip_and_bed_gain_are_written_when_played_and_omitted_when_silent_or_unset() {
        use crate::script::{Audio, AudioBed, Beat, ScriptClip};

        let script = Script {
            title: "Gain Timeline".into(),
            target_duration_s: None,
            fps: Some(Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            framing: None,
            captions: None,
            title_style: None,
            card_style: None,
            caption_edits: Vec::new(),
            agents: None,
            beats: vec![
                Beat {
                    id: "b1".into(),
                    purpose: "speak".into(),
                    narration: None,
                    on_screen_text: None,
                    title_layout: None,
                    notes: None,
                    card: None,
                    bed: None,
                    clips: vec![
                        ScriptClip {
                            video_id: 1,
                            in_s: 0.0,
                            out_s: 2.0,
                            audio: Audio::Source,
                            why: None,
                            tracks: None,
                            framing: None,
                            sentences: None,
                            exact: false,
                            gain_db: Some(6.0),
                        },
                        ScriptClip {
                            video_id: 1,
                            in_s: 2.0,
                            out_s: 4.0,
                            audio: Audio::Mute,
                            why: None,
                            tracks: None,
                            framing: None,
                            sentences: None,
                            exact: false,
                            gain_db: Some(9.0),
                        },
                        ScriptClip {
                            video_id: 1,
                            in_s: 4.0,
                            out_s: 6.0,
                            audio: Audio::Source,
                            why: None,
                            tracks: None,
                            framing: None,
                            sentences: None,
                            exact: false,
                            gain_db: None,
                        },
                    ],
                },
                Beat {
                    id: "b2".into(),
                    purpose: "broll".into(),
                    narration: None,
                    on_screen_text: None,
                    title_layout: None,
                    notes: None,
                    card: None,
                    bed: Some(AudioBed {
                        video_id: 2,
                        in_s: 0.0,
                        out_s: 2.0,
                        why: None,
                        inferred: false,
                        tracks: None,
                        gain_db: Some(-3.0),
                        sentences: None,
                    }),
                    clips: vec![ScriptClip {
                        video_id: 1,
                        in_s: 6.0,
                        out_s: 8.0,
                        audio: Audio::Mute,
                        why: None,
                        tracks: None,
                        framing: None,
                        sentences: None,
                        exact: false,
                        gain_db: Some(4.0),
                    }],
                },
            ],
        };

        let mut media = HashMap::new();
        for (id, path) in [(1, "/media/speak.mp4"), (2, "/media/bed.mp4")] {
            media.insert(
                id,
                ResolvedMedia {
                    video_id: id,
                    path: PathBuf::from(path),
                    duration_s: 10.0,
                    has_audio: true,
                    fps: Fps::new(25, 1),
                    content_hash: format!("h{id}"),
                    audio_track: 0,
                },
            );
        }

        let tl = build_timeline_pure(&script, Fps::new(25, 1), 1920, 1080, &media).unwrap();
        let v1 = tl["tracks"]["children"][0]["children"].as_array().unwrap();
        let a1 = tl["tracks"]["children"][1]["children"].as_array().unwrap();

        // Video clips never carry audio gain effects.
        for v in v1 {
            assert_eq!(v["effects"].as_array().unwrap().len(), 0);
            assert!(v["metadata"]["ghostreel"].get("gain_db").is_none());
        }

        // A1[0]: spoken clip with +6 dB.
        assert_eq!(a1[0]["effects"].as_array().unwrap().len(), 1);
        assert_eq!(a1[0]["effects"][0]["OTIO_SCHEMA"], "Effect.1");
        assert_eq!(a1[0]["effects"][0]["effect_name"], "AudioVolume");
        assert_eq!(a1[0]["effects"][0]["metadata"]["ghostreel"]["gain_db"], 6.0);
        assert_eq!(a1[0]["metadata"]["ghostreel"]["gain_db"], 6.0);

        // A1[1]: muted picture clip becomes a Gap (carries no gain).
        assert_eq!(a1[1]["OTIO_SCHEMA"], "Gap.1");

        // A1[2]: spoken clip with no gain emits no effect or gain metadata.
        assert_eq!(a1[2]["effects"].as_array().unwrap().len(), 0);
        assert!(a1[2]["metadata"]["ghostreel"].get("gain_db").is_none());

        // A1[3]: bed with -3 dB.
        assert_eq!(a1[3]["effects"].as_array().unwrap().len(), 1);
        assert_eq!(a1[3]["effects"][0]["effect_name"], "AudioVolume");
        assert_eq!(a1[3]["effects"][0]["metadata"]["ghostreel"]["gain_db"], -3.0);
        assert_eq!(a1[3]["metadata"]["ghostreel"]["gain_db"], -3.0);
    }
}

#[cfg(test)]
mod bed_media_tests {
    use super::*;
    use crate::projects::NewProject;
    use rusqlite::params;

    /// A J-cut's voice often comes from an interview that is never on screen: its file must be
    /// resolved too, or the render drops the bed and the beat plays silent.
    #[test]
    fn resolves_bed_sources_that_are_not_pictures() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("beds")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        for (id, name) in [(1, "broll.mp4"), (2, "interview.mp4")] {
            let path = dir.path().join(name);
            std::fs::write(&path, b"x").unwrap();
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s, fps, vfr) VALUES (?1, ?2, 1, 60.0, 30.0, 0)",
                    params![id, format!("h{id}")],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, path.to_str().unwrap()],
                )
                .unwrap();
        }
        let script: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{
                "id": "b1",
                "purpose": "montage",
                "clips": [{ "video_id": 1, "in_s": 0.0, "out_s": 4.0, "audio": "mute" }],
                "bed": { "video_id": 2, "in_s": 10.0, "out_s": 14.0 }
            }]
        }))
        .unwrap();
        let media = resolve_media_for_script(&db, project.id, &script, Fps::new(30, 1)).unwrap();
        assert!(media.contains_key(&1), "picture source resolved");
        assert!(media.contains_key(&2), "bed source resolved");
    }
}

#[cfg(test)]
mod premiere_tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn portrait_fixture_has_premiere_formats_motion_beds_links_and_bounded_ranges() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("Premiere portrait")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, dir.path(), true).unwrap();
        for (id, name, audio) in [(1, "follow speaker.mp4", 1), (2, "muted broll.mp4", 0), (3, "bed.wav", 1)] {
            let path = dir.path().join(name);
            std::fs::write(&path, b"fixture").unwrap();
            db.conn
                .execute(
                    "INSERT INTO videos(id, content_hash, size, duration_s, fps, width, height, has_audio)
                VALUES (?1, ?2, 1, 10, 25, ?3, ?4, ?5)",
                    params![
                        id,
                        format!("premiere-{id}"),
                        if id == 3 { 0 } else { 1920 },
                        if id == 3 { 0 } else { 1080 },
                        audio
                    ],
                )
                .unwrap();
            db.conn
                .execute(
                    "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                VALUES (?1, ?2, ?3, 1, 0, 0)",
                    params![id, folder.id, path.to_str().unwrap()],
                )
                .unwrap();
        }
        let script: Script = serde_json::from_value(json!({
            "title": "Premiere portrait", "width": 1080, "height": 1920, "fps": {"num":25,"den":1},
            "framing": "follow_speaker",
            "beats": [
                {"id":"speech", "purpose":"speaker", "on_screen_text":"Opening", "card":{"text":"Title","duration_s":1},
                 "clips":[{"video_id":1,"in_s":1,"out_s":3,"gain_db":6,"tracks":[0]}]},
                {"id":"bed", "purpose":"montage", "bed":{"video_id":3,"in_s":2,"out_s":4,"gain_db":-3},
                 "clips":[{"video_id":2,"in_s":9,"out_s":12,"audio":"mute"}]}
            ]
        }))
        .unwrap();
        let timeline = build_timeline(&db, project.id, &script).unwrap();
        let xml = crate::fcpxml::from_otio(&timeline).unwrap();
        assert_eq!(timeline["metadata"]["width"], 1080);
        assert_eq!(timeline["metadata"]["height"], 1920);
        assert_eq!(timeline["tracks"]["markers"].as_array().unwrap().len(), 2);
        assert!(xml.contains("<width>1080</width>"));
        assert!(xml.contains("<height>1920</height>"));
        assert!(xml.contains("<width>1920</width>"));
        assert!(xml.contains("<effectid>basic</effectid>"));
        assert!(xml.contains("<effectid>audiolevels</effectid>"));
        assert!(xml.contains("<linkclipref>"));
        assert!(xml.contains("<sourcetrack>"));
        assert!(xml.contains("<samplerate>48000</samplerate>"));
        assert!(xml.contains("file://localhost/"));
        assert!(xml.contains("follow%20speaker.mp4"));
        for track in timeline["tracks"]["children"].as_array().unwrap() {
            for clip in track["children"].as_array().unwrap().iter().filter(|c| c["OTIO_SCHEMA"] == "Clip.2") {
                let start = clip["source_range"]["start_time"]["value"].as_f64().unwrap();
                let duration = clip["source_range"]["duration"]["value"].as_f64().unwrap();
                let available =
                    clip["media_references"]["DEFAULT_MEDIA"]["available_range"]["duration"]["value"].as_f64().unwrap();
                assert!(start + duration <= available, "source out is clamped to available media");
            }
        }
        // Optional artifacts for running the real Python adapters outside the repository.
        if let Some(output) = std::env::var_os("GHOSTREEL_OTIO_VALIDATION_DIR") {
            std::fs::create_dir_all(&output).unwrap();
            let output = PathBuf::from(output);
            std::fs::write(output.join("portrait.otio"), serde_json::to_vec_pretty(&timeline).unwrap()).unwrap();
            std::fs::write(output.join("portrait.xml"), xml).unwrap();
        }
    }

    #[test]
    fn invalid_empty_source_range_is_rejected() {
        let media = ResolvedMedia {
            video_id: 1,
            path: "x.mp4".into(),
            duration_s: 1.0,
            has_audio: true,
            fps: Fps::new(25, 1),
            content_hash: "x".into(),
            audio_track: 0,
        };
        assert!(bounded_range(&media, 2.0, 3.0).is_err());
        assert_eq!(bounded_range(&media, 0.5, 4.0).unwrap(), (0.52, 1.0));
    }
}
