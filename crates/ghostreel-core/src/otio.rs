//! OpenTimelineIO timeline export (plan §4a, D14).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde_json::json;

use crate::Error;
use crate::db::Db;
use crate::script::{Audio, Fps, Script};

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
pub fn build_timeline_pure(
    script: &Script,
    project_fps: Fps,
    project_width: i64,
    project_height: i64,
    media: &HashMap<i64, ResolvedMedia>,
) -> Result<serde_json::Value, Error> {
    let seq_fps = script.fps.unwrap_or(project_fps);
    let seq_rate = seq_fps.as_f64();
    let seq_width = script.width.unwrap_or(project_width);
    let seq_height = script.height.unwrap_or(project_height);

    let mut v1_children = Vec::new();
    let mut a1_children = Vec::new();

    for beat in &script.beats {
        let beat_dur_s: f64 = beat.clips.iter().map(|c| (c.out_s - c.in_s).max(0.0)).sum();

        // A bed is one voice across the whole beat: A1 carries it once, and the pictures above
        // contribute no sound of their own. In an NLE that is a J-cut, audio and video cut apart.
        let mut bed_written = false;
        if let Some(bed) = &beat.bed
            && let Some(res) = media.get(&bed.video_id)
        {
            {
                bed_written = true;
                let rate = res.fps.as_f64();
                let in_frames = (bed.in_s * rate).round();
                let dur_frames = ((bed.out_s - bed.in_s).max(0.0) * rate).round().max(1.0);
                let available = (res.duration_s * rate).round().max(dur_frames);
                let name = res.path.file_name().and_then(|n| n.to_str()).unwrap_or("clip.mp4").to_string();
                let range = |start: f64, dur: f64| {
                    json!({
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": start},
                        "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": dur}
                    })
                };
                a1_children.push(json!({
                    "OTIO_SCHEMA": "Clip.2",
                    "metadata": {"ghostreel": {"bed": true, "beat_id": beat.id, "why": bed.why}},
                    "name": name,
                    "source_range": range(in_frames, dur_frames),
                    "effects": [],
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
            let available_dur_frames = (res.duration_s * source_rate).round().max(clip_dur_frames);

            let mut v1_markers = Vec::new();
            if clip_idx == 0 {
                let beat_dur_frames = (beat_dur_s * source_rate).round().max(1.0);
                let comment = beat.narration.as_deref().or(beat.notes.as_deref()).unwrap_or("");

                v1_markers.push(json!({
                    "OTIO_SCHEMA": "Marker.2",
                    "metadata": {},
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
                    "comment": comment
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
                        "notes": beat.notes
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
            } else if clip.audio == Audio::Source && res.has_audio {
                let a1_clip = json!({
                    "OTIO_SCHEMA": "Clip.2",
                    "metadata": {},
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
                let dur_s = (clip.out_s - clip.in_s).max(0.0);
                let gap_frames = (dur_s * seq_rate).round().max(1.0);
                let a1_gap = json!({
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
                            "value": gap_frames
                        }
                    },
                    "effects": [],
                    "markers": [],
                    "enabled": true,
                    "color": null
                });
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
                            "comment": text
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

    let timeline = json!({
        "OTIO_SCHEMA": "Timeline.1",
        "metadata": {
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

    Ok(timeline)
}

/// Resolve video metadata and file paths from the database for all clips in the script.
pub fn resolve_media_for_script(
    db: &Db,
    project_id: i64,
    script: &Script,
    project_fps: Fps,
) -> Result<HashMap<i64, ResolvedMedia>, Error> {
    let mut resolved_map = HashMap::new();

    for beat in &script.beats {
        for clip in &beat.clips {
            if resolved_map.contains_key(&clip.video_id) {
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

            let rows = st.query_map(params![project_id, clip.video_id], |r| {
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
                return Err(Error::NotFound(format!("a file for video #{} in this project's folders", clip.video_id)));
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
                clip.video_id,
                ResolvedMedia {
                    video_id: clip.video_id,
                    path,
                    duration_s,
                    has_audio,
                    fps,
                    content_hash,
                    audio_track: db.audio_track(clip.video_id).unwrap_or(0),
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
    build_timeline_pure(script, project_fps, project.width, project.height, &media)
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

        assert_eq!(timeline, expected);
    }
}
