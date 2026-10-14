//! Automatic video categorization (Interview, B-Roll, Screen Recording, Drone, Audio, Other).
//!
//! Uses multimodal signals gathered during indexing: Whisper transcript segments, speech ratio,
//! interviewer detection (off-mic), frame descriptions, shot types, detected objects, and camera steadiness.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::vision::FrameDescription;

/// Canonical category identifiers.
pub const CAT_INTERVIEW: &str = "interview";
pub const CAT_BROLL: &str = "broll";
pub const CAT_SCREENCAST: &str = "screencast";
pub const CAT_DRONE: &str = "drone";
pub const CAT_AUDIO: &str = "audio";
pub const CAT_OTHER: &str = "other";

/// Standard directory name on the filesystem for a given category.
pub fn folder_for_category(category: &str) -> String {
    match category.to_ascii_lowercase().as_str() {
        CAT_INTERVIEW => "Interviews".to_string(),
        CAT_BROLL => "B-Roll".to_string(),
        CAT_SCREENCAST => "Screen Recordings".to_string(),
        CAT_DRONE => "Drones & Aerials".to_string(),
        CAT_AUDIO => "Music & Audio".to_string(),
        CAT_OTHER => "Other".to_string(),
        custom => {
            // Capitalize custom category name for clean folder naming
            let mut c = custom.chars();
            match c.next() {
                None => "Other".to_string(),
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
            }
        }
    }
}

/// Human-readable display label for a category.
pub fn category_label(category: &str) -> String {
    match category.to_ascii_lowercase().as_str() {
        CAT_INTERVIEW => "Interview".to_string(),
        CAT_BROLL => "B-Roll".to_string(),
        CAT_SCREENCAST => "Screen Recording".to_string(),
        CAT_DRONE => "Drone / Aerial".to_string(),
        CAT_AUDIO => "Audio / Music".to_string(),
        CAT_OTHER => "Other".to_string(),
        custom => custom.to_string(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub video_id: i64,
    pub category: String,
    pub confidence: f64,
    pub source: String,
    pub scores: HashMap<String, f64>,
    pub tag: Option<String>,
}

/// Classify a single video based on its indexed transcript and frame descriptions.
pub fn classify_video(db: &Db, video_id: i64) -> Result<ClassificationResult, Error> {
    // 1. Check if category was manually set
    let (existing_cat, existing_conf, existing_source, existing_tag) = db.video_category(video_id)?;
    if existing_source == "manual" {
        return Ok(ClassificationResult {
            video_id,
            category: existing_cat,
            confidence: existing_conf.max(1.0),
            source: "manual".to_string(),
            scores: HashMap::new(),
            tag: existing_tag,
        });
    }

    // 2. Query video properties
    let (duration_s, has_audio, vcodec): (Option<f64>, Option<bool>, Option<String>) = db
        .conn
        .query_row("SELECT duration_s, has_audio, vcodec FROM videos WHERE id = ?1", [video_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .map_err(Error::from)?;

    let dur = duration_s.unwrap_or(0.0);

    // Audio-only check
    if (vcodec.is_none() || vcodec.as_deref() == Some("")) && has_audio.unwrap_or(false) {
        return Ok(ClassificationResult {
            video_id,
            category: CAT_AUDIO.to_string(),
            confidence: 0.95,
            source: "auto".to_string(),
            scores: HashMap::from([(CAT_AUDIO.to_string(), 1.0)]),
            tag: None,
        });
    }

    // 3. Transcript analysis: speech duration, off-mic count, word counts
    let mut st_segs = db
        .conn
        .prepare("SELECT start_s, end_s, text, COALESCE(off_mic, 0) FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s")?;
    let segs: Vec<(f64, f64, String, i64)> = st_segs
        .query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<_, _>>()?;

    let speech_seconds: f64 = segs.iter().map(|(s, e, _, _)| (e - s).max(0.0)).sum();
    let speech_ratio = if dur > 0.5 { (speech_seconds / dur).clamp(0.0, 1.0) } else { 0.0 };
    let off_mic_count = segs.iter().filter(|(_, _, _, off)| *off != 0).count();
    let total_words: usize = segs.iter().map(|(_, _, t, _)| t.split_whitespace().count()).sum();

    // 4. Frames analysis: shot types, tags, objects, descriptions
    let mut st_frames =
        db.conn.prepare("SELECT description_json FROM frames WHERE video_id = ?1 AND description_json IS NOT NULL")?;
    let desc_jsons: Vec<String> =
        st_frames.query_map([video_id], |r| r.get::<_, String>(0))?.collect::<Result<_, _>>()?;

    let mut shot_counts: HashMap<String, usize> = HashMap::new();
    let mut tag_counts: HashMap<String, usize> = HashMap::new();
    let mut total_described_frames = 0usize;

    for dj in &desc_jsons {
        if let Ok(desc) = serde_json::from_str::<FrameDescription>(dj) {
            total_described_frames += 1;
            *shot_counts.entry(desc.shot.to_ascii_lowercase()).or_default() += 1;
            for t in &desc.tags {
                *tag_counts.entry(t.to_ascii_lowercase()).or_default() += 1;
            }
            for o in &desc.objects {
                *tag_counts.entry(o.to_ascii_lowercase()).or_default() += 1;
            }
            // Check description text for key indicator words
            let text_lower = desc.description.to_ascii_lowercase();
            if text_lower.contains("screen recording")
                || text_lower.contains("source code")
                || text_lower.contains("computer screen")
                || text_lower.contains("slide presentation")
            {
                *tag_counts.entry("screencast_hint".to_string()).or_default() += 1;
            }
            if text_lower.contains("drone") || text_lower.contains("aerial") || text_lower.contains("bird's-eye") {
                *tag_counts.entry("drone_hint".to_string()).or_default() += 1;
            }
            if text_lower.contains("interview") || text_lower.contains("speaking to camera") {
                *tag_counts.entry("interview_hint".to_string()).or_default() += 1;
            }
            if text_lower.contains("speaking into a microphone")
                || text_lower.contains("holding a microphone")
                || text_lower.contains("addressing a group")
                || text_lower.contains("addressing a small group")
                || text_lower.contains("presentation")
                || text_lower.contains("giving a speech")
                || text_lower.contains("speaking to an audience")
            {
                *tag_counts.entry("presentation_hint".to_string()).or_default() += 1;
            }
            if text_lower.contains("singing")
                || text_lower.contains("playing guitar")
                || text_lower.contains("playing an acoustic guitar")
                || text_lower.contains("playing electric guitar")
                || text_lower.contains("live music")
                || text_lower.contains("musical performance")
            {
                *tag_counts.entry("music_hint".to_string()).or_default() += 1;
            }
            if text_lower.contains("building")
                || text_lower.contains("scenery")
                || text_lower.contains("landscape")
                || text_lower.contains("exterior")
            {
                *tag_counts.entry("broll_hint".to_string()).or_default() += 1;
            }
        }
    }

    // 5. Score calculations
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut interview_score = 0.0;
    let mut broll_score = 0.0;
    let mut screencast_score = 0.0;
    let mut drone_score = 0.0;
    let mut audio_score = 0.0;

    let presentation_hits = tag_counts.get("presentation_hint").copied().unwrap_or(0);
    let is_presentation = presentation_hits > 0;

    // --- Music & Audio / Singing scoring ---
    // Only active performance (singing, playing guitar, concerts) triggers music classification.
    // Passive background instruments (like a guitar resting against a wall during a speech) do NOT.
    let active_music_hits: usize = tag_counts
        .iter()
        .filter(|(k, _)| {
            k.contains("singing")
                || k.contains("singer")
                || k.contains("playing guitar")
                || k.contains("vocalist")
                || k.contains("musical performance")
                || k.contains("concert")
                || k.contains("music_hint")
        })
        .map(|(_, count)| count)
        .sum();

    if active_music_hits > 0 && !is_presentation {
        // Singing / musical performance on camera is treated as B-Roll (cutaways / live performance atmosphere)
        interview_score -= 50.0;
        broll_score += (active_music_hits as f64 * 16.0).min(65.0);
        audio_score += (active_music_hits as f64 * 4.0).min(20.0);
    }

    // --- Interview & Presentation scoring ---
    if presentation_hits > 0 {
        interview_score += (presentation_hits as f64 * 15.0).min(45.0);
    }

    // --- Interview scoring ---
    if speech_ratio > 0.40 {
        interview_score += 40.0;
    } else if speech_ratio > 0.20 {
        interview_score += 25.0;
    } else if speech_ratio > 0.08 {
        interview_score += 10.0;
    }

    if off_mic_count > 0 {
        interview_score += 25.0; // Strong indicator of dual-mic interview setup
    }

    if total_words > 50 {
        interview_score += 15.0;
    }

    // Visual interview cues
    let medium_shots = shot_counts.get("medium").copied().unwrap_or(0);
    let close_shots = shot_counts.get("close-up").copied().unwrap_or(0);
    let interview_shots = medium_shots + close_shots;
    if total_described_frames > 0 {
        let shot_ratio = interview_shots as f64 / total_described_frames as f64;
        if shot_ratio > 0.5 {
            interview_score += 20.0;
        }
    }

    let interview_tag_hits: usize = tag_counts
        .iter()
        .filter(|(k, _)| {
            k.contains("interview")
                || k.contains("speaking")
                || k.contains("talking")
                || k.contains("microphone")
                || k.contains("lavalier")
                || k.contains("speaker")
                || k.contains("podcaster")
                || k.contains("man speaking")
                || k.contains("woman speaking")
        })
        .map(|(_, count)| count)
        .sum();
    interview_score += (interview_tag_hits as f64 * 6.0).min(30.0);

    // --- Screencast scoring ---
    let screen_shots =
        shot_counts.get("screen recording").copied().unwrap_or(0) + shot_counts.get("slide").copied().unwrap_or(0);
    if total_described_frames > 0 && screen_shots as f64 / total_described_frames as f64 > 0.4 {
        screencast_score += 55.0;
    }

    let screen_tag_hits: usize = tag_counts
        .iter()
        .filter(|(k, _)| {
            k.as_str() == "code"
                || k.contains("source code")
                || k.contains("programming")
                || k.contains("terminal")
                || k.contains("web browser")
                || k.contains("software interface")
                || k.contains("screen recording")
                || k.contains("screencast")
                || k.contains("screenshot")
                || k.contains("computer screen")
                || k.contains("desktop screen")
                || k.as_str() == "slide"
                || k.contains("slideshow")
                || k.as_str() == "ui"
                || k.contains("user interface")
                || k.contains("screencast_hint")
        })
        .map(|(_, count)| count)
        .sum();
    screencast_score += (screen_tag_hits as f64 * 7.0).min(40.0);

    // --- Drone / Aerial scoring ---
    let drone_tag_hits: usize = tag_counts
        .iter()
        .filter(|(k, _)| {
            k.contains("aerial")
                || k.contains("drone")
                || k.contains("bird's-eye")
                || k.contains("bird's eye")
                || k.contains("birds eye")
                || k.contains("overhead")
                || k.contains("flyover")
                || k.contains("high altitude")
                || k.contains("drone_hint")
        })
        .map(|(_, count)| count)
        .sum();
    if drone_tag_hits > 0 {
        drone_score += (drone_tag_hits as f64 * 15.0).min(60.0);
    }
    let wide_shots = shot_counts.get("wide").copied().unwrap_or(0);
    if drone_tag_hits > 0 && wide_shots > 0 {
        drone_score += 15.0;
    }

    // --- B-Roll scoring ---
    if speech_ratio < 0.05 {
        broll_score += 30.0;
    } else if speech_ratio < 0.15 {
        broll_score += 15.0;
    }

    let broll_shots = shot_counts.get("b-roll").copied().unwrap_or(0) + shot_counts.get("wide").copied().unwrap_or(0);
    if total_described_frames > 0 {
        let broll_shot_ratio = broll_shots as f64 / total_described_frames as f64;
        if broll_shot_ratio > 0.4 {
            broll_score += 25.0;
        }
    }

    let broll_tag_hits: usize = tag_counts
        .iter()
        .filter(|(k, _)| {
            k.contains("b-roll")
                || k.contains("cutaway")
                || k.contains("establishing")
                || k.contains("scenery")
                || k.contains("landscape")
                || k.contains("building")
                || k.contains("architecture")
                || k.contains("sky")
                || k.contains("cloud")
                || k.contains("tree")
                || k.contains("car")
                || k.contains("vehicle")
                || k.contains("street")
                || k.contains("outdoor")
                || k.contains("nature")
                || k.contains("object")
                || k.contains("interior")
                || k.contains("exterior")
                || k.contains("action")
                || k.contains("broll_hint")
        })
        .map(|(_, count)| count)
        .sum();
    broll_score += (broll_tag_hits as f64 * 4.0).min(35.0);

    // If audio is muted/empty, B-roll is highly likely
    if has_audio == Some(false) {
        broll_score += 20.0;
    }

    // 6. Select winner
    scores.insert(CAT_INTERVIEW.to_string(), interview_score.max(0.0));
    scores.insert(CAT_BROLL.to_string(), broll_score.max(0.0));
    scores.insert(CAT_SCREENCAST.to_string(), screencast_score.max(0.0));
    scores.insert(CAT_DRONE.to_string(), drone_score.max(0.0));
    scores.insert(CAT_AUDIO.to_string(), audio_score.max(0.0));

    let mut ranked: Vec<(&String, &f64)> = scores.iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(b.0)));

    let (top_cat, top_score) = ranked[0];
    let runner_up_score = ranked.get(1).map(|(_, s)| **s).unwrap_or(0.0);

    // If the top score is very weak (< 15.0), classify as other
    let (final_cat, confidence) = if *top_score < 15.0 {
        (CAT_OTHER.to_string(), 0.3)
    } else {
        let conf = if *top_score + runner_up_score > 0.0 {
            ((*top_score - runner_up_score) / (*top_score + 10.0)).clamp(0.4, 0.98)
        } else {
            0.5
        };
        (top_cat.clone(), conf)
    };

    let detected_tag = if active_music_hits > 0 && !is_presentation { Some("music".to_string()) } else { None };

    Ok(ClassificationResult {
        video_id,
        category: final_cat,
        confidence,
        source: "auto".to_string(),
        scores,
        tag: detected_tag,
    })
}

/// Classify all videos in a project that don't have manual overrides, and update their records in the database.
pub fn classify_project_videos(db: &Db, project_id: i64) -> Result<usize, Error> {
    let mut st = db.conn.prepare(
        "SELECT DISTINCT v.id FROM videos v
         JOIN video_files vf ON vf.video_id = v.id
         JOIN project_folders pf ON pf.folder_id = vf.folder_id
         WHERE pf.project_id = ?1 AND COALESCE(v.category_source, 'auto') <> 'manual'",
    )?;
    let video_ids: Vec<i64> = st.query_map([project_id], |r| r.get(0))?.collect::<Result<_, _>>()?;

    let mut updated = 0;
    for vid in video_ids {
        if let Ok(res) = classify_video(db, vid) {
            db.set_video_category(vid, &res.category, res.confidence, "auto", res.tag.as_deref())?;
            updated += 1;
        }
    }
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_folder_names() {
        assert_eq!(folder_for_category("interview"), "Interviews");
        assert_eq!(folder_for_category("broll"), "B-Roll");
        assert_eq!(folder_for_category("screencast"), "Screen Recordings");
        assert_eq!(folder_for_category("drone"), "Drones & Aerials");
        assert_eq!(folder_for_category("audio"), "Music & Audio");
        assert_eq!(folder_for_category("other"), "Other");
        assert_eq!(folder_for_category("timelapse"), "Timelapse");
    }

    #[test]
    fn test_classify_interview_and_broll() {
        let db = Db::open_in_memory().unwrap();
        // Insert a video with high speech and interview frame tags
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, vcodec, has_audio) VALUES (1, 'hash1', 1000, 30.0, 'h264', 1)",
                [],
            )
            .unwrap();
        // Add speech segments covering 25s of the 30s
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (1, 0.0, 12.0, 'Hello welcome to the show can you tell us about your project', 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (1, 12.0, 26.0, 'Yes absolutely we started this project three years ago in our garage', 0)",
                [],
            )
            .unwrap();
        // Add interview frame
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (1, 5.0, '{\"description\": \"Man speaking to camera with microphone\", \"shot\": \"medium\", \"tags\": [\"interview\", \"man speaking\"], \"objects\": [\"microphone\"], \"setting\": \"studio\", \"visible_text\": []}')",
                [],
            )
            .unwrap();

        let res = classify_video(&db, 1).unwrap();
        assert_eq!(res.category, CAT_INTERVIEW);
        assert!(res.confidence > 0.6);

        // Insert a b-roll video (scenery, no speech)
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, vcodec, has_audio) VALUES (2, 'hash2', 1000, 20.0, 'h264', 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (2, 2.0, '{\"description\": \"Sunset over mountain landscape\", \"shot\": \"wide\", \"tags\": [\"scenery\", \"landscape\", \"sunset\"], \"objects\": [\"mountain\"], \"setting\": \"outdoors\", \"visible_text\": []}')",
                [],
            )
            .unwrap();

        let res2 = classify_video(&db, 2).unwrap();
        assert_eq!(res2.category, CAT_BROLL);

        // Test manual override persists
        db.set_video_category(2, "custom_tag", 1.0, "manual", None).unwrap();
        let res_manual = classify_video(&db, 2).unwrap();
        assert_eq!(res_manual.category, "custom_tag");
        assert_eq!(res_manual.source, "manual");

        // Test musical performance video defaults to B-Roll with 'music' tag
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, vcodec, has_audio) VALUES (3, 'hash3', 1000, 30.0, 'h264', 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (3, 5.0, 20.0, 'singing lyrics of an acoustic song', 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (3, 8.0, '{\"description\": \"Woman singing and playing acoustic guitar on a patio\", \"shot\": \"medium\", \"tags\": [\"singing\", \"acoustic guitar\", \"musician\"], \"objects\": [\"guitar\"], \"setting\": \"outdoors\", \"visible_text\": []}')",
                [],
            )
            .unwrap();
        let res_music = classify_video(&db, 3).unwrap();
        assert_eq!(res_music.category, CAT_BROLL);
        assert_eq!(res_music.tag.as_deref(), Some("music"));

        // Test presentation / speech with background instruments classifies as Interview (not music)
        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, vcodec, has_audio) VALUES (4, 'hash4', 1000, 120.0, 'h264', 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES (4, 1.0, 90.0, 'Hello everyone my name is Alina and I am from Cross Private Banking tonight I have with me Eric who is Vice President', 0)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO frames(video_id, t_s, description_json) VALUES (4, 10.0, '{\"description\": \"A woman is speaking into a microphone addressing a small group with a guitar sitting in the background\", \"shot\": \"medium\", \"tags\": [\"speaking into a microphone\", \"outdoor event\"], \"objects\": [\"guitar\", \"microphone\"], \"setting\": \"outdoors\", \"visible_text\": []}')",
                [],
            )
            .unwrap();
        let res_pres = classify_video(&db, 4).unwrap();
        assert_eq!(res_pres.category, CAT_INTERVIEW);
        assert_eq!(res_pres.tag, None);
    }
}
