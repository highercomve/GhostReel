//! Who is in the footage and how it frames: faces, scenes, people, speakers, crops.

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, ParamType as T, ToolParam, ToolSpec};
use super::util::{round2, round3};
use super::{Args, Tool, ToolCtx, ToolError, ToolImage, ToolOutput, ToolResult, tool};

pub static TOOLS: [Tool; 7] = [
    tool!(GET_FACES, get_faces),
    tool!(GET_SCENES, get_scenes),
    tool!(GET_PEOPLE, get_people),
    tool!(SET_PERSON, set_person),
    tool!(MERGE_PEOPLE, merge_people),
    tool!(GET_SPEAKERS, get_speakers),
    tool!(PREVIEW_CROP, preview_crop),
];

static GET_FACES: ToolSpec = ToolSpec {
    name: "get_faces",
    effect: Effect::Read,
    summary: "Find and follow faces through a video: face tracks overlapping a time range (or at a timestamp).",
    when: "Before framing a vertical or square cut, or to see who is on screen when. Needs the faces stage to \
           have run for the video.",
    returns: "{video_id, range, tracks}. A track is {id, person_id, start_s, end_s, n_faces, avg_box_h, framing, \
              boxes}: framing is the median box centre, height fraction and % of time cut off by the frame \
              edge; boxes are downsampled to at most 2 per second ({t_s, x, y, w, h, sharpness, cut_off}, 0..1 \
              of the frame).",
    misuse: &["No tracks and a message means the faces stage has not run: call index first, do not assume nobody is there."],
    chains_with: &["get_people names the person_id; preview_crop shows how the framing looks."],
    params: &[
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("from_s", T::Number, false, "Start timestamp in seconds."),
        ToolParam::new("to_s", T::Number, false, "End timestamp in seconds."),
        ToolParam::new("at_s", T::Number, false, "Exact timestamp in seconds (finds tracks active at this time)."),
    ],
};

async fn get_faces(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let (from_s, to_s) = if let Some(at) = a.opt_f64("at_s") {
        (at, at)
    } else {
        (a.opt_f64("from_s").unwrap_or(0.0), a.opt_f64("to_s").unwrap_or(f64::MAX))
    };

    let all_tracks = ctx.db.face_tracks(video_id)?;
    if all_tracks.is_empty() {
        return Ok(ToolOutput::json(json!({
            "video_id": video_id,
            "tracks": [],
            "message": "no face tracks found for video (run face analysis first)"
        }))
        .summary("no face tracks"));
    }

    let overlapping: Vec<_> = all_tracks.into_iter().filter(|t| t.end_s >= from_s && t.start_s <= to_s).collect();
    let track_people = ctx.db.face_track_people(video_id)?;

    let mut tracks_out = Vec::new();
    for t in overlapping {
        let boxes = ctx.db.face_boxes_for_track(t.id)?;
        let (med_cx, med_cy, med_h, pct_cut_off) = if boxes.is_empty() {
            (0.5f32, 0.5f32, 0.0f32, 0.0f64)
        } else {
            let mut cxs: Vec<f32> = boxes.iter().map(|b| b.x + b.w / 2.0).collect();
            let mut cys: Vec<f32> = boxes.iter().map(|b| b.y + b.h / 2.0).collect();
            let mut hs: Vec<f32> = boxes.iter().map(|b| b.h).collect();
            cxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            cys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cut_offs = boxes.iter().filter(|b| b.cut_off).count();
            (cxs[cxs.len() / 2], cys[cys.len() / 2], hs[hs.len() / 2], 100.0 * (cut_offs as f64) / (boxes.len() as f64))
        };

        let mut downsampled = Vec::new();
        let mut last_t = -10.0;
        for b in boxes {
            if b.t_s >= from_s && b.t_s <= to_s && b.t_s - last_t >= 0.49 {
                last_t = b.t_s;
                downsampled.push(json!({
                    "t_s": round2(b.t_s),
                    "x": round3(b.x as f64),
                    "y": round3(b.y as f64),
                    "w": round3(b.w as f64),
                    "h": round3(b.h as f64),
                    "sharpness": round2(b.sharpness as f64),
                    "cut_off": b.cut_off,
                }));
            }
        }

        tracks_out.push(json!({
            "id": t.id,
            "person_id": track_people.get(&t.id),
            "start_s": round2(t.start_s),
            "end_s": round2(t.end_s),
            "n_faces": t.n_faces,
            "avg_box_h": round3(t.avg_box_h as f64),
            "framing": {
                "median_box_center": [round3(med_cx as f64), round3(med_cy as f64)],
                "median_height_fraction": round3(med_h as f64),
                "pct_time_cut_off": (pct_cut_off * 10.0).round() / 10.0,
            },
            "boxes": downsampled,
        }));
    }

    let n = tracks_out.len();
    Ok(ToolOutput::json(json!({
        "video_id": video_id,
        "range": [round2(from_s), round2(if to_s == f64::MAX { 0.0 } else { to_s })],
        "tracks": tracks_out,
    }))
    .summary(format!("{n} face tracks")))
}

static GET_SCENES: ToolSpec = ToolSpec {
    name: "get_scenes",
    effect: Effect::Read,
    summary: "What the picture is, stretch by stretch, from the face boxes: two_shot (two or more people at a \
              framable size), single_face, screen_share (only small camera tiles over a screen) or no_face.",
    when: "To choose a layout for a vertical cut, or to avoid a screen share. It is the same segmentation \
           framing uses (each stretch gets its own layout), clip candidates rank by, and checks warn about.",
    returns: "{video_id, from_s, to_s, mix, scenes}: mix is the share of each kind over the range; scenes are \
              the stretches. Without face analysis: scenes [] and a message.",
    misuse: &[],
    chains_with: &["preview_crop on a stretch to see the crop; get_faces for the boxes."],
    params: &[
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("from_s", T::Number, false, "default 0."),
        ToolParam::new("to_s", T::Number, false, "default the video's end."),
    ],
};

async fn get_scenes(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let duration: f64 = ctx
        .db
        .conn
        .query_row("SELECT COALESCE(duration_s, 0) FROM videos WHERE id = ?1", [video_id], |r| r.get(0))
        .unwrap_or(0.0);
    let from_s = a.opt_f64("from_s").unwrap_or(0.0).max(0.0);
    let to_s = a.opt_f64("to_s").unwrap_or(duration).min(duration.max(from_s));
    match crate::scenes::for_range(ctx.db, video_id, from_s, to_s)? {
        None => Ok(ToolOutput::json(json!({
            "video_id": video_id,
            "scenes": [],
            "message": "the faces of this video aren't indexed yet: run the index tool (it includes faces)",
        }))
        .summary("faces not indexed")),
        Some(spans) => {
            let n = spans.len();
            Ok(ToolOutput::json(json!({
                "video_id": video_id,
                "from_s": from_s,
                "to_s": to_s,
                "mix": crate::scenes::mix(&spans, from_s, to_s),
                "scenes": spans,
            }))
            .summary(format!("{n} scenes")))
        }
    }
}

static GET_PEOPLE: ToolSpec = ToolSpec {
    name: "get_people",
    effect: Effect::Read,
    summary: "People in a project: face tracks grouped by identity (by the speakers stage), with name, handle and \
              role for name tags, the videos they appear in, seconds on screen and seconds speaking.",
    when: "To know who is in the footage before naming them or framing on one. Needs the speakers stage.",
    returns: "{people: [{id, name, handle, role, videos, face_tracks, screen_time_s, speaking_time_s}]}. id is the \
              person_id the other tools take.",
    misuse: &[],
    chains_with: &["set_person to name someone; merge_people when two entries are one person."],
    params: &[PROJECT],
};

async fn get_people(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let people = ctx.db.person_summaries(p.id)?;
    if people.is_empty() {
        return Ok(ToolOutput::json(json!({
            "people": [],
            "message": "no people yet: run the speakers stage (after faces) for this project"
        }))
        .summary("no people"));
    }
    let out: Vec<Value> = people
        .into_iter()
        .map(|s| {
            json!({
                "id": s.id,
                "name": s.name,
                "handle": s.handle,
                "role": s.role,
                "videos": s.videos,
                "face_tracks": s.tracks,
                "screen_time_s": round2(s.screen_time_s),
                "speaking_time_s": round2(s.speaking_time_s),
            })
        })
        .collect();
    let n = out.len();
    Ok(ToolOutput::json(json!({ "people": out })).summary(format!("{n} people")))
}

static SET_PERSON: ToolSpec = ToolSpec {
    name: "set_person",
    effect: Effect::Write,
    summary: "Name a person: set their name, handle (e.g. @someone) and/or role (e.g. Host).",
    when: "After get_people, once you know who someone is. Omitted fields stay; an empty string clears one.",
    returns: "The person as saved. Names survive re-running the speakers stage.",
    misuse: &[],
    chains_with: &[],
    params: &[
        PROJECT,
        ToolParam::new("person_id", T::Integer, true, "From get_people."),
        ToolParam::new("name", T::String, false, ""),
        ToolParam::new("handle", T::String, false, ""),
        ToolParam::new("role", T::String, false, ""),
    ],
};

async fn set_person(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let id = a.i64("person_id")?;
    let person = ctx.db.set_person(p.id, id, a.opt_str("name"), a.opt_str("handle"), a.opt_str("role"))?;
    Ok(ToolOutput::json(serde_json::to_value(person)?).summary(format!("updated person #{id}")))
}

static MERGE_PEOPLE: ToolSpec = ToolSpec {
    name: "merge_people",
    effect: Effect::Destructive,
    summary: "Two entries are the same person: move everything of merge_id (faces, speech, name if keep_id has \
              none) into keep_id and delete merge_id.",
    when: "Only when you are sure the two entries are one person; there is no undo.",
    returns: "The person that was kept. Re-running the speakers stage keeps them together.",
    misuse: &["merge_id is deleted for good: merging two different people mixes their faces and speech."],
    chains_with: &["get_people first, to read both entries."],
    params: &[
        PROJECT,
        ToolParam::new("keep_id", T::Integer, true, "The person that stays."),
        ToolParam::new("merge_id", T::Integer, true, "The duplicate, which is deleted."),
    ],
};

async fn merge_people(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let (keep, other) = (a.i64("keep_id")?, a.i64("merge_id")?);
    let person = ctx.db.merge_people(p.id, keep, other)?;
    Ok(ToolOutput::json(serde_json::to_value(person)?).summary(format!("merged #{other} into #{keep}")))
}

static GET_SPEAKERS: ToolSpec = ToolSpec {
    name: "get_speakers",
    effect: Effect::Read,
    summary: "Who talks when in a video: speaker turns (voice clusters, 1 = most speech in the project) with the \
              person whose face moves when that voice speaks, their name, and the link's confidence (0–1).",
    when: "To cut on whoever is speaking, or to caption with names. Needs the speakers stage after transcription.",
    returns: "{video_id, turns: [{start_s, end_s, speaker, person_id, name, confidence}]}. person_id null = the voice \
              was not matched to a face (off camera).",
    misuse: &[],
    chains_with: &["get_people for the person behind a person_id."],
    params: &[
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("from_s", T::Number, false, ""),
        ToolParam::new("to_s", T::Number, false, ""),
    ],
};

async fn get_speakers(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let from_s = a.opt_f64("from_s").unwrap_or(0.0);
    let to_s = a.opt_f64("to_s").unwrap_or(f64::MAX);
    let turns = ctx.db.speaker_turns(video_id, from_s, to_s)?;
    if turns.is_empty() {
        return Ok(ToolOutput::json(json!({
            "video_id": video_id,
            "turns": [],
            "message": "no speaker turns in range (run the speakers stage after transcription)"
        }))
        .summary("no speaker turns"));
    }
    let out: Vec<Value> = turns
        .into_iter()
        .map(|t| {
            json!({
                "start_s": round2(t.start_s),
                "end_s": round2(t.end_s),
                "speaker": t.speaker,
                "person_id": t.person_id,
                "name": t.name,
                "confidence": t.confidence.map(round2),
            })
        })
        .collect();
    let n = out.len();
    Ok(ToolOutput::json(json!({ "video_id": video_id, "turns": out })).summary(format!("{n} turns")))
}

static PREVIEW_CROP: ToolSpec = ToolSpec {
    name: "preview_crop",
    effect: Effect::Read,
    summary: "See how a stretch of a video frames in a vertical or square cut before you write it into a script.",
    when: "Before save_script with a framing. Plans the crop (follow_speaker: on whoever talks, hard cut when the \
           other answers, steady otherwise; stacked: speaker on top, listener below; fixed: still) from the faces \
           and speaker turns found at indexing.",
    returns: "A summary (people on screen, shots, crop size, quality: longest stretch without the face, share cut \
              off, face size, and the paths of the stills) plus 3-4 stills rendered with the crop, around the cuts.",
    misuse: &["Needs faces (and speakers for follow_speaker) indexed; with none the crop is a still centre."],
    chains_with: &["get_scenes says which stretches are a two-shot or a screen share."],
    params: &[
        PROJECT,
        ToolParam::new("video", T::Integer, true, "Video id."),
        ToolParam::new("from_s", T::Number, true, ""),
        ToolParam::new("to_s", T::Number, true, ""),
        ToolParam::new("layout", T::Enum(&["follow_speaker", "stacked", "fixed"]), true, ""),
        ToolParam::new("width", T::Integer, false, "Output width (default 1080)."),
        ToolParam::new("height", T::Integer, false, "Output height (default 1920)."),
        ToolParam::new("person_id", T::Integer, false, "Follow (or hold on) this person instead of the speaker."),
    ],
};

async fn preview_crop(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let video_id = a.i64("video")?;
    let in_project = ctx.db.conn.query_row(
        "SELECT count(*) FROM video_files vf JOIN project_folders pf ON pf.folder_id = vf.folder_id
         WHERE pf.project_id = ?1 AND vf.video_id = ?2",
        [p.id, video_id],
        |r| r.get::<_, i64>(0),
    )?;
    if in_project == 0 {
        return Err(ToolError::Invalid(format!("video {video_id} is not in project {}", p.name)));
    }
    let (from_s, to_s) = (a.f64("from_s")?, a.f64("to_s")?);
    let layout: crate::framing::Layout = a.str("layout")?.parse()?;
    let width = a.opt_u64("width").unwrap_or(1080).clamp(16, 8192) as u32;
    let height = a.opt_u64("height").unwrap_or(1920).clamp(16, 8192) as u32;
    let framing = crate::framing::Framing { layout, person_id: a.opt_i64("person_id"), box_: None, overrides: Vec::new() };
    let ffmpeg = crate::doctor::locate("ffmpeg").ok_or_else(|| ToolError::invalid("ffmpeg not found"))?;
    let dir = ctx.paths.data_dir.join("frames").join("crops");
    let (plan, stills) =
        crate::framing::preview_crop(ctx.db, &ffmpeg, video_id, from_s, to_s, &framing, width, height, &dir, 4)?;
    let mut summary = crate::framing::summary(&plan);
    summary["stills"] = json!(
        stills
            .iter()
            .map(|(t, f)| json!({ "t_s": round2(from_s + t), "path": f.display().to_string() }))
            .collect::<Vec<_>>()
    );
    let mut images = Vec::new();
    if ctx.can_see {
        for (t, f) in &stills {
            let jpeg = std::fs::read(f).map_err(|e| ToolError::Invalid(format!("cannot read {}: {e}", f.display())))?;
            images.push(ToolImage {
                caption: format!("video {video_id} at {:.2}s ({} crop)", from_s + t, plan.layout.as_str()),
                jpeg,
            });
        }
    }
    Ok(ToolOutput::json(summary).images(images).summary(format!("{} stills", stills.len())))
}
