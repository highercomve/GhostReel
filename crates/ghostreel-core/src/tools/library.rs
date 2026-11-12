//! Looking at the footage: what there is, what is said, what is seen, how steady it is.

use std::path::Path;

use serde::Serialize;
use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, ParamType as T, ToolParam, ToolSpec};
use super::util::{clip_chars, file_name, fit_items, frame_description, round1, round2, video_file_name};
use super::{Args, Tool, ToolCtx, ToolError, ToolImage, ToolOutput, ToolResult, tool};
use crate::index;
use crate::script_check::Steadiness;

pub static TOOLS: [Tool; 6] = [
    tool!(LIST_VIDEOS, list_videos),
    tool!(SEARCH, search),
    tool!(GET_TRANSCRIPT, get_transcript),
    tool!(GET_VIDEO, get_video),
    tool!(GET_FRAMES, get_frames),
    tool!(GET_STEADINESS, get_steadiness),
];

static LIST_VIDEOS: ToolSpec = ToolSpec {
    name: "list_videos",
    effect: Effect::Read,
    summary: "Every video in the project, one compact row each: id, file, duration, category (interview, \
              broll, drone…), camera style, whether anyone speaks, and what the first keyframe shows.",
    when: "Start here to learn the footage, or when searching has stopped finding anything.",
    returns: "{project, videos, rows}. A row is {id, file, duration_s, category, camera, has_speech, shows} — \
              and, when there is room, {tag, shaky_s, speech_segments, language, keyframes, says}. camera is \
              static, tripod, stabilised or handheld; shaky_s is how many seconds the camera shakes; shows is \
              the first described keyframe; says is the first spoken line.",
    misuse: &[
        "Listing is not looking: an id seen only here grounds nothing, and a clip written from it \
         survives only if something happens to be indexed where you guessed. Open it first.",
    ],
    chains_with: &["get_video(video_id) on anything that looks promising; get_transcript(video_id) where has_speech is true."],
    params: &[
        PROJECT,
        ToolParam::new("category", T::String, false, "Only this category, e.g. interview or broll."),
        ToolParam::new("speech", T::Boolean, false, "true: only videos with speech; false: only videos without."),
    ],
};

async fn list_videos(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let category = a.opt_str("category");
    let speech = a.opt_bool("speech");
    let limited = ctx.max_chars.is_some();
    let cfg = &ctx.config.script;
    let all = index::videos_with_shake(ctx.db, Some(p.id), cfg.max_shake_jerk, cfg.shake_relative, cfg.max_sway)?;
    let mut first_desc = ctx.db.conn.prepare(
        "SELECT description_json FROM frames
          WHERE video_id = ?1 AND description_json IS NOT NULL ORDER BY t_s LIMIT 1",
    )?;
    let mut first_line = ctx.db.conn.prepare(
        "SELECT text FROM transcript_segments WHERE video_id = ?1 ORDER BY start_s LIMIT 1",
    )?;
    let mut stored_summary = ctx.db.conn.prepare("SELECT summary FROM videos WHERE id = ?1")?;
    let mut rows = Vec::new();
    for v in all {
        if category.is_some_and(|c| c != v.category) || speech.is_some_and(|s| s != (v.segments > 0)) {
            continue;
        }
        let desc: Option<String> = first_desc.query_map([v.id], |r| r.get(0))?.next().transpose()?;
        let summary: Option<String> =
            stored_summary.query_row([v.id], |r| r.get::<_, Option<String>>(0)).ok().flatten().filter(|s| !s.trim().is_empty());
        let shows = summary.or_else(|| frame_description(desc.as_deref())).map(|d| clip_chars(&d, 120));
        let says: Option<String> = if v.segments > 0 {
            first_line.query_row([v.id], |r| r.get::<_, String>(0)).ok().map(|t| clip_chars(t.trim(), 100))
        } else {
            None
        };
        let mut row = json!({
            "id": v.id,
            "file": v.path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "duration_s": v.duration_s.map(round1),
            "category": v.category,
            "camera": v.camera,
            "has_speech": v.segments > 0,
            "shows": shows,
        });
        // A caller with limited room gets the row without the counts, the language and the first
        // spoken line: a context window is better spent on more videos.
        if !limited && let Value::Object(o) = &mut row {
            o.insert("tag".into(), json!(v.category_tag));
            o.insert("shaky_s".into(), json!(v.steadiness_measured.then(|| round1(v.shaky_s))));
            o.insert("speech_segments".into(), json!(v.segments));
            o.insert("language".into(), json!(v.language));
            o.insert("keyframes".into(), json!(v.frames));
            o.insert("says".into(), json!(says));
        }
        rows.push(row);
    }
    let total = rows.len();
    let dropped = fit_items(&mut rows, ctx.room(120));
    let mut value = json!({ "project": p.name, "videos": total, "rows": rows });
    if dropped > 0 {
        value["truncated"] = json!(format!("{dropped} videos left out to fit; filter with category or speech"));
    }
    Ok(ToolOutput::json(value).summary(format!("{total} videos")))
}

static SEARCH: ToolSpec = ToolSpec {
    name: "search",
    effect: Effect::Read,
    summary: "Find moments in a project by meaning or keyword across speech, on-screen text and what is visible.",
    when: "To locate a subject you have already decided to show. It searches the words people actually said, \
           so ask in the language of the footage.",
    returns: "{mode, hits}. mode says whether meaning search ran (without an embedding model it falls back to \
              keywords and says so). A hit is {video_id, file, start_s, end_s, camera, shaky, snippet, kinds, \
              matched_by, score}. start_s/end_s are the edges of the indexed window the match fell in — usually \
              a whole 20-40 s chunk, not a shot. camera is static, tripod, stabilised or handheld; shaky means \
              the window sits on a stretch the camera shakes through.",
    misuse: &[
        "Copying a hit's start_s/end_s into a clip gives a clip as long as the window, which is then \
         trimmed to its first seconds — rarely the shot you meant. Choose a sub-range yourself and open it \
         with get_video first.",
        "Two searches that return nothing mean those words are not in this footage. A third rephrasing wastes \
         a round; call list_videos and look at what there is.",
    ],
    chains_with: &[
        "The window a hit returned is legal to cut, so any sub-range of it is too.",
        "get_video(video_id, start_s, end_s) on the same window says whether the shot is usable.",
    ],
    params: &[
        PROJECT,
        ToolParam::new(
            "query",
            T::String,
            true,
            "Words or a phrase, matched against speech, visible text and keyframe descriptions together.",
        ),
        ToolParam::new("limit", T::Integer, false, "1-20, default 10. Anything outside that range is clamped."),
        ToolParam::new(
            "kinds",
            T::List(&T::Enum(&["moment", "transcript", "frame"])),
            false,
            "moment: what is visible (keyframe description) plus nearby speech; transcript: speech only; \
             frame: on-screen text only. Default: all.",
        ),
        ToolParam::new("keywords_only", T::Boolean, false, "Skip the embedding model."),
    ],
};

/// The embedding model: the caller's, or one started now when the caller allows it.
async fn embedder<'c>(ctx: &'c mut ToolCtx<'_>) -> Result<&'c mut crate::embed::Embedder, String> {
    if ctx.embedder.is_none() {
        if !ctx.embed_on_demand {
            return Err("no embedding model is loaded".into());
        }
        let setup = crate::runtime::resolve_embed(ctx.paths, ctx.config).await;
        *ctx.embedder = Some(crate::runtime::start_embedder(&setup, |_, _| {}).await?);
    }
    Ok(ctx.embedder.as_mut().expect("set above"))
}

#[derive(Serialize)]
struct CompactHit {
    video_id: i64,
    file: String,
    start_s: f64,
    end_s: f64,
    /// static, tripod, stabilised, handheld — a caller picks clips straight from a hit, so what it
    /// needs to know about the camera has to be here.
    camera: &'static str,
    /// The hit lies on a stretch the camera shakes through.
    shaky: bool,
    snippet: String,
    kinds: Vec<String>,
    matched_by: Vec<String>,
    score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    /// The closest keyframe's image, for a caller that can open files.
    #[serde(skip_serializing_if = "Option::is_none")]
    frame: Option<String>,
}

async fn search(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let query = a.opt_str("query").map(str::trim).filter(|q| !q.is_empty()).ok_or_else(|| ToolError::invalid("missing or invalid query"))?;
    let limit = a.opt_u64("limit").unwrap_or(10).clamp(1, 20) as usize;
    let kinds = Some(a.str_list("kinds")).filter(|k| !k.is_empty());

    // Embed first, then search without an await while the database is borrowed: the future stays
    // `Send` and the caller may run it on any thread.
    let mut mode = "keywords (asked for)".to_string();
    let mut vector = None;
    if !a.bool_or("keywords_only", false) {
        mode = match embedder(ctx).await {
            Ok(e) => match crate::search::query_vector(e, query).await {
                Ok(v) => {
                    vector = Some(v);
                    "meaning + keywords".to_string()
                }
                Err(why) => format!("keywords only: meaning search unavailable ({why})"),
            },
            Err(why) => format!("keywords only: meaning search unavailable ({why})"),
        };
    }
    let opts = crate::search::SearchOptions { project_id: Some(p.id), limit, kinds, ..Default::default() };
    let hits = crate::search::search_with_vector(ctx.db, &ctx.paths.data_dir, query, vector.as_deref(), &opts)?;

    let cfg = &ctx.config.script;
    let full = ctx.max_chars.is_none();
    let mut compact = Vec::with_capacity(hits.len());
    for hit in &hits {
        ctx.ground(hit.video_id, hit.start_s, hit.end_s);
        let windows = ctx.db.motion_windows(hit.video_id).unwrap_or_default();
        let shake_limit = crate::steadiness::shake_limit(&windows, cfg.max_shake_jerk, cfg.shake_relative);
        let shaky = !crate::steadiness::shaky_spans_with_sway(&windows, hit.start_s, hit.end_s, shake_limit, cfg.max_sway)
            .is_empty();
        compact.push(CompactHit {
            video_id: hit.video_id,
            file: file_name(&hit.path),
            start_s: round2(hit.start_s),
            end_s: round2(hit.end_s),
            camera: crate::steadiness::camera_style(&windows).as_str(),
            shaky,
            snippet: hit.snippet.chars().take(200).collect(),
            kinds: hit.kinds.clone(),
            matched_by: hit.matched_by.clone(),
            score: round2(hit.score),
            path: full.then(|| hit.path.display().to_string()),
            frame: hit.frame.as_ref().filter(|_| full).map(|f| f.display().to_string()),
        });
    }
    let total = compact.len();
    fit_items(&mut compact, ctx.room(mode.len() + 40));
    Ok(ToolOutput::json(json!({ "mode": mode, "hits": compact }))
        .summary(format!("{total} hits"))
        .empty(total == 0))
}

static GET_TRANSCRIPT: ToolSpec = ToolSpec {
    name: "get_transcript",
    effect: Effect::Read,
    summary: "The words spoken in one or more videos, as timed segments, optionally limited to a range.",
    when: "To read what people said. If the project's speech is already in your prompt (WHAT PEOPLE SAY), \
           use it only for a video listed there as left out.",
    returns: "format text (default): one 'start-end [off] text' line per segment under a '=== video N (file)' \
              header. format json: an array of {start, end, text, off_mic} for one video, or an object keyed \
              by video id for several. off_mic / [off] marks someone away from the microphone: in an \
              interview that is the questions and the slate, not the answers.",
    misuse: &[
        "A clip that opens on an off-mic segment is retimed to where someone answers, which moves the \
         picture with it — choose the answer yourself.",
        "A source clip that begins or ends inside a segment is pulled out to the sentence's edge, or back to \
         the previous one. Cut on the boundaries you are given.",
    ],
    chains_with: &["A segment's start and end are the only cut points a speaking clip should use."],
    params: &[
        ToolParam::new("video_id", T::Integer, false, "From a search hit or list_videos."),
        ToolParam::new("video_ids", T::List(&T::Integer), false, "Several videos at once (instead of video_id)."),
        ToolParam::new("start_s", T::Number, false, "Seconds; omit for the whole file."),
        ToolParam::new("end_s", T::Number, false, "Seconds."),
        ToolParam::new("speech_only", T::Boolean, false, "default false: drop off-mic (interviewer) lines."),
        ToolParam::new(
            "format",
            T::Enum(&["text", "json"]),
            false,
            "default text: one 'start-end [off] text' line per segment.",
        ),
    ],
};

fn transcript_ids(a: Args<'_>) -> Result<Vec<i64>, ToolError> {
    let mut ids = a.i64_list("video_ids");
    if let Some(id) = a.opt_i64("video_id") {
        ids.insert(0, id);
    }
    if ids.is_empty() {
        return Err(ToolError::invalid("give video_id or video_ids"));
    }
    Ok(ids)
}

fn transcript_text(videos: &[(i64, String, Vec<index::TranscriptSegment>)], cut: usize) -> String {
    let mut out = String::new();
    for (id, file, segs) in videos {
        out.push_str(&format!("=== video {id} ({file})\n"));
        if segs.is_empty() {
            out.push_str("(no speech)\n");
        }
        for s in segs {
            let off = if s.off_mic { " [off]" } else { "" };
            out.push_str(&format!("{:.2}-{:.2}{off} {}\n", s.start, s.end, s.text.trim()));
        }
    }
    if cut > 0 {
        out.push_str(&format!("… ({cut} more segments left out to fit; ask for a narrower range)\n"));
    }
    out
}

async fn get_transcript(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let ids = transcript_ids(a)?;
    let start = a.opt_f64("start_s").unwrap_or(0.0);
    let end = a.opt_f64("end_s").unwrap_or(f64::MAX);
    let speech_only = a.bool_or("speech_only", false);
    let as_text = a.opt_str("format").unwrap_or("text") == "text";

    let mut videos = Vec::new();
    for id in ids {
        ctx.check_video(id)?;
        ctx.ground(id, start, end);
        let segs: Vec<_> = index::transcript(ctx.db, id)?
            .into_iter()
            .filter(|s| s.end > start && s.start < end && !(speech_only && s.off_mic))
            .collect();
        videos.push((id, video_file_name(ctx.db, id), segs));
    }
    let total: usize = videos.iter().map(|v| v.2.len()).sum();

    // Cut to the room the caller has: whole segments off the end of the longest video, so what is
    // returned is exactly what is proven.
    let mut cut = 0usize;
    let room = ctx.room(0);
    while transcript_text(&videos, cut).len() > room {
        let Some(longest) = videos.iter_mut().max_by_key(|v| v.2.len()).filter(|v| !v.2.is_empty()) else { break };
        let n = (longest.2.len() / 10).max(1);
        longest.2.truncate(longest.2.len() - n);
        cut += n;
    }

    let value = if let [(_, _, segs)] = &videos[..] {
        serde_json::to_value(segs)?
    } else {
        let mut map = serde_json::Map::new();
        for (id, _, segs) in &videos {
            map.insert(id.to_string(), serde_json::to_value(segs)?);
        }
        Value::Object(map)
    };
    let mut out = ToolOutput::json(value).summary(format!("{total} segments"));
    if as_text {
        out = out.text(transcript_text(&videos, cut));
    }
    Ok(out)
}

static GET_VIDEO: ToolSpec = ToolSpec {
    name: "get_video",
    effect: Effect::Read,
    summary: "Look at a video: what is visible at each moment, how steady the camera is, and whether anyone speaks.",
    when: "Before you use any range as a picture. It is the only way to find out whether footage is worth \
           cutting to.",
    returns: "{video_id, file, duration_s, fps, width, height, language, has_speech, audio, steady, camera, \
              shaky_at, keyframes, keyframes_total, video}. has_speech and audio (source|mute) are for the \
              range asked; shaky_at lists the stretches worse than this clip's own ordinary level as \
              'start-end' seconds; keyframes are {t_s, description, visible_text} — what is actually on \
              screen, in order; video is the library row (category, copies, status).",
    misuse: &[
        "A clip whose pictures you have not seen is a guess: if nothing was described or spoken near it, the \
         clip is dropped and your beat loses its picture.",
        "Cutting from a range listed in shaky_at puts visible camera shake in the finished piece; the clip is \
         moved to the nearest steady stretch of the same shot, or dropped when the shot has none.",
    ],
    chains_with: &[
        "Opening a video this way makes the whole of it legal to cut, not only the range you asked for.",
        "has_speech false means the clip cannot carry source audio: mute it, or give the beat a bed from a \
         video where someone speaks.",
        "get_frames(video_id, times) shows the actual pictures.",
    ],
    params: &[
        ToolParam::new("video_id", T::Integer, true, "From a search hit or list_videos."),
        ToolParam::new(
            "start_s",
            T::Number,
            false,
            "Seconds. With end_s, every keyframe in that range; without, a sample across the whole file.",
        ),
        ToolParam::new("end_s", T::Number, false, "Seconds."),
        ToolParam::new("limit", T::Integer, false, "Most keyframes to return (default 40; 12 for a small context)."),
    ],
};

async fn get_video(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let cfg = ctx.config.script.clone();
    let row = index::video_by_id(ctx.db, video_id, cfg.max_shake_jerk, cfg.shake_relative, cfg.max_sway)?;
    let duration = row.duration_s.unwrap_or(0.0);
    ctx.ground(video_id, 0.0, duration);

    let small = ctx.max_chars.is_some_and(|m| m <= ctx.config.script.local_tool_result_chars);
    let limit = a.opt_u64("limit").map(|l| l as usize).unwrap_or(if small { 12 } else { 40 }).max(1);
    let desc_chars = match ctx.max_chars {
        None => usize::MAX,
        Some(_) if small => 120,
        Some(_) => 300,
    };
    let range_start = a.opt_f64("start_s").unwrap_or(0.0);
    let range_end = a.opt_f64("end_s").unwrap_or(f64::MAX);

    // Not `index::frames`: that wants the thumbnail path, and a keyframe described before its
    // thumbnail was written still has something to say.
    let mut frames: Vec<(f64, Option<String>, Option<String>)> = ctx
        .db
        .conn
        .prepare("SELECT t_s, description_json, visible_text FROM frames WHERE video_id = ?1 ORDER BY t_s")?
        .query_map([video_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    frames.retain(|(t, _, _)| *t >= range_start - 0.5 && *t <= range_end + 0.5);
    let total = frames.len();
    let step = total.div_ceil(limit).max(1);
    let mut keyframes: Vec<Value> = frames
        .iter()
        .step_by(step)
        .map(|(t, description_json, visible_text)| {
            let description = frame_description(description_json.as_deref()).map(|d| clip_chars(&d, desc_chars));
            json!({ "t_s": round2(*t), "description": description, "visible_text": visible_text })
        })
        .collect();

    // Whether anyone speaks in the asked-for range. Without this a caller has no evidence either
    // way and picks the first `audio` the schema offers — "source" — for scenery.
    let speech_end = if range_end == f64::MAX { duration } else { range_end };
    let has_speech: bool = ctx
        .db
        .conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM transcript_segments WHERE video_id = ?1 AND end_s > ?2 AND start_s < ?3)",
            rusqlite::params![video_id, range_start, speech_end],
            |r| r.get(0),
        )
        .unwrap_or(false);
    // How steady the camera is here, judged against the clip's own ordinary level as well as the
    // floor: a handheld clip's usual stretches are what it is, the worse ones are what to avoid.
    let measured = ctx.db.motion_windows(video_id).unwrap_or_default();
    let limit_j = crate::steadiness::shake_limit(&measured, cfg.max_shake_jerk, cfg.shake_relative);
    let shaky_at: Vec<String> =
        crate::steadiness::shaky_spans_with_sway(&measured, range_start, speech_end, limit_j, cfg.max_sway)
            .iter()
            .map(|(x, y)| format!("{x:.1}-{y:.1}"))
            .collect();
    let steady = if measured.is_empty() {
        "unknown"
    } else if shaky_at.is_empty() {
        "yes"
    } else {
        "not everywhere — see shaky_at"
    };

    let mut value = json!({
        "video_id": video_id,
        "file": file_name(&row.path),
        "duration_s": row.duration_s,
        "fps": row.fps,
        "width": row.width,
        "height": row.height,
        "language": row.language,
        "has_speech": has_speech,
        "audio": if has_speech { "source" } else { "mute" },
        "steady": steady,
        "camera": crate::steadiness::camera_style(&measured).as_str(),
        "shaky_at": shaky_at,
        "keyframes": keyframes,
        "keyframes_total": total,
        "video": row,
    });
    // Drop keyframes until the JSON fits.
    if let Some(max) = ctx.max_chars {
        while value.to_string().len() > max && !keyframes.is_empty() {
            let cut = (keyframes.len() / 4).max(1);
            keyframes.truncate(keyframes.len() - cut);
            value["keyframes"] = json!(keyframes);
        }
        if value.to_string().len() > max {
            value.as_object_mut().map(|o| o.remove("video"));
        }
    }
    Ok(ToolOutput::json(value).summary(format!("{}s, {} frames", duration as i64, keyframes.len())))
}

static GET_FRAMES: ToolSpec = ToolSpec {
    name: "get_frames",
    effect: Effect::Read,
    summary: "Look at a video: returns the pictures themselves (JPEG) with their times and descriptions.",
    when: "To see a shot rather than read about it. Without times, keyframes spread across the whole video; \
           with times, the keyframe nearest each time, or the exact frame when exact=true (decodes the \
           source, which must be reachable).",
    returns: "{frames: [{video_id, t_s, nearest_keyframe_s, description}]} and, for callers that can see, \
              one picture per entry.",
    misuse: &["exact=true decodes the source file: slow for many times, and it fails when the file is offline."],
    chains_with: &["get_video lists the keyframes and their descriptions without the pictures."],
    params: &[
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("times", T::List(&T::Number), false, "Seconds into the video (at most 8)."),
        ToolParam::new("limit", T::Integer, false, "Without times: how many keyframes (default 6, at most 8)."),
        ToolParam::new("exact", T::Boolean, false, "default false: decode the exact frame instead of the nearest keyframe."),
    ],
};

const MAX_FRAMES: usize = 8;
/// Long side of a decoded frame, the same size the keyframes are stored at.
const FRAME_LONG_SIDE: u32 = 768;

/// One frame of the source at `t_s`, decoded by ffmpeg and scaled to the keyframe size.
fn decode_frame(source: &Path, t_s: f64) -> Result<Vec<u8>, ToolError> {
    let ffmpeg = crate::doctor::locate("ffmpeg").ok_or_else(|| ToolError::invalid("ffmpeg not found"))?;
    let scale = format!("scale=w={FRAME_LONG_SIDE}:h={FRAME_LONG_SIDE}:force_original_aspect_ratio=decrease");
    let out = crate::proc::std_command(ffmpeg)
        .args(["-v", "error", "-ss", &format!("{t_s:.3}"), "-i"])
        .arg(source)
        .args(["-frames:v", "1", "-vf", &scale, "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
        .output()
        .map_err(|e| ToolError::invalid(format!("cannot run ffmpeg: {e}")))?;
    if !out.status.success() || out.stdout.is_empty() {
        return Err(ToolError::Invalid(format!(
            "ffmpeg could not decode {} at {t_s:.2}s: {}",
            source.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

async fn get_frames(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let frames = index::frames(ctx.db, &ctx.paths.data_dir, video_id)?;
    let exact = a.bool_or("exact", false);
    let times: Vec<f64> = a.f64_list("times").into_iter().take(MAX_FRAMES).collect();

    // (time asked for, keyframe nearest to it)
    let picks: Vec<(f64, Option<&index::FrameRow>)> = if times.is_empty() {
        if frames.is_empty() {
            return Err(ToolError::Invalid(format!(
                "video {video_id} has no keyframes yet; pass times with exact=true"
            )));
        }
        let limit = a.opt_u64("limit").unwrap_or(6).clamp(1, MAX_FRAMES as u64) as usize;
        let step = frames.len().div_ceil(limit).max(1);
        frames.iter().step_by(step).map(|f| (f.t_s, Some(f))).collect()
    } else {
        times
            .iter()
            .map(|&t| (t, frames.iter().min_by(|x, y| (x.t_s - t).abs().total_cmp(&(y.t_s - t).abs()))))
            .collect()
    };

    // Decoding is for a caller who will look at the result.
    let source = if exact && ctx.can_see { Some(index::video_by_id(ctx.db, video_id, 0.0, 0.0, 0.0)?.path) } else { None };
    let mut images = Vec::new();
    let mut listed = Vec::new();
    for (t, nearest) in picks {
        let description = nearest.and_then(|f| frame_description(f.description.as_deref()));
        let shown_t = match (&source, nearest) {
            (Some(_), _) | (None, None) => t,
            (None, Some(f)) => f.t_s,
        };
        if source.is_none() && nearest.is_none() {
            return Err(ToolError::Invalid(format!("video {video_id} has no keyframes yet; use exact=true")));
        }
        let mut caption = format!("video {video_id} at {shown_t:.2}s");
        if let Some(f) = nearest {
            if exact || (f.t_s - t).abs() > 0.01 {
                caption.push_str(&format!(" (nearest keyframe {:.2}s)", f.t_s));
            }
            if let Some(d) = &description {
                caption.push_str(&format!(": {d}"));
            }
        }
        if ctx.can_see {
            let jpeg = match (&source, nearest) {
                (Some(src), _) => decode_frame(src, t)?,
                (None, Some(f)) => std::fs::read(&f.path)
                    .map_err(|e| ToolError::Invalid(format!("cannot read keyframe {}: {e}", f.path.display())))?,
                (None, None) => unreachable!("checked above"),
            };
            images.push(ToolImage { caption, jpeg });
        }
        listed.push(json!({
            "video_id": video_id,
            "t_s": round2(shown_t),
            "nearest_keyframe_s": nearest.map(|f| round2(f.t_s)),
            "description": description,
        }));
    }
    let n = listed.len();
    Ok(ToolOutput::json(json!({ "frames": listed })).images(images).summary(format!("{n} frames")))
}

static GET_STEADINESS: ToolSpec = ToolSpec {
    name: "get_steadiness",
    effect: Effect::Read,
    summary: "Is the camera shaking in a stretch of a video?",
    when: "Before you commit to a range. From the motion measured at indexing; without a range, the whole video.",
    returns: "camera style, worst shake (jerk) and sway against the limits in use, the shaky stretches inside \
              the range, and a verdict (steady / some shake / shaky: over half the range). When the range is \
              shaky, the nearest steady stretch of the same length.",
    misuse: &["verdict 'not measured' means the steadiness stage has not run for this video: say so, do not assume steady."],
    chains_with: &["Pick the nearest steady stretch it offers, then save_script with those bounds."],
    params: &[
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("start_s", T::Number, false, ""),
        ToolParam::new("end_s", T::Number, false, ""),
        ToolParam::new("windows", T::Boolean, false, "default false: also list every measured window in the range."),
    ],
};

async fn get_steadiness(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let video_id = a.i64("video_id")?;
    ctx.check_video(video_id)?;
    let windows = ctx.db.motion_windows(video_id)?;
    let duration = windows.last().map(|w| w.end_s).unwrap_or(0.0);
    let start = a.opt_f64("start_s").unwrap_or(0.0);
    let end = a.opt_f64("end_s").unwrap_or(duration);
    if end <= start {
        return Err(ToolError::invalid("end_s must be after start_s"));
    }
    let cfg = &ctx.config.script;
    let Some(st) = Steadiness::measure(&windows, cfg, start, end) else {
        return Ok(ToolOutput::json(json!({
            "video_id": video_id,
            "range": [start, end],
            "verdict": "not measured",
            "next": "the steadiness stage has not run for this video (index it), or the range is outside the video",
        }))
        .summary("not measured"));
    };
    let mut out = json!({ "video_id": video_id, "range": [round2(start), round2(end)] });
    if let (Value::Object(o), Value::Object(r)) = (&mut out, st.to_json()) {
        o.extend(r);
    }
    if a.bool_or("windows", false) {
        out["windows"] = json!(
            windows
                .iter()
                .filter(|w| w.end_s > start && w.start_s < end)
                .map(|w| json!({
                    "start_s": round2(w.start_s),
                    "end_s": round2(w.end_s),
                    "jerk": round2(w.jerk),
                    "sway": round2(w.sway),
                    "motion": round2(w.motion),
                }))
                .collect::<Vec<_>>()
        );
    }
    let summary = out["verdict"].as_str().unwrap_or("measured").to_string();
    Ok(ToolOutput::json(out).summary(summary))
}
