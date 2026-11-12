//! Scripts: reading, checking, fixing, drafting, saving, exporting and previewing a cut.

use std::path::PathBuf;
use std::str::FromStr as _;

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, PROJECT_OPTIONAL, ParamType as T, ToolParam, ToolSpec};
use super::{Args, Tool, ToolCtx, ToolError, ToolOutput, ToolResult, tool};
use crate::{script, script_check};

pub static TOOLS: [Tool; 8] = [
    tool!(LIST_SCRIPTS, list_scripts),
    tool!(LIST_RUNS, list_runs),
    tool!(GET_SCRIPT, get_script),
    tool!(APPLY_FIX, apply_fix),
    tool!(DRAFT_SCRIPT, draft_script),
    tool!(SAVE_SCRIPT, save_script),
    tool!(EXPORT_SCRIPT, export_script),
    tool!(PREVIEW_SCRIPT, preview_script),
];

static LIST_SCRIPTS: ToolSpec = ToolSpec {
    name: "list_scripts",
    effect: Effect::Read,
    summary: "Script versions of a project (newest first).",
    when: "To find the script_id of a cut, or to see how many versions it has.",
    returns: "One entry per version: id, title, version, kind, when it was saved.",
    misuse: &[],
    chains_with: &["get_script(script_id) reads one version with its issues."],
    params: &[PROJECT],
};

async fn list_scripts(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let list = script::list(ctx.db, p.id)?;
    let n = list.len();
    Ok(ToolOutput::json(serde_json::to_value(list)?).summary(format!("{n} scripts")))
}

static LIST_RUNS: ToolSpec = ToolSpec {
    name: "list_runs",
    effect: Effect::Read,
    summary: "Recent long jobs (clip generation, indexing, faces, renders) from the app and the CLI, newest first.",
    when: "To see whether something is still running, or why it failed.",
    returns: "Each run: state, steps, summary, error and links to what it made (clip set, script, file).",
    misuse: &[],
    chains_with: &["list_clip_sets / get_script for what a run linked to."],
    params: &[PROJECT_OPTIONAL, ToolParam::new("limit", T::Integer, false, "Max runs (default 20).")],
};

async fn list_runs(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let limit = a.opt_u64("limit").unwrap_or(20) as usize;
    let named = a.has("project") || ctx.project.is_some();
    let runs = if named { ctx.db.runs_for_project(ctx.project_for(a)?.id, limit)? } else { ctx.db.recent_runs(limit)? };
    let n = runs.len();
    Ok(ToolOutput::json(serde_json::to_value(runs)?).summary(format!("{n} runs")))
}

static GET_SCRIPT: ToolSpec = ToolSpec {
    name: "get_script",
    effect: Effect::Read,
    summary: "One script version: beats, clips, narration, on-screen text, and `agents` (the cut's own agent choice \
              {orchestrator?, workers?, max_parallel?}, absent = the app's).",
    when: "Before changing a cut, and after saving one to see what is wrong with it.",
    returns: "The script plus `issues` checked against the index as it is now (never stale), each with any typed \
              `fixes`.",
    misuse: &[],
    chains_with: &["apply_fix with one of an issue's fixes; save_script to write a new version."],
    params: &[ToolParam::new("script_id", T::Integer, true, "From list_scripts.")],
};

async fn get_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let stored = script::load(ctx.db, a.i64("script_id")?)?;
    let min = crate::clips::length_range(ctx.db, stored.project_id, &stored.title).map(|r| r.0);
    let opts = script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    let issues = script_check::check(ctx.db, stored.project_id, &stored.script, &ctx.config.script, &opts)?;
    let n = issues.len();
    let mut v = serde_json::to_value(&stored)?;
    v["issues"] = serde_json::to_value(issues)?;
    Ok(ToolOutput::json(v).summary(format!("version {}, {n} issues", stored.version)))
}

static APPLY_FIX: ToolSpec = ToolSpec {
    name: "apply_fix",
    effect: Effect::Write,
    summary: "Apply one fix an issue offered (copy it from the issue's `fixes`): trim {beat_id, clip_index, in_s, \
              out_s}, set_layout {beat_id, clip_index, from_s, to_s, layout: follow_speaker|stacked|fixed|screen|fit}, \
              or run_stage {stage, enable} (runs that indexing stage for the project now — faces take ~5 min per hour \
              of video — turning it on first when enable).",
    when: "After get_script or save_script reported an issue with a fix.",
    returns: "{script_id, note, stage_run, issues}: the result is saved as a new version (unless dry_run) and the \
              issues are checked again.",
    misuse: &["run_stage can take minutes; do not call it twice for the same stage."],
    chains_with: &["get_script on the new script_id."],
    params: &[
        ToolParam::new("script_id", T::Integer, true, ""),
        ToolParam::new("fix", T::Object, true, "One entry of an issue's `fixes`: {kind, label?, ...}."),
        ToolParam::new("dry_run", T::Boolean, false, "default false: apply in memory and report, save nothing."),
    ],
};

async fn apply_fix(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let stored = script::load(ctx.db, a.i64("script_id")?)?;
    let fix: script::FixAction = serde_json::from_value(a.get("fix").cloned().ok_or_else(|| ToolError::invalid("missing 'fix' object"))?)
        .map_err(|e| ToolError::Invalid(format!("fix does not match {{kind: trim|set_layout|run_stage, ...}}: {e}")))?;
    let dry_run = a.bool_or("dry_run", false);
    let mut s = stored.script.clone();
    let applied = crate::script_fix::apply_fix(ctx.db, stored.project_id, &mut s, &fix)?;
    let mut ran = Value::Null;
    if let (Some(stage), false) = (&applied.run_stage, dry_run) {
        ran = super::project::run_stage(ctx, stored.project_id, stage).await?;
    }
    let min = crate::clips::length_range(ctx.db, stored.project_id, &stored.title).map(|r| r.0);
    let opts = script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    let issues = script_check::check(ctx.db, stored.project_id, &s, &ctx.config.script, &opts)?;
    let script_id =
        if dry_run || s == stored.script { stored.id } else { script::save_version(ctx.db, stored.project_id, &s, None)? };
    Ok(ToolOutput::json(json!({ "script_id": script_id, "note": applied.note, "stage_run": ran, "issues": issues }))
        .summary(format!("script #{script_id}")))
}

static DRAFT_SCRIPT: ToolSpec = ToolSpec {
    name: "draft_script",
    effect: Effect::Write,
    summary: "Ask GhostReel's editing model to draft or revise a script from the project's footage (it searches and \
              reads transcripts itself).",
    when: "When you want GhostReel's own model to do the whole edit. Slow: minutes. It needs that model to fit on this \
           machine. Not available from inside a script chat or a clip run, which are model runs already.",
    returns: "The saved version and its issues.",
    misuse: &["Called from inside a script chat or a clip run it is refused: do the work with the other tools."],
    chains_with: &["get_script on the version it saved; draft_script again with session_id to revise."],
    params: &[
        PROJECT,
        ToolParam::new("message", T::String, true, "e.g. 'a 60 second promo using the interviews, slower cuts'."),
        ToolParam::new("session_id", T::Integer, false, "Continue an earlier chat session (revisions)."),
    ],
};

async fn draft_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    ctx.refuse_nested("draft_script")?;
    let p = ctx.project_for(a)?;
    let message = a.str("message")?;
    let session_id = a.opt_i64("session_id");
    let config = ctx.config;
    let cut = crate::chat::cut_agents(ctx.db, None, session_id);
    let lease = crate::chat::lease_backend(ctx.paths, config, None, cut.as_ref(), |_| {}).await?;
    let embed_setup = crate::runtime::resolve_embed(ctx.paths, config).await;
    let embedder = crate::runtime::start_embedder(&embed_setup, |_, _| {}).await.ok();
    let db = crate::db::Db::open(&ctx.paths.db_file())?;
    let mut chat = crate::chat::ChatContext {
        db,
        paths: ctx.paths.clone(),
        backend: lease.backend,
        embedder,
        system_prompt: Some(config.chat.system_prompt.clone()),
        max_tool_rounds: lease.max_tool_rounds,
        script: config.script.clone(),
        jev: config.jev.clone(),
        cancel: ctx.cancel.clone(),
    };
    let images: Vec<String> = a.get("images").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let turn = crate::chat::run_turn(&mut chat, p.id, session_id, message, &images, None, &mut |_| {}).await?;
    let summary = match turn.script_id {
        Some(id) => format!("saved script #{id}"),
        None => "no script saved".to_string(),
    };
    Ok(ToolOutput::json(serde_json::to_value(turn)?).summary(summary))
}

static SAVE_SCRIPT: ToolSpec = ToolSpec {
    name: "save_script",
    effect: Effect::Write,
    summary: "Save your own script as a new version. Clips must lie inside real footage; returns the issues found \
              (errors mean the clip won't work).",
    when: "After looking at the footage (list_videos, get_video, get_transcript). Use dry_run first to check a \
           script without saving it. Beats play in order and a beat's clips play one after another. With a bed, the \
           bed's audio runs under the whole beat and its clips should be audio \"mute\" and add up to the bed's \
           length: an interview answer over b-roll. For a vertical (9:16) or square cut of wider footage set \
           width/height and framing (script-wide, or per clip to override): follow_speaker crops onto whoever talks \
           and cuts when the other answers, stacked shows the speaker on top and the listener below, fixed holds \
           still (on box or person_id). Without framing the picture is letterboxed. agents:{orchestrator?, workers?, \
           max_parallel?} sets which agents (list_agents) work on this cut; leave it out for the app's choice. Check \
           a framing with preview_crop first. Burnt-in text: a beat's title_layout sets when its on_screen_text shows \
           (start_s/end_s, seconds after the beat starts) and where (x,y 0..1 of the output frame; y is the text's \
           middle), title_style styles every title; captions:{x,y (y = bottom of the text), style, max_chars, \
           max_lines} places and styles speech captions; caption_edits fix single cues, named by video_id + \
           src_start_s/src_end_s (the cue's source range) with text, start_s/end_s (source seconds) or hidden. A \
           style is {font, size (line height / frame height), bold, uppercase, color, outline_color, outline, shadow \
           (fractions of the font size), box, box_color, box_opacity, align: left|center|right, max_width (0..1 of \
           the width)}; every field optional.",
    returns: "{script_id (null for a dry run), issues, errors, total_duration_s, clips, notes?, script?}. \
              save_script does the deterministic framing work itself (a vertical clip is trimmed off a screen share \
              at sentence edges, its closing hold kept on a face) and says what it changed in `notes`; everything \
              else comes back as issues, many with typed `fixes` you can apply with apply_fix.",
    misuse: &["A clip range nothing was described or spoken in is reported as an error and will not play."],
    chains_with: &["apply_fix with an issue's fix; preview_script to see it; export_script to hand it to an editor."],
    params: &[
        PROJECT,
        ToolParam::new(
            "script",
            T::Object,
            true,
            "{title, target_duration_s?, width?, height?, framing?:{layout: \"follow_speaker\"|\"stacked\"|\"fixed\", \
             person_id?, box?:{x,y,w,h} (0..1 of the source)}, beats:[{id, purpose, narration?, on_screen_text?, \
             title_layout?:{start_s?, end_s?, x?, y?, style?}, notes?, bed?:{video_id, in_s, out_s, why?}, \
             clips:[{video_id, in_s, out_s, audio?: \"source\"|\"mute\", why?, framing?}]}], title_style?, \
             captions?:{position?, size?, x?, y?, style?, max_chars?, max_lines?}, \
             caption_edits?:[{video_id, src_start_s, src_end_s, text?, start_s?, end_s?, hidden?}], \
             agents?:{orchestrator?, workers?, max_parallel?}}",
        ),
        ToolParam::new("dry_run", T::Boolean, false, "default false: only check it and report the issues, save nothing."),
        ToolParam::new(
            "repair",
            T::Boolean,
            false,
            "default true: do the deterministic framing work (trim a vertical clip off a screen share at sentence \
             edges, keep the closing hold on a face) and report it in notes; false keeps your bounds exactly.",
        ),
    ],
};

async fn save_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let raw = a.get("script").ok_or_else(|| ToolError::invalid("missing 'script' object"))?;
    let mut s: script::Script = serde_json::from_value(raw.clone())
        .map_err(|e| ToolError::Invalid(format!("script JSON does not match the schema: {e}")))?;
    s.fill_from_project(&p);
    if let Some(sel) = &s.agents {
        crate::agents::validate_selection(&ctx.config.agents(), sel).map_err(|e| ToolError::Invalid(e.to_string()))?;
    }
    let cfg = &ctx.config.script;
    let min = crate::clips::length_range(ctx.db, p.id, &s.title).map(|r| r.0);
    let notes = if a.bool_or("repair", true) {
        crate::script_fix::repair_framing(ctx.db, &mut s, cfg.speech_overrun_s, min)
    } else {
        Vec::new()
    };
    let opts = script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    let issues = script_check::check(ctx.db, p.id, &s, cfg, &opts)?;
    let errors = issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count();
    let dry_run = a.bool_or("dry_run", false);
    let script_id = if dry_run { None } else { Some(script::save_version(ctx.db, p.id, &s, None)?) };
    let mut out = json!({
        "script_id": script_id,
        "issues": issues,
        "errors": errors,
        "total_duration_s": s.total_duration_s(),
        "clips": s.clip_count(),
    });
    if !notes.is_empty() {
        out["notes"] = json!(notes);
        out["script"] = serde_json::to_value(&s)?;
    }
    let summary = match script_id {
        Some(id) => format!("saved script #{id}, {errors} errors"),
        None => format!("checked, {errors} errors"),
    };
    Ok(ToolOutput::json(out).summary(summary))
}

static EXPORT_SCRIPT: ToolSpec = ToolSpec {
    name: "export_script",
    effect: Effect::Write,
    summary: "Export a script version as an editable timeline: fcp_xml for Premiere Pro, or otio.",
    when: "When the cut is ready to be finished in an editor. Writes a file at `out`.",
    returns: "{path, format}.",
    misuse: &["An existing file at `out` is overwritten."],
    chains_with: &[],
    params: &[
        ToolParam::new("script_id", T::Integer, true, ""),
        ToolParam::new("format", T::Enum(&["fcp_xml", "otio"]), false, "default fcp_xml."),
        ToolParam::new("out", T::String, true, "Output file path."),
    ],
};

async fn export_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    ctx.check_cancel()?;
    let script_id = a.i64("script_id")?;
    let format = a.opt_str("format").unwrap_or("fcp_xml");
    let fmt = crate::export::ExportFormat::from_str(format)?;
    let out = PathBuf::from(a.str("out")?);
    let res = crate::export::export_script(ctx.db, script_id, fmt, &out)?;
    Ok(ToolOutput::json(json!({ "path": res.path, "format": format })).summary(format!("exported {format}")))
}

static PREVIEW_SCRIPT: ToolSpec = ToolSpec {
    name: "preview_script",
    effect: Effect::Write,
    summary: "Render a script version to a preview MP4 and return its path (and duration/encoder).",
    when: "To watch a cut. Slow: it cuts and encodes the real footage. Optionally burn in the on-screen titles and \
           the narration as subtitles.",
    returns: "The render: path, duration and encoder.",
    misuse: &["Rendering a long cut takes minutes and can be stopped; do not start two at once."],
    chains_with: &[],
    params: &[
        ToolParam::new("script_id", T::Integer, true, ""),
        ToolParam::new("burn_titles", T::Boolean, false, "default false."),
        ToolParam::new("burn_narration", T::Boolean, false, "default false: narration as burnt-in subtitles."),
        ToolParam::new("burn_captions", T::Boolean, false, "default false: speech captions from transcript as burnt-in subtitles."),
        ToolParam::new("normalize_audio", T::Boolean, false, "default false: bring every clip to the same loudness."),
        ToolParam::new(
            "full_size",
            T::Boolean,
            false,
            "default false: render at the cut's own size (e.g. 1080x1920) instead of a light 540p preview; use it \
             for a deliverable.",
        ),
        ToolParam::new("out", T::String, false, "Output file path; default is the data dir's preview folder."),
    ],
};

async fn preview_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let script_id = a.i64("script_id")?;
    let ffmpeg = crate::doctor::locate("ffmpeg").ok_or_else(|| ToolError::invalid("ffmpeg not found"))?;
    let cfg = &ctx.config.script;
    let opts = crate::preview::PreviewOptions {
        burn_titles: a.bool_or("burn_titles", false),
        burn_narration: a.bool_or("burn_narration", false),
        burn_captions: a.bool_or("burn_captions", false),
        normalize_audio: a.bool_or("normalize_audio", false),
        audio_fade_s: cfg.audio_fade_s,
        speech_overrun_s: cfg.speech_overrun_s,
        out: a.opt_str("out").map(PathBuf::from),
        cancel: ctx.cancel.clone(),
        size: if a.bool_or("full_size", false) {
            crate::preview::RenderSize::Full
        } else {
            crate::preview::RenderSize::Preview
        },
    };
    let res = crate::preview::render_preview(ctx.db, &ctx.paths.data_dir, &ffmpeg, script_id, &opts, |_, _| {})?;
    Ok(ToolOutput::json(serde_json::to_value(res)?).summary("rendered preview"))
}
