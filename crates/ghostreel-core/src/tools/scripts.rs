//! Scripts: reading, checking, fixing, drafting, saving, exporting and previewing a cut.

use std::path::PathBuf;
use std::str::FromStr as _;

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, PROJECT_OPTIONAL, ParamType as T, ToolParam, ToolSpec};
use super::{Args, Tool, ToolCtx, ToolError, ToolOutput, ToolResult, tool};
use crate::{script, script_check};

pub static TOOLS: [Tool; 10] = [
    tool!(LIST_SCRIPTS, list_scripts),
    tool!(LIST_RUNS, list_runs),
    tool!(GET_SCRIPT, get_script),
    tool!(APPLY_FIX, apply_fix),
    tool!(DRAFT_SCRIPT, draft_script),
    tool!(SAVE_SCRIPT, save_script),
    tool!(READ_CUT, read_cut),
    tool!(EXPORT_SCRIPT, export_script),
    tool!(REVIEW_RENDER, review_render),
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
    let runs =
        if named { ctx.db.runs_for_project(ctx.project_for(a)?.id, limit)? } else { ctx.db.recent_runs(limit)? };
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
              set_audio {beat_id, clip_index, audio: source|mute}, or run_stage {stage, enable} (runs that indexing stage for the project now — faces take ~5 min per hour \
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
    let fix: script::FixAction =
        serde_json::from_value(a.get("fix").cloned().ok_or_else(|| ToolError::invalid("missing 'fix' object"))?)
            .map_err(|e| {
                ToolError::Invalid(format!("fix does not match {{kind: trim|set_layout|set_audio|run_stage, ...}}: {e}"))
            })?;
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
    let script_id = if dry_run || s == stored.script {
        stored.id
    } else {
        script::save_version(ctx.db, stored.project_id, &s, None)?
    };
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
    when: "After looking at the footage (get_outline, get_manifest, get_video). Write a speaking clip by sentences: {video_id, from: \"s412\", to: \"s418\"} \
           with the ids get_manifest and get_transcript show — you choose the sentences, code turns them into seconds and places \
           the exact cut points, breaths, holds and framing on the measured speech. in_s/out_s (seconds) still \
           work for pictures and anything else; a clip with both uses the seconds. Beds take the same from/to. \
           \"s412@3\" starts at the third word of a sentence and \"s412@5\" as `to` ends with its fifth (needs word \
           timings; the error says when a video has none). Beats play in order and a beat's clips play one after another. With a bed, the \
           bed's audio runs under the whole beat and its clips should be audio \"mute\" and add up to the bed's \
           length: an interview answer over b-roll. For a vertical (9:16) or square cut of wider footage set \
           width/height and framing (script-wide, or per clip to override): follow_speaker crops onto whoever talks \
           and cuts when the other answers, stacked shows the speaker on top and the listener below, fixed holds \
           still (on box or person_id). Without framing the picture is letterboxed. agents (list_agents) picks who works on \
           this cut. Check \
           a framing with preview_crop first. Burnt-in text: a beat's title_layout sets when its on_screen_text shows \
           (start_s/end_s, seconds after the beat starts) and where (x,y 0..1 of the output frame; y is the text's \
           middle), title_style styles every title; captions:{x,y (y = bottom of the text), style, max_chars, \
           max_lines} places and styles speech captions; caption_edits fix single cues, named by video_id + \
           src_start_s/src_end_s (the cue's source range) with text, start_s/end_s (source seconds) or hidden. A \
           style is {font, size (line height / frame height), bold, uppercase, color, outline_color, outline, shadow \
           (fractions of the font size), box, box_color, box_opacity, align: left|center|right, max_width (0..1 of \
           the width)}; every field optional. A beat's card plays a full-screen title before its clips (2 s on black; \
           card_style sets the cut's default) and adds to the length.",
    returns: "{script_id (null for a dry run), issues, errors, total_duration_s, clips, notes?, script?}. A clip \
              written with sentences comes back resolved: `script` has its seconds and keeps its from/to. A sentence \
              of another video, a from after its to, or an unknown id is an error that says how to fix it. What the \
              repair changed (see `repair`) is in `notes`; everything else comes back as issues, many with typed \
              `fixes` you can apply with apply_fix.",
    misuse: &[
        "A clip range nothing was described or spoken in is reported as an error and will not play.",
        "Do not add holds, breath padding or pull cut points back to dodge the next speaker yourself: pick the \
         sentences and let code place the cut. Adding a second of hold after a clip is what causes speech/mute \
         problems.",
    ],
    chains_with: &[
        "read_cut on the saved script_id to read the cut back as text (the words heard, who is on screen, whether \
         each cut is clean); apply_fix with an issue's fix; preview_script to watch it; export_script to hand it \
         to an editor.",
    ],
    params: &[
        PROJECT,
        ToolParam::new(
            "script",
            T::Object,
            true,
            "{title, target_duration_s?, width?, height?, framing?:{layout: \"follow_speaker\"|\"stacked\"|\"fixed\", \
             person_id?, box?:{x,y,w,h} (0..1 of the source)}, beats:[{id, purpose, narration?, on_screen_text?, \
             title_layout?:{start_s?, end_s?, x?, y?, style?}, card?:{text, duration_s?, background?, style?}, notes?, bed?:{video_id, from?, to?, in_s?, out_s?, why?}, \
             clips:[{video_id, from?, to?, in_s?, out_s?, audio?: \"source\"|\"mute\", why?, framing?}]}], title_style?, \
             captions?:{position?, size?, x?, y?, style?, max_chars?, max_lines?}, \
             card_style?, \
             caption_edits?:[{video_id, src_start_s, src_end_s, text?, start_s?, end_s?, hidden?}], \
             agents?:{orchestrator?, workers?, max_parallel?}}",
        ),
        ToolParam::new(
            "dry_run",
            T::Boolean,
            false,
            "default false: only check it and report the issues, save nothing.",
        ),
        ToolParam::new(
            "section_cards",
            T::Boolean,
            false,
            "true: a card titled after the chapter before each beat that changes chapter (yours are kept); false: \
             remove the generated ones.",
        ),
        ToolParam::new(
            "repair",
            T::Boolean,
            false,
            "default true: do the deterministic framing work (trim a vertical clip off a screen share at sentence \
             edges, keep the closing hold on a face, place cuts on the measured speech), reported in notes; false keeps \
             all bounds exactly, a clip's `exact: true` just that one.",
        ),
    ],
};

/// A script as an agent wrote it, made ready to check and save: parsed, filled from the project,
/// sentence ids resolved to seconds and, unless `repair` is off, framed and placed on the sound.
/// Returns the script and a note for each thing done to it.
fn prepare_script(
    ctx: &mut ToolCtx<'_>,
    p: &crate::projects::Project,
    raw: &Value,
    repair: bool,
) -> Result<(script::Script, Option<f64>, Vec<String>), ToolError> {
    let mut s: script::Script = serde_json::from_value(raw.clone())
        .map_err(|e| ToolError::Invalid(format!("script JSON does not match the schema: {e}")))?;
    s.fill_from_project(p);
    // Clips written with sentence ids become seconds here; the repair below then places the cuts.
    let by_sentence = crate::sentences::resolve_script(ctx.db, &mut s)?;
    if let Some(sel) = &s.agents {
        crate::agents::validate_selection(&ctx.config.agents(), sel).map_err(|e| ToolError::Invalid(e.to_string()))?;
    }
    let cfg = &ctx.config.script;
    let min = crate::clips::length_range(ctx.db, p.id, &s.title).map(|r| r.0);
    let mut notes = Vec::new();
    if by_sentence > 0 {
        notes.push(format!("resolved {by_sentence} sentence range(s) to seconds (see `script`)"));
    }
    if repair {
        notes.extend(crate::script_fix::repair_framing(ctx.db, &mut s, cfg.speech_overrun_s, min));
        // Cuts on the sound itself, for footage whose speech was measured.
        notes.extend(crate::cuts::place_cuts(ctx.db, &mut s, cfg).iter().map(crate::cuts::CutMove::message));
    }
    Ok((s, min, notes))
}

async fn save_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let raw = a.get("script").ok_or_else(|| ToolError::invalid("missing 'script' object"))?;
    let (mut s, min, mut notes) = prepare_script(ctx, &p, raw, a.bool_or("repair", true))?;
    match a.get("section_cards").and_then(Value::as_bool) {
        Some(true) => {
            let n = crate::cards::generate(ctx.db, &mut s);
            notes.push(format!("section cards: {n} beat(s) open on a card ({:.1} s in all)", s.cards_duration_s()));
        }
        Some(false) => {
            let n = crate::cards::clear(&mut s, true);
            notes.push(format!("section cards: removed {n} generated card(s)"));
        }
        None => {}
    }
    let cfg = &ctx.config.script;
    let opts = script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    let issues = script_check::check(ctx.db, p.id, &s, cfg, &opts)?;
    let errors = issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count();
    let dry_run = a.bool_or("dry_run", false);
    let script_id = if dry_run { None } else { Some(script::save_version(ctx.db, p.id, &s, None)?) };
    let mut out = json!({
        "script_id": script_id,
        "issues": issues,
        "errors": errors,
        "total_duration_s": s.length_s(),
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

static READ_CUT: ToolSpec = ToolSpec {
    name: "read_cut",
    effect: Effect::Read,
    summary: "Read a cut back as text without rendering it: for every beat and clip, the exact words heard, who \
              speaks and who is on screen, whether each cut is clean, the running time against the target and the \
              check's issues with their fixes.",
    when: "After every save_script, before any preview_script: it is the review loop. It costs a read of the index, \
           not a render and a re-transcription. Fix what it flags, save again, read again. Give it a script_id, or a \
           script (as for save_script, sentence ids included) to read a draft before saving it.",
    returns: "Compact text (format json: the same as data). CUT: title, size, length, target. Per beat \
              'b1 \"purpose\" [0:00.0-0:08.2]', its bed, then each clip: '1. v104 file.mp4 src 12:03.4-12:09.8 \
              (6.4 s) at 0:00.0'; 'hears: Freddy: s412 Esto es O ‹Corres›...' (… where a sentence is cut short, ‹ › \
              on an edge word the cut is on); 'sees: single face (Freddy) · crop: ...' (two shot, single face, screen \
              share, no face, from the same scenes as get_scenes; how a 9:16 or 1:1 crop frames it); 'story: ...' \
              when the clip starts or ends mid-sentence, opens on a reply or connective that leans on a line not in \
              the cut, or ends on a question whose answer is cut; and the cut points from the measured speech: \
              'in: clean (in silence) · out: clips the last word' (or next speaker leaks in, starts mid-word). Then \
              TOTAL against the target, and ISSUES with their fixes (apply_fix).",
    misuse: &[
        "Do not answer a flag with a hold or by pulling the cut back by hand: save the script again with the same \
         sentences, or apply the fix an issue offers; the exact cut points are placed by code.",
        "'speech not measured' means the speech stage has not run for that video, so its cut points are judged by \
         the transcript only: index with the speech stage first.",
    ],
    chains_with: &[
        "save_script / apply_fix to change what it flags; get_frames only when the picture decides; preview_script \
         once the reading is clean.",
    ],
    params: &[
        ToolParam::new("script_id", T::Integer, false, "A saved version, from list_scripts or save_script."),
        ToolParam::new(
            "script",
            T::Object,
            false,
            "Instead of script_id: a script as for save_script (needs project), read as save_script would save it.",
        ),
        ToolParam::new(
            "project",
            T::String,
            false,
            "Project name, for an inline script; a saved script_id knows its own.",
        ),
        ToolParam::new("format", T::Enum(&["text", "json"]), false, "default text: the compact reading."),
        ToolParam::new(
            "captions",
            T::Boolean,
            false,
            "default false: also list the captions an mp4 burns in ('v104 src 1016.24-1019.10 [0:03.1] text'); \
             fix one with caption_edits {video_id, src_start_s, src_end_s, text} quoting that src range.",
        ),
    ],
};

async fn read_cut(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let (project_id, s, min) = if let Some(id) = a.opt_i64("script_id") {
        let stored = script::load(ctx.db, id)?;
        let min = crate::clips::length_range(ctx.db, stored.project_id, &stored.title).map(|r| r.0);
        (stored.project_id, stored.script, min)
    } else if let Some(raw) = a.get("script") {
        let p = ctx.project_for(a)?;
        let (s, min, _) = prepare_script(ctx, &p, raw, true)?;
        (p.id, s, min)
    } else {
        return Err(ToolError::invalid("give script_id (a saved version) or script (a draft)"));
    };
    let cfg = &ctx.config.script;
    let opts = script_check::CheckOptions { trim_min_s: min, ..Default::default() };
    let issues = script_check::check(ctx.db, project_id, &s, cfg, &opts)?;
    let reading = crate::readcut::read_cut(ctx.db, project_id, &s, cfg, issues)?;
    let flagged = reading
        .beats
        .iter()
        .flat_map(|b| &b.clips)
        .filter(|c| c.start.is_some_and(|x| !x.flag.is_clean()) || c.end.is_some_and(|x| !x.flag.is_clean()))
        .count();
    let summary = format!("{:.1} s, {} issues, {flagged} flagged cuts", reading.total_s, reading.issues.len());
    let cues =
        if a.bool_or("captions", false) { crate::captions::all_cues_for_script(ctx.db, &s)? } else { Vec::new() };
    let mut value = serde_json::to_value(&reading)?;
    if a.bool_or("captions", false) {
        value["captions"] = serde_json::to_value(&cues)?;
    }
    let mut out = ToolOutput::json(value).summary(summary);
    if a.opt_str("format").unwrap_or("text") == "text" {
        let mut text = reading.render();
        if a.bool_or("captions", false) {
            text.push_str(&captions_text(&cues));
        }
        out = out.text(text);
    }
    Ok(out)
}

/// The caption cues as `read_cut` lists them: each names itself the way a caption edit does.
fn captions_text(cues: &[crate::captions::Cue]) -> String {
    let mut out = format!("CAPTIONS {}\n", cues.len());
    for c in cues {
        let (Some(v), Some(a), Some(b)) = (c.video_id, c.src_start_s, c.src_end_s) else { continue };
        let hidden = if c.hidden { " (hidden)" } else { "" };
        out.push_str(&format!(
            "  v{v} src {a:.2}-{b:.2} [{}] {}{hidden}\n",
            crate::sentences::clock(c.start_s),
            c.text.replace('\n', " / ")
        ));
    }
    out
}

static EXPORT_SCRIPT: ToolSpec = ToolSpec {
    name: "export_script",
    effect: Effect::Write,
    summary: "Export a script version: fcp_xml (Premiere Pro) or otio as an editable timeline, or mp4 as the finished \
              full-size video.",
    when: "When the cut is ready: a timeline to finish in an editor, or format mp4 for the deliverable. mp4 renders at \
           the cut's own size (e.g. 1080x1920) with the script's titles and speech captions burnt in and every clip at \
           the same loudness, the way a clip set is exported; it takes minutes. For a light 540p look while you edit \
           use preview_script. Writes a file at `out`.",
    returns: "{path, format}; for mp4 also {duration_s, encoder}.",
    misuse: &[
        "An existing file at `out` is overwritten.",
        "mp4 re-encodes the real footage: do not start two at once.",
    ],
    chains_with: &["read_cut / preview_script to check the cut before exporting."],
    params: &[
        ToolParam::new("script_id", T::Integer, true, ""),
        ToolParam::new("format", T::Enum(&["fcp_xml", "otio", "mp4"]), false, "default fcp_xml."),
        ToolParam::new("out", T::String, true, "Output file path."),
        ToolParam::new(
            "burn_captions",
            T::Boolean,
            false,
            "mp4 only, default false: burn in the speech captions (read_cut captions: true lists them). A cut \
             that carries a caption style (clips do) burns them either way.",
        ),
    ],
};

async fn export_script(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    ctx.check_cancel()?;
    let script_id = a.i64("script_id")?;
    let format = a.opt_str("format").unwrap_or("fcp_xml");
    let out = PathBuf::from(a.str("out")?);
    if format.trim().trim_start_matches('.').eq_ignore_ascii_case("mp4") {
        let ffmpeg = crate::doctor::locate("ffmpeg").ok_or_else(|| ToolError::invalid("ffmpeg not found"))?;
        let mut opts = crate::preview::PreviewOptions::deliverable(&ctx.config.script, out, ctx.cancel.clone());
        opts.burn_captions = a.bool_or("burn_captions", false);
        let res = crate::preview::render_preview(ctx.db, &ctx.paths.data_dir, &ffmpeg, script_id, &opts, |_, _| {})?;
        return Ok(ToolOutput::json(json!({
            "path": res.path,
            "format": "mp4",
            "duration_s": res.duration_s,
            "encoder": res.encoder,
        }))
        .summary("exported mp4"));
    }
    let fmt = crate::export::ExportFormat::from_str(format)?;
    let res = crate::export::export_script(ctx.db, script_id, fmt, &out)?;
    Ok(ToolOutput::json(json!({ "path": res.path, "format": format })).summary(format!("exported {format}")))
}

static PREVIEW_SCRIPT: ToolSpec = ToolSpec {
    name: "preview_script",
    effect: Effect::Write,
    summary: "Render a script version to a light preview MP4 (540p unless full_size) and return its path (and \
              duration/encoder).",
    when: "To watch a cut, after read_cut is clean. Slow: it cuts and encodes the real footage. Optionally burn in the \
           on-screen titles and the narration as subtitles. For the finished full-size file use export_script with \
           format mp4 (titles, captions and loudness set for delivery).",
    returns: "The render: path, duration and encoder.",
    misuse: &["Rendering a long cut takes minutes and can be stopped; do not start two at once."],
    chains_with: &[],
    params: &[
        ToolParam::new("script_id", T::Integer, true, ""),
        ToolParam::new("burn_titles", T::Boolean, false, "default false."),
        ToolParam::new("burn_narration", T::Boolean, false, "default false: narration as burnt-in subtitles."),
        ToolParam::new(
            "burn_captions",
            T::Boolean,
            false,
            "default false: speech captions from transcript as burnt-in subtitles.",
        ),
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

static REVIEW_RENDER: ToolSpec = ToolSpec {
    name: "review_render",
    effect: Effect::Read,
    summary: "Watch and listen to a render the way a viewer gets it: the words heard in its audio (transcribed \
              again from the file), the silence at every cut, and a contact sheet of its frames.",
    when: "After preview_script or export_script, before you call the cut done: read_cut reads the index, this reads \
           the file. Proofread the heard words as a viewer who never saw the source.",
    returns: "Text: the silence at the head and tail of the file, each cut with the silence before/after it in \
              seconds (0 = inside sound), then every heard line \
              as '[0:12.4 clip 3] words', marked when no voice was measured under it (the transcriber inventing a \
              word in a pause, not a leak); two images: a frame every 2 s, timestamped, and the frames around every cut \
              (where a dissolve or a flash of another picture shows) with the first and last frame. Measurements, \
              not verdicts.",
    misuse: &["It reviews a file that exists: render the version first."],
    chains_with: &["save_script / apply_fix to change what you saw, then render and review again."],
    params: &[
        ToolParam::new("script_id", T::Integer, true, "The version the render was made from."),
        ToolParam::new("path", T::String, false, "The render; default its newest preview or export."),
    ],
};

async fn review_render(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    ctx.check_cancel()?;
    let script_id = a.i64("script_id")?;
    let stored = script::load(ctx.db, script_id)?;
    let path = match a.opt_str("path") {
        Some(p) => PathBuf::from(p),
        None => crate::render_review::latest_render(ctx.db, script_id).ok_or_else(|| {
            ToolError::invalid(format!(
                "script {script_id} has no render yet: preview_script or export_script it first"
            ))
        })?,
    };
    if !path.is_file() {
        return Err(ToolError::invalid(format!("no file at {}", path.display())));
    }
    let starts = crate::render_review::clip_starts(ctx.db, stored.project_id, &stored.script)?;
    let length = stored.script.length_s();
    let review = crate::render_review::review(ctx.paths, ctx.config, &starts, length, &path).await?;
    let text = review.render();
    let mut out = ToolOutput::json(serde_json::to_value(&review)?).text(text).summary(format!(
        "{} cuts, {} lines heard",
        review.cuts.len(),
        review.heard.len()
    ));
    if ctx.can_see {
        let mut images = Vec::new();
        if let Some(jpeg) = review.sheet_jpeg.clone() {
            images.push(super::ToolImage {
                caption: format!("render of script {script_id}: a frame every 2 s"),
                jpeg,
                name: format!("script{script_id}_sheet.jpg"),
            });
        }
        if let Some(jpeg) = review.cuts_jpeg.clone() {
            images.push(super::ToolImage {
                caption: format!(
                    "render of script {script_id}: the first frame, the frames around every cut (0.1 s before, \
                     0.1, 0.5 and 1.0 s after) and the last frame"
                ),
                jpeg,
                name: format!("script{script_id}_cuts.jpg"),
            });
        }
        out = out.images(images);
    }
    Ok(out)
}
