//! Short social clips: finding candidates, checking picks, saving a set, and the whole run.

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, ParamType as T, ToolParam, ToolSpec};
use super::{Args, Tool, ToolCtx, ToolError, ToolOutput, ToolResult, tool};
use crate::script;

pub static TOOLS: [Tool; 5] = [
    tool!(FIND_CLIP_CANDIDATES, find_clip_candidates),
    tool!(CHECK_CLIPS, check_clips),
    tool!(SAVE_CLIPS, save_clips),
    tool!(GENERATE_CLIPS, generate_clips),
    tool!(LIST_CLIP_SETS, list_clip_sets),
];

const FORMAT: T = T::Enum(&["original", "9:16", "1:1"]);

static FIND_CLIP_CANDIDATES: ToolSpec = ToolSpec {
    name: "find_clip_candidates",
    effect: Effect::Read,
    summary: "Moments: the spoken pieces a cut or a clip is made of, found by code (no model): whole sentences in \
              the length range, nobody off the microphone in them, never across a transcript hole or a shaky stretch.",
    when: "Choosing what a cut's voices say (min_s/max_s: one spoken beat, about 4-15 s for a 60 s cut), or a clip \
           run (20-60 s). Needs transcripts.",
    returns: "{total, candidates}: each has an id, video_id, from/to sentence ids, in_s/out_s, text, a visual label \
              (faces / screen / mixed), face numbers, speakers when known, and a code prescore. Best first.",
    misuse: &[
        "A moment is clean speech, not a good line: read its words and pictures (get_manifest) before choosing it. \
         Its id is per call; a cut uses its video_id and from/to.",
    ],
    chains_with: &["get_manifest, then save_script clips {video_id, from, to}; or check_clips, then save_clips."],
    params: &[
        PROJECT,
        ToolParam::new("min_s", T::Number, false, "default 20."),
        ToolParam::new("max_s", T::Number, false, "default 60."),
        ToolParam::new("format", FORMAT, false, "default original."),
        ToolParam::new("video_ids", T::List(&T::Integer), false, "default: the whole library."),
        ToolParam::new("limit", T::Integer, false, "default 200, best prescore first."),
    ],
};

fn clip_options(a: Args<'_>) -> Result<crate::clips::ClipOptions, ToolError> {
    use crate::clips::{ClipFormat, ClipOptions};
    let d = ClipOptions::default();
    let opts = ClipOptions {
        brief: a.opt_str("brief").unwrap_or_default().to_string(),
        count: a.opt_u64("count").map(|v| v as usize).unwrap_or(d.count),
        min_s: a.opt_f64("min_s").unwrap_or(d.min_s),
        max_s: a.opt_f64("max_s").unwrap_or(d.max_s),
        format: ClipFormat::parse(a.opt_str("format").unwrap_or("original"))?,
        video_ids: a.i64_list("video_ids"),
        worker_only: a.bool_or("worker_only", false),
        layout: match a.opt_str("layout") {
            Some(l) => l.parse()?,
            None => d.layout,
        },
        agents: crate::agents::AgentSelection {
            orchestrator: a.opt_str("orchestrator").map(str::to_string),
            workers: Some(a.str_list("agents")).filter(|w| !w.is_empty()),
            max_parallel: a.opt_u64("max_parallel").map(|v| v as usize),
            ..Default::default()
        },
        ..d
    };
    opts.validate()?;
    Ok(opts)
}

fn clip_picks(a: Args<'_>) -> Result<Vec<crate::clips::Pick>, ToolError> {
    let raw = a.get("clips").ok_or_else(|| ToolError::invalid("missing 'clips' array"))?;
    let picks: Vec<crate::clips::Pick> = serde_json::from_value(raw.clone()).map_err(|e| {
        ToolError::Invalid(format!("clips must be [{{video_id, from?, to?, in_s?, out_s?, title?, hook?, ...}}]: {e}"))
    })?;
    if picks.is_empty() {
        return Err(ToolError::invalid("no clips given"));
    }
    Ok(picks)
}

fn resolve_picks(ctx: &ToolCtx<'_>, picks: &mut [crate::clips::Pick]) -> Result<(), ToolError> {
    for pk in picks.iter() {
        ctx.check_video(pk.video_id)?;
    }
    let db = &*ctx.db;
    let mut surveys: std::collections::HashMap<i64, crate::clips::VideoSurvey> = std::collections::HashMap::new();
    for pk in picks {
        pk.from = pk.from.take().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        pk.to = pk.to.take().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        match (&pk.from, &pk.to) {
            (Some(from), Some(to)) => {
                let (in_s, out_s) = crate::sentences::resolve_range(db, pk.video_id, from, to)?;
                pk.in_s = (in_s * 100.0).round() / 100.0;
                pk.out_s = (out_s * 100.0).round() / 100.0;
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(ToolError::invalid(
                    "clip needs both 'from' and 'to' sentence ids (or 'in_s' and 'out_s' in seconds)",
                ));
            }
            (None, None) if pk.out_s <= pk.in_s => {
                return Err(ToolError::invalid(
                    "clip needs either 'from'/'to' sentence ids or 'in_s'/'out_s' with out_s > in_s",
                ));
            }
            (None, None) => {}
        }
        let survey = match surveys.entry(pk.video_id) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(crate::clips::VideoSurvey::load(db, pk.video_id)?),
        };
        pk.visual = survey.visual(pk.in_s, pk.out_s);
        if pk.speakers.is_empty() {
            pk.speakers = survey.speakers(pk.in_s, pk.out_s);
        }
    }
    Ok(())
}

async fn find_clip_candidates(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    for &id in &opts.video_ids {
        ctx.check_video(id)?;
    }
    let mut cands = crate::clips::find_candidates(ctx.db, p.id, &ctx.config.script, &opts)?;
    let total = cands.len();
    cands.sort_by(|x, y| y.prescore.total_cmp(&x.prescore));
    let limit = a.opt_u64("limit").unwrap_or(200) as usize;
    cands.truncate(limit);
    let mut items: Vec<Value> = cands.iter().map(serde_json::to_value).collect::<Result<_, _>>()?;
    let dropped = super::util::fit_items(&mut items, ctx.room(80));
    let mut out = json!({ "total": total, "candidates": items });
    if dropped > 0 {
        out["truncated"] = json!(format!("{dropped} candidates left out to fit; raise min_s or filter video_ids"));
    }
    Ok(ToolOutput::json(out).summary(format!("{total} candidates")))
}

static CHECK_CLIPS: ToolSpec = ToolSpec {
    name: "check_clips",
    effect: Effect::Read,
    summary: "Dry run of save_clips: resolves `from`/`to` sentence ids (`sX`, `sX@word`), runs repair passes \
              (sentences, closing hold, speech cuts) and checks story, framing, edges, duration and overlaps.",
    when: "Before save_clips (or before finishing a clip run), on picks you plan to keep. Saves nothing.",
    returns: "{clips: [{video_id, from?, to?, in_s, out_s, duration_s, visual, story, crop?, start?, end?, notes, issues}]}.",
    misuse: &[],
    chains_with: &["save_clips with the same list once errors and warnings are addressed."],
    params: &[
        PROJECT,
        ToolParam::new(
            "clips",
            T::List(&T::Object),
            true,
            "[{video_id, from?, to?, in_s?, out_s?, title?, hook?, ...}]",
        ),
        ToolParam::new("format", FORMAT, false, ""),
        ToolParam::new("min_s", T::Number, false, ""),
        ToolParam::new("max_s", T::Number, false, ""),
    ],
};

async fn check_clips(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    let mut picks = clip_picks(a)?;
    resolve_picks(ctx, &mut picks)?;
    let checked = crate::clips::check_clips(ctx.db, &p, &ctx.config.script, &picks, &opts)?;
    let mut total_errors = 0usize;
    let mut total_warnings = 0usize;
    let mut clips_json = Vec::with_capacity(checked.len());
    let mut lines = String::new();
    for (i, c) in checked.iter().enumerate() {
        let reading = crate::readcut::read_cut(ctx.db, p.id, &c.script, &ctx.config.script, c.issues.clone())?;
        let cr = reading.beats.first().and_then(|b| b.clips.first());
        let story = cr.map(|x| x.story.clone()).unwrap_or_default();
        let crop = cr.and_then(|x| x.crop.clone());
        let start_detail = cr.and_then(|x| x.start.as_ref().map(|s| s.detail()));
        let end_detail = cr.and_then(|x| x.end.as_ref().map(|e| e.detail()));
        let errors = c.issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count();
        let warnings = c.issues.iter().filter(|i| i.severity == script::IssueSeverity::Warning).count();
        total_errors += errors;
        total_warnings += warnings;
        clips_json.push(json!({
            "video_id": c.pick.video_id,
            "from": c.pick.from,
            "to": c.pick.to,
            "in_s": c.pick.in_s,
            "out_s": c.pick.out_s,
            "duration_s": (c.pick.duration_s() * 100.0).round() / 100.0,
            "visual": c.pick.visual,
            "story": story,
            "crop": crop,
            "start": cr.and_then(|x| x.start),
            "end": cr.and_then(|x| x.end),
            "notes": c.notes,
            "issues": c.issues,
            "warnings": warnings,
            "errors": errors,
        }));
        let span = match (&c.pick.from, &c.pick.to) {
            (Some(f), Some(t)) => format!(" {f}–{t}"),
            _ => String::new(),
        };
        lines.push_str(&format!(
            "{}. v{}{} {}-{} ({:.1} s) · {} · {errors} errors, {warnings} warnings\n",
            i + 1,
            c.pick.video_id,
            span,
            crate::sentences::clock(c.pick.in_s),
            crate::sentences::clock(c.pick.out_s),
            c.pick.duration_s(),
            c.pick.visual.summary(),
        ));
        if let (Some(s), Some(e)) = (&start_detail, &end_detail) {
            lines.push_str(&format!("   edges: in {s} · out {e}\n"));
        }
        if !story.is_empty() {
            lines.push_str(&format!("   story: {}\n", story.join(" · ")));
        }
        if let Some(cr_text) = &crop
            && (cr_text.contains("face lost") || cr_text.contains("cut off") || cr_text.contains("letterboxed"))
        {
            lines.push_str(&format!("   crop: {cr_text}\n"));
        }
        if !c.notes.is_empty() {
            lines.push_str(&format!("   repaired: {}\n", c.notes.join("; ")));
        }
        if !c.issues.is_empty() {
            let iss: Vec<String> = c
                .issues
                .iter()
                .map(|iss| {
                    let sev = match iss.severity {
                        script::IssueSeverity::Error => "error",
                        script::IssueSeverity::Warning => "warning",
                        script::IssueSeverity::Info => "info",
                    };
                    format!("[{sev}] {}", iss.message)
                })
                .collect();
            lines.push_str(&format!("   issues: {}\n", iss.join("; ")));
        }
    }
    Ok(ToolOutput::json(json!({ "clips": clips_json }))
        .text(lines.trim_end().to_string())
        .summary(format!("{} clips, {total_errors} errors, {total_warnings} warnings", checked.len())))
}

static SAVE_CLIPS: ToolSpec = ToolSpec {
    name: "save_clips",
    effect: Effect::Write,
    summary: "Save picks as a new clip set (one-clip scripts, kind 'clip', captions on, hook title). \
              Takes `from`/`to` sentence ids or `in_s`/`out_s`.",
    when: "After check_clips; clips with errors are skipped. Refused in a clip run.",
    returns: "{set_id, saved, clips: [{script_id, title, from?, to?, in_s, out_s, issues}]}.",
    misuse: &["Every call makes a new set: saving the same picks twice gives two sets."],
    chains_with: &["list_clip_sets to see the set with its live checks."],
    params: &[
        PROJECT,
        ToolParam::new("clips", T::List(&T::Object), true, "[{video_id, from?, to?, in_s?, out_s?, title, hook, ...}]"),
        ToolParam::new("brief", T::String, false, ""),
        ToolParam::new("format", FORMAT, false, ""),
        ToolParam::new("min_s", T::Number, false, ""),
        ToolParam::new("max_s", T::Number, false, ""),
    ],
};

async fn save_clips(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    if ctx.origin == super::Origin::ClipRun {
        return Err(ToolError::invalid(
            "save_clips cannot be called from inside a clip run: return your chosen clips as your final JSON answer instead",
        ));
    }
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    let mut picks = clip_picks(a)?;
    resolve_picks(ctx, &mut picks)?;
    let source = match ctx.origin {
        super::Origin::External => "mcp save_clips",
        super::Origin::ScriptChat => "script chat save_clips",
        super::Origin::ClipRun => "clip run save_clips",
    };
    let (set, checked) =
        crate::clips::save_clips(ctx.db, &p, &ctx.config.script, &picks, &opts, None, json!({ "source": source }))?;
    Ok(ToolOutput::json(json!({
        "set_id": set.id,
        "saved": set.items.len(),
        "clips": checked.iter().map(|c| json!({
            "script_id": c.script_id,
            "title": c.script.title,
            "from": c.pick.from,
            "to": c.pick.to,
            "in_s": c.pick.in_s,
            "out_s": c.pick.out_s,
            "issues": c.issues,
        })).collect::<Vec<_>>(),
    }))
    .summary(format!("saved {} clips", set.items.len())))
}

static GENERATE_CLIPS: ToolSpec = ToolSpec {
    name: "generate_clips",
    effect: Effect::Write,
    summary: "Whole clip run: mine sentence-bounded candidates, score with worker agents in parallel, let the \
              orchestrator pick N and write hooks, check, and save a clip set (captions on, hook title).",
    when: "When GhostReel's models should pick. Slow (minutes). Refused inside a clip run or script chat.",
    returns: "The set, each clip's script id and issues, and the run log with timings.",
    misuse: &["Refused inside a clip run or script chat: pick with find_clip_candidates and save_clips."],
    chains_with: &["list_clip_sets for the saved set."],
    params: &[
        PROJECT,
        ToolParam::new("brief", T::String, false, "Optional editorial brief."),
        ToolParam::new("count", T::Integer, false, "default 8."),
        ToolParam::new("min_s", T::Number, false, "default 20."),
        ToolParam::new("max_s", T::Number, false, "default 60."),
        ToolParam::new("format", FORMAT, false, "default original."),
        ToolParam::new("video_ids", T::List(&T::Integer), false, "default: the whole library."),
        ToolParam::new(
            "worker_only",
            T::Boolean,
            false,
            "default false: code picks instead of the orchestrator model.",
        ),
        ToolParam::new(
            "layout",
            T::Enum(&["follow_speaker", "stacked", "fixed"]),
            false,
            "framing for 9:16 / 1:1 (default follow_speaker: crop on whoever talks).",
        ),
        ToolParam::new(
            "orchestrator",
            T::String,
            false,
            "agent id that picks the final clips (default: the app's choice).",
        ),
        ToolParam::new("agents", T::List(&T::String), false, "worker agent ids (default: the app's choice)."),
        ToolParam::new(
            "max_parallel",
            T::Integer,
            false,
            "most worker jobs at once across all agents (default: their summed capacity).",
        ),
    ],
};

async fn generate_clips(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    ctx.refuse_nested("generate_clips")?;
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    for &id in &opts.video_ids {
        ctx.check_video(id)?;
    }
    let backends = crate::clips::Backends::resolve(ctx.paths, ctx.config, &opts).await?;
    let res = crate::clips::run(
        ctx.db,
        p.id,
        &ctx.config.script,
        &backends,
        &opts,
        ctx.cancel.clone(),
        &mut |_: f64, _: &str, _: &str| {},
    )
    .await?;
    let summary = format!("saved {} clips", res.set.items.len());
    Ok(ToolOutput::json(serde_json::to_value(res)?).summary(summary))
}

static LIST_CLIP_SETS: ToolSpec = ToolSpec {
    name: "list_clip_sets",
    effect: Effect::Read,
    summary: "A project's clip sets (newest first): brief, options, run log, and each clip's script id, title, hook, \
              quote, why, score, bounds, visual label and accepted/rejected status.",
    when: "To see what a run or save_clips made. Each clip's warnings and issues are as of now, not as the run saw them.",
    returns: "One entry per set with its clips.",
    misuse: &[],
    chains_with: &["get_script(script_id) for one clip's full script."],
    params: &[PROJECT],
};

async fn list_clip_sets(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let mut sets = crate::clips::list_sets(ctx.db, p.id)?;
    for set in &mut sets {
        crate::clips::with_live_checks(ctx.db, &ctx.config.script, set);
    }
    let n = sets.len();
    Ok(ToolOutput::json(serde_json::to_value(sets)?).summary(format!("{n} clip sets")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn foreign_picks_are_rejected_before_sentence_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = crate::paths::Paths::in_dir(tmp.path());
        let config = crate::config::Config::default();
        let mut db = crate::db::Db::open_in_memory().unwrap();
        let p = db.create_project(&crate::projects::NewProject::named("Scoped")).unwrap();
        let mut embedder = None;
        let mut ctx =
            ToolCtx::new(&paths, &config, &mut db, &mut embedder, super::super::Origin::External).with_project(p);
        for pick in [json!({"video_id":99,"from":"s999","to":"s999"}), json!({"video_id":99,"in_s":0,"out_s":30})] {
            let args = json!({"clips":[pick]});
            for result in [check_clips(&mut ctx, Args(&args)).await, save_clips(&mut ctx, Args(&args)).await] {
                assert!(result.unwrap_err().to_string().contains("not in project"));
            }
        }
    }
}
