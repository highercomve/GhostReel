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
    summary: "Sentence-bounded windows in the length range that could be clips, found by code from the stored \
              transcripts (no model): never starting off-mic, never across a transcript hole, not on shaky stretches.",
    when: "To orchestrate your own clip picking (e.g. with subagents), or to see what generate_clips would choose \
           from. Needs transcripts.",
    returns: "{total, candidates}: each candidate has an id, video_id, in_s/out_s, text, a visual label (faces / \
              screen / mixed, from keyframe descriptions), face numbers from face tracking, speakers when known, \
              and a code prescore. Best prescore first, at most `limit`.",
    misuse: &["A candidate is where speech is clean, not a judgement that it is good: read its transcript before saving it."],
    chains_with: &["get_transcript around the ones you like, then check_clips, then save_clips."],
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
    let picks: Vec<crate::clips::Pick> = serde_json::from_value(raw.clone())
        .map_err(|e| ToolError::Invalid(format!("clips must be [{{video_id, in_s, out_s, title?, hook?, ...}}]: {e}")))?;
    if picks.is_empty() {
        return Err(ToolError::invalid("no clips given"));
    }
    Ok(picks)
}

async fn find_clip_candidates(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
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
    summary: "Dry run of save_clips: builds each clip's script, runs the same repair passes `script import` uses \
              (snap to segments, whole sentences, closing hold) and the script checker, plus overlaps between clips.",
    when: "Before save_clips, on the picks you intend to keep. Saves nothing.",
    returns: "{clips: [{video_id, in_s, out_s, visual, notes, issues, errors}]}: the repaired bounds and the issues.",
    misuse: &[],
    chains_with: &["save_clips with the same list once errors is 0."],
    params: &[
        PROJECT,
        ToolParam::new("clips", T::List(&T::Object), true, "[{video_id, in_s, out_s, title?, hook?, quote?, why?, score?}]"),
        ToolParam::new("format", FORMAT, false, ""),
        ToolParam::new("min_s", T::Number, false, ""),
        ToolParam::new("max_s", T::Number, false, ""),
    ],
};

async fn check_clips(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    let mut picks = clip_picks(a)?;
    for pk in &mut picks {
        pk.visual = crate::clips::VideoSurvey::load(ctx.db, pk.video_id)?.visual(pk.in_s, pk.out_s);
    }
    let checked = crate::clips::check_clips(ctx.db, &p, &ctx.config.script, &picks, &opts)?;
    let errors: usize =
        checked.iter().map(|c| c.issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count()).sum();
    Ok(ToolOutput::json(json!({ "clips": checked.iter().map(|c| json!({
        "video_id": c.pick.video_id, "in_s": c.pick.in_s, "out_s": c.pick.out_s,
        "visual": c.pick.visual, "notes": c.notes, "issues": c.issues,
        "errors": c.issues.iter().filter(|i| i.severity == script::IssueSeverity::Error).count(),
    })).collect::<Vec<_>>() }))
    .summary(format!("{} clips, {errors} errors", checked.len())))
}

static SAVE_CLIPS: ToolSpec = ToolSpec {
    name: "save_clips",
    effect: Effect::Write,
    summary: "Save clips you picked as a new clip set (one-clip scripts, kind 'clip', speech captions on, hook as \
              the on-screen title, chosen frame size).",
    when: "After check_clips. Same checks as check_clips; clips with errors are skipped.",
    returns: "{set_id, saved, clips: [{script_id, title, in_s, out_s, issues}]}.",
    misuse: &["Every call makes a new set: saving the same picks twice gives two sets."],
    chains_with: &["list_clip_sets to see the set with its live checks."],
    params: &[
        PROJECT,
        ToolParam::new("clips", T::List(&T::Object), true, "[{video_id, in_s, out_s, title, hook, quote?, why?, score?}]"),
        ToolParam::new("brief", T::String, false, ""),
        ToolParam::new("format", FORMAT, false, ""),
        ToolParam::new("min_s", T::Number, false, ""),
        ToolParam::new("max_s", T::Number, false, ""),
    ],
};

async fn save_clips(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let opts = clip_options(a)?;
    let mut picks = clip_picks(a)?;
    for pk in &mut picks {
        pk.visual = crate::clips::VideoSurvey::load(ctx.db, pk.video_id)?.visual(pk.in_s, pk.out_s);
    }
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
            "script_id": c.script_id, "title": c.script.title, "in_s": c.pick.in_s, "out_s": c.pick.out_s,
            "issues": c.issues,
        })).collect::<Vec<_>>(),
    }))
    .summary(format!("saved {} clips", set.items.len())))
}

static GENERATE_CLIPS: ToolSpec = ToolSpec {
    name: "generate_clips",
    effect: Effect::Write,
    summary: "Whole clip run: mine sentence-bounded candidates from the project's videos, score them with the worker \
              agents in parallel (spread over every selected agent's free slots), let the orchestrator agent choose N \
              and write hooks, check, and save a clip set of one-clip scripts (speech captions on, hook as the \
              on-screen title).",
    when: "When GhostReel's own models should do the picking. Slow: minutes. Agents default to the app's choice (see \
           list_agents). Not available from inside a clip run or a script chat, which are model runs already.",
    returns: "The set, each clip's script id and issues, and the run log with timings and which agent ran each job.",
    misuse: &["Called from inside a clip run or a script chat it is refused: pick with find_clip_candidates and save_clips."],
    chains_with: &["list_clip_sets for the saved set."],
    params: &[
        PROJECT,
        ToolParam::new("brief", T::String, false, "Optional editorial brief."),
        ToolParam::new("count", T::Integer, false, "default 8."),
        ToolParam::new("min_s", T::Number, false, "default 20."),
        ToolParam::new("max_s", T::Number, false, "default 60."),
        ToolParam::new("format", FORMAT, false, "default original."),
        ToolParam::new("video_ids", T::List(&T::Integer), false, "default: the whole library."),
        ToolParam::new("worker_only", T::Boolean, false, "default false: code picks instead of the orchestrator model."),
        ToolParam::new(
            "layout",
            T::Enum(&["follow_speaker", "stacked", "fixed"]),
            false,
            "framing for 9:16 / 1:1 (default follow_speaker: crop on whoever talks).",
        ),
        ToolParam::new("orchestrator", T::String, false, "agent id that picks the final clips (default: the app's choice)."),
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
