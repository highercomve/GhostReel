//! Projects and their index: creating, watching folders, indexing, the glossary, which videos count.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, ParamType as T, ToolParam, ToolSpec};
use super::{Args, Tool, ToolCtx, ToolError, ToolOutput, ToolResult, tool};
use crate::projects::{NewProject, PipelineConfig};
use crate::{index, script};

pub static TOOLS: [Tool; 10] = [
    tool!(DOCTOR, doctor),
    tool!(LIST_PROJECTS, list_projects),
    tool!(CREATE_PROJECT, create_project),
    tool!(DELETE_PROJECT, delete_project),
    tool!(ADD_FOLDER, add_folder),
    tool!(INDEX, index_tool),
    tool!(INDEX_STATUS, index_status),
    tool!(GET_GLOSSARY, get_glossary),
    tool!(SET_GLOSSARY, set_glossary),
    tool!(SET_VIDEO_ROLE, set_video_role),
];

static DOCTOR: ToolSpec = ToolSpec {
    name: "doctor",
    effect: Effect::Read,
    summary: "What GhostReel would use on this machine: ffmpeg, GPU, database, models, AI backends.",
    when: "When something fails to start, or before a long run, to see what is missing.",
    returns: "A report of each component: found or not, its path or version, and what to do about it.",
    misuse: &[],
    chains_with: &["get_settings / set_settings to change the AI backends."],
    params: &[],
};

async fn doctor(ctx: &mut ToolCtx<'_>, _a: Args<'_>) -> ToolResult {
    Ok(ToolOutput::json(serde_json::to_value(crate::doctor::run(ctx.paths).await)?).summary("doctor report"))
}

static LIST_PROJECTS: ToolSpec = ToolSpec {
    name: "list_projects",
    effect: Effect::Read,
    summary: "Every project with its sequence settings and indexing progress.",
    when: "To find a project's name, or to see how far its indexing is.",
    returns: "One entry per project: {project: {id, name, fps, width, height, …}, status: per-stage progress}.",
    misuse: &[],
    chains_with: &["index to continue indexing a project with pending work."],
    params: &[],
};

async fn list_projects(ctx: &mut ToolCtx<'_>, _a: Args<'_>) -> ToolResult {
    let mut out = Vec::new();
    for p in ctx.db.projects()? {
        let st = index::status(ctx.db, Some(p.id))?;
        out.push(json!({ "project": p, "status": st }));
    }
    let n = out.len();
    Ok(ToolOutput::json(json!(out)).summary(format!("{n} projects")))
}

static CREATE_PROJECT: ToolSpec = ToolSpec {
    name: "create_project",
    effect: Effect::Write,
    summary: "Create a project. fps/width/height are the timeline settings used by exports.",
    when: "First, before add_folder.",
    returns: "The new project.",
    misuse: &["A name that already exists is refused."],
    chains_with: &["add_folder, then index."],
    params: &[
        ToolParam::new("name", T::String, true, ""),
        ToolParam::new("description", T::String, false, ""),
        ToolParam::new("fps", T::Number, false, "e.g. 25, 30, 29.97 (default 25)."),
        ToolParam::new("width", T::Integer, false, "default 1920."),
        ToolParam::new("height", T::Integer, false, "default 1080."),
    ],
};

async fn create_project(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let (fps_num, fps_den) = match a.opt_f64("fps") {
        Some(f) => {
            let fps = script::Fps::from_f64(f);
            (fps.num, fps.den)
        }
        None => (25, 1),
    };
    let mut pipeline = PipelineConfig::default();
    if a.opt_bool("no_transcribe") == Some(true) {
        pipeline.transcribe = false;
    }
    if let Some(t) = a.opt_bool("transcribe") {
        pipeline.transcribe = t;
    }
    let p = NewProject {
        name: a.str("name")?.to_string(),
        description: a.opt_str("description").unwrap_or_default().to_string(),
        fps_num,
        fps_den,
        width: a.opt_i64("width").unwrap_or(1920),
        height: a.opt_i64("height").unwrap_or(1080),
        pipeline: Some(pipeline),
    };
    let project = ctx.db.create_project(&p)?;
    let summary = format!("created {}", project.name);
    Ok(ToolOutput::json(serde_json::to_value(project)?).summary(summary))
}

static DELETE_PROJECT: ToolSpec = ToolSpec {
    name: "delete_project",
    effect: Effect::Destructive,
    summary: "Delete a project. With purge=true also delete its keyframes, preview renders and the \
              transcripts/descriptions/vectors of footage no other project uses. Video files are never touched.",
    when: "Only when the user asked for this project to go.",
    returns: "{removed, purged}: the project's name and what purge removed.",
    misuse: &[
        "The project's cuts and clip sets go with it and do not pass through the trash (the trash holds a project's \
         cuts, versions and clip sets while the project exists). purge=true cannot be undone either.",
    ],
    chains_with: &[],
    params: &[
        PROJECT,
        ToolParam::new("purge", T::Boolean, false, "default false: keep the index for reuse."),
    ],
};

async fn delete_project(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let purge = a.bool_or("purge", false);
    let stats = if purge { ctx.db.purge_project_data(&ctx.paths.data_dir, p.id)? } else { Default::default() };
    ctx.db.remove_project(p.id)?;
    Ok(ToolOutput::json(json!({ "removed": p.name, "purged": stats })).summary(format!("deleted {}", p.name)))
}

static ADD_FOLDER: ToolSpec = ToolSpec {
    name: "add_folder",
    effect: Effect::Write,
    summary: "Watch a folder of footage for a project (whole tree by default). Editor render/cache folders are skipped.",
    when: "After create_project. Nothing is read until index runs.",
    returns: "{folder, next}.",
    misuse: &["A folder nested inside one the project already watches is refused: files would belong to two folders."],
    chains_with: &["index to scan it."],
    params: &[
        PROJECT,
        ToolParam::new("path", T::String, true, "Absolute path of the folder."),
        ToolParam::new("recursive", T::Boolean, false, "default true."),
    ],
};

async fn add_folder(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let path = PathBuf::from(a.str("path")?);
    let folder = ctx.db.add_folder(p.id, &path, a.bool_or("recursive", true))?;
    Ok(ToolOutput::json(json!({ "folder": folder, "next": "call index to scan it" })).summary("folder added"))
}

/// A cancel flag that follows the caller's and also trips after a deadline. The index stops on the
/// flag it is given, so the caller's Stop and `max_seconds` both reach it.
struct LinkedCancel {
    flag: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl LinkedCancel {
    fn new(parent: Option<Arc<AtomicBool>>, deadline: Option<Duration>) -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        let f = flag.clone();
        let started = Instant::now();
        let task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(200)).await;
                if parent.as_ref().is_some_and(|p| p.load(Ordering::SeqCst))
                    || deadline.is_some_and(|d| started.elapsed() >= d)
                {
                    f.store(true, Ordering::SeqCst);
                    break;
                }
            }
        });
        Self { flag, task }
    }
}

impl Drop for LinkedCancel {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Run one indexing stage for a project now (an issue's run_stage fix), under the indexer lock.
pub(super) async fn run_stage(ctx: &mut ToolCtx<'_>, project_id: i64, stage: &str) -> Result<Value, ToolError> {
    let rt = crate::runtime::resolve(ctx.paths, ctx.config).await?;
    let link = LinkedCancel::new(ctx.cancel.clone(), None);
    let opts = index::Options {
        project_id: Some(project_id),
        stages: Some(vec![stage.to_string()]),
        cancel: Some(link.flag.clone()),
        ..Default::default()
    };
    let _lock = index::IndexLock::acquire(&ctx.paths.data_dir)?;
    let summary = index::run_pipeline(ctx.db, ctx.paths, ctx.config, &rt, &opts, |_| {}).await?;
    ctx.check_cancel()?;
    Ok(serde_json::to_value(summary)?)
}

static INDEX: ToolSpec = ToolSpec {
    name: "index",
    effect: Effect::Write,
    summary: "Run pending indexing work (scan, transcribe, keyframes, descriptions, embeddings, faces, speakers) for \
              up to max_seconds, then report progress.",
    when: "After add_folder, and again whenever it reports pending work. Resumable. The glossary is used when \
           transcribing; after changing it, redo: transcribe applies it to videos already transcribed.",
    returns: "{summary, status, pending_jobs, next}: next says whether to call again.",
    misuse: &[
        "redo resets that stage and every later one for the project first: the derived data is rebuilt, which costs the \
         time it took the first time.",
        "Another GhostReel process indexing at the same time is refused with a message, not queued.",
    ],
    chains_with: &["index_status to look without working; call index again while pending_jobs > 0."],
    params: &[
        PROJECT,
        ToolParam::new("max_seconds", T::Integer, false, "Stop after this long (default 900)."),
        ToolParam::new("redo", T::String, false, "Reset this stage and later ones first: frames | transcribe | describe | embed."),
    ],
};

async fn index_tool(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    if let Some(stage) = a.opt_str("redo") {
        if !index::STAGES.contains(&stage) {
            return Err(ToolError::Invalid(format!("unknown stage '{stage}'; valid: {}", index::STAGES.join(", "))));
        }
        index::reset_stages(ctx.db, &ctx.paths.data_dir, Some(p.id), stage)?;
    }
    let max_seconds = a.opt_u64("max_seconds").unwrap_or(900);
    let rt = crate::runtime::resolve(ctx.paths, ctx.config).await?;
    let link = LinkedCancel::new(ctx.cancel.clone(), Some(Duration::from_secs(max_seconds)));
    let opts = index::Options { project_id: Some(p.id), cancel: Some(link.flag.clone()), ..Default::default() };
    // The app or a CLI run may be indexing: say so rather than run the pipeline twice.
    let _lock = index::IndexLock::acquire(&ctx.paths.data_dir)?;
    let summary = index::run_pipeline(ctx.db, ctx.paths, ctx.config, &rt, &opts, |_| {}).await?;
    drop(link);
    ctx.check_cancel()?;
    let status = index::status(ctx.db, Some(p.id))?;
    let pending: i64 = status.stages.iter().map(|s| s.pending).sum();
    Ok(ToolOutput::json(json!({
        "summary": summary,
        "status": status,
        "pending_jobs": pending,
        "next": if pending > 0 { "call index again to continue" } else { "indexing complete" },
    }))
    .summary(format!("{pending} jobs pending")))
}

static INDEX_STATUS: ToolSpec = ToolSpec {
    name: "index_status",
    effect: Effect::Read,
    summary: "Indexing progress per stage, and per-video state.",
    when: "To check how far indexing is without doing any of it.",
    returns: "{status, videos}: status per stage (done / pending / failed); videos only when asked for.",
    misuse: &[],
    chains_with: &["index to do the pending work."],
    params: &[PROJECT, ToolParam::new("videos", T::Boolean, false, "Include the video list.")],
};

async fn index_status(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let status = index::status(ctx.db, Some(p.id))?;
    let pending: i64 = status.stages.iter().map(|s| s.pending).sum();
    let videos = if a.bool_or("videos", false) { serde_json::to_value(index::videos(ctx.db, Some(p.id))?)? } else { Value::Null };
    Ok(ToolOutput::json(json!({ "status": status, "videos": videos })).summary(format!("{pending} jobs pending")))
}

static GET_GLOSSARY: ToolSpec = ToolSpec {
    name: "get_glossary",
    effect: Effect::Read,
    summary: "Get a project's transcription glossary (terms Whisper should spell correctly; heard_as are known \
              mis-hearings).",
    when: "Before transcribing, or when a name keeps coming out wrong in the transcripts.",
    returns: "The terms: {term, heard_as, note}.",
    misuse: &[],
    chains_with: &["set_glossary to change it; then index with redo: transcribe to apply it."],
    params: &[PROJECT],
};

async fn get_glossary(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let n = p.glossary.terms.len();
    Ok(ToolOutput::json(serde_json::to_value(&p.glossary)?).summary(format!("{n} terms")))
}

const TERM_FIELDS: &[ToolParam] = &[
    ToolParam::new("term", T::String, true, ""),
    ToolParam::new("heard_as", T::List(&T::String), false, ""),
    ToolParam::new("note", T::String, false, ""),
];

static SET_GLOSSARY: ToolSpec = ToolSpec {
    name: "set_glossary",
    effect: Effect::Write,
    summary: "Update a project's transcription glossary (terms Whisper should spell correctly; heard_as are known \
              mis-hearings): add, remove, or replace terms.",
    when: "When a name, brand or jargon is mis-transcribed. Used when transcribing; after changing it, re-run index \
           with redo: transcribe to apply it to videos already transcribed.",
    returns: "The glossary as saved.",
    misuse: &["replace swaps the whole list: terms you leave out are gone."],
    chains_with: &["index with redo: transcribe."],
    params: &[
        PROJECT,
        ToolParam::new("add", T::List(&T::Shape(TERM_FIELDS)), false, "Terms to add or update."),
        ToolParam::new("remove", T::List(&T::String), false, "Terms to remove."),
        ToolParam::new("replace", T::List(&T::Shape(TERM_FIELDS)), false, "Full replacement list of terms."),
    ],
};

async fn set_glossary(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    use crate::glossary::{Glossary, GlossaryTerm};
    let p = ctx.project_for(a)?;
    let mut glossary = if let Some(replace) = a.get("replace") {
        let terms: Vec<GlossaryTerm> = serde_json::from_value(replace.clone())
            .map_err(|e| ToolError::Invalid(format!("invalid terms format in replace: {e}")))?;
        Glossary { terms }
    } else {
        p.glossary.clone()
    };
    for r in a.str_list("remove") {
        glossary.remove(&r);
    }
    if let Some(add) = a.get("add") {
        let terms: Vec<GlossaryTerm> = serde_json::from_value(add.clone())
            .map_err(|e| ToolError::Invalid(format!("invalid terms format in add: {e}")))?;
        for t in terms {
            glossary.add_term(t);
        }
    }
    let saved = ctx.db.set_project_glossary(p.id, &glossary)?;
    let n = saved.terms.len();
    Ok(ToolOutput::json(serde_json::to_value(&saved)?).summary(format!("{n} terms")))
}

static SET_VIDEO_ROLE: ToolSpec = ToolSpec {
    name: "set_video_role",
    effect: Effect::Write,
    summary: "Take a video out of a project's library: 'removed' (ignored everywhere) or 'reference' (a finished edit \
              the script chat studies but never cuts from). 'footage' puts it back.",
    when: "When a video should not be cut from (a duplicate, a bad take, an old export).",
    returns: "{video_id, project}.",
    misuse: &[],
    chains_with: &["set_video_role again with 'footage' to bring it back."],
    params: &[
        PROJECT,
        ToolParam::new("video_id", T::Integer, true, ""),
        ToolParam::new("role", T::Enum(&["footage", "removed", "reference"]), true, ""),
    ],
};

async fn set_video_role(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let video_id = a.i64("video_id")?;
    match a.str("role")? {
        "footage" => ctx.db.include_video(p.id, video_id)?,
        role => ctx.db.exclude_video(p.id, video_id, role)?,
    }
    Ok(ToolOutput::json(json!({ "video_id": video_id, "project": p.name })).summary(format!("video #{video_id}")))
}
