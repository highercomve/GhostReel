//! The machine's settings, its agents and the trash.

use serde_json::{Value, json};

use super::spec::{Effect, PROJECT, ParamType as T, ToolParam, ToolSpec};
use super::{Args, Tool, ToolCtx, ToolError, ToolOutput, ToolResult, tool};
use crate::config::{CLI_TOOLS, Config, CliAgentConfig, VisionConfig};

pub static TOOLS: [Tool; 5] = [
    tool!(GET_SETTINGS, get_settings),
    tool!(SET_SETTINGS, set_settings),
    tool!(LIST_AGENTS, list_agents),
    tool!(LIST_TRASH, list_trash),
    tool!(RESTORE_TRASH, restore_trash),
];

/// The AI settings as the tools report them: one entry per capability, plus which coding-agent
/// CLIs this machine can actually run.
pub fn settings_view(cfg: &Config) -> Value {
    let llm = |c: &VisionConfig| {
        json!({
            "backend": c.backend.to_string(),
            "url": c.url,
            "model": c.model,
            "local_model": c.local_model,
            "ctx_tokens": c.ctx_tokens,
            "kv_cache": c.kv_cache,
            "flash_attn": c.flash_attn,
            "think": c.think,
            "max_tool_rounds": c.max_tool_rounds,
            "cli": {
                "tool": c.cli.tool,
                "command": c.cli.command,
                "model": c.cli.model,
                "concurrency": c.cli.concurrency,
                "timeout_secs": c.cli.timeout_secs,
                "installed": crate::cliagent::CliAgent::new(c.cli.clone()).available().is_some(),
            },
        })
    };
    let agents: Vec<Value> = CLI_TOOLS
        .iter()
        .map(|tool| {
            let probe = CliAgentConfig { tool: (*tool).to_string(), ..CliAgentConfig::default() };
            json!({
                "tool": tool,
                "path": crate::cliagent::find_binary(&probe).map(|p| p.display().to_string()),
            })
        })
        .collect();
    json!({
        "vision": llm(&cfg.vision),
        "chat_model": llm(&cfg.chat_model()),
        "stt": { "backend": cfg.stt.backend.to_string(), "url": cfg.stt.url, "model": cfg.stt.model },
        "embed": { "backend": cfg.embed.backend.to_string(), "url": cfg.embed.url, "model": cfg.embed.model },
        "frames": { "max_interval_s": cfg.frames.max_interval_s },
        "cli_agents": agents,
    })
}

static GET_SETTINGS: ToolSpec = ToolSpec {
    name: "get_settings",
    effect: Effect::Read,
    summary: "The AI settings: for speech, frame descriptions, the script chat and embeddings — which backend each uses \
              (auto/local/server/cli), its server URL and model, the local context window, KV cache, flash attention, \
              whether the model reasons before answering and the optional cap on tool calls (0 = no cap), plus which \
              coding-agent CLIs are installed.",
    when: "Before changing a setting, or to see why a model is not being used.",
    returns: "{vision, chat_model, stt, embed, frames, cli_agents}.",
    misuse: &[],
    chains_with: &["set_settings to change one; list_agents for the agent pool."],
    params: &[],
};

async fn get_settings(ctx: &mut ToolCtx<'_>, _a: Args<'_>) -> ToolResult {
    let cfg = Config::load(&ctx.paths.config_file).unwrap_or_default();
    Ok(ToolOutput::json(settings_view(&cfg)).summary("settings"))
}

static SET_SETTINGS: ToolSpec = ToolSpec {
    name: "set_settings",
    effect: Effect::Write,
    summary: "Change AI settings, the same keys `ghostreel config set` takes, e.g. {\"chat_model.backend\": \"local\", \
              \"chat_model.think\": \"true\", \"chat_model.ctx_tokens\": \"16384\"}. Sections: vision (frame \
              descriptions), chat_model (script chat), stt, embed.",
    when: "Only when the user asked for a setting to change. chat_model.max_tool_rounds is an optional cap on a run's \
           tool calls (0 = no cap, the default).",
    returns: "{applied, settings}: the settings after the change.",
    misuse: &["A setting is saved to the config file and applies to every later run, not only this one."],
    chains_with: &["get_settings first, to see the current values."],
    params: &[ToolParam::new("set", T::Object, true, "{\"<section>.<field>\": \"<value>\"} — values as strings.")],
};

async fn set_settings(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let mut cfg = Config::load(&ctx.paths.config_file).unwrap_or_default();
    let set = a
        .get("set")
        .and_then(|v| v.as_object())
        .ok_or_else(|| ToolError::invalid("set must be an object of \"section.field\": \"value\""))?;
    let mut applied = Vec::new();
    for (key, value) in set {
        // Numbers and booleans are as natural to write here as strings; take either.
        let value = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        cfg.set_key(key, &value).map_err(|e| ToolError::Invalid(e.to_string()))?;
        applied.push(format!("{key} = {value}"));
    }
    cfg.save(&ctx.paths.config_file)?;
    let n = applied.len();
    Ok(ToolOutput::json(json!({ "applied": applied, "settings": settings_view(&cfg) })).summary(format!("{n} settings changed")))
}

static LIST_AGENTS: ToolSpec = ToolSpec {
    name: "list_agents",
    effect: Effect::Read,
    summary: "The agent pool: each agent's id, backend (local/server/cli), tool, model, how many calls it takes at once \
              (a local model always one), roles (orchestrator, worker, chat, vision), enabled and priority, plus the \
              app-wide selection and what it resolves to.",
    when: "Before generate_clips or save_script with `agents`, to see the ids. With check=true also whether each is \
           available now (installed / reachable / model listed).",
    returns: "{implicit, agents, app_selection, resolved, selection_error}.",
    misuse: &[],
    chains_with: &["Agents are edited with set_settings keys agents.<id>.<field> and agent_selection.<field>."],
    params: &[ToolParam::new("check", T::Boolean, false, "default false: also check availability (no model call).")],
};

async fn list_agents(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let config = Config::load(&ctx.paths.config_file).unwrap_or_default();
    let agents = config.agents();
    let check = a.bool_or("check", false);
    let mut views = Vec::new();
    for ag in &agents {
        let mut v = serde_json::to_value(ag.effective(&config))?;
        v["capacity"] = json!(ag.capacity());
        if !ag.api_key.is_empty() {
            v["api_key"] = json!("(set)");
        }
        if check {
            v["availability"] = serde_json::to_value(crate::agents::check_availability(ctx.paths, &config, ag).await)?;
        }
        views.push(v);
    }
    let resolved = config.resolve_agents(None, None);
    let n = views.len();
    Ok(ToolOutput::json(json!({
        "implicit": config.agents.is_empty(),
        "agents": views,
        "app_selection": config.agent_selection,
        "resolved": resolved.as_ref().ok().map(|r| json!({
            "orchestrator": r.orchestrator.as_ref().map(|a| &a.id),
            "workers": r.workers.iter().map(|a| &a.id).collect::<Vec<_>>(),
            "chat": r.chat.as_ref().map(|a| &a.id),
            "max_parallel": r.max_parallel,
            "capacity_line": r.capacity_line(),
        })),
        "selection_error": resolved.err().map(|e| e.to_string()),
    }))
    .summary(format!("{n} agents")))
}

static LIST_TRASH: ToolSpec = ToolSpec {
    name: "list_trash",
    effect: Effect::Read,
    summary: "Deleted cuts, script versions and clip sets of a project (newest first).",
    when: "To find something that was deleted. Deleting in the app moves it to the trash for 30 days.",
    returns: "Each entry has an id, a label, what it holds and when it is purged.",
    misuse: &[],
    chains_with: &["restore_trash(id) puts one back."],
    params: &[PROJECT],
};

async fn list_trash(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let p = ctx.project_for(a)?;
    let items = crate::trash::list(ctx.db, Some(p.id))?;
    let n = items.len();
    Ok(ToolOutput::json(serde_json::to_value(items)?).summary(format!("{n} in the trash")))
}

static RESTORE_TRASH: ToolSpec = ToolSpec {
    name: "restore_trash",
    effect: Effect::Write,
    summary: "Put a trash entry back exactly as it was (every version of a cut, or a clip set with its clips).",
    when: "After list_trash, on the id of the entry to bring back.",
    returns: "{restored, item}.",
    misuse: &[],
    chains_with: &["list_scripts / list_clip_sets to see it again."],
    params: &[ToolParam::new("id", T::String, true, "From list_trash.")],
};

async fn restore_trash(ctx: &mut ToolCtx<'_>, a: Args<'_>) -> ToolResult {
    let item = crate::trash::restore(ctx.db, a.str("id")?, ctx.origin.source())?;
    let summary = format!("restored {}", item.label);
    Ok(ToolOutput::json(json!({ "restored": item.label, "item": item })).summary(summary))
}
