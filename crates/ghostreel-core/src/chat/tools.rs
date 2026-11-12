//! What the editor is told about its tools.
//!
//! The tools themselves are not defined here any more: the script chat uses the whole registry
//! ([`crate::tools`]), the same set an MCP client and a clip worker get. This module is the chat's
//! view of it: the renderings for a project-scoped reader (`project` optional, because the chat
//! already works on one) and the claims the chat's prompts make about the tools, tested against
//! the descriptions the registry gives.

use serde_json::Value;

pub use crate::tools::{Detail, RenderOpts, TOOL_CONTRACT, render_openai, tools_prose};

/// The names the action schema will accept, in the order they are documented.
pub fn tool_names() -> Vec<&'static str> {
    crate::tools::names()
}

/// The full OpenAI `tools` array for a reader that works on one project.
pub fn tools_definition() -> Value {
    render_openai(RenderOpts::new(Detail::Full, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT: fn(Detail) -> RenderOpts = |detail| RenderOpts::new(detail, true);

    /// Both renderings come from one catalog, so a fact stated to one backend is stated to all.
    #[test]
    fn every_tool_is_described_to_every_backend() {
        let prose = tools_prose(CHAT(Detail::Full));
        let schema = render_openai(CHAT(Detail::Full));
        let arr = schema.as_array().unwrap();
        assert_eq!(arr.len(), crate::tools::all().count());
        for t in crate::tools::all() {
            let spec = t.spec;
            assert!(prose.contains(spec.name), "{} missing from the prose", spec.name);
            let found = arr.iter().find(|t| t["function"]["name"] == spec.name).expect("in the schema");
            let described = found["function"]["description"].as_str().unwrap();
            assert!(described.contains(spec.summary), "{} summary missing from the schema", spec.name);
            // The warnings are the point: a description that omits them is the thin one we replaced.
            if let Some(first) = spec.misuse.first() {
                assert!(described.contains(first), "{} loses its warning", spec.name);
            }
        }
    }

    /// The facts the pipeline acts on silently. Each sentence here has a test of the behaviour it
    /// describes elsewhere in this crate; if one changes, this catches the other.
    #[test]
    fn the_descriptions_state_what_the_pipeline_will_do() {
        let prose = tools_prose(CHAT(Detail::Full));
        for claim in [
            "list_videos makes nothing legal",     // Grounding::record_tool_call
            "trimmed to its first seconds",        // trimmed_clip_s
            "moved to the nearest steady stretch", // shaky repair
            "retimed to where someone answers",    // off-mic repair
            "pulled out to the sentence's edge",   // end_on_sentences
            "cannot carry source audio",           // mute_silent_clips
        ] {
            assert!(prose.contains(claim), "the tools no longer say: {claim}");
        }
    }

    /// The schema is re-sent every round, so its size is a running cost, not a one-off. Forty-one
    /// tools at full detail are about 10k tokens (a server model is also given their short prose, about
    /// 3k); against the ~11500 the project's transcripts already take, and a measured three-fold
    /// difference in task success between thin and rich tool descriptions, that is the deal. This
    /// is a ceiling, not a target: a wordier spec should have to argue for itself here. The short
    /// rendering is what a small local model gets, and it must stay small enough for a 16k window.
    #[test]
    fn the_descriptions_stay_within_their_budget() {
        let full = serde_json::to_string(&render_openai(CHAT(Detail::Full))).unwrap().len();
        let short = serde_json::to_string(&render_openai(CHAT(Detail::Short))).unwrap().len();
        assert!(full < 46_000, "full tool schema is {full} chars (~{} tokens)", full / 4);
        assert!(short < 20_000, "short tool schema is {short} chars (~{} tokens)", short / 4);
        assert!(short * 2 < full, "short rendering saves little: {short} vs {full}");
        let prose_short = tools_prose(CHAT(Detail::Short)).len();
        assert!(prose_short < 13_000, "short prose is {prose_short} chars (~{} tokens)", prose_short / 4);
        assert!(prose_short * 2 < tools_prose(CHAT(Detail::Full)).len());
        let grammar = serde_json::to_string(&crate::tools::action_branches(CHAT(Detail::Full))).unwrap().len();
        assert!(grammar < 18_000, "the action grammar's tool branches are {grammar} chars");
    }

    #[test]
    fn the_action_schema_and_the_catalog_name_the_same_tools() {
        let schema = crate::chat::local_action_schema();
        let named: Vec<&str> = schema["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["properties"]["action"]["enum"][0] == "tool")
            .map(|b| b["properties"]["tool"]["enum"][0].as_str().unwrap())
            .collect();
        assert_eq!(named, tool_names());
    }
}
