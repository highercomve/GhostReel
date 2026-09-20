//! What the editor is told about its four tools, written once.
//!
//! A tool description that names the tool is worth much less than one that says what must be true
//! before calling it, what the fields of the result mean, which tool comes next, and what this
//! pipeline will silently do to a draft that gets it wrong. The last is particular to GhostReel:
//! a dozen repair passes rewrite the draft after the fact, and until now the model only learned
//! about them afterwards, through `repair_note`.
//!
//! There are two renderings — the OpenAI `tools` array a server backend receives, and the prose
//! block in the system prompt that the local and CLI action loops read — and they used to be two
//! hand-written texts that had already drifted: the prose said "search results are search windows,
//! not clips" and the JSON schema did not. Both are generated from [`TOOLS`] now.

use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Integer,
    Number,
    String,
}

impl ParamType {
    fn as_str(self) -> &'static str {
        match self {
            ParamType::Integer => "integer",
            ParamType::Number => "number",
            ParamType::String => "string",
        }
    }
}

pub struct ToolParam {
    pub name: &'static str,
    pub ty: ParamType,
    pub required: bool,
    /// Units, range, default, and what happens when it is left out.
    pub doc: &'static str,
}

pub struct ToolSpec {
    pub name: &'static str,
    /// One sentence. Survives at every detail level.
    pub summary: &'static str,
    /// Where this sits in the editor's order of work.
    pub when: &'static str,
    /// Each field of the result, and what it means.
    pub returns: &'static str,
    /// What the pipeline does when it is misused, costliest first: `misuse[0]` is the line that
    /// survives at [`Detail::Short`].
    pub misuse: &'static [&'static str],
    /// How it leads to the other three.
    pub chains_with: &'static [&'static str],
    pub params: &'static [ToolParam],
}

/// How much of each spec to render. A roomy brain gets everything; a small local model, whose
/// window is already half transcripts, gets the summary, the parameters and the costliest warning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Detail {
    Full,
    Short,
}

/// The rules that apply to every tool, stated once instead of four times.
pub const TOOL_CONTRACT: &str = "Every clip you write must lie inside a range one of these tools \
    returned for that video. A range nothing returned is checked against the index and dropped if \
    nothing was ever described or spoken there — so look before you cut. Opening a video with \
    get_video or get_transcript makes the whole of that video legal to cut; a search hit makes \
    only the window it returned legal; list_videos makes nothing legal.";

const SEARCH_MOMENTS: ToolSpec = ToolSpec {
    name: "search_moments",
    summary: "Find moments in this project by meaning or keyword across speech, on-screen text and \
              what is visible.",
    when: "To locate a subject you have already decided to show. It searches the words people \
           actually said, so ask in the language of the footage.",
    returns: "Hits of {video_id, file, start_s, end_s, camera, shaky, snippet}. start_s/end_s are \
              the edges of the indexed window the match fell in — usually a whole 20-40 s chunk, \
              not a shot. camera is static, tripod, stabilised or handheld; shaky means the window \
              sits on a stretch the camera shakes through.",
    misuse: &[
        "Copying a hit's start_s/end_s into a clip gives a clip as long as the window, which is \
         then trimmed to its first seconds — rarely the shot you meant. Choose a sub-range \
         yourself and open it with get_video first.",
        "Two searches that return nothing mean those words are not in this footage. A third \
         rephrasing wastes a round; call list_videos and look at what there is.",
    ],
    chains_with: &[
        "The window a hit returned is legal to cut, so any sub-range of it is too.",
        "get_video(video_id, start_s, end_s) on the same window says whether the shot is usable.",
    ],
    params: &[
        ToolParam {
            name: "query",
            ty: ParamType::String,
            required: true,
            doc: "Words or a phrase, matched against speech, visible text and keyframe \
                  descriptions together.",
        },
        ToolParam {
            name: "limit",
            ty: ParamType::Integer,
            required: false,
            doc: "1-20, default 10. Anything outside that range is clamped.",
        },
    ],
};

const GET_VIDEO: ToolSpec = ToolSpec {
    name: "get_video",
    summary: "Look at a video: what is visible at each moment, how steady the camera is, and \
              whether anyone speaks.",
    when: "Before you use any range as a picture. It is the only way to find out whether footage \
           is worth cutting to.",
    returns: "duration, camera work, has_speech, shaky_at (the stretches worse than this clip's \
              own ordinary level, as timestamps), and keyframe descriptions with their times — \
              what is actually on screen, in order.",
    misuse: &[
        "A clip whose pictures you have not seen is a guess: if nothing was described or spoken \
         near it, the clip is dropped and your beat loses its picture.",
        "Cutting from a range listed in shaky_at puts visible camera shake in the finished piece; \
         the clip is moved to the nearest steady stretch of the same shot, or dropped when the \
         shot has none.",
    ],
    chains_with: &[
        "Opening a video this way makes the whole of it legal to cut, not only the range you \
         asked for.",
        "has_speech false means the clip cannot carry source audio: mute it, or give the beat a \
         bed from a video where someone speaks.",
    ],
    params: &[
        ToolParam {
            name: "video_id",
            ty: ParamType::Integer,
            required: true,
            doc: "From a search hit or list_videos.",
        },
        ToolParam {
            name: "start_s",
            ty: ParamType::Number,
            required: false,
            doc: "Seconds. With end_s, every keyframe in that range; without, a sample across the \
                  whole file.",
        },
        ToolParam { name: "end_s", ty: ParamType::Number, required: false, doc: "Seconds." },
    ],
};

const GET_TRANSCRIPT: ToolSpec = ToolSpec {
    name: "get_transcript",
    summary: "The words spoken in a video, as timed segments.",
    when: "Only for a video listed as left out of WHAT PEOPLE SAY — everything else is already in \
           front of you, with the same timestamps.",
    returns: "Segments of {start_s, end_s, text, off_mic}. off_mic marks someone away from the \
              microphone: in an interview that is the questions and the slate, not the answers.",
    misuse: &[
        "A clip that opens on an off-mic segment is retimed to where someone answers, which moves \
         the picture with it — choose the answer yourself.",
        "A source clip that begins or ends inside a segment is pulled out to the sentence's edge, \
         or back to the previous one. Cut on the boundaries you are given.",
    ],
    chains_with: &["A segment's start_s and end_s are the only cut points a speaking clip should use."],
    params: &[
        ToolParam {
            name: "video_id",
            ty: ParamType::Integer,
            required: true,
            doc: "From a search hit or list_videos.",
        },
        ToolParam { name: "start_s", ty: ParamType::Number, required: false, doc: "Seconds; omit for the whole file." },
        ToolParam { name: "end_s", ty: ParamType::Number, required: false, doc: "Seconds." },
    ],
};

const LIST_VIDEOS: ToolSpec = ToolSpec {
    name: "list_videos",
    summary: "Every video in the project, with a short summary of each.",
    when: "To see the shape of the material, or when searching has stopped finding anything.",
    returns: "For each video: id, file, duration, camera work and whether anyone speaks in it.",
    misuse: &["Listing is not looking: an id seen only here grounds nothing, and a clip written from it \
         survives only if something happens to be indexed where you guessed. Open it first."],
    chains_with: &["get_video(video_id) on anything that looks promising."],
    params: &[],
};

pub const TOOLS: &[&ToolSpec] = &[&SEARCH_MOMENTS, &GET_VIDEO, &GET_TRANSCRIPT, &LIST_VIDEOS];

/// The names the action schema will accept, in the order they are documented.
pub fn tool_names() -> Vec<&'static str> {
    TOOLS.iter().map(|t| t.name).collect()
}

fn describe(spec: &ToolSpec, detail: Detail) -> String {
    let mut out = String::from(spec.summary);
    if detail == Detail::Full {
        out.push(' ');
        out.push_str(spec.when);
        out.push_str(" Returns: ");
        out.push_str(spec.returns);
        for line in spec.chains_with {
            out.push(' ');
            out.push_str(line);
        }
    }
    // The costliest warning is worth its tokens even in the short rendering.
    let warnings: &[&str] = if detail == Detail::Full { spec.misuse } else { &spec.misuse[..spec.misuse.len().min(1)] };
    for line in warnings {
        out.push(' ');
        out.push_str(line);
    }
    out
}

/// The OpenAI `tools` array a server backend is sent.
pub fn render_openai(detail: Detail) -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|spec| {
            let mut props = serde_json::Map::new();
            let mut required = Vec::new();
            for p in spec.params {
                props.insert(p.name.to_string(), json!({ "type": p.ty.as_str(), "description": p.doc }));
                if p.required {
                    required.push(json!(p.name));
                }
            }
            json!({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": describe(spec, detail),
                    "parameters": {
                        "type": "object",
                        "properties": Value::Object(props),
                        "required": required,
                        "additionalProperties": false
                    }
                }
            })
        })
        .collect();
    Value::Array(tools)
}

/// The TOOLS block of the system prompt, for the loops that read prose rather than a schema.
pub fn tools_prose(detail: Detail) -> String {
    let mut out = String::from("\n\nTOOLS\n");
    out.push_str(TOOL_CONTRACT);
    out.push('\n');
    for spec in TOOLS {
        let args: Vec<String> =
            spec.params.iter().map(|p| if p.required { p.name.to_string() } else { format!("{}?", p.name) }).collect();
        out.push_str(&format!("- {}({}): {}\n", spec.name, args.join(", "), describe(spec, detail)));
        if detail == Detail::Full {
            for p in spec.params {
                out.push_str(&format!("    {} — {}\n", p.name, p.doc));
            }
        }
    }
    out
}

/// Backwards-compatible entry point: the full rendering, as the server backend has always had.
pub fn tools_definition() -> Value {
    render_openai(Detail::Full)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both renderings come from one catalog, so a fact stated to one backend is stated to all.
    #[test]
    fn every_tool_is_described_to_every_backend() {
        let prose = tools_prose(Detail::Full);
        let schema = render_openai(Detail::Full);
        let arr = schema.as_array().unwrap();
        assert_eq!(arr.len(), TOOLS.len());
        for spec in TOOLS {
            assert!(prose.contains(spec.name), "{} missing from the prose", spec.name);
            let found = arr.iter().find(|t| t["function"]["name"] == spec.name).expect("in the schema");
            let described = found["function"]["description"].as_str().unwrap();
            assert!(described.contains(spec.summary), "{} summary missing from the schema", spec.name);
            // The warnings are the point: a description that omits them is the thin one we replaced.
            assert!(described.contains(spec.misuse[0]), "{} loses its warning", spec.name);
        }
    }

    /// The facts the pipeline acts on silently. Each sentence here has a test of the behaviour it
    /// describes elsewhere in this crate; if one changes, this catches the other.
    #[test]
    fn the_descriptions_state_what_the_pipeline_will_do() {
        let prose = tools_prose(Detail::Full);
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

    /// The schema is re-sent every round, so its size is a running cost, not a one-off. About
    /// 1200 tokens for all four tools is the deal being struck: against the ~11500 the project's
    /// transcripts already take, and a measured three-fold difference in task success between
    /// thin and rich tool descriptions, it is cheap. This is a ceiling, not a target — a fifth
    /// tool or a wordier spec should have to argue for itself here.
    #[test]
    fn the_descriptions_stay_within_their_budget() {
        let full = serde_json::to_string(&render_openai(Detail::Full)).unwrap().len();
        let short = serde_json::to_string(&render_openai(Detail::Short)).unwrap().len();
        assert!(full < 5000, "full tool schema is {full} chars (~{} tokens)", full / 4);
        // Worth having at all: the short rendering must save enough to matter on a 16k window.
        assert!(short * 5 < full * 3, "short rendering saves little: {short} vs {full}");
        assert!(tools_prose(Detail::Short).len() < tools_prose(Detail::Full).len());
    }

    #[test]
    fn the_action_schema_and_the_catalog_name_the_same_tools() {
        assert_eq!(tool_names(), vec!["search_moments", "get_video", "get_transcript", "list_videos"]);
    }
}
