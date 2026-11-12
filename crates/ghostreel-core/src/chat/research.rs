//! Bounded research and evidence for proposed local-agent edits.
use crate::script::{Audio, Script};
use serde_json::{Value, json};

/// How much footage research a turn may do. With no cap (the default) it never runs out: the turn
/// ends when the model drafts, is stopped, or is told to by the repeat guard.
pub(super) struct ResearchBudget {
    pub tools: u32,
    pub calls: u32,
    limit: Option<u32>,
}
impl ResearchBudget {
    pub fn new(limit: Option<u32>) -> Self {
        Self { tools: 0, calls: 0, limit }
    }
    pub fn can_research(&self) -> bool {
        self.limit.is_none_or(|l| self.tools < l) && self.can_review()
    }
    /// Model calls are bounded by twice the cap plus a few retries; without a cap they are not.
    pub fn can_review(&self) -> bool {
        self.limit.is_none_or(|l| self.calls < l.saturating_mul(2).saturating_add(4))
    }
    pub fn remaining(&self) -> u32 {
        self.limit.map_or(u32::MAX, |l| l.saturating_sub(self.tools))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Inspection {
    pub tool: &'static str,
    pub video_id: i64,
    pub start: f64,
    pub end: f64,
}
impl Inspection {
    pub fn args(&self) -> Value {
        json!({"video_id":self.video_id,"start_s":(self.start-2.0).max(0.0),"end_s":self.end+2.0})
    }
}

#[derive(Default)]
pub(super) struct Evidence {
    ranges: Vec<Inspection>,
    attempted: std::collections::HashSet<String>,
}
impl Evidence {
    pub fn attempted(&self, inspection: &Inspection) -> bool {
        self.attempted.contains(&format!("{}:{}", inspection.tool, inspection.args()))
    }

    /// Search hits and index digests are candidates, not an inspection of a selected shot. `value`
    /// is the structured result of a call that ran (a failed call has none).
    pub fn record(&mut self, tool: &str, args: &Value, value: &Value) {
        if matches!(tool, "get_video" | "get_transcript") {
            self.attempted.insert(format!("{tool}:{args}"));
        }
        let Some(video_id) = args["video_id"].as_i64() else { return };
        let (tool, start, end) = match tool {
            "get_video"
                if value["video_id"].as_i64() == Some(video_id)
                    && value["keyframes"].as_array().is_some_and(|frames| {
                        frames.iter().any(|f| f["description"].as_str().is_some_and(|text| !text.trim().is_empty()))
                    }) =>
            {
                (
                    "get_video",
                    args["start_s"].as_f64().unwrap_or(0.0).max(0.0),
                    args["end_s"].as_f64().unwrap_or(f64::MAX).min(value["duration_s"].as_f64().unwrap_or(0.0)),
                )
            }
            "get_transcript" => {
                let Some(segments) = value.as_array() else { return };
                // A truncated result proves only the returned segments, not the whole request.
                let start = segments.iter().filter_map(|v| v["start"].as_f64()).reduce(f64::min);
                let end = segments.iter().filter_map(|v| v["end"].as_f64()).reduce(f64::max);
                let (Some(start), Some(end)) = (start, end) else { return };
                ("get_transcript", start, end)
            }
            _ => return,
        };
        if start.is_finite() && end.is_finite() && end > start {
            self.ranges.push(Inspection { tool, video_id, start, end });
        }
    }
    pub fn missing(&self, script: &Script, check_speech: bool) -> Vec<Inspection> {
        let mut needed = Vec::new();
        for beat in &script.beats {
            for clip in &beat.clips {
                needed.push(Inspection {
                    tool: "get_video",
                    video_id: clip.video_id,
                    start: clip.in_s,
                    end: clip.out_s,
                });
                if check_speech && clip.audio == Audio::Source {
                    needed.push(Inspection {
                        tool: "get_transcript",
                        video_id: clip.video_id,
                        start: clip.in_s,
                        end: clip.out_s,
                    });
                }
            }
            if let Some(bed) = &beat.bed
                && check_speech
            {
                needed.push(Inspection {
                    tool: "get_transcript",
                    video_id: bed.video_id,
                    start: bed.in_s,
                    end: bed.out_s,
                });
            }
        }
        needed.retain(|n| {
            !self.ranges.iter().any(|r| {
                r.tool == n.tool && r.video_id == n.video_id && n.start >= r.start - 0.05 && n.end <= r.end + 0.05
            })
        });
        needed.sort_by(|a, b| {
            a.tool
                .cmp(b.tool)
                .then(a.video_id.cmp(&b.video_id))
                .then(a.start.total_cmp(&b.start))
                .then(a.end.total_cmp(&b.end))
        });
        needed.dedup();
        needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn script() -> Script {
        serde_json::from_value(json!({"title":"test","fps":{"num":30,"den":1},"width":1920,"height":1080,
            "beats":[{"id":"b1","purpose":"test","clips":[{"video_id":1,"in_s":10.0,"out_s":15.0,"audio":"source"}]}]}))
        .unwrap()
    }
    #[test]
    fn planning_replies_do_not_spend_tools_but_model_retries_are_bounded() {
        let mut b = ResearchBudget::new(Some(4));
        b.calls += 2;
        assert_eq!(b.remaining(), 4);
        b.tools += 4;
        assert!(!b.can_research());
        assert!(b.can_review());
        b.calls = 12;
        assert!(!b.can_review());
    }
    #[test]
    fn no_cap_means_research_never_runs_out() {
        let mut b = ResearchBudget::new(None);
        b.tools = 10_000;
        b.calls = 20_000;
        assert!(b.can_research());
        assert!(b.can_review());
        assert_eq!(b.remaining(), u32::MAX);
    }
    #[test]
    fn search_and_failed_inspections_do_not_verify_proposed_clips() {
        let mut evidence = Evidence::default();
        let args = json!({"video_id":1});
        evidence.record("search", &args, &json!({"mode":"keywords","hits":[]}));
        // A failed call has no value to record, so only an empty inspection is left to refuse.
        evidence.record("get_video", &args, &json!({"video_id":1,"duration_s":60,"keyframes":[]}));
        assert_eq!(evidence.missing(&script(), true).len(), 2);
        evidence.record(
            "get_video",
            &args,
            &json!({"video_id":1,"duration_s":60,"keyframes":[{"t_s":12,"description":"street"}]}),
        );
        assert_eq!(evidence.missing(&script(), true)[0].tool, "get_transcript");
    }
    #[test]
    fn transcript_coverage_uses_returned_segments_and_video_ranges_do_not_bridge_gaps() {
        let mut evidence = Evidence::default();
        let args = json!({"video_id":1,"start_s":0,"end_s":60});
        evidence.record("get_transcript", &args, &json!([{"start":0,"end":5,"text":"truncated"}]));
        assert_eq!(evidence.missing(&script(), true).len(), 2);
        evidence.record("get_transcript", &args, &json!([{"start":9,"end":16,"text":"actual quote"}]));
        for (start, end) in [(0, 9), (16, 30)] {
            evidence.record(
                "get_video",
                &json!({"video_id":1,"start_s":start,"end_s":end}),
                &json!({"video_id":1,"duration_s":60,"keyframes":[{"description":"street"}]}),
            );
        }
        assert_eq!(evidence.missing(&script(), true).len(), 1);
        evidence.record(
            "get_video",
            &json!({"video_id":1,"start_s":10,"end_s":15}),
            &json!({"video_id":1,"duration_s":60,"keyframes":[{"description":"street"}]}),
        );
        assert!(evidence.missing(&script(), true).is_empty());
    }
}
