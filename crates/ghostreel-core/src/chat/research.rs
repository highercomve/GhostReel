//! Bounded research and evidence for proposed local-agent edits.
use crate::script::{Audio, Script};
use serde_json::{Value, json};

pub(super) struct ResearchBudget {
    pub tools: u32,
    pub calls: u32,
    limit: u32,
}
impl ResearchBudget {
    pub fn new(limit: u32) -> Self {
        Self { tools: 0, calls: 0, limit }
    }
    pub fn can_research(&self) -> bool {
        self.tools < self.limit && self.can_review()
    }
    pub fn can_review(&self) -> bool {
        self.calls < self.limit.saturating_mul(2).saturating_add(4)
    }
    pub fn remaining(&self) -> u32 {
        self.limit.saturating_sub(self.tools)
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

    /// Search hits and index digests are candidates, not an inspection of a selected shot.
    pub fn record(&mut self, tool: &str, args: &Value, result: &str) {
        if matches!(tool, "get_video" | "get_transcript") {
            self.attempted.insert(format!("{tool}:{args}"));
        }
        let Ok(value) = serde_json::from_str::<Value>(result) else { return };
        if value.get("error").is_some() {
            return;
        }
        let Some(video_id) = args["video_id"].as_i64() else { return };
        let (tool, start, end) = match tool {
            "get_video"
                if value["video_id"].as_i64() == Some(video_id)
                    && value["frames"].as_array().is_some_and(|frames| {
                        frames.iter().any(|f| f["summary"].as_str().is_some_and(|text| !text.trim().is_empty()))
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
                let start = segments.iter().filter_map(|v| v["start_s"].as_f64()).reduce(f64::min);
                let end = segments.iter().filter_map(|v| v["end_s"].as_f64()).reduce(f64::max);
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
        let mut b = ResearchBudget::new(4);
        b.calls += 2;
        assert_eq!(b.remaining(), 4);
        b.tools += 4;
        assert!(!b.can_research());
        assert!(b.can_review());
        b.calls = 12;
        assert!(!b.can_review());
    }
    #[test]
    fn search_and_failed_inspections_do_not_verify_proposed_clips() {
        let mut evidence = Evidence::default();
        let args = json!({"video_id":1});
        evidence.record("search_moments", &args, "[]");
        evidence.record("get_video", &args, r#"{"error":"missing file"}"#);
        evidence.record("get_video", &args, r#"{"video_id":1,"duration_s":60,"frames":[]}"#);
        assert_eq!(evidence.missing(&script(), true).len(), 2);
        evidence.record(
            "get_video",
            &args,
            r#"{"video_id":1,"duration_s":60,"frames":[{"t_s":12,"summary":"street"}]}"#,
        );
        assert_eq!(evidence.missing(&script(), true)[0].tool, "get_transcript");
    }
    #[test]
    fn transcript_coverage_uses_returned_segments_and_video_ranges_do_not_bridge_gaps() {
        let mut evidence = Evidence::default();
        let args = json!({"video_id":1,"start_s":0,"end_s":60});
        evidence.record("get_transcript", &args, r#"[{"start_s":0,"end_s":5,"text":"truncated"}]"#);
        assert_eq!(evidence.missing(&script(), true).len(), 2);
        evidence.record("get_transcript", &args, r#"[{"start_s":9,"end_s":16,"text":"actual quote"}]"#);
        for (start, end) in [(0, 9), (16, 30)] {
            evidence.record(
                "get_video",
                &json!({"video_id":1,"start_s":start,"end_s":end}),
                r#"{"video_id":1,"duration_s":60,"frames":[{"summary":"street"}]}"#,
            );
        }
        assert_eq!(evidence.missing(&script(), true).len(), 1);
        evidence.record(
            "get_video",
            &json!({"video_id":1,"start_s":10,"end_s":15}),
            r#"{"video_id":1,"duration_s":60,"frames":[{"summary":"street"}]}"#,
        );
        assert!(evidence.missing(&script(), true).is_empty());
    }
}
