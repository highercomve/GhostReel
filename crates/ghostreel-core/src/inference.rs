//! Measured per-call inference data. Missing provider counters stay unknown.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceCall {
    pub model: String,
    pub backend: String,
    pub prompt_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Generation only, excluding prompt processing; absent when the provider omits it.
    pub generation_seconds: Option<f64>,
    pub request_seconds: f64,
}

impl InferenceCall {
    pub fn from_response(v: &Value, backend: &str, fallback_model: &str, elapsed: f64) -> Self {
        let positive = |value: f64| (value.is_finite() && value > 0.0).then_some(value);
        let output_tokens = v["gen_tokens"].as_u64().or(v["usage"]["completion_tokens"].as_u64());
        let generation_seconds = v["generation_seconds"]
            .as_f64()
            .and_then(positive)
            .or_else(|| v["timings"]["predicted_ms"].as_f64().and_then(|ms| positive(ms / 1000.0)));
        Self {
            model: v["model"].as_str().filter(|s| !s.trim().is_empty()).unwrap_or(fallback_model).to_string(),
            backend: backend.to_string(),
            prompt_tokens: v["prompt_tokens"].as_u64().or(v["usage"]["prompt_tokens"].as_u64()),
            output_tokens,
            generation_seconds,
            request_seconds: v["secs"].as_f64().and_then(positive).or_else(|| positive(elapsed)).unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatPerformance {
    pub elapsed_seconds: f64,
    /// All primary-model calls for this answer, including research and repair passes.
    pub calls: Vec<InferenceCall>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_model_and_decode_timing_are_preserved_separately_from_request_time() {
        let call = InferenceCall::from_response(
            &json!({
                "model":"actual-model", "usage":{"completion_tokens":80,"prompt_tokens":9000},
                "timings":{"predicted_ms":2000.0}
            }),
            "server",
            "alias",
            12.0,
        );
        assert_eq!(call.model, "actual-model");
        assert_eq!(call.output_tokens, Some(80));
        assert_eq!(call.generation_seconds, Some(2.0));
        assert_eq!(call.request_seconds, 12.0);
    }

    #[test]
    fn absent_or_invalid_provider_metrics_are_not_estimated_as_decode_speed() {
        let call =
            InferenceCall::from_response(&json!({"generation_seconds":-1,"secs":0}), "cli", "model not reported", 4.0);
        assert_eq!(call.output_tokens, None);
        assert_eq!(call.generation_seconds, None);
        assert_eq!(call.request_seconds, 4.0);
    }
}
