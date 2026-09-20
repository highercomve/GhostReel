//! Jev — a System One model, asked typed questions about a cut.
//!
//! Everything else in GhostReel judges a script by counting: seconds over target, cuts landing
//! inside a sentence, pictures left hanging in silence. Those are the faults a pass can fix, and
//! `chat::metrics` drives each of them to zero. What no count reaches is whether the piece is any
//! *good* — whether the shot on screen shows what the voice is talking about, whether the opening
//! earns the next ten seconds, whether the ending lands. Every verdict of that kind so far came
//! from the editor watching a preview, which is the one measurement that does not scale.
//!
//! Jev answers questions like that as numbers rather than prose: a probability, a level on a
//! rubric. It does not write, and it is not a second brain — the script still comes from the
//! editor model. This is a judge, and an optional one: without an API key nothing here runs and
//! nothing changes.
//!
//! <https://docs.typesafe.ai/api> — one POST, many questions, answered against the state in
//! parallel. Asking twelve questions in one request rather than twelve requests is the whole
//! shape of the API, and the reason a judgement costs a fraction of a cent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::config::JevConfig;

/// A question to ask about the state. The three System One primitives, and nothing else.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes or no, answered as the probability of yes.
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<Value>,
    },
    /// One option from a set, answered with the full distribution.
    Choice { instructions: Value, criteria: BTreeMap<String, Value> },
    /// A position on ordered levels, answered as a probability-weighted value between them.
    Score { instructions: Value, criteria: Vec<Value> },
}

impl Question {
    pub fn noul(instructions: impl Into<Value>) -> Self {
        Self::Noul { instructions: instructions.into(), criteria: None }
    }

    /// A yes/no with both ends spelled out. Worth the two extra lines: "the pictures match" means
    /// nothing until the answer says what a no looks like.
    pub fn noul_between(instructions: impl Into<Value>, yes: &str, no: &str) -> Self {
        Self::Noul {
            instructions: instructions.into(),
            criteria: Some(serde_json::json!({ "true": yes, "false": no })),
        }
    }

    pub fn score(instructions: impl Into<Value>, levels: &[&str]) -> Self {
        Self::Score {
            instructions: instructions.into(),
            criteria: levels.iter().map(|l| Value::from(*l)).collect(),
        }
    }
}

/// One answer, under the id its question was asked with.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
    Score {
        score: f64,
        #[serde(default)]
        legend: BTreeMap<String, String>,
        #[serde(default)]
        confidence: f64,
    },
}

impl Answer {
    /// The answer as a 0–1 number, whatever its type: a noul's probability, a score's position
    /// along its own levels. Choices have no natural ordering, so they have no such reading.
    pub fn as_unit(&self) -> Option<f64> {
        match self {
            Self::Noul { noul } => Some(*noul),
            Self::Score { score, legend, .. } => {
                let top = (legend.len().max(2) - 1) as f64;
                Some((score / top).clamp(0.0, 1.0))
            }
            Self::Choice { .. } => None,
        }
    }

    /// How concentrated the distribution behind the answer is. A noul has none — near 0.5 it is
    /// saying yes and no are equally likely, which is an answer rather than an absence of one.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => Some(*confidence),
            Self::Noul { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Answers {
    #[serde(default)]
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl Answers {
    pub fn unit(&self, id: &str) -> Option<f64> {
        self.answers.get(id).and_then(Answer::as_unit)
    }
}

#[derive(Serialize)]
struct Request<'a> {
    state: &'a Value,
    model: &'a str,
    questions: &'a BTreeMap<String, Question>,
}

/// A configured client. Built only when there is a key to build it with.
pub struct Jev {
    key: String,
    base_url: String,
    model: String,
    timeout: std::time::Duration,
}

impl Jev {
    /// The client the config asks for, or `None` when Jev is off or unkeyed.
    ///
    /// The key comes from the environment before the config file, because a config file is
    /// committed by accident far more often than an environment is.
    pub fn from_config(cfg: &JevConfig) -> Option<Self> {
        if !cfg.enabled {
            return None;
        }
        let key = std::env::var("TYPESAFE_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| Some(cfg.api_key.clone()).filter(|k| !k.trim().is_empty()))?;
        let base_url = std::env::var("TYPESAFE_BASE_URL").unwrap_or_else(|_| cfg.base_url.clone());
        Some(Self {
            key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            timeout: std::time::Duration::from_secs(cfg.timeout_s.max(1)),
        })
    }

    /// Ask every question about one state, in a single request.
    ///
    /// 429 and 529 are the two the docs say to back off on rather than give up on; a judge that
    /// dies on a rate limit would make the eval flaky for no reason.
    pub async fn ask(&self, state: &Value, questions: &BTreeMap<String, Question>) -> Result<Answers, Error> {
        if questions.is_empty() {
            return Err(Error::Jev("no questions to ask".into()));
        }
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| Error::Jev(e.to_string()))?;
        let url = format!("{}/v1/systemone", self.base_url);
        let body = Request { state, model: &self.model, questions };

        let mut backoff = std::time::Duration::from_millis(500);
        let mut last = String::new();
        for attempt in 0..4 {
            if attempt > 0 {
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }
            let resp = match client.post(&url).bearer_auth(&self.key).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    last = e.to_string();
                    continue;
                }
            };
            let status = resp.status();
            if status.is_success() {
                return resp.json::<Answers>().await.map_err(|e| Error::Jev(format!("unreadable answer: {e}")));
            }
            let detail = resp.text().await.unwrap_or_default();
            let detail = detail.chars().take(400).collect::<String>();
            last = format!("{status}: {detail}");
            // Anything else is our fault — a bad key, a malformed question — and retrying it just
            // spends the same mistake four times.
            if status.as_u16() != 429 && status.as_u16() != 529 {
                break;
            }
        }
        Err(Error::Jev(last))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_serialise_the_way_the_api_spells_them() {
        let mut qs = BTreeMap::new();
        qs.insert("urgent".to_string(), Question::noul_between("Is this urgent?", "time-critical", "no hurry"));
        qs.insert("heat".to_string(), Question::score("How hot?", &["cold", "warm", "hot"]));
        let json = serde_json::to_value(&qs).unwrap();
        assert_eq!(json["urgent"]["type"], "noul");
        assert_eq!(json["urgent"]["instructions"], "Is this urgent?");
        assert_eq!(json["urgent"]["criteria"]["true"], "time-critical");
        assert_eq!(json["heat"]["type"], "score");
        assert_eq!(json["heat"]["criteria"][2], "hot");
    }

    #[test]
    fn answers_read_back_as_numbers() {
        let raw = r#"{
            "model": "jev-1.13.0",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.95},
                "heat": {"type": "score", "score": 1.05,
                         "legend": {"0": "cold", "1": "warm", "2": "hot"},
                         "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92},
                "team": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.88, "sales": 0.12}, "confidence": 0.81}
            },
            "usage": {"input_tokens": 296, "output_tokens": 20}
        }"#;
        let a: Answers = serde_json::from_str(raw).unwrap();
        assert_eq!(a.unit("urgent"), Some(0.95));
        // Halfway up a three-level rubric, read back on 0–1.
        assert!((a.unit("heat").unwrap() - 0.525).abs() < 1e-6);
        assert_eq!(a.unit("team"), None);
        assert_eq!(a.answers["team"].confidence(), Some(0.81));
        assert_eq!(a.answers["urgent"].confidence(), None);
        assert_eq!(a.usage.unwrap().input_tokens, 296);
    }

    #[test]
    fn a_client_needs_both_a_switch_and_a_key() {
        let mut cfg = JevConfig { enabled: false, api_key: "k".into(), ..Default::default() };
        assert!(Jev::from_config(&cfg).is_none(), "off means off, key or no key");
        cfg.enabled = true;
        cfg.api_key = String::new();
        // The environment may legitimately carry one on this machine; only assert the config path.
        if std::env::var("TYPESAFE_API_KEY").is_err() {
            assert!(Jev::from_config(&cfg).is_none(), "on with no key anywhere is still off");
        }
        cfg.api_key = "k".into();
        assert!(Jev::from_config(&cfg).is_some());
    }
}
