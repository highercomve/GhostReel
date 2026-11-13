//! How much footage research a local turn may do.

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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planning_replies_do_not_spend_tools_but_model_retries_are_bounded() {
        let mut b = ResearchBudget::new(Some(4));
        b.calls += 2;
        assert!(b.can_research());
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
    }
}
