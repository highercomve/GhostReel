//! Shared speech-opening vocabulary. The sets serve different editorial decisions: replies and
//! connectives warn/rank openings, fillers may be trimmed on a separate breath, and continuations
//! keep dependent clauses together. Keep those distinctions when changing a word's membership.

/// Replies that answer a preceding line the viewer may not hear.
pub const REPLIES: &[&str] = &[
    "exacto",
    "exactamente",
    "claro",
    "sí",
    "totalmente",
    "correcto",
    "efectivamente",
    "obvio",
    "obviamente",
    "exactly",
    "right",
    "yeah",
    "yes",
    "sure",
    "absolutely",
    "totally",
    "true",
];

/// Openers that lean on the preceding thought.
pub const CONNECTIVES: &[&str] = &[
    "pero", "entonces", "porque", "además", "también", "aunque", "sino", "pues", "and", "but", "so", "because", "also",
    "which", "although",
];

/// Reply/filler words eligible for breath-separated opening trims.
pub const FILLERS: &[&str] = &[
    "yeah",
    "yes",
    "yep",
    "right",
    "exactly",
    "sure",
    "absolutely",
    "totally",
    "okay",
    "ok",
    "so",
    "well",
    "um",
    "uh",
    "oh",
    "and",
    "but",
    "like",
    "exacto",
    "exactamente",
    "claro",
    "sí",
    "si",
    "bueno",
    "pues",
    "entonces",
    "vale",
    "y",
];

/// Prepositions and relatives that need the preceding clause.
pub const CONTINUATIONS: &[&str] = &[
    "en", "de", "del", "con", "para", "al", "a", "que", "cual", "cuales", "donde", "cuyo", "cuya", "lo", "of", "in",
    "with", "for", "to", "than", "which", "whose", "whom",
];

/// Whether a thought begins with a dependent clause. `lo` continues only as `lo cual`.
pub(crate) fn opens_as_continuation(text: &str) -> bool {
    let mut words = text.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase());
    match words.next().as_deref() {
        Some("lo") => words.next().as_deref() == Some("cual"),
        Some(w) => CONTINUATIONS.contains(&w),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_categories_keep_their_distinct_editorial_meaning() {
        for word in ["exacto", "sí", "yeah", "true"] {
            assert!(REPLIES.contains(&word));
            assert!(!CONNECTIVES.contains(&word));
        }
        for word in ["porque", "aunque", "because", "which"] {
            assert!(CONNECTIVES.contains(&word));
        }
        // Breath-separated fillers are deliberately neither a union nor a subset of replies.
        assert!(FILLERS.contains(&"um"));
        assert!(FILLERS.contains(&"yep"));
        assert!(!REPLIES.contains(&"yep"));
        assert!(REPLIES.contains(&"obviamente"));
        assert!(!FILLERS.contains(&"obviamente"));
        assert!(opens_as_continuation("¿Lo cual nos cuesta diez dólares?"));
        assert!(opens_as_continuation("WITH the original adapter"));
        assert!(opens_as_continuation("which we already have"));
        assert!(!opens_as_continuation("lo que pasa es"));
        assert!(!opens_as_continuation("because we can"));
        assert!(!opens_as_continuation(""));
    }
}
