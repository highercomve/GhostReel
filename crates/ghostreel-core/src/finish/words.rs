//! Words that say what a line or a picture is about: the overlap a cut uses to pair a line with a
//! shot. Shared by the finishing chain (the closing picture) and the chat's cut builder.

/// Words that appear in every sentence and so distinguish nothing.
const STOPWORDS: &[&str] = &[
    "the", "and", "that", "this", "with", "you", "your", "for", "are", "was", "were", "have", "has", "had", "but",
    "not", "they", "them", "their", "there", "here", "what", "when", "where", "who", "how", "just", "like", "really",
    "kind", "sort", "very", "much", "more", "most", "some", "any", "all", "our", "out", "about", "from", "into",
    "than", "then", "been", "would", "could", "should", "will", "can", "get", "got", "know", "think", "going", "one",
    "its", "it's", "i'm", "we're", "that's", "yeah", "okay", "thing", "things", "stuff", "because", "well",
];

/// The content words of a line or a description, lowercased, long enough to mean something.
pub fn content_words(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| w.len() >= 4 && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// How well a shot's description answers a line, as shared content words.
///
/// Deliberately not embeddings: the builder's whole shape is that code enumerates and Jev judges,
/// and pulling the search subsystem in here would widen what a change to this file can break. The
/// descriptions are keyword-rich — "deer", "fence", "patio", "hillside" — so overlap is enough to
/// shortlist, and Jev still does the choosing.
pub fn relevance(line: &std::collections::HashSet<String>, shot: &str) -> usize {
    let words = content_words(shot);
    line.iter().filter(|w| words.contains(*w)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwords_and_short_words_do_not_count_as_a_match() {
        // "that", "with", "have" appear in every sentence and distinguish nothing.
        let words = content_words("And that is the thing that we have here with you, because well.");
        assert!(words.is_empty(), "nothing here names anything: {words:?}");
        assert_eq!(relevance(&content_words("deer at the fence"), "A wooden fence at dawn."), 1);
    }
}
