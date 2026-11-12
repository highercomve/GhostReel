//! From transcript segments to embedding windows, and from labelled windows to speaker turns.

use serde::Serialize;

/// Segments shorter than this carry too little voice to embed.
pub const MIN_WINDOW_S: f64 = 0.8;
/// Long segments are cut into windows of about this length (1.33–3 s).
pub const TARGET_WINDOW_S: f64 = 2.0;
/// Same-speaker windows closer than this join into one turn.
pub const MAX_TURN_GAP_S: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSpan {
    pub segment_id: i64,
    pub start_s: f64,
    pub end_s: f64,
}

/// Embedding windows for transcript segments `(id, start_s, end_s)`.
pub fn windows_from_segments(segments: &[(i64, f64, f64)]) -> Vec<WindowSpan> {
    let mut out = Vec::new();
    for &(segment_id, start_s, end_s) in segments {
        let d = end_s - start_s;
        if d.is_nan() || d < MIN_WINDOW_S {
            continue;
        }
        let n = ((d / TARGET_WINDOW_S).round() as usize).max(1);
        let w = d / n as f64;
        for k in 0..n {
            out.push(WindowSpan { segment_id, start_s: start_s + k as f64 * w, end_s: start_s + (k + 1) as f64 * w });
        }
    }
    out.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    out
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Turn {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: i64,
}

/// Merge time-ordered `(start_s, end_s, speaker)` windows into turns: consecutive windows of the
/// same speaker separated by at most `max_gap` seconds become one turn.
pub fn merge_turns(windows: &[(f64, f64, i64)], max_gap: f64) -> Vec<Turn> {
    let mut sorted = windows.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut turns: Vec<Turn> = Vec::new();
    for (s, e, spk) in sorted {
        if let Some(last) = turns.last_mut()
            && last.speaker == spk
            && s - last.end_s <= max_gap
        {
            last.end_s = last.end_s.max(e);
            continue;
        }
        turns.push(Turn { start_s: s, end_s: e, speaker: spk });
    }
    turns
}

/// The speaker overlapping a segment the most, if any turn overlaps it.
pub fn majority_speaker(start_s: f64, end_s: f64, turns: &[Turn]) -> Option<i64> {
    let mut acc: Vec<(i64, f64)> = Vec::new();
    for t in turns {
        let overlap = end_s.min(t.end_s) - start_s.max(t.start_s);
        if overlap > 0.0 {
            match acc.iter_mut().find(|(s, _)| *s == t.speaker) {
                Some((_, d)) => *d += overlap,
                None => acc.push((t.speaker, overlap)),
            }
        }
    }
    acc.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(s, _)| s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_split_long_and_skip_short() {
        let w = windows_from_segments(&[(1, 0.0, 0.5), (2, 1.0, 2.0), (3, 3.0, 9.0), (4, 10.0, 12.9)]);
        assert!(w.iter().all(|x| x.segment_id != 1), "0.5 s is too short");
        assert_eq!(w.iter().filter(|x| x.segment_id == 2).count(), 1);
        let s3: Vec<_> = w.iter().filter(|x| x.segment_id == 3).collect();
        assert_eq!(s3.len(), 3);
        assert!((s3[0].end_s - 5.0).abs() < 1e-9 && (s3[2].end_s - 9.0).abs() < 1e-9);
        assert_eq!(w.iter().filter(|x| x.segment_id == 4).count(), 1, "2.9 s stays whole");
        for x in &w {
            let d = x.end_s - x.start_s;
            assert!((0.8..=3.0).contains(&d), "window {d}");
        }
    }

    #[test]
    fn turns_merge_adjacent_same_speaker() {
        let turns = merge_turns(
            &[(0.0, 2.0, 0), (2.0, 4.0, 0), (4.5, 6.0, 0), (6.0, 8.0, 1), (8.0, 9.0, 0), (12.0, 13.0, 0)],
            1.0,
        );
        assert_eq!(
            turns,
            vec![
                Turn { start_s: 0.0, end_s: 6.0, speaker: 0 },
                Turn { start_s: 6.0, end_s: 8.0, speaker: 1 },
                Turn { start_s: 8.0, end_s: 9.0, speaker: 0 },
                Turn { start_s: 12.0, end_s: 13.0, speaker: 0 },
            ]
        );
    }

    #[test]
    fn segment_takes_the_majority_speaker() {
        let turns = vec![
            Turn { start_s: 0.0, end_s: 3.0, speaker: 0 },
            Turn { start_s: 3.0, end_s: 10.0, speaker: 1 },
        ];
        assert_eq!(majority_speaker(2.0, 6.0, &turns), Some(1));
        assert_eq!(majority_speaker(0.0, 3.5, &turns), Some(0));
        assert_eq!(majority_speaker(11.0, 12.0, &turns), None);
    }
}
