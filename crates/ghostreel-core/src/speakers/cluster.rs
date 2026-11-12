//! Agglomerative average-linkage clustering on cosine similarity, shared by faces and voices.

use crate::faces::{cosine_similarity, l2_normalize};

/// Clusters of `embeddings` (indices), merging while the average pairwise cosine similarity
/// between two clusters is >= `threshold`.
///
/// Builds the full dendrogram with the nearest-neighbour chain algorithm (O(n²) time and memory)
/// and cuts it at `threshold`. Average linkage has no inversions, so this gives the same clusters
/// as greedily merging the most similar pair until none is above the threshold.
pub fn cluster_average_linkage(embeddings: &[&[f32]], threshold: f32) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![0]];
    }
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = cosine_similarity(embeddings[i], embeddings[j]);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut remaining = n;

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    while remaining > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).unwrap_or(0));
        }
        let a = *chain.last().unwrap_or(&0);
        let prev = if chain.len() >= 2 { Some(chain[chain.len() - 2]) } else { None };
        // Nearest active neighbour of `a`; prefer the previous chain element on ties so the
        // chain terminates.
        let mut best = usize::MAX;
        let mut best_sim = f32::NEG_INFINITY;
        if let Some(p) = prev {
            best = p;
            best_sim = sim[a * n + p];
        }
        for c in 0..n {
            if c != a && active[c] && sim[a * n + c] > best_sim {
                best_sim = sim[a * n + c];
                best = c;
            }
        }
        if Some(best) == prev {
            chain.pop();
            chain.pop();
            let (keep, gone) = (a.min(best), a.max(best));
            if best_sim >= threshold {
                let (ra, rb) = (find(&mut parent, keep), find(&mut parent, gone));
                parent[rb] = ra;
            }
            // Lance-Williams update for average linkage.
            let (sk, sg) = (size[keep] as f32, size[gone] as f32);
            for m in 0..n {
                if active[m] && m != keep && m != gone {
                    let s = (sk * sim[keep * n + m] + sg * sim[gone * n + m]) / (sk + sg);
                    sim[keep * n + m] = s;
                    sim[m * n + keep] = s;
                }
            }
            size[keep] += size[gone];
            active[gone] = false;
            remaining -= 1;
        } else {
            chain.push(best);
        }
    }

    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Mean of the given embeddings, L2-normalised.
pub fn centroid(embeddings: &[&[f32]]) -> Vec<f32> {
    let Some(first) = embeddings.first() else { return Vec::new() };
    let mut mean = vec![0.0f32; first.len()];
    for e in embeddings {
        for (m, v) in mean.iter_mut().zip(e.iter()) {
            *m += v;
        }
    }
    l2_normalize(&mean)
}

/// Thresholds for turning voice-window clusters into speakers.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerClusterParams {
    /// Average-linkage cut (cosine). Within a speaker, 2 s CAM++ windows average ~0.45–0.5;
    /// two different hosts ~0.27, so 0.4 splits people and leaves a few small fragments.
    pub threshold: f32,
    /// A cluster smaller than this many seconds is a fragment, not a speaker.
    pub min_speaker_s: f64,
    /// ... nor one below this share of all speech.
    pub min_speaker_share: f64,
    /// Speakers whose centroids are this similar are the same voice.
    pub merge_centroid_cos: f32,
    /// At most this many windows go into the O(n²) clustering; the rest join the nearest centroid.
    pub max_windows: usize,
}

impl Default for SpeakerClusterParams {
    fn default() -> Self {
        Self { threshold: 0.4, min_speaker_s: 15.0, min_speaker_share: 0.02, merge_centroid_cos: 0.6, max_windows: 3000 }
    }
}

/// Speaker label (0-based, most speech first) for each window, plus each speaker's centroid.
///
/// Clusters (a sample of) the windows, keeps clusters with enough speech as speakers, merges
/// speakers whose centroids are near-identical, then gives every window the nearest speaker.
pub fn label_speakers(
    embeddings: &[&[f32]],
    durations: &[f64],
    params: &SpeakerClusterParams,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let sample: Vec<usize> = if n > params.max_windows {
        (0..params.max_windows).map(|i| i * n / params.max_windows).collect()
    } else {
        (0..n).collect()
    };
    let sample_embs: Vec<&[f32]> = sample.iter().map(|&i| embeddings[i]).collect();
    let clusters = cluster_average_linkage(&sample_embs, params.threshold);
    let total: f64 = sample.iter().map(|&i| durations[i]).sum();
    let min_s = params.min_speaker_s.max(params.min_speaker_share * total);
    let mut scored: Vec<(f64, Vec<usize>)> = clusters
        .into_iter()
        .map(|c| {
            let members: Vec<usize> = c.iter().map(|&k| sample[k]).collect();
            (members.iter().map(|&i| durations[i]).sum(), members)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut speakers: Vec<(f64, Vec<usize>)> = scored.iter().filter(|(d, _)| *d >= min_s).cloned().collect();
    if speakers.is_empty() {
        speakers.push(scored[0].clone());
    }
    // Merge speakers with near-identical centroids (one voice split by the cut).
    loop {
        let cents: Vec<Vec<f32>> =
            speakers.iter().map(|(_, m)| centroid(&m.iter().map(|&i| embeddings[i]).collect::<Vec<_>>())).collect();
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..cents.len() {
            for j in (i + 1)..cents.len() {
                let s = cosine_similarity(&cents[i], &cents[j]);
                if s >= params.merge_centroid_cos && best.is_none_or(|(_, _, b)| s > b) {
                    best = Some((i, j, s));
                }
            }
        }
        let Some((i, j, _)) = best else { break };
        let (d, mut m) = speakers.remove(j);
        speakers[i].0 += d;
        speakers[i].1.append(&mut m);
    }
    let mut cents: Vec<Vec<f32>> =
        speakers.iter().map(|(_, m)| centroid(&m.iter().map(|&i| embeddings[i]).collect::<Vec<_>>())).collect();
    let nearest = |e: &[f32], cents: &[Vec<f32>]| -> usize {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (k, c) in cents.iter().enumerate() {
            let s = cosine_similarity(e, c);
            if s > best.1 {
                best = (k, s);
            }
        }
        best.0
    };
    let labels: Vec<usize> = embeddings.iter().map(|e| nearest(e, &cents)).collect();
    // Order speakers by total speech after reassignment, and recompute centroids from it.
    let mut speech = vec![0.0f64; cents.len()];
    for (l, d) in labels.iter().zip(durations) {
        speech[*l] += d;
    }
    let mut order: Vec<usize> = (0..cents.len()).filter(|&k| speech[k] > 0.0).collect();
    order.sort_by(|&a, &b| speech[b].total_cmp(&speech[a]));
    let mut remap = vec![usize::MAX; cents.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    let labels: Vec<usize> = labels.into_iter().map(|l| remap[l]).collect();
    cents = (0..order.len())
        .map(|k| {
            centroid(&labels.iter().enumerate().filter(|(_, l)| **l == k).map(|(i, _)| embeddings[i]).collect::<Vec<_>>())
        })
        .collect();
    (labels, cents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        l2_normalize(v)
    }

    /// Reference implementation: the greedy O(n³) loop faces used before.
    #[allow(clippy::needless_range_loop)]
    fn greedy(embs: &[&[f32]], threshold: f32) -> Vec<Vec<usize>> {
        let n = embs.len();
        let mut sim: Vec<Vec<f32>> =
            (0..n).map(|i| (0..n).map(|j| cosine_similarity(embs[i], embs[j])).collect()).collect();
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut sizes = vec![1usize; n];
        loop {
            let k = clusters.len();
            if k <= 1 {
                break;
            }
            let (mut bs, mut bi, mut bj) = (f32::NEG_INFINITY, 0, 0);
            for i in 0..k {
                for j in (i + 1)..k {
                    if sim[i][j] > bs {
                        (bs, bi, bj) = (sim[i][j], i, j);
                    }
                }
            }
            if bs < threshold {
                break;
            }
            let (si, sj) = (sizes[bi] as f32, sizes[bj] as f32);
            for m in 0..k {
                if m != bi && m != bj {
                    let s = (si * sim[bi][m] + sj * sim[bj][m]) / (si + sj);
                    sim[bi][m] = s;
                    sim[m][bi] = s;
                }
            }
            sim.remove(bj);
            for row in &mut sim {
                row.remove(bj);
            }
            let mut moved = clusters.remove(bj);
            clusters[bi].append(&mut moved);
            sizes[bi] += sizes.remove(bj);
        }
        clusters
    }

    fn normalised(mut c: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        for g in &mut c {
            g.sort();
        }
        c.sort();
        c
    }

    #[test]
    fn nn_chain_matches_greedy() {
        // Deterministic pseudo-random points around three directions.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let mut pts = Vec::new();
        for k in 0..60 {
            let mut v = vec![0.0f32; 8];
            v[k % 3] = 1.0;
            for x in v.iter_mut() {
                *x += rnd() * 0.9;
            }
            pts.push(unit(&v));
        }
        let refs: Vec<&[f32]> = pts.iter().map(|v| v.as_slice()).collect();
        for th in [0.9, 0.6, 0.4, 0.2, -0.2] {
            assert_eq!(normalised(cluster_average_linkage(&refs, th)), normalised(greedy(&refs, th)), "threshold {th}");
        }
    }

    #[test]
    fn two_voices_and_a_fragment() {
        let a = unit(&[1.0, 0.1, 0.0, 0.0]);
        let a2 = unit(&[0.95, 0.0, 0.1, 0.0]);
        let b = unit(&[0.0, 1.0, 0.1, 0.0]);
        let b2 = unit(&[0.1, 0.9, 0.0, 0.1]);
        let odd = unit(&[0.0, 0.0, 0.0, 1.0]);
        let mut embs = Vec::new();
        let mut durs = Vec::new();
        for i in 0..40 {
            embs.push(if i % 2 == 0 { a.clone() } else { a2.clone() });
            durs.push(2.0);
        }
        for i in 0..20 {
            embs.push(if i % 2 == 0 { b.clone() } else { b2.clone() });
            durs.push(2.0);
        }
        embs.push(odd);
        durs.push(2.0);
        let refs: Vec<&[f32]> = embs.iter().map(|v| v.as_slice()).collect();
        let (labels, cents) = label_speakers(&refs, &durs, &SpeakerClusterParams::default());
        assert_eq!(cents.len(), 2, "the 2 s fragment is not a speaker");
        assert!(labels[..40].iter().all(|&l| l == 0), "most speech is speaker 0");
        assert!(labels[40..60].iter().all(|&l| l == 1));
        assert!(labels[60] < 2, "the fragment joins its nearest speaker");
    }

    #[test]
    fn split_voice_is_merged_by_centroid() {
        // One voice, two tight sub-clusters 0.7 apart in cosine: centroid merge makes it one.
        let a = unit(&[1.0, 0.0, 0.0]);
        let b = unit(&[0.7, 0.714, 0.0]);
        let embs: Vec<Vec<f32>> = (0..40).map(|i| if i < 20 { a.clone() } else { b.clone() }).collect();
        let refs: Vec<&[f32]> = embs.iter().map(|v| v.as_slice()).collect();
        let params = SpeakerClusterParams { threshold: 0.9, ..Default::default() };
        let (_, cents) = label_speakers(&refs, &vec![2.0; 40], &params);
        assert_eq!(cents.len(), 1);
    }
}
