use crate::faces::embedder::{cosine_similarity, l2_normalize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceTrack {
    pub id: i64,
    pub video_id: i64,
    pub start_s: f64,
    pub end_s: f64,
    pub n_faces: usize,
    pub mean_embedding: Vec<f32>,
    pub avg_box_h: f32,
    pub boxes: Vec<FaceBoxRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaceBoxRecord {
    pub track_id: i64,
    pub t_s: f64,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub landmarks_json: String,
    pub sharpness: f32,
    pub cut_off: bool,
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub t_s: f64,
    pub box_raw: (f32, f32, f32, f32), // (x, y, w, h) in detection coordinate space
    pub norm_x: f32,
    pub norm_y: f32,
    pub norm_w: f32,
    pub norm_h: f32,
    pub score: f32,
    pub landmarks_json: String,
    pub sharpness: f32,
    pub cut_off: bool,
    pub embedding: Vec<f32>,
    pub size_px: f32,
}

#[derive(Debug)]
pub struct ActiveTrack {
    pub track_id: i64,
    pub last_t_s: f64,
    pub last_box_raw: (f32, f32, f32, f32),
    pub last_embedding: Vec<f32>,
    pub faces: Vec<DetectedFace>,
}

pub struct FaceTracker {
    video_id: i64,
    next_track_id: i64,
    active_tracks: Vec<ActiveTrack>,
    finished_tracks: Vec<FaceTrack>,
}

impl FaceTracker {
    pub fn new(video_id: i64) -> Self {
        Self { video_id, next_track_id: 1, active_tracks: Vec::new(), finished_tracks: Vec::new() }
    }

    /// Process detections for a single frame at `t_s`.
    pub fn process_frame(&mut self, t_s: f64, detections: Vec<DetectedFace>) {
        // 1. Expire tracks without a match for > 1.0 s
        let mut still_active = Vec::new();
        for mut track in self.active_tracks.drain(..) {
            if t_s - track.last_t_s > 1.0 + 1e-4 {
                let finished = Self::finalize_track(self.video_id, &mut track);
                self.finished_tracks.push(finished);
            } else {
                still_active.push(track);
            }
        }
        self.active_tracks = still_active;

        if detections.is_empty() {
            return;
        }

        // 2. Compute candidate pairs with IoU >= 0.3
        struct MatchCandidate {
            track_idx: usize,
            det_idx: usize,
            iou: f32,
            cos: f32,
        }

        let mut candidates = Vec::new();
        for (t_idx, track) in self.active_tracks.iter().enumerate() {
            for (d_idx, det) in detections.iter().enumerate() {
                let iou = raw_iou(track.last_box_raw, det.box_raw);
                if iou >= 0.3 {
                    let cos = cosine_similarity(&track.last_embedding, &det.embedding);
                    candidates.push(MatchCandidate { track_idx: t_idx, det_idx: d_idx, iou, cos });
                }
            }
        }

        // 3. Sort candidate pairs: tie-break by embedding cosine, then IoU
        candidates.sort_by(|a, b| {
            b.cos
                .partial_cmp(&a.cos)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.iou.partial_cmp(&a.iou).unwrap_or(std::cmp::Ordering::Equal))
        });

        // 4. Greedy assignment
        let mut matched_tracks = vec![false; self.active_tracks.len()];
        let mut matched_dets = vec![false; detections.len()];

        for cand in candidates {
            if !matched_tracks[cand.track_idx] && !matched_dets[cand.det_idx] {
                matched_tracks[cand.track_idx] = true;
                matched_dets[cand.det_idx] = true;

                let det = detections[cand.det_idx].clone();
                let track = &mut self.active_tracks[cand.track_idx];
                track.last_t_s = det.t_s;
                track.last_box_raw = det.box_raw;
                track.last_embedding = det.embedding.clone();
                track.faces.push(det);
            }
        }

        // 5. Unmatched detections start new tracks
        for (d_idx, det) in detections.into_iter().enumerate() {
            if !matched_dets[d_idx] {
                let track_id = self.next_track_id;
                self.next_track_id += 1;
                self.active_tracks.push(ActiveTrack {
                    track_id,
                    last_t_s: det.t_s,
                    last_box_raw: det.box_raw,
                    last_embedding: det.embedding.clone(),
                    faces: vec![det],
                });
            }
        }
    }

    /// Finish tracking across all frames and return all tracks.
    pub fn finish(mut self) -> Vec<FaceTrack> {
        for mut track in self.active_tracks.drain(..) {
            let finished = Self::finalize_track(self.video_id, &mut track);
            self.finished_tracks.push(finished);
        }
        self.finished_tracks.sort_by(|a, b| {
            a.start_s.partial_cmp(&b.start_s).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id))
        });
        self.finished_tracks
    }

    fn finalize_track(video_id: i64, track: &mut ActiveTrack) -> FaceTrack {
        let n_faces = track.faces.len();
        let start_s = track.faces.first().map(|f| f.t_s).unwrap_or(0.0);
        let end_s = track.faces.last().map(|f| f.t_s).unwrap_or(0.0);
        let avg_box_h =
            if n_faces > 0 { track.faces.iter().map(|f| f.norm_h).sum::<f32>() / n_faces as f32 } else { 0.0 };

        // Median sharpness
        let mut sharps: Vec<f32> = track.faces.iter().map(|f| f.sharpness).collect();
        sharps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_sharp = if sharps.is_empty() { 0.0 } else { sharps[sharps.len() / 2] };

        // Faces with sharpness >= median and size >= 40 px
        let eligible: Vec<&DetectedFace> =
            track.faces.iter().filter(|f| f.sharpness >= median_sharp && f.size_px >= 40.0).collect();

        let pool = if !eligible.is_empty() {
            eligible
        } else {
            // Fallback to all faces if none matched criteria
            track.faces.iter().collect()
        };

        let mut mean_v = vec![0.0f32; 128];
        if !pool.is_empty() {
            for face in &pool {
                for (i, val) in face.embedding.iter().enumerate() {
                    mean_v[i] += *val;
                }
            }
            mean_v = l2_normalize(&mean_v);
        }

        let boxes = track
            .faces
            .iter()
            .map(|f| FaceBoxRecord {
                track_id: track.track_id,
                t_s: f.t_s,
                x: f.norm_x,
                y: f.norm_y,
                w: f.norm_w,
                h: f.norm_h,
                score: f.score,
                landmarks_json: f.landmarks_json.clone(),
                sharpness: f.sharpness,
                cut_off: f.cut_off,
            })
            .collect();

        FaceTrack { id: track.track_id, video_id, start_s, end_s, n_faces, mean_embedding: mean_v, avg_box_h, boxes }
    }
}

pub fn raw_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = (a.0 + a.2).min(b.0 + b.2);
    let y2 = (a.1 + a.3).min(b.1 + b.3);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a.2 * a.3;
    let area_b = b.2 * b.3;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
}

/// Agglomerative hierarchical clustering of tracks by mean embedding (average-linkage).
/// Clusters are merged while the average pairwise cosine similarity between them is >= `threshold`.
pub fn cluster_tracks_average_linkage(tracks: &[FaceTrack], threshold: f32) -> Vec<Vec<usize>> {
    let embs: Vec<&[f32]> = tracks.iter().map(|t| t.mean_embedding.as_slice()).collect();
    crate::speakers::cluster::cluster_average_linkage(&embs, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_det(t_s: f64, x: f32, y: f32, w: f32, h: f32, emb_val: f32) -> DetectedFace {
        let mut emb = vec![0.0f32; 128];
        emb[0] = emb_val;
        let embedding = l2_normalize(&emb);
        DetectedFace {
            t_s,
            box_raw: (x, y, w, h),
            norm_x: x / 640.0,
            norm_y: y / 640.0,
            norm_w: w / 640.0,
            norm_h: h / 640.0,
            score: 0.95,
            landmarks_json: "[]".into(),
            sharpness: 100.0,
            cut_off: false,
            embedding,
            size_px: w.max(h),
        }
    }

    #[test]
    fn test_tracking_crossing_faces_keep_ids() {
        let mut tracker = FaceTracker::new(1);

        // Frame 0 at 0.0s: Face 1 on left, Face 2 on right
        let d1_0 = make_det(0.0, 140.0, 100.0, 80.0, 80.0, 1.0);
        let d2_0 = make_det(0.0, 260.0, 100.0, 80.0, 80.0, -1.0);
        tracker.process_frame(0.0, vec![d1_0, d2_0]);

        // Frame 1 at 0.2s: Both move closer towards center (IoU with prev frame = 0.33 >= 0.3)
        let d1_1 = make_det(0.2, 180.0, 100.0, 80.0, 80.0, 1.0);
        let d2_1 = make_det(0.2, 220.0, 100.0, 80.0, 80.0, -1.0);
        tracker.process_frame(0.2, vec![d1_1, d2_1]);

        // Frame 2 at 0.4s: They cross paths!
        // Face 1 moves to 220 (IoU 1.0 with Face 2's previous box, but matching embedding 1.0)
        // Face 2 moves to 180 (IoU 1.0 with Face 1's previous box, but matching embedding -1.0)
        let d1_2 = make_det(0.4, 220.0, 100.0, 80.0, 80.0, 1.0);
        let d2_2 = make_det(0.4, 180.0, 100.0, 80.0, 80.0, -1.0);
        tracker.process_frame(0.4, vec![d1_2, d2_2]);

        let tracks = tracker.finish();
        assert_eq!(tracks.len(), 2, "There must be exactly 2 tracks");

        // Verify that each track followed the same face (consistent embedding)
        let t1 = &tracks[0];
        let t2 = &tracks[1];

        assert_eq!(t1.n_faces, 3);
        assert_eq!(t2.n_faces, 3);

        // Face 1 started at x=140 and ended at x=220
        let t1_x0 = t1.boxes[0].x * 640.0;
        let t1_x2 = t1.boxes[2].x * 640.0;
        let t2_x0 = t2.boxes[0].x * 640.0;
        let t2_x2 = t2.boxes[2].x * 640.0;

        assert!((t1_x0 - 140.0).abs() < 1.0);
        assert!((t1_x2 - 220.0).abs() < 1.0);

        assert!((t2_x0 - 260.0).abs() < 1.0);
        assert!((t2_x2 - 180.0).abs() < 1.0);
    }

    #[test]
    fn test_tracking_gap_over_1s_splits() {
        let mut tracker = FaceTracker::new(1);

        // Face appears at 0.0s and 0.2s
        let d0 = make_det(0.0, 100.0, 100.0, 80.0, 80.0, 1.0);
        tracker.process_frame(0.0, vec![d0]);
        let d1 = make_det(0.2, 100.0, 100.0, 80.0, 80.0, 1.0);
        tracker.process_frame(0.2, vec![d1]);

        // Gap of 1.2s (> 1.0s) until 1.4s
        let d2 = make_det(1.4, 100.0, 100.0, 80.0, 80.0, 1.0);
        tracker.process_frame(1.4, vec![d2]);

        let tracks = tracker.finish();
        assert_eq!(tracks.len(), 2, "Gap > 1s must split into 2 tracks");
        assert_eq!(tracks[0].start_s, 0.0);
        assert_eq!(tracks[0].end_s, 0.2);
        assert_eq!(tracks[0].n_faces, 2);

        assert_eq!(tracks[1].start_s, 1.4);
        assert_eq!(tracks[1].end_s, 1.4);
        assert_eq!(tracks[1].n_faces, 1);
        assert_ne!(tracks[0].id, tracks[1].id, "Track IDs must be distinct");
    }
}
