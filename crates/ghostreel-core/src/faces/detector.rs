use crate::Error;
use ort::value::Tensor as OrtTensor;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub landmarks: [(f32, f32); 5],
}

pub struct OrtDetector {
    session: ort::session::Session,
}

impl OrtDetector {
    pub fn new(env: &Arc<ort::environment::Environment>, model_path: &str) -> Result<Self, Error> {
        let session = ort::session::Session::builder(env)
            .map_err(|e| Error::Invalid(format!("Failed to build ort session: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| Error::Invalid(format!("Failed to create YuNet ort session from {model_path}: {e}")))?;
        Ok(Self { session })
    }

    pub fn detect(&mut self, bgr_nchw: &[f32], score_thresh: f32, nms_thresh: f32) -> Result<Vec<FaceBox>, Error> {
        let ort_in = OrtTensor::from_array(([1, 3, 640, 640], bgr_nchw.to_vec()))
            .map_err(|e| Error::Invalid(format!("YuNet tensor error: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![ort_in])
            .map_err(|e| Error::Invalid(format!("YuNet inference error: {e}")))?;

        let (_, s0) = outputs[0].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s1) = outputs[1].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s2) = outputs[2].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s3) = outputs[3].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s4) = outputs[4].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s5) = outputs[5].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s6) = outputs[6].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s7) = outputs[7].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s8) = outputs[8].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s9) = outputs[9].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s10) = outputs[10].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        let (_, s11) = outputs[11].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;

        Ok(decode_yunet([s0, s1, s2], [s3, s4, s5], [s6, s7, s8], [s9, s10, s11], 640, 640, score_thresh, nms_thresh))
    }
}

pub fn decode_yunet(
    cls_outputs: [&[f32]; 3],
    obj_outputs: [&[f32]; 3],
    bbox_outputs: [&[f32]; 3],
    kps_outputs: [&[f32]; 3],
    pad_w: usize,
    pad_h: usize,
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<FaceBox> {
    let strides = [8, 16, 32];
    let mut candidates = Vec::new();

    for i in 0..3 {
        let stride = strides[i];
        let cols = pad_w / stride;
        let rows = pad_h / stride;

        let cls_v = cls_outputs[i];
        let obj_v = obj_outputs[i];
        let bbox_v = bbox_outputs[i];
        let kps_v = kps_outputs[i];

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let cls_score = cls_v[idx].clamp(0.0, 1.0);
                let obj_score = obj_v[idx].clamp(0.0, 1.0);
                let score = (cls_score * obj_score).sqrt();

                if score < score_threshold {
                    continue;
                }

                let cx = (c as f32 + bbox_v[idx * 4 + 0]) * stride as f32;
                let cy = (r as f32 + bbox_v[idx * 4 + 1]) * stride as f32;
                let w = (bbox_v[idx * 4 + 2]).exp() * stride as f32;
                let h = (bbox_v[idx * 4 + 3]).exp() * stride as f32;

                let x1 = cx - w / 2.0;
                let y1 = cy - h / 2.0;

                let mut landmarks = [(0.0, 0.0); 5];
                for n in 0..5 {
                    let lx = (kps_v[idx * 10 + 2 * n] + c as f32) * stride as f32;
                    let ly = (kps_v[idx * 10 + 2 * n + 1] + r as f32) * stride as f32;
                    landmarks[n] = (lx, ly);
                }

                candidates.push(FaceBox { x: x1, y: y1, w, h, score, landmarks });
            }
        }
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    for cand in candidates {
        let mut suppress = false;
        for kept in &keep {
            if box_iou(&cand, kept) > nms_threshold {
                suppress = true;
                break;
            }
        }
        if !suppress {
            keep.push(cand);
        }
    }

    keep
}

pub fn box_iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
}
