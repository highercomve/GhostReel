use anyhow::{Context, Result};
use ort::value::Tensor as OrtTensor;
use std::sync::Arc;
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub landmarks: [(f32, f32); 5],
}

pub trait Detector {
    fn detect(&mut self, bgr_nchw: &[f32], score_thresh: f32, nms_thresh: f32) -> Result<Vec<FaceBox>>;
}

pub struct TractDetector {
    model: Arc<tract_onnx::prelude::RunnableModel<TypedFact, Box<dyn TypedOp>>>,
}

impl TractDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("Failed to load tract model from {model_path}"))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }
}

impl Detector for TractDetector {
    fn detect(&mut self, bgr_nchw: &[f32], score_thresh: f32, nms_thresh: f32) -> Result<Vec<FaceBox>> {
        let tensor: tract_onnx::prelude::Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, 640, 640), bgr_nchw.to_vec())?.into();
        let outputs = self.model.run(tvec!(tensor.into()))?;

        let p0 = outputs[0].as_plain_ram().context("output 0 plain ram")?;
        let p1 = outputs[1].as_plain_ram().context("output 1 plain ram")?;
        let p2 = outputs[2].as_plain_ram().context("output 2 plain ram")?;
        let p3 = outputs[3].as_plain_ram().context("output 3 plain ram")?;
        let p4 = outputs[4].as_plain_ram().context("output 4 plain ram")?;
        let p5 = outputs[5].as_plain_ram().context("output 5 plain ram")?;
        let p6 = outputs[6].as_plain_ram().context("output 6 plain ram")?;
        let p7 = outputs[7].as_plain_ram().context("output 7 plain ram")?;
        let p8 = outputs[8].as_plain_ram().context("output 8 plain ram")?;
        let p9 = outputs[9].as_plain_ram().context("output 9 plain ram")?;
        let p10 = outputs[10].as_plain_ram().context("output 10 plain ram")?;
        let p11 = outputs[11].as_plain_ram().context("output 11 plain ram")?;

        let s0 = p0.as_slice::<f32>()?;
        let s1 = p1.as_slice::<f32>()?;
        let s2 = p2.as_slice::<f32>()?;
        let s3 = p3.as_slice::<f32>()?;
        let s4 = p4.as_slice::<f32>()?;
        let s5 = p5.as_slice::<f32>()?;
        let s6 = p6.as_slice::<f32>()?;
        let s7 = p7.as_slice::<f32>()?;
        let s8 = p8.as_slice::<f32>()?;
        let s9 = p9.as_slice::<f32>()?;
        let s10 = p10.as_slice::<f32>()?;
        let s11 = p11.as_slice::<f32>()?;

        Ok(decode_yunet(
            [s0, s1, s2],
            [s3, s4, s5],
            [s6, s7, s8],
            [s9, s10, s11],
            640,
            640,
            score_thresh,
            nms_thresh,
        ))
    }
}

pub struct OrtDetector {
    session: ort::session::Session,
}

impl OrtDetector {
    pub fn new(env: &Arc<ort::environment::Environment>, model_path: &str) -> Result<Self> {
        let session = ort::session::Session::builder(env)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to create ort session from {model_path}"))?;
        Ok(Self { session })
    }
}

impl Detector for OrtDetector {
    fn detect(&mut self, bgr_nchw: &[f32], score_thresh: f32, nms_thresh: f32) -> Result<Vec<FaceBox>> {
        let ort_in = OrtTensor::from_array(([1, 3, 640, 640], bgr_nchw.to_vec()))?;
        let outputs = self.session.run(ort::inputs![ort_in])?;

        let (_, s0) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, s1) = outputs[1].try_extract_tensor::<f32>()?;
        let (_, s2) = outputs[2].try_extract_tensor::<f32>()?;
        let (_, s3) = outputs[3].try_extract_tensor::<f32>()?;
        let (_, s4) = outputs[4].try_extract_tensor::<f32>()?;
        let (_, s5) = outputs[5].try_extract_tensor::<f32>()?;
        let (_, s6) = outputs[6].try_extract_tensor::<f32>()?;
        let (_, s7) = outputs[7].try_extract_tensor::<f32>()?;
        let (_, s8) = outputs[8].try_extract_tensor::<f32>()?;
        let (_, s9) = outputs[9].try_extract_tensor::<f32>()?;
        let (_, s10) = outputs[10].try_extract_tensor::<f32>()?;
        let (_, s11) = outputs[11].try_extract_tensor::<f32>()?;

        Ok(decode_yunet(
            [s0, s1, s2],
            [s3, s4, s5],
            [s6, s7, s8],
            [s9, s10, s11],
            640,
            640,
            score_thresh,
            nms_thresh,
        ))
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

                candidates.push(FaceBox {
                    x: x1,
                    y: y1,
                    w,
                    h,
                    score,
                    landmarks,
                });
            }
        }
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    for cand in candidates {
        let mut suppress = false;
        for kept in &keep {
            if iou(&cand, kept) > nms_threshold {
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

fn iou(a: &FaceBox, b: &FaceBox) -> f32 {
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

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}
