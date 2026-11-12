use anyhow::{Context, Result};
use ort::value::Tensor as OrtTensor;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub trait Embedder {
    fn embed(&mut self, rgb_nchw: &[f32]) -> Result<Vec<f32>>;
}

pub struct TractEmbedder {
    model: Arc<tract_onnx::prelude::RunnableModel<TypedFact, Box<dyn TypedOp>>>,
}

impl TractEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("Failed to load tract model from {model_path}"))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }
}

impl Embedder for TractEmbedder {
    fn embed(&mut self, rgb_nchw: &[f32]) -> Result<Vec<f32>> {
        let tract_in: tract_onnx::prelude::Tensor =
            tract_ndarray::Array4::from_shape_vec((1, 3, 112, 112), rgb_nchw.to_vec())?.into();
        let tract_out = self.model.run(tvec!(tract_in.into()))?;
        let raw = tract_out[0].as_plain_ram().context("plain ram")?.as_slice::<f32>()?.to_vec();
        Ok(l2_normalize(&raw))
    }
}

pub struct OrtEmbedder {
    session: ort::session::Session,
}

impl OrtEmbedder {
    pub fn new(env: &Arc<ort::environment::Environment>, model_path: &str) -> Result<Self> {
        let session = ort::session::Session::builder(env)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to create ort session from {model_path}"))?;
        Ok(Self { session })
    }
}

impl Embedder for OrtEmbedder {
    fn embed(&mut self, rgb_nchw: &[f32]) -> Result<Vec<f32>> {
        let ort_in = OrtTensor::from_array(([1, 3, 112, 112], rgb_nchw.to_vec()))?;
        let ort_out = self.session.run(ort::inputs![ort_in])?;
        let (_, ort_slice) = ort_out[0].try_extract_tensor::<f32>()?;
        Ok(l2_normalize(ort_slice))
    }
}

pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
