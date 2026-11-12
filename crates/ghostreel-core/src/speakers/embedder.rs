//! Voice embeddings with a WeSpeaker CAM++ ONNX model (input `[1, T, 80]` fbank, output 512-d).

use std::sync::Arc;

use ort::value::Tensor as OrtTensor;

use super::fbank::{Fbank, N_MELS};
use crate::Error;
use crate::faces::l2_normalize;

pub struct OrtSpeakerEmbedder {
    session: ort::session::Session,
    fbank: Fbank,
}

impl OrtSpeakerEmbedder {
    pub fn new(env: &Arc<ort::environment::Environment>, model_path: &str) -> Result<Self, Error> {
        let session = ort::session::Session::builder(env)
            .map_err(|e| Error::Invalid(format!("Failed to build ort session: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| Error::Invalid(format!("Failed to create speaker ort session from {model_path}: {e}")))?;
        Ok(Self { session, fbank: Fbank::new() })
    }

    /// L2-normalised embedding of 16 kHz mono PCM in [-1, 1]; None when too short for a frame.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>, Error> {
        let feats = self.fbank.compute(samples, 32768.0);
        let frames = feats.len() / N_MELS;
        if frames < 10 {
            return Ok(None);
        }
        let input = OrtTensor::from_array(([1usize, frames, N_MELS], feats))
            .map_err(|e| Error::Invalid(format!("speaker tensor error: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![input])
            .map_err(|e| Error::Invalid(format!("speaker embedding error: {e}")))?;
        let (_, out) = outputs[0].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(Some(l2_normalize(out)))
    }
}
