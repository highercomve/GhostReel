use crate::Error;
use ort::value::Tensor as OrtTensor;
use std::sync::Arc;

/// Standard ArcFace 112×112 reference landmarks (right eye, left eye, nose tip, right mouth, left mouth).
pub const ARCFACE_TEMPLATE_112: [(f32, f32); 5] =
    [(38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041)];

/// 2D similarity transform: dst = [a, -b; b, a] * src + [tx, ty].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
}

impl SimilarityTransform {
    /// Estimate the similarity transform from 5 source points to 5 destination points using least-squares.
    pub fn estimate(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> Option<Self> {
        let mut src_mx = 0.0f32;
        let mut src_my = 0.0f32;
        let mut dst_mu = 0.0f32;
        let mut dst_mv = 0.0f32;
        for i in 0..5 {
            src_mx += src[i].0;
            src_my += src[i].1;
            dst_mu += dst[i].0;
            dst_mv += dst[i].1;
        }
        src_mx /= 5.0;
        src_my /= 5.0;
        dst_mu /= 5.0;
        dst_mv /= 5.0;

        let mut var_src = 0.0f32;
        let mut num_a = 0.0f32;
        let mut num_b = 0.0f32;

        for i in 0..5 {
            let dx = src[i].0 - src_mx;
            let dy = src[i].1 - src_my;
            let du = dst[i].0 - dst_mu;
            let dv = dst[i].1 - dst_mv;

            var_src += dx * dx + dy * dy;
            num_a += du * dx + dv * dy;
            num_b += dv * dx - du * dy;
        }

        if var_src < 1e-8 {
            return None;
        }

        let a = num_a / var_src;
        let b = num_b / var_src;
        let tx = dst_mu - (a * src_mx - b * src_my);
        let ty = dst_mv - (b * src_mx + a * src_my);

        Some(Self { a, b, tx, ty })
    }

    /// Forward transform: map a point from source space to destination space.
    pub fn transform_point(&self, p: (f32, f32)) -> (f32, f32) {
        let u = self.a * p.0 - self.b * p.1 + self.tx;
        let v = self.b * p.0 + self.a * p.1 + self.ty;
        (u, v)
    }

    /// Inverse transform: map a point from destination space back to source space.
    pub fn inverse_transform_point(&self, dst_p: (f32, f32)) -> Option<(f32, f32)> {
        let det = self.a * self.a + self.b * self.b;
        if det < 1e-8 {
            return None;
        }
        let du = dst_p.0 - self.tx;
        let dv = dst_p.1 - self.ty;
        let x = (self.a * du + self.b * dv) / det;
        let y = (-self.b * du + self.a * dv) / det;
        Some((x, y))
    }
}

/// Bilinear interpolation of an interleaved BGR buffer (e.g. 640x640 frame).
#[inline]
pub fn sample_bgr_bilinear(frame_bgr: &[u8], width: usize, height: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let max_x = (width - 1) as f32;
    let max_y = (height - 1) as f32;
    let clamped_x = x.clamp(0.0, max_x);
    let clamped_y = y.clamp(0.0, max_y);

    let x0 = clamped_x.floor() as usize;
    let y0 = clamped_y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = clamped_x - x0 as f32;
    let fy = clamped_y - y0 as f32;

    let idx00 = (y0 * width + x0) * 3;
    let idx10 = (y0 * width + x1) * 3;
    let idx01 = (y1 * width + x0) * 3;
    let idx11 = (y1 * width + x1) * 3;

    let b00 = frame_bgr[idx00] as f32;
    let g00 = frame_bgr[idx00 + 1] as f32;
    let r00 = frame_bgr[idx00 + 2] as f32;

    let b10 = frame_bgr[idx10] as f32;
    let g10 = frame_bgr[idx10 + 1] as f32;
    let r10 = frame_bgr[idx10 + 2] as f32;

    let b01 = frame_bgr[idx01] as f32;
    let g01 = frame_bgr[idx01 + 1] as f32;
    let r01 = frame_bgr[idx01 + 2] as f32;

    let b11 = frame_bgr[idx11] as f32;
    let g11 = frame_bgr[idx11 + 1] as f32;
    let r11 = frame_bgr[idx11 + 2] as f32;

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let b = b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11;
    let g = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11;
    let r = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;

    (b, g, r)
}

/// Warps a 640x640 BGR frame into an aligned 112x112 RGB NCHW buffer and computes aligned gray pixels.
pub fn align_face_112(
    frame_bgr: &[u8],
    frame_w: usize,
    frame_h: usize,
    landmarks: &[(f32, f32); 5],
    rgb_nchw: &mut [f32; 1 * 3 * 112 * 112],
    gray_112: &mut [f32; 112 * 112],
) {
    let transform = SimilarityTransform::estimate(landmarks, &ARCFACE_TEMPLATE_112);

    for v in 0..112 {
        for u in 0..112 {
            let (src_x, src_y) = match transform.and_then(|t| t.inverse_transform_point((u as f32, v as f32))) {
                Some((x, y)) => (x, y),
                None => (u as f32, v as f32),
            };

            let (b, g, r) = sample_bgr_bilinear(frame_bgr, frame_w, frame_h, src_x, src_y);

            rgb_nchw[0 * 112 * 112 + v * 112 + u] = r;
            rgb_nchw[1 * 112 * 112 + v * 112 + u] = g;
            rgb_nchw[2 * 112 * 112 + v * 112 + u] = b;

            gray_112[v * 112 + u] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
}

/// Computes the sharpness of an aligned 112x112 gray crop using the variance of the 3x3 Laplacian operator.
pub fn laplacian_variance(gray: &[f32; 112 * 112]) -> f32 {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0.0f64;

    for y in 1..111 {
        for x in 1..111 {
            let center = gray[y * 112 + x];
            let up = gray[(y - 1) * 112 + x];
            let down = gray[(y + 1) * 112 + x];
            let left = gray[y * 112 + (x - 1)];
            let right = gray[y * 112 + (x + 1)];

            let lap = (up + down + left + right - 4.0 * center) as f64;
            sum += lap;
            sum_sq += lap * lap;
            count += 1.0;
        }
    }

    if count <= 0.0 {
        return 0.0;
    }

    let mean = sum / count;
    let variance = (sum_sq / count) - (mean * mean);
    variance.max(0.0) as f32
}

pub struct OrtEmbedder {
    session: ort::session::Session,
}

impl OrtEmbedder {
    pub fn new(env: &Arc<ort::environment::Environment>, model_path: &str) -> Result<Self, Error> {
        let session = ort::session::Session::builder(env)
            .map_err(|e| Error::Invalid(format!("Failed to build ort session: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| Error::Invalid(format!("Failed to create SFace ort session from {model_path}: {e}")))?;
        Ok(Self { session })
    }

    pub fn embed(&mut self, rgb_nchw: &[f32; 1 * 3 * 112 * 112]) -> Result<Vec<f32>, Error> {
        let ort_in = OrtTensor::from_array(([1, 3, 112, 112], rgb_nchw.to_vec()))
            .map_err(|e| Error::Invalid(format!("SFace tensor error: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![ort_in])
            .map_err(|e| Error::Invalid(format!("SFace inference error: {e}")))?;
        let (_, ort_slice) = outputs[0].try_extract_tensor::<f32>().map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(l2_normalize(ort_slice))
    }
}

pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 { vec.iter().map(|x| x / norm).collect() } else { vec.to_vec() }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_transform_alignment() {
        let template = ARCFACE_TEMPLATE_112;
        // Known similarity transform: scale s = 1.6, theta = 0.25 radians, tx = 35.0, ty = -18.0
        let s = 1.6f32;
        let theta = 0.25f32;
        let a = s * theta.cos();
        let b = s * theta.sin();
        let tx = 35.0f32;
        let ty = -18.0f32;

        // Apply known transform to template points to produce src_points:
        let mut src_points = [(0.0f32, 0.0f32); 5];
        for i in 0..5 {
            let u = template[i].0;
            let v = template[i].1;
            let x = a * u - b * v + tx;
            let y = b * u + a * v + ty;
            src_points[i] = (x, y);
        }

        // Estimate similarity transform from src_points back to template
        let transform = SimilarityTransform::estimate(&src_points, &template).expect("transform must exist");

        for i in 0..5 {
            let mapped = transform.transform_point(src_points[i]);
            let dx = (mapped.0 - template[i].0).abs();
            let dy = (mapped.1 - template[i].1).abs();
            assert!(
                dx < 1e-3 && dy < 1e-3,
                "Point {i}: mapped ({}, {}) vs template ({}, {}), dx={dx}, dy={dy}",
                mapped.0,
                mapped.1,
                template[i].0,
                template[i].1
            );
        }
    }
}
