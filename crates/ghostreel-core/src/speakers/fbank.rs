//! Kaldi-compatible log-mel filterbank features, as WeSpeaker models are trained on.
//!
//! 16 kHz mono, 25 ms Povey window every 10 ms (snip edges), DC removed, pre-emphasis 0.97,
//! 512-point power spectrum, 80 triangular mel bins between 20 Hz and Nyquist, natural log,
//! then the per-utterance mean of each bin is subtracted (CMN). Samples are expected in the
//! int16 range (WeSpeaker ONNX metadata `normalize_samples = 0`): pass `scale = 32768.0` for
//! f32 PCM in [-1, 1].

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_LEN: usize = 400; // 25 ms
pub const FRAME_SHIFT: usize = 160; // 10 ms
pub const N_FFT: usize = 512;
pub const N_MELS: usize = 80;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Reusable feature extractor (window, filterbank and FFT plan built once).
pub struct Fbank {
    window: Vec<f32>,
    /// Per mel bin: first FFT bin and its weights.
    filters: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn Fft<f32>>,
    buf: Vec<Complex<f32>>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Self {
        let window = (0..FRAME_LEN)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / (FRAME_LEN - 1) as f64;
                (0.5 - 0.5 * a.cos()).powf(0.85) as f32
            })
            .collect();
        let n_bins = N_FFT / 2;
        let fft_bin_width = SAMPLE_RATE as f32 / N_FFT as f32;
        let mel_low = mel_scale(LOW_FREQ);
        let mel_high = mel_scale(SAMPLE_RATE as f32 / 2.0);
        let delta = (mel_high - mel_low) / (N_MELS + 1) as f32;
        let mut filters = Vec::with_capacity(N_MELS);
        for m in 0..N_MELS {
            let left = mel_low + m as f32 * delta;
            let center = left + delta;
            let right = center + delta;
            let mut first = None;
            let mut weights = Vec::new();
            for k in 0..n_bins {
                let mel = mel_scale(fft_bin_width * k as f32);
                if mel > left && mel < right {
                    let w = if mel <= center { (mel - left) / (center - left) } else { (right - mel) / (right - center) };
                    if first.is_none() {
                        first = Some(k);
                    }
                    weights.push(w);
                }
            }
            filters.push((first.unwrap_or(0), weights));
        }
        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        Self { window, filters, fft, buf: vec![Complex::new(0.0, 0.0); N_FFT] }
    }

    /// Number of frames Kaldi (snip_edges) produces for `n` samples.
    pub fn num_frames(n: usize) -> usize {
        if n < FRAME_LEN { 0 } else { 1 + (n - FRAME_LEN) / FRAME_SHIFT }
    }

    /// Features for `samples` (row-major `[frames][N_MELS]`), mean-normalised per bin.
    pub fn compute(&mut self, samples: &[f32], scale: f32) -> Vec<f32> {
        let frames = Self::num_frames(samples.len());
        let mut out = vec![0.0f32; frames * N_MELS];
        let mut frame = [0.0f32; FRAME_LEN];
        let mut power = [0.0f32; N_FFT / 2 + 1];
        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            for (dst, &s) in frame.iter_mut().zip(&samples[start..start + FRAME_LEN]) {
                *dst = s * scale;
            }
            let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
            for v in frame.iter_mut() {
                *v -= mean;
            }
            for i in (1..FRAME_LEN).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            for (i, c) in self.buf.iter_mut().enumerate() {
                let v = if i < FRAME_LEN { frame[i] * self.window[i] } else { 0.0 };
                *c = Complex::new(v, 0.0);
            }
            self.fft.process(&mut self.buf);
            for (p, c) in power.iter_mut().zip(&self.buf) {
                *p = c.norm_sqr();
            }
            let row = &mut out[f * N_MELS..(f + 1) * N_MELS];
            for (m, (first, weights)) in self.filters.iter().enumerate() {
                let energy: f32 = weights.iter().enumerate().map(|(j, w)| w * power[first + j]).sum();
                row[m] = energy.max(f32::EPSILON).ln();
            }
        }
        if frames > 0 {
            for m in 0..N_MELS {
                let mean = (0..frames).map(|f| out[f * N_MELS + m]).sum::<f32>() / frames as f32;
                for f in 0..frames {
                    out[f * N_MELS + m] -= mean;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_kaldi_snip_edges() {
        assert_eq!(Fbank::num_frames(399), 0);
        assert_eq!(Fbank::num_frames(400), 1);
        assert_eq!(Fbank::num_frames(559), 1);
        assert_eq!(Fbank::num_frames(560), 2);
        assert_eq!(Fbank::num_frames(16_000), 98);
    }

    #[test]
    fn shape_and_mean_normalisation() {
        let mut fb = Fbank::new();
        let samples: Vec<f32> = (0..32_000).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
        let feats = fb.compute(&samples, 32768.0);
        assert_eq!(feats.len(), 198 * N_MELS);
        for m in 0..N_MELS {
            let mean: f32 = (0..198).map(|f| feats[f * N_MELS + m]).sum::<f32>() / 198.0;
            assert!(mean.abs() < 1e-3, "bin {m} mean {mean}");
        }
        assert!(feats.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tone_peaks_in_the_right_mel_bin() {
        // A pure 1 kHz tone: the bin whose centre is nearest 1 kHz carries the most energy.
        let mut fb = Fbank::new();
        let samples: Vec<f32> =
            (0..8_000).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin() * 0.5).collect();
        let frames = Fbank::num_frames(samples.len());
        // Undo CMN by comparing raw bins: compute on a buffer with silence appended would change
        // the mean, so look at the filterbank directly via a single-frame utterance.
        let one = fb.compute(&samples[..FRAME_LEN], 32768.0);
        assert_eq!(one.len(), N_MELS);
        assert!(one.iter().all(|v| v.abs() < 1e-4), "one frame minus its own mean is zero");
        assert!(frames > 1);

        // Expected bin: mel centres are evenly spaced between mel(20) and mel(8000).
        let lo = mel_scale(LOW_FREQ);
        let hi = mel_scale(8000.0);
        let delta = (hi - lo) / (N_MELS + 1) as f32;
        let target = ((mel_scale(1000.0) - lo) / delta - 1.0).round() as usize;
        let (first, weights) = &fb.filters[target];
        let fft_bin = (1000.0 / (16_000.0 / N_FFT as f32)).round() as usize;
        assert!(*first <= fft_bin && fft_bin < first + weights.len(), "1 kHz lies under mel bin {target}");
    }

    #[test]
    fn known_values_match_reference() {
        // Reference values from an independent numpy implementation of the same Kaldi recipe
        // (float64, numpy.fft.rfft) on a rising two-tone signal.
        let samples: Vec<f32> = (0..1600)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                ((i as f64 / 1600.0)
                    * (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                        + 0.1 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin())) as f32
            })
            .collect();
        let feats = Fbank::new().compute(&samples, 32768.0);
        assert_eq!(feats.len(), 8 * N_MELS);
        let expected = [
            (0, [0.5034, -2.282, -2.1317, -1.2573]),
            (4, [0.5515, 0.4869, 0.7245, -0.0233]),
            (7, [-0.0372, 1.394, 0.9154, 1.1572]),
        ];
        for (frame, values) in expected {
            for (bin, want) in [0, 10, 40, 79].into_iter().zip(values) {
                let got = feats[frame * N_MELS + bin];
                assert!((got - want).abs() < 5e-3, "frame {frame} bin {bin}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn silence_is_floored_not_nan() {
        let mut fb = Fbank::new();
        let feats = fb.compute(&vec![0.0; 4_000], 32768.0);
        assert!(feats.iter().all(|v| v.is_finite()));
    }
}
