//! Short-time Fourier transform with the same conventions as `scipy.signal.stft`
//! at its defaults: periodic Hann window, half a window of zeros at each end,
//! zero padding up to a whole number of hops, "spectrum" scaling.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Samples whose summed squared window is below this get no correction
/// (scipy uses the same guard when it inverts).
const MIN_OVERLAP_NORM: f32 = 1e-10;

pub struct Layout {
    pub nper: usize,
    pub hop: usize,
    pub frames: usize,
    pub bins: usize,
    /// Zeros added before the first sample (half a window).
    pub lead: usize,
}

impl Layout {
    pub fn new(len: usize, nper: usize, hop: usize) -> Self {
        let lead = nper / 2;
        let ext = len + 2 * lead;
        let rem = (ext - nper) % hop;
        let padded = if rem == 0 { ext } else { ext + hop - rem };
        Layout { nper, hop, frames: (padded - nper) / hop + 1, bins: nper / 2 + 1, lead }
    }
}

/// Periodic Hann window, as `scipy.signal.get_window("hann", n)`.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (core::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

pub struct Stft {
    pub layout: Layout,
    window: Vec<f32>,
    /// 1 / sum(window): scipy's "spectrum" scaling.
    scale: f32,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    buf: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Stft {
    pub fn new(len: usize, nper: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(nper);
        let inverse = planner.plan_fft_inverse(nper);
        let window = hann(nper);
        let sum: f64 = window.iter().map(|&w| w as f64).sum();
        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        Stft {
            layout: Layout::new(len, nper, hop),
            window,
            scale: (1.0 / sum) as f32,
            forward,
            inverse,
            buf: vec![Complex32::new(0.0, 0.0); nper],
            scratch: vec![Complex32::new(0.0, 0.0); scratch_len],
        }
    }

    /// How much louder independent noise must be written into the frames to come
    /// out of the overlap-add at the level it was written at.
    ///
    /// Frames of a real signal overlap coherently. Frames of fresh noise do not:
    /// each sample is a weighted mean of several unrelated values, which lowers
    /// its power. For a Hann window at a quarter-window hop the loss is 6 dB,
    /// so this returns 2.
    pub fn incoherent_makeup(&self) -> f32 {
        let Layout { nper, hop, .. } = self.layout;
        let squared = |i: usize| (self.window[i] as f64).powi(2);
        let mean_square = (0..nper).map(squared).sum::<f64>() / nper as f64;
        // 1 / (summed squared window) at each position within one hop.
        let mean_inverse_overlap = (0..hop)
            .map(|i| 1.0 / (i..nper).step_by(hop).map(squared).sum::<f64>().max(MIN_OVERLAP_NORM as f64))
            .sum::<f64>()
            / hop as f64;
        (1.0 / (mean_square * mean_inverse_overlap)).sqrt() as f32
    }

    /// Transforms frame `j` of `x` and hands the positive-frequency bins to `sink`.
    pub fn frame(&mut self, x: &[f32], j: usize, mut sink: impl FnMut(&[Complex32])) {
        let start = (j * self.layout.hop) as isize - self.layout.lead as isize;
        for (i, slot) in self.buf.iter_mut().enumerate() {
            let k = start + i as isize;
            let v = if k >= 0 && (k as usize) < x.len() { x[k as usize] } else { 0.0 };
            *slot = Complex32::new(v * self.window[i] * self.scale, 0.0);
        }
        self.forward.process_with_scratch(&mut self.buf, &mut self.scratch);
        sink(&self.buf[..self.layout.bins]);
    }

    /// Adds the inverse transform of a sparse spectrum to `x`, in place.
    ///
    /// `fill(j, bins)` writes frame `j`'s positive-frequency bins into a zeroed
    /// slice. Overlap-add runs through a sliding buffer one window long, so no
    /// second copy of the signal is needed.
    pub fn add_inverse(&mut self, x: &mut [f32], mut fill: impl FnMut(usize, &mut [Complex32])) {
        let Layout { nper, hop, frames, bins, lead } = self.layout;
        let unscale = 1.0 / (self.scale * nper as f32);
        let mut acc = vec![0.0f32; nper];
        let mut norm = vec![0.0f32; nper];
        for j in 0..frames {
            self.buf.iter_mut().for_each(|c| *c = Complex32::new(0.0, 0.0));
            fill(j, &mut self.buf[..bins]);
            // Mirror into the negative frequencies so the result is real.
            for k in 1..nper - bins + 1 {
                self.buf[nper - k] = self.buf[k].conj();
            }
            self.inverse.process_with_scratch(&mut self.buf, &mut self.scratch);
            for i in 0..nper {
                let w = self.window[i];
                acc[i] += self.buf[i].re * unscale * w;
                norm[i] += w * w;
            }
            // The first `hop` samples of the buffer are complete: emit and slide.
            let done = if j + 1 == frames { nper } else { hop.min(nper) };
            let base = (j * hop) as isize - lead as isize;
            for i in 0..done {
                let k = base + i as isize;
                if k >= 0 && (k as usize) < x.len() && norm[i] > MIN_OVERLAP_NORM {
                    x[k as usize] += acc[i] / norm[i];
                }
            }
            acc.copy_within(hop.min(nper).., 0);
            norm.copy_within(hop.min(nper).., 0);
            let tail = nper - hop.min(nper);
            acc[tail..].iter_mut().for_each(|v| *v = 0.0);
            norm[tail..].iter_mut().for_each(|v| *v = 0.0);
        }
    }
}
