//! Detect mains hum that carries the electrical network frequency (ENF).
//!
//! A port of `flyover_scrub/enf.py`. The grid's frequency wanders by a few
//! hundredths of a hertz around 50 or 60 Hz, the same way at every outlet on one
//! interconnection. Hum picked up by a microphone records that wander, so a few
//! minutes of it can be matched against a grid-frequency log to date the
//! recording, and its nominal frequency says which grid it was made on.
//!
//! A line counts as ENF only if it is the grid and not a fan or an engine: at
//! least two harmonics of 50 or 60 Hz stand out, imply the same grid frequency
//! second by second, and sit within 0.2 Hz of nominal.

use crate::stft::hann;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

/// Everything is analysed at this rate: hum harmonics sit far below 4 kHz.
const ANALYSIS_RATE: f64 = 8000.0;
/// 8 s window: 0.125 Hz bins.
const WINDOW: usize = 64_000;
/// 1 s hop: one grid-frequency reading per second.
const HOP: usize = 8_000;
/// Needs at least this many windows of audio (about 32 s).
const MIN_WINDOWS: usize = 4;
const NOMINALS: [u32; 2] = [50, 60];
const HARMONICS: [u32; 4] = [1, 2, 3, 4];
/// Highest frequency any test looks at, Hz (4th harmonic of 60 plus the side band).
const TOP_HZ: f64 = 245.0;
const ON_LINE_HZ: f64 = 0.6;
const SIDE_NEAR_HZ: f64 = 1.0;
const SIDE_FAR_HZ: f64 = 3.0;
const TRACK_HZ: f64 = 0.5;
const POWER_EPS: f64 = 1e-30;
const MIN_STRONG: usize = 2;
/// Harmonics of the real grid agree to within this, Hz.
const MAX_DISAGREEMENT_HZ: f64 = 0.02;
const MAX_OFF_NOMINAL_HZ: f64 = 0.2;
/// Resampler: pass band as a share of the lower Nyquist, and zero crossings kept per side.
const RESAMPLE_PASS: f64 = 0.9;
const RESAMPLE_LOBES: f64 = 8.0;
const RATE_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug, Default)]
pub struct EnfResult {
    pub present: bool,
    pub nominal_hz: u32,
    /// Harmonics that stand out: (Hz, dB above neighbours).
    pub lines: Vec<(u32, f64)>,
    /// Median implied grid frequency.
    pub grid_hz: f64,
    /// Its standard deviation over the clip.
    pub wander_hz: f64,
}

/// Windowed-sinc resampling to the analysis rate.
fn resample(x: &[f32], sr: f64) -> Vec<f32> {
    if (sr - ANALYSIS_RATE).abs() < RATE_TOLERANCE {
        return x.to_vec();
    }
    let cutoff = RESAMPLE_PASS * 0.5 * sr.min(ANALYSIS_RATE) / sr; // cycles per input sample
    let half = RESAMPLE_LOBES / (2.0 * cutoff);
    let step = sr / ANALYSIS_RATE;
    let n_out = (x.len() as f64 / step).floor() as usize;
    let mut out = Vec::with_capacity(n_out);
    for k in 0..n_out {
        let centre = k as f64 * step;
        let first = (centre - half).ceil().max(0.0) as usize;
        let last = ((centre + half).floor() as usize).min(x.len().saturating_sub(1));
        let mut acc = 0.0f64;
        for m in first..=last {
            let tau = m as f64 - centre;
            let arg = core::f64::consts::TAU * cutoff * tau;
            let sinc = if arg.abs() < 1e-9 { 1.0 } else { arg.sin() / arg };
            let taper = 0.5 + 0.5 * (core::f64::consts::PI * tau / half).cos();
            acc += x[m] as f64 * 2.0 * cutoff * sinc * taper;
        }
        out.push(acc as f32);
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        return values[mid];
    }
    0.5 * (values[mid - 1] + values[mid])
}

/// Power spectrogram of the hum region: `p[k * frames + t]`, 0.125 Hz bins.
struct HumSpectrogram {
    p: Vec<f64>,
    frames: usize,
    df: f64,
}

impl HumSpectrogram {
    fn new(x: &[f32]) -> Self {
        let frames = (x.len() - WINDOW) / HOP + 1;
        let df = ANALYSIS_RATE / WINDOW as f64;
        let bins = (TOP_HZ / df).ceil() as usize + 1;
        let window = hann(WINDOW);
        let scale = 1.0 / window.iter().map(|&w| w as f64).sum::<f64>();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(WINDOW);
        let mut buf = vec![Complex32::new(0.0, 0.0); WINDOW];
        let mut p = vec![0.0f64; bins * frames];
        for t in 0..frames {
            let seg = &x[t * HOP..t * HOP + WINDOW];
            for i in 0..WINDOW {
                buf[i] = Complex32::new(seg[i] * window[i], 0.0);
            }
            fft.process(&mut buf);
            for k in 0..bins {
                let (re, im) = (buf[k].re as f64 * scale, buf[k].im as f64 * scale);
                p[k * frames + t] = re * re + im * im;
            }
        }
        HumSpectrogram { p, frames, df }
    }

    /// Bins strictly inside (lo, hi) Hz.
    fn bins_between(&self, lo: f64, hi: f64) -> core::ops::Range<usize> {
        let first = (lo / self.df).floor() as usize + 1;
        let last = (hi / self.df).ceil() as usize; // exclusive
        first..last.max(first)
    }

    /// How far the line at `fc` stands above its neighbours, dB.
    fn line_db(&self, fc: f64) -> f64 {
        let on = self.bins_between(fc - ON_LINE_HZ, fc + ON_LINE_HZ);
        let mut peaks: Vec<f64> = (0..self.frames)
            .map(|t| on.clone().map(|k| self.p[k * self.frames + t]).fold(0.0, f64::max))
            .collect();
        let mut side: Vec<f64> = self
            .bins_between(fc - SIDE_FAR_HZ, fc - SIDE_NEAR_HZ)
            .chain(self.bins_between(fc + SIDE_NEAR_HZ, fc + SIDE_FAR_HZ))
            .flat_map(|k| self.p[k * self.frames..(k + 1) * self.frames].iter().copied())
            .collect();
        10.0 * (median(&mut peaks) / median(&mut side)).log10()
    }

    /// Frequency of the line near `fc`, one reading per frame (parabolic peak fit).
    fn track(&self, fc: f64) -> Vec<f64> {
        let band = self.bins_between(fc - TRACK_HZ, fc + TRACK_HZ);
        let at = |k: usize, t: usize| (self.p[k * self.frames + t] + POWER_EPS).ln();
        (0..self.frames)
            .map(|t| {
                let top = band
                    .clone()
                    .max_by(|&a, &b| self.p[a * self.frames + t].total_cmp(&self.p[b * self.frames + t]))
                    .unwrap_or(band.start);
                let k = top.clamp(band.start + 1, band.end - 2);
                let (y0, y1, y2) = (at(k - 1, t), at(k, t), at(k + 1, t));
                let den = y0 - 2.0 * y1 + y2;
                let den = if den.abs() > 1e-12 { den } else { 1e-12 };
                let d = (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5);
                (k as f64 + d) * self.df
            })
            .collect()
    }
}

/// Looks for ENF in mono audio. Needs about 30 s or more.
pub fn detect_enf(audio: &[f32], sr: f64, min_db: f64) -> EnfResult {
    if sr.is_nan() || sr <= 0.0 {
        return EnfResult::default();
    }
    let x = resample(audio, sr);
    if x.len() < MIN_WINDOWS * WINDOW {
        return EnfResult::default();
    }
    let spec = HumSpectrogram::new(&x);
    let mut best = EnfResult::default();
    let mut best_total = f64::MIN;
    for nominal in NOMINALS {
        // (harmonic Hz, dB, implied grid frequency per frame)
        let strong: Vec<(u32, f64, Vec<f64>)> = HARMONICS
            .iter()
            .map(|&k| (k, spec.line_db((nominal * k) as f64)))
            .filter(|&(_, db)| db >= min_db)
            .map(|(k, db)| {
                let grid = spec.track((nominal * k) as f64).iter().map(|f| f / k as f64).collect();
                (nominal * k, db, grid)
            })
            .collect();
        if strong.len() < MIN_STRONG {
            continue;
        }
        let consensus: Vec<f64> = (0..spec.frames)
            .map(|t| median(&mut strong.iter().map(|s| s.2[t]).collect::<Vec<_>>()))
            .collect();
        let mut disagreement: Vec<f64> = strong
            .iter()
            .map(|s| median(&mut s.2.iter().zip(&consensus).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()))
            .collect();
        let grid_hz = median(&mut strong.iter().flat_map(|s| s.2.iter().copied()).collect::<Vec<_>>());
        // Harmonics that drift apart or sit off-nominal: a machine, not the grid.
        if median(&mut disagreement) > MAX_DISAGREEMENT_HZ || (grid_hz - nominal as f64).abs() > MAX_OFF_NOMINAL_HZ {
            continue;
        }
        let total: f64 = strong.iter().map(|s| s.1).sum();
        if best.present && total <= best_total {
            continue;
        }
        // The wander is read off the strongest harmonic.
        let top = strong.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|s| &s.2).unwrap_or(&consensus);
        let mean = top.iter().sum::<f64>() / top.len() as f64;
        let wander_hz = (top.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / top.len() as f64).sqrt();
        best_total = total;
        best = EnfResult {
            present: true,
            nominal_hz: nominal,
            lines: strong.iter().map(|s| (s.0, s.1)).collect(),
            grid_hz,
            wander_hz,
        };
    }
    best
}
