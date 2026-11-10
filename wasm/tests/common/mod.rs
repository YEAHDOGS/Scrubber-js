//! Test-only ports of `flyover_scrub/physics.py` and `flyover_scrub/attack.py`.
//!
//! The attack scorer is here so the defence can be measured, as upstream does.
//! It is not part of the library and is never compiled into the WebAssembly module.
#![allow(dead_code)]

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use scrubber_core::rng::Rng;
use scrubber_core::stft::hann;
use std::f64::consts::TAU;

pub const SPEED_OF_SOUND: f64 = 343.0;
const TRACK_DT: f64 = 0.05;

/// A sampled flight path: local east/north/up metres against seconds.
pub struct Track {
    pub t: Vec<f64>,
    pub pos: Vec<[f64; 3]>,
}

#[derive(Clone, Copy, Default)]
pub struct Turn {
    pub rate_deg_s: f64,
    pub start: f64,
    pub duration: f64,
}

pub fn make_track(t0: f64, duration: f64, start_xy: (f64, f64), altitude: f64, speed: f64, heading_deg: f64, turn: Turn) -> Track {
    let n = ((duration + TRACK_DT) / TRACK_DT).ceil() as usize;
    let mut t = Vec::with_capacity(n);
    let mut pos = Vec::with_capacity(n);
    let (mut x, mut y) = start_xy;
    let mut prev = (0.0, 0.0);
    for i in 0..n {
        let tau = i as f64 * TRACK_DT;
        let heading = (heading_deg + turn.rate_deg_s * (tau - turn.start).clamp(0.0, turn.duration)).to_radians();
        let v = (speed * heading.cos(), speed * heading.sin());
        if i > 0 {
            x += 0.5 * (v.0 + prev.0) * TRACK_DT;
            y += 0.5 * (v.1 + prev.1) * TRACK_DT;
        }
        prev = v;
        t.push(t0 + tau);
        pos.push([x, y, altitude]);
    }
    Track { t, pos }
}

/// `numpy.gradient` for evenly spaced samples.
fn gradient(y: &[f64], dx: f64) -> Vec<f64> {
    let n = y.len();
    (0..n)
        .map(|i| match i {
            0 => (y[1] - y[0]) / dx,
            i if i == n - 1 => (y[n - 1] - y[n - 2]) / dx,
            i => (y[i + 1] - y[i - 1]) / (2.0 * dx),
        })
        .collect()
}

/// `numpy.interp` for one point inside the range of `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let i = xp.partition_point(|&v| v <= x).clamp(1, xp.len() - 1);
    let w = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
    fp[i - 1] + w * (fp[i] - fp[i - 1])
}

/// Per track sample: reception time, Doppler factor and range for one observer.
pub fn emission_geometry(track: &Track, observer: [f64; 3]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r: Vec<f64> = track
        .pos
        .iter()
        .map(|p| (0..3).map(|k| (p[k] - observer[k]).powi(2)).sum::<f64>().sqrt())
        .collect();
    let t_rec = track.t.iter().zip(&r).map(|(t, r)| t + r / SPEED_OF_SOUND).collect();
    let g = gradient(&r, TRACK_DT).iter().map(|v| 1.0 / (1.0 + v / SPEED_OF_SOUND)).collect();
    (t_rec, g, r)
}

/// What one observer hears from a track: reception times against Doppler factor, range and emission time.
pub struct Heard {
    t_rec: Vec<f64>,
    g: Vec<f64>,
    r: Vec<f64>,
    t_emit: Vec<f64>,
}

impl Heard {
    pub fn new(track: &Track, observer: [f64; 3]) -> Self {
        let (t_rec, g, r) = emission_geometry(track, observer);
        Heard { t_rec, g, r, t_emit: track.t.clone() }
    }

    /// (Doppler factor, range, emission time) at reception time `t`, or None outside the audible span.
    pub fn at(&self, t: f64) -> Option<(f64, f64, f64)> {
        if t < self.t_rec[0] || t > self.t_rec[self.t_rec.len() - 1] {
            return None;
        }
        Some((interp(t, &self.t_rec, &self.g), interp(t, &self.t_rec, &self.r), interp(t, &self.t_rec, &self.t_emit)))
    }
}

/// Second-order section (RBJ cookbook), used to shape the engine's broadband noise.
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biquad {
    fn new(highpass: bool, freq: f64, sr: f64) -> Self {
        let w = TAU * freq / sr;
        let alpha = w.sin() / std::f64::consts::SQRT_2;
        let c = w.cos();
        let a0 = 1.0 + alpha;
        let b = if highpass {
            [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0]
        } else {
            [(1.0 - c) / 2.0 / a0, (1.0 - c) / a0, (1.0 - c) / 2.0 / a0]
        };
        Biquad { b, a: [-2.0 * c / a0, (1.0 - alpha) / a0], z: [0.0; 2] }
    }

    fn step(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// Physically modelled propeller flyover as heard at `observer`.
///
/// The received waveform is `sin(2*pi*k*f0*t_e(t_r))`: writing phase in emission
/// time gives exact Doppler (and retardation) for free. Amplitude falls as 1/r.
pub fn synthesize_flyover(track: &Track, observer: [f64; 3], t_start: f64, n: usize, sr: f64, f0: f64, broadband: f64, rng: &mut Rng) -> Vec<f64> {
    const HARMONICS: [f64; 5] = [1.0, 0.7, 0.45, 0.3, 0.2];
    let heard = Heard::new(track, observer);
    let samples: Vec<Option<(f64, f64, f64)>> = (0..n).map(|i| heard.at(t_start + i as f64 / sr)).collect();
    let te: Vec<f64> = samples.iter().map(|s| s.map_or(0.0, |v| v.2)).collect();
    let mut amp: Vec<f64> = samples.iter().map(|s| s.map_or(0.0, |v| 1.0 / v.1.max(1.0))).collect();
    let peak = amp.iter().cloned().fold(0.0, f64::max);
    if peak > 0.0 {
        amp.iter_mut().for_each(|a| *a /= peak);
    }
    // Slow blade-rate wander so it is not a perfect synthetic tone.
    let wander_phase = rng.uniform() * 6.28;
    let mut phase = Vec::with_capacity(n);
    let mut acc = 0.0;
    for (d, te) in gradient(&te, 1.0).iter().zip(&te) {
        acc += d * (1.0 + 0.002 * (TAU * 0.07 * te + wander_phase).sin());
        phase.push(TAU * f0 * acc);
    }
    let mut sig = vec![0.0f64; n];
    for (k, a) in HARMONICS.iter().enumerate() {
        let offset = rng.uniform() * 6.28;
        sig.iter_mut().zip(&phase).for_each(|(s, p)| *s += a * ((k + 1) as f64 * p + offset).sin());
    }
    if broadband > 0.0 {
        let mut hp = Biquad::new(true, 60.0, sr);
        let mut lp = Biquad::new(false, 900.0, sr);
        sig.iter_mut().for_each(|s| *s += broadband * 3.0 * lp.step(hp.step(rng.normal())));
    }
    sig.iter().zip(&amp).map(|(s, a)| s * a).collect()
}

/// Whitened log-power spectrogram: `s[bin * frames + frame]`.
pub struct Spectrogram {
    pub s: Vec<f32>,
    pub bins: usize,
    pub frames: usize,
    pub df: f64,
    /// Seconds from the start of the audio, one per frame.
    pub times: Vec<f64>,
}

impl Spectrogram {
    pub fn at(&self, bin: usize, frame: usize) -> f32 {
        self.s[bin * self.frames + frame]
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        return values[mid];
    }
    0.5 * (values[mid - 1] + values[mid])
}

/// Long-window spectrogram, whitened per bin so faint lines stand out.
pub fn whitened_spectrogram(audio: &[f32], sr: f64, window_s: f64) -> Spectrogram {
    const HOP_S: f64 = 0.25;
    const FMAX: f64 = 1200.0;
    const FREQ_MEDIAN: isize = 9;
    let nper = (window_s * sr).round() as usize;
    let hop = (HOP_S * sr).round() as usize;
    let frames = (audio.len() - nper) / hop + 1;
    let df = sr / nper as f64;
    let bins = (FMAX / df).floor() as usize + 1;
    let window = hann(nper);
    let scale = 1.0 / window.iter().map(|&w| w as f64).sum::<f64>();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(nper);
    let mut buf = vec![Complex32::new(0.0, 0.0); nper];
    let mut logp = vec![0.0f32; bins * frames];
    for j in 0..frames {
        for i in 0..nper {
            buf[i] = Complex32::new(audio[j * hop + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..bins {
            let (re, im) = (buf[k].re as f64 * scale, buf[k].im as f64 * scale);
            logp[k * frames + j] = (re * re + im * im + 1e-20).ln() as f32;
        }
    }
    // Whiten: divide each bin by its median over time, then by a local median over frequency.
    for k in 0..bins {
        let row = &mut logp[k * frames..(k + 1) * frames];
        let m = median(&mut row.to_vec());
        row.iter_mut().for_each(|v| *v -= m);
    }
    let mut s = vec![0.0f32; bins * frames];
    let half = FREQ_MEDIAN / 2;
    for j in 0..frames {
        for k in 0..bins as isize {
            let mut win: Vec<f32> = (-half..=half).map(|d| logp[(k + d).clamp(0, bins as isize - 1) as usize * frames + j]).collect();
            s[k as usize * frames + j] = (logp[k as usize * frames + j] - median(&mut win)).clamp(-3.0, 8.0);
        }
    }
    let times = (0..frames).map(|j| (j * hop) as f64 / sr + window_s / 2.0).collect();
    Spectrogram { s, bins, frames, df, times }
}

pub struct Search {
    pub f0_range: (f64, f64),
    pub offsets: (f64, f64),
}

/// Scores every candidate location: energy under the Doppler curve the track
/// would produce there, maximised over the unknown fundamental and clock offset.
pub fn score_candidates(spec: &Spectrogram, track: &Track, audio_t0: f64, candidates: &[[f64; 3]], search: &Search) -> Vec<f64> {
    const STEP: f64 = 0.25;
    const HARMONICS: usize = 3;
    const MAX_RANGE: f64 = 12_000.0;
    const NEAR_RANGE: f64 = 300.0;
    let steps = |lo: f64, hi: f64| (0..=((hi - lo) / STEP + 1e-6).floor() as usize).map(move |i| lo + i as f64 * STEP);
    let f0s: Vec<f64> = steps(search.f0_range.0, search.f0_range.1).collect();
    let top = spec.bins as f64 - 1.0;
    candidates
        .iter()
        .map(|&observer| {
            let heard = Heard::new(track, observer);
            let mut best = f64::NEG_INFINITY;
            let mut any = false;
            for offset in steps(search.offsets.0, search.offsets.1) {
                // (frame, Doppler factor, weight) for the frames this track is audible in.
                let mut line: Vec<(usize, f64, f64)> = spec
                    .times
                    .iter()
                    .enumerate()
                    .filter_map(|(j, t)| heard.at(audio_t0 + offset + t).map(|(g, r, _)| (j, g, r)))
                    .filter(|&(_, _, r)| r < MAX_RANGE)
                    .map(|(j, g, r)| (j, g, 1.0 / r.max(NEAR_RANGE)))
                    .collect();
                if line.is_empty() {
                    continue;
                }
                any = true;
                let total: f64 = line.iter().map(|l| l.2).sum::<f64>() + 1e-12;
                line.iter_mut().for_each(|l| l.2 /= total);
                for &f0 in &f0s {
                    let mut score = 0.0;
                    for k in 1..=HARMONICS {
                        for &(j, g, w) in &line {
                            let fb = f0 * k as f64 * g / spec.df;
                            if fb < top {
                                score += spec.at(fb.round_ties_even() as usize, j) as f64 * w;
                            }
                        }
                    }
                    best = best.max(score / HARMONICS as f64);
                }
            }
            if any { best } else { 0.0 }
        })
        .collect()
}

/// A square grid of ground-level candidates.
pub fn grid(center: (f64, f64), half_width: f64, step: f64) -> Vec<[f64; 3]> {
    let n = (2.0 * half_width / step + 1e-6).floor() as usize;
    let axis = |c: f64| (0..=n).map(move |i| c - half_width + i as f64 * step);
    axis(center.1).flat_map(|y| axis(center.0).map(move |x| [x, y, 0.0])).collect()
}

pub fn std_dev(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / x.len() as f64).sqrt()
}

pub fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}
