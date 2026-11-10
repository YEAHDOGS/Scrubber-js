//! Remove aircraft Doppler signatures from audio while keeping the foreground.
//!
//! A port of `flyover_scrub/scrub.py`. Why not just a filter: the attack is a
//! matched filter. It adds up energy along a long, precisely predicted curve, so
//! it recovers lines far below the noise floor and anything merely attenuated.
//! A defence has to leave the bins with no information about the curve.
//!
//! Per STFT bin inside the protected band:
//!   * background bins (near the local noise floor) are re-synthesised: replaced
//!     with fresh random-phase noise at the local floor level;
//!   * foreground bins that form long, slowly gliding narrowband ridges
//!     (aircraft, drones, distant traffic, HVAC, hum) are replaced the same way;
//!   * other foreground (speech, music, transients) passes through untouched.
//!
//! Only the protected band is ever held in memory. The output is the input plus
//! the inverse transform of (replacement - original) for the replaced bins,
//! which is the same signal scipy's `istft` would return, without a second copy
//! of the full spectrogram.

use crate::filters::{box_nearest, box_reflect, median_nearest, nearest};
use crate::rng::Rng;
use crate::stft::Stft;
use rustfft::num_complex::Complex32;

const MAG_EPS: f32 = 1e-12;
const MIN_BACKGROUND_WEIGHT: f32 = 1e-3;
const MIN_NOISE_POWER: f64 = 1e-20;
const MIN_REPLACED_POWER: f64 = 1e-30;
const SHORTEST_WINDOW: usize = 4;
/// Progress is reported once every this many frames.
const PROGRESS_EVERY: usize = 64;
/// Share of the progress bar reached after each of the first two stages.
const PROGRESS_ANALYSED: f32 = 0.4;
const PROGRESS_PLANNED: f32 = 0.6;
/// Marks a pixel the component search has already visited.
const SEEN: u8 = 2;

#[derive(Clone, Debug)]
pub struct ScrubConfig {
    /// Protected band, Hz.
    pub band: (f64, f64),
    /// ~4 Hz bins: long enough to separate slow tones from speech.
    pub window_s: f64,
    pub hop_frac: f64,
    /// Width of the median across frequency for the floor estimate.
    pub floor_freq_hz: f64,
    /// Smoothing of the floor over time.
    pub floor_time_s: f64,
    /// Bins this far above the floor are kept; 8 leaks medium-loud lines.
    pub foreground_db: f64,
    /// Low on purpose: false ridges in pure noise cost nothing (it is re-synthesised anyway).
    pub ridge_db: f64,
    /// Ridges at least this long are treated as signatures.
    pub ridge_min_s: f64,
    /// Aircraft glide slowly; speech pitch moves fast.
    pub ridge_max_slope_hz_s: f64,
    pub ridge_pad_hz: f64,
    /// Steady tones carry no Doppler information, but mains hum enables ENF
    /// forensics. Removing them is the safer default.
    pub keep_stationary_tones: bool,
    pub stationary_max_drift_hz: f64,
    pub seed: u64,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        ScrubConfig {
            band: (20.0, 2000.0),
            window_s: 0.25,
            hop_frac: 0.25,
            floor_freq_hz: 120.0,
            floor_time_s: 3.0,
            foreground_db: 12.0,
            ridge_db: 4.0,
            ridge_min_s: 1.5,
            ridge_max_slope_hz_s: 40.0,
            ridge_pad_hz: 8.0,
            keep_stationary_tones: false,
            stationary_max_drift_hz: 1.5,
            seed: 0,
        }
    }
}

/// One long narrowband tone that was found (and, unless checking only, removed).
#[derive(Clone, Debug, Default)]
pub struct Ridge {
    pub t_start: f64,
    pub t_end: f64,
    pub f_low: f64,
    pub f_high: f64,
    /// Pitch at its first and last frame: a flyover falls, a hum stays put.
    pub f_first: f64,
    pub f_last: f64,
    /// Strongest point, dB above the local floor.
    pub peak_db: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ScrubReport {
    /// Share of in-band bins that were replaced.
    pub replaced_fraction: f64,
    /// Share of in-band bins that sat on a ridge.
    pub ridge_fraction: f64,
    pub ridges: Vec<Ridge>,
}

/// Python's `round`: halves go to the even neighbour.
fn round_even(v: f64) -> usize {
    v.round_ties_even().max(0.0) as usize
}

/// The protected band on the STFT grid, and what has to be stored to analyse it.
struct Grid {
    frames: usize,
    bins: usize,
    df: f64,
    dt: f64,
    /// First and last in-band bin.
    lo: usize,
    hi: usize,
    floor_bins: usize,
    floor_frames: usize,
}

impl Grid {
    fn band_rows(&self) -> usize {
        self.hi - self.lo + 1
    }
}

struct RidgeMask {
    mask: Vec<u8>,
    ridges: Vec<Ridge>,
}

/// Local maxima across frequency that stand above the floor, with one-frame
/// dropouts bridged (a binary closing along time, OR-ed back in).
fn ridge_candidates(ratio_db: &[f32], rows: usize, frames: usize, ridge_db: f32) -> Vec<u8> {
    let at = |b: usize, t: usize| ratio_db[b * frames + t];
    let mut peak = vec![0u8; rows * frames];
    for b in 0..rows {
        for t in 0..frames {
            let v = at(b, t);
            let below = b == 0 || v >= at(b - 1, t);
            let above = b + 1 == rows || v >= at(b + 1, t);
            peak[b * frames + t] = (v > ridge_db && below && above) as u8;
        }
    }
    let mut bridged = peak.clone();
    let last = frames as isize - 1;
    for b in 0..rows {
        let row = &peak[b * frames..(b + 1) * frames];
        let dilated = |t: isize| (t - 1..=t + 1).any(|k| k >= 0 && k <= last && row[k as usize] != 0);
        for t in 1..last {
            if dilated(t - 1) && dilated(t) && dilated(t + 1) {
                bridged[b * frames + t as usize] = 1;
            }
        }
    }
    bridged
}

/// Widens a mask by one frame either side and `pad` rows up and down.
fn dilate(mask: &[u8], rows: usize, frames: usize, pad: isize) -> Vec<u8> {
    let mut wide = vec![0u8; rows * frames];
    for b in 0..rows {
        let row = &mask[b * frames..(b + 1) * frames];
        let out = &mut wide[b * frames..(b + 1) * frames];
        for t in 0..frames {
            out[t] = row[t.saturating_sub(1)] | row[t] | row[(t + 1).min(frames - 1)];
        }
    }
    let mut out = vec![0u8; rows * frames];
    for b in 0..rows as isize {
        let dst = &mut out[b as usize * frames..(b as usize + 1) * frames];
        for nb in (b - pad).max(0)..=(b + pad).min(rows as isize - 1) {
            let src = &wide[nb as usize * frames..(nb as usize + 1) * frames];
            dst.iter_mut().zip(src).for_each(|(o, &s)| *o |= s);
        }
    }
    out
}

/// Long narrowband ridges with slow frequency drift, on the in-band grid.
fn find_ridges(ratio_db: &[f32], g: &Grid, cfg: &ScrubConfig) -> RidgeMask {
    let (rows, frames) = (g.band_rows(), g.frames);
    let mut bridged = ridge_candidates(ratio_db, rows, frames, cfg.ridge_db as f32);

    // Connected components: a couple of bins of drift per frame are allowed.
    let max_bins = ((cfg.ridge_max_slope_hz_s * g.dt / g.df).ceil() as isize).max(1);
    let min_frames = (cfg.ridge_min_s / g.dt).ceil() as usize;
    let mut mask = vec![0u8; rows * frames];
    let mut ridges = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let mut pixels: Vec<u32> = Vec::new();
    for seed in 0..rows * frames {
        if bridged[seed] != 1 {
            continue;
        }
        pixels.clear();
        stack.push(seed as u32);
        bridged[seed] = SEEN;
        let (mut b_min, mut b_max, mut t_min, mut t_max) = (rows, 0usize, frames, 0usize);
        let mut peak_db = f32::MIN;
        while let Some(p) = stack.pop() {
            pixels.push(p);
            let (b, t) = (p as usize / frames, p as usize % frames);
            b_min = b_min.min(b);
            b_max = b_max.max(b);
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            peak_db = peak_db.max(ratio_db[p as usize]);
            for nb in (b as isize - max_bins).max(0)..=(b as isize + max_bins).min(rows as isize - 1) {
                for nt in t.saturating_sub(1)..=(t + 1).min(frames - 1) {
                    let q = nb as usize * frames + nt;
                    if bridged[q] != 1 {
                        continue;
                    }
                    bridged[q] = SEEN;
                    stack.push(q as u32);
                }
            }
        }
        let t_extent = t_max - t_min + 1;
        if t_extent < min_frames {
            continue;
        }
        // Reject components that are really broad (sustained wideband noise).
        let f_extent = (b_max - b_min + 1) as f64 * g.df;
        if f_extent > cfg.ridge_max_slope_hz_s * t_extent as f64 * g.dt + 6.0 * g.df {
            continue;
        }
        if cfg.keep_stationary_tones && f_extent <= cfg.stationary_max_drift_hz + 2.0 * g.df {
            continue;
        }
        let mean_row_at = |frame: usize| -> f64 {
            let (sum, n) = pixels
                .iter()
                .filter(|&&p| p as usize % frames == frame)
                .fold((0.0f64, 0u32), |(s, n), &p| (s + (p as usize / frames) as f64, n + 1));
            sum / n.max(1) as f64
        };
        ridges.push(Ridge {
            t_start: t_min as f64 * g.dt,
            t_end: t_max as f64 * g.dt,
            f_low: (g.lo + b_min) as f64 * g.df,
            f_high: (g.lo + b_max) as f64 * g.df,
            f_first: (g.lo as f64 + mean_row_at(t_min)) * g.df,
            f_last: (g.lo as f64 + mean_row_at(t_max)) * g.df,
            peak_db: peak_db as f64,
        });
        for &p in &pixels {
            mask[p as usize] = 1;
        }
    }
    let pad = round_even(cfg.ridge_pad_hz / g.df).max(1) as isize;
    RidgeMask { mask: dilate(&mask, rows, frames, pad), ridges }
}

/// Mean over a `floor_bins` x `floor_frames` box, edges held. `src` covers the
/// in-band rows only; every other row of the full array counts as zero.
fn box_in_band(src: &[f32], g: &Grid) -> Vec<f32> {
    let (rows, frames) = (g.band_rows(), g.frames);
    let mut smooth_t = vec![0.0f32; rows * frames];
    for b in 0..rows {
        box_nearest(&src[b * frames..(b + 1) * frames], g.floor_frames, &mut smooth_t[b * frames..(b + 1) * frames]);
    }
    let half = (g.floor_bins / 2) as isize;
    let scale = 1.0 / g.floor_bins as f32;
    let mut out = vec![0.0f32; rows * frames];
    for b in 0..rows {
        let row = &mut out[b * frames..(b + 1) * frames];
        for k in -half..=half {
            let full = nearest((g.lo + b) as isize + k, g.bins);
            if full < g.lo || full > g.hi {
                continue;
            }
            let src_row = &smooth_t[(full - g.lo) * frames..(full - g.lo + 1) * frames];
            row.iter_mut().zip(src_row).for_each(|(o, &s)| *o += s);
        }
        row.iter_mut().for_each(|v| *v *= scale);
    }
    out
}

/// One value per in-band row, smoothed across the full frequency axis (zero out of band).
fn smooth_rows(per_row: &[f64], g: &Grid) -> Vec<f64> {
    let mut full = vec![0.0f64; g.bins];
    full[g.lo..=g.hi].copy_from_slice(per_row);
    box_reflect(&full, g.floor_bins)[g.lo..=g.hi].to_vec()
}

/// Scrubs mono audio in place and reports what was found.
///
/// With `check_only` the audio is left alone and only the report is filled in.
/// `progress` is called with a fraction from 0 to 1.
pub fn scrub(x: &mut [f32], sr: f64, cfg: &ScrubConfig, check_only: bool, progress: &mut dyn FnMut(f32)) -> ScrubReport {
    if x.is_empty() || sr.is_nan() || sr <= 0.0 {
        return ScrubReport::default();
    }
    let mut nper = round_even(cfg.window_s * sr).max(SHORTEST_WINDOW);
    nper += nper % 2;
    let hop = round_even(nper as f64 * cfg.hop_frac).clamp(1, nper);
    let mut stft = Stft::new(x.len(), nper, hop);
    let (frames, bins) = (stft.layout.frames, stft.layout.bins);
    let df = sr / nper as f64;
    let dt = hop as f64 / sr;

    let lo = (0..bins).find(|&k| k as f64 * df >= cfg.band.0);
    let hi = (0..bins).rev().find(|&k| k as f64 * df <= cfg.band.1);
    let (lo, hi) = match (lo, hi) {
        (Some(lo), Some(hi)) if lo <= hi => (lo, hi),
        _ => return ScrubReport::default(),
    };
    let floor_bins = round_even(cfg.floor_freq_hz / df).max(3) | 1;
    let floor_frames = round_even(cfg.floor_time_s / dt).max(1) | 1;
    let g = Grid { frames, bins, df, dt, lo, hi, floor_bins, floor_frames };
    // Stored block: the band plus half a floor window either side.
    let row0 = lo.saturating_sub(floor_bins / 2);
    let stored = (hi + floor_bins / 2).min(bins - 1) - row0 + 1;
    let rows = g.band_rows();
    let band0 = lo - row0;
    let cells = rows * frames;

    // Pass 1: the band (plus its floor margin) of every frame.
    let mut re = vec![0.0f32; stored * frames];
    let mut im = vec![0.0f32; stored * frames];
    for j in 0..frames {
        stft.frame(x, j, |spec| {
            for r in 0..stored {
                let c = spec[row0 + r];
                re[r * frames + j] = c.re;
                im[r * frames + j] = c.im;
            }
        });
        if j % PROGRESS_EVERY == 0 {
            progress(PROGRESS_ANALYSED * j as f32 / frames as f32);
        }
    }

    // Noise floor: median across frequency, then a mean over time, in the log domain.
    let mut floor = vec![0.0f32; cells];
    let mut col = vec![0.0f32; stored];
    let mut med = vec![0.0f32; rows];
    for t in 0..frames {
        for r in 0..stored {
            let i = r * frames + t;
            col[r] = ((re[i] * re[i] + im[i] * im[i]).sqrt() + MAG_EPS).ln();
        }
        median_nearest(&col, row0, bins, floor_bins / 2, lo, hi, &mut med);
        for b in 0..rows {
            floor[b * frames + t] = med[b];
        }
    }
    let mut smooth = vec![0.0f32; frames];
    for b in 0..rows {
        let row = &mut floor[b * frames..(b + 1) * frames];
        box_nearest(row, floor_frames, &mut smooth);
        row.iter_mut().zip(&smooth).for_each(|(f, &s)| *f = s.exp());
    }

    // In-band magnitude, and how far each bin stands above the floor.
    let mut mag = vec![0.0f32; cells];
    let mut ratio_db = vec![0.0f32; cells];
    for o in 0..cells {
        let i = band0 * frames + o;
        let m = (re[i] * re[i] + im[i] * im[i]).sqrt();
        mag[o] = m;
        ratio_db[o] = 20.0 * ((m + MAG_EPS) / (floor[o] + MAG_EPS)).log10();
    }

    let RidgeMask { mask: ridge, ridges } = find_ridges(&ratio_db, &g, cfg);
    let foreground_db = cfg.foreground_db as f32;
    let is_replaced = |o: usize| ratio_db[o] < foreground_db || ridge[o] != 0;
    let report = ScrubReport {
        replaced_fraction: (0..cells).filter(|&o| is_replaced(o)).count() as f64 / cells as f64,
        ridge_fraction: ridge.iter().filter(|&&m| m != 0).count() as f64 / cells as f64,
        ridges,
    };
    progress(PROGRESS_PLANNED);
    if check_only {
        progress(1.0);
        return report;
    }

    // Replacement noise is circular complex Gaussian with E|n|^2 = sigma^2.
    //  * background bins: sigma^2 = local mean power of the background bins it
    //    replaces. Keeps the tonal balance; a narrow sub-floor line is diluted
    //    across the window and its shape is lost.
    //  * ridge bins: noise at the floor level (Rayleigh median = floor), i.e. removal.
    let mut weight = vec![0.0f32; cells];
    let mut power = vec![0.0f32; cells];
    for o in 0..cells {
        if ratio_db[o] < foreground_db && ridge[o] == 0 {
            weight[o] = 1.0;
            power[o] = mag[o] * mag[o];
        }
    }
    // The two box means become the noise's real and imaginary parts in place.
    let mut n_re = box_in_band(&power, &g);
    let mut n_im = box_in_band(&weight, &g);
    drop(power);
    let floor_sigma = 1.0 / core::f32::consts::LN_2.sqrt();
    let mut rng = Rng::new(cfg.seed);
    for o in 0..cells {
        let (num, den) = (n_re[o], n_im[o]);
        let sigma = if ridge[o] == 0 && den > MIN_BACKGROUND_WEIGHT {
            (num / den).max(0.0).sqrt()
        } else {
            floor[o] * floor_sigma
        };
        let s = sigma * core::f32::consts::FRAC_1_SQRT_2;
        n_re[o] = s * rng.normal() as f32;
        n_im[o] = s * rng.normal() as f32;
    }
    drop(floor);

    // Per-frequency level correction for background bins, from whole-file totals
    // only (one number per bin, smoothed across frequency): fixes fill bias
    // without reintroducing any time-varying structure.
    let row_sums = |value: &dyn Fn(usize) -> f64| -> Vec<f64> {
        (0..rows).map(|b| (b * frames..(b + 1) * frames).map(value).sum()).collect()
    };
    let p_in = row_sums(&|o| (weight[o] * mag[o] * mag[o]) as f64);
    let p_out = row_sums(&|o| (weight[o] * (n_re[o] * n_re[o] + n_im[o] * n_im[o])) as f64);
    let gain: Vec<f32> = smooth_rows(&p_in, &g)
        .iter()
        .zip(smooth_rows(&p_out, &g))
        .map(|(&a, b)| (a.max(0.0) / b.max(MIN_NOISE_POWER)).sqrt() as f32)
        .collect();
    for o in 0..cells {
        if weight[o] != 0.0 {
            n_re[o] *= gain[o / frames];
            n_im[o] *= gain[o / frames];
        }
    }
    // Never add energy: per frequency, replaced bins may not total more power
    // than the originals they replace. Stops edge-of-band bloat.
    let p_orig = row_sums(&|o| if is_replaced(o) { (mag[o] * mag[o]) as f64 } else { 0.0 });
    let p_new = row_sums(&|o| if is_replaced(o) { (n_re[o] * n_re[o] + n_im[o] * n_im[o]) as f64 } else { 0.0 });
    let cap: Vec<f32> = p_orig
        .iter()
        .zip(&p_new)
        .map(|(&a, &b)| (a / b.max(MIN_REPLACED_POWER)).sqrt().min(1.0) as f32)
        .collect();

    // Upstream stops here and inverts. Its noise has the right power per bin,
    // but unrelated noise in overlapping frames partly cancels when the frames
    // are added back, so the rebuilt background comes out 6 dB low. The makeup
    // gain is a constant of the window and hop: it restores the level and
    // carries nothing about the recording.
    let makeup = stft.incoherent_makeup();

    // Turn the stored spectrum into (replacement - original), zero where kept.
    for i in 0..stored * frames {
        let r = i / frames;
        let o = i.wrapping_sub(band0 * frames);
        if r < band0 || r >= band0 + rows || !is_replaced(o) {
            re[i] = 0.0;
            im[i] = 0.0;
            continue;
        }
        re[i] = n_re[o] * cap[r - band0] * makeup - re[i];
        im[i] = n_im[o] * cap[r - band0] * makeup - im[i];
    }
    drop((n_re, n_im, mag, weight));

    // Pass 2: add the difference back in the time domain.
    stft.add_inverse(x, |j, spec| {
        for r in 0..stored {
            spec[row0 + r] = Complex32::new(re[r * frames + j], im[r * frames + j]);
        }
        if j % PROGRESS_EVERY == 0 {
            progress(PROGRESS_PLANNED + (1.0 - PROGRESS_PLANNED) * j as f32 / frames as f32);
        }
    });
    progress(1.0);
    report
}
