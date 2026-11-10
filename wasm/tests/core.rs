//! Ports of the upstream tests in `tests/test_core.py`, plus a full grid search.

mod common;

use common::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use scrubber_core::enf::detect_enf;
use scrubber_core::rng::Rng;
use scrubber_core::scrub::{scrub, ScrubConfig, ScrubReport};
use std::f64::consts::TAU;

const SR: f64 = 8000.0;
const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];
const HUM_MIN_DB: f64 = 8.0;

fn straight(speed: f64) -> Track {
    make_track(0.0, 200.0, (-8000.0, 500.0), 600.0, speed, 0.0, Turn::default())
}

fn noise(n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| rng.normal()).collect()
}

fn run_scrub(x: &[f64], sr: f64, seed: u64) -> (Vec<f32>, ScrubReport) {
    let mut y = to_f32(x);
    let report = scrub(&mut y, sr, &ScrubConfig { seed, ..ScrubConfig::default() }, false, &mut |_| {});
    (y, report)
}

/// Flyover level against the background, dB: inaudible, and plainly audible.
const FAINT_DB: f64 = -28.0;
const LOUD_DB: f64 = -10.0;

/// A turning propeller flyover in white noise, heard from the origin.
struct Scene {
    track: Track,
    background: Vec<f64>,
    mix: Vec<f64>,
    t0: f64,
}

fn scene(seed: u64, seconds: usize, level_db: f64) -> Scene {
    let mut rng = Rng::new(seed);
    let turn = Turn { rate_deg_s: -1.0, start: 100.0, duration: 25.0 };
    let track = make_track(0.0, 260.0, (-9000.0, -400.0), 700.0, 70.0, 5.0, turn);
    let n = SR as usize * seconds;
    let t0 = 100.0;
    let mut plane = synthesize_flyover(&track, ORIGIN, t0, n, SR, 90.0, 0.15, &mut rng);
    let background = noise(n, &mut rng);
    let audible: Vec<f64> = plane.iter().copied().filter(|&v| v != 0.0).collect();
    let level = 10f64.powf(level_db / 20.0) * std_dev(&background) / std_dev(&audible);
    plane.iter_mut().for_each(|v| *v *= level);
    let mix = background.iter().zip(&plane).map(|(b, p)| b + p).collect();
    Scene { track, background, mix, t0 }
}

#[test]
fn doppler_limits_match_textbook() {
    let (_, g, _) = emission_geometry(&straight(80.0), ORIGIN);
    // Far approach and far recede: c / (c -/+ v) for a source moving straight at you.
    assert!((g[5] - SPEED_OF_SOUND / (SPEED_OF_SOUND - 80.0)).abs() < 0.01);
    assert!((g[g.len() - 5] - SPEED_OF_SOUND / (SPEED_OF_SOUND + 80.0)).abs() < 0.01);
}

#[test]
fn synth_puts_energy_on_predicted_curve() {
    let track = straight(80.0);
    let n = SR as usize * 60;
    let mut rng = Rng::new(0);
    let mut x = synthesize_flyover(&track, ORIGIN, 70.0, n, SR, 100.0, 0.0, &mut rng);
    x.iter_mut().for_each(|v| *v += 1e-3 * rng.normal());
    let spec = whitened_spectrogram(&to_f32(&x), SR, 1.0);
    let heard = Heard::new(&track, ORIGIN);
    let (mut on, mut off) = (0.0, 0.0);
    for (j, t) in spec.times.iter().enumerate() {
        let bin = (100.0 * heard.at(70.0 + t).unwrap().0 / spec.df).round() as usize;
        on += spec.at(bin, j) as f64;
        off += spec.at(bin + 15, j) as f64;
    }
    assert!(on / spec.frames as f64 > off / spec.frames as f64 + 2.0);
}

#[test]
fn attack_prefers_truth_and_scrub_removes_it() {
    const SEEDS: [u64; 6] = [3, 4, 5, 6, 7, 8];
    let candidates = [[0.0, 0.0, 0.0], [0.0, 800.0, 0.0], [600.0, -600.0, 0.0], [-1500.0, 900.0, 0.0]];
    let search = Search { f0_range: (87.0, 93.0), offsets: (-2.0, 2.0) };
    let mut residual = 0.0;
    for seed in SEEDS {
        let s = scene(seed, 80, FAINT_DB);
        let score = |audio: &[f32]| score_candidates(&whitened_spectrogram(audio, SR, 1.0), &s.track, s.t0, &candidates, &search);
        let raw = score(&to_f32(&s.mix));
        let best = raw.iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(raw[0], best, "seed {seed}: the attack should rank the true spot first");

        let (y, _) = run_scrub(&s.mix, SR, seed);
        let scrubbed = score(&y);
        let control = score(&to_f32(&s.background));
        // After scrubbing, the true location must score like it does with no
        // aircraft at all. (Comparing the winner over 4 candidates is
        // meaningless: geometry alone can put truth first even in the control.)
        // Two noise-only recordings differ by about +/-0.07 here by chance, so
        // one run is held to 0.1 and the average over all runs to 0.03.
        assert!(raw[0] - control[0] > 0.3, "seed {seed}: raw {} vs control {}", raw[0], control[0]);
        assert!(scrubbed[0] - control[0] < 0.1, "seed {seed}: scrubbed {} vs control {}", scrubbed[0], control[0]);
        residual += (scrubbed[0] - control[0]) / SEEDS.len() as f64;
    }
    assert!(residual < 0.03, "mean score left after scrubbing: {residual}");
}

/// The upstream evaluation in miniature: a 10 x 10 km search for the recording.
/// Runs the search on four recordings and returns where the true spot ranks after scrubbing.
fn grid_search(level_db: f64) -> (Vec<usize>, usize) {
    const SEEDS: [u64; 4] = [40, 41, 42, 43];
    const HALF_WIDTH: f64 = 5000.0;
    const STEP: f64 = 250.0;
    const NEAR_M: f64 = 150.0;
    let candidates = grid((0.0, 0.0), HALF_WIDTH, STEP);
    let search = Search { f0_range: (86.0, 94.0), offsets: (-2.0, 2.0) };
    let mut scrubbed_ranks = Vec::new();
    for seed in SEEDS {
        let s = scene(seed, 80, level_db);
        // Rank (1 = best) of the best candidate within NEAR_M of the true spot.
        let truth_rank = |audio: &[f32]| -> usize {
            let scores = score_candidates(&whitened_spectrogram(audio, SR, 1.0), &s.track, s.t0, &candidates, &search);
            let near = candidates
                .iter()
                .zip(&scores)
                .filter(|(c, _)| c[0].hypot(c[1]) <= NEAR_M)
                .map(|(_, &v)| v)
                .fold(f64::MIN, f64::max);
            1 + scores.iter().filter(|&&v| v > near).count()
        };
        let raw = truth_rank(&to_f32(&s.mix));
        let scrubbed = truth_rank(&run_scrub(&s.mix, SR, seed).0);
        println!("{level_db} dB, seed {seed}: true spot ranks {raw} raw and {scrubbed} scrubbed, of {}", candidates.len());
        assert_eq!(raw, 1, "seed {seed}: the attack should find the raw recording");
        scrubbed_ranks.push(scrubbed);
    }
    scrubbed_ranks.sort_unstable();
    (scrubbed_ranks, candidates.len())
}

#[test]
fn grid_search_finds_raw_recording_and_loses_scrubbed_one() {
    for level_db in [FAINT_DB, LOUD_DB] {
        let (ranks, candidates) = grid_search(level_db);
        // Chance puts the true spot anywhere in the list; the middle run must be deep in the crowd.
        assert!(ranks[ranks.len() / 2] > candidates / 10, "{level_db} dB: scrubbed recordings still rank near the top: {ranks:?}");
    }
}

#[test]
fn scrub_preserves_foreground_and_level() {
    let mut rng = Rng::new(1);
    let n = SR as usize * 20;
    let mut phase = 0.0;
    let mut env = Vec::with_capacity(n);
    let mut voice = Vec::with_capacity(n);
    // Bursty "speech-like" harmonic content: short syllables, fast pitch movement.
    for i in 0..n {
        let t = i as f64 / SR;
        phase += TAU * (180.0 + 40.0 * (TAU * 3.0 * t).sin()) / SR;
        let on = (TAU * 2.5 * t).sin() > 0.3;
        env.push(on);
        let sum: f64 = (1..8).map(|k| 0.2 / k as f64 * (k as f64 * phase).sin()).sum();
        voice.push(if on { sum } else { 0.0 });
    }
    let x: Vec<f64> = voice.iter().map(|v| v + 0.01 * rng.normal()).collect();
    let (y, report) = run_scrub(&x, SR, 0);
    assert!(report.ridges.len() <= 3, "syllables must not be chained into long ridges");
    let pick = |v: &[f64]| -> Vec<f64> { v.iter().zip(&env).filter(|(_, &on)| on).map(|(v, _)| *v).collect() };
    let diff: Vec<f64> = y.iter().zip(&x).map(|(a, b)| *a as f64 - b).collect();
    // The quiet edges of the voice (under 12 dB above the floor) are rebuilt
    // from noise at their own level, so the waveform differs there by design.
    // Upstream holds this to 0.35 with a fill that comes out 6 dB low; at the
    // corrected level the same bins cost a little more.
    let err = std_dev(&pick(&diff)) / std_dev(&pick(&voice));
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let level_db = 20.0 * (std_dev(&y64) / std_dev(&x)).log10();
    println!("foreground error {err:.3}, level change {level_db:.2} dB");
    assert!(err < 0.4);
    assert!(level_db.abs() < 1.0);
}

#[test]
fn nothing_above_the_band_changes() {
    const TOP_OF_BAND_HZ: f64 = 2100.0;
    let n = 1 << 16;
    let x = noise(n, &mut Rng::new(7));
    let (y, _) = run_scrub(&x, SR, 0);
    let mut diff: Vec<Complex32> = y.iter().zip(&x).map(|(a, b)| Complex32::new(a - *b as f32, 0.0)).collect();
    FftPlanner::<f32>::new().plan_fft_forward(n).process(&mut diff);
    let power = |c: &Complex32| (c.re * c.re + c.im * c.im) as f64;
    let split = (TOP_OF_BAND_HZ / SR * n as f64) as usize;
    let below: f64 = diff[..split].iter().map(power).sum();
    let above: f64 = diff[split..n / 2].iter().map(power).sum();
    assert!(below > 0.0, "the scrub should change the protected band");
    // Random noise per frame is not a consistent STFT, so the synthesis window
    // spreads a trace of it past the band edge (scipy does the same): 40 dB down.
    assert!(above < 1e-4 * below, "leak above the band: {above} vs {below}");
}

#[test]
fn check_only_leaves_audio_alone() {
    let x = to_f32(&scene(3, 20, FAINT_DB).mix);
    let mut y = x.clone();
    let report = scrub(&mut y, SR, &ScrubConfig::default(), true, &mut |_| {});
    assert_eq!(x, y);
    assert!(report.replaced_fraction > 0.5);
}

fn grid_hum(n: usize, sr: f64) -> Vec<f64> {
    // Grid frequency wandering by a few hundredths of a hertz around 60 Hz.
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            phase += TAU * (60.0 + 0.02 * (TAU * t / 37.0).sin() + 0.01 * (TAU * t / 11.0).sin()) / sr;
            // 120 and 240 Hz, as transformers hum.
            0.05 * (2.0 * phase).sin() + 0.03 * (4.0 * phase).sin()
        })
        .collect()
}

#[test]
fn enf_detected_in_grid_hum_and_removed_by_scrub() {
    let n = SR as usize * 90;
    let quiet: Vec<f64> = noise(n, &mut Rng::new(2)).iter().map(|v| v * 0.3).collect();
    let humming: Vec<f64> = grid_hum(n, SR).iter().zip(&quiet).map(|(h, q)| h + q).collect();
    let found = detect_enf(&to_f32(&humming), SR, HUM_MIN_DB);
    assert!(found.present && found.nominal_hz == 60);
    assert!((found.grid_hz - 60.0).abs() < 0.01);
    assert!(!detect_enf(&to_f32(&quiet), SR, HUM_MIN_DB).present);
    // A machine tone that drifts independently at each "harmonic" is not ENF.
    let fan: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / SR;
            0.05 * (TAU * 120.3 * t).sin() + 0.03 * (TAU * 239.1 * t).sin() + quiet[i]
        })
        .collect();
    assert!(!detect_enf(&to_f32(&fan), SR, HUM_MIN_DB).present);
    let (y, _) = run_scrub(&humming, SR, 0);
    assert!(!detect_enf(&y, SR, HUM_MIN_DB).present);
}

#[test]
fn enf_survives_resampling_from_48k() {
    const SR_48K: f64 = 48_000.0;
    let n = SR_48K as usize * 60;
    let mut rng = Rng::new(9);
    let humming: Vec<f64> = grid_hum(n, SR_48K).iter().map(|h| h + 0.3 * rng.normal()).collect();
    let found = detect_enf(&to_f32(&humming), SR_48K, HUM_MIN_DB);
    assert!(found.present && found.nominal_hz == 60);
    assert!((found.grid_hz - 60.0).abs() < 0.01);
}

/// Power of `x` between two frequencies, by FFT.
fn band_power(x: &[f32], sr: f64, lo: f64, hi: f64) -> f64 {
    let n = x.len();
    let mut spec: Vec<Complex32> = x.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    FftPlanner::<f32>::new().plan_fft_forward(n).process(&mut spec);
    let bin = |hz: f64| (hz / sr * n as f64) as usize;
    spec[bin(lo)..bin(hi)].iter().map(|c| (c.re * c.re + c.im * c.im) as f64).sum()
}

#[test]
fn rebuilt_background_keeps_its_level() {
    let n = 1 << 18;
    let x = noise(n, &mut Rng::new(12));
    let (y, report) = run_scrub(&x, SR, 0);
    assert!(report.replaced_fraction > 0.99, "plain noise is all background");
    let change_db = 10.0 * (band_power(&y, SR, 200.0, 1800.0) / band_power(&to_f32(&x), SR, 200.0, 1800.0)).log10();
    println!("level change in the rebuilt band: {change_db:.2} dB");
    assert!(change_db.abs() < 1.0, "rebuilt band is off by {change_db:.2} dB");
}
