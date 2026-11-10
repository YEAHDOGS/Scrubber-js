//! Scrubber core: remove aircraft Doppler signatures and grid hum from audio.
//!
//! A Rust port of [flyover-scrub](https://github.com/lizTheDeveloper/flyover-scrub),
//! built for WebAssembly. The functions at the bottom of this file are the whole
//! WebAssembly interface; `js/index.js` wraps them.

pub mod enf;
pub mod filters;
pub mod rng;
pub mod scrub;
pub mod stft;

use enf::detect_enf;
use scrub::{scrub, Ridge, ScrubConfig};
use std::alloc::{alloc, dealloc, Layout};

/// Every buffer handed across the boundary is aligned for f64.
const BUFFER_ALIGN: usize = 8;
/// Values in the config array read by `scrubber_scrub`.
pub const CONFIG_VALUES: usize = 13;
/// Values at the head of the scrub report, before the ridges.
pub const REPORT_HEAD: usize = 3;
/// Values written per ridge.
pub const RIDGE_VALUES: usize = 7;
/// Values in the hum report: 5 at the head, then up to 4 (Hz, dB) pairs.
pub const HUM_VALUES: usize = 13;
const HUM_HEAD: usize = 5;

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// Supplied by the host: called with a fraction from 0 to 1 while scrubbing.
    fn scrubber_progress(fraction: f32);
}

fn report_progress(fraction: f32) {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: the host supplies this import; it takes a plain number.
    unsafe {
        scrubber_progress(fraction)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let _ = fraction;
}

/// Allocates `bytes` of memory for the host to fill. Returns null on failure.
#[no_mangle]
pub extern "C" fn scrubber_alloc(bytes: usize) -> *mut u8 {
    let Ok(layout) = Layout::from_size_align(bytes.max(1), BUFFER_ALIGN) else {
        return core::ptr::null_mut();
    };
    // SAFETY: the layout has a non-zero size.
    unsafe { alloc(layout) }
}

/// Frees a buffer from `scrubber_alloc`. `bytes` must be the size it was made with.
///
/// # Safety
/// `ptr` must come from `scrubber_alloc(bytes)` and not have been freed.
#[no_mangle]
pub unsafe extern "C" fn scrubber_free(ptr: *mut u8, bytes: usize) {
    if ptr.is_null() {
        return;
    }
    dealloc(ptr, Layout::from_size_align_unchecked(bytes.max(1), BUFFER_ALIGN));
}

fn config_from(values: &[f64], seed: u64) -> ScrubConfig {
    ScrubConfig {
        band: (values[0], values[1]),
        window_s: values[2],
        hop_frac: values[3],
        floor_freq_hz: values[4],
        floor_time_s: values[5],
        foreground_db: values[6],
        ridge_db: values[7],
        ridge_min_s: values[8],
        ridge_max_slope_hz_s: values[9],
        ridge_pad_hz: values[10],
        keep_stationary_tones: values[11] != 0.0,
        stationary_max_drift_hz: values[12],
        seed,
    }
}

fn write_ridge(out: &mut [f64], r: &Ridge) {
    out.copy_from_slice(&[r.t_start, r.t_end, r.f_low, r.f_high, r.f_first, r.f_last, r.peak_db]);
}

/// Scrubs `len` mono f32 samples in place (or only inspects them when `check_only`).
///
/// `config` points at `CONFIG_VALUES` f64 values in `ScrubConfig` field order.
/// `out` receives: replaced fraction, ridge fraction, ridges found, then
/// `RIDGE_VALUES` per ridge, strongest first, as many as `out_len` allows.
/// Returns the number of ridges written.
///
/// # Safety
/// `samples`, `config` and `out` must be valid for the lengths given.
#[no_mangle]
pub unsafe extern "C" fn scrubber_scrub(
    samples: *mut f32,
    len: usize,
    sample_rate: f64,
    config: *const f64,
    seed_high: u32,
    seed_low: u32,
    check_only: u32,
    out: *mut f64,
    out_len: usize,
) -> usize {
    let x = core::slice::from_raw_parts_mut(samples, len);
    let cfg = config_from(core::slice::from_raw_parts(config, CONFIG_VALUES), (seed_high as u64) << 32 | seed_low as u64);
    let out = core::slice::from_raw_parts_mut(out, out_len);
    let mut report = scrub(x, sample_rate, &cfg, check_only != 0, &mut report_progress);
    if out.len() < REPORT_HEAD {
        return 0;
    }
    out[0] = report.replaced_fraction;
    out[1] = report.ridge_fraction;
    out[2] = report.ridges.len() as f64;
    report.ridges.sort_by(|a, b| b.peak_db.total_cmp(&a.peak_db));
    let slots = out[REPORT_HEAD..].chunks_exact_mut(RIDGE_VALUES);
    slots.zip(&report.ridges).map(|(slot, ridge)| write_ridge(slot, ridge)).count()
}

/// Looks for grid hum in `len` mono f32 samples.
///
/// `out` receives `HUM_VALUES` f64 values: present (0/1), nominal Hz, grid Hz,
/// wander Hz, number of lines, then (Hz, dB) for each line.
///
/// # Safety
/// `samples` must be valid for `len` values and `out` for `HUM_VALUES`.
#[no_mangle]
pub unsafe extern "C" fn scrubber_hum(samples: *const f32, len: usize, sample_rate: f64, min_db: f64, out: *mut f64) {
    let x = core::slice::from_raw_parts(samples, len);
    let out = core::slice::from_raw_parts_mut(out, HUM_VALUES);
    let found = detect_enf(x, sample_rate, min_db);
    out.fill(0.0);
    out[0] = found.present as u32 as f64;
    out[1] = found.nominal_hz as f64;
    out[2] = found.grid_hz;
    out[3] = found.wander_hz;
    out[4] = found.lines.len() as f64;
    for (slot, &(hz, db)) in out[HUM_HEAD..].chunks_exact_mut(2).zip(&found.lines) {
        slot[0] = hz as f64;
        slot[1] = db;
    }
}
