//! The few `scipy.ndimage` filters the scrubber needs, with the same edge rules.

/// Index outside `0..n` is pulled to the nearest edge (`mode="nearest"`).
#[inline]
pub fn nearest(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// Index outside `0..n` is mirrored about the edge (`mode="reflect"`: d c b a | a b c d).
#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    // Loops only for windows longer than the array itself.
    while i < 0 || i >= n {
        i = if i < 0 { -i - 1 } else { 2 * n - i - 1 };
    }
    i as usize
}

/// Running mean of odd width `size` along a slice, edges held (`mode="nearest"`).
pub fn box_nearest(src: &[f32], size: usize, out: &mut [f32]) {
    let n = src.len();
    let half = (size / 2) as isize;
    let mut sum = 0.0f64;
    for k in -half..=half {
        sum += src[nearest(k, n)] as f64;
    }
    for i in 0..n {
        out[i] = (sum / size as f64) as f32;
        let i = i as isize;
        sum += src[nearest(i + half + 1, n)] as f64 - src[nearest(i - half, n)] as f64;
    }
}

/// Running mean of odd width `size` along a slice, edges mirrored (`mode="reflect"`).
pub fn box_reflect(src: &[f64], size: usize) -> Vec<f64> {
    let n = src.len();
    let half = (size / 2) as isize;
    (0..n as isize)
        .map(|i| (-half..=half).map(|k| src[reflect(i + k, n)]).sum::<f64>() / size as f64)
        .collect()
}

/// Sliding median of odd width along `col`, edges held, for rows `first..=last`.
///
/// `col[0]` is row `row0` of an array `rows` long; the caller guarantees every
/// row the window can reach (after clamping to the array) is inside `col`.
pub fn median_nearest(col: &[f32], row0: usize, rows: usize, half: usize, first: usize, last: usize, out: &mut [f32]) {
    let at = |r: isize| col[nearest(r, rows) - row0];
    let half = half as isize;
    let mut win: Vec<f32> = (-half..=half).map(|k| at(first as isize + k)).collect();
    win.sort_unstable_by(f32::total_cmp);
    let mid = win.len() / 2;
    for (o, r) in (first..=last).enumerate() {
        out[o] = win[mid];
        if r == last {
            break;
        }
        let r = r as isize;
        let gone = at(r - half);
        let come = at(r + half + 1);
        let pos = win.partition_point(|&v| v.total_cmp(&gone).is_lt());
        win.remove(pos);
        let pos = win.partition_point(|&v| v.total_cmp(&come).is_lt());
        win.insert(pos, come);
    }
}
