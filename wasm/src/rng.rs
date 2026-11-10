//! Small seedable generator for the replacement noise (xoshiro256++).
//!
//! The scrub does not rely on this being secret: a replaced bin holds nothing
//! of the original whatever noise fills it. It only has to be Gaussian and
//! different on every run.

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MUL_A: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL_B: u64 = 0x94D0_49BB_1331_11EB;
/// 2^-53: turns the top 53 bits of a u64 into a float in [0, 1).
const UNIT_53: f64 = 1.0 / 9_007_199_254_740_992.0;

pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in s.iter_mut() {
            z = z.wrapping_add(SPLITMIX_GAMMA);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(SPLITMIX_MUL_A);
            v = (v ^ (v >> 27)).wrapping_mul(SPLITMIX_MUL_B);
            *slot = v ^ (v >> 31);
        }
        Rng { s, spare: None }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * UNIT_53
    }

    /// Standard normal (Box-Muller, both values used).
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u = 1.0 - self.uniform(); // (0, 1]: safe for ln
        let r = (-2.0 * u.ln()).sqrt();
        let a = core::f64::consts::TAU * self.uniform();
        self.spare = Some(r * a.sin());
        r * a.cos()
    }
}
