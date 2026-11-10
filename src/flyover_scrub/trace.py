"""Trace-fit attack: the method that worked on real recordings.

The brightness scorer in ``attack.py`` sums energy under every modelled curve.
On real iPhone recordings a stronger and simpler attack is the one colsto
used for faint lines:

1. **Trace** the visible ridge: in a whitened long-window spectrogram, follow
   one line frame by frame from its strongest point, keeping only frames where
   it is strong and continuous.
2. **Fit** each candidate location: the track gives the Doppler factor g(t) at
   that spot, the unknown source pitch is solved in closed form
   (log f_obs = log f0 + log g), and the score is the RMS residual in cents.
   No f0 sweep is needed, so nothing is tuned at the true location.
3. The candidate (and clock offset) with the lowest residual wins.

``reversal_ratio`` is the chance check: the same fit on the time-reversed
ridge. A real flyover only ever falls in pitch, so its reversal fits badly; a
steady hum or a random wander fits both ways about equally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .attack import AttackResult, Spectrogram, grid
from .physics import Track, received_doppler

CENTS_PER_LOG = 1200.0 / np.log(2.0)


@dataclass
class Ridge:
    t: np.ndarray  # seconds from the start of the spectrogram's audio
    f: np.ndarray  # Hz

    def __len__(self) -> int:
        return len(self.t)

    def reversed(self) -> "Ridge":
        """Same frames and frequencies, played backwards (rising instead of falling)."""
        return Ridge(t=self.t, f=self.f[::-1].copy())


def trace_ridge(
    spec: Spectrogram,
    *,
    f_range: tuple[float, float] = (30.0, 1000.0),
    t_range: tuple[float, float] | None = None,
    follow_hz: float = 0.6,
    min_strength: float = 2.5,
    start: tuple[int, int] | None = None,
) -> Ridge:
    """Follow one ridge from its strongest point (or from ``start`` = (freq_idx, frame_idx)
    inside the selected band) and keep the strong, continuous frames."""
    fm = (spec.freqs >= f_range[0]) & (spec.freqs <= f_range[1])
    tm = np.ones(len(spec.times), bool) if t_range is None else (spec.times >= t_range[0]) & (spec.times <= t_range[1])
    S = spec.S[np.ix_(fm, tm)]
    f = spec.freqs[fm]
    t = spec.times[tm]
    if S.size == 0:
        return Ridge(np.zeros(0), np.zeros(0))
    w = max(1, int(round(follow_hz / spec.df)))
    if start is None:
        j0 = int(S.max(0).argmax())
        k0 = int(S[:, j0].argmax())
    else:
        k0, j0 = start
    k = np.zeros(len(t), int)
    k[j0] = k0
    for rng in (range(j0 + 1, len(t)), range(j0 - 1, -1, -1)):
        prev = k0
        for j in rng:
            lo, hi = max(prev - w, 0), min(prev + w + 1, len(f))
            k[j] = lo + int(S[lo:hi, j].argmax())
            prev = k[j]
    cols = np.arange(len(t))
    pk = S[k, cols]
    # parabolic sub-bin interpolation
    kk = np.clip(k, 1, len(f) - 2)
    y0, y1, y2 = S[kk - 1, cols], S[kk, cols], S[kk + 1, cols]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1e-9)
    fr = f[kk] + np.clip(d, -0.5, 0.5) * spec.df
    smooth = median_filter(fr, 9)
    keep = (pk > min_strength) & (np.abs(fr - smooth) < 1.5)
    return Ridge(t=t[keep], f=fr[keep])


def fit_candidates(ridge: Ridge, track: Track, audio_t0: float, candidates: np.ndarray, offsets: np.ndarray):
    """RMS residual (cents), best f0 (Hz) and best clock offset (s) per candidate."""
    lf = np.log(ridge.f)
    n = len(ridge)
    cents = np.full(len(candidates), np.inf)
    f0 = np.zeros(len(candidates))
    off = np.zeros(len(candidates))
    tr = audio_t0 + offsets[:, None] + ridge.t[None, :]
    for i, c in enumerate(candidates):
        g, _, _ = received_doppler(track, c, tr.ravel())
        lg = np.log(np.where(np.isfinite(g) & (g > 0), g, np.nan)).reshape(tr.shape)
        ok = np.isfinite(lg).sum(1) > 0.8 * n
        if not ok.any():
            continue
        with np.errstate(invalid="ignore"):
            lf0 = np.nanmean(lf[None, :] - lg, axis=1)
            rms = np.sqrt(np.nanmean((lf[None, :] - lg - lf0[:, None]) ** 2, axis=1))
        rms = np.where(ok, rms, np.inf)
        j = int(np.argmin(rms))
        cents[i], f0[i], off[i] = rms[j] * CENTS_PER_LOG, np.exp(lf0[j]), offsets[j]
    return cents, f0, off


def locate_trace(
    ridge: Ridge,
    track: Track,
    audio_t0: float,
    search_center: tuple[float, float],
    *,
    half_width: float = 5000.0,
    coarse: float = 200.0,
    fine: float = 25.0,
    refine_top: int = 15,
    offsets: np.ndarray | None = None,
) -> AttackResult:
    """Coarse grid then refinement. ``scores`` are negative cents (higher = better fit)."""
    if offsets is None:
        offsets = np.arange(-8.0, 8.01, 0.25)
    c = grid(search_center, half_width, coarse)
    s1, f1, o1 = fit_candidates(ridge, track, audio_t0, c, offsets)
    if np.isfinite(s1).any():
        seeds = c[np.argsort(s1)[:refine_top]]
        fp = np.unique(np.round(np.concatenate([grid(tuple(p[:2]), coarse, fine) for p in seeds]), 1), axis=0)
        s2, f2, o2 = fit_candidates(ridge, track, audio_t0, fp, offsets)
    else:
        fp, s2, f2, o2 = np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
    return AttackResult(
        candidates=np.concatenate([c, fp]),
        scores=-np.concatenate([s1, s2]),
        best_f0=np.concatenate([f1, f2]),
        best_offset=np.concatenate([o1, o2]),
    )


def reversal_ratio(ridge: Ridge, track: Track, audio_t0: float, result: AttackResult, *, top: int = 200,
                   offsets: np.ndarray | None = None) -> float:
    """Best reversed-ridge fit / best forward fit, over the forward search's top candidates.

    ~1 means the ridge fits a flyover no better backwards than forwards (chance);
    a real flyover gives several times more."""
    if offsets is None:
        offsets = np.arange(-8.0, 8.01, 0.25)
    fwd = -float(np.nanmax(result.scores))
    if not np.isfinite(fwd):
        return float("nan")
    # the reversed ridge gets its own best shot anywhere on the coarse+fine grid's top region
    cand = result.candidates[result.ranked()[:top]]
    rest = result.candidates[np.isfinite(result.scores)]
    sub = rest[np.linspace(0, len(rest) - 1, min(len(rest), 400)).astype(int)]
    rev, _, _ = fit_candidates(ridge.reversed(), track, audio_t0, np.concatenate([cand, sub]), offsets)
    return float(np.nanmin(rev) / max(fwd, 1e-6))
