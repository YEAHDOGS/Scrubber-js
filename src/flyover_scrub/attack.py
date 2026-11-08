"""Reference implementation of the Doppler-curve geolocation attack.

This exists so the mitigation can be *measured*: a defence is only real if this
scorer can no longer rank the true recording location first.

Method (after colsto, "I discovered a sound that leaks your exact location"):
  1. Long-window spectrogram, whitened per frequency bin so faint lines that
     are invisible to the ear stand out (equivalent to "boosting" after a
     high-pass).
  2. For each candidate observer location, model the exact Doppler curve that
     the known flight track would produce there (emission-time correct).
  3. Sum the whitened energy under that curve across several harmonics,
     sweeping the unknown fundamental and a small clock offset.
  4. The candidate whose modelled curve best covers the observed line wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from .physics import Track, received_doppler


@dataclass
class Spectrogram:
    """Whitened log-power spectrogram. ``S`` is (freq, time)."""

    S: np.ndarray
    freqs: np.ndarray
    times: np.ndarray  # seconds from start of audio

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def whitened_spectrogram(
    audio: np.ndarray, sr: int, *, window_s: float = 1.0, hop_s: float = 0.25, fmax: float = 1200.0
) -> Spectrogram:
    nper = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    f, t, Z = stft(audio, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    keep = f <= fmax
    P = np.abs(Z[keep]) ** 2 + 1e-20
    # whiten: divide each bin by its median over time, then local median over freq
    P /= np.median(P, axis=1, keepdims=True)
    L = np.log(P)
    from scipy.ndimage import median_filter

    L -= median_filter(L, size=(9, 1), mode="nearest")
    return Spectrogram(S=np.clip(L, -3, 8).astype(np.float32), freqs=f[keep], times=t)


@dataclass
class AttackResult:
    candidates: np.ndarray  # (N, 3)
    scores: np.ndarray  # (N,)
    best_f0: np.ndarray
    best_offset: np.ndarray

    def ranked(self) -> np.ndarray:
        return np.argsort(-self.scores)

    def summary(self, truth_xy: np.ndarray) -> dict:
        order = self.ranked()
        d = np.linalg.norm(self.candidates[:, :2] - truth_xy[None, :], axis=1)
        best = order[0]
        # rank of the best candidate within 150 m of truth
        near = np.where(d[order] <= 150.0)[0]
        s = self.scores
        z = (s[best] - np.median(s)) / (np.std(s) + 1e-12)
        return {
            "top_error_m": float(d[best]),
            "truth_rank": int(near[0]) + 1 if near.size else None,
            "n_candidates": int(len(s)),
            "top_score": float(s[best]),
            "top_zscore": float(z),
        }


def score_candidates(
    spec: Spectrogram,
    tracks: list[Track],
    audio_t0: float,
    candidates: np.ndarray,
    *,
    f0_range: tuple[float, float],
    f0_step: float = 0.25,
    offsets: np.ndarray | None = None,
    n_harmonics: int = 3,
) -> AttackResult:
    """Score every candidate location against the spectrogram.

    ``audio_t0`` is the attacker's estimate of the recording start time on the
    ADS-B clock; ``offsets`` sweeps the error in that estimate. With several
    tracks, the offset is shared (locked) between them, as in the video, which
    is what makes two flights so constraining.
    """
    if offsets is None:
        offsets = np.arange(-4.0, 4.0 + 1e-9, 0.25)
    f0s = np.arange(f0_range[0], f0_range[1] + 1e-9, f0_step)
    S = spec.S
    nF, nT = S.shape
    flat = S.ravel()
    ks = np.arange(1, n_harmonics + 1)

    scores = np.full(len(candidates), -np.inf)
    best_f0 = np.zeros(len(candidates))
    best_off = np.zeros(len(candidates))

    for i, obs in enumerate(candidates):
        total = np.zeros((len(offsets), len(f0s)))
        for trk in tracks:
            # reception times for every (offset, frame)
            tr = audio_t0 + offsets[:, None] + spec.times[None, :]
            g, r, _ = received_doppler(trk, obs, tr.ravel())
            g = g.reshape(tr.shape)
            r = r.reshape(tr.shape)
            ok = np.isfinite(g) & (r < 12000.0)
            if not ok.any():
                continue
            w = np.where(ok, 1.0 / np.maximum(np.nan_to_num(r, nan=1e9), 300.0), 0.0)
            w /= w.sum(axis=1, keepdims=True) + 1e-12
            gz = np.nan_to_num(g)
            # bins: (offset, f0, harmonic, frame)
            fb = (f0s[None, :, None, None] * ks[None, None, :, None] * gz[:, None, None, :]) / spec.df
            b = np.clip(np.rint(fb).astype(np.int64), 0, nF - 1)
            frames = np.arange(nT)[None, None, None, :]
            vals = flat[b * nT + frames]  # (O, F0, K, T)
            vals = np.where(fb < nF - 1, vals, 0.0)
            total += np.einsum("ofkt,ot->of", vals, w) / n_harmonics
        j = np.unravel_index(np.argmax(total), total.shape)
        scores[i] = total[j]
        best_off[i] = offsets[j[0]]
        best_f0[i] = f0s[j[1]]
    return AttackResult(candidates=candidates, scores=scores, best_f0=best_f0, best_offset=best_off)


def grid(center_xy: tuple[float, float], half_width: float, step: float, z: float = 0.0) -> np.ndarray:
    xs = np.arange(-half_width, half_width + 1e-9, step) + center_xy[0]
    ys = np.arange(-half_width, half_width + 1e-9, step) + center_xy[1]
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def locate(
    spec: Spectrogram,
    tracks: list[Track],
    audio_t0: float,
    search_center: tuple[float, float],
    *,
    half_width: float = 5000.0,
    coarse: float = 250.0,
    fine: float = 50.0,
    refine_top: int = 12,
    **kw,
) -> AttackResult:
    """Coarse grid, then refine around the best coarse cells (as in the video)."""
    c = grid(search_center, half_width, coarse)
    res = score_candidates(spec, tracks, audio_t0, c, **kw)
    seeds = c[res.ranked()[:refine_top]]
    fine_pts = np.concatenate([grid(tuple(s[:2]), coarse, fine) for s in seeds])
    fine_pts = np.unique(np.round(fine_pts, 1), axis=0)
    res2 = score_candidates(spec, tracks, audio_t0, fine_pts, **kw)
    return AttackResult(
        candidates=np.concatenate([c, fine_pts]),
        scores=np.concatenate([res.scores, res2.scores]),
        best_f0=np.concatenate([res.best_f0, res2.best_f0]),
        best_offset=np.concatenate([res.best_offset, res2.best_offset]),
    )
