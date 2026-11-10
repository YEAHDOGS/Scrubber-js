"""Detect mains hum that carries the electrical network frequency (ENF).

The grid's frequency wanders by a few hundredths of a hertz around 50 or 60 Hz,
the same way at every outlet on one interconnection. Hum picked up by a
microphone records that wander, so a few minutes of it can be matched against a
grid-frequency log to date the recording, and its nominal frequency says which
grid it was made on. This matters here because the aircraft attack needs the
recording time: stripping the file's date doesn't help if the hum still has it.

A line counts as ENF only if it is the grid and not a fan or an engine: at least
two harmonics of 50 or 60 Hz stand out, imply the same grid frequency second by
second, and sit within 0.2 Hz of nominal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly, stft


@dataclass
class ENFResult:
    present: bool
    nominal_hz: int | None = None
    harmonics_hz: list[int] = field(default_factory=list)  # harmonics that stand out
    line_db: dict[int, float] = field(default_factory=dict)  # harmonic -> dB above neighbours
    grid_hz: float | None = None  # median implied grid frequency
    wander_hz: float | None = None  # its standard deviation over the clip
    trace: np.ndarray | None = None  # implied grid frequency, one value per second

    def describe(self) -> str:
        if not self.present:
            return "no grid hum (ENF) detected"
        lines = ", ".join(f"{h} Hz +{self.line_db[h]:.0f} dB" for h in self.harmonics_hz)
        return (f"grid hum (ENF) present: {self.nominal_hz} Hz grid at {self.grid_hz:.3f} Hz "
                f"(wander {self.wander_hz:.3f} Hz) from {lines}. It can date the recording "
                f"and tells which power grid it was made on.")


def _track(P, f, fc):
    band = (f > fc - 0.5) & (f < fc + 0.5)
    S = P[band] + 1e-30
    k = np.clip(S.argmax(0), 1, band.sum() - 2)
    c = np.arange(S.shape[1])
    y0, y1, y2 = np.log(S[k - 1, c]), np.log(S[k, c]), np.log(S[k + 1, c])
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-12, den, 1e-12)
    return f[band][k] + np.clip(d, -0.5, 0.5) * (f[1] - f[0])


def detect_enf(audio: np.ndarray, sr: int, *, min_db: float = 8.0, harmonics=(1, 2, 3, 4)) -> ENFResult:
    """Look for ENF in mono ``audio``. Needs about 30 s or more."""
    x = audio if audio.ndim == 1 else audio.mean(axis=1)
    if sr != 8000:
        from math import gcd

        g = gcd(8000, sr)
        x = resample_poly(x, 8000 // g, sr // g)
    n = 8000 * 8  # 8 s window: 0.125 Hz bins; 1 s hop
    if len(x) < 4 * n:
        return ENFResult(False)
    f, _, Z = stft(x, 8000, nperseg=n, noverlap=n - 8000, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    best = ENFResult(False)
    for nominal in (50, 60):
        strong, db, tracks = [], {}, {}
        for k in harmonics:
            fc = nominal * k
            on = (f > fc - 0.6) & (f < fc + 0.6)
            side = ((f > fc - 3) & (f < fc - 1)) | ((f > fc + 1) & (f < fc + 3))
            snr = float(10 * np.log10(np.median(P[on].max(0)) / np.median(P[side])))
            db[fc] = snr
            if snr >= min_db:
                strong.append(k)
                tracks[k] = _track(P, f, fc) / k
        if len(strong) < 2:
            continue
        T = np.array([tracks[k] for k in strong])
        agree = np.median(np.abs(T - np.median(T, axis=0)), axis=1)
        grid = float(np.median(T))
        if np.median(agree) > 0.02 or abs(grid - nominal) > 0.2:
            continue  # harmonics drift apart or sit off-nominal: a machine, not the grid
        # weight the trace towards the strongest harmonic
        top = strong[int(np.argmax([db[nominal * k] for k in strong]))]
        res = ENFResult(True, nominal, [nominal * k for k in strong], {nominal * k: db[nominal * k] for k in strong},
                        grid, float(np.std(tracks[top])), tracks[top])
        if not best.present or sum(res.line_db.values()) > sum(best.line_db.values()):
            best = res
    return best
