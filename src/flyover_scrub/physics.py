"""Acoustic flyover physics: flight tracks, propagation delay, and Doppler.

All positions are local east/north/up metres. Times are seconds (any epoch,
as long as tracks and recordings agree).

The one subtle point (and the one that makes the attack work): the frequency an
observer hears at reception time ``t_r`` is set by the aircraft's radial
velocity at the *emission* time ``t_e``, where ``t_r = t_e + |p(t_e) - o| / c``.
Evaluating the radial velocity at ``t_r`` instead gives a curve that is not just
delayed but the wrong shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s, ~20 C


@dataclass
class Track:
    """A sampled flight path. ``t`` shape (M,), ``pos`` shape (M, 3)."""

    t: np.ndarray
    pos: np.ndarray

    def resample(self, t_new: np.ndarray) -> "Track":
        pos = np.stack([np.interp(t_new, self.t, self.pos[:, i]) for i in range(3)], axis=1)
        return Track(t=np.asarray(t_new, float), pos=pos)


def make_track(
    t0: float,
    duration: float,
    start_xy: tuple[float, float],
    altitude: float,
    speed: float,
    heading_deg: float,
    *,
    dt: float = 0.05,
    turn_rate_deg_s: float = 0.0,
    turn_start: float = 0.0,
    turn_duration: float = 0.0,
    accel: float = 0.0,
    climb_rate: float = 0.0,
) -> Track:
    """Generate a smooth track with an optional constant-rate turn window.

    Heading is the math angle from +x (east), counter-clockwise, in degrees.
    Real aircraft are never perfectly straight and level; the small turn and
    acceleration terms are what make the Doppler curve location-specific.
    """
    tau = np.arange(0.0, duration + dt, dt)
    turned = turn_rate_deg_s * np.clip(tau - turn_start, 0.0, turn_duration)
    hd = np.radians(heading_deg + turned)
    v = speed + accel * tau
    vx, vy = v * np.cos(hd), v * np.sin(hd)
    x = start_xy[0] + np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt)])
    y = start_xy[1] + np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt)])
    z = altitude + climb_rate * tau
    return Track(t=t0 + tau, pos=np.stack([x, y, z], axis=1))


def degrade_like_adsb(
    track: Track, rng: np.random.Generator, *, period: float = 1.0, pos_sigma: float = 8.0
) -> Track:
    """What an attacker actually gets: ~1 Hz positions with metre-scale noise."""
    t_s = np.arange(track.t[0], track.t[-1], period)
    s = track.resample(t_s)
    noisy = s.pos + rng.normal(0.0, pos_sigma, s.pos.shape) * np.array([1.0, 1.0, 0.5])
    return Track(t=t_s, pos=noisy)


def smooth_track(track: Track, *, out_dt: float = 0.25, smoothing_s: float = 8.0) -> Track:
    """Attacker-side cleanup of ADS-B: smoothing spline, resampled densely.

    Differentiating raw 1 Hz positions turns metres of noise into tens of m/s
    of fake radial velocity, which wrecks the modelled Doppler curve.
    """
    from scipy.interpolate import make_smoothing_spline

    t_new = np.arange(track.t[0], track.t[-1], out_dt)
    lam = (smoothing_s / 2.0) ** 3  # roughly: features shorter than smoothing_s are suppressed
    pos = np.stack(
        [make_smoothing_spline(track.t, track.pos[:, i], lam=lam)(t_new) for i in range(3)], axis=1
    )
    return Track(t=t_new, pos=pos)


def emission_geometry(track: Track, observer: np.ndarray, c: float = SPEED_OF_SOUND):
    """Per track sample: reception time, Doppler factor and range for one observer.

    Returns ``(t_rec, g, r)``; the received frequency is ``f0 * g``.
    """
    d = track.pos - observer[None, :]
    r = np.linalg.norm(d, axis=1)
    t_rec = track.t + r / c
    v_r = np.gradient(r, track.t)  # radial velocity at emission (+ = receding)
    g = 1.0 / (1.0 + v_r / c)
    return t_rec, g, r


def received_doppler(
    track: Track, observer: np.ndarray, t_r: np.ndarray, c: float = SPEED_OF_SOUND
):
    """Doppler factor, range and emission time *as heard* at reception times ``t_r``.

    Samples outside the track's audible span come back as NaN.
    """
    t_rec, g, r = emission_geometry(track, observer, c)
    out = np.full_like(t_r, np.nan, dtype=float)
    inside = (t_r >= t_rec[0]) & (t_r <= t_rec[-1])
    g_r, r_r, te_r = out.copy(), out.copy(), out.copy()
    g_r[inside] = np.interp(t_r[inside], t_rec, g)
    r_r[inside] = np.interp(t_r[inside], t_rec, r)
    te_r[inside] = np.interp(t_r[inside], t_rec, track.t)
    return g_r, r_r, te_r


def synthesize_flyover(
    track: Track,
    observer: np.ndarray,
    t_start: float,
    n_samples: int,
    sr: int,
    *,
    f0: float = 92.0,
    harmonics: tuple[float, ...] = (1.0, 0.7, 0.45, 0.3, 0.2),
    broadband: float = 0.15,
    rng: np.random.Generator | None = None,
    c: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Physically-modelled propeller flyover as heard at ``observer``.

    The received waveform is ``sin(2*pi*k*f0*t_e(t_r))``: writing phase in
    emission time gives exact Doppler (and retardation) for free.
    Amplitude falls as 1/r; output is normalised so its peak RMS is ~1.
    """
    rng = rng or np.random.default_rng(0)
    t_r = t_start + np.arange(n_samples) / sr
    _, r_r, te_r = received_doppler(track, observer, t_r, c)
    ok = np.isfinite(te_r)
    te = np.where(ok, te_r, 0.0)
    amp = np.where(ok, 1.0 / np.maximum(r_r, 1.0), 0.0)
    amp /= amp.max() if amp.max() > 0 else 1.0
    # slow blade-rate wander so it is not a perfect synthetic tone
    wander = 1.0 + 0.002 * np.sin(2 * np.pi * 0.07 * te + rng.uniform(0, 6.28))
    phase = 2 * np.pi * f0 * np.cumsum(np.gradient(te) * wander)
    sig = np.zeros(n_samples)
    for k, a in enumerate(harmonics, start=1):
        sig += a * np.sin(k * phase + rng.uniform(0, 6.28))
    if broadband > 0:
        from scipy.signal import butter, sosfilt

        sos = butter(2, [60, 900], btype="band", fs=sr, output="sos")
        sig += broadband * 3.0 * sosfilt(sos, rng.normal(0, 1, n_samples))
    return sig * amp
