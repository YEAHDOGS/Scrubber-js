"""End-to-end check: can the Doppler attack locate a recording, before and after scrubbing?

A physically-modelled propeller flyover is mixed into a real recording (the
recording's GPS position is the ground-truth observer). The attacker gets an
ADS-B-like degraded copy of the track, a recording start time that is wrong by
up to a few seconds, and a 10 km x 10 km search box. We report where the attack
puts the recording under each processing condition.

    uv run python eval/run_eval.py data/IMG_2625.wav --level-db -30
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from flyover_scrub.attack import locate, whitened_spectrogram
from flyover_scrub.physics import degrade_like_adsb, make_track, smooth_track, synthesize_flyover
from flyover_scrub.scrub import ScrubConfig, scrub


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    e = est - a * ref
    return float(10 * np.log10(np.dot(a * ref, a * ref) / (np.dot(e, e) + 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("background")
    ap.add_argument("--level-db", type=float, default=-30.0, help="flyover peak RMS vs background RMS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clock-error", type=float, default=2.5, help="attacker's start-time error (s)")
    ap.add_argument("--half-width", type=float, default=5000.0)
    ap.add_argument("--conditions", default="control,raw,highpass,scrub")
    ap.add_argument("--save-audio", default=None, help="directory to write processed wavs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bg, sr = sf.read(args.background)
    if bg.ndim == 2:
        bg = bg.mean(axis=1)
    n = len(bg)
    dur = n / sr

    # --- ground truth: observer at the recording's GPS fix (local origin) ---
    observer = np.array([0.0, 0.0, 0.0])
    T0 = 1_000_000.0  # recording start on the ADS-B clock
    # GA propeller aircraft, ~700 m AGL, gentle turn + slight acceleration
    trk = make_track(
        T0 - 120.0, dur + 240.0, start_xy=(-7600.0, -6800.0), altitude=700.0, speed=68.0,
        heading_deg=42.0, turn_rate_deg_s=-0.9, turn_start=95.0, turn_duration=30.0, accel=0.04,
    )
    d = np.linalg.norm(trk.pos - observer, axis=1)
    i_cpa = int(np.argmin(d))
    lat = float(np.linalg.norm(trk.pos[i_cpa, :2]))
    print(f"truth: CPA slant {d[i_cpa]:.0f} m (horizontal {lat:.0f} m) at t+{trk.t[i_cpa] - T0:.1f}s into clip")

    plane = synthesize_flyover(trk, observer, T0, n, sr, f0=92.0, rng=rng)
    plane *= np.sqrt(np.mean(bg**2)) / np.sqrt(np.mean(plane[np.abs(plane) > 0] ** 2) + 1e-12)
    plane *= 10 ** (args.level_db / 20)
    mix = bg + plane

    # --- attacker's view ---
    adsb = smooth_track(degrade_like_adsb(trk, rng))
    t0_guess = T0 + args.clock_error
    attack_kw = dict(f0_range=(86.0, 98.0), offsets=np.arange(-4.0, 4.01, 0.25))

    def hp(x):
        sos = butter(8, 300, btype="high", fs=sr, output="sos")
        return sosfiltfilt(sos, x)

    conds = {
        "control": lambda: (bg, {}),  # no aircraft at all
        "raw": lambda: (mix, {}),
        "highpass": lambda: (hp(mix), {}),
        "scrub": lambda: scrub(mix, sr, ScrubConfig(seed=args.seed)),
        "scrub_hum": lambda: scrub(mix, sr, ScrubConfig(seed=args.seed, keep_stationary_tones=True)),
    }
    results = {}
    for name in args.conditions.split(","):
        t_s = time.time()
        out = conds[name]()
        audio, rep = out if isinstance(out, tuple) else (out, {})
        if hasattr(rep, "replaced_fraction"):
            rep = {"replaced_fraction": round(rep.replaced_fraction, 3), "ridges": rep.ridges_found}
        spec = whitened_spectrogram(audio, sr)
        res = locate(spec, [adsb], t0_guess, (0.0, 0.0), half_width=args.half_width, **attack_kw)
        summ = res.summary(observer[:2])
        summ["si_sdr_vs_background_db"] = round(si_sdr(bg, audio), 2) if name != "control" else None
        summ.update(rep)
        summ["secs"] = round(time.time() - t_s, 1)
        results[name] = summ
        print(f"{name:9s} {json.dumps(summ)}", flush=True)
        if args.save_audio:
            import os

            os.makedirs(args.save_audio, exist_ok=True)
            sf.write(f"{args.save_audio}/{name}.wav", audio / max(1.0, np.abs(audio).max()), sr)
    print(json.dumps({"level_db": args.level_db, "results": results}))


if __name__ == "__main__":
    main()
