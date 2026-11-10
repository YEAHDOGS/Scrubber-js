"""Reproduce the band-clip proof of concept: a blind trace-fit attack over every aircraft in the log.

    uv run --with matplotlib python research/scripts/run_poc.py research/band-poc

For each aircraft in flights/, the traced ridge is fitted at every candidate
location in a 10 x 10 km box (the box centre is randomly offset from the truth,
so the grid gives no hint). The aircraft whose curve fits best is the
attacker's pick. The truth from recording.json is only used to score the result.
Pass --scrubbed to attack the flyover-scrub output instead.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from flyover_scrub.adsb import load_tracks
from flyover_scrub.attack import whitened_spectrogram
from flyover_scrub.cli import _read
from flyover_scrub.trace import locate_trace, reversal_ratio, trace_ridge

ap = argparse.ArgumentParser()
ap.add_argument("poc_dir", type=Path)
ap.add_argument("--scrubbed", action="store_true")
ap.add_argument("--band", default="33,68,55,84", help="t_lo,t_hi,f_lo,f_hi the attacker reads off the spectrogram")
ap.add_argument("--plot", type=Path)
a = ap.parse_args()

rec = json.loads((a.poc_dir / "recording.json").read_text())
name = rec["file"].replace(".m4a", "_scrubbed.m4a") if a.scrubbed else rec["file"]
import tempfile

with tempfile.TemporaryDirectory() as td:
    x, sr = _read(a.poc_dir / name, Path(td))
audio = resample_poly(x.mean(axis=1), 160, sr // 100)
t0 = datetime.fromisoformat(rec["start"]).timestamp()
tlo, thi, flo, fhi = map(float, a.band.split(","))

spec = whitened_spectrogram(audio, 16000, window_s=2.0)
ridge = trace_ridge(spec, f_range=(flo, fhi), t_range=(tlo, thi))
print(f"{name}: traced {len(ridge)} frames in {flo:g}-{fhi:g} Hz")
if len(ridge) < 20:
    print("no traceable line: nothing to fit")
    raise SystemExit

tracks = {}
for f in sorted((a.poc_dir / "flights").glob("*.jsonl")):
    for hx, d in load_tracks(f, rec["truth_lat"], rec["truth_lon"], rec["truth_alt_m"]).items():
        tracks[hx] = d
centre = tuple(np.random.default_rng(7).uniform(-2000, 2000, 2))
best = None
for hx, d in tracks.items():
    res = locate_trace(ridge, d["track"], t0, centre, half_width=5000)
    if not np.isfinite(res.scores).any():
        continue
    fit = -np.nanmax(res.scores)
    print(f"  {d['flight'] or hx:9s} {d['type'] or '?':5s} best fit {fit:6.1f} cents")
    if best is None or fit < best[0]:
        best = (fit, hx, d, res)
fit, hx, d, res = best
s = res.summary(np.zeros(2))
i = res.ranked()[0]
rr = reversal_ratio(ridge, d["track"], t0, res)
print(f"\nattacker's pick: {d['flight'] or hx} ({d['type']}), fit {fit:.1f} cents, f0 {res.best_f0[i]:.1f} Hz, "
      f"clock offset {res.best_offset[i]:+.2f} s, reversal ratio {rr:.1f}")
print(f"top guess {s['top_error_m']:.0f} m from the true spot; true spot ranked {s['truth_rank']}/{s['n_candidates']}")
print(f"right aircraft: {hx == rec['target_aircraft']['hex']}")

if a.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from flyover_scrub.physics import received_doppler

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    C, S = res.candidates, -res.scores
    m = np.isfinite(S)
    ax[0].scatter(C[m, 0], C[m, 1], c=-np.log(S[m]), s=5, cmap="viridis")
    ax[0].plot(0, 0, "r*", ms=15, label="true location")
    ax[0].plot(C[i, 0], C[i, 1], "wx", ms=12, mew=2, label="top guess")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("east (m)")
    ax[0].set_ylabel("north (m)")
    ax[0].legend()
    ax[0].set_title(f"fit quality for {d['flight'] or hx} (brighter = better)")
    ax[1].imshow(spec.S, aspect="auto", origin="lower", extent=[spec.times[0], spec.times[-1], spec.freqs[0], spec.freqs[-1]],
                 vmin=0, vmax=5, cmap="magma")
    ax[1].plot(ridge.t, ridge.f, "c.", ms=3, label="traced ridge")
    g, _, _ = received_doppler(d["track"], C[i], t0 + res.best_offset[i] + spec.times)
    ax[1].plot(spec.times, res.best_f0[i] * g, "w--", lw=1, label="model at top guess")
    ax[1].set_ylim(flo - 25, fhi + 25)
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("Hz")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(a.plot, dpi=70)
