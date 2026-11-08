"""Exposure audit: could the Doppler attack locate *this* recording of *yours*?

You supply where you actually were (or the file's own GPS tag) and a flight log
captured with ``flyover-capture``. For every aircraft that passed within range
while you recorded, we run the attack in a box around your true position and
report whether it would have found you, before and (optionally) after scrubbing.

It needs your true location by design: this answers "am I exposed?", it is
not a tool for locating recordings you did not make.

    flyover-capture --at 42.36,-83.13 --minutes 15 -o flights.jsonl   # while recording
    flyover-audit recording.mov --flights flights.jsonl --scrub
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .adsb import load_tracks
from .attack import Spectrogram, locate, score_candidates, whitened_spectrogram
from .physics import received_doppler
from .scrub import ScrubConfig, scrub

SR = 16000


def file_metadata(path: Path) -> dict:
    """Recording start time and GPS from QuickTime/MP4 tags, when present."""
    out = subprocess.run(["ffprobe", "-hide_banner", str(path)], capture_output=True, text=True).stderr
    meta = {}
    m = re.search(r"com\.apple\.quicktime\.creationdate\s*:\s*(\S+)", out)
    if m:
        meta["start"] = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    m = re.search(r"ISO6709\s*:\s*([+-][\d.]+)([+-][\d.]+)([+-][\d.]+)?", out)
    if m:
        meta["lat"], meta["lon"] = float(m.group(1)), float(m.group(2))
        if m.group(3):
            meta["alt"] = float(m.group(3))
    return meta


def audit_flight(audio, t0, trk, *, half_width, label, log):
    """Attack one flight. Returns a summary dict (truth = local origin)."""
    obs = np.zeros(3)
    # crop to +-75 s around when the truth location hears closest approach
    t_axis = t0 + np.arange(0, len(audio) / SR, 0.5)
    g, r, _ = received_doppler(trk, obs, t_axis)
    if not np.isfinite(r).any():
        return None
    i = int(np.nanargmin(r))
    t_cpa = t_axis[i]
    a = max(0, int((t_cpa - t0 - 75) * SR))
    b = min(len(audio), int((t_cpa - t0 + 75) * SR))
    if (b - a) / SR < 40:
        return None
    clip = audio[a:b]
    spec = whitened_spectrogram(clip, SR)
    t_clip = t0 + a / SR
    offsets = np.arange(-4.0, 4.01, 0.25)
    # fundamental: an attacker reads it off the visible line; we sweep it at truth
    wide = score_candidates(spec, [trk], t_clip, obs[None, :], f0_range=(40.0, 400.0), f0_step=0.5, offsets=offsets)
    f0 = float(wide.best_f0[0])
    res = locate(spec, [trk], t_clip, (0.0, 0.0), half_width=half_width,
                 f0_range=(f0 - 3, f0 + 3), offsets=offsets)
    s = res.summary(np.zeros(2))
    # Null: the same search on the time-reversed spectrogram. It keeps every
    # statistic of the recording, but aircraft lines now rise instead of fall, so
    # a modelled flyover can only match by chance. Compare best-of-grid to
    # best-of-grid so the max-over-many-candidates bias cancels.
    rev = Spectrogram(S=spec.S[:, ::-1].copy(), freqs=spec.freqs, times=spec.times)
    null = locate(rev, [trk], t_clip, (0.0, 0.0), half_width=half_width,
                  f0_range=(f0 - 3, f0 + 3), offsets=offsets).scores
    null = null[np.isfinite(null)]
    med = float(np.median(null))
    strength = (s["top_score"] - med) / max(float(null.max()) - med, 1e-9)
    s.update({"cpa_horizontal_m": float(np.nanmin(r)), "f0_hz": f0, "variant": label,
              "strength_vs_chance": float(strength)})
    real = strength >= 4.0  # measured: aircraft-free ~0.5, scrubbed ~2, inaudible flyover ~25
    if real and s["top_error_m"] <= 300:
        s["verdict"] = "EXPOSED"
    elif real and s["truth_rank"] and s["truth_rank"] <= 50:
        s["verdict"] = "partly exposed"
    elif real:
        s["verdict"] = "signal, but points elsewhere"
    else:
        s["verdict"] = "not located"
    log(f"    {label:9s} top guess {s['top_error_m']:6.0f} m from you, your spot ranked "
        f"{s['truth_rank']}/{s['n_candidates']}, signal {strength:4.1f}x chance  -> {s['verdict']}")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flyover-audit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--flights", type=Path, required=True, help="log from flyover-capture")
    ap.add_argument("--at", help="your true LAT,LON (default: the file's GPS tag)")
    ap.add_argument("--ground-alt", type=float, help="your altitude in m (default: file tag or 0)")
    ap.add_argument("--start", help="recording start, ISO 8601 with offset (default: file tag)")
    ap.add_argument("--max-cpa-km", type=float, default=8.0, help="only audit aircraft passing this close")
    ap.add_argument("--half-width", type=float, default=3000.0, help="search box half-width around you (m)")
    ap.add_argument("--scrub", action="store_true", help="also audit the scrubbed audio")
    ap.add_argument("--json", type=Path, help="write results here")
    a = ap.parse_args(argv)

    meta = file_metadata(a.recording)
    if a.at:
        lat, lon = map(float, a.at.split(","))
    elif "lat" in meta:
        lat, lon = meta["lat"], meta["lon"]
    else:
        ap.error("no GPS tag in file; pass --at LAT,LON (where you actually were)")
    t0 = datetime.fromisoformat(a.start).timestamp() if a.start else meta.get("start")
    if t0 is None:
        ap.error("no recording start time in file; pass --start 2026-11-07T17:03:12-05:00")
    ground = a.ground_alt if a.ground_alt is not None else meta.get("alt", 0.0)

    from .cli import _read

    with tempfile.TemporaryDirectory() as td:
        x, sr = _read(a.recording, Path(td))
    mono = x.mean(axis=1)
    from math import gcd

    gg = gcd(SR, sr)
    audio = resample_poly(mono, SR // gg, sr // gg)
    dur = len(audio) / SR
    print(f"{a.recording.name}: {dur:.0f}s, start {datetime.fromtimestamp(t0).isoformat()}, at {lat:.4f},{lon:.4f}")

    tracks = load_tracks(a.flights, lat, lon, ground)
    near = []
    for hx, d in tracks.items():
        trk = d["track"]
        span = (trk.t >= t0 - 120) & (trk.t <= t0 + dur + 120)
        if span.sum() < 10:
            continue
        cpa = float(np.min(np.linalg.norm(trk.pos[span, :2], axis=1)))
        if cpa <= a.max_cpa_km * 1000:
            near.append((cpa, hx, d))
    near.sort(key=lambda z: z[0])
    print(f"{len(tracks)} aircraft in log, {len(near)} passed within {a.max_cpa_km:g} km while recording")
    if not near:
        print("nothing to test: no logged aircraft passed close enough during the recording")
        return

    scrubbed = None
    if a.scrub:
        scrubbed, rep = scrub(audio, SR, ScrubConfig(seed=0))
        print(f"scrubbed: {rep.ridges_found} tonal ridges, {rep.replaced_fraction:.0%} of in-band bins resynthesised")

    results = []
    for cpa, hx, d in near:
        print(f"  {d['flight'] or hx} ({d['type'] or '?'}, cat {d['category'] or '?'}) closest {cpa / 1000:.1f} km")
        for label, sig in [("original", audio)] + ([("scrubbed", scrubbed)] if scrubbed is not None else []):
            s = audit_flight(sig, t0, d["track"], half_width=a.half_width, label=label, log=print)
            if s:
                s.update({"hex": hx, **{k: d[k] for k in ("flight", "type", "category")}})
                results.append(s)
    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
    worst = [r for r in results if r["variant"] == "original" and r["verdict"] == "EXPOSED"]
    print(f"\n{len(worst)} flight(s) would have located the original recording.")


if __name__ == "__main__":
    main()
