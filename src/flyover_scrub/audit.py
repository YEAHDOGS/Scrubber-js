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
from .trace import locate_trace, reversal_ratio, trace_ridge

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


def trace_flight(audio, t0, trk, band, *, half_width, label, log):
    """Trace-fit attack on a band you can see in the spectrogram (t_lo, t_hi, f_lo, f_hi)."""
    tlo, thi, flo, fhi = band
    spec = whitened_spectrogram(audio, SR, window_s=2.0)
    ridge = trace_ridge(spec, f_range=(flo, fhi), t_range=(tlo, thi))
    if len(ridge) < 20:
        log(f"    {label:9s} trace-fit: no traceable line in {flo:g}-{fhi:g} Hz ({len(ridge)} frames)")
        return {"trace_frames": len(ridge), "trace_verdict": "no line"}
    res = locate_trace(ridge, trk, t0, (0.0, 0.0), half_width=half_width)
    if not np.isfinite(res.scores).any():
        return {"trace_frames": len(ridge), "trace_verdict": "no coverage"}
    s = res.summary(np.zeros(2))
    rr = reversal_ratio(ridge, trk, t0, res)
    fit = -s["top_score"]
    v = ("EXPOSED" if s["top_error_m"] <= 300 else "partly exposed" if s["top_error_m"] <= 1000 else "points elsewhere") \
        if rr >= 3 else "fits backwards too (chance)"
    log(f"    {label:9s} trace-fit: {len(ridge)} frames, fit {fit:.1f} cents, top guess {s['top_error_m']:.0f} m from you, "
        f"rank {s['truth_rank']}/{s['n_candidates']}, reversal ratio {rr:.1f}  -> {v}")
    return {"trace_frames": len(ridge), "trace_fit_cents": fit, "trace_top_error_m": s["top_error_m"],
            "trace_truth_rank": s["truth_rank"], "trace_reversal_ratio": rr, "trace_verdict": v}


def audit_flight(audio, t0, trk, *, half_width, label, log, trace=None):
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
    # Fundamental: sweep it on a 3 x 3 grid of spots spread over the search box and
    # keep the best, as an attacker without the answer would. (Sweeping it at the
    # true spot tuned f0 for the truth and inflated its rank, even on scrubbed or
    # time-reversed audio.)
    h = half_width / 2
    probes = np.array([[x, y, 0.0] for x in (-h, 0.0, h) for y in (-h, 0.0, h)])
    probes[:, :2] += np.random.default_rng(11).uniform(-h / 2, h / 2, (len(probes), 2))
    wide = score_candidates(spec, [trk], t_clip, probes, f0_range=(40.0, 400.0), f0_step=0.5, offsets=offsets)
    f0 = float(wide.best_f0[int(np.argmax(wide.scores))])
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
    real = strength >= 4.0  # measured: aircraft-free ~0.5, scrubbed ~1-2, inaudible synthetic flyover ~25
    near = s["top_error_m"] <= 300 or (s["truth_rank"] is not None and s["truth_rank"] <= 10)
    if real and s["top_error_m"] <= 300:
        s["verdict"] = "EXPOSED"
    elif real and s["truth_rank"] and s["truth_rank"] <= 50:
        s["verdict"] = "partly exposed"
    elif real:
        s["verdict"] = "signal, but points elsewhere"
    elif near and strength >= 2.0:
        # A real Cessna flyover scored ~2-3x chance here yet put the true spot first:
        # real lines are weaker for this scorer than synthetic ones. Err towards warning.
        s["verdict"] = "possibly exposed"
    else:
        s["verdict"] = "not located"
    if trace is not None:
        s.update(trace_flight(audio, t0, trk, trace, half_width=half_width, label=label, log=log))
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
    ap.add_argument("--trace", help="also run the trace-fit attack on a line you can see: T_LO,T_HI,F_LO,F_HI "
                    "(seconds from the start, Hz), e.g. 10,75,65,95")
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
            s = audit_flight(sig, t0, d["track"], half_width=a.half_width, label=label, log=print,
                             trace=tuple(map(float, a.trace.split(","))) if a.trace else None)
            if s:
                s.update({"hex": hx, **{k: d[k] for k in ("flight", "type", "category")}})
                results.append(s)
    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
    orig = [r for r in results if r["variant"] == "original"]
    worst = [r for r in orig if "EXPOSED" in (r["verdict"], r.get("trace_verdict"))]
    partly = [r for r in orig if r not in worst and {"partly exposed", "possibly exposed"} & {r["verdict"], r.get("trace_verdict")}]
    print(f"\n{len(worst)} flight(s) would have located the original recording; {len(partly)} more partly or possibly.")


if __name__ == "__main__":
    main()
