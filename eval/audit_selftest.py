"""Exercise flyover-audit end to end with a synthetic flight log in adsb.lol format.

    uv run python eval/audit_selftest.py data/IMG_2625.wav
"""
import json, sys, math
from pathlib import Path
import numpy as np, soundfile as sf
from flyover_scrub.physics import make_track, synthesize_flyover
from flyover_scrub.adsb import FT
from flyover_scrub.audit import main as audit

LAT0, LON0, T0 = 41.90, -87.70, 1_790_000_000.0   # neutral point, arbitrary epoch
bg, sr = sf.read(sys.argv[1]); bg = bg if bg.ndim == 1 else bg.mean(1)
rng = np.random.default_rng(7)
trk = make_track(T0 - 150, len(bg) / sr + 300, start_xy=(-8200.0, -6100.0), altitude=750.0, speed=65.0,
                 heading_deg=38.0, turn_rate_deg_s=1.1, turn_start=110.0, turn_duration=25.0, accel=0.05)
plane = synthesize_flyover(trk, np.zeros(3), T0, len(bg), sr, f0=104.0, rng=rng)
plane *= 10 ** (-28 / 20) * bg.std() / plane[plane != 0].std()
out = Path("out/selftest"); out.mkdir(parents=True, exist_ok=True)
sf.write(out / "rec.wav", bg + plane, sr)

R = 6371008.8
with open(out / "flights.jsonl", "w") as fh:
    fh.write(json.dumps({"meta": {"lat": LAT0, "lon": LON0}}) + "\n")
    for now in np.arange(trk.t[0] + 3, trk.t[-1] - 3, 5.0):        # one poll every 5 s
        seen = rng.uniform(0, 2.0)                                    # position age
        p = trk.resample(np.array([now - seen])).pos[0] + rng.normal(0, 6, 3) * [1, 1, 0.5]
        lat = LAT0 + math.degrees(p[1] / R)
        lon = LON0 + math.degrees(p[0] / (R * math.cos(math.radians(LAT0))))
        ac = [{"hex": "abc123", "flight": "N123AB", "t": "C172", "category": "A1", "lat": lat, "lon": lon,
               "alt_geom": round(p[2] / FT), "alt_baro": None, "gs": 126, "track": 50, "seen_pos": seen}]
        fh.write(json.dumps({"now": float(now), "ac": ac}) + "\n")

start = __import__("datetime").datetime.fromtimestamp(T0).astimezone().isoformat()
audit([str(out / "rec.wav"), "--flights", str(out / "flights.jsonl"), "--at", f"{LAT0},{LON0}",
       "--ground-alt", "0", "--start", start, "--scrub", "--json", str(out / "result.json")])
