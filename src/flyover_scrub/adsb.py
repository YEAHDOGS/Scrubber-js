"""Gently log live ADS-B around a point you control, and turn the log into tracks.

This exists so you can audit your *own* recordings against *real* flyovers:
start a capture where you are, record audio, then run ``flyover-audit``.

Etiquette (the data is a free community service; please keep it that way):
  * one small point query at a time, default every 5 s, never faster than 2 s;
  * honest User-Agent with a project link;
  * backs off exponentially on errors / 429s and gives up after repeated failure;
  * stores only the few fields needed, for aircraft within your radius.
There is deliberately no bulk/historical download here.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .physics import Track, smooth_track

API = "https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{nm:.1f}"
USER_AGENT = "flyover-scrub/0.1 (+https://github.com/lizTheDeveloper/flyover-scrub)"
FIELDS = ("hex", "flight", "t", "category", "lat", "lon", "alt_geom", "alt_baro", "gs", "track", "seen_pos")
FT = 0.3048
MIN_INTERVAL_S = 2.0


def _get(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def capture(
    lat: float,
    lon: float,
    out: Path,
    *,
    radius_km: float = 15.0,
    minutes: float = 10.0,
    interval_s: float = 5.0,
    log=print,
) -> int:
    """Poll aircraft near (lat, lon) and append snapshots to ``out`` (JSONL). Returns snapshots written."""
    interval_s = max(MIN_INTERVAL_S, interval_s)
    url = API.format(lat=lat, lon=lon, nm=radius_km / 1.852)
    end = time.time() + minutes * 60
    backoff, failures, n = interval_s, 0, 0
    with open(out, "a") as fh:
        fh.write(json.dumps({"meta": {"lat": lat, "lon": lon, "radius_km": radius_km, "source": "api.adsb.lol"}}) + "\n")
        while time.time() < end:
            t_req = time.time()
            try:
                d = _get(url)
                ac = [{k: a.get(k) for k in FIELDS} for a in d.get("ac", []) if a.get("lat") is not None]
                fh.write(json.dumps({"now": d.get("now", t_req * 1000) / 1000.0, "ac": ac}) + "\n")
                fh.flush()
                n += 1
                failures, backoff = 0, interval_s
                log(f"{time.strftime('%H:%M:%S')}  {len(ac):3d} aircraft within {radius_km:g} km")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                failures += 1
                backoff = min(backoff * 2, 120.0)
                log(f"{time.strftime('%H:%M:%S')}  request failed ({e}); backing off {backoff:.0f}s")
                if failures >= 6:
                    log("too many failures; stopping")
                    break
            time.sleep(max(0.0, (backoff if failures else interval_s) - (time.time() - t_req)))
    return n


def enu(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local east/north metres (equirectangular; fine within a few tens of km)."""
    R = 6371008.8
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def load_tracks(log_path: Path, lat0: float, lon0: float, ground_alt_m: float = 0.0) -> dict[str, dict]:
    """Read a capture log into smoothed local tracks keyed by ICAO hex.

    Heights: ADS-B geometric altitude (ft, WGS84) minus your ground altitude. The
    geoid/ellipsoid difference (tens of metres) is small next to typical slant ranges.
    """
    pts: dict[str, list] = {}
    info: dict[str, dict] = {}
    for line in Path(log_path).read_text().splitlines():
        snap = json.loads(line)
        if "now" not in snap:
            continue
        for a in snap["ac"]:
            alt = a.get("alt_geom") if isinstance(a.get("alt_geom"), (int, float)) else a.get("alt_baro")
            if not isinstance(alt, (int, float)):
                continue  # "ground" or missing
            t = snap["now"] - float(a.get("seen_pos") or 0.0)
            pts.setdefault(a["hex"], []).append((t, a["lat"], a["lon"], alt * FT - ground_alt_m))
            info[a["hex"]] = {"flight": (a.get("flight") or "").strip(), "type": a.get("t"), "category": a.get("category")}
    tracks = {}
    for hx, p in pts.items():
        arr = np.array(sorted(set(p)))
        _, keep = np.unique(arr[:, 0], return_index=True)
        arr = arr[keep]
        if len(arr) < 6:
            continue
        x, y = enu(arr[:, 1], arr[:, 2], lat0, lon0)
        raw = Track(t=arr[:, 0], pos=np.stack([x, y, arr[:, 3]], axis=1))
        try:
            trk = smooth_track(raw, smoothing_s=12.0)
        except Exception:
            continue
        tracks[hx] = {"track": trk, **info[hx]}
    return tracks


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="flyover-capture",
        description="Gently log live aircraft around where you are, while you record. "
        "Polls api.adsb.lol (a free community service) every few seconds.",
    )
    ap.add_argument("--at", required=True, help="LAT,LON where you are recording")
    ap.add_argument("--radius-km", type=float, default=15.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--interval", type=float, default=5.0, help=f"seconds between requests (min {MIN_INTERVAL_S:g})")
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args(argv)
    lat, lon = map(float, a.at.split(","))
    n = capture(lat, lon, a.output, radius_km=a.radius_km, minutes=a.minutes, interval_s=a.interval)
    print(f"wrote {n} snapshots to {a.output}")
