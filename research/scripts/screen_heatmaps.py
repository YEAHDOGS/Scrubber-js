"""Screen clips for close aircraft passes using adsb.lol heatmaps (10 s positions),
fetched with HTTP range requests from the daily tar (no full-day download)."""
import json, gzip, math, os, subprocess, sys, time, datetime as dt
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
clips = json.load(open(sys.argv[1]))
os.makedirs(f"{H}/cache", exist_ok=True)

_parts = {}
def parts(day):
    """[(url, size), ...] tar parts of that day's release (prod preferred)."""
    if day in _parts: return _parts[day]
    p = f"{H}/cache/{day}.parts.json"
    if os.path.exists(p): _parts[day] = json.load(open(p)); return _parts[day]
    y = day[:4]; d = day.replace('-', '.'); found = None
    for kind in ("prod", "staging"):
        for n in range(3):
            tag = f"v{d}-planes-readsb-{kind}-{n}"
            r = subprocess.run(["gh", "api", f"repos/adsblol/globe_history_{y}/releases/tags/{tag}", "--jq",
                                "[.assets[] | [.browser_download_url, .size]] | sort"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip().startswith("[["):
                found = json.loads(r.stdout); break
        if found: break
    if not found: raise IOError(f"no release for {day}")
    json.dump(found, open(p, "w")); _parts[day] = found; return found

def rng(day, a, n):
    """Bytes [a, a+n) of the concatenated tar parts."""
    out = b""; base = 0
    for u, size in parts(day):
        lo, hi = max(a, base), min(a + n, base + size)
        if lo < hi:
            for k in range(4):
                r = subprocess.run(["curl", "-sfL", "-r", f"{lo - base}-{hi - base - 1}", u], capture_output=True)
                if r.returncode == 0 and len(r.stdout) == hi - lo: break
                time.sleep(2 ** k)
            else:
                raise IOError(f"range failed {u}")
            out += r.stdout
        base += size
    return out

def hdr(b):
    if len(b) < 512 or b[257:262] != b"ustar": return None
    try:
        return b[:100].rstrip(b"\0").decode(), int(b[124:136].rstrip(b"\0 ") or b"0", 8)
    except ValueError:
        return None

def first_header_after(day, a, win=1 << 16):
    """Scan [a, a+win) at 512-aligned offsets for a tar header; return (off, name, size)."""
    a -= a % 512
    b = rng(day, a, win)
    for o in range(0, len(b) - 511, 512):
        h = hdr(b[o:o + 512])
        if h: return a + o, h[0], h[1]
    return None

class Walker:
    """Sequential tar header reader over HTTP ranges, buffering 256 KB per request."""
    def __init__(self, day, off):
        self.day, self.off, self.buf, self.base = day, off, b"", off
    def next(self):
        rel = self.off - self.base
        if rel < 0 or rel + 512 > len(self.buf):
            self.base = self.off; self.buf = rng(self.day, self.off, 1 << 18); rel = 0
        h = hdr(self.buf[rel:rel + 512])
        if not h: return None
        here = self.off
        self.off += 512 + ((h[1] + 511) // 512) * 512
        return here, h[0], h[1]

def walk_heatmaps(day, off):
    idx = {}; seen = False; w = Walker(day, off)
    while True:
        e = w.next()
        if not e: break
        o, name, size = e
        if "/heatmap/" in name and name.endswith(".bin.ttf"):
            idx[name.split("/")[-1].split(".")[0]] = [o + 512, size]; seen = True
        elif seen and "/heatmap/" not in name:
            break
    return idx

def index(day):
    p = f"{H}/cache/{day}.idx.json"
    if os.path.exists(p): return json.load(open(p))
    total = sum(s for _, s in parts(day))
    # Fast path: heatmaps near the start (newer archives).
    head = rng(day, 0, 1 << 20); off = 0
    while off + 512 <= len(head):
        h = hdr(head[off:off + 512])
        if not h: break
        name, size = h
        if "/heatmap/" in name or "/traces/" in name: break
        off += 512 + ((size + 511) // 512) * 512
    if "/heatmap/" in name:
        idx = walk_heatmaps(day, off)
    else:
        # Traces first: binary-search the end of the (dense, small-file) traces region,
        # probing 64 KB windows; inside big heatmap files there are no headers.
        lo, hi = off, total
        while hi - lo > (1 << 20):
            mid = (lo + hi) // 2
            h = first_header_after(day, mid, 1 << 18)
            if h and "/traces/" in h[1]: lo = h[0]
            else: hi = mid
            time.sleep(0.05)
        h = first_header_after(day, lo, 1 << 20)
        idx = walk_heatmaps(day, h[0]); hh = h
        if not idx:  # heatmaps may come before traces' end in some other order; give up cleanly
            raise IOError(f"heatmaps not found for {day} (after traces at {o}: {hh})")
    json.dump(idx, open(p, "w")); return idx

def chunk(day, k):
    p = f"{H}/cache/{day}_{k}.npy"
    if os.path.exists(p): return np.load(p)
    idx = index(day)
    key = next((x for x in (f"{k:02d}", str(k)) if x in idx), None)
    if key is None: return None
    a, n = idx[key]
    arr = np.frombuffer(gzip.decompress(rng(day, a, n)), dtype="<i4").reshape(-1, 4)
    # rows: (t, hex, lat, lon, alt_ft, gs_kt)
    out = []; t = None
    marks = np.where(arr[:, 0] == 0xE7F7C9D)[0]
    for j, m in enumerate(marks):
        t = ((int(arr[m, 1]) << 32) + (int(arr[m, 2]) & 0xFFFFFFFF)) / 1000
        e = arr[m + 1: marks[j + 1] if j + 1 < len(marks) else len(arr)]
        e = e[np.abs(e[:, 1]) < 90_000_000]
        ag = e[:, 3].copy().view("<i2").reshape(-1, 2).astype(float)
        out.append(np.column_stack([np.full(len(e), t), e[:, 0] & 0xFFFFFF, e[:, 1] / 1e6, e[:, 2] / 1e6, ag[:, 0] * 25, ag[:, 1] / 10]))
    res = np.concatenate(out) if out else np.zeros((0, 6))
    np.save(p, res); return res

rows = []
by_day = {}
for c in clips:
    by_day.setdefault(dt.datetime.fromtimestamp(c["t0"], dt.timezone.utc).strftime("%Y-%m-%d"), []).append(c)
for day in sorted(by_day):
    for c in by_day[day]:
        t0, t1 = c["t0"], c["t0"] + c["dur"]
        ks = sorted({int(((t - dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp()) // 1800)) for t in (t0 - 60, t1 + 60)})
        try:
            data = [chunk(day, k) for k in range(ks[0], ks[-1] + 1) if 0 <= k < 48]
        except Exception as e:
            print(f"{day} {c['fn']}: {e}", flush=True); continue
        data = [d for d in data if d is not None and len(d)]
        if not data: print(f"{day} {c['fn']}: no heatmap", flush=True); continue
        d = np.concatenate(data)
        d = d[(d[:, 0] >= t0 - 60) & (d[:, 0] <= t1 + 60)]
        dy = (d[:, 2] - c["lat"]) * 111195; dx = (d[:, 3] - c["lon"]) * 111195 * math.cos(math.radians(c["lat"]))
        r = np.hypot(dx, dy)
        passes = []
        for hx in np.unique(d[r < 6000, 1]):
            s = d[:, 1] == hx; i = np.argmin(np.where(s, r, 1e12))
            passes.append(dict(hex=f"{int(hx):06x}", cpa_m=float(r[i]), t_rel=float(d[i, 0] - t0), alt_ft=float(d[i, 4]), gs_kt=float(d[i, 5])))
        passes.sort(key=lambda p: p["cpa_m"])
        rows.append(dict(**c, day=day, passes=passes))
        good = [p for p in passes if p["cpa_m"] < 3000 and 15 < p["t_rel"] < c["dur"] - 15]
        print(f"{day} {c['fn']:14s} {c['dur']:5.0f}s  {len(passes)} within 6 km" + "".join(
            f"  | {p['hex']} {p['cpa_m'] / 1000:.1f}km @{p['t_rel']:.0f}s {p['alt_ft']:.0f}ft {p['gs_kt']:.0f}kt" for p in good), flush=True)
    json.dump(rows, open(f"{H}/screen.json", "w"))
