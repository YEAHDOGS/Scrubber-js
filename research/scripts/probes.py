"""Paired platform measurement: steady tonal peaks in the TikTok-app local save, re-measured in the TikTok and Instagram versions."""
import numpy as np, soundfile as sf, subprocess, json, os, tempfile
from scipy.signal import stft
from scipy.ndimage import median_filter, uniform_filter1d
# local-save uuid -> (tiktok id, instagram shortcode). The 10 videos used are recent posts; their IDs are
# kept out of the repo. Provide your own as pairs.json: {"<uuid>": ["<tiktok id>", "<ig shortcode>"], ...}
pairs = {k: tuple(v) for k, v in json.load(open("pairs.json")).items()}
SR = 16000
def load(p, off=0.0):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", p, "-ac", "1", "-ar", str(SR), tmp], check=True)
    x, _ = sf.read(tmp); os.remove(tmp)
    k = int(round(-off * SR))  # shift so it aligns with local
    return np.r_[np.zeros(max(k, 0)), x[max(-k, 0):]]
def prom(x):
    f, t, Z = stft(x, SR, nperseg=SR, noverlap=SR // 2)  # 1 s window, 0.5 s hop, 1 Hz bins
    P = 10 * np.log10(np.abs(Z) ** 2 + 1e-14)
    fm = (f >= 20) & (f <= 1000); P = P[fm]; f = f[fm]
    nb = median_filter(P, size=(21, 1), mode="nearest")      # neighbourhood level +-10 Hz
    return f, t, P - nb
bands = [(20, 60), (60, 150), (150, 400), (400, 1000)]
rows = {"tiktok": [], "instagram": []}
for u, (tid, ig) in pairs.items():
    L = load(f"local/{u}.m4a"); f, t, R0 = prom(L)
    # probes: >=6 dB prominent and persistent >=5 frames (2.5 s) at that bin (+-1)
    pk = R0 >= 6
    pers = uniform_filter1d(np.maximum.reduce([np.roll(pk, s, 0) for s in (-1, 0, 1)]).astype(float), 5, axis=1) >= 0.99
    probe = pk & pers
    for name, path, off in (("tiktok", f"tiktok/recent/{tid}.m4a", 0.0), ("instagram", f"instagram/a/{ig}.m4a", -0.1)):
        X = load(path, off); n = min(len(L), len(X)); _, t2, R1 = prom(X[:n])
        m = min(R0.shape[1], R1.shape[1]); pr = probe[:, :m]
        fi, ti = np.nonzero(pr)
        for a, b in zip(fi, ti):
            rows[name].append((f[a], R0[a, b], R1[a, b]))
out = {}
for name, r in rows.items():
    r = np.array(r); print(f"\n{name}: {len(r)} probe points from {len(pairs)} videos")
    for lo, hi in bands:
        s = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
        if not len(s): continue
        faint = s[s[:, 1] < 12]
        d = s[:, 2] - s[:, 1]
        print(f"  {lo:4d}-{hi:<4d} Hz  n={len(s):6d}  median change {np.median(d):+5.1f} dB  (IQR {np.percentile(d,25):+.1f}..{np.percentile(d,75):+.1f})"
              f"  still >=6 dB: {np.mean(s[:,2] >= 6)*100:5.1f}%   faint(6-12 dB) still >=6: {np.mean(faint[:,2] >= 6)*100 if len(faint) else float('nan'):5.1f}%")
    out[name] = r.tolist()
json.dump(out, open("probes.json", "w"))
