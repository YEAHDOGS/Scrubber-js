"""Window-match platform copies (aud/*.wav) against originals (orig/*.wav) by multi-band log-energy envelopes."""
import glob, os, json, numpy as np, soundfile as sf
from scipy.signal import stft
HOP = 0.05
BANDS = [(150, 400), (400, 1000), (1000, 2000), (2000, 3800)]
def feats(path):
    try:
        x, sr = sf.read(path)
    except Exception:
        return None
    if x.ndim > 1: x = x.mean(1)
    if len(x) < sr * 4: return None
    f, t, Z = stft(x, sr, nperseg=int(sr * 0.1), noverlap=int(sr * 0.05))
    P = np.abs(Z) ** 2
    E = np.stack([np.log(P[(f >= a) & (f < b)].sum(0) + 1e-10) for a, b in BANDS])
    E = E - np.convolve(E.mean(0), np.ones(41) / 41, "same")  # remove slow loudness drift (AGC / normalisation)
    return E
def ncc(a, b):
    """normalised xcorr of window b (bands x w) sliding over a (bands x n); returns per-lag r."""
    w = b.shape[1]; n = a.shape[1] - w + 1
    if n <= 0: return np.array([])
    bz = (b - b.mean(1, keepdims=True)); bz /= (np.linalg.norm(bz, axis=1, keepdims=True) + 1e-9)
    r = np.zeros(n)
    for k in range(a.shape[0]):
        ak = a[k]; cs = np.concatenate([[0], np.cumsum(ak)]); cs2 = np.concatenate([[0], np.cumsum(ak ** 2)])
        mu = (cs[w:] - cs[:-w]) / w; sd = np.sqrt(np.maximum((cs2[w:] - cs2[:-w]) - w * mu ** 2, 1e-9))
        r += np.correlate(ak, bz[k], "valid") / sd
    return r / a.shape[0]
O = {os.path.basename(p)[:-4]: feats(p) for p in sorted(glob.glob("orig/*.wav"))}
W = int(8 / HOP); STEP = int(4 / HOP)
res = []
done = set()
if os.path.exists("matches.json"):
    res = json.load(open("matches.json")); done = {r["uuid"] for r in res}
for p in sorted(glob.glob("aud/*.wav")):
    u = os.path.basename(p)[:-4]
    if u in done: continue
    C = feats(p); best = dict(uuid=u, orig=None, frac=0, lag=None, nwin=0, rmed=0)
    if C is not None and C.shape[1] >= W:
        for name, A in O.items():
            lags, rs = [], []
            for s in range(0, C.shape[1] - W + 1, STEP):
                r = ncc(A, C[:, s:s + W])
                if len(r) == 0: continue
                k = int(r.argmax()); lags.append((k - s) * HOP); rs.append(r[k])
            if not lags: continue
            lags = np.array(lags); rs = np.array(rs)
            med = np.median(lags[rs > 0.5]) if (rs > 0.5).any() else np.median(lags)
            ok = (np.abs(lags - med) < 0.3) & (rs > 0.5)
            frac = ok.mean()
            if frac > best["frac"]:
                best = dict(uuid=u, orig=name, frac=float(frac), lag=float(med), nwin=int(len(lags)), rmed=float(np.median(rs[ok])) if ok.any() else 0)
    res.append(best)
    if best["frac"] >= 0.4 and best["nwin"] >= 4 and best["rmed"] >= 0.6: print(f"MATCH {u} -> {best['orig']} frac {best['frac']:.2f} of {best['nwin']} windows, offset {best['lag']:+.1f}s, r {best['rmed']:.2f}", flush=True)
json.dump(res, open("matches.json", "w"))
print("checked", len(res))
