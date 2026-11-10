"""Blind trace-fit attack (colsto's method for faint curves).
1. Trace the visible ridge: per frame, peak inside a band, kept only where it is strong and continuous.
2. For each candidate location and clock offset, model g(t); best f0 is solved in closed form
   (log f_obs = log f0 + log g), score = RMS log-residual (in cents). Lower is better.
usage: trace_attack.py NAME LOG t_lo t_hi f_lo f_hi [half_width]"""
import sys, os, re, json, numpy as np, soundfile as sf, datetime as dt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.ndimage import median_filter
from flyover_scrub.attack import whitened_spectrogram, grid
from flyover_scrub.adsb import load_tracks
from flyover_scrub.physics import received_doppler
from flyover_scrub.scrub import scrub, ScrubConfig
name, log = sys.argv[1:3]; tlo, thi, flo, fhi = map(float, sys.argv[3:7]); hw = float(sys.argv[7]) if len(sys.argv) > 7 else 5000
meta = open(f"aud/{name}.meta").read()
lat, lon, alt = map(float, re.search(r"ISO6709: ([+-][\d.]+)([+-][\d.]+)([+-][\d.]+)", meta).groups())
t0 = dt.datetime.strptime(re.search(r"creationdate: (\S+)", meta).group(1), "%Y-%m-%dT%H:%M:%S%z").timestamp()
a, sr = sf.read(f"aud/{name}.wav"); a = resample_poly(a, 1, 3)
import glob as _g
TR = {}
for lg in log.split(","):
    for k, d in load_tracks(lg, lat, lon, alt).items(): TR[k] = d["track"]
trks = list(TR.values())

def trace(sig):
    s = whitened_spectrogram(sig, 16000, window_s=2.0)
    fm = (s.freqs >= flo) & (s.freqs <= fhi); tm = (s.times >= tlo) & (s.times <= thi)
    S = s.S[np.ix_(fm, tm)]; f = s.freqs[fm]; t = s.times[tm]
    k = S.argmax(0); pk = S.max(0)
    if os.environ.get("FOLLOW"):
        # Follow one ridge from the strongest point: per frame, local max within +-w bins of the previous.
        w = max(1, int(round(float(os.environ.get("FOLLOW_HZ", "0.6")) / (f[1] - f[0]))))
        j0 = int(pk.argmax()); k = np.zeros(len(t), int); k[j0] = S[:, j0].argmax()
        for rng_ in (range(j0 + 1, len(t)), range(j0 - 1, -1, -1)):
            prev = k[j0]
            for j in rng_:
                lo_, hi_ = max(prev - w, 0), min(prev + w + 1, len(f))
                k[j] = lo_ + S[lo_:hi_, j].argmax(); prev = k[j]
        pk = S[k, range(len(t))]
    # parabolic sub-bin interpolation
    kk = np.clip(k, 1, len(f) - 2); y0, y1, y2 = S[kk - 1, range(len(t))], S[kk, range(len(t))], S[kk + 1, range(len(t))]
    d = 0.5 * (y0 - y2) / np.where(np.abs(y0 - 2 * y1 + y2) > 1e-9, y0 - 2 * y1 + y2, 1e-9)
    fr = f[kk] + np.clip(d, -0.5, 0.5) * (f[1] - f[0])
    smooth = median_filter(fr, 9)
    keep = (pk > float(os.environ.get("PK", "2.5"))) & (np.abs(fr - smooth) < 1.5)
    return t[keep], fr[keep], s

def score(tt, ff, cands, offs, trk):
    lf = np.log(ff); res = np.full(len(cands), np.inf); bf = np.zeros(len(cands)); bo = np.zeros(len(cands))
    for i, c in enumerate(cands):
        tr = t0 + offs[:, None] + tt[None, :]
        g, r, _ = received_doppler(trk, c, tr.ravel()); g = g.reshape(tr.shape)
        lg = np.log(np.where(np.isfinite(g), g, np.nan))
        lf0 = np.nanmean(lf[None, :] - lg, axis=1)
        rms = np.sqrt(np.nanmean((lf[None, :] - lg - lf0[:, None]) ** 2, axis=1))
        rms = np.where(np.isfinite(lg).sum(1) > 0.8 * len(tt), rms, np.inf)
        j = np.nanargmin(rms); res[i] = rms[j]; bf[i] = np.exp(lf0[j]); bo[i] = offs[j]
    return res * 1731.2, bf, bo   # cents

rng = np.random.default_rng(7); centre = rng.uniform(-2000, 2000, 2)
offs = np.arange(-8, 8.01, 0.25)
figs = []
for label, sig in ([("original", a)] if os.environ.get("NOSCRUB") else [("original", a), ("scrubbed", scrub(a, 16000, ScrubConfig(seed=0))[0])]):
    tt, ff, s = trace(sig)
    if len(tt) < 20:
        print(f"{label:9s} no traceable line ({len(tt)} frames)"); continue
    best = None
    for hx, trk in TR.items():
        c = grid(tuple(centre), hw, 200.0)
        sc, bf, bo = score(tt, ff, c, offs, trk)
        if not np.isfinite(sc).any():
            print(f"   {hx}: no coverage"); continue
        seeds = c[np.argsort(sc)[:15]]
        fine = np.unique(np.round(np.concatenate([grid(tuple(p[:2]), 200, 25.0) for p in seeds]), 1), axis=0)
        sc2, bf2, bo2 = score(tt, ff, fine, offs, trk)
        Ck = np.concatenate([c, fine]); Sk = np.concatenate([sc, sc2])
        print(f"   {hx}: best fit {np.nanmin(Sk):6.1f} cents at {np.linalg.norm(Ck[np.argmin(Sk), :2]):5.0f} m from truth", flush=True)
        if best is None or np.nanmin(Sk) < np.nanmin(best[1]):
            best = (Ck, Sk, np.concatenate([bf, bf2]), np.concatenate([bo, bo2]), hx, trk)
    C, S_, F, O, besthx, besttrk = best
    print(f"   -> best flight {besthx}")
    order = np.argsort(S_); err = np.linalg.norm(C[:, :2], axis=1)
    rank = int(np.argmax(err[order] <= 100)) + 1
    i = order[0]
    print(f"{label:9s} traced {len(tt)} frames; top guess {err[i]:5.0f} m from truth (fit {S_[i]:.1f} cents, f0 {F[i]:.1f} Hz, offset {O[i]:+.2f}s); "
          f"truth (within 100 m) rank {rank}/{len(C)}; best fit at truth {(S_[err <= 100].min() if (err <= 100).any() else float("nan")):.1f} cents; median {np.median(S_[np.isfinite(S_)]):.1f}")
    figs.append((label, C, S_, tt, ff, s, i, F, O)); TRK = besttrk
fig, ax = plt.subplots(len(figs), 2, figsize=(14, 6 * len(figs)), squeeze=False)
for row, (label, C, S_, tt, ff, s, i, F, O) in zip(ax, figs):
    m = np.isfinite(S_); row[0].scatter(C[m, 0], C[m, 1], c=-np.log(S_[m]), s=5, cmap="viridis")
    row[0].plot(0, 0, "r*", ms=15, label="true location"); row[0].plot(C[i, 0], C[i, 1], "wx", ms=12, mew=2, label="top guess")
    row[0].set_aspect("equal"); row[0].legend(); row[0].set_title(f"{name} {label}: fit quality (brighter = better)")
    row[1].imshow(s.S, aspect="auto", origin="lower", extent=[s.times[0], s.times[-1], s.freqs[0], s.freqs[-1]], vmin=0, vmax=5, cmap="magma")
    row[1].plot(tt, ff, "c.", ms=3, label="traced")
    g, _, _ = received_doppler(TRK, C[i], t0 + O[i] + s.times); row[1].plot(s.times, F[i] * g, "w--", lw=1, label="model at top guess")
    row[1].set_ylim(flo - 25, fhi + 25); row[1].legend()
plt.tight_layout(); plt.savefig(f"trace_{name}.png", dpi=55)
