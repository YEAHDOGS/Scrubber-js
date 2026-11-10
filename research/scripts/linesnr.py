"""Line strength along the known Doppler path, per processing chain.
Path = ridge traced on the clean reference (all harmonics' path given by user-chosen band).
Metric: mean whitened power on the path vs median power 4-12 Hz either side (dB)."""
import sys, os, subprocess, numpy as np, soundfile as sf, tempfile
from scipy.signal import resample_poly
from scipy.ndimage import median_filter
from flyover_scrub.attack import whitened_spectrogram
clip, tlo, thi, flo, fhi = sys.argv[1], *map(float, sys.argv[2:6])
chains = [l.split("|") for l in open("ladder_chains.txt").read().strip().splitlines()]
def load(path):
    a, sr = sf.read(path); a = a if a.ndim == 1 else a.mean(1); return resample_poly(a, 16000, sr)
def path(sig):
    s = whitened_spectrogram(sig, 16000, window_s=2.0)
    fm = (s.freqs >= flo) & (s.freqs <= fhi); tm = (s.times >= tlo) & (s.times <= thi)
    S = s.S[np.ix_(fm, tm)]; k = S.argmax(0); fr = s.freqs[fm][k]
    keep = np.abs(fr - median_filter(fr, 9)) < 1.0
    return s.times[tm][keep], fr[keep]
from scipy.signal import stft
def strength(sig, T, F):
    f, t, Z = stft(sig, 16000, nperseg=32000, noverlap=28000); P = np.abs(Z) ** 2
    ti = np.clip(np.searchsorted(t, T), 0, len(t) - 1); df = f[1] - f[0]
    on, off = [], []
    for j, fr in zip(ti, F):
        i = int(round(fr / df)); on.append(P[i - 1:i + 2, j].max())
        side = np.r_[P[i - int(12 / df):i - int(4 / df), j], P[i + int(4 / df):i + int(12 / df), j]]
        off.append(np.median(side))
    return 10 * np.log10(np.median(np.array(on) / np.array(off)))
ref = load(f"aud/{clip}.wav"); T, F = path(ref)
print(f"{clip}: path {len(T)} frames {F.min():.0f}-{F.max():.0f} Hz")
tmp = tempfile.mkdtemp()
for name, opts, ext in chains:
    enc = f"{tmp}/x.{ext}"; dec = f"{tmp}/y.wav"
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f"aud/{clip}.wav", "-ac", "1", *opts.split(), enc]).returncode \
        or subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", enc, "-ac", "1", "-ar", "48000", dec]).returncode
    if r: print(f"  {name:28s} encode failed"); continue
    print(f"  {name:28s} line {strength(load(dec), T, F):5.1f} dB above neighbours", flush=True)
