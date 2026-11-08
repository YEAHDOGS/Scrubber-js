"""Quality of processed audio vs the aircraft-free original (phase-insensitive).

* content_err: dB error on bins that are real content in the original
  (>8 dB above the local noise floor: speech, actions, music).
* ltas_err:    max deviation of the long-term 1/3-octave spectrum (tonal balance).
Raw independent noise resynthesis is expected to be ~0 dB on ltas and small on content.
"""
import sys
import numpy as np, soundfile as sf
from scipy.signal import stft, welch
from scipy import ndimage

def content_err(ref, est, sr, band):
    f, _, A = stft(ref, sr, nperseg=1024); _, _, B = stft(est, sr, nperseg=1024)
    m = (f >= band[0]) & (f < band[1])
    a = 20*np.log10(np.abs(A[m]) + 1e-9); b = 20*np.log10(np.abs(B[m]) + 1e-9)
    floor = ndimage.uniform_filter1d(ndimage.median_filter(a, size=(15, 1)), 25, axis=1)
    c = a > floor + 8
    return float(np.median(np.abs(a - b)[c])), float(c.mean())

def ltas_err(ref, est, sr):
    f, P = welch(ref, sr, nperseg=4096); _, Q = welch(est, sr, nperseg=4096)
    edges = 25 * 2 ** (np.arange(0, 28) / 3)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.any() and hi < sr / 2:
            out.append(10*np.log10(Q[m].sum() / P[m].sum()))
    return float(np.max(np.abs(out)))

bg, sr = sf.read(sys.argv[1])
for name in sys.argv[3:]:
    x, _ = sf.read(f"{sys.argv[2]}/{name}.wav")
    pass
    ce, frac = content_err(bg, x, sr, (50, 2000))
    ce_hi, _ = content_err(bg, x, sr, (2000, 8000))
    print(f"{name:9s} content err <2k {ce:5.2f} dB  >2k {ce_hi:5.2f} dB  (content={frac:.0%} of bins)  LTAS max dev {ltas_err(bg, x, sr):5.2f} dB")
