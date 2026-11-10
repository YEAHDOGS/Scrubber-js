import numpy as np, soundfile as sf, subprocess, os, tempfile, sys
from scipy.signal import stft
from scipy.ndimage import median_filter, uniform_filter1d
SR = 16000
def load(p):
    tmp = tempfile.mktemp(suffix=".wav"); subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-i",p,"-ac","1","-ar",str(SR),tmp],check=True)
    x,_ = sf.read(tmp); os.remove(tmp); return x
def prom(x):
    f,t,Z = stft(x,SR,nperseg=SR,noverlap=SR//2); P = 10*np.log10(np.abs(Z)**2+1e-14)
    fm=(f>=20)&(f<=1000); P=P[fm]; return f[fm], P - median_filter(P,size=(21,1),mode="nearest")
# (original wav, tiktok m4a, lag: tiktok time = orig time - lag  i.e. orig[t] == tik[t - lag])
P = [("IMG_3318","7244654039332326698",0.0),("IMG_0207","<withheld>",0.0),("IMG_0027","7426376969249410346",-101.7)]
allr=[]
for o,tk,lag in P:
    A = load(f"../hist/aud/{o}.wav"); B = load(f"tiktok/a/{tk}.m4a")
    s = int(round(-lag*SR))  # tiktok sample index where original starts
    if s>0: B = B[s:]
    else: A = A[-s:]
    n = min(len(A),len(B)); A, B = A[:n], B[:n]
    f, R0 = prom(A); _, R1 = prom(B)
    pk = R0>=6; pers = uniform_filter1d(np.maximum.reduce([np.roll(pk,k,0) for k in (-1,0,1)]).astype(float),5,axis=1)>=0.99
    fi,ti = np.nonzero(pk&pers); r = np.c_[f[fi], R0[fi,ti], R1[fi,ti]]; allr.append(r)
    d = r[:,2]-r[:,1]; faint = r[r[:,1]<12]
    print(f"{o} -> TikTok ({n/SR:.0f} s overlap): {len(r)} probes, median change {np.median(d):+.1f} dB (IQR {np.percentile(d,25):+.1f}..{np.percentile(d,75):+.1f}), faint still >=6 dB: {np.mean(faint[:,2]>=6)*100:.0f}%")
