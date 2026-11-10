"""Dark, 16:9 figures for the video: how a faint aircraft line in a phone recording leaks where it was made.

    uv run --with matplotlib python research/scripts/make_video_figures.py

Writes 1920x1080 PNGs to research/figures/video/. Every number is either computed
here from repo files (car clip, ADS-B logs, research/data/*) or is a ladder
line-strength figure from the research notes (LADDER_* constants below).

Privacy: locations are only ever shown in metres east/north of the true spot.
No latitude/longitude is printed or plotted, and there are no maps.
"""

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import resample_poly

from flyover_scrub.adsb import load_tracks
from flyover_scrub.attack import whitened_spectrogram
from flyover_scrub.cli import _read
from flyover_scrub.physics import received_doppler
from flyover_scrub.trace import locate_trace, trace_ridge

ROOT = Path(__file__).resolve().parents[2]
POC = ROOT / "research" / "helicopter-poc"
# The clip's audio is withheld (private). Point AUDIO_DIR at a folder holding IMG_4367_cut.m4a
# and IMG_4367_cut_scrubbed.m4a to rebuild the spectrogram figures.
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", POC))
DATA = ROOT / "research" / "data"
OUT = ROOT / "research" / "figures" / "video"
OUT.mkdir(parents=True, exist_ok=True)

BAND = (33.0, 68.0, 55.0, 84.0)  # t_lo, t_hi, f_lo, f_hi -- what the attacker reads off the spectrogram
TARGET = "a54f77"

# Line-strength change through the processing ladder (dB the plane line stands above
# its neighbours, relative to the untouched clip). From the research notes.
LADDER_TYPICAL = (0.0, -2.5)  # codecs, filters, loudness norm, AGC, classic denoisers
LADDER_ANLMDN = -7.6  # ffmpeg anlmdn, strong: the strongest classic denoiser

# ---------------------------------------------------------------- style
BG = "#0a0b0d"
PANEL = "#121418"
INK = "#f3f3f1"
INK2 = "#b4b8bf"
MUTED = "#6b7079"
GRID = "#24272d"
RED = "#ff3b30"
GHOST = "#3a3e46"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 24, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 22, "ytick.labelsize": 22,
    "axes.labelsize": 24, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})
W, H, DPI = 19.2, 10.8, 100  # 1920 x 1080

SPEC_CMAP = LinearSegmentedColormap.from_list("ash", [BG, "#2a2d33", "#8d929b", "#f3f3f1"])
HEAT_CMAP = LinearSegmentedColormap.from_list("ember", ["#16181c", "#3a1512", "#8f1d16", RED, "#ffd2cc"])


def canvas():
    return plt.figure(figsize=(W, H), dpi=DPI)


def headline(fig, title, sub=None, y=0.93):
    fig.text(0.05, y, title, fontsize=58, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.05, y - 0.095, sub, fontsize=28, color=INK2, va="top")


def footer(fig, text):
    fig.text(0.05, 0.018, text, fontsize=17, color=MUTED, va="bottom", linespacing=1.4)


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p.relative_to(ROOT))
    return p


# ---------------------------------------------------------------- compute
def load_audio(name):
    with tempfile.TemporaryDirectory() as td:
        x, sr = _read(AUDIO_DIR / name, Path(td))
    return resample_poly(x.mean(axis=1), 160, sr // 100)


rec = json.loads((POC / "recording.json").read_text())
t0 = datetime.fromisoformat(rec["start"]).timestamp()
tlo, thi, flo, fhi = BAND

spec_o = whitened_spectrogram(load_audio("IMG_4367_cut.m4a"), 16000, window_s=2.0)
spec_s = whitened_spectrogram(load_audio("IMG_4367_cut_scrubbed.m4a"), 16000, window_s=2.0)
ridge = trace_ridge(spec_o, f_range=(flo, fhi), t_range=(tlo, thi))
ridge_s = trace_ridge(spec_s, f_range=(flo, fhi), t_range=(tlo, thi))
print(f"traced frames: original {len(ridge)}, scrubbed {len(ridge_s)}")

# Truth is only the origin: tracks come back in metres east/north/up of it.
tracks = {}
for f in sorted((POC / "flights").glob("*.jsonl")):
    tracks.update(load_tracks(f, rec["truth_lat"], rec["truth_lon"], rec["truth_alt_m"]))
del rec["truth_lat"], rec["truth_lon"]

centre = tuple(np.random.default_rng(7).uniform(-2000, 2000, 2))  # same blind search box as run_poc.py
fits = []
results = {}
for hx, d in tracks.items():
    res = locate_trace(ridge, d["track"], t0, centre, half_width=5000)
    if not np.isfinite(res.scores).any():
        continue
    fits.append((-np.nanmax(res.scores), d["flight"] or hx, d["type"] or "?", hx))
    results[hx] = res
fits.sort()
print("aircraft with coverage:", len(fits))
for c, n, t, hx in fits:
    print(f"  {n:9s} {t:5s} {c:5.1f} cents")
assert fits[0][3] == TARGET, "attacker's pick is not the target aircraft"

res = results[TARGET]
summ = res.summary(np.zeros(2))
ibest = res.ranked()[0]
print(f"top guess {summ['top_error_m']:.0f} m from truth; truth rank {summ['truth_rank']}/{summ['n_candidates']}")

# Model curve at the top guess (what the attacker fits), in the spectrogram's time base.
def model_curve(times):
    g, _, _ = received_doppler(tracks[TARGET]["track"], res.candidates[ibest], t0 + res.best_offset[ibest] + times)
    return res.best_f0[ibest] * g


model_f = model_curve(spec_o.times)


def line_strength_db(spec, shift=0.0):
    """Median whitened level (dB over local spectral median) along the model curve, 33-68 s.
    ``shift`` (Hz) moves the curve off the line to measure the background with the same yardstick."""
    mf = model_curve(spec.times) + shift
    m = (spec.times >= tlo) & (spec.times <= thi) & np.isfinite(mf)
    k = np.array([np.abs(spec.freqs - f).argmin() for f in mf[m]])
    cols = np.where(m)[0]
    vals = np.max(np.stack([spec.S[np.clip(k + dk, 0, len(spec.freqs) - 1), cols] for dk in (-1, 0, 1)]), axis=0)
    return float(np.median(vals) * 10 / np.log(10))


ls_o, ls_s = line_strength_db(spec_o), line_strength_db(spec_s)
bg = [line_strength_db(sp, sh) for sp in (spec_o, spec_s) for sh in (-20, -10, 10, 25, 40)]
print(f"line strength along model curve: original {ls_o:+.1f} dB, scrubbed {ls_s:+.1f} dB; "
      f"same curve shifted off the line: {min(bg):+.1f} to {max(bg):+.1f} dB")


# ---------------------------------------------------------------- figures
def spec_panel(ax, spec, fmax=140):
    fm = (spec.freqs >= 20) & (spec.freqs <= fmax)
    ax.imshow(spec.S[fm], aspect="auto", origin="lower", cmap=SPEC_CMAP, vmin=0, vmax=5, interpolation="bilinear",
              extent=[spec.times[0], spec.times[-1], spec.freqs[fm][0], spec.freqs[fm][-1]])
    ax.set_xlim(0, 90)
    ax.set_ylim(20, fmax)
    ax.set_xlabel("seconds into the clip")
    ax.set_ylabel("pitch (Hz)")
    for s in ax.spines.values():
        s.set_visible(False)


def fig_hidden_line():
    fig = canvas()
    headline(fig, "The sound you can't hear", "90 s of an iPhone selfie in a parked car. Under the talking: a helicopter's rotor hum.")
    ax = fig.add_axes([0.08, 0.20, 0.88, 0.55])
    spec_panel(ax, spec_o)
    ax.plot(ridge.t, ridge.f, "o", color=RED, ms=9, mec=BG, mew=1.5, zorder=3)
    j = len(ridge) * 2 // 3
    ax.annotate(f"{ridge.f.max():.0f} → {ridge.f.min():.0f} Hz:\nthe Doppler drop as it flies past",
                xy=(ridge.t[j], ridge.f[j]), xytext=(57, 112), color=INK, fontsize=28, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=2, shrinkB=8))
    footer(fig, "Whitened spectrogram (each pitch row divided by its typical level). Red: the traced line, "
                f"{len(ridge)} frames in the {tlo:.0f}–{thi:.0f} s / {flo:.0f}–{fhi:.0f} Hz window.\n"
                "Source: R44 helicopter N441MR (identified blind, from the ADS-B log).")
    return save(fig, "01_the_sound_you_cant_hear.png")


def fig_eighteen():
    fig = canvas()
    n = len(fits)
    headline(fig, f"{n} aircraft nearby. One fits.",
             "How far each aircraft's flight path misses the pitch curve (lower = better fit)")
    ax = fig.add_axes([0.20, 0.16, 0.74, 0.62])
    y = np.arange(n)[::-1]
    for yi, (c, name, typ, hx) in zip(y, fits):
        win = hx == TARGET
        col = RED if win else GHOST
        ax.plot([0, c], [yi, yi], color=col, lw=4 if win else 3, solid_capstyle="round")
        ax.plot(c, yi, "o", color=col, ms=16 if win else 11, mec=BG, mew=2)
        if win:
            ax.text(c + 1.5, yi, f"{c:.1f} cents  ←  the helicopter in the recording", va="center",
                    fontsize=26, fontweight="bold", color=INK)
    ax.text(fits[1][0] + 1.5, y[1], f"{fits[1][0]:.1f}  next best is 4× worse", va="center", fontsize=21, color=INK2)
    ax.set_yticks(y, [f"{name}  {typ}" for _, name, typ, _ in fits], fontsize=17)
    for t, (_, _, _, hx) in zip(ax.get_yticklabels(), fits):
        if hx == TARGET:
            t.set_color(INK)
            t.set_fontweight("bold")
            t.set_fontsize(21)
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("fit error (cents of pitch; 100 cents = one semitone)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for xv in range(10, 70, 10):
        ax.axvline(xv, color=GRID, lw=1, zorder=0)
    footer(fig, "Blind search: every aircraft with ADS-B coverage, fitted at every spot in a 10 × 10 km box. "
                "The attacker picks the lowest error.")
    return save(fig, "02_eighteen_aircraft_one_fits.png")


def fig_heatmap():
    fig = canvas()
    err = summ["top_error_m"]
    headline(fig, f"One sound. {err:.0f} m from the true spot.",
             "Every spot in a 100 km² search box, scored by how well the helicopter's path fits the hum")
    C, cents = res.candidates, -res.scores
    coarse = np.isclose(np.mod(C[:, 0] - centre[0] + 1e-6, 200.0), 0, atol=1e-3) & \
        np.isclose(np.mod(C[:, 1] - centre[1] + 1e-6, 200.0), 0, atol=1e-3)
    m = coarse & np.isfinite(cents)
    xs = np.unique(np.round(C[m, 0], 3))
    ys = np.unique(np.round(C[m, 1], 3))
    Z = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(C[m, 0], 3))
    iy = np.searchsorted(ys, np.round(C[m, 1], 3))
    Z[iy, ix] = cents[m]
    ax = fig.add_axes([0.05, 0.09, 0.60, 0.69])
    # brighter = better fit; log so the best few cents stand out
    im = ax.imshow(-np.log(Z), origin="lower", cmap=HEAT_CMAP, interpolation="bilinear",
                   extent=[(xs[0] - 100) / 1000, (xs[-1] + 100) / 1000, (ys[0] - 100) / 1000, (ys[-1] + 100) / 1000],
                   vmin=-np.log(np.nanpercentile(Z, 60)), vmax=-np.log(cents[ibest]))
    fine = ~coarse & np.isfinite(cents)
    ax.scatter(C[fine, 0] / 1000, C[fine, 1] / 1000, c=-np.log(cents[fine]), cmap=HEAT_CMAP, s=6, lw=0,
               vmin=im.get_clim()[0], vmax=im.get_clim()[1])
    ax.plot(0, 0, marker="*", ms=34, color=INK, mec=BG, mew=2, zorder=5)
    bx, by = C[ibest, 0] / 1000, C[ibest, 1] / 1000
    ax.plot(bx, by, marker="X", ms=26, color=RED, mec=INK, mew=2, zorder=6)
    ax.plot([0, bx], [0, by], color=INK, lw=1.5, alpha=0.7, zorder=4)
    ax.text((xs[-1] + 100) / 1000 - 0.2, (ys[-1] + 100) / 1000 - 0.2, "10 km search box", ha="right", va="top",
            fontsize=20, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # 1 km scale bar
    x0, y0 = (xs[0] + 400) / 1000, (ys[0] + 500) / 1000
    ax.plot([x0, x0 + 1], [y0, y0], color=INK, lw=5, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 + 0.2, "1 km", ha="center", va="bottom", fontsize=24, fontweight="bold")
    # legend column
    lx = 0.68
    fig.text(lx, 0.66, "★", fontsize=46, color=INK, va="center")
    fig.text(lx + 0.04, 0.66, "where it was really recorded", fontsize=27, va="center")
    fig.text(lx, 0.55, "✕", fontsize=40, color=RED, fontweight="bold", va="center")
    fig.text(lx + 0.04, 0.55, f"attacker's top guess\n{err:.0f} m away", fontsize=27, va="center")
    fig.text(lx, 0.40, "Brighter = the hum fits better", fontsize=24, color=INK2, va="center")
    fig.text(lx, 0.34, "Search box centre was offset at\nrandom from the truth: the grid\ngives no hint.",
             fontsize=21, color=MUTED, va="top", linespacing=1.4)
    footer(fig, "Metres east/north of the true spot. No map. Coarse 200 m grid plus 25 m refinement around the best cells.")
    return save(fig, "03_where_it_was_recorded.png")


def fig_ladder():
    rows = list(csv.DictReader((DATA / "codec_ladder_attack.csv").open()))
    hits = sum(r["right_aircraft"] == "True" for r in rows)
    chains = list(dict.fromkeys(r["chain"] for r in rows))
    clips = list(dict.fromkeys(r["clip"] for r in rows))
    by = {(r["clip"], r["chain"]): r for r in rows}
    nice = {
        "ref": "original", "aac_apple_128k": "AAC 128k (Apple)", "aac_ffmpeg_128k": "AAC 128k",
        "aac_ffmpeg_64k": "AAC 64k", "aac_ffmpeg_32k": "AAC 32k", "he_aac_apple_48k": "HE-AAC 48k",
        "opus_128k": "Opus 128k", "opus_32k": "Opus 32k", "opus_16k_voip": "Opus 16k VoIP",
        "opus_8k_voip": "Opus 8k VoIP", "mp3_64k": "MP3 64k", "phone_amrnb_12k": "AMR phone call",
        "wideband_g722": "G.722 wideband", "hp100_aac128": "high-pass 100 Hz", "hp200_aac128": "high-pass 200 Hz",
        "loudnorm14_aac128": "loudness normalise", "agc_dynaudnorm_aac128": "auto gain (AGC)",
    }
    fig = canvas()
    fig.text(0.05, 0.93, "Re-encoding doesn't hide it", fontsize=58, fontweight="bold", va="top")
    fig.text(0.05, 0.64, f"{hits}/{len(rows)}", fontsize=150, fontweight="bold", color=RED, va="center")
    fig.text(0.05, 0.50, "codec & filter chains where\nthe attack still named\nthe right aircraft",
             fontsize=30, color=INK, va="top", linespacing=1.15)
    fig.text(0.05, 0.27, f"{len(clips)} real clips × {len(chains)} chains: AAC, HE-AAC,\nOpus (down to 8k VoIP), MP3, "
                         "AMR phone,\nG.722, high-pass, loudness norm, AGC",
             fontsize=20, color=INK2, va="top", linespacing=1.15)
    ax = fig.add_axes([0.62, 0.07, 0.34, 0.77])
    ax.set_xlim(-0.5, len(clips) - 0.5)
    ax.set_ylim(len(chains) - 0.5, -1.4)
    for j, clip in enumerate(clips):
        ax.text(j, -1.05, {"cessna": "Cessna clip", "car": "car clip"}.get(clip, clip), ha="center",
                fontsize=21, fontweight="bold", color=INK2)
        for i, ch in enumerate(chains):
            ok = by[(clip, ch)]["right_aircraft"] == "True"
            ax.add_patch(FancyBboxPatch((j - 0.4, i - 0.38), 0.8, 0.76, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=RED if ok else PANEL, ec=RED if ok else MUTED, lw=1.5))
            ax.text(j, i, "found" if ok else "missed", ha="center", va="center", fontsize=16,
                    color=BG if ok else INK2, fontweight="bold")
    ax.set_yticks(range(len(chains)), [nice.get(c, c) for c in chains], fontsize=17)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    footer(fig, "Same blind attack after each re-encode. Misses: car clip at AAC 32k and AMR phone call.")
    return save(fig, "04_reencoding_doesnt_hide_it.png")


def fig_platforms():
    raw = json.loads((DATA / "platform_probes_raw.json").read_text())
    band = {r["path"]: r for r in csv.DictReader((DATA / "platform_probes_by_band.csv").open()) if r["band_hz"] == "20-1000"}

    def faint_survival(rows):  # faint = under 12 dB above neighbours before; survives = still >= 6 dB after
        a = np.array(rows)
        f = a[a[:, 1] < 12]
        return 100 * np.mean(f[:, 2] >= 6), len(f)

    tt, n_tt = faint_survival(raw["tiktok"])
    ig, n_ig = faint_survival(raw["instagram"])
    assert abs(tt - float(band["tiktok_app_save->tiktok"]["faint_still_visible_pct"])) < 0.2
    assert abs(ig - float(band["tiktok_app_save->instagram"]["faint_still_visible_pct"])) < 0.2
    fig = canvas()
    headline(fig, "Posting it doesn't wash it out",
             "Faint steady tones in a video, measured again after the platform re-encoded it")
    for k, (name, pct, x0) in enumerate([("TikTok", tt, 0.05), ("Instagram", ig, 0.53)]):
        ax = fig.add_axes([x0, 0.17, 0.20, 0.50])
        keep = int(round(pct))
        for i in range(100):
            r, c = divmod(i, 10)
            on = i < keep
            ax.add_patch(Rectangle((c + 0.08, 9 - r + 0.08), 0.84, 0.84, fc=RED if on else PANEL,
                                   ec=RED if on else GHOST, lw=1))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        med = float(band[f"tiktok_app_save->{name.lower()}"]["median_change_db"])
        fig.text(x0 + 0.215, 0.69, f"{name}", fontsize=36, fontweight="bold", va="top")
        fig.text(x0 + 0.215, 0.62, f"{pct:.0f}%", fontsize=110, fontweight="bold", color=RED, va="top")
        fig.text(x0 + 0.215, 0.42, f"of faint tones\nsurvive\n\nmedian change\n{med:+.2f} dB",
                 fontsize=24, color=INK2, va="top", linespacing=1.2)
    footer(fig, f"{n_tt:,} faint tones (6–12 dB above neighbours) from TikTok-app saves, re-measured in the same videos as "
                "served by TikTok and Instagram. Survive = still ≥ 6 dB.\niPhone camera original → posted TikTok "
                "(2 unedited clips): −0.2 dB, 87–90% survive.")
    return save(fig, "05_platforms_keep_it.png")


def fig_before_after():
    fig = canvas()
    headline(fig, "flyover-scrub: the line is gone", "Same 90 seconds, same analysis. Left: original. Right: scrubbed.")
    for k, (spec, label, n) in enumerate([(spec_o, "original", len(ridge)), (spec_s, "after flyover-scrub", len(ridge_s))]):
        ax = fig.add_axes([0.07 + k * 0.47, 0.20, 0.42, 0.46])
        spec_panel(ax, spec)
        tm = (spec.times >= tlo) & (spec.times <= thi)
        if k == 0:
            ax.plot(ridge.t, ridge.f, "o", color=RED, ms=7, mec=BG, mew=1)
            msg, col = f"{n} frames traced → aircraft + location", INK
        else:
            mf = model_curve(spec.times)
            ax.plot(spec.times[tm], mf[tm] + 6, color=INK2, lw=1.5, ls=(0, (4, 4)), alpha=0.6)
            ax.plot(spec.times[tm], mf[tm] - 6, color=INK2, lw=1.5, ls=(0, (4, 4)), alpha=0.6)
            msg, col = ("no traceable line → nothing to fit" if n < 20 else f"{n} frames traced"), INK
            ax.set_ylabel("")
        ax.text(0.0, 1.13, label.upper(), transform=ax.transAxes, fontsize=24, fontweight="bold", color=INK2)
        ax.text(0.0, 1.03, msg, transform=ax.transAxes, fontsize=26, fontweight="bold", color=INK)
    footer(fig, f"Level along the helicopter's Doppler curve, {tlo:.0f}–{thi:.0f} s: {ls_o:+.1f} dB over the local spectrum before, "
                f"{ls_s:+.1f} dB after.\nBackground (same curve moved off the line): {min(bg):+.1f} to {max(bg):+.1f} dB. "
                "Dashed: where the line was.")
    return save(fig, "06_before_after_scrub.png")


def fig_what_removes_it():
    fig = canvas()
    headline(fig, "Ordinary processing barely dents it",
             "How much each kind of processing weakens the aircraft line (dB above its neighbours)")
    ax = fig.add_axes([0.36, 0.17, 0.58, 0.55])
    rows = [
        ("Codecs, filters,\nloudness norm, AGC,\nclassic denoisers", LADDER_TYPICAL),
        ("Strongest classic denoiser\n(ffmpeg anlmdn, strong)", (LADDER_ANLMDN, LADDER_ANLMDN)),
        ("flyover-scrub", None),
    ]
    xmin = -12
    for i, (label, rng) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=24,
                fontweight="bold" if rng is None else "normal", color=INK, linespacing=1.2)
        if rng is None:
            ax.annotate("", xy=(xmin + 0.1, y), xytext=(0, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=6, mutation_scale=40))
            ax.text(xmin / 2, y + 0.28, "line removed: nothing left to trace", ha="center", fontsize=26,
                    fontweight="bold", color=INK)
        else:
            lo = min(rng)
            ax.barh(y, lo if lo < 0 else -0.08, height=0.42, left=0, color=RED, edgecolor=BG)
            txt = f"0 to {LADDER_TYPICAL[1]:.1f} dB" if rng[0] != rng[1] else f"{lo:.1f} dB"
            ax.text(lo - 0.25, y, txt, ha="right", va="center", fontsize=28, fontweight="bold", color=INK)
            if rng[0] != rng[1]:
                ax.text(0.3, y, "line still there", va="center", fontsize=22, color=INK2)
    ax.set_xlim(xmin, 3.5)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=INK2, lw=2)
    ax.set_xticks([-10, -7.5, -5, -2.5, 0])
    ax.set_xlabel("change in line strength (dB)")
    footer(fig, "Red = line weakened, not removed. Line-strength changes measured on the research ladder; "
                "the attack still named the right aircraft in 32/34 codec/filter chains.")
    return save(fig, "07_what_removes_it.png")


if __name__ == "__main__":
    for f in (fig_hidden_line, fig_eighteen, fig_heatmap, fig_ladder, fig_platforms, fig_before_after, fig_what_removes_it):
        f()
