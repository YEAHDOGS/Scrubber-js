"""flyover-scrub: remove aircraft Doppler signatures from audio and video files.

    flyover-scrub input.mov -o output.mov      # video: audio scrubbed, video stream copied
    flyover-scrub input.wav -o output.wav
    flyover-scrub --check input.mp4            # report long tonal ridges, write nothing

Video and compressed audio go through ffmpeg (must be on PATH). Metadata is
stripped from video outputs by default, since GPS tags are a far easier leak.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .scrub import ScrubConfig, scrub

NATIVE = {".wav", ".flac", ".aiff", ".aif", ".ogg"}


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg is required for this file type (brew install ffmpeg / apt install ffmpeg)")
    return exe


def _read(path: Path, tmp: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() in NATIVE:
        return sf.read(path, always_2d=True)
    wav = tmp / "in.wav"
    subprocess.run(
        [_ffmpeg(), "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-c:a", "pcm_f32le", str(wav)],
        check=True,
    )
    return sf.read(wav, always_2d=True)


def _has_video(path: Path) -> bool:
    if not shutil.which("ffprobe"):
        return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="flyover-scrub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--check", action="store_true", help="only report detected tonal ridges")
    ap.add_argument("--band", type=float, nargs=2, default=(20.0, 2000.0), metavar=("LO", "HI"))
    ap.add_argument("--keep-hum", action="store_true", help="keep steady tones (mains hum, appliances)")
    ap.add_argument("--keep-metadata", action="store_true", help="do not strip container metadata (GPS, dates)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)

    if not a.check and not a.output:
        ap.error("-o/--output is required unless --check")
    cfg = ScrubConfig(band=tuple(a.band), keep_stationary_tones=a.keep_hum, seed=a.seed)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        x, sr = _read(a.input, tmp)
        y, rep = scrub(x, sr, cfg)
        print(
            f"{a.input.name}: {x.shape[0] / sr:.1f}s @ {sr} Hz, {x.shape[1]} ch | "
            f"tonal ridges: {rep.ridges_found} | in-band bins replaced: {rep.replaced_fraction:.0%}"
        )
        if a.check:
            if rep.ridges_found:
                print("  long narrowband tones present (aircraft, drones, traffic, HVAC or hum).")
            print("  note: faint lines below the noise floor are not detectable per-frame but are still")
            print("  recoverable by a matched filter, so scrub anyway if location matters.")
            return

        peak = float(np.abs(y).max())
        if peak > 1.0:
            y = y / peak
        out = a.output
        if out.suffix.lower() in NATIVE:
            sf.write(out, y, sr, subtype="FLOAT" if out.suffix.lower() == ".wav" else None)
            print(f"wrote {out}")
            return

        wav = tmp / "scrubbed.wav"
        sf.write(wav, y, sr, subtype="FLOAT")
        meta = [] if a.keep_metadata else ["-map_metadata", "-1", "-map_chapters", "-1"]
        if _has_video(a.input):
            cmd = [_ffmpeg(), "-v", "error", "-y", "-i", str(a.input), "-i", str(wav),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                   *meta, "-shortest", str(out)]
        else:
            cmd = [_ffmpeg(), "-v", "error", "-y", "-i", str(wav), *meta, str(out)]
        subprocess.run(cmd, check=True)
        print(f"wrote {out}" + ("" if a.keep_metadata else " (metadata stripped)"))


if __name__ == "__main__":
    main()
