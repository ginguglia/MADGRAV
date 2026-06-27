#!/usr/bin/env python
"""Provenance / regeneration script for the bundled GW190521 demo segment.

The COMMITTED demo data is the OUTPUT npz (demo/strain/GW190521_{H1,L1}.npz), NOT this
script's input. This script documents exactly how that small (~256 s) segment was carved
out of the full ~1 hr O3a cached strain, so the provenance is auditable. It reads from an
ABSOLUTE source path that only exists on the build machine -- on any other clone it simply
prints a notice and exits without doing anything (the bundled npz is already present).

What it does (when the source exists):
  - load search_mode/strain_o3a/GW190521_{H1,L1}.npz  (full ~1 hr, keys: strain, gps_start, fs)
  - locate the GW190521 merger sample = round((MERGER - gps_start)*fs)
  - slice a centred [merger-128 s, merger+128 s] window (256 s = 256*4096 samples)
  - write demo/strain/GW190521_{H1,L1}.npz with keys: strain (float32), gps_start (new
    window start GPS), fs (4096)
"""
import os, sys
import numpy as np

EV = "GW190521"
MERGER = 1242442967.4
HALF_S = 128.0                      # half-window in seconds -> 256 s total

SRC_DIR = "/scratch/ginguglia/GW/orchestrated-pipeline/search_mode/strain_o3a"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strain")


def main():
    srcH = os.path.join(SRC_DIR, f"{EV}_H1.npz")
    srcL = os.path.join(SRC_DIR, f"{EV}_L1.npz")
    if not (os.path.isfile(srcH) and os.path.isfile(srcL)):
        print(f"[make_demo_segment] source strain not found under {SRC_DIR}")
        print("[make_demo_segment] this is expected on a fresh clone -- the bundled")
        print(f"[make_demo_segment] demo/strain/{EV}_*.npz output is already committed; nothing to do.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for det, src in (("H1", srcH), ("L1", srcL)):
        f = np.load(src)
        strain = f["strain"]
        g0 = float(f["gps_start"])
        fs = float(f["fs"])
        assert abs(fs - 4096.0) < 1e-6, f"unexpected fs={fs}"
        fs = int(round(fs))
        merger_idx = int(round((MERGER - g0) * fs))
        half = int(round(HALF_S * fs))
        start_idx = merger_idx - half
        end_idx = merger_idx + half               # 256 s = 2*half samples
        assert start_idx >= 0 and end_idx <= len(strain), (
            f"window [{start_idx},{end_idx}) out of range for len {len(strain)}")
        # merger must sit comfortably inside the window
        assert half <= merger_idx - start_idx <= (end_idx - start_idx) - half + 1
        sl = strain[start_idx:end_idx].astype(np.float32)
        new_g0 = g0 + start_idx / fs
        out = os.path.join(OUT_DIR, f"{EV}_{det}.npz")
        np.savez(out, strain=sl, gps_start=np.float64(new_g0), fs=np.int64(fs))
        sz = os.path.getsize(out)
        total += sz
        moff = (MERGER - new_g0)
        print(f"[{det}] wrote {out}")
        print(f"      samples={len(sl)} ({len(sl)/fs:.0f}s)  new gps_start={new_g0:.3f}  "
              f"merger offset in window={moff:.3f}s  size={sz/1e6:.2f} MB")
    print(f"[make_demo_segment] total bundled demo strain = {total/1e6:.2f} MB")


if __name__ == "__main__":
    main()
