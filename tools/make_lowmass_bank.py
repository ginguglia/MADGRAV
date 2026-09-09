#!/usr/bin/env python
"""Generate a LOW-MASS projected signal bank (source-frame Mtot 10-22) in the exact
convention of data/o1_o3_signal_bank_projected_2s_x10, so it can be used as a third
injection stratum without biasing the comparison against the existing 20-400 coverage.

Conventions reproduced (all verified against the existing bank, 2026-09-04):
  approximant     IMRPhenomPv2, NON-SPINNING (the existing CSV carries no spin columns)
  sample rate     4096 Hz, f_lower 10 Hz
  distance        UNIFORM IN DISTANCE over [100, 5000] Mpc  (measured: counts per
                  equal-width distance bin are flat; it is a reference distance, not a
                  population prior -- the population weight is the rho^-4 dRho grid)
  orientation     cos(inclination) ~ U(-1,1); coa_phase, polarization ~ U(0,2pi)
  sky             ra ~ U(0,2pi); sin(dec) ~ U(-1,1)
  projection      h = F_plus h_plus + F_cross h_cross per detector at REF_GPS
  crop            8192 samples (2 s) centred on EACH DETECTOR'S OWN |h| peak, i.e. the
                  inter-detector arrival delay is REMOVED (existing bank: both stored
                  channels peak at sample 4096 while the CSV peak indices differ by up
                  to 33 samples). The coherence statistic maxes over +/-45 samples, so
                  this is consistent, but it must be reproduced exactly.
  layout          signals_NNNN.npz {H1, L1} float32 (n, 8192) + signals_NNNN.csv with
                  the same 20 columns, 1024 sources per shard.

Run with the pycbc environment:
  python tools/make_lowmass_bank.py --out DIR \
      [--n 10240] [--shard 1024] [--mtot-lo 10 --mtot-hi 22] [--seed 20260904]
"""
import argparse
import csv
import os

import numpy as np

FS = 4096
NCROP = 8192                     # 2 s, peak-centred
F_LOWER = 10.0
APX = "IMRPhenomPv2"
REF_GPS = 1126259462.0           # O1 era, matches the existing bank's reference_time_gps
D_LO, D_HI = 100.0, 5000.0
QMAX = 6.0
COLS = ["source_id", "projection_mode", "approximant", "mass1", "mass2", "mass_ratio",
        "total_mass", "distance_mpc", "inclination_rad", "coa_phase_rad", "ra_rad",
        "dec_rad", "polarization_rad", "reference_time_gps", "H1_f_plus", "H1_f_cross",
        "H1_peak_index", "L1_f_plus", "L1_f_cross", "L1_peak_index",
        # extra columns, absent from the existing banks and additive-only:
        # the injected SNR label must be the REAL (full-signal) network SNR, the same
        # quantity the catalog quotes -- not the SNR of the 2 s crop. Whatever the 1 s
        # analysis tile then fails to recover is a pipeline limitation and must show up
        # as reduced recovery, not be absorbed into the axis.
        "optimal_snr_full_ref", "optimal_snr_crop_ref", "snr_full_over_crop"]
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

# reference PSD for those columns: the SAME leakage-free PSD inject.py normalises
# against under SM_INJ_NORM_FIXED=1 / SM_ASD_TAG=fixed2, so the ratio is consistent
# with how the amplitude scale is applied.
REF_PSD = {d: f"{MADGRAV_ROOT}/data/o4a_search_prep/reference_psd_fixed2_{d}.npz"
           for d in ("H1", "L1")}


def load_ref_psd():
    """{det: (freq, psd)} interpolated onto whatever grid optimal_snr() needs."""
    out = {}
    for d, path in REF_PSD.items():
        z = np.load(path)
        f, S = z["freq"], z["psd"]
        ok = np.isfinite(S) & (S > 0) & (f > 0)
        out[d] = (f[ok], S[ok])
    return out


def optimal_snr(h, psd, f_lower=F_LOWER):
    """sqrt(4 Int |h~(f)|^2 / S_n(f) df) for a real time series at FS."""
    n = len(h)
    hf = np.fft.rfft(np.asarray(h, np.float64)) / FS          # one-sided, s units
    f = np.fft.rfftfreq(n, 1.0 / FS)
    fg, Sg = psd
    S = np.interp(f, fg, Sg, left=np.inf, right=np.inf)
    band = (f >= f_lower) & np.isfinite(S) & (S > 0)
    if not band.any():
        return 0.0
    df = f[1] - f[0]
    return float(np.sqrt(4.0 * np.sum(np.abs(hf[band]) ** 2 / S[band]) * df))


def draw(rng, a):
    """One source, using the EXISTING bank's mass prior shape with lowered floors.

    Reverse-engineered from the existing bank (2026-09-04): mean m1 = 65.2 = midpoint of
    [10,120] and mean m2 = 39.3 = midpoint of [10, 65], i.e. m1 ~ U(m1_lo, m1_hi),
    m2 ~ U(m2_min, m1), reject q > QMAX. Drawing Mtot uniform instead made the pairs too
    asymmetric: median chirp mass 19.20 vs 20.21 over Mtot 40-60, which alone put the
    new bank 15-22% quiet at fixed distance. Amplitude and projection were already
    correct (chirp-mass-normalised SNR*d agreed to 1.7%).
    """
    while True:
        m1 = rng.uniform(a.m1_lo, a.m1_hi)
        m2 = rng.uniform(a.m2_min, m1)
        mtot = m1 + m2
        if m2 >= a.m2_min and 1.0 <= m1 / m2 <= QMAX and a.mtot_lo <= mtot <= a.mtot_hi:
            break
    return dict(
        mass1=m1, mass2=m2, mass_ratio=m1 / m2, total_mass=mtot,
        distance_mpc=rng.uniform(D_LO, D_HI),
        inclination_rad=float(np.arccos(rng.uniform(-1.0, 1.0))),
        coa_phase_rad=rng.uniform(0.0, 2 * np.pi),
        ra_rad=rng.uniform(0.0, 2 * np.pi),
        dec_rad=float(np.arcsin(rng.uniform(-1.0, 1.0))),
        polarization_rad=rng.uniform(0.0, 2 * np.pi),
    )


def project_and_crop(p, dets, psds):
    """(rows, peaks) and records full vs crop optimal SNR on p."""
    from pycbc.waveform import get_td_waveform            # noqa: PLC0415 - needs pycbc env

    hp, hc = get_td_waveform(approximant=APX, mass1=p["mass1"], mass2=p["mass2"],
                             spin1z=0.0, spin2z=0.0, distance=p["distance_mpc"],
                             inclination=p["inclination_rad"], coa_phase=p["coa_phase_rad"],
                             delta_t=1.0 / FS, f_lower=F_LOWER)
    rows, peaks = {}, {}
    s2_full = s2_crop = 0.0
    for name, det in dets.items():
        fp, fc = det.antenna_pattern(p["ra_rad"], p["dec_rad"], p["polarization_rad"], REF_GPS)
        h = fp * np.asarray(hp) + fc * np.asarray(hc)
        pk = int(np.abs(h).argmax())
        lo, hi = pk - NCROP // 2, pk + NCROP // 2
        seg = np.zeros(NCROP, np.float32)                  # zero-pad if the peak sits near an end
        a, b = max(lo, 0), min(hi, len(h))
        seg[a - lo:b - lo] = h[a:b]
        rows[name] = seg
        peaks[name] = pk
        p[f"{name}_f_plus"], p[f"{name}_f_cross"] = float(fp), float(fc)
        s2_full += optimal_snr(h, psds[name]) ** 2          # untruncated: the REAL SNR
        s2_crop += optimal_snr(seg, psds[name]) ** 2        # what the 2 s crop retains
    full, crop = np.sqrt(s2_full), np.sqrt(s2_crop)
    p["optimal_snr_full_ref"] = float(full)
    p["optimal_snr_crop_ref"] = float(crop)
    p["snr_full_over_crop"] = float(full / crop) if crop > 0 else float("nan")
    return rows, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10240)
    ap.add_argument("--shard", type=int, default=1024)
    ap.add_argument("--mtot-lo", type=float, default=10.0)
    ap.add_argument("--mtot-hi", type=float, default=22.0)
    ap.add_argument("--m1-lo", type=float, default=5.0)
    ap.add_argument("--m1-hi", type=float, default=19.0)
    ap.add_argument("--m2-min", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--shard-index", type=int, default=None,
                    help="write signals_<index>.{npz,csv} and offset source_id by "
                         "index*n, so parallel tasks do not collide")
    a = ap.parse_args()

    from pycbc.detector import Detector                    # noqa: PLC0415
    dets = {"H1": Detector("H1"), "L1": Detector("L1")}
    psds = load_ref_psd()
    rng = np.random.default_rng(a.seed if a.shard_index is None
                                else a.seed + 1000 * a.shard_index)
    os.makedirs(a.out, exist_ok=True)

    sid = 0 if a.shard_index is None else a.shard_index * a.n
    for shard0 in range(0, a.n, a.shard):
        nsh = min(a.shard, a.n - shard0)
        H, L, meta = [], [], []
        for _ in range(nsh):
            p = draw(rng, a)
            rows, peaks = project_and_crop(p, dets, psds)
            H.append(rows["H1"]); L.append(rows["L1"])
            p.update(source_id=sid, projection_mode="projected", approximant=APX,
                     reference_time_gps=REF_GPS,
                     H1_peak_index=peaks["H1"], L1_peak_index=peaks["L1"])
            meta.append(p)
            sid += 1
        tag = (f"signals_{shard0 // a.shard:04d}" if a.shard_index is None
               else f"signals_{a.shard_index:04d}")
        np.savez(os.path.join(a.out, tag + ".npz"),
                 H1=np.stack(H).astype(np.float32), L1=np.stack(L).astype(np.float32))
        with open(os.path.join(a.out, tag + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for m in meta:
                w.writerow({c: m[c] for c in COLS})
        _r = np.array([m["snr_full_over_crop"] for m in meta])
        print(f"[bank] {tag}: {nsh} sources, Mtot "
              f"{min(m['total_mass'] for m in meta):.1f}-{max(m['total_mass'] for m in meta):.1f}"
              f"  full/crop SNR ratio med {np.median(_r):.3f} max {_r.max():.3f}", flush=True)
    print(f"[bank] done: {sid} sources -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
