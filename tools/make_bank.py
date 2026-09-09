#!/usr/bin/env python
"""Generate a projected signal bank in the repository's established convention.

Covers the full injection mass range with one tool. Presets encode each existing bank's
documented convention; --preset lowmass fills the 10-22 Msun gap that the stellar bank
leaves open (both its component masses are floored at 10, so it cannot reach below 20).

  preset        approximant      spins        Mtot            f_min   q
  stellar       IMRPhenomPv2     none         20-238          10 Hz   <=6
  ultramassive  IMRPhenomXPHM    precessing   150-400         20 Hz   <=4
  heavy         IMRPhenomXPHM    precessing   150-800         10 Hz   <=4
  lowmass       IMRPhenomPv2     none         10-22           10 Hz   <=6

CONVENTIONS REPRODUCED (verified against the existing banks, 2026-09-04)
  * projection      h = F_plus h_plus + F_cross h_cross per detector at reference_time_gps;
                    antenna factors reproduce the recorded H1_f_plus/H1_f_cross exactly
                    (max |diff| 0.0000, correlation 1.0000)
  * crop            8192 samples (2 s) centred on EACH DETECTOR'S OWN |h| peak, so the
                    inter-detector arrival delay is REMOVED from the stored series while
                    being recorded in H1_delay_s / L1_delay_s. Both stored channels peak
                    at sample 4096, as in every existing bank. The coherence statistic
                    searches +/-45 samples, so this is consistent.
  * mass prior      m1 ~ U(m1_lo, m1_hi), m2 ~ U(m2_min, m1), reject q > qmax. Recovered
                    from the stellar bank: mean m1 65.2 = midpoint of [10,120], mean m2
                    39.3 = midpoint of [10,65]. Drawing Mtot uniform instead makes the
                    pairs too asymmetric (median chirp mass 19.2 vs 20.2 over Mtot 40-60)
                    and leaves the bank 15-22% quiet at fixed distance.
  * distance        uniform in DISTANCE over [d_lo, d_hi] (measured: counts per
                    equal-width distance bin are flat). It is a reference distance, not a
                    population prior -- the population weight is the rho^-4 dRho grid
                    applied downstream, and inject.py renormalises every injection to a
                    target SNR regardless.
  * orientation     cos(inclination) ~ U(-1,1); coa_phase, polarization ~ U(0,2pi)
  * sky             ra ~ U(0,2pi); sin(dec) ~ U(-1,1)
  * layout          signals_NNNN.npz {H1, L1} float32 (n, 8192) + signals_NNNN.csv

NOT REPRODUCED
  Per-source absolute amplitude does not match the existing banks: regenerating an
  existing entry from its own CSV row gives waveform overlap 0.81-0.996 (same source,
  same morphology, correct alignment) but an amplitude ratio scattered over 0.72-2.62.
  This is immaterial for injection recovery, because inject.py rescales every injection
  to a target SNR and only the morphology and the H1/L1 amplitude ratio survive. It does
  matter for anything built from the bank's amplitude-versus-distance relation -- notably
  the VT horizons in vt_relabel_comoving.py -- so a bank from this tool must not be mixed
  with the existing ones for VT until the discrepancy is understood.

SCHEMA
  Emits the base 20 columns, the 9 spin columns for precessing presets, and the
  self-describing columns introduced by heavy_bank_v1: f_min, H1_delay_s, L1_delay_s and
  mtot_det_frame (a FLAG, always 1: total_mass is detector-frame). Also records
  optimal_snr_full_ref / optimal_snr_crop_ref / snr_full_over_crop against the
  leakage-free reference PSD, so a campaign can normalise to the FULL-signal network SNR
  rather than to the SNR retained by the 2 s crop. Those differ by ~10% at Mtot 10-22 and
  by 0.3% above Mtot 20.

Run with the pycbc environment:
  wfgen-venv/bin/python tools/make_bank.py --preset lowmass --out DIR [--n 10240]
      [--shard 1024] [--shard-index K] [--seed S]
"""
import argparse
import csv
import os

import numpy as np

FS = 4096
NCROP = 8192
REF_GPS = 1126259462.0
NCROP_HALF = NCROP // 2

PRESETS = {
    #             apx               f_low chi_max m1_lo  m1_hi  m2_min qmax  d_lo   d_hi
    "stellar":      ("IMRPhenomPv2", 10.0, 0.0,   10.0,  120.0, 10.0,  6.0,  100.0, 5000.0),
    "ultramassive": ("IMRPhenomXPHM", 20.0, 0.7,  79.0,  320.0, 31.0,  4.0,  100.0, 5000.0),
    "heavy":        ("IMRPhenomXPHM", 10.0, 0.99, 80.0,  640.0, 31.0,  4.0,  700.0, 8000.0),
    "lowmass":      ("IMRPhenomPv2", 10.0, 0.0,    5.0,   19.0,  3.0,  6.0,  100.0, 5000.0),
}
MTOT_RANGE = {"stellar": (20.0, 240.0), "ultramassive": (150.0, 400.0),
              "heavy": (150.0, 800.0), "lowmass": (10.0, 22.0)}

BASE = ["source_id", "projection_mode", "approximant", "mass1", "mass2", "mass_ratio",
        "total_mass"]
SPIN = ["chi1", "chi2", "tilt1", "tilt2", "spin1x", "spin1y", "spin1z",
        "spin2x", "spin2y", "spin2z"]
TAIL = ["distance_mpc", "inclination_rad", "coa_phase_rad", "ra_rad", "dec_rad",
        "polarization_rad", "reference_time_gps", "H1_f_plus", "H1_f_cross",
        "H1_peak_index", "L1_f_plus", "L1_f_cross", "L1_peak_index",
        "f_min", "H1_delay_s", "L1_delay_s", "mtot_det_frame",
        "optimal_snr_full_ref", "optimal_snr_crop_ref", "snr_full_over_crop"]

REF_PSD = {d: os.path.join(os.environ.get("MADGRAV_ROOT", "."), "data", "o4a_search_prep",
                           f"reference_psd_fixed2_{d}.npz") for d in ("H1", "L1")}


def load_ref_psd():
    out = {}
    for d, path in REF_PSD.items():
        z = np.load(path)
        f, S = z["freq"], z["psd"]
        ok = np.isfinite(S) & (S > 0) & (f > 0)
        out[d] = (f[ok], S[ok])
    return out


def optimal_snr(h, psd, f_lower):
    n = len(h)
    hf = np.fft.rfft(np.asarray(h, np.float64)) / FS
    f = np.fft.rfftfreq(n, 1.0 / FS)
    fg, Sg = psd
    S = np.interp(f, fg, Sg, left=np.inf, right=np.inf)
    b = (f >= f_lower) & np.isfinite(S) & (S > 0)
    if not b.any():
        return 0.0
    return float(np.sqrt(4.0 * np.sum(np.abs(hf[b]) ** 2 / S[b]) * (f[1] - f[0])))


def draw(rng, cfg, mtot_lo, mtot_hi):
    apx, f_low, chi_max, m1_lo, m1_hi, m2_min, qmax, d_lo, d_hi = cfg
    while True:
        m1 = rng.uniform(m1_lo, m1_hi)
        m2 = rng.uniform(m2_min, m1)
        mt = m1 + m2
        if m2 >= m2_min and 1.0 <= m1 / m2 <= qmax and mtot_lo <= mt <= mtot_hi:
            break
    p = dict(mass1=m1, mass2=m2, mass_ratio=m1 / m2, total_mass=mt,
             distance_mpc=rng.uniform(d_lo, d_hi),
             inclination_rad=float(np.arccos(rng.uniform(-1.0, 1.0))),
             coa_phase_rad=rng.uniform(0.0, 2 * np.pi),
             ra_rad=rng.uniform(0.0, 2 * np.pi),
             dec_rad=float(np.arcsin(rng.uniform(-1.0, 1.0))),
             polarization_rad=rng.uniform(0.0, 2 * np.pi),
             approximant=apx, f_min=f_low, mtot_det_frame=1,
             projection_mode="projected", reference_time_gps=REF_GPS)
    if chi_max > 0.0:                                   # precessing: isotropic tilts
        for i in (1, 2):
            chi = rng.uniform(0.0, chi_max)
            tilt = float(np.arccos(rng.uniform(-1.0, 1.0)))
            phi = rng.uniform(0.0, 2 * np.pi)
            p[f"chi{i}"] = chi
            p[f"tilt{i}"] = tilt
            p[f"spin{i}x"] = chi * np.sin(tilt) * np.cos(phi)
            p[f"spin{i}y"] = chi * np.sin(tilt) * np.sin(phi)
            p[f"spin{i}z"] = chi * np.cos(tilt)
    else:
        for i in (1, 2):
            p[f"chi{i}"] = p[f"tilt{i}"] = 0.0
            for c in "xyz":
                p[f"spin{i}{c}"] = 0.0
    return p


def build(p, dets, psds, f_low):
    from pycbc.waveform import get_td_waveform                       # noqa: PLC0415

    hp, hc = get_td_waveform(approximant=p["approximant"],
                             mass1=p["mass1"], mass2=p["mass2"],
                             spin1x=p["spin1x"], spin1y=p["spin1y"], spin1z=p["spin1z"],
                             spin2x=p["spin2x"], spin2y=p["spin2y"], spin2z=p["spin2z"],
                             distance=p["distance_mpc"],
                             inclination=p["inclination_rad"],
                             coa_phase=p["coa_phase_rad"],
                             delta_t=1.0 / FS, f_lower=f_low)
    rows = {}
    s2f = s2c = 0.0
    for name, det in dets.items():
        fp, fc = det.antenna_pattern(p["ra_rad"], p["dec_rad"],
                                     p["polarization_rad"], REF_GPS)
        h = fp * np.asarray(hp) + fc * np.asarray(hc)
        pk = int(np.abs(h).argmax())
        seg = np.zeros(NCROP, np.float32)
        lo, hi = pk - NCROP_HALF, pk + NCROP_HALF
        a, b = max(lo, 0), min(hi, len(h))
        seg[a - lo:b - lo] = h[a:b]
        rows[name] = seg
        p[f"{name}_f_plus"], p[f"{name}_f_cross"] = float(fp), float(fc)
        p[f"{name}_peak_index"] = pk
        # recorded, not applied: the stored series is peak-centred per detector
        p[f"{name}_delay_s"] = float(
            det.time_delay_from_earth_center(p["ra_rad"], p["dec_rad"], REF_GPS))
        s2f += optimal_snr(h, psds[name], f_low) ** 2
        s2c += optimal_snr(seg, psds[name], f_low) ** 2
    full, crop = np.sqrt(s2f), np.sqrt(s2c)
    p["optimal_snr_full_ref"] = float(full)
    p["optimal_snr_crop_ref"] = float(crop)
    p["snr_full_over_crop"] = float(full / crop) if crop > 0 else float("nan")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10240)
    ap.add_argument("--shard", type=int, default=1024)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--mtot-lo", type=float, default=None)
    ap.add_argument("--mtot-hi", type=float, default=None)
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    f_low = cfg[1]
    mlo, mhi = MTOT_RANGE[a.preset]
    mlo = a.mtot_lo if a.mtot_lo is not None else mlo
    mhi = a.mtot_hi if a.mtot_hi is not None else mhi
    cols = BASE + SPIN + TAIL

    from pycbc.detector import Detector                              # noqa: PLC0415
    dets = {"H1": Detector("H1"), "L1": Detector("L1")}
    psds = load_ref_psd()
    rng = np.random.default_rng(a.seed if a.shard_index is None
                                else a.seed + 1000 * a.shard_index)
    os.makedirs(a.out, exist_ok=True)
    sid = 0 if a.shard_index is None else a.shard_index * a.n

    for s0 in range(0, a.n, a.shard):
        nsh = min(a.shard, a.n - s0)
        H, L, meta = [], [], []
        for _ in range(nsh):
            p = draw(rng, cfg, mlo, mhi)
            rows = build(p, dets, psds, f_low)
            H.append(rows["H1"]); L.append(rows["L1"])
            p["source_id"] = sid
            meta.append(p)
            sid += 1
        tag = (f"signals_{s0 // a.shard:04d}" if a.shard_index is None
               else f"signals_{a.shard_index:04d}")
        np.savez(os.path.join(a.out, tag + ".npz"),
                 H1=np.stack(H).astype(np.float32), L1=np.stack(L).astype(np.float32))
        with open(os.path.join(a.out, tag + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for m in meta:
                w.writerow({c: m[c] for c in cols})
        r = np.array([m["snr_full_over_crop"] for m in meta])
        mt = np.array([m["total_mass"] for m in meta])
        print(f"[bank] {tag}: {nsh} sources  Mtot {mt.min():.1f}-{mt.max():.1f}"
              f"  full/crop SNR med {np.median(r):.3f}", flush=True)
    print(f"[bank] done: {sid} sources -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
