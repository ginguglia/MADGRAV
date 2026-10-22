"""MADGRAV2 unified injection bank v2 (2026-09-23: "regenerate the bank fully, a single bank from mtot=10 to 800";
PREREG_SPECIALISTS_20260923 §2 amendment A1). Replaces sig / heavy / lowmass for the specialist retrain.

Population (one bank, one recipe):
  Mtot DETECTOR-FRAME log-flat on [10, 800] Msun; q uniform [1, 6]; IMRPhenomXPHM; precessing spins, |chi| uniform [0, 0.99],
  isotropic tilts; isotropic sky, cos(inc) uniform, psi and coalescence phase uniform; distance volumetric on [100, 8000] Mpc
  (irrelevant downstream: the injector rescales to a target SNR); GPS uniform over O3a-O4b.
Physical projection: pycbc Detector.project_wave(hp, hc, ra, dec, psi) -- antenna patterns AND the geocentre->detector
  time delay (sub-sample), so the H1-L1 delay (|dt| <= 10 ms) is IN the waveforms (the v1 banks logged it but did not apply it).
Window: 4 s @ 4096 Hz (16384 samples) on a common geocentric grid, geocentric merger at sample 3*4096 (3 s pre, 1 s post);
  both detectors share the grid, so each leg's peak sits at 12288 + dt_det*4096 (exact sub-sample FFT shift, no rounding).
  0.5 s Tukey onset taper at the window start.
Start frequency: f_lower = max(10 Hz, f at which the Newtonian (2,2) time to merger is 10 s) -- only content more than 10 s
  before merger is dropped, and the stored window keeps the last 3 s, so the kept signal is unaffected; makes Mtot~10 fast.
Output per chunk: signals_XXXX.npz (H1, L1 float32 (n,16384)) + signals_XXXX.csv (parameters, f_lower, dt_H1, dt_L1, dt_L1_minus_H1).
Usage (venv): python gen_bank_v2.py --out DIR --chunk K [--n 500] [--seed 20260923]"""
import argparse, os, csv, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=20260923)
ap.add_argument("--mtot-lo", type=float, default=10.); ap.add_argument("--mtot-hi", type=float, default=800.); ap.add_argument("--qmax", type=float, default=6.)
ap.add_argument("--dlo", type=float, default=100.); ap.add_argument("--dhi", type=float, default=8000.); ap.add_argument("--apx", default="IMRPhenomXPHM")
ap.add_argument("--chi-max", type=float, default=0.99)   # 2026-09-25: restricted-population banks (bank_r_*); default = bank v2 behaviour
a = ap.parse_args()
FS = 4096; NS = 4 * FS; PRE = 3.0; TAPER = 0.5; TAU = 10.0; FMIN_FLOOR = 10.0
GPS_LO, GPS_HI = 1238166018, 1421000000          # O3a start .. end of O4b (antenna patterns only)
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from scipy.signal.windows import tukey
dets = {d: Detector(d) for d in ("H1", "L1")}; rng = np.random.default_rng([a.seed, a.chunk])
ntap = int(TAPER * FS); onset = tukey(2 * ntap, 1.0)[:ntap]
def f_start(m1, m2):                              # Newtonian (2,2): tau = 5/256 (G Mc/c^3)^(-5/3) (pi f)^(-8/3)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2 * 4.925491e-6
    return max(FMIN_FLOOR, (TAU * 256 / 5 * mc ** (5 / 3)) ** (-3 / 8) / np.pi)
rows = []; H = {"H1": [], "L1": []}; i = 0; tries = 0
while i < a.n and tries < 20 * a.n:
    tries += 1
    mt = float(np.exp(rng.uniform(np.log(a.mtot_lo), np.log(a.mtot_hi)))); q = float(rng.uniform(1., a.qmax)); m1 = mt * q / (1 + q); m2 = mt / (1 + q)
    c1, c2 = rng.uniform(0, a.chi_max, 2); t1, t2 = np.arccos(rng.uniform(-1, 1, 2)); p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    s1 = (c1 * np.sin(t1) * np.cos(p1), c1 * np.sin(t1) * np.sin(p1), c1 * np.cos(t1)); s2 = (c2 * np.sin(t2) * np.cos(p2), c2 * np.sin(t2) * np.sin(p2), c2 * np.cos(t2))
    d = float((rng.uniform(a.dlo ** 3, a.dhi ** 3)) ** (1 / 3)); inc = float(np.arccos(rng.uniform(-1, 1))); phi = float(rng.uniform(0, 2 * np.pi))
    ra = float(rng.uniform(0, 2 * np.pi)); dec = float(np.arcsin(rng.uniform(-1, 1))); psi = float(rng.uniform(0, np.pi)); gps = float(rng.uniform(GPS_LO, GPS_HI))
    fl = f_start(m1, m2)
    try:
        hp, hc = get_td_waveform(approximant=a.apx, mass1=m1, mass2=m2, spin1x=s1[0], spin1y=s1[1], spin1z=s1[2], spin2x=s2[0], spin2y=s2[1], spin2z=s2[2],
                                 distance=d, inclination=inc, coa_phase=phi, delta_t=1. / FS, f_lower=fl)
    except Exception:
        continue
    hp.start_time += gps; hc.start_time += gps; t0 = gps - PRE; out = {}; ok = True
    for dn, det in dets.items():
        h = det.project_wave(hp, hc, ra, dec, psi)          # antenna pattern + geocentre->detector delay (sub-sample)
        u = (t0 - float(h.start_time)) * FS; k0 = int(np.floor(u)); frac = u - k0   # common grid: sample k <-> t0 + k/FS
        x = h.numpy(); PAD = 256
        if k0 - PAD < 0: ok = False; break                      # waveform must cover the whole pre-merger window (+pad)
        seg = np.zeros(NS + 2 * PAD); src = x[k0 - PAD:k0 + NS + PAD]; seg[:len(src)] = src
        F = np.fft.rfft(seg); F *= np.exp(2j * np.pi * np.fft.rfftfreq(len(seg)) * frac)   # exact sub-sample advance by frac
        w = np.fft.irfft(F, len(seg))[PAD:PAD + NS]; w[:ntap] *= onset
        fp, fc = det.antenna_pattern(ra, dec, psi, gps)
        out[dn] = (w.astype(np.float32), fp, fc, int(np.argmax(np.abs(w))), float(det.time_delay_from_earth_center(ra, dec, gps)))
    if not ok or not all(np.all(np.isfinite(out[k][0])) and np.any(out[k][0]) for k in out): continue
    rows.append(dict(source_id=a.chunk * a.n + i, approximant=a.apx, mass1=m1, mass2=m2, mass_ratio=q, total_mass=mt, chi1=c1, chi2=c2, tilt1=t1, tilt2=t2,
        spin1x=s1[0], spin1y=s1[1], spin1z=s1[2], spin2x=s2[0], spin2y=s2[1], spin2z=s2[2], distance_mpc=d, inclination_rad=inc, coa_phase_rad=phi,
        ra_rad=ra, dec_rad=dec, polarization_rad=psi, reference_time_gps=gps, f_lower=fl,
        H1_f_plus=out["H1"][1], H1_f_cross=out["H1"][2], H1_peak_index=out["H1"][3], L1_f_plus=out["L1"][1], L1_f_cross=out["L1"][2], L1_peak_index=out["L1"][3],
        H1_delay_s=out["H1"][4], L1_delay_s=out["L1"][4], dt_L1_minus_H1_s=out["L1"][4] - out["H1"][4], merger_index=int(PRE * FS), mtot_det_frame=1))
    for dn in H: H[dn].append(out[dn][0])
    i += 1
os.makedirs(a.out, exist_ok=True); base = os.path.join(a.out, f"signals_{a.chunk:04d}")
np.savez(base + ".npz", H1=np.stack(H["H1"]), L1=np.stack(H["L1"]))
with open(base + ".csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print(f"[bank-v2] chunk {a.chunk}: {i} waveforms ({tries} tries) -> {base}.npz  Mtot {min(r['total_mass'] for r in rows):.1f}-{max(r['total_mass'] for r in rows):.1f}", flush=True)
