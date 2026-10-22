"""Physical H1->L1 time delay for every row of the three injection banks (PREREG_SPECIALISTS_20260923 §2 delay rule).
dt = t_L1 - t_H1 from the row's own (ra, dec, reference_time_gps), pycbc Detector.time_delay_from_detector.
Heavy bank rows are cross-checked against L1_delay_s - H1_delay_s. Also records the offset the bank itself carries
(L1_peak_index - H1_peak_index) so the renderer can remove it before applying dt.
Run with venv (pycbc); output /scratch/.../madgrav2/spec/bank_delays_<bank>.npz keyed by (src, row)."""
import glob, os, numpy as np, pandas as pd
from pycbc.detector import Detector
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/MADGRAV1/data"; OUT = MADGRAV_SCRATCH + "/madgrav2/spec"; FS = 4096
BANKS = {"sig": "o1_o3_signal_bank_projected_2s_x10", "heavy": "heavy_bank_v1", "lowmass": "lowmass_bank_v1"}
H, L = Detector("H1"), Detector("L1")
for b, sub in BANKS.items():
    fs = sorted(glob.glob(f"{D}/{sub}/projectedbank*/signals_*.csv")); F, R, DT, OFF = [], [], [], []
    for f in fs:
        d = pd.read_csv(f)
        dt = np.array([L.time_delay_from_detector(H, r.ra_rad, r.dec_rad, r.reference_time_gps) for r in d.itertuples()])
        if "H1_delay_s" in d:
            res = np.abs(dt - (d.L1_delay_s - d.H1_delay_s).values).max()
            assert res < 1e-9, f"{f}: pycbc vs bank delay columns differ by {res}"
        F += [os.path.basename(f).replace(".csv", "")] * len(d); R += list(range(len(d))); DT += list(dt)
        OFF += list((d.L1_peak_index - d.H1_peak_index).values / FS)
    DT, OFF = np.array(DT), np.array(OFF)
    np.savez(f"{OUT}/bank_delays_{b}.npz", src=np.array(F), row=np.array(R), dt=DT, bank_offset=OFF)
    print(f"{b}: {len(DT)} rows in {len(fs)} files | physical dt ms p5/p50/p95 {np.percentile(DT*1e3,5):.1f}/{np.median(DT*1e3):.1f}/"
          f"{np.percentile(DT*1e3,95):.1f} max|dt| {np.abs(DT).max()*1e3:.1f} | bank-carried offset p95 |.| {np.percentile(np.abs(OFF)*1e3,95):.1f} ms", flush=True)
