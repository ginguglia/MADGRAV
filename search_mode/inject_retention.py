"""Signal-retention test for the local-ASD consistency veto (veto #1).

Inject bank signals across a net-SNR grid into real O3b noise, score the CNN specialists under
(a) the reference ASD and (b) a local +/-64s median-Welch ASD, and measure how many recovered
signals survive the local re-whitening. If retention ~= 100% (esp. at low SNR) the veto is safe to
adopt as a general post-detection veto. Uses the pipeline's EXACT cnn path via the strain cache.

Run from MADGRAV_ROOT: SM_STRAIN=... SM_ALLOW_CPU=1 BLIND_DEV=cpu python inject_retention.py
"""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
ROOT = os.environ["MADGRAV_ROOT"]
for p in ("search_mode", "improved", "spectrogram_cascade"):
    sys.path.insert(0, os.path.join(ROOT, p))
import driver_blindscan as B
import improved_pipeline as ip
from gwpy.timeseries import TimeSeries
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


FS = 4096; WN = 4 * FS
GRID = [6., 7., 8., 10., 12., 15., 20.]
N_PER = int(os.environ.get("INJ_N", "40"))
UM_FRAC = 0.5
NOISE_SEG = os.environ.get("INJ_SEG", "o3b_1264297924")   # inject into this O3b segment's noise
STRAIN = os.environ.get("SM_STRAIN", MADGRAV_SCRATCH + "/strain_o3b_full")
RNG = np.random.default_rng(20260706)

# reference ASD to adopt = the O3b run-average
O3B = {d: ip.load_detector_asd_o1(f"{ROOT}/data/o3b_search_prep", d) for d in ("H1", "L1")}

_INJ = {}
_orig = B._strain
def _fake(n, d):
    return _INJ[(n, d)] if str(n).startswith("inj") else _orig(n, d)
B._strain = _fake


def cnn(pipe, hH, hL):
    _INJ[("inj", "H1")] = hH.astype(np.float32); _INJ[("inj", "L1")] = hL.astype(np.float32)
    return B.cnn_hm_lm("inj", 0, "inj", 0)   # _win("inj",d,0) = hX[0:WN]


def run():
    pipe = B.cpipe()
    pb = ip.load_o1_signal_bank(os.path.join(ROOT, "data", "o1_o3_signal_bank_projected_2s_x10"))
    ub = ip.load_o1_signal_bank(os.path.join(ROOT, "data", "ultramassive_bank"))
    banks = {"sig": (pb["H1"], pb["L1"]), "um": (ub["H1"], ub["L1"])}
    rawH = np.load(f"{STRAIN}/{NOISE_SEG}_H1.npz")["strain"]
    rawL = np.load(f"{STRAIN}/{NOISE_SEG}_L1.npz")["strain"]
    print(f"[inj-ret] noise={NOISE_SEG} sig {len(pb['H1'])}/UM {len(ub['H1'])}; ref=o3b_search_prep; N={N_PER}/snr\n", flush=True)
    print(f"{'snr':>4} {'n':>4} {'reco(ref>0.5)':>13} {'retain(loc>0.5)':>16} {'medΔcnn':>8} {'min_cnn_loc':>11}", flush=True)
    print("-" * 62, flush=True)
    rows = []
    for snr in GRID:
        recs, rets, dcs, mins = 0, 0, [], []
        for _ in range(N_PER):
            um = RNG.random() < UM_FRAC; WH, WL = banks["um" if um else "sig"]
            k = int(RNG.integers(0, len(WH))); wH = np.asarray(WH[k], np.float32); wL = np.asarray(WL[k], np.float32); L = len(wH)
            s0 = np.sqrt(ip.compute_optimal_snr(wH, O3B["H1"]) ** 2 + ip.compute_optimal_snr(wL, O3B["L1"]) ** 2)
            if s0 <= 0:
                continue
            sc = np.float32(snr / s0)
            base = int(RNG.integers(64 * FS, len(rawH) - 64 * FS - WN))     # leave +/-64s for local ASD
            hH = rawH[base:base + WN].copy(); hL = rawL[base:base + WN].copy()
            c = WN // 2
            hH[c - L // 2:c - L // 2 + L] += wH * sc; hL[c - L // 2:c - L // 2 + L] += wL * sc
            # reference-ASD CNN
            sav = {d: pipe.asd[d] for d in ("H1", "L1")}
            pipe.asd["H1"], pipe.asd["L1"] = O3B["H1"], O3B["L1"]
            hm_r, lm_r = cnn(pipe, hH, hL)
            # local-ASD CNN (ASD from CLEAN surrounding strain, no injection)
            laH = TimeSeries(rawH[base - 64 * FS:base + 64 * FS].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")
            laL = TimeSeries(rawL[base - 64 * FS:base + 64 * FS].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")
            pipe.asd["H1"], pipe.asd["L1"] = laH, laL
            hm_l, lm_l = cnn(pipe, hH, hL)
            pipe.asd["H1"], pipe.asd["L1"] = sav["H1"], sav["L1"]
            cref, cloc = max(hm_r, lm_r), max(hm_l, lm_l)
            if cref > 0.5:
                recs += 1
                mins.append(cloc); dcs.append(cloc - cref)
                if cloc > 0.5:
                    rets += 1
        ret_frac = (rets / recs) if recs else float("nan")
        print(f"{snr:4.0f} {N_PER:4d} {recs:13d} {rets:>10d} ({ret_frac:4.0%}) {np.median(dcs) if dcs else 0:8.3f} {min(mins) if mins else 0:11.3f}", flush=True)
        rows.append(dict(snr=snr, n=N_PER, recovered=recs, retained=rets, retention=ret_frac,
                         med_dcnn=float(np.median(dcs)) if dcs else None, min_cnn_loc=float(min(mins)) if mins else None))
    json.dump(rows, open(os.path.join(ROOT, "search_mode", "inj_retention_o3b.json"), "w"), indent=2)
    print("\nDONE -> search_mode/inj_retention_o3b.json", flush=True)


if __name__ == "__main__":
    run()
