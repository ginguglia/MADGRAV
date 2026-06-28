"""p_astro campaign — STAGE 1: inject the XPHM banks into same-run noise and score
through the FROZEN pipeline. Importance-sampling design: store per-injection
(masses, drawn network SNR, recovered net sigma, LR score, gated flag, noise half);
ALL population weights + p_s are computed in stage 2 (pastro_fgmc.py), so the f_HM
grid and population never require re-scoring (PASTRO_SPEC §5/§6, review BINDING #6).

Proposal injection set = p1 bank (50k XPHM, Mtot 10-236) + ultra-massive bank
(2k, Mtot 150-400), UM oversampled x4 for >=2000 recovered in the >200 bin. Network
SNR drawn ~ rho^-4 (uniform-in-comoving-volume leading behaviour, PASTRO_SPEC §4),
rho in [4.5, 80]. Noise = the run's prepared dataset, split into two disjoint halves
(seg parity) for the BINDING #5 disjoint-noise control. Scoring is byte-identical to
p7b_o4a.py / p7c_o4b.py: run-averaged ASD whiten -> QT -> CAE sigma (run sigma-norm)
-> BA coherence -> centroids -> 5-seed ensemble g -> quantile-align (p7g map) ->
lr_model_v5; net sigma>=4 gate.

Run per run:  P7_RUN=O4a P7_DEVICE=cuda:0 python pastro_campaign.py
Output: p4/pastro_inj_<run>.npz, PASTRO_INJ_<run>_DONE.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool
from scipy.ndimage import zoom

import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR = f"{ROOT}/lr_cascade"
sys.path.insert(0, LR); import p7_lib as L
sys.path.insert(0, f"{ROOT}/improved")
import improved_pipeline as ip

RUN = os.environ.get("P7_RUN", "O4a")
DEVICE = torch.device(os.environ.get("P7_DEVICE", "cuda:0"))
OUT, P0, P1V4E = f"{LR}/p4", f"{LR}/p0", f"{LR}/p1v42_ens"
SC = f"{ROOT}/spectrogram_cascade"
BANK = f"{ROOT}/bank"
POOL_P1 = f"{BANK}/p1_signal_bank"
UM = f"{BANK}/ultramassive_bank"
PREP = {"O3a": os.environ.get("MADGRAV_O3A_PREP", "/scratch/ginguglia/GW/codex-agent/data/o3a_prepared_4s_crop2s"),
        "O3b": f"{ROOT}/data/o3b_prepared_4s_crop2s",
        "O4a": f"{ROOT}/data/o4a_prepared_4s_crop2s",
        "O4b": f"{ROOT}/data/o4b_prepared_4s_crop2s"}[RUN]
FS, sqrt2, GCLIP, QG = ip.FS, np.sqrt(2.0), 6.0, np.linspace(0, 1, 513)
LAG = int(0.011 * FS)
UM_OVERSAMPLE = 4
SNR_MIN, SNR_MAX, SNR_POW = 4.5, 80.0, 4.0
SEED = 20260613
rng = np.random.default_rng(SEED + (0 if RUN == "O4a" else 1))
log = lambda m: print(f"[pastro-{RUN}] {m}", flush=True)
t0 = time.time()
log(f"device={DEVICE}")

# ---------- frozen scoring stack ----------
asd = {d: ip.load_detector_asd_o1(PREP, d) for d in ("H1", "L1")}
def whiten(raw, det):
    return np.asarray(ip.whiten_batch_gwpy_o1(np.asarray(raw, np.float32), [det] * len(raw), asd, True, "o1"), np.float32)
def build_qt(white):
    qi = ip.center_crop_waveforms(np.asarray(white, np.float32), sample_rate=FS, context_seconds=2.0)
    args = [(w, FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for w in qi]
    with Pool(16) as p:
        mags = p.map(ip._compute_qt_image_worker, args)
    return ip.min_max_norm(np.stack([zoom(m, (256 / m.shape[0], 128 / m.shape[1]), order=1) for m in mags]).astype(np.float32)).astype(np.float32)
cal = json.load(open(f"{SC}/massive_calibration_BA.json"))
cae = ip.BaselineCAE(dropout=0.20).to(DEVICE); cae.load_state_dict(torch.load(cal["model_path"], map_location=DEVICE)); cae.eval()
# sigma-norm: O3a is the anchor run (cal sigma_norm); other runs have per-run norms
sn = cal["sigma_norm"] if RUN == "O3a" else json.load(open(f"{OUT}/{RUN.lower()}_sigma_norm.json"))
seeds = L.load_seeds(DEVICE)
def cae_sigma(qts, det):
    mu_, sd_ = (sn["muH"], sn["sdH"]) if det == "H1" else (sn["muL"], sn["sdL"])
    out = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(torch.from_numpy(qts[:, None])), batch_size=128):
            out.extend(ip.compute_reconstruction_loss(cae, x.to(DEVICE)).cpu().numpy())
    return (np.asarray(out, np.float64) - mu_) / sd_
def ens_g(qts):
    _, e = L.score_all_seeds(seeds, qts, DEVICE); return e
def _band(x, lo=20., hi=400.):
    X = np.fft.rfft(x * np.hanning(x.shape[1])[None, :], axis=1); f = np.fft.rfftfreq(x.shape[1], 1. / FS)
    X[:, (f < lo) | (f > hi)] = 0; return np.fft.irfft(X, n=x.shape[1], axis=1)
def ba_coh(h, l, sec=1.0):
    c = h.shape[1] // 2; half = int(sec * FS / 2)
    x = _band(h[:, c - half:c + half].astype(np.float64)); y = _band(l[:, c - half:c + half].astype(np.float64))
    x -= x.mean(1, keepdims=True); y -= y.mean(1, keepdims=True); ex = (x * x).sum(1); out = np.zeros(len(x), np.float32)
    for lag in range(-LAG, LAG + 1):
        ys = np.roll(y, lag, axis=1); out = np.maximum(out, (np.abs(2 * (x * ys).sum(1)) / (ex + (ys * ys).sum(1) + 1e-30)).astype(np.float32))
    return out
def centroids(white):
    cc = white.shape[1] // 2; w = white[:, cc - FS // 4:cc + FS // 4].astype(np.float64); w -= w.mean(1, keepdims=True)
    Pw = np.abs(np.fft.rfft(w * np.hanning(w.shape[1])[None, :], axis=1)) ** 2; f = np.fft.rfftfreq(w.shape[1], 1. / FS)
    b = (f >= 20) & (f <= 400); return ((Pw[:, b] * f[b]).sum(1) / (Pw[:, b].sum(1) + 1e-30)).astype(np.float32)

# quantile-align map (p7g): source = run bg g (sig>=1) -> O3a ensemble target.
# O3a IS the target run -> identity map (its deploy g is already in the reference frame).
sgo3 = np.load(f"{P1V4E}/segment_g.npz"); o3seg = np.load(f"{P0}/segment_table.npz")
tgt = {d: np.quantile(sgo3[f"raw_{d}_deploy"][o3seg[f"sigma_{d}"] >= 1.0], QG) for d in ("H1", "L1")}
if RUN == "O3a":
    qmap = {d: (tgt[d], tgt[d]) for d in ("H1", "L1")}
else:
    graw = np.load(f"{OUT}/v5_g_{RUN.lower()}_raw.npz"); rseg = np.load(f"{OUT}/{RUN.lower()}_segment_table.npz")
    qmap = {d: (np.quantile(graw[f"g_{d}_raw"][rseg[f"sigma_{d}"] >= 1.0], QG), tgt[d]) for d in ("H1", "L1")}
def amap(x, d): return np.interp(np.asarray(x, float), qmap[d][0], qmap[d][1], left=qmap[d][1][0], right=qmap[d][1][-1])
m5 = json.load(open(f"{OUT}/lr_model_v5.json")); mu5, sd5, be5, co5 = np.array(m5["mu"]), np.array(m5["sd"]), np.array(m5["beta"]), m5["cols"]
def gate(g, s): return np.clip(g, -GCLIP, GCLIP) * np.clip(np.asarray(s) / 3.0, 0, 1)
def lr(F): Fc = np.asarray(F)[:, co5]; return np.column_stack([np.ones(len(Fc)), (Fc - mu5) / sd5]) @ be5

# ---------- noise pool (run prepared dataset), split into 2 disjoint halves ----------
noise = {"H1": [], "L1": []}
for split in ("train", "test", "val"):
    arr = np.load(f"{PREP}/noise_{split}.npy").astype(np.float32)
    meta = ip.load_noise_metadata(f"{PREP}/noise_{split}_metadata.csv")
    for i, r in enumerate(meta):
        noise[r["detector"]].append(arr[i])
nH = np.stack(noise["H1"]); nL = np.stack(noise["L1"]); nN = min(len(nH), len(nL))
log(f"noise pool: {nN} H1 / {len(nL)} L1 segments")
SEGLEN = nH.shape[1]

# ---------- proposal injection set from the banks ----------
def load_bank(path, csv):
    b = ip.load_o1_signal_bank(path); df = pd.read_csv(csv)
    by = {int(s): i for i, s in enumerate(b["source_ids"])}
    rows = []
    for _, r in df.iterrows():
        sid = int(r["source_id"])
        if sid in by:
            rows.append((b["H1"][by[sid]], b["L1"][by[sid]], float(r["mass1"]), float(r["mass2"]),
                         float(r["spin1z"]), float(r["spin2z"])))
    return rows
banks = load_bank(POOL_P1, f"{BANK}/p1_bank_parameters.csv") \
        + load_bank(UM, f"{BANK}/ultramassive_bank_parameters.csv") * UM_OVERSAMPLE
log(f"proposal sources: {len(banks)} (UM x{UM_OVERSAMPLE})")
order = rng.permutation(len(banks))
NMAX = int(os.environ.get("P7_NMAX", "0"))
if NMAX > 0:
    order = order[:NMAX]; log(f"SMOKE TEST: limiting to {NMAX} injections")

# ---------- inject + score in batches ----------
def draw_snr(n):
    # p(rho) ~ rho^-POW on [SNR_MIN, SNR_MAX] (inverse-CDF)
    a, b, p = SNR_MIN, SNR_MAX, SNR_POW
    u = rng.uniform(0, 1, n)
    return (a ** (1 - p) + u * (b ** (1 - p) - a ** (1 - p))) ** (1.0 / (1 - p))

BATCH = 3000
recs = {k: [] for k in ("m1", "m2", "mtot", "snr_net", "sig_H1", "sig_L1", "coh", "g_H1", "g_L1",
                         "loglr", "net", "gated", "noise_half")}
def opt_snr(wave, det):
    s = ip.compute_optimal_snr(np.asarray(wave, np.float32), asd[det]); return float(s)
for b0 in range(0, len(order), BATCH):
    idx = order[b0:b0 + BATCH]
    snr_t = draw_snr(len(idx))
    rawH, rawL, meta_b = [], [], []
    for j, k in enumerate(idx):
        wH, wL, m1, m2, s1z, s2z = banks[int(k)]
        rb = float(np.sqrt(max(opt_snr(wH, "H1"), 1e-9) ** 2 + max(opt_snr(wL, "L1"), 1e-9) ** 2))
        fac = snr_t[j] / max(rb, 1e-9)
        ci = int(rng.integers(0, nN)); half = int(ci % 2)
        xH = nH[ci] + ip.place_signal_in_segment(np.asarray(wH, np.float32) * np.float32(fac), SEGLEN, rng=None)
        xL = nL[ci] + ip.place_signal_in_segment(np.asarray(wL, np.float32) * np.float32(fac), SEGLEN, rng=None)
        rawH.append(xH); rawL.append(xL); meta_b.append((m1, m2, snr_t[j], half))
    wH = whiten(np.stack(rawH), "H1"); wL = whiten(np.stack(rawL), "L1")
    qH, qL = build_qt(wH), build_qt(wL)
    sH = cae_sigma(qH, "H1"); sL = cae_sigma(qL, "L1")
    gH = amap(ens_g(qH), "H1"); gL = amap(ens_g(qL), "L1")
    coh = ba_coh(wH, wL); cH, cL = centroids(wH), centroids(wL)
    net = (sH + sL) / sqrt2
    F = np.column_stack([sH, sL, coh, cH, cL, gate(gH, sH), gate(gL, sL)])
    s = lr(F)
    for j, (m1, m2, snrt, half) in enumerate(meta_b):
        recs["m1"].append(m1); recs["m2"].append(m2); recs["mtot"].append(m1 + m2)
        recs["snr_net"].append(snrt); recs["sig_H1"].append(sH[j]); recs["sig_L1"].append(sL[j])
        recs["coh"].append(float(coh[j])); recs["g_H1"].append(float(gH[j])); recs["g_L1"].append(float(gL[j]))
        recs["loglr"].append(float(s[j])); recs["net"].append(float(net[j]))
        recs["gated"].append(bool(net[j] >= 4.0)); recs["noise_half"].append(half)
    log(f"  {b0 + len(idx)}/{len(order)} injected ({time.time()-t0:.0f}s); "
        f"gated so far {int(np.sum(recs['gated']))}")
np.savez(f"{OUT}/pastro_inj_{RUN}.npz", **{k: np.array(v) for k, v in recs.items()},
         snr_pow=SNR_POW, snr_min=SNR_MIN, snr_max=SNR_MAX, run=RUN)
open(f"{OUT}/PASTRO_INJ_{RUN}_DONE", "w").close()
ng = int(np.sum(recs["gated"]))
log(f"DONE: {len(recs['gated'])} injections, {ng} gated ({ng/len(recs['gated'])*100:.0f}%) in {(time.time()-t0)/60:.0f} min")
