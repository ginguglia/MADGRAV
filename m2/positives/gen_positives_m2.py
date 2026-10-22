"""MADGRAV2 positive population (measurement set, not a training set yet).
Injections of bank waveforms (sig m1 10-120 | heavy 150-800 | lowmass Mtot 10-22) into real quiet carriers of run SM_RUN,
BOTH detectors per injection (independent carriers, time-slide style), total mass flat-in-log from M2_MT_LO (15) to 800,
amplitude set by a TARGET NETWORK OPTIMAL SNR drawn log-uniform in [M2_SNR_LO, M2_SNR_HI] against the frozen reference PSD
(inject.py convention, pipe.asd), injected PRE-whitening at the window centre +-0.4 s (g8 convention).
Every draw is kept (no CAE-survivor selection) with the frozen CAE sigma per detector and pass flags, so the CAE pass curve
vs Mtot and SNR comes for free, and the tiles of faint light injections are available for the centroid tail.
Tile = frozen model input (256x128, min-max), same builder as production (driver_streams.build_qt). Read-only on MADGRAV1.
Env: SM_RUN, MADGRAV_ROOT (=MADGRAV1 for data/model/prep), SM_STREAMS, SM_STRAIN (from launchers/templates/_common.sh),
     M2_N pairs (5000), M2_SNR_LO/HI (5, 30), M2_MT_LO (15), M2_MT_HI (800), M2_SHARD (1000), M2_SEED, M2_OUT, M2_TAG.
"""
import os, sys, glob, csv, json, hashlib, time
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ROOT = os.environ.get("MADGRAV_ROOT", MADGRAV_SCRATCH + "/MADGRAV1")
for _p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/glitch_arm"): sys.path.insert(0, os.path.join(ROOT, _p))
import det_flags; _F = det_flags.apply()
import driver_blindscan as B, driver_streams as DS, improved_pipeline as ip

RUN = os.environ["SM_RUN"]; FS = B.FS; WN = 4 * FS
N = int(os.environ.get("M2_N", "5000")); SHARD = int(os.environ.get("M2_SHARD", "1000"))
SNR_LO, SNR_HI = float(os.environ.get("M2_SNR_LO", "5")), float(os.environ.get("M2_SNR_HI", "30"))
MT_LO, MT_HI = float(os.environ.get("M2_MT_LO", "15")), float(os.environ.get("M2_MT_HI", "800"))
SEED = int(os.environ.get("M2_SEED", "20260917")); TAG = os.environ.get("M2_TAG", "")
OUT = os.environ.get("M2_OUT", f"{MADGRAV_SCRATCH}/madgrav2/positives"); os.makedirs(OUT, exist_ok=True)
BANKS = {"sig": f"{ROOT}/data/o1_o3_signal_bank_projected_2s_x10", "heavy": f"{ROOT}/data/heavy_bank_v1", "lowmass": f"{ROOT}/data/lowmass_bank_v1"}
FLOOR = 4.0
OFF = float(os.environ.get("M2_OFF", "0.4"))   # merger placement half-range in s around the window centre

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

def load_bank(d):
    H, L, rows = [], [], []
    for npz in sorted(glob.glob(os.path.join(d, "**", "signals_*.npz"), recursive=True)):
        z = np.load(npz); H.append(np.asarray(z["H1"], np.float32)); L.append(np.asarray(z["L1"], np.float32))
        with open(os.path.splitext(npz)[0] + ".csv", newline="") as fh:
            r = list(csv.DictReader(fh))
        assert len(r) == len(z["H1"]), npz
        rows += [(float(x.get("total_mass") or float(x["mass1"]) + float(x["mass2"])), float(x["mass1"]), float(x["mass2"])) for x in r]
    return np.concatenate(H), np.concatenate(L), np.array(rows)

def carriers_of(det, rng, n=3000, per_seg=4):
    metas = sorted(glob.glob(f"{os.environ['SM_STREAMS']}/*_{det}_meta.npz")); out = []
    for k in rng.permutation(len(metas)):
        m = metas[int(k)]; seg = os.path.basename(m).replace(f"_{det}_meta.npz", ""); z = np.load(m)
        ok = np.where((np.abs(z["sigma"]) >= 0.5) & (np.abs(z["sigma"]) < 1.5))[0]
        if not len(ok): continue
        for j in rng.choice(ok, size=min(per_seg, len(ok)), replace=False): out.append((seg, int(j), float(z["gps"][j])))
        if len(out) >= n: break
    print(f"[carriers {det}] {len(out)} quiet windows from {len({s for s,_,_ in out})} segments of {len(metas)}", flush=True)
    return out

def main():
    rng = np.random.default_rng(SEED)
    pipe = B.cpipe()
    prov = dict(cae_sd_hash=_sha(f"{ROOT}/assets/models/baseline_cae_weaksup_best.pt"), cae_cal_hash=_sha(f"{ROOT}/spectrogram_cascade/massive_calibration_BA.json"),
                floor=FLOOR, tile_geom="256x128_zoom_minmax", harness_sha=_sha(os.path.abspath(__file__)), seed=SEED, run=RUN, snr_lo=SNR_LO, snr_hi=SNR_HI, mt_lo=MT_LO, mt_hi=MT_HI)
    print(f"[m2pos] run {RUN} N={N} SNR log-uniform [{SNR_LO},{SNR_HI}] Mtot flat-in-log [{MT_LO},{MT_HI}] seed {SEED}; CAE sd {prov['cae_sd_hash'][:12]} cal {prov['cae_cal_hash'][:12]}; det_flags {_F}", flush=True)
    banks = {}; rows = []
    for bid, d in BANKS.items():
        H, L, r = load_bank(d); banks[bid] = (H, L)
        for i, (mt, m1, m2) in enumerate(r):
            if MT_LO <= mt <= MT_HI: rows.append((bid, i, mt, m1, m2))
        print(f"[bank {bid}] {len(H)} waveforms, {sum(1 for x in rows if x[0]==bid)} rows in mass range", flush=True)
    mt = np.array([x[2] for x in rows]); lg = np.log(mt); edges = np.linspace(np.log(MT_LO), np.log(MT_HI), 25)
    hist, _ = np.histogram(lg, bins=edges); dens = hist[np.clip(np.digitize(lg, edges) - 1, 0, 23)].astype(float)
    w = 1.0 / np.maximum(dens, 1); w /= w.sum()
    # optimal SNR per waveform at bank amplitude, both detectors, against the frozen reference PSD (cache per row)
    cars = {d: carriers_of(d, rng) for d in ("H1", "L1")}
    fields = ("mtot", "m1", "m2", "bank", "row", "snr_target", "snr_opt_H", "snr_opt_L", "scale", "off_s", "seg_H", "j_H", "gps_H", "day_H",
              "seg_L", "j_L", "gps_L", "day_L", "sigma_H", "sigma_L", "sigma_net", "pass_H", "pass_L", "pass_net")
    buf = {k: [] for k in fields}; buf["qt_H"] = []; buf["qt_L"] = []; shard = 0; done = 0; t0 = time.time(); snr_cache = {}
    def flush():
        nonlocal shard, buf
        if not buf["mtot"]: return
        np.savez(f"{OUT}/pos_{RUN}{TAG}_shard{shard:03d}.npz", **{k: np.array(v) for k, v in buf.items() if not k.startswith("qt")},
                 qt_H=np.stack(buf["qt_H"]).astype(np.float16), qt_L=np.stack(buf["qt_L"]).astype(np.float16), **prov)
        print(f"[m2pos] shard {shard} written ({len(buf['mtot'])} pairs) {(time.time()-t0)/60:.1f} min", flush=True); shard += 1; buf = {k: [] for k in buf}
    while done < N:
        nb = min(256, N - done); raw = {"H1": [], "L1": []}; meta = []
        for _ in range(nb):
            bid, i, m, m1, m2 = rows[int(rng.choice(len(rows), p=w))]
            key = (bid, i)
            if key not in snr_cache:
                snr_cache[key] = (ip.compute_optimal_snr(banks[bid][0][i], pipe.asd["H1"]), ip.compute_optimal_snr(banks[bid][1][i], pipe.asd["L1"]))
            sH, sL = snr_cache[key]; s0 = np.hypot(sH, sL)
            snr = float(np.exp(rng.uniform(np.log(SNR_LO), np.log(SNR_HI)))); sc = snr / s0
            off = float(rng.uniform(-OFF, OFF)); a0 = FS + int(off * FS)   # 2026-09-26: M2_OFF widens the merger placement (default 0.4 = as before)
            cw = {}
            for d, k in (("H1", 0), ("L1", 1)):
                seg, j, gps = cars[d][int(rng.integers(len(cars[d])))]
                r = B._win(seg, d, j).astype(np.float64); x = banks[bid][k][i].astype(np.float64); Lc = min(len(x), WN - a0)
                inj = r.copy(); inj[a0:a0 + Lc] += sc * x[:Lc]; raw[d].append(inj.astype(np.float32)); cw[d] = (seg, j, gps)
            meta.append((m, m1, m2, bid, i, snr, sH * sc, sL * sc, sc, off, cw))
        sig = {}; qt = {}
        for d in ("H1", "L1"):
            wh = pipe._whiten(np.stack(raw[d]), d); qt[d] = DS.build_qt(pipe, wh); sig[d] = np.asarray(DS.sigma_from_qt(pipe, qt[d], d), float)
        for n_, (m, m1, m2, bid, i, snr, soH, soL, sc, off, cw) in enumerate(meta):
            sh, sl = float(sig["H1"][n_]), float(sig["L1"][n_]); sn = (sh + sl) / np.sqrt(2)
            vals = dict(mtot=m, m1=m1, m2=m2, bank=bid, row=i, snr_target=snr, snr_opt_H=soH, snr_opt_L=soL, scale=sc, off_s=off,
                        seg_H=cw["H1"][0], j_H=cw["H1"][1], gps_H=cw["H1"][2], day_H=int(cw["H1"][2] // 86400), seg_L=cw["L1"][0], j_L=cw["L1"][1], gps_L=cw["L1"][2], day_L=int(cw["L1"][2] // 86400),
                        sigma_H=sh, sigma_L=sl, sigma_net=sn, pass_H=int(sh >= FLOOR), pass_L=int(sl >= FLOOR), pass_net=int(sn >= FLOOR))
            for k, v in vals.items(): buf[k].append(v)
            buf["qt_H"].append(qt["H1"][n_]); buf["qt_L"].append(qt["L1"][n_])
        done += nb
        if len(buf["mtot"]) >= SHARD: flush()
        print(f"[m2pos] {done}/{N} pairs, {(time.time()-t0)/60:.1f} min; batch pass_net {np.mean([(sig['H1'][k]+sig['L1'][k])/np.sqrt(2)>=FLOOR for k in range(nb)]):.2f}", flush=True)
    flush()
    # pass curve summary over everything written
    fs = sorted(glob.glob(f"{OUT}/pos_{RUN}{TAG}_shard*.npz")); M, S, P = [], [], []
    for f in fs:
        z = np.load(f); M.append(z["mtot"]); S.append(z["snr_target"]); P.append(z["pass_net"])
    M, S, P = map(np.concatenate, (M, S, P)); print("\n[pass curve] fraction with sigma_net>=4 by Mtot bin x network SNR bin:")
    mb = [15, 24, 38, 60, 95, 150, 240, 380, 800]; sb = [5, 8, 12, 18, 30]
    print("Mtot \\ SNR   " + " ".join(f"{a}-{b:<5}" for a, b in zip(sb[:-1], sb[1:])))
    for a, b in zip(mb[:-1], mb[1:]):
        m = (M >= a) & (M < b); line = f"{a:3d}-{b:<3d}      "
        for c, d in zip(sb[:-1], sb[1:]):
            mm = m & (S >= c) & (S < d); line += f"{P[mm].mean():.2f}({mm.sum():3d}) " if mm.any() else "   -       "
        print(line)
    print("[m2pos] DONE", flush=True)

if __name__ == "__main__":
    main()
