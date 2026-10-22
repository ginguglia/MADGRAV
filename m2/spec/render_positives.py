"""Specialist-retrain POSITIVES (PREREG_SPECIALISTS_20260923 §2 + A1): bank v2 waveforms injected into real quiet carriers
of run SM_RUN, taken through the PRODUCTION MADGRAV2 chain up to the specialists, rendered as native specialist tiles.

Per draw: one bank-v2 waveform (each used ONCE: bank chunk c belongs to run c % 4 = o3a,o3b,o4a,o4b; this task takes every
R_NTASK-th of that run's chunks), target network optimal SNR log-uniform [5, 30] against the frozen reference ASD (full 4 s
bank waveform), merger at window centre + U(-0.4, 0.4) s, SAME placement in both detectors (so the physical H1-L1 delay
carried by the bank survives), carriers = quiet windows (0.5 <= |sigma| < 1.5) of this run, H1 and L1 carrier from the SAME
day-fold (fold drawn uniformly), outside the quarantine.
Chain (production code, MADGRAV2-gx): whiten -> build_qt -> CAE sigma -> centroid2 (clu98) -> arm v2 g2 (cross-fit by day).
ADMISSION = what reaches the specialists: sigma_net >= 3, centroid2 < 220 Hz in both legs, louder-leg g2 >= -2.05.
Admitted pairs are rendered exactly like _pb_cnn_precompute._fullmag_batch (full QT, H1 Grad-CAM t0 of the v1 arm,
placement a = round((t0-14)/128*T_H), both legs cropped at a with T_H) but stored UNNORMALISED over 20-500 Hz
(961 rows x 113 cols, float16); any band crop + min-max at train time reproduces driver_blindscan._crop.
Also stored: S0 (June HM/LM) scores from the same crops. TRIPWIRE: on the first admitted pairs, S0 from the stored crop
must equal driver_blindscan.cnn_from_wh on the same whitened rows (|diff| < 1e-3), else the task aborts.
Output: $R_OUT/pos_<run>_t<task>_s<shard>.npz (tiles + meta) and $R_OUT/draws_<run>_t<task>.npz (every draw, no tiles).
Env: SM_RUN + the campaign env (launcher), R_TASK, R_NTASK, R_SEED, R_OUT, R_SHARD (2000), R_MAXCHUNKS (smoke)."""
import os, sys, glob, time, numpy as np, pandas as pd
ROOT = os.environ["MADGRAV_ROOT"]
for _p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/glitch_arm"): sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quarantine as QU

RUN = os.environ["SM_RUN"]; RUNS = ("o3a", "o3b", "o4a", "o4b"); RIDX = RUNS.index(RUN)
TASK = int(os.environ.get("R_TASK", "0")); NTASK = int(os.environ.get("R_NTASK", "4")); SEED = int(os.environ.get("R_SEED", "20260923"))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

OUT = os.environ.get("R_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/pos"); SHARD = int(os.environ.get("R_SHARD", "2000"))
MAXCH = int(os.environ.get("R_MAXCHUNKS", "0")); BANK = MADGRAV_SCRATCH + "/madgrav2_spec/bank_v2"
SNR_LO, SNR_HI, OFF = 5.0, 30.0, 0.4; NET_MIN, ARM_MIN = 3.0, -2.05; FLO, FHI = 20.0, 500.0; BATCH = 256

def main():
    import det_flags; _F = det_flags.apply()
    import torch, driver_blindscan as B, driver_streams as DS, improved_pipeline as ip, centroid_v2 as C2, arm_v2, morph_roi as mr
    FS = B.FS; WN = 4 * FS; WT = B.WT; DEV = B.DEV
    rng = np.random.default_rng([SEED, RIDX, TASK]); os.makedirs(OUT, exist_ok=True)
    pipe = B.cpipe(); ARM2 = arm_v2.ArmV2(DEV); Q = QU.build()
    print(f"[pos] run {RUN} task {TASK}/{NTASK} seed {SEED} dev {DEV} arm2 {ARM2.n} cen2_cut {C2.CEN2_CUT} det_flags {_F}", flush=True)
    # ---- bank chunks of this run / task ----
    ch = sorted(glob.glob(f"{BANK}/signals_*.csv")); ch = [c for c in ch if int(c[-8:-4]) % 4 == RIDX][TASK::NTASK]
    if MAXCH: ch = ch[:MAXCH]
    print(f"[pos] {len(ch)} bank chunks for this task", flush=True)
    # ---- carriers: quiet windows, per detector and fold, quarantine applied ----
    car = {(d, f): [] for d in ("H1", "L1") for f in (0, 1)}
    for d in ("H1", "L1"):
        for m in sorted(glob.glob(f"{os.environ['SM_STREAMS']}/*_{d}_meta.npz")):
            seg = os.path.basename(m).replace(f"_{d}_meta.npz", ""); z = np.load(m); s = np.abs(z["sigma"]); g = np.asarray(z["gps"], float)
            ok = np.where((s >= 0.5) & (s < 1.5))[0]
            if not len(ok): continue
            ok = ok[~QU.quarantined(Q, d, g[ok])]
            for j in ok: car[(d, QU.fold_of(int(g[j] // 86400)))].append((seg, int(j), float(g[j])))
    print("[pos] carriers " + ", ".join(f"{d}/f{f}: {len(v)}" for (d, f), v in car.items()), flush=True)
    fax = ip.QTRANSFORM_FRANGE[0] + 0.5 * np.arange(10000); bandf = fax[(fax >= FLO) & (fax <= FHI)]   # rows of the stored 20-500 crop
    def crop_raw(mag, a, T):                         # driver_blindscan._crop WITHOUT the min-max norm, 20-500 Hz
        m = mag[(fax[:mag.shape[0]] >= FLO) & (fax[:mag.shape[0]] <= FHI)]
        o = np.zeros((m.shape[0], WT), dtype=np.float32); sa = max(0, a); sb = min(T, a + WT)
        if sb > sa: o[:, sa - a:sb - a] = m[:, sa:sb]
        return o
    def norm(o): return ((o - o.min()) / (o.max() - o.min() + 1e-9)).astype(np.float32)
    hm_rows = (bandf >= B.FLO) & (bandf <= B.FHI); lm_rows = (bandf >= B.LFLO) & (bandf <= B.LFHI)   # S0 bands inside the crop
    net_hm, net_lm = B.cnet(), B.lmnet()
    def s0(crH, crL):
        with torch.no_grad():
            xh = torch.from_numpy(np.stack([norm(crH[hm_rows]), norm(crL[hm_rows])])[None]).float().to(DEV)
            xl = torch.from_numpy(np.stack([norm(crH[lm_rows]), norm(crL[lm_rows])])[None]).float().to(DEV)
            return float(torch.sigmoid(net_hm(xh)).item()), float(torch.sigmoid(net_lm(xl)).item())
    def mags(wh):
        qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
        args = [(qi[k], FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for k in range(len(wh))]
        return [np.asarray(m, np.float32) for m in DS.pool().map(ip._compute_qt_image_worker, args, chunksize=max(1, len(args) // (DS._QT_WORKERS * 4)))]
    D = {k: [] for k in ("chunk", "row", "source_id", "mtot", "q", "dt_LH", "snr_target", "off", "scale", "fold", "seg_H", "j_H", "gps_H", "seg_L", "j_L",
                         "gps_L", "sigma_H", "sigma_L", "sigma_net", "g2_H", "g2_L", "cen2_H", "cen2_L", "admit")}
    A = {k: [] for k in ("idx", "t0", "a", "T_H", "s0_hm", "s0_lm")}; TL = []; shard = 0; tripped = False; t_start = time.time(); ndraw = 0
    def flush(final=False):
        nonlocal shard, TL, A
        if not TL or (len(TL) < SHARD and not final): return
        meta = {k: np.array(v) for k, v in D.items()}; ai = np.array(A["idx"])
        np.savez(f"{OUT}/pos_{RUN}_t{TASK}_s{shard:02d}.npz", tiles=np.stack(TL).astype(np.float16), flo=FLO, fhi=FHI,
                 **{k: meta[k][ai] for k in meta}, **{k: np.array(v) for k, v in A.items() if k != "idx"}, run=RUN)
        print(f"[pos] shard {shard}: {len(TL)} admitted pairs written ({(time.time()-t_start)/60:.1f} min)", flush=True)
        shard += 1; TL = []; A = {k: [] for k in A}
    pending = []
    def process(batch):
        nonlocal tripped
        raw = {"H1": [], "L1": []}; gps_c = {"H1": [], "L1": []}
        for b in batch:
            c = WN // 2 + int(round(b["off"] * FS)); s = c - int(b["merger_index"])
            for d in ("H1", "L1"):
                seg, j, g = b["car"][d]; r = B._win(seg, d, j).astype(np.float64); x = b["w"][d].astype(np.float64) * b["scale"]
                lo, hi = max(0, s), min(WN, s + len(x)); r[lo:hi] += x[lo - s:hi - s]; raw[d].append(r.astype(np.float32)); gps_c[d].append(g)
        wh, qt, sg, c2, g2 = {}, {}, {}, {}, {}
        for d in ("H1", "L1"):
            wh[d] = pipe._whiten(np.stack(raw[d]), d); qt[d] = DS.build_qt(pipe, wh[d]); sg[d] = np.asarray(DS.sigma_from_qt(pipe, qt[d], d), float)
            c2[d] = np.asarray(C2.cluster_centroid(qt[d]), float); g2[d] = np.asarray(ARM2.g2_from_qt(qt[d], np.array(gps_c[d])), float)
        net = (sg["H1"] + sg["L1"]) / np.sqrt(2); lead = np.where(sg["H1"] >= sg["L1"], g2["H1"], g2["L1"])
        adm = (net >= NET_MIN) & C2.cen2_pass(c2["H1"], c2["L1"]) & (lead >= ARM_MIN); ia = np.where(adm)[0]
        base = len(D["mtot"])
        for n_, b in enumerate(batch):
            for k, v in (("chunk", b["chunk"]), ("row", b["row"]), ("source_id", b["sid"]), ("mtot", b["mtot"]), ("q", b["q"]), ("dt_LH", b["dt"]),
                         ("snr_target", b["snr"]), ("off", b["off"]), ("scale", b["scale"]), ("fold", b["fold"]), ("seg_H", b["car"]["H1"][0]),
                         ("j_H", b["car"]["H1"][1]), ("gps_H", b["car"]["H1"][2]), ("seg_L", b["car"]["L1"][0]), ("j_L", b["car"]["L1"][1]),
                         ("gps_L", b["car"]["L1"][2]), ("sigma_H", sg["H1"][n_]), ("sigma_L", sg["L1"][n_]), ("sigma_net", net[n_]),
                         ("g2_H", g2["H1"][n_]), ("g2_L", g2["L1"][n_]), ("cen2_H", c2["H1"][n_]), ("cen2_L", c2["L1"][n_]), ("admit", int(adm[n_]))):
                D[k].append(v)
        if not len(ia): return
        t0 = np.asarray(mr.cam_t0_batch(B.carm(), qt["H1"][ia], DEV), int); mH = mags(wh["H1"][ia]); mL = mags(wh["L1"][ia])
        for k, i in enumerate(ia):
            T = mH[k].shape[1]; a = int(round((t0[k] - 14) / 128 * T)); cH = crop_raw(mH[k], a, T); cL = crop_raw(mL[k], a, T)
            hm, lm = s0(cH, cL)
            if not tripped and k < 8:                    # TRIPWIRE: identical to the production CNN path on the same rows
                hp, lp = B.cnn_from_wh(wh["H1"][i][None, :], wh["L1"][i][None, :])
                if abs(hp - hm) > 1e-3 or abs(lp - lm) > 1e-3: raise SystemExit(f"[pos] TRIPWIRE: stored-crop S0 ({hm:.5f},{lm:.5f}) != cnn_from_wh ({hp:.5f},{lp:.5f})")
                if k == 7: tripped = True; print(f"[pos] tripwire OK: stored crops reproduce cnn_from_wh on 8 pairs", flush=True)
            TL.append(np.stack([cH, cL])); A["idx"].append(base + i); A["t0"].append(int(t0[k])); A["a"].append(a); A["T_H"].append(T); A["s0_hm"].append(hm); A["s0_lm"].append(lm)
    for cf in ch:
        rows = pd.read_csv(cf); z = np.load(cf.replace(".csv", ".npz")); cid = int(cf[-8:-4])
        for r_i in range(len(rows)):
            r = rows.iloc[r_i]; w = {"H1": z["H1"][r_i], "L1": z["L1"][r_i]}
            s0H = ip.compute_optimal_snr(w["H1"], pipe.asd["H1"]); s0L = ip.compute_optimal_snr(w["L1"], pipe.asd["L1"]); s0n = float(np.hypot(s0H, s0L))
            if not np.isfinite(s0n) or s0n <= 0: continue
            snr = float(np.exp(rng.uniform(np.log(SNR_LO), np.log(SNR_HI)))); f = int(rng.integers(2))
            carp = {d: car[(d, f)][int(rng.integers(len(car[(d, f)])))] for d in ("H1", "L1")}
            pending.append(dict(chunk=cid, row=r_i, sid=int(r.source_id), mtot=float(r.total_mass), q=float(r.mass_ratio), dt=float(r.dt_L1_minus_H1_s),
                                merger_index=int(r.merger_index), snr=snr, scale=snr / s0n, off=float(rng.uniform(-OFF, OFF)), fold=f, car=carp, w=w))
            if len(pending) == BATCH: process(pending); ndraw += len(pending); pending = []; flush()
        print(f"[pos] chunk {cid} done: {ndraw} draws, admitted {int(np.sum(D['admit']))} ({np.mean(D['admit']) if D['admit'] else 0:.2f}), "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)
    if pending: process(pending); ndraw += len(pending)
    flush(final=True)
    np.savez(f"{OUT}/draws_{RUN}_t{TASK}.npz", **{k: np.array(v) for k, v in D.items()})
    ad = np.array(D["admit"]); mt = np.array(D["mtot"])
    print(f"[pos] DONE {ndraw} draws, {ad.sum()} admitted ({ad.mean():.3f}); admission by Mtot: " +
          ", ".join(f"{a}-{b}: {ad[(mt>=a)&(mt<b)].mean():.2f}" for a, b in [(10, 30), (30, 100), (100, 300), (300, 800)] if ((mt>=a)&(mt<b)).any()), flush=True)

if __name__ == "__main__":
    main()
