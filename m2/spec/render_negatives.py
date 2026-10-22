"""Specialist-retrain NEGATIVES, stage 2 = RENDER (GPU; PREREG_SPECIALISTS_20260923 §2, A2).
Reads neg_final.npz (repair_negatives.py: campaign pairs + A3 re-pairs), keeps rows of run SM_RUN, takes every R_NTASK-th row from R_TASK, and renders each
time-slid pair EXACTLY like render_positives.py / _pb_cnn_precompute._fullmag_batch: raw 4 s windows via driver_blindscan._win
(H1 at (H1_seg, iH), L1 at (L1_seg, iL)), whiten, build_qt, H1 Grad-CAM t0 of the v1 arm, full QT magnitudes, both legs cropped at
the H1 placement, stored UNNORMALISED over 20-500 Hz (961 x 113, float16). Legs shared by several slid pairs are computed once.
TRIPWIRE: S0 (June HM/LM) recomputed from the stored crops must equal the production CNN cache values carried in neg_select
(cnn_hm / cnn_lm) to |diff| < 1e-3 on the first 16 CAMPAIGN pairs (abort otherwise); the max |diff| over all campaign pairs is reported; re-pairs get S0 only.
Output: $R_OUT/<R_PREFIX|neg>_<run>_t<task>_s<shard>.npz. Env: SM_RUN + campaign env, R_TASK, R_NTASK, R_OUT, R_SHARD (2000), R_MAX (smoke),
R_SEL (selection npz, default neg_final.npz), R_PREFIX."""
import os, sys, time, numpy as np
ROOT = os.environ["MADGRAV_ROOT"]
for _p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/glitch_arm"): sys.path.insert(0, os.path.join(ROOT, _p))
RUN = os.environ["SM_RUN"]; TASK = int(os.environ.get("R_TASK", "0")); NTASK = int(os.environ.get("R_NTASK", "4"))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

OUT = os.environ.get("R_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/neg"); SHARD = int(os.environ.get("R_SHARD", "2000")); RMAX = int(os.environ.get("R_MAX", "0"))
SEL = os.environ.get("R_SEL", MADGRAV_SCRATCH + "/madgrav2_spec/neg_final.npz"); PFX = os.environ.get("R_PREFIX", "neg"); FLO, FHI = 20.0, 500.0; BATCH = 256

def main():
    import det_flags; _F = det_flags.apply()
    import torch, driver_blindscan as B, driver_streams as DS, improved_pipeline as ip, morph_roi as mr
    FS = B.FS; WT = B.WT; DEV = B.DEV; pipe = B.cpipe(); os.makedirs(OUT, exist_ok=True)
    z = np.load(SEL, allow_pickle=True); rows = np.where(z["run"] == RUN)[0][TASK::NTASK]
    if RMAX: rows = rows[:RMAX]
    S = {k: z[k][rows] for k in z.files if k != "seed"}
    print(f"[neg] run {RUN} task {TASK}/{NTASK}: {len(rows)} pairs (campaign S0-pass {int((S['s0_pass']==1).sum())}, re-pairs {int((S['s0_pass']==-1).sum())}) dev {DEV} det_flags {_F}", flush=True)
    fax = ip.QTRANSFORM_FRANGE[0] + 0.5 * np.arange(10000); bandf = fax[(fax >= FLO) & (fax <= FHI)]
    hm_rows = (bandf >= B.FLO) & (bandf <= B.FHI); lm_rows = (bandf >= B.LFLO) & (bandf <= B.LFHI); net_hm, net_lm = B.cnet(), B.lmnet()
    def crop_raw(mag, a, T):
        m = mag[(fax[:mag.shape[0]] >= FLO) & (fax[:mag.shape[0]] <= FHI)]
        o = np.zeros((m.shape[0], WT), dtype=np.float32); sa = max(0, a); sb = min(T, a + WT)
        if sb > sa: o[:, sa - a:sb - a] = m[:, sa:sb]
        return o
    def norm(o): return ((o - o.min()) / (o.max() - o.min() + 1e-9)).astype(np.float32)
    def s0(cH, cL):
        with torch.no_grad():
            xh = torch.from_numpy(np.stack([norm(cH[hm_rows]), norm(cL[hm_rows])])[None]).float().to(DEV)
            xl = torch.from_numpy(np.stack([norm(cH[lm_rows]), norm(cL[lm_rows])])[None]).float().to(DEV)
            return float(torch.sigmoid(net_hm(xh)).item()), float(torch.sigmoid(net_lm(xl)).item())
    def legs(keys, det, want_t0):                    # _fullmag_batch verbatim for unique (seg, idx) legs
        raw = np.stack([B._win(s, det, int(i)) for s, i in keys]); wh = pipe._whiten(raw, det)
        t0 = np.asarray(mr.cam_t0_batch(B.carm(), DS.build_qt(pipe, wh), DEV), int) if want_t0 else np.zeros(len(keys), int)
        qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
        args = [(qi[k], FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for k in range(len(keys))]
        mg = DS.pool().map(ip._compute_qt_image_worker, args, chunksize=max(1, len(args) // (DS._QT_WORKERS * 4)))
        return {k: (np.asarray(m, np.float32), int(t)) for k, m, t in zip(keys, mg, t0)}
    TL = []; A = {k: [] for k in ("sel_row", "t0", "a", "T_H", "s0_hm", "s0_lm")}; shard = 0; dmax = 0.0; nchk = 0; t_start = time.time()
    def flush(final=False):
        nonlocal shard, TL, A
        if not TL or (len(TL) < SHARD and not final): return
        ix = np.array(A["sel_row"])
        np.savez(f"{OUT}/{PFX}_{RUN}_t{TASK}_s{shard:02d}.npz", tiles=np.stack(TL).astype(np.float16), flo=FLO, fhi=FHI,
                 **{k: S[k][ix] for k in S}, **{k: np.array(v) for k, v in A.items()})   # "run" already in S
        print(f"[neg] shard {shard}: {len(TL)} pairs written ({(time.time()-t_start)/60:.1f} min, max|S0 diff| so far {dmax:.2e})", flush=True)
        shard += 1; TL = []; A = {k: [] for k in A}
    for b0 in range(0, len(rows), BATCH):
        ib = np.arange(b0, min(b0 + BATCH, len(rows)))
        kH = list(dict.fromkeys((S["H1_seg"][i], int(S["iH"][i])) for i in ib)); kL = list(dict.fromkeys((S["L1_seg"][i], int(S["iL"][i])) for i in ib))
        LH = legs(kH, "H1", True); LL = legs(kL, "L1", False)
        for i in ib:
            mH, t0 = LH[(S["H1_seg"][i], int(S["iH"][i]))]; mL, _ = LL[(S["L1_seg"][i], int(S["iL"][i]))]
            T = mH.shape[1]; a = int(round((t0 - 14) / 128 * T)); cH = crop_raw(mH, a, T); cL = crop_raw(mL, a, T); hm, lm = s0(cH, cL)
            if np.isfinite(S["cnn_hm"][i]):              # campaign pair: production cache value exists (re-pairs have none)
                d_ = max(abs(hm - float(S["cnn_hm"][i])), abs(lm - float(S["cnn_lm"][i]))); dmax = max(dmax, d_); nchk += 1
                if nchk <= 16 and d_ > 1e-3:
                    raise SystemExit(f"[neg] TRIPWIRE row {i}: stored-crop S0 ({hm:.5f},{lm:.5f}) != production cache ({S['cnn_hm'][i]:.5f},{S['cnn_lm'][i]:.5f})")
                if nchk == 16: print("[neg] tripwire OK: stored crops reproduce the production CNN cache on 16 campaign pairs", flush=True)
            TL.append(np.stack([cH, cL])); A["sel_row"].append(int(i)); A["t0"].append(t0); A["a"].append(a); A["T_H"].append(T); A["s0_hm"].append(hm); A["s0_lm"].append(lm)
        flush()
    flush(final=True)
    print(f"[neg] DONE {len(rows)} pairs in {(time.time()-t_start)/60:.1f} min; max |S0(stored crop) - production cache| = {dmax:.2e}", flush=True)

if __name__ == "__main__":
    main()
