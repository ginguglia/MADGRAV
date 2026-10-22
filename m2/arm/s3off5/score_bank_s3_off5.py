"""_off5 copy of m2/rank/score_bank_s3.py (2026-10-02). Changes: bgfeats read from R_RANK_DIR (madgrav2/s3_off5/rank); zero-lag
triggers from the _off5 merge's blindscan.json (every catalogue-matched trigger is scored even below the lnL>=3 floor, so the
read-out can quote S3 for the net-sigma detections too); output to R_OUT (madgrav2/s3_off5/s3bank). Models = the PRODUCTION S3
ensemble (madgrav2_spec/models, hm 20-300 Hz, 2 folds x 5 seeds); crop path, fold rule and June tripwire unchanged.
Env: _off5 campaign env, SM_RUN, R_TASK, R_NTASK, R_OUT, R_RANK_DIR, R_MAX (smoke)."""
import os, sys, json, time, hashlib, numpy as np
ROOT = os.environ["MADGRAV_ROOT"]
for _p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/glitch_arm"): sys.path.insert(0, os.path.join(ROOT, _p))
RUN = os.environ["SM_RUN"]; TASK = int(os.environ.get("R_TASK", "0")); NTASK = int(os.environ.get("R_NTASK", "16")); RMAX = int(os.environ.get("R_MAX", "0"))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

OUT = os.environ["R_OUT"]; MD = MADGRAV_SCRATCH + "/madgrav2_spec/models"; D = os.environ["R_RANK_DIR"]
assert "madgrav2_spec" not in OUT and "madgrav2_spec" not in D, (OUT, D)
FLO, FHI = 20.0, 500.0; HLO, HHI = 20.0, 300.0; BATCH = 256; FLOOR = 3.0; SEEDS = (0, 1, 2, 3, 4)
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2

def main():
    import det_flags; _F = det_flags.apply()
    import torch, torch.nn as nn, driver_blindscan as B, driver_streams as DS, improved_pipeline as ip, morph_roi as mr
    FS = B.FS; WT = B.WT; DEV = B.DEV; pipe = B.cpipe(); os.makedirs(OUT, exist_ok=True)
    class Net2(nn.Module):
        def __init__(s):
            super().__init__(); ch = [2, 16, 32, 64, 128]
            s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
            s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)); s.register_buffer("temperature", torch.ones(1))
        def forward(s, x):
            for bl in s.b: x = bl(x)
            return s.h(x).squeeze(-1)
    ENS = {}
    for f in (0, 1):
        ENS[f] = []
        for s_ in SEEDS:
            m = Net2().to(DEV); m.load_state_dict(torch.load(f"{MD}/spec_hm_f{f}_s{s_}_nfull.pt", map_location=DEV)); m.eval(); ENS[f].append(m)
    # ---- pairs: bank rows of this task + (task 0) the zero-lag triggers with lnL >= 3 (and every catalogue-matched trigger) ----
    Bf = np.load(f"{D}/bgfeats_{RUN}.npz", allow_pickle=True); ok = np.where(np.isfinite(Bf["bg_loglr"]) & (Bf["bg_loglr"] >= FLOOR))[0][TASK::NTASK]
    if RMAX: ok = ok[:RMAX]
    P = dict(H1_seg=list(Bf["bg_H1_seg"][ok]), iH=list(Bf["bg_iH"][ok].astype(int)), L1_seg=list(Bf["bg_L1_seg"][ok]), iL=list(Bf["bg_iL"][ok].astype(int)), gps_H=list(Bf["bg_gps_H"][ok]), gps_L=list(Bf["bg_gps_L"][ok]),
             loglr=list(Bf["bg_loglr"][ok]), cnn_hm=list(Bf["bg_cnn_hm"][ok]), cnn_lm=list(Bf["bg_cnn_lm"][ok]), kind=["bank"] * len(ok)); ntrig = 0
    if TASK == 0:
        import driver_search_multi as M
        names = {M.segname(a, nm): None for (a, b, d_, nm) in M.SEG["segments"]}
        for t in json.load(open(f"{M.OUT}/blindscan.json"))["triggers"]:
            if t.get("loglr") is None or (t["loglr"] < FLOOR and not t.get("matches_known")) or t["seg"] not in names: continue
            z = np.load(f"{M.S_BG}/{t['seg']}_H1_meta.npz"); i = int(np.argmin(np.abs(z["gps"] - t["gps"])))
            for k_, v_ in (("H1_seg", t["seg"]), ("iH", i), ("L1_seg", t["seg"]), ("iL", i), ("gps_H", float(z["gps"][i])), ("gps_L", float(z["gps"][i])), ("loglr", float(t["loglr"])),
                           ("cnn_hm", t.get("cnn_hm") if t.get("cnn_hm") is not None else np.nan), ("cnn_lm", t.get("cnn_lm") if t.get("cnn_lm") is not None else np.nan), ("kind", "trig")): P[k_].append(v_)
            ntrig += 1
    n = len(P["iH"]); print(f"[s3] {RUN} task {TASK}/{NTASK}: {n - ntrig} bank pairs + {ntrig} zero-lag triggers; dev {DEV}; det_flags {_F}", flush=True)
    fax = ip.QTRANSFORM_FRANGE[0] + 0.5 * np.arange(10000); bandf = fax[(fax >= FLO) & (fax <= FHI)]
    hm_rows = (bandf >= B.FLO) & (bandf <= B.FHI); lm_rows = (bandf >= B.LFLO) & (bandf <= B.LFHI); s3_rows = (bandf >= HLO) & (bandf <= HHI); net_hm, net_lm = B.cnet(), B.lmnet()
    def crop_raw(mag, a, T):
        m = mag[(fax[:mag.shape[0]] >= FLO) & (fax[:mag.shape[0]] <= FHI)]
        o = np.zeros((m.shape[0], WT), dtype=np.float32); sa = max(0, a); sb = min(T, a + WT)
        if sb > sa: o[:, sa - a:sb - a] = m[:, sa:sb]
        return o
    def norm(o): return ((o - o.min()) / (o.max() - o.min() + 1e-9)).astype(np.float32)
    def legs(keys, det, want_t0):
        raw = np.stack([B._win(s, det, int(i)) for s, i in keys]); wh = pipe._whiten(raw, det)
        t0 = np.asarray(mr.cam_t0_batch(B.carm(), DS.build_qt(pipe, wh), DEV), int) if want_t0 else np.zeros(len(keys), int)
        qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
        args = [(qi[k], FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for k in range(len(keys))]
        mg = DS.pool().map(ip._compute_qt_image_worker, args, chunksize=max(1, len(args) // (DS._QT_WORKERS * 4)))
        return {k: (np.asarray(m, np.float32), int(t)) for k, m, t in zip(keys, mg, t0)}
    def ens(x, f):
        with torch.no_grad(): return float(np.mean([torch.sigmoid(m(x) / m.temperature).item() for m in ENS[f]]))
    p0 = np.full(n, np.nan); p1 = np.full(n, np.nan); s3 = np.full(n, np.nan); foldH = np.zeros(n, int); foldL = np.zeros(n, int); dmax = 0.0; nchk = 0; t_start = time.time()
    for b0 in range(0, n, BATCH):
        ib = np.arange(b0, min(b0 + BATCH, n))
        kH = list(dict.fromkeys((P["H1_seg"][i], P["iH"][i]) for i in ib)); kL = list(dict.fromkeys((P["L1_seg"][i], P["iL"][i]) for i in ib))
        LH = legs(kH, "H1", True); LL = legs(kL, "L1", False)
        for i in ib:
            mH, t0 = LH[(P["H1_seg"][i], P["iH"][i])]; mL, _ = LL[(P["L1_seg"][i], P["iL"][i])]
            T = mH.shape[1]; a = B.crop_start(mH, mL, t0, T); cH = crop_raw(mH, a, T); cL = crop_raw(mL, a, T)   # SM_CROP switch (gradcam = frozen)
            if nchk < 16 and P["kind"][i] == "bank" and np.isfinite(P["cnn_hm"][i]):
                with torch.no_grad():
                    xh = torch.from_numpy(np.stack([norm(cH[hm_rows]), norm(cL[hm_rows])])[None]).float().to(DEV); xl = torch.from_numpy(np.stack([norm(cH[lm_rows]), norm(cL[lm_rows])])[None]).float().to(DEV)
                    hm, lm = float(torch.sigmoid(net_hm(xh)).item()), float(torch.sigmoid(net_lm(xl)).item())
                d_ = max(abs(hm - P["cnn_hm"][i]), abs(lm - P["cnn_lm"][i])); dmax = max(dmax, d_); nchk += 1
                if d_ > 1e-3: raise SystemExit(f"[s3] TRIPWIRE: crop path does not reproduce the production CNN cache ({hm:.4f},{lm:.4f}) vs ({P['cnn_hm'][i]:.4f},{P['cnn_lm'][i]:.4f})")
                if nchk == 16: print(f"[s3] tripwire OK on 16 bank pairs (max |diff| {dmax:.2e})", flush=True)
            x = torch.from_numpy(np.stack([norm(cH[s3_rows]), norm(cL[s3_rows])])[None]).float().to(DEV)
            fH = fold_of(P["gps_H"][i] // 86400); fL = fold_of(P["gps_L"][i] // 86400); foldH[i] = fH; foldL[i] = fL
            # models trained on fold F must not see a pair whose legs lie in fold F -> use ENS[1-f] for a fold-pure pair, mean of both if mixed
            if fH == fL: s3[i] = ens(x, 1 - fH); (p0 if fH == 1 else p1).__setitem__(i, s3[i])
            else: p0[i] = ens(x, 0); p1[i] = ens(x, 1); s3[i] = 0.5 * (p0[i] + p1[i])
        if (b0 // BATCH) % 20 == 0: print(f"[s3] {b0 + len(ib)}/{n} pairs ({(time.time()-t_start)/60:.1f} min)", flush=True)
    np.savez(f"{OUT}/s3_{RUN}_t{TASK}.npz", run=RUN, **{k: np.array(v) for k, v in P.items()}, s3=s3, p_ens0=p0, p_ens1=p1, foldH=foldH, foldL=foldL)
    print(f"[s3] DONE {n} pairs in {(time.time()-t_start)/60:.1f} min; mixed-fold pairs {int((foldH != foldL).sum())}; tripwire max |diff| {dmax:.2e} -> s3_{RUN}_t{TASK}.npz", flush=True)

if __name__ == "__main__":
    main()
