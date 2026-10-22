"""Specialist-retrain tiles with GENERATED GLITCHES (2026-09-30, analysis lead plan "start from the glitches"). Two modes:
  R_MODE=gneg : GlitchGen glitch (raw strain, 2 s @ 4096 Hz) added to real QUIET carriers of run SM_RUN, NO chirp -> NEGATIVES.
                One leg gets a glitch (the other stays quiet) with P = 1-R_PBOTH, both legs independent glitches with P = R_PBOTH (0.2).
  R_MODE=gpos : render_positives.py draw (bank-v2 waveform on quiet carriers) + a glitch on each leg with P = R_GFRAC (0.3) -> POSITIVES.
Glitch amplitude: optimal SNR log-uniform [R_GSNR_LO, R_GSNR_HI] (8, 80) against the frozen reference ASD; centre anywhere in [1, 3] s of
the 4 s window. Chain, admission, rendering, S0 tripwire and output schema identical to render_positives.py (961 x 113 float16 crops over
20-500 Hz); gneg shards carry the neg schema of assemble_spec.py (source = 'gg_<clsH>|<clsL>', loglr = nan, cnn_* = S0, s0_pass).
Env: SM_RUN + campaign env, R_MODE, R_TASK, R_NTASK, R_SEED, R_OUT, R_SHARD (2000), R_NDRAW (gneg draws per task, 5000), R_MAXCHUNKS (gpos smoke),
     R_GG_BANK (glitchgen_v1), R_GG_CLASSES (comma list; default = every class dir incl. param/), R_GG_MAXCHUNK (chunk files per class, 2)."""
import os, sys, glob, time, numpy as np, pandas as pd
ROOT = os.environ["MADGRAV_ROOT"]
for _p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/glitch_arm"): sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quarantine as QU

RUN = os.environ["SM_RUN"]; RUNS = ("o3a", "o3b", "o4a", "o4b"); RIDX = RUNS.index(RUN); MODE = os.environ.get("R_MODE", "gneg"); assert MODE in ("gneg", "gpos")
TASK = int(os.environ.get("R_TASK", "0")); NTASK = int(os.environ.get("R_NTASK", "2")); SEED = int(os.environ.get("R_SEED", "20260930"))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

OUT = os.environ.get("R_OUT", f"{MADGRAV_SCRATCH}/madgrav2_spec/{'neg_gneg' if MODE == 'gneg' else 'pos_gpos'}"); SHARD = int(os.environ.get("R_SHARD", "2000"))
NDRAW = int(os.environ.get("R_NDRAW", "5000")); MAXCH = int(os.environ.get("R_MAXCHUNKS", "0")); BANK = MADGRAV_SCRATCH + "/madgrav2_spec/bank_v2"
GG_BANK = os.environ.get("R_GG_BANK", MADGRAV_SCRATCH + "/madgrav2/arm/synth/glitchgen_v1"); GG_CLASSES = [c for c in os.environ.get("R_GG_CLASSES", "").split(",") if c]
GG_MAXCHUNK = int(os.environ.get("R_GG_MAXCHUNK", "2")); GSNR = (float(os.environ.get("R_GSNR_LO", "8")), float(os.environ.get("R_GSNR_HI", "80")))
PBOTH = float(os.environ.get("R_PBOTH", "0.2")); GFRAC = float(os.environ.get("R_GFRAC", "0.3"))
SNR_LO, SNR_HI, OFF = 5.0, 30.0, 0.4; NET_MIN, ARM_MIN = 3.0, -2.05; FLO, FHI = 20.0, 500.0; BATCH = 256

def gg_load():
    import h5py; bank = {}
    for f in sorted(glob.glob(f"{GG_BANK}/*/*_chunk*.hdf5") + glob.glob(f"{GG_BANK}/param/*/*_chunk*.hdf5")):
        cls = os.path.basename(os.path.dirname(f))
        if GG_CLASSES and cls not in GG_CLASSES: continue
        if len(bank.get(cls, [])) >= GG_MAXCHUNK: continue
        with h5py.File(f) as h: bank.setdefault(cls, []).append(np.asarray(h["strain"], np.float32))
    bank = {c: np.concatenate(v) for c, v in bank.items()}
    if not bank: raise SystemExit(f"no GlitchGen chunks under {GG_BANK}")
    print(f"[{MODE}] glitch bank: {len(bank)} classes, " + ", ".join(f"{c}:{len(v)}" for c, v in bank.items()), flush=True); return bank

def main():
    import det_flags; _F = det_flags.apply()
    import torch, driver_blindscan as B, driver_streams as DS, improved_pipeline as ip, centroid_v2 as C2, arm_v2, morph_roi as mr
    FS = B.FS; WN = 4 * FS; WT = B.WT; DEV = B.DEV
    rng = np.random.default_rng([SEED, RIDX, TASK, 7 if MODE == "gneg" else 11]); os.makedirs(OUT, exist_ok=True)
    pipe = B.cpipe(); ARM2 = arm_v2.ArmV2(DEV); Q = QU.build(); GG = gg_load(); GCLS = sorted(GG)
    print(f"[{MODE}] run {RUN} task {TASK}/{NTASK} seed {SEED} dev {DEV} arm2 {ARM2.n} cen2_cut {C2.CEN2_CUT} det_flags {_F} gsnr {GSNR} pboth {PBOTH} gfrac {GFRAC}", flush=True)
    car = {(d, f): [] for d in ("H1", "L1") for f in (0, 1)}
    for d in ("H1", "L1"):
        for m in sorted(glob.glob(f"{os.environ['SM_STREAMS']}/*_{d}_meta.npz")):
            seg = os.path.basename(m).replace(f"_{d}_meta.npz", ""); z = np.load(m); s = np.abs(z["sigma"]); g = np.asarray(z["gps"], float)
            ok = np.where((s >= 0.5) & (s < 1.5))[0]
            if not len(ok): continue
            ok = ok[~QU.quarantined(Q, d, g[ok])]
            for j in ok: car[(d, QU.fold_of(int(g[j] // 86400)))].append((seg, int(j), float(g[j])))
    print(f"[{MODE}] carriers " + ", ".join(f"{d}/f{f}: {len(v)}" for (d, f), v in car.items()), flush=True)
    fax = ip.QTRANSFORM_FRANGE[0] + 0.5 * np.arange(10000); bandf = fax[(fax >= FLO) & (fax <= FHI)]
    def crop_raw(mag, a, T):
        m = mag[(fax[:mag.shape[0]] >= FLO) & (fax[:mag.shape[0]] <= FHI)]
        o = np.zeros((m.shape[0], WT), dtype=np.float32); sa = max(0, a); sb = min(T, a + WT)
        if sb > sa: o[:, sa - a:sb - a] = m[:, sa:sb]
        return o
    def norm(o): return ((o - o.min()) / (o.max() - o.min() + 1e-9)).astype(np.float32)
    hm_rows = (bandf >= B.FLO) & (bandf <= B.FHI); lm_rows = (bandf >= B.LFLO) & (bandf <= B.LFHI)
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
    gsnr_cache = {}
    def draw_glitch(d):
        cls = GCLS[int(rng.integers(len(GCLS)))]; i = int(rng.integers(len(GG[cls]))); g = GG[cls][i].astype(np.float64)
        key = (cls, i, d)
        if key not in gsnr_cache: gsnr_cache[key] = float(ip.compute_optimal_snr(g, pipe.asd[d]))
        snr = float(np.exp(rng.uniform(np.log(GSNR[0]), np.log(GSNR[1])))); t0 = int(rng.integers(0, WN - len(g) + 1))
        return dict(cls=cls, i=i, snr=snr, scale=snr / max(gsnr_cache[key], 1e-30), t0=t0, g=g)
    POSK = ("chunk", "row", "source_id", "mtot", "q", "dt_LH", "snr_target", "off", "scale")
    D = {k: [] for k in (POSK if MODE == "gpos" else ()) + ("fold", "seg_H", "j_H", "gps_H", "seg_L", "j_L", "gps_L", "gcls_H", "gcls_L", "gsnr_H", "gsnr_L",
                         "sigma_H", "sigma_L", "sigma_net", "g2_H", "g2_L", "cen2_H", "cen2_L", "admit")}
    A = {k: [] for k in ("idx", "t0", "a", "T_H", "s0_hm", "s0_lm")}; TL = []; shard = 0; tripped = False; t_start = time.time(); ndraw = 0
    def flush(final=False):
        nonlocal shard, TL, A
        if not TL or (len(TL) < SHARD and not final): return
        meta = {k: np.array(v) for k, v in D.items()}; ai = np.array(A["idx"]); m = {k: meta[k][ai] for k in meta}; n = len(ai)
        extra = dict(**{k: np.array(v) for k, v in A.items() if k != "idx"}, run=RUN, flo=FLO, fhi=FHI)
        if MODE == "gneg":
            s0h, s0l = np.array(A["s0_hm"]), np.array(A["s0_lm"])
            extra.update(source=np.array([f"gg_{h}|{l}" for h, l in zip(m["gcls_H"], m["gcls_L"])]), loglr=np.full(n, np.nan), cnn_hm=s0h, cnn_lm=s0l, s0_pass=(np.maximum(s0h, s0l) > 0.5).astype(int))
        np.savez(f"{OUT}/{'neg' if MODE == 'gneg' else 'pos'}_{RUN}_t{TASK}_s{shard:02d}.npz", tiles=np.stack(TL).astype(np.float16), **m, **extra)
        print(f"[{MODE}] shard {shard}: {len(TL)} admitted pairs written ({(time.time()-t_start)/60:.1f} min)", flush=True)
        shard += 1; TL = []; A = {k: [] for k in A}
    pending = []
    def process(batch):
        nonlocal tripped
        raw = {"H1": [], "L1": []}; gps_c = {"H1": [], "L1": []}
        for b in batch:
            for d in ("H1", "L1"):
                seg, j, g = b["car"][d]; r = B._win(seg, d, j).astype(np.float64)
                if MODE == "gpos":
                    c = WN // 2 + int(round(b["off"] * FS)); s = c - int(b["merger_index"]); x = b["w"][d].astype(np.float64) * b["scale"]
                    lo, hi = max(0, s), min(WN, s + len(x)); r[lo:hi] += x[lo - s:hi - s]
                gl = b["gl"][d]
                if gl is not None: r[gl["t0"]:gl["t0"] + len(gl["g"])] += gl["scale"] * gl["g"]
                raw[d].append(r.astype(np.float32)); gps_c[d].append(g)
        wh, qt, sg, c2, g2 = {}, {}, {}, {}, {}
        for d in ("H1", "L1"):
            wh[d] = pipe._whiten(np.stack(raw[d]), d); qt[d] = DS.build_qt(pipe, wh[d]); sg[d] = np.asarray(DS.sigma_from_qt(pipe, qt[d], d), float)
            c2[d] = np.asarray(C2.cluster_centroid(qt[d]), float); g2[d] = np.asarray(ARM2.g2_from_qt(qt[d], np.array(gps_c[d])), float)
        net = (sg["H1"] + sg["L1"]) / np.sqrt(2); lead = np.where(sg["H1"] >= sg["L1"], g2["H1"], g2["L1"])
        adm = (net >= NET_MIN) & C2.cen2_pass(c2["H1"], c2["L1"]) & (lead >= ARM_MIN); ia = np.where(adm)[0]
        base = len(D["fold"])
        for n_, b in enumerate(batch):
            kv = [("fold", b["fold"]), ("seg_H", b["car"]["H1"][0]), ("j_H", b["car"]["H1"][1]), ("gps_H", b["car"]["H1"][2]), ("seg_L", b["car"]["L1"][0]), ("j_L", b["car"]["L1"][1]), ("gps_L", b["car"]["L1"][2]),
                  ("gcls_H", b["gl"]["H1"]["cls"] if b["gl"]["H1"] else ""), ("gcls_L", b["gl"]["L1"]["cls"] if b["gl"]["L1"] else ""), ("gsnr_H", b["gl"]["H1"]["snr"] if b["gl"]["H1"] else 0.0), ("gsnr_L", b["gl"]["L1"]["snr"] if b["gl"]["L1"] else 0.0),
                  ("sigma_H", sg["H1"][n_]), ("sigma_L", sg["L1"][n_]), ("sigma_net", net[n_]), ("g2_H", g2["H1"][n_]), ("g2_L", g2["L1"][n_]), ("cen2_H", c2["H1"][n_]), ("cen2_L", c2["L1"][n_]), ("admit", int(adm[n_]))]
            if MODE == "gpos": kv += [("chunk", b["chunk"]), ("row", b["row"]), ("source_id", b["sid"]), ("mtot", b["mtot"]), ("q", b["q"]), ("dt_LH", b["dt"]), ("snr_target", b["snr"]), ("off", b["off"]), ("scale", b["scale"])]
            for k, v in kv: D[k].append(v)
        if not len(ia): return
        t0 = np.asarray(mr.cam_t0_batch(B.carm(), qt["H1"][ia], DEV), int); mH = mags(wh["H1"][ia]); mL = mags(wh["L1"][ia])
        for k, i in enumerate(ia):
            T = mH[k].shape[1]; a = int(round((t0[k] - 14) / 128 * T)); cH = crop_raw(mH[k], a, T); cL = crop_raw(mL[k], a, T)
            hm, lm = s0(cH, cL)
            if not tripped and k < 8:
                hp, lp = B.cnn_from_wh(wh["H1"][i][None, :], wh["L1"][i][None, :])
                if abs(hp - hm) > 1e-3 or abs(lp - lm) > 1e-3: raise SystemExit(f"[{MODE}] TRIPWIRE: stored-crop S0 ({hm:.5f},{lm:.5f}) != cnn_from_wh ({hp:.5f},{lp:.5f})")
                if k == 7: tripped = True; print(f"[{MODE}] tripwire OK: stored crops reproduce cnn_from_wh on 8 pairs", flush=True)
            TL.append(np.stack([cH, cL])); A["idx"].append(base + i); A["t0"].append(int(t0[k])); A["a"].append(a); A["T_H"].append(T); A["s0_hm"].append(hm); A["s0_lm"].append(lm)
    def glitch_legs():
        if MODE == "gneg":
            if rng.uniform() < PBOTH: return {"H1": draw_glitch("H1"), "L1": draw_glitch("L1")}
            d = "H1" if rng.uniform() < 0.5 else "L1"; return {d: draw_glitch(d), ("L1" if d == "H1" else "H1"): None}
        return {d: (draw_glitch(d) if rng.uniform() < GFRAC else None) for d in ("H1", "L1")}
    if MODE == "gneg":
        for _ in range(NDRAW):
            f = int(rng.integers(2)); carp = {d: car[(d, f)][int(rng.integers(len(car[(d, f)])))] for d in ("H1", "L1")}
            pending.append(dict(fold=f, car=carp, gl=glitch_legs()))
            if len(pending) == BATCH: process(pending); ndraw += len(pending); pending = []; flush(); print(f"[{MODE}] {ndraw}/{NDRAW} draws, admitted {int(np.sum(D['admit']))} ({(time.time()-t_start)/60:.1f} min)", flush=True)
    else:
        ch = sorted(glob.glob(f"{BANK}/signals_*.csv")); ch = [c for c in ch if int(c[-8:-4]) % 4 == RIDX][TASK::NTASK]
        if MAXCH: ch = ch[:MAXCH]
        print(f"[{MODE}] {len(ch)} bank chunks for this task", flush=True)
        for cf in ch:
            rows = pd.read_csv(cf); z = np.load(cf.replace(".csv", ".npz")); cid = int(cf[-8:-4])
            for r_i in range(len(rows)):
                r = rows.iloc[r_i]; w = {"H1": z["H1"][r_i], "L1": z["L1"][r_i]}
                s0H = ip.compute_optimal_snr(w["H1"], pipe.asd["H1"]); s0L = ip.compute_optimal_snr(w["L1"], pipe.asd["L1"]); s0n = float(np.hypot(s0H, s0L))
                if not np.isfinite(s0n) or s0n <= 0: continue
                snr = float(np.exp(rng.uniform(np.log(SNR_LO), np.log(SNR_HI)))); f = int(rng.integers(2))
                carp = {d: car[(d, f)][int(rng.integers(len(car[(d, f)])))] for d in ("H1", "L1")}
                pending.append(dict(chunk=cid, row=r_i, sid=int(r.source_id), mtot=float(r.total_mass), q=float(r.mass_ratio), dt=float(r.dt_L1_minus_H1_s), merger_index=int(r.merger_index),
                                    snr=snr, scale=snr / s0n, off=float(rng.uniform(-OFF, OFF)), fold=f, car=carp, w=w, gl=glitch_legs()))
                if len(pending) == BATCH: process(pending); ndraw += len(pending); pending = []; flush()
            print(f"[{MODE}] chunk {cid} done: {ndraw} draws, admitted {int(np.sum(D['admit']))} ({(time.time()-t_start)/60:.1f} min)", flush=True)
    if pending: process(pending); ndraw += len(pending)
    flush(final=True)
    np.savez(f"{OUT}/draws_{RUN}_t{TASK}.npz", **{k: np.array(v) for k, v in D.items()})
    ad = np.array(D["admit"]); print(f"[{MODE}] DONE {ndraw} draws, {ad.sum()} admitted ({ad.mean():.3f}); S0 pass among admitted: {np.mean(np.maximum(A['s0_hm'], A['s0_lm']) > 0.5) if A['s0_hm'] else float('nan'):.3f} (last shard only)", flush=True)

if __name__ == "__main__":
    main()
