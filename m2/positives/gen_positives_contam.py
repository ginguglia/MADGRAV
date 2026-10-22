"""MADGRAV2 CONTAMINATED positive population (2026-09-29, arm sim-to-real line): gen_positives_m2.py with the carrier population
switchable. The quiet-carrier positives (|sigma| in [0.5,1.5)) teach both networks "signal = chirp on quiet noise"; real events sit in
ordinary and sometimes glitchy noise (12/22 arm-vetoed catalogue windows also fail S3; o4a/o4b positives 1.5-3.5 logit units easier than
real events for _off5). Carrier modes (M2_CARRIER): quiet (as before) | murky (CAE-passing time-slid background windows of the run, the
FAR-setting population, g8/murky shards) | glitch (mined real glitch windows of the run, retrain_v2/mined*) | mix (M2_MIX=q,m,g fractions,
default 0.4,0.4,0.2), drawn independently per leg. Optional M2_SYNTH=gengli adds a GAN blip (gengli, Lopez et al. 2022) to a fraction
M2_SYNTH_FRAC (0.3) of legs at a random offset within the tile, amplitude log-uniform in
M2_SYNTH_SNR_LO/HI (8,60), lalsuite white-noise SNR against the whitened series' robust sigma. Records per leg: car_H/car_L (mode), csig_H/csig_L (carrier CAE sigma before injection),
synth_H/synth_L (0/1). Everything else (banks, SNR, Mtot, placement, tile builder, output format) identical to gen_positives_m2.py.
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
CARRIER = os.environ.get("M2_CARRIER", "quiet"); MIX = [float(x) for x in os.environ.get("M2_MIX", "0.4,0.4,0.2").split(",")]
SYNTH = os.environ.get("M2_SYNTH", ""); SYNTH_FRAC = float(os.environ.get("M2_SYNTH_FRAC", "0.3"))   # gengli (whitened-domain blips) | glitchgen (raw-strain GlitchGen bank, HDF5 per class)
GG_BANK = os.environ.get("M2_GG_BANK", MADGRAV_SCRATCH + "/madgrav2/arm/synth/glitchgen_v1"); GG_CLASSES = [c for c in os.environ.get("M2_GG_CLASSES", "").split(",") if c]
GG_SNR = (float(os.environ.get("M2_GG_SNR_LO", "8")), float(os.environ.get("M2_GG_SNR_HI", "80")))   # optimal SNR of the glitch vs the run's reference PSD
NO_SIGNAL = os.environ.get("M2_NO_SIGNAL", "0") == "1"   # glitch-only tiles (glitch-class augmentation): no chirp, every leg gets a glitch
SYNTH_SNR = (float(os.environ.get("M2_SYNTH_SNR_LO", "8")), float(os.environ.get("M2_SYNTH_SNR_HI", "60")))   # lalsuite white-noise SNR of the blip
M1 = MADGRAV_SCRATCH + "/MADGRAV1"

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

def carriers_quiet(det, rng, n=3000, per_seg=4):
    metas = sorted(glob.glob(f"{os.environ['SM_STREAMS']}/*_{det}_meta.npz")); out = []
    for k in rng.permutation(len(metas)):
        m = metas[int(k)]; seg = os.path.basename(m).replace(f"_{det}_meta.npz", ""); z = np.load(m)
        ok = np.where((np.abs(z["sigma"]) >= 0.5) & (np.abs(z["sigma"]) < 1.5))[0]
        if not len(ok): continue
        for j in rng.choice(ok, size=min(per_seg, len(ok)), replace=False): out.append((seg, int(j), float(z["gps"][j]), float(z["sigma"][j])))
        if len(out) >= n: break
    print(f"[carriers {det} quiet] {len(out)} windows from {len({s for s,_,_,_ in out})} segments of {len(metas)}", flush=True)
    return out

def _gps_of(seg, det, j):
    z = np.load(f"{os.environ['SM_STREAMS']}/{seg}_{det}_meta.npz"); return float(z["gps"][j])

def carriers_murky(det, rng, n=3000):
    out = []
    for f in sorted(glob.glob(f"{M1}/dev_v2/g8/murky/murky_{RUN}_{det}_shard*.npz")):
        z = np.load(f)
        for seg, j, sg in zip(z["seg"], z["j"], z["sigma"]): out.append((str(seg), int(j), float(sg)))
    if not out: raise SystemExit(f"no murky shards for {RUN} {det}")
    k = rng.permutation(len(out))[:n]; res = []
    for i in k:
        seg, j, sg = out[i]
        try: res.append((seg, j, _gps_of(seg, det, j), sg))
        except Exception: continue
    print(f"[carriers {det} murky] {len(res)} of {len(out)} CAE-passing background windows (sigma {min(r[3] for r in res):.1f}-{max(r[3] for r in res):.1f})", flush=True)
    return res

def carriers_glitch(det, rng, n=3000):
    out = []
    for d in ("mined", "mined_o3b", "mined_o4a", "mined_o4b"):
        for f in sorted(glob.glob(f"{M1}/dev_v2/glitch_arm/retrain_v2/{d}/*shard*.npz")):
            z = np.load(f, allow_pickle=True)
            if str(z["det"]) != det: continue
            for seg, j, g, sg in zip(z["seg"], z["j"], z["gps"], z["sigma"]):
                if str(seg).startswith(RUN + "_"): out.append((str(seg), int(j), float(g), float(sg)))
    if not out: raise SystemExit(f"no mined glitch windows for {RUN} {det}")
    k = rng.permutation(len(out))[:n]; res = [out[i] for i in k]
    print(f"[carriers {det} glitch] {len(res)} of {len(out)} mined glitch windows (sigma {min(r[3] for r in res):.1f}-{max(r[3] for r in res):.1f})", flush=True)
    return res

def carriers_of(det, rng):
    """dict mode -> list of (seg, j, gps, carrier_sigma); modes present per M2_CARRIER."""
    if CARRIER == "quiet": return {"quiet": carriers_quiet(det, rng)}
    if CARRIER == "murky": return {"murky": carriers_murky(det, rng)}
    if CARRIER == "glitch": return {"glitch": carriers_glitch(det, rng)}
    if CARRIER == "mix": return {"quiet": carriers_quiet(det, rng), "murky": carriers_murky(det, rng), "glitch": carriers_glitch(det, rng)}
    raise SystemExit(f"M2_CARRIER={CARRIER}?")

_gg = None
def gg_load():
    """GlitchGen bank: {class: strain array (n, 8192)} from <bank>/<cls>/<cls>_chunk*.hdf5 and <bank>/param/<cls>/..., raw strain 2 s @ FS."""
    global _gg
    if _gg is not None: return _gg
    import h5py; _gg = {}
    for f in sorted(glob.glob(f"{GG_BANK}/*/*_chunk*.hdf5") + glob.glob(f"{GG_BANK}/param/*/*_chunk*.hdf5")):
        cls = os.path.basename(os.path.dirname(f))
        if GG_CLASSES and cls not in GG_CLASSES: continue
        with h5py.File(f) as h:
            assert int(h.attrs.get("sample_rate_hz", FS)) == FS, f; x = np.asarray(h["strain"], np.float32)
        _gg.setdefault(cls, []).append(x)
    _gg = {c: np.concatenate(v) for c, v in _gg.items()}
    if not _gg: raise SystemExit(f"no GlitchGen chunks under {GG_BANK}")
    print(f"[glitchgen] {len(_gg)} classes, " + ", ".join(f"{c}:{len(v)}" for c, v in _gg.items()), flush=True); return _gg

def gg_draw(rng):
    """(class, raw strain 1-D) drawn uniformly over classes then uniformly within the class."""
    bank = gg_load(); cls = list(bank)[int(rng.integers(len(bank)))]; return cls, bank[cls][int(rng.integers(len(bank[cls])))].astype(np.float64)

def draw_carrier(cars, rng):
    modes = list(cars); p = np.array(MIX[:len(modes)] if CARRIER == "mix" else [1.0]); p = p / p.sum()
    mode = modes[int(rng.choice(len(modes), p=p))]; seg, j, gps, sg = cars[mode][int(rng.integers(len(cars[mode])))]
    return mode, seg, j, gps, sg

_synth = None
def synth_glitch(det, rng):
    """One unit-peak GAN blip (gengli bank, whitened-domain excess power) and its sum of squares; None when off."""
    global _synth
    if SYNTH != "gengli": return None
    if _synth is None:
        _synth = {}
        for d in ("H1", "L1"):
            z = np.load(f"{os.environ.get('M2_SYNTH_BANK', MADGRAV_SCRATCH + '/madgrav2/arm/synth')}/gengli_blips_{d}.npz"); assert float(z["srate"]) == FS
            _synth[d] = (np.asarray(z["x"], np.float64), np.asarray(z["sumsq"], float)); print(f"[synth {d}] {len(_synth[d][0])} gengli blips of {_synth[d][0].shape[1]} samples", flush=True)
    i = int(rng.integers(len(_synth[det][0]))); return _synth[det][0][i], float(_synth[det][1][i])

def main():
    rng = np.random.default_rng(SEED)
    pipe = B.cpipe()
    prov = dict(carrier=CARRIER, mix=",".join(map(str, MIX)), synth=SYNTH, synth_frac=SYNTH_FRAC, no_signal=int(NO_SIGNAL), gg_bank=GG_BANK, gg_snr=f"{GG_SNR[0]}-{GG_SNR[1]}", synth_snr=f"{SYNTH_SNR[0]}-{SYNTH_SNR[1]}", cae_sd_hash=_sha(f"{ROOT}/assets/models/baseline_cae_weaksup_best.pt"), cae_cal_hash=_sha(f"{ROOT}/spectrogram_cascade/massive_calibration_BA.json"),
                floor=FLOOR, tile_geom="256x128_zoom_minmax", harness_sha=_sha(os.path.abspath(__file__)), seed=SEED, run=RUN, snr_lo=SNR_LO, snr_hi=SNR_HI, mt_lo=MT_LO, mt_hi=MT_HI)
    print(f"[m2pos] CONTAM carrier={CARRIER} mix={MIX} synth={SYNTH!r} frac={SYNTH_FRAC} no_signal={NO_SIGNAL} gg_bank={GG_BANK} gg_snr={GG_SNR}"); print(f"[m2pos] run {RUN} N={N} SNR log-uniform [{SNR_LO},{SNR_HI}] Mtot flat-in-log [{MT_LO},{MT_HI}] seed {SEED}; CAE sd {prov['cae_sd_hash'][:12]} cal {prov['cae_cal_hash'][:12]}; det_flags {_F}", flush=True)
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
              "seg_L", "j_L", "gps_L", "day_L", "car_H", "car_L", "csig_H", "csig_L", "synth_H", "synth_L", "synth_snr_H", "synth_snr_L", "gcls_H", "gcls_L", "sigma_H", "sigma_L", "sigma_net", "pass_H", "pass_L", "pass_net")
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
                mode, seg, j, gps, csg = draw_carrier(cars[d], rng)
                r = B._win(seg, d, j).astype(np.float64); x = banks[bid][k][i].astype(np.float64); Lc = min(len(x), WN - a0)
                inj = r.copy()
                if not NO_SIGNAL: inj[a0:a0 + Lc] += sc * x[:Lc]
                gcls, gsnr = "", 0.0
                if SYNTH == "glitchgen" and (NO_SIGNAL or rng.uniform() < SYNTH_FRAC):
                    gcls, g = gg_draw(rng); s_g = ip.compute_optimal_snr(g, pipe.asd[d]); gsnr = float(np.exp(rng.uniform(np.log(GG_SNR[0]), np.log(GG_SNR[1]))))
                    t0 = int(rng.integers(0, WN - len(g) + 1)); inj[t0:t0 + len(g)] += (gsnr / max(s_g, 1e-30)) * g   # glitch centre anywhere in [1 s, 3 s] of the 4 s window
                raw[d].append(inj.astype(np.float32)); cw[d] = (seg, j, gps, mode, csg, gcls, gsnr)
            meta.append((m, m1, m2, bid, i, snr, sH * sc, sL * sc, sc, off, cw))
        sig = {}; qt = {}; synth = {"H1": np.zeros(nb, int), "L1": np.zeros(nb, int)}; ssnr = {"H1": np.zeros(nb), "L1": np.zeros(nb)}
        for d in ("H1", "L1"):
            wh = pipe._whiten(np.stack(raw[d]), d)
            if SYNTH == "gengli":   # add GAN blips in the whitened domain, peak amplitude in units of the whitened series' robust sigma
                for n_ in range(nb):
                    if rng.uniform() >= SYNTH_FRAC: continue
                    g, ss = synth_glitch(d, rng); rs = 1.4826 * np.median(np.abs(wh[n_] - np.median(wh[n_]))) + 1e-12
                    snr_g = float(np.exp(rng.uniform(np.log(SYNTH_SNR[0]), np.log(SYNTH_SNR[1])))); amp = rs * snr_g * np.sqrt(FS / 2.0 / ss)
                    t0 = int(rng.integers(FS // 2, WN - len(g) - FS // 2)); wh[n_, t0:t0 + len(g)] += amp * g; synth[d][n_] = 1; ssnr[d][n_] = snr_g
            qt[d] = DS.build_qt(pipe, wh); sig[d] = np.asarray(DS.sigma_from_qt(pipe, qt[d], d), float)
        for n_, (m, m1, m2, bid, i, snr, soH, soL, sc, off, cw) in enumerate(meta):
            sh, sl = float(sig["H1"][n_]), float(sig["L1"][n_]); sn = (sh + sl) / np.sqrt(2)
            vals = dict(mtot=m, m1=m1, m2=m2, bank=bid, row=i, snr_target=snr, snr_opt_H=soH, snr_opt_L=soL, scale=sc, off_s=off,
                        seg_H=cw["H1"][0], j_H=cw["H1"][1], gps_H=cw["H1"][2], day_H=int(cw["H1"][2] // 86400), seg_L=cw["L1"][0], j_L=cw["L1"][1], gps_L=cw["L1"][2], day_L=int(cw["L1"][2] // 86400),
                        car_H=cw["H1"][3], car_L=cw["L1"][3], csig_H=cw["H1"][4], csig_L=cw["L1"][4],
                        synth_H=int(synth["H1"][n_] or bool(cw["H1"][5])), synth_L=int(synth["L1"][n_] or bool(cw["L1"][5])), synth_snr_H=float(ssnr["H1"][n_] or cw["H1"][6]), synth_snr_L=float(ssnr["L1"][n_] or cw["L1"][6]),
                        gcls_H=cw["H1"][5], gcls_L=cw["L1"][5],
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
