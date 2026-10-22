"""Render both legs of every FAR-setting background family in bg_top50/bg_families_all.csv (532 families) as the frozen
model-input tile, compute centroid definitions, save feat_fam532.npz. Read-only on MADGRAV1 strain caches."""
import os, sys, csv, time, numpy as np
sys.path.insert(0, "$MADGRAV_SCRATCH/madgrav_qt_build"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiles import tile, WIN, FS, WN
from centroid_defs import compute_all
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH; OUT = f"{SC}/madgrav2/centroid"
def strain_path(run, seg, det):
    for d in (f"{SC}/strain_{run}_full", f"{SC}/strain_{run}_56"):
        p = f"{d}/{seg}_{det}.npz"
        if os.path.exists(p): return p
def window_at(path, gps):
    z = np.load(path, allow_pickle=True); gs = float(z["gps_start"]); assert int(z["fs"]) == FS
    i0 = int(round((gps - WIN / 2 - gs) * FS)); w = np.asarray(z["strain"][i0:i0 + WN], dtype=np.float32); assert len(w) == WN; return w
rows = list(csv.DictReader(open("$MADGRAV_SCRATCH/madgrav_qt_build/bg_top50/bg_families_all.csv")))
if "--smoke" in sys.argv: rows = rows[:3]
tiles, meta = [], {k: [] for k in ("fam", "run", "det", "lnL", "sigma", "cen_old", "outranks", "rank_overall", "seg", "gps")}
t0 = time.time(); fail = 0
for i, r in enumerate(rows, 1):
    run = r["run"].lower()
    for det, seg, gps, sig, cen in (("H1", r["H1_seg"], float(r["gps_H"]), r["sigma_H"], r["cen_H_Hz"]), ("L1", r["L1_seg"], float(r["gps_L"]), r["sigma_L"], r["cen_L_Hz"])):
        p = strain_path(run, seg, det)
        try:
            if p is None: raise FileNotFoundError(seg)
            t = tile(run, det, window_at(p, gps))
        except Exception as e:
            fail += 1; print(f"FAIL {r['fam']} {det}: {e}", flush=True); continue
        tiles.append(t)
        for k, v in dict(fam=r["fam"], run=r["run"], det=det, lnL=float(r["lnLambda"]), sigma=float(sig), cen_old=float(cen), outranks=r["outranks"],
                         rank_overall=int(r["rank_overall"]), seg=seg, gps=gps).items(): meta[k].append(v)
    if i % 25 == 0: print(f"[{i}/{len(rows)}] {(time.time()-t0)/60:.1f} min, fails {fail}", flush=True)
tiles = np.stack(tiles); f = compute_all(tiles); sfx = "_smoke" if "--smoke" in sys.argv else ""
np.savez(f"{OUT}/feat_fam532{sfx}.npz", **f, **{k: np.array(v) for k, v in meta.items()})
np.savez(f"{OUT}/fam532_tiles{sfx}.npz", qt=tiles.astype(np.float16), fam=np.array(meta["fam"]), det=np.array(meta["det"]))
print(f"saved {len(tiles)} legs, {fail} failures, {(time.time()-t0)/60:.1f} min", flush=True)
