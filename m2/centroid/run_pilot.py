"""Compute every centroid definition on the cached tile populations and write one npz per population.
Populations (all frozen-geometry 256x128 tiles, read-only from MADGRAV1 caches):
  pos      injections into real quiet carriers (retrain_v2/trainset0 *pos*), with mtot, sigma, det
  det47    the 47 adopted detections, H1 and L1 (madgrav_qt_build/tiles.npz)
  glitch   mined real single-detector glitch tiles with relabel-v2 class (retrain_v2/mined + relabel/mined_labels.csv)
  top50    the top-50 FAR-setting background families, both legs (madgrav_qt_build/bg_top50/bg_tiles.npz)
  murky    CAE-passing time-slid background tiles, all runs/dets (dev_v2/g8/murky)
Usage: python run_pilot.py [pos det47 glitch top50 murky] [--smoke]
"""
import sys, glob, csv, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centroid_defs import compute_all

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M1 = MADGRAV_SCRATCH + "/MADGRAV1"
QB = "$MADGRAV_SCRATCH/madgrav_qt_build"
OUT = MADGRAV_SCRATCH + "/madgrav2/centroid"
SMOKE = "--smoke" in sys.argv
os.makedirs(OUT, exist_ok=True)

def save(pop, feats, **meta):
    np.savez(f"{OUT}/feat_{pop}{'_smoke' if SMOKE else ''}.npz", **feats, **meta)
    print(f"[{pop}] {len(next(iter(feats.values())))} tiles -> feat_{pop}.npz", flush=True)

def run_pos():
    fs = sorted(glob.glob(f"{M1}/dev_v2/glitch_arm/retrain_v2/trainset0/*_pos_*.npz"))
    if SMOKE: fs = fs[:1]
    F, meta = [], {k: [] for k in ("mtot", "sigma", "det", "bank", "day")}
    for f in fs:
        z = np.load(f); F.append(compute_all(z["qt"]))
        det = os.path.basename(f).split("_")[0]
        for k in meta:
            meta[k].append(z[k] if k != "det" else np.full(len(z["qt"]), det))
    save("pos", {k: np.concatenate([d[k] for d in F]) for k in F[0]}, **{k: np.concatenate(v) for k, v in meta.items()})

def run_posg8():
    """g8 shakedown positives: CAE-survivor injections at lower sigma (4.0-8.4, median 6.0), mtot 32-790, fp16 tiles."""
    F, meta = [], {k: [] for k in ("mtot", "sigma", "det", "bank")}
    for d in ("H1", "L1"):
        z = np.load(f"{M1}/dev_v2/g8/shakedown/pos_{d}.npz", allow_pickle=True); q = np.asarray(z["qt"], dtype=np.float32)
        F.append(compute_all(q)); n = len(q)
        meta["mtot"].append(z["mtot"]); meta["sigma"].append(z["sigma"]); meta["det"].append(np.full(n, d)); meta["bank"].append(z["bank"] if "bank" in z.files else np.full(n, "?"))
    save("posg8", {k: np.concatenate([x[k] for x in F]) for k in F[0]}, **{k: np.concatenate(v) for k, v in meta.items()})

def run_det47():
    z = np.load(f"{QB}/tiles.npz"); names = sorted(z.files)
    tiles = np.stack([z[n] for n in names])
    save("det47", compute_all(tiles), name=np.array(names), det=np.array([n[-2:] for n in names]),
         event=np.array([n[:-3] for n in names]))

def run_glitch():
    lab = {}
    for r in csv.DictReader(open(f"{M1}/dev_v2/glitch_arm/retrain_v2/relabel/mined_labels.csv")):
        lab[(r["det"], r["seg"], int(r["j"]))] = r["label"]
    fs = sorted(glob.glob(f"{M1}/dev_v2/glitch_arm/retrain_v2/mined/*_shard*.npz"))
    if SMOKE: fs = fs[:1]
    F, meta = [], {k: [] for k in ("det", "label", "sigma", "centroid_old", "g_old", "stratum", "day")}
    for f in fs:
        z = np.load(f); F.append(compute_all(z["qt"])); det = str(z["det"])
        n = len(z["qt"]); meta["det"].append(np.full(n, det))
        meta["label"].append(np.array([lab.get((det, s, int(j)), "NA") for s, j in zip(z["seg"], z["j"])]))
        meta["sigma"].append(z["sigma"]); meta["centroid_old"].append(z["centroid"]); meta["g_old"].append(z["g_old"])
        meta["stratum"].append(z["stratum"]); meta["day"].append(z["day"])
    save("glitch", {k: np.concatenate([d[k] for d in F]) for k in F[0]}, **{k: np.concatenate(v) for k, v in meta.items()})

def run_top50():
    z = np.load(f"{QB}/bg_top50/bg_tiles.npz"); names = sorted(z.files)
    tiles = np.stack([z[n] for n in names])
    import json
    fam = json.load(open(f"{QB}/bg_top50/bg_top50.json"))
    runs = {}
    if isinstance(fam, list):
        for i, e in enumerate(fam, 1): runs[f"bg{i:02d}"] = e.get("run", "?")
    save("top50", compute_all(tiles), name=np.array(names), det=np.array([n[-2:] for n in names]),
         fam=np.array([n[:-3] for n in names]), run=np.array([runs.get(n[:-3], "?") for n in names]))

def run_murky():
    fs = sorted(glob.glob(f"{M1}/dev_v2/g8/murky/murky_*_shard*.npz"))
    if SMOKE: fs = fs[:1]
    F, meta = [], {k: [] for k in ("run", "det", "sigma", "split", "seg")}
    t0 = time.time()
    for i, f in enumerate(fs, 1):
        z = np.load(f, mmap_mode="r"); q = z["qt"]
        F.append(compute_all(q)); b = os.path.basename(f).split("_")
        n = len(q); meta["run"].append(np.full(n, b[1])); meta["det"].append(np.full(n, b[2]))
        meta["sigma"].append(np.asarray(z["sigma"])); meta["split"].append(np.asarray(z["split"])); meta["seg"].append(np.asarray(z["seg"]))
        print(f"  murky shard {i}/{len(fs)} {os.path.basename(f)} n={n} {(time.time()-t0)/60:.1f} min", flush=True)
    save("murky", {k: np.concatenate([d[k] for d in F]) for k in F[0]}, **{k: np.concatenate(v) for k, v in meta.items()})

if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["pos", "det47", "glitch", "top50", "murky"]
    for w in which:
        t0 = time.time(); globals()[f"run_{w}"](); print(f"[{w}] done in {(time.time()-t0)/60:.1f} min", flush=True)
