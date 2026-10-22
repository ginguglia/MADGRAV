"""Pack the rendered specialist training shards (madgrav2_spec/{pos,neg}/*.npz) into per-fold memmaps for the trainer.
Output: $SPEC_DATA/{pos,neg}_f{0,1}.npy  (N, 2, 961, 113) float16 = RAW 20-500 Hz crops (H1, L1), no normalisation
        $SPEC_DATA/{pos,neg}_f{0,1}_meta.npz  (per-pair meta incl. calib day flag, S0 scores, run, source)
Fold = the shard's 'fold' field (day fold of the pair, fold-pure by construction); calib = calib_of(UTC day of gps_H).
Two passes: meta only (tiles member not touched), then a process pool copies each shard into its precomputed rows."""
import os, sys, glob, time, numpy as np
from multiprocessing import Pool
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SRC = MADGRAV_SCRATCH + "/madgrav2_spec"; OUT = os.environ.get("SPEC_DATA", f"{SRC}/train"); os.makedirs(OUT, exist_ok=True)
KEEP = {"pos": ("run", "fold", "gps_H", "gps_L", "mtot", "q", "snr_target", "sigma_H", "sigma_L", "sigma_net", "dt_LH", "source_id", "s0_hm", "s0_lm"),
        "neg": ("run", "fold", "gps_H", "gps_L", "source", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "s0_hm", "s0_lm", "s0_pass")}

def copy(job):
    fn, cls, f, rows, dst = job
    z = np.load(fn); sel = np.where(np.asarray(z["fold"]) == f)[0]; X = open_memmap(dst, mode="r+")
    t = z["tiles"]; X[rows[0]:rows[0] + len(sel)] = t[sel]; X.flush(); del X
    return fn, f, len(sel)

if __name__ == "__main__":
    t0 = time.time(); ALLJOBS = {}
    for cls in ("pos", "neg"):
        srcdir = os.environ.get(f"SPEC_SRC_{cls.upper()}", f"{SRC}/{cls}")   # 2026-09-30: SPEC_SRC_POS/NEG = alternative shard dirs (extra sets for SPEC_EXTRA in train_spec.py)
        files = sorted(glob.glob(f"{srcdir}/{cls}_*.npz")); files = files[::max(1, len(files) // int(os.environ["SPEC_MAXFILES"]))] if os.environ.get("SPEC_MAXFILES") else files; meta = {0: {k: [] for k in KEEP[cls]}, 1: {k: [] for k in KEEP[cls]}}; jobs = []; n = {0: 0, 1: 0}; shape = None
        for fn in files:
            z = np.load(fn); fold = np.asarray(z["fold"]); N = len(fold)
            if shape is None: shape = z["tiles"].shape[1:]   # header read only (zip member stays lazy after this)
            for f in (0, 1):
                sel = np.where(fold == f)[0]
                if not len(sel): continue
                jobs.append((fn, cls, f, (n[f],), None)); n[f] += len(sel)
                for k in KEEP[cls]:
                    v = z[k]; v = np.full(N, v.item()) if v.ndim == 0 else np.asarray(v)
                    meta[f][k].append(v[sel])
        if not files: print(f"[assemble] {cls}: no shards in {srcdir}, skipped", flush=True); continue
        print(f"[assemble] {cls}: {len(files)} shards, fold0 {n[0]}, fold1 {n[1]}, tile {shape}", flush=True)
        for f in (0, 1):
            dst = f"{OUT}/{cls}_f{f}.npy"; open_memmap(dst, mode="w+", dtype=np.float16, shape=(n[f], *shape)).flush()
            m = {k: np.concatenate(v) for k, v in meta[f].items()}; day = (m["gps_H"] // 86400).astype(int)
            assert np.all(np.array([QU.fold_of(d) for d in np.unique(day)])[np.searchsorted(np.unique(day), day)] == f), "fold field != fold_of(gps_H day)"
            m["calib"] = np.array([QU.calib_of(d) for d in day]); m["day"] = day
            np.savez(f"{OUT}/{cls}_f{f}_meta.npz", **m)
            print(f"[assemble] {cls} f{f}: n={n[f]} calib {int(m['calib'].sum())} | " + " ".join(f"{r}:{int((m['run']==r).sum())}" for r in np.unique(m["run"]))
                  + ("" if cls == "pos" else " | " + " ".join(f"{s}:{int((m['source']==s).sum())}" for s in np.unique(m["source"]))), flush=True)
        jobs = [(fn, c, f, r, f"{OUT}/{c}_f{f}.npy") for fn, c, f, r, _ in jobs]; ALLJOBS[cls] = jobs
        with Pool(int(os.environ.get("SPEC_WORKERS", "16"))) as p:
            for i, (fn, f, k) in enumerate(p.imap_unordered(copy, jobs)):
                if i % 20 == 0: print(f"[assemble] {cls} {i+1}/{len(jobs)} copied ({(time.time()-t0)/60:.1f} min)", flush=True)
    # spot check: first/last row of every shard (per fold) vs the source, in copy order (legs recur across negatives -> no gps matching)
    for cls in ("pos", "neg"):
        for f in (0, 1):
            if not os.path.exists(f"{OUT}/{cls}_f{f}.npy"): continue
            X = np.load(f"{OUT}/{cls}_f{f}.npy", mmap_mode="r"); m = np.load(f"{OUT}/{cls}_f{f}_meta.npz"); bad = 0; nchk = 0
            for fn, _, ff, r, _ in ALLJOBS[cls]:
                if ff != f: continue
                z = np.load(fn); sel = np.where(z["fold"] == f)[0]; o = r[0]
                for j in (0, len(sel) - 1):
                    nchk += 1; bad += not (np.array_equal(X[o + j], z["tiles"][sel[j]]) and m["gps_H"][o + j] == z["gps_H"][sel[j]] and m["gps_L"][o + j] == z["gps_L"][sel[j]])
            zf = np.mean([float(np.abs(X[i]).max()) == 0.0 for i in np.random.default_rng(0).choice(len(X), 200, replace=False)])
            print(f"[assemble] check {cls} f{f}: {nchk - bad}/{nchk} shard-edge rows identical; zero-tile fraction (200 random) = {zf:.3f}", flush=True)
            assert bad == 0 and zf == 0.0, "assembly check FAILED"
    print(f"[assemble] DONE {(time.time()-t0)/60:.1f} min", flush=True)
