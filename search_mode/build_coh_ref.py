"""Build the per-(segment, detector) coherence reference spectra for coh_rewhiten from a series cache.
Usage: SM_SERIESCACHE=<cache dir> [SM_COHREF_DIR=<out>] [SM_COHREF_ROWS=512] python build_coh_ref.py [--procs 6]
Idempotent: existing files are skipped. Output: SM_COHREF_DIR/<name>_<det>_cohref.npy (+ a manifest json)."""
import os, sys, glob, json, time, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, coh_rewhiten as CR
def one(npy):
    b = os.path.basename(npy)[: -len("_series.npy")]; name, det = b.rsplit("_", 1)
    if os.path.exists(CR.ref_path(name, det)): return (name, det, "skip")
    CR.build_ref_from_cache(name, det, npy); return (name, det, "built")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); a = ap.parse_args()
    cache = os.environ["SM_SERIESCACHE"]; files = sorted(glob.glob(f"{cache}/*_series.npy")); t0 = time.time()
    print(f"[cohref] {len(files)} series files from {cache} -> {CR.REF_DIR} (rows/ref {CR.NREF}, band {CR.BAND})", flush=True)
    os.makedirs(CR.REF_DIR, exist_ok=True); n = {"built": 0, "skip": 0}
    with Pool(a.procs) as p:
        for i, (name, det, st) in enumerate(p.imap_unordered(one, files, chunksize=4)):
            n[st] += 1
            if i % 200 == 0: print(f"  {i+1}/{len(files)} ({(time.time()-t0)/60:.1f} min) built {n['built']} skip {n['skip']}", flush=True)
    json.dump(dict(cache=cache, n_files=len(files), rows_per_ref=CR.NREF, band=CR.BAND, built=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{CR.REF_DIR}/_manifest.json", "w"), indent=1)
    print(f"[cohref] done: built {n['built']} skipped {n['skip']} in {(time.time()-t0)/60:.1f} min", flush=True)
