"""Concatenate the per-task S3 score files of one run into madgrav2_spec/s3/s3_<run>.npz (the SM_S3_HOOK file)."""
import sys, glob, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

run = sys.argv[1]; D = MADGRAV_SCRATCH + "/madgrav2_spec/s3"; fs = sorted(glob.glob(f"{D}/s3_{run}_t*.npz")); assert len(fs) == 16, f"{len(fs)} task files"
Z = [np.load(f, allow_pickle=True) for f in fs]; keys = [k for k in Z[0].files if k != "run"]
out = {k: np.concatenate([z[k] for z in Z]) for k in keys}; assert np.isfinite(out["s3"]).all(), "NaN S3 scores"
np.savez(f"{D}/s3_{run}.npz", run=run, **out); print(f"[concat] {run}: {len(out['s3'])} rows ({int((out['kind']=='trig').sum())} triggers), s3 median {np.median(out['s3']):.3f}")
