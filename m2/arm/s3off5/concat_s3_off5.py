"""_off5 copy of m2/rank/concat_s3.py (2026-10-02): concatenate the per-task S3 score files in R_OUT (madgrav2/s3_off5/s3bank) into
R_OUT/s3_<run>.npz (the SM_S3_HOOK file of the _off5 _s3rank merge). Never touches madgrav2_spec/s3."""
import os, sys, glob, numpy as np
run = sys.argv[1]; D = os.environ["R_OUT"]; assert "madgrav2_spec" not in D, D
fs = sorted(glob.glob(f"{D}/s3_{run}_t*.npz")); assert len(fs) == 16, f"{len(fs)} task files"
Z = [np.load(f, allow_pickle=True) for f in fs]; keys = [k for k in Z[0].files if k != "run"]
out = {k: np.concatenate([z[k] for z in Z]) for k in keys}; assert np.isfinite(out["s3"]).all(), "NaN S3 scores"
np.savez(f"{D}/s3_{run}.npz", run=run, **out); print(f"[concat] {run}: {len(out['s3'])} rows ({int((out['kind']=='trig').sum())} triggers), s3 median {np.median(out['s3']):.3f}")
