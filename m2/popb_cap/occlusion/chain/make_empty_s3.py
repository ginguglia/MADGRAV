"""Empty S3 'production bank' per run (same schema, zero rows) so score_bank_s3_nf.py re-scores EVERY bank row under the new crop."""
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for run in ("o3a","o3b","o4a","o4b"):
    z=np.load(f"{MADGRAV_SCRATCH}/madgrav2/s3_off5/s3bank/s3_{run}.npz",allow_pickle=True)
    out={k:(z[k] if z[k].shape==() else z[k][:0]) for k in z.files}; np.savez(f"{MADGRAV_SCRATCH}/madgrav2/popb_occl/s3_empty_{run}.npz",**out)
    print(run,{k:v.shape for k,v in out.items() if k!="run"}["s3"])
