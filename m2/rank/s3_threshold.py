"""A6: S3 operating threshold = score keeping 92% of cross-fitted held-out injection pairs (both folds pooled, 5-seed ensemble mean),
from the trainer's held-out score files (spec_hm_f{F}_s{S}_nfull_heldout.npz holds fold 1-F positives). Also reports held-out
background pass (as drawn) at that threshold. Writes m2/rank/s3_threshold.json."""
import json, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MD = MADGRAV_SCRATCH + "/madgrav2_spec/models"; P = []; N = []
for F in (0, 1):
    Z = [np.load(f"{MD}/spec_hm_f{F}_s{s}_nfull_heldout.npz") for s in range(5)]
    P.append(np.mean([z["p_pos"] for z in Z], 0)); N.append(np.mean([z["p_neg"] for z in Z], 0))
P = np.concatenate(P); N = np.concatenate(N); thr = float(np.percentile(P, 8.0))
out = dict(threshold=thr, target_eff=0.92, eff_at_thr=float((P >= thr).mean()), bg_pass_as_drawn=float((N >= thr).mean()), n_pos=int(len(P)), n_neg=int(len(N)),
           eff_at_0628=float((P >= 0.628).mean()), rule="8th percentile of cross-fitted held-out injection ensemble scores (A6)")
json.dump(out, open(MADGRAV_ROOT + "/m2/rank/s3_threshold.json", "w"), indent=1); print(out)
