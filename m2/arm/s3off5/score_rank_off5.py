"""Score the _off5 FAR bank and zero-lag triggers with the PRODUCTION ranker models (madgrav2_spec/rank/models_<run>.joblib,
trained 2026-09-24 on campaign noise + injections; NOTHING is refitted here). Same statistic as train_prod.py lines 67-78:
stat = lnL if lnL < 3 else 3 + 20*pbar, fold-h rows scored by the fold-(1-h) models.
In: R_RANK_DIR/bgfeats_<run>.npz (build_bg_feats_off5 + add_missing_bank_off5), R_RANK_DIR/feats_<run>.npz (build_feats_off5, trigger block).
Out: R_RANK_DIR/stat_<run>.npz (SM_STAT_HOOK of the _off5 _s3rank merge) + stat_identity_<run>.npz + stat_<run>.json."""
import os, json, time, numpy as np, joblib
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

RUN = os.environ["SM_RUN"]; D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"; OD = os.environ["R_RANK_DIR"]; assert "madgrav2_spec" not in OD, OD
RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1"); FLOOR = 3.0; SCALE = 20.0
def raw(A, p, idx=None): X = np.column_stack([A[f"{p}_{k}"] for k in RAW]); return X if idx is None else X[idx]
models = joblib.load(f"{D}/models_{RUN}.joblib"); assert set(models) == {0, 1} and all(len(models[g]) == 5 for g in models)
B = np.load(f"{OD}/bgfeats_{RUN}.npz", allow_pickle=True); ok = np.isfinite(B["bg_loglr"]) & (B["bg_loglr"] >= FLOOR) & np.isfinite(B["bg_coh"]) & np.isfinite(B["bg_gx"])
Xb = raw({f"bg_{k}": B[f"bg_{k}"][ok] for k in RAW}, "bg"); bf = B["bg_fold"][ok]; bl = B["bg_loglr"][ok]; pb = np.full(len(bl), np.nan)
for h in (0, 1):
    m = bf == h; pb[m] = np.mean([mdl.predict_proba(Xb[m])[:, 1] for mdl in models[1 - h]], 0)
Z = np.load(f"{OD}/feats_{RUN}.npz", allow_pickle=True)
tok = np.isfinite(Z["trig_coh"]) & np.isfinite(Z["trig_gx"]) & np.isfinite(Z["trig_loglr"]); Xt = raw(Z, "trig", tok); tf = Z["trig_fold"][tok]; tl = Z["trig_loglr"][tok]; pt = np.full(len(tl), np.nan)
for h in (0, 1):
    m = tf == h; pt[m] = np.mean([mdl.predict_proba(Xt[m])[:, 1] for mdl in models[1 - h]], 0)
n_hi_missing = int((np.isfinite(Z["trig_loglr"]) & (Z["trig_loglr"] >= FLOOR) & ~tok).sum()); assert n_hi_missing == 0, f"{n_hi_missing} triggers with lnL>=3 lack features"
stat_b = np.where(bl < FLOOR, bl, FLOOR + SCALE * pb); stat_t = np.where(tl < FLOOR, tl, FLOOR + SCALE * pt)
common = dict(bank_H1_seg=B["bg_H1_seg"][ok], bank_iH=B["bg_iH"][ok], bank_L1_seg=B["bg_L1_seg"][ok], bank_iL=B["bg_iL"][ok], bank_loglr=bl, bank_p=pb, trig_seg=Z["trig_seg"][tok], trig_gps=Z["trig_gps"][tok], trig_loglr=tl, trig_p=pt)
np.savez(f"{OD}/stat_{RUN}.npz", bank_stat=stat_b, trig_stat=stat_t, **common); np.savez(f"{OD}/stat_identity_{RUN}.npz", bank_stat=bl, trig_stat=tl, **common)
json.dump(dict(run=RUN, models=f"{D}/models_{RUN}.joblib", bgfeats=f"{OD}/bgfeats_{RUN}.npz", feats=f"{OD}/feats_{RUN}.npz", floor=FLOOR, scale=SCALE, n_bank=int(len(bl)), n_trig=int(len(tl)), built=time.strftime("%F %T")), open(f"{OD}/stat_{RUN}.json", "w"), indent=1)
print(f"[rank-off5] {RUN}: bank {len(bl)} rows (p median {np.nanmedian(pb):.3f}), triggers {len(tl)} (lnL>=3: {int((tl>=FLOOR).sum())}) -> {OD}/stat_{RUN}.npz", flush=True)
