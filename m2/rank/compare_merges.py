"""PREREG_RANKER_20260924 §3.4 + §4: compare the campaign detections (detections_lm.json) with the identity-hook merge (_rankid; must be
identical) and the ranker merge (_rank). Per run and total: kept / lost / new detections (catalogue-matched or not), FAR before/after
for each detection, counts at FAR < 1 / 0.1 / 0.01 per yr, loudest non-catalogue candidates. Usage: compare_merges.py [runs...]"""
import os, sys, json, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH + "/madgrav2"; runs = [a for a in sys.argv[1:] if not a.startswith("_")] or ["o3a", "o3b", "o4a", "o4b"]
TAGS = [a for a in sys.argv[1:] if a.startswith("_")] or ["_rank", "_s3", "_s3rank"]   # merges to compare against the campaign (_lm)
def load(run, tag):
    f = f"{SC}/search_out_{run}_m2_gx/detections{tag}.json"
    return {(d.get("matches_known") or f"{run}_{d['gps']:.0f}"): d for d in json.load(open(f))} if os.path.exists(f) else None
out = []; TOT = {}; KLN = {}
for run in runs:
    A = load(run, "_lm"); I = load(run, "_rankid")
    if A is None: out.append(f"== {run}: no campaign detections_lm.json"); continue
    out.append(f"== {run}: campaign {len(A)} detections | identity merge {'MISSING' if I is None else len(I)}")
    if I is not None:
        same = set(A) == set(I) and all(abs((A[k]["best_far"] or 0) - (I[k]["best_far"] or 0)) < 1e-9 for k in A)
        out.append(f"  identity check: {'PASS (same events, same best_far)' if same else 'FAIL'}" + ("" if same else f" -> only in campaign {sorted(set(A)-set(I))}, only in identity {sorted(set(I)-set(A))}"))
    for j, t in enumerate((1, 0.1, 0.01)): TOT.setdefault("lm", [0, 0, 0])[j] += sum(1 for d in A.values() if d.get("matches_known") and (d["best_far"] or 0) < t)
    for tag in TAGS:
        R = load(run, tag)
        if R is None: out.append(f"  [{tag}] MISSING"); continue
        for j, t in enumerate((1, 0.1, 0.01)): TOT.setdefault(tag, [0, 0, 0])[j] += sum(1 for d in R.values() if d.get("matches_known") and (d["best_far"] or 0) < t)
        k_ = sorted(set(A) & set(R)); l_ = sorted(set(A) - set(R)); n_ = sorted(set(R) - set(A)); KLN.setdefault(tag, [0, 0, 0]); KLN[tag][0] += len(k_); KLN[tag][1] += len(l_); KLN[tag][2] += len(n_)
        out.append(f"  [{tag}] {len(R)} detections: kept {len(k_)}, LOST {len(l_)} {l_}, NEW {len(n_)} " + str([(k, f"{R[k]['best_far']:.3g}/yr", R[k]["channel"], "catalogue" if R[k].get("matches_known") else "non-catalogue") for k in n_]))
        rows = [(k, A[k]["best_far"], R[k]["best_far"]) for k in k_]; imp = [r for r in rows if r[1] and r[1] > 0]
        out.append(f"  [{tag}] kept: best_far zero under campaign {sum(1 for r in rows if not r[1])}, now {sum(1 for r in rows if not r[2])}" + (f"; median FAR ratio campaign/new over {len(imp)} with campaign FAR>0: {np.median([r[1]/(r[2] or 1e-6) for r in imp]):.1f}" if imp else ""))
        worse = [r for r in rows if (r[2] or 0) > (r[1] or 0) * 1.5 + 1e-9]
        if worse: out.append(f"  [{tag}] WORSE FAR: " + "; ".join(f"{r[0]} {r[1]:.3g}->{r[2]:.3g}" for r in worse))
        nc = sorted([d for d in R.values() if not d.get("matches_known")], key=lambda d: d["best_far"] or 0)
        if nc: out.append(f"  [{tag}] non-catalogue: " + "; ".join(f"{run}_{d['gps']:.0f} ({(d['best_far'] or 0):.3g}/yr, {d['channel']})" for d in nc))
out.append("== TOTAL catalogue detections at FAR <1 / <0.1 / <0.01 per yr: campaign " + str(TOT.get("lm")) + " | " + " | ".join(f"{t} {TOT[t]} (kept/lost/new {KLN[t]})" for t in TAGS if t in TOT))
txt = "\n".join(out); print(txt); open(MADGRAV_SCRATCH + "/madgrav2_spec/rank/compare_merges.txt", "w").write(txt + "\n")
