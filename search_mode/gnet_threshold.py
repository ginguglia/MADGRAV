"""Glitch-arm gate threshold, set on the INJECTION population only (2026-09-09).

Mirrors the sigma_net<10.6 upper veto (p99.9 of gate-passing injections): the gate statistic is the
network glitch-arm score g_net = (gH + gL)/sqrt(2) of the deploy-arm ensemble logits (positive = signal-like),
and the threshold is the SM_GNET_PCT-th percentile (default 1.0, i.e. a 1% loss of gate-passing injections)
of g_net over the ADOPTED injection campaign (inj_fixed2, main + lowsnr, the files pastro_final.py reads),
restricted to injections that would be candidates: sigma_net>4 trigger AND max(s_HM, s_LM)>0.5.
Pooled over the four runs, as the 10.6 threshold was. Candidates, detections and the time-slide background
are never read here. Output: details/successor_statistic/gnet_threshold.json
"""
import os, glob, json, time
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH; DET = MADGRAV_ROOT + "/details/successor_statistic"
PCT = float(os.environ.get("SM_GNET_PCT", "1.0")); CAMP = os.environ.get("SM_GNET_CAMPAIGN", "inj_fixed2")
per = {}; pooled = []; pooled_min = []
for run in ("o3a", "o3b", "o4a", "o4b"):
    v = []; vm = []; nf = 0; ntot = 0
    for d in (f"{SC}/{CAMP}/{run}", f"{SC}/{CAMP}/{run}_lowsnr"):
        for f in sorted(glob.glob(f"{d}/*_inj.npz")):
            z = np.load(f); nf += 1; ntot += len(z["net"])
            k = (z["net"] > 4.0) & (np.maximum(z["cnn_hm"], z["cnn_lm"]) > 0.5)
            v.append(((z["gH"] + z["gL"]) / np.sqrt(2))[k]); vm.append(np.minimum(z["gH"], z["gL"])[k])
    v = np.concatenate(v); vm = np.concatenate(vm); pooled.append(v); pooled_min.append(vm)
    per[run] = dict(files=nf, injections=ntot, candidates=int(len(v)),
                    gnet_pct={str(p): float(np.percentile(v, p)) for p in (0.1, 0.5, 1, 2, 5, 50)},
                    gnet_at_pct=float(np.percentile(v, PCT)))
v = np.concatenate(pooled); vm = np.concatenate(pooled_min)
thr = float(np.percentile(v, PCT))
out = dict(statistic="g_net = (gH + gL)/sqrt(2), deploy-arm 5-seed ensemble logit, positive = signal-like",
           rule=f"gate passes iff g_net >= threshold; threshold = p{PCT:g} of g_net over pooled gate-passing (max(s_HM,s_LM)>0.5), "
                f"triggering (sigma_net>4) injections of campaign {CAMP} (main + lowsnr)",
           campaign=CAMP, percentile=PCT, n_candidates_pooled=int(len(v)), threshold=thr,
           pooled_gnet_pct={str(p): float(np.percentile(v, p)) for p in (0.1, 0.5, 1, 2, 5, 50)},
           pooled_min_g_pct={str(p): float(np.percentile(vm, p)) for p in (0.1, 0.5, 1, 2, 5, 50)},
           per_run=per, date=time.strftime("%F %T"))
json.dump(out, open(f"{DET}/gnet_threshold.json", "w"), indent=1)
print(f"[gnet] {len(v)} pooled candidate injections; p{PCT:g} threshold g_net >= {thr:.3f}  -> {DET}/gnet_threshold.json")
for r, p in per.items(): print(f"   {r}: {p['candidates']} candidates, own p{PCT:g} = {p['gnet_at_pct']:.2f}")
