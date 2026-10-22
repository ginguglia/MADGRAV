"""Centroid (clu98) on the MADGRAV2 positive population: pass curve, centroid tail vs Mtot and SNR, loss of the 200 Hz cut
among TRIGGERING injections (sigma_net>=4), mass-agnostic p99/p99.9 of the centroid over triggering injections.
Usage: python analyze_positives.py [tag]   (tag '' = production shards, '_smoke' = smoke)"""
import sys, glob, os, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0, MADGRAV_ROOT + "/m2/centroid"); from centroid_defs import cluster_centroid
D = MADGRAV_SCRATCH + "/madgrav2/positives"; TAG = sys.argv[1] if len(sys.argv) > 1 else ""
fs = sorted(f for f in glob.glob(f"{D}/pos_*{TAG}_shard*.npz") if (TAG or "smoke" not in f))
cols = {k: [] for k in ("run", "mtot", "snr_target", "sigma_H", "sigma_L", "sigma_net", "pass_net", "cH", "cL")}
for f in fs:
    z = np.load(f, allow_pickle=True); n = len(z["mtot"]); run = os.path.basename(f).split("_")[1]
    cols["run"].append(np.full(n, run))
    for k in ("mtot", "snr_target", "sigma_H", "sigma_L", "sigma_net", "pass_net"): cols[k].append(z[k])
    for k, q in (("cH", "qt_H"), ("cL", "qt_L")):
        out = []
        for i in range(0, n, 512): out.append(cluster_centroid(np.asarray(z[q][i:i+512], np.float32), 98.0))
        cols[k].append(np.concatenate(out))
c = {k: np.concatenate(v) for k, v in cols.items()}; np.savez(f"{D}/feat_posm2{TAG}.npz", **c)
trig = c["pass_net"] == 1; C = 200.0
print(f"{len(c['mtot'])} injection pairs from {len(fs)} shards; triggering (sigma_net>=4): {trig.sum()} ({100*trig.mean():.0f}%)")
MB = [15, 24, 38, 60, 95, 150, 240, 380, 800]; SB = [5, 8, 12, 18, 30]
def cell(m, fn): return fn(m) if m.sum() else float("nan")
print("\n## CAE pass fraction (sigma_net>=4), Mtot bin x network SNR bin (n)")
print("Mtot \\ SNR     " + "".join(f"{a}-{b:<9}" for a, b in zip(SB[:-1], SB[1:])))
for a, b in zip(MB[:-1], MB[1:]):
    m = (c["mtot"] >= a) & (c["mtot"] < b); print(f"{a:3d}-{b:<3d}        " + "".join(f"{c['pass_net'][m&(c['snr_target']>=x)&(c['snr_target']<y)].mean():.2f}({(m&(c['snr_target']>=x)&(c['snr_target']<y)).sum():4d})  " if (m&(c['snr_target']>=x)&(c['snr_target']<y)).any() else "    -         " for x, y in zip(SB[:-1], SB[1:])))
fail = lambda m: float(np.mean((np.isfinite(c["cH"][m]) & (c["cH"][m] >= C)) | (np.isfinite(c["cL"][m]) & (c["cL"][m] >= C))))
print(f"\n## Fraction of TRIGGERING injections lost by clu98 >= {C:.0f} Hz on either leg, Mtot x SNR (n)")
print("Mtot \\ SNR     " + "".join(f"{a}-{b:<9}" for a, b in zip(SB[:-1], SB[1:])))
for a, b in zip(MB[:-1], MB[1:]):
    m = trig & (c["mtot"] >= a) & (c["mtot"] < b); line = f"{a:3d}-{b:<3d}        "
    for x, y in zip(SB[:-1], SB[1:]):
        mm = m & (c["snr_target"] >= x) & (c["snr_target"] < y); line += f"{fail(mm):.3f}({mm.sum():4d})  " if mm.any() else "    -         "
    print(line)
print(f"\noverall loss among triggering injections at {C:.0f} Hz: {100*fail(trig):.2f}%  (by run: " + ", ".join(f"{r} {100*fail(trig&(c['run']==r)):.2f}%" for r in np.unique(c["run"])) + ")")
for cc in (190, 200, 210, 220, 250): print(f"  cut {cc}: loss {100*np.mean((c['cH'][trig]>=cc)|(c['cL'][trig]>=cc)):.2f}%")
legs = np.concatenate([c["cH"][trig], c["cL"][trig]]); legs = legs[np.isfinite(legs)]
print(f"\nmass-agnostic centroid quantiles over triggering legs: p50 {np.percentile(legs,50):.0f}, p90 {np.percentile(legs,90):.0f}, p99 {np.percentile(legs,99):.0f}, p99.9 {np.percentile(legs,99.9):.0f} Hz")
print("\n## centroid p50/p99 of triggering legs by Mtot bin")
for a, b in zip(MB[:-1], MB[1:]):
    m = trig & (c["mtot"] >= a) & (c["mtot"] < b); l = np.concatenate([c["cH"][m], c["cL"][m]]); l = l[np.isfinite(l)]
    if len(l): print(f"  {a:3d}-{b:<3d}: n={len(l):5d} p50 {np.percentile(l,50):4.0f} p90 {np.percentile(l,90):4.0f} p99 {np.percentile(l,99):4.0f} Hz; loss@200 {100*fail(m):.2f}%")
