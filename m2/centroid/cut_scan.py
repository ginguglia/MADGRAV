"""Fixed-cut scan for the cluster centroid definitions: what a per-detector morphology cut at C Hz does to every population."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

O = MADGRAV_SCRATCH + "/madgrav2/centroid"
L = lambda p: np.load(f"{O}/feat_{p}.npz", allow_pickle=True)
pos, pg, det, gl, top, mk, cat = L("pos"), L("posg8"), L("det47"), L("glitch"), L("top50"), L("murky"), L("cat")
ad = cat["adopted"].astype(bool)
CUTS = [170, 180, 190, 200, 210, 220, 240, 260, 300]
def frac_above(z, k, c): v = z[k]; return float(np.mean(np.isfinite(v) & (v >= c)))
def fam_removed(z, k, c, key):
    ok = {}
    for f, v in zip(z[key], z[k]): ok[f] = ok.get(f, True) & (not (np.isfinite(v) and v >= c))
    return 1 - np.mean(list(ok.values())), len(ok)
for k in ("clu98", "clu99", "exc99_f500"):
    print(f"\n### {k}: per-detector cut C (tile passes if centroid < C)\n")
    print("| C [Hz] | adopted events lost (of 45) | missed events with a leg above (of 93) | inj loss sigma6.5-10 | inj loss g8 sigma4-8 | glitch removed | murky removed | murky O3a/O3b/O4a/O4b | top50 fam removed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in CUTS:
        adl, nad = fam_removed({"name": cat["name"][ad], k: cat[k][ad]}, k, c, "name")
        msl, nms = fam_removed({"name": cat["name"][~ad], k: cat[k][~ad]}, k, c, "name")
        tr, _ = fam_removed(top, k, c, "fam")
        byrun = "/".join(f"{100*frac_above({k: mk[k][mk['run']==r]}, k, c):.0f}" for r in ("o3a", "o3b", "o4a", "o4b"))
        print(f"| {c} | {round(adl*nad)} | {round(msl*nms)} | {100*frac_above(pos,k,c):.1f}% | {100*frac_above(pg,k,c):.1f}% | {100*frac_above(gl,k,c):.1f}% | {100*frac_above(mk,k,c):.1f}% | {byrun} | {100*tr:.0f}% |")
# glitch classes at 200 Hz for clu98
k, c = "clu98", 200
print(f"\n### {k} at {c} Hz: removal per glitch class (n, % removed)\n")
for cl in sorted(set(gl["label"]), key=lambda x: -np.sum(gl["label"] == x)):
    m = gl["label"] == cl; print(f"- {cl}: n={m.sum()}, removed {100*frac_above({k: gl[k][m]}, k, c):.0f}%")
print(f"\n### murky removal at {c} Hz by run and detector ({k})\n")
for r in ("o3a", "o3b", "o4a", "o4b"):
    for d in ("H1", "L1"):
        m = (mk["run"] == r) & (mk["det"] == d); print(f"- {r} {d}: n={m.sum()}, removed {100*frac_above({k: mk[k][m]}, k, c):.0f}%")
print(f"\n### top-50 families: legs above {c} Hz ({k}); family rank, run, H1/L1 centroid\n")
fam = {}
for f, d, v, r in zip(top["fam"], top["det"], top[k], top["run"]): fam.setdefault(f, {})[d] = v; fam[f]["run"] = r
rem = [(f, x) for f, x in fam.items() if any(np.isfinite(x[d]) and x[d] >= c for d in ("H1", "L1"))]
print(f"{len(rem)}/50 removed: " + ", ".join(f"{f}({x['run']}) {x['H1']:.0f}/{x['L1']:.0f}" for f, x in sorted(rem)[:50]))
print("kept: " + ", ".join(f"{f}({x['run']}) {x['H1']:.0f}/{x['L1']:.0f}" for f, x in sorted(fam.items()) if (f, x) not in rem))
