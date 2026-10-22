"""Read-out of the local-ASD veto check (2026-10-07): injection cost and background removal per clause, per run.
Clauses: CNN  = max(hm_loc, lm_loc) < 0.5 (the production clause);  NET = net_loc < 3.0 (net-sigma channel only);
         S3   = s3_loc < 0.5357 given s3 >= 0.5357 (candidate replacement for the loglr channel / S3-gated rows).
Injections: madgrav2/inj_<run>_m2_22veto (seed 20261004, CNN+S3+local). Background: madgrav2/asdveto/bg_<run>.npz."""
import glob, sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH + "/madgrav2"; THR = 0.5356733202934265; RUNS = ["o3a", "o3b", "o4a", "o4b"]
out = ["# Local-ASD veto check — read-out", ""]
def pct(m, n): return f"{100*m/n:.1f}% ({m}/{n})" if n else "n/a"
out += ["## Injection side: P(removed | CNN gate passed and net >= 3), by run and network SNR", "",
        "| run | SNR | gate-pass | CNN clause | NET clause | either (production) | S3-stability (of S3-passers) |", "|---|---|---|---|---|---|---|"]
tot = {}
for run in RUNS:
    fs = sorted(glob.glob(f"{SC}/inj_{run}_m2_22veto/*/*_inj.npz"))
    if not fs: out.append(f"| {run} | — | no files yet | | | | |"); continue
    D = {}
    for f in fs:
        z = np.load(f, allow_pickle=True)
        for k in ("net_snr", "mtot", "net", "cnn_hm", "cnn_lm", "s3", "net_loc", "hm_loc", "lm_loc", "s3_loc"):
            if k in z.files: D.setdefault(k, []).append(np.asarray(z[k], float))
    D = {k: np.concatenate(v) for k, v in D.items()}
    gp = (np.maximum(D["cnn_hm"], D["cnn_lm"]) > 0.5) & (D["net"] >= 3.0)
    cnn = np.maximum(D["hm_loc"], D["lm_loc"]) < 0.5; net = D["net_loc"] < 3.0
    s3p = D["s3"] >= THR; s3v = D["s3_loc"] < THR
    for lo, hi, lab in [(0, 10, "8"), (10, 12, "10"), (12, 15, "12"), (15, 20, "15"), (20, 99, ">=20"), (0, 99, "all")]:
        m = gp & (D["net_snr"] >= lo) & (D["net_snr"] < hi); n = int(m.sum()); ms = s3p & (D["net_snr"] >= lo) & (D["net_snr"] < hi)
        out.append(f"| {run} | {lab} | {n} of {int(((D['net_snr']>=lo)&(D['net_snr']<hi)).sum())} | {pct(int(cnn[m].sum()), n)} | {pct(int(net[m].sum()), n)} | {pct(int((cnn[m]|net[m]).sum()), n)} | {pct(int(s3v[ms].sum()), int(ms.sum()))} |")
    tot[run] = (int(gp.sum()), int(cnn[gp].sum()), int(net[gp].sum()))
    out.append(f"| {run} | files | {len(fs)} | | | | |")
out += ["", "## Injection side by total mass (SNR >= 12, gate-passed)", "", "| run | mtot | gate-pass | CNN clause | NET clause | S3-stability |", "|---|---|---|---|---|---|"]
for run in RUNS:
    fs = sorted(glob.glob(f"{SC}/inj_{run}_m2_22veto/*/*_inj.npz"))
    if not fs: continue
    D = {}
    for f in fs:
        z = np.load(f, allow_pickle=True)
        for k in ("net_snr", "mtot", "net", "cnn_hm", "cnn_lm", "s3", "net_loc", "hm_loc", "lm_loc", "s3_loc"): D.setdefault(k, []).append(np.asarray(z[k], float))
    D = {k: np.concatenate(v) for k, v in D.items()}
    gp = (np.maximum(D["cnn_hm"], D["cnn_lm"]) > 0.5) & (D["net"] >= 3.0) & (D["net_snr"] >= 12)
    cnn = np.maximum(D["hm_loc"], D["lm_loc"]) < 0.5; net = D["net_loc"] < 3.0; s3p = (D["s3"] >= THR) & (D["net_snr"] >= 12); s3v = D["s3_loc"] < THR
    for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 200), (200, 1e9)]:
        m = gp & (D["mtot"] >= lo) & (D["mtot"] < hi); ms = s3p & (D["mtot"] >= lo) & (D["mtot"] < hi); n = int(m.sum())
        out.append(f"| {run} | {lo}-{hi if hi<1e8 else ''} | {n} | {pct(int(cnn[m].sum()), n)} | {pct(int(net[m].sum()), n)} | {pct(int(s3v[ms].sum()), int(ms.sum()))} |")
out += ["", "## Background side: fraction of the K loudest CNN-passing slid pairs removed per clause", "",
        "| run | K | net range | CNN clause | NET clause | either | S3 frozen-pass | S3 local-pass (of frozen-pass) |", "|---|---|---|---|---|---|---|---|"]
for run in RUNS:
    try: z = np.load(f"{SC}/asdveto/bg_{run}.npz", allow_pickle=True)
    except FileNotFoundError: out.append(f"| {run} | — | not done | | | | | |"); continue
    A = {k: np.asarray(z[k], float) for k in ("net", "net_loc", "hm_loc", "lm_loc", "s3", "s3_loc")}; K = len(A["net"])
    cnn = np.maximum(A["hm_loc"], A["lm_loc"]) < 0.5; net = A["net_loc"] < 3.0; s3p = A["s3"] >= THR; s3l = A["s3_loc"] >= THR
    out.append(f"| {run} | {K} | {A['net'].min():.2f}-{A['net'].max():.2f} | {pct(int(cnn.sum()), K)} | {pct(int(net.sum()), K)} | {pct(int((cnn|net).sum()), K)} | {pct(int(np.nansum(s3p)), K)} | {pct(int(np.nansum(s3p & s3l)), int(np.nansum(s3p)))} |")
out += ["", "Decision rule (DECISIONS_ASDVETO.md, written before this read-out): a clause is admissible only if its injection cost is < 2% absolute at",
        "SNR >= 12 in every run and uniform across runs, AND its background removal on O3 is >= the CNN clause's. The sealed catalogue rows are read",
        "once, after the rule is fixed (CENSUS_20261007.md)."]
txt = "\n".join(out); open(sys.argv[1] if len(sys.argv) > 1 else "READOUT_ASDVETO.md", "w").write(txt); print(txt)
