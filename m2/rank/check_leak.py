"""Leakage check for the sideband read-out: replay build_feats.py's deterministic noise sampler (seed 20260923, same NCAP) to recover the
(segment, window) identity of every training-noise leg per fold, then ask: for the FAR-family pairs of fold h (scored by the model
trained on fold g = 1-h), how many have their H1 leg / L1 leg / either leg among that model's training-noise legs? Same question
for the injection carriers is moot (injections are synthetic). Env: campaign env + R_NCAP (default 60000)."""
import os, sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]; sys.path.insert(0, f"{MADGRAV_ROOT}/search_mode"); os.chdir(MADGRAV_ROOT)
os.environ.setdefault("SM_DEV", "cpu"); os.environ.setdefault("SM_ARM_V2", "1")
import driver_search_multi as M
RUN = os.environ["SM_RUN"]; NETMIN = float(os.environ.get("SM_LR_NETMIN", "3.0")); NCAP = int(os.environ.get("R_NCAP", "60000")); D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"
rng = np.random.default_rng(20260923)
SEG = M.SEG["segments"]; avail = [(a, nm) for (a, b, d, nm) in SEG]; segs = [M.load_seg(a, nm) for a, nm in avail]; fold = [i % 2 for i in range(len(segs))]
EVG = np.array(sorted(M.EVENTS.values()), float)
def is_ev(si, idx):
    if not len(EVG): return np.zeros(len(idx), bool)
    g = segs[si]["gps"][idx]; j = np.searchsorted(EVG, g); lo = np.clip(j - 1, 0, len(EVG) - 1); hi = np.clip(j, 0, len(EVG) - 1)
    return (np.abs(EVG[lo] - g) <= 4.0) | (np.abs(EVG[hi] - g) <= 4.0)
npr = {0: [], 1: []}; tries = 0
while (len(npr[0]) < NCAP or len(npr[1]) < NCAP) and tries < 800000:
    tries += 1; ai, bi = rng.integers(len(segs), size=2)
    if ai == bi: continue
    g = fold[ai]
    if len(npr[g]) >= NCAP: continue
    A, B = segs[ai], segs[bi]; k = int(rng.integers(-60, 61)); off = 4 * k + 2
    iH, iL = M.pair_idx(A["n"], B["n"], off)
    if not len(iH): continue
    net = (A["sH"][iH] + B["sL"][iL]) / np.sqrt(2.0)
    m = np.where((net >= NETMIN) & M.cen_gate(A["cH"][iH], B["cL"][iL]) & M.arm_gate(A["gH"][iH], B["gL"][iL], A["sH"][iH], B["sL"][iL]) & ~(is_ev(ai, iH) | is_ev(bi, iL)))[0]
    for j in m[:200]:
        npr[g].append((ai, int(iH[j]), bi, int(iL[j]), off))
        if len(npr[g]) >= NCAP: break
Z = np.load(f"{D}/feats_{RUN}.npz", allow_pickle=True); assert all(len(npr[g]) == len(Z[f"noise{g}_sH"]) for g in (0, 1)), "replay size mismatch"
assert all(np.allclose(np.array([p[4] for p in npr[g]]), Z[f"noise{g}_lag"]) for g in (0, 1)), "replay lag mismatch (sampler not reproduced)"
legsH = {g: {(segs[p[0]]["name"], p[1]) for p in npr[g]} for g in (0, 1)}; legsL = {g: {(segs[p[2]]["name"], p[3]) for p in npr[g]} for g in (0, 1)}
S = np.load(MADGRAV_SCRATCH + "/madgrav2_spec/protected_sel.npz", allow_pickle=True); sel = np.where((S["run"] == RUN) & (S["source"] == "sideband"))[0]
W = np.load(f"{D}/feats_side_{RUN}.npz", allow_pickle=True); sf = W["side_fold"]
out = [f"[leak] {RUN}: training-noise legs per fold: H1 {len(legsH[0])}/{len(legsH[1])}, L1 {len(legsL[0])}/{len(legsL[1])} (of {sum(len(npr[g]) for g in (0,1))} pairs)"]
for h in (0, 1):
    g = 1 - h; m = np.where(sf == h)[0]
    inH = np.array([(S["H1_seg"][sel[i]], int(S["iH"][sel[i]])) in legsH[g] for i in m]); inL = np.array([(S["L1_seg"][sel[i]], int(S["iL"][sel[i]])) in legsL[g] for i in m])
    out.append(f"[leak] {RUN} sideband fold {h} ({len(m)} pairs, scored by the fold-{g} model): H1 leg in its training noise {100*inH.mean():.1f}%, L1 leg {100*inL.mean():.1f}%, either {100*(inH|inL).mean():.1f}%")
    # same-segment (not same window) exposure: legs from the same segment within +-64 s
    byseg = {}
    for s_, w in legsL[g]: byseg.setdefault(s_, []).append(w)
    byseg = {s_: np.sort(np.array(v)) for s_, v in byseg.items()}
    def near(s_, w):
        a = byseg.get(s_)
        if a is None or not len(a): return False
        j = np.searchsorted(a, w); return (j < len(a) and a[j] - w <= 64) or (j > 0 and w - a[j - 1] <= 64)
    nearL = np.array([near(S['L1_seg'][sel[i]], int(S['iL'][sel[i]])) for i in m]) if len(m) else np.array([])
    out.append(f"[leak] {RUN} sideband fold {h}: L1 leg within +-64 s of a training-noise L1 leg of the same segment: {100*nearL.mean():.1f}%")
txt = "\n".join(out); print(txt, flush=True); open(f"{D}/leak_{RUN}.txt", "w").write(txt + "\n")
