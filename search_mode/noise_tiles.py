"""Render the time-frequency tiles of BACKGROUND families that pass the final detection checks.

Picks time-slide pseudo-candidates whose per-arm conditioned FAR is below 1/yr -- i.e. noise that the
statistic would call a detection -- and plots the whitened Q-transform the CAE actually sees, beside
a real detection for scale. The point is to look at what is passing.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MG = MADGRAV_ROOT
sys.path.insert(0, f"{MG}/search_mode")
import successor_stat as S
import driver_streams as DS
from massive_pipeline import MassiveEventPipeline
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

FS = 4096; WN = 4 * FS
RUN = os.environ.get("SM_TILE_RUN", "O3a")
SEED = int(os.environ.get("SM_TILE_SEED", "11"))
FOLD = int(os.environ.get("SM_TILE_FOLD", "0"))
REF_EVENT = os.environ.get("SM_TILE_EVENT", "GW190828_063405")
# SM_TILE_NETMAX: show only families that SURVIVE the sigma_net veto -- i.e. the residual
# problem population after the cut, which is what defines the next step.
NETMAX = float(os.environ.get("SM_TILE_NETMAX", "0")) or None
# SM_TILE_LRONLY=1 -> rank on the lnLambda channel alone (sigma_net channel dropped from the FAR).
LR_ONLY = os.environ.get("SM_TILE_LRONLY", "0") == "1"
STR = f"{MADGRAV_SCRATCH}/strain_{RUN.lower()}_full"
PREP = f"{MG}/data/{RUN.lower()}_search_prep"

def counts_veto(bg, p):
    """Per-arm conditioned counts for background pair p, with the sigma_net veto applied to the
    background too (same rule as the candidate side)."""
    Sx = int(bg.hseg[p]); f = int(bg.fold[p])
    keep = (bg.hseg != Sx) & (bg.lseg != int(bg.lseg[p]))
    if NETMAX is not None:
        keep &= bg.net < NETMAX
    m = (bg.fold == f) & keep
    ll, hm, lm, net = float(bg.ll[p]), float(bg.hm[p]), float(bg.lm[p]), float(bg.net[p])
    mm = m & (bg.ll > ll)
    F = bg.F[f]; rep = F["rep_net"]
    rk = (bg.hseg[rep] != Sx) & (bg.net[rep] >= net)
    if NETMAX is not None:
        rk &= bg.net[rep] < NETMAX
    rr = rep[rk]
    out = dict(lr_hm=len(np.unique(bg.fam[mm & (bg.hm >= hm)])),
               lr_lm=len(np.unique(bg.fam[mm & (bg.lm >= lm)])),
               net_hm=int((bg.hm[rr] >= hm).sum()), net_lm=int((bg.lm[rr] >= lm).sum()),
               T=float(F["T"]))
    if LR_ONLY:
        out["net_hm"] = out["net_lm"] = 10**9      # channel removed from the minimum
    return out


def window(segname, gps, det):
    z = np.load(f"{STR}/{segname}_{det}.npz")
    t0 = float(z["gps_start"]); x = z["strain"]
    i = int(round((gps - 2.0 - t0) * FS))
    if i < 0 or i + WN > len(x): return None
    return x[i:i + WN].astype(np.float32)

def main():
    bg = S.Background(RUN, f"{MG}/details/successor_statistic/bg_veto_{RUN.lower()}.npz", verbose=False)
    inv = {v: k for k, v in bg.seg_ix.items()}
    z = np.load(f"{MG}/details/successor_statistic/pseudo_fg_{RUN.lower()}_f{FOLD}.npz")
    pop = z["pop"][~z["vetoed"]]
    rng = np.random.default_rng(SEED)
    cand = rng.choice(pop, 900, replace=False)
    hits = []
    for p in cand:
        p = int(p)
        if NETMAX is not None and float(bg.net[p]) >= NETMAX:
            continue                      # vetoed as a candidate
        if LR_ONLY and not (np.isfinite(bg.ll[p]) and float(bg.ll[p]) >= 4.0):
            continue                      # no lnLambda channel -> cannot be ranked at all
        c = counts_veto(bg, p)
        N = min(min(c["lr_hm"], c["lr_lm"]), min(c["net_hm"], c["net_lm"]))
        far = N / c["T"]
        if far < 1.0 and (N + 2.302585) / c["T"] < 1.0:
            hits.append((far, p, N))
    hits.sort()
    print(f"[noise-tiles] {len(hits)} of {len(cand)} sampled background families pass FAR<1/yr "
          f"({100*len(hits)/len(cand):.1f}%)", flush=True)
    pick = [hits[0], hits[len(hits)//3], hits[2*len(hits)//3], hits[-1]] if len(hits) >= 4 else hits
    pipe = MassiveEventPipeline(calib_path=f"{MG}/spectrogram_cascade/massive_calibration_BA.json",
                                prep=PREP, device="cpu")
    rows = []
    for far, p, N in pick:
        hs, ls = inv[int(bg.hseg[p])], inv[int(bg.lseg[p])]
        wh = window(hs, float(bg.gpsH[p]), "H1"); wl = window(ls, float(bg.gpsL[p]), "L1")
        if wh is None or wl is None:
            print(f"  skip pair {p}: window off segment"); continue
        rows.append(dict(kind="background", far=far, N=N, ll=float(bg.ll[p]), net=float(bg.net[p]),
                         hm=float(bg.hm[p]), lm=float(bg.lm[p]),
                         lab=f"time-slide  H1 {hs[4:]}  L1 {ls[4:]}", H=wh, L=wl))
    import csv as _csv
    ev = [r for r in _csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"))
          if r["run"] == RUN and r["name"] == REF_EVENT]
    if ev:
        r = ev[0]; g = float(r["gps"]); sname = r["seg"]
        wh = window(sname, g, "H1"); wl = window(sname, g, "L1")
        if wh is not None:
            rows.append(dict(kind="detection", far=float(r["far"]), N=int(r["N_bg"]), ll=float(r["loglr"]),
                             net=float(r["net"]), hm=float(r["cnn_hm"]), lm=float(r["cnn_lm"]),
                             lab=f"real event  {REF_EVENT}", H=wh, L=wl))
    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(3.1 * n, 5.6))
    for j, r in enumerate(rows):
        for i, det in enumerate(("H", "L")):
            w = pipe._whiten(r[det][None, :], "H1" if det == "H" else "L1")
            qt = DS.build_qt(pipe, w)[0]
            a = ax[i][j] if n > 1 else ax[i]
            a.imshow(qt, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                col = "#B00" if r["kind"] == "background" else "#060"
                a.set_title(f"{r['lab']}\nFAR={r['far']:.3f}/yr  N={r['N']}\n"
                            f"$\\ln\\Lambda$={r['ll']:.1f}  $\\sigma_{{net}}$={r['net']:.1f}  "
                            f"HM={r['hm']:.2f} LM={r['lm']:.2f}", fontsize=7.5, color=col)
            if j == 0:
                a.set_ylabel(f"{'Hanford' if det=='H' else 'Livingston'}\nfrequency", fontsize=8)
    fig.suptitle("What passes the final checks: time-slide BACKGROUND (red) vs a real detection (green)\n"
                 "whitened Q-transform tiles, exactly as the autoencoder sees them", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{MG}/figures/noise_tiles_{('lronly' if LR_ONLY else 'survivors') if NETMAX else 'passing'}_{RUN.lower()}_s{SEED}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
