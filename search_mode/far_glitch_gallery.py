"""Gallery of the background families that SET the quoted FARs, per run.

Configuration: lnLambda channel only, sigma_net<10.6 veto, whole-segment self-exclusion --
the adopted stack. For each calibrated detection we take the families counted against it on the
winning arm; the union over detections is the FAR-setting population. This renders a
representative sample of that population as whitened Q-transform tiles (H1 above L1), which is
what has to survive inspection.

Selection is blind: families are ordered by lnLambda and sampled at even rank intervals, so the
gallery spans the population rather than showing the extremes.

Usage: SM_GAL_RUN=O3a SM_GAL_N=16 far_glitch_gallery.py
"""
import os, sys, csv
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

DET = f"{MG}/details/successor_statistic"
FS = 4096; WN = 4 * FS; NETMAX = 10.6
RUN = os.environ.get("SM_GAL_RUN", "O3a")
NGAL = int(os.environ.get("SM_GAL_N", "16"))
KE = {'O3a': 8.91, 'O3b': 9.46, 'O4a': 2.83, 'O4b': 5.50}[RUN]
STR = f"{MADGRAV_SCRATCH}/strain_{RUN.lower()}_full"


def window(seg, gps, det):
    z = np.load(f"{STR}/{seg}_{det}.npz")
    t0 = float(z["gps_start"]); x = z["strain"]
    i = int(round((gps - 2.0 - t0) * FS))
    return x[i:i + WN].astype(np.float32) if 0 <= i and i + WN <= len(x) else None


def main():
    bg = S.Background(RUN, f"{DET}/bg_veto_{RUN.lower()}.npz", verbose=False)
    inv = {v: k for k, v in bg.seg_ix.items()}
    rows = [r for r in csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"))
            if r["run"] == RUN and float(r["net"]) < NETMAX]
    counts = {}
    for r in rows:
        Sx = bg.seg_ix[r["seg"]]; f = int(bg.seg_fold[Sx])
        keep = (bg.hseg != Sx) & (bg.lseg != Sx) & (bg.net < NETMAX) & (bg.fold == f)
        mm = keep & (bg.ll > float(r["loglr"]))
        fh = np.unique(bg.fam[mm & (bg.hm >= float(r["cnn_hm"]))])
        fl = np.unique(bg.fam[mm & (bg.lm >= float(r["cnn_lm"]))])
        win = fh if len(fh) <= len(fl) else fl
        T = float(bg.F[f]["T"]); N = len(win)
        if (N / T) * KE < 1 and ((N + 2.302585) / T) * KE < 1:
            for fam in win.tolist():
                counts[fam] = counts.get(fam, 0) + 1
    print(f"[{RUN}] {len(counts)} FAR-setting families over {len(rows)} candidates", flush=True)
    # one representative pair per family: its loudest-lnLambda member
    reps = {}
    for fam in counts:
        idx = np.flatnonzero((bg.fam == fam) & (bg.net < NETMAX))
        if len(idx): reps[fam] = int(idx[np.argmax(bg.ll[idx])])
    fams = sorted(reps, key=lambda k: -bg.ll[reps[k]])
    pick = [fams[i] for i in np.linspace(0, len(fams) - 1, min(NGAL, len(fams))).round().astype(int)]
    pipe = MassiveEventPipeline(calib_path=f"{MG}/spectrogram_cascade/massive_calibration_BA.json",
                                prep=f"{MG}/data/{RUN.lower()}_search_prep", device="cpu")
    tiles = []
    for fam in pick:
        p = reps[fam]
        wh = window(inv[int(bg.hseg[p])], float(bg.gpsH[p]), "H1")
        wl = window(inv[int(bg.lseg[p])], float(bg.gpsL[p]), "L1")
        if wh is None or wl is None: continue
        tiles.append((p, counts[fam],
                      DS.build_qt(pipe, pipe._whiten(wh[None, :], "H1"))[0],
                      DS.build_qt(pipe, pipe._whiten(wl[None, :], "L1"))[0]))
        print(f"  rendered {len(tiles)}/{len(pick)}", flush=True)
    n = len(tiles); ncol = 8; nblk = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(2 * nblk, ncol, figsize=(2.05 * ncol, 2.9 * nblk))
    ax = np.atleast_2d(ax)
    for a in ax.ravel(): a.axis("off")
    for j, (p, c, qh, ql) in enumerate(tiles):
        b, k = divmod(j, ncol)
        for i, q in ((0, qh), (1, ql)):
            a = ax[2 * b + i][k]; a.axis("on")
            a.imshow(q, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(f"$\\ln\\Lambda$={bg.ll[p]:.1f} $\\sigma$={bg.net[p]:.1f}\n"
                            f"HM={bg.hm[p]:.2f} LM={bg.lm[p]:.2f}  x{c}", fontsize=6.5, color="#B00")
            if k == 0:
                a.set_ylabel("H1" if i == 0 else "L1", fontsize=8)
    fig.suptitle(f"{RUN}: background families that SET the quoted FARs "
                 f"({len(counts)} in total, {n} shown, sampled evenly in $\\ln\\Lambda$)\n"
                 f"lnLambda-only ranking, $\\sigma_{{net}}<10.6$ veto.  "
                 f"'xN' = number of detections this family counts against", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95 if nblk > 1 else 0.90])
    out = f"{MG}/figures/far_glitches_{RUN.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
