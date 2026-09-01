#!/usr/bin/env python
"""Sigma-p check (pre-registered decider for the amended four-epoch figure).

For each of the 15 pooled O3a cWB-found events (the cWB set of the pooled
named check, r = 9/15): direct per-event prediction of MADGRAV recovery at
the paper criterion (best_far < 1/yr AND UL90 < 1/yr).

  eps(SNR)  : measured recovery sigmoid from the O3a injection campaign
              (inj_scored_o3a.npz, det_frac = mean over the 47 empirical cnn
              pairs at the paper criterion), empirical mean at each injected
              net-SNR grid point {5,6,7,8,10,12,15,20,25}, linear
              interpolation between points, clamped at the ends.
  SNR_ev    : H1-L1-only network SNR from the catalog per-detector
              matched-filter SNR posterior medians (GWTC-2.1 PEDataRelease
              mixed_cosmo, campaign/pe_perdet_snrs.json):
              sqrt(snr_H1^2 + snr_L1^2).  Virgo excluded by construction —
              MADGRAV is H1-L1 only; the catalog network SNR would overcount.
  P(recover) = eps(SNR_ev);  Sigma_p = sum over the 15 events.

Pre-registered interpretation: observed 9 inside the central Poisson 90%
interval of lambda = Sigma_p -> residual x1.32 = contrast floor on a faint
real-event population, amended figure clears with caption (h) + one
sentence.  9 below the interval -> STOP and report.

Supplementary (reported, not decision inputs): stellar-bank-only sigmoid
variant; observed-8 note (MADGRAV-and-cWB subset — the gate's 9 counts
GW190513_205428, MADGRAV-found but cWB-missed, outside the 15).

Run: madgrav-venv python sigmap_check.py
Outputs: sigmap_check.json + sigmap_check.txt
"""
import glob, json, os
import numpy as np
from scipy.stats import poisson
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SC = MADGRAV_SCRATCH
OBSERVED = 9          # pre-registered comparison count (gate numerator)
OBSERVED_SUBSET = 8   # MADGRAV recoveries among the 15 cWB-found events

# ---- events: cWB-found O3a set from the cross-recovery matrix ----------
rows = [l.split(",") for l in
        open(f"{HERE}/cross_recovery_matrix.csv").read().strip().split("\n")[1:]]
cwb15 = [(r[1], float(r[2]), float(r[3]), int(r[4]))
         for r in rows if r[0] == "o3a" and r[5] == "1"]
assert len(cwb15) == 15, f"expected 15 cWB-found O3a events, got {len(cwb15)}"
assert sum(m for *_, m in cwb15) == OBSERVED_SUBSET

# ---- sigmoid from the injection campaign -------------------------------
inj = np.load(f"{HERE}/inj_scored_o3a.npz")
snr_i, frac_i = inj["net_snr"], inj["det_frac"]
grid = np.array(sorted(set(np.round(snr_i, 6))))
eps_grid = np.array([frac_i[snr_i == g].mean() for g in grid])
n_grid = np.array([(snr_i == g).sum() for g in grid])
# binomial-style standard error of the mean det_frac at each grid point
se_grid = np.array([frac_i[snr_i == g].std() / np.sqrt((snr_i == g).sum())
                    for g in grid])


def eps(x, e=eps_grid):
    return float(np.interp(x, grid, e))


# stellar-only variant: rebuild is_um in inj_scored order (vt_search recipe)
def load_is_um():
    dirs = [f"{SC}/inj_out_o3a_56", f"{MG}/search_mode/inj_out_o3a_lowsnr"]
    parts = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*_inj.npz")):
            parts.append(np.load(f)["is_um"].astype(bool))
    return np.concatenate(parts)


try:
    is_um = load_is_um()
    assert len(is_um) == len(snr_i)
    eps_sig = np.array([frac_i[(snr_i == g) & ~is_um].mean() for g in grid])
except Exception as ex:                       # variant is supplementary only
    is_um, eps_sig = None, None
    print("stellar-only variant unavailable:", ex)

# ---- per-event evaluation ----------------------------------------------
pe = json.load(open(f"{HERE}/campaign/pe_perdet_snrs.json"))
events, sp, sp_sig = [], 0.0, 0.0
for name, mtot, snr_cat, mg in sorted(cwb15, key=lambda t: t[1]):
    e = pe[name]
    h1, l1 = e["snr_H1"], e["snr_L1"]
    shl = float(np.hypot(h1, l1))
    p = eps(shl)
    p_sig = eps(shl, eps_sig) if eps_sig is not None else None
    sp += p
    sp_sig += (p_sig or 0.0)
    events.append(dict(name=name, mtot=mtot, madgrav=mg,
                       snr_H1=round(h1, 2), snr_L1=round(l1, 2),
                       snr_V1=round(e.get("snr_V1", float("nan")), 2),
                       snr_HL=round(shl, 2),
                       snr_cat_net=snr_cat,
                       p_recover=round(p, 4),
                       p_recover_stellar_only=(round(p_sig, 4)
                                               if p_sig is not None else None),
                       pe_analysis=e.get("analysis"),
                       snr_key=e.get("snr_H1_key")))

# ---- Poisson 90% interval (pre-registered) -----------------------------
lam = sp
k_lo = int(poisson.ppf(0.05, lam))
k_hi = int(poisson.ppf(0.95, lam))
inside = k_lo <= OBSERVED <= k_hi
# supplementary: exact Poisson-binomial CDF at the observed counts
ps = np.array([e["p_recover"] for e in events])
pb = np.zeros(len(ps) + 1)
pb[0] = 1.0
for p in ps:
    pb[1:] = pb[1:] * (1 - p) + pb[:-1] * p
    pb[0] *= (1 - p)
cdf = np.cumsum(pb)
pb_p_le_obs = float(cdf[OBSERVED])
pb_p_le_8 = float(cdf[OBSERVED_SUBSET])

verdict = ("CLEARS — 9 inside Poisson 90% interval; residual x1.32 = "
           "contrast floor on a faint real-event population"
           if inside and OBSERVED >= k_lo else
           "STOP — 9 below the interval; 3d per-segment spread becomes the "
           "next named suspect; no clearance"
           if OBSERVED < k_lo else
           "9 ABOVE the interval — outside pre-registered branches; report")

rep = dict(
    spec="Sigma-p check, pre-registered 2026-08-13",
    criterion="best_far<1/yr AND UL90<1/yr (det_frac over 47 cnn pairs)",
    sigmoid=dict(grid=grid.tolist(),
                 eps=[round(x, 4) for x in eps_grid],
                 se=[round(x, 4) for x in se_grid],
                 n=n_grid.tolist(),
                 eps_stellar_only=([round(x, 4) for x in eps_sig]
                                   if eps_sig is not None else None),
                 source="inj_scored_o3a.npz (5 event-hosting segments, "
                        "folds f0/f1, Mtot 20-400)"),
    events=events,
    sigma_p=round(sp, 3),
    sigma_p_stellar_only=(round(sp_sig, 3) if eps_sig is not None else None),
    poisson90=[k_lo, k_hi],
    observed=OBSERVED,
    observed_subset_cwb_and_madgrav=OBSERVED_SUBSET,
    inside=bool(inside),
    poisson_binomial_P_le_9=round(pb_p_le_obs, 4),
    poisson_binomial_P_le_8=round(pb_p_le_8, 4),
    verdict=verdict)

json.dump(rep, open(f"{HERE}/sigmap_check.json", "w"), indent=1)

lines = ["SIGMA-P CHECK (pre-registered decider, amended four-epoch figure)",
         f"criterion: {rep['criterion']}",
         "sigmoid eps(net SNR H1L1), measured, linear interp:",
         "  " + "  ".join(f"{g:.0f}:{e:.3f}" for g, e in zip(grid, eps_grid)),
         "", f"{'event':<18}{'Mtot':>7}{'H1':>7}{'L1':>7}{'HL':>7}"
             f"{'catSNR':>8}{'P(rec)':>8}  MADGRAV"]
for e in events:
    lines.append(f"{e['name']:<18}{e['mtot']:>7.1f}{e['snr_H1']:>7.2f}"
                 f"{e['snr_L1']:>7.2f}{e['snr_HL']:>7.2f}"
                 f"{e['snr_cat_net']:>8.1f}{e['p_recover']:>8.3f}"
                 f"  {'YES' if e['madgrav'] else 'no'}")
lines += ["",
          f"Sigma_p = {sp:.2f}   Poisson 90% interval = [{k_lo}, {k_hi}]   "
          f"observed = {OBSERVED}  ->  {'INSIDE' if inside else 'OUTSIDE'}",
          f"(stellar-only sigmoid variant: Sigma_p = {sp_sig:.2f})"
          if eps_sig is not None else "",
          f"supplementary Poisson-binomial: P(N<=9) = {pb_p_le_obs:.3f}, "
          f"P(N<=8) = {pb_p_le_8:.3f}",
          f"note: observed-in-set = {OBSERVED_SUBSET} (gate's 9 includes "
          "GW190513_205428, MADGRAV-found / cWB-missed, outside the 15)",
          "", "VERDICT: " + verdict]
open(f"{HERE}/sigmap_check.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
