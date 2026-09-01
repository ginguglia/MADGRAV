#!/usr/bin/env python
"""Corrected Sigma-p check (pre-registered follow-up, decision 2026-08-13).

Same 15 pooled O3a cWB-found events and measured sigmoid as sigmap_check.py,
with the recovery probability corrected by the 3f in-sample contrast:
det_frac = eps_trigger x FAR-conversion, and the conversion layer carries the
in-sample term bounded per det-frame Mtot bin by contrast_per_detbin
(vt_o3a_band.json, 1.15-1.40) -> p_corr = eps(SNR_HL) / contrast(bin).

Events are placed in det-frame bins at (1+z) * Mtot_src with catalog
redshifts (cached GWOSC eventapi GWTC-2.1-confident).

Decision statistic (pre-registered, exact): Poisson-binomial P(N <= 9) on
the 15 corrected Bernoullis.  P >= 0.05 -> consistent: residual = in-sample
term, triangulated by three independent methods (band gate, direct Sigma-p,
3f contrast); amended figure clears, caption (h) states the triangulation.
P < 0.05 -> still deficient: 3d per-segment spread is the next named
suspect, figure stays WITHHELD.

Run: madgrav-venv python sigmap_corrected.py
Outputs: sigmap_corrected.json + sigmap_corrected.txt
"""
import json
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
OBSERVED = 9
OBSERVED_SUBSET = 8
EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])

# ---- inputs ------------------------------------------------------------
rows = [l.split(",") for l in
        open(f"{HERE}/cross_recovery_matrix.csv").read().strip().split("\n")[1:]]
cwb15 = [(r[1], float(r[2]), float(r[3]), int(r[4]))
         for r in rows if r[0] == "o3a" and r[5] == "1"]
assert len(cwb15) == 15 and sum(m for *_, m in cwb15) == OBSERVED_SUBSET

inj = np.load(f"{HERE}/inj_scored_o3a.npz")
snr_i, frac_i = inj["net_snr"], inj["det_frac"]
grid = np.array(sorted(set(np.round(snr_i, 6))))
eps_grid = np.array([frac_i[snr_i == g].mean() for g in grid])

pe = json.load(open(f"{HERE}/campaign/pe_perdet_snrs.json"))
band = json.load(open(f"{HERE}/vt_o3a_band.json"))
contrast = np.array(band["contrast_per_detbin"])

cat = json.load(open(
    MADGRAV_EXTDATA + "/gwosc_eventapi/GWTC-2.1-confident.json"))["events"]
by_common = {e["commonName"]: e for e in cat.values()}


def redshift(name):
    e = by_common.get(name)
    if e is None:                      # GW190412_053044 -> GW190412 etc.
        e = by_common.get(name.split("_")[0])
    assert e is not None, name
    return float(e["redshift"]), e["commonName"]


# ---- per-event corrected probabilities ---------------------------------
events = []
for name, mtot_src, snr_cat, mg in sorted(cwb15, key=lambda t: t[1]):
    e = pe[name]
    shl = float(np.hypot(e["snr_H1"], e["snr_L1"]))
    z, cname = redshift(name)
    mtot_det = mtot_src * (1 + z)
    b = int(np.clip(np.digitize(mtot_det, EDGES) - 1, 0, len(contrast) - 1))
    c = float(contrast[b])
    p_raw = float(np.interp(shl, grid, eps_grid))
    p_cor = p_raw / c
    events.append(dict(name=name, mtot_src=mtot_src, z=z,
                       mtot_det=round(mtot_det, 1),
                       bin=f"{EDGES[b]:.0f}-{EDGES[b+1]:.0f}",
                       contrast=round(c, 4), snr_HL=round(shl, 2),
                       p_raw=round(p_raw, 4), p_corr=round(p_cor, 4),
                       madgrav=mg))

ps = np.array([e["p_corr"] for e in events])
sp_raw = sum(e["p_raw"] for e in events)
sp = float(ps.sum())

# exact Poisson-binomial
pb = np.zeros(len(ps) + 1)
pb[0] = 1.0
for p in ps:
    pb[1:] = pb[1:] * (1 - p) + pb[:-1] * p
    pb[0] *= (1 - p)
cdf = np.cumsum(pb)
P_le_9 = float(cdf[OBSERVED])
P_le_8 = float(cdf[OBSERVED_SUBSET])
consistent = P_le_9 >= 0.05

verdict = ("CONSISTENT (P(N<=9) >= 0.05) — residual = in-sample term, "
           "triangulated by band gate, direct Sigma-p, and 3f contrast at "
           "mutual magnitude; amended figure CLEARS, caption (h) states the "
           "triangulation" if consistent else
           "STILL DEFICIENT (P(N<=9) < 0.05) — 3d per-segment spread is the "
           "next named suspect; figure stays WITHHELD")

rep = dict(spec="Corrected Sigma-p check, pre-registered 2026-08-13 "
                "(exact Poisson-binomial; Poisson-interval spec retracted "
                "logged)",
           correction="p_corr = eps(SNR_HL) / contrast_per_detbin at "
                      "(1+z)*Mtot_src; contrast = vt_o3a_band.json (3f "
                      "in-sample FAR-conversion bound, 1.15-1.40)",
           events=events,
           sigma_p_raw=round(sp_raw, 3),
           sigma_p_corrected=round(sp, 3),
           observed=OBSERVED,
           observed_subset_cwb_and_madgrav=OBSERVED_SUBSET,
           P_le_9=round(P_le_9, 4), P_le_8=round(P_le_8, 4),
           threshold=0.05, consistent=bool(consistent),
           gw190513_note="GW190513_205428 is MADGRAV-found but cWB-missed "
                         "(outside the 15-event cWB comparison set); the "
                         "gate's observed 9 includes it, the in-set count "
                         "is 8 — a one-sided comparison set names its "
                         "out-of-set recovery",
           verdict=verdict)
json.dump(rep, open(f"{HERE}/sigmap_corrected.json", "w"), indent=1)

lines = ["CORRECTED SIGMA-P CHECK (exact Poisson-binomial; pre-registered)",
         rep["correction"], "",
         f"{'event':<18}{'Msrc':>6}{'z':>6}{'Mdet':>7}{'bin':>9}"
         f"{'ctr':>7}{'HL':>7}{'p_raw':>7}{'p_cor':>7}  MADGRAV"]
for e in events:
    lines.append(f"{e['name']:<18}{e['mtot_src']:>6.1f}{e['z']:>6.2f}"
                 f"{e['mtot_det']:>7.1f}{e['bin']:>9}{e['contrast']:>7.3f}"
                 f"{e['snr_HL']:>7.2f}{e['p_raw']:>7.3f}{e['p_corr']:>7.3f}"
                 f"  {'YES' if e['madgrav'] else 'no'}")
lines += ["",
          f"Sigma_p raw = {sp_raw:.2f}  ->  corrected = {sp:.2f}   "
          f"observed = {OBSERVED} (in-set {OBSERVED_SUBSET})",
          f"exact Poisson-binomial: P(N<={OBSERVED}) = {P_le_9:.3f}   "
          f"P(N<={OBSERVED_SUBSET}) = {P_le_8:.3f}   threshold 0.05",
          "", rep["gw190513_note"], "", "VERDICT: " + verdict]
open(f"{HERE}/sigmap_corrected.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
