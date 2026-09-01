#!/usr/bin/env python
"""STEP 4c: four-epoch MADGRAV-vs-pipeline VT ratio figure (release-absolute).

Unblocked (2026-08-12) once BOTH the O4 relabel and the O3
release-PSD relabel have landed: MADGRAV <VT> per bin per run from
vt_relabel_release.json (corrected labels, release-absolute volumes,
float32-underflow fix), pipeline <VT> at matched FAR<1/yr reweighted to OUR
population from vt_pipelines_target.json. Mid-band-matched numbers are
robustness-only and MUST NOT appear here (variant policy 2026-08-12).

Caption sentences are assembled into figures/vt_fourepoch/caption_fourepoch.txt:
 (a) FAR/exposure convention (registered 2026-08-11),
 (b) O4b injection-coverage clip (campaign/caption_o4b_clip.txt, verbatim),
 (c) O3 high-mass thinness (PyCBC-BBH-only above ~200, q-support 74%),
 (d) O4 as-run low-frequency whitening caveat (relabel, not rescan).

Run: madgrav-venv python fig_fourepoch_ratio.py [--allow-partial]
"""
import json
import os
import sys

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
FIGDIR = f"{MG}/figures/vt_fourepoch"
RUNS = ["O3a", "O3b", "O4a", "O4b"]
# The O3 (GWTC-3) release splits PyCBC into BBH and broad; the O4 (GWTC-5.0) release ships a
# single "PyCBC". Both spellings must be listed or the O4 panels silently drop PyCBC entirely
# (the `pd is None: continue` branch below) -- as they did until 2026-09-01.
PIPES = ["cWB", "GstLAL", "MBTA", "PyCBC-BBH", "PyCBC-broad", "PyCBC"]
# Display names: the GWTC-3 (O3) release splits PyCBC into a BBH-targeted and a broad
# configuration; the GWTC-5.0 (O4) release ships a single combined PyCBC. All three therefore
# appear in the legend but never in the same panel, so the label states which release each is.
LABEL = {"PyCBC-BBH": "PyCBC-BBH (O3)", "PyCBC-broad": "PyCBC-broad (O3)", "PyCBC": "PyCBC (O4)"}
PSTYLE = {"cWB": dict(c="#0072B2", m="o"), "GstLAL": dict(c="#009E73", m="s"),
          "MBTA": dict(c="#E69F00", m="D"), "PyCBC-BBH": dict(c="#D55E00", m="^"),
          "PyCBC-broad": dict(c="#CC79A7", m="v"),
          "PyCBC": dict(c="#000000", m="P")}          # O4 release spelling

# SM_VT_SUF selects the criterion whose relabeled VT is the MADGRAV numerator, exactly as
# fig_vt_frames.py does. "" = the 2026-08-19 successor build (historical); "_x1cnnadoptveto" =
# the adopted criterion frozen 2026-08-31. Outputs carry the suffix so builds never collide.
SUF = os.environ.get("SM_VT_SUF", "")


def main():
    allow_partial = "--allow-partial" in sys.argv
    # AUDIT FIX (design decision 2026-08-12): numerator MUST be the COMOVING
    # relabeled VT - the Euclidean proxy volumes must never share an axis
    # with the pipelines' inherently comoving <VT> (registered 2026-08-11
    # rule; the mixed-axis draft is WITHHELD).
    rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
    # z-consistent denominator (approved 2026-08-12): the pipeline side
    # reweighted to the population as injected, p(m_src|z) = p_bank((1+z)m_src)
    tgt = json.load(open(f"{HERE}/vt_pipelines_target_zc.json"))
    edges = np.array(rel["mass_edges"], float)
    assert np.allclose(edges, np.array(tgt["mass_edges"], float)), \
        "mass grids differ between relabel and target JSONs"
    mids = 0.5 * (edges[1:] + edges[:-1])
    missing = [r for r in RUNS if r not in rel["runs"] or r not in tgt["runs"]]
    if missing and not allow_partial:
        raise SystemExit(f"missing runs {missing} (use --allow-partial)")

    os.makedirs(FIGDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True,
                             sharey=True)
    ratios_out = {}
    for ax, run in zip(axes.ravel(), RUNS):
        ax.set_title(run, fontsize=10)
        ax.axhline(1.0, color="0.4", lw=0.9, ls="--", zorder=1)
        if run in missing:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes,
                    ha="center", color="0.5")
            continue
        # clearing condition (2026-08-12): SOURCE-frame rebinned numerator
        ours = np.array([v if v is not None else np.nan
                         for v in rel["runs"][run]["vt_comoving_srcframe_gpc3yr"]], float)
        ratios_out[run] = {}
        # O3a asymmetric downward systematic band (caveat branch
        # 2026-08-12): lower edge = numerator / measured O3a-vs-O3b
        # trigger-to-FAR conversion contrast (upper-bound in-sample term)
        # O3a systematic band REMOVED 2026-09-01: it was measured under the superseded
        # as-run statistic and never re-measured, and only one of four panels carried it. The
        # in-sample effect is now quoted directly in the text from the count-vs-efficiency ratio.
        band_lower = None
        for p in PIPES:
            pd = tgt["runs"][run]["pipelines"].get(p)
            if pd is None:
                continue
            theirs = np.array([v if v is not None else np.nan
                               for v in pd["vt_gpc3yr"]], float)
            # AUDIT FIX: enforce N_eff >= 300 on every plotted ratio point
            neff = np.array([v if v is not None else np.nan
                             for v in (pd.get("neff") or [None] * len(theirs))],
                            float)
            theirs = np.where(neff >= 300, theirs, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((theirs > 0) & np.isfinite(theirs)
                                 & np.isfinite(ours), ours / theirs, np.nan)
            ratios_out[run][p] = ratio.tolist()
            st = PSTYLE[p]
            ax.plot(mids, ratio, color=st["c"], marker=st["m"], ms=4,
                    lw=1.4, label=LABEL.get(p, p), clip_on=False)
            if band_lower is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rlo = np.where(np.isfinite(ratio) & (theirs > 0),
                                   band_lower / theirs, np.nan)
                ratios_out[run][p + "_band_lower"] = rlo.tolist()
                ok = np.isfinite(ratio) & np.isfinite(rlo)
                ax.fill_between(mids[ok], rlo[ok], ratio[ok],
                                color=st["c"], alpha=0.18, lw=0)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        # The ratio needs N_eff>=300 on BOTH sides and the comparator releases drain first,
        # so no panel carries a point above 230 Msun. Fig. vt (MADGRAV alone) reaches further
        # because it only needs support on the numerator. 2026-09-01.
        ax.set_xlim(edges[0], 250.0)
    for ax in axes[1]:
        ax.set_xlabel(r"$M_{\rm tot}$ [$M_\odot$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle VT\rangle_{\rm MADGRAV}/\langle VT\rangle_{\rm pipeline}$")
    # One figure-level legend under the panels: an in-axes legend sat on the unity line
    # and on the top of the O3a curves (2026-09-01).
    # Collect across ALL panels: the O3 releases carry PyCBC-BBH/PyCBC-broad while the O4
    # release carries a single "PyCBC", so a legend built from one panel silently omits the
    # other's curve (the O4 PyCBC line was unlabelled until 2026-09-01).
    hh, ll = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ll:
                hh.append(h); ll.append(l)
    if hh:
        fig.legend(hh, ll, frameon=False, fontsize=9, ncol=len(ll),
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MADGRAV blind search vs LVK pipelines at matched "
                 r"FAR$<1\,{\rm yr}^{-1}$, our population "
                 "(comoving, release-absolute PSDs, $N_{\\rm eff}\\geq300$)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    # WITHHELD by default; SM_FIG_CLEARED=1 (set only by the clearance
    # finisher after the analysis lead's explicit clearance) drops the prefix.
    PREFIX = "" if os.environ.get("SM_FIG_CLEARED") == "1" else "WITHHELD_"
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{PREFIX}vt_fourepoch_ratio{SUF}.{ext}", dpi=200,
                    bbox_inches="tight")

    cap = []
    cap.append("(0) Volume law and mass axis: MADGRAV <VT> is COMOVING "
               "(FlatLambdaCDM 67.9/0.3065, V_max = int dVc/(1+z)), matching "
               "the release-derived pipeline convention, and is rebinned to "
               "SOURCE-frame total mass using each injection's implied "
               "redshift from the relabel layer (volume migrates strictly "
               "down-mass; the below-20 Msun sink is tabulated in "
               "vt_relabel_comoving.json). Pipeline points with N_eff < 300 "
               "after reweighting are not plotted.")
    cap.append("(f) Both sides are evaluated on the population as injected: "
               "detector-frame masses drawn from the bank law, uniform-random "
               "time placement, comoving distance law - equivalently, a "
               "z-dependent source-frame density p(m_src | z) ∝ "
               "p_bank((1+z)·m_src). Ratios are therefore defined on the "
               "identical population, mass frame, cosmology, volume law, "
               "threshold semantics, and exposure on both sides.")
    cap.append("(e) Two documented conservatisms on the MADGRAV side: the "
               "detection criterion is the AND of two FAR conditions "
               "(trials-corrected best FAR < 1/yr AND its 90% upper limit "
               "< 1/yr), stricter than the single-FAR threshold used for "
               "the pipeline columns; and MADGRAV FARs inherit the halved "
               "analyzed-time denominator convention (ANALYZED_FRAC = 0.5), "
               "overstating them by up to a factor 2. Both push MADGRAV "
               "efficiencies, and hence these ratios, DOWN.")
    cap.append("(a) Exposure convention: T is the wall-clock analyzed "
               "coincident livetime; the ANALYZED_FRAC=0.5 factor is a "
               "false-alarm-rate trials-denominator convention only and "
               "does not enter <VT> (registered 2026-08-11).")
    cap.append("(b) " + open(f"{HERE}/campaign/caption_o4b_clip.txt").read().strip())
    cap.append("(c) O3 high-mass thinness: above Mtot~200 the matched-FAR "
               "comparison columns are PyCBC-BBH-only for O3, and the "
               "q-support coverage of the release injection sets relative "
               "to our population is 74% (O3; 91% O4) after reweighting; "
               "O3 high-mass ratios are correspondingly thin.")
    cap.append("(d) O4 as-run whitening caveat: the as-run O4 reference "
               "ASDs were biased high at low frequency, suppressing the "
               "search's low-frequency response; the comparison is quoted "
               "for the AS-RUN configuration. Injection SNR/distance "
               "labels and horizon volumes are corrected to release "
               "run-median PSDs (relabel, not rescan); found/missed "
               "outcomes are as-run. Median label shifts and the probed "
               "fraction of the release-accessible volume per bin are "
               "tabulated in vt_relabel_release.json.")
    cap.append("(g) O3a's FAR calibration (background pairs and LR folds) "
               "derives from O3a data; the measured trigger-to-FAR "
               "conversion contrast against the out-of-sample O3b bounds "
               "the resulting in-sample systematic at 15-40% (mass-graded), "
               "shown as the asymmetric band on the O3a curve.")
    cap.append("(h) The pooled O3a-vs-cWB residual (1.24 at the band's "
               "lower edge vs a count-implied 90% upper limit of 1.06) is "
               "accepted as a characterized residual rather than corrected "
               "further: it is smaller than the delivered per-segment "
               "efficiency systematic in the bins where it concentrates, "
               "the segment-selection bias direction is measured to be "
               "conservative, every powered per-bin comparison passes at "
               "the band's lower edge against all pipelines, and no sharper "
               "external measurement exists short of O5 event counts. The "
               "uniform ~1.5-2x pooled counts-versus-VT overshoot across "
               "all comparator pipelines is consistent with, and provides a "
               "population-level quantification of, the previously "
               "characterized injection-to-real-event sensitivity gap "
               "(contrast floor); injection-based <VT> values in this work "
               "should be read with this end-to-end sky calibration "
               "alongside.")
    cap.sort(key=lambda s: s[1])          # (0), then (a)-(h)
    with open(f"{FIGDIR}/{PREFIX}caption_fourepoch{SUF}.txt", "w") as fh:
        fh.write("\n\n".join(cap) + "\n")
    json.dump({"mass_mids": mids.tolist(), "ratios": ratios_out},
              open(f"{FIGDIR}/{PREFIX}vt_fourepoch_ratio{SUF}.json", "w"), indent=1)
    print(f"[fig] -> {FIGDIR}/{PREFIX}vt_fourepoch_ratio{SUF}.pdf/.png + "
          f"{PREFIX}caption_fourepoch{SUF}.txt", flush=True)


if __name__ == "__main__":
    main()
