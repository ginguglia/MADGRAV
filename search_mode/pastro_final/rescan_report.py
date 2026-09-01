#!/usr/bin/env python
"""O4a corrected-refs rescan (o4ars) completion report - PRE-REGISTERED SPEC
(decision 2026-08-13, before chain completion):

 (1) rebuilt-background loglr->FAR curves overlaid on as-run, full tail
     (per fold, log axis) -> figures/rescan_o4a/far_overlay.{png,pdf} + table
 (2) candidate list at the paper criterion (best_far<1/yr & UL90<1/yr) with
     per-event AS-RUN status (recovered / as-run sub-threshold trigger / no
     as-run trigger) and catalog match (merged catalog CSV, |dt|<=2 s)
 (3) DeltaN split: catalog-events-gained vs NON-CATALOG candidates; the
     latter are routed to the VETTING PROTOCOL and are NOT tallied
     (pre-commitment 2026-08-13: full vetting standard, post-conference)
 (4) corrected O4a VT(Mtot) from the injection leg - eff at the paper
     criterion via the accepted pastro_final scoring (PF.RUNS extended with
     the rescan; pooled cnn pairs = PF.PAIR_RUNS = 47 as-run pairs ONLY, ruling 2026-08-15 -
     flagged), volumes = per-bank-entry release volumes from the relabel
     layer (c==1 under corrected refs), comoving via the harmonized
     cosmology. Output = after-panel CANDIDATE; it enters the four-epoch
     figure only through the amendment review.

Run AFTER the o4ars chain finalizer has written detections.json.
Out: rescan_o4a_report.{json,txt}, figures/rescan_o4a/
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
NEW = f"{SC}/search_out_o4ars_far"
OLD = f"{SC}/search_out_o4a_far"
FIGD = f"{MG}/figures/rescan_o4a"
CATF = f"{MG}/figures/catalog_o3o4/merged_plot_v2.csv"
DET_FAR = 1.0
MATCH_S = 2.0
ASRUN_MATCH_S = 4.0
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])

sys.path.insert(0, HERE)


def far_curve(bg, fold, grid):
    x = bg["loglr"][bg["fold"] == fold]
    flt = float(bg["far_live"][fold])
    return np.array([(x >= g).sum() / flt for g in grid])


def main():
    os.makedirs(FIGD, exist_ok=True)
    rep = {}
    lines = ["O4A CORRECTED-REFS RESCAN (o4ars) - COMPLETION REPORT", ""]

    # ---- (1) FAR overlay ----
    bg_o = np.load(f"{OLD}/bg_cache_o4a.npz")
    bg_n = np.load(f"{NEW}/bg_cache_o4ars.npz")
    grid = np.linspace(0.0, max(bg_o["loglr"].max(), bg_n["loglr"].max()) + 0.5, 400)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, bg, color in (("as-run", bg_o, "0.55"), ("rescan", bg_n, "C3")):
        for fold in (0, 1):
            c = far_curve(bg, fold, grid)
            ax.plot(grid, np.maximum(c, 1e-4), color=color, ls="-" if fold == 0 else "--",
                    label=f"{tag} fold{fold}")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(4.0, color="k", lw=0.6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("loglr threshold")
    ax.set_ylabel("background FAR [1/yr]")
    ax.set_title("O4a background loglr -> FAR: as-run vs corrected-refs rescan")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGD}/far_overlay.{ext}", dpi=160)
    tab = {}
    for thr in (4, 5, 6, 8, 10, 12, 14):
        tab[thr] = {tag: [float((bg["loglr"][bg["fold"] == f] >= thr).sum() /
                          bg["far_live"][f]) for f in (0, 1)]
                    for tag, bg in (("asrun", bg_o), ("rescan", bg_n))}
    rep["far_table"] = tab
    lines += ["(1) FAR at loglr thresholds [fold0, fold1] (1/yr):"] + [
        f"    loglr>={t}: as-run [{v['asrun'][0]:.3g}, {v['asrun'][1]:.3g}]  "
        f"rescan [{v['rescan'][0]:.3g}, {v['rescan'][1]:.3g}]"
        for t, v in tab.items()] + [
        f"    figure -> figures/rescan_o4a/far_overlay.png", ""]

    # ---- (2)+(3) candidates vs as-run and catalog ----
    dets_n = json.load(open(f"{NEW}/detections.json"))
    dets_o = json.load(open(f"{OLD}/detections.json"))
    trig_o = json.load(open(f"{OLD}/blindscan.json"))["triggers"]
    # catalog match = the chain's own matches_known (accepted matching);
    # merged catalog CSV cross-references name -> confident-catalog row
    cat_rows = {r["name"]: r for r in csv.DictReader(open(CATF))}
    # GPS fallback (docstring: |dt|<=MATCH_S vs merged catalog) when the chain's
    # matches_known is unpopulated (o4ars label had no known-event list wired)
    cat_gps = [(r["name"], float(r["gps"])) for r in csv.DictReader(open(f"{MG}/figures/master.csv"))
               if r["name"] in cat_rows]

    def gps_match(g):
        best = min(cat_gps, key=lambda nr: abs(nr[1] - g), default=None)
        return best[0] if best and abs(best[1] - g) <= MATCH_S else None

    def asrun_status(g):
        for d in dets_o:
            if abs(d["gps"] - g) <= ASRUN_MATCH_S:
                return "RECOVERED-ASRUN", d
        best = None
        for t in trig_o:
            if abs(t["gps"] - g) <= ASRUN_MATCH_S:
                if best is None or t["loglr"] > best["loglr"]:
                    best = t
        if best:
            return "ASRUN-SUBTHRESHOLD", best
        return "NO-ASRUN-TRIGGER", None

    cands, gained, noncat, kept = [], [], [], []
    for d in dets_n:
        far, ul = d.get("best_far"), d.get("best_ul90", np.inf)
        if far is None or far >= DET_FAR or ul >= DET_FAR:
            continue
        st, ref = asrun_status(d["gps"])
        cm = d.get("matches_known") or gps_match(d["gps"])
        crow = cat_rows.get(cm) if cm else None
        row = dict(seg=d["seg"], gps=d["gps"], net=d["net"], loglr=d["loglr"],
                   far=far, ul90=ul, asrun_status=st, catalog=cm,
                   catalog_mtot=(crow or {}).get("total_mass_source"),
                   catalog_madgrav_asrun=(crow or {}).get("madgrav"),
                   asrun_loglr=(ref or {}).get("loglr"))
        cands.append(row)
        if st == "RECOVERED-ASRUN":
            kept.append(row)
        elif cm:
            gained.append(row)
        else:
            noncat.append(row)
    lost = [d for d in dets_o
            if not any(abs(d["gps"] - c["gps"]) <= ASRUN_MATCH_S for c in cands)]
    rep["candidates"] = cands
    rep["delta"] = dict(n_asrun=len(dets_o), n_rescan=len(cands),
                        kept=len(kept), catalog_gained=len(gained),
                        noncatalog_to_vetting=len(noncat), lost_vs_asrun=len(lost))
    lines += [f"(2) candidates at best_far<{DET_FAR}/yr & UL90<{DET_FAR}/yr: {len(cands)}"]
    for c in cands:
        lines.append(f"    {c['seg']} gps={c['gps']:.0f} loglr={c['loglr']:.2f} "
                     f"far={c['far']:.3g}  [{c['asrun_status']}]  "
                     f"catalog={c['catalog'] or 'NONE'}")
    lines += ["", f"(3) DeltaN: as-run {len(dets_o)} -> rescan {len(cands)} "
              f"(kept {len(kept)}, catalog-gained {len(gained)}, "
              f"LOST vs as-run {len(lost)})",
              f"    NON-CATALOG candidates -> VETTING PROTOCOL (not tallied): "
              f"{len(noncat)}"]
    for c in noncat:
        lines.append(f"      VETTING: {c['seg']} gps={c['gps']:.0f} "
                     f"loglr={c['loglr']:.2f} far={c['far']:.3g}")
    for d in lost:
        lines.append(f"      LOST vs as-run: {d['seg']} gps={d['gps']:.0f} "
                     f"(as-run loglr {d['loglr']:.2f})")
    lines.append("")

    # ---- (4) corrected O4a VT from the injection leg ----
    try:
        import pastro_final as PF
        PF.RUNS = dict(PF.RUNS)
        PF.RUNS["O4ars"] = dict(out=NEW, inj=[f"{SM}/inj_out_o4ars"])
        # RULING 2026-08-15: pooled cnn pairs = PF.PAIR_RUNS = the 47 as-run
        # pairs ONLY; the rescan's own detections are NOT pooled in (the
        # flagged deviation is resolved as out-of-sample by construction).
        assert "O4ars" not in PF.PAIR_RUNS and len(PF.PAIR_RUNS) == 4
        PF.main()
        z = np.load(f"{HERE}/inj_scored_o4ars.npz")
        rel = json.load(open(f"{HERE}/vt_relabel_release.json"))
        T = rel["runs"]["O4a"]["T_obs_yr"]
        mb = np.clip(np.digitize(z["mtot"], MASS_EDGES) - 1, 0, len(MASS_EDGES) - 2)
        eff = [float((z["det_frac"][mb == i] * z["w0"][mb == i]).sum() /
                     max(z["w0"][mb == i].sum(), 1e-300))
               for i in range(len(MASS_EDGES) - 1)]
        rep["vt_after"] = dict(mass_edges=list(MASS_EDGES), eff_at_far1=eff,
                               T_obs_yr=T,
                               note="eff leg from accepted PF scoring; volume "
                                    "assembly (per-bank-entry release volumes, "
                                    "c==1, comoving) slots in at amendment "
                                    "review -> four-epoch 'after' panel")
        lines += ["(4) corrected O4a efficiency at paper criterion, per Mtot bin:",
                  "    " + "  ".join(f"{MASS_EDGES[i]:.0f}-{MASS_EDGES[i+1]:.0f}:"
                                     f"{eff[i]:.2f}" for i in range(len(eff))),
                  "    (volume assembly -> 'after' panel at amendment review)", ""]
    except Exception as e:
        rep["vt_after"] = {"error": str(e)}
        lines += [f"(4) VT leg FAILED: {e} - run wrapper manually", ""]

    json.dump(rep, open(f"{HERE}/rescan_o4a_report.json", "w"), indent=1, default=str)
    with open(f"{HERE}/rescan_o4a_report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
