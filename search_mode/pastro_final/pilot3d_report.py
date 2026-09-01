#!/usr/bin/env python
"""STEP 3d report: segment-representativeness of epsilon (specification 2026-08-12).

Compares, per run and mass bin, the frozen chain's injection efficiency on
RANDOM non-event segments (pilot3d draw) against the EVENT-HOSTING injection
segments the quoted VT uses - plus the segment-to-segment spread of epsilon
as the systematic on the VT. Expected outcome is agreement; if eps_random
sits below eps_event-hosting, the VT curves carry that as a downward
correction.

Implementation (confirmed 2026-08-12): outcomes come from the frozen
chain under the AS-RUN prep (matching the quoted search), while injection
LABELS go through the relabel layer (matching the quoted VT): per-injection
c from relabel_c_<run>.npz (written by step-4's vt_relabel_release.py, exact
mtot->entry match), physical SNR = nominal * c, kept = rho_phys >= 5,
pooled-efficiency weights w = s^-4 dr * c^-3. O3 runs fall back to c=1 with
a loud warning if the relabel files are absent; O4 runs REQUIRE them.

Detection proxy: net sigma >= 4.0 at the peak grid window - the blindscan
STAGE-1 trigger definition, identical for both samples (the quoted det_frac
is FAR-level; representativeness at trigger level is the driver, and any
tension found here escalates to a FAR-level confirmation).

Errors: per-bin Wilson 90% intervals on unweighted recovered/total counts
(both samples), reported next to the weighted pooled epsilon. Spread: std
across segments of the per-segment pooled epsilon per bin (segments with
>= 50 injections in the bin; needs >= 3 such segments).

Inputs : search_mode/inj_out_pilot3d_<run>/*.npz (random), campaign inj dirs
         (event-hosting, same as vt_relabel INJ_DIRS), relabel_c_<run>.npz.
Outputs: pilot3d_report.json + text table on stdout.
"""
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
SM = f"{MG}/search_mode"
HERE = f"{MG}/search_mode/pastro_final"
RUNS = ["o3a", "o3b", "o4a", "o4b"]
NET_CUT = 4.0
RHO_TH = 5.0
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
INJ_DIRS = {"o3a": [f"{SC}/inj_out_o3a_56", f"{SM}/inj_out_o3a_lowsnr"],
            "o3b": [f"{SM}/inj_out_o3b", f"{SM}/inj_out_o3b_lowsnr"],
            "o4a": [f"{SM}/inj_out_o4a", f"{SM}/inj_out_o4a_lowsnr"],
            "o4b": [f"{SM}/inj_out_o4b", f"{SM}/inj_out_o4b_lowsnr"]}
MIN_PER_SEG_BIN = 50


def wilson90(k, n):
    if n == 0:
        return (None, None)
    z = 1.6449
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def c_lookup(run):
    path = f"{HERE}/relabel_c_{run}.npz"
    if not os.path.exists(path):
        if run in ("o4a", "o4b"):
            raise SystemExit(f"FATAL: {path} missing - O4 labels MUST go "
                             "through the relabel layer (run step-4 first)")
        print(f"[{run}] WARNING: {path} missing -> c=1 fallback (O3 only)",
              flush=True)
        return None
    z = np.load(path)
    maps = {}
    for tag in ("sig", "um"):
        mt, c, ok = z[f"mt_{tag}"], z[f"c_{tag}"], z[f"ok_{tag}"]
        d = {}
        for m, cv in zip(mt[ok], c[ok]):
            d.setdefault(float(m), []).append(float(cv))
        maps[tag] = {m: float(np.mean(v)) for m, v in d.items()}
    return maps


def load_sample(files, cmaps, run):
    P = {k: [] for k in ("mtot", "net_snr", "is_um", "net", "seg")}
    for f in sorted(files):
        z = np.load(f)
        n = len(z["mtot"])
        for k in ("mtot", "net_snr", "is_um", "net"):
            P[k].append(np.asarray(z[k], float))
        P["seg"].append(np.full(n, os.path.basename(f).replace("_inj.npz", "")))
    if not P["mtot"]:
        return None
    S = {k: np.concatenate(v) for k, v in P.items()}
    if cmaps is None:
        S["c"] = np.ones(len(S["mtot"]))
    else:
        c = np.empty(len(S["mtot"]))
        miss = 0
        for i, (m, u) in enumerate(zip(S["mtot"], S["is_um"])):
            v = cmaps["um" if u else "sig"].get(float(m))
            if v is None:
                miss += 1
                c[i] = np.nan
            else:
                c[i] = v
        assert miss == 0, f"{run}: {miss} injections unmatched in relabel_c"
        S["c"] = c
    S["det"] = (S["net"] >= NET_CUT).astype(float)
    S["kept"] = S["net_snr"] * S["c"] >= RHO_TH
    levels = np.unique(S["net_snr"])
    dr = np.gradient(levels)
    wmap = {float(s): float(s ** -4.0 * d) for s, d in zip(levels, dr)}
    S["w"] = np.array([wmap[float(s)] for s in S["net_snr"]]) * S["c"] ** -3.0
    S["bin"] = np.digitize(S["mtot"], MASS_EDGES) - 1
    return S


def pooled_eff(S, sel):
    sel = sel & S["kept"]
    w = S["w"][sel]
    if not sel.sum() or w.sum() <= 0:
        return None
    return float((S["det"][sel] * w).sum() / w.sum())


def main():
    rep = {"net_cut": NET_CUT, "proxy": "stage-1 trigger (net>=4), both "
           "samples identically; labels via relabel layer",
           "mass_edges": MASS_EDGES.tolist(), "runs": {}}
    lines = ["STEP 3d REPORT - segment-representativeness of epsilon", ""]
    for run in RUNS:
        cmaps = c_lookup(run)
        rand = load_sample(glob.glob(f"{SM}/inj_out_pilot3d_{run}/*_inj.npz"),
                           cmaps, run)
        evt_files = [f for d in INJ_DIRS[run]
                     for f in glob.glob(f"{d}/*_inj.npz")]
        evt = load_sample(evt_files, cmaps, run)
        if rand is None or evt is None:
            lines.append(f"[{run}] MISSING SAMPLE (random: "
                         f"{'ok' if rand else 'absent'}, event: "
                         f"{'ok' if evt else 'absent'}) - skipped")
            continue
        # grid comparability
        lr, le = set(np.unique(rand["net_snr"])), set(np.unique(evt["net_snr"]))
        common = sorted(lr & le)
        note = ""
        if lr != le:
            note = (f"grids differ (rand {sorted(lr)} vs evt {sorted(le)}); "
                    f"restricted to common {common}")
            for S in (rand, evt):
                S["_gridmask"] = np.isin(S["net_snr"], common)
        else:
            for S in (rand, evt):
                S["_gridmask"] = np.ones(len(S["net_snr"]), bool)
        R = {"bins": [], "note": note,
             "n_random_segments": int(len(np.unique(rand["seg"]))),
             "n_event_segments": int(len(np.unique(evt["seg"])))}
        lines.append(f"[{run}] random segs={R['n_random_segments']} vs "
                     f"event-hosting segs={R['n_event_segments']}"
                     + (f"  ({note})" if note else ""))
        for k in range(len(MASS_EDGES) - 1):
            row = {"bin": f"{MASS_EDGES[k]:.0f}-{MASS_EDGES[k+1]:.0f}"}
            for lab, S in (("random", rand), ("event", evt)):
                sel = (S["bin"] == k) & S["_gridmask"]
                eff = pooled_eff(S, sel)
                selk = sel & S["kept"]
                n = int(selk.sum())
                krec = int(S["det"][selk].sum())
                lo, hi = wilson90(krec, n)
                row[lab] = dict(eff=eff, n=n, recovered=krec,
                                wilson90=[lo, hi])
            # per-segment spread (random sample)
            effs = []
            for sname in np.unique(rand["seg"]):
                sel = ((rand["bin"] == k) & (rand["seg"] == sname)
                       & rand["_gridmask"] & rand["kept"])
                if sel.sum() >= MIN_PER_SEG_BIN:
                    w = rand["w"][sel]
                    effs.append(float((rand["det"][sel] * w).sum() / w.sum()))
            row["seg_spread_std"] = (float(np.std(effs, ddof=1))
                                     if len(effs) >= 3 else None)
            row["seg_spread_n"] = len(effs)
            er_, ee_ = row["random"]["eff"], row["event"]["eff"]
            row["ratio"] = (er_ / ee_ if er_ is not None and ee_ not in
                            (None, 0) else None)
            R["bins"].append(row)
            er, ee = row["random"]["eff"], row["event"]["eff"]
            if er is not None and ee is not None:
                sp = row["seg_spread_std"]
                lines.append(
                    f"  {row['bin']:>8s}: rand {er:.3f} "
                    f"[{row['random']['wilson90'][0]:.3f},"
                    f"{row['random']['wilson90'][1]:.3f}] "
                    f"(n={row['random']['n']})  evt {ee:.3f} "
                    f"[{row['event']['wilson90'][0]:.3f},"
                    f"{row['event']['wilson90'][1]:.3f}] "
                    f"(n={row['event']['n']})  ratio "
                    f"{er / ee if ee > 0 else float('nan'):.3f}  "
                    f"seg-std {sp if sp is None else round(sp, 3)}"
                    f" ({row['seg_spread_n']} segs)")
        # ---- pooled ratio + PRE-REGISTERED correction (2026-08-12):
        # per det-frame bin use the per-bin ratio when BOTH samples have
        # >= 200 kept injections in the bin and eff_event > 0; otherwise the
        # run-pooled ratio. Applied uniformly to every run by the gate
        # preview; single correction, no iteration beyond it.
        pooled = {}
        for lab, S in (("random", rand), ("event", evt)):
            sel = S["_gridmask"]
            pooled[lab] = pooled_eff(S, sel)
        R["ratio_pooled"] = (pooled["random"] / pooled["event"]
                             if pooled["event"] else None)
        corr, corr_src = [], []
        for row in R["bins"]:
            ok = (row["random"]["n"] >= 200 and row["event"]["n"] >= 200
                  and row["event"]["eff"] not in (None, 0)
                  and row["ratio"] is not None)
            corr.append(row["ratio"] if ok else R["ratio_pooled"])
            corr_src.append("per-bin" if ok else "run-pooled")
        R["correction_per_bin"] = corr
        R["correction_source"] = corr_src
        R["correction_rule"] = ("per-bin ratio if n_kept>=200 both samples "
                                "and eff_event>0, else run-pooled ratio "
                                "(pre-registered 2026-08-12)")
        lines.append(f"  [{run}] pooled eff ratio rand/evt = "
                     f"{R['ratio_pooled']:.3f}" if R["ratio_pooled"]
                     else f"  [{run}] pooled ratio unavailable")

        # ---- per-segment breakout (decision-ready form, 2026-08-12):
        # full per-segment eff distribution; for O3a additionally mark each
        # random segment against the TRAINING-ERA set (the 56
        # calibration/arm-development segments, o3a_bg_segments_56.json -
        # the frozen CAE's strain training is O1-legacy; O3a in-sample
        # status enters via the arm/calibration layer).
        segcfg = json.load(open(f"{SM}/pilot3d_{run}_segs.json"))
        train_iv = None
        if run == "o3a":
            t56 = json.load(open(f"{SM}/o3a_bg_segments_56.json"))["segments"]
            train_iv = np.array([[s[0], s[1]] for s in t56], float)
        per_seg = []
        for sname in sorted(np.unique(rand["seg"])):
            sel = (rand["seg"] == sname) & rand["_gridmask"] & rand["kept"]
            n = int(sel.sum())
            e = (float((rand["det"][sel] * rand["w"][sel]).sum()
                       / rand["w"][sel].sum()) if n else None)
            t0 = float(segcfg[sname]["coincident_lock"][0])
            entry = dict(seg=sname, t0=t0, n_kept=n, eff_pooled=e)
            if train_iv is not None:
                dmin = float(np.min(np.maximum(train_iv[:, 0] - t0,
                                               t0 - train_iv[:, 1])))
                entry["min_dist_to_training_seg_s"] = max(0.0, dmin)
                entry["inside_training_span"] = bool(
                    train_iv[:, 0].min() <= t0 <= train_iv[:, 1].max())
            per_seg.append(entry)
        R["per_segment"] = per_seg
        if train_iv is not None:
            R["training_era_span_gps"] = [float(train_iv[:, 0].min()),
                                          float(train_iv[:, 1].max())]
            n_out = sum(1 for p in per_seg if not p["inside_training_span"])
            effs_sorted = sorted([p["eff_pooled"] for p in per_seg
                                  if p["eff_pooled"] is not None])
            lines.append(f"  [O3a breakout] per-segment pooled eff "
                         f"distribution ({len(effs_sorted)} segs): "
                         + " ".join(f"{e:.3f}" for e in effs_sorted))
            lines.append(f"  [O3a breakout] random segs outside the "
                         f"training-era span: {n_out}/{len(per_seg)} "
                         f"(span = {R['training_era_span_gps']})")
        rep["runs"][run] = R
        lines.append("")
    json.dump(rep, open(f"{HERE}/pilot3d_report.json", "w"), indent=1)
    print("\n".join(lines), flush=True)
    print(f"[pilot3d] -> {HERE}/pilot3d_report.json", flush=True)


if __name__ == "__main__":
    main()
