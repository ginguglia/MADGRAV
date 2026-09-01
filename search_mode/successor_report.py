#!/usr/bin/env python
"""successor_report.py -- S7 completion report (details/successor_statistic/COMPLETION_REPORT.md) of the
successor-statistic amendment (2026-08-18). Reads only files written by S0-S6. Also usable for the STOP report."""
import os, sys, json, csv, glob, hashlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S

DET = os.environ.get("SUCC_DET", S.DET); MG = S.MG; OUTDIR = os.environ.get("SUCC_OUT_DIR")
AT_RISK = ["GW190828_063405", "GW190521_074359", "GW190513_205428", "GW190706_222641", "GW190727_060333", "GW190519_153544",
           "GW190408_181802", "GW190602_175927", "GW230922_040658", "GW230824_033047", "GW240514_121713", "GW241130_034908", "GW250108_152221"]


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "n/a"
def rd(p): return open(p).read() if os.path.exists(p) else ""


def main(stop=False):
    L = []; P = L.append
    P(f"# MADGRAV successor-statistic amendment — {'STOP REPORT (acceptance gate FAILED)' if stop else 'COMPLETION REPORT'}")
    P(f"Generated {time.strftime('%F %T')} on VSC login node. Spec PREREG_AMENDMENT_successor_statistic.md md5 {md5(f'{MG}/PREREG_AMENDMENT_successor_statistic.md')} (required {S.SPEC_MD5}).")
    P(f"Shared module search_mode/successor_stat.py md5 {S.module_md5()}; neff_freeze.json md5 {md5(f'{DET}/neff_freeze.json')}.")
    P("")
    P("## Stage log"); P("```"); P(rd(f"{DET}/STAGE_LOG.md").strip()); P("```"); P("")
    P("## S0 — shared module + self-test"); 
    for tag in ("S0", "S5"):
        p = f"{DET}/successor_stat_selftest_{tag}.json"
        if os.path.exists(p):
            j = json.load(open(p)); st = j.get("selftest", j)
            P(f"- {tag}: " + "; ".join(f"{r}: {'PASS' if st[r]['PASS'] else 'FAIL'} (as-run per-arm counts {st[r]['asrun_repro']['n']-st[r]['asrun_repro']['n_bad']}/{st[r]['asrun_repro']['n']}; entry-point mismatches {sum(v['n_entrypoint_mismatch'] for v in st[r]['folds'].values())}; brute mismatches {sum(v['n_brute_mismatch'] for v in st[r]['folds'].values())})" for r in st if isinstance(st[r], dict) and "PASS" in st[r]))
    P("")
    P("## S1 — veto cost pilot"); P("```"); P(rd(f"{DET}/veto_pilot_cost.txt").strip()); P("```"); P("")
    P("## S2 — background veto counts (top-M rep pairs per run/fold/channel)"); P("```"); P(rd(f"{DET}/bg_veto_counts.txt").strip()); P("```"); P("")
    P("## S3 — N_eff per run/fold (Garwood 90%, endpoint rule) and freeze decision"); P("```"); P(rd(f"{DET}/neff_table.txt").strip()); P("```")
    fz = json.load(open(f"{DET}/neff_freeze.json")) if os.path.exists(f"{DET}/neff_freeze.json") else None
    if fz:
        P("Frozen values (neff_freeze.json):")
        for r, d in fz["runs"].items():
            for f in ("0", "1"):
                e = d[f]; P(f"- {r} fold {f}: mode={e['mode']}, N_eff(1/yr)={e['N_eff']:.3f} [{e['ci90'][0]:.3f}-{e['ci90'][1]:.3f}], homogeneity p_chi2={e['p_homogeneity_chi2']:.3g} (p_mc={e['p_homogeneity_mc']:.3g})" + (f"; piecewise: " + ", ".join(f"{k}:{v:.2f}" for k, v in e['piecewise'].items() if v is not None) if e['mode'] == 'piecewise' else ""))
    P("")
    P("## S4 — acceptance (successor null calibration, in-fold + cross-fold)"); P("```"); P(rd(f"{DET}/null_calibration_successor.txt").strip()); P("```"); P("")
    if stop:
        P("## STOP: the cross-fold acceptance gate FAILED for at least one run. No foreground stage was run (Sec. 4)."); 
        open(f"{DET}/COMPLETION_REPORT.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); return "\n".join(L)
    # ---- S5 detection set
    P("## S5 — foreground: successor vs as-run detection sets")
    asrun = list(csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final.csv")))
    SCSV = f"{OUTDIR}/madgrav_far_successor.csv" if OUTDIR else f"{MG}/figures/catalog_o3o4/madgrav_far_successor.csv"
    succ = list(csv.DictReader(open(SCSV))) if os.path.exists(SCSV) else []
    A = {(r["run"], r["name"]): r for r in asrun}; B = {(r["run"], r["name"]): r for r in succ}
    P("| run | as-run dets | successor dets (FAR<1) | successor (FAR<1 AND UL90<1) | kept | lost | gained |"); P("|---|---|---|---|---|---|---|")
    tot = dict(a=0, s=0, su=0, k=0, l=0, g=0)
    for run in S.MAIN_RUNS:
        a = {k for k in A if k[0] == run}; b = {k for k in B if k[0] == run}; bu = {k for k in B if k[0] == run and B[k]["detection_and_ul"] == "True"}
        k, l, g = len(a & b), len(a - b), len(b - a)
        tot["a"] += len(a); tot["s"] += len(b); tot["su"] += len(bu); tot["k"] += k; tot["l"] += l; tot["g"] += g
        P(f"| {run} | {len(a)} | {len(b)} | {len(bu)} | {k} | {l} | {g} |")
    P(f"| total | {tot['a']} | {tot['s']} | {tot['su']} | {tot['k']} | {tot['l']} | {tot['g']} |"); P("")
    P("### Per-event table (union of as-run and successor detections); FAR in 1/yr; UL = N=0 (90% UL quoted)")
    P("| run | event | channel as-run | FAR as-run | channel succ | N_min | N_eff | FAR succ [90% CI] | UL90 succ | status |"); P("|---|---|---|---|---|---|---|---|---|---|")
    fgf = {r: json.load(open(f"{DET}/fg_final_{r.lower()}.json")) if os.path.exists(f"{DET}/fg_final_{r.lower()}.json") else None for r in S.MAIN_RUNS}
    def find_cand(run, name, gps=None):
        if not fgf[run]: return None
        for c in fgf[run]["candidates"]:
            if c.get("matches_known") == name or (gps is not None and abs(c["gps"] - gps) < 2.5): return c
        return None
    lost_rows = []
    for run in S.MAIN_RUNS:
        keys = sorted({k for k in list(A) + list(B) if k[0] == run}, key=lambda k: float((B.get(k) or A.get(k))["far"] or 0))
        for k in keys:
            a = A.get(k); b = B.get(k)
            if b:
                st = "kept" if a else "GAINED"
                P(f"| {run} | {k[1]} | {a['channel'] if a else '-'} | {float(a['far']):.3g} | {b['channel']} | {b['N_bg']} | {float(b['N_eff']):.2f} | {float(b['far']):.3g} [{float(b['far_lo90']):.3g}-{float(b['far_hi90']):.3g}] | {float(b['far_ul90']):.3g} | {st}{' (N=0: UL only)' if int(float(b['N_bg']))==0 else ''}{'' if b['detection_and_ul']=='True' else ' (UL90>=1)'} |")
            else:
                # find its successor numbers among candidates
                gps = None
                for d in json.load(open(f"{S.RUNS[run]}/detections.json")):
                    if abs(d["loglr"] - float(a["loglr"])) < 1e-9: gps = d["gps"]
                c = find_cand(run, k[1], gps)
                if c:
                    why = "vetoed (local-ASD)" if c.get("veto_keep") is False else ("FAR>=1/yr" if not c.get("far_lt1") else "?")
                    P(f"| {run} | {k[1]} | {a['channel']} | {float(a['far']):.3g} | {c['channel']} | {c['N_min']} | {c['N_eff']:.2f} | {c['far']:.3g} | {c['ul90']:.3g} | LOST: {why} |")
                else:
                    P(f"| {run} | {k[1]} | {a['channel']} | {float(a['far']):.3g} | - | - | - | - | - | LOST (candidate not found) |")
    P("")
    P("### The 13 crude-rescale at-risk events (Sec. 7.4), individually")
    P("| event | run | as-run FAR | successor FAR | successor UL90 | channel | N_min | outcome |"); P("|---|---|---|---|---|---|---|---|")
    for nm in AT_RISK:
        ka = [k for k in A if k[1] == nm]; run = ka[0][0] if ka else "?"
        a = A.get((run, nm)); b = B.get((run, nm))
        if b: P(f"| {nm} | {run} | {float(a['far']):.3g} | {float(b['far']):.3g} | {float(b['far_ul90']):.3g} | {b['channel']} | {b['N_bg']} | DETECTED (kept){'' if b['detection_and_ul']=='True' else ' [UL90>=1]'} |")
        else:
            gps = None
            for d in json.load(open(f"{S.RUNS[run]}/detections.json")) if run != "?" else []:
                if a and abs(d["loglr"] - float(a["loglr"])) < 1e-9: gps = d["gps"]
            c = find_cand(run, nm, gps) if run != "?" else None
            if c: P(f"| {nm} | {run} | {float(a['far']):.3g} | {c['far']:.3g} | {c['ul90']:.3g} | {c['channel']} | {c['N_min']} | LOST ({'vetoed' if c.get('veto_keep') is False else 'FAR>=1/yr'}) |")
            else: P(f"| {nm} | {run} | {float(a['far']) if a else '-'} | - | - | - | - | not found |")
    P("")
    if os.path.exists(f"{S.RUNS['O4ars']}/detections_successor.json"):
        j = json.load(open(f"{S.RUNS['O4ars']}/detections_successor.json")); ad = json.load(open(f"{S.RUNS['O4ars']}/detections.json"))
        P(f"### O4ars (supplementary rescan): successor detections {j['n_detections']} (AND-UL {j['n_detections_and_ul']}) vs as-run {len(ad)}: " + ", ".join(f"{d.get('matches_known') or d['seg']} FAR {d['far']:.3g}" for d in j["detections"])); P("")
    P("### Novel (unmatched) successor detections, if any")
    nov = [r for r in succ if not r["name"].startswith("GW")]
    P("none" if not nov else "\n".join(f"- {r['run']} {r['name']} FAR {float(r['far']):.3g}/yr channel {r['channel']} N={r['N_bg']}" for r in nov)); P("")
    # ---- foreground veto comparison
    P("### Foreground local-ASD veto: recomputed values vs as-run (rule unchanged; channel = successor channel)")
    for run in S.MAIN_RUNS:
        if fgf[run]:
            cs = [c for c in fgf[run]["candidates"] if c.get("asd_veto")]
            nv = sum(1 for c in cs if not c["asd_veto"]["keep"])
            P(f"- {run}: {len(cs)} candidates with FAR<1 or UL90<1 recomputed; {nv} vetoed: " + ", ".join(f"{c.get('matches_known') or c['seg']}({c['asd_veto']['reason']})" for c in cs if not c["asd_veto"]["keep"]))
    P("")
    # ---- S6
    P("## S6 — injections / VT / p_astro under the successor statistic")
    pj = f"{OUTDIR or MG+'/search_mode/pastro_final'}/pastro_final_successor.json"
    if os.path.exists(pj):
        j = json.load(open(pj))
        for r, s in j["summary"].items(): P(f"- {r}: efficiency (w0-weighted, all injections) {s['eff']:.3f}; T_fg={s['T_fg']:.4f} yr; FGMC Ln={s['Ln']:.3f}, Ls={s['Ls']:.2f}; n_det={s['n_det']}")
        pa = [d["p_astro"] for d in j["detections"]]
        if pa: P(f"- p_astro over {len(pa)} successor detections: min {min(pa):.3f}, median {np.median(pa):.3f}, n>=0.9: {sum(1 for v in pa if v>=0.9)}; table pastro_final_successor.csv")
        P("```"); P(rd(pj.replace(".json", ".csv")).strip()); P("```")
    else: P("- PENDING: pastro_final_successor.json not produced (see driver log).")
    vj = f"{MG}/search_mode/pastro_final/vt_paper_numbers_successor.json"
    if os.path.exists(vj):
        v = json.load(open(vj)); P(f"- VT (comoving relabel path, successor inputs): " + json.dumps({k: v[k] for k in v if k in ('T_obs_yr', 'peak_o3a', 'peak_o3', 'peak_total', 'masked_bins')}))
        P(f"- figure figures/vt_search/vt_vs_mass_paper_successor.png; vt_relabel_comoving_successor.json")
    else: P("- VT: PENDING (vt_relabel_comoving_successor.py / fig_vt_paper_successor.py not completed; plan: run them on the login node, ~10 min, inputs inj_scored_<run>_successor.npz).")
    P("")
    P("## Files"); 
    for p in sorted(glob.glob(f"{DET}/*")): 
        if os.path.isfile(p): P(f"- {p} ({os.path.getsize(p)} B, md5 {md5(p)})")
    for p in [f"{S.RUNS[r]}/detections_successor.json" for r in S.RUNS] + [f"{MG}/figures/catalog_o3o4/madgrav_far_successor.csv", pj, f"{MG}/search_mode/pastro_final/pastro_final_successor.csv", vj]:
        if os.path.exists(p): P(f"- {p} (md5 {md5(p)})")
    txt = "\n".join(L) + "\n"; open(f"{DET}/COMPLETION_REPORT.md", "w").write(txt); print(txt); return txt


if __name__ == "__main__":
    main(stop=(len(sys.argv) > 1 and sys.argv[1] == "stop"))
