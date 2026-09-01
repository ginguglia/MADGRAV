#!/usr/bin/env python
"""successor_foreground.py -- Sec. 7.1 of the successor-statistic amendment (2026-08-18): foreground FARs.

  python successor_foreground.py prelim  [runs]   (CPU) every as-run CNN-scored candidate of blindscan.json ->
        successor N_c / FAR / UL against the veto-symmetric background of its fold (narrowed exclusion), frozen
        N_eff (neff_freeze.json, md5 asserted) -> details/successor_statistic/fg_prelim_<run>.json + the list of
        candidates that need the (unchanged) foreground local-ASD veto (all with FAR<1/yr or UL90<1/yr) ->
        details/successor_statistic/fg_veto_request_<run>.json
  [GPU job: successor_fg_veto.py recomputes asd_consistency.recompute_local for the requested candidates]
  python successor_foreground.py final   [runs]   (CPU) apply the veto rule (channel = successor channel), detection
        = FAR<1/yr (AND-UL variant reported), catalog cross-match (post-FAR) -> search_out_<run>_far*/detections_successor.json
        (NEW file) + figures/catalog_o3o4/madgrav_far_successor.csv (NEW file; as-run files untouched).
"""
import os, sys, json, csv, glob, hashlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S
from successor_neff import neff_of
import os as _os
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

DET = os.environ.get("SUCC_DET", S.DET); MG = S.MG
OUTDIR = os.environ.get("SUCC_OUT_DIR")   # dry-run redirection of detections_successor.json / csv (default: real locations)
CATDIR = MADGRAV_EXTDATA + "/gwosc_eventapi"
MERGED = f"{MG}/figures/catalog_o3o4/merged_plot_v2.csv"
CSV_ASRUN = f"{MG}/figures/catalog_o3o4/madgrav_far_final.csv"
DET_FAR = 1.0; MATCH_S = 2.0
AT_RISK = ["GW190828_063405", "GW190521_074359", "GW190513_205428", "GW190706_222641", "GW190727_060333", "GW190519_153544",
           "GW190408_181802", "GW190602_175927", "GW230922_040658", "GW230824_033047", "GW240514_121713", "GW241130_034908", "GW250108_152221"]


def freeze():
    p = f"{DET}/neff_freeze.json"; fz = json.load(open(p)); fz["_md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return fz


def catalog():
    """name -> list of GPS; API jsons + name-derived GPS for merged_plot_v2 names (as-run CSV: all 44 within 0.5 s)."""
    from gwpy.time import to_gps
    cat = {}
    for f in glob.glob(f"{CATDIR}/GWTC*.json"):
        d = json.load(open(f)); ev = d.get("events", d)
        for k, v in ev.items():
            nm = v["commonName"]; nm = nm if nm.startswith("GW") else "GW" + nm
            cat.setdefault(nm, {}).setdefault("gps", set()).add(float(v["GPS"]))
            if v.get("total_mass_source") is not None: cat[nm]["mtot"] = v["total_mass_source"]
    mp = {}
    for r in csv.DictReader(open(MERGED)):
        mp[r["name"]] = r
        try:
            s = r["name"][2:]; g = float(to_gps(f"20{s[:2]}-{s[2:4]}-{s[4:6]} {s[7:9]}:{s[9:11]}:{s[11:13]}"))
            cat.setdefault(r["name"], {}).setdefault("gps", set()).add(g)
        except Exception: pass
    for nm, r in mp.items():
        if r.get("total_mass_source", "").strip(): cat.setdefault(nm, {})["mtot"] = float(r["total_mass_source"])
        cat.setdefault(nm, {})["cwb"] = r.get("cwb_detected", "")
    return cat


def match(cat, gps):
    best = None
    for nm, v in cat.items():
        for g in v.get("gps", ()):
            dt = abs(g - gps)
            if dt <= MATCH_S and (best is None or dt < best[1]): best = (nm, dt)
    return best


def cmd_prelim(runs):
    S.assert_spec(); md5 = S.module_md5(); fz = freeze()
    st = {r: S.self_test(r, (f"{S.DET}/bg_veto_{r.lower()}.npz" if os.path.exists(f"{S.DET}/bg_veto_{r.lower()}.npz") else None), n_sample=150, n_brute=30, log=False) for r in runs}
    json.dump(dict(module_md5=md5, neff_freeze_md5=fz["_md5"], selftest=st), open(f"{DET}/successor_stat_selftest_S5.json", "w"), indent=1)
    assert all(st[r]["PASS"] for r in runs), "self-test FAILED at foreground stage -> STOP"
    print(f"[fg] module md5 {md5}; neff_freeze md5 {fz['_md5']}; self-test PASS on {runs}", flush=True)
    for run in runs:
        vp = f"{S.DET}/bg_veto_{run.lower()}.npz"; vp = vp if os.path.exists(vp) else None; bg = S.Background(run, vp)
        b = json.load(open(f"{S.RUNS[run]}/blindscan.json")); trig = b["triggers"]
        cands = [t for t in trig if t.get("cnn_hm") is not None]           # CNN-scored as-run candidates
        out = []; req = []
        for t in cands:
            f = int(t["fold"]); ne = neff_of(fz["runs"], run, f)
            r = bg.score_candidate(dict(seg=t["seg"], gps=t["gps"], loglr=t["loglr"], net=t["net"], cnn_hm=t["cnn_hm"], cnn_lm=t["cnn_lm"], fold=f), neff=ne)
            asr = [c for c in (t.get("far_lr_perarm"), t.get("far_net_perarm")) if c is not None]
            far_asrun = (len(asr) * min(asr)) if asr else None
            rec = dict(seg=t["seg"], gps=t["gps"], idx=t.get("idx"), fold=f, net=t["net"], loglr=t["loglr"], cnn_hm=t["cnn_hm"], cnn_lm=t["cnn_lm"],
                       is_glitch=bool(t.get("is_glitch")), kept_by=t.get("kept_by"), matches_known_asrun=t.get("matches_known", ""),
                       far_asrun=far_asrun, channel_asrun=(None if not asr else ("net-sigma" if (t.get("far_net_perarm") is not None and (t.get("far_lr_perarm") is None or t["far_net_perarm"] < t["far_lr_perarm"])) else "loglr")),
                       N_lr=r["N_lr"], N_net=r["N_net"], n_excluded_pairs=r["n_excluded_pairs"], channel=r["channel"], N_min=r["N_min"],
                       N_eff=r["N_eff"], T_f=r["T"], far=r["far"], ul90=r["ul90"], far_lr=r["far_lr"], far_net=r["far_net"],
                       above_M=(r["N_min"] is not None and r["N_min"] > S.top_M(r["T"])))
            out.append(rec)
            if (not rec["is_glitch"]) and rec["far"] is not None and (rec["far"] < DET_FAR or rec["ul90"] < DET_FAR):
                req.append(dict(seg=t["seg"], gps=t["gps"], idx=t.get("idx"), channel=r["channel"], net=t["net"], cnn_hm=t["cnn_hm"], cnn_lm=t["cnn_lm"]))
        json.dump(dict(run=run, module_md5=md5, neff_freeze_md5=fz["_md5"], veto_path=vp, n_candidates=len(cands), candidates=out), open(f"{DET}/fg_prelim_{run.lower()}.json", "w"), indent=1)
        json.dump(req, open(f"{DET}/fg_veto_request_{run.lower()}.json", "w"), indent=1)
        nd = sum(1 for x in out if (not x["is_glitch"]) and x["far"] is not None and x["far"] < DET_FAR)
        print(f"[fg:{run}] {len(cands)} candidates scored; {nd} with successor FAR<1/yr (pre-veto); {len(req)} veto recomputations requested -> fg_veto_request_{run.lower()}.json", flush=True)


def cmd_final(runs):
    S.assert_spec(); md5 = S.module_md5(); fz = freeze(); cat = catalog()
    asrun_csv = {(r["run"], r["name"]): r for r in csv.DictReader(open(CSV_ASRUN))}
    rows = []; summary = {}
    for run in runs:
        P = json.load(open(f"{DET}/fg_prelim_{run.lower()}.json")); assert P["neff_freeze_md5"] == fz["_md5"]
        V = {}
        vf = f"{DET}/fg_veto_{run.lower()}.json"
        if os.path.exists(vf):
            for v in json.load(open(vf)): V[(v["seg"], round(v["gps"], 1))] = v
        dets = []; n_and = 0
        for c in P["candidates"]:
            c["far_lt1"] = bool(c["far"] is not None and c["far"] < DET_FAR); c["ul_lt1"] = bool(c["ul90"] is not None and c["ul90"] < DET_FAR)
            if c["is_glitch"] or c["far"] is None or not (c["far_lt1"] or c["ul_lt1"]): continue
            v = V.get((c["seg"], round(c["gps"], 1)))
            if v is None:
                raise SystemExit(f"[fg:{run}] {c['seg']} {c['gps']}: foreground veto not computed (fg_veto_{run.lower()}.json incomplete) -> STOP, nothing written")
            keep = True; reason = "keep"
            if max(v["hm_loc"], v["lm_loc"]) < S.GATE: keep = False; reason = "cnn"
            elif c["channel"] == "net-sigma" and v["net_loc"] < S.NET_FLOOR: keep = False; reason = "net<floor"
            c["asd_veto"] = dict(net_loc=v["net_loc"], hm_loc=v["hm_loc"], lm_loc=v["lm_loc"], keep=keep, reason=reason, source=v.get("source", "recomputed"))
            c["veto_keep"] = keep
            m = match(cat, c["gps"]); c["matches_known"] = m[0] if m else ""; c["match_dt_s"] = m[1] if m else None
            if keep and c["far_lt1"]:
                c["detection"] = True; c["detection_and_ul"] = bool(c["ul_lt1"]); n_and += int(c["ul_lt1"]); dets.append(c)
        dets.sort(key=lambda d: d["far"])
        outp = f"{OUTDIR or S.RUNS[run]}/detections_successor{'_'+run.lower() if OUTDIR else ''}.json"
        assert not os.path.exists(outp) or os.environ.get("SUCC_OVERWRITE") == "1"
        json.dump(dict(run=run, statistic="successor (unconditioned scalar rank, veto-symmetric bg, narrowed exclusion, measured N_eff)", module_md5=md5,
                       neff_freeze_md5=fz["_md5"], spec_md5=S.SPEC_MD5, det_far=DET_FAR, n_detections=len(dets), n_detections_and_ul=n_and,
                       detections=[{k: v for k, v in d.items()} for d in dets]), open(outp, "w"), indent=1)
        json.dump(P, open(f"{DET}/fg_final_{run.lower()}.json", "w"), indent=1)
        # CSV rows
        for d in dets:
            f = d["fold"]; T = d["T_f"]; N = d["N_min"]; ne = d["N_eff"]; lo, hi = S.garwood(N)
            fe = fz["runs"][run][str(f)]; ci = fe["ci90"]
            name = d["matches_known"] or f"{d['seg']}@{d['gps']:.0f}"
            ar = asrun_csv.get((run, name))
            rows.append(dict(run=run, name=name, net=d["net"], loglr=d["loglr"], channel=d["channel"], N_bg=N, livetime_yr=round(T, 3), trials=ne,
                             far=d["far"], far_lo90=ne * lo / T, far_hi90=ne * hi / T, far_ul90=d["ul90"], ifar=(1 / d["far"] if d["far"] > 0 else ""),
                             mtot=(cat.get(name, {}).get("mtot", "") if name in cat else ""), cwb=(cat.get(name, {}).get("cwb", "") if name in cat else ""),
                             N_eff=ne, neff_lo=ci[0], neff_hi=ci[1], far_asrun=(ar["far"] if ar else (d["far_asrun"] if d["far_asrun"] is not None else "")),
                             detection_and_ul=d["detection_and_ul"], gps=d["gps"], seg=d["seg"], fold=f))
        summary[run] = dict(n_det=len(dets), n_det_and_ul=n_and, names=[d["matches_known"] or d["seg"] for d in dets])
        print(f"[fg:{run}] successor detections (FAR<1/yr, veto applied): {len(dets)} (AND UL90<1: {n_and}) -> {outp}", flush=True)
    outc = f"{OUTDIR or MG+'/figures/catalog_o3o4'}/madgrav_far_successor{'_o4ars' if set(runs) == {'O4ars'} else ''}.csv"
    assert not os.path.exists(outc) or os.environ.get("SUCC_OVERWRITE") == "1"
    cols = ["run", "name", "net", "loglr", "channel", "N_bg", "livetime_yr", "trials", "far", "far_lo90", "far_hi90", "far_ul90", "ifar", "mtot", "cwb", "N_eff", "neff_lo", "neff_hi", "far_asrun", "detection_and_ul", "gps", "seg", "fold"]
    with open(outc, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    json.dump(summary, open(f"{DET}/fg_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)); print(f"-> {outc}")


if __name__ == "__main__":
    what = sys.argv[1]; runs = sys.argv[2:] or S.MAIN_RUNS
    {"prelim": cmd_prelim, "final": cmd_final}[what](runs)
