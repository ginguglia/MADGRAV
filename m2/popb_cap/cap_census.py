"""Population B, cap census (panel D1, 2026-10-08): slid-background L1 families at the loglr cap, by gate status, per run (folds pooled).
A gate-bypass rule 'loglr >= L and gates fail -> admit on the ungated FAR' pays the families above L; this prints N_fam(>=L) and
N_fam/livetime for L in the cap region, split by gate status (CNN hm/lm < 0.5; S3 < thr from the s3_<run>.npz bank scores)."""
import json, sys, numpy as np, time
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH + "/madgrav2"; THR=json.load(open(MADGRAV_ROOT + "/m2/rank/s3_threshold.json"))["threshold"]; t0=time.time()
LS=(22.0,22.5,22.9,22.95,22.99,22.995)
for run in sys.argv[1:]:
    d=f"{SC}/search_out_{run}_m2_gx_off5hw" if run.startswith("o3") else f"{SC}/search_out_{run}_m2_gx_off5"
    B=json.load(open(f"{d}/blindscan_v2.json")); live=B["far_live_yr"]["0"]+B["far_live_yr"]["1"]
    S=json.load(open(f"{d}/survivors_bg_v2.json"))["survivors"]; print(f"[{run}] {len(S)} bg survivors loaded ({time.time()-t0:.0f}s); livetime pooled {live:.0f} slide-yr",flush=True)
    z=np.load(f"{MADGRAV_SCRATCH}/madgrav2_spec/s3/s3_{run}.npz",allow_pickle=True)
    s3={(str(a),int(b),str(c),int(e)):float(v) for a,b,c,e,v,k in zip(z["H1_seg"],z["iH"],z["L1_seg"],z["iL"],z["s3"],z["kind"]) if k=="bank"}
    rows=[(r["loglr"],(r["L1_seg"],int(r["gps_L"]//4)),max(r.get("cnn_hm") or 0,r.get("cnn_lm") or 0),s3.get((r["H1_seg"],int(r["iH"]),r["L1_seg"],int(r["iL"])))) for r in S if r.get("loglr") is not None and r["loglr"]>=LS[0]]
    print(f"[{run}] rows with loglr>={LS[0]}: {len(rows)}; with S3 score: {sum(1 for r in rows if r[3] is not None)}")
    print(f"  {'L':>7s} {'fam_all':>8s} {'FAR_all/yr':>10s} {'fam_gatefail':>12s} {'FAR_gf/yr':>9s} {'fam_cnnfail_s3pass':>18s} {'fam_cnnpass':>11s} {'fam_noS3(cnnfail)':>17s}")
    for L in LS:
        fam={}; 
        for ll,k,cnn,s in rows:
            if ll<L: continue
            st="cnnpass" if cnn>=0.5 else ("noS3" if s is None else ("s3pass" if s>=THR else "gatefail"))
            # a family counts in the WORST (most signal-like) status among its pairs: cnnpass > s3pass > noS3 > gatefail
            o={"cnnpass":3,"s3pass":2,"noS3":1,"gatefail":0}; fam[k]=max(fam.get(k,-1),o[st])
        n=len(fam); c=np.bincount(np.array(list(fam.values()),int),minlength=4) if n else np.zeros(4,int)
        print(f"  {L:7.3f} {n:8d} {n/live:10.3f} {c[0]:12d} {c[0]/live:9.3f} {c[2]:18d} {c[3]:11d} {c[1]:17d}")
    ll=np.array([r[0] for r in rows]); print(f"  loglr distribution of bg rows >=22: p50 {np.median(ll):.3f} p90 {np.quantile(ll,0.9):.3f} max {ll.max():.4f}; rows >=22.99: {(ll>=22.99).sum()} ({(ll>=22.99).mean():.1%})",flush=True)
    # ---- panel 2026-10-08: fine grid 22.950-23.000 step 0.001, GATE-FAILING family counting, L_cap at 0.2/yr (and 0.25, 0.1) ----
    fam_gf={}; fam_all={}
    for ll,k,cnn,s in rows:
        if ll<22.95: continue
        fam_all[k]=max(fam_all.get(k,-9),ll)
        if cnn<0.5 and (s is None or s<THR): fam_gf[k]=max(fam_gf.get(k,-9),ll)   # noS3 counted as gate-failing (conservative)
    import numpy as _np
    la=_np.array(list(fam_all.values())); lg=_np.array(list(fam_gf.values())); grid=_np.round(_np.arange(22.950,23.0005,0.001),3)
    print(f"  fine grid (gate-failing families incl. unscored-S3 / all families) per L, FAR/yr pooled:")
    for L in grid:
        ng=int((lg>=L-1e-9).sum()); na=int((la>=L-1e-9).sum())
        if L in (22.95,22.96,22.97,22.98,22.985,22.99,22.995,23.0) or ng/live<=0.25: print(f"    L={L:.3f}: gate-fail {ng:6d} ({ng/live:7.3f}/yr)  all {na:6d} ({na/live:7.3f}/yr)")
    for f_ in (0.25,0.2,0.1):
        Lc=next((float(L) for L in grid if (lg>=L-1e-9).sum()/live<=f_),None); print(f"  L_cap(gate-fail counting, {f_}/yr) = {Lc}")
