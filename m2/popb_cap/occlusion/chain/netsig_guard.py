"""Net-sigma glitch guard, POST-HOC level (2026-10-11). Rule, written blind for every net-sigma-channel row:
   keep iff ( Grad-CAM CNN pass [max(hm,lm) under the production crop >= 0.5]  OR  loglr > 0 )  AND  cnn_lm > 0.5
Grad-CAM scores come from the production nf275 blindscan_p205 trigger at the same GPS (same trigger, same tile, different crop).
Foreground: filters search_out_<run>_m2_gx_off5_crop/detections_v2.json -> detections_v2_nsguard.json (+ report). FARs are NOT recomputed
(survivors keep their crop _v2 FAR = conservative). Background: fraction of the FAR bank rows that pass the coincident-crop CNN gate and the
LM floor but fail the rule (join of the crop and production S3 banks by row key) = indicative FAR-setting-background reduction.
Usage: python -I netsig_guard.py o3b o4a o4b [o3a]"""
import sys, json, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M=MADGRAV_SCRATCH + "/madgrav2"; S=MADGRAV_SCRATCH + "/madgrav2_spec/s3_crop"; THR=0.5; LMF=0.0382
def rule(gpass,loglr,lm): return (gpass or loglr>0) and lm>0.5
tot_keep=tot_rm=0
for run in sys.argv[1:]:
    out=f"{M}/search_out_{run}_m2_gx_off5_crop"; det=json.load(open(f"{out}/detections_v2.json"))
    P=json.load(open(f"{M}/search_out_{run}_m2_gx_off5_nf275/blindscan_p205.json"))["triggers"]; pg=np.array([t["gps"] for t in P])
    _T=json.load(open(f"{out}/blindscan_p205.json"))["triggers"]; CROP_T={run:(_T,np.array([t["gps"] for t in _T]))}
    keep=[]; rep=[f"[nsguard] {run}: crop _v2 {len(det)} rows; rule (gradcam-pass OR loglr>0) AND lm>0.5 on net-sigma rows"]
    for d in sorted(det,key=lambda x:x["best_far"]):
        if d.get("channel")!="net-sigma": keep.append(d); continue
        j=int(np.argmin(np.abs(pg-d["gps"]))); p=P[j] if abs(pg[j]-d["gps"])<1.5 else None
        gpass=(max(p["cnn_hm"],p["cnn_lm"])>=THR) if p else None
        Tc=CROP_T.get(run); jc=int(np.argmin(np.abs(Tc[1]-d["gps"]))); raw=Tc[0][jc]["loglr"] if abs(Tc[1][jc]-d["gps"])<1.5 else d["loglr"]   # RAW arm-stage loglr (the _v2 field holds the ranker stat for S3-admitted rows)
        ok=rule(bool(gpass),raw,d["cnn_lm"]) if p else True
        d["nsguard"]=dict(gradcam_hm=None if not p else round(p["cnn_hm"],3),gradcam_lm=None if not p else round(p["cnn_lm"],3),gradcam_pass=gpass,loglr_raw=round(raw,2),loglr_v2field=round(d["loglr"],2),lm=round(d["cnn_lm"],3),keep=ok)
        rep.append(f"  {(d.get('matches_known') or '** NON-CATALOGUE **'):22s} {d['gps']:12.1f} far {d['best_far']:.3f} | gradcam {('%.2f/%.2f'%(p['cnn_hm'],p['cnn_lm'])) if p else 'no trigger'} loglr_raw {raw:6.2f} lm {d['cnn_lm']:.2f} -> {'KEEP' if ok else 'REMOVE'}")
        if ok: keep.append(d)
    rm=[d for d in det if d not in keep]; cat_rm=[d["matches_known"] for d in rm if d.get("matches_known")]
    rep.append(f"[nsguard] {run}: removed {len(rm)} (catalogue among them {len(cat_rm)} {cat_rm}); survivors {len(keep)} = catalogue {sum(1 for d in keep if d.get('matches_known'))} + non-catalogue {sum(1 for d in keep if not d.get('matches_known'))}")
    # background side
    try:
        z=np.load(f"{S}/s3_{run}.npz",allow_pickle=True); q=np.load(f"{M}/s3_off5/s3bank/s3_{run}.npz",allow_pickle=True)
        kq={k:i for i,k in enumerate(zip(q["H1_seg"],q["iH"],q["L1_seg"],q["iL"],q["kind"]))}
        idx=[(i,kq.get(k)) for i,k in enumerate(zip(z["H1_seg"],z["iH"],z["L1_seg"],z["iL"],z["kind"])) if z["kind"][i]=="bank"]
        both=[(i,j) for i,j in idx if j is not None]; iz=np.array([i for i,_ in both]); jq=np.array([j for _,j in both])
        cpass=(np.maximum(z["cnn_hm"][iz],z["cnn_lm"][iz])>=THR)&(z["cnn_lm"][iz]>LMF)       # rows open to the net-sigma channel under the crop
        gpass=np.maximum(q["cnn_hm"][jq],q["cnn_lm"][jq])>=THR
        ok=(gpass|(z["loglr"][iz]>0))&(z["cnn_lm"][iz]>0.5)
        n=int(cpass.sum()); r=int((cpass&~ok).sum()); r1=int((cpass&~(gpass|(z["loglr"][iz]>0))).sum()); r2=int((cpass&~(z["cnn_lm"][iz]>0.5)).sum())
        rep.append(f"[nsguard] {run} background (bank rows in both banks {len(both)}): net-sigma-open under crop {n}; rule removes {r} ({100*r/max(n,1):.1f}%) [clause1 gradcam|loglr {r1}, clause2 lm>0.5 {r2}]")
    except Exception as e: rep.append(f"[nsguard] {run} background join failed: {e}")
    json.dump(keep,open(f"{out}/detections_v2_nsguard.json","w"),indent=1); open(f"{MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/NSGUARD_{run}.txt","w").write("\n".join(rep)+"\n")
    print("\n".join(rep)); tot_keep+=sum(1 for d in keep if d.get("matches_known")); tot_rm+=len(rm)
print(f"\n[nsguard] TOTAL catalogue survivors {tot_keep}; rows removed {tot_rm}")
