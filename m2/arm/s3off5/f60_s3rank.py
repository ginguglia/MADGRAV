"""Mains-line (f60) veto on the _s3rank detection list of one run (2026-10-05, Decision: adopt the rule).

RULE (madgrav2-mains-line-cut, 2026-09-21): reject a candidate if max(H1,L1) fraction of power in 59-61 Hz
exceeds SM_F60_MAX (0.10 = injection p99.5) in the pipeline's own representation, the re-whitened coherence
series (20-500 Hz, 1 s, series_for from the run's series cache) -- identical to mains_inj.py / mains_det_run.py,
where the 0.357 % injection cost was measured. The gwpy strain-whitened fraction (apply_f60_veto.py) is
recorded alongside for reference only.
Reads  $SM_OUT/detections_s3rank.json + blindscan_s3rank.json
Writes $SM_OUT/f60_report_s3rank.json (every row, both representations) and detections_s3rank_f60.json (survivors)."""
import os, sys, json, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ROOT=MADGRAV_ROOT; sys.path.insert(0,f"{ROOT}/search_mode")
import driver_search_multi as M
SC=MADGRAV_SCRATCH; FS=4096.0
RUN=os.environ["SM_RUN"]; OUT=os.environ["SM_OUT"]; FMAX=float(os.environ.get("SM_F60_MAX","0.10"))
TAG=os.environ.get("SM_DET_TAG","_s3rank")
def f60(x):
    x=np.asarray(x,float); P=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(len(x),1/FS)
    k=(f>=20)&(f<=500); f,P=f[k],P[k]
    return float(P[(f>=59)&(f<=61)].sum()/P.sum()), float(f[int(P.argmax())])
def f60_strain(seg,dt,gps):
    try:
        from gwpy.timeseries import TimeSeries
    except Exception: return None,None
    z=np.load(f"{SC}/strain_{RUN}_full/{seg}_{dt}.npz"); s=z["strain"]; t0=float(z["gps_start"]); fs=int(z["fs"])
    i0=max(0,int((gps-t0-32)*fs)); i1=min(len(s),int((gps-t0+32)*fs))
    w=TimeSeries(s[i0:i1].astype(np.float64),sample_rate=fs,t0=t0+i0/fs).whiten(fftlength=4,overlap=2).bandpass(20,500)
    tt=w.times.value-gps; m=np.abs(tt)<=0.100; x=w.value[m]*np.hanning(int(m.sum()))
    P=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(len(x),1/fs); k=(f>=20)&(f<=500); f,P=f[k],P[k]
    return float(P[(f>=59)&(f<=61)].sum()/P.sum()), float(f[int(P.argmax())])
det=json.load(open(f"{OUT}/detections{TAG}.json"))
T={round(t["gps"]):t for t in json.load(open(f"{OUT}/blindscan{TAG}.json"))["triggers"]}
rows=[]; keep=[]
print(f"[f60] {RUN} detections{TAG}: {len(det)} rows, rule max(H1,L1) f60 > {FMAX} (coherence-series representation)\n")
print(f"{'event':24}{'gps':>12}{'chan':>10}{'far':>8} | {'cohH':>7}{'cohL':>7}{'pkH':>7}{'pkL':>7} | {'strH':>7}{'strL':>7}   verdict")
for r in sorted(det,key=lambda x:x.get("best_far",9e9)):
    t=T.get(round(r["gps"]))
    if t is None: print(f"{'(no trigger row)':24}{r['gps']:12.0f}"); keep.append(r); continue
    j=t["idx"]; seg=r["seg"]
    sh=M.series_for(seg,"H1",[j]); sl=M.series_for(seg,"L1",[j])
    a,pa=f60(sh[j]); b,pb=f60(sl[j]); worst=max(a,b); bad=worst>FMAX
    sa,_=f60_strain(seg,"H1",r["gps"]); sb,_=f60_strain(seg,"L1",r["gps"])
    nm=r.get("matches_known") or "** NON-CATALOGUE **"
    fs_=lambda v: f"{100*v:6.1f}%" if v is not None else "    n/a"
    print(f"{nm:24}{r['gps']:12.0f}{r.get('channel','?'):>10}{r.get('best_far',float('nan')):8.3f} | {100*a:6.1f}%{100*b:6.1f}%{pa:7.1f}{pb:7.1f} | {fs_(sa)}{fs_(sb)}   {'** VETOED (mains) **' if bad else 'keep'}")
    rows.append(dict(name=r.get("matches_known"),gps=r["gps"],seg=seg,idx=int(j),channel=r.get("channel"),best_far=r.get("best_far"),
                     f60H=a,f60L=b,pkH=pa,pkL=pb,f60=worst,strainH=sa,strainL=sb,vetoed=bool(bad)))
    if bad: r=dict(r,f60_vetoed=True,f60=worst)
    else: keep.append(r)
nv=sum(1 for x in rows if x["vetoed"]); ncat=sum(1 for x in rows if x["vetoed"] and x["name"])
cat=[x["f60"] for x in rows if x["name"]]
print(f"\n[f60] vetoed {nv} of {len(rows)}; catalogue among them {ncat} (MUST be 0); catalogue max f60 = {100*max(cat):.2f}%" if cat else "")
print(f"[f60] survivors {len(keep)} = catalogue {sum(1 for r in keep if r.get('matches_known'))} + non-catalogue {sum(1 for r in keep if not r.get('matches_known'))}")
json.dump(dict(rule=f"max(H1,L1) f60 > {FMAX} on re-whitened coherence series 20-500 Hz",f60_max=FMAX,rows=rows),open(f"{OUT}/f60_report{TAG}.json","w"),indent=1)
json.dump(keep,open(f"{OUT}/detections{TAG}_f60.json","w"),indent=1)
print(f"[f60] wrote {OUT}/f60_report{TAG}.json and detections{TAG}_f60.json")
