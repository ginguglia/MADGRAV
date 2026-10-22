"""Fetch the GWOSC hardware-injection flags (NO_DETCHAR/BURST/STOCH/CBC_HW_INJ) and DATA for one run and write
dq/hwinj_flags_<run>.json + the union of injection windows inside coincident DATA, dq/hwinj_windows_<run>.json.
Usage: python fetch_hwinj_flags.py <run> [<dataset> <gps_start> <gps_end>]   (defaults for o3a/o3b/o4a/o4b below)
Login node (network). 2026-10-05."""
import sys, json, time
from gwosc.api import fetch_json
RUNS={"o3a":("O3a_4KHZ_R1",1238166018,1253977218),"o3b":("O3b_4KHZ_R1",1256655618,1269363618),
      "o4a":("O4a_4KHZ_R1",1368975618,1389456018),"o4b":("O4b_4KHZ_R1",1396796418,1422118818)}
FL=["NO_DETCHAR_HW_INJ","NO_BURST_HW_INJ","NO_STOCH_HW_INJ","NO_CBC_HW_INJ","DATA"]
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

DQ=MADGRAV_SCRATCH + "/madgrav2/dq"
def inv(active,span):
    out=[]; cur=span[0]
    for a,b in sorted(active):
        if a>cur: out.append([cur,a])
        cur=max(cur,b)
    if cur<span[1]: out.append([cur,span[1]])
    return out
def inter(A,B):
    out=[]; i=j=0; A=sorted(A); B=sorted(B)
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]); hi=min(A[i][1],B[j][1])
        if lo<hi: out.append([lo,hi])
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return out
def union(S):
    m=[]
    for a,b in sorted(S):
        if m and a<=m[-1][1]: m[-1][1]=max(m[-1][1],b)
        else: m.append([a,b])
    return m
def main():
    run=sys.argv[1]; ds,a,b=RUNS[run] if len(sys.argv)<5 else (sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    out={}
    for ifo in ("H1","L1"):
        for fl in FL:
            segs=None
            for k in range(6):
                try: segs=fetch_json(f"https://gwosc.org/timeline/segments/json/{ds}/{ifo}_{fl}/{a}/{b-a}/").get("segments",[]); break
                except Exception as e: print(run,ifo,fl,"retry",k,str(e)[:80],flush=True); time.sleep(5)
            if segs is None: raise SystemExit(f"FAILED {ifo}_{fl}")
            out[f"{ifo}_{fl}"]=segs; print(f"{run} {ifo} {fl:18s} n={len(segs)} active_total={sum(s[1]-s[0] for s in segs)}",flush=True)
    json.dump(dict(dataset=ds,span=[a,b],flags=out),open(f"{DQ}/hwinj_flags_{run}.json","w"))
    data=inter(out["H1_DATA"],out["L1_DATA"]); bad=[]
    for ifo in ("H1","L1"):
        for fl in FL[:4]: bad+=inter(inv(out[f"{ifo}_{fl}"],[a,b]),data)
    W=union(bad); tot=lambda S: sum(y-x for x,y in S)
    print(f"{run}: HW-inj windows inside coincident DATA: {len(W)} windows, {tot(W)/3600:.2f} h = {100*tot(W)/max(tot(data),1):.3f}%")
    json.dump(dict(run=run,hwinj_windows_union=W,flags=FL[:4],source=f"{DQ}/hwinj_flags_{run}.json"),open(f"{DQ}/hwinj_windows_{run}.json","w"))
if __name__=="__main__": main()
