"""Veto mask v2 = existing segment mask (CBC+BURST CAT2 union with the GW200129 exception, dq/veto_mask_<run>.json)
INTERSECTED with the hardware-injection-free time (complement of dq/hwinj_windows_<run>.json). Writes
dq/veto_mask_<run>_hw.json in the same {segname: {good_segments: [[lo,hi],...]}} layout read by driver_streams_bg.py
(SM_VETO) and apply_veto_to_streams.py. The v1 file is left untouched so the existing passes stay reproducible.
Usage: python build_veto_mask_hw.py <run> [--write]      (2026-10-05)"""
import sys, json
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

DQ=MADGRAV_SCRATCH + "/madgrav2/dq"
def inter(A,B):
    out=[]; i=j=0; A=sorted(A); B=sorted(B)
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]); hi=min(A[i][1],B[j][1])
        if lo<hi: out.append([lo,hi])
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return out
def main():
    run=sys.argv[1]; write="--write" in sys.argv
    v1=json.load(open(f"{DQ}/veto_mask_{run}.json")); W=json.load(open(f"{DQ}/hwinj_windows_{run}.json"))["hwinj_windows_union"]
    v2={}; removed=0.0; nseg=0; touched=[]
    for name,v in v1.items():
        g=[[float(a),float(b)] for a,b in v["good_segments"]]
        if not g: v2[name]=dict(v); continue
        lo=min(a for a,_ in g); hi=max(b for _,b in g)
        good_hw=[]; cur=lo
        for a,b in W:
            if b<=lo or a>=hi: continue
            if a>cur: good_hw.append([cur,a])
            cur=max(cur,b)
        if cur<hi: good_hw.append([cur,hi])
        if not any(not(b<=lo or a>=hi) for a,b in W): v2[name]=dict(v); continue
        g2=inter(g,good_hw); d=sum(b-a for a,b in g)-sum(b-a for a,b in g2)
        v2[name]={**v,"good_segments":g2,"hwinj_removed_s":d}; removed+=d; nseg+=1; touched.append((name,d))
    print(f"[mask_hw] {run}: {len(v1)} segments, {nseg} touched by HW-inj windows, {removed/3600:.3f} h removed from good time")
    for name,d in touched: print(f"    {name}: -{d:.0f} s")
    if write:
        json.dump(v2,open(f"{DQ}/veto_mask_{run}_hw.json","w")); print(f"[mask_hw] wrote {DQ}/veto_mask_{run}_hw.json")
    else: print("[mask_hw] dry run (--write to save)")
if __name__=="__main__": main()
