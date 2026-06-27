"""Prep the 56-segment O3a blind-search config: (1) CAT2+HWInj veto mask for batch-2 segs (same DQ
convention as q_cat2_o3a.py), (2) merge batch-1+batch-2 veto masks -> veto_mask_o3a_56.json,
(3) merge batch-1+batch-2 segment lists -> o3a_bg_segments_56.json. Runs alongside the fetch (DQ only)."""
import json,time
from gwpy.segments import DataQualityFlag, SegmentList, Segment
B1=json.load(open("search_mode/o3a_bg_segments.json"))
B2=json.load(open("search_mode/o3a_bg_segments_batch2.json"))
V1=json.load(open("search_mode/veto_mask_o3a.json"))
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
def fr(flag,t0,t1,tries=8):
    delay=15
    for k in range(tries):
        try: return DataQualityFlag.fetch_open_data(flag,t0,t1)
        except Exception as e:
            if k==tries-1: raise
            time.sleep(delay); delay=min(delay*2,300)
V2={}
for a,b,d,nm in B2["segments"]:
    name=segname(a,nm); t0,t1=float(a),float(b); lock=SegmentList([Segment(t0,t1)]); T=t1-t0; good=SegmentList([Segment(t0,t1)]); info={}
    for det in ("H1","L1"):
        for flag in (f"{det}_CBC_CAT2",f"{det}_NO_CBC_HW_INJ"):
            try: act=fr(flag,t0,t1).active & lock; good=good & act; info[flag]=float(sum(float(s[1]-s[0]) for s in act))
            except Exception as e: info[flag]=None; print(f"  WARN {name} {flag}: {str(e)[:50]}",flush=True)
    vetoed=lock-good; vs=float(sum(float(s[1]-s[0]) for s in vetoed))
    V2[name]=dict(lock_dur_h=T/3600.0, good_s=float(sum(float(s[1]-s[0]) for s in good)), vetoed_s=vs,
                  vetoed_frac=vs/T if T else 0.0, n_veto_intervals=len(vetoed), flag_good_s=info,
                  good_segments=[[float(s[0]),float(s[1])] for s in good])
    print(f"[{name}] {T/3600:.2f}h CAT2+HW veto {vs:.0f}s ({100*V2[name]['vetoed_frac']:.2f}%)",flush=True)
# merge
V56={**V1,**V2}; json.dump(V56,open("search_mode/veto_mask_o3a_56.json","w"),indent=2)
seg56=B1["segments"]+B2["segments"]
tot=sum(d for a,b,d,nm in seg56)
out=dict(run="O3a-56", n_segments=len(seg56), total_coincident_s=tot, total_days=tot/86400.0, segments=seg56)
json.dump(out,open("search_mode/o3a_bg_segments_56.json","w"),indent=2)
print(f"\nWROTE veto_mask_o3a_56.json ({len(V56)} entries) + o3a_bg_segments_56.json ({len(seg56)} segs = {tot/86400:.2f}d)",flush=True)
