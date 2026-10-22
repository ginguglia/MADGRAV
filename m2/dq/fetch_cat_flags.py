"""Fetch GWOSC CBC/BURST CAT2 and CAT3 flags (both detectors, full run) -> dq/cat_flags_<run>.json. Login node. 2026-10-05.
Usage: python fetch_cat_flags.py <run>"""
import sys, json, time
from gwosc.api import fetch_json
from fetch_hwinj_flags import RUNS, DQ
FL=["CBC_CAT2","BURST_CAT2","CBC_CAT3","BURST_CAT3"]
run=sys.argv[1]; ds,a,b=RUNS[run]; out={}
for ifo in ("H1","L1"):
    for fl in FL:
        segs=None
        for k in range(6):
            try: segs=fetch_json(f"https://gwosc.org/timeline/segments/json/{ds}/{ifo}_{fl}/{a}/{b-a}/").get("segments",[]); break
            except Exception as e: print(run,ifo,fl,"retry",k,str(e)[:80],flush=True); time.sleep(5)
        out[f"{ifo}_{fl}"]=segs; print(f"{run} {ifo} {fl:11s} n={len(segs) if segs is not None else 'FAIL'} active_total={sum(s[1]-s[0] for s in segs) if segs else None}",flush=True)
json.dump(dict(dataset=ds,span=[a,b],flags=out),open(f"{DQ}/cat_flags_{run}.json","w")); print("wrote",f"{DQ}/cat_flags_{run}.json")
