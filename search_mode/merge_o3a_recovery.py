"""Merge per-chunk recovery.json from the full-O3a foreground scan, GPS-match the CNN-passing candidates
to the 44 O3a confident events (o3a_events_full.json), and print a summary. Returns the summary text on
stdout so the orchestrator can email it."""
import json, glob, os, sys
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH; SM=os.path.dirname(os.path.abspath(__file__))
cat=json.load(open(f"{SM}/o3a_events_full.json"))                 # {event: gps}
files=sorted(glob.glob(f"{SC}/search_out_o3a_full/chunk_*/recovery.json"))
cands=[]
for f in files:
    d=json.load(open(f))
    cands.extend(d.get("recovered",[]))
passed=[c for c in cands if not c.get("is_glitch")]               # pass the CNN glitch gate
TOL=2.0
def match(gps):
    best=None;bd=1e9
    for ev,eg in cat.items():
        dd=abs(gps-eg)
        if dd<bd: bd=dd;best=ev
    return (best,bd) if bd<=TOL else (None,bd)
recovered={}
for c in passed:
    ev,dd=match(float(c["gps"]))
    if ev and (ev not in recovered or (c.get("loglr") or 0)>(recovered[ev].get("loglr") or 0)):
        recovered[ev]=c
nrec=len(recovered); ncat=len(cat)
lines=[]
lines.append(f"FULL-O3a FOREGROUND RECOVERY (chunks merged: {len(files)}/13)")
lines.append(f"candidates passing CNN gate: {len(passed)} (of {len(cands)} scored across 725 segs)")
lines.append(f"RECOVERED O3a confident events: {nrec}/{ncat}")
lines.append("")
lines.append("recovered (event  loglr  net  HM  LM):")
for ev,c in sorted(recovered.items(), key=lambda kv:-(kv[1].get('loglr') or 0)):
    lines.append(f"  {ev:20s} loglr={(c.get('loglr') or 0):5.2f} net={c.get('net',0):5.2f} HM={c.get('cnn_hm',0):.3f} LM={c.get('cnn_lm',0):.3f}")
missed=[ev for ev in cat if ev not in recovered]
lines.append("")
lines.append(f"not recovered ({len(missed)}): "+", ".join(sorted(missed)))
out="\n".join(lines)
json.dump(dict(n_candidates_pass=len(passed),n_recovered=nrec,n_catalog=ncat,
               recovered={k:{kk:v.get(kk) for kk in ('gps','loglr','net','cnn_hm','cnn_lm')} for k,v in recovered.items()},
               missed=missed), open(f"{SC}/search_out_o3a_full/recovery_summary.json","w"), indent=2)
print(out)
