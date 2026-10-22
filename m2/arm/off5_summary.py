"""Summary + mail of the O4b _off5 validation pass: new merges vs the production o4b gx merges. Env: JIDS (for SM_OUT), NOTIFY_* (site.conf)."""
import os, sys, json, glob, re
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ROOT=MADGRAV_ROOT; sys.path.insert(0,f"{ROOT}/launchers"); SC=MADGRAV_SCRATCH
EXP=[l for l in open(os.environ["JIDS"]) if l.startswith("EXP=")][-1].strip()[4:]; E=dict(kv.split("=",1) for kv in EXP.split(",") if "=" in kv)
RUN=os.environ.get("OFF5_RUN",E.get("SM_RUN","o4b")); OUT=E["SM_OUT"]; PROD=f"{SC}/madgrav2/search_out_{RUN}_m2_gx"
def load(p):
    try: return json.load(open(p))
    except Exception as e: return None
def dets(d): return {(x.get("matches_known") or f"gps{int(x['gps'])}"):(x["best_far"],x["best_ul90"],x.get("channel",""),x.get("net")) for x in d} if d else None
lines=[f"MADGRAV 2.1 arm _off5 {RUN.upper()} validation pass -- {OUT}",f"EXP: {EXP[:400]}...",""]
ref={"prod max:-2.05 (_def)":f"{PROD}/detections.json","prod _s3rank (_def)":f"{PROD}/detections_s3rank.json","prod none (_def)":f"{PROD}/detections_none.json"}
new={"off5 max:-2.49":f"{OUT}/detections.json","off5 max:-2.05":f"{OUT}/detections_p205.json","off5 none":f"{OUT}/detections_none.json"}
if RUN!="o4b": new={"off5 max:-2.05":f"{OUT}/detections.json"}
R={k:dets(load(v)) for k,v in ref.items()}; N={k:dets(load(v)) for k,v in new.items()}
for name,D in list(R.items())+list(N.items()):
    if D is None: lines.append(f"{name:24s}: MISSING"); continue
    cat=[k for k in D if k.startswith("GW")]; lines.append(f"{name:24s}: {len(D)} detections, {len(cat)} catalogue, {len(D)-len(cat)} non-catalogue")
lines.append("")
base=R["prod max:-2.05 (_def)"]
for name,D in N.items():
    if D is None or base is None: continue
    new_=sorted(set(D)-set(base)); lost=sorted(set(base)-set(D)); lines.append(f"[{name}] vs prod max:-2.05: NEW {len(new_)}: "+", ".join(f"{k} (FAR {D[k][0]:.3g}/yr, {D[k][2]})" for k in new_))
    lines.append(f"[{name}] LOST {len(lost)}: "+", ".join(f"{k} (prod FAR {base[k][0]:.3g}/yr)" for k in lost))
    common=sorted(set(D)&set(base)); ch=[(k,base[k][0],D[k][0]) for k in common if abs(D[k][0]-base[k][0])>1e-9]
    lines.append(f"[{name}] FAR changed on {len(ch)}/{len(common)} common: "+", ".join(f"{k} {a:.3g}->{b:.3g}" for k,a,b in sorted(ch,key=lambda t:-abs(t[2]-t[1]))[:12])); lines.append("")
for tag,f in ((("off5 max:-2.49","blindscan.json"),("off5 max:-2.05","blindscan_p205.json"),("off5 none","blindscan_none.json")) if RUN=="o4b" else (("off5 max:-2.05","blindscan.json"),)):
    b=load(f"{OUT}/{f}")
    if isinstance(b,dict): lines.append(f"[{tag}] blindscan keys: livetime {b.get('far_live_yr',b.get('livetime'))} candidates {len(b.get('triggers',b.get('candidates',[])) or [])}")
sm=sorted(glob.glob(f"{E['SM_STREAMS']}/*_g2_off5.json")); tw=[json.load(open(s)).get("tripwire_max_abs") for s in sm]; tw=[t for t in tw if t is not None]
lines.append(f"rescoring: {len(sm)} det-segments, tripwire max|d| over all = {max(tw) if tw else 'n/a'}; rescored windows = {sum(json.load(open(s))['n_rescored'] for s in sm)}")
for st in ("RESCORE","INJ","LR","PRUNE","TOPUP","MERGE_249","MERGE_205","MERGE_NONE","MERGE"):
    jid=[l.split("=")[1].strip() for l in open(os.environ["JIDS"]) if l.startswith(st+"=")]
    if jid: lines.append(f"{st}: job {jid[-1]} -> "+os.popen(f"sacct -j {jid[-1]} -X -n -o State,Elapsed | sort | uniq -c | tr -s ' ' | tr '\\n' ';'").read().strip())
try:
    import subprocess; lines.append(""); lines.append(subprocess.run([sys.executable,MADGRAV_ROOT + "/m2/arm/off5_readout.py","--out",OUT],capture_output=True,text=True,timeout=1200).stdout)
except Exception as e: lines.append(f"read-out failed: {e}")
txt="\n".join(lines); print(txt); open(f"{OUT}/SUMMARY_OFF5.txt","w").write(txt+"\n")
try:
    import chain_notify as notify; notify.deliver(OUT,"SUMMARY_OFF5.txt",f"[MADGRAV] {RUN.upper()} arm _off5 validation pass: summary",txt)
except Exception as e: print("notify failed:",e)
