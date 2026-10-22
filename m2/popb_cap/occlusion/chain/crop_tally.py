"""Final recount after the crop re-pass: per run, the crop _v2 detections vs the nf275 _v2 detections and vs the set-88 table; totals,
new (with FAR), lost, non-catalogue; writes TALLY_CROP_<date>.txt."""
import json, os, datetime
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH + "/madgrav2"; H=MADGRAV_ROOT + "/m2/popb_cap/occlusion/chain"
S88={n for n,v in json.load(open(f"{SC}/figures_m2/hl_snr_o3o4_m2_v2cap88.json")).items() if v.get("mad")}
def dets(p):
    out={}
    for x in json.load(open(p)): out[x.get("matches_known") or f"gps{int(x['gps'])}"]=x
    return out
lines=[f"Crop re-pass tally {datetime.date.today()} (SM_CROP=coinc, floor 2.75, nf275 shards, full CNN+S3 re-score, LR frozen, ranker applied)"]
tot=0; tot88=0; allnew=[]; alllost=[]; noncat=[]
for run in ("o3a","o3b","o4a","o4b"):
    pc=f"{SC}/search_out_{run}_m2_gx_off5_crop/detections_v2.json"; pp=f"{SC}/search_out_{run}_m2_gx_off5_nf275/detections_v2.json"
    if not os.path.exists(pc): lines.append(f"[{run}] crop detections_v2.json MISSING"); continue
    dc,dp=dets(pc),dets(pp); cat=[n for n in dc if n.startswith("GW")]; nc=[n for n in dc if not n.startswith("GW")]
    new=sorted(set(dc)-set(dp)); lost=sorted(set(dp)-set(dc)); tot+=len(cat); tot88+=len([n for n in cat if n in S88]); noncat+=[(run,n,dc[n].get("best_far")) for n in nc]
    lines.append(f"[{run}] crop _v2: {len(dc)} ({len(cat)} catalogue, {len(nc)} non-cat) | nf275 _v2: {len(dp)} | new {len(new)}: "+", ".join(f"{n} ({dc[n].get('best_far'):.3g}/yr, {dc[n].get('channel')})" for n in new)+f" | lost {len(lost)}: {lost}")
    allnew+=[(run,n) for n in new]; alllost+=[(run,n) for n in lost]
lines.append(f"\nTOTAL catalogue detections at FAR<1/yr: {tot} (of which in the 88-set: {tot88}; 88-set members lost: {len([1 for r,n in alllost if n in S88])}); new catalogue rows: {len([1 for r,n in allnew if n.startswith('GW')])}; non-catalogue rows: {len(noncat)} {noncat}")
txt="\n".join(lines); open(f"{H}/TALLY_CROP_{datetime.date.today():%Y%m%d}.txt","w").write(txt+"\n"); print(txt)
