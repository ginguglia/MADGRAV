"""Post-merge audit (2026-10-05): run after EVERY merge. For one run and merge tag it reports
  (1) every detection row against the LVK flags: CBC/BURST CAT2 and CAT3 (dq/cat_flags_<run>.json) and the hardware-injection
      windows (dq/hwinj_windows_<run>.json), per detector;
  (2) the S3-overrules-CNN population (June CNN is_glitch AND s3 > threshold) with the loud members flagged;
  (3) the reverse population (CNN pass, S3 reject) and any catalogue event in it.
Writes <OUT>/audit<tag>.json and prints the tables. Login node or CPU node (JSON only, no network).
Usage: python audit_merge.py <run> [<tag>=_s3rank] [<OUT dir>=search_out_<run>_m2_gx_off5]"""
import sys, os, json
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH + "/madgrav2"; DQ=f"{SC}/dq"
run=sys.argv[1]; tag=sys.argv[2] if len(sys.argv)>2 else "_s3rank"; OUT=sys.argv[3] if len(sys.argv)>3 else f"{SC}/search_out_{run}_m2_gx_off5"
THR=json.load(open(MADGRAV_ROOT + "/m2/rank/s3_threshold.json"))["threshold"]
def inside(g,S): return any(a<g<b for a,b in S)                       # strictly inside (trigger gps is the integer cluster second)
def near_edge(g,S,tol=1.0): return any(abs(g-a)<=tol or abs(g-b)<=tol for a,b in S)
def inv(active,span):
    out=[]; cur=span[0]
    for a,b in sorted(active):
        if a>cur: out.append([cur,a])
        cur=max(cur,b)
    if cur<span[1]: out.append([cur,span[1]])
    return out
flags={}
cf=f"{DQ}/cat_flags_{run}.json"
if os.path.exists(cf):
    C=json.load(open(cf)); span=C["span"]
    for k,v in C["flags"].items():
        if v is not None: flags[k]=inv(v,span)          # vetoed (inactive) intervals
else: print(f"[audit] WARNING no {cf} -> CAT flags not checked")
hw=json.load(open(f"{DQ}/hwinj_windows_{run}.json"))["hwinj_windows_union"] if os.path.exists(f"{DQ}/hwinj_windows_{run}.json") else None
mask=json.load(open(f"{DQ}/veto_mask_{run}_hw.json")) if os.path.exists(f"{DQ}/veto_mask_{run}_hw.json") else (json.load(open(f"{DQ}/veto_mask_{run}.json")) if os.path.exists(f"{DQ}/veto_mask_{run}.json") else None)
def mask_good(seg,g):                                                    # what the streams actually saw (mask incl. the catalogue exception)
    if mask is None or seg not in mask: return None
    return any(a<=g<=b for a,b in mask[seg]["good_segments"])
B=json.load(open(f"{OUT}/blindscan{tag}.json")); T=B["triggers"]; det=json.load(open(f"{OUT}/detections{tag}.json"))
rep=dict(run=run,tag=tag,out=OUT,s3_thr=THR,detections=[],s3_overrules_cnn=[],cnn_pass_s3_reject=[])
print(f"[audit] {run}{tag}: {len(det)} detections, {len(T)} triggers; flags: {sorted(flags)}; hwinj windows: {len(hw) if hw is not None else 'n/a'}\n")
print(f"{'event':22}{'gps':>12}{'chan':>10}{'far':>8}  flags hit (vetoed intervals containing the trigger)")
bad=0
for r in sorted(det,key=lambda x:x.get("best_far",9e9)):
    g=r["gps"]; hits=[k for k,S in flags.items() if inside(g,S)]; edges=[k for k,S in flags.items() if near_edge(g,S) and k not in hits]
    if hw is not None and inside(g,hw): hits.append("HW_INJ")
    mg=mask_good(r.get("seg"),g); note=("" if mg is None else (" [mask: GOOD -> catalogue exception]" if (hits and mg) else (" [mask: vetoed]" if (hits and not mg) else "")))
    if edges: note+=f" (edge within 1 s: {', '.join(edges)})"
    nm=r.get("matches_known") or "** NON-CATALOGUE **"
    print(f"{nm:22}{g:12.0f}{str(r.get('channel','?')):>10}{r.get('best_far',float('nan')):8.3f}  {', '.join(hits) if hits else '-'}{note}")
    rep["detections"].append(dict(name=r.get("matches_known"),gps=g,channel=r.get("channel"),best_far=r.get("best_far"),flags=hits,edge_flags=edges,mask_good=mg)); bad+=bool(hits)
print(f"\n[audit] detections inside a vetoed/injection interval: {bad} (catalogue among them: {sum(1 for d in rep['detections'] if d['flags'] and d['name'])})")
scored=[t for t in T if t.get("s3") is not None]; cg=[t for t in scored if t.get("is_glitch")]
flip=[t for t in cg if t["s3"]>THR]; rev=[t for t in scored if not t.get("is_glitch") and t["s3"]<=THR]
dg={round(r["gps"]) for r in det}
print(f"\n[audit] S3 overrules CNN: {len(flip)}/{len(cg)} CNN-glitch triggers admitted by S3 (loud net>=8: {sum(1 for t in flip if t['net']>=8)})")
print(f"{'event':22}{'gps':>12}{'net':>6}{'s3':>6}{'cnn_hm':>8}{'far_lr':>8}  det?  flags")
for t in sorted(flip,key=lambda t:-t["net"]):
    g=t["gps"]; hits=[k for k,S in flags.items() if inside(g,S)]+(["HW_INJ"] if hw is not None and inside(g,hw) else [])
    nm=t.get("matches_known") or ("** LOUD NON-CAT **" if t["net"]>=8 else "-")
    print(f"{nm:22}{g:12.0f}{t['net']:6.1f}{t['s3']:6.2f}{t.get('cnn_hm',0):8.3f}{(t.get('far_lr_perarm') if t.get('far_lr_perarm') is not None else float('nan')):8.3f}  {'DET' if round(g) in dg else '   '}  {', '.join(hits) if hits else '-'}")
    rep["s3_overrules_cnn"].append(dict(name=t.get("matches_known"),gps=g,net=t["net"],s3=t["s3"],cnn_hm=t.get("cnn_hm"),far_lr_perarm=t.get("far_lr_perarm"),detection=round(g) in dg,flags=hits))
print(f"\n[audit] CNN pass, S3 reject: {len(rev)}; catalogue among them: {[t['matches_known'] for t in rev if t.get('matches_known')]}")
rep["cnn_pass_s3_reject"]=[dict(name=t.get("matches_known"),gps=t["gps"],net=t["net"],s3=t["s3"]) for t in rev]
json.dump(rep,open(f"{OUT}/audit{tag}.json","w"),indent=1); print(f"[audit] wrote {OUT}/audit{tag}.json")
