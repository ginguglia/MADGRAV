"""MERGE + FINISH driver for the 4-GPU shard blind scan.

Loads the N shard partials (search_mode/_pb_shard_parts/shard_k.npz) written by driver_blindscan.py
in partial-output mode (SM_NSHARD=N SM_SHARD=k), merges them, then runs STAGE-1 / STAGE-2 / STAGE-1b
+ SAVE exactly as driver_blindscan.main() does on the unsharded background.

merge rules (bit-identical -- sharding is pure partitioning of the pairing loop):
  bg[g]        = UNION  (concat the per-shard partial bg lists; order-independent set)
  famN         = per-(fold,L1key) MAX by net
  face_live    = SUM
  cohmax_seen  = MAX (tripwire)

The deterministic setup (segs/fold/pairings/mdl/fgt/floor/COH_CEIL) is rebuilt from the FULL pairings
via _pb_shard_common.build_setup(), which is a verbatim lift of the head of driver_blindscan.main();
each shard ran that same setup so the merged fgt/floor here are identical. STAGE-2 uses the SAME
batched CNN precompute the integrated single-process driver uses (PB.precompute_cnn) -> bit-identical.
Needs BLIND_DEV set to a GPU for the CNN stage. Imports driver_blindscan for its CNN helpers (lazy).
"""
import os,sys,json,time
import numpy as np
from scipy.stats import chi2
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import _pb_shard_common as C
import driver_search_multi as M
import driver_blindscan as B        # CNN helpers (cnn_hm_lm, cnet, lmnet, _win, cpipe, carm, GLITCH_THRESH...)
import _pb_cnn_precompute as PB     # batched CNN precompute (bit-identical lookup)

SHARD_DIR=os.environ.get("SM_SHARD_DIR","search_mode/_pb_shard_parts")
NSHARD=int(os.environ["SM_NSHARD"])

# ---- PRE-REGISTRATION HOLD (2026-06-23) ----------------------------------------------------------
# The merge currently computes the OR-veto statistic (best_far=min(far_cnn,far_net)) at the contaminated
# floor 4.5. Per the pre-registration it MUST NOT produce a quotable result until: (1) the per-arm
# branch is ported in from driver_blindscan.py (BLOCKER 1/2), (2) the floor is re-derived blind as L*
# (BLOCKER 3), (3) detection decided on UL90 (BLOCKER 4), (4) hard-fail on missing shards (BLOCKER 6).
# This guard makes the auto-merge DEFER cleanly instead of unblinding
# a wrong-statistic result. Run intentionally with SM_MERGE_OK=1 only after the fixes land.
if os.environ.get("SM_MERGE_OK")!="1":
    print("[merge] PRE-REGISTRATION HOLD: merge deferred (set SM_MERGE_OK=1 to run). "
          "Shards are preserved in "+SHARD_DIR+". "
          "Run only once the pre-registered conditions are met: per-arm branch ported, blind floor L* "
          "re-derived, detection decided on the 90% Poisson UL, and a hard-fail on missing shards.",flush=True)
    sys.exit(0)
# --------------------------------------------------------------------------------------------------

OUT=C.OUT; YR=C.YR; STRIDE=C.STRIDE
NET_CUT=C.NET_CUT; CLUSTER_S=C.CLUSTER_S; GLITCH_THRESH=B.GLITCH_THRESH
NETSIG_FLOOR=C.NETSIG_FLOOR; N_CNN_CANDIDATES=B.N_CNN_CANDIDATES

def load_partials():
    # Graceful degradation: merge whatever shards SUCCEEDED. A shard that OOM'd at flush leaves no .npz;
    # the survivors still hold a statistically valid background (livetime=SUM, famN=per-key MAX over the
    # present subset) -- just less livetime (=> a proportionally higher FAR floor). Missing shards are
    # reported, never fatal.
    parts=[]; missing=[]
    for k in range(NSHARD):
        f=f"{SHARD_DIR}/shard_{k}.npz"
        if os.path.exists(f): parts.append(np.load(f,allow_pickle=False))
        else: missing.append(k)
    if not parts:
        raise SystemExit(f"[merge] FATAL: 0/{NSHARD} shards produced output in {SHARD_DIR} -- nothing to merge")
    # Gate 6 (PREREGISTRATION §3): the quotable merge must be FULL-SHARD -- graceful degradation on a missing
    # shard silently changes the livetime/FAR and is forbidden for a quotable result. Hard-fail by default.
    # SM_ALLOW_PARTIAL=1 re-enables the old reduced-livetime merge for NON-prereg/diagnostic use only.
    if missing:
        if os.environ.get("SM_ALLOW_PARTIAL")=="1":
            print(f"[merge] WARNING (SM_ALLOW_PARTIAL=1, NON-PREREG): shards {missing} missing; merging "
                  f"{len(parts)}/{NSHARD} survivors -> reduced livetime, FAR floor ~{NSHARD/len(parts):.2f}x higher",flush=True)
        else:
            raise SystemExit(f"[merge] FATAL (Gate 6): shards {missing} missing of {NSHARD}. A quotable per-arm "
                             f"merge requires ALL shards. Re-run the missing shards, or set SM_ALLOW_PARTIAL=1 "
                             f"for a diagnostic (non-quotable) reduced-livetime merge.")
    return parts

def merge(parts):
    bg={0:[],1:[]}; famN={}; face_live={0:0.0,1:0.0}; cohmax_seen=0.0
    for p in parts:
        for g,key in ((0,"bg0"),(1,"bg1")):
            for row in p[key]:
                bg[g].append((float(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4])))
        for kk,vv in zip(p["fam_keys"],p["fam_vals"]):
            g=int(kk[0]); kky=(g,(int(kk[1]),int(kk[2])))   # (fold,(bi,l1bin))
            net=float(vv[0]); rep=(net,int(vv[1]),int(vv[2]),int(vv[3]),int(vv[4]))
            if kky not in famN or net>famN[kky][0]: famN[kky]=rep
        fl=p["face_live"]; face_live[0]+=float(fl[0]); face_live[1]+=float(fl[1])
        cohmax_seen=max(cohmax_seen,float(p["cohmax_seen"][0]))
    return bg,famN,face_live,cohmax_seen

def main():
    S=C.build_setup()
    segs=S["segs"]; nseg=S["nseg"]; fold=S["fold"]; evloc=S["evloc"]; evrad=S["evrad"]
    mdl=S["mdl"]; fgt=S["fgt"]; floor=S["floor"]; COH_CEIL=S["COH_CEIL"]
    L1key=lambda bi,iL:C.L1key(segs,bi,iL)

    parts=load_partials()
    bg,famN,face_live,cohmax_seen_v=merge(parts)
    far_live={g:face_live[g]*M.ANALYZED_FRAC/YR for g in (0,1)}
    print(f"[merge] livetime fold0={far_live[0]:.3f}yr fold1={far_live[1]:.3f}yr; "
          f"{len(bg[0])+len(bg[1])} loud bg survivors (merged from {NSHARD} shards)",flush=True)
    _trip="OK" if cohmax_seen_v<COH_CEIL else "VIOLATED -> ceiling too tight, FAR may be biased"
    print(f"[merge] PREFILTER TRIPWIRE: max realized survivor coh {cohmax_seen_v:.3f} vs COH_CEIL {COH_CEIL:.3f} -> {_trip}",flush=True)
    for g in (0,1): bg[g].sort(key=lambda r:-r[0])

    # ---- PER-ARM CNN-RANK helpers (Gate 1/2: ported VERBATIM from driver_blindscan.py:373-377; panel wf_9019f7d5) ----
    PERARM=int(os.environ.get("SM_PERARM","1"))        # DEFAULT 1: per-arm is the headline best_far statistic
    def _ul90(n,flt): return float(chi2.ppf(0.90,2*(n+1))/2/flt) if (flt and flt>0) else None
    def _perarm(a,b): return 2*min(a,b) if (a is not None and b is not None) else None   # x2 = arm-choice Bonferroni
    # ---- L* hook (Gate 3): candidate-admission floor. Defaults INERT (==prune floor) until an L* is pre-committed;
    #      SM_LSTAR>=prune floor only (bg below the prune floor was never saved -> cannot lower it post-hoc). ----
    LSTAR=float(os.environ.get("SM_LSTAR",floor))
    if LSTAR<floor: raise SystemExit(f"[merge] SM_LSTAR={LSTAR} < prune floor {floor}: bg below {floor} not saved; re-run shards to lower.")

    # ===== STAGE-1 FAR_LR (full-band) per trigger =====  (verbatim from driver_blindscan)
    fgt.sort(key=lambda d:-d["loglr"])
    for t in fgt:
        g=t["fold"]; flt=far_live[g]
        if t["loglr"]<floor:
            t["_louder"]=[]; t["n_louder_L1fam"]=None; t["far_lr_per_yr"]=None; t["far_lr_ul90"]=None
            t["below_floor"]=True; continue
        louder=[(ai,iH,bi,iL) for ll,ai,iH,bi,iL in bg[g] if ll>t["loglr"]]
        fams=set(L1key(bi,iL) for ai,iH,bi,iL in louder)
        t["_louder"]=louder; t["n_louder_L1fam"]=len(fams)
        t["far_lr_per_yr"]=len(fams)/flt if flt>0 else None
        t["far_lr_ul90"]=float(chi2.ppf(0.90,2*(len(fams)+1))/2/flt) if flt>0 else None

    # ===== STAGE-2 HM CNN (20-140) veto on CANDIDATE triggers =====
    above=[i for i,t in enumerate(fgt) if t["loglr"]>=LSTAR]   # admission floor = L* (inert default ==prune floor)
    cand_idx=set(above[:N_CNN_CANDIDATES])
    if NETSIG_FLOOR>0:
        for i,t in enumerate(fgt):
            if t["net"]>=NETSIG_FLOOR: cand_idx.add(i)
    for i,t in enumerate(fgt):
        for ev,(esi,ew) in evloc.items():
            if t["si"]==esi and abs(t["idx"]-ew)<=evrad: t["matches_known"]=ev; cand_idx.add(i)
    # ---- CNN precompute (bit-identical to lazy per-pair scoring), mirrors driver_blindscan STAGE-2 ----
    need_pairs=set()
    for i in cand_idx:
        if fgt[i].get("far_lr_per_yr") is not None: need_pairs.update(fgt[i]["_louder"])
    if NETSIG_FLOOR>0:
        _fams=sorted(famN.values(),key=lambda r:-r[0])
        for i in (j for j in cand_idx if fgt[j]["net"]>=NETSIG_FLOOR):
            g=fgt[i]["fold"]
            for r in _fams:
                if r[0]>=fgt[i]["net"] and fold[r[1]]==g: need_pairs.add((r[1],r[2],r[3],r[4]))
    cnn_cache=PB.precompute_cnn(need_pairs,segs,B.cpipe,B.carm,B.cnet,B.lmnet,B._win)
    print(f"[merge] STAGE-2 CNN precompute: {len(cnn_cache)} deduped louder-bg pairs scored up-front",flush=True)
    def surv_cnn(ai,iH,bi,iL):
        key=(int(ai),int(iH),int(bi),int(iL))
        if key not in cnn_cache: cnn_cache[key]=B.cnn_hm_lm(segs[ai]["name"],iH,segs[bi]["name"],iL)
        hm,lm=cnn_cache[key]; return max(hm,lm)
    _fallback_hits=[0]
    def surv_cnn_pair(ai,iH,bi,iL):                         # per-arm (hm,lm) lookup; reuses the same precompute cache
        key=(int(ai),int(iH),int(bi),int(iL))
        if key not in cnn_cache: _fallback_hits[0]+=1; cnn_cache[key]=B.cnn_hm_lm(segs[ai]["name"],iH,segs[bi]["name"],iL)
        return cnn_cache[key]                               # (hm,lm)
    def _f(v,fmt): return format(v,fmt) if v is not None else "NA"
    print(f"[merge] STEP2/3: CNN glitch-gate (>{GLITCH_THRESH}) + lag FAR on {len(cand_idx)} candidates",flush=True)
    for ci,i in enumerate(sorted(cand_idx)):
        t=fgt[i]; g=t["fold"]; flt=far_live[g]
        t["cnn_hm"],t["cnn_lm"]=B.cnn_hm_lm(t["seg"],t["idx"],t["seg"],t["idx"])
        t["cnn"]=max(t["cnn_hm"],t["cnn_lm"])
        t["kept_by"]=("HM" if t["cnn_hm"]>GLITCH_THRESH else "")+("+LM" if t["cnn_lm"]>GLITCH_THRESH else "") or "none"
        t["is_glitch"]=bool(t["cnn"]<GLITCH_THRESH)
        if t.get("far_lr_per_yr") is None:
            t["n_louder_L1fam_cnn"]=None; t["far_cnn_per_yr"]=None; t["far_cnn_ul90"]=None
            t["far_lr_hm"]=t["far_lr_lm"]=t["far_lr_perarm"]=t["far_lr_perarm_ul90"]=None  # sub-floor: no spurious 0
        else:
            kept=[(ai,iH,bi,iL) for (ai,iH,bi,iL) in t["_louder"] if surv_cnn(ai,iH,bi,iL)>GLITCH_THRESH]
            fams=set(L1key(bi,iL) for ai,iH,bi,iL in kept)
            t["n_louder_L1fam_cnn"]=len(fams); t["far_cnn_per_yr"]=len(fams)/flt if flt>0 else None
            t["far_cnn_ul90"]=float(chi2.ppf(0.90,2*(len(fams)+1))/2/flt) if flt>0 else None
            # --- PER-ARM RANK (loglr channel): louder bg L1-families by EACH arm vs the event's own cnn, self-seg guarded ---
            si=t["si"]; fam_hm=set(); fam_lm=set()
            for (ai,iH,bi,iL) in t["_louder"]:
                if ai==si or bi==si: continue
                hm,lm=surv_cnn_pair(ai,iH,bi,iL); k=L1key(bi,iL)
                if hm>=t["cnn_hm"]: fam_hm.add(k)
                if lm>=t["cnn_lm"]: fam_lm.add(k)
            t["far_lr_hm"]=len(fam_hm)/flt if flt>0 else None; t["far_lr_lm"]=len(fam_lm)/flt if flt>0 else None
            t["far_lr_perarm"]=_perarm(t["far_lr_hm"],t["far_lr_lm"])
            t["far_lr_perarm_ul90"]=_perarm(_ul90(len(fam_hm),flt),_ul90(len(fam_lm),flt))
        print(f"  cand {ci+1}/{len(cand_idx)} {t['seg']} loglr={_f(t.get('loglr'),'.2f')} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')} ({t['kept_by']})"
              f"{' GLITCH' if t['is_glitch'] else ''} FAR_LR={_f(t.get('far_lr_per_yr'),'.3g')} -> FAR_CNN={_f(t.get('far_cnn_per_yr'),'.3g')}/yr {t.get('matches_known','')}",flush=True)

    # ===== STAGE-1b NET-SIGMA OR-CHANNEL =====
    if NETSIG_FLOOR>0:
        fams=sorted(famN.values(),key=lambda r:-r[0])
        netcands=[i for i in cand_idx if fgt[i]["net"]>=NETSIG_FLOOR]
        print(f"[merge] NET-SIGMA channel: floor={NETSIG_FLOOR}; {len(famN)} bg L1-families, {len(netcands)} fg net-candidates",flush=True)
        for i in sorted(netcands,key=lambda i:-fgt[i]["net"]):
            t=fgt[i]; g=t["fold"]; flt=far_live[g]
            kept=[r for r in fams if r[0]>=t["net"] and fold[r[1]]==g and surv_cnn(r[1],r[2],r[3],r[4])>GLITCH_THRESH]
            t["n_louder_netfam_cnn"]=len(kept)
            t["far_net_per_yr"]=len(kept)/flt if flt>0 else None
            t["far_net_ul90"]=float(chi2.ppf(0.90,2*(len(kept)+1))/2/flt) if flt>0 else None
            # --- PER-ARM RANK (net-sigma channel): louder-in-net famN families, by EACH arm, self-seg guarded ---
            si=t["si"]; nhm=nlm=0
            for r in fams:
                if not (r[0]>=t["net"] and fold[r[1]]==g): continue
                if r[1]==si or r[3]==si: continue
                hm,lm=surv_cnn_pair(r[1],r[2],r[3],r[4])
                if hm>=t["cnn_hm"]: nhm+=1
                if lm>=t["cnn_lm"]: nlm+=1
            t["far_net_hm"]=nhm/flt if flt>0 else None; t["far_net_lm"]=nlm/flt if flt>0 else None
            t["far_net_perarm"]=_perarm(t["far_net_hm"],t["far_net_lm"])
            t["far_net_perarm_ul90"]=_perarm(_ul90(nhm,flt),_ul90(nlm,flt))
            print(f"  net-cand {t['seg']} net={t['net']:.2f} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')}"
                  f"{' GLITCH' if t.get('is_glitch') else ''} -> FAR_net={_f(t.get('far_net_per_yr'),'.3g')}/yr (UL90 {_f(t.get('far_net_ul90'),'.3g')}) {t.get('matches_known','')}",flush=True)

    if PERARM: print(f"[merge] PER-ARM RANK active (SM_PERARM=1); surv_cnn_pair live-fallback hits = {_fallback_hits[0]} (must be 0)",flush=True)
    def best_stat(t):                                        # (point_far, ul90) for the detection statistic
        if PERARM:                                           # per-arm headline: x n_channels trials on BOTH point and UL90
            ch=[(t.get("far_lr_perarm"),t.get("far_lr_perarm_ul90")),(t.get("far_net_perarm"),t.get("far_net_perarm_ul90"))]
            ch=[c for c in ch if c[0] is not None]
            if not ch: return (None,None)
            n=len(ch); pt,ul=min(ch,key=lambda c:c[0])
            return (n*pt,(n*ul if ul is not None else None))
        ch=[(t.get("far_cnn_per_yr"),t.get("far_cnn_ul90")),(t.get("far_net_per_yr"),t.get("far_net_ul90"))]  # legacy OR-veto: no channel trials
        ch=[c for c in ch if c[0] is not None]
        return min(ch,key=lambda c:c[0]) if ch else (None,None)
    def best_far(t): return best_stat(t)[0]
    def best_ul90(t): return best_stat(t)[1]
    def _chan(d):                                            # which channel set best_far
        a,b=( (d.get("far_lr_perarm"),d.get("far_net_perarm")) if PERARM else (d.get("far_cnn_per_yr"),d.get("far_net_per_yr")) )
        if a is None: return "net-sigma"
        if b is None: return "loglr"
        return "net-sigma" if b<a else "loglr"
    # ===== SAVE EVERY SURVIVING BACKGROUND TRIGGER =====
    llmap={}
    for g in (0,1):
        for ll,ai,iH,bi,iL in bg[g]: llmap[(ai,int(iH),bi,int(iL))]=float(ll)
    def _prov(ai,iH,bi,iL,hm,lm):
        return dict(H1_seg=segs[ai]["name"],iH=int(iH),gps_H=float(segs[ai]["gps"][iH]),
                    sigma_H=float(segs[ai]["sH"][iH]),cen_H=float(segs[ai]["cH"][iH]),
                    L1_seg=segs[bi]["name"],iL=int(iL),gps_L=float(segs[bi]["gps"][iL]),
                    sigma_L=float(segs[bi]["sL"][iL]),cen_L=float(segs[bi]["cL"][iL]),
                    lag_s=float((iH-iL)*STRIDE),loglr=llmap.get((ai,int(iH),bi,int(iL))),
                    cnn_hm=float(hm),cnn_lm=float(lm),cnn=float(max(hm,lm)),
                    is_glitch=bool(max(hm,lm)<GLITCH_THRESH))
    bg_surv=[_prov(ai,iH,bi,iL,hm,lm) for (ai,iH,bi,iL),(hm,lm) in cnn_cache.items()]
    bg_surv.sort(key=lambda r:-(r["loglr"] if r["loglr"] is not None else -1e9))
    n_pass=sum(1 for r in bg_surv if not r["is_glitch"])
    json.dump(dict(glitch_thresh=GLITCH_THRESH,n_scored=len(bg_surv),n_cnn_survivors=n_pass,survivors=bg_surv),
              open(f"{OUT}/survivors_bg.json","w"),indent=2)
    print(f"[merge] saved {len(bg_surv)} CNN-scored bg survivors ({n_pass} pass CNN gate) -> survivors_bg.json",flush=True)

    for t in fgt: t.pop("_louder",None)
    json.dump(dict(net_cut=NET_CUT,cluster_s=CLUSTER_S,n_segments=nseg,far_live_yr=far_live,netsig_floor=NETSIG_FLOOR,
                   n_triggers=len(fgt),n_cnn_vetoed=len(cand_idx),triggers=fgt),
              open(f"{OUT}/blindscan.json","w"),indent=2)
    DET_FAR=float(os.environ.get("DET_FAR","1.0"))
    # Gate 4 (PREREGISTRATION §3): detection requires BOTH the point per-arm FAR AND its 90% Poisson UL < DET_FAR.
    dets=[t for t in fgt if not t.get("is_glitch")
          and best_far(t) is not None and best_far(t)<DET_FAR
          and best_ul90(t) is not None and best_ul90(t)<DET_FAR]
    dets.sort(key=lambda d:best_far(d))
    json.dump([dict(seg=d["seg"],gps=d["gps"],net=d["net"],loglr=d["loglr"],cnn_hm=d.get("cnn_hm"),cnn_lm=d.get("cnn_lm"),
                    kept_by=d.get("kept_by"),far_cnn_per_yr=d.get("far_cnn_per_yr"),far_cnn_ul90=d.get("far_cnn_ul90"),
                    far_net_per_yr=d.get("far_net_per_yr"),far_net_ul90=d.get("far_net_ul90"),
                    far_lr_perarm=d.get("far_lr_perarm"),far_lr_perarm_ul90=d.get("far_lr_perarm_ul90"),
                    far_net_perarm=d.get("far_net_perarm"),far_net_perarm_ul90=d.get("far_net_perarm_ul90"),
                    far_lr_hm=d.get("far_lr_hm"),far_lr_lm=d.get("far_lr_lm"),
                    far_net_hm=d.get("far_net_hm"),far_net_lm=d.get("far_net_lm"),
                    best_far=best_far(d),best_ul90=best_ul90(d),stat=("per-arm" if PERARM else "OR-veto"),
                    channel=_chan(d),matches_known=d.get("matches_known","")) for d in dets],
              open(f"{OUT}/detections.json","w"),indent=2)
    print(f"\n=== DETECTIONS (point FAR<{DET_FAR}/yr AND UL90<{DET_FAR}/yr; stat={'per-arm' if PERARM else 'OR-veto'}): {len(dets)} ===",flush=True)
    for d in dets:
        print(f"  {str(d.get('matches_known') or d['seg']):22s} {d['gps']:12.1f} net={d['net']:.2f} loglr={d['loglr']:.2f} "
              f"best_far={best_far(d):.3g} UL90={best_ul90(d):.3g} ({_chan(d)})",flush=True)

if __name__=="__main__": main()
