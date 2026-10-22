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
import os,sys,json,time,glob
import numpy as np
from scipy.stats import chi2
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import _pb_shard_common as C
import driver_search_multi as M
import driver_blindscan as B        # CNN helpers (cnn_hm_lm, cnet, lmnet, _win, cpipe, carm, GLITCH_THRESH...)
import _pb_cnn_precompute as PB     # batched CNN precompute (bit-identical lookup)

SHARD_DIR=os.environ.get("SM_SHARD_DIR","search_mode/_pb_shard_parts")
NSHARD=int(os.environ["SM_NSHARD"])

# ---- PRE-REGISTRATION GUARD (opened 2026-06-23; status updated 2026-07-01) -----------------------
# STATISTIC IS PER-ARM (not OR-veto): best_far = _perarm = 2*min(hm,lm) x n_channels, SM_PERARM=1 default
# (see lines ~108-224). Pre-reg BLOCKERS now RESOLVED IN CODE: (1/2) per-arm branch ported; (4) detection
# on BOTH point-FAR AND 90% Poisson UL90 (line ~257); (6) hard-fail on missing shards (line ~73).
# STILL OPEN: (3) blind-floor L* re-derivation -- the L* hook (line ~111) defaults INERT, so the floor is
# CAND_LOGLR_FLOOR (blind via SM_BLIND_FLOOR=1) not a separately-pre-committed L*. Does NOT affect the statistic.
# The SM_MERGE_OK=1 guard remains: run intentionally, with N_eff/L* fixed blind before reading the count.
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
# --- NET-SIGMA LM RULE (2026-09-22) ---------------------------------------------------------------
# In the NET-SIGMA channel only, require cnn_lm > SM_NETSIG_LM_MIN. Rationale: net=(sH+sL)/sqrt(2)
# lets ONE loud leg carry a pair, so a single-detector low-frequency transient can reach this channel
# with the low-mass specialist reading ~0. The threshold is the 1st percentile of the LM distribution
# of net-sigma-channel INJECTIONS (0.0382 on inj_o4b_m2_cnn) -- signal-derived, not tuned on the
# detections. It removes the 5 HM-only non-catalogue candidates (LM 0.0001-0.0184) and costs no
# catalogue detection in any of the four runs; the faintest catalogue net-sigma LM is 0.0929.
# SUPERSEDES the LM>0.5 proposal of 2026-09-21, which the held-out o4a rejected: it would have cost
# GW230704_212616 (LM 0.0929) and GW231127_165300 (LM 0.3305).
# Applied to BOTH the foreground net-candidates and the FAR-setting background families, so the FAR
# is computed against the population the rule admits. Default 0.0 = inert (pre-rule behaviour).
NETSIG_LM_MIN=float(os.environ.get("SM_NETSIG_LM_MIN","0"))

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
    # ---- STAT HOOK (PREREG_RANKER_20260924 §3.3): replace the loglr-channel statistic of bg rows and triggers from a file ----
    HOOK=os.environ.get("SM_STAT_HOOK"); _HOOK_MISS=set()
    if HOOK:
        _h=np.load(HOOK,allow_pickle=True)
        _bk={(str(a),int(b),str(c),int(d)):float(s) for a,b,c,d,s in zip(_h["bank_H1_seg"],_h["bank_iH"],_h["bank_L1_seg"],_h["bank_iL"],_h["bank_stat"])}
        _tk={(str(a),round(float(b),1)):float(s) for a,b,s in zip(_h["trig_seg"],_h["trig_gps"],_h["trig_stat"])}
        _nm={g:0 for g in bg}
        for g in bg:
            new=[]
            for (ll,ai,iH,bi,iL) in bg[g]:
                k=(segs[ai]["name"],int(iH),segs[bi]["name"],int(iL))
                if k in _bk: new.append((_bk[k],ai,iH,bi,iL)); _nm[g]+=1
                else: new.append((ll,ai,iH,bi,iL)); _HOOK_MISS.add((int(ai),int(iH),int(bi),int(iL)))
            bg[g]=new
        _tm=0
        for t in fgt:
            k=(t["seg"],round(float(t["gps"]),1))
            if k in _tk: t["loglr"]=_tk[k]; _tm+=1
        print(f"[merge] STAT HOOK {HOOK}: bg rows substituted fold0 {_nm[0]} fold1 {_nm[1]}, unmatched {len(_HOOK_MISS)}; triggers substituted {_tm}/{len(fgt)}",flush=True)
    # ---- S3 HOOK (PREREG_SPECIALISTS A6, option c): loglr-channel gate + single-arm FAR from the hm 20-300 Hz fold ensemble ----
    S3HOOK=os.environ.get("SM_S3_HOOK"); S3_THR=None; _s3b={}; _s3t={}; _S3ENS=None
    if S3HOOK:
        import hashlib as _hl, torch as _th, torch.nn as _nn
        S3_THR=float(os.environ["SM_S3_THR"]); _z=np.load(S3HOOK,allow_pickle=True)
        _s3b={(str(a_),int(b_),str(c_),int(d_)):float(v_) for a_,b_,c_,d_,v_,k_ in zip(_z["H1_seg"],_z["iH"],_z["L1_seg"],_z["iL"],_z["s3"],_z["kind"]) if k_=="bank"}
        _s3t={(str(a_),round(float(g_),1)):float(v_) for a_,g_,v_,k_ in zip(_z["H1_seg"],_z["gps_H"],_z["s3"],_z["kind"]) if k_=="trig"}
        class _Net2(_nn.Module):
            def __init__(s_):
                super().__init__(); ch=[2,16,32,64,128]
                s_.b=_nn.ModuleList([_nn.Sequential(_nn.Conv2d(ch[i],ch[i+1],3,padding=1),_nn.BatchNorm2d(ch[i+1]),_nn.ReLU(),_nn.MaxPool2d(2)) for i in range(4)])
                s_.h=_nn.Sequential(_nn.AdaptiveAvgPool2d(1),_nn.Flatten(),_nn.Dropout(0.3),_nn.Linear(128,64),_nn.ReLU(),_nn.Linear(64,1)); s_.register_buffer("temperature",_th.ones(1))
            def forward(s_,x):
                for bl in s_.b: x=bl(x)
                return s_.h(x).squeeze(-1)
        class _S3Ens(_nn.Module):                      # returns the LOGIT of the 5-seed mean probability, so torch.sigmoid(ens(x)) = p_S3
            def __init__(s_,models): super().__init__(); s_.m=_nn.ModuleList(models)
            def forward(s_,x):
                p=_th.stack([_th.sigmoid(m(x)/m.temperature) for m in s_.m]).mean(0).clamp(1e-6,1-1e-6); return _th.log(p/(1-p))
        _MD=os.environ.get("SM_S3_MODELS",MADGRAV_SCRATCH + "/madgrav2_spec/models"); _S3ENS={}
        for _f in (0,1):
            _ms=[]
            for _sd in range(5):
                _m=_Net2().to(B.DEV); _m.load_state_dict(_th.load(f"{_MD}/spec_hm_f{_f}_s{_sd}_nfull.pt",map_location=B.DEV)); _m.eval(); _ms.append(_m)
            _S3ENS[_f]=_S3Ens(_ms).to(B.DEV).eval()
        def _s3_fold_of_gps(g_): return int(_hl.sha1(str(int(g_//86400)).encode()).hexdigest(),16)%2
        print(f"[merge] S3 HOOK {S3HOOK}: {len(_s3b)} bank scores, {len(_s3t)} trigger scores, threshold {S3_THR}; loglr channel = S3 single arm, net-sigma channel = June",flush=True)
    # ---- MADGRAV2 pair-level arm rule applied at MERGE time (so one background prune serves every rule) ----
    # SM_ARM_RULE = none | max:<thr> | mean:<thr> | both:<thr> | ramp:<thr>  (thresholds = p1 over triggering injection pairs,
    # m2/arm/pair_rule.py 2026-09-17: max -2.05, mean -1.29, both -2.76, ramp -2.47). Applied identically to zero-lag triggers,
    # loglr-channel background survivors and net-sigma-channel families. Requires SM_ARM_V2=1 (segs gH/gL = g2).
    RULE=os.environ.get("SM_ARM_RULE","none")
    def _rule_pass(gH,gL,sH,sL):
        if RULE=="none": return True
        name,thr=RULE.split(":"); thr=float(thr); gH,gL,sH,sL=(float(x) for x in (gH,gL,sH,sL))
        if name=="max": v=gH if sH>=sL else gL
        elif name=="mean": v=0.5*(gH+gL)
        elif name=="both": v=min(gH,gL)
        elif name=="ramp": v=min(1.0,max(0.0,sH/3.0))*gH+min(1.0,max(0.0,sL/3.0))*gL
        else: raise SystemExit(f"[merge] unknown SM_ARM_RULE {RULE}")
        return v>=thr
    def _pair_ok(ai,iH,bi,iL): A_,B_=segs[ai],segs[bi]; return _rule_pass(A_["gH"][iH],B_["gL"][iL],A_["sH"][iH],B_["sL"][iL])
    if RULE!="none":
        n_bg0=sum(len(v) for v in bg.values()); n_fam0=len(famN); n_fg0=len(fgt)
        bg={g:[r for r in bg[g] if _pair_ok(r[1],r[2],r[3],r[4])] for g in bg}
        famN={k:v for k,v in famN.items() if _pair_ok(v[1],v[2],v[3],v[4])}
        fgt=[t for t in fgt if _pair_ok(t["si"],t["idx"],t["si"],t["idx"])]
        print(f"[merge] ARM RULE {RULE}: bg survivors {n_bg0}->{sum(len(v) for v in bg.values())}, net-families {n_fam0}->{len(famN)}, zero-lag triggers {n_fg0}->{len(fgt)}",flush=True)
    if HOOK and _HOOK_MISS:                                   # every bank row that survives the arm rule must carry the substituted statistic
        _left=[r for g in bg for r in bg[g] if (int(r[1]),int(r[2]),int(r[3]),int(r[4])) in _HOOK_MISS]
        if _left: raise SystemExit(f"[merge] STAT HOOK: {len(_left)} bank rows after the arm rule have no substituted statistic (e.g. {_left[0][1:]})")
        print(f"[merge] STAT HOOK: all {sum(len(v) for v in bg.values())} surviving bank rows carry the substituted statistic",flush=True)
    MTAG=os.environ.get("SM_MERGE_TAG","")
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
    if NETSIG_FLOOR>0:                                # cap net-candidates to the loudest SM_FG_MAXNET
        _maxnet=int(os.environ.get("SM_FG_MAXNET","400"))   # MIRRORS the approved foreground cap driver_blindscan.py:292-295
        _netpass=sorted((i for i,t in enumerate(fgt) if t["net"]>=NETSIG_FLOOR), key=lambda i:-fgt[i]["net"])
        for i in _netpass[:_maxnet]: cand_idx.add(i)
    for i,t in enumerate(fgt):
        for ev,(esi,ew) in evloc.items():
            if t["si"]==esi and abs(t["idx"]-ew)<=evrad: t["matches_known"]=ev; cand_idx.add(i)
    # ---- CNN precompute (bit-identical to lazy per-pair scoring), mirrors driver_blindscan STAGE-2 ----
    need_pairs=set()
    for i in cand_idx:
        if fgt[i].get("far_lr_per_yr") is not None: need_pairs.update(fgt[i]["_louder"])
    if NETSIG_FLOOR>0:                                # net-sigma channel: family r is needed iff SOME same-fold
        min_net={0:float("inf"),1:float("inf")}        # net-candidate has net<=r[0]  <=>  r[0]>=min(candidate nets in that fold).
        for i in cand_idx:                             # O(cand+fam) rewrite of the old O(cand*fam) nested scan
            if fgt[i]["net"]>=NETSIG_FLOOR:            # (verified bit-identical to the nested loop by 2 reviewers, 2026-07-02).
                g=fgt[i]["fold"]; min_net[g]=min(min_net[g],fgt[i]["net"])
        for r in famN.values():
            if r[0]>=min_net[fold[r[1]]]: need_pairs.add((r[1],r[2],r[3],r[4]))
    # ---- SLURM CNN sharding (SM_CNN_SHARD="k:N" worker / SM_CNN_CACHE=1 consumer). need_pairs is built by the
    #      SAME deterministic upstream code in every job -> sorted(need_pairs)[k::N] is a true partition. ----
    _cshard=os.environ.get("SM_CNN_SHARD")
    if _cshard:                                       # WORKER: score my strided slice, save shard, run NOTHING downstream
        k,N=(int(x) for x in _cshard.split(":"))
        # SM_CNN_TOPUP=1 (2026-10-07): score ONLY the need_pairs absent from the cnn_shard_*.npz already in OUT and
        # write them to cnn_shard_topup<k>.npz (picked up by the consumer glob). Needed when a re-pass on a masked
        # bank promotes new family representatives (HW-inj re-pass: 836/189881 o3b, 257/483519 o3a uncached pairs).
        _topup=os.environ.get("SM_CNN_TOPUP")=="1"
        _fsh=f"{OUT}/cnn_shard_topup{k}.npz" if _topup else f"{OUT}/cnn_shard_{k}.npz"
        if os.path.exists(_fsh):
            print(f"[merge] CNN shard {k}/{N}: {_fsh} already exists -- skipping (resumable)",flush=True); raise SystemExit(0)
        # CONTIGUOUS slice (not strided): plist is sorted by (ai,iH,...) so a contiguous block shares H1 keys
        # -> each worker builds ~|H|/N H1 mags instead of ~all of them (16x less duplicated QT work), and the
        # retained-H1 block fits the budget whole -> each L1 key is built exactly once per worker.
        plist=sorted(need_pairs)
        if _topup:
            _have=set()
            for _f in sorted(glob.glob(f"{OUT}/cnn_shard_*.npz")):
                _P=np.load(_f)["pairs"]; _have.update((int(a),int(h),int(b),int(l)) for a,h,b,l in _P)
            plist=[q for q in plist if q not in _have]
            print(f"[merge] CNN TOP-UP: {len(_have)} pairs already cached, {len(need_pairs)} needed, {len(plist)} to score",flush=True)
        mine=plist[(k*len(plist))//N:((k+1)*len(plist))//N]; _t0=time.time()
        # NOTE (2026-07-03): a per-sub-block checkpoint variant was tried and REVERTED -- repeated precompute_cnn
        # calls inflate the QT forkserver pool's resident memory past the 186 GB per-GPU cgroup share on this
        # partition (OOM-killed shards 3/5/8). The single-call path below fit in 186 GB (shard 6 completed).
        # For O3b, checkpoint INSIDE precompute_cnn's existing streaming loop (no extra pool churn) + lower
        # SM_PRECOMPUTE_MAXGB, not via repeated top-level calls. The 10h walltime already prevents timeouts here.
        _ckpt=f"{OUT}/_ckpt_shard_{k}" if os.environ.get("SM_BLOCK_CKPT") else None   # default-off; set SM_BLOCK_CKPT=1 to enable per-block resume
        cache=PB.precompute_cnn(mine,segs,B.cpipe,B.carm,B.cnet,B.lmnet,B._win,ckpt_dir=_ckpt)
        np.savez(_fsh,pairs=np.array(mine,dtype=np.int64).reshape(len(mine),4),
                 hm=np.array([cache[p][0] for p in mine],dtype=np.float64),
                 lm=np.array([cache[p][1] for p in mine],dtype=np.float64))
        if _ckpt and os.path.isdir(_ckpt):                # shard fully written -> drop the now-redundant per-block checkpoints
            import shutil; shutil.rmtree(_ckpt,ignore_errors=True)
        print(f"[merge] CNN shard {k}/{N}: scored {len(mine)}/{len(plist)} pairs in {time.time()-_t0:.0f}s -> {_fsh}",flush=True)
        raise SystemExit(0)
    if os.environ.get("SM_CNN_CACHE")=="1":           # CONSUMER: assemble cnn_cache from shard files; NO inline scoring
        cnn_cache={}
        for _f in sorted(glob.glob(f"{OUT}/cnn_shard_*.npz")):
            _d=np.load(_f); _P=_d["pairs"]; _H=_d["hm"]; _L=_d["lm"]
            for j in range(len(_P)):
                cnn_cache[(int(_P[j,0]),int(_P[j,1]),int(_P[j,2]),int(_P[j,3]))]=(float(_H[j]),float(_L[j]))
        _miss=[p for p in need_pairs if p not in cnn_cache]
        print(f"[merge] STAGE-2 CNN cache: {len(cnn_cache)} shard-scored pairs loaded, {len(need_pairs)} needed",flush=True)
        if _miss:                                     # a partial cache must NEVER silently produce a wrong FAR
            raise SystemExit(f"[merge] FATAL: {len(_miss)}/{len(need_pairs)} need_pairs missing from CNN shard cache "
                             f"(e.g. {sorted(_miss)[0]}) -- re-run the missing SM_CNN_SHARD workers before merging.")
    else:                                             # DEFAULT (no env): inline precompute, byte-identical to before
        print(f"[merge] STAGE-2 CNN precompute STARTING on {len(need_pairs)} louder-bg pairs (this is the slow step)...",flush=True)
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
        if S3HOOK:                                           # A6(c): loglr channel gated and ranked by S3 alone; June values kept for the net-sigma channel
            t["s3"]=_s3t.get((t["seg"],round(float(t["gps"]),1)))
            t["is_glitch_lr"]=bool(t["s3"] is None or t["s3"]<S3_THR)
            if t.get("far_lr_per_yr") is None or t["is_glitch_lr"]:
                t["n_louder_L1fam_cnn"]=None; t["far_cnn_per_yr"]=None; t["far_cnn_ul90"]=None
                t["far_lr_hm"]=t["far_lr_lm"]=t["far_lr_perarm"]=t["far_lr_perarm_ul90"]=None
            else:
                _miss=[(ai,iH,bi,iL) for (ai,iH,bi,iL) in t["_louder"] if (segs[ai]["name"],int(iH),segs[bi]["name"],int(iL)) not in _s3b]
                if _miss: raise SystemExit(f"[merge] S3 HOOK: {len(_miss)} louder bank pairs without an S3 score (e.g. {_miss[0]})")
                _sc=lambda ai,iH,bi,iL:_s3b[(segs[ai]["name"],int(iH),segs[bi]["name"],int(iL))]
                fams=set(L1key(bi,iL) for (ai,iH,bi,iL) in t["_louder"] if _sc(ai,iH,bi,iL)>=S3_THR)
                t["n_louder_L1fam_cnn"]=len(fams); t["far_cnn_per_yr"]=len(fams)/flt if flt>0 else None
                t["far_cnn_ul90"]=float(chi2.ppf(0.90,2*(len(fams)+1))/2/flt) if flt>0 else None
                si=t["si"]; fam_s3=set(L1key(bi,iL) for (ai,iH,bi,iL) in t["_louder"] if ai!=si and bi!=si and _sc(ai,iH,bi,iL)>=t["s3"])
                t["far_lr_hm"]=t["far_lr_lm"]=len(fam_s3)/flt if flt>0 else None
                t["far_lr_perarm"]=t["far_lr_hm"]; t["far_lr_perarm_ul90"]=_ul90(len(fam_s3),flt)   # ONE arm: no x2 arm-choice factor
        print(f"  cand {ci+1}/{len(cand_idx)} {t['seg']} loglr={_f(t.get('loglr'),'.2f')} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')} ({t['kept_by']})"
              f"{' GLITCH' if t['is_glitch'] else ''} FAR_LR={_f(t.get('far_lr_per_yr'),'.3g')} -> FAR_CNN={_f(t.get('far_cnn_per_yr'),'.3g')}/yr {t.get('matches_known','')}",flush=True)

    # ===== STAGE-1b NET-SIGMA OR-CHANNEL =====
    if NETSIG_FLOOR>0:
        fams=sorted(famN.values(),key=lambda r:-r[0])
        netcands=[i for i in cand_idx if fgt[i]["net"]>=NETSIG_FLOOR]
        if NETSIG_LM_MIN>0:
            _drop=[i for i in netcands if float(fgt[i].get("cnn_lm") or 0.0)<=NETSIG_LM_MIN]
            netcands=[i for i in netcands if i not in set(_drop)]
            for i in _drop: fgt[i]["netsig_lm_vetoed"]=True
            print(f"[merge] NET-SIGMA LM rule: cnn_lm>{NETSIG_LM_MIN} drops {len(_drop)} fg net-candidates",flush=True)
            for i in sorted(_drop,key=lambda i:-fgt[i]["net"])[:20]:
                t=fgt[i]
                print(f"    LM-VETO {t['seg']} net={t['net']:.2f} loglr={t['loglr']:.2f} "
                      f"HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.4f')} {t.get('matches_known','')}",flush=True)
        print(f"[merge] NET-SIGMA channel: floor={NETSIG_FLOOR}; {len(famN)} bg L1-families, {len(netcands)} fg net-candidates",flush=True)
        # 2026-09-20 VECTORISED, EXACT: the old per-candidate Python scan over ALL famN families was O(n_cand x n_fam)
        # (21509 x 2.4M under SM_ARM_RULE=none) -> the 12 h merge timed out at ~25% of this channel (pilot 1742615,
        # coh 1747214). Same arithmetic on numpy arrays: prefix of the descending-net list (r[0]>=t.net, ties included),
        # per-fold cumulative counts of CNN-passing families, per-arm counts on the prefix slice. Only families with
        # net >= the smallest candidate net of their fold are ever compared (== need_pairs), so the cache lookups match.
        _t0=time.time()
        F_net=np.array([r[0] for r in fams],dtype=np.float64)
        F_ai=np.array([r[1] for r in fams],dtype=np.int64); F_bi=np.array([r[3] for r in fams],dtype=np.int64)
        F_fold=np.array([fold[r[1]] for r in fams],dtype=np.int64)
        _minnet={g:min([float(fgt[i]["net"]) for i in netcands if fgt[i]["fold"]==g],default=float("inf")) for g in (0,1)}
        _need=np.array([r[0]>=_minnet[fold[r[1]]] for r in fams],dtype=bool)
        F_hm=np.full(len(fams),-np.inf); F_lm=np.full(len(fams),-np.inf)      # -inf = never within any candidate's prefix+fold
        for j in np.flatnonzero(_need):
            r=fams[j]; hm,lm=surv_cnn_pair(r[1],r[2],r[3],r[4]); F_hm[j]=hm; F_lm[j]=lm
        F_pass=np.maximum(F_hm,F_lm)>GLITCH_THRESH
        if NETSIG_LM_MIN>0:                           # same rule on the FAR-setting background
            _b4=int(F_pass.sum()); F_pass&=(F_lm>NETSIG_LM_MIN)
            print(f"[merge] NET-SIGMA LM rule on background: {_b4} -> {int(F_pass.sum())} "
                  f"FAR-setting families ({100*(1-F_pass.sum()/max(_b4,1)):.1f}% removed)",flush=True)
        _cs={g:np.concatenate([[0],np.cumsum((F_fold==g)&F_pass)]) for g in (0,1)}   # _cs[g][k] = CNN-passing fold-g fams among the first k
        _negnet=-F_net                                                                    # ascending for searchsorted
        print(f"[merge] NET-SIGMA arrays: {int(_need.sum())} fams within candidate reach; setup {time.time()-_t0:.0f}s",flush=True)
        for i in sorted(netcands,key=lambda i:-fgt[i]["net"]):
            t=fgt[i]; g=t["fold"]; flt=far_live[g]
            k=int(np.searchsorted(_negnet,-float(t["net"]),side="right"))                 # fams with r[0]>=t.net
            nk=int(_cs[g][k])
            t["n_louder_netfam_cnn"]=nk
            t["far_net_per_yr"]=nk/flt if flt>0 else None
            t["far_net_ul90"]=float(chi2.ppf(0.90,2*(nk+1))/2/flt) if flt>0 else None
            # --- PER-ARM RANK (net-sigma channel): louder-in-net famN families, by EACH arm, self-seg guarded ---
            si=t["si"]
            _m=(F_fold[:k]==g)&(F_ai[:k]!=si)&(F_bi[:k]!=si)
            if NETSIG_LM_MIN>0: _m&=(F_lm[:k]>NETSIG_LM_MIN)   # per-arm rank over the admitted population
            nhm=int(np.count_nonzero(_m&(F_hm[:k]>=np.float64(t["cnn_hm"]))))
            nlm=int(np.count_nonzero(_m&(F_lm[:k]>=np.float64(t["cnn_lm"]))))
            t["far_net_hm"]=nhm/flt if flt>0 else None; t["far_net_lm"]=nlm/flt if flt>0 else None
            t["far_net_perarm"]=_perarm(t["far_net_hm"],t["far_net_lm"])
            t["far_net_perarm_ul90"]=_perarm(_ul90(nhm,flt),_ul90(nlm,flt))
            print(f"  net-cand {t['seg']} net={t['net']:.2f} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')}"
                  f"{' GLITCH' if t.get('is_glitch') else ''} -> FAR_net={_f(t.get('far_net_per_yr'),'.3g')}/yr (UL90 {_f(t.get('far_net_ul90'),'.3g')}) {t.get('matches_known','')}",flush=True)

    if S3HOOK:
        for i in cand_idx:
            t=fgt[i]
            if t.get("is_glitch"):                          # June gate failed -> no net-sigma channel for this trigger
                for k_ in ("far_net_per_yr","far_net_ul90","far_net_hm","far_net_lm","far_net_perarm","far_net_perarm_ul90"): t[k_]=None
        print(f"[merge] S3 HOOK: loglr channel open for {sum(1 for i in cand_idx if not fgt[i].get('is_glitch_lr',True))} candidates, net-sigma channel open for {sum(1 for i in cand_idx if not fgt[i].get('is_glitch'))}",flush=True)
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
              open(f"{OUT}/survivors_bg{MTAG}.json","w"),indent=2)
    print(f"[merge] saved {len(bg_surv)} CNN-scored bg survivors ({n_pass} pass CNN gate) -> survivors_bg.json",flush=True)

    for t in fgt: t.pop("_louder",None)
    json.dump(dict(net_cut=NET_CUT,cluster_s=CLUSTER_S,n_segments=nseg,far_live_yr=far_live,netsig_floor=NETSIG_FLOOR,netsig_lm_min=NETSIG_LM_MIN,
                   n_segments_requested=len(M.SEG["segments"]),partial=bool(os.environ.get("SM_ALLOW_PARTIAL")),
                   n_triggers=len(fgt),n_cnn_vetoed=len(cand_idx),triggers=fgt),
              open(f"{OUT}/blindscan{MTAG}.json","w"),indent=2)
    DET_FAR=float(os.environ.get("DET_FAR","1.0"))
    # Gate 4 (PREREGISTRATION §3): detection requires BOTH the point per-arm FAR AND its 90% Poisson UL < DET_FAR.
    dets=[t for t in fgt if (S3HOOK or not t.get("is_glitch"))
          and best_far(t) is not None and best_far(t)<DET_FAR
          and best_ul90(t) is not None and best_ul90(t)<DET_FAR]
    dets.sort(key=lambda d:best_far(d))
    # ---- LOCAL-ASD CONSISTENCY VETO (veto #1, ON by default; SM_ASD_VETO=0 disables -> byte-identical pre-veto) ----
    # A candidate must stay significant when its tile is re-whitened with a LOCAL +/-64s ASD, else it
    # is an ASD-mismatch artifact (the fixed reference ASD mis-estimated that segment's noise floor).
    # Validated on the O3b/O3a-ASD run: removes net27/net23 (sigma collapse) + net16 (CNN flips), keeps
    # all 4 real events. Post-processing on the candidate list only -> survivors keep their reference FAR.
    # ADOPTED 2026-07-06: inject_retention signal-retention >=97% at SNR>=8 (med dcnn~0) confirms it is
    # signal-safe, so the validated veto is now the default. SM_ASD_VETO=0 reproduces the frozen pre-veto runs.
    if os.environ.get("SM_ASD_VETO","1")!="0":
        import asd_consistency as AV
        # ---- CONDITIONAL FEATURE CLAUSE (m2/asdveto/DECISIONS_ASDVETO.md, 2026-10-07): the CNN / S3 local-consistency clause is a
        # feature-consistency test, not an ASD test; it applies to a run only if the run's plateau livetime fraction
        # (m2/asdveto/plateau_metric.py, computed from the sigma streams before any merge) exceeds SM_ASD_PLATEAU_THR.
        # The net clause (net_loc < floor, net-sigma channel) is unconditional. SM_ASD_CNN_CLAUSE=on (default) is byte-identical to before.
        _cc=os.environ.get("SM_ASD_CNN_CLAUSE","on")
        if _cc=="auto":
            _pj=os.environ.get("SM_ASD_PLATEAU_JSON",f"{MADGRAV_ROOT}/m2/asdveto/plateau_{os.environ['SM_RUN']}.json")
            _pm=json.load(open(_pj)); _pthr=float(os.environ.get("SM_ASD_PLATEAU_THR","0.04")); CNN_CLAUSE_ON=bool(_pm["max_frac"]>_pthr)
            print(f"  [asd-veto] feature clause {'ON' if CNN_CLAUSE_ON else 'OFF'}: plateau max_frac {_pm['max_frac']:.4f} "
                  f"(H1 {_pm['H1']['plateau_livetime_frac']:.4f}, L1 {_pm['L1']['plateau_livetime_frac']:.4f}) vs thr {_pthr} [{_pj}]",flush=True)
        else:
            CNN_CLAUSE_ON=(_cc!="off"); print(f"  [asd-veto] feature clause {'ON' if CNN_CLAUSE_ON else 'OFF'} (SM_ASD_CNN_CLAUSE={_cc})",flush=True)
        kept=[]
        for d in dets:
            d.setdefault("idx", None)
            if S3HOOK and _chan(d)=="loglr":                 # A6(c): the local-ASD veto tests S3's stability for loglr-channel detections
                _sv=(B.cnet,B.lmnet,B.FLO,B.FHI,B.LFLO,B.LFHI); _e=_S3ENS[1-_s3_fold_of_gps(float(d["gps"]))]
                B.cnet=lambda:_e; B.lmnet=lambda:_e; B.FLO,B.FHI=20.0,300.0; B.LFLO,B.LFHI=20.0,300.0
                try: keep,r=AV.survives({**d,"channel":"loglr"},floor=NETSIG_FLOOR,glitch_thresh=S3_THR)
                finally: B.cnet,B.lmnet,B.FLO,B.FHI,B.LFLO,B.LFHI=_sv
                r["s3_veto"]=True
            else:
                keep,r=AV.survives({**d,"channel":_chan(d)},floor=NETSIG_FLOOR)
            if not CNN_CLAUSE_ON and not keep and r["reason"]=="cnn":
                keep=not (_chan(d)=="net-sigma" and r["net_loc"]<NETSIG_FLOOR)
                r["reason"]="keep(feature clause off)" if keep else "net<floor"; r["keep"]=keep
            r["feature_clause_on"]=CNN_CLAUSE_ON
            d["asd_veto"]=r
            print(f"  [asd-veto] {str(d.get('matches_known') or d['seg']):22s} {d['gps']:12.1f} ({_chan(d)}) "
                  f"net {d['net']:.2f}->{r['net_loc']:.2f} cnn {d.get('cnn_hm',0):.2f}/{d.get('cnn_lm',0):.2f}->"
                  f"{r['hm_loc']:.2f}/{r['lm_loc']:.2f}  {'KEEP' if keep else 'VETO('+r['reason']+')'}",flush=True)
            if keep: kept.append(d)
        print(f"  [asd-veto] {len(kept)}/{len(dets)} survive local-ASD consistency",flush=True)
        dets=kept
    json.dump([dict(seg=d["seg"],gps=d["gps"],net=d["net"],loglr=d["loglr"],cnn_hm=d.get("cnn_hm"),cnn_lm=d.get("cnn_lm"),
                    kept_by=d.get("kept_by"),far_cnn_per_yr=d.get("far_cnn_per_yr"),far_cnn_ul90=d.get("far_cnn_ul90"),
                    far_net_per_yr=d.get("far_net_per_yr"),far_net_ul90=d.get("far_net_ul90"),
                    far_lr_perarm=d.get("far_lr_perarm"),far_lr_perarm_ul90=d.get("far_lr_perarm_ul90"),
                    far_net_perarm=d.get("far_net_perarm"),far_net_perarm_ul90=d.get("far_net_perarm_ul90"),
                    far_lr_hm=d.get("far_lr_hm"),far_lr_lm=d.get("far_lr_lm"),
                    far_net_hm=d.get("far_net_hm"),far_net_lm=d.get("far_net_lm"),
                    best_far=best_far(d),best_ul90=best_ul90(d),stat=("per-arm" if PERARM else "OR-veto"),s3=d.get("s3"),is_glitch_lr=d.get("is_glitch_lr"),
                    channel=_chan(d),matches_known=d.get("matches_known","")) for d in dets],
              open(f"{OUT}/detections{MTAG}.json","w"),indent=2)
    print(f"\n=== DETECTIONS (point FAR<{DET_FAR}/yr AND UL90<{DET_FAR}/yr; stat={'per-arm' if PERARM else 'OR-veto'}): {len(dets)} ===",flush=True)
    for d in dets:
        print(f"  {str(d.get('matches_known') or d['seg']):22s} {d['gps']:12.1f} net={d['net']:.2f} loglr={d['loglr']:.2f} "
              f"best_far={best_far(d):.3g} UL90={best_ul90(d):.3g} ({_chan(d)})",flush=True)

if __name__=="__main__": main()
