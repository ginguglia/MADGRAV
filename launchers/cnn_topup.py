"""Top up the CNN shard cache for a RELAXED arm rule, so the re-merge does not FATAL.

The merge under SM_CNN_CACHE=1 refuses to run if any need_pair is absent from the cnn_shard_*.npz
cache ("a partial cache must NEVER silently produce a wrong FAR"). The cache was built for the
arm -2.05 population; relaxing to SM_ARM_TOPUP admits background pairs that were never scored
(measured 2026-09-22: net-families 0% missing in all four runs, bg survivors 0.61-9.69% missing,
i.e. 206-1852 pairs per run -- 0.06% of the 3.2M the original CNN stage scored).

This scores EVERY pair of the relaxed population that is missing (bg survivors AND net-families,
not just the newly-admitted band) and writes them as an extra cache shard. The merge consumer
globs cnn_shard_*.npz, so an extra index is picked up with no other change.

Run: SM_RUN_SEL=o4a SM_ARM_TOPUP=-3.00 sbatch cnn_topup.sbatch     (needs GPU)
"""
import os, sys, glob, time, numpy as np
ROOT=os.environ["MADGRAV_ROOT"]; sys.path.insert(0,f"{ROOT}/search_mode")
import _pb_shard_common as C, driver_blindscan as B, _pb_cnn_precompute as PB
SH=os.environ["SM_SHARD_DIR"]; OUT=os.environ["SM_OUT"]; RUN=os.environ["SM_RUN"]
ARM=float(os.environ.get("SM_ARM_TOPUP","-3.00"))
IDX=int(os.environ.get("SM_TOPUP_IDX","900"))
FSH=f"{OUT}/cnn_shard_{IDX}.npz"

# spawn-safe: precompute_cnn starts 'spawn' workers that re-import this module; without the guard every
# worker re-ran build_setup + model load on the GPU and died OOM in a restart loop (2026-09-22, jobs 1757608-11).
def main():
    if os.path.exists(FSH): print(f"[topup] {FSH} exists -- nothing to do"); raise SystemExit(0)
    S=C.build_setup(); segs=S["segs"]
    def v(ai,iH,bi,iL):
        A,Bs=segs[ai],segs[bi]
        return float(A["gH"][iH]) if float(A["sH"][iH])>=float(Bs["sL"][iL]) else float(Bs["gL"][iL])

    cache=set()
    for f in sorted(glob.glob(f"{OUT}/cnn_shard_*.npz")):
        P=np.load(f)["pairs"]
        for j in range(len(P)): cache.add((int(P[j,0]),int(P[j,1]),int(P[j,2]),int(P[j,3])))
    print(f"[topup:{RUN}] cache holds {len(cache)} pairs; target arm rule max:{ARM}",flush=True)

    pop=set(); fam={}
    for f in sorted(glob.glob(f"{SH}/shard_*.npz")):
        p=np.load(f,allow_pickle=False)
        for key in ("bg0","bg1"):
            for row in p[key]:
                q=(int(row[1]),int(row[2]),int(row[3]),int(row[4]))
                if v(*q)>=ARM: pop.add(q)
        for kk,vv in zip(p["fam_keys"],p["fam_vals"]):
            k=(int(kk[0]),(int(kk[1]),int(kk[2]))); net=float(vv[0])
            if k not in fam or net>fam[k][0]: fam[k]=(net,int(vv[1]),int(vv[2]),int(vv[3]),int(vv[4]))
    for r in fam.values():
        q=(r[1],r[2],r[3],r[4])
        if v(*q)>=ARM: pop.add(q)
    missing=sorted(pop-cache)
    print(f"[topup:{RUN}] arm>={ARM} population {len(pop)} pairs; MISSING {len(missing)} -> scoring",flush=True)
    if not missing:
        np.savez(FSH,pairs=np.zeros((0,4),dtype=np.int64),hm=np.zeros(0),lm=np.zeros(0))
        print(f"[topup:{RUN}] nothing missing; wrote empty {FSH}"); raise SystemExit(0)
    t0=time.time()
    _ck=os.environ.get("SM_TOPUP_CKPT","") or None   # per-H-block checkpoint dir (resume after a timeout); default off = unchanged behaviour
    if _ck: print(f"[topup:{RUN}] block checkpoints -> {_ck}",flush=True)
    c=PB.precompute_cnn(missing,segs,B.cpipe,B.carm,B.cnet,B.lmnet,B._win,ckpt_dir=_ck)
    np.savez(FSH,pairs=np.array(missing,dtype=np.int64).reshape(len(missing),4),
             hm=np.array([c[p][0] for p in missing],dtype=np.float64),
             lm=np.array([c[p][1] for p in missing],dtype=np.float64))
    print(f"[topup:{RUN}] scored {len(missing)} pairs in {time.time()-t0:.0f}s -> {FSH}",flush=True)

if __name__=="__main__": main()
