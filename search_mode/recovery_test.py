#!/usr/bin/env python
"""Ceiling signal-safety test. Uses the EXACT fitted LR model (dumped by driver_blindscan SM_DUMP_MODEL)
and the injection bank. For each injection: true loglr L (at its real coherence) and the prefilter upper
bound up(C) = loglr with coh forced to ceiling C. A *detectable* injection (L>floor) is LOST at ceiling C
iff up(C)<=floor (its real coh exceeds C so the 'upper bound' underestimates it). Zero lost => C is
empirically signal-safe. Pure numpy; coherence is precomputed in the inj files (no GPU needed)."""
import numpy as np, glob, os
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

GCLIP=6.0; FLOORS=[4.5,5.0]; CEILINGS=[0.43,0.50,0.614,0.65,0.85]
ROOT=MADGRAV_ROOT + "/search_mode"
INJ=MADGRAV_SCRATCH + "/inj_out_o3a_56"

def gate(g,s): return np.clip(g,-GCLIP,GCLIP)*np.clip(np.asarray(s)/3.0,0,1)
def feats(sH,sL,coh,cH,cL,gH,gL): return np.column_stack([sH,sL,coh,cH,cL,gate(gH,sH),gate(gL,sL)])
def loglr(mu,sd,beta,F): return beta[0]+((np.asarray(F,float)-mu)/sd)@beta[1:]

d=np.load(f"{ROOT}/recovery_model_5seg.npz")
mdl={0:(d["mu0"],d["sd0"],d["be0"]), 1:(d["mu1"],d["sd1"],d["be1"])}     # mdl[g]=fit on fold g; a fold-g inj uses mdl[1-g]
floor_dump=float(d["floor"])
print(f"model loaded; dump floor={floor_dump}")
for g in (0,1):
    mu,sd,be=mdl[g]; print(f"  fold{g} model: coh coeff beta[3]={be[3]:.3f}  sd_coh={sd[2]:.4f}  -> dloglr/dcoh = {be[3]/sd[2]:.2f} per unit coh")

# 5-seg avail order -> folds [0,1,0,1,0]; map each inj file (event) to its fold
seg_order=["GW190521_074359","GW190412","GW190828_063405","GW190408_181802","GW190521"]
ev_fold={ev:(i%2) for i,ev in enumerate(seg_order)}

rows=[]   # (net_snr, mtot, coh_true, L, fold)
for ev,g in ev_fold.items():
    za=np.load(f"{INJ}/{ev}_inj.npz")
    F=feats(za["sigH"],za["sigL"],za["coh"],za["cenH"],za["cenL"],za["gH"],za["gL"])
    mu,sd,be=mdl[1-g]                              # cross-fit: fold-g event scored by mdl[1-g]
    L=loglr(mu,sd,be,F)
    for k in range(len(L)):
        rows.append((float(za["net_snr"][k]),float(za["mtot"][k]),float(za["coh"][k]),float(L[k]),g))
R=np.array(rows); snr,mtot,coh,L,fold=R[:,0],R[:,1],R[:,2],R[:,3],R[:,4].astype(int)
print(f"\n{len(R)} injections | true loglr range [{L.min():.2f},{L.max():.2f}]")

def up_at(C):                                       # prefilter upper bound for every inj at ceiling C
    up=np.empty(len(R))
    for g in (0,1):
        m=fold==g; mu,sd,be=mdl[1-g]
        za_idx=m
        # rebuild feats for these rows with coh replaced by C
    # simpler: recompute per event with coh=C
    out=np.empty(len(R)); off=0
    for ev,g in ev_fold.items():
        za=np.load(f"{INJ}/{ev}_inj.npz"); n=len(za["net_snr"])
        F=feats(za["sigH"],za["sigL"],np.full(n,C),za["cenH"],za["cenL"],za["gH"],za["gL"])
        mu,sd,be=mdl[1-g]; out[off:off+n]=loglr(mu,sd,be,F); off+=n
    return out

print("\n=== CEILING SIGNAL-SAFETY (fraction of detectable injections wrongly pruned) ===")
for floor in FLOORS:
    det=L>floor
    print(f"\n--- floor={floor}  (detectable injections: {det.sum()} of {len(R)}) ---")
    print(f"{'ceiling':>8} {'N_lost':>7} {'recall_loss':>12} {'lost SNR range':>20} {'lost coh range':>18}")
    for C in CEILINGS:
        up=up_at(C); lost=det&(up<=floor)
        if lost.sum():
            sr=f"[{snr[lost].min():.0f},{snr[lost].max():.0f}]"; cr=f"[{coh[lost].min():.3f},{coh[lost].max():.3f}]"
        else: sr="-"; cr="-"
        print(f"{C:>8.3f} {lost.sum():>7} {lost.sum()/max(1,det.sum()):>11.3%} {sr:>20} {cr:>18}")

# also: where do lost injections sit? by SNR bin at the aggressive 0.43 ceiling, floor 4.5
print("\n=== detail @ ceiling 0.43, floor 4.5: lost-injection SNR/mass profile ===")
up=up_at(0.43); det=L>4.5; lost=det&(up<=4.5)
for lo,hi in [(8,12),(12,15),(15,20),(20,30)]:
    s=(snr>=lo)&(snr<hi)
    print(f"  SNR {lo:>2}-{hi:<2}: detectable {int((det&s).sum()):>4}, lost {int((lost&s).sum()):>4}  ({(lost&s).sum()/max(1,(det&s).sum()):.1%})")
