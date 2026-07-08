"""Sensitive-volume / detection-efficiency vs MASS for the O3a search (Aframe's headline metric).
RE-ANALYSIS of the existing importance-sampling injection campaign (pastro_inj_O3a.npz, SNR drawn ~rho^-4 =
uniform-in-Euclidean-volume) against the O3a LR background. For each injection: detected = gated(net>=4)
AND loglr >= thr(FAR). Per Mtot bin: efficiency eff = N_det/N_inj (= volume-averaged detectability, since the
SNR draw IS the volume measure), with Clopper-Pearson 95% CI. Absolute V_sens(m) = eff(m) * V_inj(m) needs the
per-mass horizon (TODO add from bank optimal-SNR); this first pass reports the efficiency shape vs mass.

Statistic = loglr channel (the records lack 2-det cnn_hm/lm -> per-arm VT needs a CNN re-score, later).
Background = LR time-slide (p0/lr_bg_features_sig4.npz); refines with the search-mode 56-seg/full-485 bg.
"""
import json, numpy as np
from scipy.stats import beta as betadist
import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR=f"{ROOT}/lr_cascade"; YR=3.1557e7
m=json.load(open(f"{LR}/p4/lr_model_v5.json")); mu,sd,be=np.array(m["mu"]),np.array(m["sd"]),np.array(m["beta"]); GCLIP=float(m["g_clip"])
def gate(g,s): return np.clip(g,-GCLIP,GCLIP)*np.clip(np.asarray(s)/3.0,0,1)
def loglr(X): return np.column_stack([np.ones(len(X)),(X-mu)/sd])@be

# ---- thr(FAR) from the O3a LR background ----
bg=np.load(f"{LR}/p0/lr_bg_features_sig4.npz"); sg=np.load(f"{LR}/p1/segment_g.npz"); I,J=bg["I"],bg["J"]
FB=np.column_stack([bg["sigma_H1"],bg["sigma_L1"],bg["coh_BA"],bg["centroid_H1"],bg["centroid_L1"],
                    gate(sg["g_H1"][I],bg["sigma_H1"]),gate(sg["g_L1"][J],bg["sigma_L1"])]).astype(np.float64)
lt=float(bg["bg_livetime_s"])/YR
lr_bg=np.sort(loglr(FB))[::-1]; f=np.arange(1,len(lr_bg)+1)/lt
def thr_at(far):
    sel=lr_bg[f<=far]; return float(sel[-1]) if len(sel) else float(lr_bg[0])
thr1=thr_at(1.0); thr_floor=float(lr_bg[0])  # floor = loudest bg (FAR ~ 1/livetime)
print(f"O3a LR background: {len(lr_bg)} families, livetime {lt:.1f} yr")
print(f"  thr(FAR=1/yr)   loglr >= {thr1:.2f}")
print(f"  thr(FAR floor)  loglr >= {thr_floor:.2f}  (FAR ~ {1/lt:.3f}/yr)\n")

# ---- injections (volume-weighted) ----
inj=np.load(f"{LR}/p4/pastro_inj_O3a.npz")
mtot=inj["mtot"]; llr=inj["loglr"]; gated=inj["gated"]; snr=inj["snr_net"]
print(f"injections: {len(mtot)} (mtot {mtot.min():.0f}-{mtot.max():.0f}, snr {snr.min():.1f}-{snr.max():.1f})\n")

BINS=[(10,30),(30,50),(50,80),(80,120),(120,180),(180,250),(250,400)]
def cp(k,n):  # Clopper-Pearson 95%
    lo=0.0 if k==0 else betadist.ppf(0.025,k,n-k+1); hi=1.0 if k==n else betadist.ppf(0.975,k+1,n-k); return lo,hi
print(f"{'Mtot bin':>12} {'N_inj':>7} | {'eff@1/yr':>14} {'95% CI':>16} | {'eff@floor':>12}")
print("-"*72)
rows=[]
for lo,hi in BINS:
    inb=(mtot>=lo)&(mtot<hi); n=int(inb.sum())
    if n==0: continue
    det1=int((gated&(llr>=thr1)&inb).sum()); detf=int((gated&(llr>=thr_floor)&inb).sum())
    e1=det1/n; ef=detf/n; clo,chi=cp(det1,n)
    print(f"{f'[{lo},{hi})':>12} {n:>7} | {e1*100:12.1f}% {f'[{clo*100:.1f},{chi*100:.1f}]':>16} | {ef*100:10.1f}%")
    rows.append(dict(bin=[lo,hi],n=n,eff_1yr=e1,ci95=[clo,chi],eff_floor=ef))
np.savez(f"{LR}/p4/vt_vs_mass_O3a.npz",bins=np.array(BINS),
         **{k:np.array([r[k] for r in rows]) for k in ('n','eff_1yr','eff_floor')})
json.dump({"run":"O3a","statistic":"loglr-channel","bg_livetime_yr":lt,"thr_1yr":thr1,"thr_floor":thr_floor,
           "rows":rows},open(f"{LR}/p4/vt_vs_mass_O3a.json","w"),indent=1)
print(f"\nwrote vt_vs_mass_O3a.json / .npz  (eff(Mtot) at FAR<=1/yr; absolute V_sens = eff * V_inj(m) TODO)")
