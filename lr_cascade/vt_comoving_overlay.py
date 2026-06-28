"""COMOVING sensitive volume vs Mtot at FAR<=1/yr, MADGRAV vs the O3 BBH-search 6-pipeline benchmark
(Aframe paper arXiv:2403.18661 Fig; slide p22 Image82). PANEL-REVIEW FIXES applied:
 - MADGRAV restricted to NEAR-EQUAL-MASS (q>=0.8) injections+bank, to match the benchmark's equal-mass line
   (q-averaging previously handicapped MADGRAV). Both eff AND <V_max> use the q>=0.8 subset.
 - Full 6-pipeline SPREAD shown as a band (cWB/GstLAL/MBTA/PyCBC-BBH/PyCBC-Broad/Aframe), not just the top 2.
 - Comoving V_max = int dV_c/(1+z) (z-corrected). MADGRAV labelled PROXY (loglr channel + LR-cascade bg).
 - IMBH region: benchmark ML/MF pipelines stop at Mtot=140; dedicated LVK O3 IMBH search is the IMBH comparison.
Run with cern-aigw (numpy+astropy+mpl)."""
import json, numpy as np
from astropy.cosmology import Planck18; import astropy.units as u
from scipy.integrate import cumulative_trapezoid
from scipy.stats import beta as betadist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR=f"{ROOT}/lr_cascade"; YR=3.1557e7; QCUT=0.9
# --- thr(FAR=1/yr) from LR background ---
m=json.load(open(f"{LR}/p4/lr_model_v5.json")); mu,sd,be=np.array(m["mu"]),np.array(m["sd"]),np.array(m["beta"]); GC=float(m["g_clip"])
gate=lambda g,s: np.clip(g,-GC,GC)*np.clip(np.asarray(s)/3.0,0,1); loglr=lambda X: np.column_stack([np.ones(len(X)),(X-mu)/sd])@be
bg=np.load(f"{LR}/p0/lr_bg_features_sig4.npz"); sg=np.load(f"{LR}/p1/segment_g.npz"); I,J=bg["I"],bg["J"]
FB=np.column_stack([bg["sigma_H1"],bg["sigma_L1"],bg["coh_BA"],bg["centroid_H1"],bg["centroid_L1"],
                    gate(sg["g_H1"][I],bg["sigma_H1"]),gate(sg["g_L1"][J],bg["sigma_L1"])]).astype(np.float64)
lt=float(bg["bg_livetime_s"])/YR; lr_bg=np.sort(loglr(FB))[::-1]; thr1=float(lr_bg[min(int(round(lt)),len(lr_bg))-1])
# --- comoving V_max per bank source (q>=QCUT) ---
h=np.load(f"{LR}/p4/vt_horizons_O3a.npz"); hM,hm1,hm2,Dh=h["mtot"],h["m1"],h["m2"],h["Dh_lum"]
zg=np.linspace(1e-4,3,4000); dL=Planck18.luminosity_distance(zg).to(u.Mpc).value
dVdz=Planck18.differential_comoving_volume(zg).to(u.Gpc**3/u.sr).value*4*np.pi
Vmaxz=np.concatenate([[0],cumulative_trapezoid(dVdz/(1+zg),zg)]); Vmax=np.interp(np.interp(Dh,dL,zg),zg,Vmaxz)
hq=np.minimum(hm1,hm2)/np.maximum(hm1,hm2)
# --- injections (q>=QCUT) ---
inj=np.load(f"{LR}/p4/pastro_inj_O3a.npz"); iM,im1,im2,illr,igat=inj["mtot"],inj["m1"],inj["m2"],inj["loglr"],inj["gated"]
iq=np.minimum(im1,im2)/np.maximum(im1,im2)
BINS=[(10,30),(30,50),(50,80),(80,120),(120,180),(180,250),(250,400)]
ctr=[];v=[];vlo=[];vhi=[]
for lo,hi in BINS:
    ib=(iM>=lo)&(iM<hi)&(iq>=QCUT); sb=(hM>=lo)&(hM<hi)&(hq>=QCUT); n=int(ib.sum())
    if n<30 or sb.sum()<10: continue
    det=igat[ib]&(illr[ib]>=thr1); k=int(det.sum()); e=k/n; Vc=float(np.mean(Vmax[sb]))
    clo=0 if k==0 else betadist.ppf(.025,k,n-k+1); chi=1 if k==n else betadist.ppf(.975,k+1,n-k)
    ctr.append((lo+hi)/2); v.append(e*Vc); vlo.append((e-clo)*Vc); vhi.append((chi-e)*Vc)
    print(f"  Mtot[{lo},{hi}) q>={QCUT}: n={n} eff={e*100:.0f}% <Vmax>={Vc:.1f} -> V_sens={e*Vc:.2f} Gpc^3")
# --- benchmark: 6 O3 pipelines at FAR=1/yr (read off slide Image82), component m1=m2 -> Mtot=2m ---
MtotB=2*np.array([10,15,20,25,30,35,40,45,50,55,60,65,70])
P={"cWB":[0.2,0.7,1.3,2.6,4.1,6.0,8.3,10.1,11.8,13.3,14.5,15.2,15.8],
   "GstLAL":[0.5,1.6,3.0,5.0,7.1,10.0,12.5,14.8,16.6,18.4,19.8,20.6,21.0],
   "MBTA":[0.5,1.4,2.3,3.5,4.9,6.7,9.0,11.1,13.0,14.8,16.0,16.4,16.4],
   "PyCBC-BBH":[0.5,1.5,2.5,4.0,7.0,9.5,12.0,14.1,15.6,16.8,17.7,18.0,18.1],
   "PyCBC-Broad":[0.5,1.3,2.2,4.5,5.5,7.1,10.1,10.1,11.1,12.4,13.4,13.7,13.9],
   "Aframe":[0.6,1.7,3.1,5.2,7.4,10.4,13.6,16.5,18.9,20.6,22.0,23.0,23.5]}
bmin=np.min(list(P.values()),0); bmax=np.max(list(P.values()),0)
# --- LVK O3 IMBH search (arXiv:2105.15120 / VT table 2110.01879): PyCBC <VT> Gpc^3 yr at IFAR=100yr,
#     q=1 nonspin. Converted VT->V by /T_O3a (~0.5yr). NB different FAR (IFAR=100yr vs MADGRAV 1/yr). ---
T_O3a=0.5
LVK_M=np.array([120,150,200,400]); LVK_VT=np.array([11.5,12.0,14.8,4.8]); LVK_V=LVK_VT/T_O3a

def make(scale):
    fig,ax=plt.subplots(figsize=(9,6))
    ax.axvspan(140,400,color="orange",alpha=.07)
    ax.fill_between(MtotB,bmin,bmax,color="grey",alpha=.30,label="O3 BBH-search benchmark\n(6 pipelines, FAR=1/yr)")
    ax.plot(MtotB,P["Aframe"],"-",color="purple",lw=1.3,alpha=.8,label="Aframe (ML, top of band)")
    ax.plot(LVK_M,LVK_V,"^--",color="darkred",ms=8,lw=1.5,label="LVK O3 IMBH search (PyCBC,\nIFAR=100yr; VT$\\div$T$_{O3a}$~0.5yr)")
    ax.errorbar(ctr,v,yerr=[vlo,vhi],fmt="o-",color="C0",lw=2,capsize=3,ms=7,label="MADGRAV O3a  q$\geq$0.9  (PROXY: loglr+LR bg)")
    ax.set_yscale(scale); ax.set_xlim(0,400)
    if scale=="log": ax.set_ylim(0.3,70); yi=0.6
    else:            ax.set_ylim(0,44);   yi=5
    ax.text(300,yi,"IMBH regime\n(BBH benchmark stops at 140)\n⚠ FAR MISMATCH: MADGRAV @1/yr vs\nLVK @IFAR=100yr (100× stricter) —\ncompare SHAPE not height",ha="center",fontsize=7.5,color="darkorange")
    ax.set_xlabel("total mass  M$_{tot}$  [M$_\\odot$]"); ax.set_ylabel("comoving sensitive volume @ FAR=1/yr  [Gpc$^3$]")
    ax.set_title("MADGRAV vs O3 BBH-search benchmark — sensitive volume vs total mass")
    ax.legend(loc="upper left",fontsize=8.5,framealpha=.95); ax.grid(alpha=.3,which="both")
    plt.tight_layout(rect=[0,0.055,1,1])
    fig.text(.5,.012,f"MADGRAV: O3a, comoving, equal-mass (q$\geq$0.9) to match the benchmark line; PROXY statistic (loglr, not per-arm) + preliminary LR bg (floor {1/lt:.3f}/yr). "
             "Benchmark from arXiv:2403.18661.",ha="center",fontsize=7,color="dimgrey")
    tag="" if scale=="log" else "_linear"
    for e in ("pdf","png"): fig.savefig(f"{LR}/p4/vt_comoving_overlay_O3a{tag}.{e}",dpi=140,bbox_inches="tight")
    plt.close()
make("log"); make("linear")
json.dump({"qcut":QCUT,"thr1":thr1,"bg_livetime_yr":lt,"mtot_ctr":ctr,"vsens_comoving_gpc3":v,
           "benchmark_Mtot":MtotB.tolist(),"benchmark_min":bmin.tolist(),"benchmark_max":bmax.tolist(),"pipelines":P},
          open(f"{LR}/p4/vt_comoving_overlay_O3a.json","w"),indent=1)
print(f"wrote vt_comoving_overlay_O3a(_linear).pdf/png/json  (q>={QCUT}, 6-pipeline band, thr1={thr1:.2f})")
