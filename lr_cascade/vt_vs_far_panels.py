"""Sensitive volume vs FALSE ALARM RATE at fixed component-mass cells (Aframe slide-p22 Image81 format:
LOG-x FAR, LINEAR-y V_sens Gpc^3, comoving). 4 cells (35-35,35-20,20-20,20-10). For each cell:
eff(FAR)=N_det/N_inj over equal-mass injections; V_sens(FAR)=eff(FAR)*<V_max_comoving>_cell. MADGRAV curve
TRUNCATES at its background floor (0.021/yr) -- it cannot go below, where Aframe reaches 0.01 (refines with
full-485 bg). Aframe curves overlaid (approx read off the slide). Run with cern-aigw (numpy+astropy+mpl)."""
import json, numpy as np
from astropy.cosmology import Planck18; import astropy.units as u
from scipy.integrate import cumulative_trapezoid
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR=f"{ROOT}/lr_cascade"; YR=3.1557e7
# --- loglr model + background thr(FAR) (numpy) ---
m=json.load(open(f"{LR}/p4/lr_model_v5.json")); mu,sd,be=np.array(m["mu"]),np.array(m["sd"]),np.array(m["beta"]); GC=float(m["g_clip"])
gate=lambda g,s: np.clip(g,-GC,GC)*np.clip(np.asarray(s)/3.0,0,1)
loglr=lambda X: np.column_stack([np.ones(len(X)),(X-mu)/sd])@be
bg=np.load(f"{LR}/p0/lr_bg_features_sig4.npz"); sg=np.load(f"{LR}/p1/segment_g.npz"); I,J=bg["I"],bg["J"]
FB=np.column_stack([bg["sigma_H1"],bg["sigma_L1"],bg["coh_BA"],bg["centroid_H1"],bg["centroid_L1"],
                    gate(sg["g_H1"][I],bg["sigma_H1"]),gate(sg["g_L1"][J],bg["sigma_L1"])]).astype(np.float64)
lt=float(bg["bg_livetime_s"])/YR; lr_bg=np.sort(loglr(FB))[::-1]
FAR_FLOOR=1.0/lt
def thr(far): k=int(np.floor(far*lt)); return lr_bg[min(max(k,1),len(lr_bg))-1]
# --- comoving V_max per bank source ---
h=np.load(f"{LR}/p4/vt_horizons_O3a.npz"); hm1,hm2,Dh=h["m1"],h["m2"],h["Dh_lum"]
zg=np.linspace(1e-4,3,4000); dL=Planck18.luminosity_distance(zg).to(u.Mpc).value
dVdz=Planck18.differential_comoving_volume(zg).to(u.Gpc**3/u.sr).value*4*np.pi
Vmaxz=np.concatenate([[0],cumulative_trapezoid(dVdz/(1+zg),zg)])
Vmax=np.interp(np.interp(Dh,dL,zg),zg,Vmaxz)              # comoving V_max per source [Gpc^3]
# --- injections ---
inj=np.load(f"{LR}/p4/pastro_inj_O3a.npz"); im1,im2,illr,igat=inj["m1"],inj["m2"],inj["loglr"],inj["gated"]
def cell(x1,x2,a,b,tol): return ((np.abs(x1-a)<=tol)&(np.abs(x2-b)<=tol))|((np.abs(x1-b)<=tol)&(np.abs(x2-a)<=tol))

# (m1,m2,tol); ordered by Mtot. First 4 = Aframe's stellar cells; last 2 = IMBH (benchmark absent).
CELLS=[(20,10,5),(20,20,5),(35,20,5),(35,35,5),(100,100,15),(120,120,20)]
FARg=np.logspace(np.log10(FAR_FLOOR),np.log10(300),40)
AF_FAR=[.01,.1,1,10,100,300]
AF={(35,35):[7,8,9.5,12,16,18],(35,20):[3.1,3.5,4.3,6,7.5,8],(20,20):[1.65,1.9,2.3,3,3.7,3.95],(20,10):[.6,.7,.95,1.15,1.35,1.4]}
fig,axs=plt.subplots(2,3,figsize=(14,8)); axs=axs.ravel()
for ax,(a,b,tol) in zip(axs,CELLS):
    ci=cell(im1,im2,a,b,tol); cs=cell(hm1,hm2,a,b,tol); Vc=float(np.mean(Vmax[cs])); n=int(ci.sum())
    v=[(igat[ci]&(illr[ci]>=thr(far))).mean()*Vc for far in FARg]
    ax.plot(FARg,v,"o-",color="C0",ms=3,lw=2,label=f"MADGRAV (comoving, n={n})")
    if (a,b) in AF: ax.plot(AF_FAR,AF[(a,b)],"D--",color="purple",ms=5,alpha=.8,label="Aframe (approx, slide)")
    else:
        ax.set_facecolor("#fff6ec"); ax.text(.5,.92,"IMBH — benchmark\npipelines NOT evaluated",transform=ax.transAxes,ha="center",va="top",fontsize=8,color="darkorange")
    ax.axvspan(0.008,FAR_FLOOR,color="grey",alpha=.12)
    ax.set_xscale("log"); ax.set_title(f"$m_1={a},\\ m_2={b}$  ($M_{{tot}}={a+b}$)",fontsize=11)
    ax.grid(alpha=.3,which="both"); ax.set_xlim(0.008,400); ax.legend(fontsize=7,loc="lower right")
for ax in axs[3:]: ax.set_xlabel("False Alarm Rate  [yr$^{-1}$]")
for ax in (axs[0],axs[3]): ax.set_ylabel("Sensitive Volume  [Gpc$^3$, comoving]")
fig.suptitle("MADGRAV vs Aframe: sensitive volume vs FAR at fixed mass (log-linear; cf. slide p22) — O3a, prelim. bg",fontsize=12)
plt.tight_layout(rect=[0,0.04,1,0.96])
fig.text(.5,.01,f"grey band = below MADGRAV bg floor {FAR_FLOOR:.3f}/yr (cannot reach lower until full-485 bg). loglr channel; q-averaged cells; Aframe curve read approx off slide.",ha="center",fontsize=7,color="dimgrey")
for e in ("pdf","png"): plt.savefig(f"{LR}/p4/vt_vs_far_panels_O3a.{e}",dpi=140,bbox_inches="tight")
print(f"wrote vt_vs_far_panels_O3a.pdf/.png (FAR floor {FAR_FLOOR:.3f}/yr)")
