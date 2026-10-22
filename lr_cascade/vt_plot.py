"""MADGRAV O3a detection EFFICIENCY vs total mass at FAR<=1/yr (support figure; efficiency has no volume
convention so it's comoving-consistent with the headline overlay). Absolute volume is the comoving overlay
(vt_comoving_overlay_O3a*) -- the old Euclidean V_sens panel is RETIRED to avoid a convention mismatch.
Run with cern-aigw."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR=f"{ROOT}/lr_cascade"
eff=json.load(open(f"{LR}/p4/vt_vs_mass_O3a.json"))
rows=eff["rows"]; lt=eff["bg_livetime_yr"]
ctr=[(r["bin"][0]+r["bin"][1])/2 for r in rows]; xb=[[c-r["bin"][0],r["bin"][1]-c] for c,r in zip(ctr,rows)]
e=[r["eff_1yr"]*100 for r in rows]; elo=[(r["eff_1yr"]-r["ci95"][0])*100 for r in rows]; ehi=[(r["ci95"][1]-r["eff_1yr"])*100 for r in rows]
ef=[r["eff_floor"]*100 for r in rows]; n=[r["n"] for r in rows]
fig,ax=plt.subplots(figsize=(8.5,5.5))
ax.errorbar(ctr,e,yerr=[elo,ehi],xerr=np.array(xb).T,fmt="o-",color="C0",capsize=3,lw=2,ms=7,label="FAR $\\leq$ 1/yr")
ax.plot(ctr,ef,"s--",color="C3",alpha=.7,label=f"FAR $\\leq$ floor ({1/lt:.3f}/yr)")
ax.axvspan(60,80,color="grey",alpha=.12); ax.text(70,4,"benchmark\n35+35 M$_\\odot$",ha="center",fontsize=8,color="grey")
for c,y,nn in zip(ctr,e,n): ax.annotate(f"n={nn//1000}k" if nn>=1000 else f"n={nn}",(c,y),textcoords="offset points",xytext=(0,9),fontsize=6.5,ha="center",color="C0")
ax.set_xlabel("total mass  M$_{tot}$  [M$_\\odot$]"); ax.set_ylabel("detection efficiency  [%]")
ax.set_title(f"MADGRAV O3a detection efficiency vs total mass (loglr proxy; LR bg {lt:.0f} yr)")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylim(0,60); ax.set_xlim(0,400)
plt.tight_layout()
for ext in ("pdf","png"): plt.savefig(f"{LR}/p4/vt_vs_mass_O3a.{ext}",dpi=140,bbox_inches="tight")
print("wrote vt_vs_mass_O3a.pdf/.png (efficiency-only; Euclidean panel retired)")
