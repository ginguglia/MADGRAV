"""Absolute sensitive VOLUME vs mass (Aframe units): V_sens(m) = eff(m) * <V_max(source)>_bin.
V_max(source) = (4/3)pi * D_h^3, D_h = distance_mpc * rho_ref / RHO_TH, rho_ref = optimal NETWORK SNR of the
projected bank waveform at its reference distance. Combines the per-source horizon (banks+O3a ASD) with the
FAR<=1/yr efficiency from vt_vs_mass_O3a.json. Also reports sensitive distance D_sens=(3 V_sens/4pi)^(1/3).
Run with gwdev (improved_pipeline). Output: vt_vs_mass_O3a_absolute.json (Mpc^3 / Gpc^3 / Mpc)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(ROOT, "improved"))
import improved_pipeline as ip
import os; ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); LR=f"{ROOT}/lr_cascade"; BANK=f"{ROOT}/bank"
RHO_TH=4.5                                                       # detection-threshold SNR (matches injection SNR_MIN)
PREP=f"{ROOT}/data/o3a_search_prep"
asd={d: ip.load_detector_asd_o1(PREP, d) for d in ("H1","L1")}
def load(path,csv):
    b=ip.load_o1_signal_bank(path); df=pd.read_csv(csv); by={int(r.source_id):i for i,r in df.iterrows()}
    out=[]
    for sid,i in by.items():
        wH,wL=b["H1"][sid],b["L1"][sid]
        if np.sqrt((np.asarray(wH)**2).sum())<=0 or np.sqrt((np.asarray(wL)**2).sum())<=0: continue
        rho=float(np.hypot(ip.compute_optimal_snr(np.asarray(wH,np.float32),asd["H1"]),
                           ip.compute_optimal_snr(np.asarray(wL,np.float32),asd["L1"])))
        out.append((float(df.total_mass[i]), float(df.distance_mpc[i]), rho,
                    float(df.mass1[i]), float(df.mass2[i])))
    return out
print("computing per-source horizons (banks + O3a ASD)...",flush=True)
src=load(f"{BANK}/p1_signal_bank", f"{BANK}/p1_bank_parameters.csv") \
   +load(f"{BANK}/ultramassive_bank", f"{BANK}/ultramassive_bank_parameters.csv")
mtot=np.array([s[0] for s in src]); dref=np.array([s[1] for s in src]); rho=np.array([s[2] for s in src])
m1=np.array([s[3] for s in src]); m2=np.array([s[4] for s in src])
Dh=dref*rho/RHO_TH                                              # horizon LUMINOSITY distance (Mpc) for rho_th
Vmax=(4/3)*np.pi*Dh**3                                          # Mpc^3 (EUCLIDEAN)
np.savez(f"{LR}/p4/vt_horizons_O3a.npz",mtot=mtot,m1=m1,m2=m2,dref=dref,rho=rho,Dh_lum=Dh,rho_th=RHO_TH)
print(f"{len(src)} sources; rho_ref med {np.median(rho):.1f}; D_h med {np.median(Dh):.0f} Mpc -> saved vt_horizons_O3a.npz",flush=True)

eff=json.load(open(f"{LR}/p4/vt_vs_mass_O3a.json"))
rows=[]
print(f"\n{'Mtot bin':>12} {'eff@1/yr':>9} {'<D_h> Mpc':>10} {'V_sens Gpc^3':>13} {'D_sens Mpc':>11}")
print("-"*60)
for r in eff["rows"]:
    lo,hi=r["bin"]; inb=(mtot>=lo)&(mtot<hi)
    if inb.sum()==0: continue
    vmax_bin=float(np.mean(Vmax[inb]))                          # mean injectable volume per source in bin
    vsens=r["eff_1yr"]*vmax_bin                                 # Mpc^3
    vsens_lo=r["ci95"][0]*vmax_bin; vsens_hi=r["ci95"][1]*vmax_bin
    dsens=(3*vsens/(4*np.pi))**(1/3) if vsens>0 else 0.0
    rows.append(dict(bin=[lo,hi],eff_1yr=r["eff_1yr"],Dh_mean_mpc=float(np.mean(Dh[inb])),
                     Vsens_mpc3=vsens,Vsens_gpc3=vsens/1e9,Vsens_gpc3_ci=[vsens_lo/1e9,vsens_hi/1e9],
                     Dsens_mpc=dsens))
    print(f"{f'[{lo},{hi})':>12} {r['eff_1yr']*100:8.1f}% {np.mean(Dh[inb]):10.0f} {vsens/1e9:13.3f} {dsens:11.0f}")
json.dump({"run":"O3a","rho_th":RHO_TH,"statistic":"loglr-channel","note":"V_sens=eff*<V_max>_bank-bin; loglr ch; LR bg",
           "rows":rows},open(f"{LR}/p4/vt_vs_mass_O3a_absolute.json","w"),indent=1)
print("\nwrote vt_vs_mass_O3a_absolute.json (V_sens in Gpc^3, sensitive distance in Mpc)")
