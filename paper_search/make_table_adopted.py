"""Table I for the ADOPTED configuration: 46 calibrated detections.

Statistic: sigma_net>4 trigger -> CNN glitch gate -> sigma_net<10.6 veto -> lnLambda-channel
per-arm FAR against the foreground-excluded time-slide background -> multiplied by the per-run
null-calibration factor measured on the full pseudo-foreground population. A candidate is a
detection when the calibrated FAR and its calibrated 90% upper limit are both below 1/yr.

Sources
  far_lronly_g106.csv          FAR and UL, inclusive and exclusive, under the adopted ranking
  ke_adopted.json              per-run calibration factors
  madgrav_far_final_x1.csv     catalog mass/SNR, sigma_net, cWB flag
  pastro_final_x1cnnadopt48f.csv  p_astro: per-run FGMC on the foreground-excluded FAR axis
                               (candidates and injections on one ruler; SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl)
  hl_snr_o3o4.json             rho_H1+L1, the H1+L1-only network SNR (build_hl_snr.py), which
                               is also the axis of Fig. 1. Both SNRs are tabulated: rho_net as
                               published, so the row can be matched against GWTC, and
                               rho_H1+L1 as the quantity this coincident H1+L1 search could
                               actually reach.
"""
import csv, json, os
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MG = MADGRAV_ROOT
# Final statistic (glitch-arm gate, 2026-09-09): defaults = the gated products; SM_KE_JSON / SM_FAR_LR_CSV /
# SM_PASTRO_CSV / SM_TABLE_OUT select others (e.g. the pre-gate ke_adopted.json + far_lronly_g106.csv).
KE = json.load(open(os.environ.get("SM_KE_JSON", f"{MG}/details/successor_statistic/ke_gnet.json")))
FAR_LR_CSV = os.environ.get("SM_FAR_LR_CSV", f"{MG}/figures/catalog_o3o4/far_lronly_g106_gnet.csv")
PASTRO_CSV = os.environ.get("SM_PASTRO_CSV", f"{MG}/search_mode/pastro_final/pastro_final_x1cnnadopt48fg.csv")
TABLE_OUT = os.environ.get("SM_TABLE_OUT", f"{MG}/paper_search/detections_table_gnet.tex")
KESTR = f"{KE['o3a']:.2f}$, ${KE['o3b']:.2f}$, ${KE['o4a']:.2f}$, ${KE['o4b']:.2f}"
NETMAX = 10.6
base = {(r["run"], r["name"]): r for r in csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"))}
far = {(r["run"], r["name"]): r for r in csv.DictReader(open(FAR_LR_CSV))}
pa = {(r["run"], r["name"]): r for r in csv.DictReader(open(PASTRO_CSV))}
# Seeds come from the pipeline's own seed file, never a hand-copy (2026-09-02: the previous
# hardcoded set substituted GW190513_205428 for GW190408_181802).
SEEDS = set(json.load(open(f"{MG}/search_mode/o3a_events.json")))
HL = json.load(open(f"{MG}/figures/catalog_o3o4/hl_snr_o3o4.json"))
ALIAS = {"GW190412": "GW190412_053044", "GW190521": "GW190521_030229"}
flt = lambda x: float(x) if str(x).strip() not in ("", "nan") else float("nan")

rows = []
for k, b in base.items():
    run, name = k
    if flt(b["net"]) >= NETMAX:
        continue
    if "gnet_pass" in far[k] and int(float(far[k]["gnet_pass"])) == 0:   # glitch-arm gate
        continue
    ke = KE[run.lower()]
    f_i, u_i = flt(far[k]["far_incl"]) * ke, flt(far[k]["ul90_incl"]) * ke
    f_e, u_e = flt(far[k]["far_excl"]) * ke, flt(far[k]["ul90_excl"]) * ke
    if not (f_e < 1.0):        # point estimate alone (design decision 2026-09-01)
        continue
    hl = HL.get(ALIAS.get(name, name)) or HL.get(name)
    rows.append(dict(run=run, name=name, mtot=b["mtot"], snr=b["snr_cat"],
                     hl=(hl or {}).get("snr_hl"), net=flt(b["net"]),
                     cwb=b["cwb"], N=int(flt(far[k]["N_excl"])), fi=f_i, ui=u_i, fe=f_e, ue=u_e,
                     pa=flt(pa[k]["p_astro"]) if k in pa else float("nan")))
order = {"O3a": 0, "O3b": 1, "O4a": 2, "O4b": 3}
rows.sort(key=lambda r: (order[r["run"]], r["fe"]))
assert len(rows) == 47, len(rows)


def cell(f, u, N):
    return rf"$<{u:.3f}$" if N == 0 else rf"${f:.3f}$"


def esc(n):
    return n.replace("_", r"\_") + (r"$^{\dagger}$" if n in SEEDS else "")


L = [r"\begin{table*}",
     r"\caption{\label{tab:detections}The \NDET{} MADGRAV detections at calibrated",
     r"$\FAR<1\,{\rm yr}^{-1}$. Of the \NCAND{} candidates passing the raw threshold, these are the ones",
     r"that remain below $1\,{\rm yr}^{-1}$ once the measured null calibration is applied. The ranking",
     r"statistic is the per-arm false-alarm rate on the $\ln\Lambda$ channel,",
     r"$\FAR=\min(N^{\rm HM},N^{\rm LM})/T_{\rm bg}$, evaluated against the time-slide background with the",
     r"zero-lag foreground removed and multiplied by the per-run calibration factor $\mathcal{K}$",
     r"measured on the full pseudo-foreground population (Sec.~\ref{sec:nullcal};",
     rf"$\mathcal{{K}}={KESTR}$ for O3a--O4b). $\FAR_{{\rm incl}}$ is the same quantity",
     r"against the background inclusive of the foreground; the two agree except where a detection's own",
     r"windows enter its background. Candidates with $\sigma_{\rm net}\geq10.6$ are vetoed",
     r"(Sec.~\ref{sec:method}); none of the \NCAND{} is affected. $M_{\rm tot}$ is the published",
     r"source-frame total mass and $\rho_{\rm net}$ the published network matched-filter SNR",
     r"\cite{gwtc21,gwtc3,gwtc40,gwtc50}, over the full observing network; $\rho_{\rm H1+L1}$ is",
     r"the H1+L1-only value of Eq.~(\ref{eq:snrhl}), the quantity this coincident search could",
     r"reach and the axis of Fig.~\ref{fig:population}. The two differ only where Virgo also",
     r"observed. $\sigma_{\rm net}$ is the MADGRAV network excess",
     r"significance. Events with no louder background family ($N=0$) carry a $90\%$ upper limit.",
     r"$p_{\rm astro}$ is a per-run Poisson-mixture (FGMC) fit on the foreground-excluded FAR axis:",
     r"signal density from the injection campaign of this same criterion, noise density uniform in FAR,",
     r"and the noise count pinned to $\FAR_{\rm det}\times T_{\rm fg}$ rather than fitted. The two events",
     r"above $0.1\,{\rm yr}^{-1}$ fall in the last populated injection bin, so their values rest on sparse",
     r"support and should be read as indicative. The last column marks",
     r"candidates also identified by cWB at ${\rm FAR}<1\,{\rm yr}^{-1}$. Events marked $^{\dagger}$ are",
     r"O3a development seed events (Sec.~\ref{sec:method}).}",
     r"\begin{ruledtabular}",
     r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{llrrrrccccc}",
     r"Event & Run & $M_{\rm tot}$ [$M_\odot$] & $\rho_{\rm net}$ & $\rho_{\rm H1+L1}$ & $\sigma_{\rm net}$ & $N$ & "
     r"$\FAR$ [yr$^{-1}$] & $\FAR_{\rm incl}$ & $p_{\rm astro}$ & cWB \\",
     r"\hline"]
def emit(sel, label, cap):
    L = [r"\begin{table*}", cap, r"\begin{ruledtabular}",
         r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{llrrrrccccc}",
         r"Event & Run & $M_{\rm tot}$ [$M_\odot$] & $\rho_{\rm net}$ & $\rho_{\rm H1+L1}$ & $\sigma_{\rm net}$ & $N$ & "
         r"$\FAR$ [yr$^{-1}$] & $\FAR_{\rm incl}$ & $p_{\rm astro}$ & cWB \\",
         r"\hline"]
    for r in sel:
        mt = f'{flt(r["mtot"]):.0f}' if str(r["mtot"]).strip() else r"\ldots"
        sn = f'{flt(r["snr"]):.1f}' if str(r["snr"]).strip() else r"\ldots"
        hl = f'{r["hl"]:.1f}' if r["hl"] is not None else r"\ldots"
        cw = r"\checkmark" if r["cwb"] == "True" else ""
        pa = f'${r["pa"]:.2f}$' if r["pa"] == r["pa"] else r"\ldots"
        L.append(f'{esc(r["name"])} & {r["run"]} & {mt} & {sn} & {hl} & {r["net"]:.1f} & {r["N"]} & '
                 f'{cell(r["fe"], r["ue"], r["N"])} & {cell(r["fi"], r["ui"], r["N"])} & {pa} & {cw} \\\\')
    L += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    return "\n".join(L)

o3 = [r for r in rows if r["run"].startswith("O3")]
o4 = [r for r in rows if r["run"].startswith("O4")]
cap_o4 = (r"\caption{\label{tab:detections-o4}Continuation of Table~\ref{tab:detections} for the "
          r"O4a and O4b runs. Columns, statistic and conventions are as in that table.}")
out = emit(o3, "tab:detections", "\n".join(L[1:L.index(r"\begin{ruledtabular}")])) + "\n\n" + emit(o4, "tab:detections-o4", cap_o4)
open(TABLE_OUT, "w").write(out + "\n")
from collections import Counter
print(f"wrote {os.path.basename(TABLE_OUT)}  (O3 {len(o3)} rows + O4 {len(o4)} rows = {len(rows)})")
print(f"  N=0: {sum(1 for r in rows if r['N']==0)}   cWB: {sum(1 for r in rows if r['cwb']=='True')}")
