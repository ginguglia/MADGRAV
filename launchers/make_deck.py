#!/usr/bin/env python
"""Deterministic MADGRAV results deck. Reads detections.json / blindscan.json from DECK_OUT and renders
matplotlib plots + an HTML table whose NUMBERS are taken VERBATIM from the JSON (nothing is generated).
Usage: DECK_OUT=<dir> [DECK_FILE=..] [DECK_TITLE=..] python make_deck.py
"""
import os, io, json, base64, html, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT   = os.environ.get("DECK_OUT") or os.path.join(os.environ.get("SCRATCH", "."), "search_out")
FILE  = os.environ.get("DECK_FILE", os.path.join(OUT, "deck.html"))
TITLE = os.environ.get("DECK_TITLE", "MADGRAV — results")

def load(name):
    try: return json.load(open(os.path.join(OUT, name)))
    except Exception: return None

dets  = load("detections_pastro.json") or load("detections.json") or []   # prefer the p_astro-augmented list
has_pastro = any(d.get("p_astro") is not None for d in dets)
blind = load("blindscan.json") or {}
live  = blind.get("far_live_yr", "?")

def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

imgs = []
if dets:
    fig, ax = plt.subplots(figsize=(6,4))
    x=[d.get("net",0) for d in dets]; y=[d.get("loglr",0) for d in dets]
    c=[(d.get("best_far") if d.get("best_far") is not None else 1.0) for d in dets]
    sc=ax.scatter(x,y,c=c,cmap="viridis_r",s=90,edgecolor="k")
    ax.set_xlabel("net SNR"); ax.set_ylabel("loglr"); ax.set_title("Detections: net SNR vs loglr")
    fig.colorbar(sc,label="best FAR /yr"); imgs.append(("net SNR vs loglr", png(fig)))
    fig, ax = plt.subplots(figsize=(6,4))
    labels=[str(d.get("matches_known") or d.get("seg",""))[:18] for d in dets]
    fars=[(d.get("best_far") if d.get("best_far") is not None else 0) for d in dets]
    ax.barh(range(len(fars)), fars, color="steelblue"); ax.set_yticks(range(len(fars))); ax.set_yticklabels(labels,fontsize=7)
    ax.set_xlabel("best FAR /yr"); ax.set_title("best FAR per detection"); imgs.append(("best FAR per detection", png(fig)))

def cell(v, p=".3g"):
    if v is None: return "n/a"
    try: return format(v,p)
    except Exception: return html.escape(str(v))
rows=""
for d in sorted(dets, key=lambda x:(x.get("best_far") if x.get("best_far") is not None else 9e9)):
    cells=[html.escape(str(d.get("matches_known") or d.get("seg",""))), cell(d.get("gps"),".0f"),
           cell(d.get("net"),".2f"), cell(d.get("loglr"),".2f"), cell(d.get("best_far")),
           cell(d.get("best_ul90")), html.escape(str(d.get("channel","")))]
    if has_pastro: cells.append(cell(d.get("p_astro"),".3f"))
    rows+="<tr>"+"".join(f"<td>{c}</td>" for c in cells)+"</tr>"
pastro_th = "<th>p_astro*</th>" if has_pastro else ""
pastro_note = ('<p style="color:#b8860b;font-size:11px">* p_astro is PROVISIONAL (pre-calibration): '
               'single-fold, pooled p_s, survivors-bg p_n; not the calibrated headline.</p>') if has_pastro else ""

stamp="PROVISIONAL - UNOFFICIAL - UNREVIEWED"
live_str = (" ".join(f"fold{k}={float(x):.4f}" for k,x in sorted(live.items()))+" yr") if isinstance(live,dict) else html.escape(str(live))
plots_html="".join(f'<h3>{html.escape(t)}</h3><img src="data:image/png;base64,{b}" style="max-width:640px">' for t,b in imgs)
doc=f"""<html><head><meta charset="utf-8"></head><body style="font-family:sans-serif;max-width:780px;margin:24px auto;color:#222">
<div style="background:#dc3545;color:#fff;padding:6px;text-align:center;font-weight:bold;letter-spacing:1px">{stamp}</div>
<h1>{html.escape(TITLE)}</h1>
<p>{time.strftime('%F %H:%M')} | {len(dets)} detection(s) | bg livetime: {live_str}</p>
<table border=1 cellpadding=5 style="border-collapse:collapse;font-size:13px">
<tr style="background:#eee"><th>event/seg</th><th>gps</th><th>net</th><th>loglr</th><th>best FAR/yr</th><th>UL90</th><th>chan</th>{pastro_th}</tr>
{rows or '<tr><td colspan=8>no detections</td></tr>'}</table>
{pastro_note}
{plots_html}
<hr><p style="color:#888;font-size:11px">All numbers verbatim from detections.json/blindscan.json in {html.escape(OUT)}. {stamp}.</p>
</body></html>"""
open(FILE,"w").write(doc)
print(f"[deck] wrote {FILE} — {len(dets)} detections")
