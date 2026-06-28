"""p_astro reliability diagram (per-run) from the FGMC calibration_reliability bins.
-> p4/fig_pastro_calibration.pdf (and copy next to the manuscript)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR = f"{ROOT}/lr_cascade"
r = json.load(open(f"{LR}/p4/pastro_results.json"))

fig, ax = plt.subplots(figsize=(5.0, 4.6))
ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
colors = {"O3a": "#1f77b4", "O3b": "#2ca02c", "O4a": "#d62728", "O4b": "#9467bd"}
for run in ("O3a", "O3b", "O4a", "O4b"):
    cr = r["runs"][run]["calibration_reliability"]
    pred = [b["predicted"] for b in cr]
    real = [b["realized"] for b in cr]
    ax.plot(pred, real, "-o", ms=4, lw=1.3, color=colors[run], label=run)

ax.axvspan(0.9, 1.0, color="gold", alpha=0.18, zorder=0)
ax.text(0.945, 0.04, "used\nbin", ha="center", va="bottom", fontsize=7, color="darkgoldenrod")
ax.set_xlabel(r"predicted $p_\mathrm{astro}$ (bin centre)")
ax.set_ylabel(r"realised signal fraction")
ax.set_title(r"FGMC $p_\mathrm{astro}$ reliability (per run)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout()
for p in (f"{LR}/p4/fig_pastro_calibration.pdf",
          f"{ROOT}/fig_pastro_calibration.pdf"):
    fig.savefig(p, dpi=160)

# verify caption-claimed numbers
def get(run, b):
    for x in r["runs"][run]["calibration_reliability"]:
        if x["bin"] == b:
            return x["predicted"], x["realized"]
print("saved fig_pastro_calibration.pdf")
print("O4a 0.5-0.6:", get("O4a", "0.5-0.6"))
print("O4b 0.6-0.7:", get("O4b", "0.6-0.7"))
print("pooled-ish top bin O4b 0.9-1.0:", get("O4b", "0.9-1.0"))
