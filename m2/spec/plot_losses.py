"""Loss per epoch for specialist training runs (train_spec.py / train_single.py history JSONs).
One column per run: top = train loss vs test loss (class-balanced BCE; the fold pilot has no test loss, only train),
bottom = test background pass at t99 (the selection metric); dashed line = selected epoch.
Usage: plot_losses.py <out.png> <json> [<json> ...]"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C_TRAIN, C_TEST, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e6e5df"
out, files = sys.argv[1], sys.argv[2:]
runs = [json.load(open(f)) for f in files]
fig, ax = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6.2), sharex="col", squeeze=False)
for j, r in enumerate(runs):
    H = r["history"]; ep = np.array([h["epoch"] for h in H]); tr = np.array([h["loss"] for h in H])
    te = np.array([h.get("test_loss", np.nan) for h in H]); bg = np.array([h["ho"]["bg_pass"] for h in H]); be = r["best_epoch"]
    a = ax[0, j]; a.plot(ep, tr, color=C_TRAIN, lw=2, label="train loss")
    if np.isfinite(te).any(): a.plot(ep, te, color=C_TEST, lw=2, label="test loss")
    for y, c, t in ((tr[-1], C_TRAIN, "train"), (te[-1], C_TEST, "test")):
        if np.isfinite(y): a.annotate(t, (ep[-1], y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    a.set_title(f"{r['name']}  ({r['band'][0]:.0f}-{r['band'][1]:.0f} Hz)", fontsize=10, color=INK); a.set_ylabel("BCE loss" if j == 0 else "", color=MUTED)
    b = ax[1, j]; b.plot(ep, 100 * bg, color=C_TRAIN, lw=2); b.set_xlabel("epoch", color=MUTED); b.set_ylabel("test bg pass @ t99 (%)" if j == 0 else "", color=MUTED)
    b.annotate(f"selected: epoch {be}, {100*bg[be-1]:.1f}%", (be, 100 * bg[be - 1]), xytext=(6, 10), textcoords="offset points", fontsize=9, color=INK)
    b.plot([be], [100 * bg[be - 1]], "o", ms=8, color=C_TRAIN, mec="white", mew=2)
    for x in (a, b):
        x.axvline(be, color=MUTED, lw=1, ls="--"); x.grid(True, color=GRID, lw=0.8); x.set_axisbelow(True)
        for s in ("top", "right"): x.spines[s].set_visible(False)
        for s in ("left", "bottom"): x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=9)
ax[0, 0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=130, facecolor="white"); print(f"wrote {out}")
