"""Re-score every bottleneck checkpoint with the DEPLOYED read-out (reconstruction error) and
compare against the read-out that actually selected it (the weak-sup classifier head).

Why: in `joint` mode -- the production recipe -- improved_pipeline selects the best checkpoint and
early-stops on `val_inj_n_above_3sigma` / `val_sep_k1` computed with score_mode="classifier", and
BaselineCAE.score() returns get_logit(), the classifier head on the latent. But the deployed
pipeline scores with the reconstruction error (massive_pipeline._recon ->
compute_reconstruction_loss), and score_real_events.py does the same. So the epoch we keep is the
one that maximises separation in a statistic we never deploy. This measures the consequence.

Three questions, all answerable from weights already on disk:
  Q1  does the variant ranking (A vs C_k32 vs C_k20) change between the two read-outs?
  Q2  is the classifier-selected checkpoint (_best) WORSE under recon than the final epoch?
      If yes, selection is actively picking the wrong epoch for the deployed statistic.
  Q3  how big is the spread, i.e. is any of this resolvable at all given the draw-to-draw variance?

Usage: recon_vs_clf_report.py            (all run dirs under bottleneck/ and bottleneck_reps/)
"""
import glob
import json
import os
import re

import numpy as np
import torch

SC = MADGRAV_SCRATCH
MG = MADGRAV_ROOT
import importlib.util
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

_spec = importlib.util.spec_from_file_location("ip", f"{MG}/improved/improved_pipeline.py")
ip = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(ip)
except SystemExit:
    pass

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 256
N_NOISE = 6000          # same noise-calibration size score_real_events.py uses


def load_model(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    k = sd["bott_down.weight"].shape[0] if "bott_down.weight" in sd else None
    m = ip.BaselineCAE(latent_channels=k)
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval(), k


def _batched(model, arr, mode):
    out = np.empty(len(arr), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(arr), BATCH):
            x = torch.from_numpy(np.asarray(arr[i:i + BATCH])).float().to(DEV)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if mode == "recon":
                y = model(x)
                v = ((y - x) ** 2).mean(dim=(1, 2, 3))
            else:
                v = model.score(x)          # == get_logit: the selection read-out
            out[i:i + BATCH] = v.cpu().numpy()
    return out


def metrics(model, noise, sig, mode):
    n = _batched(model, noise, mode)
    s = _batched(model, sig, mode)
    mu, sd = float(n.mean()), float(n.std())
    if sd <= 0:
        return dict(n3=0, sep=float("nan"))
    return dict(n3=int((s > mu + 3.0 * sd).sum()), sep=float((s.mean() - mu) / sd))


def main():
    caches = {}
    for d in (1, 2, 3):
        c = f"{SC}/bottleneck/qtcache_draw{d}"
        noise = np.load(f"{c}/noise_qt_te.npy", mmap_mode="r")
        idx = np.linspace(0, len(noise) - 1, min(N_NOISE, len(noise))).astype(int)
        caches[d] = (np.asarray(noise[idx]),
                     np.asarray(np.load(f"{c}/sig_qt_val_benchmark.npy", mmap_mode="r")))
        print(f"[cache draw{d}] noise {caches[d][0].shape} signal {caches[d][1].shape}", flush=True)

    dirs = sorted(glob.glob(f"{SC}/bottleneck_reps/*") + glob.glob(f"{SC}/bottleneck/*"))
    rows = []
    for rd in dirs:
        name = os.path.basename(rd)
        m = re.search(r"draw(\d)", name)
        if not m or not os.path.isdir(f"{rd}/models"):
            continue
        noise, sig = caches[int(m.group(1))]
        for ck, label in (("baseline_cae_weaksup_best.pt", "best(clf-selected)"),
                          ("baseline_cae_weaksup.pt", "final-epoch")):
            p = f"{rd}/models/{ck}"
            if not os.path.exists(p):
                continue
            model, k = load_model(p)
            r = metrics(model, noise, sig, "recon")
            c = metrics(model, noise, sig, "clf")
            variant = name.split("_draw")[0]
            rows.append(dict(run=name, variant=variant, ckpt=label, k=k,
                             n3_recon=r["n3"], sep_recon=r["sep"],
                             n3_clf=c["n3"], sep_clf=c["sep"]))
            print(f"  {name:<22} {label:<19} recon n3={r['n3']:>4} sep={r['sep']:>7.2f} | "
                  f"clf n3={c['n3']:>4} sep={c['sep']:>7.2f}", flush=True)
            del model
            torch.cuda.empty_cache()

    json.dump(rows, open(f"{MG}/search_mode/pastro_final/recon_vs_clf.json", "w"), indent=1)

    print("\nQ1 -- variant ranking under each read-out (best checkpoints, mean +/- sd over runs):")
    best = [r for r in rows if r["ckpt"].startswith("best")]
    for v in sorted({r["variant"] for r in best}):
        g = [r for r in best if r["variant"] == v]
        print(f"  {v:<10} n={len(g):<3} recon n3 {np.mean([x['n3_recon'] for x in g]):7.1f}"
              f" +/- {np.std([x['n3_recon'] for x in g]):5.1f}   "
              f"clf n3 {np.mean([x['n3_clf'] for x in g]):7.1f}"
              f" +/- {np.std([x['n3_clf'] for x in g]):5.1f}")

    print("\nQ2 -- does classifier selection pick a WORSE epoch for the deployed (recon) statistic?")
    pair = {}
    for r in rows:
        pair.setdefault(r["run"], {})[r["ckpt"]] = r
    worse = tot = 0
    for run, d in sorted(pair.items()):
        if len(d) < 2:
            continue
        b, f = d["best(clf-selected)"], d["final-epoch"]
        tot += 1
        flag = ""
        if f["n3_recon"] > b["n3_recon"]:
            worse += 1; flag = "  <- final beats 'best' under recon"
        print(f"  {run:<22} recon n3: best {b['n3_recon']:>4} vs final {f['n3_recon']:>4}{flag}")
    print(f"  => classifier-selected checkpoint is worse under recon in {worse}/{tot} runs")

    print("\nQ3 -- correlation between the two read-outs across all checkpoints:")
    a = np.array([r["n3_recon"] for r in rows], float)
    b = np.array([r["n3_clf"] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0:
        from scipy.stats import spearmanr
        print(f"  Pearson r = {np.corrcoef(a[ok], b[ok])[0,1]:.3f}   "
              f"Spearman rho = {spearmanr(a[ok], b[ok]).statistic:.3f}   (n={int(ok.sum())})")
        print("  A high rho means the two read-outs agree and the mismatch is harmless;")
        print("  a low or negative rho means selection was effectively blind to the deployed score.")
    print(f"\n-> {MG}/search_mode/pastro_final/recon_vs_clf.json")


if __name__ == "__main__":
    main()
