"""MADGRAV2: Punzi-style fine-tune stage for the CAE (training_mode=punzi), replacing the margin fine-tune.
Ranking statistic = reconstruction error r(x) (as in the production margin recipe). Per epoch the noise training scores
are standardised (mu, sd) and a hard threshold z_t is set at the (K_B+1)-th loudest noise tile (K_B = punzi_b_target
expected background tiles above cut). The loudest `punzi_active` noise tiles form the active set (all others have
s ~ 0 under the soft cut). Each step: soft cut s = sigmoid((z - z_t)/tau) on a signal batch (with total-mass bins) and
on a random sub-batch of the active set (rescaled to the full active set);
    B = sum s_bg,   eff_b = mean_{i in bin b} s_i,   loss = log S_min^Poisson(B) - mean_b log(eff_b)    (Punzi log-mean,
    R_min = S_min/VT with constant per-bin volumes dropping out of the geometric mean) + reconstruction MSE on noise.
Per-tile total masses of the cached injection tiles are recovered by replaying the seeded RNG of the cache builder
(build_o1_injections_cache_dataset: integers(pool) -> uniform(SNR range) if projected -> integers(shift)); only the
bank CSVs are read. Checkpoint selection: same criterion as the margin recipe (val injections above the noise 3-sigma
threshold, tie-break on separation), so the two recipes are compared like for like."""
import os, sys, glob, csv, json, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

PUNZI_DIR = os.environ.get("PUNZI_GW_DIR", "$MADGRAV_SCRATCH/punzi-gw")
if PUNZI_DIR not in sys.path: sys.path.insert(0, PUNZI_DIR)
from punzi_gw import SminPoisson
SMIN = SminPoisson()

def _bank_csv_meta(bank_dir):
    sids, mt, mode = [], [], None
    for npz in sorted(glob.glob(os.path.join(bank_dir, "**", "signals_*.npz"), recursive=True)):
        with open(os.path.splitext(npz)[0] + ".csv", newline="") as fh: rows = list(csv.DictReader(fh))
        for r in rows:
            sids.append(int(r["source_id"])); mode = mode or r.get("projection_mode")
            mt.append(float(r["total_mass"]) if r.get("total_mass") else float(r["mass1"]) + float(r["mass2"]))
    return np.array(sids), np.array(mt), mode

def replay_injection_masses(ip, cfg, cache_dir, split, n_tiles):
    """Total mass per cached injection tile for split in ('tr','val'), exactly as drawn by the cache builder."""
    out_path = os.path.join(cache_dir, f"sig_qt_{split}_mass.npy")
    if os.path.exists(out_path):
        m = np.load(out_path); assert len(m) == n_tiles, (out_path, len(m), n_tiles); return m
    sids, mt, mode = _bank_csv_meta(cfg.o1_signal_bank_dir)
    uniq = np.unique(sids); rng = np.random.default_rng(ip.SEED); perm = rng.permutation(uniq)
    n_eval = max(1, min(cfg.o1_inj_eval_count, int(len(perm) * 0.1))); n_val = max(1, int(len(perm) * 0.2)); n_tr = max(1, len(perm) - n_val - n_eval)
    keep = set(int(x) for x in (perm[:n_tr] if split == "tr" else perm[n_tr:n_tr + n_val]))
    mask = np.array([s in keep for s in sids]); pool_mt = mt[mask]; npool = int(mask.sum())
    rng = np.random.default_rng(ip.SEED); shift = int(0.5 * ip.FS); masses = np.empty(n_tiles)
    for i in range(n_tiles):
        j = int(rng.integers(npool)); masses[i] = pool_mt[j]
        if mode == "projected": rng.uniform(*ip.PROJECTED_TARGET_NETWORK_SNR_RANGE)
        rng.integers(-shift, shift + 1)
    np.save(out_path, masses); print(f"[punzi] replayed {n_tiles} {split} injection masses (pool {npool} sources, mode {mode}); median Mtot {np.median(masses):.1f}", flush=True)
    return masses

class SigBinDataset(Dataset):
    def __init__(self, base, bins): self.base, self.bins = base, torch.as_tensor(bins, dtype=torch.long)
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.base[i][0], self.bins[i]

class IndexSubset(Dataset):
    def __init__(self, base, idx): self.base, self.idx = base, np.asarray(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.base[int(self.idx[i])]

def _scores(ip, model, ds, device, bs, nw):
    return ip.compute_scores(model, DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw), device, score_mode="recon").astype(np.float64)

def train_punzi_model(ip, model, noise_train_ds, noise_val_loader, sig_train_ds, sig_val_loader, cfg, device, cache_dir,
                      val_benchmark_loader=None, val_benchmark_meta=None, phase_tag="punzi", num_epochs=None):
    F = torch.nn.functional
    if num_epochs is None: num_epochs = cfg.weaksup_epochs
    edges = np.array([float(x) for x in cfg.punzi_bins.split(",")]); nb = len(edges) - 1
    m_tr = replay_injection_masses(ip, cfg, cache_dir, "tr", len(sig_train_ds))
    b_tr = np.clip(np.digitize(m_tr, edges) - 1, 0, nb - 1)
    print(f"[punzi] bins {cfg.punzi_bins}: train tiles per bin {np.bincount(b_tr, minlength=nb).tolist()}", flush=True)
    sig_loader = DataLoader(SigBinDataset(sig_train_ds, b_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {k: [] for k in ("train_total", "val_total", "train_noise", "val_noise", "train_sig", "val_sig", "val_inj_n_above_3sigma", "val_inj_mean_score",
                               "val_noise_threshold_3sigma", "val_sep_k1", "punzi_B", "punzi_eff", "z_t", "tau")}
    history["val_mass_bin_n_above_3sigma"] = {l: [] for l in ip.MASS_BIN_LABELS}; history["val_mass_bin_hit_rate"] = {l: [] for l in ip.MASS_BIN_LABELS}
    vb_bins = [row["mass_bin"] for row in val_benchmark_meta] if val_benchmark_meta is not None else None
    best = dict(n=-1, sep=-np.inf, epoch=-1, state=None); es = 0; n_act = int(cfg.punzi_active); n_sub = min(n_act, cfg.batch_size)
    for epoch in range(num_epochs):
        tau = cfg.punzi_tau0 * (cfg.punzi_tau1 / cfg.punzi_tau0) ** (epoch / max(1, num_epochs - 1))
        # --- epoch prologue: standardise noise scores, threshold, active set (no grad) ---
        zn = _scores(ip, model, noise_train_ds, device, cfg.batch_size, cfg.num_workers); mu, sd = float(zn.mean()), float(zn.std() + 1e-12)
        order = np.argsort(-zn); active = order[:n_act]; z_t = float((zn[order[int(cfg.punzi_b_target)]] - mu) / sd)
        act_ds = IndexSubset(noise_train_ds, active); act_loader = DataLoader(act_ds, batch_size=n_sub, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        noise_loader = DataLoader(noise_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        model.train(); tl, tn, tB, teff = [], [], [], []
        act_it = iter(act_loader); noi_it = iter(noise_loader)
        for x_s, b_s in sig_loader:
            try: (x_a,) = next(act_it)
            except StopIteration: act_it = iter(act_loader); (x_a,) = next(act_it)
            try: (x_n,) = next(noi_it)
            except StopIteration: noi_it = iter(noise_loader); (x_n,) = next(noi_it)
            x_s, x_a, x_n, b_s = x_s.to(device), x_a.to(device), x_n.to(device), b_s.to(device)
            optimizer.zero_grad()
            z_s = (ip.compute_reconstruction_loss(model, x_s) - mu) / sd; z_a = (ip.compute_reconstruction_loss(model, x_a) - mu) / sd
            s_s = torch.sigmoid((z_s - z_t) / tau); s_a = torch.sigmoid((z_a - z_t) / tau)
            B = s_a.sum() * (n_act / len(s_a))
            eff = torch.zeros(nb, device=device, dtype=s_s.dtype).index_add_(0, b_s, s_s) / torch.bincount(b_s, minlength=nb).clamp_min(1).to(s_s.dtype)
            present = torch.bincount(b_s, minlength=nb) > 0
            smin = SMIN(B.to("cpu", torch.float64)).to(s_s.device, s_s.dtype)      # table lives on CPU; .to() keeps the gradient
            punzi = torch.log(smin) - torch.log(eff[present].clamp_min(cfg.punzi_eff_floor)).mean()
            noise_loss = ip.compute_reconstruction_loss(model, x_n).mean()
            total = punzi + cfg.lambda_anom * noise_loss / max(mu, 1e-12)      # noise MSE normalised to its epoch-start level
            total.backward(); optimizer.step()
            tl.append(float(total.item())); tn.append(float(noise_loss.item())); tB.append(float(B.item())); teff.append(eff.detach().cpu().numpy())
        # --- validation, same criterion as the margin recipe ---
        model.eval()
        vn_scores = ip.compute_scores(model, noise_val_loader, device, score_mode="recon").astype(np.float64)
        v_mu, v_sd = float(vn_scores.mean()), float(vn_scores.std()); v_t3 = v_mu + 3.0 * v_sd
        v_sig = ip.compute_scores(model, sig_val_loader, device, score_mode="recon").astype(np.float64)
        v_crit = ip.compute_scores(model, val_benchmark_loader, device, score_mode="recon").astype(np.float64) if val_benchmark_loader is not None else v_sig
        n_above = int((v_crit > v_t3).sum()); sep = float(v_crit.mean() - v_mu)
        mb_n = {l: 0 for l in ip.MASS_BIN_LABELS}; mb_r = {l: float("nan") for l in ip.MASS_BIN_LABELS}
        if vb_bins is not None:
            above = v_crit > v_t3
            for l in ip.MASS_BIN_LABELS:
                mk = np.array([b == l for b in vb_bins]); mb_n[l] = int(above[mk].sum()) if mk.any() else 0; mb_r[l] = float(mb_n[l] / mk.sum()) if mk.any() else float("nan")
        eff_ep = np.nanmean(np.stack(teff), 0)
        for k, v in (("train_total", np.mean(tl)), ("val_total", float("nan")), ("train_noise", np.mean(tn)), ("val_noise", v_mu), ("train_sig", float("nan")), ("val_sig", float(v_sig.mean())),
                     ("val_inj_n_above_3sigma", n_above), ("val_inj_mean_score", float(v_crit.mean())), ("val_noise_threshold_3sigma", v_t3), ("val_sep_k1", sep),
                     ("punzi_B", float(np.mean(tB))), ("punzi_eff", eff_ep.tolist()), ("z_t", z_t), ("tau", tau)):
            history[k].append(v)
        for l in ip.MASS_BIN_LABELS: history["val_mass_bin_n_above_3sigma"][l].append(mb_n[l]); history["val_mass_bin_hit_rate"][l].append(mb_r[l])
        improved = n_above > best["n"] or (n_above == best["n"] and sep > best["sep"])
        if improved:
            best.update(n=n_above, sep=sep, epoch=epoch + 1, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, v_mu=v_mu, v_t3=v_t3, mb_n=dict(mb_n), mb_r=dict(mb_r), sig=float(v_sig.mean()), inj_mean=float(v_crit.mean()))
            es = 0; tag = " <-- best"
        else: es += 1; tag = f" (no improve {es}/{cfg.weaksup_es_patience})"
        print(f"[{phase_tag} {epoch+1:02d}/{num_epochs}] loss={np.mean(tl):.4f} noise_mse={np.mean(tn):.6f} B={np.mean(tB):.2f} z_t={z_t:.2f} tau={tau:.2f} "
              f"eff_bins={np.array2string(eff_ep, precision=2)} val>3sigma={n_above} val_sep_k1={sep:.6f} mass_hit=" + ",".join(f"{l}:{mb_n[l]}" for l in ip.MASS_BIN_LABELS) + tag, flush=True)
        if es >= cfg.weaksup_es_patience: print(f"Early stopping at epoch {epoch+1} (best val>3sigma {best['n']})", flush=True); break
    if best["state"] is not None: model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    best_info = {"epoch": best["epoch"], "val_inj_n_above_3sigma": best["n"], "val_noise_mean": best.get("v_mu"), "val_noise_threshold_3sigma": best.get("v_t3"),
                 "val_sig": best.get("sig"), "val_inj_mean_score": best.get("inj_mean"), "val_sep_k1": best["sep"],
                 "val_mass_bin_n_above_3sigma": best.get("mb_n"), "val_mass_bin_hit_rate": best.get("mb_r"), "recipe": "punzi", "punzi_bins": cfg.punzi_bins,
                 # fields the shared summary writer (save_o1_outputs) expects from the margin/joint recipes
                 "criterion": "val_inj_n_above_3sigma (punzi recipe)", "val_noise": float(best.get("v_mu") or float("nan")), "val_total": float("nan"),
                 "val_d_signal": float("nan"), "val_d_noise": float("nan"), "val_latent_sep": float("nan")}
    return history, best_info
