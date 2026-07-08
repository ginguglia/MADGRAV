"""
Massive-event detection pipeline (remnant BH >= 80 Msun).

Stages (all validated; see COHERENCE_REPORT.md):
  whiten -> CAE per-detector anomaly score -> network significance net_sigma
         -> coherence veto (H1-L1)          -> low-frequency morphology gate
         -> calibrated false-alarm rate from a frozen 47-yr time-slide background.

A candidate that passes the coherence veto AND the morphology gate is ranked against the
MASS-CONDITIONED background (FAR for high-mass-like coincidences only); otherwise it is
ranked against the general coherence-gated background. Calibration is frozen in
massive_calibration.json (build with build_massive_calibration.py).

Usage:
  from massive_pipeline import MassiveEventPipeline
  pipe = MassiveEventPipeline()
  result = pipe.score(h1_strain, l1_strain)      # 1D (one event) or 2D (n events), FS=4096, 4 s

  # CLI:
  python massive_pipeline.py --h1 h1.npy --l1 l1.npy
  python massive_pipeline.py --selftest
"""
import os, sys, json, argparse
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool
from scipy.ndimage import zoom
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import improved_pipeline as ip

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CALIB = os.path.join(HERE, "massive_calibration_BA.json")
DEF_PREP = os.path.join(MADGRAV_ROOT, "data", "o3a_search_prep")


class MassiveEventPipeline:
    def __init__(self, calib_path=DEF_CALIB, prep=DEF_PREP, device=None):
        self.calib = json.load(open(calib_path))
        c = self.calib
        self.fs = int(c["fs"]); self.ctx = float(c["context_seconds"])
        self.lag = int(c["lag_samples"]); self.f_cut = float(c["f_cut_hz"]); self.tcoh = float(c["tcoh"])
        self.cband = tuple(c["centroid_band_hz"]); self.cwin = float(c["centroid_window_s"])
        # coherence statistic: "pearson" (broadband 1s |Pearson| max-lag, legacy) or
        # "band_symnorm" (band-limited + symmetric-norm, the BA upgrade).
        self.coh_mode = c.get("coherence_mode", "pearson")
        self.coh_band = tuple(c.get("coherence_band_hz", [20, 400]))
        self.coh_win = float(c.get("coherence_window_s", 1.0))
        self.norm = c["sigma_norm"]
        self.cond = np.array(c["far_curve_cond_coh"], float)      # ascending net_sigma
        self.glob = np.array(c["far_curve_global_coh"], float)
        for cur in (self.cond, self.glob): cur[:] = cur[np.argsort(cur[:, 0])]
        self.bg_livetime_yr = float(c["bg_livetime_yr"])
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.asd = {d: ip.load_detector_asd_o1(prep, d) for d in ("H1", "L1")}
        self.model = ip.BaselineCAE(dropout=0.20).to(self.device)
        mp = c["model_path"]; mp = mp if os.path.isabs(mp) else os.path.join(MADGRAV_ROOT, mp)
        self.model.load_state_dict(torch.load(mp, map_location=self.device)); self.model.eval()

    # ---- stages ----
    def _whiten(self, raw, det):
        return np.asarray(ip.whiten_batch_gwpy_o1(np.asarray(raw, np.float32), [det]*len(raw), self.asd, True, "o1"), np.float32)

    def _qt(self, white):
        qi = ip.center_crop_waveforms(white, sample_rate=self.fs, context_seconds=self.ctx)
        args = [(w, self.fs, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for w in qi]
        with Pool(min(16, len(qi))) as p: mags = p.map(ip._compute_qt_image_worker, args)
        return ip.min_max_norm(np.stack([zoom(m, (256/m.shape[0], 128/m.shape[1]), order=1)
                                         for m in mags]).astype(np.float32)).astype(np.float32)

    def _recon(self, qt):
        o = []
        with torch.no_grad():
            for (x,) in DataLoader(TensorDataset(torch.from_numpy(qt[:, None])), batch_size=128):
                o.extend(ip.compute_reconstruction_loss(self.model, x.to(self.device)).cpu().numpy())
        return np.asarray(o, np.float64)

    def _centroid(self, white):                       # energy-weighted mean freq, central window
        cc = white.shape[1] // 2; half = int(self.cwin * self.fs / 2)
        w = white[:, cc-half:cc+half].astype(np.float64); w = w - w.mean(1, keepdims=True)
        P = np.abs(np.fft.rfft(w * np.hanning(w.shape[1])[None, :], axis=1))**2
        f = np.fft.rfftfreq(w.shape[1], 1.0/self.fs); band = (f >= self.cband[0]) & (f <= self.cband[1])
        return (P[:, band]*f[band]).sum(1) / (P[:, band].sum(1) + 1e-30)

    def _bandlimit(self, x):                           # zero FFT bins outside coherence band
        X = np.fft.rfft(x * np.hanning(x.shape[1])[None, :], axis=1)
        f = np.fft.rfftfreq(x.shape[1], 1.0/self.fs)
        X[:, (f < self.coh_band[0]) | (f > self.coh_band[1])] = 0
        return np.fft.irfft(X, n=x.shape[1], axis=1)

    def _coherence(self, wh1, wl1):                    # max coherence over +/-lag, central window
        cc = wh1.shape[1] // 2; half = int(self.coh_win * self.fs / 2)
        a = wh1[:, cc-half:cc+half].astype(np.float64); b = wl1[:, cc-half:cc+half].astype(np.float64)
        if self.coh_mode == "band_symnorm":            # BA: band-limit + symmetric-norm 2<x,y>/(|x|^2+|y|^2)
            a, b = self._bandlimit(a), self._bandlimit(b)
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
            ea = (a*a).sum(1); out = np.zeros(len(a), np.float32)
            for lag in range(-self.lag, self.lag+1):
                bs = np.roll(b, lag, axis=1); eb = (bs*bs).sum(1)
                out = np.maximum(out, (np.abs(2.0*(a*bs).sum(1)) / (ea+eb+1e-30)).astype(np.float32))
            return out
        a = a - a.mean(1, keepdims=True); a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b0 = b - b.mean(1, keepdims=True); out = np.zeros(len(a), np.float32)   # legacy |Pearson|
        for lag in range(-self.lag, self.lag+1):
            bs = np.roll(b0, lag, axis=1); bs = bs / (np.linalg.norm(bs, axis=1, keepdims=True) + 1e-12)
            out = np.maximum(out, np.abs((a*bs).sum(1)))
        return out

    def _far(self, net, curve):
        xs, ys = curve[:, 0], curve[:, 1]             # ys is FAR/yr, decreasing in net_sigma
        if net <= xs[0]:  return float(ys[0]), "lower_bound"     # FAR >= ys[0]
        if net >= xs[-1]: return float(ys[-1]), "upper_limit"    # FAR <= floor (bg-limited)
        return float(np.exp(np.interp(net, xs, np.log(ys)))), None

    # ---- public ----
    def score(self, h1, l1):
        h1 = np.atleast_2d(np.asarray(h1, np.float32)); l1 = np.atleast_2d(np.asarray(l1, np.float32))
        single = h1.shape[0] == 1
        wh1, wl1 = self._whiten(h1, "H1"), self._whiten(l1, "L1")
        sH = (self._recon(self._qt(wh1)) - self.norm["muH"]) / self.norm["sdH"]
        sL = (self._recon(self._qt(wl1)) - self.norm["muL"]) / self.norm["sdL"]
        net = (sH + sL) / np.sqrt(2.0)
        cH, cL = self._centroid(wh1), self._centroid(wl1)
        coh = self._coherence(wh1, wl1)
        out = []
        for i in range(len(net)):
            morph = bool(cH[i] < self.f_cut and cL[i] < self.f_cut)
            cohp = bool(coh[i] >= self.tcoh)
            massive = morph and cohp
            far, flag = self._far(net[i], self.cond if massive else self.glob)
            out.append({
                "sigma_H1": float(sH[i]), "sigma_L1": float(sL[i]), "net_sigma": float(net[i]),
                "coherence": float(coh[i]), "coherence_pass": cohp,
                "centroid_H1_hz": float(cH[i]), "centroid_L1_hz": float(cL[i]),
                "morphology_pass": morph, "is_massive_candidate": massive,
                "channel": "massive(cond)" if massive else "general(coh)",
                "far_per_yr": far, "far_flag": flag,
                "ifar_days": (365.0/far if far > 0 else float("inf")),
            })
        return out[0] if single else out


def _selftest(pipe, n=8):
    """Inject n high-mass bank signals into noise + score n noise pairs."""
    import collections
    # dev-only --selftest paths (NOT on the search run path); env-overridable, no host paths shipped
    PREP = DEF_PREP; QC = os.environ.get("MADGRAV_QT_CACHE", "")
    BANK = os.environ.get("MADGRAV_SIGNAL_BANK", "")
    pool = ip.load_o1_signal_bank(BANK); sid2 = {int(s): i for i, s in enumerate(pool["source_ids"])}
    ntest = np.load(f"{PREP}/noise_test.npy").astype(np.float32)
    ntm = ip.load_noise_metadata(f"{PREP}/noise_test_metadata.csv")
    em = json.load(open(f"{QC}/sig_eval_meta.json"))
    det_te = {dd: [i for i, r in enumerate(ntm) if r["detector"] == dd] for dd in ("H1", "L1")}
    pr = collections.defaultdict(dict)
    for r in em: pr[r["pair_id"]][r["detector"]] = r
    def inj(r):
        s = np.asarray(pool[r["detector"]][sid2[r["source_id"]]], np.float32)*np.float32(r["scale_factor"])
        return ntest[det_te[r["detector"]][int(r["pair_id"])]] + ip.place_signal_in_segment(s, ntest.shape[1], rng=None)
    hi = [(p, d) for p, d in pr.items() if "H1" in d and "L1" in d and 0.95*d["H1"]["total_mass"] >= 80][:n]
    print(f"\n=== SELFTEST: {len(hi)} high-mass injections + {n} noise pairs ===")
    print(f"{'kind':>16} {'netσ':>6} {'coh':>6} {'cH':>5} {'cL':>5} {'morph':>6} {'massive':>8} {'FAR/yr':>9} {'IFAR_d':>8}")
    for p, d in hi:
        r = pipe.score(inj(d["H1"]), inj(d["L1"]))
        print(f"{'inj Mtot=%.0f'%d['H1']['total_mass']:>16} {r['net_sigma']:>6.2f} {r['coherence']:>6.3f} "
              f"{r['centroid_H1_hz']:>5.0f} {r['centroid_L1_hz']:>5.0f} {str(r['morphology_pass']):>6} "
              f"{str(r['is_massive_candidate']):>8} {r['far_per_yr']:>9.2f} {r['ifar_days']:>8.0f}")
    pairs = [(det_te["H1"][k], det_te["L1"][k]) for k in range(n)]
    for h, l in pairs:
        r = pipe.score(ntest[h], ntest[l])
        print(f"{'noise':>16} {r['net_sigma']:>6.2f} {r['coherence']:>6.3f} "
              f"{r['centroid_H1_hz']:>5.0f} {r['centroid_L1_hz']:>5.0f} {str(r['morphology_pass']):>6} "
              f"{str(r['is_massive_candidate']):>8} {r['far_per_yr']:>9.2f} {r['ifar_days']:>8.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1"); ap.add_argument("--l1")
    ap.add_argument("--calib", default=DEF_CALIB); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    pipe = MassiveEventPipeline(calib_path=a.calib)
    print(f"[pipe] loaded calibration: f_cut={pipe.f_cut:.1f}Hz tcoh={pipe.tcoh:.4f} "
          f"bg={pipe.bg_livetime_yr:.1f}yr (FAR floor {pipe.cond[:,1].min():.2f}/yr)")
    if a.selftest:
        _selftest(pipe)
    elif a.h1 and a.l1:
        res = pipe.score(np.load(a.h1), np.load(a.l1))
        print(json.dumps(res, indent=2))
    else:
        print("provide --h1 and --l1 (npy strain, FS=4096, 4 s) or --selftest")
