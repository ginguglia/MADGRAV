"""MADGRAV self-contained RECOVERY demo -- recover GW190521 (the IMBH) from a small bundled segment.

Self-contained, de-hardcoded recovery demo. Reads a ~256 s strain
segment that ships INSIDE the package (demo/strain/{EV}_{det}.npz, ~8 MB), computes a
whole-segment Welch ASD (4 s / 2 s) from that segment, injects it into the shared MassiveEventPipeline, runs the
per-detector CAE sigma stream @1s stride, clusters net=(sH+sL)/sqrt2 > 4 triggers, and scores
the top ones with the vendored HM/LM CNN glitch-gate. No GWOSC fetch; ~2 min on a GPU.

NO FAR (one short segment has no lag livetime) -- this asks only: does the event window produce
a net>4 trigger that the CNN glitch-gate (>0.5) KEEPS (RECOVER)?

WHITENING: whole-segment Welch ASD (4 s / 2 s) over THIS 256 s segment -- the signal is in-band in
the PSD estimate, which mildly self-suppresses the event (conservative for a recovery test). Matches
the retrospective O3 convention that fired GW190521. CROSS-RUN CAVEAT: only the LEARNED weights
(CAE, 5-seed glitch arm, HM/LM CNNs)
remain O4a-trained applied here to local-ASD O3 data -- documented, expected, intentional.

Run:
  cd <package> && MADGRAV_ROOT=$(pwd) DEV=cuda:0 SM_ALLOW_CPU=1 python demo/recover_event.py
or simply:  bash demo/run_demo.sh
"""
import os, sys
import numpy as np

# ---- package closure: locate MADGRAV_ROOT and put the vendored modules on sys.path ----
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["MADGRAV_ROOT"] = MADGRAV_ROOT

# ---- event config (override via argv[1]=EV argv[2]=MERGER, or env EV / MERGER) ----
EV = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EV", "GW190521"))
MERGER = float(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MERGER", "1242442967.4"))

# ---- DEVICE: honor DEV / BLIND_DEV, default cuda:0 (NEVER hardcode cuda:1). driver_blindscan's
# _resolve_dev() degrades gracefully (absent GPU -> cuda:0; no CUDA + SM_ALLOW_CPU=1 -> CPU, which
# is NOT byte-identical to the frozen GPU calibration but lets the demo run anywhere). ----
DEV_REQ = os.environ.get("DEV") or os.environ.get("BLIND_DEV") or "cuda:0"
os.environ["BLIND_DEV"] = DEV_REQ          # driver_blindscan reads this and resolves it
os.environ["SM_DEV"] = DEV_REQ             # driver_streams reads this

# ---- de-hardcoded data env: the demo strain lives in the package, NOT in a run dir ----
DEMO_STRAIN = os.path.join(MADGRAV_ROOT, "demo", "strain")
os.environ["SM_STRAIN"] = DEMO_STRAIN      # driver_search_multi.STRAIN <- cnn_hm_lm reads here
# vendored JSONs so driver_search_multi imports without a provisioned run dir (mirrors check_install.py)
os.environ.setdefault("SM_BGJSON", os.path.join(MADGRAV_ROOT, "search_mode", "o3a_bg_segments_56.json"))
os.environ.setdefault("SM_EVENTSJSON", os.path.join(MADGRAV_ROOT, "search_mode", "o3a_events.json"))
os.environ.setdefault("SM_SEGJSON_EV", os.path.join(MADGRAV_ROOT, "search_mode", "o3a_segments_event.json"))
# driver_search_multi reads spectrogram_cascade/massive_calibration_BA.json relative to CWD -> run from ROOT
os.chdir(MADGRAV_ROOT)
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(MADGRAV_ROOT, _p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

import torch
import driver_streams as DS
from gwpy.timeseries import TimeSeries
import driver_blindscan as B                 # also imports driver_search_multi (reads SM_STRAIN at import)
DEV = DS.DEV                                  # resolved device (cuda:0 / fallback / cpu)

FS = DS.FS; WN = DS.WN; STRIDE = 1.0; step = int(STRIDE * FS)
NET_CUT = 4.0; GTH = B.GLITCH_THRESH


def main():
    # NOTE: the body runs under `if __name__ == "__main__"` because driver_streams' QT step uses a
    # `forkserver` multiprocessing pool; without the guard each worker re-imports this script as
    # __main__ and re-launches the pool (RuntimeError: process started before bootstrapping finished).

    # ---- load the bundled demo segment (keys: strain, gps_start, fs) ----
    bH = os.path.join(DEMO_STRAIN, f"{EV}_H1.npz")
    bL = os.path.join(DEMO_STRAIN, f"{EV}_L1.npz")
    for p in (bH, bL):
        if not os.path.isfile(p):
            sys.exit(f"[{EV}] missing bundled segment: {p}\n  (run demo/make_demo_segment.py on a build machine, or check the package)")
    H = np.load(bH); L = np.load(bL)
    sH = H["strain"].astype(np.float32); sL = L["strain"].astype(np.float32); g0 = float(H["gps_start"])
    print(f"[{EV}] device={DEV}  H1 {len(sH)/FS:.0f}s L1 {len(sL)/FS:.0f}s  start={g0:.1f}  merger offset={MERGER-g0:.1f}s", flush=True)

    # ---- STAGING for cnn_hm_lm: it reads strain from SM_STRAIN/{EV}_{det}.npz["strain"]. The bundled
    # segment already lives at SM_STRAIN (= demo/strain) and already carries the 'strain' key, so no copy
    # is needed and the committed demo data stays pristine (running the demo writes nothing into it). If a
    # user points SM_STRAIN elsewhere, stage a strain-only copy there. ----
    if os.path.abspath(os.environ["SM_STRAIN"]) != os.path.abspath(DEMO_STRAIN):
        os.makedirs(os.environ["SM_STRAIN"], exist_ok=True)
        np.savez(os.path.join(os.environ["SM_STRAIN"], f"{EV}_H1.npz"), strain=sH)
        np.savez(os.path.join(os.environ["SM_STRAIN"], f"{EV}_L1.npz"), strain=sL)
    assert "strain" in np.load(os.path.join(os.environ["SM_STRAIN"], f"{EV}_H1.npz")).files

    # ---- LOCAL O3 ASD (whole-segment Welch 4,2), inject into the SHARED pipeline ----
    pipe = DS.MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json", prep=DS.O4A, device=DEV)
    def local_asd(x):
        return TimeSeries(x.astype(np.float64), sample_rate=FS).psd(4, 2) ** 0.5
    pipe.asd = {"H1": local_asd(sH), "L1": local_asd(sL)}
    B._pipe = pipe                                # cnn_hm_lm uses the SAME local-ASD pipe
    arms = [DS.GlitchArm().to(DEV) for _ in range(5)]
    for i, a in enumerate(arms):
        a.load_state_dict(torch.load(f"{DS.LRD}/p1v42/arm_deploy_seed{i}.pt", map_location=DEV)); a.eval()
    print(f"[{EV}] local-ASD whitening; pipe on {pipe.device}; cross-run weights = CAE+arm+CNN (O4a-trained)", flush=True)

    # ---- per-detector CAE sigma stream @1s stride ----
    n = (len(sH) - WN) // step + 1; widx = np.arange(n); gps = g0 + widx * STRIDE + WN / FS / 2.0
    def sigma_stream(strain, det):
        out = np.empty(n)
        for c0 in range(0, n, 1024):
            cs = widx[c0:c0 + 1024]
            wb = pipe._whiten(np.stack([strain[w * step:w * step + WN] for w in cs]).astype(np.float32), det)
            out[c0:c0 + len(cs)] = DS.sigma_from_qt(pipe, DS.build_qt(pipe, wb), det)
        return out
    SH = sigma_stream(sH, "H1"); SL = sigma_stream(sL, "L1"); net = (SH + SL) / np.sqrt(2.0)
    ev = int(np.abs(gps - MERGER).argmin())
    print(f"[{EV}] {n} windows; net>{NET_CUT}: {int((net > NET_CUT).sum())}; event window={ev} (gps={gps[ev]:.1f})", flush=True)
    print(f"[{EV}] event-window sigma: H1={SH[ev]:.2f} L1={SL[ev]:.2f} net={net[ev]:.2f}", flush=True)

    # ---- cluster net>cut triggers (4s), score top by CNN HM/LM ----
    hit = np.where(net > NET_CUT)[0]; gap = int(round(4.0 / STRIDE)); groups = []; cur = [hit[0]] if len(hit) else []
    for i in hit[1:]:
        if i - cur[-1] <= gap: cur.append(i)
        else: groups.append(cur); cur = [i]
    if cur: groups.append(cur)
    trig = sorted([gr[int(np.argmax(net[gr]))] for gr in groups], key=lambda j: -net[j])
    print(f"\n{'rank':>4} {'win':>4} {'gps':>11} {'net':>5} {'HM':>6} {'LM':>6} {'OR':>6}  verdict", flush=True)
    for r, j in enumerate(trig[:8]):
        hm, lm = B.cnn_hm_lm(EV, int(j), EV, int(j)); orv = max(hm, lm)
        kb = ("HM" if hm > GTH else "") + ("+LM" if lm > GTH else "") or "none"
        is_ev = abs(j - ev) <= 2
        tag = f" <-- {EV}" if is_ev else ""
        print(f"{r+1:>4} {j:>4} {gps[j]:>11.1f} {net[j]:5.2f} {hm:6.3f} {lm:6.3f} {orv:6.3f}  {'RECOVER('+kb+')' if orv>GTH else 'veto'}{tag}", flush=True)

    # ---- always show the CNN at the true event window, even if it didn't make the top-8 ----
    hm_ev, lm_ev = B.cnn_hm_lm(EV, int(ev), EV, int(ev)); or_ev = max(hm_ev, lm_ev)
    print(f"\n[event window {ev}] net={net[ev]:.2f}  HM={hm_ev:.3f}  LM={lm_ev:.3f}  OR={or_ev:.3f}  (shown regardless of net cut)", flush=True)

    recovered = (net[ev] > NET_CUT) and (or_ev > GTH)
    print(f"\n=== {EV}: {'RECOVERED' if recovered else 'not recovered'} "
          f"(net={net[ev]:.2f} {'>' if net[ev]>NET_CUT else '<='} {NET_CUT}; CNN OR={or_ev:.3f} {'>' if or_ev>GTH else '<='} {GTH}) ===", flush=True)
    print("\nDONE", flush=True)
    return 0 if recovered else 2


if __name__ == "__main__":
    sys.exit(main())
