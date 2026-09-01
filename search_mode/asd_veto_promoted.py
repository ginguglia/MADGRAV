"""Apply the local-ASD consistency veto (veto #1) to NAMED candidates from a run's blindscan.json.

Same rule and same code path as driver_blindscan_merge.py:309 (asd_consistency.survives), but applied to
candidates selected by GPS rather than to the run's detections.json. Used to test the candidates that the
as-run x n_channels trials factor kept out of the detection set.

Run inside the run's env:  SM_VETO_GPS="1242442967.0,1249205578.0" python asd_veto_promoted.py
"""
import os, sys, json
import asd_consistency as AV
import driver_blindscan as B

NETSIG_FLOOR = float(os.environ.get("SM_NETSIG_FLOOR", "4.0"))
OUT = os.environ["SM_OUT"]
TARGETS = [float(x) for x in os.environ["SM_VETO_GPS"].split(",") if x.strip()]


def _chan(d):
    a, b = d.get("far_lr_perarm"), d.get("far_net_perarm")
    if a is None: return "net-sigma"
    if b is None: return "loglr"
    return "net-sigma" if b < a else "loglr"


def main():
    trig = json.load(open(f"{OUT}/blindscan.json"))["triggers"]
    print(f"[asd-veto-promoted] {OUT}: {len(trig)} triggers; local ASD = +/-{AV.ASD_HALF:.0f}s, "
          f"floor={NETSIG_FLOOR}, glitch_thresh={B.GLITCH_THRESH}", flush=True)
    res = []
    for g in TARGETS:
        cand = [t for t in trig if abs(t["gps"] - g) < 0.6]
        if len(cand) != 1:
            print(f"  gps {g:.1f}: {len(cand)} matching triggers -- SKIP", flush=True); continue
        d = cand[0]; d["channel"] = _chan(d)
        keep, r = AV.survives(d, floor=NETSIG_FLOOR)
        tag = "KEEP" if keep else f"VETO({r['reason']})"
        print(f"  [asd-veto] {str(d.get('matches_known') or d['seg']):22s} {d['gps']:12.1f} ({d['channel']}) "
              f"net {d['net']:.2f}->{r['net_loc']:.2f} cnn {d.get('cnn_hm',float('nan')):.2f}/"
              f"{d.get('cnn_lm',float('nan')):.2f}->{r['hm_loc']:.2f}/{r['lm_loc']:.2f}  {tag}", flush=True)
        res.append(dict(gps=d["gps"], seg=d["seg"], channel=d["channel"], net=d["net"],
                        cnn_hm=d.get("cnn_hm"), cnn_lm=d.get("cnn_lm"), keep=bool(keep), asd_veto=r))
    json.dump(res, open(f"{OUT}/asd_veto_promoted.json", "w"), indent=2)
    print(f"\n[asd-veto-promoted] {sum(1 for r in res if r['keep'])}/{len(res)} survive "
          f"-> {OUT}/asd_veto_promoted.json", flush=True)


if __name__ == "__main__":
    main()
