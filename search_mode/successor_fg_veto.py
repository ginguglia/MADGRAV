"""Foreground local-ASD veto recomputation for the successor candidate list (rule UNCHANGED, asd_consistency.recompute_local).
Run in the run's env on a GPU node:  python successor_fg_veto.py <run> <request.json> <out.json>
For candidates that were as-run detections, the as-run veto numbers appear in the merge logs; both are kept for comparison."""
import sys, json, time


def main():
    import asd_consistency as AV
    run, req, out = sys.argv[1:4]
    R = json.load(open(req)); res = []; t0 = time.time()
    for i, c in enumerate(R):
        r = AV.recompute_local(c["seg"], AV.idx_of(c)); r.update(seg=c["seg"], gps=c["gps"], channel=c["channel"], source="recomputed-2026-08-18")
        res.append(r)
        print(f"[fg-veto:{run}] {i+1}/{len(R)} {c['seg']} {c['gps']:.1f} ({c['channel']}) net {c['net']:.2f}->{r['net_loc']:.2f} cnn {c['cnn_hm']:.2f}/{c['cnn_lm']:.2f}->{r['hm_loc']:.2f}/{r['lm_loc']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[fg-veto:{run}] DONE {len(res)} -> {out}", flush=True)


if __name__ == "__main__":   # INFRASTRUCTURE REPAIR 2026-08-18 22:3x: main-guard so the forkserver QT-pool workers do not re-execute
    main()                   # this script on import (crash loop in jobs 1661480-84); no logic change.
