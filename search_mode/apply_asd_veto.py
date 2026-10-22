"""Apply the local-ASD consistency veto (veto #1) to an EXISTING run's detections.json.

Recomputes each detection under a local +/-64s median-Welch ASD and drops ASD-mismatch artifacts,
without re-running the merge. Writes detections_asdveto.json next to the input.

Run: SM_STRAIN=... SM_ALLOW_CPU=1 BLIND_DEV=cpu python apply_asd_veto.py <run_dir>
"""
import os, sys, json
import asd_consistency as AV
import driver_blindscan as B

NETSIG_FLOOR = float(os.environ.get("SM_NETSIG_FLOOR", "4.0"))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "search_out"
    dets = json.load(open(f"{run}/detections.json"))
    print(f"[asd-veto] {run}: {len(dets)} detections; local ASD = +/-{AV.ASD_HALF:.0f}s, floor={NETSIG_FLOOR}, "
          f"glitch_thresh={B.GLITCH_THRESH}\n", flush=True)
    hdr = f"{'match/seg':>22} {'gps':>12} {'chan':>9} | {'net':>6} {'net_loc':>7} | {'hm_loc':>6} {'lm_loc':>6}  verdict"
    print(hdr); print("-" * len(hdr), flush=True)
    kept = []
    for d in sorted(dets, key=lambda x: x.get("best_far", 9e9)):
        keep, r = AV.survives(d, floor=NETSIG_FLOOR)
        tag = "KEEP" if keep else f"VETO({r['reason']})"
        lab = str(d.get("matches_known") or d["seg"])
        print(f"{lab:>22} {d['gps']:12.1f} {d.get('channel',''):>9} | {d['net']:6.2f} {r['net_loc']:7.2f} | "
              f"{r['hm_loc']:6.3f} {r['lm_loc']:6.3f}  {tag}", flush=True)
        d["asd_veto"] = r
        if keep:
            kept.append(d)
    json.dump(kept, open(f"{run}/detections_asdveto.json", "w"), indent=2)
    print(f"\n[asd-veto] {len(kept)}/{len(dets)} survive -> {run}/detections_asdveto.json", flush=True)


if __name__ == "__main__":
    main()
