r"""
Dump the detector's RAW proposals, so seeding can be studied without a GPU.

WHY
---
Every seeding question left is about which boxes the detector offers, not about
what is done with them - and the sidecars only record what the tracker
eventually produced. Two open questions both need the proposals themselves:

1. GS. The "shallowest vessel" rule picks the GSV on 81.1% of expert frames,
   but the pipeline lands on the vein only 12.5% of the time and on an ARTERY
   46.7%. Size is not the filter: 0% of GSV polygons fall below
   `plausible_vessel`'s bound, and their median relative side (0.114) matches
   every other vessel. So either the GSV is not being detected, or it is
   detected and something downstream discards it. The sidecars cannot tell
   these apart.

2. ARTERY-ANCHORING. The detector is much better at arteries than veins -
   TOP1 0.7701 against 0.6038, recall 0.9008 against 0.7706 - because arteries
   stay round and patent while veins collapse. Anchoring the pair on the
   artery and deriving the vein would compose 0.77 with a ~0.9 rule instead of
   0.60 with the same rule. That is roughly 0.60 -> 0.69 on paper, and it is
   untested.

Both become offline CSV work once the proposals are on disk.

WHAT IT WRITES
--------------
One row per detection per frame:

    clip, frame, cls, role, score, x0, y0, x1, y1, W, H, site

`role` is the detector's own class name, before any geometry is applied - so
the raw material, not a decision. Sampling every `--stride` frames keeps the
file manageable; the seeding questions are about which vessels exist in a
frame, not about temporal detail.

USAGE
-----
    python scripts/dump_detections.py \
        --detector checkpoints/vessel_detector_split.pth \
        --dicoms dicoms_annotated --annotations annotations_flat \
        --out detections.csv --stride 5 --min-score 0.05 --device cuda

Use a LOW --min-score. The point is to find out whether a vessel was proposed
at all and at what rank; filtering at the pipeline's 0.5 would hide exactly the
evidence being looked for.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--annotations", default=None,
                    help="if given, only clips that have expert annotations "
                         "are dumped - which is all that can be scored anyway")
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--stride", type=int, default=5,
                    help="dump every Nth frame")
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="keep detections down to this score. Deliberately far "
                         "below the pipeline's 0.5: the question is whether a "
                         "vessel was proposed at all and where it ranked.")
    ap.add_argument("--max-frames-per-clip", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from dvt import step8_compression_phase as S8
    from dvt import step10_auto_seed as S10
    from dvt import step13_detector_seed as S13

    model, img_size, names = S13.load_detector(a.detector, a.device)
    print(f"detector loaded ({img_size}px, classes {sorted(names.values())})",
          flush=True)

    wanted = None
    if a.annotations:
        wanted = set()
        for p in Path(a.annotations).glob("*.json"):
            parts = p.stem.split("__")
            if len(parts) == 4:
                wanted.add(parts[3])
        print(f"restricting to {len(wanted)} annotated clips", flush=True)

    dcms = sorted(Path(a.dicoms).rglob("*.dcm"))
    if wanted is not None:
        dcms = [p for p in dcms if p.stem in wanted]
    if a.limit:
        dcms = dcms[:a.limit]
    print(f"{len(dcms)} clips to process", flush=True)

    n_rows = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "frame", "cls", "role", "score",
                    "x0", "y0", "x1", "y1", "W", "H", "site"])
        for n, dcm in enumerate(dcms, 1):
            site = S10.parse_filename_tag(dcm.stem)["site"] or ""
            try:
                frames = S8.load_clip_frames(dcm, max_frames=0, downsample=1)
            except Exception as e:
                print(f"  skip {dcm.stem}: {e}", flush=True)
                continue
            idx = list(range(0, len(frames), max(a.stride, 1)))
            if a.max_frames_per_clip:
                idx = idx[:a.max_frames_per_clip]
            h, wd = frames.shape[1:3]
            for i in idx:
                dets = S13.detect_frame(model, img_size, frames[i], a.device,
                                        a.min_score, names)
                for d in dets:
                    w.writerow([dcm.stem, i, d.cls, d.role, round(d.score, 4),
                                round(d.box[0], 1), round(d.box[1], 1),
                                round(d.box[2], 1), round(d.box[3], 1),
                                wd, h, site])
                    n_rows += 1
            if n % 25 == 0:
                print(f"  {n}/{len(dcms)} clips   {n_rows} detections",
                      flush=True)

    print(f"\nwrote {a.out}: {n_rows} detections")
    print("\nnext: this is enough to answer offline, with no GPU -")
    print("  * is the GSV among the proposals on GS frames, and at what rank")
    print("  * does anchoring the pair on the ARTERY beat anchoring on the vein")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
