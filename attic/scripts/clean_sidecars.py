r"""
Strip vein/artery-contaminated frames from existing sidecars.

WHY THIS EXISTS SEPARATELY FROM THE RERUN
-----------------------------------------
`qc_overlap.py` found that 185 of 268 clips (class C) are contaminated only
late and only on ~10% of frames. Those do not need re-tracking - dropping the
bad frames recovers them, at no GPU cost. Only the 83 class A/B clips, where
the tracks converged early and never recovered, need the propagation fixes in
`step14`. Running this first tells you how much of the run is salvageable
before you spend anything.

WHAT IT DOES
------------
For each frame, rasterizes the vein and artery polygons, computes containment
(|A n B| / min(|A|,|B|)), and deletes BOTH vessels' polygons on frames at or
above the threshold. Deleting both is deliberate: when two tracks have merged
there is no way from the JSON alone to say which one is still on its vessel.

Clot polygons are never touched. Clot legitimately overlaps the vein - median
clot/vein area 0.62 across 783 expert polygons, per `step12_clot_detect` - so
it is exempt from the whole notion of a containment conflict.

The output is a valid Darwin sidecar the app still loads: the four importer
constraints from `step7_write_annotations` are preserved because annotations
are removed wholesale rather than rewritten.

USAGE
-----
    python scripts/clean_sidecars.py --dir sidecars/ --out cleaned/
    python scripts/clean_sidecars.py --dir sidecars/ --out cleaned/ --dry-run
    python scripts/clean_sidecars.py --dir sidecars/ --out cleaned/ \
        --threshold 0.2 --min-keep 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_overlap import frame_overlap, load_clip, role  # noqa: E402


def clean_one(path: Path, thr: float, min_keep: float):
    """-> (payload, stats). payload is None when nothing needed removing."""
    frames, meta = load_clip(path)
    if not frames:
        return None, {"reason": "no polygons"}

    bad = set()
    for idx, roles in frames.items():
        if "vein" not in roles or "artery" not in roles:
            continue
        aa, _ab, _inter, _iou, contain = frame_overlap(roles["vein"],
                                                       roles["artery"])
        if aa < 0:                       # bounding boxes disjoint
            continue
        if contain >= thr:
            bad.add(idx)

    n_vessel_frames = sum(1 for r in frames.values()
                          if {"vein", "artery"} & set(r))
    if not bad:
        return None, {"reason": "clean", "bad": 0, "frames": n_vessel_frames}

    keep_frac = 1 - len(bad) / max(n_vessel_frames, 1)
    if keep_frac < min_keep:
        return None, {"reason": "unsalvageable", "bad": len(bad),
                      "frames": n_vessel_frames, "keep_frac": keep_frac}

    with open(path) as fh:
        doc = json.load(fh)

    removed = 0
    kept_anns = []
    for ann in doc.get("annotations", []):
        r = role(ann.get("name", ""))
        if r not in ("vein", "artery"):
            kept_anns.append(ann)        # clot and tags pass through untouched
            continue
        fr = ann.get("frames") or {}
        surviving = {k: v for k, v in fr.items()
                     if not (k.isdigit() and int(k) in bad)}
        removed += len(fr) - len(surviving)
        if not surviving:
            continue                     # drop the annotation entirely
        ann = dict(ann)
        ann["frames"] = surviving
        # `ranges` must still cover the surviving keys or the importer can
        # place the polygon outside the clip's frame span.
        idxs = sorted(int(k) for k in surviving if k.isdigit())
        if idxs:
            ann["ranges"] = [[idxs[0], idxs[-1] + 1]]
        kept_anns.append(ann)

    doc["annotations"] = kept_anns
    return doc, {"reason": "cleaned", "bad": len(bad),
                 "frames": n_vessel_frames, "removed": removed,
                 "keep_frac": keep_frac}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="containment at which a frame is dropped")
    ap.add_argument("--min-keep", type=float, default=0.40,
                    help="if cleaning would leave less than this fraction of "
                         "vessel frames, the clip is reported as needing a "
                         "re-run rather than silently gutted (default 0.40)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.dir)
    dst = Path(args.out)
    files = sorted(src.glob("*_darwin.json")) or sorted(src.glob("*.json"))
    if not files:
        print(f"no sidecars in {src}")
        return 1
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    tally = {"clean": 0, "cleaned": 0, "unsalvageable": 0, "no polygons": 0}
    removed_total = 0
    rerun = []

    for p in files:
        payload, st = clean_one(p, args.threshold, args.min_keep)
        tally[st["reason"]] = tally.get(st["reason"], 0) + 1
        if st["reason"] == "unsalvageable":
            rerun.append((p.name, st["keep_frac"]))
        if payload is None:
            if not args.dry_run and st["reason"] in ("clean", "no polygons"):
                shutil.copyfile(p, dst / p.name)
            continue
        removed_total += st["removed"]
        if not args.dry_run:
            with open(dst / p.name, "w") as fh:
                json.dump(payload, fh, indent=2)

    print(f"\n{'='*60}\nSIDECAR CLEANING (containment >= {args.threshold})\n{'='*60}")
    print(f"  already clean        {tally['clean']}")
    print(f"  cleaned in place     {tally['cleaned']}")
    print(f"  need a re-run        {tally['unsalvageable']}")
    print(f"  no polygons          {tally['no polygons']}")
    print(f"  frame-polygons removed {removed_total}")

    if rerun:
        print(f"\nclips needing the step14 fixes (keep fraction < {args.min_keep}):")
        for name, kf in sorted(rerun, key=lambda x: x[1])[:25]:
            print(f"  {kf:5.0%}  {name}")
        if len(rerun) > 25:
            print(f"  ... and {len(rerun)-25} more")
        Path("rerun_list.txt").write_text(
            "\n".join(n for n, _ in rerun) + "\n")
        print("\nwrote rerun_list.txt")

    if args.dry_run:
        print("\n(dry run - nothing written)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
