r"""
QC - vein/artery mask overlap in auto-annotated sidecars.

WHY THIS EXISTS
---------------
MedSAM2 propagation (now in attic/) assumed that giving each vessel its own SAM 2
object id keeps the tracks apart. It does not. SAM 2 propagates every object id
independently; nothing in the model prevents two ids converging on the same
pixels. Observed on a real CFV clip: at peak compression the vein polygon jumped
onto the artery, so both labels described one structure.

Neither existing guard sees this:

  * `max_area_ratio` fires only on GROWTH (`a > ratio * median`). The vein is the
    larger vessel, so landing on the artery makes its area FALL. No trigger.
  * mask-contrast gating cannot help - the artery is anechoic too, so a mask
    sitting on it scores as a good mask.

But vein and artery are physically disjoint, so their masks should never
intersect. That is a hard anatomical constraint requiring no ground truth, and
it catches this failure exactly.

WHY THE ERROR MATTERS MORE THAN IT LOOKS
----------------------------------------
The compression ratio is min/max of vein area. If the vein mask lands on the
patent artery at peak compression, the minimum becomes the artery's area instead
of near-zero, so the ratio RISES - a normal compressible vein reads as rigid.
The corruption lands precisely on the frames the measurement depends on, and it
biases toward false-positive DVT. `--ratio-impact` quantifies this per clip.

TWO MEASURES, NOT ONE
---------------------
IoU alone is the wrong test. A small artery sitting entirely inside a large vein
mask has low IoU but has been completely captured. So we report both:

    iou          = |A n B| / |A u B|
    containment  = |A n B| / min(|A|, |B|)      <- catches full capture

`containment` is the primary flag; `iou` is reported for context.

USAGE
-----
    python scripts/qc_overlap.py --dir sidecars/
    python scripts/qc_overlap.py --dir sidecars/ --ratio-impact --csv overlap.csv
    python scripts/qc_overlap.py --dir sidecars/ --worst 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    sys.exit("needs pillow: pip install pillow")


# ---------------------------------------------------------------- label roles

def role(label: str) -> str | None:
    """Map a Darwin label name to vein / artery / clot / None.

    Handles both the US-DICOMizer panel names ('Common Femoral Vein # 1') and
    the ThrombUS+ expert vocabulary ('CFV Scanning Location 1'), where the
    vessel code ends in V for veins and A for arteries. the expert label scheme
    documents the second scheme; matching only on the words 'vein'/'artery'
    silently drops it.
    """
    low = label.lower()
    if "clot" in low or "thromb" in low:
        return "clot"
    if "vein" in low:
        return "vein"
    if "artery" in low or "arterial" in low:
        return "artery"
    head = label.split()[0].upper() if label.split() else ""
    if head in {"CFV", "FV", "PV", "GSV", "SFV", "DFV"}:
        return "vein"
    if head in {"CFA", "FA", "PA", "DFA", "SFA"}:
        return "artery"
    return None


# ---------------------------------------------------------------- geometry

def _poly_pts(path) -> list[tuple[float, float]]:
    return [(float(p["x"]), float(p["y"])) for p in path]


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _bboxes_disjoint(a, b) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def rasterize(paths, origin, size) -> np.ndarray:
    """Render polygon paths into a local boolean raster.

    Only the first path of each annotation is an outer boundary; the rest are
    holes (Darwin 2.0). US-DICOMizer does not support holes on import, and
    `step7` emits one path per annotation, so extra paths are treated as holes
    here rather than filled solid - filling them overstated lumen area by 20%
    on an annulus in the step2 tests.
    """
    ox, oy = origin
    w, h = size
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for i, path in enumerate(paths):
        pts = [(x - ox, y - oy) for x, y in _poly_pts(path)]
        if len(pts) < 3:
            continue
        d.polygon(pts, fill=0 if i else 1)
    return np.array(img, dtype=bool)


# ---------------------------------------------------------------- parsing

def load_clip(path: Path):
    """-> (frames, meta) where frames[idx][role] = list of polygon path-lists."""
    with open(path) as fh:
        doc = json.load(fh)

    slot = (doc.get("item", {}).get("slots") or [{}])[0]
    meta = {
        "name": doc.get("item", {}).get("name", path.stem),
        "width": slot.get("width"),
        "height": slot.get("height"),
        "frame_count": slot.get("frame_count"),
    }

    frames: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ann in doc.get("annotations", []):
        r = role(ann.get("name", ""))
        if r is None:
            continue                      # tags: _reviewed, Protocol Deviation
        for key, fr in (ann.get("frames") or {}).items():
            poly = fr.get("polygon")
            if not poly or not poly.get("paths"):
                continue
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue                  # frame index IS the dict key
            frames[idx][r].append(poly["paths"])
    return frames, meta


# ---------------------------------------------------------------- per-frame

def frame_overlap(paths_a: list, paths_b: list):
    """Union each role's polygons, then intersect the two roles.

    Returns (area_a, area_b, inter, iou, containment) in pixels.
    """
    flat_a = [p for paths in paths_a for p in paths]
    flat_b = [p for paths in paths_b for p in paths]
    if not flat_a or not flat_b:
        return 0, 0, 0, 0.0, 0.0

    box_a = _bbox_union([_bbox(_poly_pts(p)) for p in flat_a])
    box_b = _bbox_union([_bbox(_poly_pts(p)) for p in flat_b])

    # cheap reject before rasterizing anything
    if _bboxes_disjoint(box_a, box_b):
        return -1, -1, 0, 0.0, 0.0

    box = _bbox_union([box_a, box_b])
    ox, oy = int(np.floor(box[0])) - 1, int(np.floor(box[1])) - 1
    w = int(np.ceil(box[2])) - ox + 2
    h = int(np.ceil(box[3])) - oy + 2
    if w <= 0 or h <= 0 or w * h > 40_000_000:
        return 0, 0, 0, 0.0, 0.0

    ma = np.zeros((h, w), bool)
    for paths in paths_a:
        ma |= rasterize(paths, (ox, oy), (w, h))
    mb = np.zeros((h, w), bool)
    for paths in paths_b:
        mb |= rasterize(paths, (ox, oy), (w, h))

    aa, ab = int(ma.sum()), int(mb.sum())
    inter = int((ma & mb).sum())
    union = aa + ab - inter
    iou = inter / union if union else 0.0
    contain = inter / min(aa, ab) if min(aa, ab) else 0.0
    return aa, ab, inter, iou, contain


def polygon_area(paths_list) -> int:
    """Rasterized area of one role's polygons on one frame (px)."""
    flat = [p for paths in paths_list for p in paths]
    if not flat:
        return 0
    box = _bbox_union([_bbox(_poly_pts(p)) for p in flat])
    ox, oy = int(np.floor(box[0])) - 1, int(np.floor(box[1])) - 1
    w = int(np.ceil(box[2])) - ox + 2
    h = int(np.ceil(box[3])) - oy + 2
    if w <= 0 or h <= 0 or w * h > 40_000_000:
        return 0
    m = np.zeros((h, w), bool)
    for paths in paths_list:
        m |= rasterize(paths, (ox, oy), (w, h))
    return int(m.sum())


def compression_ratio(areas: dict[int, int], exclude: set[int] | None = None):
    """min/max over frames with a real measurement.

    A blank frame is stored as 0 and means 'no measurement', never 'the vein
    closed' - the step9 bug where one gap reported perfect compressibility.
    """
    exclude = exclude or set()
    vals = [a for i, a in areas.items() if a > 0 and i not in exclude]
    if len(vals) < 2:
        return None
    return min(vals) / max(vals)


# ---------------------------------------------------------------- per-clip

def audit_clip(path: Path, thr: float, ratio_impact: bool):
    frames, meta = load_clip(path)
    if not frames:
        return None

    rows = []
    vein_areas: dict[int, int] = {}
    for idx in sorted(frames):
        roles = frames[idx]
        has_v, has_a = "vein" in roles, "artery" in roles
        if ratio_impact and has_v:
            vein_areas[idx] = polygon_area(roles["vein"])
        if not (has_v and has_a):
            continue
        aa, ab, inter, iou, contain = frame_overlap(roles["vein"], roles["artery"])
        if aa < 0:                       # bbox-disjoint fast path
            aa = ab = inter = 0
        rows.append({
            "frame": idx, "vein_px": aa, "artery_px": ab,
            "inter_px": inter, "iou": iou, "containment": contain,
        })

    if not rows:
        return None

    flagged = [r for r in rows if r["containment"] >= thr]
    n = len(rows)
    max_idx = max(r["frame"] for r in rows) or 1

    # direction: is the vein swallowed by the artery, or the artery by the vein?
    absorbed_vein = sum(1 for r in flagged
                        if r["vein_px"] and r["inter_px"] / r["vein_px"] >= thr)
    absorbed_art = sum(1 for r in flagged
                       if r["artery_px"] and r["inter_px"] / r["artery_px"] >= thr)

    out = {
        "clip": meta["name"],
        "frames_both": n,
        "frames_flagged": len(flagged),
        "frac_flagged": len(flagged) / n,
        "max_containment": max(r["containment"] for r in rows),
        "median_iou": float(np.median([r["iou"] for r in rows])),
        "vein_absorbed": absorbed_vein,
        "artery_absorbed": absorbed_art,
        "flag_pos_mean": (float(np.mean([r["frame"] for r in flagged])) / max_idx
                          if flagged else None),
        "ratio_raw": None, "ratio_clean": None, "ratio_delta": None,
    }

    if ratio_impact and vein_areas:
        raw = compression_ratio(vein_areas)
        clean = compression_ratio(vein_areas, {r["frame"] for r in flagged})
        out["ratio_raw"] = raw
        out["ratio_clean"] = clean
        if raw is not None and clean is not None:
            out["ratio_delta"] = clean - raw

    return out, rows


# ---------------------------------------------------------------- reporting

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder of *_darwin.json sidecars")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="containment at or above which a frame is flagged "
                         "(default 0.30)")
    ap.add_argument("--worst", type=int, default=15,
                    help="how many worst clips to list")
    ap.add_argument("--ratio-impact", action="store_true",
                    help="recompute the compression ratio without flagged frames")
    ap.add_argument("--csv", help="write per-clip rows here")
    ap.add_argument("--frames-csv", help="write per-frame rows here")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*_darwin.json")) or \
        sorted(Path(args.dir).glob("*.json"))
    if not files:
        return print(f"no sidecars in {args.dir}") or 1

    results, all_frames, skipped = [], [], 0
    for p in files:
        got = audit_clip(p, args.threshold, args.ratio_impact)
        if got is None:
            skipped += 1
            continue
        summary, rows = got
        results.append(summary)
        for r in rows:
            r["clip"] = summary["clip"]
        all_frames.extend(rows)

    if not results:
        return print("no clip had vein and artery on the same frame") or 1

    tot_f = sum(r["frames_both"] for r in results)
    tot_flag = sum(r["frames_flagged"] for r in results)
    affected = [r for r in results if r["frames_flagged"]]

    print(f"\n{'='*66}\nVEIN / ARTERY OVERLAP AUDIT\n{'='*66}")
    print(f"sidecars scanned        {len(files)}")
    print(f"  usable (vein+artery)  {len(results)}")
    print(f"  skipped               {skipped}")
    print(f"frames with both roles  {tot_f}")
    print(f"\nthreshold: containment >= {args.threshold}")
    print(f"  frames flagged        {tot_flag}  ({tot_flag/tot_f:.1%})")
    print(f"  clips affected        {len(affected)}  "
          f"({len(affected)/len(results):.1%})")

    if affected:
        fr = [r["frac_flagged"] for r in affected]
        print(f"  per-clip flagged frac  median {np.median(fr):.1%}  "
              f"max {max(fr):.1%}")
        v = sum(r["vein_absorbed"] for r in affected)
        a = sum(r["artery_absorbed"] for r in affected)
        print(f"\ndirection of capture")
        print(f"  vein mostly inside artery   {v}")
        print(f"  artery mostly inside vein   {a}")
        pos = [r["flag_pos_mean"] for r in affected if r["flag_pos_mean"] is not None]
        if pos:
            print(f"\nwhere in the clip (0=start, 1=end)")
            print(f"  mean position of flagged frames  {np.mean(pos):.2f}")
            print(f"  (compression peaks mid-clip; ~0.5 supports the "
                  f"peak-compression hypothesis)")

    if args.ratio_impact:
        deltas = [r["ratio_delta"] for r in results if r["ratio_delta"] is not None]
        if deltas:
            d = np.array(deltas)
            print(f"\ncompression-ratio impact ({len(d)} clips recomputable)")
            print(f"  clips whose ratio changes   {(np.abs(d) > 1e-9).sum()}")
            print(f"  median change               {np.median(d):+.4f}")
            print(f"  ratio DROPS when cleaned    {(d < -1e-9).sum()}  "
                  f"(contamination had inflated it)")
            print(f"  largest single change       {d[np.argmax(np.abs(d))]:+.4f}")

    if affected:
        print(f"\nworst {min(args.worst, len(affected))} clips by flagged fraction")
        print(f"{'clip':<44}{'flagged':>12}{'max cont':>10}")
        for r in sorted(affected, key=lambda r: -r["frac_flagged"])[:args.worst]:
            print(f"  {r['clip'][:42]:<42}"
                  f"{r['frames_flagged']:>4}/{r['frames_both']:<4} "
                  f"{r['frac_flagged']:>5.0%}"
                  f"{r['max_containment']:>10.2f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {args.csv}")

    if args.frames_csv and all_frames:
        with open(args.frames_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["clip", "frame", "vein_px",
                                               "artery_px", "inter_px",
                                               "iou", "containment"])
            w.writeheader()
            w.writerows(all_frames)
        print(f"wrote {args.frames_csv}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
