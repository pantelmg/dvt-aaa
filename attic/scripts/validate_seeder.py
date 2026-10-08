r"""
Does the auto-annotator segment the RIGHT vessel?

WHY THIS IS THE DECIDING EXPERIMENT
-----------------------------------
The degradation test showed the compression ratio is remarkably robust to
segmentation error, measured on expert masks:

    uniform dilate/erode +-20%   AUC 0.7167  (unchanged - a min/max ratio is
                                              exactly invariant to scaling)
    vertex jitter 10px           AUC 0.6820
    lose the thin vessel on 40%  AUC 0.6762
    drop 60% of frames           AUC 0.6924

Nothing takes it below 0.68. Yet the automated pipeline scored 0.345 -
inverted. That gap cannot be boundary quality, so the masks are probably on the
WRONG STRUCTURE.

This script tests that directly: for every clip with both an auto-annotation
and an expert annotation, it compares the predicted vein polygon against the
expert vein polygon on the same frame.

READING THE RESULT
------------------
    high IoU  -> the seeder is fine, the hypothesis is wrong, look elsewhere
    low IoU, high artery-IoU -> vein/artery confusion, fixable in the role
                                assignment in step10_auto_seed
    low IoU everywhere       -> the seeder finds neither vessel; fix detection

Only the third case argues for a better segmentation model. The first two are
seeding logic, which is far cheaper to fix.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def poly_pts(path) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in path], dtype=np.float32)


def rasterize(pts: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def contains(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` inside `outer` - catches a mask nested in the wrong
    structure, which IoU alone scores as a near miss."""
    s = int(inner.sum())
    return int((inner & outer).sum()) / s if s else 0.0


def kind_of(name: str) -> str:
    n = (name or "").strip()
    low = n.lower()
    if low == "clot":
        return "clot"
    if low == "other":
        return "other"
    tok = n.split()[0].upper() if n else ""
    # expert vocabulary: "CFV Scanning Location 1"
    # auto vocabulary  : "Common Femoral Vein # 1"
    if tok.endswith("V") or "vein" in low:
        return "vein"
    if tok.endswith("A") or "artery" in low:
        return "artery"
    return "other"


def load_polys(path: Path):
    """-> {frame: {kind: [points, ...]}}, plus the slot dimensions."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    slots = (j.get("item") or {}).get("slots") or [{}]
    h = int(slots[0].get("height") or 0)
    w = int(slots[0].get("width") or 0)
    out: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for a in j.get("annotations") or []:
        k = kind_of(a.get("name", ""))
        if k == "other":
            continue
        for fk, fv in (a.get("frames") or {}).items():
            if "polygon" not in fv or not str(fk).lstrip("-").isdigit():
                continue
            out[int(fk)][k].append(poly_pts(fv["polygon"]["paths"][0]))
    return out, h, w


def stem_of(name: str) -> str:
    """Normalise both naming schemes to a comparable clip key."""
    s = Path(name).name
    for suf in ("_darwin.json", ".json", ".dcm"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    if "__" in s:                       # CENTRE__GROUP__SUBJECT__clip
        s = s.split("__")[-1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", required=True,
                    help="folder of auto-generated *_darwin.json sidecars")
    ap.add_argument("--expert", required=True,
                    help="folder of expert annotations (annotations_flat)")
    ap.add_argument("--out", default="seeder_validation.json")
    a = ap.parse_args()

    expert_files = {stem_of(p.name): p for p in Path(a.expert).glob("*.json")}
    auto_files = {stem_of(p.name): p for p in Path(a.auto).glob("*.json")}
    shared = sorted(set(expert_files) & set(auto_files))

    print(f"auto: {len(auto_files)} | expert: {len(expert_files)} | "
          f"overlapping clips: {len(shared)}")
    if not shared:
        print("\nNo clips in common. The auto sidecars and the expert set must")
        print("cover the same clips for this comparison to be possible.")
        print("Run run_autoannotate.py over the ANNOTATED dicoms:")
        print("  --dir dicoms_annotated")
        return

    rows = []
    for stem in shared:
        exp, h, w = load_polys(expert_files[stem])
        aut, ah, aw = load_polys(auto_files[stem])
        h = h or ah
        w = w or aw
        if not h or not w:
            continue
        common_frames = sorted(set(exp) & set(aut))
        for f in common_frames:
            ev = exp[f].get("vein") or []
            ea = exp[f].get("artery") or []
            av = aut[f].get("vein") or []
            if not ev or not av:
                continue
            # largest polygon of each, which is the one the ratio uses
            e_vein = rasterize(max(ev, key=lambda p: len(p)), h, w)
            a_vein = rasterize(max(av, key=lambda p: len(p)), h, w)
            e_art = rasterize(max(ea, key=lambda p: len(p)), h, w) if ea else None

            rows.append({
                "clip": stem, "frame": f,
                "iou_vein": round(iou(a_vein, e_vein), 4),
                "iou_artery": round(iou(a_vein, e_art), 4) if e_art is not None else None,
                "pred_in_expert_vein": round(contains(a_vein, e_vein), 4),
                "expert_vein_px": int(e_vein.sum()),
                "pred_px": int(a_vein.sum()),
            })

    if not rows:
        print("no comparable frames found")
        return

    iv = np.array([r["iou_vein"] for r in rows])
    ia = np.array([r["iou_artery"] for r in rows if r["iou_artery"] is not None])
    ratio = np.array([r["pred_px"] / max(r["expert_vein_px"], 1) for r in rows])

    print(f"\n{len(rows)} comparable frames across {len({r['clip'] for r in rows})} clips")
    print(f"\nIoU(predicted vein, EXPERT VEIN):")
    print(f"  median {np.median(iv):.3f} | mean {iv.mean():.3f} | "
          f">0.5 in {(iv > 0.5).mean():.0%} | <0.1 in {(iv < 0.1).mean():.0%}")
    if ia.size:
        print(f"IoU(predicted vein, EXPERT ARTERY):")
        print(f"  median {np.median(ia):.3f} | >0.5 in {(ia > 0.5).mean():.0%}")
        conf = float(np.mean(ia > iv[: len(ia)]))
        print(f"  frames where the prediction matches the ARTERY better than "
              f"the vein: {conf:.0%}")
    print(f"\npredicted area / expert vein area: median {np.median(ratio):.2f} "
          f"(1.0 would be perfect)")

    verdict = ("seeder is broadly correct - look elsewhere for the 0.345"
               if np.median(iv) > 0.5 else
               "vein/artery confusion - fix role assignment in step10_auto_seed"
               if ia.size and np.median(ia) > np.median(iv) else
               "predictions match neither vessel - detection itself is failing")
    print(f"\nVERDICT: {verdict}")

    Path(a.out).write_text(json.dumps(
        {"n_frames": len(rows), "median_iou_vein": float(np.median(iv)),
         "median_iou_artery": float(np.median(ia)) if ia.size else None,
         "median_area_ratio": float(np.median(ratio)),
         "verdict": verdict, "rows": rows}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
