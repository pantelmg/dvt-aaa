r"""
How much segmentation error does the compression ratio tolerate?

THE QUESTION
------------
Expert masks give ratio AUC 0.717 (0.848 per subject). Automated masks give
0.345 - inverted. Before investing in a better segmenter (nnU-Net or similar),
it is worth knowing whether the failure is general quality or something
specific, because those call for different fixes.

METHOD
------
Take the EXPERT polygons and degrade them in controlled ways, recomputing the
ratio AUC at each level:

  dilate / erode   uniform boundary error - the "slightly too big or small"
                   failure a decent segmenter makes
  jitter           random per-vertex noise - unstable boundaries
  shrink_min       scale down ONLY the smallest-area frame of each vessel,
                   which is what a model that loses the vessel at peak
                   compression does
  drop             randomly discard annotated frames, mimicking blank frames

READING THE RESULT
------------------
If AUC degrades smoothly with boundary error, quality is the binding
constraint and a better segmenter should help proportionally.

If AUC survives large boundary error but collapses under `shrink_min`, then
general quality is NOT the problem - the specific failure at peak compression
is - and a higher Dice score will not fix it.

The distinction matters: the fine-tuned model had BETTER Dice (0.74 vs
zero-shot) and WORSE clinical signal (0.345 vs 0.496), so Dice and downstream
utility are not aligned here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def scale_about_centroid(pts: np.ndarray, factor: float) -> np.ndarray:
    """Approximate dilation/erosion: scale the polygon about its centroid.

    Area scales as factor^2, so a factor of 1.05 is roughly a 10% area
    increase - the magnitude of a boundary that is a couple of pixels out on a
    vessel a few tens of pixels across.
    """
    c = pts.mean(axis=0)
    return c + (pts - c) * factor


def jitter(pts: np.ndarray, sigma_px: float, rng) -> np.ndarray:
    return pts + rng.normal(0.0, sigma_px, size=pts.shape)


def load_vessels(ann_dir: Path):
    """-> list of {dvt, subject, centre, label, frames: {frame: points}}."""
    out = []
    for f in sorted(Path(ann_dir).glob("*.json")):
        parts = f.name.split("__", 3)
        if len(parts) != 4:
            continue
        centre, group, subject, _rest = parts
        j = json.loads(f.read_text(encoding="utf-8"))
        slots = (j.get("item") or {}).get("slots") or [{}]
        if int(slots[0].get("frame_count") or 1) < 2:
            continue                       # ratio needs a multi-frame clip

        per = defaultdict(dict)
        for a in j.get("annotations") or []:
            name = (a.get("name") or "").strip()
            tok = name.split()[0].upper() if name else ""
            if not tok.endswith("V"):      # veins only
                continue
            for fk, fv in (a.get("frames") or {}).items():
                if "polygon" not in fv or not str(fk).lstrip("-").isdigit():
                    continue
                path = fv["polygon"]["paths"][0]
                per[name][int(fk)] = np.array([[p["x"], p["y"]] for p in path],
                                              dtype=float)
        for label, frames in per.items():
            if len(frames) >= 2:
                out.append({"dvt": 1 if group.upper() == "DVT" else 0,
                            "subject": subject, "centre": centre,
                            "label": label, "frames": frames})
    return out


def ratio_from(frames: dict[int, np.ndarray]) -> float | None:
    areas = [poly_area(p) for p in frames.values()]
    areas = [a for a in areas if a > 0]
    if len(areas) < 2:
        return None
    return min(areas) / max(areas)


def auc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        k = s == v
        if k.sum() > 1:
            r[k] = r[k].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def evaluate(vessels, transform, rng, subject_level: bool = False) -> float:
    scores, labels, by_subj = [], [], defaultdict(lambda: [0.0, 0])
    for v in vessels:
        frames = transform(v["frames"], rng)
        r = ratio_from(frames)
        if r is None:
            continue
        scores.append(r)
        labels.append(v["dvt"])
        cur = by_subj[v["subject"]]
        cur[0] = max(cur[0], r)
        cur[1] = v["dvt"]
    if subject_level:
        s = [x[0] for x in by_subj.values()]
        y = [x[1] for x in by_subj.values()]
        return auc(s, y)
    return auc(scores, labels)


# ------------------------------------------------------------- transforms
def t_identity(frames, rng):
    return frames


def make_scale(factor):
    def t(frames, rng):
        return {k: scale_about_centroid(p, factor) for k, p in frames.items()}
    return t


def make_jitter(sigma):
    def t(frames, rng):
        return {k: jitter(p, sigma, rng) for k, p in frames.items()}
    return t


def make_shrink_min(factor):
    """Shrink the smallest frame by a FIXED factor.

    Kept only to document a null result: this cannot change AUC at all. The
    ratio is min/max, so scaling the minimum by a constant multiplies every
    vessel's ratio by the same constant, and AUC is rank-based. The same logic
    makes the ratio exactly invariant to uniform dilation or erosion - which is
    itself a useful property, since a segmenter with a consistent boundary bias
    costs nothing.
    """
    def t(frames, rng):
        areas = {k: poly_area(p) for k, p in frames.items()}
        kmin = min(areas, key=areas.get)
        out = dict(frames)
        out[kmin] = scale_about_centroid(frames[kmin], factor)
        return out
    return t


def make_random_shrink_min(lo, hi):
    """Shrink the smallest frame by a RANDOM per-vessel factor.

    This is the realistic version: a tracker does not fail by the same amount
    everywhere, it fails unpredictably. Only differential error can change the
    ranking, so this is the transform that actually tests robustness.
    """
    def t(frames, rng):
        areas = {k: poly_area(p) for k, p in frames.items()}
        kmin = min(areas, key=areas.get)
        out = dict(frames)
        out[kmin] = scale_about_centroid(frames[kmin], rng.uniform(lo, hi))
        return out
    return t


def make_thin_vessel_failure(prob, factor=0.4):
    """Lose the vessel where it is thinnest, with some probability.

    This models the observed failure directly. A vein at peak compression is
    thin and low-contrast, so the segmenter under-segments or loses it - and
    that happens on SOME clips, not all. Because it corrupts the minimum, which
    is the numerator of the ratio, it should be the most damaging error mode.
    """
    def t(frames, rng):
        if rng.random() > prob:
            return frames
        areas = {k: poly_area(p) for k, p in frames.items()}
        kmin = min(areas, key=areas.get)
        out = dict(frames)
        out[kmin] = scale_about_centroid(frames[kmin], factor)
        return out
    return t


def make_over_segment_min(prob, factor=1.8):
    """Where the vein is thinnest, grab surrounding tissue instead.

    The mirror failure: rather than losing the vessel, the mask spills into
    neighbouring structures and reports a LARGER area at peak compression. That
    raises the minimum and makes a non-compressible vein look compressible -
    the direction that would invert the signal.
    """
    def t(frames, rng):
        if rng.random() > prob:
            return frames
        areas = {k: poly_area(p) for k, p in frames.items()}
        kmin = min(areas, key=areas.get)
        out = dict(frames)
        out[kmin] = scale_about_centroid(frames[kmin], factor)
        return out
    return t


def make_drop(prob):
    def t(frames, rng):
        keys = list(frames)
        keep = [k for k in keys if rng.random() > prob]
        if len(keep) < 2:
            keep = keys[:2]
        return {k: frames[k] for k in keep}
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--repeats", type=int, default=5,
                    help="repeats for stochastic transforms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="degradation_report.json")
    a = ap.parse_args()

    vessels = load_vessels(Path(a.annotations))
    print(f"{len(vessels)} veins with >=2 annotated frames, "
          f"{sum(v['dvt'] for v in vessels)} from DVT subjects, "
          f"{len({v['subject'] for v in vessels})} subjects\n")

    rng = np.random.default_rng(a.seed)
    base = evaluate(vessels, t_identity, rng)
    base_s = evaluate(vessels, t_identity, rng, subject_level=True)
    print(f"baseline (expert masks): vessel AUC {base:.4f} | subject AUC {base_s:.4f}\n")

    tests = []
    for f in (1.20, 1.10, 1.05, 0.95, 0.90, 0.80):
        tests.append((f"scale x{f:.2f} (area x{f**2:.2f})", make_scale(f), False))
    for s in (1, 3, 5, 10):
        tests.append((f"vertex jitter {s}px", make_jitter(s), True))
    tests.append(("shrink MIN fixed x0.5 (null)", make_shrink_min(0.5), False))
    for lo, hi in ((0.8, 1.0), (0.5, 1.0), (0.3, 1.0)):
        tests.append((f"random shrink MIN {lo}-{hi}",
                      make_random_shrink_min(lo, hi), True))
    for pr in (0.2, 0.4, 0.6):
        tests.append((f"lose thin vessel p={pr}", make_thin_vessel_failure(pr), True))
    for pr in (0.2, 0.4, 0.6):
        tests.append((f"over-segment thin p={pr}", make_over_segment_min(pr), True))
    for p in (0.2, 0.4, 0.6):
        tests.append((f"drop {int(p*100)}% of frames", make_drop(p), True))

    results = {"baseline_vessel_auc": round(base, 4),
               "baseline_subject_auc": round(base_s, 4),
               "n_vessels": len(vessels), "tests": {}}

    print(f"{'transform':28s} {'vesselAUC':>10s} {'delta':>8s} {'subjAUC':>9s}")
    print("-" * 60)
    for name, fn, stochastic in tests:
        reps = a.repeats if stochastic else 1
        v = [evaluate(vessels, fn, np.random.default_rng(a.seed + i)) for i in range(reps)]
        s = [evaluate(vessels, fn, np.random.default_rng(a.seed + i), True) for i in range(reps)]
        vm, sm = float(np.mean(v)), float(np.mean(s))
        results["tests"][name] = {"vessel_auc": round(vm, 4),
                                  "subject_auc": round(sm, 4),
                                  "delta_vs_baseline": round(vm - base, 4)}
        print(f"{name:28s} {vm:10.4f} {vm-base:+8.4f} {sm:9.4f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")

    print("\n--- how to read this ---")
    print("Smooth decline under scale/jitter -> segmentation QUALITY is the")
    print("binding constraint, and a better segmenter should help.")
    print("Robust to scale/jitter but collapsing under 'shrink MIN' -> the")
    print("problem is the specific failure at peak compression, and a higher")
    print("Dice score will NOT fix it.")


if __name__ == "__main__":
    main()
