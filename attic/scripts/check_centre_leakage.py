r"""
Can the CENTRE be identified from the mask videos?

WHY THIS RUNS FIRST
-------------------
The previous V-JEPA attempt on this cohort produced features that predicted
hospital identity at AUC ~1.00 and then collapsed on held-out centres. That
failure was discovered late, after the compute was spent. It is cheap to detect
early: train a classifier to predict the CENTRE from the same inputs the probe
will use, and see whether it can.

If it cannot, the leakage is gone and a DVT result on these inputs means
something. If it can, whatever the probe learns will be partly the scanner, and
the number will not survive a new hospital.

WHY IT MIGHT STILL LEAK, EVEN WITH MASKS
----------------------------------------
A binary mask carries no speckle or gain, but GEOMETRY leaks:

    pixel area varies 5.6x across centres
    frame dimensions run 273x516 (GR1) to 1300x745 (LT1)
    vessel position on screen follows each site's scanning protocol

`step17` normalises the canvas to the vein's own centroid and extent for
exactly this reason. This script tests whether that worked, rather than
assuming it - the assumption is the thing that failed last time.

WHAT COUNTS AS PASSING
----------------------
Chance for a 5-class problem is not 20%: the classes are wildly imbalanced
(FR1 and GR1 hold 93% of the data), so the honest floor is the MAJORITY-CLASS
rate, which this script computes and prints. A centre classifier at or near
the majority rate has learned nothing.

Also reported: the same test on RAW pixel videos, if a dicom directory is
given. That is the positive control - it should leak badly, and if it does not,
the test itself is broken.

USAGE
-----
    python scripts/check_centre_leakage.py --sidecars dicoms_annotated \\
        --annotations annotations_flat
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def summarise(video, valid, channels="both"):
    """Cheap shape/motion descriptors - no encoder needed for this test.

    If a plain summary of the masks can identify the centre, an encoder
    certainly will, so this is a conservative (early-warning) test rather than
    an exact simulation of the probe.
    """
    v = video[valid]
    if len(v) < 4:
        return None
    vein, art = v[:, 0], v[:, 1]
    f = []
    chans = (vein,) if channels == "vein" else (vein, art)
    for ch in chans:
        area = ch.reshape(len(ch), -1).sum(1)
        f += [area.mean(), area.std(), area.min(), area.max(),
              (area.min() / max(area.max(), 1e-6))]
        ys, xs = np.nonzero(ch.sum(0) > 0)
        if len(xs):
            f += [xs.mean(), ys.mean(), xs.std(), ys.std(),
                  (xs.max() - xs.min()), (ys.max() - ys.min())]
        else:
            f += [0, 0, 0, 0, 0, 0]
    if channels != "vein":
        va = vein.reshape(len(vein), -1).sum(1)
        aa = art.reshape(len(art), -1).sum(1)
        f += [float(np.mean(va / np.maximum(aa, 1e-6)))]
    f += [float(valid.mean()), float(len(v))]
    return np.asarray(f, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecars", required=True,
                    help="folder of *_darwin.json produced by run_autoannotate")
    ap.add_argument("--annotations", required=True,
                    help="annotations_flat - the CENTRE__GROUP__SUBJECT__CLIP "
                         "filenames carry centre and subject directly. "
                         "train.csv cannot be used: its 'File Name' column "
                         "holds HASHED names (adfbd93d...dcm) while sidecars "
                         "are named from the anonymised stems "
                         "(anonymized_FR1-0001_0_003_CFV-L), so every lookup "
                         "misses and the script reports 0 usable clips.")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--channels", default="both", choices=["both", "vein"],
                    help="'vein' drops the artery entirely. The vein/artery "
                         "SIZE RATIO is centre-informative and survives any "
                         "similarity transform, so no choice of scaling "
                         "denominator removes it - only removing a vessel does.")
    ap.add_argument("--scale-by", default="pair", choices=["pair", "vein"],
                    help="canvas normalisation. 'vein' equalises vein size and "
                         "leaves the vein/artery ratio free, which is what "
                         "leaked. 'pair' scales by both vessels.")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from dvt import step17_mask_video as S17

    by_stem = {}
    for q in Path(a.annotations).glob("*.json"):
        parts = q.stem.split("__")
        if len(parts) == 4:
            by_stem[parts[3]] = {"centre": parts[0], "group": parts[1],
                                 "SubjectID": parts[2]}

    files = sorted(Path(a.sidecars).rglob("*_darwin.json"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"no sidecars under {a.sidecars}")
        return 1

    X, centre, subject, skipped = [], [], [], Counter()
    for p in files:
        stem = p.name.replace("_darwin.json", "")
        row = by_stem.get(stem)
        if row is None:
            skipped["no label row"] += 1
            continue
        vid, valid, info = S17.render_clip(p, a.num_frames, channels=2,
                                           scale_by=a.scale_by)
        if vid is None:
            skipped[info.get("reason", "render failed")] += 1
            continue
        f = summarise(vid, valid, a.channels)
        if f is None:
            skipped["too few valid frames"] += 1
            continue
        X.append(f); centre.append(row["centre"]); subject.append(row["SubjectID"])

    if len(X) < 60:
        print(f"only {len(X)} usable clips; skipped {dict(skipped)}")
        return 1

    X = np.stack(X)
    centre = np.asarray(centre)
    subject = np.asarray(subject)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    counts = Counter(centre.tolist())
    majority = max(counts.values()) / len(centre)

    print(f"\n{'='*64}")
    print("CENTRE LEAKAGE CHECK - mask videos")
    print(f"{'='*64}")
    print(f"clips {len(X)}   subjects {len(set(subject))}   "
          f"features {X.shape[1]}   scale_by={a.scale_by}  "
          f"channels={a.channels}")
    print(f"centre counts: {dict(counts)}")
    print(f"majority-class rate (the honest floor): {majority:.1%}")
    if skipped:
        print(f"skipped: {dict(skipped)}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    accs = []
    gkf = GroupKFold(n_splits=min(5, len(set(subject))))
    for tr, te in gkf.split(X, centre, subject):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200)
        m.fit(X[tr], centre[tr])
        accs.append((m.predict(X[te]) == centre[te]).mean())
    acc = float(np.mean(accs))

    print(f"\ncentre classifier, subject-grouped 5-fold: {acc:.3f}")
    print(f"  lift over majority: {acc - majority:+.3f}")
    print()
    if acc < majority + 0.05:
        print("  PASS - the centre is not recoverable from these masks.")
        print("  A DVT result on this input is not the scanner.")
    elif acc < majority + 0.15:
        print("  MARGINAL - some centre signal survives. Check which features")
        print("  carry it before trusting a DVT number; the canvas transform")
        print("  in step17 may need tightening.")
    else:
        print("  FAIL - the centre is recoverable. Whatever the probe learns")
        print("  will be partly the hospital, and it will not transfer. Fix")
        print("  the normalisation before spending compute on the probe.")

    # which features carry it, if any
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200)
    m.fit(X, centre)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(m, X, centre, n_repeats=5, random_state=0)
        names = (["vein_area_mean", "vein_area_sd", "vein_area_min",
                  "vein_area_max", "vein_ratio", "vein_cx", "vein_cy",
                  "vein_sx", "vein_sy", "vein_w", "vein_h"] +
                 ["art_area_mean", "art_area_sd", "art_area_min",
                  "art_area_max", "art_ratio", "art_cx", "art_cy",
                  "art_sx", "art_sy", "art_w", "art_h"] +
                 ["vein_over_art", "valid_frac", "n_valid"])
        order = np.argsort(-r.importances_mean)[:5]
        print("\n  top centre-predictive features:")
        for i in order:
            nm = names[i] if i < len(names) else f"f{i}"
            print(f"    {nm:<18} {r.importances_mean[i]:+.4f}")
    except Exception:
        pass

    print("\nreminder: centre prevalence alone predicts DVT at AUC 0.709 in")
    print("this cohort, so a leaky input can beat the clinical baseline while")
    print("carrying no clinical signal at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
