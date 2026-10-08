r"""
Does an automated run's masks predict DVT?

Ground truth comes from the folder structure captured in the flattened
annotation filenames written by `extract_annotations.py`:

    CENTRE__GROUP__SUBJECT__clip.json    GROUP is DVT or NON_DVT

Two features are scored:

  compression ratio   min/max vein area across frames. The clinical test is
                      whether the vein collapses under probe pressure, so a
                      HIGH ratio (does not collapse) indicates thrombus.

  clot fraction       peak clot area as a share of the vein it sits in,
                      available when the run used --propose-clot. This
                      measures thrombus directly rather than inferring it
                      from geometry.

Subject level takes the MAX across a subject's vessels: a patient has DVT if
ANY scanned site is thrombosed, so the least-compressible vessel is the one
that matters. This is also how the expert benchmark was computed, which keeps
the comparison honest.

    python scripts/score_dvt.py --report report_medsam2.json \
        --expert annotations_flat
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# measured on expert polygons, for comparison
EXPERT_RATIO_VESSEL = 0.7167
EXPERT_RATIO_SUBJECT = 0.8491
EXPERT_CLOT_SUBJECT = 0.8432


def auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if len(s) < 2:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        k = s == v
        if k.sum() > 1:
            r[k] = r[k].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def operating_point(scores, labels):
    """Threshold maximising Youden's J, reported with sens/spec.

    Sensitivity matters more than accuracy for a screening tool: at ~7%
    prevalence a model predicting "no DVT" everywhere scores 93% accuracy and
    catches nobody.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    best = None
    for t in np.unique(np.round(s, 3)):
        p = (s >= t).astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if best is None or sens + spec > best["sens"] + best["spec"]:
            best = {"threshold": round(float(t), 4),
                    "sens": round(sens, 3), "spec": round(spec, 3),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return best


def load_ground_truth(expert_dir: Path) -> dict[str, tuple[str, str, str]]:
    gt = {}
    for f in Path(expert_dir).glob("*.json"):
        parts = f.name.split("__", 3)
        if len(parts) != 4:
            continue
        centre, group, subject, rest = parts
        gt[rest.replace(".json", "")] = (centre, group, subject)
    return gt


def load_split(split_csv, which):
    """-> set of clip stems on the requested side, or None for 'all'."""
    if not split_csv or which == "all":
        return None
    import csv as _csv
    keep = set()
    with open(split_csv) as fh:
        for row in _csv.DictReader(fh):
            if row.get("split") == which:
                keep.add(row["clip"])
    if not keep:
        raise SystemExit(f"no split={which} rows in {split_csv}")
    return keep


def collect(report_path: Path, gt: dict, keep=None):
    rows, unmatched, off_split = [], 0, 0
    for clip in json.loads(Path(report_path).read_text()):
        key = clip["clip"]
        stem = key.replace(".dcm", "")
        if keep is not None and stem not in keep:
            off_split += 1
            continue
        if key not in gt:
            unmatched += 1
            continue
        centre, group, subject = gt[key]
        veins = [v for v in clip.get("vessels", []) if "Vein" in v.get("label", "")]
        if not veins:
            continue
        # the vessel the run tracked as primary is the first entry
        v = veins[0]
        tag = key.split("_")[-1]
        rows.append({
            "clip": key, "centre": centre, "subject": subject,
            "dvt": 1 if group.upper() == "DVT" else 0,
            "site": tag.split("-")[0],
            "ratio": v.get("ratio"),
            # The same measurement with vein/artery-contaminated frames
            # removed. Scoring both is the only way to tell whether cleaning
            # helps PREDICTION or merely makes the audit look tidy: argmax
            # zeroes measured overlap by construction, so a clean sidecar
            # proves nothing on its own.
            "ratio_clean": clip.get("compression_ratio_clean"),
            "clot_fraction": clip.get("clot_fraction"),
            "frac_overlapping": (clip.get("overlap") or {}).get(
                "frac_overlapping"),
            "frames_valid": v.get("frames_valid"),
            "frames_blank": v.get("frames_blank"),
        })
    return rows, unmatched


def score(rows, feature: str, higher_is_dvt: bool = True) -> dict:
    vals = [r[feature] for r in rows]
    y = [r["dvt"] for r in rows]
    have = [v is not None for v in vals]
    if sum(have) < 10 or len(set(np.array(y)[have])) < 2:
        return {"available": False, "n": int(sum(have))}

    s = np.array([v if v is not None else np.nan for v in vals], dtype=float)
    y = np.array(y)
    sign = 1.0 if higher_is_dvt else -1.0

    out = {"available": True, "n": int(sum(have)),
           "vessel_auc": round(auc(sign * s, y), 4)}

    # subject level: max across a subject's vessels
    by_subj = defaultdict(lambda: [-np.inf, 0])
    for v, lab, r in zip(s, y, rows):
        if np.isnan(v):
            continue
        cur = by_subj[r["subject"]]
        cur[0] = max(cur[0], sign * v)
        cur[1] = max(cur[1], lab)
    sv = np.array([a for a, _ in by_subj.values()])
    sy = np.array([b for _, b in by_subj.values()])
    if len(set(sy.tolist())) > 1:
        out["subject_auc"] = round(auc(sv, sy), 4)
        out["n_subjects"] = int(len(sy))
        out["subject_operating_point"] = operating_point(sv, sy)

    per_site = {}
    for site in sorted({r["site"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["site"] == site]
        ys = y[idx]
        if len(set(ys.tolist())) > 1:
            per_site[site] = round(auc(sign * s[idx], ys), 4)
    out["by_site"] = per_site

    per_centre = {}
    for c in sorted({r["centre"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["centre"] == c]
        ys = y[idx]
        if len(set(ys.tolist())) > 1:
            per_centre[c] = round(auc(sign * s[idx], ys), 4)
    out["by_centre"] = per_centre
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, nargs="+",
                    help="one or more autoannotate report JSONs")
    ap.add_argument("--split-csv", default=None,
                    help="frozen split from scripts/make_split.py")
    ap.add_argument("--split", default="all",
                    choices=["all", "train", "test"],
                    help="which side to score. Report TEST only when the "
                         "pipeline is final - every look at it costs a little "
                         "of its independence.")
    ap.add_argument("--expert", required=True,
                    help="annotations_flat, for the DVT/NON_DVT labels")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt = load_ground_truth(Path(a.expert))
    _keep = load_split(a.split_csv, a.split)
    if _keep is not None:
        print(f"scoring the {a.split.upper()} side only: {len(_keep)} clips")
    print(f"ground truth: {len(gt)} clips "
          f"({sum(1 for v in gt.values() if v[1].upper() == 'DVT')} from DVT subjects)\n")

    results = {}
    for rp in a.report:
        rows, unmatched = collect(Path(rp), gt, _keep)
        name = Path(rp).stem
        if not rows:
            print(f"{name}: no clips matched ground truth")
            continue
        npos = sum(r["dvt"] for r in rows)
        blanks = [r["frames_blank"] for r in rows if r["frames_blank"] is not None]

        ratio = score(rows, "ratio", higher_is_dvt=True)
        ratio_clean = score(rows, "ratio_clean", higher_is_dvt=True)
        clot = score(rows, "clot_fraction", higher_is_dvt=True)
        results[name] = {"n_clips": len(rows), "n_dvt": npos,
                         "unmatched": unmatched,
                         "median_blank_frames": float(np.median(blanks)) if blanks else None,
                         "compression_ratio": ratio,
                         "compression_ratio_clean": ratio_clean,
                         "clot_fraction": clot}

        print(f"=== {name} ===")
        print(f"  {len(rows)} clips ({npos} DVT+), {unmatched} unmatched, "
              f"median blank frames {np.median(blanks) if blanks else '-'}")
        if ratio_clean.get("available"):
            print(f"  ratio (overlap-cleaned): vessel {ratio_clean['vessel_auc']} | "
                  f"subject {ratio_clean.get('subject_auc')}")
        if ratio.get("available"):
            print(f"  compression ratio: vessel {ratio['vessel_auc']} | "
                  f"subject {ratio.get('subject_auc')}")
            print(f"    by site  : {ratio['by_site']}")
            print(f"    by centre: {ratio['by_centre']}")
            op = ratio.get("subject_operating_point")
            if op:
                print(f"    subject operating point: sens {op['sens']} spec {op['spec']} "
                      f"(tp{op['tp']} fp{op['fp']} fn{op['fn']} tn{op['tn']})")
        if clot.get("available"):
            print(f"  clot fraction    : vessel {clot['vessel_auc']} | "
                  f"subject {clot.get('subject_auc')}  (n={clot['n']})")
        else:
            print("  clot fraction    : not available "
                  "(re-run with --propose-clot to populate it)")
        print()

    print("--- expert benchmark ---")
    print(f"  compression ratio: vessel {EXPERT_RATIO_VESSEL} | subject {EXPERT_RATIO_SUBJECT}")
    print(f"  clot presence    : subject {EXPERT_CLOT_SUBJECT} "
          "(sens 0.727, spec 0.959)")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
