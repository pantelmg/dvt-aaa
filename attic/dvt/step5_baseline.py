"""
Step 5 - The baseline you must beat.

Compressibility is, clinically, a geometric fact: does the vein lumen collapse
under probe pressure? Given per-frame lumen masks, that is a threshold on the
min/max area ratio. No deep temporal model required.

Run this BEFORE any V-JEPA work. If a one-parameter threshold on the area curve
already separates compressible from non-compressible clips, the honest project
is "MedSAM + geometry", and a video transformer is unjustified complexity.
This script produces the number that decides that.

Uses only numpy - no torch - so it can run on day one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from . import step2_json_to_masks as S2


@dataclass
class ClipFeature:
    clip_id: str
    patient_id: str
    vessel: str
    site: str | None
    ratio: float
    n_frames_tracked: int
    label: str          # ground-truth Compressibility from US-DICOMizer
    thrombosis: str


def extract_features(manifest_path: Path | None = None) -> list[ClipFeature]:
    manifest_path = manifest_path or (C.INDEX_DIR / "clips.json")
    clips = json.loads(Path(manifest_path).read_text())
    feats: list[ClipFeature] = []

    for clip in clips:
        if not clip.get("json_path"):
            continue
        ann = S2.parse_annotation_json(Path(clip["json_path"]), clip["clip_id"])
        if ann.compressibility is None:
            continue
        for vessel, curve in S2.area_curves(ann).items():
            r = S2.compression_ratio(curve)
            if r is None:
                continue  # need >=2 annotated frames to judge compression
            feats.append(ClipFeature(
                clip_id=clip["clip_id"],
                patient_id=clip["patient_id"],
                vessel=vessel,
                site=C.label_to_site(vessel),
                ratio=r,
                n_frames_tracked=len(curve),
                label=str(ann.compressibility),
                thrombosis=str(ann.thrombosis),
            ))
    return feats


def sweep_threshold(feats: list[ClipFeature], positive="No"):
    """Find the ratio threshold best separating non-compressible from the rest.

    `positive` = the Compressibility value meaning 'did NOT compress', i.e. the
    clinically abnormal class we want to detect.
    """
    y = np.array([1 if f.label == positive else 0 for f in feats])
    x = np.array([f.ratio for f in feats])
    if y.sum() == 0 or y.sum() == len(y):
        return {"error": "only one class present", "n": len(y),
                "n_positive": int(y.sum())}

    # Sweep midpoints between consecutive observed values rather than the
    # values themselves, so the chosen threshold never sits exactly on a
    # training point (where a future sample at that value flips arbitrarily).
    vals = np.unique(np.round(x, 4))
    mids = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
    candidates = np.concatenate([[vals[0] - 1e-3], mids, [vals[-1] + 1e-3]])

    best = None
    for thr in candidates:
        pred = (x >= thr).astype(int)  # high ratio = did not collapse = abnormal
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        youden = sens + spec - 1
        if best is None or youden > best["youden"]:
            best = {"threshold": float(thr), "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4), "youden": round(youden, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    best["auc"] = round(float(_auc(x, y)), 4)
    best["n"] = len(y)
    best["n_positive"] = int(y.sum())
    return best


def _auc(scores, labels):
    """Rank-based AUC (Mann-Whitney), tie-aware."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s = np.asarray(scores)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos, neg = labels == 1, labels == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(feats: list[ClipFeature]) -> dict:
    out = {"overall": sweep_threshold(feats)}
    by_site: dict[str, list[ClipFeature]] = {}
    for f in feats:
        by_site.setdefault(f.site or "unknown", []).append(f)
    out["by_site"] = {k: sweep_threshold(v) for k, v in sorted(by_site.items())}
    return out


if __name__ == "__main__":
    feats = extract_features()
    print(f"{len(feats)} vessel-clips with a computable compression ratio")
    if not feats:
        raise SystemExit("No usable data - need >=2 annotated frames per vessel.")
    res = report(feats)
    print(json.dumps(res, indent=2))
    (C.INDEX_DIR / "baseline_report.json").write_text(json.dumps(res, indent=2))
    o = res["overall"]
    if "auc" in o:
        print(f"\nBASELINE AUC = {o['auc']}  (sens {o['sensitivity']}, "
              f"spec {o['specificity']} @ ratio>={o['threshold']})")
        print("Any learned temporal model must beat this to be worth the complexity.")
