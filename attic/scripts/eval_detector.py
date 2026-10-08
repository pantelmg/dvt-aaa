r"""
Score one or more detector checkpoints on a COMMON validation set.

WHY THIS IS SEPARATE FROM TRAINING
----------------------------------
Two training runs produced these numbers:

    --keyframes-only    vessel TOP1 0.9038   clot 0.6477   (val: 1120 / 105 boxes)
    all frames          vessel TOP1 0.8264   clot 0.4550   (val: 1472 / 260 boxes)

They held out the SAME 16 subjects but evaluated on DIFFERENT box sets: the
all-frames run's validation includes tool-INTERPOLATED polygons, which are the
annotation software's guesses rather than anything a human drew. A model can be
marked wrong for disagreeing with an interpolation. So the gap admits two
readings with opposite consequences:

    (a) interpolated polygons poison TRAINING      -> drop them
    (b) interpolated polygons poison MEASUREMENT   -> train on them, never
                                                      evaluate on them

`--keyframes-only` here forces every checkpoint onto the human-drawn subset, so
the comparison is like-for-like and (a) and (b) separate.

PAIR RECALL - the metric the pipeline actually needs
----------------------------------------------------
TOP1 asks whether the single highest-scoring vessel box matches ANY ground-truth
vessel. Under the merged scheme that is easier than the pipeline's requirement:
a frame contains two vessels and `step13.select_vessel_pair` needs BOTH. A model
that finds the artery perfectly and never finds the vein scores 1.00 on vessel
TOP1 and is useless as a seeder.

PAIR asks whether the top two vessel predictions match two DISTINCT ground-truth
vessels at IoU >= thr. That is what determines whether a clip gets an artery
track at all - and a clip seeded from one vessel has no second track, so no
vein/artery collision can even be detected on it later.

Expect PAIR to be well below TOP1. It is the honest number.

USAGE
-----
    python scripts/eval_detector.py --annotations annotations_flat \
        --dicoms dicoms_annotated --keyframes-only --device cuda \
        --checkpoint checkpoints/vessel_detector_kf.pth \
        --checkpoint checkpoints/vessel_detector_all.pth
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from train_detector import (  # noqa: E402
    DetectionDataset, build_index, collate, split_by_subject,
)


def box_iou(a, b):
    """IoU of one box against many. a: (4,), b: (N, 4) -> (N,)."""
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(b) == 0:
        return np.zeros(0)
    a = np.asarray(a, dtype=float)
    ix0 = np.maximum(a[0], b[:, 0]); iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2]); iy1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa + bb - inter, 1e-9)


@torch.no_grad()
def score(model, loader, device, vessel_ids, clot_id, names, iou_thr=0.5):
    """`vessel_ids` is a SET, because a split checkpoint has two vessel classes.

    The first version took a single id, so on a split checkpoint PAIR asked
    whether two VEINS were found on one frame - which almost never happens,
    since a frame carries one vein and one artery. It reported 133 qualifying
    frames against 458 for the merged checkpoint and a PAIR of 0.26 that
    measured nothing.
    """
    model.eval()
    hit, total, top1, imgs_with = Counter(), Counter(), Counter(), Counter()
    pair_ok = pair_total = 0

    for imgs, targets in loader:
        outs = model([i.to(device) for i in imgs])
        for out, t in zip(outs, targets):
            pb = out["boxes"].cpu().numpy()
            pl = out["labels"].cpu().numpy()
            ps = out["scores"].cpu().numpy()
            gb = t["boxes"].numpy()
            gl = t["labels"].numpy()

            for g, cls in zip(gb, gl):
                total[int(cls)] += 1
                cand = pb[pl == cls]
                if len(cand) and box_iou(g, cand).max() >= iou_thr:
                    hit[int(cls)] += 1

            for cls in {int(c) for c in gl}:
                imgs_with[cls] += 1
                sel = pl == cls
                if not sel.any():
                    continue
                best = pb[sel][int(np.argmax(ps[sel]))]
                if box_iou(best, gb[gl == cls]).max() >= iou_thr:
                    top1[cls] += 1

            # ---- pair recall, only on frames that HAVE two vessels
            gsel = np.isin(gl, list(vessel_ids))
            gv = gb[gsel]
            if len(gv) < 2:
                continue
            pair_total += 1
            sel = np.isin(pl, list(vessel_ids))
            if sel.sum() < 2:
                continue
            order = np.argsort(-ps[sel])[:2]
            top2 = pb[sel][order]
            # greedy assignment: each prediction must claim a DIFFERENT gt box
            claimed, ok = set(), 0
            for p in top2:
                ious = box_iou(p, gv)
                for j in np.argsort(-ious):
                    if ious[j] >= iou_thr and j not in claimed:
                        claimed.add(int(j))
                        ok += 1
                        break
            if ok == 2:
                pair_ok += 1

    return {
        "recall": {names.get(c, str(c)): round(hit[c] / total[c], 4)
                   for c in sorted(total) if total[c]},
        "top1": {names.get(c, str(c)): round(top1[c] / imgs_with[c], 4)
                 for c in sorted(imgs_with) if imgs_with[c]},
        "pair": (round(pair_ok / pair_total, 4) if pair_total else None),
        "pair_frames": pair_total,
        "boxes": {names.get(c, str(c)): total[c] for c in sorted(total)},
    }


class _SiteDataset(torch.utils.data.Dataset):
    """DetectionDataset plus the site tag and the UNMERGED role labels.

    Role evaluation needs ground truth the merged scheme throws away: after
    `merge_class` every vessel is class 1, so the checkpoint cannot be asked
    whether it put the vein label on the vein. The records here keep the
    original 1=vein / 2=artery / 3=clot labels while the MODEL still predicts
    the merged classes - the two are compared geometrically, and the labels are
    used only to judge the role assignment.
    """

    def __init__(self, records, img_size):
        from dvt.step10_auto_seed import parse_filename_tag
        self.inner = DetectionDataset(records, img_size, augment=False)
        self.sites = [parse_filename_tag(r["clip"])["site"] or ""
                      for r in records]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x, t = self.inner[i]
        t = dict(t)
        t["site"] = self.sites[i]
        return x, t


@torch.no_grad()
def score_selector(model, records, img_size, device, vessel_ids, names,
                   iou_thr=0.5, batch_size=4, workers=4):
    """Run the REAL seeding logic and check what it produces.

    `score()`'s PAIR metric takes the top two boxes by score, which is stricter
    than the pipeline: `select_vessel_pair` searches all candidate pairs and
    rejects duplicates on box IoU, so where the model emits vein, a duplicate
    of vein, and the artery, the naive metric fails and the real selector
    recovers. True seeding performance sits between the two.

    Two things measured here that nothing else measures:

    SELECTOR PAIR  does select_vessel_pair return two boxes matching two
                   DISTINCT ground-truth vessels
    ROLE           is the box it labels 'vein' actually on the vein

    ROLE has never been measured end to end. The 95-96% figure behind
    `step10.SITE_PRIOR` came from EXPERT boxes; applying the same priors to
    detector output is a different question, and a worse one, because a box
    that is off-vessel makes the depth comparison meaningless.
    """
    from dvt import step13_detector_seed as S13

    loader = torch.utils.data.DataLoader(
        _SiteDataset(records, img_size), batch_size=batch_size, shuffle=False,
        num_workers=workers,
        collate_fn=lambda b: ([x for x, _ in b], [t for _, t in b]))

    model.eval()
    n_pair_frames = 0
    sel_pair = naive_pair = 0
    role_total = role_ok = 0
    role_by_site = defaultdict(lambda: [0, 0])
    note_counts = Counter()

    for imgs, targets in loader:
        outs = model([i.to(device) for i in imgs])
        for img, out, t in zip(imgs, outs, targets):
            gb = t["boxes"].numpy()
            gl = t["labels"].numpy()
            gv = gb[gl == 1]          # expert VEIN boxes (unmerged labels)
            ga = gb[gl == 2]          # expert ARTERY boxes
            if len(gv) < 1 or len(ga) < 1:
                continue
            n_pair_frames += 1
            shape = (img.shape[1], img.shape[2])

            pb = out["boxes"].cpu().numpy()
            pl = out["labels"].cpu().numpy()
            ps = out["scores"].cpu().numpy()
            sel = np.isin(pl, list(vessel_ids))
            dets = [S13.Detection(box=[float(v) for v in pb[k]],
                                  cls=int(pl[k]), score=float(ps[k]))
                    for k in np.where(sel)[0]]
            # Preserve the detector's OWN role where it has one. A split
            # checkpoint's vein/artery call is what the margin gate in
            # select_vessel_pair falls back to when the depth margin is
            # ambiguous; flattening everything to "vessel" would silently
            # evaluate the merged code path on a split model.
            for d in dets:
                d.role = names.get(d.cls, "vessel")
                if d.role not in ("vein", "artery"):
                    d.role = "vessel"

            # ---- naive top-2, for a like-for-like comparison with score()
            top2 = sorted(dets, key=lambda d: -d.score)[:2]
            if len(top2) == 2:
                claimed, ok = set(), 0
                allgt = np.vstack([gv, ga])
                for d in top2:
                    ious = box_iou(d.box, allgt)
                    for j in np.argsort(-ious):
                        if ious[j] >= iou_thr and j not in claimed:
                            claimed.add(int(j)); ok += 1
                            break
                naive_pair += (ok == 2)

            vein, art, note = S13.select_vessel_pair(dets, shape,
                                                     t.get("site") or None)
            note_counts[note] += 1
            if vein is None or art is None:
                continue

            allgt = np.vstack([gv, ga])
            claimed, ok = set(), 0
            for d in (vein, art):
                ious = box_iou(d.box, allgt)
                for j in np.argsort(-ious):
                    if ious[j] >= iou_thr and j not in claimed:
                        claimed.add(int(j)); ok += 1
                        break
            if ok != 2:
                continue
            sel_pair += 1

            # ---- role: is the box called 'vein' on the vein?
            role_total += 1
            correct = box_iou(vein.box, gv).max() > box_iou(vein.box, ga).max()
            role_ok += bool(correct)
            site = t.get("site") or "?"
            role_by_site[site][0] += bool(correct)
            role_by_site[site][1] += 1

    return {
        "frames": n_pair_frames,
        "naive_pair": round(naive_pair / n_pair_frames, 4) if n_pair_frames else None,
        "selector_pair": round(sel_pair / n_pair_frames, 4) if n_pair_frames else None,
        "role": round(role_ok / role_total, 4) if role_total else None,
        "role_n": role_total,
        "role_by_site": {k: (round(v[0] / v[1], 4), v[1])
                         for k, v in sorted(role_by_site.items()) if v[1]},
        "notes": dict(note_counts),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="repeat to compare several checkpoints")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="evaluate ONLY on human-drawn polygons. Use this for "
                         "any cross-checkpoint comparison: a model marked wrong "
                         "for disagreeing with an interpolation is being "
                         "measured against the tool, not the radiologist.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0,
                    help="must match training, or the 'validation' set will "
                         "contain subjects the checkpoint trained on")
    ap.add_argument("--holdout-centre", default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    from dvt import step13_detector_seed as S13

    records = build_index(Path(a.annotations), Path(a.dicoms),
                          keyframes_only=a.keyframes_only)
    if not records:
        print("no records found")
        return 1

    results = {}
    for ck_path in a.checkpoint:
        model, img_size, names = S13.load_detector(ck_path, a.device)
        inv = {v: k for k, v in names.items()}
        merged = "vessel" in inv
        clot_id = inv.get("clot", 3)
        vessel_ids = {i for i, nm in names.items()
                      if nm in ("vessel", "vein", "artery")} or {1}

        # remap the split-scheme labels onto this checkpoint's scheme
        recs = [{**r, "boxes": [list(b) for b in r["boxes"]]} for r in records]
        if merged:
            vid = min(vessel_ids)
            for r in recs:
                for b in r["boxes"]:
                    b[4] = {1: vid, 2: vid, 3: clot_id}[int(b[4])]

        tr, va, how = split_by_subject(recs, a.holdout_centre, a.val_frac,
                                       seed=a.seed, clot_id=clot_id)
        loader = torch.utils.data.DataLoader(
            DetectionDataset(va, img_size, augment=False),
            batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
            collate_fn=collate)

        res = score(model, loader, a.device, vessel_ids, clot_id, names, a.iou)
        res["split"] = how
        res["scheme"] = "merged" if merged else "split"

        # Selector evaluation uses the UNMERGED records: the roles are the
        # thing being judged, so they cannot be merged away first. The split
        # is recomputed on the same seed and clot id so the held-out subjects
        # are identical.
        _tr2, va2, _how2 = split_by_subject(
            [{**r, "boxes": [list(b) for b in r["boxes"]]} for r in records],
            a.holdout_centre, a.val_frac, seed=a.seed, clot_id=3)
        res["selector"] = score_selector(
            model, va2, img_size, a.device, vessel_ids, names, a.iou,
            a.batch_size, a.workers)
        results[Path(ck_path).name] = res
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()

    src = "human keyframes only" if a.keyframes_only else "ALL polygons"
    print(f"\n{'='*72}")
    print(f"COMMON VALIDATION SET - {src}, IoU >= {a.iou}")
    print(f"{'='*72}")
    first = next(iter(results.values()))
    print(f"{first['split']}")
    print(f"val boxes: {first['boxes']}")
    print(f"frames with two vessels: {first['pair_frames']}\n")

    print(f"{'checkpoint':<34}{'vein/vessel':>13}{'artery':>9}{'clot':>9}{'PAIR':>9}")
    for name, r in results.items():
        v = r["top1"].get("vein", r["top1"].get("vessel"))
        ar = r["top1"].get("artery")
        c = r["top1"].get("clot")
        p = r["pair"]
        print(f"{name[:32]:<34}{v if v is not None else '-':>13}"
              f"{ar if ar is not None else '-':>9}"
              f"{c if c is not None else '-':>9}{p if p is not None else '-':>9}")

    print("\nrecall, for continuity with the old numbers:")
    for name, r in results.items():
        print(f"  {name[:32]:<34}{r['recall']}")

    print(f"\n{'-'*72}")
    print("THE REAL SEEDING LOGIC (select_vessel_pair on detector output)")
    print(f"{'-'*72}")
    print(f"{'checkpoint':<34}{'naive':>8}{'selector':>10}{'ROLE':>8}{'n':>7}")
    for name, r in results.items():
        sl = r["selector"]
        print(f"{name[:32]:<34}"
              f"{sl['naive_pair'] if sl['naive_pair'] is not None else '-':>8}"
              f"{sl['selector_pair'] if sl['selector_pair'] is not None else '-':>10}"
              f"{sl['role'] if sl['role'] is not None else '-':>8}"
              f"{sl['role_n']:>7}")

    for name, r in results.items():
        sl = r["selector"]
        print(f"\n  {name[:40]}")
        print(f"    outcomes: {sl['notes']}")
        if sl["role_by_site"]:
            print("    role accuracy by site (n):")
            for site, (acc, n) in sl["role_by_site"].items():
                print(f"      {site:<5} {acc:>7.3f}  ({n})")

    print("\nnaive     top two boxes by score - the stricter figure")
    print("selector  what select_vessel_pair actually returns; it searches all")
    print("          pairs and rejects duplicates, so it should be >= naive")
    print("ROLE      of the pairs it got right, how often the box labelled")
    print("          'vein' is on the vein. The 95-96% behind SITE_PRIOR came")
    print("          from EXPERT boxes; this is the same rule on detector")
    print("          output, which is a harder and more honest question.")
    print("\nA clip where only one vessel is found gets no artery track, so no")
    print("vein/artery collision can be detected on it downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
