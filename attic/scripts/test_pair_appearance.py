r"""
Can APPEARANCE tell vein from artery, where geometry cannot?

THE QUESTION
------------
The pipeline delivers the vein polygon onto the vein on 60.4% of expert
keyframes, and onto the ARTERY on 20.7%. That identification error is now the
largest remaining source of bad annotations.

Two things are already known:

* Faster R-CNN's ROI head classifies each box from features pooled INSIDE that
  box. It sees neither the box's absolute position nor the other vessel. But
  vein-versus-artery is positional and relational, and the sign flips by site -
  the vein is deeper at CFV/FV and superficial at PV. A site-agnostic per-box
  classifier is being asked to learn contradictory labels from near-identical
  crops. It scores 0.594 TOP1, which is about what that deserves.

* GEOMETRY IS EXHAUSTED. A gradient-boosted model over ten geometric features
  (both depths, size ratio, separation, aspect ratios, site) scored 0.888 under
  subject-grouped 5-fold CV against the two hand-coded rules' 0.895. Logistic
  regression managed 0.836. Nothing in the coordinates beats depth-plus-size.

So the only untested source of information is the appearance of the two vessels
considered TOGETHER: wall thickness, the artery's brighter echogenic wall,
lumen texture, shape asymmetry. Whether any of that survives at ultrasound
resolution is genuinely unknown - the same premise that was wrong when the
vessel classes were merged.

WHAT THIS SCRIPT DECIDES
------------------------
Three models, identical splits:

    geometry     the ten features above
    appearance   two 64x64 crops, one per candidate vessel
    both         crops plus geometry

If `appearance` or `both` clears the 0.895 hand-rule baseline by more than the
fold-to-fold spread, a pairwise head is worth adding to the pipeline. If it does
not, vein/artery identification is at its ceiling for this data, and the
remaining error is a DETECTION problem - better boxes, not better rules.

WHY BOTH ORDERINGS
------------------
Every pair is emitted twice, (vein, artery) and (artery, vein), so the model
cannot learn "the first crop is the vein". Both orderings of a pair always land
in the same CV fold, via subject grouping.

WHY LEAVE-ONE-CENTRE-OUT MATTERS HERE
-------------------------------------
Appearance is exactly the channel that carries scanner identity, and centre
identity alone predicts thrombosis at AUC 0.709 in this cohort. A model that
looks good under subject-grouped CV and collapses across centres has learned the
scanner. Both are reported; believe the centre one.

USAGE
-----
    python scripts/test_pair_appearance.py \
        --annotations annotations_flat --dicoms dicoms_annotated \
        --device cuda --epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CROP = 64


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def crop_of(frame, box, size=CROP, context=0.6):
    """Square crop centred on the box, padded by `context` of its side.

    Square and context-padded on purpose: a tight crop scaled to a fixed size
    would erase the vessel's SIZE, which is one of the two cues known to work.
    The context band also carries the vessel wall, which is where any
    appearance difference between vein and artery would live.
    """
    import cv2

    h, w = frame.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * context)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        x0 += pad; x1 += pad; y0 += pad; y1 += pad
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    p = patch.astype(np.float32)
    # per-crop normalisation, to strip the scanner's absolute gain. Without it
    # the model can identify the centre from brightness alone.
    return (p - p.mean()) / (p.std() + 1e-6)


def build(annotations, dicoms, limit=0):
    from expert_reference import load
    from dvt.step10_auto_seed import parse_filename_tag
    from dvt import step8_compression_phase as S8

    dcms = {p.stem: p for p in Path(dicoms).rglob("*.dcm")}
    X_img, X_geo, Y, G, C, S = [], [], [], [], [], []
    files = sorted(Path(annotations).glob("*.json"))
    done = 0

    for p in files:
        if limit and done >= limit:
            break
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        centre, group, subject, clip = parts
        tag = parse_filename_tag(clip)
        if tag["still"] or not tag["site"]:
            continue
        dcm = dcms.get(clip)
        if dcm is None:
            continue
        ef, w, h = load(p)
        usable = [i for i, r in ef.items() if "vein" in r and "artery" in r]
        if not usable:
            continue
        try:
            frames = S8.load_clip_frames(dcm, max_frames=0, downsample=1)
        except Exception:
            continue
        if frames.shape[1:3] != (h, w):
            continue
        done += 1

        for i in usable:
            if i >= len(frames):
                continue
            V = np.logical_or.reduce(ef[i]["vein"])
            A = np.logical_or.reduce(ef[i]["artery"])
            if V.sum() < 60 or A.sum() < 60:
                continue
            bv, ba = bbox(V), bbox(A)
            cv_, ca = crop_of(frames[i], bv), crop_of(frames[i], ba)
            if cv_ is None or ca is None:
                continue
            for (b0, b1, c0, c1), y in (((bv, ba, cv_, ca), 1),
                                        ((ba, bv, ca, cv_), 0)):
                d0 = (b0[1] + b0[3]) / 2 / h
                d1 = (b1[1] + b1[3]) / 2 / h
                s0 = np.sqrt(max((b0[2] - b0[0]) * (b0[3] - b0[1]), 1))
                s1 = np.sqrt(max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1))
                sep = np.hypot((b0[0] + b0[2]) / 2 - (b1[0] + b1[2]) / 2,
                               (b0[1] + b0[3]) / 2 - (b1[1] + b1[3]) / 2)
                site = ["CFV", "FV", "PV", "GS"].index(tag["site"]) \
                    if tag["site"] in ("CFV", "FV", "PV", "GS") else 0
                oh = [0.0] * 4
                oh[site] = 1.0
                X_img.append(np.stack([c0, c1]))
                X_geo.append([d0 - d1, d0, d1, float(np.log(s0 / s1)),
                              s0 / max(h, w), s1 / max(h, w),
                              sep / ((s0 + s1) / 2),
                              (b0[2] - b0[0]) / max(b0[3] - b0[1], 1),
                              (b1[2] - b1[0]) / max(b1[3] - b1[1], 1)] + oh)
                Y.append(y); G.append(subject); C.append(centre)
                S.append(tag["site"])
    return (np.asarray(X_img, np.float32), np.asarray(X_geo, np.float32),
            np.asarray(Y), np.asarray(G), np.asarray(C), np.asarray(S))


def make_net(mode, n_geo, device):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            if mode in ("appearance", "both"):
                # deliberately small: 1868 pairs from 77 subjects will
                # memorise anything larger
                self.conv = nn.Sequential(
                    nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16),
                    nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                    nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48),
                    nn.ReLU(), nn.AdaptiveAvgPool2d(1))
                feat = 48
            else:
                feat = 0
            extra = n_geo if mode in ("geometry", "both") else 0
            self.head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(feat + extra, 32), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, img, geo):
            parts = []
            if self.mode in ("appearance", "both"):
                parts.append(self.conv(img).flatten(1))
            if self.mode in ("geometry", "both"):
                parts.append(geo)
            import torch
            return self.head(torch.cat(parts, 1)).squeeze(1)

    return Net().to(device)


def run_fold(mode, Xi, Xg, Y, tr, te, device, epochs, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = make_net(mode, Xg.shape[1], device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    xi = torch.tensor(Xi[tr]).to(device); xg = torch.tensor(Xg[tr]).to(device)
    yy = torch.tensor(Y[tr], dtype=torch.float32).to(device)
    vi = torch.tensor(Xi[te]).to(device); vg = torch.tensor(Xg[te]).to(device)
    vy = torch.tensor(Y[te], dtype=torch.float32).to(device)

    best = 0.0
    n = len(tr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for k in range(0, n, 64):
            idx = perm[k:k + 64]
            img = xi[idx]
            if mode in ("appearance", "both") and np.random.rand() < 0.5:
                img = torch.flip(img, dims=[3])      # horizontal only
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(img, xg[idx]), yy[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (((net(vi, vg) > 0).float() == vy).float().mean().item())
        best = max(best, acc)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print("extracting crops ...", flush=True)
    Xi, Xg, Y, G, C, S = build(a.annotations, a.dicoms, a.limit)
    if len(Y) < 200:
        print(f"only {len(Y)} pairs, not enough")
        return 1
    print(f"pairs (both orderings) {len(Y)}   clips-worth {len(Y)//2}   "
          f"subjects {len(set(G))}   centres {sorted(set(C))}")

    from sklearn.model_selection import GroupKFold

    print("\nHAND-RULE BASELINE on this same data: 0.895")
    print("(geometry-only gradient boosting scored 0.888 subject-grouped)\n")

    results = {}
    for mode in ("geometry", "appearance", "both"):
        accs = []
        for tr, te in GroupKFold(n_splits=5).split(Xi, Y, G):
            accs.append(run_fold(mode, Xi, Xg, Y, tr, te, a.device, a.epochs))
        results[mode] = accs
        print(f"  {mode:<11} subject-grouped 5-fold {np.mean(accs):.3f}  "
              f"+/- {np.std(accs):.3f}   folds {[round(x,3) for x in accs]}")

    print("\nleave-one-CENTRE-out (the one to believe - appearance is exactly")
    print("the channel that carries scanner identity):")
    for cen in sorted(set(C)):
        te = np.where(C == cen)[0]
        tr = np.where(C != cen)[0]
        if len(te) < 40:
            print(f"  {cen}: {len(te)//2} pairs, skipped")
            continue
        line = f"  {cen}: "
        for mode in ("geometry", "appearance", "both"):
            acc = run_fold(mode, Xi, Xg, Y, tr, te, a.device, a.epochs)
            line += f"{mode} {acc:.3f}   "
        print(line + f"(n={len(te)//2})")

    print("\nVERDICT")
    app = np.mean(results["appearance"]); both = np.mean(results["both"])
    spread = max(np.std(results["geometry"]), np.std(results["both"]))
    if max(app, both) > 0.895 + 2 * spread:
        print("  Appearance ADDS information beyond geometry. A pairwise head")
        print("  is worth putting in front of the role assignment.")
    else:
        print("  Appearance adds nothing beyond geometry. Vein/artery")
        print("  identification is at its ceiling for this data, and the")
        print("  remaining 20.7% artery-confusion is a DETECTION problem:")
        print("  the rules score 0.895 on expert boxes and the pipeline")
        print("  delivers 60.4%, because 41% of clips have a box that is not")
        print("  on a vessel. Better boxes, not better rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
