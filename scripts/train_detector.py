r"""
Train a vessel detector to replace the heuristic seeder.

WHY
---
`validate_seeder.py` compared 1790 auto-annotated frames against expert masks:

    median IoU vs expert vein   0.0004
    clips with median IoU >0.5      6%
    clips with median IoU <0.1     66%
    within-clip IoU spread (p90-p10) 0.018

The seeder picks one structure at frame 0 and tracks it faithfully for the
whole clip - the tracking works, the initial choice is wrong two thirds of the
time. And the degradation test showed the compression ratio tolerates almost
any boundary error (floor 0.68 even when the vessel is lost on 60% of clips),
so segmentation quality was never the bottleneck. Seeding was.

This replaces ~200 lines of thresholds and anatomical rules with a detector
trained on the expert polygons, converted to boxes:

    2720 images, 7154 boxes -> vein 3357, artery 2977, clot 820

The pipeline becomes: detector proposes boxes -> MedSAM segments them, which is
how SAM was designed to be used inside a larger system.

WHAT IT PREDICTS
----------------
Three classes: vein, artery, clot. Clot gets its own class here, which the
promptable MedSAM formulation could not express - prompting the vein box with a
"Clot" label returned a mask byte-identical to the vein on 201/201 frames.

    python scripts/train_detector.py --annotations annotations_flat \
        --dicoms dicoms_annotated --device cuda --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt import step10_auto_seed as S10

#: Two schemes. MERGED is the default and the one to use.
#:
#: Vein and artery are not visually separable on a still B-mode frame - that
#: is the clinical reason compression is the diagnostic test. Training a
#: detector to tell them apart from appearance asks for information the image
#: does not carry, and it measurably failed: the auto "vein" sat on the expert
#: ARTERY on 13-27% of frames.
#:
#: Merging gives the vessel class 6316 boxes instead of 3344 and 2972, and
#: frees the classification head from a distinction it cannot learn. The
#: vein/artery decision moves to geometry, which is good at it: 96.0% at CFV,
#: 95.7% at FV, 95.2% at PV, 85.4% at GS, measured clip-level on this same
#: expert set (`step10.SITE_PRIOR`).
#:
#: Clot is out of scope for this pipeline (see README §7), so the detector is
#: trained on vein and artery only. Clot polygons are still RECOGNISED by
#: `kind_of` and then dropped, rather than silently falling through as
#: background, so the count is reported.
#:
#: The merged single-"vessel" scheme that used to be the default is gone with
#: it: merging only paid off when there was a second class to keep, and v14
#: trained SPLIT. `attic/scripts/train_detector.py` has both schemes, the clot
#: class, and the oversampling that went with it.
CLASS_NAMES = {1: "vein", 2: "artery"}
NUM_CLASSES = 3
CLOT_ID = 3                                 # kind_of's code, dropped on index


def kind_of(name: str) -> int | None:
    n = (name or "").strip()
    if n.lower() == "clot":
        return 3
    if n.lower() == "other":
        return None
    tok = n.split()[0].upper() if n else ""
    if tok.endswith("V"):
        return 1
    if tok.endswith("A"):
        return 2
    return None


def build_index(ann_dir: Path, dicom_dir: Path, include_stills: bool = True,
                keyframes_only: bool = False):
    """One record per annotated FRAME, with every box on that frame.

    Stills are kept by default: a compression ratio is undefined for them, but
    a detector only needs to see vessels, and they are 683 of 1044 files. More
    views of the anatomy is exactly what a detector wants.
    """
    dicoms = {p.stem: p for p in Path(dicom_dir).rglob("*.dcm")}
    records, missing = [], 0

    for f in sorted(Path(ann_dir).glob("*.json")):
        parts = f.name.split("__", 3)
        if len(parts) != 4:
            continue
        centre, group, subject, rest = parts
        clip = rest.replace(".json", "")
        dcm = dicoms.get(clip)
        if dcm is None:
            missing += 1
            continue
        tag = S10.parse_filename_tag(clip)
        if tag["still"] and not include_stills:
            continue

        j = json.loads(f.read_text(encoding="utf-8"))
        slots = (j.get("item") or {}).get("slots") or [{}]
        h = int(slots[0].get("height") or 0)
        w = int(slots[0].get("width") or 0)

        per = defaultdict(list)
        for a in j.get("annotations") or []:
            cls = kind_of(a.get("name", ""))
            if cls is None:
                continue
            for fk, fv in (a.get("frames") or {}).items():
                if "polygon" not in fv or not str(fk).lstrip("-").isdigit():
                    continue
                if keyframes_only and not fv.get("keyframe", True):
                    continue        # tool-interpolated, not human-drawn
                pts = np.array([[p["x"], p["y"]]
                                for p in fv["polygon"]["paths"][0]], dtype=float)
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                if x1 - x0 < 4 or y1 - y0 < 4:      # degenerate box
                    continue
                per[int(fk)].append([float(x0), float(y0), float(x1), float(y1), cls])

        for frame, boxes in per.items():
            if boxes:
                records.append({"dcm": str(dcm), "frame": frame, "boxes": boxes,
                                "centre": centre, "subject": subject,
                                "group": group, "clip": clip, "h": h, "w": w})
    if missing:
        print(f"WARNING {missing} annotation files had no matching DICOM")
    return records


class DetectionDataset(torch.utils.data.Dataset):
    """Frames plus boxes, in torchvision detection format.

    Module-level class, not defined inside a factory: DataLoader workers on
    Windows and Python 3.14+ are spawned via pickle, and a locally-defined
    class raises `Can't pickle local object`.
    """

    def __init__(self, records, img_size=800, augment=False):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def __len__(self):
        return len(self.records)

    def _frame(self, dcm, frame):
        key = (dcm, frame)
        if key in self._cache:
            return self._cache[key]
        import pydicom
        from pydicom.pixels import iter_pixels
        from dvt import step8_load_clips as S1

        ds = pydicom.dcmread(dcm, stop_before_pixels=True)
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        idx = min(frame, n - 1)
        arr = None
        for a in iter_pixels(dcm, indices=[idx]):
            arr = a
            break
        g = S1._to_luma(arr)
        if g.dtype != np.uint8:
            # `_scale_clip` passes uint8 through untouched, so for 8-bit source
            # data (the common case here) train and inference are identical.
            # For deeper data this uses SINGLE-frame statistics while inference
            # via step8.load_clip_frames uses CLIP-WIDE ones - a mismatch in
            # gain between training and deployment. Flagged rather than
            # silently differing; fixing it needs the whole clip decoded here,
            # which is the memory blow-up load_clip_frames exists to avoid.
            g = S1._scale_clip([g])[0]
        if len(self._cache) > 3000:
            self._cache.clear()
        self._cache[key] = g
        return g

    def __getitem__(self, i):
        import torch.nn.functional as F

        r = self.records[i]
        gray = self._frame(r["dcm"], r["frame"])
        h, w = gray.shape

        x = torch.from_numpy(gray).float()[None, None] / 255.0
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)[0]

        # Annotation coordinates are in the JSON slot's frame, which may differ
        # from the decoded pixel array; rescale before applying the resize.
        sx = (w / r["w"]) if r["w"] else 1.0
        sy = (h / r["h"]) if r["h"] else 1.0

        boxes, labels = [], []
        for x0, y0, x1, y1, cls in r["boxes"]:
            b = [x0 * sx * scale, y0 * sy * scale, x1 * sx * scale, y1 * sy * scale]
            b[0], b[2] = max(0.0, b[0]), min(float(nw), b[2])
            b[1], b[3] = max(0.0, b[1]), min(float(nh), b[3])
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            boxes.append(b)
            labels.append(int(cls))

        if self.augment and boxes:
            x, boxes = self._augment(x, boxes, nw)

        if boxes:
            tb = torch.tensor(boxes, dtype=torch.float32)
            tl = torch.tensor(labels, dtype=torch.int64)
        else:
            # An EMPTY target, not a fake one. This used to emit
            # [[0,0,1,1]] labelled VEIN, which taught the detector that a
            # 1x1 patch in the top-left corner is a vein - the corner being
            # where ultrasound frames carry their black border. torchvision
            # accepts zero-box targets as long as the shapes are right.
            tb = torch.zeros((0, 4), dtype=torch.float32)
            tl = torch.zeros((0,), dtype=torch.int64)

        return x, {"boxes": tb, "labels": tl}

    @staticmethod
    def _clip_boxes(boxes, nw, nh, min_side=4.0):
        out = []
        for b in boxes:
            b = [max(0.0, min(b[0], nw)), max(0.0, min(b[1], nh)),
                 max(0.0, min(b[2], nw)), max(0.0, min(b[3], nh))]
            if b[2] - b[0] >= min_side and b[3] - b[1] >= min_side:
                out.append(b)
        return out

    def _augment(self, x, boxes, nw):
        """Horizontal flip plus photometric jitter.

        There was no augmentation before this: `augment` was stored on the
        dataset and never read, so every epoch saw the identical 2720 frames.
        With five centres whose images differ in size, gain and speckle - and
        whose scanner identity alone predicts thrombosis at AUC 0.709 - that
        is exactly the setup in which a detector memorises the centre.

        HORIZONTAL flip only. A vertical flip would invert depth, and depth is
        the cue that separates vein from artery: the femoral vein lies deeper
        than its artery (96% at CFV), the popliteal vein superficial to its
        own (95%). Training on upside-down frames would destroy the signal
        `step10.arbitrate_detections` depends on. Left-right is safe because
        the medial/lateral relationship is not recorded in this cohort and is
        not used anywhere in the pipeline.
        """
        import torch.nn.functional as F

        nh = x.shape[1]
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            boxes = [[nw - b[2], b[1], nw - b[0], b[3]] for b in boxes]

        # ---- scale jitter. Vessel size varies 5.6x across centres (pixel
        # area), and the model must not key on absolute size; it also makes
        # every epoch see a slightly different image, which matters when the
        # effective sample size is 64 subjects.
        if random.random() < 0.7:
            f = random.uniform(0.80, 1.25)
            sh, sw = max(int(round(nh * f)), 16), max(int(round(nw * f)), 16)
            x = F.interpolate(x[None], size=(sh, sw), mode="bilinear",
                              align_corners=False)[0]
            boxes = [[b[0] * sw / nw, b[1] * sh / nh,
                      b[2] * sw / nw, b[3] * sh / nh] for b in boxes]
            nh, nw = sh, sw

        # ---- translation by padding and cropping, so vessels are not always
        # near the frame centre where the annotator framed them
        if random.random() < 0.5:
            dx = int(random.uniform(-0.08, 0.08) * nw)
            dy = int(random.uniform(-0.08, 0.08) * nh)
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                x[:, :dy, :] = 0
            elif dy < 0:
                x[:, dy:, :] = 0
            if dx > 0:
                x[:, :, :dx] = 0
            elif dx < 0:
                x[:, :, dx:] = 0
            boxes = [[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                     for b in boxes]
            boxes = self._clip_boxes(boxes, nw, nh)

        # ---- gain and contrast. Measured ~29% frame-to-frame variation on a
        # real clip and far more between scanners.
        if random.random() < 0.8:
            gain = 1.0 + random.uniform(-0.30, 0.30)
            bias = random.uniform(-0.10, 0.10)
            x = (x * gain + bias).clamp(0.0, 1.0)

        # ---- gamma, which is where scanner presets differ most visibly
        if random.random() < 0.5:
            x = x.clamp(1e-4, 1.0).pow(random.uniform(0.7, 1.45))

        # ---- multiplicative speckle. Ultrasound noise is speckle, not
        # Gaussian: adding Gaussian noise would train against the wrong
        # corruption. Kept mild so anechoic lumen stays anechoic.
        if random.random() < 0.4:
            x = (x * (1.0 + torch.randn_like(x) * 0.06)).clamp(0.0, 1.0)

        return x, boxes


def collate(batch):
    return tuple(zip(*batch))


def split_from_csv(records, split_csv, val_frac=0.2, seed=0, clot_id=2):
    """Honour a FROZEN split: test subjects are excluded entirely.

    The detector must never see a test subject, or the pipeline's output on
    that subject is not an out-of-sample measurement. So `split=test` rows are
    dropped here, and the remaining training subjects are split again into
    train and internal validation - the detector still needs a validation set
    to select a checkpoint, and it has to come from inside the training half.
    """
    import csv as _csv

    keep = set()
    with open(split_csv) as fh:
        for row in _csv.DictReader(fh):
            if row.get("split") == "train":
                keep.add(row["subject"])
    if not keep:
        raise SystemExit(f"no split=train rows in {split_csv}")

    inside = [r for r in records if r["subject"] in keep]
    dropped = len(records) - len(inside)
    tr, va, how = split_by_subject(inside, None, val_frac, seed, clot_id)
    return tr, va, (f"{how}, within the {len(keep)} TRAIN subjects of "
                    f"{Path(split_csv).name} ({dropped} frames from held-out "
                    f"test subjects excluded)")


def split_by_subject(records, holdout_centre=None, val_frac=0.2, seed=0,
                     clot_id=2):
    """Never split by frame: a subject contributes many near-duplicate frames.

    STRATIFIED by whether the subject has any clot annotation. An unstratified
    draw put 431 of 2091 val boxes (20.6%) in the clot class against roughly
    11.5% in train, and clot validation then swung between 0.215 and 0.463
    across epochs - noise large enough to drive checkpoint selection if the
    selection metric averages over classes.
    """
    if holdout_centre:
        tr = [r for r in records if r["centre"] != holdout_centre]
        va = [r for r in records if r["centre"] == holdout_centre]
        return tr, va, f"holdout centre {holdout_centre}"

    has_clot = defaultdict(bool)
    for r in records:
        if any(b[4] == clot_id for b in r["boxes"]):
            has_clot[r["subject"]] = True
    subs = sorted({r["subject"] for r in records})

    rng = random.Random(seed)
    val = set()
    for flag in (True, False):
        group = [s for s in subs if has_clot[s] is flag]
        rng.shuffle(group)
        k = int(round(len(group) * val_frac))
        if group and k == 0:
            k = 1
        val.update(group[:k])

    nclot = sum(1 for s in val if has_clot[s])
    return ([r for r in records if r["subject"] not in val],
            [r for r in records if r["subject"] in val],
            f"stratified subject split ({len(val)} of {len(subs)} subjects "
            f"held out, {nclot} with clot)")


#: Anchor aspect ratios, MEASURED on the 7136 expert boxes. torchvision
#: defines aspect_ratio as HEIGHT/WIDTH, and these vessels are consistently
#: wider than they are tall:
#:
#:     class    h/w p10   h/w median   h/w p90
#:     vein       0.27        0.64        0.94
#:     artery     0.52        0.81        1.08
#:     clot       0.44        0.67        0.99
#:
#: The torchvision default of (0.5, 1.0, 2.0) spends a third of its anchors on
#: tall boxes that never occur here, and has nothing below 0.5 for the wide
#: veins that make up the p10-p25 of the vein distribution.
ANCHOR_RATIOS = (0.3, 0.55, 1.0)

#: Anchor sizes per FPN level, at 800px input. Measured box side
#: (sqrt(w*h)) after the same resize: p5 38, median 92-112, p95 165-232 px.
#: The default (32, 64, 128, 256, 512) puts two of its five levels above
#: anything in the data.
ANCHOR_SIZES = ((24,), (48,), (96,), (160,), (256,))

#: The input size those sizes were measured at. Anchors are in INPUT PIXELS,
#: so they only mean what they were measured to mean at this resolution.
ANCHOR_REF_SIZE = 800


def scaled_anchor_sizes(img_size: int):
    """Anchor sizes rescaled to the actual input resolution.

    WHY THIS EXISTS
    ---------------
    `ANCHOR_SIZES` was measured on this cohort at 800px: vessel boxes had a
    median side of 92-112px and a p5 near 38. Those numbers are in input
    pixels and were hard-coded, so training at another resolution silently
    mismatched them - at 1024px every object is 1.28x larger while the anchor
    ladder stayed put.

    MEASURED consequence of that mismatch, 800px against 1024px on the same
    held-out split:

        vein TOP1     0.6038 -> 0.6737     artery TOP1  0.7701 -> 0.8036
        clot TOP1     0.6250 -> 0.4545     clot recall  0.6952 -> 0.4762

    Vessels improved because they sit mid-ladder with anchors either side.
    CLOT is the smallest class - it was already relying on the bottom rungs
    (24, 48), and shifting the ladder down relative to the objects left
    nothing well matched beneath it. A 22-point recall drop on the class with
    the highest expert ceiling (subject AUC 0.828) is not a resolution effect;
    it is an anchor effect.

    Aspect ratios are dimensionless and need no scaling.
    """
    scale = float(img_size) / ANCHOR_REF_SIZE
    return tuple((max(int(round(s[0] * scale)), 4),) for s in ANCHOR_SIZES)


def build_model(num_classes=None, pretrained=True, tune_anchors=True,
                trainable_layers=2, img_size=ANCHOR_REF_SIZE):
    """Faster R-CNN R50-FPNv2 with a fresh head.

    `trainable_layers` counts ResNet stages left unfrozen, from the top: 5 is
    everything, 3 is torchvision's pretrained default, 0 freezes the backbone
    entirely. Default 2 here, because the first run overfit immediately -
    validation TOP1 for the vessel class PEAKED AT EPOCH 0 (0.7564) and never
    recovered while training loss fell 0.301 -> 0.211. 2045 training frames
    come from 64 subjects and frames within a subject are near-duplicates, so
    the effective sample size is nearer 64 than 2045; fine-tuning three
    ResNet stages on that is far too much capacity.
    """
    import torchvision
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Read the module global HERE, not as a default argument. As a default it
    # is evaluated once at definition time, so main()'s `global NUM_CLASSES`
    # rebinding for the merged scheme never took effect and every merged model
    # was built with the SPLIT scheme's 4-wide head.
    if num_classes is None:
        num_classes = NUM_CLASSES

    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        trainable_backbone_layers=(trainable_layers if pretrained else 5))
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if tune_anchors:
        sizes = scaled_anchor_sizes(img_size)
        if sizes != ANCHOR_SIZES:
            print(f"  anchors rescaled for {img_size}px: "
                  f"{[s[0] for s in ANCHOR_SIZES]} -> {[s[0] for s in sizes]}")
        m.rpn.anchor_generator = AnchorGenerator(
            sizes=sizes,
            aspect_ratios=(ANCHOR_RATIOS,) * len(sizes))
        # the RPN head's conv is shared across levels and depends only on the
        # number of anchors per location, which is unchanged at 3
    return m


def box_iou(a, b):
    ix0 = np.maximum(a[0], b[:, 0]); iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2]); iy1 = np.minimum(a[3], b[:, 3])
    iw = np.clip(ix1 - ix0, 0, None); ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


@torch.no_grad()
def evaluate(model, loader, device, iou_thr=0.5):
    """Two metrics, because the obvious one is not the one that matters.

    RECALL - is there ANY prediction of the right class matching this box?
    That is what was reported before (vein 0.743, artery 0.716, clot 0.587).

    TOP1 - is the HIGHEST-SCORING prediction of that class the right one?
    This is what the pipeline actually consumes: `detector_seed` keeps "the
    strongest of each class" and hands that single box to the tracker, which
    then follows it for the entire clip. Recall counts a correct box that
    ranked third; the seeder never sees it. Measured against expert masks,
    41% of clips had at least one seed box that was not on a vessel at all
    while recall read 74% - that gap is this distinction.

    Optimise TOP1. Recall is kept for continuity with the old numbers.

    Reported per class rather than pooled: clot is 820 of 7154 boxes, and a
    model ignoring it entirely would still look respectable on an average.
    """
    model.eval()
    hit, top1, total, imgs_with = Counter(), Counter(), Counter(), Counter()
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        outs = model(imgs)
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
            # top-1, once per (image, class) present in the ground truth
            for cls in set(int(c) for c in gl):
                imgs_with[cls] += 1
                sel = pl == cls
                if not sel.any():
                    continue
                best = pb[sel][np.argmax(ps[sel])]
                gt = gb[gl == cls]
                if len(gt) and box_iou(best, gt).max() >= iou_thr:
                    top1[cls] += 1
    rec = {CLASS_NAMES[c]: round(hit[c] / total[c], 4)
           for c in sorted(total) if total[c]}
    t1 = {CLASS_NAMES[c]: round(top1[c] / imgs_with[c], 4)
          for c in sorted(imgs_with) if imgs_with[c]}
    return rec, t1, dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--out", default="checkpoints/vessel_detector.pth")
    ap.add_argument("--img-size", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="5e-4 (the previous default) is high for fine-tuning "
                         "a pretrained R50-FPN and matched the epoch-to-epoch "
                         "thrashing seen in the first run")
    ap.add_argument("--warmup-iters", type=int, default=250,
                    help="linear LR warmup, as in torchvision's detection "
                         "reference. Without it the fresh randomly-initialised "
                         "head sends large gradients into the backbone on the "
                         "first steps, which is when a small dataset is most "
                         "easily damaged.")
    ap.add_argument("--trainable-layers", type=int, default=2,
                    help="ResNet stages left unfrozen, from the top (5=all, "
                         "3=torchvision default, 0=frozen backbone). Default 2: "
                         "the first run's vessel TOP1 peaked at EPOCH 0 while "
                         "training loss kept falling, and 2045 frames from 64 "
                         "subjects cannot support three trainable stages.")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max gradient norm, 0 to disable")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--holdout-centre", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-stills", action="store_true")
    ap.add_argument("--split-csv", default=None,
                    help="frozen split from scripts/make_split.py. Subjects "
                         "marked split=test are EXCLUDED from training "
                         "entirely; the remaining train subjects are split "
                         "again for internal validation. Without this the "
                         "detector picks its own subjects and may train on "
                         "clips that later appear in the test set.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--select-on", default="vessel_top1",
                    choices=["vessel_top1", "top1", "recall"],
                    help="which validation metric picks the saved checkpoint. "
                         "TOP1 is what the seeder consumes - it keeps the "
                         "strongest box of each class, so a correct box ranked "
                         "third is worth nothing. Default vessel_top1 uses the "
                         "VESSEL class alone: macro TOP1 averages vessel and "
                         "clot, and clot swings 0.215-0.463 between epochs "
                         "while vessel sits at 0.68-0.76, so the macro tracks "
                         "clot noise. It saved epoch 2 (vessel 0.7459) over "
                         "epoch 4 (vessel 0.7549). Clot is separately "
                         "thresholded and bounded downstream; the seed is not.")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable flip and photometric jitter. NOTE: before "
                         "this revision augmentation was passed but never "
                         "applied, so the previous checkpoint was trained with "
                         "none at all.")
    ap.add_argument("--no-tune-anchors", action="store_true",
                    help="use torchvision's default anchors instead of the "
                         "ones measured from these boxes")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="drop polygons the tool INTERPOLATED (keyframe=false) "
                         "- 1051 of 7144. They are the interpolation "
                         "algorithm's guesses, not drawn by a human.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    RETIRED = {
        "--drop-clot": "now unconditional; clot is out of scope",
        "--no-merge-vessels": "split vein/artery is now the only scheme",
        "--clot-oversample": "clot is out of scope",
    }
    hit = [x for x in sys.argv[1:] if x.split("=")[0] in RETIRED]
    if hit:
        print("These flags no longer exist (their behaviour is now the default):\n")
        for x in hit:
            print(f"    {x.split('=')[0]:22s} {RETIRED[x.split('=')[0]]}")
        sys.exit(2)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("--device cuda but torch.cuda.is_available() is False")

    print("class scheme: vein + artery (clot out of scope)")

    print("indexing ...", flush=True)
    records = build_index(Path(a.annotations), Path(a.dicoms),
                          include_stills=not a.no_stills,
                          keyframes_only=a.keyframes_only)
    # Clot boxes are dropped here, before the split, exactly where the old
    # --drop-clot flag dropped them - so the records the splitter sees, and
    # therefore the split itself, are unchanged from the run that produced
    # checkpoints/vessel_detector_9926.pth.
    before = sum(len(r["boxes"]) for r in records)
    for r in records:
        r["boxes"] = [b for b in r["boxes"] if int(b[4]) != CLOT_ID]
    records = [r for r in records if r["boxes"]]
    after = sum(len(r["boxes"]) for r in records)
    print(f"  clot: {before - after} boxes removed, {after} vessel boxes "
          f"remain across {len(records)} frames")

    nbox = Counter(b[4] for r in records for b in r["boxes"])
    print(f"{len(records)} annotated frames, {sum(nbox.values())} boxes: "
          f"{ {CLASS_NAMES[k]: v for k, v in sorted(nbox.items())} }")

    if a.split_csv:
        tr, va, how = split_from_csv(records, a.split_csv, a.val_frac,
                                     clot_id=CLOT_ID)
    else:
        tr, va, how = split_by_subject(records, a.holdout_centre, a.val_frac,
                                       clot_id=CLOT_ID)
    print(f"split: {how} -> train {len(tr)}, val {len(va)}")
    vbox = Counter(b[4] for r in va for b in r["boxes"])
    print(f"  val boxes: { {CLASS_NAMES[k]: v for k, v in sorted(vbox.items())} }")
    if not va:
        sys.exit("empty validation split")

    ds_tr = DetectionDataset(tr, a.img_size, augment=not a.no_augment)
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=a.batch_size, shuffle=True,
        num_workers=a.workers, collate_fn=collate,
        pin_memory=a.device.startswith("cuda"))
    dl_va = torch.utils.data.DataLoader(
        DetectionDataset(va, a.img_size), batch_size=a.batch_size,
        num_workers=a.workers, collate_fn=collate,
        pin_memory=a.device.startswith("cuda"))

    model = build_model(num_classes=NUM_CLASSES,
                        tune_anchors=not a.no_tune_anchors,
                        img_size=a.img_size,
                        trainable_layers=a.trainable_layers).to(a.device)
    print(f"detection head: {NUM_CLASSES} outputs "
          f"(background + {sorted(CLASS_NAMES.values())})")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    print(f"trainable parameters: {sum(p.numel() for p in params)/1e6:.1f}M "
          f"({a.trainable_layers} ResNet stages unfrozen)")

    steps_per_epoch = max(1, (len(tr) + a.batch_size - 1) // a.batch_size)
    total_steps = steps_per_epoch * a.epochs
    warmup = min(a.warmup_iters, max(total_steps // 10, 1))

    def lr_at(step):
        """Linear warmup, then cosine decay to 1% of the peak."""
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"schedule: {warmup} warmup steps then cosine over {total_steps}")

    best, bad, history = -1.0, 0, []
    for epoch in range(a.epochs):
        model.train()
        tot = n = 0
        t0 = time.time()
        for imgs, targets in dl_tr:
            imgs = [i.to(a.device) for i in imgs]
            targets = [{k: v.to(a.device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
            opt.step()
            sched.step()
            tot += loss.detach().item() * len(imgs)
            n += len(imgs)
            if n % 200 < a.batch_size:
                print(f"\r  epoch {epoch:02d} {n}/{len(tr)} loss {tot/max(n,1):.4f}",
                      end="", flush=True)
        rates, t1, counts = evaluate(model, dl_va, a.device)
        macro_rec = float(np.mean(list(rates.values()))) if rates else 0.0
        macro_t1 = float(np.mean(list(t1.values()))) if t1 else 0.0
        vessel_t1 = t1.get("vessel", t1.get("vein", 0.0))
        macro = {"vessel_top1": vessel_t1,
                 "top1": macro_t1,
                 "recall": macro_rec}[a.select_on]
        print(f"\repoch {epoch:02d}  loss {tot/max(n,1):.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  "
              f"TOP1 {macro_t1:.4f} {t1}  |  recall {macro_rec:.4f} {rates}"
              f"  [select {a.select_on}={macro:.4f}]"
              f"  ({time.time()-t0:.0f}s)")
        history.append({"epoch": epoch, "loss": tot / max(n, 1),
                        "lr": opt.param_groups[0]["lr"],
                        "vessel_top1": vessel_t1,
                        "macro_top1": macro_t1, "top1_per_class": t1,
                        "macro_detect": macro_rec, "per_class": rates,
                        "selected_on": macro})

        if macro > best:
            best, bad = macro, 0
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "img_size": a.img_size,
                        "class_names": CLASS_NAMES,
                        # the anchors travel with the weights: a checkpoint
                        # trained with tuned anchors will not load into a model
                        # built with the defaults
                        # the ACTUAL sizes used, already scaled - load_detector
                        # rebuilds from these, so a checkpoint carries its own
                        # anchor ladder and cannot be mismatched on load
                        "anchors": (None if a.no_tune_anchors else
                                    {"sizes": [list(x) for x in
                                               scaled_anchor_sizes(a.img_size)],
                                     "ratios": list(ANCHOR_RATIOS)}),
                        "merged_vessels": False,
                        "select_on": a.select_on}, a.out)
            print(f"  saved -> {a.out}")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop: no improvement in {a.patience} epochs")
                break

    Path(a.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest macro detection rate {best:.4f}")
    print("Benchmark: the heuristic seeder achieved IoU>0.5 on 6% of clips.")


if __name__ == "__main__":
    main()
