r"""
Tests for the detector training changes.

torch is not importable in every environment these run in, so this exercises
the parts that are pure logic - augmentation geometry, the top-1 metric, the
anchor constants - against a minimal stub rather than importing the trainer.
The point is to pin the DECISIONS, each of which came from a measurement:

  * augmentation existed as a flag and was never applied
  * the empty-box fallback labelled a corner pixel as VEIN
  * anchors were tuned for natural images, not for wide vessels
  * the selection metric was recall, but the seeder consumes only the top-1 box

    python scripts/test_detector_training.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "scripts" / "train_detector.py").read_text()

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


# ------------------------------------------------------- constants by parsing
tree = ast.parse(SRC)
consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            consts[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            pass

print("anchor constants match the measured box distribution")
ratios = consts.get("ANCHOR_RATIOS")
sizes = consts.get("ANCHOR_SIZES")
check(ratios is not None and sizes is not None, "constants are defined")
# measured h/w: vein p10 0.27, median 0.64, p90 0.94; artery median 0.81
check(max(ratios) <= 1.0,
      f"no anchor is taller than wide {ratios} - h/w never exceeds ~1.08 here")
check(min(ratios) <= 0.35,
      "there is an anchor for the widest veins (h/w p10 = 0.27), which the "
      "torchvision default of 0.5 could not reach")
check(2.0 not in ratios,
      "the default 2.0 anchor is dropped - it matched nothing in 7136 boxes")
flat = [x for lvl in sizes for x in lvl]
check(max(flat) <= 300,
      f"largest anchor {max(flat)} is near the measured p95 of ~232px, not 512")
check(min(flat) <= 32,
      f"smallest anchor {min(flat)} reaches the measured p5 of ~38px")

# ------------------------------------------------------- augmentation applied
print("\naugmentation is actually applied")
check("self.augment and boxes" in SRC or "if self.augment" in SRC,
      "self.augment is READ in __getitem__, not merely stored")
check("_augment" in SRC, "an _augment method exists")
check("dims=[2]" in SRC, "the flip is horizontal (width axis)")
check("dims=[1]" not in SRC,
      "there is NO vertical flip - it would invert depth, the cue that "
      "separates vein from artery (96% at CFV, 95% at PV)")

print("\n  flip geometry")


def flip_boxes(boxes, nw):
    return [[nw - b[2], b[1], nw - b[0], b[3]] for b in boxes]


nw = 800
b = [[100.0, 50.0, 300.0, 250.0]]
f = flip_boxes(b, nw)
check(f[0][0] < f[0][2], "flipped box keeps x0 < x1")
check(abs((f[0][2] - f[0][0]) - (b[0][2] - b[0][0])) < 1e-9,
      "width is preserved")
check(f[0][1] == b[0][1] and f[0][3] == b[0][3],
      "y is untouched - depth must survive the flip")
check(flip_boxes(f, nw)[0] == b[0], "flipping twice is the identity")

# ------------------------------------------------------- no fake empty target
print("\nempty targets are empty")
check("[[0.0, 0.0, 1.0, 1.0]], [1]" not in SRC,
      "the [0,0,1,1]-labelled-VEIN fallback is gone - it taught the detector "
      "that the frame's black top-left corner is a vein")
check("torch.zeros((0, 4)" in SRC, "a properly shaped zero-box target is used")

# ------------------------------------------------------- top-1 metric
print("\ntop-1 metric, which is what the seeder consumes")
check("top1" in SRC, "top1 is computed")
check('"top1", "recall"' in SRC or "'top1', 'recall'" in SRC,
      "the selection metric is a choice, defaulting to top1")


def box_iou(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[0], b[:, 0]); iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2]); iy1 = np.minimum(a[3], b[:, 3])
    iw = np.clip(ix1 - ix0, 0, None); ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


gt = [100.0, 100.0, 200.0, 200.0]
good = [102.0, 98.0, 198.0, 204.0]
junk = [600.0, 600.0, 700.0, 700.0]

check(box_iou(good, gt).max() >= 0.5, "the good box matches the truth")
check(box_iou(junk, gt).max() < 0.5, "the junk box does not")

# the case the old metric scored as a success and the pipeline scored as a
# failure: correct box present, but ranked below a wrong one
preds = [(junk, 0.91), (good, 0.62)]
recall_hit = any(box_iou(b, gt).max() >= 0.5 for b, _ in preds)
top1_box = max(preds, key=lambda p: p[1])[0]
top1_hit = box_iou(top1_box, gt).max() >= 0.5
check(recall_hit, "RECALL counts this frame as a hit")
check(not top1_hit,
      "TOP1 counts it as a miss - and the seeder takes the top box, so this "
      "is the frame that lands a vessel track on nothing")
check(recall_hit != top1_hit,
      "the two metrics disagree exactly where the pipeline breaks")

# ---------------------------------------------------- augmentation geometry
print("\naugmentation keeps boxes on the anatomy")
src_aug = SRC[SRC.index("def _augment"):SRC.index("def collate")]

check("torch.roll" in src_aug, "translation is implemented")
check("pow(" in src_aug, "gamma jitter is implemented")
check("randn_like" in src_aug, "speckle noise is implemented")
check("x * (1.0 + torch.randn_like" in src_aug,
      "speckle is MULTIPLICATIVE - ultrasound noise is speckle, and additive "
      "Gaussian would train against the wrong corruption")
check("_clip_boxes" in src_aug,
      "boxes are re-clipped after translation, or a shifted box could hang "
      "outside the image and become a negative-area target")


def scale_boxes(boxes, nw, nh, sw, sh):
    return [[b[0] * sw / nw, b[1] * sh / nh, b[2] * sw / nw, b[3] * sh / nh]
            for b in boxes]


nw = nh = 800
b0 = [[100.0, 200.0, 300.0, 400.0]]
sb = scale_boxes(b0, nw, nh, 1000, 1000)
check(abs(sb[0][0] - 125.0) < 1e-6 and abs(sb[0][2] - 375.0) < 1e-6,
      "scale jitter moves box coordinates by the same factor as the image")
back = scale_boxes(sb, 1000, 1000, nw, nh)
check(all(abs(x - y) < 1e-6 for x, y in zip(back[0], b0[0])),
      "scaling up then down is the identity")


def clip(boxes, nw, nh, min_side=4.0):
    out = []
    for b in boxes:
        b = [max(0.0, min(b[0], nw)), max(0.0, min(b[1], nh)),
             max(0.0, min(b[2], nw)), max(0.0, min(b[3], nh))]
        if b[2] - b[0] >= min_side and b[3] - b[1] >= min_side:
            out.append(b)
    return out


shifted = [[-50.0, 200.0, 150.0, 400.0]]
c = clip(shifted, 800, 800)
check(len(c) == 1 and c[0][0] == 0.0, "a partly shifted-out box is clipped, not dropped")
gone = [[-300.0, 200.0, -100.0, 400.0]]
check(clip(gone, 800, 800) == [],
      "a box shifted entirely off-image is DROPPED, not clamped to a sliver "
      "at the edge - a zero-width target at x=0 is exactly the corner-pixel "
      "false positive removed above")

# ------------------------------------------------- the default-argument trap
print("\nnum_classes is read at CALL time, not bound as a default")
check("def build_model(num_classes=None" in SRC,
      "build_model takes None, not NUM_CLASSES, as its default")
check("num_classes = NUM_CLASSES" in SRC,
      "the global is read inside the body, where main()'s rebinding is visible")
check("def build_model(num_classes=NUM_CLASSES" not in SRC,
      "the default-argument form is gone - it evaluated once at definition "
      "time, so every merged model was built with the SPLIT scheme's 4-wide "
      "head and would not load into a head sized from class_names")

G = 4


def as_default(n=G):
    return n


def at_call_time(n=None):
    return G if n is None else n


G = 3
check(as_default() == 4, "a default argument keeps the value from definition time")
check(at_call_time() == 3, "reading the global in the body sees the rebinding")

seed_src = (ROOT / "dvt" / "step13_detector_seed.py").read_text()
check("state[key].shape[0]" in seed_src,
      "load_detector sizes the head from the WEIGHTS, so checkpoints trained "
      "before the fix still load")

# ------------------------------------------------ anchors follow the input
print("\nanchor sizes scale with --img-size")
import ast as _ast

_tree = _ast.parse(SRC)
_fn = next(n for n in _tree.body
           if isinstance(n, _ast.FunctionDef) and n.name == "scaled_anchor_sizes")
_ns = {k: _ast.literal_eval(n.value) for n in _tree.body
       if isinstance(n, _ast.Assign) and isinstance(n.targets[0], _ast.Name)
       and (k := n.targets[0].id) in ("ANCHOR_SIZES", "ANCHOR_REF_SIZE")}
exec(compile(_ast.Module(body=[_fn], type_ignores=[]), "<f>", "exec"), _ns)
_scale = _ns["scaled_anchor_sizes"]

check(_ns["ANCHOR_REF_SIZE"] == 800,
      "the reference size is recorded - anchors are in INPUT PIXELS and only "
      "mean what they were measured to mean at that resolution")
check(_scale(800) == _ns["ANCHOR_SIZES"],
      "scaling is the identity at the reference size, so existing checkpoints "
      "are unaffected")
check([x[0] for x in _scale(1024)] == [31, 61, 123, 205, 328],
      "1024px scales the ladder by 1.28x")
check(_scale(512)[0][0] < _ns["ANCHOR_SIZES"][0][0],
      "and a smaller input scales it down")
check(all(x[0] >= 4 for x in _scale(128)),
      "no anchor collapses below 4px at extreme resolutions")

check("scaled_anchor_sizes(a.img_size)" in SRC,
      "the checkpoint stores the ACTUAL sizes used, so load_detector rebuilds "
      "the right ladder and cannot be mismatched on load")
check("0.6952 -> 0.4762" in SRC,
      "the measurement that motivated this is recorded: training at 1024 with "
      "800px anchors cost 22 points of CLOT recall while vessels improved - "
      "clot is the smallest class and was already on the bottom rungs")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
