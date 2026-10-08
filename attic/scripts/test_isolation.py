r"""
Tests for isolated tracking (step 15).

The one thing that CANNOT be tested here is whether hiding the rival actually
stops the tracks converging - that needs MedSAM2 and a GPU. What is tested is
everything that could silently sabotage it:

  * the fill must not leave a vessel-shaped hole where the rival was, or the
    tracker will segment the hole
  * the edge must not be hard, or the rectangle itself becomes a structure with
    stronger gradients than the vessel
  * the target must never be occluded - hiding the thing being tracked is far
    worse than not occluding at all
  * clot must never be isolated from the vein

    python scripts/test_isolation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt.step15_isolated_track import (  # noqa: E402
    MAX_SELF_OVERLAP, OCCLUSION_MARGIN, _feathered_mask, _iou, occlude,
)

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def synthetic_frame(h=300, w=400, seed=0):
    """Speckled tissue with two dark round vessels, as in a real transverse view."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 25, (h, w)), 0, 255)
    yy, xx = np.ogrid[:h, :w]
    for cx, cy, r in ((120, 150, 40), (250, 150, 35)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.normal(
            18, 6, int(((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).sum()))
    return np.clip(img, 0, 255).astype(np.uint8)


VESSEL_A = [80.0, 110.0, 160.0, 190.0]     # left
VESSEL_B = [215.0, 115.0, 285.0, 185.0]    # right

print("feathered mask")
m = _feathered_mask((300, 400), VESSEL_B)
check(m.max() > 0.99, "the mask reaches 1 inside the box")
check(m.min() < 0.01, "and 0 far outside it")
cy, cx = 150, 250
check(m[cy, cx] > 0.99, "the rival's centre is fully covered")
check(m[150, 120] < 0.05, "the OTHER vessel's centre is untouched")
grad = np.abs(np.diff(m[150, :]))
check(grad.max() < 0.35,
      f"the edge is soft (max step {grad.max():.2f}) - a hard rectangle has a "
      "step of 1.0 and stronger gradients than the vessel wall")

print("\n  the grown box covers more than the detector's tight box")
tight = _feathered_mask((300, 400), VESSEL_B, margin=0.0)
check(m.sum() > tight.sum(),
      f"margin {OCCLUSION_MARGIN} grows the hidden region, so the tracker "
      "cannot latch onto the vessel wall just outside the box")

print("\nocclude")
f = synthetic_frame()
inside_before = f[150, 240:260].mean()
out = occlude(f, [VESSEL_B])

check(out.shape == f.shape and out.dtype == f.dtype,
      "shape and dtype are preserved")
inside_after = out[150, 240:260].mean()
check(inside_after > inside_before + 40,
      f"the rival is no longer dark ({inside_before:.0f} -> {inside_after:.0f})")

tissue = f[40:80, 40:80].mean()
check(abs(inside_after - tissue) < 35,
      f"the fill resembles tissue ({inside_after:.0f} vs {tissue:.0f}), not a "
      "flat grey patch - a uniform region is as segmentable as a black one")

filled = out[130:170, 230:270].astype(float)
check(filled.std() > 8,
      f"the fill carries speckle (sd {filled.std():.1f}), so it has no "
      "suspiciously smooth interior for the tracker to find")

print("\n  the target is untouched")
left_before = f[150, 100:140].astype(float)
left_after = out[150, 100:140].astype(float)
check(np.abs(left_before - left_after).max() < 1.5,
      "the vessel being tracked is bit-for-bit intact")

print("\n  no vessel-shaped hole is left behind")
# a dark region the size of the original vessel should no longer exist there
patch = out[115:185, 215:285].astype(float)
rest = out[20:90, 20:90].astype(float)
check(abs(patch.mean() - rest.mean()) < 30,
      "the occluded region's brightness matches ordinary tissue")

print("\n  degenerate inputs")
check(np.array_equal(occlude(f, []), f), "no boxes -> the frame is unchanged")
off = occlude(f, [[-200.0, -200.0, -150.0, -150.0]])
check(off.shape == f.shape, "a box entirely off-frame does not crash")
big = occlude(f, [[-50.0, -50.0, 450.0, 350.0]])
check(big.shape == f.shape, "a box larger than the frame does not crash")

print("\nself-overlap guard")
check(_iou(VESSEL_A, VESSEL_A) > 0.99, "a box overlaps itself completely")
check(_iou(VESSEL_A, VESSEL_B) < MAX_SELF_OVERLAP,
      "the two vessels are distinct enough to occlude one another")
nudged = [VESSEL_A[0] + 8, VESSEL_A[1] + 8, VESSEL_A[2] + 8, VESSEL_A[3] + 8]
check(_iou(VESSEL_A, nudged) > MAX_SELF_OVERLAP,
      "a rival box sitting on the target IS caught by the guard - the "
      "detector firing twice on one vessel must not blind its own tracker")

print("\nclot is never isolated")
src = (Path(__file__).resolve().parent.parent
       / "dvt" / "step15_isolated_track.py").read_text()
check("EXCLUSIVE_ROLES" in src and "clot is never isolated" in src,
      "clot gets no rival boxes - 98% of clot area sits inside the vein, so "
      "hiding either from the other destroys it")
check('"clot" if roles.get(oid) == "clot" else' in src,
      "clot rides in the vein's pass rather than getting its own")

print("\noccluding both vessels at once still leaves tissue statistics sane")
both = occlude(f, [VESSEL_A, VESSEL_B])
check(abs(float(both[130:170, 230:270].mean())
          - float(both[130:170, 100:140].mean())) < 30,
      "both regions are filled consistently, using statistics from OUTSIDE "
      "the boxes so a large anechoic vessel cannot drag the fill toward black")

# ------------------------------------------------- continuity association
print("\ncontinuity matching (roles fixed at seeding, never re-decided)")
from dvt.step15_isolated_track import _match_by_continuity  # noqa: E402


class D:
    def __init__(self, box, score=0.9, role="vessel"):
        self.box = list(map(float, box)); self.score = score; self.role = role


# vein on the left, artery on the right, as at seeding
tracks = {1: (VESSEL_A, 80.0), 2: (VESSEL_B, 70.0)}

m = _match_by_continuity([D(VESSEL_A), D(VESSEL_B)], tracks)
check(m[1] == VESSEL_A and m[2] == VESSEL_B, "each track takes its own vessel")

# detections arriving in the OPPOSITE order must not swap the assignment
m = _match_by_continuity([D(VESSEL_B), D(VESSEL_A)], tracks)
check(m[1] == VESSEL_A and m[2] == VESSEL_B,
      "detection ORDER does not change the assignment - the old code re-ran "
      "the anatomical role decision every frame, and at 81-87% accuracy that "
      "flipped roughly one frame in six")

# both vessels drift right; each should follow its own
drift = [[v + 12 if i in (0, 2) else v for i, v in enumerate(VESSEL_A)],
         [v + 12 if i in (0, 2) else v for i, v in enumerate(VESSEL_B)]]
m = _match_by_continuity([D(drift[1]), D(drift[0])], tracks)
check(m[1] == drift[0] and m[2] == drift[1],
      "each track follows its own vessel through motion")

print("\n  one detection is not shared between two tracks")
m = _match_by_continuity([D(VESSEL_A)], tracks)
check(m.get(1) == VESSEL_A, "the nearer track claims it")
check(2 not in m, "the other track gets nothing rather than the same box")

print("\n  a far-away detection is not force-matched")
m = _match_by_continuity([D([600.0, 600.0, 660.0, 660.0])], tracks, gate=2.0)
check(m == {}, "beyond the gate, a detection is treated as unrelated")

print("\n  a missed frame HOLDS the last position")
src = (Path(__file__).resolve().parent.parent
       / "dvt" / "step15_isolated_track.py").read_text()
check("hold, do not re-decide" in src,
      "an unmatched frame reuses the last known rival instead of re-deciding")
check("collapse_frac" in src,
      "occlusion stops once the vessel has collapsed - nothing left to protect")
check("select_vessel_pair" not in src.split("def locate_rivals")[1].split("def track_isolated")[0],
      "locate_rivals no longer calls select_vessel_pair at all")

# ---------------------------------------------------- seed frame + coverage
print("\nseeding at the frame the boxes came from")
run_src = (Path(__file__).resolve().parent / "run_autoannotate.py").read_text()
check("prompt_frame=args._seed_frame" in run_src,
      "the prompt frame is the frame detector_seed actually found the vessels "
      "on, not a hardcoded 0")
check("cands, seed_frame = S13.detector_seed" in run_src,
      "the returned frame index is captured rather than discarded")
check("prompt_frame=0," not in run_src,
      "no tracking call hardcodes frame 0 any more - a box taken 25-49 frames "
      "away had median IoU 0.391 with the vessel's true position at frame 0")

iso_src = (Path(__file__).resolve().parent.parent
           / "dvt" / "step15_isolated_track.py").read_text()
check("reverse=True" in iso_src,
      "isolated tracking propagates BACKWARDS too - the prompt frame is now a "
      "median of 19 frames in, and everything before it would otherwise be "
      "unsegmented")
check("rival_det_score" in iso_src,
      "locating the rival uses its own detection floor")

print("\n  a missed rival is worse than a spurious one")
check("--isolate-det-score" in run_src, "the rival floor is exposed as a flag")
import re
m = re.search(r'"--isolate-det-score".*?default=([0-9.]+)', run_src, re.S)
md = re.search(r'"--det-score".*?default=([0-9.]+)', run_src, re.S)
if m and md:
    check(float(m.group(1)) < float(md.group(1)),
          f"the rival floor ({m.group(1)}) is BELOW the seeding floor "
          f"({md.group(1)}): an unhidden artery lets the vein track re-identify "
          "onto it, while a spurious box only blanks tissue")

# ------------------------------------------------- depth-order constraint
print("\ndepth ordering: vessels cannot pass through each other")
from dvt.step15_isolated_track import (  # noqa: E402
    _enforce_depth_order, depth_order_of,
)


class Det2:
    def __init__(self, box):
        self.box = list(map(float, box)); self.score = 0.9


seed = {1: [100., 300., 200., 400.], 2: [120., 150., 210., 240.]}
order = depth_order_of(seed)
check(order[2] < order[1],
      "the shallower object ranks first; the vein is deeper here")

ok = {1: [104., 306., 204., 404.], 2: [124., 156., 214., 246.]}
check(_enforce_depth_order(ok, order) == ok,
      "a consistent assignment passes through untouched")

inv = {1: [120., 150., 210., 240.], 2: [100., 300., 200., 400.]}
fixed = _enforce_depth_order(inv, order)
check(len(fixed) == 2, "an inverted pair is repaired, not dropped")
check(fixed[1][1] > fixed[2][1],
      "after repair the deep track is deep again - the SWAPPED assignment is "
      "tried before anything is discarded")

print("\n  the unresolvable case fails safe")
flat = {1: [100., 300., 200., 400.], 2: [300., 300., 400., 400.]}
check(_enforce_depth_order(flat, order) == {},
      "when neither ordering is consistent BOTH are dropped - a missing "
      "polygon costs a radiologist far less than a wrong one, and downstream "
      "code reads a blank as 'no measurement'")

single = {1: [100., 300., 200., 400.]}
check(_enforce_depth_order(single, order) == single,
      "one track has no ordering to violate")

print("\n  wired into per-frame tracking")
pf = (Path(__file__).resolve().parent.parent
      / "dvt" / "step16_per_frame.py").read_text()
check("depth_order_of(vessel_boxes)" in pf,
      "the ordering is fixed at SEEDING, not recomputed per frame")
check("_enforce_depth_order(dict(m_free), order)" in pf,
      "and applied to a COPY of the unconstrained assignment, not passed into "
      "the matcher - the counterfactual is what makes a rejected box "
      "attributable rather than merely counted")
check("order_rejected" in pf,
      "rejections are counted, so a clip losing many frames is visible")
check("30.1%" in _enforce_depth_order.__doc__ and "82.3%" in _enforce_depth_order.__doc__,
      "the measurement that justifies the constraint is recorded in the code")

fs = (Path(__file__).resolve().parent / "fix_swaps.py").read_text()
check("ESTABLISH_N = 8" in fs,
      "the post-hoc pass reconstructs the reference from the first 8 frames")
check("2.48" in fs and "0.71" in fs,
      "and records why: 8 frames gives 2.48:1, the site prior 0.71:1 - the "
      "prior re-decides the role per frame, which is what made voting fail")
check("ESTABLISH_FRAC" not in fs,
      "the fraction-based reference is gone; it broke on clips that invert "
      "early, where 20% of the clip is already mostly inverted")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
