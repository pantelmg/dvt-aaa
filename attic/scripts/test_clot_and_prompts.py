r"""
Tests for negative point prompting and the clot plausibility bound.

Both exist because of measurements, and the tests restate those measurements
so a future change has to argue with the data rather than with an opinion:

  * 82% of mask collisions were UNIONS (one track outlining both vessels), not
    slips onto the wrong vessel. Three post-hoc repairs were simulated against
    expert masks; two were harmful and one was neutral. Separation therefore
    has to happen at PROMPT time.
  * Clot geometry over 339 expert co-annotated frames: area ratio median 0.80,
    max 1.68; 0.98 of clot area inside the vein at the median. The v3 run
    produced a median of 3.43 and a max of 159.

    python scripts/test_clot_and_prompts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt.step14_medsam2_track import (  # noqa: E402
    EXCLUSIVE_ROLES, constrain_clot, negatives_for, role_of,
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


def disc(h, w, cx, cy, r):
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r * r)


# ---------------------------------------------------------------- negatives
print("negatives_for")
boxes = {1: [10, 10, 50, 50], 2: [60, 10, 100, 50], 3: [20, 20, 40, 40]}
roles = {1: "vein", 2: "artery", 3: "clot"}

check(negatives_for(1, boxes, roles) == [boxes[2]],
      "the vein is warned off the artery")
check(negatives_for(2, boxes, roles) == [boxes[1]],
      "the artery is warned off the vein")
check(negatives_for(3, boxes, roles) == [],
      "clot gets NO negatives - it must be free to sit inside the vein")
check(all(boxes[3] not in negatives_for(o, boxes, roles) for o in (1, 2)),
      "nothing is warned off the clot either, or the vein would get a hole "
      "punched in it on every thrombosed clip")
check("clot" not in EXCLUSIVE_ROLES, "clot is not an exclusive role")

two_veins = {1: [0, 0, 10, 10], 4: [50, 50, 60, 60]}
check(negatives_for(1, two_veins, {1: "vein", 4: "vein"}) == [],
      "two veins do not warn each other off (a duplicate site, not a rival)")
check(negatives_for(1, {1: [0, 0, 1, 1]}, {1: "vein"}) == [],
      "a lone vessel has nothing to be warned off")

# ---------------------------------------------------------------- clot bound
print("\nconstrain_clot")
H = W = 120
vein = np.stack([disc(H, W, 60, 60, 30) for _ in range(5)])       # ~2827 px
v_area = np.array([int(m.sum()) for m in vein])

# frame 0 plausible (inside, smaller), 1 runaway, 2 outside, 3 empty, 4 ok
clot = np.stack([
    disc(H, W, 60, 60, 20),        # inside, ratio ~0.44   -> keep
    disc(H, W, 60, 60, 55),        # ratio ~3.4            -> reject
    disc(H, W, 15, 15, 12),        # outside the vein      -> reject
    np.zeros((H, W), bool),        # nothing               -> untouched
    disc(H, W, 65, 60, 25),        # mostly inside         -> keep
])
c_area = np.array([int(m.sum()) for m in clot])
q = np.zeros(5)

per_seed = [("Femoral Vein # 3", v_area.copy(), vein.copy(), q),
            ("Clot", c_area.copy(), clot.copy(), q)]
per_seed, n = constrain_clot(per_seed)
ca = per_seed[1][1]

check(ca[0] > 0, "a clot smaller than the vein and inside it survives")
check(ca[1] == 0, f"a clot {c_area[1]/v_area[1]:.1f}x the lumen is rejected")
check(ca[2] == 0, "a clot outside the vein is rejected")
check(ca[3] == 0, "an empty frame stays empty")
check(ca[4] > 0, "a mostly-inside clot survives")
check(n == 2, f"exactly two frames rejected (got {n})")
check(not per_seed[1][2][1].any(), "the rejected MASK is blanked, not just area")
check(per_seed[0][1].tolist() == v_area.tolist(),
      "the vein track is never touched")

print("\n  calibration against the TRACKED vein, not the expert one")
from dvt.step14_medsam2_track import constrain_clot as _cc
import inspect
sig = inspect.signature(_cc)
mr = sig.parameters["max_ratio"].default
mc = sig.parameters["min_containment"].default
check(abs(mr - 2.5) < 1e-9,
      f"max_ratio is {mr}: against the TRACKED vein 2.0 rejects 3.9% of real "
      "expert clot frames and 2.5 rejects 2.7%, because the tracked vein is "
      "sometimes much smaller than the truth")
check(abs(mc - 0.50) < 1e-9,
      f"min_containment is {mc}: tracked containment is bimodal (~0.99 or "
      "~0.00), so rejection is flat from 0.20 to 0.50 and 0.50 separates the "
      "modes cleanly")
check(3.43 > mr, "the v3 run's median of 3.43 is still caught")
check(1.68 < mr, "and the expert maximum of 1.68 is still kept")

print("\n  degenerate inputs")
only_vein = [("Common Femoral Vein # 1", v_area.copy(), vein.copy(), q)]
_ps, n2 = constrain_clot(only_vein)
check(n2 == 0, "no clot track: nothing to do")
only_clot = [("Clot", c_area.copy(), clot.copy(), q)]
_ps, n3 = constrain_clot(only_clot)
check(n3 == 0, "no vein to measure against: clot is left alone rather than "
               "blanked on a guess")

print("\n  the bound is a bound, not a filter on presence")
tiny = np.stack([disc(H, W, 60, 60, 4) for _ in range(5)])
ps = [("Femoral Vein # 3", v_area.copy(), vein.copy(), q),
      ("Clot", np.array([int(m.sum()) for m in tiny]), tiny.copy(), q)]
_ps, n4 = constrain_clot(ps)
check(n4 == 0, "a very small clot is NOT rejected - only impossible ones are")

print("\nrole_of on the labels the writer emits")
check(role_of("Femoral Vein # 3") == "vein", "vein label")
check(role_of("Clot") == "clot", "clot label")

# ---------------------------------------------------------------- shape
from dvt.step14_medsam2_track import (  # noqa: E402
    MIN_ARTERY_SOLIDITY, constrain_shape, implausible_shape, lobe_count,
    solidity,
)

print("\nsolidity")
H = W = 160
circle = disc(H, W, 80, 80, 40)
check(solidity(circle) > 0.94, f"a disc is convex ({solidity(circle):.3f})")
ell = np.zeros((H, W), bool)
yy, xx = np.ogrid[:H, :W]
ell[((xx - 80) / 55.0) ** 2 + ((yy - 80) / 25.0) ** 2 <= 1] = True
check(solidity(ell) > 0.94, f"an ellipse is convex ({solidity(ell):.3f})")

# two vessels joined at a NARROW waist, as in a real union
fig8 = disc(H, W, 46, 80, 30) | disc(H, W, 114, 80, 30) | (
    (np.abs(xx - 80) < 34) & (np.abs(yy - 80) < 7))
check(solidity(fig8) < 0.90,
      f"a figure-8 across two vessels is not ({solidity(fig8):.3f})")
check(lobe_count(fig8) == 1,
      "the LOBE test does NOT catch it - a 14px waist survives erosion at "
      "12% of the blob radius, and widening the erosion would cost the 0% "
      "false-positive rate. Solidity is the workhorse; lobes caught only 1.5% "
      "of real collisions against solidity's 13.4%")

# ...but a MILD union is not caught, and that is a real limitation
mild = disc(H, W, 52, 80, 30) | disc(H, W, 108, 80, 30)
check(solidity(mild) > 0.90 and lobe_count(mild) == 1,
      f"two heavily OVERLAPPING vessels ({solidity(mild):.3f}) slip through - "
      "the waist is too thick. Shape catches only 13.4% of real collisions; "
      "step15 occlusion is what addresses the rest")
check(lobe_count(circle) == 1, "a disc erodes to one")
check(lobe_count(ell) == 1, "an ellipse erodes to one")

print("\n  a COMPRESSED vein must survive")
slit = np.zeros((H, W), bool)
slit[((xx - 80) / 55.0) ** 2 + ((yy - 80) / 4.0) ** 2 <= 1] = True
crescent = disc(H, W, 80, 80, 40) & ~disc(H, W, 80, 52, 40)
for name, m in (("slit", slit), ("crescent", crescent)):
    check(implausible_shape(m, "vein") is None,
          f"a {name} vein is accepted (solidity {solidity(m):.3f}) - expert "
          "veins reach p1 = 0.382 and a bound would reject the very frames "
          "the ratio needs")
check(lobe_count(slit) == 1, "a slit is still one lobe")

print("\n  the artery bound sits below every expert artery")
check(MIN_ARTERY_SOLIDITY < 0.944,
      "0.90 is below the expert artery p0.5 of 0.944")
check(implausible_shape(circle, "artery") is None, "a round artery passes")
check(implausible_shape(fig8, "artery") is not None,
      f"a narrow-waisted figure-8 artery is caught "
      f"({implausible_shape(fig8, 'artery')})")
check(implausible_shape(fig8, "vein") is None,
      "a figure-8 VEIN is NOT caught, and cannot be. Expert veins reach "
      "solidity p1 = 0.382 because a compressed vein is a slit, so no vein "
      "bound exists that would not also reject real compressed veins. The "
      "shape penalty is one-directional: it detects a union labelled ARTERY, "
      "never one labelled VEIN")

print("\nconstrain_shape")
q5 = np.zeros(4)
masks = np.stack([circle, fig8, ell, np.zeros((H, W), bool)])
areas = np.array([int(m.sum()) for m in masks])
ps = [("Common Femoral Artery # 1", areas.copy(), masks.copy(), q5)]
ps, n, why = constrain_shape(ps)
check(n == 1, f"exactly one frame rejected (got {n})")
check(ps[0][1][1] == 0 and not ps[0][2][1].any(),
      "the figure-8 frame is blanked, mask and area together")
check(ps[0][1][0] > 0 and ps[0][1][2] > 0, "the good frames are untouched")
check(any("artery" in k for k in why), f"the reason is recorded: {why}")

vein_ps = [("Femoral Vein # 3",
            np.array([int(slit.sum())]), np.stack([slit]), np.zeros(1))]
_ps, nv, _w = constrain_shape(vein_ps)
check(nv == 0, "a compressed vein is never rejected on convexity")

clot_ps = [("Clot", np.array([int(fig8.sum())]), np.stack([fig8]), np.zeros(1))]
_ps, nc, _w = constrain_shape(clot_ps)
check(nc == 0, "clot shape is not constrained here - it is bounded by area "
               "and containment against the vein instead")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
