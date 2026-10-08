r"""
Tests for pair-based vessel seeding.

Every constant here came from 2516 expert frames carrying both a vein and an
artery, and the bounds are deliberately set OUTSIDE the observed p1/p99 so a
real pair is never rejected:

    centroid separation / mean box side   p1 0.62   median 1.13   p99 4.16
    box IoU between the two               median 0  p99 0.136     max 0.29
    size ratio larger/smaller             median 1.41             p99 3.83
    vessel side / max(W, H)               p1 0.048  median 0.128  p99 0.277

The design this replaces kept "the strongest of each class" independently, so
a confident box on a shadow could become the vein while the real vein sat one
rank below. Measured consequence: on 41% of clips at least one seed box was not
on a vessel at all.

    python scripts/test_pair_seeding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt.step13_detector_seed import (  # noqa: E402
    CLASS_NAMES_MERGED, CLASS_NAMES_SPLIT, Detection, MAX_PAIR_BOX_IOU,
    MAX_REL_SIZE, MAX_SEPARATION, MAX_SIZE_RATIO, MIN_REL_SIZE,
    MIN_SEPARATION, pair_plausibility, plausible_vessel, select_clot,
    select_vessel_pair,
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


SHAPE = (600, 400)          # h, w -> max side 600


def det(x, y, w, h, score=0.9, role="vessel"):
    d = Detection(box=[float(x), float(y), float(x + w), float(y + h)],
                  cls=1, score=score)
    d.role = role
    return d


print("bounds sit outside every observed TRUE pair")
# These are the per-quantity extremes over 1932 expert vein+artery frames.
# An earlier cut rejected 7.9% of true pairs because MAX_REL_SIZE was taken
# from the geometric mean of the pair rather than from individual boxes.
check(MIN_SEPARATION < 0.23, "min separation is below the observed min 0.23")
check(MAX_SEPARATION > 3.63, "max separation is above the observed max 3.63")
check(MAX_PAIR_BOX_IOU > 0.447, "box-IoU bound is above the observed max 0.447")
check(MAX_SIZE_RATIO > 6.18, "size-ratio bound is above the observed max 6.18")
check(MAX_REL_SIZE > 0.698,
      "relative-size bound is above the observed max 0.698 - a single vein can "
      "fill most of the frame, which the pair's geometric mean hid")
check(MIN_REL_SIZE < 0.020, "and below the observed min 0.020")

print("\n  the observed extremes all pass")
big = det(0, 0, 300, 450)              # side 367 / 600 = 0.61
check(plausible_vessel(big, SHAPE), "a vein filling 61% of the frame is kept")
wide = det(50, 100, 60, 60), det(250, 110, 62, 62)   # sep ~3.3
check(pair_plausibility(*wide) > 0, "a pair at the observed max separation is kept")

print("\nplausible_vessel")
check(plausible_vessel(det(100, 100, 77, 77), SHAPE),
      "a vessel at the median relative size (0.128) is accepted")
check(not plausible_vessel(det(0, 0, 5, 5), SHAPE),
      "a 5px speck is rejected")
check(not plausible_vessel(det(0, 0, 400, 600), SHAPE),
      "a box covering the ENTIRE frame is rejected. The bound is deliberately "
      "loose - real veins reach 0.698 - so this only catches the degenerate "
      "whole-image blob, not the near-field band the heuristic seeder picked")

print("\npair_plausibility")
a = det(100, 200, 80, 80)
b = det(190, 210, 70, 70)                 # adjacent, sep ~1.2 box-sides
check(pair_plausibility(a, b) > 0, "an adjacent, similar-sized pair is legal")

far = det(20, 20, 40, 40), det(360, 560, 38, 38)      # ~11 box-widths
check(pair_plausibility(*far) == 0.0,
      "two vessels 11 box-widths apart are not a vein/artery pair (the "
      "observed maximum is 3.63)")

dup = det(100, 200, 80, 80), det(104, 203, 78, 78)
check(pair_plausibility(*dup) == 0.0,
      "two nearly identical boxes are a duplicate detection, not a pair")

lopsided = det(100, 200, 15, 15), det(140, 200, 160, 160)   # ~10.7x
check(pair_plausibility(*lopsided) == 0.0,
      "a 10x size difference exceeds anything observed (max 6.18)")

print("\n  the score prefers ordinary anatomy")
near_median = pair_plausibility(det(100, 200, 80, 80), det(190, 205, 78, 78))
near_edge = pair_plausibility(det(100, 200, 80, 80), det(400, 205, 78, 78))
check(near_median > near_edge,
      "a pair near the median separation scores above one at the extreme")

print("\nselect_vessel_pair")
# the failure the old rule could not see: a confident box on nothing, plus the
# real pair one rank below
shadow = det(10, 10, 60, 60, score=0.97)
vein_box = det(100, 300, 90, 70, score=0.80)       # deeper
art_box = det(200, 200, 70, 70, score=0.78)        # shallower
v, ar, note = select_vessel_pair([shadow, vein_box, art_box], SHAPE, "CFV")
check(note == "pair", "a pair is found")
check(v is not None and ar is not None, "both roles assigned")
check(v.box == vein_box.box,
      "the DEEPER box is the vein at CFV, even though the shadow scored 0.97 "
      "and would have won the old strongest-of-class rule")
check(ar.box == art_box.box, "the shallower box is the artery")
check(shadow.box not in (v.box, ar.box), "the shadow is not selected at all")

print("\n  the site prior flips the roles, not the pair")
v2, a2, _ = select_vessel_pair([vein_box, art_box], SHAPE, "PV")
check(v2.box == art_box.box,
      "at PV the SUPERFICIAL box becomes the vein - same pair, opposite roles")

print("\n  degenerate cases")
v3, a3, n3 = select_vessel_pair([vein_box], SHAPE, "CFV")
check(n3 == "single" and v3 is not None and a3 is None,
      "one vessel: seeded as the vein, no artery track")
check(v3.role == "vein", "the single vessel is labelled vein, the protocol's "
                         "target")
_v, _a, n4 = select_vessel_pair([], SHAPE, "CFV")
check(n4 == "none", "no detections: reported, not invented")
_v, _a, n5 = select_vessel_pair([det(0, 0, 4, 4)], SHAPE, "CFV")
check(n5 == "none", "a box too small to be a vessel does not become one")

print("\n  no prior for the site")
v6, a6, n6 = select_vessel_pair([vein_box, art_box], SHAPE, None)
check(n6 == "pair", "a pair is still formed without a site")
check(v6.side >= a6.side,
      "without a prior the larger box is taken as the vein - correct on "
      "relaxed frames, which is where seeding happens")

print("\nselect_clot")
vein = det(100, 300, 90, 70, role="vein")
on_vein = det(110, 310, 60, 50, score=0.6, role="clot")
elsewhere = det(300, 50, 60, 50, score=0.95, role="clot")
c = select_clot([elsewhere, on_vein], vein, SHAPE, min_score=0.3)
check(c is not None and c.box == on_vein.box,
      "the clot ON the vein wins over a higher-scoring one elsewhere")
check(select_clot([elsewhere], vein, SHAPE, 0.3) is None,
      "a clot nowhere near the vein is rejected - this is what produced "
      "clot_fraction values up to 159")
check(select_clot([on_vein], None, SHAPE, 0.3) is None,
      "no vein means no anchor, so no clot proposal")
check(select_clot([on_vein], vein, SHAPE, 0.9) is None,
      "the score floor is honoured")

print("\nclass schemes")
check(CLASS_NAMES_MERGED == {1: "vessel", 2: "clot"}, "merged scheme")
check("vein" not in CLASS_NAMES_MERGED.values(),
      "the merged checkpoint has NO vein class - callers must read .role")
check(CLASS_NAMES_SPLIT == {1: "vein", 2: "artery", 3: "clot"},
      "the legacy scheme still exists so old checkpoints load")

print("\nthe veto is loose ON PURPOSE")
# Tested against a real run's seeds, these constraints did NOT separate good
# pairs from bad: 87.3% of on-vessel pairs judged legal against 88.2% of
# off-vessel ones. They are calibrated to cost nothing, and the value of the
# pair formulation is in RANKING and in geometric role assignment, not in
# vetoing. This test exists so nobody tightens them expecting discrimination.
check(pair_plausibility(det(100, 200, 80, 80), det(300, 260, 75, 75)) > 0,
      "a merely unusual pair is still legal - tightening this to reject bad "
      "seeds was measured and does not work")

# ------------------------------------------- appearance vs geometry gate
print("\nmargin gate (split checkpoint only)")


def labelled(box, cls, score):
    d = Detection(box=list(box), cls=cls, score=score)
    d.role = {1: "vein", 2: "artery"}[cls]
    return d


# CFV: the prior says the DEEPER box is the vein. Put the detector's "vein"
# on the SHALLOW box, so the two sources disagree.
deep = [100.0, 300.0, 190.0, 380.0]
shallow = [210.0, 292.0, 296.0, 372.0]   # nearly the same depth -> tiny margin

near = [labelled(shallow, 1, 0.9), labelled(deep, 2, 0.85)]
v, a, note = select_vessel_pair(near, SHAPE, "CFV")
check(note == "pair", "a pair is formed")
check(v.box == shallow,
      "with an AMBIGUOUS depth margin the detector's call is kept - CFV has a "
      "median margin of 0.25, so 80.7% of its frames land here")

far_deep = [100.0, 380.0, 190.0, 470.0]
far_shal = [210.0, 150.0, 296.0, 230.0]
far = [labelled(far_shal, 1, 0.9), labelled(far_deep, 2, 0.85)]
v2, a2, _ = select_vessel_pair(far, SHAPE, "CFV")
check(v2.box == far_deep,
      "with a CLEAR depth margin geometry overrides the detector - the rule "
      "is 0.92-0.97 accurate above a margin of 0.5")

print("\n  the gate is inert without a detector opinion")
plain = [det(*[shallow[0], shallow[1], shallow[2]-shallow[0], shallow[3]-shallow[1]]),
         det(*[deep[0], deep[1], deep[2]-deep[0], deep[3]-deep[1]])]
v3, a3, _ = select_vessel_pair(plain, SHAPE, "CFV")
check(v3 is not None and a3 is not None,
      "a MERGED checkpoint (role 'vessel') still produces a pair")
check(v3.box == deep,
      "and geometry decides alone, since there is no appearance call to keep")

print("\n  min_margin is tunable and 0 restores geometry-only")
v4, a4, _ = select_vessel_pair(near, SHAPE, "CFV", min_margin=0.0)
check(v4.box == deep,
      "min_margin=0 reproduces the v4 behaviour that cost 0.13 subject AUC")

# --------------------------------------------- multi-vein sites (GS)
print("\nGS is a MULTI-VEIN field, not a pair")
from dvt.step10_auto_seed import (  # noqa: E402
    MULTI_VEIN_RULE, MULTI_VEIN_SITES, SITE_PRIOR,
)

check(SITE_PRIOR["GS"] == ("depth", "superficial"),
      "the GS prior is superficial, not larger. Measured over 415 GS frames: "
      "the GSV is the SHALLOWEST vein on 99.3% and the LARGEST on 11.6% - the "
      "old rule was backwards")
check("GS" in MULTI_VEIN_SITES,
      "GS is flagged multi-vein: the expert drew >1 vein on 88.3% of its frames")
check(MULTI_VEIN_RULE["GS"][0] == "shallowest",
      "shallowest leads, size breaks ties (88.2% on its own)")


def tagged(x, y, w, h, score, tag):
    d = Detection([float(x), float(y), float(x + w), float(y + h)], 1, score)
    d.role = "vessel"
    d.tag = tag
    return d


# the saphenofemoral junction: small shallow GSV, artery, deep large CFV
gsv = tagged(150, 100, 45, 45, 0.80, "GSV")
cfa = tagged(260, 215, 80, 80, 0.88, "CFA")
cfv = tagged(120, 240, 110, 100, 0.95, "CFV")

v, a, note = select_vessel_pair([cfv, cfa, gsv], SHAPE, "GS")
check(note == "pair", "a pair is returned")
check(v.tag == "GSV",
      "the SMALL SHALLOW vein is the target, even though the CFV scores higher "
      "(0.95 vs 0.80) and makes a more ordinary pair - that preference is what "
      "gave 8.4% site-named-vein accuracy at GS")
check(a is not None and a.tag == "CFA",
      "the companion is the ARTERY, not the deep vein. Among non-GSV vessels "
      "the artery is shallower than the deep vein on 89.2% of 389 frames")

print("\n  other sites are untouched")
v2, a2, _ = select_vessel_pair([cfv, cfa], SHAPE, "CFV")
check(v2.tag == "CFV" and a2.tag == "CFA",
      "CFV still takes the deeper box as the vein")
# separated well past the 0.5 margin gate, or the detector's own call stands
deep = tagged(120, 330, 110, 100, 0.95, "deep")
shal = tagged(260, 120, 80, 80, 0.88, "shallow")
v3, a3, _ = select_vessel_pair([deep, shal], SHAPE, "PV")
check(v3.tag == "shallow",
      "PV takes the SHALLOWER box as the vein - the sign genuinely flips "
      "between sites, and this only applies above the 0.5 depth-margin gate")
v3b, a3b, _ = select_vessel_pair([deep, shal], SHAPE, "CFV")
check(v3b.tag == "deep",
      "and CFV takes the deeper one, on the identical geometry")

print("\n  degenerate multi-vein cases")
v4, a4, n4 = select_vessel_pair([gsv, cfa], SHAPE, "GS")
check(n4 == "pair" and v4.tag == "GSV",
      "two vessels at GS: the shallower is still the vein")
v5, a5, n5 = select_vessel_pair([gsv], SHAPE, "GS")
check(n5 == "single" and v5 is not None,
      "one vessel at GS falls back to the single-vessel path")

print("\ntargeting the DEEP vein at GS instead of the saphenous")
from dvt.step10_auto_seed import (  # noqa: E402
    DEEP_VEIN_LABEL, MULTI_VEIN_TARGET_DEFAULT,
)

v6, a6, _ = select_vessel_pair([cfv, cfa, gsv], SHAPE, "GS",
                               multi_vein_target="deep")
check(v6.tag == "CFV",
      "target='deep' picks the common femoral vein. The DEEPEST vein is a "
      "deep-system vein on 99.3% of 418 GS frames (415 CFV against 3 GSV) - "
      "the most reliable rule measured anywhere in this pipeline")
check(a6 is not None and a6.tag == "CFA", "and still pairs it with the artery")

v7, a7, _ = select_vessel_pair([cfv, cfa, gsv], SHAPE, "GS",
                               multi_vein_target="gsv")
check(v7.tag == "GSV", "target='gsv' is unchanged")
check(MULTI_VEIN_TARGET_DEFAULT == "gsv",
      "and remains the default - switching target is a clinical choice")

check(DEEP_VEIN_LABEL["GS"][0] == "Common Femoral Vein",
      "with target='deep' the LABEL follows the vessel. v8 tracked the common "
      "femoral vein in GS clips and called it 'Great Saphenous Vein', because "
      "the label came from the filename - correct masks, wrong name")

print("\n  other sites ignore the target entirely")
for tgt in ("gsv", "deep"):
    v8_, a8_, _ = select_vessel_pair([cfv, cfa], SHAPE, "CFV",
                                     multi_vein_target=tgt)
    check(v8_.tag == "CFV",
          f"CFV unaffected by target={tgt!r} - only MULTI_VEIN_SITES route "
          "through the multi-vein path")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
