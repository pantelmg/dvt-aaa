r"""
Tests for the vein/artery exclusion logic added to step14.

These run without torch, MedSAM2, or a GPU: every function under test operates
on numpy logits, which is why the exclusion was put there rather than inside
the propagation loop.

    python scripts/test_overlap_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt.step14_medsam2_track import (  # noqa: E402
    EXCLUSIVE_ROLES, containment, resolve_exclusive, role_of,
    worst_containment,
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


def blob(h, w, cx, cy, r, val=3.0, bg=-3.0):
    """A disc of positive logits on a negative background."""
    y, x = np.ogrid[:h, :w]
    m = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    return np.where(m, val, bg).astype(np.float32)


print("role_of")
check(role_of("Common Femoral Vein # 1") == "vein", "panel vein name")
check(role_of("Common Femoral Artery # 1") == "artery", "panel artery name")
check(role_of("Clot") == "clot", "clot")
check(role_of("CFV Scanning Location 1") == "vein",
      "expert vocabulary vein (would fall through a 'vein' substring match)")
check(role_of("CFA Scanning Location 1") == "artery",
      "expert vocabulary artery")
check(role_of("PA Scanning Location 4") == "artery", "popliteal artery code")
check(role_of("GSV Scanning Location 2") == "vein", "great saphenous code")
check(role_of("Other") == "other", "unmapped label is not a vessel")
check(role_of("") == "other", "empty label does not crash")
check("clot" not in EXCLUSIVE_ROLES,
      "clot is exempt from exclusion (it overlaps the vein by design)")

print("\ncontainment")
a = np.zeros((20, 20), bool); a[2:12, 2:12] = True      # 100 px
b = np.zeros((20, 20), bool); b[4:9, 4:9] = True        # 25 px, inside a
check(abs(containment(a, b) - 1.0) < 1e-9,
      "small mask fully inside a large one scores 1.0")
inter = (a & b).sum(); union = a.sum() + b.sum() - inter
check(inter / union < 0.3,
      f"...while IoU for the same pair is only {inter/union:.2f} - which is "
      "why containment is the flag")
check(containment(a, np.zeros((20, 20), bool)) == 0.0, "empty mask -> 0.0")
c = np.zeros((20, 20), bool); c[15:19, 15:19] = True
check(containment(a, c) == 0.0, "disjoint masks -> 0.0")

print("\nresolve_exclusive")
h = w = 40
vein = blob(h, w, 15, 20, 10, val=5.0)
art = blob(h, w, 22, 20, 8, val=2.0)          # overlaps the vein, weaker
stack = {1: vein, 2: art}
res = resolve_exclusive(stack, {1, 2})
check(not (res[1] & res[2]).any(), "exclusive ids share no pixel")
check(res[1].sum() > 0 and res[2].sum() > 0, "both objects survive")
raw1 = (vein > 0)
check(res[1].sum() == raw1.sum(),
      "the stronger logit keeps all of its own pixels")
check(res[2].sum() < (art > 0).sum(),
      "the weaker logit loses the contested region")

print("\n  clot is not excluded")
clot = blob(h, w, 15, 20, 5, val=1.0)         # inside the vein, weakest
res2 = resolve_exclusive({1: vein, 2: art, 3: clot}, {1, 2})
check((res2[3] & res2[1]).any(),
      "clot may overlap the vein even though its logit is lower")
check(res2[3].sum() == (clot > 0).sum(),
      "clot keeps every pixel it claimed")
check(not (res2[1] & res2[2]).any(), "vein/artery still disjoint alongside it")

print("\n  degenerate inputs")
check(resolve_exclusive({1: vein}, {1}).__len__() == 1, "single object")
check(resolve_exclusive({1: vein}, set())[1].sum() == (vein > 0).sum(),
      "no exclusive ids -> plain threshold")
allneg = np.full((h, w), -1.0, dtype=np.float32)
r3 = resolve_exclusive({1: allneg, 2: allneg}, {1, 2})
check(r3[1].sum() == 0 and r3[2].sum() == 0,
      "argmax does not invent a winner where every logit is negative")

print("\nworst_containment")
roles = {1: "vein", 2: "artery", 3: "clot"}
masks = {1: (vein > 0), 2: (art > 0), 3: (clot > 0)}
worst, pair = worst_containment(masks, roles)
check(worst > 0, "vein/artery overlap is reported")
check(set(pair) == {1, 2}, "the reported pair is vein/artery, not clot")

only_clot = {1: (vein > 0), 3: (clot > 0)}
w2, p2 = worst_containment(only_clot, {1: "vein", 3: "clot"})
check(w2 == 0.0 and p2 is None,
      "clot inside the vein is NOT a conflict (it would be a false alarm on "
      "every thrombosed clip)")

two_veins = {1: (vein > 0), 4: (vein > 0)}
w3, _ = worst_containment(two_veins, {1: "vein", 4: "vein"})
check(w3 == 0.0, "two veins are a duplicate site, not a merge")

print("\n  the guard sees what the area test could not")
# Reproduce the measured case: artery is 0.52x the vein, so a total merge
# grows vein area by 1.52x - under a 1.8 threshold.
v = np.zeros((60, 60), bool); v[10:40, 10:40] = True          # 900 px
extra = np.zeros((60, 60), bool); extra[10:31, 40:62][:, :22] = True
merged = v | extra[:60, :60]
growth = merged.sum() / v.sum()
check(growth < 1.8,
      f"a merge of this size grows area only {growth:.2f}x, below the 1.8 "
      "threshold the run used")
check(containment(merged, (extra[:60, :60])) > 0.9,
      "...but containment against the absorbed structure is near 1.0")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
