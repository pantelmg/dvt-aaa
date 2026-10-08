r"""
Tests for vein/artery arbitration at seed time.

The priors encoded in `step10.SITE_PRIOR` were MEASURED on the 1044-file
expert set, clip-level majority over early frames:

    CFV  depth (vein deeper)       96.0%     size 74.0%
    FV   depth (vein deeper)       95.7%     size 60.9%
    PV   depth (vein superficial)  95.2%     size 64.3%
    GS   size  (vein larger)       85.4%     depth 63.4%

Two of those contradict what this codebase used to do, so these tests exist
mostly to stop the old rules creeping back:

  * depth was applied only to PV, when it is the best cue at CFV and FV too;
  * the femoral levels were decided on SIZE, which is 61-74% on relaxed frames
    and falls to 51% across whole clips - because the vein is being compressed,
    so it is only reliably larger before the probe presses.

    python scripts/test_arbitration.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt.step10_auto_seed import (  # noqa: E402
    SITE_PRIOR, arbitrate_detections, prior_picks_vein,
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


@dataclass
class Det:
    box: list
    cls: int
    score: float

    @property
    def role(self):
        return {1: "vein", 2: "artery", 3: "clot"}.get(self.cls, "unknown")


# boxes are xyxy; larger y is deeper (further from the transducer)
SHALLOW = [100, 100, 160, 160]      # depth 130
DEEP = [100, 200, 160, 260]         # depth 230
BIG = [100, 200, 200, 300]          # depth 250, area 10000
SMALL = [300, 200, 340, 240]        # depth 220, area 1600

print("SITE_PRIOR contents")
check(SITE_PRIOR["CFV"] == ("depth", "deeper"),
      "CFV uses depth, vein deeper (was size, 74%)")
check(SITE_PRIOR["FV"] == ("depth", "deeper"),
      "FV uses depth, vein deeper (was size, 61%)")
check(SITE_PRIOR["PV"] == ("depth", "superficial"),
      "PV uses depth, vein superficial")
check(SITE_PRIOR["GS"] == ("depth", "superficial"),
      "GS uses depth, vein SUPERFICIAL. The old ('size','larger') was "
      "backwards: over 415 GS frames the GSV is the shallowest vein on 99.3% "
      "and the largest on 11.6%. The earlier '63.4%' figure compared the vein "
      "against the ARTERY, where GSV/CFA really is ambiguous - vein against "
      "VEIN is near-perfect, so that measured the wrong pair")
check(SITE_PRIOR["CFV"][1] != SITE_PRIOR["PV"][1],
      "the depth SIGN differs between femoral and popliteal")
check(SITE_PRIOR["GS"][1] == SITE_PRIOR["PV"][1],
      "GS shares PV's sign - both target a vein lying above its artery")

print("\nprior_picks_vein")
i, m, k = prior_picks_vein(SHALLOW, DEEP, "PV")
check(i == 0 and k == "depth", "PV: the shallower box is the vein")
i, m, k = prior_picks_vein(SHALLOW, DEEP, "CFV")
check(i == 1, "CFV: the deeper box is the vein")
i, m, k = prior_picks_vein(SHALLOW, DEEP, "FV")
check(i == 1, "FV: the deeper box is the vein")
i, m, k = prior_picks_vein(SHALLOW, DEEP, "GS")
check(i == 0 and k == "depth",
      "GS: the SHALLOWER box is the vein - the great saphenous runs in the "
      "superficial fascia above the deep system")
i, m, k = prior_picks_vein(SHALLOW, DEEP, "OPT")
check(i is None, "unknown site yields no verdict rather than a guess")

print("\n  margins")
_i, m_far, _ = prior_picks_vein(SHALLOW, DEEP, "CFV")
_i, m_near, _ = prior_picks_vein([100, 100, 160, 160], [100, 104, 160, 164],
                                 "CFV")
check(m_far > 1.0, f"well-separated boxes give a large margin ({m_far:.2f})")
check(m_near < 0.15, f"near-identical depths give a small margin ({m_near:.2f})")

print("\narbitrate_detections")
# detector called the SHALLOW box the vein at a femoral site -> wrong
dets = [Det(SHALLOW, 1, 0.9), Det(DEEP, 2, 0.8)]
dets, v = arbitrate_detections(dets, "CFV")
check(v == "swapped", "CFV: a shallow 'vein' is overruled")
check(dets[0].role == "artery" and dets[1].role == "vein",
      "the labels actually swap on the objects")

# same geometry at PV -> detector was right
dets = [Det(SHALLOW, 1, 0.9), Det(DEEP, 2, 0.8)]
dets, v = arbitrate_detections(dets, "PV")
check(v == "agreed", "PV: the same geometry is left alone")
check(dets[0].role == "vein", "no swap means no mutation")

print("\n  refusals")
dets = [Det([100, 100, 160, 160], 1, 0.9), Det([200, 104, 260, 164], 2, 0.8)]
dets, v = arbitrate_detections(dets, "CFV")
check(v == "ambiguous",
      "a 4px depth difference is a coin flip - refuse rather than swap")
check(dets[0].role == "vein", "an ambiguous verdict leaves the detector's call")

dets = [Det(SHALLOW, 1, 0.9)]
_d, v = arbitrate_detections(dets, "CFV")
check(v == "n/a", "one vessel only: nothing to arbitrate")

dets = [Det(SHALLOW, 1, 0.9), Det(DEEP, 3, 0.8)]
_d, v = arbitrate_detections(dets, "CFV")
check(v == "n/a", "vein + clot is not a vein/artery pair")

dets = [Det(SHALLOW, 1, 0.9), Det(DEEP, 2, 0.8)]
_d, v = arbitrate_detections(dets, "OPT")
check(v == "n/a", "no prior for the site: leave it alone")

print("\n  the highest-scoring detection of each class is the one arbitrated")
dets = [Det(SHALLOW, 1, 0.95), Det([500, 500, 560, 560], 1, 0.3),
        Det(DEEP, 2, 0.9)]
dets, v = arbitrate_detections(dets, "CFV")
check(v == "swapped", "a low-scoring duplicate does not derail the decision")
check(dets[0].role == "artery" and dets[2].role == "vein",
      "the swap applies to the top-scoring pair")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
