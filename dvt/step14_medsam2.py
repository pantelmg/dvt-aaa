r"""
MedSAM2 predictor loading, vessel roles, and the shape constraint.

This module holds what the pipeline still uses from the MedSAM2 integration -
loading the video predictor, deciding which role a label denotes, measuring
mask overlap, and rejecting masks whose SHAPE cannot be a single vessel.
The propagation code it was built around is gone, and the reason is the
project's central finding. It is recorded here because the pruned pipeline
otherwise looks like an arbitrary design choice.

THE MODEL
---------
SAM2.1-tiny fine-tuned on a large medical corpus including ~19k annotated
ultrasound frames, at 512px input. A memory bank and memory-attention module
condition each frame on previous frames and predictions, so temporal
propagation is learned rather than hand-coded. That is why it was adopted:
every hand-written propagation strategy in the old MedSAM1 path failed
measurably (reseed-from-mask leaked 2090 -> 34426 px; shrink-with-mask
ratcheted to a 0.294 ratio with visibly under-segmented masks).

THE VEIN/ARTERY COLLAPSE - measured on a 317-clip run
-----------------------------------------------------
This module used to claim that giving each vessel its own SAM 2 object id was
itself a safeguard: "SAM 2 keeps object ids apart, so an artery that is
explicitly tracked is far less likely to be absorbed into the vein mask."

That claim was false. SAM 2 propagates every object id INDEPENDENTLY; there is
no cross-object competition at inference, so nothing prevents two ids settling
on the same pixels. Audited with `attic/scripts/qc_overlap.py` over 317 sidecars
(268 with vein and artery on the same frame, 35865 such frames):

    frames where the two masks overlap (containment >= 0.30)   33.4%
    clips affected                                             56.0%

    class A  tracks fully converged     19 clips   90.4% of frames
    class B  progressive contamination  64 clips   76.2%
    class C  late and transient        185 clips   10.1%

Two facts kill the obvious explanations:

* The SEEDS were fine. Frame-0 vein/artery box IoU is a median 0.050 on class A
  and 0.043 on class B. The detector put the boxes on different structures and
  the TRACKS converged afterwards. This is propagation, not seeding.
* Class B contamination ramps monotonically through the clip (149 flagged
  frames in the first decile, 937 in the ninth) with no mid-clip peak and no
  recovery. That is a merged mask entering the memory bank and being inherited
  forward, not a collapse-driven jump.

The error also biases the primary measurement in the dangerous direction. The
contaminated frames sit near the vein's area MAXIMUM (1.02-1.46x the clip
median), so min/max is deflated: removing them raised the ratio in all 124
clips where it moved, median 0.170 -> 0.462. A deflated ratio reads as a freely
compressible vein, i.e. a false NEGATIVE for thrombosis.

WHY THE AREA GUARD COULD NOT SEE IT
-----------------------------------
The original defence was a `max_area_ratio` cap. Replaying its own logic over
the run: artery/vein area on clean frames has median 0.52, so a TOTAL merge
multiplies vein area by only ~1.52x - smaller than the 1.8 threshold it ran
with. Worse, the rolling median ratchets: at flagged frames the growth factor
against the previous 9 frames has median 1.01, because the contamination is
already in the history that defines "normal". Measured catch rate: 4.1% at
threshold 1.8, 13.8% at 1.2. Area was simply the wrong channel.

WHAT WAS TRIED, AND WHAT ACTUALLY FIXED IT
------------------------------------------
Four repairs were built on top of propagation and measured. None worked well
enough: per-pixel argmax between vessel ids (`exclusive`), re-prompting on
observed overlap (`max_containment`), periodic memory-bank resets (`chunk`),
and hiding the rival vessel behind a feathered patch so each tracker could not
see it (`attic/dvt/step15_isolated_track.py`). Every one of them is a way to
repair or avoid a corrupted memory bank while still depending on it.

Removing propagation altogether is what worked. Measured on 211 frames with
expert boxes:

    propagated (best run)   collision 28.8%   median IoU 0.507
    per frame, no memory    collision  5.2%   median IoU 0.815

The encoder was never the problem; the memory bank was. `step16_per_frame`
segments each frame as its own one-frame video, and the propagation code -
`track_multi`, `resolve_exclusive`, the chunking and re-prompting - is in
`attic/dvt/step14_medsam2_track.py`.

WHAT REMAINS HERE
-----------------
`load_predictor`  build the MedSAM2 video predictor.
`write_frames`    dump a clip to the JPEG directory SAM 2 wants as input.
`role_of`         map an annotation label to "vein" / "artery" / "clot".
`containment`,
`worst_containment`
                  overlap between two masks, used as a diagnostic only. The
                  pipeline reports it; it no longer acts on it, because
                  per-frame segmentation removed the cause.
`constrain_shape` reject a mask that cannot be one vessel. This is the only
                  check that can say WHICH of two overlapping masks is broken:
                  a figure-8 spanning both vessels is wrong and the round mask
                  beside it is right. 0.0% false positives on 6384 clean masks,
                  but only ~13% sensitivity - the common failure is two ROUND
                  masks on one vessel, which shape cannot see. Kept because it
                  is free and never fires wrongly, not because it is a defence.

SETUP
-----
    git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2
    pip install -e ".[dev]"
    bash download.sh          # -> checkpoints/MedSAM2_latest.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_CKPT = "checkpoints/MedSAM2_latest.pt"

#: Roles that are physically disjoint and may therefore never share a pixel.
#: Clot is deliberately absent - see the module docstring.
EXCLUSIVE_ROLES = ("vein", "artery")


def role_of(label: str) -> str:
    """Map a polygon label to 'vein' | 'artery' | 'clot' | 'other'.

    Handles the US-DICOMizer panel names ('Common Femoral Vein # 1') and the
    ThrombUS+ expert vocabulary ('CFV Scanning Location 1'), where the vessel
    code ends in V for veins and A for arteries. Matching only on the words
    'vein'/'artery' drops the second scheme silently - the bug
    the site-prior arbitration in `step10_auto_seed` exists to prevent.
    """
    low = (label or "").lower()
    if "clot" in low or "thromb" in low:
        return "clot"
    if "vein" in low:
        return "vein"
    if "artery" in low or "arterial" in low:
        return "artery"
    head = (label or "").split()[0].upper() if (label or "").split() else ""
    if head in {"CFV", "FV", "PV", "GSV", "SFV", "DFV"}:
        return "vein"
    if head in {"CFA", "FA", "PA", "DFA", "SFA"}:
        return "artery"
    return "other"


def load_predictor(checkpoint: str = DEFAULT_CKPT, cfg: str = DEFAULT_CFG,
                   device: str = "cuda"):
    """Build the MedSAM2 video predictor.

    Requires the MedSAM2 repo on sys.path (pip install -e . inside it).
    """
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(cfg, checkpoint, device=device)


def write_frames(frames: np.ndarray, out_dir: Path) -> Path:
    """MedSAM2's video predictor reads a DIRECTORY of numbered JPEGs.

    Names must sort in temporal order, hence the zero padding: '10.jpg' sorts
    before '2.jpg' otherwise, silently reversing parts of the sequence.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        img = f if f.ndim == 2 else f[..., 0]
        Image.fromarray(img.astype(np.uint8)).convert("RGB").save(
            out_dir / f"{i:05d}.jpg", quality=95)
    return out_dir


# ------------------------------------------------------------------ exclusion

def containment(a: np.ndarray, b: np.ndarray) -> float:
    """|A n B| / min(|A|, |B|).

    Containment rather than IoU: a small artery mask sitting entirely inside a
    large vein mask has low IoU but has been completely captured, which is the
    failure that matters. Returns 0.0 when either mask is empty.
    """
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 or sb == 0:
        return 0.0
    return int((a & b).sum()) / min(sa, sb)


def worst_containment(masks: dict[int, np.ndarray],
                      roles: dict[int, str]) -> tuple[float, tuple | None]:
    """Largest containment between any two DIFFERENT exclusive roles.

    Two veins in one field are not a conflict (a duplicated site, not a merge),
    so only cross-role pairs are compared.
    """
    worst, pair = 0.0, None
    ids = [i for i in masks if roles.get(i) in EXCLUSIVE_ROLES]
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            i, j = ids[x], ids[y]
            if roles.get(i) == roles.get(j):
                continue
            c = containment(masks[i], masks[j])
            if c > worst:
                worst, pair = c, (i, j)
    return worst, pair


# ------------------------------------------------------------------ tracking

# ---------------------------------------------------------------- shape

#: Solidity (area / convex-hull area) below which an ARTERY mask is impossible.
#: Measured over 2593 expert artery polygons: p0.5 = 0.944, p1 = 0.962,
#: p5 = 0.990, median 1.006. An artery is round in transverse section and stays
#: round under compression - that is what makes compression diagnostic.
MIN_ARTERY_SOLIDITY = 0.90

#: Lobes after proportional erosion. 99.3% of 5570 expert vessel polygons erode
#: to a single blob; a figure-8 spanning two vessels erodes to two.
MAX_LOBES = 1

#: NOTE there is deliberately NO vein solidity bound. Expert VEIN polygons have
#: p1 = 0.382 and p5 = 0.752, because a compressed vein is legitimately a slit
#: or crescent. A solidity penalty on veins would reject exactly the frames the
#: compression ratio is measured from.


def _hull_area(mask: np.ndarray) -> float:
    """Convex-hull area by monotone chain. No cv2 dependency."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float(mask.sum())
    pts = sorted(set(zip(xs.tolist(), ys.tolist())))
    if len(pts) < 3:
        return float(mask.sum())

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    a = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0 or float(mask.sum())


def solidity(mask: np.ndarray) -> float:
    """area / convex-hull area. 1.0 for a disc or ellipse, ~0.8 for a figure-8."""
    a = float(mask.sum())
    if a < 20:
        return 1.0
    h = _hull_area(mask)
    return a / h if h > 0 else 1.0


def lobe_count(mask: np.ndarray) -> int:
    """Blobs remaining after eroding by 12% of the mask's own radius.

    Proportional rather than fixed, so the test means the same thing for a
    small compressed vein and a large relaxed one. A union of two vessels
    joined at a waist separates; a single oval does not.
    """
    from scipy import ndimage

    a = int(mask.sum())
    if a < 40:
        return 1
    r = max(int(round(0.12 * np.sqrt(a / np.pi))), 1)
    eroded = ndimage.binary_erosion(mask, np.ones((2 * r + 1, 2 * r + 1), bool))
    if not eroded.any():
        return 1
    lab, n = ndimage.label(eroded)
    if n <= 1:
        return int(n) or 1
    sizes = np.array(ndimage.sum(eroded, lab, range(1, n + 1)))
    return int((sizes > 0.10 * sizes.max()).sum())


def implausible_shape(mask: np.ndarray, role: str) -> str | None:
    """Why this mask cannot be a single vessel, or None if it can.

    Only ever says no when a real vessel would not look like this. Measured
    false-positive rates against 6384 clean auto masks: artery solidity 0.0%,
    lobe test 0.0-0.1%.
    """
    if int(mask.sum()) < 40:
        return None
    if lobe_count(mask) > MAX_LOBES:
        return "multi-lobed"
    if role == "artery" and solidity(mask) < MIN_ARTERY_SOLIDITY:
        return "artery not convex"
    return None


def constrain_shape(per_seed, min_artery_solidity: float = MIN_ARTERY_SOLIDITY):
    """Blank masks that cannot be a single vessel.

    WHAT THIS FIXES, AND WHAT IT DOES NOT
    -------------------------------------
    Every earlier repair failed for the same reason: given two overlapping
    masks, nothing told it WHICH one was broken. Argmax used logit magnitude,
    which is arbitrary. Shape is an arbiter - a figure-8 spanning both vessels
    is wrong and the round mask beside it is right.

    But it reaches only part of the problem. Against real collisions:

        artery solidity < 0.90    sensitivity 13.4%   false positives 0.0%
        >= 2 lobes                sensitivity  1.5%   false positives 0.0%

    The dominant failure is two perfectly ROUND masks on ONE vessel. Each is a
    plausible vessel; they are simply the same one, and shape has nothing to
    object to. That case is what per-frame segmentation removes.

    It is also ONE-DIRECTIONAL. A union labelled ARTERY is caught by the
    solidity bound; the identical union labelled VEIN is not, and cannot be,
    because no vein solidity bound exists that would not also reject the
    compressed veins the ratio is measured from. So this helps when the artery
    track swallows the vein and does nothing when the vein track swallows the
    artery.

    The lobe test is weaker still: a figure-8 with a 14px waist survives
    erosion at 12% of the blob radius, and widening the erosion would cost the
    zero false-positive rate. Measured sensitivity 1.5% against solidity's
    13.4%.

    Kept because it is free and never wrong, not because it solves anything.

    Returns (per_seed, n_rejected, reasons).
    """
    from collections import Counter

    rejected, reasons = 0, Counter()
    for label, areas, masks, _q in per_seed:
        role = role_of(label)
        if role not in EXCLUSIVE_ROLES:
            continue                     # clot shape is not constrained here
        for i in range(len(areas)):
            if areas[i] <= 0:
                continue
            why = implausible_shape(masks[i], role)
            if why:
                masks[i] = False
                areas[i] = 0
                rejected += 1
                reasons[f"{role}: {why}"] += 1
    return per_seed, rejected, dict(reasons)


