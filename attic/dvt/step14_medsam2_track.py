r"""
Step 14 - MedSAM2 video tracking.

WHY
---
Every tracking strategy in `step9_medsam_infer.track_vessel` was hand-written,
and each failed in a measurable way on real clips:

    fixed box            drifts off the vessel as tissue shifts
    reseed from mask     leaked 2090 -> 34426 px (16x runaway)
    shrink with mask     ratcheted the other way, ratio 0.294 with visibly
                         under-segmented masks
    recenter (current)   works, but is translation-only and hand-tuned
    mask-prompt feedback no measurable effect (0.625 vs 0.627)

MedSAM2 replaces all of it. A memory bank and memory-attention module condition
each frame on previous frames and predictions, so temporal propagation is
learned rather than coded. The model is SAM2.1-tiny fine-tuned on a large
medical corpus including ~19k annotated ultrasound frames, at 512px input.

WHAT STAYS THE SAME
-------------------
The detector still supplies the frame-0 box. That component measurably works -
it moved clips with IoU>0.5 from 6% (heuristic seeder) to 20%, and the ratio
AUC from 0.345 to 0.604. MedSAM2 only replaces the propagation.

THE VEIN/ARTERY COLLAPSE - measured on a 317-clip run
-----------------------------------------------------
This module used to claim that giving each vessel its own SAM 2 object id was
itself a safeguard: "SAM 2 keeps object ids apart, so an artery that is
explicitly tracked is far less likely to be absorbed into the vein mask."

That claim was false. SAM 2 propagates every object id INDEPENDENTLY; there is
no cross-object competition at inference, so nothing prevents two ids settling
on the same pixels. Audited with `scripts/qc_overlap.py` over 317 sidecars
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

WHY `max_area_ratio` COULD NOT SEE IT
-------------------------------------
Replaying the guard's own logic over the run:

    artery area / vein area on clean frames        median 0.52
    => a total merge multiplies vein area by only  ~1.52x

so the effect is SMALLER than the 1.8 threshold it was run with. Worse, the
rolling median ratchets: at flagged frames the growth factor against the
previous 9 frames has median 1.01 - the contaminated frame looks normal because
the contamination is already in the history that defines "normal". Rejected
areas were appended unconditionally, so the reference absorbed the error it was
meant to detect. Measured catch rate:

    threshold 1.8 -> 4.1% of contaminated frames
    threshold 1.2 -> 13.8%

Lowering the threshold is not the fix. Area is the wrong channel.

THE FIX - three layers, cheapest first
--------------------------------------
1. `exclusive=True` (default). Where two VESSEL ids both claim a pixel, the
   higher logit wins. Free - the logits for every id on a frame are already in
   hand - and it makes the mutual-exclusion claim true instead of assumed.
2. `max_containment` (default 0.30). Re-prompt from the detector when vein and
   artery masks overlap, which is directly observable and catches the events
   the area test misses. Rejected frames no longer enter the rolling history,
   so the reference cannot ratchet.
3. `chunk` re-runs `init_state` every N frames, which is the only way to be
   sure the memory bank is actually clear. Mid-propagation re-prompting adds a
   prompt but does not evict the merged frames already in memory. Class A
   contamination starts at a median frame 21, with 7 of 19 clips contaminated
   inside the first 10 frames, so recovery has to be available early.

CLOT IS EXEMPT FROM EXCLUSION
-----------------------------
Clot is not a separate structure from the vein - `step12_clot_detect` measured
median clot/vein area 0.62 over 783 expert polygons, and only 12% of clots lie
fully inside the vein outline. Clot and vein SHOULD overlap. Applying argmax
across all ids would tear the clot away from the lumen it sits in, so exclusion
is applied only within `EXCLUSIVE_ROLES`.

KNOWN LIMITATION
----------------
MedSAM2 uses a FIXED memory bank. The authors note degraded tracking when a
target moves rapidly or leaves the memory range and reappears. A vein at peak
compression collapses to near-nothing, which is close to that case, so this is
the specific failure mode to watch for.

SETUP
-----
    git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2
    pip install -e ".[dev]"
    bash download.sh          # -> checkpoints/MedSAM2_latest.pt
"""
from __future__ import annotations

import os
import shutil
import tempfile
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
    `step11_expert_labels` exists to prevent.
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


def _chunk_dir(src: Path, dst: Path, lo: int, hi: int) -> Path:
    """Symlink frames [lo, hi) into their own directory, renumbered from 0.

    `init_state` wants a directory whose frames start at 0. Symlinks rather
    than copies: re-encoding 300 JPEGs per chunk would cost more than the
    inference the reset protects.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(lo, hi)):
        link = dst / f"{j:05d}.jpg"
        target = src / f"{i:05d}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.symlink(target, link)
        except OSError:                      # filesystems without symlinks
            shutil.copyfile(target, link)
    return dst


# ------------------------------------------------------------------ exclusion

def resolve_exclusive(logit_stack: dict[int, np.ndarray],
                      exclusive_ids) -> dict[int, np.ndarray]:
    """Turn per-object logits into masks, letting the highest logit win.

    `logit_stack` maps obj_id -> float array (H, W) of raw logits for ONE
    frame. Objects outside `exclusive_ids` are thresholded independently and
    may overlap anything, which is what clot requires.

    Masks for exclusive ids come back disjoint by construction. This cannot
    rescue a fully converged pair: if both ids sit on the artery, argmax splits
    the artery between them rather than restoring the vein. That is what
    `max_containment` and `chunk` are for - and the overlap is measured on the
    RAW masks, before this runs, so the diagnostic survives the mitigation.
    """
    exclusive_ids = set(exclusive_ids or ())
    out: dict[int, np.ndarray] = {}
    ex = [i for i in logit_stack if i in exclusive_ids]

    for oid, lg in logit_stack.items():
        if oid not in exclusive_ids:
            out[oid] = lg > 0.0

    if not ex:
        return out
    if len(ex) == 1:
        out[ex[0]] = logit_stack[ex[0]] > 0.0
        return out

    stack = np.stack([logit_stack[i] for i in ex], axis=0)
    winner = stack.argmax(axis=0)
    positive = stack.max(axis=0) > 0.0
    for k, oid in enumerate(ex):
        out[oid] = (winner == k) & positive
    return out


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

def constrain_clot(per_seed, max_ratio: float = 2.5,
                   min_containment: float = 0.50):
    """Blank clot masks that are geometrically impossible.

    MEASURED on 339 expert frames carrying both a clot and a vein polygon:

        clot area / vein area   median 0.80   p90 1.04   p99 1.57   max 1.68
        clot area inside vein   median 0.98   p10 0.88

    A bound of 2.0 rejects ZERO expert frames, so it cannot be discarding real
    thrombus. The v3 run produced a median of 3.43 and a maximum of 159 - the
    clot track had grown to several times the lumen it is supposed to sit in,
    on 43 of 67 clips exceeding the vein's own maximum area.

    That happened because clot is deliberately exempt from vessel exclusion
    (it genuinely overlaps the vein), which left it the one track in the run
    with no constraint at all. This restores a bound without reintroducing
    exclusion: clot may overlap the vein freely, it simply may not dwarf it or
    wander outside it.

    BOUNDS CALIBRATED AGAINST THE TRACKED VEIN, NOT THE EXPERT ONE
    ---------------------------------------------------------------
    Both thresholds operate on the mask the pipeline produces, so that is what
    they were measured against - 330 expert clot frames scored against v8's
    tracked vein masks (median vein IoU 0.743 on those frames):

        containment of expert clot   in EXPERT vein   median 0.982, p10 0.890
                                     in TRACKED vein  median 0.994, p10 0.000

    The tracked figure is BIMODAL: either the clot sits squarely inside the
    tracked vein, or the tracked vein is on the wrong vessel and containment
    is zero. Nothing lives in between. So the rejection rate is flat across
    the middle -

        bound 0.20 -> rejects 14.8% of real clot frames
        bound 0.30 -> 15.2%
        bound 0.50 -> 16.1%      <- default
        bound 0.70 -> 22.1%

    - and 0.50 costs 0.9 points over 0.30 while separating the two modes
    cleanly. The ~15% floor is frames where the vein track itself is wrong;
    rejecting a clot anchored to the wrong vessel is the correct outcome.

    `max_ratio` was 2.0, chosen because it rejects ZERO of 339 expert frames
    measured against the EXPERT vein. Against the TRACKED vein it rejects
    3.9%, because the tracked vein is sometimes much smaller than the truth:

        bound 1.5 -> 7.3%    2.0 -> 3.9%    2.5 -> 2.7%    3.0 -> 1.5%

    2.5 recovers 1.2 points of real clot and still catches the pathology it
    exists for - the v3 run had a median of 3.43 and a maximum of 159.

    A NOTE ON THE README's FIGURES
    ------------------------------
    The claim that only 12% of clots lie fully inside the vein and 24% lie
    mostly outside is a bookkeeping artifact. 7.3% of expert clot polygons sit
    on frames with NO vein polygon, where containment is undefined; scoring
    those as zero reproduces "42% fully inside, 19% mostly outside" almost
    exactly. Restricted to frames where both were drawn, clot is 98.6%
    contained at the median and only 2.7% lies mostly outside.

    A rejected frame becomes area 0, which downstream code already reads as
    "no measurement" rather than "no clot" - the distinction the step9 blank
    -frame bug turned on.

    Returns (per_seed, n_rejected). `per_seed` entries are
    (label, areas, masks, quality); areas and masks are modified in place.
    """
    vein = next(((a, m) for l, a, m, _q in per_seed
                 if role_of(l) == "vein"), None)
    if vein is None:
        return per_seed, 0
    v_areas, v_masks = vein

    rejected = 0
    for label, areas, masks, _q in per_seed:
        if role_of(label) != "clot":
            continue
        for i in range(min(len(areas), len(v_areas))):
            ca = int(areas[i])
            va = int(v_areas[i])
            if ca <= 0:
                continue
            bad = False
            if va > 20 and ca > max_ratio * va:
                bad = True                      # dwarfs the lumen
            elif va > 20:
                inside = int((masks[i] & v_masks[i]).sum()) / max(ca, 1)
                if inside < min_containment:
                    bad = True                  # not in the vein at all
            if bad:
                masks[i] = False
                areas[i] = 0
                rejected += 1
    return per_seed, rejected


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
    object to. That case needs `step15.track_isolated`.

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


def track_with_memory(predictor, frames: np.ndarray, box_xyxy,
                      prompt_frame: int = 0, bidirectional: bool = True,
                      obj_id: int = 1):
    """Propagate one box through the whole clip using learned memory.

    `bidirectional` propagates backwards from the prompt frame as well as
    forwards. This matters when the detector's most confident frame is not
    frame 0: without it, everything before the prompt is unsegmented. An
    evaluation of SAM 2 across 21 medical datasets found bidirectional
    propagation outperforms front-to-end.

    Returns (areas, masks) with the same shape contract as
    `step9_medsam_infer.track_vessel`, so callers can swap between them.
    """
    import torch

    tmp = Path(tempfile.mkdtemp(prefix="medsam2_"))
    try:
        write_frames(frames, tmp)
        state = predictor.init_state(video_path=str(tmp))
        predictor.reset_state(state)

        box = np.asarray(box_xyxy, dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=int(prompt_frame),
            obj_id=obj_id, box=box)

        h, w = frames.shape[1:3]
        masks = np.zeros((len(frames), h, w), dtype=bool)

        with torch.inference_mode():
            for idx, _ids, logits in predictor.propagate_in_video(state):
                masks[idx] = (logits[0] > 0.0).cpu().numpy()[0]
            if bidirectional and prompt_frame > 0:
                for idx, _ids, logits in predictor.propagate_in_video(
                        state, reverse=True):
                    masks[idx] = (logits[0] > 0.0).cpu().numpy()[0]

        predictor.reset_state(state)
        areas = masks.reshape(len(masks), -1).sum(axis=1).astype(int)
        return areas, masks
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _centroid(box):
    """Centre of an xyxy box, as (x, y) floats."""
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _prompt_object(predictor, state, frame_idx, obj_id, box, negatives):
    """Prompt one object with its box, plus negative points on its rivals.

    WHY NEGATIVE POINTS
    -------------------
    Measured over 389 colliding frames, 320 (82%) were UNIONS - one track's
    mask fully containing the other's - not slips onto the wrong vessel. The
    model is not confusing the two vessels; it is treating the pair as a single
    anechoic object and outlining both.

    Three post-hoc repairs were simulated against expert masks and all failed:

        argmax exclusivity   median IoU 0.564 -> 0.470, and DOUBLED the frames
                             where the vein sat on the artery
        frame suppression    killed 94 good polygons to remove 84 junk ones
        subtract + largest   median IoU 0.298 -> 0.237 on colliding frames

    The common flaw is that all three act AFTER the mask exists, and none can
    recover a boundary the model never found. A negative point acts BEFORE:
    SAM 2 takes label-0 points as "not this object", so marking the artery's
    centre when prompting the vein states at prompt time that the two are
    distinct objects. That is the mechanism the architecture provides for
    exactly this case.

    `clear_old_points=False` on the second call so the box prompt survives.
    """
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=int(frame_idx), obj_id=int(obj_id),
        box=np.asarray(box, dtype=np.float32))
    if not negatives:
        return
    pts = np.asarray([_centroid(b) for b in negatives], dtype=np.float32)
    lbl = np.zeros(len(pts), dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=int(frame_idx), obj_id=int(obj_id),
        points=pts, labels=lbl, clear_old_points=False)


def negatives_for(obj_id: int, boxes: dict, roles: dict) -> list:
    """Boxes that obj_id must NOT claim.

    Only cross-role vessel pairs count. Clot is excluded in both directions:
    it legitimately sits inside the vein - 98% of clot area falls within the
    vein outline at the median across 339 expert co-annotated frames - so a
    negative point on the vein would suppress the clot entirely, and a
    negative point on the clot would punch a hole in the vein.
    """
    mine = roles.get(obj_id)
    if mine not in EXCLUSIVE_ROLES:
        return []
    return [b for oid, b in boxes.items()
            if oid != obj_id
            and roles.get(oid) in EXCLUSIVE_ROLES
            and roles.get(oid) != mine]


def track_multi(predictor, frames: np.ndarray, boxes: dict[int, list],
                prompt_frame: int = 0, bidirectional: bool = True,
                detector=None, det_size: int = 800, device: str = "cuda",
                reprompt_every: int = 0, max_area_ratio: float = 0.0,
                det_score: float = 0.5,
                roles: dict[int, str] | None = None,
                site: str | None = None,
                exclusive: bool = True,
                max_containment: float = 0.30,
                chunk: int = 0,
                negative_points: bool = True,
                return_diag: bool = False):
    """Track several objects (vein, artery, clot) in ONE propagation pass.

    MedSAM2 supports multiple object ids natively, so prompting all of them
    before propagating costs one pass rather than three - the image encoder
    runs once per frame regardless of how many objects are tracked.

    Contrary to what this docstring used to claim, separate object ids do NOT
    keep the vessels apart on their own; see the module docstring for the
    317-clip measurement. Three guards now do:

    `exclusive`
        Where two ids whose role is in `EXCLUSIVE_ROLES` both claim a pixel,
        the higher logit takes it. Costs nothing. Clot is exempt, since clot
        and vein genuinely overlap.

    `max_containment`
        Re-prompt from the detector when the RAW (pre-exclusion) vein and
        artery masks overlap by at least this fraction of the smaller one.
        This is the guard that actually fires: replayed over a 317-clip run,
        the area test caught 4.1% of contaminated frames. Set 0 to disable.

    `max_area_ratio`
        Kept for comparison, but do not rely on it. It fires only on GROWTH,
        while a total vein/artery merge multiplies area by a median 1.52x, and
        its rolling reference ratchets. Rejected frames are now excluded from
        that history, which removes the ratchet but not the weak effect size.

    `negative_points`
        Prompt each vessel with the OTHER vessel's centre as a label-0 point.
        82% of collisions are unions rather than slips, and every post-hoc
        repair tried made things worse or was neutral; this separates the
        objects at prompt time instead. Clot is excluded in both directions.

    `chunk`
        Re-run `init_state` every N frames, re-seeding from the detector. This
        is the only mechanism here that genuinely CLEARS the memory bank;
        `add_new_points_or_box` mid-propagation adds a prompt without evicting
        the merged frames already stored. Needs `detector`. 0 disables.

    `roles` maps obj_id -> 'vein' | 'artery' | 'clot'. Without it, exclusion
    and the containment guard are both inert, because there is no way to know
    which ids are allowed to overlap - so callers should always pass it.

    `boxes` maps obj_id -> xyxy box on `prompt_frame`.
    Returns {obj_id: (areas, masks)}, or (that, diag) if `return_diag`.
    """
    import torch

    roles = roles or {}
    exclusive_ids = ({i for i in boxes if roles.get(i) in EXCLUSIVE_ROLES}
                     if exclusive else set())

    h, w = frames.shape[1:3]
    n = len(frames)
    out = {oid: np.zeros((n, h, w), dtype=bool) for oid in boxes}
    cont = np.zeros(n, dtype=float)
    diag = {"frames": n, "overlap_events": 0, "reprompts": 0,
            "chunk_resets": 0, "max_containment": 0.0}

    tmp = Path(tempfile.mkdtemp(prefix="medsam2_"))
    try:
        write_frames(frames, tmp)

        # One chunk == the whole clip unless chunking is asked for AND a
        # detector exists to re-seed each chunk.
        if chunk and chunk > 0 and detector is not None:
            bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
        else:
            bounds = [(0, n)]

        recent = {oid: [] for oid in boxes}

        def _redetect(frame_idx):
            """Fresh detector boxes on one frame, keyed by object id.

            Object ids were assigned in detection order at frame 0, so the
            mapping is reconstructed the same way: strongest detection of each
            class, in the same order.
            """
            from dvt import step13_detector_seed as S13
            dets = S13.detect_frame(detector, det_size, frames[frame_idx],
                                    device, det_score)
            # Match by ROLE, not by class id and not by detection order. The
            # merged-vessel checkpoint has no vein/artery classes at all, and
            # the old "strongest of each class, in the same order" rule could
            # silently hand the vein's object id an artery box whenever the
            # detector's ranking changed between frames.
            vein, art, _note = S13.select_vessel_pair(
                dets, frames[frame_idx].shape, site)
            fresh_by_role = {"vein": vein, "artery": art}
            clot = S13.select_clot(dets, vein, frames[frame_idx].shape,
                                   det_score)
            fresh_by_role["clot"] = clot
            return {oid: fresh_by_role[roles[oid]].box
                    for oid in boxes
                    if roles.get(oid) in fresh_by_role
                    and fresh_by_role[roles[oid]] is not None}

        def _apply(ids, logits, idx, measure):
            """Threshold, measure overlap, resolve exclusivity, store."""
            stack = {int(o): logits[k].detach().float().cpu().numpy()[0]
                     for k, o in enumerate(ids) if int(o) in out}
            raw = {o: lg > 0.0 for o, lg in stack.items()}
            worst = 0.0
            if measure:
                worst, _ = worst_containment(raw, roles)
                cont[idx] = worst
            resolved = (resolve_exclusive(stack, exclusive_ids)
                        if exclusive_ids else raw)
            for o, m in resolved.items():
                out[o][idx] = m
            return worst

        for ci, (lo, hi) in enumerate(bounds):
            sub = tmp if len(bounds) == 1 else _chunk_dir(
                tmp, tmp / f"chunk{ci:03d}", lo, hi)
            state = predictor.init_state(video_path=str(sub))
            predictor.reset_state(state)
            if ci:
                diag["chunk_resets"] += 1

            if ci == 0:
                chunk_boxes = dict(boxes)
                local_prompt = int(prompt_frame) - lo
            else:
                chunk_boxes = _redetect(lo) or dict(boxes)
                local_prompt = 0
            if not 0 <= local_prompt < (hi - lo):
                local_prompt = 0

            for oid, box in chunk_boxes.items():
                if oid not in out:
                    continue
                neg = (negatives_for(oid, chunk_boxes, roles)
                       if negative_points else [])
                _prompt_object(predictor, state, local_prompt, oid, box, neg)

            with torch.inference_mode():
                for local_idx, ids, logits in predictor.propagate_in_video(state):
                    idx = local_idx + lo
                    if idx >= n:
                        continue
                    worst = _apply(ids, logits, idx, measure=True)
                    diag["max_containment"] = max(diag["max_containment"], worst)

                    if detector is None or idx <= prompt_frame:
                        continue

                    # ---- guard 2: overlap. Measured on the RAW masks, since
                    # argmax makes the resolved ones disjoint by construction
                    # and would hide the very event being looked for.
                    need = False
                    if max_containment and worst >= max_containment:
                        need = True
                        diag["overlap_events"] += 1

                    if reprompt_every and (idx - prompt_frame) % reprompt_every == 0:
                        need = True

                    # ---- guard 3: area. Weak; retained for comparison only.
                    if max_area_ratio and not need:
                        for oid in out:
                            a = int(out[oid][idx].sum())
                            hist = recent[oid]
                            if a > 20 and hist:
                                med = float(np.median(hist))
                                if med > 20 and a > max_area_ratio * med:
                                    need = True
                                    break

                    # Only ACCEPTED frames update the rolling reference. The
                    # old code appended unconditionally, so contaminated areas
                    # entered the median meant to detect them and the growth
                    # factor at flagged frames sat at 1.01.
                    if not need:
                        for oid in out:
                            a = int(out[oid][idx].sum())
                            if a > 20:
                                recent[oid].append(a)
                                if len(recent[oid]) > 9:
                                    recent[oid].pop(0)
                    else:
                        fresh = _redetect(idx)
                        diag["reprompts"] += 1
                        for oid, b in fresh.items():
                            neg = (negatives_for(oid, fresh, roles)
                                   if negative_points else [])
                            _prompt_object(predictor, state, local_idx, oid,
                                           b, neg)

                if bidirectional and local_prompt > 0:
                    for local_idx, ids, logits in predictor.propagate_in_video(
                            state, reverse=True):
                        idx = local_idx + lo
                        if idx < n:
                            _apply(ids, logits, idx, measure=False)

            predictor.reset_state(state)

        result = {oid: (m.reshape(len(m), -1).sum(axis=1).astype(int), m)
                  for oid, m in out.items()}
        if return_diag:
            thr = max_containment or 0.30
            diag["containment"] = [round(float(c), 4) for c in cont]
            diag["frames_overlapping"] = int((cont >= thr).sum())
            diag["frac_overlapping"] = round(float((cont >= thr).mean()), 4)
            return result, diag
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
