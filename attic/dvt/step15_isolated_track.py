r"""
Step 15 - isolated tracking: hide the rival vessel from the tracker.

THE FAILURE THIS ADDRESSES
--------------------------
MedSAM2's memory bank stores appearance embeddings of the tracked object. Two
anechoic circles a millimetre apart produce near-identical embeddings. When the
vein compresses it shrinks and deforms, so its match to the STORED memory of a
round anechoic vessel degrades - while the artery still matches that memory
perfectly. The tracker re-identifies onto the artery.

This is not gradual drift. It is re-acquisition onto a better-matching
candidate, which is why it is abrupt, why it clusters after compression begins,
and why it never recovers: the artery then enters the memory bank and is
reinforced by every later frame. Observed end state is all three tracks - vein,
artery and clot - on one vessel while the other is unannotated.

WHAT WAS TRIED FIRST, AND MEASURED
----------------------------------
Four interventions, all validated against expert masks, none kept:

    argmax exclusivity        median IoU 0.564 -> 0.470; frames with the vein
                              on the artery DOUBLED (12.9% -> 26.6%)
    suppress collided frames  killed 94 good polygons to remove 84 junk
    subtract + largest blob   median IoU 0.298 -> 0.237 on collided frames
    negative point prompts    overlap 0.239 -> 0.238 (no effect)

The first three act AFTER the mask exists and cannot recover a boundary the
model never found. The fourth acts only at the prompt frame: a label-0 point
shapes frame 0, and nothing in the memory bank carries "not that object"
forward.

Shape helps only a little. Artery solidity < 0.90 flags a union with ZERO false
positives over 6384 clean masks (expert arteries sit at p1 = 0.962), but only
13.4% sensitivity - because the dominant failure is two perfectly ROUND masks
on one vessel. Each is a plausible vessel; they are simply the same one. Vein
solidity cannot be used at all: its p1 is 0.382, since a compressed vein is
legitimately a slit.

WHY OCCLUSION, AND WHY IT MUST BE DYNAMIC
-----------------------------------------
If the artery is not visible in the frames fed to the vein's tracker, the vein
track cannot re-identify onto it. The constraint holds on every frame instead
of only the prompt frame.

A STATIC window does not work, and this was measured rather than assumed.
Vessels travel as the probe compresses tissue:

    max centroid displacement / box side   median 0.72-0.79   p90 1.56-1.75

A frozen ROI around the seed box holds its vessel for a whole clip only 52.5%
of the time at k=0.5 and 81.7% at k=1.0 - and k=1.0 spans three box-widths,
which swallows the rival anyway, since the two vessels sit a median 1.13
box-widths apart.

WHY THE OCCLUSION COMES FROM THE DETECTOR, NOT THE TRACKER
-----------------------------------------------------------
Deriving the mask to hide from the tracker's own output is self-defeating: once
the tracks have converged, the "rival" region IS the vessel being tracked, so
occluding it destroys the one good track. The detector is independent of the
tracker and finds both vessels on 99.4% of seed frames, so it can say where the
rival is on a frame where the tracker is already wrong.

A SIDE EFFECT WORTH HAVING
--------------------------
When the vein is compressed to near-nothing and the artery is hidden, the
tracker returns NOTHING rather than confidently grabbing the artery. A blank
frame is read downstream as "no measurement", which is safe. A wrong mask is
read as a real measurement, which is what corrupts the compression ratio.

COST
----
Two tracking passes plus per-frame detection, so roughly 3x `track_multi`.
Intended to be trialled on ten clips and LOOKED AT, not run on a thousand and
summarised.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from dvt.step14_medsam2_track import (EXCLUSIVE_ROLES, resolve_exclusive,
                                      role_of, worst_containment, write_frames)

#: How far to grow the rival's detected box before hiding it, as a fraction of
#: the box side. The detector's box is tight; the tracker can latch onto the
#: vessel wall just outside it.
OCCLUSION_MARGIN = 0.25

#: Never hide a region that overlaps the target's own predicted position by
#: more than this. Guards the case where the detector fires spuriously on the
#: vessel being tracked - hiding the target is far worse than not occluding.
MAX_SELF_OVERLAP = 0.20


def _feathered_mask(shape, box, margin=OCCLUSION_MARGIN, feather=0.35):
    """Soft-edged weight map, 1 inside the grown box, falling to 0 outside.

    A hard-edged black rectangle is itself a structure with strong gradients,
    and SAM 2 will happily segment it. The feather keeps the replacement from
    introducing a boundary the tracker can chase.
    """
    h, w = shape
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0 -= margin * bw; x1 += margin * bw
    y0 -= margin * bh; y1 += margin * bh

    fx, fy = max(feather * bw, 2.0), max(feather * bh, 2.0)
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]

    def ramp(v, lo, hi, f):
        return np.clip(np.minimum((v - lo) / f + 1.0, (hi - v) / f + 1.0), 0.0, 1.0)

    return ramp(xs, x0, x1, fx) * ramp(ys, y0, y1, fy)


def occlude(frame: np.ndarray, boxes, rng=None) -> np.ndarray:
    """Replace the given regions with plausible tissue, feathered at the edges.

    The fill is the local median plus speckle scaled to the surrounding
    variance, NOT a constant. A flat grey patch is as segmentable as a black
    one; the aim is a region that carries no vessel-shaped evidence at all.
    """
    if not len(boxes):
        return frame
    rng = rng or np.random.default_rng(0)
    out = frame.astype(np.float32)
    h, w = out.shape[:2]

    # tissue statistics from OUTSIDE the boxes, so a big anechoic vessel does
    # not drag the fill toward black
    keep = np.ones((h, w), bool)
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        keep[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = False
    ref = out[keep] if keep.any() else out
    med = float(np.median(ref))
    sd = float(np.std(ref)) * 0.6

    for b in boxes:
        wgt = _feathered_mask((h, w), b)
        if wgt.max() <= 0:
            continue
        fill = med + rng.normal(0.0, max(sd, 1.0), size=(h, w)).astype(np.float32)
        out = out * (1.0 - wgt) + fill * wgt

    return np.clip(out, 0, 255).astype(frame.dtype)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ub = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def _centroid_box(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _match_by_continuity(dets, tracks, gate=2.0, depth_order=None):
    """Assign detections to tracked objects by nearest previous position.

    WHY NOT RE-DECIDE ROLES EACH FRAME
    ----------------------------------
    The first version of this called `select_vessel_pair` on every frame
    independently, which re-ran the anatomical role decision from scratch each
    time. That decision is ~87% accurate overall and 81% at CFV, so on roughly
    one frame in six the vein's pass hid THE VEIN and left the artery fully
    visible - actively pushing the vein track onto the artery for a frame or
    two before recovering.

    Measured consequence: frame-to-frame jumps above 25% of the mask side rose
    from 1.4% to 4.0% for veins under severe compression, and the excursions
    were brief and self-recovering, which is the signature of a per-frame flip
    rather than a lost track.

    The role is decided ONCE at seeding. After that, a detection belongs to
    whichever track it is nearest to. `gate` is in units of the track's own
    box side; beyond it the detection is treated as unrelated rather than
    force-matched.

    `depth_order` (from `depth_order_of` at seeding) additionally forbids the
    tracks from passing through each other - see `_enforce_depth_order`.
    """
    out = {}
    used = set()
    order = sorted(tracks, key=lambda o: -tracks[o][1])   # larger track first
    for oid in order:
        box, side = tracks[oid]
        cx, cy = _centroid_box(box)
        best, best_d = None, gate
        for k, d in enumerate(dets):
            if k in used:
                continue
            dx, dy = _centroid_box(d.box)
            dist = float(np.hypot(cx - dx, cy - dy)) / max(side, 1.0)
            if dist < best_d:
                best, best_d = k, dist
        if best is not None:
            used.add(best)
            out[oid] = dets[best].box

    if depth_order:
        out = _enforce_depth_order(out, depth_order)
    return out


def depth_order_of(boxes: dict) -> dict:
    """Rank object ids by depth, shallowest first. Used to fix the ordering
    at seeding so later frames can be checked against it."""
    ranked = sorted(boxes, key=lambda o: _centroid_box(boxes[o])[1])
    return {oid: i for i, oid in enumerate(ranked)}


def _enforce_depth_order(assigned: dict, order: dict) -> dict:
    """Reject an assignment that would make two tracks pass through each other.

    Two vessels running parallel down a limb cannot swap depths. If the vein
    was deeper than the artery at seeding, it is deeper on every frame. Any
    inversion is the tracker's error, and it is detectable without ground
    truth.

    MEASURED on the v8 run, 1586 expert keyframes matched to a depth-order
    state:

        ordering as seeded   n=1369   vein mask on the vein 82.3%   IoU 0.673
        ordering INVERTED    n= 216   vein mask on the vein 30.1%   IoU 0.005

    Here the reference IS the seed, which is why this module can constrain the
    assignment directly. `scripts/fix_swaps.py`, working post-hoc without
    access to the seed, has to reconstruct it - and the reconstruction matters:
    taking it from the first 8 frames gives a 2.48:1 filter, while the site
    prior gives 0.71:1 and a whole-clip majority 0.77:1. Both of those re-decide
    the role per frame, which is what made depth voting fail.

    Median IoU 0.005 on inverted frames - not degraded, wrong. 42.4% of clips
    invert for 10+ consecutive frames, covering a median 41.2% of the clip.

    The mechanism: `_match_by_continuity` assigns each track to its nearest
    previous position. When the vein compresses, the two centroids close in -
    on one observed clip to 19 pixels - and the greedy assignment crosses.
    Continuity then locks the swap in permanently, so a clip is correctly
    labelled to frame 125 and wrongly labelled for the remaining 88.

    When an assignment inverts the order, the SWAPPED assignment is tried
    instead; if that is consistent it is used, otherwise both are dropped for
    that frame. Dropping is the safe failure - a missing polygon costs a
    radiologist far less than a wrong one, and downstream code already reads a
    blank as "no measurement".
    """
    ids = [o for o in assigned if o in order]
    if len(ids) < 2:
        return assigned

    def consistent(mapping):
        ranked = sorted(ids, key=lambda o: _centroid_box(mapping[o])[1])
        return all(order[ranked[i]] < order[ranked[i + 1]]
                   for i in range(len(ranked) - 1))

    if consistent(assigned):
        return assigned

    if len(ids) == 2:
        a, b = ids
        swapped = dict(assigned)
        swapped[a], swapped[b] = assigned[b], assigned[a]
        if consistent(swapped):
            return swapped

    # neither ordering works: the detections do not describe these tracks
    return {o: v for o, v in assigned.items() if o not in ids}


def locate_rivals(detector, det_size, frames, device, det_score, roles, boxes,
                  site=None, stride=1, verbose=False, collapse_frac=0.0):
    """Per-frame rival boxes for every tracked object, from the DETECTOR.

    Returns ({obj_id: {frame_idx: [boxes to hide]}}, stats).

    Roles are fixed at seeding; per-frame association is by CONTINUITY, never
    by re-running the anatomical decision. See `_match_by_continuity`.

    `collapse_frac` suppresses occlusion once the object's own detection has
    shrunk below this fraction of its largest detection in the clip. When the
    vein has collapsed there is nothing left to protect, and hiding the only
    other dark structure leaves the tracker nothing to hold - which turns a
    stable wrong track into a wandering one. 0 disables.
    """
    from dvt import step13_detector_seed as S13

    n = len(frames)
    vessel_ids = [o for o in boxes if roles.get(o) in EXCLUSIVE_ROLES]
    tracks = {o: (list(boxes[o]),
                  float(np.sqrt(max((boxes[o][2] - boxes[o][0]) *
                                    (boxes[o][3] - boxes[o][1]), 1.0))))
              for o in vessel_ids}

    matched_per_frame = {}
    stats = {"matched": 0, "unmatched": 0, "held": 0, "collapse_skips": 0}
    last_seen = {o: list(boxes[o]) for o in vessel_ids}
    sizes = {o: [] for o in vessel_ids}

    cached = {}
    for idx in range(n):
        if idx % max(stride, 1) == 0:
            dets = [d for d in S13.detect_frame(detector, det_size, frames[idx],
                                                device, det_score)
                    if d.role in ("vessel", "vein", "artery")
                    and S13.plausible_vessel(d, frames[idx].shape)]
            cached = _match_by_continuity(dets, tracks)
        m = {}
        for o in vessel_ids:
            if o in cached:
                m[o] = cached[o]
                last_seen[o] = cached[o]
                side = float(np.sqrt(max((cached[o][2] - cached[o][0]) *
                                         (cached[o][3] - cached[o][1]), 1.0)))
                tracks[o] = (cached[o], side)       # follow the motion
                sizes[o].append(side * side)
                stats["matched"] += 1
            else:
                m[o] = last_seen[o]                 # hold, do not re-decide
                sizes[o].append(np.nan)
                stats["held"] += 1
        matched_per_frame[idx] = m

    peak = {o: (np.nanmax(sizes[o]) if np.any(~np.isnan(sizes[o])) else 1.0)
            for o in vessel_ids}

    rivals = {oid: {} for oid in boxes}
    for idx in range(n):
        m = matched_per_frame[idx]
        for oid in boxes:
            mine = roles.get(oid)
            if mine not in EXCLUSIVE_ROLES:
                rivals[oid][idx] = []               # clot is never isolated
                continue
            own = sizes[oid][idx] if idx < len(sizes[oid]) else np.nan
            if (collapse_frac and not np.isnan(own)
                    and own < collapse_frac * peak[oid]):
                rivals[oid][idx] = []               # nothing left to protect
                stats["collapse_skips"] += 1
                continue
            rivals[oid][idx] = [m[o] for o in m if o != oid]

    if verbose:
        tot = stats["matched"] + stats["held"]
        print(f"      rival association: {stats['matched']}/{tot} matched, "
              f"{stats['held']} held, {stats['collapse_skips']} collapse skips")
    return rivals, stats


def track_isolated(predictor, frames: np.ndarray, boxes: dict[int, list],
                   roles: dict[int, str], detector, det_size: int,
                   device: str = "cuda", det_score: float = 0.5,
                   site: str | None = None, prompt_frame: int = 0,
                   stride: int = 1, seed: int = 0, verbose: bool = False,
                   collapse_frac: float = 0.15, bidirectional: bool = True,
                   rival_det_score: float | None = None):
    """Track each vessel in its own pass, with the other vessel hidden.

    Clot is tracked in the VEIN's pass with nothing hidden, because clot must
    be free to sit inside the vein - 98% of clot area falls within the vein
    outline at the median across 339 expert frames - and hiding either from the
    other would destroy it.

    Returns ({obj_id: (areas, masks)}, diag) with the same contract as
    `track_multi(..., return_diag=True)`.
    """
    import torch

    rng = np.random.default_rng(seed)
    n, h, w = len(frames), frames.shape[1], frames.shape[2]
    out = {oid: np.zeros((n, h, w), dtype=bool) for oid in boxes}
    diag = {"frames": n, "passes": 0, "occluded_frames": 0,
            "self_overlap_skips": 0, "reprompts": 0, "chunk_resets": 0,
            "max_containment": 0.0, "overlap_events": 0}

    # A MISSED rival is worse than a spurious one here: if the detector does
    # not find the artery on a frame, nothing is hidden and the vein track is
    # free to re-identify onto it - which is the failure this whole module
    # exists to prevent. A spurious box merely blanks a patch of tissue, and
    # the self-overlap guard stops it blanking the target. So the floor for
    # LOCATING the rival is lower than the floor for SEEDING.
    rivals, assoc = locate_rivals(detector, det_size, frames, device,
                                  rival_det_score if rival_det_score is not None
                                  else det_score,
                                  roles, boxes, site, stride,
                                  verbose, collapse_frac=collapse_frac)
    diag.update({f"assoc_{k}": v for k, v in assoc.items()})

    # objects that share a pass: clot rides with the vein
    groups = {}
    for oid in boxes:
        key = "clot" if roles.get(oid) == "clot" else roles.get(oid, str(oid))
        groups.setdefault("vein" if key == "clot" else key, []).append(oid)

    for gname, oids in groups.items():
        lead = next((o for o in oids if roles.get(o) == gname), oids[0])
        tmp = Path(tempfile.mkdtemp(prefix=f"iso_{gname}_"))
        try:
            occluded = np.empty_like(frames)
            for i in range(n):
                hide = list(rivals[lead].get(i, []))
                # never hide the target itself
                safe = [b for b in hide if _iou(b, boxes[lead]) <= MAX_SELF_OVERLAP]
                diag["self_overlap_skips"] += len(hide) - len(safe)
                occluded[i] = occlude(frames[i], safe, rng)
                if safe:
                    diag["occluded_frames"] += 1

            write_frames(occluded, tmp)
            state = predictor.init_state(video_path=str(tmp))
            predictor.reset_state(state)
            diag["passes"] += 1

            for oid in oids:
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=int(prompt_frame),
                    obj_id=int(oid),
                    box=np.asarray(boxes[oid], dtype=np.float32))

            with torch.inference_mode():
                for idx, ids, logits in predictor.propagate_in_video(state):
                    if idx >= n:
                        continue
                    for k, o in enumerate(ids):
                        o = int(o)
                        if o in out:
                            out[o][idx] = (logits[k] > 0.0).cpu().numpy()[0]

                # Backwards from the prompt frame. `detector_seed` picks the
                # frame it is most confident on, anywhere in the first third,
                # so without this every frame before it is left unsegmented -
                # a median of 19 frames and a p90 of 33. An evaluation of SAM 2
                # across 21 medical datasets found bidirectional propagation
                # beats front-to-end.
                if bidirectional and prompt_frame > 0:
                    for idx, ids, logits in predictor.propagate_in_video(
                            state, reverse=True):
                        if idx >= n:
                            continue
                        for k, o in enumerate(ids):
                            o = int(o)
                            if o in out:
                                out[o][idx] = (logits[k] > 0.0).cpu().numpy()[0]

            predictor.reset_state(state)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # measure the thing this exists to fix, on the FINAL masks
    cont = np.zeros(n, dtype=float)
    for i in range(n):
        worst, _ = worst_containment({o: out[o][i] for o in out}, roles)
        cont[i] = worst
    diag["containment"] = [round(float(c), 4) for c in cont]
    diag["frames_overlapping"] = int((cont >= 0.30).sum())
    diag["frac_overlapping"] = round(float((cont >= 0.30).mean()), 4)
    diag["max_containment"] = round(float(cont.max()), 4)

    result = {oid: (m.reshape(len(m), -1).sum(axis=1).astype(int), m)
              for oid, m in out.items()}
    return result, diag
