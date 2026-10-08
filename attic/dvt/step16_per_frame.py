r"""
Step 16 - per-frame segmentation. No memory, no propagation.

THE MEASUREMENT THAT JUSTIFIES THIS MODULE
-------------------------------------------
`scripts/test_memory_hypothesis.py` segmented 211 frames across 40 clips with
NO propagation - each frame prompted directly, one object at a time - and
compared against the v7 propagated run on the same kind of data:

                              single-frame      propagated (v7)
    vein/artery collision          5.2%              28.8%
    median containment             0.000              -
    fully merged (>= 0.9)          1.9%               -
    median IoU vs expert vein      0.815             0.507
    median IoU vs expert artery    0.845              -

    (sanity: vein mask on the vein 99.5%, artery on artery 99.1%)

Two conclusions, eight standard errors apart from chance:

1. The collisions come from the MEMORY BANK, not the encoder. Prompted
   directly, the typical vein/artery pair does not touch at all.
2. Propagation was costing about 0.3 IoU. The encoder already produces
   near-expert masks; the tracker was destroying them.

Six inference-time interventions had failed to move collisions out of a 24-31%
band - argmax exclusivity, frame suppression, mask subtraction, negative point
prompts, occlusion, and correcting the seed frame. All six tried to make
propagation behave. None of them could, because the problem is that propagation
happens at all.

WHAT THIS DOES INSTEAD
----------------------
Detect on every frame, associate detections to tracks by continuity, and prompt
each frame independently. The detector supplies the box that the memory bank
used to supply implicitly, and no state carries forward, so no frame can
inherit another frame's mistake.

The components already existed and were being thrown away: `step15` locates
vessels per frame with the detector and then runs propagation anyway.

WHAT IT COSTS
-------------
An encoder pass per frame per object, so slower than propagation - though the
occlusion path was already paying detector-per-frame. And masks flicker
slightly frame to frame, since nothing enforces temporal consistency. Boxes are
smoothed across frames rather than masks, which keeps the flicker down without
reintroducing the state that caused the original problem.

THE HONEST CAVEAT
-----------------
The 0.815 figure came from EXPERT boxes. The detector's boxes are worse, so
real per-frame IoU will land below that. The collision result should hold
regardless: collisions came from memory, and there is no memory here.
"""
from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from dvt.step14_medsam2_track import (EXCLUSIVE_ROLES, role_of,
                                      worst_containment, write_frames)
from dvt.step15_isolated_track import _iou as _iou_box
from dvt.step15_isolated_track import (_enforce_depth_order,
                                       _match_by_continuity, depth_order_of)

#: Frames either side used to smooth a box before prompting. Vessels move a
#: median 0.72-0.79 box-sides over a whole clip, so motion between adjacent
#: frames is small and a short window removes detector jitter without lagging
#: behind real movement. 0 disables.
DEFAULT_SMOOTH = 2

#: Grow the detector's box slightly before prompting. The detector is trained
#: on tight expert boxes and SAM 2 does better with a little context; measured
#: expert boxes are the target, not the prompt.
BOX_MARGIN = 0.08


def _smooth_boxes(seq, k=DEFAULT_SMOOTH):
    """Moving median over a +/-k window, per coordinate.

    Median rather than mean: a single frame where the detector jumps to the
    wrong structure should not drag its neighbours, and the median ignores it
    entirely while the mean would not.
    """
    if k <= 0 or len(seq) < 3:
        return list(seq)
    arr = np.array([b if b is not None else [np.nan] * 4 for b in seq],
                   dtype=float)
    out = []
    for i in range(len(arr)):
        lo, hi = max(0, i - k), min(len(arr), i + k + 1)
        win = arr[lo:hi]
        if np.all(np.isnan(win)):
            out.append(None)
            continue
        with np.errstate(all="ignore"):
            out.append(np.nanmedian(win, axis=0).tolist())
    return out


def _grow(box, margin=BOX_MARGIN, shape=None):
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0 -= margin * bw; x1 += margin * bw
    y0 -= margin * bh; y1 += margin * bh
    if shape is not None:
        h, w = shape[:2]
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
    return [x0, y0, x1, y1]


#: How far a candidate may sit from the clip-wide consensus, in units of the
#: consensus box's own side, before the frame is left empty. 2.5 was the value
#: simulated; tighter starts refusing frames where the vessel has genuinely
#: moved, looser lets a neighbouring vessel back in.
ANCHOR_MAX_DIST = 2.5


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ub = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _side(b):
    return max(float(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))), 1.0)


def _modal_box(boxes, thr=0.30):
    """The box with the most neighbours at IoU >= thr - the consensus choice.

    A mode rather than a mean: averaging boxes that sit on two different
    vessels produces a box on neither, which is exactly the artifact this
    exists to avoid.
    """
    best, best_n = None, -1
    for b in boxes:
        n = sum(1 for o in boxes if _box_iou(b, o) >= thr)
        if n > best_n:
            best, best_n = b, n
    return best


def _anchor_to_mode(raw, cands, vessel_ids, roles, stats):
    """Re-resolve each frame against a clip-wide consensus, without chaining.

    `raw[oid]` holds the per-frame verdicts and `cands[idx]` the candidates
    that produced them. For each object, the modal verdict across the whole
    clip becomes the anchor, and every frame then takes whichever candidate
    lies nearest it. Frames whose nearest candidate is further than
    `ANCHOR_MAX_DIST` are left empty rather than forced.

    No frame depends on its predecessor, so a single bad frame cannot
    propagate - which is what separates this from every continuity-based
    variant tried (see `locate_all_frames`).
    """
    out = {}
    for oid in vessel_ids:
        seq = raw.get(oid, [])
        present = [b for b in seq if b is not None]
        if len(present) < 3:
            out[oid] = seq
            continue
        anchor = _modal_box(present)
        if anchor is None:
            out[oid] = seq
            continue
        aside = _side(anchor)
        acx, acy = _centre(anchor)

        fixed = []
        for idx, b in enumerate(seq):
            pool = cands.get(idx) or []
            if not pool:
                fixed.append(None)
                continue
            best, bd = None, None
            for d in pool:
                cx, cy = _centre(d.box)
                dist = float(np.hypot(acx - cx, acy - cy)) / aside
                if bd is None or dist < bd:
                    best, bd = d.box, dist
            if best is not None and bd <= ANCHOR_MAX_DIST:
                if b is None or _box_iou(best, b) < 0.9:
                    stats["anchor_changed"] += 1
                fixed.append(list(best))
            else:
                fixed.append(None)
                stats["anchor_dropped"] += 1
        out[oid] = fixed
    return out


def locate_all_frames(detector, det_size, frames, device, det_score,
                      roles, boxes, site=None, smooth=DEFAULT_SMOOTH,
                      verbose=False, multi_vein_target="gsv",
                      depth_order=True, redecide_multivein=True):
    """A box per object per frame, from the detector, associated by continuity.

    Roles are fixed at seeding. Re-deciding them per frame was measured at 87%
    accuracy overall and 81% at CFV, which flipped roughly one frame in six -
    the cause of the brief vein excursions seen in v5.

    A frame with no detection for an object gets None, and the object is simply
    not segmented on that frame. That is the correct outcome: a missing polygon
    costs a radiologist far less than a wrong one, and downstream code already
    reads a blank as "no measurement" rather than "no vessel".
    """
    from dvt import step10_auto_seed as S10
    from dvt import step13_detector_seed as S13

    n = len(frames)
    vessel_ids = [o for o in boxes if roles.get(o) in EXCLUSIVE_ROLES]
    tracks = {o: (list(boxes[o]),
                  float(np.sqrt(max((boxes[o][2] - boxes[o][0]) *
                                    (boxes[o][3] - boxes[o][1]), 1.0))))
              for o in vessel_ids}

    # Depth ranking fixed at seeding. Two vessels cannot pass through each
    # other, so any later assignment that inverts this is the tracker
    # crossing its own tracks - measured at 42.4% of clips, and the vein mask
    # is on the vein only 30.1% of the time on inverted frames (median IoU
    # 0.005) against 82.3% when the order holds.
    # The constraint is optional because it was never cleanly ablated. It
    # shipped in v9 alongside the GS changes, and its attributed gains
    # (CFV/FV/PV site-named-vein 70.7->74.0, 64.9->74.1, 59.3->71.0) are an
    # inference, not a measurement. What IS measured: inversions fell 18.7% ->
    # 2.1%, mask IoU on frames that have a polygon was unchanged (0.7210 ->
    # 0.7233 on non-GS sites), and coverage fell 11.7% -> 13.3%. So it trades
    # ~1.6 points of coverage for removing frames whose masks are noise
    # anyway - on inverted frames the vein mask is on the vein 30.1% of the
    # time with median IoU 0.005.
    vessel_boxes = {o: boxes[o] for o in vessel_ids}
    order = (depth_order_of(vessel_boxes)
             if depth_order and len(vessel_boxes) >= 2 else None)

    clot_ids = [o for o in boxes if roles.get(o) == "clot"]
    raw = {o: [] for o in vessel_ids}
    raw_clot = {o: [] for o in clot_ids}
    per_frame_cands: dict[int, list] = {}
    clot_tracks = {o: (list(boxes[o]),
                       float(np.sqrt(max((boxes[o][2] - boxes[o][0]) *
                                         (boxes[o][3] - boxes[o][1]), 1.0))))
                   for o in clot_ids}
    stats = Counter()

    # MULTI-VEIN SITES RE-DECIDE EVERY FRAME; the rest follow the seed.
    #
    # Simulated on the detection dump, asking the same question of the same
    # boxes - does the chosen vein land on the SITE-NAMED vein:
    #
    #     site   seed-and-follow (v9)   decide per frame
    #     CFV          74.0%                 70.4%
    #     FV           74.1%                 65.6%
    #     PV           71.0%                 68.2%
    #     GS           36.4%                 51.0%
    #
    # Re-deciding costs accuracy where the seed is usually right and pays where
    # it is usually wrong. At GS the field holds three or four vessels and the
    # GSV ranks only 2nd among vein proposals by score (top-1 just 19.4% of the
    # time), so a single seed frame is a coin flip that then propagates through
    # the whole clip. At the other sites the seed is right ~70% of the time and
    # following it beats re-deciding, which is what caused the vein excursions
    # in v5.
    #
    # Scoped, not general - that is the whole point.
    redecide = redecide_multivein and (site or "") in S10.MULTI_VEIN_SITES

    for idx in range(n):
        dets = S13.detect_frame(detector, det_size, frames[idx], device,
                                det_score)
        vessels = [d for d in dets
                   if d.role in ("vessel", "vein", "artery")
                   and S13.plausible_vessel(d, frames[idx].shape)]

        if redecide and len(vessels) >= 2:
            # Record the per-frame verdict; the decision is resolved AFTER the
            # loop by clip-wide consensus. Deciding here and using it directly
            # is what made v10's GS tracks jump - see `_anchor_to_mode`.
            v_d, a_d, _note = S13.select_vessel_pair(
                vessels, frames[idx].shape, site,
                multi_vein_target=multi_vein_target)
            m = {}
            for oid in vessel_ids:
                pick = v_d if roles.get(oid) == "vein" else a_d
                if pick is not None:
                    m[oid] = pick.box
            per_frame_cands[idx] = vessels
            stats["redecided"] += 1
        else:
            # Match TWICE - once unconstrained, once with the depth-order
            # constraint - so a missing box can be attributed.
            #
            # The first version of this counter incremented on every miss
            # whenever ordering was enabled, which made it a restatement of
            # `missing` rather than a measurement: on the v10 run it read 9654
            # against 10099 missing, i.e. "ordering was on". Attribution needs
            # the counterfactual, so the unconstrained assignment is computed
            # and compared.
            m_free = _match_by_continuity(vessels, tracks)
            m = (_enforce_depth_order(dict(m_free), order)
                 if order else m_free)

        for o in vessel_ids:
            if o in m:
                raw[o].append(m[o])
                side = float(np.sqrt(max((m[o][2] - m[o][0]) *
                                         (m[o][3] - m[o][1]), 1.0)))
                tracks[o] = (m[o], side)
                stats["matched"] += 1
            else:
                raw[o].append(None)
                stats["missing"] += 1
                if redecide:
                    stats["miss_redecide"] += 1
                elif order is not None and o in m_free:
                    # continuity DID find a box; the ordering constraint
                    # rejected it
                    stats["miss_order_rejected"] += 1
                else:
                    # nothing matched this track at all - a detection failure,
                    # not one of our gates
                    stats["miss_no_detection"] += 1

        # ---- clot gets its OWN per-frame detections.
        #
        # It must NOT inherit the vein's box. `step12` established that
        # prompting the vein's box while calling the result "Clot" returns a
        # mask byte-identical to the vein on 201/201 frames - the label is a
        # string, and SAM 2 segments whatever the box encloses. An earlier
        # revision of this function did exactly that, which would have made
        # every clot polygon a copy of its vein.
        vein_id = next((v for v in vessel_ids if roles.get(v) == "vein"), None)
        vein_box = m.get(vein_id) if vein_id is not None else None
        clots = [d for d in dets if d.role == "clot"]
        cm = _match_by_continuity(clots, clot_tracks) if clots else {}
        for o in clot_ids:
            b = cm.get(o)
            # anchored to the vein: a clot box away from the lumen is not
            # thrombus. 98% of clot area sits inside the vein at the median
            # across 339 expert co-annotated frames.
            if b is not None and vein_box is not None and _iou_box(b, vein_box) < 0.05:
                b = None
                stats["clot_unanchored"] += 1
            if b is None:
                raw_clot[o].append(None)
                stats["clot_missing"] += 1
            else:
                raw_clot[o].append(b)
                side = float(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0)))
                clot_tracks[o] = (b, side)
                stats["clot_matched"] += 1

    # ---- MULTI-VEIN SITES: resolve by clip-wide consensus, not per frame.
    #
    # v10 used each frame's verdict directly. That lifted GS site-named-vein
    # accuracy from 36.4% to 61.0% and made the tracks jump: frame-to-frame
    # displacement above 25% of the mask side rose from 2.1% to 6.8%, and
    # above 60% from 1.0% to 4.9%. CFV/FV/PV were untouched, so the cause is
    # the re-decision, not anything shared.
    #
    # Simulated on the detection dump over 432 GS frames:
    #
    #     strategy                     named-vein   jumpy>25%
    #     seed once, follow (v9)          33.2%       18.2%
    #     re-decide every frame (v10)     49.9%       42.0%
    #     vote ALL, anchor to mode        47.9%       27.4%   <- this
    #     vote ALL, then follow           38.4%       17.3%
    #     vote first 15, then hold        36.8%       17.4%
    #     re-decide + jump gate           32.6%       21.1%
    #
    # Anchoring keeps 96% of the accuracy gain and cuts instability by 35%.
    # Every CHAINING variant collapses to 33-38%, because a single decision
    # held by continuity drifts and one wrong choice persists. Anchoring has no
    # accumulation: each frame picks the candidate nearest a consensus built
    # from all ~130 frames, so a bad frame cannot propagate.
    if redecide and per_frame_cands:
        raw = _anchor_to_mode(raw, per_frame_cands, vessel_ids, roles, stats)

    per_frame = {o: _smooth_boxes(raw[o], smooth) for o in vessel_ids}
    for o in clot_ids:
        per_frame[o] = _smooth_boxes(raw_clot[o], smooth)

    if verbose:
        tot = stats["matched"] + stats["missing"]
        print(f"      per-frame boxes: {stats['matched']}/{tot} detected, "
              f"{stats['missing']} missing"
              + (f" [{stats.get('miss_no_detection',0)} no detection, "
                 f"{stats.get('miss_order_rejected',0)} order-rejected, "
                 f"{stats.get('miss_redecide',0)} re-decide]")
              + (f" | re-decided per frame ({stats.get('redecided',0)})"
                 if redecide else "")
              + (f" | clot {stats.get('clot_matched',0)} detected, "
                 f"{stats.get('clot_missing',0)} missing, "
                 f"{stats.get('clot_unanchored',0)} off-vein"
                 if clot_ids else ""))
    return per_frame, dict(stats)


def segment_per_frame(predictor, frames: np.ndarray, boxes: dict[int, list],
                      roles: dict[int, str], detector, det_size: int,
                      device: str = "cuda", det_score: float = 0.5,
                      site: str | None = None, smooth: int = DEFAULT_SMOOTH,
                      batch: int = 24, verbose: bool = False,
                      multi_vein_target: str = "gsv",
                      depth_order: bool = True,
                      redecide_multivein: bool = True):
    """Segment every frame independently. Returns ({oid: (areas, masks)}, diag).

    Same contract as `track_multi(..., return_diag=True)` so callers can swap.

    Frames are written once and each is prompted as its own single-frame video.
    `init_state` is the expensive part, so frames are processed in chunks and
    the state is rebuilt per chunk rather than per frame.
    """
    import torch

    n, h, w = len(frames), frames.shape[1], frames.shape[2]
    out = {oid: np.zeros((n, h, w), dtype=bool) for oid in boxes}
    diag = {"frames": n, "mode": "per_frame", "passes": 0,
            "reprompts": 0, "chunk_resets": 0}

    per_frame, stats = locate_all_frames(detector, det_size, frames, device,
                                         det_score, roles, boxes, site,
                                         smooth, verbose, multi_vein_target,
                                         depth_order, redecide_multivein)
    diag.update({f"box_{k}": v for k, v in stats.items()})

    # Each frame is written once, then presented to SAM 2 as its OWN
    # one-frame video via a symlink. This is exactly the pattern
    # `test_memory_hypothesis.py` used to produce the 5.2% / 0.815 result, so
    # it is known to work against this build of MedSAM2 - preferable to
    # `propagate_in_video(start_frame_idx=..., max_frame_num_to_track=0)`,
    # whose semantics vary between SAM 2 revisions.
    #
    # Both objects are prompted into the SAME state, so it is one `init_state`
    # per FRAME rather than per frame per object. That is the dominant cost of
    # this module and the reason it is slower than propagation.
    tmp = Path(tempfile.mkdtemp(prefix="perframe_"))
    try:
        write_frames(frames, tmp)
        one = tmp / "single"
        one.mkdir(exist_ok=True)
        link = one / "00000.jpg"

        for idx in range(n):
            prompts = {}
            for oid in boxes:
                b = per_frame[oid][idx] if idx < len(per_frame[oid]) else None
                if b is None or any(np.isnan(v) for v in b):
                    continue
                prompts[oid] = _grow(b, shape=(h, w))
            if not prompts:
                diag["frames_unprompted"] = diag.get("frames_unprompted", 0) + 1
                continue

            if link.exists() or link.is_symlink():
                link.unlink()
            src = tmp / f"{idx:05d}.jpg"
            try:
                import os
                os.symlink(src, link)
            except OSError:
                shutil.copyfile(src, link)

            state = predictor.init_state(video_path=str(one))
            predictor.reset_state(state)
            diag["passes"] += 1
            for oid, b in prompts.items():
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=int(oid),
                    box=np.asarray(b, dtype=np.float32))
            with torch.inference_mode():
                for _f, ids, logits in predictor.propagate_in_video(state):
                    for k, o in enumerate(ids):
                        o = int(o)
                        if o in out:
                            out[o][idx] = (logits[k] > 0.0).cpu().numpy()[0]
                    break            # one-frame video: the first yield is it
            predictor.reset_state(state)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- WHERE A FRAME WAS LOST, per object.
    #
    # Replaying 235 no-polygon frames against the detector dump split them:
    # 31.1% had the correct vein DETECTED, above the score floor, and CHOSEN by
    # pair selection - and still produced nothing. That bucket is everything
    # between selection and output, and the report could not distinguish its
    # causes because it hardcoded which diagnostics to copy.
    #
    # Two counters separate them:
    #   no_box    - continuity or the depth-order constraint supplied no box
    #   empty_mask - a box was prompted and MedSAM2 returned nothing
    for oid in boxes:
        no_box = empty = 0
        for idx in range(n):
            b = per_frame[oid][idx] if idx < len(per_frame[oid]) else None
            has_box = b is not None and not any(np.isnan(v) for v in b)
            if not has_box:
                no_box += 1
            elif not out[oid][idx].any():
                empty += 1
        diag[f"obj{oid}_no_box"] = no_box
        diag[f"obj{oid}_empty_mask"] = empty

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
