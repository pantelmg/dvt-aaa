r"""
Step 13 - Seed boxes from the trained detector.

WHY THIS REPLACES THE HEURISTIC SEEDER
--------------------------------------
`validate_seeder.py` compared 1790 auto-annotated frames against expert masks
and found the heuristic seeder in `step10_auto_seed` picks the wrong structure
most of the time:

    median IoU vs expert vein        0.0004
    clips with median IoU > 0.5           6%
    clips with median IoU < 0.1          66%
    within-clip IoU spread (p90-p10)  0.018

That last number matters: the seeder chooses one structure at frame 0 and
tracks it faithfully all clip. Tracking was never broken - the initial choice
was. And the degradation test showed the compression ratio tolerates almost any
boundary error (floor 0.68 even losing the vessel on 60% of clips), so
segmentation quality was not the bottleneck either.

WHY THE FIRST DETECTOR STILL LEFT 41% OF CLIPS OFF-VESSEL
---------------------------------------------------------
`expert_reference.py` measured the seeds actually used in a 317-clip run: on
**41% of clips at least one seed box was not on a vessel at all** (IoU < 0.3
against BOTH expert masks), while the detector's headline detection rate read
0.743. Both numbers are right; they measure different things.

    RECALL   is there ANY prediction of the right class matching this box
    TOP1     is the HIGHEST-SCORING prediction of that class the right one

The seeder consumes TOP1 - it takes the strongest box of each class and hands
it to the tracker for the whole clip. A correct box ranked third counts toward
recall and is invisible here.

Three changes follow from that, and this module implements the last two.

1. `train_detector.py` now selects checkpoints on TOP1, augments (it never did
   - the flag was stored and never read), and uses anchors measured from these
   boxes rather than torchvision's natural-image defaults.

2. **Vein and artery are ONE detection class.** They are not visually
   separable on a still B-mode frame - that is the clinical reason compression
   is the diagnostic test, and asking a detector to do it from appearance is
   asking for something the image does not contain. Merging gives the class
   6316 boxes instead of 3344 and 2972, and stops the loss spending capacity
   on a distinction it cannot learn. Role assignment moves entirely to
   geometry, which is measurably good at it: 96.0% at CFV, 95.7% at FV, 95.2%
   at PV, 85.4% at GS (`step10.SITE_PRIOR`).

3. **The two vessels are selected as a PAIR, not independently.** Measured
   over 2516 expert frames carrying both:

       centroid separation / mean box side   p1 0.62   median 1.13   p99 4.16
       box IoU between the two               median 0    p99 0.136   max 0.29
       size ratio larger/smaller             median 1.41 p99 3.83
       vessel side / max(W,H)                p1 0.048  median 0.128  p99 0.277

   Vein and artery are always adjacent, barely overlap, and are within ~4x of
   each other in size. A candidate pair that violates those is not a vein and
   an artery, however confident the detector is about each box separately -
   which is exactly the case the old "strongest of each class" rule could not
   see.

   HOW MUCH THIS BUYS - honestly. The bounds are calibrated to reject ZERO of
   1932 true expert pairs, so they cost nothing. But tested as a VETO on the
   seeds of a real 317-clip run, they did not discriminate: pairs where both
   boxes were on vessels were judged legal 87.3% of the time, and pairs with a
   box on nothing 88.2% of the time. Geometry alone cannot tell a vessel from a
   plausible-looking dark blob at a plausible distance.

   That test used TRACKED MASK bounding boxes as a stand-in for detections,
   because raw detector output for that run was not retained - and a tracked
   mask is blob-like by construction, so the stand-in is biased toward looking
   legal. The veto may do better on real candidate lists. It should not be
   relied on to.

   What the pair formulation does buy, independent of that: the two vessels
   are now chosen TOGETHER and ranked by anatomical ordinariness
   (`pair_plausibility` peaks at the observed median separation of 1.13), so
   among several legal candidates the ordinary one wins rather than whichever
   box the network happened to score highest. And roles are assigned by
   geometry rather than appearance, which is the part that measures well.

The trained detector reached 74% vein RECALL at IoU>=0.5 against the heuristic
seeder's 6% of clips. It also detects clot as a genuine class, which the
promptable MedSAM formulation structurally could not express: prompting the
vein box with a "Clot" label returned a mask byte-identical to the vein on
201/201 frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Two label schemes are supported so an old checkpoint still loads.
#: MERGED is what `train_detector.py --merge-vessels` (the default) produces.
CLASS_NAMES_MERGED = {1: "vessel", 2: "clot"}
CLASS_NAMES_SPLIT = {1: "vein", 2: "artery", 3: "clot"}
CLASS_NAMES = CLASS_NAMES_SPLIT          # kept for backwards compatibility

# ---------------------------------------------------------------- pair priors
# All measured on 2516 expert frames carrying both a vein and an artery.
# Bounds are set outside the observed p1/p99 so a real pair is never rejected.
# Calibrated so that ZERO of 1932 true expert pairs is rejected. An earlier
# cut of these numbers rejected 7.9% of them, almost all through MAX_REL_SIZE:
# the 0.277 p99 it was set from measured the GEOMETRIC MEAN of the pair, while
# an individual vein reaches 0.70 of the frame's longest side. Bounds derived
# from a summary statistic have to be re-derived per quantity.
MIN_SEPARATION = 0.15       # observed min 0.23 (p0.1)
MAX_SEPARATION = 8.00       # observed max 3.63
MAX_PAIR_BOX_IOU = 0.55     # observed max 0.447
MAX_SIZE_RATIO = 8.0        # observed max 6.18
MIN_REL_SIZE = 0.015        # observed min 0.020
MAX_REL_SIZE = 0.800        # observed max 0.698


@dataclass
class Detection:
    box: list[float]          # xyxy in ORIGINAL frame coordinates
    cls: int
    score: float
    role: str = "unknown"     # set by seeding, not by the network

    @property
    def area(self) -> float:
        return max(self.box[2] - self.box[0], 0.0) * \
            max(self.box[3] - self.box[1], 0.0)

    @property
    def side(self) -> float:
        return float(np.sqrt(max(self.area, 0.0)))

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0,
                (self.box[1] + self.box[3]) / 2.0)


def load_detector(checkpoint: str | Path, device: str = "cuda"):
    """Load the Faster R-CNN trained by scripts/train_detector.py.

    Returns (model, img_size, class_names). The class scheme travels in the
    checkpoint so a merged-vessel model and a legacy three-class model can both
    be used without a flag - guessing from the head width would break silently
    the day someone adds a class.
    """
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    img_size = ck.get("img_size", 800) if isinstance(ck, dict) else 800
    names = (ck.get("class_names") if isinstance(ck, dict) else None) or CLASS_NAMES_SPLIT
    names = {int(k): v for k, v in names.items()}

    # Head width comes from the WEIGHTS, never from the metadata. A checkpoint
    # trained before this fix has a 4-wide head even though its class_names say
    # two classes: `build_model(num_classes=NUM_CLASSES)` bound NUM_CLASSES as
    # a DEFAULT ARGUMENT at definition time, so main()'s `global NUM_CLASSES`
    # rebinding never reached it. The unused slot saw zero boxes and its logits
    # were trained never to fire, so those checkpoints are usable as they are -
    # but they will not load into a head sized from class_names.
    key = "roi_heads.box_predictor.cls_score.weight"
    n_out = int(state[key].shape[0]) if key in state else len(names) + 1
    if n_out != len(names) + 1:
        print(f"note: {Path(checkpoint).name} has a {n_out}-wide head for "
              f"{len(names)} named classes; building to match the weights")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_out)

    anchors = ck.get("anchors") if isinstance(ck, dict) else None
    if anchors:
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        sizes = tuple(tuple(s) for s in anchors["sizes"])
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=sizes,
            aspect_ratios=(tuple(anchors["ratios"]),) * len(sizes))

    model.load_state_dict(state)
    model.eval().to(device)
    return model, img_size, names


def detect_frame(model, img_size: int, gray: np.ndarray, device: str = "cuda",
                 min_score: float = 0.5,
                 class_names: dict | None = None) -> list[Detection]:
    """Detect vessels in one frame. Boxes come back in ORIGINAL coordinates.

    The resize scales the LONGEST side and preserves aspect, matching training
    and matching how SAM preprocesses. Squaring the frame would distort the
    anatomy: GR1 frames are 273x516, so a square resize stretches them 1.89x
    horizontally and presents every circular artery as a wide ellipse.
    """
    import torch
    import torch.nn.functional as F

    names = class_names or CLASS_NAMES_SPLIT
    h, w = gray.shape
    x = torch.from_numpy(gray).float()[None, None] / 255.0
    scale = img_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)[0].to(device)

    with torch.no_grad():
        out = model([x])[0]

    dets = []
    for b, l, s in zip(out["boxes"].cpu().numpy(),
                       out["labels"].cpu().numpy(),
                       out["scores"].cpu().numpy()):
        if s < min_score:
            continue
        d = Detection(box=[float(b[0] / scale), float(b[1] / scale),
                           float(b[2] / scale), float(b[3] / scale)],
                      cls=int(l), score=float(s))
        d.role = names.get(int(l), "unknown")
        dets.append(d)
    return dets


# ---------------------------------------------------------------- pair logic

def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ub = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def plausible_vessel(det: Detection, shape) -> bool:
    """Is this box the right SIZE to be a vessel in this frame?

    Expert vessels occupy 4.8%-27.7% of the frame's longest side (p1-p99).
    A box far outside that is a near-field artifact or a whole-image blob, and
    the near-field band in particular scores excellent contrast because it
    abuts the bright surface - the failure that made the heuristic seeder pick
    it.
    """
    h, w = shape[:2]
    rel = det.side / max(h, w, 1)
    return MIN_REL_SIZE <= rel <= MAX_REL_SIZE


def pair_plausibility(a: Detection, b: Detection) -> float:
    """0.0 if this cannot be a vein/artery pair, else a score in (0, 1].

    Rejects on the three measured constraints. The returned score prefers pairs
    near the observed median separation, so that when several pairs are legal
    the most anatomically ordinary one wins rather than the one whose boxes the
    network happened to score highest.
    """
    if _box_iou(a.box, b.box) > MAX_PAIR_BOX_IOU:
        return 0.0                       # duplicate detection, not a pair
    sa, sb = a.side, b.side
    if min(sa, sb) <= 0:
        return 0.0
    if max(sa, sb) / min(sa, sb) > MAX_SIZE_RATIO:
        return 0.0
    (ax, ay), (bx, by) = a.centroid, b.centroid
    sep = float(np.hypot(ax - bx, ay - by)) / ((sa + sb) / 2.0)
    if not (MIN_SEPARATION <= sep <= MAX_SEPARATION):
        return 0.0
    # median separation is 1.13; fall off gently either side of it
    return float(np.exp(-abs(np.log(sep / 1.13))))


def select_vessel_pair(dets: list[Detection], shape,
                       site: str | None = None, min_margin: float = 0.5,
                       multi_vein_target: str = "gsv"):
    """Choose the vein and the artery together, then label them by anatomy.

    The old rule kept "the strongest of each class" independently, so a
    confident box on a shadow could become the vein while the real vein sat
    one rank below. Scoring PAIRS instead lets the geometry veto that: the two
    vessels are adjacent, barely overlapping and within ~4x in size, and a
    spurious box almost never satisfies all three against a real one.

    Returns (vein, artery, note). Either may be None. `note` is one of
    'pair', 'single', 'none' - and is worth logging, because a clip seeded
    from a single vessel has no artery track and therefore no collision to
    detect later.

THE MARGIN GATE APPLIES ONLY WITH A SPLIT CHECKPOINT
    ----------------------------------------------------
    When the detector distinguishes vein from artery, geometry overrides it
    only if the depth margin clears `min_margin`; below that the detector's
    call stands. Appearance and geometry are each unreliable in different
    places, and the margin is where they trade places.

    With a MERGED checkpoint there is no detector opinion, so geometry always
    decides and the gate is inert. That is the configuration in which the
    fallbacks below were tested and rejected - they are alternatives to
    geometry, not to the detector. Measured over 1869 expert frames:

        site   depth rule   size rule
        CFV      0.843        0.523
        FV       0.914        0.536
        PV       0.947        0.499
        GS       0.465        0.879

    Size is a coin flip at the femoral and popliteal levels, and at CFV on the
    low-margin frames specifically it gets WORSE the more confident it is -
    0.547 overall, 0.469 when the area difference exceeds 50%, 0.341 above 80%.
    A hybrid (depth when the margin clears 0.5, size otherwise) scores 0.620 at
    CFV against 0.843 for depth alone. Gating would only lose coverage.

    Voting across the frames `detector_seed` already searches was also tested
    and does not help: 0.917 single-frame against 0.895 majority and 0.898
    margin-weighted, over 314 clips.

    WHY CFV IS THE WEAK SITE, AND WHY THAT IS ANATOMY
    -------------------------------------------------
    The depth margin at CFV has median 0.25 against 0.71 at FV and 0.72 at PV:
    at the common femoral level the vein and artery sit side by side rather
    than stacked, so small box errors flip the comparison. Accuracy tracks the
    margin at every site - 0.77-0.91 below 0.5, 0.92-0.97 above it.

    On detector output the roles come out at CFV 0.81, FV 0.97, GS 0.80,
    PV 1.00, which is parity with the expert-box figures above. The priors
    transfer; CFV is simply hard.

    A TEMPTING CUE THAT MUST NOT BE USED
    ------------------------------------
    The obvious better discriminator is compressibility: the vein collapses
    under the probe and the artery does not. It is the clinical test, the whole
    clip is available, and it needs no new model.

    It is also circular here, in the direction that destroys the study. A
    THROMBOSED vein does not compress - that is the finding being detected - so
    a rule that defines the vein as "the vessel that compresses more" would
    relabel every DVT-positive vein as an artery, and the compression ratio of
    the surviving "vein" would be low by construction. The feature would look
    excellent and measure nothing.
    """
    from dvt import step10_auto_seed as S10

    vessels = [d for d in dets
               if d.role in ("vessel", "vein", "artery")
               and plausible_vessel(d, shape)]
    vessels.sort(key=lambda d: -d.score)
    if not vessels:
        return None, None, "none"
    if len(vessels) == 1:
        v = vessels[0]
        v.role = "vein"                  # the target of the protocol
        return v, None, "single"

    # ---- multi-vein sites need the TARGET vein identified first
    #
    # At GS the field is the saphenofemoral junction: great saphenous, common
    # femoral vein and common femoral artery all in view, with the expert
    # drawing more than one vein on 88.3% of frames. Choosing the best PAIR
    # there can settle on CFV+CFA and never consider the GSV at all - which is
    # what happened, giving 8.4% site-named-vein accuracy against 71% for
    # "matches any vein".
    #
    # So: pick the target vein by its own rule, then give it the nearest
    # plausible companion as the artery.
    if (site or "") in S10.MULTI_VEIN_SITES and len(vessels) >= 2:
        return _select_multivein(vessels, site, multi_vein_target)

    best, best_val = None, 0.0
    for i in range(len(vessels)):
        for j in range(i + 1, len(vessels)):
            a, b = vessels[i], vessels[j]
            geom = pair_plausibility(a, b)
            if geom <= 0:
                continue
            val = geom * (a.score * b.score) ** 0.5
            if val > best_val:
                best, best_val = (a, b), val

    if best is None:
        v = vessels[0]
        v.role = "vein"
        return v, None, "single"

    a, b = best
    idx, margin, _kind = S10.prior_picks_vein(a.box, b.box, site or "")

    # Does the DETECTOR have an opinion? A split (vein/artery/clot) checkpoint
    # does; a merged (vessel/clot) one does not, and everything below has to
    # degrade gracefully to geometry-only in that case.
    det_says = None
    if a.role == "vein" and b.role == "artery":
        det_says = (a, b)
    elif b.role == "vein" and a.role == "artery":
        det_says = (b, a)

    if idx is None:
        # no prior for this site
        if det_says is not None:
            vein, art = det_says
        else:
            # the femoral vein is the larger on relaxed frames, which is where
            # seeding happens - 74% at CFV, and only 51% across whole clips
            vein, art = (a, b) if a.side >= b.side else (b, a)
    elif det_says is not None and margin < min_margin:
        # AMBIGUOUS GEOMETRY, and the detector has something to say.
        #
        # This gate was absent while the merged checkpoint was in use, because
        # a merged model has no vein/artery opinion to fall back to. With a
        # split checkpoint it matters a great deal: v4 replaced appearance with
        # geometry-only and subject AUC fell 0.7415 -> 0.6141, with the vessel
        # labelled "vein" no longer even reliably the larger of the pair (peak
        # area ratio 1.61 -> 1.19).
        #
        # The margin is where the two sources trade places. Measured on expert
        # boxes, the depth rule scores 0.77-0.91 below a margin of 0.5 and
        # 0.92-0.97 above it; a swept comparison put arbitration at 87.3% with
        # this gate against 83.9% without it - WORSE than doing nothing.
        #
        # CFV is where this bites hardest: its median depth margin is 0.25
        # against 0.71-0.72 elsewhere, because the vessels sit side by side
        # rather than stacked, so 80.7% of CFV frames land under the gate and
        # keep the detector's call.
        vein, art = det_says
    else:
        vein, art = (a, b) if idx == 0 else (b, a)

    vein.role, art.role = "vein", "artery"
    return vein, art, "pair"


def _select_multivein(vessels, site, target="gsv"):
    """Identify the target vein among several, then pair it with an artery.

    Measured on 415 GS frames across 73 clips carrying a GSV plus at least one
    other vein: the GSV is the SHALLOWEST vein on 99.3% of them and the
    SMALLEST on 88.2%. Both agree on 87.5%, and at least one is right on 100%.
    Shallowest leads; size breaks ties.

    The companion artery is then whichever remaining detection makes the most
    ordinary pair with it - the same `pair_plausibility` used elsewhere, so a
    box on a shadow still cannot become the artery.
    """
    from dvt import step10_auto_seed as S10

    def depth(d):
        return d.centroid[1]

    if target == "deep":
        # The DEEP-system vein, not the one the filename names. Measured on 418
        # GS frames with two or more veins, the deepest vein is a deep-system
        # vein 99.3% of the time (415 CFV against 3 GSV) - the most reliable
        # rule in this pipeline. See `S10.MULTI_VEIN_TARGET_DEFAULT` for why
        # this is a clinical choice rather than a technical one.
        ordered = sorted(vessels, key=lambda d: (-depth(d), -d.side))
        vein = ordered[0]
        rest = [d for d in vessels if d is not vein]
        art, best = None, 0.0
        for d in rest:
            val = pair_plausibility(vein, d) * (d.score ** 0.5)
            if val > best:
                art, best = d, val
        if art is None and rest:
            art = max(rest, key=lambda d: d.score)
        if art is None:
            vein.role = "vein"
            return vein, None, "single"
        vein.role, art.role = "vein", "artery"
        return vein, art, "pair"

    rule = S10.MULTI_VEIN_RULE.get(site, ("shallowest", "smallest"))
    prime = rule[0]

    if prime == "shallowest":
        ordered = sorted(vessels, key=lambda d: (depth(d), d.side))
    else:
        ordered = sorted(vessels, key=lambda d: (d.side, depth(d)))
    vein = ordered[0]

    # a near-tie on the primary cue is broken by the secondary one
    if len(ordered) > 1 and prime == "shallowest":
        gap = abs(depth(ordered[1]) - depth(vein)) / max(vein.side, 1.0)
        if gap < 0.35 and ordered[1].side < vein.side:
            vein = ordered[1]

    rest = [d for d in vessels if d is not vein]
    if not rest:
        vein.role = "vein"
        return vein, None, "single"

    # The companion must be the ARTERY, not the deep vein that also sits in
    # this field. Among the non-GSV vessels the artery is SHALLOWER than the
    # deep vein on 89.2% of 389 expert frames - the same femoral rule (vein
    # deeper than artery) applied one level down. Size is only 67.1% and is
    # not used.
    #
    # Without this the plausibility score alone tends to pick the common
    # femoral VEIN as the artery, because it is the larger, more ordinary
    # partner for the little saphenous.
    if len(rest) >= 2:
        rest = sorted(rest, key=lambda d: d.centroid[1])   # shallowest first
        deepest = rest[-1]
        candidates = [d for d in rest if d is not deepest]
    else:
        candidates = rest

    art, best = None, 0.0
    for d in candidates:
        val = pair_plausibility(vein, d) * (d.score ** 0.5)
        if val > best:
            art, best = d, val
    if art is None:                       # geometry vetoed every candidate
        for d in candidates:
            if art is None or d.score > art.score:
                art = d
    if art is None:
        vein.role = "vein"
        return vein, None, "single"

    vein.role, art.role = "vein", "artery"
    return vein, art, "pair"


def select_clot(dets: list[Detection], vein: Detection | None,
                shape, min_score: float = 0.3):
    """Best clot box, required to sit on the vein.

    Expert clot geometry over 339 co-annotated frames: median 0.80 of the vein's
    area, max 1.68, and 0.98 of the clot inside the vein outline at the median.
    A clot box far from the vein is not thrombus, so the vein - once chosen -
    is the strongest filter available for it.

    Without a vein there is nothing to anchor to, and an unanchored clot
    proposal is what produced `clot_fraction` values up to 159 in the v3 run.
    """
    clots = [d for d in dets if d.role == "clot" and d.score >= min_score]
    if not clots:
        return None
    if vein is None:
        return None
    clots.sort(key=lambda d: -d.score)
    for c in clots:
        if _box_iou(c.box, vein.box) >= 0.10:
            return c
    return None


def detector_seed(model, img_size: int, frames: np.ndarray, device: str = "cuda",
                  search_frames: int = 8, min_score: float = 0.5,
                  want_clot: bool = True, clot_score: float | None = None,
                  site: str | None = None, class_names: dict | None = None,
                  multi_vein_target: str = "gsv"):
    """Choose seed boxes by detecting on several frames and taking the best.

    Several frames are searched because a single frame can be shadowed or
    mid-compression. Frames are scored by the PAIR they yield rather than by
    the best single detection: a frame showing one crisp vessel and nothing
    else is worse to seed from than one showing both, even if its top box
    scores higher.

    `clot_score` sets a separate floor for the clot class. One shared threshold
    starves it - clot detects at 0.587 against the vein's 0.743, so at
    --det-score 0.5 only 78 of 317 clips came back with any clot track at all,
    and clot presence has the best expert ceiling of any feature here (subject
    AUC 0.828 against the compression ratio's 0.785). A missed clot is a missed
    diagnosis; a spurious one is bounded by `step14.constrain_clot`.

    Returns (detections, frame_index) where detections carry `.role` already
    assigned - callers should read `.role`, never the raw class id, because the
    merged-vessel checkpoint has no vein/artery classes to read.
    """
    n = len(frames)
    if n == 0:
        return [], 0
    names = class_names or CLASS_NAMES_SPLIT
    floor = min(min_score, clot_score) if clot_score is not None else min_score

    # sample across the first third: the vein is widest before compression,
    # which is where it is easiest to detect
    idx = np.unique(np.linspace(0, max(n // 3, 1), min(search_frames, n))
                    .round().astype(int))

    best_out, best_frame, best_val = [], int(idx[0]), -1.0
    for i in idx:
        i = int(min(i, n - 1))
        dets = detect_frame(model, img_size, frames[i], device, floor, names)
        if not dets:
            continue
        vein, art, note = select_vessel_pair(dets, frames[i].shape, site,
                                             multi_vein_target=multi_vein_target)
        if vein is None:
            continue
        val = vein.score * (1.0 + (art.score if art is not None else 0.0))
        if note == "pair":
            val *= 1.5                   # prefer a frame showing both vessels
        if val <= best_val:
            continue

        keep = [vein] + ([art] if art is not None else [])
        if want_clot:
            clot = select_clot(dets, vein, frames[i].shape,
                               clot_score if clot_score is not None else min_score)
            if clot is not None:
                keep.append(clot)
        best_out, best_frame, best_val = keep, i, val

    return best_out, best_frame
