"""
Step 10 - Find the vein automatically, with no human seed.

The radiologist should be correcting proposals, not creating them. This module
produces the box prompts that step 9 needs, so `run_autoannotate.py --auto`
works on a cold folder of DICOMs.

Two things make this tractable without a trained detector:

1. The anatomical site is already known. It is encoded in the filename tag
   (CFV/GS/FV/PV plus limb), so we know which vessel we are looking for and how
   it sits relative to its artery.
2. Vessel lumen is anechoic. Blood returns almost no echo, so vein and artery
   are the darkest compact structures in the field.

Anatomy that actually discriminates vein from artery:

* Popliteal (PV): scanned from the popliteal fossa, the vein lies SUPERFICIAL
  to the artery - closer to the transducer, i.e. smaller y. This is a reliable
  geometric rule and does not depend on probe rotation.
* Femoral levels (CFV, GS, FV): the vein is normally the LARGER of the pair and
  sits medial to the artery. Medial/lateral depends on probe orientation, which
  is not recorded, so size is the usable cue rather than side.

Both rules fail sometimes, which is the point of the correction step. The
module therefore proposes SEVERAL labelled candidates rather than one, because
deleting a wrong polygon in US-DICOMizer is far quicker than drawing a right
one.

An earlier naive attempt at this - global dark threshold plus connected
components - merged vein with artery on 52/158 frames of a real clip and merged
the vein with dark border shadow on 76/158. The filters below exist because of
those specific failures: border contact, extreme elongation and implausible
size are all rejected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# site -> (vein label, artery label) as offered by the US-DICOMizer panel
SITE_LABELS = {
    "CFV": ("Common Femoral Vein # 1", "Common Femoral Artery # 1"),
    "GS": ("Great Saphenous Vein # 3", "Common Femoral Artery # 2"),
    "FV": ("Femoral Vein # 3", "Femoral Artery # 3"),
    "FV-D": ("Femoral Vein # 3", "Deep Femoral Artery # 2"),
    "PV": ("Popliteal Vein # 4", "Popliteal Artery # 4"),
}

# Site codes as they appear in real ThrombUS+ filenames, longest first so that
# FV-D and GSV are matched before FV and GS. Counts from the 1044-file expert
# set: PVr-R 95, CFVr-R 95, FVr-R 95, GSVr-R 88, ... FV-Dr-L 2, OPT-L 21.
#
# Two conventions the original parser missed entirely:
#   * GSV, not GS - 245 files use the three-letter form
#   * a trailing "r" means a RANDOM STILL IMAGE (CFVr-L), not a compression
#     cine. Those are single-frame, so a compression ratio is undefined.
FILENAME_SITE_CODES = ("FV-D", "CFV", "GSV", "GS", "FV", "PV")

# OPT clips are operator-chosen extra views with no fixed anatomy, so the site
# cannot be inferred from the name and must be supplied with --site.
UNKNOWN_SITE_CODES = ("OPT",)


@dataclass
class SeedResult:
    candidates: list
    frame: int
    ambiguous: bool = False      # vein/artery assignment was a near tie
    weak: bool = False           # no frame had a convincing vessel
    note: str = ""


@dataclass
class Candidate:
    box: list[float]          # xyxy in frame coordinates
    area_px: int
    centroid: tuple[float, float]
    darkness: float           # mean intensity inside, lower is more anechoic
    ellipticity: float        # minor/major axis, 1.0 is circular
    depth_frac: float         # centroid y as a fraction of image height
    score: float
    contrast: float = 1.0     # interior vs surrounding wall; lumen is low
    role: str = "unknown"     # vein | artery | unknown


def _components(gray: np.ndarray, pct: float):
    """Dark connected components at a percentile threshold."""
    import cv2

    thr = np.percentile(gray, pct)
    m = (gray <= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    return n, lab, stats, cent


def find_candidates(gray: np.ndarray, max_candidates: int = 4,
                    percentiles=(6, 10, 15, 22),
                    max_contrast: float = 0.7) -> list[Candidate]:
    """Locate anechoic vessel-like structures in one frame.

    Several thresholds are tried because a single global percentile either
    misses a small artery or floods into acoustic shadow, depending on gain
    settings that differ across the five scanners in this cohort.
    """
    import cv2

    h, w = gray.shape
    img_area = h * w
    # anything proposed as lumen must sit below this intensity
    dark_ceiling = float(np.percentile(gray, 25))
    found: list[Candidate] = []

    for pct in percentiles:
        n, lab, stats, cent = _components(gray, pct)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]

            # reject: too small to be a vessel, or so large it is shadow
            if area < 0.0015 * img_area or area > 0.15 * img_area:
                continue
            # reject: touching the frame edge (border shadow, not anatomy)
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue
            # reject: extreme elongation (fascial planes, acoustic shadow)
            ell = min(bw, bh) / max(bw, bh)
            if ell < 0.18:
                continue
            # reject: near-field band or very deep shadow.
            #
            # 0.06 was far too permissive. On FR1 popliteal clips the dark band
            # just under the transducer footprint scored excellent contrast
            # (0.17) because it abuts the bright near-field, and BOTH proposals
            # landed there instead of on any vessel. The vessels targeted by
            # this protocol are never in the top eighth of the image.
            cy = cent[i][1]
            depth = cy / h
            if depth < 0.15 or depth > 0.85:
                continue

            # reject: sitting directly against the top of the imaging sector.
            # The near-field band is wide and shallow; real vessels are not.
            if y < 0.10 * h and bw > 2.5 * bh:
                continue

            comp = lab == i
            darkness = float(gray[comp].mean())
            # local contrast against the surrounding wall. A global reference
            # is useless here: ultrasound frames carry large black borders, so
            # on one FR1 clip the image median was 31 while true perivascular
            # tissue was ~47.
            ring = cv2.dilate(comp.astype(np.uint8),
                              np.ones((15, 15), np.uint8)) - comp.astype(np.uint8)
            surround = gray[ring > 0]
            contrast = (darkness / max(float(surround.mean()), 1e-6)
                        if surround.size > 10 else 1.0)
            if contrast > max_contrast:
                continue
            # hard anechoic gate: vessel lumen must be genuinely dark relative
            # to this image, not merely the darkest thing at a loose threshold
            if darkness > dark_ceiling:
                continue
            fill = area / max(bw * bh, 1)          # solidity within the box
            if fill < 0.45:
                continue

            found.append(Candidate(
                box=[float(x), float(y), float(x + bw), float(y + bh)],
                area_px=int(area), centroid=(float(cent[i][0]), float(cy)),
                darkness=darkness, ellipticity=float(ell), depth_frac=float(depth),
                score=0.0, contrast=float(contrast),
            ))

    found = _dedupe(found)
    for c in found:
        # darker, rounder, reasonably sized and not at the extremes of depth
        c.score = ((1 - c.contrast) * 3.0          # local contrast dominates
                   + c.ellipticity
                   + min(c.area_px / (0.02 * img_area), 2.0) * 0.5
                   - abs(c.depth_frac - 0.35) * 1.2)
    found.sort(key=lambda c: -c.score)
    return found[:max_candidates]


def _dedupe(cands: list[Candidate], iou_thr: float = 0.45,
            contain_thr: float = 0.6) -> list[Candidate]:
    """Drop duplicates by IoU AND by containment.

    IoU alone is not enough: a small artery box sitting entirely inside a large
    vein box has low IoU but is not a separate detection. Without the
    containment test, half the clips proposed an artery nested inside the vein.
    """
    keep: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.area_px):
        dup = False
        for k in keep:
            if _iou(c.box, k.box) >= iou_thr or _containment(c.box, k.box) >= contain_thr:
                dup = True
                break
        if not dup:
            keep.append(c)
    return keep


def _containment(inner, outer) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / a if a > 0 else 0.0


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def assign_roles(cands: list[Candidate], site: str,
                 tie_frac: float = 0.25) -> tuple[list[Candidate], bool]:
    """Label candidates vein/artery. Returns (candidates, ambiguous)."""
    if len(cands) < 2:
        for c in cands:
            c.role = "vein"          # single vessel found: assume the target
        return cands, False

    top = cands[:2]
    ambiguous = False
    kind, direction = SITE_PRIOR.get(site, ("size", "larger"))
    if kind == "depth":
        # Depth, with a per-site sign: the popliteal vein is SUPERFICIAL to
        # its artery, the femoral vein is DEEPER than its own. Measured
        # clip-level accuracy on the expert set: PV 95.2%, CFV 96.0%,
        # FV 95.7%. Ambiguous when the two sit at nearly the same depth.
        top.sort(key=lambda c: c.centroid[1], reverse=(direction == "deeper"))
        dy = abs(top[0].centroid[1] - top[1].centroid[1])
        span = max(abs(top[0].box[3] - top[0].box[1]), 1)
        ambiguous = dy < 0.5 * span
    else:
        # GS only: the great saphenous is itself superficial, so depth against
        # the common femoral artery is 63.4% - near a coin flip - and size is
        # the better cue at 85.4%. Note size works ONLY on relaxed frames;
        # across whole clips it falls to 51%, because the vein is being
        # compressed. Measured on a real GS clip, 4268 vs 3926 px, an 8%
        # difference deciding which vessel gets called the vein.
        top.sort(key=lambda c: -c.area_px)
        ambiguous = (top[0].area_px - top[1].area_px) < tie_frac * top[0].area_px
    top[0].role, top[1].role = "vein", "artery"
    for c in cands[2:]:
        c.role = "unknown"
    return cands, ambiguous


#: Which anatomical prior separates vein from artery, per site, MEASURED on
#: the 1044-file expert set (early frames, clip-level majority vote):
#:
#:     site  depth rule                 size rule
#:     CFV   96.0%  vein DEEPER         74.0%
#:     FV    95.7%  vein DEEPER         60.9%
#:     PV    95.2%  vein SUPERFICIAL    64.3%
#:     GS    63.4%                      85.4%
#:
#: Two corrections to what this module used to do. Depth was applied only to
#: PV, when it is the strongest cue at CFV and FV as well - with the sign
#: FLIPPED, since the femoral vein lies deeper than its artery while the
#: popliteal vein lies superficial to it. And the femoral levels were decided
#: on size, which is 61-74% here and collapses to 51% when measured across
#: whole clips: the vein is being COMPRESSED, so it is only reliably the
#: larger vessel before the probe presses. Depth survives compression because
#: the vessels move together.
#:
#: GS is the exception and keeps the size rule - the great saphenous is itself
#: superficial, so depth against the common femoral artery is nearly a coin
#: flip at 63.4%.
SITE_PRIOR = {
    "CFV": ("depth", "deeper"),
    "FV": ("depth", "deeper"),
    "FV-D": ("depth", "deeper"),
    "PV": ("depth", "superficial"),
    # GS was ("size", "larger") and that was BACKWARDS. See MULTI_VEIN_SITES.
    "GS": ("depth", "superficial"),
}

#: Sites where the field contains more than one VEIN, so picking a vein/artery
#: pair is the wrong formulation - the target vein has to be identified among
#: its neighbours first.
#:
#: GS clips are acquired at the saphenofemoral junction, where the great
#: saphenous, common femoral vein and common femoral artery are all in view.
#: The expert drew more than one vein on 88.3% of GS frames, a median of three
#: vessels per frame.
#:
#: Measured over 415 GS frames across 73 clips where the expert labelled a GSV
#: alongside at least one other vein:
#:
#:     GSV is the SHALLOWEST vein     99.3%     <- the rule
#:     GSV is the SMALLEST vein       88.2%
#:     GSV is the LARGEST vein        11.6%     <- what was coded
#:     GSV is the DEEPEST vein         0.7%
#:
#: The GSV is a median 0.35x the area of the other veins present, which sit a
#: median 1.22 GSV-widths deeper. It is the small superficial vein running in
#: the superficial fascia above the deep system - exactly as anatomy says.
#:
#: The earlier finding that "depth is only 63.4% at GS" compared the vein
#: against the ARTERY, where the GSV/CFA relationship really is ambiguous. The
#: informative comparison is vein against VEIN, and there it is near-perfect.
#: That was the wrong pair, not a weak cue.
MULTI_VEIN_SITES = {"GS"}

#: How the target vein is picked from several candidate veins at a
#: multi-vein site. Both cues agree on 87.5% of frames and at least one is
#: correct on 100%.
MULTI_VEIN_RULE = {"GS": ("shallowest", "smallest")}

#: Which vein to target at a multi-vein site.
#:
#: "gsv"  the vessel the FILENAME names - the great saphenous
#: "deep" the deep-system vein sharing the field - the common femoral
#:
#: This is a clinical choice, not a technical one, and the two differ sharply.
#: Measured on 418 GS frames where the expert drew two or more veins:
#:
#:     the DEEPEST vein is a deep-system vein   99.3%   (415 CFV, 3 GSV)
#:     the LARGEST vein is a deep-system vein   88.8%
#:
#: so "deep" is the most reliable rule found anywhere in this pipeline. Experts
#: annotate the CFV on 447 GS frames against the GSV's 441, so both are equally
#: supported as ground truth.
#:
#: The trade, measured across v8-v11:
#:
#:     target   GS mask IoU   GS named-vein   GS ratio AUC
#:     deep         0.626         n/a             0.6515
#:     gsv          0.246-0.275   54-61%          0.4451-0.5574
#:
#: The saphenous is small, superficial and often partly collapsed, so it
#: segments far worse. It is also superficial: saphenous thrombosis is a
#: different and less serious entity than DVT, and every GS subject in this
#: cohort has three deep-vein clips per site anyway, so the GSV adds no subject
#: coverage.
#:
#: `deep` reproduces what v8 did by accident - but names the result honestly.
#: v8 labelled the common femoral vein "Great Saphenous Vein" because the label
#: came from the filename; here the label follows the vessel.
MULTI_VEIN_TARGET_DEFAULT = "gsv"

#: Label override when the deep vein is targeted at a site whose filename names
#: something else. The label must describe what was tracked.
DEEP_VEIN_LABEL = {"GS": ("Common Femoral Vein", "Common Femoral Artery")}


def _box_depth(box) -> float:
    """Centroid y of an xyxy box. Larger y is deeper (further from probe)."""
    return (float(box[1]) + float(box[3])) / 2.0


def _box_area(box) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * \
        max(float(box[3]) - float(box[1]), 0.0)


def prior_picks_vein(box_a, box_b, site: str):
    """Which of two boxes does the site's anatomical prior call the vein?

    Returns (index, margin, rule) where index is 0 or 1, and `margin` is the
    normalised separation - small margins mean the rule is close to a coin
    flip on this frame and the caller should not trust the swap.

    Returns (None, 0.0, rule) when no prior is known for the site.
    """
    rule = SITE_PRIOR.get(site)
    if rule is None:
        return None, 0.0, None
    kind, direction = rule

    if kind == "depth":
        da, db = _box_depth(box_a), _box_depth(box_b)
        span = max(abs(float(box_a[3]) - float(box_a[1])),
                   abs(float(box_b[3]) - float(box_b[1])), 1.0)
        margin = abs(da - db) / span
        deeper = 0 if da > db else 1
        idx = deeper if direction == "deeper" else (1 - deeper)
    else:
        aa, ab = _box_area(box_a), _box_area(box_b)
        big = 0 if aa > ab else 1
        margin = abs(aa - ab) / max(aa, ab, 1.0)
        idx = big if direction == "larger" else (1 - big)
    return idx, margin, kind


def arbitrate_detections(dets, site: str, min_margin: float = 0.5):
    """Re-decide vein vs artery on the detector's two vessel boxes.

    The detector assigns classes from appearance, and vein and artery look
    nearly identical on a still B-mode frame - which is the whole clinical
    reason compression is the diagnostic test. Measured against expert masks,
    the auto 'vein' track sat on the expert ARTERY on 13% of frames in the
    first run and 27% in the second. The tracker then follows that choice
    faithfully for the entire clip, so one wrong decision at frame 0 costs
    every frame.

    Geometry is the fix, because it is information the detector does not use.
    `SITE_PRIOR` reaches 95-96% at CFV/FV/PV and 85% at GS.

    `min_margin` refuses to act when the two boxes are too close for the rule
    to discriminate - a swap on a coin flip is worse than leaving the
    detector's own call alone, and the sweep below shows that is not
    hypothetical.

    SIMULATED against expert masks on 267 clips of the first run, replaying
    each clip's actual seed pair:

        no arbitration        84.3%
        min_margin 0.15       83.9%   (31 swaps)  <- WORSE than doing nothing
        min_margin 0.30       85.8%   (24 swaps)
        min_margin 0.50       87.3%   (16 swaps)  <- default
        min_margin 0.80       86.5%   (10 swaps)
        min_margin 1.20       85.0%   ( 4 swaps)

    The gain is real but small - about 8 clips in 267. The reason it is not
    the 95% the priors score in isolation: on 41% of clips at least one of the
    detector's two boxes is not on a vessel at all (IoU < 0.3 against both
    expert masks), and geometry between two arbitrary boxes decides nothing.
    Restricted to the 157 clips where both boxes ARE on vessels, assignment
    runs 93.6% -> 94.9%.

    So this fixes the vein/artery MIX-UP, which was never the main defect.
    The main defect is boxes that are not on vessels.

    Mutates and returns `dets`, plus a note: 'swapped', 'agreed', 'ambiguous',
    or 'n/a'.
    """
    vein = max((d for d in dets if d.role == "vein"),
               key=lambda d: d.score, default=None)
    art = max((d for d in dets if d.role == "artery"),
              key=lambda d: d.score, default=None)
    if vein is None or art is None:
        return dets, "n/a"

    idx, margin, _kind = prior_picks_vein(vein.box, art.box, site)
    if idx is None:
        return dets, "n/a"
    if margin < min_margin:
        return dets, "ambiguous"
    if idx == 0:
        return dets, "agreed"

    # the prior says the box the detector called ARTERY is the vein
    vein.cls, art.cls = art.cls, vein.cls
    return dets, "swapped"


def parse_filename_tag(name: str) -> dict:
    """Split a ThrombUS+ filename tag into site, limb and still-image flag.

    Handles every form observed in the expert set:

        anonymized_FR1-0001_0_003_CFV-L      -> CFV, limb L, cine
        anonymized_FR1-0001_0_001_CFVr-L     -> CFV, limb L, STILL image
        anonymized_LT1-0008_0_005_GSVr-L     -> GS,  limb L, still
        anonymized_FR1-0030_0_010_FV-Dr-L    -> FV-D, limb L, still
        anonymized_FR1-0030_0_016_OPT-L      -> site unknown (operator view)

    Returns {"site", "limb", "still", "tag"}; `site` is None when it cannot be
    determined, which is a signal to pass --site rather than to guess.
    """
    stem = str(name).rsplit(".", 1)[0]
    if stem.endswith("_darwin"):
        stem = stem[: -len("_darwin")]
    tag = stem.split("_")[-1]

    limb = None
    body = tag
    for suffix in ("-L", "-R"):
        if tag.endswith(suffix):
            limb = suffix[-1]
            body = tag[: -len(suffix)]
            break

    still = body.endswith("r")
    if still:
        body = body[:-1]

    site = None
    for code in FILENAME_SITE_CODES:
        if body.upper() == code:
            site = "GS" if code == "GSV" else code
            break

    return {"site": site, "limb": limb, "still": still, "tag": tag}


def site_from_filename(name: str) -> str | None:
    """Anatomical site from the filename tag, or None if undeterminable."""
    return parse_filename_tag(name)["site"]


def auto_seed(frames: np.ndarray, site: str | None,
              search_frames: int = 12, max_candidates: int = 3,
              min_contrast_gain: float = 0.35, as_result: bool = False):
    """Pick seed boxes from the clearest frame in the clip.

    Searches WIDELY rather than only the first few frames. Seeding on frame 0
    is a real failure mode: on an FR1 popliteal clip frame 0 was essentially
    featureless and both proposals landed on the near-field artifact rather
    than on any vessel. Scanning the first half of the clip and keeping the
    frame with the best-contrast candidate avoids depending on one arbitrary
    frame being diagnostic.

    `min_contrast_gain` is how much darker than surrounding tissue the best
    candidate must be (1 - contrast). Below it the clip is reported `weak`
    rather than seeded on noise.
    """
    n = min(search_frames, len(frames))
    hi = max(len(frames) // 2, 1)
    idx = np.linspace(0, hi, n).round().astype(int)

    best, best_frame, best_score = [], 0, -1e9
    for i in idx:
        cands = find_candidates(frames[int(i)], max_candidates=max_candidates)
        if not cands:
            continue
        if cands[0].score > best_score:
            best, best_frame, best_score = cands, int(i), cands[0].score

    weak = (not best) or (1.0 - best[0].contrast) < min_contrast_gain
    cands, ambiguous = assign_roles(best, site or "")

    if as_result:
        note = ""
        if weak:
            note = "no convincing vessel found - check this clip by hand"
        elif ambiguous:
            note = "vein/artery assignment near tie - verify before trusting"
        return SeedResult(cands, best_frame, ambiguous, weak, note)
    return cands, best_frame
