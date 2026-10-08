"""
Anatomical knowledge: what the filename says, and which vessel is the vein.

This module is pure prior knowledge - no model, no pixels. It answers two
questions that the detector cannot:

1. WHERE ARE WE? `parse_filename_tag` reads the US-DICOMizer filename template
   (`anonymized_<PATIENT>_<ORDER>_<SEQ>_<TAG>.dcm`) and recovers the anatomical
   site (CFV / GS / FV / FV-D / PV), the limb, and whether the file is a
   compression cine or a random STILL image. Stills are single-frame, so a
   compression ratio is undefined on them and `run_autoannotate` skips them.

2. WHICH OF THE PAIR IS THE VEIN? `prior_picks_vein` applies the site prior:

   * Popliteal (PV) and saphenous (GS): scanned from the popliteal fossa or the
     groin surface, the target vein lies SUPERFICIAL to the artery - closer to
     the transducer, i.e. smaller y. Depth is reliable and does not depend on
     probe rotation.
   * Femoral levels (CFV, FV, FV-D): the vein is DEEPER than the artery at the
     scanning angles used in this protocol.

   Measured against expert masks: the priors are 95-96% accurate at CFV/FV/PV
   and 85% at GS, against a detector that put the vein label on the artery
   13-27% of the time. The prior wins, which is why `select_vessel_pair`
   consults it rather than trusting the class head. Medial/lateral is NOT used:
   it depends on probe orientation, which the DICOM does not record.

MULTI-VEIN SITES. At GS the field holds three or four vessels and the great
saphenous ranks only 2nd among vein proposals by detector score (top-1 just
19.4% of the time). `MULTI_VEIN_SITES` and `MULTI_VEIN_RULE` mark that site as
ambiguous. With `--multi-vein-target deep` the pipeline deliberately tracks the
COMMON FEMORAL vein in GS clips instead, and `DEEP_VEIN_LABEL` renames the
output accordingly - v8 tracked the right vessel and called it "Great Saphenous
Vein", which was correct masks with a wrong name.

HISTORICAL NOTE. This module also held a heuristic seeder: threshold the
anechoic lumen, take connected components, filter by border contact, elongation
and size, then assign roles by the priors above. It reached IoU>0.5 on 6% of
clips and was replaced by the trained detector (74% vein detection at IoU>=0.5).
Seeding was the measured bottleneck of the whole project, not segmentation. The
code is in `attic/dvt/step10_auto_seed.py`.
"""
from __future__ import annotations


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


