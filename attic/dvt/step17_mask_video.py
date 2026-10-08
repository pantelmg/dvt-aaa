r"""
Step 17 - render sidecar polygons into mask videos for V-JEPA.

WHY MASKS AND NOT PIXELS
------------------------
The last V-JEPA attempt on this cohort learned the hospital: features predicted
centre identity at AUC ~1.00 and collapsed on held-out centres. That is not a
tuning failure. Centre prevalence ALONE gives AUC 0.709 here, each site used a
different scanner, and FR1 plus GR1 hold 93% of the annotated frames - so a
model with access to speckle, gain and depth-compensation will find the scanner
before it finds the thrombus, and a within-centre score cannot tell the two
apart.

A binary mask has no speckle, no gain curve, no scanner signature. The leakage
is not mitigated, it is ABSENT - the information is not in the input.

What survives is everything the compression ratio throws away: how fast the
vein collapses and whether it rebounds, deformation shape through compression,
vein-to-artery size ratio over time, asymmetry, and the artery as an internal
control for probe pressure. The ratio compresses all of that into min/max,
which is why the expert-mask ceiling is only 0.698 vessel / 0.783 subject.

WHAT IS LOST
------------
Texture, and with it clot. Clot presence has the higher expert ceiling of the
two features (subject AUC 0.828 against the ratio's 0.783), and thrombus is
MADE of echotexture. A mask can carry a clot polygon's shape but not whether a
lumen looks thrombosed. `--channels 3` adds a normalised pixel crop as a third
channel for exactly that reason - run masks-only FIRST, so that whatever the
crop buys is measured against a leak-free baseline rather than assumed.

GEOMETRY LEAKS TOO, IF YOU LET IT
---------------------------------
Masks are not automatically centre-free. Pixel area varies 5.6x across centres,
frame dimensions run from 273x516 (GR1) to 1300x745 (LT1), and vessel position
on screen follows site protocol. Rendering a mask at raw scale would let the
model read "this frame is 1300 wide" as a centre label.

So the canvas is normalised to the VEIN's own geometry: centred on the vein's
median centroid and scaled by its own maximum extent, not by the frame. After
that a GR1 clip and an LT1 clip of the same anatomy look the same.
`scripts/check_centre_leakage.py` verifies this rather than assuming it.

FRAMES WITH NO POLYGON
----------------------
Left EMPTY, and recorded in a per-frame validity mask. 10.7% of expert
keyframes have no polygon in v8. An empty channel says "no vessel here", which
is a different lie from a wrong mask but still a lie - the validity mask lets
the probe exclude those frames rather than train on them.

Collided frames (vein/artery containment >= 0.3) are marked invalid too, on the
same reasoning: on collided frames the vein polygon is on the vein only 48.3%
of the time, against 78.0% on clean frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Output canvas. Small on purpose - a mask carries shape, not detail, and
#: V-JEPA resizes to its own input size anyway.
CANVAS = 128

#: How much of the canvas the vein's maximum extent should span. Leaves room
#: for the artery, which sits a median 1.13 vein-widths away.
VEIN_FRACTION = 0.32

#: Fraction of the canvas the VEIN+ARTERY union should span when scaling by the
#: pair. Larger than VEIN_FRACTION because the union is bigger - the two sit a
#: median 1.13 vein-widths apart, so the pair spans roughly twice a vein.
PAIR_FRACTION = 0.58


def _raster(paths, w, h):
    from PIL import Image, ImageDraw

    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for i, path in enumerate(paths):
        pts = [(float(q["x"]), float(q["y"])) for q in path]
        if len(pts) >= 3:
            d.polygon(pts, fill=0 if i else 1)   # extra paths are holes
    return np.array(img, dtype=bool)


def _role(label: str) -> str:
    low = (label or "").lower()
    if "clot" in low or "thromb" in low:
        return "clot"
    if "vein" in low:
        return "vein"
    if "artery" in low:
        return "artery"
    head = (label or "").split()[0].upper() if (label or "").split() else ""
    if head in {"CFV", "FV", "PV", "GSV", "SFV", "DFV"}:
        return "vein"
    if head in {"CFA", "FA", "PA", "DFA", "SFA"}:
        return "artery"
    return "other"


def load_sidecar(path: Path):
    """-> (frames{idx: {role: mask}}, w, h, n_frames)."""
    doc = json.loads(Path(path).read_text())
    slot = (doc.get("item", {}).get("slots") or [{}])[0]
    w, h = slot.get("width"), slot.get("height")
    n = slot.get("frame_count") or 0
    if not w or not h:
        return {}, 0, 0, 0
    out = {}
    for ann in doc.get("annotations", []):
        r = _role(ann.get("name", ""))
        if r not in ("vein", "artery", "clot"):
            continue
        for k, fr in (ann.get("frames") or {}).items():
            if not str(k).lstrip("-").isdigit():
                continue
            poly = fr.get("polygon")
            if not poly or not poly.get("paths"):
                continue
            i = int(k)
            m = _raster(poly["paths"], w, h)
            slot_d = out.setdefault(i, {})
            slot_d[r] = (slot_d[r] | m) if r in slot_d else m
            n = max(n, i + 1)
    return out, w, h, n


def _canvas_transform(frames, w, h, scale_by="pair"):
    """Centre and scale the canvas by the vessels' own geometry, not the frame's.

    Returns (cx, cy, scale) mapping source pixels to the canvas.

    WHY "pair" AND NOT "vein"
    -------------------------
    Scaling by the VEIN alone equalises vein size across centres but leaves the
    ARTERY free, so the vein-to-artery size ratio survives - and that ratio
    varies with probe and site protocol. `check_centre_leakage.py` on the v12
    sidecars found exactly that:

        centre classifier, subject-grouped   0.572 against a 0.474 floor
        lift over majority                   +0.099

        top centre-predictive features
          art_area_mean   +0.0743      <- four times the next
          vein_over_art   +0.0180
          vein_cy         +0.0161

    `vein_area_mean` does NOT appear, which confirms the vein was equalised and
    the artery was not. Scaling by the union of both vessels removes the free
    parameter: whatever the probe, the pair fills the same fraction of canvas.

    The cost is that vein size is no longer perfectly equalised either - it now
    varies with the ratio. That is the point: the ratio has to live somewhere,
    and splitting it between the two channels leaks less than parking all of it
    in one.

    `scale_by="vein"` restores the old behaviour for comparison.
    """
    cxs, cys, sides = [], [], []
    for r in frames.values():
        parts = []
        if scale_by == "pair":
            for role in ("vein", "artery"):
                m = r.get(role)
                if m is not None and m.sum() >= 20:
                    parts.append(m)
        else:
            m = r.get("vein")
            if m is not None and m.sum() >= 20:
                parts.append(m)
        if not parts:
            continue
        u = np.logical_or.reduce(parts)
        ys, xs = np.nonzero(u)
        cxs.append(xs.mean()); cys.append(ys.mean())
        sides.append(max(xs.max() - xs.min(), ys.max() - ys.min()))
    if not sides:
        return w / 2, h / 2, CANVAS / max(w, h)
    # median centre (robust to a frame where a track jumped), max extent
    frac = PAIR_FRACTION if scale_by == "pair" else VEIN_FRACTION
    return (float(np.median(cxs)), float(np.median(cys)),
            CANVAS * frac / max(float(np.max(sides)), 1.0))


def render_clip(sidecar: Path, num_frames: int = 32, channels: int = 2,
                dicom: Path | None = None, collided_thr: float = 0.30,
                scale_by: str = "pair"):
    """-> (video (T,C,CANVAS,CANVAS) float32, valid (T,) bool, info dict).

    Channel 0 vein, channel 1 artery, channel 2 (optional) a normalised pixel
    crop on the same canvas.
    """
    import cv2

    frames, w, h, n = load_sidecar(sidecar)
    if not frames or n < 2:
        return None, None, {"reason": "no polygons"}

    cx, cy, scale = _canvas_transform(frames, w, h, scale_by)
    M = np.array([[scale, 0, CANVAS / 2 - scale * cx],
                  [0, scale, CANVAS / 2 - scale * cy]], dtype=np.float32)

    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    vid = np.zeros((num_frames, channels, CANVAS, CANVAS), np.float32)
    valid = np.zeros(num_frames, bool)

    pixels = None
    if channels >= 3 and dicom is not None:
        from dvt import step8_compression_phase as S8
        try:
            pixels = S8.load_clip_frames(dicom, max_frames=0, downsample=1)
        except Exception:
            pixels = None

    n_collided = n_empty = 0
    for t, i in enumerate(idx):
        r = frames.get(int(i), {})
        v = r.get("vein"); a = r.get("artery")
        if v is None or v.sum() < 20:
            n_empty += 1
            continue
        if a is not None and a.sum() >= 20:
            inter = int((v & a).sum())
            if inter / min(int(v.sum()), int(a.sum())) >= collided_thr:
                n_collided += 1
                continue          # the vein is on the vein only 48.3% here
        vid[t, 0] = cv2.warpAffine(v.astype(np.float32), M, (CANVAS, CANVAS))
        if channels >= 2 and a is not None:
            vid[t, 1] = cv2.warpAffine(a.astype(np.float32), M,
                                       (CANVAS, CANVAS))
        if pixels is not None and i < len(pixels):
            p = cv2.warpAffine(pixels[int(i)].astype(np.float32), M,
                               (CANVAS, CANVAS))
            p = (p - p.mean()) / (p.std() + 1e-6)   # strip absolute gain
            vid[t, 2] = p
        valid[t] = True

    if valid.sum() < 4:
        return None, None, {"reason": "too few valid frames",
                            "empty": n_empty, "collided": n_collided}
    return vid, valid, {"frames": int(n), "valid": int(valid.sum()),
                        "empty": n_empty, "collided": n_collided,
                        "scale": float(scale)}
