"""
Measurements taken off a finished mask sequence.

Three small functions, each guarding against a specific way the pipeline
reported a confident wrong number:

`compression_ratio` - the diagnostic feature. A rejected frame is stored as
area 0, and including it reported perfect compressibility on every clip with a
single gap.

`mask_quality` - the mean intensity inside the mask relative to a surrounding
ring. A vein lumen is anechoic, so a mask that has slipped onto tissue reads
bright. `run_autoannotate --max-contrast` uses it to propose NOTHING on such a
frame rather than propose something wrong.

HISTORICAL NOTE. This module was the MedSAM1 inference path: encoder loading
with position-embedding retargeting for a 6 GB card, box prompting, lumen
refinement, and five hand-written box-propagation strategies (fixed box,
reseed-from-mask, shrink-with-mask, recenter, mask-prompt feedback). Each
failed measurably - reseeding leaked 2090 -> 34426 px, shrinking ratcheted the
other way to a 0.294 ratio with visibly under-segmented masks - and all of it
was replaced by MedSAM2 per-frame segmentation. The code is in
`attic/dvt/step9_medsam_infer.py`.
"""
from __future__ import annotations

import numpy as np


def compression_ratio(areas) -> float | None:
    """min/max area over VALID frames only.

    Frames the tracker rejected are stored as 0. Including them makes the
    minimum 0 and the ratio 0.0 for any clip with a single gap - which on real
    data was every clip, reporting perfect compressibility where the truth was
    0.416. A zero means "no measurement", never "the vein closed completely".
    """
    a = np.asarray(areas, dtype=float)
    v = a[a > 0]
    if v.size < 2:
        return None
    return float(v.min() / v.max())


def mask_quality(mask: np.ndarray, gray: np.ndarray, ring_px: int = 21) -> float:
    """Contrast of the mask interior against the surrounding wall.

    Returns mean(inside) / mean(surrounding ring). Vessel lumen is anechoic, so
    a good mask scores low; a mask that has slipped onto tissue scores near 1.

    The surrounding ring is the reference, NOT the whole image: ultrasound
    frames carry large black borders which drag any global statistic down. On
    an FR1 clip the image median was 31 while the true perivascular tissue was
    ~47, so a global threshold called every mask "too bright".

    Measured on a real compression clip, this correlates -0.63 with mask area:
    as the vein closes and the mask starts covering tissue, contrast rises.
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < 20:
        return float("nan")
    ring = cv2.dilate(m, np.ones((ring_px, ring_px), np.uint8)) - m
    if ring.sum() < 10:
        return float("nan")
    return float(gray[m > 0].mean()) / max(float(gray[ring > 0].mean()), 1e-6)


