"""
Detect the compression phase of a clip, with no segmentation and no labels.

STATUS: TESTED ON 40 REAL CLIPS AND NOT RELIABLE ENOUGH TO DEPEND ON.
Kept because the diagnostics are reusable and the failure is instructive, but
do not build the annotation workflow on it. See `KNOWN LIMITATIONS` below and
use `phase_from_area_curve()` once segmentation exists.

This is the biggest single reduction in annotation effort available. A clip has
21-301 frames, but compressibility is decided by comparing exactly two of them:
one with the vein relaxed and one at peak probe pressure. Asking a radiologist
to find those two frames in a 300-frame cine loop, forty times over, is most of
the cost. Finding them automatically turns "annotate this video" into "check
these two frames".

Signal used: as the probe presses, tissue is displaced toward the transducer,
so image content translates upward. Frame-to-frame vertical translation is
recovered by phase correlation and accumulated. The result is a probe-pressure
proxy that needs no vessel model, and works identically on every scanner
because it measures bulk tissue motion rather than image intensity.

Vertical displacement is reported in millimetres using the DICOM ultrasound
calibration, so values are comparable across the five centres despite their
pixel spacing differing by a factor of ~1.8 in this cohort.

KNOWN LIMITATIONS (measured, not theoretical)
---------------------------------------------
1. Bulk motion is a weak proxy. Median travel was 1.36 mm (translation) and
   1.06 mm (strain) across 40 clips whose veins visibly collapse. Compression
   squashes tissue locally around the vessel; the global image barely moves.
2. Visual check on 6 clips: frame pairs looked correct on 2 (GR1 popliteal),
   near-identical on 3. Not good enough to hand a radiologist unreviewed.
3. Cumulative frame-to-frame phase correlation has unbounded drift. One GR1
   clip returned 856 mm of "strain" because a few low-texture frame pairs
   injected spurious steps that never wash out. Registering every frame to a
   single reference instead of chaining would bound this, but does not fix
   limitation 1.
4. LT1 clips appear time-reversed relative to other centres under the
   translation method (relaxed near the end, peak at frame 0). Whether that is
   a real protocol difference or an artefact of limitation 1 is unresolved.

RECOMMENDED PATH: derive the phase from the vein area curve produced by
segmentation instead. Peak compression is by definition the area minimum, so
once MedSAM runs there is no separate detection problem to solve. Phase
detection was an attempt to avoid segmentation; segmentation is needed anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CompressionPhase:
    n_frames: int
    relaxed_frame: int          # least compressed
    peak_frame: int             # most compressed
    displacement_px: float      # peak-to-relaxed vertical travel
    displacement_mm: float | None
    monotonic_fraction: float   # how cleanly the ramp behaves
    confidence: str             # ok | weak | suspect
    curve: list[float]

    def to_dict(self):
        d = asdict(self)
        d.pop("curve")
        return d


def vertical_drift(frames: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Cumulative vertical translation per frame, in pixels.

    Positive values mean content has moved DOWN, negative means UP (toward the
    probe, i.e. compression). Phase correlation is used rather than optical
    flow because it is global, cheap, and robust to ultrasound speckle.
    """
    import cv2

    prev = np.float64(frames[0])
    cum = 0.0
    out = [0.0]
    for i in range(1, len(frames)):
        cur = np.float64(frames[i])
        (_dx, dy), _resp = cv2.phaseCorrelate(prev, cur)
        cum += dy
        out.append(cum)
        prev = cur
    curve = np.asarray(out, dtype=float)
    if smooth > 1 and len(curve) >= smooth:
        k = np.ones(smooth) / smooth
        curve = np.convolve(curve, k, mode="same")
    return curve


def axial_strain(frames: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Depth-dependent tissue convergence - a direct compression measure.

    Bulk vertical translation turned out to be a weak proxy: probe pressure
    squashes tissue rather than sliding it, so a clip can compress hard while
    the image barely translates. Measured on this cohort, translation gave
    0.85-2.1 mm on clips whose veins visibly collapse.

    Strain measures the right thing. The tissue column between the transducer
    and deep structures shortens under pressure, so superficial and deep bands
    converge. Tracking each band separately and taking the DIFFERENCE cancels
    whole-image drift (probe sliding, patient motion) and leaves compression.

    Returns convergence in pixels: positive means the bands have moved closer,
    i.e. tissue is compressed.
    """
    import cv2

    h = frames.shape[1]
    top = frames[:, : h // 3, :]
    bot = frames[:, 2 * h // 3:, :]

    def drift(stack):
        prev = np.float64(stack[0])
        cum, out = 0.0, [0.0]
        for i in range(1, len(stack)):
            cur = np.float64(stack[i])
            (_dx, dy), _r = cv2.phaseCorrelate(prev, cur)
            cum += dy
            out.append(cum)
            prev = cur
        return np.asarray(out)

    conv = drift(top) - drift(bot)
    if smooth > 1 and len(conv) >= smooth:
        conv = np.convolve(conv, np.ones(smooth) / smooth, mode="same")
    return conv


def detect(frames: np.ndarray, px_spacing_mm: float | None = None,
           smooth: int = 5, method: str = "strain") -> CompressionPhase:
    """Locate the relaxed and peak-compression frames of a clip."""
    if method == "strain":
        curve = axial_strain(frames, smooth=smooth)
        relaxed = int(np.argmin(curve))   # least convergence = relaxed
        peak = int(np.argmax(curve))      # most convergence  = compressed
        travel = float(curve[peak] - curve[relaxed])
    else:
        curve = vertical_drift(frames, smooth=smooth)
        relaxed = int(np.argmax(curve))
        peak = int(np.argmin(curve))
        travel = float(curve[relaxed] - curve[peak])

    # How much of the path from relaxed to peak moves in a single direction.
    # A clean compression is a monotonic ramp; a noisy or absent one is not.
    lo, hi = sorted((relaxed, peak))
    seg = curve[lo:hi + 1]
    if len(seg) > 2:
        d = np.diff(seg)
        mono = float(max((d > 0).mean(), (d < 0).mean()))
    else:
        mono = 0.0

    mm = travel * px_spacing_mm if px_spacing_mm else None
    if travel < 3 or mono < 0.6:
        conf = "suspect"
    elif travel < 8 or mono < 0.75:
        conf = "weak"
    else:
        conf = "ok"

    return CompressionPhase(
        n_frames=len(frames), relaxed_frame=relaxed, peak_frame=peak,
        displacement_px=round(travel, 2),
        displacement_mm=round(mm, 2) if mm is not None else None,
        monotonic_fraction=round(mono, 3), confidence=conf,
        curve=[round(float(v), 3) for v in curve],
    )


def suggest_annotation_frames(phase: CompressionPhase, n_extra: int = 0) -> list[int]:
    """Frames to put in front of the radiologist.

    Two by default: relaxed and peak. `n_extra` adds evenly spaced frames
    between them, useful when a full area-vs-time curve is wanted rather than
    just a ratio.
    """
    a, b = sorted((phase.relaxed_frame, phase.peak_frame))
    if n_extra <= 0 or b - a < 2:
        return sorted({a, b})
    mids = np.linspace(a, b, n_extra + 2)[1:-1].round().astype(int).tolist()
    return sorted({a, b, *mids})


def phase_from_area_curve(areas: dict[int, float]) -> tuple[int, int]:
    """The recommended replacement: read the phase off the vein area curve.

    `areas` maps frame index to lumen area for ONE vessel. Peak compression is
    the area minimum by definition, so this needs no motion model, no
    thresholds, and no per-centre calibration - and it is correct even when the
    probe barely moves, which is exactly where the motion-based detectors fail.
    """
    if len(areas) < 2:
        raise ValueError("need at least two annotated frames")
    relaxed = max(areas, key=areas.get)
    peak = min(areas, key=areas.get)
    return int(relaxed), int(peak)


def frame_indices(dcm_path, max_frames: int | None = None) -> list[int]:
    """The SOURCE frame numbers that `load_clip_frames` will return.

    Needed because sampling is `linspace`, not a uniform stride. Reconstructing
    the mapping as `i * round(total / n_sampled)` is wrong: for a 158-frame clip
    sampled to 8, linspace gives [0,22,45,67,90,112,135,157] while the stride
    formula gives [0,20,40,...,140] - up to 17 frames adrift. Annotations built
    on the stride assumption land on the wrong frames.
    """
    import pydicom
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if max_frames and n > max_frames:
        return np.linspace(0, n - 1, max_frames).round().astype(int).tolist()
    return list(range(n))


def load_clip_frames(dcm_path, max_frames: int | None = None,
                    downsample: int = 1) -> np.ndarray:
    """Decode a DICOM to a greyscale (T,H,W) array, one frame at a time.

    `ds.pixel_array` materialises the whole clip: a 301-frame 430x310 RGB study
    is 120 MB decoded and peaked at 1.1 GB RSS in practice, which is enough to
    get the process OOM-killed on a modest machine. Streaming with
    `iter_pixels(indices=...)` decodes only the frames requested and peaked at
    65 MB on the same file - a 17x reduction, and the difference between the
    batch completing and dying halfway.
    """
    import pydicom
    from pydicom.pixels import iter_pixels
    from . import step1_extract_frames as S1

    indices = frame_indices(dcm_path, max_frames)

    frames = []
    for arr in iter_pixels(str(dcm_path), indices=indices):
        g = S1._to_luma(arr)
        if downsample > 1:
            g = g[::downsample, ::downsample]
        frames.append(np.ascontiguousarray(g))

    return np.stack(S1._scale_clip(frames))
