"""
Decode a compression-ultrasound DICOM into a greyscale clip.

This is the pipeline's only entry point for pixel data. Everything downstream -
detection, segmentation, the compression ratio - reads the (T,H,W) uint8 array
this module returns, so the two decisions made here (which frames, and what
normalisation) propagate into every measurement.

WHICH FRAMES. `frame_indices` is the authoritative mapping from processed index
back to DICOM frame number, and annotations must be written against it rather
than against a reconstructed stride. See its docstring for the 17-frame error
the stride assumption produces.

NORMALISATION. `_scale_clip` rescales using CLIP-WIDE min/max, never per-frame.
Per-frame rescaling is the subtle bug for this modality: frame maxima vary ~29%
over a cine loop, so identical tissue would appear at different brightness
across the sequence - spurious temporal variation injected into exactly the
signal a compressibility model reads.

HISTORICAL NOTE. This module began as a compression-PHASE detector: find the
relaxed and peak-pressure frames from bulk tissue motion (phase correlation,
accumulated vertical drift) so a radiologist would only review two frames per
clip. Measured on 40 clips whose veins visibly collapse, median travel was
1.36 mm (translation) and 1.06 mm (strain) - compression squashes tissue
locally around the vessel while the global image barely moves - and cumulative
frame-to-frame correlation drifted without bound (one clip reported 856 mm).
The approach was abandoned: peak compression is by definition the vein's area
minimum, so once segmentation exists the phase is free. The code is in
`attic/dvt/step8_compression_phase.py`.
"""
from __future__ import annotations

import numpy as np


def _to_luma(arr: np.ndarray) -> np.ndarray:
    """Collapse one frame to a single channel.

    B-mode ultrasound is greyscale but is routinely stored with
    SamplesPerPixel=3 (RGB, or YBR which pydicom converts to RGB on read). If
    the three channels are identical it is plain greyscale and any channel will
    do. If they differ, a colour overlay is present (Doppler, annotation
    burn-in) and we take ITU-R BT.601 luma rather than one arbitrary channel.
    """
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] != 3:
        return arr[..., 0]
    if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        return arr[..., 0]
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]).round().astype(arr.dtype)


def _scale_clip(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Convert a whole clip to uint8 using CLIP-WIDE statistics.

    Normalizing each frame independently is a subtle but serious bug for
    compression ultrasound: per-frame max varies substantially over a cine loop
    (measured at ~29% on a Philips EPIQ clip), so per-frame min-max rescaling
    applies a different gain to every frame. Identical tissue then appears at
    different brightness across the sequence, injecting exactly the kind of
    spurious temporal variation a compressibility model is trying to read.

    8-bit data is passed through untouched, since it is already in range.
    """
    if all(f.dtype == np.uint8 for f in frames):
        return frames
    stack_min = min(float(f.min()) for f in frames)
    stack_max = max(float(f.max()) for f in frames)
    if stack_max <= stack_min:
        return [np.zeros(f.shape, dtype=np.uint8) for f in frames]
    rng = stack_max - stack_min
    return [(((f.astype(np.float32) - stack_min) / rng) * 255.0)
            .round().astype(np.uint8) for f in frames]


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


def frame_count(dcm_path) -> int:
    """How many frames the DICOM actually holds. Header only, no pixel decode.

    The authority on whether a file is a cine or a still. The FILENAME is not:
    measured over the 1044 annotated clips, the US-DICOMizer `r` suffix
    disagrees with the header in both directions - 18 clips have one frame
    without the suffix, and 12 carry the suffix while holding 55-213 real
    frames. Compressibility is undefined on one frame and measurable on many,
    so the caller should gate on this rather than on the tag.
    """
    import pydicom
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    return int(getattr(ds, "NumberOfFrames", 1) or 1)


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
    import pydicom  # noqa: F401  (ensures the pixel handlers are registered)
    from pydicom.pixels import iter_pixels

    indices = frame_indices(dcm_path, max_frames)

    frames = []
    for arr in iter_pixels(str(dcm_path), indices=indices):
        g = _to_luma(arr)
        if downsample > 1:
            g = g[::downsample, ::downsample]
        frames.append(np.ascontiguousarray(g))

    return np.stack(_scale_clip(frames))
