"""
Step 1 - Unpack US-DICOMizer exports and extract frame sequences.

US-DICOMizer exports a .zip containing anonymized DICOM files plus one
Darwin-V7 .json per DICOM. This module unpacks that, decodes each multi-frame
DICOM into a PNG sequence, and writes a manifest.

Filenames follow the US-DICOMizer template:
    anonymized_PATIENTID_X_XXXX_tag.dcm
so patient id, acquisition order and anatomical tag are all recoverable.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from . import config as C

# anonymized_<PATIENTID>_<X>_<XXXX>_<tag>.dcm
FNAME_RE = re.compile(
    r"^anonymized_(?P<patient>.+?)_(?P<order>\d+)_(?P<seq>\d+)_(?P<tag>.+)$"
)


@dataclass
class ClipMeta:
    clip_id: str
    patient_id: str
    order: int
    tag: str
    n_frames: int
    height: int
    width: int
    dicom_path: str
    frame_dir: str
    json_path: str | None
    # calibration, needed to convert lumen area from px^2 to mm^2
    px_spacing_mm: tuple[float, float] | None = None
    cine_rate: float | None = None
    photometric: str | None = None
    lossy: bool = False


def parse_filename(stem: str) -> dict:
    """Recover patient / order / anatomical tag from the US-DICOMizer filename."""
    m = FNAME_RE.match(stem)
    if not m:
        # Fall back gracefully rather than dropping the clip silently.
        return {"patient": "unknown", "order": -1, "tag": "untagged"}
    d = m.groupdict()
    return {"patient": d["patient"], "order": int(d["order"]), "tag": d["tag"]}


def unpack_zips(zip_dir: Path = C.RAW_ZIP_DIR,
                dicom_dir: Path = C.DICOM_DIR,
                json_dir: Path = C.JSON_DIR) -> int:
    """Extract .dcm files and .json annotation files from US-DICOMizer ZIPs."""
    dicom_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for zp in sorted(zip_dir.glob("*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or member.endswith("/"):
                    continue
                low = name.lower()
                if low.endswith(".dcm"):
                    target = dicom_dir / name
                elif low.endswith(".json"):
                    target = json_dir / name
                else:
                    continue
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                n += 1
    return n


# Transfer syntaxes that discard information. pydicom UIDs expose no
# `is_lossy` attribute, so a getattr-with-default silently reports every file
# as lossless. Detect explicitly instead.
LOSSY_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2.4.50",   # JPEG Baseline (Process 1)
    "1.2.840.10008.1.2.4.51",   # JPEG Extended (Process 2 & 4)
    "1.2.840.10008.1.2.4.81",   # JPEG-LS Lossy
    "1.2.840.10008.1.2.4.91",   # JPEG 2000 (lossy allowed)
    "1.2.840.10008.1.2.4.101",  # MPEG2 Main
    "1.2.840.10008.1.2.4.102",  # MPEG-4 AVC/H.264
    "1.2.840.10008.1.2.4.107",  # HEVC/H.265
}


def _is_lossy(ds) -> bool:
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is not None and str(ts) in LOSSY_TRANSFER_SYNTAXES:
        return True
    # (0028,2110) '01' means the pixel data has been lossily compressed.
    return str(getattr(ds, "LossyImageCompression", "")) == "01"


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


def _frame_rate(ds) -> float | None:
    """Frames per second, falling back to FrameTime.

    CineRate is absent on some scanners in this cohort (ESAOTE 6150, Philips
    iU22). FrameTime is in milliseconds, so fps = 1000 / FrameTime.
    """
    cr = getattr(ds, "CineRate", None)
    if cr:
        return float(cr)
    ft = getattr(ds, "FrameTime", None)
    if ft:
        try:
            ft = float(ft)
            if ft > 0:
                return round(1000.0 / ft, 3)
        except (TypeError, ValueError):
            pass
    return None


def _calibration(ds) -> tuple[float, float] | None:
    """Physical pixel spacing in mm, from SequenceOfUltrasoundRegions.

    PhysicalUnitsXDirection == 3 means centimetres.
    """
    seq = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not seq:
        return None
    r = seq[0]
    dx, dy = getattr(r, "PhysicalDeltaX", None), getattr(r, "PhysicalDeltaY", None)
    if dx is None or dy is None:
        return None
    ux = getattr(r, "PhysicalUnitsXDirection", 3)
    factor = 10.0 if ux == 3 else 1.0  # cm -> mm
    return (float(dx) * factor, float(dy) * factor)


def extract_clip(dcm_path: Path, frame_root: Path = C.FRAME_DIR) -> ClipMeta:
    """Decode one (possibly multi-frame) DICOM into a PNG sequence."""
    ds = pydicom.dcmread(str(dcm_path))
    px = ds.pixel_array  # (H,W), (H,W,3), (T,H,W) or (T,H,W,3)

    if px.ndim == 2:
        raw = [px]
    elif px.ndim == 3 and px.shape[-1] == 3:
        raw = [px]                          # single colour frame
    else:
        raw = [px[i] for i in range(px.shape[0])]

    frames = _scale_clip([_to_luma(f) for f in raw])

    stem = dcm_path.stem
    meta = parse_filename(stem)
    out_dir = frame_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    h = w = 0
    for i, g in enumerate(frames):
        h, w = g.shape[:2]
        Image.fromarray(g).save(out_dir / f"frame_{i:04d}.png")

    json_candidate = C.JSON_DIR / f"{stem}.json"
    return ClipMeta(
        clip_id=stem,
        patient_id=meta["patient"],
        order=meta["order"],
        tag=meta["tag"],
        n_frames=len(frames),
        height=h,
        width=w,
        dicom_path=str(dcm_path),
        frame_dir=str(out_dir),
        json_path=str(json_candidate) if json_candidate.exists() else None,
        px_spacing_mm=_calibration(ds),
        cine_rate=_frame_rate(ds),
        photometric=str(getattr(ds, "PhotometricInterpretation", "")) or None,
        lossy=_is_lossy(ds),
    )


def build_manifest(dicom_dir: Path = C.DICOM_DIR,
                   out_path: Path | None = None) -> list[ClipMeta]:
    out_path = out_path or (C.INDEX_DIR / "clips.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metas = [extract_clip(p) for p in sorted(dicom_dir.glob("*.dcm"))]
    out_path.write_text(json.dumps([asdict(m) for m in metas], indent=2))
    return metas


if __name__ == "__main__":
    C.ensure_dirs()
    n = unpack_zips()
    print(f"extracted {n} files from ZIPs")
    metas = build_manifest()
    missing = [m.clip_id for m in metas if m.json_path is None]
    print(f"{len(metas)} clips, {sum(m.n_frames for m in metas)} frames")
    if missing:
        print(f"WARNING: {len(missing)} clips have no annotation JSON")
