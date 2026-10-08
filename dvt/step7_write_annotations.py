"""
Step 7 - Write model proposals as US-DICOMizer-readable sidecar JSON.

The radiologist never leaves US-DICOMizer. We drop a `<stem>_darwin.json` next
to each anonymized DICOM; the app finds it automatically, loads the polygons
into the Annotations tab, and the radiologist corrects and re-exports. No
modification to the application is required.

Verified against US-DICOMizer v5.6 source (MIT licensed,
github.com/thrombusplus/US-DICOMizer). Four constraints come straight from
`import_darwin_annotation`, and violating any of them silently drops data:

1. The payload MUST contain both "annotations" and "item" keys, or
   `detect_annotation_format_payload` classifies it as LabelMe and the Darwin
   branch never runs.
2. A polygon annotation MUST carry `properties: []`. The importer checks
   `if ann.get("properties"): ... continue` -- a non-empty properties list
   makes it a tag annotation and the polygon is skipped entirely.
3. Each path inside `polygon.paths` becomes a SEPARATE polygon with the same
   label. Holes are not supported on import, so emit one path per annotation.
4. Classification values are codes, not words: COMP2/COMP1/COMP0 and
   DVT1/DVT0. ACEP uses the exact property name "ACEP Grading Score" with the
   "Grade " prefix stripped from the value.

Sidecar location follows `package_io_utils.get_preferred_sidecar_paths`:
`<same_dir>/<stem>_darwin.json`.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

# US-DICOMizer canonical value codes
THROMBOSIS_CODE = {"Yes": "DVT1", "No": "DVT0", 1: "DVT1", 0: "DVT0"}
COMPRESSIBILITY_CODE = {"Yes": "COMP2", "Partial": "COMP1", "No": "COMP0",
                        2: "COMP2", 1: "COMP1", 0: "COMP0"}

# Polygon labels offered by the US-DICOMizer annotation panel
VESSEL_LABELS = {
    "CFV": ("Common Femoral Vein # 1", "Common Femoral Artery # 1"),
    "GS": ("Great Saphenous Vein # 3", "Common Femoral Artery # 2"),
    "FV": ("Femoral Vein # 3", "Femoral Artery # 3"),
    "PV": ("Popliteal Vein # 4", "Popliteal Artery # 4"),
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _actor(name: str = "", email: str = ""):
    name, email = str(name or "").strip(), str(email or "").strip()
    if not name or not email:
        return None
    return {"full_name": name, "email": email}


def polygon_annotation(label: str, frame: int, points, annotator=None) -> dict:
    """One polygon on one frame, in the exact shape the importer expects."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError(f"polygon needs >=3 points, got {len(pts)}")
    xs, ys = pts[:, 0], pts[:, 1]
    return {
        "annotators": [annotator] if annotator else [],
        "frames": {
            str(int(frame)): {
                "bounding_box": {
                    "x": float(xs.min()), "y": float(ys.min()),
                    "w": float(xs.max() - xs.min()),
                    "h": float(ys.max() - ys.min()),
                },
                "keyframe": True,
                "polygon": {"paths": [[{"x": float(x), "y": float(y)}
                                       for x, y in pts]]},
            }
        },
        "global_sub_types": {},
        "id": str(uuid.uuid4()),
        "interpolate_algorithm": "linear-1.1",
        "interpolated": True,
        "name": label,
        # MUST stay empty: a non-empty list makes the importer treat this as a
        # tag annotation and discard the polygon.
        "properties": [],
        "ranges": [[int(frame), int(frame) + 1]],
        "reviewers": [],
        "slot_names": ["0"],
        "updated_at": _now(),
    }


def tag_annotation(name: str, value, frame: int = 0, annotator=None) -> dict:
    """A classification tag (Thrombosis, Compressibility, ACEP, ...)."""
    return {
        "annotators": [annotator] if annotator else [],
        "frames": {str(frame): {"keyframe": True, "tag": {}}},
        "id": str(uuid.uuid4()),
        "name": name,
        "properties": [{"frame_index": frame, "name": name, "value": str(value)}],
        "ranges": [[frame, frame + 1]],
        "reviewers": [],
        "slot_names": ["0"],
        "updated_at": _now(),
    }


def build_sidecar(dcm_filename: str, width: int, height: int, frame_count: int,
                  polygons: list[dict] | None = None,
                  acep_by_frame: dict[int, int] | None = None,
                  thrombosis=None, compressibility=None,
                  patient_id: str = "", clinician_name: str = "",
                  clinician_email: str = "", reviewed: bool = False) -> dict:
    """Assemble a complete Darwin 2.0 sidecar payload.

    `polygons` is a list of {"label", "frame", "points"} dicts, typically the
    model's proposals. `reviewed` stays False so the radiologist's own tick is
    what marks a file as validated -- never pre-tick it on the model's behalf.
    """
    annotator = _actor(clinician_name, clinician_email)
    anns = []

    for p in (polygons or []):
        anns.append(polygon_annotation(p["label"], p["frame"], p["points"], annotator))

    for frame, grade in (acep_by_frame or {}).items():
        anns.append(tag_annotation("ACEP Grading Score",
                                   str(grade).replace("Grade ", "").strip(),
                                   int(frame), annotator))

    if thrombosis is not None:
        anns.append(tag_annotation("Thrombosis",
                                   THROMBOSIS_CODE.get(thrombosis, thrombosis),
                                   0, annotator))
    if compressibility is not None:
        anns.append(tag_annotation("Compressibility",
                                   COMPRESSIBILITY_CODE.get(compressibility, compressibility),
                                   0, annotator))

    anns.append(tag_annotation("_reviewed", "true" if reviewed else "false", 0, annotator))
    anns.append(tag_annotation("Protocol Deviation", "false", 0, annotator))

    return {
        "version": "2.0",
        "schema_ref": ("https://darwin-public.s3.eu-west-1.amazonaws.com/"
                       "darwin_json/2.0/schema.json"),
        "item": {
            "name": dcm_filename,
            "path": "/",
            "source_info": {
                "item_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "dataset": {"name": "", "slug": "", "dataset_management_url": ""},
                "team": {"name": "", "slug": ""},
                "workview_url": "",
            },
            "slots": [{
                "type": "dicom", "slot_name": "0",
                "width": int(width), "height": int(height),
                "fps": None, "thumbnail_url": "",
                "source_files": [{"file_name": dcm_filename, "url": ""}],
                "frame_count": int(frame_count), "frame_urls": [],
                "metadata": {"handler": None, "shape": None,
                             "colorspace": "RGB", "primary_plane": "AXIAL"},
            }],
        },
        "annotations": anns,
        "properties": [],
    }


def sidecar_path(dcm_path: Path) -> Path:
    """Where US-DICOMizer looks for it: <stem>_darwin.json beside the DICOM."""
    dcm_path = Path(dcm_path)
    return dcm_path.with_name(f"{dcm_path.stem}_darwin.json")


def write_sidecar(dcm_path: Path, payload: dict) -> Path:
    out = sidecar_path(dcm_path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return out


def mask_to_polygon(mask, max_points: int = 40, min_area: int = 30):
    """Largest external contour of a binary mask, simplified for hand editing.

    The manual instructs annotators to draw with as few points as possible, so
    proposals are simplified: a 400-vertex machine contour is far more tedious
    for a radiologist to correct than to redraw from scratch, which would
    defeat the point of proposing it.
    """
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    peri = cv2.arcLength(c, True)
    for eps in (0.002, 0.004, 0.008, 0.015, 0.03):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) <= max_points:
            break
    pts = approx.reshape(-1, 2).astype(float)
    return pts if len(pts) >= 3 else None
