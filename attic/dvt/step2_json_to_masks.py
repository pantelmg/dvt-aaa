"""
Step 2 - Convert US-DICOMizer annotation JSON into training-ready arrays.

Produces, per annotated frame:
  * a uint8 semantic mask (0=bg, 1=vein, 2=artery, 3=clot)
  * one bounding box per instance, for MedSAM box prompts
and, per clip:
  * thrombosis / compressibility / ACEP grades / anatomical tag

The exporter's JSON layout is Darwin-V7-like but the exact nesting varies by
version, so parsing is defensive: we search for the fields rather than assume
one fixed path, and we record what we could not interpret instead of silently
dropping it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import config as C


@dataclass
class Instance:
    """One polygon on one frame.

    `paths` follows the Darwin polygon spec: a list of closed paths where the
    first is the outer boundary and any others are holes.
    """
    frame: int
    label: str
    cls: int
    site: str | None
    paths: list[np.ndarray]
    keyframe: bool = True
    interpolated: bool = False

    @property
    def points(self) -> np.ndarray:
        """Outer boundary, for area/box computations."""
        return self.paths[0]


@dataclass
class ClipAnnotation:
    clip_id: str
    tag: str | None = None
    thrombosis: str | None = None
    compressibility: str | None = None
    reviewed: bool = False
    protocol_deviation: bool = False
    acep_by_frame: dict[int, int] = field(default_factory=dict)
    instances: list[Instance] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    schema: str = "unknown"

    @property
    def annotated_frames(self) -> list[int]:
        return sorted({i.frame for i in self.instances})


# --------------------------------------------------------------- helpers
def _walk(obj: Any) -> Iterable[dict]:
    """Yield every dict nested anywhere inside a JSON structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _find_first(obj: Any, keys: set[str]):
    """Return the first value found for any of `keys`, searching depth-first."""
    for d in _walk(obj):
        for k in d:
            if k.lower().replace(" ", "_") in keys:
                return d[k]
    return None


# US-DICOMizer stores classification as codes, not words.
THROMBOSIS_DECODE = {"DVT1": "Yes", "DVT0": "No"}
COMPRESSIBILITY_DECODE = {"COMP2": "Yes", "COMP1": "Partial", "COMP0": "No"}


def decode_classification(value, table):
    v = str(value or "").strip()
    return table.get(v.upper(), v)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def _digits(v) -> int | None:
    s = "".join(c for c in str(v) if c.isdigit())
    return int(s) if s else None


def _points(seq) -> np.ndarray | None:
    pts = []
    for p in seq or []:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    return np.asarray(pts, dtype=np.float32) if len(pts) >= 3 else None


def _polygon_paths(node: dict) -> list[np.ndarray] | None:
    """Extract polygon paths per the Darwin spec.

    Spec: polygon.paths is a list of closed paths (first = outer boundary,
    remainder = holes). The deprecated complex_polygon and the legacy
    polygon.path form are also accepted.
    """
    poly = node.get("polygon")
    if isinstance(poly, dict):
        if isinstance(poly.get("paths"), list):
            out = []
            for path in poly["paths"]:
                a = _points(path)
                if a is not None:
                    out.append(a)
            if out:
                return out
        a = _points(poly.get("path"))
        if a is not None:
            return [a]
    cpoly = node.get("complex_polygon")          # deprecated in V2
    if isinstance(cpoly, dict):
        for key in ("paths", "path"):
            val = cpoly.get(key)
            if isinstance(val, list) and val and isinstance(val[0], list):
                out = [a for a in (_points(p) for p in val) if a is not None]
                if out:
                    return out
            a = _points(val)
            if a is not None:
                return [a]
    return None


# --------------------------------------------- Darwin JSON 2.0 (canonical)
def _parse_properties(node: dict, ann: ClipAnnotation) -> None:
    """Read Darwin `properties`: [{frame_index, name, value}, ...].

    Darwin properties are per-frame, which is how a per-frame ACEP grade and
    per-file classification labels both land in the same list.
    """
    for prop in node.get("properties") or []:
        if not isinstance(prop, dict):
            continue
        name = str(prop.get("name", "")).strip().lower().replace(" ", "_")
        value = prop.get("value")
        if value is None:
            continue
        fi = prop.get("frame_index", 0)
        # US-DICOMizer v5.6 writes this as exactly "ACEP Grading Score",
        # which normalises to "acep_grading_score" - absent from the old key
        # set, so every frame grade was being dropped.
        if name in {"acep", "acep_grade", "frame_grade", "grade",
                    "acep_grading_score"}:
            g = _digits(value)
            if g is not None:
                ann.acep_by_frame[int(fi)] = g
        elif name in {"thrombosis", "dvt_status"}:
            ann.thrombosis = decode_classification(value, THROMBOSIS_DECODE)
        elif name in {"compressibility", "compressible"}:
            ann.compressibility = decode_classification(value, COMPRESSIBILITY_DECODE)
        elif name in {"tag", "anatomical_tag", "site"}:
            ann.tag = str(value)
        elif name.lstrip("_") in {"reviewed", "review", "review_status"}:
            ann.reviewed = _as_bool(value)
        elif name in {"protocol_deviation", "deviation"}:  # "Protocol Deviation"
            ann.protocol_deviation = _as_bool(value)


def _parse_darwin_v2(raw: dict, ann: ClipAnnotation) -> bool:
    """Parse canonical Darwin JSON 2.0. Returns True if it applied."""
    annotations = raw.get("annotations")
    if not isinstance(annotations, list):
        return False
    ann.schema = f"darwin-{raw.get('version', '2.0')}"

    for node in annotations:
        if not isinstance(node, dict):
            continue
        _parse_properties(node, ann)
        label = node.get("name")
        if not isinstance(label, str):
            continue
        cls = C.label_to_class(label)
        site = C.label_to_site(label)
        interpolated = bool(node.get("interpolated", False))

        frames = node.get("frames")
        if isinstance(frames, dict):
            # Sequence annotation: the frame index is the DICT KEY, not a
            # field. Reading a "frame" field here instead would silently
            # collapse every polygon in the clip onto frame 0.
            for fkey, fnode in frames.items():
                if not str(fkey).isdigit() or not isinstance(fnode, dict):
                    continue
                paths = _polygon_paths(fnode)
                if paths is None:
                    continue
                if cls == C.CLASS_BACKGROUND:
                    ann.unparsed.append(label)
                    continue
                ann.instances.append(Instance(
                    frame=int(fkey), label=label, cls=cls, site=site,
                    paths=paths,
                    keyframe=bool(fnode.get("keyframe", True)),
                    interpolated=interpolated,
                ))
        else:
            paths = _polygon_paths(node)
            if paths is None:
                continue
            if cls == C.CLASS_BACKGROUND:
                ann.unparsed.append(label)
                continue
            # Canonical image annotations carry no frame field and default to
            # 0. Some exporters emit Darwin-shaped JSON with an explicit frame
            # field instead of a `frames` map; honour it, or every polygon in
            # the clip collapses onto frame 0.
            ann.instances.append(Instance(
                frame=_extract_frame_idx(node, default=0),
                label=label, cls=cls, site=site, paths=paths))
    return True


# --------------------------------------------- generic fallback (non-spec)
def _extract_frame_idx(node: dict, default: int = 0) -> int:
    for k in ("frame", "frame_index", "frame_number", "frameIndex", "slice"):
        if k in node:
            try:
                return int(node[k])
            except (TypeError, ValueError):
                pass
    return default


def _parse_generic(raw: Any, ann: ClipAnnotation) -> None:
    """Best-effort parse for exports that do not follow the Darwin layout."""
    ann.schema = "generic"
    for node in _walk(raw):
        for k in node:
            if k.lower().replace(" ", "_") in {"acep", "acep_grade", "frame_grade"}:
                fi = _extract_frame_idx(node, default=-1)
                g = _digits(node[k])
                if fi >= 0 and g is not None:
                    ann.acep_by_frame[fi] = g
        label = None
        for k in ("name", "label", "class", "annotation_class"):
            if isinstance(node.get(k), str):
                label = node[k]
                break
        if label is None:
            continue
        paths = _polygon_paths(node)
        if paths is None:
            a = _points(node.get("points"))
            paths = [a] if a is not None else None
        if paths is None:
            continue
        cls = C.label_to_class(label)
        if cls == C.CLASS_BACKGROUND:
            ann.unparsed.append(label)
            continue
        ann.instances.append(Instance(
            frame=_extract_frame_idx(node), label=label, cls=cls,
            site=C.label_to_site(label), paths=paths))


# ----------------------------------------------------------- main parser
def parse_annotation_json(path: Path, clip_id: str | None = None) -> ClipAnnotation:
    raw = json.loads(Path(path).read_text())
    clip_id = clip_id or Path(path).stem
    ann = ClipAnnotation(clip_id=clip_id)

    handled = _parse_darwin_v2(raw, ann) if isinstance(raw, dict) else False
    if not handled or not ann.instances:
        before = len(ann.instances)
        _parse_generic(raw, ann)
        if handled and len(ann.instances) == before:
            ann.schema = "darwin-empty"

    # ACEP grades may live in Darwin `properties`, or in a sibling structure
    # depending on exporter version. Scan the whole document as a supplement;
    # already-parsed entries win.
    for node in _walk(raw):
        for k in node:
            if k.lower().replace(" ", "_") in {"acep", "acep_grade", "frame_grade"}:
                fi = _extract_frame_idx(node, default=-1)
                g = _digits(node[k])
                if fi >= 0 and g is not None:
                    ann.acep_by_frame.setdefault(fi, g)

    # Clip-level classification may sit outside `properties` in some exports.
    ann.tag = ann.tag or _find_first(raw, {"tag", "anatomical_tag", "site"})
    ann.thrombosis = ann.thrombosis or decode_classification(
        _find_first(raw, {"thrombosis"}), THROMBOSIS_DECODE) or None
    ann.compressibility = ann.compressibility or decode_classification(
        _find_first(raw, {"compressibility", "compressible"}),
        COMPRESSIBILITY_DECODE) or None
    ann.reviewed = ann.reviewed or _as_bool(_find_first(raw, {"reviewed", "review"}))
    ann.protocol_deviation = ann.protocol_deviation or _as_bool(
        _find_first(raw, {"protocol_deviation", "protocoldeviation"}))
    return ann


# ------------------------------------------------------------ rasterizing
def rasterize_frame(ann: ClipAnnotation, frame: int,
                    height: int, width: int,
                    keyframes_only: bool = True) -> np.ndarray:
    """Build a uint8 semantic mask for one frame.

    Painting order matters: clot is drawn last so that a clot polygon drawn
    inside a vein polygon wins. US-DICOMizer explicitly allows labels to
    overlap or be enclosed within other labels.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    order = [C.CLASS_ARTERY, C.CLASS_VEIN, C.CLASS_CLOT]
    for cls in order:
        for inst in ann.instances:
            if inst.frame != frame or inst.cls != cls:
                continue
            if keyframes_only and not inst.keyframe:
                continue
            _fill_with_holes(mask, inst.paths, int(cls))
    return mask


def _fill_with_holes(mask: np.ndarray, paths: list[np.ndarray], value: int) -> None:
    """Fill the outer path, then carve out any hole paths.

    Darwin polygons carry holes as additional paths after the first. Filling
    every path with the same value would paint the holes solid, silently
    inflating lumen area - which is the exact quantity the compressibility
    measurement depends on.
    """
    outer = np.round(paths[0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [outer], color=value)
    for hole in paths[1:]:
        h = np.round(hole).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [h], color=0)


def instance_boxes(ann: ClipAnnotation, frame: int,
                   height: int, width: int,
                   keyframes_only: bool = True) -> list[dict]:
    """Per-instance boxes in xyxy, for MedSAM prompting."""
    out = []
    for inst in _frame_instances(ann, frame, keyframes_only):
        x0, y0 = inst.points.min(axis=0)
        x1, y1 = inst.points.max(axis=0)
        out.append({
            "label": inst.label,
            "cls": int(inst.cls),
            "site": inst.site,
            "box": [float(max(0, x0)), float(max(0, y0)),
                    float(min(width - 1, x1)), float(min(height - 1, y1))],
        })
    return out


def instance_mask(ann: ClipAnnotation, frame: int, inst_idx: int,
                  height: int, width: int,
                  keyframes_only: bool = True) -> np.ndarray:
    """Binary mask for a single instance (MedSAM trains one box -> one mask)."""
    inst = _frame_instances(ann, frame, keyframes_only)[inst_idx]
    m = np.zeros((height, width), dtype=np.uint8)
    _fill_with_holes(m, inst.paths, 1)
    return m


def _frame_instances(ann: ClipAnnotation, frame: int,
                     keyframes_only: bool = True) -> list[Instance]:
    return [i for i in ann.instances
            if i.frame == frame and (i.keyframe or not keyframes_only)]


def area_curves(ann: ClipAnnotation,
                keyframes_only: bool = True) -> dict[str, dict[int, float]]:
    """Per-vessel cross-sectional area over time, keyed by polygon label.

    Critically this is keyed by *label*, not by class. A single frame routinely
    contains several distinct veins (e.g. a compressed common femoral vein plus
    an uncompressed popliteal vein). Summing vein pixels across a frame mixes
    separate vessels together and destroys the compression signal, so each
    labelled vessel gets its own curve.

    Area is in pixels. Convert to mm^2 with the DICOM PhysicalDelta tags before
    comparing across machines or scanning depths.
    """
    curves: dict[str, dict[int, float]] = {}
    for inst in ann.instances:
        if inst.cls == C.CLASS_ARTERY:
            continue  # arteries do not compress; they are landmarks, not signal
        if keyframes_only and not inst.keyframe:
            continue  # interpolated frames are the tool's guess, not ground truth
        area = float(cv2.contourArea(inst.paths[0].astype(np.float32)))
        for hole in inst.paths[1:]:
            area -= float(cv2.contourArea(hole.astype(np.float32)))
        curves.setdefault(inst.label, {})[inst.frame] = max(area, 0.0)
    return curves


def compression_ratio(curve: dict[int, float]) -> float | None:
    """Minimum-to-maximum area ratio for one vessel.

    ~0 means the vein fully collapsed (normal). Near 1 means it did not
    compress (suspicious for thrombus). This is the interpretable baseline that
    any learned model must beat to justify itself.
    """
    if len(curve) < 2:
        return None
    vals = list(curve.values())
    hi = max(vals)
    if hi <= 0:
        return None
    return min(vals) / hi


def usable_frames(ann: ClipAnnotation,
                  min_acep: int = C.MIN_ACEP_GRADE,
                  keyframes_only: bool = True) -> list[int]:
    """Frames that have polygons AND pass the ACEP quality gate.

    If no ACEP grade was recorded for a frame we keep it: US-DICOMizer will not
    let a polygon be drawn on a frame graded below 3, so the presence of a
    polygon is itself evidence the gate was passed.
    """
    frames = sorted({i.frame for i in ann.instances
                     if i.keyframe or not keyframes_only})
    return [f for f in frames if ann.acep_by_frame.get(f, min_acep) >= min_acep]
