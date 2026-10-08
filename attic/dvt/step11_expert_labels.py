r"""
Step 11 - Parse the ThrombUS+ expert annotation vocabulary.

The expert set uses a different label scheme from the US-DICOMizer panel that
`step2_json_to_masks.py` was written for. Verified across all 1047 files:

    Other                     1063      FV Scanning Location 3     773
    Clot                       820      CFV Scanning Location 2    760
    FA Scanning Location 3     723      PV Scanning Location 4     720
    PA Scanning Location 4     695      GSV Scanning Location 2    633
    CFA Scanning Location 1    489      CFV Scanning Location 1    471
    FA Scanning Location 2     397      DFA Scanning Location 2    349
    CFA Scanning Location 2    324

The pattern is `<VESSEL> Scanning Location <N>`, where the vessel code ends in
V for veins and A for arteries:

    CFV/CFA  common femoral vein/artery
    FV /FA   femoral vein/artery
    PV /PA   popliteal vein/artery
    GSV      great saphenous vein
    DFA      deep femoral artery

`step2.label_to_class()` matches the substrings "vein" and "artery", so every
one of these labels would fall through to background and be SILENTLY DROPPED.
That is the bug this module exists to prevent.

`Other` is deliberately mapped to background: it appears 1063 times across only
196 files, 154 of which also carry a proper vessel label, so it is a catch-all
for incidental structures rather than a class worth learning.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASS_BACKGROUND = 0
CLASS_VEIN = 1
CLASS_ARTERY = 2
CLASS_CLOT = 3
NUM_CLASSES = 4
CLASS_NAMES = {0: "background", 1: "vein", 2: "artery", 3: "clot"}

VESSEL_RE = re.compile(r"^\s*([A-Z]{2,4})\s+Scanning\s+Location\s+(\d+)\s*$", re.I)

VESSEL_KIND = {
    "CFV": ("vein", "common_femoral"),
    "FV": ("vein", "femoral"),
    "PV": ("vein", "popliteal"),
    "GSV": ("vein", "great_saphenous"),
    "CFA": ("artery", "common_femoral"),
    "FA": ("artery", "femoral"),
    "PA": ("artery", "popliteal"),
    "DFA": ("artery", "deep_femoral"),
}


def parse_label(name: str):
    """-> (class_id, vessel_code, site_index) ; class 0 means ignore."""
    n = (name or "").strip()
    low = n.lower()
    if low == "clot":
        return CLASS_CLOT, "CLOT", None
    if low == "other":
        return CLASS_BACKGROUND, "OTHER", None
    m = VESSEL_RE.match(n)
    if not m:
        return CLASS_BACKGROUND, None, None
    code, loc = m.group(1).upper(), int(m.group(2))
    kind = VESSEL_KIND.get(code)
    if kind is None:                    # unknown code: fall back on the suffix
        kind = ("vein" if code.endswith("V") else
                "artery" if code.endswith("A") else None, None)
    cls = (CLASS_VEIN if kind[0] == "vein" else
           CLASS_ARTERY if kind[0] == "artery" else CLASS_BACKGROUND)
    return cls, code, loc


@dataclass
class Poly:
    frame: int
    label: str
    cls: int
    code: str | None
    location: int | None
    paths: list[np.ndarray]
    keyframe: bool = True

    @property
    def outer(self) -> np.ndarray:
        return self.paths[0]

    def area(self) -> float:
        a = _shoelace(self.paths[0])
        for h in self.paths[1:]:
            a -= _shoelace(h)
        return max(a, 0.0)


@dataclass
class Expert:
    path: str
    centre: str = "?"
    group: str = "?"          # DVT | NON_DVT
    subject: str = "?"
    clip: str = "?"
    width: int = 0
    height: int = 0
    frame_count: int = 1
    acep: dict[int, int] = field(default_factory=dict)
    polys: list[Poly] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def dvt(self) -> int:
        return 1 if self.group.upper() == "DVT" else 0

    def frames(self, cls: int | None = None) -> list[int]:
        return sorted({p.frame for p in self.polys
                       if cls is None or p.cls == cls})

    def by_vessel(self, cls: int = CLASS_VEIN) -> dict[str, dict[int, float]]:
        """label -> {frame: area}, for compression-ratio work."""
        out: dict[str, dict[int, float]] = {}
        for p in self.polys:
            if p.cls != cls:
                continue
            out.setdefault(p.label, {})[p.frame] = p.area()
        return out


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points(seq) -> np.ndarray | None:
    pts = []
    for p in seq or []:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    return np.asarray(pts, dtype=np.float32) if len(pts) >= 3 else None


def parse_expert_json(path: Path) -> Expert:
    """Read one expert annotation file.

    Context is recovered from the flattened filename written by
    `scripts/extract_annotations.py`:
        CENTRE__GROUP__SUBJECT__<original>.json
    The GROUP field is the DVT label and exists nowhere inside the JSON, so it
    must survive flattening.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    e = Expert(path=str(path))

    parts = path.name.split("__", 3)
    if len(parts) == 4:
        e.centre, e.group, e.subject, rest = parts
        e.clip = rest.replace(".json", "")

    item = raw.get("item") or {}
    slots = item.get("slots") or [{}]
    e.width = int(slots[0].get("width") or 0)
    e.height = int(slots[0].get("height") or 0)
    e.frame_count = int(slots[0].get("frame_count") or 1)

    for a in raw.get("annotations") or []:
        name = a.get("name", "")
        for prop in a.get("properties") or []:
            if str(prop.get("name", "")).lower().startswith("acep"):
                digits = "".join(c for c in str(prop.get("value", "")) if c.isdigit())
                if digits:
                    e.acep[int(prop.get("frame_index", 0))] = int(digits)

        cls, code, loc = parse_label(name)
        for fk, fv in (a.get("frames") or {}).items():
            if not str(fk).lstrip("-").isdigit() or "polygon" not in fv:
                continue
            paths = []
            for p in fv["polygon"].get("paths") or []:
                arr = _points(p)
                if arr is not None:
                    paths.append(arr)
            if not paths:
                continue
            if cls == CLASS_BACKGROUND:
                e.unknown.append(name)
                continue
            e.polys.append(Poly(frame=int(fk), label=name, cls=cls, code=code,
                                location=loc, paths=paths,
                                keyframe=bool(fv.get("keyframe", True))))
    return e


def rasterize(e: Expert, frame: int, height: int | None = None,
              width: int | None = None) -> np.ndarray:
    """Semantic mask for one frame: 0 bg, 1 vein, 2 artery, 3 clot.

    Clot is painted last so a clot drawn inside a vein wins - it is a region
    WITHIN the lumen, and painting vein over it would erase the class the whole
    diagnosis rests on.
    """
    import cv2

    h = height or e.height
    w = width or e.width
    m = np.zeros((h, w), dtype=np.uint8)
    for cls in (CLASS_ARTERY, CLASS_VEIN, CLASS_CLOT):
        for p in e.polys:
            if p.frame != frame or p.cls != cls:
                continue
            cv2.fillPoly(m, [np.round(p.paths[0]).astype(np.int32).reshape(-1, 1, 2)],
                         color=int(cls))
            for hole in p.paths[1:]:
                cv2.fillPoly(m, [np.round(hole).astype(np.int32).reshape(-1, 1, 2)],
                             color=0)
    return m


def instance_masks(e: Expert, frame: int, height: int | None = None,
                   width: int | None = None):
    """One binary mask + box per polygon - the unit MedSAM trains on."""
    import cv2

    h = height or e.height
    w = width or e.width
    out = []
    for p in e.polys:
        if p.frame != frame:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(p.paths[0]).astype(np.int32).reshape(-1, 1, 2)], 1)
        for hole in p.paths[1:]:
            cv2.fillPoly(m, [np.round(hole).astype(np.int32).reshape(-1, 1, 2)], 0)
        if m.sum() < 20:
            continue
        ys, xs = np.where(m)
        out.append({"mask": m, "cls": p.cls, "label": p.label, "code": p.code,
                    "location": p.location,
                    "box": [float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())]})
    return out


def scan(folder: Path) -> list[Expert]:
    return [parse_expert_json(p) for p in sorted(Path(folder).glob("*.json"))]


if __name__ == "__main__":
    import sys
    from collections import Counter

    items = scan(Path(sys.argv[1] if len(sys.argv) > 1 else "annotations_flat"))
    print(f"{len(items)} files")
    cls = Counter(p.cls for e in items for p in e.polys)
    print("polygons by class:",
          {CLASS_NAMES[k]: v for k, v in sorted(cls.items())})
    print("multi-frame clips:", sum(1 for e in items if e.frame_count > 1))
    print("files with clot   :", sum(1 for e in items
                                     if any(p.cls == CLASS_CLOT for p in e.polys)))
    print("subjects          :", len({e.subject for e in items}))
    unk = Counter(u for e in items for u in e.unknown)
    if unk:
        print("labels mapped to background:", dict(unk))
