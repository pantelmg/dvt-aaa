"""
Round-trip test against the REAL US-DICOMizer v5.6 source.

Writes a sidecar with our writer, then feeds it to the app's own
`import_darwin_annotation` and `detect_annotation_format_payload`. If this
passes, the radiologist will see our proposals in the Annotations tab.

Requires the app modules on sys.path (fetched from the public MIT repo).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/claude/usd")           # US-DICOMizer source

from dvt import step7_write_annotations as W

import annotation_metadata_utils as AMU          # the app's own code
import package_roundtrip_utils as PRU
import package_io_utils as PIO

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def ring(cx, cy, r, n=20):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


payload = W.build_sidecar(
    dcm_filename="anonymized_P042_1_0001_CFV-R.dcm",
    width=640, height=480, frame_count=120,
    polygons=[
        {"label": "Common Femoral Vein # 1", "frame": 10, "points": ring(300, 200, 45)},
        {"label": "Common Femoral Artery # 1", "frame": 10, "points": ring(400, 210, 25)},
        {"label": "Common Femoral Vein # 1", "frame": 60, "points": ring(300, 200, 12)},
    ],
    acep_by_frame={10: 4, 60: 3},
    thrombosis="No", compressibility="Yes",
    patient_id="P042", clinician_name="A Model", clinician_email="model@x.org",
)

print("== format detection (app's own detector) ==")
fmt = PRU.detect_annotation_format_payload(payload)
check(fmt == "darwin", f"detected as {fmt!r} (needs both 'annotations' and 'item')")

print("== import via the app's own import_darwin_annotation ==")
data = AMU.import_darwin_annotation({}, payload)
data = AMU.ensure_annotation_schema(data)

frames = data["frames"]
check(set(frames) == {"10", "60"}, f"frames imported: {sorted(frames)}")
check(len(frames["10"]) == 2, f"frame 10 has 2 polygons: {len(frames.get('10', []))}")
check(len(frames["60"]) == 1, f"frame 60 has 1 polygon: {len(frames.get('60', []))}")

labels10 = sorted(p["label"] for p in frames["10"])
check(labels10 == ["Common Femoral Artery # 1", "Common Femoral Vein # 1"],
      f"labels survive: {labels10}")
check(len(frames["10"][0]["points"]) == 20,
      f"all 20 vertices survive: {len(frames['10'][0]['points'])}")

print("== classification tags ==")
cls = data["classification"]
check(cls["thrombosis"] == "DVT0", f"thrombosis code: {cls['thrombosis']!r}")
check(cls["compressibility"] == "COMP2", f"compressibility code: {cls['compressibility']!r}")
check(cls["_reviewed"] is False,
      "NOT pre-marked reviewed - the radiologist's tick must mean something")
check(cls["protocol_deviation"] is False, "protocol deviation defaults false")

print("== ACEP frame grading ==")
check(data["frame_grading"] == {"10": "Grade 4", "60": "Grade 3"},
      f"ACEP re-inflated with 'Grade ' prefix: {data['frame_grading']}")

print("== the properties trap ==")
broken = W.build_sidecar("x.dcm", 640, 480, 10,
                         polygons=[{"label": "Popliteal Vein # 4", "frame": 5,
                                    "points": ring(100, 100, 20)}])
for a in broken["annotations"]:
    if a["name"] == "Popliteal Vein # 4":
        a["properties"] = [{"frame_index": 5, "name": "note", "value": "x"}]
bad = AMU.ensure_annotation_schema(AMU.import_darwin_annotation({}, broken))
check(bad["frames"] == {},
      "non-empty properties on a polygon => SILENTLY DROPPED (confirms the trap)")

print("== sidecar path matches the app's lookup ==")
dcm = Path("/tmp/rt/anonymized_P042_1_0001_CFV-R.dcm")
dcm.parent.mkdir(parents=True, exist_ok=True)
ours = W.sidecar_path(dcm)
theirs = PIO.get_preferred_sidecar_paths(str(dcm))["darwin"]
check(str(ours) == theirs, f"path agreement:\n         ours   {ours}\n         theirs {theirs}")

W.write_sidecar(dcm, payload)
dcm.write_bytes(b"")                      # placeholder so the lookup sees a file
found = PIO.find_existing_sidecar_paths(str(dcm))
check(found["darwin"] and Path(found["darwin"][0]).exists(),
      f"app discovers our file: {found['darwin']}")

print("== polygon simplification ==")
m = np.zeros((200, 200), np.uint8)
yy, xx = np.mgrid[:200, :200]
m[((yy - 100) ** 2 + (xx - 100) ** 2) < 45 ** 2] = 1
pts = W.mask_to_polygon(m, max_points=40)
check(pts is not None and 3 <= len(pts) <= 40,
      f"circle mask -> {len(pts)} vertices (hand-editable, not 400)")
check(W.mask_to_polygon(np.zeros((50, 50), np.uint8)) is None,
      "empty mask returns None rather than a degenerate polygon")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
