"""Verify the annotation parser on a synthetic US-DICOMizer-style export."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import config as C
from dvt import step2_json_to_masks as S2


def circle(cx, cy, r, n=16):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [{"x": float(cx + r * np.cos(a)), "y": float(cy + r * np.sin(a))} for a in t]


SYNTHETIC = {
    "dataset": "thrombus",
    "image": {"filename": "anonymized_P001_1_0001_CFV-R.dcm",
              "width": 640, "height": 480},
    "classification": {
        "tag": "Common Femoral Vein (Right)",
        "thrombosis": "No",
        "compressibility": "Yes",
        "reviewed": True,
        "protocol_deviation": False,
    },
    "annotations": [
        {"name": "Common Femoral Vein # 1", "frame": 10,
         "polygon": {"path": circle(300, 200, 40)}},
        {"name": "Common Femoral Artery # 1", "frame": 10,
         "polygon": {"path": circle(380, 210, 25)}},
        {"name": "Clot", "frame": 10,
         "polygon": {"path": circle(300, 200, 15)}},
        {"name": "Common Femoral Vein # 1", "frame": 40,
         "polygon": {"path": circle(300, 200, 12)}},   # compressed
        {"name": "Popliteal Vein # 4", "frame": 40,
         "polygon": {"path": circle(120, 300, 30)}},
        {"name": "Other", "frame": 10,
         "polygon": {"path": circle(50, 50, 10)}},     # should be skipped
    ],
    "acep": [
        {"frame": 10, "frame_grade": "Grade 4"},
        {"frame": 40, "frame_grade": "Grade 3"},
        {"frame": 55, "frame_grade": "Grade 2"},
    ],
}

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "anonymized_P001_1_0001_CFV-R.json"
    p.write_text(json.dumps(SYNTHETIC))
    ann = S2.parse_annotation_json(p)

print("== clip-level fields ==")
check(ann.thrombosis == "No", f"thrombosis parsed: {ann.thrombosis!r}")
check(ann.compressibility == "Yes", f"compressibility parsed: {ann.compressibility!r}")
check(ann.reviewed is True, f"reviewed parsed: {ann.reviewed}")
check(ann.protocol_deviation is False, f"protocol_deviation: {ann.protocol_deviation}")

print("== instances ==")
check(len(ann.instances) == 5, f"5 usable instances (Other dropped): got {len(ann.instances)}")
check(ann.unparsed == ["Other"], f"unparsed recorded: {ann.unparsed}")
check(ann.annotated_frames == [10, 40], f"annotated frames: {ann.annotated_frames}")

by_cls = {}
for i in ann.instances:
    by_cls.setdefault(i.cls, []).append(i.label)
check(len(by_cls.get(C.CLASS_VEIN, [])) == 3, f"3 vein polys: {by_cls.get(C.CLASS_VEIN)}")
check(len(by_cls.get(C.CLASS_ARTERY, [])) == 1, f"1 artery poly: {by_cls.get(C.CLASS_ARTERY)}")
check(len(by_cls.get(C.CLASS_CLOT, [])) == 1, f"1 clot poly: {by_cls.get(C.CLASS_CLOT)}")

print("== site mapping ==")
sites = {i.label: i.site for i in ann.instances}
check(sites["Common Femoral Vein # 1"] == "common_femoral", f"CFV site: {sites['Common Femoral Vein # 1']}")
check(sites["Popliteal Vein # 4"] == "popliteal", f"Pop site: {sites['Popliteal Vein # 4']}")

print("== ACEP gate ==")
check(ann.acep_by_frame == {10: 4, 40: 3, 55: 2}, f"acep: {ann.acep_by_frame}")
check(S2.usable_frames(ann) == [10, 40], f"usable frames: {S2.usable_frames(ann)}")

print("== rasterization ==")
m10 = S2.rasterize_frame(ann, 10, 480, 640)
m40 = S2.rasterize_frame(ann, 40, 480, 640)
check(set(np.unique(m10)) == {0, 1, 2, 3}, f"frame10 classes: {sorted(set(np.unique(m10).tolist()))}")
check(m10[200, 300] == C.CLASS_CLOT, "clot painted over vein at centre (paint order)")
check(m10[200, 265] == C.CLASS_VEIN, "vein ring survives outside clot")
check(m10[210, 380] == C.CLASS_ARTERY, "artery painted")

print("== per-vessel area curves ==")
curves = S2.area_curves(ann)
check("Common Femoral Artery # 1" not in curves, "arteries excluded from curves")
cfv = curves["Common Femoral Vein # 1"]
check(set(cfv) == {10, 40}, f"CFV tracked on both frames: {sorted(cfv)}")
ratio = S2.compression_ratio(cfv)
check(ratio is not None and ratio < 0.2,
      f"CFV compresses (r=40 -> r=12), ratio={ratio:.3f}")
pop = curves["Popliteal Vein # 4"]
check(S2.compression_ratio(pop) is None,
      "single-frame vessel yields no ratio (cannot judge compressibility)")

# Demonstrate why per-frame class totals are the wrong statistic:
naive10 = int((m10 == C.CLASS_VEIN).sum() + (m10 == C.CLASS_CLOT).sum())
naive40 = int((m40 == C.CLASS_VEIN).sum())
check(naive40 > naive10 * 0.3,
      f"naive per-frame vein sum HIDES the collapse ({naive10}->{naive40}) "
      "- confirms per-instance tracking is required")

print("== boxes ==")
boxes = S2.instance_boxes(ann, 10, 480, 640)
check(len(boxes) == 3, f"3 boxes on frame 10: {len(boxes)}")
vb = [b for b in boxes if b["label"].startswith("Common Femoral Vein")][0]["box"]
check(abs(vb[0] - 260) < 2 and abs(vb[2] - 340) < 2, f"vein box x-range ~260-340: {vb}")

im = S2.instance_mask(ann, 10, 0, 480, 640)
check(set(np.unique(im)) == {0, 1}, "instance mask is binary")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
