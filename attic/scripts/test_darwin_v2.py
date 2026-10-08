"""Validate parsing of canonical Darwin JSON 2.0 sequence (video) annotations."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import config as C
from dvt import step2_json_to_masks as S2


def ring(cx, cy, r, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [{"x": float(cx + r * np.cos(a)), "y": float(cy + r * np.sin(a))} for a in t]


# Canonical Darwin 2.0: frame index is the DICT KEY, keyframe flags mark
# human-annotated vs interpolated frames.
DARWIN_V2 = {
    "version": "2.0",
    "item": {"name": "anonymized_P007_2_0002_POP-L.dcm", "path": "/"},
    "annotations": [
        {
            "name": "Popliteal Vein # 4",
            "ranges": [[10, 51]],
            "interpolated": True,
            "properties": [
                {"frame_index": 0, "name": "Compressibility", "value": "No"},
                {"frame_index": 0, "name": "Thrombosis", "value": "Yes"},
                {"frame_index": 10, "name": "ACEP Grade", "value": "Grade 4"},
                {"frame_index": 50, "name": "ACEP Grade", "value": "Grade 3"},
            ],
            "frames": {
                # outer ring with a HOLE (clot lumen) -> paths[1] must be carved
                "10": {"keyframe": True,
                       "polygon": {"paths": [ring(200, 150, 50), ring(200, 150, 20)]}},
                "30": {"keyframe": False,      # interpolated - NOT ground truth
                       "polygon": {"paths": [ring(200, 150, 45)]}},
                "50": {"keyframe": True,
                       "polygon": {"paths": [ring(200, 150, 46)]}},
            },
        },
        {
            "name": "Popliteal Artery # 4",
            "ranges": [[10, 11]],
            "frames": {"10": {"keyframe": True,
                              "polygon": {"paths": [ring(300, 120, 22)]}}},
        },
    ],
}

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "anonymized_P007_2_0002_POP-L.json"
    p.write_text(json.dumps(DARWIN_V2))
    ann = S2.parse_annotation_json(p)

print("== schema detection ==")
check(ann.schema == "darwin-2.0", f"detected schema: {ann.schema}")

print("== frame keys (the silent-failure case) ==")
frames = sorted({i.frame for i in ann.instances})
check(frames == [10, 30, 50], f"frame indices read from dict KEYS: {frames}")
check(frames != [0], "polygons did NOT collapse onto frame 0")

print("== keyframe vs interpolated ==")
kf = sorted({i.frame for i in ann.instances if i.keyframe})
check(kf == [10, 50], f"keyframes only: {kf} (frame 30 was interpolated)")
check(S2.usable_frames(ann) == [10, 50],
      f"usable_frames excludes interpolated: {S2.usable_frames(ann)}")
check(S2.usable_frames(ann, keyframes_only=False) == [10, 30, 50],
      "opt-in to interpolated frames still available")

print("== Darwin properties ==")
check(ann.compressibility == "No", f"compressibility from properties: {ann.compressibility!r}")
check(ann.thrombosis == "Yes", f"thrombosis from properties: {ann.thrombosis!r}")
check(ann.acep_by_frame == {10: 4, 50: 3},
      f"per-frame ACEP from properties frame_index: {ann.acep_by_frame}")

print("== polygon holes ==")
vein10 = [i for i in ann.instances if i.frame == 10 and i.cls == C.CLASS_VEIN][0]
check(len(vein10.paths) == 2, f"outer + hole parsed: {len(vein10.paths)} paths")
m = S2.rasterize_frame(ann, 10, 400, 500)
check(m[150, 200] == 0, "hole is carved out at centre, not painted solid")
check(m[150, 165] == C.CLASS_VEIN, "annulus between hole and outer edge is vein")
check(m[120, 300] == C.CLASS_ARTERY, "artery painted separately")

print("== area accounts for holes ==")
curves = S2.area_curves(ann)
c = curves["Popliteal Vein # 4"]
check(sorted(c) == [10, 50], f"interpolated frame excluded from curve: {sorted(c)}")
ideal = np.pi * (50 ** 2 - 20 ** 2)
check(abs(c[10] - ideal) / ideal < 0.05,
      f"frame10 area ~= pi(R^2-r^2)={ideal:.0f}, got {c[10]:.0f}")
ratio = S2.compression_ratio(c)
check(ratio is not None and ratio > 0.5,
      f"non-compressible vein -> high ratio {ratio:.3f} (correctly flagged abnormal)")

print("== hole handling matters ==")
naive = np.pi * 50 ** 2
check(abs(c[10] - naive) / naive > 0.1,
      f"ignoring the hole would overstate lumen area by {100*(naive-c[10])/c[10]:.0f}%")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
