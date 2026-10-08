"""
Self-service annotation-JSON inspector.

Run this on YOUR machine against a real US-DICOMizer export. It reports whether
the parser handles your file and, if not, exactly what is missing — without
requiring you to share the file.

    python scripts/inspect_json.py path/to/annotations.json

Add --share to print a redacted structural summary (key names and value TYPES
only, no coordinates, no names, no identifiers) that is safe to paste
elsewhere if you want help debugging.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import config as C
from dvt import step2_json_to_masks as S2

WARN, OK = "[WARN]", "[ ok ]"


def skeleton(obj, depth=0, max_depth=6):
    """Structure only: key names mapped to value types. No values retained."""
    if depth > max_depth:
        return "..."
    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:40]:
            key = k if not str(k).isdigit() else "<int_key>"
            out[key] = skeleton(v, depth + 1, max_depth)
        return out
    if isinstance(obj, list):
        if not obj:
            return []
        return [skeleton(obj[0], depth + 1, max_depth), f"...x{len(obj)}"]
    return type(obj).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--share", action="store_true",
                    help="print a redacted structure summary safe to share")
    a = ap.parse_args()
    path = Path(a.path)
    raw = json.loads(path.read_text())

    print(f"=== {path.name} ===\n")
    print("--- top level ---")
    if isinstance(raw, dict):
        print(f"  keys: {list(raw.keys())}")
        print(f"  version field: {raw.get('version', '<absent>')}")
        anns = raw.get("annotations")
        print(f"  annotations: {'list of ' + str(len(anns)) if isinstance(anns, list) else type(anns).__name__}")
    else:
        print(f"  document is a {type(raw).__name__}, not an object")

    # --- what the parser actually recovers ---
    ann = S2.parse_annotation_json(path)
    print("\n--- parse result ---")
    print(f"  schema detected      : {ann.schema}")
    print(f"  polygon instances    : {len(ann.instances)}")
    print(f"  distinct frames      : {len(ann.annotated_frames)}")
    print(f"  keyframe instances   : {sum(1 for i in ann.instances if i.keyframe)}")
    print(f"  interpolated dropped : {sum(1 for i in ann.instances if not i.keyframe)}")
    print(f"  ACEP grades found    : {len(ann.acep_by_frame)}")
    print(f"  compressibility      : {ann.compressibility!r}")
    print(f"  thrombosis           : {ann.thrombosis!r}")
    print(f"  anatomical tag       : {ann.tag!r}")

    print("\n--- diagnosis ---")
    ok = True
    if not ann.instances:
        ok = False
        print(f"{WARN} NO polygons recovered. Run with --share and send the output.")
    else:
        print(f"{OK} polygons parsed")

    if ann.instances and len(ann.annotated_frames) == 1 and ann.annotated_frames[0] == 0:
        ok = False
        print(f"{WARN} every polygon landed on frame 0. If this clip is multi-frame,")
        print("       the frame index is stored somewhere the parser is not reading.")
    elif len(ann.annotated_frames) > 1:
        print(f"{OK} multiple frames resolved: {ann.annotated_frames[:10]}")

    if ann.compressibility is None:
        ok = False
        print(f"{WARN} no Compressibility label found — the clip-level task has no target.")
    if ann.thrombosis is None:
        print(f"{WARN} no Thrombosis label found (secondary label, not fatal).")
    if not ann.acep_by_frame:
        print(f"{WARN} no ACEP grades found; the quality gate will pass everything.")

    labels = Counter(i.label for i in ann.instances)
    if labels:
        print("\n--- labels seen ---")
        for k, v in labels.most_common():
            cls = C.CLASS_NAMES[C.label_to_class(k)]
            site = C.label_to_site(k) or "-"
            print(f"  {k!r:36s} x{v:<4d} -> {cls:10s} site={site}")
    if ann.unparsed:
        ok = False
        print(f"\n{WARN} labels NOT in config.py (add them to VEIN/ARTERY/OTHER lists):")
        for k, v in Counter(ann.unparsed).most_common():
            print(f"       {k!r} x{v}")

    n_paths = Counter(len(i.paths) for i in ann.instances)
    if any(k > 1 for k in n_paths):
        print(f"\n{OK} polygons with holes present: {dict(n_paths)} (paths per instance)")

    curves = S2.area_curves(ann)
    print("\n--- compressibility computability ---")
    if not curves:
        print(f"{WARN} no vein curves — cannot compute compression ratio.")
    for vessel, curve in curves.items():
        r = S2.compression_ratio(curve)
        status = f"ratio={r:.3f}" if r is not None else "UNCOMPUTABLE (needs >=2 frames)"
        print(f"  {vessel!r:36s} {len(curve)} frame(s)  {status}")

    print()
    print("RESULT: parser handles this file." if ok
          else "RESULT: parser needs adjustment — see warnings above.")

    if a.share:
        print("\n" + "=" * 60)
        print("REDACTED STRUCTURE (key names + value types only — no coordinates,")
        print("no patient data, no label text). Safe to share for debugging.")
        print("=" * 60)
        print(json.dumps(skeleton(raw), indent=2)[:6000])


if __name__ == "__main__":
    main()
