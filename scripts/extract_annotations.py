r"""
Extract every `annotations/` folder from the nested ThrombUS+ archive.

Layout:

    thrombus_s3_annotated/
        FR1/
            DVT/     FR1-0001.zip   ->  annotations/*.json
            NON_DVT/ FR1-0002.zip   ->  annotations/*.json
        GR1/ ...

Each subject is its own zip containing an `annotations/` folder. This walks the
tree (or a spanned/normal outer zip), opens each subject zip, and pulls out only
the annotation JSONs - a few tens of MB rather than the 3.6 GB of pixel data.

Output preserves centre and DVT status in the filename, because that grouping IS
the label and losing it would make the files useless:

    out/FR1__DVT__FR1-0001__anonymized_FR1-0001_0_003_CFV-L.json

Usage
-----
    # after extracting the spanned archive locally
    python scripts/extract_annotations.py --root thrombus_s3_annotated --out annotations_flat

    # straight from a single (non-spanned) outer zip
    python scripts/extract_annotations.py --zip thrombus_s3_annotated.zip --out annotations_flat

    # then re-archive just the JSONs
    #   PowerShell: Compress-Archive annotations_flat\* annotations_only.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path


def parse_context(path_parts: list[str], subject_zip: str) -> tuple[str, str, str]:
    """Recover (centre, group, subject) from the path around a subject zip."""
    centre = group = "UNKNOWN"
    for p in path_parts:
        if p.upper() in ("DVT", "NON_DVT", "NONDVT"):
            group = p.upper()
        elif len(p) == 3 and p[:2].isalpha() and p[2].isdigit():
            centre = p.upper()
    subject = Path(subject_zip).stem
    if centre == "UNKNOWN" and "-" in subject:
        centre = subject.split("-")[0]
    return centre, group, subject


def harvest_subject(data: bytes, centre: str, group: str, subject: str,
                    out_dir: Path, stats: Counter,
                    dicom_dir: Path | None = None,
                    annotated_only: bool = True) -> int:
    """Pull annotation JSONs - and optionally the DICOMs - out of one subject zip.

    DICOM filenames are kept EXACTLY as-is, with no centre/group prefix. The
    training script pairs an annotation to its pixels by matching the `clip`
    field against the .dcm stem, and renaming would break that lookup.

    `annotated_only` extracts just the DICOMs that actually have an annotation.
    Of 1047 annotation files there are far fewer DICOMs worth pulling than the
    full archive contains, and pixels are ~3.5 GB against ~40 MB of JSON.
    """
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        stats["subject_zip_unreadable"] += 1
        return 0

    names = z.namelist()
    # prefer the annotations/ folder; fall back to any JSON if absent
    ann = [n for n in names if n.lower().endswith(".json") and "annotations/" in n.lower()]
    if not ann:
        ann = [n for n in names if n.lower().endswith(".json")]
        if ann:
            stats["subject_no_annotations_folder"] += 1

    written = 0
    wanted_stems = set()
    for n in ann:
        try:
            payload = z.read(n)
        except Exception:
            stats["json_read_error"] += 1
            continue
        stem = Path(n).name
        target = out_dir / f"{centre}__{group}__{subject}__{stem}"
        target.write_bytes(payload)
        wanted_stems.add(Path(stem).stem)
        written += 1

    if dicom_dir is not None:
        dicom_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            if not n.lower().endswith(".dcm"):
                continue
            stem = Path(n).stem
            if annotated_only and stem not in wanted_stems:
                stats["dcm_skipped_unannotated"] += 1
                continue
            target = dicom_dir / Path(n).name
            if target.exists():
                stats["dcm_duplicate"] += 1
                continue
            try:
                target.write_bytes(z.read(n))
                stats["dcm_written"] += 1
            except Exception:
                stats["dcm_read_error"] += 1
    return written


def walk_tree(root: Path, out_dir: Path, stats: Counter,
              dicom_dir: Path | None = None, annotated_only: bool = True) -> int:
    total = 0
    for zp in sorted(root.rglob("*.zip")):
        centre, group, subject = parse_context(list(zp.parts), zp.name)
        n = harvest_subject(zp.read_bytes(), centre, group, subject, out_dir,
                            stats, dicom_dir, annotated_only)
        stats[f"{centre}/{group}"] += 1
        total += n
        print(f"  {centre:5s} {group:8s} {subject:12s} -> {n:3d} json")
    return total


def walk_zip(zip_path: Path, out_dir: Path, stats: Counter,
             dicom_dir: Path | None = None, annotated_only: bool = True) -> int:
    outer = zipfile.ZipFile(zip_path)
    inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
    total = 0
    for n in inner:
        parts = n.split("/")
        centre, group, subject = parse_context(parts, parts[-1])
        try:
            data = outer.read(n)
        except Exception as e:
            # Spanned archives (.z01/.z02/...) store segment-relative offsets;
            # a flat concatenation cannot resolve them. Extract the outer
            # archive with a tool that understands spanning first, then use
            # --root on the extracted folder.
            stats["outer_read_error"] += 1
            print(f"  SKIP {subject}: {type(e).__name__} - extract the outer "
                  "archive first, then re-run with --root")
            continue
        got = harvest_subject(data, centre, group, subject, out_dir, stats,
                              dicom_dir, annotated_only)
        stats[f"{centre}/{group}"] += 1
        total += got
        print(f"  {centre:5s} {group:8s} {subject:12s} -> {got:3d} json")
    return total


def summarise(out_dir: Path) -> dict:
    """Report what was actually recovered, so gaps are visible immediately."""
    files = sorted(out_dir.glob("*.json"))
    labels, frames, clot_clips, multiframe = Counter(), Counter(), 0, 0
    for f in files:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        slots = (j.get("item") or {}).get("slots") or [{}]
        fc = slots[0].get("frame_count", 1) or 1
        if fc > 1:
            multiframe += 1
        has_clot = False
        for a in j.get("annotations") or []:
            nm = a.get("name", "")
            for fk, fv in (a.get("frames") or {}).items():
                if "polygon" in fv:
                    labels[nm] += 1
                    frames[nm] += 1
                    if nm.strip().lower() == "clot":
                        has_clot = True
        if has_clot:
            clot_clips += 1
    return {"json_files": len(files), "multiframe_clips": multiframe,
            "clips_with_clot": clot_clips, "labels": dict(labels.most_common())}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="extracted thrombus_s3_annotated folder")
    src.add_argument("--zip", help="single outer zip (not spanned)")
    ap.add_argument("--out", default="annotations_flat")
    ap.add_argument("--dicom-out", default=None,
                    help="also extract DICOMs here (needed for training)")
    ap.add_argument("--all-dicoms", action="store_true",
                    help="extract every DICOM, not just annotated ones")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    dicom_dir = Path(a.dicom_out) if a.dicom_out else None
    total = (walk_tree(Path(a.root), out_dir, stats, dicom_dir, not a.all_dicoms)
             if a.root else
             walk_zip(Path(a.zip), out_dir, stats, dicom_dir, not a.all_dicoms))

    print(f"\n{total} annotation files -> {out_dir}")
    print("\nsubjects by centre/group:")
    for k, v in sorted(stats.items()):
        if "/" in k:
            print(f"  {k:16s} {v:3d}")
    if dicom_dir:
        print(f"\nDICOMs written: {stats['dcm_written']} -> {dicom_dir}")
        if stats["dcm_skipped_unannotated"]:
            print(f"  skipped (no annotation): {stats['dcm_skipped_unannotated']}"
                  " - pass --all-dicoms to keep them")
        pairs = sum(1 for j in out_dir.glob('*.json')
                    if (dicom_dir / (j.name.split('__', 3)[-1].replace('.json', '.dcm'))).exists())
        print(f"  annotation/DICOM pairs resolvable: {pairs}/{total}")
        if pairs < total:
            print("  WARNING unpaired annotations will be skipped in training")

    for k in ("subject_zip_unreadable", "subject_no_annotations_folder",
              "json_read_error", "outer_read_error", "dcm_read_error",
              "dcm_duplicate"):
        if stats[k]:
            print(f"  WARNING {k}: {stats[k]}")

    print("\ncontent summary:")
    print(json.dumps(summarise(out_dir), indent=2))


if __name__ == "__main__":
    main()
