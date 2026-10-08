"""
Make the Zenodo DICOMs visible to US-DICOMizer.

The ThrombUS+ release ships files named by SHA-256 hash. The DICOM headers are
already anonymized and the images already cropped, but US-DICOMizer decides
whether a file is anonymized purely from the filename:

    # anonymized_filename_utils.py
    def looks_anonymized_filename(filename):
        return stem.startswith("anonymized_")

A hash name fails that test, so the file never reaches the "Anonymized DICOM
files" list and the Annotations tab stays greyed out. Nothing is wrong with the
data - it just needs the expected name.

Expected pattern, verified against the app's own builder and parser:

    anonymized_<SubjectID>_<file_no>_<TAG>.dcm
    e.g. anonymized_GR1-0252_0_001_GS-R.dcm

`file_no` uses Python underscore grouping, so 1 renders as "0_001", and TAG is
site plus limb: CFV-L, CFV-R, FV-L, FV-R, PV-L, PV-R, GS-L, GS-R.

Both values come straight from train.csv, so the mapping is exact rather than
guessed. The original hash is preserved in a CSV so results can be mapped back
for submission.

Usage:
    python scripts/rename_for_usdicomizer.py --csv train.csv --dir dcms --out renamed
    python scripts/rename_for_usdicomizer.py --csv train.csv --dir dcms --out renamed \
        --file-list filenames_40.txt          # just the 40-clip subset
    ... --copy      # copy instead of symlink/move (safer, uses more disk)
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

SITE_TAG = {"CFV": "CFV", "FV": "FV", "PV": "PV", "GS": "GS"}


def build_name(subject_id: str, file_no: int, site: str, limb: str) -> str:
    tag = f"{SITE_TAG[site]}-{limb.upper()}"
    return f"anonymized_{subject_id}_{file_no:04_}_{tag}.dcm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv")
    ap.add_argument("--dir", required=True, help="folder of hash-named DICOMs")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--file-list", default=None,
                    help="restrict to filenames listed in this text file")
    ap.add_argument("--copy", action="store_true",
                    help="copy files (default: hardlink, falling back to copy)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df.columns = [c.strip() for c in df.columns]

    if a.file_list:
        keep = {ln.strip() for ln in Path(a.file_list).read_text().splitlines()
                if ln.strip()}
        keep |= {k + ".dcm" for k in keep}
        df = df[df["File Name"].isin(keep)]
        print(f"restricted to {len(df)} rows from --file-list")

    src = Path(a.dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # file_no must be unique and stable within a subject; order by site so the
    # numbering follows the acquisition protocol down the leg
    site_order = {"CFV": 1, "GS": 2, "FV": 3, "PV": 4}
    df = df.assign(_ord=df["Anatomical Site"].map(site_order).fillna(9))
    df = df.sort_values(["SubjectID", "_ord", "Limb"])
    df["file_no"] = df.groupby("SubjectID").cumcount() + 1

    mapping, missing, clashes = [], [], []
    seen = set()
    for _, r in df.iterrows():
        srcp = src / r["File Name"]
        if not srcp.exists():
            missing.append(r["File Name"])
            continue
        new = build_name(r["SubjectID"], int(r["file_no"]),
                         r["Anatomical Site"], r["Limb"])
        if new in seen:
            clashes.append(new)
            continue
        seen.add(new)
        mapping.append({"original": r["File Name"], "renamed": new,
                        "SubjectID": r["SubjectID"], "site": r["Anatomical Site"],
                        "limb": r["Limb"]})
        if a.dry_run:
            continue
        dst = out / new
        if dst.exists():
            continue
        if a.copy:
            shutil.copy2(srcp, dst)
        else:
            try:
                dst.hardlink_to(srcp)      # no extra disk, instant
            except (OSError, AttributeError):
                shutil.copy2(srcp, dst)

    mp = out / "rename_map.csv"
    if not a.dry_run:
        pd.DataFrame(mapping).to_csv(mp, index=False)

    print(f"\n{len(mapping)} files {'would be ' if a.dry_run else ''}renamed -> {out}")
    if mapping:
        print(f"  example: {mapping[0]['original'][:16]}... -> {mapping[0]['renamed']}")
    if missing:
        print(f"  WARNING {len(missing)} listed in CSV but not found on disk")
        for m in missing[:5]:
            print(f"    {m}")
    if clashes:
        print(f"  WARNING {len(clashes)} name clashes skipped")
    if not a.dry_run:
        print(f"  mapping saved to {mp} - keep it, you need it to map results back")


if __name__ == "__main__":
    main()
