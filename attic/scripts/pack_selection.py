"""
Pack the 40 selected clips into a ZIP.

    python scripts/pack_selection.py /path/to/zenodo/train_dicoms /path/to/out.zip

Reads filenames_40.txt (or --list), finds each file anywhere under the source
directory, verifies every one is present, and writes a ZIP. Deliberately does
NOT recompress: the DICOMs are already JPEG-baseline, so zip deflate adds time
for almost no size saving.
"""
import argparse
import sys
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="directory containing the training DICOMs")
    ap.add_argument("out", help="output .zip path")
    ap.add_argument("--list", default=None, help="text file of filenames")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent.parent
    list_path = Path(a.list) if a.list else here / "filenames_40.txt"
    if not list_path.exists():
        sys.exit(f"filename list not found: {list_path}")
    wanted = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]

    src = Path(a.source)
    index = {}
    for p in src.rglob("*"):
        if p.is_file():
            index.setdefault(p.name, p)

    found, missing = [], []
    for name in wanted:
        p = index.get(name)
        (found if p else missing).append(p or name)

    if missing:
        print(f"MISSING {len(missing)}/{len(wanted)}:")
        for m in missing[:10]:
            print(f"   {m}")
        sys.exit("aborting - check the source directory")

    total = sum(p.stat().st_size for p in found)
    print(f"packing {len(found)} files, {total/1e6:.0f} MB uncompressed")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for p in found:
            zf.write(p, arcname=p.name)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
