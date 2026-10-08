"""
Pick `--max-contrast` from your own data instead of guessing.

Runs the tracker on one clip, then reports mask contrast per frame alongside
area, and renders a contact sheet so you can SEE which frames are wrong and
read off the contrast value that separates them.

    python scripts/tune_quality.py --checkpoint medsam_vit_b.pth \
        --dcm renamed/anonymized_FR1-0030_0_002_GS-L.dcm --site GS

Reading the output:
  * contrast is mask interior / surrounding wall. Lumen is anechoic, so a good
    mask is LOW. A mask that has slipped onto tissue approaches 1.0.
  * find the lowest contrast among the frames that look wrong in the contact
    sheet, and set --max-contrast just below it.
  * a threshold that skips more than about a third of frames is usually a sign
    the seed box is on the wrong structure, not that the threshold is wrong.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt import step8_compression_phase as S8
from dvt import step9_medsam_infer as M9
from dvt import step10_auto_seed as S10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dcm", required=True)
    ap.add_argument("--site", default=None, help="CFV|GS|FV|PV")
    ap.add_argument("--box", default=None, help="x0,y0,x1,y1 to override auto-seed")
    ap.add_argument("--img-size", type=int, default=None,
                    help="inferred from the checkpoint when omitted")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--track", default="recenter",
                    choices=["recenter", "fixed", "reseed"])
    ap.add_argument("--sheet", default="quality_sheet.png")
    a = ap.parse_args()

    dcm = Path(a.dcm)
    site = a.site or S10.site_from_filename(dcm.name)
    frames = S8.load_clip_frames(dcm, max_frames=a.max_frames,
                                 downsample=a.downsample)

    if a.box:
        box = [float(v) for v in a.box.split(",")]
    else:
        if site is None:
            sys.exit("no site tag in filename; pass --site")
        cands, _ = S10.auto_seed(frames, site)
        vein = [c for c in cands if c.role == "vein"]
        if not vein:
            sys.exit("auto-seed found no vein candidate on this clip")
        box = vein[0].box
        print(f"auto-seed box {[round(v) for v in box]} (site {site})")

    sam, img_size = M9.load_medsam(a.checkpoint, a.img_size, a.device)
    areas, masks, quality = M9.track_vessel(
        sam, img_size, frames, box, device=a.device, mode=a.track)

    q = np.asarray(quality, dtype=float)
    ar = np.asarray(areas, dtype=float)
    ok = ~np.isnan(q)

    print(f"\n{len(ar)} frames  |  area {int(ar.min())}-{int(ar.max())} px  "
          f"ratio {ar.min()/max(ar.max(),1):.3f}")
    print(f"contrast: min {np.nanmin(q):.3f}  median {np.nanmedian(q):.3f}  "
          f"max {np.nanmax(q):.3f}")

    print("\nhow many frames each threshold would SKIP:")
    for thr in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        n = int(np.nansum(q > thr))
        bar = "#" * int(30 * n / max(len(q), 1))
        print(f"  --max-contrast {thr:.2f} -> skip {n:3d}/{len(q)} {bar}")

    print("\nworst 10 frames by contrast (check these in the sheet):")
    order = np.argsort(-np.nan_to_num(q, nan=-1))[:10]
    for i in order:
        print(f"  frame {int(i):3d}  contrast {q[i]:.3f}  area {int(ar[i]):6d}")

    # contact sheet, ordered worst-first so failures are easy to spot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = list(order[:6]) + list(np.argsort(np.nan_to_num(q, nan=9))[:6])
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    for k, i in enumerate(show):
        axp = axes[k // 6, k % 6]
        ov = np.stack([frames[i] / 255.0] * 3, -1)
        ov[masks[i] > 0] = [1, 0.15, 0.15]
        axp.imshow(ov)
        axp.axis("off")
        tag = "WORST" if k < 6 else "best"
        axp.set_title(f"{tag} f{int(i)}  c={q[i]:.2f}  {int(ar[i])}px", fontsize=8)
    plt.tight_layout()
    plt.savefig(a.sheet, dpi=90)
    print(f"\ncontact sheet -> {a.sheet}")
    print("Top row is worst contrast, bottom row best. If the top row looks")
    print("fine, the threshold is not your problem - the seed box is.")


if __name__ == "__main__":
    main()
