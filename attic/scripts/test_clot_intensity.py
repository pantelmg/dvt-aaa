r"""
Is clot separable from patent lumen by INTENSITY alone?

THE IDEA
--------
Detect clot after segmentation rather than alongside it. Right now clot is a
third SAM 2 object sharing one state with the vein and artery, which is why
adding it puts the vessel masks at risk. A post-hoc pass leaves them frozen.

It is also the more natural formulation. Thrombus is defined RELATIVE to the
lumen - echogenic material where anechoic blood should be. With the lumen
already outlined, that question can be asked directly, and clot becomes a
SUB-MASK of the vein by construction. That matches the measured anatomy: on
frames where both are drawn, clot is 98.6% contained in the vein at the median
and only 2.7% lies mostly outside.

All the anchoring machinery then becomes unnecessary - box-IoU >= 0.05,
containment >= 0.50, area <= 2.5x - because nothing outside the vein is ever
considered.

WHAT COULD GO WRONG, AND WHY THIS TEST EXISTS
---------------------------------------------
Intraluminal echogenicity is not specific to thrombus. Slow flow, rouleaux
formation and reverberation artefact all brighten a patent lumen, and FRESH
thrombus can be nearly anechoic - which is precisely why compression, not
brightness, is the clinical test. So clot may simply not be separable from
lumen by intensity at this resolution.

THREE QUESTIONS, IN INCREASING ORDER OF IMPORTANCE
--------------------------------------------------
1. PIXEL separability. Within a vein that contains clot, are clot pixels
   brighter than patent ones? Reported as an AUC over pixels. This is the
   easiest version of the question and an upper bound on the rest - if it
   fails, nothing downstream can work.

2. FRAME separability. Given a vein mask, can a simple statistic
   (echogenic fraction inside the lumen) tell a thrombosed frame from a
   patent one? This is what a detector would actually consume.

3. CLIP-level prediction of DVT. Does that statistic predict the label,
   subject-level, without any clot segmentation at all? If it does, it is a
   feature in its own right - and clot presence has the higher expert ceiling
   of the two (subject AUC 0.828 against the compression ratio's 0.783).

Question 3 can succeed even if 1 and 2 are mediocre, and it is the one worth
having.

NORMALISATION
-------------
Intensity is normalised against TISSUE OUTSIDE the vein on the same frame, not
against a global constant. Scanner gain, depth compensation and centre-specific
presets all shift absolute brightness, and centre identity alone predicts
thrombosis at AUC 0.709 in this cohort - an unnormalised brightness feature
would rediscover the hospital.

Both EXPERT and v8 TRACKED vein masks are scored. The expert version is the
ceiling; the tracked version is what a real pipeline would achieve.

USAGE
-----
    python scripts/test_clot_intensity.py \
        --annotations annotations_flat --dicoms dicoms_annotated \
        --sidecars dicoms_annotated --labels train.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    if len(pos) * len(neg) > 4_000_000:      # subsample huge pixel sets
        rng = np.random.default_rng(0)
        pos = rng.choice(pos, min(len(pos), 2000), replace=False)
        neg = rng.choice(neg, min(len(neg), 2000), replace=False)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def ring(mask, width=12):
    """Tissue just outside the vein - the local gain reference.

    Local rather than whole-frame: depth-gain compensation makes brightness a
    function of depth, so a reference taken from elsewhere in the image would
    encode the vessel's depth rather than its contents.
    """
    from scipy import ndimage

    grown = ndimage.binary_dilation(mask, np.ones((width * 2 + 1,) * 2, bool))
    return grown & ~ndimage.binary_dilation(mask, np.ones((5, 5), bool))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--sidecars", default=None,
                    help="folder of v8 *_darwin.json; omit to test expert masks only")
    ap.add_argument("--labels", default=None, help="train.csv, for question 3")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd

    from expert_reference import load
    from dvt import step8_compression_phase as S8

    dcms = {p.stem: p for p in Path(a.dicoms).rglob("*.dcm")}
    files = sorted(Path(a.annotations).glob("*.json"))

    pix_auc, frame_rows, clip_rows = [], [], []
    done = 0
    # Progress, flushed. Every result print in this script happens after the
    # loop, and the loop decodes one DICOM per annotation file - 1044 of them
    # by default. Without this it looks like a hang for 20 minutes.
    total = len(files) if not a.limit else min(a.limit, len(files))
    print(f"scanning up to {total} clips (decoding one DICOM each) ...",
          flush=True)

    for p in files:
        if a.limit and done >= a.limit:
            break
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        centre, group, subject, clip = parts
        dcm = dcms.get(clip)
        if dcm is None:
            continue
        ef, w, h = load(p)
        if not ef:
            continue
        has_vein = [i for i, r in ef.items() if "vein" in r]
        if not has_vein:
            continue
        try:
            frames = S8.load_clip_frames(dcm, max_frames=0, downsample=1)
        except Exception:
            continue
        if frames.shape[1:3] != (h, w):
            continue
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{total} clips   {len(pix_auc)} clot frames so far",
                  flush=True)

        af = {}
        if a.sidecars:
            sc = Path(a.sidecars) / f"{clip}_darwin.json"
            if not sc.exists():
                sc = next(Path(a.sidecars).rglob(f"{clip}_darwin.json"), None)
            if sc is not None and sc.exists():
                af, aw, ah, *_ = (*load(sc), 0)
                if (aw, ah) != (w, h):
                    af = {}

        clip_frac_exp, clip_frac_trk = [], []
        for i in has_vein:
            if i >= len(frames):
                continue
            img = frames[i].astype(np.float32)
            EV = np.logical_or.reduce(ef[i]["vein"])
            if EV.sum() < 80:
                continue
            C = (np.logical_or.reduce(ef[i]["clot"])
                 if "clot" in ef[i] else None)

            for tag, V in (("expert", EV),
                           ("tracked", (np.logical_or.reduce(af[i]["vein"])
                                        if i in af and "vein" in af[i] else None))):
                if V is None or V.sum() < 80:
                    continue
                ref = img[ring(V)]
                if ref.size < 50:
                    continue
                mu, sd = float(ref.mean()), float(ref.std()) + 1e-6
                z = (img - mu) / sd                     # local gain reference
                inside = z[V]

                # question 2/3 statistic: how much of the lumen is echogenic
                frac = float((inside > -0.5).mean())
                (clip_frac_exp if tag == "expert" else clip_frac_trk).append(frac)
                frame_rows.append({"clip": clip, "frame": i, "mask": tag,
                                   "frac": frac, "has_clot": C is not None,
                                   "group": group, "subject": subject,
                                   "centre": centre})

                # question 1: pixel separability, expert masks only
                if tag == "expert" and C is not None and (C & V).sum() > 40:
                    inC = z[C & V]
                    outC = z[V & ~C]
                    if len(inC) > 30 and len(outC) > 30:
                        pix_auc.append(auc(inC, outC))

        if clip_frac_exp:
            clip_rows.append({"clip": clip, "subject": subject, "centre": centre,
                              "group": group,
                              "frac_expert": float(np.median(clip_frac_exp)),
                              "frac_tracked": (float(np.median(clip_frac_trk))
                                               if clip_frac_trk else np.nan)})

    print(f"\n{'='*68}")
    print("CLOT vs LUMEN BY INTENSITY")
    print(f"{'='*68}")
    print(f"clips processed {done}")

    # ---- Q1
    print(f"\n1. PIXEL separability (expert masks, {len(pix_auc)} clot frames)")
    if pix_auc:
        pa = np.array(pix_auc)
        print(f"   AUC of clot pixels vs patent lumen pixels, within the vein:")
        print(f"     median {np.median(pa):.3f}   p25 {np.percentile(pa,25):.3f}"
              f"   p75 {np.percentile(pa,75):.3f}")
        print(f"     frames above 0.70: {(pa>0.70).mean():.1%}   "
              f"above 0.60: {(pa>0.60).mean():.1%}")
        if np.median(pa) < 0.60:
            print("   -> clot is NOT reliably brighter than the lumen around it.")
            print("      Fresh thrombus is near-anechoic, which is why")
            print("      compression rather than brightness is the clinical test.")
    else:
        print("   no usable clot frames")

    fr = pd.DataFrame(frame_rows)
    # ---- Q2
    print("\n2. FRAME separability (echogenic fraction inside the lumen)")
    for tag in ("expert", "tracked"):
        d = fr[fr["mask"] == tag]
        if len(d) < 50:
            continue
        withc = d[d.has_clot]["frac"].values
        without = d[~d.has_clot]["frac"].values
        print(f"   {tag:<8} n={len(d):>5}  clot frames {len(withc)}  "
              f"AUC {auc(withc, without):.3f}")

    # ---- Q3
    if clip_rows and a.labels:
        cl = pd.DataFrame(clip_rows)
        cl["y"] = (cl.group == "DVT").astype(int)
        print("\n3. CLIP / SUBJECT prediction of DVT, no clot segmentation at all")
        for col in ("frac_expert", "frac_tracked"):
            d = cl.dropna(subset=[col])
            if len(d) < 40:
                continue
            va = auc(d[d.y == 1][col], d[d.y == 0][col])
            sub = d.groupby("subject").agg(y=("y", "max"), v=(col, "max"))
            sa = auc(sub[sub.y == 1]["v"], sub[sub.y == 0]["v"])
            print(f"   {col:<14} vessel AUC {va:.3f}   subject AUC {sa:.3f}  "
                  f"(n={len(d)} clips, {len(sub)} subjects)")
        print("\n   for reference on the same cohort:")
        print("     compression ratio, v8 automated   subject 0.7045")
        print("     compression ratio, expert masks   subject 0.7831")
        print("     clot presence,     expert masks   subject 0.8281")
        cl.to_csv("clot_intensity_clips.csv", index=False)
        print("\n   wrote clot_intensity_clips.csv")

    print("\nnote: intensity is normalised against tissue in a ring OUTSIDE the")
    print("vein on the same frame. An unnormalised brightness feature would")
    print("rediscover the scanner - centre alone predicts DVT at AUC 0.709 here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
