r"""
Is the MEMORY BANK the cause, or the encoder?

THE QUESTION
------------
Six inference-time interventions failed to move vein/artery overlap out of a
24-31% band:

    argmax exclusivity     IoU 0.564 -> 0.470, vein-on-artery frames DOUBLED
    suppress collided      94 good polygons killed to remove 84 junk
    subtract + largest     IoU 0.298 -> 0.237 on collided frames
    negative point prompts overlap 0.239 -> 0.238
    occlusion (step 15)    overlap 0.238 -> 0.275
    seed-frame correction  overlap 0.311 -> 0.288

Two of those were genuine bugs worth fixing on their own (34.7% of seed boxes
were applied at the wrong frame; the rival role was re-decided every frame at
87% accuracy). Neither was the cause.

The working hypothesis is that MedSAM2's memory bank re-identifies the vein
onto the artery: when the vein compresses, its match to the STORED embedding of
a round anechoic vessel degrades while the artery still matches it perfectly.
That hypothesis has never been tested directly, and it should be before anyone
spends a GPU-week on fine-tuning or a new architecture.

THE TEST
--------
Segment each frame with NO propagation - prompt the frame directly with the
expert's own boxes for that frame, one object at a time, and measure whether
the resulting vein and artery masks overlap.

Expert boxes are used deliberately. This is not a test of the detector; it
isolates one variable. If the boxes are right and the masks still collide, the
fault is in how the image encoder represents two adjacent anechoic circles, and
neither a per-frame model nor a memory fine-tune will help. If they do NOT
collide, the memory bank is the culprit and a per-frame multi-class segmenter -
where a softmax makes the collision structurally impossible - is the answer.

WHAT COUNTS AS AN ANSWER
------------------------
Compare the single-frame collision rate against the 28.8% of propagated frames
in the v7 run, measured on the SAME clips.

    single-frame collision << 28.8%   -> memory bank. Go per-frame.
    single-frame collision ~= 28.8%   -> encoder. Neither fix helps.

USAGE
-----
    python scripts/test_memory_hypothesis.py \
        --annotations annotations_flat --dicoms dicoms_annotated \
        --medsam2 MedSAM2/checkpoints/MedSAM2_latest.pt \
        --limit 40 --device cuda
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def containment(a, b) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 or sb == 0:
        return 0.0
    return int((a & b).sum()) / min(sa, sb)


def iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--medsam2", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--limit", type=int, default=40,
                    help="clips to test. 40 gives ~230 co-annotated frames, "
                         "enough to separate 28.8% from anything below ~15%.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="containment at which a frame counts as collided, "
                         "matching qc_overlap.py")
    a = ap.parse_args()

    from expert_reference import load
    from dvt import step14_medsam2_track as S14
    from dvt import step8_compression_phase as S8

    predictor = S14.load_predictor(
        a.medsam2, a.cfg or S14.DEFAULT_CFG, a.device)

    dicoms = {p.stem: p for p in Path(a.dicoms).rglob("*.dcm")}
    files = sorted(Path(a.annotations).glob("*.json"))

    per_frame, done = [], 0
    for p in files:
        if done >= a.limit:
            break
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        clip = parts[3]
        dcm = dicoms.get(clip)
        if dcm is None:
            continue
        ef, w, h = load(p)
        both = sorted(i for i, r in ef.items()
                      if "vein" in r and "artery" in r)
        if len(both) < 2:
            continue

        try:
            frames = S8.load_clip_frames(dcm, max_frames=0, downsample=1)
        except Exception as e:
            print(f"  skip {clip}: {e}")
            continue
        if frames.shape[1:3] != (h, w):
            continue

        for idx in both:
            if idx >= len(frames):
                continue
            EV = np.logical_or.reduce(ef[idx]["vein"])
            EA = np.logical_or.reduce(ef[idx]["artery"])
            if EV.sum() < 60 or EA.sum() < 60:
                continue

            # ONE frame, written as a one-frame video. No propagation, no
            # memory to carry anything forward - the whole point of the test.
            tmp = Path(tempfile.mkdtemp(prefix="single_"))
            try:
                S14.write_frames(frames[idx:idx + 1], tmp)
                masks = {}
                for role, gt in (("vein", EV), ("artery", EA)):
                    state = predictor.init_state(video_path=str(tmp))
                    predictor.reset_state(state)
                    predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=1,
                        box=np.asarray(bbox(gt), dtype=np.float32))
                    import torch
                    with torch.inference_mode():
                        for _i, _ids, logits in predictor.propagate_in_video(state):
                            masks[role] = (logits[0] > 0.0).cpu().numpy()[0]
                            break
                    predictor.reset_state(state)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            if "vein" not in masks or "artery" not in masks:
                continue
            V, A = masks["vein"], masks["artery"]
            if V.sum() < 20 or A.sum() < 20:
                continue
            per_frame.append({
                "clip": clip, "frame": idx,
                "containment": containment(V, A),
                "iou_vein": iou(V, EV), "iou_artery": iou(A, EA),
                # did each mask land on its OWN vessel at all?
                "vein_on_vein": iou(V, EV) > iou(V, EA),
                "artery_on_artery": iou(A, EA) > iou(A, EV),
            })
        done += 1

    if not per_frame:
        print("no usable frames")
        return 1

    import pandas as pd
    df = pd.DataFrame(per_frame)

    # Write BEFORE reporting. The inference is the expensive part, and an
    # exception in the summary should not throw it away.
    df.to_csv("single_frame_test.csv", index=False)

    # NOTE: `df.clip` is pandas' DataFrame.clip METHOD, not the column - it
    # silently returns a bound method and every attribute access on it fails.
    # Bracket access throughout for this column.
    coll = (df["containment"] >= a.threshold).mean()

    print(f"\n{'='*66}")
    print("SINGLE-FRAME TEST - expert boxes, no propagation")
    print(f"{'='*66}")
    print(f"clips {df['clip'].nunique()}   frames {len(df)}")
    print()
    print(f"  collided (containment >= {a.threshold})   {coll:.1%}")
    print(f"  median containment                  {df['containment'].median():.3f}")
    print(f"  fully merged (>= 0.9)               {(df['containment']>=0.9).mean():.1%}")
    print()
    print(f"  vein mask on the vein               {df['vein_on_vein'].mean():.1%}")
    print(f"  artery mask on the artery           {df['artery_on_artery'].mean():.1%}")
    print(f"  median IoU  vein {df['iou_vein'].median():.3f}   "
          f"artery {df['iou_artery'].median():.3f}")
    print()
    print("  reference: the v7 PROPAGATED run collided on 28.8% of frames")
    print()
    if coll < 0.15:
        print("  VERDICT: the memory bank is the culprit. Prompted directly,")
        print("  the encoder keeps the two vessels apart. A per-frame")
        print("  multi-class segmenter makes the collision structurally")
        print("  impossible via softmax and is worth building.")
    elif coll > 0.24:
        print("  VERDICT: the ENCODER is the culprit - it cannot separate two")
        print("  adjacent anechoic circles even with correct boxes and no")
        print("  propagation. Neither a per-frame model nor a memory")
        print("  fine-tune will help; the representation is the limit.")
    else:
        print("  VERDICT: ambiguous. Both mechanisms contribute. A per-frame")
        print("  model would help but not solve it - budget for that before")
        print("  committing.")
    print()
    print("wrote single_frame_test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
