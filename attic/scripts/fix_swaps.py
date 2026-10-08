r"""
Flag or repair vein/artery swaps in sidecars that already exist.

WHY A SEPARATE PASS
-------------------
`step16` now prevents the swap at tracking time, but every sidecar produced
before that still carries it, and re-running the pipeline costs hours. This
operates on the JSON alone: no GPU, no DICOMs, seconds per run.

THE SIGNAL
----------
Two vessels running down a limb cannot pass through each other. If the vein was
deeper than the artery at the start of a clip, it is deeper on every frame. Any
inversion is the tracker crossing its own tracks - detectable with no ground
truth at all.

MEASURED on the v8 run, 1585 expert keyframes matched to a depth-order state:

    ordering as seeded   n=1369   vein mask on the vein 82.3%   median IoU 0.673
    ordering INVERTED    n= 216   vein mask on the vein 30.1%   median IoU 0.005

Median IoU 0.005 on inverted frames. Not degraded - wrong. And 42.4% of clips
invert for 10 or more consecutive frames, covering a median 41.2% of the clip.

TWO ACTIONS, AND THEY SUIT DIFFERENT CONSUMERS
-----------------------------------------------
    --mode flag      mark inverted frames, change nothing
    --mode drop      remove both polygons on inverted frames
    --mode relabel   swap the vein and artery labels on inverted frames

Measured outcomes over the same 1585 frames:

    current                75.2% correct, 100% of frames
    drop inverted          83.7% correct,  84.4% of frames   (2.48:1)
    relabel inverted       ~78%  correct, 100% of frames

DROP is better for V-JEPA: a frame that is 52% correct is barely better than
noise as a training target, and masking 13.6% of frames out of the loss is
cheap. RELABEL is better for the annotation product: a radiologist would rather
correct a wrong label than redraw a missing polygon.

Note that relabelled frames only reach 52.3%, not the 82.3% of clean frames -
so an inversion is a swap PLUS degradation, not a clean exchange. Often both
tracks have drifted onto the same structure.

FLAG is the default, because it forecloses nothing: the per-frame state is
written into the sidecar and the consumer decides.

USAGE
-----
    python scripts/fix_swaps.py --dir sidecars_v8 --out sidecars_v8_fixed
    python scripts/fix_swaps.py --dir sidecars_v8 --mode drop --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

#: How many frames from the START establish the reference ordering.
#:
#: A FIXED SMALL COUNT, not a fraction, and not anything derived from the whole
#: clip. Compared on 1586 expert keyframes of the v8 run:
#:
#:     reference        flagged   kept on-vein   kept frac   ratio
#:     first 8 frames    15.6%       83.7%         84.4%     2.48   <- this
#:     first 20%         13.7%       82.3%         86.3%     2.34
#:     majority          11.2%       77.5%         88.8%     0.77
#:     site prior        30.3%       82.5%         69.7%     0.71
#:
#: The site prior looks more principled and is worse: it re-decides the role
#: per frame, which is the same thing that made clip-level depth voting fail
#: (75.7% -> 74.8%). Majority is worse still, because on a clip that swaps
#: early the majority IS the swapped state.
#:
#: What works is consistency with the EARLIEST frames, where the seed is
#: freshest and the tracks have not had time to cross. A fraction fails on
#: clips that invert early: one CFV clip inverts from frame 9, so 20% of 144
#: frames takes the reference from a region already mostly inverted.
ESTABLISH_N = 8

#: Consecutive inverted frames before a run is treated as a genuine swap
#: rather than jitter around a near-tie.
MIN_RUN = 5


def _role(label: str) -> str:
    low = (label or "").lower()
    if "clot" in low or "thromb" in low:
        return "clot"
    if "vein" in low:
        return "vein"
    if "artery" in low:
        return "artery"
    return "other"


def _poly_depth(poly) -> float | None:
    """Mean y of a polygon's vertices - cheap and enough for an ordering."""
    ys = [float(q["y"]) for path in poly.get("paths", []) for q in path]
    return float(np.mean(ys)) if ys else None


def analyse(doc) -> tuple[dict, dict]:
    """-> ({frame: inverted}, stats). Frames without both roles are absent."""
    depths: dict[int, dict[str, list]] = {}
    for ann in doc.get("annotations", []):
        r = _role(ann.get("name", ""))
        if r not in ("vein", "artery"):
            continue
        for k, fr in (ann.get("frames") or {}).items():
            if not str(k).lstrip("-").isdigit():
                continue
            poly = fr.get("polygon")
            if not poly:
                continue
            d = _poly_depth(poly)
            if d is None:
                continue
            depths.setdefault(int(k), {}).setdefault(r, []).append(d)

    usable = sorted(i for i, v in depths.items()
                    if "vein" in v and "artery" in v)
    if len(usable) < 10:
        return {}, {"reason": "too few frames with both roles"}

    order = {i: (np.mean(depths[i]["vein"]) > np.mean(depths[i]["artery"]))
             for i in usable}
    k = min(ESTABLISH_N, len(usable))
    established = np.mean([order[i] for i in usable[:k]]) > 0.5

    raw = {i: (order[i] != established) for i in usable}

    # keep only runs of >= MIN_RUN, so a one-frame wobble at a near-tie is not
    # treated as the tracks having crossed
    inverted, run = {}, []
    for i in usable:
        if raw[i]:
            run.append(i)
        else:
            if len(run) >= MIN_RUN:
                inverted.update({j: True for j in run})
            run = []
    if len(run) >= MIN_RUN:
        inverted.update({j: True for j in run})

    return inverted, {"frames": len(usable), "inverted": len(inverted),
                      "frac": len(inverted) / len(usable),
                      "established_vein_deeper": bool(established)}


def apply_mode(doc, inverted, mode):
    """Returns (doc, n_changed). `doc` is modified in place."""
    if not inverted:
        return doc, 0
    changed = 0
    for ann in doc.get("annotations", []):
        r = _role(ann.get("name", ""))
        if r not in ("vein", "artery"):
            continue
        frames = ann.get("frames") or {}

        if mode == "flag":
            for k in frames:
                if k.lstrip("-").isdigit() and int(k) in inverted:
                    frames[k]["depth_order_inverted"] = True
                    changed += 1
        elif mode == "drop":
            keep = {k: v for k, v in frames.items()
                    if not (k.lstrip("-").isdigit() and int(k) in inverted)}
            changed += len(frames) - len(keep)
            ann["frames"] = keep
            idxs = sorted(int(k) for k in keep if k.lstrip("-").isdigit())
            if idxs:
                ann["ranges"] = [[idxs[0], idxs[-1] + 1]]

    if mode == "relabel":
        # exchange the vein and artery polygons on inverted frames
        vein = next((a for a in doc.get("annotations", [])
                     if _role(a.get("name", "")) == "vein"), None)
        art = next((a for a in doc.get("annotations", [])
                    if _role(a.get("name", "")) == "artery"), None)
        if vein is not None and art is not None:
            vf, af = vein.get("frames") or {}, art.get("frames") or {}
            for k in list(set(vf) | set(af)):
                if not (k.lstrip("-").isdigit() and int(k) in inverted):
                    continue
                a_had, v_had = af.get(k), vf.get(k)
                if v_had is not None:
                    af[k] = v_had
                elif k in af:
                    del af[k]
                if a_had is not None:
                    vf[k] = a_had
                elif k in vf:
                    del vf[k]
                changed += 1
    return doc, changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None,
                    help="output folder; omit with --dry-run")
    ap.add_argument("--mode", default="flag",
                    choices=["flag", "drop", "relabel"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(Path(a.dir).rglob("*_darwin.json"))
    if not files:
        print(f"no sidecars under {a.dir}")
        return 1
    if not a.dry_run:
        if not a.out:
            print("--out is required unless --dry-run")
            return 1
        Path(a.out).mkdir(parents=True, exist_ok=True)

    tally = Counter()
    total_frames = total_inv = total_changed = 0
    worst = []

    for p in files:
        doc = json.loads(p.read_text())
        inverted, st = analyse(doc)
        if "reason" in st:
            tally[st["reason"]] += 1
            if not a.dry_run:
                shutil.copyfile(p, Path(a.out) / p.name)
            continue
        total_frames += st["frames"]
        total_inv += st["inverted"]
        if st["inverted"]:
            tally["clips with a swap"] += 1
            worst.append((p.name, st["frac"], st["inverted"], st["frames"]))
        else:
            tally["clean"] += 1

        doc, n = apply_mode(doc, inverted, a.mode)
        total_changed += n
        if not a.dry_run:
            (Path(a.out) / p.name).write_text(json.dumps(doc, indent=2))

    print(f"\n{'='*64}")
    print(f"DEPTH-ORDER SWAP PASS  (mode: {a.mode})")
    print(f"{'='*64}")
    print(f"sidecars            {len(files)}")
    for k, v in tally.most_common():
        print(f"  {k:<28} {v}")
    if total_frames:
        print(f"\nframes with both roles  {total_frames}")
        print(f"  inverted              {total_inv}  ({total_inv/total_frames:.1%})")
        print(f"  frame-polygons {'flagged' if a.mode=='flag' else 'changed'}  "
              f"{total_changed}")

    if worst:
        print(f"\nworst {min(12, len(worst))} clips by inverted fraction")
        for nm, fr, inv, n in sorted(worst, key=lambda x: -x[1])[:12]:
            print(f"  {nm[:48]:<48} {fr:>6.1%}  {inv}/{n}")

    print("\nexpected effect, measured on 1585 expert keyframes of the v8 run:")
    print("  current            75.2% correct on 100% of frames")
    print("  --mode drop        83.7% correct on  84.4% of frames  (2.48:1)")
    print("  --mode relabel     ~78%  correct on 100% of frames")
    print("\ndrop suits V-JEPA (mask the loss); relabel suits the annotation")
    print("product (correcting a label beats redrawing a polygon).")
    if a.dry_run:
        print("\n(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
