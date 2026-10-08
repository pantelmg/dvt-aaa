"""
Run the auto-annotator over a folder of compression-ultrasound DICOMs.

For each clip: read the anatomical site from the filename, detect the vein and
artery as a PAIR on several candidate frames, then segment both on EVERY frame
independently with MedSAM2 - no temporal propagation. Write the proposed
polygons as a `<stem>_darwin.json` sidecar that US-DICOMizer can reimport, and
record the vein area curve and its compression ratio in the report.

The radiologist's role is to correct proposals, not to draw from scratch.

PIPELINE
    filename  -> site tag                         step10_auto_seed
    site      -> which vessel is the vein         step10_auto_seed (priors)
    frames    -> vein/artery boxes, every frame   step13_detector_seed
    boxes     -> masks, every frame               step16_per_frame + MedSAM2
    masks     -> polygons + compression ratio     step7, step9_measures

Standard run (this is the v14 configuration that produced the reported
numbers - subject AUC 0.7221 train, vessel AUC 0.6233 test):

    python scripts/run_autoannotate.py \
        --medsam2 MedSAM2/checkpoints/MedSAM2_latest.pt \
        --detector checkpoints/vessel_detector_9926.pth \
        --propose-artery --multi-vein-target deep \
        --no-depth-order --no-redecide-multivein \
        --dir dicoms_annotated --auto --device cuda \
        --downsample 1 --max-frames 0 --all-frames --quiet-skips \
        --out report_v14.json

Both --medsam2 and --detector are required: the heuristic seeder and the
MedSAM1 tracker they replaced are in `attic/`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt import step7_write_annotations as W
from dvt import step8_load_clips as S8
from dvt import step9_measures as M9
from dvt import step10_auto_seed as S10
from dvt import step13_detector_seed as S13
from dvt import step14_medsam2 as S14
from dvt import step16_per_frame as S16

#: Flags removed when the pipeline was pruned to the per-frame path. Named
#: explicitly so an old command line fails with an explanation rather than an
#: argparse "unrecognized arguments".
RETIRED_FLAGS = {
    "--per-frame": "now the only mode; drop the flag",
    "--isolate": "occlusion experiment, see attic/dvt/step15_isolated_track.py",
    "--isolate-stride": "occlusion experiment",
    "--isolate-collapse-frac": "occlusion experiment",
    "--isolate-det-score": "occlusion experiment",
    "--propose-clot": "clot is out of scope; see attic/dvt/step12_clot_detect.py",
    "--clot-det-score": "clot is out of scope",
    "--max-clot-ratio": "clot is out of scope",
    "--min-clot-containment": "clot is out of scope",
    "--reprompt-every": "propagation only; per-frame segmentation has no memory",
    "--medsam2-chunk": "propagation only",
    "--medsam2-max-area-ratio": "propagation only",
    "--exclusive": "propagation only",
    "--no-negative-points": "propagation only",
    "--keep-overlap-frames": "folded into --max-containment",
    "--box": "manual seeding removed; --auto with a detector is the only mode",
    "--label": "manual seeding removed; labels come from the site priors",
    "--from-sidecar": "sidecar seeding removed; see attic/scripts/run_autoannotate.py",
    "--file-list": "removed with filenames_40.txt; use --limit or a smaller --dir",
    "--checkpoint": "MedSAM1 path removed; --medsam2 is required",
    "--img-size": "MedSAM1 path removed",
    "--track": "MedSAM1 path removed",
    "--reseed": "MedSAM1 path removed",
    "--stretch": "MedSAM1 path removed",
    "--no-mask-prompt": "MedSAM1 path removed",
    "--no-lumen-refine": "MedSAM1 path removed",
    "--max-jump": "MedSAM1 path removed",
    "--max-area-ratio": "MedSAM1 path removed",
    "--reacquire-after": "MedSAM1 path removed",
}


def _reject_retired(argv):
    hit = [a for a in argv if a.split("=")[0] in RETIRED_FLAGS]
    if not hit:
        return
    print("These flags no longer exist:\n")
    for a in hit:
        k = a.split("=")[0]
        print(f"    {k:28s} {RETIRED_FLAGS[k]}")
    print("\nSee README.md for the current command line.")
    sys.exit(2)


def process(dcm: Path, seeds, args, medsam2):
    """seeds: list of (box, label). Every seed is tracked and written."""
    frames = S8.load_clip_frames(dcm, max_frames=args.max_frames,
                                 downsample=args.downsample)
    per_seed = []
    m2diag = None

    boxes = {i + 1: box for i, (box, _lbl) in enumerate(seeds)}
    # The role is what the pipeline reasons about, not the label string. Label
    # text varies by site ("Popliteal Vein # 4", "Common Femoral Vein"); the
    # role does not.
    roles = {i + 1: S14.role_of(lbl) for i, (_box, lbl) in enumerate(seeds)}

    # NO PROPAGATION. Each frame is segmented as its own one-frame video.
    # Measured on 211 frames with expert boxes: single-frame collision 5.2%
    # against 28.8% propagated, and median IoU 0.815 against the 0.507 the best
    # propagated run achieved. The encoder was never the problem; the memory
    # bank was. See step14_medsam2's docstring for the full diagnosis.
    tracked, m2diag = S16.segment_per_frame(
        medsam2, frames, boxes, roles, args._detector, args._det_size,
        device=args.device, det_score=args.det_score,
        site=args._site, smooth=args.box_smooth,
        verbose=not args.quiet_skips,
        multi_vein_target=args.multi_vein_target,
        depth_order=not args.no_depth_order,
        redecide_multivein=not args.no_redecide_multivein)
    for i, (_box, label) in enumerate(seeds):
        areas, masks = tracked[i + 1]
        quality = np.array([M9.mask_quality(m, frames[k])
                            for k, m in enumerate(masks)])
        per_seed.append((label, areas, masks, quality))

    # A mask that cannot be a single vessel is wrong regardless of which
    # vessel it is on. This is the only check that identifies WHICH of two
    # overlapping masks is broken - a figure-8 spanning both vessels is wrong
    # and the round mask beside it is right. 0.0% false positives on 6384
    # clean masks, but only ~13% sensitivity: the common failure is two ROUND
    # masks on one vessel, which shape cannot see.
    n_shape_rejected, shape_reasons = 0, {}
    if not args.no_shape_penalty:
        per_seed, n_shape_rejected, shape_reasons = S14.constrain_shape(
            per_seed, min_artery_solidity=args.min_artery_solidity)

    # The VEIN drives the compression ratio. This used to be per_seed[0] -
    # whichever seed happened to be first - so on any clip seeded artery-first
    # the reported clip ratio described the artery. score_dvt reads the
    # vessel-level ratio, so the two disagreed silently.
    vein_entry = next((e for e in per_seed if S14.role_of(e[0]) == "vein"),
                      per_seed[0] if per_seed else None)
    areas, masks = vein_entry[1], vein_entry[2]

    # true source frame number for each processed frame - NOT i * stride
    src_idx = S8.frame_indices(dcm, args.max_frames)

    if args.all_frames or args.n_polygons <= 0:
        keep = list(range(len(frames)))
    else:
        keep = np.linspace(0, len(frames) - 1,
                           min(args.n_polygons, len(frames)))
        keep = sorted(set(keep.round().astype(int).tolist()))

    # Frames where the vein and artery masks still overlap. Per-frame
    # segmentation cut this from 28.8% of frames to 5.2%, so it is a residual
    # rather than the dominant failure - but where it happens the frame is
    # still wrong, so propose nothing on it. Same rule as --max-contrast: the
    # radiologist draws one polygon rather than untangling a misplaced one.
    # OFF by default (--max-containment 0): the remaining 5.2% has not been
    # shown to hurt the measurement, and dropping frames costs coverage.
    bad_frames = set()
    if m2diag and args.max_containment > 0:
        bad_frames = {i for i, c in enumerate(m2diag.get("containment", []))
                      if c >= args.max_containment}

    polys = []
    n_lowq = 0
    n_overlap = 0
    for lbl, ar, mk, q in per_seed:
        for i in keep:
            if i in bad_frames:
                n_overlap += 1
                continue
            # Skip frames where the mask has slipped onto tissue. Proposing
            # nothing is better than proposing something wrong: the radiologist
            # draws one polygon rather than untangling a misplaced one.
            if args.max_contrast > 0 and not np.isnan(q[i]) and q[i] > args.max_contrast:
                n_lowq += 1
                continue
            pts = W.mask_to_polygon(mk[i], max_points=args.max_points)
            if pts is None:
                continue
            polys.append({"label": lbl, "frame": int(src_idx[i]),
                          "points": pts * args.downsample})

    import pydicom
    ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
    payload = W.build_sidecar(
        dcm_filename=dcm.name,
        width=int(ds.Columns), height=int(ds.Rows),
        frame_count=int(getattr(ds, "NumberOfFrames", 1) or 1),
        polygons=polys,
        clinician_name=args.clinician, clinician_email=args.email,
    )
    out = W.write_sidecar(dcm, payload)

    ratio = M9.compression_ratio(areas)
    return {
        "clip": dcm.stem, "n_frames_processed": len(frames),
        "vessels": [{"label": l,
                     "ratio": (round(M9.compression_ratio(a), 4)
                               if M9.compression_ratio(a) is not None else None),
                     "ratio_clean": (
                         round(_ratio_excluding(a, bad_frames), 4)
                         if _ratio_excluding(a, bad_frames) is not None
                         else None),
                     "area_max_px": int(a.max()),
                     "area_min_px": int(a[a > 0].min()) if (a > 0).any() else 0,
                     "frames_valid": int((a > 0).sum()),
                     "frames_blank": int((a == 0).sum()),
                     "median_contrast": round(float(np.nanmedian(q)), 3),
                     "frames_low_quality": int(np.nansum(q > args.max_contrast))}
                    for l, a, _, q in per_seed],
        "polygons_skipped_low_quality": n_lowq,
        "shape_frames_rejected": n_shape_rejected,
        "shape_reasons": shape_reasons or None,
        "polygons_skipped_overlap": n_overlap,
        # EVERY scalar diagnostic, not a hardcoded subset. The previous version
        # copied five named keys, so `box_missing`, `order_rejected` and the
        # per-object loss counters never reached the report - and the largest
        # coverage bucket (31.1% of no-polygon frames) could not be attributed
        # without them. The per-frame `containment` series is kept because it
        # is the only way to ask whether a guard RECOVERS rather than merely
        # fires; other list-valued entries are dropped to keep the file small.
        "overlap": ({**{k: v for k, v in m2diag.items()
                        if not isinstance(v, (list, dict))},
                     "containment": m2diag.get("containment")}
                    if m2diag else None),
        # The ratio over CLEAN frames only. On the previous run the
        # contaminated frames sat near the vein's area maximum, so including
        # them deflated min/max - median 0.170 raw against 0.462 cleaned - and
        # a deflated ratio reads as a freely compressible vein.
        "compression_ratio_clean": (
            round(_ratio_excluding(areas, bad_frames), 4)
            if _ratio_excluding(areas, bad_frames) is not None else None),
        "area_max_px": int(areas.max()),
        "area_min_px": int(areas[areas > 0].min()) if (areas > 0).any() else 0,
        "compression_ratio": round(ratio, 4) if ratio is not None else None,
        "relaxed_idx": int(areas.argmax()), "peak_idx": int(areas.argmin()),
        "n_polygons_written": len(polys), "sidecar": str(out),
        "areas": areas.tolist(),
    }


def _ratio_excluding(areas, bad):
    """min/max over frames that are neither blank nor contaminated.

    A zero area means "no measurement", never "the vein closed" - the step9
    bug where one rejected frame reported perfect compressibility. Frames in
    `bad` are dropped for the same reason: a vein mask that has merged with
    the artery is not a measurement of the vein.
    """
    vals = [a for i, a in enumerate(np.asarray(areas)) if a > 0 and i not in bad]
    if len(vals) < 2:
        return None
    return float(min(vals)) / float(max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcm", help="single DICOM")
    ap.add_argument("--dir", help="folder of DICOMs")
    ap.add_argument("--auto", action="store_true",
                    help="find the vein automatically - no radiologist seed needed")
    ap.add_argument("--medsam2", default=None,
                    help="path to MedSAM2_latest.pt. Replaces the hand-written "
                         "box tracking with learned memory propagation.")
    ap.add_argument("--medsam2-cfg", default="configs/sam2.1_hiera_t512.yaml",
                    help="MedSAM2 model config")
    ap.add_argument("--max-containment", type=float, default=0.0,
                    help="vein/artery mask overlap at which a frame is treated "
                         "as contaminated: re-prompt and write no polygon. "
                         "DEFAULT OFF. At 0.30 it suppressed 48%% of the frames "
                         "experts had annotated (up from 27%%), destroying 94 "
                         "good polygons to remove 84 junk ones. Use "
                         "attic/scripts/qc_overlap.py to MEASURE overlap "
                         "instead.")
    ap.add_argument("--no-arbitrate", action="store_true",
                    help="do NOT re-decide vein/artery from anatomy at seed "
                         "time. Arbitration is ON by default: the site priors "
                         "are 95-96%% accurate at CFV/FV/PV and 85%% at GS, "
                         "against a detector that put the vein label on the "
                         "artery 13-27%% of the time.")
    ap.add_argument("--arbitrate-margin", type=float, default=0.5,
                    help="refuse to swap when the two boxes are closer than "
                         "this (depth difference as a fraction of box height, "
                         "or area difference as a fraction of the larger). A "
                         "swap decided on a coin flip is worse than leaving "
                         "the detector's own call alone. Swept against expert "
                         "masks on 267 clips: 0.15 gives 83.9%% (worse than "
                         "doing nothing), 0.5 gives 87.3%%, 1.2 gives 85.0%%.")
    ap.add_argument("--no-shape-penalty", action="store_true",
                    help="do not blank masks that cannot be a single vessel. "
                         "The penalty is ON by default and costs nothing: 0.0%% "
                         "false positives over 6384 clean masks. It only "
                         "reaches ~13%% of collisions though - the common "
                         "failure is two ROUND masks on one vessel.")
    ap.add_argument("--min-artery-solidity", type=float, default=0.90,
                    help="reject an ARTERY mask less convex than this. Expert "
                         "arteries sit at p1 = 0.962, p0.5 = 0.944. There is "
                         "deliberately NO vein equivalent: expert veins reach "
                         "p1 = 0.382 because a compressed vein is legitimately "
                         "a slit, so a vein bound would reject exactly the "
                         "frames the ratio is measured from.")
    ap.add_argument("--no-depth-order", action="store_true",
                    help="disable the depth-order constraint, restoring v8 "
                         "behaviour at CFV/FV/PV. The constraint cut track "
                         "inversions from 18.7%% to 2.1%% and left mask IoU "
                         "unchanged (0.7210 -> 0.7233 on non-GS frames that "
                         "have a polygon), but cost 1.6 points of coverage "
                         "(11.7%% -> 13.3%% missing). It was never ablated on "
                         "its own, so this flag exists to do that.")
    ap.add_argument("--no-redecide-multivein", action="store_true",
                    help="at multi-vein sites, seed once and follow instead of "
                         "re-deciding each frame. Restores v9 behaviour at GS.")
    ap.add_argument("--multi-vein-target", default="gsv",
                    choices=["gsv", "deep"],
                    help="which vein to track at multi-vein sites (GS). 'gsv' "
                         "tracks the vessel the filename names. 'deep' tracks "
                         "the deep-system vein sharing the field - the common "
                         "femoral - which the deepest-vein rule identifies "
                         "99.3%% of the time and which segments far better "
                         "(GS mask IoU 0.626 against 0.246-0.275). It is also "
                         "the clinically relevant vessel: saphenous thrombosis "
                         "is a different entity from DVT, and every GS subject "
                         "here has deep-vein clips anyway. With 'deep' the "
                         "POLYGON LABEL follows the vessel, not the filename.")
    ap.add_argument("--box-smooth", type=int, default=2,
                    help="moving-MEDIAN window, in frames either side, applied "
                         "to the detector boxes before prompting. Median not "
                         "mean, so a single frame where the detector jumps to "
                         "the wrong structure cannot drag its neighbours. 0 "
                         "disables.")
    ap.add_argument("--detector", default=None,
                    help="path to vessel_detector.pth from train_detector.py. "
                         "Replaces the heuristic seeder, which picked the wrong "
                         "structure on 66%% of clips.")
    ap.add_argument("--det-score", type=float, default=0.5,
                    help="minimum detector confidence to accept a box")
    ap.add_argument("--include-stills", action="store_true",
                    help="also process clips whose DICOM holds a single frame. "
                         "A compression ratio is undefined on one frame, so "
                         "these are skipped by default. The decision uses the "
                         "frame count, NOT the filename's 'r' suffix - the "
                         "suffix is wrong on 30 of the 1044 annotated clips, "
                         "in both directions.")
    ap.add_argument("--quiet-skips", action="store_true",
                    help="suppress per-clip SKIP lines, print a summary instead")
    ap.add_argument("--site", default=None,
                    help="CFV|GS|FV|PV, if the filename carries no site tag")
    ap.add_argument("--propose-artery", action="store_true",
                    help="also propose the artery, so the radiologist can "
                         "verify the vein/artery assignment at a glance")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="frames to process per clip; 0 means all of them")
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--n-polygons", type=int, default=8,
                    help="proposed polygons per clip per vessel; "
                         "ignored when --all-frames is set")
    ap.add_argument("--all-frames", action="store_true",
                    help="write a polygon on EVERY processed frame. Combine "
                         "with --max-frames 0 to process the whole clip.")
    ap.add_argument("--max-points", type=int, default=40)
    ap.add_argument("--skip-weak", action="store_true",
                    help="skip clips where no convincing vessel was found, "
                         "instead of seeding on noise")
    ap.add_argument("--max-contrast", type=float, default=0.75,
                    help="skip proposing a polygon when mask interior / "
                         "surrounding tissue exceeds this (mask has slipped off "
                         "the lumen). 0 disables the check.")
    ap.add_argument("--clinician", default="")
    ap.add_argument("--email", default="")
    ap.add_argument("--out", default="autoannotate_report.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N clips (use while testing)")
    _reject_retired(sys.argv[1:])
    a = ap.parse_args()

    if not a.medsam2:
        sys.exit("--medsam2 is required (path to MedSAM2_latest.pt).")
    if not a.detector:
        sys.exit("--detector is required (path to the trained vessel detector).\n"
                 "The heuristic seeder it replaced reached IoU>0.5 on 6% of "
                 "clips and is in attic/dvt/step10_auto_seed.py.")
    if not a.auto:
        sys.exit("--auto is required: detector seeding is the only mode. The "
                 "manual --box and\n--from-sidecar seeding paths are in "
                 "attic/scripts/run_autoannotate.py.")

    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            sys.exit(
                "--device cuda requested but torch has no CUDA support.\n"
                "On Windows, `pip install torch` installs the CPU-only build.\n"
                "Reinstall with the CUDA wheel index:\n"
                "    pip uninstall -y torch torchvision\n"
                "    pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu124\n"
                "Then check: python -c \"import torch;print(torch.cuda.is_available())\"")

    if a.max_frames is not None and a.max_frames <= 0:
        a.max_frames = None                    # None = every frame

    medsam2 = S14.load_predictor(a.medsam2, a.medsam2_cfg, a.device)
    print(f"MedSAM2 video predictor loaded from {a.medsam2}")

    detector, det_size, det_names = S13.load_detector(a.detector, a.device)
    print(f"detector loaded from {a.detector} (input {det_size}px, "
          f"classes {sorted(det_names.values())})")

    targets = []
    if a.dcm:
        targets = [Path(a.dcm)]
    elif a.dir:
        targets = sorted(Path(a.dir).glob("*.dcm"))
        if a.limit:
            targets = targets[:a.limit]
    else:
        sys.exit("need --dcm or --dir")

    if not targets:
        sys.exit("no DICOMs matched")

    from collections import Counter
    skip_reasons = Counter()
    arbitration = Counter()
    tag_disagrees = Counter()
    results, skipped = [], 0
    for i, dcm in enumerate(targets, 1):
        if a.auto:
            tag = S10.parse_filename_tag(dcm.name)
            site = tag["site"] or a.site

            # A compression ratio needs at least two frames, so single-frame
            # studies are skipped. The decision is made on the DICOM HEADER,
            # not on the filename.
            #
            # A trailing "r" in the US-DICOMizer tag is supposed to mark a
            # random still, but measured over the 1044 annotated clips it
            # disagrees with the header in BOTH directions:
            #
            #     frame_count == 1, no "r" suffix      18 clips
            #     "r" suffix, frame_count 55-213       12 clips
            #
            # Trusting the suffix therefore discarded 12 real compression
            # cines - 3.7% of the 325 that were measured, 5 DVT and 7 non-DVT,
            # so lost statistical power rather than bias. The 18 in the other
            # direction were harmless: `step9_measures.compression_ratio`
            # returns None below two valid frames and `score_dvt` masks that
            # out, so they never reached a number. Coverage was the real cost.
            n_src = S8.frame_count(dcm)
            if tag["still"] != (n_src <= 1):
                tag_disagrees[("single, no r suffix" if n_src <= 1
                               else "r suffix, multi-frame")] += 1
            if n_src <= 1 and not a.include_stills:
                skipped += 1
                skip_reasons["single frame (ratio undefined)"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - single "
                          f"frame ({tag['tag']}); pass --include-stills to keep it")
                continue

            if site is None:
                skipped += 1
                skip_reasons[f"unrecognised site tag {tag['tag']!r}"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - site tag "
                          f"{tag['tag']!r} not recognised; pass --site")
                continue
            fr = S8.load_clip_frames(dcm, max_frames=24, downsample=a.downsample)

            # Trained detector: 74% vein detection at IoU>=0.5 on its own
            # held-out set, against the heuristic seeder's 6% of clips.
            # Seeding was the measured bottleneck, not segmentation.
            # detector_seed selects the vein and artery as a PAIR and assigns
            # the roles by anatomy itself: the two vessels are adjacent, barely
            # overlapping and within ~4x in size, and a spurious box rarely
            # satisfies all three against a real one.
            #
            # The seed frame is NOT carried forward. Per-frame segmentation
            # re-detects on every frame, so the only thing seeding decides is
            # WHICH vessel is the vein - a decision, not a position.
            cands, _seed_frame = S13.detector_seed(
                detector, det_size, fr, device=a.device,
                min_score=a.det_score,
                site=(None if a.no_arbitrate else site),
                class_names=det_names,
                multi_vein_target=a.multi_vein_target)
            # When the deep vein is targeted the filename no longer describes
            # what is tracked, so the LABEL must follow the vessel. v8 tracked
            # the common femoral vein in GS clips and called it "Great
            # Saphenous Vein" - correct masks, wrong name.
            label_override = (S10.DEEP_VEIN_LABEL[site]
                              if a.multi_vein_target == "deep"
                              and site in S10.DEEP_VEIN_LABEL else None)
            arbitration[
                "pair" if sum(1 for c in cands
                              if c.role in ("vein", "artery")) == 2
                else ("single" if cands else "none")] += 1

            if not cands and a.skip_weak:
                skipped += 1
                skip_reasons["detector found no vessel"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - "
                          "detector found no vessel above --det-score")
                continue

            vein_lbl, art_lbl = label_override or S10.SITE_LABELS[site]
            seeds = []
            for c in cands:
                if c.role == "vein":
                    seeds.append((c.box, vein_lbl))
                elif c.role == "artery" and a.propose_artery:
                    seeds.append((c.box, art_lbl))
            if not seeds:
                skipped += 1
                skip_reasons["no vessel candidate"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - no vessel "
                          "candidate found (image likely shadowed)")
                continue
        try:
            a._detector, a._det_size = detector, det_size
            a._site = site
            r = process(dcm, seeds, a, medsam2)
            results.append(r)
            vs = " ".join(f"{v['label'].split('#')[0].strip()}="
                          f"{v['ratio'] if v['ratio'] is not None else float('nan'):.3f}"
                          f"({v['frames_valid']}/{v['frames_valid']+v['frames_blank']})"
                          for v in r["vessels"])
            print(f"[{i}/{len(targets)}] {dcm.stem[:8]} {vs} "
                  f"-> {r['n_polygons_written']} polygons")
        except Exception as e:
            print(f"[{i}/{len(targets)}] {dcm.stem[:8]} ERROR {e!r}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out} ({len(results)} processed, {skipped} skipped)")
    if skip_reasons:
        print("skip reasons:")
        for r, n in skip_reasons.most_common():
            print(f"  {n:5d}  {r}")
    if tag_disagrees:
        print("filename tag disagreed with the DICOM frame count:")
        for r, n in tag_disagrees.most_common():
            print(f"  {n:5d}  {r}")
        print("  The header decided. A 'r'-suffixed clip holding many frames is "
              "a real\n  compression cine and was processed; a single-frame clip "
              "without the suffix\n  has no defined ratio and was skipped.")
    if arbitration:
        tot = sum(arbitration.values())
        print("vessel seeding:")
        for r, n in arbitration.most_common():
            print(f"  {n:5d}  {r}")
        pr = arbitration.get("pair", 0)
        print(f"  both vessels found on {pr}/{tot} clips ({pr/max(tot,1):.1%}). "
              f"A clip seeded from one vessel has no artery track, so no "
              f"collision can be detected on it later.")


if __name__ == "__main__":
    main()
