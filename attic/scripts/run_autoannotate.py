"""
Run the auto-annotator on a clip, or a whole folder.

Workflow:
  1. Radiologist draws ONE polygon on ONE frame in US-DICOMizer and exports.
  2. This script reads that seed, derives a box, and propagates MedSAM across
     every frame of the clip.
  3. It writes a `<stem>_darwin.json` sidecar containing the proposed polygons,
     plus the area curve and compression ratio.
  4. Radiologist reopens the clip in US-DICOMizer, corrects, re-exports.

Usage:
    # single clip with a manual box (x0,y0,x1,y1 in ORIGINAL pixel coords)
    python scripts/run_autoannotate.py --dcm clip.dcm --box 25,65,95,115 \
        --checkpoint checkpoints/medsam_vit_b.pth --label "Popliteal Vein # 4"

    # whole folder, seeding each clip from its existing sidecar polygon
    python scripts/run_autoannotate.py --dir data/01_dicom --from-sidecar \
        --checkpoint checkpoints/medsam_vit_b.pth

On a 6 GB card use the default --img-size 512. Add --device cuda.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt import step2_json_to_masks as S2
from dvt import step7_write_annotations as W
from dvt import step8_compression_phase as S8
from dvt import step9_medsam_infer as M9
from dvt import step10_auto_seed as S10
from dvt import step13_detector_seed as S13


def process(dcm: Path, seeds, sam, img_size, args, medsam2=None):
    """seeds: list of (box, label). Every seed is tracked and written."""
    frames = S8.load_clip_frames(dcm, max_frames=args.max_frames,
                                 downsample=args.downsample)
    per_seed = []
    m2diag = None

    if medsam2 is not None:
        # One propagation pass for every vessel: MedSAM2 tracks multiple object
        # ids natively, and the image encoder runs once per frame regardless.
        from dvt import step14_medsam2_track as S14
        boxes = {i + 1: box for i, (box, _lbl) in enumerate(seeds)}
        # Roles drive mutual exclusion. Without them S14 cannot tell which ids
        # are allowed to overlap, and clot - which legitimately sits inside the
        # vein - would be torn away from the lumen by the argmax.
        roles = {i + 1: S14.role_of(lbl) for i, (_box, lbl) in enumerate(seeds)}
        if args.per_frame and args._detector is not None:
            # No propagation at all. Measured on 211 frames with expert boxes:
            # single-frame collision 5.2% against 28.8% propagated, and median
            # IoU 0.815 against the 0.507 the best propagated run achieved.
            # The encoder was never the problem; the memory bank was.
            from dvt import step16_per_frame as S16
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
            tracked = None
        elif args.isolate and args._detector is not None:
            # Hide the rival vessel from each tracker instead of trying to
            # separate the masks afterwards. Four post-hoc repairs were
            # measured and none worked; see step15's docstring.
            from dvt import step15_isolated_track as S15
            tracked, m2diag = S15.track_isolated(
                medsam2, frames, boxes, roles, args._detector, args._det_size,
                device=args.device, det_score=args.det_score,
                site=args._site, prompt_frame=args._seed_frame,
                stride=args.isolate_stride, verbose=not args.quiet_skips,
                collapse_frac=args.isolate_collapse_frac,
                bidirectional=True,
                rival_det_score=args.isolate_det_score)
            for i, (_box, label) in enumerate(seeds):
                areas, masks = tracked[i + 1]
                quality = np.array([M9.mask_quality(m, frames[k])
                                    for k, m in enumerate(masks)])
                per_seed.append((label, areas, masks, quality))
            tracked = None
        else:
            tracked, m2diag = S14.track_multi(
            medsam2, frames, boxes,
            prompt_frame=args._seed_frame, bidirectional=True,
            detector=args._detector, det_size=args._det_size,
            device=args.device, reprompt_every=args.reprompt_every,
            max_area_ratio=args.medsam2_max_area_ratio,
            det_score=args.det_score, roles=roles, site=args._site,
            exclusive=args.exclusive,
            max_containment=args.max_containment,
            chunk=args.medsam2_chunk,
            negative_points=not args.no_negative_points, return_diag=True)
        if tracked is not None:
            for i, (_box, label) in enumerate(seeds):
                areas, masks = tracked[i + 1]
                quality = np.array([M9.mask_quality(m, frames[k])
                                    for k, m in enumerate(masks)])
                per_seed.append((label, areas, masks, quality))
    else:
        for box, label in seeds:
            areas, masks, quality = M9.track_vessel(sam, img_size, frames, box,
                                           device=args.device, reseed=args.reseed,
                                           preserve_aspect=not args.stretch,
                                           mode=args.track,
                                           use_mask_prompt=not args.no_mask_prompt,
                                           refine_lumen=not args.no_lumen_refine,
                                           max_jump_px=args.max_jump,
                                           max_area_ratio=args.max_area_ratio,
                                           reacquire_after=args.reacquire_after)
            per_seed.append((label, areas, masks, quality))

    # Clot may overlap the vein but may not dwarf it or sit outside it. The
    # bound is 2.0x the lumen, which rejects zero of 339 expert frames; the
    # v3 run had a median of 3.43 and a maximum of 159.
    from dvt import step14_medsam2_track as _S14

    # A mask that cannot be a single vessel is wrong regardless of which
    # vessel it is on. This is the only check that identifies WHICH of two
    # overlapping masks is broken - a figure-8 spanning both vessels is wrong
    # and the round mask beside it is right. 0.0% false positives on 6384
    # clean masks, but only ~13% sensitivity: the common failure is two ROUND
    # masks on one vessel, which shape cannot see.
    n_shape_rejected, shape_reasons = 0, {}
    if not args.no_shape_penalty:
        per_seed, n_shape_rejected, shape_reasons = _S14.constrain_shape(
            per_seed, min_artery_solidity=args.min_artery_solidity)

    per_seed, n_clot_rejected = _S14.constrain_clot(
        per_seed, max_ratio=args.max_clot_ratio,
        min_containment=args.min_clot_containment)

    # The VEIN drives the compression ratio. This used to be per_seed[0] -
    # whichever seed happened to be first - so on any clip seeded artery-first
    # the reported clip ratio described the artery. score_dvt reads the
    # vessel-level ratio, so the two disagreed silently.
    vein_entry = next((e for e in per_seed if _S14.role_of(e[0]) == "vein"),
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

    # Frames where the vein and artery tracks collided. Exclusivity makes the
    # stored masks disjoint, but on a converged pair that just splits one
    # vessel in two - the frame is still wrong, so propose nothing on it. Same
    # rule as --max-contrast: the radiologist draws one polygon rather than
    # untangling a misplaced one.
    bad_frames = set()
    if m2diag and args.max_containment > 0 and not args.keep_overlap_frames:
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
        "clot_frames_rejected": n_clot_rejected,
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
        "clot_fraction": _clot_fraction(per_seed),
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


def args_ds(a):
    return a.downsample


def _clot_fraction(per_seed):
    """Peak clot area as a fraction of the vein lumen it sits inside.

    This is the direct measurement: the compression ratio infers thrombus from
    geometry, whereas this looks for it. On expert masks the ratio inverted
    once segmentation chased dark pixels instead of the vessel wall, because a
    thrombosed vein has little anechoic lumen left - so measuring the clot
    itself sidesteps the failure mode entirely.
    """
    vein = next((a for l, a, _m, _q in per_seed if "Vein" in l), None)
    clot = next((a for l, a, _m, _q in per_seed if l == "Clot"), None)
    # NOTE: this is a per-frame max of clot/vein, so it is far more sensitive
    # to a single bad frame than the peak-area ratio is. On the v3 run the
    # peak-area ratio had a median of 1.14 - inside the expert envelope - while
    # this statistic read 3.26, because one runaway frame sets the maximum.
    # step14.constrain_clot blanks those frames before this runs.
    if vein is None or clot is None:
        return None
    v = np.asarray(vein, dtype=float)
    c = np.asarray(clot, dtype=float)
    n = min(len(v), len(c))
    if n == 0:
        return None
    v, c = v[:n], c[:n]
    ok = v > 20
    if not ok.any():
        return None
    return round(float(np.max(c[ok] / v[ok])), 4)


def _stride(dcm, n_processed):
    import pydicom
    ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
    total = int(getattr(ds, "NumberOfFrames", 1) or 1)
    return max(1, round(total / max(n_processed, 1)))


def seed_from_sidecar(dcm: Path):
    """Read the radiologist's single seed polygon from an existing sidecar.

    Returns (box, label) or (None, reason) so the caller can explain WHY a clip
    was skipped rather than printing an opaque SKIP.
    """
    cand = dcm.with_name(f"{dcm.stem}_darwin.json")
    if not cand.exists():
        alt = dcm.parent / "annotations" / f"{dcm.stem}_darwin.json"
        cand = alt if alt.exists() else dcm.with_suffix(".json")
    if not cand.exists():
        return None, "no sidecar file"
    ann = S2.parse_annotation_json(cand, dcm.stem)
    vein = [i for i in ann.instances if i.cls == 1]
    if not vein:
        n = len(ann.instances)
        return None, (f"sidecar has no VEIN polygon "
                      f"({n} other annotation(s); labels must contain 'Vein')")
    inst = vein[0]
    return M9.box_from_polygon(inst.points), inst.label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="MedSAM weights. Not needed with --medsam2, which "
                         "replaces the tracking path entirely.")
    ap.add_argument("--dcm", help="single DICOM")
    ap.add_argument("--dir", help="folder of DICOMs")
    ap.add_argument("--box", help="x0,y0,x1,y1 in original pixel coordinates")
    ap.add_argument("--auto", action="store_true",
                    help="find the vein automatically - no radiologist seed needed")
    ap.add_argument("--medsam2", default=None,
                    help="path to MedSAM2_latest.pt. Replaces the hand-written "
                         "box tracking with learned memory propagation.")
    ap.add_argument("--medsam2-cfg", default="configs/sam2.1_hiera_t512.yaml",
                    help="MedSAM2 model config")
    ap.add_argument("--reprompt-every", type=int, default=0,
                    help="re-supply the detector box every N frames. Counters "
                         "a merged mask entering the memory bank and being "
                         "inherited by every later frame. Needs --detector.")
    ap.add_argument("--medsam2-max-area-ratio", type=float, default=0.0,
                    help="re-prompt when area exceeds this multiple of the "
                         "rolling median (e.g. 1.8). 0 disables. WEAK - a "
                         "total vein/artery merge only grows area ~1.52x, so "
                         "this caught 4.1%% of contaminated frames on a "
                         "317-clip run. Prefer --max-containment.")
    ap.add_argument("--max-containment", type=float, default=0.0,
                    help="vein/artery mask overlap at which a frame is treated "
                         "as contaminated: re-prompt and write no polygon. "
                         "DEFAULT OFF. At 0.30 it suppressed 48%% of the frames "
                         "experts had annotated (up from 27%%), destroying 94 "
                         "good polygons to remove 84 junk ones. Use "
                         "scripts/qc_overlap.py to MEASURE overlap instead.")
    ap.add_argument("--exclusive", action="store_true",
                    help="let the higher logit win where two vessel tracks "
                         "claim the same pixel. DEFAULT OFF: measured "
                         "head-to-head on 1194 expert keyframes it moved "
                         "median IoU 0.564 -> 0.470 and DOUBLED the frames "
                         "where the 'vein' sat on the artery (12.9%% -> 26.6%%). "
                         "Argmax cannot restore the vein - when both tracks "
                         "converge it just splits one vessel in two, which "
                         "zeroes the overlap metric without fixing anatomy. "
                         "Fix the seed with --arbitrate instead.")
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
    ap.add_argument("--no-negative-points", action="store_true",
                    help="do NOT prompt each vessel with the other's centre as "
                         "a label-0 point. Negatives are ON by default: 82%% of "
                         "mask collisions are UNIONS (one track outlining both "
                         "vessels), and every post-hoc repair tried made things "
                         "worse or was neutral, so the separation has to happen "
                         "at prompt time. Clot is exempt in both directions.")
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
    ap.add_argument("--per-frame", action="store_true",
                    help="segment every frame INDEPENDENTLY - detector box per "
                         "frame, no memory, no propagation. Measured on 211 "
                         "frames with expert boxes: vein/artery collision 5.2%% "
                         "against 28.8%% propagated, median IoU 0.815 against "
                         "0.507. Six attempts to make propagation behave all "
                         "failed; this removes it. Slower (one encoder pass per "
                         "frame) and masks flicker slightly. Needs --detector.")
    ap.add_argument("--box-smooth", type=int, default=2,
                    help="moving-MEDIAN window, in frames either side, applied "
                         "to the detector boxes before prompting. Median not "
                         "mean, so a single frame where the detector jumps to "
                         "the wrong structure cannot drag its neighbours. 0 "
                         "disables.")
    ap.add_argument("--isolate", action="store_true",
                    help="track each vessel in its OWN pass with the rival "
                         "vessel hidden from the frames, located per-frame by "
                         "the detector. Costs ~3x. This is the only "
                         "intervention that constrains every frame rather than "
                         "just the prompt frame; the four that did not "
                         "(argmax, suppression, subtraction, negative points) "
                         "are documented in step15. Needs --detector.")
    ap.add_argument("--isolate-collapse-frac", type=float, default=0.15,
                    help="stop occluding once a vessel's own detection falls "
                         "below this fraction of its largest in the clip. When "
                         "the vein has collapsed there is nothing left to "
                         "protect, and hiding the only other dark structure "
                         "leaves the tracker nothing to hold - which turns a "
                         "stable wrong track into a wandering one. 0 disables.")
    ap.add_argument("--isolate-det-score", type=float, default=0.25,
                    help="detection floor used to LOCATE the rival vessel for "
                         "occlusion, separate from --det-score. Lower on "
                         "purpose: a missed rival means nothing is hidden and "
                         "the tracker is free to re-identify onto it, whereas "
                         "a spurious box only blanks a patch of tissue and the "
                         "self-overlap guard stops it blanking the target.")
    ap.add_argument("--isolate-stride", type=int, default=1,
                    help="run the detector every N frames when locating the "
                         "rival (1 = every frame). Vessels move a median 0.72-"
                         "0.79 box-sides over a clip, so a large stride will "
                         "leave the occlusion behind the vessel.")
    ap.add_argument("--max-clot-ratio", type=float, default=2.5,
                    help="blank a clot mask larger than this multiple of the "
                         "vein lumen on the same frame. Calibrated against the "
                         "TRACKED vein, which is what it sees: 2.0 rejects "
                         "3.9%% of real expert clot frames, 2.5 rejects 2.7%%. "
                         "Still catches the v3 pathology (median 3.43, max "
                         "159).")
    ap.add_argument("--min-clot-containment", type=float, default=0.50,
                    help="blank a clot mask with less than this fraction of "
                         "its area inside the vein. Containment against the "
                         "TRACKED vein is bimodal - either ~0.99 or ~0.00, "
                         "nothing between - so rejection is flat from 0.20 "
                         "(14.8%%) to 0.50 (16.1%%) and climbs only at 0.70 "
                         "(22.1%%). 0.50 separates the two modes cleanly.")
    ap.add_argument("--clot-det-score", type=float, default=None,
                    help="separate detection threshold for the clot class. "
                         "Clot detects at 0.587 against the vein's 0.743, so "
                         "one shared threshold starves it: only 78 of 317 "
                         "clips got a clot track. Defaults to --det-score.")
    ap.add_argument("--keep-overlap-frames", action="store_true",
                    help="write polygons even on frames flagged by "
                         "--max-containment (for auditing what was rejected)")
    ap.add_argument("--medsam2-chunk", type=int, default=0,
                    help="re-run init_state every N frames, re-seeding from "
                         "the detector. Needs --detector. Untested in "
                         "isolation: the v2 run used it together with "
                         "--exclusive, so its effect was never separated from "
                         "the harm that flag did.")
    ap.add_argument("--detector", default=None,
                    help="path to vessel_detector.pth from train_detector.py. "
                         "Replaces the heuristic seeder, which picked the wrong "
                         "structure on 66%% of clips.")
    ap.add_argument("--det-score", type=float, default=0.5,
                    help="minimum detector confidence to accept a box")
    ap.add_argument("--include-stills", action="store_true",
                    help="also process single-frame random images (tags ending "
                         "in 'r'); a compression ratio is undefined for these")
    ap.add_argument("--quiet-skips", action="store_true",
                    help="suppress per-clip SKIP lines, print a summary instead")
    ap.add_argument("--site", default=None,
                    help="CFV|GS|FV|PV, if the filename carries no site tag")
    ap.add_argument("--propose-clot", action="store_true",
                    help="also prompt for clot inside the vein box. Clot "
                         "presence IS the diagnosis, so this is usually a "
                         "stronger signal than the compression ratio.")
    ap.add_argument("--propose-artery", action="store_true",
                    help="also propose the artery, so the radiologist can "
                         "verify the vein/artery assignment at a glance")
    ap.add_argument("--from-sidecar", action="store_true",
                    help="seed each clip from its existing seed polygon")
    ap.add_argument("--label", default="Popliteal Vein # 4")
    ap.add_argument("--img-size", type=int, default=None,
                    help="inferred from the checkpoint when omitted; a "
                         "fine-tuned model only fits its training resolution")
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
    ap.add_argument("--track", default="recenter",
                    choices=["recenter", "fixed", "reseed"],
                    help="recenter (default): box keeps its size and follows the "
                         "vessel as tissue shifts under compression. fixed: box "
                         "never moves. reseed: box resizes to the mask - leaks, "
                         "kept only for comparison.")
    ap.add_argument("--skip-weak", action="store_true",
                    help="skip clips where no convincing vessel was found, "
                         "instead of seeding on noise")
    ap.add_argument("--reacquire-after", type=int, default=3,
                    help="blank frames before retrying the original seed box; "
                         "prevents the tracker getting stuck after losing the "
                         "vessel")
    ap.add_argument("--max-jump", type=float, default=25.0,
                    help="max centroid movement between consecutive frames (px) "
                         "before the mask is treated as having jumped to another "
                         "structure. Lower = stricter.")
    ap.add_argument("--max-area-ratio", type=float, default=2.5,
                    help="max area change vs the rolling median before a frame "
                         "is rejected. Lower = stricter.")
    ap.add_argument("--max-contrast", type=float, default=0.75,
                    help="skip proposing a polygon when mask interior / "
                         "surrounding tissue exceeds this (mask has slipped off "
                         "the lumen). 0 disables the check.")
    ap.add_argument("--no-lumen-refine", action="store_true",
                    help="disable re-thresholding the mask to the anechoic "
                         "lumen; refinement is ON by default and is what lets "
                         "the mask shrink as the vein collapses")
    ap.add_argument("--no-mask-prompt", action="store_true",
                    help="disable feeding the previous frame's mask as a dense "
                         "prompt (temporal continuity); on by default")
    ap.add_argument("--stretch", action="store_true",
                    help="stretch frames to a square instead of padding "
                         "(old behaviour; distorts anatomy, not recommended)")
    ap.add_argument("--reseed", action="store_true",
                    help="let the box follow the mask (bounded); off by default "
                         "because unbounded re-seeding makes masks leak")
    ap.add_argument("--clinician", default="")
    ap.add_argument("--email", default="")
    ap.add_argument("--out", default="autoannotate_report.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N clips (use while testing)")
    ap.add_argument("--file-list", default=None,
                    help="text file of filenames to restrict to, "
                         "e.g. filenames_40.txt")
    a = ap.parse_args()

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

    medsam2 = None
    if a.medsam2:
        from dvt import step14_medsam2_track as S14
        medsam2 = S14.load_predictor(a.medsam2, a.medsam2_cfg, a.device)
        print(f"MedSAM2 video predictor loaded from {a.medsam2}")

    detector, det_size, det_names = (None, 0, None)
    if a.detector:
        detector, det_size, det_names = S13.load_detector(a.detector, a.device)
        print(f"detector loaded from {a.detector} (input {det_size}px, "
              f"classes {sorted(det_names.values())})")

    # Skip loading MedSAM entirely when MedSAM2 does the tracking: process()
    # only calls M9.track_vessel on the non-medsam2 branch, so the weights
    # would sit unused in VRAM alongside the MedSAM2 predictor.
    if medsam2 is not None:
        sam, img_size = None, a.img_size or 512
        print("MedSAM not loaded: MedSAM2 handles tracking")
    else:
        sam, img_size = M9.load_medsam(a.checkpoint, a.img_size, a.device)
    print(f"MedSAM loaded at {img_size}px on {a.device}")

    targets = []
    if a.dcm:
        targets = [Path(a.dcm)]
    elif a.dir:
        targets = sorted(Path(a.dir).glob("*.dcm"))
        if a.file_list:
            keep = {ln.strip() for ln in Path(a.file_list).read_text().splitlines()
                    if ln.strip()}
            keep |= {Path(k).stem for k in keep}
            targets = [t for t in targets if t.name in keep or t.stem in keep]
            print(f"--file-list restricted {len(targets)} clips")
        if a.limit:
            targets = targets[:a.limit]
    else:
        sys.exit("need --dcm or --dir")

    if not targets:
        sys.exit("no DICOMs matched")

    # Fail fast rather than printing thousands of SKIPs.
    if a.from_sidecar:
        probe = [seed_from_sidecar(t)[0] is not None for t in targets[:25]]
        if not any(probe):
            print("\n" + "=" * 66)
            print("NO SEED POLYGONS FOUND in the first", len(probe), "clips.")
            print("=" * 66)
            print("--from-sidecar needs a polygon you drew yourself. For each clip")
            print("it looks for, beside the .dcm:")
            print(f"    <name>_darwin.json")
            print("\nDo this first:")
            print("  1. Open the clip in US-DICOMizer")
            print("  2. Anonymize (F8) so the Annotations tab activates")
            print("  3. Draw ONE polygon round the vein on ONE clear frame,")
            print("     using a label containing the word 'Vein'")
            print("  4. Export to ZIP, and put the JSON beside the .dcm")
            print("\nOr skip seeding and test one clip manually:")
            print("  python scripts/run_autoannotate.py --checkpoint <ckpt> \\")
            print("      --dcm <one>.dcm --box 25,65,95,115 --device cpu")
            sys.exit(1)

    from collections import Counter
    skip_reasons = Counter()
    arbitration = Counter()
    results, skipped = [], 0
    for i, dcm in enumerate(targets, 1):
        if a.auto:
            tag = S10.parse_filename_tag(dcm.name)
            site = tag["site"] or a.site

            # A trailing "r" marks a RANDOM STILL image, not a compression
            # cine. These are single-frame, so a compression ratio is
            # undefined - skip unless explicitly asked for.
            if tag["still"] and not a.include_stills:
                skipped += 1
                skip_reasons["still image (single frame)"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - still image "
                          f"({tag['tag']}); pass --include-stills to keep it")
                continue

            if site is None:
                skipped += 1
                skip_reasons[f"unrecognised site tag {tag['tag']!r}"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - site tag "
                          f"{tag['tag']!r} not recognised; pass --site")
                continue
            fr = S8.load_clip_frames(dcm, max_frames=24, downsample=args_ds(a))

            if detector is not None:
                # Trained detector: 74% vein detection at IoU>=0.5 on its own
                # held-out set, against the heuristic seeder's 6% of clips.
                # Seeding was the measured bottleneck, not segmentation.
                # detector_seed now selects the vein and artery as a PAIR and
                # assigns the roles by anatomy itself, so no separate
                # arbitration step is needed on this path: the two vessels are
                # adjacent, barely overlapping and within ~4x in size, and a
                # spurious box rarely satisfies all three against a real one.
                cands, seed_frame = S13.detector_seed(
                    detector, det_size, fr, device=a.device,
                    min_score=a.det_score, want_clot=a.propose_clot,
                    clot_score=a.clot_det_score,
                    site=(None if a.no_arbitrate else site),
                    class_names=det_names,
                    multi_vein_target=a.multi_vein_target)
                a._seed_frame = int(seed_frame)
                # When the deep vein is targeted the filename no longer
                # describes what is tracked, so the LABEL must follow the
                # vessel. v8 tracked the common femoral vein in GS clips and
                # called it "Great Saphenous Vein" - correct masks, wrong name.
                if (a.multi_vein_target == "deep"
                        and site in S10.DEEP_VEIN_LABEL):
                    a._label_override = S10.DEEP_VEIN_LABEL[site]
                else:
                    a._label_override = None
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
            else:
                sr = S10.auto_seed(fr, site, as_result=True)
                cands = sr.candidates
                if sr.weak and a.skip_weak:
                    skipped += 1
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - {sr.note}")
                    continue
                if sr.note:
                    print(f"      note: {dcm.stem[:8]} {sr.note}")
            vein_lbl, art_lbl = S10.SITE_LABELS[site]
            # the label must describe the vessel that was tracked
            if getattr(a, "_label_override", None):
                vein_lbl, art_lbl = a._label_override
            seeds = []
            for c in cands:
                if c.role == "clot":
                    # the detector predicts clot directly, so no second prompt
                    seeds.append((c.box, "Clot"))
                elif c.role == "vein":
                    seeds.append((c.box, vein_lbl))
                    if a.propose_clot and detector is None:
                        # Prompt the SAME box a second time with the clot label.
                        # A fine-tuned model learns clot as a distinct class
                        # (Dice ~0.66 here) but the pipeline never asked for it,
                        # so the class carrying the diagnosis was trained and
                        # then discarded. Clot lives inside the vein lumen, so
                        # the vein box is the correct prompt for it.
                        seeds.append((c.box, "Clot"))
                elif c.role == "artery" and a.propose_artery:
                    seeds.append((c.box, art_lbl))
            if not seeds:
                skipped += 1
                skip_reasons["no vessel candidate"] += 1
                if not a.quiet_skips:
                    print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - no vessel "
                          "candidate found (image likely shadowed)")
                continue
        elif a.from_sidecar:
            box, label = seed_from_sidecar(dcm)
            if box is None:
                skipped += 1
                print(f"[{i}/{len(targets)}] {dcm.stem[:8]} SKIP - {label}")
                continue
            seeds = [(box, label)]
        else:
            if not a.box:
                sys.exit("need --box, --from-sidecar or --auto")
            seeds = [([float(v) for v in a.box.split(",")], a.label)]
        try:
            a._detector, a._det_size = detector, det_size
            a._site = site if a.auto else None
            # `detector_seed` searches the first third of the clip and returns
            # the frame it actually found the vessels on. That index used to be
            # discarded and the boxes prompted at frame 0 regardless. Measured
            # against expert masks: a box taken 25-49 frames away has median
            # IoU 0.391 with the vessel's true position at frame 0, and 34.7%
            # of seed boxes overall land below IoU 0.5. An offset box overlaps
            # the neighbouring vessel, which is one way the two tracks come to
            # share a structure from the very first frame.
            a._seed_frame = 0
            r = process(dcm, seeds, sam, img_size, a, medsam2)
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
