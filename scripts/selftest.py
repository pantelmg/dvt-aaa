"""
Self-test for the live pipeline surface. No GPU, no weights, no data needed.

Run this after changing anything in `dvt/`. It exercises the parts that are
pure logic - filename parsing, the anatomical priors, pair selection, box
bookkeeping between frames, the shape constraint, the compression ratio and
polygon conversion - with synthetic inputs whose correct answers are known.

It does NOT test MedSAM2 segmentation or the detector weights; those need a
GPU and a real clip. What it does catch is the failure mode that actually
happened during development: a refactor that silently moves a function and
breaks one caller.

    python scripts/selftest.py        # exit 0 = all good
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvt import step7_write_annotations as W
from dvt import step9_measures as M9
from dvt import step10_auto_seed as S10
from dvt import step13_detector_seed as S13
from dvt import step14_medsam2 as S14
from dvt import step16_per_frame as S16
from dvt.step13_detector_seed import Detection

FAILS: list[str] = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name
          + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def disc(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_geometry():
    print("box geometry (step16, moved here from the occlusion experiment)")
    check("_box_iou overlap",
          abs(S16._box_iou([0, 0, 10, 10], [5, 5, 15, 15]) - 25 / 175) < 1e-9)
    check("_box_iou disjoint", S16._box_iou([0, 0, 1, 1], [9, 9, 10, 10]) == 0.0)
    check("_centre", S16._centre([0, 0, 10, 20]) == (5.0, 10.0))


def test_continuity_and_depth_order():
    print("per-frame box bookkeeping")
    boxes = {1: [0, 50, 10, 60], 2: [0, 10, 10, 20]}       # 1 deep, 2 shallow
    order = S16.depth_order_of(boxes)
    check("depth_order_of ranks shallowest first",
          order[2] == 0 and order[1] == 1, order)

    tracks = {1: ([0, 50, 10, 60], 10.0), 2: ([0, 10, 10, 20], 10.0)}
    dets = [Detection([1, 51, 11, 61], 1, 0.9),
            Detection([1, 11, 11, 21], 1, 0.9)]
    m = S16._match_by_continuity(dets, tracks, depth_order=order)
    check("_match_by_continuity keeps each track on its own vessel",
          m[1][1] == 51 and m[2][1] == 11, m)

    fixed = S16._enforce_depth_order({1: [0, 10, 10, 20], 2: [0, 50, 10, 60]},
                                     order)
    check("_enforce_depth_order un-swaps an inversion",
          fixed[1][1] == 50 and fixed[2][1] == 10, fixed)

    sm = S16._smooth_boxes([[0, 0, 10, 10]] * 2 + [[100, 100, 110, 110]]
                           + [[0, 0, 10, 10]] * 2, 2)
    check("_smooth_boxes medians out a single outlier", sm[2][0] == 0, sm[2])
    check("_smooth_boxes preserves None", S16._smooth_boxes([None] * 5, 2)[2] is None)


def test_measures():
    print("measurements")
    check("compression_ratio ignores blank frames",
          abs(M9.compression_ratio([100, 50, 0, 80]) - 0.5) < 1e-9)
    check("compression_ratio None on fewer than 2 valid frames",
          M9.compression_ratio([0, 0, 100]) is None)
    check("role_of vein", S14.role_of("Common Femoral Vein # 1") == "vein")
    check("role_of artery", S14.role_of("Popliteal Artery # 4") == "artery")


def test_shape_constraint():
    print("shape constraint")
    round_m = disc(60, 60, 30, 30, 10)
    fig8 = disc(60, 60, 30, 18, 9) | disc(60, 60, 30, 42, 9)
    ps = [("Popliteal Artery # 4",
           np.array([round_m.sum(), fig8.sum()]),
           np.stack([round_m, fig8]),
           np.array([0.1, 0.1]))]
    out, n_rej, reasons = S14.constrain_shape(ps)
    check("keeps the round artery mask", out[0][2][0].any())
    check("rejects the figure-8 artery",
          n_rej == 1 and not out[0][2][1].any(), (n_rej, reasons))
    check("solidity: round > figure-8",
          S14.solidity(round_m) > S14.solidity(fig8))


def test_filenames():
    print("filename parsing")
    cases = [("anonymized_FR1-0001_0_003_CFV-L.dcm", "CFV", "L", False),
             ("anonymized_FR1-0001_0_001_CFVr-L.dcm", "CFV", "L", True),
             ("anonymized_LT1-0008_0_005_GSVr-L.dcm", "GS", "L", True),
             ("anonymized_FR1-0030_0_010_FV-Dr-L.dcm", "FV-D", "L", True),
             ("anonymized_FR1-0030_0_016_OPT-L.dcm", None, "L", False)]
    for fn, site, limb, still in cases:
        t = S10.parse_filename_tag(fn)
        check(f"tag {fn.split('_')[-1]}",
              (t["site"], t["limb"], t["still"]) == (site, limb, still), t)
    check("strips the _darwin sidecar suffix",
          S10.parse_filename_tag(
              "anonymized_FR1-0001_0_003_CFV-L_darwin.json")["site"] == "CFV")


def test_priors_and_pairing():
    print("anatomical priors and pair selection")
    shallow, deep = [0, 10, 10, 20], [0, 50, 10, 60]
    check("PV prior picks the SUPERFICIAL box as vein",
          S10.prior_picks_vein(shallow, deep, "PV")[0] == 0,
          S10.prior_picks_vein(shallow, deep, "PV"))
    check("CFV prior picks the DEEPER box as vein",
          S10.prior_picks_vein(deep, shallow, "CFV")[0] == 0,
          S10.prior_picks_vein(deep, shallow, "CFV"))
    check("unknown site yields no prior",
          S10.prior_picks_vein(deep, shallow, "ZZZ")[0] is None)
    check("GS is a multi-vein site", "GS" in S10.MULTI_VEIN_SITES)
    check("GS deep-vein label is the common femoral",
          S10.DEEP_VEIN_LABEL["GS"][0] == "Common Femoral Vein")

    # At CFV the deeper box must win even though the shallow one scores higher.
    dets = [Detection([20, 20, 40, 40], 1, 0.95, "vessel"),
            Detection([20, 60, 42, 82], 1, 0.80, "vessel")]
    vein, art, note = S13.select_vessel_pair(dets, (200, 200), "CFV")
    check("returns a pair at CFV", note == "pair" and art is not None, note)
    check("follows the prior, not the detector score",
          vein is not None and vein.box[1] == 60, vein)
    check("rejects a box covering the whole frame",
          S13.select_vessel_pair(
              [Detection([0, 0, 199, 199], 1, 0.99, "vessel")],
              (200, 200), "CFV")[0] is None)


def _tiny_dicom(path, nframes, h=16, w=20):
    """Smallest DICOM that `frame_count` and `load_clip_frames` will read."""
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import (ExplicitVRLittleEndian, SecondaryCaptureImageStorage,
                             generate_uid)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = h, w
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    if nframes > 1:
        ds.NumberOfFrames = nframes
    ds.PixelData = np.random.default_rng(0).integers(
        0, 255, (nframes, h, w), dtype=np.uint8).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_frame_count_gate():
    """The still/cine decision must come from the header, not the filename.

    Measured over the 1044 annotated clips, the US-DICOMizer `r` suffix is
    wrong in both directions: 18 clips hold one frame without it, and 12 carry
    it while holding 55-213 real frames. The second group is the costly one -
    they are real compression cines, and gating on the filename discarded them.
    """
    import tempfile

    from dvt import step8_load_clips as S8

    print("still/cine gate (header, not filename)")
    cases = [("anonymized_FR1-0001_0_003_CFV-L.dcm", 5, False, "process"),
             ("anonymized_FR1-0001_0_002_CFV-L.dcm", 1, False, "SKIP"),
             ("anonymized_FR1-0058_0_014_PVr-L.dcm", 5, True, "process"),
             ("anonymized_GR2-0021_0_001_CFVr-R.dcm", 1, True, "SKIP")]
    with tempfile.TemporaryDirectory() as td:
        for name, n, tag_still, want in cases:
            p = _tiny_dicom(Path(td) / name, n)
            check(f"parse_filename_tag still={tag_still} for {name.split('_')[-1]}",
                  S10.parse_filename_tag(name)["still"] is tag_still)
            check(f"frame_count reads {n} from the header", S8.frame_count(p) == n,
                  S8.frame_count(p))
            got = "SKIP" if S8.frame_count(p) <= 1 else "process"
            check(f"gate: {name.split('_')[-1]} with {n} frames -> {want}",
                  got == want, got)
            fr = S8.load_clip_frames(p, downsample=1)
            check(f"load_clip_frames returns {n} uint8 frames",
                  fr.shape[0] == n and fr.dtype.name == "uint8", fr.shape)


def test_polygons():
    print("polygon conversion")
    pts = W.mask_to_polygon(disc(60, 60, 30, 30, 10), max_points=40)
    check("mask_to_polygon returns a polygon", pts is not None and len(pts) >= 3)
    check("mask_to_polygon rejects a mask too small to be a vessel",
          W.mask_to_polygon(disc(60, 60, 30, 30, 1), max_points=40) is None)


def main():
    for fn in (test_geometry, test_continuity_and_depth_order, test_measures,
               test_shape_constraint, test_filenames, test_priors_and_pairing,
               test_frame_count_gate, test_polygons):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
