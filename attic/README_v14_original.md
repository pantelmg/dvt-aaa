# DVT Compression Ultrasound — Annotation & Analysis Pipeline

US-DICOMizer → MedSAM → V-JEPA 2.1, with a geometric baseline as the control.

## Install

```bash
pip install numpy opencv-python pydicom pillow          # steps 1,2,5 + QC
pip install torch torchvision segment-anything          # steps 3,4
git clone https://github.com/facebookresearch/vjepa2    # step 6
```

Checkpoints into `checkpoints/`:
- `medsam_vit_b.pth` — github.com/bowang-lab/MedSAM
- V-JEPA 2.1 ViT-L — github.com/facebookresearch/vjepa2

## The cohort (train.csv)

`train.csv` supplies every clip-level label, so **the clip-level task needs no
annotation JSON at all**. Polygon JSON is only required for segmentation.

2341 clips, 594 subjects, 5 centres, 4 sites per subject (CFV/FV/PV/GS).

| Compressibility | meaning | n |
|---|---|---|
| 2 | fully compressible (normal) | 2174 |
| 1 | partially compressible | 19 |
| 0 | non-compressible | 148 |

### Three findings that constrain the whole study design

**1. Compressibility and Thrombosis are near-duplicates.** `Compressibility==0`
implies `Thrombosis==1` in 148/148 cases; `Compressibility==2` implies
`Thrombosis==0` in 2174/2174. They agree 99.8% of the time — only the 19
"partial" clips differ. Feeding one to a model predicting the other is label
leakage, not feature engineering.

**2. Centre and scanner are perfectly confounded, and prevalence varies 5.8x.**

| centre | scanner | resolution | clips | thrombosis rate |
|---|---|---|---|---|
| GR1 | Philips EPIQ 7G | 327x273 | 811 | 13.4% |
| GR2 | Philips iU22 | 475x450 | 104 | 13.5% |
| FR1 | MINDRAY TEX20 | 606x581 | 247 | 4.9% |
| IT1 | ESAOTE 6150 | 479x304 | 711 | 2.3% |
| LT1 | GE LOGIQ S8 | 509x650 | 468 | 2.4% |

Image resolution alone identifies the centre, and **centre prevalence alone
gives AUC 0.709**. A model that learns nothing about veins but recognises the
scanner starts from there. Report leave-one-centre-out
(`step0_labels.leave_one_centre_out`) — a within-centre split cannot
distinguish signal from scanner.

**3. Pixel areas are not comparable across centres.** Pixel spacing ranges
0.0668-0.1578 mm (5.6x in mm² per pixel). A 50 mm² vein occupies ~2000 px at
FR1 and ~11000 px at LT1. Convert with `step5_baseline.to_mm2()`. The
compression *ratio* is dimensionless and therefore already centre-invariant,
which is a further argument for using the ratio over absolute area.

At 7.1% prevalence, report AUC and sensitivity at fixed specificity — never
accuracy, which a constant "normal" predictor scores 92.9% on.

## Execution order

```bash
python -m dvt.step0_labels data/train.csv    # cohort audit first
python -m dvt.step1_extract_frames      # ZIP → DICOM → PNG frames + manifest
python scripts/qc_audit.py              # audit BEFORE training anything
python -m dvt.step5_baseline            # ← DECISION GATE 1
python -m dvt.step3_dataset             # build + split the index
python -m dvt.step4_finetune_medsam --amp
python -m dvt.step6_vjepa_probe         # ← DECISION GATE 2
```

Note that step 5 runs *before* steps 3–4. That ordering is deliberate.

## The two decision gates

**Gate 1 — is this problem already solved by geometry?**
`step5_baseline` needs no GPU, no torch, and no model. It measures min/max lumen
area ratio per vessel and sweeps a threshold. If that separates compressible from
non-compressible clips at high AUC, the project is "MedSAM + a threshold" and the
video model is unjustified complexity. Run it on your first 20 annotated clips.

**Gate 2 — does V-JEPA transfer to ultrasound?**
`step6_vjepa_probe` freezes the encoder entirely and fits a linear probe — the
same protocol the V-JEPA authors use for their own dense evaluations. It prints
the probe AUC next to the Gate 1 baseline AUC. If the probe does not beat the
baseline, do not fine-tune; either continue self-supervised pretraining on
unlabelled ultrasound (needs no labels) or drop the video model.

## Design decisions worth knowing

**Area curves are per-vessel, not per-frame.** A frame often contains a
compressed common femoral vein *and* an uncompressed popliteal vein. Summing
vein pixels per frame mixes them and destroys the signal. `area_curves()` keys
by polygon label. There is a regression test for exactly this
(`scripts/test_parser.py`).

**Splits are by patient, never by frame.** Frames within one compression clip
are near-duplicates; a frame-level split leaks validation into training.

**Veins and arteries are separate classes.** MedSAM's published limitation is
difficulty with vessel-like branching structures, where a bounding box is
ambiguous — arteries and veins share one box. Supervising the distinction
explicitly is the mitigation.

**Boxes are jittered 0–20px during training**, matching MedSAM's own recipe.
Polygon-derived boxes hug the vessel; a clinician's drawn box will not.

**Clot is painted last** in rasterization, so a clot polygon drawn inside a vein
polygon wins. US-DICOMizer permits overlapping and enclosed labels.

**ROC thresholds are midpoints**, not observed values, so the operating point
never sits on a training data point.

## US-DICOMizer settings for ML export

| Setting | Default | Use | Why |
|---|---|---|---|
| Compression level | 75 | **>95** | Lossy JPEG destroys the speckle texture separating thrombus from flowing blood |
| Annotation format | Darwin V7 | Darwin V7 | What `step2` parses |
| Crop method | — | Auto / Mode | Saved per-serial coordinates break when scanning depth changes, which it does across vein sites |
| Clinician name | blank | fill in | Lands in the JSON; needed for inter-rater analysis |

## Annotation protocol requirements

The tool permits polygons only on frames graded ACEP ≥ 3. Beyond that, for
compressibility to be computable at all, **each vessel needs polygons on at
least two frames**: one pre-compression and one at peak compression. A vessel
annotated on a single frame is unusable for the primary task —
`scripts/qc_audit.py` reports the fraction affected.

## No JSON to hand? Run the inspector

You do not need to share an export. `scripts/inspect_json.py` runs locally and
reports whether the parser handles your file:

```bash
python scripts/inspect_json.py /path/to/annotations.json
python scripts/inspect_json.py /path/to/annotations.json --share   # redacted
```

It reports schema detected, polygons recovered, frame indices resolved,
keyframe vs interpolated counts, which labels are missing from `config.py`, and
whether a compression ratio is computable per vessel. `--share` prints key names
and value *types* only — no coordinates, no label text, no identifiers — which
is safe to paste if you want help adjusting the parser.

## Integration with US-DICOMizer (no app changes needed)

US-DICOMizer v5.6 source is public and MIT licensed
(github.com/thrombusplus/US-DICOMizer). Reading it removed all guesswork.

`package_io_utils.find_existing_sidecar_paths` means the app **already** loads
annotations written by anything else: drop a `<stem>_darwin.json` beside the
anonymized DICOM and it appears in the Annotations tab for correction. The
auto-annotator is therefore a sidecar producer, not a fork.

```
model -> step7_write_annotations -> <stem>_darwin.json -> radiologist opens
US-DICOMizer -> corrects polygons -> Export to ZIP -> back into training
```

`scripts/test_usdicomizer_roundtrip.py` validates our output against the app's
*own* `import_darwin_annotation`, so a green test means the radiologist really
will see the proposals.

### Four constraints from their importer, each of which silently drops data

1. Payload needs **both** `annotations` and `item` keys, or the app classifies
   it as LabelMe and never runs the Darwin branch.
2. A polygon annotation must have **`properties: []`**. The importer does
   `if ann.get("properties"): ... continue` — a non-empty list turns it into a
   tag annotation and the polygon vanishes. Tested explicitly.
3. Each path in `polygon.paths` becomes a **separate** polygon with the same
   label; holes are not supported on import. Emit one path per annotation.
4. Classification uses **codes**: `COMP2/COMP1/COMP0`, `DVT1/DVT0`. ACEP is the
   exact property name `"ACEP Grading Score"` with `"Grade "` stripped.

Reading the source also fixed three parser bugs: the ACEP property is named
`"ACEP Grading Score"` (normalises to `acep_grading_score`, previously missed,
so every frame grade was dropped); review status is `"_reviewed"` not
`"reviewed"`; and classification values arrive as codes needing decode.

Proposals are never pre-marked `_reviewed` — that tick has to mean a human
looked. Polygons are simplified to <=40 vertices, since the manual asks
annotators to use as few points as possible and a 400-vertex machine contour is
harder to correct than to redraw.

## Annotation format

The parser targets **Darwin JSON 2.0**, which US-DICOMizer states it mirrors,
with a heuristic fallback for near-spec variants. Three points from the spec
that are easy to get wrong:

- **Frame index is the dict key**, not a field:
  `"frames": {"137": {"polygon": {...}, "keyframe": true}}`. Reading a `frame`
  field instead silently collapses every polygon in a clip onto frame 0.
- **`keyframe: false` means interpolated** — generated by the tool's
  interpolation algorithm, not drawn by a human. Training on those is training
  on the tool's guesses. Excluded by default; pass `keyframes_only=False` to
  opt in.
- **`polygon.paths` is a list of paths**: the first is the outer boundary, the
  rest are holes. Filling them all with one value paints holes solid. On a
  thrombosed vein annotated as an annulus this overstated lumen area by 20% in
  test — directly corrupting the compressibility measurement.

Clip-level labels (Compressibility, Thrombosis, ACEP grade) are read from the
Darwin `properties` list, which carries a `frame_index` per entry, and
supplemented by a document-wide scan for exporters that place them elsewhere.

## Files

```
dvt/config.py                  label schema, paths, class mapping
dvt/step1_extract_frames.py    ZIP → DICOM → PNG + manifest
dvt/step2_json_to_masks.py     Darwin-V7 JSON → masks, boxes, area curves
dvt/step3_dataset.py           instance-level dataset, patient-level splits
dvt/step4_finetune_medsam.py   MedSAM fine-tune (frozen prompt encoder)
dvt/step5_baseline.py          geometric baseline — GATE 1
dvt/step6_vjepa_probe.py       frozen V-JEPA linear probe — GATE 2
scripts/qc_audit.py            dataset audit
scripts/test_parser.py         21 checks on parsing/rasterization
scripts/test_baseline.py       13 checks on AUC/threshold logic
scripts/test_darwin_v2.py      18 checks on Darwin 2.0 sequence parsing
scripts/test_labels.py         22 checks against the real train.csv
scripts/test_usdicomizer_roundtrip.py  17 checks vs US-DICOMizer v5.6 source
scripts/test_vjepa_probe.py    15 checks on probe/metric logic
dvt/step7_write_annotations.py Darwin sidecar writer (the auto-annotator output)
dvt/step0_labels.py            train.csv loader, splits, confound checks
scripts/inspect_json.py        local schema diagnostic
scripts/qc_overlap.py          vein/artery collision audit over sidecars
scripts/clean_sidecars.py      drop contaminated frames, no GPU needed
scripts/test_overlap_guard.py  30 checks on exclusion/containment logic
scripts/test_arbitration.py    23 checks on the anatomy priors
scripts/test_clot_and_prompts.py 23 checks on negatives + the clot bound
scripts/test_detector_training.py 23 checks on anchors/augment/top1
scripts/test_pair_seeding.py   36 checks on pair selection + roles
scripts/eval_detector.py       common-val comparison + PAIR recall
scripts/test_isolation.py      checks on the occlusion fill and guards
scripts/test_per_frame.py      21 checks on box smoothing and growth
scripts/test_memory_hypothesis.py  the diagnostic that settled it
dvt/step16_per_frame.py        per-frame segmentation, no propagation
dvt/step17_mask_video.py       sidecars -> centre-invariant mask videos
scripts/check_centre_leakage.py  the pre-flight check for V-JEPA
dvt/step15_isolated_track.py   per-pass tracking with the rival hidden
scripts/expert_reference.py    ceilings, clot geometry, auto-vs-expert IoU
```

## Status

Steps 1, 2, 5 and the QC audit are tested against synthetic data and run today.
Steps 3, 4, 6 are syntax-checked but untested end-to-end — they need torch, the
model checkpoints, and real clips. The V-JEPA loader in `step6` is a placeholder;
adjust the import to match the repo layout you clone, as the released API has
changed between versions.

## Validated against a real clip

A Philips EPIQ 7G compression clip (158 frames, 39 fps, 4.05 s, 327x273,
YBR_FULL_422, JPEG Baseline) was used to test steps 1-2. Findings that changed
the code:

- **Per-frame normalization was a bug.** Per-frame max ranged 186-255 on this
  clip, i.e. a ~29% gain difference between frames. Rescaling each frame
  independently makes identical tissue change brightness over time — spurious
  temporal variation injected into exactly the signal being measured.
  `_scale_clip()` now uses clip-wide statistics and passes 8-bit through.
- **Lossy detection silently returned False.** pydicom UIDs have no `is_lossy`
  attribute, so `getattr(ts, "is_lossy", False)` reported every file lossless.
  Now checked against an explicit UID set plus the `LossyImageCompression` tag.
- **Source data is already lossy.** JPEG Baseline is applied by the scanner, so
  US-DICOMizer's quality setting only controls *additional* loss. Still set it
  >95, but the texture budget is already partly spent.
- **Physical calibration is available** via `SequenceOfUltrasoundRegions`
  (0.1360 mm/px here → 0.018496 mm²/px), so lumen areas convert to mm² and are
  comparable across machines and scanning depths.

### Why prompted segmentation is required, not optional

Naive intensity thresholding + connected components was tried on this clip
(vein visibly compresses and re-expands — a normal, compressible study):

| Failure | Frames affected |
|---|---|
| Vein and artery merge into one component | 52/158 |
| Vein seed goes non-dark at peak compression | 48/158 |
| Vein blob merges with dark border shadow | 76/158 |

The resulting "compression ratio" was meaningless (vein and artery both
reported an identical 18570 px maximum — they were the same blob). This is the
empirical case for MedSAM: a per-vessel box prompt supplies the spatial
constraint that intensity alone cannot.

## Zenodo files are invisible to US-DICOMizer until renamed

The release ships SHA-256 filenames. The headers are already anonymized and the
images already cropped, but the app decides from the NAME alone:

```python
# anonymized_filename_utils.py
def looks_anonymized_filename(filename):
    return stem.startswith("anonymized_")
```

A hash fails that, so files never reach the "Anonymized DICOM files" list and
the Annotations tab stays greyed out. Nothing is wrong with the data.

```bash
python scripts/rename_for_usdicomizer.py --csv train.csv --dir dcms \
    --out renamed --file-list filenames_40.txt
```

Produces `anonymized_<SubjectID>_<file_no>_<TAG>.dcm`, e.g.
`anonymized_GR1-0252_0_001_GS-R.dcm`. Subject, site and limb all come from
train.csv, so the mapping is exact. Verified: 40/40 recognised by the app's own
`looks_anonymized_filename` and `parse_anonymized_filename`.

Note `file_no` uses Python underscore grouping — 1 renders as `0_001`, not
`0001`. Hardlinks by default, so no extra disk. `rename_map.csv` records the
original hash for every file; keep it, since Kaggle submission needs the
original names back.

## Feeding V-JEPA masks, not pixels (step 17)

The previous V-JEPA attempt learned the hospital: features predicted centre
identity at AUC ~1.00 and collapsed on held-out centres. That is structural, not
a tuning failure - centre prevalence alone gives AUC 0.709 here, each site used a
different scanner, and FR1 plus GR1 hold 93% of the annotated frames.

A binary mask has no speckle, no gain curve, no scanner signature. The leakage
is not mitigated, it is ABSENT.

What survives is everything the compression ratio discards: collapse speed,
rebound, deformation shape, vein/artery size ratio over time, and the artery as
an internal control for probe pressure. Min/max keeps none of it, which is why
the expert-mask ceiling is only 0.698 vessel / 0.783 subject.

**Geometry leaks too, if you let it.** Pixel area varies 5.6x across centres and
frames run 273x516 to 1300x745. `step17` therefore normalises the canvas to the
VEIN's own centroid and extent, never the frame's. Verified on the v8 sidecars:

| centre | n | vein peak area on canvas | raw scale removed |
|---|---|---|---|
| FR1 | 153 | 824 | 1.86x |
| GR1 | 133 | 825 | 1.19x |
| GR2 | 7 | 870 | 1.09x |
| IT1 | 21 | 915 | 1.00x |

Identical after normalisation; 1.86x apart before it.

**Invalid frames are marked, not filled.** 10.7% of expert keyframes have no
polygon, and on collided frames the vein polygon is on the vein only 48.3% of
the time against 78.0% on clean ones. Both are flagged in a per-frame validity
mask so the probe can exclude them rather than train on them. Measured on v8:
median 25.5 of 32 sampled frames valid.

**What is lost is texture, and with it clot** - the feature with the HIGHER
expert ceiling (subject 0.828 against the ratio's 0.783). `--channels 3` adds a
normalised pixel crop for that, but run masks-only first so the crop's value is
measured against a leak-free baseline.

### Check the leakage before spending compute

```bash
python scripts/check_centre_leakage.py --sidecars dicoms_annotated --labels train.csv
```

Trains a classifier to predict the CENTRE from the same inputs the probe will
use. Chance is not 20% - the classes are heavily imbalanced, so the script
computes the MAJORITY-CLASS rate as the floor and reports the lift over it. This
is the check that would have caught the original failure early.

## V-JEPA 2.1 probe (step 6) - the remaining path for DVT detection

The geometric route was tested and failed. On 31 clips with ground truth, six
features derived from MedSAM masks all scored at chance for thrombosis:

| feature | AUC | | feature | AUC |
|---|---|---|---|---|
| area_ratio | 0.571 | | circ_min | 0.525 |
| minor_ratio | 0.559 | | ecc_range | 0.454 |
| ecc_max | 0.487 | | circ_range | 0.349 |

So step 6 skips masks entirely. It uses the OFFICIAL torch.hub API - the
previous placeholder (`from vjepa2 import load_encoder`) referenced a package
that does not exist.

```bash
# 1. encode once (one forward pass per clip, cached)
python -m dvt.step6_vjepa_probe encode --dir renamed --csv train.csv \
    --variant base --device cuda

# 2. probe in seconds, re-runnable without re-encoding
python -m dvt.step6_vjepa_probe probe --features data/05_index/vjepa_base_features.npz
```

Variants: `base` (start here on 6 GB), `large`, `giant` - all 384px, 64 frames.

**The number to beat is 0.2381**, the log loss of predicting the training
prevalence for every clip. Not 0.5, and not AUC. The script prints the probe
against that constant on every fold and states the verdict plainly.

### Two optimiser traps, both caught by tests

1. **Undertraining.** At lr=1e-3 / 300 epochs the probe recovered only AUC
   0.630 on a synthetic signal it should score ~0.95 on. An undertrained probe
   reports "no signal" when signal is present - the failure that would abandon
   the approach for the wrong reason.
2. **Overconfidence.** Training harder fixed the AUC but LOST on log loss: good
   separation, badly calibrated probabilities. Weight decay is now selected on
   an inner split against log loss itself.

`scripts/test_vjepa_probe.py` guards both, plus a planted centre confound to
confirm leave-one-centre-out really holds a centre out.

## Vein and artery tracks collide - and what actually fixed it

The MedSAM2 run produced clean-looking polygons and a failure that clean-looking
polygons hide: the vein and artery masks converge onto one structure. Two labels,
one vessel.

`scripts/qc_overlap.py` audits this from the sidecars alone - no GPU, no ground
truth - because vein and artery are physically disjoint, so their masks should
never intersect. Over 268 clips with both roles on the same frame (35865 frames):
**33.4% of frames, 56.0% of clips.**

**The collisions are UNIONS, not slips.** Of 389 colliding frames, 320 (82%) had
containment > 0.9 - one track's mask entirely containing the other's, with the
inner mask correct. The model is not confusing the vessels; it is treating the
pair as one anechoic object.

### Three post-hoc repairs, all measured, none kept

| repair | result vs expert masks |
|---|---|
| argmax exclusivity between vessel ids | median IoU **0.564 -> 0.470**; frames with the vein on the artery **doubled**, 12.9% -> 26.6% |
| suppress contaminated frames | killed **94 good** polygons to remove **84 junk**; expert keyframes with no polygon rose 27% -> 48% |
| subtract the smaller mask, keep largest component | median IoU **0.298 -> 0.237** on colliding frames |

The common flaw is that all three act AFTER the mask exists. None can recover a
boundary the model never found. Argmax in particular zeroes the overlap METRIC
without fixing anatomy: on a converged pair it simply splits one vessel in two.

`--exclusive` and `--max-containment` are therefore **off by default** and kept
only for reproducing the failed experiment.

### What is kept: separate the objects before the mask exists

**Negative point prompts (default on).** SAM 2 accepts label-0 points meaning
"not this object". Each vessel is prompted with the other's centre as a negative,
stating at prompt time that the two are distinct. Clot is exempt in BOTH
directions - it must be free to sit inside the vein, and a negative on the clot
would punch a hole in the vein on every thrombosed clip.

**Seed arbitration (default on).** See the next section.

### Why `--medsam2-max-area-ratio 1.8` caught almost none of it

Two independent reasons, both measured by replaying the guard over the run:

1. **The effect is smaller than the threshold.** The artery is a median 0.52x the
   vein's area, so a *total* merge grows the vein mask only ~1.52x.
2. **The reference ratchets.** At flagged frames the growth factor against the
   previous 9 frames has median **1.01**. Rejected areas were appended
   unconditionally, so the reference absorbed the error it was meant to detect.

Catch rate: 4.1% at 1.8, 13.8% even at 1.2. Area is the wrong channel.

## GS was measuring the wrong vein entirely

Every label-accuracy figure quoted before this section counted a mask on the
COMMON FEMORAL vein in a GSV-tagged clip as correct, because the loader maps
both to the role "vein". Scored against the vein the FILENAME names:

| | |
|---|---|
| matches ANY vein | 71.0% |
| matches the SITE-NAMED vein | **50.7%** |
| right role, wrong vein | 20.3% |

By site:

| site | any vein | named vein | frames with >1 vein |
|---|---|---|---|
| CFV | 75.5% | 70.7% | 3.3% |
| FV | 71.8% | 64.9% | 5.2% |
| PV | 64.8% | 59.3% | 1.5% |
| **GS** | 71.1% | **8.4%** | **88.3%** |

GS clips are acquired at the saphenofemoral junction, where the great
saphenous, common femoral vein and common femoral artery are all in view - a
median of three vessels per frame. The pipeline almost always locked onto the
CFV. Only 41 frames in the whole run matched a GSV polygon.

**The GS rule was backwards.** Measured over 415 GS frames across 73 clips
carrying a GSV alongside another vein:

| rule | picks the GSV |
|---|---|
| **shallowest vein** | **99.3%** |
| smallest vein | 88.2% |
| largest vein (what was coded) | 11.6% |
| deepest vein | 0.7% |

The GSV is a median 0.35x the area of the other veins present, which sit a
median 1.22 GSV-widths deeper. It is the small superficial vein above the deep
system, exactly as anatomy says.

The earlier finding that "depth is only 63.4% at GS" compared the vein against
the ARTERY, where the GSV/CFA relationship really is ambiguous. Vein against
VEIN is near-perfect. That measured the wrong pair, and the wrong conclusion
stood for several versions.

**And pairing is the wrong formulation there.** `select_vessel_pair` picks the
best vein/artery PAIR, so at GS it can settle on CFV+CFA and never consider the
GSV. `MULTI_VEIN_SITES` now routes GS to `_select_multivein`: identify the
target vein by its own rule first, then give it a companion artery.

The companion needs care too - among the non-GSV vessels the artery is
SHALLOWER than the deep vein on 89.2% of 389 frames (size is only 67.1%).
Without that, plausibility alone picks the CFV as the "artery", because it is
the larger and more ordinary partner for the little saphenous.

## Seed arbitration - and two anatomy rules that were backwards

The detector assigns vein/artery from appearance, and the two vessels look nearly
identical on a still frame - which is the clinical reason compression is the
diagnostic test. Measured against expert masks, the auto "vein" sat on the expert
ARTERY on 13-27% of frames, and the tracker then follows that choice for the whole
clip.

`step10.assign_roles` used depth only for PV and SIZE for the femoral levels.
Measured clip-level on the 1044-file expert set, early frames:

| site | depth rule | size rule |
|---|---|---|
| CFV | **96.0%** (vein DEEPER) | 74.0% |
| FV | **95.7%** (vein DEEPER) | 60.9% |
| PV | **95.2%** (vein superficial) | 64.3% |
| GS | 63.4% | **85.4%** |

Both codings were wrong. Depth is the strong cue at CFV and FV too, with the sign
FLIPPED - the femoral vein lies deeper than its artery while the popliteal vein
lies superficial to it. Size is only 61-74% on relaxed frames and collapses to
**51% across whole clips**, because the vein is being compressed: it is reliably
the larger vessel only before the probe presses. GS keeps size, since the great
saphenous is itself superficial.

`step10.arbitrate_detections` re-decides the pair at seed time. Simulated on 267
clips by replaying each clip's actual seed:

| min_margin | vein label correct |
|---|---|
| no arbitration | 84.3% |
| 0.15 | 83.9% (worse than nothing) |
| **0.50 (default)** | **87.3%** |
| 1.20 | 85.0% |

Modest, because on **41% of clips at least one detector box is not on a vessel at
all** (IoU < 0.3 against both expert masks) and geometry between two arbitrary
boxes decides nothing. Restricted to clips where both boxes ARE on vessels:
93.6% -> 94.9%. That 41% is the largest remaining defect in the pipeline.

## Clot was unbounded, and it broke the best feature

Clot is exempt from vessel exclusion because it genuinely overlaps the vein -
which left it the only track with no constraint at all. Measured on 339 expert
frames carrying both polygons:

| | expert | v3 run |
|---|---|---|
| clot area / vein area | median 0.80, p99 1.57, **max 1.68** | median 3.43, **max 159** |
| clot area inside the vein | median 0.98, p10 0.88 | - |

On 43 of 67 clips the clot mask exceeded the vein's own maximum area. The feature
scored **subject AUC 0.282** - inverted, because it was measuring leak size.

`step14.constrain_clot` blanks a clot mask exceeding `--max-clot-ratio` (2.0,
which rejects ZERO expert frames) or with less than `--min-clot-containment`
(0.30) of its area inside the vein. The containment bar is far below the expert
p10 of 0.88 deliberately: the tracked vein mask is itself only ~0.5 IoU with
truth, so a strict bar would reject good clot. This catches runaway, not
imprecision.

**Coverage was the other half.** Clot detects at 0.587 against the vein's 0.743,
so one shared `--det-score` starved it - only 78 of 317 clips got a clot track.
`--clot-det-score` sets a separate floor. A missed clot is a missed diagnosis and
a spurious one is now bounded, so the asymmetry favours a lower floor.

## Ceilings, measured on the clips actually processed

The 0.7167 / 0.8491 figures below come from a different clip set. Computed on the
316 clips of the v3 run with `scripts/expert_reference.py`:

| feature | vessel | subject |
|---|---|---|
| compression ratio (expert polygons) | 0.7009 | 0.7852 |
| clot presence (expert polygons) | 0.6908 | **0.8281** |
| compression ratio (v3 automated) | 0.6285 | 0.7415 |

v3 reaches **94% of the expert subject-level ceiling**, at sens 0.656 / spec
0.761. Clot presence has the higher ceiling of the two, which is why fixing it
matters more than the last 0.04 of the ratio.

Run history, for honesty about what helped:

| run | vessel | subject | change |
|---|---|---|---|
| v2 | 0.5763 | 0.6800 | exclusivity + chunking - both harmful |
| v3 | **0.6285** | **0.7415** | defaults reverted, seed arbitration added |

## Vessel detection, rebuilt

### Vein and artery are ONE detection class

They are not visually separable on a still B-mode frame. That is the clinical
reason compression is the diagnostic test, and asking a detector to make the
call from appearance asks for information the image does not carry - it sat on
the wrong vessel 13-27% of the time.

Merging gives the vessel class **6316 boxes instead of 3344 and 2972**, and
frees the head from a distinction it cannot learn. Roles move entirely to
geometry, which measures well (`step10.SITE_PRIOR`: 96.0% CFV, 95.7% FV, 95.2%
PV, 85.4% GS). Clot stays its own class - thrombus has an echotexture, so it
genuinely is separable from lumen.

`--no-merge-vessels` reproduces the old three-class scheme for comparison. The
scheme travels in the checkpoint, so both load without a flag.

### The two vessels are chosen as a PAIR

Measured over 2516 expert frames carrying both:

| | p1 | median | p99 | max |
|---|---|---|---|---|
| centroid separation / mean box side | 0.62 | 1.13 | 4.16 | 3.63 (p0.1 = 0.23) |
| box IoU between the two | 0 | 0 | 0.136 | 0.447 |
| size ratio larger/smaller | - | 1.41 | 3.83 | 6.18 |
| vessel side / max(W,H) | 0.020 | 0.128 | - | 0.698 |

`select_vessel_pair` scores pairs, not boxes. The old rule kept the strongest
of each class independently, so a confident box on a shadow could become the
vein while the real vein ranked one below.

**Bounds are calibrated to reject ZERO of 1932 true expert pairs.** An earlier
cut rejected 7.9% of them, almost entirely through the relative-size bound: it
was taken from the geometric mean of the PAIR (p99 0.277) when an individual
vein reaches 0.698. Bounds derived from a summary statistic have to be
re-derived per quantity.

**What this does NOT buy.** Tested as a veto against a real 317-clip run's
seeds, the constraints did not discriminate: pairs with both boxes on vessels
were judged legal 87.3% of the time, pairs with a box on nothing 88.2%.
Geometry cannot tell a vessel from a plausible dark blob at a plausible
distance. That test used tracked-mask bounding boxes as a stand-in for
detections - blob-like by construction, so biased toward looking legal - but
the veto should not be relied on. The value is in RANKING (pairs near the
median separation win over extreme ones) and in geometric role assignment.

### Clot is anchored to the vein

`select_clot` requires box IoU >= 0.10 with the chosen vein and returns nothing
without one. An unanchored clot proposal is what produced `clot_fraction`
values up to 159 in the v3 run.

### The first merged run overfit at epoch 0

```
epoch 00  TOP1 {vessel: 0.7564, clot: 0.2153}   loss 0.3012   <- best vessel
epoch 02  TOP1 {vessel: 0.7459, clot: 0.4484}   loss 0.2491   <- SAVED
epoch 04  TOP1 {vessel: 0.7549, clot: 0.4159}   loss 0.2169
epoch 07  TOP1 {vessel: 0.7205, clot: 0.3038}   loss 0.2209
```

Vessel TOP1 peaked on the FIRST epoch and never recovered while training loss
fell steadily. 2045 training frames come from 64 subjects, and frames within a
subject are near-duplicates, so the effective sample size is nearer 64 than
2045.

**Checkpoint selection also saved the wrong epoch.** Macro TOP1 averages vessel
and clot; vessel is stable (0.68-0.76) while clot swings 0.215-0.463, so the
macro tracked clot noise and picked epoch 2 (vessel 0.7459) over epoch 4
(vessel 0.7549). The seed depends on vessel alone - clot is separately
thresholded and bounded downstream. `--select-on vessel_top1` is now the
default.

Five changes, all aimed at the overfitting:

| change | why |
|---|---|
| `--select-on vessel_top1` | the seeder consumes the vessel box; clot noise was driving selection |
| `--lr 1e-4` + cosine decay + 250-step warmup | 5e-4 is high for fine-tuning a pretrained R50-FPN, and matches the epoch-to-epoch thrashing |
| `--trainable-layers 2` | three unfrozen ResNet stages is far too much capacity for ~64 effective samples |
| scale / translate / gamma / speckle jitter | the previous augmentation was mild, and speckle is MULTIPLICATIVE - additive Gaussian trains against the wrong corruption |
| clot-stratified subject split | an unstratified draw put clot at 20.6% of val boxes against 11.5% of train, which is why clot validation was so noisy |

Augmentation is exercised numerically, not just pattern-matched: 500
randomised trials on the real `_augment` produced 0 invalid boxes of 1000 kept
and 0 out-of-range pixels, and the stratified split reproduces the train clot
share in val to within 0.00% with zero subject leakage.

**A caution about the ceiling.** None of this creates data. With 64 training
subjects and five centres whose scanner identity alone predicts thrombosis at
AUC 0.709, a large gain would be more suspicious than a small one. If vessel
TOP1 lands near 0.80 the changes worked; if it lands at 0.95, check the split
before celebrating.

### Two training runs, and why their numbers are not comparable

```
--keyframes-only   vessel TOP1 0.9038   clot 0.6477   val 1120 / 105 boxes
all frames         vessel TOP1 0.8264   clot 0.4550   val 1472 / 260 boxes
```

Both held out the SAME 16 subjects. They did NOT evaluate on the same boxes:
the all-frames run's validation set includes tool-INTERPOLATED polygons, so a
model can be marked wrong for disagreeing with the interpolation algorithm
rather than with a radiologist. Two readings, opposite consequences:

* interpolated polygons poison TRAINING -> drop them
* interpolated polygons poison MEASUREMENT -> train on them, never evaluate
  on them

`scripts/eval_detector.py --keyframes-only` scores every checkpoint on the
human-drawn subset and separates the two.

Dropping interpolation costs 10% of frames but **38% of clot boxes** (820 ->
508), so if the all-frames checkpoint holds up on a common val set, keep it for
clot and use keyframes only for measurement.

### PAIR recall - the metric the seeder actually needs

TOP1 asks whether the single best vessel box matches ANY ground-truth vessel.
Under the merged scheme that is easier than what the pipeline requires: a frame
has TWO vessels and `select_vessel_pair` needs both. A model that finds the
artery perfectly and never finds the vein scores **1.00 on vessel TOP1** and is
useless as a seeder.

PAIR asks whether the top two vessel predictions match two DISTINCT ground-truth
boxes at IoU >= 0.5. Expect it well below TOP1; it is the honest figure, and it
predicts whether the 41% off-vessel rate has moved. A clip where only one vessel
is found gets no artery track, so no collision can be detected on it downstream.

```bash
python scripts/eval_detector.py --annotations annotations_flat \
    --dicoms dicoms_annotated --keyframes-only --device cuda \
    --checkpoint checkpoints/vessel_detector_kf.pth \
    --checkpoint checkpoints/vessel_detector_all.pth
```

### The interpolation question, settled

Scored on a COMMON keyframes-only val set, the two checkpoints are the same:

| checkpoint | vessel TOP1 | clot TOP1 | PAIR |
|---|---|---|---|
| `vessel_detector_kf.pth` | 0.9038 | 0.6477 | 0.6201 |
| `vessel_detector_all.pth` | 0.9038 | 0.6136 | 0.6266 |

Identical on vessel, within noise on clot (105 val boxes, SE ~0.05). The
earlier 0.9038-vs-0.8264 gap was **entirely a measurement artifact** - the
all-frames run was being scored against interpolated targets.

So: train on everything, evaluate on keyframes only. The all-frames checkpoint
saw 38% more clot boxes for the same performance.

### PAIR 0.62 against vessel TOP1 0.90

Finding the FIRST vessel is easy; finding the second is not. And 1 - 0.62 = 38%
lines up with the 41% off-vessel rate measured independently from the run
sidecars - two unrelated measurements landing on the same number. The seeder is
not choosing badly among good candidates; the second good candidate often does
not exist.

`score_selector` in `eval_detector.py` runs the REAL `select_vessel_pair` over
detector output and reports three things the naive metric cannot:

| | |
|---|---|
| `naive` | top two boxes by score - the stricter figure |
| `selector` | what `select_vessel_pair` actually returns; it searches all pairs and rejects duplicates on box IoU, so it should be >= naive |
| `ROLE` | of the pairs it got right, how often the box labelled 'vein' is on the vein |

ROLE matters because the 95-96% behind `SITE_PRIOR` was measured on EXPERT
boxes. The same rule on detector output is a harder question: a box that is
slightly off-vessel makes the depth comparison meaningless. Reported per site,
since the rule and its sign differ by site.

### Role assignment on detector output, and the CFV limit

`score_selector` run on both checkpoints:

| checkpoint | naive | selector | ROLE | outcomes |
|---|---|---|---|---|
| `kf` | 0.6267 | 0.6719 | 0.8855 | 410 pair / 28 single / 4 none |
| `all` | 0.6380 | 0.6629 | 0.8737 | **435 pair / 7 single** |

The selector beats the naive top-two by 0.03-0.045, so duplicate rejection is
doing real work. ROLE by site:

| site | detector output | expert boxes (frame-level) | depth margin (median) |
|---|---|---|---|
| PV | 1.000 | 0.947 | 0.72 |
| FV | 0.967 | 0.914 | 0.71 |
| CFV | **0.812** | 0.843 | **0.25** |
| GS | 0.800 | 0.879 | 0.58 |

**The priors transfer intact** - detector output is at parity with expert
boxes. CFV is weak because at the common femoral level the vein and artery sit
side by side rather than stacked, so the depth margin is a third of what it is
elsewhere and small box errors flip it. Accuracy tracks margin at every site:
0.77-0.91 below 0.5, 0.92-0.97 above.

**Correction to an earlier number in this file.** The 96.0% CFV figure came
from a 50-clip early-frame subset. Over all 84 CFV clips the honest per-frame
figure is ~0.85. The priors were never as good at CFV as first reported.

Three fixes were tested against the expert set and all rejected:

* **size fallback at low margin** - size is 0.523 at CFV and gets WORSE the
  more confident it is (0.341 when the area gap exceeds 80%)
* **hybrid depth/size** - 0.620 at CFV against 0.843 for depth alone
* **voting across the searched frames** - 0.895 against 0.917 single-frame

CFV at 0.81 is close to the ceiling for static cues. The one cue that would
break through - compressibility - is circular here and documented in
`select_vessel_pair` as a thing not to do: a thrombosed vein does not compress,
so defining the vein as the vessel that compresses would relabel every
DVT-positive vein as an artery.

### Still overfitting, just from a higher floor

The keyframes run's vessel TOP1 spans 0.8619-0.9038 across ten epochs with no
trend, while training loss falls 0.346 -> 0.172. At 487 val images the standard
error is about +/-0.013, so that spread is close to noise. Both runs early-stop
long before the cosine schedule decays (LR never left 8.6e-05), so either
`--epochs 15` or `--patience 10` would make the schedule mean something.

Clot remains noisy in both: 105 val boxes gives an SE near 0.05, so its
0.477-0.648 swing is mostly sampling. Stratification is by subject clot
PRESENCE, so box-level shares still drift (10.4% train against 15.0% val on the
all-frames run).

## v5 results, and the two defects they exposed

v5 (occlusion) did NOT fix the overlaps: 68.8% of clips affected against v4's
51.4%, and subject AUC fell to 0.5778 (v3 0.7415, v4 0.6141).

**But it changed the failure mode in a useful way, and one observation held up.**
v5's masks "survive compression better" visually - and the reason is that v5
DECLINES to draw them. At severe compression it writes on 56.8% of frames
against v1's 74.1%. What you see is a survivorship-filtered set: only the
frames where it stayed confident. With the artery hidden, a collapsing vein
yields nothing rather than a confident grab at the artery.

That is good for annotation (a missing polygon costs a radiologist far less
than a wrong one) and bad for measurement: the compression ratio's MINIMUM
lives in exactly those frames, so dropping 43% of them inflates min/max.

| | v1 writes | v5 writes | v1 IoU | v5 IoU |
|---|---|---|---|---|
| severely compressed | 74.1% | 56.8% | 0.355 | 0.338 |
| relaxed | 86.1% | 82.6% | 0.718 | 0.672 |

### Veins jumped, and it was a bug of mine

Frame-to-frame centroid jumps above 25% of the mask side, by compression state:

| | vein v1 | vein v5 | artery v1 | artery v5 |
|---|---|---|---|---|
| severely compressed | 1.4% | **4.0%** | 1.4% | 2.1% |
| relaxed | 0.5% | 2.5% | 0.3% | 2.3% |

`locate_rivals` called `select_vessel_pair` on EVERY FRAME independently,
re-running the anatomical role decision from scratch each time. That decision
is ~87% accurate overall and 81% at CFV, so on roughly one frame in six the
vein's pass hid THE VEIN and left the artery fully visible - actively pushing
the vein track onto the artery for a frame or two before recovering. Brief,
self-recovering excursions, which is exactly what was observed.

**Fixed:** the role is decided ONCE at seeding, and per-frame association is by
CONTINUITY - a detection belongs to whichever track it is nearest to, with a
gate beyond which it is treated as unrelated, and an unmatched frame holds the
last known position rather than re-deciding.

**Also added `--isolate-collapse-frac 0.15`:** stop occluding once a vessel's
own detection falls below 15% of its largest in the clip. When the vein has
collapsed there is nothing left to protect, and hiding the only other dark
structure leaves the tracker nothing to hold - turning a stable wrong track
into a wandering one.

## The memory bank was the problem all along (step 16)

Six inference-time interventions failed to move vein/artery collisions out of a
24-31% band. `scripts/test_memory_hypothesis.py` finally tested the assumption
underneath all of them: 211 frames across 40 clips, segmented with NO
propagation, each prompted directly with the expert's own box.

| | single-frame | propagated (v7) |
|---|---|---|
| vein/artery collision | **5.2%** | 28.8% |
| median containment | **0.000** | - |
| fully merged (>= 0.9) | 1.9% | - |
| median IoU vs expert vein | **0.815** | 0.507 |
| median IoU vs expert artery | 0.845 | - |

Sanity: vein mask on the vein 99.5%, artery on artery 99.1% - the test measured
what it claimed to.

Two conclusions, eight standard errors from chance:

1. **The collisions come from the memory bank, not the encoder.** Prompted
   directly, the typical vein/artery pair does not touch at all.
2. **Propagation was costing ~0.3 IoU.** The encoder already produces
   near-expert masks. The tracker was destroying them.

Everything tried before this - argmax exclusivity, frame suppression, mask
subtraction, negative point prompts, occlusion, seed-frame correction - tried to
make propagation behave. None could, because the problem was that propagation
happened at all.

### `--per-frame`

Detect on every frame, associate detections to tracks by continuity, prompt each
frame independently. No state carries forward, so no frame can inherit another
frame's mistake. The components already existed: `step15` located vessels per
frame with the detector and then ran propagation anyway.

Boxes are smoothed with a moving MEDIAN (not mean) over +/-2 frames, so a single
frame where the detector jumps to the wrong structure cannot drag its
neighbours. Smoothing is applied to BOXES, never masks - that would reintroduce
the cross-frame state that caused the original problem.

```bash
python scripts/run_autoannotate.py --medsam2 MedSAM2/checkpoints/MedSAM2_latest.pt \
    --detector checkpoints/vessel_detector_split.pth \
    --propose-artery --per-frame \
    --dir dicoms_annotated --auto --device cuda \
    --downsample 1 --max-frames 0 --all-frames --out report_v8.json
```

**Costs:** one encoder pass per frame, so slower than propagation. Masks flicker
slightly, since nothing enforces temporal consistency.

**Caveat:** the 0.815 came from EXPERT boxes. The detector's are worse, so real
per-frame IoU will land below it. The collision result should hold regardless -
collisions came from memory, and there is no memory here.

## The seed box was applied at the wrong frame

`detector_seed` searches the first third of the clip and returns the frame it
actually found the vessels on. That index was DISCARDED and both tracking calls
hardcoded `prompt_frame=0`, so a box located on frame 50 was prompted at frame 0.

Measured against expert masks - IoU between a vessel's box at frame `_fi` and
its true position at frame 0:

| gap | n | median IoU | below 0.5 | below 0.1 |
|---|---|---|---|---|
| 1-9 frames | 30 | 0.712 | 30.0% | 6.7% |
| 10-24 | 159 | 0.688 | 23.9% | 3.8% |
| 25-49 | 76 | **0.391** | **59.2%** | 15.8% |

Overall **34.7% of seed boxes landed below IoU 0.5**, with a median gap of 19
frames and a p90 of 33. An offset box overlaps the neighbouring vessel, which
is one way the two tracks come to share a structure from the very first frame -
independent of anything the memory bank does later.

Fixed: `prompt_frame` is the frame the boxes came from, and `track_isolated`
now propagates BACKWARDS as well, since everything before the prompt frame
would otherwise be unsegmented.

## Isolated tracking - hiding the rival vessel (step 15)

The end state of the collision is all three tracks - vein, artery and clot - on
one vessel while the other is unannotated.

**Mechanism.** MedSAM2's memory bank stores appearance embeddings. Two anechoic
circles produce near-identical ones. When the vein compresses it shrinks and
deforms, so its match to the STORED memory of a round anechoic vessel degrades
while the artery still matches it perfectly - and the tracker re-identifies onto
the artery. Not gradual drift: re-acquisition onto a better-matching candidate,
which is why it is abrupt, clusters after compression starts, and never
recovers.

**Five interventions, four dead.**

| intervention | acts | result |
|---|---|---|
| argmax exclusivity | after the mask | IoU 0.564 -> 0.470; vein-on-artery frames DOUBLED |
| suppress collided frames | after the mask | killed 94 good polygons to remove 84 junk |
| subtract + largest blob | after the mask | IoU 0.298 -> 0.237 on collided frames |
| negative point prompts | prompt frame only | overlap 0.239 -> 0.238 |
| **occlusion (step 15)** | **every frame** | untested on GPU |

Shape helps a little and only for one case: artery solidity < 0.90 flags a union
with ZERO false positives across 6384 clean masks (expert arteries sit at
p1 = 0.962), but 13.4% sensitivity - because the dominant failure is two
perfectly ROUND masks on one vessel. Vein solidity is unusable, p1 = 0.382,
since a compressed vein is legitimately a slit.

**Why the window must be dynamic.** Vessels travel as the probe compresses:
max centroid displacement is a median 0.72-0.79 box-sides, p90 1.56-1.75. A
frozen ROI holds its vessel for a whole clip only 52.5% of the time at k=0.5 and
81.7% at k=1.0 - and k=1.0 spans three box-widths, swallowing the rival anyway,
since the vessels sit a median 1.13 box-widths apart.

**Why the occlusion comes from the detector, not the tracker.** Deriving it from
the tracker is self-defeating: once the tracks have converged, the "rival"
region IS the vessel being tracked. The detector is independent and finds both
vessels on 99.4% of seed frames.

**The fill.** Local median plus speckle scaled to surrounding variance, with a
feathered edge (max step 0.04 against 1.0 for a hard rectangle). A black or flat
grey patch is itself a structure and SAM 2 will segment it; statistics are taken
from OUTSIDE the boxes so a large anechoic vessel cannot drag the fill dark.

**Side effect worth having.** With the artery hidden, a compressed vein yields
NOTHING rather than a confident grab at the artery. Blank reads downstream as
"no measurement"; a wrong mask reads as a real one.

```bash
python scripts/run_autoannotate.py --medsam2 MedSAM2/checkpoints/MedSAM2_latest.pt \
    --detector checkpoints/vessel_detector_all.pth \
    --propose-artery --propose-clot --clot-det-score 0.3 \
    --isolate --limit 10 \
    --dir dicoms_annotated --auto --device cuda \
    --downsample 1 --max-frames 0 --all-frames --out report_iso10.json
```

Costs ~3x: two tracking passes plus per-frame detection. **Trial it on ten clips
and LOOK at the frames.** Four consecutive fixes were confidently wrong here;
this one cannot be simulated offline because it needs the model.

## Four defects found by reading the training code

`expert_reference.py` showed that on **41% of clips at least one detector box
is not on a vessel at all** (IoU < 0.3 against both expert masks), while the
reported detection rate was 74%. Those two numbers are consistent, and the gap
between them is the first defect.

**1. The selection metric was not the one the pipeline uses.** `evaluate`
measured RECALL - is there any prediction of the right class matching this box.
But `detector_seed` keeps "the strongest of each class" and hands that single
box to the tracker for the whole clip. A correct box ranked third counts for
recall and is invisible to the seeder. `evaluate` now reports **TOP1** as well,
and `--select-on top1` (the default) chooses the checkpoint on it.

**2. There was no augmentation.** `DetectionDataset.__init__` stored
`self.augment` and `__getitem__` never read it, so `augment=True` was a no-op
and every epoch saw the identical 2720 frames. With five centres whose scanner
identity alone predicts thrombosis at AUC 0.709, that is the setup in which a
detector memorises the centre. Horizontal flip and photometric jitter are now
applied. **Horizontal only** - a vertical flip inverts depth, and depth is the
cue separating vein from artery (96% at CFV, 95% at PV), so upside-down frames
would destroy exactly the signal `arbitrate_detections` relies on.

**3. The empty-box fallback taught a false positive.** When every box on a
frame was degenerate after rescaling, the code emitted `[[0,0,1,1]]` labelled
class 1 - VEIN. A 1x1 patch in the top-left corner, which is where ultrasound
frames carry their black border. Now a properly shaped zero-box target.

**4. Anchors were tuned for natural images.** Measured over the 7136 expert
boxes, at 800px input:

| | h/w p10 | h/w median | h/w p90 | side p5 | side median | side p95 |
|---|---|---|---|---|---|---|
| vein | 0.27 | 0.64 | 0.94 | 38 | 112 | 232 |
| artery | 0.52 | 0.81 | 1.08 | 57 | 93 | 164 |
| clot | 0.44 | 0.67 | 0.99 | 54 | 108 | 206 |

torchvision defaults to aspect ratios (0.5, 1.0, 2.0) - height/width - so a
third of its anchors cover tall boxes that never occur, and nothing reaches the
wide veins below 0.5. Sizes default to (32, 64, 128, 256, 512), putting two of
five FPN levels above anything in the data. Replaced with `ANCHOR_RATIOS =
(0.3, 0.55, 1.0)` and `ANCHOR_SIZES = (24, 48, 96, 160, 256)`.

**Also: clot oversampling.** Clot is 820 of 7136 boxes and detects at 0.587
against the vein's 0.743, yet it is the feature with the best expert ceiling
(subject AUC 0.828). `--clot-oversample 3.0` weights frames CONTAINING a clot,
so the rare class is lifted while every class on that frame is fed.

```bash
python scripts/train_detector.py --annotations annotations_flat \
    --dicoms dicoms_annotated --device cuda --epochs 25 \
    --select-on top1 --clot-oversample 3.0
```

Watch TOP1, not recall. The previous checkpoint's vein 0.743 was a recall
figure; the comparable TOP1 number has never been measured, so the first epoch
of this run establishes the real baseline.

## The seeder was the bottleneck, not the segmenter

## The seeder was the bottleneck, not the segmenter

Three experiments, in order, isolated where the pipeline failed.

**1. The compression ratio tolerates almost any boundary error.**
`scripts/degradation_test.py` degrades EXPERT masks and recomputes the AUC:

| degradation | vessel AUC | delta |
|---|---|---|
| none (baseline) | 0.7167 | - |
| uniform dilate/erode +-20% | 0.7167 | 0.0000 |
| vertex jitter 10px | 0.6820 | -0.035 |
| lose the thin vessel on 40% of clips | 0.6762 | -0.041 |
| drop 60% of frames | 0.6924 | -0.024 |

Nothing falls below 0.68. Uniform scaling gives EXACTLY zero change, because
min/max is scale-invariant - a segmenter with consistent boundary bias costs
nothing. Only differential error matters.

**2. But the automated pipeline scored 0.345** - inverted, far below that
floor. So the failure could not be boundary quality.

**3. The masks were on the wrong structure.** `scripts/validate_seeder.py`
compared 1790 auto frames against expert masks:

| median IoU vs expert vein | 0.0004 |
|---|---|
| clips with median IoU > 0.5 | 6% |
| clips with median IoU < 0.1 | 66% |
| within-clip IoU spread (p90-p10) | 0.018 |

The tiny spread is the diagnosis: the seeder locks onto one structure at frame
0 and tracks it faithfully all clip. Tracking worked; the initial choice did
not.

### The fix: a trained detector

`scripts/train_detector.py` trains Faster R-CNN on the expert polygons as boxes
(2715 frames, 7136 boxes: vein 3344, artery 2972, clot 820).

| | vein | artery | clot |
|---|---|---|---|
| detection rate @ IoU>=0.5 | **0.743** | 0.716 | **0.587** |

Against the heuristic seeder's 6% of clips, that is a 12x improvement on the
component that mattered. It also detects clot as a real class, which the
promptable MedSAM formulation structurally could not express.

```bash
python scripts/train_detector.py --annotations annotations_flat \
    --dicoms dicoms_annotated --device cuda --epochs 20

python scripts/run_autoannotate.py --checkpoint medsam_thrombus_1024.pth \
    --detector checkpoints/vessel_detector.pth --propose-clot \
    --dir renamed --auto --device cuda --downsample 1 --max-frames 0 --all-frames
```

## Auto-annotator: the working loop

Verified end to end on a real GR1 popliteal clip. MedSAM segments the vein
lumen **zero-shot** from one box prompt — no fine-tuning was needed to get
clean, correctly-bounded masks across a whole compression clip.

```bash
python scripts/run_autoannotate.py \
    --checkpoint checkpoints/medsam_vit_b.pth \
    --dir data/01_dicom --from-sidecar \
    --img-size 512 --device cuda
```

The radiologist draws **one polygon on one frame** in US-DICOMizer. That single
seed becomes the box prompt; MedSAM propagates it across every frame; the
proposals are written back as a `<stem>_darwin.json` sidecar the app loads
natively. Confirmed: our output round-trips through the app's own
`import_darwin_annotation` with the right frames, labels and vertex counts.

### Environment fixes (each one is a hard crash otherwise)

1. **The checkpoint is CUDA-saved.** `sam_model_registry` calls `torch.load`
   without `map_location`, so it raises on any CPU-only machine.
2. **ViT-B at 1024x1024 needs ~8 GB** — four global-attention blocks each hold
   a 4096x4096 attention matrix. `--img-size 512` interpolates the position
   embeddings and still segments these veins correctly.
3. **Reducing img_size means retargeting TWO grids.** Interpolating
   `image_encoder.pos_embed` alone leaves `prompt_encoder.image_embedding_size`
   at 64 and the decoder fails with a 32-vs-64 mismatch.

### Resolution settings that actually matter

Native frames in this cohort are 273-478 wide by 430-610 tall, so two defaults
were wrong:

**`--downsample 2` throws away detail for nothing.** It shrinks a 430x310 frame
to 215x155, and MedSAM then upsamples back to 512. Measured cost: essentially
identical runtime (32s vs 34s, since the fixed-size encoder dominates), so
`--downsample 1` is a free accuracy gain on images this small.

**Square resize distorted the anatomy.** SAM preprocesses by scaling the
longest side and zero-padding; stretching to a square is not something MedSAM
ever saw in training. GR1 frames are 273x516, so a square resize stretched them
**1.89x horizontally** and presented every circular artery as a wide ellipse.
Aspect-preserving padding is now the default; `--stretch` restores the old
behaviour for comparison.

Measured on one GR1 popliteal clip, the correction moved the compression ratio
from 0.640 to 0.724 - large enough to change a clinical call, which is the
point.

Recommended: `--downsample 1 --img-size 1024` if VRAM allows, falling back to
`--img-size 512`.

### Annotating every frame

Two flags control how many frames get a polygon:

* `--max-frames N` - how many frames are PROCESSED (`0` = all of them)
* `--all-frames` - write a polygon on every processed frame, instead of
  subsampling to `--n-polygons`

```bash
# every frame of every clip
--max-frames 0 --all-frames
# every 2nd-ish frame, much cheaper
--max-frames 80 --all-frames
```

Cost scales linearly with frames processed: one MedSAM pass per frame per
vessel. A 158-frame clip with vein and artery is 316 passes.

**Frame-index bug fixed here.** Polygon frame numbers used to be reconstructed
as `i * round(total / n_sampled)`, but sampling uses `linspace`, not a uniform
stride. On a 123-frame clip sampled to 8, linspace gives
`[0,17,35,52,70,87,105,122]` while the stride formula gives
`[0,15,30,...,105]` — **up to 17 frames adrift**, so proposals were being
attached to the wrong frames. `step8.frame_indices()` now returns the true
source indices and both code paths use it.

### Tracking through the compression phase (`--track`)

Masks degrade around peak compression because tissue SHIFTS under probe
pressure and a static box stops containing the vessel. Three strategies were
measured on the same GR1 popliteal clip:

| `--track` | box behaviour | ratio | outcome |
|---|---|---|---|
| `reseed` | resizes to the mask | 0.06 | leaks; area ran 2090 -> 34426 px |
| `fixed` | never moves | 0.584 | stable, drifts off the vessel late |
| `recenter` **(default)** | keeps size, follows position | 0.625 | tracks through compression |

Both size-adapting strategies ratchet. Allowing growth lets the mask leak into
neighbouring tissue. Allowing shrinkage is the mirror failure: one slightly
small mask yields a smaller box, which segments smaller still - that produced a
ratio of 0.294 with visibly under-segmented masks, an apparent "compression"
that was the tracker eating itself.

Only translation is needed, and a loose box is harmless: MedSAM is trained with
jittered box prompts precisely so it tolerates them. The centroid update is
damped 50/50 against the previous box to keep the trajectory smooth, and any
frame whose area more than doubles falls back to the seed box rather than
propagating the error forward.

### Auto-seeding: what was wrong and what changed

Evaluated on six real FR1 clips. Three concrete failures, each now addressed:

**Seeded on frame 0 regardless of quality.** An FR1 popliteal clip had a
featureless frame 0 and both proposals landed on the near-field artifact.
`auto_seed` now scans 24 frames across the first HALF of the clip and keeps the
best-contrast one.

**Ranking used absolute darkness.** Replaced by local contrast against the
surrounding wall, which is what actually distinguishes lumen from a dark
background region. A global reference is useless: on one FR1 clip the image
median was 31 while true perivascular tissue was ~47.

**The near-field band was being chosen as a vessel.** It scored excellent
contrast (0.17) because it abuts the bright near-field. Minimum depth raised
from 0.06 to 0.15 of image height, plus rejection of wide-flat components in
the top 10%. The vessels in this protocol are never that superficial.

Two new honesty flags, surfaced in the run log:

* `AMBIG` - vein/artery assignment was a near tie. Measured on a real GS clip:
  4268 vs 3926 px, an 8% difference deciding which vessel gets called the vein.
* `WEAK` - no candidate was convincingly darker than its surroundings. Use
  `--skip-weak` to skip these rather than seed on noise.

### Two bugs found in the consistency check itself

Rejected frames are stored as area 0. Two consequences, both measured on a real
FR1 common femoral run:

**1. The compression ratio was computed over the zeros.** One rejected frame
made the minimum 0 and the ratio 0.0 — reporting perfect compressibility where
the true value over valid frames was **0.416**. This corrupted every clip with
a single gap. `compression_ratio()` now ignores blanks, and a zero means "no
measurement", never "the vein closed".

**2. The tracker could not recover.** 26% of frames came back blank, in runs of
up to **16 consecutive frames**: once the vessel was lost, `last_good_box` went
stale, every later frame was judged against a position the vessel had left, and
rejections cascaded. `--reacquire-after 3` (default) retries the ORIGINAL seed
box after three blank frames and accepts what it finds, on the grounds that a
stale reference is worse than no reference.

The report now carries `frames_valid` and `frames_blank` per vessel — check
these first. A ratio computed from 157 of 213 frames is a different claim from
one computed from all of them.

### The dominant failure is JUMPS, not drift

Measured on six real FR1 clips exported from an actual run:

| clip / vessel | area range | max frame-to-frame step | centroid jumps >25px |
|---|---|---|---|
| CFV-R vein | 99 - 11761 px | **153x** | 11 (frames 9, 45-47, 101-104) |
| GS-L vein | 2519 - 4623 px | 1.2x | 0 |

The mask leaves the vein for a neighbouring dark structure and stays there for
a few frames. Contrast checks cannot catch this: the structure it jumps to is
also anechoic, so the mask still scores well.

Continuity can. A vein deforms smoothly at 13-39 fps, so a large discontinuity
between consecutive frames is an error rather than anatomy. `track_vessel` now
rejects a frame whose centroid moves more than `--max-jump` px (default 25) or
whose area departs from the rolling median by more than `--max-area-ratio`
(default 2.5x), re-segments from the last KNOWN GOOD box, and emits nothing if
that also fails.

Tune both against your own data with `scripts/tune_quality.py`, which reports
per-frame contrast and area and renders a worst-first contact sheet.

### Masks degrade at peak compression - four fixes tried, one kept

MedSAM returns a valid-looking mask for ANY box prompt. When a vein collapses
there is no lumen left, so it fills the box with whatever is most blob-like -
often adjacent tissue. On a real FR1 saphenous clip the reported area moved
only 6962 -> 5383 px across a full compression cycle: the mask stayed the same
size while the vessel closed.

| attempted fix | result |
|---|---|
| box shrinks with the mask | ratchets; ratio 0.294 with visibly under-segmented masks |
| previous mask as dense prompt | no effect (0.625 vs 0.627, identical masks) |
| MedSAM predicted-IoU gate | uninformative: 0.378-0.606 on ALL frames, r=0.246 with area |
| re-threshold mask to dark pixels | ragged, fragmented boundaries; worse to correct than raw |
| **flag bad frames, propose nothing** | **kept** |

`mask_quality()` scores mask interior against the surrounding wall - lumen is
anechoic, so a good mask is dark relative to its surroundings and a slipped one
approaches 1.0. It correlates -0.63 with mask area on a real clip: as the vein
closes and the mask creeps onto tissue, contrast rises.

`--max-contrast 0.75` (default) skips writing a polygon on those frames.
Proposing nothing is better than proposing something wrong: the radiologist
draws one polygon instead of untangling a misplaced one. Set 0 to disable.

Use the surrounding ring as reference, never the whole image - ultrasound
frames carry large black borders. On the FR1 clip the image median was 31 while
true perivascular tissue was ~47, so a global threshold called every mask bad.

### Do not re-seed the box from the mask

Measured on a real popliteal clip:

| strategy | area range | verdict |
|---|---|---|
| re-seed box from each mask + pad | 2090 → 34426 px | runaway leak, looks like signal |
| fixed box | 1237 → 2257 px | stable, correct |

Padding compounds every frame. `track_vessel(reseed=False)` is the default;
`reseed=True` clamps growth to 1.5x the seed box.
