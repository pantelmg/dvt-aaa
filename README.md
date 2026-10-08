# DVT Compression Ultrasound — Automated Annotation & Scoring

Leg compression ultrasound (ThrombUS+). The pipeline finds the vein and artery
in every frame of a compression clip, writes polygons back as Darwin sidecars
that US-DICOMizer can reimport, and derives the **compression ratio** —
min/max vein area — as a DVT feature.

The radiologist's job becomes correcting proposals instead of drawing from
scratch.

This README describes **only what the pipeline actually runs**. Rejected and
superseded approaches are in `attic/`, with a short account of each in
[§7](#7-what-was-removed-and-why). That section is the point of the cleanup:
the negative results are most of what was learned, so they are recorded rather
than deleted.

---

## 1. Results

Frozen subject-grouped 80/20 split, 1044 annotated clips from 80 subjects
across 5 centres.

| | vessel AUC | subject AUC | operating point |
|---|---|---|---|
| train (64 subjects, 270 measured clips) | **0.6678** | 0.7221 | sens 0.68 / spec 0.737 |
| test (16 subjects, 56 measured clips) | **0.6233** | 0.9821 † | sens 1.00 / spec 0.875 |

† tp7 / fp1 / fn0 / tn7 on 15 subjects. Unstable — one flipped pair moves it
~1.8 points, and it exceeds the expert ceiling, which is itself evidence of
small-sample luck. **Vessel AUC is the reportable endpoint**: 56 clips behind
it rather than 15 subjects.

**The ceiling.** Expert polygons — perfect segmentation by definition — give
vessel AUC **0.7009** and subject AUC **0.7831** on the same clips. The
pipeline reaches **95% of the vessel ceiling**.

That is the single most important number here: *the compression ratio, not the
segmentation, is the limiting factor.* Better masks cannot buy much more.
Anything that improves on 0.6233 has to be a different feature.

**Baseline that must be beaten.** Centre prevalence alone predicts DVT at
**AUC 0.709** — above the pipeline's subject-level result. FR1 (36 subjects)
and GR1 (28) hold most of the data; IT1 7, LT1 5, GR2 4. Any claimed result
must be checked against this.

---

## 2. Install

```bash
pip install numpy opencv-python pydicom pillow
pip install torch torchvision          # CUDA build; see --device cuda error text
pip install scikit-learn               # attic/scripts/check_centre_leakage.py only

git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2
pip install -e ".[dev]"
bash download.sh                       # -> checkpoints/MedSAM2_latest.pt
```

Two checkpoints are needed:

| file | what | where from |
|---|---|---|
| `MedSAM2_latest.pt` | SAM2.1-tiny fine-tuned on medical video, 512px | MedSAM2 `download.sh` |
| `vessel_detector_9926.pth` | Faster R-CNN vein/artery detector, 800px | trained here, §4 step 2 |

Verify the package without a GPU or any data:

```bash
python scripts/selftest.py             # 31 checks, exit 0 = good
```

Run that after touching anything in `dvt/`.

---

## 3. How it works

Five stages. Nothing is propagated through time — that is the central design
decision and §7 explains why.

```
filename              ──▶  anatomical site, limb, still-vs-cine
  anonymized_FR1-0001_0_003_CFV-L.dcm  ──▶  CFV, left, cine
                                                  step10_auto_seed

site                  ──▶  WHICH of the two vessels is the vein
  CFV/FV/FV-D: vein is DEEPER      PV/GS: vein is SUPERFICIAL
  priors are 95-96% accurate at CFV/FV/PV, 85% at GS
                                                  step10_auto_seed

every frame           ──▶  vein box + artery box
  detect ──▶ keep plausible ──▶ select as a PAIR ──▶ match to the
  track by nearest previous position ──▶ moving-median smooth
                                     step13_detector_seed, step16_per_frame

every frame           ──▶  vein mask + artery mask
  each frame prompted as its OWN one-frame video. no memory bank.
                                          step16_per_frame + MedSAM2

masks                 ──▶  polygons + compression ratio
  reject implausible shapes, skip frames where the mask slipped onto
  tissue, write <stem>_darwin.json, report min/max vein area
                            step14_medsam2, step7_write_annotations, step9_measures
```

### Why the vessel is identified by anatomy, not by the network

The detector's own class head put the vein label on the artery 13–27% of the
time. The site prior is 95–96% accurate at CFV/FV/PV. So `select_vessel_pair`
consults the prior and treats the class head as a weaker opinion.

The pair is selected jointly rather than vessel by vessel: the two vessels are
adjacent, barely overlapping, and within ~4× in size, and a spurious box rarely
satisfies all three constraints against a real one.

### Why the role is decided once but the box is found every frame

These are different questions. *Which vessel is the vein* is an anatomical
decision — decided at seeding and then held, because re-deciding it per frame
was measurably worse at the sites where the seed is usually right:

| site | seed once, then follow | re-decide every frame |
|---|---|---|
| CFV | **74.0%** | 70.4% |
| FV | **74.1%** | 65.6% |
| PV | **71.0%** | 68.2% |
| GS | 36.4% | **51.0%** |

*Where that vessel is on this frame* is a detection question, answered afresh
every frame. GS is the exception on the left column, and it is handled by a
scoped rule rather than by changing the default — see below.

### GS: a deliberate substitution

At the saphenous site the field holds three or four vessels and the GSV ranks
only 2nd among vein proposals by detector score (top-1 just 19.4%). With
`--multi-vein-target deep` the pipeline tracks the **common femoral vein** in
GS clips instead and renames the output accordingly. v8 tracked the right
vessel and called it "Great Saphenous Vein" — correct masks, wrong name.

### Two guards, both deliberately conservative

| guard | flag | what it does |
|---|---|---|
| shape | `--no-shape-penalty` to disable | rejects a mask that cannot be one vessel (a figure-8 spanning both). 0.0% false positives on 6384 clean masks, ~13% sensitivity. Kept because it is free and never fires wrongly — not because it is a defence. |
| mask quality | `--max-contrast` (default 0.75) | a vein lumen is anechoic, so a mask that slipped onto tissue reads bright. Such frames get **no polygon at all**. |

Both follow the same rule: **a missing polygon costs a radiologist far less
than a wrong one**, and downstream code reads a blank as "no measurement"
rather than "no vessel".

---

## 4. Running it

Inputs assumed: US-DICOMizer exports in `thrombus_s3_annotated/`, and
`train.csv` for clip-level labels.

### Step 1 — extract annotations and DICOMs

```bash
python scripts/extract_annotations.py \
    --root thrombus_s3_annotated \
    --out annotations_flat \
    --dicom-out dicoms_annotated
```

Flattens to `CENTRE__GROUP__SUBJECT__clip.json`, which is where ground truth
comes from downstream — `GROUP` is `DVT` or `NON_DVT`.

### Step 2 — freeze the split, once and never again

```bash
python scripts/make_split.py --annotations annotations_flat --out split.csv
```

Subject-grouped: every clip from a subject lands in the same set. 1044 clips
from 80 subjects, median 12 clips each, so a clip-level split would put
near-duplicate frames of the same leg on both sides and every number afterwards
would be optimistic.

Result: **train 859 clips / 64 subjects (26 DVT+), test 185 clips / 16 subjects
(7 DVT+), zero subject overlap.** Stratified on DVT status, seed 20250101.
The script refuses to overwrite an existing `split.csv` without `--force` — the
whole point is that it does not move.

### Step 3 — train the vessel detector

```bash
python scripts/train_detector.py \
    --annotations annotations_flat --dicoms dicoms_annotated \
    --split-csv split.csv \
    --epochs 25 --patience 10 --device cuda \
    --out checkpoints/vessel_detector_9926.pth
```

`--split-csv` is what keeps the detector **out of the test subjects**. Without
it the detector picks its own split and a test subject it has already seen
makes the pipeline's output on that subject not out-of-sample.

Faster R-CNN ResNet50-FPNv2, 800px, 2 ResNet stages unfrozen (41.8M trainable),
anchors measured from the data. Two classes, vein and artery: clot boxes are
recognised and dropped before the split, which is what `--drop-clot` used to do
and is now unconditional. Checkpoint selection is on **vein top-1**, not on
recall: the pipeline needs the right vessel first, not more boxes.

Trained run: 2686 annotated frames, 6316 boxes (3344 vein / 2972 artery), best
vein top-1 **0.5714** at epoch 5, early stop at 15. Against the heuristic
seeder's 6% of clips at IoU>0.5, and 74% vein detection at IoU≥0.5 on its own
held-out set.

### Step 4 — annotate

This is the configuration that produced the numbers in §1.

```bash
python scripts/run_autoannotate.py \
    --medsam2 MedSAM2/checkpoints/MedSAM2_latest.pt \
    --detector checkpoints/vessel_detector_9926.pth \
    --propose-artery --multi-vein-target deep \
    --no-depth-order --no-redecide-multivein \
    --dir dicoms_annotated --auto --device cuda \
    --downsample 1 --max-frames 0 --all-frames --quiet-skips \
    --out report_v14.json
```

| flag | why |
|---|---|
| `--propose-artery` | track the artery too. It is what makes a collision detectable at all. |
| `--multi-vein-target deep` | the GS substitution above |
| `--no-depth-order` | the depth-order constraint was never cleanly ablated; it trades ~1.6 points of coverage for removing frames whose masks are noise anyway. Off by default here. Measured: inversions 18.7% → 2.1%, mask IoU unchanged (0.7210 → 0.7233), coverage 11.7% → 13.3%. |
| `--no-redecide-multivein` | hold the seeded role; see the table in §3 |
| `--max-frames 0 --all-frames` | every frame, one polygon each |
| `--downsample 1` | full resolution |

Writes `<stem>_darwin.json` beside each DICOM, plus `report_v14.json` holding
the area curves, ratios and per-stage diagnostics.

`--medsam2`, `--detector` and `--auto` are all **required**: detector seeding
is the only mode. Manual `--box` seeding and `--from-sidecar` seeding are in
`attic/scripts/run_autoannotate.py`. An old command line containing a retired
flag fails with an explanation rather than an argparse error — this applies to
`train_detector.py` too, whose `--drop-clot` and `--no-merge-vessels` are now
the unconditional behaviour.

### Step 5 — score

```bash
python scripts/score_dvt.py --report report_v14.json \
    --expert annotations_flat --split-csv split.csv --split train

python scripts/score_dvt.py --report report_v14.json \
    --expert annotations_flat --split-csv split.csv --split test
```

Subject level takes the **max** across a subject's vessels: a patient has DVT
if any scanned site is thrombosed, so the least-compressible vessel is the one
that matters. That is also how the expert benchmark was computed.

Look at `test` **once**, at the end.

### The ceiling

The one extra measurement §1 quotes: what the compression ratio achieves on
expert polygons, over the clips this run covered.

```bash
python scripts/expert_reference.py annotations_flat dicoms_annotated report_v14.json
```

Detector scoring (`eval_detector.py`), sidecar QC (`qc_overlap.py`) and the
US-DICOMizer renamer are in `attic/scripts/` — useful, but none of them produce
a number reported in §1.

---

## 5. Module reference

Everything in `dvt/` is reachable from a command in §4. Numbering is historical
— the gaps are modules that moved to `attic/`.

| module | what it holds |
|---|---|
| `step7_write_annotations.py` | Darwin sidecar writing, mask → polygon |
| `step8_load_clips.py` | DICOM → greyscale clip; frame-index mapping; clip-wide normalisation |
| `step9_measures.py` | compression ratio, mask quality, box from polygon |
| `step10_auto_seed.py` | filename tags, site priors, multi-vein rules |
| `step13_detector_seed.py` | detector loading, per-frame detection, pair selection |
| `step14_medsam2.py` | MedSAM2 predictor, roles, shape constraint, overlap measurement |
| `step16_per_frame.py` | **the core** — per-frame segmentation and box bookkeeping |

Two modules were renamed in this cleanup because their contents changed
character: `step8_compression_phase.py` → `step8_load_clips.py`, and
`step9_medsam_infer.py` → `step9_measures.py`. `step14_medsam2_track.py` →
`step14_medsam2.py`, since it no longer tracks.

Each module's docstring carries the measurements behind its design. They are
the primary record; this README is the summary.

---

## 6. Known defects and limits

**The still/cine gate reads the DICOM header, not the filename.** The
US-DICOMizer `r` suffix is supposed to mark a single-frame random still, but
measured over all 1044 annotated clips it disagrees with the header in **both**
directions:

| | clips | consequence of trusting the suffix |
|---|---|---|
| `frame_count == 1`, no `r` suffix | 18 | harmless — 15 are `OPT` and skipped earlier for an unrecognised site; the other 3 reach processing and get `compression_ratio = None`, which `score_dvt` masks out |
| `r` suffix, `frame_count` 55–213 | 12 | **12 real compression cines discarded** — all 12 have a recognised site, so all were measurable |

The second row is the one that cost something: 12 clips against the 325 that
were measured, **+3.7% of the usable set**, split 5 DVT / 7 non-DVT — so lost
statistical power rather than bias.

Earlier notes described this defect the other way round: 18 clips yielding a
compression ratio of exactly 1.0 and a false-positive DVT bias. The count of 18
is right, but the consequence was never measured and is not what happens.
Across **all ten reports from v2 to v13clot**, every one shows 3 single-frame
clips, **0** clips at ratio 1.0, and 3 at `None`. Two guards already prevent it:
`step9_measures.compression_ratio` returns `None` below two valid frames, and
`score_dvt` skips NaN *before* the per-subject max, so one such clip cannot
poison a subject either.

`run_autoannotate` now gates on `step8_load_clips.frame_count` and reports any
clip where the suffix and the header disagree. Expect it to recover the 12 and
drop the 3. **This changes measured coverage, so numbers from before this change
are not directly comparable** — rerun before quoting a new AUC.

**The ratio is never computed from too few frames.** Median valid vein frames
per measured clip is 158, and no clip derives a ratio from 2–5 frames.

**Where the 1044 clips went in the v13 run** (before the gate changed):

| | clips |
|---|---|
| `r` suffix, skipped on the filename | 681 |
| treated as cine, attempted | 363 |
| skipped as unrecognised site tag — all `OPT` (operator view) | 37 |
| errored | 1 |
| **measured** | **325** |

All 37 `OPT` clips belong to **DVT subjects** (34 FR1, 1 GR1, 2 IT1). Skipping
them is correct — an operator overview carries no compression measurement — but
it is a DVT-only subset being dropped, and should be stated rather than left
implicit.

**Residual vein/artery overlap is 5.2% of frames.** Down from 28.8% under
propagation. `--max-containment` can drop those frames, but it is **off by
default**: the remaining 5.2% has not been shown to hurt the measurement, and
dropping frames costs coverage. Measure with `qc_overlap.py` before turning it
on; at 0.30 under the old propagation path it destroyed 94 good polygons to
remove 84 junk ones.

**Detector recall is the live bottleneck**, accounting for 86.3% of frames with
no polygon. Not the segmentation, and not the guards.

**Subject-level test AUC is not reportable** at 15 subjects. See the † in §1.

**Centre and scanner are perfectly confounded** and prevalence varies 5.8×
across centres. Centre prevalence alone reaches AUC 0.709.

**Clot is out of scope.** The pipeline proposes vein and artery only, matching
the expert annotations. Its expert ceiling (subject AUC 0.8281) is the highest
of any feature in this dataset and exceeds the compression ratio's 0.7831, so
it remains the best unexploited signal — but it never worked here (§7).

---

## 7. What was removed, and why

Everything below is in `attic/`, preserved unmodified — `attic/dvt/` is a full
snapshot of the package as it was before the cleanup, so the originals of the
pruned and renamed modules are there in one piece. See `attic/README.md` for how
to restore one. It is kept because these are measurements, not mistakes: each
one closed off a direction.

### Temporal propagation — the central result

MedSAM2 carries a memory bank, so the obvious design is to prompt frame 0 and
propagate. That was the pipeline for eight versions and it failed in a way that
took a long time to see.

SAM 2 propagates every object id **independently**. There is no cross-object
competition at inference, so nothing prevents the vein and artery ids settling
on the same pixels. Audited over 317 sidecars (35865 frames with both vessels):

| | |
|---|---|
| frames where the two masks overlap (containment ≥ 0.30) | **33.4%** |
| clips affected | **56.0%** |

Two facts killed the obvious explanations:

- **The seeds were fine.** Frame-0 vein/artery box IoU was a median 0.050. The
  detector put the boxes on different structures and the *tracks* converged
  afterwards.
- **Contamination ramped monotonically** through each clip (149 flagged frames
  in the first decile, 937 in the ninth), with no mid-clip peak and no
  recovery. That is a merged mask entering the memory bank and being inherited
  forward.

And it biased the measurement in the dangerous direction: contaminated frames
sat near the vein's area *maximum*, so min/max was deflated — median 0.170 raw
against 0.462 cleaned. A deflated ratio reads as a freely compressible vein,
i.e. a **false negative** for thrombosis.

Four repairs were built on top of propagation and measured. None worked:

| repair | result |
|---|---|
| per-pixel argmax between vessel ids (`exclusive`) | median IoU 0.564 → 0.470, and *doubled* the frames where the "vein" sat on the artery (12.9% → 26.6%) |
| area cap (`max_area_ratio`) | caught 4.1% of contaminated frames at threshold 1.8. A total merge grows vein area only ~1.52×, and the rolling median ratchets because the contamination is already in the history. Area was the wrong channel. |
| re-prompt on observed overlap (`max_containment`) | at 0.30 it suppressed 48% of expert-annotated frames to remove 84 junk ones |
| occlude the rival vessel (`step15_isolated_track`) | the most elaborate attempt; still depends on the memory bank |

**Removing propagation is what worked.** Measured on 211 frames with expert
boxes:

| | collision rate | median IoU |
|---|---|---|
| propagated (best run) | 28.8% | 0.507 |
| per frame, no memory | **5.2%** | **0.815** |

The encoder was never the problem; the memory bank was. Every repair was a way
to fix or avoid a corrupted memory bank while still depending on it.

Cost: per-frame is slower, because `init_state` runs once per frame rather than
once per clip. That is the whole price.

`attic/dvt/step14_medsam2_track.py` (`track_multi`, `resolve_exclusive`,
chunking, re-prompting), `attic/dvt/step15_isolated_track.py`.

### The heuristic seeder

Threshold the anechoic lumen, take connected components, filter by border
contact / elongation / size, assign roles by the site priors. Reached IoU>0.5
on **6% of clips**; the trained detector reaches 74% vein detection at IoU≥0.5.
Seeding was the measured bottleneck of the whole project — not segmentation.

Its four geometry helpers survived and now live in `step16_per_frame.py`.

`attic/dvt/step10_auto_seed.py` (the `find_candidates` / `assign_roles` /
`auto_seed` half of it).

### The MedSAM1 tracking path

Five hand-written box-propagation strategies, each failing measurably:

| strategy | failure |
|---|---|
| fixed box | drifts off the vessel as tissue shifts |
| reseed from mask | leaked 2090 → 34426 px (16× runaway) |
| shrink with mask | ratcheted the other way, ratio 0.294 with visibly under-segmented masks |
| recenter | works, but translation-only and hand-tuned |
| mask-prompt feedback | no measurable effect (0.625 vs 0.627) |

`attic/dvt/step9_medsam_infer.py`.

### Clot

The detector learns clot as a real class (Dice ~0.66), and clot presence has
the best expert ceiling of any feature here. It still never worked:

- Clot detects at 0.587 against the vein's 0.743, so at `--det-score 0.5` only
  78 of 317 clips returned any clot track.
- Prompting the vein's box while calling the result "Clot" returns a mask
  **byte-identical to the vein on 201/201 frames**. The label is a string; SAM 2
  segments whatever the box encloses. Clot needs its own detections.
- Clot sits almost entirely *inside* the vein — remeasured here over the 630
  expert frames carrying both a clot and a vein polygon, containment is a
  median **0.984** and clot area a median **0.646** of the vein's. So clot and
  vein legitimately overlap and cannot be separated by mask exclusion, which
  would tear the clot away from the lumen it sits in.
  (An older note claimed "only 12% of clots lie fully inside the vein". That
  does not reproduce: 40.2% are fully contained at a 0.99 threshold, and 38.3%
  if the 30 clot frames with no vein polygon are scored as zero, so the figure
  is not a containment measurement of any kind. The area median is subset-
  sensitive — 0.646 over all 630 co-annotated frames here against 0.80
  previously reported over 339 — so quote it with its n.)

Scoped out to match the expert annotations, which are vein and artery only.

`attic/dvt/step12_clot_detect.py`, `attic/scripts/test_clot_intensity.py`.

### Compression-phase detection

Find the relaxed and peak-pressure frames from bulk tissue motion, so a
radiologist reviews two frames per clip instead of 300. Median travel was
1.36 mm (translation) / 1.06 mm (strain) across 40 clips whose veins visibly
collapse — compression squashes tissue locally while the global image barely
moves — and cumulative frame-to-frame correlation drifted without bound (one
clip reported 856 mm).

Abandoned for a reason worth stating: peak compression *is* the vein's area
minimum, so once segmentation exists the phase is free. This was an attempt to
avoid segmentation, and segmentation was needed anyway.

`attic/dvt/step8_compression_phase.py`.

### V-JEPA 2.1 mask-video encoding

Encode the mask sequence as video and probe the embeddings, rather than
reducing each clip to one scalar. It does not lift the ceiling, and the
centre-leakage check showed the embeddings carry centre identity — normalising
the canvas to the vein made leakage *worse* (+0.099 → +0.118), i.e. it moved
the leak rather than removing it.

`attic/dvt/step6_vjepa_probe.py`, `attic/dvt/step17_mask_video.py`,
`attic/scripts/check_centre_leakage.py`.

### Alternative seeding and measurement tooling

Not failures — just not on the path to v14's numbers, and removed on a second
pass that cut the package to exactly that path:

| | why it is not here |
|---|---|
| `run_autoannotate.py` `--box` / `--label` | manual seeding. v14 seeds from the detector, so the only thing a human supplies is the folder. |
| `run_autoannotate.py` `--from-sidecar` | seeding from a polygon the radiologist drew. Its removal also retired `dvt/step2_json_to_masks.py` (Darwin *reading*) and `dvt/config.py`, which nothing else used. |
| `--file-list` and `filenames_40.txt` | a 40-clip pilot subset. `--limit` covers the same need. |
| `eval_detector.py` | scores a detector checkpoint on a common validation set. Genuinely useful before adopting a retrained detector; produces no number in §1. |
| `qc_overlap.py` | measures vein/artery overlap in written sidecars. Its `role()` was a duplicate of `step14_medsam2.role_of`, which `expert_reference.py` now uses. |
| `rename_for_usdicomizer.py` | renames hash-named DICOMs for the *unannotated* pool. Belongs to the self-training step, not this one. |
| `train_detector.py` clot class, oversampling, merged-vessel scheme | v14 trained with `--drop-clot --no-merge-vessels`; both are now unconditional. |

Note the third row: dropping one seeding mode removed two whole modules. That
is the usual shape of this kind of cleanup — a flag nobody passes keeps a
dependency alive.

### Post-hoc repair tools

`attic/scripts/fix_swaps.py` (reconstruct depth order after the fact and
un-swap: 2.48:1 filter from the first 8 frames, but 0.71:1 from the site prior
and 0.77:1 from a whole-clip majority — both re-decide the role per frame,
which is what made depth voting fail) and `attic/scripts/clean_sidecars.py`
(delete implausible segmentations after writing). Both became unnecessary once
the masks were produced per frame.

Also in `attic/scripts/`: the exploratory `test_*.py` scripts that produced
most of the numbers quoted above — `test_memory_hypothesis.py` is the one that
measured the 5.2% / 0.815 result and decided the architecture.

---

## 8. Where to go next

In order of expected value, given that the compression ratio is at 95% of its
ceiling:

1. **A different feature.** The ratio is near-exhausted. Clot presence has the
   higher ceiling (0.8281 vs 0.7831) and has never been made to work.
2. **Detector recall** — 86.3% of missing polygons. Improves coverage, not the
   ceiling.
3. **Self-training** on the ~1300 unannotated clips, seeding from the current
   model's confident proposals.
4. **Rerun with the header-based still gate** (§6). It recovers 12 measurable
   clips and drops 3 undefined ones, so the coverage figures above are stale
   until a fresh run.

Anything claimed from here has to beat **AUC 0.709 from centre prevalence
alone**, and be measured on `--split test` exactly once.
