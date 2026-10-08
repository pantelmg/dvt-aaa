# attic — superseded and rejected code

Nothing in here runs as part of the pipeline. It is kept because these are
**measurements**, not mistakes: each approach was built, measured, and closed
off, and the numbers behind the design decisions in `../README.md` §7 come from
this code.

## What is in here

`attic/dvt/` is a **complete snapshot of the package before the cleanup**, not
just the removed files. That means the originals of the four modules that were
pruned or renamed are here in full:

| attic file | relationship to the live tree |
|---|---|
| `step8_compression_phase.py` | original of `dvt/step8_load_clips.py`, plus the abandoned phase detector |
| `step9_medsam_infer.py` | original of `dvt/step9_measures.py`, plus the whole MedSAM1 tracking path |
| `step14_medsam2_track.py` | original of `dvt/step14_medsam2.py`, plus `track_multi` and the propagation guards |
| `step10_auto_seed.py` | original, including the heuristic seeder (`find_candidates`, `assign_roles`, `auto_seed`) |
| `step13_detector_seed.py` | original, including `select_clot` |
| `step16_per_frame.py` | original, including the clot branch |
| `run_autoannotate.py`, `score_dvt.py` | originals, with every retired flag and the `--box` / `--from-sidecar` seeding modes |
| `train_detector.py` | original, with the clot class, clot oversampling and the merged-vessel scheme |
| `step2_json_to_masks.py`, `config.py` | Darwin *reading* and the label/path constants — used only by `--from-sidecar` |
| `eval_detector.py`, `qc_overlap.py`, `rename_for_usdicomizer.py`, `filenames_40.txt` | measurement and prep tooling, off the path to v14's numbers |
| `README_v14_original.md` | the pre-cleanup README (1368 lines) |

Modules with no live counterpart: `step0_labels`, `step1_extract_frames`,
`step3_dataset`, `step4_finetune_medsam`, `step5_baseline`,
`step6_vjepa_probe`, `step11_expert_labels`, `step12_clot_detect`,
`step15_isolated_track`, `step17_mask_video`.

## Running something from here

These files import each other as `dvt.*`, so they will **not** run from inside
`attic/`. To bring one back, copy the whole attic package over the live one in
a scratch checkout:

```bash
cp -r pkg pkg_attic_restore
cp attic/dvt/*.py pkg_attic_restore/dvt/
cp attic/scripts/*.py pkg_attic_restore/scripts/
```

That gives you the pre-cleanup package exactly as it was, with every flag the
old command lines used. Do not restore individual files into the live `dvt/` —
the pruned modules and the originals share names but not contents, and mixing
them gives you neither version.

## Why each one is here

See `../README.md` §7. In short:

| | |
|---|---|
| propagation (`step14_medsam2_track`, `step15_isolated_track`) | the memory bank merged the vein and artery tracks on 33.4% of frames; four repairs all failed; removing propagation cut collisions to 5.2% and raised median IoU 0.507 → 0.815 |
| heuristic seeder (in `step10_auto_seed`) | IoU>0.5 on 6% of clips, against the trained detector's 74% |
| MedSAM1 tracking (in `step9_medsam_infer`) | five hand-written strategies, each failing measurably |
| clot (`step12_clot_detect`) | best expert ceiling in the dataset (0.8281) but never made to work; scoped out to match the expert annotations |
| compression phase (in `step8_compression_phase`) | bulk tissue motion is too weak a proxy; the vein's area minimum gives the phase for free once segmentation exists |
| V-JEPA (`step6_vjepa_probe`, `step17_mask_video`) | does not lift the ceiling, and the embeddings carry centre identity |

The exploratory `test_*.py` scripts in `attic/scripts/` produced most of the
numbers quoted in the main README. `test_memory_hypothesis.py` is the one that
measured the per-frame result and decided the architecture.
