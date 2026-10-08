"""
Data QC audit - run after every annotation batch, before any training.

Catches the failure modes that silently produce meaningless models:
  1. Vessels annotated on only one frame  -> compressibility uncomputable
  2. Clips with no compressibility label   -> unusable for the global task
  3. Class imbalance so extreme that Dice is uninformative
  4. Patients contributing most of the data -> split leakage risk
  5. Unrecognised polygon labels           -> schema drift vs config.py
  6. Lossy JPEG compression                -> texture destroyed for training
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import config as C
from dvt import step2_json_to_masks as S2

WARN, OK = "[WARN]", "[ ok ]"


def main(manifest=None):
    manifest = Path(manifest or (C.INDEX_DIR / "clips.json"))
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}. Run step1 first.")
    clips = json.loads(manifest.read_text())

    n_no_json = n_no_comp = n_single_frame_vessel = 0
    cls_counter, site_counter, unparsed = Counter(), Counter(), Counter()
    patient_frames, comp_counter, thromb_counter = Counter(), Counter(), Counter()
    tracked_vessels = 0

    for clip in clips:
        if not clip.get("json_path"):
            n_no_json += 1
            continue
        ann = S2.parse_annotation_json(Path(clip["json_path"]), clip["clip_id"])
        unparsed.update(ann.unparsed)
        comp_counter[str(ann.compressibility)] += 1
        thromb_counter[str(ann.thrombosis)] += 1
        if ann.compressibility not in ("Yes", "No", "Partial"):
            n_no_comp += 1
        for inst in ann.instances:
            cls_counter[C.CLASS_NAMES[inst.cls]] += 1
            site_counter[inst.site or "unknown"] += 1
        for vessel, curve in S2.area_curves(ann).items():
            tracked_vessels += 1
            if len(curve) < 2:
                n_single_frame_vessel += 1
        patient_frames[clip["patient_id"]] += len(S2.usable_frames(ann))

    print(f"=== {len(clips)} clips, {len(patient_frames)} patients ===\n")

    print("--- coverage ---")
    print(f"{WARN if n_no_json else OK} clips with no annotation JSON: {n_no_json}")
    print(f"{WARN if n_no_comp else OK} clips with no usable Compressibility label: {n_no_comp}")
    frac = n_single_frame_vessel / max(tracked_vessels, 1)
    print(f"{WARN if frac > 0.3 else OK} vessels annotated on only ONE frame: "
          f"{n_single_frame_vessel}/{tracked_vessels} ({frac:.0%})")
    if frac > 0.3:
        print("       -> compressibility cannot be computed for these. Annotate at")
        print("          minimum one pre-compression AND one peak-compression frame.")

    print("\n--- polygon classes ---")
    for k, v in cls_counter.most_common():
        print(f"  {k:12s} {v}")
    if cls_counter:
        rarest = min(cls_counter.values())
        ratio = max(cls_counter.values()) / max(rarest, 1)
        print(f"{WARN if ratio > 20 else OK} class imbalance ratio {ratio:.1f}:1")
        if ratio > 20:
            print("       -> report per-class Dice, never a single pooled number.")
    if unparsed:
        print(f"{WARN} unrecognised labels (schema drift vs config.py):")
        for k, v in unparsed.most_common():
            print(f"       {k!r} x{v}")

    print("\n--- anatomical sites ---")
    for k, v in site_counter.most_common():
        print(f"  {k:18s} {v}")
    if len(site_counter) > 1:
        print(f"{OK} split validation BY SITE too - popliteal and femoral")
        print("       have different image characteristics.")

    print("\n--- clip labels ---")
    print(f"  compressibility: {dict(comp_counter)}")
    print(f"  thrombosis     : {dict(thromb_counter)}")
    pos = comp_counter.get("No", 0)
    tot = sum(v for k, v in comp_counter.items() if k in ("Yes", "No"))
    if tot:
        print(f"{WARN if pos / tot < 0.15 else OK} abnormal (non-compressible) "
              f"prevalence: {pos}/{tot} = {pos/tot:.0%}")
        if pos / tot < 0.15:
            print("       -> accuracy is misleading at this prevalence; use AUC")
            print("          and report sensitivity at a fixed specificity.")

    print("\n--- patient concentration ---")
    top = patient_frames.most_common(3)
    total = sum(patient_frames.values())
    for pid, n in top:
        print(f"  {pid}: {n} frames ({n/max(total,1):.0%})")
    if top and total and top[0][1] / total > 0.3:
        print(f"{WARN} one patient holds >30% of frames - split by patient (the")
        print("       pipeline does) and treat val scores with caution.")

    print("\n--- reminder ---")
    print("  US-DICOMizer JPEG compression defaults to 75. Set it >95 for")
    print("  training exports: lossy compression destroys the speckle texture")
    print("  that distinguishes thrombus from flowing blood.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)


def audit_labels(csv_path):
    """Cohort-level audit of train.csv. Run before any modelling."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dvt import step0_labels as S0

    df = S0.load_labels(csv_path)
    print(S0.summary(df))
    lk = S0.check_leakage(df)
    print(f"\n{WARN} Compressibility/Thrombosis agree {lk['agreement']:.1%} "
          f"({lk['n_partial']} partial cases differ).")
    print("       Never use one as a feature for the other.")
    a = S0.centre_shortcut_auc(df)
    print(f"\n{WARN} centre-only shortcut AUC = {a:.3f}")
    print("       Each centre used a different scanner, and prevalence ranges")
    print("       2.3%-13.5% across centres. Report leave-one-centre-out.")
    print(f"\n{WARN} prevalence {df.abnormal.mean():.1%} -> report AUC and")
    print("       sensitivity at fixed specificity, never accuracy.")
