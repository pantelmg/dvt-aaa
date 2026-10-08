r"""
Make the 80/20 train/test split. ONCE, then never again.

WHAT THIS IS
------------
A fixed, subject-grouped split of the annotated clips, written to a CSV that
every downstream script reads. Every clip belonging to a subject lands in the
same set - there are 1044 clips from 80 subjects, a median of 12 clips each, so
a clip-level split would put near-duplicate frames of the same leg on both
sides and every number afterwards would be optimistic.

Centre is deliberately NOT held out. That is a separate question from this
split, and mixing the two makes both harder to interpret.

WHY IT REFUSES TO OVERWRITE
---------------------------
The point of a frozen split is that it is frozen. Re-running with a different
seed until the test set looks favourable is the most ordinary way to fool
yourself, and it leaves no trace. `--force` exists, but using it invalidates
every number measured against the previous split.

WHAT IT STRATIFIES ON
---------------------
DVT status, so the test set keeps roughly the cohort's prevalence: 33 of 80
subjects are DVT-positive, and an unstratified draw of 16 could easily land 3
or 11. Centre balance is reported but not enforced - GR2 has 4 subjects and LT1
5, so forcing proportional representation would over-constrain a 16-subject
test set.

THE THING TO WATCH
------------------
The vessel detector was trained on its OWN 64/16 subject split, chosen with a
different seed. If subjects in this new test set were in that training set, the
detector has seen them and the test set is contaminated - the pipeline's output
on those clips is not an out-of-sample measurement. The script reports the
overlap explicitly. Retraining the detector with `--split-csv` pointing here is
the fix.

USAGE
-----
    python scripts/make_split.py --annotations annotations_flat --out split.csv
    python scripts/make_split.py --annotations annotations_flat --out split.csv \
        --check-detector-split          # report contamination only
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path


def load_subjects(annotations: Path):
    """-> (clips, subjects). Labels come from the annotation FILENAMES.

    `CENTRE__GROUP__SUBJECT__CLIP` carries everything needed. train.csv cannot
    be used: its `File Name` column holds hashed names while the sidecars and
    annotations are named from the anonymised stems, so the two do not join.
    """
    clips, subj = [], {}
    for p in sorted(Path(annotations).glob("*.json")):
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        centre, group, subject, clip = parts
        clips.append({"clip": clip, "subject": subject, "centre": centre,
                      "group": group})
        subj.setdefault(subject, {"centre": centre, "group": group, "n": 0})
        subj[subject]["n"] += 1
    return clips, subj


def make_split(subj, test_frac=0.20, seed=20250101):
    """Stratified by DVT status, grouped by subject, deterministic."""
    rng = random.Random(seed)
    test = set()
    for grp in ("DVT", "NON_DVT"):
        members = sorted(s for s, v in subj.items() if v["group"] == grp)
        rng.shuffle(members)
        k = int(round(len(members) * test_frac))
        test.update(members[:k])
    return test


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="split.csv")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20250101,
                    help="fixed on purpose; changing it invalidates every "
                         "number measured against the previous split")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split. See the docstring "
                         "before using this.")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out} already exists.\n"
              f"A frozen split is only frozen if it is not silently redrawn.\n"
              f"Use --force only if you intend to invalidate every result "
              f"measured against it.")
        return 1

    clips, subj = load_subjects(Path(a.annotations))
    if not subj:
        print(f"no CENTRE__GROUP__SUBJECT__CLIP files under {a.annotations}")
        return 1

    test = make_split(subj, a.test_frac, a.seed)

    import csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "subject", "centre",
                                           "group", "split"])
        w.writeheader()
        for c in sorted(clips, key=lambda r: (r["subject"], r["clip"])):
            w.writerow({**c,
                        "split": "test" if c["subject"] in test else "train"})

    tr_s = [s for s in subj if s not in test]
    tr_c = [c for c in clips if c["subject"] not in test]
    te_c = [c for c in clips if c["subject"] in test]

    def dvt(ss):
        return sum(1 for s in ss if subj[s]["group"] == "DVT")

    print(f"\n{'='*62}\nFROZEN SPLIT -> {out}\n{'='*62}")
    print(f"seed {a.seed}   grouped by SUBJECT   stratified on DVT status\n")
    print(f"{'':<8}{'subjects':>10}{'DVT+':>7}{'clips':>8}")
    print(f"{'train':<8}{len(tr_s):>10}{dvt(tr_s):>7}{len(tr_c):>8}")
    print(f"{'test':<8}{len(test):>10}{dvt(test):>7}{len(te_c):>8}")
    print(f"\nprevalence  train {dvt(tr_s)/max(len(tr_s),1):.1%}   "
          f"test {dvt(test)/max(len(test),1):.1%}")

    print("\ncentre balance (not enforced - GR2 has 4 subjects, LT1 5):")
    ctr = Counter(subj[s]["centre"] for s in subj)
    cte = Counter(subj[s]["centre"] for s in test)
    for c in sorted(ctr):
        print(f"  {c:<5} {ctr[c] - cte[c]:>3} train  {cte[c]:>3} test")

    # the leakage that matters
    print(f"\n{'-'*62}")
    print("DETECTOR CONTAMINATION")
    print(f"{'-'*62}")
    print("The vessel detector was trained on its own 64/16 subject split with")
    print("a different seed. Any test subject that was in ITS training set has")
    print("been seen by the model, so the pipeline's output on that subject is")
    print("not out-of-sample.")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.train_detector import split_by_subject  # noqa
        print("\nRetrain with --split-csv pointing at this file so the two")
        print("agree. Until then, treat test numbers as optimistic.")
    except Exception:
        print("\nRetrain the detector restricted to split=train.")

    print(f"\n{len(te_c)} test clips held out. Do not look at them until the")
    print("pipeline is final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
