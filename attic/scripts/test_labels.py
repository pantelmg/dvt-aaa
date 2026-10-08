"""Validate label loading and split logic against the real ThrombUS+ train.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import step0_labels as S0

CSV = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/train.csv"
fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


df = S0.load_labels(CSV)

print("== load ==")
check(len(df) == 2341, f"2341 clips loaded: {len(df)}")
check(df.SubjectID.nunique() == 594, f"594 subjects: {df.SubjectID.nunique()}")
check(set(df.centre) == {"FR1", "GR1", "GR2", "IT1", "LT1"}, f"5 centres: {sorted(set(df.centre))}")
check(df.isna().sum().sum() == 0, "no missing values")

print("== label semantics ==")
lk = S0.check_leakage(df)
check(lk["comp0_all_thrombotic"], "Compressibility=0 always thrombotic")
check(lk["comp2_all_nonthrombotic"], "Compressibility=2 never thrombotic")
check(lk["n_partial"] == 19, f"19 partial cases break the equivalence: {lk['n_partial']}")
check(lk["agreement"] > 0.99, f"Compressibility/Thrombosis agree {lk['agreement']:.1%} of the time")

print("== class balance ==")
check(abs(df.abnormal.mean() - 0.071) < 0.005, f"abnormal rate {df.abnormal.mean():.1%}")
check(df.abnormal.mean() < 0.15,
      "prevalence below 15% -> accuracy is misleading, use AUC + sensitivity@specificity")

print("== centre shortcut ==")
a = S0.centre_shortcut_auc(df)
check(a > 0.65, f"centre prevalence alone gives AUC {a:.3f}")
check(a > 0.5, "a model reading scanner signature beats chance for free")

print("== subject-level split (no clip leakage) ==")
tr, va = S0.split_by_subject(df, val_frac=0.2, seed=0)
overlap = set(tr.SubjectID) & set(va.SubjectID)
check(not overlap, f"no subject in both splits: {len(overlap)} overlaps")
check(len(tr) + len(va) == len(df), f"all clips assigned: {len(tr)}+{len(va)}={len(tr)+len(va)}")
check(va.abnormal.sum() > 0, f"val split contains positives: {va.abnormal.sum()}")
rel = abs(va.abnormal.mean() - tr.abnormal.mean()) / tr.abnormal.mean()
check(rel < 0.5, f"stratification holds rates close: train {tr.abnormal.mean():.1%} "
                 f"val {va.abnormal.mean():.1%}")

print("== leave-one-centre-out ==")
folds = list(S0.leave_one_centre_out(df))
check(len(folds) == 5, f"5 folds: {len(folds)}")
for c, t, e in folds:
    check(set(e.centre) == {c} and c not in set(t.centre),
          f"fold {c}: test is only {c}, train excludes it (test n={len(e)}, pos={e.Thrombosis.sum()})")
shift = max(abs(e.Thrombosis.mean() - t.Thrombosis.mean()) for _, t, e in folds)
check(shift > 0.05,
      f"max train/test prevalence shift across folds = {shift:.1%} "
      "-> confirms centre is a strong confound")

print("== site mapping ==")
check(set(df["Anatomical Site"]) == {"CFV", "FV", "PV", "GS"},
      f"4 sites: {sorted(set(df['Anatomical Site']))}")
check(df.site_name.notna().all(), "every site code maps to a name")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
