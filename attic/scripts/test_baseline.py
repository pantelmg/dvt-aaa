"""Validate the baseline classifier's AUC and threshold sweep numerically."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt.step5_baseline import ClipFeature, sweep_threshold, report, _auc

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


print("== AUC sanity ==")
check(abs(_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) - 1.0) < 1e-9,
      "perfectly separated -> AUC 1.0")
check(abs(_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) - 0.0) < 1e-9,
      "perfectly inverted -> AUC 0.0")
check(abs(_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])) - 0.5) < 1e-9,
      "all ties -> AUC 0.5")

print("== synthetic cohort ==")
rng = np.random.default_rng(0)
feats = []
# compressible veins collapse: ratio near 0.05-0.25
for i in range(40):
    feats.append(ClipFeature(f"c{i}", f"P{i//2}", "Common Femoral Vein # 1",
                             "common_femoral", float(rng.uniform(0.03, 0.25)),
                             5, "Yes", "No"))
# non-compressible (thrombosed) veins barely change: ratio 0.6-0.95
for i in range(15):
    feats.append(ClipFeature(f"n{i}", f"Q{i//2}", "Popliteal Vein # 4",
                             "popliteal", float(rng.uniform(0.60, 0.95)),
                             5, "No", "Yes"))

res = sweep_threshold(feats)
check(res["n"] == 55 and res["n_positive"] == 15,
      f"cohort counted: n={res['n']}, positives={res['n_positive']}")
check(res["auc"] == 1.0, f"cleanly separable cohort -> AUC {res['auc']}")
check(res["sensitivity"] == 1.0 and res["specificity"] == 1.0,
      f"sens {res['sensitivity']}, spec {res['specificity']}")
check(0.25 < res["threshold"] < 0.61, f"threshold strictly inside the gap, not on a data point: {res['threshold']}")

print("== overlapping (realistic) cohort ==")
feats2 = []
for i in range(40):
    feats2.append(ClipFeature(f"c{i}", f"P{i//2}", "v", "common_femoral",
                              float(np.clip(rng.normal(0.25, 0.15), 0, 1)), 5, "Yes", "No"))
for i in range(20):
    feats2.append(ClipFeature(f"n{i}", f"Q{i//2}", "v", "popliteal",
                              float(np.clip(rng.normal(0.55, 0.18), 0, 1)), 5, "No", "Yes"))
r2 = sweep_threshold(feats2)
check(0.6 < r2["auc"] < 1.0, f"overlapping cohort -> intermediate AUC {r2['auc']}")
check(r2["tp"] + r2["fn"] == 20, f"positives conserved: tp+fn={r2['tp']+r2['fn']}")
check(r2["tn"] + r2["fp"] == 40, f"negatives conserved: tn+fp={r2['tn']+r2['fp']}")

print("== per-site breakdown ==")
rep = report(feats)
check(set(rep["by_site"]) == {"common_femoral", "popliteal"},
      f"sites split: {sorted(rep['by_site'])}")
check("error" in rep["by_site"]["common_femoral"],
      "single-class site correctly flagged as unscoreable, not silently 0.5")

print("== degenerate input ==")
one = sweep_threshold([f for f in feats if f.label == "Yes"])
check("error" in one, "all-one-class input returns an error, not a fake score")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
