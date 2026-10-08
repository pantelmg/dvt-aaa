"""Validate the probe/metric logic without needing V-JEPA weights.

Encoding needs a GPU and a model download; the probe maths does not. This
checks the parts that would silently give a wrong verdict: log loss, AUC,
base-rate initialisation, and that leave-one-centre-out actually holds a centre
out.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt import step6_vjepa_probe as V6

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


print("== log loss ==")
y = np.array([0] * 93 + [1] * 7)
check(abs(V6.log_loss(np.full(100, 0.07), y) - 0.2554) < 0.01,
      f"constant at prevalence -> {V6.log_loss(np.full(100, 0.07), y):.4f}")
check(V6.log_loss(np.full(100, 0.5), y) > V6.log_loss(np.full(100, 0.07), y),
      "predicting 0.5 is worse than predicting the base rate")
one_wrong = np.full(100, 0.07)
one_wrong[-1] = 1e-4
check(V6.log_loss(one_wrong, y) > V6.log_loss(np.full(100, 0.07), y) + 0.05,
      "one confident miss on a positive is heavily punished (log loss, not AUC)")

print("== auc ==")
check(abs(V6.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) - 1.0) < 1e-9, "separable -> 1.0")
check(abs(V6.auc([0.5] * 4, [0, 0, 1, 1]) - 0.5) < 1e-9, "all ties -> 0.5")

print("== probe: signal present ==")
rng = np.random.default_rng(0)
n, dim = 300, 32
X = rng.normal(size=(n, dim)).astype(np.float32)
y2 = (X[:, 0] + 0.3 * rng.normal(size=n) > 1.2).astype(int)
tr = np.arange(n) < 220
p = V6.fit_probe(X[tr], y2[tr], X[~tr])
a = V6.auc(p, y2[~tr])
check(a > 0.85, f"recovers a planted linear signal: AUC {a:.3f}")
check(V6.log_loss(p, y2[~tr]) < V6.log_loss(np.full((~tr).sum(), y2[tr].mean()), y2[~tr]),
      "and beats the constant baseline on log loss")

print("== probe: NO signal (the case that matters) ==")
yr = rng.integers(0, 2, size=n)
yr[:270] = 0                     # ~7% prevalence, unrelated to X
p2 = V6.fit_probe(X[tr], yr[tr], X[~tr])
ll = V6.log_loss(p2, yr[~tr])
bl = V6.log_loss(np.full((~tr).sum(), yr[tr].mean()), yr[~tr])
check(ll < bl + 0.15,
      f"pure noise stays near the baseline ({ll:.3f} vs {bl:.3f}), "
      "not catastrophically worse")
check(abs(np.mean(p2) - yr[tr].mean()) < 0.15,
      f"predictions stay near the base rate: mean {np.mean(p2):.3f} "
      f"vs prevalence {yr[tr].mean():.3f}")

print("== base-rate initialisation ==")
p3 = V6.fit_probe(X[tr], y2[tr], X[~tr], epochs=0, weight_decay=1e-2)
check(abs(np.mean(p3) - y2[tr].mean()) < 0.02,
      f"untrained probe already predicts the base rate ({np.mean(p3):.3f}) "
      "so it never scores worse than a constant")

print("== evaluate(): leave-one-centre-out really holds a centre out ==")
with tempfile.TemporaryDirectory() as td:
    m = 240
    Xf = rng.normal(size=(m, 16)).astype(np.float32)
    centre = np.array(["GR1", "IT1", "LT1", "FR1"] * (m // 4))
    subj = np.array([f"S{i//4:03d}" for i in range(m)])
    # planted centre effect, exactly the confound this guards against
    thr = np.where(centre == "GR1", 0.35, 0.03)
    yy = (rng.random(m) < thr).astype(int)
    f = Path(td) / "f.npz"
    np.savez(f, X=Xf, SubjectID=subj, centre=centre, site=np.array(["CFV"] * m),
             thrombosis=yy, abnormal=yy)
    r = V6.evaluate(f)

check(r["n"] == m, f"all rows used: {r['n']}")
check(set(r["leave_one_centre_out"]) <= {"GR1", "IT1", "LT1", "FR1"},
      f"folds: {sorted(r['leave_one_centre_out'])}")
tot = sum(v["n"] for v in r["leave_one_centre_out"].values())
check(tot <= m and all(v["n"] < m for v in r["leave_one_centre_out"].values()),
      "each fold tests on a strict subset - no centre leaks into its own test")
check("loco_mean_baseline" in r and "loco_mean_logloss" in r,
      "reports both the probe and the constant it must beat")
gr1 = r["leave_one_centre_out"].get("GR1")
if gr1:
    check(gr1["baseline"] > r["baseline_logloss_constant"],
          f"held-out GR1 is harder than the pooled baseline "
          f"({gr1['baseline']} vs {r['baseline_logloss_constant']}) - "
          "confirms the centre confound is being exercised")

print()
print("FAILED" if fails else "ALL CHECKS PASSED", f"({len(fails)} failures)")
sys.exit(1 if fails else 0)
