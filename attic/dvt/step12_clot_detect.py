r"""
Step 12 - Frame-level clot detection.

WHY THIS AND NOT SEGMENTATION
-----------------------------
MedSAM is promptable and has no class output: prompting the vein box with a
"Clot" label returned a mask byte-identical to the vein on 201/201 frames,
because the label is a string we attach, not something the model predicts.

Clot also cannot be recovered by refining the vein mask. Measured on 783 expert
clot polygons: median clot/vein area 0.62 (max 1.90), only 12% of clots lie
fully inside the vein outline and 24% lie mostly outside it. Clot is not a
sub-region of the lumen.

But clot does not need to be SEGMENTED to be useful. On expert masks, clot
presence at subject level gives AUC 0.843 at sensitivity 0.727 / specificity
0.959 - a far better operating point than the compression ratio's 0.43/0.94.
So detect it instead.

THE LEAKAGE TRAP
----------------
Frame-level clot is nearly a restatement of subject-level DVT status:

    frames from DVT subjects     containing clot: 45.0%
    frames from NON_DVT subjects containing clot:  0.5%

A model can therefore score well by recognising the SUBJECT, not the thrombus.
Two guards, both mandatory:

  * split by subject, never by frame - a subject contributes many near
    duplicate frames
  * report leave-one-centre-out as well, because clot prevalence ranges 0% to
    40% across centres and each centre used a different scanner. V-JEPA
    features predicted the hospital at AUC 1.000 in this cohort, so scanner
    identity is demonstrably learnable

IT1 and LT1 contain no clot at all, so they can only ever be used for training
or as negative-only sanity checks - never as the sole validation set.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ labels
def build_frame_labels(ann_dir: Path) -> list[dict]:
    """One row per annotated frame: which clip, which frame, clot present."""
    rows = []
    for f in sorted(Path(ann_dir).glob("*.json")):
        parts = f.name.split("__", 3)
        if len(parts) != 4:
            continue
        centre, group, subject, rest = parts
        clip = rest.replace(".json", "")
        j = json.loads(f.read_text(encoding="utf-8"))

        per_frame: dict[int, set[str]] = defaultdict(set)
        for a in j.get("annotations") or []:
            name = (a.get("name") or "").strip()
            for fk, fv in (a.get("frames") or {}).items():
                if "polygon" in fv and str(fk).lstrip("-").isdigit():
                    per_frame[int(fk)].add(name)

        for frame, labels in per_frame.items():
            rows.append({
                "centre": centre, "group": group, "subject": subject,
                "clip": clip, "frame": frame,
                "clot": int(any(l.lower() == "clot" for l in labels)),
                "dvt": int(group.upper() == "DVT"),
            })
    return rows


# ---------------------------------------------------------------- features
def encode_frames(rows, dicom_dir: Path, backbone: str = "medsam",
                  checkpoint: str | None = None, device: str = "cuda",
                  img_size: int | None = None):
    """Embed each annotated frame once.

    `medsam` reuses the fine-tuned encoder, which has already adapted to these
    scanners; the alternative is any generic image backbone. Either way the
    encoder stays FROZEN - with 660 positive frames from 26 subjects, fitting a
    linear probe on frozen features is the only defensible option. Full
    fine-tuning saturated in a single epoch on five separate configurations of
    this dataset.
    """
    import torch

    from dvt import step8_compression_phase as S8
    from dvt import step9_medsam_infer as M9

    index = {p.stem: p for p in Path(dicom_dir).rglob("*.dcm")}
    sam, size = M9.load_medsam(checkpoint, img_size, device)
    for p in sam.parameters():
        p.requires_grad = False

    feats, keep, missing = [], [], 0
    cache: dict[str, np.ndarray] = {}

    for i, r in enumerate(rows, 1):
        dcm = index.get(r["clip"])
        if dcm is None:
            missing += 1
            continue
        key = f"{r['clip']}:{r['frame']}"
        if key in cache:
            feats.append(cache[key])
            keep.append(r)
            continue
        try:
            gray = S8.load_clip_frames(str(dcm), max_frames=None,
                                       downsample=1)[min(r["frame"], 10 ** 6)]
        except Exception:
            missing += 1
            continue
        with torch.no_grad():
            emb, _geom = M9.embed_frame(sam, size, gray, device)
            v = emb.mean(dim=(2, 3)).squeeze(0).float().cpu().numpy()
        cache[key] = v
        feats.append(v)
        keep.append(r)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)} frames encoded", flush=True)

    if missing:
        print(f"WARNING {missing} frames could not be encoded")
    return np.stack(feats), keep


# ------------------------------------------------------------------ probe
def log_loss(p, y, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels)
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _fit(X, y, wd, epochs=1500, lr=0.05):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    p0 = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    b = np.log(p0 / (1 - p0))
    mw = vw = np.zeros_like(w)
    mb = vb = 0.0
    for t in range(1, epochs + 1):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) / len(y)
        gw, gb = Z.T @ g + wd * w, g.sum()
        mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw ** 2
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return w, b, mu, sd


def fit_probe(Xtr, ytr, Xva, seed=0, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Logistic probe; weight decay chosen on an inner split against LOG LOSS.

    Selecting on log loss rather than AUC matters: an over-trained probe can
    rank well while producing badly calibrated probabilities, and calibration
    is what a clinical threshold depends on.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    n = max(1, int(len(idx) * 0.25))
    vi, ti = idx[:n], idx[n:]
    if ytr[ti].sum() == 0 or ytr[vi].sum() == 0 or len(ti) < 10:
        w, b, mu, sd = _fit(Xtr, ytr, 1e-2)
    else:
        best = (1e-2, np.inf)
        for wd in grid:
            w, b, mu, sd = _fit(Xtr[ti], ytr[ti], wd)
            p = 1 / (1 + np.exp(-(((Xtr[vi] - mu) / sd) @ w + b)))
            ll = log_loss(p, ytr[vi])
            if ll < best[1]:
                best = (wd, ll)
        w, b, mu, sd = _fit(Xtr, ytr, best[0])
    return 1 / (1 + np.exp(-(((Xva - mu) / sd) @ w + b)))


def operating_point(p, y):
    """Threshold maximising Youden's J, with sensitivity and specificity."""
    best = None
    for t in np.unique(np.round(np.asarray(p, float), 3)):
        pred = (np.asarray(p) >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if best is None or sens + spec > best["sens"] + best["spec"]:
            best = {"threshold": float(t), "sens": round(sens, 3),
                    "spec": round(spec, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return best


def evaluate(X, rows, target="clot", seed=0):
    y = np.array([r[target] for r in rows])
    subj = np.array([r["subject"] for r in rows])
    centre = np.array([r["centre"] for r in rows])

    res = {"n_frames": int(len(y)), "n_positive": int(y.sum()),
           "prevalence": round(float(y.mean()), 4),
           "n_subjects": int(len(set(subj))),
           "feature_dim": int(X.shape[1])}

    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    rng.shuffle(subs)
    val = set(subs[: max(1, len(subs) // 5)])
    va = np.array([s in val for s in subj])
    if va.sum() and (~va).sum() and y[~va].sum() and y[va].sum():
        p = fit_probe(X[~va], y[~va], X[va])
        res["subject_split"] = {
            "n_val": int(va.sum()), "n_val_positive": int(y[va].sum()),
            "auc": round(auc(p, y[va]), 4),
            "logloss": round(log_loss(p, y[va]), 4),
            "baseline_logloss": round(log_loss(np.full(int(va.sum()), float(y[~va].mean())), y[va]), 4),
            "operating_point": operating_point(p, y[va]),
        }

    loco = {}
    for c in np.unique(centre):
        te = centre == c
        # A centre with no positives cannot validate a detector; IT1 and LT1
        # have zero clot frames, so scoring them would be meaningless.
        if not (te.sum() and (~te).sum() and y[~te].sum() and y[te].sum()):
            loco[c] = {"skipped": "no positives in this centre",
                       "n": int(te.sum()), "n_pos": int(y[te].sum())}
            continue
        p = fit_probe(X[~te], y[~te], X[te])
        loco[c] = {"n": int(te.sum()), "n_pos": int(y[te].sum()),
                   "auc": round(auc(p, y[te]), 4),
                   "logloss": round(log_loss(p, y[te]), 4),
                   "baseline_logloss": round(
                       log_loss(np.full(int(te.sum()), float(y[~te].mean())), y[te]), 4)}
    res["leave_one_centre_out"] = loco
    scored = [v["auc"] for v in loco.values() if "auc" in v]
    if scored:
        res["loco_mean_auc"] = round(float(np.mean(scored)), 4)

    # Subject-level aggregation: max frame probability per subject, which is
    # how the clinical question is actually posed - does this patient have DVT.
    if "subject_split" in res:
        p_all = fit_probe(X[~va], y[~va], X[va])
        by_s = defaultdict(list)
        for pi, s in zip(p_all, subj[va]):
            by_s[s].append(pi)
        sv = np.array([max(v) for v in by_s.values()])
        sy = np.array([int(max(y[va][subj[va] == s])) for s in by_s])
        if len(set(sy.tolist())) > 1:
            res["subject_level"] = {"n": len(sy), "n_positive": int(sy.sum()),
                                    "auc": round(auc(sv, sy), 4),
                                    "operating_point": operating_point(sv, sy)}
    return res


def null_baseline(rows, target="clot", dim=64, seed=1, n_repeats=3):
    """What does a probe score on features containing NO image information?

    Each subject gets a random code; frames inherit their subject's code plus
    noise. A held-out subject's code is unseen, so an honest evaluation should
    return ~0.5.

    Measured on this dataset it returns ~0.78 on a subject split. With only 26
    of 81 subjects carrying clot, a probe can find a direction that separates
    the DVT and non-DVT subject groups without seeing a single pixel. Any real
    result must be compared against THIS number, not against 0.5 - and only the
    margin above it is evidence about thrombus.
    """
    out = []
    for k in range(n_repeats):
        rng = np.random.default_rng(seed + k)
        subs = sorted({r["subject"] for r in rows})
        code = {s: rng.normal(size=dim) for s in subs}
        X = np.stack([code[r["subject"]] + 0.01 * rng.normal(size=dim)
                      for r in rows]).astype(np.float32)
        res = evaluate(X, rows, target=target, seed=seed + k)
        out.append({"subject_split_auc": res.get("subject_split", {}).get("auc"),
                    "loco_mean_auc": res.get("loco_mean_auc")})
    def _mean(key):
        vals = [o[key] for o in out if o[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    return {"repeats": n_repeats,
            "subject_split_auc": _mean("subject_split_auc"),
            "loco_mean_auc": _mean("loco_mean_auc"),
            "note": "features contain no image content; real results must beat this"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode")
    e.add_argument("--annotations", required=True)
    e.add_argument("--dicoms", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--img-size", type=int, default=None)
    e.add_argument("--out", default="clot_features.npz")

    p = sub.add_parser("probe")
    p.add_argument("--features", required=True)
    p.add_argument("--target", default="clot", choices=["clot", "dvt"])
    p.add_argument("--skip-null", action="store_true",
                   help="skip the identity-only null baseline (it is slow)")

    a = ap.parse_args()
    if a.cmd == "encode":
        rows = build_frame_labels(Path(a.annotations))
        pos = sum(r["clot"] for r in rows)
        print(f"{len(rows)} annotated frames, {pos} with clot ({pos/max(len(rows),1):.1%})")
        X, keep = encode_frames(rows, Path(a.dicoms), checkpoint=a.checkpoint,
                                device=a.device, img_size=a.img_size)
        np.savez_compressed(
            a.out, X=X,
            **{k: np.array([r[k] for r in keep]) for k in
               ("centre", "group", "subject", "clip", "frame", "clot", "dvt")})
        print(f"encoded {len(X)} frames, dim {X.shape[1]} -> {a.out}")
        return

    d = np.load(a.features, allow_pickle=True)
    rows = [{k: d[k][i] for k in d.files if k != "X"} for i in range(len(d["X"]))]
    for r in rows:
        r["clot"] = int(r["clot"])
        r["dvt"] = int(r["dvt"])
        r["subject"] = str(r["subject"])
        r["centre"] = str(r["centre"])
    res = evaluate(d["X"], rows, target=a.target)
    if not a.skip_null:
        res["null_baseline"] = null_baseline(rows, target=a.target)
    print(json.dumps(res, indent=2))
    nb = res.get("null_baseline", {})
    if nb.get("subject_split_auc") is not None:
        got = res.get("subject_split", {}).get("auc")
        print(f"\nsubject-split AUC {got} vs NULL baseline "
              f"{nb['subject_split_auc']} (identity only, no pixels)")
        if got is not None and got <= nb["subject_split_auc"] + 0.03:
            print("WARNING: not distinguishable from subject-identity leakage.")
            print("Trust leave-one-centre-out instead.")
    print("\n--- benchmark ---")
    print("expert clot presence, subject level: AUC 0.843, sens 0.727, spec 0.959")
    print("expert compression ratio, subject level: AUC 0.848, sens 0.43, spec 0.94")


if __name__ == "__main__":
    main()
