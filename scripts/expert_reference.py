r"""
What the expert polygons say, on the clips this run actually covered.

Three questions the automated numbers cannot answer on their own:

1. What is the compression ratio's ceiling ON THESE CLIPS? The 0.7167 / 0.8491
   benchmarks were computed on a different clip set, so comparing an automated
   0.576 against them is not a fair test.
2. How large is a clot relative to the vein it sits in, really? The clot bound
   should be measured, not remembered.
3. Are the automated masks on the vein at all? Boundary error is nearly free
   (degradation floor 0.68), so a low AUC means either a weak feature or the
   wrong structure - and those imply opposite next steps.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvt.step14_medsam2 import role_of  # noqa: E402


def role(label: str):
    """`role_of` with None instead of 'other', which is what this script reads."""
    r = role_of(label)
    return None if r == "other" else r


def raster(paths, w, h):
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for i, path in enumerate(paths):
        pts = [(float(p["x"]), float(p["y"])) for p in path]
        if len(pts) >= 3:
            d.polygon(pts, fill=0 if i else 1)   # extra paths are holes
    return np.array(img, dtype=bool)


def load(path: Path):
    """-> (frames{idx: {role: [mask...]}}, w, h). Keyframes only.

    `keyframe: false` means the tool INTERPOLATED the polygon rather than a
    human drawing it, so those are excluded - training or benchmarking on them
    is measuring the tool's guesses.
    """
    doc = json.loads(path.read_text())
    slot = (doc.get("item", {}).get("slots") or [{}])[0]
    w, h = slot.get("width"), slot.get("height")
    if not w or not h:
        return {}, 0, 0
    out = defaultdict(lambda: defaultdict(list))
    for ann in doc.get("annotations", []):
        r = role(ann.get("name", ""))
        if r is None:
            continue
        for k, fr in (ann.get("frames") or {}).items():
            if not k.isdigit() or not fr.get("keyframe", True):
                continue
            poly = fr.get("polygon")
            if poly and poly.get("paths"):
                out[int(k)][r].append(raster(poly["paths"], w, h))
    return out, w, h


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return None, len(y)
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg))), len(y)


def main(expert_dir: str, auto_dir: str | None, run_report: str | None):
    files = sorted(Path(expert_dir).glob("*.json"))
    print(f"expert files: {len(files)}")

    # restrict to the clips the run covered, if a report is given
    covered = None
    if run_report:
        rep = json.loads(Path(run_report).read_text())
        covered = {r["clip"] for r in rep}
        print(f"clips in run report: {len(covered)}")

    rows, clot_ratios, clot_contain = [], [], []
    for p in files:
        centre, group, subject, clipname = p.stem.split("__")
        if covered is not None and clipname not in covered:
            continue
        frames, w, h = load(p)
        if not frames:
            continue

        vein_area, art_area = {}, {}
        for idx, roles in frames.items():
            if "vein" in roles:
                m = np.logical_or.reduce(roles["vein"])
                vein_area[idx] = int(m.sum())
                if "clot" in roles:
                    c = np.logical_or.reduce(roles["clot"])
                    if m.sum() > 20 and c.sum() > 20:
                        clot_ratios.append(c.sum() / m.sum())
                        clot_contain.append((c & m).sum() / c.sum())
            if "artery" in roles:
                art_area[idx] = int(np.logical_or.reduce(roles["artery"]).sum())

        vals = [a for a in vein_area.values() if a > 0]
        ratio = (min(vals) / max(vals)) if len(vals) >= 2 else np.nan
        has_clot = any("clot" in r for r in frames.values())
        rows.append({"clip": clipname, "centre": centre, "subject": subject,
                     "dvt": 1 if group == "DVT" else 0, "ratio": ratio,
                     "clot": 1.0 if has_clot else 0.0,
                     "n_vein_frames": len(vals)})

    print(f"matched clips with expert polygons: {len(rows)}")
    if not rows:
        return

    y = [r["dvt"] for r in rows]
    print(f"  DVT+ {sum(y)}  DVT- {len(y)-sum(y)}")
    nf = [r["n_vein_frames"] for r in rows]
    print(f"  vein keyframes per clip: median {np.median(nf):.0f}, "
          f"clips with <2 (ratio uncomputable): {sum(1 for x in nf if x < 2)}")

    print("\n--- CEILING ON THESE CLIPS (expert polygons) ---")
    a, n = auc(y, [r["ratio"] for r in rows])
    print(f"  compression ratio  vessel AUC {a if a is None else round(a,4)}  (n={n})")
    a2, n2 = auc(y, [r["clot"] for r in rows])
    print(f"  clot presence      vessel AUC {a2 if a2 is None else round(a2,4)}  (n={n2})")

    # subject level: max across the subject's vessels
    for feat, label in (("ratio", "compression ratio"), ("clot", "clot presence")):
        sub = defaultdict(list)
        lab = {}
        for r in rows:
            if not np.isnan(r[feat]):
                sub[r["subject"]].append(r[feat])
            lab[r["subject"]] = r["dvt"]
        s = [max(v) for k, v in sub.items()]
        yy = [lab[k] for k in sub]
        a3, n3 = auc(yy, s)
        print(f"  {label:<18} subject AUC {a3 if a3 is None else round(a3,4)}  "
              f"(n={n3} subjects)")

    print("\n--- CLOT GEOMETRY (for the bound) ---")
    if clot_ratios:
        cr = np.array(clot_ratios)
        cc = np.array(clot_contain)
        print(f"  co-annotated clot/vein frames: {len(cr)}")
        print(f"  clot area / vein area: median {np.median(cr):.2f}  "
              f"p90 {np.percentile(cr,90):.2f}  p99 {np.percentile(cr,99):.2f}  "
              f"max {cr.max():.2f}")
        print(f"  fraction of clot inside the vein: median {np.median(cc):.2f}  "
              f"p10 {np.percentile(cc,10):.2f}")
        for t in (1.5, 2.0, 2.5, 3.0):
            print(f"    a bound of {t:.1f} would reject "
                  f"{(cr>t).sum()}/{len(cr)} = {(cr>t).mean():.1%} of EXPERT frames")

    # --- auto vs expert IoU
    if auto_dir:
        print("\n--- ARE THE AUTO MASKS ON THE VEIN? ---")
        ious, per_clip = [], []
        for p in files:
            centre, group, subject, clipname = p.stem.split("__")
            ap = Path(auto_dir) / f"{clipname}_darwin.json"
            if not ap.exists():
                continue
            ef, w, h = load(p)
            if not ef:
                continue
            af, aw, ah = load(ap)
            if not af or (aw, ah) != (w, h):
                continue
            vals, art_vals = [], []
            for idx, er in ef.items():
                if "vein" not in er or idx not in af or "vein" not in af[idx]:
                    continue
                e = np.logical_or.reduce(er["vein"])
                a_ = np.logical_or.reduce(af[idx]["vein"])
                u = (e | a_).sum()
                vals.append(((e & a_).sum() / u) if u else 0.0)
                if "artery" in er:
                    ea = np.logical_or.reduce(er["artery"])
                    ua = (ea | a_).sum()
                    art_vals.append(((ea & a_).sum() / ua) if ua else 0.0)
            if vals:
                per_clip.append((clipname, float(np.median(vals)),
                                 float(np.median(art_vals)) if art_vals else np.nan))
                ious.extend(vals)
        if per_clip:
            med = np.array([x[1] for x in per_clip])
            art = np.array([x[2] for x in per_clip])
            print(f"  clips compared: {len(per_clip)}  frames: {len(ious)}")
            print(f"  median IoU (auto vein vs expert vein): {np.median(ious):.4f}")
            print(f"  clips with median IoU > 0.5 : {(med>0.5).sum()} "
                  f"({(med>0.5).mean():.1%})")
            print(f"  clips with median IoU < 0.1 : {(med<0.1).sum()} "
                  f"({(med<0.1).mean():.1%})")
            good_art = art[~np.isnan(art)]
            if len(good_art):
                print(f"  median IoU (auto VEIN vs expert ARTERY): "
                      f"{np.median(good_art):.4f}")
                print(f"    clips where the auto 'vein' matches the expert "
                      f"ARTERY better: "
                      f"{int(np.nansum(art > med))}/{len(per_clip)}")
            print("\n  reference: heuristic seeder scored median 0.0004, "
                  "6% of clips >0.5; the trained detector was expected ~0.74")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
