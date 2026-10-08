"""
Step 0 - Load the ThrombUS+ label table and build leakage-safe splits.

`train.csv` supplies every clip-level label the project needs, so the whole
clip-level task is doable with no annotation JSON at all. Polygon JSON is only
required for the segmentation sub-task.

Columns:
    File Name, SubjectID, Age, Sex, Height, Weight, Thigh Circumference,
    Limb, Anatomical Site, Compressibility, Thrombosis

Label semantics, verified against the data:
    Compressibility 2 = fully compressible (normal)
                    1 = partially compressible
                    0 = non-compressible
    Compressibility == 0  =>  Thrombosis == 1   (148/148 cases)
    Compressibility == 2  =>  Thrombosis == 0   (2174/2174 cases)
Only the 19 "partial" clips break the equivalence. The two columns are
therefore near-duplicates: using either as an input feature to predict the
other is label leakage, not feature engineering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# Site codes in train.csv -> the protocol index used in polygon labels
SITE_CODE_TO_INDEX = {"CFV": 1, "FV": 3, "PV": 4, "GS": 3}
SITE_CODE_NAMES = {
    "CFV": "common_femoral_vein",
    "FV": "femoral_vein",
    "PV": "popliteal_vein",
    "GS": "great_saphenous_vein",
}

COMPRESSIBILITY_MEANING = {
    2: "fully_compressible",
    1: "partially_compressible",
    0: "non_compressible",
}


def load_labels(csv_path: Path | None = None) -> pd.DataFrame:
    """Load train.csv and derive centre, clip id, and binary targets."""
    csv_path = Path(csv_path or (C.DATA / "train.csv"))
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    df["clip_id"] = df["File Name"].str.replace(r"\.dcm$", "", regex=True)
    # SubjectID is like 'GR1-0252'; the prefix is the recruiting centre, and
    # each centre used a different ultrasound machine.
    df["centre"] = df["SubjectID"].str.split("-").str[0]
    df["site_name"] = df["Anatomical Site"].map(SITE_CODE_NAMES)

    # Primary target: abnormal = anything less than fully compressible.
    df["abnormal"] = (df["Compressibility"] < 2).astype(int)
    df["compressibility_meaning"] = df["Compressibility"].map(COMPRESSIBILITY_MEANING)
    return df


def check_leakage(df: pd.DataFrame) -> dict:
    """Quantify how far Compressibility and Thrombosis are interchangeable."""
    agree = int(((df["Compressibility"] == 2) == (df["Thrombosis"] == 0)).sum())
    return {
        "n": len(df),
        "agreement": round(agree / len(df), 4),
        "comp0_all_thrombotic": bool((df.loc[df.Compressibility == 0, "Thrombosis"] == 1).all()),
        "comp2_all_nonthrombotic": bool((df.loc[df.Compressibility == 2, "Thrombosis"] == 0).all()),
        "n_partial": int((df["Compressibility"] == 1).sum()),
        "note": ("Thrombosis and Compressibility are near-deterministic. Never "
                 "feed one to a model predicting the other."),
    }


def centre_shortcut_auc(df: pd.DataFrame, target: str = "Thrombosis") -> float:
    """AUC obtainable from centre prevalence alone.

    Because centre and scanner are perfectly confounded in this cohort, this is
    the score a model gets for free by recognising the scanner. Any reported
    AUC below or near this number is evidence of shortcut learning, not
    clinical signal.
    """
    prior = df.groupby("centre")[target].transform("mean")
    return _auc(prior.to_numpy(), df[target].to_numpy())


def _auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ splits
def split_by_subject(df: pd.DataFrame, val_frac: float = 0.2,
                     seed: int = 0, target: str = "abnormal"):
    """Group-stratified split by SubjectID.

    Each subject contributes up to 4 clips (one per anatomical site) and
    thrombus frequently affects several sites in the same leg, so clips within
    a subject are strongly correlated. Splitting by clip leaks.
    """
    subj = (df.groupby("SubjectID")[target].max()
              .rename("subj_pos").reset_index())
    rng = np.random.default_rng(seed)
    val_ids = []
    for pos, grp in subj.groupby("subj_pos"):
        ids = grp["SubjectID"].to_numpy()
        rng.shuffle(ids)
        val_ids.extend(ids[:max(1, int(round(len(ids) * val_frac)))])
    val_ids = set(val_ids)
    return df[~df.SubjectID.isin(val_ids)].copy(), df[df.SubjectID.isin(val_ids)].copy()


def leave_one_centre_out(df: pd.DataFrame):
    """Yield (centre, train, test) with an entire centre held out.

    This is the split that actually tests generalisation. A within-centre split
    lets the model use scanner signature as a proxy for prevalence; holding out
    a centre forces it to transfer across machines, which is what deployment
    requires.
    """
    for centre in sorted(df.centre.unique()):
        te = df[df.centre == centre].copy()
        tr = df[df.centre != centre].copy()
        yield centre, tr, te


def summary(df: pd.DataFrame) -> str:
    lines = [f"{len(df)} clips, {df.SubjectID.nunique()} subjects, "
             f"{df.centre.nunique()} centres"]
    lines.append(f"abnormal (Compressibility<2): {df.abnormal.sum()} "
                 f"({df.abnormal.mean():.1%})")
    lines.append(f"thrombosis: {df.Thrombosis.sum()} ({df.Thrombosis.mean():.1%})")
    lines.append("\nper centre:")
    g = df.groupby("centre").agg(clips=("clip_id", "size"),
                                 subjects=("SubjectID", "nunique"),
                                 thrombosis=("Thrombosis", "sum"))
    g["rate"] = (g.thrombosis / g.clips).map("{:.1%}".format)
    lines.append(g.to_string())
    lines.append(f"\ncentre-only shortcut AUC: {centre_shortcut_auc(df):.3f}")
    lines.append("Any model scoring near this has likely learned the scanner, "
                 "not the vein.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    df = load_labels(sys.argv[1] if len(sys.argv) > 1 else None)
    print(summary(df))
    print("\nleakage check:", check_leakage(df))
