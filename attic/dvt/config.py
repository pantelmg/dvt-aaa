"""
Central configuration for the DVT compression-ultrasound pipeline.

Label schema mirrors the US-DICOMizer v5.2 "Annotations" tab exactly, so that
JSON exported by the tool maps onto training classes with no manual renaming.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_ZIP_DIR = DATA / "00_raw_zip"        # ZIPs exported by US-DICOMizer
DICOM_DIR = DATA / "01_dicom"            # unzipped anonymized DICOMs
JSON_DIR = DATA / "02_annotations"       # Darwin-V7 JSON, one per DICOM
FRAME_DIR = DATA / "03_frames"           # PNG frame sequences, one dir per clip
MASK_DIR = DATA / "04_masks"             # rasterized masks (PNG, uint8)
INDEX_DIR = DATA / "05_index"            # manifests / splits (CSV, JSON)
CKPT_DIR = ROOT / "checkpoints"
PROPOSAL_DIR = DATA / "06_proposals"     # model-proposed JSON for human review

ALL_DIRS = [RAW_ZIP_DIR, DICOM_DIR, JSON_DIR, FRAME_DIR, MASK_DIR,
            INDEX_DIR, CKPT_DIR, PROPOSAL_DIR]

# ---------------------------------------------------- polygon label schema
# Exact strings from the US-DICOMizer polygon-segmentation panel.
VEIN_LABELS = [
    "Common Femoral Vein # 1",
    "Common Femoral Vein # 2",
    "Great Saphenous Vein # 3",
    "Femoral Vein # 3",
    "Popliteal Vein # 4",
]
ARTERY_LABELS = [
    "Common Femoral Artery # 1",
    "Common Femoral Artery # 2",
    "Deep Femoral Artery # 2",
    "Femoral Artery # 3",
    "Femoral Artery # 3 ",
    "Popliteal Artery # 4",
]
OTHER_LABELS = ["Clot", "Other"]

# Semantic classes used for training. Vein-vs-artery is kept separate on
# purpose: MedSAM's documented weakness is exactly that veins and arteries
# share a bounding box, so we supervise the distinction explicitly.
CLASS_BACKGROUND = 0
CLASS_VEIN = 1
CLASS_ARTERY = 2
CLASS_CLOT = 3

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_VEIN: "vein_lumen",
    CLASS_ARTERY: "artery",
    CLASS_CLOT: "clot",
}
NUM_CLASSES = 4


def label_to_class(label: str) -> int:
    """Map a US-DICOMizer polygon label string to a training class id."""
    norm = " ".join(label.split()).lower()
    if norm == "clot":
        return CLASS_CLOT
    if "vein" in norm:
        return CLASS_VEIN
    if "artery" in norm:
        return CLASS_ARTERY
    return CLASS_BACKGROUND  # "Other" and anything unrecognized


# Anatomical site inferred from the "# N" suffix in the label, which follows
# the ThrombUS+ acquisition protocol ordering down the leg.
SITE_BY_INDEX = {
    1: "common_femoral",
    2: "proximal_femoral",
    3: "mid_femoral",
    4: "popliteal",
}


def label_to_site(label: str):
    """Extract the protocol site index from a label like 'Popliteal Vein # 4'."""
    if "#" not in label:
        return None
    tail = label.split("#")[-1].strip()
    digits = "".join(c for c in tail if c.isdigit())
    if not digits:
        return None
    return SITE_BY_INDEX.get(int(digits))


# ------------------------------------------------------- clip-level labels
# From the "Classification (per file)" panel.
COMPRESSIBILITY_VALUES = ["Yes", "No", "Partial", "Unknown"]
THROMBOSIS_VALUES = ["Yes", "No", "Unknown"]

# ACEP frame-quality gate. US-DICOMizer only permits polygon annotation on
# frames graded >= 3, so frames below this have no usable ground truth.
MIN_ACEP_GRADE = 3

# ------------------------------------------------------------- model config
MEDSAM_INPUT_SIZE = 1024      # MedSAM expects 1024x1024x3
MEDSAM_CHECKPOINT = CKPT_DIR / "medsam_vit_b.pth"
MEDSAM_FINETUNED = CKPT_DIR / "medsam_dvt_finetuned.pth"

VJEPA_CHECKPOINT = CKPT_DIR / "vjepa2.1_vitl"
VJEPA_CLIP_LEN = 16           # frames per forward pass
VJEPA_FRAME_SIZE = 224

# Box prompt jitter during fine-tuning, matching MedSAM's own training recipe
# (random perturbation of 0-20 pixels around the ground-truth box).
BOX_JITTER_PX = 20


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
