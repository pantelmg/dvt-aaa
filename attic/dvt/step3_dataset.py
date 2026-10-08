"""
Step 3 - Build the MedSAM fine-tuning dataset.

MedSAM is a *promptable* model: one box in, one binary mask out. So the unit of
training is not a frame but an (instance, frame) pair.

Splitting is done by PATIENT, never by frame. Frames from one compression clip
are near-duplicates of each other; splitting by frame leaks the validation set
into training and produces meaningless scores.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C
from . import step2_json_to_masks as S2


@dataclass
class Sample:
    clip_id: str
    patient_id: str
    frame: int
    inst_idx: int
    label: str
    cls: int
    site: str | None
    frame_path: str
    box: list[float]


def build_index(manifest_path: Path | None = None) -> list[Sample]:
    """Walk every clip's JSON and enumerate usable (frame, instance) pairs."""
    manifest_path = manifest_path or (C.INDEX_DIR / "clips.json")
    clips = json.loads(Path(manifest_path).read_text())
    samples: list[Sample] = []

    for clip in clips:
        if not clip.get("json_path"):
            continue
        ann = S2.parse_annotation_json(Path(clip["json_path"]), clip["clip_id"])
        h, w = clip["height"], clip["width"]
        for frame in S2.usable_frames(ann):
            boxes = S2.instance_boxes(ann, frame, h, w)
            fp = Path(clip["frame_dir"]) / f"frame_{frame:04d}.png"
            if not fp.exists():
                continue
            for k, b in enumerate(boxes):
                samples.append(Sample(
                    clip_id=clip["clip_id"],
                    patient_id=clip["patient_id"],
                    frame=frame,
                    inst_idx=k,
                    label=b["label"],
                    cls=b["cls"],
                    site=b["site"],
                    frame_path=str(fp),
                    box=b["box"],
                ))
    return samples


def split_by_patient(samples: list[Sample], val_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    patients = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_frac)))
    val_ids = set(patients[:n_val])
    train = [s for s in samples if s.patient_id not in val_ids]
    val = [s for s in samples if s.patient_id in val_ids]
    return train, val


def jitter_box(box, w, h, max_px=C.BOX_JITTER_PX, rng=None):
    """Perturb a box by 0-max_px per edge.

    This mirrors MedSAM's own training recipe and is essential: at inference a
    clinician's hand-drawn box will not hug the vessel the way a polygon-derived
    box does, so training on tight boxes alone produces a brittle model.
    """
    rng = rng or np.random
    x0, y0, x1, y1 = box
    d = rng.randint(0, max_px + 1, size=4)
    return [float(max(0, x0 - d[0])), float(max(0, y0 - d[1])),
            float(min(w - 1, x1 + d[2])), float(min(h - 1, y1 + d[3]))]


def load_sample(s: Sample, augment: bool = True):
    """Return (image_1024x1024x3 float32 [0,1], box_1024_scale, mask_256)."""
    img = np.array(Image.open(s.frame_path).convert("L"))
    h, w = img.shape

    clip = _clip_lookup()[s.clip_id]
    ann = _ann_cache(clip["json_path"], s.clip_id)
    mask = S2.instance_mask(ann, s.frame, s.inst_idx, h, w)

    box = jitter_box(s.box, w, h) if augment else list(s.box)

    # resize to MedSAM's fixed input, scaling the box with it
    size = C.MEDSAM_INPUT_SIZE
    img_r = np.array(Image.fromarray(img).resize((size, size), Image.BICUBIC))
    mask_r = np.array(Image.fromarray(mask * 255).resize((256, 256), Image.NEAREST))
    mask_r = (mask_r > 127).astype(np.float32)

    sx, sy = size / w, size / h
    box_r = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]

    img3 = np.repeat((img_r.astype(np.float32) / 255.0)[..., None], 3, axis=-1)
    return img3, np.array(box_r, dtype=np.float32), mask_r


# -------- tiny caches so we do not re-read/re-parse JSON per sample --------
_CLIPS = None
_ANNS: dict[str, S2.ClipAnnotation] = {}


def _clip_lookup():
    global _CLIPS
    if _CLIPS is None:
        clips = json.loads((C.INDEX_DIR / "clips.json").read_text())
        _CLIPS = {c["clip_id"]: c for c in clips}
    return _CLIPS


def _ann_cache(json_path, clip_id):
    if clip_id not in _ANNS:
        _ANNS[clip_id] = S2.parse_annotation_json(Path(json_path), clip_id)
    return _ANNS[clip_id]


def make_torch_dataset(samples, augment=True):
    """Wrap in a torch Dataset. Imported lazily so the rest of the pipeline
    (parsing, QC, baselines) runs without torch installed."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self, items, aug):
            self.items, self.aug = items, aug

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            img, box, mask = load_sample(self.items[i], augment=self.aug)
            return (torch.from_numpy(img).permute(2, 0, 1),
                    torch.from_numpy(box),
                    torch.from_numpy(mask)[None])

    return _DS(samples, augment)


if __name__ == "__main__":
    samples = build_index()
    train, val = split_by_patient(samples)
    print(f"{len(samples)} instance-frames  |  train {len(train)}  val {len(val)}")
    from collections import Counter
    print("by class:", Counter(C.CLASS_NAMES[s.cls] for s in samples))
    print("by site :", Counter(s.site for s in samples))
    (C.INDEX_DIR / "splits.json").write_text(json.dumps(
        {"train": [s.clip_id for s in train], "val": [s.clip_id for s in val]},
        indent=2))
