r"""
Fine-tune MedSAM on the ThrombUS+ expert annotations.

    python scripts/finetune_medsam.py \
        --annotations annotations_flat --dicoms /path/to/all_dcm \
        --checkpoint checkpoints/medsam_vit_b.pth \
        --img-size 1024 --batch-size 2 --epochs 30 --device cuda

Data
----
1047 expert files, 81 subjects, 5 centres:
    vein 3357 polygons | artery 2977 | clot 820

MedSAM is promptable, so the training unit is one (box, mask) pair, not one
frame. A frame with a vein, an artery and a clot yields three examples.

Splitting
---------
`--holdout-centre` is the honest evaluation. Frames within a clip are near
duplicates and a subject contributes ~13 files, so a random file split leaks
badly. Centre-holdout also answers the question that matters clinically: does
this work on a scanner it has never seen? V-JEPA scored AUC 1.000 at predicting
which hospital a clip came from, so hardware signature is demonstrably learnable
in this cohort.

Falls back to a subject-level split when no centre is held out. Never a random
file split.

Class imbalance
---------------
Per-class Dice is always reported. Clot is 820 of 7154 polygons, and a model
that ignored it entirely could still look respectable on a pooled average -
while being useless for the one task the diagnosis rests on.

Unlabelled data
---------------
Unlabelled DICOMs cannot validate anything; validation needs ground truth. They
are useful for (a) qualitative review of predictions, and (b) self-training,
where the fine-tuned model labels them and confident predictions are folded back
in. Both come after this script, not during it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported at MODULE level, not inside main(): the loss functions and
# decode_batch live at module scope and reference torch directly, so a
# function-local import leaves them with `NameError: name 'torch' is not
# defined` the moment training starts.
import torch
import torch.nn.functional as F

from dvt import step11_expert_labels as EX


# ------------------------------------------------------------------ dataset
def build_examples(ann_dir: Path, dicom_dir: Path, min_area: int = 40):
    """One example per (file, frame, polygon). Missing DICOMs are reported."""
    from dvt import step8_compression_phase as S8

    print("indexing annotations and DICOMs ...", flush=True)
    items = EX.scan(ann_dir)
    index = {p.stem: p for p in Path(dicom_dir).rglob("*.dcm")}

    examples, missing = [], []
    for e in items:
        dcm = index.get(e.clip)
        if dcm is None:
            missing.append(e.clip)
            continue
        for frame in e.frames():
            for inst in EX.instance_masks(e, frame):
                if int(inst["mask"].sum()) < min_area:
                    continue
                examples.append({
                    "dcm": str(dcm), "frame": frame, "cls": inst["cls"],
                    "box": inst["box"], "label": inst["label"],
                    "centre": e.centre, "subject": e.subject, "group": e.group,
                    "ann": e.path, "h": e.height, "w": e.width,
                })
    return examples, missing, items


def split_examples(examples, holdout_centre: str | None, val_frac: float,
                   seed: int):
    """Centre-holdout if given, else a subject-level split. Never by file."""
    if holdout_centre:
        tr = [x for x in examples if x["centre"] != holdout_centre]
        va = [x for x in examples if x["centre"] == holdout_centre]
        return tr, va, f"holdout centre {holdout_centre}"

    subs = sorted({x["subject"] for x in examples})
    rng = random.Random(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val = set(subs[:n_val])
    tr = [x for x in examples if x["subject"] not in val]
    va = [x for x in examples if x["subject"] in val]
    return tr, va, f"subject split ({len(val)} of {len(subs)} subjects held out)"


_FRAME_CACHE: dict[tuple[str, int], np.ndarray] = {}
_FRAME_CACHE_MAX = 4096


def load_frame(dcm_path: str, frame: int, cache: bool = True):
    """Decode ONE frame from a DICOM.

    The previous implementation called `load_clip_frames(max_frames=None)`,
    which decodes the entire cine - 213 frames - and then throws away all but
    one. At 5048 examples per epoch that is roughly 200x more JPEG decoding
    than required, and it left the GPU idle between short bursts of compute
    (measured: 2525 MiB resident, utilisation mostly 0% with spikes to 98%).

    `iter_pixels(indices=[frame])` decodes only the frame requested. A small
    cache then helps further, because several polygons usually share a frame:
    vein, artery and clot on the same image are three training examples that
    would otherwise each trigger a separate decode.
    """
    import pydicom
    from pydicom.pixels import iter_pixels

    from dvt import step1_extract_frames as S1

    key = (dcm_path, int(frame))
    if cache and key in _FRAME_CACHE:
        return _FRAME_CACHE[key]

    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    idx = min(int(frame), n - 1)

    arr = None
    for a in iter_pixels(dcm_path, indices=[idx]):
        arr = a
        break
    if arr is None:                       # fall back for odd encodings
        arr = pydicom.dcmread(dcm_path).pixel_array
        if arr.ndim >= 3 and arr.shape[0] > 4:
            arr = arr[idx]

    g = S1._to_luma(arr)
    if g.dtype != np.uint8:
        g = S1._scale_clip([g])[0]

    if cache:
        if len(_FRAME_CACHE) >= _FRAME_CACHE_MAX:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[key] = g
    return g


class MedSAMDataset:
    """(image, box, mask, class) examples for MedSAM fine-tuning.

    Defined at MODULE level, not inside a factory. Windows and Python 3.14+
    spawn DataLoader workers via pickle, and a class defined inside a function
    is unpicklable:

        PicklingError: Can't pickle local object make_dataset.<locals>.DS

    Subclassing torch.utils.data.Dataset is done lazily in `make_dataset` so
    the rest of this module still imports without torch.
    """

    def __init__(self, items, img_size, augment, mask_size=None):
        self.items = items
        self.img_size = img_size
        self.augment = augment
        # MedSAM's mask decoder emits at img_size // 4, so the target must
        # follow: 1024 -> 256, 512 -> 128. Hardcoding 256 works only at the
        # default resolution and fails everywhere else with
        #   "Target size (1,1,256,256) must be the same as input (1,1,128,128)"
        self.mask_size = mask_size or (img_size // 4)
        self._ann_cache = {}

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _one_mask(e, frame, label, h, w):
        import cv2

        m = np.zeros((h, w), np.uint8)
        for p in e.polys:
            if p.frame != frame or p.label != label:
                continue
            cv2.fillPoly(m, [np.round(p.paths[0]).astype(np.int32).reshape(-1, 1, 2)], 1)
            for hole in p.paths[1:]:
                cv2.fillPoly(m, [np.round(hole).astype(np.int32).reshape(-1, 1, 2)], 0)
        return m

    def _expert(self, path):
        # cache is per-worker; each process parses the JSONs it actually needs
        if path not in self._ann_cache:
            self._ann_cache[path] = EX.parse_expert_json(Path(path))
        return self._ann_cache[path]

    def __getitem__(self, i):
        it = self.items[i]
        gray = load_frame(it["dcm"], it["frame"])
        h, w = gray.shape

        # Rasterise only the ONE polygon this example needs. instance_masks()
        # builds every polygon on the frame and we discarded all but one, which
        # is 3x the work on a frame carrying vein, artery and clot.
        e = self._expert(it["ann"])
        mask = self._one_mask(e, it["frame"], it["label"], h, w)

        box = list(it["box"])
        # Polygon-derived boxes hug the vessel; a clinician's box will not.
        # MedSAM's own recipe jitters 0-20 px, and without it the model is
        # brittle to exactly the prompts it will meet at inference.
        if self.augment:
            d = np.random.randint(0, 21, size=4)
            box = [max(0, box[0] - d[0]), max(0, box[1] - d[1]),
                   min(w - 1, box[2] + d[2]), min(h - 1, box[3] + d[3])]

        size = self.img_size
        x = torch.from_numpy(gray).float()[None, None] / 255.0
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        x = F.pad(x, (0, size - nw, 0, size - nh))
        x = x.repeat(1, 3, 1, 1)[0]

        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        m = F.interpolate(m, size=(nh, nw), mode="nearest")
        m = F.pad(m, (0, size - nw, 0, size - nh))
        m = F.interpolate(m, size=(self.mask_size, self.mask_size),
                          mode="nearest")[0]

        b = torch.tensor([box[0] * scale, box[1] * scale,
                          box[2] * scale, box[3] * scale], dtype=torch.float)
        return x, b, m, int(it["cls"])


def make_dataset(examples, img_size, augment, mask_size=None):
    """Return a picklable dataset usable by DataLoader.

    DataLoader is duck-typed: it needs only __len__ and __getitem__, so
    MedSAMDataset can be handed over directly. Two earlier attempts were wrong:
    defining the class inside this function made it unpicklable on Windows
    spawn, and `Dataset.register(...)` fails because torch's Dataset is a plain
    class, not an ABC.
    """
    return MedSAMDataset(examples, img_size, augment, mask_size)


# ------------------------------------------------------------------ training
class Progress:
    """Minimal in-place progress bar.

    Deliberately not tqdm: it is an extra dependency for a script that already
    needs torch, pydicom, opencv and segment-anything, and it misbehaves in
    PowerShell when stderr is redirected. This writes a plain carriage-return
    line that works in any terminal.

    Also reports items/sec, which is the number that matters here - the earlier
    bottleneck was the data loader, not the GPU, and throughput makes that
    visible immediately rather than after an epoch.
    """

    def __init__(self, total: int, prefix: str = "", width: int = 28):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = width
        self.n = 0
        self.t0 = time.time()
        self._last = 0.0

    def update(self, k: int = 1, suffix: str = "") -> None:
        self.n += k
        now = time.time()
        # throttle redraws; flushing every step is itself a measurable cost
        if now - self._last < 0.2 and self.n < self.total:
            return
        self._last = now
        frac = self.n / self.total
        done = int(self.width * frac)
        bar = "#" * done + "-" * (self.width - done)
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        sys.stdout.write(
            f"\r  {self.prefix} [{bar}] {self.n}/{self.total} "
            f"{frac*100:5.1f}%  {rate:5.1f} it/s  eta {_hms(eta)}  {suffix}   ")
        sys.stdout.flush()

    def close(self) -> None:
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0.0
        sys.stdout.write(f"\r  {self.prefix} done {self.n} items in {_hms(el)} "
                         f"({rate:.1f} it/s)".ljust(96) + "\n")
        sys.stdout.flush()


def _hms(sec: float) -> str:
    sec = int(max(sec, 0))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def decode_batch(sam, emb, box):
    """Run the mask decoder over a BATCH of (image, box) pairs.

    SAM's mask decoder is written for one image with many prompts. Internally:

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)

    so a batch of B images with B boxes produces B*B image copies against B
    token sets, and fails with

        RuntimeError: The size of tensor a (16) must match the size of
        tensor b (4) at non-singleton dimension 0

    for batch_size=4. Batch 1 hides the bug because 1*1 == 1.

    Decoding per sample and stacking is correct and cheap: the decoder is 4M
    parameters against the encoder's 90M, and the encoder - the expensive part -
    still runs fully batched.
    """
    outs = []
    for i in range(emb.shape[0]):
        sparse, dense = sam.prompt_encoder(
            points=None, boxes=box[i:i + 1, None, :], masks=None)
        lg, _ = sam.mask_decoder(
            image_embeddings=emb[i:i + 1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        outs.append(lg)
    return torch.cat(outs, dim=0)


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = (p ** 2).sum(dim=(1, 2, 3)) + (target ** 2).sum(dim=(1, 2, 3))
    return (1 - (num + eps) / (den + eps)).mean()


def seg_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def dice_score(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (num + eps) / (den + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--dicoms", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="checkpoints/medsam_thrombus.pth")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-decoder", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="low-VRAM mode: train the mask decoder only")
    ap.add_argument("--holdout-centre", default=None,
                    help="e.g. IT1 - the honest cross-scanner evaluation")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 is safest on Windows (spawn). "
                         "On Linux try 4-8 for a real speedup.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from torch.utils.data import DataLoader

    from dvt import step9_medsam_infer as M9

    examples, missing, items = build_examples(Path(a.annotations), Path(a.dicoms))
    print(f"{len(items)} annotation files -> {len(examples)} (box, mask) examples")
    if missing:
        print(f"WARNING {len(missing)} annotated clips had no matching DICOM, e.g. {missing[:3]}")
    if not examples:
        sys.exit("no examples - check --dicoms points at the DICOM tree")

    from collections import Counter
    print("by class:", {EX.CLASS_NAMES[k]: v
                        for k, v in sorted(Counter(x["cls"] for x in examples).items())})

    tr, va, how = split_examples(examples, a.holdout_centre, a.val_frac, a.seed)
    print(f"split: {how} -> train {len(tr)}, val {len(va)}")
    if not va:
        sys.exit("validation split is empty")
    val_cls = Counter(x["cls"] for x in va)
    print("  val by class:", {EX.CLASS_NAMES[k]: v for k, v in sorted(val_cls.items())})

    train_cls = Counter(x["cls"] for x in tr)
    absent = [EX.CLASS_NAMES[c] for c in train_cls if c not in val_cls]
    if absent:
        print(f"\n  WARNING: {absent} present in training but ABSENT from validation.")
        print("  macro-Dice will be computed over the remaining classes only, so")
        print("  it cannot tell you anything about those. With --holdout-centre")
        print("  this usually means that centre has no examples of the class;")
        print("  pick a different centre or use the subject split instead.")
        if "clot" in absent:
            print("  Clot is the class the diagnosis rests on - do not evaluate")
            print("  a clot model on a split that contains no clot.")

    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit("--device cuda but torch.cuda.is_available() is False.\n"
                     "Reinstall the CUDA build:\n"
                     "  pip uninstall -y torch torchvision\n"
                     "  pip install torch torchvision --index-url "
                     "https://download.pytorch.org/whl/cu130")
        print(f"CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    ck_size = M9.checkpoint_img_size(
        torch.load(a.checkpoint, map_location="cpu", weights_only=True))
    if ck_size != a.img_size:
        print(f"note: checkpoint is {ck_size}px, training at {a.img_size}px - "
              "position embeddings will be interpolated")
    sam, img_size = M9.load_medsam(a.checkpoint, ck_size, a.device)
    if ck_size != a.img_size:
        M9._retarget(sam, a.img_size)
        sam.to(a.device)
        img_size = a.img_size

    # Confirm the weights really landed on the GPU. Passing --device is not
    # proof: a silent fallback, or a model built before .to() was applied,
    # trains on CPU at a fraction of the speed while everything still "works".
    where = next(sam.image_encoder.parameters()).device
    print(f"model parameters live on: {where}")
    if a.device.startswith("cuda") and where.type != "cuda":
        sys.exit("model is on CPU despite --device cuda - aborting rather than "
                 "training 90M parameters on the CPU")

    # Probe the decoder for its real output resolution rather than assuming
    # img_size // 4 - the retargeting in step9 changes the embedding grid, and
    # a mismatch here is a hard crash several minutes into training.
    with torch.no_grad():
        probe = torch.zeros(1, 3, img_size, img_size, device=a.device)
        emb = sam.image_encoder(probe)
        sp, dn = sam.prompt_encoder(
            points=None,
            boxes=torch.tensor([[[0.0, 0.0, 10.0, 10.0]]], device=a.device),
            masks=None)
        lg, _ = sam.mask_decoder(
            image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp, dense_prompt_embeddings=dn,
            multimask_output=False)
        mask_size = int(lg.shape[-1])
        del probe, emb, sp, dn, lg
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"decoder output resolution: {mask_size}x{mask_size}")

    # Sort by (file, frame): the three polygons on one frame then land next to
    # each other, so the frame cache hits instead of re-decoding. Shuffling
    # still happens at the batch level via DataLoader.
    tr.sort(key=lambda x: (x["dcm"], x["frame"]))
    va.sort(key=lambda x: (x["dcm"], x["frame"]))

    dl_tr = DataLoader(make_dataset(tr, a.img_size, True, mask_size),
                       batch_size=a.batch_size, shuffle=True,
                       num_workers=a.workers, drop_last=True,
                       persistent_workers=a.workers > 0,
                       pin_memory=a.device.startswith("cuda"),
                       prefetch_factor=4 if a.workers > 0 else None)
    dl_va = DataLoader(make_dataset(va, a.img_size, False, mask_size),
                       batch_size=a.batch_size, num_workers=a.workers,
                       persistent_workers=a.workers > 0,
                       pin_memory=a.device.startswith("cuda"),
                       prefetch_factor=4 if a.workers > 0 else None)
    if a.freeze_encoder:
        for p in sam.image_encoder.parameters():
            p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False              # already encodes boxes, per MedSAM

    groups = [{"params": [p for p in sam.mask_decoder.parameters() if p.requires_grad],
               "lr": a.lr_decoder}]
    if not a.freeze_encoder:
        groups.append({"params": [p for p in sam.image_encoder.parameters()
                                  if p.requires_grad], "lr": a.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=a.amp)

    n_enc = sum(p.numel() for p in sam.image_encoder.parameters() if p.requires_grad)
    n_dec = sum(p.numel() for p in sam.mask_decoder.parameters() if p.requires_grad)
    print(f"trainable: encoder {n_enc/1e6:.1f}M, decoder {n_dec/1e6:.2f}M")

    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(a.epochs):
        sam.train()
        tot = n = 0
        bar = Progress(len(dl_tr), f"epoch {epoch:02d} train")
        for x, box, m, cls in dl_tr:
            x, box, m = x.to(a.device), box.to(a.device), m.to(a.device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=a.device.split(":")[0], enabled=a.amp):
                emb = sam.image_encoder(x)
                logits = decode_batch(sam, emb, box)
                loss = seg_loss(logits, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * x.size(0)
            n += x.size(0)
            bar.update(1, f"loss {tot/max(n,1):.4f}")
        bar.close()

        sam.eval()
        per_cls: dict[int, list[float]] = {}
        vbar = Progress(len(dl_va), f"epoch {epoch:02d} val  ")
        with torch.no_grad():
            for x, box, m, cls in dl_va:
                x, box, m = x.to(a.device), box.to(a.device), m.to(a.device)
                with torch.amp.autocast(device_type=a.device.split(":")[0], enabled=a.amp):
                    emb = sam.image_encoder(x)
                    logits = decode_batch(sam, emb, box)
                d = dice_score(logits.float(), m).cpu().numpy()
                for c, v in zip(cls.numpy().tolist(), d.tolist()):
                    per_cls.setdefault(int(c), []).append(v)
                vbar.update(1)
        vbar.close()

        scores = {EX.CLASS_NAMES[c]: round(float(np.mean(v)), 4)
                  for c, v in sorted(per_cls.items())}
        # mean of PER-CLASS means, so clot is not drowned by vein and artery
        macro = float(np.mean([np.mean(v) for v in per_cls.values()]))
        mem = (f"  gpu {torch.cuda.max_memory_allocated()/1e9:.1f} GB"
               if a.device.startswith("cuda") else "")
        print(f"epoch {epoch:02d}  loss {tot/max(n,1):.4f}  "
              f"macro-Dice {macro:.4f}  {scores}{mem}")
        history.append({"epoch": epoch, "loss": tot / max(n, 1),
                        "macro_dice": macro, "per_class": scores})

        if macro > best:
            best, bad_epochs = macro, 0
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(sam.state_dict(), a.out)
            print(f"  saved -> {a.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"early stop: no improvement in {a.patience} epochs")
                break

    Path(a.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest macro-Dice {best:.4f}")


if __name__ == "__main__":
    # Required on Windows/spawn: without the guard each worker re-executes
    # module scope and re-enters main().
    import multiprocessing
    multiprocessing.freeze_support()
    main()
