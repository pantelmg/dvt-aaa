"""
Step 4 - Fine-tune MedSAM on DVT compression frames.

Follows MedSAM's published recipe: the prompt encoder is frozen (it already
knows how to encode a box), while the image encoder and mask decoder are
trained. Loss is the unweighted sum of Dice and cross-entropy, which MedSAM
reports as robust across segmentation tasks.

Setup:
    pip install torch torchvision segment-anything
    # MedSAM ViT-B checkpoint from github.com/bowang-lab/MedSAM
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import step3_dataset as S3


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = (p ** 2).sum(dim=(1, 2, 3)) + (target ** 2).sum(dim=(1, 2, 3))
    return (1 - (num + eps) / (den + eps)).mean()


def seg_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def dice_score(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((num + eps) / (den + eps))


class MedSAMWrapper(torch.nn.Module):
    """Thin wrapper exposing (image, box) -> mask logits at 256x256.

    `freeze_encoder=True` is the low-VRAM path and the default on <=8 GB cards.
    Fine-tuning the ViT-B image encoder at 1024x1024 needs roughly 14-18 GB even
    at batch size 1 with AMP, so it does not fit on a 6 GB RTX 3060 laptop.
    Training the mask decoder alone (4.06M parameters) fits in well under 1 GB,
    and because the frozen encoder then produces a fixed embedding per frame,
    those embeddings can be computed once and cached - which is both the memory
    fix and a large speedup.
    """

    def __init__(self, sam_model, freeze_encoder: bool = False):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False  # frozen, per MedSAM
        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward_from_embedding(self, emb, box):
        """Decoder-only path, for use with cached embeddings."""
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(
                points=None, boxes=box[:, None, :], masks=None)
        logits, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return logits

    def forward(self, image, box):
        emb = self.image_encoder(image)                       # (B,256,64,64)
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(
                points=None, boxes=box[:, None, :], masks=None)
        logits, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return logits                                          # (B,1,256,256)


def load_medsam(checkpoint: Path, freeze_encoder: bool = False):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    return MedSAMWrapper(sam, freeze_encoder=freeze_encoder)


@torch.no_grad()
def cache_embeddings(model, samples, out_dir: Path, device="cuda", fp16=True):
    """Run the frozen encoder once per frame and cache the 64x64x256 embedding.

    2.1 MB per frame in fp16. Budget accordingly: 40 clips x 32 frames is
    ~2.7 GB, the full 2919-clip dataset at 32 frames per clip is ~196 GB, so
    plan disk before scaling up.
    """
    from . import step3_dataset as S3
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)
    seen = {}
    for s in samples:
        key = f"{s.clip_id}_{s.frame:04d}"
        if key in seen:
            continue
        img, _, _ = S3.load_sample(s, augment=False)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        with torch.autocast(device_type=device, enabled=fp16):
            emb = model.image_encoder(x)
        torch.save(emb.squeeze(0).half().cpu(), out_dir / f"{key}.pt")
        seen[key] = True
    return len(seen)


def train(args):
    dev = torch.device(args.device)
    samples = S3.build_index()
    tr, va = S3.split_by_patient(samples, val_frac=args.val_frac, seed=args.seed)
    print(f"train {len(tr)} / val {len(va)} instance-frames")
    print(f"train patients {len({s.patient_id for s in tr})}, "
          f"val patients {len({s.patient_id for s in va})}")

    dl_tr = DataLoader(S3.make_torch_dataset(tr, augment=True),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(S3.make_torch_dataset(va, augment=False),
                       batch_size=args.batch_size, num_workers=args.workers)

    model = load_medsam(Path(args.checkpoint),
                        freeze_encoder=args.freeze_encoder).to(dev)
    if args.freeze_encoder:
        print("image encoder FROZEN (low-VRAM mode): training mask decoder only, "
              f"{sum(p.numel() for p in model.mask_decoder.parameters())/1e6:.2f}M params")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        tot = n = 0
        for img, box, mask in dl_tr:
            img, box, mask = img.to(dev), box.to(dev), mask.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, enabled=args.amp):
                loss = seg_loss(model(img, box), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * img.size(0)
            n += img.size(0)
        tr_loss = tot / max(n, 1)

        model.eval()
        scores, cls_scores = [], {}
        with torch.no_grad():
            for (img, box, mask), items in zip(dl_va, _chunks(va, args.batch_size)):
                img, box, mask = img.to(dev), box.to(dev), mask.to(dev)
                d = dice_score(model(img, box), mask).cpu().numpy()
                scores.extend(d.tolist())
                for s, val in zip(items, d.tolist()):
                    cls_scores.setdefault(C.CLASS_NAMES[s.cls], []).append(val)
        va_dice = float(np.mean(scores)) if scores else 0.0
        per_cls = {k: round(float(np.mean(v)), 4) for k, v in cls_scores.items()}
        print(f"epoch {epoch:02d}  train_loss {tr_loss:.4f}  val_dice {va_dice:.4f}  {per_cls}")
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_dice": va_dice, "per_class": per_cls})

        if va_dice > best:
            best = va_dice
            C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), C.MEDSAM_FINETUNED)
            print(f"  saved -> {C.MEDSAM_FINETUNED} (dice {best:.4f})")

    (C.INDEX_DIR / "medsam_history.json").write_text(json.dumps(history, indent=2))


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.MEDSAM_CHECKPOINT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)   # MedSAM's own LR
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="low-VRAM mode: train the mask decoder only "
                         "(required on <=8 GB cards such as an RTX 3060 laptop)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(ap.parse_args())
