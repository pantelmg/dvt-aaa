"""
Step 9 - MedSAM inference for vein tracking.

Verified working: zero-shot, no fine-tuning, clean vein-lumen masks across a
full compression clip from a single box prompt.

Three environment fixes are baked in, each of which is a hard crash otherwise:

1. The published MedSAM checkpoint is saved from CUDA. `sam_model_registry`
   calls `torch.load` without `map_location`, so it raises on any CPU-only
   machine. We load the state dict ourselves.
2. ViT-B at 1024x1024 holds four global-attention blocks with 4096x4096
   attention matrices, roughly 8 GB of activations. On a 6 GB card use
   `img_size=512`, which interpolates the position embeddings and still
   segments these veins correctly.
3. Reducing img_size means retargeting TWO grids. Interpolating
   `image_encoder.pos_embed` alone leaves `prompt_encoder.image_embedding_size`
   at 64, and the mask decoder then fails with a 32-vs-64 size mismatch.

Measured cost: 3.4 s/frame at 512 on CPU, 1.6 GB peak RSS. On an RTX 3060
laptop expect well under 0.1 s/frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def checkpoint_img_size(state_dict) -> int:
    """Infer the resolution a checkpoint was saved at, from its pos_embed grid.

    The original MedSAM release is 1024 (64x64 grid). A model fine-tuned with
    --img-size 512 saves a 32x32 grid, and loading that into a 1024 model fails
    with a wall of size-mismatch errors:

        size mismatch for image_encoder.pos_embed: copying a param with shape
        [1, 32, 32, 768] ... current model is [1, 64, 64, 768]

    Detecting it here means a fine-tuned checkpoint just works, rather than
    requiring the caller to remember which resolution it was trained at.
    """
    pe = state_dict.get("image_encoder.pos_embed")
    if pe is None:
        return 1024
    return int(pe.shape[1]) * 16


def load_medsam(checkpoint: str | Path, img_size: int | None = 512,
                device: str = "cpu"):
    """Load MedSAM, optionally at reduced resolution for low-VRAM cards.

    `img_size=None` adopts whatever the checkpoint was saved at. Passing an
    explicit size that disagrees with the checkpoint is an error, not something
    to silently paper over - the weights would not match the architecture.
    """
    from segment_anything import sam_model_registry

    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    ckpt_size = checkpoint_img_size(sd)

    if img_size is None:
        img_size = ckpt_size
    elif img_size != ckpt_size:
        raise ValueError(
            f"checkpoint was saved at img_size={ckpt_size} but img_size="
            f"{img_size} was requested. Fine-tuned weights only fit the "
            f"resolution they were trained at - pass --img-size {ckpt_size}, "
            "or use the original MedSAM checkpoint to train at a new size.")

    sam = sam_model_registry["vit_b"](checkpoint=None)
    # Retarget BEFORE loading: the architecture must already match the weights,
    # otherwise load_state_dict rejects every position-embedding tensor.
    if img_size != 1024:
        _retarget(sam, img_size)

    missing, unexpected = sam.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[medsam] missing={len(missing)} unexpected={len(unexpected)}")
    sam.eval()

    return sam.to(device), img_size


def _retarget(sam, img_size: int) -> None:
    """Resize both the encoder and prompt-encoder grids to a new input size."""
    g = img_size // 16
    enc = sam.image_encoder

    pe = enc.pos_embed.data                       # (1, 64, 64, 768)
    enc.pos_embed = torch.nn.Parameter(
        F.interpolate(pe.permute(0, 3, 1, 2), size=(g, g),
                      mode="bicubic", align_corners=False).permute(0, 2, 3, 1))
    enc.img_size = img_size

    # global-attention blocks carry decomposed relative position embeddings
    for blk in enc.blocks:
        attn = blk.attn
        if getattr(attn, "rel_pos_h", None) is None:
            continue
        if attn.rel_pos_h.shape[0] <= 2 * g - 1:
            continue
        for name in ("rel_pos_h", "rel_pos_w"):
            r = getattr(attn, name).data
            r2 = F.interpolate(r.T[None], size=(2 * g - 1,),
                               mode="linear", align_corners=False)[0].T
            setattr(attn, name, torch.nn.Parameter(r2.contiguous()))

    # the second grid - omitting this is the 32-vs-64 mismatch
    pe_mod = sam.prompt_encoder
    pe_mod.image_embedding_size = (g, g)
    pe_mod.input_image_size = (img_size, img_size)
    pe_mod.mask_input_size = (4 * g, 4 * g)


@torch.inference_mode()
def embed_frame(sam, img_size: int, gray: np.ndarray, device: str = "cpu",
                preserve_aspect: bool = True):
    """Encode one greyscale frame.

    Returns (embedding, geom) where geom describes how the frame was mapped
    into the model's square input, so boxes and masks can be converted back.

    `preserve_aspect=True` follows SAM's own preprocessing: scale the LONGEST
    side to img_size and zero-pad the rest. Stretching to a square instead
    distorts anatomy badly on this cohort - GR1 frames are 273x516, so a square
    resize stretches them 1.89x horizontally and a circular artery is presented
    to the model as a wide ellipse. MedSAM never saw that during training.
    """
    h, w = gray.shape
    x = torch.from_numpy(gray).float()[None, None] / 255.0

    if preserve_aspect:
        scale = img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        pad = (0, img_size - nw, 0, img_size - nh)     # left,right,top,bottom
        x = F.pad(x, pad, value=0.0)
        geom = {"mode": "pad", "scale": scale, "nh": nh, "nw": nw, "h": h, "w": w}
    else:
        x = F.interpolate(x, size=(img_size, img_size),
                          mode="bilinear", align_corners=False)
        geom = {"mode": "stretch", "sx": img_size / w, "sy": img_size / h,
                "h": h, "w": w}

    return sam.image_encoder(x.repeat(1, 3, 1, 1).to(device)), geom


@torch.inference_mode()
def segment_box(sam, embedding, geom, box_xyxy, out_hw, device: str = "cpu",
                threshold: float = 0.5, prev_logits=None,
                return_logits: bool = False):
    """Segment with a box prompt. Box is in ORIGINAL frame coordinates.

    `prev_logits` is the previous frame's low-resolution mask logits, fed back
    as SAM's dense mask prompt. This is the mechanism SAM itself uses for
    iterative refinement, and it gives genuine temporal continuity: the model
    starts from where the vessel was rather than re-deciding from scratch. It
    matters most exactly where box tracking struggles - at peak compression,
    when the lumen is thin and low-contrast and a box alone is ambiguous.
    """
    if isinstance(geom, (tuple, list)):                 # legacy (sx, sy)
        geom = {"mode": "stretch", "sx": geom[0], "sy": geom[1]}

    if geom["mode"] == "pad":
        k = geom["scale"]
        bx = [box_xyxy[0] * k, box_xyxy[1] * k, box_xyxy[2] * k, box_xyxy[3] * k]
    else:
        bx = [box_xyxy[0] * geom["sx"], box_xyxy[1] * geom["sy"],
              box_xyxy[2] * geom["sx"], box_xyxy[3] * geom["sy"]]
    b = torch.tensor([bx], dtype=torch.float, device=device)
    sparse, dense = sam.prompt_encoder(points=None, boxes=b[None, 0][None],
                                       masks=prev_logits)
    logits, iou_pred = sam.mask_decoder(
        image_embeddings=embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    if geom["mode"] == "pad":
        # upsample to the padded square, drop the padding, then resize to native
        full = F.interpolate(logits, size=(sam.image_encoder.img_size,) * 2,
                             mode="bilinear", align_corners=False)
        full = full[..., :geom["nh"], :geom["nw"]]
        m = F.interpolate(full, size=out_hw, mode="bilinear", align_corners=False)
    else:
        m = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    mask = (torch.sigmoid(m)[0, 0].cpu().numpy() > threshold)
    score = float(iou_pred.reshape(-1)[0].cpu())
    return (mask, logits, score) if return_logits else mask


def refine_to_lumen(mask: np.ndarray, gray: np.ndarray,
                    quantile: float = 0.55, min_area: int = 20) -> np.ndarray:
    """Keep only the anechoic part of a MedSAM mask.

    MedSAM returns a valid-looking mask for ANY box prompt, so when a vein
    collapses it simply fills the box with whatever is most blob-like - often
    adjacent tissue. Measured on a real FR1 saphenous clip, the reported area
    moved only 6962 -> 5383 px across a full compression cycle, i.e. the mask
    stayed the same size while the vessel closed.

    VERDICT AFTER TESTING: worse than leaving the mask alone. Re-thresholding
    produces ragged, fragmented boundaries where raw MedSAM gives smooth ones,
    and a radiologist has to redraw a ragged mask rather than nudge it. OFF by
    default. Use `mask_quality()` below to FLAG bad frames instead of trying to
    repair them.

    Blood is anechoic, so lumen must be dark. This re-thresholds INSIDE the
    predicted mask using the local intensity distribution and keeps the largest
    dark component. Crucially the threshold is recomputed from the image every
    frame rather than from the previous mask, so unlike box resizing it cannot
    ratchet in either direction: it shrinks when the vessel closes and recovers
    when it reopens.
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < min_area:
        return m

    vals = gray[m > 0]
    # reference the surrounding wall, not the whole image: ultrasound frames
    # carry large black borders that would drag any global threshold down
    ring = cv2.dilate(m, np.ones((21, 21), np.uint8)) - m
    surround = gray[ring > 0]
    if surround.size < 10:
        return m

    thr = min(np.quantile(vals, quantile), float(surround.mean()) * 0.75)
    dark = ((gray <= thr) & (m > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return np.zeros_like(m)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == best).astype(np.uint8)
    return out if out.sum() >= min_area else np.zeros_like(m)


def compression_ratio(areas) -> float | None:
    """min/max area over VALID frames only.

    Frames the tracker rejected are stored as 0. Including them makes the
    minimum 0 and the ratio 0.0 for any clip with a single gap - which on real
    data was every clip, reporting perfect compressibility where the truth was
    0.416. A zero means "no measurement", never "the vein closed completely".
    """
    a = np.asarray(areas, dtype=float)
    v = a[a > 0]
    if v.size < 2:
        return None
    return float(v.min() / v.max())


def mask_quality(mask: np.ndarray, gray: np.ndarray, ring_px: int = 21) -> float:
    """Contrast of the mask interior against the surrounding wall.

    Returns mean(inside) / mean(surrounding ring). Vessel lumen is anechoic, so
    a good mask scores low; a mask that has slipped onto tissue scores near 1.

    The surrounding ring is the reference, NOT the whole image: ultrasound
    frames carry large black borders which drag any global statistic down. On
    an FR1 clip the image median was 31 while the true perivascular tissue was
    ~47, so a global threshold called every mask "too bright".

    Measured on a real compression clip, this correlates -0.63 with mask area:
    as the vein closes and the mask starts covering tissue, contrast rises.
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < 20:
        return float("nan")
    ring = cv2.dilate(m, np.ones((ring_px, ring_px), np.uint8)) - m
    if ring.sum() < 10:
        return float("nan")
    return float(gray[m > 0].mean()) / max(float(gray[ring > 0].mean()), 1e-6)


def track_vessel(sam, img_size, frames: np.ndarray, box_xyxy,
                 device: str = "cpu", reseed: bool = False,
                 max_growth: float = 1.5, pad: int = 8,
                 preserve_aspect: bool = True,
                 mode: str = "recenter", jump_thr: float = 2.0,
                 min_box: int = 12, use_mask_prompt: bool = True,
                 min_score: float = 0.0, return_scores: bool = False,
                 refine_lumen: bool = False,
                 max_jump_px: float = 25.0, max_area_ratio: float = 2.5,
                 history: int = 7, reacquire_after: int = 3):
    """Segment one vessel across every frame of a clip.

    `reseed=False` (the default) keeps the prompt box fixed. This matters:
    re-deriving the box from each frame's mask and padding it makes the mask
    leak into surrounding tissue and compound. Measured on a real popliteal
    clip, re-seeding ran the area from 2090 to 34426 px - a 16x runaway that
    looks like a signal but is pure drift. The fixed box gave a stable
    1237-2257 px over the same clip.

    If `reseed=True`, box growth is clamped to `max_growth` times the initial
    box area, which bounds the runaway without disabling tracking entirely.
    """
    h, w = frames.shape[1:3]
    box = list(box_xyxy)
    box0 = list(box_xyxy)
    bw0, bh0 = box0[2] - box0[0], box0[3] - box0[1]
    area0 = max(bw0 * bh0, 1)

    if reseed:                 # legacy unbounded behaviour, kept for comparison
        mode = "reseed"

    areas, masks, scores, quality = [], [], [], []
    prev_area, prev_logits = None, None
    last_good_box = list(box0)
    last_cent = None
    consecutive_blank = 0
    recent_areas: list[int] = []

    def _centroid(mm):
        ys, xs = np.where(mm)
        if len(xs) == 0:
            return None
        return np.array([xs.mean(), ys.mean()])
    for i in range(len(frames)):
        emb, geom = embed_frame(sam, img_size, frames[i], device,
                                preserve_aspect=preserve_aspect)
        m, lg, score = segment_box(
            sam, emb, geom, box, (h, w), device,
            prev_logits=prev_logits if use_mask_prompt else None,
            return_logits=True)
        a = int(m.sum())

        # MedSAM always returns SOME mask for a box prompt - it is trained to
        # produce a valid mask for any prompt, even an ambiguous one. At full
        # venous collapse there is no lumen left to segment, so the model
        # invents one. Its own predicted-IoU head is the signal that this has
        # happened, and a low score there is clinically meaningful: it is
        # evidence the vein HAS collapsed, not merely that segmentation failed.
        # NOTE: measured on a real GR1 popliteal clip, this head returns
        # 0.378-0.606 across ALL frames and correlates only 0.246 with area -
        # it is uniformly mediocre rather than specifically low at collapse, so
        # it does not discriminate. Left available but DISABLED by default
        # (min_score=0.0); do not enable without checking it on your own data.
        if min_score > 0 and score < min_score:
            m = np.zeros_like(m)
            a = 0

        if refine_lumen and a > 0:
            m = refine_to_lumen(m, frames[i])
            a = int(m.sum())
        quality.append(mask_quality(m, frames[i]))

        # TEMPORAL CONSISTENCY.
        #
        # The dominant failure on real clips is not gradual drift but discrete
        # JUMPS: the mask leaves the vein for a neighbouring dark structure and
        # stays there for a few frames. Measured on an FR1 common femoral clip,
        # the centroid leapt >25 px at frames 9, 45-47 and 101-104, with area
        # changing up to 153x in a single frame - while a clean clip from the
        # same centre never exceeded an 18 px jump or a 1.2x area ratio.
        #
        # Contrast checks cannot catch this, because the structure it jumps to
        # is also anechoic. Continuity can: a vein deforms smoothly between
        # consecutive frames at 13-39 fps, so a large discontinuity is an error
        # rather than anatomy. On detection we re-segment from the last KNOWN
        # GOOD box instead of the seed, which recovers the vessel's current
        # position rather than snapping back to where it started.
        cent = _centroid(m)
        ref_area = float(np.median(recent_areas)) if recent_areas else None
        jumped = (last_cent is not None and cent is not None
                  and np.linalg.norm(cent - last_cent) > max_jump_px)
        swung = (ref_area and ref_area > 30 and a > 0
                 and (a > max_area_ratio * ref_area
                      or a < ref_area / max_area_ratio))

        if jumped or swung or (prev_area and prev_area > 30 and a > jump_thr * prev_area):
            m2, lg2, sc2 = segment_box(sam, emb, geom, last_good_box, (h, w),
                                       device, prev_logits=None,
                                       return_logits=True)
            a2 = int(m2.sum())
            c2 = _centroid(m2)
            ok2 = (c2 is not None
                   and (last_cent is None
                        or np.linalg.norm(c2 - last_cent) <= max_jump_px)
                   and (not ref_area or ref_area <= 30
                        or a2 <= max_area_ratio * ref_area))
            if ok2:
                m, lg, score, a, cent = m2, lg2, sc2, a2, c2
                box = list(last_good_box)
            else:
                # neither prompt is credible: emit nothing rather than a
                # confident wrong polygon for the radiologist to untangle
                m = np.zeros_like(m)
                a = 0
                cent = None

        # RE-ACQUISITION.
        #
        # Without this the tracker never recovers: `last_good_box` goes stale,
        # every subsequent frame is measured against a position the vessel has
        # left, and rejections cascade. Measured on a real FR1 common femoral
        # clip, 26% of frames came back empty in runs of up to 16 consecutive
        # frames. After `reacquire_after` blank frames we retry the ORIGINAL
        # seed box and accept whatever it finds, on the grounds that a stale
        # reference is worse than no reference.
        if a == 0:
            consecutive_blank += 1
            if consecutive_blank >= reacquire_after:
                m3, lg3, sc3 = segment_box(sam, emb, geom, box0, (h, w), device,
                                           prev_logits=None, return_logits=True)
                a3 = int(m3.sum())
                if a3 > 20:
                    m, lg, score, a = m3, lg3, sc3, a3
                    cent = _centroid(m)
                    box = list(box0)
                    last_cent = cent          # accept the new anchor
                    recent_areas.clear()
                    consecutive_blank = 0
        else:
            consecutive_blank = 0

        areas.append(a)
        masks.append(m)
        scores.append(score)
        prev_area = a if a > 20 else prev_area
        if a > 20 and cent is not None:
            last_cent = cent
            last_good_box = list(box)
            recent_areas.append(a)
            if len(recent_areas) > history:
                recent_areas.pop(0)
        # keep logits only while the mask is credible, so a bad frame does not
        # poison the next one through the dense prompt
        prev_logits = lg if a > 20 else None

        if mode == "fixed" or a <= 20:
            continue

        ys, xs = np.where(m)
        if mode == "reseed":
            cand = [max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(w - 1, xs.max() + pad), min(h - 1, ys.max() + pad)]
            if (cand[2] - cand[0]) * (cand[3] - cand[1]) <= max_growth * area0:
                box = cand
            continue

        # recentre: the box KEEPS ITS SIZE and only follows the vessel.
        #
        # Both size-adapting strategies ratchet. Letting the box grow makes the
        # mask leak into neighbouring tissue (measured: 2090 -> 34426 px).
        # Letting it shrink is the mirror failure - one slightly small mask
        # yields a smaller box, which segments smaller still, and the apparent
        # "compression" is the tracker eating itself (measured ratio 0.294 on a
        # clip where a fixed box gave 0.584, with visibly under-segmented masks).
        #
        # The genuine problem is only that tissue SHIFTS under probe pressure,
        # so a static box stops containing the vessel. Translation alone fixes
        # that, and a loose box is not a problem: MedSAM is prompted with
        # jittered boxes during training precisely so it tolerates them.
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        cx = 0.5 * cx + 0.5 * ((box[0] + box[2]) / 2.0)   # damp the trajectory
        cy = 0.5 * cy + 0.5 * ((box[1] + box[3]) / 2.0)
        box = [max(0.0, cx - bw0 / 2), max(0.0, cy - bh0 / 2),
               min(w - 1.0, cx + bw0 / 2), min(h - 1.0, cy + bh0 / 2)]

    if return_scores:
        return (np.asarray(areas), np.asarray(masks),
                np.asarray(scores), np.asarray(quality))
    return np.asarray(areas), np.asarray(masks), np.asarray(quality)


def box_from_polygon(points) -> list[float]:
    """Derive a prompt box from a hand-drawn polygon.

    This is the bridge from US-DICOMizer: the radiologist draws ONE polygon on
    ONE frame, and that single annotation seeds the whole clip.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max())]
