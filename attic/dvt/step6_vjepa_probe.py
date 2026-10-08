"""
Step 6 - V-JEPA 2.1 frozen-feature probe for DVT detection.

WHY THIS EXISTS
---------------
The geometric route was tested and failed. On 31 clips with ground truth, six
features derived from MedSAM masks all scored at chance for thrombosis:

    area_ratio 0.571 | minor_ratio 0.559 | ecc_max    0.487
    circ_min   0.525 | ecc_range   0.454 | circ_range 0.349

So the remaining idea is to skip masks entirely and learn from pixels. V-JEPA
sidesteps the whole error chain behind those numbers - wrong vessel, tracking
drift, jumps, blank frames - because it never segments anything.

THE PROTOCOL
------------
Freeze the encoder, fit a LINEAR probe. This is the authors' own evaluation
protocol and it is deliberately cheap: it answers "do these features contain
the signal" before anyone spends compute on fine-tuning. A linear probe on
frozen features is also naturally well calibrated, which matters because the
Kaggle DVT task is scored on LOG LOSS, not AUC.

THE NUMBER TO BEAT
------------------
0.2381 - the log loss of predicting the training prevalence (0.069) for every
clip. Not 0.5, and not AUC. If the probe cannot beat a constant, that is the
answer and the write-up says so.

Leave-one-centre-out is reported alongside the random split: centre prevalence
alone gives AUC 0.709 in this cohort because each hospital used a different
scanner, so a within-centre score cannot separate signal from hardware.

SETUP
-----
    pip install torch torchvision timm einops
    # weights download automatically on first use via torch.hub

Variants, from the official hubconf:
    vjepa2_1_vit_base_384    <- start here on a 6 GB card
    vjepa2_1_vit_large_384
    vjepa2_1_vit_giant_384
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C

VJEPA_VARIANTS = {
    "base": "vjepa2_1_vit_base_384",
    "large": "vjepa2_1_vit_large_384",
    "giant": "vjepa2_1_vit_giant_384",
}
DEFAULT_FRAMES = 64          # V-JEPA 2.1 backbones are built for 64 frames
DEFAULT_SIZE = 384

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Hugging Face mirrors, documented in the same README. These avoid the
# torch.hub path entirely, which matters because of the bug described below.
HF_VARIANTS = {
    "base": "facebook/vjepa2-vitl-fpc64-256",
    "large": "facebook/vjepa2-vitl-fpc64-256",
    "huge": "facebook/vjepa2-vith-fpc64-256",
    "giant": "facebook/vjepa2-vitg-fpc64-256",
    "giant384": "facebook/vjepa2-vitg-fpc64-384",
}


def _patch_hub_base_url():
    """Repair the hardcoded dev URL in the upstream repo.

    facebookresearch/vjepa2 `src/hub/backbones.py` currently ships:

        # VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
        VJEPA_BASE_URL = "http://localhost:8300"

    Someone committed a local test server, so every `torch.hub.load(...,
    pretrained=True)` tries to download from the user's own machine and fails
    with connection refused. We rewrite the constant in the cached checkout
    after torch.hub clones it.
    """
    import re

    hub_dir = Path(torch.hub.get_dir())
    hits = list(hub_dir.glob("*vjepa2*/src/hub/backbones.py"))
    patched = []
    for f in hits:
        txt = f.read_text(encoding="utf-8")
        if 'VJEPA_BASE_URL = "http://localhost' not in txt:
            continue
        txt = re.sub(r'VJEPA_BASE_URL = "http://localhost:\d+"',
                     'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"',
                     txt)
        f.write_text(txt, encoding="utf-8")
        patched.append(str(f))
    return patched


def _unwrap_encoder(obj):
    """Return the encoder from whatever the loader handed back."""
    if isinstance(obj, (tuple, list)):
        if not obj:
            raise RuntimeError("model loader returned an empty sequence")
        obj = obj[0]                      # (encoder, predictor) -> encoder
    for attr in ("encoder", "backbone", "model"):
        inner = getattr(obj, attr, None)
        if inner is not None and hasattr(inner, "forward") and not hasattr(obj, "forward"):
            return inner
    if not hasattr(obj, "forward"):
        raise RuntimeError(f"could not find an encoder in {type(obj)}")
    return obj


def load_encoder(variant: str = "base", device: str = "cuda",
                 source: str = "auto"):
    """Load a frozen V-JEPA encoder.

    `source`:
      hub  - official torch.hub entry point (needs the URL patch above)
      hf   - Hugging Face transformers, which sidesteps the broken URL
      auto - try hub, fall back to hf
    """
    if source in ("hub", "auto"):
        try:
            name = VJEPA_VARIANTS.get(variant, variant)
            try:
                enc = torch.hub.load("facebookresearch/vjepa2", name, pretrained=True)
            except Exception:
                done = _patch_hub_base_url()
                if done:
                    print(f"[vjepa] patched hardcoded localhost URL in: {done}")
                enc = torch.hub.load("facebookresearch/vjepa2", name,
                                     pretrained=True, force_reload=False)
            # The V-JEPA 2.1 entry points return (encoder, predictor); the 2.0
            # ones return a bare module. The predictor is only needed for the
            # masked-prediction objective, so for feature extraction we keep
            # the encoder and drop it.
            enc = _unwrap_encoder(enc)
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False
            return enc.to(device)
        except Exception as e:
            if source == "hub":
                raise
            print(f"[vjepa] torch.hub failed ({e!r}); falling back to Hugging Face")

    from transformers import AutoModel

    repo = HF_VARIANTS.get(variant, variant)
    model = AutoModel.from_pretrained(repo)
    enc = _unwrap_encoder(getattr(model, "encoder", model))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device)


def prepare_clip(frames: np.ndarray, num_frames: int = DEFAULT_FRAMES,
                 size: int = DEFAULT_SIZE) -> torch.Tensor:
    """(T,H,W) uint8 greyscale -> (1,3,num_frames,size,size), normalised.

    Frames are sampled uniformly across the WHOLE clip rather than from a
    window. The diagnostic content is the transition from uncompressed to
    compressed, so a window covering only one phase carries no signal at all.
    """
    import torch.nn.functional as F

    t = len(frames)
    idx = (np.linspace(0, t - 1, num_frames).round().astype(int)
           if t >= num_frames else np.clip(np.arange(num_frames), 0, t - 1))
    sel = frames[idx].astype(np.float32) / 255.0

    x = torch.from_numpy(sel)[:, None]                     # (T,1,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)                               # greyscale -> RGB
    m = torch.tensor(MEAN)[None, :, None, None]
    s = torch.tensor(STD)[None, :, None, None]
    x = (x - m) / s
    return x.permute(1, 0, 2, 3)[None]                     # (1,3,T,H,W)


def prepare_mask_clip(video: np.ndarray, valid: np.ndarray,
                      num_frames: int = DEFAULT_FRAMES,
                      size: int = DEFAULT_SIZE) -> torch.Tensor:
    """(T,C,H,W) float mask video -> (1,3,num_frames,size,size), normalised.

    WHY MASKS RATHER THAN PIXELS
    ----------------------------
    The previous attempt on this cohort encoded raw DICOM frames and the
    features predicted HOSPITAL at AUC ~1.00, then collapsed on held-out
    centres. Centre prevalence alone predicts thrombosis at AUC 0.709 here, so
    a leaky input can beat the clinical baseline while carrying no clinical
    signal.

    A binary mask has no speckle, no gain curve, no scanner signature.
    `check_centre_leakage.py` measured the difference on v12 sidecars: a centre
    classifier scores 0.541 against a 0.474 majority floor - a lift of +0.068,
    against ~1.00 on pixels. Not zero, so leave-one-centre-out remains
    mandatory, but most of the leakage is gone by construction.

    CHANNEL MAPPING
    ---------------
    The encoder wants 3 channels because it was pretrained on RGB. A 1-channel
    mask video is replicated across all three, exactly as greyscale frames are
    in `prepare_clip`. A 2- or 3-channel mask video (vein, artery, clot) maps
    one mask per colour channel - which is not what the pretrained weights
    expect a colour channel to mean, but it is the only way to present several
    masks at once without collapsing them into one.

    INVALID FRAMES
    --------------
    `step17` marks frames with no polygon, and frames where the vein and
    artery masks collided, as invalid. Those are dropped and the remaining
    frames resampled, rather than fed as empty. An empty channel asserts "no
    vessel here", which is a different lie from a wrong mask but still a lie -
    and on collided frames the vein polygon is on the vein only 48.3% of the
    time against 78.0% on clean ones.
    """
    import torch.nn.functional as F

    keep = video[valid] if valid is not None and valid.any() else video
    t = len(keep)
    if t == 0:
        raise ValueError("no valid frames")
    idx = (np.linspace(0, t - 1, num_frames).round().astype(int)
           if t >= num_frames else np.clip(np.arange(num_frames), 0, t - 1))
    sel = keep[idx]                                        # (T,C,H,W)

    x = torch.from_numpy(np.ascontiguousarray(sel)).float()
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 2:                                  # vein, artery
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    elif x.shape[1] > 3:
        x = x[:, :3]

    x = F.interpolate(x, size=(size, size), mode="bilinear",
                      align_corners=False)
    m = torch.tensor(MEAN)[None, :, None, None]
    s = torch.tensor(STD)[None, :, None, None]
    x = (x - m) / s
    return x.permute(1, 0, 2, 3)[None]                     # (1,3,T,H,W)


@torch.inference_mode()
def encode_mask_clip(encoder, video, valid, device: str = "cuda",
                     num_frames: int = DEFAULT_FRAMES,
                     size: int = DEFAULT_SIZE, fp16: bool = True):
    """Mean-pooled embedding for one mask video."""
    x = prepare_mask_clip(video, valid, num_frames, size).to(device)
    out = _forward_any_layout(encoder, x, device, fp16)
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        out = out.mean(dim=1)
    elif out.ndim > 3:
        out = out.flatten(2).mean(dim=2)
    return out.float().squeeze(0).cpu().numpy()


def _forward_any_layout(encoder, x, device, fp16):
    """Call the encoder, tolerating either video tensor convention.

    The two sources disagree on layout and neither documents it at the call
    site:

      * torch.hub backbones take (B, C, T, H, W)
      * Hugging Face video models take (B, T, C, H, W) and permute internally

    Passing the hub layout to an HF model produces
    "expected input[1, 64, 3, 384, 384] to have 3 channels, but got 64"
    because the internal permute swaps an already-correct tensor. Rather than
    hard-code a guess, try one and fall back on a channel-count error. The
    working layout is cached on the encoder so the probe only pays for the
    discovery once.
    """
    order = getattr(encoder, "_dvt_layout", None)
    attempts = [order] if order else ["bcthw", "btchw"]

    last = None
    for layout in attempts:
        inp = x if layout == "bcthw" else x.permute(0, 2, 1, 3, 4).contiguous()
        try:
            with torch.autocast(device_type=device.split(":")[0], enabled=fp16):
                out = encoder(inp)
            encoder._dvt_layout = layout
            return out
        except RuntimeError as e:
            if "channels" not in str(e) and "size" not in str(e):
                raise
            last = e
    raise last


@torch.inference_mode()
def encode_clip(encoder, frames: np.ndarray, device: str = "cuda",
                num_frames: int = DEFAULT_FRAMES, size: int = DEFAULT_SIZE,
                fp16: bool = True) -> np.ndarray:
    """Mean-pooled token embedding for one clip."""
    x = prepare_clip(frames, num_frames, size).to(device)
    out = _forward_any_layout(encoder, x, device, fp16)

    # HF models return a ModelOutput rather than a bare tensor
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:                                      # (B, N_tokens, D)
        out = out.mean(dim=1)
    elif out.ndim > 3:                                     # (B, D, t, h, w)
        out = out.flatten(2).mean(dim=2)
    return out.float().squeeze(0).cpu().numpy()


def _match_renamed(stem: str, df):
    """Map anonymized_<SubjectID>_<n>_<SITE>-<LIMB> back to a label row."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    subj, tag = parts[1], parts[-1]
    if "-" not in tag:
        return None
    site, limb = tag.split("-", 1)
    m = df[(df.SubjectID == subj) & (df["Anatomical Site"] == site)
           & (df.Limb == limb)]
    return m.iloc[0] if len(m) == 1 else None


def build_feature_table(dicom_dir: Path, labels_csv: Path, variant: str = "base",
                        device: str = "cuda", max_frames: int = 64,
                        downsample: int = 2, out: Path | None = None,
                        limit: int | None = None, source: str = "auto"):
    """Encode every clip once and cache the features.

    One forward pass per clip. Caching matters because probing is then seconds,
    so splits and targets can be re-explored without re-encoding 2341 clips.
    """
    import pandas as pd

    from . import step0_labels as S0
    from . import step8_compression_phase as S8

    df = S0.load_labels(labels_csv)
    by_stem = {r["clip_id"]: r for _, r in df.iterrows()}

    encoder = load_encoder(variant, device, source=source)
    feats, meta, errors = [], [], 0
    files = sorted(Path(dicom_dir).glob("*.dcm"))
    if limit:
        files = files[:limit]

    for i, p in enumerate(files, 1):
        row = by_stem.get(p.stem)
        if row is None:
            row = _match_renamed(p.stem, df)
        if row is None:
            print(f"[{i}/{len(files)}] {p.stem[:30]} SKIP - no label row")
            continue
        try:
            fr = S8.load_clip_frames(p, max_frames=max_frames, downsample=downsample)
            v = encode_clip(encoder, fr, device=device)
        except Exception as e:
            errors += 1
            print(f"[{i}/{len(files)}] {p.stem[:30]} ERROR {e!r}")
            continue
        feats.append(v)
        meta.append({"clip": p.stem, "SubjectID": row["SubjectID"],
                     "centre": row["centre"], "site": row["Anatomical Site"],
                     "thrombosis": int(row["Thrombosis"]),
                     "abnormal": int(row["abnormal"])})
        if i % 25 == 0:
            print(f"  {i}/{len(files)} encoded", flush=True)

    if not feats:
        raise SystemExit("no clips encoded - check --dir and --csv")

    X = np.stack(feats)
    M = pd.DataFrame(meta)
    out = Path(out or (C.INDEX_DIR / f"vjepa_{variant}_features.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, **{k: M[k].to_numpy() for k in M.columns})
    print(f"\nencoded {len(X)} clips, dim {X.shape[1]}, {errors} errors -> {out}")
    return X, M


def build_mask_feature_table(sidecar_dir: Path, annotations: Path,
                             variant: str = "base", device: str = "cuda",
                             out: Path | None = None, limit: int | None = None,
                             source: str = "auto", channels: int = 1,
                             scale_by: str = "vein",
                             num_frames: int = DEFAULT_FRAMES):
    """Encode MASK videos rendered from sidecars, and cache the features.

    Labels come from the annotation FILENAMES - `CENTRE__GROUP__SUBJECT__CLIP`
    - not from train.csv. The csv's `File Name` column holds hashed names
    (adfbd93d...dcm) while sidecars are named from the anonymised stems
    (anonymized_FR1-0001_0_003_CFV-L), so every csv lookup misses and the whole
    table comes back empty.

    `channels` selects what the model sees:

        1  vein only        - the least leaky input measured, +0.068 lift over
                              the majority-class floor
        2  vein + artery    - +0.099. The vein/artery SIZE RATIO is
                              centre-informative and survives any similarity
                              transform, so no scaling removes it
        3  + clot           - the only texture-derived signal that survives
                              into a mask input, and clot has the highest
                              expert ceiling here (subject AUC 0.828)

    Start at 1. A result on the leak-free input is unambiguous; a result on
    three channels has to be argued for.
    """
    import pandas as pd

    from . import step17_mask_video as S17

    by_stem = {}
    for q in Path(annotations).glob("*.json"):
        parts = q.stem.split("__")
        if len(parts) == 4:
            by_stem[parts[3]] = {"centre": parts[0], "group": parts[1],
                                 "SubjectID": parts[2]}
    if not by_stem:
        raise SystemExit(f"no CENTRE__GROUP__SUBJECT__CLIP files in {annotations}")

    encoder = load_encoder(variant, device, source=source)
    files = sorted(Path(sidecar_dir).rglob("*_darwin.json"))
    if limit:
        files = files[:limit]

    feats, meta, skipped = [], [], {}
    for i, p in enumerate(files, 1):
        stem = p.name.replace("_darwin.json", "")
        row = by_stem.get(stem)
        if row is None:
            skipped["no label"] = skipped.get("no label", 0) + 1
            continue
        vid, valid, info = S17.render_clip(p, num_frames=max(num_frames, 32),
                                           channels=channels,
                                           scale_by=scale_by)
        if vid is None:
            r = info.get("reason", "render failed")
            skipped[r] = skipped.get(r, 0) + 1
            continue
        try:
            v = encode_mask_clip(encoder, vid, valid, device=device,
                                 num_frames=num_frames)
        except Exception as e:
            skipped[f"encode: {type(e).__name__}"] = \
                skipped.get(f"encode: {type(e).__name__}", 0) + 1
            continue
        feats.append(v)
        meta.append({"clip": stem, "SubjectID": row["SubjectID"],
                     "centre": row["centre"],
                     "site": S10_site(stem),
                     "thrombosis": 1 if row["group"] == "DVT" else 0,
                     "abnormal": 1 if row["group"] == "DVT" else 0,
                     "valid_frames": int(valid.sum())})
        if i % 25 == 0:
            print(f"  {i}/{len(files)} encoded", flush=True)

    if not feats:
        raise SystemExit(f"no clips encoded - skipped {skipped}")

    X = np.stack(feats)
    M = pd.DataFrame(meta)
    out = Path(out or (C.INDEX_DIR / f"vjepa_{variant}_mask{channels}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, **{k: M[k].to_numpy() for k in M.columns})
    print(f"\nencoded {len(X)} clips, dim {X.shape[1]} -> {out}")
    print(f"  channels={channels}  scale_by={scale_by}  skipped={skipped}")
    print(f"  DVT+ {int(M.thrombosis.sum())}  subjects {M.SubjectID.nunique()}  "
          f"centres {sorted(M.centre.unique())}")
    return X, M


def S10_site(stem: str) -> str:
    from . import step10_auto_seed as S10
    return S10.parse_filename_tag(stem)["site"] or ""


# ------------------------------------------------------------------ metrics
def log_loss(p, y, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    order = np.argsort(s)
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_once(Xtr, ytr, Xva, epochs, lr, weight_decay, device):
    """Single logistic fit, full batch, bias initialised at the base rate."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-6
    Xtr_t, Xva_t = (Xtr_t - mu) / sd, (Xva_t - mu) / sd

    probe = nn.Linear(Xtr.shape[1], 1).to(device)
    nn.init.zeros_(probe.weight)
    p0 = float(min(max(ytr.mean(), 1e-4), 1 - 1e-4))
    with torch.no_grad():
        probe.bias.fill_(float(np.log(p0 / (1 - p0))))

    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(probe(Xtr_t).squeeze(1), ytr_t).backward()
        opt.step()

    with torch.no_grad():
        return torch.sigmoid(probe(Xva_t).squeeze(1)).cpu().numpy()


def fit_probe(Xtr, ytr, Xva, epochs: int = 2000, lr: float = 0.05,
              weight_decay: float | None = None, device: str = "cpu",
              wd_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), inner_frac: float = 0.25,
              seed: int = 0):
    """Logistic probe on frozen features, initialised at the base rate.

    The output bias starts at the log-odds of the training prevalence, so an
    untrained probe already predicts the base rate. Under log loss that IS the
    baseline, and starting there stops early epochs scoring worse than doing
    nothing at all.

    Two optimisation traps, both caught by scripts/test_vjepa_probe.py:

    1. UNDERTRAINING. At lr=1e-3 for 300 epochs this recovered only AUC 0.630
       on a synthetic signal it should score ~0.95 on. An undertrained probe
       reports "no signal" when signal is present - the one failure mode that
       would abandon the approach for the wrong reason.

    2. OVERCONFIDENCE. Training harder then fixed that AUC but LOST on log
       loss: the probe separated the classes well while producing probabilities
       too extreme to be well calibrated. Since the Kaggle DVT task is scored
       on log loss, ranking well is not enough.

    So weight decay is selected on an inner split against LOG LOSS itself
    rather than fixed. Full-batch on a few thousand samples is cheap, so the
    grid costs almost nothing.
    """
    if weight_decay is not None:
        return _fit_once(Xtr, ytr, Xva, epochs, lr, weight_decay, device)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    n_in = max(1, int(len(idx) * inner_frac))
    va_i, tr_i = idx[:n_in], idx[n_in:]

    # inner split must contain both classes, or log loss cannot rank the grid
    if ytr[tr_i].sum() == 0 or ytr[va_i].sum() == 0 or len(tr_i) < 10:
        return _fit_once(Xtr, ytr, Xva, epochs, lr, 1e-2, device)

    best_wd, best_ll = None, float("inf")
    for wd in wd_grid:
        p = _fit_once(Xtr[tr_i], ytr[tr_i], Xtr[va_i], epochs, lr, wd, device)
        ll = log_loss(p, ytr[va_i])
        if ll < best_ll:
            best_wd, best_ll = wd, ll

    return _fit_once(Xtr, ytr, Xva, epochs, lr, best_wd, device)



def normalize_per_centre(X, centre):
    """Standardise features within each centre.

    The frozen features predict which hospital a clip came from at AUC ~1.000,
    and centre prevalence ranges 2.3%-13.5%, so a probe can score well by
    recognising the scanner while learning nothing about veins. Removing each
    centre's own mean and scale deletes the bulk of that signature. Crude
    compared with adversarial training, but it costs nothing and needs no
    labels beyond the centre tag.
    """
    Xn = np.array(X, dtype=np.float32, copy=True)
    for c in np.unique(centre):
        m = centre == c
        if m.sum() < 2:
            continue
        Xn[m] = (Xn[m] - Xn[m].mean(0, keepdims=True)) / (Xn[m].std(0, keepdims=True) + 1e-6)
    return Xn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def fit_probe_adversarial(Xtr, ytr, ctr, Xva, epochs: int = 2000, lr: float = 0.05,
                          weight_decay: float = 1e-2, lambd: float = 0.3,
                          device: str = "cpu"):
    """Probe trained to predict thrombosis while being UNABLE to predict centre.

    A gradient-reversal head on the centre label pushes the shared
    representation to discard scanner identity, so the probe cannot fall back
    on the centre-prevalence shortcut. This matters here because leave-one-
    centre-out collapses precisely when that shortcut disappears.

    `lambd` trades the two objectives off. Too high and the representation is
    destroyed along with the shortcut.
    """
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-6
    Xtr_t, Xva_t = (Xtr_t - mu) / sd, (Xva_t - mu) / sd

    codes = {c: i for i, c in enumerate(np.unique(ctr))}
    ctr_t = torch.tensor([codes[c] for c in ctr], dtype=torch.long, device=device)

    dim = Xtr.shape[1]
    trunk = nn.Sequential(nn.Linear(dim, 128), nn.ReLU()).to(device)
    head = nn.Linear(128, 1).to(device)
    adv = nn.Linear(128, len(codes)).to(device)

    p0 = float(min(max(ytr.mean(), 1e-4), 1 - 1e-4))
    with torch.no_grad():
        head.bias.fill_(float(np.log(p0 / (1 - p0))))

    opt = torch.optim.AdamW(
        list(trunk.parameters()) + list(head.parameters()) + list(adv.parameters()),
        lr=lr, weight_decay=weight_decay)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()
        z = trunk(Xtr_t)
        loss = bce(head(z).squeeze(1), ytr_t)
        loss = loss + ce(adv(_GradReverse.apply(z, lambd)), ctr_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        return torch.sigmoid(head(trunk(Xva_t)).squeeze(1)).cpu().numpy()


def _split_subjects(split_csv, which):
    """Subjects on one side of the frozen split, or None for 'all'.

    Filtering by SUBJECT rather than clip: the split is subject-grouped, so a
    clip-level filter would be equivalent here but would break silently if the
    csv were ever redrawn at clip level.
    """
    if not split_csv or which == "all":
        return None
    import csv as _csv
    keep = set()
    with open(split_csv) as fh:
        for row in _csv.DictReader(fh):
            if row.get("split") == which:
                keep.add(row["subject"])
    if not keep:
        raise SystemExit(f"no split={which} rows in {split_csv}")
    return keep


def evaluate(npz_path: Path, target: str = "thrombosis", device: str = "cpu",
             seed: int = 0, per_centre_norm: bool = False,
             adversarial: bool = False, lambd: float = 0.3,
             split_csv=None, split: str = "all") -> dict:
    """Subject-level split AND leave-one-centre-out, each vs a constant.

    `split_csv`/`split` restrict everything - probe fitting AND scoring - to
    one side of the frozen split. Develop against `train`; touch `test` once,
    at the end.

    Without this the probe fits across all subjects, so its internal
    cross-validation would train on subjects held out for the final test and
    every number afterwards would be optimistic in a way no later analysis
    could undo.
    """
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    y = d[target].astype(int)
    subj = d["SubjectID"].astype(str)
    centre = d["centre"].astype(str)

    keep = _split_subjects(split_csv, split)
    if keep is not None:
        m = np.array([s in keep for s in subj])
        if m.sum() < 20:
            raise SystemExit(f"only {int(m.sum())} clips on the {split} side")
        print(f"restricted to the {split.upper()} side: {int(m.sum())} clips, "
              f"{len(set(subj[m]))} subjects")
        X, y, subj, centre = X[m], y[m], subj[m], centre[m]
    if per_centre_norm:
        X = normalize_per_centre(X, centre)

    def _fit(Xa, ya, Xb, ca=None):
        if adversarial and ca is not None and len(np.unique(ca)) > 1:
            return fit_probe_adversarial(Xa, ya, ca, Xb, lambd=lambd, device=device)
        return fit_probe(Xa, ya, Xb, device=device)

    res = {"method": ("adversarial" if adversarial else "linear")
                     + ("+centre_norm" if per_centre_norm else ""),
           "n": int(len(y)), "n_positive": int(y.sum()),
           "prevalence": round(float(y.mean()), 4),
           "feature_dim": int(X.shape[1])}
    res["baseline_logloss_constant"] = round(
        log_loss(np.full(len(y), float(y.mean())), y), 4)

    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    rng.shuffle(subs)
    val_s = set(subs[: max(1, len(subs) // 5)])
    va = np.array([s in val_s for s in subj])
    if va.sum() and (~va).sum() and y[~va].sum() and y[va].sum():
        p = _fit(X[~va], y[~va], X[va], centre[~va])
        res["subject_split"] = {
            "n_train": int((~va).sum()), "n_val": int(va.sum()),
            "n_val_positive": int(y[va].sum()),
            "logloss": round(log_loss(p, y[va]), 4),
            "baseline": round(log_loss(np.full(int(va.sum()), float(y[~va].mean())), y[va]), 4),
            "auc": round(auc(p, y[va]), 4),
        }

    loco = {}
    for c in np.unique(centre):
        te = centre == c
        if not (te.sum() and (~te).sum() and y[~te].sum() and y[te].sum()):
            continue
        p = _fit(X[~te], y[~te], X[te], centre[~te])
        loco[c] = {"n": int(te.sum()), "n_pos": int(y[te].sum()),
                   "logloss": round(log_loss(p, y[te]), 4),
                   "baseline": round(log_loss(np.full(int(te.sum()), float(y[~te].mean())), y[te]), 4),
                   "auc": round(auc(p, y[te]), 4)}
    res["leave_one_centre_out"] = loco
    if loco:
        res["loco_mean_auc"] = round(float(np.mean([v["auc"] for v in loco.values()])), 4)
        res["loco_mean_logloss"] = round(float(np.mean([v["logloss"] for v in loco.values()])), 4)
        res["loco_mean_baseline"] = round(float(np.mean([v["baseline"] for v in loco.values()])), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="encode clips into a cached feature table")
    e.add_argument("--dir", required=False)
    e.add_argument("--csv", required=False)
    e.add_argument("--masks", default=None,
                   help="folder of *_darwin.json sidecars. Encodes MASK videos "
                        "instead of DICOM pixels - the input that removes most "
                        "of the centre leakage that sank the previous attempt.")
    e.add_argument("--annotations", default=None,
                   help="annotations_flat, for labels. Required with --masks: "
                        "train.csv keys on HASHED filenames and cannot be "
                        "joined to sidecar stems.")
    e.add_argument("--channels", type=int, default=1, choices=[1, 2, 3],
                   help="1 vein only (least leaky), 2 +artery, 3 +clot")
    e.add_argument("--scale-by", default="vein", choices=["vein", "pair"],
                   help="canvas normalisation; vein measured cleanest")
    e.add_argument("--variant", default="base")
    e.add_argument("--source", default="auto", choices=["auto", "hub", "hf"],
                   help="hub = torch.hub (upstream has a hardcoded localhost "
                        "URL we patch); hf = Hugging Face, avoids it entirely")
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.add_argument("--max-frames", type=int, default=64)
    e.add_argument("--downsample", type=int, default=2)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--out", default=None)

    p = sub.add_parser("probe", help="fit and evaluate the linear probe")
    p.add_argument("--features", required=True)
    p.add_argument("--target", default="thrombosis", choices=["thrombosis", "abnormal"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--per-centre-norm", action="store_true",
                   help="standardise features within each centre first")
    p.add_argument("--adversarial", action="store_true",
                   help="gradient-reversal on centre, so the probe cannot use "
                        "the scanner shortcut")
    p.add_argument("--lambd", type=float, default=0.3,
                   help="adversarial strength")
    p.add_argument("--split-csv", default=None,
                   help="frozen split from scripts/make_split.py")
    p.add_argument("--split", default="all", choices=["all", "train", "test"],
                   help="restrict probe fitting AND scoring to one side. "
                        "Develop against train; score test once, at the end.")

    a = ap.parse_args()
    if a.cmd == "encode":
        if a.masks:
            if not a.annotations:
                raise SystemExit("--masks requires --annotations "
                                 "(train.csv keys on hashed filenames and "
                                 "cannot be joined to sidecar stems)")
            build_mask_feature_table(Path(a.masks), Path(a.annotations),
                                     a.variant, a.device,
                                     Path(a.out) if a.out else None, a.limit,
                                     source=a.source, channels=a.channels,
                                     scale_by=a.scale_by)
            return
        if not (a.dir and a.csv):
            raise SystemExit("encode needs either --masks/--annotations "
                             "or --dir/--csv")
        build_feature_table(Path(a.dir), Path(a.csv), a.variant, a.device,
                            a.max_frames, a.downsample,
                            Path(a.out) if a.out else None, a.limit,
                            source=a.source)
        return

    r = evaluate(Path(a.features), a.target, a.device,
                 split_csv=a.split_csv, split=a.split,
                 per_centre_norm=a.per_centre_norm, adversarial=a.adversarial,
                 lambd=a.lambd)
    print(json.dumps(r, indent=2))
    print("\n--- verdict ---")
    base, got = r.get("loco_mean_baseline"), r.get("loco_mean_logloss")
    if base is not None and got is not None:
        print(f"leave-one-centre-out log loss {got}  vs constant baseline {base}")
        print("PROBE BEATS THE BASELINE."
              if got < base else
              "Probe does NOT beat a constant - V-JEPA features do not transfer "
              "to this task as-is. Next option is continued self-supervised "
              "pretraining on unlabelled ultrasound (needs no labels).")


if __name__ == "__main__":
    main()
