"""Encoder initialisation by feature distillation from the public CineMA foundation model (stage S1).

The frozen CineMA SAX encoder (``mathpluscode/CineMA``, files ``pretrained/config.yaml`` and
``pretrained/cinema.safetensors``) sees each training image the way it was pre-trained: resampled to 1 x 1 x 10 mm,
centre-cropped / end-padded to 192 x 192 x 16 and min-max scaled.  Its 1/8 convolutional skip and its 1/16 ViT tokens are
mapped back onto the native student lattice through the exact crop/resample affine, and the student's stage-3 and
stage-4 features (after SliceAlign / PhysZSSM) are trained to match them with a masked cosine loss through 1x1x1
projectors that are discarded afterwards.  Only training images are used (labelled and unlabelled, no masks).
The output is an encoder-only artifact consumed by ``scripts/train.py --init``.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import SplitData
from .slicessm_student import ENCODER_PREFIXES, build_model

HF_REPO = "mathpluscode/CineMA"
HF_REVISION = "b1251ee50423bceeca84c080782fc3bc7756dea6"
HF_FILES = ("pretrained/config.yaml", "pretrained/cinema.safetensors")
SAX_SPACING_ZYX = (10.0, 1.0, 1.0)
SAX_SIZE_DHW = (16, 192, 192)
INIT_SCHEMA = "sasnet.cinema_init.v1"
DISTILL_DEFAULTS = {"updates": 10000, "updates_per_epoch": 80, "lr": 3e-4, "weight_decay": 1e-4, "holdout_fraction": 0.1,
                    "hf_repo": HF_REPO, "hf_revision": HF_REVISION}


# ---------------------------------------------------------------------------------------------------------------
# CineMA source
# ---------------------------------------------------------------------------------------------------------------

class CinemaSaxEncoder(nn.Module):
    """Official CineMA ConvMAE SAX branch exposing the 1/8 conv skip ``[B,C,D,H,W]`` and the 1/16 tokens ``[B,N,C]``."""

    def __init__(self, model: nn.Module, conv_1_8_index: int) -> None:
        super().__init__()
        self.model, self.conv_1_8_index = model, conv_1_8_index

    def forward(self, image_hwd: Tensor) -> tuple[Tensor, Tensor]:
        skips, patch_tokens = self.model.enc_down_dict["sax"](image_hwd, mask=None)
        n = patch_tokens.shape[1]
        tokens = self.model.encoder(patch_tokens)[:, 1:1 + n]
        tokens = self.model.enc_fusion_dict["sax"](skips, tokens, mask=None)
        return skips[self.conv_1_8_index].permute(0, 1, 4, 2, 3), tokens  # CineMA works in H,W,D; return D,H,W


def load_cinema(repo: str = HF_REPO, revision: str | None = HF_REVISION, device: torch.device | str = "cpu") -> CinemaSaxEncoder:
    import yaml
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from cinema.mae.mae import get_model  # official CineMA package (github.com/mathpluscode/CineMA)

    kwargs = {} if revision is None else {"revision": revision}
    config_path, weights_path = (Path(hf_hub_download(repo_id=repo, filename=f, **kwargs)) for f in HF_FILES)
    raw = yaml.safe_load(config_path.read_text())
    model_cfg, sax = raw["model"], raw["data"]["sax"]
    if tuple(float(v) for v in sax["spacing"]) != (1.0, 1.0, 10.0) or tuple(int(v) for v in sax["patch_size"]) != (192, 192, 16):
        raise ValueError("unexpected CineMA SAX preprocessing in config.yaml")
    # ConvMAE: patch (4,4,1) then one scale factor (2,2,1) per conv block -> skips at strides 4, 8, ... in-plane
    patch, scale = tuple(model_cfg["patch_size"]), tuple(model_cfg["scale_factor"])
    strides, running = [], [1, 1, 1]
    for factor in [patch, *([scale] * len(model_cfg["enc_conv_chans"]))][:-1]:
        running = [a * b for a, b in zip(running, factor)]
        strides.append(tuple(running))
    conv_index = next(i for i, s in enumerate(strides) if s[:2] == (8, 8))
    model = get_model(OmegaConf.create(raw))
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        state = {k: handle.get_tensor(k) for k in handle.keys()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"CineMA weights do not match the official model: missing={missing[:3]}, unexpected={unexpected[:3]}")
    encoder = CinemaSaxEncoder(model, conv_index).to(device).eval()
    encoder.requires_grad_(False)
    return encoder


# ---------------------------------------------------------------------------------------------------------------
# CineMA view of a native volume and the way back
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class SaxView:
    image_hwd: Tensor                 # [1, 1, 192, 192, 16] CineMA input
    native_to_processed: np.ndarray   # 4x4, native voxel (z, y, x, 1) -> processed voxel
    support_bounds: tuple[tuple[int, int], ...]  # processed-voxel extent of real (non-padded) data per axis (z, y, x)


def sax_view(image: Tensor, spacing_zyx: Sequence[float]) -> SaxView:
    """Resample ``[D, H, W]`` to 10 x 1 x 1 mm, centre-crop / end-pad to 16 x 192 x 192 and min-max scale (CineMA SAX)."""
    native = tuple(int(v) for v in image.shape)
    out_shape = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(native, spacing_zyx, SAX_SPACING_ZYX))
    if out_shape == native:
        resampled, scale = image, (1.0, 1.0, 1.0)
    else:
        resampled = F.interpolate(image[None, None], size=out_shape, mode="trilinear", align_corners=False)[0, 0]
        scale = tuple(o / n for o, n in zip(out_shape, native))
    start = [max(0, (n - t) // 2) for n, t in zip(out_shape, SAX_SIZE_DHW)]
    canvas = torch.zeros(SAX_SIZE_DHW, dtype=torch.float32)
    src = [slice(s, min(n, s + t)) for s, n, t in zip(start, out_shape, SAX_SIZE_DHW)]
    extent = [sl.stop - sl.start for sl in src]
    canvas[: extent[0], : extent[1], : extent[2]] = resampled[src[0], src[1], src[2]]
    lo, hi = canvas.amin(), canvas.amax()
    canvas = torch.zeros_like(canvas) if float(hi - lo) <= 1e-12 else ((canvas - lo) / (hi - lo)).clamp(0.0, 1.0)
    # processed voxel p <- native coordinate (p + start + 0.5) / scale - 0.5 (F.interpolate, align_corners=False)
    processed_to_native = np.diag([1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2], 1.0])
    processed_to_native[:3, 3] = [(s + 0.5) / sc - 0.5 for s, sc in zip(start, scale)]
    return SaxView(canvas.permute(1, 2, 0)[None, None], np.linalg.inv(processed_to_native), tuple((0, e) for e in extent))


def backproject(feature: Tensor, view: SaxView, target_shape: Sequence[int], *, source_stride: Sequence[int],
                target_step: Sequence[int]) -> tuple[Tensor, Tensor]:
    """Sample a CineMA feature ``[B, C, D', H', W']`` at the centres of a native student lattice; also return its support."""
    indices = [torch.arange(t, dtype=torch.float32, device=feature.device) * s for t, s in zip(target_shape, target_step)]
    zz, yy, xx = torch.meshgrid(*indices, indexing="ij")
    native = torch.stack([zz, yy, xx, torch.ones_like(zz)], dim=-1)
    processed = torch.einsum("...j,ij->...i", native, torch.as_tensor(view.native_to_processed, dtype=torch.float32, device=feature.device).T)[..., :3]
    # a kernel = stride patch-embedding output j is centred at processed voxel j * stride + (stride - 1) / 2
    source = torch.stack([processed[..., a] / source_stride[a] - 0.5 + 0.5 / source_stride[a] for a in range(3)], dim=-1)
    d, h, w = feature.shape[-3:]
    grid = torch.stack([2 * (source[..., 2] + 0.5) / w - 1, 2 * (source[..., 1] + 0.5) / h - 1, 2 * (source[..., 0] + 0.5) / d - 1], dim=-1)
    sampled = F.grid_sample(feature.float(), grid[None].expand(feature.shape[0], -1, -1, -1, -1), mode="bilinear",
                            padding_mode="border", align_corners=False)
    b = view.support_bounds
    valid = torch.ones_like(processed[..., 0], dtype=torch.bool)
    for a in range(3):
        valid &= (processed[..., a] >= b[a][0] - 0.5) & (processed[..., a] < b[a][1] - 0.5)
    return sampled, valid.to(sampled.dtype)[None, None]


def tokens_to_grid(tokens: Tensor) -> Tensor:
    """CineMA tokens ``[B, N, C]`` (flattened H, W, D at stride 16, 16, 1) -> ``[B, C, 16, 12, 12]`` (D, H, W)."""
    grid_hwd = (SAX_SIZE_DHW[1] // 16, SAX_SIZE_DHW[2] // 16, SAX_SIZE_DHW[0])
    return tokens.reshape(tokens.shape[0], *grid_hwd, tokens.shape[2]).permute(0, 4, 3, 1, 2)


def masked_cosine(projected: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    cosine = 1.0 - F.cosine_similarity(projected.float(), target.float(), dim=1, eps=1e-6)
    weight = valid[:, 0].float()
    return (cosine * weight).sum() / weight.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------------------------------------------

def distill(cfg: Mapping[str, Any], out_dir: str | Path, *, device: str = "cuda", max_updates: int | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    dcfg = {**DISTILL_DEFAULTS, **cfg.get("distill", {})}
    data = SplitData(cfg["data"]["prepared"], cfg["data"]["split"])
    # training images only: labelled frames then unlabelled frames, fixed order; the last 10 % are a holdout for monitoring
    keys = [(k, True) for k in data.labeled_keys()] + [(k, False) for k in data.unlabeled_keys()]
    n_hold = max(1, min(len(keys) - 1, int(math.ceil(len(keys) * float(dcfg["holdout_fraction"])))))
    train_keys, hold_keys = keys[: len(keys) - n_hold], keys[len(keys) - n_hold:]
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cinema = load_cinema(dcfg["hf_repo"], dcfg["hf_revision"], dev)
    model = build_model(cfg.get("model")).to(dev).train()
    encoder_params = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES)]
    optimizer = torch.optim.AdamW(encoder_params, lr=float(dcfg["lr"]), weight_decay=float(dcfg["weight_decay"]))
    projectors = nn.ModuleDict()
    phases = {"conv": model.phases[3][0], "tokens": model.phases[4][0]}

    def loss_of(key: tuple[tuple[str, int], bool]) -> Tensor:
        (pid, frame), labeled = key
        f = data.dataset.load(pid, frame, with_mask=False)
        view = sax_view(f.image[0, 0], f.geometry["spacing_zyx"])
        with torch.no_grad():
            conv, tokens = cinema(view.image_hwd.to(dev))
        s3, s4 = model.encode(f.image.to(dev), f.geometry)
        terms = []
        for name, source, student, stride in (("conv", conv, s3, (1, 8, 8)), ("tokens", tokens_to_grid(tokens), s4, (1, 16, 16))):
            target, valid = backproject(source, view, student.shape[-3:], source_stride=stride, target_step=phases[name])
            if name not in projectors:
                projectors[name] = nn.Conv3d(student.shape[1], target.shape[1], 1).to(dev)
                optimizer.add_param_group({"params": projectors[name].parameters()})
            terms.append(masked_cosine(projectors[name](student), target.detach(), valid))
        return torch.stack(terms).mean()

    updates = int(dcfg["updates"]) if max_updates is None else min(int(dcfg["updates"]), int(max_updates))
    per_epoch = int(dcfg["updates_per_epoch"])
    for update in range(updates):
        loss = loss_of(train_keys[update % len(train_keys)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (update + 1) % per_epoch == 0 or update + 1 == updates:
            model.eval()
            with torch.no_grad():
                holdout = 1.0 - float(np.mean([float(loss_of(k)) for k in hold_keys]))
            model.train()
            row = {"epoch": math.ceil((update + 1) / per_epoch), "update": update + 1, "loss": float(loss), "holdout_cosine": holdout}
            with (out / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"[distill] ep {row['epoch']}/{math.ceil(updates / per_epoch)} | loss {row['loss']:.4f} | holdout cosine {holdout:.4f}", flush=True)
    encoder = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(ENCODER_PREFIXES)}
    torch.save({"schema": INIT_SCHEMA, "encoder": encoder, "model": dict(cfg.get("model") or {}), "updates": updates,
                "source": {"hf_repo": dcfg["hf_repo"], "hf_revision": dcfg["hf_revision"]}}, out / "encoder_init.pt")
    return {"path": str(out / "encoder_init.pt"), "updates": updates, "train_frames": len(train_keys), "holdout_frames": len(hold_keys)}


__all__ = ["distill", "load_cinema", "sax_view", "backproject", "masked_cosine", "HF_REPO", "HF_REVISION"]
