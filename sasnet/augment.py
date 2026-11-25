"""Reproducible in-plane augmentation, weak/strong views and full-depth CutMix.

Everything is in-plane: no operation mixes or resamples slices.  The weak view is spatial only (rotation, scaling,
elastic deformation, flips, 90-degree turns); the strong view adds intensity perturbations (gamma, brightness, contrast,
noise, blur, low resolution) to the weak view.  All randomness comes from private CPU generators whose state is saved in
checkpoints, so a resumed run replays exactly the same views.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.nn import functional as F

CutMixBox = tuple[int, int, int, int]  # (y0, y1, x0, x1)

PROBABILITIES = {
    "rotate": .5, "scale": .5, "elastic": .2, "flip_y": .5, "flip_x": .5, "rot90": .5,
    "gamma": .3, "brightness": .15, "contrast": .15, "noise": .1, "blur": .2, "low_resolution": .25,
}


class InplaneAugment:
    """Frame-wise augmentation of ``[1, 1, D, H, W]`` volumes with one private CPU generator."""

    def __init__(self, seed: int = 42) -> None:
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + 2003)
        self.probabilities = dict(PROBABILITIES)

    def _rand(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._rand()

    def _enabled(self, name: str) -> bool:
        return self._rand() < self.probabilities[name]

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.generator.set_state(state["generator_state"].cpu())

    @staticmethod
    def _warp(image: Tensor, target: Tensor | None, matrix: Tensor, displacement: Tensor | None = None):
        # ``matrix`` maps output pixel coordinates to input coordinates around the canvas centre (align_corners=False).
        d, h, w = image.shape[2:]
        device = image.device
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device), torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
        cy, cx = (h - 1) / 2, (w - 1) / 2
        coords = torch.stack((xx - cx, yy - cy), dim=-1) @ matrix.to(device=device, dtype=torch.float32).T
        x, y = coords[..., 0] + cx, coords[..., 1] + cy
        if displacement is not None:
            displacement = displacement.to(device=device, dtype=torch.float32)
            y, x = y + displacement[0], x + displacement[1]
        grid = torch.stack(((2 * x + 1) / w - 1, (2 * y + 1) / h - 1), dim=-1).unsqueeze(0).expand(d, -1, -1, -1)
        out_image = F.grid_sample(image[0, 0].unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0][None, None]
        out_target = None
        if target is not None:
            masks = target[0].to(torch.float32).unsqueeze(1)
            out_target = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].round().long().unsqueeze(0)
        return out_image, out_target

    def spatial(self, image: Tensor, target: Tensor | None, geometry: Mapping[str, Any]):
        """Spatial draw applied identically to the image and an optional ``[1, D, H, W]`` label/validity map (fp32)."""
        with torch.autocast(device_type=image.device.type, enabled=False):
            out, lab = image.clone(), (target.clone() if target is not None else None)
            h, w = image.shape[-2:]
            if self._enabled("rotate"):
                angle = np.deg2rad(self._uniform(-180, 180))
                c, s = float(np.cos(angle)), float(np.sin(angle))
                out, lab = self._warp(out, lab, torch.tensor([[c, -s], [s, c]], dtype=torch.float32))
            if self._enabled("scale"):
                out, lab = self._warp(out, lab, torch.eye(2) / self._uniform(.7, 1.4))
            if self._enabled("elastic"):
                alpha, sigma = self._uniform(0, 200), self._uniform(9, 13)
                raw = torch.rand((2, h, w), generator=self.generator).numpy() * 2 - 1
                field = np.stack([gaussian_filter(raw[i], sigma=sigma, mode="constant") for i in range(2)])
                out, lab = self._warp(out, lab, torch.eye(2), torch.from_numpy(field).to(torch.float32) * alpha)
            if self._enabled("flip_y"):
                out = torch.flip(out, dims=(-2,))
                lab = torch.flip(lab, dims=(-2,)) if lab is not None else None
            if self._enabled("flip_x"):
                out = torch.flip(out, dims=(-1,))
                lab = torch.flip(lab, dims=(-1,)) if lab is not None else None
            spacing = geometry.get("spacing_zyx")
            square_pixels = spacing is not None and abs(float(spacing[1]) - float(spacing[2])) <= 1e-3
            if square_pixels and self._enabled("rot90"):
                angle = int(torch.randint(1, 4, (), generator=self.generator)) * np.pi / 2
                c, s = float(np.cos(angle)), float(np.sin(angle))
                out, lab = self._warp(out, lab, torch.tensor([[c, -s], [s, c]], dtype=torch.float32))
            return out, lab

    def intensity(self, image: Tensor) -> Tensor:
        with torch.autocast(device_type=image.device.type, enabled=False):
            out = image.clone()
            if self._enabled("gamma"):
                out = out.clamp_min(0).pow(self._uniform(.7, 1.5))
            if self._enabled("brightness"):
                out = out * self._uniform(.75, 1.25)
            if self._enabled("contrast"):
                mean = out.mean(dim=(-2, -1), keepdim=True)
                out = (out - mean) * self._uniform(.75, 1.25) + mean
            if self._enabled("noise"):
                std = self._uniform(0, .1)
                out = out + torch.randn(out.shape, generator=self.generator, dtype=torch.float32).to(out.device) * std
            if self._enabled("blur"):
                sigma = self._uniform(.5, 1)
                radius = max(1, int(np.ceil(3 * sigma)))
                coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
                kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
                kernel /= kernel.sum()
                k2 = torch.outer(kernel, kernel).to(out.device)
                flat = out.reshape(-1, 1, *out.shape[-2:])
                out = F.conv2d(F.pad(flat, (radius,) * 4, mode="replicate"), k2[None, None]).reshape_as(out)
            if self._enabled("low_resolution"):
                scale = self._uniform(.5, 1)
                h, w = out.shape[-2:]
                low = F.interpolate(out.reshape(-1, 1, h, w), size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="nearest")
                out = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).reshape_as(out)
            return out.clamp(0, 1)


class ViewGenerator:
    """Weak/strong views of one frame plus a CutMix generator; checkpointable."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.augment = InplaneAugment(self.seed)
        self.cutmix_generator = torch.Generator(device="cpu")
        self.cutmix_generator.manual_seed(self.seed + 3001)

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "augmentation": self.augment.state_dict(), "cutmix_generator_state": self.cutmix_generator.get_state().clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("view-generator seed mismatch")
        self.augment.load_state_dict(state["augmentation"])
        self.cutmix_generator.set_state(state["cutmix_generator_state"].cpu())

    def weak(self, image: Tensor, valid: Tensor, geometry: Mapping[str, Any], target: Tensor | None = None):
        """One spatial draw for ``image`` [1,1,D,H,W], its validity map and optional target (replayed with the same draw)."""
        replay = self.augment.state_dict()
        image_w, valid_w = self.augment.spatial(image, valid.long(), geometry)
        # a bilinear sum of [0, 1] inputs can overshoot 1 by one float32 ULP
        image_w = image_w.clamp_max(1.0)
        if target is None:
            return image_w, valid_w.bool()
        self.augment.load_state_dict(replay)
        _, target_w = self.augment.spatial(image, target, geometry)
        return image_w, valid_w.bool(), target_w

    def strong(self, weak_image: Tensor) -> Tensor:
        return self.augment.intensity(weak_image).clamp(0, 1)

    def views(self, image: Tensor, geometry: Mapping[str, Any], target: Tensor | None = None) -> dict[str, Any]:
        valid = torch.ones((image.shape[0], *image.shape[-3:]), dtype=torch.bool, device=image.device)
        out = self.weak(image, valid, geometry, target)
        view = {"weak": out[0], "valid": out[1], "geometry": geometry}
        if target is not None:
            view["target"] = out[2]
        view["strong"] = self.strong(view["weak"])
        return view


def _support(valid: Tensor) -> Tensor:
    """In-plane support common to every acquired slice of a ``[B, D, H, W]`` validity map."""
    slices = valid.bool().flatten(0, 1)
    acquired = slices.flatten(1).any(1)
    return slices[acquired].all(0) if bool(acquired.any()) else torch.zeros_like(slices[0])


def cutmix_box(valid: Tensor, generator: torch.Generator, *, partner_valid: Tensor | None = None,
               area_fraction: Sequence[float] = (0.25, 0.5), align_px: int = 16) -> CutMixBox:
    """Sample an in-plane box (full depth) inside the valid support of both volumes; corners are multiples of ``align_px``."""
    support = _support(valid)
    height, width = int(valid.shape[-2]), int(valid.shape[-1])
    if partner_valid is not None:
        support = support & _support(partner_valid)
    if not bool(support.any()):
        raise ValueError("CutMix valid intersection is empty")
    low, high = (float(v) for v in area_fraction)
    y_ends, x_ends = range(0, height + 1, align_px), range(0, width + 1, align_px)
    candidates = [(y0, y1, x0, x1) for y0, y1 in combinations(y_ends, 2) for x0, x1 in combinations(x_ends, 2)]
    if candidates:
        boxes = torch.tensor(candidates, device=support.device)
        y0, y1, x0, x1 = boxes.unbind(1)
        integral = F.pad(support.to(torch.int64).cumsum(0).cumsum(1), (1, 0, 1, 0))
        totals = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        candidates = [tuple(b) for b in boxes[totals == (y1 - y0) * (x1 - x0)].cpu().tolist()]
    target = low + (high - low) * float(torch.rand((), generator=generator))
    area = lambda b: ((b[1] - b[0]) * (b[3] - b[2])) / (height * width)  # noqa: E731
    in_range = [b for b in candidates if low <= area(b) <= high]
    if in_range:
        return in_range[int(torch.randint(len(in_range), (), generator=generator))]
    if candidates:  # no box in the area range fits: take the box closest to the drawn area
        best = min(abs(area(b) - target) for b in candidates)
        tied = sorted((b for b in candidates if abs(area(b) - target) == best), key=lambda b: abs(area(b) - target))
        return tied[int(torch.randint(len(tied), (), generator=generator))]
    raise ValueError("CutMix valid intersection contains no aligned box")


def mix_tensor(anchor: Tensor, partner: Tensor, box: CutMixBox) -> Tensor:
    """Copy the full-depth in-plane ``box`` of ``partner`` into ``anchor`` (``[B,D,H,W]`` or ``[B,C,D,H,W]``)."""
    y0, y1, x0, x1 = (int(v) for v in box)
    if partner.shape[0] != anchor.shape[0] and partner.shape[0] == 1:
        partner = partner.expand(anchor.shape[0], *partner.shape[1:])
    result = anchor.clone()
    result[..., y0:y1, x0:x1] = partner[..., y0:y1, x0:x1]
    return result


def center_crop_pad(value: Tensor, target_shape: Sequence[int], *, pad_value: float = 0) -> Tensor:
    """Centre-crop or pad the last three axes of ``[B,D,H,W]`` / ``[B,C,D,H,W]`` to ``target_shape``."""
    target = tuple(int(v) for v in target_shape)
    if tuple(value.shape[-3:]) == target:
        return value
    result = torch.full((*value.shape[:-3], *target), pad_value, dtype=value.dtype, device=value.device)
    src, dst = [], []
    for size, want in zip(value.shape[-3:], target):
        take = min(int(size), want)
        src.append(slice((int(size) - take) // 2, (int(size) - take) // 2 + take))
        dst.append(slice((want - take) // 2, (want - take) // 2 + take))
    result[..., dst[0], dst[1], dst[2]] = value[..., src[0], src[1], src[2]]
    return result


__all__ = ["InplaneAugment", "ViewGenerator", "cutmix_box", "mix_tensor", "center_crop_pad", "CutMixBox"]
