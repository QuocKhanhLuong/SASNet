"""Segmentation losses: supervised CE + Dice, and the confidence-weighted CE + Dice used on pseudo-labels."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy + soft Dice over the foreground classes present in ``labels`` (``[B, C, ...]`` vs ``[B, ...]``)."""
    classes = int(logits.shape[1])
    valid = labels != 255  # 255 = ignore (never produced by the shipped data, kept for generality)
    labels = labels.long()
    ce = F.cross_entropy(logits, labels, ignore_index=255)
    y = F.one_hot(labels.clamp(0, classes - 1), classes).movedim(-1, 1).to(logits.dtype) * valid[:, None]
    p = logits.softmax(1) * valid[:, None]
    dims = (0, *range(2, logits.ndim))
    dice = (2 * (p * y).sum(dims) + 1e-5) / (p.sum(dims) + y.sum(dims) + 1e-5)
    present = y[:, 1:].sum(dims) > 0
    dice_loss = (1 - dice[1:][present]).mean() if bool(present.any()) else logits.sum() * 0
    return ce + dice_loss


def weighted_ce(logits: Tensor, target: Tensor, w: Tensor, valid: Tensor) -> Tensor:
    """``sum(valid * w * CE) / sum(valid)``: the denominator is the valid-voxel count, not the weight mass."""
    per_voxel = F.cross_entropy(logits, target.long().clamp(0, logits.shape[1] - 1), reduction="none")
    support = valid.bool().to(per_voxel.dtype)
    denominator = support.sum()
    if float(denominator) == 0.0:
        return logits.sum() * 0.0
    return (per_voxel * w.detach().to(per_voxel.dtype) * support).sum() / denominator


def weighted_dice(probabilities: Tensor, target: Tensor, w: Tensor, valid: Tensor) -> Tensor:
    """One minus weighted soft Dice, averaged over the foreground classes with positive weighted target mass."""
    mask = w.detach().to(probabilities.dtype) * valid.bool().to(probabilities.dtype)
    labels = target.long().clamp(0, probabilities.shape[1] - 1)
    terms = []
    for c in range(1, int(probabilities.shape[1])):
        target_mask = (labels == c) & valid.bool()
        target_mass = (mask * target_mask.to(mask.dtype)).sum()
        if float(target_mass) <= 0.0:
            continue
        numerator = 2.0 * (mask * probabilities[:, c] * target_mask.to(mask.dtype)).sum()
        denominator = (mask * probabilities[:, c]).sum() + target_mass
        if float(denominator) > 0.0:
            terms.append(numerator / denominator)
    if not terms:
        return probabilities.sum() * 0.0
    return 1.0 - torch.stack(terms).mean()


def pseudo_label_loss(logits: Tensor, target: Tensor, w: Tensor, valid: Tensor) -> Tensor:
    return weighted_ce(logits, target, w, valid) + weighted_dice(logits.softmax(dim=1), target, w, valid)


__all__ = ["supervised_loss", "weighted_ce", "weighted_dice", "pseudo_label_loss"]
