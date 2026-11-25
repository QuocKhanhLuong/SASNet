"""Evaluation on the native grid.

Two metric families are reported:

* ``native``: Dice, HD95 and ASSD in mm with the true anisotropic spacing.  Per frame and class; a class absent from both
  reference and prediction is skipped, HD95/ASSD are skipped when one side is empty.  Frames are averaged within a
  patient, then patients are averaged (patient macro).  ``fg_dice`` (mean over RV, MYO, LV) on the validation split is
  the model-selection metric.
* ``ssl4mis``: the metric computation of SSL4MIS ``test_2D_fully.py`` (medpy ``dc``, ``jc``, ``hd95``, ``asd`` per volume
  and class, averaged over volumes, then classes).  HD95/ASD are in voxels (no spacing), as in SSL4MIS; a class with an
  empty prediction or reference scores Dice by medpy and HD95 = ASD = 0, and is counted in ``empty_cases``.  Passing
  ``spacing=True`` gives the same statistics in mm.

For a single-class dataset (LA) the class lives in slot 3 ("LV"): report the per-class LV values, not the class mean.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy import ndimage

CLASS_NAMES = ("RV", "MYO", "LV")


@torch.no_grad()
def predict(model: torch.nn.Module, frames: Iterable[Any], device: torch.device) -> list[dict[str, Any]]:
    """Argmax predictions on the native grid in fp32 (no autocast)."""
    was_training = model.training
    model.eval()
    records = []
    for frame in frames:
        logits = model(frame.image.to(device), frame.geometry)["logits"]
        records.append({"patient": str(frame.patient_id), "frame": int(frame.frame_index), "target": frame.target[0].numpy().astype(np.uint8),
                        "prediction": logits.argmax(1)[0].cpu().numpy().astype(np.uint8), "spacing_zyx": tuple(frame.geometry["spacing_zyx"])})
    model.train(was_training)
    return records


def surface_distances(reference: np.ndarray, prediction: np.ndarray, spacing_zyx: Sequence[float]) -> tuple[float | None, float | None]:
    """Symmetric voxel-surface HD95 and ASSD in mm (``None`` when either mask or surface is empty)."""
    ref, pred = np.asarray(reference, dtype=bool), np.asarray(prediction, dtype=bool)
    if not ref.any() or not pred.any():
        return None, None
    structure = ndimage.generate_binary_structure(3, 1)
    ref_surface = ref ^ ndimage.binary_erosion(ref, structure=structure, border_value=0)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    if not ref_surface.any() or not pred_surface.any():
        return None, None
    spacing = tuple(float(v) for v in spacing_zyx)
    d_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)
    d_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((d_pred[ref_surface], d_ref[pred_surface])).astype(np.float64)
    return float(np.percentile(distances, 95)), float(distances.mean())


def _mean(values: Iterable[float | None]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _frame_metrics(reference: np.ndarray, prediction: np.ndarray, spacing_zyx: Sequence[float]) -> dict[str, Any]:
    per_class = {}
    for c, name in enumerate(CLASS_NAMES, 1):
        ref, pred = reference == c, prediction == c
        denominator = int(ref.sum()) + int(pred.sum())
        dice = None if denominator == 0 else float(2.0 * np.count_nonzero(ref & pred) / denominator)
        hd95, assd = surface_distances(ref, pred, spacing_zyx)
        per_class[name] = {"dice": dice, "reference_present": bool(ref.any()), "hd95_mm": hd95, "assd_mm": assd}
    return per_class


def native_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_patient[r["patient"]].append(_frame_metrics(r["target"], r["prediction"], r["spacing_zyx"]))
    per_patient = {}
    for pid, frames in sorted(by_patient.items()):
        classes = {}
        for name in CLASS_NAMES:
            # Dice counts only frames where the class is in the reference; HD95/ASSD every frame where both sides exist
            classes[name] = {"dice": _mean(f[name]["dice"] for f in frames if f[name]["reference_present"]),
                             "hd95_mm": _mean(f[name]["hd95_mm"] for f in frames),
                             "assd_mm": _mean(f[name]["assd_mm"] for f in frames)}
        per_patient[pid] = {"per_class": classes, **{f"fg_{k}": _mean(classes[n][key] for n in CLASS_NAMES)
                                                    for k, key in (("dice", "dice"), ("hd95_mm", "hd95_mm"), ("assd_mm", "assd_mm"))}}
    out: dict[str, Any] = {"patient_count": len(per_patient)}
    for k in ("fg_dice", "fg_hd95_mm", "fg_assd_mm"):
        out[k] = _mean(p[k] for p in per_patient.values())
    for name in CLASS_NAMES:
        out[name] = {key: _mean(p["per_class"][name][key] for p in per_patient.values()) for key in ("dice", "hd95_mm", "assd_mm")}
        out[name]["patient_count"] = sum(p["per_class"][name]["dice"] is not None for p in per_patient.values())
    out["per_patient"] = per_patient
    return out


def ssl4mis_metrics(records: Sequence[Mapping[str, Any]], *, spacing: bool = False) -> dict[str, Any]:
    from medpy import metric

    per_volume, empty = [], 0
    for r in records:
        pred, gt = np.asarray(r["prediction"]), np.asarray(r["target"])
        kwargs = {"voxelspacing": tuple(float(v) for v in r["spacing_zyx"])} if spacing else {}
        row: dict[str, Any] = {"patient": r["patient"], "frame": r["frame"]}
        for c, name in enumerate(CLASS_NAMES, 1):
            p, g = pred == c, gt == c
            if p.any() and g.any():
                values = {"dice": float(metric.binary.dc(p, g)), "jaccard": float(metric.binary.jc(p, g)),
                          "hd95": float(metric.binary.hd95(p, g, **kwargs)), "asd": float(metric.binary.asd(p, g, **kwargs)), "empty": False}
            else:
                any_side = bool(p.any() or g.any())
                values = {"dice": float(metric.binary.dc(p, g)) if any_side else 0.0, "jaccard": float(metric.binary.jc(p, g)) if any_side else 0.0,
                          "hd95": 0.0, "asd": 0.0, "empty": True}
                empty += 1
            row[name] = values
        per_volume.append(row)
    keys = ("dice", "jaccard", "hd95", "asd")
    classes = {name: {k: float(np.mean([v[name][k] for v in per_volume])) for k in keys} for name in CLASS_NAMES}
    return {"unit": "mm" if spacing else "voxels", "volumes": len(per_volume), "per_class": classes,
            "mean": {k: float(np.mean([classes[n][k] for n in CLASS_NAMES])) for k in keys}, "empty_cases": empty, "per_volume": per_volume}


def evaluate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"native": native_metrics(records), "ssl4mis": ssl4mis_metrics(records), "ssl4mis_mm": ssl4mis_metrics(records, spacing=True)}


__all__ = ["predict", "native_metrics", "ssl4mis_metrics", "evaluate", "surface_distances", "CLASS_NAMES"]
