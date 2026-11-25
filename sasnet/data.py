"""Prepared datasets, split files and the patient-balanced one-L / two-U sampler.

A prepared dataset is a directory with ``dataset.json`` and ``arrays/`` (written by ``scripts/preprocess_*.py``)::

    {"schema": "sasnet.dataset.v1", "name": "acdc", "frames": ["ED", "ES"],
     "cases": {"patient001": {"image": "arrays/patient001_edes.npy",            # float32 [F, D, H, W] in [0, 1]
                              "masks": {"0": "arrays/patient001_t0_mask.npy",   # uint8 [D, H, W], labels 0..3
                                        "1": "arrays/patient001_t1_mask.npy"},
                              "shape": [D, H, W], "spacing_zyx": [sz, sy, sx]}}}

Masks are optional per case (unlabelled training cases need none).  A split file (``splits/*.json``) lists patient IDs
under ``labeled``, ``unlabeled``, ``val`` and ``test``.  Unlabelled cases are always loaded image-only, even when the
prepared directory holds their masks.  Volumes stay on their native grid: nothing is cropped or resampled.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DATASET_SCHEMA = "sasnet.dataset.v1"
FrameKey = tuple[str, int]


@dataclass(frozen=True)
class Frame:
    image: torch.Tensor          # float32 [1, 1, D, H, W]
    target: torch.Tensor | None  # int64 [1, D, H, W] or None
    geometry: dict[str, Any]     # {"shape": (D, H, W), "spacing_zyx": (sz, sy, sx)} in mm
    patient_id: str
    frame_index: int


class PreparedDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta = json.loads((self.root / "dataset.json").read_text())
        if meta.get("schema") != DATASET_SCHEMA:
            raise ValueError(f"{self.root}: not a {DATASET_SCHEMA} prepared dataset")
        self.name = meta["name"]
        self.frames = tuple(range(len(meta["frames"])))
        self.cases: dict[str, dict[str, Any]] = meta["cases"]
        self._images: dict[str, np.ndarray] = {}

    def geometry(self, pid: str) -> dict[str, Any]:
        case = self.cases[pid]
        return {"shape": tuple(int(v) for v in case["shape"]), "spacing_zyx": tuple(float(v) for v in case["spacing_zyx"])}

    def _image(self, pid: str) -> np.ndarray:
        if pid not in self._images:
            array = np.load(self.root / self.cases[pid]["image"], mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[0] != len(self.frames) or tuple(array.shape[1:]) != self.geometry(pid)["shape"]:
                raise ValueError(f"{pid}: image must be [{len(self.frames)}, D, H, W] matching its declared shape")
            self._images[pid] = array
        return self._images[pid]

    def load(self, pid: str, frame: int, *, with_mask: bool) -> Frame:
        image = torch.from_numpy(np.array(self._image(pid)[frame], dtype=np.float32, copy=True))[None, None]
        if not bool(torch.isfinite(image).all()) or float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"{pid}: image must be finite and normalized to [0, 1]")
        target = None
        if with_mask:
            masks = self.cases[pid].get("masks") or {}
            if str(frame) not in masks:
                raise ValueError(f"{pid}: no mask for frame {frame}")
            labels = np.load(self.root / masks[str(frame)], allow_pickle=False)
            if tuple(labels.shape) != self.geometry(pid)["shape"] or not np.isin(labels, (0, 1, 2, 3)).all():
                raise ValueError(f"{pid}: mask must cover the native grid with labels 0..3")
            target = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))[None]
        return Frame(image, target, self.geometry(pid), pid, frame)


class SplitData:
    """One prepared dataset seen through one split file.  Training code can reach val/test only through ``eval_frames``."""

    def __init__(self, prepared: str | Path, split: str | Path) -> None:
        self.dataset = PreparedDataset(prepared)
        ids = json.loads(Path(split).read_text())
        self.ids = {k: sorted(ids[k]) for k in ("labeled", "unlabeled", "val", "test")}
        flat = sum(self.ids.values(), [])
        if len(flat) != len(set(flat)):
            raise ValueError("split sets must be disjoint")
        missing = set(flat) - set(self.dataset.cases)
        if missing:
            raise ValueError(f"split names cases absent from the prepared dataset: {sorted(missing)[:5]}")
        if len(self.ids["unlabeled"]) < 2:
            raise ValueError("at least two unlabelled cases are needed for different-patient CutMix pairs")

    def labeled_keys(self) -> list[FrameKey]:
        return [(p, t) for p in self.ids["labeled"] for t in self.dataset.frames]

    def unlabeled_keys(self) -> list[FrameKey]:
        return [(p, t) for p in self.ids["unlabeled"] for t in self.dataset.frames]

    def load_labeled(self, key: FrameKey) -> Frame:
        if key[0] not in self.ids["labeled"]:
            raise ValueError("labelled loads are restricted to the labelled training cases")
        return self.dataset.load(key[0], key[1], with_mask=True)

    def load_unlabeled(self, key: FrameKey) -> Frame:
        if key[0] not in self.ids["unlabeled"]:
            raise ValueError("unlabelled loads are restricted to the unlabelled training cases")
        return self.dataset.load(key[0], key[1], with_mask=False)

    def eval_frames(self, split: str) -> list[Frame]:
        if split not in ("val", "test"):
            raise ValueError("eval split must be val or test")
        return [self.dataset.load(p, t, with_mask=True) for p in self.ids[split] for t in self.dataset.frames]


class PatientBalancedSampler:
    """Checkpointable one-L / two-U sampler.

    Each sampler epoch has one update per unlabelled patient.  The anchor patient follows a shuffled permutation and the
    partner is the same permutation cyclically shifted by one, so the two U frames always come from different patients
    and every patient is used once as an anchor.  The frame inside a patient is drawn at random.  Labelled frames are
    shuffled and cycled independently.
    """

    def __init__(self, data: SplitData, *, seed: int = 42) -> None:
        self.seed = int(seed)
        self.py_rng = random.Random(self.seed)
        self.epoch_rng = random.Random(self.seed ^ 0x5047_5352)
        self.lkeys = tuple(data.labeled_keys())
        ukeys = tuple(data.unlabeled_keys())
        self._u_by_patient = {pid: tuple(k for k in ukeys if k[0] == pid) for pid in dict.fromkeys(k[0] for k in ukeys)}
        self._patients = tuple(sorted(self._u_by_patient))
        self.steps_per_epoch = len(self._patients)
        self.epoch = self.update = 0
        self._new_epoch()

    def _new_epoch(self) -> None:
        self._l_stream = list(self.lkeys)
        self.py_rng.shuffle(self._l_stream)
        self._l_cursor = 0
        self._anchor = list(self._patients)
        self.epoch_rng.shuffle(self._anchor)
        self._partner = self._anchor[1:] + self._anchor[:1]
        self._u_cursor = 0

    def _next_labeled(self) -> FrameKey:
        if self._l_cursor >= len(self._l_stream):
            self._l_stream = list(self.lkeys)
            self.py_rng.shuffle(self._l_stream)
            self._l_cursor = 0
        key = self._l_stream[self._l_cursor]
        self._l_cursor += 1
        return key

    def _frame_of(self, pid: str) -> FrameKey:
        frames = self._u_by_patient[pid]
        return frames[self.epoch_rng.randrange(len(frames))]

    def next_step(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        if self._u_cursor >= self.steps_per_epoch:
            self.epoch += 1
            self._new_epoch()
        anchor, partner = self._anchor[self._u_cursor], self._partner[self._u_cursor]
        sample = (self._next_labeled(), self._frame_of(anchor), self._frame_of(partner))
        self._u_cursor += 1
        self.update += 1
        return sample

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "epoch": self.epoch, "update": self.update, "py_rng": self.py_rng.getstate(),
                "epoch_rng": self.epoch_rng.getstate(), "l_stream": list(self._l_stream), "l_cursor": self._l_cursor,
                "anchor": list(self._anchor), "partner": list(self._partner), "u_cursor": self._u_cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("sampler seed mismatch")
        self.epoch, self.update = int(state["epoch"]), int(state["update"])
        self.py_rng.setstate(_tuple_state(state["py_rng"]))
        self.epoch_rng.setstate(_tuple_state(state["epoch_rng"]))
        self._l_stream = [tuple(k) for k in state["l_stream"]]
        self._l_cursor = int(state["l_cursor"])
        self._anchor, self._partner = list(state["anchor"]), list(state["partner"])
        self._u_cursor = int(state["u_cursor"])


def _tuple_state(value: Any) -> Any:
    return tuple(_tuple_state(v) for v in value) if isinstance(value, (list, tuple)) else value


__all__ = ["Frame", "PreparedDataset", "SplitData", "PatientBalancedSampler", "DATASET_SCHEMA"]
