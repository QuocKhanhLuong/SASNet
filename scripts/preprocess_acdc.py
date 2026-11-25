#!/usr/bin/env python3
"""Prepare ACDC (training set, 100 patients) for SASNet.

    python scripts/preprocess_acdc.py --raw data/ACDC/database/training --out preprocessed_data/acdc

Per patient only the ED and ES frames are kept (float32 ``[2, D, H, W]``, z/y/x order).  Intensities are normalised over
the whole cine with the 0.5/99.5 percentiles and clipped to [0, 1].  Masks are stored for both frames (uint8, labels
0 BG, 1 RV, 2 MYO, 3 LV).  The native grid is kept: no crop, no resampling; spacing comes from the NIfTI header (mm).
Which masks are used is decided by the split file at training time (unlabelled cases are always loaded image-only).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def normalize_cine(cine_xyzt: np.ndarray) -> np.ndarray:
    """``[X, Y, Z, T]`` -> ``[T, Z, Y, X]`` scaled by the 0.5/99.5 percentiles of the whole cine, clipped to [0, 1]."""
    ordered = np.transpose(np.asarray(cine_xyzt), (3, 2, 1, 0))
    lo, hi = np.percentile(ordered, [0.5, 99.5])
    return np.clip((ordered.astype(np.float32) - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def ed_es(info: Path, frames: int) -> tuple[int, int]:
    values = {}
    for line in info.read_text().splitlines():
        match = re.fullmatch(r"\s*(ED|ES)\s*:\s*(\d+)\s*", line, flags=re.I)
        if match:
            values[match.group(1).upper()] = int(match.group(2)) - 1
    if set(values) != {"ED", "ES"} or not 0 <= values["ED"] < frames or not 0 <= values["ES"] < frames:
        raise SystemExit(f"{info}: invalid ED/ES")
    return values["ED"], values["ES"]


def find(folder: Path, stem: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        if (folder / f"{stem}{suffix}").is_file():
            return folder / f"{stem}{suffix}"
    raise SystemExit(f"missing {folder / stem}.nii(.gz)")


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="ACDC 'training' folder with patient001 ... patient100")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arrays = a.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    cases = {}
    for folder in sorted(p for p in a.raw.iterdir() if p.is_dir() and p.name.startswith("patient")):
        pid = folder.name
        nii = nib.load(str(find(folder, f"{pid}_4d")))
        cine = normalize_cine(np.asarray(nii.dataobj))
        zooms = [float(v) for v in nii.header.get_zooms()[:3]]  # x, y, z in mm
        ed, es = ed_es(folder / "Info.cfg", cine.shape[0])
        np.save(arrays / f"{pid}_edes.npy", np.ascontiguousarray(np.stack([cine[ed], cine[es]])), allow_pickle=False)
        masks = {}
        for index, frame in ((0, ed), (1, es)):
            labels = np.asarray(nib.load(str(find(folder, f"{pid}_frame{frame + 1:02d}_gt"))).dataobj).transpose(2, 1, 0)
            if labels.shape != cine.shape[1:] or not np.isin(labels, (0, 1, 2, 3)).all():
                raise SystemExit(f"{pid}: invalid mask for frame {frame}")
            np.save(arrays / f"{pid}_t{index}_mask.npy", labels.astype(np.uint8), allow_pickle=False)
            masks[str(index)] = f"arrays/{pid}_t{index}_mask.npy"
        cases[pid] = {"image": f"arrays/{pid}_edes.npy", "masks": masks, "shape": list(cine.shape[1:]),
                      "spacing_zyx": [zooms[2], zooms[1], zooms[0]], "source_frames": {"ED": ed, "ES": es}}
        print(f"{pid} ED {ed} ES {es} shape {list(cine.shape[1:])}", flush=True)
    (a.out / "dataset.json").write_text(json.dumps({"schema": "sasnet.dataset.v1", "name": "acdc", "frames": ["ED", "ES"], "cases": cases}, indent=1) + "\n")
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
