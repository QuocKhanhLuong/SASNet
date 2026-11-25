#!/usr/bin/env python3
"""Prepare the 2018 Left Atrium Segmentation Challenge (Utah, 100 training volumes) for SASNet.

    python scripts/preprocess_la2018.py --raw "data/LA2018/Training Set" --out preprocessed_data/la2018

Conversion as in UA-MT (``la_heart_processing.py``): ``lgemri.nrrd`` / ``laendo.nrrd``, label = (laendo == 255), x/y crop
around the label box with random 10-20 px margins plus padding up to 112 x 112, whole-volume z-score; z is not cropped.
The upstream margins are unseeded; here they are drawn with ``numpy.random.default_rng(42)`` per case in sorted order,
so the conversion is reproducible.  The z-scored volume is then mapped to [0, 1] with its 0.5/99.5 percentiles.
One volume per case (frames ``["LGE"]``); the atrium is stored as class 3, so report the class-3 ("LV") metrics.
Voxels are 0.625 mm isotropic.  The split files follow the UA-MT ``train.list`` / ``test.list`` (test = 20 cases).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUTPUT_XY = (112, 112)
SPACING_MM = 0.625
LA_CLASS = 3


def crop_box(label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    w, h, _ = label.shape
    nz = np.nonzero(label)
    minx, maxx, miny, maxy = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()
    px, py = max(OUTPUT_XY[0] - (maxx - minx), 0) // 2, max(OUTPUT_XY[1] - (maxy - miny), 0) // 2
    minx = max(minx - int(rng.integers(10, 20)) - px, 0)
    maxx = min(maxx + int(rng.integers(10, 20)) + px, w)
    miny = max(miny - int(rng.integers(10, 20)) - py, 0)
    maxy = min(maxy + int(rng.integers(10, 20)) + py, h)
    rng.integers(5, 10), rng.integers(5, 10)  # upstream z margins: drawn and unused, kept so the stream matches
    return int(minx), int(maxx), int(miny), int(maxy)


def main() -> None:
    import nrrd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="the 100-volume 'Training Set' folder")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arrays = a.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cases = {}
    for folder in sorted(p for p in a.raw.iterdir() if p.is_dir()):
        cid = folder.name
        image, _ = nrrd.read(str(folder / "lgemri.nrrd"))
        label_raw, _ = nrrd.read(str(folder / "laendo.nrrd"))
        label = (label_raw == 255).astype(np.uint8)
        x0, x1, y0, y1 = crop_box(label, rng)
        z = ((image - np.mean(image)) / np.std(image)).astype(np.float32)[x0:x1, y0:y1].transpose(2, 1, 0)  # -> z, y, x
        lo, hi = np.percentile(z, [0.5, 99.5])
        img01 = np.clip((z - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
        np.save(arrays / f"{cid}_vol.npy", np.ascontiguousarray(img01[None]), allow_pickle=False)
        np.save(arrays / f"{cid}_t0_mask.npy", np.ascontiguousarray(label[x0:x1, y0:y1].transpose(2, 1, 0) * LA_CLASS), allow_pickle=False)
        cases[cid] = {"image": f"arrays/{cid}_vol.npy", "masks": {"0": f"arrays/{cid}_t0_mask.npy"}, "shape": list(img01.shape),
                      "spacing_zyx": [SPACING_MM] * 3, "crop_xy": [x0, x1, y0, y1]}
        print(f"{cid} shape {list(img01.shape)}", flush=True)
    (a.out / "dataset.json").write_text(json.dumps({"schema": "sasnet.dataset.v1", "name": "la2018", "frames": ["LGE"], "cases": cases}, indent=1) + "\n")
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
