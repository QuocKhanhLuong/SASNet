#!/usr/bin/env python3
"""Prepare M&Ms-1 for SASNet (official split: Training/Labeled 150, Validation 34, Testing 136).

    python scripts/preprocess_mnms1.py --raw data/MnM1 \
        --csv "data/MnM1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv" --out preprocessed_data/mnms1

``--raw`` must contain ``Training/Labeled``, ``Validation`` and ``Testing`` with ``<id>/<id>_sa.nii.gz`` and
``<id>_sa_gt.nii.gz``.  ``Training/Unlabeled`` (no ED/ES annotation) is not used.  ED/ES frames come from the official
CSV.  Labels are remapped to the ACDC order (1 RV, 2 MYO, 3 LV).  Intensities are normalised over the whole cine with the
0.5/99.5 percentiles.  The native grid is kept; spacing is the header zooms, which the M&Ms description gives in mm
(the headers declare unknown units).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from preprocess_acdc import normalize_cine

MNMS_TO_ACDC = np.array([0, 3, 2, 1], dtype=np.uint8)  # M&Ms: 1 LV, 2 MYO, 3 RV


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with a.csv.open(newline="") as stream:
        rows = {r["External code"]: r for r in csv.DictReader(stream)}
    arrays = a.out / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    cases = {}
    for sub in ("Training/Labeled", "Validation", "Testing"):
        for folder in sorted(p for p in (a.raw / sub).iterdir() if p.is_dir()):
            pid = folder.name
            nii = nib.load(str(folder / f"{pid}_sa.nii.gz"))
            cine = normalize_cine(np.asarray(nii.dataobj))
            zooms = [float(v) for v in nii.header.get_zooms()[:3]]
            ed, es = int(rows[pid]["ED"]), int(rows[pid]["ES"])
            np.save(arrays / f"{pid}_edes.npy", np.ascontiguousarray(np.stack([cine[ed], cine[es]])), allow_pickle=False)
            gt = np.asarray(nib.load(str(folder / f"{pid}_sa_gt.nii.gz")).dataobj)
            masks = {}
            for index, frame in ((0, ed), (1, es)):
                raw = np.asarray(gt[..., frame], dtype=np.float64)
                labels = np.rint(raw)  # some masks are float64 with values such as 1.0000000196
                if float(np.abs(raw - labels).max()) > 1e-3 or not np.isin(labels, (0, 1, 2, 3)).all() or not labels.any():
                    raise SystemExit(f"{pid}: invalid mask at frame {frame}")
                np.save(arrays / f"{pid}_t{index}_mask.npy", MNMS_TO_ACDC[labels.transpose(2, 1, 0).astype(np.int64)], allow_pickle=False)
                masks[str(index)] = f"arrays/{pid}_t{index}_mask.npy"
            cases[pid] = {"image": f"arrays/{pid}_edes.npy", "masks": masks, "shape": list(cine.shape[1:]),
                          "spacing_zyx": [zooms[2], zooms[1], zooms[0]], "source_frames": {"ED": ed, "ES": es},
                          "vendor": rows[pid]["Vendor"], "subset": sub}
            print(f"{pid} {sub} ED {ed} ES {es} shape {list(cine.shape[1:])}", flush=True)
    (a.out / "dataset.json").write_text(json.dumps({"schema": "sasnet.dataset.v1", "name": "mnms1", "frames": ["ED", "ES"], "cases": cases}, indent=1) + "\n")
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
