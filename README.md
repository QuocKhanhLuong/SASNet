# SASNet: Slice-Aware State-Space Network for Semi-Supervised Anisotropic Cardiac MRI Segmentation

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**SASNet** is a 2.53 M-parameter network for semi-supervised segmentation of anisotropic cine and LGE MRI.

Short-axis stacks have 5–10 mm between slices against about 1.5 mm in plane, and neighbouring slices are misaligned
because each slice is acquired in its own breath-hold. SASNet keeps every convolution in-plane (no z down-sampling) and
adds through-plane context with two modules:

- a **physically stepped z state-space model (PhysZSSM)**;
- a **misalignment-aware slice fusion (SliceAlign)**, supervised for free by a shift-recovery task.

The encoder is initialised by distillation from the CineMA foundation model. The network is then trained with
UniMatch-style self-training from a few labelled patients.

### Key Features

| Feature | Description |
|---------|-------------|
| **Native anisotropic grid** | 1×3×3 convolutions, no z down-sampling, no resampling or cropping of the input volume |
| **PhysZSSM** | Bidirectional selective scan along z at every in-plane location. The step is scaled by the slice spacing, `Δ = softplus(dt(x)) · s_z / 10 mm` |
| **SliceAlign** | Predicts in-plane offsets between adjacent slices, warps both neighbours onto each slice and fuses them (identity at initialisation) |
| **Shift recovery** | Label-free supervision of SliceAlign: injected per-slice shifts must be recovered by the predicted offsets |
| **CineMA distillation** | Stage S1. Masked cosine distillation of the CineMA SAX encoder (1/8 and 1/16 grids) into the SASNet encoder, using training images only |
| **Semi-supervised training** | Stage S0. Confidence-thresholded self-training with two CutMix strong views and a feature-perturbation stream |
| **Small** | 2.53 M parameters at deployment (< 3 M budget) |

---

## Installation

### Requirements

- Python 3.12
- PyTorch 2.x with CUDA (tested with PyTorch 2.14 / CUDA 13.2 on an RTX 5000 Ada, 32 GB)

### Setup Environment

```bash
# 1. Clone the repository
git clone https://github.com/QuocKhanhLuong/SASNet.git
cd SASNet

# 2. Create conda environment
conda create -n sasnet python=3.12 -y
conda activate sasnet

# 3. Install PyTorch for your CUDA version (see pytorch.org), then the dependencies
pip install -r requirements.txt

# 4. Official CineMA package (needed for the S1 distillation only)
pip install git+https://github.com/mathpluscode/CineMA.git
```

---

## Dataset Preparation

### Supported Datasets

| Dataset | Task | Classes | Download Link |
|---------|------|---------|---------------|
| **ACDC** | Cine cardiac segmentation | 4 (BG, RV, MYO, LV) | [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| **M&Ms-1** | Multi-centre, multi-vendor cine | 4 (BG, RV, MYO, LV) | [M&Ms Challenge](https://www.ub.edu/mnms/) |
| **LA 2018** | Left atrium, LGE MRI | 2 (BG, LA) | [Atrial Segmentation Challenge 2018](https://www.cardiacatlas.org/atriaseg2018-challenge/) |

### Splits

Each split file in `splits/` lists the labelled, unlabelled, validation and test patients.

| Split | Labelled / Unlabelled / Val / Test (patients) | Notes |
|-------|-----------------------------------------------|-------|
| `acdc_10`, `acdc_5` | 7 / 63 / 10 / 20 and 3 / 67 / 10 / 20 | SSL4MIS ACDC split, 10 % and 5 % labels; ED + ES frames |
| `mnms1_10`, `mnms1_5` | 15 / 135 / 34 / 136 and 7 / 143 / 34 / 136 | official split (Training-Labeled / Validation / Testing) |
| `la2018_10`, `la2018_5` | 7 / 63 / 10 / 20 and 3 / 67 / 10 / 20 | UA-MT test list; 10 training cases held out for validation |

Unlabelled patients are always loaded without their masks.

### Directory Structure

Datasets must be downloaded, unzipped and placed in `data/`:

```
SASNet/
├── data/
│   ├── ACDC/database/training/         patient001 ... patient100
│   ├── MnM1/                           Training/Labeled, Validation, Testing, <info>.csv
│   └── LA2018/Training Set/            100 case folders (lgemri.nrrd, laendo.nrrd)
│
├── preprocessed_data/                  written by scripts/preprocess_*.py
└── runs/                               checkpoints and results
```

---

## Usage

### Preprocessing

Raw volumes are converted to `.npy` arrays plus a `dataset.json` with each patient's shape and spacing.
- **ACDC and M&Ms-1:** only the ED and ES frames are kept, and intensities are scaled by the 0.5/99.5 percentiles of the
  whole cine.
- **LA 2018:** converted as in UA-MT (x/y crop around the atrium, z-score), with seeded crop margins.

No volume is resampled.

```bash
# ACDC
python scripts/preprocess_acdc.py --raw data/ACDC/database/training --out preprocessed_data/acdc

# M&Ms-1
python scripts/preprocess_mnms1.py --raw data/MnM1 \
    --csv "data/MnM1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv" --out preprocessed_data/mnms1

# LA 2018
python scripts/preprocess_la2018.py --raw "data/LA2018/Training Set" --out preprocessed_data/la2018
```

### Training

Each run has two stages. The commands below are for ACDC 10 %; use any config in `configs/` for the other settings.

```bash
# Stage S1: CineMA encoder distillation (about 3 h on one GPU)
python scripts/distill.py --config configs/acdc_10.json --out runs/acdc_10/distill

# Stage S0: semi-supervised training, 150 epochs (about 1.5 h); the checkpoint is selected on the validation split
python scripts/train.py --config configs/acdc_10.json \
    --init runs/acdc_10/distill/encoder_init.pt --out runs/acdc_10/train
```

Running `train.py` again resumes from `runs/<name>/train/last.pt`. Use `--seed 43` for a replicate run.

| Config | Dataset / labels |
|--------|------------------|
| `configs/acdc_10.json`, `configs/acdc_5.json` | ACDC 10 % / 5 % |
| `configs/mnms1_10.json`, `configs/mnms1_5.json` | M&Ms-1 10 % / 5 % |
| `configs/la2018_10.json`, `configs/la2018_5.json` | LA 2018 10 % / 5 % |

### Evaluation

```bash
python scripts/evaluate.py --config configs/acdc_10.json --run runs/acdc_10/train
```

The script evaluates the validation-selected checkpoint on the test split once and refuses to run again. It writes
`TEST_RESULT.json` (per patient and per class) and the deployable model `sasnet.pt`:

```python
from sasnet import load_model
model = load_model("runs/acdc_10/train/sasnet.pt")   # input [B,1,D,H,W] in [0,1], geometry {"shape", "spacing_zyx"}
```

The reported metrics are:
- **Dice, HD95 and ASSD in mm** with the true anisotropic spacing, averaged over frames per patient, then over patients.
- **Dice, Jaccard, HD95 and ASD in voxels** following the SSL4MIS/medpy protocol, for comparison with published
  semi-supervised results.

For LA the single class is stored in slot 3, so read the `LV` entries.

---

## Results

Test-set results of the validation-selected checkpoint, seed 42:
- Dice and Jaccard are in %.
- For ACDC and M&Ms-1, values are the mean over RV, MYO and LV; for LA they are the single class.
- In the voxel columns, ASD is one-directional, following SSL4MIS.

| Dataset | Labels | Dice ↑ | Jaccard ↑ | HD95 mm ↓ | ASSD mm ↓ | HD95 vox ↓ | ASD vox ↓ |
|---------|--------|-------:|----------:|----------:|----------:|-----------:|----------:|
| ACDC | 10 % | 90.56 | 83.18 | 3.37 | 0.75 | 1.19 | 0.28 |
| ACDC | 5 % | 87.65 | 78.76 | 5.74 | 1.35 | 1.88 | 0.55 |
| M&Ms-1 | 10 % | 82.96 | 72.32 | 11.98 | 2.69 | 7.71 | 2.37 |
| M&Ms-1 | 5 % | 77.63 | 65.70 | 16.20 | 3.93 | 7.40 | 2.23 |
| LA 2018 | 10 % | 90.91 | 83.68 | 4.59 | 1.23 | 7.34 | 2.34 |
| LA 2018 | 5 % | 89.08 | 80.85 | 5.20 | 1.43 | 8.33 | 2.63 |

M&Ms-1 uses the official split. Its test set contains vendors that are absent or rare among the labelled training
patients.

---

## Ablation Study

### Components (ACDC 10 %, test set, mean of seeds 42 and 43)

| Arm | PhysZSSM | Spacing-scaled step | SliceAlign | Shift recovery | Dice ↑ | HD95 mm ↓ | ASSD mm ↓ |
|-----|:--------:|:-------------------:|:----------:|:--------------:|-------:|----------:|----------:|
| **Full** | ✓ | ✓ | ✓ | ✓ | 90.56 | **3.65** | **0.83** |
| A2 | ✓ | ✓ | | | 90.28 | 4.38 | 0.93 |
| A3 | ✓ | ✓ | ✓ | | 90.38 | 4.44 | 0.99 |
| A4 | | | ✓ | ✓ | 91.11 | 6.06 | 1.15 |
| A5 | | | | | 89.90 | 8.76 | 1.79 |

Paired per-patient tests pooled over both seeds (40 pairs):

- **SliceAlign with shift recovery (A4 vs A5):** +1.20 Dice (p < 0.001) and −2.70 mm HD95 (p = 0.011).
- **Adding PhysZSSM (Full vs A4):** −0.54 Dice (p = 0.019), but mean HD95 drops by 2.4 mm. The HD95 gain comes from
  removing gross 3D failures on a few patients, so it is not significant.

A1 (step not scaled by spacing, seed 42 only) gives 90.12 Dice and 3.90 mm HD95.

### Width (ACDC 10 %, seed 42)

| Size | Channels | Params | Dice ↑ | HD95 mm ↓ |
|------|----------|-------:|-------:|----------:|
| S | 16-32-64-112-128 | 1.20 M | 89.60 | 5.44 |
| **M** | 24-48-96-160-192 | 2.53 M | **90.56** | **3.37** |
| L | 40-72-144-240-288 | 5.66 M | 90.40 | 5.25 |

### Reproducing the ablations

```bash
python scripts/train.py --config configs/ablations/acdc_10_A4_no_physzssm.json \
    --init runs/acdc_10/distill/encoder_init.pt --out runs/acdc_10_A4/train
```

| Config | Change |
|--------|--------|
| `acdc_10_A1_no_physical_step` | step not scaled by the slice spacing |
| `acdc_10_A2_no_slicealign` | no SliceAlign, no shift recovery |
| `acdc_10_A3_no_shift_aux` | no shift recovery |
| `acdc_10_A4_no_physzssm` | no PhysZSSM |
| `acdc_10_A5_inplane_only` | none of the three |
| `acdc_10_width_S`, `acdc_10_width_L` | narrower / wider network |
| `acdc_10_gated_physzssm` | token-gated PhysZSSM (explored, not adopted) |

The ablations reuse the full model's S1 encoder; modules an arm does not have are skipped at loading. The S and L
widths need their own S1 run with the same config.

---

## Project Structure

```
SASNet/
├── sasnet/
│   ├── slicessm_student.py    SASNet: PhysZSSM, SliceAlign, shift recovery, export / load
│   ├── cinema_distill.py      stage S1: CineMA encoder distillation
│   ├── train.py               stage S0: semi-supervised training and model selection
│   ├── augment.py             in-plane weak / strong views, CutMix
│   ├── data.py                preprocessed datasets, split files, patient-balanced sampler
│   ├── losses.py, metrics.py
│   └── config.py
├── scripts/                   preprocess_*.py, distill.py, train.py, evaluate.py
├── configs/                   base.json, per-dataset configs, ablations/
├── splits/                    patient lists
└── tests/                     unit tests (python -m pytest -q tests)
```

---

## Citation

The citation will be added when the paper is published.

---

## License

This project is for **research purposes only**.

For commercial use, please contact the authors.

---

## Acknowledgements

This project builds upon the knowledge and codebases of:

- **CineMA**: a foundation model for cine cardiac MRI ([mathpluscode/CineMA](https://github.com/mathpluscode/CineMA)), used for encoder initialisation.
- **UniMatch**: Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation.
- **SSL4MIS** ([HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS)): ACDC semi-supervised split and evaluation protocol.
- **UA-MT** ([yulequan/UA-MT](https://github.com/yulequan/UA-MT)): LA 2018 preprocessing and split.
- **Mamba**: selective state-space models.
