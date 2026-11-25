"""CPU unit tests (software checks only; they say nothing about segmentation quality)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sasnet.augment import ViewGenerator, center_crop_pad, cutmix_box, mix_tensor  # noqa: E402
from sasnet.config import load_config  # noqa: E402
from sasnet.data import PatientBalancedSampler, SplitData  # noqa: E402
from sasnet.losses import supervised_loss, weighted_ce, weighted_dice  # noqa: E402
from sasnet.metrics import native_metrics, ssl4mis_metrics  # noqa: E402
from sasnet.slicessm_student import (PhysZSSM, SliceAlign, apply_slice_shifts, build_model, export_model, load_model,  # noqa: E402
                                     shift_recovery_loss, slice_shift_aux)
from sasnet.train import training_step, TRAIN_DEFAULTS  # noqa: E402

GEOM = {"shape": (5, 40, 36), "spacing_zyx": (8.0, 1.5, 1.5)}


def _model(**kw):
    torch.manual_seed(0)
    return build_model({**load_config(ROOT / "configs/acdc_10.json")["model"], **kw})


def test_parameter_budget_and_shapes():
    model = _model()
    assert sum(p.numel() for p in model.parameters()) == 2_531_734
    out = model.eval()(torch.rand(2, 1, 5, 40, 36), GEOM)
    assert out["logits"].shape == (2, 4, 5, 40, 36)
    with pytest.raises(ValueError):
        build_model({"channels": [40, 72, 144, 240, 288]})  # above the 3 M cap unless parameter_cap is lifted


def test_slicealign_is_identity_at_init_and_physzssm_uses_spacing():
    x = torch.randn(1, 16, 4, 6, 6)
    out, flow = SliceAlign(16)(x)
    assert torch.equal(out, x) and torch.count_nonzero(flow) == 0
    torch.manual_seed(1)
    block = PhysZSSM(16).eval()
    a, b = block(x, torch.tensor([5.0])), block(x, torch.tensor([10.0]))
    assert not torch.allclose(a, b)
    none = PhysZSSM(16, spacing_mode="none")
    none.load_state_dict(block.state_dict())
    assert torch.allclose(none(x, torch.tensor([5.0])), none(x, torch.tensor([10.0])))


def test_shift_recovery_target_and_determinism():
    flow = torch.zeros(1, 2, 3, 4, 4)
    shifts = torch.tensor([[[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 0.0]]])
    target = flow.clone()
    target[:, 0, 0], target[:, 1, 1], target[:, 0, 2], target[:, 1, 2] = 1.0, 1.0, -1.0, -1.0
    assert float(shift_recovery_loss(target, flow, shifts)) == 0.0
    moved, inside = apply_slice_shifts(torch.rand(1, 1, 4, 16, 16), shifts)
    assert moved.shape == (1, 1, 4, 16, 16) and inside.dtype == torch.bool
    model, image = _model().train(), torch.rand(1, 1, 5, 40, 36)
    a = slice_shift_aux(model, [(image, GEOM)], seed=42, update=7, max_px=8)
    b = slice_shift_aux(model, [(image, GEOM)], seed=42, update=7, max_px=8)
    assert torch.equal(a, b)
    assert slice_shift_aux(_model(align=False), [(image, GEOM)], seed=42, update=7, max_px=8) is None


def test_losses():
    logits = torch.randn(1, 4, 3, 8, 8, requires_grad=True)
    labels = torch.randint(0, 4, (1, 3, 8, 8))
    supervised_loss(logits, labels).backward()
    assert logits.grad is not None
    zero_w = torch.zeros(1, 3, 8, 8)
    valid = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    assert float(weighted_ce(logits, labels, zero_w, valid)) == 0.0
    assert float(weighted_dice(logits.softmax(1), labels, zero_w, valid)) == 0.0


def test_cutmix_box_is_aligned_inside_support():
    valid = torch.zeros(1, 4, 64, 80, dtype=torch.bool)
    valid[:, :, 8:60, 4:76] = True
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        y0, y1, x0, x1 = cutmix_box(valid, gen)
        assert all(v % 16 == 0 for v in (y0, y1, x0, x1)) and bool(valid[0, 0, y0:y1, x0:x1].all())
    a, b = torch.zeros(1, 1, 2, 4, 4), torch.ones(1, 1, 2, 4, 4)
    assert float(mix_tensor(a, b, (0, 2, 0, 2)).sum()) == 2 * 4
    assert center_crop_pad(torch.ones(1, 2, 3, 5), (2, 5, 3)).shape == (1, 2, 5, 3)


def test_export_roundtrip(tmp_path):
    model = _model().eval()
    export_model(model, tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    x = torch.rand(1, 1, 5, 40, 36)
    assert torch.allclose(model(x, GEOM)["logits"], loaded(x, GEOM)["logits"])


def _toy_dataset(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    (root / "arrays").mkdir(parents=True)
    cases = {}
    for i in range(6):
        pid = f"p{i}"
        np.save(root / "arrays" / f"{pid}.npy", rng.random((2, 5, 40, 36), dtype=np.float32))
        masks = {}
        for t in range(2):
            m = np.zeros((5, 40, 36), np.uint8)
            m[:, 10:30, 8:28] = 1
            m[:, 14:26, 12:24] = 2
            m[:, 17:23, 15:21] = 3
            np.save(root / "arrays" / f"{pid}_t{t}.npy", m)
            masks[str(t)] = f"arrays/{pid}_t{t}.npy"
        cases[pid] = {"image": f"arrays/{pid}.npy", "masks": masks, "shape": [5, 40, 36], "spacing_zyx": [8.0, 1.5, 1.5]}
    (root / "dataset.json").write_text(json.dumps({"schema": "sasnet.dataset.v1", "name": "toy", "frames": ["ED", "ES"], "cases": cases}))
    split = root / "split.json"
    split.write_text(json.dumps({"labeled": ["p0"], "unlabeled": ["p1", "p2", "p3"], "val": ["p4"], "test": ["p5"]}))
    return root, split


def test_sampler_pairs_and_training_step(tmp_path):
    data = SplitData(*_toy_dataset(tmp_path / "toy"))
    sampler = PatientBalancedSampler(data, seed=42)
    steps = [sampler.next_step() for _ in range(9)]
    assert all(a[0] != b[0] for _, a, b in steps)
    state = sampler.state_dict()
    nxt = sampler.next_step()
    other = PatientBalancedSampler(data, seed=42)
    other.load_state_dict(state)
    assert other.next_step() == nxt
    views = ViewGenerator(42)
    lk, ka, kb = steps[0]
    lf, fa, fb = data.load_labeled(lk), data.load_unlabeled(ka), data.load_unlabeled(kb)
    assert fa.target is None
    lab = views.views(lf.image, lf.geometry, lf.target)
    ua, ub = views.views(fa.image, fa.geometry), views.views(fb.image, fb.geometry)
    ua["strong2"], ub["strong2"] = views.strong(ua["weak"]), views.strong(ub["weak"])
    boxes = [cutmix_box(ua["valid"], views.cutmix_generator, partner_valid=ub["valid"]) for _ in range(2)]
    cfg = {**TRAIN_DEFAULTS, "lambda_u_warmup": 0}
    model = _model().train()
    terms = training_step(model, cfg, 5, lab, ua, ub, boxes, fp_generator=torch.Generator().manual_seed(1), seed=42)
    terms["loss"].backward()
    assert torch.isfinite(terms["loss"]) and float(terms["loss_u"]) >= 0 and "loss_shift" in terms


def test_metrics_perfect_prediction():
    target = np.zeros((4, 20, 20), np.uint8)
    target[:, 5:15, 5:15] = 1
    target[:, 7:13, 7:13] = 2
    target[:, 9:11, 9:11] = 3
    rec = [{"patient": "a", "frame": 0, "target": target, "prediction": target.copy(), "spacing_zyx": (8.0, 1.5, 1.5)}]
    native = native_metrics(rec)
    assert native["fg_dice"] == 1.0 and native["fg_hd95_mm"] == 0.0
    assert ssl4mis_metrics(rec)["mean"]["dice"] == 1.0
