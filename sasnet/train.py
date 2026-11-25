"""Semi-supervised training (UniMatch-style self-training) of SASNet, model selection on the validation split.

One update draws one labelled frame and two unlabelled frames from different patients (A = anchor, B = partner):

* supervised: CE + Dice of the labelled strong view against its (spatially augmented) mask;
* unlabelled, after ``lambda_u_warmup`` labelled-only updates: the model's own clean weak prediction on A and B gives
  pseudo-labels and confidence; two strong views are mixed with one full-depth CutMix box each, and a third stream is
  the weak view of A with channel dropout on every decoder input (feature perturbation).  Each stream is trained with
  CE + Dice weighted by ``confidence >= tau``; stream weights 0.25 / 0.25 / 0.5;
* SliceAlign shift recovery: per-slice shifts are injected into the labelled strong view (and the weak view of A once
  the unlabelled loss is on) and the predicted stage-3 offsets must change by the injected relative shift.

``total = L_sup + lambda_u * L_u + w_shift * L_shift``; ``lambda_u`` ramps linearly to its maximum.  Every
``updates_per_epoch`` updates the model is evaluated on the validation split (fp32) and the best patient-macro
foreground Dice is kept.  The run is resumable from ``last.pt`` with identical random streams.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .augment import ViewGenerator, center_crop_pad, cutmix_box, mix_tensor
from .data import PatientBalancedSampler, SplitData
from .losses import pseudo_label_loss, supervised_loss
from .metrics import native_metrics, predict
from .slicessm_student import ENCODER_PREFIXES, build_model, slice_shift_aux

TRAIN_DEFAULTS: dict[str, Any] = {
    "updates": 12000, "updates_per_epoch": 80, "lr": 1e-3, "weight_decay": 1e-4, "warmup_fraction": 0.05, "grad_clip": 1.0,
    "lambda_u_max": 1.0, "lambda_u_warmup": 800, "lambda_u_ramp_fraction": 0.1, "tau": 0.95,
    "stream_weights": [0.25, 0.25, 0.5], "fp_dropout": 0.5, "cutmix_area": [0.25, 0.5], "cutmix_align_px": 16,
    "shift_aux": {"weight": 0.1, "max_px": 8},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(device: torch.device) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=device.type == "cuda")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False


def cosine_lr(update: int, total: int, base_lr: float, warmup_fraction: float) -> float:
    warmup = max(1, int(math.ceil(total * warmup_fraction))) if warmup_fraction > 0 else 0
    if warmup and update < warmup:
        return base_lr * float(update + 1) / float(warmup)
    progress = min(1.0, max(0.0, (update - warmup + 1) / max(1, total - warmup)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def lambda_u(update: int, cfg: Mapping[str, Any]) -> float:
    warmup = int(cfg["lambda_u_warmup"])
    if update < warmup:
        return 0.0
    return min(float(cfg["lambda_u_max"]), (update - warmup + 1) / max(1, math.ceil(float(cfg["lambda_u_ramp_fraction"]) * int(cfg["updates"]))))


def load_encoder(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    """Import the encoder weights of a CineMA-distilled initialisation (``scripts/distill.py``).

    Modules the model does not have (an ablation without SliceAlign or z-SSM blocks) are dropped; any other mismatch is an
    error.  Decoder and head keep their own initialisation.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in payload["encoder"].items() if k.startswith(ENCODER_PREFIXES)}
    own = model.state_dict()
    dropped = sorted(k for k in state if k not in own)
    state = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k.startswith(ENCODER_PREFIXES) and k not in state and ".zgate." not in k]
    if missing:
        raise ValueError(f"encoder initialisation lacks {missing[:5]}")
    model.load_state_dict(state, strict=False)
    return {"path": str(path), "loaded": len(state), "dropped_absent_modules": len(dropped)}


def training_step(model: torch.nn.Module, cfg: Mapping[str, Any], update: int, labeled: Mapping[str, Any], ua: Mapping[str, Any],
                  ub: Mapping[str, Any], boxes: list[tuple[int, int, int, int]], *, fp_generator: torch.Generator, seed: int) -> dict[str, Tensor]:
    """Loss of one update.  ``labeled``/``ua``/``ub`` are view dicts (weak, strong, strong2, valid, geometry[, target])."""
    out_l = model(labeled["strong"], labeled["geometry"])
    loss_sup = supervised_loss(out_l["logits"], labeled["target"])
    loss_u = loss_sup * 0.0
    u_active = update >= int(cfg["lambda_u_warmup"])
    lam_u = lambda_u(update, cfg)
    if u_active:
        anchor = tuple(int(v) for v in ua["weak"].shape[-3:])
        valid_a = ua["valid"]
        valid_b = center_crop_pad(ub["valid"], anchor, pad_value=0).bool()
        with torch.no_grad():
            p_a = model(ua["weak"], ua["geometry"])["logits"].detach().softmax(1)
            p_b = center_crop_pad(model(ub["weak"], ub["geometry"])["logits"].softmax(1), anchor)
        images = [mix_tensor(ua["strong"], center_crop_pad(ub["strong"], anchor), boxes[0]),
                  mix_tensor(ua["strong2"], center_crop_pad(ub["strong2"], anchor), boxes[1])]
        logits = [model(img, ua["geometry"])["logits"] for img in images]
        logits.append(model(ua["weak"], ua["geometry"], skip_dropout_p=float(cfg["fp_dropout"]), dropout_generator=fp_generator)["logits"])
        with torch.no_grad():
            valid = [mix_tensor(valid_a, valid_b, box) for box in boxes] + [valid_a]
            targets = []
            for p in [mix_tensor(p_a, p_b, box) for box in boxes] + [p_a]:
                confidence, label = p.max(1)
                targets.append((label, (confidence >= float(cfg["tau"])).float()))
        weights = [float(w) for w in cfg["stream_weights"]]
        loss_u = sum(w * pseudo_label_loss(lg, y, conf, v) for w, lg, (y, conf), v in zip(weights, logits, targets, valid))
    total = loss_sup + lam_u * loss_u
    terms = {"loss_sup": loss_sup.detach(), "loss_u": loss_u.detach(), "lambda_u": torch.tensor(lam_u)}
    shift = cfg.get("shift_aux") or {}
    if float(shift.get("weight", 0.0)) > 0 and update >= int(shift.get("start_update", 0)):
        pairs = [(labeled["strong"], labeled["geometry"])] + ([(ua["weak"], ua["geometry"])] if u_active else [])
        loss_shift = slice_shift_aux(model, pairs, seed=seed, update=update, max_px=int(shift.get("max_px", 8)))
        if loss_shift is not None:
            total = total + float(shift["weight"]) * loss_shift
            terms["loss_shift"] = loss_shift.detach()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(f"non-finite loss at update {update}")
    terms["loss"] = total
    return terms


def _views(view_gen: ViewGenerator, frame: Any, device: torch.device, *, labeled: bool) -> dict[str, Any]:
    image = frame.image.to(device)
    target = frame.target.to(device) if labeled else None
    return view_gen.views(image, frame.geometry, target)


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def train(cfg: Mapping[str, Any], out_dir: str | Path, *, device: str = "cuda", init: str | Path | None = None,
          max_updates: int | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    configure_determinism(dev)
    seed = int(cfg.get("seed", 42))
    tcfg = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    data = SplitData(cfg["data"]["prepared"], cfg["data"]["split"])
    val_frames = data.eval_frames("val")

    seed_everything(seed)
    model = build_model(cfg.get("model"))
    init_receipt = load_encoder(model, init) if init else None
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    sampler = PatientBalancedSampler(data, seed=seed)
    view_gen = ViewGenerator(seed)
    fp_generator = torch.Generator(device=dev).manual_seed(seed + 2003)
    updates = int(tcfg["updates"]) if max_updates is None else min(int(tcfg["updates"]), int(max_updates))
    per_epoch = int(tcfg["updates_per_epoch"])
    start, best, best_epoch, best_state, history = 0, -math.inf, None, None, []
    last = out / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"])
        view_gen.load_state_dict(ck["view_generator"])
        fp_generator.set_state(ck["fp_generator"])
        start, best, best_epoch, best_state, history = ck["update"], ck["best"], ck["best_epoch"], ck["best_state"], ck["history"]
        print(f"[train] resumed at update {start}", flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    sums, steps, tick = {"loss": 0.0, "loss_sup": 0.0, "loss_u": 0.0, "loss_shift": 0.0}, 0, time.perf_counter()
    for update in range(start, updates):
        lk, uk_a, uk_b = sampler.next_step()
        labeled = _views(view_gen, data.load_labeled(lk), dev, labeled=True)
        ua = _views(view_gen, data.load_unlabeled(uk_a), dev, labeled=False)
        ub = _views(view_gen, data.load_unlabeled(uk_b), dev, labeled=False)
        ua["strong2"], ub["strong2"] = view_gen.strong(ua["weak"]), view_gen.strong(ub["weak"])
        partner_valid = center_crop_pad(ub["valid"], ua["valid"].shape[-3:], pad_value=0)
        boxes = [cutmix_box(ua["valid"], view_gen.cutmix_generator, partner_valid=partner_valid,
                            area_fraction=tcfg["cutmix_area"], align_px=int(tcfg["cutmix_align_px"])) for _ in range(2)]
        optimizer.zero_grad(set_to_none=True)
        lr = cosine_lr(update, updates, float(tcfg["lr"]), float(tcfg["warmup_fraction"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            terms = training_step(model, tcfg, update, labeled, ua, ub, boxes, fp_generator=fp_generator, seed=seed)
        terms["loss"].backward()
        if float(tcfg["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        optimizer.step()
        for k in sums:
            if k in terms:
                sums[k] += float(terms[k].detach())
        steps += 1
        done = update + 1
        if done % per_epoch == 0 or done == updates:
            epoch = math.ceil(done / per_epoch)
            val = native_metrics(predict(model, val_frames, dev))
            row = {"epoch": epoch, "update": done, "lr": lr, **{k: v / max(1, steps) for k, v in sums.items()},
                   "val_fg_dice": val["fg_dice"], "val_fg_hd95_mm": val["fg_hd95_mm"], "seconds": time.perf_counter() - tick}
            if val["fg_dice"] is not None and val["fg_dice"] > best:
                best, best_epoch, best_state = float(val["fg_dice"]), epoch, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            row["best_val_fg_dice"], row["best_epoch"] = best, best_epoch
            history.append(row)
            with (out / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"[train] ep {epoch}/{math.ceil(updates / per_epoch)} | loss {row['loss']:.4f} | val FG {val['fg_dice']:.4f} "
                  f"(best {best:.4f}@{best_epoch}) | {row['seconds']:.0f}s", flush=True)
            _save(last, {"update": done, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
                         "view_generator": view_gen.state_dict(), "fp_generator": fp_generator.get_state(), "best": best,
                         "best_epoch": best_epoch, "best_state": best_state, "history": history})
            sums, steps, tick = {k: 0.0 for k in sums}, 0, time.perf_counter()
    result = {"selected_epoch": best_epoch, "best_val_fg_dice": best, "updates": updates, "init": init_receipt}
    (out / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


__all__ = ["train", "training_step", "load_encoder", "cosine_lr", "lambda_u", "TRAIN_DEFAULTS"]
