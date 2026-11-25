#!/usr/bin/env python3
"""Evaluate the validation-selected checkpoint on the test split, once, and export the deployable model.

    python scripts/evaluate.py --config configs/acdc_10.json --run runs/acdc_10/train

Writes ``<run>/TEST_RESULT.json`` (native mm metrics and SSL4MIS voxel/mm metrics for the selected and the last
checkpoint) and ``<run>/sasnet.pt``.  An existing TEST_RESULT.json is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import torch

    from sasnet.config import load_config
    from sasnet.data import SplitData
    from sasnet.metrics import evaluate, native_metrics, predict
    from sasnet.slicessm_student import build_model, export_model, load_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = a.run / "TEST_RESULT.json"
    if out.exists():
        print(f"{out} exists: the test split is evaluated once", flush=True)
        return 0
    cfg, device = load_config(a.config), torch.device(a.device)
    ck = torch.load(a.run / "last.pt", map_location="cpu", weights_only=False)
    data = SplitData(cfg["data"]["prepared"], cfg["data"]["split"])
    model = build_model(cfg.get("model"))
    model.load_state_dict(ck["best_state"], strict=True)
    model.to(device).eval()
    val = native_metrics(predict(model, data.eval_frames("val"), device))
    if abs(float(val["fg_dice"]) - float(ck["best"])) > 1e-4:
        raise SystemExit(f"validation recomputation {val['fg_dice']:.6f} differs from the logged best {ck['best']:.6f}")
    test_frames = data.eval_frames("test")
    selected = evaluate(predict(model, test_frames, device))
    model.load_state_dict(ck["model"], strict=True)
    last = evaluate(predict(model, test_frames, device))
    model.load_state_dict(ck["best_state"], strict=True)
    export = export_model(model.cpu(), a.run / "sasnet.pt", epoch=ck["best_epoch"])
    deployed = load_model(export["path"]).to(device)
    frame = test_frames[0]
    with torch.no_grad():
        diff = float((model.to(device).eval()(frame.image.to(device), frame.geometry)["logits"]
                      - deployed(frame.image.to(device), frame.geometry)["logits"]).abs().max())
    result = {"selected_epoch": ck["best_epoch"], "val_fg_dice": val["fg_dice"], "test": selected, "test_last_epoch": last,
              "export": export, "export_max_abs_logit_diff": diff}
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    n = selected["native"]
    s = selected["ssl4mis"]
    print(f"[test] epoch {ck['best_epoch']} | Dice {n['fg_dice']:.4f} | Jaccard {s['mean']['jaccard']:.4f} | HD95 {n['fg_hd95_mm']:.2f} mm "
          f"({s['mean']['hd95']:.2f} vox) | ASSD {n['fg_assd_mm']:.2f} mm (ASD {s['mean']['asd']:.2f} vox)", flush=True)
    if cfg["data"].get("single_class"):
        lv, sl = n["LV"], s["per_class"]["LV"]
        print(f"[test] single class (slot 3) | Dice {lv['dice']:.4f} | Jaccard {sl['jaccard']:.4f} | HD95 {lv['hd95_mm']:.2f} mm "
              f"({sl['hd95']:.2f} vox) | ASSD {lv['assd_mm']:.2f} mm (ASD {sl['asd']:.2f} vox)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
