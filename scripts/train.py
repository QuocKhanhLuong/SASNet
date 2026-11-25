#!/usr/bin/env python3
"""Stage S0: semi-supervised training from the distilled encoder; the checkpoint is selected on the validation split.

    python scripts/train.py --config configs/acdc_10.json --init runs/acdc_10/distill/encoder_init.pt --out runs/acdc_10/train

Re-running the same command resumes from ``<out>/last.pt``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sasnet.config import load_config  # noqa: E402
from sasnet.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="encoder_init.pt from scripts/distill.py (omit to train from scratch)")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-updates", type=int, default=None, help="smoke runs only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    print(json.dumps(train(cfg, a.out, device=a.device, init=a.init, max_updates=a.max_updates), indent=1))


if __name__ == "__main__":
    main()
