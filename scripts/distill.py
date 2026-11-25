#!/usr/bin/env python3
"""Stage S1: distil the CineMA SAX encoder into the SASNet encoder (training images only).

    python scripts/distill.py --config configs/acdc_10.json --out runs/acdc_10/distill
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sasnet.cinema_distill import distill  # noqa: E402
from sasnet.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-updates", type=int, default=None, help="smoke runs only")
    a = ap.parse_args()
    print(json.dumps(distill(load_config(a.config), a.out, device=a.device, max_updates=a.max_updates), indent=1))


if __name__ == "__main__":
    main()
