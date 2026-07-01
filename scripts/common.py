"""Shared CLI helpers for pipeline scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset", nargs="*", default=None, help="Dataset names: harm_c harm_p memotion facebook.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--no-save", action="store_true", help="Run without writing result files.")
    return parser


def print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))
