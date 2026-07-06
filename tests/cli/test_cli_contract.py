from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run  # noqa: E402


def test_core_cli_commands_still_exist():
    parser = run.build_parser()
    help_text = parser.format_help()
    for command in ["train", "baseline", "stage", "assets", "evaluate", "ablation", "audit", "suite", "preflight"]:
        assert command in help_text


def test_progress_options_exist_on_relevant_commands():
    parser = run.build_parser()
    commands = ["train", "baseline", "stage", "evaluate", "ablation", "suite"]
    for command in commands:
        args = parser.parse_args([command, *_required_args(command), "--disable-tqdm", "--tqdm-mininterval", "0.25", "--tqdm-leave"])
        assert args.disable_tqdm is True
        assert args.tqdm_mininterval == 0.25
        assert args.tqdm_leave is True


def _required_args(command: str) -> list[str]:
    if command == "suite":
        return ["--suite", "core_smoke"]
    if command == "evaluate":
        return ["--dataset", "harm_c"]
    if command == "assets":
        return ["inspect"]
    return []
