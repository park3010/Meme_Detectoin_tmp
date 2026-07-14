from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run  # noqa: E402


def test_research_cli_supports_explicit_dry_run_and_resume():
    parser = run.build_parser()
    args = parser.parse_args(["research", "run", "--suite", "harmeme_to_fhm_smoke", "--dry-run"])
    assert args.dry_run is True
    assert args.resume is False
    resume = parser.parse_args(["research", "resume", "--suite", "harmeme_to_fhm_smoke", "--execute"])
    assert resume.execute is True
    assert resume.resume is True


def test_research_cli_supports_explicit_experiment_selection():
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "research",
            "run",
            "--suite",
            "harmeme_to_fhm_1seed",
            "--experiment",
            "ours_full",
            "ablation_w_o_retrieval",
            "--dry-run",
        ]
    )
    assert args.experiment == ["ours_full", "ablation_w_o_retrieval"]


def test_research_cli_exposes_safety_and_reporting_commands():
    parser = run.build_parser()
    help_text = parser.parse_args(["research", "plan", "--suite", "harmeme_to_fhm_smoke"])
    assert help_text.suite == "harmeme_to_fhm_smoke"
    assert "research" in parser.format_help()
