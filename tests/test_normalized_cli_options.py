from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_baseline_text_only  # noqa: E402
import run_ours_full  # noqa: E402


def test_run_ours_full_normalized_cli_mapping():
    parser = run_ours_full.build_parser()
    args = parser.parse_args(
        [
            "--label-set",
            "clean",
            "--no-normalized-labels",
            "--allow-missing-normalized-label",
            "--no-sample-weight",
        ]
    )
    cfg = run_ours_full.config_from_args(args, dataset="harm_c", seed=42)

    assert cfg.label_set == "clean"
    assert cfg.use_normalized_labels is False
    assert cfg.require_normalized_label is False
    assert cfg.use_sample_weight is False


def test_run_ours_full_normalized_cli_defaults_match_config():
    parser = run_ours_full.build_parser()
    args = parser.parse_args([])
    cfg = run_ours_full.config_from_args(args, dataset="harm_c", seed=42)

    assert cfg.label_set == "full"
    assert cfg.use_normalized_labels is True
    assert cfg.require_normalized_label is True
    assert cfg.use_sample_weight is True


def test_baseline_normalized_cli_mapping_and_legacy_default():
    parser = run_baseline_text_only.build_parser()
    default_args = parser.parse_args([])
    default_cfg = run_baseline_text_only.config_from_args(default_args, "text_only_encoder", "harm_c", 42)
    assert default_cfg.use_normalized_labels is False
    assert default_cfg.require_normalized_label is True
    assert default_cfg.use_sample_weight is False

    args = parser.parse_args(
        [
            "--use-normalized-labels",
            "--label-set",
            "clean",
            "--allow-missing-normalized-label",
            "--use-sample-weight",
        ]
    )
    cfg = run_baseline_text_only.config_from_args(args, "text_only_encoder", "harm_c", 42)
    assert cfg.use_normalized_labels is True
    assert cfg.label_set == "clean"
    assert cfg.require_normalized_label is False
    assert cfg.use_sample_weight is True
