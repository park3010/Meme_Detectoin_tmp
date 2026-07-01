from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run  # noqa: E402


def test_run_ours_full_normalized_cli_mapping():
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--label-set",
            "clean",
            "--no-normalized-labels",
            "--allow-missing-normalized-label",
            "--no-sample-weight",
        ]
    )
    cfg = run.OursRunConfig(
        dataset_name="harm_c",
        seed=42,
        config_path=args.config,
        label_set=args.label_set,
        use_normalized_labels=args.use_normalized_labels,
        require_normalized_label=args.require_normalized_label,
        use_sample_weight=args.use_sample_weight,
    )

    assert cfg.label_set == "clean"
    assert cfg.use_normalized_labels is False
    assert cfg.require_normalized_label is False
    assert cfg.use_sample_weight is False


def test_run_ours_full_normalized_cli_defaults_match_config():
    parser = run.build_parser()
    args = parser.parse_args(["train"])
    cfg = run.OursRunConfig(
        dataset_name="harm_c",
        seed=42,
        config_path=args.config,
        label_set=args.label_set,
        use_normalized_labels=args.use_normalized_labels,
        require_normalized_label=args.require_normalized_label,
        use_sample_weight=args.use_sample_weight,
    )

    assert cfg.label_set == "full"
    assert cfg.use_normalized_labels is True
    assert cfg.require_normalized_label is True
    assert cfg.use_sample_weight is True


def test_baseline_normalized_cli_mapping_and_legacy_default():
    parser = run.build_parser()
    default_args = parser.parse_args(["baseline"])
    default_cfg = run.BaselineRunConfig(
        model_name=default_args.baseline,
        dataset_name="harm_c",
        seed=42,
        config_path=default_args.config,
        label_set=default_args.label_set,
        use_normalized_labels=default_args.use_normalized_labels,
        require_normalized_label=default_args.require_normalized_label,
        use_sample_weight=default_args.use_sample_weight,
    )
    assert default_cfg.use_normalized_labels is False
    assert default_cfg.require_normalized_label is True
    assert default_cfg.use_sample_weight is False

    args = parser.parse_args(
        [
            "baseline",
            "--use-normalized-labels",
            "--label-set",
            "clean",
            "--allow-missing-normalized-label",
            "--use-sample-weight",
        ]
    )
    cfg = run.BaselineRunConfig(
        model_name=args.baseline,
        dataset_name="harm_c",
        seed=42,
        config_path=args.config,
        label_set=args.label_set,
        use_normalized_labels=args.use_normalized_labels,
        require_normalized_label=args.require_normalized_label,
        use_sample_weight=args.use_sample_weight,
    )
    assert cfg.use_normalized_labels is True
    assert cfg.label_set == "clean"
    assert cfg.require_normalized_label is False
    assert cfg.use_sample_weight is True
