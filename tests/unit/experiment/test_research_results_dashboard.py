from __future__ import annotations

import json
from pathlib import Path

from experiments.research_dashboard import build_research_dashboard
from experiments.research_results import aggregate_research_results


def test_research_aggregation_preserves_coverage_and_missing_status(tmp_path: Path):
    root = tmp_path / "result"
    run = root / "research_runs" / "harmeme_to_fhm_1seed" / "ours_full" / "seed_42"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({
            "experiment_id": "ours_full",
            "seed": 42,
            "completion_status": "complete",
            "source_split_manifest_sha256": "split",
            "config_sha256": "config",
            "code_sha256": "code",
        }),
        encoding="utf-8",
    )
    (run / "pipeline_audit_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({
            "harmfulness_macro_f1": 0.75,
            "target_presence_macro_f1": 0.5,
            "target_presence_valid_n": 80,
            "target_presence_total_n": 100,
            "target_presence_coverage": 0.8,
        }),
        encoding="utf-8",
    )
    (run / "runtime.json").write_text(json.dumps({"wall_seconds": 12.0}), encoding="utf-8")
    summary = aggregate_research_results(output_root=str(root))
    assert summary["metric_row_count"] == 2
    text = (root / "aggregates" / "all_results.csv").read_text(encoding="utf-8")
    assert "target_presence" in text and "0.8" in text
    status = (root / "aggregates" / "experiment_status.csv").read_text(encoding="utf-8")
    assert "blocked_api_credentials" in status

    dashboard = build_research_dashboard(output_root=str(root))
    assert dashboard.exists()
    assert "Missing results are shown as status" in dashboard.read_text(encoding="utf-8")
