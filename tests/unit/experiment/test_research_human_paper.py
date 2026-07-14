from __future__ import annotations

import csv
from pathlib import Path

from experiments.paper_export import check_paper
from experiments.research_human_eval import agreement_report, validate_human_ratings


def test_human_rating_validation_and_agreement(tmp_path: Path):
    path = tmp_path / "ratings.csv"
    rows = [
        {"item_id": "A", "annotator_id": "r1", "faithfulness": "4", "usefulness": "5"},
        {"item_id": "A", "annotator_id": "r2", "faithfulness": "4", "usefulness": "4"},
        {"item_id": "B", "annotator_id": "r1", "faithfulness": "2", "usefulness": "3"},
        {"item_id": "B", "annotator_id": "r2", "faithfulness": "3", "usefulness": "3"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    schema = tmp_path / "schema.json"
    schema.write_text('{"required_columns":["item_id","annotator_id","faithfulness","usefulness"],"rating_scale":{"minimum":1,"maximum":5}}', encoding="utf-8")
    assert validate_human_ratings(path, schema)["passed"] is True
    report = agreement_report(path, ["faithfulness", "usefulness"])
    assert report["metrics"]["faithfulness"]["paired_item_count"] == 2


def test_paper_check_allows_draft_and_blocks_final_markers(tmp_path: Path):
    root = tmp_path / "latex"
    (root / "generated").mkdir(parents=True)
    (root / "tables" / "generated").mkdir(parents=True)
    (root / "main.tex").write_text("\\PaperDraftPlaceholder{pending}\n", encoding="utf-8")
    (root / "reference.bib").write_text("", encoding="utf-8")
    (root / "generated" / "result_macros.tex").write_text("", encoding="utf-8")
    (root / "generated" / "experiment_status.tex").write_text("Not run\n", encoding="utf-8")
    (root / "generated" / "generation_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "tables" / "generated" / "main_baseline.tex").write_text("Not run\n", encoding="utf-8")
    (root / "tables" / "generated" / "main_core_ablation.tex").write_text("Not run\n", encoding="utf-8")
    (root / "tables" / "generated" / "main_structured.tex").write_text("Not run\n", encoding="utf-8")
    assert check_paper(mode="draft", latex_root=str(root))["passed"] is True
    final = check_paper(mode="final", latex_root=str(root))
    assert final["passed"] is False
    assert final["draft_markers"]
