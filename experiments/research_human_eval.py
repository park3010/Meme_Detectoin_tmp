"""Blinded human-evaluation export, import validation, and agreement tools."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from utils.io import read_jsonl, write_json


def export_human_evaluation(
    *,
    output_root: str = "result",
    suite: str = "harmeme_to_fhm_1seed",
    experiment_id: str = "ours_full",
    seed: int = 42,
    limit: int = 100,
    random_seed: int = 42,
    human_eval_root: str = "human_eval",
) -> dict[str, Any]:
    """Export randomized evidence and rationale rating sheets with blank judgments."""

    run_dir = Path(output_root) / "research_runs" / suite / experiment_id / f"seed_{seed}"
    predictions = read_jsonl(run_dir / "test_predictions.jsonl")
    rng = random.Random(random_seed)
    rng.shuffle(predictions)
    predictions = predictions[: max(0, limit)]
    root = Path(human_eval_root) / "export"
    root.mkdir(parents=True, exist_ok=True)
    evidence_rows = []
    rationale_rows = []
    key_rows = []
    for index, record in enumerate(predictions, start=1):
        item_id = f"HME-{index:04d}-{_short_hash(str(record.get('sample_key') or record.get('sample_id')))}"
        common = {
            "item_id": item_id,
            "image_path": record.get("image_path", ""),
            "ocr_text": record.get("ocr_text", record.get("ocr_text_full", "")),
            "predicted_harmfulness": record.get("pred_label", record.get("harmfulness", {}).get("label")),
        }
        supporting = record.get("supporting_evidence", {}) or {}
        evidence_rows.append(
            {
                **common,
                "selected_internal_evidence": json.dumps(supporting.get("internal", []), ensure_ascii=False),
                "selected_external_evidence": json.dumps(supporting.get("external", []), ensure_ascii=False),
                "annotator_id": "",
                "relevance": "",
                "sufficiency": "",
                "notes": "",
            }
        )
        rationale_rows.append(
            {
                **common,
                "predicted_target": json.dumps(record.get("target", {}), ensure_ascii=False),
                "predicted_intent": json.dumps(record.get("intent", {}), ensure_ascii=False),
                "predicted_tactic": json.dumps(record.get("tactic", {}), ensure_ascii=False),
                "selected_evidence": json.dumps(supporting, ensure_ascii=False),
                "rationale": record.get("rationale", ""),
                "annotator_id": "",
                "supports_prediction": "",
                "grounded_in_selected_evidence": "",
                "understandable": "",
                "unsupported_claim": "",
                "notes": "",
            }
        )
        key_rows.append({"item_id": item_id, "sample_key": record.get("sample_key"), "sample_id": record.get("sample_id")})
    evidence_path = root / "evidence_human_eval_template.csv"
    rationale_path = root / "rationale_human_eval_template.csv"
    _write_csv(evidence_path, evidence_rows)
    _write_csv(rationale_path, rationale_rows)
    _write_csv(root / "blinding_key.csv", key_rows)
    verifier_rows = _verifier_rows(predictions, key_rows)
    _write_csv(root / "verifier_human_eval_template.csv", verifier_rows)
    manifest = {
        "schema_version": "human_evaluation_export_v1",
        "source_run": str(run_dir),
        "source_prediction_count": len(predictions),
        "random_seed": random_seed,
        "contains_human_judgments": False,
        "evidence_template": str(evidence_path),
        "rationale_template": str(rationale_path),
        "verifier_template": str(root / "verifier_human_eval_template.csv"),
    }
    write_json(root / "export_manifest.json", manifest)
    return manifest


def validate_human_ratings(path: str | Path, schema_path: str | Path) -> dict[str, Any]:
    """Validate columns, rating ranges, item uniqueness, and annotator IDs."""

    rows = _read_csv(Path(path))
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    required = list(schema.get("required_columns", []))
    minimum = int(schema.get("rating_scale", {}).get("minimum", 1))
    maximum = int(schema.get("rating_scale", {}).get("maximum", 5))
    field_specs = schema.get("fields", {}) or {}
    errors: list[str] = []
    columns = set(rows[0]) if rows else set()
    missing_columns = sorted(set(required) - columns)
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
    keys = [(row.get("item_id", ""), row.get("annotator_id", "")) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if key != ("", "") and count > 1]
    if duplicates:
        errors.append(f"duplicate item/annotator rows: {duplicates[:10]}")
    rating_columns = [column for column in required if column not in {"item_id", "candidate_id", "annotator_id"}]
    completed = 0
    for line, row in enumerate(rows, start=2):
        if not row.get("item_id"):
            errors.append(f"line {line}: item_id is blank")
        values_present = False
        for column in rating_columns:
            raw = str(row.get(column, "")).strip()
            if not raw:
                continue
            values_present = True
            allowed = {str(value) for value in (field_specs.get(column, {}) or {}).get("allowed_values", [])}
            if allowed:
                if raw not in allowed:
                    errors.append(f"line {line}: {column}={raw!r} not in {sorted(allowed)}")
            else:
                try:
                    rating = int(raw)
                except ValueError:
                    errors.append(f"line {line}: {column} is not an integer")
                    continue
                if not minimum <= rating <= maximum:
                    errors.append(f"line {line}: {column}={rating} outside [{minimum}, {maximum}]")
        if values_present and not row.get("annotator_id"):
            errors.append(f"line {line}: annotator_id required when ratings are present")
        completed += int(values_present)
    return {
        "passed": not errors,
        "row_count": len(rows),
        "completed_row_count": completed,
        "duplicate_count": len(duplicates),
        "errors": errors,
    }


def import_human_ratings(
    path: str | Path,
    schema_path: str | Path,
    *,
    human_eval_root: str = "human_eval",
) -> dict[str, Any]:
    """Validate and copy supplied ratings into the controlled import area."""

    validation = validate_human_ratings(path, schema_path)
    root = Path(human_eval_root)
    report_path = root / "reports" / (Path(path).stem + "_validation.json")
    write_json(report_path, validation)
    imported_path = None
    if validation["passed"]:
        imported_path = root / "import" / Path(path).name
        imported_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, imported_path)
    return {**validation, "imported_path": str(imported_path) if imported_path else None, "report_path": str(report_path)}


def agreement_report(path: str | Path, rating_columns: list[str]) -> dict[str, Any]:
    """Compute pairwise exact agreement and quadratic weighted kappa."""

    rows = _read_csv(Path(path))
    by_item: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("item_id") and row.get("annotator_id"):
            by_item.setdefault(row["item_id"], []).append(row)
    metrics = {}
    for column in rating_columns:
        pairs = []
        for item_rows in by_item.values():
            if len(item_rows) < 2:
                continue
            values = [row.get(column, "") for row in item_rows if str(row.get(column, "")).strip()]
            if len(values) >= 2:
                pairs.append((int(values[0]), int(values[1])))
        metrics[column] = {
            "paired_item_count": len(pairs),
            "exact_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
            "quadratic_weighted_kappa": _weighted_kappa(pairs) if pairs else None,
        }
    return {"schema_version": "human_agreement_report_v1", "metrics": metrics}


def _weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    labels = sorted({value for pair in pairs for value in pair})
    if len(labels) < 2:
        return None
    index = {label: idx for idx, label in enumerate(labels)}
    size = len(labels)
    observed = [[0.0] * size for _ in range(size)]
    left = [0.0] * size
    right = [0.0] * size
    for a, b in pairs:
        observed[index[a]][index[b]] += 1
        left[index[a]] += 1
        right[index[b]] += 1
    total = float(len(pairs))
    observed_weight = expected_weight = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) / max(1, size - 1)) ** 2
            observed_weight += weight * observed[i][j] / total
            expected_weight += weight * (left[i] * right[j]) / (total * total)
    return None if expected_weight == 0 else 1.0 - observed_weight / expected_weight


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _verifier_rows(predictions: list[dict[str, Any]], key_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    item_ids = {str(row.get("sample_id")): row.get("item_id") for row in key_rows}
    rows = []
    for prediction in predictions:
        item_id = item_ids.get(str(prediction.get("sample_id")))
        external = (prediction.get("supporting_evidence", {}) or {}).get("external", []) or []
        for index, candidate in enumerate(external):
            candidate = candidate if isinstance(candidate, dict) else {"text": str(candidate)}
            rows.append(
                {
                    "item_id": item_id,
                    "candidate_id": candidate.get("evidence_id", candidate.get("candidate_id", f"external_{index}")),
                    "candidate_text": candidate.get("text", ""),
                    "stage_c_decision": candidate.get("verification_status", ""),
                    "annotator_id": "",
                    "relevance": "",
                    "validity": "",
                    "target_support": "",
                    "intent_support": "",
                    "tactic_support": "",
                    "stage_c_agreement": "",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


__all__ = ["agreement_report", "export_human_evaluation", "import_human_ratings", "validate_human_ratings"]
