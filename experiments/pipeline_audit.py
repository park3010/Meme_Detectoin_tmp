"""Audit full-pipeline experiment artifacts for contract and training readiness."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from experiments.ablation_configs import component_state_for_ablation, get_ablation_contract
from utils.io import read_jsonl


EXPECTED_LOGITS_LOSSES = {
    "harmfulness",
    "target_granularity",
    "target_presence",
    "intent_primary",
    "tactic_rhetorical",
    "tactic_multimodal_relation",
}

EXPECTED_TRAINABLE_FIELDS = {
    "harmfulness.label",
    "target.granularity",
    "target.presence",
    "intent.primary",
    "tactic.rhetorical",
    "tactic.multimodal_relation",
}

INTERNAL_EVIDENCE_FIELDS = {
    "source_stage",
    "modality",
    "grounding_type",
    "is_heuristic",
    "attribution_backend",
}

EXTERNAL_EVIDENCE_FIELDS = {
    "candidate_origin",
    "is_external_knowledge",
    "is_generated",
    "is_fallback",
    "is_retrieved",
    "verification_status",
    "attribution_backend",
}


def audit_run_artifacts(
    run_root: str | Path,
    *,
    training_log: str | Path | None = None,
    predictions: str | Path | None = None,
    metrics: str | Path | None = None,
    require_nonempty_metrics: bool = False,
    allow_empty_split: bool = False,
    strict: bool = False,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Discover and audit one full-framework experiment run."""

    root = Path(run_root)
    paths = discover_artifacts(root, training_log=training_log, predictions=predictions, metrics=metrics)
    result: dict[str, Any] = {
        "run_root": str(root),
        "strict": strict,
        "artifacts": {name: str(path) if path else None for name, path in paths.items()},
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    manifest = _load_manifest(paths["manifest"], result)
    result["manifest"] = manifest
    require_training_log = _requires_training_log(manifest)
    training_rows = _load_records(paths["training_log"], result, "training log", strict and require_training_log)
    prediction_rows = _load_records(paths["predictions"], result, "predictions", strict)
    metrics_obj = _load_object(paths["metrics"], result, "metrics", strict)
    tactic_decoding_obj = _load_object(paths["tactic_decoding"], result, "formal tactic decoding", False) if paths["tactic_decoding"] else {}

    expected_losses = set(manifest.get("expected_active_logits_losses") or EXPECTED_LOGITS_LOSSES)
    disabled_losses = set(manifest.get("expected_disabled_losses") or [])
    result["training_log"] = audit_training_log(
        training_rows,
        result,
        strict=strict,
        expected_logits_losses=expected_losses,
        expected_disabled_losses=disabled_losses,
        require_training_log=require_training_log,
    )
    result["predictions"] = audit_predictions(
        prediction_rows,
        result,
        strict=strict,
        sample_limit=sample_limit,
    )
    result["metrics"] = audit_metrics(
        metrics_obj,
        training_rows,
        prediction_rows,
        result,
        require_nonempty_metrics=require_nonempty_metrics,
        allow_empty_split=allow_empty_split,
    )
    result["formal_tactic_decoding"] = audit_formal_tactic_decoding(
        tactic_decoding_obj,
        metrics_obj,
        prediction_rows,
        result,
        strict=strict,
        allow_empty_split=allow_empty_split,
        required=_requires_formal_tactic_decoding(manifest),
    )
    result["ablation_contract"] = audit_ablation_contract(
        manifest,
        training_rows,
        prediction_rows,
        result,
        strict=strict,
    )
    result["passed"] = not result["errors"]
    result["status"] = "pass" if result["passed"] and not result["warnings"] else "warning" if result["passed"] else "fail"
    return result


def discover_artifacts(
    run_root: str | Path,
    *,
    training_log: str | Path | None = None,
    predictions: str | Path | None = None,
    metrics: str | Path | None = None,
) -> dict[str, Path | None]:
    """Resolve explicit artifact paths or common filenames below a run root."""

    root = Path(run_root)
    return {
        "training_log": _resolve_artifact(root, training_log, ["training_log.json", "training_log.jsonl"]),
        "predictions": _resolve_artifact(
            root,
            predictions,
            ["final_predictions.jsonl", "predictions.jsonl", "final_predictions.json", "predictions.json"],
        ),
        "metrics": _resolve_artifact(root, metrics, ["metrics.json", "final_metrics.json", "metrics.jsonl"]),
        "manifest": _resolve_artifact(root, None, ["run_manifest.json"]),
        "tactic_decoding": _resolve_artifact(root, None, ["tactic_rhetorical_decoding.json"]),
    }


def audit_training_log(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    strict: bool,
    expected_logits_losses: set[str] | None = None,
    expected_disabled_losses: set[str] | None = None,
    require_training_log: bool = True,
) -> dict[str, Any]:
    """Audit epoch logs, loss provenance, active losses, and gradient status."""

    expected_logits_losses = expected_logits_losses or set(EXPECTED_LOGITS_LOSSES)
    expected_disabled_losses = expected_disabled_losses or set()
    if not rows:
        _issue(result, "Training log is absent or empty.", strict=strict, critical=require_training_log)
        return {
            "epoch_count": 0,
            "expected_logits_losses_found": [],
            "training_log_required": require_training_log,
        }

    active_logits = {
        str(item)
        for row in rows
        for value in _find_values(row, "active_logits_losses")
        for item in _as_list(value)
    }
    active_proxy = {
        str(item)
        for row in rows
        for value in _find_values(row, "active_proxy_losses")
        for item in _as_list(value)
    }
    provenance_by_loss: dict[str, dict[str, Any]] = {}
    for row in rows:
        for value in _find_values(row, "loss_provenance"):
            if not isinstance(value, dict):
                continue
            for name, details in value.items():
                if isinstance(details, dict):
                    provenance_by_loss[str(name)] = details

    missing_losses = sorted(expected_logits_losses - active_logits)
    if missing_losses:
        _issue(
            result,
            f"Expected active logits losses are missing: {', '.join(missing_losses)}.",
            strict=strict,
            critical=True,
        )

    aux_checks: dict[str, Any] = {}
    aux_checks: dict[str, Any] = {}
    for name in ["target_presence", "tactic_multimodal_relation"]:
        if name in expected_disabled_losses:
            aux_checks[name] = {"disabled_by_contract": True}
            continue
        details = provenance_by_loss.get(name, {})
        provenance = str(details.get("provenance", ""))
        mean_requires_grad = _number(details.get("mean_requires_grad"))
        provenance_ok = "logits_aux" in provenance
        gradient_ok = mean_requires_grad == 1.0
        aux_checks[name] = {
            "provenance": provenance or None,
            "mean_requires_grad": mean_requires_grad,
            "provenance_ok": provenance_ok,
            "gradient_ok": gradient_ok,
        }
        if not provenance_ok:
            _issue(
                result,
                f"{name} loss provenance is not logits_aux based.",
                strict=strict,
                critical=True,
            )
        if not gradient_ok:
            _issue(
                result,
                f"{name} mean_requires_grad is not 1.0.",
                strict=strict,
                critical=True,
            )

    latest = rows[-1]
    required_log_fields = {
        "loss_components",
        "loss_provenance",
        "active_logits_losses",
        "active_proxy_losses",
        "active_logits_loss_count",
        "active_proxy_loss_count",
    }
    missing_log_fields = sorted(field for field in required_log_fields if not _find_values(latest, field))
    if missing_log_fields:
        _issue(
            result,
            f"Latest training epoch is missing audit fields: {', '.join(missing_log_fields)}.",
            strict=strict,
            critical=True,
        )

    split_sizes = _first_dict(_find_values(latest, "split_sizes"))
    return {
        "epoch_count": len(rows),
        "latest_epoch": latest.get("epoch"),
        "active_logits_losses": sorted(active_logits),
        "active_proxy_losses": sorted(active_proxy),
        "expected_logits_losses": sorted(expected_logits_losses),
        "expected_disabled_losses": sorted(expected_disabled_losses),
        "expected_logits_losses_found": sorted(expected_logits_losses & active_logits),
        "missing_expected_logits_losses": missing_losses,
        "auxiliary_loss_checks": aux_checks,
        "split_sizes": split_sizes,
        "training_log_required": require_training_log,
    }


def audit_ablation_contract(
    manifest: dict[str, Any],
    training_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    """Validate ablation-specific semantic expectations from the run manifest."""

    contract_obj = manifest.get("ablation_contract") if isinstance(manifest, dict) else None
    if not isinstance(contract_obj, dict):
        return {"ablation_contract_present": False, "ablation_contract_passed": True}

    name = str(contract_obj.get("name", ""))
    try:
        expected_contract = get_ablation_contract(name)
    except ValueError:
        _issue(result, f"Manifest references unknown ablation contract: {name}.", strict=strict, critical=True)
        return {"ablation_contract_present": True, "ablation_contract_passed": False}

    failures: list[str] = []
    component_state = manifest.get("component_state", {})
    expected_state = component_state if isinstance(component_state, dict) else {}
    canonical_state = _as_bool_dict(component_state_for_ablation(name))
    for key, expected in canonical_state.items():
        if expected_state.get(key) is not expected:
            failures.append(f"component_state.{key} expected {expected}")

    latest = training_rows[-1] if training_rows else {}
    active_logits = set(str(item) for value in _find_values(latest, "active_logits_losses") for item in _as_list(value))
    expected_losses = set(manifest.get("expected_active_logits_losses") or expected_contract.expected_active_logits_losses)
    if training_rows and active_logits and active_logits != expected_losses:
        failures.append(f"active logits losses {sorted(active_logits)} != expected {sorted(expected_losses)}")

    if name == "w_o_retrieval":
        for row in prediction_rows[:5]:
            stage_d = ((row.get("stage_metadata") or {}).get("stage_d") or {})
            if _integer(stage_d.get("verified_knowledge_count")) not in {0, None}:
                failures.append("w_o_retrieval has nonzero Stage D verified_knowledge_count")
                break
    if name == "w_o_context_generation":
        for row in prediction_rows[:5]:
            stage_b = ((row.get("stage_metadata") or {}).get("stage_b") or {})
            if _integer(stage_b.get("generated_count")) not in {0, None}:
                failures.append("w_o_context_generation has nonzero Stage B generated_count")
                break
    if name == "w_o_task_aware_gate":
        for row in prediction_rows[:5]:
            stage_d = ((row.get("stage_metadata") or {}).get("stage_d") or {})
            if "shared_gate" not in str(stage_d.get("gate_mode", "")):
                failures.append("w_o_task_aware_gate did not report shared gate mode")
                break
    if name == "w_o_structured_auxiliary":
        if sorted(expected_losses) != ["harmfulness", "intent_primary", "tactic_rhetorical", "target_granularity"]:
            failures.append("w_o_structured_auxiliary expected active losses are not the four primary logits losses")
        disabled = set(manifest.get("expected_disabled_losses") or [])
        if {"target_presence", "tactic_multimodal_relation"} - disabled:
            failures.append("w_o_structured_auxiliary disabled auxiliary losses are missing")

    for failure in failures:
        _issue(result, failure, strict=strict, critical=True)
    return {
        "ablation_contract_present": True,
        "name": name,
        "ablation_contract_passed": not failures,
        "failures": failures,
        "component_state": component_state,
    }


def audit_predictions(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    strict: bool,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Audit serialized Stage E payloads and evidence attribution records."""

    if not rows:
        _issue(result, "Prediction artifact is absent or contains zero records.", strict=strict, critical=True)
        return {"record_count": 0, "audited_count": 0}

    audited = rows[: max(1, sample_limit)]
    failures: list[dict[str, Any]] = []
    external_count = 0
    internal_count = 0
    stage_d_trace_count = 0

    for index, record in enumerate(audited):
        structured = _structured_view(record)
        sample_id = structured.get("sample_id") or record.get("sample_id") or f"record_{index}"
        missing: list[str] = []

        provenance = structured.get("output_provenance", {})
        field_provenance = provenance.get("field_provenance", {}) if isinstance(provenance, dict) else {}
        for key in ["field_provenance", "trainable_logits_fields", "proxy_fields", "template_fields", "cue_fields", "label_spaces"]:
            if not isinstance(provenance, dict) or key not in provenance:
                missing.append(f"output_provenance.{key}")
        if field_provenance.get("target.presence") != "logits_aux":
            missing.append("field_provenance.target.presence=logits_aux")
        if field_provenance.get("tactic.multimodal_relation") != "logits_aux":
            missing.append("field_provenance.tactic.multimodal_relation=logits_aux")
        if field_provenance.get("rationale") != "template":
            missing.append("field_provenance.rationale=template")

        trainable_fields = set(_as_list(provenance.get("trainable_logits_fields") if isinstance(provenance, dict) else []))
        if not EXPECTED_TRAINABLE_FIELDS <= trainable_fields:
            missing.append("output_provenance.trainable_logits_fields.expected_six")

        target = structured.get("target", {}) or {}
        _require_keys(
            target,
            {
                "presence",
                "presence_scores",
                "presence_logits",
                "presence_source",
                "presence_provenance",
                "heuristic_presence",
                "heuristic_presence_score",
            },
            "target",
            missing,
        )
        if target.get("presence_source") != "target_presence_head":
            missing.append("target.presence_source=target_presence_head")
        if target.get("presence_provenance") != "logits_aux":
            missing.append("target.presence_provenance=logits_aux")

        tactic = structured.get("tactic", {}) or {}
        _require_keys(
            tactic,
            {
                "multimodal_relation",
                "multimodal_relation_scores",
                "multimodal_relation_logits",
                "multimodal_relation_source",
                "multimodal_relation_provenance",
                "stage_a_multimodal_relation",
                "rhetorical_primary",
                "rhetorical_labels",
                "rhetorical_decoding",
                "heuristic_rhetorical_cues",
            },
            "tactic",
            missing,
        )
        if tactic.get("multimodal_relation_source") != "tactic_multimodal_relation_head":
            missing.append("tactic.multimodal_relation_source=tactic_multimodal_relation_head")
        if tactic.get("multimodal_relation_provenance") != "logits_aux":
            missing.append("tactic.multimodal_relation_provenance=logits_aux")

        hooks = structured.get("training_hooks", {}) or {}
        _require_keys(
            hooks,
            {
                "target_presence_logits",
                "target_presence_scores",
                "tactic_multimodal_relation_logits",
                "tactic_multimodal_relation_scores",
                "field_provenance",
                "trainable_logits_fields",
                "proxy_fields",
            },
            "training_hooks",
            missing,
        )

        if _stage_d_trace_available(record, structured):
            stage_d_trace_count += 1
        else:
            missing.append("stage_d_trace_available")

        evidence = structured.get("supporting_evidence", {}) or {}
        internal = evidence.get("internal", []) if isinstance(evidence, dict) else []
        external = evidence.get("external", []) if isinstance(evidence, dict) else []
        internal_count += len(internal)
        external_count += len(external)
        for item_index, item in enumerate(internal):
            absent = sorted(INTERNAL_EVIDENCE_FIELDS - set(item))
            if absent:
                missing.append(f"supporting_evidence.internal[{item_index}].{','.join(absent)}")
        for item_index, item in enumerate(external):
            absent = sorted(EXTERNAL_EVIDENCE_FIELDS - set(item))
            if absent:
                missing.append(f"supporting_evidence.external[{item_index}].{','.join(absent)}")

        if missing:
            failures.append({"sample_id": sample_id, "missing": sorted(set(missing))})

    if failures:
        preview = "; ".join(f"{item['sample_id']}: {', '.join(item['missing'][:4])}" for item in failures[:3])
        _issue(
            result,
            f"{len(failures)}/{len(audited)} audited prediction records violate the Stage E artifact contract. {preview}",
            strict=strict,
            critical=True,
        )
    if internal_count == 0:
        _issue(result, "No internal evidence records were present in audited predictions.", strict=False, critical=False)
    if external_count == 0:
        _issue(result, "No external evidence records were present in audited predictions.", strict=False, critical=False)

    return {
        "record_count": len(rows),
        "audited_count": len(audited),
        "contract_pass_count": len(audited) - len(failures),
        "contract_failures": failures,
        "internal_evidence_count": internal_count,
        "external_evidence_count": external_count,
        "stage_d_trace_available_count": stage_d_trace_count,
    }


def audit_metrics(
    metrics: dict[str, Any],
    training_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    require_nonempty_metrics: bool,
    allow_empty_split: bool,
) -> dict[str, Any]:
    """Audit final and validation metric usability with empty-split handling."""

    split_sizes = _latest_split_sizes(training_rows)
    test_size = _integer(split_sizes.get("test")) if split_sizes else None
    valid_size = _integer(split_sizes.get("valid")) if split_sizes else None
    if test_size is None and not prediction_rows:
        test_size = 0

    metric_values = {
        key: value
        for key, value in metrics.items()
        if _is_metric_key(key) and _number(value) is not None
    }
    usable = bool(metric_values)
    empty_test_detected = test_size == 0
    empty_validation_detected = valid_size == 0 or _validation_looks_empty(training_rows)
    empty_split_detected = empty_test_detected or empty_validation_detected

    if not metrics:
        _issue(result, "Metrics artifact is absent or empty.", strict=False, critical=False)
    if require_nonempty_metrics and not usable:
        if allow_empty_split and empty_split_detected:
            _issue(result, "Metrics are empty because an empty validation/test split was detected.", strict=False, critical=False)
        else:
            result["errors"].append("Non-empty metrics were required, but no usable metric values were found.")
    elif not usable:
        _issue(
            result,
            "Final metrics contain no usable non-NaN values.",
            strict=False,
            critical=False,
        )

    return {
        "metric_file_found": bool(metrics),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "structured_metrics": {
            key: value
            for key, value in metrics.items()
            if any(token in key for token in ("target_", "intent_", "tactic_", "evidence_"))
        },
        "usable_metric_count": len(metric_values),
        "metrics_usable": usable,
        "split_sizes": split_sizes,
        "empty_test_split_detected": empty_test_detected,
        "empty_validation_split_detected": empty_validation_detected,
        "empty_split_allowed": allow_empty_split,
    }


def audit_formal_tactic_decoding(
    decoding: dict[str, Any],
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    strict: bool,
    allow_empty_split: bool,
    required: bool,
) -> dict[str, Any]:
    """Audit formal logits-only rhetorical tactic decoding artifacts."""

    if not decoding:
        if required:
            _issue(
                result,
                "Formal tactic decoding artifact tactic_rhetorical_decoding.json is missing.",
                strict=strict,
                critical=True,
            )
        return {"required": required, "artifact_found": False, "passed": not required}

    failures: list[str] = []
    if decoding.get("schema_version") != "tactic_rhetorical_decoding_v1":
        failures.append("schema_version=tactic_rhetorical_decoding_v1")
    if decoding.get("prediction_source") != "tactic_logits_sigmoid":
        failures.append("prediction_source=tactic_logits_sigmoid")
    if decoding.get("rendered_labels_used") is not False:
        failures.append("rendered_labels_used=false")
    if decoding.get("threshold_policy") != "validation_grid_search":
        failures.append("threshold_policy=validation_grid_search")
    if decoding.get("test_evaluation_policy") != "fixed_validation_threshold":
        failures.append("test_evaluation_policy=fixed_validation_threshold")
    selected = _number(decoding.get("selected_threshold"))
    candidates = [_number(item) for item in _as_list(decoding.get("threshold_candidates"))]
    candidates = [item for item in candidates if item is not None]
    if selected is None:
        failures.append("selected_threshold_present")
    elif candidates and not any(abs(selected - item) <= 1e-8 for item in candidates):
        failures.append("selected_threshold_in_candidates")

    formal_status = metrics.get("tactic_rhetorical_formal_status")
    eligible = _integer(metrics.get("tactic_rhetorical_eligible_sample_count"))
    if formal_status not in {"ready", "no_eligible_samples"}:
        if not allow_empty_split:
            failures.append("metrics.tactic_rhetorical_formal_status_ready")
    if formal_status == "no_eligible_samples" and not allow_empty_split:
        failures.append("metrics.tactic_rhetorical_has_eligible_samples")
    if eligible == 0 and not allow_empty_split:
        failures.append("metrics.tactic_rhetorical_eligible_sample_count_nonzero")

    trace_failures = []
    for row in prediction_rows[:5]:
        trace = ((row.get("evaluation") or {}).get("tactic_rhetorical_formal") or {})
        if not isinstance(trace, dict) or not trace:
            trace_failures.append(f"{row.get('sample_id', '<unknown>')}: missing formal trace")
            continue
        trace_threshold = _number(trace.get("threshold"))
        if trace.get("prediction_source") != "tactic_logits_sigmoid":
            trace_failures.append(f"{row.get('sample_id', '<unknown>')}: trace source")
        if trace.get("rendered_labels_used") is not False:
            trace_failures.append(f"{row.get('sample_id', '<unknown>')}: trace rendered_labels_used")
        if selected is not None and trace_threshold is not None and abs(trace_threshold - selected) > 1e-8:
            trace_failures.append(f"{row.get('sample_id', '<unknown>')}: trace threshold differs")
    if trace_failures:
        failures.extend(trace_failures)

    for failure in failures:
        _issue(result, f"Formal tactic decoding audit failed: {failure}.", strict=strict, critical=True)
    return {
        "required": required,
        "artifact_found": True,
        "passed": not failures,
        "prediction_source": decoding.get("prediction_source"),
        "rendered_labels_used": decoding.get("rendered_labels_used"),
        "threshold_policy": decoding.get("threshold_policy"),
        "selected_threshold": selected,
        "threshold_candidate_count": len(candidates),
        "test_evaluation_policy": decoding.get("test_evaluation_policy"),
        "formal_metric_status": formal_status,
        "eligible_sample_count": eligible,
        "trace_failures": trace_failures,
        "failures": failures,
    }


def write_audit_report(result: dict[str, Any], path: str | Path) -> Path:
    """Write a concise Markdown report for one audited run."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    training = result.get("training_log", {})
    predictions = result.get("predictions", {})
    metrics = result.get("metrics", {})
    formal = result.get("formal_tactic_decoding", {})
    artifacts = result.get("artifacts", {})
    aux = training.get("auxiliary_loss_checks", {})
    manifest = result.get("manifest", {}) or {}
    contract = result.get("ablation_contract", {}) or {}

    lines = [
        "# Pipeline Audit Report",
        "",
        "## Run path",
        f"`{result.get('run_root', '')}`",
        "",
        "## Artifact discovery",
        *[f"- {name}: `{value}`" if value else f"- {name}: missing" for name, value in artifacts.items()],
        "",
        "## Run manifest",
        f"- Schema: `{manifest.get('schema')}`",
        f"- Run kind: `{manifest.get('run_kind')}`",
        f"- Run name: `{manifest.get('run_name')}`",
        f"- Expected knowledge mode: `{manifest.get('expected_knowledge_mode')}`",
        f"- Expected evidence mode: `{manifest.get('expected_evidence_mode')}`",
        "",
        "## Ablation contract",
        f"- Present: `{contract.get('ablation_contract_present', False)}`",
        f"- Passed: `{contract.get('ablation_contract_passed', True)}`",
        f"- Name: `{contract.get('name')}`",
        "",
        "## Training log audit",
        f"- Epochs: {training.get('epoch_count', 0)}",
        f"- Active logits losses: {_display_list(training.get('active_logits_losses', []))}",
        f"- Missing expected logits losses: {_display_list(training.get('missing_expected_logits_losses', []))}",
        f"- Split sizes: `{training.get('split_sizes', {})}`",
        "",
        "## Loss provenance summary",
        f"- target_presence: `{aux.get('target_presence', {})}`",
        f"- tactic_multimodal_relation: `{aux.get('tactic_multimodal_relation', {})}`",
        "",
        "## Prediction JSON audit",
        f"- Records: {predictions.get('record_count', 0)}",
        f"- Audited: {predictions.get('audited_count', 0)}",
        f"- Contract passes: {predictions.get('contract_pass_count', 0)}",
        "",
        "## Stage E output provenance",
        f"- Stage D trace available: {predictions.get('stage_d_trace_available_count', 0)}/{predictions.get('audited_count', 0)} audited records",
        "",
        "## Evidence attribution provenance",
        f"- Internal evidence records: {predictions.get('internal_evidence_count', 0)}",
        f"- External evidence records: {predictions.get('external_evidence_count', 0)}",
        "",
        "## Metrics readiness",
        f"- Metrics usable: {metrics.get('metrics_usable', False)}",
        f"- Accuracy: {metrics.get('accuracy')}",
        f"- Macro-F1: {metrics.get('macro_f1')}",
        f"- Empty split detected: {metrics.get('empty_test_split_detected') or metrics.get('empty_validation_split_detected')}",
        "",
        "## Formal tactic decoding",
        f"- Required: {formal.get('required', False)}",
        f"- Artifact found: {formal.get('artifact_found', False)}",
        f"- Passed: {formal.get('passed', False)}",
        f"- Source: `{formal.get('prediction_source')}`",
        f"- Selected threshold: `{formal.get('selected_threshold')}`",
        f"- Formal metric status: `{formal.get('formal_metric_status')}`",
        "",
        "## Warnings",
        *([f"- {warning}" for warning in result.get("warnings", [])] or ["- None"]),
        "",
        "## Pass/fail summary",
        f"**{str(result.get('status', 'unknown')).upper()}**",
    ]
    if result.get("errors"):
        lines.extend(["", "Errors:", *[f"- {error}" for error in result["errors"]]])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def format_audit_summary(result: dict[str, Any]) -> str:
    """Return a compact terminal summary."""

    training = result.get("training_log", {})
    predictions = result.get("predictions", {})
    metrics = result.get("metrics", {})
    formal = result.get("formal_tactic_decoding", {})
    expected_count = len(training.get("expected_logits_losses", EXPECTED_LOGITS_LOSSES))
    lines = [
        f"Pipeline audit: {str(result.get('status', 'unknown')).upper()}",
        f"Run root: {result.get('run_root')}",
        f"Training epochs: {training.get('epoch_count', 0)}",
        f"Expected logits losses: {len(training.get('expected_logits_losses_found', []))}/{expected_count}",
        f"Prediction contract: {predictions.get('contract_pass_count', 0)}/{predictions.get('audited_count', 0)}",
        f"Metrics usable: {metrics.get('metrics_usable', False)}",
        f"Formal tactic decoding: found={formal.get('artifact_found', False)} passed={formal.get('passed', False)}",
    ]
    contract = result.get("ablation_contract", {})
    if contract.get("ablation_contract_present"):
        lines.append(f"Ablation contract: {contract.get('name')} passed={contract.get('ablation_contract_passed')}")
    if result.get("warnings"):
        lines.append(f"Warnings: {len(result['warnings'])}")
    if result.get("errors"):
        lines.append(f"Errors: {len(result['errors'])}")
    return "\n".join(lines)


def _resolve_artifact(root: Path, explicit: str | Path | None, names: list[str]) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = sorted(root.glob(f"**/{name}")) if root.exists() else []
        if matches:
            return matches[0]
    return None


def _load_records(
    path: Path | None,
    result: dict[str, Any],
    label: str,
    strict: bool,
) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        _issue(result, f"{label.capitalize()} artifact was not found.", strict=strict, critical=True)
        return []
    try:
        if path.suffix == ".jsonl":
            return read_jsonl(path)
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _issue(result, f"Could not read {label} artifact {path}: {exc}", strict=strict, critical=True)
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    _issue(result, f"{label.capitalize()} artifact has unsupported top-level type.", strict=strict, critical=True)
    return []


def _load_object(
    path: Path | None,
    result: dict[str, Any],
    label: str,
    strict: bool,
) -> dict[str, Any]:
    rows = _load_records(path, result, label, strict)
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    return {"records": rows}


def _load_manifest(path: Path | None, result: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _issue(result, f"Could not read run manifest {path}: {exc}", strict=False, critical=False)
        return {}
    return data if isinstance(data, dict) else {}


def _requires_training_log(manifest: dict[str, Any]) -> bool:
    run_kind = manifest.get("run_kind")
    if run_kind in {"ablation", "fusion", "knowledge_comparison"}:
        return manifest.get("training_strategy") != "evaluation_time_variant"
    return run_kind in {None, "", "ours_full"}


def _requires_formal_tactic_decoding(manifest: dict[str, Any]) -> bool:
    run_kind = manifest.get("run_kind")
    if run_kind == "baseline":
        return False
    return run_kind in {None, "", "ours_full", "ablation", "fusion", "knowledge_comparison"}


def _issue(
    result: dict[str, Any],
    message: str,
    *,
    strict: bool,
    critical: bool,
) -> None:
    target = "errors" if strict and critical else "warnings"
    if message not in result[target]:
        result[target].append(message)


def _structured_view(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("structured_prediction")
    return nested if isinstance(nested, dict) else record


def _require_keys(payload: dict[str, Any], keys: set[str], prefix: str, missing: list[str]) -> None:
    for key in sorted(keys):
        if key not in payload or payload.get(key) is None:
            missing.append(f"{prefix}.{key}")


def _stage_d_trace_available(record: dict[str, Any], structured: dict[str, Any]) -> bool:
    provenance = structured.get("output_provenance", {}) or {}
    hooks = structured.get("training_hooks", {}) or {}
    metadata = record.get("stage_metadata", {}) or {}
    stage_d = metadata.get("stage_d", {}) if isinstance(metadata, dict) else {}
    stage_e = metadata.get("stage_e", {}) if isinstance(metadata, dict) else {}
    return bool(
        provenance.get("stage_d_trace_available")
        or hooks.get("stage_d_trace_available")
        or stage_e.get("stage_d_trace_available")
        or stage_d.get("attention_trace")
    )


def _find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                found.append(current_value)
            found.extend(_find_values(current_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_values(item, key))
    return found


def _latest_split_sizes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return _first_dict(_find_values(rows[-1], "split_sizes"))


def _validation_looks_empty(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    latest = rows[-1]
    primary = [_number(latest.get(key)) for key in ("val_accuracy", "val_macro_f1", "val_roc_auc")]
    confusion = sum(_integer(latest.get(key)) or 0 for key in ("val_tn", "val_fp", "val_fn", "val_tp"))
    return all(value is None for value in primary) and confusion == 0


def _is_metric_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("accuracy", "precision", "recall", "f1", "auc"))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _as_bool_dict(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(current) for key, current in value.items()}


def _first_dict(values: list[Any]) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _display_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"
