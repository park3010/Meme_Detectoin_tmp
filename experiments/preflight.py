"""Experiment 0 preflight checks for paper-quality experiment readiness."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset import MemeDataset
from dataset.labels import LabelVocab, NormalizedLabelStore, iter_normalized_label_paths
from experiments.metric_contract import resolve_metric_contract, write_metric_contract_artifact
from experiments.prediction_io import compact_stage_metadata
from experiments.run_manifest import sha256_file
from experiments.splits import build_splits_for_dataset, label_to_int, load_official_split_ids, load_split_file, normalize_dataset_names, save_splits
from module.backbone.text import TextEncoderWrapper
from module.backbone.vision import CLIPWrapper
from module.runner import HarmfulMemePipeline
from utils.annotation_utils import as_list
from utils.io import load_yaml, read_jsonl, write_json, write_jsonl
from utils.tensor_utils import tensor_to_python


@dataclass
class PreflightIssue:
    """One preflight warning or blocking error."""

    code: str
    severity: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    """Top-level preflight result artifact."""

    schema_version: str
    profile: str
    passed: bool
    strict: bool
    config_path: str
    config_sha256: str | None
    datasets: list[str]
    seeds: list[int]
    checks: dict[str, Any]
    warnings: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_preflight(
    *,
    profile: str = "smoke",
    config_path: str | Path = "configs/config.yaml",
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    label_set: str = "clean",
    normalized_root: str | Path = "dataset/annotation_normalized",
    vocab_path: str | Path = "configs/label_vocab.yaml",
    device: str = "cpu",
    output_root: str | Path = "result",
    strict: bool = False,
    fail_on_warnings: bool = False,
    create_missing_splits: bool | None = None,
    overwrite_splits: bool = False,
    probe_pipeline: bool | None = None,
    probe_limit: int = 3,
    allow_fallback: bool = False,
    allow_download: bool = False,
    write_report: bool = True,
) -> PreflightResult:
    """Run Experiment 0 preflight and write all artifacts."""

    cfg = load_yaml(config_path)
    profile_cfg = _profile_config(cfg, profile)
    selected_datasets = normalize_dataset_names(datasets) or list(cfg.get("data", {}).get("default_datasets", ["harm_c", "harm_p", "facebook", "memotion"]))
    selected_seeds = [int(seed) for seed in (seeds or [42])]
    create_splits = bool(profile_cfg.get("create_missing_splits", True) if create_missing_splits is None else create_missing_splits)
    should_probe = bool(profile_cfg.get("probe_pipeline", False) if probe_pipeline is None else probe_pipeline)
    output_dir = Path(output_root) / "preflight" / profile
    issues: list[PreflightIssue] = []

    backbone = inspect_backbone_readiness(cfg, profile_cfg, device=device, allow_download=allow_download)
    _add_backbone_issues(backbone, profile_cfg, issues, allow_fallback=allow_fallback, profile=profile)

    split_report = inspect_split_integrity(
        cfg,
        selected_datasets,
        selected_seeds,
        output_root=output_root,
        create_missing_splits=create_splits,
        overwrite_splits=overwrite_splits,
        issues=issues,
    )

    eligibility, eligibility_rows = inspect_dataset_metric_eligibility(
        cfg,
        selected_datasets,
        selected_seeds,
        split_report=split_report,
        normalized_root=normalized_root,
        label_set=label_set,
        vocab_path=vocab_path,
        issues=issues,
        require_metric_eligibility=bool(profile_cfg.get("require_metric_eligibility", False)),
    )
    retrieval = inspect_retrieval_corpus_readiness(cfg, profile_cfg, issues)
    metric_contract = inspect_metric_contract(cfg, vocab_path=vocab_path, profile_cfg=profile_cfg, issues=issues)
    snapshot = normalized_annotation_snapshot(selected_datasets, normalized_root=normalized_root, label_set=label_set, vocab_path=vocab_path)
    probe = None
    if should_probe:
        if profile == "main_experiment" and not backbone["ready_for_main_experiment"]:
            _issue(issues, "probe_blocked_backbone", "warning", "Pipeline probe skipped because strict backbone readiness failed.", {})
        else:
            probe = probe_pipeline_provenance(cfg, selected_datasets, device=device, limit=probe_limit)

    warnings = [asdict(issue) for issue in issues if issue.severity == "warning"]
    errors = [asdict(issue) for issue in issues if issue.severity == "error"]
    decision = "BLOCKED" if errors or (fail_on_warnings and warnings) else "PASS_WITH_WARNINGS" if warnings else "PASS"
    checks = {
        "decision": decision,
        "profile_description": profile_cfg.get("description", ""),
        "backbone_readiness": backbone,
        "dataset_metric_eligibility": eligibility,
        "split_integrity": split_report,
        "retrieval_corpus_audit": retrieval,
        "metric_contract": metric_contract,
        "normalized_annotation_snapshot": snapshot,
    }
    if probe is not None:
        checks["pipeline_provenance_probe"] = probe
    result = PreflightResult(
        schema_version=str(cfg.get("preflight", {}).get("schema_version", "experiment_preflight_v1")),
        profile=profile,
        passed=decision != "BLOCKED",
        strict=strict,
        config_path=str(config_path),
        config_sha256=sha256_file(config_path),
        datasets=selected_datasets,
        seeds=selected_seeds,
        checks=checks,
        warnings=warnings,
        errors=errors,
        artifacts={},
    )
    artifacts = write_preflight_artifacts(
        result,
        output_dir,
        eligibility_rows=eligibility_rows,
        write_report=write_report,
        probe=probe,
    )
    result.artifacts = {key: str(path) for key, path in artifacts.items()}
    if write_report and "preflight_report" in artifacts:
        artifacts["preflight_report"].write_text(format_preflight_report(result), encoding="utf-8")
    write_json(output_dir / "preflight_manifest.json", result.to_dict())
    return result


def inspect_backbone_readiness(
    config: dict[str, Any],
    profile: dict[str, Any],
    device: str,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Instantiate runtime backbone wrappers and return honest readiness state."""

    model_cfg = config.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    backbone_cfg = config.get("backbone", config.get("backbones", {}))
    clip_cfg = dict(backbone_cfg.get("clip", {}) or {})
    text_cfg = dict(backbone_cfg.get("text", {}) or {})
    if allow_download:
        clip_cfg["allow_download"] = True
        clip_cfg["local_files_only"] = False
        text_cfg["allow_download"] = True
        text_cfg["local_files_only"] = False

    vision = CLIPWrapper(
        hidden_dim=hidden_dim,
        prefer_pretrained=bool(clip_cfg.get("prefer_pretrained", False)),
        model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
        device=device,
        pretrained_tag=clip_cfg.get("pretrained_tag"),
        checkpoint_path=clip_cfg.get("checkpoint_path"),
        cache_dir=clip_cfg.get("cache_dir"),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        allow_download=bool(clip_cfg.get("allow_download", False)),
    )
    text = TextEncoderWrapper(
        hidden_dim=hidden_dim,
        prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
        model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
        device=device,
        checkpoint_path=text_cfg.get("checkpoint_path"),
        cache_dir=text_cfg.get("cache_dir"),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
        allow_download=bool(text_cfg.get("allow_download", False)),
    )
    vision_state = vision.readiness_state()
    text_state = text.readiness_state()
    ready_smoke = True
    vision_main = _vision_ready_for_main(vision_state, profile)
    text_main = _text_ready_for_main(text_state, profile)
    return {
        "vision": {
            "ready_for_smoke": ready_smoke,
            "ready_for_main_experiment": vision_main,
            "readiness_state": vision_state,
        },
        "text": {
            "ready_for_smoke": ready_smoke,
            "ready_for_main_experiment": text_main,
            "readiness_state": text_state,
        },
        "ready_for_smoke": ready_smoke,
        "ready_for_main_experiment": vision_main and text_main,
    }


def inspect_split_integrity(
    config: dict[str, Any],
    datasets: list[str],
    seeds: list[int],
    *,
    output_root: str | Path,
    create_missing_splits: bool,
    overwrite_splits: bool,
    issues: list[PreflightIssue],
) -> dict[str, Any]:
    """Resolve/persist splits and check duplicate/overlap/coverage integrity."""

    dataset_root = config.get("paths", {}).get("dataset_root", "dataset/source")
    annotation_root = config.get("paths", {}).get("annotation_root", "dataset/annotation")
    report: dict[str, Any] = {}
    for dataset_name in datasets:
        dataset = MemeDataset(dataset_root=dataset_root, annotation_root=annotation_root, dataset_names=[dataset_name], keep_missing_images=True)
        sample_ids = {str(sample.get("sample_id")) for sample in dataset}
        report[dataset_name] = {}
        for seed in seeds:
            split_path = Path(output_root) / "splits" / dataset_name / f"seed_{seed}.json"
            origin = "persisted" if split_path.exists() else "generated"
            if split_path.exists() and not overwrite_splits:
                splits = load_split_file(split_path)
            elif split_path.exists() and overwrite_splits:
                splits = _build_splits_with_origin(dataset_name, dataset, seed, dataset_root)
                origin = splits.pop("_origin", "generated")
                save_splits(splits, dataset_name, seed, Path(output_root) / "splits")
            elif create_missing_splits:
                splits = _build_splits_with_origin(dataset_name, dataset, seed, dataset_root)
                origin = splits.pop("_origin", "generated")
                save_splits(splits, dataset_name, seed, Path(output_root) / "splits")
            else:
                splits = {"train": [], "valid": [], "test": []}
                _issue(issues, "split_missing", "error", "Split file is missing and creation is disabled.", {"dataset": dataset_name, "seed": seed, "path": str(split_path)})

            duplicates = {name: _duplicates(ids) for name, ids in splits.items()}
            overlap = _split_overlap(splits)
            missing_ids = {name: sorted(set(ids) - sample_ids) for name, ids in splits.items()}
            if any(duplicates.values()):
                _issue(issues, "split_duplicate_ids", "error", "Duplicate sample IDs found within split.", {"dataset": dataset_name, "seed": seed, "duplicates": duplicates})
            if overlap:
                _issue(issues, "split_overlap", "error", "Sample IDs overlap across train/valid/test.", {"dataset": dataset_name, "seed": seed, "overlap": overlap})
            if any(missing_ids.values()):
                _issue(issues, "split_missing_ids", "error", "Split contains sample IDs not found in dataset.", {"dataset": dataset_name, "seed": seed, "missing_ids": missing_ids})
            report[dataset_name][str(seed)] = {
                "path": str(split_path),
                "sha256": sha256_file(split_path),
                "origin": origin,
                "split_sizes": {name: len(ids) for name, ids in splits.items()},
                "duplicates": duplicates,
                "overlap": overlap,
                "missing_ids": missing_ids,
                "harmfulness_distribution": _harmfulness_distribution(dataset, splits),
                "reusable_split_path": str(split_path),
            }
    return report


def inspect_dataset_metric_eligibility(
    config: dict[str, Any],
    datasets: list[str],
    seeds: list[int],
    *,
    split_report: dict[str, Any],
    normalized_root: str | Path,
    label_set: str,
    vocab_path: str | Path,
    issues: list[PreflightIssue],
    require_metric_eligibility: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute field/split label eligibility from normalized labels and vocab."""

    vocab = LabelVocab.from_yaml(vocab_path)
    eligibility_cfg = config.get("preflight", {}).get("eligibility", {}) or {}
    fields = list(eligibility_cfg.get("fields", []))
    hard_min = {
        "train": int(eligibility_cfg.get("hard_min_train_labeled_samples", 1)),
        "valid": int(eligibility_cfg.get("hard_min_valid_labeled_samples", 1)),
        "test": int(eligibility_cfg.get("hard_min_test_labeled_samples", 1)),
    }
    advisory_min = {
        "train": int(eligibility_cfg.get("advisory_min_train_labeled_samples", 20)),
        "valid": int(eligibility_cfg.get("advisory_min_valid_labeled_samples", 10)),
        "test": int(eligibility_cfg.get("advisory_min_test_labeled_samples", 10)),
    }
    hard_min_classes = int(eligibility_cfg.get("hard_min_eval_classes", 2))
    advisory_min_support = int(eligibility_cfg.get("advisory_min_samples_per_present_class", 2))
    store = NormalizedLabelStore(normalized_root=normalized_root, dataset_names=datasets, label_set=label_set)
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for dataset_name in datasets:
        result[dataset_name] = {}
        for seed in seeds:
            split_info = split_report.get(dataset_name, {}).get(str(seed), {})
            split_path = split_info.get("path")
            splits = load_split_file(split_path) if split_path and Path(split_path).exists() else {"train": [], "valid": [], "test": []}
            result[dataset_name][str(seed)] = {}
            for field_name in fields:
                split_records = {}
                overall_hard = True
                overall_advisory = True
                for split_name in ["train", "valid", "test"]:
                    record = _field_split_eligibility(store, vocab, field_name, dataset_name, splits.get(split_name, []))
                    record.update(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "field": field_name,
                            "split": split_name,
                            "eligible_hard": record["valid_labeled_samples"] >= hard_min[split_name],
                            "eligible_advisory": record["valid_labeled_samples"] >= advisory_min[split_name],
                        }
                    )
                    if split_name == "test":
                        record["eligible_hard"] = record["eligible_hard"] and record["number_of_observed_non_ignored_classes"] >= hard_min_classes
                        record["eligible_advisory"] = record["eligible_advisory"] and (record["minimum_class_support"] or 0) >= advisory_min_support
                    record["reason_codes"] = _eligibility_reasons(record)
                    split_records[split_name] = record
                    rows.append(record)
                    overall_hard = overall_hard and bool(record["eligible_hard"])
                    overall_advisory = overall_advisory and bool(record["eligible_advisory"])
                result[dataset_name][str(seed)][field_name] = {
                    "splits": split_records,
                    "eligible_hard": overall_hard,
                    "eligible_advisory": overall_advisory,
                }
                if not overall_hard:
                    severity = "error" if require_metric_eligibility else "warning"
                    _issue(
                        issues,
                        "metric_ineligible",
                        severity,
                        "Dataset/field is not hard-eligible for the declared metric.",
                        {"dataset": dataset_name, "seed": seed, "field": field_name},
                    )
                elif not overall_advisory:
                    _issue(
                        issues,
                        "metric_low_support",
                        "warning",
                        "Dataset/field passes hard eligibility but is below advisory support.",
                        {"dataset": dataset_name, "seed": seed, "field": field_name},
                    )
    return result, rows


def inspect_retrieval_corpus_readiness(config: dict[str, Any], profile: dict[str, Any], issues: list[PreflightIssue]) -> dict[str, Any]:
    """Audit configured retrieval corpora and retrieval provenance policy."""

    paths = list(config.get("paths", {}).get("retrieval_corpus_paths", []) or [])
    report_paths = []
    usable_count = 0
    for raw_path in paths:
        path = Path(raw_path)
        info = _audit_corpus_path(path)
        usable_count += int(info["parseable_record_count"] > 0 and info["text_field_coverage"] > 0)
        report_paths.append(info)
    retriever_cfg = config.get("backbone", {}).get("retriever", {})
    stage_b_cfg = config.get("stages", {}).get("stage_b", {})
    stage_c_cfg = config.get("stages", {}).get("stage_c", {})
    report = {
        "paths": report_paths,
        "usable_corpus_count": usable_count,
        "policy": {
            "retriever_backend": retriever_cfg.get("backend", "local"),
            "max_documents": retriever_cfg.get("max_documents"),
            "cross_encoder_rerank_enabled": retriever_cfg.get("use_cross_encoder_rerank", True),
            "fallback_candidates_enabled": retriever_cfg.get("fallback_candidates", True),
            "stage_b_top_k": stage_b_cfg.get("top_k", config.get("model", {}).get("knowledge_top_k")),
            "stage_c_min_relevance": stage_c_cfg.get("min_relevance"),
            "allow_low_relevance_fallback": stage_c_cfg.get("allow_low_relevance_fallback"),
            "semantic_note": "fallback candidate != retrieved external knowledge; generated hypothesis != retrieved external knowledge",
        },
    }
    if profile.get("require_retrieval_corpus", False) and usable_count == 0:
        _issue(issues, "retrieval_corpus_unusable", "error", "No configured retrieval corpus is present and structurally usable.", {"paths": paths})
    if retriever_cfg.get("fallback_candidates", True):
        _issue(issues, "retrieval_fallback_enabled", "warning", "Fallback candidates are enabled; provenance must not treat them as retrieved external knowledge.", {})
    return report


def inspect_metric_contract(config: dict[str, Any], *, vocab_path: str | Path, profile_cfg: dict[str, Any], issues: list[PreflightIssue]) -> dict[str, Any]:
    """Resolve metric contract and flag missing formal metric capabilities."""

    contract = resolve_metric_contract(config, vocab_path=vocab_path)
    if profile_cfg.get("require_metric_eligibility", False) and contract.get("implementation_status") != "ready":
        _issue(
            issues,
            "metric_contract_blocked",
            "error",
            "Metric contract is not fully implementable under declared formal metric policy.",
            {"missing_capabilities": contract.get("missing_capabilities", [])},
        )
    return contract


def normalized_annotation_snapshot(
    datasets: list[str],
    *,
    normalized_root: str | Path,
    label_set: str,
    vocab_path: str | Path,
) -> dict[str, Any]:
    """Return normalized label file and vocab snapshot metadata."""

    files = []
    for dataset, path in zip(datasets, iter_normalized_label_paths(normalized_root, datasets, label_set=label_set)):
        files.append({"dataset": dataset, "path": str(path), "exists": path.exists(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size if path.exists() else 0})
    return {
        "normalized_root": str(normalized_root),
        "label_set": label_set,
        "vocab_path": str(vocab_path),
        "vocab_sha256": sha256_file(vocab_path),
        "files": files,
    }


def probe_pipeline_provenance(config: dict[str, Any], datasets: list[str], *, device: str, limit: int = 3) -> dict[str, Any]:
    """Run a tiny in-memory pipeline provenance probe without training."""

    cfg = dict(config)
    cfg.setdefault("runtime", {})["device"] = device
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=datasets[:1],
        keep_missing_images=True,
        limit=limit,
    )
    pipeline = HarmfulMemePipeline(cfg).eval()
    records = []
    for sample in list(dataset)[:limit]:
        outputs = pipeline(dict(sample))
        stage_e = outputs.get("stage_e")
        structured = stage_e.structured_prediction if stage_e is not None else {}
        records.append(
            {
                "sample_id": sample.get("sample_id"),
                "stage_metadata": compact_stage_metadata(outputs),
                "stage_e_contract": {
                    "has_output_provenance": bool(structured.get("output_provenance")),
                    "target_presence_provenance": (structured.get("target") or {}).get("presence_provenance"),
                    "multimodal_relation_provenance": (structured.get("tactic") or {}).get("multimodal_relation_provenance"),
                    "trainable_logits_fields": (structured.get("output_provenance") or {}).get("trainable_logits_fields", []),
                    "proxy_fields": (structured.get("output_provenance") or {}).get("proxy_fields", []),
                },
            }
        )
    return {"probe_limit": limit, "records": tensor_to_python(records, max_elements=12)}


def write_preflight_artifacts(
    result: PreflightResult,
    output_dir: str | Path,
    *,
    eligibility_rows: list[dict[str, Any]],
    write_report: bool,
    probe: dict[str, Any] | None,
) -> dict[str, Path]:
    """Write the canonical Experiment 0 artifact bundle."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = result.checks
    artifacts: dict[str, Path] = {}
    artifacts["preflight_manifest"] = out / "preflight_manifest.json"
    artifacts["backbone_readiness"] = out / "backbone_readiness.json"
    artifacts["dataset_metric_eligibility_json"] = out / "dataset_metric_eligibility.json"
    artifacts["dataset_metric_eligibility_csv"] = out / "dataset_metric_eligibility.csv"
    artifacts["split_integrity_report"] = out / "split_integrity_report.json"
    artifacts["retrieval_corpus_audit"] = out / "retrieval_corpus_audit.json"
    artifacts["metric_contract"] = out / "metric_contract.json"
    artifacts["normalized_annotation_snapshot"] = out / "normalized_annotation_snapshot.json"
    write_json(artifacts["backbone_readiness"], checks["backbone_readiness"])
    write_json(artifacts["dataset_metric_eligibility_json"], checks["dataset_metric_eligibility"])
    _write_eligibility_csv(artifacts["dataset_metric_eligibility_csv"], eligibility_rows)
    write_json(artifacts["split_integrity_report"], checks["split_integrity"])
    write_json(artifacts["retrieval_corpus_audit"], checks["retrieval_corpus_audit"])
    write_metric_contract_artifact(checks["metric_contract"], artifacts["metric_contract"])
    write_json(artifacts["normalized_annotation_snapshot"], checks["normalized_annotation_snapshot"])
    if probe is not None:
        artifacts["pipeline_provenance_probe"] = out / "pipeline_provenance_probe.json"
        write_json(artifacts["pipeline_provenance_probe"], probe)
    if write_report:
        artifacts["preflight_report"] = out / "preflight_report.md"
        artifacts["preflight_report"].write_text(format_preflight_report(result), encoding="utf-8")
    return artifacts


def format_preflight_summary(result: PreflightResult) -> str:
    """Return a concise terminal summary."""

    decision = result.checks.get("decision", "UNKNOWN")
    lines = [
        f"Experiment 0 preflight: {decision}",
        f"Profile: {result.profile}",
        f"Datasets: {', '.join(result.datasets)}",
        f"Seeds: {', '.join(str(seed) for seed in result.seeds)}",
        f"Warnings: {len(result.warnings)}",
        f"Blocking errors: {len(result.errors)}",
    ]
    if result.errors:
        lines.append("Blocking reasons:")
        lines.extend(f"- {error['code']}: {error['message']}" for error in result.errors[:8])
    return "\n".join(lines)


def format_preflight_report(result: PreflightResult) -> str:
    """Render the human-readable Markdown report."""

    decision = result.checks.get("decision", "UNKNOWN")
    lines = [
        "# Experiment 0 Preflight Report",
        "",
        "## Profile and command",
        f"- Profile: `{result.profile}`",
        f"- Strict: `{result.strict}`",
        f"- Config: `{result.config_path}`",
        f"- Config SHA256: `{result.config_sha256}`",
        f"- Created: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Overall decision",
        f"**{decision}**",
        "",
        "## Backbone readiness",
        f"- Vision main-ready: `{result.checks['backbone_readiness']['vision']['ready_for_main_experiment']}`",
        f"- Text main-ready: `{result.checks['backbone_readiness']['text']['ready_for_main_experiment']}`",
        f"- Vision state: `{result.checks['backbone_readiness']['vision']['readiness_state']}`",
        f"- Text state: `{result.checks['backbone_readiness']['text']['readiness_state']}`",
        "",
        "## Dataset / label eligibility",
        f"- Datasets: `{result.datasets}`",
        f"- Detailed CSV: `{result.artifacts.get('dataset_metric_eligibility_csv', 'dataset_metric_eligibility.csv')}`",
        "",
        "## Split integrity",
        f"- Split report: `{result.artifacts.get('split_integrity_report', 'split_integrity_report.json')}`",
        "",
        "## Retrieval corpus readiness",
        f"- Usable corpora: `{result.checks['retrieval_corpus_audit'].get('usable_corpus_count')}`",
        f"- Provenance note: `{result.checks['retrieval_corpus_audit'].get('policy', {}).get('semantic_note')}`",
        "",
        "## Metric contract",
        f"- Implementation status: `{result.checks['metric_contract'].get('implementation_status')}`",
        f"- Missing capabilities: `{result.checks['metric_contract'].get('missing_capabilities', [])}`",
        "",
        "## Normalized annotation snapshot",
        f"- Snapshot artifact: `{result.artifacts.get('normalized_annotation_snapshot', 'normalized_annotation_snapshot.json')}`",
        "",
        "## Warnings",
        *([f"- `{item['code']}`: {item['message']} `{item.get('context', {})}`" for item in result.warnings] or ["- None"]),
        "",
        "## Blocking errors",
        *([f"- `{item['code']}`: {item['message']} `{item.get('context', {})}`" for item in result.errors] or ["- None"]),
        "",
        "## Main-experiment acceptance decision",
        "Main experiment results are valid only after `main_experiment` strict preflight passes.",
        "",
        "## Required next actions",
        *(_next_actions(result) or ["- None"]),
    ]
    return "\n".join(lines) + "\n"


def _profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = config.get("preflight", {}).get("profiles", {}) or {}
    if profile not in profiles:
        raise ValueError(f"Unknown preflight profile: {profile}")
    return dict(profiles[profile])


def _add_backbone_issues(backbone: dict[str, Any], profile_cfg: dict[str, Any], issues: list[PreflightIssue], *, allow_fallback: bool, profile: str) -> None:
    for name in ["vision", "text"]:
        state = backbone[name]["readiness_state"]
        if state.get("fallback_used"):
            severity = "warning"
            if profile == "main_experiment" and profile_cfg.get("forbid_fallback_backbones", False) and not allow_fallback:
                severity = "error"
            _issue(issues, f"{name}_fallback_backbone", severity, f"{name} backbone is using fallback features.", state)
        if state.get("random_initialization_used"):
            _issue(issues, f"{name}_random_initialization", "error", f"{name} backbone has random initialization, not pretrained weights.", state)
        if profile_cfg.get(f"require_pretrained_{name}", False) and not state.get("weights_loaded"):
            _issue(issues, f"{name}_pretrained_missing", "error", f"{name} pretrained weights are required but not loaded.", state)
        if state.get("checkpoint_path") and not state.get("checkpoint_exists"):
            _issue(issues, f"{name}_checkpoint_missing", "error", f"{name} checkpoint_path was configured but does not exist.", state)


def _vision_ready_for_main(state: dict[str, Any], profile: dict[str, Any]) -> bool:
    if profile.get("forbid_fallback_backbones") and state.get("fallback_used"):
        return False
    if state.get("random_initialization_used"):
        return False
    if profile.get("require_pretrained_vision") and not state.get("weights_loaded"):
        return False
    return True


def _text_ready_for_main(state: dict[str, Any], profile: dict[str, Any]) -> bool:
    if profile.get("forbid_fallback_backbones") and state.get("fallback_used"):
        return False
    if profile.get("require_pretrained_text") and not state.get("weights_loaded"):
        return False
    return True


def _build_splits_with_origin(dataset_name: str, dataset: MemeDataset, seed: int, dataset_root: str | Path) -> dict[str, Any]:
    official = load_official_split_ids(dataset_name, dataset_root=dataset_root)
    splits = build_splits_for_dataset(dataset_name, dataset, seed=seed, dataset_root=dataset_root)
    splits["_origin"] = "official" if official else "generated"
    return splits


def _field_split_eligibility(store: NormalizedLabelStore, vocab: LabelVocab, field_name: str, dataset_name: str, ids: list[str]) -> dict[str, Any]:
    total = len(ids)
    coverage = 0
    missing = 0
    ignored_ambiguous = 0
    ignored_unknown = 0
    valid = 0
    classes: Counter[str] = Counter()
    is_multi = field_name in vocab.multi_label_fields
    ignore = vocab.multi_ignore_labels.get(field_name, set()) if is_multi else vocab.single_ignore_labels.get(field_name, set())
    vocab_labels = set(vocab.multi_label_fields.get(field_name, []) if is_multi else vocab.single_label_fields.get(field_name, []))
    for sample_id in ids:
        row = store.get(dataset_name, str(sample_id))
        if row is None:
            missing += 1
            continue
        coverage += 1
        labels = row.labels or {}
        raw_value = labels.get(field_name)
        if field_name not in labels or raw_value is None or raw_value == "":
            missing += 1
            continue
        values = [str(value) for value in as_list(raw_value)] if is_multi else [str(raw_value)]
        usable = []
        for value in values:
            if value == "ambiguous" and value in ignore:
                ignored_ambiguous += 1
                continue
            if (value == "unknown" and value in ignore) or (value not in vocab_labels and "unknown" in ignore):
                ignored_unknown += 1
                continue
            if value in ignore:
                continue
            if value in vocab_labels:
                usable.append(value)
        if not usable:
            continue
        valid += 1
        classes.update(usable)
    return {
        "total_samples": total,
        "normalized_annotation_coverage": coverage,
        "valid_labeled_samples": valid,
        "ignored_ambiguous_count": ignored_ambiguous,
        "ignored_unknown_count": ignored_unknown,
        "missing_label_count": missing,
        "class_distribution": dict(sorted(classes.items())),
        "number_of_observed_non_ignored_classes": len(classes),
        "minimum_class_support": min(classes.values()) if classes else 0,
    }


def _eligibility_reasons(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record["valid_labeled_samples"] == 0:
        reasons.append("no_valid_labels")
    if record["split"] == "test" and record["number_of_observed_non_ignored_classes"] < 2:
        reasons.append("test_has_fewer_than_two_classes")
    if not record["eligible_advisory"]:
        reasons.append("below_advisory_support")
    return reasons


def _audit_corpus_path(path: Path) -> dict[str, Any]:
    exists = path.exists()
    line_count = 0
    parseable = 0
    empty_text = 0
    doc_id = 0
    text_field = 0
    if exists and path.is_file():
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                line_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                parseable += 1
                text = record.get("text") or record.get("contents") or record.get("content") or record.get("passage") or record.get("summary") or record.get("caption") or record.get("title") or ""
                if str(text).strip():
                    text_field += 1
                else:
                    empty_text += 1
                if record.get("id") or record.get("doc_id") or record.get("kid"):
                    doc_id += 1
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            line_count = len(text.splitlines())
            parseable = int(bool(text.strip()))
            text_field = parseable
            empty_text = int(not bool(text.strip()))
            doc_id = parseable
    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
        "sha256": sha256_file(path),
        "line_count": line_count,
        "parseable_record_count": parseable,
        "empty_text_record_count": empty_text,
        "document_id_coverage": doc_id / parseable if parseable else 0.0,
        "text_field_coverage": text_field / parseable if parseable else 0.0,
    }


def _harmfulness_distribution(dataset: MemeDataset, splits: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    by_id = {str(sample.get("sample_id")): label_to_int(sample.get("raw_label")) for sample in dataset}
    out = {}
    for split, ids in splits.items():
        counter: Counter[str] = Counter()
        for sample_id in ids:
            counter[str(by_id.get(str(sample_id)))] += 1
        out[split] = dict(counter)
    return out


def _duplicates(ids: list[str]) -> list[str]:
    counts = Counter(ids)
    return sorted(item for item, count in counts.items() if count > 1)


def _split_overlap(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    names = ["train", "valid", "test"]
    overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = sorted(set(splits.get(left, [])) & set(splits.get(right, [])))
            if common:
                overlap[f"{left}_{right}"] = common
    return overlap


def _write_eligibility_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "seed",
        "field",
        "split",
        "total_samples",
        "normalized_annotation_coverage",
        "valid_labeled_samples",
        "ignored_ambiguous_count",
        "ignored_unknown_count",
        "missing_label_count",
        "class_distribution",
        "number_of_observed_non_ignored_classes",
        "minimum_class_support",
        "eligible_hard",
        "eligible_advisory",
        "reason_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in columns})


def _next_actions(result: PreflightResult) -> list[str]:
    actions = []
    for error in result.errors:
        if "pretrained_missing" in error["code"] or "fallback_backbone" in error["code"]:
            actions.append("- Provide local pretrained vision/text checkpoints in `configs/config.yaml` and rerun strict preflight.")
            break
    if any(error["code"] == "metric_contract_blocked" for error in result.errors):
        actions.append("- Implement logits-only validation-threshold decoding for `tactic_rhetorical` formal metrics.")
    if any(error["code"] == "retrieval_corpus_unusable" for error in result.errors):
        actions.append("- Provide a parseable local retrieval corpus under `paths.retrieval_corpus_paths`.")
    return actions


def _issue(issues: list[PreflightIssue], code: str, severity: str, message: str, context: dict[str, Any]) -> None:
    issues.append(PreflightIssue(code=code, severity=severity, message=message, context=context))


__all__ = [
    "PreflightIssue",
    "PreflightResult",
    "run_preflight",
    "write_preflight_artifacts",
    "format_preflight_summary",
    "inspect_backbone_readiness",
    "inspect_dataset_metric_eligibility",
    "inspect_split_integrity",
    "inspect_retrieval_corpus_readiness",
]
