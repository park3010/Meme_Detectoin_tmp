"""Built-in baseline and framework adapters for the locked paper protocol."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.pipeline_audit import audit_baseline_run_artifacts, audit_run_artifacts, format_audit_summary
from experiments.research_automatic_analysis import analyze_prediction_artifacts
from experiments.research_protocol import sha256_file, source_tree_sha256
from experiments.run_manifest import update_run_manifest
from experiments.train import BaselineRunConfig, OursRunConfig, run_baseline_experiment, run_ours_experiment
from utils.io import load_yaml, write_json

from .base import ExperimentAdapter, RunContext
from .blocked import BlockedExternalAdapter


class _BuiltinAdapter(ExperimentAdapter):
    native_run_dir: Path | None = None
    _started_at: float | None = None
    _metrics: dict[str, Any] | None = None

    def prepare_data(self) -> dict[str, Any]:
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not self.context.force:
            raise FileExistsError(f"Refusing to overwrite completed/partial run: {self.run_dir}")
        if not Path(self.spec.split_manifest).exists():
            raise FileNotFoundError(f"Source split manifest is missing: {self.spec.split_manifest}")
        if not Path(self.context.fhm_manifest).exists():
            raise FileNotFoundError(f"FHM held-out manifest is missing: {self.context.fhm_manifest}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        return {"source_split": self.spec.split_manifest, "fhm_test": self.context.fhm_manifest}

    def predict_validation(self) -> Path | None:
        path = self._native_path("validation_predictions.jsonl")
        return path if path.exists() else None

    def freeze_selection(self) -> dict[str, Any]:
        native = self._native_path("tactic_rhetorical_decoding.json")
        if native.exists():
            payload = json.loads(native.read_text(encoding="utf-8"))
        else:
            payload = {
                "status": "not_applicable" if self.spec.tasks == ["harmfulness"] else "unavailable",
                "selection_dataset": "HarMeme validation",
            }
        write_json(self.run_dir / "thresholds.json", payload)
        return payload

    def predict_test(self) -> Path | None:
        path = self._native_path("final_predictions.jsonl")
        return path if path.exists() else None

    def evaluate(self) -> dict[str, Any]:
        metrics_path = self._native_path("metrics.json")
        self._metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        return self._metrics

    def audit(self) -> dict[str, Any]:
        if self.native_run_dir is None:
            raise RuntimeError("Adapter has not executed a native run")
        if self.spec.adapter == "builtin_baseline":
            result = audit_baseline_run_artifacts(
                self.native_run_dir,
                require_nonempty_metrics=True,
                strict=True,
            )
        else:
            result = audit_run_artifacts(
                self.native_run_dir,
                require_nonempty_metrics=True,
                allow_empty_split=False,
                strict=True,
            )
        write_json(self.run_dir / "pipeline_audit_report.json", result)
        (self.run_dir / "pipeline_audit_report.md").write_text(
            "# Pipeline audit\n\n```text\n" + format_audit_summary(result) + "\n```\n",
            encoding="utf-8",
        )
        return result

    def export(self) -> dict[str, Any]:
        if self.native_run_dir is None:
            raise RuntimeError("Adapter has not executed a native run")
        copies = {
            "run_manifest.json": "run_manifest.json",
            "training_log.json": "training_log.json",
            "validation_predictions.jsonl": "validation_predictions.jsonl",
            "final_predictions.jsonl": "test_predictions.jsonl",
            "metrics.json": "metrics.json",
            "best_model.pt": "best_model.pt",
        }
        for source_name, target_name in copies.items():
            source = self.native_run_dir / source_name
            if source.exists():
                shutil.copy2(source, self.run_dir / target_name)
        test_predictions = self.run_dir / "test_predictions.jsonl"
        if test_predictions.exists() and self.spec.adapter == "builtin_framework":
            analyze_prediction_artifacts(test_predictions, self.run_dir / "automatic_analysis.json")
        config_payload = {
            "registry_experiment": self.spec.to_dict(),
            "run_context": asdict(self.context),
            "framework_config": load_yaml(self.context.config_path),
        }
        _write_yaml(self.run_dir / "resolved_config.yaml", config_payload)
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_request": self.context.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        write_json(self.run_dir / "environment.json", environment)
        elapsed = time.perf_counter() - self._started_at if self._started_at is not None else None
        write_json(
            self.run_dir / "runtime.json",
            {
                "wall_seconds": elapsed,
                "training_seconds": None,
                "validation_seconds": None,
                "fhm_inference_seconds": None,
                "fhm_latency_seconds_per_sample": None,
                "peak_gpu_memory_mb": None,
                "status": "total_wall_measured_phase_breakdown_unavailable" if elapsed is not None else "unavailable",
            },
        )
        complexity = _checkpoint_complexity(self.run_dir / "best_model.pt")
        canonical_manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8")) if (self.run_dir / "run_manifest.json").exists() else {}
        data_snapshot = canonical_manifest.get("data_snapshot", {}) or {}
        asset_hashes = _pretrained_asset_hashes(canonical_manifest)
        code_hash = source_tree_sha256()
        complexity["parameter_count"] = canonical_manifest.get("parameter_count", complexity.get("parameter_count"))
        complexity["trainable_parameter_count"] = canonical_manifest.get("trainable_parameter_count")
        complexity["run_artifact_size_bytes"] = sum(path.stat().st_size for path in self.run_dir.rglob("*") if path.is_file())
        write_json(self.run_dir / "complexity.json", complexity)
        update_run_manifest(
            self.run_dir,
            {
                "research_schema": "harmeme_to_fhm_canonical_run_v1",
                "experiment_id": self.spec.id,
                "family": self.spec.family,
                "group": self.spec.group,
                "adapter": self.spec.adapter,
                "adapter_version": self.adapter_version,
                "registry_version": "wsmd2027_harmeme_fhm_20260714_v1",
                "code_sha256": code_hash,
                "code_commit_or_source_tree_sha256": code_hash,
                "source_split_manifest_sha256": sha256_file(self.spec.split_manifest),
                "fhm_test_manifest_sha256": sha256_file(self.context.fhm_manifest),
                "label_vocab_sha256": data_snapshot.get("label_vocab_sha256"),
                "normalized_label_snapshot_sha256": data_snapshot.get("normalized_label_snapshot_sha256"),
                "pretrained_asset_hashes": asset_hashes,
                "fhm_role": "heldout_target_test",
                "annotation_provenance": "original_fhm_label_and_agent_silver_structured_evaluation",
                "selection_data": "HarMeme validation only",
                "test_data": "FHM only",
                "completion_status": "complete",
            },
        )
        return {"status": "completed", "metrics": self._metrics or {}, "artifacts": sorted(path.name for path in self.run_dir.iterdir())}

    def _native_path(self, name: str) -> Path:
        if self.native_run_dir is None:
            raise RuntimeError("Adapter has not executed a native run")
        return self.native_run_dir / name


class BuiltinBaselineAdapter(_BuiltinAdapter):
    """Execute one supervised harmfulness baseline under the fixed protocol."""

    def train_or_fit(self) -> dict[str, Any]:
        native_root = self.run_dir / "native"
        cfg = BaselineRunConfig(
            model_name=self.spec.model,
            dataset_name="harmeme_to_fhm",
            seed=self.context.seed,
            config_path=self.context.config_path,
            output_root=str(native_root),
            epochs=10 if self.context.epochs is None else self.context.epochs,
            device=self.context.device,
            limit=self.context.limit,
            disable_tqdm=self.context.disable_tqdm,
            use_normalized_labels=True,
            require_normalized_label=True,
            label_set="clean",
            source_dataset_names=list(self.spec.source_train_datasets),
            source_split_manifest=self.spec.split_manifest,
            heldout_test_dataset=self.spec.heldout_test_datasets[0],
            heldout_test_manifest=self.context.fhm_manifest,
            annotation_provenance="original_harmfulness_and_agent_silver_structured",
            suite_name=self.context.suite,
        )
        self._metrics = run_baseline_experiment(cfg)
        self.native_run_dir = native_root / "predictions" / "harmeme_to_fhm" / self.spec.model / str(self.context.seed)
        return self._metrics


class BuiltinFrameworkAdapter(_BuiltinAdapter):
    """Execute Ours Full or a train-time ablation under the fixed protocol."""

    def train_or_fit(self) -> dict[str, Any]:
        native_root = self.run_dir / "native"
        cfg = OursRunConfig(
            dataset_name="harmeme_to_fhm",
            seed=self.context.seed,
            config_path=self.context.config_path,
            output_root=str(native_root),
            model_name=self.spec.id,
            epochs=5 if self.context.epochs is None else self.context.epochs,
            device=self.context.device,
            limit=self.context.limit,
            disable_tqdm=self.context.disable_tqdm,
            label_set="clean",
            use_normalized_labels=True,
            require_normalized_label=True,
            use_sample_weight=True,
            source_dataset_names=list(self.spec.source_train_datasets),
            source_split_manifest=self.spec.split_manifest,
            heldout_test_dataset=self.spec.heldout_test_datasets[0],
            heldout_test_manifest=self.context.fhm_manifest,
            annotation_provenance="original_harmfulness_and_agent_silver_structured",
            ablation_name=self.spec.ablation,
            structured_auxiliary=self.spec.ablation != "w_o_structured_auxiliary",
            suite_name=self.context.suite,
        )
        self._metrics = run_ours_experiment(cfg)
        self.native_run_dir = native_root / "predictions" / "harmeme_to_fhm" / self.spec.id / str(self.context.seed)
        return self._metrics


def create_adapter(spec: Any, context: RunContext) -> ExperimentAdapter:
    """Instantiate the registered adapter without executing it."""

    if spec.adapter == "builtin_baseline":
        return BuiltinBaselineAdapter(spec, context)
    if spec.adapter == "builtin_framework":
        return BuiltinFrameworkAdapter(spec, context)
    return BlockedExternalAdapter(spec, context)


def _checkpoint_complexity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "unavailable", "parameter_count": None}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", {}) if isinstance(checkpoint, dict) else {}
        count = sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))
        return {"status": "measured_from_checkpoint", "parameter_count": count}
    except Exception as exc:
        return {"status": "unavailable", "parameter_count": None, "reason": str(exc)}


def _pretrained_asset_hashes(manifest: dict[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("pretrained_asset_provenance", {}) or {}
    output: dict[str, Any] = {}
    for name in ("vision", "text"):
        payload = provenance.get(name, {}) if isinstance(provenance, dict) else {}
        if isinstance(payload, dict):
            output[name] = payload.get("sha256") or payload.get("checkpoint_sha256")
    return output


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ImportError:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["BuiltinBaselineAdapter", "BuiltinFrameworkAdapter", "create_adapter"]
