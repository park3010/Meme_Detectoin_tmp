"""Config-driven pipeline runner and result saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dataset import MemeDataset
from module.pipeline.model import HarmfulMemePipeline
from utils.io import deep_update, load_yaml, write_json, write_jsonl
from utils.logging_utils import setup_logger
from utils.seed import set_seed
from utils.tensor_utils import tensor_to_python


logger = setup_logger(__name__)


class PipelineRunner:
    """Load config/data, execute stages, and save outputs."""

    def __init__(self, config_path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> None:
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        if overrides:
            self.config = deep_update(self.config, overrides)
        set_seed(int(self.config.get("runtime", {}).get("seed", 13)))
        self.pipeline = HarmfulMemePipeline(self.config).eval()
        self.result_root = Path(self.config.get("paths", {}).get("result_root", "result"))

    def load_dataset(self, dataset_names: list[str] | None = None, limit: int | None = None) -> MemeDataset:
        """Load configured dataset."""

        runtime_limit = self.config.get("runtime", {}).get("max_samples")
        effective_limit = limit if limit is not None else runtime_limit
        return MemeDataset(
            dataset_root=self.config.get("paths", {}).get("dataset_root", "dataset/V1"),
            annotation_root=self.config.get("paths", {}).get("annotation_root", "outputs"),
            dataset_names=dataset_names,
            keep_missing_images=True,
            limit=int(effective_limit) if effective_limit else None,
        )

    @torch.no_grad()
    def run(
        self,
        dataset_names: list[str] | None = None,
        limit: int | None = None,
        run_until: str = "e",
        save: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the configured pipeline and optionally save stage outputs."""

        dataset = self.load_dataset(dataset_names=dataset_names, limit=limit)
        logger.info("Running pipeline on %s samples through Stage %s", len(dataset), run_until.upper())
        records: list[dict[str, Any]] = []
        for sample in dataset:
            outputs = self.pipeline(sample, run_until=run_until)
            records.append({"sample": sample, "outputs": outputs})
        if save:
            self.save_outputs(records, run_until=run_until)
        return records

    def save_outputs(self, records: list[dict[str, Any]], run_until: str = "e") -> None:
        """Save readable per-stage artifacts under result/."""

        stage_order = ["a", "b", "c", "d", "e"]
        active = set(stage_order[: stage_order.index(run_until.lower()) + 1])
        if "a" in active:
            write_jsonl(
                self.result_root / "stage_a" / "internal_evidence.jsonl",
                [record["outputs"]["stage_a"] for record in records if "stage_a" in record["outputs"]],
            )
        if "b" in active:
            write_jsonl(
                self.result_root / "stage_b" / "knowledge_candidates.jsonl",
                [record["outputs"]["stage_b"] for record in records if "stage_b" in record["outputs"]],
            )
        if "c" in active:
            write_jsonl(
                self.result_root / "stage_c" / "verified_knowledge.jsonl",
                [record["outputs"]["stage_c"] for record in records if "stage_c" in record["outputs"]],
            )
        if "d" in active:
            self._save_stage_d_pt(records)
        if "e" in active:
            write_jsonl(
                self.result_root / "stage_e" / "final_predictions.jsonl",
                [record["outputs"]["stage_e"].structured_prediction for record in records if "stage_e" in record["outputs"]],
                compact_tensors=True,
            )
            self._save_analysis_exports(records)
        if self.config.get("runtime", {}).get("save_per_sample", False):
            for record in records:
                sample_id = record["sample"]["sample_id"]
                write_json(self.result_root / "samples" / f"{sample_id}.json", record)

    def _save_stage_d_pt(self, records: list[dict[str, Any]]) -> None:
        path = self.result_root / "stage_d" / "fusion_states.pt"
        legacy_path = self.result_root / "stage_d" / "fusion_outputs.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in records:
            output = record["outputs"].get("stage_d")
            if output is None:
                continue
            payload.append(
                {
                    "sample_id": output.sample_id,
                    "dataset_name": output.dataset_name,
                    "shared_reasoning_state": output.shared_reasoning_state.detach().cpu(),
                    "internal_memory": output.internal_memory.detach().cpu(),
                    "fused_tokens": output.fused_tokens.detach().cpu(),
                    "cross_attention_weights": output.cross_attention_weights.detach().cpu(),
                    "task_latents": {key: value.detach().cpu() for key, value in output.task_latents.items()},
                    "gates": tensor_to_python(output.gates),
                    "metadata": tensor_to_python(output.metadata),
                }
            )
        torch.save(payload, path)
        torch.save(payload, legacy_path)
        logger.info("Saved Stage D tensor artifact to %s", path)

    def _save_analysis_exports(self, records: list[dict[str, Any]]) -> None:
        attribution_records = []
        summary_records = []
        for record in records:
            stage_e = record["outputs"].get("stage_e")
            if stage_e is None:
                continue
            attribution_records.append(
                {
                    "sample_id": stage_e.sample_id,
                    "dataset_name": stage_e.dataset_name,
                    "supporting_evidence": stage_e.supporting_evidence,
                }
            )
            structured = stage_e.structured_prediction
            summary_records.append(
                {
                    "sample_id": stage_e.sample_id,
                    "dataset_name": stage_e.dataset_name,
                    "harmfulness": structured.get("harmfulness", {}).get("label"),
                    "target": structured.get("target", {}).get("granularity"),
                    "intent": structured.get("intent", {}).get("primary"),
                    "tactic": structured.get("tactic", {}).get("label"),
                    "rationale": structured.get("rationale", ""),
                }
            )
        write_jsonl(self.result_root / "analysis" / "evidence_attribution.jsonl", attribution_records)
        write_jsonl(self.result_root / "analysis" / "sample_summaries.jsonl", summary_records)
