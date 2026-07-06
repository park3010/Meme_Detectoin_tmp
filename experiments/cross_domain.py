"""Cross-dataset robustness experiments for framework predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dataset import MemeDataset
from experiments.evaluation import compute_harmfulness_metrics
from experiments.prediction_io import save_predictions_and_metrics, stage_outputs_to_prediction_record
from experiments.progress import ProgressConfig, progress_iter
from experiments.splits import build_splits_for_dataset, label_to_int, load_split_file, save_splits, split_samples
from experiments.evaluation import evaluate_structured_predictions
from experiments.train import OursRunConfig, configure_trainable_parameters
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from module.runner import HarmfulMemePipeline
from utils.io import load_yaml
from utils.seed import set_seed


DATASETS = ["harm_c", "harm_p", "facebook", "memotion"]
SETTINGS = ["in_domain", "train_one_test_others", "mixed_train", "leave_one_domain_out"]


@dataclass
class CrossDomainConfig:
    """Runtime options for one cross-domain robustness run."""

    setting: str
    model_name: str = "ours_full"
    seed: int = 42
    config_path: str = "configs/config.yaml"
    output_root: str = "result"
    train_dataset: str | None = None
    heldout: str | None = None
    test_dataset: str | None = None
    epochs: int = 0
    lr: float = 1e-4
    device: str = "cpu"
    limit: int | None = None
    split_file: str | None = None
    disable_tqdm: bool = False
    progress: ProgressConfig | None = None


def run_cross_domain(config: CrossDomainConfig) -> list[dict[str, Any]]:
    """Run a cross-domain setting and append metric rows."""

    if config.setting not in SETTINGS:
        raise ValueError(f"Unsupported cross-domain setting: {config.setting}")
    rows: list[dict[str, Any]] = []
    plan = _plan_runs(config)
    progress_config = _progress_config(config)
    for train_domains, test_domain, label in progress_iter(
        plan,
        desc=f"cross-domain {config.setting}",
        config=progress_config,
        position=1,
        leave=progress_config.leave_epoch,
    ):
        predictions, metrics = _run_train_test(config, train_domains, test_domain)
        metrics["performance_drop_vs_in_domain"] = _performance_drop(config, test_domain, metrics)
        output_dir = (
            Path(config.output_root)
            / "predictions_cross_domain"
            / config.setting
            / label
            / test_domain
            / config.model_name
            / str(config.seed)
        )
        save_predictions_and_metrics(output_dir, predictions, metrics)
        row = _metric_row(config, label, test_domain, metrics)
        rows.append(row)
        _append_row(Path(config.output_root) / "metrics" / "cross_domain.csv", row)
    return rows


def _plan_runs(config: CrossDomainConfig) -> list[tuple[list[str], str, str]]:
    if config.setting == "in_domain":
        domains = [config.train_dataset] if config.train_dataset else DATASETS
        return [([domain], domain, domain) for domain in domains if domain]
    if config.setting == "train_one_test_others":
        train = config.train_dataset or "harm_c"
        return [([train], test, train) for test in DATASETS if test != train]
    if config.setting == "mixed_train":
        return [(DATASETS, test, "mixed") for test in DATASETS]
    heldout = config.heldout or "harm_c"
    return [([domain for domain in DATASETS if domain != heldout], heldout, heldout)]


def _run_train_test(config: CrossDomainConfig, train_domains: list[str], test_domain: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(config.seed)
    cfg = load_yaml(config.config_path)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg.setdefault("runtime", {})["device"] = str(device)
    dataset_names = sorted(set(train_domains + [test_domain]))
    dataset = MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=dataset_names,
        keep_missing_images=True,
        limit=config.limit,
    )
    samples = [dict(sample, label=label_to_int(sample.get("raw_label"))) for sample in dataset if label_to_int(sample.get("raw_label")) is not None]
    by_dataset = {name: [sample for sample in samples if sample.get("dataset_name") == name] for name in dataset_names}
    splits = {name: _load_or_create_split(config, dataset, cfg, name) for name in dataset_names}
    materialized = {name: split_samples(by_dataset[name], splits[name]) for name in dataset_names}
    train_samples = [sample for domain in train_domains for sample in materialized.get(domain, {}).get("train", [])]
    valid_samples = [sample for domain in train_domains for sample in materialized.get(domain, {}).get("valid", [])]
    test_samples = materialized.get(test_domain, {}).get("test", [])

    pipeline = HarmfulMemePipeline(cfg).to(device)
    if config.model_name not in {"ours_full", "full"}:
        raise ValueError("Cross-domain runner currently supports model_name='ours_full' for fresh pipeline execution.")
    if config.epochs > 0:
        _train_lightweight_pipeline(pipeline, train_samples, valid_samples, config, device)
    predictions = _evaluate_pipeline(pipeline, test_samples, config)
    metrics = _combined_metrics(predictions)
    return predictions, metrics


def _train_lightweight_pipeline(
    pipeline: HarmfulMemePipeline,
    train_samples: list[dict[str, Any]],
    valid_samples: list[dict[str, Any]],
    config: CrossDomainConfig,
    device: torch.device,
) -> None:
    progress_config = _progress_config(config)
    train_cfg = OursRunConfig(
        dataset_name="mixed",
        seed=config.seed,
        config_path=config.config_path,
        epochs=config.epochs,
        lr=config.lr,
        progress=progress_config,
    )
    configure_trainable_parameters(pipeline, train_cfg)
    params = [param for param in pipeline.parameters() if param.requires_grad]
    if not params:
        return
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    loss_fn = StructuredMemeLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("-inf")
    for epoch in progress_iter(
        range(1, config.epochs + 1),
        desc="cross-domain train",
        config=progress_config,
        position=1,
        leave=progress_config.leave_epoch,
    ):
        pipeline.train()
        for sample in progress_iter(
            train_samples,
            desc=f"train epoch {epoch}/{config.epochs}",
            config=progress_config,
            position=2,
            leave=progress_config.leave_batch,
        ):
            outputs = pipeline(sample)
            stage_e = outputs["stage_e"]
            losses = loss_fn(stage_e, extract_supervision_from_annotation(sample))
            fallback = F.cross_entropy(stage_e.harmfulness.logits.unsqueeze(0), torch.tensor([sample["label"]], device=device))
            loss = losses.get("total", fallback)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        valid_metrics = _combined_metrics(_evaluate_pipeline(pipeline, valid_samples, config)) if valid_samples else {"macro_f1": 0.0}
        score = valid_metrics.get("macro_f1") or valid_metrics.get("harmfulness_macro_f1") or 0.0
        if score >= best_valid:
            best_valid = score
            best_state = {key: value.detach().cpu().clone() for key, value in pipeline.state_dict().items()}
    if best_state is not None:
        pipeline.load_state_dict(best_state)


@torch.no_grad()
def _evaluate_pipeline(pipeline: HarmfulMemePipeline, samples: list[dict[str, Any]], config: CrossDomainConfig) -> list[dict[str, Any]]:
    pipeline.eval()
    progress_config = _progress_config(config)
    return [
        stage_outputs_to_prediction_record(sample, pipeline(sample), model_name=config.model_name, seed=config.seed)
        for sample in progress_iter(
            samples,
            desc=f"cross-domain eval {config.model_name}",
            config=progress_config,
            position=2,
            leave=progress_config.leave_batch,
        )
    ]


def _combined_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [record for record in predictions if record.get("gold_label") is not None]
    harmfulness = compute_harmfulness_metrics(
        [record["gold_label"] for record in labeled],
        [record["pred_label"] for record in labeled],
        [record.get("prob_harmful", 0.0) for record in labeled],
    )
    structured = evaluate_structured_predictions(predictions)
    return {**harmfulness, **structured}


def _load_or_create_split(config: CrossDomainConfig, dataset: MemeDataset, cfg: dict[str, Any], dataset_name: str) -> dict[str, list[str]]:
    if config.split_file and len(set(_plan_domains_for_split_file(config))) == 1:
        return load_split_file(config.split_file)
    split_path = Path(config.output_root) / "splits" / dataset_name / f"seed_{config.seed}.json"
    if split_path.exists():
        return load_split_file(split_path)
    splits = build_splits_for_dataset(
        dataset_name,
        dataset,
        seed=config.seed,
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
    )
    save_splits(splits, dataset_name, config.seed, Path(config.output_root) / "splits")
    return splits


def _plan_domains_for_split_file(config: CrossDomainConfig) -> list[str]:
    return [domain for train, test, _ in _plan_runs(config) for domain in [*train, test]]


def _performance_drop(config: CrossDomainConfig, test_domain: str, metrics: dict[str, Any]) -> float | None:
    path = Path(config.output_root) / "metrics" / "cross_domain.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("setting") == "in_domain" and row.get("test_dataset") == test_domain and row.get("model") == config.model_name:
                try:
                    baseline = float(row.get("harmfulness_macro_f1") or 0.0)
                    current = float(metrics.get("macro_f1") or metrics.get("harmfulness_macro_f1") or 0.0)
                    return baseline - current
                except ValueError:
                    return None
    return None


def _metric_row(config: CrossDomainConfig, train_label: str, test_domain: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "setting": config.setting,
        "train_dataset_or_heldout": train_label,
        "test_dataset": test_domain,
        "model": config.model_name,
        "seed": config.seed,
        "harmfulness_macro_f1": metrics.get("macro_f1") or metrics.get("harmfulness_macro_f1"),
        "target_macro_f1": metrics.get("target_granularity_macro_f1"),
        "intent_macro_f1": metrics.get("intent_primary_macro_f1"),
        "tactic_macro_f1": metrics.get("tactic_multimodal_relation_macro_f1"),
        "performance_drop_vs_in_domain": metrics.get("performance_drop_vs_in_domain"),
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _progress_config(config: CrossDomainConfig) -> ProgressConfig:
    return config.progress or ProgressConfig(disable=True if config.disable_tqdm else None)
