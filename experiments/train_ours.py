"""Training and evaluation runner for the full proposed framework."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dataset import MemeDataset, NormalizedMemeDataset
from experiments.early_stopping import EarlyStopping, metric_from_validation, prefix_metrics, save_checkpoint, save_training_log
from experiments.metrics import compute_harmfulness_metrics
from experiments.prediction_io import save_predictions_and_metrics, stage_outputs_to_prediction_record
from experiments.progress import progress_iter
from experiments.splits import build_splits_for_dataset, label_to_int, load_split_file, save_splits, split_samples
from experiments.structured_eval import evaluate_structured_predictions
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from module.pipeline.model import HarmfulMemePipeline
from utils.io import load_yaml
from utils.seed import set_seed


@dataclass
class OursRunConfig:
    """Runtime configuration for Ours Full experiments."""

    dataset_name: str
    seed: int = 42
    config_path: str = "configs/default.yaml"
    split_file: str | None = None
    output_root: str = "result"
    model_name: str = "ours_full"
    epochs: int = 5
    lr: float = 1e-4
    patience: int = 3
    min_delta: float = 0.0
    early_stop_metric: str = "val_macro_f1"
    early_stop_mode: str = "max"
    save_best: bool = True
    save_last: bool = True
    disable_tqdm: bool = False
    print_components: bool = False
    device: str = "cpu"
    limit: int | None = None
    freeze_backbones: bool = True
    train_relevance_mlp: bool = True
    structured_auxiliary: bool = True
    normalized_root: str = "dataset/annotation_normalized"
    label_set: str = "full"
    vocab_path: str = "configs/label_vocab.yaml"
    use_normalized_labels: bool = True
    require_normalized_label: bool = True
    use_sample_weight: bool = True


def run_ours_experiment(config: OursRunConfig) -> dict[str, Any]:
    """Train/evaluate HarmfulMemePipeline on one dataset split."""

    set_seed(config.seed)
    cfg = load_yaml(config.config_path)
    dataset = _load_ours_dataset(config, cfg)
    samples = _materialize_ours_samples(dataset, prefer_normalized=config.use_normalized_labels)
    split_dataset = dataset.base_dataset if isinstance(dataset, NormalizedMemeDataset) else dataset
    if config.use_normalized_labels and not samples:
        print(
            f"[normalized-labels] No usable normalized labels for {config.dataset_name}; "
            "falling back to raw_label supervision."
        )
        dataset = _load_legacy_dataset(config, cfg)
        split_dataset = dataset
        samples = _materialize_ours_samples(dataset, prefer_normalized=False)
    splits = _load_or_create_splits(config, split_dataset, cfg)
    materialized = split_samples(samples, splits)

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg.setdefault("runtime", {})["device"] = str(device)
    pipeline = HarmfulMemePipeline(cfg).to(device)
    if config.print_components:
        from experiments.components import print_pipeline_components

        print_pipeline_components(pipeline)
    configure_trainable_parameters(pipeline, config)
    optimizer = torch.optim.AdamW([param for param in pipeline.parameters() if param.requires_grad], lr=config.lr)
    loss_fn = StructuredMemeLoss()
    output_dir = Path(config.output_root) / "predictions" / config.dataset_name / config.model_name / str(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_state: dict[str, torch.Tensor] | None = None
    stopper = EarlyStopping(config.patience, config.min_delta, config.early_stop_mode)
    validation_samples = materialized.get("valid", [])
    early_stopping_active = stopper.enabled and bool(validation_samples)
    if stopper.enabled and not validation_samples:
        print(f"[early-stopping] Validation split is empty for {config.dataset_name}/{config.model_name}; training for all {config.epochs} epochs.")
    training_log: list[dict[str, Any]] = []
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    epoch_iter = progress_iter(
        range(1, config.epochs + 1),
        desc=f"{config.model_name} {config.dataset_name} seed={config.seed}",
        disable=config.disable_tqdm,
    )
    for epoch in epoch_iter:
        pipeline.train()
        epoch_losses: list[float] = []
        component_sums: dict[str, float] = defaultdict(float)
        component_counts: dict[str, int] = defaultdict(int)
        component_grad_counts: dict[str, int] = defaultdict(int)
        component_provenance: dict[str, str] = {}
        component_expected_grad: dict[str, bool] = {}
        train_iter = progress_iter(
            materialized.get("train", []),
            desc=f"train epoch {epoch}/{config.epochs}",
            disable=config.disable_tqdm,
            leave=False,
        )
        for sample in train_iter:
            outputs = pipeline(sample)
            stage_e = outputs["stage_e"]
            supervision = extract_supervision_from_annotation(sample)
            if not config.structured_auxiliary:
                supervision = {
                    "harmfulness": supervision.get("harmfulness"),
                    "sample_weight": supervision.get("sample_weight", sample.get("sample_weight", 1.0)),
                }
            losses = loss_fn(stage_e, supervision)
            for name, description in loss_fn.describe_losses(losses).items():
                if name == "total":
                    continue
                component_sums[name] += float(description["value"])
                component_counts[name] += 1
                component_grad_counts[name] += int(bool(description["requires_grad"]))
                component_provenance[name] = str(description["provenance"])
                component_expected_grad[name] = bool(description["differentiable_expected"])
            loss = losses.get("total")
            if loss is None:
                loss = F.cross_entropy(
                    stage_e.harmfulness.logits.unsqueeze(0),
                    torch.tensor([sample["label"]], device=stage_e.harmfulness.logits.device),
                )
            if config.use_sample_weight:
                weight = _sample_weight(supervision, sample)
                loss = loss * loss.new_tensor(weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(loss=f"{epoch_losses[-1]:.4f}")
        valid_metrics, _ = evaluate_ours_pipeline(pipeline, validation_samples, config, device=device, desc="valid")
        current_metric = metric_from_validation(
            valid_metrics,
            config.early_stop_metric,
            structured_score=config.early_stop_metric in {"val_structured_score", "structured_score"},
        )
        status = stopper.step(current_metric, epoch, active=early_stopping_active)
        if status["is_best"]:
            best_state = {key: value.detach().cpu().clone() for key, value in pipeline.state_dict().items()}
            if config.save_best:
                save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    model=pipeline,
                    optimizer=optimizer,
                    best_metric=stopper.best_metric,
                    config=config,
                    seed=config.seed,
                    dataset=config.dataset_name,
                    model_name=config.model_name,
                )
        loss_components = {
            name: component_sums[name] / component_counts[name]
            for name in sorted(component_counts)
            if component_counts[name]
        }
        loss_provenance = {
            name: {
                "provenance": component_provenance[name],
                "differentiable_expected": component_expected_grad[name],
                "mean_requires_grad": component_grad_counts[name] / component_counts[name],
            }
            for name in sorted(component_counts)
            if component_counts[name]
        }
        active_logits_losses = sorted(
            name for name, provenance in component_provenance.items() if provenance.startswith("logits")
        )
        active_proxy_losses = sorted(
            name for name, provenance in component_provenance.items() if provenance.startswith("proxy_")
        )
        log_row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
            "loss_components": loss_components,
            "loss_provenance": loss_provenance,
            "active_logits_losses": active_logits_losses,
            "active_proxy_losses": active_proxy_losses,
            "active_logits_loss_count": len(active_logits_losses),
            "active_proxy_loss_count": len(active_proxy_losses),
            "early_stop_metric": config.early_stop_metric,
            "early_stop_metric_value": current_metric,
            **prefix_metrics(valid_metrics),
            **status,
        }
        training_log.append(log_row)
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                loss=f"{log_row['train_loss']:.4f}",
                metric=current_metric,
                best=stopper.best_metric,
                patience=stopper.counter,
            )
        if status["stopped_early"]:
            break

    if config.save_last:
        save_checkpoint(
            last_checkpoint_path,
            epoch=training_log[-1]["epoch"] if training_log else 0,
            model=pipeline,
            optimizer=optimizer,
            best_metric=stopper.best_metric,
            config=config,
            seed=config.seed,
            dataset=config.dataset_name,
            model_name=config.model_name,
        )
    if best_state is not None:
        pipeline.load_state_dict(best_state)
    elif config.save_best:
        save_checkpoint(
            best_checkpoint_path,
            epoch=0,
            model=pipeline,
            optimizer=optimizer,
            best_metric=stopper.best_metric,
            config=config,
            seed=config.seed,
            dataset=config.dataset_name,
            model_name=config.model_name,
        )
    if training_log:
        training_log[-1]["stopped_early"] = bool(stopper.stopped_early)
    save_training_log(output_dir, training_log)
    metrics, predictions = evaluate_ours_pipeline(pipeline, materialized.get("test", []), config, device=device, desc="test")
    save_predictions_and_metrics(output_dir, predictions, metrics)
    print(
        f"[early-stopping] {config.dataset_name}/{config.model_name}/seed={config.seed}: "
        f"best_epoch={stopper.best_epoch} best_{config.early_stop_metric}={stopper.best_metric} "
        f"stopped_early={stopper.stopped_early} checkpoint={best_checkpoint_path if config.save_best else 'disabled'}"
    )
    return metrics


@torch.no_grad()
def evaluate_ours_pipeline(
    pipeline: HarmfulMemePipeline,
    samples: list[dict[str, Any]],
    config: OursRunConfig,
    device: torch.device | None = None,
    desc: str = "eval",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a full pipeline and return combined harmfulness/structured metrics."""

    _ = device
    pipeline.eval()
    predictions: list[dict[str, Any]] = []
    for sample in progress_iter(samples, desc=desc, disable=config.disable_tqdm, leave=False):
        outputs = pipeline(sample)
        predictions.append(stage_outputs_to_prediction_record(sample, outputs, model_name=config.model_name, seed=config.seed))
    harmfulness = compute_harmfulness_metrics(
        [record["gold_label"] for record in predictions if record.get("gold_label") is not None],
        [record["pred_label"] for record in predictions if record.get("gold_label") is not None],
        [record["prob_harmful"] for record in predictions if record.get("gold_label") is not None],
    )
    structured = evaluate_structured_predictions(predictions)
    metrics = {**harmfulness, **structured}
    return metrics, predictions


def configure_trainable_parameters(pipeline: HarmfulMemePipeline, config: OursRunConfig) -> None:
    """Freeze heavy components and unfreeze lightweight framework modules."""

    for param in pipeline.parameters():
        param.requires_grad = False
    for module in [pipeline.stage_d, pipeline.stage_e]:
        for param in module.parameters():
            param.requires_grad = True
    if config.train_relevance_mlp and hasattr(pipeline.stage_c.relevance, "feature_mlp"):
        for param in pipeline.stage_c.relevance.feature_mlp.parameters():
            param.requires_grad = True
    if not config.freeze_backbones:
        for param in pipeline.parameters():
            param.requires_grad = True


def _load_or_create_splits(config: OursRunConfig, dataset: Any, cfg: dict[str, Any]) -> dict[str, list[str]]:
    if config.split_file:
        return load_split_file(config.split_file)
    split_path = Path(config.output_root) / "splits" / config.dataset_name / f"seed_{config.seed}.json"
    if split_path.exists():
        return load_split_file(split_path)
    splits = build_splits_for_dataset(
        config.dataset_name,
        dataset,
        seed=config.seed,
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
    )
    save_splits(splits, config.dataset_name, config.seed, Path(config.output_root) / "splits")
    return splits


def _load_ours_dataset(config: OursRunConfig, cfg: dict[str, Any]) -> Any:
    if config.use_normalized_labels:
        return NormalizedMemeDataset(
            dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
            annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
            normalized_root=cfg.get("paths", {}).get("normalized_root", config.normalized_root),
            dataset_names=[config.dataset_name],
            label_set=config.label_set,
            vocab_path=config.vocab_path,
            keep_missing_images=True,
            limit=config.limit,
            require_normalized_label=config.require_normalized_label,
        )
    return _load_legacy_dataset(config, cfg)


def _load_legacy_dataset(config: OursRunConfig, cfg: dict[str, Any]) -> MemeDataset:
    return MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[config.dataset_name],
        keep_missing_images=True,
        limit=config.limit,
    )


def _materialize_ours_samples(dataset: Any, prefer_normalized: bool) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample in dataset:
        label = _normalized_harmfulness_label(sample) if prefer_normalized else None
        if label is None:
            label = label_to_int(sample.get("raw_label"))
        if label is None:
            continue
        samples.append(dict(sample, label=int(label)))
    return samples


def _normalized_harmfulness_label(sample: dict[str, Any]) -> int | None:
    targets = sample.get("targets") or {}
    class_ids = targets.get("class_ids") or {}
    masks = targets.get("masks") or {}
    if "harmfulness" not in class_ids or int(masks.get("harmfulness", 1) or 0) == 0:
        return None
    try:
        label = int(class_ids["harmfulness"])
    except (TypeError, ValueError):
        return None
    return label if label in {0, 1} else None


def _sample_weight(supervision: dict[str, Any], sample: dict[str, Any]) -> float:
    value = supervision.get("sample_weight", sample.get("sample_weight", 1.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0
