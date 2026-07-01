"""Unified training runners for Ours Full and simple baselines."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dataset import MemeDataset, NormalizedMemeDataset
from experiments.early_stopping import EarlyStopping, metric_from_validation, prefix_metrics, save_checkpoint, save_training_log
from experiments.evaluation import compute_harmfulness_metrics, evaluate_structured_predictions
from experiments.prediction_io import save_predictions_and_metrics, stage_outputs_to_prediction_record
from experiments.progress import progress_iter
from experiments.splits import build_splits_for_dataset, label_to_int, load_split_file, save_splits, split_samples
from module.baseline import CLIPTextConcatClassifier, ImageOnlyCLIPClassifier, TextOnlyEncoderClassifier
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from module.runner import HarmfulMemePipeline
from utils.io import load_yaml, write_json, write_jsonl
from utils.seed import set_seed


# =============================================================================
# Ours Full training
# =============================================================================

@dataclass
class OursRunConfig:
    """Runtime configuration for Ours Full experiments."""

    dataset_name: str
    seed: int = 42
    config_path: str = "configs/config.yaml"
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
        dataset = _load_ours_legacy_dataset(config, cfg)
        split_dataset = dataset
        samples = _materialize_ours_samples(dataset, prefer_normalized=False)
    splits = _load_or_create_ours_splits(config, split_dataset, cfg)
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
            "split_sizes": {
                "train": len(materialized.get("train", [])),
                "valid": len(validation_samples),
                "test": len(materialized.get("test", [])),
            },
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


def _load_or_create_ours_splits(config: OursRunConfig, dataset: Any, cfg: dict[str, Any]) -> dict[str, list[str]]:
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
    return _load_ours_legacy_dataset(config, cfg)


def _load_ours_legacy_dataset(config: OursRunConfig, cfg: dict[str, Any]) -> MemeDataset:
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
        label = _normalized_ours_harmfulness_label(sample) if prefer_normalized else None
        if label is None:
            label = label_to_int(sample.get("raw_label"))
        if label is None:
            continue
        samples.append(dict(sample, label=int(label)))
    return samples


def _normalized_ours_harmfulness_label(sample: dict[str, Any]) -> int | None:
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


# =============================================================================
# Baseline training
# =============================================================================

@dataclass
class BaselineRunConfig:
    """Runtime configuration for one baseline experiment."""

    model_name: str
    dataset_name: str
    seed: int = 42
    config_path: str = "configs/config.yaml"
    split_file: str | None = None
    output_root: str = "result"
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    patience: int = 3
    min_delta: float = 0.0
    early_stop_metric: str = "val_macro_f1"
    early_stop_mode: str = "max"
    save_best: bool = True
    save_last: bool = True
    disable_tqdm: bool = False
    device: str = "cpu"
    limit: int | None = None
    normalized_root: str = "dataset/annotation_normalized"
    label_set: str = "full"
    vocab_path: str = "configs/label_vocab.yaml"
    use_normalized_labels: bool = False
    require_normalized_label: bool = True
    use_sample_weight: bool = False


def run_baseline_experiment(config: BaselineRunConfig) -> dict[str, Any]:
    """Train, validate, and test a simple harmfulness baseline."""

    set_seed(config.seed)
    cfg = load_yaml(config.config_path)
    dataset = _load_baseline_dataset(config, cfg)
    sample_dicts = []
    for sample in dataset:
        try:
            sample_dicts.append(_normalize_sample(sample))
        except ValueError:
            continue
    split_dataset = dataset.base_dataset if isinstance(dataset, NormalizedMemeDataset) else dataset
    if config.use_normalized_labels and not sample_dicts:
        print(
            f"[normalized-labels] No usable normalized labels for {config.dataset_name}; "
            "falling back to raw_label supervision."
        )
        dataset = _load_baseline_legacy_dataset(config, cfg)
        split_dataset = dataset
        sample_dicts = []
        for sample in dataset:
            try:
                sample_dicts.append(_normalize_sample(sample))
            except ValueError:
                continue
    splits = _load_or_create_baseline_splits(config, split_dataset)
    materialized = split_samples(sample_dicts, splits)

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = create_baseline_model(config.model_name, cfg, device=str(device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
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
        train_loss = train_one_epoch(
            model,
            materialized.get("train", []),
            optimizer,
            config.batch_size,
            device,
            seed=config.seed + epoch,
            disable_tqdm=config.disable_tqdm,
            desc=f"train epoch {epoch}/{config.epochs}",
            use_sample_weight=config.use_sample_weight,
        )
        valid_metrics, _ = evaluate_model(model, validation_samples, config.batch_size, device, disable_tqdm=config.disable_tqdm, desc="valid")
        current_metric = metric_from_validation(valid_metrics, config.early_stop_metric)
        status = stopper.step(current_metric, epoch, active=early_stopping_active)
        if status["is_best"]:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if config.save_best:
                save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_metric=stopper.best_metric,
                    config=config,
                    seed=config.seed,
                    dataset=config.dataset_name,
                    model_name=config.model_name,
                )
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "early_stop_metric": config.early_stop_metric,
            "early_stop_metric_value": current_metric,
            **prefix_metrics(valid_metrics),
            **status,
        }
        training_log.append(log_row)
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                loss=f"{train_loss:.4f}",
                metric=current_metric,
                best=stopper.best_metric,
                patience=stopper.counter,
            )
        if status["stopped_early"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif config.save_best:
        save_checkpoint(
            best_checkpoint_path,
            epoch=0,
            model=model,
            optimizer=optimizer,
            best_metric=stopper.best_metric,
            config=config,
            seed=config.seed,
            dataset=config.dataset_name,
            model_name=config.model_name,
        )
    if config.save_last:
        save_checkpoint(
            last_checkpoint_path,
            epoch=training_log[-1]["epoch"] if training_log else 0,
            model=model,
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
    test_metrics, predictions = evaluate_model(model, materialized.get("test", []), config.batch_size, device, disable_tqdm=config.disable_tqdm, desc="test")
    save_baseline_outputs(config, model, predictions, test_metrics)
    print(
        f"[early-stopping] {config.dataset_name}/{config.model_name}/seed={config.seed}: "
        f"best_epoch={stopper.best_epoch} best_{config.early_stop_metric}={stopper.best_metric} "
        f"stopped_early={stopper.stopped_early} checkpoint={best_checkpoint_path if config.save_best else 'disabled'}"
    )
    return test_metrics


def create_baseline_model(model_name: str, cfg: dict[str, Any] | None = None, device: str = "cpu"):
    """Instantiate one supported baseline model."""

    cfg = cfg or {}
    model_cfg = cfg.get("model", {})
    backbone_cfg = cfg.get("backbone", cfg.get("backbones", {}))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    clip_cfg = backbone_cfg.get("clip", {})
    text_cfg = backbone_cfg.get("text", {})
    if model_name == "image_only_clip":
        return ImageOnlyCLIPClassifier(
            hidden_dim=hidden_dim,
            prefer_pretrained_clip=bool(clip_cfg.get("prefer_pretrained", False)),
            clip_model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
            device=device,
        )
    if model_name == "text_only_encoder":
        return TextOnlyEncoderClassifier(
            hidden_dim=hidden_dim,
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            device=device,
        )
    if model_name == "clip_text_concat":
        return CLIPTextConcatClassifier(
            hidden_dim=hidden_dim,
            prefer_pretrained_clip=bool(clip_cfg.get("prefer_pretrained", False)),
            prefer_transformers=bool(text_cfg.get("prefer_transformers", False)),
            clip_model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
            text_model_name=str(text_cfg.get("model_name", "microsoft/deberta-v3-base")),
            device=device,
        )
    raise ValueError(f"Unsupported baseline model: {model_name}")


def train_one_epoch(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    seed: int,
    disable_tqdm: bool = False,
    desc: str = "train",
    use_sample_weight: bool = False,
) -> float:
    """Train for one epoch and return mean loss."""

    if not samples:
        return 0.0
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    model.train()
    losses: list[float] = []
    batches = list(_batches(shuffled, batch_size))
    batch_iter = progress_iter(batches, desc=desc, disable=disable_tqdm, leave=False)
    for batch in batch_iter:
        labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long, device=device)
        output = model(
            image_paths=[sample.get("image_path") for sample in batch],
            ocr_texts=[sample.get("ocr_text_full", "") for sample in batch],
        )
        if use_sample_weight:
            per_sample_loss = F.cross_entropy(output["logits"], labels, reduction="none")
            weights = torch.tensor([float(sample.get("sample_weight", 1.0)) for sample in batch], dtype=per_sample_loss.dtype, device=device)
            loss = (per_sample_loss * weights).mean()
        else:
            loss = F.cross_entropy(output["logits"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(loss=f"{losses[-1]:.4f}")
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    disable_tqdm: bool = False,
    desc: str = "eval",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a baseline and return metrics plus prediction records."""

    if not samples:
        return compute_harmfulness_metrics([], [], []), []
    model.eval()
    predictions: list[dict[str, Any]] = []
    batches = list(_batches(samples, batch_size))
    for batch in progress_iter(batches, desc=desc, disable=disable_tqdm, leave=False):
        output = model(
            image_paths=[sample.get("image_path") for sample in batch],
            ocr_texts=[sample.get("ocr_text_full", "") for sample in batch],
        )
        logits = output["logits"].detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        for idx, sample in enumerate(batch):
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "dataset_name": sample["dataset_name"],
                    "gold_label": int(sample["label"]),
                    "pred_label": int(pred[idx]),
                    "prob_harmful": float(probs[idx, 1]),
                    "logits": [float(value) for value in logits[idx].tolist()],
                }
            )
    metrics = compute_harmfulness_metrics(
        [record["gold_label"] for record in predictions],
        [record["pred_label"] for record in predictions],
        [record["prob_harmful"] for record in predictions],
    )
    return metrics, predictions


def save_baseline_outputs(
    config: BaselineRunConfig,
    model: torch.nn.Module,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Save predictions and metrics."""

    output_dir = Path(config.output_root) / "predictions" / config.dataset_name / config.model_name / str(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "final_predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: metrics.get(column) for column in columns})


def _load_or_create_baseline_splits(config: BaselineRunConfig, dataset: Any) -> dict[str, list[str]]:
    if config.split_file:
        return load_split_file(config.split_file)
    split_path = Path(config.output_root) / "splits" / config.dataset_name / f"seed_{config.seed}.json"
    if split_path.exists():
        return load_split_file(split_path)
    cfg = load_yaml(config.config_path)
    splits = build_splits_for_dataset(
        config.dataset_name,
        dataset,
        seed=config.seed,
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
    )
    save_splits(splits, config.dataset_name, config.seed, output_root=Path(config.output_root) / "splits")
    return splits


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    label = _normalized_baseline_harmfulness_label(sample)
    if label is None:
        label = label_to_int(sample.get("raw_label"))
    if label is None:
        raise ValueError(f"Sample {sample.get('sample_id')} has no binary label")
    output = dict(sample)
    output["label"] = label
    return output


def _load_baseline_dataset(config: BaselineRunConfig, cfg: dict[str, Any]) -> Any:
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
    return _load_baseline_legacy_dataset(config, cfg)


def _load_baseline_legacy_dataset(config: BaselineRunConfig, cfg: dict[str, Any]) -> MemeDataset:
    return MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[config.dataset_name],
        keep_missing_images=True,
        limit=config.limit,
    )


def _normalized_baseline_harmfulness_label(sample: dict[str, Any]) -> int | None:
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


def _batches(samples: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(samples), max(1, batch_size)):
        yield samples[start : start + max(1, batch_size)]



def run_training(config: OursRunConfig | BaselineRunConfig, model_kind: str = "ours") -> dict[str, Any]:
    """Dispatch to the requested training runner."""

    if model_kind == "ours":
        return run_ours_experiment(config)  # type: ignore[arg-type]
    if model_kind == "baseline":
        return run_baseline_experiment(config)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported model_kind: {model_kind}")


train_ours = run_ours_experiment
train_baseline = run_baseline_experiment

__all__ = [
    "OursRunConfig",
    "BaselineRunConfig",
    "run_ours_experiment",
    "run_baseline_experiment",
    "run_training",
    "train_ours",
    "train_baseline",
    "configure_trainable_parameters",
    "evaluate_ours_pipeline",
    "create_baseline_model",
    "train_one_epoch",
    "evaluate_model",
    "_normalize_sample",
]
