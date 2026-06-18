"""Training and evaluation runner for simple harmfulness baselines."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dataset import MemeDataset, NormalizedMemeDataset
from experiments.early_stopping import EarlyStopping, metric_from_validation, prefix_metrics, save_checkpoint, save_training_log
from experiments.metrics import compute_harmfulness_metrics
from experiments.progress import progress_iter
from experiments.splits import build_splits_for_dataset, label_to_int, load_split_file, save_splits, split_samples
from module.baselines import CLIPTextConcatClassifier, ImageOnlyCLIPClassifier, TextOnlyEncoderClassifier
from utils.io import load_yaml, write_json, write_jsonl
from utils.seed import set_seed


@dataclass
class BaselineRunConfig:
    """Runtime configuration for one baseline experiment."""

    model_name: str
    dataset_name: str
    seed: int = 42
    config_path: str = "configs/default.yaml"
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
        dataset = _load_legacy_dataset(config, cfg)
        split_dataset = dataset
        sample_dicts = []
        for sample in dataset:
            try:
                sample_dicts.append(_normalize_sample(sample))
            except ValueError:
                continue
    splits = _load_or_create_splits(config, split_dataset)
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
    backbone_cfg = cfg.get("backbones", {})
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


def _load_or_create_splits(config: BaselineRunConfig, dataset: Any) -> dict[str, list[str]]:
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
    label = _normalized_harmfulness_label(sample)
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
    return _load_legacy_dataset(config, cfg)


def _load_legacy_dataset(config: BaselineRunConfig, cfg: dict[str, Any]) -> MemeDataset:
    return MemeDataset(
        dataset_root=cfg.get("paths", {}).get("dataset_root", "dataset/source"),
        annotation_root=cfg.get("paths", {}).get("annotation_root", "dataset/annotation"),
        dataset_names=[config.dataset_name],
        keep_missing_images=True,
        limit=config.limit,
    )


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


def _batches(samples: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(samples), max(1, batch_size)):
        yield samples[start : start + max(1, batch_size)]
