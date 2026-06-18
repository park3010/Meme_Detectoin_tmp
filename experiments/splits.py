"""Dataset split creation and loading for baseline experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from dataset.meme_dataset import DATASET_ALIASES, DATASET_FOLDER_TO_NAME, MemeDataset
from utils.io import read_jsonl, write_json


DEFAULT_SEEDS = [42, 52, 123, 777, 2026]


def normalize_dataset_names(dataset: str | list[str] | None) -> list[str] | None:
    """Normalize CLI dataset input. `None` means all supported datasets."""

    if dataset is None:
        return None
    values = dataset if isinstance(dataset, list) else [dataset]
    if not values or "all" in values:
        return None
    return [DATASET_ALIASES.get(value, value) for value in values]


def build_splits_for_dataset(
    dataset_name: str,
    dataset: MemeDataset,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    dataset_root: str | Path = "dataset/source",
) -> dict[str, list[str]]:
    """Build one dataset split, preserving official txt splits when available."""

    samples = [sample for sample in dataset.samples if sample.dataset_name == dataset_name and label_to_int(sample.raw_label) is not None]
    official = load_official_split_ids(dataset_name, dataset_root=dataset_root)
    if official and _covers_any(official, samples):
        sample_ids = {sample.sample_id for sample in samples}
        return {name: [sid for sid in ids if sid in sample_ids] for name, ids in official.items()}
    return stratified_split(samples, seed=seed, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio)


def save_splits(
    splits: dict[str, list[str]],
    dataset_name: str,
    seed: int,
    output_root: str | Path = "result/splits",
) -> Path:
    """Save split JSON and return the path."""

    path = Path(output_root) / dataset_name / f"seed_{seed}.json"
    write_json(path, splits)
    return path


def load_split_file(path: str | Path) -> dict[str, list[str]]:
    """Load a split file."""

    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {key: [str(item) for item in value] for key, value in data.items()}


def split_samples(samples: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Materialize sample dictionaries by split id lists."""

    by_id = {str(sample["sample_id"]): sample for sample in samples}
    return {name: [by_id[sid] for sid in ids if sid in by_id] for name, ids in splits.items()}


def label_to_int(label: object) -> int | None:
    """Convert raw harmfulness labels into 0/1."""

    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return 1 if label == 1 else 0 if label == 0 else None
    text = str(label).strip().lower()
    if text in {"1", "true", "harmful", "yes"}:
        return 1
    if text in {"0", "false", "non_harmful", "non-harmful", "no"}:
        return 0
    return None


def load_official_split_ids(dataset_name: str, dataset_root: str | Path = "dataset/source") -> dict[str, list[str]] | None:
    """Load official train/valid/test ids from txt JSONL files when present."""

    folder = _dataset_folder(dataset_name)
    if folder is None:
        return None
    source = Path(dataset_root) / folder / "txt"
    paths = {
        "train": source / "train.jsonl",
        "valid": source / "val.jsonl",
        "test": source / "test.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    splits: dict[str, list[str]] = {}
    for name, path in paths.items():
        splits[name] = [str(record.get("id") or record.get("sample_id")) for record in read_jsonl(path) if record.get("id") or record.get("sample_id")]
    return splits


def stratified_split(
    samples: list[Any],
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict[str, list[str]]:
    """Create deterministic stratified split by harmfulness label when possible."""

    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")
    rng = random.Random(seed)
    buckets: dict[int, list[str]] = {0: [], 1: []}
    unlabeled: list[str] = []
    for sample in samples:
        sample_id = sample.sample_id if hasattr(sample, "sample_id") else str(sample["sample_id"])
        raw_label = sample.raw_label if hasattr(sample, "raw_label") else sample.get("raw_label")
        label = label_to_int(raw_label)
        if label is None:
            unlabeled.append(sample_id)
        else:
            buckets[label].append(sample_id)
    splits = {"train": [], "valid": [], "test": []}
    for ids in buckets.values():
        rng.shuffle(ids)
        train_end = int(round(len(ids) * train_ratio))
        valid_end = train_end + int(round(len(ids) * valid_ratio))
        splits["train"].extend(ids[:train_end])
        splits["valid"].extend(ids[train_end:valid_end])
        splits["test"].extend(ids[valid_end:])
    if unlabeled:
        rng.shuffle(unlabeled)
        splits["train"].extend(unlabeled)
    for ids in splits.values():
        rng.shuffle(ids)
    return splits


def _dataset_folder(dataset_name: str) -> str | None:
    for folder, logical in DATASET_FOLDER_TO_NAME.items():
        if logical == dataset_name:
            return folder
    return None


def _covers_any(splits: dict[str, list[str]], samples: list[Any]) -> bool:
    sample_ids = {sample.sample_id for sample in samples}
    return any(sid in sample_ids for ids in splits.values() for sid in ids)
