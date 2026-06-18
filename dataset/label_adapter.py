"""Adapters that convert normalized annotation strings into training targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from dataset.meme_dataset import MemeDataset
from dataset.normalized_labels import DEFAULT_DATASETS, NormalizedLabelRow, NormalizedLabelStore
from utils.annotation_utils import as_list, parse_bool
from utils.io import load_yaml


@dataclass
class LabelVocab:
    """Stable label vocabulary with class-id and multi-hot conversion helpers."""

    single_label_fields: dict[str, list[str]]
    multi_label_fields: dict[str, list[str]]
    binary_fields: list[str]
    ignore_index: int = -100
    single_ignore_labels: dict[str, set[str]] = field(default_factory=dict)
    multi_ignore_labels: dict[str, set[str]] = field(default_factory=dict)
    binary_field_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    optional_text_fields: list[str] = field(default_factory=list)
    sample_weight_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/label_vocab.yaml") -> "LabelVocab":
        """Load a LabelVocab from YAML config."""

        cfg = load_yaml(path)
        single_cfg = cfg.get("single_label_fields", {}) or {}
        multi_cfg = cfg.get("multi_label_fields", {}) or {}
        binary_cfg = cfg.get("binary_fields", {}) or {}
        return cls(
            single_label_fields={field: list(spec.get("labels", [])) for field, spec in single_cfg.items()},
            multi_label_fields={field: list(spec.get("labels", [])) for field, spec in multi_cfg.items()},
            binary_fields=list(binary_cfg.keys()),
            single_ignore_labels={field: set(spec.get("ignore_labels", [])) for field, spec in single_cfg.items()},
            multi_ignore_labels={field: set(spec.get("ignore_labels", [])) for field, spec in multi_cfg.items()},
            binary_field_configs={field: dict(spec or {}) for field, spec in binary_cfg.items()},
            optional_text_fields=list(cfg.get("optional_text_fields", []) or []),
            sample_weight_config=dict(cfg.get("sample_weight", {}) or {}),
        )

    def label_to_id(self, field: str, label: str) -> int:
        """Map a label string to a stable class id, falling back to unknown."""

        labels = self.single_label_fields.get(field) or self.multi_label_fields.get(field)
        if not labels:
            raise KeyError(f"Unknown label field: {field}")
        value = str(label)
        if value in labels:
            return labels.index(value)
        if "unknown" in labels:
            return labels.index("unknown")
        return self.ignore_index

    def id_to_label(self, field: str, idx: int) -> str:
        """Map a stable class id back to a label string."""

        labels = self.single_label_fields.get(field) or self.multi_label_fields.get(field)
        if not labels:
            raise KeyError(f"Unknown label field: {field}")
        if idx < 0 or idx >= len(labels):
            return "unknown"
        return labels[idx]

    def multi_hot(self, field: str, labels: list[str]) -> list[int]:
        """Convert multi-label strings into a deterministic multi-hot vector."""

        vocab = self.multi_label_fields[field]
        ignore = self.multi_ignore_labels.get(field, set())
        vector = [0] * len(vocab)
        for label in labels:
            value = str(label)
            if value in ignore or (value not in vocab and "unknown" in ignore):
                continue
            idx = self.label_to_id(field, value)
            if idx != self.ignore_index and 0 <= idx < len(vector):
                vector[idx] = 1
        return vector

    def num_classes(self, field: str) -> int:
        """Return number of classes for a single or multi-label field."""

        labels = self.single_label_fields.get(field) or self.multi_label_fields.get(field)
        if labels is None:
            raise KeyError(f"Unknown label field: {field}")
        return len(labels)

    def mask_for_single(self, field: str, label: str) -> int:
        """Return 0 when a single-label value should be ignored."""

        value = str(label)
        labels = self.single_label_fields.get(field, [])
        ignore = self.single_ignore_labels.get(field, set())
        return 0 if value in ignore or (value not in labels and "unknown" in ignore) else 1

    def mask_for_multi(self, field: str, labels: list[str]) -> int:
        """Return 0 when all multi-label values should be ignored."""

        vocab = self.multi_label_fields.get(field, [])
        ignore = self.multi_ignore_labels.get(field, set())
        usable = [str(label) for label in labels if str(label) not in ignore and not (str(label) not in vocab and "unknown" in ignore)]
        return int(bool(usable))


class NormalizedLabelAdapter:
    """Attach normalized labels and model-ready targets to meme samples."""

    def __init__(
        self,
        vocab: LabelVocab | None = None,
        vocab_path: str | Path = "configs/label_vocab.yaml",
    ) -> None:
        self.vocab = vocab or LabelVocab.from_yaml(vocab_path)

    def encode_row(self, row: NormalizedLabelRow | dict[str, Any]) -> dict[str, Any]:
        """Encode a normalized row into string labels and numeric targets."""

        row_dict = row.to_dict() if isinstance(row, NormalizedLabelRow) else dict(row)
        labels = dict(row_dict.get("labels", {}) or {})
        label_strings: dict[str, Any] = {}
        class_ids: dict[str, int] = {}
        multi_hot: dict[str, list[int]] = {}
        binary: dict[str, int] = {}
        masks: dict[str, int] = {}

        for field in self.vocab.single_label_fields:
            label = str(labels.get(field, "unknown"))
            label_strings[field] = label
            class_ids[field] = self.vocab.label_to_id(field, label)
            masks[field] = self.vocab.mask_for_single(field, label)

        for field in self.vocab.multi_label_fields:
            values = [str(item) for item in as_list(labels.get(field, []))]
            label_strings[field] = values
            multi_hot[field] = self.vocab.multi_hot(field, values)
            masks[field] = self.vocab.mask_for_multi(field, values)

        for field in self.vocab.binary_fields:
            value = parse_bool(labels.get(field), default=False)
            label_strings[field] = value
            binary[field] = int(value)
            masks[field] = 1

        if "confidence" in labels:
            label_strings["confidence"] = str(labels.get("confidence"))
        if "confidence_score" in labels:
            label_strings["confidence_score"] = labels.get("confidence_score")

        sample_weight = self._sample_weight(labels)
        return {
            "label_strings": label_strings,
            "class_ids": class_ids,
            "multi_hot": multi_hot,
            "binary": binary,
            "masks": masks,
            "sample_weight": sample_weight,
            "evidence_text": dict(row_dict.get("evidence_text", {}) or {}),
            "audit_flags": list(row_dict.get("audit_flags", []) or []),
            "source_annotation": dict(row_dict.get("source_annotation", {}) or {}),
        }

    def attach_to_sample(
        self,
        sample: dict[str, Any],
        row: NormalizedLabelRow | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return a sample copy with normalized annotation and target fields."""

        enriched = dict(sample)
        if row is None:
            enriched.update(
                {
                    "normalized_annotation": None,
                    "targets": None,
                    "label_strings": {},
                    "evidence_text": {},
                    "audit_flags": [],
                    "sample_weight": 0.0,
                }
            )
            return enriched
        row_dict = row.to_dict() if isinstance(row, NormalizedLabelRow) else dict(row)
        targets = self.encode_row(row_dict)
        enriched["normalized_annotation"] = row_dict
        enriched["targets"] = targets
        enriched["label_strings"] = targets["label_strings"]
        enriched["evidence_text"] = targets["evidence_text"]
        enriched["audit_flags"] = targets["audit_flags"]
        enriched["sample_weight"] = targets["sample_weight"]
        return enriched

    def encode_batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate already-attached target dictionaries by key."""

        return {
            "targets": [sample.get("targets") for sample in samples],
            "label_strings": [sample.get("label_strings", {}) for sample in samples],
            "sample_weight": [sample.get("sample_weight", 0.0) for sample in samples],
            "audit_flags": [sample.get("audit_flags", []) for sample in samples],
            "evidence_text": [sample.get("evidence_text", {}) for sample in samples],
        }

    def _sample_weight(self, labels: dict[str, Any]) -> float:
        cfg = self.vocab.sample_weight_config
        weight = 1.0
        if cfg.get("use_confidence_score", True):
            try:
                weight = float(labels.get("confidence_score", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
        if cfg.get("downweight_not_sure", True) and parse_bool(labels.get("not_sure"), default=False):
            weight *= float(cfg.get("not_sure_weight_multiplier", 0.5))
        return max(float(cfg.get("min_weight", 0.1)), weight)


class NormalizedMemeDataset:
    """MemeDataset wrapper that attaches normalized labels and encoded targets."""

    def __init__(
        self,
        dataset_root: str | Path = "dataset/source",
        annotation_root: str | Path = "dataset/annotation",
        normalized_root: str | Path = "dataset/annotation_normalized",
        dataset_names: Iterable[str] | None = None,
        label_set: str = "full",
        vocab_path: str | Path = "configs/label_vocab.yaml",
        keep_missing_images: bool = False,
        limit: int | None = None,
        require_normalized_label: bool = True,
    ) -> None:
        self.dataset_names = _source_dataset_names(dataset_names)
        self.base_dataset = MemeDataset(
            dataset_root=dataset_root,
            annotation_root=annotation_root,
            dataset_names=self.dataset_names,
            keep_missing_images=keep_missing_images,
            limit=limit,
        )
        store_names = ["all"] if _requested_all(dataset_names) else (list(self.dataset_names) if self.dataset_names else None)
        self.label_store = NormalizedLabelStore(
            normalized_root=normalized_root,
            dataset_names=store_names,
            label_set=label_set,
        )
        self.adapter = NormalizedLabelAdapter(vocab_path=vocab_path)
        self.require_normalized_label = require_normalized_label
        self.label_set = label_set
        self.samples: list[dict[str, Any]] = []
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            row = self.label_store.get(str(sample.get("dataset_name")), str(sample.get("sample_id")))
            if require_normalized_label and row is None:
                continue
            self.samples.append(self.adapter.attach_to_sample(sample, row))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def preview(self, n: int = 3) -> list[dict[str, Any]]:
        """Return compact previews with target fields."""

        previews = []
        for sample in self.samples[:n]:
            previews.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "dataset_name": sample.get("dataset_name"),
                    "raw_label": sample.get("raw_label"),
                    "label_strings": sample.get("label_strings", {}),
                    "sample_weight": sample.get("sample_weight"),
                    "audit_flags": sample.get("audit_flags", []),
                    "has_normalized_annotation": sample.get("normalized_annotation") is not None,
                }
            )
        return previews

    def statistics(self) -> dict[str, Any]:
        """Return wrapper statistics and normalized-label coverage."""

        label_counts: dict[str, Counter[str]] = {field: Counter() for field in self.adapter.vocab.single_label_fields}
        weights = []
        for sample in self.samples:
            labels = sample.get("label_strings", {}) or {}
            for field in label_counts:
                label_counts[field][str(labels.get(field, "unknown"))] += 1
            weights.append(float(sample.get("sample_weight", 0.0)))
        return {
            "total": len(self.samples),
            "label_set": self.label_set,
            "require_normalized_label": self.require_normalized_label,
            "coverage": self.label_store.coverage_for_samples([self.base_dataset[idx] for idx in range(len(self.base_dataset))]),
            "single_label_counts": {field: dict(counter) for field, counter in label_counts.items()},
            "sample_weight": _weight_summary(weights),
            "validation": self.validate_files(),
        }

    def validate_files(self) -> dict[str, int]:
        """Count missing images and empty OCR strings in attached samples."""

        missing_images = 0
        empty_text = 0
        for sample in self.samples:
            metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
            if not metadata.get("image_exists", bool(sample.get("image_path"))):
                missing_images += 1
            if not str(sample.get("ocr_text_full", "")).strip():
                empty_text += 1
        return {"missing_images": missing_images, "empty_text": empty_text}


def _requested_all(dataset_names: Iterable[str] | None) -> bool:
    return dataset_names is None or "all" in {str(name) for name in dataset_names}


def _source_dataset_names(dataset_names: Iterable[str] | None) -> list[str] | None:
    if dataset_names is None:
        return None
    values = [str(name) for name in dataset_names]
    if "all" in values:
        return DEFAULT_DATASETS
    return values


def _weight_summary(weights: list[float]) -> dict[str, float | None]:
    if not weights:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(weights), "max": max(weights), "mean": mean(weights)}
