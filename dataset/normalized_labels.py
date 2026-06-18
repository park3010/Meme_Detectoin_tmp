"""Normalized annotation label loading and dataset attachment utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from utils.io import read_jsonl


DEFAULT_DATASETS = ["harm_c", "harm_p", "facebook", "memotion"]
LABEL_SET_FILENAMES = {
    "full": "normalized_labels.jsonl",
    "clean": "normalized_clean.jsonl",
}


@dataclass
class NormalizedLabelRow:
    """One normalized annotation row keyed by dataset and sample id."""

    sample_id: str
    dataset_name: str
    image_path: str | None
    ocr_text_full: str
    raw_label: Any
    labels: dict[str, Any] = field(default_factory=dict)
    evidence_text: dict[str, str] = field(default_factory=dict)
    source_annotation: dict[str, Any] = field(default_factory=dict)
    audit_flags: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        """Stable lookup key for normalized labels."""

        return (self.dataset_name, self.sample_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""

        return asdict(self)


def load_normalized_label_rows(path: str | Path) -> list[NormalizedLabelRow]:
    """Load normalized JSONL rows from a file."""

    rows: list[NormalizedLabelRow] = []
    for record in read_jsonl(path):
        rows.append(
            NormalizedLabelRow(
                sample_id=str(record.get("sample_id", "")),
                dataset_name=str(record.get("dataset_name", "")),
                image_path=record.get("image_path"),
                ocr_text_full=str(record.get("ocr_text_full", "")),
                raw_label=record.get("raw_label"),
                labels=dict(record.get("labels", {}) or {}),
                evidence_text=dict(record.get("evidence_text", {}) or {}),
                source_annotation=dict(record.get("source_annotation", {}) or {}),
                audit_flags=list(record.get("audit_flags", []) or []),
            )
        )
    return rows


def iter_normalized_label_paths(
    normalized_root: str | Path,
    dataset_names: list[str],
    label_set: str = "full",
) -> list[Path]:
    """Return normalized label JSONL paths for datasets and label-set type."""

    root = Path(normalized_root)
    filename = LABEL_SET_FILENAMES.get(label_set)
    if filename is None:
        if label_set == "uncertain":
            filename = LABEL_SET_FILENAMES["full"]
        else:
            raise ValueError(f"Unsupported label_set={label_set!r}; expected full or clean.")
    return [root / dataset_name / filename for dataset_name in _resolve_store_datasets(root, dataset_names)]


class NormalizedLabelStore:
    """In-memory lookup table for normalized labels keyed by dataset/sample."""

    def __init__(
        self,
        normalized_root: str | Path = "dataset/annotation_normalized",
        dataset_names: list[str] | None = None,
        label_set: str = "full",
    ) -> None:
        self.normalized_root = Path(normalized_root)
        self.dataset_names = _resolve_store_datasets(self.normalized_root, dataset_names)
        self.label_set = label_set
        self.rows: list[NormalizedLabelRow] = []
        self.index: dict[tuple[str, str], NormalizedLabelRow] = {}
        self.duplicates: list[dict[str, Any]] = []
        self.paths = iter_normalized_label_paths(self.normalized_root, self.dataset_names, label_set=label_set)
        self._load()

    def get(self, dataset_name: str, sample_id: str) -> NormalizedLabelRow | None:
        """Return the normalized row for a dataset/sample key if present."""

        return self.index.get((str(dataset_name), str(sample_id)))

    def __len__(self) -> int:
        return len(self.index)

    def coverage_for_samples(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Compute normalized-label coverage for existing dataset samples."""

        total = 0
        matched = 0
        missing_by_dataset: Counter[str] = Counter()
        total_by_dataset: Counter[str] = Counter()
        matched_by_dataset: Counter[str] = Counter()
        for sample in samples:
            total += 1
            dataset_name = str(sample.get("dataset_name", ""))
            sample_id = str(sample.get("sample_id", ""))
            total_by_dataset[dataset_name] += 1
            if self.get(dataset_name, sample_id):
                matched += 1
                matched_by_dataset[dataset_name] += 1
            else:
                missing_by_dataset[dataset_name] += 1
        return {
            "total_samples": total,
            "matched_samples": matched,
            "missing_samples": total - matched,
            "coverage_ratio": matched / total if total else 0.0,
            "matched_by_dataset": dict(sorted(matched_by_dataset.items())),
            "missing_by_dataset": dict(sorted(missing_by_dataset.items())),
            "total_by_dataset": dict(sorted(total_by_dataset.items())),
            "store_rows": len(self),
            "duplicate_count": len(self.duplicates),
            "label_set": self.label_set,
            "datasets": self.dataset_names,
        }

    def _load(self) -> None:
        for path in self.paths:
            for row in load_normalized_label_rows(path):
                if self.label_set == "uncertain" and not _is_uncertain(row):
                    continue
                self.rows.append(row)
                if row.key in self.index:
                    self.duplicates.append({"key": row.key, "path": str(path)})
                    continue
                self.index[row.key] = row


def _is_uncertain(row: NormalizedLabelRow) -> bool:
    labels = row.labels or {}
    return bool(labels.get("not_sure") or row.audit_flags)


def _resolve_store_datasets(root: Path, dataset_names: list[str] | None) -> list[str]:
    if dataset_names is None:
        discovered = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "all") if root.exists() else []
        return discovered or DEFAULT_DATASETS
    if "all" in dataset_names:
        return ["all"]
    return [str(name) for name in dataset_names]
