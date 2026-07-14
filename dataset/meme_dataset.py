"""Unified loader for the prepared meme datasets and annotations."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dataset.base_dataset import BaseMemeDataset, MemeSample
from utils.image_utils import IMAGE_EXTENSIONS
from utils.io import read_jsonl
from utils.logging_utils import setup_logger
from utils.text_utils import normalize_text


logger = setup_logger(__name__)

DATASET_FOLDER_TO_NAME = {
    "covid_img+text": "harm_c",
    "political_img+text": "harm_p",
    "memotion_img+text": "memotion",
    "facebook_img+text": "facebook",
}

DATASET_ALIASES = {
    "harmc": "harm_c",
    "harm_c": "harm_c",
    "harmp": "harm_p",
    "harm_p": "harm_p",
    "memotion": "memotion",
    "facebook": "facebook",
}

PAPER_DATASET_PROTOCOL: dict[str, dict[str, Any]] = {
    "harm_c": {
        "dataset_family": "harmeme",
        "paper_name": "HarMeme",
        "original_dataset": "harm_c",
        "domain": "covid",
        "domain_role": "source_train_validation",
        "enabled_for_paper": True,
    },
    "harm_p": {
        "dataset_family": "harmeme",
        "paper_name": "HarMeme",
        "original_dataset": "harm_p",
        "domain": "politics",
        "domain_role": "source_train_validation",
        "enabled_for_paper": True,
    },
    "facebook": {
        "dataset_family": "fhm",
        "paper_name": "FHM",
        "original_dataset": "facebook",
        "domain": "facebook",
        "domain_role": "heldout_target_test",
        "annotation_provenance": "agent_silver_structured_evaluation",
        "enabled_for_paper": True,
    },
    "memotion": {
        "dataset_family": "memotion",
        "paper_name": "Memotion",
        "original_dataset": "memotion",
        "domain": "memotion",
        "domain_role": "future_dataset",
        "enabled_for_paper": False,
        "disabled_reason": "unified harmfulness labels require future re-annotation",
    },
}


class MemeDataset(BaseMemeDataset):
    """Scan images, OCR JSONL/text files, and optional annotation JSONL files."""

    def __init__(
        self,
        dataset_root: str | Path = "dataset/source",
        annotation_root: str | Path | None = "dataset/annotation",
        dataset_names: Iterable[str] | None = None,
        keep_missing_images: bool = False,
        limit: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.annotation_root = Path(annotation_root) if annotation_root else None
        self.keep_missing_images = keep_missing_images
        requested = {_canonical_name(name) for name in dataset_names} if dataset_names else None
        annotation_index = self._load_annotations()
        samples: list[MemeSample] = []

        for source_dir in sorted(self.dataset_root.iterdir() if self.dataset_root.exists() else []):
            if not source_dir.is_dir() or source_dir.name not in DATASET_FOLDER_TO_NAME:
                continue
            dataset_name = DATASET_FOLDER_TO_NAME[source_dir.name]
            if requested and dataset_name not in requested:
                continue
            samples.extend(self._load_source(source_dir, dataset_name, annotation_index))

        if not keep_missing_images:
            samples = [sample for sample in samples if sample.image_path and Path(sample.image_path).exists()]
        self.samples = samples[:limit] if limit else samples
        logger.info("Loaded %s unified samples from %s", len(self.samples), self.dataset_root)

    def _load_source(
        self,
        source_dir: Path,
        dataset_name: str,
        annotation_index: dict[tuple[str, str], dict[str, Any]],
    ) -> list[MemeSample]:
        image_index = _scan_images(source_dir / "img")
        text_records = _load_text_records(source_dir / "txt")
        if not text_records:
            text_records = {
                stem: {
                    "id": stem,
                    "image": image_path.name,
                    "text": "",
                }
                for stem, image_path in image_index.items()
            }

        samples: list[MemeSample] = []
        for sample_id, record in sorted(text_records.items()):
            image_path = _resolve_image_path(source_dir / "img", image_index, record, sample_id)
            annotation = annotation_index.get((dataset_name, sample_id)) or annotation_index.get(("*", sample_id))
            samples.append(
                MemeSample(
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    image_path=str(image_path) if image_path else None,
                    ocr_text_full=normalize_text(
                        record.get("ocr_text_full")
                        or record.get("ocr_text")
                        or record.get("text")
                        or record.get("caption")
                        or ""
                    ),
                    raw_label=record.get("labels", record.get("label")),
                    annotation=annotation,
                    raw_record=record,
                    metadata={
                        "source_folder": source_dir.name,
                        "image_exists": image_path.exists() if image_path else False,
                        **PAPER_DATASET_PROTOCOL.get(dataset_name, {}),
                    },
                )
            )
        return samples

    def _load_annotations(self) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.annotation_root or not self.annotation_root.exists():
            return {}
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(self.annotation_root.rglob("*annotation*.jsonl")):
            for record in read_jsonl(path):
                sample_id = normalize_text(record.get("sample_id") or record.get("id"))
                if not sample_id:
                    continue
                raw_dataset_name = normalize_text(record.get("dataset_name", ""))
                dataset_name = DATASET_ALIASES.get(raw_dataset_name, raw_dataset_name)
                payload = record.get("annotation") if isinstance(record.get("annotation"), dict) else record
                if dataset_name:
                    index[(dataset_name, sample_id)] = payload
                index[("*", sample_id)] = payload
        logger.info("Indexed %s annotations from %s", len(index), self.annotation_root)
        return index

    def statistics(self) -> dict[str, Any]:
        """Return dataset counts, label distribution, and validation summary."""

        by_dataset = Counter(sample.dataset_name for sample in self.samples)
        by_label = defaultdict(Counter)
        annotations = 0
        for sample in self.samples:
            by_label[sample.dataset_name][str(sample.raw_label)] += 1
            if sample.annotation is not None:
                annotations += 1
        return {
            "total": len(self.samples),
            "by_dataset": dict(by_dataset),
            "by_label": {name: dict(counter) for name, counter in by_label.items()},
            "annotations": annotations,
            "validation": self.validate_files(),
        }


def _canonical_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _scan_images(img_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    if not img_dir.exists():
        return images
    for path in img_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem] = path
    return images


def _load_text_records(txt_dir: Path) -> dict[str, dict[str, Any]]:
    if not txt_dir.exists():
        return {}
    preferred = txt_dir / "all.jsonl"
    jsonl_paths = [preferred] if preferred.exists() else sorted(txt_dir.glob("*.jsonl"))
    records: dict[str, dict[str, Any]] = {}
    for path in jsonl_paths:
        for record in read_jsonl(path):
            sample_id = normalize_text(record.get("id") or record.get("sample_id") or Path(str(record.get("image", ""))).stem)
            if sample_id:
                records.setdefault(sample_id, record)
    if records:
        return records
    for path in sorted(txt_dir.glob("*.txt")):
        records[path.stem] = {"id": path.stem, "text": path.read_text(encoding="utf-8", errors="ignore")}
    return records


def _resolve_image_path(
    img_dir: Path,
    image_index: dict[str, Path],
    record: dict[str, Any],
    sample_id: str,
) -> Path | None:
    image_value = record.get("image") or record.get("image_path")
    if isinstance(image_value, str) and image_value.strip():
        candidate = img_dir / Path(image_value).name
        if candidate.exists():
            return candidate
    return image_index.get(sample_id)
