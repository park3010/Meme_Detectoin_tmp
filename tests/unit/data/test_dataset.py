from __future__ import annotations

import json

from dataset import MemeDataset


def test_meme_dataset_loads_jsonl_and_annotations(tmp_path):
    root = tmp_path / "V1"
    source = root / "covid_img+text"
    (source / "img").mkdir(parents=True)
    (source / "txt").mkdir()
    (source / "img" / "sample_1.png").write_bytes(b"not-a-real-image")
    (source / "txt" / "all.jsonl").write_text(
        json.dumps({"id": "sample_1", "image": "sample_1.png", "labels": 1, "text": "HELLO meme"}) + "\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs" / "harm_c"
    outputs.mkdir(parents=True)
    (outputs / "harmc_annotations.jsonl").write_text(
        json.dumps({"sample_id": "sample_1", "dataset_name": "harmc", "annotation": {"target": "demo"}}) + "\n",
        encoding="utf-8",
    )

    dataset = MemeDataset(root, outputs.parent)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["dataset_name"] == "harm_c"
    assert sample["ocr_text_full"] == "HELLO meme"
    assert sample["annotation"] == {"target": "demo"}
    assert dataset.statistics()["total"] == 1
