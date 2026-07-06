from __future__ import annotations

from module.runner import HarmfulMemePipeline


def test_full_pipeline_smoke():
    pipeline = HarmfulMemePipeline().eval()
    output = pipeline(
        {
            "sample_id": "s1",
            "dataset_name": "memotion",
            "image_path": None,
            "ocr_text_full": "CAN SOMEONE TAG BRIAN I HEARD HE WANTS TO RIDE A PONY",
        }
    )
    assert set(output) == {"stage_a", "stage_b", "stage_c", "stage_d", "stage_e"}
    assert output["stage_e"].structured_prediction["sample_id"] == "s1"
