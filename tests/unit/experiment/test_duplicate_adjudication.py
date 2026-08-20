from pathlib import Path

from PIL import Image

from experiments.duplicate_adjudication import (
    canonical_member,
    classify_group,
    decoded_pixel_sha256,
    multimodal_group_id,
)


def _row(key, raw=1, normalized="harmful", text="same", file_hash="a", split="train", match=True):
    return {"sample_key":key,"raw_label_numeric":raw,"normalized_harmfulness":normalized,"source_text":text,"normalized_text":text,"source_normalized_ocr_match":match,"raw_file_sha256":file_hash,"split":split}


def test_raw_file_can_differ_while_decoded_pixels_match(tmp_path):
    first=tmp_path/"a.png"; second=tmp_path/"b.png"
    image=Image.new("RGB",(12,9),(12,34,56)); image.save(first,compress_level=0); image.save(second,compress_level=9)
    assert first.read_bytes()!=second.read_bytes()
    assert decoded_pixel_sha256(first)[0]==decoded_pixel_sha256(second)[0]


def test_composite_hash_changes_with_ocr():
    assert multimodal_group_id("pixels","a") != multimodal_group_id("pixels","b")


def test_image_same_ocr_different_is_shared_image():
    category,_,action=classify_group([_row("harm_c::a",text="one"),_row("harm_c::b",text="two")])
    assert category=="SHARED_IMAGE_DIFFERENT_TEXT_CONFIRMED"
    assert action.startswith("KEEP_AS_DISTINCT")


def test_raw_label_conflict_precedes_normalization():
    category,_,action=classify_group([_row("harm_c::a",0,"non_harmful"),_row("harm_c::b",1,"harmful")])
    assert category=="RAW_SOURCE_LABEL_CONFLICT" and action=="EXCLUDE_ALL_CONFLICTING_MEMBERS"


def test_normalization_induced_conflict():
    category,_,_=classify_group([_row("harm_c::a",1,"harmful"),_row("harm_c::b",1,"non_harmful")])
    assert category=="NORMALIZATION_INDUCED_LABEL_CONFLICT"


def test_pairing_mismatch_is_detected():
    category,_,_=classify_group([_row("harm_c::a",match=False),_row("harm_c::b")])
    assert category=="SUSPECTED_IMAGE_TEXT_PAIRING_ERROR"


def test_format_duplicate_classification():
    category,_,_=classify_group([_row("harm_c::a",file_hash="a"),_row("harm_c::b",file_hash="b")])
    assert category=="FORMAT_DUPLICATE_ONLY"


def test_exact_duplicate_classification():
    category,_,action=classify_group([_row("harm_c::a"),_row("harm_c::b")])
    assert category=="EXACT_MULTIMODAL_DUPLICATE_SAME_LABEL"
    assert action=="RETAIN_ONE_CANONICAL_SAMPLE"


def test_canonical_choice_is_deterministic():
    rows=[_row("harm_c::z"),_row("harm_c::a")]
    assert canonical_member(rows)["sample_key"]=="harm_c::a"


def test_cross_split_fixture_preserves_canonical_choice():
    rows=[_row("harm_c::b",split="validation"),_row("harm_c::a",split="train")]
    assert len({row["split"] for row in rows})==2
    assert canonical_member(rows)["split"]=="train"


def test_module_contains_no_forbidden_dataset_loader():
    source=Path("experiments/duplicate_adjudication.py").read_text(encoding="utf-8")
    assert "MemeDataset(" not in source
