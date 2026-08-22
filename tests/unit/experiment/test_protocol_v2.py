import hashlib
import json
from pathlib import Path

from experiments.protocol_v2 import audit_source_split_v2, canonical_label, structured_conflicts


def row(**labels):
    return {"labels": labels}


def test_formal_structured_conflict_and_masked_values():
    assert structured_conflicts([row(target_presence="explicit"), row(target_presence="implicit")]) == {
        "target_presence": ["explicit", "implicit"]
    }
    assert structured_conflicts([row(target_presence="unknown"), row(target_presence="explicit")]) == {}


def test_multilabel_set_equality_is_order_independent():
    assert canonical_label(["satire", "sarcasm"], multilabel=True) == canonical_label(
        ["sarcasm", "satire", "satire"], multilabel=True
    )
    assert "tactic_rhetorical" in structured_conflicts([
        row(tactic_rhetorical=["satire"]), row(tactic_rhetorical=["sarcasm"])
    ])


def test_v2_official_artifacts_and_legacy_preservation():
    root = Path("result/splits/harmeme")
    v1 = root / "source_split_seed_42.json"
    v2 = root / "source_split_seed_42_v2.json"
    resolution = json.loads((root / "duplicate_resolution_manifest_v1.json").read_text())
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == "f02364f41169fdeac4ec1cffe68192b9d522e7fe5bca5ba2848ad6dfe777200c"
    assert v2.is_file() and audit_source_split_v2(v2)["passed"]
    assert resolution["source_split_v1_sha256"] == hashlib.sha256(v1.read_bytes()).hexdigest()
    assert resolution["structured_label_conflict_group_count"] == 9


def test_no_conflict_or_excluded_member_retained():
    root = Path("result/splits/harmeme")
    split = json.loads((root / "source_split_seed_42_v2.json").read_text())
    resolution = json.loads((root / "duplicate_resolution_manifest_v1.json").read_text())
    retained = {r["sample_key"] for p in ("train", "validation") for r in split[p]}
    excluded = {json.loads(line)["sample_key"] for line in (root / "excluded_samples_v2.jsonl").read_text().splitlines()}
    assert not retained & excluded
    for group in resolution["groups"]:
        if group["final_category"] in {"RAW_SOURCE_LABEL_CONFLICT", "STRUCTURED_LABEL_CONFLICT"}:
            assert not retained & set(group["sample_keys"])


def test_retrieval_v2_is_split_v2_train_only_and_v1_present():
    split = json.loads(Path("result/splits/harmeme/source_split_seed_42_v2.json").read_text())
    train = {r["sample_key"] for r in split["train"]}
    queries = [json.loads(line) for line in Path("dataset/retrieval/harmeme_train_v2/cache/source_queries/queries.jsonl").read_text().splitlines()]
    assert {r["sample_key"] for r in queries} == train
    assert Path("dataset/retrieval/harmeme_train_v1/retrieval_manifest.json").is_file()
    assert Path("dataset/retrieval/harmeme_train_v2/retrieval_manifest.json").is_file()


def test_active_config_rejects_legacy_for_paper():
    text = Path("configs/config.yaml").read_text()
    policy = Path("configs/paper_result_policy.yaml").read_text()
    assert "active_profile: harmeme_train_v2" in text
    assert "harmeme_train_v1:\n      enabled: true\n      paper_eligible: false" in text
    assert "protocol_version: harmeme_to_fhm_v2" in policy


def test_fhm_manifest_preserved_without_loading_dataset():
    path = Path("result/splits/fhm/heldout_test_manifest.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "a8ad8bb386b8e739b9362e141af14bf9409589f6fd4480c07dadef0163aba197"
