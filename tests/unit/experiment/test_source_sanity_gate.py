import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.source_sanity_gate import atomic_write_json, validate_gradient_gate
from experiments.source_tiny_overfit import validate_overfit_manifest
from experiments.storage_safety import atomic_torch_save


def _gate(tmp_path, monkeypatch):
    root=tmp_path/"root";root.mkdir();config=tmp_path/"config.yaml";config.write_text("x: 1\n")
    report=tmp_path/"report.json";report.write_text("{}\n")
    monkeypatch.setattr("experiments.source_sanity_gate.sha256_file",lambda p: {
        "source_split_seed_42_v2.json":"1995075ba474345702ee590bc9e291522c6ebaee5f941fc1e924a867fc64e6bf",
        "config.yaml":hashlib.sha256(config.read_bytes()).hexdigest(),
        "report.json":hashlib.sha256(report.read_bytes()).hexdigest(),
        "gradient_check_gate.json":hashlib.sha256((root/"gates/gradient_check_gate.json").read_bytes()).hexdigest(),
    }[Path(p).name])
    monkeypatch.setattr("experiments.source_sanity_gate.source_sanity_code_sha256",lambda:"code")
    payload={"schema_version":"source_sanity_gradient_gate_v1","passed":True,"decision":"READY_FOR_OVERFIT_32","ready_for_overfit_32":True,"ready_for_1seed":False,"source_only":True,"device_requested":"cuda","cuda_available":True,"physical_gpu_id":0,"visible_cuda_device":0,"output_root":str(root.resolve()),"source_split_manifest_path":"result/splits/harmeme/source_split_seed_42_v2.json","source_split_manifest_sha256":"1995075ba474345702ee590bc9e291522c6ebaee5f941fc1e924a867fc64e6bf","diagnostic_manifest_sha256":"diag","config_path":str(config),"config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),"code_sha256":"code","gradient_report_path":str(report),"gradient_report_sha256":hashlib.sha256(report.read_bytes()).hexdigest(),"all_formal_tasks_tested":True,"optimizer_membership_passed":True,"parameter_updates_passed":True,"dimensions_passed":True,"nan_or_inf":False,"fhm_or_memotion_accessed":False,"scientific_checkpoint_written":False,"created_at_utc":"2026-01-01T00:00:00+00:00"}
    path=root/"gates/gradient_check_gate.json";atomic_write_json(path,payload);Path(f"{path}.sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return root,config,report,path,payload


def test_valid_gate_accepts_separate_invocation(tmp_path,monkeypatch):
    root,config,_,_,_=_gate(tmp_path,monkeypatch)
    assert validate_gradient_gate(output_root=root,config_path=config,expected_diagnostic_manifest_sha256="diag")["passed"]


@pytest.mark.parametrize(("field","value","reason"),[
    ("source_split_manifest_path","result/splits/harmeme/source_split_seed_42.json","gradient_gate_split_hash_mismatch"),
    ("code_sha256","stale","gradient_gate_code_hash_mismatch"),
    ("config_sha256","stale","gradient_gate_config_hash_mismatch"),
    ("cuda_available",False,"gradient_gate_cuda_unverified"),
    ("optimizer_membership_passed",False,"gradient_gate_optimizer_failed"),
    ("parameter_updates_passed",False,"gradient_gate_parameter_update_failed"),
    ("fhm_or_memotion_accessed",True,"gradient_gate_fhm_access_detected"),
])
def test_gate_rejects_invalid_contract(tmp_path,monkeypatch,field,value,reason):
    root,config,_,path,payload=_gate(tmp_path,monkeypatch);payload[field]=value;atomic_write_json(path,payload);Path(f"{path}.sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    assert reason in validate_gradient_gate(output_root=root,config_path=config,expected_diagnostic_manifest_sha256="diag")["reasons"]


def test_gate_from_other_output_root_rejected(tmp_path,monkeypatch):
    root,config,_,_,_=_gate(tmp_path,monkeypatch);other=tmp_path/"other";other.mkdir();(other/"gates").symlink_to(root/"gates",target_is_directory=True)
    assert "gradient_gate_output_root_mismatch" in validate_gradient_gate(output_root=other,config_path=config)["reasons"]


def test_report_hash_mismatch_rejected(tmp_path,monkeypatch):
    root,config,report,_,_=_gate(tmp_path,monkeypatch);report.write_text("changed")
    assert "gradient_report_hash_mismatch" in validate_gradient_gate(output_root=root,config_path=config)["reasons"]


def test_force_cannot_bypass_gate_validator_signature():
    assert "force" not in validate_gradient_gate.__code__.co_varnames


def test_no_in_process_or_profile_all_requirement():
    source=Path("experiments/source_sanity.py").read_text()
    assert 'elif profile == "overfit_32"' in source
    assert "validate_gradient_gate" in Path("experiments/source_tiny_overfit.py").read_text()


def test_gradient_check_writes_gate_in_source():
    source=Path("experiments/source_sanity.py").read_text()
    assert "write_gradient_gate" in source and "gradient_gate_runs" in source


def test_gate_json_is_canonical_and_deterministic(tmp_path):
    path=tmp_path/"gate.json";payload={"z":1,"a":2};atomic_write_json(path,payload);first=hashlib.sha256(path.read_bytes()).hexdigest();atomic_write_json(path,payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==first


def test_atomic_checkpoint_success_and_failure_cleanup(tmp_path):
    final=tmp_path/"best_model.pt";atomic_torch_save({"x":torch.tensor([1])},final);assert final.is_file() and final.stat().st_size>0
    def fail(*args): raise OSError("full")
    with pytest.raises(OSError): atomic_torch_save({},tmp_path/"failed.pt",save_fn=fail)
    assert not (tmp_path/"failed.pt").exists() and not list(tmp_path.glob(".failed.pt.*.tmp"))


def test_overfit_source_contains_real_training_and_outputs():
    source=Path("experiments/source_tiny_overfit.py").read_text()
    assert "loss.backward()" in source and "optimizer.step()" in source
    assert "training_curve.csv" in source and "best_model.pt" in source
    assert 'dataset_names=["harm_c","harm_p"]' in source
    assert 'dataset_names=["facebook"]' not in source and 'dataset_names=["memotion"]' not in source


def test_manifest_validator_rejects_validation_excluded_and_forbidden(tmp_path,monkeypatch):
    split=tmp_path/"split.json";split.write_text(json.dumps({"train":[{"sample_key":"harm_c::1"}],"validation":[{"sample_key":"harm_p::2"}]}))
    excluded=tmp_path/"excluded.jsonl";excluded.write_text(json.dumps({"sample_key":"harm_c::3"})+"\n")
    monkeypatch.setattr("experiments.source_tiny_overfit.EXCLUDED_PATH",excluded)
    rows=[{"sample_key":"harm_c::1","harmfulness":"harmful"},{"sample_key":"harm_p::2","harmfulness":"non_harmful"},{"sample_key":"harm_c::3","harmfulness":"harmful"},{"sample_key":"fhm::4","harmfulness":"non_harmful"}]
    rows += [{"sample_key":f"harm_c::{i}","harmfulness":"harmful" if i%2 else "non_harmful"} for i in range(5,33)]
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps({"samples":rows}))
    result=validate_overfit_manifest(manifest,split)
    assert {"overfit_manifest_non_train_ids","overfit_manifest_validation_ids","overfit_manifest_excluded_ids","overfit_manifest_forbidden_ids"} <= set(result["errors"])


def test_strict_failure_writes_report_before_return(tmp_path,monkeypatch):
    from experiments.source_tiny_overfit import run_overfit_32
    result=run_overfit_32(output_root=str(tmp_path),config="configs/config.yaml",seed=42,device="cuda",force=False)
    assert not result["passed"] and (tmp_path/"overfit_32/audit_report.json").is_file() and (tmp_path/"overfit_32/readiness_decision.json").is_file()
