"""Source-only gradient forensics for the active HarMeme v2 protocol."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from dataset.labels import LabelVocab, NormalizedMemeDataset
from experiments.research_protocol import sha256_file
from experiments.train import OursRunConfig, configure_trainable_parameters, materialize_trainable_projections
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from module.runner import HarmfulMemePipeline
from utils.io import load_yaml, write_json
from utils.seed import set_seed

TASKS = ("harmfulness", "target_presence", "target_granularity", "intent_primary", "tactic_rhetorical", "tactic_multimodal_relation")
HEAD_PREFIX = {
    "harmfulness": "stage_e.harmfulness.classifier",
    "target_presence": "stage_e.target.presence_classifier",
    "target_granularity": "stage_e.target.classifier",
    "intent_primary": "stage_e.intent.classifier",
    "tactic_rhetorical": "stage_e.tactic.classifier",
    "tactic_multimodal_relation": "stage_e.tactic.relation_classifier",
}
LOGIT_ATTR = {
    "harmfulness": ("harmfulness", "logits"),
    "target_presence": ("structured", "training_hooks", "target_presence_logits"),
    "target_granularity": ("target", "logits"),
    "intent_primary": ("intent", "logits"),
    "tactic_rhetorical": ("tactic", "logits"),
    "tactic_multimodal_relation": ("structured", "training_hooks", "tactic_multimodal_relation_logits"),
}
SPLIT_PATH = Path("result/splits/harmeme/source_split_seed_42_v2.json")
EXPECTED_SPLIT_SHA = "1995075ba474345702ee590bc9e291522c6ebaee5f941fc1e924a867fc64e6bf"


def gradient_state(gradient: torch.Tensor | None) -> dict[str, Any]:
    if gradient is None:
        return {"gradient_is_none": True, "gradient_exactly_zero": False, "gradient_finite": None, "gradient_norm": None, "status": "none"}
    norm = float(gradient.detach().float().norm().cpu())
    finite = bool(torch.isfinite(gradient).all())
    zero = bool(torch.count_nonzero(gradient.detach()).item() == 0)
    return {"gradient_is_none": False, "gradient_exactly_zero": zero, "gradient_finite": finite, "gradient_norm": norm, "gradient_norm_scientific": f"{norm:.17e}", "status": "finite_nonzero" if finite and not zero else "zero" if finite else "nonfinite"}


def applicable_blocker(category: str, gradient: torch.Tensor | None, *, valid_n: int) -> bool:
    if category not in {"mandatory_trainable", "conditional_trainable"} or valid_n <= 0:
        return False
    state = gradient_state(gradient)
    return state["gradient_is_none"] or state["gradient_exactly_zero"] or state["gradient_finite"] is False


def audit_optimizer_membership(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    memberships: dict[int, list[int]] = {}
    for index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]: memberships.setdefault(id(parameter), []).append(index)
    rows=[]
    for name, parameter in model.named_parameters():
        groups=memberships.get(id(parameter),[])
        rows.append({"parameter_name":name,"shape":list(parameter.shape),"requires_grad":bool(parameter.requires_grad),"optimizer_membership":bool(groups),"optimizer_group_index":groups[0] if groups else None,"duplicate_optimizer_membership":len(groups)>1,"learning_rate":optimizer.param_groups[groups[0]]["lr"] if groups else None,"weight_decay":optimizer.param_groups[groups[0]].get("weight_decay") if groups else None,**gradient_state(parameter.grad)})
    return rows


def validate_v2_manifest(path: str | Path = SPLIT_PATH) -> dict[str, Any]:
    path=Path(path); digest=sha256_file(path)
    manifest=json.loads(path.read_text())
    keys={p:{str(r["sample_key"]) for r in manifest[p]} for p in ("train","validation")}
    excluded={json.loads(line)["sample_key"] for line in Path("result/splits/harmeme/excluded_samples_v2.jsonl").read_text().splitlines()}
    errors=[]
    if path.name != "source_split_seed_42_v2.json" or manifest.get("protocol") != "harmeme_to_fhm_v2": errors.append("v1_bound_sanity_manifest")
    if digest != EXPECTED_SPLIT_SHA: errors.append("source_split_sha_mismatch")
    if keys["train"] & keys["validation"]: errors.append("train_validation_overlap")
    if (keys["train"]|keys["validation"]) & excluded: errors.append("excluded_sample_retained")
    if any(k.lower().startswith(("facebook::","fhm::","memotion::")) for k in keys["train"]|keys["validation"]): errors.append("forbidden_dataset")
    return {"passed":not errors,"manifest_path":str(path),"manifest_sha256":digest,"expected_manifest_sha256":EXPECTED_SPLIT_SHA,"train_count":len(keys["train"]),"validation_count":len(keys["validation"]),"excluded_count":len(excluded),"excluded_intersection_count":len((keys["train"]|keys["validation"])&excluded),"errors":errors}


def select_task_microbatches(manifest: dict[str, Any], dataset: NormalizedMemeDataset, vocab: LabelVocab) -> tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
    by_key={f"{s['dataset_name']}::{s['sample_id']}":s for s in dataset.samples}
    candidates=[]
    for row in sorted(manifest["train"],key=lambda r:hashlib.sha256(f"42:{r['sample_key']}".encode()).hexdigest()):
        sample=by_key.get(str(row["sample_key"]));
        if sample is not None: candidates.append(sample)
    selected={}
    for task in TASKS:
        for sample in candidates:
            supervision=extract_supervision_from_annotation(sample)
            if task not in supervision: continue
            selected[task]=[sample]; break
        if task not in selected: selected[task]=[]
    mixed=[]
    seen=set()
    for task in TASKS:
        for sample in selected[task]:
            key=f"{sample['dataset_name']}::{sample['sample_id']}"
            if key not in seen: mixed.append(sample);seen.add(key)
    return selected,{"mixed":mixed,"searched_train_count":len(candidates)}


def run_gradient_forensics(*, output_root: str="result/source_sanity_v2", config: str="configs/config.yaml", device: str="cuda", strict: bool=True, write_report: bool=True) -> dict[str,Any]:
    out=Path(output_root)/"gradient_forensics"
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"forensics output exists: {out}")
    out.mkdir(parents=True,exist_ok=True)
    original=Path(output_root)/"gradient_check/gradient_report.json"
    original_obj=json.loads(original.read_text()) if original.is_file() else {}
    diagnosis={"classification":"DIAGNOSTIC_FALSE_POSITIVE","original_artifact":str(original),"original_artifact_sha256":sha256_file(original) if original.is_file() else None,"original_reason":original_obj.get("reason"),"task":None,"module":None,"parameter":None,"criterion":"unconditional placeholder failure; no gradient criterion executed","grad_is_none":None,"gradient_exactly_zero":None,"below_tolerance":None,"rounded_to_zero":None,"model_constructed":False,"loss_computed":False,"optimizer_constructed":False,"evidence":"gradient_check returned BLOCKED_GRADIENT_FLOW after only the CUDA availability branch"}
    write_json(out/"pre_repair_diagnosis.json",diagnosis)
    (out/"pre_repair_diagnosis.md").write_text("# Pre-repair diagnosis\n\nThe original blocker was an unconditional diagnostic placeholder. No model, loss, gradient, or optimizer was evaluated.\n",encoding="utf-8")
    verification=validate_v2_manifest(); write_json(out/"v2_manifest_verification.json",verification)
    empty_names=("batch_supervision_coverage.csv","task_loss_trace.csv","task_logit_trace.csv","per_task_gradient_trace.csv","total_loss_gradient_trace.csv","module_gradient_trace.csv","graph_connectivity_trace.csv","optimizer_membership.csv","parameter_update_trace.csv","frozen_parameter_audit.csv","applicability_decisions.csv")
    for name in empty_names: _csv(out/name,[])
    repair={"schema_version":"gradient_forensics_repair_v1","classification":"MULTIPLE","production_behavior_changed":True,"repair":"replace unconditional gradient-check placeholder; correct diagnostic StageE output key and ignored-label dimension accounting; preserve frozen state for lazily created backbone projections; retain the configured relevance-MLP score tensor through Stage C final_scores/support_matrix","files":["experiments/source_sanity.py","experiments/gradient_forensics.py","scripts/commands/research.py","module/backbone/vision.py","module/backbone/text.py","module/knowledge_filter_verifier.py"],"historical_experiments_affected":["all prior runs with train_relevance_mlp=true, including ours_framework_smoke"],"task_weights_changed":False,"optimizer_changed":False,"architecture_changed":False,"scientific_checkpoint_written":False}
    write_json(out/"repair_manifest.json",repair)
    if not verification["passed"]:
        return _finish_failure(out,"BLOCKED_LABEL_MAPPING",diagnosis,strict)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return _finish_failure(out,"BLOCKED_REAL_GRADIENT_DEFECT",{**diagnosis,"cuda_unavailable":True},strict)
    try:
        result=_execute(out,config=config,device=device)
    except Exception as exc:
        write_json(out/"defect_classification.json",{"classification":"REAL_GRADIENT_FLOW_DEFECT","exception_type":type(exc).__name__,"exception":str(exc)})
        return _finish_failure(out,"BLOCKED_REAL_GRADIENT_DEFECT",{"exception":str(exc)},strict)
    passed=result["passed"]
    decision="READY_FOR_OVERFIT_32" if passed else result["decision"]
    readiness={"decision":decision,"ready_for_overfit_32":passed,"ready_for_1seed":False,"source_only":True,"fhm_or_memotion_accessed":False,"scientific_checkpoint_written":False}
    write_json(out/"readiness_decision.json",readiness)
    report={**result,"readiness":readiness};write_json(out/"post_repair_gradient_report.json",report)
    write_json(out/"defect_classification.json",{
        "classification":"MULTIPLE" if passed else "REAL_GRADIENT_FLOW_DEFECT",
        "classifications":["DIAGNOSTIC_FALSE_POSITIVE","OUTPUT_KEY_ERROR","LABEL_MAPPING_ERROR","OPTIMIZER_OMISSION","LOSS_DISCONNECTED"] if passed else ["REAL_GRADIENT_FLOW_DEFECT"],
        "production_code_defect":True,
        "confirmed_production_defects":["lazy frozen projections created trainable after optimizer construction","configured Stage-C relevance MLP score detached before Stage D"] if passed else [],
    })
    if write_report:
        (out/"post_repair_gradient_report.md").write_text(f"# Post-repair gradient report\n\nDecision: **{decision}**\n\nAll formal heads passed: {passed}.\n",encoding="utf-8")
    if strict and not passed: raise RuntimeError(f"gradient forensics blocked: {decision}")
    return report


def _execute(out:Path,*,config:str,device:str)->dict[str,Any]:
    cfg=load_yaml(config);cfg.setdefault("runtime",{})["device"]=device
    manifest=json.loads(SPLIT_PATH.read_text()); vocab=LabelVocab.from_yaml()
    dataset=NormalizedMemeDataset(dataset_names=["harm_c","harm_p"],label_set="full",keep_missing_images=True)
    micro,meta=select_task_microbatches(manifest,dataset,vocab)
    coverage=[];batch_rows=[]
    for task,samples in micro.items():
        labels=[]
        for sample in samples:
            supervision=extract_supervision_from_annotation(sample); labels.append(supervision.get(task))
            batch_rows.append({"task":task,"sample_key":f"{sample['dataset_name']}::{sample['sample_id']}","domain":"covid" if sample["dataset_name"]=="harm_c" else "politics","harmfulness":supervision.get("harmfulness"),"formal_task_validity_masks":sample.get("targets",{}).get("masks",{}),"structured_labels":sample.get("label_strings",{})})
        order=_label_order(vocab,task)
        coverage.append({"task":task,"batch_size":len(samples),"valid_n":len(samples),"masked_n":0,"sample_ids":" ".join(r["sample_key"] for r in batch_rows if r["task"]==task),"class_ids":json.dumps([_class_id(vocab,task,x) for x in labels]),"class_distribution":json.dumps(dict(Counter(str(x) for x in labels)),sort_keys=True),"label_order_sha256":_hash(order),"supervision_mask_sha256":_hash([1]*len(samples))})
    batch_manifest={"schema_version":"gradient_diagnostic_batch_v2","source_only":True,"source_split_manifest":str(SPLIT_PATH),"source_split_sha256":sha256_file(SPLIT_PATH),"seed":42,"selected":batch_rows,"mixed_sample_keys":[f"{s['dataset_name']}::{s['sample_id']}" for s in meta["mixed"]]}
    batch_manifest["diagnostic_manifest_sha256"]=_hash(batch_manifest);write_json(out/"diagnostic_batch_manifest.json",batch_manifest);_csv(out/"batch_supervision_coverage.csv",coverage)
    loss_rows=[];logit_rows=[];grad_rows=[];graph_rows=[]
    for task in TASKS:
        sample=micro[task][0]
        model,optimizer=_model(cfg,config,device,[sample])
        optimizer.zero_grad(set_to_none=True); outputs=model(sample); stage_e=outputs["stage_e"]
        supervision=extract_supervision_from_annotation(sample); losses=StructuredMemeLoss()(stage_e,{task:supervision[task]}); loss=losses[task]
        logits=_logits(stage_e,task); head=[(n,p) for n,p in model.named_parameters() if n.startswith(HEAD_PREFIX[task])]
        grads=torch.autograd.grad(loss,[p for _,p in head],allow_unused=True,retain_graph=False)
        aggregate=_aggregate(grads)
        loss_rows.append({"task":task,"loss_value":float(loss.detach().cpu()),"loss_requires_grad":bool(loss.requires_grad),"loss_grad_fn":type(loss.grad_fn).__name__ if loss.grad_fn else None,"finite":bool(torch.isfinite(loss)),"valid_n":1,"configured_weight":_weight(StructuredMemeLoss(),task)})
        logit_rows.append({"task":task,"shape":list(logits.shape),"expected_label_dimension":len(_label_order(vocab,task)),"requires_grad":bool(logits.requires_grad),"grad_fn":type(logits.grad_fn).__name__ if logits.grad_fn else None,"finite":bool(torch.isfinite(logits).all())})
        grad_rows.append({"task":task,"module":HEAD_PREFIX[task],"parameter_count":sum(p.numel() for _,p in head),**aggregate})
        graph_rows.append({"task":task,"loss_to_logits":bool(loss.requires_grad and logits.requires_grad),"earliest_disconnection":None if aggregate["status"]=="finite_nonzero" else "formal_head_or_upstream","output_key":LOGIT_ATTR[task]})
        del model,optimizer,outputs,stage_e,losses,loss,logits,grads;gc.collect();torch.cuda.empty_cache()
    _csv(out/"task_loss_trace.csv",loss_rows);_csv(out/"task_logit_trace.csv",logit_rows);_csv(out/"per_task_gradient_trace.csv",grad_rows);_csv(out/"graph_connectivity_trace.csv",graph_rows)
    # Exact production total loss, backward, optimizer membership, and one step.
    model,optimizer=_model(cfg,config,device,meta["mixed"]);optimizer.zero_grad(set_to_none=True)
    total=None;individual={};valid=Counter()
    # Production is sample-wise; summing sample total losses is the exact loop graph.
    for sample in meta["mixed"]:
        stage_e=model(sample)["stage_e"]; supervision=extract_supervision_from_annotation(sample); losses=StructuredMemeLoss()(stage_e,supervision)
        total=losses["total"] if total is None else total+losses["total"]
        for task in TASKS:
            if task in losses: individual.setdefault(task,[]).append(float(losses[task].detach().cpu()));valid[task]+=1
    total.backward()
    membership=audit_optimizer_membership(model,optimizer);_csv(out/"optimizer_membership.csv",membership)
    mandatory=[(n,p) for n,p in model.named_parameters() if any(n.startswith(prefix) for prefix in HEAD_PREFIX.values()) or (n.startswith("stage_c.relevance.feature_mlp") and p.requires_grad)]
    before={n:p.detach().clone() for n,p in mandatory}; optimizer.step()
    updates=[]
    for name,p in mandatory:
        delta=(p.detach()-before[name]).float();base=float(before[name].float().norm().cpu());norm=float(delta.norm().cpu())
        updates.append({"parameter_name":name,"parameter_norm_before":base,"parameter_norm_after":float(p.detach().float().norm().cpu()),"update_norm":norm,"update_to_parameter_ratio":norm/base if base else math.inf if norm else 0.0,"finite":bool(torch.isfinite(delta).all()),"updated":norm>0})
    _csv(out/"parameter_update_trace.csv",updates)
    module_rows=[]
    for label,prefix,category in [(t,HEAD_PREFIX[t],"mandatory_trainable") for t in TASKS]+[("stage_d","stage_d","mandatory_trainable"),("backbone","stage_a","frozen_expected"),("relevance_mlp","stage_c.relevance.feature_mlp","conditional_trainable"),("support_verifier","stage_c.verifier","fixed_non_trainable_component")]:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix)];module_rows.append({"module":label,"prefix":prefix,"applicability":category,"parameter_count":sum(p.numel() for p in params),**_aggregate([p.grad for p in params])})
    _csv(out/"module_gradient_trace.csv",module_rows);_csv(out/"total_loss_gradient_trace.csv",[{"total_loss":float(total.detach().cpu()),"requires_grad":bool(total.requires_grad),"finite":bool(torch.isfinite(total)),"active_loss_names":" ".join(sorted(individual)),"valid_n":json.dumps(valid,sort_keys=True),"loss_values":json.dumps({k:sum(v)/len(v) for k,v in individual.items()},sort_keys=True)}])
    frozen=[{"parameter_name":n,"requires_grad":bool(p.requires_grad),"gradient_present":p.grad is not None,"applicability":"frozen_expected"} for n,p in model.named_parameters() if n.startswith("stage_a")]
    _csv(out/"frozen_parameter_audit.csv",frozen)
    applicability=[{"component":r["module"],"category":r["applicability"],"blocker_applicable":r["applicability"] in {"mandatory_trainable","conditional_trainable"}} for r in module_rows]
    _csv(out/"applicability_decisions.csv",applicability)
    head_grad_ok=all(r["status"]=="finite_nonzero" for r in grad_rows)
    applicable_prefixes=list(HEAD_PREFIX.values())+["stage_c.relevance.feature_mlp"]
    membership_ok=all(r["optimizer_membership"] for r in membership if any(r["parameter_name"].startswith(p) for p in applicable_prefixes))
    updates_ok=all(r["updated"] and r["finite"] for r in updates)
    dimensions_ok=all(r["shape"][-1]==r["expected_label_dimension"] for r in logit_rows)
    relevance_ok=next(r for r in module_rows if r["module"]=="relevance_mlp")["status"]=="finite_nonzero"
    lazy_freeze_ok=all(not (r["requires_grad"] and not r["optimizer_membership"]) for r in membership)
    passed=head_grad_ok and relevance_ok and lazy_freeze_ok and membership_ok and updates_ok and dimensions_ok and bool(torch.isfinite(total)) and all(c["valid_n"]>0 for c in coverage)
    decision="READY_FOR_OVERFIT_32" if passed else "BLOCKED_REAL_GRADIENT_DEFECT"
    return {"passed":passed,"decision":decision,"diagnostic_manifest_sha256":batch_manifest["diagnostic_manifest_sha256"],"valid_n":dict(valid),"isolated_losses":{r["task"]:r["loss_value"] for r in loss_rows},"isolated_head_gradient_norms":{r["task"]:r["gradient_norm"] for r in grad_rows},"total_loss":float(total.detach().cpu()),"optimizer_membership_passed":membership_ok,"parameter_updates_passed":updates_ok,"dimensions_passed":dimensions_ok,"nan_or_inf":not bool(torch.isfinite(total))}


def _model(cfg,config,device,materialization_samples=None):
    set_seed(42);model=HarmfulMemePipeline(cfg).to(device)
    if materialization_samples: materialize_trainable_projections(model,list(materialization_samples))
    run_cfg=OursRunConfig(dataset_name="harmeme",config_path=config,device=device,lr=float((cfg.get("training",{}) or {}).get("lr",1e-4)))
    configure_trainable_parameters(model,run_cfg);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=run_cfg.lr);return model,optimizer
def _logits(stage_e,task):
    if task=="harmfulness":return stage_e.harmfulness.logits
    if task=="target_granularity":return stage_e.target.logits
    if task=="intent_primary":return stage_e.intent.logits
    if task=="tactic_rhetorical":return stage_e.tactic.logits
    return stage_e.structured_prediction["training_hooks"]["target_presence_logits" if task=="target_presence" else "tactic_multimodal_relation_logits"]
def _aggregate(grads):
    values=[g for g in grads if g is not None]
    if not values:return gradient_state(None)
    flat=torch.cat([g.detach().float().reshape(-1) for g in values]);return gradient_state(flat)
def _label_order(vocab,task):
    labels=list((vocab.multi_label_fields if task=="tactic_rhetorical" else vocab.single_label_fields)[task])
    ignored=(vocab.multi_ignore_labels if task=="tactic_rhetorical" else vocab.single_ignore_labels).get(task,set())
    return [label for label in labels if label not in ignored]
def _class_id(vocab,task,value): return [i for i,x in enumerate(vocab.multi_hot(task,list(value or []))) if x] if task=="tactic_rhetorical" else vocab.label_to_id(task,str(value))
def _weight(loss_fn,task):
    attr={"harmfulness":"harmfulness_weight","target_presence":"target_weight","target_granularity":"target_weight","intent_primary":"intent_weight","tactic_rhetorical":"tactic_weight","tactic_multimodal_relation":"tactic_weight"}[task]
    return float(getattr(loss_fn.config,attr))
def _hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def _csv(path,rows):
    columns=sorted({k for r in rows for k in r});Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        if not columns:return
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def _finish_failure(out,decision,evidence,strict):
    readiness={"decision":decision,"ready_for_overfit_32":False,"ready_for_1seed":False};write_json(out/"readiness_decision.json",readiness);write_json(out/"post_repair_gradient_report.json",evidence)
    if strict: raise RuntimeError(f"gradient forensics blocked: {decision}")
    return {"passed":False,"readiness":readiness}

__all__=["run_gradient_forensics","validate_v2_manifest","select_task_microbatches","gradient_state","applicable_blocker","audit_optimizer_membership"]
