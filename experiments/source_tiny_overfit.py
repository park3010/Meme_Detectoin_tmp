"""Source-only 32-sample memorization diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.labels import LabelVocab, NormalizedMemeDataset
from experiments.ablation_configs import runtime_config_for_ablation
from experiments.gradient_forensics import HEAD_PREFIX, TASKS, _label_order, _logits, _model
from experiments.research_protocol import sha256_file
from experiments.source_sanity_gate import ACTIVE_SPLIT, ACTIVE_SPLIT_SHA256, atomic_write_json, validate_gradient_gate
from experiments.storage_safety import atomic_torch_save, storage_preflight
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from utils.io import load_yaml
from utils.seed import set_seed

EXCLUDED_PATH = Path("result/splits/harmeme/excluded_samples_v2.jsonl")
MAX_DIAGNOSTIC_EPOCHS = 30


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_overfit_manifest(path: str | Path, split_path: str | Path = ACTIVE_SPLIT) -> dict[str, Any]:
    path = Path(path); split_path = Path(split_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = {str(row["sample_key"]) for row in split["train"]}
    validation = {str(row["sample_key"]) for row in split["validation"]}
    excluded = {json.loads(line)["sample_key"] for line in EXCLUDED_PATH.read_text().splitlines() if line.strip()}
    rows = list(manifest.get("samples", [])); keys = [str(row["sample_key"]) for row in rows]
    forbidden = [key for key in keys if key.lower().startswith(("facebook::", "fhm::", "memotion::"))]
    errors=[]
    if len(rows) != 32: errors.append("overfit_manifest_wrong_size")
    if len(set(keys)) != len(keys): errors.append("overfit_manifest_duplicate_ids")
    if set(keys)-train: errors.append("overfit_manifest_non_train_ids")
    if set(keys)&validation: errors.append("overfit_manifest_validation_ids")
    if set(keys)&excluded: errors.append("overfit_manifest_excluded_ids")
    if forbidden: errors.append("overfit_manifest_forbidden_ids")
    domains={key.split("::",1)[0] for key in keys}; labels={str(row.get("harmfulness")) for row in rows}
    if not {"harm_c","harm_p"} <= domains: errors.append("overfit_manifest_domain_coverage")
    if not {"harmful","non_harmful"} <= labels: errors.append("overfit_manifest_harmfulness_coverage")
    return {"passed":not errors,"errors":errors,"sample_count":len(rows),"sample_keys":keys,"excluded_id_intersection_count":len(set(keys)&excluded),"validation_id_intersection_count":len(set(keys)&validation),"source_split_sha256":sha256_file(split_path),"manifest_sha256":sha256_file(path),"manifest":manifest}


def _evaluate(model, samples, loss_fn, ablation, vocab):
    model.eval(); sums=defaultdict(float); counts=Counter(); correct=0; harmful_scores=[]; predictions=defaultdict(list); golds=defaultdict(list)
    with torch.no_grad():
        for sample in samples:
            stage_e=model(sample,ablation=ablation)["stage_e"]; supervision=extract_supervision_from_annotation(sample); losses=loss_fn(stage_e,supervision)
            for task in TASKS:
                if task in losses: sums[task]+=float(losses[task].cpu());counts[task]+=1
            sums["total"]+=float(losses["total"].cpu());counts["total"]+=1
            harmful_logits=_logits(stage_e,"harmfulness"); prob=torch.softmax(harmful_logits,dim=-1); order=_label_order(vocab,"harmfulness"); pred=order[int(prob.argmax())]; gold=str(supervision["harmfulness"])
            correct+=int(pred==gold); harmful_scores.append(float(prob[order.index("harmful")].cpu()))
            for task in TASKS:
                if task not in supervision: continue
                logits=_logits(stage_e,task); order=_label_order(vocab,task)
                if task=="tactic_rhetorical":
                    predicted=[order[i] for i,v in enumerate(torch.sigmoid(logits).cpu()) if float(v)>=0.5]; gold=sorted(str(x) for x in supervision[task]); predictions[task].append(predicted);golds[task].append(gold)
                else:
                    predictions[task].append(order[int(logits.argmax().cpu())]);golds[task].append(str(supervision[task]))
    distribution={}
    for task in TASKS:
        pred=predictions[task]
        if task=="tactic_rhetorical":
            cardinalities=[len(x) for x in pred]; width=len(_label_order(vocab,task)); flat=Counter(x for row in pred for x in row)
            distribution[task]={"valid_n":len(pred),"average_predicted_cardinality":sum(cardinalities)/len(cardinalities) if cardinalities else None,"all_zero_ratio":sum(x==0 for x in cardinalities)/len(cardinalities) if cardinalities else None,"all_one_ratio":sum(x==width for x in cardinalities)/len(cardinalities) if cardinalities else None,"per_label_predictions":dict(flat),"mapping_status":"passed"}
        else:
            c=Counter(pred); distribution[task]={"valid_n":len(pred),"predicted_distribution":dict(c),"gold_distribution":dict(Counter(golds[task])),"unique_predicted_class_count":len(c),"mapping_status":"passed"}
    return {"losses":{k:sums[k]/counts[k] for k in counts},"harmfulness_accuracy":correct/len(samples),"harmfulness_score_variance":float(torch.tensor(harmful_scores).var(unbiased=False)) if harmful_scores else 0.0,"prediction_distributions":distribution}


def run_overfit_32(*, output_root: str, config: str, seed: int, device: str, force: bool, disable_tqdm: bool=False) -> dict[str, Any]:
    del disable_tqdm
    root=Path(output_root)
    immutable_prior=root/"overfit_32"
    out=(root/"tiny_overfit_forensics"/"diagnostic_runs"/"post_repair_overfit_32") if (immutable_prior/"training_curve.json").is_file() else immutable_prior
    out.mkdir(parents=True,exist_ok=True)
    gate_validation=validate_gradient_gate(output_root=root,config_path=config,expected_diagnostic_manifest_sha256="43205c49a4c679c72b8f83f073f8a1e12bcf33fa9a5aec7ceb4d1e26d0d0e85f")
    atomic_write_json(out/"audit_report.json",{"phase":"gradient_gate","source_only":True,**{k:v for k,v in gate_validation.items() if k!="gate"}})
    if not gate_validation["passed"]:
        result={"profile":"overfit_32","passed":False,"decision":"BLOCKED_MULTIPLE","blockers":gate_validation["reasons"],"source_only":True}
        atomic_write_json(out/"readiness_decision.json",result);(out/"audit_report.md").write_text("# Overfit-32 audit\n\nGate rejected: "+", ".join(gate_validation["reasons"])+"\n")
        return result
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        result={"profile":"overfit_32","passed":False,"decision":"BLOCKED_MULTIPLE","blockers":["overfit_cuda_unavailable"],"source_only":True};atomic_write_json(out/"readiness_decision.json",result);return result
    if any(out.iterdir()) and not force and (out/"training_curve.json").exists(): raise FileExistsError(f"output exists; use --force: {out}")
    manifest_path=root/"manifests"/"overfit_32.json"; manifest_check=validate_overfit_manifest(manifest_path)
    if not manifest_check["passed"]:
        result={"profile":"overfit_32","passed":False,"decision":"BLOCKED_LABEL_MAPPING","blockers":manifest_check["errors"],"source_only":True};atomic_write_json(out/"readiness_decision.json",result);return result
    manifest_payload={**manifest_check["manifest"],"active_source_split_path":str(ACTIVE_SPLIT),"active_source_split_sha256":ACTIVE_SPLIT_SHA256,"excluded_id_intersection_count":0,"validation_id_intersection_count":0,"diagnostic_only":True,"paper_eligible":False}
    atomic_write_json(out/"manifest.json",manifest_payload); (out/"manifest.json.sha256").write_text(f"{sha256_file(out/'manifest.json')}  manifest.json\n")
    atomic_write_json(out/"gradient_gate_snapshot.json",gate_validation["gate"])
    cfg=load_yaml(config);cfg.setdefault("runtime",{})["device"]=device;cfg["runtime"]["disable_retrieval"]=True
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=True),encoding="utf-8")
    atomic_write_json(out/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"source_only":True})
    dataset=NormalizedMemeDataset(dataset_names=["harm_c","harm_p"],label_set="full",keep_missing_images=True)
    by_key={f"{s['dataset_name']}::{s['sample_id']}":s for s in dataset.samples};samples=[by_key[key] for key in manifest_check["sample_keys"]]
    set_seed(seed);model,optimizer=_model(cfg,config,device,samples);loss_fn=StructuredMemeLoss();ablation=runtime_config_for_ablation("w_o_retrieval")
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    try:
        storage=storage_preflight(out,estimated_checkpoint_bytes=max(parameter_bytes,64*1024*1024),checkpoint_count=1)
    except RuntimeError as exc:
        storage={"status":"block","error":str(exc)}
    atomic_write_json(out/"storage_preflight.json",storage)
    if not storage.get("passed",storage.get("status") in {"pass","passed"}):
        result={"profile":"overfit_32","passed":False,"decision":"BLOCKED_STORAGE","blockers":["storage_preflight_failed"],"source_only":True};atomic_write_json(out/"readiness_decision.json",result);return result
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and any(n.startswith(prefix) for prefix in HEAD_PREFIX.values())}
    started=time.perf_counter(); initial=_evaluate(model,samples,loss_fn,ablation,LabelVocab.from_yaml()); curve=[]; per_head=defaultdict(list); gradient_seen=Counter(); update_seen=Counter()
    for epoch in range(1,MAX_DIAGNOSTIC_EPOCHS+1):
        model.train();sums=defaultdict(float);counts=Counter()
        for sample in samples:
            optimizer.zero_grad(set_to_none=True);stage_e=model(sample,ablation=ablation)["stage_e"];supervision=extract_supervision_from_annotation(sample);losses=loss_fn(stage_e,supervision);loss=losses["total"]
            loss.backward()
            for task,prefix in HEAD_PREFIX.items():
                if task in losses and any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n.startswith(prefix)): gradient_seen[task]+=1
            optimizer.step()
            for name,value in losses.items(): sums[name]+=float(value.detach().cpu());counts[name]+=1
        evaluated=_evaluate(model,samples,loss_fn,ablation,LabelVocab.from_yaml()); row={"epoch":epoch,"total_loss":evaluated["losses"]["total"],"harmfulness_accuracy":evaluated["harmfulness_accuracy"],**{f"{t}_loss":evaluated["losses"].get(t) for t in TASKS}};curve.append(row)
        for task in TASKS: per_head[task].append(evaluated["losses"].get(task))
        if epoch>=5 and evaluated["harmfulness_accuracy"]>=0.95 and all(evaluated["losses"].get(t,math.inf)<initial["losses"].get(t,-math.inf) for t in TASKS): break
    final=_evaluate(model,samples,loss_fn,ablation,LabelVocab.from_yaml());runtime_seconds=time.perf_counter()-started
    update_rows=[]
    for name,p in model.named_parameters():
        if name not in before: continue
        norm=float((p.detach()-before[name]).float().norm().cpu());task=next(t for t,prefix in HEAD_PREFIX.items() if name.startswith(prefix));update_seen[task]+=int(norm>0);update_rows.append({"parameter_name":name,"task":task,"update_norm":norm,"finite":bool(torch.isfinite(p).all())})
    per_head_metrics={task:{"initial_loss":initial["losses"].get(task),"final_loss":final["losses"].get(task),"loss_decreased":final["losses"].get(task,math.inf)<initial["losses"].get(task,-math.inf),"finite":math.isfinite(final["losses"].get(task,math.inf)),"nonzero_gradient_steps":gradient_seen[task],"updated_parameter_count":update_seen[task],"valid_n":sum(task in extract_supervision_from_annotation(s) for s in samples)} for task in TASKS}
    no_nonfinite=all(math.isfinite(float(x)) for row in curve for x in row.values() if x is not None)
    learned=all(v["finite"] and v["loss_decreased"] and v["nonzero_gradient_steps"]>0 and v["updated_parameter_count"]>0 and v["valid_n"]>0 for v in per_head_metrics.values())
    passed=final["harmfulness_accuracy"]>=.95 and final["harmfulness_score_variance"]>0 and no_nonfinite and learned
    checkpoint={"schema_version":"diagnostic_trainable_state_v1","epoch":curve[-1]["epoch"],"model_trainable_state_dict":{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},"source_split_sha256":ACTIVE_SPLIT_SHA256,"manifest_sha256":sha256_file(out/"manifest.json"),"paper_eligible":False}
    atomic_torch_save(checkpoint,out/"best_model.pt");checkpoint_integrity={"passed":True,"path":str(out/"best_model.pt"),"sha256":sha256_file(out/"best_model.pt"),"size_bytes":(out/"best_model.pt").stat().st_size,"atomic_save":True,"diagnostic_only":True};atomic_write_json(out/"checkpoint_integrity.json",checkpoint_integrity)
    columns=list(curve[0]);
    with (out/"training_curve.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(curve)
    atomic_write_json(out/"training_curve.json",{"initial":initial,"epochs":curve,"final":final});atomic_write_json(out/"per_head_training_metrics.json",per_head_metrics);atomic_write_json(out/"per_head_loss_curves.json",dict(per_head));atomic_write_json(out/"prediction_distributions.json",final["prediction_distributions"]);atomic_write_json(out/"gradient_summary.json",{"nonzero_gradient_steps":dict(gradient_seen),"all_valid_heads_received_gradients":all(gradient_seen[t]>0 for t in TASKS)});atomic_write_json(out/"parameter_update_summary.json",{"rows":update_rows,"all_valid_heads_updated":all(update_seen[t]>0 for t in TASKS)});atomic_write_json(out/"runtime.json",{"runtime_seconds_total":runtime_seconds,"training_seconds":runtime_seconds,"epochs_completed":len(curve),"retrieval_enabled":False})
    run_manifest={"schema_version":"source_overfit_32_v1","profile":"overfit_32","condition":"ablation_w_o_retrieval","source_only":True,"paper_eligible":False,"seed":seed,"device":device,"source_split_sha256":ACTIVE_SPLIT_SHA256,"manifest_sha256":sha256_file(out/"manifest.json"),"gradient_gate_sha256":gate_validation["gate_sha256"],"fhm_or_memotion_access_count":0,"validation_early_stopping":False,"diagnostic_epoch_override":MAX_DIAGNOSTIC_EPOCHS,"created_at_utc":datetime.now(timezone.utc).isoformat()};atomic_write_json(out/"run_manifest.json",run_manifest)
    blockers=[]
    if final["harmfulness_accuracy"]<.95: blockers.append("harmfulness_memorization_failed")
    if not learned: blockers.append("structured_head_learning_failed")
    if not no_nonfinite: blockers.append("nonfinite_training_value")
    decision="READY_FOR_OVERFIT_128" if passed else "BLOCKED_TINY_OVERFIT"
    result={"profile":"overfit_32","passed":passed,"decision":decision,"blockers":blockers,"initial_total_loss":initial["losses"]["total"],"final_total_loss":final["losses"]["total"],"harmfulness_final_training_accuracy":final["harmfulness_accuracy"],"per_head":per_head_metrics,"source_only":True,"fhm_or_memotion_access_count":0,"checkpoint_integrity":checkpoint_integrity}
    atomic_write_json(out/"audit_report.json",{**result,"gate_validation_passed":True,"manifest_validation":{k:v for k,v in manifest_check.items() if k!="manifest"},"storage_preflight_passed":True});(out/"audit_report.md").write_text(f"# Overfit-32 audit\n\nDecision: **{decision}**\n\nInitial total loss: {result['initial_total_loss']:.6f}; final: {result['final_total_loss']:.6f}; harmfulness accuracy: {result['harmfulness_final_training_accuracy']:.4f}.\n",encoding="utf-8");atomic_write_json(out/"readiness_decision.json",result)
    return result


__all__=["run_overfit_32","validate_overfit_manifest"]
