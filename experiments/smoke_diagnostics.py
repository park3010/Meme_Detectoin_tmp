"""Read-only forensics for completed engineering-smoke artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.evaluation import _roc_auc, evaluate_structured_predictions
from experiments.formal_tasks import FORMAL_TASKS
from experiments.research_protocol import sha256_file
from utils.io import read_jsonl, write_json
from experiments.tokenizer_compatibility import tokenizer_compatibility_report

SINGLE_TASKS = {
    "harmfulness": (lambda r: r.get("gold_harmfulness"), lambda r: r.get("pred_harmfulness")),
    "target_presence": (lambda r: (r.get("gold_target") or {}).get("target_presence"), lambda r: (r.get("target") or {}).get("presence")),
    "target_granularity": (lambda r: (r.get("gold_target") or {}).get("target_granularity"), lambda r: (r.get("target") or {}).get("granularity")),
    "intent_primary": (lambda r: (r.get("gold_intent") or {}).get("intent_primary"), lambda r: (r.get("intent") or {}).get("primary")),
    "tactic_multimodal_relation": (lambda r: (r.get("gold_tactic") or {}).get("tactic_multimodal_relation"), lambda r: (r.get("tactic") or {}).get("multimodal_relation")),
}
LOGITS = {
    "harmfulness": lambda r: (r.get("training_hooks") or {}).get("harmfulness_logits"),
    "target_presence": lambda r: (r.get("target") or {}).get("presence_logits"),
    "target_granularity": lambda r: (r.get("target") or {}).get("logits"),
    "intent_primary": lambda r: (r.get("intent") or {}).get("logits"),
    "tactic_rhetorical": lambda r: r.get("tactic_rhetorical_logits"),
    "tactic_multimodal_relation": lambda r: (r.get("tactic") or {}).get("multimodal_relation_logits"),
}


def canonical_prediction_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in sorted(rows, key=lambda x: str(x.get("sample_id"))))
    return hashlib.sha256(payload.encode()).hexdigest()


def index_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ids = [str(row.get("sample_id")) for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample IDs: {duplicates[:10]}")
    if any(value in {"", "None"} for value in ids):
        raise ValueError("missing sample ID")
    return dict(zip(ids, rows))


def diagnose_smoke(*, suite: str, output_root: str = "result", reference_experiment: str = "ours_full", strict: bool = False, write_report: bool = False, force: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    suite_root = root / "research_runs" / suite
    diag = root / "diagnostics"
    if diag.exists() and any(diag.iterdir()) and not force:
        raise FileExistsError(f"diagnostics already exist; use --force: {diag}")
    diag.mkdir(parents=True, exist_ok=True)
    runs = {path.parent.name: path for path in sorted(suite_root.glob("*/seed_*"))}
    if reference_experiment not in runs:
        raise FileNotFoundError(f"reference experiment not found: {reference_experiment}")
    files: dict[tuple[str, str], list[dict[str, Any]]] = {}
    hashes, integrity, collapse, logits, polarity, distributions = [], [], [], [], [], []
    validation_metric_rows: list[dict[str, Any]] = []
    validation_domain_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for experiment, run_dir in runs.items():
        for split, filename in (("test", "test_predictions.jsonl"), ("validation", "validation_predictions.jsonl")):
            path = run_dir / filename
            rows = read_jsonl(path)
            try:
                indexed = index_predictions(rows)
                duplicate_ids: list[str] = []
            except ValueError as exc:
                indexed, duplicate_ids = {}, [str(exc)]
                blockers.append("label_mapping_or_prediction_identity")
            ids = [str(row.get("sample_id")) for row in rows]
            files[(experiment, split)] = rows
            hashes.append({"experiment_id": experiment, "split": split, "path": str(path), "raw_sha256": sha256_file(path), "canonical_row_content_sha256": canonical_prediction_hash(rows), "sample_id_sequence_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "sample_count": len(rows)})
            integrity.append({"experiment_id": experiment, "split": split, "sample_count": len(rows), "duplicate_ids": duplicate_ids, "missing_ids": sum(i in {"", "None"} for i in ids), "status": "pass" if indexed else "blocker"})
            c, d, l = collapse_checks(rows, experiment, split)
            collapse.extend(c); distributions.extend(d); logits.extend(l)
            polarity.append(auroc_polarity(rows, experiment, split))
        val_rows = files[(experiment, "validation")]
        for dataset, domain, subset in (("harmeme", "pooled", val_rows), ("harm_c", "covid", [r for r in val_rows if r.get("dataset_name") == "harm_c" or r.get("domain") == "covid"]), ("harm_p", "politics", [r for r in val_rows if r.get("dataset_name") == "harm_p" or r.get("domain") == "politics"])):
            metrics = evaluate_structured_predictions(subset, disable_tqdm=True)
            target = validation_metric_rows if dataset == "harmeme" else validation_domain_rows
            target.extend(_metric_rows(experiment, dataset, domain, metrics))
    agreements, disagreements = agreement_checks(files, reference_experiment)
    raw_groups = _duplicate_groups(hashes, "raw_sha256")
    suspicious = [group for group in raw_groups if len({row["experiment_id"] for row in group}) > 1]
    if suspicious:
        blockers.append("prediction_reuse")
    contracts = [_ablation_contract(name, path) for name, path in runs.items()]
    if any(row["status"] != "pass" for row in contracts): blockers.append("ablation_contract")
    if any(row["status"] == "blocker" for row in collapse): blockers.append("prediction_collapse")
    evidence = [_evidence_semantics(name, files[(name, "test")]) for name in runs]
    runtime = [_runtime_row(name, path) for name, path in runs.items()]
    _csv(diag / "prediction_file_hashes.csv", hashes)
    _csv(diag / "prediction_agreement.csv", agreements)
    _csv(diag / "sample_disagreements.csv", disagreements)
    _csv(diag / "task_prediction_distributions.csv", distributions)
    _csv(diag / "logit_statistics.csv", logits)
    _csv(diag / "collapse_checks.csv", collapse)
    _csv(diag / "auroc_polarity_checks.csv", polarity)
    _csv(diag / "ablation_contract_checks.csv", contracts)
    _csv(diag / "evidence_metric_semantics.csv", evidence)
    _csv(diag / "validation_metrics.csv", validation_metric_rows)
    _csv(diag / "validation_domain_metrics.csv", validation_domain_rows)
    _csv(diag / "runtime_phase_summary.csv", runtime)
    _csv(diag / "manifest_integrity.csv", integrity)
    decision = _decision(blockers)
    readiness = {"decision": decision, "ready_for_1seed": False, "blockers": sorted(set(blockers)), "fhm_used_for_selection": False, "note": "READY_FOR_SOURCE_SANITY is not authorization to launch a one-seed experiment", "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(diag / "readiness_decision.json", readiness)
    report = {"schema_version": "smoke_diagnostics_v1", "suite": suite, "reference_experiment": reference_experiment, "formal_tasks": list(FORMAL_TASKS), "prediction_schema": sorted(files[(reference_experiment, "test")][0]), "prediction_hashes": hashes, "suspicious_artifact_reuse": bool(suspicious), "agreements": agreements, "collapse_checks": collapse, "auroc_polarity": polarity, "ablation_contracts": contracts, "evidence_semantics": evidence, "readiness": readiness}
    write_json(diag / "smoke_diagnostics.json", report)
    tokenizer_compatibility_report(diag)
    if write_report:
        (diag / "smoke_diagnostics.md").write_text(_markdown(report), encoding="utf-8")
    if strict and blockers:
        raise RuntimeError(f"strict smoke diagnostics blocked: {sorted(set(blockers))}")
    return report


def collapse_checks(rows: list[dict[str, Any]], experiment: str, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checks, distributions, stats = [], [], []
    for task, (gold_fn, pred_fn) in SINGLE_TASKS.items():
        gold, pred = [gold_fn(r) for r in rows], [pred_fn(r) for r in rows]
        pred_count = Counter(str(x) for x in pred if x is not None)
        gold_count = Counter(str(x) for x in gold if x is not None)
        n = sum(pred_count.values()); majority = max(pred_count.values(), default=0) / n if n else None
        severity = "blocker" if n == 0 else "warning" if len(pred_count) == 1 else "pass"
        checks.append({"experiment_id": experiment, "split": split, "task": task, "status": severity, "unique_predicted_class_count": len(pred_count), "majority_prediction_rate": majority, "reason": "zero_valid_predictions" if n == 0 else "single_class_prediction" if len(pred_count) == 1 else "ok"})
        distributions.append({"experiment_id": experiment, "split": split, "task": task, "gold_distribution": dict(gold_count), "predicted_distribution": dict(pred_count), "class_coverage": len(pred_count), "most_frequent_prediction": pred_count.most_common(1)[0][0] if pred_count else None, "normalized_prediction_entropy": _entropy(pred_count)})
    labels = rows[0].get("tactic_rhetorical_label_order", []) if rows else []
    sets = [set((r.get("evaluation") or {}).get("tactic_rhetorical_formal", {}).get("predicted_labels_with_none_fallback") or (r.get("tactic") or {}).get("rhetorical") or []) for r in rows]
    all_zero = sum(not (s - {"none"}) for s in sets) / len(sets) if sets else None
    all_one = sum(len(s - {"none"}) == len([x for x in labels if x != "none"]) for s in sets) / len(sets) if sets else None
    checks.append({"experiment_id": experiment, "split": split, "task": "tactic_rhetorical", "status": "warning" if all_zero == 1 or all_one == 1 else "pass", "all_zero_ratio": all_zero, "all_one_ratio": all_one, "reason": "all_zero_or_all_one" if all_zero == 1 or all_one == 1 else "ok"})
    distributions.append({"experiment_id": experiment, "split": split, "task": "tactic_rhetorical", "predicted_label_cardinality": sum(map(len, sets))/len(sets) if sets else None, "all_zero_ratio": all_zero, "all_one_ratio": all_one, "label_order": labels})
    for task, fn in LOGITS.items():
        vectors = [_vector(fn(r)) for r in rows]
        flat = [x for v in vectors for x in v]
        nan = sum(math.isnan(x) for x in flat); inf = sum(math.isinf(x) for x in flat)
        finite = [x for x in flat if math.isfinite(x)]
        stats.append({"experiment_id": experiment, "split": split, "task": task, "vector_count": sum(bool(v) for v in vectors), "min": min(finite) if finite else None, "max": max(finite) if finite else None, "mean": sum(finite)/len(finite) if finite else None, "std": _std(finite), "nan_count": nan, "inf_count": inf, "zero_variance": len(set(finite)) <= 1 if finite else None})
        if nan or inf: checks.append({"experiment_id": experiment, "split": split, "task": task, "status": "blocker", "reason": "nan_or_inf_logits"})
        elif not any(vectors): checks.append({"experiment_id": experiment, "split": split, "task": task, "status": "blocker", "reason": "trainable_head_logits_absent"})
        elif finite and len(set(finite)) == 1: checks.append({"experiment_id": experiment, "split": split, "task": task, "status": "warning", "reason": "zero_variance_logits"})
    return checks, distributions, stats


def auroc_polarity(rows: list[dict[str, Any]], experiment: str, split: str) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("gold_label") is not None]
    gold = [int(r["gold_label"]) for r in labeled]
    declared = [float(r.get("prob_harmful")) for r in labeled]
    ambiguous = [float(r.get("harmfulness_score")) for r in labeled]
    misuse = any(abs(a-d) > 1e-8 for a,d in zip(ambiguous, declared))
    return {"experiment_id": experiment, "split": split, "declared_positive_class": "harmful", "harmfulness_score_semantics": "probability_of_harmful", "auroc_declared_harmful_score": _roc_auc(gold, declared), "auroc_one_minus_declared": _roc_auc(gold, [1-x for x in declared]), "legacy_harmfulness_score_auroc": _roc_auc(gold, ambiguous), "legacy_field_is_predicted_class_confidence": misuse, "status": "pass" if not misuse else "warning_historical_legacy_field_invalid"}


def agreement_checks(files: dict[tuple[str,str], list[dict[str,Any]]], reference: str) -> tuple[list[dict[str,Any]], list[dict[str,Any]]]:
    output, examples = [], []
    for (experiment, split), rows in files.items():
        if experiment == reference: continue
        a, b = index_predictions(files[(reference, split)]), index_predictions(rows)
        shared = sorted(set(a)&set(b)); score_diffs=[]; any_diff=0
        agreements = Counter()
        for sid in shared:
            x,y=a[sid],b[sid]; score_diffs.append(abs(float(x["prob_harmful"])-float(y["prob_harmful"])))
            fields = {"harmfulness": x.get("pred_label")==y.get("pred_label"), "target_presence": (x.get("target")or{}).get("presence")== (y.get("target")or{}).get("presence"), "target_granularity": (x.get("target")or{}).get("granularity")== (y.get("target")or{}).get("granularity"), "intent_primary": (x.get("intent")or{}).get("primary")== (y.get("intent")or{}).get("primary"), "multimodal_relation": (x.get("tactic")or{}).get("multimodal_relation")== (y.get("tactic")or{}).get("multimodal_relation"), "rhetorical": set((x.get("tactic")or{}).get("rhetorical")or[])==set((y.get("tactic")or{}).get("rhetorical")or[])}
            agreements.update({k:int(v) for k,v in fields.items()})
            if not all(fields.values()):
                any_diff += 1
                if len(examples)<100: examples.append({"experiment_id":experiment,"split":split,"sample_id":sid,"differing_tasks":" ".join(k for k,v in fields.items() if not v)})
        output.append({"reference_experiment":reference,"experiment_id":experiment,"split":split,"sample_count":len(shared),"missing_ids":len(set(a)-set(b)),"unexpected_ids":len(set(b)-set(a)),"harmfulness_label_agreement_ratio":agreements["harmfulness"]/len(shared) if shared else None,"harmfulness_score_exact_equality_count":sum(d==0 for d in score_diffs),"harmfulness_score_mean_absolute_difference":sum(score_diffs)/len(score_diffs) if score_diffs else None,"harmfulness_score_maximum_absolute_difference":max(score_diffs,default=None),"target_presence_agreement":agreements["target_presence"]/len(shared) if shared else None,"target_granularity_agreement":agreements["target_granularity"]/len(shared) if shared else None,"intent_primary_agreement":agreements["intent_primary"]/len(shared) if shared else None,"multimodal_relation_agreement":agreements["multimodal_relation"]/len(shared) if shared else None,"rhetorical_tactic_exact_set_agreement":agreements["rhetorical"]/len(shared) if shared else None,"samples_differing_any_formal_task":any_diff})
    return output, examples


def _ablation_contract(name: str, run: Path) -> dict[str,Any]:
    manifest=json.loads((run/"run_manifest.json").read_text()); logs=json.loads((run/"training_log.json").read_text()); last=logs[-1] if logs else {}; runtime=manifest.get("ablation_runtime") or {}; expected=4 if name=="ablation_w_o_structured_auxiliary" else 6
    checks=[last.get("active_logits_loss_count")==expected]
    if name=="ablation_w_o_retrieval": checks.append(runtime.get("disable_retrieval") is True)
    if name=="ablation_w_o_support_verifier": checks.append(runtime.get("disable_support_verifier") is True and not runtime.get("disable_retrieval"))
    if name=="ablation_w_o_task_aware_gate": checks.append(runtime.get("disable_task_aware_gate") is True)
    if name=="ablation_w_o_structured_auxiliary": checks.append(runtime.get("disable_structured_auxiliary") is True and set(last.get("active_logits_losses",[]))=={"harmfulness","target_granularity","intent_primary","tactic_rhetorical"})
    resolved=sha256_file(run/"resolved_config.yaml"); contract=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(); checkpoint=sha256_file(run/"best_model.pt")
    return {"experiment_id":name,"status":"pass" if all(checks) else "blocker","active_loss_names":last.get("active_logits_losses"),"active_logits_loss_count":last.get("active_logits_loss_count"),"mean_requires_grad":{k:v.get("mean_requires_grad") for k,v in (last.get("loss_provenance")or{}).items()},"ablation_runtime_config":runtime,"checkpoint_sha256":checkpoint,"resolved_config_sha256":resolved,"ablation_contract_sha256":contract}


def _evidence_semantics(name: str, rows: list[dict[str,Any]]) -> dict[str,Any]:
    external=[item for r in rows for item in ((r.get("supporting_evidence")or{}).get("external")or[]) if item.get("is_external_knowledge")]
    retrieved=[x for x in external if x.get("is_retrieved")]; verified=[x for x in retrieved if x.get("verification_status")=="accepted"]
    disabled=name=="ablation_w_o_retrieval"
    return {"experiment_id":name,"legacy_metric_semantics":"combined_internal_and_external_weak_text_overlap_proxy","internal_evidence_metric":"applicable","external_retrieval_metric":"not_applicable" if disabled else "not_separately_measured_historically","retrieved_candidate_count":len(retrieved),"verified_external_count":len(verified),"external_evidence_count":len(external),"status":"pass" if not disabled or not external else "blocker","reason":"retrieval_disabled; external metrics are not_applicable" if disabled else "legacy metric pools supporting_evidence groups before weak lexical matching"}


def _metric_rows(experiment: str,dataset: str,domain: str,metrics: dict[str,Any])->list[dict[str,Any]]:
    aliases={"harmfulness_accuracy":("harmfulness","accuracy"),"harmfulness_macro_f1":("harmfulness","macro_f1"),"harmfulness_weighted_f1":("harmfulness","weighted_f1"),"harmfulness_precision":("harmfulness","harmful_class_precision"),"harmfulness_recall":("harmfulness","harmful_class_recall"),"harmfulness_roc_auc":("harmfulness","roc_auc"),"target_presence_macro_f1":("target_presence","macro_f1"),"target_granularity_macro_f1":("target_granularity","macro_f1"),"intent_primary_macro_f1":("intent_primary","macro_f1"),"tactic_multimodal_relation_macro_f1":("tactic_multimodal_relation","macro_f1"),"tactic_rhetorical_macro_f1_logits_only":("tactic_rhetorical","macro_f1_logits_only")}
    return [{"experiment_id":experiment,"dataset_family":"harmeme","dataset":dataset,"domain":domain,"domain_role":"source_validation" if domain=="pooled" else "source_validation_domain","task":task,"metric":metric,"value":metrics.get(key),"valid_n":metrics.get(f"{task}_valid_n"),"total_n":metrics.get(f"{task}_total_n"),"coverage":metrics.get(f"{task}_coverage"),"unknown_count":metrics.get(f"{task}_unknown_count"),"ambiguous_count":metrics.get(f"{task}_ambiguous_count"),"masked_count":metrics.get(f"{task}_masked_count"),"class_distribution":metrics.get(f"{task}_class_distribution")} for key,(task,metric) in aliases.items()]


def _runtime_row(name:str,run:Path)->dict[str,Any]:
    data=json.loads((run/"runtime.json").read_text()); env=json.loads((run/"environment.json").read_text())
    return {"experiment_id":name,"runtime_seconds_total":data.get("wall_seconds"),"training_seconds":data.get("training_seconds"),"validation_seconds":data.get("validation_seconds"),"test_inference_seconds":data.get("fhm_inference_seconds"),"retrieval_seconds":None,"verification_seconds":None,"context_generation_seconds":None,"serialization_seconds":None,"gpu_model":env.get("gpu_model"),"gpu_count":None,"peak_gpu_memory_mb":data.get("peak_gpu_memory_mb"),"gpu_hours":None,"measurement_status":"not_recorded"}


def _vector(value:Any)->list[float]:
    if isinstance(value,dict): value=value.get("values") or value.get("preview") or []
    return [float(x) for x in value] if isinstance(value,(list,tuple)) else []


def _std(values:list[float])->float|None:
    if not values:return None
    mean=sum(values)/len(values);return (sum((x-mean)**2 for x in values)/len(values))**.5


def _entropy(counts:Counter)->float|None:
    n=sum(counts.values()); k=len(counts)
    if not n:return None
    if k<=1:return 0.0
    return -sum((c/n)*math.log(c/n) for c in counts.values())/math.log(k)


def _duplicate_groups(rows:list[dict[str,Any]],key:str)->list[list[dict[str,Any]]]:
    groups={}
    for row in rows:groups.setdefault(row[key],[]).append(row)
    return [v for v in groups.values() if len(v)>1]


def _decision(blockers:list[str])->str:
    unique=set(blockers)
    if not unique:return "READY_FOR_SOURCE_SANITY"
    mapping={"prediction_reuse":"BLOCKED_PREDICTION_REUSE","ablation_contract":"BLOCKED_ABLATION_CONTRACT","label_mapping_or_prediction_identity":"BLOCKED_LABEL_MAPPING"}
    return mapping.get(next(iter(unique)),"BLOCKED_MULTIPLE") if len(unique)==1 else "BLOCKED_MULTIPLE"


def _csv(path:Path,rows:list[dict[str,Any]])->None:
    columns=sorted({k for r in rows for k in r}); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        if not columns:return
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _markdown(report:dict[str,Any])->str:
    return "# Smoke artifact diagnostics\n\n"+f"Suite: `{report['suite']}`  \nReference: `{report['reference_experiment']}`  \nFHM use: schema, polarity, consistency and provenance only.\n\n"+f"Suspicious artifact reuse: **{report['suspicious_artifact_reuse']}**  \nReadiness: **{report['readiness']['decision']}**\n"


__all__=["auroc_polarity","canonical_prediction_hash","collapse_checks","diagnose_smoke","index_predictions"]
