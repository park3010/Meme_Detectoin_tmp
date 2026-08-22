"""Source-only integrity, manifest, gradient, and tiny-overfit diagnostics.

This module has a hard dataset allowlist and must never construct or inspect
FHM/Facebook or Memotion data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from dataset.labels import LabelVocab, NormalizedLabelAdapter, NormalizedLabelStore
from dataset.meme_dataset import MemeDataset
from experiments.research_protocol import sha256_file
from utils.io import write_json

SOURCE_DATASETS = frozenset({"harm_c", "harm_p"})
FORBIDDEN_MARKERS = ("facebook", "fhm", "memotion")
FORMAL_FIELDS = ("harmfulness", "target_presence", "target_granularity", "intent_primary", "tactic_rhetorical", "tactic_multimodal_relation")


def assert_source_only_dataset_names(dataset_names: list[str] | tuple[str, ...] | set[str]) -> None:
    values = {str(value).lower() for value in dataset_names}
    if not values or not values <= SOURCE_DATASETS:
        raise ValueError(f"source-sanity permits only {sorted(SOURCE_DATASETS)}; received {sorted(values)}")


def reject_forbidden_manifest(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        marker = " ".join(str(row.get(key, "")).lower() for key in ("dataset_name", "dataset_family", "domain", "sample_id", "sample_key"))
        if any(forbidden in marker for forbidden in FORBIDDEN_MARKERS):
            raise ValueError(f"forbidden non-source sample in manifest: {row}")


def run_source_sanity(*, profile: str, output_root: str = "result/source_sanity", config: str = "configs/config.yaml", seed: int = 42, device: str = "cpu", strict: bool = False, force: bool = False, disable_tqdm: bool = False) -> dict[str, Any]:
    if profile not in {"data_integrity", "gradient_check", "overfit_32", "overfit_128", "shuffled_label", "domain_probe", "all"}:
        raise ValueError(f"unknown source-sanity profile: {profile}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if profile == "all":
        results = {}
        for name in ("data_integrity", "gradient_check", "overfit_32", "overfit_128", "shuffled_label", "domain_probe"):
            results[name] = run_source_sanity(profile=name, output_root=output_root, config=config, seed=seed, device=device, strict=strict, force=force)
            if not results[name].get("passed", False): break
        return {"profile": "all", "passed": all(r.get("passed") for r in results.values()), "profiles": results}
    if profile == "data_integrity":
        result = audit_source_data(root, seed=seed, force=force)
    elif profile == "gradient_check":
        result = gradient_check(root, config=config, seed=seed, device=device, force=force)
    elif profile == "overfit_32":
        from experiments.source_tiny_overfit import run_overfit_32
        result = run_overfit_32(output_root=str(root), config=config, seed=seed, device=device, force=force, disable_tqdm=disable_tqdm)
    else:
        result = _not_run_profile(root, profile, reason="profile implementation requires a passing CUDA gradient check and explicit sequential launch")
    # The v2 gradient repair preserves the original source-sanity readiness
    # artifact; certification is written under gradient_forensics instead.
    if profile == "overfit_32":
        write_json(root/"readiness_decision.json",result)
    elif profile != "gradient_check" or not (root/"gradient_forensics/post_repair_gradient_report.json").is_file():
        _write_readiness(root)
    if strict and not result.get("passed", False):
        raise RuntimeError(f"source-sanity {profile} blocked: {result.get('blockers') or result.get('reason')}")
    return result


def audit_source_data(root: Path, *, seed: int = 42, force: bool = False) -> dict[str, Any]:
    out = root / "data_integrity"
    _prepare_output(out, force)
    split_path = Path("result/splits/harmeme/source_split_seed_42_v2.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_rows = {part: list(split.get(part, [])) for part in ("train", "validation")}
    reject_forbidden_manifest(split_rows["train"] + split_rows["validation"])
    membership: dict[str, str] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    overlap = []
    for part, rows in split_rows.items():
        for row in rows:
            key = str(row.get("sample_key") or f"{row.get('dataset_name')}::{row.get('sample_id')}")
            if key in membership and membership[key] != part: overlap.append(key)
            membership[key] = part
            split_metadata[key] = row
    all_rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    duplicate_ids: list[dict[str, Any]] = []
    duplicate_images: list[dict[str, Any]] = []
    label_conflicts: list[dict[str, Any]] = []
    summaries = {}
    vocab = LabelVocab.from_yaml("configs/label_vocab.yaml")
    adapter = NormalizedLabelAdapter(vocab=vocab)
    for dataset_name in sorted(SOURCE_DATASETS):
        dataset = MemeDataset(dataset_names=[dataset_name], keep_missing_images=True)
        store = NormalizedLabelStore(dataset_names=[dataset_name], label_set="full")
        counts = Counter(str(sample.sample_id) for sample in dataset.samples)
        for sample_id, count in counts.items():
            if count > 1: duplicate_ids.append({"dataset_name": dataset_name, "sample_id": sample_id, "count": count})
        image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        records = []
        for sample in dataset.samples:
            sid = str(sample.sample_id); key = f"{dataset_name}::{sid}"
            # v2 integrity is evaluated over the effective supervised pool;
            # excluded raw records remain immutable and auditable via the
            # duplicate-resolution manifest.
            if key not in membership:
                continue
            normalized = store.get(dataset_name, sid)
            image = Path(sample.image_path) if sample.image_path else None
            source_annotation_exists = sample.annotation is not None
            normalized_exists = normalized is not None
            image_exists = bool(image and image.exists())
            decode_ok = False; image_sha = None; decode_error = None
            if image_exists:
                try:
                    with Image.open(image) as handle:
                        handle.verify()
                    image_sha = sha256_file(image)
                    decode_ok = True
                except Exception as exc:
                    decode_error = f"{type(exc).__name__}: {exc}"
                    decode_failures.append({"dataset_name": dataset_name, "sample_id": sid, "image_path": str(image), "error": decode_error})
            labels = normalized.labels if normalized else {}
            encoded = adapter.encode_row(normalized) if normalized else None
            mask_consistent = _masks_consistent(vocab, labels, encoded)
            ocr = str(sample.ocr_text_full or "")
            normalized_ocr = str(normalized.ocr_text_full or "") if normalized else None
            ocr_aligned = normalized is not None and ocr == normalized_ocr
            raw_label = _harmfulness(sample.raw_label)
            normalized_label = str(labels.get("harmfulness")) if labels else None
            label_consistent = normalized_label == raw_label
            row = {"dataset_name":dataset_name,"sample_id":sid,"sample_key":key,"split":membership.get(key),"source_annotation_exists":source_annotation_exists,"normalized_annotation_exists":normalized_exists,"image_path":str(image) if image else None,"image_exists":image_exists,"image_decode_ok":decode_ok,"image_sha256":image_sha,"ocr_exists":ocr is not None,"ocr_aligned":ocr_aligned,"ocr_empty":ocr == "","ocr_whitespace_only":bool(ocr) and not ocr.strip(),"ocr_length":len(ocr),"raw_harmfulness":raw_label,"normalized_harmfulness":normalized_label,"label_consistent":label_consistent,"structured_label_eligible":bool(split_metadata.get(key,{}).get("structured_label_eligible",False)),"structured_masks_consistent":mask_consistent,**{f"valid_{field}":bool(encoded and encoded["masks"].get(field,0)) for field in FORMAL_FIELDS}}
            records.append(row)
            if image_sha: image_groups[image_sha].append(row)
            reasons=[]
            if not source_annotation_exists: reasons.append("missing_source_annotation")
            if not normalized_exists: reasons.append("missing_normalized_annotation")
            if not image_exists: reasons.append("missing_image")
            if membership.get(key) not in {"train","validation"}: reasons.append("unknown_sample_origin")
            if not ocr_aligned: reasons.append("ocr_sample_mismatch")
            if not mask_consistent: reasons.append("mask_inconsistent")
            if not label_consistent: reasons.append("label_conflict")
            if reasons: missing.append({"dataset_name":dataset_name,"sample_id":sid,"reasons":" ".join(reasons)})
            if not label_consistent: label_conflicts.append({"dataset_name":dataset_name,"sample_id":sid,"raw_harmfulness":raw_label,"normalized_harmfulness":normalized_label})
        for digest, group in image_groups.items():
            if len(group)>1:
                duplicate_images.append({"dataset_name":dataset_name,"image_sha256":digest,"count":len(group),"sample_ids":" ".join(r["sample_id"] for r in group),"labels":" ".join(sorted({str(r["normalized_harmfulness"]) for r in group})),"conflicting_labels":len({r["normalized_harmfulness"] for r in group})>1})
        all_rows[dataset_name]=records
        summaries[dataset_name]=_domain_summary(records)
    blockers=[]
    if overlap: blockers.append("BLOCKED_SPLIT_OVERLAP")
    if any(not r["image_exists"] for rows in all_rows.values() for r in rows): blockers.append("BLOCKED_MISSING_IMAGE")
    if decode_failures: blockers.append("BLOCKED_IMAGE_DECODE")
    if label_conflicts: blockers.append("BLOCKED_LABEL_CONFLICT")
    if any(bool(row["conflicting_labels"]) for row in duplicate_images): blockers.append("BLOCKED_LABEL_CONFLICT")
    if any("unknown_sample_origin" in r["reasons"] for r in missing): blockers.append("BLOCKED_UNKNOWN_SAMPLE_ORIGIN")
    if any(any(token in r["reasons"] for token in ("missing_source_annotation","missing_normalized_annotation","ocr_sample_mismatch","mask_inconsistent")) for r in missing): blockers.append("BLOCKED_SAMPLE_PAIRING")
    for name, rows in all_rows.items(): _csv(out / f"{name}_samples.csv", rows)
    _csv(out / "missing_samples.csv", missing); _csv(out / "decode_failures.csv", decode_failures); _csv(out / "duplicate_ids.csv", duplicate_ids); _csv(out / "duplicate_images.csv", duplicate_images); _csv(out / "label_conflicts.csv", label_conflicts)
    _csv(out / "ocr_statistics.csv", [{"dataset_name":name,**{k:v for k,v in summary.items() if k.startswith("ocr_")}} for name,summary in summaries.items()])
    _csv(out / "domain_comparison.csv", [{"dataset_name":name,**summary} for name,summary in summaries.items()])
    manifests = create_sanity_manifests(root, split_rows, all_rows, seed=seed, force=force)
    report={"schema_version":"source_data_integrity_v1","source_only":True,"source_split_manifest":str(split_path),"source_split_sha256":sha256_file(split_path),"domains":summaries,"split_overlap_count":len(overlap),"duplicate_id_count":len(duplicate_ids),"duplicate_image_group_count":len(duplicate_images),"duplicate_image_conflicting_label_group_count":sum(bool(r["conflicting_labels"]) for r in duplicate_images),"missing_sample_issue_count":len(missing),"decode_failure_count":len(decode_failures),"label_conflict_count":len(label_conflicts),"blockers":sorted(set(blockers)),"passed":not blockers,"manifests":manifests,"generated_at_utc":datetime.now(timezone.utc).isoformat()}
    write_json(out/"data_integrity.json",report)
    (out/"data_integrity.md").write_text(_integrity_markdown(report),encoding="utf-8")
    return report


def create_sanity_manifests(root: Path, split_rows: dict[str,list[dict[str,Any]]], audited: dict[str,list[dict[str,Any]]], *, seed: int, force: bool) -> dict[str,Any]:
    out=root/"manifests"; out.mkdir(parents=True,exist_ok=True)
    audit_index={r["sample_key"]:r for rows in audited.values() for r in rows}
    train=[dict(r) for r in split_rows["train"]]; validation=[dict(r) for r in split_rows["validation"]]
    reject_forbidden_manifest(train+validation)
    result={}
    for size in (32,128):
        rows=_stratified_select(train,size,seed,audit_index)
        result[f"overfit_{size}"]=_write_manifest(out/f"overfit_{size}.json",rows,"source_train",seed,{"size":size,"strata":["dataset_name","harmfulness"],"maximize_structured_validity":True},force)
    probe=[]
    for domain in ("harm_c","harm_p"):
        probe.extend(_stratified_select([r for r in validation if r.get("dataset_name")==domain],100,seed,audit_index))
    result["domain_probe_200"]=_write_manifest(out/"domain_probe_200.json",probe,"source_validation",seed,{"per_domain":100,"stratify":"harmfulness","maximize_structured_validity":True},force)
    base=_stratified_select(train,128,seed,audit_index); labels=[r["harmfulness"] for r in base]; shuffled=list(labels); random.Random(seed).shuffle(shuffled)
    shuffled_rows=[{**r,"original_harmfulness":r["harmfulness"],"shuffled_harmfulness":label} for r,label in zip(base,shuffled)]
    result["shuffled_label_manifest"]=_write_manifest(out/"shuffled_label_manifest.json",shuffled_rows,"ephemeral_source_train_view",seed,{"permutation_hash":hashlib.sha256(json.dumps(shuffled,separators=(",",":")).encode()).hexdigest(),"annotations_modified":False},force)
    return result


def gradient_check(root: Path, *, config: str, seed: int, device: str, force: bool) -> dict[str,Any]:
    del seed
    from experiments.gradient_forensics import _execute
    from experiments.source_sanity_gate import atomic_write_json, source_sanity_code_sha256, write_gradient_gate
    if str(device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            return {"profile":"gradient_check","passed":False,"decision":"BLOCKED_GRADIENT_FLOW","reason":"CUDA requested but unavailable","source_only":True}
    # Every forced canonical check executes the current graph. Versioned run
    # evidence avoids rewriting the immutable original failed report.
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+source_sanity_code_sha256()[:12]
    run_dir=root/"gradient_gate_runs"/run_id
    run_dir.mkdir(parents=True,exist_ok=False)
    report=_execute(run_dir,config=config,device=device)
    report={**report,"profile":"gradient_check","source_only":True,"fhm_or_memotion_accessed":False,"scientific_checkpoint_written":False}
    report_path=run_dir/"gradient_report.json";atomic_write_json(report_path,report)
    if report.get("passed"):
        gate=write_gradient_gate(output_root=root,config_path=config,gradient_report_path=report_path,report=report,device_requested=device)
        report["gradient_gate_path"]=str(root/"gates/gradient_check_gate.json")
        report["gradient_gate_written"]=True
        report["gradient_gate_code_sha256"]=gate["code_sha256"]
    else:
        report["gradient_gate_written"]=False
    return report


def deterministic_manifest_hash(rows: list[dict[str,Any]]) -> str:
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def deterministic_shuffle(values: list[Any], seed: int=42) -> tuple[list[Any],str]:
    output=list(values); random.Random(seed).shuffle(output)
    return output,hashlib.sha256(json.dumps(output,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def parameter_update(before: Any, after: Any) -> dict[str,float|bool]:
    import torch
    delta=(after.detach()-before.detach()).float(); norm=float(delta.norm())
    base=float(before.detach().float().norm())
    return {"parameter_norm_before":base,"parameter_norm_after":float(after.detach().float().norm()),"update_norm":norm,"update_to_parameter_ratio":norm/base if base else math.inf if norm else 0.0,"updated":norm>0,"finite":bool(torch.isfinite(delta).all())}


def _stratified_select(rows:list[dict[str,Any]],size:int,seed:int,audit_index:dict[str,dict[str,Any]])->list[dict[str,Any]]:
    groups=defaultdict(list)
    for row in rows:
        key=str(row.get("sample_key") or f"{row.get('dataset_name')}::{row.get('sample_id')}")
        audit=audit_index.get(key,{})
        enriched={**row,"audit_structured_label_eligible":bool(audit.get("structured_label_eligible")),**{f"valid_{field}":bool(audit.get(f"valid_{field}")) for field in FORMAL_FIELDS}}
        groups[(row.get("dataset_name"),row.get("harmfulness"))].append(enriched)
    rng=random.Random(seed)
    for group in groups.values(): group.sort(key=lambda r:(not r["audit_structured_label_eligible"],hashlib.sha256(f"{seed}:{r.get('sample_key')}".encode()).hexdigest()))
    selected=[]; keys=sorted(groups); target=size//len(keys)
    for key in keys: selected.extend(groups[key][:target])
    leftovers=[r for key in keys for r in groups[key][target:] if r not in selected]
    leftovers.sort(key=lambda r:hashlib.sha256(f"extra:{seed}:{r.get('sample_key')}".encode()).hexdigest())
    selected.extend(leftovers[:size-len(selected)])
    selected.sort(key=lambda r:str(r.get("sample_key")))
    return selected


def _write_manifest(path:Path,rows:list[dict[str,Any]],role:str,seed:int,rules:dict[str,Any],force:bool)->dict[str,Any]:
    reject_forbidden_manifest(rows); digest=deterministic_manifest_hash(rows)
    payload={"schema_version":"source_sanity_manifest_v1","source_only":True,"role":role,"seed":seed,"selection_rules":rules,"sample_count":len(rows),"sample_rows_sha256":digest,"domain_distribution":dict(Counter(r.get("dataset_name") for r in rows)),"harmfulness_distribution":dict(Counter(r.get("harmfulness") for r in rows)),"structured_valid_n":{field:sum(bool(r.get(f"valid_{field}")) for r in rows) for field in FORMAL_FIELDS},"samples":rows}
    if path.exists() and not force:
        existing=json.loads(path.read_text());
        if existing.get("sample_rows_sha256")!=digest: raise FileExistsError(f"immutable manifest differs: {path}")
        return {"path":str(path),"sha256":sha256_file(path),"sample_rows_sha256":digest,"sample_count":len(rows)}
    write_json(path,payload); return {"path":str(path),"sha256":sha256_file(path),"sample_rows_sha256":digest,"sample_count":len(rows)}


def _domain_summary(rows:list[dict[str,Any]])->dict[str,Any]:
    lengths=[int(r["ocr_length"]) for r in rows]
    return {"total_sample_count":len(rows),"train_count":sum(r["split"]=="train" for r in rows),"validation_count":sum(r["split"]=="validation" for r in rows),"harmful_count":sum(r["normalized_harmfulness"]=="harmful" for r in rows),"non_harmful_count":sum(r["normalized_harmfulness"]=="non_harmful" for r in rows),"structured_label_eligible_count":sum(r["structured_label_eligible"] for r in rows),"empty_ocr_count":sum(r["ocr_empty"] for r in rows),"empty_ocr_ratio":sum(r["ocr_empty"] for r in rows)/len(rows),"whitespace_only_ocr_count":sum(r["ocr_whitespace_only"] for r in rows),"ocr_length_min":min(lengths),"ocr_length_mean":mean(lengths),"ocr_length_median":median(lengths),"ocr_length_p95":sorted(lengths)[min(len(lengths)-1,int(.95*len(lengths)))],"ocr_length_max":max(lengths),"image_decode_failure_count":sum(not r["image_decode_ok"] for r in rows),"missing_annotation_count":sum(not r["source_annotation_exists"] or not r["normalized_annotation_exists"] for r in rows),"missing_image_count":sum(not r["image_exists"] for r in rows),"label_conflict_count":sum(not r["label_consistent"] for r in rows)}


def _masks_consistent(vocab:LabelVocab,labels:dict[str,Any],encoded:dict[str,Any]|None)->bool:
    if not encoded:return False
    for field in vocab.single_label_fields:
        if encoded["masks"][field]!=vocab.mask_for_single(field,str(labels.get(field,"unknown"))):return False
    for field in vocab.multi_label_fields:
        values=labels.get(field,[]) if isinstance(labels.get(field,[]),list) else [labels.get(field)]
        if encoded["masks"][field]!=vocab.mask_for_multi(field,[str(v) for v in values]):return False
    return True


def _harmfulness(value:Any)->str|None:
    if value in (1,"1","harmful",True):return "harmful"
    if value in (0,"0","non_harmful",False):return "non_harmful"
    return None


def _prepare_output(path:Path,force:bool)->None:
    if path.exists() and any(path.iterdir()) and not force: raise FileExistsError(f"output exists; use --force: {path}")
    path.mkdir(parents=True,exist_ok=True)


def _not_run_profile(root:Path,profile:str,reason:str)->dict[str,Any]:
    out=root/profile; out.mkdir(parents=True,exist_ok=True); result={"profile":profile,"passed":False,"status":"not_run","reason":reason,"source_only":True}; write_json(out/"status.json",result); return result


def _write_readiness(root:Path)->None:
    integrity=_read(root/"data_integrity"/"data_integrity.json"); gradient=_read(root/"gradient_check"/"gradient_report.json")
    blockers=[]; pending=[]
    if integrity and not integrity.get("passed"):blockers.append("BLOCKED_DATA_INTEGRITY")
    if gradient and not gradient.get("passed"):blockers.append("BLOCKED_GRADIENT_FLOW")
    elif not gradient:pending.append("gradient_check")
    if not (root/"overfit_32"/"audit_report.json").exists():pending.append("overfit_32")
    if not (root/"domain_probe"/"metrics.json").exists():pending.append("domain_probe")
    decision=blockers[0] if len(blockers)==1 else "BLOCKED_MULTIPLE" if blockers else "READY_FOR_1SEED"
    payload={"decision":decision,"blockers":blockers,"pending_profiles":pending,"ready_for_1seed":decision=="READY_FOR_1SEED" and not pending,"tokenizer_policy_status":"frozen_with_documented_warning","fhm_or_memotion_accessed":False,"generated_at_utc":datetime.now(timezone.utc).isoformat()}
    write_json(root/"readiness_decision.json",payload); (root/"readiness_decision.md").write_text(f"# Source-sanity readiness\n\nDecision: **{decision}**\n\nBlockers: {', '.join(blockers) or 'none'}\n",encoding="utf-8")


def _read(path:Path)->dict[str,Any]:
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def _csv(path:Path,rows:list[dict[str,Any]])->None:
    columns=sorted({k for r in rows for k in r}); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        if not columns:return
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def _integrity_markdown(report:dict[str,Any])->str:
    lines=["# HarMeme source data integrity","","No Facebook/FHM or Memotion dataset was instantiated.",""]
    for name,summary in report["domains"].items():lines.extend([f"## {name}","",f"Samples: {summary['total_sample_count']}; missing images: {summary['missing_image_count']}; decode failures: {summary['image_decode_failure_count']}; label conflicts: {summary['label_conflict_count']}.",""])
    lines.append(f"Result: **{'PASS' if report['passed'] else 'BLOCKED'}**")
    return "\n".join(lines)+"\n"


__all__=["SOURCE_DATASETS","assert_source_only_dataset_names","audit_source_data","create_sanity_manifests","deterministic_manifest_hash","deterministic_shuffle","parameter_update","reject_forbidden_manifest","run_source_sanity"]
