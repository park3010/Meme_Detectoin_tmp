"""Versioned, source-only HarMeme duplicate resolution and split-v2 migration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.research_protocol import content_sha256, sha256_file, validate_source_manifest
from utils.io import read_jsonl

FORMAL_FIELDS = ("harmfulness", "target_presence", "target_granularity", "intent_primary", "tactic_rhetorical", "tactic_multimodal_relation")
MULTILABEL_FIELDS = {"tactic_rhetorical"}
FORBIDDEN_PREFIXES = ("facebook::", "fhm::", "memotion::")
V1_SPLIT = Path("result/splits/harmeme/source_split_seed_42.json")
V2_SPLIT = Path("result/splits/harmeme/source_split_seed_42_v2.json")
RESOLUTION = Path("result/splits/harmeme/duplicate_resolution_manifest_v1.json")


class ProtocolV2Error(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}"); self.code = code


def canonical_label(value: Any, *, multilabel: bool = False) -> Any:
    if value is None: return None
    if multilabel:
        return tuple(sorted({str(item) for item in (value if isinstance(value, list) else [value]) if str(item) not in {"", "unknown", "ambiguous"}}))
    text = str(value)
    return None if text in {"", "unknown", "ambiguous", "not_applicable"} else text


def structured_conflicts(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Return disagreements only where at least two members have valid values."""
    conflicts: dict[str, list[Any]] = {}
    for field in FORMAL_FIELDS:
        values = [canonical_label((row.get("labels") or {}).get(field), multilabel=field in MULTILABEL_FIELDS) for row in rows]
        valid = [value for value in values if value is not None]
        if len(valid) >= 2 and len(set(valid)) > 1:
            conflicts[field] = [list(v) if isinstance(v, tuple) else v for v in values]
    return conflicts


def build_protocol_v2(*, resolution_root: str | Path, output_root: str | Path = "result", strict: bool = True) -> dict[str, Any]:
    root = Path(resolution_root)
    required = ["adjudication_summary.json", "duplicate_groups.csv", "conflict_groups.csv", "raw_label_trace.csv", "ocr_pairing_trace.csv", "proposed_resolution_plan.json", "split_v2_impact_preview.json", "retrieval_impact.csv", "readiness_decision.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing: raise ProtocolV2Error("duplicate_resolution_manifest_missing", f"Missing approved inputs: {missing}")
    split_path = Path(output_root) / "splits/harmeme/source_split_seed_42.json"
    out = split_path.parent
    v2_path = out / "source_split_seed_42_v2.json"
    resolution_path = out / "duplicate_resolution_manifest_v1.json"
    v1_bytes = split_path.read_bytes(); v1_sha = hashlib.sha256(v1_bytes).hexdigest()
    v1 = json.loads(v1_bytes); all_rows = {str(r["sample_key"]): dict(r) for p in ("train", "validation") for r in v1[p]}
    old_partition = {str(r["sample_key"]): p for p in ("train", "validation") for r in v1[p]}
    groups = list(csv.DictReader((root / "duplicate_groups.csv").open(encoding="utf-8")))
    normalized = _normalized_rows()
    resolution_groups=[]; excluded: dict[str, dict[str, Any]]={}; canonical_map={}; image_by_key={}; mm_by_key={}
    ocr = {r["sample_key"]: r for r in csv.DictReader((root / "ocr_pairing_trace.csv").open(encoding="utf-8"))}
    for group in groups:
        keys=group["sample_keys"].split(); category=group["primary_category"]
        conflicts = structured_conflicts([normalized[key] for key in keys])
        # Shared-image/different-text rows are genuinely distinct multimodal records;
        # the guard canonicalizes only annotations proposed for deduplication.
        if conflicts and category in {"EXACT_MULTIMODAL_DUPLICATE_SAME_LABEL", "FORMAT_DUPLICATE_ONLY"}:
            category="STRUCTURED_LABEL_CONFLICT"
        canonical=min(keys)
        if category in {"RAW_SOURCE_LABEL_CONFLICT", "STRUCTURED_LABEL_CONFLICT"}:
            action="EXCLUDE_ALL_CONFLICTING_MEMBERS"; excluded_keys=keys
        elif category in {"EXACT_MULTIMODAL_DUPLICATE_SAME_LABEL", "FORMAT_DUPLICATE_ONLY"}:
            action="RETAIN_ONE_CANONICAL_SAMPLE"; excluded_keys=[k for k in keys if k != canonical]
        elif category == "SHARED_IMAGE_DIFFERENT_TEXT_CONFIRMED":
            action="KEEP_AS_DISTINCT_MULTIMODAL_SAMPLES_GROUP_BY_IMAGE_SPLIT"; excluded_keys=[]
        else: raise ProtocolV2Error("canonical_duplicate_violation", f"Unapproved category {category}")
        for key in excluded_keys: excluded[key]={"sample_key":key,"group_id":group["group_id"],"reason":category,"action":action}
        for key in keys:
            canonical_map[key]=canonical; image_by_key[key]=group["image_group_id"]; mm_by_key[key]=ocr[key]["multimodal_group_id"]
        resolution_groups.append({"group_id":group["group_id"],"image_group_id":group["image_group_id"],"sample_keys":keys,"original_category":group["primary_category"],"final_category":category,"structured_conflicting_fields":conflicts,"action":action,"canonical_sample_key":canonical,"excluded_keys":excluded_keys})
    retained={k:r for k,r in all_rows.items() if k not in excluded}
    assigned={k:old_partition[k] for k in retained}
    for group in resolution_groups:
        keys=[k for k in group["sample_keys"] if k in retained]
        if keys and len({assigned[k] for k in keys}) > 1:
            target=assigned[group["canonical_sample_key"]]
            for key in keys: assigned[key]=target
    train=sorted((r for k,r in retained.items() if assigned[k]=="train"),key=lambda r:r["sample_key"])
    validation=sorted((r for k,r in retained.items() if assigned[k]=="validation"),key=lambda r:r["sample_key"])
    moved=sorted(k for k in retained if assigned[k] != old_partition[k])
    manifest={"schema_version":"harmeme_source_split_v2","protocol":"harmeme_to_fhm_v2","source_family":"HarMeme","split_seed":42,"model_seed_independent":True,"train_ratio":0.8,"validation_ratio":0.2,"stratification_fields":["original_dataset","harmfulness"],"grouping_field":"decoded_rgb_pixel_sha256","resolution_manifest_path":str(resolution_path),"resolution_policy_version":"duplicate_resolution_v1","label_set":v1.get("label_set","clean"),"harmfulness_supervision":v1.get("harmfulness_supervision"),"structured_supervision":v1.get("structured_supervision"),"train":train,"validation":validation}
    manifest["statistics"]=_statistics(train,validation,normalized); manifest["content_sha256"]=content_sha256(manifest)
    retained_ids=sorted(retained); excluded_ids=sorted(excluded)
    resolution={"schema_version":"duplicate_resolution_manifest_v1","protocol":"harmeme_to_fhm_v2","source_only":True,"approved_evidence_root":str(root),"approved_evidence_sha256":{n:sha256_file(root/n) for n in required},"source_split_v1_path":str(split_path),"source_split_v1_sha256":v1_sha,"raw_total":len(all_rows),"retained_total":len(retained),"excluded_total":len(excluded),"moved_ids":moved,"groups":resolution_groups,"structured_label_conflict_group_count":sum(g["final_category"]=="STRUCTURED_LABEL_CONFLICT" for g in resolution_groups),"retained_sample_ids_sha256":_lines_hash(retained_ids),"excluded_sample_ids_sha256":_lines_hash(excluded_ids)}
    resolution["content_sha256"]=_json_hash(resolution)
    audit=audit_source_split_v2(manifest, resolution=resolution, normalized=normalized)
    if strict and not audit["passed"]: raise ProtocolV2Error(audit["errors"][0]["code"], audit["errors"][0]["message"])
    out.mkdir(parents=True,exist_ok=True)
    _write_new_json(resolution_path,resolution); _write_sha(resolution_path)
    _write_new_json(v2_path,manifest); _write_sha(v2_path)
    _write_new_jsonl(out/"excluded_samples_v2.jsonl",[excluded[k] for k in excluded_ids])
    _write_new_json(out/"canonical_duplicate_map_v2.json",{"schema_version":"canonical_duplicate_map_v2","mapping":canonical_map})
    migration={"schema_version":"protocol_migration_v1_to_v2","status":"built_not_activated","protocol_from":"harmeme_to_fhm_v1","protocol_to":"harmeme_to_fhm_v2","split_v1_sha256":v1_sha,"split_v2_sha256":sha256_file(v2_path),"resolution_manifest_sha256":sha256_file(resolution_path),"counts":manifest["statistics"],"excluded_ids":excluded_ids,"moved_ids":moved,"audit":audit}
    _write_new_json(out/"protocol_migration_v1_to_v2.json",migration)
    if hashlib.sha256(split_path.read_bytes()).hexdigest()!=v1_sha: raise ProtocolV2Error("split_v1_modified","v1 changed during build")
    return {"status":"pass","split_v2_path":str(v2_path),"split_v2_sha256":sha256_file(v2_path),"resolution_manifest_sha256":sha256_file(resolution_path),"statistics":manifest["statistics"],"excluded_ids":excluded_ids,"moved_ids":moved,"audit":audit}


def audit_source_split_v2(manifest_or_path: dict[str,Any]|str|Path, *, resolution:dict[str,Any]|None=None, normalized:dict[str,Any]|None=None) -> dict[str,Any]:
    manifest=json.loads(Path(manifest_or_path).read_text()) if not isinstance(manifest_or_path,dict) else manifest_or_path
    if resolution is None:
        p=Path(str(manifest.get("resolution_manifest_path",RESOLUTION))); resolution=json.loads(p.read_text()) if p.is_file() else None
    errors=[]
    def err(code,msg): errors.append({"code":code,"message":msg})
    if not resolution: err("duplicate_resolution_manifest_missing","Resolution manifest missing"); return {"passed":False,"errors":errors}
    rows={p:list(manifest.get(p,[]) or []) for p in ("train","validation")}; keys={p:{str(r.get("sample_key")) for r in rows[p]} for p in rows}; retained=keys["train"]|keys["validation"]
    excluded={k for g in resolution.get("groups",[]) for k in g.get("excluded_keys",[])}
    if retained & excluded: err("excluded_sample_retained",f"Excluded retained: {sorted(retained&excluded)[:5]}")
    for prefix in FORBIDDEN_PREFIXES:
        if any(k.lower().startswith(prefix) for k in retained): err("forbidden_dataset_retained",prefix)
    image_groups={g["image_group_id"]:{k for k in g["sample_keys"] if k in retained} for g in resolution.get("groups",[])}
    image_overlap=sum(bool(v&keys["train"] and v&keys["validation"]) for v in image_groups.values())
    if image_overlap: err("decoded_image_cross_split_overlap",str(image_overlap))
    mm_by_partition={"train":defaultdict(set),"validation":defaultdict(set)}
    # Exact multimodal overlap is necessarily contained in retained duplicate groups.
    for g in resolution.get("groups",[]):
        if g["original_category"]=="EXACT_MULTIMODAL_DUPLICATE_SAME_LABEL":
            present={p for p in rows if set(g["sample_keys"])&keys[p]}
            if len(present)>1: err("multimodal_cross_split_overlap",g["group_id"])
        if g["final_category"]=="RAW_SOURCE_LABEL_CONFLICT" and set(g["sample_keys"])&retained: err("raw_label_conflict_retained",g["group_id"])
        if g["final_category"]=="STRUCTURED_LABEL_CONFLICT" and set(g["sample_keys"])&retained: err("structured_label_conflict_retained",g["group_id"])
        if g["action"]=="RETAIN_ONE_CANONICAL_SAMPLE" and set(g["sample_keys"])&retained != {g["canonical_sample_key"]}: err("canonical_duplicate_violation",g["group_id"])
    base=validate_source_manifest(manifest)
    for message in base["errors"]: err("source_split_invalid",message)
    return {"passed":not errors,"errors":errors,"train_count":len(rows["train"]),"validation_count":len(rows["validation"]),"retained_count":len(retained),"excluded_count":len(excluded),"decoded_image_cross_split_overlap":image_overlap,"multimodal_cross_split_overlap":sum(e["code"]=="multimodal_cross_split_overlap" for e in errors)}


def _normalized_rows():
    out={}; clean={}
    for ds in ("harm_c","harm_p"):
        for row in read_jsonl(Path("dataset/annotation_normalized")/ds/"normalized_labels.jsonl"): out[f"{ds}::{row['sample_id']}"]=row
        for row in read_jsonl(Path("dataset/annotation_normalized")/ds/"normalized_clean.jsonl"): clean[f"{ds}::{row['sample_id']}"]=row
    for key, row in out.items():
        full_labels=dict(row.get("labels") or {}); clean_labels=dict((clean.get(key) or {}).get("labels") or {})
        row["labels"]={field:(full_labels.get(field) if field=="harmfulness" else clean_labels.get(field)) for field in FORMAL_FIELDS}
    return out


def _statistics(train,validation,normalized):
    all_rows=train+validation
    return {"raw_total":7013,"retained_total":len(all_rows),"excluded_total":7013-len(all_rows),"train_total":len(train),"validation_total":len(validation),"domain_counts":dict(sorted(Counter(r["original_dataset"] for r in all_rows).items())),"harmfulness_counts":dict(sorted(Counter(r["harmfulness"] for r in all_rows).items())),"structured_eligible_total":sum(bool(r.get("structured_label_eligible")) for r in all_rows),"structured_eligible_by_field":{f:sum(canonical_label((normalized[str(r["sample_key"])].get("labels") or {}).get(f),multilabel=f in MULTILABEL_FIELDS) is not None for r in all_rows) for f in FORMAL_FIELDS},"stratum_counts":dict(sorted(Counter(f"{p}::{r['original_dataset']}::{r['harmfulness']}" for p,rows in (("train",train),("validation",validation)) for r in rows).items()))}


def _json_hash(obj): return hashlib.sha256(json.dumps({k:v for k,v in obj.items() if k!="content_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def _lines_hash(lines): return hashlib.sha256(("\n".join(lines)+"\n").encode()).hexdigest()
def _atomic(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def _write_new_json(path,obj):
    data=(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode(); path=Path(path)
    if not path.exists() or path.read_bytes()!=data: _atomic(path,data)
def _write_new_jsonl(path,rows):
    data="".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows).encode(); path=Path(path)
    if not path.exists() or path.read_bytes()!=data: _atomic(path,data)
def _write_sha(path): _atomic(Path(str(path)+".sha256"),f"{sha256_file(path)}  {Path(path).name}\n".encode())

__all__=["build_protocol_v2","audit_source_split_v2","structured_conflicts","canonical_label","ProtocolV2Error"]
