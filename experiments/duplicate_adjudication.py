"""Read-only duplicate-image adjudication for HarMeme source data."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from experiments.research_protocol import sha256_file
from experiments.source_sanity import FORBIDDEN_MARKERS, SOURCE_DATASETS, reject_forbidden_manifest
from utils.io import read_jsonl, write_json
from utils.text_utils import normalize_text


def decoded_pixel_sha256(path: str | Path) -> tuple[str, tuple[int, int], str]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = rgb.tobytes()
        digest = hashlib.sha256(rgb.width.to_bytes(8,"big") + rgb.height.to_bytes(8,"big") + payload).hexdigest()
        small = rgb.resize((8,8)).convert("L")
        values = list(small.getdata()); threshold = sum(values)/len(values)
        phash = f"{sum((1 << index) for index,value in enumerate(values) if value >= threshold):016x}"
        return digest, rgb.size, phash


def multimodal_group_id(pixel_hash: str, exact_ocr: str) -> str:
    return hashlib.sha256((pixel_hash + "\0" + exact_ocr).encode("utf-8")).hexdigest()


def canonical_member(rows: list[dict[str,Any]]) -> dict[str,Any]:
    return min(rows,key=lambda row:str(row["sample_key"]))


def classify_group(rows: list[dict[str,Any]]) -> tuple[str,list[str],str]:
    raw={str(r.get("raw_label_numeric")) for r in rows}; normalized={str(r.get("normalized_harmfulness")) for r in rows}
    source_ocr={str(r.get("source_text")) for r in rows}; normalized_ocr={str(r.get("normalized_text")) for r in rows}
    if any(not r.get("source_normalized_ocr_match") for r in rows):
        return "SUSPECTED_IMAGE_TEXT_PAIRING_ERROR",["pairing_mismatch"],"FIX_PROVEN_PAIRING_ERROR"
    if len(raw)==1 and len(normalized)>1:
        return "NORMALIZATION_INDUCED_LABEL_CONFLICT",["normalization_mismatch"],"FIX_PROVEN_NORMALIZATION_ERROR"
    if len(raw)>1:
        return "RAW_SOURCE_LABEL_CONFLICT",["raw_binary_labels_conflict"],"EXCLUDE_ALL_CONFLICTING_MEMBERS"
    if len(source_ocr)>1:
        return "SHARED_IMAGE_DIFFERENT_TEXT_CONFIRMED",["official_source_records_use_different_text"],"KEEP_AS_DISTINCT_MULTIMODAL_SAMPLES_GROUP_BY_IMAGE_SPLIT"
    if len(normalized_ocr)==1:
        if len({r.get("raw_file_sha256") for r in rows})>1:
            return "FORMAT_DUPLICATE_ONLY",["decoded_pixels_identical_raw_bytes_differ"],"RETAIN_ONE_CANONICAL_SAMPLE"
        return "EXACT_MULTIMODAL_DUPLICATE_SAME_LABEL",[],"RETAIN_ONE_CANONICAL_SAMPLE"
    return "UNRESOLVED_REQUIRES_MANUAL_REVIEW",[],"MANUAL_REVIEW_REQUIRED"


def run_duplicate_adjudication(*, output_root: str="result/source_sanity", strict: bool=False, write_report: bool=False, force: bool=False) -> dict[str,Any]:
    root=Path(output_root); out=root/"duplicate_adjudication"
    if out.exists() and any(out.iterdir()) and not force: raise FileExistsError(f"output exists; use --force: {out}")
    out.mkdir(parents=True,exist_ok=True)
    split_path=Path("result/splits/harmeme/source_split_seed_42.json"); split=json.loads(split_path.read_text())
    split_rows=list(split["train"])+list(split["validation"]); reject_forbidden_manifest(split_rows)
    split_index={str(r["sample_key"]):("train" if r in split["train"] else "validation") for r in split_rows}
    structured_index={str(r["sample_key"]):bool(r.get("structured_label_eligible")) for r in split_rows}
    source_records, source_locations=_source_records()
    annotations, annotation_locations=_indexed_jsonl("dataset/annotation", lambda r:(str(r.get("dataset_name","")).replace("harmc","harm_c").replace("harmp","harm_p"),str(r.get("sample_id") or r.get("id") or "")), annotation=True)
    normalized, normalized_locations=_normalized_records()
    members=[]
    for dataset_name,folder,domain in (("harm_c","covid_img+text","covid"),("harm_p","political_img+text","politics")):
        for key,record in source_records.items():
            if key[0]!=dataset_name: continue
            sid=key[1]; sample_key=f"{dataset_name}::{sid}"; image=Path("dataset/source")/folder/"img"/Path(str(record.get("image") or f"{sid}.png")).name
            pixel_hash,dimensions,phash=decoded_pixel_sha256(image); exact_text=str(record.get("text") or record.get("ocr_text_full") or "")
            norm=normalized.get(key,{}) or {}; norm_text=str(norm.get("ocr_text_full") or "")
            raw=record.get("labels",record.get("label")); normalized_label=(norm.get("labels") or {}).get("harmfulness")
            members.append({"sample_key":sample_key,"dataset_name":dataset_name,"original_dataset":dataset_name,"domain":domain,"sample_id":sid,"image_path":str(image),"raw_file_sha256":sha256_file(image),"decoded_rgb_pixel_sha256":pixel_hash,"perceptual_hash":phash,"file_size":image.stat().st_size,"width":dimensions[0],"height":dimensions[1],"ocr_text":exact_text,"exact_ocr_sha256":hashlib.sha256(exact_text.encode()).hexdigest(),"normalized_ocr_sha256":hashlib.sha256(normalize_text(unicodedata.normalize("NFKC",exact_text)).encode()).hexdigest(),"unicode_normalized_ocr":unicodedata.normalize("NFKC",exact_text),"whitespace_normalized_ocr":normalize_text(exact_text),"lowercase_diagnostic_ocr":normalize_text(exact_text).lower(),"source_text":exact_text,"normalized_text":norm_text,"source_normalized_ocr_match":normalize_text(exact_text)==norm_text,"raw_label_original":raw,"raw_label_text":str(raw),"raw_label_numeric":raw if isinstance(raw,(int,float)) else None,"normalized_harmfulness":normalized_label,"binary_merge_rule":"harm_c/harm_p direct 1->harmful, 0->non_harmful","split":split_index.get(sample_key),"structured_labels":norm.get("labels",{}),"structured_label_eligible":structured_index.get(sample_key),"annotation_source_file":annotation_locations.get(key,{}).get("path"),"annotation_row_index":annotation_locations.get(key,{}).get("index"),"normalized_annotation_source":normalized_locations.get(key,{}).get("path"),"normalized_annotation_row_index":normalized_locations.get(key,{}).get("index"),"confidence":(norm.get("labels") or {}).get("confidence"),"confidence_score":(norm.get("labels") or {}).get("confidence_score"),"audit_flags":norm.get("audit_flags",[]),"image_group_id":pixel_hash,"multimodal_group_id":multimodal_group_id(pixel_hash,exact_text)})
    by_pixel=defaultdict(list)
    for row in members: by_pixel[row["image_group_id"]].append(row)
    groups=[]; traces=[]; ocr_traces=[]
    for number,(gid,rows) in enumerate(sorted((k,v) for k,v in by_pixel.items() if len(v)>1),1):
        category,flags,action=classify_group(rows); splits={r["split"] for r in rows}; crosses=len(splits)>1
        flags.extend(flag for flag,condition in (("crosses_train_validation",crosses),("affects_image_only_baseline",crosses),("affects_multimodal_baseline",crosses and len({r['multimodal_group_id'] for r in rows})<len(rows)),("affects_structured_labels",any(r["structured_label_eligible"] for r in rows)),("paper_protocol_blocker",category in {"RAW_SOURCE_LABEL_CONFLICT","NORMALIZATION_INDUCED_LABEL_CONFLICT","SUSPECTED_IMAGE_TEXT_PAIRING_ERROR","UNRESOLVED_REQUIRES_MANUAL_REVIEW"})) if condition)
        group_id=f"dup_{number:03d}"; canonical=canonical_member(rows)["sample_key"]
        group={"group_id":group_id,"image_group_id":gid,"member_count":len(rows),"sample_keys":" ".join(r["sample_key"] for r in rows),"splits":" ".join(sorted(splits)),"crosses_train_validation":crosses,"raw_file_hash_count":len({r["raw_file_sha256"] for r in rows}),"multimodal_group_count":len({r["multimodal_group_id"] for r in rows}),"exact_multimodal_duplicate":len({r["multimodal_group_id"] for r in rows})==1,"same_normalized_label":len({r["normalized_harmfulness"] for r in rows})==1,"primary_category":category,"secondary_flags":" ".join(sorted(set(flags))),"proposed_action":action,"canonical_sample_key":canonical}
        groups.append(group)
        for row in rows:
            row["group_id"]=group_id; row["primary_category"]=category; row["proposed_action"]=action
            traces.append({k:row.get(k) for k in ("group_id","sample_key","raw_label_original","raw_label_text","raw_label_numeric","normalized_harmfulness","binary_merge_rule","annotation_source_file","annotation_row_index","normalized_annotation_source","normalized_annotation_row_index","confidence","confidence_score","audit_flags")}|{"conflict_exists_in_raw":len({r["raw_label_numeric"] for r in rows})>1,"introduced_by_normalization":len({r["raw_label_numeric"] for r in rows})==1 and len({r["normalized_harmfulness"] for r in rows})>1,"classification":category})
            ocr_traces.append({k:row.get(k) for k in ("group_id","sample_key","image_group_id","multimodal_group_id","ocr_text","exact_ocr_sha256","whitespace_normalized_ocr","unicode_normalized_ocr","lowercase_diagnostic_ocr","source_text","normalized_text","source_normalized_ocr_match")}|{"official_source_different_text":len({r["source_text"] for r in rows})>1})
    retrieval=_retrieval_impact([r for r in members if any(r["sample_key"] in g["sample_keys"].split() for g in groups)])
    retrieval_by_key={r["sample_key"]:r for r in retrieval}
    for group in groups:
        if any(retrieval_by_key.get(key,{}).get("is_retrieval_query_origin") for key in group["sample_keys"].split()): group["secondary_flags"] += " included_in_retrieval_origin"
    preview=_split_preview(split_rows,groups,members)
    changed_keys=set(preview["excluded_ids"]) | set(preview["moved_ids"])
    for row in retrieval:
        requires=bool(row["is_retrieval_query_origin"] and row["sample_key"] in changed_keys)
        row["exclusion_or_movement_requires_rebuild"]=requires
        row["expected_rebuild_scope"]="source query cache + corpus provenance + sparse/dense indexes" if requires else "none"
    counts=Counter(g["primary_category"] for g in groups); cross=[g for g in groups if g["crosses_train_validation"]]
    summary={"schema_version":"duplicate_adjudication_v1","source_only":True,"total_duplicate_groups":len(groups),"total_duplicate_members":sum(g["member_count"] for g in groups),"category_counts":dict(counts),"cross_split_group_count":len(cross),"cross_split_image_overlap_count":len(cross),"cross_split_multimodal_overlap_count":sum(g["crosses_train_validation"] and g["exact_multimodal_duplicate"] for g in groups),"same_label_cross_split_group_count":sum(g["crosses_train_validation"] and g["same_normalized_label"] for g in groups),"conflicting_label_cross_split_group_count":sum(g["crosses_train_validation"] and not g["same_normalized_label"] for g in groups),"train_only_group_count":sum(g["splits"]=="train" for g in groups),"validation_only_group_count":sum(g["splits"]=="validation" for g in groups),"split_v2_preview":preview,"generated_at_utc":datetime.now(timezone.utc).isoformat()}
    decision=_decision(groups); readiness={"decision":decision,"ready_for_1seed":False,"source_only":True,"scientific_artifacts_modified":False}
    _csv(out/"duplicate_groups.csv",groups); _csv(out/"conflict_groups.csv",[g for g in groups if not g["same_normalized_label"]]); _csv(out/"cross_split_groups.csv",cross); _csv(out/"raw_label_trace.csv",traces); _csv(out/"ocr_pairing_trace.csv",ocr_traces); _csv(out/"retrieval_impact.csv",retrieval)
    write_json(out/"adjudication_summary.json",summary); write_json(out/"split_v2_impact_preview.json",preview); write_json(out/"readiness_decision.json",readiness)
    write_json(out/"proposed_resolution_plan.json",{"preview_only":True,"default_policy":{"exact_same_label":"RETAIN_ONE_CANONICAL_SAMPLE","unresolved_conflicting":"EXCLUDE_ALL_CONFLICTING_MEMBERS","shared_image_different_text":"KEEP_AS_DISTINCT_MULTIMODAL_SAMPLES_GROUP_BY_IMAGE_SPLIT"},"groups":[{"group_id":g["group_id"],"sample_keys":g["sample_keys"],"proposed_action":g["proposed_action"],"canonical_sample_key":g["canonical_sample_key"]} for g in groups]})
    if write_report: (out/"adjudication_summary.md").write_text(_markdown(summary,readiness),encoding="utf-8")
    (out/"review_gallery.html").write_text(_gallery(groups,members),encoding="utf-8")
    result={**summary,"readiness":readiness,"passed":decision=="READY_FOR_RESOLUTION_APPROVAL"}
    if strict and decision not in {"READY_FOR_RESOLUTION_APPROVAL","BLOCKED_RAW_LABEL_CONFLICT","BLOCKED_MANUAL_REVIEW"}: raise RuntimeError(f"duplicate adjudication strict blocker: {decision}")
    return result


def _source_records():
    records={}; locations={}
    for dataset,folder in (("harm_c","covid_img+text"),("harm_p","political_img+text")):
        path=Path("dataset/source")/folder/"txt"/"all.jsonl"
        for index,row in enumerate(read_jsonl(path),1):
            sid=str(row.get("id") or row.get("sample_id")); records[(dataset,sid)]=row; locations[(dataset,sid)]={"path":str(path),"index":index}
    return records,locations


def _indexed_jsonl(root, key_fn, annotation=False):
    index={}; locations={}
    for path in sorted(Path(root).rglob("*annotation*.jsonl")):
        if "raw_responses" in path.name: continue
        for number,row in enumerate(read_jsonl(path),1):
            key=key_fn(row)
            if key[0] in SOURCE_DATASETS and key[1]: index[key]=row.get("annotation",row) if annotation else row; locations[key]={"path":str(path),"index":number}
    return index,locations


def _normalized_records():
    index={};locations={}
    for dataset in sorted(SOURCE_DATASETS):
        path=Path("dataset/annotation_normalized")/dataset/"normalized_labels.jsonl"
        for number,row in enumerate(read_jsonl(path),1): index[(dataset,str(row["sample_id"]))]=row;locations[(dataset,str(row["sample_id"]))]={"path":str(path),"index":number}
    return index,locations


def _retrieval_impact(members):
    query_path=Path("dataset/retrieval/harmeme_train_v1/cache/source_queries/queries.jsonl"); corpus_path=Path("dataset/retrieval/harmeme_train_v1/corpus/corpus_texts.jsonl")
    queries=defaultdict(list)
    for row in read_jsonl(query_path): queries[str(row.get("sample_key"))].append(row)
    unique_docs=Counter()
    for row in read_jsonl(corpus_path):
        origins=list(row.get("origin_sample_ids",[]) or [])
        if len(origins)==1: unique_docs[str(origins[0])]+=1
    output=[]
    for member in members:
        key=member["sample_key"]; rows=queries.get(key,[]); docs={doc for row in rows for doc in row.get("retrieved_document_ids",[])}
        affected=member.get("split")=="train" and bool(rows)
        output.append({"sample_key":key,"split":member.get("split"),"is_retrieval_query_origin":bool(rows),"query_row_count":len(rows),"retrieved_document_count":len(docs),"unique_corpus_documents_attributable_only_to_sample":unique_docs[key],"exclusion_or_movement_requires_rebuild":affected,"expected_rebuild_scope":"source query cache + corpus provenance + sparse/dense indexes" if affected else "none"})
    return output


def _split_preview(split_rows,groups,members):
    action_by_key={key:g for g in groups for key in g["sample_keys"].split()}; member_by_key={m["sample_key"]:m for m in members}
    retained=[];excluded=[]
    for row in split_rows:
        key=str(row["sample_key"]); group=action_by_key.get(key)
        if group and group["proposed_action"]=="EXCLUDE_ALL_CONFLICTING_MEMBERS": excluded.append(key);continue
        if group and group["proposed_action"]=="RETAIN_ONE_CANONICAL_SAMPLE" and key!=group["canonical_sample_key"]:excluded.append(key);continue
        retained.append(dict(row))
    original={str(row["sample_key"]):("train" if index < 5611 else "validation") for index,row in enumerate(split_rows)}
    assigned={member_by_key[str(row["sample_key"])]["image_group_id"]:original[str(row["sample_key"])] for row in retained}
    # Only cross-split retained groups require movement. Preserve all other
    # immutable-v1 assignments and choose the canonical member's partition.
    for group in groups:
        keys=[key for key in group["sample_keys"].split() if key not in excluded]
        if keys and len({original[key] for key in keys})>1:
            assigned[group["image_group_id"]]=original[group["canonical_sample_key"]]
    moved=[]; preview_rows=[]
    for row in retained:
        image_id=member_by_key[str(row["sample_key"])]["image_group_id"]; new=assigned[image_id]
        old=original[str(row["sample_key"])]
        if old!=new:moved.append(str(row["sample_key"]))
        preview_rows.append({**row,"preview_partition":new,"image_group_id":image_id,"multimodal_group_id":member_by_key[str(row["sample_key"])]["multimodal_group_id"]})
    train=[r for r in preview_rows if r["preview_partition"]=="train"]; val=[r for r in preview_rows if r["preview_partition"]=="validation"]
    train_img={r["image_group_id"] for r in train};val_img={r["image_group_id"] for r in val};train_mm={r["multimodal_group_id"] for r in train};val_mm={r["multimodal_group_id"] for r in val}
    return {"preview_only":True,"policy":"group-aware decoded-pixel split; exact same-label duplicates canonicalized; conflicting groups excluded","total_retained_sample_count":len(preview_rows),"train_count":len(train),"validation_count":len(val),"domain_counts":dict(Counter(r["dataset_name"] for r in preview_rows)),"harmfulness_counts":dict(Counter(r["harmfulness"] for r in preview_rows)),"structured_eligible_count":sum(bool(r.get("structured_label_eligible")) for r in preview_rows),"stratum_counts":dict(Counter(f"{r['preview_partition']}::{r['dataset_name']}::{r['harmfulness']}" for r in preview_rows)),"moved_id_count":len(moved),"moved_ids":moved,"excluded_id_count":len(excluded),"excluded_ids":excluded,"image_overlap_after_policy":len(train_img&val_img),"multimodal_overlap_after_policy":len(train_mm&val_mm),"prospective_split_sha256":hashlib.sha256(json.dumps(preview_rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def _decision(groups):
    categories={g["primary_category"] for g in groups}
    blockers=[]
    if "SUSPECTED_IMAGE_TEXT_PAIRING_ERROR" in categories:blockers.append("BLOCKED_PAIRING_ERROR")
    if "NORMALIZATION_INDUCED_LABEL_CONFLICT" in categories:blockers.append("BLOCKED_NORMALIZATION_ERROR")
    if "RAW_SOURCE_LABEL_CONFLICT" in categories:blockers.append("BLOCKED_RAW_LABEL_CONFLICT")
    if "UNRESOLVED_REQUIRES_MANUAL_REVIEW" in categories:blockers.append("BLOCKED_MANUAL_REVIEW")
    return blockers[0] if len(blockers)==1 else "BLOCKED_MULTIPLE" if blockers else "READY_FOR_RESOLUTION_APPROVAL"


def _gallery(groups,members):
    by={g["group_id"]:g for g in groups}; rows=[]
    for member in members:
        group=next((g for g in groups if member["sample_key"] in g["sample_keys"].split()),None)
        if not group:continue
        src="../../../"+member["image_path"]
        rows.append(f'<article><h2>{html.escape(group["group_id"])} — {html.escape(group["primary_category"])}</h2><img src="{html.escape(src)}" loading="lazy"><pre>{html.escape(json.dumps({k:member.get(k) for k in ("sample_key","raw_file_sha256","decoded_rgb_pixel_sha256","ocr_text","raw_label_original","normalized_harmfulness","split","proposed_action")},ensure_ascii=False,indent=2))}</pre></article>')
    return '<!doctype html><meta charset="utf-8"><title>Local duplicate review</title><style>img{max-width:360px;max-height:300px}article{border-bottom:1px solid #999;padding:1rem}pre{white-space:pre-wrap}</style><h1>HarMeme duplicate review — local only</h1>'+"".join(rows)


def _markdown(summary,readiness):return f"# Duplicate-image adjudication\n\nGroups: {summary['total_duplicate_groups']}  \nCross-split image groups: {summary['cross_split_image_overlap_count']}  \nDecision: **{readiness['decision']}**\n"


def _csv(path,rows):
    columns=sorted({k for r in rows for k in r})
    with Path(path).open("w",encoding="utf-8",newline="") as handle:
        if not columns:return
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


__all__=["canonical_member","classify_group","decoded_pixel_sha256","multimodal_group_id","run_duplicate_adjudication"]
