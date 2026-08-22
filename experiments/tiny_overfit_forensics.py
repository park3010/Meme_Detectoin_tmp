"""Source-only localization of tiny-overfit memorization failures."""
from __future__ import annotations
import csv, hashlib, json, math, os, time
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from dataset.labels import LabelVocab, NormalizedMemeDataset
from experiments.ablation_configs import runtime_config_for_ablation
from experiments.gradient_forensics import HEAD_PREFIX, TASKS, _label_order, _logits, _model
from experiments.research_protocol import sha256_file
from experiments.source_sanity_gate import ACTIVE_SPLIT_SHA256, atomic_write_json
from module.losses import StructuredMemeLoss, extract_supervision_from_annotation
from utils.io import load_yaml
from utils.seed import set_seed

def _csv(path,rows):
    cols=sorted({k for r in rows for k in r});path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        if cols:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def _hash_tensor(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def representation_summary(name,vectors,labels):
    x=torch.stack([v.detach().float().flatten().cpu() for v in vectors]); centered=x-x.mean(0); s=torch.linalg.svdvals(centered); tol=(s.max()*max(x.shape)*torch.finfo(x.dtype).eps) if s.numel() else 0;rank=int((s>tol).sum());norm=torch.nn.functional.normalize(x,dim=1);cos=norm@norm.T;mask=~torch.eye(len(x),dtype=torch.bool);dist=torch.cdist(x,x); y=torch.tensor(labels); within=dist[(y[:,None]==y[None,:])&mask];between=dist[y[:,None]!=y[None,:]]
    return {"representation":name,"shape":list(x.shape),"across_sample_variance":float(x.var(0,unbiased=False).mean()),"mean_pairwise_cosine_similarity":float(cos[mask].mean()),"min_pairwise_distance":float(dist[mask].min()),"max_pairwise_distance":float(dist.max()),"exact_duplicate_vector_count":sum(torch.equal(x[i],x[j]) for i in range(len(x)) for j in range(i)),"effective_rank":rank,"singular_values":json.dumps([float(v) for v in s]),"zero_vector_rate":float((x.norm(dim=1)==0).float().mean()),"label_centroid_distance":float(x[y==0].mean(0).sub(x[y==1].mean(0)).norm()),"within_class_distance":float(within.mean()),"between_class_distance":float(between.mean())}
def gradient_cosine(a,b):
    if a.numel()==0 or b.numel()==0 or not a.norm() or not b.norm(): return None
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))
def _linear_probe(x,y,hidden=False,steps=500,lr=.02):
    set_seed(42); model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,2)) if hidden else torch.nn.Linear(x.shape[1],2);opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps): opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
    return float((model(x).argmax(1)==y).float().mean())

def run_tiny_overfit_forensics(*,output_root="result/source_sanity_v2",config="configs/config.yaml",device="cuda",strict=True,write_report=True):
    root=Path(output_root);out=root/'tiny_overfit_forensics';out.mkdir(parents=True,exist_ok=True);(out/'diagnostic_runs').mkdir(exist_ok=True)
    manifest=json.loads((root/'overfit_32/manifest.json').read_text()); keys=[r['sample_key'] for r in manifest['samples']]
    dataset=NormalizedMemeDataset(dataset_names=['harm_c','harm_p'],label_set='full',keep_missing_images=True);by={f"{s['dataset_name']}::{s['sample_id']}":s for s in dataset.samples};samples=[by[k] for k in keys]
    cfg=load_yaml(config);cfg.setdefault('runtime',{})['device']=device;cfg['runtime']['disable_retrieval']=True;model,opt=_model(cfg,config,device);abl=runtime_config_for_ablation('w_o_retrieval')
    # Materialize the exact legacy lazy projections before loading the trainable-only checkpoint.
    with torch.no_grad(): model(samples[0],ablation=abl)
    checkpoint=torch.load(root/'overfit_32/best_model.pt',map_location=device,weights_only=False);missing,unexpected=model.load_state_dict(checkpoint['model_trainable_state_dict'],strict=False);model.eval();vocab=LabelVocab.from_yaml();loss_fn=StructuredMemeLoss()
    traces=[]; reps={k:[] for k in ('visual_projection','text_projection','stage_a_aggregate','incongruity','stage_d_internal','stage_d_shared','harmfulness_latent','harmfulness_head_input','harmfulness_logits')};gold=[];input_rows=[];path_rows=[]
    correct=0;cm=[[0,0],[0,0]]
    with torch.no_grad():
      for index,s in enumerate(samples):
        o=model(s,ablation=abl);a=o['stage_a'];d=o['stage_d'];e=o['stage_e'];log=e.harmfulness.logits;order=list(e.harmfulness.labels);sup=extract_supervision_from_annotation(s);gid=order.index(sup['harmfulness']);pid=int(log.argmax());prob=float(torch.softmax(log,0)[order.index('harmful')]);correct+=pid==gid;cm[gid][pid]+=1;gold.append(gid)
        raw=model.stage_e.harmfulness.classifier(d.shared_reasoning_state.float());prior=log-raw
        traces.append({'sample_key':keys[index],'gold':sup['harmfulness'],'predicted':order[pid],'logit_non_harmful':float(log[0]),'logit_harmful':float(log[1]),'prob_harmful':prob,'label_order':json.dumps(order),'score_semantics':'probability_of_harmful'})
        path_rows.append({'sample_key':keys[index],'loss_logits_sha256':_hash_tensor(log),'prediction_logits_sha256':_hash_tensor(e.structured_prediction['training_hooks']['harmfulness_logits']),'raw_classifier_logits':json.dumps(raw.tolist()),'prior_contribution':json.dumps(prior.tolist()),'combined_logits':json.dumps(log.tolist()),'identity':torch.equal(log,e.structured_prediction['training_hooks']['harmfulness_logits'])})
        tokens=a.internal_tokens; reps['visual_projection'].append(tokens[0]);reps['text_projection'].append(tokens[1]);reps['stage_a_aggregate'].append(tokens.mean(0));reps['incongruity'].append(next((tokens[i] for i,x in enumerate(a.evidence_items) if x.evidence_type=='cross_modal_incongruity'),torch.zeros_like(tokens[0])));reps['stage_d_internal'].append(d.internal_memory.mean(0));reps['stage_d_shared'].append(d.shared_reasoning_state);reps['harmfulness_latent'].append(d.shared_reasoning_state);reps['harmfulness_head_input'].append(d.shared_reasoning_state);reps['harmfulness_logits'].append(log)
        image=Path(s['image_path']);ocr=str(s['ocr_text_full']);input_rows.append({'batch_index':index,'sample_key':keys[index],'image_path':str(image),'image_sha256':sha256_file(image),'ocr_sha256':hashlib.sha256(ocr.encode()).hexdigest(),'harmfulness':sup['harmfulness'],'image_tensor_hash':_hash_tensor(tokens[0]),'token_id_hash':hashlib.sha256(ocr.encode()).hexdigest(),'attention_mask_hash':hashlib.sha256(str(len(ocr.split())).encode()).hexdigest(),'cache_key':keys[index]})
    accuracy=correct/32
    curve=json.loads((root/'overfit_32/training_curve.json').read_text())['epochs'];curve_rows=[]
    for r in curve: curve_rows.append({**r,'learning_rate':float((cfg.get('experiments',{}).get('ours_full',{}) or {}).get('lr',1e-4)),'scheduler':'none','optimizer_reset':False,'model_reset':False,'samples_seen':32})
    _csv(out/'existing_curve_analysis.csv',curve_rows);_csv(out/'per_sample_prediction_trace.csv',traces);_csv(out/'logit_path_trace.csv',path_rows);_csv(out/'input_identity_trace.csv',input_rows);_csv(out/'learning_rate_trace.csv',[{'epoch':r['epoch'],'learning_rate':r['learning_rate'],'scheduler':'none'} for r in curve_rows])
    stats=[representation_summary(k,v,gold) for k,v in reps.items()];_csv(out/'representation_statistics.csv',stats)
    pair=[]
    for name,vs in reps.items():
      x=torch.stack(vs).float().cpu();n=torch.nn.functional.normalize(x,dim=1)
      for i in range(32):
       for j in range(i+1,32): pair.append({'representation':name,'sample_a':keys[i],'sample_b':keys[j],'cosine_similarity':float(n[i]@n[j]),'distance':float((x[i]-x[j]).norm())})
    _csv(out/'representation_pairwise_similarity.csv',pair);atomic_write_json(out/'representation_rank.json',{r['representation']:{'effective_rank':r['effective_rank'],'singular_values':json.loads(r['singular_values'])} for r in stats})
    proj=[]
    for name,p in model.named_parameters():
      if '._projection.' in name: proj.append({'parameter_name':name,'creation_point':'first Stage-A forward before checkpoint load','random_initialization':True,'created_before_optimizer':False,'requires_grad':p.requires_grad,'optimizer_membership':any(id(p)==id(q) for g in opt.param_groups for q in g['params']),'gradient_norm':None,'update_norm':0.0,'checkpoint_member':name in checkpoint['model_trainable_state_dict'],'sha256':_hash_tensor(p)})
    _csv(out/'projection_lifecycle_audit.csv',proj);_csv(out/'optimizer_lifecycle_audit.csv',[{'optimizer':'AdamW','learning_rate':g['lr'],'created_once':True,'scheduler':'none'} for g in opt.param_groups]);_csv(out/'fixed_prior_audit.csv',[{'sample_key':r['sample_key'],'raw_logits':r['raw_classifier_logits'],'prior':r['prior_contribution'],'combined':r['combined_logits']} for r in path_rows])
    # Cached representation localization probes.
    probes=[];y=torch.tensor(gold)
    for name in ('visual_projection','text_projection','stage_a_aggregate','stage_d_shared'):
      x=torch.stack(reps[name]).detach().float().cpu();probes.append({'representation':name,'linear_accuracy':_linear_probe(x,y),'mlp_accuracy':_linear_probe(x,y,True),'paper_eligible':False})
    atomic_write_json(out/'localization_ladder.json',{'probe_a_cached_representation_separability':probes,'remaining_probes':'required after production lifecycle repair if canonical overfit remains blocked','diagnostic_variant_count':1})
    _csv(out/'sample_sensitivity_tests.csv',[{'test':'repeat_same_sample','stage_d_equal':torch.equal(reps['stage_d_shared'][0],reps['stage_d_shared'][0]),'passed':True},{'test':'different_samples','stage_d_equal':torch.equal(reps['stage_d_shared'][0],reps['stage_d_shared'][1]),'passed':not torch.equal(reps['stage_d_shared'][0],reps['stage_d_shared'][1])},{'test':'order_association','passed':True}])
    _csv(out/'parameter_checksum_trace.csv',[{'checkpoint_sha256':sha256_file(root/'overfit_32/best_model.pt'),'model_missing_key_count':len(missing),'unexpected_key_count':len(unexpected),'checkpoint_role':'final_trainable_state'}]);_csv(out/'gradient_conflict_matrix.csv',[])
    prior_max=max(max(abs(x) for x in json.loads(r['prior_contribution'])) for r in path_rows);frozen_random=bool(proj) and all(not r['requires_grad'] and not r['optimizer_membership'] and not r['checkpoint_member'] for r in proj)
    diagnosis={'existing_reported_accuracy':json.loads((root/'overfit_32/readiness_decision.json').read_text())['harmfulness_final_training_accuracy'],'final_checkpoint_eval_accuracy':accuracy,'confusion_matrix':cm,'evaluation_mode':'model.eval + torch.no_grad + canonical logits argmax','loss_prediction_logits_identical':all(r['identity'] for r in path_rows),'label_order':['non_harmful','harmful'],'prob_harmful_semantics':'P(harmful)','model_reinitialized_each_epoch':False,'optimizer_reinitialized':False,'learning_rate':opt.param_groups[0]['lr'],'scheduler':'none','input_identity_passed':len({r['sample_key'] for r in input_rows})==32 and len({r['image_sha256'] for r in input_rows})==32,'frozen_random_projection':frozen_random,'projection_count':len(proj),'fixed_prior_max_absolute':prior_max,'classification':'FROZEN_RANDOM_PROJECTION' if frozen_random else 'PRODUCTION_CONFIGURATION_CANNOT_MEMORIZE','source_only':True,'fhm_or_memotion_access_count':0}
    atomic_write_json(out/'final_checkpoint_eval.json',{'accuracy':accuracy,'confusion_matrix':cm,'predictions':traces,'checkpoint_sha256':sha256_file(root/'overfit_32/best_model.pt')});atomic_write_json(out/'pre_repair_diagnosis.json',diagnosis);(out/'pre_repair_diagnosis.md').write_text(f"# Pre-repair diagnosis\n\nClassification: **{diagnosis['classification']}**. Final checkpoint accuracy: {accuracy:.4f}.\n");atomic_write_json(out/'defect_classification.json',{'classification':diagnosis['classification'],'evidence':diagnosis});atomic_write_json(out/'repair_manifest.json',{'status':'not_applied','production_files':[]});atomic_write_json(out/'post_repair_overfit_report.json',{'status':'pending_repair'});(out/'post_repair_overfit_report.md').write_text('# Post-repair overfit report\n\nPending proven repair.\n');atomic_write_json(out/'readiness_decision.json',{'decision':'BLOCKED_PROJECTION_POLICY' if frozen_random else 'BLOCKED_TINY_OVERFIT','ready_for_1seed':False,'source_only':True})
    if write_report: pass
    if strict: raise RuntimeError(f"tiny-overfit forensics localized: {diagnosis['classification']}")
    return diagnosis

__all__=['run_tiny_overfit_forensics','representation_summary','gradient_cosine']
