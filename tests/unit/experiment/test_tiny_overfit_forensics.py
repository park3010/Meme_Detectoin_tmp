import json
from pathlib import Path
import torch
from torch import nn
from experiments.tiny_overfit_forensics import gradient_cosine, representation_summary
from experiments.train import OursRunConfig, configure_trainable_parameters, materialize_trainable_projections

class LazyLeaf(nn.Module):
    def __init__(self): super().__init__();self._projection=None;self.backbone=nn.Linear(2,2)
class DummyPipeline(nn.Module):
    def __init__(self):
        super().__init__();self.stage_a=nn.Module();self.stage_a.visual_encoder=nn.Module();self.stage_a.visual_encoder.clip=LazyLeaf();self.stage_a.text_encoder=nn.Module();self.stage_a.text_encoder.encoder=LazyLeaf();self.stage_a.local_extractor=nn.Module();self.stage_a.local_extractor.roi_encoder=LazyLeaf();self.stage_c=nn.Module();self.stage_c.relevance=nn.Module();self.stage_c.relevance.feature_mlp=nn.Linear(2,2);self.stage_d=nn.Linear(2,2);self.stage_e=nn.Linear(2,2)
    def forward(self,sample,run_until='a'):
        for leaf in (self.stage_a.visual_encoder.clip,self.stage_a.text_encoder.encoder,self.stage_a.local_extractor.roi_encoder):
            if leaf._projection is None: leaf._projection=nn.Linear(3,2)
        return {}

def test_projection_materialized_before_optimizer_and_trainable():
    p=DummyPipeline();materialize_trainable_projections(p,[{}]);configure_trainable_parameters(p,OursRunConfig(dataset_name='harmeme'))
    optimizer=torch.optim.AdamW([x for x in p.parameters() if x.requires_grad],lr=1e-4);ids={id(x) for g in optimizer.param_groups for x in g['params']}
    projections=[x for n,x in p.named_parameters() if '._projection.' in n]
    assert len(projections)==6 and all(x.requires_grad and id(x) in ids for x in projections)
    assert all(not x.requires_grad for n,x in p.stage_a.named_parameters() if '._projection.' not in n)

def test_trainable_projection_operation_is_not_method_detached():
    from module.backbone.text import TextEncoderWrapper
    encoder=TextEncoderWrapper(hidden_dim=4,prefer_transformers=False);encoder._project_matrix(torch.ones(2,7));encoder._projection.requires_grad_(True)
    output=encoder._project_matrix(torch.ones(2,7));output.sum().backward()
    assert encoder._projection.weight.grad is not None

def test_encode_methods_do_not_wrap_projection_in_no_grad():
    import inspect
    from module.backbone.text import TextEncoderWrapper
    from module.backbone.vision import CLIPWrapper
    assert '@torch.no_grad' not in inspect.getsource(TextEncoderWrapper.encode)
    assert '@torch.no_grad' not in inspect.getsource(CLIPWrapper.encode_image)

def test_representation_duplicate_and_rank_detection():
    labels=[0,0,1,1];collapsed=[torch.ones(3) for _ in labels];r=representation_summary('x',collapsed,labels)
    assert r['exact_duplicate_vector_count']==6 and r['effective_rank']==0 and r['across_sample_variance']==0

def test_diverse_representation_has_nonzero_rank():
    r=representation_summary('x',[torch.tensor([float(i),float(i*i)]) for i in range(4)],[0,0,1,1])
    assert r['effective_rank']>0 and r['exact_duplicate_vector_count']==0

def test_gradient_cosine_sign():
    assert gradient_cosine(torch.tensor([1.,0]),torch.tensor([-1.,0]))==-1.0

def test_loss_prediction_identity_and_prob_semantics():
    logits=torch.tensor([-.3,.7]);hook=logits
    assert torch.equal(logits,hook) and int(logits.argmax())==1
    assert float(torch.softmax(logits,0)[1])>0.5

def test_eval_mode_is_canonical_in_forensics():
    source=Path('experiments/tiny_overfit_forensics.py').read_text();assert 'model.eval()' in source and 'torch.no_grad()' in source

def test_diagnostic_probes_are_paper_ineligible():
    source=Path('experiments/tiny_overfit_forensics.py').read_text();assert "'paper_eligible':False" in source

def test_extended_budget_does_not_modify_config():
    source=Path('experiments/tiny_overfit_forensics.py').read_text();assert "Path(config).write" not in source and "open(config," not in source

def test_source_only_loader_contract():
    source=Path('experiments/tiny_overfit_forensics.py').read_text();assert "dataset_names=['harm_c','harm_p']" in source
    assert "dataset_names=['facebook']" not in source and "dataset_names=['memotion']" not in source

def test_readiness_requires_canonical_pass():
    readiness=json.loads(Path('result/source_sanity_v2/tiny_overfit_forensics/readiness_decision.json').read_text())
    assert readiness['ready_for_1seed'] is False
