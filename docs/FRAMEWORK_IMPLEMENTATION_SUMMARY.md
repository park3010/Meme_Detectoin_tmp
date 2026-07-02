# Framework Implementation Summary

## Current Code Layout

```text
module/
├── internal_evidence_extractor.py       # Stage A schemas, encoders, ROI cues, incongruity, extractor
├── external_knowledge_acquisition.py    # Stage B schemas, query construction, linking, retrieval, hypotheses
├── knowledge_filter_verifier.py         # Stage C schemas, relevance, support, validity, redundancy, verifier
├── evidence_fusion_reasoning.py         # Stage D schemas, aggregation, cross-attention, gates, task latents
├── structured_interpretation_head.py    # Stage E schemas, heads, evidence attribution, rationale, provenance
├── baseline.py                          # Simple baseline classifiers and registry
├── losses.py                            # Structured losses and supervision extraction
├── runner.py                            # HarmfulMemePipeline, PipelineRunner, run_single_sample
└── backbone/
    ├── vision.py                        # CLIP fallback and detector adapter
    ├── text.py                          # HF/hash text encoder
    ├── retrieval.py                     # local retriever and cross-encoder fallback
    └── generation.py                    # template hypothesis generator
```

## Config Layout

`configs/config.yaml` is the canonical runtime/model/experiment configuration. Label and normalization schemas stay separate:

```text
configs/config.yaml
configs/label_vocab.yaml
configs/annotation_normalization.yaml
```

## Stage Roles

- Stage A extracts internal image/text/ROI/incongruity evidence and auxiliary cues.
- Stage B builds evidence-aware queries, links entities/concepts, retrieves candidates, and generates context hypotheses.
- Stage C scores relevance, verifies support/contradiction, validates source/time/culture metadata, and produces verified knowledge.
- Stage D fuses internal evidence with verified knowledge using attention and gates, producing shared and task-specific latents.
- Stage E predicts structured harmfulness, target, intent, tactic, evidence attribution, rationale, and training hooks.

## Public Imports

```python
from module.runner import HarmfulMemePipeline, PipelineRunner
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.structured_interpretation_head import StructuredInterpretationHead
from module.baseline import build_baseline, BASELINE_REGISTRY
from module.losses import StructuredMemeLoss
```
