from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from module.baseline import MLPClassifierHead
from module.baseline import CLIPTextConcatClassifier
from module.baseline import ImageOnlyCLIPClassifier
from module.baseline import TextOnlyEncoderClassifier
from module.internal_evidence_extractor import InternalEvidenceExtractor
from module.external_knowledge_acquisition import ExternalKnowledgeAcquisition
from module.knowledge_filter_verifier import KnowledgeRelevanceFilterVerifier
from module.evidence_fusion_reasoning import EvidenceFusionReasoning
from module.structured_interpretation_head import StructuredInterpretationHead
from module.runner import print_pipeline_components
from module.runner import HarmfulMemePipeline
import run


def test_consolidated_public_imports_work():
    assert ImageOnlyCLIPClassifier.model_name == "image_only_clip"
    assert TextOnlyEncoderClassifier.model_name == "text_only_encoder"
    assert CLIPTextConcatClassifier.model_name == "clip_text_concat"
    assert MLPClassifierHead(2).net is not None
    assert InternalEvidenceExtractor is not None
    assert ExternalKnowledgeAcquisition is not None
    assert KnowledgeRelevanceFilterVerifier is not None
    assert EvidenceFusionReasoning is not None
    assert StructuredInterpretationHead is not None


def test_unified_cli_exposes_required_subcommands():
    parser = run.build_parser()
    for command in ["train", "baseline", "stage", "assets", "evaluate", "ablation", "audit", "suite", "preflight", "data", "report", "analysis"]:
        assert command in parser.format_help()


def test_required_consolidated_files_exist():
    for relative in [
        "module/internal_evidence_extractor.py",
        "module/external_knowledge_acquisition.py",
        "module/knowledge_filter_verifier.py",
        "module/evidence_fusion_reasoning.py",
        "module/structured_interpretation_head.py",
        "module/baseline.py",
        "module/losses.py",
        "module/runner.py",
        "module/backbone/vision.py",
        "module/backbone/text.py",
        "module/backbone/retrieval.py",
        "module/backbone/generation.py",
        "configs/config.yaml",
        "experiments/data_preparation.py",
        "experiments/reporting.py",
        "experiments/statistics.py",
        "experiments/posthoc_error_analysis.py",
        "experiments/posthoc_quality_evaluation.py",
        "scripts/commands/experiment.py",
        "scripts/commands/data.py",
        "scripts/commands/report.py",
        "scripts/commands/analysis.py",
        "scripts/presets/run_preflight.sh",
        "scripts/presets/run_core_smoke.sh",
        "scripts/presets/run_core_1seed.sh",
        "scripts/presets/run_core_5seed.sh",
    ]:
        assert (ROOT / relative).exists()


def test_legacy_paths_are_physically_removed():
    def script(name: str) -> str:
        return f"scripts/{name}.py"

    def config(name: str) -> str:
        return f"configs/{name}.yaml"

    removed_paths = [
        "module/stage_a",
        "module/stage_b",
        "module/stage_c",
        "module/stage_d",
        "module/stage_e",
        "module/pipeline",
        "module/baselines",
        "module/losses",
        "module/backbones",
        "dataset/label_adapter.py",
        "dataset/normalized_labels.py",
        "experiments/train_ours.py",
        "experiments/train_baseline.py",
        "experiments/metrics.py",
        "experiments/structured_eval.py",
        "experiments/evaluate_predictions.py",
        "experiments/annotation_normalization.py",
        "experiments/annotation_audit.py",
        "experiments/dataset_stats.py",
        "experiments/aggregate_results.py",
        "experiments/paper_tables.py",
        "experiments/significance.py",
        "experiments/error_case_analysis.py",
        "experiments/subset_analysis.py",
        "experiments/rationale_eval.py",
        "experiments/verifier_eval.py",
        "experiments/components.py",
        script("audit_annotations"),
        script("build_normalized_labels"),
        script("inspect_dataset"),
        script("inspect_normalized_labels"),
        script("export_intermediate_results"),
        "scripts/run_all_experiments.sh",
        "scripts/run_exp_phase1.sh",
        "scripts/run_exp_phase2.sh",
        "scripts/run_exp_phase3.sh",
        "scripts/run_paper_tables_only.sh",
        "scripts/run_pipeline_audit_smoke.sh",
        "scripts/run_smoke_experiments.sh",
        script("run_ours_full"),
        script("run_stage_a"),
        script("run_stage_b"),
        script("run_stage_c"),
        script("run_stage_d"),
        script("run_stage_e"),
        script("run_pipeline"),
        script("run_baseline_text_only"),
        script("run_baseline_image_only"),
        script("run_baseline_clip_concat"),
        script("run_ablation"),
        config("default"),
        config("pipeline"),
        config("stage_a"),
        config("stage_b"),
        config("stage_c"),
        config("stage_d"),
        config("stage_e"),
        "configs/experiments",
    ]
    for relative in removed_paths:
        assert not (ROOT / relative).exists(), relative


def test_config_directory_contains_only_canonical_runtime_files():
    files = sorted(path.name for path in (ROOT / "configs").iterdir())
    assert files == ["annotation_normalization.yaml", "config.yaml", "label_vocab.yaml"]


def test_active_source_has_no_legacy_imports_or_deleted_cli_references():
    forbidden = [
        "module." + "stage_a",
        "module." + "stage_b",
        "module." + "stage_c",
        "module." + "stage_d",
        "module." + "stage_e",
        "module." + "pipeline",
        "module." + "baselines",
        "module." + "losses.",
        "module." + "backbones",
        "dataset." + "label_adapter",
        "dataset." + "normalized_labels",
        "experiments." + "train_ours",
        "experiments." + "train_baseline",
        "experiments." + "metrics",
        "experiments." + "structured_eval",
        "experiments." + "evaluate_predictions",
        "experiments." + "annotation_normalization",
        "experiments." + "annotation_audit",
        "experiments." + "dataset_stats",
        "experiments." + "aggregate_results",
        "experiments." + "paper_tables",
        "experiments." + "significance",
        "experiments." + "error_case_analysis",
        "experiments." + "subset_analysis",
        "experiments." + "rationale_eval",
        "experiments." + "verifier_eval",
        "experiments." + "components",
        "run_ours_full" + ".py",
        "run_stage_a" + ".py",
        "run_stage_b" + ".py",
        "run_stage_c" + ".py",
        "run_stage_d" + ".py",
        "run_stage_e" + ".py",
        "run_pipeline" + ".py",
        "run_baseline_text_only" + ".py",
        "run_baseline_image_only" + ".py",
        "run_baseline_clip_concat" + ".py",
        "run_ablation" + ".py",
    ]
    roots = ["module", "dataset", "experiments", "scripts", "tests", "tool", "utils"]
    suffixes = {".py", ".sh"}
    offenders = []
    for root_name in roots:
        for path in (ROOT / root_name).rglob("*"):
            if path.is_dir() or path.suffix not in suffixes or "__pycache__" in path.parts:
                continue
            if path.name == "test_repository_layout.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert offenders == []


def test_print_pipeline_components(capsys):
    pipeline = HarmfulMemePipeline().eval()
    print_pipeline_components(pipeline)
    captured = capsys.readouterr()
    assert "Pipeline components:" in captured.out
    assert "module.internal_evidence_extractor.InternalEvidenceExtractor" in captured.out
