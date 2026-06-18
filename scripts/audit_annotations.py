"""Audit LLM-generated silver annotations and normalized label coverage."""

from __future__ import annotations

import argparse

from common import ROOT

from experiments.annotation_audit import format_audit_summary, run_annotation_audit
from experiments.annotation_normalization import load_normalization_config, resolve_dataset_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "annotation_normalization.yaml"))
    parser.add_argument("--dataset", default="all", help="all, harm_c, harm_p, facebook, or memotion")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--annotation-root", default=None)
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples-per-flag", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    cfg = load_normalization_config(args.config)
    paths = cfg.get("paths", {})
    dataset_names = resolve_dataset_names(args.dataset, cfg)
    result = run_annotation_audit(
        dataset_names=dataset_names,
        dataset_root=args.dataset_root or paths.get("dataset_root", "dataset/source"),
        annotation_root=args.annotation_root or paths.get("annotation_root", "dataset/annotation"),
        config=cfg,
        audit_root=args.audit_root or paths.get("audit_root", "result/annotation_audit"),
        limit=args.limit,
        max_examples_per_flag=args.max_examples_per_flag,
        disable_tqdm=args.disable_tqdm,
    )
    print(format_audit_summary(result))


if __name__ == "__main__":
    main()
