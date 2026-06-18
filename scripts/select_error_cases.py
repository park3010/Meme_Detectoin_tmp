"""Select representative TP/TN/FP/FN error-analysis cases."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.error_case_analysis import select_error_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model", default="ours_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()
    cases = select_error_cases(
        args.dataset,
        args.model,
        args.seed,
        args.result_root,
        disable_tqdm=args.disable_tqdm,
    )
    print(f"Selected {len(cases)} cases.")


if __name__ == "__main__":
    main()
