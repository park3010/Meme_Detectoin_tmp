"""Export paper-ready CSV tables from result metrics."""

from __future__ import annotations

import argparse

from common import ROOT  # noqa: F401

from experiments.paper_tables import export_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    paths = export_paper_tables(args.result_root, args.output_root)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
