"""Export a compact manifest of available intermediate result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import print_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--output", default="result/intermediate_manifest.json")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(result_root.rglob("*"))
        if path.is_file()
    ]
    manifest = {"result_root": str(result_root), "files": files}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print_json(manifest)


if __name__ == "__main__":
    main()
