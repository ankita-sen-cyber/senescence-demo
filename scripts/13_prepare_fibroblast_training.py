"""Download and prepare balanced GSE226225 fibroblasts for Geneformer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.fibroblast_training import download_gse226225, prepare_gse226225


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fibroblast/GSE226225"))
    parser.add_argument("--archive", type=Path, default=Path("data/external/GSE226225/raw/GSE226225_RAW.tar"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    archive = download_gse226225(args.archive)
    output = args.output_dir / "GSE226225_fibroblast_train.h5ad"
    stats = prepare_gse226225(archive, output, seed=args.seed)
    manifest = {
        "accession": "GSE226225",
        "role": "fibroblast_training",
        "output_h5ad": str(output),
        **stats,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
