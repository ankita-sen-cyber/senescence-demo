"""Download and prepare labelled GSE282425 as an independent test dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.external_data import convert_gse282425, download_gse282425


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/GSE282425"))
    parser.add_argument(
        "--gene-name-map", type=Path,
        default=Path("Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"),
        help="Geneformer V2 gene-symbol to Ensembl mapping",
    )
    args = parser.parse_args()
    paths = download_gse282425(args.output_dir / "raw")
    output = args.output_dir / "GSE282425_external.h5ad"
    stats = convert_gse282425(args.output_dir / "raw", output, args.gene_name_map)
    manifest = {
        "accession": "GSE282425",
        "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282425",
        "role": "independent_test_only",
        "output_h5ad": str(output),
        "files": {name: str(path) for name, path in paths.items()},
        **stats,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
