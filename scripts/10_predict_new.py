"""Run a trained Geneformer classifier on a new unlabeled dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.deployment import prepare_new_anndata, predict_tokenized
from rnaseq_loop.utils import get_logger

log = get_logger("predict-new")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-h5ad", type=Path, help="Raw-count AnnData to validate and tokenize")
    source.add_argument("--tokenized-dataset", type=Path, help="Existing Geneformer .dataset directory")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--label-map", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=None, help="Tokenization work directory for --input-h5ad")
    parser.add_argument("--prefix", default="unseen")
    parser.add_argument("--metadata", nargs="*", default=["cell_id", "dataset_id", "donor_id", "cell_type"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-version", default="V2", choices=["V1", "V2"])
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()

    tokenized = args.tokenized_dataset
    metadata = list(args.metadata)
    if args.input_h5ad:
        workdir = args.workdir or args.output_csv.parent / f"{args.output_csv.stem}_tokenized"
        tokenized = prepare_new_anndata(
            args.input_h5ad, workdir, args.prefix, metadata,
            model_version=args.model_version, nproc=args.nproc,
        )
        if "cell_id" not in metadata:
            metadata.append("cell_id")
    label_map = args.label_map or args.model_dir / "id_class_dict.json"
    predictions = predict_tokenized(
        args.model_dir, tokenized, label_map,
        metadata_columns=metadata,
        batch_size=args.batch_size,
        device=args.device,
        model_version=args.model_version,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    log.info(f"Wrote {len(predictions):,} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
