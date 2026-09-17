"""Train on GSE226225 raw-count pseudobulks and test on GSE282425."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.pseudobulk import (
    aggregate_gse226225_archive,
    aggregate_raw_counts,
    run_external_pseudobulk,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-archive",
        type=Path,
        default=Path("data/external/GSE226225/raw/GSE226225_RAW.tar"),
    )
    parser.add_argument(
        "--test-h5ad",
        type=Path,
        default=Path("data/external/GSE282425/GSE282425_external.h5ad"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external/GSE282425"),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-cpm", type=float, default=1.0)
    parser.add_argument("--min-training-samples", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_counts, train_meta, train_features = aggregate_gse226225_archive(
        args.train_archive
    )
    test = ad.read_h5ad(args.test_h5ad)
    test_counts, test_meta = aggregate_raw_counts(test.X, test.obs)
    test_ids = (
        test.var["ensembl_id"].astype(str)
        if "ensembl_id" in test.var
        else test.var_names.astype(str)
    )
    predictions, metrics, selected = run_external_pseudobulk(
        train_counts,
        train_meta,
        train_features,
        test_counts,
        test_meta,
        test_ids,
        top_n=args.top_n,
        min_cpm=args.min_cpm,
        min_training_samples=args.min_training_samples,
        regularization_c=args.regularization_c,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "pseudobulk_predictions.csv", index=False)
    train_meta.to_csv(args.output_dir / "pseudobulk_training_samples.csv", index=False)
    selected.to_csv(args.output_dir / "pseudobulk_selected_features.csv", index=False)
    run = {
        "method": "raw-count sum by biological sample; log1p CPM; "
        "training-only expression filter and Welch-t feature selection; "
        "class-balanced L2 logistic regression",
        "training_study": "GSE226225",
        "external_test_study": "GSE282425",
        "top_n": args.top_n,
        "min_cpm": args.min_cpm,
        "min_training_samples": args.min_training_samples,
        "regularization_c": args.regularization_c,
        "seed": args.seed,
        **metrics,
    }
    (args.output_dir / "pseudobulk_metrics.json").write_text(
        json.dumps(run, indent=2) + "\n"
    )
    print(predictions.to_string(index=False))
    print("\n" + json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
