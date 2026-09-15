"""Compare classifiers on cell lines excluded completely from training.

This evaluates within-study domain generalization. It is stronger than a random
sample split, but it is not a substitute for validation on an independent study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rnaseq_loop.evaluation import DEFAULT_METHODS, evaluate_group_holdout, summarize_predictions


DEFAULT_DATA = Path("data/GSE63577_counts_rpkm_exvivo_jenage_data.xls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    return parser.parse_args()


def sample_label(column: str) -> tuple[str, str]:
    if column.startswith("IMR90"):
        cell_line = "IMR90"
    elif column.startswith("MRC_5"):
        cell_line = "MRC5"
    elif column.startswith("WI_"):
        cell_line = "WI38"
    else:
        cell_line = column.split("_")[0]
    condition = "young" if ("_Y" in column or "PD16" in column or "PD32" in column) else "senescent"
    return cell_line, condition


def load_expression(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Data not found: {path}\nRun `python scripts/download_data.py` first.")
    raw = pd.read_excel(path, engine="xlrd")
    metadata = {"ensembl_gene_id", "external_gene_id", "description", "gene_biotype"}
    sample_columns = [column for column in raw.columns if column not in metadata]
    labels = [sample_label(column) for column in sample_columns]

    X = raw[sample_columns].to_numpy(dtype=np.float64).T
    genes = raw["external_gene_id"].astype(str).to_numpy()
    unique = ~pd.Series(genes).duplicated(keep="first").to_numpy()
    X, genes = X[:, unique], genes[unique]
    expressed = (X > 0).sum(axis=0) >= 3
    X, genes = X[:, expressed], genes[expressed]
    library_size = X.sum(axis=1, keepdims=True)
    if np.any(library_size <= 0):
        raise ValueError("at least one sample has zero total expression")
    X = np.log1p(X / library_size * 10_000.0)

    groups = np.asarray([cell_line for cell_line, _ in labels])
    y = np.asarray([condition == "senescent" for _, condition in labels], dtype=int)
    return X, y, groups, genes, np.asarray(sample_columns)


def main() -> None:
    args = parse_args()
    X, y, groups, genes, samples = load_expression(args.data)
    predictions, folds = evaluate_group_holdout(
        X, y, groups,
        feature_names=genes,
        sample_names=samples,
        top_n=args.top_n,
        methods=args.methods,
        random_state=args.seed,
    )
    summary = summarize_predictions(
        predictions, n_bootstrap=args.bootstrap, random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "generalization_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "generalization_by_group.csv", index=False)
    summary.to_csv(args.output_dir / "generalization_method_comparison.csv", index=False)
    # Preserve the historical single-method schema for existing consumers.
    compatibility = (
        folds.loc[folds["method"] == "logistic_top",
                  ["held_out_group", "n_train", "n_test", "accuracy"]]
        .rename(columns={"held_out_group": "held_out_cell_line"})
    )
    compatibility.to_csv(args.output_dir / "generalization_leave_one_out.csv", index=False)

    shown = summary[[
        "method", "balanced_accuracy", "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high", "auroc", "macro_f1", "mcc",
    ]].copy()
    print("Leave-one-cell-line-out comparison (all predictions are out of fold):\n")
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n95% intervals use a cell-line cluster bootstrap.")
    print("This tests unseen cell lines within GSE63577, not an independent external study.")


if __name__ == "__main__":
    main()
