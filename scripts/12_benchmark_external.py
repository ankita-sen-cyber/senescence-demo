"""Score frozen Geneformer and classical baselines on an external test study."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.evaluation.generalization import (
    _continuous_score,
    _estimator,
    select_top_features,
    summarize_predictions,
)


def _normalize_log1p(X: sparse.spmatrix) -> sparse.csr_matrix:
    X = X.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(X.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Every cell must have a positive library size")
    X = sparse.diags(10_000.0 / totals) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def _labels(values: pd.Series) -> np.ndarray:
    unknown = sorted(set(values.astype(str)) - {"proliferating", "senescent"})
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return (values.astype(str) == "senescent").to_numpy(dtype=int)


def _load_aligned(
    train_path: Path, test_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ad.AnnData]:
    train = ad.read_h5ad(train_path)
    test = ad.read_h5ad(test_path)
    train_ids = pd.Index(train.var["ensembl_id"].astype(str))
    test_ids = pd.Index(test.var["ensembl_id"].astype(str))
    shared = train_ids.intersection(test_ids, sort=False)
    if len(shared) == 0:
        raise ValueError("Training and external datasets have no shared Ensembl genes")
    train_index = train_ids.get_indexer(shared)
    test_index = test_ids.get_indexer(shared)
    X_train = _normalize_log1p(train.X[:, train_index])
    X_test = _normalize_log1p(test.X[:, test_index])
    return X_train, _labels(train.obs["senescence_label"]), X_test, shared.to_numpy(), test


def _geneformer_rows(predictions_path: Path, test: ad.AnnData) -> pd.DataFrame:
    frame = pd.read_csv(predictions_path)
    required = {"cell_id", "predicted_label", "probability_senescent"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Geneformer predictions missing columns: {sorted(missing)}")
    truth = test.obs[["senescence_label", "sample_id", "culture_dimension"]].copy()
    truth.index.name = None
    truth["cell_id"] = test.obs_names.astype(str)
    merged = frame.merge(truth, on="cell_id", how="inner", validate="one_to_one", suffixes=("", "_truth"))
    if len(merged) != test.n_obs:
        raise ValueError(f"Matched {len(merged)} predictions for {test.n_obs} test cells")
    true_column = "senescence_label_truth" if "senescence_label_truth" in merged else "senescence_label"
    return pd.DataFrame({
        "sample": merged["cell_id"],
        "held_out_group": merged["sample_id_truth"] if "sample_id_truth" in merged else merged["sample_id"],
        "culture_dimension": merged["culture_dimension_truth"] if "culture_dimension_truth" in merged else merged["culture_dimension"],
        "method": "geneformer_v1",
        "label": (merged[true_column] == "senescent").astype(int),
        "prediction": (merged["predicted_label"] == "senescent").astype(int),
        "score": merged["probability_senescent"].astype(float),
        "selected_features": "Geneformer rank-value tokens",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5ad", type=Path, default=Path("data/senescence/census_slice.h5ad"))
    parser.add_argument("--test-h5ad", type=Path, default=Path("data/external/GSE282425/GSE282425_external.h5ad"))
    parser.add_argument("--geneformer-predictions", type=Path, default=Path("outputs/senescence/predictions/GSE282425_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/GSE282425"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_test, genes, test = _load_aligned(args.train_h5ad, args.test_h5ad)
    y_test = _labels(test.obs["senescence_label"])
    selected = select_top_features(X_train.toarray(), y_train, args.top_n)
    rows = [_geneformer_rows(args.geneformer_predictions, test)]
    methods = ("dummy_majority", "nearest_centroid_top", "linear_svm_top", "logistic_top")
    for method in methods:
        use = np.array([selected[0]]) if method == "dummy_majority" else selected
        estimator = _estimator(method, args.seed)
        train_values = X_train[:, use].toarray()
        test_values = X_test[:, use].toarray()
        estimator.fit(train_values, y_train)
        prediction = np.asarray(estimator.predict(test_values), dtype=int)
        score = _continuous_score(estimator, test_values)
        rows.append(pd.DataFrame({
            "sample": test.obs_names.astype(str),
            "held_out_group": test.obs["sample_id"].astype(str).to_numpy(),
            "culture_dimension": test.obs["culture_dimension"].astype(str).to_numpy(),
            "method": method,
            "label": y_test,
            "prediction": prediction,
            "score": score,
            "selected_features": ";".join(genes[selected]),
        }))

    predictions = pd.concat(rows, ignore_index=True)
    summary = summarize_predictions(
        predictions, n_bootstrap=args.bootstrap, random_state=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "external_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "external_method_comparison.csv", index=False)
    per_sample = (
        predictions.assign(correct=lambda x: x.label == x.prediction)
        .groupby(["method", "held_out_group", "culture_dimension"], as_index=False)
        .agg(n_cells=("label", "size"), accuracy=("correct", "mean"),
             true_senescent_rate=("label", "mean"), predicted_senescent_rate=("prediction", "mean"),
             mean_senescence_score=("score", "mean"))
    )
    per_sample.to_csv(args.output_dir / "external_by_sample.csv", index=False)
    print(summary[["method", "n_samples", "balanced_accuracy", "auroc", "macro_f1", "mcc"]].to_string(index=False))
    print(f"\nShared genes: {len(genes):,}; top features fitted on training cells only: {len(selected)}")


if __name__ == "__main__":
    main()
