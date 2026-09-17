"""Leakage-safe raw-count pseudobulk training and external evaluation."""
from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rnaseq_loop.fibroblast_training import GSE226225_SAMPLES, _read_member


VALID_LABELS = {"proliferating", "senescent"}


def _validate_labels(values: pd.Series) -> None:
    unknown = sorted(set(values.astype(str)) - VALID_LABELS)
    if unknown:
        raise ValueError(f"Unknown senescence labels: {unknown}")


def aggregate_raw_counts(
    X: sparse.spmatrix | np.ndarray,
    obs: pd.DataFrame,
    *,
    sample_column: str = "sample_id",
    label_column: str = "senescence_label",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sum cell counts into one raw-count vector per biological sample."""
    required = {sample_column, label_column}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Missing observation columns: {sorted(missing)}")
    if X.shape[0] != len(obs):
        raise ValueError("Expression rows and observation rows do not match")
    _validate_labels(obs[label_column])
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Pseudobulk requires finite, non-negative raw counts")

    rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    sample_values = obs[sample_column].astype(str).to_numpy()
    for sample_id in sorted(pd.unique(sample_values)):
        indices = np.flatnonzero(sample_values == sample_id)
        labels = obs.iloc[indices][label_column].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(f"Sample {sample_id!r} contains multiple labels")
        total = np.asarray(X[indices].sum(axis=0)).ravel()
        rows.append(total)
        records.append({
            "sample_id": sample_id,
            "true_label": labels[0],
            "n_cells": int(len(indices)),
            "library_size": int(total.sum()),
        })
    return np.vstack(rows), pd.DataFrame(records)


def aggregate_gse226225_archive(
    archive: str | Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Aggregate all cells in the selected GSE226225 endpoint samples."""
    rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    reference_features: pd.DataFrame | None = None

    with tarfile.open(archive, mode="r") as table:
        for prefix, (label, induction, _target_cells) in GSE226225_SAMPLES.items():
            features = _read_member(
                table,
                f"{prefix}_features.tsv.gz",
                sep="\t",
                header=None,
                names=["ensembl_id", "gene_symbol", "feature_type"],
            )
            if reference_features is None:
                reference_features = features
            elif not features.equals(reference_features):
                raise ValueError(f"Feature ordering differs for {prefix}")
            matrix_member = table.extractfile(f"{prefix}_matrix.mtx.gz")
            if matrix_member is None:
                raise FileNotFoundError(f"Matrix missing for {prefix}")
            with gzip.GzipFile(fileobj=matrix_member) as stream:
                counts = mmread(stream).tocsr().T
            total = np.asarray(counts.sum(axis=0)).ravel()
            rows.append(total)
            records.append({
                "sample_id": prefix.split("_", 1)[0],
                "true_label": label,
                "induction": induction,
                "n_cells": int(counts.shape[0]),
                "library_size": int(total.sum()),
            })

    if reference_features is None:
        raise ValueError("No GSE226225 samples were found")
    return np.vstack(rows), pd.DataFrame(records), reference_features


def log_cpm(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return counts per million and log(1 + CPM) matrices."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("counts must be a finite, non-negative 2-D matrix")
    library_sizes = counts.sum(axis=1)
    if np.any(library_sizes <= 0):
        raise ValueError("Every pseudobulk sample must have a positive library size")
    cpm = counts / library_sizes[:, None] * 1_000_000.0
    return cpm, np.log1p(cpm)


def run_external_pseudobulk(
    train_counts: np.ndarray,
    train_meta: pd.DataFrame,
    train_features: pd.DataFrame,
    test_counts: np.ndarray,
    test_meta: pd.DataFrame,
    test_feature_ids: pd.Index,
    *,
    top_n: int = 100,
    min_cpm: float = 1.0,
    min_training_samples: int = 2,
    regularization_c: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit only on training pseudobulks and predict untouched test samples."""
    if top_n <= 0 or min_cpm < 0 or min_training_samples <= 0:
        raise ValueError("Invalid feature-filtering parameters")
    _validate_labels(train_meta["true_label"])
    _validate_labels(test_meta["true_label"])

    train_ids = pd.Index(train_features["ensembl_id"].astype(str))
    test_ids = pd.Index(test_feature_ids.astype(str))
    if not train_ids.is_unique or not test_ids.is_unique:
        raise ValueError("Gene identifiers must be unique")
    shared = train_ids.intersection(test_ids, sort=False)
    if shared.empty:
        raise ValueError("Training and test datasets have no shared genes")
    train_aligned = train_counts[:, train_ids.get_indexer(shared)]
    test_aligned = test_counts[:, test_ids.get_indexer(shared)]

    train_cpm, X_train = log_cpm(train_aligned)
    _, X_test = log_cpm(test_aligned)
    expressed = (train_cpm >= min_cpm).sum(axis=0) >= min_training_samples
    if not expressed.any():
        raise ValueError("No genes pass the training-only expression filter")
    X_train = X_train[:, expressed]
    X_test = X_test[:, expressed]
    filtered_genes = shared.to_numpy()[expressed]

    y_train = (train_meta["true_label"].astype(str) == "senescent").to_numpy(dtype=int)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training pseudobulks must contain both classes")
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = stats.ttest_ind(
            X_train[y_train == 1],
            X_train[y_train == 0],
            axis=0,
            equal_var=False,
            nan_policy="omit",
        ).statistic
    scores = np.nan_to_num(np.abs(statistic), nan=0.0, posinf=np.finfo(float).max)
    selected = np.argsort(-scores, kind="stable")[: min(top_n, len(scores))]

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        ),
    )
    classifier.fit(X_train[:, selected], y_train)
    probability = classifier.predict_proba(X_test[:, selected])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    y_test = (test_meta["true_label"].astype(str) == "senescent").to_numpy(dtype=int)

    predictions = test_meta.copy()
    predictions["probability_senescent"] = probability
    predictions["predicted_label"] = np.where(
        prediction == 1, "senescent", "proliferating"
    )
    predictions["correct"] = prediction == y_test
    confusion = confusion_matrix(y_test, prediction, labels=[0, 1])
    metrics: dict[str, object] = {
        "n_training_samples": int(len(train_meta)),
        "n_test_samples": int(len(test_meta)),
        "n_shared_genes": int(len(shared)),
        "n_genes_after_training_filter": int(expressed.sum()),
        "n_selected_genes": int(len(selected)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "auroc": float(roc_auc_score(y_test, probability)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
        "confusion_matrix": {
            "true_proliferating_pred_proliferating": int(confusion[0, 0]),
            "true_proliferating_pred_senescent": int(confusion[0, 1]),
            "true_senescent_pred_proliferating": int(confusion[1, 0]),
            "true_senescent_pred_senescent": int(confusion[1, 1]),
        },
        "decision_threshold": 0.5,
        "threshold_tuned_on_external_test": False,
    }

    symbol_by_id = (
        train_features.assign(ensembl_id=train_features["ensembl_id"].astype(str))
        .drop_duplicates("ensembl_id")
        .set_index("ensembl_id")["gene_symbol"]
    )
    selected_features = pd.DataFrame({
        "rank": np.arange(1, len(selected) + 1),
        "ensembl_id": filtered_genes[selected],
        "gene_symbol": [symbol_by_id.get(gene, "") for gene in filtered_genes[selected]],
        "absolute_welch_t": scores[selected],
    })
    return predictions, metrics, selected_features
