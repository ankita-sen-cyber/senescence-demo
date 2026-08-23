"""Slice-based evaluation and bootstrap stability for the closed loop.

For every iteration we report AUROC / macro-F1 stratified by:
  - dataset_id  (study-level batch check)
  - tissue      (biological slice)
  - assay       (technology confound)
  - sex, donor age band (demographic confounds)

Any slice ≥10pp worse than overall is a failure signal to feed back into the loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from rnaseq_loop.utils import get_logger, save_json

log = get_logger(__name__)


def slice_metrics(
    predictions: pd.DataFrame,
    label_col: str = "label",
    pred_col: str = "pred",
    score_col: str = "score",
    slice_cols: Iterable[str] = ("dataset_id", "tissue", "assay", "sex"),
    min_slice_size: int = 50,
) -> pd.DataFrame:
    """
    Args:
      predictions: DataFrame with per-cell true label, hard prediction, and
                   probability score for the positive class, plus slice cols.
    Returns:
      DataFrame with one row per (slice_col, slice_value) reporting n, AUROC,
      macro-F1, and the gap versus overall.
    """
    rows: list[dict] = []
    overall_auc = roc_auc_score(predictions[label_col], predictions[score_col])
    overall_f1 = f1_score(predictions[label_col], predictions[pred_col], average="macro")
    rows.append({
        "slice_col": "OVERALL", "slice_value": "OVERALL",
        "n": len(predictions), "auroc": overall_auc, "macro_f1": overall_f1,
        "auroc_gap": 0.0, "f1_gap": 0.0,
    })

    for col in slice_cols:
        if col not in predictions.columns:
            continue
        for val, sub in predictions.groupby(col, observed=True):
            if len(sub) < min_slice_size or sub[label_col].nunique() < 2:
                continue
            try:
                auc = roc_auc_score(sub[label_col], sub[score_col])
            except ValueError:
                auc = np.nan
            f1 = f1_score(sub[label_col], sub[pred_col], average="macro")
            rows.append({
                "slice_col": col, "slice_value": str(val),
                "n": len(sub), "auroc": auc, "macro_f1": f1,
                "auroc_gap": auc - overall_auc,
                "f1_gap": f1 - overall_f1,
            })

    df = pd.DataFrame(rows).sort_values("auroc_gap")
    return df


def flag_failing_slices(
    slice_df: pd.DataFrame,
    gap_threshold: float = -0.10,
) -> pd.DataFrame:
    """Return only slices where AUROC is >= gap_threshold pp below overall."""
    return slice_df[
        (slice_df["slice_col"] != "OVERALL") & (slice_df["auroc_gap"] <= gap_threshold)
    ].copy()


def bootstrap_topk_stability(
    per_gene_scores_per_bootstrap: list[pd.Series],
    k: int = 100,
) -> pd.DataFrame:
    """
    For each gene, the fraction of bootstraps in which it appears in the top-k.
    per_gene_scores_per_bootstrap[i] is a Series indexed by gene.
    """
    top_sets: list[set[str]] = []
    for s in per_gene_scores_per_bootstrap:
        top_sets.append(set(s.nlargest(k).index))
    all_genes = set().union(*top_sets)
    n_boot = len(top_sets)
    freqs = {
        g: sum(g in ts for ts in top_sets) / n_boot
        for g in all_genes
    }
    return (
        pd.DataFrame({"gene": list(freqs), "topk_freq": list(freqs.values())})
        .sort_values("topk_freq", ascending=False)
        .reset_index(drop=True)
    )


def counterfactual_probe(
    isp_stats: pd.DataFrame,
    top_k: int = 50,
    reversal_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    For each of the top-k genes by attribution, check whether the model's
    predicted state actually shifts toward the goal upon in-silico deletion.

    `isp_stats` is expected to have columns
        `gene`, `attribution`, `goal_state_shift`, `pval`.
    A gene passes the probe iff `goal_state_shift > reversal_threshold` and pval < .05.
    """
    top = isp_stats.nlargest(top_k, "attribution").copy()
    top["passes_counterfactual"] = (
        (top["goal_state_shift"] > reversal_threshold) & (top["pval"] < 0.05)
    )
    log.info(
        f"Counterfactual probe: {top['passes_counterfactual'].sum()}/{top_k} "
        f"top-attributed genes pass"
    )
    return top


def save_report(
    slice_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    payload = {
        "slice_metrics": slice_df.to_dict(orient="records"),
        "failing_slices": flag_failing_slices(slice_df).to_dict(orient="records"),
        "topk_stability_summary": stability_df.head(200).to_dict(orient="records"),
        "counterfactual_probe": counterfactual_df.to_dict(orient="records"),
    }
    save_json(payload, output_path)
    log.info(f"Wrote failure-analysis report → {output_path}")
