"""Run slice-based evaluation + counterfactual probing + active-learning selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rnaseq_loop.active import acquire
from rnaseq_loop.failure import (
    bootstrap_topk_stability,
    counterfactual_probe,
    save_report,
    slice_metrics,
)
from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger("failure_active")


def _load_predictions(path: Path) -> pd.DataFrame:
    """Load the per-cell predictions saved by Classifier.validate(predict_eval=True).

    Geneformer's `predictions.pkl` is a dict with numpy arrays. We convert to
    a DataFrame with columns: label, pred, score, plus any slice attrs
    forwarded through tokenization.
    """
    import pickle
    with path.open("rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    # Expected keys vary by Geneformer version; harmonize.
    rename = {"labels": "label", "predictions": "pred", "probabilities": "prob"}
    df = df.rename(columns=rename)
    if "score" not in df.columns and "prob" in df.columns:
        # Binary case: take positive-class probability.
        probs = np.asarray(df["prob"].tolist())
        df["score"] = probs[:, 1] if probs.ndim == 2 else probs
    return df


def _load_isp_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = {"Gene_name": "gene", "Shift_to_goal_end": "attribution",
            "Goal_end_vs_random_pval": "pval", "Goal_end_vs_random_shift": "goal_state_shift"}
    df = df.rename(columns=keep)
    for c in ("attribution", "pval", "goal_state_shift"):
        if c not in df.columns:
            df[c] = np.nan
    return df[["gene", "attribution", "goal_state_shift", "pval"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-pkl", required=True, type=Path,
                   help="Geneformer's per-cell predictions from validate(predict_eval=True)")
    p.add_argument("--isp-stats-csv", required=True, type=Path)
    p.add_argument("--bootstrap-csvs", nargs="*", type=Path, default=[],
                   help="Optional: per-bootstrap ISP stats for stability")
    p.add_argument("--output-report", required=True, type=Path)
    p.add_argument("--acquisition-config", type=Path, default=None,
                   help="If provided, also run active-learning selection")
    args = p.parse_args()

    preds = _load_predictions(args.predictions_pkl)
    log.info(f"Predictions loaded: {preds.shape}")

    slices = slice_metrics(preds)
    log.info(f"Slice metrics:\n{slices.head(20)}")

    isp = _load_isp_stats(args.isp_stats_csv)
    cf = counterfactual_probe(isp, top_k=50)

    per_boot = []
    for path in args.bootstrap_csvs:
        b = _load_isp_stats(path)
        per_boot.append(b.set_index("gene")["attribution"])
    if per_boot:
        stab = bootstrap_topk_stability(per_boot, k=100)
    else:
        stab = pd.DataFrame(columns=["gene", "topk_freq"])

    save_report(slices, stab, cf, args.output_report)

    if args.acquisition_config is not None:
        acq = yaml.safe_load(args.acquisition_config.read_text())
        pool_emb = np.load(acq["pool_embeddings_npy"])
        lab_emb = np.load(acq["labeled_embeddings_npy"]) if Path(acq["labeled_embeddings_npy"]).exists() else np.empty((0, pool_emb.shape[1]))
        ens = np.load(acq["ensemble_probs_npy"]) if Path(acq["ensemble_probs_npy"]).exists() else None
        pred_tf = np.load(acq["predicted_tf_activity_npy"]) if Path(acq["predicted_tf_activity_npy"]).exists() else None
        prior_tf = np.load(acq["prior_tf_activity_npy"]) if Path(acq["prior_tf_activity_npy"]).exists() else None
        already_txt = Path(acq["already_labeled_ids_txt"])
        already = set(int(x) for x in already_txt.read_text().split()) if already_txt.exists() else set()

        picks = acquire(
            ensemble_probs=ens,
            pool_embeddings=pool_emb,
            labeled_embeddings=lab_emb,
            predicted_tf_activity=pred_tf,
            prior_tf_activity=prior_tf,
            weights=acq["acquisition"]["weights"],
            k=acq["acquisition"]["k"],
            already_labeled=already,
        )
        out = Path(acq["output_picks_txt"])
        ensure_dir(out.parent)
        out.write_text("\n".join(str(i) for i in picks))
        log.info(f"Wrote {len(picks)} next-experiment picks → {out}")


if __name__ == "__main__":
    main()
