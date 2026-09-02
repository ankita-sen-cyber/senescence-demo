"""Run slice-based evaluation + counterfactual probing + active-learning selection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.active import acquire
from rnaseq_loop.failure import (
    bootstrap_topk_stability,
    counterfactual_probe,
    save_report,
    slice_metrics,
)
from rnaseq_loop.mechanism.prep import resolve_predictions_pkl
from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger("failure_active")


def _softmax_pos(logits) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        return arr
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp[:, 1] / exp.sum(axis=1)


def _load_predictions(path: Path) -> pd.DataFrame:
    """Load the per-cell predictions saved by Classifier.validate(predict_eval=True).

    Geneformer's `*_pred_dict.pkl` is a dict of numpy arrays (`pred_ids`,
    `label_ids`, `predictions` logits). Older docs called this `predictions.pkl`.
    """
    import pickle

    path = resolve_predictions_pkl(path)
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        n = None
        for key in ("label_ids", "pred_ids", "labels", "predictions"):
            if key in data:
                n = len(data[key])
                break
        cols = {}
        for k, v in data.items():
            arr = np.asarray(v)
            if n is not None and arr.ndim == 2 and arr.shape[0] == n:
                cols[k] = list(arr)
            else:
                cols[k] = v
        df = pd.DataFrame(cols)

    df = df.rename(columns={
        "labels": "label",
        "label_ids": "label",
        "pred_ids": "pred",
        "probabilities": "prob",
    })
    if "score" not in df.columns:
        if "prob" in df.columns:
            probs = np.asarray(df["prob"].tolist())
            df["score"] = probs[:, 1] if probs.ndim == 2 else probs
        elif "predictions" in df.columns:
            df["score"] = _softmax_pos(df["predictions"].tolist())
    if "pred" not in df.columns and "score" in df.columns:
        df["pred"] = (df["score"] >= 0.5).astype(int)
    if "label" not in df.columns:
        raise KeyError(f"Predictions pickle missing labels; columns={list(df.columns)}")
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


def _maybe_acquire(acq: dict) -> None:
    required = ["pool_embeddings_npy"]
    missing = [k for k in required if not Path(acq[k]).exists()]
    if missing:
        log.warning(
            "Skipping active-learning selection; missing "
            + ", ".join(f"{k}={acq[k]}" for k in missing)
            + ". Write those arrays after embedding the unlabeled pool."
        )
        return

    pool_emb = np.load(acq["pool_embeddings_npy"])
    lab_path = Path(acq["labeled_embeddings_npy"])
    lab_emb = np.load(lab_path) if lab_path.exists() else np.empty((0, pool_emb.shape[1]))
    ens_path = Path(acq.get("ensemble_probs_npy") or "")
    ens = np.load(ens_path) if ens_path.exists() else None
    pred_tf_path = Path(acq.get("predicted_tf_activity_npy") or "")
    pred_tf = np.load(pred_tf_path) if pred_tf_path.exists() else None
    prior_tf_path = Path(acq.get("prior_tf_activity_npy") or "")
    prior_tf = np.load(prior_tf_path) if prior_tf_path.exists() else None
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
        _maybe_acquire(acq)


if __name__ == "__main__":
    main()
