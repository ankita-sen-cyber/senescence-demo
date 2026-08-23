"""Active learning acquisition functions.

We support three acquisition strategies over a pool of candidate perturbations
(each represented by an expected transcriptomic response vector):

1. BALD (Bayesian Active Learning by Disagreement) — pick perturbations where
   the model ensemble disagrees most about the predicted post-perturbation state.
2. Expected TF-prior disagreement — pick perturbations whose predicted TF
   activity pattern most disagrees with prior CollecTRI-based expectations,
   surfacing novel mechanism.
3. Coverage — pick perturbations that maximize embedding-space coverage of the
   unlabeled pool (facility location / k-medoids).

All acquisition functions return an ordered list of pool indices.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bald(
    ensemble_probs: np.ndarray,
) -> np.ndarray:
    """
    Args:
      ensemble_probs: shape (n_models, n_pool, n_classes). Softmax probs from
                      each ensemble member (or MC-dropout samples) for each
                      pool item.
    Returns:
      Per-pool-item BALD score = H(mean p) − E[H(p)]. Higher = more informative.
    """
    eps = 1e-12
    mean_p = ensemble_probs.mean(axis=0)                # (n_pool, n_classes)
    entropy_mean = -(mean_p * np.log(mean_p + eps)).sum(axis=1)  # (n_pool,)
    entropy_per_model = -(ensemble_probs * np.log(ensemble_probs + eps)).sum(axis=2)
    mean_entropy = entropy_per_model.mean(axis=0)       # (n_pool,)
    return entropy_mean - mean_entropy                  # mutual information


def tf_prior_disagreement(
    predicted_tf_activity: np.ndarray,     # (n_pool, n_tfs)
    prior_tf_activity: np.ndarray,          # (n_pool, n_tfs) from CollecTRI target lists
) -> np.ndarray:
    """L2 disagreement between model-predicted and prior-expected TF activities."""
    return np.linalg.norm(predicted_tf_activity - prior_tf_activity, axis=1)


def coverage_scores(
    pool_embeddings: np.ndarray,       # (n_pool, d)
    labeled_embeddings: np.ndarray,    # (n_labeled, d)
) -> np.ndarray:
    """
    Facility-location-style coverage score: distance to the nearest already-labeled
    point. Higher score = further from labeled set = better for coverage.
    """
    if labeled_embeddings.shape[0] == 0:
        # Nothing labeled yet → uniform score.
        return np.ones(pool_embeddings.shape[0])
    # Chunked to avoid the full n_pool × n_labeled matrix.
    n_pool = pool_embeddings.shape[0]
    out = np.empty(n_pool, dtype=np.float32)
    chunk = 1024
    for start in range(0, n_pool, chunk):
        end = min(start + chunk, n_pool)
        block = pool_embeddings[start:end]           # (b, d)
        d2 = ((block[:, None, :] - labeled_embeddings[None, :, :]) ** 2).sum(-1)
        out[start:end] = np.sqrt(d2.min(axis=1))
    return out


def combine_acquisitions(
    scores: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Z-score each acquisition, then weighted sum."""
    stacked = []
    for name, s in scores.items():
        z = (s - s.mean()) / (s.std() + 1e-9)
        stacked.append(weights.get(name, 0.0) * z)
    return np.stack(stacked, axis=0).sum(axis=0)


def select_top_k(
    combined: np.ndarray,
    k: int,
    already_labeled: set[int] | None = None,
) -> list[int]:
    """Return the top-k pool indices, excluding already-labeled ones."""
    if already_labeled is None:
        already_labeled = set()
    order = np.argsort(-combined)
    picked: list[int] = []
    for i in order:
        if int(i) in already_labeled:
            continue
        picked.append(int(i))
        if len(picked) >= k:
            break
    return picked


def acquire(
    ensemble_probs: np.ndarray | None,
    pool_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    predicted_tf_activity: np.ndarray | None = None,
    prior_tf_activity: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
    k: int = 20,
    already_labeled: set[int] | None = None,
) -> list[int]:
    """High-level entry point. Any acquisition can be disabled by omitting its inputs."""
    if weights is None:
        weights = {"bald": 0.5, "tf_disagreement": 0.3, "coverage": 0.2}
    scores: dict[str, np.ndarray] = {}
    if ensemble_probs is not None:
        scores["bald"] = bald(ensemble_probs)
    if predicted_tf_activity is not None and prior_tf_activity is not None:
        scores["tf_disagreement"] = tf_prior_disagreement(
            predicted_tf_activity, prior_tf_activity
        )
    scores["coverage"] = coverage_scores(pool_embeddings, labeled_embeddings)
    combined = combine_acquisitions(scores, weights)
    return select_top_k(combined, k=k, already_labeled=already_labeled)
