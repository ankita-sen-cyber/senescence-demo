"""Fuse per-gene attribution scores with prior sources into a final target ranking.

Design:
  - Every input is a Series indexed by (Ensembl ID or gene symbol).
  - We z-score each source column, then take a weighted sum.
  - Weights are configurable (default: 0.5 attribution, 0.25 OT, 0.25 DepMap).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rnaseq_loop.utils import get_logger

log = get_logger(__name__)


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "isp_attribution": 0.5,
        "ot_overall_score": 0.25,
        "depmap_frac_essential": 0.25,
    })
    invert: dict[str, bool] = field(default_factory=lambda: {
        # DepMap effect is more negative = more essential; frac_essential is already
        # positive-oriented so no inversion needed for that column.
    })
    min_sources: int = 2  # a gene must have signal from at least this many sources


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (s - s.mean()) / (s.std() + 1e-9)


def rank_targets(
    sources: dict[str, pd.Series],
    cfg: ScoringConfig,
    key: str = "symbol",
) -> pd.DataFrame:
    """
    Args:
      sources: {name: Series indexed by gene identifier} — keys should be a
               subset of `cfg.weights`.
      cfg: scoring config with weights.
      key: name of the gene identifier column in the output.
    """
    aligned = pd.DataFrame(sources)  # outer join on index
    aligned.index.name = key
    log.info(f"Fused {aligned.shape[1]} sources for {aligned.shape[0]} genes")

    # Apply per-source inversion.
    for col, flip in cfg.invert.items():
        if flip and col in aligned.columns:
            aligned[col] = -aligned[col]

    z = aligned.apply(zscore, axis=0)
    # Weighted mean over available sources per row.
    w = pd.Series({k: v for k, v in cfg.weights.items() if k in z.columns})
    if w.empty:
        raise ValueError("No configured source columns present in `sources`.")
    w = w / w.sum()

    weighted = (z * w).sum(axis=1)
    n_sources_present = z.notna().sum(axis=1)
    weighted[n_sources_present < cfg.min_sources] = np.nan

    out = aligned.copy()
    out["target_score"] = weighted
    out["n_sources"] = n_sources_present
    out = out.sort_values("target_score", ascending=False)
    return out
