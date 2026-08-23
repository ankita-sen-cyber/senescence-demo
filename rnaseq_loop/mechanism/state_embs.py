"""Compute per-state mean cell embeddings.

`InSilicoPerturber` needs a `state_embs_dict` mapping each label to a
`torch.tensor` representing that state's centroid embedding.

We compute state centroids on a held-out set of labeled cells using
`geneformer.EmbExtractor` (documented behavior) — for each cell we take
the CLS embedding at `emb_layer=-1` (2nd-to-last layer, per docs
recommendation for general representations) and mean-pool per class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)


def extract_state_embeddings(
    model_directory: str,
    labeled_dataset: str,
    state_key: str,
    states: Sequence[str],
    output_dir: str,
    output_prefix: str = "state_embs",
    model_version: str = "V2",
    max_ncells_per_state: int = 5_000,
    emb_layer: int = -1,
    forward_batch_size: int = 100,
    nproc: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Returns a dict {state: mean_embedding_tensor} in the shape ISP expects.
    Also writes each tensor to disk for reuse across loop iterations.
    """
    from geneformer import EmbExtractor

    out = ensure_dir(output_dir)
    log.info(f"Extracting embeddings from {model_directory}")

    ex = EmbExtractor(
        model_type="CellClassifier",
        num_classes=len(states),
        emb_mode="cls",
        cell_emb_style="mean_pool",
        filter_data={state_key: list(states)},
        max_ncells=max_ncells_per_state * len(states),
        emb_layer=emb_layer,
        emb_label=[state_key],
        labels_to_plot=[state_key],
        forward_batch_size=forward_batch_size,
        nproc=nproc,
        model_version=model_version,
    )

    emb_df: pd.DataFrame = ex.extract_embs(
        model_directory=model_directory,
        input_data_file=labeled_dataset,
        output_directory=str(out),
        output_prefix=output_prefix,
    )

    # `emb_df` columns: embedding dims + the state column we asked for.
    log.info(f"Extracted {len(emb_df):,} cell embeddings")

    state_embs: dict[str, torch.Tensor] = {}
    for state in states:
        sub = emb_df[emb_df[state_key] == state]
        if sub.empty:
            log.warning(f"No cells for state '{state}' — skipping")
            continue
        emb_only = sub.drop(columns=[state_key]).values.astype(np.float32)
        mean_emb = torch.tensor(emb_only.mean(axis=0))
        state_embs[state] = mean_emb
        torch.save(mean_emb, out / f"{output_prefix}_{state}.pt")
        log.info(f"  {state}: n={len(sub):,}  ‖μ‖={mean_emb.norm().item():.3f}")

    torch.save(state_embs, out / f"{output_prefix}_state_embs_dict.pt")
    return state_embs


def load_state_embs(path: str | Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=False)
