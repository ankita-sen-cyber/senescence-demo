"""In silico perturbation with Geneformer's InSilicoPerturber.

We use the "delete" perturbation on every detected gene in each cell, and
measure the shift in cell embedding toward the goal state versus the start
state. The resulting per-gene "cosine_shift_toward_goal" (or "delta" in
newer builds) is our mechanism-attribution score.

API reference (verified):
  https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass
class ISPConfig:
    perturb_type: Literal["delete", "overexpress", "inhibit", "activate"] = "delete"
    genes_to_perturb: str | list[str] = "all"       # or list of Ensembl IDs
    combos: int = 0
    anchor_gene: str | None = None
    model_type: str = "CellClassifier"              # or "Pretrained", "MTLCellClassifier"
    num_classes: int = 2
    emb_mode: Literal["cls", "cell", "cls_and_gene", "cell_and_gene"] = "cls_and_gene"
    filter_data: dict | None = None
    cell_states_to_model: dict = field(default_factory=dict)   # {state_key, start_state, goal_state, alt_states}
    state_embs_dict_path: str = ""                  # path to torch.save'd dict
    max_ncells: int | None = 2_000
    emb_layer: int = -1
    forward_batch_size: int = 100
    nproc: int = 8
    model_version: str = "V2"
    clear_mem_ncells: int = 1000


def run_isp(
    cfg: ISPConfig,
    model_directory: str,
    input_data_file: str,
    output_directory: str,
    output_prefix: str,
) -> Path:
    """Run the perturber and return the output directory (batched pickle files)."""
    from geneformer import InSilicoPerturber

    out = ensure_dir(output_directory)
    state_embs = torch.load(cfg.state_embs_dict_path, weights_only=False)
    log.info(f"Loaded state_embs_dict with keys: {list(state_embs.keys())}")

    isp = InSilicoPerturber(
        perturb_type=cfg.perturb_type,
        genes_to_perturb=cfg.genes_to_perturb,
        combos=cfg.combos,
        anchor_gene=cfg.anchor_gene,
        model_type=cfg.model_type,
        num_classes=cfg.num_classes,
        emb_mode=cfg.emb_mode,
        filter_data=cfg.filter_data,
        cell_states_to_model=cfg.cell_states_to_model,
        state_embs_dict=state_embs,
        max_ncells=cfg.max_ncells,
        emb_layer=cfg.emb_layer,
        forward_batch_size=cfg.forward_batch_size,
        nproc=cfg.nproc,
        model_version=cfg.model_version,
        clear_mem_ncells=cfg.clear_mem_ncells,
    )

    log.info(f"Running ISP ({cfg.perturb_type}) on {input_data_file}")
    isp.perturb_data(
        model_directory=model_directory,
        input_data_file=input_data_file,
        output_directory=str(out),
        output_prefix=output_prefix,
    )
    log.info(f"ISP outputs in {out}")
    return out


def summarize_isp(
    isp_output_dir: str,
    stats_output_dir: str,
    output_prefix: str,
    cell_states_to_model: dict,
    genes_perturbed: str | list[str] = "all",
    model_version: str = "V2",
) -> Path:
    """
    Run InSilicoPerturberStats to aggregate the batched ISP outputs into
    a single per-gene ranking with p-values.
    """
    from geneformer import InSilicoPerturberStats

    stats_dir = ensure_dir(stats_output_dir)

    stats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed=genes_perturbed,
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states_to_model,
        model_version=model_version,
    )

    stats.get_stats(
        input_data_directory=isp_output_dir,
        null_dist_data_directory=None,
        output_directory=str(stats_dir),
        output_prefix=output_prefix,
    )
    log.info(f"ISP stats written to {stats_dir}")
    return stats_dir
