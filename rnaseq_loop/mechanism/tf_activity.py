"""Upstream regulator inference via decoupler + CollecTRI.

Given a ranked gene list from ISP (or classic differential expression), we
infer transcription-factor activities and pathway activities. This is the
biologist-readable mechanism layer.

Reference: https://decoupler-py.readthedocs.io/
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)


def _import_decoupler():
    import decoupler as dc

    return dc


def get_collectri(organism: str = "human") -> pd.DataFrame:
    """Fetch the CollecTRI TF-target regulon network (curated by decoupler team)."""
    dc = _import_decoupler()
    if hasattr(dc, "get_collectri"):
        net = dc.get_collectri(organism=organism, split_complexes=False)
    else:
        net = dc.op.collectri(organism=organism, remove_complexes=False)
    log.info(f"CollecTRI: {len(net):,} interactions, "
             f"{net['source'].nunique():,} TFs")
    return net


def get_progeny(organism: str = "human", top: int = 500) -> pd.DataFrame:
    """Fetch the PROGENy pathway-footprint gene sets."""
    dc = _import_decoupler()
    if hasattr(dc, "get_progeny"):
        net = dc.get_progeny(organism=organism, top=top)
    else:
        net = dc.op.progeny(organism=organism, top=top)
    log.info(f"PROGENy: {net['source'].nunique()} pathways, {len(net):,} weights")
    return net


def _run_linear_model(mat: pd.DataFrame, net: pd.DataFrame, method: str, min_n: int):
    """Run MLM/ULM on both decoupler 1.x (`run_*`) and 2.x (`mt.*`) APIs."""
    dc = _import_decoupler()
    method = method.lower()
    old_name = {"mlm": "run_mlm", "ulm": "run_ulm", "wsum": "run_wsum"}.get(method)
    if old_name and hasattr(dc, old_name):
        return getattr(dc, old_name)(
            mat=mat,
            net=net,
            source="source",
            target="target",
            weight="weight",
            min_n=min_n,
            verbose=False,
        )
    if not hasattr(dc, "mt") or not hasattr(dc.mt, method):
        raise AttributeError(f"decoupler has no method {method!r}")
    return getattr(dc.mt, method)(mat, net, tmin=min_n, verbose=False)


def tf_activity_from_ranked_genes(
    gene_scores: pd.Series,
    net: pd.DataFrame | None = None,
    method: Literal["mlm", "ulm", "wsum"] = "mlm",
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Score TF activities from a Series of gene → attribution_score.

    `gene_scores.index` must be gene symbols (or Ensembl IDs matching the
    network's `target` column). CollecTRI's default is gene symbols.
    """
    if net is None:
        net = get_collectri()

    # decoupler wants a matrix-shaped input; wrap the series as a 1-row DataFrame.
    mat = gene_scores.to_frame().T.rename(index={0: "phenotype"})
    if mat.index[0] != "phenotype":
        mat.index = ["phenotype"]

    est, pval = _run_linear_model(mat, net, method=method, min_n=min_n)

    out = pd.DataFrame({
        "tf": est.columns,
        "activity": est.iloc[0].values,
        "pval": pval.iloc[0].values,
    })
    out["signed_neglog10p"] = -np.log10(out["pval"].clip(lower=1e-300)) * np.sign(out["activity"])
    return out.sort_values("signed_neglog10p", ascending=False, key=lambda s: s.abs())


def pathway_activity_from_ranked_genes(
    gene_scores: pd.Series,
    net: pd.DataFrame | None = None,
    min_n: int = 5,
) -> pd.DataFrame:
    """PROGENy pathway activity."""
    if net is None:
        net = get_progeny()

    mat = gene_scores.to_frame().T.rename(index={0: "phenotype"})
    est, pval = _run_linear_model(mat, net, method="mlm", min_n=min_n)
    out = pd.DataFrame({
        "pathway": est.columns,
        "activity": est.iloc[0].values,
        "pval": pval.iloc[0].values,
    })
    return out.sort_values("activity", ascending=False, key=lambda s: s.abs())


def gsea_from_ranked_genes(
    gene_scores: pd.Series,
    gene_sets: str = "MSigDB_Hallmark_2020",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Preranked GSEA via gseapy (fgsea equivalent in Python)."""
    import gseapy as gp

    ranked = gene_scores.sort_values(ascending=False)
    ranked.index = ranked.index.astype(str).str.upper()

    outdir = str(ensure_dir(output_dir)) if output_dir else None
    pre_res = gp.prerank(
        rnk=ranked.reset_index().rename(columns={"index": "gene", 0: "score",
                                                 ranked.name or 0: "score"}),
        gene_sets=gene_sets,
        min_size=15,
        max_size=500,
        permutation_num=1000,
        outdir=outdir,
        seed=0,
        verbose=False,
    )
    return pre_res.res2d
