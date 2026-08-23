"""Tokenize an h5ad into a Geneformer .dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from rnaseq_loop.tokenize import tokenize_anndata
from rnaseq_loop.utils import get_logger

log = get_logger("tokenize")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="labeled .h5ad")
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--prefix", required=True)
    p.add_argument("--model-version", default="V2")
    p.add_argument("--nproc", type=int, default=8)
    args = p.parse_args()

    adata = ad.read_h5ad(args.input)
    log.info(f"Loaded {adata.n_obs:,} × {adata.n_vars:,} from {args.input}")

    # Carry through label + slice columns so they become dataset attributes.
    custom_attrs = {
        col: col for col in [
            "senescence_label", "age_label", "cell_type", "tissue", "tissue_general",
            "disease", "assay", "sex", "development_stage", "dataset_id", "donor_id",
        ] if col in adata.obs.columns
    }

    ds_path = tokenize_anndata(
        adata=adata,
        workdir=args.workdir,
        output_prefix=args.prefix,
        custom_attrs=custom_attrs,
        nproc=args.nproc,
        model_version=args.model_version,
    )
    log.info(f"Tokenized → {ds_path}")


if __name__ == "__main__":
    main()
