"""Pull a Census slice, build labels, and write h5ad."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rnaseq_loop.data import CensusQuery, label_age_binary, label_senescence_binary, pull_slice
from rnaseq_loop.utils import get_logger, set_seed

log = get_logger("pull_census")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(0)

    q = CensusQuery(**cfg["census"])
    adata = pull_slice(q, cfg["output_h5ad"])

    label_cfg = cfg["label"]
    if label_cfg["kind"] == "senescence_binary":
        adata = label_senescence_binary(
            adata,
            upper_quantile=label_cfg.get("upper_quantile", 0.25),
            lower_quantile=label_cfg.get("lower_quantile", 0.25),
        )
    elif label_cfg["kind"] == "age_binary":
        adata = label_age_binary(
            adata,
            young_max_years=label_cfg.get("young_max_years", 30.0),
            old_min_years=label_cfg.get("old_min_years", 65.0),
        )
    else:
        raise ValueError(f"Unknown label kind: {label_cfg['kind']}")

    # Overwrite with the labeled + filtered version.
    out = Path(cfg["output_h5ad"])
    adata.write_h5ad(out, compression="gzip")
    log.info(f"Final labeled slice: {adata.n_obs:,} cells  →  {out}")


if __name__ == "__main__":
    main()
