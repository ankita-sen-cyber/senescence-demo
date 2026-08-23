"""Fuse ISP attribution × Open Targets × DepMap into a final target ranking."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rnaseq_loop.scoring import (
    ScoringConfig,
    depmap_essentiality_summary,
    load_depmap_essentiality,
    opentargets_association,
    rank_targets,
)
from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger("score_targets")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    # 1. Attribution
    attr = pd.read_csv(cfg["attribution_csv"])
    attr_series = (
        attr.set_index("Gene_name")["Shift_to_goal_end"]
        .astype(float).groupby(level=0).mean()
    )
    attr_series.name = "isp_attribution"

    sources: dict[str, pd.Series] = {"isp_attribution": attr_series}

    # 2. Open Targets
    ot_cfg = cfg["opentargets"]
    ot = opentargets_association(ot_cfg["disease_efo"], top_n=ot_cfg["top_n"])
    sources["ot_overall_score"] = ot.set_index("symbol")["ot_overall_score"]

    # 3. DepMap (optional)
    dm_cfg = cfg.get("depmap") or {}
    if dm_cfg.get("csv_path"):
        gene_df = load_depmap_essentiality(
            dm_cfg["csv_path"],
            subset_cell_lines=dm_cfg.get("subset_cell_lines"),
        )
        dm = depmap_essentiality_summary(gene_df)
        sources["depmap_frac_essential"] = dm["depmap_frac_essential"]

    scfg = ScoringConfig(
        weights=cfg["weights"],
        min_sources=cfg.get("min_sources", 2),
    )
    ranked = rank_targets(sources, scfg, key="symbol")

    out_csv = Path(cfg["output_csv"])
    ensure_dir(out_csv.parent)
    ranked.to_csv(out_csv)
    log.info(f"Wrote {out_csv}. Top 10:\n{ranked.head(10)}")


if __name__ == "__main__":
    main()
