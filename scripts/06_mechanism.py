"""Post-hoc mechanism: TF activities (CollecTRI), pathway activities (PROGENy),
   and GSEA (MSigDB Hallmark) from the ISP stats output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rnaseq_loop.mechanism import (
    get_collectri,
    get_progeny,
    gsea_from_ranked_genes,
    pathway_activity_from_ranked_genes,
    tf_activity_from_ranked_genes,
)
from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger("mechanism")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isp-stats-csv", required=True, type=Path,
                   help="output of Geneformer InSilicoPerturberStats, e.g. senescence.csv")
    p.add_argument("--symbol-col", default="Gene_name")
    p.add_argument("--score-col", default="Shift_to_goal_end")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    ensure_dir(args.output_dir)
    stats = pd.read_csv(args.isp_stats_csv)
    log.info(f"Loaded ISP stats: {stats.shape}")

    # Turn into a Series indexed by gene symbol.
    scores = (
        stats.set_index(args.symbol_col)[args.score_col]
        .astype(float)
        .groupby(level=0).mean()
    )
    scores.name = "attribution"

    log.info("Computing CollecTRI TF activity")
    tf = tf_activity_from_ranked_genes(scores, net=get_collectri(), method="mlm")
    tf.to_csv(args.output_dir / "tf_activity.csv", index=False)

    log.info("Computing PROGENy pathway activity")
    pw = pathway_activity_from_ranked_genes(scores, net=get_progeny())
    pw.to_csv(args.output_dir / "pathway_activity.csv", index=False)

    log.info("Running GSEA (MSigDB Hallmark)")
    gsea = gsea_from_ranked_genes(scores, gene_sets="MSigDB_Hallmark_2020",
                                   output_dir=args.output_dir / "gsea")
    gsea.to_csv(args.output_dir / "gsea_hallmark.csv", index=False)

    log.info(f"Done. Mechanism artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
