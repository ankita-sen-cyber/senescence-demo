"""Compute per-state mean cell embeddings (needed by InSilicoPerturber)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.mechanism import extract_state_embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["state_embs"]
    extract_state_embeddings(
        model_directory=cfg["model_directory"],
        labeled_dataset=cfg["labeled_dataset"],
        state_key=cfg["state_key"],
        states=cfg["states"],
        output_dir=cfg["output_dir"],
        output_prefix=cfg["output_prefix"],
        model_version=cfg["model_version"],
        max_ncells_per_state=cfg.get("max_ncells_per_state", 5000),
        emb_layer=cfg.get("emb_layer", -1),
        forward_batch_size=cfg.get("forward_batch_size", 100),
        nproc=cfg.get("nproc", 4),
    )


if __name__ == "__main__":
    main()
