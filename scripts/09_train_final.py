"""Train a versioned Geneformer deployment model on all labeled cells."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.deployment import train_deployment_model
from rnaseq_loop.train import FinetuneConfig
from rnaseq_loop.utils import get_logger, set_seed

log = get_logger("train-final")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New versioned directory, e.g. outputs/senescence/deployment/v1",
    )
    args = parser.parse_args()
    cfg = FinetuneConfig(**yaml.safe_load(args.config.read_text()))
    set_seed(cfg.seed)
    result = train_deployment_model(cfg, args.output_dir)
    log.info(f"Deployment model ready: {result['model']}")


if __name__ == "__main__":
    main()
