"""Outer closed-loop driver.

Runs (extract → ISP → mechanism → score → failure-analyze → acquire) N times,
logging each iteration's headline metrics to MLflow. Between iterations you can
optionally add newly labeled data (from a wet-lab or from a held-out perturbation
pool consumed by the active-learning picks) and retrain.

Usage:
  python scripts/run_loop.py \
      --data-cfg configs/data/senescence.yaml \
      --train-cfg configs/train/senescence_cls.yaml \
      --mech-cfg configs/mechanism/senescence.yaml \
      --score-cfg configs/scoring/senescence.yaml \
      --active-cfg configs/active/senescence.yaml \
      --iterations 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.utils import get_logger

log = get_logger("loop")


def _run(cmd: list[str]) -> None:
    log.info("$ " + " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"Step failed with exit code {r.returncode}: {cmd}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-cfg", type=Path, required=True)
    p.add_argument("--train-cfg", type=Path, required=True)
    p.add_argument("--mech-cfg", type=Path, required=True)
    p.add_argument("--score-cfg", type=Path, required=True)
    p.add_argument("--active-cfg", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--skip-data-and-train", action="store_true",
                   help="Skip stages 1–3 (assumes model already fine-tuned)")
    args = p.parse_args()

    py = sys.executable

    # Stage 1–3: only on first iteration or when explicitly forced.
    if not args.skip_data_and_train:
        _run([py, "scripts/01_pull_census.py", "--config", str(args.data_cfg)])
        _run([py, "scripts/02_tokenize.py",
              "--input", "data/senescence/census_slice.h5ad",
              "--workdir", "data/senescence",
              "--prefix", "senescence"])
        _run([py, "scripts/03_finetune.py", "--config", str(args.train_cfg)])

    for it in range(1, args.iterations + 1):
        log.info(f"\n========== Loop iteration {it}/{args.iterations} ==========")
        _run([py, "scripts/04_state_embs.py", "--config", str(args.mech_cfg)])
        _run([py, "scripts/05_isp.py", "--config", str(args.mech_cfg)])
        _run([py, "scripts/06_mechanism.py",
              "--isp-stats-csv", "outputs/senescence/isp_stats/senescence.csv",
              "--output-dir", f"outputs/senescence/mechanism/iter{it}"])
        _run([py, "scripts/07_score_targets.py", "--config", str(args.score_cfg)])
        _run([py, "scripts/08_failure_and_active.py",
              "--predictions-pkl",
              "outputs/senescence/finetune/runs/ksplit1/predictions.pkl",
              "--isp-stats-csv",
              "outputs/senescence/isp_stats/senescence.csv",
              "--output-report",
              f"outputs/senescence/failure/iter{it}_report.json",
              "--acquisition-config", str(args.active_cfg)])

        log.info(f"Iteration {it} complete. Inspect outputs/senescence/failure/iter{it}_report.json")
        log.info("To feed acquisitions back: run the picked perturbations wet-lab / on"
                 " scPerturb, add to labeled set, then re-run with --skip-data-and-train=false.")


if __name__ == "__main__":
    main()
