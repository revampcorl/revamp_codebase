#!/usr/bin/env python3
from __future__ import annotations

import argparse

from revamp.launch.config import load_config
from revamp.launch.pipeline import run_world_model_online_retrain_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REVAMP online world-model retraining.")
    parser.add_argument(
        "--config",
        default="configs/turn_on_sink_faucet.json",
        help="Path to the JSON experiment config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned retrain command without execution.")
    parser.add_argument(
        "--with-rollout-check",
        action="store_true",
        help="Run the rollout sanity check after retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_world_model_online_retrain_pipeline(
        config,
        dry_run=args.dry_run,
        with_rollout_check=args.with_rollout_check,
    )


if __name__ == "__main__":
    main()
