#!/usr/bin/env python3
from __future__ import annotations

import argparse

from revamp.launch.config import load_config
from revamp.launch.pipeline import run_imagination_policy_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run REVAMP two-chunk imagination policy training.")
    parser.add_argument(
        "--config",
        default="configs/turn_on_sink_faucet.json",
        help="Path to the JSON experiment config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command graph without execution.")
    parser.add_argument("--smoke-steps", type=int, default=None, help="Override policy training steps for a quick test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_imagination_policy_pipeline(config, dry_run=args.dry_run, smoke_steps=args.smoke_steps)


if __name__ == "__main__":
    main()
