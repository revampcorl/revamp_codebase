"""CLI surface for the Stage 3 OpenPI/QAM trainer.

Most training choices live in YAML. These arguments are runtime overrides:
server URLs, resume location, smoke-test length, and launch-time sharding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ARGUMENTS = (
    (("--config",), {"required": True, "help": "Path to the Stage3 QAM yaml config."}),
    (("--q-server-url",), {"default": "http://127.0.0.1:8765", "help": "Local Q-gradient server URL."}),
    (("--imag-server-url",), {"default": None, "help": "World-model imagination server URL for two-chunk QAM."}),
    (("--output-dir",), {"default": None, "help": "Override config.system.checkpoint_dir."}),
    (("--resume-from",), {"default": None, "help": "OpenPI checkpoint directory used to initialize policy weights."}),
    (("--start-step",), {"type": int, "default": None, "help": "Initial global step. Defaults to suffix in --resume-from."}),
    (("--max-steps",), {"type": int, "default": None, "help": "Override config.training.max_steps."}),
    (("--log-interval",), {"type": int, "default": 10, "help": "Print JSON metrics every N steps."}),
    (("--num-fsdp-devices",), {"type": int, "default": None, "help": "Override the number of JAX devices in the FSDP mesh."}),
    (("--fsdp-min-mbytes",), {"type": int, "default": 4, "help": "Minimum leaf size for OpenPI FSDP sharding."}),
    (
        ("--q-payload-compression",),
        {
            "choices": ("compressed", "uncompressed"),
            "default": None,
            "help": "Override Q server npz payload compression.",
        },
    ),
    (("--skip-save",), {"action": "store_true", "help": "Skip final OpenPI checkpoint save for speed tests."}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed OpenPI/QAM policy training.")
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
    return parser.parse_args()


def start_step_from_args(args: argparse.Namespace) -> int:
    start_step = int(args.start_step if args.start_step is not None else 0)
    if args.resume_from and args.start_step is None:
        match = re.search(r"(?:^|_)step_(\d+)$", Path(args.resume_from).name)
        if match:
            start_step = int(match.group(1))
    return start_step
