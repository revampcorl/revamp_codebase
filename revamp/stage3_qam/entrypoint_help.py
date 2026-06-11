"""Compatibility entrypoint for two-chunk imagination QAM.

The implementation lives in:
  - world_model_imag_server.py
  - openpi_qam_train_distributed.py

This file intentionally avoids the old sketch-only imports such as
``WorldModelClient`` or ``sample_actions_jit``.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-chunk imagination QAM is integrated into "
            "revamp.stage3_qam.train_policy. "
            "Use config_phase3_qam_turn_openpi_two_chunk_imag.yaml and run the "
            "Q server plus world_model_imag_server alongside the trainer."
        )
    )
    parser.parse_args()
    print(
        "Use:\n"
        "  python -m revamp.stage3_qam.q_gradient_server ...\n"
        "  python -m revamp.stage3_qam.imagination_server ...\n"
        "  python -m revamp.stage3_qam.train_policy "
        "--config world_model/config_phase3_qam_turn_openpi_two_chunk_imag.yaml ...",
        flush=True,
    )


if __name__ == "__main__":
    main()
