# REVAMP: Reliability-Aware Dynamics-Value World Models for VLA Policy Improvement

<a href="https://revampcorl.github.io/REVAMP/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-REVAMP-1565C0?style=for-the-badge"></a>

## Overview

REVAMP improves a pretrained VLA policy (OpenPI π0) beyond supervised
fine-tuning by pairing a learned video world model with a reliability gate.
It has three parts:

- **Unified world model** — a single Wan2.2-based DiT backbone, conditioned on
  the action chunk and a contact signal, with two heads sharing one encoder: a
  dynamics head that predicts future multi-view frames and a Q-head that
  predicts action-and-contact-conditioned Q-values. The shared encoder lets the
  Q-head capture subtle contact effects such as failed grasps where the visual
  change is minimal.
- **Reliability-aware mechanism** — scores every prediction with two signals:
  an intrinsic velocity-field self-consistency residual (geometric faithfulness)
  and an external visual-probe-plus-inverse-dynamics check that recovers the
  implied action chunk (action faithfulness).
- **Closed-loop improvement** — a frequent imagination loop fine-tunes the
  policy with reliability-gated QAM, while a sparse real-interaction loop
  triggers targeted real rollouts at flagged states and refits the world model
  exactly where it was unreliable.

## Installation

```bash
conda env create -f environment.yml
conda activate revamp
python -m pip install -r requirements.txt
```

For full experiments, install CUDA-matched PyTorch/JAX first, then the
remaining requirements. See `ENVIRONMENT.md` for the detailed setup order,
GPU notes, and optional `flash-attn`.

## Usage

### `python run_imagination_policy.py`

Runs the imagination-loop policy improvement: it starts the Q-gradient server,
the world-model imagination server, and the OpenPI policy trainer, then
fine-tunes π0 with two-chunk reliability-gated QAM.

### `python run_world_model_update.py`

Runs the real-interaction-loop world-model update: retrains the world model on
the original demonstrations (`datasets/origin`) together with the online
rollouts collected for correction (`datasets/online_recovery_rollouts`,
`datasets/new_scene_data`).

```bash
python run_imagination_policy.py      # 1. policy improvement (imagination loop)
python run_world_model_update.py      # 2. world-model update (real-interaction loop)
```

## Repository Layout

```text
repository/
├── run_imagination_policy.py          # imagination-loop policy improvement
├── run_world_model_update.py          # merged online world-model update
├── configs/
│   ├── turn_on_sink_faucet.json       # central experiment/asset/GPU config
│   └── world_model/*.source.yaml      # source training configs
├── revamp/                            # common/, launch/, world-model, reliability, qam, online-update
├── third_party/                       # trimmed openpi/, robocasa/, wan/ subsets
├── ENVIRONMENT.md
├── ASSETS.md
├── environment.yml
└── requirements.txt
```

## Configuration

All launchers default to `configs/turn_on_sink_faucet.json`, which controls
dataset/checkpoint/output paths, third-party code paths, ports, and GPU
assignment. Adjust the `cuda_visible_devices` fields before running on a
different machine. Outputs are written to `outputs/turn_on_sink_faucet/`; for
machines without online logging, run with `export WANDB_MODE=offline`.

The included `third_party/` subsets are used by default, so the path variables
below are optional — set them only to point at a different local checkout:

```bash
export REVAMP_OPENPI_ROOT=$PWD/third_party/openpi
export REVAMP_OPENPI_CLIENT_SRC=$PWD/third_party/openpi/packages/openpi-client/src
export REVAMP_ROBOCASA_ROOT=$PWD/third_party/robocasa
export REVAMP_WAN_CODE_ROOT=$PWD/third_party/wan
```
