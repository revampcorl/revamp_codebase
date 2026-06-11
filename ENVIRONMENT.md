# Environment Setup

Use one merged conda environment for this release. The Stage3 OpenPI trainer,
world-model/Q servers, Wan video world model, and RoboCasa branch collection are
launched from the same repository and expect the same Python environment.

`requirements.txt` is a pip dependency surface, not a complete conda lockfile.
CUDA-sensitive packages are deliberately handled in the setup steps below
because the correct wheels depend on the GPU driver, CUDA runtime, and PyTorch
ABI.

## Local Environment Comparison

The current machine has two source environments:

| Environment | Role | Key packages |
| --- | --- | --- |
| `openpi` | OpenPI/JAX/RoboCasa side | Python `3.11.15`, `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `jax 0.5.3`, `jaxlib 0.5.3`, `flax`, `optax`, `orbax-checkpoint`, `robosuite`, `mujoco` |
| `viva` | ViVa/Wan/video side | Python `3.11.15`, `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `diffusers`, `transformers`, `flash-attn`, video/data packages |

Relevant packages found in both environments include `torch`, `torchvision`,
`accelerate`, `av`, `datasets`, `decord`, `diffusers`, `einops`, `fsspec`,
`ftfy`, `gymnasium`, `h5py`, `imageio`, `jaxtyping`, `lerobot`, `msgpack`,
`numpy`, `omegaconf`, `opencv-python`, `pandas`, `pillow`, `pyarrow`,
`pydantic`, `pyyaml`, `regex`, `safetensors`, `scipy`, `tensorboard`,
`tokenizers`, `tqdm`, `transformers`, `tyro`, and `wandb`.

OpenPI-only packages that are needed by this release include `jax`, `jaxlib`,
`flax`, `optax`, `orbax-checkpoint`, `augmax`, `beartype`, `dm-tree`, `etils`,
`ml-collections`, `mujoco`, `numpydantic`, `robosuite`, `tree`,
`tqdm-loggable`, and `websockets`.

Some shared packages have different versions across the two source
environments, especially `diffusers`, `transformers`, `numpy`, `opencv-python`,
`pillow`, `safetensors`, `scipy`, and `wandb`. The safest merge policy is to
pin only the GPU ABI-critical packages when reproducing a run, then let
`requirements.txt` describe the remaining import surface.

The merged environment should therefore start from the `openpi` stack and add
the ViVa/Wan dependencies. In practice that means:

- Keep Python `3.11`.
- Use one CUDA-compatible PyTorch install for both OpenPI and Wan.
- Add JAX/Flax/Optax/Orbax for OpenPI QAM training.
- Add Wan/video dependencies such as `diffusers`, `transformers`, `ftfy`,
  `regex`, `easydict`, `decord`, and `safetensors`.
- Treat `flash-attn` as optional but recommended on the training GPU. It is
  fast, but it is tied to the local Torch/CUDA ABI.

## Create The Environment

```bash
cd /path/to/repository

conda env create -f environment.yml
conda activate revamp
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first. On this machine the existing `openpi` and `viva`
environments both use CUDA 12.6 wheels:

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

If the target machine uses a different CUDA/runtime policy, use the official
PyTorch selector and keep `torch` and `torchvision` compatible.

Install JAX next. For current JAX releases, the official CUDA 12 path is:

```bash
python -m pip install --upgrade "jax[cuda12]"
```

For closest parity with the existing `openpi` environment, use `jax==0.5.3` and
`jaxlib==0.5.3` if those wheels are available for the target machine.

Then install the remaining project import surface:

```bash
python -m pip install -r requirements.txt
```

If Wan inference is too slow or attention fallback warnings appear, install
`flash-attn` after PyTorch is already installed:

```bash
python -m pip install flash-attn --no-build-isolation
```

Skip `flash-attn` on machines where it cannot build cleanly; the Wan attention
code has a PyTorch SDPA fallback.

## Third-Party Source Trees

This repository includes trimmed compatibility subsets for OpenPI, RoboCasa,
and Wan under `third_party/`. They provide the OpenPI config/client,
TurnOnSinkFaucet replay utilities, and Python `wan` package expected by the
launchers. If you want to use different local checkouts, use the runtime
variables below.

## Runtime Variables

The launch helpers add the repository and configured third-party paths to
`PYTHONPATH` automatically. For manual shells, this is the equivalent setup:

```bash
export REVAMP_OPENPI_ROOT=$PWD/third_party/openpi
export REVAMP_OPENPI_CLIENT_SRC=$PWD/third_party/openpi/packages/openpi-client/src
export REVAMP_ROBOCASA_ROOT=$PWD/third_party/robocasa
export REVAMP_WAN_CODE_ROOT=$PWD/third_party/wan

export PYTHONPATH=$PWD:$REVAMP_OPENPI_ROOT/src:$REVAMP_OPENPI_CLIENT_SRC:$REVAMP_ROBOCASA_ROOT:$REVAMP_WAN_CODE_ROOT:$PYTHONPATH
export MUJOCO_GL=egl

export REVAMP_WAN_CHECKPOINT_DIR=/path/to/Wan2.2-TI2V-5B
export REVAMP_ROBOCASA_ASSETS_ROOT=/path/to/robocasa_models/assets
```

`REVAMP_WAN_CODE_ROOT` points to Wan Python source code.
`REVAMP_WAN_CHECKPOINT_DIR` points to Wan model weights. They are intentionally
separate because code and weights may come from different locations.

## Smoke Tests

Run these before starting long jobs:

```bash
python - <<'PY'
import jax
import torch
import flax
import optax
import orbax.checkpoint
import diffusers
import transformers
import robosuite
import mujoco

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("jax", jax.__version__, "devices", jax.devices())
PY

python run_imagination_policy.py --dry-run
python run_world_model_update.py --dry-run
```

Dry-runs may recreate `outputs/`; that directory is generated and can be
removed before packaging.

## References

- PyTorch install selector and previous-version commands:
  <https://pytorch.org/get-started/previous-versions/>
- JAX CUDA installation notes:
  <https://docs.jax.dev/en/latest/installation.html>
