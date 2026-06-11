"""Constants for the WorldModel.

Core layout shared by the world-model trainer, Q head, and imagination server.

3-frame latent layout (1 cond + 2 target):
  Index | Slot                  | timestep | Notes
  ------|-----------------------|----------|------------------------------------
  0     | cam_concat cond       | 0 clean  | VAE-encoded raw[t-3..t]   (4 raw frame)
  1     | cam_concat target Q1  | t_rand   | VAE-encoded raw[t+1..t+4] (4 raw frame)
  2     | cam_concat target Q2  | t_rand   | VAE-encoded raw[t+5..t+8] (4 raw frame)

cam_concat = 3 cameras (left_wrist, right_wrist, high) concatenated along the
W axis (horizontal), producing a single [3, 12, 384, 960] video tensor per
sample. VAE 4x temporal + 16x spatial compress, then patch_embedding 2x spatial:
final patched grid is (T_p=3, H_p=12, W_p=30).

State and action injection do NOT occupy latent slots:
  - StateContextMLP : state    → 1 cross-attn token
  - StateTimeMLP    : state    → 1 instance offset, broadcast to all latent frames
  - ActionContextMLP: chunk    → CHUNK_LENGTH cross-attn tokens + learnable step
                                  embedding
  - ActionTimeMLP   : chunk    → per-latent-frame offset [B, 3, dim]
                                  latent[0] offset = 0 (cond unaffected)
                                  latent[1] offset = mlp(action[0..3])
                                  latent[2] offset = mlp(action[4..7])

future_state head removed: legacy joint-target variants can use
action_chunk[-1] as a next-state shortcut when action and state conventions
match. Contact-aware configs should use the real next-state fields carried by
the dataset/server path.

Q forward layout:
  - latent[0] = VAE encode raw[t-3..t]                       (clean cond)
  - latent[1..2] = torch.randn(...)                          (placeholder)
  - per_frame_timestep = [0, Q_PLACEHOLDER_TIMESTEP=999, 999]
  Rationale: aligns with Forward A main training distribution (sigma > 0.95
  occupies ~20.8% under shift=5 logit-warp). Q forward uses partial DiT
  (first Q_DIT_N_BLOCKS), pools latent[0] cam tokens, concatenates with
  state_proj + action_embed, feeds Q ensemble MLP.

Stage 2 / Stage 3 detach asymmetry (key contract; see ARCHITECTURE.md §3.7):
  - Stage 2 (train Q):  cam_pool.detach() KEPT      (saves ~30% backward)
  - Stage 3 (QAM):      cam_pool.detach() DROPPED   (let grad_a_Q flow via DiT)
  DiT params are permanently frozen (requires_grad=False) in both stages.
"""
from typing import List

# ===== 3-frame latent layout =====
COND_LATENT_IDX: int = 0
TARGET_LATENT_INDICES: List[int] = [1, 2]
NUM_LATENT_FRAMES: int = 3

# ===== Dataset yield shape =====
# Each sample yields 12 raw frames per camera: [t-3, t-2, ..., t+8].
# Camera tensor shape per sample: [3 cams, 12 raw_T, H=384, W=320].
# After W-axis concat: [3, 12, 384, 960] (cam_concat).
# After VAE encode + patch: latent [48, 3, 12, 30] (latent_T = 3).
NUM_RAW_FRAMES_PER_SAMPLE: int = 12
NUM_COND_RAW_FRAMES: int = 4         # raw[t-3..t]   → latent[0]
NUM_TARGET_RAW_FRAMES: int = 8       # raw[t+1..t+8] → latent[1..2]
COND_RAW_END_OFFSET: int = 0         # cond ends at raw[t + COND_RAW_END_OFFSET]
TARGET_RAW_START_OFFSET: int = 1     # target starts at raw[t + TARGET_RAW_START_OFFSET]

# ===== Action chunk shape =====
DEFAULT_CHUNK_LENGTH: int = 8
DEFAULT_ACTION_DIM: int = 14
# Per-latent-frame action segmentation (ActionTimeMLP):
#   latent[0] uses no action  → offset = 0
#   latent[1] uses action[0..CHUNK_HALF-1]   (front half of chunk)
#   latent[2] uses action[CHUNK_HALF..CHUNK_LENGTH-1] (back half)
CHUNK_SEGMENT_BOUNDARY: int = DEFAULT_CHUNK_LENGTH // 2   # = 4

# ===== State shape =====
DEFAULT_STATE_DIM: int = 14

# ===== Latent shape (Wan2.2-TI2V-5B) =====
LATENT_CHANNELS: int = 48

# ===== Future offset (raw frames) =====
# CRITICAL: future_offset is NOT an independent knob; it equals chunk_length
# x frames_per_action_step. See [PERF-4] deprecated for rationale.
DEFAULT_FUTURE_OFFSET: int = 8

# ===== Q forward configuration =====
DEFAULT_Q_DIT_N_BLOCKS: int = 8                   # partial DiT depth for Q
DEFAULT_Q_HIDDEN_DIM: int = 512
DEFAULT_Q_PLACEHOLDER_TIMESTEP: int = 999         # ODE start-point timestep
# Placeholder strategy: "randn" matches noisy target latents; "zeros" is kept
# only as an out-of-distribution ablation hook.
DEFAULT_Q_PLACEHOLDER_TYPE: str = "randn"

# ===== RL hyperparameters =====
DEFAULT_GAMMA: float = 0.95
DEFAULT_ALPHA: float = 1.0
DEFAULT_BETA_SUCC: float = 10.0
DEFAULT_BETA_FAIL: float = 5.0
DEFAULT_BETA_INT: float = 1.0
DEFAULT_LAMBDA_RHO: float = 5.0
DEFAULT_LAMBDA_W_Q: float = 2.0
DEFAULT_LAMBDA_W_PI: float = 3.0
DEFAULT_TAU_POLYAK: float = 0.005

# ===== Stage =====
STAGE_DYNAMICS_ONLY: int = 1     # Stage 1: train dynamics only
STAGE_RL: int = 2                # Stage 2: train Q (DiT frozen)
