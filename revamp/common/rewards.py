"""Reward utilities: PBRS shaping + uncertainty-driven gates.

Implements the reward design from ARCHITECTURE.md §5:

  Φ(s) = ρ(s) · α · ViVa(s)
  r̂_t = (γ Φ(s_{t+1}) - Φ(s_t)) + r_sparse,t
  ρ(s) = phase-dependent gate
  w_Q(s) = data-source-dependent trust weight (Q loss)
  w_π(s) = data-source-dependent trust weight (policy loss)

Phase 1 simplification: ρ ≡ 1, w_Q ≡ 1, w_π ≡ 1 for all data
(no online OOD data exists yet).

Phase 2 (TODO): ρ, w_Q, w_π become functions of u_epistemic(s) +
data source flag.
"""
from __future__ import annotations

from typing import Optional
import torch

from revamp.common.constants import (
    DEFAULT_GAMMA, DEFAULT_ALPHA,
    DEFAULT_BETA_SUCC, DEFAULT_BETA_FAIL, DEFAULT_BETA_INT,
    DEFAULT_LAMBDA_RHO, DEFAULT_LAMBDA_W_Q, DEFAULT_LAMBDA_W_PI,
)


# -------- Data-source flags (used by Phase 2; ignored in Phase 1) ----------
SOURCE_DEMO = 0      # offline demonstration data, always ID
SOURCE_IMAG = 1      # imagined rollout from world model
SOURCE_REAL = 2      # real-robot trigger data, always OOD by definition


def potential(
    viva_value: torch.Tensor,             # [B] ∈ [0, 1] from frozen ViVa
    rho: torch.Tensor,                    # [B] ∈ [0, 1]
    alpha: float = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Φ(s) = ρ(s) · α · ViVa(s)"""
    return rho * alpha * viva_value


def pbrs_step(
    viva_t: torch.Tensor,                 # [B] ViVa(s_t)
    viva_tp1: torch.Tensor,               # [B] ViVa(s_{t+1})
    rho_t: torch.Tensor,                  # [B] ρ(s_t)
    rho_tp1: torch.Tensor,                # [B] ρ(s_{t+1})
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """One step PBRS shaping: F_t = γ Φ(s_{t+1}) - Φ(s_t).

    With ρ gate, OOD states automatically have Φ ≈ 0, so shaping decays.
    """
    phi_t = potential(viva_t, rho_t, alpha)
    phi_tp1 = potential(viva_tp1, rho_tp1, alpha)
    return gamma * phi_tp1 - phi_t


def sparse_reward(
    success_mask: torch.Tensor,           # [B] 1/0
    fail_mask: Optional[torch.Tensor] = None,
    intervention_mask: Optional[torch.Tensor] = None,
    beta_succ: float = DEFAULT_BETA_SUCC,
    beta_fail: float = DEFAULT_BETA_FAIL,
    beta_int: float = DEFAULT_BETA_INT,
) -> torch.Tensor:
    """r_sparse,t = β_succ · 1[succ] - β_fail · 1[fail] - β_int · 1[interv]"""
    r = beta_succ * success_mask.float()
    if fail_mask is not None:
        r = r - beta_fail * fail_mask.float()
    if intervention_mask is not None:
        r = r - beta_int * intervention_mask.float()
    return r


def step_reward(
    viva_t: torch.Tensor,
    viva_tp1: torch.Tensor,
    rho_t: torch.Tensor,
    rho_tp1: torch.Tensor,
    success_mask: torch.Tensor,
    fail_mask: Optional[torch.Tensor] = None,
    intervention_mask: Optional[torch.Tensor] = None,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    beta_succ: float = DEFAULT_BETA_SUCC,
    beta_fail: float = DEFAULT_BETA_FAIL,
    beta_int: float = DEFAULT_BETA_INT,
) -> torch.Tensor:
    """Single-step reward: PBRS + sparse."""
    f = pbrs_step(viva_t, viva_tp1, rho_t, rho_tp1, alpha, gamma)
    r = sparse_reward(
        success_mask, fail_mask, intervention_mask,
        beta_succ, beta_fail, beta_int,
    )
    return f + r


def chunk_pbrs(
    viva_chunk: torch.Tensor,             # [B, H+1] ViVa values along chunk: s_t..s_{t+H}
    rho_chunk: torch.Tensor,              # [B, H+1] ρ values along chunk
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """Per-step PBRS along an H-step chunk.

    Returns:
        F: [B, H] per-step shaping rewards
    """
    phi = potential(viva_chunk, rho_chunk, alpha)              # [B, H+1]
    return gamma * phi[:, 1:] - phi[:, :-1]                    # [B, H]


def chunk_step_rewards(
    viva_chunk: torch.Tensor,             # [B, H+1]
    rho_chunk: torch.Tensor,              # [B, H+1]
    success_mask_chunk: torch.Tensor,     # [B, H]
    fail_mask_chunk: Optional[torch.Tensor] = None,
    intervention_mask_chunk: Optional[torch.Tensor] = None,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    beta_succ: float = DEFAULT_BETA_SUCC,
    beta_fail: float = DEFAULT_BETA_FAIL,
    beta_int: float = DEFAULT_BETA_INT,
) -> torch.Tensor:
    """Full per-step rewards along a chunk: F + sparse.

    Returns:
        r: [B, H]
    """
    F = chunk_pbrs(viva_chunk, rho_chunk, alpha, gamma)
    s = beta_succ * success_mask_chunk.float()
    if fail_mask_chunk is not None:
        s = s - beta_fail * fail_mask_chunk.float()
    if intervention_mask_chunk is not None:
        s = s - beta_int * intervention_mask_chunk.float()
    return F + s


# ---------- ρ, w_Q, w_π gates -----------

def rho(
    u_epistemic: torch.Tensor,            # [B] uncertainty values, ≥ 0
    source_flag: torch.Tensor,            # [B] one of SOURCE_*
    phase: int = 1,
    lambda_rho: float = DEFAULT_LAMBDA_RHO,
) -> torch.Tensor:
    """ρ(s): ViVa shaping gate.

    Phase 1: ρ ≡ 1 (no OOD data yet).
    Phase 2:
      - SOURCE_DEMO: ρ = 1 (in distribution)
      - SOURCE_REAL: ρ = 0 (always OOD by definition; force ViVa off)
      - SOURCE_IMAG: ρ = exp(-λ_ρ · u_epistemic)
    """
    if phase == 1:
        return torch.ones_like(u_epistemic)
    # Phase 2:
    out = torch.empty_like(u_epistemic)
    out[source_flag == SOURCE_DEMO] = 1.0
    out[source_flag == SOURCE_REAL] = 0.0
    imag_mask = source_flag == SOURCE_IMAG
    out[imag_mask] = torch.exp(-lambda_rho * u_epistemic[imag_mask])
    return out


def trust_weight_q(
    u_epistemic: torch.Tensor,
    source_flag: torch.Tensor,
    phase: int = 1,
    lambda_w: float = DEFAULT_LAMBDA_W_Q,
) -> torch.Tensor:
    """w_Q(s): trust weight on Q loss.

    Phase 1: w_Q ≡ 1.
    Phase 2:
      - SOURCE_DEMO: 1
      - SOURCE_REAL: 1 (real data is high-quality even if OOD)
      - SOURCE_IMAG: exp(-λ_w · u_epistemic)
    """
    if phase == 1:
        return torch.ones_like(u_epistemic)
    out = torch.ones_like(u_epistemic)
    imag_mask = source_flag == SOURCE_IMAG
    out[imag_mask] = torch.exp(-lambda_w * u_epistemic[imag_mask])
    return out


def trust_weight_pi(
    u_epistemic: torch.Tensor,
    source_flag: torch.Tensor,
    phase: int = 1,
    lambda_w: float = DEFAULT_LAMBDA_W_PI,
) -> torch.Tensor:
    """w_π(s): trust weight on policy loss.

    Phase 1: w_π ≡ 1.
    Phase 2:
      - SOURCE_DEMO: 1
      - SOURCE_REAL: 1 (we want policy to learn real-robot behaviors)
      - SOURCE_IMAG: exp(-λ_w · u_epistemic)  (more aggressive than w_Q)
    """
    if phase == 1:
        return torch.ones_like(u_epistemic)
    out = torch.ones_like(u_epistemic)
    imag_mask = source_flag == SOURCE_IMAG
    out[imag_mask] = torch.exp(-lambda_w * u_epistemic[imag_mask])
    return out
