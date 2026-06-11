"""Chunk Q head: Q(s_t, ā_t) → scalar.

Input is a fused feature vector built upstream by WorldModel:
    feats = concat(pool(partial_DiT(curr_obs, action)), state_proj, action_embed)
    q1, q2 = QEnsemble(feats)

Implements:
  - K=2 ensemble (Q_θ1, Q_θ2) for TD3-style clipped double Q (no entropy term)
  - Target networks (Q_θ̄1, Q_θ̄2) updated via Polyak averaging
  - chunk-level n-step TD target

Stage 2 trains Q via Fitted Q Evaluation (FQE) of the demonstration policy:
ā' for the bootstrap is taken from the dataset's next chunk (no learned
policy network in the loop). The resulting Q^{π_demo} is the value signal
consumed by Stage 3 QAM (Adjoint Matching) when fine-tuning π0.
See PI0_FINETUNING.md for the full pipeline rationale.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from revamp.common.constants import (
    DEFAULT_GAMMA, DEFAULT_TAU_POLYAK,
)


class QHead(nn.Module):
    """3-layer MLP: features → scalar Q."""

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class QEnsemble(nn.Module):
    """K=2 Q ensemble + Polyak target networks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        tau: float = DEFAULT_TAU_POLYAK,
    ):
        super().__init__()
        self.tau = tau

        self.q1 = QHead(in_dim, hidden_dim)
        self.q2 = QHead(in_dim, hidden_dim)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self._reinit_q2()

    def _reinit_q2(self):
        for module in self.q2.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("linear") * 0.85,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.q2_target.load_state_dict(self.q2.state_dict())

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(features), self.q2(features)

    def target(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.q1_target(features), self.q2_target(features)

    def target_min(self, features: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.target(features)
        return torch.min(q1, q2)

    def online_min(self, features: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(features)
        return torch.min(q1, q2)

    def polyak_update(self) -> None:
        with torch.no_grad():
            for p, p_tgt in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_tgt in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)


def compute_td_target(
    chunk_rewards: torch.Tensor,           # [B, H_chunk]
    bootstrap_q: torch.Tensor,             # [B] min Q at chunk end
    gamma: float = DEFAULT_GAMMA,
    done_chunk: Optional[torch.Tensor] = None,  # [B, H], 1 at terminal event
) -> torch.Tensor:
    """y(s_t, ā_t) = Σ γ^i r_{t+i} + 1[not done] γ^H Q_target(s_{t+H}, ā').

    If a terminal event appears inside the chunk, rewards after that event are
    masked out and the bootstrap term is removed.
    """
    B, H = chunk_rewards.shape
    device = chunk_rewards.device
    rewards = chunk_rewards
    if done_chunk is not None:
        done = done_chunk.to(device=device, dtype=chunk_rewards.dtype).clamp(0.0, 1.0)
        done_before = torch.cumsum(done, dim=1) - done
        reward_mask = (done_before <= 0).to(dtype=chunk_rewards.dtype)
        rewards = rewards * reward_mask
        nonterminal = (done.sum(dim=1) <= 0).to(dtype=chunk_rewards.dtype)
    else:
        nonterminal = torch.ones(B, device=device, dtype=chunk_rewards.dtype)
    discounts = gamma ** torch.arange(H, device=device, dtype=chunk_rewards.dtype)
    discounted_sum = (rewards * discounts.unsqueeze(0)).sum(dim=1)
    return discounted_sum + nonterminal * (gamma ** H) * bootstrap_q


def q_loss(
    q_pred_1: torch.Tensor,                # [B]
    q_pred_2: torch.Tensor,                # [B]
    q_target: torch.Tensor,                # [B]
    sample_weight: Optional[torch.Tensor] = None,  # [B] w_Q
) -> torch.Tensor:
    """L_Q = E[w_Q · ((Q1 - y)² + (Q2 - y)²)]."""
    target = q_target.detach()
    err1 = (q_pred_1 - target) ** 2
    err2 = (q_pred_2 - target) ** 2
    per_sample = err1 + err2
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    return per_sample.mean()


