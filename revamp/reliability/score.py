"""Combine the two reliability signals and feed the closed loop (Sec. 3.4).

- ``combined_reliability`` — the per-state scalar r(s, ā) (Eq. 3):
      r = w_intr * r_intr + w_ext * r_ext
  with weights chosen so the in-distribution baseline E[r] is of order one.
- ``gate_weight`` — the imagined-update gate (Eq. 4):
      w(s, ā) = exp(-λ * r(s, ā))
  scaling each sample's contribution to the policy gradient so highly
  unreliable samples (r ≫ 0) contribute negligibly.
- ``is_triggered`` — the discrete real-rollout trigger (Eq. 5):
      s ∈ T  iff  r(s, ā) > τ
  states above τ are sent for a targeted real-environment rollout. τ is set so
  |T| matches the per-iteration real-rollout budget.
"""
from __future__ import annotations

import math


def combined_reliability(
    r_intr: float,
    r_ext: float,
    w_intr: float = 1.0,
    w_ext: float = 1.0,
) -> float:
    """Per-state reliability scalar r(s, ā) (Eq. 3). Higher = less reliable."""
    return w_intr * r_intr + w_ext * r_ext


def gate_weight(r: float, lam: float = 1.0) -> float:
    """Imagined-update gate weight w = exp(-λ r) (Eq. 4), in (0, 1]."""
    return math.exp(-lam * r)


def is_triggered(r: float, tau: float) -> bool:
    """Real-rollout trigger: True iff r exceeds the threshold τ (Eq. 5)."""
    return r > tau
