"""Reliability-aware mechanism (paper Sec. 3.3).

Scores every world-model prediction with two complementary signals and combines
them into a single per-state reliability scalar consumed by the closed-loop
improvement (Sec. 3.4):

- Geometric faithfulness (intrinsic) — ``geometric.velocity_field_residual``:
  velocity-field self-consistency residual ``r_intr`` (Eq. 1). Catches
  predictions that have drifted off the training manifold (objects distort,
  change color, disappear).
- Action faithfulness (external) — ``action_faithfulness.implied_action_residual``:
  a visual probe maps decoded RGB to end-effector pose and an inverse-dynamics
  module recovers the implied action chunk; ``r_ext`` is its L1 discrepancy from
  the commanded chunk (Eq. 2). Catches plausible-looking predictions that do not
  faithfully reflect the commanded action.

``score`` combines the two signals (Eq. 3), turns them into the imagined-update
gate weight (Eq. 4), and into the real-rollout trigger (Eq. 5).
"""
from __future__ import annotations

from revamp.reliability.geometric import velocity_field_residual
from revamp.reliability.action_faithfulness import (
    PoseProbe,
    InverseDynamicsMLP,
    probe_decoded,
    implied_action_residual,
)
from revamp.reliability.score import (
    combined_reliability,
    gate_weight,
    is_triggered,
)

__all__ = [
    "velocity_field_residual",
    "PoseProbe",
    "InverseDynamicsMLP",
    "probe_decoded",
    "implied_action_residual",
    "combined_reliability",
    "gate_weight",
    "is_triggered",
]
