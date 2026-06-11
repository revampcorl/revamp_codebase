"""Action faithfulness — external behavioral verification (Eq. 2).

Two lightweight networks, both trained on REAL demonstration data only, verify
that a prediction faithfully reflects the commanded action chunk:

- ``PoseProbe``: a frozen ResNet-18 backbone + small MLP that maps decoded
  multi-view RGB to a 7-D end-effector pose [pos(3) | quat(4)].
- ``InverseDynamicsMLP``: maps consecutive end-effector states (s_k, s_{k+1}) to
  the action a_k that produced the transition.

At inference the chain is:

    cond, state_t, action_chunk  --world model-->  H future RGB frames
    probe(RGB)                   -->  s̃_{t+1..t+H}      (per-frame EE pose)
    inverse-dynamics(s̃_k, s̃_{k+1})  -->  ã_k            (implied action chunk)
    r_ext = mean_k || ã_k - action_chunk[k] ||₁

Because the probe and inverse-dynamics module are trained on data whose support
contains the world model's RGB output (the *containment* property), a large
r_ext is attributable to the world model failing to render the commanded action
rather than to probe/ID error.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# EE pose slice within the proprioceptive state: pos (3) + quat (4).
STATE_EE_SLICE = slice(7, 14)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# Visual probe: RGB -> 7-D EE pose
# --------------------------------------------------------------------------- #
class PoseProbe(nn.Module):
    """Frozen ResNet-18 backbone (weight-shared across cameras) + 2-layer MLP."""

    def __init__(self, hidden_dim: int = 256, num_cams: int = 3,
                 freeze_backbone: bool = True):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        bb.fc = nn.Identity()
        self.backbone = bb
        feat_dim = 512
        self.num_cams = num_cams
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * num_cams),
            nn.Linear(feat_dim * num_cams, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7),
        )
        self._freeze_bb = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_bb:
            # Keep BN in eval to avoid statistics drift on small data.
            self.backbone.eval()
        return self

    def _features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, num_cams, 3, H, W) -> (B, num_cams * 512)."""
        B, C, _, H, W = frames.shape
        x = frames.view(B * C, 3, H, W)
        if self._freeze_bb:
            with torch.no_grad():
                f = self.backbone(x)
        else:
            f = self.backbone(x)
        return f.view(B, C * 512)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        out = self.head(self._features(frames))
        pos, rot = out[..., :3], out[..., 3:7]
        rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([pos, rot], dim=-1)


# --------------------------------------------------------------------------- #
# Inverse dynamics: (s_k, s_{k+1}) -> a_k
# --------------------------------------------------------------------------- #
class InverseDynamicsMLP(nn.Module):
    """f(s_k, s_{k+1}) -> a_k, trained on real (state, action) pairs only."""

    def __init__(self, state_ee_dim: int = 7, action_dim: int = 12,
                 hidden: int = 256):
        super().__init__()
        in_dim = state_ee_dim * 3  # s_k, s_{k+1}, Δ
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s_k: torch.Tensor, s_kp1: torch.Tensor) -> torch.Tensor:
        delta = s_kp1 - s_k
        return self.net(torch.cat([s_k, s_kp1, delta], dim=-1))


# --------------------------------------------------------------------------- #
# Inference chain
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_decoded(probe: PoseProbe, decoded_per_cam: Dict[str, torch.Tensor],
                  device) -> torch.Tensor:
    """Run the probe over decoded RGB frames -> per-frame EE pose.

    Args:
        decoded_per_cam: {'left': [1,3,H,h,w], 'right': ..., 'high': ...} float
            in [-1, 1] (from ``predict_chunk`` keys ``next_cam_{left,right,high}``).

    The probe was trained on per-frame stacks in camera order
    (agentview_left, agentview_right, eye_in_hand). The world-model interface
    maps those to ("high", "right", "left"), so we read the decoded cameras in
    that order.

    Returns:
        Tensor of shape (n_frames, 7).
    """
    n_frames = decoded_per_cam["left"].shape[2]
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    preds = []
    for f in range(n_frames):
        cams = []
        for cam_key in ("high", "right", "left"):
            rgb = (decoded_per_cam[cam_key][0, :, f] + 1.0) * 0.5  # [-1,1] -> [0,1]
            cams.append(rgb.clamp(0.0, 1.0))
        cams_t = torch.stack(cams, dim=0).to(device)  # (3, 3, h, w)
        cams_t = (cams_t - mean) / std
        preds.append(probe(cams_t.unsqueeze(0))[0])
    return torch.stack(preds, dim=0)  # (n_frames, 7)


@torch.no_grad()
def implied_action_residual(
    probe: PoseProbe,
    id_model: InverseDynamicsMLP,
    decoded_per_cam: Dict[str, torch.Tensor],
    action_chunk_commanded: torch.Tensor,  # (H, action_dim) on device
    device,
) -> dict:
    """External reliability score r_ext (Eq. 2): L1 between implied and commanded.

    probe(RGB) -> H EE states -> inverse-dynamics -> (H-1) implied actions,
    compared against the first (H-1) commanded actions.

    Returns:
        dict with ``score`` (scalar r_ext = mean L1), ``per_step_l1`` (per-step
        L1), ``implied`` and ``commanded`` chunks.
    """
    s_chunk = probe_decoded(probe, decoded_per_cam, device)  # (H, 7)
    implied = id_model(s_chunk[:-1], s_chunk[1:])            # (H-1, action_dim)
    cmd = action_chunk_commanded[: implied.shape[0]]         # (H-1, action_dim)
    diff = (implied - cmd).abs()
    return {
        "score": diff.mean().item(),
        "per_step_l1": diff.mean(dim=-1).detach().cpu(),
        "implied": implied.detach().cpu(),
        "commanded": cmd.detach().cpu(),
    }
