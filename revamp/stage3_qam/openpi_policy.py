"""OpenPI policy bridge for Stage 3 integration.

This bridge loads the local JAX/Flax OpenPI policy and exposes a small
PyTorch-facing sampling interface. It is intentionally inference-oriented:
JAX parameters are not PyTorch `nn.Parameter`s, so PyTorch autograd cannot
update pi0 through this wrapper. Full QAM training needs a JAX-side training
loop or an explicit cross-framework gradient bridge.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class OpenPiPolicyBridge(nn.Module):
    """Loads OpenPI and returns action chunks compatible with Stage 2 Q."""

    def __init__(
        self,
        *,
        openpi_root: str,
        config_name: str,
        checkpoint_path: str,
        prompt: str,
        q_action_horizon: int,
        q_action_dim: int,
        pi0_state_dim: int = 16,
        num_flow_steps: int = 10,
        data_dirs: List[Dict[str, Any]] | None = None,
    ):
        super().__init__()
        src = Path(openpi_root).expanduser().resolve() / "src"
        if not src.exists():
            raise FileNotFoundError(f"OpenPI src directory not found: {src}")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        self.prompt = prompt
        self.q_action_horizon = int(q_action_horizon)
        self.q_action_dim = int(q_action_dim)
        self.pi0_state_dim = int(pi0_state_dim)
        self.num_flow_steps = int(num_flow_steps)

        train_config = _config.get_config(config_name)
        if data_dirs is not None:
            train_config = dataclasses.replace(
                train_config,
                data=dataclasses.replace(train_config.data, data_dirs=data_dirs),
            )
        self.openpi_action_horizon = int(train_config.model.action_horizon)
        self.openpi_action_dim = int(train_config.model.action_dim)
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_path,
            default_prompt=prompt,
            sample_kwargs={"num_steps": self.num_flow_steps},
        )

    @staticmethod
    def _to_hwc_uint8(img_chw: torch.Tensor) -> np.ndarray:
        img = img_chw.detach().float().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 2.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _obs_from_batch_item(
        self,
        state: torch.Tensor,
        cam_left_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
    ) -> Dict[str, Any]:
        # WorldModelDataset state_dim=17 includes contact; OpenPI RoboCasa
        # policy was trained on the proprio state, so use the leading state dims.
        proprio = state.detach().float().cpu().numpy()[: self.pi0_state_dim]
        current_left = cam_left_cond[:, -1] if cam_left_cond.ndim == 4 else cam_left_cond
        current_high = cam_high_cond[:, -1] if cam_high_cond.ndim == 4 else cam_high_cond
        return {
            "observation/state": proprio,
            "observation/image": self._to_hwc_uint8(current_high),
            "observation/wrist_image": self._to_hwc_uint8(current_left),
            "prompt": self.prompt,
        }

    def flow_forward(
        self,
        state: torch.Tensor,
        *,
        cam_left_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
        n_steps: int | None = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, float]]]:
        """Sample pi0 actions and return the first Stage2-Q-compatible chunk."""
        device = state.device
        actions: List[torch.Tensor] = []
        traj_by_step: List[List[torch.Tensor]] = []
        traj_times = None
        steps = int(n_steps or self.num_flow_steps)

        for i in range(state.shape[0]):
            obs = self._obs_from_batch_item(state[i], cam_left_cond[i], cam_high_cond[i])
            out = self.policy.infer_with_traj(obs, num_steps=steps)
            a = torch.as_tensor(out["actions"][: self.q_action_horizon, : self.q_action_dim], device=device).float()
            actions.append(a)

            traj = out.get("traj_actions", None)
            if traj is not None:
                traj = traj[:, : self.q_action_horizon, : self.q_action_dim]
                if len(traj_by_step) == 0:
                    traj_by_step = [[] for _ in range(traj.shape[0])]
                for j in range(traj.shape[0]):
                    traj_by_step[j].append(torch.as_tensor(traj[j], device=device).float())
                traj_times = out.get("traj_times", None)

        a1 = torch.stack(actions, dim=0)
        if not traj_by_step:
            return a1, [(a1, 0.0)]

        trajectory: List[Tuple[torch.Tensor, float]] = []
        for j, items in enumerate(traj_by_step):
            t = float(traj_times[j]) if traj_times is not None else float(j + 1) / len(traj_by_step)
            trajectory.append((torch.stack(items, dim=0), t))
        return a1, trajectory

    def vector_field(self, *args, **kwargs):
        raise NotImplementedError(
            "OpenPiPolicyBridge can sample OpenPI actions for Q scoring, but it "
            "cannot train pi0 with PyTorch autograd. Full Stage 3 QAM needs a "
            "JAX-side QAM trainer or an explicit JAX<->PyTorch gradient bridge."
        )
