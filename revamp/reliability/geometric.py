"""Geometric faithfulness — intrinsic velocity-field self-consistency (Eq. 1).

After the world model's ``predict_chunk`` produces a clean target latent x̂, we
probe the flow-matching velocity field v_θ at multiple sampled (noise, σ) pairs:

    for i in 1..k:
        z_i ~ N(0, I)
        σ_i ~ uniform over the FM scheduler's training σ band
        x_t = (1 - σ_i) * x̂ + σ_i * z_i        (target slots only; cond untouched)
        v_pred  = v_θ(x_t, σ_i, cond, action, state, t5)
        v_target = z_i - x̂                      (the world model's FM target)
        r_i = || v_pred - v_target ||²          (target slots only)
    r_intr = mean_i r_i

When (cond, action) is on-manifold the velocity field models the (clean, noise,
σ) manifold smoothly and the residual sits near the training-loss floor; when
the prediction has departed off-manifold the field is inconsistent and the
residual grows. This is a self-consistency check against the model's OWN
prediction x̂ — it needs no ground-truth future.

Cost: one DiT forward per probe sample, i.e. ≈ ``k_probes`` × a single forward.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from revamp.common.constants import (
    TARGET_LATENT_INDICES,
    NUM_LATENT_FRAMES,
)


@torch.no_grad()
def velocity_field_residual(
    wm,
    state: torch.Tensor,                 # [B, state_dim]
    cam_left_segment: torch.Tensor,      # [B, 3, T, H, W]
    cam_right_segment: torch.Tensor,
    cam_high_segment: torch.Tensor,
    action_chunk: torch.Tensor,          # [B, horizon, action_dim]
    t5_embeddings: torch.Tensor,         # [B, L, D]
    clean_target_latent: torch.Tensor,   # [B, C, len(TARGET_LATENT_INDICES), H', 3W']
    k_probes: int = 6,
    sigma_range: tuple = (0.2, 0.95),
    seed: Optional[int] = None,
) -> dict:
    """Probe the velocity field at k random (z, σ) pairs; return the mean residual.

    Args:
        wm: a ``revamp.common.world_model.WorldModel`` instance.
        clean_target_latent: x̂, the target latent from a prior ``predict_chunk``.

    Returns:
        dict with ``score`` (scalar r_intr in FM-loss units), ``per_probe``
        (list of per-sample residuals), ``sigma_used`` and ``k``.
    """
    device = state.device
    B = state.shape[0]
    assert B == 1, "velocity_field_residual is implemented for B=1 only"

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

    # Build the cond-frame latent from the camera segments, then overwrite the
    # target slots with the world model's own prediction x̂ to form the "clean"
    # full latent sequence we probe around.
    cond_clean = wm.build_full_latent_sequence(
        cam_left_segment, cam_right_segment, cam_high_segment,
    )
    clean_full = cond_clean.clone()
    for i, idx in enumerate(TARGET_LATENT_INDICES):
        clean_full[:, :, idx] = clean_target_latent[:, :, i]

    # Sample σ uniformly from the FM scheduler's training σ band.
    fm_sched = wm.fm_scheduler
    sigmas_full = fm_sched.sigmas.to(device=device, dtype=wm.dtype)
    timesteps_full = fm_sched.timesteps.to(device=device, dtype=wm.dtype)
    sigma_min, sigma_max = sigma_range
    mask = (sigmas_full >= sigma_min) & (sigmas_full <= sigma_max)
    valid_indices = mask.nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        valid_indices = torch.arange(
            len(sigmas_full) // 4, 3 * len(sigmas_full) // 4, device=device,
        )

    per_probe = []
    sigmas_used = []
    for _ in range(k_probes):
        if gen is not None:
            pick = torch.randint(0, valid_indices.numel(), (1,), generator=gen).item()
        else:
            pick = torch.randint(0, valid_indices.numel(), (1,)).item()
        idx_in_full = valid_indices[pick].item()
        sigma = sigmas_full[idx_in_full]
        t = timesteps_full[idx_in_full]
        sigmas_used.append(float(sigma.item()))

        if gen is not None:
            z = torch.randn(clean_full.shape, generator=gen).to(
                device=device, dtype=wm.dtype,
            )
        else:
            z = torch.randn_like(clean_full)

        # Per-frame σ: zero on the cond slot, σ on the target slots only.
        sigma_per_frame = torch.zeros(
            B, 1, NUM_LATENT_FRAMES, 1, 1, device=device, dtype=wm.dtype,
        )
        for idx in TARGET_LATENT_INDICES:
            sigma_per_frame[:, :, idx, :, :] = sigma
        noisy = clean_full * (1 - sigma_per_frame) + z * sigma_per_frame

        per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=wm.dtype)
        for idx in TARGET_LATENT_INDICES:
            per_frame_t[:, idx] = t

        video_tokens, time_emb, grid_sizes = wm._run_dit(
            noisy, per_frame_t, t5_embeddings, action_chunk, state, n_blocks=None,
        )
        pred = wm._dit_to_pred(video_tokens, time_emb, grid_sizes)

        # FM target velocity (world model's convention): v = noise - clean.
        target_v = z - clean_full
        pred_tgt = pred[:, :, TARGET_LATENT_INDICES]
        target_v_tgt = target_v[:, :, TARGET_LATENT_INDICES]
        residual = F.mse_loss(pred_tgt.float(), target_v_tgt.float()).item()
        per_probe.append(residual)

    return {
        "score": float(sum(per_probe) / len(per_probe)),
        "per_probe": per_probe,
        "sigma_used": sigmas_used,
        "k": k_probes,
    }
