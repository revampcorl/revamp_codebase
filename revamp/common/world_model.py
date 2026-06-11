"""WorldModel: dynamics + Q for offline-to-online VLA RL.

[BUG-4] post-refactor (see OPEN_ISSUES.md + ARCHITECTURE.md §3.7).

3-frame latent layout (T=3, 1 cond + 2 target), cam_concat along W:
  Index | Content                          | timestep | Notes
  ------|----------------------------------|----------|------------------------------
  [0]   | cam_concat cond  (raw[t-3..t])   |    0     | clean, VAE-encoded 4 raw frames
  [1]   | cam_concat target1 (raw[t+1..t+4])| t_rand  | noisy target, flow matching loss
  [2]   | cam_concat target2 (raw[t+5..t+8])| t_rand  | noisy target, flow matching loss

cam_concat = concat([cam_left, cam_right, cam_high], dim=W)
  3 cameras × W=320 = W_concat=960; H=384 unchanged.
  VAE encodes [B, 3, 12, 384, 960] → latent [B, 48, 3, 24, 60].

State and action injection (do NOT occupy latent slots):
  state_t        → StateContextMLP → 1 cross-attn token
                 → StateTimeMLP    → 1 instance offset, broadcast to all 3 latent frames
  action_chunk   → ActionContextMLP → 8 cross-attn tokens + learnable step emb
                 → ActionTimeMLP    → per-latent-frame offset [B, 3, dim]
                     latent[0] offset = 0  (cond unaffected)
                     latent[1] offset = mlp(action[0..3])
                     latent[2] offset = mlp(action[4..7])

Per-frame timestep (Forward A training):
  cond frames carry t=0, target frames carry the random training t.
  Loss is masked to TARGET_LATENT_INDICES = [1, 2] explicitly (do NOT
  rely on σ=0 implicit mask, see [BUG-4] §3).

Q forward (Stage 2 train + Stage 3 use):
  build_q_input_latent: latent[0] = cond VAE; latent[1..2] = randn placeholder.
  per_frame_timestep = [0, 999, 999] (aligns with Forward A main training dist).
  partial DiT (first q_dit_n_blocks=8) → pool latent[0] cam tokens → Q head.
  detach_dit controls whether Q backward flows through DiT:
    Stage 2 (train Q):  detach_dit=True  — saves ~30% backward FLOPs
    Stage 3 (QAM):      detach_dit=False — let ∇_a Q flow via DiT (motivation!)
  DiT params are PERMANENTLY frozen in both stages.

Forward B (future_state head) is REMOVED. Phase 2 imagination chain uses
action_chunk[-1] as next_state (see [BUG-4] §4 + [RES-11] D3 validation).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revamp.common.wan_model import WanVideoModel
from wan.utils.fm import FlowMatchScheduler
from wan.modules.model import sinusoidal_embedding_1d

from revamp.common.constants import (
    NUM_LATENT_FRAMES, COND_LATENT_IDX, TARGET_LATENT_INDICES,
    NUM_RAW_FRAMES_PER_SAMPLE, NUM_COND_RAW_FRAMES, NUM_TARGET_RAW_FRAMES,
    LATENT_CHANNELS, DEFAULT_CHUNK_LENGTH, DEFAULT_ACTION_DIM,
    DEFAULT_STATE_DIM, DEFAULT_Q_DIT_N_BLOCKS, DEFAULT_Q_HIDDEN_DIM,
    DEFAULT_Q_PLACEHOLDER_TIMESTEP, DEFAULT_Q_PLACEHOLDER_TYPE,
    STAGE_DYNAMICS_ONLY, STAGE_RL,
)
from revamp.common.action_chunk import (
    ActionContextMLP, ActionTimeMLP, ActionChunkEmbedder,
    StateContextMLP, StateTimeMLP,
)
from revamp.stage1_2_world_model.q_head import QEnsemble

logger = logging.getLogger(__name__)


class WorldModel(nn.Module):
    """Dynamics (WAN DiT) + chunk Q (partial-DiT features → MLP)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16

        # ---- WAN backbone ----
        self.video_model = WanVideoModel.from_pretrained(
            checkpoint_path=config.model.wan.checkpoint_path,
            vae_path=config.model.wan.vae_path,
            config_path=config.model.wan.config_path,
            precision=config.model.wan.precision,
        )
        self.device = next(self.video_model.parameters()).device

        self.num_latent_frames = NUM_LATENT_FRAMES                  # = 3
        self.latent_h = config.common.video_height // 16            # 384/16 = 24
        # NOTE: latent_w applies to per-camera (un-concatenated) W. After
        # concatenating 3 cameras along W, the latent's W' = 3 * latent_w.
        self.latent_w = config.common.video_width // 16             # 320/16 = 20
        self.latent_w_concat = self.latent_w * 3                    # 60
        self.latent_c = LATENT_CHANNELS

        self.dit_dim = self.video_model.wan_model.dim
        self.text_len = self.video_model.wan_model.text_len

        # ---- Shape params ----
        self.chunk_length = getattr(
            config.common, "action_chunk_length", DEFAULT_CHUNK_LENGTH,
        )
        self.action_dim = getattr(
            config.common, "action_dim", DEFAULT_ACTION_DIM,
        )
        self.state_dim = getattr(
            config.common, "state_dim", DEFAULT_STATE_DIM,
        )

        # ---- Action injection (cross-attn + per-latent-frame time-mod) ----
        self.action_context_mlp = ActionContextMLP(
            action_dim=self.action_dim,
            dim=self.dit_dim,
            hidden_dim=getattr(config.model, "action_context_hidden", 512),
            chunk_length=self.chunk_length,
        )
        self.action_time_mlp = ActionTimeMLP(
            chunk_length=self.chunk_length,
            action_dim=self.action_dim,
            dim=self.dit_dim,
            hidden_dim=getattr(config.model, "action_time_hidden", 512),
            num_latent_frames=self.num_latent_frames,
        )

        # ---- State injection (dual to action: ContextMLP + TimeMLP) ----
        self.state_context_mlp = StateContextMLP(
            state_dim=self.state_dim,
            dim=self.dit_dim,
            hidden_dim=getattr(config.model, "state_context_hidden", 512),
        )
        self.state_time_mlp = StateTimeMLP(
            state_dim=self.state_dim,
            dim=self.dit_dim,
            hidden_dim=getattr(config.model, "state_time_hidden", 512),
        )

        # ---- Q head ----
        self.q_dit_n_blocks = getattr(
            config.model, "q_dit_n_blocks", DEFAULT_Q_DIT_N_BLOCKS,
        )
        action_embed_dim = getattr(config.model, "action_embed_dim", 256)
        state_proj_dim = getattr(config.model, "state_proj_dim", 256)

        self.action_chunk_embedder = ActionChunkEmbedder(
            chunk_length=self.chunk_length,
            action_dim=self.action_dim,
            embed_dim=action_embed_dim,
        )
        self.state_proj = nn.Linear(self.state_dim, state_proj_dim)

        q_in_dim = self.dit_dim + state_proj_dim + action_embed_dim
        self.q_ensemble = QEnsemble(
            in_dim=q_in_dim,
            hidden_dim=getattr(config.model, "q_hidden_dim", DEFAULT_Q_HIDDEN_DIM),
            tau=getattr(config.training, "tau_polyak", 0.005),
        )

        # ---- Q forward placeholder config ----
        self.q_placeholder_timestep = getattr(
            config.model, "q_placeholder_timestep", DEFAULT_Q_PLACEHOLDER_TIMESTEP,
        )
        self.q_placeholder_type = getattr(
            config.model, "q_placeholder_type", DEFAULT_Q_PLACEHOLDER_TYPE,
        )
        assert self.q_placeholder_type in ("randn", "zeros"), (
            f"q_placeholder_type must be 'randn' or 'zeros' (got {self.q_placeholder_type})"
        )

        # ---- Loss weights ----
        # future_state_loss removed (Forward B deleted). loss_weight_future_state
        # kept for backward-compat config files but unused.
        self.loss_weight_future_cam = getattr(
            config.training, "loss_weight_future_cam", 1.0,
        )
        self.loss_weight_q = getattr(config.training, "loss_weight_q", 1.0)

        # ---- Flow matching scheduler ----
        self.fm_scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000,
        )
        self.fm_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"WorldModel initialized: {total_p / 1e9:.2f}B total, "
            f"{train_p / 1e9:.2f}B trainable. "
            f"Latent[{self.latent_c},{self.num_latent_frames},"
            f"{self.latent_h},{self.latent_w_concat}] (cam_concat-W) "
            f"DiT dim={self.dit_dim}, Q n_blocks={self.q_dit_n_blocks}"
        )

    # ============================================================
    # VAE
    # ============================================================

    def encode_video_segment(self, video: torch.Tensor) -> torch.Tensor:
        """[B, 3, T_raw, H, W] RGB in [0,1] → [B, 48, T_latent, H', W'] latent.

        VAE 4× temporal + 16× spatial compress. T_latent = (T_raw - 1) // 4 + 1.
        For our standard 12-frame segment: T_latent = (12-1)//4+1 = 3.
        For Q forward cond-only 4-frame: T_latent = (4-1)//4+1 = 1.
        """
        video = video.to(dtype=self.dtype)
        video_normalized = video * 2.0 - 1.0
        with torch.no_grad():
            latents = self.video_model.encode_video(video_normalized)
        return latents.to(dtype=self.dtype)

    def decode_cam_latents(self, cam_latents: torch.Tensor) -> torch.Tensor:
        """[B, 48, T, H', W'] latent → [B, 3, T_raw, H, W] RGB in [-1,1]."""
        with torch.no_grad():
            return self.video_model.decode_video(cam_latents.to(self.dtype))

    def decode_target_latents_with_cond(
        self,
        cond_latent: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target RGB frames with the preceding condition latent.

        Wan VAE decodes a temporal latent sequence of length T to
        ``1 + 4 * (T - 1)`` RGB frames. Decoding the two target latents alone
        therefore yields only 5 frames because the first target latent is
        treated as the initial VAE chunk. Prepended with the condition latent,
        the 3-latent sequence decodes to 9 frames; dropping the first condition
        frame leaves the 8 target frames for the action chunk horizon.
        """
        full_latents = torch.cat([cond_latent, target_latents], dim=2)
        decoded = self.decode_cam_latents(full_latents)
        return decoded[:, :, 1:]

    @staticmethod
    def concat_cams_along_w(
        cam_left: torch.Tensor,
        cam_right: torch.Tensor,
        cam_high: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate 3 cam tensors along the W axis (last dim).

        Inputs:  each [B, 3, T, H, W]  (or [B, 3, H, W] if single frame)
        Output:  [B, 3, T, H, 3*W]
        """
        return torch.cat([cam_left, cam_right, cam_high], dim=-1)

    # ============================================================
    # Build latent sequences
    # ============================================================

    def build_full_latent_sequence(
        self,
        cam_left_segment: torch.Tensor,        # [B, 3, 12, H, W]
        cam_right_segment: torch.Tensor,       # [B, 3, 12, H, W]
        cam_high_segment: torch.Tensor,        # [B, 3, 12, H, W]
    ) -> torch.Tensor:
        """Full T=3 layout: VAE-encode the 12-frame cam_concat segment.

        Returns clean latent of shape [B, 48, 3, latent_h, 3*latent_w].
          latent[:, :, 0] = VAE encode raw[t-3..t]   (cond)
          latent[:, :, 1] = VAE encode raw[t+1..t+4] (target Q1)
          latent[:, :, 2] = VAE encode raw[t+5..t+8] (target Q2)

        VAE 4× temporal compress maps 12 raw → 3 latent frames in one pass.
        State and action are NOT in the latent (they are injected via MLPs
        through cross-attn + time modulation in _build_context and
        _get_time_embedding).
        """
        cam_concat = self.concat_cams_along_w(
            cam_left_segment, cam_right_segment, cam_high_segment,
        )                                                   # [B, 3, 12, H, 3W]
        latent = self.encode_video_segment(cam_concat)      # [B, 48, 3, H', 3W']
        return latent

    def build_q_input_latent(
        self,
        cam_left_cond: torch.Tensor,           # [B, 3, 4, H, W] or [B, 3, H, W]
        cam_right_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
        noise_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build Q forward input: T=3 latent with cond + 2 randn placeholders.

        latent[0] = VAE encode the first condition latent from the supplied
                   camera segment. Prefer the full 12 raw frames raw[t-3..t+8]
                   so the video VAE sees the same temporal length as Stage 2
                   dynamics training; legacy 4-frame or single-frame inputs
                   are still accepted when the VAE backend supports them.
        latent[1..2] = randn placeholder (or zeros if q_placeholder_type=='zeros')
        per_frame_timestep = [0, q_placeholder_timestep, q_placeholder_timestep]

        Inputs must already be cam_concat-W ready in the channel/H/W axes;
        accept 12-frame segments (preferred), 4-frame cond segments, or single
        raw frames (legacy/inference fallback). When the VAE produces more
        than one latent frame, Q uses only latent[0] as the condition frame and
        replaces latent[1..2] with placeholders. This avoids short temporal
        VAE inputs in Stage 3 while preserving the Q head's T=3 layout.

        Returns:
          latent: [B, 48, 3, H', 3W']
          per_frame_timestep: [B, 3]
        """
        # Promote single-frame inputs to T_raw=1 for video encode.
        if cam_left_cond.dim() == 4:
            cam_left_cond = cam_left_cond.unsqueeze(2)
        if cam_right_cond.dim() == 4:
            cam_right_cond = cam_right_cond.unsqueeze(2)
        if cam_high_cond.dim() == 4:
            cam_high_cond = cam_high_cond.unsqueeze(2)

        cam_concat = self.concat_cams_along_w(
            cam_left_cond, cam_right_cond, cam_high_cond,
        )                                                   # [B, 3, T_raw, H, 3W]
        cond_latent = self.encode_video_segment(cam_concat)  # [B, 48, T_latent_cond, H', 3W']
        B = cond_latent.shape[0]
        device = cond_latent.device

        if cond_latent.shape[2] < 1:
            raise ValueError(
                f"build_q_input_latent expected at least one latent frame, "
                f"got {cond_latent.shape[2]} (raw input shape {cam_concat.shape})"
            )
        cond_latent = cond_latent[:, :, :1]

        # Build placeholder for the 2 target frames.
        placeholder_shape = (
            B, self.latent_c, len(TARGET_LATENT_INDICES),
            self.latent_h, self.latent_w_concat,
        )
        if self.q_placeholder_type == "randn":
            if noise_generator is not None:
                placeholder = torch.randn(
                    *placeholder_shape, dtype=self.dtype, device=device,
                    generator=noise_generator,
                )
            else:
                placeholder = torch.randn(
                    *placeholder_shape, dtype=self.dtype, device=device,
                )
        else:  # zeros are kept only as an out-of-distribution ablation hook
            placeholder = torch.zeros(
                *placeholder_shape, dtype=self.dtype, device=device,
            )

        latent = torch.cat([cond_latent, placeholder], dim=2)  # [B, 48, 3, H', 3W']

        # per-frame timestep: cond=0, target=q_placeholder_timestep (999)
        per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=self.dtype)
        for idx in TARGET_LATENT_INDICES:
            per_frame_t[:, idx] = float(self.q_placeholder_timestep)

        return latent, per_frame_t

    # ============================================================
    # Time embedding (per-frame) + state/action time-mod injection
    # ============================================================

    def _get_time_embedding(
        self,
        per_frame_timesteps: torch.Tensor,   # [B, T_latent]
        seq_len: int,
        action_chunk: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (time_emb, time_proj) with state/action time-mod offsets.

        state_offset: instance-level, broadcast to all latent frames.
        action_offset: per-latent-frame (latent[0]=0, latent[1..2]=action segments).

        Returns:
            time_emb:  [B, seq_len, dim]
            time_proj: [B, seq_len, 6, dim]
        """
        B, T_latent = per_frame_timesteps.shape
        assert seq_len % T_latent == 0, (
            f"seq_len {seq_len} not divisible by T_latent {T_latent}"
        )
        tokens_per_frame = seq_len // T_latent
        # Expand each frame's timestep to all its spatial tokens.
        timesteps = (
            per_frame_timesteps.unsqueeze(2)
            .expand(B, T_latent, tokens_per_frame)
            .reshape(B, seq_len)
        )

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_flat = timesteps.flatten().to(dtype=torch.float32)
            freq_dim = self.video_model.wan_model.freq_dim
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat)
                .unflatten(0, (B, seq_len))
                .float()
            )                                                # [B, seq_len, dim]

            # State offset: [B, dim] → broadcast across all tokens.
            if state is not None:
                state_offset = self.state_time_mlp(state).float()         # [B, dim]
                t_emb = t_emb + state_offset.unsqueeze(1)                 # broadcast

            # Action offset: per-latent-frame [B, T_latent, dim] → expand to tokens.
            if action_chunk is not None:
                action_offset_per_frame = self.action_time_mlp(action_chunk).float()
                # [B, T_latent, dim] → [B, T_latent, tokens_per_frame, dim] → [B, seq_len, dim]
                action_offset_per_token = (
                    action_offset_per_frame.unsqueeze(2)
                    .expand(B, T_latent, tokens_per_frame, self.dit_dim)
                    .reshape(B, seq_len, self.dit_dim)
                )
                t_emb = t_emb + action_offset_per_token

            t_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(
                2, (6, self.dit_dim)
            )
        return t_emb, t_proj

    # ============================================================
    # Context (text + state + action)
    # ============================================================

    def _build_context(
        self,
        t5_embeddings: torch.Tensor,
        action_chunk: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """text(t5) || state_context(1 token) || action_context(H tokens)."""
        text_len = self.text_len
        if t5_embeddings.shape[1] < text_len:
            pad = t5_embeddings.new_zeros(
                t5_embeddings.shape[0],
                text_len - t5_embeddings.shape[1],
                t5_embeddings.shape[2],
            )
            t5_embeddings = torch.cat([t5_embeddings, pad], dim=1)
        elif t5_embeddings.shape[1] > text_len:
            t5_embeddings = t5_embeddings[:, :text_len]

        t5_embeddings = t5_embeddings.to(dtype=self.dtype)
        text_context = self.video_model.wan_model.text_embedding(t5_embeddings)

        parts = [text_context]
        if state is not None:
            state_context = self.state_context_mlp(state).to(dtype=text_context.dtype)
            parts.append(state_context)
        if action_chunk is not None:
            action_context = self.action_context_mlp(action_chunk).to(dtype=text_context.dtype)
            parts.append(action_context)
        return torch.cat(parts, dim=1)

    # ============================================================
    # Core DiT forward (supports partial)
    # ============================================================

    def _run_dit(
        self,
        latent: torch.Tensor,                 # [B, C, T, H', W']
        per_frame_timesteps: torch.Tensor,    # [B, T]
        t5_embeddings: torch.Tensor,
        action_chunk: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        n_blocks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DiT (or first n_blocks). Returns (video_tokens, time_emb, grid_sizes)."""
        wan = self.video_model.wan_model

        patched = wan.patch_embedding(latent.to(dtype=self.dtype))
        video_tokens = patched.flatten(2).transpose(1, 2)
        seq_len = video_tokens.shape[1]
        B = patched.shape[0]

        _, _, T_p, H_p, W_p = patched.shape
        grid_sizes = (
            torch.tensor([T_p, H_p, W_p], dtype=torch.long, device=patched.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        time_emb, time_proj = self._get_time_embedding(
            per_frame_timesteps, seq_len, action_chunk, state,
        )
        context = self._build_context(t5_embeddings, action_chunk, state)

        freqs = wan.freqs
        if freqs.device != video_tokens.device:
            freqs = freqs.to(video_tokens.device)
        seq_lens = torch.full(
            (B,), seq_len, dtype=torch.long, device=video_tokens.device,
        )

        blocks_to_run = wan.blocks if n_blocks is None else wan.blocks[:n_blocks]

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for block in blocks_to_run:
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    modulation = (block.modulation.unsqueeze(0) + time_proj).chunk(6, dim=2)
                norm_x = (
                    block.norm1(video_tokens).float() * (1 + modulation[1].squeeze(2))
                    + modulation[0].squeeze(2)
                )
                attn_out = block.self_attn(norm_x, seq_lens, grid_sizes, freqs)
                video_tokens = video_tokens + attn_out * modulation[2].squeeze(2)

                cross_out = block.cross_attn(
                    block.norm3(video_tokens), context, None,
                )
                video_tokens = video_tokens + cross_out

                ffn_in = (
                    block.norm2(video_tokens).float() * (1 + modulation[4].squeeze(2))
                    + modulation[3].squeeze(2)
                )
                ffn_out = block.ffn(ffn_in)
                video_tokens = video_tokens + ffn_out * modulation[5].squeeze(2)

        return video_tokens, time_emb, grid_sizes

    def _dit_to_pred(
        self,
        video_tokens: torch.Tensor,
        time_emb: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        wan = self.video_model.wan_model
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            pred = wan.head(video_tokens, time_emb)
            pred = wan.unpatchify(pred, grid_sizes)
            pred = torch.stack([u for u in pred], dim=0)
        return pred.to(dtype=self.dtype)

    # ============================================================
    # Q forward (partial DiT + pool latent[0] + MLP)
    # ============================================================

    def _q_features(
        self,
        state: torch.Tensor,
        cam_left_cond: torch.Tensor,       # [B, 3, 4, H, W] or [B, 3, H, W]
        cam_right_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        detach_dit: bool = True,
        noise_generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Q forward: T=3 layout with randn placeholder, partial DiT first
        q_dit_n_blocks blocks, pool latent[0] cam tokens, concat with
        state_proj + action_embed.

        detach_dit controls whether Q backward flows through DiT:
          Stage 2 (train Q):  True  — saves ~30% backward FLOPs (DiT frozen anyway)
          Stage 3 (QAM):      False — let ∇_a Q flow via DiT (project motivation!)
        DiT params should be set requires_grad=False from Stage 2 onward
        regardless of detach_dit (the two flags control different things:
        detach_dit affects the autograd graph, requires_grad affects parameter
        updates).
        """
        latent, per_frame_t = self.build_q_input_latent(
            cam_left_cond, cam_right_cond, cam_high_cond,
            noise_generator=noise_generator,
        )

        video_tokens, _time_emb, _grid = self._run_dit(
            latent, per_frame_t, t5_embeddings, action_chunk, state,
            n_blocks=self.q_dit_n_blocks,
        )
        # video_tokens: [B, T_p * H_p * W_p, dim] where T_p = NUM_LATENT_FRAMES = 3
        # patch_size temporal = 1 so T_p = T_latent = 3.
        B = video_tokens.shape[0]
        seq_len = video_tokens.shape[1]
        tokens_per_frame = seq_len // NUM_LATENT_FRAMES
        per_frame_tokens = video_tokens.reshape(
            B, NUM_LATENT_FRAMES, tokens_per_frame, self.dit_dim,
        )
        # Pool only the cond frame (latent[0]) cam tokens. latent[1..2] are
        # randn placeholders; their tokens are "noise representations" with
        # no semantic value for Q evaluation.
        cam_pool = per_frame_tokens[:, COND_LATENT_IDX].mean(dim=1)        # [B, dim]

        if detach_dit:
            cam_pool = cam_pool.detach()

        state_proj = self.state_proj(state.float())
        action_embed = self.action_chunk_embedder(action_chunk)
        return torch.cat([cam_pool.float(), state_proj, action_embed], dim=-1)

    def forward_q(
        self,
        state: torch.Tensor,
        cam_left_cond: torch.Tensor,
        cam_right_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        detach_dit: bool = True,
        noise_generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._q_features(
            state, cam_left_cond, cam_right_cond, cam_high_cond,
            action_chunk, t5_embeddings,
            detach_dit=detach_dit, noise_generator=noise_generator,
        )
        return self.q_ensemble(feats)

    # NOTE: @torch.no_grad() decorator REMOVED ([BUG-4-1] fix).
    # Stage 2 callers must wrap with `with torch.no_grad():` themselves;
    # Stage 3 (QAM) needs gradient flow through this function for ∇_a Q.
    def forward_q_target_min(
        self,
        state: torch.Tensor,
        cam_left_cond: torch.Tensor,
        cam_right_cond: torch.Tensor,
        cam_high_cond: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        detach_dit: bool = True,
        noise_generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        feats = self._q_features(
            state, cam_left_cond, cam_right_cond, cam_high_cond,
            action_chunk, t5_embeddings,
            detach_dit=detach_dit, noise_generator=noise_generator,
        )
        return self.q_ensemble.target_min(feats)

    # ============================================================
    # Training step (stage gating)
    # ============================================================

    def training_step(
        self,
        # Cam segments [B, 3, 12, H, W] for the new T=3 layout.
        cam_left_segment: torch.Tensor,
        cam_right_segment: torch.Tensor,
        cam_high_segment: torch.Tensor,
        # Per-sample state and action.
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        # Stage 2 only:
        stage: int = STAGE_DYNAMICS_ONLY,
        chunk_rewards: Optional[torch.Tensor] = None,        # [B, H]
        bootstrap_q_target: Optional[torch.Tensor] = None,   # [B]
        done_chunk: Optional[torch.Tensor] = None,           # [B, H]
        gamma: float = 0.95,
        sample_weight_q: Optional[torch.Tensor] = None,      # [B]
        freeze_dit: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        from revamp.stage1_2_world_model.q_head import compute_td_target, q_loss

        B = state.shape[0]
        device = state.device
        if B == 0:
            return self._dummy_output(device)

        if freeze_dit:
            # Stage 2 FQE with DiT frozen: skip dynamics forward entirely.
            zero = torch.zeros((), device=device, dtype=torch.float32)
            future_cam_loss = zero
            dynamics_loss = zero
            sigma_target_mean = 0.0
        else:
            # ---- Build full clean latent (T=3) ----
            clean_latent = self.build_full_latent_sequence(
                cam_left_segment, cam_right_segment, cam_high_segment,
            )                                              # [B, 48, 3, H', 3W']

            # ---- Per-frame timestep: cond=0, targets=random shared per sample ----
            timestep_ids = torch.randint(
                0, self.fm_scheduler.num_train_timesteps, (B,), device="cpu",
            )
            timestep_target = self.fm_scheduler.timesteps[timestep_ids].to(
                dtype=self.dtype, device=device,
            )                                                    # [B]
            sigma_target = self.fm_scheduler.sigmas[timestep_ids].to(
                dtype=self.dtype, device=device,
            )                                                    # [B]

            per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=self.dtype)
            for idx in TARGET_LATENT_INDICES:
                per_frame_t[:, idx] = timestep_target

            # Per-frame sigma: zero on cond, σ on target.
            sigma_per_frame = torch.zeros(
                B, 1, NUM_LATENT_FRAMES, 1, 1, device=device, dtype=self.dtype,
            )
            for idx in TARGET_LATENT_INDICES:
                sigma_per_frame[:, :, idx, :, :] = sigma_target.view(B, 1, 1, 1)

            noise = torch.randn_like(clean_latent, dtype=self.dtype)
            noisy_latent = clean_latent * (1 - sigma_per_frame) + noise * sigma_per_frame

            # ---- Dynamics forward ----
            video_tokens, time_emb, grid_sizes = self._run_dit(
                noisy_latent, per_frame_t, t5_embeddings, action_chunk, state,
                n_blocks=None,
            )
            pred = self._dit_to_pred(video_tokens, time_emb, grid_sizes)

            # ---- Velocity flow matching MSE, EXPLICITLY masked to target frames ----
            # ([BUG-4] §3: do NOT rely on σ=0 implicit mask. Explicit slice along
            # T axis ensures cond frames don't pull gradient toward an arbitrary
            # noise-clean target.)
            target = (noise - clean_latent)                        # full target
            pred_target = pred[:, :, TARGET_LATENT_INDICES]        # [B, 48, 2, H', 3W']
            target_target = target[:, :, TARGET_LATENT_INDICES]    # same

            future_cam_loss = F.mse_loss(pred_target.float(), target_target.float())

            dynamics_loss = self.loss_weight_future_cam * future_cam_loss
            sigma_target_mean = float(sigma_target.float().mean().item())

        # ---- Stage gating: Q TD loss ----
        ql = torch.zeros((), device=device)
        q1_mean = q2_mean = 0.0
        if (
            stage == STAGE_RL
            and chunk_rewards is not None
            and bootstrap_q_target is not None
        ):
            # Q forward uses cond-only inputs (raw[t-3..t] segment slice).
            cam_left_cond = cam_left_segment[:, :, :NUM_COND_RAW_FRAMES]
            cam_right_cond = cam_right_segment[:, :, :NUM_COND_RAW_FRAMES]
            cam_high_cond = cam_high_segment[:, :, :NUM_COND_RAW_FRAMES]

            q1, q2 = self.forward_q(
                state, cam_left_cond, cam_right_cond, cam_high_cond,
                action_chunk, t5_embeddings,
                detach_dit=True,    # Stage 2 keeps DiT detached
            )
            y = compute_td_target(
                chunk_rewards,
                bootstrap_q_target,
                gamma,
                done_chunk=done_chunk,
            )
            ql = q_loss(q1, q2, y, sample_weight=sample_weight_q)
            q1_mean = float(q1.mean().item())
            q2_mean = float(q2.mean().item())

        total_loss = dynamics_loss + (
            self.loss_weight_q * ql if stage == STAGE_RL else 0.0
        )

        if return_dict:
            return {
                "total_loss": total_loss,
                "dynamics_loss": dynamics_loss,
                "future_cam_loss": future_cam_loss,
                "q_loss": ql,
                "q1_mean": q1_mean,
                "q2_mean": q2_mean,
                "sigma_mean": sigma_target_mean,
            }
        return total_loss

    def _dummy_output(self, device):
        z = lambda: torch.zeros([], device=device, requires_grad=True)
        return {
            "total_loss": z(),
            "dynamics_loss": z(),
            "future_cam_loss": z(),
            "q_loss": z(),
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "sigma_mean": 0.0,
        }

    # ============================================================
    # Inference: predict_chunk (used by Phase 2 imagination)
    # ============================================================

    @torch.no_grad()
    def predict_chunk_from_cond_latent(
        self,
        cond_latent: torch.Tensor,          # [B, 48, 1, H', 3W']
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        decode_rgb: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """One chunk-step from an already encoded condition latent.

        This is the latent-space autoregressive path: it avoids decoding a
        predicted chunk to RGB and then re-encoding it just to build the next
        condition. The caller is responsible for carrying a condition latent
        whose temporal dimension is exactly 1.
        """
        if cond_latent.dim() != 5 or cond_latent.shape[2] != 1:
            raise ValueError(
                "cond_latent must have shape [B, C, 1, H, W], "
                f"got {tuple(cond_latent.shape)}"
            )

        B = state.shape[0]
        device = state.device
        cond_latent = cond_latent.to(device=device, dtype=self.dtype)

        target_shape = (
            B, self.latent_c, len(TARGET_LATENT_INDICES),
            self.latent_h, self.latent_w_concat,
        )
        target_noise = torch.randn(*target_shape, device=device, dtype=self.dtype)
        latent = torch.cat([cond_latent, target_noise], dim=2)

        self.fm_scheduler.set_timesteps(num_inference_steps)
        for t in self.fm_scheduler.timesteps:
            per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=self.dtype)
            for idx in TARGET_LATENT_INDICES:
                per_frame_t[:, idx] = t

            video_tokens, time_emb, grid_sizes = self._run_dit(
                latent, per_frame_t, t5_embeddings, action_chunk, state,
                n_blocks=None,
            )
            pred = self._dit_to_pred(video_tokens, time_emb, grid_sizes)

            for idx in TARGET_LATENT_INDICES:
                latent[:, :, idx] = self.fm_scheduler.step(
                    pred[:, :, idx], t, latent[:, :, idx],
                )

        next_state = action_chunk[:, -1, :]
        target_latents = latent[:, :, TARGET_LATENT_INDICES]
        out: Dict[str, torch.Tensor] = {
            "next_state": next_state,
            "next_cam_concat_latent": target_latents,
        }

        if decode_rgb:
            decoded = self.decode_target_latents_with_cond(
                cond_latent=latent[:, :, COND_LATENT_IDX : COND_LATENT_IDX + 1],
                target_latents=target_latents,
            )
            W = decoded.shape[-1] // 3
            out["next_cam_left"] = decoded[..., :W]
            out["next_cam_right"] = decoded[..., W:2 * W]
            out["next_cam_high"] = decoded[..., 2 * W:]

        return out

    @torch.no_grad()
    def predict_chunk(
        self,
        state: torch.Tensor,
        cam_left_segment: torch.Tensor,     # [B, 3, 12, H, W]  (or 4-cond raw frame seg)
        cam_right_segment: torch.Tensor,
        cam_high_segment: torch.Tensor,
        action_chunk: torch.Tensor,
        t5_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        decode_rgb: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """One chunk-step world model forward (Phase 2 imagination).

        Inputs:  s_t = (state, 3 cam_segments covering raw[t-3..t+8]),
                 action chunk ā_t, language.
                 NOTE: cam_segments must include at least raw[t-3..t]; the
                 target portion [t+1..t+8] is overwritten by noise + denoised.
        Outputs:
          next_state: action_chunk[-1]                  ([BUG-4] Forward B removed)
          next_cam_{left,right,high}_latent: denoised target latent[1..2]
          next_cam_{left,right,high} (RGB): decoded if decode_rgb=True
          q1, q2: Q evaluated at (s_t, ā_t)
        """
        B = state.shape[0]
        device = state.device

        # Build full T=3 latent from cam_segments. We will OVERWRITE the target
        # frames (latent[1..2]) with randn and denoise them.
        clean_latent = self.build_full_latent_sequence(
            cam_left_segment, cam_right_segment, cam_high_segment,
        )                                                  # [B, 48, 3, H', 3W']
        latent = clean_latent.clone()
        for idx in TARGET_LATENT_INDICES:
            latent[:, :, idx : idx + 1] = torch.randn_like(
                latent[:, :, idx : idx + 1]
            )

        self.fm_scheduler.set_timesteps(num_inference_steps)
        for t in self.fm_scheduler.timesteps:
            per_frame_t = torch.zeros(B, NUM_LATENT_FRAMES, device=device, dtype=self.dtype)
            for idx in TARGET_LATENT_INDICES:
                per_frame_t[:, idx] = t

            video_tokens, time_emb, grid_sizes = self._run_dit(
                latent, per_frame_t, t5_embeddings, action_chunk, state,
                n_blocks=None,
            )
            pred = self._dit_to_pred(video_tokens, time_emb, grid_sizes)

            for idx in TARGET_LATENT_INDICES:
                latent[:, :, idx] = self.fm_scheduler.step(
                    pred[:, :, idx], t, latent[:, :, idx],
                )

        # ---- Extract outputs ----
        # Legacy rollout shortcut for configs where action/state semantics match.
        # Contact-aware training paths use real next-state fields from the dataset.
        next_state = action_chunk[:, -1, :]                # [B, action_dim]

        # The two target latent frames cover raw[t+1..t+4] and raw[t+5..t+8].
        # Decode and split along W to recover each camera (W_concat = 3 * W).
        target_latents = latent[:, :, TARGET_LATENT_INDICES]   # [B, 48, 2, H', 3W']

        out: Dict[str, torch.Tensor] = {
            "next_state": next_state,
            "next_cam_concat_latent": target_latents,           # latent[1..2]
        }

        if decode_rgb:
            # Decode [cond, target1, target2] and drop the first condition
            # frame, yielding 8 target RGB frames for the chunk horizon.
            decoded = self.decode_target_latents_with_cond(
                cond_latent=latent[:, :, COND_LATENT_IDX : COND_LATENT_IDX + 1],
                target_latents=target_latents,
            )                                                   # [B, 3, 8, H, 3W]
            # Split W axis back into 3 cameras.
            W = decoded.shape[-1] // 3
            out["next_cam_left"] = decoded[..., :W]
            out["next_cam_right"] = decoded[..., W:2 * W]
            out["next_cam_high"] = decoded[..., 2 * W:]

        # ---- Q at (s_t, ā_t), using cond slice of cam_segments ----
        cam_left_cond = cam_left_segment[:, :, :NUM_COND_RAW_FRAMES]
        cam_right_cond = cam_right_segment[:, :, :NUM_COND_RAW_FRAMES]
        cam_high_cond = cam_high_segment[:, :, :NUM_COND_RAW_FRAMES]
        q1, q2 = self.forward_q(
            state, cam_left_cond, cam_right_cond, cam_high_cond,
            action_chunk, t5_embeddings,
            detach_dit=True,
        )
        out["q1"] = q1
        out["q2"] = q2

        return out
