"""Action and state injection MLPs for the WorldModel.

[BUG-4] post-refactor: action_chunk and state are injected via dedicated MLPs
that do NOT occupy any latent T slot. The latent T axis is reserved entirely
for camera frames (3 cameras concatenated along W → 1 cam_concat stream of
T=3 latent frames).

Four injection MLPs (closely modeled on RLinf/diffsynth-studio
`action_mlp1` / `action_mlp2`, plus state dual to action):

  1. ActionContextMLP — chunk → per-step context tokens [B, H_chunk, dim].
     Includes a learnable per-step positional embedding so cross-attention can
     distinguish front-of-chunk vs back-of-chunk actions explicitly. Output is
     appended to T5 text-embedding context; every DiT block's cross-attention
     attends jointly over (text + state + action).

  2. ActionTimeMLP — chunk → per-latent-frame time-embedding offset
     [B, T_latent=3, dim]. **NEW per-frame design**:
       latent[0] offset = 0                       (cond clean, no action mod)
       latent[1] offset = mlp(action[0..3])       (front half of chunk)
       latent[2] offset = mlp(action[4..7])       (back half of chunk)
     This gives each target frame a chunk-segment-specific FiLM signal aligned
     with the raw frames it covers (latent[1]=raw[t+1..t+4],
     latent[2]=raw[t+5..t+8]).

  3. StateContextMLP — state → 1 cross-attn context token [B, 1, dim].

  4. StateTimeMLP — state → 1 instance offset [B, dim], broadcast to all
     latent frames in the time embedding.

Also exports `ActionChunkEmbedder`, the compact action embedder used by the Q
head's MLP top (separate, shallow path; see [RES-10] D1 for whether this short
path C dominates ∇_a Q in Stage 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from revamp.common.constants import (
    DEFAULT_CHUNK_LENGTH, DEFAULT_ACTION_DIM, DEFAULT_STATE_DIM,
    NUM_LATENT_FRAMES, CHUNK_SEGMENT_BOUNDARY,
)


class ActionContextMLP(nn.Module):
    """Encode action chunk into per-step context tokens with positional embedding.

    Output is appended to the T5 text-embedding context, so DiT cross-attention
    treats action as additional conditioning tokens. The learnable step
    embedding lets cross-attention distinguish the 8 tokens' chunk position
    explicitly (required because the per-latent-frame ActionTimeMLP segments
    the chunk by index).

    Shape:
        in : [B, H_chunk, action_dim]
        out: [B, H_chunk, dim]
    """

    def __init__(
        self,
        action_dim: int = DEFAULT_ACTION_DIM,
        dim: int = 3072,
        hidden_dim: int = 512,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Learnable per-step positional embedding (one per chunk step).
        self.step_emb = nn.Parameter(torch.zeros(chunk_length, dim))
        nn.init.normal_(self.step_emb, std=0.02)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.dim() == 2:
            action_chunk = action_chunk.unsqueeze(1)
        x = self.net(action_chunk)               # [B, H_chunk, dim]
        return x + self.step_emb.unsqueeze(0)    # broadcast step emb [B, H_chunk, dim]


class ActionTimeMLP(nn.Module):
    """Encode the action chunk into per-latent-frame time-embedding offsets.

    Per-latent-frame design (RLinf `action_mlp2` style + chunk-aligned):
      latent[0] offset = 0  (cond frame, unaffected by future action)
      latent[1] offset = mlp(action[0..CHUNK_SEGMENT_BOUNDARY-1])
                         covers raw[t+1..t+4]
      latent[2] offset = mlp(action[CHUNK_SEGMENT_BOUNDARY..H-1])
                         covers raw[t+5..t+8]

    The offsets are added to per-frame time embeddings, then run through WAN's
    `time_projection` 6-way unflatten, modulating every DiT block's pre-norm
    scale/shift per frame.

    Shape:
        in : [B, H_chunk, action_dim]
        out: [B, NUM_LATENT_FRAMES, dim]
    """

    def __init__(
        self,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        action_dim: int = DEFAULT_ACTION_DIM,
        dim: int = 3072,
        hidden_dim: int = 512,
        num_latent_frames: int = NUM_LATENT_FRAMES,
        chunk_segment_boundary: int = CHUNK_SEGMENT_BOUNDARY,
    ):
        super().__init__()
        self.chunk_length = chunk_length
        self.num_latent_frames = num_latent_frames
        self.chunk_segment_boundary = chunk_segment_boundary

        # Each target latent frame uses half the chunk (4 steps × 14-D = 56-D)
        front_in = chunk_segment_boundary * action_dim
        back_in = (chunk_length - chunk_segment_boundary) * action_dim
        assert front_in == back_in, (
            f"per-segment MLP expects equal-size chunks "
            f"(got front={front_in} back={back_in}); "
            f"non-symmetric splits require separate weights."
        )

        # Shared MLP across the two target segments (same input size).
        self.net = nn.Sequential(
            nn.Linear(front_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        B = action_chunk.shape[0]
        boundary = self.chunk_segment_boundary

        # Split chunk into front (latent[1]) and back (latent[2]).
        front = action_chunk[:, :boundary].reshape(B, -1)        # [B, boundary*action_dim]
        back = action_chunk[:, boundary:].reshape(B, -1)         # [B, (H-boundary)*action_dim]

        offset_front = self.net(front)                            # [B, dim]
        offset_back = self.net(back)                              # [B, dim]
        zero = torch.zeros_like(offset_front)                     # latent[0] cond, no action mod

        # Layout: [latent[0]=0, latent[1]=front, latent[2]=back]
        # Assumes num_latent_frames == 3; if you change this, also change the layout.
        assert self.num_latent_frames == 3, (
            f"ActionTimeMLP currently assumes NUM_LATENT_FRAMES=3 "
            f"(got {self.num_latent_frames}); revise per-segment split for other layouts."
        )
        return torch.stack([zero, offset_front, offset_back], dim=1)  # [B, 3, dim]


class StateContextMLP(nn.Module):
    """Encode current state into a single cross-attn context token.

    State is a single-instant snapshot (no temporal structure), so 1 token is
    sufficient (a 3072-D token easily holds a 14-D vector's information).

    Shape:
        in : [B, state_dim]
        out: [B, 1, dim]
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        dim: int = 3072,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).unsqueeze(1)   # [B, 1, dim]


class StateTimeMLP(nn.Module):
    """Encode state into an instance-level time-embedding offset.

    State is a single-instant value, broadcast to all latent frames (no
    per-frame split). The offset is added to time embeddings before WAN's
    `time_projection`, modulating every DiT block's pre-norm scale/shift.

    Shape:
        in : [B, state_dim]
        out: [B, dim]   (caller broadcasts to all latent frames)
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        dim: int = 3072,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)   # [B, dim]


class ActionChunkEmbedder(nn.Module):
    """Compact action-chunk embedder for the Q head's MLP top.

    Q's q_input = [cam_pool, state_proj, action_embed]. The DiT-pooled features
    (cam_pool) carry visual+language+action information jointly because
    `ActionContextMLP` / `ActionTimeMLP` already injected the action into DiT,
    but a separate compact action embedding at the MLP layer gives Q an
    explicit short path for action sensitivity.

    NOTE: This is "path C" in the Stage 3 ∇_a Q backward (see [RES-10] D1).
    If diagnostic shows path C dominates path A/B (the DiT-mediated paths),
    consider removing this embedder from the Q head input to force Q to
    learn through DiT.

    Shape:
        in : [B, H_chunk, action_dim]
        out: [B, embed_dim]
    """

    def __init__(
        self,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        action_dim: int = DEFAULT_ACTION_DIM,
        embed_dim: int = 256,
    ):
        super().__init__()
        in_dim = chunk_length * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.dim() == 3:
            B = action_chunk.shape[0]
            flat = action_chunk.reshape(B, -1)
        else:
            flat = action_chunk
        return self.net(flat)
