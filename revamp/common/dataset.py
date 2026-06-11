"""WorldModelDataset: extends VivaDataset to load action chunks, future
cameras, per-step ViVa values (from cache), FQE-style next-chunk labels for
Stage 2 Q training, and (post-[BUG-4]) 12-frame raw camera video segments
for the new T=3 latent layout.

Per sample, returns:
  - state, current cameras (single frame), future_state, future cameras
    (single frame) — inherited from VivaDataset, retained for backward
    compatibility and [RES-11] D3 diagnostic
  - cam_{left_wrist,right_wrist,high}_segment: [3, 12, H, W] — 12 raw
    frames per camera covering raw[t-3..t+8]. Used by the new T=3 latent
    layout: VAE encode these → latent [48, 3, H', W'] per camera, then
    concat along W axis in model.py
  - action_chunk:      [H, action_dim] current chunk ā_t
  - next_action_chunk: [H, action_dim] next chunk ā_{t+H} (used as ā' for
                       FQE bootstrap; zeros past episode end)
  - state_next_chunk:  [state_dim] normalized proprio/contact state at t+H.
                       Kept for legacy t+H two-chunk experiments. Invalid
                       near episode end is marked by has_state_next_chunk=False.
  - state_imag_anchor: [state_dim] normalized proprio/contact state at
                       t+imag_anchor_offset. For two-chunk imagination QAM,
                       set this to the WM RGB horizon: t+8 for the current
                       8-frame decode path.
  - viva_chunk:        [H+1] dense ViVa values at frames [t, ..., t+H]
                       loaded from precomputed cache; falls back to
                       heuristic linear interp if cache absent (with warning).
  - per-step success/fail masks for sparse reward
  - source_flag for Phase 2 trust gating

Edge handling for 12-frame segment:
  - target_frame < 0: clamp to frame 0 of the episode (repeat first frame)
  - target_frame >= episode_length: clamp to last frame (repeat last)
This matches the standard "edge-pad" treatment in video models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from revamp.common.viva_dataset import VivaDataset, CAMERA_KEY_ALIASES
from revamp.common.constants import (
    DEFAULT_CHUNK_LENGTH, DEFAULT_ACTION_DIM, DEFAULT_FUTURE_OFFSET,
    NUM_RAW_FRAMES_PER_SAMPLE, NUM_COND_RAW_FRAMES, NUM_TARGET_RAW_FRAMES,
)
from revamp.common.rewards import SOURCE_DEMO

try:
    from lerobot.datasets.video_utils import decode_video_frames
    _BATCH_DECODE_AVAILABLE = True
except Exception:
    decode_video_frames = None
    _BATCH_DECODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorldModelDataset(VivaDataset):
    """Subclass of VivaDataset that additionally loads action chunks,
    future cameras, and (optionally) cached ViVa values for dense
    PBRS shaping.
    """

    def __init__(
        self,
        *args,
        action_chunk_length: int = DEFAULT_CHUNK_LENGTH,
        action_dim: int = DEFAULT_ACTION_DIM,
        action_stride: int = 1,
        success_at_episode_end: bool = True,
        terminal_outcomes: Optional[List[str]] = None,
        viva_cache_dir: Optional[str] = None,
        imag_anchor_offset: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_chunk_length = action_chunk_length
        self.action_dim = action_dim
        self.action_stride = action_stride
        self.success_at_episode_end = success_at_episode_end
        self.terminal_outcomes = self._normalize_terminal_outcomes(terminal_outcomes)
        self.imag_anchor_offset = int(imag_anchor_offset) if imag_anchor_offset is not None else None

        self.viva_cache_dir = Path(viva_cache_dir) if viva_cache_dir else None
        self._viva_cache_warned = False

        expected_offset = action_chunk_length * action_stride
        if self.future_offset != expected_offset:
            logger.warning(
                f"future_offset ({self.future_offset}) != "
                f"action_chunk_length × stride ({expected_offset}). "
                f"Q targets and dynamics horizon may misalign."
            )

        self._cam_video_key: Dict[Tuple[int, str], str] = {}
        self._sub_ds_meta: Dict[int, Dict] = {}
        if _BATCH_DECODE_AVAILABLE:
            self._build_video_decode_caches()

    def _normalize_terminal_outcomes(
        self, terminal_outcomes: Optional[List[str]],
    ) -> Optional[List[str]]:
        """Normalize optional per-subdataset terminal labels.

        When unset, preserve the historical behavior controlled by
        success_at_episode_end. When set, labels must align with flattened
        data_paths order and be one of: success, failure/fail, none/unknown.
        """
        if terminal_outcomes is None:
            return None

        outcomes = [str(x).strip().lower() for x in terminal_outcomes]
        n_subdatasets = len(self.lerobot_dataset.datasets)
        if len(outcomes) != n_subdatasets:
            raise ValueError(
                f"terminal_outcomes length ({len(outcomes)}) must match the "
                f"number of flattened data_paths/subdatasets ({n_subdatasets})."
            )

        aliases = {
            "success": "success",
            "succeed": "success",
            "succeeded": "success",
            "failure": "failure",
            "fail": "failure",
            "failed": "failure",
            "none": "none",
            "unknown": "none",
            "neutral": "none",
            "": "none",
        }
        normalized = []
        for outcome in outcomes:
            if outcome not in aliases:
                raise ValueError(
                    f"Unsupported terminal outcome {outcome!r}. Expected one of "
                    "success, failure, fail, none, unknown."
                )
            normalized.append(aliases[outcome])
        logger.info(f"Terminal outcomes by subdataset: {normalized}")
        return normalized

    def _mark_terminal_outcome(
        self,
        success: torch.Tensor,
        fail: torch.Tensor,
        sub_ds_idx: int,
        chunk_idx: int,
    ) -> None:
        if self.terminal_outcomes is None:
            if self.success_at_episode_end:
                success[chunk_idx] = 1.0
            return

        outcome = self.terminal_outcomes[sub_ds_idx]
        if outcome == "success":
            success[chunk_idx] = 1.0
        elif outcome == "failure":
            fail[chunk_idx] = 1.0

    def _build_video_decode_caches(self) -> None:
        for sub_ds_idx, lds in enumerate(self.lerobot_dataset.datasets):
            sub_ds = getattr(lds, "dataset", lds)
            try:
                video_keys = list(sub_ds.meta.video_keys)
            except Exception:
                continue
            for logical, aliases in CAMERA_KEY_ALIASES.items():
                for k in aliases:
                    if k in video_keys:
                        self._cam_video_key[(sub_ds_idx, logical)] = k
                        break
            try:
                fps = float(sub_ds.meta.info["fps"])
            except Exception:
                fps = 20.0
            self._sub_ds_meta[sub_ds_idx] = dict(
                fps=fps,
                root=Path(sub_ds.root),
                meta=sub_ds.meta,
                hf_dataset=sub_ds.hf_dataset,
            )

    def _local_idx(self, base_actual_idx: int, sub_ds_idx: int) -> int:
        if sub_ds_idx == 0:
            return base_actual_idx
        return base_actual_idx - int(self.lerobot_dataset.cumulative_sizes[sub_ds_idx - 1])

    def _hf_row_fast(self, base_actual_idx: int, sub_ds_idx: int):
        meta = self._sub_ds_meta.get(sub_ds_idx)
        if meta is None:
            return None
        try:
            return meta["hf_dataset"][self._local_idx(base_actual_idx, sub_ds_idx)]
        except Exception:
            return None

    def _process_frames_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """[T, C, H_in, W_in] float in [0,1] -> [C, T, video_height, video_width]
        with the same scale-fit + center-pad policy as _process_image."""
        import torch.nn.functional as F
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
        th, tw = self.video_height, self.video_width
        H_in, W_in = frames.shape[-2], frames.shape[-1]
        if (H_in, W_in) != (th, tw):
            scale = min(th / H_in, tw / W_in)
            new_h, new_w = int(H_in * scale), int(W_in * scale)
            frames = F.interpolate(
                frames, size=(new_h, new_w), mode="bilinear", align_corners=False,
            )
            pad_h, pad_w = th - new_h, tw - new_w
            frames = F.pad(
                frames,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                value=0,
            )
        return frames.permute(1, 0, 2, 3).contiguous()

    def _load_action_chunk(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H = self.action_chunk_length
        stride = self.action_stride
        action_dim = self.action_dim

        sub_ds_idx = self._get_subdataset_index(base_actual_idx)
        meta = self._sub_ds_meta.get(sub_ds_idx)
        hf = meta["hf_dataset"] if meta is not None else None
        local_base = (
            self._local_idx(base_actual_idx, sub_ds_idx)
            if meta is not None else None
        )

        actions = []
        success = torch.zeros(H, dtype=torch.float32)
        fail = torch.zeros(H, dtype=torch.float32)
        terminal_marked = False

        for i in range(H):
            offset = i * stride
            target_frame = frame_index + offset
            if target_frame >= episode_length:
                actions.append(torch.zeros(action_dim, dtype=torch.float32))
                if not terminal_marked and i > 0:
                    self._mark_terminal_outcome(success, fail, sub_ds_idx, i)
                    terminal_marked = True
                continue

            actual_idx = base_actual_idx + offset
            action_raw = None
            if hf is not None:
                try:
                    # Parquet-only lookup: avoids the video decode that
                    # lerobot_dataset[idx] would otherwise trigger.
                    action_raw = hf[local_base + offset].get("action", None)
                except Exception:
                    action_raw = None
            if action_raw is None:
                try:
                    sample = self.lerobot_dataset[actual_idx]
                    action_raw = sample.get("action", None)
                except Exception as e:
                    logger.debug(f"Action load failed at {actual_idx}: {e}")

            if action_raw is None:
                actions.append(torch.zeros(action_dim, dtype=torch.float32))
            else:
                if isinstance(action_raw, torch.Tensor):
                    a = action_raw.float()
                elif isinstance(action_raw, list):
                    a = torch.tensor(action_raw, dtype=torch.float32)
                else:
                    a = torch.tensor(action_raw, dtype=torch.float32)
                if a.numel() != action_dim:
                    if a.numel() > action_dim:
                        a = a[:action_dim]
                    else:
                        a = torch.cat([a, torch.zeros(action_dim - a.numel())])
                actions.append(a)

            if (
                not terminal_marked
                and target_frame + stride >= episode_length - 1
                and (self.terminal_outcomes is not None or self.success_at_episode_end)
            ):
                self._mark_terminal_outcome(success, fail, sub_ds_idx, i)
                terminal_marked = True

        return torch.stack(actions, dim=0), success, fail

    def _load_next_action_chunk(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
    ) -> torch.Tensor:
        """Load the action chunk starting at frame_index + future_offset.

        Used by Stage 2 (FQE) as ā' for the TD bootstrap target — i.e., the
        next chunk that the demonstration policy executed at s_{t+H}. If the
        next chunk is past the episode end, returns zeros (terminal state).
        """
        H = self.action_chunk_length
        action_dim = self.action_dim
        next_frame = frame_index + self.future_offset
        # If the n-step target lands on the terminal state, the current chunk's
        # done mask owns the terminal reward and the TD target must not
        # bootstrap from Q(s_terminal, a').
        if next_frame >= max(episode_length - 1, 0):
            return torch.zeros(H, action_dim, dtype=torch.float32)
        next_actual = base_actual_idx + self.future_offset
        actions, _, _ = self._load_action_chunk(
            next_actual, next_frame, episode_length,
        )
        return actions

    def _load_state_at_offset(
        self, base_actual_idx: int, frame_index: int, episode_length: int, offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load normalized state at frame t + offset for imagination QAM.

        Unlike ``future_state`` from VivaDataset, this method does not clamp to
        the terminal frame. The caller can filter invalid samples with
        the returned validity bit so A2 never silently trains on a padded state.
        """
        next_frame = frame_index + int(offset)
        if next_frame >= episode_length:
            return (
                torch.zeros(self.state_dim, dtype=torch.float32),
                torch.tensor(False, dtype=torch.bool),
            )

        sub_ds_idx = self._get_subdataset_index(base_actual_idx)
        next_actual_idx = base_actual_idx + (next_frame - frame_index)
        sample = self._hf_row_fast(next_actual_idx, sub_ds_idx)
        if sample is None:
            sample = self.lerobot_dataset[next_actual_idx]

        episode_index = sample.get("episode_index", None)
        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.item())
        elif episode_index is None:
            # Same episode by construction; fall back to the local frame query.
            current = self._hf_row_fast(base_actual_idx, sub_ds_idx)
            if current is None:
                current = self.lerobot_dataset[base_actual_idx]
            episode_index = current.get("episode_index", 0)
            episode_index = int(episode_index.item()) if isinstance(episode_index, torch.Tensor) else int(episode_index)
        else:
            episode_index = int(episode_index)

        state = self._load_state(
            sample,
            sub_ds_idx,
            episode_index,
            next_frame,
            episode_length,
        )
        return state, torch.tensor(True, dtype=torch.bool)

    def _load_state_next_chunk(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load normalized state at frame t + H for legacy two-chunk paths."""
        return self._load_state_at_offset(
            base_actual_idx,
            frame_index,
            episode_length,
            self.action_chunk_length * self.action_stride,
        )

    def _load_viva_chunk(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
    ) -> torch.Tensor:
        """Return [H+1] tensor of per-step ViVa values at frames
        [t, t+1, ..., t+H]. If a cache directory is configured, every value
        must be present; otherwise Stage 2 would silently train on mixed reward
        semantics. Without a cache, falls back to the legacy heuristic.
        """
        H = self.action_chunk_length
        stride = self.action_stride

        values = torch.zeros(H + 1, dtype=torch.float32)

        for i in range(H + 1):
            target_frame = min(frame_index + i * stride, episode_length - 1)
            actual_idx = base_actual_idx + (target_frame - frame_index)
            if self.viva_cache_dir is not None:
                p = self.viva_cache_dir / f"{actual_idx}.pt"
                if not p.exists():
                    raise FileNotFoundError(
                        f"ViVa cache miss for actual_idx={actual_idx}, "
                        f"frame_index={target_frame}: {p}"
                    )
                try:
                    v = torch.load(p, map_location="cpu")
                    values[i] = float(v) if not isinstance(v, torch.Tensor) else float(v.item())
                    continue
                except Exception as e:
                    raise RuntimeError(f"ViVa cache read failed at {p}: {e}") from e
            values[i] = self._compute_value(target_frame, episode_length)

        return values

    def _load_cam_segment(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
        logical_cam_name: str,
    ) -> torch.Tensor:
        """Load NUM_RAW_FRAMES_PER_SAMPLE (=12) raw frames for one camera,
        covering raw[t-3..t+8]. Uses a single batched video decode per camera
        (one call to lerobot.datasets.video_utils.decode_video_frames covering
        all 12 timestamps) instead of 12 individual lerobot_dataset[idx] calls,
        which each re-opens the video and decodes all cameras for one frame.

        Edge handling (clamp / edge-pad):
          target_frame < 0:                  clamp to frame 0 of episode
          target_frame >= episode_length:    clamp to last frame

        Returns: [3, NUM_RAW_FRAMES_PER_SAMPLE, video_height, video_width]
                 RGB tensor in [0, 1].
        """
        sub_ds_idx = self._get_subdataset_index(base_actual_idx)
        vid_key = self._cam_video_key.get((sub_ds_idx, logical_cam_name))
        meta = self._sub_ds_meta.get(sub_ds_idx)

        if _BATCH_DECODE_AVAILABLE and vid_key is not None and meta is not None:
            try:
                row = self._hf_row_fast(base_actual_idx, sub_ds_idx)
                if row is None:
                    raise RuntimeError("hf row lookup failed")
                ep_val = row["episode_index"]
                if isinstance(ep_val, torch.Tensor):
                    episode_index = int(ep_val.item())
                else:
                    episode_index = int(ep_val)

                fps = meta["fps"]
                rel_path = meta["meta"].get_video_file_path(episode_index, vid_key)
                video_path = meta["root"] / rel_path

                start_offset = -NUM_COND_RAW_FRAMES + 1   # -3
                end_offset = NUM_TARGET_RAW_FRAMES        #  8
                indices = [
                    max(0, min(frame_index + o, episode_length - 1))
                    for o in range(start_offset, end_offset + 1)
                ]
                timestamps = [i / fps for i in indices]
                tol = 0.5 / fps

                frames = decode_video_frames(
                    str(video_path), timestamps, tol, backend="pyav",
                )  # [T, C, H_in, W_in], float in [0, 1]
                return self._process_frames_batch(frames)
            except Exception as e:
                logger.debug(
                    f"batch cam decode failed for cam={logical_cam_name} "
                    f"idx={base_actual_idx}: {e}; falling back to per-frame"
                )

        return self._load_cam_segment_legacy(
            base_actual_idx, frame_index, episode_length, logical_cam_name,
        )

    def _load_cam_segment_legacy(
        self, base_actual_idx: int, frame_index: int, episode_length: int,
        logical_cam_name: str,
    ) -> torch.Tensor:
        start_offset = -NUM_COND_RAW_FRAMES + 1
        end_offset = NUM_TARGET_RAW_FRAMES
        frames = []
        for offset in range(start_offset, end_offset + 1):
            target_frame = frame_index + offset
            clamped_frame = max(0, min(target_frame, episode_length - 1))
            actual_idx = base_actual_idx + (clamped_frame - frame_index)
            try:
                sample = self.lerobot_dataset[actual_idx]
                img = self._get_camera_image(sample, logical_cam_name)
                if img is None:
                    frames.append(torch.zeros(
                        3, self.video_height, self.video_width,
                        dtype=torch.float32,
                    ))
                else:
                    frames.append(self._process_image(img))
            except Exception as e:
                logger.debug(
                    f"cam segment load failed at idx={actual_idx} "
                    f"cam={logical_cam_name}: {e}"
                )
                frames.append(torch.zeros(
                    3, self.video_height, self.video_width,
                    dtype=torch.float32,
                ))
        return torch.stack(frames, dim=1)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        base = super().__getitem__(idx)
        if base is None:
            return None

        actual_idx = idx
        if self._sample_indices is not None:
            actual_idx = int(self._sample_indices[idx])

        try:
            frame_index = base["frame_idx"]
            episode_length = base["episode_length"]
            if isinstance(frame_index, torch.Tensor):
                frame_index = int(frame_index.item())
            if isinstance(episode_length, torch.Tensor):
                episode_length = int(episode_length.item())

            action_chunk, success_chunk, fail_chunk = self._load_action_chunk(
                actual_idx, frame_index, episode_length,
            )
            next_action_chunk = self._load_next_action_chunk(
                actual_idx, frame_index, episode_length,
            )
            state_next_chunk, has_state_next_chunk = self._load_state_next_chunk(
                actual_idx, frame_index, episode_length,
            )
            imag_anchor_offset = (
                self.imag_anchor_offset
                if self.imag_anchor_offset is not None
                else self.action_chunk_length * self.action_stride
            )
            state_imag_anchor, has_state_imag_anchor = self._load_state_at_offset(
                actual_idx, frame_index, episode_length, imag_anchor_offset,
            )
            viva_chunk = self._load_viva_chunk(
                actual_idx, frame_index, episode_length,
            )

            # [BUG-4] 12-frame raw camera segments for new T=3 latent layout.
            # Each segment covers raw[t-3..t+8] = 4 cond frames + 8 target frames.
            # model.py concatenates the 3 cameras along the W axis and VAE-encodes
            # the whole stack to latent [48, 3, H', W'].
            cam_left_segment = self._load_cam_segment(
                actual_idx, frame_index, episode_length,
                "cam_left_wrist",
            )
            cam_right_segment = self._load_cam_segment(
                actual_idx, frame_index, episode_length,
                "cam_right_wrist",
            )
            cam_high_segment = self._load_cam_segment(
                actual_idx, frame_index, episode_length,
                "cam_high",
            )

            base.update(
                {
                    "action_chunk": action_chunk,                  # [H, action_dim]
                    "next_action_chunk": next_action_chunk,        # [H, action_dim] FQE ā'
                    "state_next_chunk": state_next_chunk,          # [state_dim] real s[t+H]
                    "has_state_next_chunk": has_state_next_chunk,  # bool, false near episode end
                    "state_imag_anchor": state_imag_anchor,        # [state_dim] real s[t+imag_anchor_offset]
                    "has_state_imag_anchor": has_state_imag_anchor,
                    "success_chunk": success_chunk,                # [H]
                    "fail_chunk": fail_chunk,                      # [H]
                    "viva_chunk": viva_chunk,                      # [H+1] dense Φ
                    "source_flag": torch.tensor(SOURCE_DEMO, dtype=torch.long),
                    "cam_left_segment":  cam_left_segment,         # [3, 12, H, W]
                    "cam_right_segment": cam_right_segment,        # [3, 12, H, W]
                    "cam_high_segment":  cam_high_segment,         # [3, 12, H, W]
                }
            )
            return base

        except Exception as e:
            raise RuntimeError(f"WorldModelDataset error idx={idx}: {e}") from e


def world_model_collate_fn(batch):
    if any(b is None for b in batch):
        raise RuntimeError("WorldModelDataset returned None; inspect dataset error above.")

    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        elif isinstance(v0, (int, float)):
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            try:
                out[k] = torch.tensor([b[k] for b in batch])
            except Exception:
                out[k] = [b[k] for b in batch]
    return out
