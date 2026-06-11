"""Roll out a Stage-1 WorldModel checkpoint for one action chunk.

Example:
    python -m revamp.stage1_2_world_model.rollout_check \
        --config outputs/turn_on_sink_faucet/materialized_configs/phase1_wm_online_update.release.yaml \
        --checkpoint checkpoints/world_model_after_online_update \
        --sample-index 0 \
        --out-dir outputs/turn_on_sink_faucet/rollout_check \
        --num-inference-steps 50 \
        --device cuda \
        --fps 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from revamp.common.dataset import WorldModelDataset
from revamp.common.world_model import WorldModel


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "model.safetensors"
        if candidate.exists():
            return candidate
        candidate = path / "pytorch_model.bin"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    return path


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def _build_dataset(config) -> WorldModelDataset:
    task_configs = [
        {"task_id": tc.task_id, "data_paths": list(tc.data_paths)}
        for tc in config.dataset.tasks
    ]
    return WorldModelDataset(
        task_configs=task_configs,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        future_offset=config.dataset.future_offset,
        action_chunk_length=config.common.action_chunk_length,
        action_dim=config.common.action_dim,
        action_stride=config.dataset.get("action_stride", 1),
        max_samples=config.dataset.get("max_samples", None),
        viva_cache_dir=config.dataset.get("viva_cache_dir", None),
        terminal_outcomes=config.dataset.get("terminal_outcomes", None),
        state_dim=config.common.get("state_dim", None),
        use_contact=config.dataset.get("use_contact", False),
        left_contact_key=config.dataset.get("left_contact_key", "observation.contact.left"),
        right_contact_key=config.dataset.get("right_contact_key", "observation.contact.right"),
        require_contact=config.dataset.get("require_contact", True),
        contact_sidecar_name=config.dataset.get("contact_sidecar_name", "contact_features.npy"),
    )


def _next_valid_sample(dataset: WorldModelDataset, start_idx: int) -> tuple[int, Dict[str, torch.Tensor]]:
    for idx in range(start_idx, len(dataset)):
        sample = dataset[idx]
        if sample is not None:
            return idx, sample
    raise RuntimeError(f"No valid sample found at or after index {start_idx}")


def _effective_index_from_actual(dataset: WorldModelDataset, actual_idx: int) -> int:
    if dataset._sample_indices is None:
        return actual_idx
    matches = (dataset._sample_indices == actual_idx).nonzero()[0]
    if len(matches) == 0:
        raise ValueError(
            f"Actual dataset index {actual_idx} is not present after max_samples "
            "subsampling. Set dataset.max_samples: null for episode/frame lookup."
        )
    return int(matches[0])


def _index_from_episode_frame(
    dataset: WorldModelDataset,
    episode_index: int,
    frame_index: int,
    subdataset_index: int,
) -> int:
    if subdataset_index < 0 or subdataset_index >= len(dataset._subdataset_episode_lengths):
        raise ValueError(
            f"subdataset_index={subdataset_index} out of range "
            f"[0, {len(dataset._subdataset_episode_lengths) - 1}]"
        )

    lengths = dataset._subdataset_episode_lengths[subdataset_index]
    if episode_index < 0 or episode_index >= len(lengths):
        raise ValueError(
            f"episode_index={episode_index} out of range [0, {len(lengths) - 1}] "
            f"for subdataset {subdataset_index}"
        )
    episode_length = int(lengths[episode_index])
    if frame_index < 0 or frame_index >= episode_length:
        raise ValueError(
            f"frame_index={frame_index} out of range [0, {episode_length - 1}] "
            f"for episode {episode_index}"
        )

    subdataset_offset = 0
    if subdataset_index > 0:
        subdataset_offset = int(dataset.lerobot_dataset.cumulative_sizes[subdataset_index - 1])
    actual_idx = subdataset_offset + sum(int(v) for v in lengths[:episode_index]) + frame_index
    return _effective_index_from_actual(dataset, actual_idx)


def _batch_to_device(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = [
        "state",
        "action_chunk",
        "cam_left_segment",
        "cam_right_segment",
        "cam_high_segment",
    ]
    return {
        key: sample[key].unsqueeze(0).to(device=device, non_blocking=True)
        for key in keys
    }


def _make_segment_from_cond(cond_frames: torch.Tensor) -> torch.Tensor:
    """[3, 4, H, W] -> [3, 12, H, W].

    The model overwrites target latents during sampling. We still need a
    syntactically valid 12-frame segment for VAE encoding, so fill the target
    portion with repeats of the last condition frame.
    """
    tail = cond_frames[:, -1:].expand(-1, 8, -1, -1)
    return torch.cat([cond_frames, tail], dim=1).contiguous()


def _sample_with_visual_cond(
    sample: Dict[str, torch.Tensor],
    cond_by_camera: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    sample = dict(sample)
    for key, cond in cond_by_camera.items():
        sample[key] = _make_segment_from_cond(cond)
    return sample


def _pred_last_cond(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "cam_left_segment": pred["next_cam_left"][0, :, -4:].detach().cpu(),
        "cam_right_segment": pred["next_cam_right"][0, :, -4:].detach().cpu(),
        "cam_high_segment": pred["next_cam_high"][0, :, -4:].detach().cpu(),
    }


def _to_uint8_video(video: torch.Tensor) -> list:
    """[3, T, H, W] in [-1, 1] or [0, 1] -> list of [H, W, 3] uint8."""
    video = video.detach().float().cpu()
    if video.min() < -0.05:
        video = (video + 1.0) * 0.5
    video = video.clamp(0, 1)
    video = (video.permute(1, 2, 3, 0).numpy() * 255.0).round().astype("uint8")
    return [frame for frame in video]


def _camera_frames(
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[List, List, List]]:
    cameras = {
        "left": ("next_cam_left", "cam_left_segment"),
        "right": ("next_cam_right", "cam_right_segment"),
        "high": ("next_cam_high", "cam_high_segment"),
    }
    frames = {}
    for name, (pred_key, gt_key) in cameras.items():
        pred_frames = _to_uint8_video(pred[pred_key][0])
        # Model decodes [cond, target1, target2] and drops the condition frame,
        # so prediction length should match the 8-frame action chunk horizon.
        pred_len = len(pred_frames)
        gt_future = sample[gt_key][:, 4:4 + pred_len]
        gt_frames = _to_uint8_video(gt_future)
        side_by_side = [
            torch.cat(
                [
                    torch.from_numpy(gt).permute(2, 0, 1),
                    torch.from_numpy(pr).permute(2, 0, 1),
                ],
                dim=2,
            ).permute(1, 2, 0).numpy()
            for gt, pr in zip(gt_frames, pred_frames)
        ]
        frames[name] = (pred_frames, gt_frames, side_by_side)
    return frames


def _write_camera_videos(
    pred: Dict[str, torch.Tensor],
    sample: Dict[str, torch.Tensor],
    out_dir: Path,
    fps: int,
) -> Dict[str, Tuple[List, List, List]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _camera_frames(pred, sample)
    for name, (pred_frames, gt_frames, side_by_side) in frames.items():
        imageio.mimsave(out_dir / f"pred_{name}.mp4", pred_frames, fps=fps)
        imageio.mimsave(out_dir / f"gt_{name}.mp4", gt_frames, fps=fps)
        imageio.mimsave(out_dir / f"compare_{name}_gt_left_pred_right.mp4", side_by_side, fps=fps)
    return frames


def _extend_combined(
    combined: Dict[str, Dict[str, List]],
    chunk_frames: Dict[str, Tuple[List, List, List]],
) -> None:
    for name, (pred_frames, gt_frames, side_by_side) in chunk_frames.items():
        combined[name]["pred"].extend(pred_frames)
        combined[name]["gt"].extend(gt_frames)
        combined[name]["compare"].extend(side_by_side)


def _write_combined(combined: Dict[str, Dict[str, List]], out_dir: Path, fps: int) -> None:
    for name, groups in combined.items():
        imageio.mimsave(out_dir / f"all_pred_{name}.mp4", groups["pred"], fps=fps)
        imageio.mimsave(out_dir / f"all_gt_{name}.mp4", groups["gt"], fps=fps)
        imageio.mimsave(
            out_dir / f"all_compare_{name}_gt_left_pred_right.mp4",
            groups["compare"],
            fps=fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage-1 YAML config")
    parser.add_argument("--checkpoint", required=True, help="Accelerate checkpoint dir or model file")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument(
        "--subdataset-index",
        type=int,
        default=0,
        help="Which data_path to use when config has multiple data_paths.",
    )
    parser.add_argument("--out-dir", default="rollouts/stage1")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=None,
        help=(
            "Dataset-frame stride between chunk starts. Default uses "
            "dataset.future_offset/action horizon."
        ),
    )
    parser.add_argument(
        "--rollout-mode",
        choices=("teacher_forced", "visual_autoregressive", "latent_autoregressive"),
        default="teacher_forced",
        help=(
            "teacher_forced: each chunk uses dataset state/camera/action. "
            "visual_autoregressive: state/action still come from dataset, "
            "but camera condition comes from the previous predicted RGB chunk. "
            "latent_autoregressive: state/action still come from dataset, "
            "but camera condition is carried as latent to avoid RGB decode/encode."
        ),
    )
    args = parser.parse_args()

    _setup_logging()
    config = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    logger.info("Building dataset ...")
    dataset = _build_dataset(config)
    if (args.episode_index is None) != (args.frame_index is None):
        raise ValueError("--episode-index and --frame-index must be provided together")
    sample_index = args.sample_index
    if args.episode_index is not None:
        sample_index = _index_from_episode_frame(
            dataset,
            episode_index=args.episode_index,
            frame_index=args.frame_index,
            subdataset_index=args.subdataset_index,
        )
    actual_idx, first_sample = _next_valid_sample(dataset, sample_index)
    logger.info(
        "Using sample index %s (episode=%s frame=%s)",
        actual_idx,
        first_sample.get("episode_index", args.episode_index),
        first_sample.get("frame_idx", args.frame_index),
    )

    logger.info("Building WorldModel ...")
    model = WorldModel(config)
    ckpt = _checkpoint_file(args.checkpoint)
    logger.info("Loading checkpoint from %s", ckpt)
    incompatible = model.load_state_dict(_load_state_dict(ckpt), strict=False)
    if incompatible.missing_keys:
        logger.warning("Missing keys: %s", incompatible.missing_keys[:20])
    if incompatible.unexpected_keys:
        logger.warning("Unexpected keys: %s", incompatible.unexpected_keys[:20])
    model.to(device)
    model.eval()

    t5_path = config.dataset.tasks[0].t5_embedding_path
    t5 = torch.load(t5_path, map_location="cpu").to(device=device, dtype=model.dtype)
    if t5.dim() == 2:
        t5 = t5.unsqueeze(0)

    out_dir = Path(args.out_dir)
    step = int(args.chunk_stride or config.dataset.future_offset)
    if args.rollout_mode == "latent_autoregressive" and step != int(config.dataset.future_offset):
        logger.warning(
            "latent_autoregressive is semantically aligned with future_offset=%s; "
            "got chunk stride %s. This may mismatch carried latent condition "
            "with dataset state/action timing.",
            config.dataset.future_offset,
            step,
        )
    cond_by_camera = None
    cond_latent = None
    combined = {
        name: {"pred": [], "gt": [], "compare": []}
        for name in ("left", "right", "high")
    }

    start_frame = int(first_sample["frame_idx"])
    episode_length = int(first_sample["episode_length"])
    for chunk_idx in range(args.num_chunks):
        if start_frame + chunk_idx * step >= episode_length:
            logger.info("Stopping at chunk %s: reached episode end.", chunk_idx)
            break

        chunk_start = actual_idx + chunk_idx * step
        chunk_actual_idx, sample = _next_valid_sample(dataset, chunk_start)
        if chunk_actual_idx != chunk_start:
            logger.warning(
                "Requested chunk index %s, using next valid index %s",
                chunk_start,
                chunk_actual_idx,
            )

        if args.rollout_mode == "visual_autoregressive" and cond_by_camera is not None:
            sample_for_model = _sample_with_visual_cond(sample, cond_by_camera)
        else:
            sample_for_model = sample

        logger.info(
            "Predicting chunk %s/%s from dataset index %s frame %s",
            chunk_idx + 1,
            args.num_chunks,
            chunk_actual_idx,
            sample["frame_idx"],
        )
        batch = _batch_to_device(sample_for_model, device)
        with torch.no_grad():
            if args.rollout_mode == "latent_autoregressive" and cond_latent is not None:
                pred = model.predict_chunk_from_cond_latent(
                    cond_latent=cond_latent,
                    state=batch["state"],
                    action_chunk=batch["action_chunk"],
                    t5_embeddings=t5.expand(1, -1, -1),
                    num_inference_steps=args.num_inference_steps,
                    decode_rgb=True,
                )
            else:
                pred = model.predict_chunk(
                    state=batch["state"],
                    cam_left_segment=batch["cam_left_segment"],
                    cam_right_segment=batch["cam_right_segment"],
                    cam_high_segment=batch["cam_high_segment"],
                    action_chunk=batch["action_chunk"],
                    t5_embeddings=t5.expand(1, -1, -1),
                    num_inference_steps=args.num_inference_steps,
                    decode_rgb=True,
                )

        chunk_dir = out_dir / f"chunk_{chunk_idx:03d}_idx_{chunk_actual_idx}"
        chunk_frames = _write_camera_videos(pred, sample, chunk_dir, fps=args.fps)
        _extend_combined(combined, chunk_frames)
        cond_by_camera = _pred_last_cond(pred)
        cond_latent = pred["next_cam_concat_latent"][:, :, -1:].detach()
        if "q1" in pred and "q2" in pred:
            logger.info(
                "chunk=%s q1=%s q2=%s",
                chunk_idx,
                pred["q1"].detach().float().cpu().tolist(),
                pred["q2"].detach().float().cpu().tolist(),
            )
        else:
            logger.info("chunk=%s carried latent cond shape=%s", chunk_idx, tuple(cond_latent.shape))

    _write_combined(combined, out_dir, fps=args.fps)
    logger.info("Saved rollout videos to %s", out_dir)


if __name__ == "__main__":
    main()
