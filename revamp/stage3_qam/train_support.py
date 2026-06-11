"""Support code for the Stage 3 OpenPI/QAM trainer.

QAM schedule/path helpers are adapted from the MIT-licensed "Q-learning with
Adjoint Matching" project and integrated with REVAMP/OpenPI training. See
NOTICE for attribution.

Keep this module stage-local: these helpers know about OpenPI observation
transforms, Q-gradient payloads, and two-chunk imagination. They are not
general enough for revamp.common.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader

from revamp.common.dataset import WorldModelDataset, world_model_collate_fn
from revamp.stage3_qam import runtime
from revamp.stage3_qam.openpi_policy import OpenPiPolicyBridge


@dataclass(frozen=True)
class ImagQAMSettings:
    enabled: bool
    lambda_weight: float
    update_every: int
    warmup_steps: int
    num_inference_steps: int
    state_offset: int
    pred_history_frames: int
    pred_history_strategy: str
    max_batches: int | None
    server_url: str
    payload_compression: str

    def should_update(self, step: int, batches_done: int) -> bool:
        return (
            self.enabled
            and step >= self.warmup_steps
            and step % self.update_every == 0
            and (self.max_batches is None or batches_done < self.max_batches)
        )


def load_imag_qam_settings(config, args) -> ImagQAMSettings:
    max_batches_cfg = config.training.get("imag_max_batches", None)
    settings = ImagQAMSettings(
        enabled=bool(config.training.get("two_chunk_imag_qam", False)),
        lambda_weight=runtime.cfg_get_float(config.training, "imag_lambda", 0.25),
        update_every=runtime.cfg_get_int(config.training, "imag_update_every", 1),
        warmup_steps=runtime.cfg_get_int(config.training, "imag_warmup_steps", 0),
        num_inference_steps=runtime.cfg_get_int(config.training, "imag_num_inference_steps", 20),
        state_offset=runtime.cfg_get_int(config.training, "imag_state_offset", int(config.common.action_chunk_length)),
        pred_history_frames=runtime.cfg_get_int(config.training, "imag_pred_history_frames", 4),
        pred_history_strategy=str(config.training.get("imag_pred_history_strategy", "last")),
        max_batches=None if max_batches_cfg is None else int(max_batches_cfg),
        server_url=args.imag_server_url or config.system.get("imag_server_url", "http://127.0.0.1:8766"),
        payload_compression=str(config.system.get("imag_payload_compression", "uncompressed")),
    )
    if settings.update_every <= 0:
        raise ValueError(f"training.imag_update_every must be positive, got {settings.update_every}")
    if settings.lambda_weight < 0.0:
        raise ValueError(f"training.imag_lambda must be non-negative, got {settings.lambda_weight}")
    if settings.pred_history_frames != 4:
        raise ValueError(
            "OpenPI observation preparation expects 4 condition frames; "
            f"got training.imag_pred_history_frames={settings.pred_history_frames}"
        )
    if settings.pred_history_strategy != "last":
        raise ValueError(
            f"training.imag_pred_history_strategy={settings.pred_history_strategy!r} is unsupported; "
            "use 'last'"
        )
    return settings


def build_dataset(config) -> WorldModelDataset:
    return WorldModelDataset(
        task_configs=[{"task_id": tc.task_id, "data_paths": list(tc.data_paths)} for tc in config.dataset.tasks],
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        future_offset=config.dataset.future_offset,
        action_chunk_length=config.common.action_chunk_length,
        action_dim=config.common.action_dim,
        action_stride=config.dataset.get("action_stride", 1),
        max_samples=config.dataset.get("max_samples", None),
        use_contact=config.dataset.get("use_contact", False),
        left_contact_key=config.dataset.get("left_contact_key", "observation.contact.left"),
        right_contact_key=config.dataset.get("right_contact_key", "observation.contact.right"),
        require_contact=config.dataset.get("require_contact", False),
        contact_sidecar_name=config.dataset.get("contact_sidecar_name", "contact_features.npy"),
        viva_cache_dir=None,
        terminal_outcomes=config.dataset.get("terminal_outcomes", None),
        imag_anchor_offset=config.training.get("imag_state_offset", None),
    )


def build_train_dataloader(config, dataset: WorldModelDataset, batch_size: int) -> tuple[DataLoader, int, int, dict[str, Any]]:
    dataloader_seed = int(config.system.get("dataloader_seed", 0))
    num_workers = int(config.system.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": world_model_collate_fn,
        "pin_memory": bool(config.system.get("pin_memory", False)),
        "shuffle": True,
        "drop_last": bool(config.system.get("drop_last", True)),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.system.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.system.get("prefetch_factor", 2))
        kwargs["multiprocessing_context"] = str(config.system.get("multiprocessing_context", "spawn"))
    return DataLoader(dataset, **kwargs), dataloader_seed, num_workers, kwargs


def prepare_openpi_observation(policy, bridge: OpenPiPolicyBridge, batch: dict[str, torch.Tensor]):
    from openpi.models import model as _model

    state = batch["state"]
    cam_left_cond = batch["cam_left_segment"][:, :, :4]
    cam_high_cond = batch["cam_high_segment"][:, :, :4]
    items = []
    for i in range(state.shape[0]):
        obs = bridge._obs_from_batch_item(state[i], cam_left_cond[i], cam_high_cond[i])
        items.append(policy._input_transform(obs))

    merged: dict[str, Any] = {}
    for key in items[0].keys():
        if isinstance(items[0][key], dict):
            merged[key] = {
                sub_key: jnp.asarray(np.stack([it[key][sub_key] for it in items], axis=0))
                for sub_key in items[0][key]
            }
        else:
            merged[key] = jnp.asarray(np.stack([it[key] for it in items], axis=0))
    return _model.Observation.from_dict(merged)


def q_server_payload(batch: dict[str, torch.Tensor], action_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Build Q-gradient request with cond-only cameras.

    QAM consumes Q(s_t, a). Keep the HTTP payload to raw[t-3..t] and let the
    Q server repeat the current frame to form a VAE-safe 12-frame segment,
    avoiding future-frame leakage from demo videos.
    """
    return {
        "state": np.asarray(batch["state"], dtype=np.float32),
        "cam_left_cond": np.ascontiguousarray(batch["cam_left_segment"][:, :, :4], dtype=np.float32),
        "cam_right_cond": np.ascontiguousarray(batch["cam_right_segment"][:, :, :4], dtype=np.float32),
        "cam_high_cond": np.ascontiguousarray(batch["cam_high_segment"][:, :, :4], dtype=np.float32),
        "action": np.ascontiguousarray(action_raw, dtype=np.float32),
    }


def wm_imag_payload(
    batch: dict[str, torch.Tensor],
    action_raw: np.ndarray,
    *,
    num_inference_steps: int,
) -> dict[str, np.ndarray]:
    """Build no-leak world-model request for A1 -> imagined future frames."""

    def cond_padded_segment(key: str) -> np.ndarray:
        cam = batch[key].detach().float().cpu()
        out = torch.zeros_like(cam)
        out[:, :, :4] = cam[:, :, :4]
        out[:, :, 4:] = cam[:, :, 3:4].expand(-1, -1, out.shape[2] - 4, -1, -1)
        return np.ascontiguousarray(out.numpy(), dtype=np.float32)

    return {
        "state": np.asarray(batch["state"], dtype=np.float32),
        "cam_left_segment": cond_padded_segment("cam_left_segment"),
        "cam_right_segment": cond_padded_segment("cam_right_segment"),
        "cam_high_segment": cond_padded_segment("cam_high_segment"),
        "action": np.ascontiguousarray(action_raw, dtype=np.float32),
        "num_inference_steps": np.asarray(int(num_inference_steps), dtype=np.int32),
    }


def build_two_chunk_imagined_batch(
    batch: dict[str, torch.Tensor],
    pred: dict[str, np.ndarray],
    *,
    history_frames: int,
    history_strategy: str,
) -> dict[str, torch.Tensor]:
    state_key = "state_imag_anchor" if "state_imag_anchor" in batch else "state_next_chunk"
    valid_key = "has_state_imag_anchor" if "has_state_imag_anchor" in batch else "has_state_next_chunk"
    if state_key not in batch:
        raise RuntimeError("two-chunk imagination requires state_imag_anchor/state_next_chunk; update WorldModelDataset")
    if valid_key in batch and not bool(batch[valid_key].bool().all().item()):
        raise RuntimeError(f"two-chunk imagination received a batch with invalid {state_key}")

    imag_batch = dict(batch)
    imag_batch["state"] = batch[state_key].detach().clone()
    if history_strategy != "last":
        raise ValueError(f"Unsupported imag_pred_history_strategy={history_strategy!r}; currently only 'last' is implemented")
    if history_frames <= 0:
        raise ValueError(f"imag_pred_history_frames must be positive, got {history_frames}")

    for batch_key, pred_key in (
        ("cam_left_segment", "next_cam_left"),
        ("cam_right_segment", "next_cam_right"),
        ("cam_high_segment", "next_cam_high"),
    ):
        frames = torch.as_tensor(pred[pred_key], dtype=batch[batch_key].dtype)
        expected = batch[batch_key]
        if (
            frames.shape[0] != expected.shape[0]
            or frames.shape[1] != expected.shape[1]
            or frames.shape[2] < history_frames
            or frames.shape[3:] != expected.shape[3:]
        ):
            raise RuntimeError(
                f"{pred_key} shape {tuple(frames.shape)} is incompatible with "
                f"{batch_key} shape {tuple(batch[batch_key].shape)}; expected at least "
                f"{history_frames} predicted frames for the imagined observation history"
            )
        if not torch.isfinite(frames).all():
            raise RuntimeError(f"{pred_key} contains non-finite values")
        segment = torch.zeros_like(batch[batch_key])
        # A2 consumes a fixed 4-frame visual history. For the current
        # 8-frame WM decode this selects predicted raw[t+5..t+8].
        imag_cond = frames[:, :, -history_frames:]
        segment[:, :, :4] = imag_cond
        segment[:, :, 4:] = imag_cond[:, :, -1:].expand(-1, -1, segment.shape[2] - 4, -1, -1)
        imag_batch[batch_key] = segment
    return imag_batch


def run_q_preflight(
    dataset: WorldModelDataset,
    *,
    batch_size: int,
    q_h: int,
    q_a: int,
    q_client: runtime.QGradientClient,
    log,
) -> None:
    log("running Q server preflight on one deterministic batch")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=world_model_collate_fn,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
    )
    batch = next(iter(loader))
    action = np.zeros((int(batch["state"].shape[0]), q_h, q_a), dtype=np.float32)
    q_grad, q_values, q_mean, timings = q_client.post_grad(q_server_payload(batch, action))
    runtime.validate_q_response(
        q_grad_raw=q_grad,
        q_values=q_values,
        q_mean=q_mean,
        expected_shape=action.shape,
    )
    log(
        "Q server preflight ok: "
        f"q={q_mean:.4f}, grad_std={float(q_grad.std()):.4g}, "
        f"timings={json.dumps(timings)}"
    )


def query_q_gradient(
    q_client: runtime.QGradientClient,
    batch: dict[str, torch.Tensor],
    action_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    import time

    t0 = time.time()
    q_grad_raw, q_values, q_mean, _timings = q_client.post_grad(q_server_payload(batch, action_raw))
    runtime.validate_q_response(
        q_grad_raw=q_grad_raw,
        q_values=q_values,
        q_mean=q_mean,
        expected_shape=action_raw.shape,
    )
    return q_grad_raw, q_values, q_mean, time.time() - t0


def update_metric_ema(
    ema_state: dict[str, float],
    row: dict[str, Any],
    keys: list[str],
    beta: float,
) -> dict[str, float]:
    out = {}
    for key in keys:
        value = row.get(key, None)
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        prev = ema_state.get(key, value)
        ema = beta * prev + (1.0 - beta) * value
        ema_state[key] = ema
        out[f"{key}_ema"] = ema
    return out


def wandb_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Minimal W&B surface for live monitoring."""
    base = {
        "train/loss": row["loss"],
        "train/qam_loss": row["qam_loss"],
        "train/reference_reg_loss": row["reference_reg_loss"],
        "train/q_mean": row["q_mean"],
        "train/q_mean_ema": row.get("q_mean_ema"),
        "train/trainable_update_norm_ema": row.get("trainable_update_norm_ema"),
        "train/action_raw_std_ema": row.get("action_raw_std_ema"),
        "time/q_server_seconds": row["q_s"],
        "time/jax_train_seconds": row["train_s"],
        "time/openpi_sample_seconds": row["sample_s"],
        "time/data_wait_seconds": row.get("data_wait_s"),
    }
    if row.get("imag_enabled", False):
        base.update(
            {
                "imag/lambda": row.get("imag_lambda"),
                "imag/q_mean": row.get("imag_q_mean"),
                "imag/loss": row.get("imag_loss"),
                "imag/qam_loss": row.get("imag_qam_loss"),
                "imag/action_raw_std": row.get("imag_action_raw_std"),
                "time/imag_world_model_seconds": row.get("imag_world_model_s"),
                "time/imag_q_server_seconds": row.get("imag_q_s"),
                "time/imag_train_seconds": row.get("imag_train_s"),
            }
        )
    return {k: v for k, v in base.items() if v is not None}


def empty_imag_row() -> dict[str, Any]:
    return {"imag_enabled": False, "imag_skip_reason": None}


def imag_log_row(
    settings: ImagQAMSettings,
    *,
    qam_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    world_model_s: float,
) -> dict[str, Any]:
    return {
        "imag_enabled": True,
        "imag_skip_reason": None,
        "imag_lambda": settings.lambda_weight,
        "imag_q_mean": qam_metrics["q_mean"],
        "imag_q_s": qam_metrics["q_s"],
        "imag_world_model_s": world_model_s,
        "imag_train_s": qam_metrics["train_s"],
        "imag_action_raw_std": qam_metrics["action_raw_std"],
        "imag_loss": float(train_metrics["loss"]),
        "imag_qam_loss": float(train_metrics["qam_loss"]),
    }


def train_log_row(
    *,
    step: int,
    train_metrics: dict[str, Any],
    qam_metrics: dict[str, Any],
    data_wait_s: float,
    imag_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": step,
        "loss": float(train_metrics["loss"]),
        "qam_loss": float(train_metrics["qam_loss"]),
        "reference_reg_loss": float(train_metrics["reference_reg_loss"]),
        "q_mean": qam_metrics["q_mean"],
        "q_s": qam_metrics["q_s"],
        "train_s": qam_metrics["train_s"],
        "sample_s": qam_metrics["sample_s"],
        "data_wait_s": data_wait_s,
        "action_raw_std": qam_metrics["action_raw_std"],
        "trainable_update_norm": float(train_metrics["trainable_update_norm"]),
        **imag_row,
    }


def progress_postfix(row: dict[str, Any]) -> dict[str, str]:
    return {
        "loss": f"{row['loss']:.3g}",
        "q": f"{row['q_mean']:.3f}",
        "upd": f"{row['trainable_update_norm']:.2g}",
        "a_std": f"{row['action_raw_std']:.2g}",
        "sample_s": f"{row['sample_s']:.1f}",
        "train_s": f"{row['train_s']:.1f}",
    }


def init_wandb(config, args, batch_size: int, per_device_batch_size: int, fsdp_devices: int, log):
    report_to = str(runtime.cfg_get(config, "logging.report_to", "")).lower()
    if "wandb" not in report_to and report_to not in {"all", "true", "1"}:
        return None

    import wandb

    wandb_cfg = runtime.cfg_to_container(config)
    if isinstance(wandb_cfg, dict):
        wandb_cfg.setdefault("runtime", {})
        wandb_cfg["runtime"].update(
            {
                "global_batch_size": batch_size,
                "per_device_batch_size": per_device_batch_size,
                "fsdp_devices": fsdp_devices,
                "jax_device_count": jax.device_count(),
                "q_server_url": args.q_server_url,
                "skip_save": bool(args.skip_save),
                "resume_from": args.resume_from,
                "start_step": args.start_step,
            }
        )
    project = runtime.cfg_get(config, "logging.wandb_project", "world-model-rl")
    entity = runtime.cfg_get(config, "logging.wandb_entity", None)
    run_name = runtime.cfg_get(config, "logging.run_name", "phase3-qam-turn-openpi")
    mode = runtime.cfg_get(config, "logging.wandb_mode", None)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        config=wandb_cfg,
    )
    log(f"W&B logging enabled: project={project}, run={run.name}")
    return run


def noise_schedule(t, h: float, mode: str):
    """Noise scale used by the QAM residual."""
    eps = 1e-3
    if mode == "qam_sqrt":
        return jnp.sqrt(2.0 * (1.0 - t + h) / jnp.maximum(t + h, eps))
    if mode == "one_minus_t":
        return jnp.maximum(1.0 - t, eps)
    if mode == "t":
        return jnp.maximum(t, eps)
    raise ValueError(f"Unknown qam_sigma_schedule={mode!r}")


def qam_time(t_openpi, mode: str):
    """Map OpenPI denoising time to the time convention used by QAM sigma."""
    if mode == "one_minus_t":
        return 1.0 - t_openpi
    if mode == "identity":
        return t_openpi
    raise ValueError(f"Unknown qam_time_transform={mode!r}")


def qam_path_indices(n_flow_steps: int, qam_path_samples: int) -> tuple[int, ...]:
    if qam_path_samples <= 0:
        raise ValueError(f"qam_path_samples must be positive, got {qam_path_samples}")
    if qam_path_samples >= n_flow_steps:
        return tuple(range(n_flow_steps))
    if qam_path_samples == 1:
        return (n_flow_steps - 1,)
    return tuple(np.linspace(0, n_flow_steps - 1, qam_path_samples, dtype=np.int32).tolist())


def masked_tree_l2_norm(tree, mask):
    total = jnp.array(0.0, dtype=jnp.float32)
    for leaf, is_trainable in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(mask)):
        if hasattr(leaf, "shape") and bool(is_trainable):
            total = total + jnp.sum(jnp.square(leaf.astype(jnp.float32)))
    return jnp.sqrt(total)


def make_fsdp_sharding(mesh, tree, min_size_mbytes: int):
    from openpi.training import sharding as openpi_sharding

    return openpi_sharding.fsdp_sharding(tree, mesh, min_size_mbytes=min_size_mbytes, log=False)


def make_trainable_mask(tree, regex: str):
    pattern = re.compile(regex)

    def match(path, leaf):
        del leaf
        return bool(pattern.search(jax.tree_util.keystr(path)))

    return jax.tree_util.tree_map_with_path(match, tree)


def mask_stats(tree, mask) -> tuple[int, int]:
    leaves = jax.tree_util.tree_leaves(tree)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    total = sum(1 for x in leaves if hasattr(x, "shape"))
    trainable = sum(1 for x, m in zip(leaves, mask_leaves) if hasattr(x, "shape") and bool(m))
    return trainable, total
