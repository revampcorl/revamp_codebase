"""Phase 1 offline training (two-stage).

Stage 1 (`training.stage: 1`) — Dynamics only:
  - Loss: future_state MSE + Σ future_cam MSE
  - No Q, no policy. Train DiT to predict t+H state + 3 future cams from
    (current state, current cams, action chunk, language).
  - Run until dynamics converges, save checkpoint.

Stage 2 (`training.stage: 2`) — Fitted Q Evaluation (FQE) of π_demo:
  - Load Stage 1 dynamics weights, freeze (or stop-grad).
  - Loss: dynamics (optional, skipped under DiT freeze) + chunk Q TD.
  - Bootstrap ā' = batch["next_action_chunk"] (taken from the dataset),
    so Q learns Q^{π_demo} with no learned policy in the loop.
  - Reward = PBRS shaping (ρ · α · ViVa) + sparse success/fail.
  - The trained Q is consumed by Stage 3 (QAM) to fine-tune π0.
    See PI0_FINETUNING.md and stage3_qam_train.py.

Usage:
    # Multi-GPU (default, plain DDP + bf16):
    accelerate launch --num_processes 8 --mixed_precision bf16 \
        -m revamp.stage1_2_world_model.train --config configs/world_model/phase1_wm_online_update.source.yaml

    # With DeepSpeed ZeRO-2 (opt-in for tighter memory / bigger batch):
    accelerate launch --num_processes 8 --mixed_precision bf16 \
        -m revamp.stage1_2_world_model.train --config configs/world_model/phase1_wm_online_update.source.yaml

    # Single GPU (debug):
    python -m revamp.stage1_2_world_model.train --config configs/world_model/phase1_wm_online_update.source.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings(
    "ignore",
    message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from revamp.common.constants import STAGE_DYNAMICS_ONLY, STAGE_RL
from revamp.common.dataset import WorldModelDataset, world_model_collate_fn
from revamp.common.rewards import chunk_pbrs, chunk_step_rewards, trust_weight_q
from revamp.common.world_model import WorldModel
from revamp.stage1_2_world_model.q_head import compute_td_target

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", is_main_process: bool = True):
    level = getattr(logging, log_level.upper()) if is_main_process else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ----------------------------------------------------------------------
# Logging integrations
# ----------------------------------------------------------------------

def _logging_targets(config) -> set[str]:
    report_to = getattr(getattr(config, "logging", None), "report_to", "tensorboard")
    if report_to is None:
        return set()
    if isinstance(report_to, str):
        values = [item.strip().lower() for item in report_to.split(",")]
    else:
        values = [str(item).strip().lower() for item in report_to]

    targets = set()
    for value in values:
        if value in {"all", "both"}:
            targets.update({"tensorboard", "wandb"})
        elif value not in {"", "none", "disabled", "false"}:
            targets.add(value)
    return targets


def use_tensorboard(config) -> bool:
    return "tensorboard" in _logging_targets(config)


def use_wandb(config) -> bool:
    return "wandb" in _logging_targets(config)


def init_wandb(config, *, enabled: bool, is_main_process: bool):
    if not enabled or not is_main_process:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "logging.report_to includes 'wandb', but wandb is not installed."
        ) from exc

    logging_config = getattr(config, "logging", None)
    project = (
        getattr(logging_config, "wandb_project", None)
        or getattr(logging_config, "project", None)
        or "world-model-rl"
    )
    entity = getattr(logging_config, "wandb_entity", None) or getattr(logging_config, "entity", None)
    mode = getattr(logging_config, "wandb_mode", None) or os.environ.get("WANDB_MODE")
    tags = getattr(logging_config, "wandb_tags", None)
    run_name = getattr(logging_config, "run_name", None)
    notes = getattr(logging_config, "wandb_notes", None)
    run_id = getattr(logging_config, "wandb_id", None)
    resume = getattr(logging_config, "wandb_resume", None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        resume=resume,
        mode=mode,
        tags=list(tags) if tags is not None else None,
        notes=notes,
        config=OmegaConf.to_container(config, resolve=True),
        dir=str(config.system.checkpoint_dir),
    )
    logger.info(f"Weights & Biases: project={project}, name={run_name}, url={run.url}")
    return run


def finish_wandb(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def update_ema_metrics(
    ema_metrics: Dict[str, float],
    metrics: Dict[str, float],
    decay: float,
) -> Dict[str, float]:
    """Update scalar metric EMAs and return the current EMA values."""
    out: Dict[str, float] = {}
    skip = {"step", "stage"}
    for key, value in metrics.items():
        if key in skip or not isinstance(value, (int, float)):
            continue
        value_f = float(value)
        if key not in ema_metrics:
            ema_metrics[key] = value_f
        else:
            ema_metrics[key] = decay * ema_metrics[key] + (1.0 - decay) * value_f
        out[key] = ema_metrics[key]
    return out


# ----------------------------------------------------------------------
# Stage 1: dynamics-only step
# ----------------------------------------------------------------------

def train_step_stage1(
    world_model,                # DDP-wrapped
    batch: Dict[str, torch.Tensor],
    optimizer_wm,
    scheduler_wm,
    accelerator: Accelerator,
    config,
    t5_embeddings: torch.Tensor,
    metrics: Dict[str, float],
):
    state = batch["state"]
    # [BUG-4] new T=3 layout: cam_*_segment is [B, 3, 12, H, W] covering
    # raw[t-3..t+8]. model.training_step VAE-encodes the cam_concat once
    # (sum-along-W) to latent [B, 48, 3, H', 3W'].
    cam_left_segment = batch["cam_left_segment"]
    cam_right_segment = batch["cam_right_segment"]
    cam_high_segment = batch["cam_high_segment"]
    action_chunk = batch["action_chunk"]

    unwrapped = accelerator.unwrap_model(world_model)

    with accelerator.accumulate(world_model):
        out = unwrapped.training_step(
            cam_left_segment=cam_left_segment,
            cam_right_segment=cam_right_segment,
            cam_high_segment=cam_high_segment,
            state=state,
            action_chunk=action_chunk,
            t5_embeddings=t5_embeddings,
            stage=STAGE_DYNAMICS_ONLY,
        )
        accelerator.backward(out["total_loss"])
        if accelerator.sync_gradients and config.training.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(
                world_model.parameters(), config.training.grad_clip_norm,
            )
        optimizer_wm.step()
        scheduler_wm.step()
        optimizer_wm.zero_grad()

    metrics.update({
        "total_loss": float(out["total_loss"].item()),
        "dynamics_loss": float(out["dynamics_loss"].item()),
        # [BUG-4] future_state_loss and per-cam losses removed
        # (Forward B deleted; cam_concat is encoded as one tensor).
        "future_cam_loss": float(out["future_cam_loss"].item()),
        "sigma_mean": out["sigma_mean"],
        "lr": float(scheduler_wm.get_last_lr()[0]),
    })


# ----------------------------------------------------------------------
# Stage 2: FQE step (Q TD with ā' from dataset, no policy update)
# ----------------------------------------------------------------------

def train_step_stage2(
    world_model,                # DDP-wrapped
    batch: Dict[str, torch.Tensor],
    optimizer_wm,
    scheduler_wm,
    accelerator: Accelerator,
    config,
    t5_embeddings: torch.Tensor,
    metrics: Dict[str, float],
):
    state = batch["state"]
    # [BUG-4] new T=3 layout
    cam_left_segment = batch["cam_left_segment"]
    cam_right_segment = batch["cam_right_segment"]
    cam_high_segment = batch["cam_high_segment"]
    # next-state quantities (for FQE bootstrap). future_state is the dataset
    # ground-truth s_{t+H}; future_cam_* single-frame fields are still yielded
    # by VivaDataset for legacy compatibility.
    future_state = batch["future_state"]
    action_chunk = batch["action_chunk"]
    next_action_chunk = batch["next_action_chunk"]   # FQE ā'
    success_chunk = batch["success_chunk"]
    fail_chunk = batch["fail_chunk"]
    viva_chunk = batch["viva_chunk"]                 # [B, H+1] dense Φ
    source_flag = batch["source_flag"]

    B = state.shape[0]
    H = action_chunk.shape[1]
    device = accelerator.device
    gamma = config.training.gamma
    alpha = config.training.alpha
    beta_succ = config.training.beta_succ
    beta_fail = config.training.beta_fail

    # Phase 1 ρ ≡ 1 across the whole chunk.
    u_dummy = torch.zeros(B, device=device)
    rho_chunk = torch.ones(B, H + 1, device=device)

    pbrs_rewards = chunk_pbrs(
        viva_chunk=viva_chunk * alpha,         # potential = α · ViVa(s)
        rho_chunk=rho_chunk,
        alpha=1.0,                             # already folded into viva_chunk
        gamma=gamma,
    )
    chunk_rewards = chunk_step_rewards(
        viva_chunk=viva_chunk * alpha,         # potential = α · ViVa(s)
        rho_chunk=rho_chunk,
        success_mask_chunk=success_chunk,
        fail_mask_chunk=fail_chunk,
        intervention_mask_chunk=None,
        alpha=1.0,                             # already folded into viva_chunk
        gamma=gamma,
        beta_succ=beta_succ, beta_fail=beta_fail,
    )
    sparse_rewards = chunk_rewards - pbrs_rewards
    done_chunk = (success_chunk + fail_chunk).clamp(max=1.0)

    unwrapped = accelerator.unwrap_model(world_model)

    # ---- FQE bootstrap: ā' from dataset (next chunk executed by π_demo) ----
    # Q therefore fits Q^{π_demo}: no learned policy in the loop, target
    # is stationary, no Q–π circular coupling. The trained Q is consumed
    # by Stage 3 QAM. See PI0_FINETUNING.md.
    #
    # [BUG-4] next-state cond cam: segment slice covering raw[t+5..t+8],
    # i.e., the 4-frame cond window centered on s_{t+H} (H=8). Using the
    # last 4 raw frames of our current segment gives raw[t+5..t+8] which
    # ends at s_{t+H}, exactly the cond window we need for Q at s_{t+H}.
    next_cam_left_cond = cam_left_segment[:, :, -4:]      # raw[t+5..t+8]
    next_cam_right_cond = cam_right_segment[:, :, -4:]
    next_cam_high_cond = cam_high_segment[:, :, -4:]
    with torch.no_grad():
        bootstrap_q = unwrapped.forward_q_target_min(
            future_state, next_cam_left_cond, next_cam_right_cond, next_cam_high_cond,
            next_action_chunk, t5_embeddings,
            detach_dit=True,    # Stage 2: keep DiT detached
        )

    sample_weight_q = trust_weight_q(u_dummy, source_flag, phase=1)

    freeze_dit = config.training.get("freeze_dit_in_stage2", True)

    # ---- World model + Q step ----
    with accelerator.accumulate(world_model):
        out = unwrapped.training_step(
            cam_left_segment=cam_left_segment,
            cam_right_segment=cam_right_segment,
            cam_high_segment=cam_high_segment,
            state=state,
            action_chunk=action_chunk,
            t5_embeddings=t5_embeddings,
            stage=STAGE_RL,
            chunk_rewards=chunk_rewards,
            bootstrap_q_target=bootstrap_q,
            done_chunk=done_chunk,
            gamma=gamma,
            sample_weight_q=sample_weight_q,
            freeze_dit=freeze_dit,
        )
        accelerator.backward(out["total_loss"])
        if accelerator.sync_gradients and config.training.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(
                world_model.parameters(), config.training.grad_clip_norm,
            )
        optimizer_wm.step()
        scheduler_wm.step()
        optimizer_wm.zero_grad()

    # Polyak EMA on actual update boundary only (not per microbatch).
    if accelerator.sync_gradients:
        unwrapped.q_ensemble.polyak_update()

    with torch.no_grad():
        done_before = torch.cumsum(done_chunk, dim=1) - done_chunk
        reward_mask = (done_before <= 0).to(dtype=chunk_rewards.dtype)
        masked_rewards = chunk_rewards * reward_mask
        nonterminal = (done_chunk.sum(dim=1) <= 0).to(dtype=chunk_rewards.dtype)
        q_target = compute_td_target(
            chunk_rewards, bootstrap_q, gamma, done_chunk=done_chunk,
        )

    metrics.update({
        "total_loss": float(out["total_loss"].item()),
        "q_loss": float(out["q_loss"].item()),
        "q1_mean": out["q1_mean"],
        "q2_mean": out["q2_mean"],
        "reward_mean": float(chunk_rewards.mean().item()),
        "pbrs_reward_mean": float(pbrs_rewards.mean().item()),
        "sparse_reward_mean": float(sparse_rewards.mean().item()),
        "masked_reward_mean": float(masked_rewards.mean().item()),
        "bootstrap_q_mean": float(bootstrap_q.mean().item()),
        "nonterminal_frac": float(nonterminal.mean().item()),
        "q_target_mean": float(q_target.mean().item()),
        "q_target_std": float(q_target.float().std(unbiased=False).item()),
        "success_count_max": float(success_chunk.sum(dim=1).max().item()),
        "fail_count_max": float(fail_chunk.sum(dim=1).max().item()),
        "terminal_in_chunk_frac": float(
            ((success_chunk + fail_chunk).sum(dim=1) > 0).float().mean().item()
        ),
        "lr": float(scheduler_wm.get_last_lr()[0]),
    })
    if not freeze_dit:
        metrics.update({
            "dynamics_loss": float(out["dynamics_loss"].item()),
            # [BUG-4] future_state_loss removed (Forward B deleted)
            "future_cam_loss": float(out["future_cam_loss"].item()),
        })


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--deepspeed", type=str, default=None,
        help="Optional path to a DeepSpeed JSON config (e.g. ZeRO-2). "
             "Default is plain Accelerator DDP + bf16.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # ---- Build Accelerator ----
    checkpoint_dir = Path(config.system.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    project_config = ProjectConfiguration(
        project_dir=str(checkpoint_dir),
        total_limit=int(config.system.get("ckpt_total_limit", 20)),
    )
    accelerator = Accelerator(
        deepspeed_plugin=(
            DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed else None
        ),
        gradient_accumulation_steps=int(
            config.training.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision="bf16",
        project_config=project_config,
    )

    setup_logging(
        config.system.get("log_level", "INFO"),
        is_main_process=accelerator.is_main_process,
    )
    logger.info(f"Loaded config: {args.config}")
    logger.info(
        f"Accelerator: num_processes={accelerator.num_processes}, "
        f"mixed_precision={accelerator.mixed_precision}, "
        f"deepspeed={'on' if args.deepspeed else 'off'}, "
        f"grad_accum={accelerator.gradient_accumulation_steps}"
    )

    accelerator.wait_for_everyone()

    tb_writer = None
    if accelerator.is_main_process and use_tensorboard(config):
        tb_log_dir = checkpoint_dir / config.logging.get("tensorboard_log_dir", "tensorboard")
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard: {tb_log_dir}")

    wandb_run = init_wandb(
        config,
        enabled=use_wandb(config),
        is_main_process=accelerator.is_main_process,
    )

    stage = int(config.training.get("stage", STAGE_DYNAMICS_ONLY))
    logger.info(f"Training stage: {stage}")

    # ---- Build world model (no manual device placement; accelerator handles it) ----
    logger.info("Building WorldModel ...")
    world_model = WorldModel(config)

    # ---- Stage 2: optionally freeze the backbone (BEFORE optimizer/prepare) ----
    if stage == STAGE_RL and config.training.get("freeze_dit_in_stage2", True):
        for p in world_model.video_model.wan_model.parameters():
            p.requires_grad = False
        logger.info("Stage 2: WAN DiT frozen.")

    # ---- Optimizer (only params with requires_grad) ----
    optimizer_wm = torch.optim.AdamW(
        [p for p in world_model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # ---- LR scheduler: linear warmup → constant (both stages) ----
    warmup_steps = int(config.training.get("warmup_steps", 0))
    scheduler_wm = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer_wm,
        num_warmup_steps=warmup_steps,
    )

    # ---- Dataset ----
    task_configs = [
        {"task_id": tc.task_id, "data_paths": list(tc.data_paths)}
        for tc in config.dataset.tasks
    ]
    dataset = WorldModelDataset(
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

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        collate_fn=world_model_collate_fn,
        pin_memory=config.system.pin_memory,
        shuffle=True,
    )

    # ---- accelerator.prepare ----
    world_model, optimizer_wm, scheduler_wm, dataloader = accelerator.prepare(
        world_model, optimizer_wm, scheduler_wm, dataloader,
    )

    # ---- T5 embeddings (not trainable; lives on accelerator.device) ----
    t5_emb_path = config.dataset.tasks[0].t5_embedding_path
    dtype = accelerator.unwrap_model(world_model).dtype
    t5_emb = torch.load(t5_emb_path, map_location="cpu").to(
        device=accelerator.device, dtype=dtype,
    )
    if t5_emb.dim() == 2:
        t5_emb = t5_emb.unsqueeze(0)

    # ---- Resume ----
    start_step = 0
    resume_path = config.resume.get("checkpoint_path", None)
    if resume_path:
        resume_path = Path(resume_path)
        only_model = bool(config.resume.get("only_model", False))
        reset_step = bool(config.resume.get("reset_step", False))
        if only_model:
            # Stage transition: only model weights (no optim / sched / step).
            # accelerator.save_state may write pytorch_model.bin or model.safetensors.
            model_bin = resume_path / "pytorch_model.bin"
            model_safe = resume_path / "model.safetensors"
            if model_bin.exists():
                state_dict = torch.load(model_bin, map_location="cpu")
            elif model_safe.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(model_safe), device="cpu")
            else:
                raise FileNotFoundError(
                    f"only_model=true but neither {model_bin} nor {model_safe} exists. "
                    f"Re-save the source ckpt with accelerator.save_state."
                )
            incompatible = accelerator.unwrap_model(world_model).load_state_dict(
                state_dict, strict=False,
            )
            if hasattr(incompatible, "missing_keys"):
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
            else:
                missing = list(incompatible[0])
                unexpected = list(incompatible[1])
            logger.info(
                "Loaded model-only state from %s: missing=%d unexpected=%d",
                resume_path, len(missing), len(unexpected),
            )
            if missing:
                logger.info("First missing keys: %s", missing[:20])
            if unexpected:
                logger.info("First unexpected keys: %s", unexpected[:20])
        else:
            accelerator.load_state(str(resume_path))
            meta_path = resume_path / "meta.pt"
            if meta_path.exists() and not reset_step:
                meta = torch.load(meta_path, map_location="cpu")
                start_step = int(meta.get("step", 0))
            logger.info(
                f"Resumed full state from {resume_path} at step {start_step}"
            )

    # ---- Train loop ----
    log_file = checkpoint_dir / "training_log.json"
    log_records = []

    step = start_step
    max_steps = int(config.training.get("max_steps", None) or 1_000_000)
    log_interval = int(config.system.log_interval)
    save_interval = int(config.system.save_interval)
    ema_decay = float(config.logging.get("ema_decay", 0.98))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(f"logging.ema_decay must be in [0, 1), got {ema_decay}")
    ema_metrics: Dict[str, float] = {}

    logger.info(f"Starting Stage {stage} training for up to {max_steps} steps.")
    epoch = 0
    pbar = tqdm(
        initial=step,
        total=max_steps,
        desc=f"Stage {stage}",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
        smoothing=0.1,
    )

    try:
        while step < max_steps:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            for batch in dataloader:
                if batch is None:
                    raise RuntimeError(
                        "Dataloader produced an empty batch. In distributed "
                        "training this would desynchronize ranks; inspect "
                        "dataset warnings above for the bad samples."
                    )
                B = batch["state"].shape[0]
                t5_batch = t5_emb.expand(B, -1, -1)
                metrics: Dict[str, float] = {"step": step, "stage": stage}
                step_start = time.time()

                try:
                    if stage == STAGE_DYNAMICS_ONLY:
                        train_step_stage1(
                            world_model=world_model, batch=batch,
                            optimizer_wm=optimizer_wm,
                            scheduler_wm=scheduler_wm,
                            accelerator=accelerator,
                            config=config,
                            t5_embeddings=t5_batch, metrics=metrics,
                        )
                    else:
                        train_step_stage2(
                            world_model=world_model, batch=batch,
                            optimizer_wm=optimizer_wm,
                            scheduler_wm=scheduler_wm,
                            accelerator=accelerator,
                            config=config,
                            t5_embeddings=t5_batch, metrics=metrics,
                        )
                except Exception as e:
                    logger.exception(f"Step {step} failed: {e}")
                    # In DDP, letting only one rank skip a backward pass causes
                    # the other ranks to hang in NCCL collectives. Fail fast so
                    # the real error is visible instead of surfacing later as a
                    # watchdog timeout.
                    raise

                # Only count step / log / save on actual update boundaries.
                if not accelerator.sync_gradients:
                    continue

                metrics["step_time"] = time.time() - step_start
                ema_payload = update_ema_metrics(ema_metrics, metrics, ema_decay)

                if accelerator.is_main_process:
                    pbar.set_postfix({
                        "loss": f"{metrics.get('total_loss', 0):.4f}",
                        "dyn": f"{metrics.get('dynamics_loss', 0):.4f}",
                        "fcam": f"{metrics.get('future_cam_loss', 0):.4f}",
                        "q": f"{metrics.get('q_loss', 0):.4f}",
                        "lr": f"{metrics.get('lr', 0):.2e}",
                    }, refresh=False)
                    pbar.update(1)

                if step % log_interval == 0 and accelerator.is_main_process:
                    parts = [
                        f"step={step}",
                        f"stage={stage}",
                        f"L_total={metrics.get('total_loss', 0):.4f}",
                    ]
                    if "dynamics_loss" in metrics:
                        parts.append(f"L_dyn={metrics['dynamics_loss']:.4f}")
                        parts.append(f"L_fcam={metrics['future_cam_loss']:.4f}")
                    parts.append(f"L_Q={metrics.get('q_loss', 0):.4f}")
                    parts.append(f"q1={metrics.get('q1_mean', 0):.3f}")
                    parts.append(f"boot={metrics.get('bootstrap_q_mean', 0):.3f}")
                    parts.append(f"lr={metrics.get('lr', 0):.2e}")
                    logger.info(" ".join(parts))

                    log_records.append(metrics)
                    with open(log_file, "w") as f:
                        json.dump(log_records, f, indent=2)

                    log_payload = {
                        f"train/{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }
                    log_payload.update({
                        f"train_ema/{k}": v
                        for k, v in ema_payload.items()
                    })
                    if tb_writer is not None:
                        for k, v in log_payload.items():
                            tb_writer.add_scalar(k, v, step)
                        tb_writer.flush()
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=step)

                if step > 0 and step % save_interval == 0:
                    save_dir = (
                        checkpoint_dir / f"world_model_stage{stage}_step_{step}"
                    )
                    accelerator.save_state(str(save_dir))
                    if accelerator.is_main_process:
                        meta = {
                            "step": step,
                            "stage": stage,
                            "config": OmegaConf.to_container(config, resolve=True),
                        }
                        torch.save(meta, save_dir / "meta.pt")
                        logger.info(f"Saved checkpoint to {save_dir}")
                    accelerator.wait_for_everyone()

                step += 1
                if step >= max_steps:
                    break
            epoch += 1

        logger.info("Training complete.")
    finally:
        pbar.close()
        if tb_writer is not None:
            tb_writer.close()
        accelerator.end_training()
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
