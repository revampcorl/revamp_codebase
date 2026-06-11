"""Distributed JAX/OpenPI Stage 3 QAM trainer.

QAM update logic is adapted from the MIT-licensed "Q-learning with Adjoint
Matching" project and extended here for OpenPI, two-chunk imagination, and the
REVAMP world-model/Q-gradient server setup. See NOTICE for attribution.

This is the production-oriented replacement for the earlier single-process
JAX+PyTorch prototype:
  - JAX owns OpenPI params, optimizer, sharding, and checkpointing.
  - PyTorch Stage2 Q runs in a separate local Q-gradient server.
  - The trainer sends only action chunks and observation tensors to that
    server, receives dQ/da, then performs a JAX vector-field regression update.

Launch pattern:
  Terminal A, reserve one GPU for Q:
    CUDA_VISIBLE_DEVICES=0 python -m revamp.stage3_qam.q_gradient_server ...

  Terminal B, reserve remaining GPUs for JAX/OpenPI:
    CUDA_VISIBLE_DEVICES=1,2,3,4 python -m revamp.stage3_qam.train_policy ...
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from revamp.stage3_qam import runtime
from revamp.stage3_qam import train_support as support
from revamp.stage3_qam.train_cli import parse_args
from revamp.stage3_qam.train_cli import start_step_from_args
from revamp.stage3_qam.openpi_policy import OpenPiPolicyBridge


# ----------------------------------------------------------------------
# Logging / path setup
# ----------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[openpi_qam_dist] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _add_openpi_to_path(openpi_root: str) -> None:
    src = Path(openpi_root).expanduser().resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ----------------------------------------------------------------------
# OpenPI sampling
# ----------------------------------------------------------------------

def _sample_raw_actions(
    sample_actions_fn,
    params,
    rng,
    observation,
    *,
    replicated_sharding,
    act_stats: dict[str, np.ndarray],
    use_quantiles: bool,
    q_h: int,
    q_a: int,
    context: str,
):
    """Sample OpenPI actions and convert them back to the Q model's raw action space."""
    rng, sample_rng = jax.random.split(rng, 2)
    t0 = time.time()
    final_norm, (traj, times) = sample_actions_fn(
        params,
        jax.device_put(sample_rng, replicated_sharding),
        observation,
    )
    final_norm.block_until_ready()
    sample_s = time.time() - t0
    action_raw = runtime.unnormalize_actions(np.asarray(final_norm), act_stats, use_quantiles)[:, :q_h, :q_a]
    if not np.isfinite(action_raw).all():
        raise RuntimeError(f"OpenPI produced non-finite {context} actions during Stage3 QAM")
    return rng, action_raw, traj, times, sample_s


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    pi0_cfg = config.model.pi0
    policy_checkpoint_path = str(args.resume_from or pi0_cfg.checkpoint_path)
    start_step = start_step_from_args(args)
    _add_openpi_to_path(pi0_cfg.openpi_root)

    from openpi.training import config as _config
    from openpi.training import sharding as openpi_sharding

    # ---- Runtime mesh, logging, and external services ----
    fsdp_devices = int(args.num_fsdp_devices or jax.device_count())
    mesh = openpi_sharding.make_mesh(fsdp_devices)
    per_device_batch_size = runtime.cfg_get_int(
        config.training,
        "per_device_batch_size",
        runtime.cfg_get_int(config.training, "batch_size", 1),
    )
    batch_size = per_device_batch_size * jax.device_count()
    data_shard_factor = int(mesh.shape[openpi_sharding.BATCH_AXIS] * mesh.shape[openpi_sharding.FSDP_AXIS])
    if batch_size % data_shard_factor != 0:
        raise ValueError(
            f"global_batch_size={batch_size} must be divisible by "
            f"{data_shard_factor} for OpenPI DATA_AXIS sharding with mesh={mesh.shape}. "
            f"per_device_batch_size={per_device_batch_size}, visible devices={jax.device_count()}."
        )
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    action_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS, None, None),
    )
    traj_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(None, openpi_sharding.DATA_AXIS, None, None),
    )
    _log(f"jax devices={jax.devices()}, mesh={mesh.shape}")
    wandb_run = support.init_wandb(config, args, batch_size, per_device_batch_size, fsdp_devices, log=_log)
    q_payload_compression = str(
        args.q_payload_compression
        or config.system.get("q_payload_compression", "uncompressed")
    )
    q_client = runtime.QGradientClient(
        args.q_server_url,
        compressed_payload=(q_payload_compression == "compressed"),
    )
    _log(f"Q server payload mode={q_payload_compression}, url={args.q_server_url}")
    imag_settings = support.load_imag_qam_settings(config, args)
    imag_batches_done = 0
    imag_client = None
    if imag_settings.enabled:
        imag_client = runtime.WorldModelImagClient(
            imag_settings.server_url,
            compressed_payload=(imag_settings.payload_compression == "compressed"),
        )
        _log(
            "two-chunk imagination QAM enabled: "
            f"url={imag_settings.server_url}, lambda={imag_settings.lambda_weight}, "
            f"update_every={imag_settings.update_every}, warmup={imag_settings.warmup_steps}, "
            f"num_inference_steps={imag_settings.num_inference_steps}, "
            f"state_offset={imag_settings.state_offset}, "
            f"pred_history={imag_settings.pred_history_strategy}:{imag_settings.pred_history_frames}, "
            f"payload={imag_settings.payload_compression}"
        )

    # ---- OpenPI policy and trainable parameter mask ----
    # The bridge gives us OpenPI's loaded model and exact transform stack. It
    # lives only in this JAX process; PyTorch Q is in q_gradient_server.py.
    _log("loading OpenPI policy bridge")
    bridge = OpenPiPolicyBridge(
        openpi_root=pi0_cfg.openpi_root,
        config_name=pi0_cfg.config_name,
        checkpoint_path=policy_checkpoint_path,
        prompt=pi0_cfg.get("prompt", "turn on sink faucet"),
        q_action_horizon=pi0_cfg.get("q_action_horizon", config.common.action_chunk_length),
        q_action_dim=pi0_cfg.get("q_action_dim", config.common.action_dim),
        pi0_state_dim=pi0_cfg.get("pi0_state_dim", 16),
        num_flow_steps=pi0_cfg.get("num_flow_steps", 2),
        data_dirs=runtime.optional_container(pi0_cfg.get("data_dirs", None)),
    )
    policy = bridge.policy
    model = policy._model
    ref_model = copy.deepcopy(model)
    graphdef, params = nnx.split(model)
    ref_graphdef, ref_params = nnx.split(ref_model)

    param_sharding = support.make_fsdp_sharding(mesh, params, int(args.fsdp_min_mbytes))
    params = jax.device_put(params, param_sharding)
    ref_params = jax.device_put(ref_params, param_sharding)
    trainable_regex = str(
        config.training.get(
            "trainable_regex",
            "action_in_proj|action_time_mlp|action_out_proj|state_proj",
        )
    )
    trainable_mask = support.make_trainable_mask(params, trainable_regex)
    n_trainable, n_total = support.mask_stats(params, trainable_mask)
    if n_trainable == 0:
        raise ValueError(f"trainable_regex={trainable_regex!r} matched 0 parameter leaves")
    _log(f"trainable leaves={n_trainable}/{n_total}, regex={trainable_regex!r}")

    train_config = _config.get_config(pi0_cfg.config_name)
    pi0_data_dirs = runtime.optional_container(pi0_cfg.get("data_dirs", None))
    if pi0_data_dirs is not None:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, data_dirs=pi0_data_dirs),
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    use_quantiles = bool(data_config.use_quantile_norm)
    act_stats = runtime.action_norm_stats(policy_checkpoint_path)
    act_scale = runtime.normalization_scale(act_stats, use_quantiles)

    # ---- Optimizer, resume state, and QAM knobs ----
    lr = runtime.cfg_get_float(config.training, "learning_rate", 1e-5)
    tx = optax.chain(
        optax.clip_by_global_norm(runtime.cfg_get_float(config.training, "grad_clip_norm", 1.0)),
        optax.adamw(lr, weight_decay=runtime.cfg_get_float(config.training, "weight_decay", 0.01)),
    )
    opt_state = tx.init(params)
    opt_state_sharding = support.make_fsdp_sharding(mesh, opt_state, int(args.fsdp_min_mbytes))
    opt_state = jax.device_put(opt_state, opt_state_sharding)
    restored_full_state = False
    resume_dir = Path(policy_checkpoint_path)
    if args.resume_from and (resume_dir / "opt_state").exists():
        restored = runtime.restore_orbax_item(resume_dir / "opt_state")
        restored_opt_state = restored.get("opt_state", restored) if isinstance(restored, dict) else restored
        opt_state = jax.device_put(restored_opt_state, opt_state_sharding)
        restored_full_state = True
        _log(f"restored optimizer state from {resume_dir / 'opt_state'}")
    n_flow_steps = int(pi0_cfg.get("num_flow_steps", 2))
    q_h = int(pi0_cfg.get("q_action_horizon", config.common.action_chunk_length))
    q_a = int(pi0_cfg.get("q_action_dim", config.common.action_dim))
    qam_path_samples = runtime.cfg_get_int(config.training, "qam_path_samples", 1)
    qam_sigma_schedule = str(config.training.get("qam_sigma_schedule", "qam_sqrt"))
    qam_time_transform = str(config.training.get("qam_time_transform", "one_minus_t"))
    use_discrete_adjoint = bool(config.training.get("use_discrete_adjoint", True))
    h_flow = 1.0 / float(n_flow_steps)
    qam_path_indices = support.qam_path_indices(n_flow_steps, qam_path_samples)
    _log(
        "QAM vector-field loss uses "
        f"path_indices={qam_path_indices}, num_flow_steps={n_flow_steps}, "
        f"sigma_schedule={qam_sigma_schedule!r}, time_transform={qam_time_transform!r}, "
        f"discrete_adjoint={use_discrete_adjoint}"
    )
    adjoint_solver_steps_cfg = config.training.get("adjoint_solver_steps", None)
    if adjoint_solver_steps_cfg is not None and int(adjoint_solver_steps_cfg) != n_flow_steps:
        _log(
            "warning: training.adjoint_solver_steps is currently documentation-only in this "
            f"discrete OpenPI trainer; adjoint propagation follows model.pi0.num_flow_steps={n_flow_steps}, "
            f"not adjoint_solver_steps={int(adjoint_solver_steps_cfg)}"
        )
    if bool(config.training.get("use_amp", False)):
        _log(
            "warning: training.use_amp is not used by the JAX/OpenPI trainer; "
            "OpenPI/JAX dtype comes from the checkpoint/config, and Q-server precision is controlled in PyTorch"
        )
    reference_kl_weight = runtime.cfg_get_float(config.training, "reference_kl_weight", 0.0)
    metric_ema_beta = runtime.cfg_get_float(config.training, "metric_ema_beta", 0.98)
    if not 0.0 <= metric_ema_beta < 1.0:
        raise ValueError(f"training.metric_ema_beta must be in [0,1), got {metric_ema_beta}")
    if reference_kl_weight > 0:
        _log(f"explicit reference regularization enabled: weight={reference_kl_weight}")
    _log(f"metric EMA enabled: beta={metric_ema_beta}")

    # ---- JIT-compiled OpenPI sampling and QAM update ----
    @functools.partial(
        jax.jit,
        in_shardings=(param_sharding, replicated_sharding, data_sharding),
        out_shardings=(action_sharding, (traj_sharding, replicated_sharding)),
    )
    def sample_actions_jit(params, rng, observation):
        m = nnx.merge(graphdef, params)
        return m.sample_actions_with_traj(rng, observation, num_steps=n_flow_steps)

    metric_sharding = {
        "loss": replicated_sharding,
        "qam_loss": replicated_sharding,
        "reference_reg_loss": replicated_sharding,
        "trainable_update_norm": replicated_sharding,
    }

    @functools.partial(
        jax.jit,
        in_shardings=(
            param_sharding,
            param_sharding,
            opt_state_sharding,
            data_sharding,
            action_sharding,
            traj_sharding,
            replicated_sharding,
        ),
        out_shardings=(param_sharding, opt_state_sharding, metric_sharding),
    )
    def train_step(params, ref_params, opt_state, observation, g_norm_q, traj, times):
        """One QAM optimizer step on an already-sampled OpenPI flow trajectory."""
        def q_slice(x):
            return x[:, :q_h, :q_a]

        def reference_adjoint_path(ref, traj_sg, times_sg, g_sg):
            if not use_discrete_adjoint:
                return [g_sg] * n_flow_steps

            # Propagate the terminal critic gradient backward along the frozen
            # reference flow path, matching QAM's slow/base actor.
            adj_full = jnp.zeros_like(traj_sg[-1], dtype=jnp.float32)
            adj_full = adj_full.at[:, :q_h, :q_a].set(g_sg)
            adjs_rev = []
            for j in reversed(range(n_flow_steps)):
                t_j = times_sg[j]
                t_batch = jnp.broadcast_to(t_j, (traj_sg.shape[1],))

                def ref_vf(a):
                    return ref.vector_field(observation, a, t_batch).astype(jnp.float32)

                _, vjp_fn = jax.vjp(ref_vf, traj_sg[j])
                jt_adj = vjp_fn(adj_full)[0].astype(jnp.float32)
                # OpenPI integrates x_next = x_t - h * f(x_t, t), so the
                # cotangent update is g_t = g_next - h * J_f^T g_next.
                adj_full = adj_full - h_flow * jt_adj
                adjs_rev.append(q_slice(adj_full))
            return list(reversed(adjs_rev))

        def qam_loss_terms(m, ref, traj_i, t, adj_i):
            t_qam = support.qam_time(t, qam_time_transform)
            t_batch = jnp.broadcast_to(t, (traj_i.shape[0],))
            f_theta = m.vector_field(observation, traj_i, t_batch)
            f_ref = jax.lax.stop_gradient(ref.vector_field(observation, traj_i, t_batch))
            sigma = support.noise_schedule(t_qam, h_flow, qam_sigma_schedule)
            f_delta = q_slice(f_theta).astype(jnp.float32) - q_slice(f_ref).astype(jnp.float32)
            residual = 2.0 * f_delta / sigma + sigma * adj_i
            qam_loss = jnp.mean(jnp.square(residual))
            ref_reg = jnp.mean(jnp.square(f_delta / sigma))
            return qam_loss, ref_reg

        def loss_fn(params):
            effective_params = jax.tree.map(
                lambda p, m: p if m else jax.lax.stop_gradient(p),
                params,
                trainable_mask,
            )
            m = nnx.merge(graphdef, effective_params)
            ref = nnx.merge(ref_graphdef, ref_params)
            with openpi_sharding.set_mesh(mesh):
                traj_sg = jax.lax.stop_gradient(traj)
                times_sg = jax.lax.stop_gradient(times)
                g_sg = jax.lax.stop_gradient(g_norm_q)
                adjs = reference_adjoint_path(ref, traj_sg, times_sg, g_sg)

                loss = 0.0
                qam_loss_sum = 0.0
                ref_reg_sum = 0.0
                for i in qam_path_indices:
                    qam_loss_i, ref_reg_i = qam_loss_terms(m, ref, traj_sg[i], times_sg[i], adjs[i])
                    loss = loss + qam_loss_i + reference_kl_weight * ref_reg_i
                    qam_loss_sum = qam_loss_sum + qam_loss_i
                    ref_reg_sum = ref_reg_sum + ref_reg_i

                denom = float(len(qam_path_indices))
                metrics = {
                    "loss": loss / denom,
                    "qam_loss": qam_loss_sum / denom,
                    "reference_reg_loss": ref_reg_sum / denom,
                }
                return metrics["loss"], metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree.map(lambda g, m: g if m else jnp.zeros_like(g), grads, trainable_mask)
        updates, opt_state = tx.update(grads, opt_state, params)
        updates = jax.tree.map(lambda u, m: u if m else jnp.zeros_like(u), updates, trainable_mask)
        update_norm = support.masked_tree_l2_norm(updates, trainable_mask)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["trainable_update_norm"] = update_norm
        return params, opt_state, metrics

    def run_qam_update(params, opt_state, rng, batch, observation, *, context: str, grad_scale: float = 1.0):
        """Run the Stage3 update unit: sample policy action, query Q gradient, fit the vector field."""
        rng, action_raw, traj, times, sample_s = _sample_raw_actions(
            sample_actions_jit,
            params,
            rng,
            observation,
            replicated_sharding=replicated_sharding,
            act_stats=act_stats,
            use_quantiles=use_quantiles,
            q_h=q_h,
            q_a=q_a,
            context=context,
        )
        q_grad_raw, _q_values, q_mean, q_s = support.query_q_gradient(q_client, batch, action_raw)
        g_norm = -q_grad_raw * act_scale[:q_a][None, None, :] * float(grad_scale)

        t0 = time.time()
        params, opt_state, train_metrics = train_step(
            params,
            ref_params,
            opt_state,
            observation,
            jax.device_put(jnp.asarray(g_norm, dtype=jnp.float32), action_sharding),
            traj,
            jax.device_put(times, replicated_sharding),
        )
        train_metrics["loss"].block_until_ready()
        qam_metrics = {
            "q_mean": q_mean,
            "q_s": q_s,
            "train_s": time.time() - t0,
            "sample_s": sample_s,
            "action_raw_std": float(action_raw.std()),
        }
        return params, opt_state, rng, action_raw, train_metrics, qam_metrics

    def run_imagination_update(params, opt_state, rng, batch, final_raw, step: int, batches_done: int):
        """Optional A2 update: use the world model's predicted future as the next policy observation."""
        if imag_client is None or not imag_settings.should_update(step, batches_done):
            return params, opt_state, rng, batches_done, support.empty_imag_row()
        if "has_state_imag_anchor" in batch and not bool(batch["has_state_imag_anchor"].bool().all().item()):
            return params, opt_state, rng, batches_done, {
                "imag_enabled": False,
                "imag_skip_reason": "invalid_state_imag_anchor",
            }

        t0 = time.time()
        pred, _imag_server_timings = imag_client.post_predict(
            support.wm_imag_payload(
                batch,
                final_raw,
                num_inference_steps=imag_settings.num_inference_steps,
            )
        )
        imag_world_model_s = time.time() - t0
        imag_batch = support.build_two_chunk_imagined_batch(
            batch,
            pred,
            history_frames=imag_settings.pred_history_frames,
            history_strategy=imag_settings.pred_history_strategy,
        )
        observation2 = support.prepare_openpi_observation(policy, bridge, imag_batch)
        observation2 = jax.device_put(observation2, data_sharding)

        params, opt_state, rng, _final_raw2, imag_train_metrics, imag_qam_metrics = run_qam_update(
            params,
            opt_state,
            rng,
            imag_batch,
            observation2,
            context="A2",
            grad_scale=imag_settings.lambda_weight,
        )
        return params, opt_state, rng, batches_done + 1, support.imag_log_row(
            imag_settings,
            qam_metrics=imag_qam_metrics,
            train_metrics=imag_train_metrics,
            world_model_s=imag_world_model_s,
        )

    # ---- Dataset, preflight, and logging state ----
    _log("building dataset")
    dataset = support.build_dataset(config)
    dataloader, dataloader_seed, num_workers, dataloader_kwargs = support.build_train_dataloader(
        config,
        dataset,
        batch_size,
    )
    steps_per_epoch = max(len(dataloader), 1)
    resume_skip_batches = int(start_step % steps_per_epoch)
    _log(
        f"dataset ready len={len(dataset)}, "
        f"per_device_batch_size={per_device_batch_size}, global_batch_size={batch_size}, "
        f"num_workers={num_workers}, pin_memory={dataloader_kwargs['pin_memory']}"
    )

    support.run_q_preflight(
        dataset,
        batch_size=batch_size,
        q_h=q_h,
        q_a=q_a,
        q_client=q_client,
        log=_log,
    )

    max_steps = int(args.max_steps or config.training.get("max_steps", 1000))
    save_interval = int(config.system.get("save_interval", 0))
    out_dir = Path(args.output_dir or config.system.checkpoint_dir)
    assets_src = Path(policy_checkpoint_path) / "assets"
    rng = jax.random.key(0)
    trainer_state_path = resume_dir / "trainer_state.json"
    if args.resume_from and trainer_state_path.exists():
        trainer_state = json.loads(trainer_state_path.read_text())
        if "rng" in trainer_state:
            rng = runtime.rng_from_jsonable(trainer_state["rng"])
            _log(f"restored JAX rng from {trainer_state_path}")
    logs: list[dict[str, Any]] = []
    ema_state: dict[str, float] = {}
    ema_metric_keys = [
        "q_mean",
        "loss",
        "qam_loss",
        "reference_reg_loss",
        "trainable_update_norm",
        "action_raw_std",
        "imag_q_mean",
        "imag_loss",
        "imag_qam_loss",
        "imag_action_raw_std",
    ]
    last_saved_step = -1
    if args.skip_save:
        _log("skip-save enabled; periodic and final checkpoint saves are disabled")
    elif save_interval > 0:
        _log(f"periodic checkpoint saving enabled every {save_interval} steps")
    else:
        _log("periodic checkpoint saving disabled; final checkpoint will still be saved")

    # ---- Train loop ----
    step = start_step
    if start_step > 0:
        _log(
            f"resuming policy weights from {policy_checkpoint_path}; "
            f"step counter starts at {start_step} "
            f"({'optimizer/RNG restored' if restored_full_state else 'optimizer state is fresh'})"
        )
    pbar = tqdm(total=max_steps, desc="openpi_qam", dynamic_ncols=True)
    try:
        first_epoch_after_resume = True
        last_step_end = time.time()
        while step < max_steps:
            iterator = iter(dataloader)
            if first_epoch_after_resume and start_step > 0 and resume_skip_batches > 0:
                _log(f"skipping {resume_skip_batches} dataloader batches to restore resume cursor")
                for _ in range(resume_skip_batches):
                    next(iterator)
            first_epoch_after_resume = False
            for batch in iterator:
                data_wait_s = time.time() - last_step_end
                if batch is None:
                    raise RuntimeError(
                        "Dataloader produced an empty batch. Inspect dataset warnings above "
                        "for bad samples instead of silently skipping training."
                    )
                observation = support.prepare_openpi_observation(policy, bridge, batch)
                observation = jax.device_put(observation, data_sharding)
                # A1: update OpenPI on the real observation using Stage-2 Q gradients.
                params, opt_state, rng, final_raw, train_metrics, qam_metrics = run_qam_update(
                    params,
                    opt_state,
                    rng,
                    batch,
                    observation,
                    context="A1",
                )

                # A2: optionally let the world model imagine the next observation
                # and apply a second, down-weighted QAM update.
                params, opt_state, rng, imag_batches_done, imag_row = run_imagination_update(
                    params,
                    opt_state,
                    rng,
                    batch,
                    final_raw,
                    step,
                    imag_batches_done,
                )

                row = support.train_log_row(
                    step=step,
                    train_metrics=train_metrics,
                    qam_metrics=qam_metrics,
                    data_wait_s=data_wait_s,
                    imag_row=imag_row,
                )
                row.update(support.update_metric_ema(ema_state, row, ema_metric_keys, metric_ema_beta))
                logs.append(row)
                if wandb_run is not None:
                    wandb_run.log(support.wandb_metrics(row), step=step)
                pbar.set_postfix(support.progress_postfix(row))
                pbar.update(1)
                if step % args.log_interval == 0:
                    pbar.write(json.dumps(row))
                step += 1
                last_step_end = time.time()
                if (not args.skip_save) and save_interval > 0 and step % save_interval == 0:
                    pbar.write(f"[openpi_qam_dist] saving checkpoint at step={step}")
                    runtime.save_checkpoint(
                        out_dir=out_dir,
                        step=step,
                        params=params,
                        opt_state=opt_state,
                        rng=rng,
                        dataloader_seed=dataloader_seed,
                        logs=logs,
                        assets_src=assets_src,
                        wandb_run=wandb_run,
                        log=_log,
                    )
                    last_saved_step = step
                if step >= max_steps:
                    break
    finally:
        q_client.close()
        if imag_client is not None:
            imag_client.close()
        pbar.close()

    if args.skip_save:
        step_dir = out_dir / f"openpi_qam_dist_step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "openpi_qam_log.json").write_text(json.dumps(logs, indent=2) + "\n")
        _log(f"skip-save enabled; wrote log to {step_dir / 'openpi_qam_log.json'}")
        if wandb_run is not None:
            wandb_run.finish()
        return
    if last_saved_step != step:
        step_dir = runtime.save_checkpoint(
            out_dir=out_dir,
            step=step,
            params=params,
            opt_state=opt_state,
            rng=rng,
            dataloader_seed=dataloader_seed,
            logs=logs,
            assets_src=assets_src,
            wandb_run=wandb_run,
            log=_log,
        )
    else:
        step_dir = out_dir / f"openpi_qam_dist_step_{step}"
        _log(f"final step={step} already checkpointed at {step_dir}")
    if wandb_run is not None:
        wandb_run.summary["checkpoint_dir"] = str(step_dir)
        wandb_run.finish()


if __name__ == "__main__":
    main()
