#!/usr/bin/env python3

"""
Generate failure trajectories for RoboCasa LeRobot datasets.

This script is tailored for manipulation tasks like TurnOnSinkFaucet:
- read successful episodes from an existing RoboCasa LeRobot dataset
- restore the exact scene XML and simulator state
- find when the end effector gets close to the faucet handle
- perturb actions around that contact window
- keep only failed rollouts
- write the result directly as a new LeRobot dataset

The generated dataset contains standard LeRobot files under:
- meta/
- data/
- videos/ (optional)
- extras/

The extras directory also stores simulator states and generation metadata so the
failures can be replayed exactly inside RoboCasa.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import robosuite
from robosuite import make

import robocasa.utils.lerobot_utils as LU


OBS_STATE_KEYS = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)

ACTION_SLICES = {
    "eef_pos": slice(0, 3),
    "eef_rot": slice(3, 6),
    "gripper": slice(6, 7),
    "base": slice(7, 11),
    "control_mode": slice(11, 12),
}


@dataclass
class GenConfig:
    src_dataset_dir: str = "datasets/origin/success"
    out_dataset_dir: str = "outputs/generated_failures/TurnOnSinkFaucet"

    env_name: str = "TurnOnSinkFaucet"
    robots: str = "PandaOmron"
    control_freq: int = 20

    camera_names: Tuple[str, ...] = (
        "robot0_eye_in_hand",
        "robot0_agentview_left",
        "robot0_agentview_right",
    )
    camera_height: int = 256
    camera_width: int = 256
    save_video: bool = True
    video_fps: int = 20

    start_source_episode: int = 0
    max_source_episodes: Optional[int] = None
    failure_modes: Tuple[str, ...] = (
        "eef_pose_noise",
        "eef_pose_drift",
        "reverse_rotation",
    )

    fallback_late_start_ratio_min: float = 0.70
    fallback_late_start_ratio_max: float = 0.90

    handle_distance_threshold: float = 0.25
    handle_distance_fallback_threshold: float = 0.20
    proximity_lead_steps_min: int = 2
    proximity_lead_steps_max: int = 8
    critical_lead_steps_min: int = 6
    critical_lead_steps_max: int = 16
    handle_motion_epsilon: float = 0.03

    perturb_window_min: int = 20
    perturb_window_max: int = 50
    smooth_noise_knot_stride: int = 10
    window_fade_ratio: float = 0.1
    drift_rise_ratio: float = 0.1
    drift_settle_ratio: float = 0.1
    drift_residual_ratio: float = 0.10
    reverse_rotation_blend_peak: float = 0.50

    # Translation noise is intentionally stronger than before so the robot is
    # more likely to miss or poorly engage the faucet handle.
    eef_pos_noise_sigma_min: float = 0.05
    eef_pos_noise_sigma_max: float = 0.6
    eef_rot_noise_sigma_min: float = 0.1
    eef_rot_noise_sigma_max: float = 0.6

    image_writer_threads: int = 8
    image_writer_processes: int = 0
    save_success_rollouts: bool = False
    overwrite_output: bool = False
    seed: int = 0


CFG = GenConfig()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_ready(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    return x


def json_dump(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(obj), f, ensure_ascii=False, indent=2)


def prepare_output_root(out_root: Path, overwrite: bool) -> None:
    if out_root.exists():
        if overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(
                f"Output dataset directory already exists: {out_root}. "
                "Pass --overwrite-output to replace it."
            )


def get_observations(env) -> Optional[dict]:
    getter = getattr(env, "_get_observations", None)
    if getter is None:
        return None
    try:
        return getter(force_update=True)
    except TypeError:
        return getter()


def extract_state_from_obs(obs: Any) -> Optional[np.ndarray]:
    if not isinstance(obs, dict):
        return None

    if all(k in obs for k in OBS_STATE_KEYS):
        chunks = [to_numpy(obs[k]).reshape(-1) for k in OBS_STATE_KEYS]
        return np.concatenate(chunks, axis=0).astype(np.float64)

    if "state" in obs:
        return to_numpy(obs["state"]).astype(np.float64)

    if "robot0_proprio-state" in obs:
        return to_numpy(obs["robot0_proprio-state"]).astype(np.float64)

    return None


def restore_env_state(env, model_xml: str, ep_meta: dict, sim_state: np.ndarray) -> None:
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)

    env.reset()
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        xml = postprocess_model_xml(model_xml)
    else:
        xml = env.edit_model_xml(model_xml)

    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(sim_state)
    env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def check_success(env) -> bool:
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    raise RuntimeError("Environment does not expose a success check method.")


def get_eef_position(env) -> np.ndarray:
    eef_site_id = env.robots[0].eef_site_id
    if isinstance(eef_site_id, dict):
        site_id = eef_site_id.get("right")
        if site_id is None:
            site_id = next(iter(eef_site_id.values()))
    else:
        site_id = eef_site_id
    return np.array(env.sim.data.site_xpos[site_id], dtype=np.float64)


def get_handle_position(env) -> Tuple[np.ndarray, str]:
    prefix = env.sink.naming_prefix
    candidates = [
        ("body", f"{prefix}handle"),
        ("geom", f"{prefix}handle_main"),
        ("site", f"{prefix}default_site"),
    ]

    for kind, name in candidates:
        try:
            if kind == "body":
                idx = env.sim.model.body_name2id(name)
                return np.array(env.sim.data.body_xpos[idx], dtype=np.float64), name
            if kind == "geom":
                idx = env.sim.model.geom_name2id(name)
                return np.array(env.sim.data.geom_xpos[idx], dtype=np.float64), name
            if kind == "site":
                idx = env.sim.model.site_name2id(name)
                return np.array(env.sim.data.site_xpos[idx], dtype=np.float64), name
        except Exception:
            continue

    raise RuntimeError(
        f"Could not resolve sink handle pose for naming prefix '{prefix}'."
    )


def build_output_features(source_info: dict, include_videos: bool) -> dict:
    features = deepcopy(source_info["features"])
    for feature_cfg in features.values():
        if "shape" in feature_cfg and feature_cfg["shape"] is not None:
            feature_cfg["shape"] = tuple(feature_cfg["shape"])
    if include_videos:
        return features
    return {
        key: value
        for key, value in features.items()
        if value.get("dtype") != "video"
    }


def copy_source_metadata_templates(src_dataset_dir: Path, out_root: Path) -> None:
    src_meta = src_dataset_dir / "meta"
    dst_meta = out_root / "meta"
    for fname in ("modality.json", "embodiment.json"):
        src = src_meta / fname
        if src.exists():
            shutil.copyfile(src, dst_meta / fname)


def write_dataset_meta(
    src_dataset_dir: Path,
    out_root: Path,
    cfg: GenConfig,
) -> None:
    extras_dir = out_root / "extras"
    ensure_dir(extras_dir)

    dataset_meta = load_json(src_dataset_dir / "extras" / "dataset_meta.json")
    dataset_meta["failure_generation"] = {
        "source_dataset_dir": str(src_dataset_dir.resolve()),
        "generator": "robocasa/scripts/generate_fail_data.py",
        "save_video": cfg.save_video,
        "seed": cfg.seed,
        "handle_distance_threshold": cfg.handle_distance_threshold,
    }
    json_dump(dataset_meta, extras_dir / "dataset_meta.json")
    json_dump(cfg.__dict__, extras_dir / "failure_generation_config.json")


def load_task_entries(tasks_path: Path) -> List[dict]:
    if not tasks_path.exists():
        return []
    entries = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def append_task_entry_if_missing(tasks_path: Path, task_name: str) -> int:
    entries = load_task_entries(tasks_path)
    existing = {entry["task"]: entry["task_index"] for entry in entries}
    if task_name in existing:
        return existing[task_name]

    next_index = 0 if not entries else max(entry["task_index"] for entry in entries) + 1
    with open(tasks_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": next_index, "task": task_name}) + "\n")
    return next_index


@dataclass
class SourceEpisode:
    episode_index: int
    episode_id: str
    actions: np.ndarray
    model_xml: str
    initial_state: np.ndarray
    sim_states: np.ndarray
    ep_meta: Dict[str, Any]
    env_meta: Dict[str, Any]
    task_name: str
    layout_id: Optional[int]
    style_id: Optional[int]
    instruction: Optional[str]


class SuccessDatasetReader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.env_meta = LU.get_env_metadata(self.dataset_dir)
        self.info = load_json(self.dataset_dir / "meta" / "info.json")

    def load_episodes(
        self,
        start_episode: int = 0,
        max_episodes: Optional[int] = None,
    ) -> List[SourceEpisode]:
        episode_dirs = LU.get_episodes(self.dataset_dir)
        if start_episode < 0:
            raise ValueError(
                f"start_episode must be non-negative, got {start_episode}."
            )
        episode_dirs = episode_dirs[start_episode:]
        if max_episodes is not None:
            episode_dirs = episode_dirs[:max_episodes]

        episodes: List[SourceEpisode] = []
        for ep_dir in episode_dirs:
            ep_num = int(ep_dir.name.split("_")[-1])
            ep_meta = LU.get_episode_meta(self.dataset_dir, ep_num)
            states = LU.get_episode_states(self.dataset_dir, ep_num)
            actions = LU.get_episode_actions(self.dataset_dir, ep_num)
            model_xml = LU.get_episode_model_xml(self.dataset_dir, ep_num)

            if len(states) != len(actions):
                raise ValueError(
                    f"Episode {ep_num} has mismatched states/actions lengths: "
                    f"{len(states)} vs {len(actions)}"
                )

            episodes.append(
                SourceEpisode(
                    episode_index=ep_num,
                    episode_id=f"episode_{ep_num:06d}",
                    actions=actions.astype(np.float32),
                    model_xml=model_xml,
                    initial_state=states[0].astype(np.float64),
                    sim_states=states.astype(np.float64),
                    ep_meta=ep_meta,
                    env_meta=deepcopy(self.env_meta),
                    task_name=self.env_meta["env_name"],
                    layout_id=ep_meta.get("layout_id"),
                    style_id=ep_meta.get("style_id"),
                    instruction=ep_meta.get("lang"),
                )
            )

        return episodes


def build_env(cfg: GenConfig, ep: SourceEpisode):
    env_kwargs = deepcopy(ep.env_meta["env_kwargs"])
    env_kwargs["env_name"] = ep.env_meta["env_name"]
    env_kwargs["robots"] = cfg.robots
    env_kwargs["control_freq"] = cfg.control_freq
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = cfg.save_video
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["render_gpu_device_id"] = getattr(
        cfg,
        "render_gpu_device_id",
        env_kwargs.get("render_gpu_device_id", -1),
    )
    env_kwargs["camera_names"] = list(cfg.camera_names)
    env_kwargs["camera_heights"] = cfg.camera_height
    env_kwargs["camera_widths"] = cfg.camera_width
    return make(**env_kwargs)


def locate_critical_step(
    env,
    ep: SourceEpisode,
    rng: np.random.Generator,
    cfg: GenConfig,
) -> Tuple[int, Dict[str, Any]]:
    restore_env_state(env, ep.model_xml, ep.ep_meta, ep.initial_state)

    handle_pos, handle_name = get_handle_position(env)
    initial_handle_state = env.sink.get_handle_state(env)
    initial_handle_joint = float(initial_handle_state.get("handle_joint", 0.0))

    first_handle_motion_step = None
    first_water_on_step = None
    first_proximity_step = None
    closest_handle_step = None
    closest_handle_distance = float("inf")

    handle_trace: List[float] = []
    handle_distance_trace: List[float] = []

    for step_idx, sim_state in enumerate(ep.sim_states):
        env.sim.set_state_from_flattened(sim_state)
        env.sim.forward()
        if hasattr(env, "update_state"):
            env.update_state()

        handle_state = env.sink.get_handle_state(env)
        handle_joint = float(handle_state.get("handle_joint", 0.0))
        handle_trace.append(handle_joint)

        if (
            first_handle_motion_step is None
            and abs(handle_joint - initial_handle_joint) > cfg.handle_motion_epsilon
        ):
            first_handle_motion_step = step_idx

        if first_water_on_step is None and bool(handle_state.get("water_on", False)):
            first_water_on_step = step_idx

        eef_pos = get_eef_position(env)
        handle_pos, _ = get_handle_position(env)
        distance = float(np.linalg.norm(eef_pos - handle_pos))
        handle_distance_trace.append(distance)

        if distance < closest_handle_distance:
            closest_handle_distance = distance
            closest_handle_step = step_idx

        if first_proximity_step is None and distance <= cfg.handle_distance_threshold:
            first_proximity_step = step_idx

    event_step = None
    detection_mode = None
    if first_proximity_step is not None:
        event_step = first_proximity_step
        detection_mode = "handle_proximity"
    elif (
        closest_handle_step is not None
        and closest_handle_distance <= cfg.handle_distance_fallback_threshold
    ):
        event_step = closest_handle_step
        detection_mode = "closest_handle_distance"
    elif first_handle_motion_step is not None:
        event_step = first_handle_motion_step
        detection_mode = "handle_motion"
    elif first_water_on_step is not None:
        event_step = first_water_on_step
        detection_mode = "water_on"
    else:
        event_step = int(
            len(ep.actions)
            * rng.uniform(
                cfg.fallback_late_start_ratio_min,
                cfg.fallback_late_start_ratio_max,
            )
        )
        detection_mode = "fallback_ratio"

    if detection_mode in {"handle_proximity", "closest_handle_distance"}:
        lead = int(
            rng.integers(cfg.proximity_lead_steps_min, cfg.proximity_lead_steps_max + 1)
        )
    else:
        lead = int(
            rng.integers(cfg.critical_lead_steps_min, cfg.critical_lead_steps_max + 1)
        )

    anchor_step = max(0, min(len(ep.actions) - 1, event_step - lead))
    meta = {
        "anchor_step": anchor_step,
        "event_step": int(event_step),
        "lead_steps": lead,
        "detection_mode": detection_mode,
        "handle_name": handle_name,
        "first_handle_motion_step": first_handle_motion_step,
        "first_water_on_step": first_water_on_step,
        "first_proximity_step": first_proximity_step,
        "closest_handle_step": closest_handle_step,
        "closest_handle_distance": closest_handle_distance,
        "initial_handle_joint": initial_handle_joint,
        "handle_trace_preview": handle_trace[:20],
        "handle_distance_preview": handle_distance_trace[:20],
    }
    return anchor_step, meta


def sample_window(
    anchor_step: int,
    horizon: int,
    rng: np.random.Generator,
    cfg: GenConfig,
) -> Tuple[int, int]:
    window = int(rng.integers(cfg.perturb_window_min, cfg.perturb_window_max + 1))
    start = max(0, min(anchor_step, horizon - 1))
    stop = min(horizon, start + window)
    return start, stop


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_window_envelope(length: int, cfg: GenConfig) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)

    fade = max(1, int(round(length * cfg.window_fade_ratio)))
    if fade * 2 >= length:
        phase = np.linspace(0.0, np.pi, length, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(phase)

    envelope = np.ones((length,), dtype=np.float32)
    ramp = smoothstep01(np.linspace(0.0, 1.0, fade, dtype=np.float32))
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    return envelope


def build_drift_profile(length: int, cfg: GenConfig) -> np.ndarray:
    if length <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    if length == 1:
        return np.ones((1, 1), dtype=np.float32)

    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    rise_end = np.clip(cfg.drift_rise_ratio, 0.05, 0.90)
    settle_start = max(rise_end + 0.05, 1.0 - cfg.window_fade_ratio - 0.15)
    settle_start = min(settle_start, 0.90)

    profile = np.ones((length,), dtype=np.float32)
    rise_mask = t < rise_end
    profile[rise_mask] = smoothstep01(t[rise_mask] / rise_end)

    settle_mask = t > settle_start
    if np.any(settle_mask):
        settle_t = (t[settle_mask] - settle_start) / (1.0 - settle_start)
        settle = smoothstep01(settle_t)
        profile[settle_mask] = (
            1.0 + (cfg.drift_settle_ratio - 1.0) * settle
        )

    return profile[:, None]


def sample_smooth_offset(
    length: int,
    dims: int,
    sigma: float,
    rng: np.random.Generator,
    cfg: GenConfig,
    envelope: Optional[np.ndarray] = None,
) -> np.ndarray:
    if length <= 0:
        return np.zeros((0, dims), dtype=np.float32)

    if envelope is None:
        envelope = np.ones((length,), dtype=np.float32)

    num_knots = max(4, int(np.ceil(length / max(1, cfg.smooth_noise_knot_stride))) + 1)
    num_knots = min(length, num_knots)
    knot_steps = np.linspace(0, length - 1, num_knots, dtype=np.float32)
    frame_steps = np.arange(length, dtype=np.float32)
    knot_values = rng.normal(0.0, sigma, size=(num_knots, dims)).astype(np.float32)

    smooth = np.empty((length, dims), dtype=np.float32)
    for dim in range(dims):
        smooth[:, dim] = np.interp(frame_steps, knot_steps, knot_values[:, dim])

    return smooth * envelope[:, None]


def perturb_actions(
    actions: np.ndarray,
    anchor_step: int,
    mode: str,
    rng: np.random.Generator,
    cfg: GenConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = actions.copy()
    horizon, _ = out.shape
    start, stop = sample_window(anchor_step, horizon, rng, cfg)
    window_len = stop - start

    pos_sigma = float(
        rng.uniform(cfg.eef_pos_noise_sigma_min, cfg.eef_pos_noise_sigma_max)
    )
    rot_sigma = float(
        rng.uniform(cfg.eef_rot_noise_sigma_min, cfg.eef_rot_noise_sigma_max)
    )
    envelope = build_window_envelope(window_len, cfg)

    if mode == "eef_pose_noise":
        out[start:stop, ACTION_SLICES["eef_pos"]] += sample_smooth_offset(
            window_len, 3, pos_sigma, rng, cfg, envelope
        )
        out[start:stop, ACTION_SLICES["eef_rot"]] += sample_smooth_offset(
            window_len, 3, rot_sigma, rng, cfg, envelope
        )
    elif mode == "eef_pose_drift":
        drift_profile = build_drift_profile(window_len, cfg)
        pos_bias = rng.normal(0.0, pos_sigma, size=(1, 3)).astype(np.float32)
        rot_bias = rng.normal(0.0, rot_sigma, size=(1, 3)).astype(np.float32)
        pos_residual = sample_smooth_offset(
            window_len,
            3,
            pos_sigma * cfg.drift_residual_ratio,
            rng,
            cfg,
            envelope,
        )
        rot_residual = sample_smooth_offset(
            window_len,
            3,
            rot_sigma * cfg.drift_residual_ratio,
            rng,
            cfg,
            envelope,
        )
        out[start:stop, ACTION_SLICES["eef_pos"]] += envelope[:, None] * (
            drift_profile * pos_bias
        ) + pos_residual
        out[start:stop, ACTION_SLICES["eef_rot"]] += envelope[:, None] * (
            drift_profile * rot_bias
        ) + rot_residual
    elif mode == "reverse_rotation":
        rot_slice = out[start:stop, ACTION_SLICES["eef_rot"]].copy()
        blend = (cfg.reverse_rotation_blend_peak * envelope)[:, None]
        out[start:stop, ACTION_SLICES["eef_rot"]] = (
            (1.0 - blend) * rot_slice + blend * (-rot_slice)
        )
        out[start:stop, ACTION_SLICES["eef_rot"]] += sample_smooth_offset(
            window_len, 3, rot_sigma * 0.35, rng, cfg, envelope
        )
        out[start:stop, ACTION_SLICES["eef_pos"]] += sample_smooth_offset(
            window_len, 3, pos_sigma * 0.30, rng, cfg, envelope
        )
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")

    return out, {
        "failure_type": mode,
        "anchor_step": anchor_step,
        "start_step": start,
        "stop_step": stop,
        "eef_pos_sigma": pos_sigma,
        "eef_rot_sigma": rot_sigma,
    }


@dataclass
class RolloutResult:
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    success: bool
    observation_state: np.ndarray
    sim_states: np.ndarray
    video_frames: Dict[str, List[np.ndarray]]
    handle_states: List[Dict[str, Any]]
    meta: Dict[str, Any]


def rollout_episode(
    env,
    ep: SourceEpisode,
    perturbed_actions: np.ndarray,
    generation_meta: Dict[str, Any],
    critical_meta: Dict[str, Any],
    cfg: GenConfig,
) -> RolloutResult:
    restore_env_state(env, ep.model_xml, ep.ep_meta, ep.initial_state)

    current_obs = get_observations(env)
    if current_obs is None:
        raise RuntimeError("Could not read observations from environment.")

    obs_state_list: List[np.ndarray] = []
    sim_state_list: List[np.ndarray] = [np.array(env.sim.get_state().flatten())]
    reward_list: List[float] = []
    done_list: List[bool] = []
    action_list: List[np.ndarray] = []
    handle_states: List[Dict[str, Any]] = []
    frames_by_cam: Dict[str, List[np.ndarray]] = {cam: [] for cam in cfg.camera_names}

    for action in perturbed_actions:
        obs_state = extract_state_from_obs(current_obs)
        if obs_state is None:
            raise RuntimeError("Failed to construct observation.state during rollout.")
        obs_state_list.append(obs_state)

        step_out = env.step(action)
        if len(step_out) == 4:
            next_obs, reward, done, _info = step_out
        elif len(step_out) == 5:
            next_obs, reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")

        success = check_success(env)
        action_list.append(action.copy())
        reward_list.append(float(reward))
        done_list.append(bool(done or success))
        sim_state_list.append(np.array(env.sim.get_state().flatten()))
        handle_states.append(env.sink.get_handle_state(env))

        if cfg.save_video:
            if hasattr(env, "update_state"):
                env.update_state()
            for camera_name in cfg.camera_names:
                frames_by_cam[camera_name].append(
                    env.sim.render(
                        height=cfg.camera_height,
                        width=cfg.camera_width,
                        camera_name=camera_name,
                    )[::-1]
                )

        current_obs = next_obs
        if done or success:
            break

    success = check_success(env)
    episode_info = {
        "source_dataset_dir": str(Path(cfg.src_dataset_dir).resolve()),
        "source_episode_id": ep.episode_id,
        "source_episode_index": ep.episode_index,
        "task_name": ep.task_name,
        "instruction": ep.instruction,
        "layout_id": ep.layout_id,
        "style_id": ep.style_id,
        "success": bool(success),
        "num_steps": len(action_list),
        "critical_meta": critical_meta,
        "generation_meta": generation_meta,
        "final_handle_state": handle_states[-1] if handle_states else None,
    }

    return RolloutResult(
        actions=np.asarray(action_list, dtype=np.float32),
        rewards=np.asarray(reward_list, dtype=np.float32),
        dones=np.asarray(done_list, dtype=np.bool_),
        success=bool(success),
        observation_state=np.stack(obs_state_list, axis=0).astype(np.float64),
        sim_states=np.asarray(sim_state_list, dtype=np.float64),
        video_frames=frames_by_cam,
        handle_states=handle_states,
        meta=episode_info,
    )


def save_failure_episode_extras(
    out_root: Path,
    episode_index: int,
    ep: SourceEpisode,
    result: RolloutResult,
) -> None:
    episode_dir = out_root / "extras" / f"episode_{episode_index:06d}"
    ensure_dir(episode_dir)

    np.savez_compressed(episode_dir / "states.npz", states=result.sim_states)
    json_dump(ep.ep_meta, episode_dir / "ep_meta.json")
    json_dump(result.meta, episode_dir / "failure_meta.json")
    json_dump(result.handle_states, episode_dir / "handle_states.json")

    with gzip.open(episode_dir / "model.xml.gz", "wb") as f:
        f.write(ep.model_xml.encode("utf-8"))


def write_rollout_to_lerobot(
    dataset,
    episode_index: int,
    ep: SourceEpisode,
    result: RolloutResult,
    cfg: GenConfig,
) -> None:
    task_description = ep.instruction or ep.task_name
    task_description_index = 0
    task_name_index = 1 if ep.task_name != task_description else 0

    for step_idx in range(len(result.actions)):
        frame = {
            "annotation.human.task_description": np.array(
                [task_description_index], dtype=np.int64
            ),
            "annotation.human.task_name": np.array([task_name_index], dtype=np.int64),
            "observation.state": result.observation_state[step_idx].astype(np.float64),
            "action": result.actions[step_idx].astype(np.float64),
            "next.reward": np.array([result.rewards[step_idx]], dtype=np.float32),
            "next.done": np.array([result.dones[step_idx]], dtype=np.bool_),
        }

        if cfg.save_video:
            for camera_name in cfg.camera_names:
                frame[f"observation.images.{camera_name}"] = result.video_frames[
                    camera_name
                ][step_idx]

        dataset.add_frame(frame, task=task_description)

    dataset.save_episode()
    save_failure_episode_extras(out_root=dataset.root, episode_index=episode_index, ep=ep, result=result)


def create_output_dataset(cfg: GenConfig, reader: SuccessDatasetReader):
    out_root = Path(cfg.out_dataset_dir)
    prepare_output_root(out_root, overwrite=cfg.overwrite_output)

    source_info = reader.info
    features = build_output_features(source_info, include_videos=cfg.save_video)
    dataset = LU.LerobotDatasetWrapper.create(
        repo_id=out_root.name,
        root=out_root,
        fps=source_info["fps"],
        robot_type=source_info.get("robot_type"),
        features=features,
        use_videos=cfg.save_video,
        image_writer_threads=cfg.image_writer_threads if cfg.save_video else 0,
        image_writer_processes=cfg.image_writer_processes if cfg.save_video else 0,
    )
    copy_source_metadata_templates(reader.dataset_dir, out_root)
    write_dataset_meta(reader.dataset_dir, out_root, cfg)
    return dataset


def finalize_output_dataset(
    dataset,
    reader: SuccessDatasetReader,
    cfg: GenConfig,
    summary: dict,
) -> None:
    tasks_path = dataset.root / "meta" / "tasks.jsonl"
    append_task_entry_if_missing(tasks_path, reader.env_meta["env_name"])

    parquet_paths = sorted((dataset.root / "data").rglob("episode_*.parquet"))
    if parquet_paths:
        stats = LU.calculate_dataset_statistics(parquet_paths)
        json_dump(stats, dataset.root / "meta" / "stats.json")

    json_dump(summary, dataset.root / "extras" / "generation_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dataset-dir", type=str, default=CFG.src_dataset_dir)
    parser.add_argument("--out-dataset-dir", type=str, default=CFG.out_dataset_dir)
    parser.add_argument(
        "--start-source-episode",
        type=int,
        default=CFG.start_source_episode,
        help="0-based source episode index to start reading from.",
    )
    parser.add_argument(
        "--num-source-episodes",
        type=int,
        default=None,
        help="Number of source episodes to load starting from --start-source-episode.",
    )
    parser.add_argument(
        "--max-source-episodes",
        type=int,
        default=CFG.max_source_episodes,
        help=(
            "Deprecated alias for --num-source-episodes. "
            "If both are provided, --num-source-episodes takes precedence."
        ),
    )
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--save-success-rollouts",
        action="store_true",
        help="Also save perturbed rollouts that still complete the task successfully.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> GenConfig:
    cfg = deepcopy(CFG)
    cfg.src_dataset_dir = args.src_dataset_dir
    cfg.out_dataset_dir = args.out_dataset_dir
    cfg.start_source_episode = args.start_source_episode
    cfg.max_source_episodes = (
        args.num_source_episodes
        if args.num_source_episodes is not None
        else args.max_source_episodes
    )
    if cfg.start_source_episode < 0:
        raise ValueError("--start-source-episode must be non-negative.")
    if cfg.max_source_episodes is not None and cfg.max_source_episodes <= 0:
        raise ValueError(
            "--num-source-episodes/--max-source-episodes must be positive."
        )
    cfg.seed = args.seed
    cfg.save_video = not args.no_video
    cfg.save_success_rollouts = args.save_success_rollouts
    cfg.overwrite_output = args.overwrite_output
    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    rng = np.random.default_rng(cfg.seed)

    reader = SuccessDatasetReader(cfg.src_dataset_dir)
    dataset = create_output_dataset(cfg, reader)
    episodes = reader.load_episodes(
        start_episode=cfg.start_source_episode,
        max_episodes=cfg.max_source_episodes,
    )

    summary = {
        "num_source_episodes": len(episodes),
        "source_episode_start_index": cfg.start_source_episode,
        "requested_num_source_episodes": cfg.max_source_episodes,
        "generated_failures": 0,
        "attempted_rollouts": 0,
        "saved_success_rollouts": 0,
        "saved_failure_rollouts": 0,
        "skipped_success_rollouts": 0,
        "save_success_rollouts": cfg.save_success_rollouts,
        "attempted_modes_per_episode": list(cfg.failure_modes),
    }

    saved_episode_index = 0
    for ep_idx, ep in enumerate(episodes):
        print(f"\n[INFO] Source episode {ep_idx + 1}/{len(episodes)}: {ep.episode_id}")

        for mode in cfg.failure_modes:
            summary["attempted_rollouts"] += 1

            env = build_env(cfg, ep)
            try:
                anchor_step, critical_meta = locate_critical_step(env, ep, rng, cfg)
                perturbed_actions, generation_meta = perturb_actions(
                    ep.actions,
                    anchor_step=anchor_step,
                    mode=mode,
                    rng=rng,
                    cfg=cfg,
                )
                result = rollout_episode(
                    env=env,
                    ep=ep,
                    perturbed_actions=perturbed_actions,
                    generation_meta=generation_meta,
                    critical_meta=critical_meta,
                    cfg=cfg,
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            if result.success:
                if not cfg.save_success_rollouts:
                    summary["skipped_success_rollouts"] += 1
                    print(
                        f"  mode {mode}: skipped success rollout "
                        f"(steps={len(result.actions)})"
                    )
                    continue
                summary["saved_success_rollouts"] += 1
                print(f"  mode {mode}: saving success rollout (steps={len(result.actions)})")
            else:
                summary["generated_failures"] += 1
                summary["saved_failure_rollouts"] += 1
                print(f"  mode {mode}: saving failure rollout (steps={len(result.actions)})")

            write_rollout_to_lerobot(
                dataset=dataset,
                episode_index=saved_episode_index,
                ep=ep,
                result=result,
                cfg=cfg,
            )
            saved_episode_index += 1

    finalize_output_dataset(dataset, reader, cfg, summary)
    print("\n[DONE]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
