"""Collect pi0 success branches from failed RoboCasa rollouts.

This script starts from failure episodes in a LeRobot-style RoboCasa dataset,
replays to an anchor before the detected failure onset, lets pi0 take over,
and writes only successful branches as a new success dataset.

By default the output episode is only the pi0 suffix starting at the anchor.
This avoids visual discontinuities when the source dataset does not contain a
full MuJoCo flattened sim state for the anchor frame. If --include-prefix is
set, the output episode is:

    original failure prefix [0, anchor) + pi0 suffix starting at anchor

In either mode, the anchor row's action is pi0's action, not the original
failed action. This is important for world-model training because each LeRobot
row is an (observation_t, action_t) transition.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOCASA_MJCF_TMPDIR", "/tmp/robocasa_mjcf_tmp")


try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - fallback for lean envs
    import yaml

    class _AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def _attrify(x):
        if isinstance(x, dict):
            return _AttrDict({k: _attrify(v) for k, v in x.items()})
        if isinstance(x, list):
            return [_attrify(v) for v in x]
        return x

    class OmegaConf:  # type: ignore[no-redef]
        @staticmethod
        def load(path):
            with open(path, "r", encoding="utf-8") as f:
                return _attrify(yaml.safe_load(f))


_RELEASE_ROOT = Path(os.environ.get("REVAMP_RELEASE_ROOT", Path.cwd())).resolve()
ROBOCASA_ROOT = Path(os.environ.get("REVAMP_ROBOCASA_ROOT", _RELEASE_ROOT / "third_party" / "robocasa")).resolve()
ROBOSUITE_ROOT_RAW = os.environ.get("REVAMP_ROBOSUITE_ROOT", "")
ROBOSUITE_ROOT = Path(ROBOSUITE_ROOT_RAW).resolve() if ROBOSUITE_ROOT_RAW else None
ROBOCASA_SCRIPTS = ROBOCASA_ROOT / "robocasa" / "scripts"
DEFAULT_OPENPI_ROOT = Path(os.environ.get("REVAMP_OPENPI_ROOT", _RELEASE_ROOT / "third_party" / "openpi")).resolve()
if ROBOSUITE_ROOT is not None and str(ROBOSUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOSUITE_ROOT))
if str(ROBOCASA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_ROOT))
if str(ROBOCASA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_SCRIPTS))
for _openpi_path in (
    DEFAULT_OPENPI_ROOT / "src",
    DEFAULT_OPENPI_ROOT / "packages" / "openpi-client" / "src",
):
    if _openpi_path.exists() and str(_openpi_path) not in sys.path:
        sys.path.insert(0, str(_openpi_path))

import robocasa.utils.lerobot_utils as LU  # noqa: E402
from generate_contact_features import (  # noqa: E402
    extract_contact_features_for_current_state,
    get_finger_geom_ids,
    get_robot_geom_ids,
    write_contact_features,
)
from generate_fail_data import (  # noqa: E402
    GenConfig,
    SourceEpisode,
    build_env,
    build_output_features,
    check_success,
    copy_source_metadata_templates,
    ensure_dir,
    extract_state_from_obs,
    finalize_output_dataset,
    get_observations,
    json_dump,
    restore_env_state,
)
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402


DEFAULT_SRC_ROOT = str(_RELEASE_ROOT / "datasets/origin")
DEFAULT_OUT_ROOT = str(_RELEASE_ROOT / "outputs/turn_on_sink_faucet/online_branch_success")
DEFAULT_CONFIG = str(
    _RELEASE_ROOT
    / "configs/world_model/phase3_qam_turn_openpi_two_chunk_imag.source.yaml"
)


def _log(msg: str) -> None:
    print(f"[branch_collect] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Policy server did not open {host}:{port}") from last_error


def _start_policy_server(
    *,
    openpi_root: Path,
    config_name: str,
    checkpoint: Path,
    prompt: str,
    port: int,
    server_gpu: str | None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if server_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = server_gpu

    cmd = [
        sys.executable,
        str(openpi_root / "scripts" / "serve_policy.py"),
        f"--port={port}",
        f"--default-prompt={prompt}",
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={checkpoint}",
    ]
    _log("starting policy server: " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(openpi_root), env=env)


def _load_openpi_robocasa_main(openpi_root: Path):
    src = openpi_root / "src"
    client_src = openpi_root / "packages" / "openpi-client" / "src"
    for p in (src, client_src):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    module_path = openpi_root / "examples" / "robocasa" / "main.py"
    spec = importlib.util.spec_from_file_location("openpi_robocasa_eval_main", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenPI RoboCasa helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _existing_episode_count(dataset_root: Path) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return 0
    return sum(1 for line in episodes_path.read_text().splitlines() if line.strip())


def _has_existing_dataset(dataset_root: Path) -> bool:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    return episodes_path.exists() and _existing_episode_count(dataset_root) > 0


class _DatasetContext:
    def __init__(self, dataset_dir: Path, env_meta: dict[str, Any]):
        self.dataset_dir = dataset_dir
        self.env_meta = env_meta
        self.info = _read_json(dataset_dir / "meta" / "info.json")


def _load_env_meta(dataset_dir: Path) -> dict[str, Any]:
    try:
        return LU.get_env_metadata(dataset_dir)
    except Exception as exc:
        raise RuntimeError(
            f"{dataset_dir}/extras/dataset_meta.json does not expose env_args; "
            "the source dataset should carry RoboCasa env_args directly."
        ) from exc


def _create_or_open_output_dataset(
    *,
    src_dataset_dir: Path,
    out_dataset_dir: Path,
    cfg: GenConfig,
    overwrite: bool,
    env_meta: dict[str, Any],
):
    if out_dataset_dir.exists() and overwrite:
        shutil.rmtree(out_dataset_dir)

    reader = _DatasetContext(src_dataset_dir, env_meta)
    if _has_existing_dataset(out_dataset_dir):
        dataset = LU.LerobotDatasetWrapper(
            repo_id=out_dataset_dir.name,
            root=out_dataset_dir,
            download_videos=cfg.save_video,
        )
        if cfg.save_video and (cfg.image_writer_threads or cfg.image_writer_processes):
            dataset.start_image_writer(
                num_processes=cfg.image_writer_processes,
                num_threads=cfg.image_writer_threads,
            )
        return dataset, reader

    if out_dataset_dir.exists():
        shutil.rmtree(out_dataset_dir)
    features = build_output_features(reader.info, include_videos=cfg.save_video)
    dataset = LU.LerobotDatasetWrapper.create(
        repo_id=out_dataset_dir.name,
        root=out_dataset_dir,
        fps=reader.info["fps"],
        robot_type=reader.info.get("robot_type"),
        features=features,
        use_videos=cfg.save_video,
        image_writer_threads=cfg.image_writer_threads if cfg.save_video else 0,
        image_writer_processes=cfg.image_writer_processes if cfg.save_video else 0,
    )
    copy_source_metadata_templates(src_dataset_dir, out_dataset_dir)
    extras_dir = out_dataset_dir / "extras"
    ensure_dir(extras_dir)
    source_meta = _read_json(src_dataset_dir / "extras" / "dataset_meta.json")
    source_meta["env_args"] = env_meta
    source_meta["branch_success_generation"] = {
        "source_dataset_dir": str(src_dataset_dir.resolve()),
        "generator": "world_model/scripts/collect_pi0_branch_success_rollouts.py",
        "save_video": cfg.save_video,
    }
    json_dump(source_meta, extras_dir / "dataset_meta.json")
    return dataset, reader


def _load_source_episode(src_failure_dir: Path, episode_index: int, env_meta: dict[str, Any]) -> SourceEpisode:
    ep_meta = LU.get_episode_meta(src_failure_dir, episode_index)
    states = LU.get_episode_states(src_failure_dir, episode_index)
    actions = LU.get_episode_actions(src_failure_dir, episode_index)
    model_xml = None
    try:
        model_xml = LU.get_episode_model_xml(src_failure_dir, episode_index)
    except FileNotFoundError:
        pass
    return SourceEpisode(
        episode_index=episode_index,
        episode_id=f"episode_{episode_index:06d}",
        actions=actions.astype(np.float32),
        model_xml=model_xml,
        initial_state=states[0].astype(np.float64),
        sim_states=states.astype(np.float64),
        ep_meta=ep_meta,
        env_meta=env_meta,
        task_name=env_meta["env_name"],
        layout_id=ep_meta.get("layout_id"),
        style_id=ep_meta.get("style_id"),
        instruction=ep_meta.get("lang"),
    )


def _current_model_xml(env) -> str:
    if hasattr(env, "model") and hasattr(env.model, "get_xml"):
        return env.model.get_xml()
    if hasattr(env, "sim") and hasattr(env.sim, "model") and hasattr(env.sim.model, "get_xml"):
        return env.sim.model.get_xml()
    raise RuntimeError("Could not get current MuJoCo model XML from env")


def _episode_parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    info = _read_json(dataset_dir / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return dataset_dir / template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _render_frames(env, camera_names: tuple[str, ...], height: int, width: int) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        frame = env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        frames[camera_name] = np.ascontiguousarray(frame).astype(np.uint8)
    return frames


def _add_frame_checked(dataset, frame: dict[str, Any], task_lang: str) -> None:
    for key, feature in dataset.features.items():
        if key.startswith("annotation.") and key not in frame:
            dtype = np.int64 if str(feature.get("dtype", "int64")).startswith("int") else np.float32
            shape = tuple(feature.get("shape") or (1,))
            frame[key] = np.zeros(shape, dtype=dtype)
    try:
        dataset.add_frame(frame, task=task_lang)
    except ValueError as exc:
        shapes = {
            key: {
                "shape": tuple(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in frame.items()
        }
        raise ValueError(f"{exc}\nframe shapes: {shapes}") from exc


def _policy_image(frame: np.ndarray, resize_size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, resize_size, resize_size))


def _policy_state_from_obs(obs: Any, fallback_state: np.ndarray) -> np.ndarray:
    """Match the state order used by openpi/examples/robocasa/main.py.

    The LeRobot rows used by the world model keep RoboCasa's dataset order:
    base pose, eef relative pose, gripper. The deployed OpenPI evaluator,
    however, feeds pi0 eef relative pose first. Keep those two contracts
    separate so branch data remains WM-compatible while pi0 sees familiar
    observations.
    """
    if isinstance(obs, dict):
        raw_keys = (
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_gripper_qpos",
        )
        if all(k in obs for k in raw_keys):
            return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in raw_keys], axis=0).astype(np.float32)

        gym_keys = (
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.base_position",
            "state.base_rotation",
            "state.gripper_qpos",
        )
        if all(k in obs for k in gym_keys):
            return np.concatenate([np.asarray(obs[k]).reshape(-1) for k in gym_keys], axis=0).astype(np.float32)

    return np.asarray(fallback_state, dtype=np.float32)


def _datanew_action_to_env_action(action: np.ndarray) -> np.ndarray:
    """Convert stored ViVa/datanew action order to RoboCasa/OpenPI action order."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 12:
        return a
    return np.concatenate(
        [
            a[5:8],    # end_effector_position
            a[8:11],   # end_effector_rotation
            a[11:12],  # gripper_close
            a[0:4],    # base_motion
            a[4:5],    # control_mode
        ],
        axis=0,
    ).astype(np.float32)


def _env_action_to_datanew_action(action: np.ndarray) -> np.ndarray:
    """Convert RoboCasa/OpenPI action order to the WM dataset action order."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 12:
        return a
    return np.concatenate(
        [
            a[7:11],   # base_motion
            a[11:12],  # control_mode
            a[0:3],    # end_effector_position
            a[3:6],    # end_effector_rotation
            a[6:7],    # gripper_close
        ],
        axis=0,
    ).astype(np.float64)


def _rollout_state_to_datanew_state(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.shape[0] < 16:
        return s.astype(np.float64)
    return np.concatenate(
        [
            s[7:10],   # base_position
            s[10:14],  # base_rotation
            s[0:3],    # end_effector_position_relative
            s[3:7],    # end_effector_rotation_relative
            s[14:16],  # gripper_qpos
        ],
        axis=0,
    ).astype(np.float64)


def _load_source_rollout_npz(src_failure_dir: Path, episode_index: int) -> tuple[dict[str, np.ndarray] | None, str | None]:
    meta_path = src_failure_dir / "extras" / f"episode_{episode_index:06d}" / "rollout_meta.json"
    if not meta_path.exists():
        return None, None
    source_rollout = _read_json(meta_path).get("source_rollout")
    if not source_rollout:
        return None, None
    path = Path(source_rollout)
    if not path.exists():
        _log(f"warning: source_rollout missing for episode_{episode_index:06d}: {path}")
        return None, str(path)
    data = np.load(path)
    return {k: data[k] for k in data.files}, str(path)


def _reset_episode_start(env, ep: SourceEpisode) -> None:
    if ep.model_xml:
        restore_env_state(env, ep.model_xml, ep.ep_meta, ep.initial_state)
        return

    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep.ep_meta)
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep.ep_meta)
    if hasattr(env, "hard_reset"):
        env.hard_reset = True
    if hasattr(env, "deterministic_reset"):
        env.deterministic_reset = False
    env.reset()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _replay_prefix_to_anchor(
    env,
    ep: SourceEpisode,
    source_df: pd.DataFrame,
    anchor_frame: int,
    cfg: GenConfig,
) -> tuple[Any, dict[str, Any]]:
    """Reset the scene, record prefix frames, and replay failed actions to anchor."""
    _reset_episode_start(env, ep)
    current_obs = get_observations(env)
    contact_sets = _resolve_contact_sets(env)
    prefix = {
        "states": [],
        "actions": [],
        "sim_states": [],
        "contacts": [],
        "frames": {cam: [] for cam in cfg.camera_names},
        "handle_states": [],
    }
    for frame_idx in range(anchor_frame):
        if current_obs is None:
            current_obs = get_observations(env)
        frames = _render_frames(env, cfg.camera_names, cfg.camera_height, cfg.camera_width)
        row = source_df.iloc[frame_idx]
        prefix["states"].append(np.asarray(row["observation.state"], dtype=np.float64))
        prefix["actions"].append(np.asarray(row["action"], dtype=np.float64))
        prefix["sim_states"].append(np.array(env.sim.get_state().flatten(), dtype=np.float64))
        prefix["contacts"].append(_current_contact(env, contact_sets))
        prefix["handle_states"].append(env.sink.get_handle_state(env) if hasattr(env, "sink") else {})
        for cam, frame in frames.items():
            prefix["frames"][cam].append(frame)
        action = _datanew_action_to_env_action(source_df.iloc[frame_idx]["action"])
        current_obs, _reward, _done, _info = _safe_step(env, action)
        if hasattr(env, "update_state"):
            env.update_state()
    current_obs = get_observations(env) or current_obs
    return current_obs, prefix


def _query_policy(
    client,
    *,
    frames: dict[str, np.ndarray],
    state: np.ndarray,
    task_lang: str,
    resize_size: int,
    replan_steps: int,
) -> collections.deque:
    element = {
        "observation/image": _policy_image(frames["robot0_agentview_left"], resize_size),
        "observation/wrist_image": _policy_image(frames["robot0_eye_in_hand"], resize_size),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": task_lang,
    }
    action_chunk = client.infer(element)["actions"]
    if len(action_chunk) < replan_steps:
        raise RuntimeError(
            f"Policy returned {len(action_chunk)} actions, shorter than replan_steps={replan_steps}"
        )
    return collections.deque(np.asarray(a, dtype=np.float32) for a in action_chunk[:replan_steps])


def _safe_step(env, action: np.ndarray):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        raise RuntimeError(f"Unexpected env.step output length: {len(out)}")
    return obs, float(reward), bool(done), info


def _resolve_contact_sets(env):
    left_ids, right_ids = get_finger_geom_ids(env, arm="right")
    robot_ids = get_robot_geom_ids(env)
    return left_ids, right_ids, robot_ids


def _current_contact(env, contact_sets) -> np.ndarray:
    left_ids, right_ids, robot_ids = contact_sets
    return extract_contact_features_for_current_state(
        env,
        left_finger_geom_ids=left_ids,
        right_finger_geom_ids=right_ids,
        robot_geom_ids=robot_ids,
    )


def _rollout_pi0_suffix(
    *,
    env,
    ep: SourceEpisode,
    source_df: pd.DataFrame,
    anchor_frame: int,
    client,
    task_lang: str,
    cfg: GenConfig,
    resize_size: int,
    replan_steps: int,
    post_success_steps: int,
    max_suffix_steps: int,
    attempt_seed: int,
) -> dict[str, Any] | None:
    current_obs, prefix = _replay_prefix_to_anchor(env, ep, source_df, anchor_frame, cfg)
    if current_obs is None:
        raise RuntimeError("Could not read observation after branch reset")

    contact_sets = _resolve_contact_sets(env)
    action_plan: collections.deque = collections.deque()
    suffix = {
        "states": [],
        "actions": [],
        "sim_states": [],
        "contacts": [],
        "frames": {cam: [] for cam in cfg.camera_names},
        "rewards": [],
        "dones": [],
        "success_trace": [],
        "handle_states": [],
        "prefix": prefix,
    }
    first_success_step = None

    for local_step in range(max_suffix_steps):
        obs_state = extract_state_from_obs(current_obs)
        if obs_state is None:
            raise RuntimeError("Could not construct observation.state during branch rollout")
        frames = _render_frames(env, cfg.camera_names, cfg.camera_height, cfg.camera_width)
        policy_state = _policy_state_from_obs(current_obs, obs_state)

        if not action_plan:
            action_plan = _query_policy(
                client,
                frames=frames,
                state=policy_state,
                task_lang=task_lang,
                resize_size=resize_size,
                replan_steps=replan_steps,
            )
        action = np.asarray(action_plan.popleft(), dtype=np.float32)

        suffix["states"].append(np.asarray(obs_state, dtype=np.float64))
        suffix["actions"].append(_env_action_to_datanew_action(action))
        suffix["sim_states"].append(np.array(env.sim.get_state().flatten(), dtype=np.float64))
        suffix["contacts"].append(_current_contact(env, contact_sets))
        for cam, frame in frames.items():
            suffix["frames"][cam].append(frame)

        next_obs, reward, done, _info = _safe_step(env, action)
        success_now = check_success(env)
        if hasattr(env, "update_state"):
            env.update_state()
        refreshed_obs = get_observations(env)
        suffix["rewards"].append(float(reward))
        suffix["success_trace"].append(bool(success_now))
        suffix["handle_states"].append(env.sink.get_handle_state(env) if hasattr(env, "sink") else {})
        suffix["dones"].append(False)

        if success_now and first_success_step is None:
            first_success_step = local_step

        current_obs = refreshed_obs if refreshed_obs is not None else next_obs
        if first_success_step is not None and local_step >= first_success_step + post_success_steps:
            break
        if done and first_success_step is None:
            break

    if first_success_step is None:
        return None
    if suffix["dones"]:
        suffix["dones"][-1] = True
    suffix["first_success_step"] = int(first_success_step)
    suffix["attempt_seed"] = int(attempt_seed)
    return suffix


def _write_branch_episode(
    *,
    dataset,
    out_dataset_dir: Path,
    env,
    ep: SourceEpisode,
    source_df: pd.DataFrame,
    suffix: dict[str, Any],
    anchor_frame: int,
    failure_onset: int,
    task_lang: str,
    cfg: GenConfig,
    branch_meta: dict[str, Any],
    source_rollout_data: dict[str, np.ndarray] | None,
    source_rollout_path: str | None,
    include_prefix: bool,
) -> int:
    episode_index = _existing_episode_count(out_dataset_dir)
    full_sim_states: list[np.ndarray] = []
    full_contacts: list[np.ndarray] = []
    full_handle_states: list[dict[str, Any]] = []

    replay_prefix = suffix.get("prefix", {})
    prefix_len = len(replay_prefix.get("actions", []))
    if prefix_len != anchor_frame:
        raise RuntimeError(f"Recorded prefix length {prefix_len} != anchor_frame {anchor_frame}")

    use_source_prefix = source_rollout_data is not None and all(
        k in source_rollout_data
        for k in ("state", "policy_action", "contact", "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
    )

    if include_prefix:
        for frame_idx in range(prefix_len):
            if use_source_prefix:
                state = _rollout_state_to_datanew_state(source_rollout_data["state"][frame_idx])
                action = _env_action_to_datanew_action(source_rollout_data["policy_action"][frame_idx])
                contact = np.asarray(source_rollout_data["contact"][frame_idx], dtype=np.float32)
                frames = {
                    "robot0_agentview_left": np.asarray(source_rollout_data["robot0_agentview_left"][frame_idx], dtype=np.uint8),
                    "robot0_agentview_right": np.asarray(source_rollout_data["robot0_agentview_right"][frame_idx], dtype=np.uint8),
                    "robot0_eye_in_hand": np.asarray(source_rollout_data["robot0_eye_in_hand"][frame_idx], dtype=np.uint8),
                }
                extra_state = state
            else:
                state = np.asarray(replay_prefix["states"][frame_idx], dtype=np.float64)
                action = np.asarray(replay_prefix["actions"][frame_idx], dtype=np.float64)
                contact = np.asarray(replay_prefix["contacts"][frame_idx], dtype=np.float32)
                frames = {cam: replay_prefix["frames"][cam][frame_idx] for cam in cfg.camera_names}
                extra_state = state

            frame = {
                "observation.state": state,
                "action": action,
            }
            if cfg.save_video:
                for cam_name in cfg.camera_names:
                    frame[f"observation.images.{cam_name}"] = frames[cam_name]
            _add_frame_checked(dataset, frame, task_lang)
            full_sim_states.append(np.asarray(replay_prefix["sim_states"][frame_idx], dtype=np.float64))
            full_contacts.append(contact)
            full_handle_states.append(replay_prefix["handle_states"][frame_idx])

    for local_idx, action in enumerate(suffix["actions"]):
        frame = {
            "observation.state": np.asarray(suffix["states"][local_idx], dtype=np.float64),
            "action": np.asarray(action, dtype=np.float64),
        }
        if cfg.save_video:
            for cam_name in cfg.camera_names:
                frame[f"observation.images.{cam_name}"] = suffix["frames"][cam_name][local_idx]
        _add_frame_checked(dataset, frame, task_lang)
        full_sim_states.append(np.asarray(suffix["sim_states"][local_idx], dtype=np.float64))
        full_contacts.append(np.asarray(suffix["contacts"][local_idx], dtype=np.float32))
        full_handle_states.append(suffix["handle_states"][local_idx])

    dataset.save_episode()

    ep_dir = out_dataset_dir / "extras" / f"episode_{episode_index:06d}"
    ensure_dir(ep_dir)
    np.savez_compressed(ep_dir / "states.npz", states=np.asarray(full_sim_states, dtype=np.float64))
    np.save(ep_dir / "contact_features.npy", np.asarray(full_contacts, dtype=np.float32))
    write_contact_features(
        episode_dir=ep_dir,
        features=np.asarray(full_contacts, dtype=np.float32),
        overwrite=True,
        write_json_flag=True,
    )
    json_dump(ep.ep_meta, ep_dir / "ep_meta.json")
    json_dump(full_handle_states, ep_dir / "handle_states.json")
    model_xml = ep.model_xml or _current_model_xml(env)
    with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
        f.write(model_xml.encode("utf-8"))

    rollout_meta = {
        "env_name": ep.task_name,
        "split": "target",
        "episode_idx": int(episode_index),
        "success": True,
        "first_success_step": int((anchor_frame if include_prefix else 0) + suffix["first_success_step"]),
        "first_success_step_in_suffix": int(suffix["first_success_step"]),
        "task_lang": task_lang,
        "num_steps": int(len(full_sim_states)),
        "source_episode_index": int(ep.episode_index),
        "source_episode_id": ep.episode_id,
        "source_split": "failure",
        "failure_onset": int(failure_onset),
        "anchor_frame": int(anchor_frame),
        "anchor_offset_before_onset": int(failure_onset - anchor_frame),
        "pi0_branch": branch_meta,
        "source_rollout": source_rollout_path,
        "episode_construction": "prefix_plus_suffix" if include_prefix else "suffix_only",
        "saved_prefix": bool(include_prefix),
        "num_prefix_frames_saved": int(prefix_len if include_prefix else 0),
        "prefix_video_source": (
            ("source_rollout_npz" if use_source_prefix else "replayed_render") if include_prefix else "not_saved"
        ),
        "suffix_only_reason": (
            None
            if include_prefix
            else "source dataset lacks full MuJoCo anchor states; replayed suffix may not be visually continuous with original prefix"
        ),
        "layout_id": ep.layout_id,
        "style_id": ep.style_id,
        "ep_meta": ep.ep_meta,
        "success_trace_suffix": [bool(x) for x in suffix["success_trace"]],
        "camera_mapping_for_viva": {
            "cam_high": "robot0_agentview_left",
            "cam_left_wrist": "robot0_eye_in_hand",
            "cam_right_wrist": "robot0_agentview_right",
        },
    }
    json_dump(rollout_meta, ep_dir / "rollout_meta.json")
    json_dump(rollout_meta, ep_dir / "success_meta.json")
    json_dump(rollout_meta, ep_dir / "branch_meta.json")
    return episode_index


def _parse_offsets(raw: str) -> tuple[int, ...]:
    offsets = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not offsets:
        raise ValueError("At least one anchor offset is required")
    return offsets


def _candidate_failure_rows(trim_manifest_path: Path, max_source_episodes: int | None) -> list[dict[str, Any]]:
    manifest = _read_json(trim_manifest_path)
    if not isinstance(manifest, list):
        raise ValueError(f"Expected list trim manifest at {trim_manifest_path}")
    rows = [r for r in manifest if r.get("split") == "failure"]
    rows = [r for r in rows if r.get("failure_onset") is not None]
    rows.sort(key=lambda r: int(r["episode_index"]))
    if max_source_episodes is not None:
        rows = rows[:max_source_episodes]
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--src-failure-dir", default=None)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--no-start-policy-server", action="store_true")
    parser.add_argument("--server-gpu", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--post-success-steps", type=int, default=3)
    parser.add_argument("--anchor-offsets", default="40,35,30,25")
    parser.add_argument("--attempts-per-anchor", type=int, default=4)
    parser.add_argument("--max-suffix-steps", type=int, default=180)
    parser.add_argument("--max-source-episodes", type=int, default=None)
    parser.add_argument("--max-branch-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--include-prefix",
        action="store_true",
        help=(
            "Save the failure prefix before the pi0 suffix. Default is suffix-only because "
            "trimmed rollout data does not include full MuJoCo anchor states, so prefix/suffix "
            "video continuity is not guaranteed."
        ),
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OmegaConf.load(args.config)
    pi0_cfg = config.model.pi0
    openpi_root = Path(pi0_cfg.openpi_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint or pi0_cfg.checkpoint_path).expanduser().resolve()
    prompt = str(pi0_cfg.get("prompt", "turn on sink faucet"))

    src_root = Path(args.src_root).expanduser().resolve()
    src_failure_dir = Path(args.src_failure_dir).expanduser().resolve() if args.src_failure_dir else src_root / "failure"
    out_dataset_dir = Path(args.out_root).expanduser().resolve() / "success"
    env_meta = _load_env_meta(src_failure_dir)
    offsets = _parse_offsets(args.anchor_offsets)

    cfg = GenConfig(
        src_dataset_dir=str(src_failure_dir),
        out_dataset_dir=str(out_dataset_dir),
        env_name=str(env_meta["env_name"]),
        save_video=not args.no_video,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        overwrite_output=args.overwrite_output,
        seed=args.seed,
    )
    cfg.camera_names = ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")

    dataset, output_reader = _create_or_open_output_dataset(
        src_dataset_dir=src_failure_dir,
        out_dataset_dir=out_dataset_dir,
        cfg=cfg,
        overwrite=args.overwrite_output,
        env_meta=env_meta,
    )

    if args.max_branch_episodes == 0:
        summary = {
            "source_root": str(src_root),
            "source_failure_dir": str(src_failure_dir),
            "out_dataset_dir": str(out_dataset_dir),
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint),
            "anchor_offsets": list(offsets),
            "attempts_per_anchor": int(args.attempts_per_anchor),
            "post_success_steps": int(args.post_success_steps),
            "include_prefix": bool(args.include_prefix),
            "attempts": [],
            "num_success_branches": 0,
            "dry_run_zero_branches": True,
        }
        finalize_output_dataset(dataset, output_reader, cfg, summary)
        _log(f"dry run complete: initialized {out_dataset_dir}")
        return

    server = None
    if not args.no_start_policy_server:
        if not (checkpoint / "params").exists():
            raise FileNotFoundError(f"Expected OpenPI checkpoint with params/: {checkpoint}")
        if not (checkpoint / "assets").exists():
            raise FileNotFoundError(f"Expected OpenPI checkpoint with assets/: {checkpoint}")
        server = _start_policy_server(
            openpi_root=openpi_root,
            config_name=str(pi0_cfg.config_name),
            checkpoint=checkpoint,
            prompt=prompt,
            port=args.port,
            server_gpu=args.server_gpu,
        )
        _wait_for_port(args.host, args.port, args.startup_timeout_s)

    _load_openpi_robocasa_main(openpi_root)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    manifest_rows = _candidate_failure_rows(src_root / "trim_manifest.json", args.max_source_episodes)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "source_root": str(src_root),
        "source_failure_dir": str(src_failure_dir),
        "out_dataset_dir": str(out_dataset_dir),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "anchor_offsets": list(offsets),
        "attempts_per_anchor": int(args.attempts_per_anchor),
        "post_success_steps": int(args.post_success_steps),
        "include_prefix": bool(args.include_prefix),
        "episode_construction": "prefix_plus_suffix" if args.include_prefix else "suffix_only",
        "attempts": [],
        "num_success_branches": 0,
    }

    env = None
    try:
        for row in manifest_rows:
            if args.max_branch_episodes is not None and summary["num_success_branches"] >= args.max_branch_episodes:
                break

            source_ep_idx = int(row["episode_index"])
            failure_onset = int(row["failure_onset"])
            ep = _load_source_episode(src_failure_dir, source_ep_idx, env_meta)
            source_df = pd.read_parquet(_episode_parquet_path(src_failure_dir, source_ep_idx))
            task_lang = prompt
            extras_meta = src_failure_dir / "extras" / f"episode_{source_ep_idx:06d}" / "rollout_meta.json"
            if extras_meta.exists():
                task_lang = str(_read_json(extras_meta).get("task_lang", task_lang))
            source_rollout_data, source_rollout_path = (None, None)
            if args.include_prefix:
                source_rollout_data, source_rollout_path = _load_source_rollout_npz(src_failure_dir, source_ep_idx)

            if env is None:
                env = build_env(cfg, ep)

            episode_success = False
            for offset in offsets:
                if episode_success:
                    break
                anchor = max(0, min(failure_onset - int(offset), len(ep.actions) - 1))
                if anchor >= len(source_df):
                    continue
                for attempt in range(args.attempts_per_anchor):
                    if args.max_branch_episodes is not None and summary["num_success_branches"] >= args.max_branch_episodes:
                        break
                    attempt_seed = int(rng.integers(0, 2**31 - 1))
                    _log(
                        f"source_ep={source_ep_idx:06d} onset={failure_onset} "
                        f"anchor={anchor} offset={offset} attempt={attempt + 1}/{args.attempts_per_anchor}"
                    )
                    suffix = _rollout_pi0_suffix(
                        env=env,
                        ep=ep,
                        source_df=source_df,
                        anchor_frame=anchor,
                        client=client,
                        task_lang=task_lang,
                        cfg=cfg,
                        resize_size=args.resize_size,
                        replan_steps=args.replan_steps,
                        post_success_steps=args.post_success_steps,
                        max_suffix_steps=args.max_suffix_steps,
                        attempt_seed=attempt_seed,
                    )
                    attempt_meta = {
                        "source_episode_index": source_ep_idx,
                        "failure_onset": failure_onset,
                        "anchor_frame": anchor,
                        "anchor_offset_before_onset": int(failure_onset - anchor),
                        "requested_offset": int(offset),
                        "attempt": int(attempt),
                        "attempt_seed": attempt_seed,
                        "success": suffix is not None,
                    }
                    if suffix is None:
                        summary["attempts"].append(attempt_meta)
                        continue

                    branch_meta = {
                        **attempt_meta,
                        "pi0_checkpoint": str(checkpoint),
                        "policy_config": str(pi0_cfg.config_name),
                        "replan_steps": int(args.replan_steps),
                        "post_success_steps": int(args.post_success_steps),
                        "suffix_steps": int(len(suffix["actions"])),
                    }
                    new_ep = _write_branch_episode(
                        dataset=dataset,
                        out_dataset_dir=out_dataset_dir,
                        env=env,
                        ep=ep,
                        source_df=source_df,
                        suffix=suffix,
                        anchor_frame=anchor,
                        failure_onset=failure_onset,
                        task_lang=task_lang,
                        cfg=cfg,
                        branch_meta=branch_meta,
                        source_rollout_data=source_rollout_data,
                        source_rollout_path=source_rollout_path,
                        include_prefix=bool(args.include_prefix),
                    )
                    attempt_meta["output_episode_index"] = int(new_ep)
                    attempt_meta["suffix_steps"] = int(len(suffix["actions"]))
                    summary["attempts"].append(attempt_meta)
                    summary["num_success_branches"] += 1
                    episode_success = True
                    _log(f"saved branch success episode_{new_ep:06d} from source episode_{source_ep_idx:06d}")
                    break

        finalize_output_dataset(dataset, output_reader, cfg, summary)
        _log(f"done: saved {summary['num_success_branches']} success branches to {out_dataset_dir}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)


if __name__ == "__main__":
    main()
