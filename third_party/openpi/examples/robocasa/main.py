import collections
import dataclasses
import logging
import pathlib
import sys
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os

# Headless nodes need EGL for MuJoCo offscreen camera rendering. These must be
# set before importing robocasa / robosuite, which import MuJoCo and OpenGL.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.models.fixtures import FixtureType
import gymnasium as gym
from robocasa.utils.env_utils import convert_action

USE_CODE_CONFIG = True

RELEASE_ROOT = pathlib.Path(
    os.environ.get("REVAMP_RELEASE_ROOT", pathlib.Path(__file__).resolve().parents[4])
).expanduser().resolve()

REFERENCE_EPISODE_META_PATH = (
    RELEASE_ROOT
    / "datasets"
    / "robocasa_env_metadata"
    / "extras"
    / "episode_000001"
    / "ep_meta.json"
)

CODE_CONFIG = {
    "host": "127.0.0.1",
    "port": 8001,
    "resize_size": 224,
    "replan_steps": 5,
    "render_gpu_device_id": 0,
    "split": "target",
    "num_trials": 40,
    "post_success_steps": 5,
    "task_set": None,
    "env_names": ["TurnOnSinkFaucet"],
    "log_dir": str(RELEASE_ROOT / "checkpoints/openpi_pi0_turn"),
    "seed": 7,
    "save_rollouts": True,
    "rollout_dir": str(RELEASE_ROOT / "outputs/turn_on_sink_faucet/rollouts"),
    # Same-scene rollout: this ep_meta fixes layout/style/object config/camera config
    # and the original robot base pose from the reference episode.
    "reference_ep_meta_path": REFERENCE_EPISODE_META_PATH,
    # Set True to sample a fresh RoboCasa scene on each reset instead of replaying
    # reference_ep_meta_path. With split="target", the default candidate scenes are
    # RoboCasa target scenes [(1, 1), ..., (10, 10)].
    "randomize_scene": True,
    # Optional scene candidates used only when randomize_scene=True.
    # Examples: [[1, 1], [2, 2], [4, 4]] or "5x5" / "5x1".
    "scene_layout_and_style_ids": [[1,1],[2,2],[3,3]],
    # Alternative to scene_layout_and_style_ids: sample all layout/style products.
    # Examples: scene_layout_ids=[1, 2, 4], scene_style_ids=[1, 2, 4].
    "scene_layout_ids": None,
    "scene_style_ids": None,
    # Edit these offsets to perturb the robot mobile-base initial pose.
    # Units: meters for xyz, radians for roll/pitch/yaw.
    "robot_base_pos_offset": [0.0, 0.0, 0.0],
    "robot_base_ori_offset": [0.0, 0.0, 0.0],
    # Per-rollout random uniform perturbation range around the pose above.
    # These defaults randomize x/y within 5 cm and yaw within about 5.7 degrees.
    "robot_base_pos_random_range": [0.25, 0.25, 0.0],
    "robot_base_ori_random_range": [0.0, 0.0, 0.2],
    # Per-rollout random arm-joint perturbation after env.reset().
    # A scalar applies to every arm joint; units are radians.
    "robot_arm_joint_random_range": 0.25,
    "robot_arm_joint_offset": None,
    # If set, these absolute values replace the reference ep_meta base pose
    # before offsets are applied.
    "robot_base_pos_override": None,
    "robot_base_ori_override": None,
}

TASK_SET_ALIASES = {
    # Training/data-soup config names use these prefixes, while RoboCasa
    # TASK_SET_REGISTRY uses the bare eval group names.
    "target_atomic_seen": "atomic_seen",
}


def _normalize_bool_cli_args(argv: list[str]) -> list[str]:
    """Accept both Tyro boolean flags and shell-style `--flag True/False`."""
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}
    bool_flag_pairs = {
        "--args.save-rollouts": ("--args.save-rollouts", "--args.no-save-rollouts"),
        "--args.save_rollouts": ("--args.save-rollouts", "--args.no-save-rollouts"),
        "--args.randomize-scene": ("--args.randomize-scene", "--args.no-randomize-scene"),
        "--args.randomize_scene": ("--args.randomize-scene", "--args.no-randomize-scene"),
    }
    negative_flags = {
        "--args.no-save-rollouts": "--args.no-save-rollouts",
        "--args.no_save_rollouts": "--args.no-save-rollouts",
        "--args.no-randomize-scene": "--args.no-randomize-scene",
        "--args.no_randomize_scene": "--args.no-randomize-scene",
    }

    normalized = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in bool_flag_pairs:
            if i + 1 < len(argv) and argv[i + 1].lower() in truthy | falsy:
                value = argv[i + 1].lower()
                positive_flag, negative_flag = bool_flag_pairs[arg]
                normalized.append(positive_flag if value in truthy else negative_flag)
                i += 2
                continue
            normalized.append(bool_flag_pairs[arg][0])
        elif arg in negative_flags:
            normalized.append(negative_flags[arg])
        else:
            normalized.append(arg)
        i += 1
    return normalized

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    render_gpu_device_id: int = 0

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    post_success_steps: int = 25
    task_set: list[str] | None = None
    env_names: list[str] | None = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str | None = None

    seed: int = 7  # Random Seed (for reproducibility)
    save_rollouts: bool = False
    rollout_dir: str | None = None
    reference_ep_meta_path: str | None = None
    randomize_scene: bool = False
    scene_layout_and_style_ids: list[list[int]] | str | None = None
    scene_layout_ids: list[int] | int | None = None
    scene_style_ids: list[int] | int | None = None
    robot_base_pos_offset: list[float] | None = None
    robot_base_ori_offset: list[float] | None = None
    robot_base_pos_random_range: list[float] | None = None
    robot_base_ori_random_range: list[float] | None = None
    robot_arm_joint_random_range: float | list[float] | None = None
    robot_arm_joint_offset: list[float] | None = None
    robot_base_pos_override: list[float] | None = None
    robot_base_ori_override: list[float] | None = None


def eval_main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    post_success_steps = args.post_success_steps
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    render_gpu_device_id = args.render_gpu_device_id
    host = args.host
    port = args.port
    reference_ep_meta = _load_reference_ep_meta(
        args.reference_ep_meta_path,
        robot_base_pos_offset=args.robot_base_pos_offset,
        robot_base_ori_offset=args.robot_base_ori_offset,
        robot_base_pos_override=args.robot_base_pos_override,
        robot_base_ori_override=args.robot_base_ori_override,
    )

    if args.env_names:
        all_env_names = list(args.env_names)
    else:
        if not args.task_set:
            raise ValueError("Specify --args.env-names or --args.task-set.")
        all_env_names = []
        for task in args.task_set:
            registry_key = TASK_SET_ALIASES.get(task, task)
            if registry_key not in TASK_SET_REGISTRY:
                available = ", ".join(sorted(TASK_SET_REGISTRY.keys()))
                raise KeyError(
                    f"Unknown RoboCasa task set {task!r}. "
                    f"Use one of: {available}. "
                    f"OpenPI config alias 'target_atomic_seen' maps to 'atomic_seen'."
                )
            env_names = TASK_SET_REGISTRY[registry_key]
            all_env_names.extend(env_names)

    for env_name in all_env_names:
        # try:
        eval_env(
            env_name,
            split,
            log_dir,
            num_trials,
            post_success_steps,
            resize_size,
            replan_steps,
            host,
            port,
            args.seed,
            render_gpu_device_id,
            reference_ep_meta=reference_ep_meta,
            randomize_scene=args.randomize_scene,
            scene_layout_and_style_ids=args.scene_layout_and_style_ids,
            scene_layout_ids=args.scene_layout_ids,
            scene_style_ids=args.scene_style_ids,
            robot_base_pos_random_range=args.robot_base_pos_random_range,
            robot_base_ori_random_range=args.robot_base_ori_random_range,
            robot_arm_joint_random_range=args.robot_arm_joint_random_range,
            robot_arm_joint_offset=args.robot_arm_joint_offset,
            save_rollouts=args.save_rollouts,
            rollout_dir=args.rollout_dir,
        )
        # except Exception as e:
        #     print("Exception!")
        #     print(e)

def _as_uint8_image(obs: dict, key: str) -> np.ndarray:
    image = np.ascontiguousarray(obs[key])
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return image.astype(np.uint8, copy=False)


def _make_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    ).astype(np.float32)


def _get_contact(obs: dict, env=None, contact_geom_ids: dict[str, set[int]] | None = None) -> np.ndarray:
    left = obs.get("observation.contact.left", obs.get("contact.left", 0.0))
    right = obs.get("observation.contact.right", obs.get("contact.right", 0.0))
    obs_contact = np.asarray([left, right], dtype=np.float32).reshape(2)
    if np.any(obs_contact > 0.5) or env is None or contact_geom_ids is None:
        return obs_contact
    return _get_sim_hand_contact(env, contact_geom_ids)


ENV_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


def _flatten_env_action(env_action: dict) -> np.ndarray:
    if not isinstance(env_action, dict):
        return np.asarray(env_action, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [np.asarray(env_action[key], dtype=np.float32).reshape(-1) for key in ENV_ACTION_KEYS],
        axis=0,
    )


def _sim_name_position(sim, kind: str, name: str) -> np.ndarray:
    if kind == "site":
        if hasattr(sim.data, "get_site_xpos"):
            return np.asarray(sim.data.get_site_xpos(name), dtype=np.float32)
        return np.asarray(sim.data.site_xpos[sim.model.site_name2id(name)], dtype=np.float32)
    if kind == "geom":
        if hasattr(sim.data, "get_geom_xpos"):
            return np.asarray(sim.data.get_geom_xpos(name), dtype=np.float32)
        return np.asarray(sim.data.geom_xpos[sim.model.geom_name2id(name)], dtype=np.float32)
    if kind == "body":
        if hasattr(sim.data, "get_body_xpos"):
            return np.asarray(sim.data.get_body_xpos(name), dtype=np.float32)
        return np.asarray(sim.data.body_xpos[sim.model.body_name2id(name)], dtype=np.float32)
    raise ValueError(f"Unsupported MuJoCo object kind: {kind}")


def _get_raw_robocasa_env(env):
    gym_env = _get_robocasa_gym_env(env)
    return object.__getattribute__(gym_env, "env")


def _get_gripper_for_arm(raw_env, arm: str = "right"):
    gripper = raw_env.robots[0].gripper
    if isinstance(gripper, dict):
        return gripper[arm]
    return gripper


def _safe_geom_ids(raw_env, geom_names) -> set[int]:
    geom_ids = set()
    for name in geom_names or []:
        try:
            geom_ids.add(int(raw_env.sim.model.geom_name2id(name)))
        except Exception:
            continue
    return geom_ids


def _split_left_right_gripper_geom_names(geom_names) -> tuple[list[str], list[str]]:
    left_names = []
    right_names = []
    for name in geom_names or []:
        lower = str(name).lower()
        if (
            "left" in lower
            or "finger1" in lower
            or "f1" in lower
            or lower.startswith("l_")
            or "_l_" in lower
        ):
            left_names.append(name)
        if (
            "right" in lower
            or "finger2" in lower
            or "f2" in lower
            or lower.startswith("r_")
            or "_r_" in lower
        ):
            right_names.append(name)
    return left_names, right_names


def _resolve_contact_geom_ids(env, arm: str = "right") -> dict[str, set[int]]:
    raw_env = _get_raw_robocasa_env(env)
    gripper = _get_gripper_for_arm(raw_env, arm=arm)
    important_geoms = getattr(gripper, "important_geoms", {}) or {}

    # Use the whole left/right gripper side, not just fingertip or fingerpad
    # geoms. important_geoms gives good semantic groups; contact_geoms catches
    # any additional collision geoms on the gripper side.
    left_names = list(important_geoms.get("left_finger") or [])
    right_names = list(important_geoms.get("right_finger") or [])
    split_left_names, split_right_names = _split_left_right_gripper_geom_names(
        getattr(gripper, "contact_geoms", [])
    )
    left_names.extend(split_left_names)
    right_names.extend(split_right_names)
    if not left_names:
        left_names = list(important_geoms.get("left_fingerpad") or [])
    if not right_names:
        right_names = list(important_geoms.get("right_fingerpad") or [])

    left_ids = _safe_geom_ids(raw_env, left_names)
    right_ids = _safe_geom_ids(raw_env, right_names)
    if not left_ids or not right_ids:
        raise RuntimeError(
            "Could not resolve left/right hand geoms from gripper important_geoms/contact_geoms. "
            f"Resolved keys: {sorted(important_geoms.keys())}"
        )

    robot_ids = set()
    robot_model = raw_env.robots[0].robot_model
    robot_ids.update(_safe_geom_ids(raw_env, getattr(robot_model, "contact_geoms", [])))
    gripper_model = raw_env.robots[0].gripper
    if isinstance(gripper_model, dict):
        for candidate in gripper_model.values():
            robot_ids.update(_safe_geom_ids(raw_env, getattr(candidate, "contact_geoms", [])))
    else:
        robot_ids.update(_safe_geom_ids(raw_env, getattr(gripper_model, "contact_geoms", [])))

    return {
        "left": left_ids,
        "right": right_ids,
        "robot": robot_ids,
        "left_names": set(str(x) for x in left_names),
        "right_names": set(str(x) for x in right_names),
    }


def _get_sim_hand_contact(env, contact_geom_ids: dict[str, set[int]] | None) -> np.ndarray:
    if contact_geom_ids is None:
        return np.zeros(2, dtype=np.float32)
    try:
        raw_env = _get_raw_robocasa_env(env)
        left_ids = contact_geom_ids["left"]
        right_ids = contact_geom_ids["right"]
        robot_ids = contact_geom_ids["robot"]
        left_contact = 0.0
        right_contact = 0.0
        for idx in range(int(raw_env.sim.data.ncon)):
            contact = raw_env.sim.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in left_ids and geom2 not in robot_ids) or (
                geom2 in left_ids and geom1 not in robot_ids
            ):
                left_contact = 1.0
            if (geom1 in right_ids and geom2 not in robot_ids) or (
                geom2 in right_ids and geom1 not in robot_ids
            ):
                right_contact = 1.0
            if left_contact == 1.0 and right_contact == 1.0:
                break
        return np.asarray([left_contact, right_contact], dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


def _get_faucet_metrics(env) -> dict[str, float]:
    missing = {
        "gripper_to_faucet_handle_dist": np.nan,
        "faucet_handle_angle": np.nan,
    }
    try:
        gym_env = _get_robocasa_gym_env(env)
        raw_env = object.__getattribute__(gym_env, "env")
        sink = getattr(raw_env, "sink", None)
        if sink is None and hasattr(raw_env, "get_fixture"):
            sink = raw_env.get_fixture(FixtureType.SINK)
        if sink is None:
            return missing

        robot = raw_env.robots[0]
        eef_site_id = robot.eef_site_id["right"] if isinstance(robot.eef_site_id, dict) else robot.eef_site_id
        gripper_pos = np.asarray(raw_env.sim.data.site_xpos[eef_site_id], dtype=np.float32)

        handle_pos = None
        for kind, name in (
            ("geom", f"{sink.naming_prefix}handle_main"),
            ("body", f"{sink.naming_prefix}handle"),
        ):
            try:
                handle_pos = _sim_name_position(raw_env.sim, kind, name)
                break
            except Exception:
                continue
        if handle_pos is None:
            return missing

        handle_joint_name = f"{sink.naming_prefix}handle_joint"
        handle_joint_addr = raw_env.sim.model.get_joint_qpos_addr(handle_joint_name)
        handle_joint_id = raw_env.sim.model.joint_name2id(handle_joint_name)
        handle_joint_range = raw_env.sim.model.jnt_range[handle_joint_id]
        handle_angle = float(raw_env.sim.data.qpos[handle_joint_addr])
        handle_angle = float(np.clip(handle_angle, handle_joint_range[0], handle_joint_range[1]))
        return {
            "gripper_to_faucet_handle_dist": float(np.linalg.norm(gripper_pos - handle_pos)),
            "faucet_handle_angle": handle_angle,
        }
    except Exception:
        return missing


def _get_rollout_metrics(env) -> dict[str, float]:
    return _get_faucet_metrics(env)


def _append_rollout_step(
    buffer: dict,
    obs: dict,
    policy_action: np.ndarray,
    env_action: dict,
    reward,
    done,
    info,
    rollout_metrics: dict[str, float] | None = None,
    contact: np.ndarray | None = None,
):
    buffer["robot0_agentview_left"].append(_as_uint8_image(obs, "video.robot0_agentview_left"))
    buffer["robot0_agentview_right"].append(_as_uint8_image(obs, "video.robot0_agentview_right"))
    buffer["robot0_eye_in_hand"].append(_as_uint8_image(obs, "video.robot0_eye_in_hand"))
    buffer["state"].append(_make_state(obs))
    buffer["contact"].append(
        np.asarray(contact, dtype=np.float32).reshape(2) if contact is not None else _get_contact(obs)
    )
    buffer["policy_action"].append(np.asarray(policy_action, dtype=np.float32))
    buffer["env_action"].append(_flatten_env_action(env_action))
    buffer["reward"].append(float(reward))
    buffer["done"].append(bool(done))
    buffer["success"].append(bool(info.get("success", done)) if isinstance(info, dict) else bool(done))
    rollout_metrics = rollout_metrics or {}
    buffer["gripper_to_faucet_handle_dist"].append(
        float(rollout_metrics.get("gripper_to_faucet_handle_dist", np.nan))
    )
    buffer["faucet_handle_angle"].append(float(rollout_metrics.get("faucet_handle_angle", np.nan)))


def _save_rollout(buffer: dict, output_dir: pathlib.Path, episode_idx: int, suffix: str, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"rollout_{episode_idx:06d}_{suffix}.npz"
    arrays = {}
    for key, values in buffer.items():
        if values:
            arrays[key] = np.stack(values)
        else:
            arrays[key] = np.empty((0,), dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)
    meta_path = output_dir / f"rollout_{episode_idx:06d}_{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return npz_path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_reference_ep_meta(
    path: str | None,
    *,
    robot_base_pos_offset: list[float] | None = None,
    robot_base_ori_offset: list[float] | None = None,
    robot_base_pos_override: list[float] | None = None,
    robot_base_ori_override: list[float] | None = None,
) -> dict | None:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as f:
        ep_meta = json.load(f)

    def update_pose(key: str, override, offset):
        base = override if override is not None else ep_meta.get(key)
        if base is None:
            return
        base = np.asarray(base, dtype=np.float64)
        delta = np.zeros_like(base)
        if offset is not None:
            delta[: len(offset)] = np.asarray(offset, dtype=np.float64)
        ep_meta[key] = (base + delta).tolist()

    update_pose("init_robot_base_pos", robot_base_pos_override, robot_base_pos_offset)
    update_pose("init_robot_base_ori", robot_base_ori_override, robot_base_ori_offset)
    return ep_meta


def _set_episode_meta(env, ep_meta: dict | None) -> None:
    if not ep_meta:
        return
    candidates = [env, getattr(env, "unwrapped", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "env", None))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "set_ep_meta"):
            candidate.set_ep_meta(_json_safe(ep_meta))
            return
    raise AttributeError("Could not find a RoboCasa env with set_ep_meta().")


def _unset_episode_meta(env) -> None:
    candidates = [env, getattr(env, "unwrapped", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "env", None))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "unset_ep_meta"):
            candidate.unset_ep_meta()
            return


def _sample_pose_offset(rng: np.random.Generator, random_range: list[float] | None, dim: int) -> np.ndarray:
    if random_range is None:
        return np.zeros(dim, dtype=np.float64)
    high = np.zeros(dim, dtype=np.float64)
    values = np.asarray(random_range, dtype=np.float64)
    high[: min(dim, len(values))] = values[:dim]
    return rng.uniform(-high, high)


def _sample_joint_offset(rng: np.random.Generator, random_range, dim: int) -> np.ndarray:
    if random_range is None:
        return np.zeros(dim, dtype=np.float64)
    if np.isscalar(random_range):
        high = np.full(dim, float(random_range), dtype=np.float64)
    else:
        high = np.zeros(dim, dtype=np.float64)
        values = np.asarray(random_range, dtype=np.float64)
        high[: min(dim, len(values))] = values[:dim]
    return rng.uniform(-high, high)


def _get_robocasa_gym_env(env):
    candidates = [env, getattr(env, "unwrapped", None)]
    seen = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        try:
            inner = object.__getattribute__(candidate, "env")
        except AttributeError:
            inner = None

        try:
            get_observation = object.__getattribute__(candidate, "get_observation")
        except AttributeError:
            get_observation = None

        try:
            robots = object.__getattribute__(inner, "robots") if inner is not None else None
        except AttributeError:
            robots = None

        if callable(get_observation) and robots is not None:
            return candidate
        candidates.append(inner)
    raise AttributeError("Could not find the RoboCasa GymWrapper inside the Gymnasium wrapper stack.")


def _refresh_observation(env):
    gym_env = _get_robocasa_gym_env(env)
    raw_env = object.__getattribute__(gym_env, "env")
    raw_obs = (
        raw_env.viewer._get_observations(force_update=True)
        if raw_env.viewer_get_obs
        else raw_env._get_observations(force_update=True)
    )
    return gym_env.get_observation(raw_obs)


def _update_fixture_state(env) -> None:
    gym_env = _get_robocasa_gym_env(env)
    raw_env = object.__getattribute__(gym_env, "env")
    if hasattr(raw_env, "update_state"):
        raw_env.update_state()


def _randomize_robot_arm_joints(
    env,
    rng: np.random.Generator,
    *,
    robot_arm_joint_random_range=None,
    robot_arm_joint_offset: list[float] | None = None,
) -> dict:
    if robot_arm_joint_random_range is None and robot_arm_joint_offset is None:
        return {}

    gym_env = _get_robocasa_gym_env(env)
    raw_env = object.__getattribute__(gym_env, "env")
    robot_metas = []

    for robot_idx, robot in enumerate(raw_env.robots):
        qpos_indexes = getattr(robot, "_ref_arm_joint_pos_indexes", None)
        joint_indexes = getattr(robot, "_ref_arm_joint_indexes", None)
        joint_names = list(getattr(robot, "robot_arm_joints", []))
        if not qpos_indexes:
            continue

        before = np.asarray(raw_env.sim.data.qpos[qpos_indexes], dtype=np.float64)
        random_offset = _sample_joint_offset(rng, robot_arm_joint_random_range, len(before))
        fixed_offset = np.zeros(len(before), dtype=np.float64)
        if robot_arm_joint_offset is not None:
            values = np.asarray(robot_arm_joint_offset, dtype=np.float64)
            fixed_offset[: min(len(before), len(values))] = values[: len(before)]

        after = before + fixed_offset + random_offset
        if joint_indexes is not None:
            ranges = np.asarray(raw_env.sim.model.jnt_range[joint_indexes], dtype=np.float64)
            limited = np.asarray(raw_env.sim.model.jnt_limited[joint_indexes], dtype=bool)
            after[limited] = np.clip(after[limited], ranges[limited, 0], ranges[limited, 1])

        raw_env.sim.data.qpos[qpos_indexes] = after
        raw_env.sim.data.qvel[getattr(robot, "_ref_arm_joint_vel_indexes", [])] = 0.0
        robot_metas.append(
            {
                "robot_idx": robot_idx,
                "joint_names": joint_names,
                "joint_qpos_before": before.tolist(),
                "joint_qpos_after": after.tolist(),
                "fixed_robot_arm_joint_offset": fixed_offset.tolist(),
                "random_robot_arm_joint_offset": random_offset.tolist(),
            }
        )

    raw_env.sim.forward()
    for robot in raw_env.robots:
        if hasattr(robot, "composite_controller"):
            robot.composite_controller.reset()
    return {"robot_arm_joint_randomization": robot_metas}


def _episode_meta_with_random_robot_pose(
    reference_ep_meta: dict | None,
    rng: np.random.Generator,
    *,
    robot_base_pos_random_range: list[float] | None,
    robot_base_ori_random_range: list[float] | None,
) -> tuple[dict | None, dict]:
    if not reference_ep_meta:
        return None, {}

    ep_meta = json.loads(json.dumps(reference_ep_meta))
    random_pos_offset = _sample_pose_offset(rng, robot_base_pos_random_range, 3)
    random_ori_offset = _sample_pose_offset(rng, robot_base_ori_random_range, 3)

    if "init_robot_base_pos" in ep_meta:
        base_pos = np.asarray(ep_meta["init_robot_base_pos"], dtype=np.float64)
        ep_meta["init_robot_base_pos"] = (base_pos + random_pos_offset).tolist()
    if "init_robot_base_ori" in ep_meta:
        base_ori = np.asarray(ep_meta["init_robot_base_ori"], dtype=np.float64)
        ep_meta["init_robot_base_ori"] = (base_ori + random_ori_offset).tolist()

    return ep_meta, {
        "random_robot_base_pos_offset": random_pos_offset.tolist(),
        "random_robot_base_ori_offset": random_ori_offset.tolist(),
    }


def _get_episode_meta(env) -> dict:
    candidates = [env, getattr(env, "unwrapped", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "env", None))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "get_ep_meta"):
            try:
                return _json_safe(candidate.get_ep_meta())
            except Exception as exc:
                logging.warning(f"Failed to read RoboCasa episode metadata: {exc}")
                return {}
    return {}


def _layout_style_key(ep_meta: dict) -> str:
    layout_id = _json_safe(ep_meta.get("layout_id"))
    style_id = _json_safe(ep_meta.get("style_id"))
    return f"{json.dumps(layout_id, sort_keys=True)},{json.dumps(style_id, sort_keys=True)}"


def _make_robocasa_env(
    env_name,
    split,
    seed,
    render_gpu_device_id,
    *,
    reference_ep_meta=None,
    randomize_scene=False,
    scene_layout_and_style_ids=None,
    scene_layout_ids=None,
    scene_style_ids=None,
):
    env_kwargs = {}
    gym_split = split

    if randomize_scene:
        has_custom_scene_candidates = (
            scene_layout_and_style_ids is not None
            or scene_layout_ids is not None
            or scene_style_ids is not None
        )
        if scene_layout_and_style_ids is not None:
            if scene_layout_ids is not None or scene_style_ids is not None:
                raise ValueError(
                    "Use either scene_layout_and_style_ids or scene_layout_ids/scene_style_ids, not both."
                )
            env_kwargs["layout_and_style_ids"] = scene_layout_and_style_ids
        else:
            if scene_layout_ids is not None:
                env_kwargs["layout_ids"] = scene_layout_ids
            if scene_style_ids is not None:
                env_kwargs["style_ids"] = scene_style_ids

        if has_custom_scene_candidates:
            # RoboCasa create_env() overwrites layout/style kwargs when split is
            # "target"/"pretrain"/"all", so use split=None and pass the object
            # instance split explicitly.
            gym_split = None
            env_kwargs["obj_instance_split"] = None if split == "all" else split
    elif (
        reference_ep_meta
        and "layout_id" in reference_ep_meta
        and "style_id" in reference_ep_meta
    ):
        env_kwargs["layout_and_style_ids"] = [
            [reference_ep_meta["layout_id"], reference_ep_meta["style_id"]]
        ]

    return gym.make(
        f"robocasa/{env_name}",
        split=gym_split,
        seed=seed,
        render_gpu_device_id=render_gpu_device_id,
        **env_kwargs,
    )


def _close_robocasa_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        try:
            env.env.close()
        except Exception:
            pass


def eval_env(
    env_name,
    split,
    log_dir,
    num_trials,
    post_success_steps,
    resize_size,
    replan_steps,
    host,
    port,
    seed,
    render_gpu_device_id,
    *,
    reference_ep_meta=None,
    randomize_scene=False,
    scene_layout_and_style_ids=None,
    scene_layout_ids=None,
    scene_style_ids=None,
    robot_base_pos_random_range=None,
    robot_base_ori_random_range=None,
    robot_arm_joint_random_range=None,
    robot_arm_joint_offset=None,
    save_rollouts=False,
    rollout_dir=None,
):
    # set args based on task
    assert split in ["pretrain", "target"]
    task_horizon = get_task_horizon(env_name)
    # set dataset path and horizon
    horizon = int(task_horizon * 1) # the policy moves slow so give the policy extra time

    now = datetime.now()
    now_formatted = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"{log_dir}/evals/{split}/{env_name}/{now_formatted}"

    stats_path = pathlib.Path(log_path) / "stats.json"
    if stats_path.exists():
        print(f"{env_name}/{split}, stats path exists at {stats_path}, skipping.")
        return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    rollout_path = pathlib.Path(rollout_dir) if rollout_dir else pathlib.Path(log_path) / "rollouts"


    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    layout_style_counts = collections.Counter()
    # Get task
    active_reference_ep_meta = None if randomize_scene else reference_ep_meta
    env = _make_robocasa_env(
        env_name,
        split,
        seed,
        render_gpu_device_id,
        reference_ep_meta=active_reference_ep_meta,
        randomize_scene=randomize_scene,
        scene_layout_and_style_ids=scene_layout_and_style_ids,
        scene_layout_ids=scene_layout_ids,
        scene_style_ids=scene_style_ids,
    )
    # Start episodes
    task_episodes, task_successes = 0, 0
    pose_rng = np.random.default_rng(seed)
    for episode_idx in tqdm.tqdm(range(num_trials)):

        # Reset environment
        episode_ep_meta, pose_random_meta = _episode_meta_with_random_robot_pose(
            active_reference_ep_meta,
            pose_rng,
            robot_base_pos_random_range=robot_base_pos_random_range,
            robot_base_ori_random_range=robot_base_ori_random_range,
        )
        if episode_ep_meta:
            _set_episode_meta(env, episode_ep_meta)
        else:
            _unset_episode_meta(env)
        obs, info = env.reset()
        arm_random_meta = _randomize_robot_arm_joints(
            env,
            pose_rng,
            robot_arm_joint_random_range=robot_arm_joint_random_range,
            robot_arm_joint_offset=robot_arm_joint_offset,
        )
        if arm_random_meta:
            obs = _refresh_observation(env)
        ep_meta = _get_episode_meta(env)
        layout_style_counts[_layout_style_key(ep_meta)] += 1
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        try:
            contact_geom_ids = _resolve_contact_geom_ids(env, arm="right")
        except Exception as exc:
            logging.warning(f"Could not resolve gripper contact geoms; contact will be zeros: {exc}")
            contact_geom_ids = None

        # Setup
        t = 0
        replay_images = []
        rollout_buffer = {
            "robot0_agentview_left": [],
            "robot0_agentview_right": [],
            "robot0_eye_in_hand": [],
            "state": [],
            "contact": [],
            "policy_action": [],
            "env_action": [],
            "reward": [],
            "done": [],
            "success": [],
            "gripper_to_faucet_handle_dist": [],
            "faucet_handle_angle": [],
        }

        logging.info(f"Starting episode {task_episodes+1}...")
        episode_success = False
        first_success_step = None
        while t < horizon:
            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )

            # Save preprocessed image for replay video
            # replay_images.append(img)

            if not action_plan:
                state = _make_state(obs)
                # state = np.ascontiguousarray(state)
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: replan_steps])

            policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
            env_action = convert_action(policy_action)
            obs_before_step = obs
            rollout_metrics_before_step = _get_rollout_metrics(env)
            contact_before_step = _get_contact(obs_before_step, env=env, contact_geom_ids=contact_geom_ids)
            # Execute action in environment
            obs, reward, _terminated, _truncated, info = env.step(env_action)
            success_now = bool(info["success"]) # for robocasa, success entry in info
            if success_now and not episode_success:
                episode_success = True
                first_success_step = t
                task_successes += 1
                total_successes += 1
            if episode_success:
                _update_fixture_state(env)
                obs = _refresh_observation(env)
            rollout_metrics_after_step = _get_rollout_metrics(env)
            contact_after_step = _get_contact(obs, env=env, contact_geom_ids=contact_geom_ids)
            obs_to_save = obs if episode_success else obs_before_step
            rollout_metrics_to_save = (
                rollout_metrics_after_step if episode_success else rollout_metrics_before_step
            )
            contact_to_save = contact_after_step if episode_success else contact_before_step
            if save_rollouts:
                _append_rollout_step(
                    rollout_buffer,
                    obs_to_save,
                    policy_action,
                    env_action,
                    reward,
                    False,
                    info,
                    rollout_metrics_to_save,
                    contact_to_save,
                )
            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            replay_img = image_tools.convert_to_uint8(
                replay_img
            )
            if t % 2 == 0 or t == horizon - 1 or success_now:
                replay_images.append(replay_img)
            t += 1
            if first_success_step is not None and t > first_success_step + post_success_steps:
                break

        if save_rollouts and rollout_buffer["done"]:
            rollout_buffer["done"][-1] = True

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if episode_success else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )
        if save_rollouts:
            saved_path = _save_rollout(
                rollout_buffer,
                rollout_path / env_name,
                episode_idx,
                suffix,
                {
                    "env_name": env_name,
                    "split": split,
                    "episode_idx": episode_idx,
                    "success": bool(episode_success),
                    "first_success_step": first_success_step,
                    "task_lang": str(task_lang),
                    "seed": seed,
                    "horizon": horizon,
                    "post_success_steps": post_success_steps,
                    "replan_steps": replan_steps,
                    "num_steps": len(rollout_buffer["state"]),
                    "layout_id": ep_meta.get("layout_id"),
                    "style_id": ep_meta.get("style_id"),
                    "randomize_scene": bool(randomize_scene),
                    "scene_layout_and_style_ids": scene_layout_and_style_ids,
                    "scene_layout_ids": scene_layout_ids,
                    "scene_style_ids": scene_style_ids,
                    "ep_meta": ep_meta,
                    "pose_randomization": {**pose_random_meta, **arm_random_meta},
                    "env_action_keys": list(ENV_ACTION_KEYS),
                    "rollout_metric_keys": [
                        "gripper_to_faucet_handle_dist",
                        "faucet_handle_angle",
                    ],
                    "contact_source": "sim.data.contact whole left/right gripper-side geoms vs non-robot geoms",
                    "contact_geom_ids": (
                        {
                            "left": sorted(int(x) for x in contact_geom_ids["left"]),
                            "right": sorted(int(x) for x in contact_geom_ids["right"]),
                            "robot": sorted(int(x) for x in contact_geom_ids["robot"]),
                            "left_names": sorted(str(x) for x in contact_geom_ids["left_names"]),
                            "right_names": sorted(str(x) for x in contact_geom_ids["right_names"]),
                        }
                        if contact_geom_ids is not None
                        else None
                    ),
                    "camera_mapping_for_viva": {
                        "cam_high": "robot0_agentview_left",
                        "cam_left_wrist": "robot0_eye_in_hand",
                        "cam_right_wrist": "robot0_agentview_right",
                    },
                },
            )
            logging.info(f"Saved rollout data: {saved_path}")

        # Log current results
        logging.info(f"Success: {episode_success}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        stats = {
            "num_episodes": total_episodes,
            "success_rate": float(total_successes) / float(total_episodes),
            "randomize_scene": bool(randomize_scene),
            "scene_layout_and_style_ids": scene_layout_and_style_ids,
            "scene_layout_ids": scene_layout_ids,
            "scene_style_ids": scene_style_ids,
            "layout_style_counts": {
                layout_style_key: count
                for layout_style_key, count in sorted(layout_style_counts.items())
            },
        }
        json.dump(stats, f, indent=4)

    # close and delete the env
    _close_robocasa_env(env)
    if hasattr(env, "env"):
        del env.env
    del env




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.argv = _normalize_bool_cli_args(sys.argv)
    if USE_CODE_CONFIG:
        eval_main(Args(**CODE_CONFIG))
    else:
        tyro.cli(eval_main)
