#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate per-frame left / right finger contact features for RoboCasa LeRobot datasets.

For each episode under:
  <dataset_root>/extras/episode_xxxxxx/

this script replays the saved simulator states from:
  - model.xml.gz
  - ep_meta.json
  - states.npz

and writes:
  - contact_features.npy   shape [T, 2], values in {0, 1}
  - contact_features.json  metadata / feature names

The two features are:
  [0] left_finger_contact
  [1] right_finger_contact

The contact is computed using the full left / right finger geom groups
(falling back to fingerpad-only groups if needed), against non-robot geoms
only, so robot self-collision does not trigger a positive label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import robosuite
from robosuite import make

import robocasa.utils.lerobot_utils as LU


FEATURE_NAMES = [
    "left_finger_contact",
    "right_finger_contact",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_episode_list(episode_str: str | None) -> list[int] | None:
    if not episode_str:
        return None
    values: list[int] = []
    for chunk in episode_str.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range: {chunk}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(chunk))
    return sorted(set(values))


def iter_episode_indices(dataset_root: Path, episodes: list[int] | None = None) -> list[int]:
    if episodes is not None:
        return episodes

    extras_dir = dataset_root / "extras"
    episode_dirs = sorted(extras_dir.glob("episode_*"))
    indices: list[int] = []
    for ep_dir in episode_dirs:
        try:
            indices.append(int(ep_dir.name.removeprefix("episode_")))
        except ValueError:
            continue
    return indices


def restore_env_state(env, model_xml: str, ep_meta: dict[str, Any], sim_state: np.ndarray) -> None:
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


def create_env(dataset_root: Path):
    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    return make(**env_kwargs)


def _get_gripper_for_arm(env, arm: str):
    gripper = env.robots[0].gripper
    if isinstance(gripper, dict):
        if arm not in gripper:
            raise KeyError(f"Requested arm '{arm}' not found in gripper dict: {list(gripper.keys())}")
        return gripper[arm]
    return gripper


def _safe_geom_ids(env, geom_names: Iterable[str]) -> set[int]:
    geom_ids: set[int] = set()
    for name in geom_names:
        try:
            geom_ids.add(int(env.sim.model.geom_name2id(name)))
        except Exception:
            continue
    return geom_ids


def get_finger_geom_ids(env, arm: str = "right") -> tuple[set[int], set[int]]:
    gripper = _get_gripper_for_arm(env, arm)
    important_geoms = getattr(gripper, "important_geoms", {}) or {}

    # Prefer the full finger geom groups so contact is less brittle than
    # pad-only detection, while still preserving left / right finger semantics.
    left_names = important_geoms.get("left_finger") or important_geoms.get("left_fingerpad") or []
    right_names = important_geoms.get("right_finger") or important_geoms.get("right_fingerpad") or []

    left_ids = _safe_geom_ids(env, left_names)
    right_ids = _safe_geom_ids(env, right_names)

    if not left_ids or not right_ids:
        raise RuntimeError(
            "Could not resolve left/right finger geoms from gripper important_geoms. "
            f"Resolved keys: {sorted(important_geoms.keys())}"
        )

    return left_ids, right_ids


def get_robot_geom_ids(env) -> set[int]:
    geom_ids: set[int] = set()
    robot_model = env.robots[0].robot_model
    geom_ids.update(_safe_geom_ids(env, getattr(robot_model, "contact_geoms", [])))

    gripper = env.robots[0].gripper
    if isinstance(gripper, dict):
        for g in gripper.values():
            geom_ids.update(_safe_geom_ids(env, getattr(g, "contact_geoms", [])))
    else:
        geom_ids.update(_safe_geom_ids(env, getattr(gripper, "contact_geoms", [])))

    return geom_ids


def extract_contact_features_for_current_state(
    env,
    left_finger_geom_ids: set[int],
    right_finger_geom_ids: set[int],
    robot_geom_ids: set[int],
) -> np.ndarray:
    left_contact = 0.0
    right_contact = 0.0

    for idx in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 in left_finger_geom_ids and geom2 not in robot_geom_ids:
            left_contact = 1.0
        elif geom2 in left_finger_geom_ids and geom1 not in robot_geom_ids:
            left_contact = 1.0

        if geom1 in right_finger_geom_ids and geom2 not in robot_geom_ids:
            right_contact = 1.0
        elif geom2 in right_finger_geom_ids and geom1 not in robot_geom_ids:
            right_contact = 1.0

        if left_contact == 1.0 and right_contact == 1.0:
            break

    return np.array([left_contact, right_contact], dtype=np.float32)


def write_contact_features(
    episode_dir: Path,
    features: np.ndarray,
    overwrite: bool,
    write_json_flag: bool,
) -> None:
    out_npy = episode_dir / "contact_features.npy"
    out_json = episode_dir / "contact_features.json"

    if out_npy.exists() and not overwrite:
        raise FileExistsError(
            f"{out_npy} already exists. Pass --overwrite to replace it."
        )

    np.save(out_npy, features.astype(np.float32))

    if write_json_flag:
        info = {
            "feature_names": FEATURE_NAMES,
            "shape": list(features.shape),
            "dtype": "float32",
            "value_semantics": {
                "0": "no contact",
                "1": "contact",
            },
            "description": (
                "Per-frame left/right full-finger contact against non-robot geoms. "
                "Computed by replaying saved MuJoCo simulator states."
            ),
            "version": "v1",
        }
        save_json(out_json, info)


def process_episode(
    env,
    dataset_root: Path,
    episode_index: int,
    arm: str,
    overwrite: bool,
    write_json_flag: bool,
) -> tuple[bool, str]:
    episode_dir = dataset_root / "extras" / f"episode_{episode_index:06d}"
    if not episode_dir.exists():
        return False, f"missing episode dir: {episode_dir}"

    states = LU.get_episode_states(dataset_root, episode_index)
    model_xml = LU.get_episode_model_xml(dataset_root, episode_index)
    ep_meta = LU.get_episode_meta(dataset_root, episode_index)

    if len(states) == 0:
        return False, f"episode {episode_index:06d} has no states"

    restore_env_state(env, model_xml, ep_meta, states[0])
    left_ids, right_ids = get_finger_geom_ids(env, arm=arm)
    robot_geom_ids = get_robot_geom_ids(env)

    features = []
    for sim_state in states:
        env.sim.set_state_from_flattened(sim_state)
        env.sim.forward()
        if hasattr(env, "update_sites"):
            env.update_sites()
        if hasattr(env, "update_state"):
            env.update_state()
        features.append(
            extract_contact_features_for_current_state(
                env,
                left_finger_geom_ids=left_ids,
                right_finger_geom_ids=right_ids,
                robot_geom_ids=robot_geom_ids,
            )
        )

    features_np = np.stack(features, axis=0).astype(np.float32)
    write_contact_features(
        episode_dir=episode_dir,
        features=features_np,
        overwrite=overwrite,
        write_json_flag=write_json_flag,
    )
    return True, (
        f"episode {episode_index:06d}: wrote {features_np.shape[0]} frames "
        f"to {episode_dir / 'contact_features.npy'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate left/right finger contact features for RoboCasa LeRobot episodes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset root, e.g. .../TurnOnSinkFaucet/.../lerobot",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episode list like '0,1,2' or ranges like '0-9,20-29'. Default: all episodes.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm/gripper to inspect. Default: right",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing contact_features.npy/json",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write contact_features.json",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    episodes = iter_episode_indices(dataset_root, parse_episode_list(args.episodes))
    if not episodes:
        raise RuntimeError(f"No episodes found under {dataset_root / 'extras'}")

    env = create_env(dataset_root)
    ok_count = 0
    fail_count = 0
    try:
        for episode_index in episodes:
            ok, message = process_episode(
                env=env,
                dataset_root=dataset_root,
                episode_index=episode_index,
                arm=args.arm,
                overwrite=args.overwrite,
                write_json_flag=not args.no_json,
            )
            print(message)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
    finally:
        if hasattr(env, "close"):
            env.close()

    print(
        f"Done. success={ok_count} failed={fail_count} "
        f"dataset_root={dataset_root}"
    )


if __name__ == "__main__":
    main()
