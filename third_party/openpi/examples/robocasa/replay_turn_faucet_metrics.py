#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Replay RoboCasa LeRobot sim states and collect TurnOnSinkFaucet metrics.

It does not modify parquet data. For each episode it writes:

    extras/episode_000000/turn_metrics.npz

The output contains faucet handle angle and gripper-to-handle distance, which
can then be consumed by value-label generation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOCASA_MJCF_TMPDIR", "/tmp/robocasa_mjcf_tmp")

try:
    import robosuite  # noqa: E402
    import robocasa  # noqa: F401,E402
    from robocasa.models.fixtures import FixtureType  # noqa: E402
except ModuleNotFoundError:
    robosuite = None
    FixtureType = None


METRIC_KEYS = [
    "gripper_to_faucet_handle_dist",
    "faucet_handle_angle",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_xml(path: Path) -> str:
    with gzip.open(path, "rb") as f:
        return f.read().decode("utf-8")


def parse_episodes(spec: str | None, available: list[int]) -> list[int]:
    if spec is None or spec == "all":
        return available
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo_s, hi_s = part.split(":", 1)
            lo = int(lo_s) if lo_s else min(available)
            hi = int(hi_s) if hi_s else max(available) + 1
            out.extend(i for i in available if lo <= i < hi)
        else:
            out.append(int(part))
    seen = set()
    return [i for i in out if i in available and not (i in seen or seen.add(i))]


def sim_name_position(sim, kind: str, name: str) -> np.ndarray:
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
    raise ValueError(f"unsupported MuJoCo object kind: {kind}")


def get_eef_pos(env) -> np.ndarray:
    robot = env.robots[0]
    eef_site_id = robot.eef_site_id["right"] if isinstance(robot.eef_site_id, dict) else robot.eef_site_id
    return np.asarray(env.sim.data.site_xpos[eef_site_id], dtype=np.float32)


def collect_faucet_metrics(env) -> dict[str, float]:
    missing = {
        "gripper_to_faucet_handle_dist": np.nan,
        "faucet_handle_angle": np.nan,
    }

    sink = getattr(env, "sink", None)
    if sink is None and FixtureType is not None and hasattr(env, "get_fixture"):
        sink = env.get_fixture(FixtureType.SINK)
    if sink is None:
        return missing

    eef_pos = get_eef_pos(env)
    handle_pos = None
    for kind, name in (
        ("geom", f"{sink.naming_prefix}handle_main"),
        ("body", f"{sink.naming_prefix}handle"),
        ("site", f"{sink.naming_prefix}handle_site"),
    ):
        try:
            handle_pos = sim_name_position(env.sim, kind, name)
            break
        except Exception:
            continue
    if handle_pos is None:
        return missing

    try:
        handle_joint_name = f"{sink.naming_prefix}handle_joint"
        handle_joint_addr = env.sim.model.get_joint_qpos_addr(handle_joint_name)
        handle_joint_id = env.sim.model.joint_name2id(handle_joint_name)
        handle_joint_range = env.sim.model.jnt_range[handle_joint_id]
        handle_angle = float(env.sim.data.qpos[handle_joint_addr])
        handle_angle = float(np.clip(handle_angle, handle_joint_range[0], handle_joint_range[1]))
    except Exception:
        handle_angle = np.nan

    return {
        "gripper_to_faucet_handle_dist": float(np.linalg.norm(eef_pos - handle_pos)),
        "faucet_handle_angle": float(handle_angle),
    }


def reset_to(
    env,
    *,
    model_xml: str | None = None,
    ep_meta: dict[str, Any] | None = None,
    state: np.ndarray | None = None,
) -> None:
    if model_xml is not None:
        if ep_meta is not None:
            if hasattr(env, "set_attrs_from_ep_meta"):
                env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(model_xml) if hasattr(env, "edit_model_xml") else model_xml
        env.reset_from_xml_string(xml)
        env.sim.reset()

    if state is not None:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def make_env(dataset_dir: Path, render_gpu_device_id: int):
    if robosuite is None:
        raise ModuleNotFoundError(
            "robosuite is required for MuJoCo replay. Use --handle-states-only "
            "for existing turn branch data with handle_states.json."
        )
    meta = load_json(dataset_dir / "extras" / "dataset_meta.json")
    env_args = meta["env_args"]
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs["env_name"] = env_args["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["render_gpu_device_id"] = render_gpu_device_id
    return robosuite.make(**env_kwargs)


def replay_episode(env, ep_dir: Path, output_name: str, overwrite: bool) -> dict[str, Any]:
    out_path = ep_dir / output_name
    if out_path.exists() and not overwrite:
        return {"episode": ep_dir.name, "status": "skip_exists", "out": str(out_path)}

    states = np.load(ep_dir / "states.npz", allow_pickle=True)["states"]
    ep_meta = load_json(ep_dir / "ep_meta.json")
    model_xml = load_model_xml(ep_dir / "model.xml.gz")

    arrays = {key: np.full(len(states), np.nan, dtype=np.float32) for key in METRIC_KEYS}
    reset_to(env, model_xml=model_xml, ep_meta=ep_meta, state=states[0])

    for t, state in enumerate(states):
        reset_to(env, state=state)
        metrics = collect_faucet_metrics(env)
        for key in METRIC_KEYS:
            arrays[key][t] = metrics[key]

    np.savez_compressed(out_path, **arrays)
    return {
        "episode": ep_dir.name,
        "status": "ok",
        "T": int(len(states)),
        "distance_final": float(arrays["gripper_to_faucet_handle_dist"][-1]),
        "angle_final": float(arrays["faucet_handle_angle"][-1]),
        "finite_distance_ratio": float(np.isfinite(arrays["gripper_to_faucet_handle_dist"]).mean()),
        "finite_angle_ratio": float(np.isfinite(arrays["faucet_handle_angle"]).mean()),
        "out": str(out_path),
    }


def replay_episode_from_handle_states(
    ep_dir: Path,
    output_name: str,
    overwrite: bool,
    fallback_distance: float,
) -> dict[str, Any]:
    out_path = ep_dir / output_name
    if out_path.exists() and not overwrite:
        return {"episode": ep_dir.name, "status": "skip_exists", "out": str(out_path)}

    handle_path = ep_dir / "handle_states.json"
    if not handle_path.exists():
        raise FileNotFoundError(f"missing {handle_path}")
    handle_states = load_json(handle_path)
    if not isinstance(handle_states, list) or not handle_states:
        raise ValueError(f"{handle_path} is empty or not a list")

    angle = np.asarray(
        [float(row.get("handle_joint", np.nan)) if isinstance(row, dict) else np.nan for row in handle_states],
        dtype=np.float32,
    )
    distance = np.full(len(angle), float(fallback_distance), dtype=np.float32)
    np.savez_compressed(
        out_path,
        gripper_to_faucet_handle_dist=distance,
        faucet_handle_angle=angle,
        source=np.asarray("handle_states_json_with_constant_distance"),
    )
    return {
        "episode": ep_dir.name,
        "status": "ok_handle_states",
        "T": int(len(angle)),
        "distance_final": float(distance[-1]),
        "angle_final": float(angle[-1]),
        "finite_distance_ratio": float(np.isfinite(distance).mean()),
        "finite_angle_ratio": float(np.isfinite(angle).mean()),
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=str, default="all", help="all, comma list, or half-open ranges like 0:10,20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="turn_metrics.npz")
    parser.add_argument("--summary-name", type=str, default="turn_metrics_summary.csv")
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--handle-states-only",
        action="store_true",
        help="Read handle_joint from handle_states.json instead of replaying MuJoCo states.",
    )
    parser.add_argument(
        "--fallback-distance",
        type=float,
        default=0.03,
        help="Distance value used with --handle-states-only; keeps labels angle-driven.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ep_dirs = sorted((args.dataset_dir / "extras").glob("episode_*"))
    available = [int(p.name.split("_")[-1]) for p in ep_dirs]
    selected_ids = parse_episodes(args.episodes, available)
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    selected = [args.dataset_dir / "extras" / f"episode_{i:06d}" for i in selected_ids]
    if not selected:
        raise RuntimeError("No episodes selected")

    rows: list[dict[str, Any]] = []
    env = None
    try:
        if not args.handle_states_only:
            env = make_env(args.dataset_dir, args.render_gpu_device_id)
        for idx, ep_dir in enumerate(selected, start=1):
            try:
                if args.handle_states_only:
                    row = replay_episode_from_handle_states(
                        ep_dir,
                        args.output_name,
                        args.overwrite,
                        args.fallback_distance,
                    )
                else:
                    row = replay_episode(env, ep_dir, args.output_name, args.overwrite)
                rows.append(row)
                print(
                    f"[{idx}/{len(selected)}] {ep_dir.name} {row['status']} "
                    f"dist={row.get('distance_final', float('nan')):.4f} "
                    f"angle={row.get('angle_final', float('nan')):.4f}"
                )
            except Exception as exc:
                row = {"episode": ep_dir.name, "status": "error", "error": repr(exc)}
                rows.append(row)
                print(f"[{idx}/{len(selected)}] {ep_dir.name} error {exc!r}")
    finally:
        if env is not None:
            env.close()

    summary_path = args.dataset_dir / "extras" / args.summary_name
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
