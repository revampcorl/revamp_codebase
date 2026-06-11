"""Small dataset-registry helpers for the TurnOnSinkFaucet release."""

from __future__ import annotations

import os
from pathlib import Path

import robocasa
import robocasa.macros as macros


def _registry():
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS, TASK_SET_REGISTRY

    return ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS, TASK_SET_REGISTRY


def get_ds_meta(task: str, split: str, source: str = "human", demo_fraction: float = 1.0) -> dict | None:
    atomic, composite, _ = _registry()
    if split not in {"target"}:
        return None
    if source != "human":
        return None
    if task in atomic:
        ds_config = atomic[task]
    elif task in composite:
        ds_config = composite[task]
    else:
        raise ValueError(f"Unknown RoboCasa task in this release: {task!r}")

    folder = ds_config.get(split, {}).get("human_path")
    if folder is None:
        return None

    if macros.DATASET_BASE_PATH is None:
        ds_base_path = os.path.join(Path(robocasa.__path__[0]).parent.absolute(), "datasets")
    else:
        ds_base_path = macros.DATASET_BASE_PATH

    num_sampled_demos = int(500 * float(demo_fraction))
    return {
        "path": os.path.join(ds_base_path, folder, "lerobot"),
        "horizon": ds_config["horizon"],
        "filter_key": f"{num_sampled_demos}_demos",
        "task": task,
        "split": split,
        "source": source,
    }


def get_ds_path(task: str, source: str, split: str = "target", return_info: bool = False):
    meta = get_ds_meta(task=task, split=split, source=source)
    path = None if meta is None else meta.get("path")
    if return_info:
        return path, (meta or {})
    return path


def get_ds_soup(split: str, task_set: str, source: str, demo_fraction: float = 1.0) -> list[dict]:
    _, _, task_sets = _registry()
    if task_set not in task_sets:
        raise KeyError(f"Unknown RoboCasa task set in this release: {task_set!r}")
    soup = []
    for task in task_sets[task_set]:
        meta = get_ds_meta(task=task, split=split, source=source, demo_fraction=demo_fraction)
        if meta is not None:
            soup.append(meta)
    return soup


def add_cotraining_weights(soup: list[dict], *_, **__) -> list[dict]:
    for item in soup:
        item["ds_weight"] = 1.0
    return soup


def get_task_horizon(task: str) -> int:
    atomic, composite, _ = _registry()
    if task in atomic:
        return int(atomic[task]["horizon"])
    if task in composite:
        return int(composite[task]["horizon"])
    raise ValueError(f"Unknown RoboCasa task in this release: {task!r}")
