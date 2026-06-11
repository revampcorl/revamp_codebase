"""OpenPI config subset for REVAMP TurnOnSinkFaucet experiments.

This release keeps the real pi0 / RoboCasa interfaces, but intentionally
removes upstream configs that are not part of this task. The only public
training config is:

    pi0_robocasa_turn_success10_weak
"""

from __future__ import annotations

import abc
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.tokenizer as _tokenizer
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

TURN_TASK_NAME = "TurnOnSinkFaucet"
TURN_PROMPT = "Turn on the sink faucet."
TURN_CONFIG_NAME = "pi0_robocasa_turn_success10_weak"


def _release_root() -> pathlib.Path:
    env_root = os.environ.get("REVAMP_RELEASE_ROOT")
    if env_root:
        return pathlib.Path(env_root).expanduser().resolve()
    return pathlib.Path(__file__).resolve().parents[5]


def _turn_success_data_dir() -> str:
    return str(
        _release_root()
        / "datasets"
        / "turn_rollout_120_trimmed_f15"
        / "success"
    )


def _pad(values: Sequence[float], target_dim: int, fill: float) -> np.ndarray:
    values = list(values)
    if len(values) > target_dim:
        return np.asarray(values[:target_dim], dtype=np.float32)
    return np.asarray(values + [fill] * (target_dim - len(values)), dtype=np.float32)


def _load_turn_dataset_norm_stats(
    data_dirs: list[dict[str, Any]] | None,
    action_dim: int,
) -> dict[str, _transforms.NormStats] | None:
    if not data_dirs:
        return None
    stats_path = pathlib.Path(data_dirs[0]["path"]).expanduser() / "meta" / "stats.json"
    if not stats_path.exists():
        return None

    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    state_raw = raw["observation.state"]
    action_raw = raw["action"]

    # LeRobot export order:
    # state = [base_pos, base_rot, eef_pos_rel, eef_rot_rel, gripper_qpos]
    # action = [base_motion, control_mode, eef_pos, eef_rot, gripper_close]
    # OpenPI RoboCasa order:
    # state = [eef_pos_rel, eef_rot_rel, base_pos, base_rot, gripper_qpos]
    # action = [eef_pos, eef_rot, gripper_close, base_motion, control_mode]
    state_order = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
    action_order = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]

    def ordered(values: Sequence[float], order: Sequence[int]) -> list[float]:
        return [float(values[i]) for i in order]

    return {
        "state": _normalize.NormStats(
            mean=_pad(ordered(state_raw["mean"], state_order), action_dim, 0.0),
            std=_pad(ordered(state_raw["std"], state_order), action_dim, 1.0),
        ),
        "actions": _normalize.NormStats(
            mean=_pad(ordered(action_raw["mean"], action_order), action_dim, 0.0),
            std=_pad(ordered(action_raw["std"], action_order), action_dim, 1.0),
        ),
    }


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Location of normalization assets used by OpenPI transforms."""

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Materialized OpenPI data / transform configuration."""

    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | Callable[[], dict[str, _transforms.NormStats]] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    rlds_data_dir: str | None = None
    action_space: Any | None = None
    action_dim: int | None = None
    data_dirs: list[dict[str, Any]] | None = None
    dataset_weights: list[float] | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group: ...


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Model-side transforms for pi0."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if model_config.model_type != _model.ModelType.PI0:
            raise ValueError(
                "This release artifact only includes the pi0 RoboCasa path; "
                f"got model_type={model_config.model_type!r}."
            )
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                ),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a materialized data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        base = self.base_config or DataConfig()
        existing_stats = base.norm_stats() if callable(base.norm_stats) else base.norm_stats
        loaded_stats = None
        if existing_stats is None and asset_id is not None:
            loaded_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        return dataclasses.replace(
            base,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=existing_stats if existing_stats is not None else loaded_stats,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str) -> dict[str, _transforms.NormStats] | None:
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logging.info("Norm stats not found in %s.", assets_dir / asset_id)
            return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs, model_config
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """RoboCasa LeRobot/GROOT-style data config for TurnOnSinkFaucet."""

    repo_id: str | None = None
    data_dirs: list[dict[str, Any]] | None = None
    dataset_weights: list[float] | None = None
    action_dim: int | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base = self.create_base_config(assets_dirs)
        norm_stats = base.norm_stats or _load_turn_dataset_norm_stats(self.data_dirs, model_config.action_dim)
        data_transforms = _transforms.Group(
            inputs=[
                robocasa_policy.RobocasaInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=TURN_PROMPT)(model_config),
            action_dim=model_config.action_dim,
            data_dirs=self.data_dirs,
            dataset_weights=self.dataset_weights,
            norm_stats=norm_stats,
        )


@dataclasses.dataclass
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    TrainConfig(
        name=TURN_CONFIG_NAME,
        model=pi0.Pi0Config(max_token_len=96),
        data=LeRobotRobocasaDataConfig(
            data_dirs=[
                {
                    "path": _turn_success_data_dir(),
                    "filter_key": None,
                    "prompt": TURN_PROMPT,
                }
            ],
        ),
        num_train_steps=10_000,
        save_interval=1000,
        keep_period=5000,
        batch_size=8,
        num_workers=4,
        ema_decay=None,
        policy_metadata={
            "task": TURN_TASK_NAME,
            "prompt": TURN_PROMPT,
            "weak_success_demos": 10,
        },
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config {config_name!r} not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
