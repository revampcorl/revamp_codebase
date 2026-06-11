from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_actions_with_traj = (
            nnx_utils.module_jit(model.sample_actions_with_traj)
            if hasattr(model, "sample_actions_with_traj")
            else None
        )
        self._vector_field = (
            nnx_utils.module_jit(model.vector_field)
            if hasattr(model, "vector_field")
            else None
        )
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _prepare_batched_observation(self, obs: dict) -> _model.Observation:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def infer_with_traj(self, obs: dict, *, num_steps: int | None = None) -> dict:
        """Infer actions and also return the pi0 denoising trajectory.

        This is intended for QAM integration/debugging. Returned `actions`
        are passed through the normal output transforms. `traj_actions` are
        also output-scale actions with shape [num_steps, action_horizon, dim].
        """
        if self._sample_actions_with_traj is None:
            raise NotImplementedError("This model does not expose sample_actions_with_traj.")

        observation = self._prepare_batched_observation(obs)
        self._rng, sample_rng = jax.random.split(self._rng)
        kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        actions, (traj_actions, traj_times) = self._sample_actions_with_traj(sample_rng, observation, **kwargs)

        outputs = {"state": observation.state, "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)

        unnorm_traj = []
        for i in range(np.asarray(traj_actions).shape[0]):
            step_out = {
                "state": np.asarray(observation.state[0, ...]),
                "actions": np.asarray(traj_actions[i, 0, ...]),
            }
            unnorm_traj.append(self._output_transform(step_out)["actions"])

        outputs["traj_actions"] = np.stack(unnorm_traj, axis=0)
        outputs["traj_times"] = np.asarray(traj_times)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
