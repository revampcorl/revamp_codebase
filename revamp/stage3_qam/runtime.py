"""Runtime helpers for distributed OpenPI QAM training.

This module keeps transport, checkpointing, and resource-management utilities
out of the training script so the script can focus on the QAM update flow.
"""

from __future__ import annotations

import http.client
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any, Callable
from urllib.parse import urlparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import DictConfig, ListConfig, OmegaConf


LogFn = Callable[[str], None]


def cfg_get(config, path: str, default=None):
    value = config
    for key in path.split("."):
        if value is None:
            return default
        try:
            value = value.get(key, default)
        except AttributeError:
            value = getattr(value, key, default)
    return value


def cfg_to_container(config):
    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return {k: cfg_to_container(v) for k, v in config.items()}
    if isinstance(config, list):
        return [cfg_to_container(v) for v in config]
    return config


def optional_container(value):
    if value is None:
        return None
    return cfg_to_container(value)


def cfg_get_int(section, key: str, default: int) -> int:
    value = section.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def cfg_get_float(section, key: str, default: float) -> float:
    value = section.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def action_norm_stats(checkpoint_path: str) -> dict[str, np.ndarray]:
    path = Path(checkpoint_path) / "assets" / "norm_stats.json"
    data = json.loads(path.read_text())["norm_stats"]["actions"]
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items() if v is not None}


def unnormalize_actions(a_norm: np.ndarray, stats: dict[str, np.ndarray], use_quantiles: bool) -> np.ndarray:
    if use_quantiles:
        return (a_norm + 1.0) / 2.0 * (stats["q99"] - stats["q01"] + 1e-6) + stats["q01"]
    return a_norm * (stats["std"] + 1e-6) + stats["mean"]


def normalization_scale(stats: dict[str, np.ndarray], use_quantiles: bool) -> np.ndarray:
    if use_quantiles:
        return (stats["q99"] - stats["q01"] + 1e-6) / 2.0
    return stats["std"] + 1e-6


def _npz_bytes(payload: dict[str, np.ndarray], *, compressed: bool = True) -> bytes:
    bio = BytesIO()
    if compressed:
        np.savez_compressed(bio, **payload)
    else:
        np.savez(bio, **payload)
    return bio.getvalue()


class QGradientClient:
    """Small keep-alive HTTP client for the local PyTorch Q oracle."""

    def __init__(self, url: str, *, compressed_payload: bool = True, timeout: float = 600.0):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", ""}:
            raise ValueError(f"Only local http Q server URLs are supported, got {url!r}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = int(parsed.port or 80)
        self.prefix = parsed.path.rstrip("/")
        self.compressed_payload = bool(compressed_payload)
        self.timeout = float(timeout)
        self._conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def post_grad(self, payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
        data = self._post_npz("/grad", payload)
        timings = {
            "q_server_decode_s": float(np.asarray(data["decode_s"]).item()) if "decode_s" in data else 0.0,
            "q_server_wait_s": float(np.asarray(data["wait_s"]).item()) if "wait_s" in data else 0.0,
            "q_server_compute_s": float(np.asarray(data["compute_s"]).item()) if "compute_s" in data else 0.0,
            "q_server_encode_s": float(np.asarray(data["encode_s"]).item()) if "encode_s" in data else 0.0,
        }
        if "q_values" not in data:
            raise RuntimeError("Q server /grad response is missing q_values; restart q_gradient_server.py")
        q_values = data["q_values"].astype(np.float32)
        return data["grad"].astype(np.float32), q_values, float(np.asarray(data["q"]).item()), timings

    def _post_npz(self, endpoint: str, payload: dict[str, np.ndarray]):
        body = _npz_bytes(payload, compressed=self.compressed_payload)
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
        }
        path = f"{self.prefix}{endpoint}" if self.prefix else endpoint
        for attempt in range(2):
            conn = self._connection()
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"Q server returned HTTP {resp.status}: {raw[:512]!r}")
                data = np.load(BytesIO(raw))
                return {k: data[k] for k in data.files}
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable QGradientClient retry state")


class WorldModelImagClient:
    """HTTP client for the local Stage-1 world-model imagination server."""

    def __init__(self, url: str, *, compressed_payload: bool = True, timeout: float = 1200.0):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", ""}:
            raise ValueError(f"Only local http world-model server URLs are supported, got {url!r}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = int(parsed.port or 80)
        self.prefix = parsed.path.rstrip("/")
        self.compressed_payload = bool(compressed_payload)
        self.timeout = float(timeout)
        self._conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def post_predict(self, payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        data = self._post_npz("/predict_chunk", payload)
        required = ("next_cam_left", "next_cam_right", "next_cam_high")
        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(f"world-model imagination response is missing keys: {missing}")
        timings = {
            "imag_server_decode_s": float(np.asarray(data["decode_s"]).item()) if "decode_s" in data else 0.0,
            "imag_server_wait_s": float(np.asarray(data["wait_s"]).item()) if "wait_s" in data else 0.0,
            "imag_server_compute_s": float(np.asarray(data["compute_s"]).item()) if "compute_s" in data else 0.0,
            "imag_server_encode_s": float(np.asarray(data["encode_s"]).item()) if "encode_s" in data else 0.0,
        }
        pred = {k: data[k].astype(np.float32) for k in required}
        return pred, timings

    def _post_npz(self, endpoint: str, payload: dict[str, np.ndarray]):
        body = _npz_bytes(payload, compressed=self.compressed_payload)
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
        }
        path = f"{self.prefix}{endpoint}" if self.prefix else endpoint
        for attempt in range(2):
            conn = self._connection()
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"world-model server returned HTTP {resp.status}: {raw[:512]!r}")
                data = np.load(BytesIO(raw))
                return {k: data[k] for k in data.files}
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable WorldModelImagClient retry state")


def validate_q_response(
    *,
    q_grad_raw: np.ndarray,
    q_values: np.ndarray,
    q_mean: float,
    expected_shape: tuple[int, ...],
) -> None:
    if q_grad_raw.shape != expected_shape:
        raise RuntimeError(f"Q server returned grad shape {q_grad_raw.shape}, expected {expected_shape}")
    expected_q_shape = (expected_shape[0],)
    if q_values.shape != expected_q_shape:
        raise RuntimeError(f"Q server returned q_values shape {q_values.shape}, expected {expected_q_shape}")
    if not np.isfinite(q_grad_raw).all() or not np.isfinite(q_values).all() or not np.isfinite(q_mean):
        raise RuntimeError("Q server returned non-finite q/grad during Stage3 training")


def restore_orbax_item(path: Path):
    ckptr = ocp.PyTreeCheckpointer()
    return ckptr.restore(path)


def rng_from_jsonable(data):
    rng_data = jnp.asarray(data, dtype=jnp.uint32)
    try:
        return jax.random.wrap_key_data(rng_data)
    except TypeError:
        return rng_data


def _to_host_pytree(tree):
    return jax.tree.map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree)


def _rng_to_jsonable(rng) -> list[int]:
    try:
        rng = jax.random.key_data(rng)
    except TypeError:
        pass
    return np.asarray(jax.device_get(rng)).astype(np.uint32).tolist()


def save_checkpoint(
    *,
    out_dir: Path,
    step: int,
    params,
    opt_state=None,
    rng=None,
    dataloader_seed: int | None = None,
    logs: list[dict[str, Any]],
    assets_src: Path,
    wandb_run=None,
    log: LogFn | None = None,
) -> Path:
    log = log or (lambda msg: print(f"[qam_runtime] {msg}", flush=True))
    step_dir = out_dir / f"openpi_qam_dist_step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "openpi_qam_log.json").write_text(json.dumps(logs, indent=2) + "\n")

    if assets_src.exists():
        assets_dst = step_dir / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)

    ckptr = ocp.PyTreeCheckpointer()
    log(f"copying sharded params to host for checkpoint save at step={step}")
    host_params = _to_host_pytree(params)
    params_dir = step_dir / "params"
    if params_dir.exists():
        shutil.rmtree(params_dir)
    for tmp_dir in step_dir.glob("params.orbax-checkpoint-tmp-*"):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    ckptr.save(params_dir, {"params": host_params})
    if hasattr(ckptr, "wait_until_finished"):
        ckptr.wait_until_finished()

    if opt_state is not None:
        opt_dir = step_dir / "opt_state"
        if opt_dir.exists():
            shutil.rmtree(opt_dir)
        for tmp_dir in step_dir.glob("opt_state.orbax-checkpoint-tmp-*"):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        host_opt_state = _to_host_pytree(opt_state)
        ckptr.save(opt_dir, {"opt_state": host_opt_state})
        if hasattr(ckptr, "wait_until_finished"):
            ckptr.wait_until_finished()

    trainer_state = {"step": int(step)}
    if rng is not None:
        trainer_state["rng"] = _rng_to_jsonable(rng)
    if dataloader_seed is not None:
        trainer_state["dataloader_seed"] = int(dataloader_seed)
    (step_dir / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2) + "\n")

    log(f"saved checkpoint to {step_dir}")
    if wandb_run is not None:
        wandb_run.summary["latest_checkpoint_dir"] = str(step_dir)
        wandb_run.summary["latest_checkpoint_step"] = step
    return step_dir
