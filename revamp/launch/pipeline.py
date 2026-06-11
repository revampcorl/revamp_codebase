from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from revamp.launch.config import repo_root, resolve_path
from revamp.launch.console import print_box
from revamp.launch.files import ensure_dir, write_json


@dataclass
class PipelineCommand:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path | None = None
    health_url: str | None = None


def run_imagination_policy_pipeline(config: dict[str, Any], *, dry_run: bool, smoke_steps: int | None = None) -> None:
    materialized = materialize_world_model_configs(config)
    commands = build_imagination_commands(config, materialized, smoke_steps=smoke_steps)
    print_commands("REVAMP Imagination Policy", commands)
    if dry_run:
        return

    output_dir = ensure_dir(config["resolved"]["output_dir"])
    write_json(Path(output_dir) / "imagination_policy_commands.json", commands_to_json(commands))
    q_server, imag_server, trainer = commands
    with ManagedProcess(q_server), ManagedProcess(imag_server):
        run_foreground(trainer)


def run_world_model_online_retrain_pipeline(
    config: dict[str, Any],
    *,
    dry_run: bool,
    with_rollout_check: bool = False,
) -> None:
    materialized = materialize_world_model_configs(config)
    commands = build_update_commands(
        config,
        materialized,
        skip_retrain=False,
        skip_rollout_check=not with_rollout_check,
    )
    print_commands("REVAMP World-Model Online Retrain", commands)
    if dry_run:
        return
    if not commands:
        raise SystemExit("No world-model online retrain command is enabled.")

    output_dir = ensure_dir(config["resolved"]["output_dir"])
    write_json(Path(output_dir) / "world_model_online_retrain_commands.json", commands_to_json(commands))
    for command in commands:
        run_foreground(command)


def materialize_world_model_configs(config: dict[str, Any]) -> dict[str, Path]:
    root = repo_root()
    settings = config["pipeline"]
    output_dir = ensure_dir(resolve_path(config["outputs"]["materialized_config_dir"]))

    phase3 = resolve_path(settings["source_configs"]["phase3_qam"])
    wm_update = resolve_path(settings["source_configs"]["wm_online_update"])
    phase3_out = Path(output_dir) / "phase3_qam_turn_openpi_two_chunk_imag.release.yaml"
    wm_update_out = Path(output_dir) / "phase1_wm_online_update.release.yaml"

    origin_dataset = resolve_path(settings["assets"]["stage3_qam_dataset"])
    branch_dataset = resolve_path(settings["assets"]["online_recovery_dataset"])
    new_scene_dataset = resolve_path(settings["assets"]["new_scene_dataset"])
    replacements = {
        "datasets/assets/t5_embedding/prompts/robocasa/TurnOnSinkFaucet.pt": str(
            resolve_path(settings["assets"]["t5_embedding"])
        ),
        "datasets/origin/success": str(origin_dataset / "success"),
        "datasets/origin/failure": str(origin_dataset / "failure"),
        "datasets/online_recovery_rollouts/success": str(branch_dataset / "success"),
        "datasets/new_scene_data/success": str(new_scene_dataset / "success"),
        "datasets/new_scene_data/failure": str(new_scene_dataset / "failure"),
        "datasets/assets/wan/Wan2.2-TI2V-5B/Wan2.2_VAE.pth": str(
            resolve_path(settings["assets"]["wan_checkpoint_dir"]) / "Wan2.2_VAE.pth"
        ),
        "datasets/assets/wan/Wan2.2-TI2V-5B": str(resolve_path(settings["assets"]["wan_checkpoint_dir"])),
        "third_party/openpi": str(resolve_path(settings["third_party"]["openpi_root"])),
        "checkpoints/openpi_pi0_turn": str(
            resolve_path(settings["checkpoints"]["openpi_policy"])
        ),
        "outputs/turn_on_sink_faucet/checkpoints/phase3_qam_turn_openpi_two_chunk_imag_online": str(
            root
            / "outputs"
            / "turn_on_sink_faucet"
            / "checkpoints"
            / "phase3_qam_turn_openpi_two_chunk_imag_online"
        ),
        "checkpoints/stage2_q_initial": str(resolve_path(settings["checkpoints"]["stage2_q_initial"])),
        "checkpoints/world_model_initial": str(
            resolve_path(settings["checkpoints"]["world_model_initial"])
        ),
        "checkpoints/world_model_after_online_update": str(
            resolve_path(settings["checkpoints"]["world_model_after_online_update"])
        ),
        "outputs/turn_on_sink_faucet/checkpoints/world_model_online_update": str(
            root / "outputs" / "turn_on_sink_faucet" / "checkpoints" / "world_model_online_update"
        ),
    }
    _write_replaced_yaml(phase3, phase3_out, replacements)
    _write_replaced_yaml(wm_update, wm_update_out, replacements)
    return {"phase3_qam": phase3_out, "wm_online_update": wm_update_out}


def build_imagination_commands(
    config: dict[str, Any],
    materialized: dict[str, Path],
    *,
    smoke_steps: int | None = None,
) -> list[PipelineCommand]:
    root = repo_root()
    settings = config["pipeline"]
    imag_cfg = settings["imagination_policy"]
    q_cfg = imag_cfg["q_server"]
    wm_cfg = imag_cfg["imagination_server"]
    trainer_cfg = imag_cfg["trainer"]
    q_port = int(q_cfg["port"])
    wm_port = int(wm_cfg["port"])
    q_url = f"http://127.0.0.1:{q_port}"
    wm_url = f"http://127.0.0.1:{wm_port}"
    output_dir = Path(config["resolved"]["output_dir"])

    q_cmd = [
        "python",
        "-m",
        "revamp.stage3_qam.q_gradient_server",
        "--config",
        str(materialized["phase3_qam"]),
        "--port",
        str(q_port),
        "--device",
        str(q_cfg["device"]),
    ]

    wm_cmd = [
        "python",
        "-m",
        "revamp.stage3_qam.imagination_server",
        "--config",
        str(materialized["wm_online_update"]),
        "--checkpoint",
        str(resolve_path(settings["checkpoints"]["stage2_q_initial"])),
        "--port",
        str(wm_port),
        "--device",
        str(wm_cfg["device"]),
    ]
    _add(wm_cmd, "--num-inference-steps", wm_cfg.get("num_inference_steps"))

    trainer_cmd = [
        "python",
        "-m",
        "revamp.stage3_qam.train_policy",
        "--config",
        str(materialized["phase3_qam"]),
        "--q-server-url",
        q_url,
        "--imag-server-url",
        wm_url,
        "--log-interval",
        str(trainer_cfg["log_interval"]),
    ]
    if smoke_steps is not None:
        trainer_cmd.extend(["--max-steps", str(smoke_steps), "--skip-save"])

    return [
        PipelineCommand(
            "q_server",
            q_cmd,
            root,
            _pipeline_env(settings, q_cfg.get("cuda_visible_devices")),
            output_dir / "logs" / "q_server.log",
            f"{q_url}/health",
        ),
        PipelineCommand(
            "world_model_imagination_server",
            wm_cmd,
            root,
            _pipeline_env(settings, wm_cfg.get("cuda_visible_devices")),
            output_dir / "logs" / "world_model_imagination_server.log",
            f"{wm_url}/health",
        ),
        PipelineCommand(
            "openpi_qam_policy_trainer",
            trainer_cmd,
            root,
            _pipeline_env(settings, trainer_cfg.get("cuda_visible_devices")),
            output_dir / "logs" / "openpi_qam_policy_trainer.log",
        ),
    ]


def build_update_commands(
    config: dict[str, Any],
    materialized: dict[str, Path],
    *,
    skip_retrain: bool,
    skip_rollout_check: bool,
) -> list[PipelineCommand]:
    root = repo_root()
    settings = config["pipeline"]
    update = settings["world_model_update"]
    output_dir = Path(config["resolved"]["output_dir"])
    commands: list[PipelineCommand] = []

    if not skip_retrain:
        retrain = update["retrain"]
        retrain_cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(retrain["num_processes"]),
            "--main_process_port",
            str(retrain["main_process_port"]),
            "--mixed_precision",
            str(retrain["mixed_precision"]),
            "-m",
            "revamp.stage1_2_world_model.train",
            "--config",
            str(materialized["wm_online_update"]),
        ]
        commands.append(
            PipelineCommand(
                "retrain_world_model",
                retrain_cmd,
                root,
                _pipeline_env(settings, retrain.get("cuda_visible_devices")),
                output_dir / "logs" / "retrain_world_model.log",
            )
        )

    if not skip_rollout_check:
        rollout = update["rollout_check"]
        rollout_cmd = [
            "python",
            "-m",
            "revamp.stage1_2_world_model.rollout_check",
            "--config",
            str(materialized["wm_online_update"]),
            "--checkpoint",
            str(resolve_path(settings["checkpoints"]["world_model_after_online_update"])),
            "--sample-index",
            str(rollout["sample_index"]),
            "--out-dir",
            str(resolve_path(rollout["out_dir"])),
            "--num-inference-steps",
            str(rollout["num_inference_steps"]),
            "--device",
            str(rollout["device"]),
            "--fps",
            str(rollout["fps"]),
        ]
        commands.append(
            PipelineCommand(
                "rollout_check",
                rollout_cmd,
                root,
                _pipeline_env(settings, rollout.get("cuda_visible_devices")),
                output_dir / "logs" / "rollout_check.log",
            )
        )
    return commands


class ManagedProcess:
    def __init__(self, command: PipelineCommand):
        self.command = command
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def __enter__(self) -> "ManagedProcess":
        if self.command.log_path is not None:
            self.command.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = self.command.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command.command,
                cwd=self.command.cwd,
                env=self.command.env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT if self.log_handle else None,
                text=True,
                preexec_fn=os.setsid,
            )
            if self.command.health_url:
                wait_for_health(self.command.health_url)
        except Exception:
            self._terminate()
            if self.log_handle is not None:
                self.log_handle.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()
        if self.log_handle is not None:
            self.log_handle.close()

    def _terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=20)


def run_foreground(command: PipelineCommand) -> None:
    if command.log_path is not None:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        with command.log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command.command,
                cwd=command.cwd,
                env=command.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            returncode = process.wait()
    else:
        result = subprocess.run(command.command, cwd=command.cwd, env=command.env, check=False)
        returncode = result.returncode
    if returncode:
        raise SystemExit(returncode)


def wait_for_health(url: str, timeout_sec: float = 300.0) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url}") from last_error


def print_commands(title: str, commands: list[PipelineCommand]) -> None:
    lines: list[str] = []
    for command in commands:
        lines.append(f"[{command.name}]")
        lines.append(f"cwd: {command.cwd}")
        lines.append("env: " + _summarize_env(command.env))
        lines.append(shlex.join(command.command))
        if command.health_url:
            lines.append(f"health: {command.health_url}")
        lines.append("")
    print_box(title, lines)


def commands_to_json(commands: list[PipelineCommand]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "cwd": str(item.cwd),
            "command": item.command,
            "log_path": str(item.log_path) if item.log_path else None,
            "health_url": item.health_url,
            "env": {
                key: item.env[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "REVAMP_OPENPI_ROOT",
                    "REVAMP_WAN_CODE_ROOT",
                    "REVAMP_ROBOCASA_ASSETS_ROOT",
                )
                if key in item.env
            },
        }
        for item in commands
    ]


def _write_replaced_yaml(source: Path, target: Path, replacements: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    pattern = re.compile("|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True)))
    text = pattern.sub(lambda match: replacements[match.group(0)], text)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _pipeline_env(settings: dict[str, Any], cuda_visible_devices: str | None, extra: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    root = repo_root()
    python_paths = [
        str(root),
        str(resolve_path(settings["third_party"]["openpi_root"]) / "src"),
        str(resolve_path(settings["third_party"]["openpi_client_src"])),
        str(resolve_path(settings["third_party"]["robocasa_root"])),
        str(resolve_path(settings["third_party"]["wan_root"])),
    ]
    robosuite_root = settings["third_party"].get("robosuite_root")
    if robosuite_root:
        python_paths.append(str(resolve_path(robosuite_root)))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    deduped_python_paths: list[str] = []
    for path in python_paths:
        if not path:
            continue
        normalized = str(Path(path))
        if normalized not in deduped_python_paths:
            deduped_python_paths.append(normalized)
    env["PYTHONPATH"] = os.pathsep.join(deduped_python_paths)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("ROBOCASA_MJCF_TMPDIR", "/tmp/robocasa_mjcf_tmp")
    env.setdefault("REVAMP_RELEASE_ROOT", str(root))
    env.setdefault("REVAMP_ROBOCASA_ROOT", str(resolve_path(settings["third_party"]["robocasa_root"])))
    env.setdefault("REVAMP_ROBOCASA_ASSETS_ROOT", str(resolve_path(settings["assets"]["robocasa_assets_root"])))
    env.setdefault("REVAMP_OPENPI_ROOT", str(resolve_path(settings["third_party"]["openpi_root"])))
    env.setdefault("REVAMP_WAN_CODE_ROOT", str(resolve_path(settings["third_party"]["wan_root"])))
    robosuite_root = settings["third_party"].get("robosuite_root")
    if robosuite_root:
        env.setdefault("REVAMP_ROBOSUITE_ROOT", str(resolve_path(robosuite_root)))
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    for key, value in (extra or {}).items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def _add(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _summarize_env(env: dict[str, str]) -> str:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "MUJOCO_GL",
        "REVAMP_OPENPI_ROOT",
        "REVAMP_WAN_CODE_ROOT",
        "REVAMP_ROBOCASA_ASSETS_ROOT",
    ]
    return ", ".join(f"{key}={env[key]}" for key in keys if key in env)
