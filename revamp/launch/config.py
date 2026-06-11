from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expand_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(f"Missing environment variable: {name}")

    return _ENV_PATTERN.sub(replace, value)


def walk_expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: walk_expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [walk_expand(item) for item in value]
    if isinstance(value, str):
        return expand_env(value)
    return value


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    root = root or repo_root()
    path = Path(expand_env(str(value))).expanduser()
    if path.is_absolute():
        return path
    return root / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = walk_expand(json.load(handle))

    root = repo_root()
    config.setdefault("resolved", {})
    config["resolved"]["repo_root"] = str(root)
    config["resolved"]["config_path"] = str(config_path)
    config["resolved"]["output_dir"] = str(resolve_path(config["outputs"]["dir"], root=root))
    return config
