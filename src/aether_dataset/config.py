from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SplitConfig:
    config: str
    split: str
    expected_audio_hours: float


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def output_path(self) -> Path:
        return Path(self.raw["output"]["path"]).expanduser()

    @property
    def target_shard_seconds(self) -> float:
        return float(self.raw["output"]["target_audio_hours_per_shard"]) * 3600.0

    @property
    def splits(self) -> dict[str, SplitConfig]:
        return {
            name: SplitConfig(
                config=value["config"],
                split=value["split"],
                expected_audio_hours=float(value["expected_audio_hours"]),
            )
            for name, value in self.raw["dataset"]["splits"].items()
        }

    def require_ready(self) -> None:
        placeholders = _find_placeholders(self.raw)
        if placeholders:
            joined = ", ".join(placeholders)
            raise ConfigError(f"configuration contains unresolved placeholders: {joined}")
        revision = str(self.raw["dataset"]["revision"])
        mimi_revision = str(self.raw["mimi"]["revision"])
        if revision in {"main", "master"} or mimi_revision in {"main", "master"}:
            raise ConfigError("dataset and Mimi revisions must be immutable commit revisions")
        fraction = float(self.raw["runtime"]["target_vram_fraction"])
        if not 0.1 <= fraction <= 0.98:
            raise ConfigError("runtime.target_vram_fraction must be between 0.1 and 0.98")


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("top-level configuration must be a mapping")
    for key in ("dataset", "mimi", "text", "output", "runtime"):
        if key not in raw:
            raise ConfigError(f"missing configuration section: {key}")
    return AppConfig(raw=raw, path=config_path)


def _find_placeholders(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_placeholders(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_placeholders(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and value.startswith("REQUIRED_"):
        found.append(prefix)
    return found
