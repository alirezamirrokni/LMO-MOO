from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return cfg


def save_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

def resolve_method_config(config: dict[str, Any], method: str, dataset: str | None = None) -> dict[str, Any]:
    """Return one method config with optional dataset-specific overrides applied.

    The top-level ``by_dataset`` mapping is selection metadata, not part of the
    method implementation itself.  Removing it after applying the current
    dataset keeps both runtime behavior and cache fingerprints independent of
    unrelated dataset overrides.
    """
    methods = config.get("methods", {})
    if method not in methods:
        raise KeyError(f"unknown method {method!r}")
    cfg = copy.deepcopy(methods[method])
    overrides = cfg.pop("by_dataset", {}) or {}
    if dataset is not None:
        chosen = overrides.get(dataset, {}) or {}
        if not isinstance(chosen, dict):
            raise ValueError(f"methods.{method}.by_dataset.{dataset} must be a mapping")
        cfg.update(copy.deepcopy(chosen))
    return cfg

