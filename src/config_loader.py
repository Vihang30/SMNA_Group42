"""Reads config.yaml and turns relative paths into full paths from the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    for key in ("raw_dir", "processed_dir", "sample_dir", "figures_dir", "tables_dir"):
        rel = cfg["paths"][key]
        cfg["paths"][key] = str(ROOT / rel)
    return cfg
