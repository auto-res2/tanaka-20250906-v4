"""
preprocess.py – configuration, data acquisition & reproducibility utilities
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import load_dataset, load_from_disk

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

CONFIG_YAML = Path("config/config.yaml")

_DEFAULT_CFG: Dict[str, Any] = {
    "datasets": {
        "gsm8k": "openai/gsm8k"
    },
    "models": {
        "base_name": "togethercomputer/RedPajama-INCITE-7B-Base",
        "base_quant": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "device_map": "auto",
        },
        "flash_writer": {
            "d_model": 1024,
            "depth": 12,
            "latents": 8,
            "vocab": 32000,
            "max_ctx": 4096,
        },
    },
    "hyperparams": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_steps": 100,
        "support_k": 8,
    },
    "seeds": [11, 29, 97],
}


def ensure_config() -> None:
    CONFIG_YAML.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_YAML.exists():
        with CONFIG_YAML.open("w") as f:
            yaml.safe_dump(_DEFAULT_CFG, f)


def load_cfg() -> Dict[str, Any]:
    ensure_config()
    with CONFIG_YAML.open() as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Data handling – lightweight wrapper around HF datasets
# -----------------------------------------------------------------------------

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(exist_ok=True)


class DataManager:
    """Download & cache datasets under ./data."""

    def __init__(self):
        self.cfg = load_cfg()["datasets"]

    # ------------------------------------------------------------------
    def _path(self, name: str) -> Path:
        return _DATA_ROOT / name

    # ------------------------------------------------------------------
    def _download(self, name: str):
        repo_id = self.cfg[name]
        print(f"[DataManager] Downloading {repo_id} …", flush=True)
        ds = load_dataset(repo_id)
        tgt = self._path(name)
        ds.save_to_disk(str(tgt))
        return tgt

    # ------------------------------------------------------------------
    def get(self, name: str):
        tgt = self._path(name)
        if tgt.exists():
            return tgt
        return self._download(name)


# -----------------------------------------------------------------------------
# Reproducibility helper
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
