"""
evaluate.py – evaluation metrics, statistical helpers & plotting utilities
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_from_disk

# -----------------------------------------------------------------------------
# Resource probes
# -----------------------------------------------------------------------------

def gpu_mem() -> int:
    """Peak allocated GPU memory in MiB (returns 0 on CPU-only hosts)."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0


def energy_joules(duration_s: float) -> float:
    """Rough energy estimate via current GPU power draw * time."""
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        power = float(smi.decode().strip().split("\n")[0])  # Watts
        return power * duration_s
    except Exception:
        return -1.0  # unavailable


# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------

def simple_accuracy(model, tokenizer, ds) -> float:
    """Extremely lightweight accuracy: substring match of gold in prediction."""
    model.eval()
    correct = 0
    total = len(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for item in ds:
        prompt = item.get("question", item.get("input", ""))
        gold = item.get("answer", "").strip().lower()
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**toks, max_new_tokens=8)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).lower()
        correct += int(gold in pred)
    return correct / max(total, 1)


# -----------------------------------------------------------------------------
# Plotting utilities – saved into .research/iteration1/images/
# -----------------------------------------------------------------------------

matplotlib.use("Agg")


def save_bar(values: List[float], names: List[str], title: str, filename: Path):
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=sns.color_palette("deep"))
    ax.set_ylabel(title)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
