"""
main.py – orchestrator that executes the experiment suite and stores results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_from_disk

from .evaluate import save_bar, simple_accuracy
from .preprocess import DataManager, load_cfg, set_seed
from .train import adapt_flashpet, adapt_lora, base_tokenizer

# -----------------------------------------------------------------------------
# Directory housekeeping (.research/iteration1 and images sub-folder)
# -----------------------------------------------------------------------------

ROOT = Path(".research") / "iteration1"
IMG_DIR = ROOT / "images"
ROOT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Run Experiment 1 – FlashP.E.T vs LoRA on GSM8K (subset)
# -----------------------------------------------------------------------------

cfg = load_cfg()
print("\n===== Experiment 1 – One-Shot FlashP.E.T vs Iterative LoRA =====", flush=True)

DM = DataManager()
dataset_path = DM.get("gsm8k")
full_train = load_from_disk(str(dataset_path))["train"]

results = {}

for seed in cfg["seeds"]:
    print(f"\n--- Seed {seed} ---", flush=True)
    set_seed(seed)

    # Support / test split (tiny to keep the demo light-weight)
    support = full_train.shuffle(seed=seed).select(range(cfg["hyperparams"]["support_k"]))
    test = full_train.shuffle(seed=seed + 1).select(range(100))

    # ------------------------------------------------------------------
    # FlashP.E.T adaptation (single forward)
    # ------------------------------------------------------------------
    tokenizer = base_tokenizer()
    sup_ids = tokenizer(
        support["question"], return_tensors="pt", padding=True, truncation=True
    ).input_ids

    model_fpet, res_fpet = adapt_flashpet(sup_ids)
    acc_fpet = simple_accuracy(model_fpet, tokenizer, test)

    # ------------------------------------------------------------------
    # Conventional LoRA fine-tuning
    # ------------------------------------------------------------------
    model_lora, res_lora = adapt_lora(support)
    acc_lora = simple_accuracy(model_lora, tokenizer, test)

    # ------------------------------------------------------------------
    results[f"seed_{seed}"] = {
        "flashpet": {**res_fpet, "accuracy": acc_fpet},
        "lora": {**res_lora, "accuracy": acc_lora},
    }

# -----------------------------------------------------------------------------
# Persist & visualise
# -----------------------------------------------------------------------------

json_path = ROOT / "experiment1_results.json"
json_path.write_text(json.dumps(results, indent=2))
print("\nExperiment 1 JSON results:\n", json.dumps(results, indent=2), flush=True)

# Average accuracy over seeds
avg_acc_fpet = sum(v["flashpet"]["accuracy"] for v in results.values()) / len(results)
avg_acc_lora = sum(v["lora"]["accuracy"] for v in results.values()) / len(results)

plot_path = IMG_DIR / "accuracy_baselines.pdf"
save_bar([avg_acc_fpet, avg_acc_lora], ["FlashP.E.T", "LoRA"], "Accuracy", plot_path)
print(f"Figures saved under {IMG_DIR}", flush=True)
