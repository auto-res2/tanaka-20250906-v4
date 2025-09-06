"""
train.py – model definitions and training / adaptation routines
Extracted from the monolithic FlashP.E.T experimental script and
re-organised to comply with the new multi-file structure.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocess import load_cfg
from .evaluate import gpu_mem, energy_joules

# -----------------------------------------------------------------------------
# Global singletons to avoid expensive re-loading
# -----------------------------------------------------------------------------
_CFG: Dict[str, Any] = load_cfg()
_TOKENIZER = None  # type: ignore
_BASE_MODEL = None  # type: ignore


def base_tokenizer():
    """Shared tokenizer instance (AutoTokenizer)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            _CFG["models"]["base_name"], use_fast=True
        )
    return _TOKENIZER


def load_base_llm() -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load (or return cached) 4-bit quantised base LLM."""
    global _BASE_MODEL
    if _BASE_MODEL is not None:
        return _BASE_MODEL, base_tokenizer()

    quant_args = _CFG["models"]["base_quant"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        _CFG["models"]["base_name"], device_map=device, **quant_args
    )
    _BASE_MODEL = model
    return _BASE_MODEL, base_tokenizer()


# -----------------------------------------------------------------------------
# LoRA helper
# -----------------------------------------------------------------------------

def attach_lora(model: torch.nn.Module, r: int = 16, alpha: int = 32) -> torch.nn.Module:
    """Wrap `model` in a PEFT-LoRA adapter and return the updated model."""
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


# -----------------------------------------------------------------------------
# FlashWriter hyper-network – emits adapter weights in a single forward pass
# -----------------------------------------------------------------------------

class FlashWriter(torch.nn.Module):
    """Compact Perceiver-I/O writer that predicts LoRA weights in one shot."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        from transformers.models.perceiver.modeling_perceiver import (
            PerceiverEncoder,
        )

        self.cfg = cfg
        self.latents = torch.nn.Parameter(torch.randn(cfg["latents"], cfg["d_model"]))
        self.encoder = PerceiverEncoder(
            d_model=cfg["d_model"],
            num_layers=cfg["depth"],
            num_heads=8,
            attention_dropout=0.0,
            dropout=0.0,
        )

        # determine adapter size from base model once
        base_model, _ = load_base_llm()
        total = 0
        for n, p in base_model.named_parameters():
            if any(k in n for k in ("q_proj", "v_proj")):
                total += p.numel()
        self.out = torch.nn.Linear(cfg["d_model"], total, bias=False)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (bs, seq)
        emb = torch.nn.functional.embedding(
            input_ids,
            torch.empty(
                self.cfg["vocab"], self.cfg["d_model"], device=input_ids.device
            ).normal_(0, 1),
        )
        lat = self.latents.expand(emb.size(0), -1, -1)
        x = torch.cat([lat, emb], dim=1)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.out(pooled)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_adapter(self, support_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = self.forward(support_ids)
        base_model, _ = load_base_llm()
        state: Dict[str, torch.Tensor] = {}
        offset = 0
        for n, p in base_model.named_parameters():
            if any(k in n for k in ("q_proj", "v_proj")):
                sz = p.numel()
                half = sz // 2
                state[f"{n}.lora_A"] = (
                    flat[..., offset : offset + half].view(p.shape[0], -1).half()
                )
                state[f"{n}.lora_B"] = (
                    flat[..., offset + half : offset + sz].view(-1, p.shape[1]).half()
                )
                offset += sz
        return state


# -----------------------------------------------------------------------------
# FlashWriter loader – aborts if checkpoint is missing (NO-FALLBACK rule)
# -----------------------------------------------------------------------------

_CKPT_PATH = Path("models/flash_writer.pt")


def load_flash_writer() -> FlashWriter:
    if not _CKPT_PATH.exists():
        raise RuntimeError(
            "FlashWriter checkpoint missing – per strict NO-FALLBACK rule we abort."
        )
    writer = FlashWriter(_CFG["models"]["flash_writer"]).half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer.to(device)
    writer.load_state_dict(torch.load(_CKPT_PATH, map_location="cpu"))
    writer.eval()
    return writer


# -----------------------------------------------------------------------------
# Adaptation methods
# -----------------------------------------------------------------------------

def adapt_flashpet(support_ids: torch.Tensor):
    """Single-pass adapter generation via FlashP.E.T."""
    model, _ = load_base_llm()
    writer = load_flash_writer()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    adapter_state = writer.generate_adapter(support_ids.to(writer.device))

    peft_model = attach_lora(
        model, r=_CFG["hyperparams"]["lora_r"], alpha=_CFG["hyperparams"]["lora_alpha"]
    )
    peft_model.load_adapter(adapter_state, adapter_name="flashpet")

    duration = time.perf_counter() - start
    mem_mb = gpu_mem()
    joules = energy_joules(duration)

    return peft_model, {"time": duration, "vram": mem_mb, "energy": joules}


def adapt_lora(support_dataset):
    """Conventional LoRA fine-tuning for the configured number of steps."""
    from torch.utils.data import DataLoader

    model, tokenizer = load_base_llm()
    model = attach_lora(
        model, r=_CFG["hyperparams"]["lora_r"], alpha=_CFG["hyperparams"]["lora_alpha"]
    )

    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)

    loader = DataLoader(support_dataset, batch_size=1, shuffle=True)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    steps = _CFG["hyperparams"]["lora_steps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        optimiser.zero_grad()
        # naive full-prompt LM training; expects keys suitable for AutoModel
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimiser.step()

    duration = time.perf_counter() - start
    mem_mb = gpu_mem()
    joules = energy_joules(duration)

    model.eval()
    return model, {"time": duration, "vram": mem_mb, "energy": joules}
