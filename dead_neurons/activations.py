#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dead Neuron Benchmark across Activation Functions
=================================================

Runs the same model/dataset under different activation functions
and logs per-layer + global dead neuron ratios into Neptune Scale.

Each activation type gets its own Run inside the same experiment.

Example:
  python benchmark_activations.py \
    --experiment-name activation-benchmark \
    --model distilgpt2 \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --max-tokens 128 --batches 20 --batch-size 8 \
    --eps-dead 1e-4
"""

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from neptune_scale import Run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Activation swap (GPT-2 family)
# -----------------------
def make_activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "relu":
        return nn.ReLU()
    if kind == "elu":
        return nn.ELU()
    if kind == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if kind == "gelu":
        return nn.GELU()
    if kind == "swish":
        return nn.SiLU()  # Swish
    raise ValueError(f"Unsupported activation: {kind}")


def swap_gpt2_block_activations(model, kind: str):
    act = make_activation(kind)
    # GPT-2 style modules: transformer.h[*].mlp.act
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("This swapper expects a GPT-2-like model (e.g., distilgpt2).")
    for block in model.transformer.h:
        block.mlp.act = make_activation(kind)  # fresh module per block
    return model


# -----------------------
# Hooks to capture per-layer post-MLP activations
# -----------------------
class ActCatcher:
    def __init__(self):
        self.buffers: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _save(self, name: str):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self.buffers.setdefault(name, []).append(x.detach().float().cpu())  # [B,S,H]
        return hook

    def register_gpt2(self, model) -> None:
        for li, block in enumerate(model.transformer.h):
            # post-activation (after act) is best seen by prehooking the next linear (c_proj) input,
            # but GPT2's MLP is: c_fc -> act -> c_proj, with 'act' as a Module. Hooking act output works fine:
            self.hooks.append(block.mlp.act.register_forward_hook(self._save(f"layer{li}/mlp_post")))

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass


# -----------------------
# Dead neuron metric (pad-aware)
# -----------------------
def masked_mean_abs(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # acts: [N,S,H], mask: [N,S,1]
    masked = acts.abs() * mask
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)  # [1] or [H] via broadcast
    return masked.sum(dim=(0, 1)) / denom  # [H]


def dead_ratio_from_acts(acts: torch.Tensor, pad_mask: torch.Tensor, eps: float) -> float:
    # acts: [N,S,H], pad_mask: [N,S,1]
    mean_abs = masked_mean_abs(acts, pad_mask)  # [H]
    return (mean_abs <= eps).float().mean().item()


# -----------------------
# Core run
# -----------------------
def run_activation_experiment(args, activation: str):
    run = Run(experiment_name=args.experiment_name)
    run.log_configs({
        "label": activation,
        "activation": activation,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "max_tokens": args.max_tokens,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "eps_dead": args.eps_dead,
    })

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        # Prefer EOS if present; otherwise add PAD
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "left"  # safer for causal models

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True).to(DEVICE).eval()
    model = swap_gpt2_block_activations(model, activation)

    # Hooks
    catcher = ActCatcher()
    catcher.register_gpt2(model)

    # Data
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = [ex["text"] for ex in ds if isinstance(ex.get("text"), str) and ex["text"].strip()]
    texts = texts[: args.batch_size * args.batches]

    # Forward passes
    pad_masks = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i : i + args.batch_size]
            enc = tok(batch, truncation=True, padding="max_length",
                      max_length=args.max_tokens, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in enc.items()}
            _ = model(**inputs)
            pad_masks.append(inputs["attention_mask"].detach().cpu())

    pad_mask = torch.cat(pad_masks, dim=0).unsqueeze(-1).float()  # [N,S,1]

    # Consolidate per-layer buffers, compute per-layer dead ratios, log with step=layer_idx
    per_layer: List[Tuple[int, float]] = []
    for key in sorted(catcher.buffers.keys()):
        # key like "layer5/mlp_post"
        try:
            layer_idx = int(key.split("/", 1)[0].replace("layer", ""))
        except Exception:
            continue
        acts = torch.cat(catcher.buffers[key], dim=0)  # [N,S,H]
        ratio = dead_ratio_from_acts(acts, pad_mask, eps=args.eps_dead)
        per_layer.append((layer_idx, ratio))
        run.log_metrics(data={f"dead_ratio/mlp_post": float(ratio)}, step=layer_idx)

    # Global summary
    if per_layer:
        global_mean = float(sum(r for _, r in per_layer) / len(per_layer))
        run.log_metrics(data={"dead_ratio/global_mean": global_mean}, step=0)

    catcher.close()
    run.close()
    print(f"[{activation}] global_mean dead ratio: {global_mean:.6f}")


# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", type=str, default="activation-benchmark")
    p.add_argument("--model", type=str, default="distilgpt2")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eps-dead", type=float, default=1e-1)
    args = p.parse_args()

    # The five activations you wanted
    activations = ["relu", "elu", "leakyrelu", "gelu", "swish"]

    for act in activations:
        run_activation_experiment(args, act)


if __name__ == "__main__":
    main()
