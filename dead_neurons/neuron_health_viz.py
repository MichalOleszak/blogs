#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neuron Health Visualizer — Activation Distributions with Neptune Scale
======================================================================

Logs for Neptune Dashboards:
- Scalars: 3 series across layers (step = layer index)
    * dead_ratio/mlp_post
    * dead_ratio/mlp_pre
    * dead_ratio/attn_out
- File series (step = layer index):
    * heatmaps/<layer>/<tensor>        (activation frequency heatmaps with colorbar)
    * histograms/<layer>/<tensor>      (activation histograms; pad tokens excluded)

Example:
  python neuron_health_viz.py \
    --experiment-name dead-neurons \
    --model distilgpt2 \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --max-tokens 128 --batches 50 --batch-size 8 \
    --eps-dead 1e-4 --fire-threshold 1e-3
"""

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from neptune_scale import Run  # pip install neptune-scale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Persistent artifact directory so the async uploader can read files safely
ARTIFACT_DIR = Path("./neptune_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)  # add to .gitignore


# -----------------------
# Small utilities
# -----------------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def save_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def is_gpt_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["gpt", "tiny-gpt2", "opt", "llama", "mpt", "gpt-neox"])


def is_bert_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["bert", "distilbert", "roberta", "albert", "electra", "mpnet"])


# -----------------------
# Activation capture
# -----------------------
class ActivationCatcher:
    def __init__(self):
        self.buffers: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _save(self, name: str):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self.buffers.setdefault(name, []).append(x.detach().float().cpu())  # [B,S,H]
        return hook

    def register_gpt_like(self, model) -> None:
        # GPT-2 style blocks
        for li, block in enumerate(model.transformer.h):
            self.hooks.append(block.mlp.c_fc.register_forward_hook(self._save(f"layer{li}/mlp_pre")))
            # Post-activation: GPT2 uses GELU module at block.mlp.act
            self.hooks.append(block.mlp.act.register_forward_hook(self._save(f"layer{li}/mlp_post")))
            self.hooks.append(block.attn.c_proj.register_forward_hook(self._save(f"layer{li}/attn_out")))

    def register_bert_like(self, model) -> None:
        """
        Normalizes hook names to mlp_pre / mlp_post / attn_out across BERT-like models.

        - DistilBertModel (encoder-only core):
            model.transformer.layer[i].ffn.lin1 / ffn.lin2 / attention.out_lin
            Activation is a function -> hook mlp_post via forward_pre_hook on lin2.
        - DistilBERT task heads (e.g., DistilBertForSequenceClassification):
            model.distilbert.transformer.layer[...]  (same submodules as above)
        - BERT/Roberta:
            encoder.layer[i].intermediate.dense (mlp_pre)
            encoder.layer[i].output.dense       (mlp_post via prehook on input)
            encoder.layer[i].attention.output.dense (attn_out)
        """
        # ---- Case 1: DistilBertModel core (no `.distilbert` wrapper) ----
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
            layers = model.transformer.layer
            # Heuristic check for DistilBERT blocks
            if len(layers) > 0 and hasattr(layers[0], "ffn") and hasattr(layers[0], "attention"):
                for li, layer in enumerate(layers):
                    # mlp_pre: output of FFN first linear
                    self.hooks.append(layer.ffn.lin1.register_forward_hook(self._save(f"layer{li}/mlp_pre")))
                    # mlp_post: input to lin2 == post-activation
                    def make_prehook(name):
                        def prehook(module, inputs):
                            x = inputs[0]
                            self.buffers.setdefault(name, []).append(x.detach().float().cpu())
                        return prehook
                    self.hooks.append(layer.ffn.lin2.register_forward_pre_hook(make_prehook(f"layer{li}/mlp_post")))
                    # attn_out: projection after self-attention
                    self.hooks.append(layer.attention.out_lin.register_forward_hook(self._save(f"layer{li}/attn_out")))
                return  # done

        # ---- Case 2: DistilBERT task heads exposing `.distilbert` ----
        if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
            for li, layer in enumerate(model.distilbert.transformer.layer):
                self.hooks.append(layer.ffn.lin1.register_forward_hook(self._save(f"layer{li}/mlp_pre")))
                def make_prehook(name):
                    def prehook(module, inputs):
                        x = inputs[0]
                        self.buffers.setdefault(name, []).append(x.detach().float().cpu())
                    return prehook
                self.hooks.append(layer.ffn.lin2.register_forward_pre_hook(make_prehook(f"layer{li}/mlp_post")))
                self.hooks.append(layer.attention.out_lin.register_forward_hook(self._save(f"layer{li}/attn_out")))
            return  # done

        # ---- Case 3: BERT / Roberta ----
        enc = getattr(model, "bert", None) or getattr(model, "roberta", None)
        if enc is not None and hasattr(enc.encoder, "layer"):
            for li, layer in enumerate(enc.encoder.layer):
                inter = layer.intermediate.dense        # mlp_pre
                outp = layer.output.dense               # mlp_post via prehook on input
                self.hooks.append(inter.register_forward_hook(self._save(f"layer{li}/mlp_pre")))
                def make_prehook(name):
                    def prehook(module, inputs):
                        x = inputs[0]
                        self.buffers.setdefault(name, []).append(x.detach().float().cpu())
                    return prehook
                self.hooks.append(outp.register_forward_pre_hook(make_prehook(f"layer{li}/mlp_post")))
                self.hooks.append(layer.attention.output.dense.register_forward_hook(self._save(f"layer{li}/attn_out")))
            return  # done

        # If none matched:
        raise ValueError("Unsupported encoder architecture for hooks (neither DistilBERT nor BERT/Roberta layout detected).")


    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass


# -----------------------
# Masked stats (ignore pads)
# -----------------------
def masked_mean_abs(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = acts.abs() * mask
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
    return masked.sum(dim=(0, 1)) / denom  # [H]


def masked_fire_freq(acts: torch.Tensor, mask: torch.Tensor, thr: float) -> torch.Tensor:
    fired = (acts.abs() >= thr).float() * mask
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
    return fired.sum(dim=(0, 1)) / denom  # [H]


def dead_ratio_per_layer(buffers: Dict[str, torch.Tensor], mask: torch.Tensor, eps: float) -> Dict[str, float]:
    out = {}
    for key, acts in buffers.items():
        acts = torch.cat(acts, dim=0) if isinstance(acts, list) else acts  # [N,S,H]
        mean_abs = masked_mean_abs(acts, mask)  # [H]
        dead = (mean_abs <= eps).float().mean().item()
        out[key] = dead
    return out


# -----------------------
# Neptune logging helpers
# -----------------------
def parse_layer_and_kind(name: str) -> Tuple[int, str]:
    # name like "layer12/mlp_post" -> (12, "mlp_post")
    parts = name.split("/", 2)
    layer = int(parts[0].replace("layer", "")) if parts and parts[0].startswith("layer") else -1
    kind = parts[1] if len(parts) > 1 else "unknown"
    return layer, kind


def log_dead_ratios_series(run: "Run", ratios: Dict[str, float]) -> None:
    by_kind: Dict[str, List[Tuple[int, float]]] = {"mlp_post": [], "mlp_pre": [], "attn_out": []}
    for name, val in ratios.items():
        layer, kind = parse_layer_and_kind(name)
        if kind in by_kind and layer >= 0:
            by_kind[kind].append((layer, float(val)))
    for kind, items in by_kind.items():
        items.sort(key=lambda x: x[0])
        for layer, val in items:
            run.log_metrics(data={f"dead_ratio/{kind}": val}, step=layer)


def log_histogram_file_series(run: "Run", layer: int, tag: str, sample: np.ndarray) -> None:
    fig = plt.figure()
    plt.hist(sample, bins=100)
    plt.title(f"Activation Histogram: layer{layer}/{tag}")
    plt.xlabel("Activation value")
    plt.ylabel("Count")
    png = fig_to_png_bytes(fig)
    plt.close(fig)

    key = f"histograms/layer{layer}/{tag}"
    path = ARTIFACT_DIR / f"{key}.png"
    local = save_bytes(path, png)
    run.log_files(files={key: local}, step=layer)


def log_heatmap_file_series(run: "Run", layer: int, tag: str, freq_sorted: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 2))
    im = plt.imshow(freq_sorted.reshape(1, -1), aspect="auto")
    plt.yticks([0], [f"layer{layer}/{tag}"])
    plt.xlabel("Neuron (sorted)")
    plt.title(f"Activation frequency heatmap: layer{layer}/{tag}\n(0=never, 1=always)")
    cbar = plt.colorbar(im)
    cbar.set_label("Act. freq. (0–1)", rotation=0, labelpad=50, ha="left")
    png = fig_to_png_bytes(fig)
    plt.close(fig)

    key = f"heatmaps/layer{layer}/{tag}"
    path = ARTIFACT_DIR / f"{key}.png"
    local = save_bytes(path, png)
    run.log_files(files={key: local}, step=layer)


# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", type=str, required=True)
    p.add_argument("--model", type=str, default="distilgpt2",
                   help="Prefer safetensors-enabled repos: 'distilgpt2', 'gpt2', 'prajjwal1/bert-tiny', 'distilbert-base-uncased'")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eps-dead", type=float, default=1e-4)
    p.add_argument("--fire-threshold", type=float, default=1e-3)
    p.add_argument("--max-hist-layers", type=int, default=6)
    p.add_argument("--max-heatmap-neurons", type=int, default=512)
    args = p.parse_args()

    run = Run(experiment_name=args.experiment_name)
    run.log_configs(vars(args))

    # Tokenizer + padding
    tok = AutoTokenizer.from_pretrained(args.model)
    added_pad = False
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            added_pad = True
    if is_gpt_like(args.model):
        tok.padding_side = "left"

    # Model (force safetensors to avoid torch.load on .bin)
    if is_gpt_like(args.model):
        model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True).to(DEVICE).eval()
    elif is_bert_like(args.model):
        model = AutoModel.from_pretrained(args.model, use_safetensors=True).to(DEVICE).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True).to(DEVICE).eval()

    if added_pad and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))

    # Hooks
    catcher = ActivationCatcher()
    if is_gpt_like(args.model):
        catcher.register_gpt_like(model)
    elif is_bert_like(args.model):
        catcher.register_bert_like(model)
    else:
        catcher.register_gpt_like(model)

    # Data
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = [ex["text"] for ex in ds if isinstance(ex.get("text"), str) and ex["text"].strip()]
    texts = texts[: args.batch_size * args.batches]

    # Forward passes
    mask_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in chunks(texts, args.batch_size):
            enc = tok(batch, truncation=True, padding="max_length", max_length=args.max_tokens, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in enc.items()}
            _ = model(**inputs)
            mask_batches.append(inputs["attention_mask"].detach().cpu())

    # Consolidate buffers/mask
    pad_mask = torch.cat(mask_batches, dim=0).unsqueeze(-1).float()  # [N,S,1]
    buffers: Dict[str, torch.Tensor] = {}
    for k, lst in catcher.buffers.items():
        try:
            buffers[k] = torch.cat(lst, dim=0)  # [N,S,H]
        except Exception:
            continue

    # --- Scalars (as 3 clean series) ---
    ratios = dead_ratio_per_layer(buffers, pad_mask, eps=args.eps_dead)
    log_dead_ratios_series(run, ratios)

    # --- File series per layer ---
    # 1) Histograms: sample real-token activations and log them (limited to a few layers for speed)
    # 2) Heatmaps: sorted activation frequency per neuron (limit to max_heatmap_neurons)
    plotted = 0
    for key in sorted(buffers.keys()):
        layer, kind = parse_layer_and_kind(key)
        acts = buffers[key]  # [N,S,H]
        N, S, H = acts.shape
        mask = pad_mask.expand(N, S, H)
        # Only real tokens
        valid = acts[mask.bool()]
        if valid.numel() == 0:
            continue

        # HISTOGRAMS (limit number of layers for speed)
        if plotted < args.max_hist_layers:
            flat = valid.view(-1)
            take = min(50000, flat.shape[0])
            idx = torch.randint(0, flat.shape[0], (take,))
            sample = flat[idx].numpy()
            log_histogram_file_series(run, layer, kind, sample)
            plotted += 1

        # HEATMAPS
        fired = (acts.abs() >= args.fire_threshold).float() * mask
        denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
        freq = (fired.sum(dim=(0, 1)) / denom).numpy()  # [H]
        order = np.argsort(freq)
        freq_sorted = freq[order][: args.max_heatmap_neurons]
        log_heatmap_file_series(run, layer, kind, freq_sorted)

    # Cleanup
    try:
        catcher.close()
    finally:
        try:
            run.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
