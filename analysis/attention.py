"""
analysis/attention.py
Attention weight extraction and visualisation.

The v1 hook approach silently returned stale tensors because PyTorch's
TransformerEncoderLayer only populates attn_output_weights when
need_weights=True is explicitly passed to MHA.forward().

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import config
from utils import device

# Extraction

def get_attention_weights(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """
    Run one forward pass and return attention weights.

    Returns:
        np.ndarray of shape [B, nhead, T+1, T+1]
    """
    store = {}
    mha = model.encoder.layers[0].self_attn
    original_forward = mha.forward

    def patched_forward(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False # keep per-head weights
        out, weights = original_forward(query, key, value, **kwargs)
        store["weights"] = weights.detach().cpu()
        return out, weights

    mha.forward = patched_forward
    model.eval()
    with torch.no_grad():
        model(x.to(device))
    mha.forward = original_forward

    return store["weights"].numpy() # [B, nhead, T+1, T+1]


def plot_attention_heatmap(
    model: nn.Module,
    task_fn,
    seq_len: int,
    task_name: str,
    n_samples: int = 64,
    save_dir: str | None = None):
    X, _ = task_fn(seq_len, n_samples)
    weights = get_attention_weights(model, X) # [B, nhead, T+1, T+1]

    nhead = weights.shape[1]
    ncols = nhead + 1 # one per head + mean
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), squeeze=False)
    axes = axes[0]

    tick_positions = [0, seq_len // 2, seq_len]
    tick_labels = ["CLS", str(seq_len // 2), str(seq_len)]

    for h in range(nhead):
        head_attn = weights.mean(axis=0)[h] # [T+1, T+1]
        im = axes[h].imshow(head_attn, aspect="auto", cmap="Blues", vmin=0)
        axes[h].set_title(f"Head {h}", fontsize=11)
        axes[h].set_xlabel("Key position")
        axes[h].set_ylabel("Query position")
        axes[h].set_xticks(tick_positions)
        axes[h].set_xticklabels(tick_labels)
        axes[h].set_yticks(tick_positions)
        axes[h].set_yticklabels(tick_labels)
        plt.colorbar(im, ax=axes[h])

    mean_attn = weights.mean(axis=(0, 1)) # [T+1, T+1]
    im = axes[nhead].imshow(mean_attn, aspect="auto", cmap="Blues", vmin=0)
    axes[nhead].set_title("Mean (all heads)", fontsize=11)
    axes[nhead].set_xlabel("Key position")
    axes[nhead].set_ylabel("Query position")
    axes[nhead].set_xticks(tick_positions)
    axes[nhead].set_xticklabels(tick_labels)
    axes[nhead].set_yticks(tick_positions)
    axes[nhead].set_yticklabels(tick_labels)
    plt.colorbar(im, ax=axes[nhead])

    fig.suptitle(
        f"{task_name} - Attention weights  (L={seq_len})", fontsize=13
    )
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"attn_{task_name.replace('*','s').replace(' ','-')}_L{seq_len}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)

    plt.show()


def run_attention_analysis(
    results_models: dict,
    task_registry: dict,
    pe: str = "learned",
    lengths_to_plot: list[int] | None = None,
    save_dir: str | None = None):
    """
    Plot attention heatmaps for each task at in-distribution and OOD lengths.

    Args:
        results_models: {task_name: trained_model} one model per task
        task_registry: {task_name: generator_fn}
        pe: which PE variant to label plots with
        lengths_to_plot: defaults to [train_len, 30, 100]
    """
    lengths_to_plot = lengths_to_plot or [config.TRAIN_LEN, 30, 100]

    for task_name, model in results_models.items():
        task_fn = task_registry[task_name]
        print(f"\n--- Attention: {task_name} ({pe} PE) ---")
        for L in lengths_to_plot:
            plot_attention_heatmap(
                model, task_fn, L, task_name, save_dir=save_dir
            )
