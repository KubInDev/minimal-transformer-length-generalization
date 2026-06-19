"""
analysis/attention_extended.py
Extended attention diagnostics targeting two open questions:

  Q1. A*B* OOD decay: mechanism correct but confidence degrades,
      or length-specific shortcut?
      -> "diagonality score": mean attention weight from pos i+1 to pos i

  Q2. First-Last CLS: does CLS actually attend to the first and last token?
      -> tracks CLS->pos1 and CLS->posL weights at each test length
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from data.generators import generate_a_star_b_star, generate_first_last
from model.transformer import MinimalTransformer
from training.trainer import train_model
from training.evaluate import evaluate
from utils import set_seed, device

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def get_attention_weights(model, x):
    """Returns [B, nhead, T+1, T+1]. T+1 because CLS is prepended."""
    store = {}
    mha  = model.encoder.layers[0].self_attn
    orig = mha.forward

    def patched(query, key, value, **kwargs):
        kwargs["need_weights"]        = True
        kwargs["average_attn_weights"] = False
        out, w = orig(query, key, value, **kwargs)
        store["w"] = w.detach().cpu()
        return out, w

    mha.forward = patched
    model.eval()
    with torch.no_grad():
        model(x.to(device))
    mha.forward = orig
    return store["w"].numpy()


def mean_attn(model, task_fn, seq_len, n=128):
    X, _ = task_fn(seq_len, n)
    w = get_attention_weights(model, X)
    return w.mean(axis=(0, 1)) # [T+1, T+1]


def per_head_attn(model, task_fn, seq_len, n=128):
    X, _ = task_fn(seq_len, n)
    w = get_attention_weights(model, X)
    return w.mean(axis=0) # [nhead, T+1, T+1]


def train_single(task_fn, pe, seed=42, verbose=True):
    set_seed(seed)
    model = MinimalTransformer(pos_encoding=pe)
    X_tr, y_tr = task_fn(config.TRAIN_LEN, config.N_TRAIN)
    train_model(model, X_tr, y_tr, verbose=verbose)
    return model


# A*B* diagonality

def diagnose_astar_ood(pe="learned", seed=42):
    print(f"\n=== Q1: A*B* OOD (PE={pe}) ===")
    model   = train_single(generate_a_star_b_star, pe, seed)
    lengths = [20, 40, 50, 60, 80, 100]

    accs = {}
    for L in lengths:
        X_t, y_t = generate_a_star_b_star(L, config.N_TEST)
        accs[L] = evaluate(model, X_t, y_t)
        print(f" L={L:3d} acc={accs[L]:.4f}")

    # heatmap grid
    fig, axes = plt.subplots(2, len(lengths), figsize=(3.5 * len(lengths), 7))
    adjacent_scores = []

    for col, L in enumerate(lengths):
        attn_m = mean_attn(model, generate_a_star_b_star, L)
        attn_h = per_head_attn(model, generate_a_star_b_star, L)

        ax0 = axes[0][col]
        ax0.imshow(attn_m, aspect="auto", cmap="Blues", vmin=0)
        ax0.set_title(f"L={L}  acc={accs[L]:.2f}", fontsize=9, color="green" if accs[L] > 0.85 else "red")
        ax0.set_xticks([0, L // 2, L])
        ax0.set_xticklabels(["CLS", str(L // 2), str(L)], fontsize=7)
        ax0.set_yticks([0, L // 2, L])
        ax0.set_yticklabels(["CLS", str(L // 2), str(L)], fontsize=7)
        if col == 0: ax0.set_ylabel("Mean heads")

        ax1 = axes[1][col]
        ax1.imshow(attn_h[0], aspect="auto", cmap="Oranges", vmin=0)
        ax1.set_xticks([0, L // 2, L])
        ax1.set_xticklabels(["CLS", str(L // 2), str(L)], fontsize=7)
        ax1.set_yticks([0, L // 2, L])
        ax1.set_yticklabels(["CLS", str(L // 2), str(L)], fontsize=7)
        if col == 0: ax1.set_ylabel("Head 0")

        diag = np.mean([attn_m[i+1, i] for i in range(1, L)])
        adjacent_scores.append((L, diag))

    fig.suptitle(f"A*B* attention heatmaps ({pe} PE)", fontsize=12)
    plt.tight_layout()
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, f"attn_astar_heatmaps_{pe}.png"), dpi=150)
    plt.show()

    # diagonality score vs length
    Ls, scores = zip(*adjacent_scores)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ls, scores, marker="o", color="#3266ad", linewidth=2)
    ax.axvline(config.TRAIN_LEN, color="gray", linestyle="--", linewidth=0.9, alpha=0.6, label="Train length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Mean adjacent attention\n(query i+1 → key i)")
    ax.set_title(f"A*B* - Diagonality vs length  ({pe} PE)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"attn_astar_diagonality_{pe}.png"), dpi=150)
    plt.show()

    print("\n Adjacent attention (diagonality) scores:")
    for L, s in adjacent_scores:
        print(f" L={L:3d}  {s:.4f}  {' ' * int(s * 40)}")

    return model, accs, adjacent_scores


# first-Last CLS endpoint attention

def diagnose_first_last_cls(pe="learned", seed=42):
    print(f"\n=== Q2: First-Last CLS probe  (PE={pe}) ===")
    model   = train_single(generate_first_last, pe, seed)
    lengths = [20, 30, 40, 50, 60, 80, 100]

    accs = {}
    cls_to_first, cls_to_last = [], []

    for L in lengths:
        X_t, y_t = generate_first_last(L, config.N_TEST)
        accs[L] = evaluate(model, X_t, y_t)

        w = get_attention_weights(model, X_t)   # [B, nhead, L+1, L+1]
        cls_row = w[:, :, 0, :].mean(axis=(0, 1))   # [L+1]
        cls_to_first.append(float(cls_row[1]))
        cls_to_last.append(float(cls_row[L]))
        print(f"  L={L:3d}  acc={accs[L]:.4f}  "
              f"CLS→pos1={cls_row[1]:.4f}  CLS→posL={cls_row[L]:.4f}")

    # endpoint attention weights vs length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(lengths, cls_to_first, marker="o", color="#3266ad",
            label="CLS -> pos 1 (first token)", linewidth=2)
    ax.plot(lengths, cls_to_last,  marker="s", color="#e07b39",
            label="CLS -> pos L (last token)",  linewidth=2)
    ax.axvline(config.TRAIN_LEN, color="gray", linestyle="--",
               linewidth=0.9, alpha=0.6, label="Train length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Attention weight from CLS")
    ax.set_title(f"First-Last - CLS endpoint attention  ({pe} PE)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, None)

    # Plot 2: full CLS row at key lengths
    ax2 = axes[1]
    for L, color, ls in [
        (40, "#aaaaaa", "--"),
        (50, "#3266ad", "--"),
        (60, "#e07b39", "--"),
        (100, "#c44b4b", "--"),
    ]:
        X_t, _ = generate_first_last(L, 256)
        w = get_attention_weights(model, X_t)
        cls_row = w[:, :, 0, :].mean(axis=(0, 1))
        positions = np.linspace(0, 1, len(cls_row))
        ax2.plot(positions, cls_row, color=color, linestyle=ls,
                 linewidth=2.0 if L == config.TRAIN_LEN else 1.5,
                 label=f"L={L}  (acc={accs[L]:.2f})", alpha=0.85)

    ax2.set_xlabel("Normalised position (0=CLS, 1=last token)")
    ax2.set_ylabel("CLS attention weight")
    ax2.set_title(f"CLS attention distribution ({pe} PE)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"attn_firstlast_cls_{pe}.png"), dpi=150)
    plt.show()

    # full heatmap at train length vs OOD
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, L, title in zip(axes2,
                              [config.TRAIN_LEN, 100],
                              [f"L={config.TRAIN_LEN} (train)", "L=100 (OOD)"]):
        attn = mean_attn(model, generate_first_last, L)
        im = ax.imshow(attn, aspect="auto", cmap="Blues", vmin=0)
        ax.set_title(f"{title}  acc={accs[L]:.2f}", fontsize=11, color="green" if accs[L] > 0.8 else "red")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_xticks([0, 1, L // 2, L])
        ax.set_xticklabels(["CLS", "pos1", str(L // 2), f"pos{L}"], fontsize=8)
        ax.set_yticks([0, 1, L // 2, L])
        ax.set_yticklabels(["CLS", "pos1", str(L // 2), f"pos{L}"], fontsize=8)
        plt.colorbar(im, ax=ax)

    fig2.suptitle(f"First-Last heatmap  ({pe} PE)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"attn_firstlast_heatmap_{pe}.png"), dpi=150)
    plt.show()

    return model, accs, cls_to_first, cls_to_last


# Contains-11 mechanism confirmation

def confirm_contains11_mechanism(pe="none", seed=42):
    from data.generators import generate_contains_11
    print(f"\n=== Q3: Contains-11 mechanism  (PE={pe}) ===")
    model   = train_single(generate_contains_11, pe, seed)
    lengths = [20, 50, 100]

    fig, axes = plt.subplots(1, len(lengths),
                              figsize=(5 * len(lengths), 4))
    for col, L in enumerate(lengths):
        X_t, y_t = generate_contains_11(L, config.N_TEST)
        acc  = evaluate(model, X_t, y_t)
        attn = mean_attn(model, generate_contains_11, L)
        diag = np.mean([attn[i+1, i] for i in range(1, L)])
        print(f"  L={L:3d}  acc={acc:.4f}  diagonality={diag:.4f}")

        ax = axes[col]
        im = ax.imshow(attn, aspect="auto", cmap="Greens", vmin=0)
        ax.set_title(f"L={L}  acc={acc:.4f}", fontsize=11, color="green")
        ax.set_xlabel("Key position")
        if col == 0: ax.set_ylabel("Query position")
        ax.set_xticks([0, L // 2, L])
        ax.set_xticklabels(["CLS", str(L // 2), str(L)], fontsize=8)
        ax.set_yticks([0, L // 2, L])
        ax.set_yticklabels(["CLS", str(L // 2), str(L)], fontsize=8)
        plt.colorbar(im, ax=ax)

    fig.suptitle(f"Contains-11 attention  (PE={pe})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, f"attn_contains11_{pe}.png"), dpi=150)
    plt.show()
    return model

if __name__ == "__main__":
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    for pe in ["learned", "sin"]:
        diagnose_astar_ood(pe=pe)
        diagnose_first_last_cls(pe=pe)
    confirm_contains11_mechanism(pe="none")