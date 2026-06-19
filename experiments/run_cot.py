"""
experiments/run_cot.py
CoT vs Standard model comparison on First-Last with learned PE.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.generators import generate_first_last
from model.transformer import MinimalTransformer
from training.evaluate import evaluate
from utils import set_seed, device


# CoT model

class MinimalTransformerCoT(MinimalTransformer):
    """
    Same as MinimalTransformer but with three auxiliary prediction heads.
    Training loss = CE(main, y) + CE(cot_first, first_tok)
                  + CE(cot_last, last_tok) + CE(cot_equal, y)
    Inference uses only the main classifier head.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d = self.d_model
        self.cot_first = nn.Linear(d, 2) # predicts seq[0]
        self.cot_last  = nn.Linear(d, 2) # predicts seq[-1]
        self.cot_equal = nn.Linear(d, 2) # predicts seq[0]==seq[-1]

    def forward_cot(self, x):
        """
        Returns (main_logits, first_logits, last_logits, equal_logits).
        Call this during training. Call forward() during evaluation.
        """
        B, L = x.shape
        tok = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)

        if self.pos_encoding_type == "learned":
            tok = tok + self.pos_embedding[: L + 1]
        elif self.pos_encoding_type == "sin":
            tok = tok + self.pos_embedding[:, : L + 1, :]

        h = self.encoder(tok) # [B, L+1, d]
        cls_repr = h[:, 0, :] # [B, d]

        return (
            self.classifier(cls_repr),
            self.cot_first(cls_repr),
            self.cot_last(cls_repr),
            self.cot_equal(cls_repr),
        )


# Training loop

def train_cot(
    model,
    X_train,
    y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    lr=config.LR,
    patience=config.PATIENCE,
    min_train_acc=config.MIN_TRAIN_ACC,
    warmup_epochs=config.WARMUP_EPOCHS,
    verbose=False,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    history = {"loss": [], "train_acc": []}
    best_acc = 0.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)


            first_tok = xb[:, 0] # [B]
            last_tok  = xb[:, -1]

            main_l, first_l, last_l, equal_l = model.forward_cot(xb)

            loss = (
                loss_fn(main_l, yb)
              + loss_fn(first_l, first_tok)
              + loss_fn(last_l, last_tok)
              + loss_fn(equal_l, yb)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (main_l.argmax(1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()

        avg_loss  = total_loss / total
        train_acc = correct / total
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        if verbose and epoch % 10 == 0:
            print(f" epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.4f}")

        if train_acc >= min_train_acc:
            if verbose:
                print(f" -> converged at epoch {epoch}")
            break

        if epoch >= warmup_epochs:
            if train_acc - best_acc > 0.005:
                best_acc   = train_acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f" -> plateau stop at epoch {epoch}")
                break
        else:
            best_acc = max(best_acc, train_acc)

    return history


# Runner

def run_cot_experiment(
    pe = "learned",
    seeds = config.SEEDS,
    train_len = config.TRAIN_LEN,
    test_lens = config.TEST_LENS,
):
    print(f"\nCoT vs Standard  |  First-Last  |  PE={pe}")

    X_train, y_train = generate_first_last(train_len, config.N_TRAIN)

    std_accs  = defaultdict(list)
    cot_accs  = defaultdict(list)
    std_hists = []
    cot_hists = []

    for seed in seeds:
        # Standard
        set_seed(seed)
        from training.trainer import train_model
        m_std  = MinimalTransformer(pos_encoding=pe)
        h_std  = train_model(m_std, X_train, y_train, verbose=False)
        std_hists.append(h_std)

        # CoT
        set_seed(seed)
        m_cot  = MinimalTransformerCoT(pos_encoding=pe)
        h_cot  = train_cot(m_cot, X_train, y_train, verbose=False)
        cot_hists.append(h_cot)

        for L in test_lens:
            X_t, y_t = generate_first_last(L, config.N_TEST)
            std_accs[L].append(evaluate(m_std, X_t, y_t))
            cot_accs[L].append(evaluate(m_cot, X_t, y_t))

        print(".", end="", flush=True)

    print(f"\n  Standard train acc: "
          f"{np.mean([h['train_acc'][-1] for h in std_hists]):.3f} ± "
          f"{np.std([h['train_acc'][-1] for h in std_hists]):.3f}")
    print(f" CoT train acc: "
          f"{np.mean([h['train_acc'][-1] for h in cot_hists]):.3f} ± "
          f"{np.std([h['train_acc'][-1] for h in cot_hists]):.3f}")

    return {
        "std_accs":  dict(std_accs),
        "cot_accs":  dict(cot_accs),
        "std_hists": std_hists,
        "cot_hists": cot_hists,
    }

def plot_cot_results(results, pe="learned", save_dir=None):
    std_accs  = results["std_accs"]
    cot_accs  = results["cot_accs"]
    std_hists = results["std_hists"]
    cot_hists = results["cot_hists"]
    lengths   = sorted(std_accs.keys())

    STD_COLOR = "#3266ad"
    COT_COLOR = "#e07b39"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: length generalisation
    ax = axes[0]
    ax.axvline(config.TRAIN_LEN, color="gray", linestyle="--",
               linewidth=0.9, alpha=0.55, label="Train length")
    ax.axhline(0.5, color="gray", linestyle=":",
               linewidth=0.7, alpha=0.4)

    for label, accs, color, ls in [
        ("Standard", std_accs, STD_COLOR, "-"),
        ("CoT",      cot_accs, COT_COLOR, "--"),
    ]:
        means = np.array([np.mean(accs[L]) for L in lengths])
        stds  = np.array([np.std(accs[L])  for L in lengths])
        ax.plot(lengths, means, marker="o", color=color,
                linestyle=ls, linewidth=2, markersize=5, label=label)
        ax.fill_between(lengths, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_title(f"First-Last: Standard vs CoT  ({pe} PE)", fontsize=12)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.40, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: training curves
    ax2 = axes[1]
    ax2.axhline(config.MIN_TRAIN_ACC, color="gray", linestyle="--",
                linewidth=0.9, alpha=0.6, label="99% threshold")

    for i, (h_std, h_cot) in enumerate(zip(std_hists, cot_hists)):
        seed = config.SEEDS[i]
        ax2.plot(h_std["train_acc"], color=STD_COLOR, alpha=0.6,
                 linewidth=1.4,
                 label=f"Standard seed {seed}" if i == 0 else "_")
        ax2.plot(h_cot["train_acc"], color=COT_COLOR, alpha=0.6,
                 linewidth=1.4, linestyle="--",
                 label=f"CoT seed {seed}" if i == 0 else "_")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=STD_COLOR, linewidth=2, label="Standard (all seeds)"),
        Line2D([0], [0], color=COT_COLOR,  linewidth=2, linestyle="--",
               label="CoT (all seeds)"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
               label="99% threshold"),
    ]
    ax2.legend(handles=handles, fontsize=9)
    ax2.set_title("Training curves: Standard vs CoT", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train accuracy")
    ax2.set_ylim(0.40, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, "cot_vs_standard.png"),
            dpi=150, bbox_inches="tight",
        )
        print(f" Saved -> {os.path.join(save_dir, 'cot_vs_standard.png')}")
    plt.show()


def print_cot_table(results):
    lengths = sorted(results["std_accs"].keys())
    print(f"\n{'Length':<10} {'Standard':>14} {'CoT':>14}")
    print("-" * 40)
    for L in lengths:
        sm = np.mean(results["std_accs"][L])
        ss = np.std(results["std_accs"][L])
        cm = np.mean(results["cot_accs"][L])
        cs = np.std(results["cot_accs"][L])
        print(f"L={L:<8} {sm:.3f}±{ss:.3f}   {cm:.3f}±{cs:.3f}")


if __name__ == "__main__":
    results = run_cot_experiment()
    plot_cot_results(results, save_dir=config.PLOTS_DIR)
    print_cot_table(results)