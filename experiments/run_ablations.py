"""
experiments/run_ablations.py
Ablation: CLS readout vs last-token readout on First-Last.

Answers: how much of v1's training instability came from the last-token
readout giving the model a positional shortcut?

"""

import os
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.generators import generate_first_last
from model.transformer import MinimalTransformer
from training.trainer import train_model
from training.evaluate import evaluate
from utils import set_seed, device


# Last-token

class MinimalTransformerLastToken(MinimalTransformer):
    """
    Identical to MinimalTransformer except it reads from the final
    sequence position instead of the CLS token.
    Used only in the ablation experiment.
    """

    def forward(self, x):
        import torch
        B, L = x.shape
        tok = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)

        if self.pos_encoding_type == "learned":
            tok = tok + self.pos_embedding[: L + 1]
        elif self.pos_encoding_type == "sin":
            tok = tok + self.pos_embedding[:, : L + 1, :]

        h = self.encoder(tok)
        return self.classifier(h[:, -1, :])   # last token, not CLS


# Runner

def run_readout_ablation(
    pe: str = "learned",
    seeds: list[int] = config.SEEDS,
    train_len: int = config.TRAIN_LEN,
    test_lens: list[int] = config.TEST_LENS,
) -> dict:
    """
    Train both readout variants on First-Last across all seeds.
    Returns ablation results dict.
    """
    print(f"\nAblation: CLS vs last-token  |  PE={pe}")

    X_train, y_train = generate_first_last(train_len, config.N_TRAIN)

    cls_accs = defaultdict(list)
    last_accs = defaultdict(list)
    cls_train_accs  = []
    last_train_accs = []

    # Multi seed check
    for seed in seeds:
        set_seed(seed)

        m_cls = MinimalTransformer(pos_encoding=pe)
        m_last = MinimalTransformerLastToken(pos_encoding=pe)

        h_cls = train_model(m_cls,  X_train, y_train)
        h_last = train_model(m_last, X_train, y_train)

        cls_train_accs.append(h_cls["train_acc"][-1])
        last_train_accs.append(h_last["train_acc"][-1])

        for L in test_lens:
            X_t, y_t = generate_first_last(L, config.N_TEST)
            cls_accs[L].append(evaluate(m_cls,  X_t, y_t))
            last_accs[L].append(evaluate(m_last, X_t, y_t))

        print(".", end="", flush=True)

    print()
    print(f"CLS  train acc: {np.mean(cls_train_accs):.3f} "
          f"+/- {np.std(cls_train_accs):.3f}")
    print(f"Last train acc: {np.mean(last_train_accs):.3f} "
          f"+/- {np.std(last_train_accs):.3f}")

    return {
        "cls_accs": dict(cls_accs),
        "last_accs": dict(last_accs),
        "cls_train_accs": cls_train_accs,
        "last_train_accs": last_train_accs,
    }


def plot_readout_ablation(
    ablation: dict,
    pe: str = "learned",
    train_len: int = config.TRAIN_LEN,
    save_dir: str | None = None,
):
    lengths = sorted(ablation["cls_accs"].keys())
    cls_means = np.array([np.mean(ablation["cls_accs"][L])  for L in lengths])
    last_means = np.array([np.mean(ablation["last_accs"][L]) for L in lengths])
    cls_stds = np.array([np.std(ablation["cls_accs"][L])   for L in lengths])
    last_stds = np.array([np.std(ablation["last_accs"][L])  for L in lengths])

    plt.figure(figsize=(7, 4))
    plt.axvline(train_len, color="gray", linestyle="--",
                linewidth=0.8, alpha=0.6, label="Train length")
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

    for means, stds, color, label in [
        (cls_means,  cls_stds,  "#3266ad", "CLS readout"),
        (last_means, last_stds, "#c44b4b", "Last-token readout"),
    ]:
        plt.plot(lengths, means, marker="o", color=color, label=label, linewidth=2)
        plt.fill_between(lengths, means - stds, means + stds,
                         alpha=0.15, color=color)

    plt.title(f"First-Last: CLS vs last-token readout ({pe} PE)")
    plt.xlabel("Sequence length")
    plt.ylabel("Accuracy")
    plt.ylim(0.40, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"ablation_readout_{pe}.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    for pe in ["learned", "sin"]:
        ablation = run_readout_ablation(pe=pe)
        plot_readout_ablation(ablation, pe=pe, save_dir=config.PLOTS_DIR)
