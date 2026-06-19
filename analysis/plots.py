"""
analysis/plots.py
All plotting functions for the length-generalisation results.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
from training.evaluate import detect_grokking


PE_COLORS = {
    "learned": "#3266ad",
    "sin": "#e07b39",
    "none": "#6a994e",
}
PE_LABELS = {
    "learned": "Learned PE",
    "sin": "Sinusoidal PE",
    "none": "No PE",
}


def plot_length_generalization(
    results: dict,
    tasks: list[str] | None = None,
    pe_types: list[str] | None = None,
    train_len: int = config.TRAIN_LEN,
    save_dir: str | None = None,
    tasks_per_fig: int = 2,
):
    """
    At most `tasks_per_fig` subplots per figure. Each PE type = one line + shaded +/- std band.
    Vertical dashed line = training length.
    Horizontal dotted line = chance level (0.5).
    """
    tasks    = tasks    or list(results.keys())
    pe_types = pe_types or list(PE_COLORS.keys())
    n_seeds  = len(config.SEEDS)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for fig_idx, chunk_start in enumerate(range(0, len(tasks), tasks_per_fig)):
        chunk = tasks[chunk_start:chunk_start + tasks_per_fig]
        ncols = len(chunk)

        fig, axes = plt.subplots(
            1, ncols,
            figsize=(7 * ncols, 5),
            squeeze=False,
        )

        for col, task in enumerate(chunk):
            ax = axes[0][col]

            ax.axvline(train_len, color="gray", linestyle="--",
                       linewidth=0.9, alpha=0.55, label="Train length")
            ax.axhline(0.5, color="gray", linestyle=":",
                       linewidth=0.7, alpha=0.4)

            for pe in pe_types:
                if pe not in results.get(task, {}):
                    continue
                data = results[task][pe]["accs"]
                lengths = sorted(data.keys())
                means = np.array([np.mean(data[L]) for L in lengths])
                stds = np.array([np.std(data[L])  for L in lengths])
                color = PE_COLORS.get(pe, "black")

                ax.plot(lengths, means, marker="o", color=color, label=PE_LABELS.get(pe, pe), linewidth=2, markersize=5)
                ax.fill_between(
                    lengths, means - stds, means + stds,
                    alpha=0.15, color=color,
                )

            ax.set_title(task, fontsize=13)
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.40, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Length generalisation  "
            f"(train L={train_len},  {n_seeds} seeds,  mean ± std)",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()

        if save_dir:
            fname = f"length_generalization_{fig_idx + 1:02d}.png"
            plt.savefig(
                os.path.join(save_dir, fname),
                dpi=150, bbox_inches="tight",
            )
        plt.show()


def plot_training_curves(
    results: dict,
    tasks: list[str] | None = None,
    pe_types: list[str] | None = None,
    save_dir: str | None = None,
    pes_per_fig: int = 2,
):
    """
    For each task: at most `pes_per_fig` PE subplots per figure.
    Helps spot bimodal convergence, grokking, and training failures.
    """
    tasks    = tasks    or list(results.keys())
    pe_types = pe_types or list(PE_COLORS.keys())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    task_slug = lambda t: t.replace("*", "s").replace(" ", "-")

    for task in tasks:
        valid_pes = [pe for pe in pe_types if pe in results.get(task, {})]
        if not valid_pes:
            continue

        for fig_idx, chunk_start in enumerate(range(0, len(valid_pes), pes_per_fig)):
            chunk_pes = valid_pes[chunk_start:chunk_start + pes_per_fig]
            ncols = len(chunk_pes)

            fig, axes = plt.subplots(
                1, ncols,
                figsize=(7 * ncols, 5),
                squeeze=False,
            )

            for col, pe in enumerate(chunk_pes):
                ax    = axes[0][col]
                hists = results[task][pe]["histories"]

                for i, hist in enumerate(hists):
                    ax.plot(
                        hist["train_acc"],
                        alpha=0.75,
                        linewidth=1.5,
                        label=f"seed {config.SEEDS[i]}",
                    )

                ax.axhline(
                    config.MIN_TRAIN_ACC,
                    linestyle="--", color="gray",
                    linewidth=0.9, alpha=0.6,
                    label=f"{config.MIN_TRAIN_ACC:.0%} threshold",
                )
                ax.set_title(f"{task}  |  {PE_LABELS.get(pe, pe)}", fontsize=11)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Train accuracy")
                ax.set_ylim(0.40, 1.05)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_dir:
                fname = f"training_curves_{task_slug(task)}_{fig_idx + 1:02d}.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=150)
            plt.show()

def print_summary_table(
    results: dict,
    test_lens: list[int] = config.TEST_LENS):
    focal_lens = [config.TRAIN_LEN, max(test_lens)]

    header = f"{'Task':<15} {'PE':<10} {'TrainAcc':>12}"
    for L in focal_lens:
        header += f"  {'L='+str(L):>9}"
    header += f"  {'Grokking':>8}  {'Converged':>9}"
    print(header)
    print("-" * len(header))

    for task in results:
        for pe in results[task]:
            d = results[task][pe]

            tr_mean = np.mean(d["train_accs"])
            tr_std  = np.std(d["train_accs"])
            converged = np.mean([a > 0.95 for a in d["train_accs"]]) >= 0.6
            grokking  = any(detect_grokking(h) for h in d["histories"])

            row = (f"{task:<15} {pe:<10} "
                   f"{tr_mean:.3f}±{tr_std:.3f}  ")
            for L in focal_lens:
                if L in d["accs"]:
                    m = np.mean(d["accs"][L])
                    s = np.std(d["accs"][L])
                    row += f"  {m:.2f}±{s:.2f}"
                else:
                    row += f"  {'N/A':>9}"

            row += f" {'yes' if grokking  else 'no':>8}"
            row += f" {'yes' if converged else 'NO':>9}"
            print(row)
        print()
