"""
experiments/run_length_gen.py
Multi-seed length-generalisation experiment runner.

Results structure (returned and saved):
    results[task_name][pe_type] = {
        "accs": {L: [acc_seed0, acc_seed1, ...]},
        "histories": [history_dict_seed0, history_dict_seed1, ...],
        "train_accs": [final_train_acc_seed0, ...],
    }
"""

import os
import sys
from collections import defaultdict

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.generators import TASK_REGISTRY
from model.transformer import MinimalTransformer
from training.trainer import train_model
from training.evaluate import evaluate
from utils import set_seed, save_results

def run_condition(
    task_name: str,
    pos_encoding: str,
    seeds: list[int] = config.SEEDS,
    train_len: int = config.TRAIN_LEN,
    test_lens: list[int] = config.TEST_LENS,
    verbose: bool = False,
) -> tuple[dict, list, list]:
    """
    Run one (task, PE) condition across all seeds.

    Returns:
        accs_by_len: {L: [acc per seed]}
        histories: [history dict per seed]
        final_train_accs: [final train acc per seed]
    """
    task_fn = TASK_REGISTRY[task_name]
    X_train, y_train = task_fn(train_len, config.N_TRAIN)

    accs_by_len = defaultdict(list)
    histories = []
    final_train_accs = []

    for seed in seeds:
        set_seed(seed)
        model = MinimalTransformer(pos_encoding=pos_encoding)
        hist = train_model(model, X_train, y_train, verbose=verbose)
        histories.append(hist)
        final_train_accs.append(hist["train_acc"][-1])

        for L in test_lens:
            X_test, y_test = task_fn(L, config.N_TEST)
            acc = evaluate(model, X_test, y_test)
            accs_by_len[L].append(acc)

        if not verbose:
            print(".", end="", flush=True)

    mean_tr = np.mean(final_train_accs)
    std_tr = np.std(final_train_accs)
    print(f" done (train acc {mean_tr:.3f} +/- {std_tr:.3f})")

    return dict(accs_by_len), histories, final_train_accs


def run_all(
    tasks: list[str] | None = None,
    pe_types: list[str] | None = None,
    seeds: list[int] = config.SEEDS,
    save: bool = True,
) -> dict:
    """
    Run all (task * PE) conditions and return the full results dict.
    Saves results to disk by default so plotting can be rerun independently.
    """
    tasks    = tasks    or config.TASKS
    pe_types = pe_types or config.PE_TYPES

    results = {}

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"  [skip] Unknown task: {task_name}")
            continue

        results[task_name] = {}
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"{'='*50}")

        for pe in pe_types:
            print(f"PE={pe:<8}", end="", flush=True)
            accs, hists, train_accs = run_condition(
                task_name, pe, seeds=seeds
            )
            results[task_name][pe] = {
                "accs": accs,
                "histories": hists,
                "train_accs": train_accs,
            }

    if save:
        save_results(results)

    return results


if __name__ == "__main__":
    run_all()
