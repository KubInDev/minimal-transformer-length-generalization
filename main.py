"""
main.py
"""

import argparse
import os
import config
from data.generators import TASK_REGISTRY
from experiments.run_length_gen import run_all
from experiments.run_ablations import run_readout_ablation, plot_readout_ablation
from analysis.plots import (
    plot_length_generalization,
    plot_training_curves,
    print_summary_table,
)
from analysis.attention import plot_attention_heatmap
from model.transformer import MinimalTransformer
from training.trainer import train_model
from utils import set_seed, load_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal transformer - length generalisation experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--tasks", nargs="+", default=config.TASKS)
    p.add_argument("--pe", nargs="+", default=config.PE_TYPES, dest="pe_types")
    p.add_argument("--seeds", type=int, default=config.N_SEEDS)

    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--plot-only", action="store_true")
    modes.add_argument("--ablations-only", action="store_true")
    modes.add_argument("--attention-only",  action="store_true")
    modes.add_argument("--attention-extended", action="store_true")
    modes.add_argument("--multi-length", action="store_true")
    modes.add_argument("--cot", action="store_true")

    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def mode_plot_only(args):
    results = load_results()
    plot_length_generalization(results, tasks=args.tasks,
                               pe_types=args.pe_types, save_dir=config.PLOTS_DIR)
    plot_training_curves(results, tasks=args.tasks,
                         pe_types=args.pe_types, save_dir=config.PLOTS_DIR)
    print_summary_table(results)


def mode_ablations(args):
    seeds = config.SEEDS[: args.seeds]
    for pe in args.pe_types:
        if pe == "none":
            continue
        ablation = run_readout_ablation(pe=pe, seeds=seeds)
        plot_readout_ablation(ablation, pe=pe, save_dir=config.PLOTS_DIR)


def mode_attention_basic(args):
    pe = args.pe_types[0]
    for task_name in args.tasks:
        if task_name not in TASK_REGISTRY:
            continue
        task_fn = TASK_REGISTRY[task_name]
        print(f"\n--- Attention: {task_name}  ({pe} PE) ---")
        set_seed(42)
        model = MinimalTransformer(pos_encoding=pe)
        X_tr, y_tr = task_fn(config.TRAIN_LEN, config.N_TRAIN)
        train_model(model, X_tr, y_tr, verbose=True)
        for L in [30, config.TRAIN_LEN, 100]:
            plot_attention_heatmap(model, task_fn, L, task_name,
                                   save_dir=config.PLOTS_DIR)


def mode_attention_extended(args):
    from analysis.attention_extended import (
        diagnose_astar_ood,
        diagnose_first_last_cls,
        confirm_contains11_mechanism,
    )
    for pe in [p for p in args.pe_types if p != "none"]:
        print(f"\n-- PE = {pe} --")
        if "A*B*" in args.tasks:
            diagnose_astar_ood(pe=pe)
        if "First-Last" in args.tasks:
            diagnose_first_last_cls(pe=pe)
    if "Contains-11" in args.tasks:
        confirm_contains11_mechanism(pe="none")


def mode_multi_length(args):
    from experiments.run_multi_length_train import run_multi_length
    seeds = config.SEEDS[: args.seeds]
    target = [t for t in args.tasks if t in ("First-Last", "A*B*")]
    if not target:
        print("Needs First-Last or A*B*.")
        return
    run_multi_length(
        tasks = target,
        pe_types = [p for p in args.pe_types if p != "none"],
        seeds = seeds,
    )

def mode_cot(args):
    from experiments.run_cot import (
        run_cot_experiment, plot_cot_results, print_cot_table
    )
    pe = args.pe_types[0] if args.pe_types else "learned"
    seeds = config.SEEDS[: args.seeds]
    results = run_cot_experiment(pe=pe, seeds=seeds)
    plot_cot_results(results, pe=pe, save_dir=config.PLOTS_DIR)
    print_cot_table(results)

def mode_full_run(args):
    seeds = config.SEEDS[: args.seeds]
    save  = not args.no_save

    print(f"\nFull run  tasks={args.tasks}  pe={args.pe_types}  seeds={seeds}\n")

    results = run_all(tasks=args.tasks, pe_types=args.pe_types,
                      seeds=seeds, save=save)

    plot_length_generalization(results, tasks=args.tasks,
                               pe_types=args.pe_types, save_dir=config.PLOTS_DIR)
    plot_training_curves(results, tasks=args.tasks,
                         pe_types=args.pe_types, save_dir=config.PLOTS_DIR)
    print_summary_table(results)

    print("\nReadout ablation")
    for pe in args.pe_types:
        if pe == "none":
            continue
        ablation = run_readout_ablation(pe=pe, seeds=seeds)
        plot_readout_ablation(ablation, pe=pe, save_dir=config.PLOTS_DIR)


def main():
    args = parse_args()
    os.makedirs(config.PLOTS_DIR,   exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if args.plot_only:
        mode_plot_only(args)
    elif args.ablations_only:
        mode_ablations(args)
    elif args.attention_only:
        mode_attention_basic(args)
    elif args.attention_extended:
        mode_attention_extended(args)
    elif args.multi_length:
        mode_multi_length(args)
    elif args.cot:
        mode_cot(args)
    else:
        mode_full_run(args)


if __name__ == "__main__":
    main()