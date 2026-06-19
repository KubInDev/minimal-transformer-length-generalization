"""
experiments/run_multi_length_train.py

Trains on sequences drawn uniformly from [TRAIN_LEN_MIN, TRAIN_LEN_MAX]
instead of a single fixed length, then evaluates OOD beyond TRAIN_LEN_MAX.
"""

import os, sys, pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config
from data.generators import TASK_REGISTRY
from model.transformer import MinimalTransformer
from training.trainer import train_model
from training.evaluate import evaluate
from utils import set_seed, device

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRAIN_LEN_MIN = 10
TRAIN_LEN_MAX = 50
TEST_LENS = [20, 40, 50, 60, 80, 100, 120, 150]
N_TRAIN = 20_000
N_TEST = 3_000

def generate_multi_length(task_fn, n_samples, len_min=TRAIN_LEN_MIN, len_max=TRAIN_LEN_MAX):
    import random
    samples = []
    for _ in range(n_samples):
        L = random.randint(len_min, len_max)
        X_b, y_b = task_fn(L, 1)
        samples.append((X_b[0], y_b[0]))
    return samples


def collate_variable_length(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    X_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        X_pad[i, :x.size(0)] = x
    return X_pad, torch.stack(ys)


def train_multi(model, samples, epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE, lr=config.LR,
                patience=config.PATIENCE,
                min_train_acc=config.MIN_TRAIN_ACC,
                verbose=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        samples, batch_size=batch_size, shuffle=True,
        collate_fn=collate_variable_length,
    )

    history = {"loss": [], "train_acc": []}
    best_acc = 0.0
    no_imp = 0

    for epoch in range(epochs):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        avg_loss  = tot_loss / total
        train_acc = correct / total
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        if verbose and epoch % 10 == 0:
            print(f" epoch {epoch:3d} loss={avg_loss:.4f} acc={train_acc:.4f}")

        if train_acc >= min_train_acc:
            if verbose: print(f" converged epoch {epoch}")
            break
        if epoch >= config.WARMUP_EPOCHS:
            if train_acc - best_acc > 0.005:
                best_acc = train_acc; no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                if verbose: print(f" plateau stop epoch {epoch}")
                break
        else:
            best_acc = max(best_acc, train_acc)
    return history


def run_multi_length(tasks=None, pe_types=None, seeds=config.SEEDS, save=True):
    tasks = tasks or ["First-Last", "A*B*"]
    pe_types = pe_types or ["learned", "sin"]

    multi_res = {}
    single_res = {}

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            continue
        task_fn = TASK_REGISTRY[task_name]
        multi_res[task_name] = {}
        single_res[task_name] = {}

        print(f"\n{'='*55}\n  Task: {task_name}\n{'='*55}")

        for pe in pe_types:
            max_len_needed = max(TEST_LENS) + 10

            # Multi length
            print(f" [multi] PE={pe:<8}", end="", flush=True)
            accs_m = defaultdict(list); hists_m = []; tr_m = []
            for seed in seeds:
                set_seed(seed)
                samples = generate_multi_length(task_fn, N_TRAIN)
                model = MinimalTransformer(pos_encoding=pe,
                                             max_len=max_len_needed)
                hist = train_multi(model, samples)
                hists_m.append(hist); tr_m.append(hist["train_acc"][-1])
                for L in TEST_LENS:
                    X_t, y_t = task_fn(L, N_TEST)
                    accs_m[L].append(evaluate(model, X_t, y_t))
                print(".", end="", flush=True)
            print(f" done (train {np.mean(tr_m):.3f}±{np.std(tr_m):.3f})")
            multi_res[task_name][pe] = {
                "accs": dict(accs_m), "histories": hists_m, "train_accs": tr_m
            }

            # Single length
            print(f"  [single] PE={pe:<8}", end="", flush=True)
            X_tr, y_tr = task_fn(config.TRAIN_LEN, N_TRAIN)
            accs_s = defaultdict(list); hists_s = []; tr_s = []
            for seed in seeds:
                set_seed(seed)
                model = MinimalTransformer(pos_encoding=pe, max_len=max_len_needed)
                hist = train_model(model, X_tr, y_tr, verbose=False)
                hists_s.append(hist); tr_s.append(hist["train_acc"][-1])
                for L in TEST_LENS:
                    X_t, y_t = task_fn(L, N_TEST)
                    accs_s[L].append(evaluate(model, X_t, y_t))
                print(".", end="", flush=True)
            print(f" done (train {np.mean(tr_s):.3f}+/-{np.std(tr_s):.3f})")
            single_res[task_name][pe] = {
                "accs": dict(accs_s), "histories": hists_s, "train_accs": tr_s
            }

    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, "multi_length_results.pkl")
        with open(path, "wb") as f:
            pickle.dump({"multi": multi_res, "single": single_res}, f)