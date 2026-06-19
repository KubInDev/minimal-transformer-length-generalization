"""
utils.py
Shared utilities: reproducibility, device, results persistence.
"""

import os
import pickle
import random
import numpy as np
import torch
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_results(results: dict, path: str | None = None) -> None:
    """Pickle the results dict to disk."""
    if path is None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, config.RESULTS_FILE)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved -> {path}")

def load_results(path: str | None = None) -> dict:
    """Load a pickled results dict from disk."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, config.RESULTS_FILE)
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(f"Results loaded <- {path}")
    return results
