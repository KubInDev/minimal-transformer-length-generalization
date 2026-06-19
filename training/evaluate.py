"""
training/evaluate.py
"""

import torch
import torch.nn as nn
from utils import device

@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Return accuracy on (X, y)."""
    model.eval()
    X, y = X.to(device), y.to(device)
    preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()

def detect_grokking(
    history: dict,
    window: int = 5,
    threshold: float = 0.15,
    start_epoch: int = 20,
) -> bool:
    """
    Return True if training accuracy jumps by >= threshold within,
    any window of `window` epochs after `start_epoch`.
    """
    accs = history["train_acc"]
    if len(accs) < start_epoch + window:
        return False
    for i in range(start_epoch, len(accs) - window):
        if accs[i + window] - accs[i] >= threshold:
            return True
    return False
