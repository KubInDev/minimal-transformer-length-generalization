"""
training/trainer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import device


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    patience: int = config.PATIENCE,
    min_train_acc: float = config.MIN_TRAIN_ACC,
    warmup_epochs: int = config.WARMUP_EPOCHS,
    verbose: bool = False,
) -> dict:
    """
    Train model and return history dict with per-epoch 'loss' and 'train_acc'
    """
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
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()

        avg_loss = total_loss / total
        train_acc = correct / total
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        if verbose and epoch % 10 == 0:
            print(
                f"epoch {epoch:3d}  "
                f"loss={avg_loss:.4f}  "
                f"train_acc={train_acc:.4f}"
            )


        if train_acc >= min_train_acc:
            if verbose:
                print(f" -> converged at epoch {epoch}  "
                      f"(train_acc={train_acc:.4f})")
            break


        if epoch >= warmup_epochs:
            if train_acc - best_acc > 0.005:
                best_acc = train_acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f" -> plateau stop at epoch {epoch}  "
                          f"(no improve for {patience} epochs)")
                break
        else:
            best_acc = max(best_acc, train_acc)

    return history
