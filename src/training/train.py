"""
Training loop for punch and defense classifiers.

Usage:
    python -m src.training.train --model punch
    python -m src.training.train --model defense
    python -m src.training.train --model mlp_punch   # Baseline 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import config
from src.data.dataset import get_punch_loaders, get_defense_loaders
from src.models.punch_classifier import PunchClassifier
from src.models.defense_classifier import DefenseClassifier
from src.models.baselines import FeedforwardMLP


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_name: str) -> nn.Module:
    """Instantiate model by name."""
    models = {
        "punch": PunchClassifier,
        "defense": DefenseClassifier,
        "mlp_punch": lambda: FeedforwardMLP(
            num_classes=config.NUM_PUNCH_CLASSES
        ),
        "mlp_defense": lambda: FeedforwardMLP(
            input_dim=config.SEQUENCE_LENGTH * 66,  # head features
            num_classes=config.NUM_DEFENSE_CLASSES
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()


def get_loaders(model_name: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get appropriate data loaders for the model."""
    if "defense" in model_name:
        return get_defense_loaders()
    return get_punch_loaders()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train(model_name: str, num_epochs: int = config.NUM_EPOCHS,
          lr: float = config.LEARNING_RATE, batch_size: int = config.BATCH_SIZE):
    """
    Full training pipeline with early stopping and LR scheduling.
    """
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Build model
    model = build_model(model_name).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} ({param_count:,} parameters)")

    # Data
    train_loader, val_loader, test_loader = get_loaders(model_name)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config.LR_SCHEDULER_PATIENCE,
        factor=config.LR_SCHEDULER_FACTOR,
    )

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Output paths
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"

    print(f"\nTraining for up to {num_epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6d} {train_loss:>11.4f} {train_acc:>9.1%} {val_loss:>10.4f} "
              f"{val_acc:>8.1%} {current_lr:>10.6f}  ({elapsed:.1f}s)")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.EARLY_STOPPING_PATIENCE})")
                break

    # Save training history
    history_path = config.RESULTS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest checkpoint saved: {checkpoint_path}")
    print(f"Training history saved: {history_path}")

    # Load best model and evaluate on test set
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.1%}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train boxing classifier.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["punch", "defense", "mlp_punch", "mlp_defense"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(args.model, args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
