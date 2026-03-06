"""
Evaluation: accuracy, F1-scores, confusion matrices, model comparison.

Usage:
    python -m src.training.evaluate --model punch
    python -m src.training.evaluate --model defense
    python -m src.training.evaluate --compare
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader

import config
from src.data.dataset import get_punch_loaders, get_defense_loaders
from src.models.punch_classifier import PunchClassifier
from src.models.defense_classifier import DefenseClassifier
from src.models.baselines import RuleBasedClassifier, FrameSVM, FeedforwardMLP


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(model_name: str, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_name == "punch":
        model = PunchClassifier()
    elif model_name == "defense":
        model = DefenseClassifier()
    elif model_name == "mlp_punch":
        model = FeedforwardMLP(num_classes=config.NUM_PUNCH_CLASSES)
    elif model_name == "mlp_defense":
        model = FeedforwardMLP(
            input_dim=config.SEQUENCE_LENGTH * 66,
            num_classes=config.NUM_DEFENSE_CLASSES
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Get all predictions and true labels from a DataLoader."""
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list[str], title: str, save_path: Path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} — Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_history(history: dict, model_name: str, save_path: Path):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {save_path}")


def evaluate_model(model_name: str):
    """Full evaluation of a trained model on the test set."""
    device = get_device()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 50}")

    # Load model and data
    model = load_trained_model(model_name, device)

    if "defense" in model_name:
        _, _, test_loader = get_defense_loaders()
        class_names = config.DEFENSE_CLASSES
    else:
        _, _, test_loader = get_punch_loaders()
        class_names = config.PUNCH_CLASSES

    # Predictions
    y_pred, y_true = get_predictions(model, test_loader, device)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n  Accuracy:    {acc:.1%}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")

    # Classification report — only include classes present in the test set
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names = [class_names[i] for i in present_labels]
    report = classification_report(y_true, y_pred, labels=present_labels,
                                   target_names=present_names, zero_division=0)
    print(f"\n{report}")

    # Save report
    report_path = config.RESULTS_DIR / f"{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm_path = config.RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, model_name, cm_path)

    # Training history plot
    history_path = config.RESULTS_DIR / f"{model_name}_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        curves_path = config.RESULTS_DIR / f"{model_name}_training_curves.png"
        plot_training_history(history, model_name, curves_path)

    return {"model": model_name, "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def evaluate_rule_based(task: str = "punch"):
    """Evaluate the rule-based baseline."""
    print(f"\n{'=' * 50}")
    print(f"Evaluating: Rule-Based ({task})")
    print(f"{'=' * 50}")

    rb = RuleBasedClassifier()

    if task == "punch":
        _, _, test_loader = get_punch_loaders()
        class_names = config.PUNCH_CLASSES
    else:
        _, _, test_loader = get_defense_loaders()
        class_names = config.DEFENSE_CLASSES

    all_preds = []
    all_labels = []

    for X_batch, y_batch in test_loader:
        preds = rb.predict_batch(X_batch.numpy(), task=task)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Accuracy: {acc:.1%}, F1 (macro): {f1_macro:.4f}")

    return {"model": f"rule_based_{task}", "accuracy": acc, "f1_macro": f1_macro}


def evaluate_svm(task: str = "punch"):
    """Evaluate the SVM baseline (must be trained first)."""
    svm = FrameSVM()
    model_path = config.CHECKPOINTS_DIR / f"svm_{task}.pkl"

    if not model_path.exists():
        print(f"  SVM model not found at {model_path}. Training...")
        if task == "punch":
            train_loader, _, test_loader = get_punch_loaders()
        else:
            train_loader, _, test_loader = get_defense_loaders()

        # Collect training data
        X_train, y_train = [], []
        for X_batch, y_batch in train_loader:
            X_train.append(X_batch.numpy())
            y_train.append(y_batch.numpy())
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        svm.fit(X_train, y_train)
        config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        svm.save(model_path)
    else:
        svm.load(model_path)

    if task == "punch":
        _, _, test_loader = get_punch_loaders()
    else:
        _, _, test_loader = get_defense_loaders()

    X_test, y_test = [], []
    for X_batch, y_batch in test_loader:
        X_test.append(X_batch.numpy())
        y_test.append(y_batch.numpy())
    X_test = np.concatenate(X_test)
    y_true = np.concatenate(y_test)

    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  SVM ({task}) — Accuracy: {acc:.1%}, F1 (macro): {f1_macro:.4f}")

    return {"model": f"svm_{task}", "accuracy": acc, "f1_macro": f1_macro}


def compare_all():
    """Run evaluation on all models and print comparison table."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = []

    # Try each model (skip if checkpoint not found)
    for model_name in ["punch", "defense", "mlp_punch", "mlp_defense"]:
        try:
            r = evaluate_model(model_name)
            results.append(r)
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")

    # Baselines
    for task in ["punch", "defense"]:
        try:
            results.append(evaluate_rule_based(task))
            results.append(evaluate_svm(task))
        except Exception as e:
            print(f"  Baseline error ({task}): {e}")

    if results:
        print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 (macro)':>12}")
        print("-" * 45)
        for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
            print(f"{r['model']:<20} {r['accuracy']:>9.1%} {r['f1_macro']:>11.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate boxing classifiers.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["punch", "defense", "mlp_punch", "mlp_defense"],
                        help="Specific model to evaluate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models")
    args = parser.parse_args()

    if args.compare:
        compare_all()
    elif args.model:
        evaluate_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
