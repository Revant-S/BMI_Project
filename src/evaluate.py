"""
Evaluate trained models on the test set.

Computes confusion matrix, accuracy, sensitivity, specificity, precision,
F1-score, ROC curve + AUC.  Optionally generates a side-by-side comparison
table if both models have been evaluated.

Usage:
    python -m src.evaluate --model smallnet
    python -m src.evaluate --model googlenet
    python -m src.evaluate --compare
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from src.config import RESULTS_DIR, BATCH_SIZE, CLASS_NAMES
from src.dataset import get_dataloaders
from src.model import get_model, count_parameters


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on all batches, return ground-truth and predicted arrays.

    Returns
    -------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)
    y_prob : np.ndarray, shape (N,)  — probability of class 1 (Abnormal)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)

        if hasattr(outputs, 'logits'):
            outputs = outputs.logits

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    return (np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics.

    Returns
    -------
    metrics : dict
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'sensitivity': recall_score(y_true, y_pred) * 100,       # recall for positive class
        'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0,
        'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, zero_division=0) * 100,
        'auc': roc_auc_score(y_true, y_prob) * 100,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    return metrics


def print_metrics(metrics: dict, model_name: str):
    """Pretty-print evaluation metrics."""
    print(f"\n  {'-' * 40}")
    print(f"  Evaluation Results - {model_name.upper()}")
    print(f"  {'-' * 40}")
    print(f"  Accuracy    : {metrics['accuracy']:.2f}%")
    print(f"  Sensitivity : {metrics['sensitivity']:.2f}%")
    print(f"  Specificity : {metrics['specificity']:.2f}%")
    print(f"  Precision   : {metrics['precision']:.2f}%")
    print(f"  F1 Score    : {metrics['f1_score']:.2f}%")
    print(f"  AUC         : {metrics['auc']:.2f}%")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  "
          f"FP={metrics['fp']}  FN={metrics['fn']}")


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: Path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    fig.tight_layout()

    path = save_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {path}")


def plot_roc_curve(y_true, y_prob, model_name: str, save_dir: Path, auc_val=None):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc_val or roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#4e79a7', lw=2,
            label=f'{model_name}  (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = save_dir / f"roc_curve_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved to {path}")


def plot_roc_comparison(save_dir: Path):
    """Plot ROC curves for both models side by side (if both exist)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {'smallnet': '#4e79a7', 'googlenet': '#e15759'}

    found = False
    for name in ['smallnet', 'googlenet']:
        metrics_path = save_dir / f"metrics_{name}.json"
        if not metrics_path.exists():
            continue
        found = True

        # We need the raw predictions to plot — reload test set
        # For simplicity, store fpr/tpr in the metrics JSON during evaluate
        data = json.loads(metrics_path.read_text())
        if 'fpr' in data and 'tpr' in data:
            fpr = data['fpr']
            tpr = data['tpr']
            auc_val = data['auc'] / 100.0
            ax.plot(fpr, tpr, color=colors[name], lw=2,
                    label=f'{name}  (AUC = {auc_val:.4f})')

    if not found:
        return

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = save_dir / "roc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ROC comparison saved to {path}")


def compare_models(save_dir: Path):
    """Print a side-by-side comparison table if both models are evaluated."""
    rows = []
    for name in ['smallnet', 'googlenet']:
        path = save_dir / f"metrics_{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            rows.append((name, data))

    if len(rows) < 2:
        print("  [SKIP] Need both models evaluated to compare. "
              "Run evaluate for both first.")
        return

    header = f"{'Metric':<15s} | {'SmallNet':>10s} | {'GoogLeNet':>10s}"
    print(f"\n  {'=' * len(header)}")
    print(f"  Model Comparison")
    print(f"  {'=' * len(header)}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for key in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'auc']:
        v1 = rows[0][1].get(key, 0)
        v2 = rows[1][1].get(key, 0)
        print(f"  {key:<15s} | {v1:>9.2f}% | {v2:>9.2f}%")

    print()

    # Also plot the comparison ROC
    plot_roc_comparison(save_dir)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG classifier")
    parser.add_argument('--model', type=str, default='smallnet',
                        choices=['smallnet', 'googlenet'])
    parser.add_argument('--compare', action='store_true',
                        help='Compare both models (no model arg needed)')
    args = parser.parse_args()

    if args.compare:
        compare_models(RESULTS_DIR)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"  Phase 6 - Evaluation  [{args.model.upper()}]")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load test data
    _, _, test_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)

    # Load model + checkpoint
    model = get_model(args.model, num_classes=len(class_names))
    ckpt_path = RESULTS_DIR / f"best_{args.model}.pth"

    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        print("  Train the model first: python -m src.train --model", args.model)
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.2f}%)")

    # Collect predictions
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, args.model)

    # ROC data for later comparison
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()

    # Save metrics
    metrics_path = RESULTS_DIR / f"metrics_{args.model}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Plots
    plot_confusion_matrix(y_true, y_pred, args.model, RESULTS_DIR)
    plot_roc_curve(y_true, y_prob, args.model, RESULTS_DIR,
                   auc_val=metrics['auc'] / 100.0)

    # Classification report
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")


if __name__ == "__main__":
    main()
