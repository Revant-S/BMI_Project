"""
Training loop for SmallNet and GoogLeNet.

Supports SGD with momentum, validation monitoring, early stopping,
checkpoint saving, and training-curve plotting.

Usage:
    python -m src.train --model smallnet --epochs 50
    python -m src.train --model googlenet --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    BATCH_SIZE, LEARNING_RATE, MOMENTUM, EPOCHS,
    EARLY_STOP_PATIENCE, RESULTS_DIR, RANDOM_SEED,
)
from src.dataset import get_dataloaders
from src.model import get_model, count_parameters


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    is_googlenet_train=False):
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float (0-100)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # GoogLeNet returns GoogLeNetOutputs during training (with aux)
        if is_googlenet_train and hasattr(outputs, 'logits'):
            loss_main = criterion(outputs.logits, labels)
            loss_aux1 = criterion(outputs.aux_logits1, labels) if outputs.aux_logits1 is not None else 0
            loss_aux2 = criterion(outputs.aux_logits2, labels) if outputs.aux_logits2 is not None else 0
            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            preds = outputs.logits.argmax(dim=1)
        else:
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation.

    Returns
    -------
    avg_loss : float
    accuracy : float (0-100)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # In eval mode GoogLeNet returns plain tensor
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits

        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_history(history: dict, model_name: str, save_dir: Path):
    """Plot training / validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#4e79a7')
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#e15759')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#4e79a7')
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc', color='#e15759')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / f"training_curves_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument('--model', type=str, default='smallnet',
                        choices=['smallnet', 'googlenet'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    _set_seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"  Phase 5 - Training  [{args.model.upper()}]")
    print(f"  Device: {device}")
    print("=" * 60)

    # Data
    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=args.batch_size,
    )

    # Model
    model = get_model(args.model, num_classes=len(class_names))
    model = model.to(device)
    print(f"  Trainable params: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM)

    is_googlenet = args.model == 'googlenet'

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            is_googlenet_train=is_googlenet,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:3d}/{args.epochs}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}%  |  "
              f"{elapsed:.1f}s")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = RESULTS_DIR / f"best_{args.model}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"    * Saved best model (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # Save history
    history_path = RESULTS_DIR / f"history_{args.model}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to {history_path}")

    # Plot curves
    plot_history(history, args.model, RESULTS_DIR)

    print(f"\n  [OK] Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
