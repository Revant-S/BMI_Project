"""
train.py
--------
End-to-end training script for the Small-CNN arrhythmia classifier.

Usage:
    python train.py

Outputs:
    checkpoints/best_model.pth   – best model weights (by val accuracy)
    checkpoints/last_model.pth   – weights after the final epoch
    results/training_curves.png  – loss & accuracy curves
    results/confusion_matrix.png – confusion matrix on the test set
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use('Agg')           # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SmallCNN1D

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PROCESSED_DIR = "./processed"
CKPT_DIR      = "./checkpoints"
RESULTS_DIR   = "./results"
CLASSES       = ['N', 'S', 'V']
NUM_CLASSES   = len(CLASSES)
INPUT_LENGTH  = 200           # samples per heartbeat segment

BATCH_SIZE    = 256
EPOCHS        = 30
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15          # must sum to 1.0

# automatically use Apple Silicon MPS, CUDA, or CPU
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class HeartbeatDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 200), y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,200)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_weighted_sampler(y_train):
    """Over-sample minority classes so the model doesn't just predict 'N'."""
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_train]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds  = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss per Epoch'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   label='Val Acc')
    axes[1].set_title('Accuracy per Epoch'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {save_path}")


def plot_confusion(preds, labels, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix – Test Set')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load preprocessed data ───────────────────────────────────────────────
    X_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")
    if not os.path.exists(X_path):
        raise FileNotFoundError(
            "Preprocessed data not found. Run `python preprocess.py` first!"
        )
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Loaded dataset: {X.shape}, labels: {y.shape}")

    # 2. Train / Val / Test split ─────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED, stratify=y_tmp
    )
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}  Test: {len(y_te):,}")

    # 3. Build DataLoaders ────────────────────────────────────────────────────
    train_ds  = HeartbeatDataset(X_tr,  y_tr)
    val_ds    = HeartbeatDataset(X_val, y_val)
    test_ds   = HeartbeatDataset(X_te,  y_te)
    sampler   = make_weighted_sampler(y_tr)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # 4. Model, loss, optimizer ───────────────────────────────────────────────
    model     = SmallCNN1D(num_classes=NUM_CLASSES, input_length=INPUT_LENGTH).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    # 5. Training loop ────────────────────────────────────────────────────────
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        flag = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'best_model.pth'))
            flag = '  ← best ✓'

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%{flag}")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'last_model.pth'))

    # 6. Evaluate on test set ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model.pth'),
                                     map_location=DEVICE))
    _, te_acc, te_preds, te_labels = evaluate(model, test_loader, criterion)

    print(f"\n{'='*55}")
    print(f"TEST SET RESULTS")
    print(f"{'='*55}")
    print(f"Overall Accuracy: {te_acc*100:.2f}%")
    print(f"\n{classification_report(te_labels, te_preds, target_names=CLASSES)}")

    # 7. Save plots ───────────────────────────────────────────────────────────
    plot_curves(history,       os.path.join(RESULTS_DIR, 'training_curves.png'))
    plot_confusion(te_preds, te_labels,
                              os.path.join(RESULTS_DIR, 'confusion_matrix.png'))

    print("\nDone! 🎉")
    print(f"  Best validation accuracy : {best_val_acc*100:.2f}%")
    print(f"  Final test accuracy      : {te_acc*100:.2f}%")
    print(f"  Model checkpoint         : {CKPT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
