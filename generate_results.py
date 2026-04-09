"""
Generate all results from the current SmallNet checkpoint.
Produces: training curves, confusion matrix, ROC curve, and metrics summary.

Usage:
    python generate_results.py
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from pathlib import Path

# Setup
sys.path.insert(0, '.')
from src.config import RESULTS_DIR, BATCH_SIZE, CLASS_NAMES
from src.dataset import get_dataloaders
from src.model import get_model, count_parameters

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Plot training curves from saved history
# ──────────────────────────────────────────────
print("=" * 60)
print("  Generating Results for SmallNet (8 epochs)")
print("=" * 60)

history_path = RESULTS_DIR / "history_smallnet.json"
with open(history_path) as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)

# --- Training Curves (Loss + Accuracy side by side) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss',
         color='#4e79a7', linewidth=2, markersize=6)
ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss',
         color='#e15759', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('SmallNet - Training & Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(list(epochs))

ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc',
         color='#4e79a7', linewidth=2, markersize=6)
ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc',
         color='#e15759', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('SmallNet - Training & Validation Accuracy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(list(epochs))

fig.tight_layout()
fig.savefig(RESULTS_DIR / "training_curves_smallnet.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [OK] Training curves saved")

# ──────────────────────────────────────────────
# 2. Load model and run test-set evaluation
# ──────────────────────────────────────────────
device = torch.device('cpu')

print("  Loading test data...")
_, _, test_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)
print(f"  Test set: {len(test_loader.dataset):,} samples")

model = get_model('smallnet', num_classes=2)
ckpt_path = RESULTS_DIR / "best_smallnet.pth"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(val_acc={checkpoint['val_acc']:.2f}%)")
print(f"  Model params: {count_parameters(model):,}")

# Inference
all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        preds = outputs.argmax(dim=1).numpy()
        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)
y_prob = np.concatenate(all_probs)

# ──────────────────────────────────────────────
# 3. Compute metrics
# ──────────────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

metrics = {
    'accuracy': accuracy_score(y_true, y_pred) * 100,
    'sensitivity': recall_score(y_true, y_pred) * 100,
    'specificity': tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0,
    'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
    'f1_score': f1_score(y_true, y_pred, zero_division=0) * 100,
    'auc': roc_auc_score(y_true, y_prob) * 100,
    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
}

print(f"\n  {'-' * 45}")
print(f"  TEST SET RESULTS - SMALLNET (8 epochs)")
print(f"  {'-' * 45}")
print(f"  Accuracy    : {metrics['accuracy']:.2f}%")
print(f"  Sensitivity : {metrics['sensitivity']:.2f}%")
print(f"  Specificity : {metrics['specificity']:.2f}%")
print(f"  Precision   : {metrics['precision']:.2f}%")
print(f"  F1 Score    : {metrics['f1_score']:.2f}%")
print(f"  AUC         : {metrics['auc']:.2f}%")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

# Save metrics
fpr_roc, tpr_roc, _ = roc_curve(y_true, y_prob)
metrics['fpr'] = fpr_roc.tolist()
metrics['tpr'] = tpr_roc.tolist()
with open(RESULTS_DIR / "metrics_smallnet.json", 'w') as f:
    json.dump(metrics, f, indent=2)
print("  [OK] Metrics saved")

# Full classification report
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open(RESULTS_DIR / "classification_report_smallnet.txt", 'w') as f:
    f.write(report)

# ──────────────────────────────────────────────
# 4. Confusion Matrix plot
# ──────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            annot_kws={"size": 16})
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix - SmallNet (8 epochs)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix_smallnet.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [OK] Confusion matrix saved")

# ──────────────────────────────────────────────
# 5. ROC Curve plot
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_roc, tpr_roc, color='#4e79a7', lw=2.5,
        label=f'SmallNet  (AUC = {metrics["auc"]/100:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
ax.fill_between(fpr_roc, tpr_roc, alpha=0.1, color='#4e79a7')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve - SmallNet (8 epochs)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "roc_curve_smallnet.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [OK] ROC curve saved")

# ──────────────────────────────────────────────
# 6. Summary dashboard (single image with all key info)
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle('ECG Arrhythmia Detection - SmallNet Results (8 Epochs, CPU Training)',
             fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: Training curves - Loss
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(epochs, history['train_loss'], 'o-', label='Train', color='#4e79a7', lw=2)
ax1.plot(epochs, history['val_loss'], 's-', label='Val', color='#e15759', lw=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Training curves - Accuracy
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(epochs, history['train_acc'], 'o-', label='Train', color='#4e79a7', lw=2)
ax2.plot(epochs, history['val_acc'], 's-', label='Val', color='#e15759', lw=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Confusion Matrix
ax3 = fig.add_subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax3,
            annot_kws={"size": 14})
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title('Confusion Matrix')

# Subplot 4: ROC Curve
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(fpr_roc, tpr_roc, color='#4e79a7', lw=2.5,
         label=f'AUC = {metrics["auc"]/100:.4f}')
ax4.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
ax4.fill_between(fpr_roc, tpr_roc, alpha=0.1, color='#4e79a7')
ax4.set_xlabel('FPR')
ax4.set_ylabel('TPR')
ax4.set_title('ROC Curve')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# Subplot 5: Metrics table
ax5 = fig.add_subplot(2, 3, 5)
ax5.axis('off')
table_data = [
    ['Metric', 'Value'],
    ['Accuracy', f'{metrics["accuracy"]:.2f}%'],
    ['Sensitivity', f'{metrics["sensitivity"]:.2f}%'],
    ['Specificity', f'{metrics["specificity"]:.2f}%'],
    ['Precision', f'{metrics["precision"]:.2f}%'],
    ['F1 Score', f'{metrics["f1_score"]:.2f}%'],
    ['AUC', f'{metrics["auc"]:.2f}%'],
]
table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)
for i in range(len(table_data[0])):
    table[0, i].set_facecolor('#4e79a7')
    table[0, i].set_text_props(color='white', fontweight='bold')
ax5.set_title('Performance Metrics', fontsize=12, fontweight='bold')

# Subplot 6: Model info
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
info_text = (
    f"Model: SmallNet (Custom CNN)\n"
    f"Parameters: {count_parameters(model):,}\n"
    f"Architecture: 4x(Conv-BN-ReLU-MaxPool) + FC\n"
    f"Filter sizes: 16 -> 4 -> 8 -> 16\n"
    f"\n"
    f"Dataset: MIT-BIH Arrhythmia (PhysioNet)\n"
    f"Total samples: 21,182 (balanced)\n"
    f"Train/Val/Test: 14,827 / 3,177 / 3,178\n"
    f"\n"
    f"Training: SGD (lr=0.0001, momentum=0.9)\n"
    f"Epochs completed: 8 / 50\n"
    f"Best checkpoint: Epoch 2\n"
    f"Device: CPU"
)
ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
ax6.set_title('Model & Training Info', fontsize=12, fontweight='bold')

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(RESULTS_DIR / "dashboard_smallnet.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [OK] Summary dashboard saved")

# ──────────────────────────────────────────────
# 7. Print final summary
# ──────────────────────────────────────────────
print(f"\n  {'=' * 50}")
print(f"  ALL RESULTS SAVED TO: {RESULTS_DIR}")
print(f"  {'=' * 50}")
print(f"  Files generated:")
print(f"    - training_curves_smallnet.png")
print(f"    - confusion_matrix_smallnet.png")
print(f"    - roc_curve_smallnet.png")
print(f"    - dashboard_smallnet.png  (all-in-one)")
print(f"    - metrics_smallnet.json")
print(f"    - classification_report_smallnet.txt")
print(f"\n{report}")
