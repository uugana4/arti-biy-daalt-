"""
F.CSM303 - Бие даалт №2
Сэдэв: Convolutional Neural Network (CNN)
Зорилго: Fashion-MNIST дата дээр CNN-ийг сургаж, baseline ба improved
загваруудын үр дүнг харьцуулан, сайжруулалт яаж нөлөөлж байгааг харуулах.
"""

from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# -------------------------------------------------------------------
# 0. Тогтвортой үр дүнгийн төлөө random seed
# -------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"[INFO] Тооцооллын төхөөрөмж: {DEVICE}")

WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR / "data"
OUT_DIR = WORK_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# -------------------------------------------------------------------
# 1. Өгөгдлийг бэлдэх (Fashion-MNIST)
# -------------------------------------------------------------------
def get_loaders(augment: bool, batch_size: int = 256):
    """Fashion-MNIST дата ачаалах. augment=True үед сургалтын дата
    дээр data augmentation хэрэглэнэ."""
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST mean/std
    ]
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(28, padding=2),
            *base_transform,
        ])
    else:
        train_transform = transforms.Compose(base_transform)

    test_transform = transforms.Compose(base_transform)

    train_set = datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True, transform=test_transform
    )

    # CPU дээр хурдан ажиллуулахын тулд train-аас 20000 жишээ авна
    train_indices = list(range(20000))
    train_set = Subset(train_set, train_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


# -------------------------------------------------------------------
# 2. Загваруудын тодорхойлолт
# -------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Энгийн CNN: 2 Conv + Pool + 1 FC. Ямар ч regularization байхгүй."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14x32
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ImprovedCNN(nn.Module):
    """Сайжруулсан CNN:
    - Илүү гүн (3 conv block)
    - BatchNorm: сургалтыг тогтворжуулна
    - Dropout: overfitting-ийг бууруулна
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),               # 14x14
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),               # 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------------------------
# 3. Сургалт ба үнэлгээний функцууд
# -------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, all_preds, all_targets = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return running_loss / len(y_true), y_true, y_pred


def run_experiment(name, model, epochs, lr, augment, weight_decay=0.0):
    """Нэг загварыг бүхэлд нь сургах, үнэлэх, түүх гаргах."""
    print(f"\n{'='*60}\n[RUN] {name}\n{'='*60}")
    train_loader, test_loader = get_loaders(augment=augment)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    start = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, y_true, y_pred = evaluate(model, test_loader, criterion)
        va = accuracy_score(y_true, y_pred)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"  epoch {ep:>2}/{epochs} | "
              f"train_loss={tl:.4f} acc={ta:.4f} | "
              f"val_loss={vl:.4f} acc={va:.4f}")
    duration = time.time() - start

    # Эцсийн үнэлгээний metric
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall":    recall_score(y_true, y_pred, average="macro"),
        "f1":        f1_score(y_true, y_pred, average="macro"),
        "duration_sec": duration,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    print(f"  [DONE] {name} -> acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1']:.4f}  time={duration:.1f}s")

    return history, metrics, y_true, y_pred, model


# -------------------------------------------------------------------
# 4. Үр дүнг зурагт буулгах
# -------------------------------------------------------------------
def plot_curves(hist_a, hist_b, name_a, name_b, path):
    epochs_a = range(1, len(hist_a["val_acc"]) + 1)
    epochs_b = range(1, len(hist_b["val_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_a, hist_a["val_acc"], "o-", label=name_a)
    axes[0].plot(epochs_b, hist_b["val_acc"], "s-", label=name_b)
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_a, hist_a["val_loss"], "o-", label=name_a)
    axes[1].plot(epochs_b, hist_b["val_loss"], "s-", label=name_b)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_sample_predictions(model, path, n=12):
    """Тестийн дата дээрх таамаглалын жишээ зургийг хадгална."""
    _, test_loader = get_loaders(augment=False, batch_size=n)
    model.eval()
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
    yb = yb.numpy()
    fig, axes = plt.subplots(2, 6, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(xb[i, 0].numpy(), cmap="gray")
        color = "green" if preds[i] == yb[i] else "red"
        ax.set_title(
            f"T:{CLASS_NAMES[yb[i]]}\nP:{CLASS_NAMES[preds[i]]}",
            color=color, fontsize=9,
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# -------------------------------------------------------------------
# 5. Үндсэн pipeline
# -------------------------------------------------------------------
def main():
    # --- Baseline ---
    base_hist, base_metrics, by_true, by_pred, _ = run_experiment(
        name="Baseline CNN (augmentation байхгүй, regularization байхгүй)",
        model=BaselineCNN(),
        epochs=4,
        lr=1e-3,
        augment=False,
    )

    # --- Improved ---
    imp_hist, imp_metrics, iy_true, iy_pred, improved_model = run_experiment(
        name="Improved CNN (BatchNorm + Dropout + Augmentation + WeightDecay)",
        model=ImprovedCNN(),
        epochs=6,
        lr=1e-3,
        augment=True,
        weight_decay=1e-4,
    )

    # --- Графикууд ---
    plot_curves(base_hist, imp_hist,
                "Baseline", "Improved",
                OUT_DIR / "training_curves.png")
    plot_confusion(by_true, by_pred,
                   "Confusion Matrix - Baseline",
                   OUT_DIR / "cm_baseline.png")
    plot_confusion(iy_true, iy_pred,
                   "Confusion Matrix - Improved",
                   OUT_DIR / "cm_improved.png")

    # Сургагдсан Improved загвараар тестийн дээж таамаглах жишээ
    plot_sample_predictions(improved_model,
                            OUT_DIR / "sample_predictions.png")

    # --- Хүснэгт ---
    summary = {
        "Baseline":  base_metrics,
        "Improved":  imp_metrics,
        "delta_accuracy_pct": (imp_metrics["accuracy"]
                               - base_metrics["accuracy"]) * 100,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== ҮР ДҮНГИЙН ХҮСНЭГТ ==========")
    print(f"{'Metric':<12}{'Baseline':>12}{'Improved':>12}{'Delta':>10}")
    for k in ["accuracy", "precision", "recall", "f1"]:
        b, i = base_metrics[k], imp_metrics[k]
        print(f"{k:<12}{b:>12.4f}{i:>12.4f}{(i-b):>+10.4f}")
    print(f"{'params':<12}{base_metrics['n_params']:>12,}"
          f"{imp_metrics['n_params']:>12,}")
    print(f"{'time(s)':<12}{base_metrics['duration_sec']:>12.1f}"
          f"{imp_metrics['duration_sec']:>12.1f}")

    # Classification report (Improved)
    rep = classification_report(iy_true, iy_pred,
                                target_names=CLASS_NAMES, digits=4)
    with open(OUT_DIR / "classification_report_improved.txt", "w") as f:
        f.write(rep)
    print("\nImproved загварын classification report:")
    print(rep)

    print(f"\n[OK] Бүх үр дүн '{OUT_DIR}' хавтсанд хадгалагдлаа.")


if __name__ == "__main__":
    main()
