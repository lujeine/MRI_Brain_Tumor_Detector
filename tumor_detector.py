"""Brain-tumor MRI classifier: data loading, model, training and overfitting checks.

Dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
Expected layout (class folders hold .jpg files)::

    brain_tumor_dataset/
        yes/   # tumor     -> label 1
        no/    # healthy   -> label 0
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASS_NAMES = ["Healthy", "Tumor"]  # index == label


# --------------------------------------------------------------------------- #
# Configuration / reproducibility
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    data_dir: str = "brain_tumor_dataset"
    img_size: int = 128
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    patience: int = 15  # early-stopping patience (epochs without val-loss gain)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_images(data_dir: str, img_size: int = 128):
    """Read every image once. Returns (images uint8 NHWC RGB, labels float32, n_duplicates_removed).

    Unreadable files are skipped and exact duplicates (after resizing) are dropped,
    because a duplicate that lands in both train and test inflates test accuracy.
    """
    root = Path(data_dir)
    images, labels, seen = [], [], set()
    n_dupes = 0
    for folder, label in (("no", 0.0), ("yes", 1.0)):
        files = sorted(p for p in (root / folder).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise FileNotFoundError(f"No images found in {root / folder}")
        for path in files:
            img = cv2.imread(str(path))
            if img is None:
                print(f"Skipping unreadable file: {path}")
                continue
            img = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2RGB)
            digest = hashlib.md5(img.tobytes()).hexdigest()
            if digest in seen:
                n_dupes += 1
                continue
            seen.add(digest)
            images.append(img)
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.float32), n_dupes


def make_splits(labels: np.ndarray, cfg: Config):
    """Stratified train / val / test index split."""
    idx = np.arange(len(labels))
    trainval, test = train_test_split(idx, test_size=cfg.test_size, stratify=labels, random_state=cfg.seed)
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(trainval, test_size=rel_val, stratify=labels[trainval], random_state=cfg.seed)
    return train, val, test


def channel_stats(images: np.ndarray):
    """Per-channel mean/std from the *training* images only (no test leakage)."""
    x = images.astype(np.float32) / 255.0
    return x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2)) + 1e-7


class MRIDataset(Dataset):
    """Images are NHWC uint8; items come out as normalised CHW float tensors."""

    def __init__(self, images, labels, mean, std, augment: bool = False):
        self.images, self.labels = images, labels
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.aug = (
            v2.Compose([v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))])
            if augment
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0  # HWC -> CHW (a real transpose)
        if self.aug is not None:
            x = self.aug(x)
        return (x - self.mean) / self.std, torch.tensor(self.labels[i])


def build_datasets(images, labels, train_idx, val_idx, test_idx, train_labels=None):
    """`train_labels` (optional) replaces the labels of the training images only (used by the shuffled-label test)."""
    mean, std = channel_stats(images[train_idx])
    y_train = labels[train_idx] if train_labels is None else train_labels
    return {
        "train": MRIDataset(images[train_idx], y_train, mean, std, augment=True),
        "train_eval": MRIDataset(images[train_idx], y_train, mean, std),  # no augmentation: used to measure train accuracy
        "val": MRIDataset(images[val_idx], labels[val_idx], mean, std),
        "test": MRIDataset(images[test_idx], labels[test_idx], mean, std),
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CNN(nn.Module):
    """LeNet-style CNN. Outputs raw logits (use BCEWithLogitsLoss; apply sigmoid for probabilities)."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.Tanh(), nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(kernel_size=2, stride=5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(120, 84), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_proba(model, dataset, device, batch_size: int = 64):
    model.eval()
    probs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def loss_and_acc(model, dataset, device):
    p, y = predict_proba(model, dataset, device)
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return loss, float(accuracy_score(y, p >= 0.5))


def train_model(datasets, cfg: Config, device, early_stopping: bool = True, epochs: int | None = None, verbose: bool = True):
    """Train a fresh CNN. Returns (model, history).

    With early_stopping=True the weights from the best validation-loss epoch are restored.
    The validation set is only used for that choice, never the test set.
    """
    epochs = epochs or cfg.epochs
    model = CNN(cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True)

    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()

        tr_loss, tr_acc = loss_and_acc(model, datasets["train_eval"], device)
        va_loss, va_acc = loss_and_acc(model, datasets["val"], device)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)

        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if early_stopping and stale >= cfg.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch if early_stopping else len(history["train_loss"])
    return model, history


def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for an accuracy measured on n samples."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def evaluate(model, dataset, device, threshold: float = 0.5) -> dict:
    p, y = predict_proba(model, dataset, device)
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]),
        "n": len(y),
    }
    out["accuracy_ci95"] = wilson_interval(int((pred == y).sum()), len(y))
    return out


# --------------------------------------------------------------------------- #
# Overfitting checks
# --------------------------------------------------------------------------- #
def generalisation_gap(model, datasets, device) -> dict:
    """Train / val / test accuracy side by side (train measured without augmentation)."""
    accs = {name: loss_and_acc(model, datasets[key], device)[1] for name, key in
            (("train", "train_eval"), ("val", "val"), ("test", "test"))}
    accs["train_minus_test"] = accs["train"] - accs["test"]
    return accs


def cross_validate(images, labels, cfg: Config, device, holdout_idx=None, n_splits: int = 5):
    """Stratified k-fold CV with the same recipe (early stopping on an inner val split).

    `holdout_idx` (the final test set) is excluded so CV never touches it.
    """
    pool = np.setdiff1d(np.arange(len(labels)), holdout_idx if holdout_idx is not None else [])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(pool, labels[pool]), 1):
        tr_idx, te_idx = pool[tr], pool[te]
        inner_tr, inner_val = train_test_split(tr_idx, test_size=0.15, stratify=labels[tr_idx], random_state=cfg.seed)
        ds = build_datasets(images, labels, inner_tr, inner_val, te_idx)
        seed_everything(cfg.seed + fold)
        model, _ = train_model(ds, cfg, device, verbose=False)
        g = generalisation_gap(model, ds, device)
        rows.append({"fold": fold, "train_acc": g["train"], "test_acc": g["test"]})
        print(f"fold {fold}: train {g['train']:.3f}  held-out {g['test']:.3f}")
    return rows


def shuffled_label_test(images, labels, train_idx, val_idx, test_idx, cfg: Config, device, epochs: int = 60):
    """Randomisation test: train on *random* labels with no early stopping.

    If the model still reaches high train accuracy, it can memorise this data, so
    train accuracy alone says nothing. Held-out accuracy should sit near chance (~majority rate).
    """
    rng = np.random.default_rng(cfg.seed)
    ds = build_datasets(images, labels, train_idx, val_idx, test_idx, train_labels=rng.permutation(labels[train_idx]))
    model, hist = train_model(ds, cfg, device, early_stopping=False, epochs=epochs, verbose=False)
    return {
        "train_acc_on_random_labels": hist["train_acc"][-1],
        "test_acc_true_labels": loss_and_acc(model, ds["test"], device)[1],
        "majority_class_rate": float(max(labels[test_idx].mean(), 1 - labels[test_idx].mean())),
    }


def verdict(gap: float) -> str:
    """Rule-of-thumb read-out; thresholds are heuristics, not statistical tests."""
    if gap > 0.10:
        return "OVERFITTING LIKELY: train accuracy exceeds held-out accuracy by more than 10 points."
    if gap > 0.05:
        return "MILD GAP: 5-10 points between train and held-out accuracy; watch it."
    return "NO CLEAR OVERFITTING: train and held-out accuracy are within 5 points."


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_samples(images, labels, per_class: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, per_class, figsize=(3 * per_class, 6))
    for r, cls in enumerate((0, 1)):
        for ax, i in zip(axes[r], rng.choice(np.where(labels == cls)[0], per_class, replace=False)):
            ax.imshow(images[i]); ax.set_title(CLASS_NAMES[cls]); ax.axis("off")
    fig.tight_layout()
    return fig


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(epochs, history["train_loss"], label="train"); a.plot(epochs, history["val_loss"], label="val")
    a.set(title="Loss", xlabel="epoch"); a.legend(); a.grid(alpha=.3)
    b.plot(epochs, history["train_acc"], label="train"); b.plot(epochs, history["val_acc"], label="val")
    b.set(title="Accuracy", xlabel="epoch", ylim=(0, 1.02)); b.legend(); b.grid(alpha=.3)
    for ax in (a, b):
        ax.axvline(history["best_epoch"], color="gray", ls="--", lw=1)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=16, color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_feature_maps(model, sample: torch.Tensor, device):
    """Show the feature maps after each Conv2d layer for one normalised CHW tensor."""
    model.eval()
    x = sample.unsqueeze(0).to(device)
    figs = []
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            maps = x.squeeze(0).cpu().numpy()
            fig, axes = plt.subplots(2, math.ceil(len(maps) / 2), figsize=(2 * math.ceil(len(maps) / 2), 4))
            for ax, m in zip(axes.flat, maps):
                ax.imshow(m); ax.axis("off")
            fig.suptitle(f"{layer} feature maps")
            figs.append(fig)
    return figs
