"""
LSM — Script de entrenamiento 1D CNN
======================================
Uso:
    python train.py --dataset ./output --epochs 100 --batch 64
    python train.py --dataset ./output --epochs 100 --batch 64 --resume checkpoint.pt
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import load_raw_dataset, apply_augmentation, N_FRAMES, FEATURE_DIM
from model import LSM_CNN, count_parameters


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class LSMDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training utilities ────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  GPU: Apple MPS")
    else:
        device = torch.device("cpu")
        print("  CPU (sin GPU)")
    return device


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def top_k_accuracy(model, loader, device, k=5):
    model.eval()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits  = model(X_batch)
        topk    = logits.topk(k, dim=1).indices
        correct += (topk == y_batch.unsqueeze(1)).any(dim=1).sum().item()
        total   += len(y_batch)
    return correct / total


def save_checkpoint(state, path):
    torch.save(state, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    # ── Carga de datos ────────────────────────────────────────────────────────
    print("\nCargando dataset...")
    X_raw, y_raw, classes, meta = load_raw_dataset(args.dataset, min_videos=args.min_videos)
    n_classes = len(classes)

    # Guardar lista de clases (necesaria para Core ML)
    classes_path = os.path.join(args.output, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"  Clases guardadas: {classes_path}")

    # ── Split ANTES del augmentation ──────────────────────────────────────────
    # Garantiza que ningún video original aparece en train y val a la vez
    n_total  = len(X_raw)
    n_val    = max(1, int(n_total * args.val_split))
    n_train  = n_total - n_val

    rng_split = np.random.default_rng(42)
    idx_all   = rng_split.permutation(n_total)
    idx_train = idx_all[:n_train]
    idx_val   = idx_all[n_train:]

    X_train_raw, y_train_raw = X_raw[idx_train], y_raw[idx_train]
    X_val,       y_val       = X_raw[idx_val],   y_raw[idx_val]

    # Augmentation solo sobre train
    print(f"\n  Split (sobre videos originales):")
    print(f"    Train originales : {len(X_train_raw)}")
    print(f"    Val originales   : {len(X_val)}  ← sin augmentation")
    X_train, y_train = apply_augmentation(X_train_raw, y_train_raw,
                                          augment_factor=args.augment, seed=42)
    print(f"    Train aumentado  : {len(X_train)}  ({args.augment}x augment)")

    train_ds = LSMDataset(X_train, y_train)
    val_ds   = LSMDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = LSM_CNN(n_classes=n_classes, dropout=args.dropout).to(device)
    print(f"\n  Parámetros: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history      = ckpt.get("history", history)
        print(f"  Resumiendo desde epoch {start_epoch}, mejor val_acc: {best_val_acc:.4f}")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print(f"{'─'*65}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " ✓" if val_acc > best_val_acc else ""
        print(f"  {epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {lr_now:>8.6f}  {elapsed:>5.1f}s{marker}")

        # Checkpoint cada mejora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "history":      history,
                "classes":      classes,
                "n_frames":     N_FRAMES,
                "feature_dim":  FEATURE_DIM,
            }, os.path.join(args.output, "best_model.pt"))

        # Checkpoint periódico cada 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "history":      history,
                "classes":      classes,
                "n_frames":     N_FRAMES,
                "feature_dim":  FEATURE_DIM,
            }, os.path.join(args.output, f"checkpoint_epoch{epoch+1}.pt"))

    print(f"{'─'*65}")

    # ── Top-5 accuracy final ──────────────────────────────────────────────────
    best_ckpt = torch.load(os.path.join(args.output, "best_model.pt"),
                           map_location=device)
    model.load_state_dict(best_ckpt["model"])
    top5 = top_k_accuracy(model, val_loader, device, k=5)

    print(f"\n  Mejor modelo:")
    print(f"    Val Acc  (Top-1) : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"    Val Acc  (Top-5) : {top5:.4f} ({top5*100:.2f}%)")
    print(f"    Guardado en      : {args.output}/best_model.pt")

    # ── Curvas de entrenamiento ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LSM 1D CNN — Curvas de entrenamiento", fontsize=13, fontweight="bold")

        epochs_x = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs_x, history["train_loss"], label="Train", color="#4C9BE8")
        ax1.plot(epochs_x, history["val_loss"],   label="Val",   color="#E85C5C")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs_x, history["train_acc"], label="Train", color="#4C9BE8")
        ax2.plot(epochs_x, history["val_acc"],   label="Val",   color="#E85C5C")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(args.output, "training_curves.png")
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Curvas guardadas : {curves_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento LSM 1D CNN")
    parser.add_argument("--dataset",    required=True,        help="Carpeta raíz del dataset extraído")
    parser.add_argument("--output",     default="./runs",     help="Carpeta de salida para checkpoints")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--val_split",  type=float, default=0.15, help="Fracción de validación")
    parser.add_argument("--augment",    type=int,   default=10,   help="Versiones augmentadas por video")
    parser.add_argument("--min_videos",  type=int,   default=2,    help="Mínimo de videos por clase para incluirla")
    parser.add_argument("--resume",     default=None,             help="Checkpoint para continuar")
    args = parser.parse_args()
    main(args)
