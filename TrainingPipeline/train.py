"""
LSM — Training script (CNN / TCN / 3DCNN)
==========================================
Uso:
    python train.py --dataset ./output --model cnn --epochs 150 --batch 64
    python train.py --cache prepared/lab_v2_cap5.npz --model tcn --causal 0 --pool attn
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from augment import expand_with_augmentation
from dataset import apply_augmentation, load_raw_dataset
from features import LAYOUT_V1, LAYOUT_V2, N_FRAMES, feature_dim_for_layout
from models import TRAIN_MODEL_MAP, count_parameters, create_model
from split_utils import cap_and_split_indices, export_split_manifest, stratified_split_indices


class LSMDataset(Dataset):
    def __init__(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_epoch(model, loader, optimizer, criterion, device, layout, augment_factor, joint_dropout, legacy_augment):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if not legacy_augment and augment_factor > 0:
            X_batch, y_batch = expand_with_augmentation(
                X_batch, y_batch, factor=augment_factor,
                layout=layout, joint_dropout=joint_dropout,
            )
            # Shuffle within the expanded batch
            perm = torch.randperm(len(y_batch), device=device)
            X_batch, y_batch = X_batch[perm], y_batch[perm]

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def top_k_accuracy(model, loader, device, k=5):
    model.eval()
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        k_eff = min(k, logits.size(1))
        topk = logits.topk(k_eff, dim=1).indices
        correct += (topk == y_batch.unsqueeze(1)).any(dim=1).sum().item()
        total += len(y_batch)
    return correct / total


@torch.no_grad()
def evaluate_split(model, X, y, device, batch_size=64, k=5):
    if len(y) == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0}
    ds = LSMDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        logits = model(Xb)
        correct1 += (logits.argmax(1) == yb).sum().item()
        k_eff = min(k, logits.size(1))
        topk = logits.topk(k_eff, dim=1).indices
        correct5 += (topk == yb.unsqueeze(1)).any(dim=1).sum().item()
        total += len(yb)
    return {
        "n": total,
        "top1": correct1 / total,
        "top5": correct5 / total,
        "top1_correct": correct1,
        "top5_correct": correct5,
    }


def save_checkpoint(state, path):
    torch.save(state, path)


def load_cache(path: str):
    data = np.load(path, allow_pickle=True)
    settings = json.loads(str(data["settings"]))
    classes = [str(c) for c in data["classes"].tolist()]
    meta = [
        {
            "clase": str(data["class_names"][i]),
            "categoria": str(data["categories"][i]),
            "archivo": str(data["paths"][i]),
        }
        for i in range(len(data["y"]))
    ]
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "classes": classes,
        "meta": meta,
        "idx_train": data["idx_train"].astype(np.int64),
        "idx_val": data["idx_val"].astype(np.int64),
        "idx_extra": data["idx_extra"].astype(np.int64),
        "settings": settings,
    }


def main(args):
    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    layout = args.layout
    causal = bool(args.causal)
    pool = args.pool
    joint_dropout = args.joint_dropout
    legacy_augment = args.legacy_augment

    # ── Data ──────────────────────────────────────────────────────────────────
    idx_extra = np.array([], dtype=np.int64)
    if args.cache:
        print(f"\nCargando cache: {args.cache}")
        cache = load_cache(args.cache)
        X_raw, y_raw = cache["X"], cache["y"]
        classes, meta = cache["classes"], cache["meta"]
        idx_train, idx_val = cache["idx_train"], cache["idx_val"]
        idx_extra = cache["idx_extra"]
        layout = cache["settings"].get("layout", layout)
        print(f"  Layout from cache: {layout}")
        print(f"  Train/Val/Extra: {len(idx_train)}/{len(idx_val)}/{len(idx_extra)}")
    else:
        if not args.dataset:
            raise SystemExit("--dataset or --cache is required")
        print("\nCargando dataset...")
        X_raw, y_raw, classes, meta = load_raw_dataset(
            args.dataset,
            min_videos=args.min_videos,
            shoulder_norm=not args.no_shoulder_norm,
            layout=layout,
            max_videos_per_class=args.max_videos_per_class,
        )
        if args.max_videos_per_class is not None:
            idx_train, idx_val, idx_extra = cap_and_split_indices(
                y_raw, meta,
                max_videos_per_class=args.max_videos_per_class,
                val_fraction=args.val_split,
                seed=42,
            )
        else:
            idx_train, idx_val = stratified_split_indices(
                y_raw, meta, val_fraction=args.val_split, seed=42,
            )

    n_classes = len(classes)
    feature_dim = int(X_raw.shape[-1])
    print(f"  Feature dim : {feature_dim} (layout={layout})")
    print(f"  Clases      : {n_classes}")

    classes_path = os.path.join(args.output, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(args.output, "split_manifest.json")
    export_split_manifest(
        meta, y_raw, classes, idx_train, idx_val, manifest_path, idx_extra=idx_extra,
    )
    print(f"  Split manifest : {manifest_path}")

    X_train_raw, y_train_raw = X_raw[idx_train], y_raw[idx_train]
    X_val, y_val = X_raw[idx_val], y_raw[idx_val]
    X_extra = X_raw[idx_extra] if len(idx_extra) else np.zeros((0, *X_raw.shape[1:]), dtype=np.float32)
    y_extra = y_raw[idx_extra] if len(idx_extra) else np.zeros((0,), dtype=np.int64)

    print(f"\n  Split:")
    print(f"    Train originales : {len(X_train_raw)}")
    print(f"    Val originales   : {len(X_val)}")
    print(f"    Extra test       : {len(X_extra)}")

    if legacy_augment and args.augment > 0:
        print(f"    Legacy augment   : {args.augment}x (static, CPU)")
        X_train, y_train = apply_augmentation(
            X_train_raw, y_train_raw, augment_factor=args.augment, seed=42,
        )
        runtime_augment = 0
    else:
        X_train, y_train = X_train_raw, y_train_raw
        runtime_augment = args.augment
        print(f"    GPU augment/epoch: {runtime_augment}x (joint_dropout={joint_dropout})")

    train_ds = LSMDataset(X_train, y_train)
    val_ds = LSMDataset(X_val, y_val)

    if args.workers is not None:
        num_workers = args.workers
    elif device.type == "cuda":
        num_workers = 4
    else:
        num_workers = 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = {
        "input_dim": feature_dim,
        "layout": layout,
        "causal": causal,
        "pool": pool,
        "dropout": args.dropout,
    }
    registry_name = TRAIN_MODEL_MAP[args.model]
    model_kwargs = {
        "dropout": args.dropout,
        "input_dim": feature_dim,
        "pool": pool,
    }
    if args.model == "tcn":
        model_kwargs["causal"] = causal

    model = create_model(registry_name, num_classes=n_classes, **model_kwargs)
    if hasattr(model, "receptive_field"):
        print(f"  Campo recep.: {model.receptive_field()}")
    model = model.to(device)
    print(f"\n  Parámetros: {count_parameters(model):,}")
    print(f"  Config    : {model_config}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = max(0, args.warmup)
    if warmup_epochs > 0 and warmup_epochs < args.epochs:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_acc = 0.0
    patience_left = args.patience if args.patience > 0 else None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    resume_path = args.resume
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", history)
        print(f"  Resumiendo desde epoch {start_epoch}, mejor val_acc: {best_val_acc:.4f}")

    def ckpt_payload(epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "classes": classes,
            "n_frames": N_FRAMES,
            "feature_dim": feature_dim,
            "model_type": args.model,
            "model_config": model_config,
        }

    print(f"\n{'─'*65}")
    print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print(f"{'─'*65}")

    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            layout=layout, augment_factor=runtime_augment,
            joint_dropout=joint_dropout, legacy_augment=legacy_augment,
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        marker = " ✓" if improved else ""
        print(f"  {epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {lr_now:>8.6f}  {elapsed:>5.1f}s{marker}")

        save_checkpoint(ckpt_payload(epoch), os.path.join(args.output, "last.pt"))

        if improved:
            best_val_acc = val_acc
            save_checkpoint(ckpt_payload(epoch), os.path.join(args.output, "best_model.pt"))
            if patience_left is not None:
                patience_left = args.patience
        elif patience_left is not None:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={args.patience})")
                stopped_early = True
                break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                ckpt_payload(epoch),
                os.path.join(args.output, f"checkpoint_epoch{epoch+1}.pt"),
            )

    print(f"{'─'*65}")

    # ── Final metrics ─────────────────────────────────────────────────────────
    best_path = os.path.join(args.output, "best_model.pt")
    print(f"\n  Mejor modelo:")
    print(f"    Val Acc  (Top-1) : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    metrics = {
        "model": args.model,
        "layout": layout,
        "model_config": model_config,
        "n_classes": n_classes,
        "best_val_acc": best_val_acc,
        "stopped_early": stopped_early,
        "epochs_run": len(history["val_acc"]),
    }
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        top5 = top_k_accuracy(model, val_loader, device, k=5)
        print(f"    Val Acc  (Top-5) : {top5:.4f} ({top5*100:.2f}%)")
        metrics["val"] = {"top1": best_val_acc, "top5": top5, "n": len(y_val)}
        if len(y_extra):
            extra = evaluate_split(model, X_extra, y_extra, device, batch_size=args.batch)
            print(f"    Extra Top-1       : {extra['top1']:.4f} ({extra['top1']*100:.2f}%)  n={extra['n']}")
            print(f"    Extra Top-5       : {extra['top5']:.4f} ({extra['top5']*100:.2f}%)")
            metrics["extra_test"] = extra
        print(f"    Guardado en      : {best_path}")
    else:
        print("    (Sin mejora en val_acc — best_model.pt no guardado)")

    with open(os.path.join(args.output, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── Curves ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        title = f"LSM {args.model.upper()} — Curvas de entrenamiento"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        epochs_x = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs_x, history["train_loss"], label="Train", color="#4C9BE8")
        ax1.plot(epochs_x, history["val_loss"], label="Val", color="#E85C5C")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs_x, history["train_acc"], label="Train", color="#4C9BE8")
        ax2.plot(epochs_x, history["val_acc"], label="Val", color="#E85C5C")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy")
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        curves_path = os.path.join(args.output, "training_curves.png")
        plt.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Curvas guardadas : {curves_path}")
    except ImportError:
        pass


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults (skipping None/False) and keep the line breaks of the epilog."""

    def _get_help_string(self, action):
        text = action.help or ""
        if "%(default)" in text or action.default is None or action.default is False \
                or action.default is argparse.SUPPRESS:
            return text
        return text + " (default: %(default)s)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an LSM sign classifier (CNN / TCN / 3D-CNN) on landmark sequences. "
                    "Snake_case flags (--val_split, ...) also accept kebab-case aliases (--val-split, ...).",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --cache ../Workspace/prepared/lab_v2_cap5.npz --model cnn "
            "--output ../Runs/lab_cap5_cnn\n"
            "  python train.py --cache ../Workspace/prepared/lab_v2_cap5.npz --model tcn --causal 0 --pool attn "
            "--output ../Runs/lab_cap5_tcn\n"
            "  python train.py --dataset ../Workspace/landmarks --model cnn --layout v2_162 --max-videos-per-class 5\n\n"
            "Outputs (in --output): best_model.pt, last.pt, checkpoint_epoch*.pt, classes.json, split_manifest.json,\n"
            "  history.json, metrics.json, training_curves.png.\n"
            "Prefer the orchestrator, which also copies best_model.pt to Models/Pytorch/: ./lsm train --help"
        ),
    )
    data = parser.add_argument_group("data (--cache or --dataset is required)")
    data.add_argument("--cache", default=None, help="Prepared .npz from prepare_dataset.py (recommended)")
    data.add_argument("--dataset", default=None, help="Raw landmark JSON folder (parsed on the fly)")
    data.add_argument("--layout", default=LAYOUT_V1, choices=[LAYOUT_V1, LAYOUT_V2],
                      help="Feature layout for --dataset (a --cache carries its own)")
    data.add_argument("--val_split", "--val-split", dest="val_split", type=float, default=0.15,
                      help="--dataset only: validation fraction")
    data.add_argument("--min_videos", "--min-videos", dest="min_videos", type=int, default=2,
                      help="--dataset only: drop words with fewer than N videos")
    data.add_argument("--max_videos_per_class", "--max-videos-per-class", dest="max_videos_per_class",
                      type=int, default=None, help="--dataset only: cap videos per word (rest = extra test)")
    data.add_argument("--no-shoulder-norm", action="store_true",
                      help="--dataset only: disable shoulder normalisation (v1 layout)")

    model = parser.add_argument_group("model")
    model.add_argument("--model", default="cnn", choices=["cnn", "tcn", "3dcnn"], help="Architecture")
    model.add_argument("--pool", default="avg", choices=["avg", "attn"], help="Temporal pooling")
    model.add_argument("--causal", type=int, default=1, help="TCN only: 1 = causal padding (legacy), 0 = symmetric")
    model.add_argument("--dropout", type=float, default=0.3, help="Classifier dropout")
    model.add_argument("--joint_dropout", "--joint-dropout", dest="joint_dropout", type=float, default=0.1,
                       help="Probability of dropping whole joints during training")

    opt = parser.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    opt.add_argument("--batch", type=int, default=64, help="Batch size")
    opt.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (AdamW + cosine schedule)")
    opt.add_argument("--warmup", type=int, default=10, help="Linear warm-up epochs")
    opt.add_argument("--patience", type=int, default=15, help="Early-stopping patience in epochs; 0 disables")

    aug = parser.add_argument_group("augmentation")
    aug.add_argument("--augment", type=int, default=10,
                     help="Augmented views per train video per epoch (GPU) or static factor if --legacy-augment")
    aug.add_argument("--legacy-augment", action="store_true",
                     help="Use static CPU augmentation (old pipeline) instead of on-the-fly GPU augmentation")

    run = parser.add_argument_group("run control")
    run.add_argument("--output", default="./runs", help="Run directory for checkpoints and metrics")
    run.add_argument("--resume", default=None, help="Checkpoint (e.g. <output>/last.pt) to resume from")
    run.add_argument("--workers", type=int, default=None, help="DataLoader workers (auto if omitted)")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
