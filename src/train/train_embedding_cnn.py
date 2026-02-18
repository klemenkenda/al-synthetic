from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.train.train_baseline import (
    SurfaceDataset,
    build_class_mapping,
    read_label_overrides,
    read_metadata,
    split_rows,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochStats:
    loss: float
    acc: float


class EmbeddingCNN(nn.Module):
    """
    Backbone -> GAP -> bottleneck embedding -> classifier.
    """

    def __init__(self, num_classes: int, emb_dim: int = 256, dropout: float = 0.2, l2_normalize: bool = True) -> None:
        super().__init__()
        self.l2_normalize = l2_normalize

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.emb_head = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.gap(h).flatten(1)
        emb = self.emb_head(h)
        if self.l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embeddings(x)
        return self.classifier(emb)


def run_epoch(
    model: EmbeddingCNN,
    loader: DataLoader[tuple[torch.Tensor, int, str]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochStats:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((preds == y).sum().item())
        total += int(x.size(0))

    return EpochStats(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))


@torch.no_grad()
def evaluate(model: EmbeddingCNN, loader: DataLoader[tuple[torch.Tensor, int, str]], device: torch.device) -> EpochStats:
    return run_epoch(model, loader, device, optimizer=None)


@torch.no_grad()
def extract_embeddings(
    model: EmbeddingCNN,
    loader: DataLoader[tuple[torch.Tensor, int, str]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_emb: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_paths: list[str] = []

    for x, y, paths in loader:
        x = x.to(device)
        emb = model.forward_embeddings(x).cpu().numpy()
        all_emb.append(emb)
        all_y.append(y.numpy())
        all_paths.extend(paths)

    if not all_emb:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.concatenate(all_emb, axis=0), np.concatenate(all_y, axis=0), all_paths


def save_embedding_split(
    out_dir: Path,
    split_name: str,
    emb: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    idx_to_class: dict[int, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split_name}_embeddings.npz"
    csv_path = out_dir / f"{split_name}_embeddings_index.csv"

    np.savez_compressed(npz_path, embeddings=emb, labels=y, paths=np.array(paths, dtype=object))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "filepath", "label_idx", "label"])
        writer.writeheader()
        for i, (fp, yi) in enumerate(zip(paths, y)):
            yi_int = int(yi)
            writer.writerow(
                {
                    "row_id": i,
                    "filepath": fp,
                    "label_idx": yi_int,
                    "label": idx_to_class[yi_int],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN and export embeddings for each split.")
    parser.add_argument("--data-root", type=Path, default=Path("data/synth_surface_defects"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional human labels override file.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/embedding_cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-l2-normalize", action="store_true", help="Disable L2 normalization on embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metadata is None:
        args.metadata = args.data_root / "metadata.csv"

    seed_everything(args.seed)
    start_t = time.time()

    rows = read_metadata(args.metadata)
    label_overrides = read_label_overrides(args.labels_csv)
    train_rows, val_rows, test_rows = split_rows(rows)
    class_to_idx = build_class_mapping(rows)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_ds = SurfaceDataset(args.data_root, train_rows, class_to_idx, label_overrides=label_overrides)
    val_ds = SurfaceDataset(args.data_root, val_rows, class_to_idx)
    test_ds = SurfaceDataset(args.data_root, test_rows, class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingCNN(
        num_classes=len(class_to_idx),
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        l2_normalize=not args.no_l2_normalize,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.artifacts_dir / "best.pt"
    metrics_csv = args.artifacts_dir / "training_metrics.csv"
    summary_json = args.artifacts_dir / "run_summary.json"
    embed_dir = args.artifacts_dir / "embeddings"

    print(f"[SETUP] device={device} classes={class_to_idx}", flush=True)
    print(f"[SETUP] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} emb_dim={args.emb_dim}", flush=True)

    best_val_acc = -1.0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer=optimizer)
        va = evaluate(model, val_loader, device)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "val_loss": va.loss,
                "val_acc": va.acc,
            }
        )
        print(
            f"[TRAIN] epoch={epoch:02d}/{args.epochs:02d} "
            f"train_loss={tr.loss:.4f} train_acc={tr.acc:.4f} val_loss={va.loss:.4f} val_acc={va.acc:.4f}",
            flush=True,
        )

        if va.acc > best_val_acc:
            best_val_acc = va.acc
            ckpt_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "best_val_acc": best_val_acc,
                    "config": ckpt_config,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    te = evaluate(model, test_loader, device)
    print(f"[EVAL] best_val_acc={best_val_acc:.4f} test_loss={te.loss:.4f} test_acc={te.acc:.4f}", flush=True)

    # Save epoch metrics
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(epoch_rows)

    # Extract embeddings from best checkpoint
    print("[EMBED] extracting embeddings for train/val/test", flush=True)
    tr_emb, tr_y, tr_paths = extract_embeddings(model, train_loader, device)
    va_emb, va_y, va_paths = extract_embeddings(model, val_loader, device)
    te_emb, te_y, te_paths = extract_embeddings(model, test_loader, device)

    save_embedding_split(embed_dir, "train", tr_emb, tr_y, tr_paths, idx_to_class)
    save_embedding_split(embed_dir, "val", va_emb, va_y, va_paths, idx_to_class)
    save_embedding_split(embed_dir, "test", te_emb, te_y, te_paths, idx_to_class)

    summary = {
        "elapsed_seconds": round(time.time() - start_t, 3),
        "device": str(device),
        "best_val_acc": best_val_acc,
        "test_loss": te.loss,
        "test_acc": te.acc,
        "embedding_dim": args.emb_dim,
        "l2_normalize": not args.no_l2_normalize,
        "paths": {
            "checkpoint": str(best_ckpt),
            "training_metrics": str(metrics_csv),
            "embeddings_dir": str(embed_dir),
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVE] checkpoint: {best_ckpt}", flush=True)
    print(f"[SAVE] training metrics: {metrics_csv}", flush=True)
    print(f"[SAVE] embeddings dir: {embed_dir}", flush=True)
    print(f"[SAVE] run summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
