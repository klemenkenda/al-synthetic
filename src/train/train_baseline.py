from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ANSI_RESET = "\033[0m"
PHASE_STYLES = {
    "SETUP": ("\033[38;5;39m", "⚙"),
    "TRAIN": ("\033[38;5;45m", "🏋"),
    "EVAL": ("\033[38;5;220m", "📊"),
    "SAVE": ("\033[38;5;111m", "💾"),
    "DONE": ("\033[38;5;82m", "✓"),
}
PHASE_ASCII_ICONS = {
    "SETUP": "[S]",
    "TRAIN": "[T]",
    "EVAL": "[E]",
    "SAVE": "[W]",
    "DONE": "[OK]",
}


def _supports_color() -> bool:
    return os.environ.get("NO_COLOR") is None


def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "⚙".encode(enc)
        return True
    except Exception:
        return False


def log(message: str, phase: str = "SETUP") -> None:
    ts = time.strftime("%H:%M:%S")
    color, icon = PHASE_STYLES.get(phase, ("", "•"))
    if not _supports_unicode():
        icon = PHASE_ASCII_ICONS.get(phase, "*")
    plain = f"[{ts}] [{phase}] {message}"
    if _supports_color() and color:
        print(f"{color}{icon} {plain}{ANSI_RESET}", flush=True)
    else:
        print(f"{icon} {plain}", flush=True)


class SurfaceDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, Any]],
        class_to_idx: dict[str, int],
        label_overrides: dict[str, str] | None = None,
    ) -> None:
        self.data_root = data_root
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.label_overrides = label_overrides or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        filepath = row["filepath"]
        image_path = self.data_root / filepath
        image = Image.open(image_path).convert("L")

        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        x = x.unsqueeze(0)
        target_label = self.label_overrides.get(filepath, row["label"])
        y = self.class_to_idx[target_label]
        return x, y, filepath


class SurfaceDefectNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(96, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        emb = self.dropout(F.relu(self.embedding(h)))
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


@dataclass
class EpochStats:
    loss: float
    acc: float


def read_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_label_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for row in rows:
        fp = row.get("filepath", "").strip()
        label = row.get("label", "").strip()
        if fp and label:
            out[fp] = label
    return out


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    return train, val, test


def build_class_mapping(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({r["label"] for r in rows})
    return {label: idx for idx, label in enumerate(labels)}


def run_epoch(
    model: SurfaceDefectNet,
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

    return EpochStats(
        loss=total_loss / max(total, 1),
        acc=total_correct / max(total, 1),
    )


@torch.no_grad()
def evaluate(model: SurfaceDefectNet, loader: DataLoader[tuple[torch.Tensor, int, str]], device: torch.device) -> EpochStats:
    return run_epoch(model, loader, device, optimizer=None)


@torch.no_grad()
def export_pool_scores(
    model: SurfaceDefectNet,
    loader: DataLoader[tuple[torch.Tensor, int, str]],
    device: torch.device,
    out_csv: Path,
) -> None:
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath",
                "pred_label_idx",
                "max_prob",
                "least_confidence",
                "margin",
                "entropy",
                "embedding",
            ],
        )
        writer.writeheader()

        for x, _, paths in loader:
            x = x.to(device)
            logits, emb = model(x, return_embedding=True)
            probs = F.softmax(logits, dim=1)

            top2 = torch.topk(probs, k=2, dim=1).values
            max_prob, pred_idx = torch.max(probs, dim=1)
            least_conf = 1.0 - max_prob
            margin = top2[:, 0] - top2[:, 1]
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=1)

            for i, path in enumerate(paths):
                writer.writerow(
                    {
                        "filepath": path,
                        "pred_label_idx": int(pred_idx[i].item()),
                        "max_prob": float(max_prob[i].item()),
                        "least_confidence": float(least_conf[i].item()),
                        "margin": float(margin[i].item()),
                        "entropy": float(entropy[i].item()),
                        "embedding": json.dumps([round(float(v), 6) for v in emb[i].cpu().tolist()]),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surface defect classifier (active learning ready).")
    parser.add_argument("--data-root", type=Path, default=Path("data/synth_surface_defects"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/surface_classifier"))
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional human labels override file.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metadata is None:
        args.metadata = args.data_root / "metadata.csv"

    seed_everything(args.seed)

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
    model = SurfaceDefectNet(
        num_classes=len(class_to_idx),
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.artifacts_dir / "best.pt"
    metrics_csv_path = args.artifacts_dir / "training_metrics.csv"
    run_summary_path = args.artifacts_dir / "training_run_summary.json"
    best_val_acc = -1.0
    epoch_rows: list[dict[str, Any]] = []
    started_at = time.time()

    log(f"Training on {device} | classes={class_to_idx}", phase="SETUP")
    log(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}", phase="SETUP")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, optimizer)
        val_stats = evaluate(model, val_loader, device)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_acc": train_stats.acc,
                "val_loss": val_stats.loss,
                "val_acc": val_stats.acc,
            }
        )
        log(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
            f"val_loss={val_stats.loss:.4f} val_acc={val_stats.acc:.4f}",
            phase="TRAIN",
        )

        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            ckpt_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "config": ckpt_config,
                    "best_val_acc": best_val_acc,
                },
                best_ckpt_path,
            )

    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_stats = evaluate(model, test_loader, device)
    log(f"Best val_acc={best_val_acc:.4f} | test_loss={test_stats.loss:.4f} test_acc={test_stats.acc:.4f}", phase="EVAL")

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    run_summary = {
        "elapsed_seconds": round(time.time() - started_at, 3),
        "device": str(device),
        "num_samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "classes": class_to_idx,
        "best_val_acc": best_val_acc,
        "test_loss": test_stats.loss,
        "test_acc": test_stats.acc,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    export_pool_scores(
        model=model,
        loader=test_loader,
        device=device,
        out_csv=args.artifacts_dir / "test_uncertainty_scores.csv",
    )
    log(f"Saved model: {best_ckpt_path}", phase="SAVE")
    log(f"Saved training metrics: {metrics_csv_path}", phase="SAVE")
    log(f"Saved training summary: {run_summary_path}", phase="SAVE")
    log(f"Saved uncertainty+embedding export: {args.artifacts_dir / 'test_uncertainty_scores.csv'}", phase="SAVE")
    log(f"Training completed in {time.time() - started_at:.1f}s", phase="DONE")


if __name__ == "__main__":
    main()
