from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.train.train_baseline import (
    SurfaceDataset,
    SurfaceDefectNet,
    build_class_mapping,
    evaluate,
    read_metadata,
    read_label_overrides,
    run_epoch,
)


ANSI_RESET = "\033[0m"
PHASE_STYLES = {
    "SETUP": ("\033[38;5;39m", "⚙"),
    "ROUND": ("\033[38;5;141m", "🔁"),
    "TRAIN": ("\033[38;5;45m", "🏋"),
    "EVAL": ("\033[38;5;220m", "📊"),
    "QUERY": ("\033[38;5;208m", "🎯"),
    "SAVE": ("\033[38;5;111m", "💾"),
    "DONE": ("\033[38;5;82m", "✓"),
}
PHASE_ASCII_ICONS = {
    "SETUP": "[S]",
    "ROUND": "[R]",
    "TRAIN": "[T]",
    "EVAL": "[E]",
    "QUERY": "[Q]",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-loop active learning simulation for surface defect classifier.")
    parser.add_argument("--data-root", type=Path, default=Path("data/synth_surface_defects"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/surface_classifier/best.pt"))
    parser.add_argument("--warm-start", action="store_true", help="Initialize round-1 model from --checkpoint weights.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--query-size", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="entropy", choices=["entropy", "margin", "least_confidence"])
    parser.add_argument("--diversity", action="store_true", help="Enable greedy diversity in embedding space.")
    parser.add_argument("--seed-size", type=int, default=80, help="Initial labeled set size sampled from train split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional human labels override file for training labels.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/active_learning"))
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/active_learning/al_query.csv"))
    parser.add_argument("--log-file", type=Path, default=Path("artifacts/active_learning/simulation.log"))
    return parser.parse_args()


def uncertainty_scores(probs: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "entropy":
        return -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=1)
    if strategy == "least_confidence":
        return 1.0 - probs.max(dim=1).values
    if strategy == "margin":
        top2 = torch.topk(probs, k=2, dim=1).values
        return 1.0 - (top2[:, 0] - top2[:, 1])
    raise ValueError(f"Unsupported strategy: {strategy}")


def greedy_diverse_topk(emb: np.ndarray, utility: np.ndarray, k: int) -> list[int]:
    if len(emb) == 0 or k <= 0:
        return []

    chosen: list[int] = [int(np.argmax(utility))]
    remaining = set(range(len(emb)))
    remaining.remove(chosen[0])

    emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    utility_norm = (utility - utility.min()) / max(1e-8, utility.max() - utility.min())

    while len(chosen) < min(k, len(emb)) and remaining:
        rem_idx = np.array(sorted(remaining), dtype=np.int64)
        sims = emb_norm[rem_idx] @ emb_norm[np.array(chosen)].T
        max_sim = sims.max(axis=1)
        diversity = 1.0 - max_sim
        score = 0.7 * utility_norm[rem_idx] + 0.3 * diversity
        pick_local = int(np.argmax(score))
        pick = int(rem_idx[pick_local])
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def maybe_load_checkpoint(args: argparse.Namespace, device: torch.device) -> dict[str, Any] | None:
    if args.checkpoint is None:
        return None
    if not args.checkpoint.exists():
        return None
    return torch.load(args.checkpoint, map_location=device, weights_only=False)


def build_model(
    class_to_idx: dict[str, int],
    device: torch.device,
    embedding_dim: int,
    dropout: float,
) -> SurfaceDefectNet:
    return SurfaceDefectNet(
        num_classes=len(class_to_idx),
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)


def train_one_round(
    model: SurfaceDefectNet,
    train_loader: DataLoader[tuple[torch.Tensor, int, str]],
    val_loader: DataLoader[tuple[torch.Tensor, int, str]],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    logger: Callable[[str], None] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, device, optimizer)
        val_stats = evaluate(model, val_loader, device)
        if logger is not None:
            logger(
                f"  [epoch {epoch:02d}/{epochs:02d}] "
                f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
                f"val_loss={val_stats.loss:.4f} val_acc={val_stats.acc:.4f}"
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_stats.loss),
                "train_acc": float(train_stats.acc),
                "val_loss": float(val_stats.loss),
                "val_acc": float(val_stats.acc),
            }
        )
        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, history


@torch.no_grad()
def query_from_pool(
    model: SurfaceDefectNet,
    pool_loader: DataLoader[tuple[torch.Tensor, int, str]],
    strategy: str,
    diversity: bool,
    query_size: int,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    all_rows: list[dict[str, Any]] = []

    for x, y, paths in pool_loader:
        x = x.to(device)
        logits, emb = model(x, return_embedding=True)
        probs = F.softmax(logits, dim=1)
        util = uncertainty_scores(probs, strategy=strategy)

        for i in range(x.size(0)):
            y_idx = int(y[i].item())
            all_rows.append(
                {
                    "filepath": paths[i],
                    "true_label_idx": y_idx,
                    "true_label": idx_to_class[y_idx],
                    "uncertainty": float(util[i].item()),
                    "embedding": emb[i].detach().cpu().numpy(),
                }
            )

    if not all_rows:
        return [], {"pool_uncertainty_mean": 0.0, "pool_uncertainty_std": 0.0}

    utility = np.array([r["uncertainty"] for r in all_rows], dtype=np.float32)
    actual_k = min(query_size, len(all_rows))
    if diversity:
        embeddings = np.stack([r["embedding"] for r in all_rows], axis=0)
        chosen_idx = greedy_diverse_topk(embeddings, utility, actual_k)
    else:
        chosen_idx = np.argsort(-utility)[:actual_k].tolist()

    selected = [all_rows[i] for i in chosen_idx]
    selected_unc = np.array([r["uncertainty"] for r in selected], dtype=np.float32) if selected else np.array([0.0], dtype=np.float32)
    summary = {
        "pool_uncertainty_mean": float(utility.mean()),
        "pool_uncertainty_std": float(utility.std()),
        "selected_uncertainty_mean": float(selected_unc.mean()),
        "selected_uncertainty_std": float(selected_unc.std()),
        "selected_uncertainty_min": float(selected_unc.min()),
        "selected_uncertainty_max": float(selected_unc.max()),
    }
    return selected, summary


def main() -> None:
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metadata_path = args.metadata if args.metadata is not None else args.data_root / "metadata.csv"
    rows = read_metadata(metadata_path)
    label_overrides = read_label_overrides(args.labels_csv)
    class_to_idx = build_class_mapping(rows)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    random.shuffle(train_rows)

    seed_size = min(args.seed_size, len(train_rows))
    labeled = list(train_rows[:seed_size])
    unlabeled = list(train_rows[seed_size:])
    if not labeled:
        raise RuntimeError("No initial labeled samples. Increase dataset size or reduce constraints.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = maybe_load_checkpoint(args, device)
    embedding_dim = args.embedding_dim
    dropout = args.dropout
    if ckpt is not None and "config" in ckpt:
        embedding_dim = int(ckpt["config"].get("embedding_dim", embedding_dim))
        dropout = float(ckpt["config"].get("dropout", dropout))

    val_ds = SurfaceDataset(args.data_root, val_rows, class_to_idx)
    test_ds = SurfaceDataset(args.data_root, test_rows, class_to_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.metrics_dir / "al_metrics.csv"
    run_summary_json = args.metrics_dir / "al_run_summary.json"
    epoch_history_csv = args.metrics_dir / "al_epoch_history.csv"

    metric_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    query_rows_all: list[dict[str, Any]] = []

    log_f = args.log_file.open("w", encoding="utf-8")

    def log(msg: str, phase: str = "SETUP") -> None:
        ts = time.strftime("%H:%M:%S")
        color, icon = PHASE_STYLES.get(phase, ("", "•"))
        if not _supports_unicode():
            icon = PHASE_ASCII_ICONS.get(phase, "*")
        plain = f"[{ts}] [{phase}] {msg}"
        if _supports_color() and color:
            print(f"{color}{icon} {plain}{ANSI_RESET}", flush=True)
        else:
            print(f"{icon} {plain}", flush=True)
        log_f.write(plain + "\n")
        log_f.flush()

    log(
        f"Starting AL loop on {device} | rounds={args.rounds} | seed_size={seed_size} | "
        f"query_size={args.query_size} | strategy={args.strategy} | diversity={args.diversity}",
        phase="SETUP",
    )
    log(
        f"Dataset sizes: train_total={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"| initial_labeled={len(labeled)} initial_pool={len(unlabeled)}",
        phase="SETUP",
    )

    for round_idx in range(1, args.rounds + 1):
        if not labeled:
            break

        log(f"[round {round_idx:02d}] build train set and model", phase="ROUND")
        train_ds = SurfaceDataset(args.data_root, labeled, class_to_idx, label_overrides=label_overrides)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        model = build_model(class_to_idx, device, embedding_dim=embedding_dim, dropout=dropout)

        if round_idx == 1 and args.warm_start and ckpt is not None and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            log(f"[round {round_idx:02d}] warm-started model from checkpoint: {args.checkpoint}", phase="ROUND")

        log(f"[round {round_idx:02d}] train start | labeled={len(labeled)} epochs={args.epochs_per_round}", phase="TRAIN")
        best_val_acc, round_history = train_one_round(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs_per_round,
            lr=args.lr,
            weight_decay=args.weight_decay,
            logger=lambda m: log(m, phase="TRAIN"),
        )
        test_stats = evaluate(model, test_loader, device)
        log(
            f"[round {round_idx:02d}] eval done | best_val_acc={best_val_acc:.4f} "
            f"test_loss={test_stats.loss:.4f} test_acc={test_stats.acc:.4f}",
            phase="EVAL",
        )

        for r in round_history:
            epoch_rows.append(
                {
                    "round": round_idx,
                    "epoch": int(r["epoch"]),
                    "train_loss": r["train_loss"],
                    "train_acc": r["train_acc"],
                    "val_loss": r["val_loss"],
                    "val_acc": r["val_acc"],
                }
            )

        pool_size_before = len(unlabeled)
        selected: list[dict[str, Any]] = []
        unc_summary = {
            "pool_uncertainty_mean": 0.0,
            "pool_uncertainty_std": 0.0,
            "selected_uncertainty_mean": 0.0,
            "selected_uncertainty_std": 0.0,
            "selected_uncertainty_min": 0.0,
            "selected_uncertainty_max": 0.0,
        }

        if unlabeled:
            log(f"[round {round_idx:02d}] query start | pool_size={len(unlabeled)}", phase="QUERY")
            pool_ds = SurfaceDataset(args.data_root, unlabeled, class_to_idx)
            pool_loader = DataLoader(pool_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            selected, unc_summary = query_from_pool(
                model=model,
                pool_loader=pool_loader,
                strategy=args.strategy,
                diversity=args.diversity,
                query_size=args.query_size,
                idx_to_class=idx_to_class,
                device=device,
            )

            selected_paths = {r["filepath"] for r in selected}
            selected_lookup = {r["filepath"]: r for r in selected}

            for row in unlabeled:
                fp = row["filepath"]
                if fp in selected_paths:
                    labeled.append(row)
                    sr = selected_lookup[fp]
                    query_rows_all.append(
                        {
                            "round": round_idx,
                            "filepath": fp,
                            "true_label_idx": sr["true_label_idx"],
                            "true_label": sr["true_label"],
                            "uncertainty": sr["uncertainty"],
                            "strategy": args.strategy,
                            "diversity": args.diversity,
                            "embedding": json.dumps([round(float(v), 6) for v in sr["embedding"].tolist()]),
                        }
                    )

            unlabeled = [row for row in unlabeled if row["filepath"] not in selected_paths]
            log(
                f"[round {round_idx:02d}] query done | selected={len(selected)} "
                f"pool_mean_unc={unc_summary['pool_uncertainty_mean']:.4f} "
                f"selected_mean_unc={unc_summary['selected_uncertainty_mean']:.4f}",
                phase="QUERY",
            )

        metric_rows.append(
            {
                "round": round_idx,
                "labeled_size": len(labeled),
                "pool_size_before_query": pool_size_before,
                "queried_this_round": len(selected),
                "pool_size_after_query": len(unlabeled),
                "best_val_acc": best_val_acc,
                "test_loss": float(test_stats.loss),
                "test_acc": float(test_stats.acc),
                "pool_uncertainty_mean": unc_summary["pool_uncertainty_mean"],
                "pool_uncertainty_std": unc_summary["pool_uncertainty_std"],
                "selected_uncertainty_mean": unc_summary["selected_uncertainty_mean"],
                "selected_uncertainty_std": unc_summary["selected_uncertainty_std"],
                "selected_uncertainty_min": unc_summary["selected_uncertainty_min"],
                "selected_uncertainty_max": unc_summary["selected_uncertainty_max"],
            }
        )

        log(
            f"[round {round_idx:02d}] labeled={len(labeled)} pool={len(unlabeled)} "
            f"queried={len(selected)} best_val_acc={best_val_acc:.4f} test_acc={test_stats.acc:.4f}",
            phase="ROUND",
        )

        if not unlabeled:
            log("Unlabeled pool exhausted. Stopping early.", phase="DONE")
            break

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "labeled_size",
                "pool_size_before_query",
                "queried_this_round",
                "pool_size_after_query",
                "best_val_acc",
                "test_loss",
                "test_acc",
                "pool_uncertainty_mean",
                "pool_uncertainty_std",
                "selected_uncertainty_mean",
                "selected_uncertainty_std",
                "selected_uncertainty_min",
                "selected_uncertainty_max",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    with epoch_history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "filepath", "true_label_idx", "true_label", "uncertainty", "strategy", "diversity", "embedding"],
        )
        writer.writeheader()
        writer.writerows(query_rows_all)

    run_summary = {
        "elapsed_seconds": round(time.time() - started_at, 3),
        "num_rounds_completed": len(metric_rows),
        "initial_seed_size": seed_size,
        "final_labeled_size": len(labeled),
        "final_pool_size": len(unlabeled),
        "strategy": args.strategy,
        "diversity": args.diversity,
        "device": str(device),
        "paths": {
            "query_csv": str(args.out_csv),
            "metrics_csv": str(metrics_csv),
            "epoch_history_csv": str(epoch_history_csv),
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with run_summary_json.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    log(f"Saved AL query history: {args.out_csv}", phase="SAVE")
    log(f"Saved AL round metrics: {metrics_csv}", phase="SAVE")
    log(f"Saved AL epoch history: {epoch_history_csv}", phase="SAVE")
    log(f"Saved AL summary: {run_summary_json}", phase="SAVE")
    log(f"Saved simulation log: {args.log_file}", phase="SAVE")
    log(f"AL simulation completed in {time.time() - started_at:.1f}s", phase="DONE")
    log_f.close()


if __name__ == "__main__":
    main()
