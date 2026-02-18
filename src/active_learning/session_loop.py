from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.active_learning.simulate import build_model, query_from_pool, train_one_round
from src.train.train_baseline import (
    SurfaceDataset,
    build_class_mapping,
    evaluate,
    read_label_overrides,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stateful active-learning session loop.")
    parser.add_argument("--action", required=True, choices=["init", "run_round", "finalize_query"])
    parser.add_argument("--data-root", type=Path, default=Path("data/synth_surface_defects"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=Path("artifacts/labels/human_labels.csv"))
    parser.add_argument("--state-json", type=Path, default=Path("artifacts/active_learning/session_state.json"))
    parser.add_argument("--query-csv", type=Path, default=Path("artifacts/active_learning/current_query.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("artifacts/active_learning/al_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-size", type=int, default=80)
    parser.add_argument("--query-size", type=int, default=50)
    parser.add_argument("--epochs-per-round", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--strategy", type=str, default="entropy", choices=["entropy", "margin", "least_confidence"])
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _metadata_and_mapping(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int], dict[int, str]]:
    metadata_path = args.metadata if args.metadata is not None else args.data_root / "metadata.csv"
    rows = read_metadata(metadata_path)
    by_path = {r["filepath"]: r for r in rows}
    class_to_idx = build_class_mapping(rows)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return rows, by_path, class_to_idx, idx_to_class


def action_init(args: argparse.Namespace) -> None:
    _log("Initializing AL session")
    rows, _, _, _ = _metadata_and_mapping(args)
    train_rows = [r for r in rows if r["split"] == "train"]
    random.Random(args.seed).shuffle(train_rows)
    seed_size = min(args.seed_size, len(train_rows))

    state = {
        "round": 0,
        "seed": args.seed,
        "seed_size": seed_size,
        "strategy": args.strategy,
        "diversity": bool(args.diversity),
        "query_size": args.query_size,
        "epochs_per_round": args.epochs_per_round,
        "labeled_paths": [r["filepath"] for r in train_rows[:seed_size]],
        "unlabeled_paths": [r["filepath"] for r in train_rows[seed_size:]],
        "pending_query_paths": [],
    }
    _save_state(args.state_json, state)
    _log(f"Initialized AL session at {args.state_json}")
    _log(f"labeled={len(state['labeled_paths'])} unlabeled={len(state['unlabeled_paths'])}")


def action_run_round(args: argparse.Namespace) -> None:
    _log("Starting action: run_round")
    state = _load_state(args.state_json)
    if state.get("pending_query_paths"):
        raise RuntimeError("Pending queried samples exist. Label + finalize them before running next round.")

    rows, by_path, class_to_idx, idx_to_class = _metadata_and_mapping(args)
    label_overrides = read_label_overrides(args.labels_csv)

    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    labeled_rows = [by_path[p] for p in state["labeled_paths"] if p in by_path]
    unlabeled_rows = [by_path[p] for p in state["unlabeled_paths"] if p in by_path]
    _log(
        f"Round {int(state.get('round', 0)) + 1} setup | "
        f"labeled={len(labeled_rows)} unlabeled={len(unlabeled_rows)} val={len(val_rows)} test={len(test_rows)}"
    )

    if not labeled_rows:
        raise RuntimeError("No labeled samples available for training.")
    if not unlabeled_rows:
        raise RuntimeError("No unlabeled pool available for querying.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")
    model = build_model(
        class_to_idx=class_to_idx,
        device=device,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )

    train_ds = SurfaceDataset(args.data_root, labeled_rows, class_to_idx, label_overrides=label_overrides)
    val_ds = SurfaceDataset(args.data_root, val_rows, class_to_idx)
    test_ds = SurfaceDataset(args.data_root, test_rows, class_to_idx)
    pool_ds = SurfaceDataset(args.data_root, unlabeled_rows, class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pool_loader = DataLoader(pool_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    _log(f"Training started | epochs={args.epochs_per_round} batch_size={args.batch_size} lr={args.lr}")
    best_val_acc, _ = train_one_round(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs_per_round,
        lr=args.lr,
        weight_decay=args.weight_decay,
        logger=lambda m: _log(m),
    )
    _log("Training completed")
    test_stats = evaluate(model, test_loader, device)
    _log(f"Evaluation completed | best_val_acc={best_val_acc:.4f} test_acc={test_stats.acc:.4f} test_loss={test_stats.loss:.4f}")
    _log(
        f"Querying pool | strategy={args.strategy} diversity={bool(args.diversity)} "
        f"query_size={args.query_size}"
    )
    selected, unc = query_from_pool(
        model=model,
        pool_loader=pool_loader,
        strategy=args.strategy,
        diversity=bool(args.diversity),
        query_size=args.query_size,
        idx_to_class=idx_to_class,
        device=device,
    )

    selected_paths = [s["filepath"] for s in selected]
    _log(
        f"Query done | selected={len(selected_paths)} "
        f"pool_unc_mean={unc.get('pool_uncertainty_mean', 0.0):.4f} "
        f"selected_unc_mean={unc.get('selected_uncertainty_mean', 0.0):.4f}"
    )
    state["round"] = int(state["round"]) + 1
    state["pending_query_paths"] = selected_paths
    _save_state(args.state_json, state)

    args.query_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.query_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "filepath", "true_label", "uncertainty"])
        writer.writeheader()
        for s in selected:
            writer.writerow(
                {
                    "round": state["round"],
                    "filepath": s["filepath"],
                    "true_label": s["true_label"],
                    "uncertainty": s["uncertainty"],
                }
            )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.metrics_csv.exists()
    with args.metrics_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "round": state["round"],
                "labeled_size": len(labeled_rows),
                "pool_size_before_query": len(unlabeled_rows),
                "queried_this_round": len(selected_paths),
                "pool_size_after_query": max(0, len(unlabeled_rows) - len(selected_paths)),
                "best_val_acc": float(best_val_acc),
                "test_loss": float(test_stats.loss),
                "test_acc": float(test_stats.acc),
                "pool_uncertainty_mean": unc.get("pool_uncertainty_mean", 0.0),
                "pool_uncertainty_std": unc.get("pool_uncertainty_std", 0.0),
                "selected_uncertainty_mean": unc.get("selected_uncertainty_mean", 0.0),
                "selected_uncertainty_std": unc.get("selected_uncertainty_std", 0.0),
                "selected_uncertainty_min": unc.get("selected_uncertainty_min", 0.0),
                "selected_uncertainty_max": unc.get("selected_uncertainty_max", 0.0),
            }
        )

    _log(
        f"Round {state['round']} complete: labeled={len(labeled_rows)} "
        f"queried={len(selected_paths)} pending={len(selected_paths)} test_acc={test_stats.acc:.4f}"
    )


def action_finalize_query(args: argparse.Namespace) -> None:
    _log("Starting action: finalize_query")
    state = _load_state(args.state_json)
    pending = list(state.get("pending_query_paths", []))
    if not pending:
        _log("No pending queried samples to finalize.")
        return

    label_overrides = read_label_overrides(args.labels_csv)
    labeled_paths = set(state["labeled_paths"])
    unlabeled_paths = set(state["unlabeled_paths"])

    moved = 0
    unresolved: list[str] = []
    for fp in pending:
        if fp in label_overrides:
            labeled_paths.add(fp)
            if fp in unlabeled_paths:
                unlabeled_paths.remove(fp)
            moved += 1
        else:
            unresolved.append(fp)

    state["labeled_paths"] = sorted(labeled_paths)
    state["unlabeled_paths"] = sorted(unlabeled_paths)
    state["pending_query_paths"] = unresolved
    _save_state(args.state_json, state)

    _log(f"Finalize complete: moved={moved}, unresolved={len(unresolved)}")
    if unresolved:
        _log("Unresolved samples still need human label in labels csv.")


def main() -> None:
    args = parse_args()
    if args.action == "init":
        action_init(args)
    elif args.action == "run_round":
        action_run_round(args)
    elif args.action == "finalize_query":
        action_finalize_query(args)
    else:
        raise ValueError(args.action)


if __name__ == "__main__":
    main()
